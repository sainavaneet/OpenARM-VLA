import os
from datetime import datetime
from collections.abc import Sequence
from pathlib import Path

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

def _resolve_dataset_root(dataset_root: str | Path) -> str:
    path = Path(dataset_root).expanduser()
    if path.exists():
        return str(path.resolve())

    dataset_name = path.name
    candidates: list[Path] = []

    env_root = os.environ.get("OPENARM_DATASET_ROOT") or os.environ.get("DATASET_ROOT")
    if env_root:
        env_base = Path(env_root).expanduser()
        candidates.append(env_base)
        candidates.append(env_base / dataset_name)

    repo_root = Path(__file__).resolve().parents[1]
    candidates.append(repo_root / "datasets" / dataset_name)
    candidates.append(repo_root / dataset_name)

    cwd = Path.cwd()
    candidates.append(cwd / "datasets" / dataset_name)
    candidates.append(cwd / dataset_name)

    tried: list[str] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        tried.append(str(candidate))
        if candidate.exists():
            return str(candidate)

    detail = "\n  - " + "\n  - ".join(tried[:8])
    extra = "" if len(tried) <= 8 else f"\n  - (and {len(tried) - 8} more)"
    raise SystemExit(
        "Dataset root not found. Update eval_model.dataset_root or set OPENARM_DATASET_ROOT/DATASET_ROOT.\n"
        f"Tried:{detail}{extra}"
    )



def _resolve_latest_checkpoint(checkpoints_root: str | Path, model_type: str, dataset_root: str | Path) -> str:
    root = Path(checkpoints_root)
    if not root.exists():
        raise SystemExit(f"checkpoints_root not found: {root}")
    dataset_name = Path(dataset_root).resolve().name
    candidates = list(root.rglob(f"train/{model_type}/{dataset_name}/*.pth"))
    if not candidates:
        candidates = list(root.rglob(f"{model_type}/**/*.pth"))
    if not candidates:
        candidates = list(root.rglob("*.pth"))
    if not candidates:
        raise SystemExit(f"No .pth checkpoints found under {root}")
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def _default_output_metrics(dataset_root: str, checkpoint_path: str, model_type: str) -> str:
    dataset_name = Path(dataset_root).resolve().name
    ckpt_path = Path(checkpoint_path)
    date_dir = ckpt_path.parent.parent.name if ckpt_path.parent.parent else ""
    time_dir = ckpt_path.parent.name
    if len(date_dir) != 10 or len(time_dir) != 8:
        date_dir = datetime.now().strftime("%Y-%m-%d")
        time_dir = datetime.now().strftime("%H-%M-%S")
    return str(
        Path("outputs")
        / "eval"
        / model_type
        / dataset_name
        / date_dir
        / time_dir
        / "metrics.json"
    )


def _prepare_eval_cfg(eval_cfg) -> None:
    if not eval_cfg.get("dataset_root") or not eval_cfg.get("task"):
        raise SystemExit("eval_model must define dataset_root and task.")

    eval_cfg.dataset_root = _resolve_dataset_root(eval_cfg.dataset_root)
    print(f"[INFO] Using dataset_root: {eval_cfg.dataset_root}")

    if not eval_cfg.get("model_checkpoint"):
        repo_root = Path(__file__).resolve().parents[1]
        ckpt_root = eval_cfg.get("checkpoints_root") or (repo_root / "checkpoints")
        model_type = eval_cfg.get("model_type", "mamba")
        eval_cfg.model_checkpoint = _resolve_latest_checkpoint(
            ckpt_root,
            model_type,
            eval_cfg.dataset_root,
        )
        print(f"[INFO] Using latest checkpoint: {eval_cfg.model_checkpoint}")

    if not eval_cfg.get("output_metrics"):
        model_type = eval_cfg.get("model_type", "mamba")
        eval_cfg.output_metrics = _default_output_metrics(
            eval_cfg.dataset_root,
            eval_cfg.model_checkpoint,
            model_type,
        )


def _build_eval_metrics(
    *,
    model_type: str | None,
    rollouts_done: int,
    success_count: int,
    episode_lengths: Sequence[int],
    infer_time_total: float,
    infer_calls: int,
    task_name: str | None = None,
    tasks: int | None = None,
) -> dict[str, float | int | str]:
    success_rate = (success_count / rollouts_done) if rollouts_done else 0.0
    avg_episode_steps = (sum(episode_lengths) / len(episode_lengths)) if episode_lengths else 0.0
    avg_infer_ms = (infer_time_total / infer_calls * 1000.0) if infer_calls else 0.0
    metrics: dict[str, float | int | str] = {
        "model_type": model_type or "mamba",
        "num_rollouts": rollouts_done,
        "successes": success_count,
        "success_rate": success_rate,
        "avg_episode_steps": avg_episode_steps,
        "avg_inference_ms": avg_infer_ms,
    }
    if task_name is not None:
        metrics["task"] = task_name
    if tasks is not None:
        metrics["tasks"] = tasks
    return metrics


def _sanitize_name(name):
    safe = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "task"


def _render_camera_frame(env, sensor_name):
    cam = env.unwrapped.scene.sensors[sensor_name].data.output["rgb"]
    frame = cam[0] if hasattr(cam, "__getitem__") else cam
    try:
        import torch

        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
    except Exception:
        pass
    return frame


def _setup_eval_video_recording(env, eval_cfg, record_video_cls, print_dict=None, sensor_name="main_camera"):
    metrics_path = eval_cfg.get("output_metrics")
    if metrics_path:
        metrics_path = os.path.abspath(os.path.expanduser(metrics_path))
        video_root = os.path.join(os.path.dirname(metrics_path), "videos")
    else:
        video_root = os.path.abspath(os.path.join("outputs", "eval", "videos"))
    os.makedirs(video_root, exist_ok=True)
    print(f"[INFO] Video root: {video_root}")
    if sensor_name and hasattr(env.unwrapped.scene, "sensors") and sensor_name in env.unwrapped.scene.sensors:
        env.render = lambda: _render_camera_frame(env, sensor_name)
        print(f"[INFO] Recording video from sensor: {sensor_name}")
    else:
        if sensor_name:
            print(f"[WARN] {sensor_name} sensor not found; falling back to default render.")
    video_length = eval_cfg.video_length
    if video_length is None and eval_cfg.get("max_steps") is not None:
        video_length = eval_cfg.max_steps
    if video_length is None:
        video_length = 200
    video_kwargs = {
        "video_folder": video_root,
        "episode_trigger": lambda episode_id: True,
        "video_length": video_length,
        "disable_logger": True,
    }
    print("[INFO] Recording videos during play.")
    if print_dict is not None:
        print_dict(video_kwargs, nesting=4)
    env = record_video_cls(env, **video_kwargs)
    return env, video_root


def _set_eval_video_task_folder(env, video_root, task_index, task_name):
    task_slug = _sanitize_name(task_name)
    task_video_dir = os.path.join(video_root, f"task{task_index + 1}_{task_slug}")
    os.makedirs(task_video_dir, exist_ok=True)
    if hasattr(env, "video_folder"):
        env.video_folder = task_video_dir
    if hasattr(env, "name_prefix"):
        env.name_prefix = f"task{task_index + 1}_{task_slug}"
    if hasattr(env, "episode_id"):
        env.episode_id = 0
    print(f"[INFO] Video folder for task {task_index}: {task_video_dir}")
    return task_video_dir


def _init_wandb(config_path, eval_cfg):
    if not WANDB_AVAILABLE:
        return None

    try:
        import yaml
    except ImportError:
        return None

    # Load the config file as a plain dictionary
    try:
        with open(config_path, "r") as f:
            base_cfg = yaml.safe_load(f)
    except Exception:
        base_cfg = {}

    wandb_cfg = base_cfg.get("wandb", {})
    if not wandb_cfg or not wandb_cfg.get("enabled", False):
        return None

    init_kwargs = {
        "project": wandb_cfg.get("project"),
        "entity": wandb_cfg.get("entity"),
        "name": wandb_cfg.get("name"),
    }
    run_id = os.environ.get("WANDB_RUN_ID")
    if run_id:
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = os.environ.get("WANDB_RESUME", "allow")
    run = wandb.init(**init_kwargs)

    if run is not None:
        # Use plain Python objects for config update
        try:
            run.config["eval_cfg"] = dict(eval_cfg) if hasattr(eval_cfg, "items") else eval_cfg
            run.config["model_cfg"] = base_cfg
            run.config["model_checkpoint"] = eval_cfg.get("model_checkpoint", None)
            run.config["model_type"] = eval_cfg.get("model_type", None)
            run.config["dataset_root"] = eval_cfg.get("dataset_root", None)
        except Exception:
            pass
    return run


def _log_wandb_metrics(run, prefix, metrics):
    if run is None or not WANDB_AVAILABLE:
        return
    log_dict = {}
    for key, value in metrics.items():
        if key == "task":
            continue
        log_dict[f"{prefix}/{key}"] = value
    if log_dict:
        wandb.log(log_dict)
