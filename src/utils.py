import os

from pathlib import Path

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



