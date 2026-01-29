"""Run OpenARM-VLA inference in the OpenArm fixed-position task (dataset-consistent)."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


CLI_ARGS_DIR = "openarm_isaac_lab/scripts/reinforcement_learning/rsl_rl"
if CLI_ARGS_DIR not in sys.path:
    sys.path.insert(0, CLI_ARGS_DIR)

import cli_args  # isort: skip

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--mamba_checkpoint", type=str, required=True, help="Path to MambaVLA checkpoint (.pth).")
parser.add_argument("--run_dir", type=str, default=None, help="Run dir for model_scaler.pkl (defaults to checkpoint directory).")
parser.add_argument("--lang_emb", type=str, default=None, help="Path to language embedding pkl for OpenArm tasks.")
parser.add_argument("--config", type=str, default="conf/config.yaml", help="Path to OpenARM-VLA config.yaml (defaults to ./config.yaml).")
parser.add_argument("--model_type", type=str, default=None, choices=["mamba", "transformer"], help="Model type to use (mamba or transformer).")
parser.add_argument("--dataset_root", type=str, required=True, help="Dataset root to rebuild scaler if model_scaler.pkl is missing.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts to run for the chosen slot.")
parser.add_argument("--output_metrics", type=str, default=None, help="Optional path to write JSON metrics.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-OpenArm-Play-v0", help="Name of the task.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--settle_steps", type=int, default=50)
parser.add_argument("--max_steps", type=int, default=100)
parser.add_argument("--target_slot", type=int, default=-1, help="Target slot index 0/1. If not set, random once at startup.")
parser.add_argument("--success_threshold", type=float, default=0.04, help="Success threshold for the task.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)


args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

if args_cli.target_slot not in (0, 1):
    args_cli.target_slot = random.choice([0, 1])
    print(f"[INFO] No --target_slot provided. Randomly selected target_slot={args_cli.target_slot}.")

_slot_to_object = {0: "object_red", 1: "object_green"}
os.environ["OPENARM_TARGET_OBJECT"] = _slot_to_object[args_cli.target_slot]

if args_cli.lang_emb is None:
    dataset_root = Path(args_cli.dataset_root).resolve()
    dataset_name = dataset_root.name
    repo_root = dataset_root.parents[1] if len(dataset_root.parents) > 1 else dataset_root.parent
    candidate_file = repo_root / "MambaVLA" / "language_embeddings" / f"{dataset_name}.pkl"
    if not candidate_file.exists():
        raise SystemExit(
            "Could not find language embedding file based on --dataset_root. "
            "Pass --lang_emb explicitly."
        )
    args_cli.lang_emb = str(candidate_file)

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import openarm.tasks  # noqa: F401

RIGHT_SLOT_POS = (0.40, 0.0, 0.055)
LEFT_SLOT_POS = (0.35, -0.20, 0.055)
DROP_HEIGHT = 0.02


def _set_object_pose(obj, xyz: tuple[float, float, float]) -> None:
    with torch.inference_mode():
        root_state = obj.data.root_state_w.clone()
        root_state[:, 0:3] = torch.tensor(xyz, device=root_state.device)
        root_state[:, 7:13] = 0.0
        obj.write_root_state_to_sim(root_state)
        zero_vel = torch.zeros((root_state.shape[0], 6), device=root_state.device)
        obj.write_root_velocity_to_sim(zero_vel)
        if hasattr(obj, "root_physx_view") and hasattr(obj.root_physx_view, "wake_up"):
            obj.root_physx_view.wake_up()
        if hasattr(obj, "root_physx_view") and hasattr(obj.root_physx_view, "set_rigid_body_disable_sleeping"):
            obj.root_physx_view.set_rigid_body_disable_sleeping(True)


def _move_cubes_to_fixed_slots(env, object_map: dict[str, str]) -> None:
    scene = env.unwrapped.scene
    left_slot = (LEFT_SLOT_POS[0], LEFT_SLOT_POS[1], LEFT_SLOT_POS[2] + DROP_HEIGHT)
    right_slot = (RIGHT_SLOT_POS[0], RIGHT_SLOT_POS[1], RIGHT_SLOT_POS[2] + DROP_HEIGHT)

    if "red" in object_map:
        _set_object_pose(scene[object_map["red"]], right_slot)
    if "green" in object_map:
        _set_object_pose(scene[object_map["green"]], left_slot)


def _zero_actions(env) -> torch.Tensor:
    return torch.zeros(
        (env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim),
        device=env.unwrapped.device,
    )


def _step_env(env, actions):
    out = env.step(actions)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, terminated | truncated, info
    obs, reward, done, info = out
    return obs, reward, done, info


def _update_episode_state(env, target_object_name: str, dones: torch.Tensor, target_pos: tuple[float, float, float] | None = None):
    obj = env.unwrapped.scene[target_object_name]
    obj_pos = obj.data.root_pos_w[:, :3]
    if target_pos is None:
        cmd = env.unwrapped.command_manager.get_command("object_pose")[:, :3]
        error = torch.norm(cmd - obj_pos, dim=1)
    else:
        target = torch.tensor(target_pos, device=obj_pos.device, dtype=obj_pos.dtype)
        error = torch.norm(target - obj_pos, dim=1)
    success_mask = error < args_cli.success_threshold
    done_mask = (dones > 0.5)
    return done_mask, success_mask, error


def _print_episode_success(env, target_color: str, target_object_name: str, done_mask: torch.Tensor, success_mask: torch.Tensor, episode_steps: torch.Tensor, error: torch.Tensor) -> None:
    if not torch.any(done_mask):
        return
    obj = env.unwrapped.scene[target_object_name]
    heights = obj.data.root_pos_w[:, 2]
    done_ids = torch.where(done_mask)[0].tolist()
    for i in done_ids:
        status = "SUCCESS" if success_mask[i].item() else "FAIL"
        steps = int(episode_steps[i].item())
        print(
            f"[EPISODE] env={i} color={target_color} {status} height={heights[i].item():.3f} "
            f"steps={steps} err={error[i].item():.3f}"
        )


def _load_mamba_lang_embeddings(path: str, device: str) -> dict[str, torch.Tensor]:
    import pickle

    with open(path, "rb") as f:
        task_embs = pickle.load(f)
    for k, v in task_embs.items():
        if isinstance(v, torch.Tensor):
            task_embs[k] = v.detach().to(device).float()
        else:
            task_embs[k] = torch.tensor(v, device=device, dtype=torch.float32)
    return task_embs


_SLOT_TO_LANG_KEY = {
    0: "pick_the_cube_and_lift_it_to_the_left_side_of_the_table",
    1: "pick_the_cube_and_reach_to_the_right_side_but_slighlty_lower",
}

_SLOT_TO_TARGET_POS = {
    0: (0.25, 0.3, 0.25),
    1: (0.25, -0.2, 0.2),
}


def _resolve_scene_layout(env) -> dict[str, str]:
    scene_keys = set(env.unwrapped.scene.keys())
    if {"object"}.issubset(scene_keys):
        # Single-object task layout.
        return {"red": "object", "green": "object"}
    if {"object", "distractor"}.issubset(scene_keys):
        # In TwoCube tasks, the target choice swaps which color is "object".
        target_object_name = os.environ.get("OPENARM_TARGET_OBJECT", "object_red")
        if target_object_name == "object_green":
            return {"green": "object", "red": "distractor"}
        return {"red": "object", "green": "distractor"}
    if {"object_red", "object_green"}.issubset(scene_keys):
        return {"red": "object_red", "green": "object_green"}
    raise KeyError(f"Unsupported scene layout. Available Entities: {sorted(scene_keys)}")


def _build_policy(
    checkpoint_path: str,
    run_dir: str | None,
    dataset_root: str,
    device: str,
    lang_emb_dim: int,
    config_path: str,
    model_type: str | None,
):
    from omegaconf import OmegaConf
    from MambaVLA.model_factory import create_mambavla_model
    from MambaVLA.utils.scaler import MinMaxScaler
    from src.dataloader.load import OpenArmDataset

    cfg = OmegaConf.load(config_path)

    resolved_model_type = model_type or cfg.get("model_type", "mamba")
    transformer_cfg = cfg.get("transformer", None)

    print(f"[INFO] Model type: {resolved_model_type}")
    model = create_mambavla_model(
        dataloader=None,
        camera_names=["agentview", "eye_in_hand"],
        latent_dim=cfg.latent_dim,
        action_dim=cfg.action_dim,
        lang_emb_dim=lang_emb_dim,
        embed_dim=cfg.embed_dim,
        obs_tok_len=cfg.obs_tok_len,
        action_seq_len=cfg.action_seq_len,
        perception_seq_len=1,
        state_dim=cfg.state_dim,
        device=device,
        n_layer=cfg.n_layer,
        d_intermediate=cfg.d_intermediate,
        sampling_steps=cfg.sampling_steps,
        transformer_weight_decay=cfg.transformer_weight_decay,
        obs_encoder_weight_decay=cfg.obs_encoder_weight_decay,
        learning_rate=cfg.learning_rate,
        betas=list(cfg.betas) if cfg.betas is not None else None,
        use_language_encoder=True,
        freeze_language_encoder=True,
        clip_model_name="ViT-B/32",
        model_type=resolved_model_type,
        transformer_cfg=transformer_cfg,
    )

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)

    scaler_dir = run_dir or os.path.dirname(checkpoint_path)
    scaler_path = os.path.join(scaler_dir, "model_scaler.pkl")
    if os.path.exists(scaler_path):
        model.load_model_scaler(scaler_dir)
    else:
        dataset = OpenArmDataset(
            data_directory=dataset_root,
            device=device,
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            state_dim=cfg.state_dim,
            max_len_data=cfg.max_len_data,
            chunck_size=cfg.chunck_size,
            demos_per_task=cfg.demos_per_task,
        )
        scaler = MinMaxScaler(dataset.get_all_actions(), True, device)
        model.set_scaler(scaler)

    model.eval()
    model.reset()
    return model


def _build_mamba_obs(env, lang_emb: torch.Tensor, device: str) -> dict[str, torch.Tensor]:
    cam_link0 = env.unwrapped.scene.sensors["camera_link0"].data.output["rgb"]
    cam_fixed = env.unwrapped.scene.sensors["camera_fixed"].data.output["rgb"]
    robot = env.unwrapped.scene["robot"]
    joint_pos = robot.data.joint_pos[:, :7]
    gripper = robot.data.joint_pos[:, -2:]
    robot_states = torch.cat([joint_pos, gripper], dim=1)

    agentview = cam_fixed.float().permute(0, 3, 1, 2) / 255.0
    eye_in_hand = cam_link0.float().permute(0, 3, 1, 2) / 255.0

    batch = agentview.shape[0]
    if lang_emb.dim() == 1:
        lang_batch = lang_emb.unsqueeze(0).repeat(batch, 1)
    else:
        lang_batch = lang_emb
        if lang_batch.shape[0] != batch:
            lang_batch = lang_batch.repeat(batch, 1)

    return {
        "agentview_image": agentview.unsqueeze(1).to(device),
        "eye_in_hand_image": eye_in_hand.unsqueeze(1).to(device),
        "lang_emb": lang_batch.unsqueeze(1).to(device),
        "robot_states": robot_states.unsqueeze(1).to(device),
    }


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.episode_length_s = 30.0
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    render_mode = "rgb_array" if (args_cli.video or args_cli.enable_cameras) else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    object_map = _resolve_scene_layout(env)
    if args_cli.target_slot not in (0, 1):
        args_cli.target_slot = random.choice([0, 1])
    target_color = "red" if args_cli.target_slot == 0 else "green"
    if target_color not in object_map:
        raise SystemExit(f"Scene does not contain expected color '{target_color}'. Available: {sorted(object_map.keys())}")
    target_object_name = object_map[target_color]
    target_pos = _SLOT_TO_TARGET_POS.get(args_cli.target_slot)


    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("logs", "mamba_vla", "videos"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    lang_embeddings = _load_mamba_lang_embeddings(args_cli.lang_emb, env.unwrapped.device)
    lang_key = _SLOT_TO_LANG_KEY.get(args_cli.target_slot)
    if lang_key is None:
        raise SystemExit(f"Unsupported target_slot={args_cli.target_slot}; expected 0 or 1.")
    if lang_key not in lang_embeddings:
        raise SystemExit(f"Missing lang embedding key: {lang_key}")
    lang_emb = lang_embeddings[lang_key]

    policy = _build_policy(
        checkpoint_path=args_cli.mamba_checkpoint,
        run_dir=args_cli.run_dir,
        dataset_root=args_cli.dataset_root,
        device=env.unwrapped.device,
        lang_emb_dim=lang_emb.shape[-1],
        config_path=args_cli.config,
        model_type=args_cli.model_type,
    )

    dt = env.unwrapped.step_dt
    env.reset()
    _move_cubes_to_fixed_slots(env, object_map)

    for _ in range(args_cli.settle_steps):
        _step_env(env, _zero_actions(env))

    timestep = 0
    rollouts_done = 0
    success_count = 0
    episode_lengths: list[int] = []
    infer_time_total = 0.0
    infer_calls = 0
    episode_steps = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device, dtype=torch.int32)

    try:
        while simulation_app.is_running() and rollouts_done < args_cli.num_rollouts:
            start_time = time.time()

            mamba_obs = _build_mamba_obs(env, lang_emb, env.unwrapped.device)
            with torch.inference_mode():
                infer_start = time.perf_counter()
                actions = policy.predict(mamba_obs)
                infer_time_total += time.perf_counter() - infer_start
                infer_calls += 1
            if actions.dim() == 1:
                actions = actions.unsqueeze(0)
            _, _, dones, _ = _step_env(env, actions)

            episode_steps += 1
            done_mask, success_mask, error = _update_episode_state(env, target_object_name, dones, target_pos)

            max_mask = episode_steps >= args_cli.max_steps
            done_mask = success_mask | max_mask
            if torch.any(done_mask):
                _print_episode_success(env, target_color, target_object_name, done_mask, success_mask, episode_steps, error)
                done_ids = torch.where(done_mask)[0].tolist()
                for i in done_ids:
                    rollouts_done += 1
                    if success_mask[i].item():
                        success_count += 1
                    episode_lengths.append(int(episode_steps[i].item()))
                    if rollouts_done >= args_cli.num_rollouts:
                        break
                episode_steps = torch.where(done_mask, torch.zeros_like(episode_steps), episode_steps)
                if rollouts_done < args_cli.num_rollouts:
                    env.reset()
                    _move_cubes_to_fixed_slots(env, object_map)
                    policy.reset()
                    for _ in range(args_cli.settle_steps):
                        _step_env(env, _zero_actions(env))

            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    break

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user. Shutting down...")
    finally:
        avg_infer_ms = (infer_time_total / infer_calls * 1000.0) if infer_calls else 0.0
        avg_episode_steps = (sum(episode_lengths) / len(episode_lengths)) if episode_lengths else 0.0
        success_rate = (success_count / rollouts_done) if rollouts_done else 0.0
        metrics = {
            "model_type": args_cli.model_type or "mamba",
            "target_slot": args_cli.target_slot,
            "num_rollouts": rollouts_done,
            "successes": success_count,
            "success_rate": success_rate,
            "avg_episode_steps": avg_episode_steps,
            "avg_inference_ms": avg_infer_ms,
        }
        print(f"[METRICS] {json.dumps(metrics, indent=2)}")
        if args_cli.output_metrics:
            out_path = Path(args_cli.output_metrics).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(metrics, indent=2))
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
