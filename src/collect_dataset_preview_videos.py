from __future__ import annotations

"""
Collect a small number of successful demos and save an MP4 video from `main_camera`.

This is intended for qualitative visualization of how the dataset is generated.
It mirrors `src/generate_dataset.py` logic (teacher policy rollout + success filter),
but additionally records `main_camera` frames and writes them to disk.
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from omegaconf import DictConfig, OmegaConf
from isaaclab.app import AppLauncher


def _stack(value):
    return np.stack(value, axis=0) if isinstance(value, (list, tuple)) else np.asarray(value)


def append_demo_hdf5(h5f: h5py.File, demo_key: str, data: dict, meta: dict):
    data_grp = h5f.require_group("data")
    g = data_grp.create_group(demo_key)
    for k, v in data.items():
        if isinstance(v, dict):
            sg = g.create_group(k)
            for kk, vv in v.items():
                arr = _stack(vv)
                sg.create_dataset(kk, data=arr, dtype=arr.dtype)
        else:
            arr = _stack(v)
            g.create_dataset(k, data=arr, dtype=arr.dtype)
    for k, v in meta.items():
        g.attrs[k] = v


def _npify(x):
    # torch import must happen after Isaac Sim launches.
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _squeeze_env(x):
    arr = _npify(x)
    if isinstance(arr, dict):
        return {k: _squeeze_env(v) for k, v in arr.items()}
    arr = np.asarray(arr)
    if arr.ndim >= 1 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _parse_target_poses(raw: str) -> list[tuple[float, float, float]]:
    poses = []
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = [p.strip() for p in entry.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid target pose '{entry}'. Expected x,y,z.")
        poses.append(tuple(float(p) for p in parts))
    if len(poses) < 1:
        raise ValueError("No target poses provided.")
    return poses


def _set_fixed_object_pose_env(env, pose: tuple[float, float, float]) -> None:
    cmd_cfg = getattr(getattr(env.unwrapped, "command_manager", None), "cfg", None)
    if cmd_cfg is None or not hasattr(cmd_cfg, "object_pose"):
        raise RuntimeError("Could not access command_manager.cfg.object_pose to set target pose.")
    cmd_cfg.object_pose.ranges.pos_x = (pose[0], pose[0])
    cmd_cfg.object_pose.ranges.pos_y = (pose[1], pose[1])
    cmd_cfg.object_pose.ranges.pos_z = (pose[2], pose[2])


def _reset_env(env):
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def _settle_env(env, policy, steps: int):
    import torch

    if steps <= 0:
        return
    for _ in range(steps):
        obs = env.get_observations()
        zero_actions = torch.zeros_like(policy(obs))
        env.step(zero_actions)


def _near_goal(env, tol: float) -> bool:
    import torch

    obj = env.unwrapped.scene["object"]
    obj_pos = obj.data.root_pos_w[:, :3]
    cmd = env.unwrapped.command_manager.get_command("object_pose")[:, :3]
    err = torch.norm(obj_pos - cmd, dim=1)
    return bool(torch.all(err <= tol).item())


def _record_step(env, episode, actions, rewards, dones):
    # dataset cameras
    cam_link0 = env.unwrapped.scene.sensors["camera_link0"].data.output["rgb"]
    cam_fixed = env.unwrapped.scene.sensors["camera_fixed"].data.output["rgb"]
    episode["obs"]["eye_in_hand_rgb"].append(_squeeze_env(cam_link0))
    episode["obs"]["agentview_rgb"].append(_squeeze_env(cam_fixed))

    # main camera for video
    cam_main = env.unwrapped.scene.sensors["main_camera"].data.output["rgb"]
    episode["main_camera_rgb"].append(_squeeze_env(cam_main))

    robot = env.unwrapped.scene["robot"]
    joint_pos = robot.data.joint_pos[:, :7]
    gripper = robot.data.joint_pos[:, -2:]
    episode["obs"]["joint_states"].append(_squeeze_env(joint_pos))
    episode["obs"]["gripper_states"].append(_squeeze_env(gripper))
    episode["robot_states"].append(_squeeze_env(np.concatenate([_npify(joint_pos), _npify(gripper)], axis=1)))

    episode["actions"].append(_squeeze_env(actions))
    episode["rewards"].append(_squeeze_env(rewards))
    episode["dones"].append(_squeeze_env(dones))


def _write_mp4(frames: np.ndarray, out_path: Path, fps: int):
    import cv2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames (T,H,W,3), got {frames.shape}")

    h, w = int(frames.shape[1]), int(frames.shape[2])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {out_path}")
    try:
        for rgb in frames:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()


def _collect_task(
    *,
    env,
    policy,
    cfg: DictConfig,
    task_cfg: Any,
    simulation_app,
    out_root: Path,
    demos_per_task: int,
    max_attempts: int,
    fps: int,
):
    import torch
    from isaaclab.utils.math import combine_frame_transforms

    target_poses = _parse_target_poses(task_cfg.target_pose)
    h5_dir = out_root / "hdf5"
    video_dir = out_root / "videos" / str(task_cfg.name)
    h5_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    out_path = h5_dir / f"{task_cfg.name}.hdf5"
    demo_global_idx = 0

    all_frames: list[np.ndarray] = []
    with h5py.File(out_path, "w") as out_f:
        for pose_idx, pose in enumerate(target_poses):
            save_pose_idx = cfg.save_pose_index if cfg.save_pose_index is not None else pose_idx
            print(f"[TASK] {task_cfg.name} pose{save_pose_idx} target={pose}")
            _set_fixed_object_pose_env(env, pose)

            successes = 0
            attempts = 0
            while successes < demos_per_task and attempts < max_attempts:
                attempts += 1
                obs = _reset_env(env)
                _settle_env(env, policy, int(cfg.get("settle_steps", 0) or 0))

                episode = {
                    "obs": {
                        "eye_in_hand_rgb": [],
                        "agentview_rgb": [],
                        "joint_states": [],
                        "gripper_states": [],
                    },
                    "main_camera_rgb": [],
                    "actions": [],
                    "rewards": [],
                    "dones": [],
                    "robot_states": [],
                }

                success = False
                timed_out = False
                ep_steps = 0

                while simulation_app.is_running():
                    with torch.inference_mode():
                        actions = policy(obs)
                    next_obs, rewards, dones, _ = env.step(actions)
                    _record_step(env, episode, actions, rewards, dones)

                    obj = env.unwrapped.scene["object"]
                    obj_pos = obj.data.root_pos_w[:, :3]
                    robot = env.unwrapped.scene["robot"]
                    cmd = env.unwrapped.command_manager.get_command("object_pose")[:, :3]
                    des_pos_w, _ = combine_frame_transforms(
                        robot.data.root_pos_w,
                        robot.data.root_quat_w,
                        cmd,
                    )
                    err = torch.norm(des_pos_w - obj_pos, dim=1)
                    if torch.any(err < float(cfg.success_threshold)):
                        success = True
                        break

                    ep_steps += 1
                    if int(cfg.max_steps) > 0 and ep_steps >= int(cfg.max_steps):
                        timed_out = True
                        obs = next_obs
                        break
                    obs = next_obs

                status = "success" if success and _near_goal(env, float(cfg.success_threshold)) else "fail"
                if status == "success":
                    successes += 1
                    demo_key = f"demo_{demo_global_idx}"
                    demo_global_idx += 1

                    # save hdf5 demo
                    data = {
                        "obs": {k: np.stack(v, axis=0) for k, v in episode["obs"].items()},
                        "actions": np.stack(episode["actions"], axis=0),
                        "rewards": np.stack(episode["rewards"], axis=0),
                        "dones": np.stack(episode["dones"], axis=0),
                        "robot_states": np.stack(episode["robot_states"], axis=0),
                    }
                    append_demo_hdf5(
                        out_f,
                        demo_key,
                        data,
                        {"success": True, "target_pose": pose, "pose_index": int(save_pose_idx)},
                    )

                    # collect frames for a single, concatenated preview video
                    frames = np.stack(episode["main_camera_rgb"], axis=0)
                    all_frames.append(frames)
                    print(f"[SAVED] {out_path}::{demo_key}  frames={frames.shape[0]}")
                else:
                    reason = "timed_out" if timed_out else "failed"
                    print(f"[SKIP] {task_cfg.name} attempt={attempts} ({reason})")

            if successes < demos_per_task:
                print(
                    f"[WARN] {task_cfg.name} pose{save_pose_idx}: got {successes}/{demos_per_task} successes "
                    f"after {attempts} attempts"
                )

        out_f.attrs["task"] = str(task_cfg.name)
        out_f.attrs["num_demos"] = int(demo_global_idx)

    if all_frames:
        concat = np.concatenate(all_frames, axis=0)
        mp4_path = video_dir / "all_demos.mp4"
        _write_mp4(concat, mp4_path, fps=fps)
        print(f"[VIDEO] wrote concatenated demo video: {mp4_path}")
    print(f"[DONE] wrote: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect a few demos + main_camera videos (preview).")
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--tasks_config", type=str, required=True)
    parser.add_argument("--demos_per_task", type=int, default=5)
    parser.add_argument("--max_attempts_per_task", type=int, default=200)
    parser.add_argument("--output_root", type=str, default="outputs/dataset_preview")
    parser.add_argument("--fps", type=int, default=30)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Isaac Sim consumes CLI args; wipe them so downstream imports don't re-parse.
    sys.argv = [sys.argv[0]]

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    import gymnasium as gym
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.hydra import hydra_task_config
    import openarm.tasks  # noqa: F401

    cfg: DictConfig = OmegaConf.load(args.dataset_config)
    tasks_cfg: DictConfig = OmegaConf.load(args.tasks_config)
    if not cfg.get("checkpoint"):
        raise ValueError("checkpoint is required in generate_dataset.yaml")
    if "tasks" not in tasks_cfg:
        raise ValueError("tasks not found in tasks.yaml.")

    cfg.checkpoint = os.path.abspath(os.path.expanduser(str(cfg.checkpoint)))
    cfg.dataset_root = os.path.abspath(os.path.expanduser(str(cfg.dataset_root)))

    task_keys = list(cfg.task_keys) if cfg.get("task_keys") else sorted(tasks_cfg.tasks.keys())

    out_root = Path(args.output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    # Hydra (via `hydra_task_config`) defaults to writing to `outputs/<date>/<time>` under CWD.
    # Some repos end up with root-owned `outputs/<date>` folders, which breaks local runs.
    # Switching CWD to a user-writable output_root keeps all Hydra artifacts contained.
    os.chdir(out_root)

    @hydra_task_config(cfg.task, "rsl_rl_cfg_entry_point")
    def _entry(env_cfg, agent_cfg):
        random.seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))
        torch.manual_seed(int(cfg.seed))

        env_cfg.scene.num_envs = int(cfg.num_envs)
        env_cfg.seed = agent_cfg.seed

        # create env (cameras must be enabled via AppLauncher args: --enable_cameras)
        base_env = gym.make(
            cfg.task,
            cfg=env_cfg,
            render_mode="rgb_array" if getattr(args, "headless", False) else None,
        )

        # wrap env for rsl-rl + policy loading
        env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(retrieve_file_path(cfg.checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        for task_index, task_key in enumerate(task_keys):
            task_cfg = tasks_cfg.tasks.get(task_key)
            if task_cfg is None:
                raise ValueError(f"task_key '{task_key}' not found in tasks.")
            if "target_pose" not in task_cfg or "name" not in task_cfg:
                raise ValueError(f"task '{task_key}' must define name and target_pose.")

            task_root = out_root / f"task{task_index}_{task_cfg.name}"
            _collect_task(
                env=env,
                policy=policy,
                cfg=cfg,
                task_cfg=task_cfg,
                simulation_app=simulation_app,
                out_root=task_root,
                demos_per_task=int(args.demos_per_task),
                max_attempts=int(args.max_attempts_per_task),
                fps=int(args.fps),
            )

        env.close()

    _entry()
    simulation_app.close()


if __name__ == "__main__":
    main()
