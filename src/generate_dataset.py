from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from omegaconf import DictConfig, OmegaConf
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate cube-lift HDF5 demos.")
parser.add_argument(
    "--dataset_config",
    type=str,
    required=True,
    help="Path to generate_dataset.yaml (dataset settings).",
)
parser.add_argument(
    "--tasks_config",
    type=str,
    required=True,
    help="Path to tasks.yaml (task names and target poses).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch


def npify(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _squeeze_env(x):
    arr = npify(x)
    if isinstance(arr, dict):
        return {k: _squeeze_env(v) for k, v in arr.items()}
    arr = np.asarray(arr)
    if arr.ndim >= 1 and arr.shape[0] == 1:
        return arr[0]
    return arr


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


def near_goal(env, tol: float) -> bool:
    obj = env.unwrapped.scene["object"]
    obj_pos = obj.data.root_pos_w[:, :3]
    cmd = env.unwrapped.command_manager.get_command("object_pose")[:, :3]
    err = torch.norm(obj_pos - cmd, dim=1)
    return bool(torch.all(err <= tol).item())


def record_step(env, episode, actions, rewards, dones):
    cam_link0 = env.unwrapped.scene.sensors["camera_link0"].data.output["rgb"]
    cam_fixed = env.unwrapped.scene.sensors["camera_fixed"].data.output["rgb"]
    episode["obs"]["eye_in_hand_rgb"].append(_squeeze_env(cam_link0))
    episode["obs"]["agentview_rgb"].append(_squeeze_env(cam_fixed))

    robot = env.unwrapped.scene["robot"]
    joint_pos = robot.data.joint_pos[:, :7]
    gripper = robot.data.joint_pos[:, -2:]
    episode["obs"]["joint_states"].append(_squeeze_env(joint_pos))
    episode["obs"]["gripper_states"].append(_squeeze_env(gripper))
    episode["robot_states"].append(_squeeze_env(torch.cat([joint_pos, gripper], dim=1)))

    episode["actions"].append(_squeeze_env(actions))
    episode["rewards"].append(_squeeze_env(rewards))
    episode["dones"].append(_squeeze_env(dones))


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
    if steps <= 0:
        return
    for _ in range(steps):
        obs = env.get_observations()
        zero_actions = torch.zeros_like(policy(obs))
        env.step(zero_actions)


def _collect_task(env, policy, cfg: DictConfig, task_cfg: Any, simulation_app):
    target_poses = _parse_target_poses(task_cfg.target_pose)
    out_path = Path(cfg.dataset_root) / f"{task_cfg.name}.hdf5"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    demo_global_idx = 0
    with h5py.File(out_path, "w") as out_f:
        for pose_idx, pose in enumerate(target_poses):
            save_pose_idx = cfg.save_pose_index if cfg.save_pose_index is not None else pose_idx
            print(f"[POSE] start pose{save_pose_idx} target={pose}")
            _set_fixed_object_pose_env(env, pose)

            counts = 0
            attempts = 0
            while counts < cfg.num_demos:
                attempts += 1
                if attempts > cfg.max_attempts_per_pose:
                    print(f"[SKIP] pose{pose_idx} reached max attempts ({cfg.max_attempts_per_pose})")
                    break
                obs = _reset_env(env)
                _settle_env(env, policy, cfg.settle_steps)
                obs = env.get_observations()
                if attempts == 1:
                    cmd = env.unwrapped.command_manager.get_command("object_pose")[:, :3]
                    print(f"[POSE] pose{save_pose_idx} command={_squeeze_env(cmd)}")

                episode = {
                    "obs": {"agentview_rgb": [], "eye_in_hand_rgb": [], "joint_states": [], "gripper_states": []},
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
                    record_step(env, episode, actions, rewards, dones)

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
                    if cfg.debug_interval > 0 and (ep_steps % cfg.debug_interval) == 0:
                        print(f"[POSE] pose{save_pose_idx} step={ep_steps} err={float(err.max().item()):.4f}")
                    if torch.any(err < cfg.success_threshold):
                        success = True
                        break

                    ep_steps += 1
                    if cfg.max_steps > 0 and ep_steps >= cfg.max_steps:
                        timed_out = True
                        obs = next_obs
                        break
                    obs = next_obs

                if success:
                    if not near_goal(env, cfg.success_threshold):
                        print(f"[SKIP] episode not near goal (tol={cfg.success_threshold:.4f})")
                        continue
                    counts += 1
                    data = {
                        "obs": {k: np.stack(v, axis=0) for k, v in episode["obs"].items()},
                        "actions": np.stack(episode["actions"], axis=0),
                        "rewards": np.stack(episode["rewards"], axis=0),
                        "dones": np.stack(episode["dones"], axis=0),
                        "robot_states": np.stack(episode["robot_states"], axis=0),
                    }
                    demo_key = f"demo_{demo_global_idx}"
                    append_demo_hdf5(
                        out_f,
                        demo_key,
                        data,
                        {"success": success, "target_pose": pose, "pose_index": save_pose_idx},
                    )
                    demo_global_idx += 1
                    print(f"[SAVED] {out_path}::{demo_key}")
                else:
                    if timed_out:
                        print(f"[SKIP] failed episode (max_steps={cfg.max_steps})")
                    else:
                        print("[SKIP] failed episode")

        out_f.attrs["task"] = task_cfg.name
        out_f.attrs["num_demos"] = demo_global_idx


def main():
    cfg: DictConfig = OmegaConf.load(args_cli.dataset_config)
    tasks_cfg: DictConfig = OmegaConf.load(args_cli.tasks_config)
    if not cfg.get("checkpoint"):
        raise ValueError("checkpoint is required in generate_dataset.yaml")
    if "tasks" not in tasks_cfg:
        raise ValueError("tasks not found in tasks.yaml.")

    cfg.checkpoint = os.path.abspath(cfg.checkpoint)
    cfg.dataset_root = os.path.abspath(cfg.dataset_root)

    global gym, OnPolicyRunner, retrieve_file_path, RslRlVecEnvWrapper, combine_frame_transforms
    global hydra_task_config
    import gymnasium as gym
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.hydra import hydra_task_config
    from isaaclab.utils.math import combine_frame_transforms
    import openarm.tasks  # noqa: F401

    task_keys = list(cfg.task_keys) if cfg.get("task_keys") else sorted(tasks_cfg.tasks.keys())

    @hydra_task_config(cfg.task, "rsl_rl_cfg_entry_point")
    def _entry(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        env_cfg.scene.num_envs = cfg.num_envs
        env_cfg.seed = agent_cfg.seed

        env = gym.make(
            cfg.task,
            cfg=env_cfg,
            render_mode="rgb_array" if getattr(args_cli, "headless", False) else None,
        )
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(retrieve_file_path(cfg.checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        for task_key in task_keys:
            task_cfg = tasks_cfg.tasks.get(task_key)
            if task_cfg is None:
                raise ValueError(f"task_key '{task_key}' not found in tasks.")
            if "target_pose" not in task_cfg:
                raise ValueError(f"task '{task_key}' must define target_pose.")
            if "name" not in task_cfg:
                raise ValueError(f"task '{task_key}' must define name.")
            _collect_task(env, policy, cfg, task_cfg, simulation_app)

        env.close()

    _entry()
    simulation_app.close()


if __name__ == "__main__":
    main()
