from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import h5py
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate cube-lift HDF5 demos.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-OpenArm-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_demos", type=int, default=50)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--dataset_root", type=str, default="./datasets/cube_lift")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--success_threshold", type=float, default=0.035)
parser.add_argument(
    "--max_steps",
    type=int,
    default=100,
    help="Max steps per episode. Episodes that don't succeed within this are skipped.",
)
parser.add_argument(
    "--settle_steps",
    type=int,
    default=10,
    help="Zero-action steps after reset to let the cube settle.",
)
parser.add_argument(
    "--debug_interval",
    type=int,
    default=0,
    help="If >0, print pose error every N steps.",
)
parser.add_argument(
    "--max_attempts_per_pose",
    type=int,
    default=500,
    help="Max episodes to try per pose before skipping to the next pose.",
)
parser.add_argument(
    "--target_poses",
    type=str,
    default="0.25,0.0,0.25",
    help="Semicolon-separated list of target poses as x,y,z",
)
parser.add_argument(
    "--save_pose_index",
    type=int,
    default=None,
    help="If set, use this index in output filename demo_pose<idx>_*.hdf5.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Pass through Hydra overrides only.
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.utils.math import combine_frame_transforms

import openarm.tasks  # noqa: F401


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


def save_hdf5(path: Path, data: dict, meta: dict):
    def _stack(value):
        return np.stack(value, axis=0) if isinstance(value, (list, tuple)) else np.asarray(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group("data").create_group("demo")
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
            f.attrs[k] = v


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


def _set_fixed_object_pose(env_cfg: ManagerBasedRLEnvCfg, pose: tuple[float, float, float]) -> None:
    env_cfg.commands.object_pose.ranges.pos_x = (pose[0], pose[0])
    env_cfg.commands.object_pose.ranges.pos_y = (pose[1], pose[1])
    env_cfg.commands.object_pose.ranges.pos_z = (pose[2], pose[2])


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


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed

    target_poses = _parse_target_poses(args_cli.target_poses)
    for pose_idx, pose in enumerate(target_poses):
        save_pose_idx = args_cli.save_pose_index if args_cli.save_pose_index is not None else pose_idx
        print(f"[POSE] start pose{save_pose_idx} target={pose}")
        _set_fixed_object_pose(env_cfg, pose)

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.headless else None)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        print(f"[POSE] env created for pose{pose_idx}")

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(retrieve_file_path(args_cli.checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        counts = 0
        attempts = 0
        while counts < args_cli.num_demos:
            attempts += 1
            if attempts > args_cli.max_attempts_per_pose:
                print(f"[SKIP] pose{pose_idx} reached max attempts ({args_cli.max_attempts_per_pose})")
                break
            obs = _reset_env(env)
            _settle_env(env, policy, args_cli.settle_steps)
            obs = env.get_observations()
            if attempts == 1:
                cmd = env.unwrapped.command_manager.get_command("object_pose")[:, :3]
                print(f"[POSE] pose{save_pose_idx} command={_squeeze_env(cmd)}")

            cmd = env.unwrapped.command_manager.get_command("object_pose")[:, :3]

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
                des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, cmd)
                err = torch.norm(des_pos_w - obj_pos, dim=1)
                if args_cli.debug_interval > 0 and (ep_steps % args_cli.debug_interval) == 0:
                    print(f"[POSE] pose{save_pose_idx} step={ep_steps} err={float(err.max().item()):.4f}")
                if torch.any(err < args_cli.success_threshold):
                    success = True
                    break

                ep_steps += 1
                if args_cli.max_steps > 0 and ep_steps >= args_cli.max_steps:
                    timed_out = True
                    obs = next_obs
                    break
                obs = next_obs

            if success:
                if not near_goal(env, args_cli.success_threshold):
                    print(f"[SKIP] episode not near goal (tol={args_cli.success_threshold:.4f})")
                    continue
                counts += 1
                ep_idx = counts - 1
                data = {
                    "obs": {k: np.stack(v, axis=0) for k, v in episode["obs"].items()},
                    "actions": np.stack(episode["actions"], axis=0),
                    "rewards": np.stack(episode["rewards"], axis=0),
                    "dones": np.stack(episode["dones"], axis=0),
                    "robot_states": np.stack(episode["robot_states"], axis=0),
                }
                out_dir = Path(args_cli.dataset_root) / f"pose{save_pose_idx}"
                out_path = out_dir / f"demo_pose{save_pose_idx}_{ep_idx}.hdf5"
                save_hdf5(out_path, data, {"success": success, "target_pose": pose})
                print(f"[SAVED] {out_path}")
            else:
                if timed_out:
                    print(f"[SKIP] failed episode (max_steps={args_cli.max_steps})")
                else:
                    print("[SKIP] failed episode")

        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
