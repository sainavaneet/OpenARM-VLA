from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import h5py
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate two-cube HDF5 demos (fixed positions).")
parser.add_argument("--task", type=str, default="Isaac-Lift-TwoCube-OpenArm-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_demos", type=int, default=50)
parser.add_argument("--target_slot", type=int, default=-1, help="0=red, 1=green, -1=random")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--dataset_root", type=str, default="./datasets/two_cube")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--success_threshold", type=float, default=0.035)
parser.add_argument("--slot_tolerance", type=float, default=0.05)
parser.add_argument("--max_reset_attempts", type=int, default=5)
parser.add_argument("--settle_steps", type=int, default=10)
parser.add_argument("--max_steps", type=int, default=50)
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

RIGHT_SLOT_POS = (0.40, 0.0, 0.055)
LEFT_SLOT_POS = (0.35, -0.20, 0.055)
DROP_HEIGHT = 0.02

SLOT_TO_COLOR = {0: "red", 1: "green"}
SLOT_TO_OBJECT = {0: "object_red", 1: "object_green"}


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


def set_object_pose(obj, xyz):
    root_state = obj.data.root_state_w.clone()
    root_state[:, 0:3] = torch.tensor(xyz, device=root_state.device)
    root_state[:, 7:13] = 0.0
    obj.write_root_state_to_sim(root_state)
    obj.write_root_velocity_to_sim(torch.zeros((root_state.shape[0], 6), device=root_state.device))
    if hasattr(obj, "root_physx_view"):
        if hasattr(obj.root_physx_view, "wake_up"):
            obj.root_physx_view.wake_up()
        elif hasattr(obj.root_physx_view, "wake"):
            obj.root_physx_view.wake()
        if hasattr(obj.root_physx_view, "set_rigid_body_disable_sleeping"):
            obj.root_physx_view.set_rigid_body_disable_sleeping(True)


def place_objects(env):
    scene = env.unwrapped.scene
    obj_target = scene["object"]
    obj_other = scene["distractor"]
    slot_red = (RIGHT_SLOT_POS[0], RIGHT_SLOT_POS[1], RIGHT_SLOT_POS[2] + DROP_HEIGHT)
    slot_green = (LEFT_SLOT_POS[0], LEFT_SLOT_POS[1], LEFT_SLOT_POS[2] + DROP_HEIGHT)
    if os.environ.get("OPENARM_TARGET_OBJECT", "object_red") == "object_green":
        set_object_pose(obj_target, slot_green)
        set_object_pose(obj_other, slot_red)
    else:
        set_object_pose(obj_target, slot_red)
        set_object_pose(obj_other, slot_green)


def slots_ok(env, tol):
    scene = env.unwrapped.scene
    obj_target = scene["object"].data.root_pos_w[:, :3]
    obj_other = scene["distractor"].data.root_pos_w[:, :3]
    red = torch.tensor(RIGHT_SLOT_POS, device=obj_target.device)
    green = torch.tensor(LEFT_SLOT_POS, device=obj_target.device)
    if os.environ.get("OPENARM_TARGET_OBJECT", "object_red") == "object_green":
        err_target = torch.norm(obj_target - green, dim=1)
        err_other = torch.norm(obj_other - red, dim=1)
    else:
        err_target = torch.norm(obj_target - red, dim=1)
        err_other = torch.norm(obj_other - green, dim=1)
    max_err = torch.max(torch.stack([err_target, err_other], dim=1), dim=1).values
    max_err_val = float(max_err.max().item())
    return max_err_val <= tol, max_err_val


def settle_and_check(env, policy) -> bool:
    for attempt in range(args_cli.max_reset_attempts):
        place_objects(env)
        for _ in range(args_cli.settle_steps):
            obs = env.get_observations()
            obs, _, _, _ = env.step(torch.zeros_like(policy(obs)))
        ok, max_err = slots_ok(env, args_cli.slot_tolerance)
        if ok:
            return True
        print(
            f"[RESET] placement error (attempt {attempt + 1}/{args_cli.max_reset_attempts}) "
            f"max_err={max_err:.4f} tol={args_cli.slot_tolerance:.4f}"
        )
    print(f"[SKIP] failed to place cubes (tol={args_cli.slot_tolerance:.4f})")
    return False


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


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.headless else None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    counts = {"red": 0, "green": 0}
    while sum(counts.values()) < args_cli.num_demos:
        slot = args_cli.target_slot if args_cli.target_slot >= 0 else random.randint(0, 1)
        os.environ["OPENARM_TARGET_OBJECT"] = SLOT_TO_OBJECT[slot]
        target_color = SLOT_TO_COLOR[slot]
        obs = env.get_observations()
        if not settle_and_check(env, policy):
            continue

        cmd = env.unwrapped.command_manager.get_command("object_pose")[:, :3]


        episode = {
            "obs": {"agentview_rgb": [], "eye_in_hand_rgb": [], "joint_states": [], "gripper_states": []},
            "actions": [],
            "rewards": [],
            "dones": [],
            "robot_states": [],
        }

        success = False
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
            if torch.any(torch.norm(des_pos_w - obj_pos, dim=1) < args_cli.success_threshold):
                success = True

            ep_steps += 1
            if ep_steps >= args_cli.max_steps:
                obs = next_obs
                break
            obs = next_obs

        if success:
            if not near_goal(env, args_cli.slot_tolerance):
                print(f"[SKIP] episode for {target_color} not near goal (tol={args_cli.slot_tolerance:.4f})")
                continue
            counts[target_color] += 1
            ep_idx = counts[target_color] - 1
            data = {
                "obs": {k: np.stack(v, axis=0) for k, v in episode["obs"].items()},
                "actions": np.stack(episode["actions"], axis=0),
                "rewards": np.stack(episode["rewards"], axis=0),
                "dones": np.stack(episode["dones"], axis=0),
                "robot_states": np.stack(episode["robot_states"], axis=0),
            }
            out_path = Path(args_cli.dataset_root) / f"demo_{target_color}_{ep_idx}.hdf5"
            save_hdf5(out_path, data, {"color": target_color, "success": success})
            print(f"[SAVED] {out_path}")
        else:
            print(f"[SKIP] failed episode for {target_color}")

    env.close()


if __name__ == "__main__":
    if args_cli.target_slot >= 0:
        os.environ["OPENARM_TARGET_OBJECT"] = SLOT_TO_OBJECT.get(args_cli.target_slot, "object_red")
    main()
    simulation_app.close()
