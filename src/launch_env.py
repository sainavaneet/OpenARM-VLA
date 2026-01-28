from __future__ import annotations

import argparse
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Lift-TwoCube-OpenArm-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--real-time", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_tasks.utils.hydra import hydra_task_config
import openarm.tasks  # noqa: F401




@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, _agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.headless else None,
    )
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env.reset()

    while simulation_app.is_running():
        action_np = env.action_space.sample()
        action = torch.zeros_like(torch.as_tensor(action_np, device=env.unwrapped.device))
        env.step(action)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
