"""Run OpenARM-VLA inference in the OpenArm fixed-position task (dataset-consistent)."""

import argparse
import json
import os
import sys
import time

import torch
from omegaconf import OmegaConf

from src.utils import (
    _build_eval_metrics,
    _prepare_eval_cfg,
    _set_eval_video_task_folder,
    _setup_eval_video_recording,
    _rename_latest_video,
)


CLI_ARGS_DIR = "openarm_isaac_lab/scripts/reinforcement_learning/rsl_rl"
if CLI_ARGS_DIR not in sys.path:
    sys.path.insert(0, CLI_ARGS_DIR)

import cli_args  # isort: skip

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="conf/config.yaml", help="Path to OpenARM-VLA config.yaml (defaults to ./config.yaml).")
parser.add_argument("--model_type", type=str, default=None, help="Override eval model type (e.g., mamba).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)


args_cli, hydra_args = parser.parse_known_args()


def _load_eval_cfg(config_path):
    cfg = OmegaConf.load(config_path)
    eval_cfg = cfg.get("eval_model")
    if eval_cfg is None:
        cfg_dir = os.path.dirname(os.path.abspath(config_path))
        fallback = os.path.join(cfg_dir, "eval_model.yaml")
        if os.path.exists(fallback):
            eval_cfg = OmegaConf.load(fallback)
        else:
            raise SystemExit("eval_model section is required in config.")
    return eval_cfg





eval_cfg = _load_eval_cfg(args_cli.config)
if args_cli.checkpoint:
    eval_cfg.model_checkpoint = args_cli.checkpoint
if args_cli.model_type:
    eval_cfg.model_type = args_cli.model_type
_prepare_eval_cfg(eval_cfg)
print(f"[INFO] Using checkpoint: {eval_cfg.model_checkpoint}")

if eval_cfg.get("video", False):
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import openarm.tasks  # noqa: F401



def _zero_actions(env):
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



def _update_episode_state(env, dones, success_threshold, target_pos=None):
    obj = env.unwrapped.scene[_auto_target_object(env)]
    obj_pos = obj.data.root_pos_w[:, :3]
    if target_pos is None:
        cmd = env.unwrapped.command_manager.get_command("object_pose")[:, :3]
        error = torch.norm(cmd - obj_pos, dim=1)
    else:
        target = torch.tensor(target_pos, device=obj_pos.device, dtype=obj_pos.dtype)
        error = torch.norm(target - obj_pos, dim=1)
    success_mask = error < success_threshold
    done_mask = (dones > 0.5)
    return done_mask, success_mask, error


def _print_episode_success(
    env,
    task_key,
    task_name,
    task_index,
    done_mask,
    success_mask,
    episode_steps,
    error,
):
    if not torch.any(done_mask):
        return
    obj = env.unwrapped.scene[_auto_target_object(env)]
    heights = obj.data.root_pos_w[:, 2]
    done_ids = torch.where(done_mask)[0].tolist()
    for i in done_ids:
        status = "SUCCESS" if success_mask[i].item() else "FAIL"
        steps = int(episode_steps[i].item())
        print(
            f"[EPISODE] env={i} task={task_index} name={task_name} {status} "
            f"height={heights[i].item():.3f} "
            f"steps={steps} err={error[i].item():.3f}"
        )




def _parse_target_pose(raw):
    entry = raw.split(";")[0].strip()
    if not entry:
        raise ValueError("Empty target_pose in tasks.yaml.")
    parts = [p.strip() for p in entry.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid target_pose '{entry}'. Expected x,y,z.")
    return tuple(float(p) for p in parts)


def _set_fixed_object_pose_env(env, pose):
    cmd_cfg = getattr(getattr(env.unwrapped, "command_manager", None), "cfg", None)
    if cmd_cfg is None or not hasattr(cmd_cfg, "object_pose"):
        raise RuntimeError("Could not access command_manager.cfg.object_pose to set target pose.")
    cmd_cfg.object_pose.ranges.pos_x = (pose[0], pose[0])
    cmd_cfg.object_pose.ranges.pos_y = (pose[1], pose[1])
    cmd_cfg.object_pose.ranges.pos_z = (pose[2], pose[2])


def _auto_target_object(env):
    scene_keys = set(env.unwrapped.scene.keys())
    if "object" in scene_keys:
        return "object"
    candidates = [
        key for key in scene_keys
        if key not in {"robot", "terrain", "ground", "table"}
        and not key.startswith("camera")
    ]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        chosen = sorted(candidates)[0]
        print(f"[WARN] Multiple scene objects found; using '{chosen}'.")
        return chosen
    raise KeyError(f"Could not infer target object. Scene keys: {sorted(scene_keys)}")




def _build_policy(
    checkpoint_path,
    run_dir,
    dataset_root,
    device,
    lang_emb_dim,
    config_path,
    model_type,
    allowed_tasks,
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
            allowed_tasks=allowed_tasks,
        )
        scaler = MinMaxScaler(dataset.get_all_actions(), True, device)
        model.set_scaler(scaler)

    model.eval()
    model.reset()
    return model


def _build_mamba_obs(env, lang_input, device):
    cam_link0 = env.unwrapped.scene.sensors["camera_link0"].data.output["rgb"]
    cam_fixed = env.unwrapped.scene.sensors["camera_fixed"].data.output["rgb"]
    robot = env.unwrapped.scene["robot"]
    joint_pos = robot.data.joint_pos[:, :7]
    gripper = robot.data.joint_pos[:, -2:]
    robot_states = torch.cat([joint_pos, gripper], dim=1)

    agentview = cam_fixed.float().permute(0, 3, 1, 2) / 255.0
    eye_in_hand = cam_link0.float().permute(0, 3, 1, 2) / 255.0

    batch = agentview.shape[0]
    obs = {
        "agentview_image": agentview.unsqueeze(1).to(device),
        "eye_in_hand_image": eye_in_hand.unsqueeze(1).to(device),
        "robot_states": robot_states.unsqueeze(1).to(device),
    }
    if isinstance(lang_input, torch.Tensor):
        if lang_input.dim() == 1:
            lang_batch = lang_input.unsqueeze(0).repeat(batch, 1)
        else:
            lang_batch = lang_input
            if lang_batch.shape[0] != batch:
                lang_batch = lang_batch.repeat(batch, 1)
        obs["lang_emb"] = lang_batch.unsqueeze(1).to(device)
    else:
        if isinstance(lang_input, (list, tuple)):
            if len(lang_input) == batch:
                lang_list = [str(x) for x in lang_input]
            elif len(lang_input) == 1:
                lang_list = [str(lang_input[0])] * batch
            else:
                lang_list = [str(lang_input[0])] * batch
        else:
            lang_list = [str(lang_input)] * batch
        obs["lang"] = lang_list
    return obs


@hydra_task_config(eval_cfg.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = eval_cfg.num_envs if eval_cfg.get("num_envs") is not None else env_cfg.scene.num_envs
    env_cfg.episode_length_s = 30.0
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    tasks_cfg = OmegaConf.load(eval_cfg.tasks_config)
    if "tasks" not in tasks_cfg:
        raise SystemExit(f"tasks not found in {eval_cfg.tasks_config}.")
    task_keys = (
        [k.strip() for k in eval_cfg.task_keys.split(",") if k.strip()]
        if eval_cfg.get("task_keys")
        else sorted(tasks_cfg.tasks.keys())
    )

    render_mode = "rgb_array" if (eval_cfg.get("video", False) or args_cli.enable_cameras) else None
    env = gym.make(eval_cfg.task, cfg=env_cfg, render_mode=render_mode)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    video_root = None
    if eval_cfg.get("video", False):
        env, video_root = _setup_eval_video_recording(
            env,
            eval_cfg,
            gym.wrappers.RecordVideo,
            print_dict=print_dict,
            sensor_name="main_camera",
        )

    policy = _build_policy(
        checkpoint_path=eval_cfg.model_checkpoint,
        run_dir=eval_cfg.run_dir,
        dataset_root=eval_cfg.dataset_root,
        device=env.unwrapped.device,
        lang_emb_dim=int(eval_cfg.get("lang_emb_dim", 512)),
        config_path=args_cli.config,
        model_type=eval_cfg.model_type,
        allowed_tasks=[tasks_cfg.tasks[k].name for k in task_keys if k in tasks_cfg.tasks],
    )

    dt = env.unwrapped.step_dt
    all_metrics = []
    total_rollouts = 0
    total_successes = 0
    total_episode_lengths = []
    total_infer_time = 0.0
    total_infer_calls = 0

    try:
        for task_index, task_key in enumerate(task_keys):
            task_cfg = tasks_cfg.tasks.get(task_key)
            if task_cfg is None:
                print(f"[WARN] task_key '{task_key}' not found in tasks.yaml. Skipping.")
                continue
            if "target_pose" not in task_cfg or "name" not in task_cfg:
                print(f"[WARN] task '{task_key}' missing name/target_pose. Skipping.")
                continue

            lang_text = task_cfg.name
            task_video_dir = None
            video_episode_index = 0
            if eval_cfg.get("video", False):
                task_video_dir = _set_eval_video_task_folder(env, video_root, task_index, task_cfg.name)

            target_pos = _parse_target_pose(task_cfg.target_pose)
            _set_fixed_object_pose_env(env, target_pos)

            env.reset()
            policy.reset()
            for _ in range(eval_cfg.settle_steps):
                _step_env(env, _zero_actions(env))

            rollouts_done = 0
            success_count = 0
            episode_lengths = []
            infer_time_total = 0.0
            infer_calls = 0
            episode_steps = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device, dtype=torch.int32)

            while simulation_app.is_running() and rollouts_done < eval_cfg.num_rollouts:
                start_time = time.time()

                mamba_obs = _build_mamba_obs(env, lang_text, env.unwrapped.device)
                with torch.inference_mode():
                    infer_start = time.perf_counter()
                    actions = policy.predict(mamba_obs)
                    infer_time_total += time.perf_counter() - infer_start
                    infer_calls += 1
                if actions.dim() == 1:
                    actions = actions.unsqueeze(0)
                _, _, dones, _ = _step_env(env, actions)

                episode_steps += 1
                done_mask, success_mask, error = _update_episode_state(
                    env,
                    dones,
                    eval_cfg.success_threshold,
                    target_pos,
                )

                max_mask = episode_steps >= eval_cfg.max_steps
                done_mask = success_mask | max_mask
                if torch.any(done_mask):
                    _print_episode_success(
                        env,
                        task_key,
                        task_cfg.name,
                        task_index,
                        done_mask,
                        success_mask,
                        episode_steps,
                        error,
                    )
                    done_ids = torch.where(done_mask)[0].tolist()
                    for i in done_ids:
                        rollouts_done += 1
                        if success_mask[i].item():
                            success_count += 1
                        episode_lengths.append(int(episode_steps[i].item()))
                    if (
                        eval_cfg.get("video", False)
                        and task_video_dir
                        and 0 in done_ids
                    ):
                        status = "success" if success_mask[0].item() else "fail"
                        _rename_latest_video(task_video_dir, video_episode_index, status)
                        video_episode_index += 1
                    if rollouts_done >= eval_cfg.num_rollouts:
                        break
                    episode_steps = torch.where(done_mask, torch.zeros_like(episode_steps), episode_steps)
                    if rollouts_done < eval_cfg.num_rollouts:
                        env.reset()
                        policy.reset()
                        for _ in range(eval_cfg.settle_steps):
                            _step_env(env, _zero_actions(env))

                sleep_time = dt - (time.time() - start_time)
                if eval_cfg.get("real_time", False) and sleep_time > 0:
                    time.sleep(sleep_time)

            total_rollouts += rollouts_done
            total_successes += success_count
            total_episode_lengths.extend(episode_lengths)
            total_infer_time += infer_time_total
            total_infer_calls += infer_calls

            
            metrics = _build_eval_metrics(
                model_type=eval_cfg.model_type,
                rollouts_done=rollouts_done,
                success_count=success_count,
                episode_lengths=episode_lengths,
                infer_time_total=infer_time_total,
                infer_calls=infer_calls,
                task_name=task_cfg.name,
            )
            all_metrics.append(metrics)
            print(f"[METRICS] {json.dumps(metrics, indent=2)}")
        if all_metrics:
            overall_metrics = _build_eval_metrics(
                model_type=eval_cfg.model_type,
                rollouts_done=total_rollouts,
                success_count=total_successes,
                episode_lengths=total_episode_lengths,
                infer_time_total=total_infer_time,
                infer_calls=total_infer_calls,
                tasks=len(all_metrics),
            )
            print(f"[OVERALL_METRICS] {json.dumps(overall_metrics, indent=2)}")
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user. Shutting down...")
    finally:
        if eval_cfg.get("output_metrics"):
            out_path = os.path.abspath(os.path.expanduser(eval_cfg.output_metrics))
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "w") as f:
                f.write(json.dumps(all_metrics, indent=2))
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
