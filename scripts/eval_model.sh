#!/usr/bin/env bash
# export LD_LIBRARY_PATH=/home/navaneet/miniconda3/envs/openarm/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
unset DISPLAY
export PYOPENGL_PLATFORM=egl

TARGET_SLOT=0

/workspace/isaaclab/_isaac_sim/python.sh src/eval.py \
--task Isaac-Lift-Cube-OpenArm-Play-v0 \
--mamba_checkpoint /workspace/OpenARM-VLA/checkpoints/openarm_cube_lift_two_pose_tasks/2026-01-29/12-40-08/epoch_00100.pth \
--dataset_root /workspace/OpenARM-VLA/datasets/openarm_cube_lift_two_pose_tasks \
--model_type mamba \
--num_envs 1 \
--settle_steps 50 \
--target_slot "$TARGET_SLOT" \
--max_steps 100 \
--enable_cameras \
--headless


