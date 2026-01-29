#!/usr/bin/env bash
export LD_LIBRARY_PATH=/home/navaneet/miniconda3/envs/openarm/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH

TARGET_SLOT=1

python src/eval.py \
--task Isaac-Lift-Cube-OpenArm-Play-v0 \
--mamba_checkpoint /home/navaneet/OPENARM/OpenARM-VLA/checkpoints/openarm_cube_lift_two_pose_tasks/2026-01-29/05-10-54/epoch_02000.pth \
--dataset_root /home/navaneet/OPENARM/OpenARM-VLA/datasets/openarm_cube_lift_two_pose_tasks \
--num_envs 1 \
--settle_steps 50 \
--target_slot "$TARGET_SLOT" \
--max_steps 100 \
--enable_cameras \
# --headless
