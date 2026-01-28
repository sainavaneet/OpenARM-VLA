#!/usr/bin/env bash
# export LD_LIBRARY_PATH=/home/navaneet/miniconda3/envs/openarm/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH

python src/eval.py \
--task Isaac-Lift-TwoCube-OpenArm-Play-v0 \
--mamba_checkpoint /home/navaneet/OPENARM/OpenARM-VLA/checkpoints/openarm_pick_the_color_cube_tasks/2026-01-29/00-54-44/epoch_00800.pth \
--dataset_root /home/navaneet/OPENARM/OpenARM-VLA/datasets/openarm_pick_the_color_cube_tasks \
--target_slot 0 \
--num_envs 1 \
--settle_steps 50 \
--max_steps 100 \
--enable_cameras \
--headless
