#!/bin/bash


/workspace/isaaclab/_isaac_sim/python.sh src/train.py \
data_directory=/workspace/OpenARM-VLA/datasets/openarm_cube_lift_two_pose_tasks \
batch_size=256 \
num_epochs=500 \
save_freq=100 \
max_len_data=100 \
demos_per_task=100 \
learning_rate=3e-5 \
model_type=transformer \
wandb.name=cube_lift_two_pose_tasks \
wandb.project=OpenARM \
wandb.entity=sainavaneet \
wandb.enabled=false \
seed=42
