#!/bin/bash


python src/train.py \
data_directory=/home/navaneet/OPENARM/OpenARM-VLA/datasets/openarm_cube_lift_two_pose_tasks \
batch_size=256 \
num_epochs=2000 \
save_freq=100 \
max_len_data=100 \
demos_per_task=100 \
learning_rate=1e-4 \
wandb.name=cube_lift_two_pose_tasks \
wandb.project=OpenARM \
wandb.entity=sainavaneet \
wandb.enabled=true \
seed=42
