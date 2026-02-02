#!/bin/bash


/workspace/isaaclab/_isaac_sim/python.sh src/train.py \
batch_size=256 \
num_epochs=500 \
save_freq=100 \
max_len_data=100 \
demos_per_task=50 \
learning_rate=1e-4 \
model_type=mamba \
wandb.name=cube_lift_direction_tasks \
wandb.project=OpenARM \
wandb.entity=sainavaneet \
wandb.enabled=false \
seed=42
