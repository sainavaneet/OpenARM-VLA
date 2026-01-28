#!/bin/bash


python src/train.py \
data_directory=/home/navaneet/OPENARM/OpenARM-VLA/datasets/openarm_pick_the_color_cube_tasks \
batch_size=256 \
num_epochs=2000 \
save_freq=100 \
demos_per_task=50 \
learning_rate=3e-5 \
wandb.name=pick_the_color_cube_tasks \
wandb.project=OpenARM \
wandb.entity=sainavaneet \
wandb.enabled=true \
seed=42
