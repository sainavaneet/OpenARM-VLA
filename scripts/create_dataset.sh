#!/bin/bash


unset DISPLAY
export PYOPENGL_PLATFORM=egl
# export LD_LIBRARY_PATH=/home/navaneet/miniconda3/envs/openarm/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH


NUM_DEMOS=100

python src/generate_dataset.py \
  --task Isaac-Lift-Cube-OpenArm-Play-v0 \
  --num_demos $NUM_DEMOS \
  --num_envs 1 \
  --dataset_root /home/navaneet/OPENARM/OpenARM-VLA/datasets/cube_lift \
  --target_poses "0.25,0.3,0.25" \
  --save_pose_index 0 \
  --checkpoint /home/navaneet/OPENARM/OpenARM-VLA/src/RL_policy/model_1999.pt \
  --enable_cameras \
  --settle_steps 100 \
  --headless 



python src/generate_dataset.py \
  --task Isaac-Lift-Cube-OpenArm-Play-v0 \
  --num_demos $NUM_DEMOS \
  --num_envs 1 \
  --dataset_root /home/navaneet/OPENARM/OpenARM-VLA/datasets/cube_lift \
  --target_poses "0.25,-0.2,0.2" \
  --save_pose_index 1 \
  --checkpoint /home/navaneet/OPENARM/OpenARM-VLA/src/RL_policy/model_1999.pt \
  --enable_cameras \
  --settle_steps 100 \
  --headless 

TASK_MAP_JSON='{
  "pose0_task": "pick_the_cube_and_lift_it_to_the_left_side_of_the_table",
  "pose1_task": "pick_the_cube_and_reach_to_the_right_side_but_slighlty_lower"
}'


python src/merge_demos.py \
  --root /home/navaneet/OPENARM/OpenARM-VLA/datasets/cube_lift \
  --out /home/navaneet/OPENARM/OpenARM-VLA/datasets/openarm_cube_lift_two_pose_tasks \
  --task_map_json "$TASK_MAP_JSON"
