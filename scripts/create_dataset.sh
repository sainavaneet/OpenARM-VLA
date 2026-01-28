#!/bin/bash


# unset DISPLAY
# export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH=/home/navaneet/miniconda3/envs/openarm/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH


NUM_DEMOS=50

# python src/generate_dataset.py \
#   --task Isaac-Lift-TwoCube-OpenArm-Play-v0 \
#   --target_slot 0 \
#   --num_demos $NUM_DEMOS \
#   --num_envs 1 \
#   --slot_tolerance 0.05 \
#   --settle_steps 50 \
#   --dataset_root /home/navaneet/OPENARM/OpenARM-VLA/datasets/two_cube \
#   --checkpoint /home/navaneet/OPENARM/OpenARM-VLA/src/RL_policy/model_1999.pt \
#   --enable_cameras \
#   # --headless 


#   python src/generate_dataset.py \
#   --task Isaac-Lift-TwoCube-OpenArm-Play-v0 \
#   --target_slot 1 \
#   --num_demos $NUM_DEMOS \
#   --num_envs 1 \
#   --slot_tolerance 0.05 \
#   --settle_steps 50 \
#   --dataset_root /home/navaneet/OPENARM/OpenARM-VLA/datasets/two_cube \
#   --checkpoint /home/navaneet/OPENARM/OpenARM-VLA/src/RL_policy/model_1999.pt \
#   --enable_cameras \
#   # --headless 



  python src/merge_demos.py \
  --root /home/navaneet/OPENARM/OpenARM-VLA/datasets/two_cube \
  --out /home/navaneet/OPENARM/OpenARM-VLA/datasets/openarm_pick_the_color_cube_tasks
