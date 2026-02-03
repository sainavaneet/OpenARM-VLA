#!/bin/bash


ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"


unset DISPLAY
export PYOPENGL_PLATFORM=egl

# export LD_LIBRARY_PATH=/home/navaneet/miniconda3/envs/openarm/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH

/workspace/isaaclab/_isaac_sim/python.sh "${ROOT_DIR}/src/generate_dataset.py" \
  --headless \
  --enable_cameras \
  --dataset_config "${ROOT_DIR}/conf/generate_dataset.yaml" \
  --tasks_config "${ROOT_DIR}/conf/tasks.yaml"
  
