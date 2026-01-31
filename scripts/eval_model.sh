#!/usr/bin/env bash
# export LD_LIBRARY_PATH=/home/navaneet/miniconda3/envs/openarm/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
unset DISPLAY
export PYOPENGL_PLATFORM=egl

/workspace/isaaclab/_isaac_sim/python.sh src/eval.py \
    --enable_cameras \
    --headless \
    --model_type transformer