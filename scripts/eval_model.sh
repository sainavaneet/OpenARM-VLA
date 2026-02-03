#!/usr/bin/env bash
# export LD_LIBRARY_PATH=/home/navaneet/miniconda3/envs/openarm/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
unset DISPLAY
export PYOPENGL_PLATFORM=egl

LOG_DIR="/workspace/OpenARM-VLA/outputs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/eval_model_$(date +%Y-%m-%d_%H-%M-%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

python src/eval.py \
    --enable_cameras \
    --headless \
    --model_type mamba \
    --checkpoint /workspace/OpenARM-VLA/outputs/2026-02-02/11-19-37/train/mamba/openarm_cube_lift_direction_tasks2/epoch_00200.pth
