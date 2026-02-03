#!/usr/bin/env bash
# export LD_LIBRARY_PATH=/home/navaneet/miniconda3/envs/openarm/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
unset DISPLAY
export PYOPENGL_PLATFORM=egl

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/outputs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/eval_model_$(date +%Y-%m-%d_%H-%M-%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

CHECKPOINT="${1:-}"
if [[ -z "${CHECKPOINT}" ]]; then
  echo "Usage: $0 /path/to/checkpoint.pth" >&2
  exit 1
fi

python src/eval.py \
    --enable_cameras \
    --headless \
    --model_type mamba \
    --checkpoint "${CHECKPOINT}"
