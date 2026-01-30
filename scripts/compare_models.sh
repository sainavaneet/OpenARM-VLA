#!/usr/bin/env bash
set -euo pipefail

unset DISPLAY
export PYOPENGL_PLATFORM=egl

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif [[ -x "/workspace/isaaclab/_isaac_sim/python.sh" ]]; then
  PYTHON_BIN="/workspace/isaaclab/_isaac_sim/python.sh"
else
  echo "ERROR: python not found (tried python3, python, /workspace/isaaclab/_isaac_sim/python.sh)" >&2
  exit 1
fi
DATA_DIR_CFG="$("$PYTHON_BIN" - <<'PY'
from omegaconf import OmegaConf
from pathlib import Path
print(Path(OmegaConf.load("conf/generate_dataset.yaml").dataset_root).resolve())
PY
)"
DATASET_NAME="$(basename "${DATA_DIR_CFG}")"
if [[ -d "${DATA_DIR_CFG}" ]]; then
  DATA_DIR="${DATA_DIR_CFG}"
elif [[ -d "${ROOT_DIR}/datasets/${DATASET_NAME}" ]]; then
  DATA_DIR="${ROOT_DIR}/datasets/${DATASET_NAME}"
else
  echo "ERROR: dataset directory not found: ${DATA_DIR_CFG}" >&2
  exit 1
fi
TASK_NAME="Isaac-Lift-Cube-OpenArm-Play-v0"
NUM_EPOCHS=3000
SAVE_FREQ=500
BATCH_SIZE=256
MAX_LEN=100
DEMOS=100
LR=1e-4
SEED=42
ROLLOUTS_PER_SLOT=50
NUM_ENVS=1
SETTLE_STEPS=50
MAX_STEPS=100
WANDB_ENABLED=False
WANDB_NAME="cube_lift_direction_tasks_compare"
WANDB_PROJECT="OpenARM_compare_runs"
WANDB_ENTITY="sainavaneet"

DATE_STAMP="$(date +%Y-%m-%d)"
TIME_STAMP="$(date +%H-%M-%S)"
OUT_ROOT="${ROOT_DIR}/outputs/compare_runs"
OUT_DIR="${OUT_ROOT}/${DATE_STAMP}/${TIME_STAMP}"
export OUT_DIR

if ! mkdir -p "${OUT_DIR}"; then
  OUT_ROOT="${HOME}/OpenARM-outputs/compare_runs"
  OUT_DIR="${OUT_ROOT}/${DATE_STAMP}/${TIME_STAMP}"
  export OUT_DIR
  if ! mkdir -p "${OUT_DIR}"; then
    OUT_ROOT="/tmp/OpenARM-outputs/compare_runs"
    OUT_DIR="${OUT_ROOT}/${DATE_STAMP}/${TIME_STAMP}"
    export OUT_DIR
    mkdir -p "${OUT_DIR}"
  fi
fi
LOG_FILE="${OUT_DIR}/compare_run.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

resolve_checkpoint() {
  local run_dir="$1"
  if [[ -f "${run_dir}/final_model.pth" ]]; then
    echo "${run_dir}/final_model.pth"
    return
  fi
  if [[ -f "${run_dir}/final_model.pt" ]]; then
    echo "${run_dir}/final_model.pt"
    return
  fi
  local latest
  latest="$(ls -1t "${run_dir}"/epoch_*.pth "${run_dir}"/epoch_*.pt 2>/dev/null | head -n1 || true)"
  if [[ -z "${latest}" ]]; then
    echo "ERROR: No checkpoint found in ${run_dir}" >&2
    exit 1
  fi
  echo "${latest}"
}

train_model() {
  local model_type="$1"
  local run_dir="${ROOT_DIR}/outputs/${DATE_STAMP}/${TIME_STAMP}/train/${model_type}/$(basename "${DATA_DIR}")"
  mkdir -p "${run_dir}"

  echo "[INFO] Training ${model_type} -> ${run_dir}" >&2
  local start_ts end_ts
  start_ts="$(date +%s)"

  (
    /workspace/isaaclab/_isaac_sim/python.sh src/train.py \
      data_directory="${DATA_DIR}" \
      batch_size="${BATCH_SIZE}" \
      num_epochs="${NUM_EPOCHS}" \
      save_freq="${SAVE_FREQ}" \
      max_len_data="${MAX_LEN}" \
      demos_per_task="${DEMOS}" \
      learning_rate="${LR}" \
      model_type="${model_type}" \
      wandb.enabled=${WANDB_ENABLED} \
      wandb.name=${WANDB_NAME} \
      wandb.project=${WANDB_PROJECT} \
      wandb.entity=${WANDB_ENTITY} \
      seed="${SEED}" \
      save_dir="${run_dir}"
  ) 1>&2

  end_ts="$(date +%s)"
  local train_seconds=$((end_ts - start_ts))
  printf "%s\n" "${train_seconds}" > "${OUT_DIR}/train_${model_type}_seconds.txt"

  echo "${run_dir}"
}

eval_model() {
  local model_type="$1"
  local ckpt="$2"
  local eval_cfg="${ROOT_DIR}/conf/eval_model.yaml"
  local eval_cfg_bak
  eval_cfg_bak="$(mktemp)"
  cp "${eval_cfg}" "${eval_cfg_bak}"
  cat > "${eval_cfg}" <<EOF
model_checkpoint: ${ckpt}
run_dir: null
lang_emb: null
dataset_root: ${DATA_DIR}
model_type: ${model_type}
checkpoints_root: ${ROOT_DIR}/outputs
num_envs: ${NUM_ENVS}
num_rollouts: ${ROLLOUTS_PER_SLOT}
output_metrics: ${OUT_DIR}/${model_type}.json
video: false
video_length: 200
task: ${TASK_NAME}
real_time: false
settle_steps: ${SETTLE_STEPS}
max_steps: ${MAX_STEPS}
success_threshold: 0.04
task_keys: null
tasks_config: ${ROOT_DIR}/conf/tasks.yaml
EOF

  /workspace/isaaclab/_isaac_sim/python.sh src/eval.py \
    --config "${ROOT_DIR}/conf/config.yaml" \
    --enable_cameras \
    --headless

  mv "${eval_cfg_bak}" "${eval_cfg}"
}

run_dir_mamba="$(train_model mamba)"
ckpt_mamba="$(resolve_checkpoint "${run_dir_mamba}")"
eval_model mamba "${ckpt_mamba}"

run_dir_transformer="$(train_model transformer)"
ckpt_transformer="$(resolve_checkpoint "${run_dir_transformer}")"
eval_model transformer "${ckpt_transformer}"

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])

def load_metrics(model):
    path = out_dir / f"{model}.json"
    return json.loads(path.read_text())

def load_train_seconds(model):
    path = out_dir / f"train_{model}_seconds.txt"
    return int(path.read_text().strip())

summary = {"run_dir": str(out_dir), "models": {}}

for model in ("mamba", "transformer"):
    metrics = load_metrics(model)
    summary["models"][model] = {
        "train_seconds": load_train_seconds(model),
        "tasks": metrics,
    }

summary_path = out_dir / "compare_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(f"[INFO] Wrote {summary_path}")
PY
