#!/usr/bin/env bash
set -euo pipefail

unset DISPLAY
export PYOPENGL_PLATFORM=egl

DATA_DIR="/workspace/OpenARM-VLA/datasets/openarm_cube_lift_two_pose_tasks"
TASK_NAME="Isaac-Lift-Cube-OpenArm-Play-v0"
NUM_EPOCHS=500
SAVE_FREQ=100
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
WANDB_NAME="cube_lift_two_pose_tasks"
WANDB_PROJECT="OpenARM_compare_runs"
WANDB_ENTITY="sainavaneet"
RUN_ROOT="/workspace/OpenARM-VLA/checkpoints/compare_runs"
OUT_ROOT="/workspace/OpenARM-VLA/outputs/compare_runs"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_DIR="${OUT_ROOT}/${STAMP}"

mkdir -p "${OUT_DIR}"

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
  local run_dir="${RUN_ROOT}/${model_type}/${STAMP}"
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

  for slot in 0 1; do
    /workspace/isaaclab/_isaac_sim/python.sh src/eval.py \
      --task "${TASK_NAME}" \
      --mamba_checkpoint "${ckpt}" \
      --dataset_root "${DATA_DIR}" \
      --model_type "${model_type}" \
      --num_envs "${NUM_ENVS}" \
      --settle_steps "${SETTLE_STEPS}" \
      --target_slot "${slot}" \
      --max_steps "${MAX_STEPS}" \
      --enable_cameras \
      --headless \
      --num_rollouts "${ROLLOUTS_PER_SLOT}" \
      --output_metrics "${OUT_DIR}/${model_type}_slot${slot}.json"
  done
}

run_dir_mamba="$(train_model mamba)"
ckpt_mamba="$(resolve_checkpoint "${run_dir_mamba}")"
eval_model mamba "${ckpt_mamba}"

run_dir_transformer="$(train_model transformer)"
ckpt_transformer="$(resolve_checkpoint "${run_dir_transformer}")"
eval_model transformer "${ckpt_transformer}"

/workspace/isaaclab/_isaac_sim/python.sh - <<'PY'
import json
from pathlib import Path

out_dir = Path("/workspace/OpenARM-VLA/outputs/compare_runs")
latest = sorted(out_dir.glob("*"))[-1]

def load_metrics(model, slot):
    path = latest / f"{model}_slot{slot}.json"
    return json.loads(path.read_text())

def load_train_seconds(model):
    path = latest / f"train_{model}_seconds.txt"
    return int(path.read_text().strip())

summary = {"run_dir": str(latest), "models": {}}

for model in ("mamba", "transformer"):
    m0 = load_metrics(model, 0)
    m1 = load_metrics(model, 1)
    summary["models"][model] = {
        "train_seconds": load_train_seconds(model),
        "slots": {"left": m0, "right": m1},
        "avg_success_rate": (m0["success_rate"] + m1["success_rate"]) / 2.0,
        "avg_inference_ms": (m0["avg_inference_ms"] + m1["avg_inference_ms"]) / 2.0,
        "avg_episode_steps": (m0["avg_episode_steps"] + m1["avg_episode_steps"]) / 2.0,
    }

summary_path = latest / "compare_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(f"[INFO] Wrote {summary_path}")
PY
