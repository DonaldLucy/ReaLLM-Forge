#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="${OUT_DIR:-results/side_rehearsal_last_mlp_down_strict_fast_1xa100_20k}"
RUN_NAME="${RUN_NAME:-side_rehearsal_last_mlp_down_strict_fast_1xa100_20k}"
DATASET="${DATASET:-minipile}"
PYTHON_BIN="${PYTHON_BIN:-/home/cmx/miniconda3/envs/realforge/bin/python}"

N_LAYER="${N_LAYER:-18}"
N_HEAD="${N_HEAD:-16}"
N_KV_GROUP="${N_KV_GROUP:-8}"
N_EMBD="${N_EMBD:-1024}"
QUANT_BITS="${QUANT_BITS:-8}"

BLOCK_SIZE="${BLOCK_SIZE:-512}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-6}"
MAX_ITERS="${MAX_ITERS:-20000}"

EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
EVAL_ITERS="${EVAL_ITERS:-10}"
SAVE_MAJOR_CKPT_INTERVAL="${SAVE_MAJOR_CKPT_INTERVAL:-2000}"
LOG_FILE="${LOG_FILE:-${OUT_DIR}/launcher.log}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUT_DIR}"

LATEST_CKPT=""
if [ -f "${OUT_DIR}/ckpt.pt" ]; then
  LATEST_CKPT="ckpt.pt"
else
  LATEST_CKPT="$(find "${OUT_DIR}" -maxdepth 1 -type f -name '[0-9]*.pt' | sed 's#^.*/##' | sort -V | tail -n 1)"
fi

RESUME_ARGS=()
if [ -n "${LATEST_CKPT}" ]; then
  echo "[resume] found ${OUT_DIR}/${LATEST_CKPT}, resuming" | tee -a "${LOG_FILE}"
  RESUME_ARGS+=(--init_from resume --init_from_ckpt "${LATEST_CKPT}")
else
  echo "[resume] no checkpoint found, starting fresh" | tee -a "${LOG_FILE}"
fi

TRAIN_ARGS=(
  train.py
  --dataset "${DATASET}"
  --out_dir "${OUT_DIR}"
  --tensorboard_run_name "${RUN_NAME}"
  --csv_name "${RUN_NAME}"
  --device cuda
  --dtype bfloat16
  --backend nccl
  --no-compile
  --n_layer "${N_LAYER}"
  --n_head "${N_HEAD}"
  --n_kv_group "${N_KV_GROUP}"
  --n_embd "${N_EMBD}"
  --block_size "${BLOCK_SIZE}"
  --batch_size "${BATCH_SIZE_PER_GPU}"
  --gradient_accumulation_steps 1
  --max_iters "${MAX_ITERS}"
  --eval_interval "${EVAL_INTERVAL}"
  --eval_iters "${EVAL_ITERS}"
  --log_interval 20
  --always_save_checkpoint
  --save_major_ckpt_interval "${SAVE_MAJOR_CKPT_INTERVAL}"
  --learning_rate 3e-4
  --warmup_iters 200
  --lr_scheduler cosine
  --quantization_warmup_iters 0
  --linear_variant_mlp_down quantized_linear
  --quantize_linear_mlp_down_method symmetric_quant
  --quantize_linear_mlp_down_bits "${QUANT_BITS}"
  --use_side_rehearsal
  --side_rehearsal_targets last_mlp_down
  --side_rehearsal_every 10
  --side_rehearsal_hidden 32
  --side_rehearsal_lr 3e-4
  --side_rehearsal_piggyback_ratio 1.0
  --side_rehearsal_accept_margin 5e-5
  --side_rehearsal_commit
  "${RESUME_ARGS[@]}"
)

printf '[launch]' | tee -a "${LOG_FILE}"
printf ' %q' "${PYTHON_BIN}" "${TRAIN_ARGS[@]}" | tee -a "${LOG_FILE}"
printf '\n' | tee -a "${LOG_FILE}"

"${PYTHON_BIN}" "${TRAIN_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
