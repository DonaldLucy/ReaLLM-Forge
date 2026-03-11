#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Estimated 3-5h on 2x A100 for the default settings below.
# If you are on 40GB cards, start by lowering BATCH_SIZE_PER_GPU from 6 to 4.

OUT_DIR="${OUT_DIR:-results/side_rehearsal_last_mlp_down_2xa100}"
RUN_NAME="${RUN_NAME:-side_rehearsal_last_mlp_down_2xa100}"
DATASET="${DATASET:-minipile}"
NPROC="${NPROC:-2}"

N_LAYER="${N_LAYER:-18}"
N_HEAD="${N_HEAD:-16}"
N_KV_GROUP="${N_KV_GROUP:-8}"
N_EMBD="${N_EMBD:-1024}"
BLOCK_SIZE="${BLOCK_SIZE:-1024}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-6}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_ITERS="${MAX_ITERS:-6000}"

mkdir -p "${OUT_DIR}"

torchrun --standalone --nproc_per_node="${NPROC}" train.py \
  --dataset "${DATASET}" \
  --out_dir "${OUT_DIR}" \
  --tensorboard_run_name "${RUN_NAME}" \
  --csv_name "${RUN_NAME}" \
  --device cuda \
  --dtype bfloat16 \
  --backend nccl \
  --compile false \
  --n_layer "${N_LAYER}" \
  --n_head "${N_HEAD}" \
  --n_kv_group "${N_KV_GROUP}" \
  --n_embd "${N_EMBD}" \
  --block_size "${BLOCK_SIZE}" \
  --batch_size "${BATCH_SIZE_PER_GPU}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --max_iters "${MAX_ITERS}" \
  --eval_interval 200 \
  --eval_iters 50 \
  --log_interval 20 \
  --learning_rate 3e-4 \
  --warmup_iters 200 \
  --lr_scheduler cosine \
  --quantization_warmup_iters 0 \
  --linear_variant_mlp_down quantized_linear \
  --quantize_linear_mlp_down_method symmetric_quant \
  --quantize_linear_mlp_down_bits 8 \
  --use_side_rehearsal \
  --side_rehearsal_targets last_mlp_down \
  --side_rehearsal_every 10 \
  --side_rehearsal_hidden 32 \
  --side_rehearsal_lr 1e-4 \
  --side_rehearsal_piggyback_ratio 0.25 \
  --side_rehearsal_commit
