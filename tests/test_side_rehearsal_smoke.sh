#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="minipile"

OUT_ROOT="results/side_rehearsal_smoke"
CSV_DIR="${OUT_ROOT}/csv"
PLOT_DIR="${OUT_ROOT}/plots"
mkdir -p "${OUT_ROOT}" "${CSV_DIR}" "${PLOT_DIR}"

COMMON_ARGS=(
  --dataset "${DATASET}"
  --device cpu
  --dtype float32
  --no-compile
  --no-tensorboard_log
  --no-wandb_log
  --csv_dir "${CSV_DIR}"
  --max_iters 1200
  --eval_interval 6
  --eval_iters 6
  --log_interval 2
  --batch_size 4
  --block_size 64
  --n_layer 2
  --n_head 2
  --n_kv_group 2
  --n_embd 64
  --dropout 0.0
  --quantization_warmup_iters 0
  --linear_variant_mlp_down quantized_linear
  --quantize_linear_mlp_down_method symmetric_quant
  --quantize_linear_mlp_down_bits 8
)

python3 train.py \
  "${COMMON_ARGS[@]}" \
  --out_dir "${OUT_ROOT}/baseline" \
  --csv_name "baseline_smoke"

python3 train.py \
  "${COMMON_ARGS[@]}" \
  --out_dir "${OUT_ROOT}/side" \
  --csv_name "side_smoke" \
  --use_side_rehearsal \
  --side_rehearsal_targets last_mlp_down \
  --side_rehearsal_every 4 \
  --side_rehearsal_hidden 16 \
  --side_rehearsal_lr 1e-3 \
  --side_rehearsal_piggyback_ratio 0.25

python3 tools/plot_side_rehearsal_report.py \
  --baseline_bulk "${CSV_DIR}/bulk_baseline_smoke.csv" \
  --rehearsal_bulk "${CSV_DIR}/bulk_side_smoke.csv" \
  --side_eval_csv "${CSV_DIR}/side_side_smoke.csv" \
  --out_dir "${PLOT_DIR}" \
  --title "Side-Rehearsal Smoke Test"

echo "[OK] side rehearsal smoke test finished"
