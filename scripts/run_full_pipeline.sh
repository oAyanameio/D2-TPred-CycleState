#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-$ROOT_DIR/experiments/full_pipeline_smoke}"
MODEL_TYPE="${MODEL_TYPE:-cyclestate}"
TRAIN_STAGE="${TRAIN_STAGE:-warmup}"
DEVICE="${DEVICE:-cpu}"
DATASET_NAME="${DATASET_NAME:-VTP_C}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-1}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-1}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-0}"
VAL_DSET_TYPE="${VAL_DSET_TYPE:-val}"
EVAL_DSET_TYPE="${EVAL_DSET_TYPE:-test}"
NUM_VAL_SAMPLES="${NUM_VAL_SAMPLES:-2}"
NUM_EVAL_SAMPLES="${NUM_EVAL_SAMPLES:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"

mkdir -p "$LOG_DIR"

echo "[run_full_pipeline] train start"
python D2TP/train.py \
  --model_type "$MODEL_TYPE" \
  --train_stage "$TRAIN_STAGE" \
  --device "$DEVICE" \
  --dataset_name "$DATASET_NAME" \
  --log_dir "$LOG_DIR" \
  --num_epochs "$NUM_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --max_train_batches "$MAX_TRAIN_BATCHES" \
  --max_val_batches "$MAX_VAL_BATCHES" \
  --val_dset_type "$VAL_DSET_TYPE" \
  --num_val_samples "$NUM_VAL_SAMPLES" \
  --loader_num_workers 0 \
  ${EXTRA_TRAIN_ARGS}

CKPT_PATH="$LOG_DIR/checkpoint/model_best.pth.tar"
if [[ ! -f "$CKPT_PATH" ]]; then
  echo "[run_full_pipeline] expected checkpoint missing: $CKPT_PATH" >&2
  exit 1
fi

echo "[run_full_pipeline] evaluate start"
python D2TP/evaluate_model.py \
  --model_type "$MODEL_TYPE" \
  --device "$DEVICE" \
  --dataset_name "$DATASET_NAME" \
  --dset_type "$EVAL_DSET_TYPE" \
  --resume "$CKPT_PATH" \
  --num_samples "$NUM_EVAL_SAMPLES" \
  --max_eval_batches "$MAX_EVAL_BATCHES" \
  --batch_size "$BATCH_SIZE" \
  --loader_num_workers 0 \
  ${EXTRA_EVAL_ARGS}

echo "[run_full_pipeline] done: $CKPT_PATH"
