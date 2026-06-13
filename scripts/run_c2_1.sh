#!/usr/bin/env bash
# C2-1 第一变体 (``C2-1-MV1``) 实验脚本: 2 层 stacked trajectory encoder
#
# 设计目标:
# 1. 把 ``TrajectoryGenerator`` 的 trajectory encoder 由单层
#    ``nn.LSTMCell`` 升级为 2 层 stacked LSTMCell, 保持 hidden_size 不变
#    以隔离"深度 vs 宽度"这两个变量。
# 2. **不**改 state 分支、decoder、light embedding、loss — C2-1 是
#    孤立改动, 与 PLAN.md §6.3 分支 C2-1 的 "trajectory-level
#    modeling" 方向一致, **完全离开 state injection 路线**。
# 3. 与现有 50b warmup -> 50b refine 协议对齐, 保证可比较。
# 4. 训练完成后, 用 val + num_samples=20 与 test + num_samples=20
#    双口径评估。
#
# 用法:
#   C21_DEVICE=cuda C21_NUM_EPOCHS=100 ./scripts/run_c2_1.sh
# 或逐步:
#   ./scripts/run_c2_1.sh train
#   ./scripts/run_c2_1.sh eval_val
#   ./scripts/run_c2_1.sh eval_test
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-$ROOT_DIR/experiments/cyclestate/C2_1_trajectory_level}"
DEVICE="${C21_DEVICE:-cuda}"
DATASET_NAME="${C21_DATASET:-VTP_C}"
NUM_EPOCHS="${C21_NUM_EPOCHS:-100}"
BATCH_SIZE="${C21_BATCH_SIZE:-64}"
VAL_BATCH_SIZE="${C21_VAL_BATCH_SIZE:-8}"
NUM_VAL_SAMPLES="${C21_NUM_VAL_SAMPLES:-20}"
NUM_EVAL_SAMPLES="${C21_NUM_EVAL_SAMPLES:-20}"
VAL_DSET_TYPE="${C21_VAL_DSET_TYPE:-val}"
TRAIN_STAGE="${C21_TRAIN_STAGE:-warmup}"
SEED="${C21_SEED:-72}"

# 训练阶段: 50 warmup batches + 50 refine batches
WARMUP_EPOCHS="${C21_WARMUP_EPOCHS:-50}"
REFINE_EPOCHS="${C21_REFINE_EPOCHS:-50}"
# 50b 协议：与 ``warmup50_refine50_p0_seqgat_relation_v1`` 对齐——
# 每个 epoch 最多跑 50 个 batch（用 ``max_train_batches`` 控制）。
MAX_TRAIN_BATCHES="${C21_MAX_TRAIN_BATCHES:-50}"
MAX_VAL_BATCHES="${C21_MAX_VAL_BATCHES:-20}"

# model_type 默认走 cyclestate (与 DE-3 对齐), 也可以通过环境变量
# 切到 d2tpred 做 cleanest isolation test:
#   C21_MODEL_TYPE=d2tpred ./scripts/run_c2_1.sh train
MODEL_TYPE="${C21_MODEL_TYPE:-cyclestate}"
# minimal_viable_mode 默认走 DE-3 路径, 即 c2_1 + state hidden init 拼接
# 组合. 也可以关闭做 c2_1 alone 测试:
#   C21_MINIMAL_VIABLE=0 ./scripts/run_c2_1.sh train
MINIMAL_VIABLE_FLAG=""
if [[ "${C21_MINIMAL_VIABLE:-1}" == "1" ]]; then
  MINIMAL_VIABLE_FLAG="--minimal_viable_mode"
fi

mkdir -p "$LOG_DIR"

TRAIN_COMMON_ARGS=(
  --model_type "$MODEL_TYPE"
  --c2_1_trajectory_level_mode
  $MINIMAL_VIABLE_FLAG
  --train_stage "$TRAIN_STAGE"
  --device "$DEVICE"
  --dataset_name "$DATASET_NAME"
  --log_dir "$LOG_DIR"
  --batch_size "$BATCH_SIZE"
  --val_dset_type "$VAL_DSET_TYPE"
  --num_val_samples "$NUM_VAL_SAMPLES"
  --loader_num_workers 0
  --seed "$SEED"
  --graph_lstm_hidden_size 32
  --traj_lstm_hidden_size 32
  --max_train_batches "$MAX_TRAIN_BATCHES"
  --max_val_batches "$MAX_VAL_BATCHES"
)

EVAL_COMMON_ARGS=(
  --model_type "$MODEL_TYPE"
  --c2_1_trajectory_level_mode
  $MINIMAL_VIABLE_FLAG
  --device "$DEVICE"
  --dataset_name "$DATASET_NAME"
  --batch_size "$VAL_BATCH_SIZE"
  --num_samples "$NUM_EVAL_SAMPLES"
  --loader_num_workers 0
  --pin_memory
)

case "${1:-all}" in
  train)
    echo "[C2-1] training $WARMUP_EPOCHS warmup batches (stage=$TRAIN_STAGE, max_train_batches=$MAX_TRAIN_BATCHES) -> $LOG_DIR"
    # num_epochs=2 是为了让 ``range(0, num_epochs+1) == [0, 1]`` 真的
    # 跑 2 个 epoch（每个 epoch 内由 max_train_batches 截断），与
    # ``warmup50_refine50_p0_seqgat_relation_v1`` 的实际行为对齐。
    python D2TP/train.py \
      "${TRAIN_COMMON_ARGS[@]}" \
      --num_epochs 2
    ;;

  train_refine)
    REFINED_CKPT="$LOG_DIR/checkpoint/model_best.pth.tar"
    if [[ ! -f "$REFINED_CKPT" ]]; then
      echo "[C2-1] expected warmup checkpoint missing: $REFINED_CKPT" >&2
      exit 1
    fi
    echo "[C2-1] refining $REFINE_EPOCHS refine batches from $REFINED_CKPT -> $LOG_DIR"
    python D2TP/train.py \
      "${TRAIN_COMMON_ARGS[@]}" \
      --train_stage refine \
      --resume "$REFINED_CKPT" \
      --num_epochs 3
    ;;

  eval_val)
    CKPT="${2:-$LOG_DIR/checkpoint/model_best.pth.tar}"
    echo "[C2-1] evaluating val + num_samples=$NUM_EVAL_SAMPLES from $CKPT"
    python D2TP/evaluate_model.py \
      "${EVAL_COMMON_ARGS[@]}" \
      --dset_type val \
      --eval_print_every 10 \
      --resume "$CKPT"
    ;;

  eval_test)
    CKPT="${2:-$LOG_DIR/checkpoint/model_best.pth.tar}"
    echo "[C2-1] evaluating test + num_samples=$NUM_EVAL_SAMPLES from $CKPT"
    python D2TP/evaluate_model.py \
      "${EVAL_COMMON_ARGS[@]}" \
      --dset_type test \
      --eval_print_every 10 \
      --resume "$CKPT"
    ;;

  smoke)
    echo "[C2-1] smoke: 1-batch train + 1-batch val (no real training)"
    python D2TP/train.py \
      "${TRAIN_COMMON_ARGS[@]}" \
      --num_epochs 0 \
      --max_train_batches 1 \
      --max_val_batches 1 \
      --print_every 1
    CKPT="$LOG_DIR/checkpoint/model_best.pth.tar"
    if [[ ! -f "$CKPT" ]]; then
      echo "[C2-1] smoke produced no checkpoint; skipping eval"
      exit 1
    fi
    echo "[C2-1] smoke eval on val/test (1 batch each, num_samples=2)"
    python D2TP/evaluate_model.py \
      "${EVAL_COMMON_ARGS[@]}" \
      --dset_type val \
      --max_eval_batches 1 \
      --num_samples 2 \
      --resume "$CKPT"
    python D2TP/evaluate_model.py \
      "${EVAL_COMMON_ARGS[@]}" \
      --dset_type test \
      --max_eval_batches 1 \
      --num_samples 2 \
      --resume "$CKPT"
    echo "[C2-1] smoke OK"
    ;;

  all|*)
    "$0" smoke
    "$0" train
    "$0" eval_val
    "$0" eval_test
    ;;

esac
