#!/usr/bin/env bash
# DE-3 决定性实验脚本：Minimum Viable CycleState
#
# 设计目标:
# 1. 把 CycleState 降到"最简可行"形态:
#    - 强制关闭 state_gating / queue_rollout / lane_queue_anchor /
#      decoder_state_residual / aux_losses 五个开关
#    - 观测期最后时刻的 [queue_last, cycle_last] 直接拼接到 decoder
#      初始化向量后面, 替代原加性残差
# 2. 与现有 50b warmup -> 50b refine 协议对齐, 保证可比较
# 3. 训练完成后, 用 val + num_samples=20 与 test + num_samples=20 双口径评估
#
# 用法:
#   DE3_DEVICE=cuda DE3_NUM_EPOCHS=100 ./scripts/run_de3.sh
# 或逐步:
#   ./scripts/run_de3.sh train
#   ./scripts/run_de3.sh eval_val
#   ./scripts/run_de3.sh eval_test
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-$ROOT_DIR/experiments/cyclestate/DE3_minimal_viable}"
DEVICE="${DE3_DEVICE:-cuda}"
DATASET_NAME="${DE3_DATASET:-VTP_C}"
NUM_EPOCHS="${DE3_NUM_EPOCHS:-100}"
BATCH_SIZE="${DE3_BATCH_SIZE:-64}"
VAL_BATCH_SIZE="${DE3_VAL_BATCH_SIZE:-8}"
NUM_VAL_SAMPLES="${DE3_NUM_VAL_SAMPLES:-20}"
NUM_EVAL_SAMPLES="${DE3_NUM_EVAL_SAMPLES:-20}"
VAL_DSET_TYPE="${DE3_VAL_DSET_TYPE:-val}"
TRAIN_STAGE="${DE3_TRAIN_STAGE:-warmup}"
SEED="${DE3_SEED:-72}"

# 训练阶段: 50 warmup batches + 50 refine batches
WARMUP_EPOCHS="${DE3_WARMUP_EPOCHS:-50}"
REFINE_EPOCHS="${DE3_REFINE_EPOCHS:-50}"
# 50b 协议：与 ``warmup50_refine50_p0_seqgat_relation_v1`` 对齐——
# 每个 epoch 最多跑 50 个 batch（用 ``max_train_batches`` 控制）。
MAX_TRAIN_BATCHES="${DE3_MAX_TRAIN_BATCHES:-50}"
MAX_VAL_BATCHES="${DE3_MAX_VAL_BATCHES:-20}"

mkdir -p "$LOG_DIR"

TRAIN_COMMON_ARGS=(
  --model_type cyclestate
  --minimal_viable_mode
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
  --model_type cyclestate
  --minimal_viable_mode
  --device "$DEVICE"
  --dataset_name "$DATASET_NAME"
  --batch_size "$VAL_BATCH_SIZE"
  --num_samples "$NUM_EVAL_SAMPLES"
  --loader_num_workers 0
  --pin_memory
)

case "${1:-all}" in
  train)
    echo "[DE-3] training $WARMUP_EPOCHS warmup batches (stage=$TRAIN_STAGE, max_train_batches=$MAX_TRAIN_BATCHES) -> $LOG_DIR"
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
      echo "[DE-3] expected warmup checkpoint missing: $REFINED_CKPT" >&2
      exit 1
    fi
    echo "[DE-3] refining $REFINE_EPOCHS refine batches from $REFINED_CKPT -> $LOG_DIR"
    # 用 num_epochs=3 是因为 checkpoint 会把 ``start_epoch`` 恢复到自身
    # 的 ``epoch`` 字段（DE-3 warmup 收尾于 epoch=2），主循环
    # ``range(start_epoch, num_epochs+1)`` 必须留出至少 1 个 slot 才会
    # 真的进入训练；留 3 是给后续可能的 cont 留余地。
    python D2TP/train.py \
      "${TRAIN_COMMON_ARGS[@]}" \
      --train_stage refine \
      --resume "$REFINED_CKPT" \
      --num_epochs 3
    ;;

  eval_val)
    CKPT="${2:-$LOG_DIR/checkpoint/model_best.pth.tar}"
    echo "[DE-3] evaluating val + num_samples=$NUM_EVAL_SAMPLES from $CKPT"
    python D2TP/evaluate_model.py \
      "${EVAL_COMMON_ARGS[@]}" \
      --dset_type val \
      --eval_print_every 10 \
      --resume "$CKPT"
    ;;

  eval_test)
    CKPT="${2:-$LOG_DIR/checkpoint/model_best.pth.tar}"
    echo "[DE-3] evaluating test + num_samples=$NUM_EVAL_SAMPLES from $CKPT"
    python D2TP/evaluate_model.py \
      "${EVAL_COMMON_ARGS[@]}" \
      --dset_type test \
      --eval_print_every 10 \
      --resume "$CKPT"
    ;;

  smoke)
    echo "[DE-3] smoke: 1-batch train + 1-batch val (no real training)"
    python D2TP/train.py \
      "${TRAIN_COMMON_ARGS[@]}" \
      --num_epochs 0 \
      --max_train_batches 1 \
      --max_val_batches 1 \
      --print_every 1
    CKPT="$LOG_DIR/checkpoint/model_best.pth.tar"
    if [[ ! -f "$CKPT" ]]; then
      echo "[DE-3] smoke produced no checkpoint; skipping eval"
      exit 1
    fi
    echo "[DE-3] smoke eval on val/test (1 batch each, num_samples=2)"
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
    echo "[DE-3] smoke OK"
    ;;

  all|*)
    "$0" smoke
    "$0" train
    "$0" eval_val
    "$0" eval_test
    ;;

esac
