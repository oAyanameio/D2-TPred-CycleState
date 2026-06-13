#!/usr/bin/env bash
# AR-2 决定性实验脚本：乘法门控 (Multiplicative Gating)
#
# 设计目标:
# 1. 把 CycleState 改为"乘法门控"形态:
#    - 沿用 DE-3 的 init 拼接 (观测期最后时刻的 [queue_last, cycle_last]
#      拼接到 decoder 初始化向量)
#    - 叠加一个新机制: ``pred_lstm_model`` 每步更新 ``pred_lstm_hidden``
#      后, 用一个 2 层 MLP + sigmoid 计算逐元素门控
#      ``gate = ar2_hidden_gate([pred_lstm_hidden, state_context])``,
#      然后 ``pred_lstm_hidden = pred_lstm_hidden * gate``。
#    - 这是与 AR-1 (加性拼接) 不同的耦合方式: AR-2 用 state context
#      通过 sigmoid 门控"调制" pred_lstm_hidden, 让 state context 决定
#      哪些隐状态维度被放大/抑制。
#    - 隐含 ``minimal_viable_mode=True`` (init 拼接), 即
#      AR-2 = DE-3 + per-step multiplicative gate。
#    - 强制关闭 state_gating / queue_rollout / lane_queue_anchor /
#      decoder_state_residual / aux_losses 五个开关
#    - 训练时不再消费 queue_lstm_model / cycle_lstm_model 的 hidden
# 2. 训练完成后, 用 val + num_samples=20 与 test + num_samples=20 双口径评估
# 3. AR-2 与 ``--oracle_inject_mode`` / ``--ar1_direct_inject_mode`` 互斥,
#    互斥校验在模型内部完成
#
# 用法:
#   DE2_DEVICE=cuda ./scripts/run_ar2.sh
# 或逐步:
#   ./scripts/run_ar2.sh train
#   ./scripts/run_ar2.sh eval_val
#   ./scripts/run_ar2.sh eval_test
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-$ROOT_DIR/experiments/cyclestate/AR2_multiplicative_gating}"
DEVICE="${AR2_DEVICE:-cuda}"
DATASET_NAME="${AR2_DATASET:-VTP_C}"
BATCH_SIZE="${AR2_BATCH_SIZE:-64}"
VAL_BATCH_SIZE="${AR2_VAL_BATCH_SIZE:-8}"
NUM_VAL_SAMPLES="${AR2_NUM_VAL_SAMPLES:-20}"
NUM_EVAL_SAMPLES="${AR2_NUM_EVAL_SAMPLES:-20}"
VAL_DSET_TYPE="${AR2_VAL_DSET_TYPE:-val}"
TRAIN_STAGE="${AR2_TRAIN_STAGE:-warmup}"
SEED="${AR2_SEED:-73}"

# 训练阶段: 50 warmup batches + 50 refine batches — 与 AR-1 / DE-1 对齐
WARMUP_EPOCHS="${AR2_WARMUP_EPOCHS:-50}"
REFINE_EPOCHS="${AR2_REFINE_EPOCHS:-50}"
# 50b 协议: 与 ``warmup50_refine50_p0_seqgat_relation_v1`` 对齐——
# 每个 epoch 最多跑 50 个 batch (用 ``max_train_batches`` 控制)。
MAX_TRAIN_BATCHES="${AR2_MAX_TRAIN_BATCHES:-50}"
MAX_VAL_BATCHES="${AR2_MAX_VAL_BATCHES:-20}"

mkdir -p "$LOG_DIR"

TRAIN_COMMON_ARGS=(
  --model_type cyclestate
  --ar2_multiplicative_gating_mode
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
  --ar2_multiplicative_gating_mode
  --device "$DEVICE"
  --dataset_name "$DATASET_NAME"
  --batch_size "$VAL_BATCH_SIZE"
  --num_samples "$NUM_EVAL_SAMPLES"
  --loader_num_workers 0
  --pin_memory
)

case "${1:-all}" in
  train)
    echo "[AR-2] training $WARMUP_EPOCHS warmup batches (stage=$TRAIN_STAGE, max_train_batches=$MAX_TRAIN_BATCHES) -> $LOG_DIR"
    # num_epochs=2 是为了让 ``range(0, num_epochs+1) == [0, 1]`` 真的
    # 跑 2 个 epoch (每个 epoch 内由 max_train_batches 截断), 与
    # ``warmup50_refine50_p0_seqgat_relation_v1`` 的实际行为对齐。
    python D2TP/train.py \
      "${TRAIN_COMMON_ARGS[@]}" \
      --num_epochs 2
    ;;

  train_refine)
    REFINED_CKPT="$LOG_DIR/checkpoint/model_best.pth.tar"
    if [[ ! -f "$REFINED_CKPT" ]]; then
      echo "[AR-2] expected warmup checkpoint missing: $REFINED_CKPT" >&2
      exit 1
    fi
    echo "[AR-2] refining $REFINE_EPOCHS refine batches from $REFINED_CKPT -> $LOG_DIR"
    # 用 num_epochs=3 是因为 checkpoint 会把 ``start_epoch`` 恢复到自身
    # 的 ``epoch`` 字段 (warmup 收尾于 epoch=2), 主循环
    # ``range(start_epoch, num_epochs+1)`` 必须留出至少 1 个 slot 才会
    # 真的进入训练; 留 3 是给后续可能的 cont 留余地。
    python D2TP/train.py \
      "${TRAIN_COMMON_ARGS[@]}" \
      --train_stage refine \
      --resume "$REFINED_CKPT" \
      --num_epochs 3
    ;;

  eval_val)
    CKPT="${2:-$LOG_DIR/checkpoint/model_best.pth.tar}"
    echo "[AR-2] evaluating val + num_samples=$NUM_EVAL_SAMPLES from $CKPT"
    python D2TP/evaluate_model.py \
      "${EVAL_COMMON_ARGS[@]}" \
      --dset_type val \
      --eval_print_every 10 \
      --resume "$CKPT"
    ;;

  eval_test)
    CKPT="${2:-$LOG_DIR/checkpoint/model_best.pth.tar}"
    echo "[AR-2] evaluating test + num_samples=$NUM_EVAL_SAMPLES from $CKPT"
    python D2TP/evaluate_model.py \
      "${EVAL_COMMON_ARGS[@]}" \
      --dset_type test \
      --eval_print_every 10 \
      --resume "$CKPT"
    ;;

  smoke)
    echo "[AR-2] smoke: 1-batch train + 1-batch val (no real training)"
    python D2TP/train.py \
      "${TRAIN_COMMON_ARGS[@]}" \
      --num_epochs 0 \
      --max_train_batches 1 \
      --max_val_batches 1 \
      --print_every 1
    CKPT="$LOG_DIR/checkpoint/model_best.pth.tar"
    if [[ ! -f "$CKPT" ]]; then
      echo "[AR-2] smoke produced no checkpoint; skipping eval"
      exit 1
    fi
    echo "[AR-2] smoke eval on val/test (1 batch each, num_samples=2)"
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
    echo "[AR-2] smoke OK"
    ;;

  smoke_unit)
    echo "[AR-2] smoke_unit: run ar2_multiplicative_gating_smoke_test.py directly (no training)"
    python D2TP/ar2_multiplicative_gating_smoke_test.py
    ;;

  all|*)
    "$0" smoke_unit
    "$0" smoke
    "$0" train
    "$0" train_refine
    "$0" eval_val
    "$0" eval_test
    ;;

esac
