#!/usr/bin/env bash
# DE-1 决定性实验脚本：Oracle State 直注
#
# 设计目标:
# 1. 把 CycleState 改为"oracle 交通状态直注"形态:
#    - 强制关闭 state_gating / queue_rollout / lane_queue_anchor /
#      decoder_state_residual / aux_losses 五个开关
#    - 训练时不再消费 queue_lstm_model / cycle_lstm_model 的 hidden
#    - 替换 ``pred_lstm_model`` 为新版本:输入维度由
#      ``traj_lstm_input_size`` 扩大到 ``traj_lstm_input_size + 10``,
#      oracle 特征 (phase one-hot / elapsed / remaining / phase_change /
#      distance / direction / speed, 10 维) 直接拼接到每一步的 LSTM
#      输入后面,等价于"把真实交通状态(不做任何学习)塞进 decoder 内部信号"
# 2. 训练完成后, 用 val + num_samples=20 与 test + num_samples=20 双口径评估
#
# 用法:
#   DE1_DEVICE=cuda DE1_NUM_EPOCHS=100 ./scripts/run_de1.sh
# 或逐步:
#   ./scripts/run_de1.sh train
#   ./scripts/run_de1.sh eval_val
#   ./scripts/run_de1.sh eval_test
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-$ROOT_DIR/experiments/cyclestate/DE1_oracle_inject}"
DEVICE="${DE1_DEVICE:-cuda}"
DATASET_NAME="${DE1_DATASET:-VTP_C}"
NUM_EPOCHS="${DE1_NUM_EPOCHS:-100}"
BATCH_SIZE="${DE1_BATCH_SIZE:-64}"
VAL_BATCH_SIZE="${DE1_VAL_BATCH_SIZE:-8}"
NUM_VAL_SAMPLES="${DE1_NUM_VAL_SAMPLES:-20}"
NUM_EVAL_SAMPLES="${DE1_NUM_EVAL_SAMPLES:-20}"
VAL_DSET_TYPE="${DE1_VAL_DSET_TYPE:-val}"
TRAIN_STAGE="${DE1_TRAIN_STAGE:-warmup}"
SEED="${DE1_SEED:-73}"

# 训练阶段: 50 warmup batches + 50 refine batches
WARMUP_EPOCHS="${DE1_WARMUP_EPOCHS:-50}"
REFINE_EPOCHS="${DE1_REFINE_EPOCHS:-50}"
# 50b 协议：与 ``warmup50_refine50_p0_seqgat_relation_v1`` 对齐——
# 每个 epoch 最多跑 50 个 batch（用 ``max_train_batches`` 控制）。
MAX_TRAIN_BATCHES="${DE1_MAX_TRAIN_BATCHES:-50}"
MAX_VAL_BATCHES="${DE1_MAX_VAL_BATCHES:-20}"

mkdir -p "$LOG_DIR"

TRAIN_COMMON_ARGS=(
  --model_type cyclestate
  --oracle_inject_mode
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
  --oracle_inject_mode
  --device "$DEVICE"
  --dataset_name "$DATASET_NAME"
  --batch_size "$VAL_BATCH_SIZE"
  --num_samples "$NUM_EVAL_SAMPLES"
  --loader_num_workers 0
  --pin_memory
)

case "${1:-all}" in
  train)
    echo "[DE-1] training $WARMUP_EPOCHS warmup batches (stage=$TRAIN_STAGE, max_train_batches=$MAX_TRAIN_BATCHES) -> $LOG_DIR"
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
      echo "[DE-1] expected warmup checkpoint missing: $REFINED_CKPT" >&2
      exit 1
    fi
    echo "[DE-1] refining $REFINE_EPOCHS refine batches from $REFINED_CKPT -> $LOG_DIR"
    # 用 num_epochs=3 是因为 checkpoint 会把 ``start_epoch`` 恢复到自身
    # 的 ``epoch`` 字段（DE-1 warmup 收尾于 epoch=2），主循环
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
    echo "[DE-1] evaluating val + num_samples=$NUM_EVAL_SAMPLES from $CKPT"
    python D2TP/evaluate_model.py \
      "${EVAL_COMMON_ARGS[@]}" \
      --dset_type val \
      --eval_print_every 10 \
      --resume "$CKPT"
    ;;

  eval_test)
    CKPT="${2:-$LOG_DIR/checkpoint/model_best.pth.tar}"
    echo "[DE-1] evaluating test + num_samples=$NUM_EVAL_SAMPLES from $CKPT"
    python D2TP/evaluate_model.py \
      "${EVAL_COMMON_ARGS[@]}" \
      --dset_type test \
      --eval_print_every 10 \
      --resume "$CKPT"
    ;;

  smoke)
    echo "[DE-1] smoke: 1-batch train + 1-batch val (no real training)"
    python D2TP/train.py \
      "${TRAIN_COMMON_ARGS[@]}" \
      --num_epochs 0 \
      --max_train_batches 1 \
      --max_val_batches 1 \
      --print_every 1
    CKPT="$LOG_DIR/checkpoint/model_best.pth.tar"
    if [[ ! -f "$CKPT" ]]; then
      echo "[DE-1] smoke produced no checkpoint; skipping eval"
      exit 1
    fi
    echo "[DE-1] smoke eval on val/test (1 batch each, num_samples=2)"
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
    echo "[DE-1] smoke OK"
    ;;

  smoke_unit)
    echo "[DE-1] smoke_unit: run de1_oracle_smoke_test.py directly (no training)"
    python D2TP/de1_oracle_smoke_test.py
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
