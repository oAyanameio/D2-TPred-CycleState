# D2-TPred-CycleState

本仓库同时承担两个角色：

1. 复现与审计原始 `D2-TPred` baseline。
2. 开发新研究方向 `CycleState: Full-Cycle Traffic-State Memory for Trajectory Prediction at Signalized Intersections`。

当前开发主线不是给 baseline 简单叠加一个通用时序模块，而是围绕一个更明确的科研判断：

> 信号灯路口轨迹预测应被建模为 `full-cycle traffic-state memory` 问题。

也就是说，未来轨迹不仅由个体短时运动和局部交互决定，还受到车道级排队/释放波与信号周期状态的共同约束。

## 科研故事

`CycleState` 把信号灯路口预测拆成三层状态：

- `micro`：继承原始 `D2-TPred` 的轨迹 LSTM、空间图交互和局部时间交互，保住强运动预测基线。
- `meso`：显式构造车道级 `queue-state memory`，建模排队密度、等待比例、释放比例、队首状态和 stop-line occupancy。
- `macro`：显式构造 `cycle-state memory`，建模 phase one-hot、elapsed/remaining phase time 和 phase-change。

当前最有辨识度的机制是 `Phase-Rolling Queue Memory`：

- queue-state 不在 decoder 初始化后冻结；
- 预测期每一步都会根据 phase progression、上一时刻预测运动和 lane-level anchor 滚动更新；
- decoder 使用 `observed queue context + bounded rollout residual`，避免早期 warmup 中 rollout hidden 直接接管主解码器。

这条叙事的边界也很明确：本仓库当前不引入通用 Transformer/扩散式重构，不照搬现有 intersection forecasting 方法的完整套路，而是把交通工程中的 queue/cycle 状态显式纳入轨迹预测。

## 当前实现

核心文件：

- `D2TP/models.py`：原始 `TrajectoryGenerator` 与新增 `CycleStateTrajectoryGenerator`。
- `D2TP/train.py`：训练协议、structured auxiliary losses、训练内验证和 checkpoint 保存。
- `D2TP/evaluate_model.py`：离线 `val/test` 评估，支持 `d2tpred` 与 `cyclestate`。
- `tests/test_cyclestate_protocol.py`：协议、消融开关、rollout 路径和评估聚合的回归测试。
- `EXPERIMENT_LOG.md`：完整优化日志和实验记录。

当前 `CycleState` 已支持：

- `tuple -> traffic_context` adapter，为后续 INT2 迁移保留统一接口；
- `warmup / refine / adversarial` 分阶段训练；
- structured queue/cycle auxiliary losses；
- `queue rollout / lane anchor / state gating / decoder residual` 可控消融；
- `aux_rollout_weight` 独立调节；
- `val` split 训练内选模，`test` split 最终复核；
- `lr` 命令行参数真实生效；
- `grad_clip`、`rollout_residual_scale`、`detach_rollout_state` 稳定化协议；
- 状态稳定性日志：`DInitNorm / DStepNorm / QRollHNorm / PredOffsetNorm / GradNorm`。

## 评测口径

所有实验结果必须按下面三类记录：

- `smoke`：只验证 forward/backward、日志、checkpoint、消融开关和损失项是否工作。常见设置为 `max_train_batches=1` 或 `num_epochs=0`。
- `protocol-check`：验证训练协议、验证调度、采样逻辑、checkpoint 恢复和统计口径是否一致。不能直接当作论文结论。
- `comparable`：只有 split、checkpoint 来源、ADE/FDE 聚合方式、采样次数和评估脚本口径都对齐后，才能用于和 baseline 或论文指标比较。

当前训练内默认使用 `--val_dset_type val` 做模型选择；`test` split 只用于最终复核或兼容历史协议。

## 当前证据

仓库内已经确认的 baseline 可比线：

- `D2TP/model_best.pth.tar`，完整 `val` split，`num_samples=4`：
  - `ADE 38.493 / FDE 78.706`
- `D2TP/model_best.pth.tar`，完整 `test` split，`num_samples=4`：
  - `ADE 17.812 / FDE 37.568`

当前 `CycleState` 的关键中间证据：

- 旧调度下 `warmup_main_v2` 短程下降到 `ADE 56.827 / FDE 107.416`，只作为“模型在学”的证据。
- 修复前短协议中 rollout-on 曾输给 no-rollout：`78.227 / 152.544` vs `71.863 / 140.974`。
- 修复 training step-0 rollout alignment 与 anchored rollout decode context 后，rollout-on 恢复到 `66.793 / 132.168`，重新优于对应 no-rollout。
- `aux_rollout_weight=2.5` 的 50-batch rollout-on 达到 `66.761 / 122.728`，优于 `no_rollout@50b` 的 `67.747 / 124.741`。
- 100-batch matched warmup 曾出现后半程整体崩坏，因此当前优先级是协议稳定化，而不是继续堆新结构。

目前还不能宣称稳定超过 baseline 或论文指标。下一步必须先在 true-val 协议下证明 100/200 batch 稳定，再推进 refine 和 test 复核。

## 复现入口

### 单元测试

```bash
python -m unittest discover -s tests -p 'test_cyclestate_protocol.py'
```

### Baseline 审计

```bash
python D2TP/evaluate_model.py \
  --model_type d2tpred \
  --device cuda \
  --loader_num_workers 0 \
  --batch_size 16 \
  --num_samples 4 \
  --eval_print_every 10 \
  --resume D2TP/model_best.pth.tar \
  --dset_type val
```

最终 `test` 复核只需把 `--dset_type val` 改为 `--dset_type test`。更接近论文口径时使用 `--num_samples 20`。

### CycleState 协议优先 warmup

```bash
python D2TP/train.py \
  --log_dir experiments/cyclestate/warmup_protocol_stable_v1 \
  --model_type cyclestate \
  --train_stage warmup \
  --device cuda \
  --pin_memory \
  --loader_num_workers 0 \
  --batch_size 8 \
  --best_k 4 \
  --num_val_samples 4 \
  --aux_rollout_weight 2.5 \
  --resume D2TP/model_best.pth.tar \
  --num_epochs 0 \
  --print_every 50 \
  --max_train_batches 100 \
  --max_val_batches 20 \
  --val_dset_type val
```

默认 warmup 稳定化参数：

- `--grad_clip 1.0`
- `--rollout_residual_scale 0.35`
- `--detach_rollout_state`

显式关闭 rollout-state detach：

```bash
--no_detach_rollout_state
```

## 下一轮实验顺序

1. 补齐 baseline：`val/test` 各跑 `num_samples=4` 与 `20`。
2. 稳定性梯度实验：
   - 当前参考：`rollfix + aux_rollout_weight=2.5`
   - 协议硬化：`true-val + lr 生效 + grad_clip`
   - 状态注入稳定化：`rollout_residual_scale + detach_rollout_state`
3. 每组按 `50 -> 100 -> 200 batch` 递进；快速筛选用 `val + num_samples=4`，入围后完整 `val + num_samples=20`。
4. 稳定性门槛：
   - 同配置 `100-batch` 的 `val ADE` 不得比 `50-batch` 恶化超过 `15%`；
   - rollout-on 必须优于匹配的 no-rollout；
   - 通过后才进入 `refine`。
5. 消融顺序：
   - rollout on/off
   - decoder residual on/off
   - lane anchor on/off
   - state gating on/off
6. 只推进 1 个长程候选进入 `refine`，最后在 `test` split 上复核。

## 优化日志摘要

完整日志见 `EXPERIMENT_LOG.md`。当前累计优化包括：

- Stage 0-4：baseline 兼容、实验仓库克隆、CycleState 原型、quick smoke、warm-start。
- Stage 5-14：generator-only、structured auxiliary losses、phase-conditioned gating、traffic_context adapter、staged protocol、实验日志规范。
- Stage 15-19：Phase-Rolling Queue Memory、queue rollout 消融、Lane-Consensus Meso Anchor、predictive anchor trace、baseline-compatible decoder residual。
- Stage 20-23：训练/评估指标口径对齐、验证调度修复、rollout 路径根因修复、matched warmup stability、`aux_rollout_weight` 独立调节。
- Stage 24：协议优先稳定化，包含 true-val 选模、`lr` 生效、`grad_clip`、bounded rollout residual injection、warmup rollout-state detach 和状态稳定性日志。
