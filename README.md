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
- `docs/AI_EXPERIMENT_DELEGATION_GUIDE.md`：委托其他 AI 执行修改/实验/验证时的统一作战手册。

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
- Stage 24 true-val 复核显示默认稳定化仍未通过：
  - `warmup_protocol_stable_v1_50b`: `ADE 87.082 / FDE 175.723`
  - `warmup_protocol_stable_v1_100b`: `ADE 231.420 / FDE 420.862`
  - `100b` 相对 `50b` 恶化 `165.7%`，超过 15% 稳定性门槛
- 单变量降低学习率到 `3e-4` 后，`batch 50` 状态范数更温和，但 `batch 100` 仍崩：
  - `88.598 / 171.890 -> 226.302 / 411.163`
- 对当前代码进一步审查后，已确认两个更基础的高优先级问题：
  - `seqGAT` 前向被 `torch.no_grad()` 包裹，局部时序图注意力实际上不参与训练
  - `relation_Matrix` 的正常方向区间分支会把距离内邻居无条件连边，削弱了方向约束
- Stage 25 P0 已完成并通过测试：
  - 已恢复 baseline/CycleState 中 `seqGAT` 的梯度流
  - 已修复 `relation_Matrix` 正常方向区间的无条件连边
  - 当前单元测试增至 `39` 项并全部通过
- Stage 25 true-val 复测结果：
  - `warmup_p0_seqgat_relation_v1_50b`: `ADE 88.956 / FDE 175.380`
  - `warmup_p0_seqgat_relation_v1_100b`: `ADE 204.730 / FDE 375.054`
  - `100b` 相对 `50b` 仍恶化 `130.1%`，未通过 15% 稳定性门槛
  - 但相较 Stage 24 的 `231.420 / 420.862`，P0 修复已把 `100b` 崩坏幅度压低
- Stage 26 最小变量实验结果：
  - 单独降低 warmup `teacher_forcing_ratio` 到 `0.6` 并不能解决问题：
    - `warmup_p0_seqgat_relation_tf06_100b`
    - `92.735 / 182.898 -> 266.671 / 468.079`
    - `100b` 相对 `50b` 恶化 `187.6%`
  - 但 `50b warmup -> 50b refine` 的阶段切换候选显著更稳：
    - `warmup50_refine50_p0_seqgat_relation_v1`
    - quick `val + num_samples=4`: `ADE 84.772 / FDE 170.878`
    - full `val + num_samples=20`: `ADE 75.078 / FDE 154.690`
  - 这说明“继续压 warmup teacher forcing”不是主因修复方向，更有价值的是在 `50b` 左右把长链状态学习交给 `refine` 接管。

目前还不能宣称稳定超过 baseline 或论文指标。到当前为止，最合理的科研叙事是：基础交互建模 bug 会放大崩坏，但 warmup 后半程失稳主要更像“训练阶段职责划分错误”，而不只是 exposure-bias；因此下一步主线应围绕 `50b warmup -> refine` 候选做正式复核和消融，而不是继续在 warmup 中堆新技巧。

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

### 当前最佳协议候选：`50b warmup -> 50b refine`

```bash
python D2TP/train.py \
  --log_dir experiments/cyclestate/warmup50_refine50_p0_seqgat_relation_v1 \
  --model_type cyclestate \
  --train_stage refine \
  --device cuda \
  --pin_memory \
  --loader_num_workers 0 \
  --batch_size 8 \
  --best_k 4 \
  --num_val_samples 4 \
  --aux_rollout_weight 2.5 \
  --resume experiments/cyclestate/warmup_p0_seqgat_relation_v1_50b/checkpoint/model_best.pth.tar \
  --num_epochs 0 \
  --print_every 50 \
  --max_train_batches 50 \
  --max_val_batches 20 \
  --val_dset_type val
```

完整 `val + num_samples=20` 复核：

```bash
python D2TP/evaluate_model.py \
  --model_type cyclestate \
  --device cuda \
  --pin_memory \
  --loader_num_workers 0 \
  --batch_size 8 \
  --num_samples 20 \
  --eval_print_every 10 \
  --resume experiments/cyclestate/warmup50_refine50_p0_seqgat_relation_v1/checkpoint/model_best.pth.tar \
  --dset_type val \
  --rollout_residual_scale 0.7
```

## 下一轮实验顺序

1. 先补齐 baseline 的正式可比线：
   - `baseline_audit_v2_val_full_num_samples20`
   - `baseline_audit_v2_test_full_num_samples20`
2. 再对当前最佳候选做正式复核：
   - `warmup50_refine50_p0_seqgat_relation_v1_test20`
   - 目标是确认 `50b warmup -> 50b refine` 在 `test + num_samples=20` 上是否仍优于纯 warmup 候选
3. 若 `test@20` 结果仍站得住，再围绕该候选做消融：
   - `disable_queue_rollout`
   - `disable_decoder_state_residual`
   - `disable_lane_queue_anchor`
   - `disable_state_gating`
4. 若 `test@20` 明显退化，再回到协议层而不是加结构：
   - 优先检查 `rollout offset` 一致性、`phase_change` 预测期缺失、`D_step` 重置
   - 暂不继续尝试 warmup `teacher_forcing_ratio` 单变量，因为 `tf=0.6` 已被证伪

### 下一轮将直接执行的命令

1. baseline `val + num_samples=20`

```bash
python D2TP/evaluate_model.py \
  --model_type d2tpred \
  --device cuda \
  --pin_memory \
  --loader_num_workers 0 \
  --batch_size 16 \
  --num_samples 20 \
  --eval_print_every 10 \
  --resume D2TP/model_best.pth.tar \
  --dset_type val
```

2. baseline `test + num_samples=20`

```bash
python D2TP/evaluate_model.py \
  --model_type d2tpred \
  --device cuda \
  --pin_memory \
  --loader_num_workers 0 \
  --batch_size 16 \
  --num_samples 20 \
  --eval_print_every 10 \
  --resume D2TP/model_best.pth.tar \
  --dset_type test
```

3. current best candidate `test + num_samples=20`

```bash
python D2TP/evaluate_model.py \
  --model_type cyclestate \
  --device cuda \
  --pin_memory \
  --loader_num_workers 0 \
  --batch_size 8 \
  --num_samples 20 \
  --eval_print_every 10 \
  --resume experiments/cyclestate/warmup50_refine50_p0_seqgat_relation_v1/checkpoint/model_best.pth.tar \
  --dset_type test \
  --rollout_residual_scale 0.7
```

### 进入下一步的判定标准

- 若 `warmup50_refine50_p0_seqgat_relation_v1_test20` 相比纯 warmup 候选仍保持优势，则继续做消融，不改结构。
- 若该候选在 `test@20` 上明显失去优势，则下一轮优先修正协议正确性项，而不是继续试新的训练 trick。

### 委托其他 AI 的执行规范

若后续把实验委托给其他 AI 执行，统一按：

- [docs/AI_EXPERIMENT_DELEGATION_GUIDE.md](./docs/AI_EXPERIMENT_DELEGATION_GUIDE.md)

它定义了：

- 允许做什么 / 禁止做什么
- 固定实验顺序
- 必跑验证命令
- 交付模板
- 必须停下来交给主审的条件

## 优化日志摘要

完整日志见 `EXPERIMENT_LOG.md`。当前累计优化包括：

- Stage 0-4：baseline 兼容、实验仓库克隆、CycleState 原型、quick smoke、warm-start。
- Stage 5-14：generator-only、structured auxiliary losses、phase-conditioned gating、traffic_context adapter、staged protocol、实验日志规范。
- Stage 15-19：Phase-Rolling Queue Memory、queue rollout 消融、Lane-Consensus Meso Anchor、predictive anchor trace、baseline-compatible decoder residual。
- Stage 20-23：训练/评估指标口径对齐、验证调度修复、rollout 路径根因修复、matched warmup stability、`aux_rollout_weight` 独立调节。
- Stage 24：协议优先稳定化，包含 true-val 选模、`lr` 生效、`grad_clip`、bounded rollout residual injection、warmup rollout-state detach 和状态稳定性日志。
- Stage 25：修复 `seqGAT` 梯度冻结与 `relation_Matrix` 方向约束 bug，并完成 true-val 复测；结果显示 `100b` 崩坏有所缓和，但仍未通过稳定性门槛。
- Stage 26：最小变量稳定化复测；确认单独降低 warmup teacher forcing 无效，但 `50b warmup -> 50b refine` 在 quick `val` 与 full `val@20` 上均优于纯 warmup 候选。
