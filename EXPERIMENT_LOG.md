# CycleState 实验日志

> **更新时间**: 2026-06-10 12:00
> **覆盖范围**: 当前 protocol-check 与 comparable 结论
> **问题索引**: [docs/ENGINEERING_ISSUES.md](./docs/ENGINEERING_ISSUES.md) ·
> [docs/COMPREHENSIVE_ANALYSIS.md](./docs/COMPREHENSIVE_ANALYSIS.md) ·
> [docs/METHOD_AND_ARCHITECTURE_ANALYSIS.md](./docs/METHOD_AND_ARCHITECTURE_ANALYSIS.md)
> **说明**: 本文件只保留当前证据、决定性里程碑和推荐下一步；更早的长时间线默认回溯 git 历史。

## 1. 协议标签

| 标签 | 用途 | 常见配置 |
|------|------|----------|
| `smoke` | 验证 forward/backward、日志、checkpoint、开关 | `max_train_batches=1` 或 `num_epochs=0` |
| `protocol-check` | 验证训练协议、恢复逻辑、采样口径 | 小 batch、短流程 |
| `comparable` | 正式比较 baseline 与候选 | split、checkpoint、采样次数、评估脚本全部对齐 |

正式对比必须记录这些字段：

- `split`
- `num_samples`
- `checkpoint / resume 路径`
- `teacher_forcing_ratio`
- `rollout_residual_scale`
- `aux_rollout_weight`
- `detach_rollout_state`
- `disable_*` 开关

## 2. 当前可比证据

### 2.1 Baseline

`D2TP/model_best.pth.tar`，`num_samples=4`

| Split | ADE | FDE |
|-------|-----|-----|
| val | 38.493 | 78.706 |
| test | 17.812 | 37.568 |

关键边界：

- `val` 与 `test` 有明显落差。
- `num_samples=20` 的 baseline 审计仍未补齐，现阶段不能把 `4-sample` 结果外推成论文口径。

### 2.2 当前最佳 CycleState 候选

Run: `experiments/cyclestate/warmup50_refine50_p0_seqgat_relation_v1`

配置摘要：

- `50b warmup -> 50b refine`
- `teacher_forcing_ratio=0.6`
- `aux_rollout_weight=2.5`
- `rollout_residual_scale=0.7`

| Split | num_samples | ADE | FDE | 备注 |
|-------|-------------|-----|-----|------|
| val | 20 | 75.078 | 154.690 | 当前最佳 true-val comparable |
| val | 4 | 84.772 | 170.878 | quick 复核口径 |

当前结论：

- 这是仓库内最强的 CycleState 可比证据。
- 它优于纯 warmup `50b` 候选，但仍明显落后于 baseline。
- 当前最重要的问题不是“能不能再加模块”，而是“这条协议能否在 `test + num_samples=20` 上站住”。

## 3. 决定性里程碑

1. **Stage 21-22: rollout 路径一致性修复**
   - 修复 step-0 offset 与 queue context 锚定后，短协议里 rollout-on 重新优于 no-rollout。

2. **Stage 24: 协议硬化落地**
   - true-val 选模、`lr` 生效、`grad_clip`、`detach_rollout_state`、状态稳定性日志均已落地。

3. **Stage 25: P0 基础正确性修复**
   - `seqGAT` 梯度恢复。
   - `relation_Matrix` 正常方向区间逻辑修复。
   - `100b` 崩坏从 `165.7%` 缓和到 `130.1%`，但未根治。

4. **Stage 26: 最小变量结论明确**
   - 单独降低 warmup `teacher_forcing_ratio` 到 `0.6` 无效。
   - `50b warmup -> 50b refine` 是当前最佳协议候选。

5. **Stage 44: 消融协议收口**
   - `AblationConfig` 集中管理 `disable_*` 开关。
   - train / evaluate / protocol-log 共用一套配置语义，减少实验口径漂移。

## 4. 推荐下一步

固定顺序如下：

1. baseline `val + num_samples=20`
2. baseline `test + num_samples=20`
3. 当前最佳候选 `test + num_samples=20`

如果第 3 步成立，再按下面顺序做单变量消融：

1. `disable_queue_rollout`
2. `disable_decoder_state_residual`
3. `disable_lane_queue_anchor`
4. `disable_state_gating`

如果第 3 步不成立，优先回到协议正确性与状态进入预测期的稳定性，不要继续试：

- 单独 warmup `teacher_forcing_ratio`
- 同一轮混改多个大变量
- 新增通用 Transformer / Diffusion / Scene Encoder

## 5. 结论边界

- 当前还**不能**宣称 CycleState 超过 baseline。
- 任何结论都必须显式区分：
  - `val` vs `test`
  - `num_samples=4` vs `20`
  - warmup-only vs `50b warmup -> refine`
- oracle 信号假设已经文档化，但其性能贡献仍需通过 [docs/PLAN.md](./docs/PLAN.md)
  中的 Phase 0.5 退化实验量化。
