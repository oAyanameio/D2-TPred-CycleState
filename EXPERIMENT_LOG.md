# CycleState 实验日志

> **更新时间**: 2026-06-10 18:10
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

`D2TP/model_best.pth.tar`

| Split | num_samples | ADE | FDE |
|-------|-----|-----|
| val | 20 | 35.022 | 70.658 |
| test | 20 | 15.359 | 31.514 |
| val | 4 | 38.493 | 78.706 |
| test | 4 | 17.812 | 37.568 |

关键边界：

- `val` 与 `test` 仍有明显落差。
- baseline 的 `num_samples=20` 口径现已补齐，可作为后续所有 comparable 的硬基线。

### 2.2 当前最佳 CycleState 候选

Run: `experiments/cyclestate/warmup50_refine50_p0_seqgat_relation_v1`

配置摘要：

- `50b warmup -> 50b refine`
- `teacher_forcing_ratio=0.6`
- `aux_rollout_weight=2.5`
- `rollout_residual_scale=0.7`

| Split | num_samples | ADE | FDE | 备注 |
|-------|-------------|-----|-----|------|
| val | 20 | 74.947 | 154.411 | 接入 prediction-time cycle rollout 后复核 |
| test | 20 | 43.736 | 85.691 | 接入 prediction-time cycle rollout 后复核 |
| val | 4 | 84.772 | 170.878 | quick 复核口径 |

当前结论：

- `prediction-time cycle rollout` 直接接入旧 checkpoint 后，`val + 20` 有轻微改善
  (`75.078 -> 74.947`, `154.690 -> 154.411`)。
- 但 `test + 20` 仍显著落后于 baseline (`43.736 / 85.691` vs `15.359 / 31.514`)。
- 这说明当前信号更像“方向可能对，但旧参数没有适配新 macro rollout”，而不是
  “只靠推理侧改动就能翻盘”。

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

6. **Stage 45: prediction-time cycle rollout 最小方案落地**
   - 在 `CycleStateTrajectoryGenerator` 的 decoder loop 内补上 `cycle hidden/cell`
     的逐步 rollout。
   - decoder 的 macro 条件从“单步 `cycle_step_embedding`”切换为“rollout 后的
     cycle hidden”，使 `macro memory` 与 `meso queue rollout` 一样具备跨步状态。
   - 已补定向测试，证明：
     - `cycle hidden` 会在预测期跨步变化；
     - 下一步 rollout 以上一步 `cycle hidden` 为输入；
     - decoder state residual 确实消费 rollout 后的 cycle context。
   - 目前这仍是 `protocol-check` 级证据，不代表 comparable 指标已经提升。

7. **Stage 46: comparable 口径补齐与旧 checkpoint 复核**
   - baseline `val/test + num_samples=20` 已补齐：
     - `val`: `35.022 / 70.658`
     - `test`: `15.359 / 31.514`
   - 为了让旧 `CycleState` checkpoint 能在新代码下评估，补了 legacy
     `queue_aux_head / cycle_aux_head` 到新拆头结构的兼容加载。
   - 在此基础上，`warmup50_refine50_p0_seqgat_relation_v1` 复核得到：
     - `val + 20`: `74.947 / 154.411`
     - `test + 20`: `43.736 / 85.691`
   - 当前判断：需要最小增量训练验证，而不是继续只做推理侧改动。

## 4. 推荐下一步

固定顺序如下：

1. 基于 `Stage 45` 的代码改动做最小增量训练/微调
2. 先复核 `val + num_samples=20`
3. 只有 `val` 出现稳定正向信号时再补 `test + num_samples=20`

如果最小训练后仍没有明确正向信号，再优先按下面顺序做诊断式单变量消融：

1. `disable_queue_rollout`
2. `disable_decoder_state_residual`
3. `disable_state_gating`
4. `disable_lane_queue_anchor`

如果最小训练后仍不成立，优先回到预测期状态路径与协议稳定性，不要继续试：

- 单独 warmup `teacher_forcing_ratio`
- 同一轮混改多个大变量
- 新增通用 Transformer / Diffusion / Scene Encoder

在这条主线下，G6 现在的默认顺序变为：

1. 先保留已落地的 `cycle hidden rollout` 最小方案
2. 先跑 comparable / 诊断式消融确认它是否提供净收益
3. 只有在它表现出正向信号时，才考虑更重的显式 cycle feature rollout

## 5. 结论边界

- 当前还**不能**宣称 CycleState 超过 baseline。
- 任何结论都必须显式区分：
  - `val` vs `test`
  - `num_samples=4` vs `20`
  - warmup-only vs `50b warmup -> refine`
- oracle 信号假设已经文档化，但其性能贡献仍需通过 [docs/PLAN.md](./docs/PLAN.md)
  中的 Phase 0.5 退化实验量化。
- `Stage 45` 当前只证明“机制已进入代码并被 decoder 使用”，还没有证明
  “它已让 CycleState 超过 baseline”。
- `Stage 46` 进一步证明：即便补齐了 comparable 口径，当前 candidate 仍远未达到
  baseline，因此后续所有叙事都必须围绕“为什么新 macro rollout 还没被参数学会”
  来组织，而不是提前声称有效。
