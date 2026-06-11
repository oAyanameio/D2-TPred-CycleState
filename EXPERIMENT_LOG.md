# CycleState 实验日志

> **更新时间**: 2026-06-11 14:10
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
| val | 20 | 51.607 | 103.955 | `..._cycle_rollout_refine1/model_best` full-val 复核 |
| test | 20 | 34.911 | 69.133 | `..._cycle_rollout_refine1/model_best` full-test 复核 |
| val | 4 | 84.772 | 170.878 | quick 复核口径 |

当前结论：

- `prediction-time cycle rollout` 直接接入旧 checkpoint 后，`val + 20` 有轻微改善
  (`75.078 -> 74.947`, `154.690 -> 154.411`)。
- 但 `test + 20` 仍显著落后于 baseline (`43.736 / 85.691` vs `15.359 / 31.514`)。
- 这说明当前信号更像“方向可能对，但旧参数没有适配新 macro rollout”，而不是
  “只靠推理侧改动就能翻盘”。
- 在旧 best 上做 1 轮短程 refine（`max_train_batches=50`、`max_val_batches=20`，
  `aux_rollout_weight=2.5`、`rollout_residual_scale=0.7`）后，新的
  `model_best` 经 full eval 复核达到：
  - `val + 20`: `51.607 / 103.955`
  - `test + 20`: `34.911 / 69.133`
- 这比旧 candidate 明显更好，说明 `macro cycle rollout` 需要参数适配后才能把信号
  传到最终轨迹；但它仍明显落后 baseline，因此当前结论是“方向成立但强度不足”，
  还不能讲成“已经超过 baseline”。

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

8. **Stage 47: 最小增量 refine 证明 macro rollout 不是伪信号**
   - 从 `warmup50_refine50_p0_seqgat_relation_v1/checkpoint/model_best.pth.tar`
     warm-start，保持 `refine` 协议与 `aux_rollout_weight=2.5`、`teacher_forcing=0.6`、
     `rollout_residual_scale=0.7` 一致，只追加 1 轮短程 `50` 个 train batch。
   - 训练内 `20-batch val` 快照曾到 `54.491 / 106.453`，随后又退化到
     `83.834 / 169.097`，说明这条线有明显“早收益、后过冲”的特征。
   - 对最终 `model_best` 做 full eval 后得到：
     - `val + 20`: `51.607 / 103.955`
     - `test + 20`: `34.911 / 69.133`
   - 结论：prediction-time cycle rollout 不是仅在 `val` 上漂漂亮亮的假信号，
     它确实把 `test + 20` 从 `43.736 / 85.691` 拉到了 `34.911 / 69.133`；
     但和 baseline (`15.359 / 31.514`) 仍有大差距，后续重点应转到
     “如何更稳地利用这条信号”，而不是再争论它是否存在。

9. **Stage 48: continuation 判负，状态分支“在动但耦合太弱”**
   - 从 `Stage 47` 的 `model_best` 出发，尝试更保守的 continuation：
     `lr=3e-4`、`max_train_batches=30`、`max_val_batches=20`。
   - 训练内 `20-batch val` 快照依次为：
     - `77.511 / 154.035`
     - `69.903 / 138.006`
     - `83.454 / 165.355`
   - 没有任何一次接近或刷新当前 best；这条 low-lr continuation 当前应视为负结果。
   - 基于 `Stage 47` checkpoint 做四个单变量 `val + 20` 离线消融：
     - base: `51.607 / 103.955`
     - `disable_queue_rollout`: `51.605 / 103.947`
     - `disable_decoder_state_residual`: `51.693 / 104.259`
     - `disable_state_gating`: `51.649 / 104.078`
     - `disable_lane_queue_anchor`: `51.607 / 103.955`
   - 四个开关都能改变单 batch 前向输出，但 `best-of-20` comparable 指标几乎不变。
   - 额外诊断：
     - base 与 `all_off` 在 `val + num_samples=1` 上几乎重合
       (`59.349 / 121.138` vs `59.410 / 121.388`)
     - decoder step residual 平均范数约 `0.0278`
     - queue hidden 平均范数约 `2.1306`
     - `step_residual / queue_hidden ≈ 1.3%`
   - 当前判断：瓶颈不是“状态分支不存在”，而是“状态分支对 decoder 的数值耦合仍过弱”。

10. **Stage 49: 引入独立的 decoder-state residual scale，并确认它只在 val 侧给出弱正信号**
    - 为了把 `decoder state residual` 与 `rollout queue residual` 解耦，新增
      `decoder_state_residual_scale`，默认 `1.0`，train/eval 两侧都可显式覆盖。
    - 定向测试已补齐并通过：
      - `decoder_state_residual_scale=0.0` 时 state residual 归零
      - train stage defaults 会补上该字段
      - eval parser / get_generator 能正确透传该字段
    - 基于 `Stage 47` checkpoint 的推理侧扫描结果：
      - base `val + 1`: `59.349 / 121.138`
      - `scale=2.0` `val + 1`: `59.295 / 120.904`
      - `scale=4.0` `val + 1`: `59.205 / 120.487`
      - base `val + 20`: `51.607 / 103.955`
      - `scale=4.0` `val + 20`: `51.383 / 103.133`
      - base `test + 20`: `34.911 / 69.133`
      - `scale=4.0` `test + 20`: `35.147 / 69.575`
    - 结论：
      - 增强 `decoder state residual` 的确能把单样本指标与 `val + 20` 往下拉，
        说明“residual 太弱”这个判断是对的。
      - 但这条收益还没有稳定迁移到 `test + 20`，说明仅靠推理时放大 residual
        不够，后续需要训练阶段一起重新适配。

## 4. 推荐下一步

固定顺序如下：

1. 保留新的 `decoder_state_residual_scale` 旋钮，并把下一轮短程 refine 建立在
   `decoder_state_residual_scale > 1.0` 的协议上
2. 训练后先复核 `val + num_samples=20`
3. 只有 `val` 继续保持正向时，再补 `test + num_samples=20`

如果上述强化仍没有明确正向信号，再优先按下面顺序回到结构/依赖诊断：

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
- `Stage 46` 证明“只做推理侧改动”不够。
- `Stage 47` 证明“最小再训练后它确实有效”，但离 baseline 仍远，因此后续所有叙事
  都必须围绕“如何稳定放大这条已存在的正向信号”，而不是提前声称已经赢了。
- `Stage 48` 进一步说明：当前最弱的一环不是“状态分支不存在”，而是
  “状态分支对 decoder 的数值耦合仍过弱”，因此后续实验应优先围绕注入强度展开。
- `Stage 49` 进一步说明：只在推理时放大 state residual 可以改善 `val`，但还不能稳定
  改善 `test`；下一步必须把这条增强协议带回训练阶段一起适配。
