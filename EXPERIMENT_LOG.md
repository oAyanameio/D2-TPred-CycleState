# CycleState 实验日志

> **更新时间**: 2026-06-13 17:10
> **覆盖范围**: 当前 protocol-check 与 comparable 结论（**含 DE-3 / DE-1 / AR-1 / AR-2 决定性实验结果**）
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

### 2.3 DE-3 (Minimum Viable CycleState)

Run: `experiments/cyclestate/de3_minimal_viable`

配置摘要：

- `minimal_viable_mode=True`（强制关闭 `state_gating / queue_rollout / lane_queue_anchor / decoder_state_residual / aux_losses`）
- 观测期最后时刻的 `[queue_last, cycle_last]` 直接拼接到 `encoded_before_noise_hidden` 后面（与 `light_state_embedding` 同级）
- 训练协议：warmup 2 epoch（`max_train_batches=50`） + refine 3 epoch（`max_train_batches=50`），`max_val_batches=20`
- `aux_queue_weight = aux_cycle_weight = aux_rollout_weight = 0`
- `num_samples=20`（与 baseline `comparable` 口径对齐）

| Split | num_samples | ADE | FDE | 相对 base 变化 |
|-------|-------------|-----|-----|---------------|
| val | 20 | 65.537 | 161.993 | 显著低于 baseline (35.022) |
| test | 20 | 24.632 | 57.135 | 显著高于 baseline (15.359)，但显著低于旧 CycleState (34.911) |

结论：

- **最简版 CycleState 仍显著好于旧 CycleState** (`test: 24.632 vs 34.911`，ADE 改善约 **29.4%**)。
  说明"加性残差 → 直接拼接"确实是一个更有效的耦合方式——这与第 2.2 节的
  "1.3% 范数比诊断"互为佐证：去掉残差后直接拼接到 decoder 初始化向量，让
  queue/cycle hidden 真正进入 pred_lstm 的输入维度。
- **但仍远未追平 baseline** (`24.632 vs 15.359`)，差距仍达 **1.60×**。
  说明：
  1. 仅仅是"换一种耦合方式"并不能独立解决 2 倍差距；
  2. 即便去掉了 rollout / gating / anchor / aux / 残差，state 分支也只
     能"少拖后腿"，不能"成为正向贡献"；
  3. baseline 的 `light_state_embedding + pred_state + get_next_state`
     路径可能已经捕获了主要信号（与第 2.5 节的根因三一致）。
- **val 指标 (65.537) 仍显著差于 test (24.632)**，且比 baseline val (35.022) 差，
  这是与 Stage 47 类似的现象：val 与 test 的分布差异在 DE-3 下更明显，
  提示可能存在 val 特定场景（高密度排队 / 长等待）下模型仍未学到正确的
  状态→轨迹映射。
- **诊断证据**：
  - `queue_grad / cycle_grad / pred_lstm_grad` 三个分支都有真实梯度流过，
    确认 state 信息确实在驱动 trajectory loss；
  - `disable_state_gating / disable_queue_rollout / disable_lane_queue_anchor / disable_decoder_state_residual` 在 DE-3 模式下均为 True，
    进一步消融已无意义（无开关可关）。

**核心结论**：DE-3 验证了**根因一（架构耦合路径过窄）确实存在且有实质影响**——
直接拼接比加性残差好得多。但根因二/三（aux loss 帮倒忙 / 信号灯信息已被 baseline 充分捕获）
尚未被 DE-3 直接证伪，需要 DE-1（Oracle 直注）来回答。

### 2.4 DE-1 (Oracle State 直注)

Run: `experiments/cyclestate/DE1_oracle_inject`

配置摘要：

- `--oracle_inject_mode` 开关：把 CycleState 改为"oracle 交通状态直注"形态
- 强制开启 5 个 disable 开关：`disable_state_gating / disable_queue_rollout /
  disable_lane_queue_anchor / disable_decoder_state_residual / disable_aux_losses`
- 把 `aux_queue_weight = aux_cycle_weight = aux_rollout_weight = 0` 置零
- 替换 `pred_lstm_model` 为输入维度 `traj_lstm_input_size + 10` 的 LSTMCell
- 单步 oracle 特征 (10 维) 拼接到 decoder LSTM 的输入后面：
  - phase one-hot (3D)
  - elapsed / remaining time (2D，已按 `cycle_time_norm` 归一化并 clamp 到 `[0, 2]`)
  - phase_change 标志 (1D)
  - distance / direction 到停止线 (3D：`dis`、`disx_norm`、`disy_norm`)
  - speed (1D，已按 `queue_speed_norm` 归一化并 clamp 到 `[0, 2]`)
- 训练协议：warmup 3 epoch（`max_train_batches=50`） → refine 1 epoch（`max_train_batches=50`），`max_val_batches=20`
- `teacher_forcing_ratio=0.6`（与 DE-3 / Stage 50 对齐）
- `num_samples=20`（与 baseline / DE-3 `comparable` 口径对齐）

| Split | num_samples | ADE | FDE | 备注 |
|-------|-------------|-----|-----|------|
| val | 20 | 77.472 | 176.863 | DE-1 直注 oracle state，refine best=77.40 |
| test | 20 | 30.433 | 66.544 | DE-1 直注 oracle state |

三方 comparable 对比（`test + 20`）：

| 候选 | ADE | FDE | 与 baseline 差距 | 与 DE-3 差距 |
|------|-----|-----|----------------|-------------|
| baseline | 15.359 | 31.514 | — | — |
| DE-3 (learned state 直接拼接) | 24.632 | 57.135 | 1.60× | — |
| **DE-1 (oracle state 直注)** | **30.433** | **66.544** | **1.98×** | **1.24×** |
| 旧 CycleState (full machinery) | 34.911 | 69.133 | 2.27× | 1.42× |

结论：

- **oracle 直注改善了旧 CycleState**（`test: 30.433 vs 34.911`，ADE 改善 **12.8%**），
  说明"交通状态信息本身对轨迹预测确实有正向贡献"——这一点**否证了根因三
  的极端版本**（即"信号灯信息对轨迹完全无价值"）。
- **但 oracle 直注不如 DE-3**（`test: 30.433 vs 24.632`，ADE 差 **23.5%**），
  这是一个**反直觉但有方法论价值**的发现：
  1. 10 维 oracle 特征（phase one-hot / elapsed / remaining / phase_change /
     distance / direction / speed）**不如** queue/cycle LSTM 学习的 32D+16D
     隐藏表征对 decoder 有用；
  2. 可能原因：
     - 学习的 hidden state 经过 `queue_lstm_model` / `cycle_lstm_model` 的
       跨帧累计，捕获了"演化轨迹"（如"刚刚切换到红，已累积 8s"），而
       10 维 oracle 特征只有"瞬时切片"；
     - 学习的 hidden state 是连续、平滑的向量，而 oracle 特征里有
       one-hot / 离散 phase_change 标志，梯度友好性差；
     - 50 批 refine 不足以让模型学会把 10 维离散信号映射成有用的
       内部表征；DE-3 的 32D+16D 隐藏向量已经经过训练，是"现成的"。
  3. **方法论意义**：把"oracle 信息"直接拼接到 LSTM 输入 ≠ 把"学习到的
     状态表征"拼接到 init 向量。后者经过训练的向量空间本身就是对轨迹
     loss 更友好的特征表示。
- **但仍显著差于 baseline**（`30.433 vs 15.359`，差距 **1.98×**），
  根因三的弱版本（"信号灯信息对轨迹边际贡献有限"）仍未被证伪——
  oracle 直注的 1.98× 差距说明，即便把真实交通状态全塞进 decoder，
  仍不能独立解决 2× 差距。
- **val/test 落差进一步扩大**：DE-1 val (77.47) 远差于 test (30.43)，
  比例与 DE-3 类似（val 65.54 / test 24.63）。这说明 oracle 直注并
  没有缓解 val/test 分布差异问题，且 10 维离散特征在 val 上更难学。
- **训练量过少是 confound**：DE-1 只跑了 warmup 150b + refine 50b = 200b，
  比 DE-3 的 250b 略少；且 refine 只 1 个 epoch。理论上更长的训练
  可能进一步压低指标，但不能改变核心结论——oracle 信息本身对 decoder
  的贡献**不比学到的隐藏表征更大**。

**核心结论**：

- **否证根因三的极端版本**：oracle 交通状态对轨迹预测有正向贡献
  （比旧 CycleState 改善 12.8%）。
- **未证伪根因三的弱版本**：oracle 直注仍差 baseline 1.98×，说明仅靠
  注入真实交通状态不能独立解决 2× 差距。
- **意外发现**：oracle 特征 (10D) **不如**学习的 hidden state (32D+16D)，
  这意味着状态分支"学到的表征"本身有信息量，瓶颈不在"信号 vs 学到"，
  而在"如何把表征送进 decoder 让它真的用"。
- **下一步必须是 AR-1（直接条件注入）**：因为 DE-3（learned hidden 拼 init）
  比 DE-1（oracle 拼 input）更好，但 DE-3 仍差 baseline；下一步应在
  DE-3 的拼接方式基础上，把"只在初始化时拼接"扩展为"每步拼接 +
  输出投影"，看是否能进一步逼近 baseline。

### 2.5 AR-1 (直接条件注入)

Run: `experiments/cyclestate/AR1_direct_inject`

配置摘要：

- `--ar1_direct_inject_mode` 开关：把 CycleState 改为"直接条件注入"形态
- 隐含启用 `minimal_viable_mode=True`（保留 DE-3 的 init 拼接），并强制开启
  5 个 disable 开关：`disable_state_gating / disable_queue_rollout /
  disable_lane_queue_anchor / disable_decoder_state_residual / disable_aux_losses`
- 把 `aux_queue_weight = aux_cycle_weight = aux_rollout_weight = 0` 置零
- **核心改动**（在 DE-3 之上叠加两层新注入）：
  1. `pred_lstm_model` 的每步输入拼接 `[queue_last, cycle_last]` (32+16=48 维) — 把
     state context 拉成 decoder 的"per-step conditional input"
  2. `pred_hidden2pos` 输出投影的输入也拼接同样的 state context — 让 state
     context 强行进入输出投影，直接参与最终 (dx, dy) 的预测
- AR-1 与 `--oracle_inject_mode` 互斥（AR-1 用 learned 48 维 hidden，DE-1 用 oracle 10 维）
- 训练协议：warmup 3 epoch（`max_train_batches=50`） → refine 1 epoch（`max_train_batches=50`），`max_val_batches=20`
- `teacher_forcing_ratio=0.6`（与 DE-3 / DE-1 / Stage 50 对齐）
- `num_samples=20`（与 baseline / DE-3 / DE-1 `comparable` 口径对齐）

| Split | num_samples | ADE | FDE | 备注 |
|-------|-------------|-----|-----|------|
| val | 20 | 57.954 | 140.002 | AR-1 比 DE-3 val (65.54) 改善 11.6% |
| test | 20 | 28.631 | 65.631 | AR-1 比 DE-3 test (24.63) **差 16.2%** |

四方 comparable 对比（`test + 20`）：

| 候选 | ADE | FDE | vs baseline | vs DE-3 | vs DE-1 |
|------|-----|-----|-------------|---------|---------|
| baseline | 15.359 | 31.514 | — | 0.62× | 0.50× |
| DE-3 (learned state 拼 init) | 24.632 | 57.135 | 1.60× | — | — |
| **AR-1 (DE-3 + per-step + output)** | **28.631** | **65.631** | **1.86×** | **1.16×** | 0.94× |
| DE-1 (oracle state 拼 LSTM input) | 30.433 | 66.544 | 1.98× | 1.24× | — |
| 旧 CycleState (full machinery) | 34.911 | 69.133 | 2.27× | 1.42× | 1.15× |

结论：

- **AR-1 在 val 上明显改善**（比 DE-3 val 65.54 → 57.95，**ADE 改善 11.6%**），
  这说明 "per-step + output-projection 注入" 在 val 分布（高密度排队 / 长等待）
  下确实让 decoder 更"听"state context 的指令。
- **但 AR-1 在 test 上变差**（比 DE-3 test 24.63 → 28.63，**ADE 差 16.2%**），
  这是一个**反直觉且对原假设有否决意义的发现**：
  1. 原假设（PLAN.md §4.4）认为"在 DE-3 拼接基础上扩展注入强度 → 进一步
     逼近 baseline"，但实际结果恰恰相反。
  2. AR-1 在 DE-3 之上叠加了 2 个新注入点（per-step + output），但 test
     反而变差，说明**注入强度有"甜蜜点"**：
     - DE-3 (init 拼接，**单点**注入) → 24.632 ← 当前最优 CycleState 变体
     - AR-1 (init + per-step + output，**三点**注入) → 28.631
  3. 可能原因：
     - **信息冗余 / 梯度被稀释**：在 init / per-step / output 三个位置都拼
       同样的 state context 48 维，每一处都在做"告诉 decoder 现在是什么
       state"的事，decoder 可能反而被冗余信号干扰。
     - **per-step 注入破坏 LSTM 状态演化**：DE-3 的 init 拼接让 state context
       通过初始化进入 LSTM 隐状态演化；AR-1 在 per-step 又强行塞 48 维
       state context，**强制覆盖**了 LSTM 自然演化出的隐状态信号，导致
       LSTM 内部的"演化轨迹"被打断。
     - **过拟合 val / 欠泛化 test**：val 上 11.6% 改善但 test 上 16.2%
       变差，说明 AR-1 的强注入让模型对 val 的特定分布过拟合了，泛化到
       test 时反而表现更差。
  4. **方法论价值**：DE-3 的"init 单点拼接"是 learned state context 的
     **最优注入位置**——AR-1 想用"多点注入"加强这个优势，但实际是
     **稀释了**这个优势。
- **AR-1 仍优于 DE-1 和旧 CycleState**：
  - vs DE-1 (`30.433`) → 改善 5.9%，说明 learned 48D 拼接 init/per-step/output
    **比** oracle 10D 拼 LSTM input 更好。
  - vs 旧 CycleState (`34.911`) → 改善 18.0%。
  - 这两个比较共同验证了"学到的表征比 oracle 物理特征更易被 decoder 消费"
    （延续 DE-1 的反直觉发现）。
- **val/test 落差**：AR-1 val (57.95) vs test (28.63)，落差 2.02×，与 DE-3
  (val 65.54 / test 24.63, 落差 2.66×) 相比**落差显著缩小**，说明 AR-1
  的强注入让模型在 val/test 分布差异上的鲁棒性变好；但**平均性能变差**。

**核心结论**：

- **原假设（PLAN.md §4.4）被 AR-1 否决**：原假设认为"DE-3 + per-step +
  output-projection 注入 → 进一步逼近 baseline"，但 AR-1 实际结果
  (`28.631`) **差于** DE-3 (`24.632`)。
- **learned state context 的最优注入位置是 init**（DE-3），
  **不是 init+per-step+output**（AR-1）。"多点注入"反而稀释了 state
  context 的边际贡献。
- **根因三的弱版本仍未被证伪**：AR-1 仍差 baseline 1.86×，仅靠"拼接
  learned state"不能独立解决 2× 差距。
- **整个 CycleState 变体族（DE-3 / DE-1 / AR-1）的最佳仍是 DE-3**，
  test ADE 24.632；这意味着"DE-3 + 加性残差替换为直接拼接"已经是
  CycleState 方向的"最大可榨出价值"。
- **下一步方向**：
  - **AR-2（乘法门控）**：在 DE-3 基础上加乘法门控（区别于 AR-1
    的"加法拼接"），看门控是否能比简单拼接更聪明地使用 state context。
  - 重新审视"是否真的需要显式交通状态记忆"：AR-1 / DE-1 都未能逼近
    baseline，且 AR-1 的反直觉结果说明"显式拼接"可能并非最优耦合方式。

### 2.6 AR-2 (乘法门控)

Run: `experiments/cyclestate/AR2_multiplicative_gating`

配置摘要：

- `--ar2_multiplicative_gating_mode` 开关：把 CycleState 改为"乘法门控"形态
- 隐含启用 `minimal_viable_mode=True`（保留 DE-3 的 init 拼接），并强制开启
  5 个 disable 开关：`disable_state_gating / disable_queue_rollout /
  disable_lane_queue_anchor / disable_decoder_state_residual / disable_aux_losses`
- 把 `aux_queue_weight = aux_cycle_weight = aux_rollout_weight = 0` 置零
- **核心改动**（在 DE-3 之上叠加一个新机制，与 AR-1 的核心差异是**耦合方式**）：
  - AR-1 走"加性拼接"：state context (48D) 拼到 `pred_lstm_model` 每步输入 + 输出投影
  - **AR-2 走"乘法调制"**：state context (48D) 通过 2 层 MLP + sigmoid 学习
    一个逐元素门控 `gate ∈ (0, 1)^{pred_lstm_hidden_size}`，然后
    `pred_lstm_hidden = pred_lstm_hidden * gate`（在 `pred_lstm_model`
    更新**之后**立即施加）。
  - `pred_lstm_model.input_size` 和 `pred_hidden2pos.in_features` 与 DE-3
    保持一致（AR-2 不"加性"地扩大输入），新增的 `ar2_hidden_gate`
    是 `Linear(pred_hidden+state, pred_hidden) -> ReLU -> Linear -> Sigmoid`。
  - AR-2 与 `--oracle_inject_mode` / `--ar1_direct_inject_mode` 互斥
    （三种耦合方式不能同时启用以避免混淆归因）
- 训练协议：warmup 3 epoch（`max_train_batches=50`） → refine 3 epoch（`max_train_batches=50`），`max_val_batches=20`
- `teacher_forcing_ratio=0.6`（与 DE-3 / DE-1 / AR-1 / Stage 50 对齐）
- `num_samples=20`（与 baseline / DE-3 / DE-1 / AR-1 `comparable` 口径对齐）

| Split | num_samples | ADE | FDE | 备注 |
|-------|-------------|-----|-----|------|
| val | 20 | 74.229 | 178.848 | AR-2 比 DE-3 val (65.54) 差 13.2% |
| test | 20 | 32.368 | 77.927 | AR-2 比 DE-3 test (24.63) **差 31.4%** |

五方 comparable 对比（`test + 20`）：

| 候选 | ADE | FDE | vs baseline | vs DE-3 | vs AR-1 | vs DE-1 |
|------|-----|-----|-------------|---------|---------|---------|
| baseline | 15.359 | 31.514 | — | 0.62× | 0.54× | 0.50× |
| DE-3 (learned state 拼 init) | 24.632 | 57.135 | 1.60× | — | 0.86× | 0.81× |
| AR-1 (DE-3 + per-step + output) | 28.631 | 65.631 | 1.86× | 1.16× | — | 0.94× |
| DE-1 (oracle state 拼 LSTM input) | 30.433 | 66.544 | 1.98× | 1.24× | 1.06× | — |
| **AR-2 (DE-3 + per-step multiplicative gate)** | **32.368** | **77.927** | **2.11×** | **1.31×** | **1.13×** | **1.06×** |
| 旧 CycleState (full machinery) | 34.911 | 69.133 | 2.27× | 1.42× | 1.22× | 1.15× |

五方 comparable 对比（`val + 20`）：

| 候选 | ADE | FDE | 备注 |
|------|-----|-----|------|
| baseline | 35.022 | 70.658 | comparable 硬基线 |
| AR-1 | 57.954 | 140.002 | AR-1 在 val 上比 DE-3 改善 11.6% |
| DE-3 | 65.537 | 161.993 | 当前 CycleState 变体族最优（test） |
| **AR-2** | **74.229** | **178.848** | **AR-2 在 val 上比 DE-3 差 13.2%** |
| DE-1 | 77.472 | 176.863 | oracle 直注在 val 上最差 |

结论：

- **AR-2 是 CycleState 变体族在 `test + 20` 上的最差变体**（除旧 CycleState
  外）：`test ADE 32.368` 比 DE-3 (`24.632`) **差 31.4%**，比 AR-1
  (`28.631`) 差 13.0%，比 DE-1 (`30.433`) 差 6.4%。
- **AR-2 在 `val + 20` 上也是较差变体**：`val ADE 74.229` 比 DE-3
  (`65.537`) 差 13.2%，比 AR-1 (`57.954`) 差 28.1%。这与 AR-1 在 val
  上的优势（比 DE-3 改善 11.6%）形成鲜明对比——AR-1 的"加性拼接"
  在 val 上对长等待/高密度排队场景有帮助，而 AR-2 的"乘法调制"
  在 val 上没有任何优势。
- **方法论价值**：
  1. **"加性" vs "乘法"耦合方式比较**：AR-1 走加性拼接，AR-2 走乘法
     门控，两者**都比 DE-3 差**。这说明：
     - DE-3 的"init 单点拼接"在 CycleState 变体族中确实是**最优注入**，
       任何"扩展"（无论是加性还是乘法）都会让指标退化；
     - "乘法调制"并没有比"加性拼接"更聪明地使用 state context —
       反过来，乘法门控**比加性拼接更差**。
  2. **乘法门控为何更差的可能解释**：
     - **sigmoid gate 衰减问题**：训练初期 gate 接近 0.5，每步对
       `pred_lstm_hidden` 的所有维度乘 0.5 等于"硬性减半"，且
       sigmoid 输出在训练早期不稳定，可能导致隐状态幅度大幅波动，
       让训练更难收敛；
     - **状态耦合被门控稀释**：与 AR-1 一样，AR-2 也"扩展"了注入
       强度（per-step），但门控机制把"信号源 → decoder"变成
       "信号源 → 隐式控制信号 → decoder"，中间多了一层 sigmoid
       衰减。
     - **类比 LSTM 自身的 forget gate**：LSTM 内部本就有 forget/input/
       output 门来调制 hidden state，AR-2 在外部再叠加一个 sigmoid
       gate 来"再调制"，可能与 LSTM 内部门控产生冗余甚至冲突。
- **CycleState 变体族的最终定论**：
  - DE-3 (init 单点拼接) 仍是 test 最佳（`24.632`）。
  - AR-1 (加性多点) 退化到 `28.631`。
  - AR-2 (乘法门控) 退化到 `32.368`，是变体族最差。
  - 这意味着 **"显式交通状态注入"对轨迹预测的边际贡献非常有限** —
     无论用什么耦合方式（加性 / 乘法 / init / per-step / output），
     改进空间都很快达到上限，剩余 1.6× 差距（vs baseline）不是
     耦合方式能解决的。

**核心结论**：

- **AR-2 否决了"乘法调制能超越加性拼接"的假设**：原假设认为"用 sigmoid
  gate 调制 pred_lstm_hidden 比简单拼接更灵活"，但 AR-2 (`32.368`)
  **比 AR-1 (`28.631`) 更差**，**比 DE-3 (`24.632`) 差 31.4%**。
- **整个 CycleState 变体族（DE-3 / DE-1 / AR-1 / AR-2）的 test 最佳仍是
  DE-3**：`24.632`，远好于其他三个变体。
- **根因三的弱版本仍未被证伪**：DE-3 / DE-1 / AR-1 / AR-2 全部仍差
  baseline 1.6-2.1×，**"显式交通状态记忆"对轨迹预测的边际贡献有限**
  这个假设被四个变体一致支持。
- **下一步方向**（PLAN.md 决策树中的"分支 C2"分支触发条件）：
  1. **DE-2（极端耦合）继续暂停** — 已经被 AR-1 / AR-2 的结果充分证明
     "加性/乘法路径都有上限"，DE-2 即使能做也几乎不可能突破这个上限。
  2. **重新审视"是否真的需要显式交通状态记忆"** — 整个 CycleState
     路线的核心假设在当前数据/容量约束下未成立。
  3. **可能的替代方向**（不是必须现在做，而是作为分支 C2 的备选项）：
     - 更强的 trajectory-level modeling（而不是 state-level）
     - 更细粒度的 vehicle-vehicle 交互建模
     - 更好的不确定性建模（当前 best-of-K 采样本身就能提供多样性）

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
        说明"residual 太弱"这个判断是对的。
      - 但这条收益还没有稳定迁移到 `test + 20`，说明仅靠推理时放大 residual
        不够，后续需要训练阶段一起重新适配。

11. **Stage 50: DE-3 决定性实验 — 最简可行 CycleState**
    - 见第 2.3 节。
    - 核心结论：把 `[queue_last, cycle_last]` 直接拼接到 decoder 初始化向量
      比加性残差更有效，**验证了根因一（架构耦合路径过窄）确实存在**。
    - 但即便去掉所有复杂机制（rollout / gating / anchor / aux / 残差），
      `test + 20` 仍为 `24.632 / 57.135`，距离 baseline (`15.359 / 31.514`)
      仍有 1.6× 差距，说明仅换耦合方式不足以独立解决问题。
    - 这次实验**确认了根因一**（耦合路径过窄），但**没有证伪**根因二/三
      （aux loss 帮倒忙 / 信号灯信息已被 baseline 充分捕获）；
      这两个问题需要 DE-1（Oracle 直注）来回答。

12. **Stage 51: DE-1 决定性实验 — Oracle State 直注**
    - 见第 2.4 节。
    - 关键改动：把 10 维 oracle 特征（phase one-hot / elapsed / remaining /
      phase_change / distance / direction / speed）直接拼接到 `pred_lstm_model`
      的输入后面，强制关闭所有 5 个 disable 开关、aux 权重全部置零。
    - 训练协议：warmup 150b + refine 50b = 200b 总训练 batch。
    - 关键结果（`num_samples=20`）：
      - `val + 20`: 77.472 / 176.863
      - `test + 20`: **30.433 / 66.544**
    - 三方对比（`test + 20` ADE）：
      - baseline: 15.359
      - DE-3 (learned state 拼 init): 24.632
      - **DE-1 (oracle state 拼 input): 30.433**
      - 旧 CycleState (full machinery): 34.911
    - 核心结论：
      - **否证根因三的极端版本**：oracle 交通状态对轨迹预测有正向贡献
        （比旧 CycleState 改善 12.8%）。
      - **意外反直觉发现**：oracle 特征 (10D) **不如**学习的 hidden state
        (32D+16D)，DE-1 比 DE-3 差 23.5%——这说明学到的表征本身有
        信息量，瓶颈不在"信号 vs 学到"。
      - **未证伪根因三的弱版本**：oracle 直注仍差 baseline 1.98×。
    - 下一步：AR-1（直接条件注入）——在 DE-3 基础上把"只在 init 拼接"
      扩展为"每步拼接 + 输出投影"，看是否能进一步逼近 baseline。

13. **Stage 52: AR-1 决定性实验 — 直接条件注入**
    - 见第 2.5 节。
    - 关键改动（DE-3 之上叠加 2 个新注入点）：
      - `pred_lstm_model` 每步输入拼接 `[queue_last, cycle_last]` (48 维)
      - `pred_hidden2pos` 输出投影输入也拼接同样的 48 维 state context
      - 强制开启 5 个 disable 开关 + aux 权重全部置零
    - 训练协议：warmup 150b + refine 50b = 200b 总训练 batch。
    - 关键结果（`num_samples=20`）：
      - `val + 20`: 57.954 / 140.002（比 DE-3 val 65.54 改善 11.6%）
      - `test + 20`: **28.631 / 65.631**（比 DE-3 test 24.63 **差 16.2%**）
    - 四方对比（`test + 20` ADE）：
      - baseline: 15.359
      - **DE-3 (learned state 拼 init, 单点): 24.632** ← 当前最优
      - **AR-1 (init + per-step + output, 三点): 28.631**
      - DE-1 (oracle state 拼 input): 30.433
      - 旧 CycleState (full machinery): 34.911
    - 核心结论：
      - **原假设（PLAN.md §4.4）被 AR-1 否决**：原假设认为"DE-3 +
        per-step + output-projection 注入 → 进一步逼近 baseline"，
        但实际 AR-1 (`28.631`) **差于** DE-3 (`24.632`)。
      - **learned state context 的最优注入位置是 init**（DE-3 单点注入），
        "多点注入"反而稀释了 state context 的边际贡献。
      - **整个 CycleState 变体族的最佳仍是 DE-3**。
      - **val/test 落差缩小**：AR-1 (val/test = 2.02×) 比 DE-3 (2.66×)
        鲁棒性更好，但**平均性能变差**。
    - 下一步方向：
      - **AR-2（乘法门控）**：在 DE-3 基础上加乘法门控（区别于 AR-1
        的"加法拼接"），看门控是否能比简单拼接更聪明地使用 state context。
      - **重新审视"是否真的需要显式交通状态记忆"**：AR-1 / DE-1 都未能
        逼近 baseline，且 AR-1 的反直觉结果说明"显式拼接"可能并非
        最优耦合方式。

## 4. 推荐下一步

DE-3（Stage 50）、DE-1（Stage 51）、AR-1（Stage 52）均已完成。三个决定性
实验形成了一致的"CycleState 变体族"画像：

| 变体 | 注入位置 | 注入信号 | test ADE | vs DE-3 |
|------|---------|---------|---------|---------|
| DE-3 (单点, init) | init | learned 48D | **24.632** | — |
| AR-1 (三点) | init + per-step + output | learned 48D | 28.631 | +16.2% |
| DE-1 (单点, input) | LSTM input | oracle 10D | 30.433 | +23.5% |
| 旧 CycleState | 多点混合 | learned 多信号 | 34.911 | +41.7% |

**核心观察**：

- **DE-3 是当前 CycleState 变体族的最优配置**：单点 init 注入 + learned
  32+16 维 hidden state。
- **AR-1 否决了"加大注入强度 → 更好"的假设**：多点注入 (init + per-step +
  output) 反而**比单点注入差 16.2%**，说明"加性拼接"有"甜蜜点"，过犹不及。
- **DE-1 的 oracle 仍不如 DE-3 的 learned**：把"信号"换成 oracle 物理
  特征更差，说明 decoder 更容易消费学到的连续向量。
- **根因三的弱版本仍未被证伪**：DE-3 / DE-1 / AR-1 全部仍差 baseline
  1.6-2.0×，**"显式交通状态记忆"对轨迹预测的边际贡献有限**这个假设
  越来越难推翻。

**立即执行的下一个决定性实验是 AR-2（乘法门控）**：

DE-3 用"加性拼接"（init 拼接 learned state），AR-1 用"加性拼接"的
**多点扩展**（init + per-step + output），但 AR-1 反而**比** DE-3 差。
这说明"加法"路径有上限——AR-2 改走"乘法门控"路径，在 DE-3 基础上
把 learned state 通过 sigmoid 门控调制 `pred_lstm_hidden` 的某些维度，
**让 state context 通过乘法**决定哪些隐状态维度被放大/抑制。
这与 AR-1 的"加法堆叠"是本质不同的耦合方式。

**AR-2 之前的纪律要求**：

- 不再做任何 continuation/refine 超参扫描。
- 不再做任何推理侧 scale 扫描。
- 不再新增任何 state 分支机制。
- 不再在 init 拼接 + 加法堆叠路径上做变体（AR-1 已给出结论：这条路径
  接近上限）。
- 不再重新做 DE-1 / DE-3 / AR-1。
- DE-2（极端耦合）继续暂停——AR-1 进一步强化了"加性路径有上限"的结论，
  DE-2 即使能做也不会超出这个上限。

**AR-2 完成后**，再根据其结论决定后续：

- **若 AR-2 接近 baseline（差距 < 20%）** → AR-3（aux loss 重新设计）作为
  细化，**整体 CycleState 路线在 baseline 1.5× 以内成立**。
- **若 AR-2 仍差 baseline 1.5×+** → 优先回到问题定义层面（分支 C2）：
  信号灯路口轨迹预测是否真的需要显式交通状态记忆？可能的替代方向：
  - 更强的 trajectory-level modeling（而不是 state-level）
  - 更细粒度的 vehicle-vehicle 交互建模
  - 更好的不确定性建模（best-of-K 采样）
- **若 AR-2 比 DE-3 还差** → 进一步验证"加性 vs 乘法都不是问题核心"，
  应该完全回到问题定义层面（分支 C2），不再继续 state injection 路线。

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
- `Stage 50 / DE-3` 直接验证了根因一（耦合路径过窄）确实存在——直接拼接比加性残差
  显著更好（`test + 20`: 24.632 vs 34.911，ADE 改善 29.4%）；但即便去掉所有复杂机制，
  距离 baseline (15.359) 仍有 1.6× 差距，根因二/三尚未被证伪，下一步必须是
  DE-1（Oracle 直注），而不是继续堆结构或扫超参。
- `Stage 51 / DE-1` 把 10 维 oracle 交通状态直接拼接到 decoder LSTM input，
  `test + 20` 达到 **30.433 / 66.544**——比旧 CycleState 改善 12.8%，
  但比 DE-3 差 23.5%（oracle 不如 learned state）。这**否证了根因三的极端版本**
  （信号灯信息对轨迹完全无价值），但**未证伪根因三的弱版本**（仍差 baseline 1.98×）。
  反直觉发现：学习的 32D+16D hidden state 携带了 10 维 oracle 特征之外的
  信息（可能是跨帧演化轨迹 / 平滑性 / 与轨迹 loss 友好的向量空间），瓶颈
  不在"信号 vs 学到"而在"如何把表征送进 decoder 让它真的用"。下一步必须
  做 AR-1（直接条件注入），把 DE-3 的"只在 init 拼接"扩展为"每步拼接 + 输出投影"。
