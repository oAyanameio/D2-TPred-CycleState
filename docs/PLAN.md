# CycleState Live Plan

> **更新时间**: 2026-06-11
> **作用**: 只维护当前活跃 backlog、执行顺序和验收门槛。
> **规则**: 已解决项不再在本文件展开维护；历史修复细节以 git 历史、测试与当前代码为准。

## 1. 当前目标

把项目从"当前最佳候选仍明显落后 baseline"推进到"协议正确、口径清楚、可复现地评估下一步是否还有方法价值"。

当前默认原则：

- 先修协议与状态进入预测期的稳定性，再谈结构增强。
- 不混用 split、不混用 `num_samples`、不用 `test` 做模型选择。
- 所有正式数字都按 `comparable` 口径记录。

---

## 2. 研究诊断：为什么 CycleState 一直不如 baseline

> **说明**: 本节是 2026-06-11 综合诊断的正式版本，目的是在继续实验之前，先把"问题到底是什么"写清楚。诊断基于当前所有可比证据、消融结果和架构分析，不引入新实验假设。

### 2.1 差距量级

| 指标 | Baseline (test, 20) | CycleState Best (test, 20) | 倍率 |
|------|---------------------|---------------------------|------|
| ADE | 15.359 | 34.911 | **2.27×** |
| FDE | 31.514 | 69.133 | **2.19×** |

这不是"接近但差一点"，而是**2 倍以上的系统性落后**。这个量级本身就说明问题不在调参层面——它指向更深层的架构或方法论问题。

### 2.2 最关键的证据：创新模块几乎不起作用

基于 Stage 47 checkpoint 的四个单变量消融 (`val + 20`)：

| 消融条件 | ADE | FDE | 相对 base 变化 |
|----------|-----|-----|---------------|
| base (全部开启) | 51.607 | 103.955 | — |
| `disable_queue_rollout` | 51.605 | 103.947 | 几乎不变 |
| `disable_decoder_state_residual` | 51.693 | 104.259 | 几乎不变 |
| `disable_state_gating` | 51.649 | 104.078 | 几乎不变 |
| `disable_lane_queue_anchor` | 51.607 | 103.955 | 完全不变 |

**把四个核心机制全部关掉 vs 全部打开，best-of-20 指标几乎完全一样。** 这不是某个开关接错了——所有开关都确认能改变单 batch 前向输出，但在 best-of-20 聚合后，差异淹没在采样方差里。

配合这个证据链：

- `val + num_samples=1` 上，base 与 `all_off` 也几乎重合 (`59.349 / 121.138` vs `59.410 / 121.388`)
- decoder step residual 平均范数约 `0.0278`
- queue hidden 平均范数约 `2.1306`
- **`step_residual / queue_hidden ≈ 1.3%`**

结论很明确：**meso/macro memory 分支在"物理上存在"（前向传播能跑通），但在"功能上不存在"（对最终预测的数值影响可以忽略）。**

### 2.3 根因一：架构耦合路径过窄

当前 meso/macro 信息进入 decoder 的唯一物理路径是：

```
state_residual = MLP([light_embedding, queue_context, cycle_context])
state_residual *= gate([...])   # sigmoid gating
pred_lstm_hidden = pred_lstm_hidden + state_residual   # 初始化时一次
pred_lstm_hidden = pred_lstm_hidden + state_residual   # 每步一次（训练+推理）
```

问题出在四个层面：

1. **数值淹没**：`pred_lstm_hidden`（32-64 维）承载了 trajectory LSTM、graph interaction、scene noise 的全部信息。一个由 queue/cycle context 经过 2 层 MLP 生成的残差向量，在初始化后就天然地会被 trajectory path 的持续更新覆盖。1.3% 的范数比不是意外，是这种设计的必然结果。

2. **加性残差本身就是弱耦合**：decoder LSTM 在每一步接收 `[input_offset, pred_lstm_hidden]`，然后更新 hidden state。加性残差只在**输入侧**做了一次修正，LSTM 内部的 forget/input/output 门完全由 trajectory hidden 自身的动力学驱动。下一时刻的 hidden state 没有任何机制保留 state residual 的贡献。

3. **初始化修正被后续更新洗掉**：即使初始化时 `state_residual` 成功在 `pred_lstm_hidden` 中留下印记，经过 12 步（pred_len=12）的自回归更新后，LSTM 内部的 gating 机制对初始化方向的记忆呈指数衰减。

4. **state gating 是训练出的，但训练信号太弱**：gate 由 sigmoid 输出，训练早期接近 0.5。当 state residual 本身对 loss 的贡献微小时，gate 的梯度也极小，gate 很快收敛到接近 0——这就是"state branch 在动但 decoder 不理会它"的动力学解释。

**这不是一个调参问题，是信息注入的架构设计问题。** 加性残差到 hidden state 这条路径，在当前容量和数据量下，可能从根本上就不足以让交通状态信息对轨迹解码产生实质性约束。

### 2.4 根因二：aux loss 可能在帮倒忙

当前 aux 监督设计：

| aux 头 | 监督目标 | 维度 |
|--------|---------|------|
| `queue_aux_reg_head` | 排队车辆数、等待比例、释放比例、排队长度 | 4 |
| `queue_aux_cls_head` | 停止线占用、队首标记 | 2 |
| `cycle_aux_phase_head` | 当前相位类别（红/绿/黄） | 3 |
| `cycle_aux_time_head` | elapsed time、remaining time | 2 |
| `cycle_aux_change_head` | 相位切换二分类 | 1 |

这些 aux 任务的梯度反向传播到 queue/cycle LSTM 中，塑形它们的 hidden representation。问题在于：

- **"擅长预测排队长度"和"提供对轨迹预测有用的表征"可能是正交甚至冲突的目标。** 如果 queue LSTM 被 aux loss 驱动去学习一个精确的排队长度回归器，它的 hidden state 可能编码了大量在 aux 维度上准确、但在轨迹维度上无关的信息。

- 当这个 hidden state 再通过一个**微弱的残差路径**（见 2.3）注入 decoder 时，decoder 完全可以学会忽略它——甚至**主动学会忽略它**，因为来自 aux-driven state 的信号对轨迹 loss 没有帮助。

- aux loss 的权重 (`aux_rollout_weight=2.5`) 当前是手动设定的，没有经过针对轨迹指标的校准。如果 aux loss 占主导，state branch 就会往"做好 aux 预测"的方向优化，而这个方向可能与"帮助轨迹预测"无关。

**换句话说：你可能花了大量计算在训练一个"精密的排队检测器"，然后期望它顺便帮助轨迹预测——但它没有。**

### 2.5 根因三：信号灯信息可能已被 baseline 充分捕获

一个需要认真面对的方法论问题：

- Baseline 已经有 `light_state_embedding`，从 `light_state`（停止线坐标、灯态、持续时间等 5 维）中提取特征，**直接拼接**到 decoder 初始化向量中。
- Baseline 也使用 `pred_state`（未来真实信号状态），通过 `get_next_state` 在预测期每步更新灯态信息。

一个必须考虑的可能性是：**对于"在信号灯路口预测轨迹"这个任务，baseline 的 light_state_embedding + get_next_state 已经捕获了信号灯的主要影响。**

你在此基础上增加的：
- queue rollout（排队波/释放波动力学）
- cycle rollout（宏观周期状态演化）
- state gating（相位条件调制）
- lane queue anchor（车道级共识）

这些都在试图建模更细粒度的**二阶交通流效应**。但：

1. 这些二阶效应在当前数据量和模型容量下可能无法稳定学习
2. 即使学到了，对轨迹预测的**边际贡献**可能本身就很小
3. 数据集中交叉口场景的变异性可能不足以让这些精细机制产生可泛化的信号

### 2.6 综合判断

当前的问题**不是调参不够，不是 warmup 长度不对，不是 teacher forcing ratio 没选好，不是 continuation 的 lr 没设对**——那些是 5-10% 级别的优化问题，而你的差距是 100%+。

核心矛盾可以总结为一句话：

> **你设计了一套理论上合理的交通状态记忆机制，但它与轨迹解码器之间的信息通道在架构上就不足以传递有效信号。四个消融开关全部无效这一事实，已经明确告诉你：创新模块虽然在"物理上存在"（前向传播能跑通），但在"功能上不存在"（对输出没有实质影响）。**

诊断优先级排序：

| 优先级 | 问题 | 严重程度 | 可操作性 |
|--------|------|---------|---------|
| **P0** | 架构耦合路径过窄（1.3% 范数比） | 致命 | 需要重新设计注入方式 |
| **P1** | aux loss 可能使 state branch 学到轨迹无关表征 | 严重 | 可通过对比实验验证 |
| **P2** | 二阶交通流效应对轨迹预测的边际贡献存疑 | 根本性 | 需要决定性实验判断路线价值 |
| P3 | warmup 长程稳定性 (G3) | 中等 | 耦合问题解决后再看 |
| P4 | meso/macro 分支容量偏小 (G4) | 低 | 耦合问题解决后才有意义 |

---

## 3. 当前证据快照

| 项目 | Split | num_samples | ADE | FDE | 说明 |
|------|-------|-------------|-----|-----|------|
| baseline | val | 4 | 38.493 | 78.706 | 当前仓库基线 |
| baseline | test | 4 | 17.812 | 37.568 | 当前仓库基线 |
| baseline | val | 20 | 35.022 | 70.658 | comparable 硬基线 |
| baseline | test | 20 | 15.359 | 31.514 | comparable 硬基线 |
| CycleState old best under new rollout code | val | 20 | 74.947 | 154.411 | `warmup50_refine50_p0_seqgat_relation_v1` |
| CycleState old best under new rollout code | test | 20 | 43.736 | 85.691 | `warmup50_refine50_p0_seqgat_relation_v1` |
| CycleState short refine after rollout | val | 20 | 51.607 | 103.955 | `..._cycle_rollout_refine1/model_best` |
| CycleState short refine after rollout | test | 20 | 34.911 | 69.133 | `..._cycle_rollout_refine1/model_best` |
| CycleState best candidate | val | 4 | 84.772 | 170.878 | quick 口径 |

当前判断：

- 预测期 `cycle hidden rollout` 直接接到旧 checkpoint 上时，只带来很小的推理侧改善。
- 但在旧 best 上做最小增量 refine 后，`val/test + 20` 都有实质下降，说明 `macro rollout` 方向成立，只是当前强度还不足以追平 baseline。
- 更保守的 low-lr continuation 已经判负。
- **综合诊断结论（见第 2 节）：目前最该做的不再是继续扫 continuation 超参或堆新结构，而是从根本上重新设计 state-to-decoder 的耦合方式，并在架构改动前先用决定性实验验证"交通状态信息本身是否足以改善轨迹预测"。**

---

## 4. 决定性实验（新增，最高优先级）

> **目标**: 在改动架构之前，先用最少变量判断"交通状态 → 轨迹预测"这条因果链本身是否成立，以及当前架构瓶颈到底在耦合方式还是在信息本身。

### 4.1 DE-1: Oracle State 直注实验

**问题**: 如果连**真实交通状态**直接注入都不能显著改善轨迹预测，那么学习这些状态的整个 CycleState 假设就需要重新审视。

**方案**:

1. 从数据集中直接构造每步的"真实 queue/cycle state"：
   - 排队车辆数（从轨迹和停止线位置直接计算）
   - 等待/释放比例（从速度阈值判断）
   - 当前相位和持续时间（`pred_state` 中已有）
   - 到停止线距离（可直接计算）
2. 将这些真实值（不做任何学习）直接作为 decoder LSTM 的**额外输入**（与 trajectory offset 拼接），等价于 bypass 整个 queue/cycle LSTM 分支。
3. 训练一个最小版本的"oracle-injected decoder"，与 baseline 做 comparable 对比。

**期望**:

- 如果 oracle state 直注后，指标从 `34.911 / 69.133` 大幅下降到接近 baseline 的 `15.359 / 31.514` → **交通状态信息确实有价值，问题出在当前的学习/耦合方式上。**
- 如果 oracle state 直注后，指标改善有限（比如只到 `28 / 55`）→ **即使知道真实排队/相位信息，对轨迹预测的边际贡献也有限，CycleState 的核心假设需要重新评估。**

**工作量估计**: 1-2 天（主要是构造 oracle features + 修改 decoder 输入层）

**完成标志**: 产出 oracle-injected vs baseline vs CycleState 的三方 comparable 对比。

### 4.2 DE-2: 极端耦合压力测试

**问题**: 当前 state residual 范数只有 queue hidden 的 1.3%。如果通过极端手段把耦合强度拉满，能否看到实质改善？

**方案**:

1. 不改架构，但做以下极端修改：
   - `queue_lstm_hidden_size`: 32 → 128
   - `cycle_lstm_hidden_size`: 16 → 64
   - `decoder_state_residual_scale`: 1.0 → 10.0（训练阶段就设大，不是只在推理时放大）
   - 去掉所有 aux loss（`aux_rollout_weight=0`），让 state branch 纯粹靠 trajectory loss 驱动
2. 从头训练 50 epoch（warmup 阶段），观察 state branch 是否能在"强耦合 + 无 aux 干扰"的条件下学到对轨迹有用的表征。

**期望**:

- 如果极端耦合后，test 指标仍远不如 baseline → **问题不在耦合强度和容量，而在架构本身的路径设计。**
- 如果极端耦合后，test 指标接近或超过 baseline → **说明耦合强度是核心瓶颈，可以在此基础上做更优雅的架构重设计。**

**工作量估计**: 1-2 天（主要是改配置 + 训练 + 评估）

**完成标志**: 产出极端耦合版 vs baseline 的 comparable 对比。

### 4.3 DE-3: 最简可行版本（Minimum Viable CycleState）

**问题**: 是否当前复杂的 rollout + gating + anchor 机制本身就是问题的来源？

**方案**:

1. 创建一个**最简版 CycleState**：
   - 只保留观测期最后时刻的 `queue_hidden` 和 `cycle_hidden`
   - **去掉** queue rollout
   - **去掉** cycle rollout
   - **去掉** state gating
   - **去掉** lane queue anchor
   - **去掉** decoder state residual 的加性注入方式
   - **改为**：将 `[queue_last, cycle_last]` 直接拼接到 decoder 初始化向量 `encoded_before_noise_hidden` 后面（与 light_state_embedding 同级）
2. 从头训练，做 comparable 对比。

**期望**:

- 如果最简版不能靠近 baseline → rollout/gating/anchor 这些复杂机制是在一个不成立的基础上叠床架屋。
- 如果最简版接近甚至超过 baseline → 可以在最简版基础上，逐个加回 rollout/gating/anchor，每次验证边际收益。

**工作量估计**: 0.5-1 天

**完成标志**: 产出最简版 vs baseline 的 comparable 对比。

---

## 5. 活跃 backlog（重新排序）

> **说明**: 以下 backlog 已根据第 2 节诊断重新排序。P0 级（决定性实验）必须在任何架构改动或继续训练之前完成。

### Phase 0: 决定性实验（最高优先级）

见第 4 节。执行顺序：**DE-3（最简版）→ DE-1（Oracle 直注）→ DE-2（极端耦合）**。

DE-3 排在最前，因为它工作量最小、变量最少，能最快判断"把 queue/cycle state 直接拼进 decoder 初始化"是否本身就是一个比"加性残差"更有效的耦合方式。如果 DE-3 已经给出正向信号，DE-1 和 DE-2 的优先级可以下调。

### Phase 0.5: Oracle 假设量化验证（保留，但在 DE-1 之后执行）

**问题**

CycleState 与 baseline 都使用未来真实信号 `pred_state`，当前 comparable 在这个假设上是对齐的，但其性能贡献还没有被量化。更重要的是，`pred_state` 不只进入 cycle 路径，也进入 `light_state_embedding`、`get_next_state` 和 queue rollout 路径；如果不先拆清楚这部分贡献，后续对 G3 / G6 / G4 / G5 的判断容易跑偏。

**关注文件**

- `README.md`
- `docs/PLAN.md`
- `D2TP/models.py`
- `D2TP/evaluate_model.py`

**执行原则**

- DE-1 中的 oracle state 直注实验，与 Phase 0.5 的"量化 oracle 贡献"高度重合。DE-1 完成后，Phase 0.5 的大部分问题应该已有答案。
- 如果 DE-1 显示 oracle state 的边际贡献有限，Phase 0.5 的后续步骤（噪声注入等）可以取消。
- 如果 DE-1 显示 oracle state 贡献显著，再按原计划做更细粒度的 `cycle_step_embedding` / `decode_cycle_gate` 消融。

先做**不引入新模块**的量化，再决定是否需要真正实现 predicted-signal / predicted-cycle 路径。`predicted` 不是一个现成开关，不能把"新增一个 predicted-cycle evaluator"误写成低成本动作。

**执行步骤**

1. 明确当前 `train oracle -> eval oracle` 上界，作为后续所有退化实验的参考点。
2. 做不引入新模块的敏感度实验，优先量化：
   - `cycle_step_embedding` 注入强度
   - `decode_cycle_gate` / cycle-conditioned path 的影响
   - 必要时再做轻量噪声注入，观察 `pred_state` 扰动对 ADE/FDE 的敏感度
3. 只有在上述证据显示 cycle-path oracle 贡献足够大时，才立项实现真正的 `predicted signal / predicted cycle` 评估路径。

**完成标志**

1. 明确 `oracle -> oracle` 的参考结果。
2. 对 cycle-path oracle 依赖给出可比较的定量证据，而不是只停留在文档提醒。
3. 再决定是否需要真实的 `train oracle -> eval predicted` 实验，而不是默认把它当成 Phase 0.5 的第一步。

### 架构重设计（取决于决定性实验结果）

> **前提**: DE-1/DE-2/DE-3 至少有一个给出明确正向信号。

如果决定性实验证明"交通状态信息确实有价值，问题出在耦合方式"，则按以下优先级推进架构改动：

#### AR-1: 替换加性残差为直接条件注入（最高优先级）

**当前设计 (弱耦合)**:
```
state_residual = gate * MLP([light, queue, cycle])     # 范数 ~0.028
pred_lstm_hidden = pred_lstm_hidden + state_residual   # 加性修正，范数比 1.3%
pred_lstm_hidden, pred_lstm_cell = LSTM(input, (pred_lstm_hidden, pred_lstm_cell))
output = MLP(pred_lstm_hidden)
```

**改为 (强耦合)**:
```
state_context = MLP([light, queue, cycle])              # 独立于 decoder hidden
lstm_input = concat([trajectory_offset, state_context]) # 直接拼接作为 LSTM 输入
pred_lstm_hidden, pred_lstm_cell = LSTM(lstm_input, (pred_lstm_hidden, pred_lstm_cell))
output = MLP([pred_lstm_hidden, state_context])         # state context 也进入输出层
```

关键改动：
1. state context **不再通过残差修正 hidden state**，而是作为 LSTM 每步的**显式输入**。
2. state context 同时进入输出投影层，进一步强化其对最终预测的直接影响。
3. 这等价于告诉模型："交通状态信息与你自己的轨迹历史同等重要"——而不是"顺便修正一下你的内部状态"。

#### AR-2: 乘法门控替代加性残差（备选）

如果 AR-1 不可行（比如破坏了 baseline 的 warm-start 逻辑），备选是乘法门控：

```
state_gate = sigmoid(MLP([pred_lstm_hidden, queue_context, cycle_context]))
pred_lstm_hidden = pred_lstm_hidden * state_gate   # 逐元素乘法，而不是加法
```

乘法耦合天然比加法耦合更强——它不只是"修正"，而是"重新缩放"。每个 hidden 维度都可以被 state 信息独立调制。

#### AR-3: aux loss 重新设计

当前 aux 监督的是"排队长度/相位"等可能与轨迹无关的中间量。改为：

1. **轨迹相关 aux**：预测每步的"到停止线距离变化量"、"是否处于等待状态（速度 < 阈值）"——这些 aux 与轨迹预测直接相关。
2. **对比实验**：`aux_rollout_weight=0` vs `aux_rollout_weight=2.5` 的 full comparable 对比，验证 aux loss 到底是正向还是负向贡献。
3. **渐进式 aux**：warmup 阶段 aux weight 很低（让 state branch 先学会帮轨迹），refine 阶段再逐渐增大 aux weight（让 state branch 在不破坏轨迹性能的前提下学习可解释表征）。

### G6: cycle memory 在预测期退化（已完成最小方案）

**问题**

预测期当前主要依赖单步 `cycle feature -> cycle_step_embedding` 路径，缺少与 queue rollout 对称的 cycle-state rollout。观测期的 `cycle_lstm_model` 会累计 `cycle_last`，但解码期每步消费的主要是单帧 `cycle_step_embedding`，不是 rollout 的 cycle hidden。

**关注文件**

- `D2TP/models.py`
- `tests/test_cyclestate_protocol.py`

**执行原则**

先做最小可验证改动，确认"有没有预测期 cycle memory"本身是否重要；不要一上来就把完整的显式 cycle dynamics 和 predicted-signal 路径同时做掉。

**当前状态**

- **最小方案已落地**：预测期 decoder loop 内已引入 cycle hidden/cell 的逐步更新，decoder 读到 rollout 后的 cycle hidden。
- **验证方案已通过**：定向测试证明 cycle hidden 在预测期跨步变化、且确实参与 decoder 条件输入。
- **但消融结果显示这个改动对最终指标几乎无影响**（见第 2.2 节），说明问题不在"有没有 cycle rollout"，而在"cycle rollout 的结果如何影响 decoder 输出"。
- **下一步**: G6 的完整方案（显式 feature rollout）暂停，先等决定性实验和架构重设计的结果。

**完成标志**

1. ✅ 预测期 cycle 状态不再只是单帧静态条件。
2. ✅ 能通过定向测试证明 cycle-state 在预测期按时间推进。
3. ⏸️ 能说明这一改动是否实质影响当前 best candidate 的评估结果 → **已确认：不影响，根因在耦合路径，不在 cycle rollout 是否存在。**

### G3: warmup 长程稳定性仍未根治

**现状**

- Stage 24: `100b` 相对 `50b` 恶化 `165.7%`
- Stage 25: 缓和到 `130.1%`
- Stage 26: 单独降 `teacher_forcing_ratio` 到 `0.6` 更差

**判断（根据第 2 节诊断更新）**

问题更像模块耦合与阶段职责划分错误，而不是单一超参能修好。但更重要的是：**在决定性实验和架构重设计完成之前，G3 不应再作为独立线索推进**——因为如果 state-to-decoder 耦合方式被根本改变，当前所有 warmup 稳定性结论都可能不再适用。

**诊断顺序（在架构重设计完成后）**

1. 在新架构基础上，优先做**诊断式消融**：
   - `disable_queue_rollout`
   - `disable_decoder_state_residual`
   - `disable_state_gating`
   - `disable_lane_queue_anchor`
2. 如果某个模块一关掉就明显缓解长程退化，优先回到该模块路径定位，而不是继续扫 warmup 超参。
3. 如果消融仍无法定位，再讨论：
   - warmup 是否应更短
   - refine 是否应更早接管
   - `rollout_residual_scale` 等稳定化参数是否需要小范围 sweep

**完成标志**

1. 至少定位出一个最可能的退化主因，而不是继续盲调。
2. 保留 `100b ADE <= 50b ADE * 1.15` 作为稳定性 KPI，但不把它当成 G3 唯一出口。
3. 如果证据显示 warmup 本身不应承担长链状态耦合学习，则明确收口为更短 warmup + 更早 refine。

### G4: meso / macro 分支容量偏小（前提：DE-2 给出正向信号）

**问题**

`queue LSTM hidden=32`、`cycle LSTM hidden=16` 仍偏保守，可能限制状态表征能力。

**前提**

- 在决定性实验（特别是 DE-2）给出正向信号后再做。
- 如果 DE-2 显示极端增容也不能改善指标，G4 可以直接关闭。

**完成标志**

- 结构增容后仍保持协议可解释。
- 变更能通过单变量消融与 comparable 结果说明价值。

### G5: decoder residual 注入位置单一（前提：架构重设计完成后）

**问题**

当前状态残差主要注入 `pred_lstm_hidden`，还没验证是否应扩展到 cell 或输出侧。

**前提**

- 架构重设计（AR-1/AR-2）完成后，G5 的大部分问题应该已被覆盖。
- 如果 AR-1 已将 state context 同时注入 LSTM input 和 output projection，G5 自动解决。

**完成标志**

- 有明确的注入设计与对应消融。
- 新设计不会破坏 baseline-compatible decoder 的 warm-start 逻辑。

### G7: 消融设计仍不够分离

**问题**

当前已有 4 个二元开关，但使用方式更像"候选通过后再证明模块价值"，还没有把它们当成诊断工具。

**当前状态（根据第 2.2 节更新）**

四个开关已经完成了一轮消融，结果非常清楚：**四个核心机制全部关掉后指标几乎不变。** 这已经提供了极强的诊断信号——问题不在某个具体模块的实现细节，而在所有模块共享的底层耦合方式。

**推荐顺序（在架构重设计后复用）**

1. `disable_queue_rollout`
2. `disable_decoder_state_residual`
3. `disable_state_gating`
4. `disable_lane_queue_anchor`

**完成标志**

- ✅ 已完成一轮围绕当前最佳候选的单变量消融，关键结论已写入第 2 节。
- 在架构重设计后，复用同一套消融来验证新架构的每个模块是否真正提供了边际贡献。

### G8: 实验推进纪律

**问题**

仓库已经有 `smoke / protocol-check / comparable` 三级标签，但执行上仍容易过早推进新结构。

**当前状态（根据第 2 节诊断更新）**

G8 比之前更加重要。诊断显示过去多轮实验（G6 最小方案、continuation、推理侧 scale 扫描）都在一个根本性的架构问题被定位之前就推进了——结果是大量实验在"优化一个不工作的系统"。

**新增强制规则**

1. 任何新训练或架构改动之前，必须先确认"这个改动解决了第 2 节诊断中的哪个具体问题"。
2. 如果改动不涉及 state-to-decoder 耦合路径的实质变化，默认预期是"对指标无影响"（如第 2.2 节的所有消融）。
3. 每轮实验后更新本文档的第 2 节诊断状态（确认/推翻/修正某个根因假设）。

**完成标志**

- 每条新结果都带 split、`num_samples`、checkpoint、标签。
- 任何"超过 baseline"的说法都有完整 comparable 证据。
- 第 2 节诊断随着实验证据的积累而动态更新，而不是写成一次性的静态分析。

---

## 6. 固定执行顺序（重新编排）

### Phase A: 必跑可比线（保持不变）

1. baseline `val + num_samples=20`
2. baseline `test + num_samples=20`
3. current best candidate `test + num_samples=20`
4. `..._cycle_rollout_refine1/model_best` 的 `val/test + num_samples=20`

建议命令：

```bash
python D2TP/evaluate_model.py --model_type d2tpred --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --num_samples 20 --eval_print_every 10 --resume D2TP/model_best.pth.tar --dset_type val
python D2TP/evaluate_model.py --model_type d2tpred --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --num_samples 20 --eval_print_every 10 --resume D2TP/model_best.pth.tar --dset_type test
python D2TP/evaluate_model.py --model_type cyclestate --device cuda --pin_memory --loader_num_workers 0 --batch_size 8 --num_samples 20 --eval_print_every 10 --resume experiments/cyclestate/warmup50_refine50_p0_seqgat_relation_v1/checkpoint/model_best.pth.tar --dset_type test --rollout_residual_scale 0.7
python D2TP/evaluate_model.py --model_type cyclestate --device cuda --pin_memory --loader_num_workers 0 --batch_size 8 --num_samples 20 --eval_print_every 10 --resume experiments/cyclestate/warmup50_refine50_p0_seqgat_relation_v1_cycle_rollout_refine1/checkpoint/model_best.pth.tar --dset_type test --rollout_residual_scale 0.7
```

### Phase B: 决定性实验（新增，最高优先级）

按顺序执行：

1. **DE-3**: 最简可行版本（0.5-1 天）
   - 只保留 queue/cycle last 直接拼接进 decoder 初始化
   - 去掉所有 rollout/gating/anchor/residual 机制
   - 结果会直接告诉你：把 state 直接拼进 decoder 这条更简单的路径本身是否成立

2. **DE-1**: Oracle State 直注实验（1-2 天，仅在 DE-3 不足以解释问题时执行）
   - 用真实交通状态直接注入 decoder
   - 结果会告诉你："交通状态 → 轨迹"这条因果链本身强度如何

3. **DE-2**: 极端耦合压力测试（1-2 天，仅在 DE-1 给出正向信号后执行）
   - 极端增容 + 极端 scale + 去掉 aux loss
   - 结果会告诉你：在足够强的耦合下，state branch 能否自发学到有用表征

### Phase C: 根据决定性实验结果分支

#### 分支 C1: 决定性实验全部正向

→ 推进架构重设计 AR-1（直接条件注入）或 AR-2（乘法门控）
→ 在新架构上重新训练
→ 复用 G7 消融验证每个模块的边际贡献
→ 更新第 2 节诊断状态

#### 分支 C2: DE-1 显示 oracle state 边际贡献有限

→ 整个 CycleState 的核心假设需要重新审视
→ 优先回到问题定义层面：信号灯路口的轨迹预测，是否真的需要一个显式的交通状态记忆？
→ 可能的替代方向：
  - 更强的 trajectory-level modeling（而不是 state-level）
  - 更细粒度的交互建模（vehicle-vehicle 而非 vehicle-infrastructure）
  - 更好的不确定性建模（当前 best-of-K 采样本身就能提供多样性）

#### 分支 C3: DE-1 正向但 DE-2 负向

→ 问题不在耦合强度，而在"学习 vs 使用"的分离
→ 重点转向 AR-3：aux loss 重新设计
→ 先跑 `aux_rollout_weight=0` 的 full comparable 对比

### Phase D: 诊断式消融（在架构重设计后复用）

如果候选在架构重设计后站得住，复用诊断式消融结果，把重点从"定位问题"升级成"确认主故事中的有效模块"。

优先补强这些方向：

1. `disable_queue_rollout`
2. `disable_decoder_state_residual`
3. `disable_state_gating`
4. `disable_lane_queue_anchor`

此时消融目标从"定位问题"升级成"确认主故事中的有效模块"，并按需要追加 `test` 复核。

### Phase E: 如果架构重设计后候选仍站不住

优先回到问题定义层面，不加新结构：

1. 承认当前 `micro / meso / macro` 三层架构在当前数据量和容量下可能不成立
2. 重新审视 baseline 本身的改进空间（更强的交互建模、更好的不确定性估计）
3. 或者转向更轻量的"trajectory-conditioned-on-signal"而不是"full-cycle traffic state memory"

---

## 7. Comparable 门槛

正式比较必须同时满足：

1. split 明确
2. `num_samples` 明确
3. checkpoint 来源明确
4. `evaluate_model.py` 口径一致
5. 不用 `test` 做模型选择

额外稳定性门槛：

- 若继续讨论 warmup 长程训练，`100b` 相对 `50b` 的 ADE 恶化不得超过 `15%`
- 但该指标是稳定性 KPI，不代替"先定位主因"的诊断工作

---

## 8. 默认取舍

- **当前默认优先完成决定性实验（第 4 节），而不是继续推进任何新结构或训练。**
- 当前默认继续把 `README.md` 当项目入口，把 `EXPERIMENT_LOG.md` 当证据入口，把本文件当活跃待办入口。
- 任何影响研究叙事、协议口径或当前结论的改动，都应同步更新：
  - `README.md`
  - `EXPERIMENT_LOG.md`
  - `docs/PLAN.md`
- 第 2 节诊断不是一次性文档——每轮决定性实验完成后，应更新对应根因的确认/推翻状态。
- 在决定性实验给出明确结论之前：
  - **不再**做任何 continuation/refine 超参扫描
  - **不再**做推理侧 scale 扫描
  - **不再**新增任何 state 分支机制
  - **不再**在未改变耦合方式的前提下堆结构

---

## 9. 风险提醒

- oracle 假设若不量化，论文叙事会有明显缺口；但如果把 `predicted cycle` 误当成低成本评估开关，也会把 Phase 0.5 做成一个隐形大功能。
- **若在决定性实验（第 4 节）未完成前继续推进架构改动或训练，容易再次得到"结构改了但无法归因"的结果——这是过去多轮实验重复出现的问题。**
- 若 baseline `num_samples=20` 不补齐，后续任何"更接近论文口径"的比较都会站不稳。
- **最大的方法论风险**：整个 CycleState 的核心假设——"显式的交通状态记忆能改善信号灯路口轨迹预测"——可能在当前数据/容量约束下不成立。决定性实验 DE-1 和 DE-3 将直接面对这个风险，而不是继续绕开它。
- **次大风险**：即使决定性实验正向，架构重设计（AR-1）也可能引入新的训练不稳定或收敛问题，需要预留足够的调试时间。
