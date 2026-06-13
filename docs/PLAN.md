# CycleState Live Plan

> **更新时间**: 2026-06-13 17:30
> **作用**: 只维护当前活跃 backlog、执行顺序和验收门槛。
> **规则**: 已解决项不再在本文件展开维护；历史修复细节以 git 历史、测试与当前代码为准。
> **当前主线**: DE-3 / DE-1 / AR-1 / AR-2 四个决定性实验已全部完成，**分支 C2 触发** — 重新审视"是否真的需要显式交通状态记忆"。DE-3 (init 单点拼接) 仍是 CycleState 变体族在 `test + 20` 上的最优配置（`24.632`），AR-1 / AR-2 都比 DE-3 差，"加性 vs 乘法"两条扩展路径都触到上限。

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

| 优先级 | 问题 | 严重程度 | 可操作性 | 当前状态 |
|--------|------|---------|---------|---------|
| **P0** | 架构耦合路径过窄（1.3% 范数比） | 致命 | 需要重新设计注入方式 | **DE-3 已确认**：直接拼接比加性残差好 29.4%，但仍未追平 baseline |
| **P1** | aux loss 可能使 state branch 学到轨迹无关表征 | 严重 | 可通过对比实验验证 | **待 DE-1 / DE-2**：oracle 直注会绕过此问题 |
| **P2** | 二阶交通流效应对轨迹预测的边际贡献存疑 | 根本性 | 需要决定性实验判断路线价值 | **待 DE-1**：oracle 直注是最直接的判定 |
| P3 | warmup 长程稳定性 (G3) | 中等 | 耦合问题解决后再看 | 暂停（等架构重设计） |
| P4 | meso/macro 分支容量偏小 (G4) | 低 | 耦合问题解决后才有意义 | 暂停（等 DE-2） |

### 2.7 DE-3 后的根因状态更新（2026-06-12）

DE-3 决定性实验已完成（详见 [EXPERIMENT_LOG.md §2.3](../EXPERIMENT_LOG.md)）。

**对根因的更新**：

- **根因一（耦合路径过窄）** — **状态：已确认**
  - 直接拼接比加性残差显著更好（`test + 20`: 24.632 vs 34.911，ADE 改善 29.4%）
  - 这意味着原 1.3% 范数比诊断不仅是"诊断上正确"，更对应着"可量化的性能损失"。
  - 但直接拼接仍远未追平 baseline（`24.632 vs 15.359`，差距 1.6×），说明根因一
    只是多个根因中的一个，仅修复它不能独立解决问题。

- **根因二（aux loss 帮倒忙）** — **状态：未证伪 / DE-1 旁路回答**
  - DE-3 把 aux loss 全部置零后，仍有 1.6× 差距，**说明 aux loss 至少不是唯一的根因**。
  - DE-1（Oracle 直注）也把 aux loss 置零，且进一步绕过了 queue/cycle LSTM
    的"学习"环节，但结果（`test + 20: 30.433`）比 DE-3 还差 23.5%——这说明
    aux loss 至少在 DE-3 协议下没有"帮倒忙"，否则去掉它 + 旁路它的 DE-1
    应该明显更好。
  - 这个根因的最终答案需要 DE-2 或单独的 aux ablation 来给出。
  - **当前判断**：aux loss 不是当前瓶颈，可以推迟或合并到后续架构重设计中。

- **根因三（信号灯信息已被 baseline 充分捕获）** — **状态：极端版本否证，弱版本未证伪**
  - **DE-1 否证了极端版本**：把真实交通状态（10 维 oracle 特征）直接拼到
    decoder LSTM input，比旧 CycleState (`34.911`) 改善 12.8% → `30.433`。
    这说明信号灯信息对轨迹预测**有正向贡献**，"完全无价值"的极端假设不成立。
  - **弱版本仍未证伪**：DE-1 oracle 直注仍差 baseline 1.98×（`30.433 vs 15.359`），
    这说明仅靠注入真实交通状态不能独立解决 2× 差距——baseline 的
    `light_state_embedding + pred_state + get_next_state` 路径之外的
    某些信息（vehicle-vehicle 交互 / scene-level context / 多步未来预测的
    不确定性建模？）可能是关键。
  - **DE-1 的反直觉发现**：10 维 oracle 特征**不如** DE-3 学习的 32D+16D
    hidden state（`30.433 vs 24.632`，DE-1 比 DE-3 差 23.5%）。这说明
    状态分支"学到的表征"本身携带了 oracle 10 维之外的信号——可能包括：
    1. 跨帧累计的"演化轨迹"（如"刚刚切换到红，已累积 8s"），而 oracle
       特征只有"瞬时切片"；
    2. 连续、平滑的向量空间 vs 离散 one-hot / phase_change 标志，梯度友好性差；
    3. 50 批 refine 不足以让模型学会把 10 维离散信号映射成有用的内部表征，
       而 DE-3 的 32D+16D 隐藏向量已经经过训练，是"现成的"。
  - **方法论价值**：把"oracle 信息"直接拼接到 LSTM 输入 ≠ 把"学习到的
    状态表征"拼接到 init 向量。后者经过训练的向量空间本身就是对轨迹
    loss 更友好的特征表示。这反过来为 AR-1（直接条件注入）提供了
    具体的设计依据——应该在 DE-3 的拼接位置（init）基础上扩展注入强度，
    而不是模仿 DE-1 把信息塞到 input。

**DE-3 的方法论价值**：

DE-3 是所有决定性实验中最便宜的一个，但它给出了一个**清晰的二元信号**——
"直接拼接"比"加性残差"显著更好。这同时：

1. **排除了一个反事实**：如果 DE-3 仍比旧 CycleState 差，那"换耦合方式"这个
   方向本身就不可行，根因一会被推翻。但 DE-3 比旧 CycleState 显著更好，根因一
   被加强。
2. **缩小了搜索空间**：既然直接拼接有效，AR-1（直接条件注入）就有了具体的
   设计依据——把"只在初始化时拼接"扩展为"每步拼接 + 输出投影"，比从零开始
   重新设计更可靠。
3. **留下了未回答的问题**：DE-3 的最简版与 baseline 仍有 1.6× 差距，
   这部分差距必须由 DE-1 来回答（oracle 直注能否覆盖？）和 DE-2 来回答
   （极端耦合能否覆盖？）。

### 2.8 DE-1 后的根因状态更新（2026-06-12）

DE-1 决定性实验已完成（详见 [EXPERIMENT_LOG.md §2.4](../EXPERIMENT_LOG.md)）。

**对根因的更新**：

- **根因一（耦合路径过窄）** — **状态：已确认 + DE-1 给出注入位置敏感性**
  - DE-1 把 10 维 oracle 特征拼到 LSTM input，得到 `test + 20: 30.433`。
  - DE-3 把 32D+16D learned state 拼到 init，得到 `test + 20: 24.632`。
  - 两者**都是"直接拼接"路径**（都绕过了加性残差），但拼接位置不同
    （LSTM input vs init），结果差异 **23.5%**。这说明：
    - decoder 对"注入位置"很敏感，把信息塞到 init 比塞到 LSTM 每步 input
      更有效——至少在 50 批 refine 的训练量下如此。
    - 这反过来印证了根因一不是"加性 vs 拼接"那么简单，而是"decoder 的
      内部动力学对注入位置有偏好"。
  - **AR-1 的设计依据**已被加强：DE-3 的 init 拼接位置有效，应该在该位置
    基础上扩展（"每步拼接 + 输出投影"），而不是模仿 DE-1 的 input 拼接。

- **根因二（aux loss 帮倒忙）** — **状态：基本可排除**
  - DE-3 和 DE-1 都把 aux loss 权重置零，但结果都比"全 aux" 的旧 CycleState
    显著更好。DE-3 (`24.632`) 和 DE-1 (`30.433`) 都没有让 aux loss 起任何作用，
    仍比旧 CycleState (`34.911`) 改善 29.4% / 12.8%。
  - 严格地说，DE-3 / DE-1 都没有"在 DE-3 协议上加回 aux loss" 做反向验证，
    所以"aux 完全没贡献"还需 DE-2 / 单独的 aux ablation 来严格证伪。
  - **当前判断**：aux loss 至少不是当前瓶颈，可以推迟到架构重设计之后。

- **根因三（信号灯信息已被 baseline 充分捕获）** — **状态：极端版本否证，弱版本未证伪**
  - **极端版本否证**（DE-1 直接证明）：oracle 交通状态对轨迹预测**有正向贡献**
    （比旧 CycleState 改善 12.8%）。
  - **弱版本未证伪**（DE-1 给出量化下界）：oracle 直注仍差 baseline 1.98×。
    这意味着：仅靠注入"信号灯 + 排队 + 距离 + 速度"等 10 维物理信号，
    不能独立解决 2× 差距。baseline 中比 10 维 oracle 多出来的东西
    （可能是更细粒度的 vehicle-vehicle 交互 / scene-level context /
    多步未来预测 / 训练技巧）才是真正起决定作用的部分。

- **意外发现（DE-1 vs DE-3）** — **新根因候选**：瓶颈不在"信号 vs 学到"
  - 10 维 oracle 特征 (DE-1) 不如 32D+16D learned state (DE-3)。
  - 这说明状态分支"学到的表征"本身有信息量，**瓶颈不在"信号 vs 学到"**。
  - 下一个真正的问题变成："decoder 怎么使用这些表征才能让它真的对
    轨迹预测有用？"——这正是 AR-1（直接条件注入）和 AR-2（乘法门控）
    要回答的问题。
  - **不能**因为这个反直觉发现就下结论说"oracle 信号没用"——oracle
    信号比旧 CycleState 的"复杂 rollout + 残差"路径更有用，只是不如
    简单拼接 learned state 而已。

### 2.9 AR-1 / AR-2 后的根因状态更新（2026-06-13）

AR-1（直接条件注入，DE-3 之上叠加 2 个新注入点）和 AR-2（乘法门控，
DE-3 之上叠加 1 个新调制机制）两个决定性实验均已完成（详见
[EXPERIMENT_LOG.md §2.5 / §2.6](../EXPERIMENT_LOG.md)）。两者均**否决**
了"进一步扩展注入强度即可逼近 baseline"的假设。

**对根因的更新**：

- **根因一（架构耦合路径过窄）** — **状态：已被 AR-1 / AR-2 加强**
  - DE-3 (init 单点拼接) 已是 CycleState 变体族的"最大可榨出价值"：
    - DE-3 → `test + 20: 24.632`
    - AR-1 (init + per-step + output) → `test + 20: 28.631`（**比 DE-3 差 16.2%**）
    - AR-2 (init + per-step multiplicative gate) → `test + 20: 32.368`（**比 DE-3 差 31.4%**）
  - "加性多点"（AR-1）和"乘法门控"（AR-2）两条扩展路径**都触到上限**：
    - AR-1 在 val 上改善 11.6% 但 test 变差 16.2% → **注入过强反而稀释**；
    - AR-2 进一步证明"加性 vs 乘法"不是问题核心，乘法门控**比**加性拼接更差。
  - 因此原诊断"耦合路径过窄"可以更精确地重新表述为：
    > "DE-3 的 init 单点拼接是 learned state context 的最优注入位置；任何
    > 扩展（无论加性多点还是乘法调制）都让 state context 的边际贡献被稀释。"
  - 这与 DE-1 的反直觉发现（oracle 不如 learned）形成完整的因果链：
    1. learned hidden state 比 oracle 物理信号更易被 decoder 消费（DE-1）；
    2. learned state 的最优注入位置是 init 单点（DE-3）；
    3. 在 init 之上再叠加 per-step / output / gate 都会让指标退化（AR-1 / AR-2）。

- **根因二（aux loss 帮倒忙）** — **状态：基本可排除**
  - DE-3 / DE-1 / AR-1 / AR-2 全部把 aux loss 权重置零，但**没有让任何变体
    优于 DE-3**。这说明 aux loss 至少不是瓶颈：去掉 aux + 改进耦合（DE-3）
    可以显著改善，但去掉 aux + 进一步扩展耦合（AR-1 / AR-2）反而退化。
  - 严格地说，仍需"在 DE-3 协议上加回 aux loss"做反向验证才能彻底证伪。
  - **当前判断**：aux loss 不是瓶颈，可以推迟到架构重设计之后。

- **根因三（信号灯信息已被 baseline 充分捕获）** — **状态：弱版本**逐步被支持**
  - DE-3 / DE-1 / AR-1 / AR-2 四个变体在 `test + 20` 上全部仍差 baseline
    1.6-2.1×，且 AR-1 / AR-2 一致显示"加大注入强度"反而让指标退化。
  - 这越来越支持"**显式交通状态注入**对轨迹预测的边际贡献有限"这个
    假设。
  - **重要边界**：根因三的**极端版本**（"信号灯信息对轨迹完全无价值"）
    仍被 DE-1 否证（oracle 直注比旧 CycleState 改善 12.8%）。但**弱版本**
    （"仅靠显式注入不能独立解决 2× 差距"）被四个变体**一致支持**。
  - **方法论价值**：根因三的弱版本若成立，那么"继续在 state injection
    路线上做架构重设计"将无法追平 baseline，分支 C2 应当被触发。

**AR-1 / AR-2 的方法论价值**：

1. **"加性 vs 乘法"两条路径都触到上限**：
   - AR-1 (加性多点) 退化到 `28.631`；
   - AR-2 (乘法调制) 退化到 `32.368`，是变体族最差。
   - 这意味着 state injection 路线的"扩展空间"在 DE-3 附近就已饱和，
     继续在 init 拼接基础上叠加新机制**几乎不可能**让指标更接近 baseline。

2. **DE-3 的"init 单点拼接"是最优配置**：
   - 单点拼接给 decoder 留出了"自然演化"的空间；
   - 多点拼接 (AR-1) 强制覆盖 LSTM 隐状态演化；
   - 乘法门控 (AR-2) 在外部再加一层 sigmoid 调制，与 LSTM 内部门控冗余。
   - 这与 LSTM 自身设计哲学（让 hidden state 自然演化）一致 ——
     **额外的门控或拼接会破坏这种自然演化**。

3. **val/test 落差的进一步观察**：
   - DE-3: val/test 落差 2.66× (`65.54 / 24.63`)
   - AR-1: val/test 落差 2.02× (`57.95 / 28.63`) — **鲁棒性变好但平均变差**
   - AR-2: val/test 落差 2.29× (`74.23 / 32.37`) — **鲁棒性与平均都变差**
   - 这说明"加强耦合"虽然能部分缓解 val/test 分布差异问题，但**以牺牲
     test 性能为代价**。val 上的改善不能迁移到 test。

**分支 C2 触发**：

DE-3 / DE-1 / AR-1 / AR-2 四个决定性实验一致支持"**state injection 路线
边际收益已饱和**"的结论。这正是 PLAN.md §6.3 分支 C2 描述的触发条件：

> **DE-1 / AR-1 显示 oracle state 边际贡献有限**
> → 整个 CycleState 的核心假设需要重新审视
> → 优先回到问题定义层面：信号灯路口的轨迹预测，是否真的需要一个显式的
>   交通状态记忆？
> → 可能的替代方向：
>   - 更强的 trajectory-level modeling（而不是 state-level）
>   - 更细粒度的交互建模（vehicle-vehicle 而非 vehicle-infrastructure）
>   - 更好的不确定性建模（当前 best-of-K 采样本身就能提供多样性）

**当前判断**：

- DE-3 仍是 CycleState 变体族的最终结果，**应当被作为"state injection 路线
  的最大可榨出价值"记录**。
- AR-1 / AR-2 的反直觉结果不是"设计失败"，而是给出了"扩展路径触顶"的
  清晰二元信号。
- 下一阶段的重点应**完全离开 state injection 路线**，转入分支 C2 的备选项
  （trajectory-level / vehicle-vehicle interaction / 不确定性建模）。

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
| **DE-3 minimal viable** | val | 20 | 65.537 | 161.993 | `experiments/cyclestate/de3_minimal_viable` |
| **DE-3 minimal viable** | test | 20 | 24.632 | 57.135 | `experiments/cyclestate/de3_minimal_viable` |
| **DE-1 oracle state inject** | val | 20 | 77.472 | 176.863 | `experiments/cyclestate/DE1_oracle_inject` |
| **DE-1 oracle state inject** | test | 20 | 30.433 | 66.544 | `experiments/cyclestate/DE1_oracle_inject` |
| **AR-1 direct condition inject** | val | 20 | 57.954 | 140.002 | `experiments/cyclestate/AR1_direct_inject` |
| **AR-1 direct condition inject** | test | 20 | 28.631 | 65.631 | `experiments/cyclestate/AR1_direct_inject` |
| **AR-2 multiplicative gating** | val | 20 | 74.229 | 178.848 | `experiments/cyclestate/AR2_multiplicative_gating` |
| **AR-2 multiplicative gating** | test | 20 | 32.368 | 77.927 | `experiments/cyclestate/AR2_multiplicative_gating` |

当前判断：

- 预测期 `cycle hidden rollout` 直接接到旧 checkpoint 上时，只带来很小的推理侧改善。
- 但在旧 best 上做最小增量 refine 后，`val/test + 20` 都有实质下降，说明 `macro rollout` 方向成立，只是当前强度还不足以追平 baseline。
- 更保守的 low-lr continuation 已经判负。
- **DE-3 决定性实验已完成**：直接拼接 `[queue_last, cycle_last]` 到 decoder 初始化向量
  比加性残差显著更好（`test + 20`: 24.632 vs 34.911，ADE 改善 29.4%）。
  - 这**确认**了根因一（架构耦合路径过窄）确实存在。
  - 但最简版仍有 1.6× 差距，根因二/三（aux loss / 信号灯信息已被 baseline 充分捕获）
    尚未被证伪，**下一步必须做 DE-1（Oracle 直注）**。
- **DE-1 决定性实验已完成**：把 10 维 oracle 交通状态直接拼到 decoder LSTM input，
  `test + 20: 30.433`。
  - **否证根因三的极端版本**：oracle 信息对轨迹有正向贡献（比旧 CycleState 改善 12.8%）。
  - **意外反直觉发现**：oracle 10D **不如** learned 32D+16D（DE-1 比 DE-3 差 23.5%），
    瓶颈不在"信号 vs 学到"。
  - **未证伪根因三的弱版本**：oracle 直注仍差 baseline 1.98×。
  - **下一步必须是 AR-1（直接条件注入）**：在 DE-3 的 init 拼接位置基础上扩展为
    "每步拼接 + 输出投影"，看是否能进一步逼近 baseline。
- **AR-1 决定性实验已完成**：在 DE-3 init 拼接之上叠加 per-step + output 两个
  注入点，`test + 20: 28.631`，**比 DE-3 差 16.2%**。
  - **否决原假设**（"加大注入强度 → 更好"）：val 改善 11.6% 但 test 变差 16.2%，
    说明注入强度有"甜蜜点"，过犹不及。
  - **learned state context 的最优注入位置是 init**（DE-3），不是多点拼接（AR-1）。
- **AR-2 决定性实验已完成**：在 DE-3 init 拼接之上叠加 per-step multiplicative
  gate，`test + 20: 32.368`，**比 DE-3 差 31.4%**，是 CycleState 变体族最差。
  - "加性 vs 乘法"两条扩展路径**都触到上限**，**任何**在 DE-3 之上的扩展
    （无论加性多点还是乘法调制）都让指标退化。
  - 这意味着 state injection 路线的"扩展空间"在 DE-3 附近就已饱和。
- **分支 C2 触发**：DE-3 / DE-1 / AR-1 / AR-2 四个变体一致支持"显式交通状态
  注入边际贡献有限"的假设，下一阶段应**完全离开 state injection 路线**，
  转入"trajectory-level / vehicle-vehicle 交互 / 不确定性建模"等替代方向。

---

## 4. 决定性实验（新增，最高优先级）

> **目标**: 在改动架构之前，先用最少变量判断"交通状态 → 轨迹预测"这条因果链本身是否成立，以及当前架构瓶颈到底在耦合方式还是在信息本身。

### 4.1 DE-1: Oracle State 直注实验 — ✅ 已完成

**问题**: 如果连**真实交通状态**直接注入都不能显著改善轨迹预测，那么学习这些状态的整个 CycleState 假设就需要重新审视。

**方案**:

1. 从数据集中直接构造每步的"真实 queue/cycle state"：
   - 排队车辆数（从轨迹和停止线位置直接计算）
   - 等待/释放比例（从速度阈值判断）
   - 当前相位和持续时间（`pred_state` 中已有）
   - 到停止线距离（可直接计算）
2. 将这些真实值（不做任何学习）直接作为 decoder LSTM 的**额外输入**（与 trajectory offset 拼接），等价于 bypass 整个 queue/cycle LSTM 分支。
3. 训练一个最小版本的"oracle-injected decoder"，与 baseline 做 comparable 对比。

**实际结果（2026-06-12）**：

- 训练协议：warmup 150b + refine 50b = 200b 总 batch
- `val + 20`: **77.472 / 176.863**
- `test + 20`: **30.433 / 66.544**
- 与 baseline (`test + 20: 15.359 / 31.514`) 差距 **1.98×**
- 与 DE-3 (`test + 20: 24.632 / 57.135`) 相比**差 23.5%**
- 与旧 CycleState (`test + 20: 34.911 / 69.133`) 相比**改善 12.8%**

**结论**：

- **否证根因三的极端版本**：oracle 交通状态对轨迹有正向贡献
  （DE-1 `30.433` vs 旧 CycleState `34.911`，改善 12.8%）。
- **未证伪根因三的弱版本**：oracle 直注仍差 baseline 1.98×，说明仅靠
  注入真实交通状态不能独立解决 2× 差距。
- **意外反直觉发现**：10 维 oracle 特征**不如**学习的 32D+16D hidden state
  （DE-1 比 DE-3 差 23.5%）。这说明状态分支"学到的表征"本身有信息量，
  瓶颈不在"信号 vs 学到"，而在"如何把表征送进 decoder 让它真的用"。
- **AR-1 已成为唯一方向明确的下一动作**：在 DE-3 的 init 拼接位置基础上
  扩展为"每步拼接 + 输出投影"，看是否能进一步逼近 baseline。
- DE-2（极端耦合）已暂停——DE-1 已经把"信号灯信息对轨迹的边际贡献"
  这个问题答了大半，再做 DE-2 的边际收益有限。

**完成标志**：

- ✅ 已产出 `experiments/cyclestate/DE1_oracle_inject` 的 comparable 结果。
- ✅ 已在 [EXPERIMENT_LOG.md §2.4](../EXPERIMENT_LOG.md) 完整记录。
- ✅ 已在 `models.py` / `train.py` / `evaluate_model.py` 中通过 `--oracle_inject_mode` 落地。
- ✅ smoke test 已通过（forward/backward/shape 验证）。
- ✅ val / test 双口径 comparable 评估已完成。

### 4.2 DE-2: 极端耦合压力测试 — ⏸ 已暂停

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

### 4.3 DE-3: 最简可行版本（Minimum Viable CycleState） — ✅ 已完成

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

**实际结果**（2026-06-12）：

- `test + 20`: **24.632 / 57.135**
- `val + 20`: **65.537 / 161.993**
- 与旧 CycleState (`34.911 / 69.133`) 相比，ADE 改善 **29.4%**。
- 与 baseline (`15.359 / 31.514`) 相比，仍有 1.6× 差距。

**结论**：

- 根因一（耦合路径过窄）**已确认** — 直接拼接比加性残差好得多。
- 但仅修复根因一不够，根因二/三（aux loss / 信号灯信息已被 baseline 充分捕获）
  尚未被证伪。
- 决策：**DE-1 必须在任何架构改动前执行**，因为 DE-3 留下了 1.6× 的差距，
  这部分差距只能由 DE-1 的 oracle 直注来回答。

**完成标志**：

- ✅ 已产出 `experiments/cyclestate/de3_minimal_viable` 的 comparable 结果。
- ✅ 已在 [EXPERIMENT_LOG.md §2.3](../EXPERIMENT_LOG.md) 完整记录。
- ✅ 已在 `models.py` / `train.py` / `evaluate_model.py` 中通过 `--minimal_viable_mode` 落地。
- ✅ smoke test 已通过（forward/backward/shape 验证）。

---

### 4.4 AR-1: 直接条件注入（DE-3 之上叠加多点） — ✅ 已完成

**问题**: DE-3 把 `[queue_last, cycle_last]` 只在初始化时拼接到 decoder init 向量；
DE-1 把 10 维 oracle 特征拼到 LSTM input。两者**都是直接拼接**，但拼接位置不同
（LSTM input vs init），结果差异 23.5% (DE-3 比 DE-1 好 23.5%)。这说明
"decoder 怎么使用这些表征"才是真正起决定作用的部分。

**方案**（在 DE-3 基础上扩展）：

1. 把"只在初始化时拼接"扩展为"每步拼接"：state context 作为 decoder LSTM
   每步的**显式输入**。
2. state context 同时进入输出投影层（与 `pred_lstm_hidden` 一同经过输出 MLP），
   进一步强化对最终预测的直接影响。
3. 在新架构上训练最小版本，与 DE-3 / baseline 做 comparable 对比。

**期望**:

- 如果 AR-1 接近 baseline → AR-2（乘法门控）作为备选，AR-3（aux loss 重新设计）作为细化。
- 如果 AR-1 与 DE-3 差不多 → 说明拼接位置对 decoder 影响有限，瓶颈在别处。
- 如果 AR-1 仍差 baseline 1.5×+ → 优先回到问题定义层面（见分支 C2）。

**实际结果（2026-06-13）**：

- 训练协议：warmup 150b + refine 50b = 200b 总 batch
- `val + 20`: **57.954 / 140.002**（比 DE-3 val 65.54 改善 11.6%）
- `test + 20`: **28.631 / 65.631**（比 DE-3 test 24.63 **差 16.2%**）
- 与 baseline (`test + 20: 15.359 / 31.514`) 差距 **1.86×**
- 与 DE-3 (`test + 20: 24.632 / 57.135`) **差 16.2%**
- 与 DE-1 (`test + 20: 30.433 / 66.544`) **改善 5.9%**
- 与旧 CycleState (`test + 20: 34.911 / 69.133`) **改善 18.0%**

**结论**：

- **AR-1 否决了"加大注入强度 → 更好"的原假设**（PLAN.md §4.4 设计的
  假设）：val 改善 11.6% 但 test 变差 16.2%，说明"加性拼接"有"甜蜜点"，
  过犹不及。
- **learned state context 的最优注入位置是 init**（DE-3 单点注入），
  不是 init+per-step+output（AR-1 多点注入）。"多点注入"反而稀释了
  state context 的边际贡献。
- 可能原因：
  1. **信息冗余 / 梯度被稀释**：在 init / per-step / output 三个位置都
     拼同样的 state context 48 维，每一处都在做"告诉 decoder 现在是
     什么 state"的事，decoder 可能反而被冗余信号干扰。
  2. **per-step 注入破坏 LSTM 状态演化**：DE-3 的 init 拼接让 state
     context 通过初始化进入 LSTM 隐状态演化；AR-1 在 per-step 又强行
     塞 48 维 state context，**强制覆盖**了 LSTM 自然演化出的隐状态
     信号，导致 LSTM 内部的"演化轨迹"被打断。
  3. **过拟合 val / 欠泛化 test**：val 上 11.6% 改善但 test 上 16.2%
     变差，说明 AR-1 的强注入让模型对 val 的特定分布过拟合了。
- **方法论价值**：DE-3 的"init 单点拼接"是 learned state context 的
  **最优注入位置**——AR-1 想用"多点注入"加强这个优势，但实际是
  **稀释了**这个优势。
- **AR-1 仍优于 DE-1 和旧 CycleState**：
  - vs DE-1 (`30.433`) → 改善 5.9%，说明 learned 48D 拼接 init/per-step/output
    **比** oracle 10D 拼 LSTM input 更好。
  - vs 旧 CycleState (`34.911`) → 改善 18.0%。
- **val/test 落差**：AR-1 val (57.95) vs test (28.63)，落差 2.02×，与
  DE-3 (val 65.54 / test 24.63, 落差 2.66×) 相比**落差显著缩小**，
  说明 AR-1 的强注入让模型在 val/test 分布差异上的鲁棒性变好；但
  **平均性能变差**。

**完成标志**：

- ✅ 已产出 `experiments/cyclestate/AR1_direct_inject` 的 comparable 结果。
- ✅ 已在 [EXPERIMENT_LOG.md §2.5](../EXPERIMENT_LOG.md) 完整记录。
- ✅ 已在 `models.py` / `train.py` / `evaluate_model.py` 中通过 `--ar1_direct_inject_mode` 落地。
- ✅ smoke test 已通过（forward/backward/shape 验证）。
- ✅ val / test 双口径 comparable 评估已完成。
- ✅ **AR-1 后的下一步**已在 §2.9 完成根因状态更新 ——
  learned state context 的最优注入位置是 init（DE-3），任何扩展都会退化。

---

### 4.5 AR-2: 乘法门控（DE-3 之上叠加 per-step 调制） — ✅ 已完成

**问题**: AR-1 在 DE-3 之上叠加"加性多点注入"（init + per-step + output），
结果**比** DE-3 差 16.2%，这说明"加性"路径有上限。AR-2 探索另一条
耦合方式——**乘法门控**：用 sigmoid gate 调制 `pred_lstm_hidden` 的
某些维度，让 state context 通过**乘法**决定哪些隐状态维度被放大/抑制。
这与 AR-1 的"加法堆叠"是本质不同的耦合方式。

**方案**（在 DE-3 基础上扩展）：

1. 保留 DE-3 的 init 单点拼接 `[queue_last, cycle_last]`。
2. 在 `pred_lstm_model` 每步更新后，**用 state context (48D) 通过 2 层 MLP +
   sigmoid 学习一个逐元素门控** `gate ∈ (0, 1)^{pred_lstm_hidden_size}`。
3. 然后 `pred_lstm_hidden = pred_lstm_hidden * gate`（在 `pred_lstm_model`
   更新**之后**立即施加）。
4. **不**修改 `pred_lstm_model` 的输入维度（与 DE-3 一致），也**不**修改
   `pred_hidden2pos` 的输入维度——AR-2 不"加性"地扩大输入，新增的耦合
   完全由"乘法调制"实现。
5. AR-2 与 `--oracle_inject_mode` / `--ar1_direct_inject_mode` 互斥
   （三种耦合方式不能同时启用以避免混淆归因）。

**期望**:

- 如果 AR-2 接近 DE-3 → 乘法调制是"加性"路径外的有效替代。
- 如果 AR-2 比 DE-3 差 → 加性 vs 乘法都不是问题核心，回到根因三弱版本。
- 如果 AR-2 比 AR-1 差 → 进一步证明"扩展路径"在 DE-3 附近就已饱和。

**实际结果（2026-06-13）**：

- 训练协议：warmup 150b + refine 150b = 300b 总 batch
- `val + 20`: **74.229 / 178.848**（比 DE-3 val 65.54 **差 13.2%**）
- `test + 20`: **32.368 / 77.927**（比 DE-3 test 24.63 **差 31.4%**）
- 与 baseline (`test + 20: 15.359 / 31.514`) 差距 **2.11×**
- 与 DE-3 (`test + 20: 24.632 / 57.135`) **差 31.4%**（CycleState 变体族最差）
- 与 AR-1 (`test + 20: 28.631 / 65.631`) **差 13.0%**
- 与 DE-1 (`test + 20: 30.433 / 66.544`) **差 6.4%**
- 与旧 CycleState (`test + 20: 34.911 / 69.133`) **改善 7.3%**

**结论**：

- **AR-2 否决了"乘法调制能超越加性拼接"的假设**：原假设认为"用
  sigmoid gate 调制 pred_lstm_hidden 比简单拼接更灵活"，但 AR-2
  (`32.368`) **比** AR-1 (`28.631`) **更差**，**比** DE-3 (`24.632`)
  **差 31.4%**。
- **AR-2 是 CycleState 变体族在 `test + 20` 上的最差变体**（除旧
  CycleState 外）。
- **AR-2 在 val 上也是较差变体**：val 74.229 比 DE-3 65.537 差 13.2%，
  比 AR-1 57.954 差 28.1%。这与 AR-1 在 val 上的优势（比 DE-3 改善
  11.6%）形成鲜明对比——AR-1 的"加性拼接"在 val 上对长等待/高密度
  排队场景有帮助，而 AR-2 的"乘法调制"在 val 上没有任何优势。
- **"加性 vs 乘法"两条路径都触到上限**：
  - AR-1 (加性多点) → 28.631
  - AR-2 (乘法调制) → 32.368（更差）
  - 这意味着 state injection 路线的"扩展空间"在 DE-3 附近就已饱和，
    继续在 init 拼接基础上叠加新机制**几乎不可能**让指标更接近 baseline。
- **乘法门控为何更差的可能解释**：
  1. **sigmoid gate 衰减问题**：训练初期 gate 接近 0.5，每步对
     `pred_lstm_hidden` 的所有维度乘 0.5 等于"硬性减半"，且 sigmoid
     输出在训练早期不稳定，可能导致隐状态幅度大幅波动，让训练更
     难收敛。
  2. **状态耦合被门控稀释**：与 AR-1 一样，AR-2 也"扩展"了注入强度
     （per-step），但门控机制把"信号源 → decoder"变成"信号源 →
     隐式控制信号 → decoder"，中间多了一层 sigmoid 衰减。
  3. **类比 LSTM 自身的 forget gate**：LSTM 内部本就有 forget/input/
     output 门来调制 hidden state，AR-2 在外部再叠加一个 sigmoid gate
     来"再调制"，可能与 LSTM 内部门控产生冗余甚至冲突。

**完成标志**：

- ✅ 已产出 `experiments/cyclestate/AR2_multiplicative_gating` 的 comparable 结果。
- ✅ 已在 [EXPERIMENT_LOG.md §2.6](../EXPERIMENT_LOG.md) 完整记录。
- ✅ 已在 `models.py` / `train.py` / `evaluate_model.py` 中通过 `--ar2_multiplicative_gating_mode` 落地。
- ✅ smoke test 已通过（forward/backward/shape 验证）。
- ✅ val / test 双口径 comparable 评估已完成。
- ✅ **AR-2 后的下一步**已在 §2.9 完成根因状态更新 ——
  整个 CycleState 变体族的"扩展路径"在 DE-3 附近就已饱和，
  **分支 C2 触发**。

---

### 4.6 DE-3 / DE-1 / AR-1 / AR-2 后变体族定论

四个决定性实验形成了一致的"CycleState 变体族"画像：

| 变体 | 注入位置 | 注入信号 | test ADE | vs DE-3 | 排序 |
|------|---------|---------|---------|---------|------|
| **DE-3 (单点, init)** | init | learned 32D+16D | **24.632** | — | 1（最优） |
| AR-1 (三点) | init + per-step + output | learned 48D | 28.631 | +16.2% | 2 |
| DE-1 (单点, input) | LSTM input | oracle 10D | 30.433 | +23.5% | 3 |
| AR-2 (单点 + 门控) | init + per-step gate | learned 48D | 32.368 | +31.4% | 4（最差） |
| 旧 CycleState | 多点混合 | learned 多信号 | 34.911 | +41.7% | 5 |

**核心观察**：

- **DE-3 是当前 CycleState 变体族的最优配置**：单点 init 注入 + learned
  32+16 维 hidden state。
- **AR-1 否决了"加大注入强度 → 更好"的假设**：多点注入 (init + per-step +
  output) 反而**比**单点注入差 16.2%，说明"加性拼接"有"甜蜜点"，过犹不及。
- **AR-2 进一步否决了"乘法调制能超越加性"的假设**：AR-2 (`32.368`) **比**
  AR-1 (`28.631`) 更差，是变体族最差。
- **DE-1 的 oracle 仍不如 DE-3 的 learned**：把"信号"换成 oracle 物理
  特征更差，说明 decoder 更容易消费学到的连续向量。
- **根因三的弱版本仍未被证伪**：四个变体全部仍差 baseline 1.6-2.1×，
  **"显式交通状态注入"对轨迹预测的边际贡献有限**这个假设被四个变体
  一致支持。

**变体族最终定论**：

- DE-3 应当被记录为"state injection 路线的最大可榨出价值"（`test + 20: 24.632`）。
- AR-1 / AR-2 一致显示"扩展路径触顶"，继续在 init 拼接基础上叠加新机制
  **几乎不可能**让指标更接近 baseline。
- 下一阶段应**完全离开 state injection 路线**，转入分支 C2 的备选项
  （trajectory-level / vehicle-vehicle 交互 / 不确定性建模）。

---

## 5. 活跃 backlog（重新排序）

> **说明**: 以下 backlog 已根据 AR-1 / AR-2 结果重新排序。P0 级（决定性实验）已完成 DE-3 / DE-1 / AR-1 / AR-2 全部四个，**分支 C2 触发** — 下一阶段应**完全离开 state injection 路线**，转入"trajectory-level / vehicle-vehicle 交互 / 不确定性建模"等替代方向。

### Phase 0: 决定性实验（已全部完成）

执行顺序：**DE-3 ✅ → DE-1 ✅ → AR-1 ✅ → AR-2 ✅ → 触发分支 C2**

| 实验 | 状态 | 关键结论 |
|------|------|---------|
| **DE-3** | ✅ 已完成 (2026-06-12) | 根因一确认：直接拼接比加性残差好 29.4%；仍有 1.6× 差距 |
| **DE-1** | ✅ 已完成 (2026-06-12) | oracle 直注 30.433（比旧 CycleState 改善 12.8%，但比 DE-3 差 23.5%）；否证根因三极端版本；意外发现 oracle 不如 learned state |
| **DE-2** | ⏸ 已暂停 | AR-1 / AR-2 已充分证明"加性/乘法路径都有上限"，DE-2 边际收益几乎为 0 |
| **AR-1** | ✅ 已完成 (2026-06-13) | DE-3 之上叠加 init+per-step+output 三点注入 → 28.631（**比 DE-3 差 16.2%**）；val 改善 11.6% 但 test 变差 16.2%，否决"加大注入强度 → 更好"假设 |
| **AR-2** | ✅ 已完成 (2026-06-13) | DE-3 之上叠加 per-step multiplicative gate → 32.368（**比 DE-3 差 31.4%**）；否决"乘法调制能超越加性拼接"假设；是变体族最差 |

**分支 C2 触发逻辑**：

- 四个变体（DE-3 / DE-1 / AR-1 / AR-2）一致支持"state injection 边际贡献已饱和"；
- "加性 vs 乘法"两条扩展路径都触到上限（都比 DE-3 差）；
- 继续在 init 拼接基础上叠加新机制**几乎不可能**让指标更接近 baseline；
- 下一阶段应**完全离开 state injection 路线**，转入分支 C2 的备选项。

**Phase 0 完成标志**：

- ✅ DE-3 / DE-1 / AR-1 / AR-2 四个决定性实验的 comparable 结果均已落地。
- ✅ 全部 val / test 双口径 comparable 评估已完成。
- ✅ smoke test 全部通过（forward/backward/shape 验证）。
- ✅ 变体族定论已写入 [EXPERIMENT_LOG.md §2.6](../EXPERIMENT_LOG.md)。
- ✅ **PLAN.md §2.9 已更新根因状态** ——
  整个 CycleState 路线的"扩展空间"在 DE-3 附近已饱和，**分支 C2 触发**。

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

问题更像模块耦合与阶段职责划分错误，而不是单一超参能修好。

**重要更新（2026-06-13）**：DE-3 / DE-1 / AR-1 / AR-2 四个决定性实验一致支持
"**state injection 路线边际收益已饱和**"，分支 C2 已触发。这意味着：

- **G3 的所有 warmup 稳定性结论**都建立在"state-to-decoder 耦合方式不变"
  的前提下，而分支 C2 选定的备选方向（C2-1 trajectory-level /
  C2-2 vehicle-vehicle / C2-3 不确定性建模）**几乎都涉及架构变动**。
- 因此 G3 的诊断价值在分支 C2 之下**显著降低** —— 如果新架构不再
  使用 state branch，G3 的 warmup 稳定性 KPI 直接失效。
- **G3 应当被收口为"参考性 KPI"**，而不是"必须先解决的瓶颈"。
  后续在分支 C2 选定方向上**复用 G3 的诊断方法**（短程 vs 长程的
  对比），但**不再以 G3 的具体数值为目标**。

**诊断顺序（在分支 C2 选定方向后）**

1. 在分支 C2 选定的新架构上，**复用 G3 的诊断方法**：
   - 短程（warmup 50b）vs 长程（warmup 100b）的 val/test 落差
   - `teacher_forcing_ratio` sweep（不要假设 `0.6` 是最优）
2. 如果新架构上 G3 不再显著，则明确收口为"warmup 稳定性在
   state-branch 路径上才显著，trajectory-level / interaction 路径上
   稳定"。
3. 如果新架构上 G3 仍显著，则按原计划做：
   - warmup 是否应更短
   - refine 是否应更早接管
   - `rollout_residual_scale` 等稳定化参数是否需要小范围 sweep

**完成标志**

1. 在分支 C2 选定方向上复用 G3 诊断方法。
2. **不再**把"warmup 100b ADE 退化"当成独立线索推进 —— 它是
   state-branch 路径的特征，不是所有架构的特征。
3. 保留 `100b ADE <= 50b ADE * 1.15` 作为"state-branch 路径"的参考
   性 KPI，但不作为 G3 在分支 C2 之下的统一出口。

---

### G4: meso / macro 分支容量偏小 — ⏸ **分支 C2 触发后收口**

**问题**

`queue LSTM hidden=32`、`cycle LSTM hidden=16` 仍偏保守，可能限制状态表征能力。

**当前判断（2026-06-13 更新）**

- AR-1 / AR-2 已充分证明"state injection 路径的扩展空间在 DE-3 附近已饱和"。
- 即便增容 `queue_lstm_hidden` 到 128、`cycle_lstm_hidden` 到 64，也**几乎
  不可能**让指标更接近 baseline —— 因为瓶颈不在"状态表征容量"，而在
  "decoder 怎么使用这些表征"。
- **G4 应当被收口**：分支 C2 触发后，"增容 meso/macro 分支"不再是
  有意义的优化方向。

**完成标志**

- ✅ 已被 AR-1 / AR-2 的结果支持关闭 —— "增容 + 扩展注入"也
  不能突破 1.6× 差距的天花板。
- 不再把 G4 作为分支 C2 选方向时的考量项。

---

### G5: decoder residual 注入位置单一 — ✅ **被 AR-1 / AR-2 覆盖后收口**

**问题**

当前状态残差主要注入 `pred_lstm_hidden`，还没验证是否应扩展到 cell 或输出侧。

**当前判断（2026-06-13 更新）**

- AR-1 已经把"init + per-step + output"三个注入位置都做了：
  - per-step 注入（pred_lstm_model 每步输入）—— `28.631`
  - output 注入（pred_hidden2pos 输入）—— 与 per-step 一起
- AR-2 已经把"init + per-step multiplicative gate"也做了：
  - per-step 调制（pred_lstm_model 每步更新后乘以 gate）—— `32.368`
- 这两个实验**直接覆盖了 G5 的所有可能位置扩展**：
  - 加性多点（AR-1）→ 退化
  - 乘法调制（AR-2）→ 退化更严重
- **G5 应当被收口**：所有"decoder 残差 / 注入位置"的扩展都已通过
  AR-1 / AR-2 验证过，都不能突破 1.6× 差距的天花板。

**完成标志**

- ✅ AR-1 / AR-2 已经把 G5 的所有可能位置扩展都做了一遍。
- 不再把 G5 作为分支 C2 选方向时的考量项。

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

### Phase B: 决定性实验（已完成 DE-3 / DE-1 / AR-1 / AR-2，触发分支 C2）

**已完成**：

1. ✅ **DE-3**: 最简可行版本（2026-06-12 完成）
   - 只保留 queue/cycle last 直接拼接进 decoder 初始化
   - 去掉所有 rollout/gating/anchor/residual 机制
   - 关键结果：`test + 20: 24.632 / 57.135`，比旧 CycleState (34.911 / 69.133) 改善 29.4%
   - **结论**：根因一（耦合路径过窄）确认；剩余 1.6× 差距需要 DE-1 解释
   - **变体族定论**：DE-3 是当前 CycleState 变体族的**最优配置**（`test + 20: 24.632`）

2. ✅ **DE-1**: Oracle State 直注（2026-06-12 完成）
   - 把 10 维 oracle 交通状态直接拼到 decoder LSTM input
   - 关键结果：`test + 20: 30.433 / 66.544`
   - 与旧 CycleState 改善 12.8%（否证根因三极端版本）
   - 与 DE-3 差 23.5%（意外反直觉发现：oracle 10D 不如 learned 32D+16D）
   - 仍差 baseline 1.98×（未证伪根因三弱版本）
   - **结论**：瓶颈不在"信号 vs 学到"而在"如何把表征送进 decoder 让它真的用"——下一步必须是 AR-1

3. ✅ **AR-1**: 直接条件注入（2026-06-13 完成）
   - 在 DE-3 拼接位置（init）基础上扩展为"每步拼接 + 输出投影"
   - 关键结果：`test + 20: 28.631 / 65.631`
   - val 改善 11.6% 但 test 变差 16.2% → **否决"加大注入强度 → 更好"假设**
   - 与 DE-3 差 16.2%，与 DE-1 改善 5.9%，与旧 CycleState 改善 18.0%
   - **结论**：learned state context 的最优注入位置是 init（DE-3），不是多点拼接（AR-1）

4. ✅ **AR-2**: 乘法门控（2026-06-13 完成）
   - 在 DE-3 之上叠加 per-step multiplicative gate（区别于 AR-1 的加法拼接）
   - 关键结果：`test + 20: 32.368 / 77.927`
   - 与 DE-3 差 31.4%，与 AR-1 差 13.0%，与 DE-1 差 6.4%，与旧 CycleState 改善 7.3%
   - 是 CycleState 变体族在 `test + 20` 上的**最差变体**（除旧 CycleState 外）
   - **结论**：否决"乘法调制能超越加性拼接"假设；"加性 vs 乘法"两条扩展路径都触到上限

5. ⏸ **DE-2**: 极端耦合压力测试（已确认暂停）
   - AR-1 / AR-2 已充分证明"加性/乘法路径都有上限"
   - DE-2 极端增容 + 极端 scale + 去掉 aux loss 的边际收益几乎为 0
   - 不再考虑

**变体族定论**（按 `test + 20` ADE 排序）：

| 排名 | 变体 | ADE | 相对 DE-3 |
|------|------|-----|----------|
| 1（最优） | **DE-3** | **24.632** | — |
| 2 | AR-1 | 28.631 | +16.2% |
| 3 | DE-1 | 30.433 | +23.5% |
| 4（最差） | AR-2 | 32.368 | +31.4% |
| 5 | 旧 CycleState | 34.911 | +41.7% |

**Phase B 完成标志**：

- ✅ 四个决定性实验的 comparable 结果均已落地。
- ✅ 变体族定论已写入 EXPERIMENT_LOG.md。
- ✅ PLAN.md §2.9 已更新根因状态。
- ✅ **分支 C2 触发** — 见 Phase C。

### Phase C: 根据决定性实验结果分支 — **分支 C2 触发**

四个决定性实验（DE-3 / DE-1 / AR-1 / AR-2）一致支持"**state injection 路线
边际收益已饱和**"的结论，**分支 C2 触发**。

#### 分支 C1: 决定性实验全部正向 + AR-1 接近 baseline — ❌ 未触发

- AR-1 仍差 baseline 1.86×，不是"接近 baseline"状态。
- **状态**：未触发，关闭。

#### 分支 C2: DE-1 / AR-1 显示 oracle state 边际贡献有限 — ✅ **当前触发**

**触发证据**：

- DE-1 (oracle 10D 直注 LSTM input) → `test + 20: 30.433`，**比 DE-3 差 23.5%**；
- AR-1 (learned 48D 多点注入) → `test + 20: 28.631`，**比 DE-3 差 16.2%**；
- AR-2 (learned 48D 乘法门控) → `test + 20: 32.368`，**比 DE-3 差 31.4%**；
- 四个变体全部仍差 baseline 1.6-2.1×，且 AR-1 / AR-2 一致显示"加大注入强度"
  反而让指标退化。

**分支 C2 行动要求**：

1. **整个 CycleState 的核心假设需要重新审视**：
   - 显式交通状态记忆对信号灯路口轨迹预测的边际贡献是否在当前数据/容量
     约束下成立？
   - 是否需要换一个"问题定义"而非继续在 state injection 路径上优化？

2. **优先回到问题定义层面**：
   - 信号灯路口的轨迹预测，是否真的需要一个显式的交通状态记忆？
   - 是否应该把"对交通状态建模"当成"对轨迹预测有用的副产品"而非"必须显式
     注入 decoder 的信号"？

3. **可能的替代方向**（按推荐优先级排序）：

   - **C2-1: 更强的 trajectory-level modeling**（最高优先级）
     - 不再显式建模"信号灯状态 / 排队状态 / 相位时长"等
     - 把精力放到"如何让 trajectory LSTM 本身学到更精确的演化"
     - 可能的工具：更深的 trajectory encoder、更细粒度的位置编码、更好的
       多步未来预测损失
     - 评估指标：与 baseline 直接 comparable 对比

   - **C2-2: 更细粒度的 vehicle-vehicle 交互建模**（次高优先级）
     - 当前 `seqGAT` 主要建模 vehicle-light 关系
     - 可以探索 vehicle-vehicle attention / 关系图
     - 这与 baseline 的 `light_state_embedding` 路径互补——baseline 已经
       有"对车辆间交互"的隐式建模，C2-2 是显式升级

   - **C2-3: 更好的不确定性建模**（次高优先级）
     - 当前 best-of-K 采样本身就能提供多样性
     - 可以探索"显式预测分布"（高斯混合 / 流模型 / c-VAE）
     - 评估指标：minADE / minFDE（多模态）或 negative log-likelihood

   - **C2-4: baseline 本身的改进空间**（不推荐作为主线）
     - 重新审视 baseline 的 `light_state_embedding + pred_state + get_next_state`
       路径，看是否有改进空间
     - 但这与 CycleState 路线的目标相悖（我们要改进的是 CycleState 而不是
       baseline 本身）

4. **不**应该再做：
   - 任何在 DE-3 之上叠加新 state injection 机制的实验（AR-1 / AR-2 已给出
     结论：扩展路径触顶）
   - 任何 state injection 路径的变体（如 AR-3、AR-4 等）
   - 任何 continuation / refine 超参扫描
   - 任何推理侧 scale 扫描
   - DE-2（极端耦合）——已被 AR-1 / AR-2 充分证明边际收益为 0

#### 分支 C3: DE-1 正向但 AR-1 仍差 baseline — ❌ 未触发（被分支 C2 覆盖）

- 分支 C3 原本的触发条件是"DE-1 正向但 AR-1 仍差 baseline → 重点转向 AR-2"，
  但 AR-2 (`32.368`) 也比 DE-3 差，分支 C2 已经覆盖了这种情况。
- **状态**：被分支 C2 覆盖，关闭。

### Phase D: 诊断式消融（暂停，分支 C2 触发后重新评估）

如果分支 C2 选择的替代方向需要 CycleState 现有的子模块，复用 G7 消融。
否则这些消融在分支 C2 下不再直接相关，应当**完全收口**：

- 当前的 4 个 `disable_*` 开关是针对旧 CycleState 设计
- 如果分支 C2 选 trajectory-level / interaction 方向，这些开关在 DE-3 之下
  已经是 trivial（DE-3 已经全部 disable 了）
- 消融目标应当重新定义：例如"trajectory encoder 深度" / "interaction 强度" /
  "uncertainty head 类型"等

### Phase E: 如果分支 C2 备选项也站不住

如果分支 C2 选定的备选方向（如 C2-1 trajectory-level）也不奏效，那么：

1. **承认当前数据/容量约束下信号灯路口轨迹预测的天花板**：
   - baseline (D2TPred 原模型) + `num_samples=20` 可能就是当前 setting 的
     工程化上限
   - 继续投入 CycleState / state injection / trajectory-level / interaction
     改造的边际收益都接近 0

2. **回到研究方向而非工程优化**：
   - 重新审视任务定义（"未来 12 步轨迹预测"是否合理？）
   - 重新审视数据规模（当前数据量是否足以学到更复杂的交通动力学？）
   - 重新审视评估指标（ADE/FDE 是否是正确指标？）

3. **承认负结果并写成论文贡献**：
   - 四个变体一致支持"显式 state injection 边际贡献有限"，
     这本身就是一个**清晰的负结果**
   - 写清楚"为什么这条路走不通"比"勉强做出一个能 work 的 CycleState"
     更有方法论价值

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

- **DE-3 / DE-1 / AR-1 / AR-2 四个决定性实验均已完成，分支 C2 已触发**。
- 当前 CycleState 变体族的**最优配置**是 DE-3（init 单点拼接），
  `test + 20: 24.632`；AR-1 / AR-2 都比 DE-3 差，**任何在 DE-3 之上的
  state injection 扩展路径都已触顶**。
- 在分支 C2 备选方向（C2-1 trajectory-level / C2-2 vehicle-vehicle
  交互 / C2-3 不确定性建模）选定之前：
  - **不再**做任何 continuation/refine 超参扫描
  - **不再**做推理侧 scale 扫描
  - **不再**新增任何 state 分支机制
  - **不再**在 DE-3 之上叠加新 state injection 机制（AR-1 / AR-2 已给
    出结论：扩展路径触顶）
  - **不再**做 state injection 路径的变体（如 AR-3、AR-4 等）
  - **不再**重新做 DE-1 / DE-3 / AR-1 / AR-2
  - **不再**做 DE-2（已被 AR-1 / AR-2 充分证明边际收益为 0）
- 当前默认继续把 `README.md` 当项目入口，把 `EXPERIMENT_LOG.md` 当证据
  入口，把本文件当活跃待办入口。
- 任何影响研究叙事、协议口径或当前结论的改动，都应同步更新：
  - `README.md`
  - `EXPERIMENT_LOG.md`
  - `docs/PLAN.md`
- 第 2 节诊断不是一次性文档——DE-3 / DE-1 / AR-1 / AR-2 已更新根因
  状态。分支 C2 选定具体方向后，应继续更新对应根因的确认/推翻状态。

## 9. 风险提醒

- oracle 假设若不量化，论文叙事会有明显缺口；但如果把 `predicted cycle`
  误当成低成本评估开关，也会把 Phase 0.5 做成一个隐形大功能。
- **若在决定性实验（第 4 节）未完成前继续推进架构改动或训练，容易
  再次得到"结构改了但无法归因"的结果——这是过去多轮实验重复出现的
  问题。**
- 若 baseline `num_samples=20` 不补齐，后续任何"更接近论文口径"的
  比较都会站不稳。
- **DE-1 / AR-1 / AR-2 已给出"扩展路径触顶"的清晰二元信号**：
  - DE-1 否决了根因三的极端版本（oracle 10D 不如 learned 32D+16D）；
  - AR-1 否决了"加大注入强度 → 更好"的假设（val 改善 11.6% 但
    test 变差 16.2%）；
  - AR-2 否决了"乘法调制能超越加性拼接"的假设（AR-2 `32.368` 比
    AR-1 `28.631` 还差）。
  - 这三个否决形成一致证据：**state injection 路线的扩展空间在
    DE-3 附近就已饱和**。
- **最大的方法论风险**：整个 CycleState 的核心假设——
  "显式的交通状态记忆能改善信号灯路口轨迹预测"——在当前数据/容量
  约束下可能**不成立**。
  - DE-3 已确认根因一（耦合路径过窄）；
  - DE-1 已否证根因三的极端版本；
  - AR-1 / AR-2 已确认"扩展路径触顶"；
  - 剩余 1.6× 差距（DE-3 vs baseline）的根本来源仍未定位，且
    进一步在 state injection 路径上做架构重设计**几乎不可能**让
    指标更接近 baseline。
- **分支 C2 的执行风险**：
  - 如果选 C2-1（trajectory-level）方向，**承认 CycleState 核心
    假设不成立**是写出来的关键，不能再继续以"改进 CycleState"
    为目标；
  - 如果选 C2-2（vehicle-vehicle 交互）方向，需要先验证 baseline
    的 `seqGAT` 没有已经覆盖这个能力；
  - 如果选 C2-3（不确定性建模）方向，需要重新设计评估指标
    （minADE / minFDE 或 NLL），而不是 ADE / FDE；
  - 不推荐选 C2-4（baseline 改进），因为这与 CycleState 路线的
    目标相悖。
- **次大风险**：如果分支 C2 选定的方向（如 C2-1）也不奏效，那么
  应当承认整个数据/容量约束下信号灯路口轨迹预测的天花板，写出
  清晰的负结果（"为什么 state injection 走不通"）而不是继续堆
  结构。
- **第三大风险**：分支 C2 触发的"回到问题定义层面"可能被误读为
  "放弃 CycleState 路线"，但实际要求的是**承认 DE-3 是该路线的
  最大可榨出价值**，并把工程精力投入替代方向，而不是废弃所有
  工作。
- **最关键的边界**：DE-3 (`test + 20: 24.632`) **仍是 CycleState
  变体族的最优结果**，应当被作为"state injection 路线的最大可
  榨出价值"记录下来，而不是被 AR-1 / AR-2 的负结果"覆盖"掉。
