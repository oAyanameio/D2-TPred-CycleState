# CycleState 未来网络优化参考

> 更新时间：2026-06-08
> 目的：沉淀可持续补充的网络优化思路，重点回答“如何让 `queue/cycle memory` 更强地支配预测过程”，而不是简单堆叠更大的 backbone。

---

## 1. 文档定位

这份文档不是正式实验结论，也不是立刻执行的开发计划。

它的定位是：

1. 记录与 `CycleState` 高度相关的未来网络优化方向；
2. 区分“值得优先尝试的机制”和“暂时只做远期储备的机制”；
3. 让后续每次补充新论文、新想法、新实验时，有统一的落点。

当前方法主线见：

- [README.md](/home/lbh/D2-TPred-CycleState/README.md)
- [cyclestate_research_story.md](/home/lbh/D2-TPred-CycleState/docs/cyclestate_research_story.md)
- [models.py](/home/lbh/D2-TPred-CycleState/D2TP/models.py:806)

---

## 2. 当前 CycleState 的核心结构

当前 `CycleStateTrajectoryGenerator` 可以概括为：

`micro backbone + meso queue memory + macro cycle memory + phase-rolling queue rollout + decoder residual modulation`

对应代码位置：

- `micro backbone`：`Trajectory LSTM -> GAT -> seqGAT -> autoregressive decoder`
  - [models.py](/home/lbh/D2-TPred-CycleState/D2TP/models.py:666)
- `queue memory`
  - [models.py](/home/lbh/D2-TPred-CycleState/D2TP/models.py:898)
- `cycle memory`
  - [models.py](/home/lbh/D2-TPred-CycleState/D2TP/models.py:908)
- `phase-rolling queue memory`
  - [models.py](/home/lbh/D2-TPred-CycleState/D2TP/models.py:1237)
- `decoder residual modulation`
  - [models.py](/home/lbh/D2-TPred-CycleState/D2TP/models.py:1509)

这条路线的优点很明确：

- 不是把交通灯只当一个普通特征；
- 已经显式区分了 `micro / meso / macro`；
- 已经有“预测期状态继续演化”的雏形。

但当前最明显的结构性瓶颈也很明确：

1. `queue/cycle memory` 目前更像“旁路条件”，而不是“主干控制器”；
2. `queue/cycle` 对 decoder 的作用主要通过 `residual + gate` 注入，控制力度有限；
3. `lane_queue_anchor` 目前本质上还是均值池化，不是真正的 lane-level memory；
4. `phase-rolling queue memory` 虽然已经 rollout，但还没有上升到完整的显式状态转移系统。

因此，未来优化的主问题不是“换不换 Transformer/Mamba”，而是：

> 如何把 `queue/cycle memory` 从“辅助条件”升级成“真正支配轨迹生成过程的状态控制变量”。

---

## 3. 优化总原则

未来网络优化优先遵循下面四条原则：

1. 优先增强 `queue/cycle memory` 对预测过程的控制力，而不是优先替换整个 backbone。
2. 优先选择与现有 `CycleState` 叙事兼容的机制，而不是完全推翻当前研究故事。
3. 优先做可解释、可消融、可逐步接入的改造，而不是一次性引入过重的新体系。
4. 优先改“memory 的读写和支配方式”，再考虑更大规模的生成主干。

---

## 4. 重点可迁移方向

### 4.1 FiLM-style 条件调制

**来源领域**

- 视觉推理、条件生成
- 代表工作：`FiLM: Visual Reasoning with a General Conditioning Layer`

**核心思想**

不用只在末端拼接条件特征，而是让条件变量直接生成逐通道的 `scale/shift`，持续调制主干特征。

**对 CycleState 的迁移方式**

让 `queue/cycle memory` 不只生成一个 decoder residual，而是生成：

- decoder hidden 的逐通道 `gamma/beta`
- decoder cell state 的逐通道 `gamma/beta`
- `pred_hidden2pos` 前特征的逐通道调制

可写成：

```text
queue/cycle memory
 -> FiLM generator
 -> gamma, beta
 -> modulate decoder hidden/cell/output features
```

**为什么比当前更强**

当前实现是“加一点残差”；FiLM 是“每一步都改写主干内部表征”。  
这更容易让 `queue/cycle` 真正主导输出分布。

**与当前代码的结合点**

- 替换或扩展 `build_decoder_state_residual(...)`
  - [models.py](/home/lbh/D2-TPred-CycleState/D2TP/models.py:1509)
- 在 decoder step 内对 `pred_lstm_hidden`、必要时对 `pred_lstm_c_t` 做条件调制
  - [models.py](/home/lbh/D2-TPred-CycleState/D2TP/models.py:1726)

**优先级**

`P1`

---

### 4.2 HyperNetwork / 条件化参数生成

**来源领域**

- 条件生成、元学习、动态网络
- 代表工作：`HyperNetworks`

**核心思想**

条件变量不只是调特征，而是直接生成另一部分网络的参数。

**对 CycleState 的迁移方式**

让 `cycle-state memory` 生成 decoder 中小范围、低秩、可控的动态参数，例如：

- step decoder gate 的权重偏置；
- `pred_hidden2pos` 的低秩 adapter；
- `queue rollout gate` 的动态偏移项。

推荐先做“小超网络 + 低秩适配”，不要上全量动态权重。

**适用场景**

- 想让不同 phase 下 decoder 的工作方式发生真实变化；
- 想强调“红灯 / 绿灯 / 切换期不是同一个生成机制”。

**与当前代码的结合点**

- `decoder_state_gate`
  - [models.py](/home/lbh/D2-TPred-CycleState/D2TP/models.py:1017)
- `pred_hidden2pos`
  - [models.py](/home/lbh/D2-TPred-CycleState/D2TP/models.py:530)
- `decode_cycle_gate`
  - [models.py](/home/lbh/D2-TPred-CycleState/D2TP/models.py:993)

**优先级**

`P2`

---

### 4.3 Mixture-of-Experts（状态分专家）

**来源领域**

- 大模型、条件计算、稀疏专家路由
- 代表工作：`Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer`

**核心思想**

不是所有样本都用同一套解码器；由状态门控选择更适合当前样本的专家。

**对 CycleState 的迁移方式**

将 decoder 或 decoder 头拆成若干交通状态专家，例如：

- `red / queue buildup`
- `green / release`
- `phase transition`
- `free-flow / weak-control`

由 `queue/cycle memory` 产生 routing weight。

**为什么适合路口问题**

信号灯路口天然具有 regime-switching dynamics。  
同样的历史轨迹，在不同 phase 与不同 queue state 下，后续动力学不是同一套机制。

**与当前代码的结合点**

- `pred_hidden2pos` 可先拆成多 expert heads
- 再逐步考虑 `pred_lstm_model` 后接 expert adapter

**优先级**

`P2`

---

### 4.4 Lane Memory Bank（替代均值 anchor）

**来源领域**

- 外部记忆、可寻址 memory、结构化场景表示
- 代表工作：`Memory Networks`、`Differentiable Neural Computer`

**核心思想**

不要把 lane-level state 压成简单均值；应保留可寻址、可更新、可读写的 lane memory slot。

**对 CycleState 的迁移方式**

把每条 lane 建成一个 memory slot，slot 中存：

- lane queue density
- stop-line occupancy
- front-of-queue tendency
- release tendency
- phase compatibility

agent 在解码时依据：

- `lane_id`
- `stopline_distance`
- `current phase`
- `pred_offset`

去查询 lane memory。

**为什么比当前更强**

当前 `lane_queue_anchor` 本质上是 lane 内均值：

- [models.py](/home/lbh/D2-TPred-CycleState/D2TP/models.py:1209)

这会抹掉“谁更靠近停止线、谁更像队首、lane 当前释放到哪一段”这些结构信息。

**优先级**

`P1`

---

### 4.5 RSSM / 世界模型式显式状态转移

**来源领域**

- 世界模型、model-based RL、latent dynamics
- 代表工作：`PlaNet`、`Dreamer`

**核心思想**

把未来期 rollout 视为一个显式状态转移系统，而不是只在 decoder 外挂一个启发式更新器。

**对 CycleState 的迁移方式**

将当前 `phase-rolling queue memory` 升级为：

```text
previous latent traffic state
+ current cycle feature
+ previous predicted motion
 -> latent transition
 -> next traffic state
 -> decoder reads next traffic state
```

其中 latent traffic state 可以包含：

- queue latent
- cycle latent
- lane latent
- transition uncertainty

**为什么值得做**

你现在已经有 rollout 的原型，只是还偏 heuristic + residual。  
这条路线可以把“full-cycle traffic-state memory”真正系统化。

**风险**

- 改动明显更大；
- 训练稳定性要求更高；
- 需要更好的辅助监督与可观测状态设计。

**优先级**

`P3`

---

### 4.6 Hierarchical Maneuver / Option State

**来源领域**

- 层级强化学习、行为预测、选项策略
- 代表工作：`Options Framework`、`Director`

**核心思想**

让高层状态先决定“当前属于哪一类子行为或交通子目标”，低层再生成连续轨迹。

**对 CycleState 的迁移方式**

增加一个轻量高层 head，让 `queue/cycle memory` 先预测：

- `stop-before-line`
- `hold-queue`
- `start-release`
- `commit-turn`
- `proceed-straight`

低层 decoder 再条件化生成位移。

**意义**

这比直接只看连续位置更容易把 `cycle-state` 解释成策略层变量。

**优先级**

`P2`

---

### 4.7 Constraint-guided Decoding / Projection

**来源领域**

- 约束生成、神经投影、神经符号解码

**核心思想**

不仅让状态影响特征，还让状态参与“输出可行性”的筛选或投影。

**对 CycleState 的迁移方式**

在解码后加入显式约束模块，例如：

- red-light stop-line consistency
- lane compliance
- impossible release suppression
- phase-incompatible motion penalty

可以先从两种轻量形式开始：

1. `candidate reranking`
2. `projection-style correction`

**意义**

这类方法不一定提升表达力上限，但能显著增强 `queue/cycle` 对最终轨迹的支配力。

**优先级**

`P2`

---

## 5. 建议优先级排序

当前最推荐的尝试顺序如下：

1. `Lane Memory Bank`
2. `FiLM-style 条件调制`
3. `状态分专家 MoE`
4. `层级 maneuver / option state`
5. `约束引导解码`
6. `RSSM / 世界模型式状态转移`
7. `HyperNetwork` 更强动态参数化

排序依据：

- 与当前代码兼容性；
- 对“memory 主导预测”的提升幅度；
- 实验可控性；
- 失败成本。

---

## 6. 不建议当前优先做的方向

### 6.1 整体替换成 Transformer

原因：

- 当前问题主矛盾不是长序列表达力不足；
- `obs_len=8, pred_len=12` 的有效时序并不长；
- 更大的注意力主干不自动等于更强的 `queue/cycle` 支配力。

### 6.2 整体替换成 Mamba

原因：

- 当前场景并不体现 Mamba 的长序列优势；
- 方法创新点会被 backbone 替换弱化；
- 你真正缺的是状态控制机制，而不是线性复杂度序列模型。

### 6.3 继续只在 warmup 上堆训练技巧

原因：

- 训练协议当然要稳；
- 但从长期看，结构上 `queue/cycle` 对 decoder 控制力不够，仍是更本质的瓶颈。

---

## 7. 与当前代码的直接改造入口

后续如果开始逐步实现，建议优先观察和改造这些位置：

- `build_decoder_state_residual(...)`
  - [models.py](/home/lbh/D2-TPred-CycleState/D2TP/models.py:1509)
- `build_rollout_decode_queue_context(...)`
  - [models.py](/home/lbh/D2-TPred-CycleState/D2TP/models.py:1525)
- `rollout_queue_step(...)`
  - [models.py](/home/lbh/D2-TPred-CycleState/D2TP/models.py:1378)
- `build_lane_queue_anchor_seq(...)`
  - [models.py](/home/lbh/D2-TPred-CycleState/D2TP/models.py:1209)
- `CycleStateTrajectoryGenerator.forward(...)`
  - [models.py](/home/lbh/D2-TPred-CycleState/D2TP/models.py:1571)

可以把它们分别理解为：

- `state -> decoder` 的主接口；
- `rollout state -> decoder` 的主接口；
- `queue state` 的预测期转移核心；
- `lane-level meso state` 的当前简化实现；
- 整体结构升级的总装配点。

---

## 8. 建议的短期实验主线

在不推翻现有故事的前提下，短期最推荐的实验序列是：

1. `lane memory bank` 替换 `lane mean anchor`
2. `FiLM decoder modulation` 替换单纯 residual injection
3. `MoE decoder head` 由 `queue/cycle` 做路由

短期不建议直接同时做多项大改。  
每次只动一条主结构，才有可能判断性能变化究竟来自哪里。

---

## 9. 参考来源

下面这些工作不是都做了 signalized intersection，但它们提供了“让状态变量更强地主导输出”的方法原型：

- `FiLM: Visual Reasoning with a General Conditioning Layer`
  - https://arxiv.org/abs/1709.07871
- `HyperNetworks`
  - https://arxiv.org/abs/1609.09106
- `Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer`
  - https://arxiv.org/abs/1701.06538
- `End-To-End Memory Networks`
  - https://arxiv.org/abs/1503.08895
- `Differentiable Neural Computers`
  - https://www.nature.com/articles/nature20101
- `Learning Latent Dynamics for Planning from Pixels`
  - https://arxiv.org/abs/1811.04551
- `Mastering Diverse Domains through World Models`
  - https://arxiv.org/abs/2301.04104
- `Director: Deep Hierarchical Planning from Pixels`
  - https://arxiv.org/abs/2206.04114

与 signalized intersection 直接相关、可作为问题背景和对照的工作：

- `D2-TPred: Discontinuous Dependency for Trajectory Prediction under Traffic Lights`
  - https://arxiv.org/abs/2207.10398
- `KI-GAN: Knowledge-Informed Generative Adversarial Networks for Enhanced Multi-Vehicle Trajectory Forecasting at Signalized Intersections`
  - https://arxiv.org/abs/2404.11181
- `Knowledge-Informed Multi-Agent Trajectory Prediction at Signalized Intersections for Infrastructure-to-Everything`
  - https://arxiv.org/abs/2501.13461
- `Fusing transportation rules and diverse motion behaviors for trajectory prediction in traffic intersections`
  - https://www.nature.com/articles/s41598-026-46123-7

---

## 10. 后续补充模板

后续每新增一个方向，按下面模板补：

```markdown
### X.X 模块名

**来源领域**

**代表论文**

**核心思想**

**迁移到 CycleState 的方式**

**预期收益**

**主要风险**

**建议代码入口**

**实验优先级**

**当前状态**
```

建议状态值统一为：

- `idea-only`
- `paper-read`
- `design-ready`
- `implemented`
- `ablation-done`
- `dropped`

---

## 11. 当前结论

到 2026-06-08 为止，这份调研的核心结论是：

> `CycleState` 后续最值得强化的方向，不是“把 LSTM 换成 Transformer/Mamba”，而是把 `queue/cycle memory` 升级成更强的控制器，包括更强的条件调制、更显式的 lane memory、更清晰的状态分专家，以及更系统化的状态转移。

这条路线与当前研究叙事连续，也最有机会形成真正有辨识度的方法差异。
