# CycleState 科研故事文档

## 0. 一句话主线

我的研究不是简单在 `D2-TPred` 上增加一个时序模块，而是尝试重新定义**信号灯路口轨迹预测**这个问题：

> **signalized intersection trajectory prediction should be modeled as a full-cycle traffic-state memory problem**

中文就是：

> **信号灯路口轨迹预测，本质上不是单纯的轨迹外推问题，而是一个“全周期交通状态记忆与演化建模”问题。**

这个表述的核心是：未来轨迹不仅由个体短时运动和邻居交互决定，还受到**车道级排队/释放波**与**信号周期级相位状态**的共同约束。

---

## 1. 这个研究领域的特点是什么

### 1.1 轨迹预测本身就是一个强交互、强不确定、多模态的问题

自动驾驶中的轨迹预测要根据历史轨迹、周围交通参与者、道路结构和场景语义，预测目标车辆未来可能的运动轨迹。这个任务有三个公认特点：

1. **多模态**：同样的历史轨迹可能对应多种未来行为；
2. **交互性强**：周边车辆、行人和道路参与者之间存在耦合；
3. **场景依赖强**：车道拓扑、交通规则、地图先验都会显著影响未来运动。

### 1.2 路口尤其是信号灯路口，比普通道路更难

在普通道路上，很多预测任务主要处理“跟驰、换道、并线”等连续行为；但在信号灯路口附近，车辆行为通常会出现：

- **启停切换明显**；
- **受灯态驱动的阶段性变化明显**；
- **队列形成、排队传播、绿灯释放等中观交通现象明显**；
- **左转、直行、等待、抢行等意图切换更频繁**。

因此，信号灯路口并不是一个“普通轨迹预测场景 + 一个灯态特征”这么简单。

### 本节参考文献与引用理由

- **Lefèvre et al., 2014, A survey on motion prediction and risk assessment for intelligent vehicles**  
  链接：[Springer](https://link.springer.com/article/10.1186/s40648-014-0001-z)  
  理由：这篇综述很适合支撑“轨迹预测是自动驾驶核心任务，且存在 physics-based / maneuver-based / interaction-aware 等范式”的背景叙述。它也适合用来说明该领域长期关注的是“如何预测交通场景未来演化”。  

- **Social LSTM, CVPR 2016**  
  链接：[CVF Open Access](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Alahi_Social_LSTM_Human_CVPR_2016_paper.pdf)  
  理由：这是“交互感知轨迹预测”的经典早期工作之一，适合说明领域长期关注“邻居交互影响未来轨迹”。  

- **LaneGCN, ECCV 2020**  
  链接：[ECCV PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470528.pdf)  
  理由：适合支撑“地图结构、车道拓扑、actor-map interaction 对预测很重要”这一点。  

---

## 2. 这个领域通常怎么解决问题

从研究演化上看，这个领域大致形成了四类思路。

### 2.1 第一类：基于运动学或驾驶意图的经典方法

这类方法通常假设车辆服从某种动力学规律，或先识别高层意图，再生成连续轨迹。  
优点是可解释；缺点是对复杂交互和复杂场景的适应能力有限。

### 2.2 第二类：基于交互建模的深度学习方法

代表工作包括：

- `Social LSTM`：通过 social pooling 建模邻居影响；
- `Convolutional Social Pooling`：把车辆社会池化做成卷积结构；
- `Trajectron++`：使用图结构与异构输入建模多智能体预测。

这一路线的核心思想是：

> 未来轨迹不是单车独立演化，而是由多车相互作用共同决定。

### 2.3 第三类：结合地图/车道结构/全局注意力的方法

代表工作包括：

- `LaneGCN`：将 lane graph 明确建模；
- `Scene Transformer`：用 attention 统一建模 agent、road elements 和 time steps；
- `Wayformer`：尝试用更统一、更高效的注意力框架整合多种输入。

这类方法解决的是：

> 如何把地图、拓扑和全局场景信息更强地融入预测。

### 2.4 第四类：针对信号灯路口的专门方法

相比前面几类更通用的 motion forecasting 方法，近年来也出现了一些更加关注**signalized intersections** 的工作，例如：

- `Impact of Traffic Lights on Trajectory Forecasting of Human-driven Vehicles Near Signalized Intersections`：强调交通灯信息确实会影响预测；
- `D2-TPred`：围绕 traffic lights 下的 discontinuous dependency 建模；
- `A hierarchical behavior prediction framework at signalized intersections`：通过分层行为建模利用信号信息；
- `Knowledge-Informed Multi-Agent Trajectory Prediction at Signalized Intersections for Infrastructure-to-Everything`：结合实时信号、先验驾驶策略和交互建模。

### 本节参考文献与引用理由

- **Convolutional Social Pooling for Vehicle Trajectory Prediction, CVPR Workshops 2018**  
  链接：[CVF PDF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w29/Deo_Convolutional_Social_Pooling_CVPR_2018_paper.pdf)  
  理由：适合代表“车辆轨迹预测中的交互建模路线”。  

- **Trajectron++, ECCV 2020**  
  链接：[ECCV PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630664.pdf)  
  理由：适合代表“异构输入 + 多智能体图结构预测”。  

- **Scene Transformer, arXiv 2021**  
  链接：[arXiv](https://arxiv.org/abs/2106.08417)  
  理由：适合代表“统一注意力建模 agent、road、time”这一类大模型式 motion forecasting。  

- **Wayformer, arXiv 2022**  
  链接：[arXiv](https://arxiv.org/abs/2207.05844)  
  理由：适合说明很多新方法在追求更统一的输入建模，但仍然主要围绕“如何更好融合输入”，未必显式刻画信号控制交通流状态。  

- **Impact of Traffic Lights on Trajectory Forecasting of Human-driven Vehicles Near Signalized Intersections, arXiv 2019/2020**  
  链接：[arXiv](https://arxiv.org/abs/1906.00486)  
  理由：适合支撑“交通灯对轨迹预测确实重要，而且不能忽略”。  

- **D2-TPred: Discontinuous Dependency for Trajectory Prediction under Traffic Lights, ECCV 2022**  
  链接：[ECCV PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680512.pdf)  
  理由：这是你当前最直接的 baseline，也是你故事中“已有 traffic-light-aware 方法”的最好参照。  

- **A hierarchical behavior prediction framework at signalized intersections, arXiv 2021**  
  链接：[arXiv](https://arxiv.org/abs/2110.15465)  
  理由：适合说明已经有工作开始把“信号灯路口”当作一个特定问题建模，而不是普通道路预测。  

- **Knowledge-Informed Multi-Agent Trajectory Prediction at Signalized Intersections for Infrastructure-to-Everything, arXiv 2025**  
  链接：[arXiv](https://arxiv.org/abs/2501.13461)  
  理由：适合说明这个方向仍在发展，大家也在持续尝试把信号、先验策略、交互等知识结合起来。  

---

## 3. 这些做法的主要缺点是什么

这是你科研故事里最关键的一段。不是说前人方法“不好”，而是说：

> **它们已经解决了一部分问题，但还没有把“信号灯路口”的本质结构充分建模出来。**

我建议把缺点总结成下面四点。

### 3.1 多数方法仍然以“微观轨迹交互”为中心

无论是 LSTM、GNN 还是 Transformer，很多方法最终还是围绕：

- 个体历史轨迹；
- 邻居交互；
- 地图上下文；

来直接预测未来位置。

这当然有效，但它们主要捕捉的是**微观层面的运动与交互**。

### 3.2 交通灯常被当作“额外特征”，而不是“系统状态”

一些 intersection-aware 方法已经把 traffic light 纳入输入，但很多情况下它更像一个：

- one-hot 条件；
- 额外 context；
- proposal 约束；

而不是一个**会持续演化、并且会支配场景行为组织方式的宏观状态变量**。

换句话说，许多方法知道“当前是什么灯”，但没有真正建模：

- 当前相位持续了多久；
- 距离相位切换还有多久；
- 不同 phase 下同样的交互模式为何会产生不同结果。

### 3.3 很少有方法显式建模“车道级 queue-wave state”

这是你最容易讲出特色的一点。

信号灯路口中的很多行为并不是由单车局部交互直接决定的，而是由：

- 是否已经形成排队；
- 队列是否已逼近停止线；
- 当前是否处于释放波启动阶段；
- 同车道车辆是否已经形成共同的出队节奏；

这些**中观层交通状态**所支配。

而这类信息在多数轨迹预测方法中，要么被隐式吸收在 hidden feature 里，要么根本没有被建模。

### 3.4 多数方法预测“轨迹”，但不预测“状态如何演化”

很多模型会在编码阶段抽取上下文，然后直接进入 decoder 做 future rollout。  
但在信号灯路口里，真正决定未来轨迹的那些状态本身也在变化：

- queue 在增长还是消散；
- phase 在推进还是即将切换；
- lane-level consensus 是否正在形成；

如果这些状态被当作静态 context，一开始注入一次，然后整个未来期都不更新，那么就会损失很多关键信息。

### 这一段怎么说更稳

这里有一部分是**基于文献的归纳判断**，不是某篇文章原句直接写出来的结论。  
你汇报时可以明确说：

> “这是我对现有方法设计重点的归纳：它们大多强于微观交互和地图融合，但对信号灯路口里的中观 queue-wave 与宏观 cycle-state 缺少显式状态建模。”

### 本节参考文献与引用理由

- **D2-TPred, ECCV 2022**  
  链接：[ECCV PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680512.pdf)  
  理由：D2-TPred 已经意识到 traffic lights 下存在 discontinuous dependency，这是你论证“问题已经从普通道路推进到了 intersection-specific”的重要支点。  
  但“它主要围绕 discontinuous motion dependency，而非显式 meso/macro traffic-state memory”这一点，是**你基于方法结构做的归纳**。  

- **Impact of Traffic Lights..., arXiv**  
  链接：[arXiv](https://arxiv.org/abs/1906.00486)  
  理由：可支撑“signal information matters”，但同时也可引出“只加 traffic light 还不够，关键是怎么建模其时序约束”。  

- **Wayformer, arXiv 2022**  
  链接：[arXiv](https://arxiv.org/abs/2207.05844)  
  理由：适合说明不少新方法在追求统一输入建模与全局注意力，但未必显式建模信号控制场景中的交通状态层次。  

- **Shock Wave Approach for Estimating Queue Length at Signalized Intersections..., 2014**  
  链接：[SAGE](https://journals.sagepub.com/doi/10.3141/2422-09)  
  理由：这类交通工程文献非常适合证明“queue length / queue formation / queue dissipation 是 signalized intersection 的核心中观状态”，给你的 `queue-state` 分支提供交通学上的合理性。  

- **Real-time queue length estimation for congested signalized intersections, 2009**  
  链接：[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0968090X09000230)  
  理由：适合补充说明“队列长度和队列演化是交通信号路口长期研究的重要对象”，你不是凭空发明 queue-state，而是在把交通工程中的关键状态引入轨迹预测。  

---

## 4. 我的核心创新点是什么

你的创新点最好不要讲成“我加了 7 个模块”，而要讲成一个**统一假设**下的几个结构化设计。

### 4.1 核心创新：重新表述任务

我的第一创新不是某个网络结构，而是**问题重构**：

> 我把信号灯路口轨迹预测，从“短时轨迹交互建模”推进为“全周期交通状态记忆建模”。

这个表述的含义是：

- `micro`：个体车辆运动与邻居交互；
- `meso`：车道级 queue-wave state；
- `macro`：信号周期级 cycle-state。

### 4.2 创新二：提出 `micro-meso-macro` 分层建模框架

相比只做单层 feature fusion，我的方法显式区分三种层次：

1. **Micro**：保留 D2-TPred 的微观轨迹与交互建模；
2. **Meso**：构造 queue-state feature，并通过 queue memory 学习车道级排队/释放状态；
3. **Macro**：构造 cycle-state feature，并通过 cycle memory 学习相位/周期状态。

这个分层结构的好处是：

- 交通语义更清楚；
- 更适合做消融；
- 更容易讲“为什么这个分支存在”。

### 4.3 创新三：提出 `Phase-Rolling Queue Memory`

我不是把 queue-state 当作静态上下文，而是在预测期中继续滚动更新它。  
这相当于从：

- “编码一个状态”

走向：

- “让状态在未来期持续演化”

这是你方法里最有辨识度的点之一。

### 4.4 创新四：提出 `phase-conditioned gating`

同样的 queue-state memory，在红灯、黄灯、绿灯下对未来行为的作用不可能完全一样。  
因此，我不是简单拼接 state feature，而是通过 `phase-conditioned gating` 让状态作用方式随相位变化。

### 4.5 创新五：提出 `lane-consensus anchor`

车道级交通状态不应完全由单个 agent 的局部噪声决定。  
所以，我加入了 `lane-consensus anchor`，使同车道车辆共享一个更稳定的中观共识参考。

### 4.6 创新六：提出 `baseline-compatible decoder residual`

很多新结构加进去之后，首先会破坏 baseline 原有能力。  
我这里的策略不是改宽整个 decoder，而是以**残差方式**将 state memory 注入，尽量保护原始 D2-TPred 的 decoder 能力与 warm-start 兼容性。

### 4.7 创新七：结构化辅助监督与分阶段训练

为了让 `meso/macro` 分支不是“存在但不可解释”的黑箱，我加入了：

- `queue_aux_head`
- `cycle_aux_head`
- structured auxiliary losses
- warmup / refine / adversarial 分阶段训练协议

这样做的目的不是“多加几个 loss”，而是：

1. 提高训练稳定性；
2. 提高状态分支可解释性；
3. 支持后续分析“这个分支到底学到了什么”。

---

## 5. 我的方法是如何对应解决现有缺点的

| 现有缺点 | 我的对应设计 | 解决逻辑 |
|---|---|---|
| 多数方法主要建模微观交互 | `micro-meso-macro` 分层结构 | 从单层轨迹特征提升为分层交通状态建模 |
| 交通灯常被当成额外特征 | `cycle-state memory` + `phase-conditioned gating` | 把灯态从静态标签变成可演化、可调制的宏观状态 |
| 缺少显式中观 queue-wave state | `queue-state feature` + `queue memory` + `lane-consensus anchor` | 显式刻画车道级排队/释放状态，而不是隐式依赖 hidden feature |
| 上下文只编码一次，不滚动演化 | `Phase-Rolling Queue Memory` | 让中观状态在预测阶段持续更新 |
| 新结构易破坏 baseline 能力 | `baseline-compatible decoder residual` | 在尽量保住基线能力的前提下引入新状态记忆 |
| 状态分支难解释、难验证 | `queue/cycle aux heads` + structured losses | 让中观/宏观状态具备明确可监督目标 |

---

## 6. 适合向导师讲的科研故事版本

下面这段可以直接当汇报主线。

### 6.1 第一层：先讲问题背景

轨迹预测是自动驾驶决策中的核心问题。现有大量工作已经证明，历史轨迹、邻居交互和地图结构对于未来预测都很重要，因此出现了 Social-LSTM、Trajectron++、LaneGCN、Scene Transformer、Wayformer 等一系列代表性方法。

### 6.2 第二层：再指出“信号灯路口是一个特殊场景”

但我认为，信号灯路口与普通道路不同。  
在这个场景里，车辆未来行为不仅由个体运动和局部交互决定，还会被信号周期、排队传播、绿灯释放等机制强约束。  
因此，把信号灯只当作一个额外特征，或者只建模 agent-level interaction，可能不足以充分刻画这个场景。

### 6.3 第三层：给出你的核心判断

基于这个观察，我提出：

> signalized intersection trajectory prediction should be modeled as a full-cycle traffic-state memory problem.

也就是说，模型不仅要记住车怎么动，还要记住并预测：

- 车道级 queue-wave 如何演化；
- 相位/周期级 cycle-state 如何推进；
- 它们如何共同作用于未来轨迹。

### 6.4 第四层：再讲你的方法如何落地

具体来说，我在 D2-TPred 的微观建模能力基础上，增加了：

- `meso queue-state memory`
- `macro cycle-state memory`
- `phase-conditioned gating`
- `phase-rolling queue memory`
- `lane-consensus anchor`
- `baseline-compatible decoder residual`

从而形成一个 `micro-meso-macro` 的分层框架。

### 6.5 第五层：最后讲你的学术目标

我的目标不是只把 ADE/FDE 再压低一点，而是证明：

1. 信号灯路口预测可以被更好地表述为**全周期交通状态记忆问题**；
2. 显式建模 `queue-state` 与 `cycle-state` 比只建模微观交互更适合 signalized intersection forecasting；
3. 这种分层交通状态建模既能提升性能，也能增强可解释性，并有潜力迁移到更复杂的数据接口和场景中。

---

## 7. 汇报时可以直接使用的总结句

### 7.1 一句话摘要

> 我的工作不是在 D2-TPred 上简单堆模块，而是尝试把信号灯路口轨迹预测从“短时交互建模”推进为“全周期交通状态记忆建模”。  

### 7.2 创新点概括

> 我提出了一个 `micro-meso-macro` 的分层框架，在保留 baseline 微观建模能力的基础上，显式建模车道级 `queue-state` 与周期级 `cycle-state`，并让这些状态通过 `phase-conditioned gating` 和 `phase-rolling queue memory` 持续作用于未来预测。  

### 7.3 价值概括

> 如果这条主线成立，那么 signalized intersection forecasting 将不再只是“预测未来坐标”，而是“预测受信号周期和排队传播共同约束的交通状态演化”。  

---

## 8. 参考文献清单（按故事用途组织）

### A. 领域背景 / 综述

1. Lefèvre, S., Vasquez, D., Laugier, C. **A survey on motion prediction and risk assessment for intelligent vehicles**. ROBOMECH Journal, 2014.  
   链接：https://link.springer.com/article/10.1186/s40648-014-0001-z  
   用途：讲领域背景、经典范式分类。  

2. A Survey of Autonomous Driving Trajectory Prediction: Methodologies, Challenges, and Future Prospects, 2025.  
   链接：https://www.mdpi.com/2075-1702/13/9/818  
   用途：可作为补充综述，支撑“当前方法很多，但挑战仍在”。  

### B. 交互建模 / 深度预测代表作

3. Alahi et al. **Social LSTM: Human Trajectory Prediction in Crowded Spaces**. CVPR, 2016.  
   链接：https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Alahi_Social_LSTM_Human_CVPR_2016_paper.pdf  
   用途：交互建模经典起点。  

4. Deo and Trivedi. **Convolutional Social Pooling for Vehicle Trajectory Prediction**. CVPR Workshops, 2018.  
   链接：https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w29/Deo_Convolutional_Social_Pooling_CVPR_2018_paper.pdf  
   用途：车辆轨迹预测中的社会交互建模代表。  

5. Salzmann et al. **Trajectron++: Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data**. ECCV, 2020.  
   链接：https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630664.pdf  
   用途：异构输入、多智能体图结构建模。  

### C. 地图/车道/全局注意力建模

6. Liang et al. **Learning Lane Graph Representations for Motion Forecasting (LaneGCN)**. ECCV, 2020.  
   链接：https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470528.pdf  
   用途：支撑 lane graph / actor-map interaction 很重要。  

7. Ngiam et al. **Scene Transformer: A unified architecture for predicting multiple agent trajectories**. 2021.  
   链接：https://arxiv.org/abs/2106.08417  
   用途：支撑统一注意力式场景建模。  

8. Nayakanti et al. **Wayformer: Motion Forecasting via Simple & Efficient Attention Networks**. 2022.  
   链接：https://arxiv.org/abs/2207.05844  
   用途：支撑当前方法追求统一输入融合与高效建模。  

### D. 信号灯路口 / 交通灯相关预测

9. Oh and Peng. **Impact of Traffic Lights on Trajectory Forecasting of Human-driven Vehicles Near Signalized Intersections**. 2019/2020.  
   链接：https://arxiv.org/abs/1906.00486  
   用途：支撑“traffic light matters”。  

10. Zhang et al. **D2-TPred: Discontinuous Dependency for Trajectory Prediction under Traffic Lights**. ECCV, 2022.  
    链接：https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680512.pdf  
    用途：你的直接 baseline，也是“signalized intersection 任务化”的关键参照。  

11. **A hierarchical behavior prediction framework at signalized intersections**. 2021.  
    链接：https://arxiv.org/abs/2110.15465  
    用途：支撑“已经有人尝试针对信号灯路口做专门行为建模”。  

12. Yin et al. **Knowledge-Informed Multi-Agent Trajectory Prediction at Signalized Intersections for Infrastructure-to-Everything**. 2025.  
    链接：https://arxiv.org/abs/2501.13461  
    用途：支撑“这一方向仍在快速发展，大家在尝试把信号、策略知识与交互结合”。  

13. Lee et al. **Deep Learning-Based Multimodal Trajectory Prediction with Traffic Light**. Applied Sciences, 2023.  
    链接：https://www.mdpi.com/2076-3417/13/22/12339  
    用途：支撑“已有工作将 traffic light 融入预测，但通常仍偏向将其作为输入条件或上下文特征”。  

### E. 交通工程 / queue-state 合理性支撑

14. Liu et al. **Real-time queue length estimation for congested signalized intersections**. Transportation Research Part C, 2009.  
    链接：https://www.sciencedirect.com/science/article/pii/S0968090X09000230  
    用途：支撑“queue length / queue evolution 是 signalized intersection 长期重要研究对象”。  

15. Cai et al. **Shock Wave Approach for Estimating Queue Length at Signalized Intersections by Fusing Data from Point and Mobile Sensors**. Transportation Research Record, 2014.  
    链接：https://journals.sagepub.com/doi/10.3141/2422-09  
    用途：支撑“queue formation / dissipation / shockwave 是信号路口的关键中观状态”。  

16. **Towards Data-Driven Vehicle Estimation for Signalised Intersections in a Partially Connected Environment**. Sensors, 2021.  
    链接：https://www.mdpi.com/1424-8220/21/24/8477  
    用途：可补充支撑“信号化路口中的车流/车辆数量/排队估计本身就是重要问题”。  

---

## 9. 使用提醒

1. 上面关于前人工作的“缺点总结”，有些是**你基于方法侧重点做的归纳判断**，汇报时最好明确说“这是我的归纳”，不要冒充成原文作者自己的结论。  
2. 目前你的实验状态仍然是**原型完成 + smoke run 验证可训练**，所以故事应强调“问题重构与方法框架”，不要过早把故事压成“我已经稳定超过 baseline”。  
3. 导师如果追问“你这个故事和 D2-TPred 的本质差别是什么”，你就回答：  
   - D2-TPred 强在 `discontinuous dependency`；  
   - CycleState 进一步强调 `full-cycle traffic-state memory`；  
   - 前者更像在 traffic-light scene 下强化微观行为依赖建模，后者更像在 traffic-light scene 下显式加入中观/宏观交通状态记忆。  

