# `D2-TPred-CycleState` 对话上下文

这个文件的作用，是把当前这条长研究对话中积累下来的实际项目上下文，
同步记录到 `D2-TPred-CycleState` 工作区中。

它并不复制聊天记录本身，而是记录：
- 当前项目状态
- 研究方向
- 已完成的实现进展
- 接下来推荐推进的动作

## 当前工作项目
- 项目路径：`/home/lbh/D2-TPred-CycleState`
- 当前分支：`main`
- 目标：
  - 继续实现并迭代新的研究方向：
    `CycleState: Full-Cycle Traffic-State Memory for Trajectory Prediction at Signalized Intersections`

## 为什么会有这个项目
原始 `D2-TPred` 仓库被保留下来作为可复现 baseline。
这个实验性克隆仓库的用途是：
1. 避免污染 baseline 仓库
2. 为快速研究迭代保留独立项目空间
3. 把之前围绕论文的分析讨论，真正落成一个可训练原型

## 核心科研叙事
当前思路并不是被表述成“再加一个多尺度时间模块”。
相反，它被表述为：

`signalized intersection trajectory prediction should be modeled as a full-cycle traffic-state memory problem`

对应的层级结构是：
1. 微观层 `micro`：个体运动与局部交互
2. 中观层 `meso`：车道级 queue-wave 状态
3. 宏观层 `macro`：信号周期状态

采用这套表述的原因是：
- 科研故事更强
- 相比通用 temporal encoder 升级更不容易显得增量化
- 更容易与已有方法做新颖性区分

## 已经完成的内容

### 1. 实验生成器
- 已在 `D2TP/models.py` 中加入 `CycleStateTrajectoryGenerator`
- 它复用了原始 `D2-TPred` 的微观轨迹与图交互分支
- 在此基础上新增：
  - `queue-state memory` 分支
  - `cycle-state memory` 分支
  - 分层 decoder 初始化与状态调制

### 2. 训练入口集成
- 已在训练/评估中加入 `--model_type {d2tpred, cyclestate}`
- 已加入从原始 `D2-TPred` checkpoint 的部分 warm-start
- 对于 `CycleState`：
  - 兼容权重会被复用
  - 不兼容的新层随机初始化
  - 训练从 epoch 0 开始

### 3. 已通过的基础验证
- forward pass 正常
- backward pass 正常
- partial checkpoint warm-start 正常

### 4. 训练稳定化工具
- `--generator_only`
- `--aux_queue_weight`
- `--aux_cycle_weight`
- `--gan_weight`
- `--max_train_batches`
- `--max_val_batches`

### 5. Auxiliary supervision 升级
早期的辅助监督版本是直接监督 hidden state 切片。
这部分已经被升级为：
- `queue_aux_head`
- `cycle_aux_head`

现在 queue/cycle 分支会显式预测辅助状态目标。

## 已观察到的快速实验现象

### 早期带 GAN 的原型
`CycleState` 可以训练，但优化过程比较噪。

### 仅生成器 + auxiliary supervision
这一模式明显更稳定。

从 quick smoke run 中已经观察到的趋势包括：
- queue auxiliary loss 明显下降
- cycle auxiliary loss 也在下降
- 短训练下验证 ADE/FDE 呈现下降趋势

这说明：
- `CycleState` 分支是可训练的
- 状态监督是有价值的
- 在早期阶段，`generator-only` 比完整 GAN 训练更合适

## 后续最重要的文件
- `README.md`
  - 记录累计优化日志和当前科研叙事
- `D2TP/models.py`
  - 包含当前 `CycleState` 原型实现
- `D2TP/train.py`
  - 包含训练控制逻辑和辅助损失接线
- `D2TP/evaluate_model.py`
  - 支持 `--model_type cyclestate`

## 当前高优先级下一步
1. 继续加强 `queue-state targets`：
   - queue length
   - release order
   - stop-line occupancy
   - front-of-queue status
2. 跑一轮更完整的 `generator-only` 训练实验
3. 增加评估阶段的辅助状态分析
4. 在稳定收敛之后，以更小权重重新引入 GAN

## 当前开发主线转向
当前主线已经明确转向：

`training-protocol strengthening + INT2 interface compatibility`

这意味着当前重点不再是快速继续堆模型拓扑，
而是优先让 `CycleState` 具备以下能力：
1. 按 staged protocol 训练
2. 用 structured losses 监督 queue/cycle states
3. 支持可控 gating ablation
4. 暴露 tuple-to-context adapter，使未来 INT2 迁移只需替换 adapter 层

## 最新的重要建模升级
协议稳定之后，一个关键建模升级是：

`Phase-Rolling Queue Memory`

其含义是：
- queue-state 不应该在 decoder 初始化后就被冻结
- 模型应该在预测期内持续滚动 `meso queue-wave state`
- 这个 rollout 依赖：
  - cycle progression
  - predicted motion
  - last observed queue-state anchor

这让科研故事保持一致：
`signalized intersection forecasting as full-cycle traffic-state memory modeling`
现在不仅包括状态编码，也包括未来阶段的状态演化。

## 最新的重要结构修正
当前最新的重要修正是：

`baseline-compatible decoder state residual`

其含义是：
- 早期 `CycleState` 版本会直接改宽 decoder，
  从而破坏原始 `D2-TPred` decoder 的完整 warm-start 兼容性
- 这很可能是早期指标严重退化的关键原因之一
- 当前版本保持原始 decoder 形状不变，
  改为通过 gated residual pathway 注入状态记忆

当前启示是：
- 后续优化必须把 `protect baseline decoder capability`
  视作一阶约束，而不是次要工程细节

## 当前研究目标带来的重要约束
后续优化必须持续遵守以下要求：
1. 讲好一个强而完整的科研故事
2. 目标是超过原始论文的实际性能
3. 避免简单照搬轨迹预测领域里常见的已有套路

## 实际协作说明
从现在开始，这条研究协作默认发生在：

`D2-TPred-CycleState`

并且当前以分支：

`main`

作为持续推进的主线。
