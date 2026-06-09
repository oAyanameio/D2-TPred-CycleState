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
- `--val_every`

### 5. Auxiliary supervision 升级
早期的辅助监督版本是直接监督 hidden state 切片。
这部分已经被升级为：
- `queue_aux_reg_head` + `queue_aux_cls_head`
- `cycle_aux_phase_head` + `cycle_aux_time_head` + `cycle_aux_change_head`

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

## 最新的重要实现对齐
当前最新的重要实现对齐是：

`dynamic rollout recursion + dynamic lane anchor aggregation + metric audit alignment`

其含义是：
- `queue rollout` 不再在每一步都从最后观测帧的静态 `base_queue_feature` 近似展开，
  而是显式依赖上一步 rolled meso-state
- `lane-consensus anchor` 不再固定为最后观测帧 anchor，
  而是在预测期中动态重聚合
- 训练内 `validate` 与独立 `evaluate_model.py`
  现在共用同一套 raw/average displacement 计算函数

当前启示是：
- 接下来的收益验证会更可信，
  因为“实现是否真的对应科研故事”与“训练/评估是否口径一致”这两个基础问题
  已经先被收口了

## 最新的重要协议修正
当前最新的重要协议修正是：

`smoke/protocol-check validation schedule fix`

其含义是：
- 旧版 `should_run_validation()` 会在 `max_train_batches > 0` 或 `num_epochs=0`
  时，于 `batch 0` 后立刻触发验证
- 当 `max_val_batches` 已经扩大到 20 这类中等预算短实验时，
  这种行为会让早期噪声过度影响 checkpoint 选择，也会拖慢协议检查
- 当前版本改为：
  在 `print_every` 的区间末尾触发验证，并保证最后一个 batch 一定验证

当前启示是：
- 后续所有 `protocol-check` 短实验都应该基于修复后的调度来读结论，
  旧 run 只能作为“模型在学”的辅助证据

## 最新的重要可比结果
当前已经确认两条仓库内可比参考线：

1. `D2TP/model_best.pth.tar` 在统一口径 `num_samples=4` 下，
   完整 `val` split 得到：
   - `ADE 38.493`
   - `FDE 78.706`
2. 同一 checkpoint 在统一口径 `num_samples=4` 下，
   完整 `test` split 得到：
   - `ADE 17.812`
   - `FDE 37.568`

这说明：
- baseline checkpoint 在 `test` split 上仍然很强
- 当前 `CycleState` 与真正“超过 baseline”之间仍有明显距离
- 后续讨论性能时必须明确 split 与采样口径，不能混用 `val/test`

## 最新的重要 hypothesis 结果
当前已经完成一次修复后协议下的 rollout 短对照：

- `warmup_main_v2_schedfix`（rollout on）：
  - `ADE 78.227`
  - `FDE 152.544`
- `warmup_no_rollout_v2_schedfix`（仅关闭 rollout）：
  - `ADE 71.863`
  - `FDE 140.974`

当前启示是：
- “真正递推的 queue rollout” 这个主假设还没有在短程 warmup 协议下兑现
- 下一步优先级不是继续堆新模块，
  而是修正 rollout 的训练入口、监督强度和状态注入方式

## 最新的重要 rollout 修复
当前已经完成两处更具体的 rollout 路径修正：

1. `training step-0 rollout alignment`
   - 训练态 step 0 不再直接吃 teacher-forced future offset
   - 现在改为与推理态一致，使用最后观测 offset 作为上一时刻已知运动
2. `anchored rollout decode context`
   - decoder 不再直接用 `rollout_queue_h_t` 整块替换 queue context
   - 现在改为：
     `observed queue context + gated rollout delta`

这两点的直接动机是：
- 让 training/inference 真正共享单步 meso rollout 逻辑
- 让 rollout memory 以“锚定残差”方式调制 decoder，而不是早期短训时直接接管 decoder

## 最新的重要结果更新
修复后的短协议对照结果变成：

- `warmup_main_v2_schedfix_rollfix_v2`（rollout on）：
  - `ADE 66.793`
  - `FDE 132.168`
- 对照参考 `warmup_no_rollout_v2_schedfix`：
  - `ADE 71.863`
  - `FDE 140.974`

当前启示更新为：
- rollout 主线并没有 conceptually 失败
- 一旦把训练态驱动与 decoder 注入方式修顺，
  `rollout on` 已经重新超过 `no_rollout`
- 下一步应继续沿 rollout 主线检查：
  - 更长 warmup 下这种优势是否稳定
  - rollout auxiliary supervision 是否仍然过强

## 最新的重要 stability 结论
当前已经完成更长的 matched warmup 检查：

- 默认 rollout-on：
  - `batch 50`: `71.978 / 135.287`
  - `batch 100`: `185.583 / 322.389`
- `no_rollout`：
  - `batch 50`: `67.747 / 124.741`
  - `batch 100`: `196.345 / 331.583`

当前启示是：
- 当前 warmup 在更长 short-run 训练上整体不稳
- 默认 rollout-on 到 `batch 50` 仍然略输 `no_rollout`
- 但到 `batch 100` 已经不是 rollout 单独崩，而是协议整体都在崩

## 最新的重要 rollout-aux 结论
当前已经把 rollout auxiliary supervision 从 queue aux 中拆成独立权重：

- 新参数：`aux_rollout_weight`
- 默认兼容旧行为：
  - 若不显式指定，则等于 `aux_queue_weight`

当前实验结论：
- `aux_rollout_weight=2.5`
  - `50-batch rollout-on`: `66.761 / 122.728`
  - 优于 `no_rollout@50b`: `67.747 / 124.741`
- `aux_rollout_weight=1.0`
  - `batch 50`: `71.872 / 134.018`
  - `batch 100`: `190.641 / 320.417`

当前启示更新为：
- rollout aux 的确不该继续和 `aux_queue_weight=10.0` 完全绑死
- 但它也不能被简单砍到很低；
  当前短程最佳点更接近 `2.5`，而不是 `1.0`
- 下一步更应该考虑：
  - 用 `aux_rollout_weight≈2.5` 作为短程最佳设置
  - 缩短 warmup 或提早切 refine，
    避免 `batch 100` 附近整体崩坏

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
4. 先区分 `smoke / protocol-check / comparable`，
   再讨论是否“超过 baseline/论文指标”

## 实际协作说明
从现在开始，这条研究协作默认发生在：

`D2-TPred-CycleState`

并且当前以分支：

`main`

作为持续推进的主线。
