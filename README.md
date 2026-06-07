# D2 TPred : Discontinuous Dependency for Trajectory Prediction under Traffic Lights

# 项目总览
本仓库当前同时承担两个角色：
1. 原始 `D2-TPred` 基线代码的可复现实验仓库。
2. 新研究方向 `CycleState` 的持续开发仓库。

## 当前开发约束
- 当前只保留 `main` 作为持续开发主线，不再维护长期实验分支。
- `origin/main` 是默认推送目标；`upstream/main` 仅用于对照原始 baseline/论文。
- 核心说明文档默认使用中文，代码字段、参数名和论文术语保留必要英文原文。

当前主线不是简单给原模型“再加一个时序模块”，而是围绕下面这条科研叙事展开：

`信号灯路口轨迹预测，本质上可以被建模为全周期交通状态记忆（full-cycle traffic-state memory）问题。`

也就是说，我们希望模型不仅记住：
- 个体车辆短时运动怎么演化；
- 场景内车辆之间如何交互；

还要进一步记住并预测：
- 车道尺度的排队/释放波如何演化；
- 信号周期尺度的相位状态如何约束未来行为。

# 当前框架
当前 `CycleState` 的整体框架可以概括为四层：

1. 微观层 `micro`
   - 继承原始 `D2-TPred` 的轨迹编码器、空间图交互和局部时间交互。
   - 这是当前性能基础，也是必须尽量保住的强基线能力。

2. 中观层 `meso`
   - 从观测窗口构造 queue-state 特征。
   - 通过 queue LSTM 建立车道级排队状态记忆。
   - 在预测期使用 `phase-rolling queue memory` 显式滚动中观状态。
   - 进一步加入 `lane-consensus anchor`，让同车道车辆共享更稳定的车道级共识。

3. 宏观层 `macro`
   - 从信号状态构造 cycle feature。
   - 通过 cycle LSTM 建立相位/周期记忆。
   - 通过 phase-conditioned gating 让不同灯态下的状态作用方式可学习、可消融。

4. 解码与训练层
   - 使用分阶段训练协议：`warmup / refine / adversarial`。
   - 使用 `tuple -> traffic_context` adapter，为后续 INT2 迁移保留统一接口。
   - 使用 structured auxiliary losses，分别监督 queue/cycle 的不同语义变量。
   - 当前最新版本使用 `baseline-compatible decoder warm-start`，避免新结构破坏原始 D2-TPred 解码器的已学习能力。

# 评测可比性说明
当前仓库里的实验结果需要分成三类看：
1. `smoke`
   - 目标是验证 forward/backward、日志、checkpoint、消融开关和损失项是否正常工作。
   - 常见特征是 `max_train_batches=1`、`max_val_batches=1` 或 `num_epochs=0`。
   - 这类结果不能直接用于宣称“超过 baseline”或“超过论文指标”。
2. `protocol-check`
   - 目标是验证训练协议、验证触发频率、采样逻辑、checkpoint 恢复与评估脚本是否彼此一致。
   - 可以用于排查“为什么训练内验证和独立评估看起来不一致”，
     但仍不应直接当成论文结论。
3. `comparable`
   - 只有在数据 split、checkpoint 来源、ADE/FDE 统计方式、采样次数和评估脚本口径
     都明确对齐后，结果才可用于和 baseline 或论文表格做正式比较。

当前主线先做的是：
- 先把 `train.py` 的验证逻辑与 `evaluate_model.py` 的误差聚合口径对齐；
- 再用统一口径复核 baseline checkpoint；
- 最后再启动长程 `warmup/refine` 正式实验。

当前已经拿到四条更有参考价值的中间证据：
- `baseline_audit_v1_val_full_num_samples4`
  - 在统一口径、`num_samples=4` 的完整 `val` split 复核中，
    原始 baseline checkpoint 得到 `ADE 38.493 / FDE 78.706`
  - 这条结果已经可以作为仓库内 `4-sample` 口径下的 `comparable` 参考线。
- `baseline_audit_v1_test_full_num_samples4`
  - 在统一口径、`num_samples=4` 的完整 `test` split 复核中，
    原始 baseline checkpoint 得到 `ADE 17.812 / FDE 37.568`
  - 这说明仓库内 baseline checkpoint 在 `test` split 上仍然很强，
    当前 `CycleState` 距离真正“超过 baseline”还有明显差距。
- `experiments/cyclestate/warmup_main_v2`
  - 在旧的 `protocol-check` 验证调度下，
    主配置 `CycleState` 第二个验证点下降到 `ADE 56.827 / FDE 107.416`
  - 但该 run 会在 `batch 0` 之后立刻做一次 20-batch 验证，
    因此只能作为“模型在学”的证据，不能当成最干净的短对照。
- `experiments/cyclestate/warmup_main_v2_schedfix` vs
  `experiments/cyclestate/warmup_no_rollout_v2_schedfix`
  - 在修正后的同口径短协议下，
    主配置得到 `78.227 / 152.544`，
    `no_rollout` 对照得到 `71.863 / 140.974`
  - 这说明“真正递推的 queue rollout”在当前短程 warmup 设置下
    还没有兑现预期收益，
    下一步必须先修 rollout 路径本身，而不是急着进入 refine/GAN。
- `experiments/cyclestate/warmup_main_v2_schedfix_rollfix_v2`
  - 在补上两处 rollout 路径修正后，
    主配置重新提升到 `ADE 66.793 / FDE 132.168`
  - 它已经重新优于前一轮 `no_rollout` 对照的 `71.863 / 140.974`，
    说明 scientific story 本身没有塌，
    问题主要出在训练态 rollout 的驱动方式与 decoder 注入方式。
- `100-batch matched warmup`
  - 进一步拉长后的 matched warmup 说明：
    当前默认 warmup 协议在 `batch 50` 附近还能比较，
    但到 `batch 100` 左右整体会一起崩。
  - 默认 rollout-on：
    - `batch 50`: `71.978 / 135.287`
    - `batch 100`: `185.583 / 322.389`
  - `no_rollout`：
    - `batch 50`: `67.747 / 124.741`
    - `batch 100`: `196.345 / 331.583`
  - 这说明问题已经不只是“rollout 路径本身”，
    而是当前 warmup 协议在更长短程训练里整体不稳。
- `rollout aux` 独立降权探测
  - 新增 `aux_rollout_weight` 后，
    `aux_rollout_weight=2.5` 的 `50-batch` rollout-on
    达到 `66.761 / 122.728`，
    重新优于 `no_rollout@50b` 的 `67.747 / 124.741`。
  - `aux_rollout_weight=1.0` 反而退到 `71.872 / 134.018`，
    说明 rollout aux 不能简单一刀砍得过低。

# 当前状态总结
## 已完成
- 已完成 `CycleStateTrajectoryGenerator` 的原型实现。
- 已完成 queue-state / cycle-state 两条状态分支。
- 已完成 `traffic_context` 统一接口适配。
- 已完成分阶段训练协议与 teacher forcing 调度。
- 已完成 structured auxiliary losses。
- 已完成 `state gating / queue rollout / lane anchor / decoder residual` 等可控消融开关。
- 已完成单元测试与基础语法验证框架。
- 已完成实验日志与对话上下文文档沉淀。

## 当前还没完成
- 还没有做足够长的 `warmup/refine` 正式训练，当前大部分结论仍然来自短 smoke run。
- 已经在短协议下初步恢复 `rollout on > no_rollout`，
  但还没有证明这种优势在更长 warmup / refine 中稳定存在。
- 已确认 `aux_rollout_weight=2.5` 比默认更适合当前短程 warmup，
  但还没有解决 `100-batch` 后半程整体崩坏的问题。
- 还没有证明 `decoder residual` 在更长训练下稳定带来收益。
- 还没有完成辅助状态预测质量的系统评估与可视化。
- 还没有完成 INT2 数据接口的正式接入。
- 还没有拿到“稳定超过 baseline/论文结果”的正式实验结论。
- 还没有完成更接近论文口径的 `num_samples=20` baseline 全量审计。
- 还没有形成最终论文需要的完整消融、误差分析和可视化证据链。

## 接下来要做的任务
1. 做更长的 `warmup` 与 `refine` 实验，先验证 `CycleState` 是否能稳定收敛。
2. 重点比较：
   - `decoder residual on/off`
   - `queue rollout on/off`
   - `lane anchor on/off`
   - `state gating on/off`
3. 增加辅助状态预测分析，确认中观/宏观状态分支到底学到了什么。
4. 如果状态分支被证明有用，再谨慎重新引入小权重 GAN。
5. 为 INT2 准备新的 adapter / loader，而不是重写模型主体。

## 最终要完成的目标
1. 在不简单照搬该领域现有套路的前提下，形成一条自洽、可投稿的科研主线：
   `signalized intersection forecasting as full-cycle traffic-state memory modeling`。
2. 在 `VTP_C` 上把 `CycleState` 训练到稳定优于原始 `D2-TPred` baseline 和论文指标的水平。
3. 通过系统消融证明：
   - 中观 queue-state memory 有效；
   - 宏观 cycle-state memory 有效；
   - 它们的交互方式是必要的，而不是可有可无的附属分支。
4. 把当前框架整理成可扩展到 `INT2` 的统一接口版本。
5. 最终沉淀为一套可用于 CCFA / SCI 一区论文写作的完整实验与叙事框架。

# CycleState 实验分支说明
这个仓库同时也被用作一个新研究方向的工作分支：
`CycleState: Full-Cycle Traffic-State Memory for Trajectory Prediction at Signalized Intersections`。

这个方向的核心动机，是把原始 `D2-TPred` 里以短历史交互为主的建模方式，
进一步推进成一种分层交通状态视角：
1. 微观层 `micro`：个体车辆的短时运动与交互。
2. 中观层 `meso`：车道级 queue-wave 状态。
3. 宏观层 `macro`：信号周期状态。

目前，这部分实现仍然是构建在 `D2-TPred` 之上的实验性原型
（`CycleState v0/v1/v2`），但已经可以训练，并且支持从原始 checkpoint
进行 warm-start。

# 当前实验相关文件
- `D2TP/models.py`
  - 新增了 `CycleStateTrajectoryGenerator`。
- `D2TP/train.py`
  - 新增了 `--model_type cyclestate`、`--generator_only`、辅助损失相关选项，
    以及 quick-smoke 训练控制参数。
- `D2TP/evaluate_model.py`
  - 新增了 `--model_type cyclestate`。

# CycleState 设计概述
相较于原始 `D2-TPred` 生成器，当前 `CycleState` 主要额外加入了：
1. `queue-state` 分支：
   - 从同车道邻居中提取弱监督的 queue-wave 统计量，
     包括 queue count、lane density、waiting ratio、release ratio、
     mean lane speed、当前信号状态、已持续时间以及到 stop-line 的距离。
2. `cycle-state` 分支：
   - 将 phase one-hot、elapsed phase time 和 phase-change indicator
     编码成紧凑的 signal-cycle memory。
3. 分层解码器：
   - 解码器初始化不再只依赖运动和图交互特征，
     还会引入 queue-state memory 和 cycle-state memory。
4. 显式辅助预测头：
   - queue-state 和 cycle-state 不再直接监督 hidden state 的切片，
     而是通过专门的 auxiliary head 来预测。
5. `phase-conditioned state gating`：
   - queue-state memory 和 cycle-state memory 不再只是简单拼接，
     它们对解码的作用会受到当前交通灯状态的显式调制。
6. 更强的 `meso/macro` 监督目标：
   - queue-state 和 cycle-state 的监督信号，比最初的弱目标版本更丰富。

# 优化日志
下面记录从最初原型到当前版本的优化过程。

## Stage 0: 原始基线
- 保留原始 `D2-TPred` 作为 baseline。
- 在基础分支中修复了 PyTorch 兼容性问题，包括：
  - 设备无关执行；
  - 针对短时间窗口更安全的归一化；
  - 更稳定的评估入口。

## Stage 1: 实验性克隆仓库
- 单独创建了一个实验仓库副本：
  `/home/lbh/D2-TPred-CycleState`
- 这样可以避免污染原始可复现的 `D2-TPred` 仓库。

## Stage 2: CycleState v0 原型
- 新增 `CycleStateTrajectoryGenerator`。
- 复用原始的微观轨迹编码器和图交互分支。
- 新增：
  - 车道级 `queue-state memory`
  - `signal-cycle memory`
- 已验证：
  - forward pass 正常；
  - backward pass 正常；
  - 能从原始 `model_best.pth.tar` 部分 warm-start。

## Stage 3: Quick Smoke 训练支持
- 在 `train.py` 中新增了 quick-debug 控制参数：
  - `--max_train_batches`
  - `--max_val_batches`
- 这使得我们能够快速验证新分支到底能不能学起来。

## Stage 4: Warm-Start 策略
- 新增了兼容型 checkpoint 加载器。
- 对于 `cyclestate`，当前模型会：
  - 复用原始 `D2-TPred` 中形状兼容的权重；
  - 让新的 queue/cycle 模块随机初始化；
  - 从 epoch 0 开始训练，而不是继承旧的 epoch 计数。

## Stage 5: 仅生成器稳定化
- 新增 `--generator_only`。
- 动机：
  - 在 `CycleState` 的早期实验中，应该先验证 queue/cycle memory
    是否有价值，而不是让 GAN 的不稳定性掩盖真实效果。

## Stage 6: 辅助状态监督
- 新增：
  - `--aux_queue_weight`
  - `--aux_cycle_weight`
  - `--gan_weight`
- 最初的实现是间接监督 queue/cycle hidden state。
- 当前实现升级为显式 auxiliary head：
  - `queue_aux_head`
  - `cycle_aux_head`
- 这样优化目标更容易解释，也更方便分析。

## Stage 7: 仅生成器 + 辅助监督验证
- 纯生成器 + auxiliary supervision 比直接混入 GAN 更稳定。
- 在短 smoke run 中，queue/cycle 的辅助损失下降得比较明确。
- 快速实验中的验证 ADE/FDE 也呈现下降趋势，
  说明 `CycleState` 分支是可训练的，并且已经带来了正向信号。

## Stage 8: 更强的交通状态目标
- 原始的 queue 监督只覆盖三个较弱统计量：
  - queue count
  - waiting ratio
  - release ratio
- 当前版本把 queue targets 升级成更丰富的 `meso-state` 集合：
  - queue count
  - waiting ratio
  - release ratio
  - lane queue length
  - stop-line occupancy
  - front-of-queue flag
- `cycle-state` 分支也得到了加强：
  - phase one-hot
  - elapsed phase time
  - remaining phase time
  - phase-change indicator
- 这一步让模型更接近预期的科研叙事：
  - 模型不再只是学习 trajectory-conditioned features，
    而是显式学习结构化的 `meso/macro traffic states`。

## Stage 9: Phase-Conditioned State Gating
- 新增了一个优化，避免朴素特征拼接。
- 动机：
  - 同样的 queue-wave memory，在红灯、黄灯和绿灯下，
    对解码的影响不应该完全一样。
- 新增：
  - `queue-state context gate`
  - `cycle-state context gate`
  - `decode-time cycle gate`
- 这使 `CycleState` 更贴近目标论文叙事：
  - 信号灯路口应被建模为条件交通状态系统，
    而不只是一般性的时空交互图。

## Stage 10: Tuple-to-Context Adapter
- 原因：
  - 原始 `VTP_C` 训练流程基于 tuple，便于快速迭代，
    但也把数据集语义硬编码进了调用路径。
- 改动：
  - 训练和评估阶段现在会构建结构化的 `traffic_context`
  - `CycleState` 生成器可以直接接收外部 `traffic_context`
  - 模型内部语义被重新整理为：
    - `agent`
    - `signal`
    - `scene`
    - `meso`
- 对科研叙事的贡献：
  - 这支持这样一个说法：
    模型学习的是 full-cycle traffic-state memory，
    而不是单纯依赖固定 tuple 布局。
- 对 INT2 迁移的意义：
  - 未来迁移时可以只替换 adapter，
    而不必重写模型主体。

## Stage 11: 分阶段训练协议
- 原因：
  - 早期实验表明，过早混入 GAN 会让我们难以判断
    新的 traffic-state 分支到底是否真的有用。
- 改动：
  - 新增 `--train_stage {warmup, refine, adversarial}`
  - 每个阶段都有明确默认协议：
    - `warmup`：只训生成器，不启用 GAN，状态监督更强
    - `refine`：只训生成器，不启用 GAN，辅助损失权重降低
    - `adversarial`：重新引入较小权重的 GAN
  - `teacher forcing` 也做成了分阶段调度，而不是固定值
- 对科研叙事的贡献：
  - 优化过程更符合我们的层级逻辑：
    先学会状态记忆，再证明状态记忆能帮助轨迹预测，
    最后再做 adversarial 精修。

## Stage 12: Structured Queue/Cycle Auxiliary Losses
- 原因：
  - 用单一的 MSE 统一监督所有 queue/cycle 目标太粗糙，
    也不符合这些状态变量本身的语义。
- 改动：
  - queue 监督被拆成：
    - 回归目标
    - 二分类目标
  - cycle 监督被拆成：
    - 相位分类
    - elapsed/remaining time 回归
    - phase-change 分类
  - 训练日志中也显式暴露这些结构化分项
- 对科研叙事的贡献：
  - 这提升了可解释性，也让 `meso/macro` 监督更贴合
    结构化交通状态建模的论文表述。
- 对原创性的贡献：
  - 目标不是加一个通用 auxiliary trick，
    而是显式编码“不同交通状态变量应该如何学习”的语义差异。

## Stage 13: 可控的 Phase-Conditioned Gating 消融
- 原因：
  - 如果 `phase-conditioned state modulation` 不能被干净地关闭，
    那就很难证明这个机制到底有没有帮助。
- 改动：
  - 新增 `--disable_state_gating`
  - queue-state gating、cycle-state gating、
    以及 decode-time cycle modulation 都可以一致关闭
  - 训练与评估都支持这条消融路径
- 对科研叙事的贡献：
  - 这让 gating 从一个隐藏的工程细节，
    变成一个可以被验证的科学主张。

## Stage 14: 实验日志规范
- 原因：
  - 一旦训练协议的变体多起来，只靠 README 级别叙事就不够了。
- 改动：
  - 新增 `EXPERIMENT_LOG.md`
  - 统一推荐把实验结果组织到 `experiments/` 目录下
  - 每个实验现在都可以保留自己的：
    - 命令
    - 协议阶段
    - gating 设置
    - auxiliary weights
    - checkpoints
    - 最佳 ADE/FDE 摘要
- 对科研叙事的贡献：
  - 这让当前分支从一次性原型，逐步变成一个可复现的研究开发轨道。
- 对 INT2 迁移的意义：
  - 更清晰的实验协议，也更方便未来做跨数据集比较。

## Stage 15: Phase-Rolling Queue Memory
- 原因：
  - 之前的 `CycleState` 解码器只在初始化时注入一次 queue-state memory。
  - 这对信号灯路口来说过于静态，因为 queue-wave state 会随着相位推进
    在预测期内持续演化。
- 改动：
  - 新增了预测期 `queue rollout` 分支
  - queue-state 现在会在预测期逐步向前滚动，依赖：
    - 当前 `cycle feature`
    - 预测运动偏移
    - 最后观测到的 `meso-state anchor`
  - 解码器使用滚动后的 queue memory，
    而不是始终复用一个固定 queue 向量
  - queue rollout 的预测结果也被纳入 structured queue loss 监督
- 对科研叙事的贡献：
  - 这让 `CycleState` 从“静态中观状态注入”变成了
    “相位演化中观状态记忆”。
  - 换句话说，模型不再假设 queue-wave state 在预测开始后被冻结，
    而是把 queue dynamics 视为未来本身的一部分。
- 对原创性的贡献：
  - 这里的关键不是一个通用 temporal refinement trick，
    而是一个特定于交通灯预测场景的 signal-conditioned
    `meso-state rollout` 机制。
- 实用价值：
  - 它更自然地连接了：
    - `macro cycle progression`
    - `meso queue-wave evolution`
    - `micro trajectory decoding`

### Stage 15 之后的当前理解
- queue-state 现在被建模为两层：
  - 来自历史观测窗口的 queue memory
  - 跨预测期滚动的 queue memory
- 目标论文叙事因此更完整了：
  - `full-cycle traffic-state memory`
    不只是记住观察到的交通状态，
    还包括在相位推进下把状态继续传播到未来。

## Stage 16: Queue Rollout 作为可控科学变量
- 原因：
  - 在引入 `phase-rolling queue memory` 之后，
    它必须变成一个可测试变量，而不是一个始终开启的隐藏实现选择。
- 改动：
  - 新增 `--disable_queue_rollout`
  - 训练和评估现在可以在以下两者之间切换：
    - 动态 `queue rollout`
    - 静态 `queue-state injection fallback`
  - rollout 相关损失会继续显示在日志中：
    - `QRollReg`
    - `QRollCls`
- 对科研叙事的贡献：
  - 这让我们可以提出一个更尖锐的问题：
    `未来 queue-wave 演化，是否比静态 meso-state context 更有助于轨迹预测？`
- 首个 smoke 比较：
  - rollout `on`：ADE `156.592`，FDE `300.038`
  - rollout `off`：ADE `156.663`，FDE `300.138`
- 当前理解：
  - 在这个单 batch smoke 设置里，优势还非常小，
    但方向是正向的，更重要的是这个变量已经可控、可验证。

## Stage 17: Lane-Consensus Meso Anchor
- 原因：
  - 即便加入了 `queue rollout`，中观状态仍可能过于 agent-local。
  - 在信号灯路口，queue-wave state 更适合被理解成一种车道级共识，
    而不是每个 agent 完全独立的局部上下文。
- 改动：
  - 在 `traffic_context` 中新增 `lane_queue_anchor_seq`
  - 每条 lane 现在都有一个由同车道 agent 计算出的共享 `meso anchor`
  - `queue rollout` 会被软约束拉向这个车道级共识 anchor
  - 新增 `--disable_lane_queue_anchor`，
    便于后续做干净消融
- 对科研叙事的贡献：
  - 这强化了论文叙事中的中观层主张：
    模型不只是滚动个体 queue-state，
    还显式尊重车道级集体交通状态结构。
- 对原创性的贡献：
  - 这依然不是通用 aggregation 模块，
    而是明确插入到 full-cycle meso-state evolution 路径中的
    `lane-consensus prior`。
- 首个 smoke 观察：
  - 打开 `lane-consensus anchor` 后，
    一次短 `warmup` smoke run 得到：
    ADE `156.591`，FDE `300.036`
  - 同时 rollout losses 仍然保持激活。

## Stage 18: Predictive Lane-Anchor Traceability
- 原因：
  - 引入车道级 `meso anchor` 后，
    它也应该在预测期中可被观察到，
    否则它只会停留在一个隐藏的内部偏置上。
- 改动：
  - 在模型 debug 输出中新增 `lane_queue_rollout_anchor_seq`
  - 这样就可以逐步追踪预测期内的车道级 `meso anchor`
  - 当使用 `--disable_lane_queue_anchor` 时，
    这条 trace 会干净地消失
- 对科研叙事的贡献：
  - 现在中观层故事不只是：
    `模型使用了 lane-level consensus`
  - 还变成了：
    `模型把 lane-level consensus 一直携带到了未来状态演化过程中`

## Stage 19: Baseline-Compatible Decoder State Residual
- 原因：
  - 早期的 `CycleState` 版本会通过拼接 queue/cycle memory
    直接把 decoder hidden state 变宽。
  - 这种做法在建模上看似自然，但它破坏了原始 `D2-TPred`
    最强部分之一的完整 warm-start 兼容性，也就是 decoder 本身。
- 改动：
  - `CycleState` 解码器现在保持与原始 `D2-TPred` 形状兼容
  - queue/cycle state 改为通过 gated residual pathway 注入，
    而不是直接加宽 decoder LSTM
  - 新增 `--disable_decoder_state_residual` 便于显式消融
  - 新增测试来验证：
    - `CycleState` decoder 形状与 baseline 一致
    - 原始生成器 checkpoint 可以在 decoder 相关部分零跳过加载
- 对科研叙事的贡献：
  - 这一步让我们的科学主张更尖锐：
    `CycleState` 应该通过交通状态记忆去调制一个强运动解码器，
    而不是让模型从头重学整个 decoder backbone。
- 实际价值：
  - 这显著提高了 warm-start 的保真度，
    也让稳定优化的路径更清晰。

### Stage 19 之后的首个 Smoke 观察
- `cyclestate + warmup + decoder residual on`：
  - ADE `54.784`，FDE `114.138`
- `cyclestate + warmup + decoder residual off`：
  - ADE `45.856`，FDE `93.743`

当前理解：
- 这仍然只是单 batch smoke 比较，还不能当成正式性能结论。
- 但它非常明确地说明：
  一旦恢复了完整的 decoder warm-start 兼容性，
  `CycleState` 就不再像之前那样处于严重退化原型状态。
- 这也确认了：
  “保护 baseline decoder 能力”已经成为下一阶段研究中的首要优化轴之一。

## Stage 20: 动态 Queue Rollout / Dynamic Lane Anchor / 评测口径收口
- 原因：
  - 之前的 `queue rollout` 虽然已经进入训练流程，但每一步仍主要从最后观测帧
    的静态 `base_queue_feature` 近似展开。
  - `lane-consensus anchor` 也主要复用最后观测帧 anchor，
    这和“预测期内中观状态持续演化”的故事并不完全一致。
  - 同时，训练内验证与独立评估脚本虽然方向一致，但底层误差聚合函数没有完全共用，
    不利于做严格的可比性审计。
- 改动：
  - `queue rollout` 现在改成真正递推：
    每一步使用上一步 rolled meso-state，而不是反复从静态末帧展开。
  - `lane-consensus anchor` 现在改成预测期动态重聚合，
    而不是始终复用最后观测帧的共享 anchor。
  - 新增统一的 raw/average displacement 计算函数，
    让 `train.py` 的 `validate` 与 `evaluate_model.py` 共用同一套误差定义。
  - 新增 `val_every`，把“正式训练按 epoch 验证”和“smoke run 按极小 batch 快速验证”分开。
  - 新增调试信号：
    - `queue_rollout_feature_seq`
    - `decoder_state_init_residual_norm`
    - `decoder_state_step_residual_norm_seq`
- 对科研叙事的贡献：
  - 这一步把“full-cycle traffic-state memory”的故事从
    “我们有这个想法”推进成
    “我们真的在预测期里递推中观状态，而且能跟踪它如何调制 decoder”。
- 当前结论：
  - 这一步主要是让方法实现和科研叙事重新对齐，
    不是新的性能结论。
  - 接下来真正需要看的，是统一口径下更长的 `warmup/refine` 结果。
  - 当前已经用一次 `protocol-check` 级短实验验证：
    `experiments/cyclestate/warmup_dynamic_protocol_v1`
    在 `skipped 0 keys` 的完整 warm-start 下成功跑通，
    得到 `ADE 59.663 / FDE 126.717`，并且新的 rollout losses 正常激活。
  - 随后又补了一次“训练内验证也走多采样 best-of-K”的 `protocol-check`：
    `experiments/cyclestate/warmup_bestofk_protocol_v1`
    在 `num_val_samples=4` 下得到 `ADE 52.322 / FDE 109.916`。
  - 这说明训练内验证的采样口径会真实影响数值和 checkpoint 选择，
    因此后续正式实验必须把这一点固定下来，再谈是否“超过 baseline”。

## Stage 21: 验证调度修复与 Rollout Sanity Check
- 原因：
  - 旧的 `should_run_validation()` 在 `smoke / protocol-check` 模式下，
    会在 `batch 0` 之后立刻触发验证。
  - 当 `max_val_batches` 已经放大到 20 这种“中等预算短实验”时，
    这种早期大验证既耗时，也会把极早期噪声混进 checkpoint 选择。
- 改动：
  - `smoke / protocol-check` 现在改为：
    按 `print_every` 的区间末尾触发验证，
    并保证最后一个 batch 一定触发验证。
  - 已为该行为补充并通过 `unittest` 回归测试。
- 新证据：
  - 原始 `D2TP/model_best.pth.tar` 在统一 `4-sample` 口径下，
    完整 `val` 为 `38.493 / 78.706`，
    完整 `test` 为 `17.812 / 37.568`。
  - 修正验证调度后做 20-train-batch / 20-val-batch 的匹配短对照：
    - `warmup_main_v2_schedfix`：`78.227 / 152.544`
    - `warmup_no_rollout_v2_schedfix`：`71.863 / 140.974`
- 当前解释：
  - `CycleState` 主线并不是“完全训不动”，
    因为旧 run 的第二个验证点确实降到了 `56.827 / 107.416`；
  - 但在更干净的短协议对照里，
    `rollout on` 目前还不如 `no_rollout`，
    所以当前最需要优化的不是叠新模块，
    而是把 rollout 的训练入口、监督强度和状态注入方式调顺。

## Stage 22: Rollout Path Root-Cause Fix
- 原因：
  - 针对 `rollout on < no_rollout` 的异常现象做代码级排查后，
    发现了两个具体失配点：
  - 第一，训练态 rollout 在第 0 步直接使用 teacher-forced future offset，
    而推理态使用的是最后观测到的历史 offset，
    两者不是同一套单步状态演化逻辑。
  - 第二，decoder 在 rollout 开启后会直接把 `rollout_queue_h_t`
    整块替换为 queue decode context，
    缺少对观测期 `gated_queue_last` 的锚定。
- 改动：
  - 训练态 step-0 rollout 现在改为与推理态一致，
    使用最后观测 offset 作为上一时刻已知运动。
  - rollout queue context 现在改为
    `observed queue context + gated rollout delta` 的锚定残差注入，
    不再把 rollout hidden 整块替换进 decoder。
  - 为这两点都新增了回归测试。
- 新证据：
  - 修复前：
    - `warmup_main_v2_schedfix`：`78.227 / 152.544`
  - 修复后：
    - `warmup_main_v2_schedfix_rollfix_v2`：`66.793 / 132.168`
  - 对照参考：
    - `warmup_no_rollout_v2_schedfix`：`71.863 / 140.974`
- 当前解释：
  - 这说明 rollout 主线并不是概念上无效，
    而是之前的训练/注入路径把它训歪了。
  - 当前 `rollout on` 已经重新优于 `no_rollout`，
    所以下一步应该继续沿 rollout 主线微调监督强度和长程稳定性，
    而不是退回静态 queue context。

## Stage 23: Matched Warmup Stability And Rollout-Aux Decoupling
- 原因：
  - 在 `20-batch` 级别，rollout-on 已经重新优于 `no_rollout`，
    但这还不足以证明优势稳定。
  - 因此需要先做更长的 matched warmup，
    再判断是否应该把 rollout aux 从 queue aux 中拆开调节。
- 新证据 1：`100-batch matched warmup`
  - `rollout on`：
    - `batch 50`: `71.978 / 135.287`
    - `batch 100`: `185.583 / 322.389`
  - `no_rollout`：
    - `batch 50`: `67.747 / 124.741`
    - `batch 100`: `196.345 / 331.583`
  - 解释：
    - 默认 warmup 配方下，`rollout on` 到 `batch 50` 仍略输 `no_rollout`
    - 到 `batch 100` 时两边都明显崩坏
    - 说明当前更大的问题是：warmup 协议在更长短训上整体不稳
- 新证据 2：`aux_rollout_weight` 独立探测
  - 新增独立 `aux_rollout_weight`，默认仍兼容旧行为
  - `aux_rollout_weight=2.5`：
    - `50-batch rollout on`: `66.761 / 122.728`
    - 已重新优于 `no_rollout@50b`: `67.747 / 124.741`
  - `aux_rollout_weight=1.0`：
    - `100-batch rollout on`: `batch 50` 为 `71.872 / 134.018`
    - 比 `2.5` 明显更差
- 当前解释：
  - rollout aux 确实不该继续和 `aux_queue_weight=10.0` 完全绑死
  - 但它也不能被简单粗暴降得太低；
    当前最优短程点更像落在 `2.5` 左右，而不是 `1.0`
  - 下一步的主要方向不再是继续往下砍 rollout aux，
    而是：
    1. 用 `aux_rollout_weight≈2.5` 作为当前短程最佳设置
    2. 把 warmup 改成更短阶段，或提早切换到 refine，
       避免在 `batch 100` 附近整体崩坏

### 新训练协议下的首批 Smoke 观察
- `cyclestate + warmup + aux + gating`：
  - ADE `155.554`，FDE `298.113`
- `cyclestate + warmup + aux + no gating`：
  - ADE `155.569`，FDE `298.149`
- `cyclestate + warmup + no aux + gating`：
  - ADE `155.555`，FDE `298.116`
- `d2tpred baseline quick run`：
  - ADE `84.391`，FDE `172.836`

当前理解：
- 新协议现在已经可以在受控短实验中稳定执行和复现；
- 在当前这个极小的 smoke 设置里，gating 显示出轻微正向趋势；
- structured auxiliary supervision 现在已经稳定、可控、可分析，
  即便它在单 batch 尺度下的指标收益还不明显；
- 原始 `D2-TPred` baseline 目前仍然明显更强，
  这也符合预期，因为 `CycleState` 仍处于新分支 warm-start 原型阶段。

# 当前推荐训练模式
对于早期 `CycleState` 实验，目前推荐的模式是：
1. 使用 `--model_type cyclestate`
2. 使用 `--generator_only`
3. 从原始 `model_best.pth.tar` 进行 warm-start
4. 开启 `--aux_queue_weight` 和 `--aux_cycle_weight`
5. 在 queue/cycle 分支稳定前，不启用 GAN
6. 正式长程实验显式设置 `--val_every 1` 或更大的 epoch 间隔；
   `smoke` 才继续依赖极小 `max_train_batches/max_val_batches`
7. 在 `smoke / protocol-check` 模式下，
   现在默认按 `print_every` 的区间末尾做验证，不再在 `batch 0` 立刻触发大验证

示例：
```bash
CUDA_VISIBLE_DEVICES=2 python D2TP/train.py \
  --log_dir experiments/cyclestate/warmup_v1 \
  --model_type cyclestate \
  --train_stage warmup \
  --device cuda \
  --pin_memory \
  --resume ./model_best.pth.tar
```

# 下一步计划优化
当前 `CycleState` 分支仍然是一个早期原型。下一步的主要优化方向包括：
1. 以当前 `4-sample` baseline 全量结果为仓库内可比参考线，
   再继续补更接近论文口径的 `num_samples=20` 审计。
2. 继续修 rollout 主线，再谈完整长程 refine：
   - 当前短程最佳设置先采用 `aux_rollout_weight≈2.5`
   - 优先检查“缩短 warmup / 提前 refine”能否解决 `batch 100` 附近整体崩坏
3. 在 rollout 已恢复短程优势的基础上，再做匹配短对照：
   - `queue rollout on/off`
   - `lane anchor on/off`
   - `state gating on/off`
   - `decoder residual on/off`
4. 只有当新的 staged protocol 能避免 `100-batch` 级别崩坏，
   才值得继续扩大 warmup/refine 训练长度。
5. 增加评估阶段的辅助状态预测质量分析与可视化。
6. 只有在 `warmup/refine` 已证明稳定并且主配置优于匹配 baseline 时，
   才谨慎重新引入小权重 adversarial 训练。

# 数据是如何采集的？
`VTP-TL` 数据集采自带有交通信号灯的城市路口，用于预测车辆在一天中不同时段的轨迹，
覆盖了较丰富的真实驾驶场景。采集时，使用无人机在距离地面约 70 到 120 米的高度
尽可能静止悬停，从俯视视角记录车辆在非高峰、早晚高峰和夜晚等时段穿过路口区域的轨迹。


<div align=center>
<img src="https://github.com/VTP-TL/D2-TPred/blob/main/drone.png" width="780" height="312" alt=" "/><br/>
</div>

# 数据是在哪些场景采集的？
数据采自 3 类不同交通路口场景，包括十字路口、T 字路口和环岛。
这些场景具有不同数量的道路和信号灯，因此会诱发不同类型的车辆运动行为。

<div align=center>
<img src="https://github.com/VTP-TL/D2-TPred/blob/main/scenarios.png" width="762" height="628" alt=" "/><br/>
</div>

# 数据集概览
在 [VTP-TL](https://pan.baidu.com/s/1gAdWP58RCKl0RrsvtQotpw) 数据集中，
我们使用无人机采集了 3 类不同交通场景的数据。数据概览见下表。

<div align=center>
<img src="https://github.com/VTP-TL/D2-TPred/blob/main/summary.png" width="772" height="503" alt=" "/><br/>
</div>

# 数据包含内容
对于这 3 类采集场景，我们为每个场景提供两类文件：
1. 视频片段样例（`xxx.mp4`）
2. 车辆轨迹记录文件（`xxx.txt`）

其中，轨迹信息以像素坐标形式提供。

# 车辆轨迹记录文件（xxx.txt）
**F_id：** 第 1 列。对于每个 agent（按 `Agent_id` 区分），`frame_id` 表示该 agent 在视频中出现的帧。  
**A_id：** 第 2 列。对于每个 `xxx.txt` 文件，`Agent_id` 从 0 开始，表示车辆编号。  
**x：** 第 3 列。车辆在每一帧中的 x 坐标，单位为像素。  
**y：** 第 4 列。车辆在每一帧中的 y 坐标，单位为像素。  
**Lane_id：** 第 5 列。对于每个 `xxx.txt` 文件，`Lane_id` 从 0 开始，表示车道编号。  
**pa：** 第 6 列。对于每个 `xxx.txt` 文件，`inperception` 取值为 0 或 1，表示车辆是否位于交通灯影响区域内。  
**f：** 第 7 列。对于每个 `xxx.txt` 文件，`isfirstobj` 取值为 0 或 1，表示车辆是否是交通灯影响区域内的第一个 agent。  
**Lig_id：** 第 8 列。对于每个 `xxx.txt` 文件，`Lig_id` 从 0 开始，表示交通灯编号。  
**ls：** 第 9 列。对于每个 `xxx.txt` 文件，`Ls` 取值为 0、1、2，表示交通灯状态。  
**mb：** 第 10 列。对于每个 `xxx.txt` 文件，`Mb` 取值为 0、1、2，表示车辆运动行为。  
**lt：** 第 11 列。对于每个 `xxx.txt` 文件，`Ldurtime` 表示交通灯持续时间。  

**示例：**
<div align=center>
<img src="https://github.com/VTP-TL/D2-TPred/blob/main/smaple.png" alt=" "/><br/>
</div>
