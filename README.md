# D2 TPred : Discontinuous Dependency for Trajectory Prediction under Traffic Lights

# 项目总览
本仓库当前同时承担两个角色：
1. 原始 `D2-TPred` 基线代码的可复现实验仓库。
2. 新研究方向 `CycleState` 的持续开发仓库。

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
- 还没有证明 `decoder residual` 在更长训练下稳定带来收益。
- 还没有完成辅助状态预测质量的系统评估与可视化。
- 还没有完成 INT2 数据接口的正式接入。
- 还没有拿到“稳定超过 baseline/论文结果”的正式实验结论。
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

# CycleState Experiment Branch
This repository is also used as a working branch for a new research idea:
`CycleState: Full-Cycle Traffic-State Memory for Trajectory Prediction at Signalized Intersections`.

The core motivation is to move from the original short-history interaction modeling
in D2-TPred towards a hierarchical traffic-state view:
1. Micro level: short-term agent motion and interaction.
2. Meso level: lane-level queue-wave state.
3. Macro level: signal-cycle state.

At the current stage, the implementation is still an experimental prototype
(`CycleState v0/v1/v2`) built on top of D2-TPred, but it is already trainable and
supports warm-start from the original checkpoint.

# Current Experimental Files
- `D2TP/models.py`
  Added `CycleStateTrajectoryGenerator`.
- `D2TP/train.py`
  Added `--model_type cyclestate`, `--generator_only`, auxiliary-loss options,
  and quick-smoke training controls.
- `D2TP/evaluate_model.py`
  Added `--model_type cyclestate`.

# CycleState Design Summary
Compared with the original D2-TPred generator, CycleState currently adds:
1. A queue-state branch:
   weak queue-wave statistics are extracted from lane-level neighbors,
   including queue count, lane density, waiting ratio, release ratio,
   mean lane speed, current light state, elapsed light time, and stop-line distance.
2. A cycle-state branch:
   phase one-hot, elapsed phase time, and phase-change indicators are encoded
   as a compact signal-cycle memory.
3. A hierarchical decoder:
   the decoder is initialized not only by motion and graph interaction features,
   but also by queue-state memory and cycle-state memory.
4. Explicit auxiliary heads:
   queue-state and cycle-state are now predicted by dedicated heads instead of
   directly supervising hidden-state slices.
5. Phase-conditioned state gating:
   queue-state memory and cycle-state memory are no longer only concatenated.
   Their influence is explicitly modulated by the current traffic-light condition.
6. Stronger meso/macro targets:
   queue-state and cycle-state supervision now contain richer traffic-state
   information than the earliest weak-target version.

# Optimization Log
Below is the optimization record from the first prototype to the current version.

## Stage 0: Original Baseline
- Original D2-TPred was kept as the baseline.
- PyTorch compatibility for evaluation was fixed in the base branch:
  device-agnostic execution, safer normalization for short temporal windows,
  and stable evaluation entry.

## Stage 1: Experimental Clone
- A separate experimental copy was created at:
  `/home/lbh/D2-TPred-CycleState`
- This avoids polluting the original reproducible D2-TPred repository.

## Stage 2: CycleState v0 Prototype
- Added `CycleStateTrajectoryGenerator`.
- Reused the original micro trajectory encoder and graph interaction branch.
- Added:
  - lane-level queue-state memory
  - signal-cycle memory
- Verified:
  - forward pass works
  - backward pass works
  - partial warm-start from original `model_best.pth.tar` works

## Stage 3: Quick Smoke Training Support
- Added quick-debug controls in `train.py`:
  - `--max_train_batches`
  - `--max_val_batches`
- This made it possible to rapidly test whether the new branch can learn at all.

## Stage 4: Warm-Start Strategy
- Added a compatible partial checkpoint loader.
- For `cyclestate`, the model now:
  - reuses compatible weights from original D2-TPred
  - keeps new queue/cycle modules randomly initialized
  - starts training from epoch 0 instead of inheriting the old epoch counter

## Stage 5: Generator-Only Stabilization
- Added `--generator_only`.
- Motivation:
  early CycleState experiments should first validate whether queue/cycle memory
  is useful, without GAN instability masking the effect.

## Stage 6: Auxiliary State Supervision
- Added:
  - `--aux_queue_weight`
  - `--aux_cycle_weight`
  - `--gan_weight`
- First implementation supervised queue/cycle hidden states indirectly.
- Current implementation upgrades this to explicit auxiliary prediction heads:
  - `queue_aux_head`
  - `cycle_aux_head`
- This makes the optimization target more interpretable and easier to analyze.

## Stage 7: Generator-Only + Auxiliary Supervision Validation
- Pure generator + auxiliary supervision is more stable than directly mixing GAN.
- In short smoke runs, queue/cycle auxiliary losses decreased clearly.
- Validation ADE/FDE in quick experiments also showed a downward trend, indicating
  that the CycleState branch is trainable and already contributes positive signal.

## Stage 8: Stronger Traffic-State Targets
- The original queue supervision only covered three weak statistics:
  queue count, waiting ratio, and release ratio.
- The current version upgrades the queue targets to a richer meso-state set:
  - queue count
  - waiting ratio
  - release ratio
  - lane queue length
  - stop-line occupancy
  - front-of-queue flag
- The cycle-state branch is also strengthened:
  - phase one-hot
  - elapsed phase time
  - remaining phase time
  - phase-change indicator
- This change pushes the model closer to the intended scientific story:
  the model is no longer only learning trajectory-conditioned features,
  but explicitly learning structured meso/macro traffic states.

## Stage 9: Phase-Conditioned State Gating
- A new optimization is added to avoid naive feature concatenation.
- Motivation:
  the same queue-wave memory should not influence decoding in the same way
  under red, yellow, and green phases.
- Added:
  - queue-state context gate
  - cycle-state context gate
  - decode-time cycle gate
- This makes CycleState more aligned with the target paper story:
  signalized intersections should be modeled as conditional traffic-state systems,
  not only as generic temporal interaction graphs.

## Stage 10: Tuple-to-Context Adapter
- Why:
  the original VTP_C training loop is tuple-based and convenient for fast iteration,
  but it hardcodes dataset semantics into the call path.
- What changed:
  - training and evaluation now build a structured `traffic_context`
  - the `CycleState` generator can directly accept external `traffic_context`
  - the internal model semantics are now organized into:
    - `agent`
    - `signal`
    - `scene`
    - `meso`
- Research-story contribution:
  this supports the claim that the model is learning a full-cycle traffic-state memory,
  not just consuming a fixed dataset tuple layout.
- INT2 readiness:
  future migration can focus on replacing the adapter instead of rewriting the model.

## Stage 11: Stage-Based Training Protocol
- Why:
  earlier experiments showed that directly mixing GAN too early makes it hard to tell
  whether the new traffic-state branches are actually useful.
- What changed:
  - added `--train_stage {warmup, refine, adversarial}`
  - each stage now has a clear default protocol:
    - `warmup`: generator only, no GAN, stronger state supervision
    - `refine`: generator only, no GAN, reduced auxiliary emphasis
    - `adversarial`: reintroduce GAN with smaller weight
  - teacher forcing is also stage-aware instead of staying fixed
- Research-story contribution:
  this makes the optimization process match the intended hierarchy:
  state memory should be learned first, then shown to help trajectory prediction,
  and only then be refined adversarially.

## Stage 12: Structured Queue/Cycle Auxiliary Losses
- Why:
  using one uniform MSE for all queue/cycle targets is too coarse and does not match
  the semantics of the states being supervised.
- What changed:
  - queue supervision is now split into:
    - regression targets
    - binary classification targets
  - cycle supervision is now split into:
    - phase classification
    - elapsed/remaining time regression
    - phase-change classification
  - training logs now expose the structured components explicitly
- Research-story contribution:
  this strengthens interpretability and makes the meso/macro supervision more aligned
  with the paper framing of structured traffic-state modeling.
- Originality contribution:
  the goal is not to add a generic auxiliary trick, but to explicitly encode how
  different traffic-state variables should be learned according to their semantics.

## Stage 13: Controllable Phase-Conditioned Gating Ablation
- Why:
  if `phase-conditioned state modulation` cannot be switched off cleanly,
  then it is difficult to prove whether the mechanism is actually helping.
- What changed:
  - added `--disable_state_gating`
  - queue-state gating, cycle-state gating, and decode-time cycle modulation
    can now be disabled consistently
  - training/evaluation both support this ablation path
- Research-story contribution:
  this turns gating from a hidden implementation detail into a testable scientific claim.

## Stage 14: Experiment Logging Convention
- Why:
  README-level narrative is not enough once training protocol variants multiply.
- What changed:
  - added `EXPERIMENT_LOG.md`
  - standardized the preferred run layout under `experiments/`
  - each run can now keep its own:
    - command
    - protocol stage
    - gating setting
    - auxiliary weights
    - checkpoints
    - best ADE/FDE summary
- Research-story contribution:
  this helps turn the current branch from a one-off prototype into a reproducible
  research development track.
- INT2 readiness:
  a cleaner run protocol makes cross-dataset transfer experiments easier to compare later.

## Stage 15: Phase-Rolling Queue Memory
- Why:
  the previous CycleState decoder only injected queue-state memory once at initialization.
  This is too static for signalized intersections, because queue-wave state keeps evolving
  during the prediction horizon as the phase progresses.
- What changed:
  - added a prediction-time queue rollout branch
  - queue-state is now explicitly rolled forward step by step using:
    - current cycle feature
    - predicted motion offset
    - the last observed meso-state anchor
  - the decoder now consumes the rolled queue memory instead of reusing one fixed queue vector
  - queue rollout predictions are also supervised as part of the structured queue loss
- Research-story contribution:
  this turns CycleState from a model with static meso-state injection into a model with
  phase-evolving meso-state memory.
  In other words, the model is no longer assuming that queue-wave state is frozen when
  prediction starts; it treats queue dynamics as part of the future to be modeled.
- Originality contribution:
  the key idea is not a generic temporal refinement trick, but a signal-conditioned
  meso-state rollout mechanism that is specific to the traffic-light forecasting setting.
- Practical contribution:
  this creates a more natural bridge between:
  - macro cycle progression
  - meso queue-wave evolution
  - micro trajectory decoding

### Current Interpretation After Stage 15
- queue-state is now modeled in two layers:
  - observed queue memory from the history window
  - rolled queue memory across the prediction horizon
- the intended paper narrative becomes stronger:
  `full-cycle traffic-state memory` now includes not only remembering observed traffic state,
  but also propagating that state forward under phase progression.

## Stage 16: Queue Rollout As A Controllable Scientific Variable
- Why:
  after introducing phase-rolling queue memory, it must become a testable variable rather
  than an always-on hidden implementation choice.
- What changed:
  - added `--disable_queue_rollout`
  - training and evaluation can now switch between:
    - dynamic queue rollout
    - static queue-state injection fallback
  - rollout losses remain visible in logs:
    - `QRollReg`
    - `QRollCls`
- Research-story contribution:
  this lets us phrase and test a sharper claim:
  `Does future queue-wave evolution help trajectory prediction beyond static meso-state context?`
- First smoke comparison:
  - rollout `on`: ADE `156.592`, FDE `300.038`
  - rollout `off`: ADE `156.663`, FDE `300.138`
- Current reading:
  the advantage is still very small in this one-batch smoke setup, but the direction is
  favorable and, more importantly, the variable is now experimentally controllable.

## Stage 17: Lane-Consensus Meso Anchor
- Why:
  even with queue rollout, meso-state can still be too agent-local.
  At a signalized intersection, queue-wave state is better viewed as a lane-level consensus
  than as completely independent per-agent context.
- What changed:
  - added `lane_queue_anchor_seq` into `traffic_context`
  - each lane now has a shared meso anchor computed from same-lane agents
  - queue rollout is softly pulled toward this lane-level consensus anchor
  - added `--disable_lane_queue_anchor` so this variable can later be ablated cleanly
- Research-story contribution:
  this strengthens the meso-level claim in the paper story:
  the model is not only rolling individual queue-state estimates, but also respecting
  lane-level collective traffic-state structure.
- Originality contribution:
  this is still not a generic aggregation block; it is an explicit lane-consensus prior
  inserted into the full-cycle meso-state evolution path.
- First smoke observation:
  with lane-consensus anchor enabled, a short warmup smoke run reached:
  ADE `156.591`, FDE `300.036`
  while keeping the new rollout losses active.

## Stage 18: Predictive Lane-Anchor Traceability
- Why:
  once lane-level meso anchors are introduced, they should also become observable during
  the prediction horizon, otherwise they remain only a hidden internal bias.
- What changed:
  - added `lane_queue_rollout_anchor_seq` into model debug outputs
  - this makes the predictive lane-level meso anchor traceable step by step
  - when `--disable_lane_queue_anchor` is used, this trace disappears cleanly
- Research-story contribution:
  now the meso story is not only:
  `the model uses lane-level consensus`
  but also:
  `the model carries lane-level consensus forward during future-state evolution`

## Stage 19: Baseline-Compatible Decoder State Residual
- Why:
  earlier CycleState versions directly enlarged the decoder hidden state by concatenating
  queue/cycle state memories into the decoder core. Although this looked natural from a
  modeling perspective, it broke full warm-start compatibility with the strongest part of
  the original D2-TPred generator: the decoder itself.
- What changed:
  - the CycleState decoder is now kept shape-compatible with the original D2-TPred decoder
  - queue/cycle state is injected through a gated residual pathway instead of directly
    widening the decoder LSTM
  - added `--disable_decoder_state_residual` for explicit ablation
  - added tests to verify:
    - CycleState decoder shape matches baseline
    - original generator checkpoint can be fully loaded with zero skipped decoder keys
- Research-story contribution:
  this change sharpens the scientific claim:
  CycleState should improve trajectory prediction by modulating a strong motion decoder with
  traffic-state memory, not by forcing the model to relearn the entire decoder backbone from scratch.
- Practical contribution:
  this dramatically improves warm-start faithfulness and creates a much cleaner path toward
  stable optimization.

### First Smoke Observation After Stage 19
- `cyclestate + warmup + decoder residual on`:
  ADE `54.784`, FDE `114.138`
- `cyclestate + warmup + decoder residual off`:
  ADE `45.856`, FDE `93.743`

Current reading:
- this is only a 1-batch smoke comparison and cannot yet be treated as a formal performance claim
- however, it clearly shows that once full decoder warm-start compatibility is restored,
  CycleState no longer behaves like the earlier severely degraded prototype
- this confirms that "protecting the baseline decoder" is now a primary optimization axis
  for the next stage of research

### First Smoke Observations Under The New Protocol
- `cyclestate + warmup + aux + gating`:
  ADE `155.554`, FDE `298.113`
- `cyclestate + warmup + aux + no gating`:
  ADE `155.569`, FDE `298.149`
- `cyclestate + warmup + no aux + gating`:
  ADE `155.555`, FDE `298.116`
- `d2tpred baseline quick run`:
  ADE `84.391`, FDE `172.836`

Current reading:
- the new protocol is now executable and reproducible under controlled short runs
- gating shows a small favorable direction in the current tiny smoke setup
- structured auxiliary supervision is now stable, controllable, and analyzable, even though
  its metric gain is not yet visible at this one-batch scale
- the original D2-TPred baseline is still substantially stronger, which is expected before
  CycleState receives longer stabilization/refinement training

# Recommended Current Training Mode
For early-stage CycleState experiments, the recommended mode is:
1. use `--model_type cyclestate`
2. use `--generator_only`
3. warm-start from original `model_best.pth.tar`
4. enable `--aux_queue_weight` and `--aux_cycle_weight`
5. keep GAN disabled until the queue/cycle branches become stable

Example:
```bash
CUDA_VISIBLE_DEVICES=2 python D2TP/train.py \
  --log_dir experiments/cyclestate/warmup_v1 \
  --model_type cyclestate \
  --train_stage warmup \
  --device cuda \
  --pin_memory \
  --resume ./model_best.pth.tar
```

# Next Planned Optimizations
The current CycleState branch is still an early prototype. The next planned
optimizations are:
1. run longer `warmup` / `refine` experiments around `decoder residual on/off`.
2. add evaluation-time analysis for auxiliary state prediction quality.
3. complete multi-factor ablations across:
   - decoder residual
   - queue rollout
   - lane anchor
   - state gating
4. after warmup/refine are stable, reintroduce adversarial training with smaller GAN weight.
5. extend the current context adapter for later INT2 integration without breaking the main model path.
# How was the data collected?
The data in VTP-TL comes from at urban intersections with traffic lights is used to predict vehicles trajectory in different times of day and provides a broad range of real-world driving scenarios. We use drones to hover at 70 to 120 meters above the traffic intersections, as statically as possible, to record vehicle trajectories passing through the area with a bird’s-eye view in the daytime of the non-rush hours, rush hours, and the evening.


<div align=center>
<img src="https://github.com/VTP-TL/D2-TPred/blob/main/drone.png" width="780" height="312" alt=" "/><br/>
</div>

# Where was the data collected?
We choose 3 different traffic intersections, including crossroad, T-junction, and roundabout scenarios. In these scenario, they own the different number of roads and traffic lights, and cause to different movement behaviors for vehicles.

<div align=center>
<img src="https://github.com/VTP-TL/D2-TPred/blob/main/scenarios.png" width="762" height="628" alt=" "/><br/>
</div>

# Summary of the Dataset 
In the [VTP-TL](https://pan.baidu.com/s/1gAdWP58RCKl0RrsvtQotpw) dataset, we have collected data from 3 different categories of traffic scenarios using drones. The summary of the data is listed in the following table. 

<div align=center>
<img src="https://github.com/VTP-TL/D2-TPred/blob/main/summary.png" width="772" height="503" alt=" "/><br/>
</div>

# Included Materials
For the 3 recording scenarios, we include 2 files for each scenarios: 
1. The sample of video clips (xxx.mp4) 
2. Recorded vehicle trajectory file (xxx.txt) 
where, we provide trajectories information in pixel.

# Recorded Vehicle Trajectory files (xxx.txt)
**F_id:** column 1. For each agent (per Agent_id), frame_id represents the frames the agent appears in the video.    
**A_id:** column 2. For each xxx.txt file, the Agent_id starts from 0, and represent the ID of the agent.   
**x:** column 3, the x position of the agent at each frame. The unit is pixel.     
**y:** column 4, the y position of the agent at each frame. The unit is pixel.   
**Lane_id:** column 5, For each xxx.txt file, the Lane_id starts from 0, and represent the ID of the traffic lane.   
**pa:** column 6, For each xxx.txt file, the inperception is set as 0 or 1, and represent whether vehicle locates in the influencing area of traffic light.   
**f:** column 7, For each xxx.txt file, the isfirstobj is set as 0 or 1, and represent whether vehicle is the first agent in the influencing area of traffic light.   
**Lig_id:** column 8, For each xxx.txt file, the Lig_id starts from 0, and represent the ID of the traffic light.   
**ls:** column 9, For each xxx.txt file, the Ls is set as 0, 1 and 2, and represents the state of traffic light.   
**mb:** column 10, For each xxx.txt file, the Mb is set as 0, 1 and 2, and represents the movement behaviors of vehicle.   
**lt:** column 11, For each xxx.txt file, the Ldurtime represents the durtime of traffic light.   

**Example:**
<div align=center>
<img src="https://github.com/VTP-TL/D2-TPred/blob/main/smaple.png" alt=" "/><br/>
</div>
