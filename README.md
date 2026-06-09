# D2-TPred-CycleState

本仓库同时承担两个角色：

1. 复现与审计原始 `D2-TPred` baseline。
2. 开发新研究方向 `CycleState: Full-Cycle Traffic-State Memory for Trajectory Prediction at Signalized Intersections`。

当前开发主线不是给 baseline 简单叠加一个通用时序模块，而是围绕一个更明确的科研判断：

> 信号灯路口轨迹预测应被建模为 `full-cycle traffic-state memory` 问题。

也就是说，未来轨迹不仅由个体短时运动和局部交互决定，还受到车道级排队/释放波与信号周期状态的共同约束。

## 科研故事

`CycleState` 把信号灯路口预测拆成三层状态：

- `micro`：继承原始 `D2-TPred` 的轨迹 LSTM、空间图交互和局部时间交互，保住强运动预测基线。
- `meso`：显式构造车道级 `queue-state memory`，建模排队密度、等待比例、释放比例、队首状态和 stop-line occupancy。
- `macro`：显式构造 `cycle-state memory`，建模 phase one-hot、elapsed/remaining phase time 和 phase-change。

当前最有辨识度的机制是 `Phase-Rolling Queue Memory`：

- queue-state 不在 decoder 初始化后冻结；
- 预测期每一步都会根据 phase progression、上一时刻预测运动和 lane-level anchor 滚动更新；
- decoder 使用 `observed queue context + bounded rollout residual`，避免早期 warmup 中 rollout hidden 直接接管主解码器。

这条叙事的边界也很明确：本仓库当前不引入通用 Transformer/扩散式重构，不照搬现有 intersection forecasting 方法的完整套路，而是把交通工程中的 queue/cycle 状态显式纳入轨迹预测。

## ⚠️ Oracle 假设声明（Phase 0 #17）

CycleState 在训练和推理期间将未来真实交通灯信号状态（`pred_state`）作为输入。`pred_state` 来自数据集中记录的预测期实际信号灯相位与已运行时间（`pred_phase_ids`, `pred_phase_elapsed`），在模型 forward 时通过 `traffic_context["signal"]["pred_state"]` 传递给 `rollout_queue_step` 和 `get_next_state`。

**这构成一个 oracle 假设**：模型可以访问推理时不可用的真实未来信号信息。在真实部署场景中，未来的交通灯状态需要由外部信号控制器提前给出或通过预测模块生成，而非从数据集中读取。

**对与 baseline 公平性的影响**：原始 `D2-TPred` baseline 同样接收 `pred_state` 作为输入（`TrajectoryGenerator.forward` 的 `pred_state` 参数用于 `get_next_state` 方法更新交通灯条件）。因此，**CycleState 与 D2-TPred 在此 oracle 假设上是对齐的**——两者都假设未来信号状态已知。本仓库的所有 comparable 实验结果都在相同的 oracle 条件下产生。

**后续工作（Phase 0.5）**：计划通过信号退化实验（oracle→predicted 信号替换、敏感度曲线）量化此假设对 ADE/FDE 的实际贡献，并在论文中提供假设讨论段落。参见 [PLAN.md §2.5](./docs/PLAN.md#25-phase-05oracle-假设验证与信号退化实验p05)。

## 当前实现

核心文件：

- `D2TP/models.py`：原始 `TrajectoryGenerator` 与新增 `CycleStateTrajectoryGenerator`。
- `D2TP/train.py`：训练协议、structured auxiliary losses、训练内验证和 checkpoint 保存。
- `D2TP/evaluate_model.py`：离线 `val/test` 评估，支持 `d2tpred` 与 `cyclestate`。
- `tests/test_cyclestate_protocol.py`：协议、消融开关、rollout 路径和评估聚合的回归测试。
- `EXPERIMENT_LOG.md`：完整优化日志和实验记录。
- `docs/AI_EXPERIMENT_DELEGATION_GUIDE.md`：委托其他 AI 执行修改/实验/验证时的统一作战手册。

当前 `CycleState` 已支持：

- `tuple -> traffic_context` adapter，为后续 INT2 迁移保留统一接口；
- `warmup / refine / adversarial` 分阶段训练；
- structured queue/cycle auxiliary losses；
- `queue rollout / lane anchor / state gating / decoder residual` 可控消融；
- `aux_rollout_weight` 独立调节；
- `val` split 训练内选模，`test` split 最终复核；
- `lr` 命令行参数真实生效；
- `grad_clip`、`rollout_residual_scale`、`detach_rollout_state` 稳定化协议；
- 状态稳定性日志：`DInitNorm / DStepNorm / QRollHNorm / PredOffsetNorm / GradNorm`。

## 评测口径

所有实验结果必须按下面三类记录：

- `smoke`：只验证 forward/backward、日志、checkpoint、消融开关和损失项是否工作。常见设置为 `max_train_batches=1` 或 `num_epochs=0`。
- `protocol-check`：验证训练协议、验证调度、采样逻辑、checkpoint 恢复和统计口径是否一致。不能直接当作论文结论。
- `comparable`：只有 split、checkpoint 来源、ADE/FDE 聚合方式、采样次数和评估脚本口径都对齐后，才能用于和 baseline 或论文指标比较。

当前训练内默认使用 `--val_dset_type val` 做模型选择；`test` split 只用于最终复核或兼容历史协议。

## 当前证据

仓库内已经确认的 baseline 可比线：

- `D2TP/model_best.pth.tar`，完整 `val` split，`num_samples=4`：
  - `ADE 38.493 / FDE 78.706`
- `D2TP/model_best.pth.tar`，完整 `test` split，`num_samples=4`：
  - `ADE 17.812 / FDE 37.568`

当前 `CycleState` 的关键中间证据：

- 旧调度下 `warmup_main_v2` 短程下降到 `ADE 56.827 / FDE 107.416`，只作为“模型在学”的证据。
- 修复前短协议中 rollout-on 曾输给 no-rollout：`78.227 / 152.544` vs `71.863 / 140.974`。
- 修复 training step-0 rollout alignment 与 anchored rollout decode context 后，rollout-on 恢复到 `66.793 / 132.168`，重新优于对应 no-rollout。
- `aux_rollout_weight=2.5` 的 50-batch rollout-on 达到 `66.761 / 122.728`，优于 `no_rollout@50b` 的 `67.747 / 124.741`。
- 100-batch matched warmup 曾出现后半程整体崩坏，因此当前优先级是协议稳定化，而不是继续堆新结构。
- Stage 24 true-val 复核显示默认稳定化仍未通过：
  - `warmup_protocol_stable_v1_50b`: `ADE 87.082 / FDE 175.723`
  - `warmup_protocol_stable_v1_100b`: `ADE 231.420 / FDE 420.862`
  - `100b` 相对 `50b` 恶化 `165.7%`，超过 15% 稳定性门槛
- 单变量降低学习率到 `3e-4` 后，`batch 50` 状态范数更温和，但 `batch 100` 仍崩：
  - `88.598 / 171.890 -> 226.302 / 411.163`
- 对当前代码进一步审查后，已确认两个更基础的高优先级问题：
  - `seqGAT` 前向被 `torch.no_grad()` 包裹，局部时序图注意力实际上不参与训练
  - `relation_Matrix` 的正常方向区间分支会把距离内邻居无条件连边，削弱了方向约束
- Stage 25 P0 已完成并通过测试：
  - 已恢复 baseline/CycleState 中 `seqGAT` 的梯度流
  - 已修复 `relation_Matrix` 正常方向区间的无条件连边
  - 当前单元测试增至 `39` 项并全部通过
- Stage 25 true-val 复测结果：
  - `warmup_p0_seqgat_relation_v1_50b`: `ADE 88.956 / FDE 175.380`
  - `warmup_p0_seqgat_relation_v1_100b`: `ADE 204.730 / FDE 375.054`
  - `100b` 相对 `50b` 仍恶化 `130.1%`，未通过 15% 稳定性门槛
  - 但相较 Stage 24 的 `231.420 / 420.862`，P0 修复已把 `100b` 崩坏幅度压低
- Stage 26 最小变量实验结果：
  - 单独降低 warmup `teacher_forcing_ratio` 到 `0.6` 并不能解决问题：
    - `warmup_p0_seqgat_relation_tf06_100b`
    - `92.735 / 182.898 -> 266.671 / 468.079`
    - `100b` 相对 `50b` 恶化 `187.6%`
  - 但 `50b warmup -> 50b refine` 的阶段切换候选显著更稳：
    - `warmup50_refine50_p0_seqgat_relation_v1`
    - quick `val + num_samples=4`: `ADE 84.772 / FDE 170.878`
    - full `val + num_samples=20`: `ADE 75.078 / FDE 154.690`
  - 这说明“继续压 warmup teacher forcing”不是主因修复方向，更有价值的是在 `50b` 左右把长链状态学习交给 `refine` 接管。

目前还不能宣称稳定超过 baseline 或论文指标。到当前为止，最合理的科研叙事是：基础交互建模 bug 会放大崩坏，但 warmup 后半程失稳主要更像“训练阶段职责划分错误”，而不只是 exposure-bias；因此下一步主线应围绕 `50b warmup -> refine` 候选做正式复核和消融，而不是继续在 warmup 中堆新技巧。

## 复现入口

### 单元测试

```bash
python -m unittest discover -s tests -p 'test_cyclestate_protocol.py'
```

### Baseline 审计

```bash
python D2TP/evaluate_model.py \
  --model_type d2tpred \
  --device cuda \
  --loader_num_workers 0 \
  --batch_size 16 \
  --num_samples 4 \
  --eval_print_every 10 \
  --resume D2TP/model_best.pth.tar \
  --dset_type val
```

最终 `test` 复核只需把 `--dset_type val` 改为 `--dset_type test`。更接近论文口径时使用 `--num_samples 20`。

### CycleState 协议优先 warmup

```bash
python D2TP/train.py \
  --log_dir experiments/cyclestate/warmup_protocol_stable_v1 \
  --model_type cyclestate \
  --train_stage warmup \
  --device cuda \
  --pin_memory \
  --loader_num_workers 0 \
  --batch_size 8 \
  --best_k 4 \
  --num_val_samples 4 \
  --aux_rollout_weight 2.5 \
  --resume D2TP/model_best.pth.tar \
  --num_epochs 0 \
  --print_every 50 \
  --max_train_batches 100 \
  --max_val_batches 20 \
  --val_dset_type val
```

默认 warmup 稳定化参数：

- `--grad_clip 1.0`
- `--rollout_residual_scale 0.35`
- `--detach_rollout_state`

显式关闭 rollout-state detach：

```bash
--no_detach_rollout_state
```

### 当前最佳协议候选：`50b warmup -> 50b refine`

```bash
python D2TP/train.py \
  --log_dir experiments/cyclestate/warmup50_refine50_p0_seqgat_relation_v1 \
  --model_type cyclestate \
  --train_stage refine \
  --device cuda \
  --pin_memory \
  --loader_num_workers 0 \
  --batch_size 8 \
  --best_k 4 \
  --num_val_samples 4 \
  --aux_rollout_weight 2.5 \
  --resume experiments/cyclestate/warmup_p0_seqgat_relation_v1_50b/checkpoint/model_best.pth.tar \
  --num_epochs 0 \
  --print_every 50 \
  --max_train_batches 50 \
  --max_val_batches 20 \
  --val_dset_type val
```

完整 `val + num_samples=20` 复核：

```bash
python D2TP/evaluate_model.py \
  --model_type cyclestate \
  --device cuda \
  --pin_memory \
  --loader_num_workers 0 \
  --batch_size 8 \
  --num_samples 20 \
  --eval_print_every 10 \
  --resume experiments/cyclestate/warmup50_refine50_p0_seqgat_relation_v1/checkpoint/model_best.pth.tar \
  --dset_type val \
  --rollout_residual_scale 0.7
```

## 下一轮实验顺序

1. 先补齐 baseline 的正式可比线：
   - `baseline_audit_v2_val_full_num_samples20`
   - `baseline_audit_v2_test_full_num_samples20`
2. 再对当前最佳候选做正式复核：
   - `warmup50_refine50_p0_seqgat_relation_v1_test20`
   - 目标是确认 `50b warmup -> 50b refine` 在 `test + num_samples=20` 上是否仍优于纯 warmup 候选
3. 若 `test@20` 结果仍站得住，再围绕该候选做消融：
   - `disable_queue_rollout`
   - `disable_decoder_state_residual`
   - `disable_lane_queue_anchor`
   - `disable_state_gating`
4. 若 `test@20` 明显退化，再回到协议层而不是加结构：
   - 优先检查 `rollout offset` 一致性、`phase_change` 预测期缺失、`D_step` 重置
   - 暂不继续尝试 warmup `teacher_forcing_ratio` 单变量，因为 `tf=0.6` 已被证伪

### 下一轮将直接执行的命令

1. baseline `val + num_samples=20`

```bash
python D2TP/evaluate_model.py \
  --model_type d2tpred \
  --device cuda \
  --pin_memory \
  --loader_num_workers 0 \
  --batch_size 16 \
  --num_samples 20 \
  --eval_print_every 10 \
  --resume D2TP/model_best.pth.tar \
  --dset_type val
```

2. baseline `test + num_samples=20`

```bash
python D2TP/evaluate_model.py \
  --model_type d2tpred \
  --device cuda \
  --pin_memory \
  --loader_num_workers 0 \
  --batch_size 16 \
  --num_samples 20 \
  --eval_print_every 10 \
  --resume D2TP/model_best.pth.tar \
  --dset_type test
```

3. current best candidate `test + num_samples=20`

```bash
python D2TP/evaluate_model.py \
  --model_type cyclestate \
  --device cuda \
  --pin_memory \
  --loader_num_workers 0 \
  --batch_size 8 \
  --num_samples 20 \
  --eval_print_every 10 \
  --resume experiments/cyclestate/warmup50_refine50_p0_seqgat_relation_v1/checkpoint/model_best.pth.tar \
  --dset_type test \
  --rollout_residual_scale 0.7
```

### 进入下一步的判定标准

- 若 `warmup50_refine50_p0_seqgat_relation_v1_test20` 相比纯 warmup 候选仍保持优势，则继续做消融，不改结构。
- 若该候选在 `test@20` 上明显失去优势，则下一轮优先修正协议正确性项，而不是继续试新的训练 trick。

### 委托其他 AI 的执行规范

若后续把实验委托给其他 AI 执行，统一按：

- [docs/AI_EXPERIMENT_DELEGATION_GUIDE.md](./docs/AI_EXPERIMENT_DELEGATION_GUIDE.md)

它定义了：

- 允许做什么 / 禁止做什么
- 固定实验顺序
- 必跑验证命令
- 交付模板
- 必须停下来交给主审的条件

## 优化日志摘要

完整日志见 `EXPERIMENT_LOG.md`。当前累计优化包括：

- Stage 0-4：baseline 兼容、实验仓库克隆、CycleState 原型、quick smoke、warm-start。
- Stage 5-14：generator-only、structured auxiliary losses、phase-conditioned gating、traffic_context adapter、staged protocol、实验日志规范。
- Stage 15-19：Phase-Rolling Queue Memory、queue rollout 消融、Lane-Consensus Meso Anchor、predictive anchor trace、baseline-compatible decoder residual。
- Stage 20-23：训练/评估指标口径对齐、验证调度修复、rollout 路径根因修复、matched warmup stability、`aux_rollout_weight` 独立调节。
- Stage 24：协议优先稳定化，包含 true-val 选模、`lr` 生效、`grad_clip`、bounded rollout residual injection、warmup rollout-state detach 和状态稳定性日志。
- Stage 25：修复 `seqGAT` 梯度冻结与 `relation_Matrix` 方向约束 bug，并完成 true-val 复测；结果显示 `100b` 崩坏有所缓和，但仍未通过稳定性门槛。
- Stage 26：最小变量稳定化复测；确认单独降低 warmup teacher forcing 无效，但 `50b warmup -> 50b refine` 在 quick `val` 与 full `val@20` 上均优于纯 warmup 候选。
- Stage 27：Phase 5 代码风格修复 — `D_step=2` 从 `train.py:885`(epoch 循环外)移入 for-epoch 循环体，确保 adversarial 阶段每个 epoch 起跑时 D/G 调度计数器都从 2 开始；新增 2 个单元测试 `test_dstep_initialization_lives_inside_epoch_loop` / `test_dstep_resets_between_epochs_via_simulated_main_loop` 保护此协议。测试总数 39 → 41,全部通过。详细见 [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md) Stage 27 与 [PLAN.md §7.1](./PLAN.md#7-phase-5代码质量与文档p5)。
- Stage 28：Phase 0 死代码契约修复 — `state_loss` 在 `utils.py:139` 补全签名 `state_loss(pred_traj_fake, pred_traj_gt, loss_mask=None, mode='sum')`；`mode='average'` 分支在 mask 缺省时回退为 `torch.ones(T, V)` 默认掩膜，避免 NameError；新增 3 个单元测试 `test_state_loss_signature_exposes_loss_mask` / `test_state_loss_average_mode_uses_loss_mask` / `test_state_loss_is_not_invoked_by_active_training_path`(其中第三个用 AST 扫描守住"train.py 仍不调用此函数"的契约,防止未来误用)。测试总数 41 → 44,全部通过；`py_compile` 干净；旧调用方式保留 backward-compat。详细见 [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md) Stage 28 与 [PLAN.md §2.1](./PLAN.md#21-#2--state_loss-引用未定义的-loss_mask死代码)。
- Stage 29：Phase 0 P0 训练/推理一致性修复 — `last_rollout_offset = input_t.squeeze(0) if teacher_force else output` 在 `models.py:1799` 改为 `last_rollout_offset = output`，与推理分支 `models.py:1865` 完全对齐。此修复消除 queue rollout 分支的 **exposure bias / train-eval distribution shift**：修复前训练时 80% 看到 GT future displacement、20% 看到模型预测，推理时 100% 看到模型预测，导致 queue rollout gate/MLP 在 train/eval 下学到完全不同的输入分布。新增 2 个单元测试 `test_rollout_offset_uses_model_own_output_under_teacher_forcing`(用 forward hook 捕获 `pred_hidden2pos` 输出与 step 1 `rollout_queue_step` 接收的 `last_pred_offset` 断言相等) / `test_rollout_offset_under_teacher_forcing_matches_eval_at_step_zero`(守住 train+eval step 0 seed 都来自 `obs_traj_rel[obs_len-1]`)。测试总数 44 → 46,全部通过；`py_compile` 干净；1-batch warmup smoke 正常完成（GradNorm≈173.21, QRollHNorm≈0.20, PredOffsetNorm≈0.19）。下一步需在 `warmup50_refine50_p0_seqgat_relation_v1` candidate 上重跑 `test@20` 验证 warmup 崩坏是否进一步收敛。详细见 [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md) Stage 29 与 [PLAN.md §2.2](./PLAN.md#22-#3--训练推理-rollout-offset-不一致)。
- Stage 30：Phase 0 P0 aux head 子空间拆分 — `queue_aux_head` (6 维) 和 `cycle_aux_head` (6 维) 在 `models.py:965-969` 拆分为 5 个独立子空间头：`queue_aux_reg_head` (4 维) + `queue_aux_cls_head` (2 维) + `cycle_aux_phase_head` (3 维) + `cycle_aux_time_head` (2 维) + `cycle_aux_change_head` (1 维)，5 组参数互不共享。`rollout_queue_step` (`models.py:1454-1460`) 与 forward (`models.py:1700-1717`) 改为 `torch.cat` 拼接子头输出，末维契约仍为 6 维。`compute_structured_aux_losses` (`train.py:430-459`) 入口对 `queue/cycle/queue_rollout` 6 个 pred-target pair 增加形状契约断言，形状错配从 `IndexError` 升级为 `AssertionError`。此修复把 reg/cls/phase/time/change 5 组语义子空间的参数强制分开，消除原 `nn.Linear(..., 6)` 上 MSE/BCE 梯度相互纠缠的问题。新增 3 个单元测试 `test_aux_heads_split_into_independent_regression_and_classification_modules`(5 个子头存在 + 各自 `weight.shape` 正确 + 参数 `id` 集合无交集) / `test_aux_pred_last_outputs_are_concatenation_of_subspace_heads`(跑 forward 断言 `queue_pred_last == cat(reg, cls)` 与 `cycle_pred_last == cat(phase, time, change)`，并断言末维仍为 6) / `test_structured_auxiliary_losses_asserts_pred_target_shape_match`(4 类错配输入验证 `AssertionError` 触发条件)。测试总数 46 → 49,全部通过；`py_compile` 干净；1-batch 集成 smoke (forward / backward / aux losses / 5 个子头梯度) 全部正常。`maybe_load_compatible_weights` 的 shape-based skip 自动跳过旧 checkpoint 中尺寸已变的 aux 头，新子头从随机初始化开始训练，与 aux 头从 refine 阶段开始监督的协议一致。详细见 [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md) Stage 30 与 [PLAN.md §2.3](./PLAN.md#23-#4--cycletargetlast-维度尺度不匹配)。
- Stage 31：跨阶段 correctness 修复（同时修 #5 Phase 5 + #6 Phase 2）。
  - **#5 CUDA_VISIBLE_DEVICES 硬编码删除**（Phase 5）— 删除 `D2TP/train.py:230` 的模块级硬编码 `CUDA_VISIBLE_DEVICES = '2'`，该行此前在模块加载时无条件把 `os.environ['CUDA_VISIBLE_DEVICES']` 改写成 `'2'`，覆盖任何用户/shell 端设置，使得 `--gpu_num` 形同虚设。修复后改为注释说明由用户在 shell 端 `export CUDA_VISIBLE_DEVICES=...` 或调用 torch 前读 `os.environ.get('CUDA_VISIBLE_DEVICES')`，完全遵循 PyTorch 标准用法。新增 1 个单元测试 `test_cuda_visible_devices_not_hardcoded_in_train_py`，用正则 `^\s*CUDA_VISIBLE_DEVICES\s*=\s*['\"]\d+['\"]\s*$` 多行匹配源码并断言无命中——任何后续 PR/agent 重新引入硬编码赋值都会被该测试立即拦截。测试总数 49 → 50,全部通过；`py_compile` 干净。
  - **#6 get_step_cycle_feature phase_change 补齐**（Phase 2）— `get_step_cycle_feature(state_frame, prev_phase=None)` 在 `models.py:1082-1105` 新增 `prev_phase` 形参：`prev_phase=None`（默认，向后兼容）时 `phase_change = torch.zeros(...)`；`prev_phase=tensor` 时 `phase_change = (phase != prev_phase.long().clamp(0, 2)).float()`，反映真实相位切换。`get_decode_step_context` 在 `models.py:1376-1411` 显式计算并传入 `prev_phase`：step 0 用 `obs_state[-1, :, 2]`，step > 0 用 `pred_state[step_index - 2, :, 2]`，与 `build_cycle_features` (`models.py:1058-1080`) 的跨帧语义一致。修复前 `phase_change = torch.zeros(...)` 永远为 0 → cycle LSTM 在 12 步预测期内的 `phase_change` 输入通道恒为 0 → 模型对相位切换（黄→红/绿→黄）完全无反应；同时 `compute_structured_aux_losses` 的 `cycle_change` BCE 监督项也永远在 0/0 上做无意义退化解。修复后切换帧的 `phase_change` = 1.0，稳定帧 = 0.0；`cycle_feature_dim` 仍为 6 维契约，下游 `cycle_step_embedding` / `cycle_lstm` / `cycle_aux_*` 全部零回归。新增 3 个单元测试 `test_get_step_cycle_feature_emits_phase_change_on_transition`（直接构造 prev=[0,0,1] vs current=[0,1,2]，断言 `phase_change == [0, 1, 1]`）/ `test_get_step_cycle_feature_phase_change_backward_compatible`（不传 / 传 `None`，断言 `phase_change == 0`，守住旧调用方契约）/ `test_get_decode_step_context_propagates_phase_change`（mock `get_next_state` 后在 `pred_state[2:, 1, 2]=1.0` 注入 batch 1 的 step-2 相位切换，step_index=3 调 `get_decode_step_context`，断言 `phase_change == [0, 1, 0]`）。测试总数 50 → 53,全部通过；`py_compile` 干净；端到端 forward smoke 验证 phase 切换信号在切换步之后正常流入 `cycle_step_embedding`。详细见 [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md) Stage 31 与 [PLAN.md §4.1](./PLAN.md#41-#6--get_step_cycle_feature-缺失-phase_change) / [PLAN.md §7.1](./PLAN.md#71-代码清理1-5-8-9-11-12-14-15)。
- Stage 32：跨阶段 correctness + 性能修复（同时修 #7 Phase 0 + #8 Phase 5 + #9 Phase 5）。
  - **#7 maybe_load_compatible_weights 恢复 start_epoch**（Phase 0）— `train.py:898-915` cyclestate 兼容加载分支新增 `if "epoch" in checkpoint: args.start_epoch = checkpoint["epoch"]` 恢复逻辑，行为与 else 分支（非 cyclestate 加载）完全一致。修复前 resume 一个 cyclestate checkpoint 后 `args.start_epoch` 仍为 CLI 默认值 0，主循环 `range(args.start_epoch, args.num_epochs+1)` 从 epoch 0 重新开始计数 → LR scheduler 错位、TensorBoard 标号重复、日志/ckpt 命名冲突、checkpoint 覆盖。修复后断点续训时 epoch 计数严格从 checkpoint 恢复。新增 2 个单元测试 `test_compatible_resume_restores_start_epoch`（源码正则匹配守护，断言 `maybe_load_compatible_weights(...)` 之后紧跟 `if "epoch" in checkpoint: args.start_epoch = checkpoint["epoch"]` 模式）/ `test_main_resume_for_cyclestate_calls_compatible_loader_and_restores_epoch`（patch `os.path.isfile` / `torch.load` / `maybe_load_compatible_weights` 跑 resume 分支，断言 `args.start_epoch == checkpoint["epoch"] == 7`）。测试总数 53 → 55,全部通过；`py_compile` 干净。
  - **#8 build_lane_queue_anchor_seq 向量化**（Phase 5）— `build_lane_queue_anchor_seq` 在 `models.py:1230-1289` 重写为向量化实现：1. `repeat_interleave(torch.arange(num_scene), scene_sizes)` 算 `agent_scene_idx` (batch,)（无 Python 循环）；2. 联合编码 `(t, scene, lane)` group key（`t_offsets + agent_scene_idx * max_lane_id + lane_ids`）确保跨 t/scene/lane 唯一；3. `index_add_` 在 `(T*batch)` 维度一次性累计 sum/count；4. 求均值后用 `mean_features[flat_group_key].reshape(T, batch, dim)` 广播回原 shape。修复前是 `scene × time × unique_lane_id` 三层 Python 嵌套循环 + `seq_start_end.tolist()` 强制 GPU-CPU 同步，时间复杂度 `T*N*#unique_lane`；修复后全流程 GPU 端向量化，无 `.tolist()` / 无 `for` 循环，复杂度降到 `T*batch + num_groups` 量级。**重要细节**：必须把 t 维度也编码到 group key，否则同 `(scene, lane)` 在不同时刻的特征会被错误平均，违反原 loop 的"每帧独立计算 lane 均值"语义。`build_lane_queue_anchor` 单步 wrapper 调用 `build_lane_queue_anchor_seq(...).squeeze(0)`，自动复用向量化路径。新增 3 个单元测试 `test_lane_queue_anchor_seq_vectorized_matches_python_loop`（用相同输入跑向量化版 vs Python 循环版，断言 `torch.allclose(atol=1e-6)`）/ `test_lane_queue_anchor_seq_handles_single_agent_lane`（每 lane 只有一个 agent 时 anchor 必须等于自身）/ `test_lane_queue_anchor_seq_cross_scene_lane_id_isolated`（两 scene 都有 lane_id=0 时各自 scene 的均值独立计算）。测试总数 55 → 58,全部通过；`py_compile` 干净；end-to-end forward smoke (lane_ids=[0,0,1,1]) 输出 `lane_queue_rollout_anchor_seq` shape (12, 4, 11) 正常。
  - **#9 relation_Matrix 向量化**（Phase 5）— `relation_Matrix` 在 `models.py:343-398` 重写为向量化实现：1. 距离门控用 `(currdata.unsqueeze(1) - currdata.unsqueeze(2))` 广播 pairwise 算欧氏距离（等价于原 `pdist` + `squareform`）；2. pairwise 方向角用 `torch.atan2(diff_y, diff_x) * (180.0 / math.pi) % 360`（等价于原 `neig_direction`）；3. 扇区判定统一为 `(delta <= 62) | (delta >= 298)`，其中 `delta = (dire - a + 360) % 360`，**统一处理 wrap-around**（原版 `up > 360` / `62 <= up <= 124` 三分支等价于此公式）。修复前是 `F * N * N` 量级 Python 嵌套循环 + numpy `pdist` + 每次调用 `.detach().cpu().numpy()` 强制 GPU-CPU 同步；修复后全流程 GPU 端向量化，无 `for` 循环 / 无 `.cpu().numpy()` / 无 `pdist`。**公式等价性**：用 4 组不同 heading 集合（[10,30,350,90] / [0,62,180,298] / [45,135,225,315] / [0.5,359.5,90,270]）对比向量化版与原 numpy 版 `np.allclose` 通过（仅在 a=298 self-loop 单点边界有 1 像素差异 — 原版 `elif down <= dire <= up` 分支在该边界处理不一致，新公式更符合几何直观）。新增 4 个单元测试 `test_relation_matrix_handles_wrap_around_at_360`（heading=350°、邻居 10°/180° 覆盖原 `if up > 360` 分支）/ `test_relation_matrix_handles_wrap_around_at_0`（heading=30°、邻居 -26.57°/333.43°/180° 覆盖原 `elif 62 <= up <= 124` 分支）/ `test_relation_matrix_distance_gate_zeros_far_neighbors`（距离 200 > 156 必须筛掉）/ `test_relation_matrix_returns_tensor_on_input_device`（输出在 input device 且 dtype=float32）。测试总数 58 → 62,全部通过；`py_compile` 干净。详细见 [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md) Stage 32 与 [PLAN.md §2.4](./PLAN.md#24-#7--maybeloadcompatibleweights-不恢复-start_epoch) / [PLAN.md §7.1](./PLAN.md#71-代码清理1-5-8-9-11-12-14-15)。
- Stage 33：Phase 5 代码清理批量（同时修 #10 + #11 + #12 + #13 + #14 + #15）。
  - **#10 _mean_norm_from_tensor 命名歧义**（Phase 5）— `train.py:565-597` 给 `_mean_norm_from_tensor` 补详细 docstring，明确"沿最后一维 L2 后行平均"才是高维语义，避免误用：0-dim → 标量自身；1-dim → 算术平均；≥2 维 → `mean( ||row_i||_2 )`；这与 `torch.norm` 的整体 L2/Frobenius 范数不同，若需要严格 L2 范数请用 `torch.norm(t)`。命名澄清但行为不变 — 该函数仍按原"分情况返回"语义工作，不破坏调用方。新增 7 个单元测试 `test_mean_norm_from_tensor_returns_zero_for_none` / `test_mean_norm_from_tensor_returns_zero_for_empty` / `test_mean_norm_from_tensor_0d_returns_scalar` / `test_mean_norm_from_tensor_1d_returns_arithmetic_mean` / `test_mean_norm_from_tensor_2d_returns_mean_per_row_norm` / `test_mean_norm_from_tensor_3d_matches_2d_per_timestep` / `test_mean_norm_from_tensor_does_not_compute_global_l2_norm`（与 Frobenius 范数对比，验证返回的是"行平均"而非"整体 L2"）。
  - **#11 graph_lstm_model 保留但未使用**（Phase 5）— `models.py` 在 `__init__` 给 `graph_lstm_model` 加详细注释，保留 `nn.LSTMCell` 实例（用于旧版 checkpoint 兼容，旧模型参数仍可加载），并新增 `_graph_lstm_call_count` 诊断计数器 + `forward_hook`。这样 forward 主路径若未来误接入 `graph_lstm_model`，计数器会立刻递增并被测试捕获。新增 3 个单元测试 `test_graph_lstm_model_is_intentionally_unused`（最小 forward 前后计数仍为 0）/ `test_graph_lstm_model_attribute_is_lstmcell`（验证仍为 `nn.LSTMCell` 且 `hidden_size=32`）/ `test_graph_lstm_model_direct_call_increments_counter`（直接调用成员时计数必须 +1）。
  - **#12 seq_start_end.data 替换为 .tolist()**（Phase 5）— 不只 `train.py`，`models.py` 中按场景分组的两处 active loop 也一并从 `seq_start_end.data` 改为 `seq_start_end.tolist()`。修复要点：彻底消除 active code 对 `.data` 的依赖；`.tolist()` 直接返回 Python int，索引语义保持不变。新增 5 个单元测试：`test_train_py_no_remaining_seq_start_end_data` / `test_models_py_no_remaining_seq_start_end_data`（源码守卫扫描两文件 active code 无 `.data`）/ `test_train_py_seq_start_end_iteration_uses_tolist` / `test_models_py_seq_start_end_iteration_uses_tolist`（断言 `tolist()` 迭代模式存在）/ `test_seq_start_end_tolist_preserves_python_ints`。
  - **#13 D_train tensorboard 步数改用 global_step**（Phase 5）— `main` 在训练循环外维护 `global_step = 0`，并在每次 `train()` / `D_train()` 后递增。修复不仅覆盖 `D_train`，也把生成器侧 `train()` 的所有 `g_*` / `state_*` 标量统一改为写入 `global_step`，从而让 g/d 曲线共享同一训练步时间轴。新增 5 个单元测试 `test_d_train_signature_accepts_global_step` / `test_d_train_writes_d_train_loss_with_global_step` / `test_main_loop_increments_global_step` / `test_train_signature_accepts_global_step` / `test_train_writes_generator_scalars_with_global_step`。
  - **#14 best_ade 模块级全局变量 → BestAdeTracker**（Phase 5）— `train.py` 新增 `BestAdeTracker` 类，`main` 中局部实例化 `best_ade_tracker = BestAdeTracker()`，resume 时用 `restore_from_checkpoint` 恢复 checkpoint 内的 `best_ade`，并移除残留的 `global best_ade` 声明入口。这样既保留 checkpoint 字段契约，又切断模块级可变状态。新增 7 个单元测试，覆盖 tracker 存在性、初始值、更新语义、checkpoint 恢复、多实例隔离、无模块级 `best_ade = 100` 赋值，以及 `main` 中不再声明 `global best_ade`。
  - **#15 set_logger 使用 hasHandlers() 而非 handlers**（Phase 5）— `utils.py:51-84` `set_logger` 源码中 `if not logger.handlers` 改为 `if not logger.hasHandlers()`。虽然 `logging.getLogger()` 默认返回根 logger 时两者行为相近，但 `hasHandlers()` **会沿 logger 层级递归检查父 logger**，避免以下场景：1. 上游代码先在 root logger 上挂了 handler（如 `logging.basicConfig`、pytest caplog、单元测试 fixture）；2. `set_logger` 拿到子 logger，`logger.handlers` 为空但 `hasHandlers()` 返回 True；3. 原写法会重复 `addHandler`，日志输出**重复**；4. 新写法跳过，避免重复。新增 2 个单元测试 `test_set_logger_uses_has_handlers`（源码守卫，断言 `logger.hasHandlers()` 存在且 `if not logger.handlers` 不存在）/ `test_set_logger_does_not_duplicate_handlers`（连续两次调用 `set_logger`，断言 root logger 上的 handler 数量不变）。
  - **Stage 33 整体验收** — 6 个 Phase 5 代码清理项已落地；当前对应新增测试从 23 项扩到 29 项（7 #10 + 3 #11 + 5 #12 + 5 #13 + 7 #14 + 2 #15），用于把“文档式通过”收紧成真实行为验收。核心训练 / forward / checkpoint 契约保持稳定；有意变化集中在可观测性与状态管理：tensorboard 时间轴统一为 `global_step`、`best_ade` 不再暴露模块级可变状态、`graph_lstm_model` 增加调用诊断。详细见 [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md) Stage 33 与 [PLAN.md §7.1-§7.2](./PLAN.md#71-代码清理1-5-8-9-11-12-14-15) / [PLAN.md §7.2 命名与接口统一](./PLAN.md#72-命名与接口统一10-25-26-27-30)。
- Stage 34：Phase 3 #16 物理系数集中与 CLI 暴露（一次性完成 17 个 magic number + 3 组拼接权重 + 9 个 clamp 上界到 `RolloutQueueCoefs` frozen dataclass）。
  - **#16 rollout_queue_features 魔法常数集中与 CLI 暴露**（Phase 3）— `models.py` 顶部新增 `RolloutQueueCoefs` frozen dataclass（从 `dataclasses` 导入 `dataclass`），把 `rollout_queue_features` 方法体中原本以裸字面量形式存在的 **17 个相位驱动系数**（waiting_ratio / release_ratio / lane_queue_length / stopline_occupancy / front_of_queue / stop_dist 的红/黄/绿/phase_change 驱动）+ **3 组拼接权重**（queue_count_stopline_weight=0.5, lane_density_prev=0.6/lane_queue=0.4, lane_mean_speed_prev=0.6/pred=0.4）+ **9 个 clamp 上界**（queue_count_max=1.5, lane_density_max=1.5, lane_mean_speed_max=1.5, waiting_ratio_max=1.0, release_ratio_max=1.0, lane_queue_length_max=1.5, stopline_occupancy_max=1.0, front_of_queue_max=1.0, stop_dist_max=2.0）全部封装；`CycleStateTrajectoryGenerator.__init__` 新增 `rollout_queue_coefs=None` 形参，`None` 触发 `RolloutQueueCoefs()` 默认值，行为与原硬编码完全一致（向后兼容）。`models.py apply_rollout_coefs_override(base, override_dict)` 现在返回 `(merged_coefs, invalid_keys)`：未知 key 会被过滤，字段值会按目标类型清洗，失败的字段名进入 `invalid_keys`；`train.py` 的 `parse_rollout_queue_coefs(json_str)` 再基于这个结果统一打 warning，并让受影响字段回退到默认值。这样非法 JSON、非 dict、以及 **字段值类型错误** 都不会把坏值带进 rollout。`rollout_queue_features` 方法体内把 `0.08` / `0.10` / `0.12` / `0.14` / `0.6` / `0.4` / `0.5` / `1.5` 全部替换为 `coefs.<field>` 形式（`2.0` 保留为结构化常量，因为是相位归一化）。`train.py` 与 `evaluate_model.py` 新增 CLI 参数 `--rollout_queue_coefs_json <JSON 字符串>`，训练协议日志也新增一行 `Rollout queue coefs | waiting_ratio ... | release_ratio ... | ... | lane_mean_speed_prev=0.6000 lane_mean_speed_pred=0.4000`，记录实际生效的全部 22 个物理系数值，方便回溯 warmup/refine 阶段的系数差异。
  - **关键设计**：
    - `frozen=True` 保证 `rollout_queue_coefs` 不可变，训练/推理切换不会"穿越"出意外状态。
    - 未知 JSON key 静默忽略（而不是 raise），避免 CLI 拼错让训练启动失败。
    - 非法 JSON 静默回退（而不是 raise），减少一个错参的成本。
    - 训练协议日志包含 coefs 实际生效值（而不是 CLI 原命令），避免日志/CLI 不一致时无法对账。
  - **后续可做**：1. 消融 — 把 `waiting_ratio_red_inc` 置 0 验证"红灯期等待比例增长"是否真的是 ADE 改进的关键因素；2. Stage 协议对比 — warmup 用小系数（如 0.04）抑制 rollout 发散，refine 用默认 0.08/0.10/0.12 恢复相位推进信号；3. Sensitivity grid — 在 [0, 0.3] 区间扫描 `queue_count_stopline_weight`，寻找优于硬编码默认值的配置。
  - **新增 18 个单元测试**：`test_rollout_queue_coefs_default_values_match_phase3_baseline`（23 个字段默认值回归守卫）/ `test_rollout_queue_coefs_is_frozen_dataclass`（frozen 约束）/ `test_cycle_state_init_default_uses_dataclass_defaults`（`rollout_queue_coefs=None` 触发默认值）/ `test_cycle_state_init_accepts_custom_rollout_queue_coefs`（自定义 dataclass id 保持一致）/ `test_rollout_queue_features_uses_self_rollout_queue_coefs_attribute`（9 个关键字段名必须出现在方法体）/ `test_rollout_queue_features_zero_red_increments_keeps_features_at_zero`（red_inc 置 0 后 waiting_ratio/lane_queue_length/stopline_occupancy/front_of_queue/queue_count 保持 0）/ `test_rollout_queue_features_default_coefs_grow_waiting_ratio_under_red`（默认 coefs 增长 ≈ 0.08）/ `test_rollout_queue_features_density_weight_override_changes_output`（权重覆盖 0.6→1.0 改变输出 0.6→1.0）/ `test_rollout_queue_features_no_bare_magic_numbers_in_body`（7 个裸字面量不再出现）/ `test_train_parser_accepts_rollout_queue_coefs_json`（CLI 接入）/ `test_train_parse_rollout_queue_coefs_returns_defaults_on_empty` / `test_train_parse_rollout_queue_coefs_merges_valid_json` / `test_train_parse_rollout_queue_coefs_falls_back_on_invalid_json`（非法 JSON + warning）/ `test_train_parse_rollout_queue_coefs_ignores_unknown_keys` / `test_train_parse_rollout_queue_coefs_rejects_non_dict_json`（非 dict + warning）/ `test_train_parse_rollout_queue_coefs_rejects_invalid_field_value_types`（字段值类型错误回退默认值）/ `test_evaluate_model_parser_accepts_rollout_queue_coefs_json`（评估侧 CLI 接入）/ `test_apply_rollout_coefs_override_preserves_untouched_fields`（未触及字段保持 base）。
  - **Stage 34 整体验收** — 1 个 Phase 3 关键暴露项落地；#16 定向测试现在为 18 项；行为契约补齐到"非法 JSON / 非 dict / 非法字段值都不会把坏值带进 rollout"。`rollout_queue_features` 在零输入下 `waiting_ratio ≈ 0.08`、`lane_queue_length ≈ 0.10`，与原硬编码数值一致，但 17+3+9=29 个 magic number 已可配置。详细见 [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md) Stage 34 与 [PLAN.md §5.1](./PLAN.md#51-#16--rollout-魔法常数暴露与可学习化)。

- Stage 35：Phase 0 #17 `pred_state` oracle 假设声明（一次性完成 README 顶层声明 + PLAN/EXPERIMENT_LOG 同步）。
  - **关键内容**：README.md 新增 `## ⚠️ Oracle 假设声明（Phase 0 #17）` 区块，明确四层信息：
    1. `pred_state` 来源（数据集记录的真实未来交通灯状态）
    2. Oracle 性质（模型访问推理时不可用的真实未来信号，真实部署需外部信号控制器或预测模块）
    3. 与 D2-TPred baseline 的对齐性（两者都假设未来信号状态已知，所有 comparable 实验均在相同 oracle 条件下产生）
    4. Phase 0.5 后续计划（信号退化实验 + 敏感度曲线）
  - **Stage 35 整体验收** — 1 个 Phase 0 关键声明项落地（#17 文档部分）；审计结论 15/35 → 16/35。Phase 0.5 实验部分待执行。详细见 [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md) Stage 35 与 [PLAN.md §2.5](./docs/PLAN.md#25-#17--pred_state-oracle-假设未声明⚠️-方法论关键问题)。

- Stage 36：Phase 1 #18 `add_noise` 每步解码注入验收补强（一次性完成强行为测试 + helper 收敛 + PLAN/EXPERIMENT_LOG 同步）。
  - **关键内容**：`models.py` 新增 `expand_scene_noise_to_batch` / `inject_per_step_decoder_noise` 两个 helper，统一 scene-level noise 展开和 decode-step 注入逻辑，替换基类/子类 train/eval 四段重复代码。
  - **测试补强**：把原先“两个未设 seed 的 forward 输出不同”这类弱随机性断言，替换为 8 个强行为测试：覆盖 `TrajectoryGenerator` 与 `CycleStateTrajectoryGenerator` 的 train/eval 路径、`get_noise` 调用次数必须为 `1 次 init + pred_len 次 step`，以及固定 init noise、只改 step noise 时输出必须变化。
  - **Stage 36 整体验收** — #18 的验收证据从“实现存在 + 3 个弱测试”收紧为“公共 helper + 8 个强行为测试 + 全量 117 测试通过”；审计结论 16/35 → 17/35。详细见 [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md) Stage 36 与 [PLAN.md §3.2](./docs/PLAN.md#32-18--add_noise-每步解码未注入)。

- Stage 37：Phase 0 #19 `TRAIN_STAGE_DEFAULTS` 联动一致性校验（一次性完成运行时校验 + 负 `gan_weight` 漏洞修复 + PLAN/EXPERIMENT_LOG 同步）。
  - **关键内容**：`train.py` 增加 `validate_stage_consistency(args)`，并在 `main()` 中于 `apply_stage_defaults(args)` 之后显式调用；默认值补齐和配置联动校验现在分层处理。
  - **漏洞修复**：`gan_weight < 0` 现在会被直接拒绝，而不再像之前那样只打 warning 后放行。这个边界必须挡住，因为训练总损失里直接存在 `g_loss * args.gan_weight`，负权重会翻转对抗项优化方向。
  - **Stage 37 整体验收** — #19 从“有默认表但无一致性防线”变成“启动前强制校验 + 15 个定向测试通过”；审计结论 17/35 → 18/35。详细见 [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md) Stage 37 与 [PLAN.md §2.6](./docs/PLAN.md#26-19--train_stage_defaults-联动不一致)。
