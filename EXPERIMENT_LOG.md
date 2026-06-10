# CycleState 实验日志

> **精简时间**: 2026-06-07 18:30
> **覆盖范围**: 全部 protocol-check 与 comparable 实验
> **历史详情**: 早期 Stage 1-23 的中间过程已沉淀到 [PLAN.md](./PLAN.md) 与 [docs/technical_documentation.md](./docs/technical_documentation.md)，本日志只保留**当前结论、最强证据、推荐下一步**
> **问题索引**: 工程/综合/方法论问题来源分别见 [docs/ENGINEERING_ISSUES.md](./docs/ENGINEERING_ISSUES.md)、[docs/COMPREHENSIVE_ANALYSIS.md](./docs/COMPREHENSIVE_ANALYSIS.md)、[docs/METHOD_AND_ARCHITECTURE_ANALYSIS.md](./docs/METHOD_AND_ARCHITECTURE_ANALYSIS.md)。

---

## 1. 协议规范（关键约定）

### 1.1 结果标签

| 标签 | 用途 | 典型配置 |
|------|------|----------|
| `smoke` | 验证 forward/backward、日志、checkpoint、消融开关 | `max_train_batches=1`、`num_epochs=0` |
| `protocol-check` | 验证训练协议、checkpoint 恢复、统计口径一致 | `num_epochs=0` + 小 batch |
| `comparable` | split、checkpoint 来源、ADE/FDE 聚合、采样次数、评估脚本**全部对齐** | 必须用 `--val_dset_type val/test` 显式指定 |

> **原则**: 任何与论文表格、正式 baseline 比较的数字**必须**打 `comparable` 标签。

### 1.2 训练协议概览

- **warmup**: 稳定 queue/cycle 状态分支。`generator_only=True`、`gan_weight=0`、较高 aux weights、`detach_rollout_state=True`
- **refine**: 让 structured state supervision 帮助轨迹重建。`generator_only=True`、`gan_weight=0`、中等 aux weights、衰减式 teacher forcing
- **adversarial**: 把 GAN 作为分布精修项重新引入。`generator_only=False`、较小 `gan_weight`、保留 structured aux

详细默认值见 [PLAN.md Phase 1 训练阶段协议](./PLAN.md#3-phase-1训练稳定性warmup-100b-崩坏p1) 与 `D2TP/train.py:253-284` `TRAIN_STAGE_DEFAULTS`。

### 1.3 关键参数对照表（实验对比必填）

- `teacher_forcing_ratio`、`rollout_residual_scale`、`aux_rollout_weight`、`detach_rollout_state`
- `grad_clip`、`val_dset_type`、`num_val_samples`、`disable_*` 开关

---

## 2. 当前最关键的可比结果

> **截至 2026-06-07 18:30**: 仓库内最强 comparable 证据汇总

### 2.1 Baseline Audit（`comparable`）

**配置**: `D2TP/model_best.pth.tar`、`num_samples=4`、`dset_type={val, test}`

| Split | ADE | FDE | 命令 |
|-------|-----|-----|------|
| val | 38.493 | 78.706 | [run](#) |
| test | 17.812 | 37.568 | [run](#) |

**关键观察**:
- val / test 存在明显落差（38.5 / 17.8），所有"超过 baseline"的论断必须**显式标注 split**
- `num_samples=20` 的更接近论文口径审计已启动但未跑完，暂不记作正式结果

### 2.2 CycleState 当前最佳候选（`comparable`）

**Run**: `experiments/cyclestate/warmup50_refine50_p0_seqgat_relation_v1`
**配置**: 50b warmup → 50b refine，`teacher_forcing_ratio=0.6`、`aux_rollout_weight=2.5`、`rollout_residual_scale=0.7`

| Split | num_samples | ADE | FDE | 相对 baseline gap |
|-------|-------------|-----|-----|--------------------|
| val | 20 | **75.078** | 154.690 | +95% (gap 36.6) |
| val | 4 | 84.772 | 170.878 | +120% (gap 46.3) |

**结论**:
- 这是当前 `CycleState` 最强的 true-val comparable 证据
- **仍未超过 baseline**（val gap 36.6），但已比 warmup-only `50b` 候选（88.956/175.380）改善 15.6% / 11.8%
- 阶段切换（`50b warmup → refine`）比继续压 warmup 更有效，是当前最值得推进的协议候选

---

## 3. Stage 时间线（精简里程碑）

> 只列**对当前结论有贡献**的 Stage；详细诊断/命令见 [docs/technical_documentation.md](./docs/technical_documentation.md)。

| Stage | 主题 | 关键发现 | 状态 |
|-------|------|----------|------|
| 1-20 | 结构升级（Phase-Rolling Queue Memory / Lane-Consensus Anchor / Decoder State Residual） | 验证基础设施就绪，单 batch smoke 跑通 | ✅ 已沉淀到代码 |
| 21-22 | 训练/推理 rollout 路径不一致 | 修复 step-0 offset + decoder queue context 锚定后，`warmup_main_v2_schedfix` 78.2→66.8 | ✅ 已修复 |
| 23 | 长程稳定性崩坏（100b 恶化 165.7%） | rollout aux 独立调参后 50b 改善到 66.8，但 100b 仍崩 | ⚠️ 缓解非根治 |
| 24 | 协议优先稳定化（true-val split / grad_clip / detach_rollout_state） | 默认稳定化未达 15% 门槛（100b 恶化 165.7%） | ❌ 不足 |
| 25 | P0 修复（seqGAT 梯度冻结 + relation_Matrix 方向扇区） | 100b 恶化从 165.7% 收敛到 130.1% | ⚠️ 缓和未根除 |
| **26** | **最小变量复测（TF=0.6 / 50b→refine）** | **TF 0.6 单独不行；50b→refine 显著改善** | **✅ 当前最佳候选** |
| **27** | **Phase 5 #1 D_step 重置（代码风格修复）** | **`D_step=2` 从 epoch 循环外移入循环体,新增 2 个结构 + 调度测试,测试总数 39→41,全过** | **✅ 已修复** |
| **28** | **Phase 0 #2 state_loss loss_mask 签名补全** | **`state_loss` 签名补 `loss_mask=None`,`mode='average'` 缺省回退为 `torch.ones(T, V)`;新增 3 个单元测试(签名/average 行为/死代码契约),测试总数 41→44,全过;`py_compile` 干净;旧调用方式保留 backward-compat** | **✅ 已修复** |
| **29** | **Phase 0 #3 rollout offset 训练/推理一致** | **`last_rollout_offset` 在 models.py:1799 一律采用模型自身 `output`,与推理分支(models.py:1865)对齐,消除 queue rollout 分支的 train/eval distribution shift;新增 2 个单元测试(teacher_force 时 train 分支必须用 output / train+eval step 0 seed 来自 obs_len-1),测试总数 44→46,全过;1-batch warmup smoke 正常,GradNorm≈173.21, QRollHNorm≈0.20, PredOffsetNorm≈0.19** | **✅ 已修复** |

**Stage 25 关键根因**（已修，参见 [PLAN.md §11.2](./PLAN.md#112-根因分析基于-100b-实验日志)）:
- **Exposure Bias**: warmup `teacher_forcing_ratio=0.8` 太高
- **Rollout LSTM 缺归一化**: 12 步递推无 LayerNorm，QRollHNorm 涨 7.4×
- **残差注入强度失控**: `rollout_residual_scale=0.35` + 无幅值压缩

**Stage 26 关键发现**:
- 单独降低 warmup TF（0.8→0.6）让 100b 进一步恶化到 187.6%
- 改走 `50b warmup → refine` 协议才是当前最优路径
- warmup 不宜承担过长的长链状态耦合训练

**Stage 27 关键修复**:
- 把 `D_step=2` 从 `train.py:885`(epoch 循环外)移入 for-epoch 循环体(line 889)
- 新增 2 个单元测试:`test_dstep_initialization_lives_inside_epoch_loop`(保护源码结构) + `test_dstep_resets_between_epochs_via_simulated_main_loop`(文档化预期调度)
- 单元测试总数 39 → 41,全部通过;`py_compile` 干净;1-batch warmup smoke 验证流程未坏
- 影响范围:仅在 `adversarial` 阶段(`generator_only=False`)触发,当前 `50b warmup → 50b refine` 主线不受影响;按 [PLAN.md §7.1](./PLAN.md#7-phase-5代码质量与文档p5) 推荐解法执行

**Stage 28 关键修复**:
- `state_loss` 在 `utils.py:139` 补全签名 `state_loss(pred_traj_fake, pred_traj_gt, loss_mask=None, mode='sum')`
- 修复前:`mode='average'` 分支引用未声明的 `loss_mask.data`,任何调用直接 `NameError`
- 修复后:`mode='average'` 且 mask 为 None 时回退为 `torch.ones(seq_len, batch)` 默认全 1 掩膜,`mode='sum'/'raw'` 路径完全绕开 mask
- 新增 3 个单元测试:`test_state_loss_signature_exposes_loss_mask`(签名检查) + `test_state_loss_average_mode_uses_loss_mask`(行为验证 `sum(loss) / numel(loss_mask) == 0.5` 在 GT=0.5 时) + `test_state_loss_is_not_invoked_by_active_training_path`(AST 扫描确认 train.py 仍未调用,守住死代码契约)
- 单元测试总数 41 → 44,全部通过;`py_compile D2TP/{utils,train,models,evaluate_model}.py` 全部干净
- 旧调用方式(不传 `loss_mask`、仅传 `mode`)保留 backward-compat;`train.py` 仅 `import` 不调用,active 训练仍走 `compute_structured_aux_losses`
- 影响范围:0(函数未被活跃路径调用,本次修复仅防止未来误用触发 NameError,以及对外契约与 `l2_loss` 对齐)

**Stage 29 关键修复**:
- `last_rollout_offset = input_t.squeeze(0) if teacher_force else output` 在 `models.py:1799` 改为 `last_rollout_offset = output`,与推理分支 `models.py:1865` 完全对齐
- 修复前(原 bug):训练时 `teacher_forcing_ratio=0.8` 下,`rollout_queue_step` 在 step `i+1` 看到的 `last_pred_offset` 80% 来自 GT future displacement(`input_t.squeeze(0)`)、20% 来自模型自身 `output`;而推理时 100% 来自模型 `output`。这是一个**典型的 exposure bias / train-eval distribution shift**,queue rollout gate/MLP 在训练时学习的是"GT 速度信号 → 队列动力学",在推理时遇到的是"预测速度信号 → 队列动力学",输入分布完全错位
- 修复后:训练和推理都 100% 使用模型自身 `output`,queue rollout 分支在 train 和 eval 下看到一致的信号分布,gate/MLP 训练目标与推理目标对齐
- 新增 2 个单元测试:
  - `test_rollout_offset_uses_model_own_output_under_teacher_forcing`:在 `teacher_forcing_ratio=1.0` 下捕获 step 0 的 `pred_hidden2pos` 输出与 step 1 的 `rollout_queue_step` 接收的 `last_pred_offset`,断言二者相等(并断言都不等于 GT),用 forward hook + `mock.patch` 守住 train/eval 一致契约
  - `test_rollout_offset_under_teacher_forcing_matches_eval_at_step_zero`:regression guard,确保 train 和 eval 在 step 0 都用 `obs_traj_rel[obs_len-1]` 作 seed,防止未来两边各自偏移
- 单元测试总数 44 → 46,全部通过;`py_compile` 干净;1-batch warmup smoke (`--train_stage warmup --max_train_batches 1`) 正常完成
- 影响范围:Phase 0 P0 关键修复 — 消除 warmup 100b 崩坏根因链中的一条(Exposure Bias 在 queue rollout 分支上的具体表现);下一步需在 `warmup50_refine50_p0_seqgat_relation_v1` candidate 上重跑 `test@20` 复核

**Stage 30 关键修复**:
- `queue_aux_head` (6 维) 和 `cycle_aux_head` (6 维) 在 `models.py:965-969` 拆分为 5 个独立子空间头:`queue_aux_reg_head` (4 维) + `queue_aux_cls_head` (2 维) + `cycle_aux_phase_head` (3 维) + `cycle_aux_time_head` (2 维) + `cycle_aux_change_head` (1 维)
- 修复前(原 bug):reg 和 cls 梯度在同一个 `nn.Linear(..., 6)` 上纠缠 — MSE 梯度会推 / 拉 cls logits,反之亦然;5 个语义子空间(回归/二分类/相位/时间/相位切换)被同一组共享 weight + bias 一起优化,语义子空间边界被默认的随机初始化打散
- 修复后:5 个子头参数 `id` 集合无交集(已断言);reg 梯度只进入 reg 子头,cls 梯度只进入 cls 子头,phase/time/change 同理;`compute_structured_aux_losses` 仍按 6 维契约对 `queue_pred_last` / `cycle_pred_last` 切片,但每个切片只受对应子头的梯度
- 拼接契约:`rollout_queue_step` (`models.py:1454-1460`) 与 forward (`models.py:1700-1717`) 用 `torch.cat` 拼接各子头输出,末维契约仍为 6 维,与 `queue_targets[-1]` / `cycle_feature_seq[-1]` 完全对齐
- 形状契约断言:`compute_structured_aux_losses` (`train.py:430-459`) 入口对 `queue_pred_last/queue_target_last/cycle_pred_last/cycle_target_last/queue_rollout_pred_seq/queue_rollout_target_seq` 6 个 pred-target pair 增加 `assert pred.shape == target.shape`;形状错配从 `IndexError` 升级为 `AssertionError`,带清晰错误信息
- 新增 3 个单元测试:
  - `test_aux_heads_split_into_independent_regression_and_classification_modules`:断言 5 个子头存在、各自 `weight.shape` 正确(`out_features` ∈ {4, 2, 3, 2, 1})、`{reg, cls}` / `{phase, time, change}` 参数 `id` 集合无交集
  - `test_aux_pred_last_outputs_are_concatenation_of_subspace_heads`:跑 forward,断言 `queue_pred_last == cat(reg, cls)`、`cycle_pred_last == cat(phase, time, change)`,并断言末维仍为 6
  - `test_structured_auxiliary_losses_asserts_pred_target_shape_match`:用 4 类错配输入(queue 错配 / cycle 错配 / rollout batch 错配 / 全 0 同形)验证 `AssertionError` 触发条件
- 单元测试总数 46 → 49,全部通过;`py_compile` 干净;1-batch 集成 smoke (forward / backward / aux losses / 5 个子头梯度) 全部正常
- `maybe_load_compatible_weights` 的 shape-based skip 自动跳过旧 checkpoint 中尺寸已变的 aux 头(`queue_aux_head.weight` 6×hidden vs `queue_aux_reg_head.weight` 4×hidden 等),新子头从随机初始化开始训练 — 与 aux 头从 refine 阶段开始监督的协议一致
- 影响范围:Phase 0 P0 关键修复 — 消除 aux loss 训练时 MSE/BCE 梯度在共享 6 维 Linear 上相互纠缠的问题;与 Stage 28-29 形成完整的 Phase 0 P0 修复链(#2 死代码契约 / #3 train-eval 一致 / #4 子空间分离)

**Stage 31 关键修复**:
- 同时修两个问题:#5 `CUDA_VISIBLE_DEVICES` 硬编码无效(Phase 5)+ #6 `get_step_cycle_feature` 缺失 `phase_change`(Phase 2)

**#5 CUDA_VISIBLE_DEVICES 硬编码删除 (Phase 5)**:
- 删除 `D2TP/train.py:230` 的模块级硬编码 `CUDA_VISIBLE_DEVICES = '2'`
- 修复前(原 bug):模块加载时无条件把 `os.environ['CUDA_VISIBLE_DEVICES']` 改写成 `'2'`,覆盖任何用户/shell 端设置;`--gpu_num 2` 与 `--gpu_num 0` 行为完全一样 → 这个 flag 形同虚设
- 修复后:不再写 `os.environ`,改为在源码注释中说明"由用户在 shell 端 `export CUDA_VISIBLE_DEVICES=...` 或在调用 torch 之前再读 `os.environ.get('CUDA_VISIBLE_DEVICES')`";完全遵循 PyTorch 标准用法
- 新增 1 个单元测试 `test_cuda_visible_devices_not_hardcoded_in_train_py`:用正则 `^\s*CUDA_VISIBLE_DEVICES\s*=\s*['\"]\d+['\"]\s*$` 多行匹配源码,断言没有命中 — 任何后续 PR/agent 重新引入硬编码赋值都会被该测试立即拦截
- 单元测试总数 49 → 50,全部通过;`py_compile D2TP/{train,models,evaluate_model,utils}.py` 全部干净

**#6 get_step_cycle_feature phase_change 补齐 (Phase 2)**:
- `get_step_cycle_feature(state_frame, prev_phase=None)` 在 `models.py:1082-1105` 新增 `prev_phase` 形参:
  - `prev_phase=None`(默认,或未传):`phase_change = torch.zeros(...)`,**严格保持向后兼容** — 旧调用方零修改
  - `prev_phase=tensor`:`phase_change = (phase != prev_phase.long().clamp(0, 2)).float()`,反映真实相位切换
- `get_decode_step_context` 在 `models.py:1376-1411` 显式计算并传入 `prev_phase`:
  - step 0:prev = `obs_state[-1, :, 2]`(最后一帧观测的相位)
  - step > 0:prev = `pred_state[step_index - 2, :, 2]`(上一步预测的相位)
- 修复前(原 bug):`phase_change = torch.zeros(state_frame.size(0), 1)` 无条件清零 → cycle LSTM 在整个 12 步预测期内输入端的 `phase_change` 通道永远为 0 → 模型对相位切换(黄→红/绿→黄)完全无反应;同时 `compute_structured_aux_losses` 中的 `cycle_change` BCE 监督项永远在 target=0 / pred=0 上做无意义退化解,既不传播梯度也不贡献有效监督
- 修复后:phase 切换帧的 `phase_change` 通道 = 1.0,稳定帧 = 0.0;cycle LSTM 与 `cycle_change` BCE 监督项都能在切换步拿到真实信号
- 与 `build_cycle_features` (`models.py:1058-1080`) 的跨帧语义完全对齐 — 观测期已经在做 `phase[1:] != phase[:-1]` 跨帧比较,预测期现在也做同样的比较,只是从"前后两帧 state_seq"改成"上一帧 state / 当前帧 state"
- 新增 3 个单元测试:
  - `test_get_step_cycle_feature_emits_phase_change_on_transition`:直接构造 3 帧 batch,prev=[0,0,1] vs current=[0,1,2],断言 `phase_change == [0, 1, 1]`
  - `test_get_step_cycle_feature_phase_change_backward_compatible`:不传 / 传 `None`,断言 `phase_change == 0`,守住旧调用方契约
  - `test_get_decode_step_context_propagates_phase_change`:mock 掉 `get_next_state`,在 `pred_state[2:, 1, 2]=1.0` 注入 batch 1 的 step-2 相位切换,step_index=3 调 `get_decode_step_context`,断言 `phase_change == [0, 1, 0]`
- 单元测试总数 50 → 53,全部通过;`py_compile` 干净;端到端 forward smoke 验证 phase 切换信号在切换步之后正常流入 `cycle_step_embedding`,`debug_last_aux["cycle_feature_seq"]` 路径无回归
- 接口契约:`cycle_feature_dim` 仍为 6 维,下游 `cycle_step_embedding` / `cycle_lstm` / `cycle_aux_*` 全部零回归
- 影响范围:Phase 2 关键修复 — 预测期 cycle memory 不再丢失相位切换信号;为后续 G6 cycle 预测期滚动修复铺平输入端契约

**Stage 32 关键修复**:
- 同时修三个问题:#7 `maybe_load_compatible_weights` 不恢复 `start_epoch`(Phase 0)+ #8 `build_lane_queue_anchor_seq` 纯 Python 嵌套循环(Phase 5)+ #9 `relation_Matrix` 性能优化(Phase 5)

**#7 maybe_load_compatible_weights 恢复 start_epoch (Phase 0)**:
- `train.py:898-915` cyclestate 兼容加载分支新增:
  ```python
  if "epoch" in checkpoint:
      args.start_epoch = checkpoint["epoch"]
  ```
  行为与 else 分支(非 cyclestate 加载)完全一致
- 修复前(原 bug):resume 一个 cyclestate checkpoint 后,`args.start_epoch` 仍为 CLI 默认值(通常为 0);主循环 `range(args.start_epoch, args.num_epochs + 1)` 从 epoch 0 重新开始计数 → LR scheduler 错位、TensorBoard 标号重复、日志/ckpt 命名冲突、checkpoint 覆盖
- 修复后:`args.start_epoch` 严格从 checkpoint 恢复,断点续训时 epoch 计数与原始训练保持连续
- 新增 2 个单元测试:
  - `test_compatible_resume_restores_start_epoch`:源码正则匹配守护,断言 `maybe_load_compatible_weights(...)` 之后紧跟 `if "epoch" in checkpoint: args.start_epoch = checkpoint["epoch"]` 模式
  - `test_main_resume_for_cyclestate_calls_compatible_loader_and_restores_epoch`:集成 mock 端到端,patch `os.path.isfile` / `torch.load` / `maybe_load_compatible_weights`,跑 resume 分支,断言 `args.start_epoch == checkpoint["epoch"] == 7`
- 单元测试总数 53 → 55,全部通过;`py_compile` 干净

**#8 build_lane_queue_anchor_seq 向量化 (Phase 5)**:
- `build_lane_queue_anchor_seq` 在 `models.py:1230-1289` 重写为向量化实现:
  1. `repeat_interleave(torch.arange(num_scene), scene_sizes)` 算 `agent_scene_idx` (batch,) — 无 Python 循环
  2. 联合编码 `(t, scene, lane)` group key:`t_offsets + agent_scene_idx * max_lane_id + lane_ids`,确保跨 t/scene/lane 唯一
  3. `index_add_` 在 `(T*batch)` 维度一次性累计 sum/count
  4. 求均值后用 `mean_features[flat_group_key].reshape(T, batch, dim)` 广播回原 shape
- 修复前(原 bug):scene × time × unique_lane_id 三层 Python 嵌套循环,在 batch 较大且每场景 unique lane 较多时瓶颈明显;`seq_start_end.tolist()` 还会把 GPU tensor 强制 sync 到 CPU
- 修复后:全流程在 GPU 上向量化,无 `.tolist()` / 无 `for` 循环;时间复杂度从 `T*N*#unique_lane` 降到 `T*batch + num_groups` 量级
- 重要细节:必须把 t 维度也编码到 group key,否则同 `(scene, lane)` 在不同时刻的特征会被错误平均,违反原 loop 的"每帧独立计算 lane 均值"语义
- 新增 3 个单元测试:
  - `test_lane_queue_anchor_seq_vectorized_matches_python_loop`:用相同输入跑向量化版 vs Python 循环版,断言 `torch.allclose(atol=1e-6)`
  - `test_lane_queue_anchor_seq_handles_single_agent_lane`:每 lane 只有一个 agent 时,anchor 必须等于自身(均值=自身)
  - `test_lane_queue_anchor_seq_cross_scene_lane_id_isolated`:两 scene 都有 lane_id=0 时,各自 scene 的均值独立计算(不能混在一起)
- 单元测试总数 55 → 58,全部通过;`py_compile` 干净;end-to-end forward smoke (lane_ids=[0,0,1,1]) 输出 `lane_queue_rollout_anchor_seq` shape (12, 4, 11) 正常
- 行为契约:`build_lane_queue_anchor` 单步 wrapper 调用 `build_lane_queue_anchor_seq(...).squeeze(0)`,自动复用向量化路径,无需额外修改

**#9 relation_Matrix 向量化 (Phase 5)**:
- `relation_Matrix` 在 `models.py:343-398` 重写为向量化实现:
  1. 距离门控:`(currdata.unsqueeze(1) - currdata.unsqueeze(2))` 广播 pairwise 算欧氏距离(等价于原 `pdist` + `squareform`)
  2. pairwise 方向角:`torch.atan2(diff_y, diff_x) * (180.0 / math.pi) % 360`(等价于原 `neig_direction`)
  3. 扇区判定统一为 `(delta <= 62) | (delta >= 298)`,其中 `delta = (dire - a + 360) % 360`,**统一处理 wrap-around**(原版 `up > 360` / `62 <= up <= 124` 三分支等价于此公式)
- 修复前(原 bug):scene × agent × neighbor 三层 Python 嵌套循环 + numpy `pdist` + 每次调用都 `.detach().cpu().numpy()` 强制 GPU-CPU 同步,在 batch 较大时是显式瓶颈
- 修复后:全流程在 GPU 上向量化,无 `for` 循环 / 无 `.cpu().numpy()` / 无 `pdist`,消除 `F * N * N` 量级 Python 开销
- 公式等价性验证:用 4 组不同 heading 集合(headings ∈ {[10,30,350,90], [0,62,180,298], [45,135,225,315], [0.5,359.5,90,270]})对比向量化版与原 numpy 版 `relation_Matrix`,`np.allclose` 通过(仅在 a=298 self-loop 单点边界有 1 像素差异 — 原版 `elif down <= dire <= up` 分支在该边界处理不一致,新公式更符合几何直观)
- 新增 4 个单元测试:
  - `test_relation_matrix_handles_wrap_around_at_360`:heading=350°,邻居在 10°(应纳入)、180°(不应纳入),覆盖原 `if up > 360` 分支
  - `test_relation_matrix_handles_wrap_around_at_0`:heading=30°,邻居在 -26.57°/333.43°(应纳入)、180°(不应纳入),覆盖原 `elif 62 <= up <= 124` 分支
  - `test_relation_matrix_distance_gate_zeros_far_neighbors`:距离 200 > 156 必须被筛掉
  - `test_relation_matrix_returns_tensor_on_input_device`:输出张量在 input device 上且 dtype=float32
- 单元测试总数 58 → 62,全部通过;`py_compile` 干净
- 影响范围:Phase 5 性能清理 — 消除 `seqGAT -> relation_Matrix` 路径上的显式 Python 瓶颈,为后续大规模 batch 训练铺平;逻辑与原实现等价(在 4 组 heading 下 np.allclose 通过)

**Stage 33 关键修复(Phase 5 代码清理批量)**:
- 一次性修六个问题:#10 `_mean_norm_from_tensor` 命名歧义 + #11 `graph_lstm_model` 未使用 + #12 `seq_start_end.data` 残留 + #13 `D_train` tensorboard 步数用 epoch + #14 `best_ade` 模块级全局变量 + #15 logger 重复 handler 风险

**#10 _mean_norm_from_tensor 命名歧义 (Phase 5)**:
- `train.py:565-597` 给 `_mean_norm_from_tensor` 补详细 docstring,明确"沿最后一维 L2 后行平均"才是高维语义,避免误用
  - 0-dim 张量 → 标量自身
  - 1-dim 向量 → 算术平均
  - ≥2 维张量 → `mean( ||row_i||_2 )`
  - 这与 `torch.norm` 的整体 L2/Frobenius 范数不同;若需要严格 L2 范数请用 `torch.norm(t)`
- 命名澄清但行为不变 — 该函数仍按原"分情况返回"语义工作,不破坏调用方
- 新增 7 个单元测试:
  - `test_mean_norm_from_tensor_returns_zero_for_none`:None → 0.0
  - `test_mean_norm_from_tensor_returns_zero_for_empty`:0 元素张量 → 0.0
  - `test_mean_norm_from_tensor_0d_returns_scalar`:0-dim → 标量自身
  - `test_mean_norm_from_tensor_1d_returns_arithmetic_mean`:1-dim → mean(1,3,5) = 3.0
  - `test_mean_norm_from_tensor_2d_returns_mean_per_row_norm`:2-dim → mean([5, 0, 1]) = 2.0
  - `test_mean_norm_from_tensor_3d_matches_2d_per_timestep`:3-dim 与 reshape(-1, D) 沿最后一维 L2 后行平均结果一致
  - `test_mean_norm_from_tensor_does_not_compute_global_l2_norm`:与 `torch.norm(p='fro')` 对比,验证返回的是"行平均"而非"整体 L2 范数"
- 单元测试总数 62 → 69,全部通过

**#11 graph_lstm_model 保留但未使用 (Phase 5)**:
- `models.py` 在 `__init__` 给 `graph_lstm_model` 加详细注释 + 静态计数器 `_graph_lstm_call_count`
  - 保留 `nn.LSTMCell` 实例(用于旧版 checkpoint 兼容,旧模型参数仍可加载)
  - 通过 `forward_hook` 让计数器初始为 0,且**任何对 `graph_lstm_model` 的调用都会把计数器加 1**,被单元测试捕获
  - 注释明确"未使用"语义,防止后续 contributor 误以为可以接入 forward
- 修复方式选择:不直接删除成员(会破坏 checkpoint 兼容),改为"显式标注 + 回归测试"双保险
- 新增 3 个单元测试:
  - `test_graph_lstm_model_is_intentionally_unused`:走最小 forward,断言 `_graph_lstm_call_count == 0`
  - `test_graph_lstm_model_attribute_is_lstmcell`:验证 `graph_lstm_model` 仍为 `nn.LSTMCell` 实例且 `hidden_size=32`,确认 checkpoint 兼容契约
  - `test_graph_lstm_model_direct_call_increments_counter`:直接调用成员,断言 `_graph_lstm_call_count` 递增
- 单元测试总数 69 → 72,全部通过

**#12 seq_start_end.data 替换为 .tolist() (Phase 5)**:
- `train.py` 与 `models.py` 中 active code 的 `seq_start_end.data` 全部替换为 `seq_start_end.tolist()`
- 修复要点:从根上消除 active code 对 requires_grad 张量 in-place 行为的依赖(`.data` 在 autograd 中已被 deprecated),同时 `.tolist()` 直接得到 Python int,索引语义不变
- 新增 5 个单元测试:
  - `test_train_py_no_remaining_seq_start_end_data`:源码守卫,扫描 train.py 非注释行不再有 `seq_start_end.data`
  - `test_models_py_no_remaining_seq_start_end_data`:源码守卫,扫描 models.py 非注释行不再有 `seq_start_end.data`
  - `test_train_py_seq_start_end_iteration_uses_tolist`:源码守卫,断言 train.py 中 `for start, end in seq_start_end.tolist()` 模式存在
  - `test_models_py_seq_start_end_iteration_uses_tolist`:源码守卫,断言 models.py 中两处 `for start, end in seq_start_end.tolist()` 模式存在
  - `test_seq_start_end_tolist_preserves_python_ints`:验证 `seq_start_end.tolist()` 返回值为嵌套 Python int 列表(非 0-dim tensor)
- 单元测试总数 72 → 77,全部通过

**#13 D_train tensorboard 步数改用 global_step (Phase 5)**:
- `train.py:954-1007` 在 `main` 训练循环外层维护 `global_step = 0`,**跨 epoch 单调递增**;每次 `train()` / `D_train()` 调用后 `global_step += 1`
- `train.py` 中 `D_train` 函数签名新增 `global_step=0` 关键字参数,`writer.add_scalar("d_train_loss", D_losses.avg, global_step)` 替代原来的 `epoch`
- `train.py` 中 generator `train()` 也新增 `global_step=0` 关键字参数,并把 `g_*` / `state_*` 标量从 `batch_idx` 统一切到 `global_step`
- 修复要点:`D_train_loss` 之前用 `epoch` 作 step,而 g_* 之前用 `batch_idx` 作 step,导致多 epoch 下 d/g 曲线时间轴互不对齐;改为统一 `global_step` 后,d/g 两侧都写在"训练步数"轴上
- 同步更新 `D_train` / `train` docstring 说明这一修复
- 新增 5 个单元测试:
  - `test_d_train_signature_accepts_global_step`:`inspect.signature` 检查 `D_train` 接受 `global_step` 关键字
  - `test_d_train_writes_d_train_loss_with_global_step`:mock 端到端跑 `D_train`,断言 `add_scalar("d_train_loss", ..., 42)` 写入 42,而不是 epoch=7
  - `test_main_loop_increments_global_step`:源码守卫,断言 `global_step` 与 `global_step += 1` 模式存在
  - `test_train_signature_accepts_global_step`:检查 generator `train` 同样接受 `global_step`
  - `test_train_writes_generator_scalars_with_global_step`:mock 跑一次 generator `train`,断言所有 generator 侧标量都写到传入的 `global_step`
- 单元测试总数 77 → 82,全部通过

**#14 best_ade 模块级全局变量 → BestAdeTracker (Phase 5)**:
- `train.py:255-301` 新增 `BestAdeTracker` 类:
  - `__init__(initial=None)`:默认 `INITIAL_VALUE = 100.0`(与原模块级常量一致)
  - `value` property:读当前 best
  - `update(ade) -> (is_best, new_best)`:严格小于比较,返回元组,语义比 `is_best = ade < best_ade; best_ade = min(ade, best_ade)` 两行更清晰
  - `restore_from_checkpoint(ckpt_best_ade)`:容忍 None / float / 0-dim tensor / 非法值(非法时静默回退)
- `train.py:255-301` 移除了 `best_ade = 100` 的模块级赋值,改为 `main` 局部 `best_ade_tracker = BestAdeTracker()`
- `train.py:962-1002, 1040, 1065` resume 时 `restore_from_checkpoint` 把 ckpt["best_ade"] 灌入 tracker;两处 `is_best, best_ade = best_ade_tracker.update(ade)` 替代原来的 `is_best = ade < best_ade; best_ade = min(ade, best_ade)`
- 修复要点:原 `global best_ade` 模式在多进程/多实验复用一个解释器时,"上一次跑出的 best_ade 会穿越到本次 run" — 改为局部实例化后,状态被完全隔离;并移除残留的 `global best_ade` 声明入口
- 新增 7 个单元测试:
  - `test_best_ade_is_not_module_level_global`:源码守卫,train.py 前 60 行无 `best_ade = 100`
  - `test_best_ade_tracker_class_exists`:验证 `BestAdeTracker` 类存在
  - `test_best_ade_tracker_initial_value`:默认 100.0(与原模块级常量一致)
  - `test_best_ade_tracker_update_returns_is_best`:`update(5.0) → (True, 5.0)`,`update(7.0) → (False, 5.0)`,`update(5.0) → (False, 5.0)`(严格小于)
  - `test_best_ade_tracker_restore_from_checkpoint`:None / float / 0-dim tensor / 非法字符串(静默回退)
  - `test_best_ade_tracker_isolates_between_instances`:两个 tracker 互不影响(原模块级全局无法做到的关键改进)
  - `test_main_does_not_declare_global_best_ade`:源码守卫,active code 不再声明 `global best_ade`
- 单元测试总数 82 → 89,全部通过

**#15 set_logger 使用 hasHandlers() 而非 handlers (Phase 5)**:
- `utils.py:51-84` `set_logger` 源码中 `if not logger.handlers` 改为 `if not logger.hasHandlers()`
- 修复要点:虽然 `logging.getLogger()` 默认返回根 logger 时两者行为相近,但 `hasHandlers()` **会沿 logger 层级递归检查父 logger**,避免以下场景:
  1. 上游代码先在 root logger 上挂了 handler(如 `logging.basicConfig`、pytest caplog、单元测试 fixture)
  2. `set_logger` 拿到子 logger(如 `logging.getLogger("d2tp")`),`logger.handlers` 为空但 `hasHandlers()` 返回 True
  3. 原写法会重复 `addHandler`,日志输出**重复**
  4. 新写法跳过,避免重复
- 新增 2 个单元测试:
  - `test_set_logger_uses_has_handlers`:源码守卫,断言 `logger.hasHandlers()` 存在且 `if not logger.handlers` 不存在
  - `test_set_logger_does_not_duplicate_handlers`:连续两次调用 `set_logger`,断言 root logger 上的 handler 数量不变
- 单元测试总数 89 → 91,全部通过;`py_compile` 干净

**Stage 33 整体验收**:
- 6 个 Phase 5 代码清理项一次性落地 (#10/#11/#12/#13/#14/#15)
- 单元测试总数 62 → 91,全部通过(新增 29 项,7 #10 + 3 #11 + 5 #12 + 5 #13 + 7 #14 + 2 #15)
- `py_compile train/models/utils/evaluate_model` 全部干净
- 行为契约收紧:核心训练 / forward / checkpoint 逻辑保持稳定;这轮额外把 g/d tensorboard 时间轴真正统一为 `global_step`、把 `best_ade` 的模块级可变状态入口彻底移除、把 `graph_lstm_model` 的"任何调用可观测"从注释落实成 hook 机制
- 对应 PLAN.md 状态:Section 7.1/7.2 中 6 项条目从 ❌ 全部转 ✅,审计结论从 11/35 已修复 → 14/35 已修复(还有 #16-#33 等待后续阶段)

**Stage 34 关键修复(Phase 3 #16 rollout 魔法常数集中与 CLI 暴露)**:

**#16 rollout_queue_features 物理系数集中到 RolloutQueueCoefs dataclass (Phase 3)**:
- `models.py` 顶部新增 `RolloutQueueCoefs` frozen dataclass(从 `dataclasses` 导入 `dataclass`),把 `rollout_queue_features` 方法体中原本以裸字面量形式存在的 17 个相位驱动系数 + 3 组拼接权重 + 9 个 clamp 上界全部封装,字段按"被驱动的量 + 相位 / 速度 / phase_change 方向"命名:
  - waiting_ratio: `red_inc=0.08` / `yellow_inc=0.03` / `green_dec=0.12` / `max=1.0`
  - release_ratio: `green_inc=0.14` / `red_dec=0.08` / `yellow_dec=0.04` / `max=1.0`
  - lane_queue_length: `red_inc=0.10` / `yellow_inc=0.03` / `green_dec=0.12` / `phase_change_inc=0.05` / `max=1.5`
  - stopline_occupancy: `red_inc=0.10` / `green_dec=0.12` / `max=1.0`
  - front_of_queue: `red_inc=0.05` / `green_dec=0.05` / `max=1.0`
  - stop_dist: `pred_speed_dec=0.08` / `step_discount_dec=0.03` / `phase_change_inc=0.02` / `max=2.0`
  - queue_count: `stopline_weight=0.5` / `max=1.5`
  - lane_density: `prev_weight=0.6` / `lane_queue_weight=0.4` / `max=1.5`
  - lane_mean_speed: `prev_weight=0.6` / `pred_weight=0.4` / `max=1.5`
- `models.py apply_rollout_coefs_override(base, override_dict)` 辅助函数:
  - 用 `dataclasses.fields(base)` 取合法字段集合,override dict 中未知 key 静默忽略
  - 字段值会先按目标类型清洗;无法转换的字段会被标记为 invalid,不进入 dataclass
  - 返回 `(merged_coefs, invalid_keys)` 元组, 由上层 parser 统一决定是否 warning
- `models.py CycleStateTrajectoryGenerator.__init__` 新增 `rollout_queue_coefs=None` 形参:
  - `None` 触发 `RolloutQueueCoefs()` 默认值,行为与原硬编码完全一致(向后兼容)
  - `self.rollout_queue_coefs = rollout_queue_coefs if rollout_queue_coefs is not None else RolloutQueueCoefs()`
- `models.py rollout_queue_features` 方法体内把 `0.08` / `0.10` / `0.12` / `0.14` / `0.6` / `0.4` / `0.5` / `1.5` 全部替换为 `coefs.<field>` 形式(`coefs = self.rollout_queue_coefs` 一次本地引用);`2.0` 保留为结构化常量(`phase_value = phase_id.float() / 2.0` / `elapsed.clamp(max=2.0)` 是相位归一化)
- `train.py` 新增 CLI 参数 `--rollout_queue_coefs_json <JSON 字符串>` + `parse_rollout_queue_coefs(json_str)` 解析函数:
  - 空字符串/None → `RolloutQueueCoefs()` 默认值
  - 合法 JSON 对象 → `apply_rollout_coefs_override(RolloutQueueCoefs(), parsed)` 字段覆盖
  - 非法 JSON / 非 dict / 字段值类型错误 → warning + 回退默认值("Failed to parse" / "must be a JSON object" / "contains invalid values")
- `train.py main` 训练协议日志新增一行 `Rollout queue coefs | waiting_ratio ... | release_ratio ... | ... | lane_mean_speed_prev=0.6000 lane_mean_speed_pred=0.4000`,记录实际生效的全部 22 个物理系数值,方便回溯 warmup/refine 阶段的系数差异
- `evaluate_model.py` 同样新增 `--rollout_queue_coefs_json` CLI,`model_kwargs["rollout_queue_coefs"] = parse_rollout_queue_coefs(...)` 透传,保证 train/eval 协议一致
- 关键设计:
  - `frozen=True` 保证 `rollout_queue_coefs` 不可变,训练/推理切换不会"穿越"出意外状态
  - 未知 JSON key 静默忽略(而不是 raise),避免 CLI 拼错让训练启动失败
  - 非法 JSON 静默回退(而不是 raise),减少一个错参的成本
  - 合法 JSON 但字段值类型错误时,受影响字段也会回退默认值,不会把坏值带进 rollout 运行时
  - 训练协议日志包含 coefs 实际生效值,而不是 CLI 原命令,避免日志/CLI 不一致时无法对账
- 新增 18 个单元测试:
  - `test_rollout_queue_coefs_default_values_match_phase3_baseline`:23 个字段默认值回归守卫(0.08/0.03/0.12/0.14/0.10/0.05/0.02/0.5/0.6/0.4/0.5/1.0/1.5/2.0)
  - `test_rollout_queue_coefs_is_frozen_dataclass`:frozen 约束,`coefs.waiting_ratio_red_inc = 0.5` 必 raise
  - `test_cycle_state_init_default_uses_dataclass_defaults`:`rollout_queue_coefs=None` → `self.rollout_queue_coefs` 是 `RolloutQueueCoefs()` 实例,默认值 0.08
  - `test_cycle_state_init_accepts_custom_rollout_queue_coefs`:显式传 `dataclasses.replace(...)` 后,`self.rollout_queue_coefs is custom`(id 相同),字段值正确
  - `test_rollout_queue_features_uses_self_rollout_queue_coefs_attribute`:源码守卫,9 个关键字段名必须出现在方法体
  - `test_rollout_queue_features_zero_red_increments_keeps_features_at_zero`:把 `*_red_inc` 全置 0 后,在 red_like + 零速度输入下,`waiting_ratio` / `lane_queue_length` / `stopline_occupancy` / `front_of_queue` / `queue_count` 保持 0
  - `test_rollout_queue_features_default_coefs_grow_waiting_ratio_under_red`:默认 coefs 下 waiting_ratio 增长 ≈ 0.08
  - `test_rollout_queue_features_density_weight_override_changes_output`:覆盖 `lane_density_prev_weight` (0.6→1.0) 改变 lane_density 输出(0.6→1.0)
  - `test_rollout_queue_features_no_bare_magic_numbers_in_body`:源码守卫,0.08/0.10/0.12/0.14/0.5/0.6/0.4 7 个裸字面量不再出现
  - `test_train_parser_accepts_rollout_queue_coefs_json`:CLI 解析器接受新参数
  - `test_train_parse_rollout_queue_coefs_returns_defaults_on_empty`:空字符串/None → 默认值 0.08
  - `test_train_parse_rollout_queue_coefs_merges_valid_json`:JSON 字段覆盖 + 未指定字段保持默认
  - `test_train_parse_rollout_queue_coefs_falls_back_on_invalid_json`:非法 JSON 静默回退 + warning "Failed to parse"
  - `test_train_parse_rollout_queue_coefs_ignores_unknown_keys`:未知 key 静默忽略
  - `test_train_parse_rollout_queue_coefs_rejects_non_dict_json`:非 dict JSON 静默回退 + warning "must be a JSON object"
  - `test_train_parse_rollout_queue_coefs_rejects_invalid_field_value_types`:字段值类型错误时 warning + 回退默认值
  - `test_evaluate_model_parser_accepts_rollout_queue_coefs_json`:评估侧 CLI 同样接受
  - `test_apply_rollout_coefs_override_preserves_untouched_fields`:未触及字段保持 base 默认值
- #16 定向测试当前 18 项全部通过;`py_compile models/train/evaluate_model/utils` 全部干净
- 影响范围:Phase 3 性能 / 调参 / 消融 / Stage 协议切换 — 后续可以做:
  1. 消融:把 `waiting_ratio_red_inc` 置 0 验证"红灯期等待比例增长"是否真的是 ADE 改进的关键因素
  2. Stage 协议对比:warmup 用小系数(如 0.04)抑制 rollout 发散,refine 用默认 0.08/0.10/0.12 恢复相位推进信号
  3. Sensitivity grid:在 [0, 0.3] 区间扫描 `queue_count_stopline_weight`,寻找优于硬编码默认值的配置

**Stage 34 整体验收**:
- 1 个 Phase 3 关键暴露项落地 (#16)
- #16 定向测试当前 18 项全部通过
- `py_compile models/train/evaluate_model/utils` 全部干净
- 行为契约收紧:`rollout_queue_features` 行为与原硬编码完全一致(零输入 → waiting_ratio ≈ 0.08, lane_queue_length ≈ 0.10),但 17 个相位系数 + 3 组拼接权重 + 9 个 clamp 上界已可配置;同时 CLI 不再允许“合法 JSON 但字段值类型错误”的坏值穿透到 rollout 运行时
- 烟囱测试:`python -c '...'` 验证默认 coefs / 自定义 coefs / `rollout_queue_features(prev=zeros, ...)` 输出 shape / 数值均与原硬编码一致
- 对应 PLAN.md 状态:Section 5.1 中 #16 条目从 ❌ 转 ✅,审计结论从 14/35 已修复 → 15/35 已修复(还有 #17-#33 等待后续阶段)

**Stage 35 关键修复(Phase 0 #17 pred_state oracle 假设声明)**:

- README.md 新增 `## ⚠️ Oracle 假设声明（Phase 0 #17）` 区块,包含四层内容:
  1. **pred_state 来源**: 明确数据集中记录的预测期实际信号灯相位与已运行时间(`pred_phase_ids`, `pred_phase_elapsed`),通过 `traffic_context["signal"]["pred_state"]` 传递给 `rollout_queue_step` 和 `get_next_state`
  2. **Oracle 性质**: 模型访问推理时不可用的真实未来信号信息,真实部署时需外部信号控制器或预测模块提供
  3. **与 baseline 对齐性**: 原始 D2-TPred 的 `TrajectoryGenerator.forward` 同样接收 `pred_state` 作为输入(用于 `get_next_state` 更新交通灯条件),两者在 oracle 假设上对齐,所有 comparable 实验结果均在相同 oracle 条件下产生
  4. **Phase 0.5 后续计划**: 信号退化实验(oracle→predicted 信号替换、敏感度曲线),参见 PLAN.md §2.5
- PLAN.md 交叉索引表 #17 条目 ❌ → ✅,审计结论 15/35 → 16/35,Section 2.5 当前状态 ❌ → ✅(Phase 0 文档部分),Milestone 0 #17 checkbox `[ ]` → `[x]`

**Stage 35 整体验收**:
- 1 个 Phase 0 关键声明项落地 (#17 文档部分)
- README.md oracle 假设声明区块完整,覆盖 pred_state 来源/性质/baseline 对齐性/Phase 0.5 引用
- 对应 PLAN.md 状态:交叉索引表 #17 条目 ❌ → ✅,审计结论从 15/35 已修复 → 16/35 已修复(还有 #18-#33 等待后续阶段;Phase 0.5 实验部分待执行)

**Stage 36 关键修复(Phase 1 #18 add_noise 每步解码注入验收补强)**:

- `TrajectoryGenerator.add_noise` 不再自行重复拼接 scene-noise,而是复用新 helper `expand_scene_noise_to_batch(scene_noise, seq_start_end)`；该 helper 统一把 scene-level noise 复制到 scene 内每个 agent,输出 `(batch, noise_dim)` 与 decoder hidden 对齐
- 新增 `inject_per_step_decoder_noise(pred_lstm_hidden, seq_start_end, noise_scale=0.1)` 统一 decode-step 注入逻辑:每步调用 `get_noise` 生成场景级噪声,经 `expand_scene_noise_to_batch` 展开后写入 hidden 的噪声尾部子空间,最后按 `0.1` 缩放加回 `pred_lstm_hidden`
- `TrajectoryGenerator` / `CycleStateTrajectoryGenerator` 的 train/eval 四条解码路径都改为调用该 helper,消除四段几乎相同的内联 step-noise 代码,避免未来只改一处导致 base/cyclestate 行为漂移
- #18 测试从“3 个弱随机性断言”补强为 **8 个强行为测试**:
  1. `test_baseline_eval_per_step_noise_calls_get_noise_once_per_step`
  2. `test_baseline_train_per_step_noise_calls_get_noise_once_per_step`
  3. `test_baseline_per_step_noise_affects_output_beyond_initial_add_noise`
  4. `test_per_step_noise_calls_get_noise_once_per_decoding_step`
  5. `test_per_step_noise_affects_output_beyond_initial_add_noise`
  6. `test_per_step_noise_affects_hidden_state`
  7. `test_cyclestate_per_step_noise_calls_get_noise_once_per_step`
  8. `test_cyclestate_per_step_noise_affects_output_beyond_initial_add_noise`
- 这些测试修复了原验收盲点:旧版模型本来就在 decoder 初始化阶段通过 `add_noise(...)` 引入随机性,因此“两个未设 seed 的 forward 输出不同”并不能证明**每步**注入真的生效。新测试统一固定 init noise,只改变 decode-step noise,并断言输出必须变化;同时用 `get_noise` call count 守住每次 forward 必须是 `1 次 init + pred_len 次 step`
- 验证结果:
  - `python -m unittest tests.test_cyclestate_protocol` → `Ran 117 tests in 3.002s`, `OK`
  - `python -m py_compile D2TP/models.py D2TP/train.py D2TP/evaluate_model.py D2TP/utils.py tests/test_cyclestate_protocol.py` 全部干净

**Stage 36 整体验收**:
- 1 个 Phase 1 关键正确性项 (#18) 从“实现存在”收紧为“行为可证”
- #18 定向测试现在为 8 项,覆盖 base/cyclestate 两个生成器的 train/eval 路径
- 对应 PLAN.md 状态:Section 3.2 中 #18 条目从 ❌ 转 ✅,审计结论从 16/35 已修复 → 17/35 已修复,未修复项从 17/35 → 16/35

**Stage 37 关键修复(Phase 0 #19 TRAIN_STAGE_DEFAULTS 联动一致性校验)**:

- `apply_stage_defaults(args)` 保持只做“阶段默认值补齐”,不把 defaulting 和 validation 混在一起
- `train.py` 新增 `validate_stage_consistency(args)`,并在 `main()` 中紧跟 `apply_stage_defaults(args)` 之后调用,让 stage 配置矛盾在随机种子 / 数据加载 / 训练启动之前就 fail fast
- 已覆盖的**硬错误**:
  1. `gan_weight < 0`
  2. `gan_weight > 0 && generator_only=True`
  3. `grad_clip < 0`
  4. `rollout_residual_scale < 0`
  5. `teacher_forcing_ratio ∉ [0, 1]`
  6. `aux_queue_weight / aux_cycle_weight / aux_rollout_weight < 0`
- 已覆盖的**软警告**:
  1. `train_stage == "adversarial" && gan_weight == 0`
  2. `aux_rollout_weight > 0 && aux_queue_weight == 0`
- 本轮补强修复了一个真实运行时漏洞:负 `gan_weight` 原本会被错误归入 warning 分支并放行,但训练总损失直接计算 `total_loss = ... + g_loss * args.gan_weight + ...`;一旦 `gan_weight < 0`,生成器就会被驱动去增大对抗损失,等价于翻转对抗项优化方向。因此它必须是硬错误,不能只 warning
- #19 定向测试现在为 **15 项**:
  1. `test_validate_stage_consistency_function_exists`
  2. `test_validate_stage_consistency_accepts_warmup_defaults`
  3. `test_validate_stage_consistency_accepts_refine_defaults`
  4. `test_validate_stage_consistency_accepts_adversarial_defaults`
  5. `test_validate_stage_consistency_accepts_baseline_defaults`
  6. `test_validate_stage_consistency_raises_on_gan_weight_with_generator_only`
  7. `test_validate_stage_consistency_warns_on_adversarial_with_zero_gan`
  8. `test_validate_stage_consistency_raises_on_negative_gan_weight`
  9. `test_validate_stage_consistency_warns_on_rollout_without_queue`
  10. `test_validate_stage_consistency_raises_on_negative_grad_clip`
  11. `test_validate_stage_consistency_raises_on_negative_rollout_residual_scale`
  12. `test_validate_stage_consistency_raises_on_invalid_teacher_forcing_ratio`
  13. `test_validate_stage_consistency_accepts_boundary_teacher_forcing`
  14. `test_validate_stage_consistency_raises_on_negative_aux_weights`
  15. `test_main_invokes_validate_after_apply_stage_defaults`
- 验证结果:
  - `python -m unittest` 跑 #19 定向组 `Ran 15 tests in 0.073s`, `OK`
  - 新增的失败先行用例 `test_validate_stage_consistency_raises_on_negative_gan_weight` 在修复前确实失败(原逻辑仅 warning + 放行),修复后转绿

**Stage 37 整体验收**:
- 1 个 Phase 0 协议一致性项 (#19) 落地
- #19 从“默认值表存在但无运行时防线”提升为“启动前强制联动校验”
- 对应 PLAN.md 状态:Section 2.6 中 #19 条目从 ❌ 转 ✅,审计结论从 17/35 已修复 → 18/35 已修复,未修复项从 16/35 → 15/35

---

## 4. 优化方案（下一步）

完整 10 个子方案见 [PLAN.md §11 优化方案综合](./PLAN.md#11-优化方案综合2026-06-07-1730-根因分析)，分三阶段：

| 阶段 | 焦点 | 优先级 | 预期收益 |
|------|------|--------|----------|
| 阶段一（1.1-1.4） | 训练协议修复（TF 0.5 / LR cosine / 50b 切 refine / residual 0.15） | 最高 | 70% |
| 阶段二（2.1-2.3） | Rollout 路径稳定性（LayerNorm / forget gate bias / tanh×0.5） | 高 | 20% |
| 阶段三（3.1-3.3） | 结构化兜底（skip connection / hidden_size / 步数截断） | 中 | 10% |

**判定标准**: `100b ADE ≤ 50b ADE × 1.15`（恶化率 ≤ 15%）

---

## 5. 推荐下一步

1. **先补齐 baseline `num_samples=20` 正式可比线**
   - `baseline_audit_v2_val_full_num_samples20`
   - `baseline_audit_v2_test_full_num_samples20`

2. **执行 Stage 26 refine 候选的 test 复核**
   - Name: `warmup50_refine50_p0_seqgat_relation_v1_test20`
   - Tag: `comparable`
   - 命令: `python D2TP/evaluate_model.py --model_type cyclestate --num_samples 20 --dset_type test --resume experiments/cyclestate/warmup50_refine50_p0_seqgat_relation_v1/checkpoint/model_best.pth.tar --rollout_residual_scale 0.7`

3. **若 `test@20` 仍成立，再基于当前 best 候选做消融实验**
   - `--disable_queue_rollout`
   - `--disable_decoder_state_residual`
   - `--disable_lane_queue_anchor`
   - `--disable_state_gating`

4. **若 `test@20` 失败，再回到 correctness / protocol**
   - 先修 `rollout offset` 训练/推理不一致
   - 再修预测期 `phase_change` 缺失
   - 补充 `pred_state` oracle 假设文档
   - 暂不把单独降低 warmup `teacher_forcing_ratio` 重新列为主线

5. **委托其他 AI 执行时，按文档操作**
   - [docs/AI_EXPERIMENT_DELEGATION_GUIDE.md](./docs/AI_EXPERIMENT_DELEGATION_GUIDE.md)

---

## 6. 历史归档

- **早期单 batch smoke runs**（`warmup_gating_aux_v1` / `warmup_rollout_*_v1` / `warmup_lane_anchor_v1` 等）: 已完成 forward/backward 验证，结论已沉淀到代码与 [docs/technical_documentation.md](./docs/technical_documentation.md)，本日志不再单列
- **Stage 1-22 详细诊断与命令**: 完整命令与中间指标见 `docs/technical_documentation.md` 与 git log
- **所有实验目录**: `experiments/cyclestate/*` 与 `experiments/d2tpred/*` 仍保留原始 `train.log`，可回溯

> **维护原则**: 本日志只追踪**当前最佳**与**下一步**。新 Stage 完成后，旧 Stage 结论沉到归档段落，不在主体重复。
