# CycleState 综合修改计划 (Comprehensive Modification Plan)

> **制定时间**: 2026-06-07 12:00 (CST)
> **最后更新**: 2026-06-07 19:30 (CST) — 计划审计与修复：重新分类 #1/#2，新增 Phase 0.5 (Oracle 验证)，整合根因分析，添加定量成功标准，修正 Phase 5 依赖关系
> **制定依据**: [ENGINEERING_ISSUES.md](./ENGINEERING_ISSUES.md) · [COMPREHENSIVE_ANALYSIS.md](./COMPREHENSIVE_ANALYSIS.md) · [METHOD_AND_ARCHITECTURE_ANALYSIS.md](./METHOD_AND_ARCHITECTURE_ANALYSIS.md) · [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md)
> **目标**: 从"当前不能稳定超越 baseline"推进到"正式 comparable 实验可超越 baseline"
> **核心原则**: **先修基础，再稳训练，后提性能** — 在稳定性问题解决之前，不新增任何功能模块
> **⚠️ 计划前置条件**: ENGINEERING_ISSUES.md、COMPREHENSIVE_ANALYSIS.md、METHOD_AND_ARCHITECTURE_ANALYSIS.md 三份分析文档尚未创建。本计划中的问题清单目前直接嵌入在交叉索引表中（§1），后续应将这些分析提取到独立文档中以支持独立验证。

---

## 目录

1. [主问题交叉索引](#1-主问题交叉索引)
2. [Phase 0：立即修复——代码正确性（P0）](#2-phase-0立即修复代码正确性p0)
3. [Phase 0.5：Oracle 假设验证与信号退化实验](#25-phase-05oracle-假设验证与信号退化实验p05)
4. [Phase 1：训练稳定性——warmup 100b 崩坏（P1）](#3-phase-1训练稳定性warmup-100b-崩坏p1)
5. [Phase 2：结构对齐——研究方法与实现 Gap（P2）](#4-phase-2结构对齐研究方法与实现-gapp2)
6. [Phase 3：特征质量与模型容量（P3）](#5-phase-3特征质量与模型容量p3)
7. [Phase 4：实验体系与可复现性（P4）](#6-phase-4实验体系与可复现性p4)
8. [Phase 5：代码质量与文档（P5）](#7-phase-5代码质量与文档p5)
9. [整体依赖关系与关键路径](#8-整体依赖关系与关键路径)
10. [风险与备选方案](#9-风险与备选方案)
11. [里程碑与检查清单](#10-里程碑与检查清单)

---

## 1. 主问题交叉索引

下表列出三份分析文档中所有问题的统一编号、来源、严重程度、修改阶段和**当前状态**（基于 commit 0b99b6e 审计）。

| 编号 | 问题简述 | 来源 | 严重程度 | 修改阶段 | 当前状态 |
|------|----------|------|----------|----------|----------|
| #1 | `D_step` 跨 epoch 未重置 | EI | 🟢 Minor | Phase 5 | ❌ 未修复 — `D_step=2` 在 epoch 循环外 (train.py:885)，但逻辑分析表明这不会影响训练正确性：D_step 总是在 0→2 之间循环，跨 epoch 不重置仅影响 epoch 边界处 D/G 交替节奏，不构成 bug。降级为代码风格问题。 |
| #2 | `state_loss` 引用未定义的 `loss_mask` | EI | 🟡 Medium | Phase 0 | ❌ 未修复 — `loss_mask.data` 引用未定义参数 (utils.py:149)，但经审计该 `state_loss` 函数未被任何活跃训练路径调用（训练使用 `compute_structured_aux_losses`），属于死代码 bug。修复成本低（添加参数或标记废弃），不阻塞其他 Phase 0 工作。 |
| #3 | 训练/推理 rollout offset 不一致 | EI | 🔴 Critical | Phase 0 | ⚠️ 部分改善 — 初始 offset 改用 `output` (models.py:1707)，但训练循环内仍用 teacher-forced input (models.py:1799) |
| #4 | `cycle_target_last` 维度尺度不匹配 | EI | 🔴 Critical | Phase 0 | ⚠️ 部分改善 — loss 计算已正确分维度 (train.py:466-475)，但 head 仍为单一 Linear 输出 6 维 |
| #5 | `CUDA_VISIBLE_DEVICES` 硬编码无效 | EI | 🟡 Medium | Phase 5 | ❌ 未修复 (train.py:230) |
| #6 | `get_step_cycle_feature` 缺失 `phase_change` | EI | 🟡 Medium | Phase 2 | ❌ 未修复 — `phase_change` 仍为零张量 (models.py:1083) |
| #7 | `maybe_load_compatible_weights` 不恢复 `start_epoch` | EI | 🟡 Medium | Phase 0 | ❌ 未修复 (train.py:864-869) |
| #8 | `build_lane_queue_anchor_seq` 纯 Python 嵌套循环 | EI | 🟡 Medium | Phase 5 | ❌ 未修复 — 仍为三层嵌套循环 (models.py:1215-1227) |
| #9 | `relation_Matrix` 方向扇区 Bug 已修复，性能慢 | EI | 🟡 Medium | Phase 5 | ⚠️ 逻辑 Bug 已修复 (bc47e72, else→elif)，但纯 Python + numpy 循环性能优化未做 |
| #10 | `_mean_norm_from_tensor` 命名歧义 | EI | 🟡 Medium | Phase 5 | ❌ 未修复 |
| #11 | `graph_lstm_model` 未使用 | EI | 🟢 Minor | Phase 5 | ❌ 未修复 (models.py:511) |
| #12 | `seq_start_end.data` 已弃用 | EI | 🟢 Minor | Phase 5 | ⚠️ 大部分已修复，残留 1 处 (train.py:1029) |
| #13 | `D_train` tensorboard 步数用 epoch | EI | 🟢 Minor | Phase 5 | ❌ 未修复 (train.py:1182) |
| #14 | `best_ade` 模块级全局变量 | EI | 🟢 Minor | Phase 5 | ❌ 未修复 (train.py:251, 858) |
| #15 | logger 重复 handler 风险 | EI | 🟢 Minor | Phase 5 | ❌ 未修复 (utils.py:56, 仍用 `logger.handlers`) |
| #16 | rollout 魔法常数未暴露 | CA | 🔴 Critical | Phase 3 | ❌ 未修复 — 0.6/0.4/0.10/0.14 等系数仍在 `rollout_queue_features` 中硬编码 |
| #17 | `pred_state` oracle 假设未声明 | CA | 🔴 Critical | Phase 0 | ❌ 未修复 — README 中无任何 oracle/假设声明 |
| #18 | `add_noise` 每步解码未注入 | CA | 🔴 Critical | Phase 1 | ❌ 未修复 — 噪声仅在第 0 步注入一次 (models.py:1691)，解码循环内无 step noise |
| #19 | `TRAIN_STAGE_DEFAULTS` 联动不一致 | CA | 🟡 Medium | Phase 0 | ❌ 未修复 — `apply_stage_defaults` 中无一致性断言 (train.py:298-316) |
| #20 | `compute_structured_aux_losses` 语义错位 | CA | 🟡 Medium | Phase 2 | ⚠️ 部分改善 — loss 按维度正确拆分 (reg/cls)，但 head 仍为单一 Linear (models.py:959-960)，回归/分类子空间共享参数 |
| #21 | best-of-K 采样次数未对齐校验 | CA | 🟡 Medium | Phase 4 | ❌ 未修复 — checkpoint 未保存 `num_val_samples` |
| #22 | `pred_lstm` 消融模式下维度 | CA | 🟡 Medium | Phase 4 | ❌ 未修复 — 无 `disable_aux_losses` 开关，消融设计未改进 |
| #23 | `phase_duration_limits` 硬编码 | CA | 🟢 Minor | Phase 3 | ❌ 未修复 — (38.0, 47.0, 2.0) 仍为默认值，未暴露到 CLI |
| #24 | `cycle_target_last` 只用最后一帧 | CA | 🟢 Minor | Phase 3 | ❌ 未修复 (train.py:1059) |
| #25 | `light_input_size` 不一致风险 | CA | 🟢 Minor | Phase 5 | ❌ 未修复 — 无单元测试验证 |
| #26 | 状态指标命名不一致 | CA | 🟢 Minor | Phase 5 | ❌ 未修复 — `_norm` 与 `_norm_seq` 仍不一致 |
| #27 | `traffic_context["scene"]` 字段重复 | CA | 🟢 Minor | Phase 5 | ❌ 未修复 (train.py:607-614, models.py:1451+) |
| #28 | 随机种子传播链路不完整 | CA | 🟡 P0-级 | Phase 0 | ❌ 未修复 — `init_hidden_*` 仍用 `torch.randn`，无 `seed_worker`，`random.random()` 仍用于 teacher forcing |
| #29 | 数据集归一化参数未持久化 | CA | 🟡 Medium | Phase 4 | ❌ 未修复 — `queue_count_norm` 等仍硬编码在 `__init__` 中，不写入 checkpoint |
| #30 | `add_noise` noise_type 字符串分发 | CA | 🟢 Minor | Phase 5 | ❌ 未修复 (models.py:23-41) |
| #31 | `disable_*` 开关未集中注册 | CA | 🟢 Minor | Phase 4 | ❌ 未修复 — 无 `AblationConfig` dataclass，开关仍散落各处 |
| #32 | 文档间缺少交叉引用 | CA | 🟢 Minor | Phase 5 | ❌ 未修复 — EXPERIMENT_LOG.md 不引用 ENGINEERING_ISSUES |
| #33 | 缺乏端到端运行示例 | CA | 🟢 Minor | Phase 5 | ❌ 未修复 — 无 `scripts/run_full_pipeline.sh` |
| G1 | seqGAT 被 `torch.no_grad()` 包裹 | MA | ✅ 已修复 (bc47e72) | — | ✅ **已修复** — `torch.no_grad()` 在 TrajectoryGenerator (models.py:730+) 和 CycleStateTrajectoryGenerator 的 seqGAT 调用中均已移除 |
| G2 | `relation_Matrix` 方向扇区失真 | MA | ✅ 已修复 (bc47e72) | — | ✅ **已修复** — `else:` 改为 `elif down <= dire_n_neig <= up:` (models.py:395)，方向扇区逻辑正确 |
| G3 | warmup 100-batch 稳定崩坏 | MA | 🔴 Critical | Phase 1 | ⚠️ 改善中 — S25 修复后恶化率从 165%→130%，但未根除。新增 `detach_rollout_state`、epoch 级 TF 衰减等缓解手段 |
| G4 | meso/macro 分支容量偏小 | MA | 🟡 Medium | Phase 3 | ❌ 未修复 — queue LSTM hidden=32, cycle LSTM hidden=16 不变 |
| G5 | decoder state residual 注入位置单一 | MA | 🟡 Medium | Phase 3 | ❌ 未修复 — 仅在 `pred_lstm_hidden` 注入，未扩展到 cell 或 output |
| G6 | cycle memory 预测期退化 | MA | 🟡 Medium | Phase 2 | ❌ 未修复 — 预测期无 cycle LSTM 滚动，仅用单步 embedding |
| G7 | 消融设计无法分离各模块贡献 | MA | 🟡 Medium | Phase 4 | ❌ 未修复 — 仍只有 4 个二元开关，无 2×2×2 设计 |
| G8 | 实验推进策略过于激进 | MA | 🟡 Medium | Phase 4 | ⚠️ 策略调整 — 已建立 `smoke/protocol-check/comparable` 三级标签体系，但 warmup 仍未解决即推进 |

> **审计结论** (2026-06-07 17:00, 更新于 2026-06-07 19:30):
> - ✅ **已修复**: **2/35** (G1, G2)
> - ⚠️ **部分改善**: **6/35** (#3, #4, #9, #12, #20, G3)
> - ❌ **未修复**: **27/35** (其余全部)
> - 🔄 **重新分类**: **#1** 从 Phase 0/🔴Critical 降级为 Phase 5/🟢Minor（代码逻辑分析表明 D_step 在 0↔2 间循环是正确行为，跨 epoch 不重置不影响正确性）；**#2** 从 🔴Critical 降级为 🟡Medium（死代码，不影响活跃训练路径）
> - **新增 Phase 0.5**: Oracle 假设验证（#17 从纯文档任务升级为独立实验阶段）
> - **关键发现**: 计划制定后新增了 `get_teacher_forcing_ratio` epoch 级衰减、`maybe_detach_rollout_state`、stability metrics 日志等缓解措施，但这些是**缓解而非修复**，核心问题均未解决。

> 来源缩写：EI = ENGINEERING_ISSUES, CA = COMPREHENSIVE_ANALYSIS, MA = METHOD_AND_ARCHITECTURE_ANALYSIS

---

## 2. Phase 0：立即修复——代码正确性（P0）

> **目标**: 消除所有已知的代码级 bug，确保训练/推理逻辑一致，声明关键实验假设。
> **完成标志**: 所有 P0 项修复完成并经过单元测试验证。
> **预计工作量**: 3-5 天（原估算 1-2 天过于乐观；#3 rollout offset 一致性和 #28 完整种子可复现性各需约 1 天独立调试与验证）
> **前置条件**: 无
> **当前进度**: **0/7 已完成** (2026-06-07 审计，移除 #1 后从 8 减为 7)

### 2.1 #2 · `state_loss` 引用未定义的 `loss_mask`（死代码）

- **文件**: `D2TP/utils.py` 行 149
- **当前状态**: ❌ 未修复。函数 `state_loss` 体内引用 `loss_mask.data`，但该参数未出现在函数签名中，调用会触发 `NameError`。
- **影响范围**: 经审计，`state_loss` 未被任何活跃训练/推理路径调用（训练使用 `compute_structured_aux_losses`），属于死代码。不影响当前实验。
- **修改方案**: 添加 `loss_mask` 到函数签名并补充默认值，或显式标记为 `@deprecated` 待移除。
- **验证**: `python -m py_compile D2TP/utils.py`
- **优先级说明**: 与 rollout offset 一致性 (#3) 等影响活跃路径的问题相比，此项可在 Phase 0 末尾或与其他 Phase 5 清理项一起处理。

### 2.2 #3 · 训练/推理 rollout offset 不一致

- **文件**: `D2TP/models.py` 行 1799 (训练) vs 1865 (推理)
- **当前状态**: ⚠️ 部分改善。初始 offset 已改用 `output`（models.py:1707），但训练循环内 `last_rollout_offset = input_t.squeeze(0) if teacher_force else output`（models.py:1799）在 teacher force 时仍用真实未来位移。
- **修改方案**:
  ```python
  # 训练时始终使用模型自身预测作为 rollout offset：
  last_rollout_offset = output  # 无论 teacher_force 与否
  ```
- **关键**: 需要确保 rollout offset 更新在 `pred_hidden2pos` 之后、下一轮 teacher force 判断之前。
- **验证**: 新增测试 `test_rollout_offset_consistency_train_vs_inference`

### 2.3 #4 · `cycle_target_last` 维度尺度不匹配

- **文件**: `D2TP/train.py` 行 466-475, `D2TP/models.py` 行 959-960
- **当前状态**: ⚠️ 部分改善。`compute_structured_aux_losses` 已按语义正确分维度计算 loss（phase→CrossEntropy, time→MSE, change→BCE），但 `queue_aux_head` 和 `cycle_aux_head` 仍为单一 `nn.Linear(..., 6)`，回归子空间和分类子空间共享参数。
- **修改方案**:
  1. 拆分为 `queue_reg_head (4 维) + queue_cls_head (2 维)` 和 `cycle_phase_head (3 维) + cycle_time_head (2 维) + cycle_change_head (1 维)`
  2. 在 `compute_structured_aux_losses` 中添加断言 `assert pred.shape == target.shape`
- **验证**: 新增测试 `test_cycle_aux_scale_consistency`

### 2.4 #7 · `maybe_load_compatible_weights` 不恢复 `start_epoch`

- **文件**: `D2TP/train.py` 行 864-869
- **当前状态**: ❌ 未修复。cyclestate 兼容加载路径跳过了 `args.start_epoch` 恢复。
- **修改方案**:
  ```python
  # 在兼容加载分支也加入：
  if "epoch" in checkpoint:
      args.start_epoch = checkpoint["epoch"]
  ```
- **验证**: 断点续训实验验证 epoch 计数正确

### 2.5 #17 · `pred_state` oracle 假设未声明（⚠️ 方法论关键问题）

- **文件**: `README.md`, `docs/cyclestate_research_story.md`
- **当前状态**: ❌ 未修复。经全文搜索，README 中无任何 "oracle" / "假设" / "ground-truth future signal" 声明。
- **为什么这不仅仅是文档问题**: CycleState 在训练和推理期间将未来真实信号状态 (`pred_state`) 作为输入。这构成了一个 **oracle 假设** — 模型可以访问推理时不可用的真实未来信息。如果审稿人拒绝此假设，整个方法的有效性和与 baseline 的公平比较都会受到质疑。这不是 Phase 5 的文档任务，而是一个必须在实验设计层面解决的方法论问题。
- **修改方案**:
  1. **(Phase 0)**: 在 README 顶部添加 oracle 假设声明块，在科研故事文档中标注此假设
  2. **(Phase 0.5)**: 设计并执行 oracle→predicted 信号退化实验（见下方 Phase 0.5）
  3. **(论文阶段)**: 在论文中显式讨论此假设、其限制、以及信号预测误差对轨迹误差的敏感度分析
- **验证**:
  - 文档审核：README 和科研故事中的声明
  - 实验验证：oracle 信号→预测信号的 ADE 退化曲线（Phase 0.5）

### 2.6 #19 · `TRAIN_STAGE_DEFAULTS` 联动不一致

- **文件**: `D2TP/train.py` 行 298-316
- **当前状态**: ❌ 未修复。`apply_stage_defaults` 中无 `gan_weight > 0` 与 `generator_only` 的一致性断言。不过，`TRAIN_STAGE_DEFAULTS` 现在包含了 `detach_rollout_state` 字段（新增），使阶段默认值更完整。
- **修改方案**:
  ```python
  def apply_stage_defaults(args):
      # ... 应用默认值 ...
      if args.gan_weight > 0 and args.generator_only:
          raise ValueError(...)
  ```
- **验证**: 运行测试 `test_stage_defaults_consistency`

### 2.7 #28 · 随机种子传播链路不完整

- **文件**: `D2TP/train.py` 行 768-770, `D2TP/models.py` 多处 init_hidden
- **当前状态**: ❌ 未修复。所有 `init_hidden_*` 仍用 `torch.randn`（models.py:549-550, 561-562, 1037-1038, 1045-1046），teacher forcing 中仍用 `random.random()`（models.py:1724），DataLoader 无 `seed_worker`。
- **修改方案**:
  1. 将 `init_hidden_*` 改为 `torch.zeros`（Social-LSTM 标准做法）
  2. 增加 `seed_worker` 函数并传给 DataLoader
  3. Teacher forcing 中的 `random.random()` 改用 `torch.rand(1).item()`
- **验证**: 两次相同 seed 的运行，loss 曲线完全一致

---

## 2.5. Phase 0.5：Oracle 假设验证与信号退化实验（P0.5）

> **目标**: 量化 `pred_state` oracle 假设对模型性能的贡献，建立信号预测误差→轨迹误差的敏感度曲线，为论文中的假设讨论提供实验证据。
> **完成标志**: oracle→predicted 信号退化曲线完成，明确在最坏情况下（纯预测信号）CycleState 相对 baseline 的保持率。
> **预计工作量**: 2-3 天（含训练时间）
> **前置条件**: Phase 0 完成（代码正确性修复后才能跑可信的实验）
> **当前进度**: **0/3 已完成**

### 为什么需要这个阶段

CycleState 在训练和推理中使用未来真实信号状态 (`pred_state`) 作为输入。这构成了一个 **oracle 假设**。如果此假设被审稿人质疑且缺乏实验证据，整个方法可能被判定为与 baseline 不公平比较。Phase 0.5 的目标是在进入长时间的训练稳定性调试之前，先确认 oracle 假设在方法中的实际权重。

### 2.5.1 实验 A: 训练 oracle → 推理 oracle（当前默认，上界）

- **目的**: 确认 CycleState 在完全 oracle 条件下的性能上限
- **配置**: 训练和推理均使用真实 `pred_state`（当前默认行为）
- **状态**: ⏸ 待执行（与 Phase 1 的 warmup50_refine50 候选复用）

### 2.5.2 实验 B: 训练 oracle → 推理 predicted（核心退化实验）

- **目的**: 测量推理时用预测信号替代 oracle 信号的性能退化
- **方案**:
  1. 训练一个轻量 GRU 信号预测器，从观测轨迹和灯态预测未来 `pred_state`
  2. 在 CycleState 推理时，用信号预测器的输出替代真实 `pred_state`
  3. 对比实验 A 和实验 B 的 ADE/FDE，得到退化曲线
- **预期**: 如果退化 < 15%，oracle 假设不是核心贡献因素；如果退化 > 30%，需要重新设计方法以减少对 oracle 信号的依赖

### 2.5.3 实验 C: 训练 predicted → 推理 predicted（下界，公平比较）

- **目的**: 建立完全不依赖 oracle 信号的 CycleState 变体
- **方案**: 训练时也用预测信号替代真实 `pred_state`（端到端可微信号预测 + 轨迹预测）
- **用途**: 如果 oracle 假设被审稿人拒绝，此变体可作为公平 baseline 比较的候选

### 2.5.4 信号预测误差→轨迹误差敏感度分析

- **方案**: 向真实 `pred_state` 注入不同幅度的高斯噪声（σ ∈ {0.01, 0.05, 0.1, 0.2, 0.5}），测量 ADE/FDE 变化
- **产出**: 敏感度曲线图（x=信号误差, y=轨迹误差），用于论文中讨论 oracle 假设的边界条件

### 优先级与时机

- **Phase 0 完成后立即启动**：实验 A 复用现有 warmup50_refine50 候选的 checkpoint
- **实验 B/C 可与 Phase 1 并行**：信号预测器训练独立于 CycleState 稳定性调试
- **敏感度分析在实验 A 的 checkpoint 上执行**：不需要额外训练，仅推理时注入噪声

---

## 3. Phase 1：训练稳定性——warmup 100b 崩坏（P1）

> **目标**: 解决 warmup 100-batch 后半程 ADE 崩坏（从 ~87 跳至 ~200+），建立稳定训练基线。
> **完成标志**: `100b ADE <= 50b ADE × 1.15`（恶化不超过 15%）
> **预计工作量**: 3-5 天
> **前置条件**: Phase 0 全部完成
> **当前进度**: **0/4 实验未执行，缓解措施已新增但未根本解决**

### 核心诊断

**实验证据**（来自 Stage 23-25）:
| 实验 | 50b ADE | 100b ADE | 恶化率 |
|------|---------|----------|--------|
| Stage 24 默认 | 87.082 | 231.420 | 165.7% |
| Stage 24 lr=3e-4 | 88.598 | 226.302 | 155.4% |
| Stage 25 P0修复 | 88.956 | 204.730 | 130.1% |

说明：
- seqGAT 梯度 + relation_Matrix 方向扇区的修复 **缓和了崩坏**（165%→130%），但未根除
- 降低学习率也不足以解决问题
- **根因在训练协议设计层面。** 具体根因链（基于 Stage 25 100b 实验日志的深入分析）：

### 根因分析（Stage 25 诊断结论）

以下三条根因链相互耦合，单一修复不足以解决 warmup 崩坏：

**根因 1: Exposure Bias（暴露偏差）**
- Warmup 阶段 `teacher_forcing_ratio=0.8` 过高，模型在训练中 80% 时间看到真实未来位移
- 当 teacher forcing 关闭时（推理模式或低 TF），模型遇到自身预测误差的累积 → 12 步递推中误差指数增长
- 证据：降低 TF 到 0.6 后 100b 恶化反而加剧（187.6%），说明模型已过拟合到高 TF 路径，单独降 TF 不足以修复

**根因 2: Rollout LSTM 缺少归一化**
- 12 步解码递推中，`pred_lstm_model` 的 hidden state 无 LayerNorm 约束
- QRollHNorm（queue rollout hidden norm）从第 1 步到第 12 步增长 7.4×
- 幅值膨胀导致后续 `pred_hidden2pos` 的输出偏移失控

**根因 3: 残差注入强度失控**
- `rollout_residual_scale=0.35` + 无幅值压缩机制
- Decoder state residual 的幅值在 warmup 后期（50b+）开始发散
- 残差注入缺乏 per-step 的 adaptive scaling，导致不同 batch 间注入强度方差大

**三条根因的耦合关系**:
```
高 TF (0.8) → 模型依赖 teacher signal → 自身递推能力弱
    +
无 LayerNorm → 递推 hidden state 膨胀 7.4×
    +
残差注入无压缩 → 状态分支干扰递增
    =
100b 崩坏：ADE 从 ~87 跳至 ~200+
```

**修复策略**: 三条根因需联合解决，不能逐个修。Phase 1 的实验序列（见 §3.1）按"成本从低到高"排列，但实验 B（提前切 refine）绕开了 warmup 长程训练的根因暴露窗口，是当前最经济的路径。

### 自计划制定以来新增的缓解措施（commit c234113 - 0b99b6e）

以下措施已在代码中实现，**缓解但未解决** warmup 稳定性问题：

1. **`get_teacher_forcing_ratio` epoch 级衰减** (train.py:495-507): refine 阶段 TF 每 epoch 衰减 0.02，最低 0.35；adversarial 阶段最低 0.2
2. **`maybe_detach_rollout_state`** (models.py:1549-1566): warmup 阶段 `detach_rollout_state=True` 默认截断 rollout 跨步反传
3. **`extract_state_stability_metrics`** (train.py:550-570): 提取 DInitNorm / DStepNorm / QRollHNorm / PredOffsetNorm 四个稳定性指标并在日志中记录
4. **`maybe_clip_gradients`** (train.py:523-527): 独立的梯度裁剪函数，warmup 默认 `grad_clip=1.0`
5. **refine 阶段协议** (commit 0b99b6e): 新增 refine-stage protocol check 实验证据

### 3.1 系统性消融实验序列

**原则: 每次只改变一个变量。**

**重要**: 实验 A 中的 `teacher_forcing_ratio` 现在通过 `get_teacher_forcing_ratio` 进行 epoch 级调度。实验时需注意：warmup 阶段直接使用 `base_ratio`（不做衰减），因此 CLI 传参 `--teacher_forcing_ratio 0.5` 即可生效。

#### 实验 A: 降低 teacher_forcing_ratio（已证伪，降级为归档）

- **状态**: ✅ 已执行，但结果否定假设
- **实验**: `experiments/cyclestate/warmup_p0_seqgat_relation_tf06_100b`
- **结果**:
  - `batch 50`: `92.735 / 182.898`
  - `batch 100`: `266.671 / 468.079`
  - 恶化率 `187.6%`
- **结论**: 单独降低 warmup teacher forcing 不能解决长链失稳，且最终更差；不再作为下一轮优先方向。

#### 实验 B: 提前切到 refine（当前最佳协议候选）

- **状态**: ✅ 已执行，并成为当前主线
- **实验**: `experiments/cyclestate/warmup50_refine50_p0_seqgat_relation_v1`
- **结果**:
  - quick `val + num_samples=4`: `84.772 / 170.878`
  - full `val + num_samples=20`: `75.078 / 154.690`
- **结论**: 阶段切换优于继续在 warmup 内调参，下一步应先做 `test + num_samples=20` 正式复核，再做消融。

#### 实验 C: 降低 rollout_residual_scale（仅在 test 复核失败后回收）

- **状态**: ⏸ 暂缓
- **原因**: 当前已有更强的 `warmup50 -> refine50` 候选，优先验证该候选在 `test@20` 上是否成立。

#### 实验 D: 切换优化器（仅在协议正确性项排除后考虑）

- **状态**: ⏸ 暂缓
- **原因**: 当前还未完成 baseline `@20` 审计和 candidate `test@20` 复核；在此之前切优化器会把科研故事打散。

### 3.2 #18 · `add_noise` 每步解码未注入

- **文件**: `D2TP/models.py` 行 1691 (仅解码器初始化时注入)
- **当前状态**: ❌ 未修复
- **修改方案**: 同原计划
- **注意**: 此修改需与 teacher_forcing_ratio 调整后的实验同步进行

### 3.3 稳定性告警机制

- **当前状态**: ⚠️ 部分实现。稳定性指标已通过 `extract_state_stability_metrics` 提取并在日志中打印（train.py:1096-1103）和写入 TensorBoard（train.py:1113-1133），但**未设置自动告警阈值**（如 `if norm > threshold: warning`）
- **待补充**: 在指标日志后增加 `if norm > threshold` 的 `logging.warning` 调用

### 3.4 下一轮直接执行顺序

1. **baseline 正式可比线**
   - `baseline_audit_v2_val_full_num_samples20`
   - `baseline_audit_v2_test_full_num_samples20`
   - 目的：明确 `num_samples=20` 口径下仓库 baseline 的真实 `val/test` 结果

2. **当前最佳候选的正式 test 复核**
   - `warmup50_refine50_p0_seqgat_relation_v1_test20`
   - 命令：
     `python D2TP/evaluate_model.py --model_type cyclestate --device cuda --pin_memory --loader_num_workers 0 --batch_size 8 --num_samples 20 --eval_print_every 10 --resume experiments/cyclestate/warmup50_refine50_p0_seqgat_relation_v1/checkpoint/model_best.pth.tar --dset_type test --rollout_residual_scale 0.7`
   - 目的：确认 `50b warmup -> 50b refine` 不是只在 val 上偶然有效

3. **若 test@20 结果成立，再做四条消融**
   - `disable_queue_rollout`
   - `disable_decoder_state_residual`
   - `disable_lane_queue_anchor`
   - `disable_state_gating`

4. **若 test@20 结果不成立，再回到协议正确性项**
   - 优先修 `rollout offset` 训练/推理不一致 (#3)
   - 再修 `get_step_cycle_feature` 缺失 `phase_change` (#6) — 可能本身是 warmup 不稳定的诱因
   - 暂不继续尝试单独降低 warmup `teacher_forcing_ratio`
   - 注：#1 (D_step) 已降级为 Phase 5 代码风格问题，不再列入协议修复路径

---

## 4. Phase 2：结构对齐——研究方法与实现 Gap（P2）

> **目标**: 修复研究方法表述与代码实现之间的结构性 gap。
> **完成标志**: 所有研究方法描述的功能在代码中均完整实现。
> **定量成功标准**:
> - `get_step_cycle_feature` 的 `phase_change` 不再恒为零（非零比例 > 10% batch）
> - `cycle_aux_head` 拆分为独立 phase/time/change 输出头，各头参数量独立
> - 预测期 cycle LSTM 状态更新频率 = 观测期更新频率（每步更新一次）
> **预计工作量**: 2-3 天
> **前置条件**: Phase 1 完成（训练稳定后才有意义验证结构性修改）。注意：部分 Phase 2 问题 (#6 phase_change, G6 cycle memory) 可能本身是 Phase 1 不稳定的诱因——如果 Phase 1 在实验 A-D 全部失败后仍未收敛，应考虑提前修复 #6 和 G6。
> **当前进度**: **0/3 已完成**

### 4.1 #6 · `get_step_cycle_feature` 缺失 `phase_change`

- **文件**: `D2TP/models.py` 行 1083
- **当前状态**: ❌ 未修复。预测期 `phase_change` 仍为零张量。观测期 `build_cycle_features` 有正确的跨帧 phase_change 计算（models.py:1067-1071），但 `get_step_cycle_feature` 缺少此逻辑。
- **修改方案**: 同原计划 — 在 `get_step_cycle_feature` 中加入 `prev_phase` 参数和比较逻辑

### 4.2 #20 · `compute_structured_aux_losses` 语义错位

- **文件**: `D2TP/train.py` 行 397-492
- **当前状态**: ⚠️ 部分改善。loss 计算已正确按维度拆分（queue_reg→MSE, queue_cls→BCE, cycle_phase→CrossEntropy, cycle_time→MSE, cycle_change→BCE），且新增了 `queue_rollout_reg/cls` 分项。但 `queue_aux_head` 和 `cycle_aux_head` 仍为单一 `nn.Linear(..., 6)`（models.py:959-960），回归/分类子空间共享参数。
- **修改方案**: 同原计划 — 拆分独立输出头

### 4.3 G6 · cycle memory 预测期退化—完整实现预测期 cycle 滚动

- **当前状态**: ❌ 未修复。预测期 cycle 仅用单步 `get_step_cycle_feature` 嵌入（`cycle_step_embedding`），无 cycle LSTM 状态更新（`cycle_lstm_cell` 在解码循环中未调用）。cycle memory 在预测期实质是静态的。
- **修改方案**: 同原计划

---

## 5. Phase 3：特征质量与模型容量（P3）

> **目标**: 提升 meso/macro 分支的表达能力和特征质量。
> **完成标志**: 消融实验显示 queue/cycle 特征相比随机特征有显著正向贡献。
> **定量成功标准**:
> - 消融实验 (`disable_queue_rollout` vs 完整模型) 的 ADE 差异 > 5%（证明 queue rollout 非冗余）
> - 容量提升后 (G4: queue LSTM 32→64, cycle LSTM 16→32) 的 ADE 改善 ≥ 3%
> - `rollout_residual_scale` 等魔法常数暴露到 CLI 后，grid search 能找到优于硬编码默认值的配置
> **预计工作量**: 3-4 天
> **前置条件**: Phase 2 完成
> **当前进度**: **0/5 已完成**

### 5.1 #16 · rollout 魔法常数暴露与可学习化

- **文件**: `D2TP/models.py` `rollout_queue_features` 方法
- **当前状态**: ❌ 未修复。0.6/0.4/0.10/0.14 等系数仍在方法体内硬编码，未暴露到 `__init__` 形参或 CLI。
- **修改方案**: 同原计划（渐进式：先暴露形参，再可学习 MLP）

### 5.2 G4 · meso/macro 分支容量提升

- **文件**: `D2TP/models.py` 行 836-837
- **当前状态**: ❌ 未修改。`queue_lstm_hidden_size=32`、`cycle_lstm_hidden_size=16` 不变。
- **修改方案**: 同原计划

### 5.3 G5 · decoder state residual 注入位置扩展

- **文件**: `D2TP/models.py` 行 1794-1795
- **当前状态**: ❌ 未修改。仅在 `pred_lstm_hidden = pred_lstm_hidden + step_state_residual` 注入，未扩展到 cell 或 output。
- **修改方案**: 同原计划

### 5.4 #23 · `phase_duration_limits` 可配置化

- **当前状态**: ❌ 未修改。`(38.0, 47.0, 2.0)` 仍为 `__init__` 默认值，未暴露到 CLI。

### 5.5 #24 · cycle 监督信号利用率提升

- **当前状态**: ❌ 未修改。仍只用 `cycle_feature_seq[-1]`。

---

## 6. Phase 4：实验体系与可复现性（P4）

> **目标**: 建立可信、可复现的实验体系，确保实验结果能支撑学术发表。
> **完成标志**: 完整的 baseline vs CycleState 消融表，所有 `comparable` 标签实验可复现。
> **定量成功标准**:
> - baseline 在 `num_samples=20` 口径下的 val/test ADE/FDE 已记录并可在 1 条命令内复现
> - 消融表覆盖 ≥ 8 种配置组合（2×2×2 核心开关 + baseline + 完整模型）
> - 两次相同 seed 的运行，最终 ADE 差异 < 0.5%（可复现性验证）
> - 所有 `comparable` 实验的 checkpoint 包含完整的 `num_val_samples` 和归一化参数
> **预计工作量**: 2-3 天（不含实验运行时间）
> **前置条件**: Phase 2+3 完成
> **当前进度**: **0/6 已完成。三级标签体系 (`smoke/protocol-check/comparable`) 已建立并用于最新实验**

### 6.1 #21 · best-of-K 采样次数对齐

- **当前状态**: ❌ 未修复。checkpoint 未保存 `num_val_samples`。

### 6.2 #29 · 数据集归一化参数持久化

- **当前状态**: ❌ 未修复。`queue_count_norm=10.0` 等仍硬编码在 `__init__` 中（models.py:840-843），未写入 checkpoint。

### 6.3 #22 · 消融实验公平性

- **当前状态**: ❌ 未修复。仍无 `disable_aux_losses` 开关，消融设计仍为 4 个独立二元开关（共 16 种组合但缺少关键组合）。

### 6.4 #31 · 消融开关集中管理

- **当前状态**: ❌ 未修复。无 `AblationConfig` dataclass。但实验日志模板已增加 `disable_state_gating` 等字段（EXPERIMENT_LOG.md:41-50），使配置记录更规范。

### 6.5 G8 · 实验推进策略调整

- **当前状态**: ⚠️ 策略已调整但未严格执行。三级标签体系 (`smoke/protocol-check/comparable`) 已建立，但 warmup 稳定性未解决前就推进了 refine stage（commit 0b99b6e），不符合"先稳定后推进"原则。

### 6.6 baseline 完整审计

- **状态**: ❌ 尚未执行

---

## 7. Phase 5：代码质量与文档（P5）

> **目标**: 清理技术债务，提升代码可维护性和文档完整性。
> **完成标志**: 所有轻微问题修复，文档交叉引用完整，端到端运行示例可用。
> **定量成功标准**:
> - 新增 ≥ 5 个单元测试覆盖 Phase 0 所有修复项
> - `scripts/run_full_pipeline.sh` 可从零开始完成训练+评估
> - 所有 `# type: ignore` 和 `.data` 弃用 API 消除（或记录 why not）
> - CI 可通过（如果有）
> **预计工作量**: 1-2 天
> **前置条件**: Phase 2 完成后方可修改涉及模型架构的清理项 (#8, #9, #11)；纯文档/命名/测试项 (#5, #10, #12, #14, #15, #25, #26, #27, #30, #32, #33) 可与 Phase 0/1 并行
> **当前进度**: **0/16 已完成**（含从 Phase 0 降级的 #1）

### 7.1 代码清理（#1, #5, #8, #9, #11, #12, #14, #15）

| 编号 | 当前状态 | 修改内容 |
|------|----------|----------|
| #1 | ❌ | 将 `D_step=2` 移入 epoch 循环（代码风格，非 bug：D_step 总是在 0↔2 间循环，跨 epoch 不重置仅影响 epoch 边界的 D/G 交替节奏） |
| #5 | ❌ | 移除 `CUDA_VISIBLE_DEVICES` 硬编码 (train.py:230) |
| #8 | ❌ | 向量化 `build_lane_queue_anchor_seq`（`torch.scatter_add`） |
| #9 | ⚠️ | 方向扇区 Bug 已修复 (bc47e72, else→elif)，但性能优化未做（纯 Python + numpy 循环保留） |
| #11 | ❌ | `graph_lstm_model` 仍在 models.py:511，未被调用 |
| #12 | ⚠️ | 大部分 `.data` 已修复，残留 train.py:1029 一处 `seq_start_end.data` |
| #14 | ❌ | `best_ade` 仍在 train.py:251（模块级），train.py:858（`global best_ade`） |
| #15 | ❌ | `logger.handlers` 未改为 `logger.hasHandlers()` (utils.py:56) |

### 7.2 命名与接口统一（#10, #25, #26, #27, #30）

| 编号 | 当前状态 | 修改内容 |
|------|----------|----------|
| #10 | ❌ | `_mean_norm_from_tensor` 添加注释说明计算逻辑 |
| #25 | ❌ | 增加 `test_light_input_size_consistency` 单元测试 |
| #26 | ❌ | 统一 `xxx_norm` / `xxx_norm_seq` 命名约定 |
| #27 | ❌ | 统一 `traffic_context["scene"]` 字段来源 |
| #30 | ❌ | 可选：`add_noise` 抽象为 `NoiseSampler` 类族 |

### 7.3 文档完善（#32, #33）

- **#32** ❌: EXPERIMENT_LOG.md 与 ENGINEERING_ISSUES.md 之间无交叉引用
- **#33** ❌: 无 `scripts/run_full_pipeline.sh`，但实验日志模板已完善（EXPERIMENT_LOG.md 包含完整的配置模板和命令记录规范）

### 7.4 新增测试覆盖

**当前已有测试**: `tests/test_cyclestate_protocol.py` (550 行，已存在并持续更新)

建议新增测试文件覆盖 Phase 0 所有修复：
- `test_dstep_reset_per_epoch`
- `test_rollout_offset_consistency`
- `test_cycle_aux_scale_consistency`
- `test_seed_reproducibility`
- `test_stage_defaults_consistency`

---

## 8. 整体依赖关系与关键路径

```
Phase 0 (代码正确性)
    │
    ├──────────────────────────┐
    ▼                          ▼
Phase 0.5 (Oracle 验证)    Phase 5a (文档/命名/测试) ← 可与 Phase 0/1 并行
    │                          │
    ▼                          │
Phase 1 (训练稳定性) ──────────┘
    │
    ▼
Phase 2 (结构对齐) ─────────── Phase 5b (模型相关清理: #8, #9, #11)
    │                              ↑ 依赖 Phase 2（共享文件修改）
    ▼
Phase 3 (特征质量/容量)
    │
    ▼
Phase 4 (实验体系/可复现性) ─── Phase 5c (端到端脚本: #33)
    ↑                              ↑ 依赖 Phase 4（需要稳定的实验流程）
```

**关键路径**: Phase 0 → Phase 0.5 → Phase 1 → Phase 2 → Phase 3 → Phase 4

**Phase 5 拆分说明**:
- **Phase 5a** (文档/命名/测试 — #5, #10, #12, #14, #15, #25, #26, #27, #30, #32): 无依赖，可与 Phase 0/1 并行
- **Phase 5b** (模型相关清理 — #1, #8, #9, #11): 依赖 Phase 2，避免与结构性修改产生合并冲突
- **Phase 5c** (端到端脚本 — #33): 依赖 Phase 4，需要稳定的实验流程后才能编写

**Phase 1 是最大瓶颈**。如果 Phase 1 无法通过所有实验（A/B/C/D），需要进入备选方案（见第 9 节）。

---

## 9. 风险与备选方案

### 9.1 风险矩阵

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| warmup 100b 崩坏无法根除 | 中 | 高 | 备选 A/B/C（见下） |
| queue feature 启发式质量不足以支撑有效监督 | 中 | 高 | 备选 D |
| oracle 假设（#17）被审稿人质疑 — **已升级为 Phase 0.5** | 高 | **高** | Phase 0.5 实验体系 + 备选 E |
| 容量提升后显存溢出 | 低 | 中 | 降 batch_size 或 dynamic batching |

### 9.2 备选方案

#### 备选 A: 放弃 warmup 长程训练

- 如果所有实验都无法通过 100b 稳定性门槛：
  - 将 warmup 固定为 50b
  - 50b warmup + 50b refine 作为一个完整的"warmup"等价阶段
  - 这不会根本性影响 staged training 的科学合理性

#### 备选 B: 渐进式训练（freeze-then-finetune）

- Step 1: Freeze baseline 参数，只训练 queue/cycle 分支（warmup 100b）
- Step 2: 解冻所有参数，联合微调（refine 100b）
- Step 3: 引入 GAN（adversarial）

#### 备选 C: 简化模型

- 先只用 queue-state branch（不用 rollout，不用 cycle）
- 验证静态 queue memory 的有效性
- 逐步增加 rollout → lane anchor → decoder residual → cycle branch

#### 备选 D: queue feature 端到端可学习化

- 如果启发式 queue feature 被证明无效：
  - 用一个小型 GNN/Transformer encoder 从观测轨迹和灯态中端到端学习 queue 表示
  - 仅保留排队/释放的物理概念框架，放弃手工统计特征

#### 备选 E: 信号预测联合实验

- 针对 oracle 假设问题（现已升级为 Phase 0.5）：
  - Phase 0.5 的实验 B/C 即为备选 E 的具体实施方案
  - 训练一个轻量 GRU 预测 `pred_state`（未来信号状态）
  - 用预测的 `pred_state` 替代真值喂给 CycleState
  - 报告信号预测误差 vs 轨迹预测误差的敏感度曲线（perturbation analysis）
  - 如果退化 < 15%，oracle 假设不是核心贡献；如果 > 30%，需考虑备选 C（简化模型）或重新设计状态依赖

---

## 10. 里程碑与检查清单

> **审计日期**: 2026-06-07 17:00 (基于 commit 0b99b6e)

### Milestone 0: Phase 0 完成
- [ ] #2 state_loss loss_mask（死代码） — ❌ 未修复
- [ ] #3 rollout offset 一致性 — ⚠️ 部分改善（初始 offset 已修正，循环内仍不一致）
- [ ] #4 cycle_aux scale — ⚠️ 部分改善（loss 按维度正确拆分，head 未拆分）
- [ ] #7 start_epoch 恢复 — ❌ 未修复
- [ ] #17 oracle 假设声明（文档部分） — ❌ 未修复
- [ ] #19 stage defaults 断言 — ❌ 未修复（但 `TRAIN_STAGE_DEFAULTS` 已扩展了 `detach_rollout_state`）
- [ ] #28 随机种子完整 — ❌ 未修复（`init_hidden_*` 仍用 randn，无 seed_worker）
- [ ] 所有 Phase 0 单元测试通过 — ❌ 测试未编写

### Milestone 0.5: Phase 0.5 完成——Oracle 假设验证
- [ ] 实验 A (训练 oracle→推理 oracle) 结果记录 — ⏸ 待执行
- [ ] 实验 B (训练 oracle→推理 predicted) 结果记录 — ⏸ 待执行
- [ ] 实验 C (训练 predicted→推理 predicted) 结果记录 — ⏸ 待执行
- [ ] 信号预测误差→轨迹误差敏感度曲线 — ⏸ 待执行
- [ ] Oracle 假设讨论段落草稿（论文用） — ⏸ 待执行

### Milestone 1: Phase 1 完成——训练稳定
- [x] 实验 A (`teacher_forcing_ratio=0.6`) 已证伪 — `100b` 更差
- [x] 实验 B (`50b warmup→50b refine`) 已跑通，并成为当前最佳协议候选
- [ ] 对实验 B 做 `test + num_samples=20` 正式复核
- [ ] 仅在实验 B 的 `test@20` 失败后，再考虑实验 C (`rollout_residual_scale`) 或实验 D (optimizer)
- [ ] #18 add_noise per-step 验证 — ❌ 未修复
- [ ] 稳定性告警机制生效 — ⚠️ 指标已记录但无自动告警阈值

### Milestone 2: Phase 2+3 完成——方法对齐
- [ ] #6 phase_change 完整实现 — ❌ 未修复
- [ ] #20 aux loss 拆分 — ⚠️ 部分改善（loss 项已拆分，输出头未拆分）
- [ ] G6 cycle 预测期滚动 — ❌ 未修复
- [ ] #16 魔法常数暴露/可学习 — ❌ 未修复
- [ ] G4 容量提升 — ❌ 未修复
- [ ] G5 残差注入扩展 — ❌ 未修复
- [ ] 消融实验对比 A/B — ❌ 未执行

### Milestone 3: Phase 4 完成——实验体系就绪
- [ ] #21 best-of-K 对齐 — ❌ 未修复
- [ ] #29 归一化参数持久化 — ❌ 未修复
- [ ] #22 消融实验公平性 — ❌ 未修复
- [ ] #31 消融开关集中管理 — ❌ 未修复
- [ ] baseline 完整审计 — ❌ 未执行
- [ ] CycleState full pipeline 完成 — ❌ 未执行
- [ ] 最终消融表 — ❌ 未执行

### Milestone 4: Phase 5 完成——代码就绪
- [ ] 所有轻微问题修复 — ❌ (0/16，含从 Phase 0 降级的 #1)
- [ ] #1 D_step 移至 epoch 循环内（代码风格） — ❌ 未修复
- [ ] 文档交叉引用完整 — ❌ 未修复
- [ ] `run_full_pipeline.sh` 可用 — ❌ 未创建
- [ ] 新增测试通过 — ⚠️ `test_cyclestate_protocol.py` 已有 550 行基础测试，但未覆盖 Phase 0 修复
- [ ] Phase 5a (文档/命名/测试) 可在 Phase 0/1 期间完成 — ⏸ 未启动
- [ ] Phase 5b (模型相关清理) 等待 Phase 2 完成 — ⏸ 阻塞中
- [ ] Phase 5c (端到端脚本) 等待 Phase 4 完成 — ⏸ 阻塞中

---

## 10.1 执行委托规范

从本轮开始，若把实验执行工作委托给其他 AI，统一遵循：

- [AI_EXPERIMENT_DELEGATION_GUIDE.md](./AI_EXPERIMENT_DELEGATION_GUIDE.md)

该手册是本计划的执行补充，不替代本计划本身。关系如下：

1. `PLAN.md` 负责列问题池、优先级、根因和路径依赖。
2. `AI_EXPERIMENT_DELEGATION_GUIDE.md` 负责告诉执行型 AI：
   - 什么能做
   - 什么不能做
   - 先跑什么
   - 出现什么情况必须停下上报
3. `README.md` 只保留摘要和当前主线。
4. `EXPERIMENT_LOG.md` 记录时间线和已完成证据。

执行顺序以委托手册第 5 节为准：

1. baseline `val@20`
2. baseline `test@20`
3. current best candidate `test@20`
4. 若站得住，再做四条消融
5. 若站不住，再回 correctness / protocol

已证伪方向不得重新升为主线：

- 单独降低 warmup `teacher_forcing_ratio`

## 附录 A：实验记录模板

每个 experiment 应记录：

```markdown
### Run
- Name:
- Phase: [0/1/2/3/4]
- Tag: [smoke/protocol-check/comparable]
- Date:
- Command:
- Key switches:
  - teacher_forcing_ratio:
  - rollout_residual_scale:
  - aux_rollout_weight:
  - optimizer:
  - lr:
  - grad_clip:
- Results:
  - Best 50b ADE / FDE:
  - Best 100b ADE / FDE (if applicable):
  - Stability pass? [yes/no] (100b/50b ≤ 1.15)
- Notes:
```

## 附录 B：文档关系图

```
ENGINEERING_ISSUES.md          COMPREHENSIVE_ANALYSIS.md        METHOD_AND_ARCHITECTURE_ANALYSIS.md
(代码级问题 #1-#15)            (系统级问题 #16-#33)             (方法论问题 G1-G8)
        │                              │                                │
        └──────────────────────────────┼────────────────────────────────┘
                                       │
                                       ▼
                          COMPREHENSIVE_PLAN.md  ← 本文件
                          (综合修改计划)
                                       │
                        ┌──────────────┼──────────────┐
                        ▼              ▼              ▼
                  EXPERIMENT_LOG.md  README.md   cyclestate_research_story.md
                  (实验执行追踪)    (使用文档)     (科研叙事更新)
```

---

## 修复记录

| 日期 | 阶段 | 修改内容 | 修改人 | 备注 |
|------|------|----------|--------|------|
| 2026-06-07 12:00 | — | 综合修改计划制定 | AI | 综合 ENGINEERING_ISSUES + COMPREHENSIVE_ANALYSIS + METHOD_AND_ARCHITECTURE_ANALYSIS |
| 2026-06-07 16:49 | Phase 1 | seqGAT `torch.no_grad()` 移除 (G1) | 用户 (bc47e72) | TrajectoryGenerator 和 CycleStateTrajectoryGenerator 的 seqGAT 调用均已移出 no_grad 上下文 |
| 2026-06-07 16:49 | Phase 0 | relation_Matrix 方向扇区 Bug 修复 (G2) | 用户 (bc47e72) | `else:` → `elif down <= dire_n_neig <= up:` 修复无条件连接问题 |
| 2026-06-07 16:49 | Phase 1 | 稳定性缓解措施 batch 1 | 用户 (bc47e72) | 新增 `detach_rollout_state`、epoch 级 TF 衰减 (`get_teacher_forcing_ratio`)、stability metrics 日志、`maybe_clip_gradients` |
| 2026-06-07 17:49 | Phase 1 | refine stage protocol-check | 用户 (0b99b6e) | 新增 refine 阶段训练协议证据 |
| 2026-06-07 17:00 | — | PLAN.md 全面状态审计 | AI | 基于最新代码 (0b99b6e) 审计全部 35 个问题状态，结论：2 已修复，6 部分改善，27 未修复 |

## 新增代码改进（超出原计划范围）

以下改进已在最新代码中实现，超越了 PLAN.md 制定时的代码状态：

1. **训练协议日志** (train.py:839-857): 启动时记录完整的训练协议配置字符串，包含 model_type、stage、所有超参、消融开关
2. **`get_teacher_forcing_ratio`** (train.py:495-507): 按阶段和 epoch 的 TF 调度，refine/adversarial 阶段自动衰减
3. **`maybe_detach_rollout_state`** (models.py:1549-1566): warmup 阶段截断 rollout 跨步反传
4. **`extract_state_stability_metrics`** (train.py:550-570): 正式提取 4 个稳定性指标并写入 TensorBoard 和日志
5. **`build_optimizers` / `maybe_clip_gradients`** (train.py:515-527): 独立的优化器构造和梯度裁剪函数
6. **`compute_structured_aux_losses` 改进**: 新增 `queue_rollout_reg/cls` 分项，loss 计算按语义正确分维度
7. **`tests/test_cyclestate_protocol.py`** (550 行): 基础训练协议单元测试
8. **EXPERIMENT_LOG.md**: 完善的实验模板和结果标签规范
