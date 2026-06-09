# CycleState 综合修改计划 (Comprehensive Modification Plan)

> **制定时间**: 2026-06-07 12:00 (CST)
> **最后更新**: 2026-06-09 21:00 (CST) — Stage 42 #29 修复 (数据集归一化参数持久化: norm_params/load_norm_params + train/evaluate checkpoint 读写), 测试总数 202 → 212, 审计结论 (28/35 已修复)
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

下表列出三份分析文档中所有问题的统一编号、来源、严重程度、修改阶段和**当前状态**（基于 commit 0b99b6e 审计，并同步 2026-06-07 Stage 27 #1、Stage 28 #2 与 Stage 29 #3 修复状态）。

| 编号 | 问题简述 | 来源 | 严重程度 | 修改阶段 | 当前状态 |
|------|----------|------|----------|----------|----------|
| #1 | `D_step` 跨 epoch 未重置 | EI | 🟢 Minor | Phase 5 | ✅ 已修复 (2026-06-07) — `D_step=2` 移入 for-epoch 循环体，新增结构 + 调度双测试覆盖 |
| #2 | `state_loss` 引用未定义的 `loss_mask` | EI | 🟡 Medium | Phase 0 | ✅ 已修复 (2026-06-07) — `state_loss` 签名补 `loss_mask=None`；`mode='average'` 分支在 mask 缺省时回退为 `torch.ones(T, V)` 默认掩膜，避免 NameError；新增 3 个单元测试（签名 / average 行为 / 死代码契约） |
| #3 | 训练/推理 rollout offset 不一致 | EI | 🔴 Critical | Phase 0 | ✅ 已修复 (2026-06-07) — 训练循环内 `last_rollout_offset` 一律采用模型自身 `output`(models.py:1799),与推理分支(models.py:1865)对齐;新增 2 个单元测试(train/eval rollout offset 一致性 + step 0 seed 不变) |
| #4 | `cycle_target_last` 维度尺度不匹配 | EI | 🔴 Critical | Phase 0 | ✅ 已修复 (2026-06-07) — `queue_aux_head`(6 维)和 `cycle_aux_head`(6 维)拆分为独立子空间头:`queue_aux_reg_head`(4 维)+`queue_aux_cls_head`(2 维)、`cycle_aux_phase_head`(3 维)+`cycle_aux_time_head`(2 维)+`cycle_aux_change_head`(1 维),reg/cls/phase/time/change 五组参数互不共享;`compute_structured_aux_losses` 增加 pred/target 形状契约断言(models.py:956-969, train.py:430-459);新增 3 个单元测试(子头结构独立性 + 拼接正确性 + 形状契约),测试总数 46 → 49 |
| #5 | `CUDA_VISIBLE_DEVICES` 硬编码无效 | EI | 🟡 Medium | Phase 5 | ✅ 已修复 (2026-06-08) — 删除 `train.py` 模块级 `CUDA_VISIBLE_DEVICES='2'` 硬编码，改为完全尊重用户 shell / 环境变量设置；新增 1 个源码守卫测试防止回归 |
| #6 | `get_step_cycle_feature` 缺失 `phase_change` | EI | 🟡 Medium | Phase 2 | ✅ 已修复 (2026-06-08) — `get_step_cycle_feature(state_frame, prev_phase=None)` 补齐跨帧 `phase_change` 比较，`get_decode_step_context` 显式传入上一帧 phase；新增 3 个单元测试覆盖切换、向后兼容和解码期传播 |
| #7 | `maybe_load_compatible_weights` 不恢复 `start_epoch` | EI | 🟡 Medium | Phase 0 | ✅ 已修复 (2026-06-08) — cyclestate 兼容加载分支补 `if "epoch" in checkpoint: args.start_epoch = checkpoint["epoch"]`；新增 2 个单元测试覆盖源码守卫与 resume 分支恢复 |
| #8 | `build_lane_queue_anchor_seq` 纯 Python 嵌套循环 | EI | 🟡 Medium | Phase 5 | ✅ 已修复 (2026-06-08) — 重写为 `repeat_interleave + index_add_` 的向量化实现；新增 3 个单元测试覆盖等价性、单 agent lane 和跨 scene 隔离 |
| #9 | `relation_Matrix` 方向扇区 Bug 已修复，性能慢 | EI | 🟡 Medium | Phase 5 | ✅ 已修复 (2026-06-08) — 在逻辑修复基础上进一步改为全向量化 `torch` 实现，移除 Python 三重循环 / numpy `pdist` / `.cpu().numpy()`；新增 4 个单元测试覆盖 wrap-around、距离门控与 device 契约 |
| #10 | `_mean_norm_from_tensor` 命名歧义 | EI | 🟡 Medium | Phase 5 | ✅ 已修复 (2026-06-08) — 详尽 docstring 标注 0/1/≥2 维分情况返回语义,明确"沿最后一维 L2 后行平均"才是高维语义,避免与 `torch.norm` 误用;新增 7 个单元测试覆盖 None/0-d/1-d/2-d/3-d/与 Frobenius 范数的差异 |
| #11 | `graph_lstm_model` 未使用 | EI | 🟢 Minor | Phase 5 | ✅ 已修复 (2026-06-08) — 保留 `nn.LSTMCell` 实例(用于旧版 checkpoint 兼容),添加 `_graph_lstm_call_count` + `forward_hook` 作为真实调用诊断,并用详细注释明确"未使用"语义;新增 3 个单元测试覆盖 forward 路径未调用 / 成员类型 / 直接调用必增计数 |
| #12 | `seq_start_end.data` 已弃用 | EI | 🟢 Minor | Phase 5 | ✅ 已修复 (2026-06-08) — train.py 与 models.py 中 active code 的 `seq_start_end.data` 均替换为 `seq_start_end.tolist()`;新增 5 个单元测试(双文件源码守卫 + 双文件 tolist 模式 + Python int 契约) |
| #13 | `D_train` tensorboard 步数用 epoch | EI | 🟢 Minor | Phase 5 | ✅ 已修复 (2026-06-08) — `main` 训练循环维护 `global_step` 跨 epoch 单调递增,且 `D_train` 与生成器 `train` 的 tensorboard 标量都统一写入 `global_step`;新增 5 个单元测试(签名检查 + D_train mock 验证 + 源码守卫 + train 签名/写入验证) |
| #14 | `best_ade` 模块级全局变量 | EI | 🟢 Minor | Phase 5 | ✅ 已修复 (2026-06-08) — 新增 `BestAdeTracker` 类(`update` / `restore_from_checkpoint` / `value` 三个公开 API),`main` 函数局部实例化并移除 `global best_ade` 声明入口;checkpoint 加载时 `restore_from_checkpoint` 把 `ckpt["best_ade"]` 灌入 tracker;新增 7 个单元测试覆盖存在性、初始值、update、restore、多实例隔离、不再模块级、无 `global best_ade` |
| #15 | logger 重复 handler 风险 | EI | 🟢 Minor | Phase 5 | ✅ 已修复 (2026-06-08) — `utils.set_logger` 把 `if not logger.handlers` 改为 `if not logger.hasHandlers()`,沿 logger 层级递归检查父 logger,避免子 logger 在父 logger 已有 handler 时重复挂载;新增 2 个单元测试(源码守卫 + 连续两次调用 handler 数量不变) |
| #16 | rollout 魔法常数未暴露 | CA | 🔴 Critical | Phase 3 | ✅ 已修复 (2026-06-08) — ``rollout_queue_features`` 内 17 个相位驱动系数 + 3 组拼接权重 + 9 个 clamp 上界集中到 ``models.RolloutQueueCoefs`` frozen dataclass,``CycleStateTrajectoryGenerator.__init__`` 新增 ``rollout_queue_coefs`` 形参 (``None`` 触发默认值, 向后兼容);``train.py`` / ``evaluate_model.py`` 新增 ``--rollout_queue_coefs_json`` CLI, 非法 JSON / 非 dict / 非法字段值都会 warning 并对受影响字段回退默认值;新增 18 个单元测试覆盖默认值守卫 / frozen / 构造路径 / 源码不再出现裸字面量 / JSON 解析与 fallback / 非法字段值回退 / 字段覆盖 / 评估侧 CLI 接入 |
| #17 | `pred_state` oracle 假设未声明 | CA | 🔴 Critical | Phase 0 | ✅ 已修复 (2026-06-08) — README.md 新增 `## ⚠️ Oracle 假设声明` 区块,明确 `pred_state` 来源、oracle 性质、与 D2-TPred baseline 的对齐性,以及 Phase 0.5 后续信号退化实验计划 |
| #18 | `add_noise` 每步解码未注入 | CA | 🔴 Critical | Phase 1 | ✅ 已修复 (2026-06-09) — `TrajectoryGenerator` 与 `CycleStateTrajectoryGenerator` 的训练/推理解码循环均新增 per-step noise injection；公共 helper `expand_scene_noise_to_batch` / `inject_per_step_decoder_noise` 统一 scene-noise 展开与 step-noise 注入，避免四处分叉漂移；新增 8 个单元测试覆盖 base/cyclestate 两个生成器的 train/eval 路径、`get_noise` 调用次数（1 次 init + `pred_len` 次 step）以及“固定 init noise、只改 step noise 必须改输出”的强行为断言；测试总数 109 → 117,全过 |
| #19 | `TRAIN_STAGE_DEFAULTS` 联动不一致 | CA | 🟡 Medium | Phase 0 | ✅ 已修复 (2026-06-09) — `train.py` 新增 `validate_stage_consistency(args)`，并在 `main()` 中于 `apply_stage_defaults(args)` 之后、训练启动之前强制调用；硬错误覆盖 `gan_weight < 0`、`gan_weight > 0 && generator_only=True`、负 `grad_clip`、负 `rollout_residual_scale`、非法 `teacher_forcing_ratio`、负 aux 权重；软警告覆盖 `adversarial + gan_weight == 0` 与 `aux_rollout_weight > 0 && aux_queue_weight == 0`；新增 15 个单元测试覆盖默认 stage 合法性、硬错误、软警告与 main 调用顺序，测试总数 117 → 118（#19 定向组 15 项全过） |
| #20 | `compute_structured_aux_losses` 语义错位 | CA | 🟡 Medium | Phase 2 | ✅ 已修复 (2026-06-09) — `train.compute_structured_aux_losses` 入口新增显式末维契约断言(`queue/cycle/queue_rollout` 末维必须严格 6,否则 `AssertionError`);`models.compute_queue_targets` / `models.build_cycle_features` 文档化 6 维 dim→loss-type 映射,并新增 `build_queue_targets_signature` / `build_cycle_features_signature` 两个源码契约 helper(被测试引用,守卫 reorder);`compute_structured_aux_losses` docstring 显式声明 **main aux = 末帧监督** 与 **rollout aux = 全序列监督** 的有意 asymmetry;新增 11 个单元测试覆盖 dim 契约、主/rollout 不对称、源码守卫与 BCE/CE 鲁棒性,测试总数 132 → 143,全过 |
| #21 | best-of-K 采样次数未对齐校验 | CA | 🟡 Medium | Phase 4 | ✅ 已修复 (2026-06-09, Stage 39 双状态模型强化) — `train.py` 新增 `NumValSamplesTracker` 类与 `build_num_val_samples_signature()` 结构化契约 helper(返回 `dict`, 锁定 `checkpoint_key` / `runtime_arg` / `eval_arg` / `must_persist_positive_int`);tracker 采用**双状态模型**(`_runtime_num_val_samples` 永不被 restore 覆盖 + `_checkpoint_num_val_samples` 仅用于 `check_alignment` 诊断), 修复了"只报警、不修正 payload"导致的"加载旧 ckpt K=20 后, 后续 save_checkpoint 写入 K=20 而非当前 args K=4"污染漏洞;`save_checkpoint` 两个调用点(generator_only / D-G 交替)均把 `num_val_samples_tracker.checkpoint_payload()` 写进 ckpt 字典;`main` 加载 checkpoint 时调用 `restore_from_checkpoint` + `check_alignment`, 一致时 info / 不一致时 warning / 旧 ckpt 缺失字段时 info;`evaluate_model.py` 同样 import tracker 做 K 对齐校验;新增 11 个单元测试(原 9 + 双状态定向 2: `test_num_val_samples_tracker_restore_does_not_override_runtime_payload` / `test_num_val_samples_tracker_keeps_runtime_k_after_ckpt_mismatch`), 测试总数 143 → 155, 全过 |
| #22 | `pred_lstm` 消融模式下维度 | CA | 🟡 Medium | Phase 4 | ✅ 已修复 (2026-06-09, Stage 40 + 二次精修日志口径) — 新增 `disable_aux_losses` 统一主开关，一次关闭所有 CycleState 特有功能（state gating / queue rollout / lane queue anchor / decoder state residual）+ 将 aux 权重置零；训练协议日志改用 `(eff)=` 有效状态字段，保证日志口径与模型运行时状态一致 |
| #23 | `phase_duration_limits` 硬编码 | CA | 🟢 Minor | Phase 3 | ✅ 已修复 (2026-06-09) — 把 ``CycleStateTrajectoryGenerator.__init__`` 中的 ``phase_duration_limits=(38.0, 47.0, 2.0)`` 暴露到 train / evaluate 的 ``--phase_duration_limits "R,Y,G"`` CLI；`validate_stage_consistency` 强制长度=3 且非负；训练协议日志打印实际生效值（None 时回退到默认 `(38.0, 47.0, 2.0)`）；17 个 #23 定向测试覆盖 CLI 解析、模型 buffer 传播、参数数量保持、校验拒绝、None 跳过、源码守卫，测试总数 166 → 183 |
| #24 | `cycle_target_last` 只用最后一帧 | CA | 🟢 Minor | Phase 3 | ✅ 已修复 (2026-06-09) — cycle target 从 ``[-1]`` 单帧改为最后 3 帧平均，docstring 与回归测试均已同步 |
| #25 | `light_input_size` 不一致风险 | CA | 🟢 Minor | Phase 5 | ✅ 已修复 (2026-06-09) — 新增 4 个单元测试锁定 Generator(5)/Discriminator(4) |
| #26 | 状态指标命名不一致 | CA | 🟢 Minor | Phase 5 | ✅ 已修复 (2026-06-09) — debug key ``decoder_state_step_residual_norm_seq`` → ``decoder_state_step_residual_norm`` |
| #27 | `traffic_context["scene"]` 字段重复 | CA | 🟢 Minor | Phase 5 | ✅ 已修复 (2026-06-09) — 移除重复的 ``seq_start_end`` 字段 |
| #28 | 随机种子传播链路不完整 | CA | 🟡 P0-级 | Phase 0 | ✅ 已修复 (2026-06-09) — ``init_hidden_*`` 全部从 ``torch.randn`` 改为 ``torch.zeros``；teacher forcing 中 ``random.random()`` 改为 ``torch.rand(1).item()``；DataLoader 新增 ``seed_worker`` + ``generator`` 确定性加载, 新增 8 个单元测试 |
| #29 | 数据集归一化参数未持久化 | CA | 🟡 Medium | Phase 4 | ✅ 已修复 (2026-06-09) — 新增 ``norm_params()`` / ``load_norm_params()`` 方法；train.py 两个 checkpoint 均写入 ``norm_params``；train/evaluate 加载时恢复；向后兼容旧 ckpt (None 不报错) |
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

> **审计结论** (2026-06-07 17:00, 更新于 2026-06-07 20:30,再次更新 2026-06-07 21:00,再次更新 2026-06-07 21:30,再次更新 2026-06-08 00:20,再次更新 2026-06-08 00:45,再次更新 2026-06-08 01:10,再次更新 2026-06-08 19:30,再次更新 2026-06-08 Stage 34,再次更新 2026-06-08 Stage 35,再次更新 2026-06-09 Stage 36,再次更新 2026-06-09 Stage 37,再次更新 2026-06-09 Stage 38,再次更新 2026-06-09 Stage 39,再次更新 2026-06-09 Stage 40,再次更新 2026-06-09 Stage 41,再次更新 2026-06-09 Stage 42 #24 终审修复 + #28 种子传播修复 + #29 归一化持久化):
> - ✅ **已修复**: **28/35** (#1, #2, #3, #4, #5, #6, #7, #8, #9, #10, #11, #12, #13, #14, #15, #16, #17, #18, #19, #20, #22, #23, #24, #25, #26, #27, #28, #29, G1, G2)
> - ⚠️ **部分改善**: **1/35** (G3)
> - ❌ **未修复**: **6/35** (其余未修复项)
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
> **当前进度**: **2/7 已完成** (2026-06-07 审计，移除 #1 后从 8 减为 7，#2 与 #3 已修复)

### 2.1 #2 · `state_loss` 引用未定义的 `loss_mask`（死代码）

- **文件**: `D2TP/utils.py` 行 149
- **当前状态**: ✅ 已修复 (2026-06-07)。`state_loss` 签名补 `loss_mask=None`；`mode='average'` 分支在 mask 缺省时回退为 `torch.ones(T, V)` 默认掩膜，避免 NameError；同时补 docstring 说明其为死代码占位。
- **修改方案**: 在函数签名中显式声明 `loss_mask=None`；仅在 `mode='average'` 且 mask 为 None 时构造默认全 1 掩膜；`mode='sum'/'raw'` 路径完全绕过 mask。
- **验证**:
  - `python -m py_compile D2TP/utils.py D2TP/train.py D2TP/models.py D2TP/evaluate_model.py`
  - 3 个新单元测试（`test_state_loss_signature_exposes_loss_mask` / `test_state_loss_average_mode_uses_loss_mask` / `test_state_loss_is_not_invoked_by_active_training_path`），测试总数 41 → 44，全过
  - 旧调用方式（不传 loss_mask）保留 backward-compat
- **影响范围**: 该函数仍属死代码（`train.py` 仅 `import` 不调用），但其对外契约已与 `l2_loss` 对齐，未来若被外部脚本调用不再触发 NameError。

### 2.2 #3 · 训练/推理 rollout offset 不一致

- **文件**: `D2TP/models.py` 行 1799 (训练) vs 1865 (推理)
- **当前状态**: ✅ 已修复 (2026-06-07)。训练循环内 `last_rollout_offset = input_t.squeeze(0) if teacher_force else output` 改为 `last_rollout_offset = output`，与推理分支完全对齐。
- **修改方案**:
  ```python
  # 训练时始终使用模型自身预测作为 rollout offset (Phase 0 #3 修复)
  last_rollout_offset = output  # 无论 teacher_force 与否
  ```
- **关键**: rollout offset 更新位置在 `pred_hidden2pos` 之后、下一轮 `rollout_queue_step` 调用之前，确保 step `i+1` 的 queue rollout 看到的是 step `i` 的模型预测。
- **bug 影响**: 修复前训练时 `teacher_forcing_ratio=0.8` 下，queue rollout 分支 80% 看到 GT future displacement，20% 看到模型预测；而推理时 100% 看到模型预测。这是典型的 exposure bias / train-eval distribution shift，会让 queue rollout gate/MLP 在训练时学习一个"信号"分布，在推理时遇到完全不同的"信号"分布。
- **验证**:
  - `python -m py_compile D2TP/models.py D2TP/train.py D2TP/evaluate_model.py D2TP/utils.py` 全部干净
  - 2 个新单元测试（`test_rollout_offset_uses_model_own_output_under_teacher_forcing` / `test_rollout_offset_under_teacher_forcing_matches_eval_at_step_zero`），测试总数 44 → 46，全过
  - 1-batch smoke (`train.py --train_stage warmup --max_train_batches 1`) 正常完成，loss 与稳定性指标数值合理（GradNorm ≈ 173.21, QRollHNorm ≈ 0.20, PredOffsetNorm ≈ 0.19）

### 2.3 #4 · `cycle_target_last` 维度尺度不匹配

- **文件**: `D2TP/train.py` 行 430-459, `D2TP/models.py` 行 959-969、1454-1460、1700-1717
- **当前状态**: ✅ 已修复。`queue_aux_head` / `cycle_aux_head` 已拆分为 5 个独立子空间头：`queue_aux_reg_head (4 维) + queue_aux_cls_head (2 维) + cycle_aux_phase_head (3 维) + cycle_aux_time_head (2 维) + cycle_aux_change_head (1 维)`，消除了回归 / 分类 / phase / time / change 梯度共享同一 `Linear(..., 6)` 参数的问题；`compute_structured_aux_losses` 入口新增 pred/target 形状契约断言，避免静默错位。
- **实现结果**:
  1. `rollout_queue_step` 与 forward 中统一用 `torch.cat` 拼接各子头输出，保持末维 6 维契约不变
  2. `compute_structured_aux_losses` 对 `queue/cycle/queue_rollout` 三类 pred-target pair 增加 `assert pred.shape == target.shape`
- **验证**:
  1. `test_aux_heads_split_into_independent_regression_and_classification_modules`
  2. `test_aux_pred_last_outputs_are_concatenation_of_subspace_heads`
  3. `test_structured_auxiliary_losses_asserts_pred_target_shape_match`

### 2.4 #7 · `maybe_load_compatible_weights` 不恢复 `start_epoch`

- **文件**: `D2TP/train.py` 行 898-915
- **当前状态**: ✅ 已修复。cyclestate 兼容加载分支在 `maybe_load_compatible_weights(...)` 之后补上 `if "epoch" in checkpoint: args.start_epoch = checkpoint["epoch"]`，行为与非 cyclestate resume 分支一致。
- **实现结果**:
  1. resume 一个 cyclestate checkpoint 时，主循环 `range(args.start_epoch, args.num_epochs + 1)` 会从 checkpoint epoch 连续恢复
  2. 避免 LR scheduler、TensorBoard 标号、日志/ckpt 命名从 epoch 0 重新错位
- **验证**:
  1. `test_compatible_resume_restores_start_epoch`
  2. `test_main_resume_for_cyclestate_calls_compatible_loader_and_restores_epoch`

### 2.5 #17 · `pred_state` oracle 假设未声明（⚠️ 方法论关键问题）

- **文件**: `README.md`, `docs/cyclestate_research_story.md`
- **当前状态**: ✅ 已修复 (2026-06-09, Stage 35)。Phase 0 文档部分已完成:README.md 与 `docs/cyclestate_research_story.md` 均新增 `## ⚠️ Oracle 假设声明` 区块,明确 `pred_state` 来源(数据集中记录的预测期实际信号灯相位与已运行时间)、oracle 性质(模型访问推理时不可用的真实未来信号)、与 D2-TPred baseline 的对齐性(两者都假设未来信号状态已知),以及 Phase 0.5 后续信号退化实验计划。Phase 0.5 实验部分(信号退化曲线)待执行。
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
- **当前状态**: ✅ 已修复 (2026-06-09)
- **实际修改**:
  1. 保持 `apply_stage_defaults(args)` 只负责“阶段默认值补齐”，不把配置校验和默认值应用混在一起
  2. 新增 `validate_stage_consistency(args)` 专门承担联动一致性检查，并在 `main()` 中于 `apply_stage_defaults(args)` 之后显式调用
  3. `gan_weight < 0` 现在是硬错误，不再被错误地降级成 warning；这是必要修复，因为训练总损失直接包含 `g_loss * args.gan_weight`
- **校验规则**:
  - 硬错误: `gan_weight < 0`、`gan_weight > 0 && generator_only=True`、负 `grad_clip`、负 `rollout_residual_scale`、`teacher_forcing_ratio ∉ [0,1]`、负 aux 权重
  - 软警告: `train_stage == "adversarial" && gan_weight == 0`、`aux_rollout_weight > 0 && aux_queue_weight == 0`
- **验证**:
  - #19 定向测试 15 项全过
  - 其中新增回归守卫 `test_validate_stage_consistency_raises_on_negative_gan_weight`，复现并封堵了“负 GAN 权重被放行、进而翻转对抗项优化方向”的漏洞

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

- **文件**: `D2TP/models.py`
- **当前状态**: ✅ 已修复 (2026-06-09)
- **实际修改**:
  1. `TrajectoryGenerator` 与 `CycleStateTrajectoryGenerator` 的 train/eval 解码循环均在每步调用 `inject_per_step_decoder_noise(...)`
  2. 新增 `expand_scene_noise_to_batch(scene_noise, seq_start_end)` 统一 scene-level noise 到 batch-level hidden 的展开逻辑
  3. 新增 `inject_per_step_decoder_noise(pred_lstm_hidden, seq_start_end, noise_scale=0.1)` 统一 step-noise 注入,避免基类/子类四处复制
- **验收证据**:
  1. 定向测试 8 项全过,覆盖 base/cyclestate 两个生成器的 train/eval 路径
  2. 强行为断言不再依赖“forward 两次输出不同”这种弱随机性证据,而是固定初始化噪声后只改变 decode-step noise,断言输出必须变化
  3. `get_noise` 调用次数守卫:每次 forward 必须恰好是 `1 次 init + pred_len 次 per-step`
  4. `python -m unittest tests.test_cyclestate_protocol` 117 项全过; `py_compile D2TP/{models,train,evaluate_model,utils}.py tests/test_cyclestate_protocol.py` 干净

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
> - `cycle` aux 输出头保持独立 `phase/time/change` 子空间，各头参数量独立
> - 预测期 cycle LSTM 状态更新频率 = 观测期更新频率（每步更新一次）
> **预计工作量**: 2-3 天
> **前置条件**: Phase 1 完成（训练稳定后才有意义验证结构性修改）。注意：部分 Phase 2 问题 (#6 phase_change, G6 cycle memory) 可能本身是 Phase 1 不稳定的诱因——如果 Phase 1 在实验 A-D 全部失败后仍未收敛，应考虑提前修复 #6 和 G6。
> **当前进度**: **0/3 已完成**

### 4.1 #6 · `get_step_cycle_feature` 缺失 `phase_change`

- **文件**: `D2TP/models.py` 行 1082-1105、1386-1406
- **当前状态**: ✅ 已修复。`get_step_cycle_feature(state_frame, prev_phase=None)` 新增 `prev_phase` 形参；当提供上一帧相位时，`phase_change = (phase != prev_phase.long().clamp(0, 2)).float()`，不再把预测期 `phase_change` 恒置 0。`get_decode_step_context` 显式传入上一帧 phase：step 0 用 `obs_state[-1, :, 2]`，step > 0 用 `pred_state[step_index - 2, :, 2]`，与 `build_cycle_features` 的跨帧语义对齐。
- **实现结果**:
  1. 不传 `prev_phase` 或显式传 `None` 时，`phase_change` 退化为全 0，保持旧调用方向后兼容
  2. 真实切换帧的 `phase_change` = 1.0，稳定帧 = 0.0，`cycle_change` BCE 监督项恢复有效信号
- **验证**:
  1. `test_get_step_cycle_feature_emits_phase_change_on_transition`
  2. `test_get_step_cycle_feature_phase_change_backward_compatible`
  3. `test_get_decode_step_context_propagates_phase_change`

### 4.2 #20 · `compute_structured_aux_losses` 语义错位

- **文件**: `D2TP/train.py` 行 608-827, `D2TP/models.py` 行 1379-1446
- **当前状态**: ✅ 已修复 (2026-06-09)
- **实际修改**:
  1. `compute_structured_aux_losses` 入口新增显式末维契约断言:queue/cycle/queue_rollout 三类 target 的末维必须严格等于 6,否则 `AssertionError`;避免 #4 修复后 idx 切片在末维被改时静默错位(MSE 吃到 cls logits / 反之)。
  2. `models.compute_queue_targets` / `models.build_cycle_features` 各自 docstring 显式声明 6 维 dim→loss-type 映射,并新增 `build_queue_targets_signature` / `build_cycle_features_signature` 两个源码契约 helper;helper 现在返回可断言的结构化签名(维度顺序 / loss 类型 / queue source index),被测试直接比对以守卫 reorder。
  3. `compute_structured_aux_losses` docstring 显式声明 **main aux = 末帧监督** (queue/cycle, 取 `aux_info["queue_targets"][-1]` / `aux_info["cycle_feature_seq"][-1]`) 与 **rollout aux = 全序列监督** (queue_rollout, 跨 `T_pred` 步) 的有意 asymmetry,并说明设计理由(末帧与 gated hidden 语义对齐 / rollout 跨步稳定性);任何把这两套方案对调的重构都会在测试中暴露。
  4. `queue_reg_idx = [0,1,2,3]` / `queue_cls_idx = [4,5]` / cycle `[:, :3]` / `[:, 3:5]` / `[:, 5:6]` 切分常量集中声明,避免散落多处。
- **新增 11 个单元测试**:
  1. `test_compute_queue_targets_returns_6_dims_in_canonical_order` - 源码契约:queue 6 维顺序与 `build_queue_targets_signature` 对齐
  2. `test_build_cycle_features_returns_6_dims_in_canonical_order` - 源码契约:cycle 6 维顺序与 `build_cycle_features_signature` 对齐
  3. `test_compute_structured_aux_losses_docstring_documents_dim_semantics` - docstring 显式声明 dim→loss-type 映射
  4. `test_compute_structured_aux_losses_docstring_documents_main_vs_rollout_asymmetry` - docstring 显式声明 main/rollout 不对称
  5. `test_train_call_site_uses_last_frame_for_main_and_sequence_for_rollout` - 源码守卫:train.py 调用点用 `[-1]` 末帧做 main、传完整 seq 做 rollout
  6. `test_compute_structured_aux_losses_rejects_queue_target_wrong_dim` - 末维 ≠ 6 必须 `AssertionError`
  7. `test_compute_structured_aux_losses_rejects_cycle_target_wrong_dim` - 末维 ≠ 6 必须 `AssertionError`
  8. `test_compute_structured_aux_losses_bce_target_must_be_probability` - 合法 0/1 target 下 BCE/CE finite 且非负
  9. `test_compute_structured_aux_losses_phase_argmax_handles_label_noise` - phase argmax 在 tie 标签下仍 finite
  10. `test_compute_structured_aux_losses_rollout_seq_flattens_over_time` - rollout 在 (T*batch) flatten 后做均匀 reg/cls 切分
  11. `test_compute_structured_aux_losses_main_uses_last_frame_not_sequence` - main aux 在 2-D (batch,6) target 上**不**做隐式时间归约
- **验收证据**:
  - `python -m unittest tests.test_cyclestate_protocol` 143 项全过(132 → 143,新增 11 项 #20 定向测试)
  - `python -m py_compile D2TP/{train,models,utils,evaluate_model}.py tests/test_cyclestate_protocol.py` 干净
  - 1-batch warmup smoke (`python D2TP/train.py --train_stage warmup --max_train_batches 1 --num_epochs 1 --batch_size 4`) 正常完成,loss 与稳定性指标合理

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
- **当前状态**: ✅ 已修复。`rollout_queue_features` 的 magic number 已集中到 `RolloutQueueCoefs`，并通过 `__init__` / `--rollout_queue_coefs_json` 暴露；非法 JSON、非 dict、字段值类型错误都会 warning 并回退默认值。
- **后续方向**: 当前只完成“暴露与可配置化”；更进一步的“可学习化”仍可按原计划评估是否引入 MLP / meta-parameter 学习。

### 5.2 G4 · meso/macro 分支容量提升

- **文件**: `D2TP/models.py` 行 836-837
- **当前状态**: ❌ 未修改。`queue_lstm_hidden_size=32`、`cycle_lstm_hidden_size=16` 不变。
- **修改方案**: 同原计划

### 5.3 G5 · decoder state residual 注入位置扩展

- **文件**: `D2TP/models.py` 行 1794-1795
- **当前状态**: ❌ 未修改。仅在 `pred_lstm_hidden = pred_lstm_hidden + step_state_residual` 注入，未扩展到 cell 或 output。
- **修改方案**: 同原计划

### 5.4 #23 · `phase_duration_limits` 可配置化

- **文件**: `D2TP/models.py` L1000 (默认) + L1055-1058 (register_buffer), `D2TP/train.py` L254-292 (CLI + 解析) + L1390-1397 (model_kwargs 透传) + L662-681 (校验) + L1454-1475 (日志), `D2TP/evaluate_model.py` L121-157 (CLI + 解析) + L241-246 (model_kwargs 透传)
- **当前状态**: ✅ 已修复 (2026-06-09)
- **实际修改**:
  1. `train.py` 新增 `--phase_duration_limits` CLI（自定义 ``_parse_phase_duration_limits`` type 解析逗号分隔 3 floats，长度/格式/非负都校验；空字符串/None 触发 ``__init__`` 默认）
  2. `train.py` 在 `validate_stage_consistency` 中补一道防御性校验（避免外部代码路径写入非 tuple 或负数）
  3. `train.py` 把 `args.phase_duration_limits` 在 `args.model_type == "cyclestate"` 分支透传给 `model_kwargs["phase_duration_limits"]`（`None` 时跳过，让 `__init__` 默认值生效）
  4. `train.py` 训练协议日志加 `phase_duration_limits=%s` 字段，`None` 时打印 `(38.0, 47.0, 2.0)` 默认值，便于审计
  5. `evaluate_model.py` 镜像同样的 CLI + type 解析 + model_kwargs 透传，行为与训练侧对齐
  6. `models.py` 无结构性改动；`phase_duration_limits` 仍走 `register_buffer` 机制，可随 `.to(device)` 迁移
- **新增 17 个单元测试**:
  1. `test_train_cli_phase_duration_limits_defaults_to_none`
  2. `test_train_cli_phase_duration_limits_parses_three_floats`
  3. `test_train_cli_phase_duration_limits_rejects_wrong_length`
  4. `test_train_cli_phase_duration_limits_rejects_non_float`
  5. `test_train_cli_phase_duration_limits_rejects_negative`
  6. `test_evaluate_model_cli_phase_duration_limits_parses_three_floats`
  7. `test_evaluate_model_cli_phase_duration_limits_defaults_to_none`
  8. `test_cyclestate_default_phase_duration_limits_is_38_47_2`
  9. `test_cyclestate_custom_phase_duration_limits_propagates_to_buffer`
  10. `test_cyclestate_phase_duration_limits_preserves_default_param_count`
  11. `test_validate_stage_consistency_rejects_wrong_length_phase_duration`
  12. `test_validate_stage_consistency_rejects_negative_phase_duration`
  13. `test_validate_stage_consistency_accepts_valid_phase_duration`
  14. `test_train_main_forwards_phase_duration_limits_to_model`
  15. `test_train_main_skips_phase_duration_limits_when_none`
  16. `test_train_log_includes_phase_duration_limits_field`
  17. `test_evaluate_model_passes_phase_duration_limits_to_constructor`
- **验收证据**:
  - `python -m unittest discover -s tests -p 'test_cyclestate_protocol.py'` 183 项全过（166 → 183，+17 项 #23 定向测试）
  - `python -m py_compile D2TP/{train,models,evaluate_model}.py` 干净
  - 1-batch cyclestate warmup smoke with `--phase_duration_limits "40,50,3"` 训练协议日志打印 `phase_duration_limits=(40.0, 50.0, 3.0)`，checkpoint 恢复 `start_epoch=129` 正常
  - 1-batch cyclestate warmup smoke without flag 训练协议日志打印 `phase_duration_limits=(38.0, 47.0, 2.0)`（`__init__` 默认），向后兼容

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
> **当前进度**: **1/6 已完成 (#21)。三级标签体系 (`smoke/protocol-check/comparable`) 已建立并用于最新实验**

### 6.1 #21 · best-of-K 采样次数对齐

- **文件**: `D2TP/train.py` 行 320-464 (`build_num_val_samples_signature` / `NumValSamplesTracker` 双状态模型), 行 1383-1432 (main 加载), 行 1481-1535 (save_checkpoint), `D2TP/evaluate_model.py` 行 15-21 + 200-219
- **当前状态**: ✅ 已修复 (2026-06-09, Stage 39 双状态模型强化)
- **实际修改**:
  1. `train.py` 新增 `build_num_val_samples_signature()` 契约 helper(同 #20 模式),**强化为返回结构化 dict**, 锁定所有关键字段名与约束:
     ```python
     return {
         "checkpoint_key": "num_val_samples",
         "runtime_arg": "num_val_samples",
         "eval_arg": "num_samples",
         "must_persist_positive_int": True,
     }
     ```
  2. `train.py` 新增 `NumValSamplesTracker` 类,封装 K 值存档 / 恢复 / 对齐校验语义,采用**双状态模型**避免旧 ckpt 污染当前运行时:
     - `_runtime_num_val_samples`: 当前这次运行真正使用的 K 值(来自 `__init__`),**永不**被 `restore_from_checkpoint` 修改;
     - `_checkpoint_num_val_samples`: 从旧 ckpt 读出的历史 K 值,只用于 `check_alignment` 做诊断比对,**不**参与 `checkpoint_payload()` 的写回逻辑;
     - `value` / `checkpoint_payload()` — 永远返回 `_runtime_num_val_samples`,保证 `save_checkpoint` 写入的是当前 args K 而非旧 ckpt K;
     - `restore_from_checkpoint(ckpt_val)` — 容忍 `None` / `int` / `tensor` / `float` / 非法值(str / 0 / 负数),非法静默回退,**不**抛异常,只更新 `_checkpoint_num_val_samples`;
     - `check_alignment(args_val)` — 使用 `_checkpoint_num_val_samples` 做诊断比对,返回 `(is_aligned: bool, message: str)`,分别处理缺失/一致/mismatch 三种情形。
  3. `main` 中实例化 `num_val_samples_tracker = NumValSamplesTracker(num_val_samples=getattr(args, "num_val_samples", None))`;checkpoint 加载分支依次调用 `restore_from_checkpoint` + `check_alignment`:
     - **一致** → `logging.info`;
     - **缺失** (旧 ckpt 升级) → `logging.info` 提示沿用当前 args;
     - **不一致** → `logging.warning` 提示"`checkpoint K=20 与 args.num_val_samples=4 不一致`, 建议 `evaluate_model.py --num_samples 20`"。
  4. 两处 `save_checkpoint` 调用(generator_only 分支 / D-G 交替分支)都把 `num_val_samples_tracker.checkpoint_payload()` 写进 checkpoint 字典。**关键**: payload 永远是当前 runtime K,不会被旧 ckpt 的 K 污染。
  5. `evaluate_model.py` 同步: import `NumValSamplesTracker`,在 `load_model_from_checkpoint` 末尾用 `args.num_samples` 做对齐校验,行为与 train.py 一致(一致 info / 缺失 info / 不一致 warning)。
- **核心闭环**: Stage 39 修复了"只报警、不修正 payload"的核心漏洞 — 旧实现下 `restore_from_checkpoint(20)` 会覆盖 runtime K=4,导致后续 `save_checkpoint` 写入 K=20 而非当前 args K=4;双状态模型确保 `checkpoint_payload()` 永远返回 runtime K,即便 `check_alignment` 已经报警不一致,落盘的也是用户当前想要保存的 K 值。
- **新增 11 个单元测试** (Stage 38 9 项 + Stage 39 双状态定向 2 项):
  1. `test_num_val_samples_signature_returns_true_and_acts_as_contract` — 契约 helper 返回**结构化 dict** 而非裸 True,断言 `checkpoint_key` / `runtime_arg` / `eval_arg` / `must_persist_positive_int` 四个字段
  2. `test_num_val_samples_tracker_stores_int_value` — 构造时把 K 存为 int,缺失为 None
  3. `test_num_val_samples_tracker_restore_from_int_ckpt` — int / tensor / None / 0 / 负数 / str 各种类型的鲁棒性,以及 `value` 不被覆盖
  4. `test_num_val_samples_alignment_silent_when_match` — K 一致时 `is_aligned=True` + message 含 "一致"
  5. `test_num_val_samples_alignment_warns_on_mismatch` — K 不一致时 `is_aligned=False` + message 含 "best-of-K" + "--num_samples"
  6. `test_num_val_samples_alignment_silent_on_missing_ckpt_key` — 旧 ckpt 缺失字段时 `is_aligned=True` + message 含 "缺失"
  7. `test_train_save_checkpoint_persists_num_val_samples` — 源码守卫:`save_checkpoint` 至少 2 处写入 `"num_val_samples"` 字段
  8. `test_train_main_restores_num_val_samples_from_checkpoint` — 源码守卫: `main` 加载 ckpt 时调 `restore_from_checkpoint` + `check_alignment`
  9. `test_evaluate_model_aligns_num_samples_with_checkpoint` — 源码守卫: `evaluate_model.py` import `NumValSamplesTracker` + 引用 `num_val_samples`
  10. `test_num_val_samples_tracker_restore_does_not_override_runtime_payload` — **Stage 39 强行为断言**: 构造时 K=4 + restore(20) 后,`checkpoint_payload()` 必须仍返回 4,而非 20
  11. `test_num_val_samples_tracker_keeps_runtime_k_after_ckpt_mismatch` — **Stage 39 强行为断言**: 模拟"加载旧 ckpt K=20 后再 save_checkpoint"场景,断言 payload 是运行时 K=4,确保落盘正确
- **验收证据**:
  - `python -m unittest tests.test_cyclestate_protocol` 155 项全过(143 → 155,新增 11 项 #21 定向测试,其中 Stage 39 新增 2 项双状态回归)
  - `python -m py_compile D2TP/{train,models,utils,evaluate_model}.py` 干净
  - 直接 `NumValSamplesTracker` smoke test 7 个分支全部通过
  - **核心验收标准** (用户验收条件): 加载 K=20 的旧 ckpt 后,`save_checkpoint` 写入新 ckpt 的 K 仍然是当前 runtime K=4,绝不变成 20

### 6.2 #29 · 数据集归一化参数持久化

- **当前状态**: ✅ 已修复 (2026-06-09)。新增 ``norm_params()`` / ``load_norm_params()`` 方法实现数据集归一化参数的显式序列化；train.py 的两个 ``save_checkpoint`` 调用均写入 ``norm_params``；train.py resume 和 evaluate_model.py 加载时恢复，向后兼容旧 checkpoint (``None`` 不报错)。新增 10 个定向单元测试。

### 6.3 #22 · 消融实验公平性

- **文件**: `D2TP/models.py` L1005-1019, `D2TP/train.py` L207-214 + L1284-1293 + L1335 + L1371-1401, `D2TP/evaluate_model.py` L97-104 + L200
- **当前状态**: ✅ 已修复 (2026-06-09, Stage 40；2026-06-09 二次精修日志口径,测试总数 164→166)
- **实际修改**:
  1. `CycleStateTrajectoryGenerator.__init__` 新增 `disable_aux_losses=False` 参数；当 True 时强制 `disable_state_gating` / `disable_queue_rollout` / `disable_lane_queue_anchor` / `disable_decoder_state_residual` 全部为 True
  2. `train.py` 新增 `--disable_aux_losses` CLI 参数（`action="store_true"`），并在 `main()` 中于 `validate_stage_consistency` 之后检查：若 `disable_aux_losses=True` 且 `model_type=cyclestate`，将 `aux_queue_weight` / `aux_cycle_weight` / `aux_rollout_weight` 全部置零，打印 info 日志
  3. `evaluate_model.py` 新增 `--disable_aux_losses` CLI 参数并经 `model_kwargs` 透传给模型构造函数
  4. 模型参数数量不变（架构兼容 checkpoint），仅运行时行为切换
  5. **(2026-06-09 精修)** 训练协议日志改用 `_eff_disable_*` 变量（`disable_aux_losses or args.disable_*`），格式串字段名加 `(eff)` 后缀，并新增 `disable_aux_losses=%s` 字段；保证日志口径与模型运行时真实状态一致（之前日志打印原始 `args.disable_*`，在 `disable_aux_losses=True` 时与模型实际行为不一致，影响消融实验可审计性）
  6. **(2026-06-09 精修)** 两条源码守卫测试改用 `Path.read_text(encoding="utf-8")`，避免 `ResourceWarning`
- **新增 9 个单元测试**:
  1. `test_disable_aux_losses_forwards_to_all_four_individual_flags` — 四个独立标志位均被强制为 True
  2. `test_disable_aux_losses_forward_output_correct_shape` — forward 仍正常生成 (12, batch, 2) 轨迹
  3. `test_disable_aux_losses_nulls_all_aux_debug_fields` — 所有 CycleState 特有 debug 字段均为 None
  4. `test_disable_aux_losses_preserves_model_architecture_parameters` — 参数数量与默认模型一致
  5. `test_train_cli_accepts_disable_aux_losses` — train.py argparse 正确识别
  6. `test_evaluate_model_cli_accepts_disable_aux_losses` — evaluate_model.py argparse 正确识别
  7. `test_train_main_zeros_aux_weights_on_disable_aux_losses` — 源码守卫: aux 权重被置零
  8. `test_evaluate_model_passes_disable_aux_losses_to_constructor` — 源码守卫: evaluate_model 透传
  9. `test_disable_aux_losses_aux_heads_still_exist_but_unused` — aux 头结构保留（checkpoint 兼容）
  10. **(2026-06-09 精修)** `test_train_log_uses_effective_disable_flags_not_raw_args` — 源码守卫：日志格式串含 `(eff)=` 与 `disable_aux_losses=%s`
  11. **(2026-06-09 精修)** `test_train_log_effective_flags_match_model_runtime_state` — 行为守卫：`_eff_disable_*` 变量在主开关 ON/OFF 下分别等于全 True / 原始 args
- **验收证据**:
  - `python -m unittest tests.test_cyclestate_protocol` 166 项全过（164 → 166，+2 项日志精修测试）
  - `python -m py_compile D2TP/{models,train,evaluate_model}.py` 干净
  - `python -W error::ResourceWarning -m unittest ...` 无 `ResourceWarning`（两条源码守卫测试改用 `Path.read_text`）
  - 1-batch cyclestate warmup smoke with `--disable_aux_losses` 正常完成: 所有 aux 损失为 0，L2_loss 合理（1856 → 1769），GradNorm 合理（414 → 401）
  - 1-batch cyclestate warmup smoke without flag（正常路径）aux 损失非零，行为未退化
  - **(2026-06-09 精修)** smoke 捕获日志：`disable_aux_losses=True` 时日志正确显示 4 个 `(eff)=True` + `disable_aux_losses=True`，与模型运行时状态一致

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
> **当前进度**: **10/16 已完成**（#1, #5, #8, #9, #10, #11, #12, #13, #14, #15 已完成；#2 在 Phase 0 已修复，故不再计入 Phase 5 计数）

### 7.1 代码清理（#1, #5, #8, #9, #11, #12, #14, #15）

| 编号 | 当前状态 | 修改内容 |
|------|----------|----------|
| #1 | ✅ | 将 `D_step=2` 移入 epoch 循环（代码风格，非 bug：D_step 总是在 0↔2 间循环，跨 epoch 不重置仅影响 epoch 边界的 D/G 交替节奏） — 2026-06-07 已修复 |
| #5 | ✅ | 移除 `CUDA_VISIBLE_DEVICES` 硬编码 (train.py:230)；改为完全遵循用户 shell / 环境变量设置，并新增 `test_cuda_visible_devices_not_hardcoded_in_train_py` 防回归 |
| #8 | ✅ | `build_lane_queue_anchor_seq` 已改为 `repeat_interleave + index_add_` 的向量化实现；保留逐帧 `(t, scene, lane)` 语义，并新增 3 个单元测试守护等价性与隔离性 |
| #9 | ✅ | `relation_Matrix` 已改为全向量化 `torch` 实现；保留距离门控和扇区 wrap-around 语义，移除 Python 循环与 numpy 路径，并新增 4 个单元测试守护 |
| #11 | ✅ | `graph_lstm_model` 保留 `nn.LSTMCell` 实例(用于 checkpoint 兼容),加 `_graph_lstm_call_count` + `forward_hook` 作为调用诊断,并用 3 个回归测试守住"forward 主路径未调用但直接调用可观测"契约 — 2026-06-08 已修复 |
| #12 | ✅ | train.py / models.py active code 的 `seq_start_end.data` 已全部替换为 `.tolist()`,新增 5 个测试(双文件源码守卫 + 双文件 tolist 模式 + Python int 契约) — 2026-06-08 已修复 |
| #14 | ✅ | 新增 `BestAdeTracker` 类,`main` 局部实例化并移除 `global best_ade` 声明入口,checkpoint 加载时 `restore_from_checkpoint` 灌入 ckpt best_ade;新增 7 个测试覆盖存在性/初始值/update/restore/多实例隔离/不再模块级/无 global 声明 — 2026-06-08 已修复 |
| #15 | ✅ | `utils.set_logger` 把 `if not logger.handlers` 改为 `if not logger.hasHandlers()`(沿 logger 层级递归检查父 logger),新增 2 个测试(源码守卫 + handler 数量不变) — 2026-06-08 已修复 |

### 7.2 命名与接口统一（#10, #25, #26, #27, #30）

| 编号 | 当前状态 | 修改内容 |
|------|----------|----------|
| #10 | ✅ | `_mean_norm_from_tensor` 补详细 docstring,标注 0/1/≥2 维分情况返回语义;新增 7 个测试覆盖 None/0-d/1-d/2-d/3-d/与 Frobenius 范数的差异 — 2026-06-08 已修复 |
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
- [x] #2 state_loss loss_mask（死代码） — ✅ 已修复 (2026-06-07)
- [x] #3 rollout offset 一致性 — ✅ 已修复 (2026-06-07)
- [x] #4 cycle_aux scale — ✅ 已修复 (2026-06-08)
- [x] #7 start_epoch 恢复 — ✅ 已修复 (2026-06-08)
- [x] #17 oracle 假设声明（文档部分） — ✅ 已修复 (2026-06-08, Stage 35)
- [x] #19 stage defaults 断言 — ✅ 已修复 (2026-06-09, Stage 37)
- [x] #28 随机种子完整 — ✅ 已修复（init_hidden→zeros, random.random→torch.rand, seed_worker+generator）
- [x] 所有 Phase 0 单元测试通过 — ✅ 相关协议测试已补齐并通过（62 tests）

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
- [x] #18 add_noise per-step 验证 — ✅ 已修复 (2026-06-09)
- [ ] 稳定性告警机制生效 — ⚠️ 指标已记录但无自动告警阈值

### Milestone 2: Phase 2+3 完成——方法对齐
- [x] #6 phase_change 完整实现 — ✅ 已修复 (2026-06-08)
- [x] #20 aux loss 拆分 — ✅ 已修复 (2026-06-09, Stage 38; 11 个 #20 定向测试新增,测试总数 132→143,全过)
- [ ] G6 cycle 预测期滚动 — ❌ 未修复
- [x] #16 魔法常数暴露/可学习 — ✅ 已修复 (2026-06-08, Stage 34; 可学习化留待后续)
- [ ] G4 容量提升 — ❌ 未修复
- [ ] G5 残差注入扩展 — ❌ 未修复
- [ ] 消融实验对比 A/B — ❌ 未执行

### Milestone 3: Phase 4 完成——实验体系就绪
- [x] #21 best-of-K 对齐 — ✅ 已修复 (2026-06-09, Stage 39 双状态模型强化; 11 个 #21 定向测试,测试总数 143→155,全过;核心闭环:"只报警、不修正 payload"已修复)
- [x] #29 归一化参数持久化 — ✅ 已修复（norm_params/load_norm_params + train/evaluate ckpt 读写）
- [x] #22 消融实验公平性 — ✅ 已修复 (2026-06-09, Stage 40; 9 个 #22 定向测试,测试总数 155→164,全过;`--disable_aux_losses` 统一主开关打通 CLI → model_kwargs → __init__ → aux 权重置零 完整链路)
- [ ] #31 消融开关集中管理 — ❌ 未修复
- [ ] baseline 完整审计 — ❌ 未执行
- [ ] CycleState full pipeline 完成 — ❌ 未执行
- [ ] 最终消融表 — ❌ 未执行

### Milestone 4: Phase 5 完成——代码就绪
- [ ] 所有轻微问题修复 — ❌ (10/16,#1/#5/#8/#9/#10/#11/#12/#13/#14/#15 已完成,其余 #25/#26/#27/#30/#32/#33 未完成)
- [x] #1 D_step 移至 epoch 循环内（代码风格） — ✅ 已修复 (2026-06-07)
- [x] #10 `_mean_norm_from_tensor` 命名歧义(分情况返回语义) — ✅ 已修复 (2026-06-08)
- [x] #11 `graph_lstm_model` 保留但未使用,加调用计数诊断与回归测试 — ✅ 已修复 (2026-06-08)
- [x] #12 `seq_start_end.data` 在 train.py / models.py active code 全部替换为 `.tolist()` — ✅ 已修复 (2026-06-08)
- [x] #13 `D_train` 与 generator tensorboard step 统一改用 `global_step` 跨 epoch 单调递增 — ✅ 已修复 (2026-06-08)
- [x] #14 `best_ade` 模块级全局变量改为 `BestAdeTracker` 局部实例,并移除 `global best_ade` 声明 — ✅ 已修复 (2026-06-08)
- [x] #15 `utils.set_logger` 改用 `logger.hasHandlers()` 避免子 logger 重复挂载 — ✅ 已修复 (2026-06-08)
- [ ] 文档交叉引用完整 — ❌ 未修复
- [ ] `run_full_pipeline.sh` 可用 — ❌ 未创建
- [x] 新增测试通过 — ✅ `test_cyclestate_protocol.py` 已覆盖 #1 DStep 调度、#2 state_loss 契约、#3 rollout offset 一致性、#4 aux 子空间拆分、#5 CUDA 环境变量硬编码、#6 phase_change 预测期传播、#7 resume epoch 恢复、#8 lane anchor 向量化、#9 relation_Matrix 向量化、#10 _mean_norm_from_tensor 分情况语义、#11 graph_lstm_model 未使用/直接调用计数、#12 train/models 双文件 tolist 替换、#13 g/d tensorboard global_step 统一、#14 BestAdeTracker 局部化/无 global 声明、#15 hasHandlers 检查
- [ ] Phase 5a (文档/命名/测试) 可在 Phase 0/1 期间完成 — ⏸ 部分启动(#10/#14 已完成,#25/#26/#27/#30/#32 未启动)
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
| 2026-06-07 17:00 | — | PLAN.md 全面状态审计 | AI | 基于最新代码 (0b99b6e) 审计全部 35 个问题状态,结论:2 已修复,6 部分改善,27 未修复 |
| 2026-06-07 19:30 | — | 重新分类 #1 / #2 | AI | #1 从 Phase 0/🔴Critical 降级为 Phase 5/🟢Minor(代码风格,非正确性 bug);#2 从 🔴Critical 降级为 🟡Medium(死代码,不影响活跃训练路径) |
| 2026-06-07 20:30 | Phase 5 | #1 D_step 移入 for-epoch 循环 | AI | `D_step=2` 从 train.py:885(epoch 循环外)移入 for-epoch 循环体(line 889);新增 2 个单元测试 `test_dstep_initialization_lives_inside_epoch_loop` / `test_dstep_resets_between_epochs_via_simulated_main_loop` 保护此协议;测试总数 39 → 41,全部通过;warmup 1-batch smoke 验证流程未坏 |
| 2026-06-07 21:00 | Phase 0 | #2 state_loss loss_mask 签名补全 | AI | `state_loss(pred_traj_fake, pred_traj_gt, loss_mask=None, mode='sum')` 在 utils.py:139 补 `loss_mask=None` 形参;`mode='average'` 分支在 mask 缺省时回退为 `torch.ones(T, V)`,避免 NameError;新增 3 个单元测试(签名检查/average 行为/死代码契约),测试总数 41 → 44,全部通过;py_compile utils/train/models/evaluate_model 全部通过;保留 backward-compat 旧调用方式 |
| 2026-06-09 10:30 | Phase 1 | #18 add_noise 每步解码注入验收补强 | AI | `TrajectoryGenerator` / `CycleStateTrajectoryGenerator` 共用 `expand_scene_noise_to_batch` + `inject_per_step_decoder_noise` 两个 helper,统一 scene-noise 展开与 decode-step 注入;把原先只证明“有随机性”的弱测试收紧为 8 个强行为测试:覆盖 base/cyclestate 两个生成器的 train/eval 路径、`get_noise` 调用次数(1 次 init + pred_len 次 step)、以及固定 init noise 后仅修改 step noise 必须改变输出;`python -m unittest tests.test_cyclestate_protocol` 117 项全过,py_compile 干净 |
| 2026-06-09 11:10 | Phase 0 | #19 stage defaults 联动一致性校验 | AI | `train.py` 新增 `validate_stage_consistency(args)` 并在 `main()` 的 `apply_stage_defaults(args)` 之后强制调用,把“默认值补齐”和“配置联动校验”分层处理;修复了负 `gan_weight` 会被 warning 放行的漏洞,现在 `gan_weight < 0` 会直接 `ValueError`;#19 定向测试 15 项全过,新增 `test_validate_stage_consistency_raises_on_negative_gan_weight` 守住“对抗项权重不能翻转损失方向”这一运行时边界 |
| 2026-06-09 14:20 | Phase 2 | #20 compute_structured_aux_losses 语义错位 | AI | `train.compute_structured_aux_losses` 入口新增显式末维契约断言(`queue/cycle/queue_rollout` 末维必须严格 6,否则 `AssertionError`);`models.compute_queue_targets` / `models.build_cycle_features` 各自 docstring 声明 6 维 dim→loss-type 映射,并新增 `build_queue_targets_signature` / `build_cycle_features_signature` 两个源码契约 helper(被测试引用,守卫 reorder);`compute_structured_aux_losses` docstring 显式声明 **main aux = 末帧监督** 与 **rollout aux = 全序列监督** 的有意 asymmetry,并说明设计理由;`queue_reg_idx` / `queue_cls_idx` / cycle 切分常量集中声明;#20 定向测试 11 项全过(末维契约、源码守卫、BCE/CE 鲁棒性、main/rollout 不对称),测试总数 132 → 143,全过;1-batch warmup smoke 验证训练流程未坏 |
| 2026-06-07 21:30 | Phase 0 | #3 rollout offset 训练/推理一致 | AI | `last_rollout_offset = input_t.squeeze(0) if teacher_force else output` 在 models.py:1799 改为 `last_rollout_offset = output`,与推理分支(models.py:1865)对齐;此修复消除 queue rollout 分支的 train/eval distribution shift(exposure bias):修复前训练时 80% 看到 GT future displacement、20% 看到模型预测,推理时 100% 看到模型预测;新增 2 个单元测试(训练分支 teacher_force 时必须用模型自身 output / 训练与推理 step0 seed 来自 obs_len-1),测试总数 44 → 46,全部通过;py_compile 干净;1-batch smoke 训练正常完成,loss 与稳定性指标数值合理(GradNorm ≈ 173.21, QRollHNorm ≈ 0.20, PredOffsetNorm ≈ 0.19) |
| 2026-06-08 01:10 | Phase 0 + Phase 5 | #7/#8/#9 状态同步 | AI | 基于 Stage 32 代码与 62 项单元测试结果,将 PLAN.md 中 #7 start_epoch 恢复、#8 lane anchor 向量化、#9 relation_Matrix 向量化 从"未修复/部分改善"同步为"已修复";同时更新审计结论(11/35 已修复)、Phase 0/Phase 5 进度与里程碑计数 |
| 2026-06-08 19:30 | Phase 5 | #10/#11/#12/#13/#14/#15 修复批量 | AI | Stage 33:六个代码清理项完成并经补强验收复核。(a) **#10** `_mean_norm_from_tensor` docstring 明确 0/1/≥2 维分情况返回语义;(b) **#11** `graph_lstm_model` 保留 `nn.LSTMCell` 实例(用于旧 checkpoint 兼容),加 `_graph_lstm_call_count` + `forward_hook` 作为真实调用诊断,补到 3 个测试(未调用/类型/直接调用计数);(c) **#12** `seq_start_end.data` 不仅在 train.py,也在 models.py active code 中全部替换为 `.tolist()`,补到 5 个测试(双文件源码守卫 + 双文件模式 + Python int 契约);(d) **#13** `main` 训练循环维护 `global_step` 跨 epoch 单调递增,且 `D_train` 与 generator `train` 的 tensorboard 标量统一写入 `global_step`,补到 5 个测试(D_train + train 双侧验证);(e) **#14** 新增 `BestAdeTracker` 类(`update`/`restore_from_checkpoint`/`value`),`main` 局部实例化并移除 `global best_ade` 声明入口,补到 7 个测试(含无 global 声明);(f) **#15** `utils.set_logger` 改用 `logger.hasHandlers()` 防重复挂载。Stage 33 对应测试从原 23 项补强到 29 项(7/3/5/5/7/2),用于把"文档式通过"收紧成真实行为验收;py_compile 校验通过;PLAN.md/README/EXPERIMENT_LOG 同步收口 |
| 2026-06-09 16:00 | Phase 4 | #21 NumValSamplesTracker 双状态模型强化 | AI | Stage 39:补强 #21 的两个弱点 — (1) `build_num_val_samples_signature()` 从裸 `True` 升级为返回结构化 `dict`(`checkpoint_key` / `runtime_arg` / `eval_arg` / `must_persist_positive_int` 四个字段),同 #20 模式可被测试断言;(2) `NumValSamplesTracker` 重构为**双状态模型**(`_runtime_num_val_samples` + `_checkpoint_num_val_samples`),修复"只报警、不修正 payload"的核心漏洞 — 旧实现下 `restore_from_checkpoint(20)` 会覆盖 runtime K=4,导致后续 `save_checkpoint` 写入 K=20 而非当前 args K=4;新实现下 `checkpoint_payload()` 永远返回 runtime K,`check_alignment` 仅使用 checkpoint K 做诊断比对,既报警又落盘正确。新增 2 个强行为回归测试 (`test_num_val_samples_tracker_restore_does_not_override_runtime_payload` / `test_num_val_samples_tracker_keeps_runtime_k_after_ckpt_mismatch`) 外加把签名测试收紧为 dict 断言;`python -m unittest tests.test_cyclestate_protocol` 155 项全过(153 → 155,新增 2 项 #21 双状态回归),py_compile 干净;PLAN.md 中 #21 行 / 6.1 节 / 审计时间戳同步更新到 Stage 39 |
| 2026-06-09 17:30 | Phase 4 | #22 disable_aux_losses 消融统一主开关 | AI | Stage 40:为 #22 消融实验公平性新增 `disable_aux_losses` 统一主开关。(a) `CycleStateTrajectoryGenerator.__init__` 新增 `disable_aux_losses=False` 参数,True 时强制 4 个独立 disable 标志位全 True;(b) `train.py` 新增 `--disable_aux_losses` CLI 参数并在 `main()` 中将三组 aux 权重置零;(c) `evaluate_model.py` 新增同名 CLI 参数并透传给模型构造。(d) 模型参数数量不变(架构兼容 checkpoint),仅运行时行为切换。新增 9 个定向测试覆盖标志位转发/forward/输出形状/debug 字段/参数数量/双 CLI 解析/源码守卫 aux 置零与 evaluate_model 透传/aux 头结构保留;`python -m unittest tests.test_cyclestate_protocol` 164 项全过(155 → 164),py_compile 干净;1-batch warmup smoke with/without flag 均正常完成,aux 损失在 flag=ON 时全零,flag=OFF 时非零且行为未退化 |

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
