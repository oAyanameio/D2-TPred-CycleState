# CycleState 实验日志

> **精简时间**: 2026-06-07 18:30
> **覆盖范围**: 全部 protocol-check 与 comparable 实验
> **历史详情**: 早期 Stage 1-23 的中间过程已沉淀到 [PLAN.md](./PLAN.md) 与 [docs/technical_documentation.md](./docs/technical_documentation.md)，本日志只保留**当前结论、最强证据、推荐下一步**

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

**Stage 25 关键根因**（已修，参见 [PLAN.md §11.2](./PLAN.md#112-根因分析基于-100b-实验日志)）:
- **Exposure Bias**: warmup `teacher_forcing_ratio=0.8` 太高
- **Rollout LSTM 缺归一化**: 12 步递推无 LayerNorm，QRollHNorm 涨 7.4×
- **残差注入强度失控**: `rollout_residual_scale=0.35` + 无幅值压缩

**Stage 26 关键发现**:
- 单独降低 warmup TF（0.8→0.6）让 100b 进一步恶化到 187.6%
- 改走 `50b warmup → refine` 协议才是当前最优路径
- warmup 不宜承担过长的长链状态耦合训练

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
