# CycleState Live Plan

> **更新时间**: 2026-06-10 12:00
> **作用**: 只维护当前活跃 backlog、执行顺序和验收门槛。
> **规则**: 已解决项不再在本文件展开维护；历史修复细节以 git 历史、测试与当前代码为准。

## 1. 当前目标

把项目从“当前最佳候选仍明显落后 baseline”推进到“协议正确、口径清楚、可复现地评估下一步是否还有方法价值”。

当前默认原则：

- 先修协议与状态进入预测期的稳定性，再谈结构增强。
- 不混用 split、不混用 `num_samples`、不用 `test` 做模型选择。
- 所有正式数字都按 `comparable` 口径记录。

## 2. 当前证据快照

| 项目 | Split | num_samples | ADE | FDE | 说明 |
|------|-------|-------------|-----|-----|------|
| baseline | val | 4 | 38.493 | 78.706 | 当前仓库基线 |
| baseline | test | 4 | 17.812 | 37.568 | 当前仓库基线 |
| CycleState best candidate | val | 20 | 75.078 | 154.690 | `warmup50_refine50_p0_seqgat_relation_v1` |
| CycleState best candidate | val | 4 | 84.772 | 170.878 | quick 口径 |

当前判断：

- `50b warmup -> 50b refine` 优于纯 warmup 候选，但仍没有追平 baseline。
- baseline 的 `num_samples=20` 审计尚未补齐。
- 当前最值得确认的不是新结构，而是：这条候选在 `test + num_samples=20` 上是否成立。

## 3. 活跃 backlog

### Phase 0.5: Oracle 假设量化验证

**问题**

CycleState 与 baseline 都使用未来真实信号 `pred_state`，当前 comparable 在这个假设上是对齐的，但其性能贡献还没有被量化。更重要的是，`pred_state` 不只进入 cycle 路径，也进入 `light_state_embedding`、`get_next_state` 和 queue rollout 路径；如果不先拆清楚这部分贡献，后续对 G3 / G6 / G4 / G5 的判断容易跑偏。

**关注文件**

- `README.md`
- `docs/PLAN.md`
- `D2TP/models.py`
- `D2TP/evaluate_model.py`

**执行原则**

先做**不引入新模块**的量化，再决定是否需要真正实现 predicted-signal / predicted-cycle 路径。`predicted` 不是一个现成开关，不能把“新增一个 predicted-cycle evaluator”误写成低成本动作。

**执行步骤**

1. 明确当前 `train oracle -> eval oracle` 上界，作为后续所有退化实验的参考点。
2. 做不引入新模块的敏感度实验，优先量化：
   - `cycle_step_embedding` 注入强度
   - `decode_cycle_gate` / cycle-conditioned path 的影响
   - 必要时再做轻量噪声注入，观察 `pred_state` 扰动对 ADE/FDE 的敏感度
3. 只有在上述证据显示 cycle-path oracle 贡献足够大时，才立项实现真正的 `predicted signal / predicted cycle` 评估路径。

**完成标志**

1. 明确 `oracle -> oracle` 的参考结果。
2. 对 cycle-path oracle 依赖给出可比较的定量证据，而不是只停留在文档提醒。
3. 再决定是否需要真实的 `train oracle -> eval predicted` 实验，而不是默认把它当成 Phase 0.5 的第一步。

### G6: cycle memory 在预测期退化

**问题**

预测期当前主要依赖单步 `cycle feature -> cycle_step_embedding` 路径，缺少与 queue rollout 对称的 cycle-state rollout。观测期的 `cycle_lstm_model` 会累计 `cycle_last`，但解码期每步消费的主要是单帧 `cycle_step_embedding`，不是 rollout 的 cycle hidden。

**关注文件**

- `D2TP/models.py`
- `tests/test_cyclestate_protocol.py`

**执行原则**

先做最小可验证改动，确认“有没有预测期 cycle memory”本身是否重要；不要一上来就把完整的显式 cycle dynamics 和 predicted-signal 路径同时做掉。

**建议执行顺序**

1. **最小方案**：在预测期 decoder loop 内引入 cycle hidden/cell 的逐步更新，保留每步 `current_cycle_feature` 输入，但让 decoder 读到 rollout 后的 cycle hidden，而不是只读静态单帧投影。
2. **验证方案**：新增定向测试，证明 cycle hidden 在预测期跨步变化、且确实参与 decoder 条件输入。
3. **完整方案**：只有在最小方案显示有正向信号时，才考虑把 cycle rollout 从“hidden-state rollout”升级为“显式 feature rollout”。

**完成标志**

1. 预测期 cycle 状态不再只是单帧静态条件。
2. 能通过定向测试证明 cycle-state 在预测期按时间推进。
3. 能说明这一改动是否实质影响当前 best candidate 的评估结果。

### G3: warmup 长程稳定性仍未根治

**现状**

- Stage 24: `100b` 相对 `50b` 恶化 `165.7%`
- Stage 25: 缓和到 `130.1%`
- Stage 26: 单独降 `teacher_forcing_ratio` 到 `0.6` 更差

**判断**

问题更像模块耦合与阶段职责划分错误，而不是单一超参能修好。G3 不应继续写成“先调协议”，而应先写成“先定位主因，再决定是否需要协议改动”。

**诊断顺序**

1. 在当前 best candidate 基础上，优先做**诊断式消融**：
   - `disable_queue_rollout`
   - `disable_decoder_state_residual`
   - `disable_state_gating`
   - `disable_lane_queue_anchor`
2. 如果某个模块一关掉就明显缓解长程退化，优先回到该模块路径定位，而不是继续扫 warmup 超参。
3. 如果消融仍无法定位，再讨论：
   - warmup 是否应更短
   - refine 是否应更早接管
   - `rollout_residual_scale` 等稳定化参数是否需要小范围 sweep

**完成标志**

1. 至少定位出一个最可能的退化主因，而不是继续盲调。
2. 保留 `100b ADE <= 50b ADE * 1.15` 作为稳定性 KPI，但不把它当成 G3 唯一出口。
3. 如果证据显示 warmup 本身不应承担长链状态耦合学习，则明确收口为更短 warmup + 更早 refine。

### G4: meso / macro 分支容量偏小

**问题**

`queue LSTM hidden=32`、`cycle LSTM hidden=16` 仍偏保守，可能限制状态表征能力。

**前提**

- 只在 G3、G6 至少有一项得到更清楚证据后再做。

**完成标志**

- 结构增容后仍保持协议可解释。
- 变更能通过单变量消融与 comparable 结果说明价值。

### G5: decoder residual 注入位置单一

**问题**

当前状态残差主要注入 `pred_lstm_hidden`，还没验证是否应扩展到 cell 或输出侧。

**前提**

- 不在协议还不稳时推进。

**完成标志**

- 有明确的注入设计与对应消融。
- 新设计不会破坏 baseline-compatible decoder 的 warm-start 逻辑。

### G7: 消融设计仍不够分离

**问题**

当前已有 4 个二元开关，但使用方式更像“候选通过后再证明模块价值”，还没有把它们当成诊断工具。

**执行原则**

G7 要前置成诊断工具，而不是只在“结果好看之后”才做的论文式补充。

**推荐顺序**

1. `disable_queue_rollout`
2. `disable_decoder_state_residual`
3. `disable_state_gating`
4. `disable_lane_queue_anchor`

**完成标志**

- 至少完成一轮围绕当前最佳候选的单变量消融，用于定位“哪个模块在拖后腿/哪个模块在提供稳定信号”。
- 如果主故事需要，再扩展成更系统的 2x2x2 设计。

### G8: 实验推进纪律

**问题**

仓库已经有 `smoke / protocol-check / comparable` 三级标签，但执行上仍容易过早推进新结构。

**完成标志**

- 每条新结果都带 split、`num_samples`、checkpoint、标签。
- 任何“超过 baseline”的说法都有完整 comparable 证据。

## 4. 固定执行顺序

### Phase A: 必跑可比线

1. baseline `val + num_samples=20`
2. baseline `test + num_samples=20`
3. current best candidate `test + num_samples=20`

建议命令：

```bash
python D2TP/evaluate_model.py --model_type d2tpred --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --num_samples 20 --eval_print_every 10 --resume D2TP/model_best.pth.tar --dset_type val
python D2TP/evaluate_model.py --model_type d2tpred --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --num_samples 20 --eval_print_every 10 --resume D2TP/model_best.pth.tar --dset_type test
python D2TP/evaluate_model.py --model_type cyclestate --device cuda --pin_memory --loader_num_workers 0 --batch_size 8 --num_samples 20 --eval_print_every 10 --resume experiments/cyclestate/warmup50_refine50_p0_seqgat_relation_v1/checkpoint/model_best.pth.tar --dset_type test --rollout_residual_scale 0.7
```

### Phase B: 诊断式消融

无论候选是否优于 baseline，只要 Phase A 的口径补齐，就可以开始最小离线消融。目标优先是定位问题，而不是写结果故事。

按下面顺序做单变量消融：

1. `disable_queue_rollout`
2. `disable_decoder_state_residual`
3. `disable_state_gating`
4. `disable_lane_queue_anchor`

默认先跑**离线 eval**：

- 优先 `val + num_samples=20`
- 只在出现明确信号时再补 `test`
- 默认先基于现有 checkpoint 做 eval 消融，不把“1 epoch 短训的 loss 趋势”当成主要证据

### Phase C: 如果候选站得住

在 Phase B 已完成的前提下，复用诊断式消融结果，把重点从“定位问题”升级成“确认主故事中的有效模块”。只有主结论需要时，才补更贵的 `test` 复核，而不是把同一套离线消融完整重跑一遍。

优先补强这些方向：

1. `disable_queue_rollout`
2. `disable_decoder_state_residual`
3. `disable_state_gating`
4. `disable_lane_queue_anchor`

此时消融目标从“定位问题”升级成“确认主故事中的有效模块”，并按需要追加 `test` 复核。

### Phase D: 如果候选站不住

优先回到协议与状态演化问题，不加新结构：

1. 先看 G7 诊断式消融有没有指出退化来源
2. 优先回到 G6，确认预测期 cycle-state 是否只是单帧条件而非 rollout memory
3. 再回到 G3，判断 warmup 是否应更短，或更早交给 refine

## 5. Comparable 门槛

正式比较必须同时满足：

1. split 明确
2. `num_samples` 明确
3. checkpoint 来源明确
4. `evaluate_model.py` 口径一致
5. 不用 `test` 做模型选择

额外稳定性门槛：

- 若继续讨论 warmup 长程训练，`100b` 相对 `50b` 的 ADE 恶化不得超过 `15%`
- 但该指标是稳定性 KPI，不代替“先定位主因”的诊断工作

## 6. 默认取舍

- 当前默认优先保住 `micro / meso / macro` 主线，不引入通用 Transformer / Diffusion / Scene Encoder 大改。
- 当前默认继续把 `README.md` 当项目入口，把 `EXPERIMENT_LOG.md` 当证据入口，把本文件当活跃待办入口。
- 任何影响研究叙事、协议口径或当前结论的改动，都应同步更新：
  - `README.md`
  - `EXPERIMENT_LOG.md`
  - `docs/PLAN.md`

## 7. 风险提醒

- oracle 假设若不量化，论文叙事会有明显缺口；但如果把 `predicted cycle` 误当成低成本评估开关，也会把 Phase 0.5 做成一个隐形大功能。
- 若在 G6 / G7 未给出更清楚证据前推进 G4/G5，容易再次得到“结构改了但无法归因”的结果。
- 若 baseline `num_samples=20` 不补齐，后续任何“更接近论文口径”的比较都会站不稳。
