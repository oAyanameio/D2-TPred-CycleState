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

### Phase 0.5: Oracle 假设验证

**问题**

CycleState 与 baseline 都使用未来真实信号 `pred_state`，当前 comparable 是对齐的，但这个 oracle 假设的性能贡献还没被量化。

**关注文件**

- `README.md`
- `docs/PLAN.md`
- 未来若做实验，主要落点在 `D2TP/models.py`、`D2TP/evaluate_model.py`

**完成标志**

1. 明确 `train oracle -> eval oracle` 上界。
2. 评估 `train oracle -> eval predicted` 的退化幅度。
3. 如果需要，再补信号噪声敏感度曲线。

### G3: warmup 长程稳定性仍未根治

**现状**

- Stage 24: `100b` 相对 `50b` 恶化 `165.7%`
- Stage 25: 缓和到 `130.1%`
- Stage 26: 单独降 `teacher_forcing_ratio` 到 `0.6` 更差

**判断**

问题更像训练阶段职责划分错误，而不是单一超参能修好。

**完成标志**

- `100b ADE <= 50b ADE * 1.15`
- 或者给出证据说明 warmup 不应承担长链状态耦合学习，并正式改成更短 warmup + 更早 refine

### G6: cycle memory 在预测期退化

**问题**

预测期目前主要依赖单步 `cycle feature` 嵌入，缺少真正的 cycle-state 滚动更新。

**关注文件**

- `D2TP/models.py`
- `tests/test_cyclestate_protocol.py`

**完成标志**

- 预测期 cycle 状态不再只是静态条件。
- 能通过定向测试证明 cycle-state 在预测期按时间推进。

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

当前只有 4 个二元开关，尚未形成足够清晰的模块贡献矩阵。

**前提**

- 先确认当前最佳候选在 `test + num_samples=20` 站得住。

**完成标志**

- 至少完成一轮围绕当前最佳候选的单变量消融。
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

### Phase B: 如果候选站得住

按下面顺序做单变量消融：

1. `disable_queue_rollout`
2. `disable_decoder_state_residual`
3. `disable_lane_queue_anchor`
4. `disable_state_gating`

默认先跑 `val + num_samples=20`，只有主结论需要时再补 `test`。

### Phase C: 如果候选站不住

优先回到协议与状态演化问题，不加新结构：

1. 复核 rollout 训练/推理一致性
2. 复核预测期 `phase_change` / cycle-state 信号是否充分进入预测期
3. 判断 warmup 是否应更短，或更早交给 refine

## 5. Comparable 门槛

正式比较必须同时满足：

1. split 明确
2. `num_samples` 明确
3. checkpoint 来源明确
4. `evaluate_model.py` 口径一致
5. 不用 `test` 做模型选择

额外稳定性门槛：

- 若继续讨论 warmup 长程训练，`100b` 相对 `50b` 的 ADE 恶化不得超过 `15%`

## 6. 默认取舍

- 当前默认优先保住 `micro / meso / macro` 主线，不引入通用 Transformer / Diffusion / Scene Encoder 大改。
- 当前默认继续把 `README.md` 当项目入口，把 `EXPERIMENT_LOG.md` 当证据入口，把本文件当活跃待办入口。
- 任何影响研究叙事、协议口径或当前结论的改动，都应同步更新：
  - `README.md`
  - `EXPERIMENT_LOG.md`
  - `docs/PLAN.md`

## 7. 风险提醒

- oracle 假设若不量化，论文叙事会有明显缺口。
- 若在 G3 未收敛前推进 G4/G5，容易再次得到“结构改了但无法归因”的结果。
- 若 baseline `num_samples=20` 不补齐，后续任何“更接近论文口径”的比较都会站不稳。
