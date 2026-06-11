# CycleState Experiment Delegation Guide

> 用途：把实验、协议复核或 correctness 修复交给执行型 AI 时，保持任务边界清晰、口径一致。

## 1. 开始前必须读

1. `/home/lbh/D2-TPred-CycleState/README.md`
2. `/home/lbh/D2-TPred-CycleState/EXPERIMENT_LOG.md`
3. `/home/lbh/D2-TPred-CycleState/docs/PLAN.md`
4. `/home/lbh/D2-TPred-CycleState/tests/test_cyclestate_protocol.py`

只有在需要改代码时，再继续读：

5. `/home/lbh/D2-TPred-CycleState/D2TP/models.py`
6. `/home/lbh/D2-TPred-CycleState/D2TP/train.py`
7. `/home/lbh/D2-TPred-CycleState/D2TP/evaluate_model.py`

## 2. 不得偏离的主线

1. `CycleState` 的核心是 `full-cycle traffic-state memory`。
2. 主结构保持 `micro / meso / macro` 三层。
3. 当前研究重点是“状态如何稳定进入预测期”。
4. 现阶段优先级是协议正确性与训练稳定性，不是继续加新结构。

## 3. 明确禁止事项

1. 引入通用 Transformer、Diffusion、Scene Encoder 大改。
2. 混用 `val` / `test` 或混用 `num_samples=4` / `20` 后直接宣称超过 baseline。
3. 用 `test` 做模型选择。
4. 在同一轮同时改多个大变量。
5. 把已证伪方向重新当主线：
   - 单独降低 warmup `teacher_forcing_ratio`
   - 在 warmup 内继续堆小 trick
6. 只给结论，不给命令、checkpoint、split、采样次数和原始指标。

## 4. 允许执行的任务

1. `baseline audit`
2. `candidate verification`
3. `correctness fix`
4. `controlled ablation`

## 5. 固定执行顺序

### Phase A: 必跑可比线

```bash
python D2TP/evaluate_model.py --model_type d2tpred --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --num_samples 20 --eval_print_every 10 --resume D2TP/model_best.pth.tar --dset_type val
python D2TP/evaluate_model.py --model_type d2tpred --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --num_samples 20 --eval_print_every 10 --resume D2TP/model_best.pth.tar --dset_type test
python D2TP/evaluate_model.py --model_type cyclestate --device cuda --pin_memory --loader_num_workers 0 --batch_size 8 --num_samples 20 --eval_print_every 10 --resume experiments/cyclestate/warmup50_refine50_p0_seqgat_relation_v1/checkpoint/model_best.pth.tar --dset_type test --rollout_residual_scale 0.7
```

### Phase B: 如果候选站得住

只做单变量消融，顺序固定：

1. `disable_queue_rollout`
2. `disable_decoder_state_residual`
3. `disable_lane_queue_anchor`
4. `disable_state_gating`

默认先做 `val + num_samples=20`。

### Phase C: 如果候选站不住

优先回修 correctness 与协议：

1. rollout 训练 / 推理一致性
2. 预测期 `phase_change` / cycle-state 信号路径
3. warmup 与 refine 的职责划分

## 6. 改代码后的最小验证

```bash
python -m unittest discover -s tests -p 'test_cyclestate_protocol.py'
python -m py_compile D2TP/train.py D2TP/models.py D2TP/evaluate_model.py
```

如果改了训练协议、日志或 checkpoint 路径，再补一个 1-batch smoke。

## 7. 结果回填模板

每次交付至少给出：

- `任务类型`
- `命令`
- `checkpoint / resume 路径`
- `split`
- `num_samples`
- `ADE / FDE`
- `标签`：`smoke` / `protocol-check` / `comparable`
- `一句话结论`

如果结果会影响当前主线，再同步更新：

- `README.md`
- `EXPERIMENT_LOG.md`
- `docs/PLAN.md`

## 8. 何时必须停下

- 缺 checkpoint / 缺依赖 / 命令无法运行
- 结果口径无法与现有 comparable 对齐
- 需要同时改多个大变量才有希望推进
- 想引入新结构才能继续

出现这些情况时，先回报阻塞，不要擅自改任务目标。
