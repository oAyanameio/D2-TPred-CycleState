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
3. 当前研究重点已经转移到”决定性实验”阶段：在最简可行版本 (DE-3)、Oracle State 直注 (DE-1)、极端耦合压力测试 (DE-2) 完成之前，不做任何新结构或训练。
4. 现阶段优先级是完成决定性实验，验证”交通状态 → 轨迹预测”因果链本身是否成立，以及当前架构瓶颈到底在耦合方式还是在信息本身。

## 3. 明确禁止事项

1. 引入通用 Transformer、Diffusion、Scene Encoder 大改。
2. 混用 `val` / `test` 或混用 `num_samples=4` / `20` 后直接宣称超过 baseline。
3. 用 `test` 做模型选择。
4. 在同一轮同时改多个大变量。
5. 把已证伪方向重新当主线：
   - 单独降低 warmup `teacher_forcing_ratio`
   - 在 warmup 内继续堆小 trick
6. 在决定性实验（见 `docs/PLAN.md` 第 4 节）给出明确结论之前：
   - 继续做 continuation/refine 超参扫描
   - 继续做推理侧 scale 扫描
   - 新增任何 state 分支机制
   - 在未改变耦合方式的前提下堆结构
7. 只给结论，不给命令、checkpoint、split、采样次数和原始指标。

## 4. 允许执行的任务

1. `baseline audit`
2. `candidate verification`
3. `correctness fix`
4. `controlled ablation`

## 5. 固定执行顺序

### Phase A: 必跑可比线

```bash
# baseline val + 20
python D2TP/evaluate_model.py --model_type d2tpred --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --num_samples 20 --eval_print_every 10 --resume D2TP/model_best.pth.tar --dset_type val
# baseline test + 20
python D2TP/evaluate_model.py --model_type d2tpred --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --num_samples 20 --eval_print_every 10 --resume D2TP/model_best.pth.tar --dset_type test
# old best cycle rollout 接旧 checkpoint
python D2TP/evaluate_model.py --model_type cyclestate --device cuda --pin_memory --loader_num_workers 0 --batch_size 8 --num_samples 20 --eval_print_every 10 --resume experiments/cyclestate/warmup50_refine50_p0_seqgat_relation_v1/checkpoint/model_best.pth.tar --dset_type test --rollout_residual_scale 0.7
# cycle_rollout_refine1 增量 refine 后最佳
python D2TP/evaluate_model.py --model_type cyclestate --device cuda --pin_memory --loader_num_workers 0 --batch_size 8 --num_samples 20 --eval_print_every 10 --resume experiments/cyclestate/warmup50_refine50_p0_seqgat_relation_v1_cycle_rollout_refine1/checkpoint/model_best.pth.tar --dset_type test --rollout_residual_scale 0.7
```

### Phase B: 如果候选站得住

只做单变量消融，顺序固定：

1. `disable_queue_rollout`
2. `disable_decoder_state_residual`
3. `disable_state_gating`
4. `disable_lane_queue_anchor`

（注意：该顺序已根据 `docs/PLAN.md` G7 更新——`state_gating` 和 `lane_queue_anchor` 的优先级互换。）

默认先做 `val + num_samples=20`。

### Phase C: 根据决定性实验结果分支

当前所有实验推进前，必须先完成 `docs/PLAN.md` 第 4 节中的三个决定性实验：

1. **DE-3（最先）**: 最简可行版本（最轻量，最快判断"直接拼接 state 进 decoder 初始化"是否比"加性残差"有效）
2. **DE-1**: Oracle State 直注实验（判断"交通状态 → 轨迹预测"因果链本身强度）
3. **DE-2**: 极端耦合压力测试（判断耦合强度是否是主瓶颈）

根据决定性实验结果，走不同分支：

- **全部正向** → 推进架构重设计 AR-1（直接条件注入）或 AR-2（乘法门控），在新架构上重新训练
- **DE-1 显示 oracle state 边际贡献有限** → 重新审视 CycleState 核心假设，回到问题定义层面
- **DE-1 正向但 DE-2 负向** → 问题不在耦合强度，重点转向 AR-3 (aux loss 重新设计)

详见 `docs/PLAN.md` 第 4-5 节。

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
- （若涉及 CycleState）`teacher_forcing_ratio`、`rollout_residual_scale`、`decoder_state_residual_scale`、`aux_rollout_weight`、`detach_rollout_state`

如果结果会影响当前主线，再同步更新：

- `README.md`
- `EXPERIMENT_LOG.md`
- `docs/PLAN.md`

## 8. 何时必须停下

- 缺 checkpoint / 缺依赖 / 命令无法运行
- 结果口径无法与现有 comparable 对齐
- 需要同时改多个大变量才有希望推进
- 想引入新结构才能继续
- 在决定性实验（DE-1/DE-2/DE-3）完成之前，试图推进新训练或架构改动
- 改动不涉及 state-to-decoder 耦合路径的实质变化（默认预期：对指标无影响，见 PLAN.md 2.2 节消融证据）

出现这些情况时，先回报阻塞，不要擅自改任务目标。
