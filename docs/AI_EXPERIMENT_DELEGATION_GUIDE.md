# CycleState Experiment Delegation Guide

> 用途：把 `CycleState` 的后续修改、实验、验证和文档回填交给执行型 AI；主审 AI 只负责判断它做得对不对。
> 适用日期：2026-06-07 之后，直到 README / EXPERIMENT_LOG / PLAN 再次改写主线。

## 1. 先记住研究主线

执行任何任务前，不得偏离下面四条主线：

1. `CycleState` 的核心问题是 `full-cycle traffic-state memory`，不是泛化的“再堆一个更强时序模块”。
2. 主结构保持 `micro / meso / macro` 三层：
   - `micro`: 原始 `D2-TPred` 运动与交互编码
   - `meso`: lane-level queue-state memory
   - `macro`: cycle-state memory
3. 当前最有辨识度的机制是 `Phase-Rolling Queue Memory`，研究重点是“状态如何稳定进入预测期”。
4. 现阶段优先级是 **协议正确性与训练稳定性**，不是继续加新结构。

## 2. 明确禁止事项

以下行为默认判定为跑偏，主审 AI 可以直接驳回：

1. 引入通用 Transformer、Diffusion、Scene Encoder 大改、额外 imitation/matching 分支。
2. 混用 `val` 和 `test`，或混用 `num_samples=4` 和 `20` 后直接宣称超过 baseline。
3. 用 `test` 做模型选择。
4. 在同一轮里同时改多个大变量，导致无法归因。
5. 把已经证伪的方向重新当作主线推进：
   - 单独降低 warmup `teacher_forcing_ratio`
   - 继续在 warmup 内堆小 trick，而不先验证 `50b warmup -> 50b refine`
6. 只给结论，不给命令、checkpoint、split、采样次数和原始指标。

## 3. 已有结论，先别重复试错

### 3.1 已完成且应视为既有事实

1. 协议硬化已落地：
   - 训练内验证默认走 `val`
   - `--lr` 生效
   - `--grad_clip`
   - `--rollout_residual_scale`
   - `--detach_rollout_state`
   - 状态稳定性日志
2. P0 修复已落地：
   - `seqGAT` 梯度恢复
   - `relation_Matrix` 正常方向扇区逻辑修复
3. Stage 26 已证伪：
   - 单独把 warmup `teacher_forcing_ratio` 从 `0.8` 降到 `0.6`，`100b` 更差
4. 当前最强候选不是 warmup-only，而是：
   - `50b warmup -> 50b refine`
   - run: `experiments/cyclestate/warmup50_refine50_p0_seqgat_relation_v1`

### 3.2 当前已知证据

- baseline comparable:
  - `val + num_samples=4`: `ADE 38.493 / FDE 78.706`
  - `test + num_samples=4`: `ADE 17.812 / FDE 37.568`
- current best CycleState evidence:
  - `warmup50_refine50_p0_seqgat_relation_v1`
  - `val + num_samples=20`: `ADE 75.078 / FDE 154.690`
  - `val + num_samples=4`: `ADE 84.772 / FDE 170.878`

结论边界：当前还不能宣称超过 baseline。

## 4. 允许执行的任务类型

执行型 AI 只应在以下四类任务里工作：

1. `baseline audit`
   - 跑官方 baseline 在不同 split / `num_samples` 下的正式可比线。
2. `candidate verification`
   - 复核当前最强候选在 `val/test` 上是否成立。
3. `correctness fix`
   - 修训练/推理协议不一致、状态特征缺失、文档假设缺漏这类 correctness 问题。
4. `controlled ablation`
   - 围绕当前 best candidate 做单变量消融。

## 5. 固定执行顺序

### Phase A: 必跑的正式可比线

先跑这三条，再谈后续优化：

1. baseline `val + num_samples=20`
2. baseline `test + num_samples=20`
3. current best candidate `test + num_samples=20`

命令固定如下。

#### A1. baseline `val + num_samples=20`

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

#### A2. baseline `test + num_samples=20`

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

#### A3. current best candidate `test + num_samples=20`

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

### Phase B: 如果 candidate `test@20` 站得住

只做消融，不加结构。

消融顺序固定为：

1. `disable_queue_rollout`
2. `disable_decoder_state_residual`
3. `disable_lane_queue_anchor`
4. `disable_state_gating`

消融默认流程：

1. 先 `val + num_samples=20`
2. 保留 split / `num_samples` / checkpoint 一致
3. 只有当结论确实影响主故事时，才追加 `test` 复核

参考命令模板：

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
  --rollout_residual_scale 0.7 \
  --disable_queue_rollout
```

其余三条消融只替换最后一个开关。

### Phase C: 如果 candidate `test@20` 站不住

不要回去继续试 `teacher_forcing_ratio` 单变量。按下面顺序回修 correctness：

1. `rollout offset` 训练 / 推理不一致
   - 文件：`D2TP/models.py`
   - 目标：训练期 rollout offset 的更新逻辑和推理期一致
2. 预测期 `phase_change` 缺失
   - 文件：`D2TP/models.py`
   - 目标：`get_step_cycle_feature` 不再把 `phase_change` 恒置零
3. `pred_state` 的 oracle 假设补文档
   - 文件：`README.md` 及相关研究文档
   - 目标：对外叙事明确“当前方法默认已知未来信号状态/或等价 oracle”
4. `D_step` 跨 epoch 重置
   - 文件：`D2TP/train.py`
   - 只在重新推进 adversarial 训练时优先处理；当前不是第一优先级

## 6. 修改代码时的固定流程

### 6.1 先读这些文件

1. `/home/lbh/D2-TPred-CycleState/README.md`
2. `/home/lbh/D2-TPred-CycleState/EXPERIMENT_LOG.md`
3. `/home/lbh/D2-TPred-CycleState/docs/PLAN.md`
4. `/home/lbh/D2-TPred-CycleState/tests/test_cyclestate_protocol.py`
5. 如需改代码，再读：
   - `/home/lbh/D2-TPred-CycleState/D2TP/train.py`
   - `/home/lbh/D2-TPred-CycleState/D2TP/models.py`

### 6.2 必须补的验证

任何代码改动后，先做这两条：

```bash
python -m unittest discover -s tests -p 'test_cyclestate_protocol.py'
python -m py_compile D2TP/train.py D2TP/models.py D2TP/evaluate_model.py
```

如果改了训练协议、日志或 checkpoint 路径，再补一个 1-batch smoke：

```bash
python D2TP/train.py \
  --log_dir experiments/cyclestate/smoke_delegate_check \
  --model_type cyclestate \
  --train_stage warmup \
  --device cuda \
  --pin_memory \
  --loader_num_workers 0 \
  --batch_size 8 \
  --best_k 2 \
  --num_val_samples 2 \
  --resume D2TP/model_best.pth.tar \
  --num_epochs 0 \
  --print_every 1 \
  --max_train_batches 1 \
  --max_val_batches 1 \
  --val_dset_type val
```

### 6.3 测试写法要求

新测试必须直接保护你改的协议，不要只测“能跑通”。

优先补这类测试：

1. rollout offset train / infer 一致性
2. `phase_change` 在预测期非恒零
3. 某个 flag 为 0 或关闭时，路径退化为旧行为
4. detach 类开关截断了梯度，但不截断前向输出

## 7. 结果标签和判断口径

所有结果必须打标签：

1. `smoke`
   - 只验证能不能跑
2. `protocol-check`
   - 验证协议是否一致
3. `comparable`
   - split、采样数、checkpoint 来源、评估脚本全部对齐后才能用来比 baseline

额外规则：

1. 不写 split 的结果一律视为不可比。
2. 不写 `num_samples` 的结果一律视为不可比。
3. `val + 4` 可以快速筛选，但不能写成正式超越结论。
4. `test` 只做最终复核，不能拿来挑模型。

## 8. 当前统一的验收标准

### 8.1 训练稳定性

对同一配置：

- `100-batch val ADE` 不得比 `50-batch val ADE` 恶化超过 `15%`

### 8.2 rollout 价值

- `rollout on` 必须优于匹配的 `no_rollout`

### 8.3 可比结论

只有同时满足下面三条，才允许写“优于 baseline”：

1. split 对齐
2. `num_samples` 对齐
3. checkpoint 来源和评估脚本一致

## 9. 执行型 AI 的交付格式

每次交付必须包含下面这些字段，缺一项就不能进入主审：

```md
## Run Summary
- Task type:
- Tag: smoke / protocol-check / comparable
- Run name:
- Commit:
- Files changed:
- Resume checkpoint:
- Split:
- num_samples:
- max_train_batches:
- max_val_batches:
- Key flags:
- Commands:
- Best ADE:
- Best FDE:
- Conclusion:
- Recommended next step:
```

若改了代码，还要补：

```md
## Validation
- Unit tests:
- Compile check:
- Smoke run:
- New regression test name:
```

## 10. 什么时候必须停下来交给主审 AI

出现以下任一情况，不要继续自由发挥，直接上报：

1. 你想加新的大模块或重写主体结构。
2. 你发现当前结果只能在 `test` 上好、在 `val` 上不好。
3. 你需要同时改 2 个以上核心变量才能让结果变好。
4. baseline `num_samples=20` 的正式结果和当前认知差异很大。
5. 当前 best candidate `test@20` 失败，并且 correctness 修复后仍无改善。
6. 你准备写“超过 baseline / 接近论文结果”这类结论。

## 11. 主审 AI 的拒收清单

如果执行型 AI 的交付满足任一条，主审 AI 可以直接判定为不合格：

1. 没有给原始命令。
2. 没写 split 或 `num_samples`。
3. 混淆 `val` 与 `test`。
4. 把 `protocol-check` 结果写成 `comparable`。
5. 改了主结构却没有明确理由。
6. 继续把 warmup `teacher_forcing_ratio` 单变量当作主优化线。
7. 没补测试，只给“看起来能跑”的口头结论。
8. 文档没有同步更新。

## 12. 推荐的最小工作习惯

如果执行环境支持技能或工作流工具，建议顺序如下：

1. 先用规划类技能梳理任务边界
2. 出现异常时用系统化调试流程
3. 准备宣称完成前做一次 verification

如果没有这些工具，也至少做到：

1. 一次只改一个主要变量
2. 每次先验证再宣称
3. 每次都把结论写回文档

## 13. 文档同步要求

只要出现下面任一变化，就必须同步文档：

1. 下一轮实验顺序改变
2. 当前 best candidate 改变
3. baseline 可比线补齐
4. 纠正了一个会影响科研叙事的 correctness 问题

最少需要同步：

1. `/home/lbh/D2-TPred-CycleState/README.md`
2. `/home/lbh/D2-TPred-CycleState/EXPERIMENT_LOG.md`
3. `/home/lbh/D2-TPred-CycleState/docs/PLAN.md`

这三份文档里，README 保留摘要，完整时间线优先写进 `EXPERIMENT_LOG.md`。
