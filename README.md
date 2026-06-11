# D2-TPred-CycleState

本仓库承担两个角色：

1. 审计原始 `D2-TPred` baseline。
2. 迭代 `CycleState: Full-Cycle Traffic-State Memory for Trajectory Prediction at Signalized Intersections`。

当前主线不是给 baseline 叠一个泛化时序模块，而是把信号灯路口轨迹预测表述成
`full-cycle traffic-state memory` 问题：未来轨迹同时受微观运动、中观排队/释放波、
宏观信号周期状态约束。当前最有辨识度的机制是 `Phase-Rolling Queue Memory`，最新
实现补上了预测期 `cycle hidden/cell` rollout，使 macro memory 不再只是一帧一帧
重算的静态条件。

## Oracle 假设

CycleState 在训练和推理中都使用未来真实信号状态 `pred_state`。这是一条明确的
oracle 假设：真实部署时，这部分信息必须来自外部信号控制器或预测模块，而不是数据集。
原始 `D2-TPred` baseline 同样使用 `pred_state`，因此当前仓库中的 comparable 结果在
这个假设上是对齐的。后续退化实验见 [docs/PLAN.md](./docs/PLAN.md)。

## 当前状态

### 研究与工程结论

- 当前优先级是协议正确性与训练稳定性，不是继续加新结构。
- 已完成的关键协议硬化包括：`val` 选模、`lr` 生效、`grad_clip`、
  `rollout_residual_scale`、`detach_rollout_state`、`AblationConfig` 集中管理。
- G6 的最小方案已经落地：预测期 decoder 现在消费 rollout 后的 cycle hidden，
  而不是只消费 `cycle_step_embedding`。
- 当前最强候选已经更新为“`50b warmup -> 50b refine` old best + prediction-time
  cycle rollout 后的最小增量 refine”，但还**不能**宣称超过 baseline。

### 当前可比证据

| 模型 | Split | num_samples | ADE | FDE | 说明 |
|------|-------|-------------|-----|-----|------|
| baseline (`D2TP/model_best.pth.tar`) | val | 20 | 35.022 | 70.658 | 新补齐的 comparable 口径 |
| baseline (`D2TP/model_best.pth.tar`) | test | 20 | 15.359 | 31.514 | 新补齐的 comparable 口径 |
| baseline (`D2TP/model_best.pth.tar`) | val | 4 | 38.493 | 78.706 | 当前仓库 quick 基线 |
| baseline (`D2TP/model_best.pth.tar`) | test | 4 | 17.812 | 37.568 | 当前仓库 quick 基线 |
| CycleState `warmup50_refine50_p0_seqgat_relation_v1` | val | 20 | 74.947 | 154.411 | 接入 prediction-time cycle rollout 后复核 |
| CycleState `warmup50_refine50_p0_seqgat_relation_v1` | test | 20 | 43.736 | 85.691 | 接入 prediction-time cycle rollout 后复核 |
| CycleState `warmup50_refine50_p0_seqgat_relation_v1_cycle_rollout_refine1` | val | 20 | 51.607 | 103.955 | 旧 best 上做最小增量 refine 后的 full-val 复核 |
| CycleState `warmup50_refine50_p0_seqgat_relation_v1_cycle_rollout_refine1` | test | 20 | 34.911 | 69.133 | 旧 best 上做最小增量 refine 后的 full-test 复核 |
| CycleState `warmup50_refine50_p0_seqgat_relation_v1` | val | 4 | 84.772 | 170.878 | 旧 quick 口径 |

### 当前判断

- `val` / `test` 差距明显，所有“超过 baseline”的论断都必须显式带上 split。
- 单独把 warmup `teacher_forcing_ratio` 从 `0.8` 降到 `0.6` 已被证伪。
- 这轮优化把研究故事进一步收口为：`micro trajectory`、`meso queue rollout`、
  `macro cycle rollout` 三条时间尺度一致推进，而不是再堆一个通用时序模块。
- `prediction-time cycle rollout` 直接接到旧 checkpoint 后，`val + 20` 只从
  `75.078 / 154.690` 小幅改善到 `74.947 / 154.411`；单靠推理侧接线不够。
- 但在旧 best 上做 1 轮短程 refine 后，`full val + 20` 提升到
  `51.607 / 103.955`，`test + 20` 也提升到 `34.911 / 69.133`，说明这条
  `macro rollout` 路径不是伪信号，而是需要参数重新适配才能生效。
- 这轮最小训练把 `test + 20` 从 `43.736 / 85.691` 拉到 `34.911 / 69.133`，
  但仍明显落后 baseline `15.359 / 31.514`。
- 更保守的 low-lr continuation 已经被证伪：从 `..._cycle_rollout_refine1`
  继续短程 refine 的 `20-batch val` 快照依次为 `77.511 / 154.035`、
  `69.903 / 138.006`、`83.454 / 165.355`，没有刷新当前 best。
- 四个单变量消融在 `val + 20` 上几乎不动，说明当前 checkpoint 不是“开关没接上”，
  而是这些状态分支虽然会改变前向输出，但和最终 ADE/FDE 的数值耦合仍然偏弱。
- 额外诊断显示 decoder state residual 的步级范数约为 queue hidden 范数的
  `1.3%`，因此当前重点已经从“继续盲试 continuation”转向“如何把 meso/macro
  memory 更强地耦合进 decoder”。
- 新增 `decoder_state_residual_scale` 这个独立旋钮后，离线扫描表明：
  - `val + num_samples=1` 可从 `59.349 / 121.138` 提到 `59.205 / 120.487`
    (`scale=4.0`)
  - `val + num_samples=20` 可从 `51.607 / 103.955` 提到 `51.383 / 103.133`
    (`scale=4.0`)
  - 但 `test + num_samples=20` 会从 `34.911 / 69.133` 小幅回退到
    `35.147 / 69.575`
- 这说明“state residual 太弱”这个判断是对的，但单纯在推理时放大它还不够稳，
  需要训练阶段一起重新适配。

## 代码入口

- `D2TP/models.py`：`TrajectoryGenerator`、`CycleStateTrajectoryGenerator`、
  `RolloutQueueCoefs`、`AblationConfig`
- `D2TP/train.py`：分阶段训练协议、aux losses、训练内验证、checkpoint
- `D2TP/evaluate_model.py`：离线 `val/test` 评估
- `tests/test_cyclestate_protocol.py`：协议与回归测试

## 文档地图

| 文档 | 作用 |
|------|------|
| `README.md` | 项目是什么、现在在哪、最小复现入口 |
| `EXPERIMENT_LOG.md` | 当前证据、关键里程碑、推荐下一步 |
| `docs/PLAN.md` | 活跃 backlog、执行顺序、验收门槛 |
| `docs/technical_documentation.md` | 架构、数据流、修改入口、实现约束 |
| `docs/AI_EXPERIMENT_DELEGATION_GUIDE.md` | 把实验或修复委托给执行型 AI 的简版规则 |
| `docs/ENGINEERING_ISSUES.md` | 工程问题索引落点 |
| `docs/COMPREHENSIVE_ANALYSIS.md` | 综合问题索引落点 |
| `docs/METHOD_AND_ARCHITECTURE_ANALYSIS.md` | 方法/结构问题索引落点 |

## 实验标签

- `smoke`：只验证 forward/backward、日志、checkpoint、开关和损失项是否工作
- `protocol-check`：验证训练协议、恢复逻辑、采样口径是否一致
- `comparable`：split、checkpoint、采样次数、评估脚本口径全部对齐后，才可用于正式比较

## 最小复现入口

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

把 `--dset_type val` 改成 `test` 可做最终复核；把 `--num_samples 4` 改成 `20`
可做更接近论文口径的审计。

### 当前最佳候选复核

```bash
python D2TP/evaluate_model.py \
  --model_type cyclestate \
  --device cuda \
  --pin_memory \
  --loader_num_workers 0 \
  --batch_size 8 \
  --num_samples 20 \
  --eval_print_every 10 \
  --resume experiments/cyclestate/warmup50_refine50_p0_seqgat_relation_v1_cycle_rollout_refine1/checkpoint/model_best.pth.tar \
  --dset_type test \
  --rollout_residual_scale 0.7
```

## 下一步

1. 保留新的 `decoder_state_residual_scale` 旋钮，并把下一轮短程 refine 建立在
   `decoder_state_residual_scale > 1.0` 的协议上，而不是只在推理时放大。
2. 新训练仍先看 `val + num_samples=20` 选模，再补 `test + 20`；重点验证
   “训练适配后，增强的 state residual 是否能把 `val` 的微弱收益传到 `test`”。
3. 若这条线仍只改善 `val` 而不改善 `test`，就回到 oracle 依赖量化与更深层的
   状态使用问题，而不是继续扫推理尺度。
