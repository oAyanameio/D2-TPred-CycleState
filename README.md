# D2-TPred-CycleState

本仓库承担两个角色：

1. 审计原始 `D2-TPred` baseline。
2. 迭代 `CycleState: Full-Cycle Traffic-State Memory for Trajectory Prediction at Signalized Intersections`。

当前主线不是给 baseline 叠一个泛化时序模块，而是把信号灯路口轨迹预测表述成
`full-cycle traffic-state memory` 问题：未来轨迹同时受微观运动、中观排队/释放波、
宏观信号周期状态约束。当前最有辨识度的机制是 `Phase-Rolling Queue Memory`。

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
- 当前最强候选是 `50b warmup -> 50b refine`，但还**不能**宣称超过 baseline。

### 当前可比证据

| 模型 | Split | num_samples | ADE | FDE | 说明 |
|------|-------|-------------|-----|-----|------|
| baseline (`D2TP/model_best.pth.tar`) | val | 4 | 38.493 | 78.706 | 当前仓库基线 |
| baseline (`D2TP/model_best.pth.tar`) | test | 4 | 17.812 | 37.568 | 当前仓库基线 |
| CycleState `warmup50_refine50_p0_seqgat_relation_v1` | val | 20 | 75.078 | 154.690 | 当前最佳 true-val 证据 |
| CycleState `warmup50_refine50_p0_seqgat_relation_v1` | val | 4 | 84.772 | 170.878 | 同一候选 quick 口径 |

### 当前判断

- `val` / `test` 差距明显，所有“超过 baseline”的论断都必须显式带上 split。
- 单独把 warmup `teacher_forcing_ratio` 从 `0.8` 降到 `0.6` 已被证伪。
- 比继续压 warmup 更值得推进的是：先补齐 baseline `num_samples=20` 审计，再复核
  当前最佳候选的 `test + num_samples=20`。

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
  --resume experiments/cyclestate/warmup50_refine50_p0_seqgat_relation_v1/checkpoint/model_best.pth.tar \
  --dset_type test \
  --rollout_residual_scale 0.7
```

## 下一步

1. 补齐 baseline 的 `val/test + num_samples=20` 可比线。
2. 复核 `warmup50_refine50_p0_seqgat_relation_v1` 在 `test + num_samples=20` 上是否成立。
3. 如果候选站得住，再按 `queue rollout -> decoder residual -> lane anchor -> state gating`
   顺序做单变量消融。
