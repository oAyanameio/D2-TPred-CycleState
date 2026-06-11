# CycleState 技术文档

> 这份文档只保留工程协作真正需要的内容：架构总览、关键数据流、实现约束、常见修改入口。

## 1. 项目结构

```text
D2-TPred-CycleState/
├── D2TP/
│   ├── models.py
│   ├── train.py
│   ├── evaluate_model.py
│   ├── utils.py
│   └── data/
├── tests/test_cyclestate_protocol.py
├── scripts/run_full_pipeline.sh
├── README.md
├── EXPERIMENT_LOG.md
└── docs/
    ├── PLAN.md
    ├── technical_documentation.md
    ├── AI_EXPERIMENT_DELEGATION_GUIDE.md
    ├── ENGINEERING_ISSUES.md
    ├── COMPREHENSIVE_ANALYSIS.md
    └── METHOD_AND_ARCHITECTURE_ANALYSIS.md
```

## 2. 核心文件职责

| 文件 | 职责 |
|------|------|
| `D2TP/models.py` | baseline 与 CycleState 模型定义，状态分支、rollout、消融配置 |
| `D2TP/train.py` | 训练入口、阶段默认值、aux losses、训练内验证、checkpoint |
| `D2TP/evaluate_model.py` | 离线 `val/test` 评估、best-of-K 聚合 |
| `D2TP/utils.py` | ADE/FDE、日志、辅助工具 |
| `tests/test_cyclestate_protocol.py` | 协议、回归、文档链路、CLI 契约 |

## 3. 模型主线

CycleState 延续原始 `D2-TPred` 的微观轨迹与交互建模，并新增两层显式状态：

- `micro`：轨迹 LSTM、空间图交互、局部时序交互
- `meso`：lane-level queue-state memory
- `macro`：cycle-state memory

当前最重要的机制是 `Phase-Rolling Queue Memory`：

- queue state 不在 decoder 初始化后冻结
- 预测期每一步根据 phase progression、上一时刻预测运动和 lane anchor 滚动更新
- decoder 使用 `observed queue context + bounded rollout residual`

## 4. 训练与评估数据流

1. `data/trajectories.py` 读取轨迹与信号相关特征。
2. `seq_collate` 组装 batch，输出原始 tuple。
3. `build_traffic_context` 把 tuple 适配成统一的 `traffic_context`。
4. `TrajectoryGenerator` / `CycleStateTrajectoryGenerator` 前向：
   - 编码观测期轨迹
   - 构造 queue / cycle features
   - 初始化 decoder
   - 自回归预测未来轨迹
5. `train.py` 计算：
   - trajectory loss
   - queue / cycle main aux
   - rollout queue aux
   - adversarial loss（若启用）
6. `evaluate_model.py` 在固定 split 和 `num_samples` 下做 best-of-K 聚合。

## 5. 当前关键实现约束

### 5.1 Oracle 信号假设

CycleState 与 baseline 都把未来真实信号状态 `pred_state` 作为输入。当前 comparable 结果在这个假设上对齐，但它仍是需要单独讨论的限制条件。

### 5.2 统一数据接口

`traffic_context` 是后续所有改造的公共入口。若要新增状态分支或修改信号输入，优先沿这条接口改，不要把 tuple 解析逻辑散落回各处。

### 5.3 消融配置只能走 `AblationConfig`

`disable_state_gating`、`disable_queue_rollout`、`disable_lane_queue_anchor`、
`disable_decoder_state_residual`、`disable_aux_losses` 统一由 `models.AblationConfig`
管理。train / evaluate / protocol-log 必须共用同一套语义。

### 5.4 rollout 系数只能走 `RolloutQueueCoefs`

queue rollout 的相位驱动系数、拼接权重、clamp 上界都集中在
`RolloutQueueCoefs`。不要把 magic number 重新写回 `rollout_queue_features`。

### 5.5 aux 监督有意区分 main 与 rollout

`compute_structured_aux_losses` 当前采用：

- `main aux`：末帧监督
- `rollout aux`：全预测期序列监督

如果要改这条约束，必须同步补测试并更新 `README.md` / `EXPERIMENT_LOG.md` / `docs/PLAN.md`。

### 5.6 checkpoint 契约

当前 checkpoint 相关契约包括：

- `num_val_samples` 对齐检查
- 数据集归一化参数持久化
- cyclestate resume 恢复 `start_epoch`

这三类逻辑分别影响实验可比性、输入尺度一致性和断点续训正确性。

## 6. 常见修改入口

| 目标 | 主要文件 | 备注 |
|------|----------|------|
| 调整训练协议 | `D2TP/train.py`, `tests/test_cyclestate_protocol.py` | 默认先做 `protocol-check`，再谈长训练 |
| 调整状态分支 | `D2TP/models.py`, `tests/test_cyclestate_protocol.py` | 关注 queue / cycle 进入预测期的路径 |
| 调整评估口径 | `D2TP/evaluate_model.py`, `README.md`, `EXPERIMENT_LOG.md` | 必须保持 split / `num_samples` 明确 |
| 调整实验叙事 | `README.md`, `EXPERIMENT_LOG.md`, `docs/PLAN.md` | 三份文档要同步 |
| 调整可运行脚本 | `scripts/run_full_pipeline.sh` | 不能打破 train -> checkpoint -> evaluate 链路 |

## 7. 变更后的最小验证

```bash
python -m unittest discover -s tests -p 'test_cyclestate_protocol.py'
python -m py_compile D2TP/train.py D2TP/models.py D2TP/evaluate_model.py
```

如果改了训练协议、日志或 checkpoint 路径，再补一个 1-batch smoke。

## 8. 文档分工

- `README.md`：项目入口
- `EXPERIMENT_LOG.md`：当前证据
- `docs/PLAN.md`：活跃 backlog
- 本文档：实现结构与修改入口
