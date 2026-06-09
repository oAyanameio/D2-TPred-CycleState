# CycleState: 全周期交通状态记忆轨迹预测 — 代码与研究方法解析

> **文档定位**: 面向研究者与工程协作者的系统化技术文档，从工程架构到学术理论，完整阐述 D2-TPred-CycleState 项目的设计思想、代码实现与研究方法。

---

## 目录

1. [工程架构分析](#1-工程架构分析)
   - 1.1 项目结构概览
   - 1.2 模块划分与职责
   - 1.3 技术栈与依赖关系
   - 1.4 数据流架构图
2. [代码实现详解](#2-代码实现详解)
   - 2.1 数据层：轨迹数据集与批次拼接
     - 2.1.1 TrajectoryDataset 初始化
     - 2.1.2 seq_collate 批次拼接
   - 2.2 编码层：D2-TPred 微观运动与交互编码
     - 2.2.1 轨迹 LSTM 编码
     - 2.2.2 空间图注意力 (GAT + relation_Matrix)
     - 2.2.3 序列图注意力 (seqGAT)
   - 2.3 状态层：CycleState 中观/宏观状态记忆
     - 2.3.0 build_traffic_context — 统一数据接口
     - 2.3.1 Queue-State Feature 构造
     - 2.3.2 Cycle-State Feature 构造
     - 2.3.3 Phase-Rolling Queue Memory (核心创新)
     - 2.3.4 Lane-Consensus Meso Anchor
     - 2.3.5 Phase-Conditioned Gating
     - 2.3.6 Warmup 阶段跨步反传截断
   - 2.4 解码层：状态调制与自回归轨迹生成
     - 2.4.1 解码器初始化
     - 2.4.2 预测期自回归解码
     - 2.4.3 锚定残差注入 (关键修复)
     - 2.4.4 状态残差构造器
     - 2.4.5 Baseline-Compatible 模块定义与初始化
   - 2.5 训练层：分阶段协议与辅助监督
     - 2.5.1 分阶段训练协议
     - 2.5.2 结构化辅助损失
     - 2.5.3 轨迹判别器 (TrajectoryDiscriminator)
     - 2.5.4 状态稳定性监控
   - 2.6 评估层：Best-of-K 采样与指标聚合
     - 2.6.1 训练内验证
     - 2.6.2 离线评估
3. [研究方法阐述](#3-研究方法阐述)
   - 3.1 问题定义与任务描述
   - 3.2 核心假设：全周期交通状态记忆
   - 3.3 技术路线选择依据
   - 3.4 创新点分析
   - 3.5 与现有研究的对比
4. [学术理论支撑](#4-学术理论支撑)
   - 4.1 图注意力网络与不连续依赖
   - 4.2 车道级排队波理论
   - 4.3 信号相位演化模型
   - 4.4 残差注入与暖启动兼容性
   - 4.5 参考文献索引
5. [附录](#附录)
   - 附录 A: 关键代码逐行注释
   - 附录 B: 当前实验状态与性能基线

---

## 1. 工程架构分析

### 1.1 项目结构概览

```
D2-TPred-CycleState/
├── D2TP/                           # 核心代码目录
│   ├── models.py                   # 模型定义（TrajectoryGenerator + CycleStateTrajectoryGenerator）
│   ├── train.py                    # 训练入口与分阶段协议
│   ├── evaluate_model.py           # 离线评估脚本
│   ├── utils.py                    # 工具函数（ADE/FDE、噪声、日志）
│   ├── data/
│   │   ├── __init__.py
│   │   ├── trajectories.py         # 轨迹数据集（TrajectoryDataset）
│   │   └── loader.py              # DataLoader 封装
│   ├── datasets/VTP_C/            # 数据集目录（train/val/test）
│   ├── model_best.pth.tar          # 预训练 D2-TPred baseline 权重
│   └── checkpoint/                 # 历史 checkpoint 存档
├── tests/
│   └── test_cyclestate_protocol.py # 单元测试与回归测试（38 项）
├── experiments/                    # 实验输出目录
│   ├── cyclestate/                 # CycleState 各阶段实验
│   └── d2tpred/                    # Baseline 审计实验
├── quick_runs/                     # 快速 smoke 实验
├── runs/                           # TensorBoard 事件文件
├── docs/
│   ├── cyclestate_research_story.md  # 科研叙事
│   ├── advisor_progress_report.md    # 导师汇报
│   ├── ENGINEERING_ISSUES.md         # 工程问题记录
│   └── plan_evaluation.md            # 计划评估
├── README.md                       # 项目总览
├── EXPERIMENT_LOG.md               # 实验日志
├── CONVERSATION_CONTEXT.md         # 对话上下文同步
├── requirements.txt                # Python 依赖
└── environment.yml                 # Conda 环境
```

### 1.2 模块划分与职责

项目采用**分层模块化**架构，各模块职责如下：

| 模块 | 文件 | 职责 | 输入 | 输出 |
|------|------|------|------|------|
| **数据层** | `data/trajectories.py`, `data/loader.py` | 轨迹数据加载、特征工程、批次拼接 | 原始 CSV 轨迹文件 | `(obs_traj, pred_traj, obs_traj_rel, ..., seq_start_end)` 元组 |
| **编码层** | `models.py` (GATEncoder, seqGATEncoder, TrajectoryGenerator) | 微观运动编码、空间图交互、局部时序交互 | 观测轨迹相对位移 | 轨迹隐藏状态、图交互特征 |
| **状态层** | `models.py` (CycleStateTrajectoryGenerator) | 中观队列状态记忆、宏观周期状态记忆、相位滚动队列 | 轨迹特征、信号状态、车道信息 | queue/cycle hidden states、rollout states |
| **解码层** | `models.py` (pred_lstm_model, decoder residual) | 自回归轨迹生成、状态调制注入 | 编码特征 + 状态记忆 + 噪声 | 未来 12 帧相对位移 `(pred_len, batch, 2)` |
| **训练层** | `train.py` | 分阶段训练协议、辅助损失计算、梯度裁剪、状态稳定性监控 | 数据批次 | 模型权重、日志 |
| **评估层** | `evaluate_model.py`, `train.py` (validate) | Best-of-K 评估、指标聚合 | 模型 + 测试数据 | ADE/FDE 指标 |

### 1.3 技术栈与依赖关系

```
核心依赖:
├── PyTorch (>=1.8)           # 深度学习框架
├── NumPy                     # 数值计算
├── SciPy                     # 距离矩阵计算 (pdist, squareform)
├── TensorBoardX              # 训练可视化
└── Standard Library
    ├── argparse              # 命令行参数解析
    ├── logging               # 日志系统
    ├── random                # 随机数（teacher forcing）
    ├── math                  # 数学函数（角度计算）
    └── unittest              # 单元测试框架
```

**依赖关系图**:

```
                    ┌──────────────┐
                    │  train.py    │ (训练入口)
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
         ┌────▼───┐  ┌────▼───┐  ┌─────▼──────┐
         │models.py│  │utils.py│  │data/loader.py│
         └────┬───┘  └────────┘  └─────┬──────┘
              │                         │
    ┌─────────┼─────────┐    ┌─────────▼──────────┐
    │         │         │    │data/trajectories.py│
    │    ┌────▼───┐     │    └────────────────────┘
    │    │ GAT    │     │
    │    │ seqGAT │     │
    │    └────────┘     │
    │                   │
    │  ┌────────────────▼──────────────────────────┐
    │  │     CycleStateTrajectoryGenerator          │
    │  │  ┌──────────┐ ┌──────────┐ ┌───────────┐  │
    │  │  │  Queue    │ │  Cycle   │ │  Rollout   │  │
    │  │  │  Memory   │ │  Memory  │ │  Memory    │  │
    │  │  └──────────┘ └──────────┘ └───────────┘  │
    │  └────────────────────────────────────────────┘
    │
    └──────────► evaluate_model.py (离线评估)
```

### 1.4 数据流架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          训练数据流 (train.py)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  VTP_C Dataset   seq_collate   batch tuple    build_traffic_context     │
│  ┌──────────┐    ┌────────┐   ┌──────────┐   ┌───────────────────┐     │
│  │ 原始CSV  │───►│ 批次拼接│──►│ (obs,    │──►│ traffic_context   │     │
│  │ 轨迹文件 │    │ 方向计算│   │  pred,   │   │ {agent, signal,   │     │
│  │          │    │ 状态构造│   │  state,  │   │  scene, meso,     │     │
│  └──────────┘    └────────┘   │  mask)   │   │  meta}            │     │
│                               └────┬─────┘   └────────┬──────────┘     │
│                                    │                   │                │
│                                    ▼                   ▼                │
│                         ┌─────────────────────────────────────┐         │
│                         │   CycleStateTrajectoryGenerator      │         │
│                         │                                     │         │
│                         │  ┌─────────────────────────────────┐│         │
│                         │  │ 1. 微观编码 (obs_len=8 帧)       ││         │
│                         │  │   traj_lstm → GAT → seqGAT      ││         │
│                         │  └─────────────────────────────────┘│         │
│                         │  ┌─────────────────────────────────┐│         │
│                         │  │ 2. 中观/宏观编码                 ││         │
│                         │  │   queue_feature → queue_lstm    ││         │
│                         │  │   cycle_feature → cycle_lstm    ││         │
│                         │  │   phase-conditioned gating      ││         │
│                         │  └─────────────────────────────────┘│         │
│                         │  ┌─────────────────────────────────┐│         │
│                         │  │ 3. 解码器初始化                  ││         │
│                         │  │   noise injection +             ││         │
│                         │  │   decoder_state_residual         ││         │
│                         │  └─────────────────────────────────┘│         │
│                         │  ┌─────────────────────────────────┐│         │
│                         │  │ 4. 预测期自回归 (pred_len=12 帧) ││         │
│                         │  │   for each step:                ││         │
│                         │  │   - pred_lstm step              ││         │
│                         │  │   - phase-rolling queue rollout ││         │
│                         │  │   - anchored rollout context    ││         │
│                         │  │   - decoder state residual      ││         │
│                         │  │   - pred_hidden2pos output      ││         │
│                         │  └─────────────────────────────────┘│         │
│                         └──────────────┬──────────────────────┘         │
│                                        │                                │
│                                        ▼                                │
│                         ┌──────────────────────────────┐                │
│                         │  损失计算                     │                │
│                         │  - L2 trajectory loss         │                │
│                         │  - structured queue aux loss  │                │
│                         │  - structured cycle aux loss  │                │
│                         │  - rollout queue aux loss     │                │
│                         │  - GAN loss (adversarial)     │                │
│                         └──────────────────────────────┘                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 代码实现详解

### 2.1 数据层：轨迹数据集与批次拼接

#### 2.1.1 TrajectoryDataset 初始化

```python
# 文件: D2TP/data/trajectories.py, 第 109-216 行
class TrajectoryDataset(Dataset):
```

**核心逻辑**：

1. **帧采样**: 原作者每 9 帧取一个时间点，将原始高频数据降采样为可管理的时间步长。
2. **滑动窗口**: 在每个场景文件中，以 `obs_len + pred_len = 20` 帧为窗口长度，`skip=1` 为步长，滑动提取训练样本。
3. **相对位移**: 对坐标通道 `(x, y)` 做帧间差分，得到 `obs_traj_rel` 和 `pred_traj_rel`，作为模型的核心输入。
4. **方向角**: `agent_direction()` 函数为每个 agent 计算运动方向角，存入第 9 通道，供 `relation_Matrix` 构图使用。
5. **信号状态构造**: 为每个 agent 关联所属交通灯的坐标和当前灯态，并在预测期推断未来灯态序列。

**关键数据结构**:

```python
# 每个样本的通道含义 (9 通道):
# [0]: frame_id       [1]: agent_id
# [2]: x 坐标         [3]: y 坐标
# [4]: lane_id        [5]: 速度
# [6]: 加速度         [7]: 灯态编号
# [8]: 灯态持续时间   [9]: 运动方向角 (agent_direction 添加)

# 信号状态 (4 通道):
# [0]: 停止线 x       [1]: 停止线 y
# [2]: 灯态编号       [3]: 灯态持续时间
```

#### 2.1.2 seq_collate 批次拼接

```python
# 文件: D2TP/data/trajectories.py, 第 39-80 行
def seq_collate(data):
```

**核心逻辑**: 将多个场景的所有 agent 沿 batch 维度拼接，同时记录每个场景的起止索引 `seq_start_end`。这样模型可以按场景边界分离处理，避免跨场景信息泄露。

**输出格式**:
```python
batch = (
    obs_traj,         # (obs_len=8,  total_agents, 10)  # 带方向角的绝对轨迹
    pred_traj,        # (pred_len=12, total_agents, 10)
    obs_traj_rel,     # (obs_len=8,  total_agents, 9)   # 相对位移
    pred_traj_rel,    # (pred_len=12, total_agents, 9)
    obs_state,        # (obs_len=8,  total_agents, 4)   # 信号灯状态
    pred_state,       # (pred_len=12, total_agents, 4)
    non_linear_ped,   # (total_agents,)
    loss_mask,        # (total_agents, 20)
    seq_start_end,    # (num_scenes, 2)
)
```

### 2.2 编码层：D2-TPred 微观运动与交互编码

#### 2.2.1 轨迹 LSTM 编码

```python
# 文件: D2TP/models.py, 第 622-627 行 (CycleState.forward 中)
for input_t in obs_traj_rel[: self.obs_len].chunk(self.obs_len, dim=0):
    inputtraj = input_t[:, :, 2:4]  # 只取相对位移 (dx, dy)
    traj_lstm_h_t, traj_lstm_c_t = self.traj_lstm_model(
        inputtraj.squeeze(0), (traj_lstm_h_t, traj_lstm_c_t)
    )
    traj_lstm_hidden_states += [traj_lstm_h_t]
```

**设计要点**: 逐帧输入相对位移 `(dx, dy)`，通过 LSTMCell 累积个体运动历史，输出 `(batch, 32)` 维隐藏状态。

#### 2.2.2 空间图注意力 (GAT + relation_Matrix)

```python
# 文件: D2TP/models.py, 第 343-398 行
def relation_Matrix(self, curr_dire):
```

**核心算法 — 不连续依赖 (Discontinuous Dependency)**:

1. **距离门控**: 以欧氏距离 156 (像素) 为阈值，筛掉空间上不相关的邻居。
2. **方向扇区约束**: 对于距离内的邻居，进一步检验其是否位于目标车辆前向 ±62° 的扇区内。只有同时满足距离和方向约束的邻居对才被视为有效交互边。
3. **逐帧逐场景构图**: 每帧独立计算关系矩阵 `r ∈ {0,1}^(F×N×N)`，然后按场景分组执行 GAT 聚合。

```python
# GAT 聚合流程 (models.py 第 401-424 行)
def forward(self, obs_traj_embedding, seq_start_end, obs_dire):
    graph_embeded_data = []
    for start, end in seq_start_end.data:
        # 1. 按场景分离
        curr_seq_embedding_traj = obs_traj_embedding[:, start:end, :]
        # 2. 构造该场景的关系矩阵
        Relation = self.relation_Matrix(obs_dire[:, start:end, :])
        # 3. 图注意力聚合
        curr_seq_graph_embedding = self.gat_net(curr_seq_embedding_traj, Relation)
        graph_embeded_data.append(curr_seq_graph_embedding)
    return torch.cat(graph_embeded_data, dim=1)
```

#### 2.2.3 序列图注意力 (seqGAT)

```python
# 文件: D2TP/models.py, 第 635-647 行 (CycleState.forward 中)
for j in range(self.obs_len):
    if j <= 6:
        staend[0, 1] = j + 1
        graph_inter_input = self.seqgatencoder(
            graph_lstm_input[0:(j + 1)].permute(1, 0, 2), staend
        )
    else:
        staend[0, 1] = 7
        graph_inter_input = self.seqgatencoder(
            graph_lstm_input[(j - 6):(j + 1)].permute(1, 0, 2), staend
        )
    graph_lstm_hidden_states += [graph_inter_input[:, -1, :]]
```

**设计要点**: 在时间维上对 GAT 输出做局部窗口 (`kl=6`) 内的注意力聚合，增强短时间窗口内的交互动态建模。早期 `j <= 6` 时使用递增窗口，之后使用固定长度 7 的滑动窗口。

### 2.3 状态层：CycleState 中观/宏观状态记忆

#### 2.3.0 build_traffic_context — 统一数据接口

```python
# 文件: D2TP/models.py, 第 1451-1507 行
def build_traffic_context(self, obs_traj_rel, obs_traj_pos, obs_state, pred_state, seq_start_end):
```

**设计动机**: 将原始 tuple 风格的输入统一组织成结构化 `traffic_context` 字典，为后续迁移到新数据集（如 INT2）提供清晰的 adapter 接口，而不需要重写核心模型。

**数据模型图**:

```
traffic_context
├── agent                          # 个体级信息
│   ├── obs_traj:      (8, N, 10)  # 绝对轨迹
│   ├── obs_traj_rel:  (8, N, 9)   # 相对位移
│   ├── lane_ids:      (N,)        # 车道 ID
│   ├── direction:     (N,)        # 运动方向角
│   └── stopline_distance: (N,)    # 到停止线距离
├── signal                         # 信号灯级信息
│   ├── obs_state:     (8, N, 4)   # 历史信号状态
│   ├── pred_state:    (12, N, 4)  # 未来信号状态
│   ├── phase_ids:     (8, N)      # 历史相位 ID
│   ├── phase_elapsed: (8, N)      # 历史相位持续时间
│   ├── pred_phase_ids:     (12, N)  # 未来相位 ID
│   ├── pred_phase_elapsed: (12, N)  # 未来相位持续时间
│   └── cycle_feature_seq: (8, N, 6)  # 周期特征序列
├── scene                          # 场景级元信息
│   └── seq_start_end:  (S, 2)     # 场景边界索引
└── meso                           # 中观车道级信息
    ├── queue_feature_seq:      (8, N, 11)  # 排队特征序列
    ├── lane_queue_anchor_seq:  (8, N, 11)  # 车道共识锚点序列
    └── queue_targets:          (8, N, 11)  # 辅助监督目标
```

**实现流程**:

```python
# 1. 构造中观队列特征
queue_feature_seq = self.build_queue_features(obs_traj_pos, obs_traj_rel, obs_state, seq_start_end)
# 2. 构造宏观周期特征
cycle_feature_seq = self.build_cycle_features(obs_state)
# 3. 构造车道级共识锚点
lane_queue_anchor_seq = self.build_lane_queue_anchor_seq(queue_feature_seq, lane_ids, seq_start_end)
# 4. 组装字典
traffic_context = {"agent": {...}, "signal": {...}, "scene": {...}, "meso": {...}}
```

#### 2.3.1 Queue-State Feature 构造

```python
# 文件: D2TP/models.py, 第 1086-1193 行
def build_queue_features(self, obs_traj_pos, obs_traj_rel, obs_state, seq_start_end):
```

**11 维 queue feature 语义**:

| 索引 | 名称 | 类型 | 含义 |
|------|------|------|------|
| 0 | queue_count | 回归 | 前方排队车辆数（归一化） |
| 1 | lane_density | 回归 | 同车道局部密度 |
| 2 | lane_mean_speed | 回归 | 同车道平均速度（归一化） |
| 3 | lane_wait_ratio | 回归 | 同车道等待比例（速度 < 阈值） |
| 4 | lane_release_ratio | 回归 | 同车道释放比例（速度 ≥ 阈值） |
| 5 | phase_value | 回归 | 当前灯态编号 / 2 |
| 6 | elapsed_value | 回归 | 灯态已持续时间（归一化） |
| 7 | own_stop_dist | 回归 | 自身到停止线距离（归一化） |
| 8 | lane_queue_length | 回归 | 同车道排队长度代理；观测期用等待车辆数（归一化）近似 |
| 9 | stopline_occupancy | 二分类 | 停止线附近是否有车辆 |
| 10 | front_of_queue | 二分类 | 是否为队首车辆 |

**构造逻辑**:

```python
# 核心伪代码
for each scene, each timestep:
    same_lane = (lane_ids == lane_ids.T)  # 同车道矩阵
    waiting   = (speed < threshold) & (stop_dist < threshold)
    releasing = (speed >= threshold) & (stop_dist < threshold)
    ahead_mask = same_lane & (stop_dist_other < stop_dist_self)

    queue_count    = sum(ahead_mask) / norm    # 前方排队数
    lane_density   = (lane_count - 1) / norm   # 同车道密度
    lane_wait_ratio = sum(waiting * same_lane) / lane_count
    ...
```

#### 2.3.2 Cycle-State Feature 构造

```python
# 文件: D2TP/models.py, 第 1049-1071 行
def build_cycle_features(self, state_seq):
```

**6 维 cycle feature 语义**:

| 索引 | 名称 | 类型 | 含义 |
|------|------|------|------|
| 0-2 | phase_one_hot | 分类 | 灯态 one-hot (红/绿/黄) |
| 3 | elapsed | 回归 | 已持续时间 (归一化) |
| 4 | remaining | 回归 | 剩余时间 (归一化) |
| 5 | phase_change | 二分类 | 是否发生相位切换 |

**设计要点**: 剩余时间通过 `phase_duration_limits[phase] - elapsed_raw` 计算，利用数据集提供的三种灯态持续时间上限 `(38, 47, 2)` 秒。

#### 2.3.3 Phase-Rolling Queue Memory (核心创新)

```python
# 文件: D2TP/models.py, 第 1237-1353 行
def rollout_queue_features(self, prev_queue_feature, current_cycle_feature, 
                           last_pred_offset, step_index):
```

**设计动机**: 观测阶段的 queue feature 只能描述"看到的最后一刻"，但在解码期，队列会继续随相位推进而积累或释放。这个函数显式建模预测期队列状态的演化。

**核心演化规则**:

```python
# 相位条件的状态更新 (简化版)
is_red_like    = (phase_id == 0).float()
is_green_like  = (phase_id == 1).float()
is_yellow_like = (phase_id == 2).float()
pred_speed_norm = last_pred_offset.norm() / speed_threshold
progress = elapsed / 2.0
remaining_progress = remaining / 2.0

# 红灯下: 等待比例上升，释放比例下降，排队长度增加
waiting_ratio += 0.08 * is_red_like * (1 - progress) - 0.12 * is_green_like * pred_speed_norm
release_ratio += 0.14 * is_green_like * (1 - remaining_progress + pred_speed_norm) - 0.08 * is_red_like
lane_queue_length += 0.10 * is_red_like - 0.12 * is_green_like * pred_speed_norm + 0.05 * phase_change

# 绿灯下: 距离停止线逐步减小 (车辆前进)
stop_dist -= 0.08 * pred_speed_norm - 0.03 * step_discount + 0.02 * phase_change
```

**关键设计**:
- 演化系数 (0.08, 0.12, 0.14 等) 是**可解释的物理先验**，反映不同灯态下排队/释放的典型速率差异。
- `pred_speed_norm` 将预测运动反馈到状态演化中，形成"轨迹 → 状态 → 轨迹"的闭环。
- 所有变量通过 `torch.clamp` 约束在合理范围内，防止数值发散。

#### 2.3.4 Lane-Consensus Meso Anchor

```python
# 文件: D2TP/models.py, 第 1209-1235 行
def build_lane_queue_anchor_seq(self, queue_feature_seq, lane_ids, seq_start_end):
    """同车道 vehicle 共享一个车道级中观共识参考。"""
    for each scene, each timestep:
        for each unique lane_id:
            lane_mean = mean(queue_features[lane_mask])
            anchor[lane_mask] = lane_mean  # 广播到同车道所有车辆
```

**设计动机**: 中观 queue-wave 状态不应完全由单个 agent 的局部噪声决定，同一条车道上的车辆应共享一个更平滑的"车道级状态共识"。在预测期，这个 anchor 会动态重聚合，追踪车道级状态的演化。

#### 2.3.5 Phase-Conditioned Gating

```python
# 文件: D2TP/models.py, 第 1666-1674 行
# 编码阶段门控
phase_gate_input = torch.cat((light_state_embedding, queue_last, cycle_last), dim=1)
gated_queue_last = queue_last * self.queue_context_gate(phase_gate_input)
gated_cycle_last = cycle_last * self.cycle_context_gate(phase_gate_input)

# 解码阶段门控 (第 1741-1753 行)
rollout_queue_h_t = rollout_queue_h_t * self.queue_rollout_gate(
    torch.cat((light_state_embedding, rollout_queue_h_t, cycle_step_embedding), dim=1)
)
```

**设计动机**: 同样的 queue/cycle 记忆，在红灯、绿灯、黄灯下的作用不可能一致。门控使状态记忆受当前灯态条件调制，而不是简单拼接。

#### 2.3.6 Warmup 阶段跨步反传截断 (maybe_detach_rollout_state)

```python
# 文件: D2TP/models.py, 第 1549-1569 行
def maybe_detach_rollout_state(
    self, rollout_queue_feature, rollout_lane_queue_anchor,
    rollout_queue_h_t, rollout_queue_c_t,
):
    """warmup 阶段截断预测期 meso rollout 跨步反传。"""
    if not self.detach_rollout_state or not self.training:
        return (rollout_queue_feature, rollout_lane_queue_anchor,
                rollout_queue_h_t, rollout_queue_c_t)
    return (
        rollout_queue_feature.detach(),
        rollout_lane_queue_anchor.detach(),
        rollout_queue_h_t.detach(),
        rollout_queue_c_t.detach(),
    )
```

**设计动机**: 在 warmup 阶段，rollout 分支尚未稳定，如果允许跨步梯度反传（即 step t 的 rollout 状态通过 step t+1 的 rollout 计算反向传播），会形成复杂的长链梯度依赖，导致训练不稳定。通过 `.detach()` 截断跨步反传，模型先专注于学习"单步可预测的 rollout 状态"，再在 refine 阶段恢复跨步反传，学习"长链状态耦合"。

**渐进式策略**:
- `warmup` 阶段: `detach_rollout_state=True`，截断跨步反传，每个预测步的 rollout 独立优化。
- `refine` 阶段: `detach_rollout_state=False`，恢复跨步反传，让模型学习完整的 rollout 链。
- 推理阶段: 始终不截断（`if not self.training` 条件），确保推理时状态信息完整流动。

### 2.4 解码层：状态调制与自回归轨迹生成

#### 2.4.1 解码器初始化

```python
# 文件: D2TP/models.py, 第 1691-1703 行
# 1. 拼接编码特征 + 噪声
pred_lstm_hidden = self.add_noise(encoded_before_noise_hidden, seq_start_end)

# 2. 注入初始状态残差
init_state_residual = self.build_decoder_state_residual(
    light_state_embedding, gated_queue_last, gated_cycle_last
)
if init_state_residual is not None:
    pred_lstm_hidden = pred_lstm_hidden + init_state_residual
```

**噪声注入策略**: 噪声不是"每辆车一份"，而是"每个场景一份"。同一场景内所有车辆复制同一条噪声向量，意味着模型采样的是"该场景未来整体演化模式"的不同可能性，而不是彼此独立的随机抖动。

#### 2.4.2 预测期自回归解码

```python
# 训练态 (第 1720-1800 行)
for i in range(pred_len):
    # 1. 预测 LSTM 单步
    pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(
        input_t, (pred_lstm_hidden, pred_lstm_c_t)
    )
    
    # 2. 获取当前步的灯态与周期上下文
    light_state_embedding, current_cycle_feature, cycle_step_embedding = \
        self.get_decode_step_context(i, pred_traj_rel, obs_traj_pos, obs_state, pred_state)
    
    # 3. Phase-Rolling Queue Memory 更新
    if not disable_queue_rollout:
        rollout_info = self.rollout_queue_step(
            rollout_queue_feature, rollout_lane_queue_anchor, ...
        )
        # 锚定残差注入
        queue_context = self.build_rollout_decode_queue_context(
            gated_queue_last, rollout_queue_h_t, light_state_embedding
        )
        # warmup 阶段截断跨步反传
        rollout_queue_feature, ... = self.maybe_detach_rollout_state(...)
    
    # 4. 解码器状态残差注入
    step_state_residual = self.build_decoder_state_residual(
        light_state_embedding, queue_context, cycle_step_embedding
    )
    pred_lstm_hidden = pred_lstm_hidden + step_state_residual
    
    # 5. 输出头
    pred_input = torch.cat((light_state_embedding, pred_lstm_hidden), dim=1)
    output = self.pred_hidden2pos(pred_input)
```

#### 2.4.3 锚定残差注入 (关键修复)

```python
# 文件: D2TP/models.py, 第 1525-1547 行
def build_rollout_decode_queue_context(self, observed_queue_context, 
                                        rollout_queue_context, light_state_embedding):
    """让 rollout queue context 以锚定残差方式进入 decoder。"""
    rollout_delta = rollout_queue_context - observed_queue_context
    rollout_delta = torch.tanh(rollout_delta)  # 幅值约束
    rollout_gate = self.rollout_decode_context_gate(
        torch.cat((light_state_embedding, observed_queue_context, rollout_queue_context), dim=1)
    )
    return observed_queue_context + self.rollout_residual_scale * rollout_gate * rollout_delta
```

**设计动机**: 早期版本直接将 `rollout_queue_h_t` 整块替换 queue context，导致 decoder 在短训练时被未稳定的 rollout 状态接管。修复后改为**"观测锚点 + 门控 rollout delta"**的残差注入方式：
- 当 `rollout_residual_scale=0` 时，退化为纯观测锚点注入；
- 当 gate 接近 0 时，rollout 信息被抑制，保持观测期状态；
- `tanh` 约束 delta 幅值，防止状态分支在早期短训中接管 decoder。

#### 2.4.4 状态残差构造器 (build_decoder_state_residual)

```python
# 文件: D2TP/models.py, 第 1509-1523 行
def build_decoder_state_residual(self, light_state_embedding, queue_context, cycle_context):
    """把交通状态记忆映射成与 baseline 解码器同维的残差调制量。"""
    if self.disable_decoder_state_residual:
        return None
    decoder_state_context = torch.cat(
        (light_state_embedding, queue_context, cycle_context), dim=1
    )
    state_residual = self.decoder_state_residual(decoder_state_context)
    state_gate = self.decoder_state_gate(decoder_state_context)
    return state_gate * state_residual
```

**数学形式**:

```
residual = σ(W_gate · context) ⊙ tanh(W_res · context)
```

其中 `context = [light_embedding; queue_context; cycle_context]`，`⊙` 为逐元素乘法。

**调用位置**: 该函数在解码器的两个位置被调用：
1. **解码器初始化时** (step 0): 注入初始状态残差，为解码器提供"起始状态调制"。
2. **每个预测步**: 注入当前步的状态残差，向解码器提供"当前交通状态应该如何影响轨迹"的信息。

#### 2.4.5 Baseline-Compatible 模块定义与初始化

```python
# 文件: D2TP/models.py, 第 1000-1030 行
self.decoder_state_residual = nn.Sequential(
    nn.Linear(light_embedding + queue_hidden + cycle_hidden, pred_lstm_hidden),
    nn.ReLU(),
    nn.Linear(pred_lstm_hidden, pred_lstm_hidden),
    nn.Tanh(),  # 输出范围 [-1, 1]，限制残差幅值
)
self.decoder_state_gate = nn.Sequential(
    nn.Linear(light_embedding + queue_hidden + cycle_hidden, pred_lstm_hidden),
    nn.ReLU(),
    nn.Linear(pred_lstm_hidden, pred_lstm_hidden),
    nn.Sigmoid(),
)
# 初始化策略: residual 权重为零，gate bias 为 -2
nn.init.zeros_(self.decoder_state_residual[2].weight)
nn.init.constant_(self.decoder_state_gate[2].bias, -2.0)
```

**设计动机**: 保持原始 D2-TPred 解码器形状不变，通过残差方式注入状态记忆。初始化策略确保训练开始时状态残差为零，模型从纯 baseline 能力起步，逐步学习状态调制。

### 2.5 训练层：分阶段协议与辅助监督

#### 2.5.1 分阶段训练协议

```python
# 文件: D2TP/train.py, 第 247-278 行
TRAIN_STAGE_DEFAULTS = {
    "warmup": {
        "generator_only": True,
        "gan_weight": 0.0,
        "aux_queue_weight": 10.0,       # 高辅助权重，迫使状态分支学习
        "aux_cycle_weight": 5.0,
        "teacher_forcing_ratio": 0.8,   # 高 teacher forcing，稳定早期解码
        "grad_clip": 1.0,
        "rollout_residual_scale": 0.35, # 低 rollout 影响，保护解码器
        "detach_rollout_state": True,   # 截断跨步反传，先学"可预测状态"
    },
    "refine": {
        "generator_only": True,
        "gan_weight": 0.0,
        "aux_queue_weight": 3.0,        # 降低辅助权重，让轨迹损失主导
        "aux_cycle_weight": 1.5,
        "teacher_forcing_ratio": 0.6,   # 衰减 teacher forcing
        "grad_clip": 1.0,
        "rollout_residual_scale": 0.7,  # 放大 rollout 影响
        "detach_rollout_state": False,  # 恢复跨步反传，学"长链状态耦合"
    },
    "adversarial": {
        "generator_only": False,
        "gan_weight": 50.0,
        ...
    },
}
```

**训练阶段设计哲学**:

| 阶段 | 目标 | 策略 |
|------|------|------|
| `warmup` | 稳定 queue/cycle 状态分支，保护 baseline 解码器 | 高 aux 权重、高 teacher forcing、低 rollout 影响、截断跨步反传 |
| `refine` | 让状态记忆更好地辅助轨迹重建 | 降低 aux、衰减 teacher forcing、放大 rollout、恢复跨步反传 |
| `adversarial` | GAN 精修轨迹分布 | 低权重 GAN 作为分布精修项引入 |

#### 2.5.2 结构化辅助损失

```python
# 文件: D2TP/train.py, 第 391-486 行
def compute_structured_aux_losses(...):
    # Queue 损失
    queue_reg_idx = [0, 1, 2, 3]  # queue_count, wait_ratio, release_ratio, lane_queue_length
    queue_cls_idx = [4, 5]        # stopline_occupancy, front_of_queue
    losses["queue_reg_loss"] = MSE(queue_pred[reg], queue_target[reg])
    losses["queue_cls_loss"] = BCE(queue_pred[cls], queue_target[cls])
    
    # Cycle 损失
    losses["cycle_phase_loss"] = CrossEntropy(phase_pred, phase_target)
    losses["cycle_time_loss"] = MSE(elapsed_remaining_pred, elapsed_remaining_target)
    losses["cycle_change_loss"] = BCE(phase_change_pred, phase_change_target)
    
    # Rollout 损失 (预测期每一步的 queue 状态监督)
    losses["queue_rollout_reg_loss"] = MSE(rollout_pred_flat[reg], rollout_target_flat[reg])
    losses["queue_rollout_cls_loss"] = BCE(rollout_pred_flat[cls], rollout_target_flat[cls])
```

**总损失**:

```
L_total = L2_trajectory 
        + gan_weight * GAN_loss 
        + aux_queue_weight * (queue_reg + queue_cls)
        + aux_rollout_weight * (queue_rollout_reg + queue_rollout_cls)
        + aux_cycle_weight * (cycle_phase + cycle_time + cycle_change)
```

#### 2.5.3 轨迹判别器 (TrajectoryDiscriminator)

```python
# 文件: D2TP/models.py, 第 1899-1977 行
class TrajectoryDiscriminator(nn.Module):
```

**设计思路**: 判别器不是只看轨迹坐标，而是同时看轨迹和交通灯状态，判断一段行为序列是否符合真实的交通场景规律。其架构与生成器对称，采用双路编码 + 时序融合的结构。

**架构图**:

```
                        ┌─────────────────────────┐
                        │  TrajectoryDiscriminator  │
                        └───────────┬─────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
         ┌────▼────┐          ┌────▼────┐          ┌─────▼─────┐
         │ 轨迹编码  │          │ 灯态编码  │          │  融合判别  │
         │ 分支      │          │ 分支      │          │  头        │
         └────┬────┘          └────┬────┘          └─────┬─────┘
              │                     │                     │
    ┌─────────▼─────────┐  ┌───────▼────────┐  ┌────────▼────────┐
    │ pos_embedding     │  │ light_embedding│  │ merge_embedding │
    │ (2→32→16)        │  │ (4→32→16)     │  │ (64→32→1)      │
    └─────────┬─────────┘  └───────┬────────┘  └────────▲────────┘
              │                     │                     │
    ┌─────────▼─────────┐  ┌───────▼────────┐           │
    │ pos_part_lstm     │  │ state_part_lstm │           │
    │ (逐帧编码轨迹坐标)  │  │ (逐帧编码灯态)   │           │
    └─────────┬─────────┘  └───────┬────────┘           │
              │                     │                     │
              └──────────┬──────────┘                     │
                         │                                │
                  ┌──────▼──────┐                         │
                  │ merge_lstm  │─────────────────────────┘
                  │ (时序融合)   │
                  └─────────────┘
```

**核心组件说明**:

| 组件 | 输入维度 | 输出维度 | 作用 |
|------|----------|----------|------|
| `pos_embedding` | 2 (dx, dy) | 16 | 将轨迹坐标映射到嵌入空间 |
| `light_embedding` | 4 (信号状态) | 16 | 将交通灯状态映射到嵌入空间 |
| `pos_part_lstm` | 16 | 64 | 在时间维上编码轨迹几何序列 |
| `state_part_lstm` | 16 | 64 | 在时间维上编码灯态序列 |
| `merge_lstm` | 128 (拼接) | 128 | 融合轨迹与灯态的时序特征 |
| `merge_embedding` | 64 | 1 | 输出真假判别分数 |

**判别流程**:

```python
# 伪代码
for each timestep in [obs_len + pred_len]:
    # 1. 轨迹坐标编码
    pos_embed = pos_embedding(traj_rel[t, :, 2:4])
    pos_h, pos_c = pos_part_lstm(pos_embed, (pos_h, pos_c))
    
    # 2. 交通灯状态编码
    light_embed = light_embedding(state[t])
    state_h, state_c = state_part_lstm(light_embed, (state_h, state_c))
    
    # 3. 时序融合
    merge_h, merge_c = merge_lstm(cat(pos_h, state_h), (merge_h, merge_c))

# 4. 判别输出
score = merge_embedding(merge_h)  # (N, 1) 实数分数
```

**与生成器的对称性**:

| 生成器组件 | 判别器对应组件 | 对称关系 |
|-----------|---------------|---------|
| `traj_lstm_model` | `pos_part_lstm` | 逐帧编码轨迹几何 |
| `light_embedding` | `light_embedding` | 灯态嵌入（共享设计） |
| `pred_lstm_model` | `merge_lstm` | 时序融合建模 |
| `pred_hidden2pos` | `merge_embedding` | 输出头（回归 vs 判别） |

#### 2.5.4 状态稳定性监控

```python
# 文件: D2TP/train.py, 第 544-559 行
def extract_state_stability_metrics(debug_info, pred_offsets):
    return {
        "decoder_state_init_residual_norm": ...,  # 初始残差幅值
        "decoder_state_step_residual_norm": ...,   # 逐步残差幅值
        "queue_rollout_hidden_norm": ...,          # rollout hidden 范数
        "pred_offset_norm": ...,                   # 预测位移范数
    }
```

**设计动机**: 用于定位 "aux 在降但 ADE/FDE 崩坏" 的根因。如果 `queue_rollout_hidden_norm` 在训练后半程急剧增长，说明 rollout 状态出现了不稳定漂移。

### 2.6 评估层：Best-of-K 采样与指标聚合

#### 2.6.1 训练内验证

```python
# 文件: D2TP/train.py, 第 1178-1241 行
def validate(args, model, val_loader, epoch, writer):
    for batch in val_loader:
        for _ in range(args.num_val_samples):
            pred_traj_fake_rel = forward_generator(batch, teacher_forcing_ratio=0.0)
            # 计算逐 agent 的原始 ADE/FDE
            ade_raw, fde_raw = compute_raw_displacement_metrics(pred_traj_gt, pred_traj_fake)
            ade_candidates.append(ade_raw)
            fde_candidates.append(fde_raw)
        # 按场景做 best-of-K 聚合
        ade_, fde_ = compute_best_of_k_metrics(
            ade_candidates, fde_candidates, seq_start_end, pred_len, total_traj
        )
```

**验证协议**:
- 训练内验证使用 `--val_dset_type` 指定的 split（默认 `val`），`test` 只用于最终复核。
- 采用 `num_val_samples` 次采样，按场景做 best-of-K 聚合，与离线评估口径一致。
- 验证调度支持 smoke run（batch 级快速反馈）和正式训练（epoch 级验证）两种模式。

#### 2.6.2 离线评估

```python
# 文件: D2TP/evaluate_model.py
# 支持 --dset_type {val, test} 和 --num_samples 控制采样次数
# 评估结果按加权平均聚合，与论文口径对齐
```

**指标定义**:

```
ADE = (1 / (N * T_pred)) * Σ_i min_k Σ_t ||pred_i^k(t) - gt_i(t)||
FDE = (1 / N)           * Σ_i min_k ||pred_i^k(T) - gt_i(T)||
```

其中 `min_k` 表示在 K 次采样中选择误差最小的那条轨迹。

---

## 3. 研究方法阐述

### 3.1 问题定义与任务描述

**任务**: 信号灯路口轨迹预测 (Trajectory Prediction at Signalized Intersections)

**形式化定义**: 给定观测期内 N 辆车的轨迹历史和交通灯状态序列，预测未来 T_pred 帧内每辆车的运动轨迹。

```
输入:
- obs_traj:  (T_obs=8,  N, 10)  # 历史轨迹
- obs_state: (T_obs=8,  N, 4)   # 历史信号状态
- pred_state: (T_pred=12, N, 4)  # 未来信号状态 (已知)

输出:
- pred_traj: (T_pred=12, N, 2)   # 未来相对位移 (dx, dy)
```

**问题特点**:
1. **多模态**: 同样的历史可能对应多种未来（如停车等待 vs 加速通过）。
2. **强交互**: 车辆之间通过排队、跟驰、让行等机制相互耦合。
3. **场景依赖**: 交通灯相位周期性地约束和重组交通流行为。
4. **状态演化**: 路口状态（排队长度、释放波、相位切换）本身在预测期内持续变化。

### 3.2 核心假设：全周期交通状态记忆

**一句话表述**:

> 信号灯路口轨迹预测本质上不是单纯的轨迹外推问题，而是一个**全周期交通状态记忆与演化建模**问题。

**三层状态分解**:

```
┌──────────────────────────────────────────────────────────┐
│                    Macro: Cycle-State                     │
│  相位 one-hot、已持续时间、剩余时间、相位切换              │
│  "当前是什么灯？还要多久变灯？"                            │
├──────────────────────────────────────────────────────────┤
│                    Meso: Queue-State                      │
│  排队车辆数、等待/释放比例、停止线占用、队首标记            │
│  "车道上有多少车在等？释放波来了吗？"                      │
├──────────────────────────────────────────────────────────┤
│                    Micro: Motion-State                    │
│  个体运动历史、邻居空间交互、局部时序交互                  │
│  "这辆车怎么动？周边车怎么影响它？"                        │
└──────────────────────────────────────────────────────────┘
```

**核心论断**:

1. 中观 queue-wave 和宏观 cycle-state 不是"附加特征"，而是**支配信号灯路口交通行为的本构状态**。
2. 这些状态在预测期内不应被冻结，而应随相位推进和车辆运动**持续演化**。
3. 显式建模这些状态，比仅用隐式 hidden feature 更能捕捉信号灯路口的特殊结构。

### 3.3 技术路线选择依据

| 技术选择 | 依据 |
|----------|------|
| 以 D2-TPred 为 baseline | D2-TPred 是当前最直接针对 signalized intersection 的预测方法，其 discontinuous dependency 建模为微观交互提供了良好基础 |
| 基于 LSTM 的编码器-解码器 | LSTM 天然适合序列建模，且与 D2-TPred baseline 兼容，便于 warm-start 和公平对比 |
| 图注意力 (GAT) 用于空间交互 | 关系矩阵约束使交互建模更精确，避免全连接图的信息稀释 |
| 显式 queue/cycle feature 而非隐式学习 | 可解释性强，支持结构化辅助监督，便于消融分析 |
| 残差注入而非直接拼接 | 保护 baseline 解码器能力，降低新分支的学习难度 |
| 分阶段训练协议 | 避免多目标优化冲突，先稳定状态分支再联合优化 |

### 3.4 创新点分析

#### 创新 1: 问题重构 — 从轨迹预测到状态记忆

**不再是**: 在 D2-TPred 上加一个时序模块。
**而是**: 重新定义信号灯路口预测为 full-cycle traffic-state memory 问题。

**创新性**: 大多数轨迹预测方法围绕"如何更好地编码历史轨迹和交互"展开，而 CycleState 的核心主张是"需要显式建模并预测交通状态本身的演化"。这是一个**问题表述层面**的创新。

#### 创新 2: Micro-Meso-Macro 分层建模框架

**对比**: 现有方法通常只做单层 feature fusion（如 LaneGCN 融合 lane graph，Scene Transformer 融合 agent-road-time）。

**优势**:
- 交通语义更清晰（排队 vs 相位 vs 运动）
- 更适合做消融（可以独立开关每层）
