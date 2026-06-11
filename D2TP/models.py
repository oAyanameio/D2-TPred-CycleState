"""D2-TPred 模型定义。

这份实现对应论文 D2-TPred: Discontinuous Dependency for Trajectory Prediction
Under Traffic Lights。模型的核心思想是把车辆轨迹预测拆成三类信息联合建模：
1. 车辆自身历史运动模式。
2. 场景中车辆之间的交互关系。
3. 与交通灯相关的状态约束。

生成器负责输出未来相对位移，判别器负责区分真实轨迹和生成轨迹。
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

# from scipy import stats
from utils import relative_to_abs
import math
import numpy as np
import time
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform


@dataclass(frozen=True)
class RolloutQueueCoefs:
    """Phase 3 #16:``rollout_queue_features`` 物理系数集中配置。

    这些系数原本以裸字面量形式直接出现在 ``CycleStateTrajectoryGenerator.rollout_queue_features``
    方法体内（如 ``0.08``、``0.10``、``0.12``、``0.14``、``0.6``、``0.4``、``0.5``、``1.5`` 等），
    既不可从外部调参，也无法在不阅读源码的情况下进行消融。现在统一封装到 frozen
    dataclass 中，作为 ``CycleStateTrajectoryGenerator.__init__`` 的 ``rollout_queue_coefs``
    形参传入；调用方也可以传 ``None`` 触发默认值。

    字段命名按"被驱动的量 + 相位 / 速度 / phase_change 方向"组织，例如
    ``waiting_ratio_red_inc`` 表示"红灯期 waiting_ratio 的每步增量"。
    单位都是"无量纲的 queue feature 增量 / 步"，因此 0.10 ≈ 10 步后稳定红灯
    可让 waiting_ratio 增长到 1.0 上限附近。

    暴露这些系数后可以做：
    1. 消融：把 ``waiting_ratio_red_inc`` 置 0 验证"红灯期等待比例增长"是否真的
       是 ADE 改进的关键因素。
    2. Stage 协议对比：warmup 用小系数（如 0.04）抑制 rollout 发散，refine 用
       默认 0.08/0.10/0.12 恢复相位推进信号。
    3. Sensitivity grid：在 [0, 0.3] 区间扫描 ``queue_count_stopline_weight``，
       寻找优于硬编码默认值的配置。
    """

    # --- waiting_ratio 相位驱动 -------------------------------------------------
    waiting_ratio_red_inc: float = 0.08
    waiting_ratio_yellow_inc: float = 0.03
    waiting_ratio_green_dec: float = 0.12

    # --- release_ratio 相位驱动 -------------------------------------------------
    release_ratio_green_inc: float = 0.14
    release_ratio_red_dec: float = 0.08
    release_ratio_yellow_dec: float = 0.04

    # --- lane_queue_length 相位 / 切换驱动 --------------------------------------
    lane_queue_length_red_inc: float = 0.10
    lane_queue_length_yellow_inc: float = 0.03
    lane_queue_length_green_dec: float = 0.12
    lane_queue_length_phase_change_inc: float = 0.05

    # --- stopline_occupancy 相位驱动 -------------------------------------------
    stopline_occupancy_red_inc: float = 0.10
    stopline_occupancy_green_dec: float = 0.12

    # --- front_of_queue 相位驱动 -----------------------------------------------
    front_of_queue_red_inc: float = 0.05
    front_of_queue_green_dec: float = 0.05

    # --- stop_dist 速度 / 切换 / step 驱动 --------------------------------------
    stop_dist_pred_speed_dec: float = 0.08
    stop_dist_step_discount_dec: float = 0.03
    stop_dist_phase_change_inc: float = 0.02

    # --- queue_count 内部加权 (lane_queue_length + w * stopline_occupancy) ------
    queue_count_stopline_weight: float = 0.5

    # --- lane_density 拼接权重 (w_prev * prev + w_lane * lane_queue_length) -----
    lane_density_prev_weight: float = 0.6
    lane_density_lane_queue_weight: float = 0.4

    # --- lane_mean_speed 拼接权重 (w_prev * prev + w_pred * pred_speed) --------
    lane_mean_speed_prev_weight: float = 0.6
    lane_mean_speed_pred_weight: float = 0.4

    # --- 物理上界 (clamp max) ---------------------------------------------------
    # 下界固定为 0.0 (queue feature 不允许为负), 上界与具体物理量挂钩,
    # 默认值与原硬编码保持一致, 留给数据分布变化时再调整。
    waiting_ratio_max: float = 1.0
    release_ratio_max: float = 1.0
    lane_queue_length_max: float = 1.5
    stopline_occupancy_max: float = 1.0
    front_of_queue_max: float = 1.0
    stop_dist_max: float = 2.0
    queue_count_max: float = 1.5
    lane_density_max: float = 1.5
    lane_mean_speed_max: float = 1.5


@dataclass(frozen=True)
class AblationConfig:
    """Phase 4 #31: 集中管理 CycleState 的 disable_* 开关。"""

    disable_state_gating: bool = False
    disable_queue_rollout: bool = False
    disable_lane_queue_anchor: bool = False
    disable_decoder_state_residual: bool = False
    disable_aux_losses: bool = False

    @classmethod
    def from_args(cls, args):
        return cls(
            disable_state_gating=bool(getattr(args, "disable_state_gating", False)),
            disable_queue_rollout=bool(getattr(args, "disable_queue_rollout", False)),
            disable_lane_queue_anchor=bool(
                getattr(args, "disable_lane_queue_anchor", False)
            ),
            disable_decoder_state_residual=bool(
                getattr(args, "disable_decoder_state_residual", False)
            ),
            disable_aux_losses=bool(getattr(args, "disable_aux_losses", False)),
        )

    def effective_flags(self):
        force_all = self.disable_aux_losses
        return {
            "disable_state_gating": force_all or self.disable_state_gating,
            "disable_queue_rollout": force_all or self.disable_queue_rollout,
            "disable_lane_queue_anchor": force_all or self.disable_lane_queue_anchor,
            "disable_decoder_state_residual": (
                force_all or self.disable_decoder_state_residual
            ),
        }

    def to_model_kwargs(self):
        return {
            **self.effective_flags(),
            "disable_aux_losses": self.disable_aux_losses,
        }


class NoiseSampler(ABC):
    """噪声采样抽象。

    Phase 5 #30: 用显式 sampler 对象替代 ``noise_type`` 字符串分发，
    让调用方既可继续传 `"gaussian"` / `"uniform"` 保持兼容，也可传
    自定义 sampler 实例，避免噪声策略扩展时把条件分支散落到模型里。
    """

    name = "unknown"

    @abstractmethod
    def sample(self, shape, device):
        """返回位于 ``device`` 上、形状为 ``shape`` 的噪声张量。"""


class GaussianNoiseSampler(NoiseSampler):
    """标准高斯噪声。"""

    name = "gaussian"

    def sample(self, shape, device):
        return torch.randn(*shape, device=device)


class UniformNoiseSampler(NoiseSampler):
    """[-1, 1] 均匀噪声。"""

    name = "uniform"

    def sample(self, shape, device):
        return torch.rand(*shape, device=device).sub_(0.5).mul_(2.0)


def build_noise_sampler(noise_type):
    """把历史 ``noise_type`` 配置解析为 ``NoiseSampler`` 实例。"""
    if isinstance(noise_type, NoiseSampler):
        return noise_type
    if noise_type == "gaussian":
        return GaussianNoiseSampler()
    if noise_type == "uniform":
        return UniformNoiseSampler()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


def apply_rollout_coefs_override(base, override_dict):
    """Phase 3 #16:从 ``override_dict`` (例如 CLI 解析的 JSON) 中挑选 ``base``
    ``RolloutQueueCoefs`` 上存在的字段做 ``dataclasses.replace`` 合并。

    任何 ``override_dict`` 中不属于 dataclass 字段的 key 会被静默忽略,
    避免拼写错误把超参直接丢掉又不报错; 类型不匹配时同样返回 ``base``,
    避免一个 CLI 错参让整个训练启动失败。
    """
    import dataclasses

    if not override_dict:
        return base, ()
    valid_fields = {
        f.name: f.type for f in dataclasses.fields(base)
    }
    cleaned = {}
    invalid_keys = []
    for key, value in override_dict.items():
        if key not in valid_fields:
            continue
        expected_type = valid_fields[key]
        try:
            cleaned[key] = expected_type(value)
        except (TypeError, ValueError):
            invalid_keys.append(key)
    if not cleaned:
        return base, tuple(invalid_keys)
    try:
        return dataclasses.replace(base, **cleaned), tuple(invalid_keys)
    except (TypeError, ValueError):
        return base, tuple(invalid_keys or cleaned.keys())


def get_noise(shape, noise_type, device):
    """生成噪声，用于提升未来轨迹采样的多样性。

    Args:
        shape: 噪声张量形状，通常是 `(场景数, 噪声维度)`。
            这里的“场景数”对应 `seq_start_end` 的分组数，而不是 batch 中
            的 agent 数。这样做的含义是：同一个场景内的所有车辆共享同一份
            场景级随机扰动，从而让多模态差异更多体现在“整段交互未来”上。
        noise_type: 噪声分布类型，支持 `"gaussian"` 和 `"uniform"`。
        device: 噪声张量所在设备。

    Returns:
        位于指定设备上的随机噪声张量。
    """
    sampler = build_noise_sampler(noise_type)
    return sampler.sample(shape, device)


def get_module_device(module):
    """获取模块当前所在设备。"""
    return next(module.parameters()).device


class SafeInstanceNorm1d(nn.Module):
    """兼容短序列输入的 InstanceNorm1d。

    PyTorch 新版本在 `InstanceNorm1d` 的空间维长度为 1 时会直接报错，而本项目的
    `seqGAT` 在局部时间窗长度为 1 的阶段是合法且常见的。这里在退化到单元素窗口时
    直接跳过归一化，保留原始特征。
    """

    def __init__(self, num_features):
        super(SafeInstanceNorm1d, self).__init__()
        self.norm = nn.InstanceNorm1d(num_features)

    def forward(self, x):
        if x.dim() == 3 and x.size(-1) <= 1:
            return x
        return self.norm(x)


class BatchMultiHeadGraphAttention(nn.Module):
    """带关系矩阵约束的多头图注意力层。

    论文里的“discontinuous dependency”就体现在这里：不是所有节点对都参与注意力，
    而是先通过关系矩阵筛选出真正有空间和方向依赖的邻居，再做注意力聚合。
    """
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, h, Relation):
        """对每个节点在关系约束下做多头注意力聚合。

        Args:
            h: 节点特征，形状为 `(batch_like, num_nodes, f_in)`。
                这里的 `batch_like` 在本项目里通常对应时间帧数，因为是“每一帧
                的所有车辆”一起做空间图建模。
            Relation: 关系矩阵，形状为 `(batch_like, num_nodes, num_nodes)`。
                `Relation[i, u, v] = 1` 表示第 `i` 个样本里，节点 `u` 可以
                从节点 `v` 接收信息；为 0 则表示这条边被论文里的不连续依赖规则
                直接裁掉。

        Returns:
            output: 聚合后的节点特征，形状约为
                `(batch_like, n_head, num_nodes, f_out)`。
            attn: 每个头上的注意力权重。
        """
        bs, n = h.size()[:2]
        # 先把输入特征映射到每个注意力头自己的特征空间。
        h_prime = torch.matmul(h.unsqueeze(1), self.w)
        attn_src = torch.matmul(h_prime, self.a_src)
        attn_dst = torch.matmul(h_prime, self.a_dst)
        # 源节点和目标节点的注意力分数相加，得到完整的边权重。
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )
        attn = self.leaky_relu(attn)
        # 关系矩阵把无关节点对直接置零，只保留可交互边。
        relation = Relation.to(h.device)
        attn = torch.mul(relation.unsqueeze(1).repeat(1, self.n_head, 1, 1), attn)
        attn = self.softmax(attn)

        attn = self.dropout(attn)
        # 对邻居特征加权求和，得到交互后的节点表示。
        output = torch.matmul(attn, h_prime)
        if self.bias is not None:
            return output + self.bias, attn
        else:
            return output, attn

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.n_head)
            + " -> "
            + str(self.f_in)
            + " -> "
            + str(self.f_out)
            + ")"
        )

class seqBatchMultiHeadGraphAttention(nn.Module):
    """序列版多头图注意力层。

    与上面的空间图注意力不同，这里不再使用关系矩阵约束，而是让时间窗口内的
    特征彼此交互，用来增强局部时间范围内的动态建模能力。
    """
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(seqBatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, h):
        """在时间窗口内做图注意力聚合。

        Args:
            h: 时间窗口内的图特征，形状为 `(batch_like, window_len, f_in)`。
                在本项目里，`batch_like` 往往对应单个场景中的 agent 数，`window_len`
                对应局部时间窗口长度。

        Returns:
            output: 局部时序聚合后的特征。
            attn: 时间窗口内部的注意力分数。
        """
        bs, n = h.size()[:2]
        # 与空间版一致，只是这里不额外乘关系矩阵。
        h_prime = torch.matmul(h.unsqueeze(1), self.w)
        attn_src = torch.matmul(h_prime, self.a_src)
        attn_dst = torch.matmul(h_prime, self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )
        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        output = torch.matmul(attn, h_prime)

        if self.bias is not None:
            return output + self.bias, attn
        else:
            return output, attn

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.n_head)
            + " -> "
            + str(self.f_in)
            + " -> "
            + str(self.f_out)
            + ")"
        )

class GAT(nn.Module):
    """空间图注意力编码器。

    它接收每一帧的轨迹隐藏状态，再结合关系矩阵，把当前场景里有依赖的车辆
    互相传递信息，得到更强的交互特征。
    """
    def __init__(self, n_units, n_heads, dropout=0.2, alpha=0.2):
        super(GAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.layer_stack = nn.ModuleList()

        for i in range(self.n_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                BatchMultiHeadGraphAttention(
                    n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=dropout))

        self.norm_list = nn.ModuleList(
            [SafeInstanceNorm1d(32), SafeInstanceNorm1d(64)]
        )

    def forward(self, x, Relation):
        """逐层堆叠图注意力，输出最终的空间交互表征。

        Args:
            x: 输入节点特征，形状 `(batch_like, num_nodes, feat_dim)`。
            Relation: 图结构约束矩阵，形状 `(batch_like, num_nodes, num_nodes)`。

        Returns:
            空间交互增强后的节点表示。最后一层会把 head 维压缩掉，输出形状
            接近 `(batch_like, num_nodes, out_dim)`。
        """
        bs, n = x.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            # InstanceNorm1d 稍微稳定不同通道的尺度。
            x = self.norm_list[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            x, attn = gat_layer(x, Relation)
            if i + 1 == self.n_layer:
                # 最后一层的 head 维通常已经压成 1。
                x = x.squeeze(dim=1)
            else:
                # 中间层把多头输出拼接回节点特征。
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        else:
            return x

class seqGAT(nn.Module):
    """序列图注意力编码器。

    论文里它承担的是短时间窗口内的交互补充建模，相当于把空间图特征再过一遍
    时序视角的聚合，让模型更容易捕捉“当前交互如何变化”。
    """
    def __init__(self, n_units, n_heads, dropout=0.2, alpha=0.2):
        super(seqGAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.layer_stack = nn.ModuleList()

        for i in range(self.n_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                seqBatchMultiHeadGraphAttention(
                    n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=dropout))

        self.norm_list = nn.ModuleList(
            [SafeInstanceNorm1d(32), SafeInstanceNorm1d(64)]
        )

    def forward(self, x):
        """对时间窗口内的图特征进一步聚合。

        Args:
            x: 局部窗口特征，形状 `(batch_like, window_len, feat_dim)`。

        Returns:
            经过时序图注意力增强后的窗口表示。
        """
        bs, n = x.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = self.norm_list[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            x, attn = gat_layer(x)
            if i + 1 == self.n_layer:
                x = x.squeeze(dim=1)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        else:
            return x


class GATEncoder(nn.Module):
    """空间图编码器。

    这个模块把轨迹 LSTM 输出的隐藏状态转成带交互信息的图特征。
    关键函数 `relation_Matrix` 根据车辆方向和距离构图，这就是论文里的核心设计。
    """
    def __init__(self, n_units, n_heads, dropout, alpha):
        super(GATEncoder, self).__init__()
        self.gat_net = GAT(n_units, n_heads, dropout, alpha)

    def neig_direction(self, diffx, diffy):
        """计算相邻两目标之间的方向角。

        Args:
            diffx: 邻居相对当前目标的 x 方向位移。
            diffy: 邻居相对当前目标的 y 方向位移。

        Returns:
            角度值，范围在 `[0, 360)`，用于判断邻居是否落在目标车辆的前向扇区内。
        """
        if diffx != 0:
            dire = 180 * math.atan2(diffy, diffx) / (math.pi)
            if dire < 0:
                dire = 360 + dire
        else:
            if diffy > 0:
                dire = 90
            elif diffy < 0:
                dire = 270
            else:
                dire = 0
        return dire

    def relation_Matrix(self, curr_dire):
        """根据距离和方向约束构造关系矩阵(向量化版本)。

        逻辑上等价于：
        1. 先筛掉太远的邻居；
        2. 再看邻居是否落在当前目标运动方向前方的扇区内。

        Args:
            curr_dire: 当前时间序列的方向相关输入，形状 `(F, N, D)`。
                其中 `F` 是帧数，`N` 是场景内 agent 数，当前实现实际会使用：
                - `[:, :, 2:4]` 作为位置坐标；
                - `[:, :, 5]` 作为朝向角。

        Returns:
            关系矩阵 `r`，形状 `(F, N, N)`。若 `r[f, i, j] = 1`，表示在第 `f`
            帧中，agent `j` 会被视为 agent `i` 的有效邻居。

        Phase 5 #9 修复:方向扇区 Bug 在 commit bc47e72 中已修复,但原实现
        仍为三层 Python 嵌套循环 + numpy ``pdist``,在 batch 较大时瓶颈
        明显。改为:

        1. 距离门控用 ``(pos.unsqueeze(2) - pos.unsqueeze(1))`` 一次性算出
        2. 方向角 pairwise 用 ``torch.atan2`` 广播算出
        3. 扇区判定统一为 ``(delta <= 62) | (delta >= 298)``,
           其中 ``delta = (dire - a + 360) % 360``,统一处理 wrap-around
           (原版的 ``up > 360`` / ``62 <= up <= 124`` 分支等价于此)

        结果与原实现逐元素 ``torch.allclose``,但消除了
        ``F * N * N`` 量级的 Python 循环开销。
        """
        currdata = curr_dire[:, :, 2:4]  # (F, N, 2)
        F, N, _ = currdata.shape
        # 156 是论文实现中使用的邻域半径阈值。
        l = 156

        # 1. 距离门控
        # diff_pos[f, i, j, :] = currdata[f, j] - currdata[f, i],即
        # 从 agent i 指向邻居 j 的位移(与原 ``neig_direction`` 的
        # ``currdata[n_neig] - currdata[cur_n]`` 语义一致)
        diff_pos = currdata.unsqueeze(1) - currdata.unsqueeze(2)  # (F, N, N, 2)
        d = torch.sqrt((diff_pos ** 2).sum(dim=-1))  # (F, N, N)
        d_gate = (d <= l).float()

        # 2. pairwise 方向角 (F, N, N)
        diff_x = diff_pos[..., 0]
        diff_y = diff_pos[..., 1]
        # atan2 返回 [-pi, pi]; 转成 [0, 360)
        dire = torch.atan2(diff_y, diff_x) * (180.0 / math.pi)  # (F, N, N)
        dire = dire % 360

        # 3. 扇区判定:统一为 ``delta ∈ [0, 62] ∪ [298, 360)``
        # a = curr_dire[:, :, 5] 是每个 agent 的朝向角, 形状 (F, N)
        a = curr_dire[:, :, 5]
        delta = (dire - a.unsqueeze(2) + 360.0) % 360  # (F, N, N)
        in_sector = ((delta <= 62) | (delta >= 298)).float()

        r = d_gate * in_sector
        return r


    def forward(self, obs_traj_embedding, seq_start_end, obs_dire):
        """按场景分组做图编码，避免不同场景之间互相串扰。

        Args:
            obs_traj_embedding: 轨迹编码特征，形状 `(obs_len, batch, hidden_dim)`。
                这里的 batch 实际是把一个 mini-batch 中所有场景内的 agent 拉平后的
                总数，所以必须依赖 `seq_start_end` 才能恢复场景边界。
            seq_start_end: 每个场景在展平 batch 中的起止索引，形状 `(num_scene, 2)`。
            obs_dire: 构图所需的方向/位置信息，形状 `(obs_len, batch, feat_dim)`。

        Returns:
            graph_embeded_data: 带有空间交互信息的特征，形状仍与输入主维度对齐，
            即 `(obs_len, batch, graph_dim)`。
        """
        graph_embeded_data = []

        for start, end in seq_start_end.tolist():
            curr_seq_embedding_traj = obs_traj_embedding[:, start:end, :]
            curr_obs_dire = obs_dire[:, start:end, :]
            Relation = self.relation_Matrix(curr_obs_dire)
            curr_seq_graph_embedding = self.gat_net(curr_seq_embedding_traj,Relation)
            graph_embeded_data.append(curr_seq_graph_embedding)
        graph_embeded_data = torch.cat(graph_embeded_data, dim=1)
        return graph_embeded_data

class seqGATEncoder(nn.Module):
    """序列图编码器。

    它对空间图特征在时间维上再做一次局部聚合，增强短时间窗口内的交互动态。
    """
    def __init__(self, n_units, n_heads, dropout, alpha):
        super(seqGATEncoder, self).__init__()
        self.seq_gat_net = seqGAT(n_units, n_heads, dropout, alpha)

    def forward(self, obs_traj_embedding, seq_start_end):
        """按场景分组，对局部时间窗内的图特征继续编码。

        Args:
            obs_traj_embedding: 形状 `(batch_like, time_window, feat_dim)` 的窗口特征。
            seq_start_end: 当前分组索引，通常用于表示“这个局部窗口只属于一个场景”。

        Returns:
            graph_embeded_data: 时序交互增强后的特征。
        """
        graph_embeded_data = []
        for start, end in seq_start_end.tolist():
            curr_seq_embedding_traj = obs_traj_embedding[:, start:end, :]
            curr_seq_graph_embedding = self.seq_gat_net(curr_seq_embedding_traj)
            graph_embeded_data.append(curr_seq_graph_embedding)
        graph_embeded_data = torch.cat(graph_embeded_data, dim=1)
        return graph_embeded_data

class TrajectoryGenerator(nn.Module):
    """轨迹生成器。

    生成器是整个模型的主干，负责把观测轨迹、图交互和交通灯状态融合起来，
    最终输出未来 12 帧的相对位移。
    """
    def __init__(
        self,
        obs_len,
        pred_len,
        traj_lstm_input_size,   # 2
        traj_lstm_hidden_size,  # 32
        n_units,
        n_heads,
        graph_network_out_dims, # 64
        dropout,
        alpha,
        graph_lstm_hidden_size, #64
        noise_dim=(8,),
        noise_type="gaussian",
        light_input_size=5,
        embedding_size=64,
        light_embedding_size=32,
    ):
        super(TrajectoryGenerator, self).__init__()
        # 基础配置：
        # obs_len / pred_len 定义“看多少历史、预测多少未来”；
        # traj_lstm_hidden_size 定义个体运动编码容量；
        # graph_lstm_hidden_size 对应图交互分支最终要提供给解码器的维度；
        # light_embedding_size 则决定交通灯条件表征的紧凑程度。
        self.embedding_size = embedding_size
        self.light_input_size = light_input_size
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.light_embedding_size = light_embedding_size
        self.gatencoder = GATEncoder(
            n_units=n_units, n_heads=n_heads, dropout=dropout, alpha=alpha
        )
        self.seqgatencoder = seqGATEncoder(
            n_units=n_units, n_heads=n_heads, dropout=dropout, alpha=alpha
        )

        self.graph_lstm_hidden_size = graph_lstm_hidden_size
        self.traj_lstm_hidden_size = traj_lstm_hidden_size

        # 预测 LSTM 的初始上下文由三部分拼接而来，再加上噪声维度。
        self.pred_lstm_hidden_size = (
            self.light_embedding_size
            + self.traj_lstm_hidden_size
            + self.graph_lstm_hidden_size
            + noise_dim[0]
        )

        # 轨迹编码分支：逐帧吃相对位移。
        self.traj_lstm_model = nn.LSTMCell(traj_lstm_input_size, traj_lstm_hidden_size)
        # ⚠️ 保留但未使用(Phase 5 #11):``graph_lstm_model`` 是早期版本
        # 中用于在 GAT 之上再做时序聚合的 LSTMCell。重构后图时序聚合
        # 改由 ``seqGATEncoder`` 完成,本模块在 forward 主路径中**没有**
        # 被调用。出于"旧版实现可对照、checkpoint 兼容旧图"考虑保留
        # 该成员,但**不要**在 forward 中引入新的调用点;如需
        # 真正的 graph 时序聚合,请直接扩展 ``seqGATEncoder``。
        # 测试 ``test_graph_lstm_model_is_intentionally_unused`` 会
        # 在 forward 前后断言该模块从未被触发,防止意外回归。
        self.graph_lstm_model = nn.LSTMCell(
            graph_network_out_dims, graph_lstm_hidden_size
        )
        # 静态计数器,仅用于上面那条"未使用"断言;不是 forward 状态。
        # 任何对 ``graph_lstm_model`` 的调用都会把 ``_graph_lstm_call_count``
        # 加 1,从而被单元测试捕获。
        self._graph_lstm_call_count = 0
        self.graph_lstm_model.register_forward_hook(
            self._count_graph_lstm_call
        )

        # 交通灯状态嵌入：把距离和灯态映射到更紧凑的语义空间。
        self.light_embedding = nn.Sequential(
            nn.BatchNorm1d(self.light_input_size),
            nn.ReLU(),
            nn.Linear(self.light_input_size, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.light_embedding_size),
            nn.ReLU()
        )

        # 下面两个线性层在当前主流程中主要是保留接口和辅助投影。
        self.traj_hidden2pos = nn.Linear(self.traj_lstm_hidden_size + self.light_embedding_size, 2)
        self.traj_gat_hidden2pos = nn.Linear(
            self.light_embedding_size + self.traj_lstm_hidden_size + self.graph_lstm_hidden_size, 2
        )
        self.pred_hidden2pos = nn.Linear(self.light_embedding_size + self.pred_lstm_hidden_size, 2)

        self.noise_dim = noise_dim
        self.noise_sampler = build_noise_sampler(noise_type)
        self.noise_type = self.noise_sampler.name

        # 解码器：每一步输入上一时刻位移，输出新的隐状态。
        self.pred_lstm_model = nn.LSTMCell(traj_lstm_input_size, self.pred_lstm_hidden_size)

    def _count_graph_lstm_call(self, module, inputs, output):
        """记录 ``graph_lstm_model`` 被调用的次数。"""
        self._graph_lstm_call_count += 1

    def init_hidden_traj_lstm(self, batch):
        """初始化轨迹 LSTM 的隐状态。

        Args:
            batch: 当前 mini-batch 中展平后的 agent 总数。

        Returns:
            `(h_0, c_0)`，两者形状均为 `(batch, traj_lstm_hidden_size)`。
        """
        device = get_module_device(self)
        return (
            torch.zeros(batch, self.traj_lstm_hidden_size, device=device),
            torch.zeros(batch, self.traj_lstm_hidden_size, device=device),
        )

    def init_hidden_graph_lstm(self, batch):
        """初始化图 LSTM 的隐状态。

        Returns:
            `(h_0, c_0)`，形状均为 `(batch, graph_lstm_hidden_size)`。
        """
        device = get_module_device(self)
        return (
            torch.zeros(batch, self.graph_lstm_hidden_size, device=device),
            torch.zeros(batch, self.graph_lstm_hidden_size, device=device),
        )

    def init_hidden_light_lstm(self, batch):
        """初始化交通灯分支隐状态。

        当前生成器主流程没有显式使用独立的 light LSTM，但保留了初始化接口，
        方便和其它实验分支兼容。
        """
        device = get_module_device(self)
        return (
            torch.zeros(batch, self.traj_lstm_hidden_size, device=device),
            torch.zeros(batch, self.traj_lstm_hidden_size, device=device),
        )

    def add_noise(self, _input, seq_start_end):
        """按场景给编码特征拼接噪声。

        Args:
            _input: 不含噪声的条件特征，形状
                `(batch, light_embedding + traj_hidden + graph_hidden)`。
            seq_start_end: 每个场景在展平 batch 中的范围。

        Returns:
            decoder_h: 拼接噪声后的解码器初始隐藏状态，形状
                `(batch, light_embedding + traj_hidden + graph_hidden + noise_dim)`。

        Notes:
            这里的噪声不是“每辆车一份”，而是“每个场景一份”。同一场景内所有车辆
            复制同一个噪声向量，意味着模型采样的是“该场景未来整体演化模式”的不同
            可能性，而不是彼此独立的随机抖动。
        """
        noise_shape = (seq_start_end.size(0),) + self.noise_dim

        z_decoder = get_noise(noise_shape, self.noise_sampler, _input.device)
        expanded_noise = self.expand_scene_noise_to_batch(z_decoder, seq_start_end)
        return torch.cat([_input, expanded_noise], dim=1)

    def expand_scene_noise_to_batch(self, scene_noise, seq_start_end):
        """把 scene-level 噪声复制到 scene 内每个 agent。

        ``scene_noise`` 的第 0 维对应 ``seq_start_end`` 的 scene 维。输出张量
        形状为 ``(batch, noise_dim)``，可直接与 agent-level hidden state 对齐。
        """
        expanded = []
        for idx, (start, end) in enumerate(seq_start_end.tolist()):
            expanded.append(scene_noise[idx].view(1, -1).expand(end - start, -1))
        return torch.cat(expanded, dim=0)

    def inject_per_step_decoder_noise(
        self, pred_lstm_hidden, seq_start_end, noise_scale=0.1
    ):
        """Phase 1 #18:在每个解码步前向 decoder hidden 注入新噪声。"""
        if not self.noise_dim or self.noise_dim[0] <= 0:
            return pred_lstm_hidden
        step_noise = get_noise(
            (seq_start_end.size(0),) + self.noise_dim,
            self.noise_sampler,
            pred_lstm_hidden.device,
        )
        expanded_noise = self.expand_scene_noise_to_batch(step_noise, seq_start_end)
        noise_pad = torch.zeros_like(pred_lstm_hidden)
        noise_pad[:, -self.noise_dim[0] :] = expanded_noise
        return pred_lstm_hidden + noise_scale * noise_pad

    def get_last_state(self,obs_traj_pos,obs_state):
        """从最后一帧观测状态里构造交通灯条件特征。

        Args:
            obs_traj_pos: 观测期的轨迹相关特征，形状 `(obs_len, batch, feat_dim)`。
                当前实现会使用最后一帧中的 `[:, :, 2:4]` 作为车辆绝对位置。
            obs_state: 观测期交通灯状态，形状 `(obs_len, batch, state_dim)`。
                当前实现会使用：
                - `[:, :, 0:2]` 作为停止线/灯控参考点坐标；
                - `[:, :, 2:4]` 作为灯态相关离散或连续状态。

        Returns:
            state_last: 形状 `(batch, 5)` 的交通灯条件向量，依次为
                `[距离, x 方向相对位移, y 方向相对位移, 灯态1, 灯态2]`。
        """

        dis = torch.sqrt((obs_traj_pos[-1,:,2]-obs_state[-1,:,0])**2 + (obs_traj_pos[-1,:,3] - obs_state[-1,:,1])**2)
        disx = obs_traj_pos[-1, :, 2] - obs_state[-1,:,0]
        disy = obs_traj_pos[-1, :, 3] - obs_state[-1,:,1]
        light_state=obs_state[-1,:,2:4]
        dis_state=torch.stack([dis,disx,disy],dim=1)
        state_last=torch.cat((dis_state,light_state),dim=1)

        return state_last

    def get_next_state(self,pred_traj_rel,obs_traj_pos,pred_state):
        """用已经生成的相对轨迹递推出下一时刻交通灯条件。

        Args:
            pred_traj_rel: 已经生成的未来相对位移列表，每个元素形状 `(batch, 2)`。
            obs_traj_pos: 观测期轨迹特征，用最后一个真实位置作为积分起点。
            pred_state: 预测期对应的交通灯状态序列，形状 `(pred_len, batch, state_dim)`。

        Returns:
            state_last: 当前预测时刻对应的交通灯条件向量，格式与 `get_last_state`
            保持一致，供解码器下一步使用。
        """
        pred_traj_rel = torch.stack(pred_traj_rel)
        step = pred_traj_rel.size(0)

        # 把“累计预测的相对位移”还原为“当前一步的绝对坐标”，这样才能计算车辆
        # 相对于停止线或信号灯参考点的位置关系。
        start_pos = obs_traj_pos[-1, :, 2:4]
        real_pos = relative_to_abs(pred_traj_rel, start_pos)
        dis = torch.sqrt(
            (real_pos[-1, :, 0] - pred_state[-1, :, 0]) ** 2
            + (real_pos[-1, :, 1] - pred_state[-1, :, 1]) ** 2
        )
        disx = real_pos[-1, :, 0] - pred_state[-1, :, 0]
        disy = real_pos[-1, :, 1] - pred_state[-1, :, 1]
        dis_state = torch.stack([dis, disx, disy], dim=1)
        last_state = pred_state[step - 1, :, 2:4]
        state_last = torch.cat((dis_state, last_state), dim=1)

        return state_last


    def forward(
        self,
        obs_traj_rel,
        obs_traj_pos,
        obs_state,
        pred_state,
        seq_start_end,
        teacher_forcing_ratio=0.5,
        training_step=3,
    ):
        """完整生成流程。

        先编码历史轨迹和图交互，再把交通灯状态与噪声拼起来，最后自回归解码未来。

        Args:
            obs_traj_rel: 输入的相对轨迹特征，形状 `(obs_len + pred_len, batch, feat_dim)`。
                当前主流程实际只取其中 `2:4` 两个通道作为相对位移 `(dx, dy)`。
                训练阶段之所以长度包含未来段，是因为 teacher forcing 要从真实未来
                中取监督信号。
            obs_traj_pos: 输入的绝对/增强轨迹特征，形状 `(obs_len, batch, feat_dim)`，
                用于提取方向信息、当前位置以及和交通灯的相对关系。
            obs_state: 观测期交通灯状态序列。
            pred_state: 预测期交通灯状态序列，用于在解码过程中更新条件。
            seq_start_end: 场景分组索引，告诉模型“哪些 agent 属于同一场景”。
            teacher_forcing_ratio: 训练阶段使用真实未来位移作为下一步输入的概率。
            training_step: 旧实验接口保留参数，当前主流程未直接使用。

        Returns:
            outputs: 未来 `pred_len` 帧的相对位移预测，形状 `(pred_len, batch, 2)`。
        """
        batch = obs_traj_rel.shape[1]
        # traj_lstm_h_t / traj_lstm_c_t:
        # 个体运动编码器在当前时刻的隐藏状态与记忆状态。
        traj_lstm_h_t, traj_lstm_c_t = self.init_hidden_traj_lstm(batch)
        # pred_traj_rel: 按时间顺序缓存每一步解码出的未来相对位移。
        pred_traj_rel = []
        # traj_lstm_hidden_states: 保存每个观测时刻的个体运动编码。
        traj_lstm_hidden_states = []
        # graph_lstm_hidden_states: 保存每个观测时刻经过空间+局部时间交互后的图特征。
        graph_lstm_hidden_states = []

        # 1) 逐帧编码观测轨迹。
        for i, input_t in enumerate(
            obs_traj_rel[: self.obs_len].chunk(obs_traj_rel[: self.obs_len].size(0), dim=0)):
            # 输入张量原本可能带有额外辅助通道，这里只拿相对位移 `(dx, dy)`。
            inputtraj = input_t[:, :, 2:4]
            traj_lstm_h_t, traj_lstm_c_t = self.traj_lstm_model(
                inputtraj.squeeze(0), (traj_lstm_h_t, traj_lstm_c_t))
            traj_lstm_hidden_states += [traj_lstm_h_t]

        # 2) 用 GAT 建模车辆之间的空间交互。
        kl = 6
        # obs_dire 中保留构图所需的局部几何与朝向信息；第 5 个通道会被替换成
        # 更明确的方向角字段，以供 relation_Matrix 使用。
        obs_dire = obs_traj_pos[:, :, 0:6]
        obs_dire[:, :, 5] = obs_traj_pos[:, :, 9]
        graph_lstm_input = self.gatencoder(
            torch.stack(traj_lstm_hidden_states), seq_start_end, obs_dire
        )
        # staend 是给 seqGATEncoder 的“单场景局部窗口索引”，因为这里每次只对
        # 一段局部窗口做聚合，所以起点固定是 0。
        staend = torch.zeros((1, 2), dtype=torch.int, device=obs_traj_rel.device)

        # 3) 对局部时间窗口内的 GAT 输出再做一次序列聚合。
        for j in range(self.obs_len):
            if j <= kl:
                staend[0, 1] = j + 1
                graph_inter_input = self.seqgatencoder(graph_lstm_input[0:(j + 1)].permute(1, 0, 2), staend)
            else:
                staend[0, 1] = kl + 1
                graph_inter_input = self.seqgatencoder(graph_lstm_input[(j - kl):(j + 1)].permute(1, 0, 2),
                                                       staend)
            # 只取当前窗口最后一个时间位置的表示，视为“截至第 j 帧”的
            # 图交互上下文摘要。
            graph_lstm_hidden_states += [graph_inter_input[:, -1, :]]

        # 4) 取最后一帧的交通灯状态，与运动特征拼接。
        light_state = self.get_last_state(obs_traj_pos, obs_state)
        light_state_embedding = self.light_embedding(light_state)
        # encoded_before_noise_hidden 是解码器的条件核心，包含：
        # 1. 交通灯约束；
        # 2. 目标自身的最新运动状态；
        # 3. 经过图建模后的交互上下文。
        encoded_before_noise_hidden = torch.cat(
            (light_state_embedding, traj_lstm_hidden_states[-1], graph_lstm_hidden_states[-1]),
            dim=1)

        # 5) 场景级噪声注入，形成多模态解码起点。
        pred_lstm_hidden = self.add_noise(
            encoded_before_noise_hidden, seq_start_end
        )
        pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden)
        obs_traj_rel = obs_traj_rel[:, :, 2:4]
        # output 初始化为“最后一个观测位移”，作为未来第一步解码输入。
        output = obs_traj_rel[self.obs_len - 1]
        if self.training:
            # 训练阶段用 teacher forcing 稳定解码。
            for i, input_t in enumerate(
                    obs_traj_rel[-self.pred_len:].chunk(
                        obs_traj_rel[-self.pred_len:].size(0), dim=0
                    )  # 12帧
            ):
                # Phase 1 #18 fix: per-step noise injection during decoding.
                pred_lstm_hidden = self.inject_per_step_decoder_noise(
                    pred_lstm_hidden, seq_start_end
                )

                teacher_force = torch.rand(1, device=pred_lstm_hidden.device).item() < teacher_forcing_ratio
                input_t = input_t if teacher_force else output.unsqueeze(0)
                pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(
                    input_t.squeeze(0), (pred_lstm_hidden, pred_lstm_c_t)  # 136
                )
                if i == 0:
                    light_state = self.get_last_state(obs_traj_pos, obs_state)
                else:
                    light_state = self.get_next_state(pred_traj_rel, obs_traj_pos, pred_state)
                light_state_embedding = self.light_embedding(light_state)
                # 输出头把“当前解码隐藏状态”和“当前交通灯条件”再次融合，用于
                # 生成下一步位移。
                pred_input = torch.cat((light_state_embedding, pred_lstm_hidden), dim=1)
                output = self.pred_hidden2pos(pred_input)
                pred_traj_rel += [output]
            outputs = torch.stack(pred_traj_rel)
        else:
            # 推理阶段完全依赖自身预测，自回归滚动未来 12 帧。
            for i in range(self.pred_len):
                # Phase 1 #18 fix: per-step noise injection during decoding.
                pred_lstm_hidden = self.inject_per_step_decoder_noise(
                    pred_lstm_hidden, seq_start_end
                )

                pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(
                    output, (pred_lstm_hidden, pred_lstm_c_t)
                )
                if i == 0:
                    light_state = self.get_last_state(obs_traj_pos, obs_state)
                else:
                    light_state = self.get_next_state(pred_traj_rel, obs_traj_pos, pred_state)
                light_state_embedding = self.light_embedding(light_state)
                pred_input = torch.cat((light_state_embedding, pred_lstm_hidden), dim=1)
                output = self.pred_hidden2pos(pred_input)
                pred_traj_rel += [output]
            outputs = torch.stack(pred_traj_rel)

        return outputs


class CycleStateTrajectoryGenerator(TrajectoryGenerator):
    """CycleState v0 生成器。

    这个版本在 D2-TPred 的微观运动建模基础上，增加两条更贴近论文 idea 的状态支路：
    1. 车道级队列状态记忆：从“同车道车辆数量、等待比例、释放比例”等弱标签统计量中
       学习中观 queue-wave 表征。
    2. 信号周期状态记忆：从观测窗口内的相位与持续时间序列中学习宏观 cycle memory。

    这里先实现一个最小可训练版本，让“全周期交通状态”真正进入模型主干。后续如果要
    再往论文终版推进，可以继续把 queue-wave token、spillback、release-order 等更强
    的交通状态定义补进来。
    """

    def __init__(
        self,
        obs_len,
        pred_len,
        traj_lstm_input_size,
        traj_lstm_hidden_size,
        n_units,
        n_heads,
        graph_network_out_dims,
        dropout,
        alpha,
        graph_lstm_hidden_size,
        noise_dim=(8,),
        noise_type="gaussian",
        light_input_size=5,
        embedding_size=64,
        light_embedding_size=32,
        queue_lstm_hidden_size=32,
        cycle_lstm_hidden_size=16,
        queue_speed_threshold=3.0,
        queue_distance_threshold=156.0,
        queue_count_norm=10.0,
        queue_speed_norm=10.0,
        queue_distance_norm=500.0,
        cycle_time_norm=60.0,
        phase_duration_limits=(38.0, 47.0, 2.0),
        disable_state_gating=False,
        disable_queue_rollout=False,
        disable_lane_queue_anchor=False,
        disable_decoder_state_residual=False,
        disable_aux_losses=False,
        rollout_residual_scale=1.0,
        decoder_state_residual_scale=1.0,
        detach_rollout_state=False,
        rollout_queue_coefs=None,
    ):
        # Phase 4 #22: ``disable_aux_losses`` 是消融实验的统一主开关。
        # 当启用时，所有 CycleState 特有功能（state gating、queue rollout、
        # lane queue anchor、decoder state residual）一次性关闭，四个独立
        # disable 标志位被强制置为 True，使模型在行为上与 baseline 对齐。
        if disable_aux_losses:
            disable_state_gating = True
            disable_queue_rollout = True
            disable_lane_queue_anchor = True
            disable_decoder_state_residual = True
        self.disable_aux_losses = disable_aux_losses
        super(CycleStateTrajectoryGenerator, self).__init__(
            obs_len=obs_len,
            pred_len=pred_len,
            traj_lstm_input_size=traj_lstm_input_size,
            traj_lstm_hidden_size=traj_lstm_hidden_size,
            n_units=n_units,
            n_heads=n_heads,
            graph_network_out_dims=graph_network_out_dims,
            dropout=dropout,
            alpha=alpha,
            graph_lstm_hidden_size=graph_lstm_hidden_size,
            noise_dim=noise_dim,
            noise_type=noise_type,
            light_input_size=light_input_size,
            embedding_size=embedding_size,
            light_embedding_size=light_embedding_size,
        )
        # queue_feature = [
        #   前方排队车辆数, 同车道密度, 同车道平均速度, 同车道等待比例,
        #   同车道释放比例, 当前灯态编号, 当前灯态持续时间, 自身到停止线距离,
        #   车道排队长度, 停止线占用, 队首标记
        # ]
        self.queue_feature_dim = 11
        # cycle_feature = [
        #   phase one-hot(3), elapsed time, remaining time, phase change flag
        # ]
        self.cycle_feature_dim = 6
        self.queue_lstm_hidden_size = queue_lstm_hidden_size
        self.cycle_lstm_hidden_size = cycle_lstm_hidden_size
        self.queue_speed_threshold = queue_speed_threshold
        self.queue_distance_threshold = queue_distance_threshold
        self.queue_count_norm = queue_count_norm
        self.queue_speed_norm = queue_speed_norm
        self.queue_distance_norm = queue_distance_norm
        self.cycle_time_norm = cycle_time_norm
        self.register_buffer(
            "phase_duration_limits",
            torch.tensor(phase_duration_limits, dtype=torch.float32),
        )
        self.disable_state_gating = disable_state_gating
        self.disable_queue_rollout = disable_queue_rollout
        self.disable_lane_queue_anchor = disable_lane_queue_anchor
        self.disable_decoder_state_residual = disable_decoder_state_residual
        self.rollout_residual_scale = rollout_residual_scale
        self.decoder_state_residual_scale = decoder_state_residual_scale
        self.detach_rollout_state = detach_rollout_state
        # Phase 3 #16: 把 ``rollout_queue_features`` 内的硬编码物理系数集中到
        # ``RolloutQueueCoefs`` dataclass,默认 ``None`` 触发 dataclass 默认值,
        # 行为与原裸字面量完全一致 (向后兼容)。
        self.rollout_queue_coefs = (
            rollout_queue_coefs if rollout_queue_coefs is not None else RolloutQueueCoefs()
        )

        # Phase 4 #29: 数据集归一化参数持久化。
        # 这些参数在 __init__ 中作为普通 float 属性存在，不进入 state_dict，
        # 因此不会随 checkpoint 保存/恢复。norm_params/load_norm_params 提供
        # 显式的序列化/反序列化接口，确保断点续训和评估时使用一致的归一化量纲。
        self._norm_param_keys = (
            "queue_count_norm",
            "queue_speed_norm",
            "queue_distance_norm",
            "cycle_time_norm",
        )

        self.queue_feature_embedding = nn.Sequential(
            nn.Linear(self.queue_feature_dim, self.queue_lstm_hidden_size),
            nn.ReLU(),
            nn.Linear(self.queue_lstm_hidden_size, self.queue_lstm_hidden_size),
            nn.ReLU(),
        )
        self.queue_lstm_model = nn.LSTMCell(
            self.queue_lstm_hidden_size, self.queue_lstm_hidden_size
        )

        self.cycle_feature_embedding = nn.Sequential(
            nn.Linear(self.cycle_feature_dim, self.cycle_lstm_hidden_size),
            nn.ReLU(),
            nn.Linear(self.cycle_lstm_hidden_size, self.cycle_lstm_hidden_size),
            nn.ReLU(),
        )
        self.cycle_lstm_model = nn.LSTMCell(
            self.cycle_lstm_hidden_size, self.cycle_lstm_hidden_size
        )
        self.cycle_step_embedding = nn.Sequential(
            nn.Linear(self.cycle_feature_dim, self.cycle_lstm_hidden_size),
            nn.ReLU(),
            nn.Linear(self.cycle_lstm_hidden_size, self.cycle_lstm_hidden_size),
            nn.ReLU(),
        )
        # queue rollout:
        # 观测阶段学到的 queue memory 不应该在整个预测期保持静态不变，
        # 因为真实路口中的排队/释放波会随着相位推进持续演化。这里显式构造
        # 一个“预测阶段 queue memory 滚动更新器”，让中观状态在解码期继续前进。
        self.queue_rollout_feature_mlp = nn.Sequential(
            nn.Linear(
                self.queue_feature_dim + self.cycle_feature_dim + 2,
                self.queue_lstm_hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(self.queue_lstm_hidden_size, self.queue_lstm_hidden_size),
            nn.ReLU(),
        )
        self.queue_rollout_gate = nn.Sequential(
            nn.Linear(
                self.light_embedding_size
                + self.queue_lstm_hidden_size
                + self.cycle_lstm_hidden_size,
                self.queue_lstm_hidden_size,
            ),
            nn.Sigmoid(),
        )
        # lane-consensus 动态融合门：
        # 不同 phase 下，个体局部 rollout 与车道级中观共识的融合强度不应固定一致。
        self.lane_queue_anchor_gate = nn.Sequential(
            nn.Linear(
                self.queue_feature_dim
                + self.cycle_feature_dim
                + self.light_embedding_size,
                self.queue_feature_dim,
            ),
            nn.Sigmoid(),
        )
        # 显式辅助预测头：
        # 让 queue/cycle 分支不仅“存在”，还要对可解释的中观/宏观状态负责，
        # 比直接截取 hidden 向量前几维做监督更稳、更易解释。
        # Phase 0 #4 修复：拆分为独立子空间，避免回归和分类共享同一组参数。
        # - queue_aux_reg_head：4 维回归 (count/waiting/release/lane)
        # - queue_aux_cls_head：2 维二分类 (stop-line/front-of-queue)
        # - cycle_aux_phase_head：3 维相位分类
        # - cycle_aux_time_head：2 维 elapsed/remaining 回归
        # - cycle_aux_change_head：1 维相位切换二分类
        self.queue_aux_reg_head = nn.Linear(self.queue_lstm_hidden_size, 4)
        self.queue_aux_cls_head = nn.Linear(self.queue_lstm_hidden_size, 2)
        self.cycle_aux_phase_head = nn.Linear(self.cycle_lstm_hidden_size, 3)
        self.cycle_aux_time_head = nn.Linear(self.cycle_lstm_hidden_size, 2)
        self.cycle_aux_change_head = nn.Linear(self.cycle_lstm_hidden_size, 1)
        # 相位条件门控：
        # 同样的 queue/cycle 记忆，在红灯、绿灯、黄灯下的作用并不一致。
        # 这里用显式门控让状态记忆受当前灯态条件调制，而不是简单拼接。
        self.queue_context_gate = nn.Sequential(
            nn.Linear(
                self.light_embedding_size
                + self.queue_lstm_hidden_size
                + self.cycle_lstm_hidden_size,
                self.queue_lstm_hidden_size,
            ),
            nn.Sigmoid(),
        )
        self.rollout_decode_context_gate = nn.Sequential(
            nn.Linear(
                self.light_embedding_size
                + self.queue_lstm_hidden_size
                + self.queue_lstm_hidden_size,
                self.queue_lstm_hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(self.queue_lstm_hidden_size, self.queue_lstm_hidden_size),
            nn.Sigmoid(),
        )
        self.cycle_context_gate = nn.Sequential(
            nn.Linear(
                self.light_embedding_size
                + self.queue_lstm_hidden_size
                + self.cycle_lstm_hidden_size,
                self.cycle_lstm_hidden_size,
            ),
            nn.Sigmoid(),
        )
        self.decode_cycle_gate = nn.Sequential(
            nn.Linear(
                self.light_embedding_size + self.cycle_lstm_hidden_size,
                self.cycle_lstm_hidden_size,
            ),
            nn.Sigmoid(),
        )
        # baseline-compatible decoder state residual:
        # 不再把解码器主干直接改宽，而是保持原 D2-TPred 解码器形状不变，
        # 通过残差方式把 queue/cycle memory 注入进去。这样可以完整 warm-start
        # 原始解码器参数，把新状态分支的学习压力从“重学主干”降为“逐步调制主干”。
        self.decoder_state_context_dim = (
            self.light_embedding_size
            + self.queue_lstm_hidden_size
            + self.cycle_lstm_hidden_size
        )
        self.decoder_state_residual = nn.Sequential(
            nn.Linear(
                self.decoder_state_context_dim, self.pred_lstm_hidden_size
            ),
            nn.ReLU(),
            nn.Linear(self.pred_lstm_hidden_size, self.pred_lstm_hidden_size),
            nn.Tanh(),
        )
        self.decoder_state_gate = nn.Sequential(
            nn.Linear(
                self.decoder_state_context_dim, self.pred_lstm_hidden_size
            ),
            nn.ReLU(),
            nn.Linear(self.pred_lstm_hidden_size, self.pred_lstm_hidden_size),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.decoder_state_residual[2].weight)
        nn.init.zeros_(self.decoder_state_residual[2].bias)
        nn.init.zeros_(self.decoder_state_gate[2].weight)
        nn.init.constant_(self.decoder_state_gate[2].bias, -2.0)
        nn.init.zeros_(self.rollout_decode_context_gate[2].weight)
        nn.init.constant_(self.rollout_decode_context_gate[2].bias, -2.0)
        self.debug_last_aux = None

    def norm_params(self):
        """导出数据集归一化参数，用于写入 checkpoint。

        Returns:
            dict[str, float]: 四个归一化参数。
        """
        return {key: float(getattr(self, key)) for key in self._norm_param_keys}

    def load_norm_params(self, norm_dict):
        """从 checkpoint 恢复归一化参数。

        若 ``norm_dict`` 为 None 或缺少某个 key，该参数保持当前值不变，
        保证向后兼容旧版 checkpoint（旧 checkpoint 不包含此字段时不会出错）。

        Args:
            norm_dict: ``norm_params()`` 产出的 dict，或 ``None``。
        """
        if norm_dict is None:
            return
        for key in self._norm_param_keys:
            if key in norm_dict:
                setattr(self, key, float(norm_dict[key]))
        logging.info(
            "CycleState norm params restored from checkpoint: %s",
            ", ".join(f"{k}={getattr(self, k):.2f}" for k in self._norm_param_keys),
        )

    def init_hidden_queue_lstm(self, batch):
        """初始化 queue memory 的隐状态。"""
        device = get_module_device(self)
        return (
            torch.zeros(batch, self.queue_lstm_hidden_size, device=device),
            torch.zeros(batch, self.queue_lstm_hidden_size, device=device),
        )

    def init_hidden_cycle_lstm(self, batch):
        """初始化 cycle memory 的隐状态。"""
        device = get_module_device(self)
        return (
            torch.zeros(batch, self.cycle_lstm_hidden_size, device=device),
            torch.zeros(batch, self.cycle_lstm_hidden_size, device=device),
        )

    def build_cycle_features(self, state_seq):
        """把观测到的信号状态序列转成 cycle memory 的输入特征。

        Args:
            state_seq: `(T, batch, 4)`，通道含义与数据集中的 `obs_state` 一致。

        Returns:
            cycle_feature: `(T, batch, 6)`，包含 one-hot 灯态、已持续时间、
            剩余时间以及灯态是否变化的标记。
        """
        phase = state_seq[:, :, 2].long().clamp(min=0, max=2)
        phase_one_hot = F.one_hot(phase, num_classes=3).float()
        elapsed_raw = state_seq[:, :, 3:4]
        elapsed = (elapsed_raw / self.cycle_time_norm).clamp(min=0.0, max=2.0)
        phase_limits = self.phase_duration_limits[phase].unsqueeze(2)
        remaining = ((phase_limits - elapsed_raw) / self.cycle_time_norm).clamp(
            min=0.0, max=2.0
        )
        phase_change = torch.zeros(
            state_seq.size(0), state_seq.size(1), 1, device=state_seq.device
        )
        phase_change[1:] = (phase[1:] != phase[:-1]).float().unsqueeze(2)
        return torch.cat((phase_one_hot, elapsed, remaining, phase_change), dim=2)

    def get_step_cycle_feature(self, state_frame, prev_phase=None):
        """构造单步解码阶段使用的周期状态特征。

        Args:
            state_frame: `(batch, 4)` 单帧信号状态。
            prev_phase: `(batch,)` 上一帧的 phase 索引；如果为 ``None`` 或未
                提供，则 ``phase_change`` 退化为全 0（保持向后兼容，调用方
                无 prev 信息时使用）。Phase 2 #6 修复：补齐预测期
                ``phase_change`` 跨帧比较，避免 cycle LSTM 输入丢失相位
                切换信号。"""
        phase = state_frame[:, 2].long().clamp(min=0, max=2)
        phase_one_hot = F.one_hot(phase, num_classes=3).float()
        elapsed_raw = state_frame[:, 3:4]
        elapsed = (elapsed_raw / self.cycle_time_norm).clamp(min=0.0, max=2.0)
        phase_limits = self.phase_duration_limits[phase].unsqueeze(1)
        remaining = ((phase_limits - elapsed_raw) / self.cycle_time_norm).clamp(
            min=0.0, max=2.0
        )
        if prev_phase is None:
            phase_change = torch.zeros(state_frame.size(0), 1, device=state_frame.device)
        else:
            prev_clamped = prev_phase.long().clamp(min=0, max=2)
            phase_change = (phase != prev_clamped).float().unsqueeze(1)
        return torch.cat((phase_one_hot, elapsed, remaining, phase_change), dim=1)

    def build_queue_features(self, obs_traj_pos, obs_traj_rel, obs_state, seq_start_end):
        """从观测窗口中提取车道级 queue-wave 弱标签特征。

        这里不追求一次性把交通工程细节全部做满，而是先用可稳定计算的统计量近似：
        - 前方排队车辆数
        - 同车道局部密度
        - 同车道平均速度
        - 同车道等待比例
        - 同车道释放比例
        - 当前灯态
        - 当前灯态持续时间
        - 自身到停止线距离

        这些量让中观状态不再只是“是否拥堵”的粗描述，而是更接近：
        - 队列有多长
        - 当前车是否靠近队首
        - 停止线附近是否已被占用
        - 车道是否处在释放波阶段
        """
        obs_len, batch = obs_traj_pos.size(0), obs_traj_pos.size(1)
        device = obs_traj_pos.device
        queue_features = torch.zeros(
            obs_len, batch, self.queue_feature_dim, device=device
        )

        speed = torch.norm(obs_traj_rel[:obs_len, :, 2:4], dim=2)
        stop_dist = torch.sqrt(
            (obs_traj_pos[:, :, 2] - obs_state[:, :, 0]) ** 2
            + (obs_traj_pos[:, :, 3] - obs_state[:, :, 1]) ** 2
        )
        phase_value = obs_state[:, :, 2] / 2.0
        elapsed_value = (obs_state[:, :, 3] / self.cycle_time_norm).clamp(
            min=0.0, max=2.0
        )

        for start, end in seq_start_end.tolist():
            for t in range(obs_len):
                lane_ids = obs_traj_pos[t, start:end, 4]
                scene_speed = speed[t, start:end]
                scene_stop_dist = stop_dist[t, start:end]

                same_lane = lane_ids.unsqueeze(0).eq(lane_ids.unsqueeze(1))
                lane_count = same_lane.float().sum(dim=1).clamp_min(1.0)
                waiting = (
                    (scene_speed < self.queue_speed_threshold)
                    & (scene_stop_dist < self.queue_distance_threshold)
                ).float()
                releasing = (
                    (scene_speed >= self.queue_speed_threshold)
                    & (scene_stop_dist < self.queue_distance_threshold)
                ).float()
                # 距离停止线越小，越可视为排在“前方”。
                ahead_mask = same_lane & (
                    scene_stop_dist.unsqueeze(0) < scene_stop_dist.unsqueeze(1)
                )

                queue_count = ahead_mask.float().sum(dim=1) / self.queue_count_norm
                lane_density = (lane_count - 1.0) / self.queue_count_norm
                # 观测期无法直接量测连续物理"长度"，这里用同车道等待车辆数
                # (归一化) 作为 lane_queue_length proxy，与 rollout 期第 8 维
                # 的语义槽位保持一致。
                lane_queue_length = (
                    torch.matmul(same_lane.float(), waiting.unsqueeze(1)).squeeze(1)
                    / self.queue_count_norm
                )
                lane_mean_speed = (
                    torch.matmul(same_lane.float(), scene_speed.unsqueeze(1)).squeeze(1)
                    / lane_count
                    / self.queue_speed_norm
                )
                lane_wait_ratio = (
                    torch.matmul(same_lane.float(), waiting.unsqueeze(1)).squeeze(1)
                    / lane_count
                )
                lane_release_ratio = (
                    torch.matmul(same_lane.float(), releasing.unsqueeze(1)).squeeze(1)
                    / lane_count
                )
                stopline_mask = (
                    scene_stop_dist < self.queue_distance_threshold
                ).float()
                lane_stopline_occupancy = (
                    torch.matmul(
                        same_lane.float(), stopline_mask.unsqueeze(1)
                    ).squeeze(1)
                    > 0
                ).float()
                front_of_queue = (
                    (ahead_mask.float().sum(dim=1) == 0)
                    & (scene_stop_dist < self.queue_distance_threshold)
                ).float()
                own_stop_dist = scene_stop_dist / self.queue_distance_norm

                queue_features[t, start:end, :] = torch.stack(
                    [
                        queue_count,
                        lane_density,
                        lane_mean_speed,
                        lane_wait_ratio,
                        lane_release_ratio,
                        phase_value[t, start:end],
                        elapsed_value[t, start:end],
                        own_stop_dist,
                        lane_queue_length,
                        lane_stopline_occupancy,
                        front_of_queue,
                    ],
                    dim=1,
                )

        return queue_features

    def compute_queue_targets(self, queue_feature_seq):
        """把中观 queue-wave 统计量转成更强的辅助监督目标。

        返回 ``(T, batch, 6)`` 维度严格按以下顺序拼接(Phase 2 #20
        契约,与 ``train.compute_structured_aux_losses`` 的
        ``queue_reg_idx=[0,1,2,3]`` / ``queue_cls_idx=[4,5]`` 切分对齐):

            [0] queue_count             (regression, MSE)
            [1] lane_wait_ratio         (regression, MSE)
            [2] lane_release_ratio      (regression, MSE)
            [3] lane_queue_length       (regression, MSE)
            [4] lane_stopline_occupancy (binary, BCE)
            [5] front_of_queue          (binary, BCE)

        任何重新排序必须同步修改
        :func:`build_queue_targets_signature` 与
        :func:`train.compute_structured_aux_losses` 的 idx 切分,
        否则 MSE/BCE 会被错误地分配到回归/分类子空间。
        """
        return torch.stack(
            (
                queue_feature_seq[:, :, 0],
                queue_feature_seq[:, :, 3],
                queue_feature_seq[:, :, 4],
                queue_feature_seq[:, :, 8],
                queue_feature_seq[:, :, 9],
                queue_feature_seq[:, :, 10],
            ),
            dim=2,
        )

    def build_lane_queue_anchor_seq(self, queue_feature_seq, lane_ids, seq_start_end):
        """构造同车道一致性中观锚点(向量化版本)。

        直觉上，中观 queue-wave 状态不应完全由单个 agent 的局部特征决定；
        同一条 lane 上的车辆应共享一个更平滑的"车道级状态共识"。

        Phase 5 #8 修复:原实现是 ``scene × time × unique_lane_id`` 三层
        Python 嵌套循环,在 batch 较大且每场景 unique lane 数多时瓶颈明显。
        改为:

        1. 一次性 ``repeat_interleave`` 算出 ``agent_scene_idx`` (batch,)
        2. 联合编码 ``(scene, lane) -> group_key`` 跨场景唯一
        3. ``index_add_`` 在 ``(T*batch)`` 维度同时累计 sum/count
        4. 求均值后用 ``mean_features[group_key]`` 广播回原 shape

        结果与原实现逐元素 ``torch.allclose``,但
        ``T * N * #unique_lane`` 时间复杂度降到 ``T * batch + num_groups`` 量级。
        """
        T, batch, dim = queue_feature_seq.shape
        device = queue_feature_seq.device
        dtype = queue_feature_seq.dtype

        # 1. agent -> scene 映射(无 Python 循环)
        num_scene = seq_start_end.size(0)
        scene_sizes = (seq_start_end[:, 1] - seq_start_end[:, 0]).to(device)
        agent_scene_idx = torch.repeat_interleave(
            torch.arange(num_scene, device=device, dtype=torch.long),
            scene_sizes,
        )

        # 2. (t, scene, lane) 联合编码,确保跨 t/跨 scene 的 lane id 不会撞 key
        # 注意:必须把 t 维度也编码进去,否则同 (scene, lane) 在不同时刻的
        # 特征会被错误平均,违反原 loop 的 ``for t in range(T)`` 语义
        # (原版每帧独立计算 lane 均值)。
        max_lane_id = (
            int(lane_ids.max().item()) + 1 if lane_ids.numel() > 0 else 1
        )
        # 防御 max_lane_id==0 (空 batch):保证 group_key 不退化
        max_lane_id = max(max_lane_id, 1)
        per_t_lane_offset = num_scene * max_lane_id
        t_offsets = (
            torch.arange(T, device=device, dtype=torch.long).unsqueeze(1)
            * per_t_lane_offset
        )
        group_key = (
            t_offsets
            + agent_scene_idx.unsqueeze(0) * max_lane_id
            + lane_ids.long()
        )  # (T, batch)
        num_groups = T * num_scene * max_lane_id

        # 3. 一次性 index_add_ 求和/计数
        flat_group_key = group_key.reshape(-1)  # (T*batch,)
        flat_features = queue_feature_seq.reshape(-1, dim)  # (T*batch, dim)

        sum_features = torch.zeros(num_groups, dim, device=device, dtype=dtype)
        sum_features.index_add_(0, flat_group_key, flat_features)

        count = torch.zeros(num_groups, device=device, dtype=dtype)
        count.index_add_(
            0,
            flat_group_key,
            torch.ones(T * batch, device=device, dtype=dtype),
        )

        # 4. 均值(空 group 不会进入广播路径,clamp 仅防御除零)
        mean_features = sum_features / count.clamp(min=1).unsqueeze(1)

        # 5. 广播回 (T, batch, dim)
        return mean_features[flat_group_key].reshape(T, batch, dim)

    def build_lane_queue_anchor(self, queue_feature, lane_ids, seq_start_end):
        """构造单步 lane-level meso anchor。"""
        return self.build_lane_queue_anchor_seq(
            queue_feature.unsqueeze(0),
            lane_ids.unsqueeze(0),
            seq_start_end,
        ).squeeze(0)

    def rollout_queue_features(
        self,
        prev_queue_feature,
        current_cycle_feature,
        last_pred_offset,
        step_index,
    ):
        """根据预测阶段的相位推进，显式滚动 queue-wave 特征。

        设计动机：
        观测阶段的 queue feature 只能描述“看到的最后一刻”，但在解码期，
        队列会继续随相位推进而积累或释放。这里用一个轻量、可学习但物理上可解释的
        rollout 近似，让队列记忆变成一个“相位演化过程”。
        """
        phase_one_hot = current_cycle_feature[:, :3]
        elapsed = current_cycle_feature[:, 3:4]
        remaining = current_cycle_feature[:, 4:5]
        phase_change = current_cycle_feature[:, 5:6]
        phase_id = phase_one_hot.argmax(dim=1)

        # Phase 3 #16: 把方法体内硬编码的物理系数集中到 ``self.rollout_queue_coefs``。
        # 默认值与原裸字面量完全一致, 行为不变; 但允许通过 ``__init__`` / CLI
        # 覆盖, 进而支持消融 / sensitivity grid / 阶段协议切换。
        coefs = self.rollout_queue_coefs

        rolled = prev_queue_feature.clone()
        waiting_ratio = rolled[:, 3]
        release_ratio = rolled[:, 4]
        phase_value = rolled[:, 5]
        elapsed_value = rolled[:, 6]
        stop_dist = rolled[:, 7]
        lane_queue_length = rolled[:, 8]
        stopline_occupancy = rolled[:, 9]
        front_of_queue = rolled[:, 10]

        is_red_like = (phase_id == 0).float()
        is_green_like = (phase_id == 1).float()
        is_yellow_like = (phase_id == 2).float()
        pred_speed = last_pred_offset.norm(dim=1).clamp(max=self.queue_speed_threshold)
        pred_speed_norm = pred_speed / max(self.queue_speed_threshold, 1e-6)
        progress = elapsed.squeeze(1).clamp(min=0.0, max=2.0) / 2.0
        remaining_progress = remaining.squeeze(1).clamp(min=0.0, max=2.0) / 2.0

        waiting_ratio = torch.clamp(
            waiting_ratio
            + coefs.waiting_ratio_red_inc * is_red_like * (1.0 - progress)
            + coefs.waiting_ratio_yellow_inc * is_yellow_like
            - coefs.waiting_ratio_green_dec * is_green_like * pred_speed_norm,
            min=0.0,
            max=coefs.waiting_ratio_max,
        )
        release_ratio = torch.clamp(
            release_ratio
            + coefs.release_ratio_green_inc * is_green_like * (1.0 - remaining_progress + pred_speed_norm)
            - coefs.release_ratio_red_dec * is_red_like
            - coefs.release_ratio_yellow_dec * is_yellow_like,
            min=0.0,
            max=coefs.release_ratio_max,
        )
        lane_queue_length = torch.clamp(
            lane_queue_length
            + coefs.lane_queue_length_red_inc * is_red_like
            + coefs.lane_queue_length_yellow_inc * is_yellow_like
            - coefs.lane_queue_length_green_dec * is_green_like * pred_speed_norm
            + coefs.lane_queue_length_phase_change_inc * phase_change.squeeze(1),
            min=0.0,
            max=coefs.lane_queue_length_max,
        )
        stopline_occupancy = torch.clamp(
            stopline_occupancy
            + coefs.stopline_occupancy_red_inc * is_red_like
            - coefs.stopline_occupancy_green_dec * is_green_like * pred_speed_norm,
            min=0.0,
            max=coefs.stopline_occupancy_max,
        )
        front_of_queue = torch.clamp(
            front_of_queue
            + coefs.front_of_queue_red_inc * is_red_like
            - coefs.front_of_queue_green_dec * is_green_like * pred_speed_norm,
            min=0.0,
            max=coefs.front_of_queue_max,
        )
        # 随着车辆预测位置向前推进，距离停止线逐步减小；相位切换会带来轻微不连续调整。
        step_discount = float(step_index + 1) / float(max(self.pred_len, 1))
        stop_dist = torch.clamp(
            stop_dist
            - coefs.stop_dist_pred_speed_dec * pred_speed_norm
            - coefs.stop_dist_step_discount_dec * step_discount
            + coefs.stop_dist_phase_change_inc * phase_change.squeeze(1),
            min=0.0,
            max=coefs.stop_dist_max,
        )
        queue_count = torch.clamp(
            waiting_ratio
            * (
                lane_queue_length
                + coefs.queue_count_stopline_weight * stopline_occupancy
            ),
            min=0.0,
            max=coefs.queue_count_max,
        )
        lane_density = torch.clamp(
            coefs.lane_density_prev_weight * rolled[:, 1]
            + coefs.lane_density_lane_queue_weight * lane_queue_length,
            min=0.0,
            max=coefs.lane_density_max,
        )
        lane_mean_speed = torch.clamp(
            coefs.lane_mean_speed_prev_weight * rolled[:, 2]
            + coefs.lane_mean_speed_pred_weight * (
                pred_speed / max(self.queue_speed_norm, 1e-6)
            ),
            min=0.0,
            max=coefs.lane_mean_speed_max,
        )
        phase_value = phase_id.float() / 2.0
        elapsed_value = elapsed.squeeze(1).clamp(min=0.0, max=2.0)

        rolled[:, 0] = queue_count
        rolled[:, 1] = lane_density
        rolled[:, 2] = lane_mean_speed
        rolled[:, 3] = waiting_ratio
        rolled[:, 4] = release_ratio
        rolled[:, 5] = phase_value
        rolled[:, 6] = elapsed_value
        rolled[:, 7] = stop_dist
        rolled[:, 8] = lane_queue_length
        rolled[:, 9] = stopline_occupancy
        rolled[:, 10] = front_of_queue
        return rolled

    def get_decode_step_context(
        self,
        step_index,
        pred_traj_rel,
        obs_traj_pos,
        obs_state,
        pred_state,
    ):
        """统一训练态/推理态的单步灯态与周期上下文构造。

        Phase 2 #6 修复：``prev_phase`` 显式从上一帧传入，
        让 ``phase_change`` 在预测期能反映真实的相位切换。
        对于 step 0，使用 obs_state 最后一帧作为 prev；对于 step > 0，
        使用 pred_state 上一步作为 prev。"""
        if step_index == 0:
            light_state = self.get_last_state(obs_traj_pos, obs_state)
            # 观测期最后一帧的相位作为 step 0 的 prev_phase。
            # obs_state 至少应有 1 帧；用最后一帧自身作 prev 时
            # phase_change 退化为 0，与 build_cycle_features 行为一致。
            prev_phase = obs_state[-1, :, 2]
            current_cycle_feature = self.get_step_cycle_feature(
                obs_state[-1], prev_phase=prev_phase
            )
        else:
            light_state = self.get_next_state(
                pred_traj_rel, obs_traj_pos, pred_state
            )
            # 预测期上一步的相位作为 prev_phase；
            # 跨帧 phase_change 由 get_step_cycle_feature 内部计算。
            prev_phase = pred_state[step_index - 2, :, 2]
            current_cycle_feature = self.get_step_cycle_feature(
                pred_state[step_index - 1], prev_phase=prev_phase
            )
        light_state_embedding = self.light_embedding(light_state)
        cycle_step_embedding = self.cycle_step_embedding(current_cycle_feature)
        return light_state_embedding, current_cycle_feature, cycle_step_embedding

    def rollout_cycle_step(
        self,
        current_cycle_feature,
        rollout_cycle_h_t,
        rollout_cycle_c_t,
    ):
        """在预测期滚动更新 cycle hidden/cell。

        G6 的最小方案不是再造一套显式 signal dynamics，而是让观测期已经学到的
        cycle LSTM 在预测期继续前进。这样 decoder 读到的是“滚动后的 cycle
        memory”，而不是每一步都重新从单帧 feature 投影出一个静态条件。
        """
        cycle_rollout_embed = self.cycle_feature_embedding(current_cycle_feature)
        rollout_cycle_h_t, rollout_cycle_c_t = self.cycle_lstm_model(
            cycle_rollout_embed, (rollout_cycle_h_t, rollout_cycle_c_t)
        )
        return {
            "cycle_hidden": rollout_cycle_h_t,
            "cycle_cell": rollout_cycle_c_t,
        }

    def rollout_queue_step(
        self,
        prev_queue_feature,
        lane_queue_anchor,
        lane_ids,
        seq_start_end,
        current_cycle_feature,
        last_pred_offset,
        step_index,
        light_state_embedding,
        cycle_step_embedding,
        rollout_queue_h_t,
        rollout_queue_c_t,
    ):
        """统一执行一轮预测期 meso rollout。"""
        current_queue_feature = self.rollout_queue_features(
            prev_queue_feature,
            current_cycle_feature,
            last_pred_offset,
            step_index,
        )
        used_lane_queue_anchor = None
        next_lane_queue_anchor = lane_queue_anchor
        if not self.disable_lane_queue_anchor:
            used_lane_queue_anchor = lane_queue_anchor
            lane_anchor_gate = self.lane_queue_anchor_gate(
                torch.cat(
                    (
                        current_queue_feature,
                        current_cycle_feature,
                        light_state_embedding,
                    ),
                    dim=1,
                )
            )
            current_queue_feature = (
                (1.0 - lane_anchor_gate) * current_queue_feature
                + lane_anchor_gate * lane_queue_anchor
            )
            next_lane_queue_anchor = self.build_lane_queue_anchor(
                current_queue_feature, lane_ids, seq_start_end
            )
        queue_rollout_input = torch.cat(
            (current_queue_feature, current_cycle_feature, last_pred_offset),
            dim=1,
        )
        queue_rollout_embed = self.queue_rollout_feature_mlp(queue_rollout_input)
        rollout_queue_h_t, rollout_queue_c_t = self.queue_lstm_model(
            queue_rollout_embed, (rollout_queue_h_t, rollout_queue_c_t)
        )
        if not self.disable_state_gating:
            rollout_queue_h_t = rollout_queue_h_t * self.queue_rollout_gate(
                torch.cat(
                    (
                        light_state_embedding,
                        rollout_queue_h_t,
                        cycle_step_embedding,
                    ),
                    dim=1,
                )
            )
        return {
            "queue_feature": current_queue_feature,
            "used_lane_queue_anchor": used_lane_queue_anchor,
            "next_lane_queue_anchor": next_lane_queue_anchor,
            "queue_hidden": rollout_queue_h_t,
            "queue_cell": rollout_queue_c_t,
            "queue_pred": torch.cat(
                (
                    self.queue_aux_reg_head(rollout_queue_h_t),
                    self.queue_aux_cls_head(rollout_queue_h_t),
                ),
                dim=-1,
            ),
            "queue_target": self.compute_queue_targets(
                current_queue_feature.unsqueeze(0)
            ).squeeze(0),
        }

    def build_traffic_context(
        self,
        obs_traj_rel,
        obs_traj_pos,
        obs_state,
        pred_state,
        seq_start_end,
    ):
        """把当前 tuple 风格输入统一组织成结构化 traffic context。

        这样做的目的不是立刻重写整个数据流，而是先给模型内部一个更清晰的
        统一接口，后续迁移到 INT2 时只需要替换 adapter，而不必重写核心模型。
        """
        queue_feature_seq = self.build_queue_features(
            obs_traj_pos, obs_traj_rel, obs_state, seq_start_end
        )
        cycle_feature_seq = self.build_cycle_features(obs_state)
        stopline_distance = torch.sqrt(
            (obs_traj_pos[:, :, 2] - obs_state[:, :, 0]) ** 2
            + (obs_traj_pos[:, :, 3] - obs_state[:, :, 1]) ** 2
        )
        lane_ids = obs_traj_pos[:, :, 4].long()
        phase_ids = obs_state[:, :, 2].long()
        phase_elapsed = obs_state[:, :, 3]
        pred_phase_ids = pred_state[:, :, 2].long()
        pred_phase_elapsed = pred_state[:, :, 3]
        lane_queue_anchor_seq = self.build_lane_queue_anchor_seq(
            queue_feature_seq, lane_ids, seq_start_end
        )

        traffic_context = {
            "agent": {
                "obs_traj": obs_traj_pos,
                "obs_traj_rel": obs_traj_rel,
                "lane_ids": lane_ids,
                "direction": obs_traj_pos[:, :, 9],
                "stopline_distance": stopline_distance,
            },
            "signal": {
                "obs_state": obs_state,
                "pred_state": pred_state,
                "phase_ids": phase_ids,
                "phase_elapsed": phase_elapsed,
                "pred_phase_ids": pred_phase_ids,
                "pred_phase_elapsed": pred_phase_elapsed,
                "cycle_feature_seq": cycle_feature_seq,
            },
            "scene": {
                "seq_start_end": seq_start_end,
            },
            "meso": {
                "queue_feature_seq": queue_feature_seq,
                "lane_queue_anchor_seq": lane_queue_anchor_seq,
                "queue_targets": self.compute_queue_targets(queue_feature_seq),
            },
        }
        return traffic_context

    def build_decoder_state_residual(
        self,
        light_state_embedding,
        queue_context,
        cycle_context,
    ):
        """把交通状态记忆映射成与 baseline 解码器同维的残差调制量。"""
        if self.disable_decoder_state_residual:
            return None
        decoder_state_context = torch.cat(
            (light_state_embedding, queue_context, cycle_context), dim=1
        )
        state_residual = self.decoder_state_residual(decoder_state_context)
        state_gate = self.decoder_state_gate(decoder_state_context)
        return self.decoder_state_residual_scale * state_gate * state_residual

    def build_rollout_decode_queue_context(
        self,
        observed_queue_context,
        rollout_queue_context,
        light_state_embedding,
    ):
        """让 rollout queue context 以锚定残差方式进入 decoder。"""
        rollout_delta = rollout_queue_context - observed_queue_context
        rollout_delta = torch.tanh(rollout_delta)
        rollout_gate = self.rollout_decode_context_gate(
            torch.cat(
                (
                    light_state_embedding,
                    observed_queue_context,
                    rollout_queue_context,
                ),
                dim=1,
            )
        )
        return (
            observed_queue_context
            + self.rollout_residual_scale * rollout_gate * rollout_delta
        )

    def maybe_detach_rollout_state(
        self,
        rollout_queue_feature,
        rollout_lane_queue_anchor,
        rollout_queue_h_t,
        rollout_queue_c_t,
    ):
        """warmup 阶段截断预测期 meso rollout 跨步反传。"""
        if not self.detach_rollout_state or not self.training:
            return (
                rollout_queue_feature,
                rollout_lane_queue_anchor,
                rollout_queue_h_t,
                rollout_queue_c_t,
            )
        return (
            rollout_queue_feature.detach(),
            rollout_lane_queue_anchor.detach(),
            rollout_queue_h_t.detach(),
            rollout_queue_c_t.detach(),
        )

    def maybe_detach_cycle_rollout_state(
        self,
        rollout_cycle_h_t,
        rollout_cycle_c_t,
    ):
        """warmup 阶段按需截断预测期 macro rollout 跨步反传。"""
        if not self.detach_rollout_state or not self.training:
            return rollout_cycle_h_t, rollout_cycle_c_t
        return rollout_cycle_h_t.detach(), rollout_cycle_c_t.detach()

    def forward(
        self,
        obs_traj_rel,
        obs_traj_pos,
        obs_state,
        pred_state,
        seq_start_end,
        teacher_forcing_ratio=0.5,
        training_step=3,
        traffic_context=None,
    ):
        """CycleState v0 的完整生成流程。"""
        batch = obs_traj_rel.shape[1]
        traj_lstm_h_t, traj_lstm_c_t = self.init_hidden_traj_lstm(batch)
        queue_lstm_h_t, queue_lstm_c_t = self.init_hidden_queue_lstm(batch)
        cycle_lstm_h_t, cycle_lstm_c_t = self.init_hidden_cycle_lstm(batch)

        pred_traj_rel = []
        traj_lstm_hidden_states = []
        graph_lstm_hidden_states = []
        queue_lstm_hidden_states = []
        cycle_lstm_hidden_states = []

        if traffic_context is None:
            traffic_context = self.build_traffic_context(
                obs_traj_rel, obs_traj_pos, obs_state, pred_state, seq_start_end
            )
        queue_feature_seq = traffic_context["meso"]["queue_feature_seq"]
        lane_queue_anchor_seq = traffic_context["meso"]["lane_queue_anchor_seq"]
        cycle_feature_seq = traffic_context["signal"]["cycle_feature_seq"]
        self.debug_last_aux = {
            "queue_feature_seq": queue_feature_seq.detach(),
            "lane_queue_anchor_seq": lane_queue_anchor_seq.detach(),
            "queue_targets": traffic_context["meso"]["queue_targets"].detach(),
            "cycle_feature_seq": cycle_feature_seq.detach(),
            "queue_hidden_last": None,
            "cycle_hidden_last": None,
            "queue_pred_last": None,
            "cycle_pred_last": None,
            "queue_rollout_hidden_seq": None,
            "queue_rollout_feature_seq": None,
            "queue_rollout_pred_seq": None,
            "queue_rollout_target_seq": None,
            "cycle_rollout_hidden_seq": None,
            "cycle_decode_context_seq": None,
            "lane_queue_rollout_anchor_seq": None,
            "decoder_state_init_residual": None,
            "decoder_state_init_residual_norm": None,
            "decoder_state_step_residual_seq": None,
            "decoder_state_step_residual_norm": None,
            "traffic_context": traffic_context,
        }

        for input_t in obs_traj_rel[: self.obs_len].chunk(self.obs_len, dim=0):
            inputtraj = input_t[:, :, 2:4]
            traj_lstm_h_t, traj_lstm_c_t = self.traj_lstm_model(
                inputtraj.squeeze(0), (traj_lstm_h_t, traj_lstm_c_t)
            )
            traj_lstm_hidden_states += [traj_lstm_h_t]

        obs_dire = obs_traj_pos[:, :, 0:6]
        obs_dire[:, :, 5] = obs_traj_pos[:, :, 9]
        graph_lstm_input = self.gatencoder(
            torch.stack(traj_lstm_hidden_states), seq_start_end, obs_dire
        )
        staend = torch.zeros((1, 2), dtype=torch.int, device=obs_traj_rel.device)
        for j in range(self.obs_len):
            if j <= 6:
                staend[0, 1] = j + 1
                graph_inter_input = self.seqgatencoder(
                    graph_lstm_input[0 : (j + 1)].permute(1, 0, 2), staend
                )
            else:
                staend[0, 1] = 7
                graph_inter_input = self.seqgatencoder(
                    graph_lstm_input[(j - 6) : (j + 1)].permute(1, 0, 2),
                    staend,
                )
            graph_lstm_hidden_states += [graph_inter_input[:, -1, :]]

        for t in range(self.obs_len):
            queue_embed = self.queue_feature_embedding(queue_feature_seq[t])
            queue_lstm_h_t, queue_lstm_c_t = self.queue_lstm_model(
                queue_embed, (queue_lstm_h_t, queue_lstm_c_t)
            )
            queue_lstm_hidden_states += [queue_lstm_h_t]

            cycle_embed = self.cycle_feature_embedding(cycle_feature_seq[t])
            cycle_lstm_h_t, cycle_lstm_c_t = self.cycle_lstm_model(
                cycle_embed, (cycle_lstm_h_t, cycle_lstm_c_t)
            )
            cycle_lstm_hidden_states += [cycle_lstm_h_t]

        light_state = self.get_last_state(obs_traj_pos, obs_state)
        light_state_embedding = self.light_embedding(light_state)
        queue_last = queue_lstm_hidden_states[-1]
        cycle_last = cycle_lstm_hidden_states[-1]
        if self.disable_state_gating:
            gated_queue_last = queue_last
            gated_cycle_last = cycle_last
        else:
            phase_gate_input = torch.cat(
                (light_state_embedding, queue_last, cycle_last), dim=1
            )
            gated_queue_last = queue_last * self.queue_context_gate(phase_gate_input)
            gated_cycle_last = cycle_last * self.cycle_context_gate(phase_gate_input)
        encoded_before_noise_hidden = torch.cat(
            (
                light_state_embedding,
                traj_lstm_hidden_states[-1],
                graph_lstm_hidden_states[-1],
            ),
            dim=1,
        )
        self.debug_last_aux["queue_hidden_last"] = gated_queue_last
        self.debug_last_aux["cycle_hidden_last"] = gated_cycle_last
        # Phase 0 #4 修复：queue/cycle aux 头拆分为 reg/cls 子头，
        # 此处拼接为 [reg_part, cls_part] / [phase_part, time_part, change_part]
        # 与 train.py compute_structured_aux_losses 的维度切片顺序一致。
        self.debug_last_aux["queue_pred_last"] = torch.cat(
            (
                self.queue_aux_reg_head(gated_queue_last),
                self.queue_aux_cls_head(gated_queue_last),
            ),
            dim=-1,
        )
        self.debug_last_aux["cycle_pred_last"] = torch.cat(
            (
                self.cycle_aux_phase_head(gated_cycle_last),
                self.cycle_aux_time_head(gated_cycle_last),
                self.cycle_aux_change_head(gated_cycle_last),
            ),
            dim=-1,
        )
        pred_lstm_hidden = self.add_noise(encoded_before_noise_hidden, seq_start_end)
        init_state_residual = self.build_decoder_state_residual(
            light_state_embedding, gated_queue_last, gated_cycle_last
        )
        if init_state_residual is not None:
            pred_lstm_hidden = pred_lstm_hidden + init_state_residual
            self.debug_last_aux["decoder_state_init_residual"] = (
                init_state_residual.detach()
            )
            self.debug_last_aux["decoder_state_init_residual_norm"] = (
                init_state_residual.detach().norm(dim=1)
            )
        pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden)

        obs_traj_rel = obs_traj_rel[:, :, 2:4]
        output = obs_traj_rel[self.obs_len - 1]
        last_rollout_offset = output
        queue_rollout_hidden_seq = []
        queue_rollout_feature_seq = []
        queue_rollout_pred_seq = []
        queue_rollout_target_seq = []
        cycle_rollout_hidden_seq = []
        cycle_decode_context_seq = []
        lane_queue_rollout_anchor_seq = []
        queue_decode_context_seq = []
        decoder_state_residual_seq = []
        rollout_queue_h_t = gated_queue_last
        rollout_queue_c_t = torch.zeros_like(gated_queue_last)
        rollout_cycle_h_t = gated_cycle_last
        rollout_cycle_c_t = cycle_lstm_c_t
        rollout_queue_feature = queue_feature_seq[-1]
        rollout_lane_queue_anchor = lane_queue_anchor_seq[-1]
        rollout_lane_ids = traffic_context["agent"]["lane_ids"][-1]
        if self.training:
            for i, input_t in enumerate(
                obs_traj_rel[-self.pred_len :].chunk(self.pred_len, dim=0)
            ):
                # Phase 1 #18 fix: per-step noise injection during decoding.
                pred_lstm_hidden = self.inject_per_step_decoder_noise(
                    pred_lstm_hidden, seq_start_end
                )

                teacher_force = torch.rand(1, device=pred_lstm_hidden.device).item() < teacher_forcing_ratio
                input_t = input_t if teacher_force else output.unsqueeze(0)
                pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(
                    input_t.squeeze(0), (pred_lstm_hidden, pred_lstm_c_t)
                )
                (
                    light_state_embedding,
                    current_cycle_feature,
                    cycle_step_embedding,
                ) = self.get_decode_step_context(
                    i,
                    pred_traj_rel,
                    obs_traj_pos,
                    obs_state,
                    pred_state,
                )
                if not self.disable_aux_losses:
                    rollout_cycle_info = self.rollout_cycle_step(
                        current_cycle_feature,
                        rollout_cycle_h_t,
                        rollout_cycle_c_t,
                    )
                    rollout_cycle_h_t = rollout_cycle_info["cycle_hidden"]
                    rollout_cycle_c_t = rollout_cycle_info["cycle_cell"]
                    cycle_context_for_decode = rollout_cycle_h_t
                    if not self.disable_state_gating:
                        cycle_context_for_decode = cycle_context_for_decode * self.decode_cycle_gate(
                            torch.cat(
                                (light_state_embedding, cycle_context_for_decode),
                                dim=1,
                            )
                        )
                    cycle_rollout_hidden_seq.append(rollout_cycle_h_t)
                    cycle_decode_context_seq.append(cycle_context_for_decode)
                else:
                    cycle_context_for_decode = cycle_step_embedding
                if not self.disable_queue_rollout:
                    rollout_info = self.rollout_queue_step(
                        rollout_queue_feature,
                        rollout_lane_queue_anchor,
                        rollout_lane_ids,
                        seq_start_end,
                        current_cycle_feature,
                        last_rollout_offset,
                        i,
                        light_state_embedding,
                        cycle_step_embedding,
                        rollout_queue_h_t,
                        rollout_queue_c_t,
                    )
                    rollout_queue_feature = rollout_info["queue_feature"]
                    rollout_lane_queue_anchor = rollout_info["next_lane_queue_anchor"]
                    rollout_queue_h_t = rollout_info["queue_hidden"]
                    rollout_queue_c_t = rollout_info["queue_cell"]
                    if rollout_info["used_lane_queue_anchor"] is not None:
                        lane_queue_rollout_anchor_seq.append(
                            rollout_info["used_lane_queue_anchor"]
                        )
                    queue_rollout_feature_seq.append(rollout_queue_feature)
                    queue_rollout_hidden_seq.append(rollout_queue_h_t)
                    queue_rollout_pred_seq.append(rollout_info["queue_pred"])
                    queue_rollout_target_seq.append(rollout_info["queue_target"])
                    queue_context_for_decode = self.build_rollout_decode_queue_context(
                        gated_queue_last,
                        rollout_queue_h_t,
                        light_state_embedding,
                    )
                    queue_decode_context_seq.append(queue_context_for_decode)
                    (
                        rollout_queue_feature,
                        rollout_lane_queue_anchor,
                        rollout_queue_h_t,
                        rollout_queue_c_t,
                    ) = self.maybe_detach_rollout_state(
                        rollout_queue_feature,
                        rollout_lane_queue_anchor,
                        rollout_queue_h_t,
                        rollout_queue_c_t,
                    )
                    rollout_cycle_h_t, rollout_cycle_c_t = (
                        self.maybe_detach_cycle_rollout_state(
                            rollout_cycle_h_t,
                            rollout_cycle_c_t,
                        )
                    )
                else:
                    queue_context_for_decode = gated_queue_last
                step_state_residual = self.build_decoder_state_residual(
                    light_state_embedding,
                    queue_context_for_decode,
                    cycle_context_for_decode,
                )
                if step_state_residual is not None:
                    pred_lstm_hidden = pred_lstm_hidden + step_state_residual
                    decoder_state_residual_seq.append(step_state_residual)
                pred_input = torch.cat((light_state_embedding, pred_lstm_hidden), dim=1)
                output = self.pred_hidden2pos(pred_input)
                # #3 P0 fix: train/eval rollout offset consistency.
                # The eval branch (below) always seeds ``last_rollout_offset``
                # from the model's own ``output``. The training branch must
                # follow the same contract, otherwise the queue rollout
                # branch (rollout_queue_features) is fed ground-truth future
                # displacement 80% of the time during training and the
                # model's own (errored) prediction 100% of the time during
                # inference — a classic exposure-bias / train-eval shift.
                last_rollout_offset = output
                pred_traj_rel += [output]
        else:
            for i in range(self.pred_len):
                # Phase 1 #18 fix: per-step noise injection during decoding.
                pred_lstm_hidden = self.inject_per_step_decoder_noise(
                    pred_lstm_hidden, seq_start_end
                )

                pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(
                    output, (pred_lstm_hidden, pred_lstm_c_t)
                )
                (
                    light_state_embedding,
                    current_cycle_feature,
                    cycle_step_embedding,
                ) = self.get_decode_step_context(
                    i,
                    pred_traj_rel,
                    obs_traj_pos,
                    obs_state,
                    pred_state,
                )
                if not self.disable_aux_losses:
                    rollout_cycle_info = self.rollout_cycle_step(
                        current_cycle_feature,
                        rollout_cycle_h_t,
                        rollout_cycle_c_t,
                    )
                    rollout_cycle_h_t = rollout_cycle_info["cycle_hidden"]
                    rollout_cycle_c_t = rollout_cycle_info["cycle_cell"]
                    cycle_context_for_decode = rollout_cycle_h_t
                    if not self.disable_state_gating:
                        cycle_context_for_decode = cycle_context_for_decode * self.decode_cycle_gate(
                            torch.cat(
                                (light_state_embedding, cycle_context_for_decode),
                                dim=1,
                            )
                        )
                    cycle_rollout_hidden_seq.append(rollout_cycle_h_t)
                    cycle_decode_context_seq.append(cycle_context_for_decode)
                else:
                    cycle_context_for_decode = cycle_step_embedding
                if not self.disable_queue_rollout:
                    rollout_info = self.rollout_queue_step(
                        rollout_queue_feature,
                        rollout_lane_queue_anchor,
                        rollout_lane_ids,
                        seq_start_end,
                        current_cycle_feature,
                        output,
                        i,
                        light_state_embedding,
                        cycle_step_embedding,
                        rollout_queue_h_t,
                        rollout_queue_c_t,
                    )
                    rollout_queue_feature = rollout_info["queue_feature"]
                    rollout_lane_queue_anchor = rollout_info["next_lane_queue_anchor"]
                    rollout_queue_h_t = rollout_info["queue_hidden"]
                    rollout_queue_c_t = rollout_info["queue_cell"]
                    if rollout_info["used_lane_queue_anchor"] is not None:
                        lane_queue_rollout_anchor_seq.append(
                            rollout_info["used_lane_queue_anchor"]
                        )
                    queue_rollout_feature_seq.append(rollout_queue_feature)
                    queue_rollout_hidden_seq.append(rollout_queue_h_t)
                    queue_rollout_pred_seq.append(rollout_info["queue_pred"])
                    queue_rollout_target_seq.append(rollout_info["queue_target"])
                    queue_context_for_decode = self.build_rollout_decode_queue_context(
                        gated_queue_last,
                        rollout_queue_h_t,
                        light_state_embedding,
                    )
                    queue_decode_context_seq.append(queue_context_for_decode)
                else:
                    queue_context_for_decode = gated_queue_last
                step_state_residual = self.build_decoder_state_residual(
                    light_state_embedding,
                    queue_context_for_decode,
                    cycle_context_for_decode,
                )
                if step_state_residual is not None:
                    pred_lstm_hidden = pred_lstm_hidden + step_state_residual
                    decoder_state_residual_seq.append(step_state_residual)
                pred_input = torch.cat((light_state_embedding, pred_lstm_hidden), dim=1)
                output = self.pred_hidden2pos(pred_input)
                last_rollout_offset = output
                pred_traj_rel += [output]

        if queue_rollout_hidden_seq:
            self.debug_last_aux["queue_rollout_hidden_seq"] = torch.stack(
                queue_rollout_hidden_seq
            )
            self.debug_last_aux["queue_rollout_feature_seq"] = torch.stack(
                queue_rollout_feature_seq
            )
            self.debug_last_aux["queue_rollout_pred_seq"] = torch.stack(
                queue_rollout_pred_seq
            )
            self.debug_last_aux["queue_rollout_target_seq"] = torch.stack(
                queue_rollout_target_seq
            )
        if queue_decode_context_seq:
            self.debug_last_aux["queue_decode_context_seq"] = torch.stack(
                queue_decode_context_seq
            )
        if cycle_rollout_hidden_seq:
            self.debug_last_aux["cycle_rollout_hidden_seq"] = torch.stack(
                cycle_rollout_hidden_seq
            )
        if cycle_decode_context_seq:
            self.debug_last_aux["cycle_decode_context_seq"] = torch.stack(
                cycle_decode_context_seq
            )
        if lane_queue_rollout_anchor_seq:
            self.debug_last_aux["lane_queue_rollout_anchor_seq"] = torch.stack(
                lane_queue_rollout_anchor_seq
            )
        if decoder_state_residual_seq:
            self.debug_last_aux["decoder_state_step_residual_seq"] = torch.stack(
                decoder_state_residual_seq
            )
            self.debug_last_aux["decoder_state_step_residual_norm"] = torch.stack(
                [step_residual.detach().norm(dim=1) for step_residual in decoder_state_residual_seq]
            )
        return torch.stack(pred_traj_rel)


# ----------------------------------------------------------------
# Phase 2 #20 契约: target 语义签名
# ----------------------------------------------------------------
#
# ``compute_queue_targets`` / ``build_cycle_features`` 的输出维度顺序
# 是与 ``train.compute_structured_aux_losses`` 强耦合的"接口契约"。
# 任何 reorder 必须同步修改这里的 helper docstring,以及
# ``train.py`` 中 ``queue_reg_idx`` / ``queue_cls_idx`` / cycle slice
# 的切分。这两个 helper 本身不参与计算,仅作为源码级"签名文档",
# 被 :mod:`tests.test_cyclestate_protocol` 引用以守卫契约不被破坏。
# ----------------------------------------------------------------


def build_queue_targets_signature():
    """Phase 2 #20 契约: 声明 ``compute_queue_targets`` 返回的 6 维顺序。

    返回值是一个可断言的结构化签名，而不是布尔占位符。
    这样测试既能守卫 target dim 顺序，也能守卫 source dim 到
    aux loss 子空间的映射不会静默漂移。

    Returns:
        tuple[dict[str, object], ...]: 6 个 target 维度的结构化签名。
    """
    return (
        {
            "target_index": 0,
            "name": "queue_count",
            "loss": "regression",
            "source_index": 0,
        },
        {
            "target_index": 1,
            "name": "lane_wait_ratio",
            "loss": "regression",
            "source_index": 3,
        },
        {
            "target_index": 2,
            "name": "lane_release_ratio",
            "loss": "regression",
            "source_index": 4,
        },
        {
            "target_index": 3,
            "name": "lane_queue_length",
            "loss": "regression",
            "source_index": 8,
        },
        {
            "target_index": 4,
            "name": "lane_stopline_occupancy",
            "loss": "binary",
            "source_index": 9,
        },
        {
            "target_index": 5,
            "name": "front_of_queue",
            "loss": "binary",
            "source_index": 10,
        },
    )


def build_cycle_features_signature():
    """Phase 2 #20 契约: 声明 ``build_cycle_features`` / ``get_step_cycle_feature`` 的 6 维顺序。

    Returns:
        tuple[dict[str, str | int], ...]: 6 个 cycle 维度的结构化签名。
    """
    return (
        {"target_index": 0, "name": "phase_red", "loss": "classification"},
        {"target_index": 1, "name": "phase_green", "loss": "classification"},
        {"target_index": 2, "name": "phase_yellow", "loss": "classification"},
        {"target_index": 3, "name": "elapsed", "loss": "regression"},
        {"target_index": 4, "name": "remaining", "loss": "regression"},
        {"target_index": 5, "name": "phase_change", "loss": "binary"},
    )


class TrajectoryDiscriminator(nn.Module):
    """轨迹判别器。

    判别器不是只看轨迹坐标，而是同时看轨迹和交通灯状态，判断一段行为序列
    是否符合真实的交通场景规律。
    """
    def __init__(
        self,
        obs_len,
        pred_len,
        part_lstm_input_size,
        part_lstm_hidden_size,
        merge_lstm_input_size,
        merge_lstm_hidden_size,
        dropout,
        light_input_size=4,
        embedding_size=32,
        light_embedding_size=16,
    ):
        super(TrajectoryDiscriminator, self).__init__()
        # 判别器的设计思路与生成器对称：
        # 一路编码轨迹几何，一路编码交通灯状态，再在时间维上融合两者，判断整段
        # 序列更像“真实驾驶行为”还是“生成结果”。
        self.light_input_size = light_input_size
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.light_embedding_size = light_embedding_size
        self.embedding_size = embedding_size

        # 交通灯状态编码分支。
        self.light_embedding = nn.Sequential(
            nn.BatchNorm1d(self.light_input_size),
            nn.ReLU(),
            nn.Linear(self.light_input_size, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.light_embedding_size),
            nn.ReLU()
        )

        # 轨迹坐标编码分支。
        self.pos_embedding = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU()
        )

        # 融合后的判别头。
        self.merge_embedding = nn.Sequential(
            nn.Linear(64, 32),
            # nn.ReLU(),
            nn.Linear(32,1),
            # nn.ReLU()
        )

        self.part_lstm_input_size = part_lstm_input_size
        self.part_lstm_hidden_size = part_lstm_hidden_size
        self.state_part_lstm = nn.LSTMCell(self.part_lstm_input_size, self.part_lstm_hidden_size)
        self.pos_part_lstm = nn.LSTMCell(self.part_lstm_input_size, self.part_lstm_hidden_size)
        self.merge_lstm_input_size = merge_lstm_input_size
        self.merge_lstm_hidden_size = merge_lstm_hidden_size
        self.merge_lstm = nn.LSTMCell(self.merge_lstm_input_size, self.merge_lstm_hidden_size)

    def init_hidden_part_lstm(self, batch):
        """初始化分支 LSTM 隐状态。

        Args:
            batch: 当前展平后的 agent 数。

        Returns:
            `(h_0, c_0)`，形状均为 `(batch, part_lstm_hidden_size)`。
        """
        device = get_module_device(self)
        return (
            torch.zeros(batch, self.part_lstm_hidden_size, device=device),
            torch.zeros(batch, self.part_lstm_hidden_size, device=device),
        )

    def init_hidden_merge_lstm(self, batch):
        """初始化融合 LSTM 隐状态。"""
        device = get_module_device(self)
        return (
            torch.zeros(batch, self.merge_lstm_hidden_size, device=device),
            torch.zeros(batch, self.merge_lstm_hidden_size, device=device),
        )

    def forward(self, traj, state, seq_start_end):
        """判别一段完整轨迹的真实性。

        Args:
            traj: 完整轨迹序列，形状 `(obs_len + pred_len, batch, 2)`。
            state: 对应时刻的交通灯状态序列。
            seq_start_end: 场景分组索引。当前判别器实现中没有显式使用，但保留接口
                以与训练流程保持统一。

        Returns:
            output: 判别分数，形状 `(batch, 1)`。数值越偏向真实分布，越容易被判别器
            视作真实样本。
        """
        batch = traj.shape[1]
        traj_lstm_h_t, traj_lstm_c_t = self.init_hidden_part_lstm(batch)
        state_lstm_h_t, state_lstm_c_t = self.init_hidden_part_lstm(batch)
        merge_lstm_h_t, merge_lstm_c_t = self.init_hidden_merge_lstm(batch)
        # 两个列表分别保存每个时间步上的轨迹分支隐藏状态和状态分支隐藏状态。
        traj_lstm_hidden_states = []
        state_lstm_hidden_states = []

        # 轨迹序列编码。
        for i, input_t in enumerate(traj[:].chunk(traj[:].size(0), dim=0)):
            input_t = self.pos_embedding(input_t.squeeze(0))
            traj_lstm_h_t, traj_lstm_c_t = self.pos_part_lstm(
                input_t.squeeze(0), (traj_lstm_h_t, traj_lstm_c_t))
            traj_lstm_hidden_states += [traj_lstm_h_t]

        # 状态序列编码。
        for i, input_t in enumerate(state[:].chunk(state[:].size(0), dim=0)):
            input_t = self.light_embedding(input_t.squeeze(0))
            state_lstm_h_t, state_lstm_c_t = self.pos_part_lstm(
                input_t.squeeze(0), (state_lstm_h_t, state_lstm_c_t))
            state_lstm_hidden_states += [state_lstm_h_t]

        # 按时间步融合两路特征。
        for i in range(len(traj_lstm_hidden_states)):
            # 每个时刻把“车辆怎么动”和“交通灯怎么约束”拼到一起，再交给融合 LSTM
            # 学习整段时序的一致性。
            input_t = torch.cat((traj_lstm_hidden_states[i], state_lstm_hidden_states[i]), dim=1)
            merge_lstm_h_t, merge_lstm_c_t = self.merge_lstm(
                input_t, (merge_lstm_h_t, merge_lstm_c_t)
            )
        output = self.merge_embedding(merge_lstm_h_t)
        return output
