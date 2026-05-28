"""
D2-TPred 模型定义文件
=====================
本文件定义了 D2-TPred (Discontinuous Dependency for Trajectory Prediction under Traffic Lights)
论文中的核心神经网络模型。该论文研究在交通灯信号影响下的车辆轨迹预测问题。

数据库：VTP-TL（无人机在交叉路口俯拍采集的车辆轨迹数据）
场景类型：十字路口 (crossroad)、T型路口 (T-junction)、环岛 (roundabout)
观测/预测长度：obs_len=8 帧 (3.2秒) 观测 → pred_len=12 帧 (4.8秒) 预测

模型架构概览：
1. 图注意力网络 (GAT) 组件 — 建模车辆间的空间交互关系
   - BatchMultiHeadGraphAttention: 带关系矩阵约束的多头图注意力层
   - seqBatchMultiHeadGraphAttention: 序列版多头图注意力层（用于时间维度上的交互建模）
   - GAT: 多层 GAT 堆叠，中间加入 LayerNorm + ELU + Dropout
   - seqGAT: 序列版多层 GAT 堆叠
   - GATEncoder: GAT 编码器，包含 relation_Matrix 方法，
     根据车辆运动方向和距离计算交互关系矩阵（空间上的间断依赖关系）
   - seqGATEncoder: 序列版 GAT 编码器，用于建模时间维度上的车辆交互

2. TrajectoryGenerator (轨迹生成器, Generator) — GAN 的生成器
     输入: 观测轨迹 (obs_len=8帧) + 4个交通灯坐标 + 交通灯状态序列
     处理流程:
       a. 轨迹 LSTM: 对每条车辆轨迹独立编码 (LSTMCell)
       b. GATEncoder + seqGATEncoder: 建模车辆间空间-时间交互
       c. 交通灯嵌入: 计算车辆与最近交通灯的相对距离和状态
       d. 噪声注入: 在预测 LSTM 前注入随机噪声实现多模态预测
       e. 预测 LSTM: 自回归生成未来12帧轨迹
       f. 使用 teacher forcing 策略训练 (默认 50% 概率使用真值)
     输出: 未来 12 帧的相对位移 (pred_len x batch x 2)

3. TrajectoryDiscriminator (轨迹判别器, Discriminator) — GAN 的判别器
     输入: 完整轨迹 (obs_len + pred_len 帧) + 对应的交通灯状态序列
     处理流程:
       a. 双路 Part-LSTM: 分别编码轨迹位置 和 交通灯状态
       b. Merge-LSTM: 融合双路特征
       c. MLP 分类头: 输出轨迹真实性得分
     输出: 每条轨迹的判别得分 (batch x 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from utils import relative_to_abs
import math


# ============================================================
# 工具函数
# ============================================================

def get_noise(shape, noise_type):
    """
    生成随机噪声向量，用于在生成器中注入随机性以实现多模态轨迹预测。

    参数:
        shape: 噪声张量的形状，例如 (batch_size, noise_dim)
        noise_type: 噪声类型，支持 "gaussian" (标准正态分布) 或 "uniform" ([-1, 1] 均匀分布)

    返回:
        CUDA 张量，形状为 shape 的随机噪声
    """
    if noise_type == "gaussian":
        return torch.randn(*shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


# ============================================================
# 图注意力网络 (GAT) 核心层
# ============================================================

class BatchMultiHeadGraphAttention(nn.Module):
    """
    带关系矩阵约束的多头图注意力层 (Multi-Head Graph Attention with Relation Matrix)。

    这是 D2-TPred 的核心创新之一。与标准 GAT 不同，本层在计算注意力权重时
    额外乘以一个 Relation 矩阵 (由 GATEncoder.relation_Matrix 计算)，
    用于建模车辆间基于运动方向和距离的"间断依赖"关系。

    输入:
        h:        节点特征, 形状 (batch_size, num_nodes, f_in)
        Relation: 关系约束矩阵, 形状 (batch_size, num_nodes, num_nodes)
                 值为 0/1，0 表示两节点间无交互关系

    输出:
        更新后的节点特征, 形状 (batch_size, num_nodes, n_head * f_out)
        注意力权重矩阵 (用于可视化分析)

    参数:
        n_head:       注意力头数
        f_in:         输入特征维度
        f_out:        每个注意力头的输出特征维度
        attn_dropout: 注意力权重的 dropout 比率
        bias:         是否使用偏置项
    """
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        # 可学习的线性变换矩阵 W: (n_head, f_in, f_out)
        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
        # 注意力机制中的源节点向量 a_src: (n_head, f_out, 1)
        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        # 注意力机制中的目标节点向量 a_dst: (n_head, f_out, 1)
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
        """
        前向传播：计算带关系约束的多头图注意力。

        步骤：
        1. 对输入特征做线性变换: h_prime = h @ W
        2. 计算源-目标节点对的原始注意力分数: e_ij = a_src(h'_i) + a_dst(h'_j)
        3. LeakyReLU 激活
        4. 乘以关系矩阵 Relation 进行掩码（屏蔽无关节点对）
        5. Softmax 归一化得到最终注意力权重
        6. 加权聚合邻居节点特征
        """
        bs, n = h.size()[:2]  # bs: batch_size, n: 节点数
        # (bs, n, f_in) -> (bs, 1, n, f_in) @ (n_head, f_in, f_out) -> (bs, n_head, n, f_out)
        h_prime = torch.matmul(h.unsqueeze(1), self.w)
        # 源节点注意力分数: (bs, n_head, n, 1)
        attn_src = torch.matmul(h_prime, self.a_src)
        # 目标节点注意力分数: (bs, n_head, n, 1)
        attn_dst = torch.matmul(h_prime, self.a_dst)
        # 完整注意力分数 e_ij = attn_src_i + attn_dst_j: (bs, n_head, n, n)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )
        attn = self.leaky_relu(attn)
        # 关键步骤：将注意力权重乘以关系矩阵，只保留有依赖关系的节点对
        attn = torch.mul(Relation.unsqueeze(1).repeat(1, self.n_head, 1, 1).cuda(), attn)
        attn = self.softmax(attn)

        attn = self.dropout(attn)
        # 加权聚合: (bs, n_head, n, n) @ (bs, n_head, n, f_out) -> (bs, n_head, n, f_out)
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
    """
    序列版多头图注意力层 (不带关系矩阵约束)。

    与 BatchMultiHeadGraphAttention 的主要区别：
    1. 不使用 Relation 矩阵 —— 所有节点对之间都允许交互
    2. 用于 seqGATEncoder 中建模时间维度上的车辆交互
       （对同一场景在不同时间步的 GAT 输出做序列级图注意力）

    输入:
        h: 节点特征, 形状 (batch_size, num_nodes, f_in)

    输出:
        更新后的节点特征, 形状 (batch_size, num_nodes, n_head * f_out)
        注意力权重矩阵

    参数:
        n_head:       注意力头数
        f_in:         输入特征维度
        f_out:        每个注意力头的输出特征维度
        attn_dropout: 注意力权重的 dropout 比率
        bias:         是否使用偏置项
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
        """
        前向传播：计算多头图注意力（无关系矩阵约束）。

        与 BatchMultiHeadGraphAttention.forward 的区别：
        跳过 Relation 矩阵乘法步骤，直接 Softmax 归一化。
        """
        bs, n = h.size()[:2]
        # 线性变换: (bs, n, f_in) -> (bs, n_head, n, f_out)
        h_prime = torch.matmul(h.unsqueeze(1), self.w)
        attn_src = torch.matmul(h_prime, self.a_src)
        attn_dst = torch.matmul(h_prime, self.a_dst)
        # 计算完整注意力分数 e_ij: (bs, n_head, n, n)
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
    """
    多层图注意力网络堆叠封装 (带关系矩阵约束)。

    将多个 BatchMultiHeadGraphAttention 层堆叠，层间使用 LayerNorm + ELU + Dropout。

    网络结构 (n_units=[32, 32, 64], n_heads=[4, 2]):
      输入 (32维) → GAT层1 (n_head=4, 32->32) → LayerNorm → ELU → Dropout
                                                          ↓
      → reshape (n_head*f_out=128 → concat为下一个输入)
                                                          ↓
                  → GAT层2 (n_head=2, 128->64) → LayerNorm → ELU → 输出 (64维)

    参数:
        n_units:  每层输出维度列表, 例如 [32, 32, 64]，首元素为输入维度
        n_heads:  每层的注意力头数, 例如 [4, 2]
        dropout:  dropout 比率
        alpha:    LeakyReLU 的负斜率 (通过 BatchMultiHeadGraphAttention 使用)
    """
    def __init__(self, n_units, n_heads, dropout=0.2, alpha=0.2):
        super(GAT, self).__init__()
        self.n_layer = len(n_units) - 1  # GAT 层数 = len(n_units) - 1
        self.dropout = dropout
        self.layer_stack = nn.ModuleList()

        for i in range(self.n_layer):
            # 第一个 GAT 层的输入维度 = n_units[0]
            # 后续层的输入维度 = 前一层 n_head * f_out (多头拼接)
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                BatchMultiHeadGraphAttention(
                    n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=dropout))

        self.norm_list = nn.ModuleList([
            torch.nn.LayerNorm(32),
            torch.nn.LayerNorm(64),
        ])

    def forward(self, x, Relation):
        """
        前向传播。

        输入:
            x:        节点特征, 形状 (batch_size, num_nodes, f_in)
            Relation: 关系矩阵, 形状 (batch_size, num_nodes, num_nodes)

        输出:
            最终 GAT 编码结果, 形状 (batch_size, num_nodes, f_out_last)
        """
        bs, n = x.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = self.norm_list[i](x)  # LayerNorm 归一化
            x, attn = gat_layer(x, Relation)  # 多头图注意力
            if i + 1 == self.n_layer:
                # 最后一层: 去除 n_head 维度 (因为 n_head=1 时可 squeeze)
                x = x.squeeze(dim=1)
            else:
                # 中间层: 将 (bs, n_head, n, f_out) 重塑为 (bs, n, n_head*f_out)
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        else:
            return x

class seqGAT(nn.Module):
    """
    序列版多层图注意力网络堆叠封装 (不带关系矩阵约束)。

    与 GAT 的区别：
    - 使用 seqBatchMultiHeadGraphAttention 层（无 Relation 矩阵约束）
    - 用于 seqGATEncoder 中，对多帧 GAT 编码结果进行时间维度的图注意力聚合

    网络结构 (n_units=[32, 32, 64], n_heads=[4, 2]):
      输入 (32维) → seqGAT层1 (n_head=4, 32->32) → LayerNorm → ELU → Dropout
                                                              ↓
                  → GAT层2 (n_head=2, 128->64) → LayerNorm → ELU → 输出 (64维)

    参数同 GAT 类。
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

        self.norm_list = nn.ModuleList([
            torch.nn.LayerNorm(32),
            torch.nn.LayerNorm(64),
        ])

    def forward(self, x):
        bs, n = x.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = self.norm_list[i](x)
            x, attn = gat_layer(x)
            if i + 1 == self.n_layer:
                x = x.squeeze(dim=1)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        else:
            return x


class GATEncoder(nn.Module):
    """
    GAT 编码器：将车辆轨迹的 LSTM 隐藏状态通过图注意力网络进行空间交互建模。

    核心功能：
    1. relation_Matrix(): 计算车辆间的"间断依赖"关系矩阵
       基于车辆运动方向和距离，判断两车之间是否存在交互关系。
       这是 D2-TPred 论文的核心创新 —— 不是所有车辆之间都有交互，
       只有满足特定方向关系的车辆对才会被认为有依赖关系。
    2. forward(): 对每个场景中的车辆执行 GAT 消息传递

    参数:
        n_units: GAT 每层输出维度列表
        n_heads: GAT 每层注意力头数
        dropout: dropout 比率
        alpha:   LeakyReLU 负斜率
    """
    def __init__(self, n_units, n_heads, dropout, alpha):
        super(GATEncoder, self).__init__()
        self.gat_net = GAT(n_units, n_heads, dropout, alpha)

    def relation_Matrix(self, curr_dire):
        """
        计算车辆间的间断依赖关系矩阵 (Relation Matrix)。

        这是 D2-TPred 的核心创新。根据车辆运动方向角度和欧几里得距离，
        判断两辆车之间是否存在有意义的交互关系。

        判断条件：
        1. 两车间欧几里得距离 <= 156.0 像素 (约等于实际空间阈值)
        2. 目标车辆的方向角落在 [a_i - 62°, a_i + 62°] 的扇形范围内
           (a_i 为源车辆的运动方向角)

        参数:
            curr_dire: 当前场景的车辆方向数据
                       形状 (F, N, D)，其中:
                       - F: 观测帧数 (obs_len)
                       - N: 当前场景中的车辆数
                       - D: 特征维度
                         indices 2:3 → 车辆坐标 (x, y)
                         index 5     → 车辆运动方向角 (度)

        返回:
            r: 关系矩阵, 形状 (F, N, N)
               r[i,j] = 1 表示在第 i 帧, 车辆 j 对车辆 i 存在依赖关系 (有交互)
               r[i,j] = 0 表示无交互
        """
        # 提取车辆坐标 (x, y): (F, N, 2)
        currdata = curr_dire[:, :, 2:4]
        F, N, D = currdata.size()
        l = 156.0  # 距离阈值 (像素)

        # 计算两两欧几里得距离: (F, N, N)
        d = torch.cdist(currdata, currdata, p=2)
        # 距离掩码: 距离 <= l 的节点对才可能有交互
        d_mask = (d <= l).float()

        # 提取车辆运动方向角: (F, N)
        a = curr_dire[:, :, 5]

        # 计算两两车辆之间的方向向量差
        diff = currdata.unsqueeze(2) - currdata.unsqueeze(1)  # (F, N, N, 2)
        diffx = diff[:, :, :, 0]  # x 方向差
        diffy = diff[:, :, :, 1]  # y 方向差

        # atan2 计算方向角度 (度): 从源车指向目标车的方向角
        dire = 180 * torch.atan2(diffy, diffx) / math.pi
        # 归一化到 [0, 360)
        dire = torch.where(dire < 0, dire + 360, dire)
        # 处理边界情况: x 方向差接近 0 时的特殊处理
        diffx_zero = diffx.abs() < 1e-8
        dire = torch.where(diffx_zero & (diffy > 0), torch.tensor(90.0, device=dire.device), dire)
        dire = torch.where(diffx_zero & (diffy < 0), torch.tensor(270.0, device=dire.device), dire)
        dire = torch.where(diffx_zero & (diffy.abs() < 1e-8), torch.tensor(0.0, device=dire.device), dire)

        # 方向扇区: [a_i - 62°, a_i + 62°] 即前方约 124° 范围
        up = a.unsqueeze(2) + 62    # 扇区上界
        down = a.unsqueeze(2) - 62  # 扇区下界

        r = torch.ones(F, N, N, device=curr_dire.device)

        # 情况1: 上界超过 360° (跨越 0° 边界)
        #   有效范围: [down, 360] ∪ [0, up-360]
        case1 = up > 360
        case1_valid = ((down <= dire) & (dire <= 360)) | ((0 <= dire) & (dire <= (up - 360)))
        r = torch.where(case1, case1_valid.float(), r)

        # 情况2: 上界在 [62°, 124°] 范围内
        case2 = (up >= 62) & (up <= 124)
        case2_valid = ((down + 360 <= dire) & (dire <= 360)) | ((0 <= dire) & (dire <= up))
        r = torch.where(case2, case2_valid.float(), r)

        # 最终关系矩阵 = 方向条件 AND 距离条件
        r = r * d_mask

        return r


    def forward(self, obs_traj_embedding, seq_start_end, obs_dire):
        """
        对每个场景中的车辆执行 GAT 编码。

        参数:
            obs_traj_embedding: LSTM 隐藏状态序列
                               形状 (obs_len, total_vehicles, hidden_dim)
            seq_start_end:      场景起止索引, 形状 (num_scenes, 2)
                               每行 [start, end] 标记一个场景中车辆的索引范围
            obs_dire:           车辆方向数据, 形状 (obs_len, total_vehicles, 6+)

        返回:
            graph_embeded_data: GAT 编码后的特征
                               形状 (obs_len, total_vehicles, gat_output_dim)
        """
        graph_embeded_data = []

        for start, end in seq_start_end.data:
            # 取出当前场景中所有车辆的嵌入
            curr_seq_embedding_traj = obs_traj_embedding[:, start:end, :]
            curr_obs_dire = obs_dire[:, start:end, :]
            # 计算当前场景的关系矩阵
            Relation = self.relation_Matrix(curr_obs_dire)
            # GAT 消息传递
            curr_seq_graph_embedding = self.gat_net(curr_seq_embedding_traj, Relation)
            graph_embeded_data.append(curr_seq_graph_embedding)
        # 合并所有场景的结果
        graph_embeded_data = torch.cat(graph_embeded_data, dim=1)
        return graph_embeded_data

class seqGATEncoder(nn.Module):
    """
    序列版 GAT 编码器：对时间维度上的 GAT 编码结果进行图注意力聚合。

    与 GATEncoder 的区别：
    - 使用 seqGAT（不带关系矩阵约束）而非 GAT
    - 不需要 Relation 矩阵和 obs_dire，所有节点对之间全连接交互
    - 用于建模车辆在不同时间步的 GAT 输出之间的时序交互关系

    在 TrajectoryGenerator 中的使用场景：
    对每帧的 GAT 编码输出，取滑动窗口（kl=6 帧）内的多帧编码结果，
    通过 seqGATEncoder 进行时间维度的图注意力聚合，
    以捕获车辆交互随时间演化的动态模式。

    参数同 GATEncoder。
    """
    def __init__(self, n_units, n_heads, dropout, alpha):
        super(seqGATEncoder, self).__init__()
        self.seq_gat_net = seqGAT(n_units, n_heads, dropout, alpha)

    def forward(self, obs_traj_embedding, seq_start_end):
        """
        对每个场景中的车辆执行序列级 GAT 编码。

        参数:
            obs_traj_embedding: 嵌入序列, 形状 (time_steps, num_vehicles, hidden_dim)
            seq_start_end:      场景起止索引, 形状 (num_scenes, 2)

        返回:
            graph_embeded_data: 编码后特征, 形状 (time_steps, total_vehicles, gat_output_dim)
        """
        graph_embeded_data = []
        for start, end in seq_start_end.data:
            # 取出当前场景的嵌入
            curr_seq_embedding_traj = obs_traj_embedding[:, start:end, :]
            # 序列级 GAT 聚合
            curr_seq_graph_embedding = self.seq_gat_net(curr_seq_embedding_traj)
            graph_embeded_data.append(curr_seq_graph_embedding)
        # 合并所有场景
        graph_embeded_data = torch.cat(graph_embeded_data, dim=1)
        return graph_embeded_data

class TrajectoryGenerator(nn.Module):
    """
    D2-TPred 轨迹生成器 (GAN Generator)。

    根据观测轨迹和交通灯状态预测车辆未来轨迹。采用编码器-解码器架构，
    结合 LSTM、GAT、交通灯嵌入和随机噪声实现多模态轨迹预测。

    ┌─────────────────────────────────────────────────────────────────┐
    │                       数据流概览                                │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  输入数据:                                                      │
    │  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
    │  │ obs_traj_rel    │  │ obs_traj_pos     │  │ obs_state /   │  │
    │  │ (8帧, V, 2)     │  │ (8帧, V, 11)     │  │ pred_state    │  │
    │  │ 相对位移        │  │ 位置+方向+灯信息  │  │ 交通灯状态    │  │
    │  └───────┬─────────┘  └────────┬─────────┘  └───────┬───────┘  │
    │          │                     │                     │          │
    │          ▼                     ▼                     │          │
    │  ┌───────────────┐    ┌───────────────┐              │          │
    │  │ Trajectory    │    │ GATEncoder    │              │          │
    │  │ LSTM (LSTMCell)│   │ + seqGATEncoder│             │          │
    │  │ 每车独立编码   │    │ 空间+时间交互  │              │          │
    │  └───────┬───────┘    └───────┬───────┘              │          │
    │          │                    │                       │          │
    │          │     ┌──────────────┴───────────┐          │          │
    │          │     │                          │          │          │
    │          ▼     ▼                          ▼          │          │
    │  ┌──────────────────────┐    ┌─────────────────────┐ │          │
    │  │ 隐藏状态拼接          │    │ light_state        │ │          │
    │  │ traj_h + graph_h     │    │ (距离+灯态, 5维)    │ │          │
    │  └──────────┬───────────┘    └──────────┬──────────┘ │          │
    │             │                           │             │          │
    │             └──────────┬────────────────┘             │          │
    │                        ▼                              │          │
    │              ┌──────────────────┐                     │          │
    │              │ 拼接 + 噪声注入   │                     │          │
    │              │ (136维 → pred_h) │                     │          │
    │              └────────┬─────────┘                     │          │
    │                       ▼                               │          │
    │              ┌──────────────────┐                     │          │
    │              │ Prediction LSTM  │                     │          │
    │              │ + light_embedding│                     │          │
    │              │ 自回归解码 12帧   │                     │          │
    │              └────────┬─────────┘                     │          │
    │                       ▼                               │          │
    │              输出: 未来12帧相对位移 (12, V, 2)          │          │
    └─────────────────────────────────────────────────────────────────┘

    关键参数说明:
        obs_len:                 观测帧数, 论文中为 8
        pred_len:                预测帧数, 论文中为 12
        traj_lstm_input_size:    轨迹 LSTM 输入维度 (2: dx, dy)
        traj_lstm_hidden_size:   轨迹 LSTM 隐藏状态维度 (32)
        n_units:                 GAT 层输出维度列表, 如 [32, 32, 64]
        n_heads:                 GAT 每层注意力头数, 如 [4, 2]
        graph_network_out_dims:  GAT 输出维度 (64)
        graph_lstm_hidden_size:  图 LSTM 隐藏维度 (64)
        noise_dim:               噪声维度 (16,)
        noise_type:              噪声类型 ("gaussian" | "uniform")
        light_input_size:        交通灯输入维度 (5: 距离x3 + 灯状态x2)
        embedding_size:          嵌入中间维度 (64)
        light_embedding_size:    交通灯嵌入输出维度 (32)
    """
    def __init__(
        self,
        obs_len,
        pred_len,
        traj_lstm_input_size,   # 2  — 相对位移 (dx, dy)
        traj_lstm_hidden_size,  # 32 — 轨迹 LSTM 隐藏维度
        n_units,
        n_heads,
        graph_network_out_dims, # 64 — GAT 输出维度
        dropout,
        alpha,
        graph_lstm_hidden_size, # 64 — 图 LSTM 隐藏维度
        noise_dim=(8,),
        noise_type="gaussian",
        light_input_size=5,     # 5 — (距离, dis_x, dis_y, 灯状态_1, 灯状态_2)
        embedding_size=64,
        light_embedding_size=32,
    ):
        super(TrajectoryGenerator, self).__init__()
        self.embedding_size = embedding_size
        self.light_input_size = light_input_size
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.light_embedding_size = light_embedding_size

        # === GAT 编码器 (空间交互 + 时间交互) ===
        self.gatencoder = GATEncoder(
            n_units=n_units, n_heads=n_heads, dropout=dropout, alpha=alpha
        )
        self.seqgatencoder = seqGATEncoder(
            n_units=n_units, n_heads=n_heads, dropout=dropout, alpha=alpha
        )

        self.graph_lstm_hidden_size = graph_lstm_hidden_size    # 64
        self.traj_lstm_hidden_size = traj_lstm_hidden_size      # 32

        # 预测 LSTM 的隐藏维度 = 交通灯嵌入 + 轨迹隐藏 + 图隐藏 + 噪声
        self.pred_lstm_hidden_size = (
            self.light_embedding_size + self.traj_lstm_hidden_size
            + self.graph_lstm_hidden_size + noise_dim[0]
        )

        # === LSTM 层 ===
        # 轨迹 LSTM: 编码每辆车的轨迹序列
        self.traj_lstm_model = nn.LSTMCell(traj_lstm_input_size, traj_lstm_hidden_size)
        # 图 LSTM: 编码 GAT 输出的图特征序列
        self.graph_lstm_model = nn.LSTMCell(
            graph_network_out_dims, graph_lstm_hidden_size
        )

        # === 交通灯状态嵌入 MLP ===
        # 将 (距离, dis_x, dis_y, 灯状态_1, 灯状态_2) 5维 → 32维嵌入
        self.light_embedding = nn.Sequential(
            nn.BatchNorm1d(self.light_input_size),
            nn.ReLU(),
            nn.Linear(self.light_input_size, self.embedding_size),    # 5 → 64
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.light_embedding_size), # 64 → 32
            nn.ReLU()
        )

        # === 输出投影层 ===
        # 从轨迹 LSTM 隐藏状态 + 交通灯嵌入 → 相对位移 (用于训练辅助)
        self.traj_hidden2pos = nn.Linear(
            self.traj_lstm_hidden_size + self.light_embedding_size, 2
        )
        # 从轨迹 + GAT + 交通灯隐藏状态 → 相对位移 (用于训练辅助)
        self.traj_gat_hidden2pos = nn.Linear(
            self.light_embedding_size + self.traj_lstm_hidden_size
            + self.graph_lstm_hidden_size, 2
        )
        # 从预测 LSTM 隐藏状态 + 交通灯嵌入 → 相对位移 (主解码器)
        self.pred_hidden2pos = nn.Linear(
            self.light_embedding_size + self.pred_lstm_hidden_size, 2
        )

        self.noise_dim = noise_dim
        self.noise_type = noise_type

        # === 预测 LSTM (解码器) ===
        # 输入维度 = 2 (dx, dy), 隐藏维度 = 32+64+64+16=176
        self.pred_lstm_model = nn.LSTMCell(
            traj_lstm_input_size, self.pred_lstm_hidden_size
        )

    def init_hidden_traj_lstm(self, batch):
        """
        初始化轨迹 LSTM 的隐藏状态和细胞状态。

        参数:
            batch: batch 大小 (总车辆数)

        返回:
            (h_0, c_0): 随机初始化的隐藏状态和细胞状态
        """
        return (
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
        )

    def init_hidden_graph_lstm(self, batch):
        """初始化图 LSTM 的隐藏状态和细胞状态。"""
        return (
            torch.randn(batch, self.graph_lstm_hidden_size).cuda(),
            torch.randn(batch, self.graph_lstm_hidden_size).cuda(),
        )

    def init_hidden_light_lstm(self, batch):
        """初始化交通灯 LSTM 的隐藏状态和细胞状态。"""
        return (
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
        )

    def add_noise(self, _input, seq_start_end):
        """
        在编码器输出上注入随机噪声，实现轨迹预测的多模态性。

        对每个场景生成独立的噪声向量，同一场景中的车辆共享该噪声。

        参数:
            _input:         编码器输出特征, 形状 (total_vehicles, hidden_dim)
            seq_start_end:  场景起止索引, 形状 (num_scenes, 2)

        返回:
            decoder_h: 拼接了噪声的解码器输入, 形状 (total_vehicles, hidden_dim + noise_dim)
        """
        noise_shape = (seq_start_end.size(0),) + self.noise_dim  # (num_scenes, 16)

        z_decoder = get_noise(noise_shape, self.noise_type)

        _list = []
        for idx, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            _vec = z_decoder[idx].view(1, -1)
            _to_cat = _vec.repeat(end - start, 1)  # 同一场景中所有车辆共享噪声
            _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
        decoder_h = torch.cat(_list, dim=0)

        return decoder_h

    def get_last_state(self, obs_traj_pos, obs_state):
        """
        计算观测阶段最后一帧时，车辆与对应交通灯的相对状态。

        参数:
            obs_traj_pos: 观测轨迹位置, 形状 (obs_len, batch, 11)
            obs_state:    观测阶段的交通灯状态, 形状 (obs_len, batch, 5+)

        返回:
            state_last: 最后一帧的车辆-交通灯状态, 形状 (batch, 5)
                        [欧几里得距离, x方向距离, y方向距离, 灯状态1, 灯状态2]
        """
        # 欧几里得距离 + x/y 方向距离
        dis = torch.sqrt(
            (obs_traj_pos[-1, :, 2] - obs_state[-1, :, 0]) ** 2
            + (obs_traj_pos[-1, :, 3] - obs_state[-1, :, 1]) ** 2
        )
        disx = obs_traj_pos[-1, :, 2] - obs_state[-1, :, 0]
        disy = obs_traj_pos[-1, :, 3] - obs_state[-1, :, 1]
        light_state = obs_state[-1, :, 2:4]  # 交通灯的两个状态值
        dis_state = torch.stack([dis, disx, disy], dim=1)  # (batch, 3)
        state_last = torch.cat((dis_state, light_state), dim=1)  # (batch, 5)

        return state_last   # (batch, 5)

    def get_next_state(self, pred_traj_rel, obs_traj_pos, pred_state):
        """
        计算预测过程中当前帧时，车辆与交通灯的相对状态。

        与 get_last_state 类似，但每次使用最新预测的位置计算。

        参数:
            pred_traj_rel: 已预测的相对位移列表 (每个元素形状 (batch, 2))
            obs_traj_pos:  观测轨迹位置, 形状 (obs_len, batch, 11)
            pred_state:    预测阶段的交通灯状态, 形状 (pred_len, batch, 5+)

        返回:
            state_last: 当前帧的车辆-交通灯状态, 形状 (batch, 5)
        """
        pred_traj_rel = torch.stack(pred_traj_rel)     # (T_pred, batch, 2)
        step = pred_traj_rel.size(0)

        # 起始位置 = 观测轨迹最后一帧的绝对位置
        start_pos = obs_traj_pos[-1, :, 2:4]           # (batch, 2)
        # 将相对位移转换为绝对位置
        real_pos = relative_to_abs(pred_traj_rel, start_pos)  # (T_pred, batch, 2)

        # 计算最新预测位置与交通灯的距离
        dis = torch.sqrt(
            (real_pos[-1, :, 0] - pred_state[-1, :, 0]) ** 2
            + (real_pos[-1, :, 1] - pred_state[-1, :, 1]) ** 2
        )
        disx = real_pos[-1, :, 0] - pred_state[-1, :, 0]
        disy = real_pos[-1, :, 1] - pred_state[-1, :, 1]
        dis_state = torch.stack([dis, disx, disy], dim=1)  # (batch, 3)

        # 取当前步的交通灯状态
        last_state = pred_state[step - 1, :, 2:4]  # (batch, 2)

        state_last = torch.cat((dis_state, last_state), dim=1)  # (batch, 5)

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
        """
        Generator 前向传播：编码观测轨迹 → 注入噪声 → 自回归解码未来轨迹。

        执行流程:
        1. 轨迹 LSTM 编码: 对 obs_len=8 帧观测轨迹逐帧编码
        2. GAT 交互编码: 对 LSTM 隐藏状态执行空间 GAT (+关系矩阵)
        3. seqGAT 时序交互: 滑动窗口聚合时间维度交互 (kl=6)
        4. 交通灯嵌入: 计算车辆与最近交通灯的状态
        5. 噪声注入: 拼接隐藏状态并注入随机噪声
        6. 自回归解码: 12 帧循环预测，每帧结合当前交通灯状态
        7. Teacher Forcing: 训练时有 p=0.5 概率使用真实值

        参数:
            obs_traj_rel:  观测轨迹相对位移, 形状 (obs_len, batch, 11)
            obs_traj_pos:  观测轨迹绝对位置+方向, 形状 (obs_len, batch, 11)
            obs_state:     观测阶段交通灯状态, 形状 (obs_len, batch, 5+)
            pred_state:    预测阶段交通灯状态, 形状 (pred_len, batch, 5+)
            seq_start_end: 场景起止索引, 形状 (num_scenes, 2)
            teacher_forcing_ratio: Teacher Forcing 概率, 默认 0.5

        返回:
            outputs: 预测轨迹相对位移, 形状 (pred_len, batch, 2)
        """
        batch = obs_traj_rel.shape[1]

        # ============ 阶段1: 初始化 ============
        traj_lstm_h_t, traj_lstm_c_t = self.init_hidden_traj_lstm(batch)
        pred_traj_rel = []           # 存储预测结果
        traj_lstm_hidden_states = [] # 存储每帧的轨迹 LSTM 隐藏状态
        graph_lstm_hidden_states = []# 存储每帧的图 LSTM 隐藏状态

        # ============ 阶段2: 轨迹 LSTM 编码 ============
        for i, input_t in enumerate(
            obs_traj_rel[: self.obs_len].chunk(obs_traj_rel[: self.obs_len].size(0), dim=0)):
            inputtraj = input_t[:, :, 2:4]  # 只取相对位移 (dx, dy)
            traj_lstm_h_t, traj_lstm_c_t = self.traj_lstm_model(
                inputtraj.squeeze(0), (traj_lstm_h_t, traj_lstm_c_t))
            traj_lstm_hidden_states += [traj_lstm_h_t]  # 收集每帧隐藏状态

        # ============ 阶段3: GAT 空间交互编码 ============
        kl = 6  # 滑动窗口大小 (用于 seqGAT 的时序交互)
        obs_dire = obs_traj_pos[:, :, 0:6]
        obs_dire[:, :, 5] = obs_traj_pos[:, :, 9]  # 将运动方向放入第5个位置
        # 对所有帧的 LSTM 隐藏状态执行 GAT 编码 (空间交互)
        graph_lstm_input = self.gatencoder(
            torch.stack(traj_lstm_hidden_states), seq_start_end, obs_dire
        )

        # ============ 阶段4: seqGAT 时序窗口交互 ============
        staend = torch.zeros((1, 2), dtype=torch.int)
        with torch.no_grad():
            for j in range(self.obs_len):
                if j <= kl:
                    # 帧数不足窗口大小时，使用所有已编码帧
                    staend[0, 1] = j + 1
                    graph_inter_input = self.seqgatencoder(
                        graph_lstm_input[0:(j + 1)].permute(1, 0, 2), staend)
                else:
                    # 使用滑动窗口 (当前帧及之前 kl=6 帧)
                    staend[0, 1] = kl + 1
                    graph_inter_input = self.seqgatencoder(
                        graph_lstm_input[(j - kl):(j + 1)].permute(1, 0, 2), staend)
                # 取最后一帧的输出
                graph_lstm_hidden_states += [graph_inter_input[:, -1, :]]

        # ============ 阶段5: 交通灯状态嵌入 + 噪声注入 ============
        light_state = self.get_last_state(obs_traj_pos, obs_state)
        light_state_embedding = self.light_embedding(light_state)  # (batch, 32)
        # 拼接: 交通灯嵌入 + 轨迹隐藏状态 (最后一帧) + 图隐藏状态 (最后一帧)
        encoded_before_noise_hidden = torch.cat(
            (light_state_embedding, traj_lstm_hidden_states[-1], graph_lstm_hidden_states[-1]),
            dim=1)  # 32 + 32 + 64 = 128 维 (实际为 32+32+64=128, 这里注释可能有误)

        # 注入噪声
        pred_lstm_hidden = self.add_noise(
            encoded_before_noise_hidden, seq_start_end
        )
        pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden).cuda()

        # ============ 阶段6: 自回归解码 12 帧 ============
        obs_traj_rel = obs_traj_rel[:, :, 2:4]  # 只保留相对位移
        output = obs_traj_rel[self.obs_len - 1] # 起始输入 = 观测最后一帧位移

        if self.training:
            # ------- 训练模式：使用 Teacher Forcing -------
            for i, input_t in enumerate(
                    obs_traj_rel[-self.pred_len:].chunk(
                        obs_traj_rel[-self.pred_len:].size(0), dim=0
                    )  # 真实未来12帧
            ):
                # Teacher Forcing: 以 probability 使用真实值
                teacher_force = random.random() < teacher_forcing_ratio
                input_t = input_t if teacher_force else output.unsqueeze(0)
                pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(
                    input_t.squeeze(0), (pred_lstm_hidden, pred_lstm_c_t)  # 隐藏维度 176
                )
                # 更新当前帧的交通灯状态
                if i == 0:
                    light_state = self.get_last_state(obs_traj_pos, obs_state)
                else:
                    light_state = self.get_next_state(pred_traj_rel, obs_traj_pos, pred_state)
                light_state_embedding = self.light_embedding(light_state)
                # 拼接交通灯嵌入和 LSTM 隐藏状态
                pred_input = torch.cat((light_state_embedding, pred_lstm_hidden), dim=1)  # 32+176=208
                output = self.pred_hidden2pos(pred_input)  # (batch, 2)
                pred_traj_rel += [output]
            outputs = torch.stack(pred_traj_rel)  # (12, batch, 2)
        else:
            # ------- 评估模式：纯自回归（不使用真值） -------
            for i in range(self.pred_len):
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


# ============================================================
# 轨迹判别器 (GAN Discriminator)
# ============================================================

class TrajectoryDiscriminator(nn.Module):
    """
    D2-TPred 轨迹判别器 (GAN Discriminator)。

    判断输入轨迹是真实的还是生成器生成的伪造轨迹。采用双路 LSTM 架构，
    分别编码轨迹位置序列和交通灯状态序列，再通过合并 LSTM 融合并给出判别得分。

    架构:
    ┌──────────────────────────────────────────────────┐
    │                                                  │
    │  traj (20帧, V, 2)         state (20帧, V, 4)    │
    │       │                          │               │
    │       ▼ pos_embedding            ▼ light_embedding
    │  ┌──────────┐              ┌──────────┐          │
    │  │ Pos LSTM │              │ State LSTM│          │
    │  │ (LSTMCell)│             │ (LSTMCell)│          │
    │  └────┬─────┘              └────┬─────┘          │
    │       │ 32维                    │ 32维            │
    │       └────────┬────────────────┘                │
    │                ▼ cat → 64维                      │
    │         ┌─────────────┐                          │
    │         │ Merge LSTM  │                          │
    │         │ (LSTMCell)  │                          │
    │         └──────┬──────┘                          │
    │                ▼ 64维                              │
    │         ┌─────────────┐                          │
    │         │ Merge MLP   │                          │
    │         │ 64→32→1     │                          │
    │         └──────┬──────┘                          │
    │                ▼                                  │
    │         判别得分 (V, 1)                          │
    └──────────────────────────────────────────────────┘

    关键参数:
        obs_len:                 观测帧数 (8)
        pred_len:                预测帧数 (12)
        part_lstm_input_size:    分路 LSTM 输入维度 (16: 嵌入后维度)
        part_lstm_hidden_size:   分路 LSTM 隐藏维度 (32)
        merge_lstm_input_size:   合并 LSTM 输入维度 (64: 32+32)
        merge_lstm_hidden_size:  合并 LSTM 隐藏维度 (64)
        light_input_size:        交通灯输入维度 (4)
        embedding_size:          嵌入中间维度 (32)
        light_embedding_size:    交通灯嵌入输出维度 (16)
    """
    def __init__(
        self,
        obs_len,
        pred_len,
        part_lstm_input_size,       # 16
        part_lstm_hidden_size,      # 32
        merge_lstm_input_size,      # 64 (32 + 32)
        merge_lstm_hidden_size,     # 64
        dropout,
        light_input_size=4,         # 4 维交通灯状态
        embedding_size=32,
        light_embedding_size=16,
    ):
        super(TrajectoryDiscriminator, self).__init__()
        self.light_input_size = light_input_size
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.light_embedding_size = light_embedding_size
        self.embedding_size = embedding_size

        # === 交通灯状态嵌入 MLP (4维 → 16维) ===
        self.light_embedding = nn.Sequential(
            nn.BatchNorm1d(self.light_input_size),
            nn.ReLU(),
            nn.Linear(self.light_input_size, self.embedding_size),    # 4 → 32
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.light_embedding_size), # 32 → 16
            nn.ReLU()
        )

        # === 轨迹位置嵌入 MLP (2维 → 16维) ===
        self.pos_embedding = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # === 合并特征分类 MLP (64维 → 32维 → 1维) ===
        self.merge_embedding = nn.Sequential(
            nn.Linear(64, 32),
            nn.Linear(32, 1),
        )

        self.part_lstm_input_size = part_lstm_input_size      # 16
        self.part_lstm_hidden_size = part_lstm_hidden_size    # 32

        # === 双路 LSTM (Part-LSTM) ===
        # 轨迹位置 LSTM: 输入嵌入后的轨迹位置 (16维) → 隐藏状态 (32维)
        self.pos_part_lstm = nn.LSTMCell(self.part_lstm_input_size, self.part_lstm_hidden_size)
        # 交通灯状态 LSTM: 输入嵌入后的交通灯状态 (16维) → 隐藏状态 (32维)
        self.state_part_lstm = nn.LSTMCell(self.part_lstm_input_size, self.part_lstm_hidden_size)

        self.merge_lstm_input_size = merge_lstm_input_size     # 64 (32 + 32)
        self.merge_lstm_hidden_size = merge_lstm_hidden_size   # 64

        # === 合并 LSTM (Merge-LSTM) ===
        # 输入拼接后的双路隐藏状态 (64维) → 最终隐藏状态 (64维)
        self.merge_lstm = nn.LSTMCell(self.merge_lstm_input_size, self.merge_lstm_hidden_size)

    def init_hidden_part_lstm(self, batch):
        """
        初始化分路 LSTM 的隐藏状态和细胞状态。

        参数:
            batch: batch 大小 (总车辆数 × 场景数)

        返回:
            (h_0, c_0): 隐藏状态和细胞状态, 各形状 (batch, part_lstm_hidden_size=32)
        """
        return (
            torch.randn(batch, self.part_lstm_hidden_size).cuda(),
            torch.randn(batch, self.part_lstm_hidden_size).cuda(),
        )

    def init_hidden_merge_lstm(self, batch):
        """
        初始化合并 LSTM 的隐藏状态和细胞状态。

        返回:
            (h_0, c_0): 隐藏状态和细胞状态, 各形状 (batch, merge_lstm_hidden_size=64)
        """
        return (
            torch.randn(batch, self.merge_lstm_hidden_size).cuda(),
            torch.randn(batch, self.merge_lstm_hidden_size).cuda(),
        )

    def forward(self, traj, state, seq_start_end):
        """
        判别器前向传播。

        执行步骤:
        1. 轨迹位置嵌入: pos_embedding(2维 dx,dy → 16维)
        2. 交通灯状态嵌入: light_embedding(4维状态 → 16维)
        3. 双路 LSTM 编码: 分别对20帧嵌入序列编码
        4. 逐帧合并: 将每帧双路隐藏状态拼接 (32+32=64维)
        5. Merge LSTM: 对拼接特征序列编码
        6. Merge MLP: 64→32→1 得到判别得分

        参数:
            traj:          完整轨迹 (观测+预测) 的相对位移,
                           形状 (obs_len+pred_len, batch, 2)
            state:         对应的交通灯状态序列,
                           形状 (obs_len+pred_len, batch, 4)
            seq_start_end: 场景起止索引 (未直接使用，保留接口一致性)

        返回:
            output: 每条轨迹的判别得分, 形状 (batch, 1)
        """
        batch = traj.shape[1]       # 总车辆数

        # 初始化所有 LSTM 隐藏状态
        traj_lstm_h_t, traj_lstm_c_t = self.init_hidden_part_lstm(batch)
        state_lstm_h_t, state_lstm_c_t = self.init_hidden_part_lstm(batch)
        merge_lstm_h_t, merge_lstm_c_t = self.init_hidden_merge_lstm(batch)

        traj_lstm_hidden_states = []  # 保存每帧轨迹 LSTM 隐藏状态
        state_lstm_hidden_states = [] # 保存每帧状态 LSTM 隐藏状态

        # ---- 第一步: 轨迹位置编码 (共20帧) ----
        for i, input_t in enumerate(traj[:].chunk(traj[:].size(0), dim=0)):
            input_t = self.pos_embedding(input_t.squeeze(0))  # 嵌入: 2→16
            traj_lstm_h_t, traj_lstm_c_t = self.pos_part_lstm(
                input_t.squeeze(0), (traj_lstm_h_t, traj_lstm_c_t))
            traj_lstm_hidden_states += [traj_lstm_h_t]  # 收集隐藏状态 (32维)

        # ---- 第二步: 交通灯状态编码 (共20帧) ----
        for i, input_t in enumerate(state[:].chunk(state[:].size(0), dim=0)):
            input_t = self.light_embedding(input_t.squeeze(0))  # 嵌入: 4→16
            state_lstm_h_t, state_lstm_c_t = self.pos_part_lstm(
                input_t.squeeze(0), (state_lstm_h_t, state_lstm_c_t))
            state_lstm_hidden_states += [state_lstm_h_t]  # 收集隐藏状态 (32维)

        # ---- 第三步: 逐帧合并编码 ----
        for i in range(len(traj_lstm_hidden_states)):
            # 拼接双路隐藏状态: (32+32=64维)
            input_t = torch.cat(
                (traj_lstm_hidden_states[i], state_lstm_hidden_states[i]), dim=1)
            merge_lstm_h_t, merge_lstm_c_t = self.merge_lstm(
                input_t, (merge_lstm_h_t, merge_lstm_c_t))

        # ---- 第四步: MLP 分类得到判别得分 ----
        output = self.merge_embedding(merge_lstm_h_t)  # 64→32→1

        return output

