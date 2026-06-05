"""D2-TPred 模型定义。

这份实现对应论文 D2-TPred: Discontinuous Dependency for Trajectory Prediction
Under Traffic Lights。模型的核心思想是把车辆轨迹预测拆成三类信息联合建模：
1. 车辆自身历史运动模式。
2. 场景中车辆之间的交互关系。
3. 与交通灯相关的状态约束。

生成器负责输出未来相对位移，判别器负责区分真实轨迹和生成轨迹。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
# from scipy import stats
from utils import relative_to_abs
import math
import numpy as np
import time
from scipy.spatial.distance import pdist, squareform

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
    if noise_type == "gaussian":
        return torch.randn(*shape, device=device)
    elif noise_type == "uniform":
        return torch.rand(*shape, device=device).sub_(0.5).mul_(2.0)
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


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
        """根据距离和方向约束构造关系矩阵。

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
        """
        currdata = curr_dire[:,:,2:4]
        F, N, D = currdata.size()
        # d: 距离门控矩阵，先依据欧氏距离筛掉过远车辆。
        d= np.zeros((F, N, N))
        # r: 最终关系矩阵，同时满足距离与方向约束的边会被置为 1。
        r=np.zeros((F, N, N))
        # 156 是论文实现中使用的邻域半径阈值。
        l = 156


        currdata = currdata.detach().cpu().numpy()
        for cur_f in range(F):
            d[cur_f] = squareform(pdist(currdata[cur_f], metric='euclidean'))
        d = np.where(d<=l,1,0)

        for cur_f in range(F):
            for cur_n in range(N):
                # 第 5 个通道是方向角。以它为中心构造一个前向扇区，只保留更可能
                # 影响当前车辆决策的邻居，体现论文的“不连续依赖”思想。
                a = curr_dire[cur_f, cur_n, 5]
                up = a + 62  #62
                down = a - 62

                for n_neig in range(N):
                    if (d[cur_f, cur_n, n_neig] == 1):
                        dire_n_neig = self.neig_direction(
                            currdata[cur_f, n_neig, 0] - currdata[cur_f, cur_n, 0],
                            currdata[cur_f, n_neig, 1] - currdata[cur_f, cur_n, 1]
                        )
                        if up> 360:
                            if (down <= dire_n_neig <= 360) or (0 <= dire_n_neig <= (up - 360)):
                                r[cur_f, cur_n, n_neig] = 1
                        elif 62 <= up <= 124:
                            if (down + 360 <= dire_n_neig <= 360) or (0 <= dire_n_neig <= up):
                                r[cur_f, cur_n, n_neig] = 1
                        else:
                            r[cur_f, cur_n, n_neig] = 1
        r = torch.tensor(r, dtype=torch.float32, device=curr_dire.device)
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

        for start, end in seq_start_end.data:
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
        for start, end in seq_start_end.data:
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
        # 图交互编码分支：历史上这里原本预留了一个 graph LSTM 用于继续处理图特征。
        # 在当前 forward 主路径中，这个模块没有被实际调用，图时序聚合改由
        # seqGATEncoder 完成，但保留该成员有助于和原始论文/旧版实现对照。
        self.graph_lstm_model = nn.LSTMCell(
            graph_network_out_dims, graph_lstm_hidden_size
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
        self.noise_type = noise_type

        # 解码器：每一步输入上一时刻位移，输出新的隐状态。
        self.pred_lstm_model = nn.LSTMCell(traj_lstm_input_size, self.pred_lstm_hidden_size)

    def init_hidden_traj_lstm(self, batch):
        """初始化轨迹 LSTM 的隐状态。

        Args:
            batch: 当前 mini-batch 中展平后的 agent 总数。

        Returns:
            `(h_0, c_0)`，两者形状均为 `(batch, traj_lstm_hidden_size)`。
        """
        device = get_module_device(self)
        return (
            torch.randn(batch, self.traj_lstm_hidden_size, device=device),
            torch.randn(batch, self.traj_lstm_hidden_size, device=device),
        )

    def init_hidden_graph_lstm(self, batch):
        """初始化图 LSTM 的隐状态。

        Returns:
            `(h_0, c_0)`，形状均为 `(batch, graph_lstm_hidden_size)`。
        """
        device = get_module_device(self)
        return (
            torch.randn(batch, self.graph_lstm_hidden_size, device=device),
            torch.randn(batch, self.graph_lstm_hidden_size, device=device),
        )

    def init_hidden_light_lstm(self, batch):
        """初始化交通灯分支隐状态。

        当前生成器主流程没有显式使用独立的 light LSTM，但保留了初始化接口，
        方便和其它实验分支兼容。
        """
        device = get_module_device(self)
        return (
            torch.randn(batch, self.traj_lstm_hidden_size, device=device),
            torch.randn(batch, self.traj_lstm_hidden_size, device=device),
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

        z_decoder = get_noise(noise_shape, self.noise_type, _input.device)

        _list = []
        for idx, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            _vec = z_decoder[idx].view(1, -1)
            _to_cat = _vec.repeat(end - start, 1)
            _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
        decoder_h = torch.cat(_list, dim=0)

        return decoder_h

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
        with torch.no_grad():
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
                teacher_force = random.random() < teacher_forcing_ratio
                input_t = input_t if teacher_force else output.unsqueeze(0)
                # 注意：图特征和交通灯特征并不是在每个时间步直接作为 LSTM 输入，
                # 而是已经注入到了 pred_lstm_hidden 的初始化里。每一步真正送入
                # pred_lstm_model 的，是上一时刻的位移向量。
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
    ):
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
        #   同车道释放比例, 当前灯态编号, 当前灯态持续时间, 自身到停止线距离
        # ]
        self.queue_feature_dim = 8
        # cycle_feature = [phase one-hot(3), elapsed time, phase change flag]
        self.cycle_feature_dim = 5
        self.queue_lstm_hidden_size = queue_lstm_hidden_size
        self.cycle_lstm_hidden_size = cycle_lstm_hidden_size
        self.queue_speed_threshold = queue_speed_threshold
        self.queue_distance_threshold = queue_distance_threshold
        self.queue_count_norm = queue_count_norm
        self.queue_speed_norm = queue_speed_norm
        self.queue_distance_norm = queue_distance_norm
        self.cycle_time_norm = cycle_time_norm

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
        # 显式辅助预测头：
        # 让 queue/cycle 分支不仅“存在”，还要对可解释的中观/宏观状态负责，
        # 比直接截取 hidden 向量前几维做监督更稳、更易解释。
        self.queue_aux_head = nn.Linear(self.queue_lstm_hidden_size, 3)
        self.cycle_aux_head = nn.Linear(self.cycle_lstm_hidden_size, 4)
        self.debug_last_aux = None

        # 新解码器的初始状态除了微观运动与图交互，还会额外注入：
        # 1. 当前交通灯约束；
        # 2. 车道级 queue-wave 摘要；
        # 3. 信号周期级 cycle memory。
        self.pred_lstm_hidden_size = (
            self.light_embedding_size
            + self.traj_lstm_hidden_size
            + self.graph_lstm_hidden_size
            + self.queue_lstm_hidden_size
            + self.cycle_lstm_hidden_size
            + noise_dim[0]
        )
        self.pred_lstm_model = nn.LSTMCell(
            traj_lstm_input_size, self.pred_lstm_hidden_size
        )
        self.pred_hidden2pos = nn.Linear(
            self.light_embedding_size
            + self.cycle_lstm_hidden_size
            + self.pred_lstm_hidden_size,
            2,
        )

    def init_hidden_queue_lstm(self, batch):
        """初始化 queue memory 的隐状态。"""
        device = get_module_device(self)
        return (
            torch.randn(batch, self.queue_lstm_hidden_size, device=device),
            torch.randn(batch, self.queue_lstm_hidden_size, device=device),
        )

    def init_hidden_cycle_lstm(self, batch):
        """初始化 cycle memory 的隐状态。"""
        device = get_module_device(self)
        return (
            torch.randn(batch, self.cycle_lstm_hidden_size, device=device),
            torch.randn(batch, self.cycle_lstm_hidden_size, device=device),
        )

    def build_cycle_features(self, state_seq):
        """把观测到的信号状态序列转成 cycle memory 的输入特征。

        Args:
            state_seq: `(T, batch, 4)`，通道含义与数据集中的 `obs_state` 一致。

        Returns:
            cycle_feature: `(T, batch, 5)`，包含 one-hot 灯态、持续时间归一化以及
            灯态是否变化的标记。
        """
        phase = state_seq[:, :, 2].long().clamp(min=0, max=2)
        phase_one_hot = F.one_hot(phase, num_classes=3).float()
        elapsed = (state_seq[:, :, 3:4] / self.cycle_time_norm).clamp(min=0.0, max=2.0)
        phase_change = torch.zeros(
            state_seq.size(0), state_seq.size(1), 1, device=state_seq.device
        )
        phase_change[1:] = (phase[1:] != phase[:-1]).float().unsqueeze(2)
        return torch.cat((phase_one_hot, elapsed, phase_change), dim=2)

    def get_step_cycle_feature(self, state_frame):
        """构造单步解码阶段使用的周期状态特征。"""
        phase = state_frame[:, 2].long().clamp(min=0, max=2)
        phase_one_hot = F.one_hot(phase, num_classes=3).float()
        elapsed = (state_frame[:, 3:4] / self.cycle_time_norm).clamp(min=0.0, max=2.0)
        phase_change = torch.zeros(state_frame.size(0), 1, device=state_frame.device)
        return torch.cat((phase_one_hot, elapsed, phase_change), dim=1)

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

        这些量足以让第一版模型学到“排队-释放-未释放”的中观状态差异。
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
                    ],
                    dim=1,
                )

        return queue_features

    def compute_queue_targets(self, queue_feature_seq):
        """把弱标签统计量转成辅助监督目标。"""
        return torch.stack(
            (
                queue_feature_seq[:, :, 0],
                queue_feature_seq[:, :, 3],
                queue_feature_seq[:, :, 4],
            ),
            dim=2,
        )

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

        queue_feature_seq = self.build_queue_features(
            obs_traj_pos, obs_traj_rel, obs_state, seq_start_end
        )
        cycle_feature_seq = self.build_cycle_features(obs_state)
        self.debug_last_aux = {
            "queue_feature_seq": queue_feature_seq.detach(),
            "queue_targets": self.compute_queue_targets(queue_feature_seq).detach(),
            "cycle_feature_seq": cycle_feature_seq.detach(),
            "queue_hidden_last": None,
            "cycle_hidden_last": None,
            "queue_pred_last": None,
            "cycle_pred_last": None,
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
        with torch.no_grad():
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
        encoded_before_noise_hidden = torch.cat(
            (
                light_state_embedding,
                traj_lstm_hidden_states[-1],
                graph_lstm_hidden_states[-1],
                queue_lstm_hidden_states[-1],
                cycle_lstm_hidden_states[-1],
            ),
            dim=1,
        )
        self.debug_last_aux["queue_hidden_last"] = queue_lstm_hidden_states[-1]
        self.debug_last_aux["cycle_hidden_last"] = cycle_lstm_hidden_states[-1]
        self.debug_last_aux["queue_pred_last"] = self.queue_aux_head(
            queue_lstm_hidden_states[-1]
        )
        self.debug_last_aux["cycle_pred_last"] = self.cycle_aux_head(
            cycle_lstm_hidden_states[-1]
        )
        pred_lstm_hidden = self.add_noise(
            encoded_before_noise_hidden, seq_start_end
        )
        pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden)

        obs_traj_rel = obs_traj_rel[:, :, 2:4]
        output = obs_traj_rel[self.obs_len - 1]
        if self.training:
            for i, input_t in enumerate(
                obs_traj_rel[-self.pred_len :].chunk(self.pred_len, dim=0)
            ):
                teacher_force = random.random() < teacher_forcing_ratio
                input_t = input_t if teacher_force else output.unsqueeze(0)
                pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(
                    input_t.squeeze(0), (pred_lstm_hidden, pred_lstm_c_t)
                )
                if i == 0:
                    light_state = self.get_last_state(obs_traj_pos, obs_state)
                    current_cycle_feature = self.get_step_cycle_feature(obs_state[-1])
                else:
                    light_state = self.get_next_state(
                        pred_traj_rel, obs_traj_pos, pred_state
                    )
                    current_cycle_feature = self.get_step_cycle_feature(
                        pred_state[i - 1]
                    )
                light_state_embedding = self.light_embedding(light_state)
                cycle_step_embedding = self.cycle_step_embedding(
                    current_cycle_feature
                )
                pred_input = torch.cat(
                    (light_state_embedding, cycle_step_embedding, pred_lstm_hidden),
                    dim=1,
                )
                output = self.pred_hidden2pos(pred_input)
                pred_traj_rel += [output]
        else:
            for i in range(self.pred_len):
                pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(
                    output, (pred_lstm_hidden, pred_lstm_c_t)
                )
                if i == 0:
                    light_state = self.get_last_state(obs_traj_pos, obs_state)
                    current_cycle_feature = self.get_step_cycle_feature(obs_state[-1])
                else:
                    light_state = self.get_next_state(
                        pred_traj_rel, obs_traj_pos, pred_state
                    )
                    current_cycle_feature = self.get_step_cycle_feature(
                        pred_state[i - 1]
                    )
                light_state_embedding = self.light_embedding(light_state)
                cycle_step_embedding = self.cycle_step_embedding(
                    current_cycle_feature
                )
                pred_input = torch.cat(
                    (light_state_embedding, cycle_step_embedding, pred_lstm_hidden),
                    dim=1,
                )
                output = self.pred_hidden2pos(pred_input)
                pred_traj_rel += [output]

        return torch.stack(pred_traj_rel)


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
            torch.randn(batch, self.part_lstm_hidden_size, device=device),
            torch.randn(batch, self.part_lstm_hidden_size, device=device),
        )

    def init_hidden_merge_lstm(self, batch):
        """初始化融合 LSTM 隐状态。"""
        device = get_module_device(self)
        return (
            torch.randn(batch, self.merge_lstm_hidden_size, device=device),
            torch.randn(batch, self.merge_lstm_hidden_size, device=device),
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
