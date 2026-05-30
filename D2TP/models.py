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

def get_noise(shape, noise_type):
    """生成噪声，用于提升未来轨迹采样的多样性。"""
    if noise_type == "gaussian":
        return torch.randn(*shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


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
        """对每个节点在关系约束下做多头注意力聚合。"""
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
        attn = torch.mul(Relation.unsqueeze(1).repeat(1, self.n_head, 1, 1).cuda(), attn)
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
        """在时间窗口内做图注意力聚合。"""
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

        self.norm_list = [
            torch.nn.InstanceNorm1d(32).cuda(),
            torch.nn.InstanceNorm1d(64).cuda(),
        ]

    def forward(self, x, Relation):
        """逐层堆叠图注意力，输出最终的空间交互表征。"""
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

        self.norm_list = [
            torch.nn.InstanceNorm1d(32).cuda(),
            torch.nn.InstanceNorm1d(64).cuda(),
        ]

    def forward(self, x):
        """对时间窗口内的图特征进一步聚合。"""
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
        """计算相邻两目标之间的方向角。"""
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
        """
        currdata = curr_dire[:,:,2:4]
        F, N, D = currdata.size()
        d= np.zeros((F, N, N))
        r=np.zeros((F, N, N))
        l = 156


        currdata = currdata.cuda().data.cpu().numpy()
        for cur_f in range(F):
            d[cur_f] = squareform(pdist(currdata[cur_f], metric='euclidean'))
        d = np.where(d<=l,1,0)

        for cur_f in range(F):
            for cur_n in range(N):
                # 第 5 个通道是方向角。
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
        r = torch.FloatTensor(r)
        return r


    def forward(self, obs_traj_embedding, seq_start_end, obs_dire):
        """按场景分组做图编码，避免不同场景之间互相串扰。"""
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
        """按场景分组，对局部时间窗内的图特征继续编码。"""
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
        # 图交互编码分支：把图特征进一步映射成时序隐藏状态。
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
        """初始化轨迹 LSTM 的隐状态。"""
        return (
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
        )

    def init_hidden_graph_lstm(self, batch):
        """初始化图 LSTM 的隐状态。"""
        return (
            torch.randn(batch, self.graph_lstm_hidden_size).cuda(),
            torch.randn(batch, self.graph_lstm_hidden_size).cuda(),
        )

    def init_hidden_light_lstm(self, batch):
        """初始化交通灯分支隐状态。"""
        return (
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
        )

    def add_noise(self, _input, seq_start_end):
        """按场景给编码特征拼接噪声。"""
        noise_shape = (seq_start_end.size(0),) + self.noise_dim

        z_decoder = get_noise(noise_shape, self.noise_type)

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
        """
        从最后一帧观测状态里构造交通灯条件特征。
        """

        dis = torch.sqrt((obs_traj_pos[-1,:,2]-obs_state[-1,:,0])**2 + (obs_traj_pos[-1,:,3] - obs_state[-1,:,1])**2)
        disx = obs_traj_pos[-1, :, 2] - obs_state[-1,:,0]
        disy = obs_traj_pos[-1, :, 3] - obs_state[-1,:,1]
        light_state=obs_state[-1,:,2:4]
        dis_state=torch.stack([dis,disx,disy],dim=1)
        state_last=torch.cat((dis_state,light_state),dim=1)

        return state_last

    def get_next_state(self,pred_traj_rel,obs_traj_pos,pred_state):
        """
        用已经生成的相对轨迹递推出下一时刻交通灯条件。
        """
        pred_traj_rel = torch.stack(pred_traj_rel)
        step = pred_traj_rel.size(0)

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
        """
        batch = obs_traj_rel.shape[1]
        traj_lstm_h_t, traj_lstm_c_t = self.init_hidden_traj_lstm(batch)
        pred_traj_rel = []
        traj_lstm_hidden_states = []
        graph_lstm_hidden_states = []

        # 1) 逐帧编码观测轨迹。
        for i, input_t in enumerate(
            obs_traj_rel[: self.obs_len].chunk(obs_traj_rel[: self.obs_len].size(0), dim=0)):
            inputtraj = input_t[:, :, 2:4]
            traj_lstm_h_t, traj_lstm_c_t = self.traj_lstm_model(
                inputtraj.squeeze(0), (traj_lstm_h_t, traj_lstm_c_t))
            traj_lstm_hidden_states += [traj_lstm_h_t]

        # 2) 用 GAT 建模车辆之间的空间交互。
        kl = 6
        obs_dire = obs_traj_pos[:, :, 0:6]
        obs_dire[:, :, 5] = obs_traj_pos[:, :, 9]
        graph_lstm_input = self.gatencoder(
            torch.stack(traj_lstm_hidden_states), seq_start_end, obs_dire
        )
        staend = torch.zeros((1, 2), dtype=torch.int)

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
                graph_lstm_hidden_states += [graph_inter_input[:, -1, :]]

        # 4) 取最后一帧的交通灯状态，与运动特征拼接。
        light_state = self.get_last_state(obs_traj_pos, obs_state)
        light_state_embedding = self.light_embedding(light_state)
        encoded_before_noise_hidden = torch.cat(
            (light_state_embedding, traj_lstm_hidden_states[-1], graph_lstm_hidden_states[-1]),
            dim=1)

        # 5) 场景级噪声注入，形成多模态解码起点。
        pred_lstm_hidden = self.add_noise(
            encoded_before_noise_hidden, seq_start_end
        )
        pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden).cuda()
        obs_traj_rel = obs_traj_rel[:, :, 2:4]
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
                pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(
                    input_t.squeeze(0), (pred_lstm_hidden, pred_lstm_c_t)  # 136
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
        """初始化分支 LSTM 隐状态。"""
        return (
            torch.randn(batch, self.part_lstm_hidden_size).cuda(),
            torch.randn(batch, self.part_lstm_hidden_size).cuda(),
        )

    def init_hidden_merge_lstm(self, batch):
        """初始化融合 LSTM 隐状态。"""
        return (
            torch.randn(batch, self.merge_lstm_hidden_size).cuda(),
            torch.randn(batch, self.merge_lstm_hidden_size).cuda(),
        )

    def forward(self, traj, state, seq_start_end):
        """判别一段完整轨迹的真实性。"""
        batch = traj.shape[1]
        traj_lstm_h_t, traj_lstm_c_t = self.init_hidden_part_lstm(batch)
        state_lstm_h_t, state_lstm_c_t = self.init_hidden_part_lstm(batch)
        merge_lstm_h_t, merge_lstm_c_t = self.init_hidden_merge_lstm(batch)
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
            input_t = torch.cat((traj_lstm_hidden_states[i], state_lstm_hidden_states[i]), dim=1)
            merge_lstm_h_t, merge_lstm_c_t = self.merge_lstm(
                input_t, (merge_lstm_h_t, merge_lstm_c_t)
            )
        output = self.merge_embedding(merge_lstm_h_t)
        return output
