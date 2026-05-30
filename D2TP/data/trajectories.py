"""轨迹数据集定义与 batch 拼接逻辑。

这个文件是整个项目的数据入口，负责：
1. 从原始 txt 文件读取轨迹。
2. 按观测长度与预测长度切分样本窗口。
3. 构造相对位移、交通灯状态以及 batch 内场景边界。
"""

import logging
import os
import math
import numpy as np

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

def agent_direction(rdata):
    """为每个目标补充朝向角度信息。

    参数：
    - rdata: 形状 `(T, N, C)` 的轨迹张量。

    返回：
    - endata: 在原始特征末尾额外加 1 维朝向角后的张量。

    这里通过相邻两帧的位置差 `(dx, dy)` 估计运动方向，
    并把角度统一映射到 `[0, 360)` 区间。
    """
    a, b, c = rdata.size()
    endata = torch.zeros(a,b,c+1)
    endata[:,:,0:c] = rdata[:, :, 0:c]

    for n in range(b):
        for f in range(a-1):
            # 第 2、3 通道存的是 x/y 坐标，因此用它们计算方向角。
            difx=rdata[f+1,n,2]-rdata[f,n,2]
            dify = rdata[f + 1, n, 3] - rdata[f, n, 3]
            if difx != 0:
                dire = 180 * math.atan2(dify, difx) / (math.pi)
                if dire < 0:
                    dire = 360 + dire
                endata[f,n,c] = dire
            else:
                if dify > 0:
                    dire = 90
                elif dify < 0:
                    dire = 270
                else:
                    dire = 0
                endata[f,n,c] = dire
        # 最后一帧没有下一帧可供求差分，因此直接复用前一帧方向。
        endata[a-1, n, c]=endata[a-2, n, c]
    return endata

def seq_collate(data):
    """把 `Dataset.__getitem__` 返回的多个场景样本拼成一个 batch。

    这里的关键点是：一个样本里可能包含多个目标，DataLoader 收到的
    `data` 是“场景列表”，而模型需要的是把所有目标拼成一个大 batch，
    同时额外记录每个场景在这个大 batch 中的起止下标。
    """
    (
        obs_seq_list,   # VCT
        pred_seq_list,
        obs_seq_rel_list,
        pred_seq_rel_list,
        obs_state_list,  # VST
        pred_state_lsit, # VST
        non_linear_ped_list,
        loss_mask_list,
    ) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [
        # 每个元组表示一个原始场景在“拼接后目标维”中的左右边界。
        [start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])
    ]

    # 原始样本的维度是 `(num_agent, channel, time)`，
    # 模型更习惯 `(time, total_agent, channel)`，所以这里统一转置。
    obs_traj0 = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)  # T NV C
    obs_traj = agent_direction(obs_traj0)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    obs_state=torch.cat(obs_state_list, dim=0).permute(2, 0, 1) # VST-T NV S
    pred_state=torch.cat(pred_state_lsit, dim=0).permute(2, 0, 1) # VST-T NV S
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj,
        pred_traj,
        obs_traj_rel,
        pred_traj_rel,
        obs_state,
        pred_state,
        non_linear_ped,
        loss_mask,
        seq_start_end,
    ]

    return tuple(out)


def read_file(_path, delim="\t"):
    """读取单个文本轨迹文件并转成 `numpy.ndarray`。"""
    data = []
    if delim == "tab":
        delim = "\t"
    elif delim == "space":
        delim = " "
    with open(_path, "r") as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """通过二次多项式拟合误差判断轨迹是否为非线性。

    这个函数是行人轨迹预测项目里常见的辅助逻辑。
    当前仓库保留了它，但主流程里没有真正使用到非线性标签。
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    """轨迹预测数据集。

    每个样本代表一个时间窗口内的一个场景，场景中可包含多个目标。
    一个目标的原始通道数为 9，通常可理解为：
    - 0: frame_id
    - 1: agent_id
    - 2,3: x,y 坐标
    - 其余通道：数据集中附带的场景或信号状态特征

    注意：这里的具体字段语义来自仓库当前数据格式，而不是标准公开数据集格式。
    """

    def __init__(
        self,
        data_dir,
        obs_len=8,
        pred_len=12,
        skip=1,
        threshold=0.002,
        min_ped=1,
        delim="\t",
    ):
        """初始化并预处理整个数据集。"""
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            # 每个文件对应一段场景轨迹记录。
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            # 原作者在这里每隔 9 帧取一个时间点，相当于做降采样。
            frames=[frames[i] for i in range(0,len(frames),9)]

            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                # 取出一个长度为 `obs_len + pred_len` 的滑动窗口。
                curr_seq_data = np.concatenate(
                    frame_data[idx : idx + self.seq_len], axis=0
                )
                # 找出该窗口内出现过的所有目标 ID。
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                # 每个目标保留 9 个通道，时间长度为 `self.seq_len`。
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 9, self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 9, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []

                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    # 计算该目标在当前窗口中的起止位置。
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1

                    # 如果目标没有完整覆盖整个窗口，则直接丢弃。
                    if pad_end - pad_front != self.seq_len:
                        continue
                    if curr_ped_seq.shape[0]<self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 0:])
                    curr_ped_seq = curr_ped_seq
                    # 构造相对位移版本，只对坐标通道做差分。
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[2:4, 1:] = curr_ped_seq[2:4, 1:] - curr_ped_seq[2:4, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # 这里本来可以顺带标记线性/非线性轨迹，当前仓库没有启用。
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    # 只保留目标数足够的场景窗口。
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)
        # 下面把 numpy 数据全部转成 torch.Tensor，方便后续直接进模型。
        self.obs_traj = torch.from_numpy(seq_list[:, :, : self.obs_len]).type(
            torch.float
        )
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len :]).type(
            torch.float
        )
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, : self.obs_len]).type(
            torch.float
        )
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len :]).type(
            torch.float
        )
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]


        # 预定义四个交通灯在地图中的绝对坐标。
        self.light_pos=torch.tensor([[522,599],[1063,860],[1381,489],[940,208]])    # 4 * 2  light pos x y
        # 每种灯态持续的上限时间，后面用于预测未来状态。
        self.time_top=torch.tensor([38,47,2])
        # `obs_state/pred_state` 每个时间步都由 4 维构成：
        # [traffic_light_x, traffic_light_y, light_phase, phase_elapsed]
        self.obs_state=torch.zeros(self.obs_traj.size(0),4,self.obs_len)
        # 前两维存交通灯位置。
        self.obs_state[:,:2,:]=self.light_pos[ self.obs_traj[:,4,0].long() ].unsqueeze(dim=2)
        # 后两维直接拷贝轨迹数据中已有的状态特征。
        self.obs_state[:,2,:]=self.obs_traj[:,7,:]
        self.obs_state[:,3,:]=self.obs_traj[:,8,:]

        self.pred_state=torch.zeros(self.obs_traj.size(0),4,self.pred_len)
        # 未来时刻默认沿用同一个交通灯位置。
        self.pred_state[:,:2,:]=self.obs_state[:,:2,0].unsqueeze(dim=2)
        # 统计观测阶段末尾连续保持某种状态的时长。
        all_state_step=[torch.unique(self.obs_state[i,3,:],return_counts=True,sorted= False)[1][-1] for i in range (self.obs_traj.size(0))]

        all_state_step=torch.stack(all_state_step)  # NV 1
        # 下面这组运算沿用了原作者的状态推进规则：
        # 通过一个近似步长把观测末尾的相位计数映射到未来时间推进量。
        fstep=(1+all_state_step)/3.1
        fstep=fstep.float()
        fstep = fstep.floor()
        fstep=fstep.long()

        for ped in range(self.pred_state.size(0)):
            # 如果当前累计时长已经超过该状态允许的上限，就切到下一种灯态。
            if self.obs_state[ped,3,-1]+fstep[ped]>\
                    self.time_top[int(self.pred_state[ped, 2, 0])].float():
                self.pred_state[ped,2,0]=(self.pred_state[ped,2,0]+2)%3
                self.pred_state[ped,3,0]=0
            else: 
                self.pred_state[ped,3,0]=self.pred_state[ped,3,0]+fstep[ped]
                

        for f in range(1,self.pred_len):
            # 逐步递推未来每一帧的灯态与已持续时长。
            all_state_step=torch.round(((1+all_state_step)%3.1).float())
            step=(1+all_state_step)/3.1
            step = step.floor()

            for ped in range(self.pred_state.size(0)):
                # 超过该灯态的最大持续时间就切换到下一灯态，否则继续累计时间。
                if self.pred_state[ped,3,f-1]+step[ped]>self.time_top[int(self.obs_state[ped,2,-1])].float():
                    self.pred_state[ped,2,f]=(self.pred_state[ped,2,f]+2)%3
                    self.pred_state[ped,3,f]=0
                else:
                    self.pred_state[ped,3,f]=self.pred_state[ped,3,f]+step[ped]


        


    def __len__(self):
        # DataLoader 通过它知道数据集一共有多少个场景窗口样本。
        return self.num_seq

    def __getitem__(self, index):
        # 取出一个场景窗口中所有目标的观测轨迹、预测轨迹及辅助状态。
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :],
            self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :],
            self.pred_traj_rel[start:end, :],
            self.obs_state[start:end, :],
            self.pred_state[start:end, :],
            self.non_linear_ped[start:end],
            self.loss_mask[start:end, :],
        ]
        return out
