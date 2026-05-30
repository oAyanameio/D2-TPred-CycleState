"""训练与评估阶段共用的工具函数。

这个文件主要包含三类内容：
1. 日志与进度显示。
2. 轨迹坐标转换与数据路径辅助函数。
3. 损失函数与评价指标。
"""

import os
import logging
import torch
import random

class AverageMeter(object):
    """记录某个标量指标的当前值、累计值和平均值。"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        # 每个 epoch 或每次统计开始前，都把累计状态清零。
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # `n` 一般表示这次更新对应多少个样本，用来做加权平均。
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """把多个 `AverageMeter` 组织成易读的训练日志行。"""

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        # 最终输出形如：Epoch: [3][10/100]    L2_Loss 0.123 (0.456)
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        # 根据总 batch 数动态决定显示宽度，让日志对齐一些。
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def set_logger(log_path):
    """同时把日志写到终端和文件。"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # 文件日志便于训练结束后回看完整过程。
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # 控制台日志便于实时观察训练状态。
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def relative_to_abs(rel_traj, start_pos):
    """把相对位移轨迹还原为绝对坐标轨迹。

    参数：
    - rel_traj: 相对轨迹，形状为 `(seq_len, batch, 2)`。
    - start_pos: 起始绝对位置，形状为 `(batch, 2)`。

    返回：
    - abs_traj: 绝对轨迹，形状仍为 `(seq_len, batch, 2)`。
    """
    # 先把时间维挪到中间，便于沿时间方向做累计和。
    rel_traj = rel_traj.permute(1, 0, 2)
    # cumsum 表示“把每一步位移累加起来”，得到相对起点的总位移。
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    # 再变回模型和评估代码习惯使用的 `(T, B, C)` 排列。
    return abs_traj.permute(1, 0, 2)


def get_dset_path(dset_name, dset_type):
    """拼出数据集目录路径。"""
    _dir = os.path.dirname(__file__)
    return os.path.join(_dir, "datasets", dset_name, dset_type)


def int_tuple(s):
    """把形如 `16,32` 的字符串解析成整数元组。"""
    return tuple(int(i) for i in s.split(","))


def l2_loss(pred_traj, pred_traj_gt, loss_mask, random=0, mode="average"):
    """计算预测轨迹与真实轨迹之间的逐点平方误差。

    参数：
    - pred_traj: 预测轨迹，形状 `(seq_len, batch, 2)`。
    - pred_traj_gt: 真实轨迹，形状 `(seq_len, batch, 2)`。
    - loss_mask: 有效时间步掩码，形状 `(batch, seq_len)`。
    - mode:
      - `sum`: 返回所有元素总和。
      - `average`: 返回按 mask 元素数归一化后的平均损失。
      - `raw`: 返回每个样本的损失。
    """
    seq_len, batch, _ = pred_traj.size()
    loss = (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)) ** 2
    if mode == "sum":
        return torch.sum(loss)
    elif mode == "average":
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == "raw":
        return loss.sum(dim=2).sum(dim=1)


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode="sum"):
    """计算 ADE 对应的逐步位移误差累计值。

    这里的实现会先在每个时间步上求欧氏距离，再对时间维求和。
    如果 `mode="raw"`，返回的是“每个样本一条轨迹”的累计误差。
    """

    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)

    loss = loss ** 2
    if consider_ped is not None:
        # 有些场景会只统计特定目标，这里用掩码控制。
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == "sum":
        return torch.sum(loss)
    elif mode == "mean":
        return torch.mean(loss)
    elif mode == "raw":
        return loss


def final_displacement_error(pred_pos, pred_pos_gt, consider_ped=None, mode="sum"):
    """计算 FDE，对比最后一个时间步的位置误差。"""

    loss = pred_pos_gt - pred_pos
    loss = loss ** 2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == "raw":
        return loss
    else:
        return torch.sum(loss)

def state_loss(pred_traj_fake,pred_traj_gt,mode='sum'):
    """计算状态分支使用的位置回归损失。

    这个函数只取输入张量中的 `x, y` 两个位置通道来比较。
    项目当前训练主流程并没有显式使用它，但保留了这部分实现。
    """
    seq_len, batch, _ = pred_traj_gt.size()
    pred_gt=pred_traj_gt[:,:,2:4]   # T V C
    loss = (pred_gt.permute(1, 0, 2) - pred_traj_fake.permute(1, 0, 2)) ** 2    # V T C
    x_= (pred_gt.permute(1, 0, 2)[:,-1,0] - pred_traj_fake.permute(1, 0, 2)[:,-1,0])
    if mode == "sum":
        return torch.sum(loss)
    elif mode == "average":
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == "raw":
        return loss.sum(dim=2).sum(dim=1)

def bce_loss(input, target):
    """数值稳定版本的二元交叉熵。

    这里没有直接调用 `nn.BCEWithLogitsLoss`，而是手动展开公式。
    """
    neg_abs = -input.abs()          # 先取绝对值再取反，便于稳定计算 log 项
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def gan_g_loss(scores_fake):
    """生成器对抗损失。

    给假样本打上接近 1 的软标签，鼓励判别器把生成结果看成真样本。
    """
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2)
    return bce_loss(scores_fake, y_fake)

def gan_d_loss(scores_real, scores_fake):
    """判别器对抗损失。

    - 真样本使用接近 1 的软标签。
    - 假样本使用接近 0 的软标签。

    这种 label smoothing 可以稍微缓和 GAN 训练的不稳定性。
    """
    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)
    y_fake = torch.zeros_like(scores_fake) *random.uniform(0.0, 0.3)     # 预测数据在0左右
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake
