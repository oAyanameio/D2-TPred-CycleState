"""训练和评估共用的工具函数。"""

import os
import logging
import torch
import random

class AverageMeter(object):
    """记录一个指标的当前值和平均值。"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """把多个指标拼成一行日志。"""
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def set_logger(log_path):
    """同时把日志写到终端和文件。"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def relative_to_abs(rel_traj, start_pos):
    """把相对位移轨迹还原成绝对坐标轨迹。"""
    # batch, seq_len, 2
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)


def get_dset_path(dset_name, dset_type):
    """拼出数据集路径。"""
    _dir = os.path.dirname(__file__)
    # _dir = _dir.split("/")[:-1]
    # _dir = "/".join(_dir)
    return os.path.join(_dir, "datasets", dset_name, dset_type)


def int_tuple(s):
    """把字符串转成整数元组。"""
    return tuple(int(i) for i in s.split(","))


def l2_loss(pred_traj, pred_traj_gt, loss_mask, random=0, mode="average"):
    """计算轨迹重建的 L2 损失。"""
    seq_len, batch, _ = pred_traj.size()
    # equation below , the first part do noing, can be delete

    loss = (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)) ** 2
    if mode == "sum":
        return torch.sum(loss)
    elif mode == "average":
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == "raw":
        return loss.sum(dim=2).sum(dim=1)


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode="sum"):
    """计算 ADE 的累计位移误差。"""

    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)

    loss = loss ** 2
    if consider_ped is not None:
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
    """计算 FDE。"""

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
    """轨迹状态辅助损失。"""
    seq_len, batch, _ = pred_traj_gt.size()
    # equation below , the first part do noing, can be delete
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
    """数值稳定版 BCE。"""
    neg_abs = -input.abs()          # 取负值
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def gan_g_loss(scores_fake):
    """生成器的对抗损失。"""
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2)
    return bce_loss(scores_fake, y_fake)

def gan_d_loss(scores_real, scores_fake):
    """判别器的对抗损失。"""
    # y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)    # 真实数据在1左右
    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)
    y_fake = torch.zeros_like(scores_fake) *random.uniform(0.0, 0.3)     # 预测数据在0左右
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake
  
# print(210.1//1)
