"""数据加载封装。"""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.trajectories import TrajectoryDataset, seq_collate


def seed_worker(worker_id):
    """为每个 DataLoader worker 设置确定性种子，确保数据加载可复现。"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def data_loader(args, path):
    """构建数据集和 DataLoader。"""
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)

    g = torch.Generator()
    if getattr(args, "seed", None) is not None:
        g.manual_seed(args.seed)

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_num_workers,
        # 自定义 collate 函数会把"一个场景内的多目标"拼成模型想要的格式。
        collate_fn=seq_collate,
        pin_memory=getattr(args, "pin_memory", True),
        worker_init_fn=seed_worker,
        generator=g,
    )
    return dset, loader
