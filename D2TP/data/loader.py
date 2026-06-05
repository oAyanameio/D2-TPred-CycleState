"""数据加载封装。"""

from torch.utils.data import DataLoader

from data.trajectories import TrajectoryDataset, seq_collate


def data_loader(args, path):
    """构建数据集和 DataLoader。"""
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_num_workers,
        # 自定义 collate 函数会把“一个场景内的多目标”拼成模型想要的格式。
        collate_fn=seq_collate,
        pin_memory=getattr(args, "pin_memory", True))
    return dset, loader
