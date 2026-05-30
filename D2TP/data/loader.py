"""DataLoader 封装。

这个文件很薄，只负责把数据集类和 PyTorch 的 DataLoader 组装起来，
让训练、评估、可视化脚本都用同一套数据读取逻辑。
"""

from torch.utils.data import DataLoader

from data.trajectories import TrajectoryDataset, seq_collate


def data_loader(args, path):
    """根据命令行参数构建数据集和 DataLoader。

    参数说明：
    - args: 训练/评估脚本解析得到的参数对象。
    - path: 当前要读取的数据目录。

    返回：
    - dset: `TrajectoryDataset` 实例，内部已经把轨迹切成样本。
    - loader: 按 batch 返回样本的 PyTorch DataLoader。
    """
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        # 原仓库这里保持 `False`，意味着按数据集原有顺序取样。
        shuffle=False,
        num_workers=args.loader_num_workers,
        # 自定义拼接函数负责把“每个场景含多个目标”的样本整理成模型输入。
        collate_fn=seq_collate,
        # pin_memory=True 可以在使用 GPU 时加快主机内存到显存的拷贝。
        pin_memory=True)
    return dset, loader
