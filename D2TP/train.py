"""训练脚本。

训练阶段采用生成器 + 判别器的对抗式学习，同时用 best-of-K 轨迹重建约束生成器。
"""

import argparse
import logging
import os
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gc
from tensorboardX import SummaryWriter
import utils
from data.loader import data_loader
from models import (
    TrajectoryGenerator,
    TrajectoryDiscriminator,
    CycleStateTrajectoryGenerator,
)
from utils import (
    displacement_error,
    final_displacement_error,
    get_dset_path,
    int_tuple,
    l2_loss,
    relative_to_abs,
    state_loss,
    bce_loss,
    gan_g_loss,
    gan_d_loss,
)

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", default="./", help="Directory containing logging file")

parser.add_argument("--dataset_name", default="VTP_C", type=str)
parser.add_argument("--delim", default="\t")
parser.add_argument("--loader_num_workers", default=4, type=int)
parser.add_argument("--obs_len", default=8, type=int)
parser.add_argument("--pred_len", default=12, type=int)
parser.add_argument("--skip", default=1, type=int)

parser.add_argument("--seed", type=int, default=72, help="Random seed.")
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--num_epochs", default=150, type=int)

parser.add_argument("--noise_dim", default=(16,), type=int_tuple)
parser.add_argument("--noise_type", default="gaussian")

parser.add_argument(
    "--traj_lstm_input_size", type=int, default=2, help="traj_lstm_input_size"
)
parser.add_argument("--traj_lstm_hidden_size", default=32, type=int)

parser.add_argument(
    "--heads", type=str, default="4,1", help="Heads in each layer, splitted with comma"
)
parser.add_argument(
    "--hidden-units",
    type=str,
    default="16",
    help="Hidden units in each hidden layer, splitted with comma",
)
parser.add_argument(
    "--graph_network_out_dims",
    type=int,
    default=32,
    help="dims of every node after through GAT module",
)
parser.add_argument("--graph_lstm_hidden_size", default=32, type=int)
parser.add_argument(
    "--model_type",
    default="d2tpred",
    choices=["d2tpred", "cyclestate"],
    help="选择训练的生成器类型。",
)

parser.add_argument(
    "--dropout", type=float, default=0, help="Dropout rate (1 - keep probability)."
)
parser.add_argument(
    "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
)


parser.add_argument(
    "--lr",
    default=1e-3,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)

parser.add_argument("--best_k", default=20, type=int)
parser.add_argument("--print_every", default=10, type=int)
parser.add_argument(
    "--max_train_batches",
    default=0,
    type=int,
    help="仅用于快速实验。大于 0 时，每个 epoch 最多训练这么多 batch。",
)
parser.add_argument(
    "--max_val_batches",
    default=0,
    type=int,
    help="仅用于快速实验。大于 0 时，验证时最多评估这么多 batch。",
)
parser.add_argument(
    "--generator_only",
    action="store_true",
    help="只训练生成器，不更新判别器，用于新模型早期稳定训练。",
)
parser.add_argument(
    "--gan_weight",
    default=1000.0,
    type=float,
    help="生成器对抗损失的缩放系数。",
)
parser.add_argument(
    "--aux_queue_weight",
    default=0.0,
    type=float,
    help="CycleState 的 queue-state 辅助损失权重。",
)
parser.add_argument(
    "--aux_cycle_weight",
    default=0.0,
    type=float,
    help="CycleState 的 cycle-state 辅助损失权重。",
)
parser.add_argument("--use_gpu", default=1, type=int)
parser.add_argument("--gpu_num", default="2", type=str)
CUDA_VISIBLE_DEVICES = '2'
parser.add_argument(
    "--device",
    default="cuda",
    choices=["cuda", "cpu"],
    help="训练设备。选择 cuda 时会在可用 GPU 上运行。",
)
parser.add_argument(
    "--pin_memory",
    action="store_true",
    help="DataLoader 是否启用 pin_memory。GPU 训练时建议打开。",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)


best_ade = 100


def maybe_load_compatible_weights(model, state_dict):
    """尽量复用旧 checkpoint 中与当前模型形状兼容的参数。"""
    model_state = model.state_dict()
    compatible_state = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible_state[key] = value
        else:
            skipped.append(key)
    model_state.update(compatible_state)
    model.load_state_dict(model_state)
    return skipped


def main(args):
    """训练入口。"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_path = get_dset_path(args.dataset_name, "train")
    val_path = get_dset_path(args.dataset_name, "test")

    logging.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)
    logging.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)

    writer = SummaryWriter()

    n_units = (
        [args.traj_lstm_hidden_size]
        + [int(x) for x in args.hidden_units.strip().split(",")]
        + [args.graph_lstm_hidden_size]
    )

    n_heads = [int(x) for x in args.heads.strip().split(",")]

    model_cls = (
        CycleStateTrajectoryGenerator
        if args.model_type == "cyclestate"
        else TrajectoryGenerator
    )
    model = model_cls(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        traj_lstm_input_size=args.traj_lstm_input_size,
        traj_lstm_hidden_size=args.traj_lstm_hidden_size,
        n_units=n_units,
        n_heads=n_heads,
        graph_network_out_dims=args.graph_network_out_dims,
        dropout=args.dropout,
        alpha=args.alpha,
        graph_lstm_hidden_size=args.graph_lstm_hidden_size,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
    )
    model.to(args.device)
    # 判别器用于判断生成轨迹是否像真实数据。
    Discriminator=TrajectoryDiscriminator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        part_lstm_input_size=16,
        part_lstm_hidden_size=32,
        merge_lstm_input_size=64,
        merge_lstm_hidden_size=64,
        dropout=0.1,
        light_input_size=4,
        embedding_size=32,
        light_embedding_size=16,
    )
    Discriminator.to(args.device)

    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
    optimizer_d = optim.RMSprop(Discriminator.parameters(), lr=1e-3)
    global best_ade
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("Restoring from checkpoint {}".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=args.device)
            if args.model_type == "cyclestate":
                skipped_keys = maybe_load_compatible_weights(
                    model, checkpoint["state_dict"]
                )
                logging.info(
                    "=> warm-started CycleState from checkpoint, skipped {} keys and kept start_epoch={}".format(
                        len(skipped_keys), args.start_epoch
                    )
                )
            else:
                args.start_epoch = checkpoint["epoch"]
                model.load_state_dict(checkpoint["state_dict"])
            logging.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    training_step = 3
    # 先多更新几步判别器，再更新一次生成器。
    D_step=2
    for epoch in range(args.start_epoch, args.num_epochs + 1):
        gc.collect() 
        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                logging.info(
                    "Reached max_train_batches=%d, stop current epoch early.",
                    args.max_train_batches,
                )
                break
            if args.generator_only:
                train(
                    args,
                    len(train_loader),
                    model,
                    batch_idx,
                    batch,
                    Discriminator,
                    optimizer,
                    epoch,
                    training_step,
                    writer,
                )
                if batch_idx % args.print_every == 0:
                    ade = validate(args, model, val_loader, epoch, writer)
                    is_best = ade < best_ade
                    best_ade = min(ade, best_ade)
                    save_checkpoint(
                        {
                            "epoch": epoch + 1,
                            "state_dict": model.state_dict(),
                            "best_ade": best_ade,
                            "optimizer": optimizer.state_dict(),
                        },
                        is_best,
                        f"./checkpoint/checkpoint{epoch}.pth.tar",
                    )
                continue
            if D_step>0:
                D_train(args,len(train_loader), model,batch_idx,batch,Discriminator, optimizer_d, epoch, training_step, writer)
                D_step=D_step-1
            else:
                train(args,len(train_loader), model, batch_idx,batch,Discriminator, optimizer, epoch, training_step, writer)
                D_step=2
                if batch_idx % args.print_every == 0:
                    ade = validate(args, model, val_loader, epoch, writer)
                    is_best = ade < best_ade
                    best_ade = min(ade, best_ade)

                    save_checkpoint(
                        {
                            "epoch": epoch + 1,
                            "state_dict": model.state_dict(),
                            "best_ade": best_ade,
                            "optimizer": optimizer.state_dict(),
                        },
                        is_best,
                        f"./checkpoint/checkpoint{epoch}.pth.tar",
                    )
    writer.close()


def train(args,lens, model,batch_idx,batch,Discriminator, optimizer, epoch, training_step, writer):
    """更新生成器。"""
    losses = utils.AverageMeter("L2_Loss", ":.6f")
    g_losses = utils.AverageMeter("G_Loss", ":.6f")
    aux_queue_losses = utils.AverageMeter("QAux", ":.6f")
    aux_cycle_losses = utils.AverageMeter("CAux", ":.6f")
    progress = utils.ProgressMeter(
        lens, [losses, g_losses, aux_queue_losses, aux_cycle_losses], prefix="Epoch: [{}]".format(epoch)
    )
    model.train()
    batch = [tensor.to(args.device) for tensor in batch]
    (
        obs_traj,
        pred_traj_gt,
        obs_traj_rel,
        pred_traj_gt_rel,
        obs_state,
        pred_state,
        non_linear_ped,
        loss_mask,
        seq_start_end,
    ) = batch

    optimizer.zero_grad()
    predtrajgt = pred_traj_gt[:, :, 2:4]
    L2_loss = torch.zeros(1).to(predtrajgt)
    l2_loss_rel = []
    loss_mask = loss_mask[:, args.obs_len :]

    model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)
    for _ in range(args.best_k):
        # best-of-K：多次采样，保留误差最小的那次。
        pred_traj_fake_rel = model(model_input, obs_traj, obs_state, pred_state, seq_start_end, 0)  # ？
        modinput = model_input[:, :, 2:4]
        l2_loss_rel.append(
            l2_loss(
                pred_traj_fake_rel,
                modinput[-args.pred_len:],
                loss_mask,
                mode="raw",
            )
        )

    pred_traj_fake = torch.cat((obs_traj[:,:,2:4],relative_to_abs(pred_traj_fake_rel, obs_traj[-1,:,2:4])),dim=0)
    traj_state=torch.cat((obs_state,pred_state),dim=0)
    if args.generator_only:
        g_loss = torch.zeros(1).to(predtrajgt)
    else:
        fakesocre=Discriminator(pred_traj_fake,traj_state,seq_start_end)
        g_loss=gan_g_loss(fakesocre)
    l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    l2_loss_rel = torch.stack(l2_loss_rel, dim=1)

    for start, end in seq_start_end.data:
        _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
        _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)
        _l2_loss_rel = torch.min(_l2_loss_rel) / (
            (pred_traj_fake_rel.shape[0]) * (end - start)
        )
        l2_loss_sum_rel += _l2_loss_rel

    L2_loss += l2_loss_sum_rel
    aux_queue_loss = torch.zeros(1).to(predtrajgt)
    aux_cycle_loss = torch.zeros(1).to(predtrajgt)
    if args.model_type == "cyclestate" and hasattr(model, "debug_last_aux"):
        aux_info = model.debug_last_aux
        if args.aux_queue_weight > 0 and aux_info["queue_pred_last"] is not None:
            queue_head = aux_info["queue_pred_last"]
            queue_target = aux_info["queue_targets"][-1]
            aux_queue_loss = torch.mean((queue_head - queue_target) ** 2)
        if args.aux_cycle_weight > 0 and aux_info["cycle_pred_last"] is not None:
            cycle_head = aux_info["cycle_pred_last"]
            cycle_target = aux_info["cycle_feature_seq"][-1][:, :4]
            aux_cycle_loss = torch.mean((cycle_head - cycle_target) ** 2)
    losses.update(L2_loss.item(), obs_traj.shape[1])
    g_losses.update(g_loss.item(),obs_traj.shape[1])
    aux_queue_losses.update(aux_queue_loss.item(), obs_traj.shape[1])
    aux_cycle_losses.update(aux_cycle_loss.item(), obs_traj.shape[1])
    total_loss = (
        L2_loss
        + g_loss * args.gan_weight
        + aux_queue_loss * args.aux_queue_weight
        + aux_cycle_loss * args.aux_cycle_weight
    )
    total_loss.backward()
    optimizer.step()
    if batch_idx % args.print_every == 0:
        progress.display(batch_idx)
    writer.add_scalar("g_l2_loss", losses.avg, batch_idx)
    writer.add_scalar("g_ad_loss", g_losses.avg * args.gan_weight, batch_idx)
    writer.add_scalar("g_queue_aux_loss", aux_queue_losses.avg, batch_idx)
    writer.add_scalar("g_cycle_aux_loss", aux_cycle_losses.avg, batch_idx)

def D_train(args,lens, model,batch_idx,batch,Discriminator, optimizer, epoch, training_step, writer):
    """更新判别器。"""
    D_losses = utils.AverageMeter("D_Loss", ":.6f")
    progress = utils.ProgressMeter(
        lens,[D_losses], prefix="Epoch: [{}]".format(epoch)
    )
    model.train()
    batch = [tensor.to(args.device) for tensor in batch]
    (
        obs_traj,
        pred_traj_gt,
        obs_traj_rel,
        pred_traj_gt_rel,
        obs_state,
        pred_state,
        non_linear_ped,
        loss_mask,
        seq_start_end,
    ) = batch
    optimizer.zero_grad()

    model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)
    pred_traj_fake_rel = model(model_input, obs_traj, obs_state, pred_state, seq_start_end, 0)  # ？

    pred_traj_fake = torch.cat((obs_traj[:,:,2:4],relative_to_abs(pred_traj_fake_rel, obs_traj[-1,:,2:4])),dim=0)
    traj_state=torch.cat((obs_state,pred_state),dim=0)
    traj_real=torch.cat((obs_traj[:,:,2:4],pred_traj_gt[:,:,2:4]),dim=0)
    fakesocre=Discriminator(pred_traj_fake,traj_state,seq_start_end)
    realsocre=Discriminator(traj_real,traj_state,seq_start_end)
    D_loss=gan_d_loss(realsocre,fakesocre)

    D_losses.update(D_loss.item(), obs_traj.shape[1])
    D_loss.backward()
    optimizer.step()
    if batch_idx % args.print_every == 0:
        progress.display(batch_idx)

    writer.add_scalar("d_train_loss", D_losses.avg, epoch)

def validate(args, model, val_loader, epoch, writer):
    ade = utils.AverageMeter("ADE", ":.6f")
    fde = utils.AverageMeter("FDE", ":.6f")
    # progress = utils.ProgressMeter(len(val_loader), [ade, fde], prefix="Test: ")

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if args.max_val_batches > 0 and i >= args.max_val_batches:
                logging.info(
                    "Reached max_val_batches=%d, stop validation early.",
                    args.max_val_batches,
                )
                break
            batch = [tensor.to(args.device) for tensor in batch]
            (
                obs_traj,
                pred_traj_gt,
                obs_traj_rel,
                pred_traj_gt_rel,
                obs_state,
                pred_state,
                non_linear_ped,
                loss_mask,
                seq_start_end,
            ) = batch

            pred_traj_fake_rel= model(obs_traj_rel, obs_traj,obs_state,pred_state, seq_start_end)

            pred_traj_fake_rel_predpart = pred_traj_fake_rel[-args.pred_len :]
            obs_traj = obs_traj[:, :, 2:4]
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel_predpart, obs_traj[-1])
            pred_traj_gt = pred_traj_gt[:, :, 2:4]
            ade_, fde_ = cal_ade_fde(pred_traj_gt, pred_traj_fake)
            ade_ = ade_ / (obs_traj.shape[1] * args.pred_len)

            fde_ = fde_ / (obs_traj.shape[1])
            ade.update(ade_, obs_traj.shape[1])
            fde.update(fde_, obs_traj.shape[1])


        logging.info(
            " * ADE  {ade.avg:.3f} FDE  {fde.avg:.3f}".format(ade=ade, fde=fde)
        )
        writer.add_scalar("val_ade", ade.avg, epoch)
    return ade.avg


def cal_ade_fde(pred_traj_gt, pred_traj_fake):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    return ade, fde


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    if is_best:
        torch.save(state, filename)
        logging.info("-------------- lower ade ----------------")
        shutil.copyfile(filename, "model_best.pth.tar")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if args.device == "cpu":
        args.pin_memory = False
    utils.set_logger(os.path.join(args.log_dir, "train.log"))
    checkpoint_dir = "./checkpoint"
    if os.path.exists(checkpoint_dir) is False:
        os.mkdir(checkpoint_dir)
    main(args)
