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
import torch.nn.functional as F
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
    "--train_stage",
    default="warmup",
    choices=["warmup", "refine", "adversarial"],
    help="CycleState 的阶段化训练协议。warmup/refine 默认只训生成器，adversarial 再引入 GAN。",
)
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
parser.add_argument(
    "--num_val_samples",
    default=20,
    type=int,
    help="训练内验证时的采样次数，用于尽量对齐离线评估的 best-of-K 口径。",
)
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
    "--val_dset_type",
    default="val",
    choices=["val", "test"],
    help="训练内验证使用的数据 split。默认 val，test 只用于最终复核或兼容旧协议。",
)
parser.add_argument(
    "--val_every",
    default=1,
    type=int,
    help="正式训练时按 epoch 间隔做验证；smoke run 可继续配合 max_*_batches 使用。",
)
parser.add_argument(
    "--generator_only",
    action="store_true",
    default=None,
    help="只训练生成器，不更新判别器，用于新模型早期稳定训练。",
)
parser.add_argument(
    "--gan_weight",
    default=None,
    type=float,
    help="生成器对抗损失的缩放系数。",
)
parser.add_argument(
    "--aux_queue_weight",
    default=None,
    type=float,
    help="CycleState 的 queue-state 辅助损失权重。",
)
parser.add_argument(
    "--aux_rollout_weight",
    default=None,
    type=float,
    help="CycleState 的 rollout queue-state 辅助损失权重。默认跟随 aux_queue_weight。",
)
parser.add_argument(
    "--aux_cycle_weight",
    default=None,
    type=float,
    help="CycleState 的 cycle-state 辅助损失权重。",
)
parser.add_argument(
    "--disable_state_gating",
    action="store_true",
    help="关闭 phase-conditioned state modulation，用于 CycleState 的 gating 消融。",
)
parser.add_argument(
    "--disable_queue_rollout",
    action="store_true",
    help="关闭预测阶段的 phase-rolling queue memory，回退到静态 queue-state 注入。",
)
parser.add_argument(
    "--disable_lane_queue_anchor",
    action="store_true",
    help="关闭 lane-level queue consensus anchor，使 queue rollout 只依赖个体局部中观状态。",
)
parser.add_argument(
    "--disable_decoder_state_residual",
    action="store_true",
    help="关闭 baseline-compatible decoder state residual，使状态分支不再残差调制原始解码器。",
)
parser.add_argument(
    "--grad_clip",
    default=None,
    type=float,
    help="生成器/判别器梯度裁剪阈值。CycleState warmup/refine 默认启用。",
)
parser.add_argument(
    "--rollout_residual_scale",
    default=None,
    type=float,
    help="CycleState rollout queue delta 注入 decoder 的缩放系数。",
)
detach_rollout_group = parser.add_mutually_exclusive_group()
detach_rollout_group.add_argument(
    "--detach_rollout_state",
    action="store_true",
    dest="detach_rollout_state",
    default=None,
    help="截断预测期 queue rollout 跨步反传，用于 warmup 稳定化。",
)
detach_rollout_group.add_argument(
    "--no_detach_rollout_state",
    action="store_false",
    dest="detach_rollout_state",
    help="显式关闭预测期 queue rollout 跨步截断，用于消融或 refine 复核。",
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

TRAIN_STAGE_DEFAULTS = {
    "warmup": {
        "generator_only": True,
        "gan_weight": 0.0,
        "aux_queue_weight": 10.0,
        "aux_cycle_weight": 5.0,
        "teacher_forcing_ratio": 0.8,
        "grad_clip": 1.0,
        "rollout_residual_scale": 0.35,
        "detach_rollout_state": True,
    },
    "refine": {
        "generator_only": True,
        "gan_weight": 0.0,
        "aux_queue_weight": 3.0,
        "aux_cycle_weight": 1.5,
        "teacher_forcing_ratio": 0.6,
        "grad_clip": 1.0,
        "rollout_residual_scale": 0.7,
        "detach_rollout_state": False,
    },
    "adversarial": {
        "generator_only": False,
        "gan_weight": 50.0,
        "aux_queue_weight": 3.0,
        "aux_cycle_weight": 1.5,
        "teacher_forcing_ratio": 0.4,
        "grad_clip": 1.0,
        "rollout_residual_scale": 0.7,
        "detach_rollout_state": False,
    },
}

BASELINE_DEFAULTS = {
    "generator_only": False,
    "gan_weight": 1000.0,
    "aux_queue_weight": 0.0,
    "aux_cycle_weight": 0.0,
    "teacher_forcing_ratio": 0.5,
    "grad_clip": 0.0,
    "rollout_residual_scale": 1.0,
    "detach_rollout_state": False,
}


def apply_stage_defaults(args):
    """根据训练阶段补齐默认超参。

    这里采用“显式阶段协议覆盖默认值、用户传参再优先覆盖”的策略：
    - parser 默认值设为 `None` 的参数，按阶段自动填充；
    - 用户如果显式传了值，就保留用户选择。
    """
    model_type = getattr(args, "model_type", "cyclestate")
    stage_defaults = (
        TRAIN_STAGE_DEFAULTS[args.train_stage]
        if model_type == "cyclestate"
        else BASELINE_DEFAULTS
    )
    for key, value in stage_defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    if getattr(args, "aux_rollout_weight", None) is None:
        args.aux_rollout_weight = getattr(args, "aux_queue_weight", 0.0)
    return args


def build_traffic_context_from_batch(batch):
    """把 dataloader 返回的 tuple batch 适配成结构化 traffic context。

    目前仍然保留原始 tuple 训练入口，避免一次性改乱仓库。
    这个 adapter 的作用是先在训练/评估层定义统一语义接口，为后续迁移
    到 INT2 或其它数据源预留“只换适配层、不重写模型主体”的空间。
    """
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
    stopline_distance = torch.sqrt(
        (obs_traj[:, :, 2] - obs_state[:, :, 0]) ** 2
        + (obs_traj[:, :, 3] - obs_state[:, :, 1]) ** 2
    )
    lane_ids = obs_traj[:, :, 4].long()
    phase_ids = obs_state[:, :, 2].long()
    phase_elapsed = obs_state[:, :, 3]
    pred_phase_ids = pred_state[:, :, 2].long()
    pred_phase_elapsed = pred_state[:, :, 3]

    scene_groups = []
    lane_grouping = []
    for start, end in seq_start_end.tolist():
        scene_groups.append((start, end))
        lane_grouping.append(lane_ids[-1, start:end].detach().cpu().tolist())

    queue_features = None
    queue_targets = None
    cycle_feature_seq = None

    traffic_context = {
        "agent": {
            "obs_traj": obs_traj,
            "pred_traj_gt": pred_traj_gt,
            "obs_traj_rel": obs_traj_rel,
            "pred_traj_gt_rel": pred_traj_gt_rel,
            "direction": obs_traj[:, :, 9],
            "lane_ids": lane_ids,
            "stopline_distance": stopline_distance,
        },
        "signal": {
            "obs_state": obs_state,
            "pred_state": pred_state,
            "phase_ids": phase_ids,
            "phase_elapsed": phase_elapsed,
            "pred_phase_ids": pred_phase_ids,
            "pred_phase_elapsed": pred_phase_elapsed,
            "observed_phase": phase_ids,
            "observed_elapsed": phase_elapsed,
            "predicted_phase": pred_phase_ids,
            "predicted_elapsed": pred_phase_elapsed,
            "cycle_feature_seq": cycle_feature_seq,
        },
        "scene": {
            "seq_start_end": seq_start_end,
            "scene_groups": scene_groups,
            "lane_grouping": lane_grouping,
        },
        "meso": {
            "queue_feature_seq": queue_features,
            "queue_targets": queue_targets,
        },
        "meta": {
            "non_linear_ped": non_linear_ped,
            "loss_mask": loss_mask,
        },
    }
    return traffic_context


def compute_structured_aux_losses(
    queue_pred_last,
    queue_target_last,
    cycle_pred_last,
    cycle_target_last,
    queue_rollout_pred_seq=None,
    queue_rollout_target_seq=None,
    device=None,
):
    """把 queue/cycle 辅助监督拆成更符合语义的分项损失。

    Queue targets:
    - regression: queue count, waiting ratio, release ratio, lane queue length
    - binary: stop-line occupancy, front-of-queue flag

    Cycle targets:
    - phase one-hot(3): classification
    - elapsed / remaining: regression
    - phase change: binary classification
    """
    if device is None:
        for tensor in (
            queue_pred_last,
            queue_target_last,
            cycle_pred_last,
            cycle_target_last,
        ):
            if tensor is not None:
                device = tensor.device
                break
    if device is None:
        device = torch.device("cpu")

    zero = torch.zeros(1, device=device)
    losses = {
        "queue_reg_loss": zero.clone(),
        "queue_cls_loss": zero.clone(),
        "queue_rollout_reg_loss": zero.clone(),
        "queue_rollout_cls_loss": zero.clone(),
        "cycle_phase_loss": zero.clone(),
        "cycle_time_loss": zero.clone(),
        "cycle_change_loss": zero.clone(),
    }

    if queue_pred_last is not None and queue_target_last is not None:
        queue_reg_idx = [0, 1, 2, 3]
        queue_cls_idx = [4, 5]
        losses["queue_reg_loss"] = F.mse_loss(
            queue_pred_last[:, queue_reg_idx], queue_target_last[:, queue_reg_idx]
        )
        losses["queue_cls_loss"] = F.binary_cross_entropy_with_logits(
            queue_pred_last[:, queue_cls_idx], queue_target_last[:, queue_cls_idx]
        )
    if (
        queue_rollout_pred_seq is not None
        and queue_rollout_target_seq is not None
        and queue_rollout_pred_seq.numel() > 0
    ):
        queue_reg_idx = [0, 1, 2, 3]
        queue_cls_idx = [4, 5]
        rollout_pred_flat = queue_rollout_pred_seq.reshape(-1, queue_rollout_pred_seq.size(-1))
        rollout_target_flat = queue_rollout_target_seq.reshape(-1, queue_rollout_target_seq.size(-1))
        losses["queue_rollout_reg_loss"] = F.mse_loss(
            rollout_pred_flat[:, queue_reg_idx], rollout_target_flat[:, queue_reg_idx]
        )
        losses["queue_rollout_cls_loss"] = F.binary_cross_entropy_with_logits(
            rollout_pred_flat[:, queue_cls_idx], rollout_target_flat[:, queue_cls_idx]
        )

    if cycle_pred_last is not None and cycle_target_last is not None:
        phase_target = cycle_target_last[:, :3].argmax(dim=1)
        losses["cycle_phase_loss"] = F.cross_entropy(
            cycle_pred_last[:, :3], phase_target
        )
        losses["cycle_time_loss"] = F.mse_loss(
            cycle_pred_last[:, 3:5], cycle_target_last[:, 3:5]
        )
        losses["cycle_change_loss"] = F.binary_cross_entropy_with_logits(
            cycle_pred_last[:, 5:6], cycle_target_last[:, 5:6]
        )

    losses["queue_main_loss"] = (
        losses["queue_reg_loss"] + losses["queue_cls_loss"]
    )
    losses["queue_rollout_loss"] = (
        losses["queue_rollout_reg_loss"] + losses["queue_rollout_cls_loss"]
    )
    losses["queue_total_loss"] = (
        losses["queue_main_loss"] + losses["queue_rollout_loss"]
    )
    losses["cycle_total_loss"] = (
        losses["cycle_phase_loss"]
        + losses["cycle_time_loss"]
        + losses["cycle_change_loss"]
    )
    return losses


def get_teacher_forcing_ratio(args, epoch):
    """按训练阶段提供更稳定的 teacher forcing 调度。"""
    base_ratio = getattr(args, "teacher_forcing_ratio", 0.5)
    if args.model_type != "cyclestate":
        return base_ratio
    if args.train_stage == "warmup":
        return base_ratio

    epoch_offset = max(epoch - args.start_epoch, 0)
    decay = 0.02 * epoch_offset
    if args.train_stage == "refine":
        return max(0.35, base_ratio - decay)
    return max(0.2, base_ratio - decay)


def get_validation_dset_path(args):
    """返回训练内验证使用的 split 路径。"""
    return get_dset_path(args.dataset_name, getattr(args, "val_dset_type", "val"))


def build_optimizers(args, model, discriminator):
    """构造优化器，确保命令行 lr 真正生效。"""
    return (
        optim.RMSprop(model.parameters(), lr=args.lr),
        optim.RMSprop(discriminator.parameters(), lr=args.lr),
    )


def maybe_clip_gradients(parameters, grad_clip):
    """按需裁剪梯度；返回裁剪前范数，便于日志分析。"""
    if grad_clip is None or grad_clip <= 0:
        return None
    return torch.nn.utils.clip_grad_norm_(parameters, grad_clip)


def _mean_norm_from_tensor(tensor):
    if tensor is None:
        return 0.0
    if tensor.numel() == 0:
        return 0.0
    if tensor.dim() == 0:
        return float(tensor.detach().item())
    if tensor.dim() == 1:
        return float(tensor.detach().float().mean().item())
    return float(tensor.detach().float().norm(dim=-1).mean().item())


def _mean_value_from_tensor(tensor):
    if tensor is None:
        return 0.0
    if tensor.numel() == 0:
        return 0.0
    return float(tensor.detach().float().mean().item())


def extract_state_stability_metrics(debug_info, pred_offsets):
    """提取状态注入稳定性指标，用于定位 warmup 后半程崩坏。"""
    if debug_info is None:
        debug_info = {}
    return {
        "decoder_state_init_residual_norm": _mean_value_from_tensor(
            debug_info.get("decoder_state_init_residual_norm")
        ),
        "decoder_state_step_residual_norm": _mean_value_from_tensor(
            debug_info.get("decoder_state_step_residual_norm_seq")
        ),
        "queue_rollout_hidden_norm": _mean_norm_from_tensor(
            debug_info.get("queue_rollout_hidden_seq")
        ),
        "pred_offset_norm": _mean_norm_from_tensor(pred_offsets),
    }


def should_run_validation(args, epoch, batch_idx, num_batches):
    """统一决定当前批次是否触发验证。

    - smoke / quick run：保留 batch 级快速反馈；
    - 正式训练：只在满足 `val_every` 的 epoch 末验证一次。
    """
    is_smoke_run = (
        getattr(args, "max_train_batches", 0) > 0
        or getattr(args, "num_epochs", 0) == 0
    )
    if is_smoke_run:
        batch_interval = max(getattr(args, "print_every", 1), 1)
        is_interval_boundary = ((batch_idx + 1) % batch_interval) == 0
        is_last_batch = batch_idx == (num_batches - 1)
        return is_interval_boundary or is_last_batch

    epoch_offset = max(epoch - getattr(args, "start_epoch", 0), 0)
    if epoch_offset % max(getattr(args, "val_every", 1), 1) != 0:
        return False
    return batch_idx == (num_batches - 1)


def prepare_traffic_context(args, model, batch, generator_input):
    """统一构造 CycleState 使用的 traffic context。"""
    base_context = build_traffic_context_from_batch(batch)
    if args.model_type != "cyclestate" or not hasattr(model, "build_traffic_context"):
        return None

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
    traffic_context = model.build_traffic_context(
        generator_input,
        obs_traj,
        obs_state,
        pred_state,
        seq_start_end,
    )
    traffic_context["scene"].update(base_context["scene"])
    traffic_context["signal"].update(
        {
            "observed_phase": base_context["signal"]["observed_phase"],
            "observed_elapsed": base_context["signal"]["observed_elapsed"],
            "predicted_phase": base_context["signal"]["predicted_phase"],
            "predicted_elapsed": base_context["signal"]["predicted_elapsed"],
        }
    )
    traffic_context["agent"].update(
        {
            "pred_traj_gt": pred_traj_gt,
            "pred_traj_gt_rel": pred_traj_gt_rel,
        }
    )
    traffic_context["meta"] = base_context["meta"]
    return traffic_context


def forward_generator(args, model, batch, teacher_forcing_ratio, training_step=3):
    """统一封装 d2tpred / cyclestate 的生成器前向调用。"""
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
    if model.training:
        generator_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)
    else:
        generator_input = obs_traj_rel

    traffic_context = prepare_traffic_context(args, model, batch, generator_input)
    if args.model_type == "cyclestate":
        pred_traj_fake_rel = model(
            generator_input,
            obs_traj,
            obs_state,
            pred_state,
            seq_start_end,
            teacher_forcing_ratio=teacher_forcing_ratio,
            training_step=training_step,
            traffic_context=traffic_context,
        )
    else:
        pred_traj_fake_rel = model(
            generator_input,
            obs_traj,
            obs_state,
            pred_state,
            seq_start_end,
            teacher_forcing_ratio=teacher_forcing_ratio,
            training_step=training_step,
        )
    return pred_traj_fake_rel, traffic_context


def evaluate_helper(error, seq_start_end):
    """把多次采样误差按场景聚合，与独立评估脚本保持一致。"""
    sum_ = 0
    error = torch.stack(error, dim=1)
    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def compute_raw_displacement_metrics(pred_traj_gt, pred_traj_fake):
    """返回逐 agent 的原始 ADE/FDE 误差，供训练验证与独立评估共用。"""
    ade_raw = displacement_error(pred_traj_fake, pred_traj_gt, mode="raw")
    fde_raw = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], mode="raw"
    )
    return ade_raw, fde_raw


def compute_average_displacement_metrics(
    pred_traj_gt,
    pred_traj_fake,
    seq_start_end=None,
):
    """统一计算平均 ADE/FDE。

    当提供 `seq_start_end` 时，按场景做 best-of-K 风格聚合；
    否则保持训练内单次采样的直接平均逻辑。
    """
    ade_raw, fde_raw = compute_raw_displacement_metrics(pred_traj_gt, pred_traj_fake)
    if seq_start_end is not None:
        ade_sum = evaluate_helper([ade_raw], seq_start_end)
        fde_sum = evaluate_helper([fde_raw], seq_start_end)
    else:
        ade_sum = ade_raw.sum()
        fde_sum = fde_raw.sum()
    batch = pred_traj_gt.size(1)
    pred_len = pred_traj_gt.size(0)
    ade = ade_sum / (batch * pred_len)
    fde = fde_sum / batch
    return ade, fde


def compute_best_of_k_metrics(
    ade_candidates,
    fde_candidates,
    seq_start_end,
    pred_len,
    total_traj,
):
    """按场景做多采样 best-of-K 聚合，和离线评估保持一致。"""
    ade_sum, fde_sum = compute_best_of_k_metric_sums(
        ade_candidates, fde_candidates, seq_start_end
    )
    ade = ade_sum / (total_traj * pred_len)
    fde = fde_sum / total_traj
    return ade, fde


def compute_best_of_k_metric_sums(
    ade_candidates,
    fde_candidates,
    seq_start_end,
):
    """返回 scene-level best-of-K 选择后的误差和。"""
    ade_sum = evaluate_helper(ade_candidates, seq_start_end)
    fde_sum = evaluate_helper(fde_candidates, seq_start_end)
    return ade_sum, fde_sum


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
    apply_stage_defaults(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_path = get_dset_path(args.dataset_name, "train")
    val_path = get_validation_dset_path(args)

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
    model_kwargs = dict(
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
    if args.model_type == "cyclestate":
        model_kwargs["disable_state_gating"] = args.disable_state_gating
        model_kwargs["disable_queue_rollout"] = args.disable_queue_rollout
        model_kwargs["disable_lane_queue_anchor"] = args.disable_lane_queue_anchor
        model_kwargs["disable_decoder_state_residual"] = (
            args.disable_decoder_state_residual
        )
        model_kwargs["rollout_residual_scale"] = args.rollout_residual_scale
        model_kwargs["detach_rollout_state"] = args.detach_rollout_state
    model = model_cls(**model_kwargs)
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

    optimizer, optimizer_d = build_optimizers(args, model, Discriminator)
    logging.info(
        "Training protocol | model_type=%s stage=%s val_split=%s lr=%.6f grad_clip=%.3f generator_only=%s gan_weight=%.3f aux_queue=%.3f aux_rollout=%.3f aux_cycle=%.3f rollout_residual_scale=%.3f detach_rollout_state=%s disable_state_gating=%s disable_queue_rollout=%s disable_lane_queue_anchor=%s disable_decoder_state_residual=%s teacher_forcing=%.3f",
        args.model_type,
        args.train_stage,
        args.val_dset_type,
        args.lr,
        args.grad_clip,
        args.generator_only,
        args.gan_weight,
        args.aux_queue_weight,
        args.aux_rollout_weight,
        args.aux_cycle_weight,
        args.rollout_residual_scale,
        args.detach_rollout_state,
        args.disable_state_gating,
        args.disable_queue_rollout,
        args.disable_lane_queue_anchor,
        args.disable_decoder_state_residual,
        args.teacher_forcing_ratio,
    )
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
                if should_run_validation(
                    args, epoch, batch_idx, len(train_loader)
                ):
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
                        os.path.join(args.checkpoint_dir, f"checkpoint{epoch}.pth.tar"),
                        best_filename=os.path.join(args.checkpoint_dir, "model_best.pth.tar"),
                    )
                continue
            if D_step>0:
                D_train(args,len(train_loader), model,batch_idx,batch,Discriminator, optimizer_d, epoch, training_step, writer)
                D_step=D_step-1
            else:
                train(args,len(train_loader), model, batch_idx,batch,Discriminator, optimizer, epoch, training_step, writer)
                D_step=2
                if should_run_validation(
                    args, epoch, batch_idx, len(train_loader)
                ):
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
                        os.path.join(args.checkpoint_dir, f"checkpoint{epoch}.pth.tar"),
                        best_filename=os.path.join(args.checkpoint_dir, "model_best.pth.tar"),
                    )
    writer.close()


def train(args,lens, model,batch_idx,batch,Discriminator, optimizer, epoch, training_step, writer):
    """更新生成器。"""
    losses = utils.AverageMeter("L2_Loss", ":.6f")
    g_losses = utils.AverageMeter("G_Loss", ":.6f")
    aux_queue_reg_losses = utils.AverageMeter("QReg", ":.6f")
    aux_queue_cls_losses = utils.AverageMeter("QCls", ":.6f")
    aux_queue_rollout_reg_losses = utils.AverageMeter("QRollReg", ":.6f")
    aux_queue_rollout_cls_losses = utils.AverageMeter("QRollCls", ":.6f")
    aux_cycle_phase_losses = utils.AverageMeter("CPhase", ":.6f")
    aux_cycle_time_losses = utils.AverageMeter("CTime", ":.6f")
    aux_cycle_change_losses = utils.AverageMeter("CChange", ":.6f")
    progress = utils.ProgressMeter(
        lens,
        [
            losses,
            g_losses,
            aux_queue_reg_losses,
            aux_queue_cls_losses,
            aux_queue_rollout_reg_losses,
            aux_queue_rollout_cls_losses,
            aux_cycle_phase_losses,
            aux_cycle_time_losses,
            aux_cycle_change_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
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
    teacher_forcing_ratio = get_teacher_forcing_ratio(args, epoch)
    pred_traj_fake_rel = None
    for _ in range(args.best_k):
        # best-of-K：多次采样，保留误差最小的那次。
        pred_traj_fake_rel, _ = forward_generator(
            args,
            model,
            batch,
            teacher_forcing_ratio=teacher_forcing_ratio,
            training_step=training_step,
        )
        modinput = pred_traj_gt_rel[:, :, 2:4]
        l2_loss_rel.append(
            l2_loss(
                pred_traj_fake_rel,
                modinput,
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
    aux_losses = compute_structured_aux_losses(
        None, None, None, None, device=predtrajgt.device
    )
    if args.model_type == "cyclestate" and hasattr(model, "debug_last_aux"):
        aux_info = model.debug_last_aux
        queue_pred_last = None
        queue_target_last = None
        queue_rollout_pred_seq = None
        queue_rollout_target_seq = None
        cycle_pred_last = None
        cycle_target_last = None
        if (
            (args.aux_queue_weight > 0 or args.aux_rollout_weight > 0)
            and aux_info["queue_pred_last"] is not None
        ):
            queue_pred_last = aux_info["queue_pred_last"]
            queue_target_last = aux_info["queue_targets"][-1]
            queue_rollout_pred_seq = aux_info.get("queue_rollout_pred_seq")
            queue_rollout_target_seq = aux_info.get("queue_rollout_target_seq")
        if args.aux_cycle_weight > 0 and aux_info["cycle_pred_last"] is not None:
            cycle_pred_last = aux_info["cycle_pred_last"]
            cycle_target_last = aux_info["cycle_feature_seq"][-1]
        aux_losses = compute_structured_aux_losses(
            queue_pred_last,
            queue_target_last,
            cycle_pred_last,
            cycle_target_last,
            queue_rollout_pred_seq=queue_rollout_pred_seq,
            queue_rollout_target_seq=queue_rollout_target_seq,
            device=predtrajgt.device,
        )
    aux_queue_main_loss = aux_losses["queue_main_loss"]
    aux_queue_rollout_loss = aux_losses["queue_rollout_loss"]
    aux_cycle_loss = aux_losses["cycle_total_loss"]
    losses.update(L2_loss.item(), obs_traj.shape[1])
    g_losses.update(g_loss.item(),obs_traj.shape[1])
    aux_queue_reg_losses.update(aux_losses["queue_reg_loss"].item(), obs_traj.shape[1])
    aux_queue_cls_losses.update(aux_losses["queue_cls_loss"].item(), obs_traj.shape[1])
    aux_queue_rollout_reg_losses.update(aux_losses["queue_rollout_reg_loss"].item(), obs_traj.shape[1])
    aux_queue_rollout_cls_losses.update(aux_losses["queue_rollout_cls_loss"].item(), obs_traj.shape[1])
    aux_cycle_phase_losses.update(aux_losses["cycle_phase_loss"].item(), obs_traj.shape[1])
    aux_cycle_time_losses.update(aux_losses["cycle_time_loss"].item(), obs_traj.shape[1])
    aux_cycle_change_losses.update(aux_losses["cycle_change_loss"].item(), obs_traj.shape[1])
    total_loss = (
        L2_loss
        + g_loss * args.gan_weight
        + aux_queue_main_loss * args.aux_queue_weight
        + aux_queue_rollout_loss * args.aux_rollout_weight
        + aux_cycle_loss * args.aux_cycle_weight
    )
    stability_metrics = extract_state_stability_metrics(
        getattr(model, "debug_last_aux", None), pred_traj_fake_rel
    )
    total_loss.backward()
    grad_norm = maybe_clip_gradients(model.parameters(), args.grad_clip)
    optimizer.step()
    if batch_idx % args.print_every == 0:
        progress.display(batch_idx)
        logging.info(
            "StateStability | DInitNorm %.6f DStepNorm %.6f QRollHNorm %.6f PredOffsetNorm %.6f GradNorm %.6f",
            stability_metrics["decoder_state_init_residual_norm"],
            stability_metrics["decoder_state_step_residual_norm"],
            stability_metrics["queue_rollout_hidden_norm"],
            stability_metrics["pred_offset_norm"],
            float(grad_norm.item() if grad_norm is not None else 0.0),
        )
    writer.add_scalar("g_l2_loss", losses.avg, batch_idx)
    writer.add_scalar("g_ad_loss", g_losses.avg * args.gan_weight, batch_idx)
    writer.add_scalar("g_queue_reg_loss", aux_queue_reg_losses.avg, batch_idx)
    writer.add_scalar("g_queue_cls_loss", aux_queue_cls_losses.avg, batch_idx)
    writer.add_scalar("g_queue_rollout_reg_loss", aux_queue_rollout_reg_losses.avg, batch_idx)
    writer.add_scalar("g_queue_rollout_cls_loss", aux_queue_rollout_cls_losses.avg, batch_idx)
    writer.add_scalar("g_cycle_phase_loss", aux_cycle_phase_losses.avg, batch_idx)
    writer.add_scalar("g_cycle_time_loss", aux_cycle_time_losses.avg, batch_idx)
    writer.add_scalar("g_cycle_change_loss", aux_cycle_change_losses.avg, batch_idx)
    writer.add_scalar(
        "state_decoder_init_residual_norm",
        stability_metrics["decoder_state_init_residual_norm"],
        batch_idx,
    )
    writer.add_scalar(
        "state_decoder_step_residual_norm",
        stability_metrics["decoder_state_step_residual_norm"],
        batch_idx,
    )
    writer.add_scalar(
        "state_queue_rollout_hidden_norm",
        stability_metrics["queue_rollout_hidden_norm"],
        batch_idx,
    )
    writer.add_scalar(
        "state_pred_offset_norm",
        stability_metrics["pred_offset_norm"],
        batch_idx,
    )
    writer.add_scalar(
        "g_grad_norm",
        float(grad_norm.item() if grad_norm is not None else 0.0),
        batch_idx,
    )

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

    pred_traj_fake_rel, _ = forward_generator(
        args,
        model,
        batch,
        teacher_forcing_ratio=0.0,
        training_step=training_step,
    )

    pred_traj_fake = torch.cat((obs_traj[:,:,2:4],relative_to_abs(pred_traj_fake_rel, obs_traj[-1,:,2:4])),dim=0)
    traj_state=torch.cat((obs_state,pred_state),dim=0)
    traj_real=torch.cat((obs_traj[:,:,2:4],pred_traj_gt[:,:,2:4]),dim=0)
    fakesocre=Discriminator(pred_traj_fake,traj_state,seq_start_end)
    realsocre=Discriminator(traj_real,traj_state,seq_start_end)
    D_loss=gan_d_loss(realsocre,fakesocre)

    D_losses.update(D_loss.item(), obs_traj.shape[1])
    D_loss.backward()
    maybe_clip_gradients(Discriminator.parameters(), args.grad_clip)
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

            obs_traj = obs_traj[:, :, 2:4]
            pred_traj_gt = pred_traj_gt[:, :, 2:4]
            ade_candidates = []
            fde_candidates = []
            for _ in range(args.num_val_samples):
                pred_traj_fake_rel, _ = forward_generator(
                    args,
                    model,
                    batch,
                    teacher_forcing_ratio=0.0,
                    training_step=3,
                )
                pred_traj_fake_rel_predpart = pred_traj_fake_rel[-args.pred_len :]
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel_predpart, obs_traj[-1]
                )
                ade_raw, fde_raw = compute_raw_displacement_metrics(
                    pred_traj_gt, pred_traj_fake
                )
                ade_candidates.append(ade_raw)
                fde_candidates.append(fde_raw)
            ade_, fde_ = compute_best_of_k_metrics(
                ade_candidates,
                fde_candidates,
                seq_start_end,
                args.pred_len,
                obs_traj.shape[1],
            )
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


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar", best_filename=None):
    if is_best:
        torch.save(state, filename)
        logging.info("-------------- lower ade ----------------")
        if best_filename is None:
            best_filename = os.path.join(os.path.dirname(filename), "model_best.pth.tar")
        shutil.copyfile(filename, best_filename)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if args.device == "cpu":
        args.pin_memory = False
    os.makedirs(args.log_dir, exist_ok=True)
    args.checkpoint_dir = os.path.join(args.log_dir, "checkpoint")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    utils.set_logger(os.path.join(args.log_dir, "train.log"))
    main(args)
