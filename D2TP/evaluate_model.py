"""评估脚本。"""

import argparse
import json
import logging
import torch

from data.loader import data_loader
from models import (
    TrajectoryGenerator,
    CycleStateTrajectoryGenerator,
    AblationConfig,
    RolloutQueueCoefs,
    apply_rollout_coefs_override,
)
from train import (
    build_traffic_context_from_batch,
    compute_best_of_k_metric_sums,
    compute_raw_displacement_metrics,
    maybe_load_compatible_weights,
    parse_rollout_queue_coefs,
    NumValSamplesTracker,
    reapply_phase_duration_limits_if_overridden,
)
from utils import (
    int_tuple,
    relative_to_abs,
    get_dset_path,
)

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", default="./", help="Directory containing logging file")

parser.add_argument("--dataset_name", default="VTP_C", type=str)
parser.add_argument("--delim", default="\t")
parser.add_argument("--loader_num_workers", default=0, type=int)
parser.add_argument("--obs_len", default=8, type=int)
parser.add_argument("--pred_len", default=12, type=int)
parser.add_argument("--skip", default=1, type=int)

parser.add_argument("--seed", type=int, default=72, help="Random seed.")
parser.add_argument("--batch_size", default=64, type=int)

parser.add_argument("--noise_dim", default=(16,), type=int_tuple)
parser.add_argument("--noise_type", default="gaussian")
parser.add_argument("--noise_mix_type", default="global")

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
    help="选择评估的生成器类型。",
)
parser.add_argument(
    "--disable_state_gating",
    action="store_true",
    help="关闭 CycleState 的 phase-conditioned state modulation，用于评估消融模型。",
)
parser.add_argument(
    "--disable_queue_rollout",
    action="store_true",
    help="关闭 CycleState 的预测期 queue rollout，用于评估静态 queue-state 版本。",
)
parser.add_argument(
    "--disable_lane_queue_anchor",
    action="store_true",
    help="关闭 CycleState 的 lane-level queue consensus anchor，用于评估局部 queue rollout 版本。",
)
parser.add_argument(
    "--disable_decoder_state_residual",
    action="store_true",
    help="关闭 CycleState 的 baseline-compatible decoder state residual，用于评估无残差状态注入版本。",
)
parser.add_argument(
    "--disable_aux_losses",
    action="store_true",
    help="Phase 4 #22 消融实验统一主开关：一次关闭所有 CycleState 特有功能"
    "（state gating、queue rollout、lane queue anchor、decoder state residual），"
    "使模型行为等价于全消融模式，用于公平评估消融基线。",
)
parser.add_argument(
    "--rollout_residual_scale",
    default=1.0,
    type=float,
    help="CycleState rollout queue delta 注入 decoder 的缩放系数，需与训练 checkpoint 协议一致。",
)
parser.add_argument(
    "--decoder_state_residual_scale",
    default=1.0,
    type=float,
    help="CycleState decoder state residual 注入 decoder hidden 的缩放系数。",
)
# Phase 3 #16: 评估侧同样支持 ``--rollout_queue_coefs_json`` JSON 字符串覆盖,
# 行为与 train.py 一致; 留空使用 ``RolloutQueueCoefs()`` 默认值, 与原硬编码兼容。
parser.add_argument(
    "--rollout_queue_coefs_json",
    default="",
    type=str,
    help="(Phase 3 #16) JSON 字符串,用于对 ``RolloutQueueCoefs`` 字段做部分覆盖。"
    "需与训练时的配置一致,否则 rollout 行为会与 checkpoint 不匹配。",
)
parser.add_argument(
    "--detach_rollout_state",
    action="store_true",
    help="评估时保留参数兼容；eval 模式下不会截断额外训练梯度。",
)


def _parse_phase_duration_limits_eval(raw):
    """Phase 3 #23: 评估侧解析 ``--phase_duration_limits`` 字符串。

    与 ``train._parse_phase_duration_limits`` 行为对齐: 逗号分隔的 3 个
    非负浮点 (R, Y, G); 留空 / ``None`` / ``"None"`` 触发模型 ``__init__``
    默认值 ``(38.0, 47.0, 2.0)``, 与训练 checkpoint 协议保持一致。
    """
    if raw is None or raw == "" or raw == "None":
        return None
    parts = [p.strip() for p in str(raw).split(",") if p.strip() != ""]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--phase_duration_limits 必须是 3 个浮点数（用逗号分隔），got {!r}".format(raw)
        )
    try:
        values = tuple(float(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--phase_duration_limits 元素无法解析为浮点: {!r}".format(raw)
        ) from exc
    if any(v < 0 for v in values):
        raise argparse.ArgumentTypeError(
            "--phase_duration_limits 元素必须非负, got {}".format(values)
        )
    return values


# Phase 3 #23: 评估侧同样把 ``phase_duration_limits`` 暴露到 CLI, 与训练
# 协议保持一致; 留空 / None 触发 ``__init__`` 默认值 ``(38.0, 47.0, 2.0)``。
parser.add_argument(
    "--phase_duration_limits",
    default=None,
    type=_parse_phase_duration_limits_eval,
    help="Phase 3 #23: 逗号分隔的 3 个非负浮点 (R,Y,G),对应 ``phase_duration_limits`` "
    "buffer。需与训练时的配置一致,否则 cycle feature 行为会偏离 checkpoint。"
    "留空或 None 触发 ``__init__`` 默认值 (38.0, 47.0, 2.0)。",
)


parser.add_argument("--num_samples", default=20, type=int)
parser.add_argument(
    "--max_eval_batches",
    default=0,
    type=int,
    help="仅用于可比性审计或快速协议检查。大于 0 时，评估最多运行这么多 batch。",
)
parser.add_argument(
    "--eval_print_every",
    default=0,
    type=int,
    help="评估进度打印间隔。大于 0 时，每处理若干个 batch 打印一次进度。",
)
parser.add_argument(
    "--device",
    default="cuda",
    choices=["cuda", "cpu"],
    help="评估设备。选择 cuda 时会在可用 GPU 上运行。",
)
parser.add_argument(
    "--pin_memory",
    action="store_true",
    help="DataLoader 是否启用 pin_memory。GPU 评估时建议打开。",
)


parser.add_argument(
    "--dropout", type=float, default=0, help="Dropout rate (1 - keep probability)."
)
parser.add_argument(
    "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
)

parser.add_argument("--dset_type", default="test", type=str)


parser.add_argument(
    "--resume",
    default="./model_best.pth.tar",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)

def get_generator(checkpoint):
    """从 checkpoint 恢复生成器。"""
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
        # Phase 4 #22/#31: ``to_model_kwargs()`` 会把 "disable_aux_losses"
        # 及四个子开关一起从 ``args.disable_aux_losses`` / ``args.disable_*``
        # 集中映射到模型构造参数，避免 train/eval 两侧再手写一份散落透传逻辑。
        ablation_cfg = AblationConfig.from_args(args)
        model_kwargs.update(ablation_cfg.to_model_kwargs())
        model_kwargs["rollout_residual_scale"] = args.rollout_residual_scale
        model_kwargs["decoder_state_residual_scale"] = (
            getattr(args, "decoder_state_residual_scale", 1.0)
        )
        model_kwargs["detach_rollout_state"] = args.detach_rollout_state
        # Phase 3 #23: 把 ``--phase_duration_limits`` 透传到模型构造函数;
        # ``None`` 触发 ``__init__`` 默认值, 与训练时的协议保持一致。
        if getattr(args, "phase_duration_limits", None) is not None:
            model_kwargs["phase_duration_limits"] = tuple(
                args.phase_duration_limits
            )
        # Phase 3 #16: 评估侧同样把 ``--rollout_queue_coefs_json`` 解析后的
        # ``RolloutQueueCoefs`` 透传; 必须与训练时使用的系数保持一致, 否则
        # rollout 行为会偏离 checkpoint 学习到的分布。
        model_kwargs["rollout_queue_coefs"] = parse_rollout_queue_coefs(
            getattr(args, "rollout_queue_coefs_json", "")
        )
    model = model_cls(**model_kwargs)
    if args.model_type == "cyclestate":
        maybe_load_compatible_weights(model, checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint["state_dict"])
    reapply_phase_duration_limits_if_overridden(
        model,
        getattr(args, "phase_duration_limits", None),
    )
    # Phase 4 #29: 从 checkpoint 恢复归一化参数,
    # 确保评估使用与训练一致的归一化量纲。
    if hasattr(model, "load_norm_params"):
        model.load_norm_params(checkpoint.get("norm_params"))
    model.to(args.device)
    model.eval()
    # Phase 4 #21: 评估时显式把 checkpoint 中保存的 K 值与 args.num_samples
    # 做对齐校验,确保最终 test 评估与训练时 best-of-K 口径一致;若不一致
    # 给出可读的 warning 但不抛异常,避免破坏 evaluate 流程。
    k_tracker = NumValSamplesTracker(num_val_samples=getattr(args, "num_samples", None))
    k_tracker.restore_from_checkpoint(checkpoint.get("num_val_samples"))
    is_aligned, alignment_msg = k_tracker.check_alignment(
        getattr(args, "num_samples", None)
    )
    if is_aligned and "缺失" not in alignment_msg:
        logging.info("[Phase 4 #21] %s", alignment_msg)
    else:
        if "缺失" in alignment_msg:
            logging.info("[Phase 4 #21] %s", alignment_msg)
        else:
            logging.warning("[Phase 4 #21] %s", alignment_msg)
    return model
def evaluate(args, loader, generator):
    """在整个数据集上做 best-of-K 评估。"""
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_eval_batches > 0 and batch_idx >= args.max_eval_batches:
                print(
                    "Reached max_eval_batches={}, stop evaluation early.".format(
                        args.max_eval_batches
                    )
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

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(args.num_samples):
                # 同一输入采样多次，得到多模态结果。
                traffic_context = None
                if args.model_type == "cyclestate":
                    base_context = build_traffic_context_from_batch(batch)
                    traffic_context = generator.build_traffic_context(
                        obs_traj_rel,
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
                    traffic_context["meta"] = base_context["meta"]
                    pred_traj_fake_rel = generator(
                        obs_traj_rel,
                        obs_traj,
                        obs_state,
                        pred_state,
                        seq_start_end,
                        0,
                        3,
                        traffic_context=traffic_context,
                    )
                else:
                    pred_traj_fake_rel = generator(
                        obs_traj_rel, obs_traj,obs_state,pred_state,seq_start_end, 0, 3
                    )
                pred_traj_fake_rel = pred_traj_fake_rel[-args.pred_len :]

                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1,:,2:4])
                ade_, fde_ = compute_raw_displacement_metrics(
                    pred_traj_gt[:, :, 2:4],
                    pred_traj_fake,
                )
                ade.append(ade_)
                fde.append(fde_)
            ade_sum, fde_sum = compute_best_of_k_metric_sums(
                ade,
                fde,
                seq_start_end,
            )

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
            if args.eval_print_every > 0 and batch_idx % args.eval_print_every == 0:
                print(
                    "Eval batch {}/? | total_traj={} | current_ade_sum={:.6f} current_fde_sum={:.6f}".format(
                        batch_idx,
                        total_traj,
                        ade_sum.item(),
                        fde_sum.item(),
                    )
                )

        if total_traj == 0:
            raise RuntimeError("评估未处理任何轨迹，请检查数据集或 max_eval_batches 设置。")
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / total_traj
        return ade, fde


def main(args):
    """评估入口。"""
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前环境中没有可用 GPU。")

    checkpoint = torch.load(args.resume, map_location=args.device)
    generator = get_generator(checkpoint)
    path = get_dset_path(args.dataset_name, args.dset_type)

    _, loader = data_loader(args, path)
    ade, fde = evaluate(args, loader, generator)
    ckpt_best_ade = checkpoint.get("best_ade")
    if torch.is_tensor(ckpt_best_ade):
        ckpt_best_ade = ckpt_best_ade.item()
    print(
        "Dataset: {}, Split: {}, Pred Len: {}, Samples: {}, Resume: {}, Checkpoint Best ADE: {}, ADE: {:.12f}, FDE: {:.12f}".format(
            args.dataset_name,
            args.dset_type,
            args.pred_len,
            args.num_samples,
            args.resume,
            "None" if ckpt_best_ade is None else "{:.12f}".format(ckpt_best_ade),
            ade,
            fde,
        )
    )


if __name__ == "__main__":
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if args.device == "cpu":
        args.pin_memory = False
    torch.manual_seed(72)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main(args)
