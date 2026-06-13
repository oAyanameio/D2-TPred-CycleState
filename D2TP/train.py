"""训练脚本。

训练阶段采用生成器 + 判别器的对抗式学习，同时用 best-of-K 轨迹重建约束生成器。
"""

import argparse
import json
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
    AblationConfig,
    RolloutQueueCoefs,
    apply_rollout_coefs_override,
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
    "--teacher_forcing_ratio",
    default=None,
    type=float,
    help="显式覆盖当前训练阶段的 teacher forcing ratio，用于 protocol-check 或消融实验。",
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
    "--disable_aux_losses",
    action="store_true",
    help="Phase 4 #22 消融实验统一主开关：同时关闭 state gating、queue rollout、"
    "lane queue anchor、decoder state residual，并将 queue/cycle/rollout aux 权重置零，"
    "使 CycleState 行为等价于全消融模式（功能上对齐 baseline）。",
)
parser.add_argument(
    "--minimal_viable_mode",
    action="store_true",
    help="DE-3 决定性实验开关：把 CycleState 降到'最简可行'形态 — "
    "强制关闭 state_gating / queue_rollout / lane_queue_anchor / "
    "decoder_state_residual / aux_losses, 并把观测期最后时刻的 "
    "``[queue_last, cycle_last]`` 直接拼接到 decoder 初始化向量 "
    "``encoded_before_noise_hidden`` 后面, 验证'直接拼接'是否比'加性残差'"
    "更有效。该开关只能在 ``--model_type cyclestate`` 时生效。",
)
parser.add_argument(
    "--oracle_inject_mode",
    action="store_true",
    help="DE-1 决定性实验开关：把 CycleState 改为'oracle 交通状态直注'形态 — "
    "强制关闭 state_gating / queue_rollout / lane_queue_anchor / "
    "decoder_state_residual / aux_losses, 并把单步 oracle 特征 "
    "(phase one-hot / elapsed / remaining / distance / direction / speed / "
    "phase_change, 10 维) 直接拼接到 ``pred_lstm_model`` 的输入后面。oracle "
    "特征**不**经过任何学习,直接由 ``pred_state`` 与当前解码位置算出,等价于 "
    "bypass 整个 queue/cycle LSTM 分支。该开关只能在 ``--model_type cyclestate`` "
    "时生效。",
)
parser.add_argument(
    "--ar1_direct_inject_mode",
    action="store_true",
    help="AR-1 决定性实验开关：把 CycleState 改为'直接条件注入'形态 — "
    "在 DE-3 (init 拼接) 之上叠加两个新机制: 1. 把观测期最后时刻的 "
    "``[queue_last, cycle_last]`` (32+16=48 维, 不参与学习) 作为 "
    "``pred_lstm_model`` 的每步拼接输入; 2. 把同样的 state context 同时拼到 "
    "``pred_hidden2pos`` 输出投影的输入。隐含启用 ``--minimal_viable_mode``, "
    "即 AR-1 = DE-3 + per-step inject + output-projection inject。AR-1 与 "
    "``--oracle_inject_mode`` 互斥: AR-1 用 learned hidden, DE-1 用 oracle "
    "物理特征。该开关只能在 ``--model_type cyclestate`` 时生效。",
)
parser.add_argument(
    "--ar2_multiplicative_gating_mode",
    action="store_true",
    help="AR-2 决定性实验开关：把 CycleState 改为'乘法门控'形态 — "
    "在 DE-3 (init 拼接) 之上叠加一个新机制: 在 ``pred_lstm_model`` 每步 "
    "更新 ``pred_lstm_hidden`` 后, 用一个 2 层 MLP + sigmoid 计算逐元素 "
    "门控 (输入 = ``[pred_lstm_hidden, queue_last, cycle_last]``, 输出维度 "
    "= ``pred_lstm_hidden_size``), 然后 ``pred_lstm_hidden = pred_lstm_hidden * gate``。"
    "这是与 AR-1 (加性拼接) 不同的耦合方式: AR-2 用 state context 调制 "
    "隐状态维度, AR-1 用 state context 扩展输入/输出。隐含启用 "
    "``--minimal_viable_mode`` + 5 个 disable 开关, 即 AR-2 = DE-3 + "
    "per-step multiplicative gate。AR-2 与 ``--oracle_inject_mode`` / "
    "``--ar1_direct_inject_mode`` 互斥。该开关只能在 ``--model_type cyclestate`` "
    "时生效。",
)
parser.add_argument(
    "--c2_1_trajectory_level_mode",
    action="store_true",
    help="C2-1 第一变体 (``C2-1-MV1``) 开关: 把 trajectory encoder 由单层 "
    "``nn.LSTMCell`` 升级为 2 层 stacked LSTMCell,保持 hidden_size 不变以"
    "隔离'深度 vs 宽度'两个变量。这是 PLAN.md §6.3 分支 C2-1 的第一个决定"
    "性实验 — 'trajectory-level capacity' 是否是当前 1.6× 差距的瓶颈,"
    "**完全离开 state injection 路线**。该开关与 DE-3 / DE-1 / AR-1 / AR-2 "
    "等 state injection 模式正交,可单独启用,也可与 ``--minimal_viable_mode`` "
    "组合验证 'C2-1 + state hidden init 拼接' 联合效果。",
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
parser.add_argument(
    "--decoder_state_residual_scale",
    default=None,
    type=float,
    help="CycleState decoder state residual 注入 decoder hidden 的缩放系数。",
)
# Phase 3 #16: 把 ``rollout_queue_features`` 内 hardcoded 的物理系数集中到
# ``RolloutQueueCoefs`` dataclass 后,允许通过 JSON 字符串做部分字段覆盖。
# JSON 字段名与 ``models.RolloutQueueCoefs`` 字段名一一对应,例如
# ``--rollout_queue_coefs_json '{"waiting_ratio_red_inc": 0.04,
# "release_ratio_green_inc": 0.18}'``。任何未识别的 key 会被静默忽略,
# 不传则使用 dataclass 默认值,行为与原硬编码完全一致 (向后兼容)。
parser.add_argument(
    "--rollout_queue_coefs_json",
    default="",
    type=str,
    help="(Phase 3 #16) JSON 字符串,用于对 ``RolloutQueueCoefs`` 字段做部分覆盖。"
    "例如 '{\"waiting_ratio_red_inc\": 0.04, \"release_ratio_green_inc\": 0.18}'。"
    "留空表示使用 dataclass 默认值。",
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


def _parse_phase_duration_limits(raw):
    """Phase 3 #23: 解析 ``--phase_duration_limits`` CLI 字符串。

    输入格式: 逗号分隔的 3 个非负浮点,顺序对应 (R, Y, G) 三种灯态
    的最大持续秒数;``CycleState.compute_cycle_features`` 用其
    反推"距下次相位变化"的剩余时间。留空 / ``None`` / ``"None"`` 触发
    调用方 fallback 到模型 ``__init__`` 默认值 ``(38.0, 47.0, 2.0)``。
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


# Phase 3 #23: 把 ``CycleStateTrajectoryGenerator.__init__`` 中硬编码的
# ``phase_duration_limits=(38.0, 47.0, 2.0)`` 暴露到 CLI;允许复现 / 消融实验
# 替换为不同数据集的相位持续时间上限。空字符串或 None 触发 ``__init__`` 默认值。
parser.add_argument(
    "--phase_duration_limits",
    default=None,
    type=_parse_phase_duration_limits,
    help="Phase 3 #23: 逗号分隔的 3 个非负浮点 (R,Y,G),对应 ``phase_duration_limits`` "
    "buffer。例如 '38.0,47.0,2.0'。空字符串或 None 触发模型 ``__init__`` 默认值 "
    "(38.0, 47.0, 2.0),保持向后兼容。",
)

parser.add_argument("--use_gpu", default=1, type=int)
parser.add_argument("--gpu_num", default="2", type=str)
# Phase 5 #5 修复：删除模块级硬编码 ``CUDA_VISIBLE_DEVICES = '2'``。
# 此前在模块加载时无条件写入 '2'，覆盖任何用户 / shell 端
# ``CUDA_VISIBLE_DEVICES`` 设置，使得 --gpu_num 形同虚设。
# 当前改为：完全交给用户在 shell 端 export ``CUDA_VISIBLE_DEVICES``，
# 或者在调用 torch 之前再读 ``os.environ.get("CUDA_VISIBLE_DEVICES")``。
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


class BestAdeTracker:
    """Phase 5 #14:替代原来的模块级 ``best_ade = 100`` 全局变量。

    之前 ``best_ade`` 是模块级 Python 全局,在 ``global best_ade``
    模式下被 ``main`` 函数读写。这种写法有几个隐患:
    1. 一旦脚本被 ``import train`` 后,``best_ade`` 就成了该进程
       的全局状态,任何执行 ``from train import best_ade`` 或
       ``train.best_ade = X`` 都会污染训练循环;
    2. 多进程 / 多实验复用一个解释器时,上一次跑出的 best_ade
       会"穿越"到本次 run,造成新 run 误判 best;
    3. 单元测试无法隔离地构造"尚未达到 best"的状态。

    改用 ``BestAdeTracker`` 实例化到 ``main`` 局部作用域后,所有
    读写都通过显式方法进行,避免上述问题;同时 ``update`` 返回
    是否为新的 best,语义比 ``is_best = ade < best_ade`` 一行更清晰。
    """

    INITIAL_VALUE = 100.0

    def __init__(self, initial=None):
        if initial is None:
            initial = self.INITIAL_VALUE
        self._best_ade = float(initial)

    @property
    def value(self):
        return self._best_ade

    def update(self, ade):
        """更新 best_ade,返回 ``(is_best, new_best)`` 元组。"""
        is_best = ade < self._best_ade
        if is_best:
            self._best_ade = float(ade)
        return is_best, self._best_ade

    def restore_from_checkpoint(self, ckpt_best_ade):
        """从 checkpoint 恢复 best_ade,容忍 tensor/scalar/None。"""
        if ckpt_best_ade is None:
            return
        if torch.is_tensor(ckpt_best_ade):
            ckpt_best_ade = ckpt_best_ade.item()
        try:
            self._best_ade = float(ckpt_best_ade)
        except (TypeError, ValueError):
            # 非法类型时静默回退到当前 best,不破坏训练循环
            return


def build_num_val_samples_signature():
    """Phase 4 #21 契约: 声明 ``num_val_samples`` 必为正 int, 并锁定
    checkpoint 中 key 名。

    用途:
        - 训练时 ``save_checkpoint`` 必须把 ``num_val_samples`` 写进
          checkpoint 字典(键名固定为 ``"num_val_samples"``);
        - 加载时 ``restore_from_checkpoint`` 容忍 ``None`` / 缺失 /
          非法值, 但**不**允许悄悄丢失信息 — 任何不一致都要走
          ``alignment_warnings`` 列表, 供测试 / 上层 logger 引用。

    Returns:
        dict: 结构化契约描述, 包含所有关键字段名与约束。
    """
    return {
        "checkpoint_key": "num_val_samples",
        "runtime_arg": "num_val_samples",
        "eval_arg": "num_samples",
        "must_persist_positive_int": True,
    }


class NumValSamplesTracker:
    """Phase 4 #21: 跟踪训练时使用的 best-of-K 采样次数 ``num_val_samples``。

    背景
    ----
    训练内验证 (``validate`` 中 ``for _ in range(args.num_val_samples)``)
    与离线评估 (``evaluate_model.py`` 中 ``for _ in range(args.num_samples)``)
    都各自使用一个 K 采样数;若两者不一致, ``best_ade`` 与最终
    ``test`` 评估就**不可比**, 容易把"采样次数变多带来的误差下降"
    误归功于模型改进。

    本 tracker 把训练时实际使用的 K 值写进 checkpoint, 加载时自动
    与 ``args.num_val_samples`` 对齐校验:

    - **完全一致** (int 与 int 相等): 静默通过, 不污染日志;
    - **缺失 / None** (旧 checkpoint 升级上来): 升级期 warning,
      但**不**抛异常, 避免破坏加载流程;
    - **类型错误** (str / float / 负数 / 0): 升级期 warning, 同样
      静默回退, 不破坏训练;
    - **值不一致**: 升级期 warning, 提示"checkpoint K=20 但当前
      args.num_val_samples=4, 评估结果不可比"。

    该设计沿用 #14 ``BestAdeTracker`` 的"实例化到 main 局部 + 容忍
    None/tensor 类型"模式, 不引入模块级全局。

    双状态模型
    ----------
    为避免旧 checkpoint 的 K 值污染当前运行时的 save_checkpoint,
    tracker 维护两个独立状态:

    - ``_runtime_num_val_samples``: 当前这次运行真正使用的 K 值,
      来自 ``__init__(num_val_samples=...)``, **永不**被
      ``restore_from_checkpoint`` 修改;
    - ``_checkpoint_num_val_samples``: 从旧 checkpoint 读出来的
      历史 K 值, 只用于 ``check_alignment`` 做诊断比对,
      不参与 ``checkpoint_payload()`` 的写回逻辑。
    """

    def __init__(self, num_val_samples=None):
        """构造时记录当前 args 的 K 值, 缺失则记 ``None``。"""
        self._runtime_num_val_samples = (
            int(num_val_samples) if num_val_samples is not None else None
        )
        self._checkpoint_num_val_samples = None

    @property
    def value(self):
        """返回当前运行时的 K 值, 保持向后兼容。"""
        return self._runtime_num_val_samples

    def checkpoint_payload(self):
        """返回要写入 checkpoint 字典的 K 值。

        必须永远返回当前运行时 K (``_runtime_num_val_samples``),
        **不允许**返回旧 checkpoint 中恢复的历史 K 值。
        """
        return self._runtime_num_val_samples

    def restore_from_checkpoint(self, ckpt_num_val_samples):
        """从 checkpoint 恢复历史 K 值到 ``_checkpoint_num_val_samples``。

        只更新对齐诊断用的历史 K, **不覆盖** ``_runtime_num_val_samples``。
        容忍缺失/类型错误, 但不抛异常。
        """
        if ckpt_num_val_samples is None:
            return
        if torch.is_tensor(ckpt_num_val_samples):
            try:
                ckpt_num_val_samples = ckpt_num_val_samples.item()
            except (RuntimeError, ValueError):
                return
        try:
            int_value = int(ckpt_num_val_samples)
        except (TypeError, ValueError):
            return
        if int_value <= 0:
            return
        self._checkpoint_num_val_samples = int_value

    def check_alignment(self, args_num_val_samples):
        """比较 checkpoint 中的 K 与 ``args.num_val_samples``。

        使用的是 ``_checkpoint_num_val_samples``(历史 K)
        而非 ``_runtime_num_val_samples``(当前 K),
        确保诊断语义是"旧 checkpoint 与当前 args 是否一致"。

        Args:
            args_num_val_samples: 当前 CLI / 配置文件中的 K 值。

        Returns:
            tuple[bool, str]: ``(is_aligned, message)``。
            - ``is_aligned=True`` 表示两边一致或升级期无法判定;
            - ``message`` 是给人 / logger 看的诊断信息, 即使对齐
              也可能非空(例如 "checkpoint 无 num_val_samples, 沿用
              当前 args")。
        """
        ckpt_k = self._checkpoint_num_val_samples
        if ckpt_k is None:
            return (
                True,
                "checkpoint 中缺失 num_val_samples (旧版 / Phase 4 #21 "
                "升级前), 沿用当前 args.num_val_samples={}".format(
                    args_num_val_samples
                ),
            )
        if int(args_num_val_samples) == ckpt_k:
            return (
                True,
                "checkpoint num_val_samples={} 与 args.num_val_samples={} "
                "一致 (best-of-K 口径对齐)".format(
                    ckpt_k, args_num_val_samples
                ),
            )
        return (
            False,
            "[Phase 4 #21] checkpoint num_val_samples={} 与 "
            "args.num_val_samples={} 不一致! 训练内验证与离线评估的 "
            "best-of-K 采样次数不同, best_ade 与最终 test 评估不可比, "
            "请在 evaluate_model.py 中显式指定 --num_samples {}".format(
                ckpt_k, args_num_val_samples,
                ckpt_k,
            ),
        )


TRAIN_STAGE_DEFAULTS = {
    "warmup": {
        "generator_only": True,
        "gan_weight": 0.0,
        "aux_queue_weight": 10.0,
        "aux_cycle_weight": 5.0,
        "teacher_forcing_ratio": 0.8,
        "grad_clip": 1.0,
        "rollout_residual_scale": 0.35,
        "decoder_state_residual_scale": 1.0,
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
        "decoder_state_residual_scale": 1.0,
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
        "decoder_state_residual_scale": 1.0,
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
    "decoder_state_residual_scale": 1.0,
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


def validate_stage_consistency(args):
    """Phase 0 #19: 校验 ``TRAIN_STAGE_DEFAULTS`` 各字段间的联动一致性。

    该函数应当在 ``apply_stage_defaults(args)`` 之后、``main`` 真正开始训练
    之前被调用,把“看似合法但语义矛盾”的训练配置挡在启动阶段,避免在
    几十分钟训练后才暴露出 silent waste / dead config。

    约束分两类:

    * **硬错误 (raise ValueError)**: 数值越界或互斥字段同时为真;
      - ``gan_weight > 0`` 与 ``generator_only=True`` 互斥
        (GAN loss 在 generator_only=True 时被短路,gan_weight 实际不生效);
      - ``grad_clip < 0``、``rollout_residual_scale < 0`` 数值越界;
      - ``teacher_forcing_ratio`` 不在 ``[0, 1]`` 区间;
      - ``aux_queue_weight / aux_cycle_weight / aux_rollout_weight`` < 0。

    * **软警告 (logging.warning)**: 语义可疑但仍能跑(可能是消融场景);
      - ``train_stage == "adversarial"`` 但 ``gan_weight == 0``
        (adversarial 阶段无对抗信号,通常是协议错误);
      - ``aux_rollout_weight > 0`` 但 ``aux_queue_weight == 0``
        (rollout aux 属于 queue aux 家族,建议同步开关)。

    返回 ``args`` 本身以便链式调用。
    """
    issues = []
    warnings = []

    gan_weight = float(getattr(args, "gan_weight", 0.0) or 0.0)
    generator_only = bool(getattr(args, "generator_only", False))
    train_stage = getattr(args, "train_stage", None)

    # 硬约束 1a: gan_weight 不能为负
    if gan_weight < 0:
        issues.append(
            "gan_weight must be non-negative, got {:.4f}. A negative gan_weight "
            "would flip the adversarial optimization direction because total_loss "
            "contains g_loss * gan_weight.".format(gan_weight)
        )

    # 硬约束 1b: gan_weight > 0 与 generator_only=True 互斥
    if gan_weight > 0 and generator_only:
        issues.append(
            "gan_weight > 0 ({:.4f}) with generator_only=True is contradictory: "
            "the GAN loss is bypassed (g_loss is zeroed) when generator_only=True, "
            "so a non-zero gan_weight has no effect. Set generator_only=False to "
            "enable adversarial training, or set gan_weight=0 to keep generator-only."
            .format(gan_weight)
        )

    # 软警告 1: train_stage=adversarial 但 gan_weight=0
    if train_stage == "adversarial" and gan_weight == 0:
        warnings.append(
            "train_stage='adversarial' with gan_weight={} means the GAN loss is "
            "zeroed out: the adversarial stage is supposed to introduce GAN "
            "signal. Confirm this is an intentional warm-start (e.g. inheriting "
            "from a refined checkpoint before flipping on the discriminator).".format(gan_weight)
        )

    # 硬约束 2: 数值越界
    grad_clip = getattr(args, "grad_clip", None)
    if grad_clip is not None and grad_clip < 0:
        issues.append("grad_clip must be non-negative, got {}".format(grad_clip))

    rollout_residual_scale = getattr(args, "rollout_residual_scale", None)
    if rollout_residual_scale is not None and rollout_residual_scale < 0:
        issues.append(
            "rollout_residual_scale must be non-negative, got {}".format(rollout_residual_scale)
        )

    teacher_forcing_ratio = getattr(args, "teacher_forcing_ratio", None)
    if teacher_forcing_ratio is not None and not (0.0 <= teacher_forcing_ratio <= 1.0):
        issues.append(
            "teacher_forcing_ratio must be in [0, 1], got {}".format(teacher_forcing_ratio)
        )

    for field in ("aux_queue_weight", "aux_cycle_weight", "aux_rollout_weight"):
        value = getattr(args, field, None)
        if value is not None and value < 0:
            issues.append("{} must be non-negative, got {}".format(field, value))

    # Phase 3 #23: ``--phase_duration_limits`` 必须恰好 3 个非负浮点。
    # 早期由 ``_parse_phase_duration_limits`` 完成了 length / float 解析;
    # 这里再补一道防御,防止外部 ``apply_stage_defaults`` / checkpoint 恢复
    # 等代码路径写入非 tuple 或负数。
    phase_limits = getattr(args, "phase_duration_limits", None)
    if phase_limits is not None:
        if not (isinstance(phase_limits, (tuple, list))
                and len(phase_limits) == 3):
            issues.append(
                "phase_duration_limits 必须是长度为 3 的 tuple/list, got {!r}".format(
                    type(phase_limits).__name__
                )
            )
        elif any((not isinstance(v, (int, float))) or v < 0
                 for v in phase_limits):
            issues.append(
                "phase_duration_limits 元素必须为非负数, got {!r}".format(
                    phase_limits
                )
            )

    # 软警告 2: aux_rollout>0 但 aux_queue==0
    aux_queue = float(getattr(args, "aux_queue_weight", 0.0) or 0.0)
    aux_rollout = float(getattr(args, "aux_rollout_weight", 0.0) or 0.0)
    if aux_rollout > 0 and aux_queue == 0:
        warnings.append(
            "aux_rollout_weight > 0 ({:.4f}) with aux_queue_weight == 0 is unusual: "
            "the rollout aux loss is a sub-loss of the queue aux loss family. "
            "If you intentionally want only the rollout term, this is OK; "
            "otherwise consider setting aux_queue_weight > 0 as well.".format(aux_rollout)
        )

    # 报告
    for w in warnings:
        logging.warning("[stage-consistency] %s", w)

    if issues:
        joined = "\n  - ".join(issues)
        raise ValueError(
            "Stage consistency check failed (Phase 0 #19):\n  - {}".format(joined)
        )

    return args


def parse_rollout_queue_coefs(json_str):
    """Phase 3 #16: 解析 ``--rollout_queue_coefs_json`` CLI 参数。

    返回 ``RolloutQueueCoefs`` 实例:
    - 空字符串 / ``None`` -> 默认值 (与原硬编码一致)
    - 合法 JSON 对象 -> 解析后用 ``apply_rollout_coefs_override`` 做字段合并
    - 解析失败 (非 JSON / 不是 dict) -> 静默回退到默认值, 但通过 logging 警告一次,
      避免一个 CLI 错参让训练直接 crash。
    """
    if not json_str:
        return RolloutQueueCoefs()
    try:
        parsed = json.loads(json_str)
    except (TypeError, ValueError) as exc:
        logging.warning(
            "Failed to parse --rollout_queue_coefs_json=%r (%s); "
            "falling back to RolloutQueueCoefs() defaults.",
            json_str,
            exc,
        )
        return RolloutQueueCoefs()
    if not isinstance(parsed, dict):
        logging.warning(
            "--rollout_queue_coefs_json must be a JSON object, got %s; "
            "falling back to RolloutQueueCoefs() defaults.",
            type(parsed).__name__,
        )
        return RolloutQueueCoefs()
    coefs, invalid_keys = apply_rollout_coefs_override(
        RolloutQueueCoefs(), parsed
    )
    if invalid_keys:
        logging.warning(
            "--rollout_queue_coefs_json contains invalid values for %s; "
            "falling back to RolloutQueueCoefs() defaults for those fields.",
            ", ".join(sorted(invalid_keys)),
        )
    return coefs


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
            # Phase 5 #27: 移除与模型 ``build_traffic_context`` 重复的
            # ``seq_start_end`` 字段，避免两边都持有同一张量的冗余。
            # ``scene_groups`` / ``lane_grouping`` 为 adapter 专属补充字段，
            # 模型内部不依赖它们。
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
    """Phase 2 #20: 把 queue/cycle 辅助监督拆成更符合语义的分项损失。

    维度契约 (dim → loss-type mapping)
    ----------------------------------

    **Queue target** 末维 6 维严格按以下顺序, 与
    :func:`D2TP.models.compute_queue_targets` 和
    :func:`D2TP.models.build_queue_targets_signature` 一致::

        queue_reg_idx = [0, 1, 2, 3]  # regression (MSE)
            [0] queue_count
            [1] lane_wait_ratio
            [2] lane_release_ratio
            [3] lane_queue_length
        queue_cls_idx = [4, 5]        # binary (BCE-with-logits)
            [4] lane_stopline_occupancy
            [5] front_of_queue

    **Cycle target** 末维 6 维严格按以下顺序, 与
    :func:`D2TP.models.build_cycle_features` 和
    :func:`D2TP.models.build_cycle_features_signature` 一致::

        cycle[:, :3]   phase_one_hot     # classification (CrossEntropy)
        cycle[:, 3:5]  elapsed+remaining # regression (MSE)
        cycle[:, 5:6]  phase_change      # binary (BCE-with-logits)

    **Rollout 序列** 末维 6 维与单帧 queue target 同顺序(同
    ``compute_queue_targets``), 按 ``(T, batch, 6) -> (T*batch, 6)``
    flatten 后做统一的 reg/cls 切分; per-step supervision 权重均匀
    (无时间衰减/末帧偏置)。

    Main vs Rollout 监督方案的不对称性 (asymmetry)
    ----------------------------------------------

    本函数对 **main aux (queue/cycle)** 和 **rollout aux (queue)** 使用
    不同的监督范围,这是有意的设计而不是 bug:

    - **main aux**  (``queue_target_last`` / ``cycle_target_last``):
      queue 仅用 **最后一帧** 的目标 (``aux_info["queue_targets"][-1]``);
      cycle 取观测期 **最后 3 帧平均** (``aux_info["cycle_feature_seq"][-3:].mean(dim=0)``)
      作为更鲁棒的监督目标。原因: queue LSTM 的"门控
      hidden" 在观测期最后一步汇总, 仅取末帧目标与 hidden 语义对齐;
      cycle 灯态在 8 帧观测窗 (~0.8s) 内变化缓慢, 最后 3 帧平均可减少单帧噪声,
      同时保持语义正确性。

    - **rollout aux** (``queue_rollout_target_seq``):
      用 **完整预测期** 序列 (shape ``(T_pred, batch, 6)``)。原因:
      rollout 期间 queue LSTM 逐帧滚动, 需要 per-step 监督保证
      跨步稳定性, 仅取末帧监督 rollout 等于砍掉 11/12 的信号。

    任何把 main aux 改成"全序列监督"或把 rollout aux 改成"末帧监督"
    的重构都必须显式记录在 ``docs/PLAN.md`` 的活跃 backlog 中, 否则 :func:`tests.test_cyclestate_protocol`
    中 ``test_train_call_site_uses_last_frame_for_main_and_sequence_for_rollout``
    会失败。

    Args:
        queue_pred_last: 末步 queue aux 头输出, ``(batch, 6)``,
            末维顺序见上。
        queue_target_last: 同形 ``(batch, 6)``, 由
            :func:`D2TP.models.compute_queue_targets` 末帧给出。
        cycle_pred_last: 末步 cycle aux 头输出, ``(batch, 6)``。
        cycle_target_last: 同形 ``(batch, 6)``, 由
            :func:`D2TP.models.build_cycle_features` 末帧给出。
        queue_rollout_pred_seq: rollout 期 queue aux 头输出,
            ``(T_pred, batch, 6)``; 或 ``None``。
        queue_rollout_target_seq: rollout 期 queue target 序列,
            ``(T_pred, batch, 6)``; 或 ``None``。
        device: 0-loss 张量所在 device; 缺省时从入参推导。

    Returns:
        dict[str, torch.Tensor]: 分项损失, 键包括
        ``queue_reg_loss`` / ``queue_cls_loss`` /
        ``queue_rollout_reg_loss`` / ``queue_rollout_cls_loss`` /
        ``cycle_phase_loss`` / ``cycle_time_loss`` /
        ``cycle_change_loss`` / ``queue_main_loss`` /
        ``queue_rollout_loss`` / ``queue_total_loss`` /
        ``cycle_total_loss``。
    """
    # Phase 0 #4 + Phase 2 #20 契约: 末维必须严格 6, 否则下方 idx
    # 切片会静默错位(MSE 吃到 cls logits 或反之)。
    if queue_pred_last is not None and queue_target_last is not None:
        assert queue_pred_last.shape == queue_target_last.shape, (
            f"queue_pred_last.shape {tuple(queue_pred_last.shape)} must equal "
            f"queue_target_last.shape {tuple(queue_target_last.shape)} "
            "(Phase 0 #4 契约: aux 头拆分后 reg/cls 子空间必须严格对齐)."
        )
        assert queue_pred_last.shape[-1] == 6, (
            f"queue_pred_last last-dim must be 6 (4 reg + 2 cls), "
            f"got {queue_pred_last.shape[-1]} "
            "(Phase 2 #20 契约: 改 dim 顺序必须同步改 queue_reg_idx / "
            "queue_cls_idx 与 build_queue_targets_signature)."
        )
    if cycle_pred_last is not None and cycle_target_last is not None:
        assert cycle_pred_last.shape == cycle_target_last.shape, (
            f"cycle_pred_last.shape {tuple(cycle_pred_last.shape)} must equal "
            f"cycle_target_last.shape {tuple(cycle_target_last.shape)} "
            "(Phase 0 #4 契约: cycle 头拆分后 3 phase + 2 time + 1 change "
            "必须与 cycle_feature_seq 末帧对齐)."
        )
        assert cycle_pred_last.shape[-1] == 6, (
            f"cycle_pred_last last-dim must be 6 (3 phase + 2 time + "
            f"1 change), got {cycle_pred_last.shape[-1]} "
            "(Phase 2 #20 契约: 改 dim 顺序必须同步改下方 [:3]/[3:5]/[5:6] "
            "切片与 build_cycle_features_signature)."
        )
    if (
        queue_rollout_pred_seq is not None
        and queue_rollout_target_seq is not None
    ):
        assert (
            queue_rollout_pred_seq.shape == queue_rollout_target_seq.shape
        ), (
            f"queue_rollout_pred_seq.shape "
            f"{tuple(queue_rollout_pred_seq.shape)} must equal "
            f"queue_rollout_target_seq.shape "
            f"{tuple(queue_rollout_target_seq.shape)} "
            "(Phase 0 #4 契约: rollout 序列 pred/target 形状必须一致)."
        )
        assert (
            queue_rollout_pred_seq.shape[-1] == 6
        ), (
            "queue_rollout_pred_seq last-dim must be 6 (4 reg + 2 cls), "
            f"got {queue_rollout_pred_seq.shape[-1]} "
            "(Phase 2 #20 契约: 改 dim 顺序必须同步改 rollout 的 "
            "reg/cls 切分与 build_queue_targets_signature)."
        )

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

    # Phase 2 #20 契约索引常量 (集中声明, 便于源码守卫审计)
    queue_reg_idx = [0, 1, 2, 3]
    queue_cls_idx = [4, 5]

    if queue_pred_last is not None and queue_target_last is not None:
        # queue_reg_idx / queue_cls_idx 与 build_queue_targets_signature
        # 中的维度契约严格对齐; 任何 reorder 必须同步修改两边。
        losses["queue_reg_loss"] = F.mse_loss(
            queue_pred_last[:, queue_reg_idx],
            queue_target_last[:, queue_reg_idx],
        )
        losses["queue_cls_loss"] = F.binary_cross_entropy_with_logits(
            queue_pred_last[:, queue_cls_idx],
            queue_target_last[:, queue_cls_idx],
        )
    if (
        queue_rollout_pred_seq is not None
        and queue_rollout_target_seq is not None
        and queue_rollout_pred_seq.numel() > 0
    ):
        # rollout 序列在 (T*batch) 维度 flatten 后做同样的 reg/cls 切分;
        # 该 per-step 监督是均匀的 (无时间衰减/末帧偏置), 与 main aux 的
        # "末帧监督" 互补, 而不是替代 (设计见 docstring "asymmetry" 段)。
        rollout_pred_flat = queue_rollout_pred_seq.reshape(-1, queue_rollout_pred_seq.size(-1))
        rollout_target_flat = queue_rollout_target_seq.reshape(-1, queue_rollout_target_seq.size(-1))
        losses["queue_rollout_reg_loss"] = F.mse_loss(
            rollout_pred_flat[:, queue_reg_idx], rollout_target_flat[:, queue_reg_idx]
        )
        losses["queue_rollout_cls_loss"] = F.binary_cross_entropy_with_logits(
            rollout_pred_flat[:, queue_cls_idx], rollout_target_flat[:, queue_cls_idx]
        )

    if cycle_pred_last is not None and cycle_target_last is not None:
        # cycle_target[:, :3] 是 phase one-hot, 用 argmax 转 phase 索引供
        # cross_entropy 使用; [3:5] / [5:6] 与 build_cycle_features_signature
        # 维度契约严格对齐。
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
    """将任意形状张量压缩为单个标量,用作稳定性指标日志。

    ⚠️ 命名澄清(Phase 5 #10 修复):
    该函数**并不是**整个张量的 L2/Frobenius 范数。命名 ``mean_norm``
    实际上指代以下分情况计算:

    - ``tensor is None`` → ``0.0``(占位,避免日志崩溃)
    - 0 维(标量张量) → 标量自身
    - 1 维向量 → 所有元素的算术平均
    - ≥2 维张量 → 沿最后一维求 L2 范数,再在所有行上取平均
      (即 ``mean( ||x_i||_2 )`` for each row ``x_i``)

    这种"先求每行 L2 范数再求平均"的语义,对**特征图 / hidden
    sequence** 这类 `(T, N, D)` 形状的张量更直观;但对 0/1 维
    情况则退化为对应特殊定义。若需要严格的整体 L2 范数,
    请使用 ``torch.norm(tensor)``。

    返回值永远是 Python ``float``,可直接写入 TensorBoard
    ``add_scalar`` 与 logging。
    """
    if tensor is None:
        return 0.0
    if tensor.numel() == 0:
        return 0.0
    if tensor.dim() == 0:
        # 标量张量:直接返回其值(不是"0 维向量的 L2 范数")
        return float(tensor.detach().item())
    if tensor.dim() == 1:
        # 一维向量:整体算术平均(不是 L2 范数)
        return float(tensor.detach().float().mean().item())
    # 二维及更高:对每行(沿最后一维)求 L2 范数,再在行上取平均
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
            debug_info.get("decoder_state_step_residual_norm")
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
    legacy_state = dict(state_dict)

    # Phase 6 #45: 兼容旧版单头 aux checkpoint。
    # 旧 CycleState 使用:
    #   queue_aux_head: 6 = [4 reg, 2 cls]
    #   cycle_aux_head: 6 = [3 phase, 2 time, 1 change]
    # 新版拆成独立子头后,加载侧需要显式做一次按槽位切分,否则
    # evaluate/train 都会因为 missing/unexpected keys 直接失败。
    queue_weight = legacy_state.pop("queue_aux_head.weight", None)
    queue_bias = legacy_state.pop("queue_aux_head.bias", None)
    if queue_weight is not None and queue_bias is not None:
        legacy_state.setdefault("queue_aux_reg_head.weight", queue_weight[:4].clone())
        legacy_state.setdefault("queue_aux_reg_head.bias", queue_bias[:4].clone())
        legacy_state.setdefault("queue_aux_cls_head.weight", queue_weight[4:].clone())
        legacy_state.setdefault("queue_aux_cls_head.bias", queue_bias[4:].clone())

    cycle_weight = legacy_state.pop("cycle_aux_head.weight", None)
    cycle_bias = legacy_state.pop("cycle_aux_head.bias", None)
    if cycle_weight is not None and cycle_bias is not None:
        legacy_state.setdefault(
            "cycle_aux_phase_head.weight", cycle_weight[:3].clone()
        )
        legacy_state.setdefault(
            "cycle_aux_phase_head.bias", cycle_bias[:3].clone()
        )
        legacy_state.setdefault(
            "cycle_aux_time_head.weight", cycle_weight[3:5].clone()
        )
        legacy_state.setdefault(
            "cycle_aux_time_head.bias", cycle_bias[3:5].clone()
        )
        legacy_state.setdefault(
            "cycle_aux_change_head.weight", cycle_weight[5:6].clone()
        )
        legacy_state.setdefault(
            "cycle_aux_change_head.bias", cycle_bias[5:6].clone()
        )

    compatible_state = {}
    skipped = []
    for key, value in legacy_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible_state[key] = value
        else:
            skipped.append(key)
    model_state.update(compatible_state)
    model.load_state_dict(model_state)
    return skipped


def reapply_phase_duration_limits_if_overridden(model, phase_duration_limits):
    """若调用方显式覆盖了 ``phase_duration_limits``，则在 checkpoint
    加载后把该 buffer 重新写回，避免被旧 checkpoint 的同名 buffer 覆盖。

    ``phase_duration_limits`` 仍走 ``register_buffer``，所以这里必须用
    原地 ``copy_`` 保持 buffer 身份不变，继续受 ``state_dict`` / ``to(device)``
    管理。
    """
    if phase_duration_limits is None or not hasattr(model, "phase_duration_limits"):
        return
    model.phase_duration_limits.copy_(
        torch.tensor(
            phase_duration_limits,
            dtype=model.phase_duration_limits.dtype,
            device=model.phase_duration_limits.device,
        )
    )


def main(args):
    """训练入口。"""
    apply_stage_defaults(args)
    # Phase 0 #19: 在进入随机种子/数据加载之前做一次 TRAIN_STAGE_DEFAULTS 联动一致性
    # 校验,把 ``gan_weight > 0 + generator_only=True`` 之类 silent 矛盾挡在启动阶段。
    validate_stage_consistency(args)
    # Phase 4 #22: ``disable_aux_losses`` 统一主开关。把所有 aux 权重置零，
    # 配合 models.py 中四个 disable 标志位的强制开启，使 CycleState 在行为上等价于
    # 全消融模式（功能上对齐 baseline），保证消融实验的公平性。
    if getattr(args, "disable_aux_losses", False) and args.model_type == "cyclestate":
        args.aux_queue_weight = 0.0
        args.aux_cycle_weight = 0.0
        args.aux_rollout_weight = 0.0
        logging.info(
            "[Phase 4 #22] disable_aux_losses=ON: 所有 CycleState 特有功能已统一关闭"
            "（state_gating / queue_rollout / lane_queue_anchor / decoder_state_residual"
            " 全为 True，aux 权重均为 0），模型在功能上等价于全消融基线。"
        )
    # DE-3: ``minimal_viable_mode`` 隐式把 aux 权重置零，因为最简版本来就不
    # 消费任何 aux loss；保留 weight > 0 会让 ``compute_structured_aux_losses``
    # 内部尝试去拿 ``debug_last_aux`` 里与 DE-3 路径不一致的字段，可能引入
    # silent 噪声梯度。
    if (
        getattr(args, "minimal_viable_mode", False)
        and args.model_type == "cyclestate"
    ):
        args.aux_queue_weight = 0.0
        args.aux_cycle_weight = 0.0
        args.aux_rollout_weight = 0.0
        logging.info(
            "[DE-3] minimal_viable_mode=ON: 模型在内部强制开启 5 个 disable 开关，"
            "并把 aux 权重全部置零；'观测期最后时刻的 queue/cycle hidden' "
            "会直接拼接到 decoder 初始化向量后面。"
        )
    # DE-1: ``oracle_inject_mode`` 隐式把 aux 权重置零，原因与
    # ``minimal_viable_mode`` 相同：oracle 直注下模型不消费任何 aux 头
    # 也不消费 queue/cycle LSTM hidden，保留 weight > 0 只会引入 silent
    # 噪声梯度。
    if (
        getattr(args, "oracle_inject_mode", False)
        and args.model_type == "cyclestate"
    ):
        args.aux_queue_weight = 0.0
        args.aux_cycle_weight = 0.0
        args.aux_rollout_weight = 0.0
        logging.info(
            "[DE-1] oracle_inject_mode=ON: 模型在内部强制开启 5 个 disable 开关，"
            "并把 aux 权重全部置零；'单步 oracle 特征 (10 dim)' 会直接拼接到 "
            "``pred_lstm_model`` 的输入后面,等价于 'oracle 交通状态 → decoder 内部信号'。"
        )
    # AR-1: ``ar1_direct_inject_mode`` 隐式把 aux 权重置零，原因与
    # ``minimal_viable_mode`` / ``oracle_inject_mode`` 相同：AR-1 隐含
    # ``minimal_viable_mode=True`` 且不消费任何 aux 头 / rollout hidden
    # 也不消费 state_gating。
    if (
        getattr(args, "ar1_direct_inject_mode", False)
        and args.model_type == "cyclestate"
    ):
        args.aux_queue_weight = 0.0
        args.aux_cycle_weight = 0.0
        args.aux_rollout_weight = 0.0
        # AR-1 隐含 minimal_viable_mode=True,这里也把它置 True 以便日志
        # 与 protocol-check 完整对齐; 模型内部会再次确认并强制开启 5 个
        # disable 开关。
        args.minimal_viable_mode = True
        logging.info(
            "[AR-1] ar1_direct_inject_mode=ON: 模型在内部强制开启 5 个 disable 开关，"
            "并把 aux 权重全部置零；'观测期最后时刻的 [queue_last, cycle_last] (48 维)' "
            "会同时拼接到: 1) decoder 初始化向量 (与 DE-3 一致), 2) ``pred_lstm_model`` "
            "每步输入, 3) ``pred_hidden2pos`` 输出投影。"
        )
    # AR-2: ``ar2_multiplicative_gating_mode`` 隐式把 aux 权重置零, 原因与
    # AR-1 相同: AR-2 隐含 ``minimal_viable_mode=True`` 且不消费任何 aux 头
    # / rollout hidden / state_gating。AR-2 与 AR-1 / oracle_inject_mode
    # 互斥, 模型内部会校验。
    if (
        getattr(args, "ar2_multiplicative_gating_mode", False)
        and args.model_type == "cyclestate"
    ):
        args.aux_queue_weight = 0.0
        args.aux_cycle_weight = 0.0
        args.aux_rollout_weight = 0.0
        # AR-2 隐含 minimal_viable_mode=True,这里也把它置 True 以便日志
        # 与 protocol-check 完整对齐; 模型内部会再次确认并强制开启 5 个
        # disable 开关。
        args.minimal_viable_mode = True
        logging.info(
            "[AR-2] ar2_multiplicative_gating_mode=ON: 模型在内部强制开启 5 个 disable 开关，"
            "并把 aux 权重全部置零；'观测期最后时刻的 [queue_last, cycle_last] (48 维)' "
            "会用于: 1) decoder 初始化向量 (与 DE-3 一致), 2) ``ar2_hidden_gate`` "
            "学习 per-step 逐元素 sigmoid 门控, 然后 ``pred_lstm_hidden = "
            "pred_lstm_hidden * gate`` (乘法调制, 与 AR-1 的加性拼接不同)。"
        )
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
        # C2-1 第一变体: 与 model_type 正交, 既能作用于 d2tpred (cleanest
        # isolation test) 也能作用于 cyclestate (与 --minimal_viable_mode
        # 组合验证 "C2-1 + state hidden init 拼接" 联合效果)。
        c2_1_trajectory_level_mode=bool(
            getattr(args, "c2_1_trajectory_level_mode", False)
        ),
    )
    if args.model_type == "cyclestate":
        ablation_cfg = AblationConfig.from_args(args)
        model_kwargs.update(ablation_cfg.to_model_kwargs())
        model_kwargs["rollout_residual_scale"] = args.rollout_residual_scale
        model_kwargs["decoder_state_residual_scale"] = (
            args.decoder_state_residual_scale
        )
        model_kwargs["detach_rollout_state"] = args.detach_rollout_state
        # Phase 3 #23: 把 ``--phase_duration_limits`` 透传到模型构造函数;
        # ``None`` 触发 ``__init__`` 默认值 ``(38.0, 47.0, 2.0)``, 与原硬编码
        # 行为一致 (向后兼容)。显式覆盖时模型 buffer 会被替换, 但 ``register_buffer``
        # 机制保证其能随 ``.to(device)`` / ``.cuda()`` 正确迁移。
        if getattr(args, "phase_duration_limits", None) is not None:
            model_kwargs["phase_duration_limits"] = tuple(
                args.phase_duration_limits
            )
        # Phase 3 #16: 把 ``--rollout_queue_coefs_json`` 解析后的 dataclass
        # 透传给 ``CycleStateTrajectoryGenerator.__init__``; 解析失败 / 空字符串
        # 都回退到 ``RolloutQueueCoefs()`` 默认值, 与原硬编码行为一致。
        model_kwargs["rollout_queue_coefs"] = parse_rollout_queue_coefs(
            getattr(args, "rollout_queue_coefs_json", "")
        )
        # DE-3: 把 ``--minimal_viable_mode`` 透传到模型构造函数。
        # 注意: 这个开关与 ``disable_aux_losses`` 互不冲突; 模型内部会先把
        # ``minimal_viable_mode=True`` 解释为强制开启 5 个 disable 开关,
        # 然后再独立修改 ``encoded_before_noise_hidden`` 的拼接方式。
        model_kwargs["minimal_viable_mode"] = bool(
            getattr(args, "minimal_viable_mode", False)
        )
        # DE-1: 把 ``--oracle_inject_mode`` 透传到模型构造函数; 必须在
        # 训练和推理两侧都传,否则 ``pred_lstm_model`` 的输入维度与
        # checkpoint 不匹配会报 shape error。
        model_kwargs["oracle_inject_mode"] = bool(
            getattr(args, "oracle_inject_mode", False)
        )
        # AR-1: 把 ``--ar1_direct_inject_mode`` 透传到模型构造函数;
        # 必须在训练和推理两侧都传,否则 ``pred_lstm_model`` /
        # ``pred_hidden2pos`` 的输入维度与 checkpoint 不匹配会报
        # shape error。AR-1 与 ``oracle_inject_mode`` 互斥,模型内部
        # 会校验。
        model_kwargs["ar1_direct_inject_mode"] = bool(
            getattr(args, "ar1_direct_inject_mode", False)
        )
        # AR-2: 把 ``--ar2_multiplicative_gating_mode`` 透传到模型构造函数;
        # 必须在训练和推理两侧都传,否则 ``ar2_hidden_gate`` 模块不会被
        # 创建,加载 checkpoint 时会报 ``unexpected key`` / ``missing key``
        # 错误。AR-2 与 ``oracle_inject_mode`` / ``ar1_direct_inject_mode``
        # 互斥,模型内部会校验。
        model_kwargs["ar2_multiplicative_gating_mode"] = bool(
            getattr(args, "ar2_multiplicative_gating_mode", False)
        )
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
    # Phase 4 #22: 日志口径必须反映 ``disable_aux_losses=True`` 强制开启四个子开关
    # 之后的"模型实际生效状态", 而不是原始 ``args.disable_*``。这样才能保证消融实验
    # 日志与运行时的真实行为一致, 满足可审计性。
    ablation_cfg = AblationConfig.from_args(args)
    _disable_aux = ablation_cfg.disable_aux_losses
    _effective_ablation = ablation_cfg.effective_flags()
    _eff_disable_state_gating = _effective_ablation["disable_state_gating"]
    _eff_disable_queue_rollout = _effective_ablation["disable_queue_rollout"]
    _eff_disable_lane_queue_anchor = _effective_ablation["disable_lane_queue_anchor"]
    _eff_disable_decoder_state_residual = _effective_ablation[
        "disable_decoder_state_residual"
    ]
    logging.info(
        "Training protocol | model_type=%s stage=%s val_split=%s lr=%.6f grad_clip=%.3f generator_only=%s gan_weight=%.3f aux_queue=%.3f aux_rollout=%.3f aux_cycle=%.3f rollout_residual_scale=%.3f detach_rollout_state=%s phase_duration_limits=%s disable_state_gating(eff)=%s disable_queue_rollout(eff)=%s disable_lane_queue_anchor(eff)=%s disable_decoder_state_residual(eff)=%s disable_aux_losses=%s teacher_forcing=%.3f",
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
        # Phase 3 #23: 把 ``--phase_duration_limits`` 的“实际生效值”打到日志
        # 口径中; 显式覆盖时打印用户传值, ``None`` 时打印 ``__init__`` 默认
        # ``(38.0, 47.0, 2.0)``, 让审计员一眼看出本轮实验用的是哪一组相位
        # 持续时间上限。
        (
            tuple(args.phase_duration_limits)
            if getattr(args, "phase_duration_limits", None) is not None
            else (38.0, 47.0, 2.0)
        ),
        _eff_disable_state_gating,
        _eff_disable_queue_rollout,
        _eff_disable_lane_queue_anchor,
        _eff_disable_decoder_state_residual,
        _disable_aux,
        args.teacher_forcing_ratio,
    )
    # Phase 3 #16: 训练协议日志单独打印一行 ``RolloutQueueCoefs`` 实际生效的
    # 物理系数 (默认或被 --rollout_queue_coefs_json 覆盖后的值), 方便在日志中
    # 直接看到 warmup/refine 阶段是否做了系数调整, 而不用回溯 CLI 完整命令。
    if args.model_type == "cyclestate":
        active_coefs = model.rollout_queue_coefs
        logging.info(
            "Rollout queue coefs | waiting_ratio red_inc=%.4f yellow_inc=%.4f green_dec=%.4f"
            " | release_ratio green_inc=%.4f red_dec=%.4f yellow_dec=%.4f"
            " | lane_queue_length red_inc=%.4f yellow_inc=%.4f green_dec=%.4f phase_change_inc=%.4f"
            " | stopline_occupancy red_inc=%.4f green_dec=%.4f"
            " | front_of_queue red_inc=%.4f green_dec=%.4f"
            " | stop_dist pred_speed_dec=%.4f step_discount_dec=%.4f phase_change_inc=%.4f"
            " | queue_count_stopline_weight=%.4f"
            " | lane_density_prev=%.4f lane_density_lane=%.4f"
            " | lane_mean_speed_prev=%.4f lane_mean_speed_pred=%.4f",
            active_coefs.waiting_ratio_red_inc,
            active_coefs.waiting_ratio_yellow_inc,
            active_coefs.waiting_ratio_green_dec,
            active_coefs.release_ratio_green_inc,
            active_coefs.release_ratio_red_dec,
            active_coefs.release_ratio_yellow_dec,
            active_coefs.lane_queue_length_red_inc,
            active_coefs.lane_queue_length_yellow_inc,
            active_coefs.lane_queue_length_green_dec,
            active_coefs.lane_queue_length_phase_change_inc,
            active_coefs.stopline_occupancy_red_inc,
            active_coefs.stopline_occupancy_green_dec,
            active_coefs.front_of_queue_red_inc,
            active_coefs.front_of_queue_green_dec,
            active_coefs.stop_dist_pred_speed_dec,
            active_coefs.stop_dist_step_discount_dec,
            active_coefs.stop_dist_phase_change_inc,
            active_coefs.queue_count_stopline_weight,
            active_coefs.lane_density_prev_weight,
            active_coefs.lane_density_lane_queue_weight,
            active_coefs.lane_mean_speed_prev_weight,
            active_coefs.lane_mean_speed_pred_weight,
        )
    # Phase 5 #14:把模块级 ``best_ade`` 替换为 ``BestAdeTracker`` 实例,
    # 状态从模块全局转为 ``main`` 局部变量;checkpoint 加载时同步
    # 把 ``ckpt["best_ade"]`` 灌入 tracker,语义与原代码完全一致。
    best_ade_tracker = BestAdeTracker()
    # Phase 4 #21:跟踪训练时使用的 best-of-K 采样次数,checkpoint 加载
    # 时与 ``args.num_val_samples`` 对齐校验,确保 ``best_ade`` 与
    # 最终 ``test`` 评估的 best-of-K 口径可比。
    num_val_samples_tracker = NumValSamplesTracker(
        num_val_samples=getattr(args, "num_val_samples", None)
    )
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("Restoring from checkpoint {}".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=args.device)
            if args.model_type == "cyclestate":
                skipped_keys = maybe_load_compatible_weights(
                    model, checkpoint["state_dict"]
                )
                reapply_phase_duration_limits_if_overridden(
                    model,
                    getattr(args, "phase_duration_limits", None),
                )
                # Phase 4 #29: 从 checkpoint 恢复归一化参数,
                # 确保断点续训使用与原始训练一致的量纲。
                model.load_norm_params(checkpoint.get("norm_params"))
                # Phase 0 #7 修复:兼容加载分支也要把 ``start_epoch`` 恢复,
                # 否则断点续训时 epoch 计数永远从 CLI 默认值(通常为 0)重新
                # 开始,导致 LR scheduler / tensorboard / log 命名错位。
                # 行为与 else 分支(非 cyclestate 加载)保持一致:
                # 使用 ``checkpoint["epoch"]`` 自身,主循环 ``range(start_epoch,
                # num_epochs + 1)`` 会从该 epoch 继续(允许重跑上一 epoch)。
                if "epoch" in checkpoint:
                    args.start_epoch = checkpoint["epoch"]
                logging.info(
                    "=> warm-started CycleState from checkpoint, "
                    "skipped {} keys, start_epoch restored to {}".format(
                        len(skipped_keys), args.start_epoch
                    )
                )
            else:
                args.start_epoch = checkpoint["epoch"]
                model.load_state_dict(checkpoint["state_dict"])
            # Phase 5 #14:把 ckpt 中的 best_ade 灌入 tracker(如果存在)
            best_ade_tracker.restore_from_checkpoint(checkpoint.get("best_ade"))
            # Phase 4 #21:把 ckpt 中的 num_val_samples 灌入 tracker,
            # 并打印与 ``args.num_val_samples`` 的对齐诊断(mismatch 时
            # 走 ``logging.warning`` 而非抛异常,避免破坏加载流程)。
            num_val_samples_tracker.restore_from_checkpoint(
                checkpoint.get("num_val_samples")
            )
            is_aligned, alignment_msg = num_val_samples_tracker.check_alignment(
                getattr(args, "num_val_samples", None)
            )
            if is_aligned and "缺失" not in alignment_msg:
                logging.info("=> [Phase 4 #21] %s", alignment_msg)
            else:
                # 缺失(旧 ckpt) 或 不一致 时, 升级期 warning 但不抛异常
                if "缺失" in alignment_msg:
                    logging.info("=> [Phase 4 #21] %s", alignment_msg)
                else:
                    logging.warning("=> [Phase 4 #21] %s", alignment_msg)
            logging.info(
                "=> loaded checkpoint '{}' (epoch {}, best_ade={})".format(
                    args.resume,
                    checkpoint["epoch"],
                    "{:.4f}".format(best_ade_tracker.value),
                )
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    training_step = 3
    # Phase 5 #13:跨 epoch 单调递增的全局步数,供 ``D_train`` / ``train``
    # 把 tensorboard scalar 写到一个统一的时间轴上,避免 d_train_loss
    # 长期使用 epoch 作 step 时与 g_*_loss 不可对照的问题。
    global_step = 0
    for epoch in range(args.start_epoch, args.num_epochs + 1):
        gc.collect()
        # 先多更新几步判别器，再更新一次生成器。
        # D_step 必须在每个 epoch 起跑时重置为 2，避免 epoch 边界处 D/G
        # 调度计数器携带上一 epoch 的残值，破坏判别器两拍热身节奏。
        D_step = 2
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
                    global_step=global_step,
                )
                global_step += 1
                if should_run_validation(
                    args, epoch, batch_idx, len(train_loader)
                ):
                    ade = validate(args, model, val_loader, epoch, writer)
                    is_best, best_ade = best_ade_tracker.update(ade)
                    save_checkpoint(
                        {
                            "epoch": epoch + 1,
                            "state_dict": model.state_dict(),
                            "best_ade": best_ade,
                            # Phase 4 #21: 记录训练时实际使用的 K 值,
                            # 加载时与 ``args.num_val_samples`` 对齐校验。
                            "num_val_samples": num_val_samples_tracker.checkpoint_payload(),
                            "optimizer": optimizer.state_dict(),
                            # Phase 4 #29: 持久化数据集归一化参数,
                            # 确保断点续训与评估使用一致量纲。
                            "norm_params": model.norm_params() if hasattr(model, "norm_params") else None,
                        },
                        is_best,
                        os.path.join(args.checkpoint_dir, f"checkpoint{epoch}.pth.tar"),
                        best_filename=os.path.join(args.checkpoint_dir, "model_best.pth.tar"),
                    )
                continue
            if D_step>0:
                D_train(args,len(train_loader), model,batch_idx,batch,Discriminator, optimizer_d, epoch, training_step, writer, global_step=global_step)
                D_step=D_step-1
                global_step += 1
            else:
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
                    global_step=global_step,
                )
                D_step=2
                global_step += 1
                if should_run_validation(
                    args, epoch, batch_idx, len(train_loader)
                ):
                    ade = validate(args, model, val_loader, epoch, writer)
                    is_best, best_ade = best_ade_tracker.update(ade)

                    save_checkpoint(
                        {
                            "epoch": epoch + 1,
                            "state_dict": model.state_dict(),
                            "best_ade": best_ade,
                            # Phase 4 #21: 同上, 把 K 值写进 ckpt
                            "num_val_samples": num_val_samples_tracker.checkpoint_payload(),
                            "optimizer": optimizer.state_dict(),
                            # Phase 4 #29: 持久化数据集归一化参数。
                            "norm_params": model.norm_params() if hasattr(model, "norm_params") else None,
                        },
                        is_best,
                        os.path.join(args.checkpoint_dir, f"checkpoint{epoch}.pth.tar"),
                        best_filename=os.path.join(args.checkpoint_dir, "model_best.pth.tar"),
                    )
    writer.close()


def train(
    args,
    lens,
    model,
    batch_idx,
    batch,
    Discriminator,
    optimizer,
    epoch,
    training_step,
    writer,
    global_step=0,
):
    """更新生成器。

    Phase 5 #13:生成器侧 tensorboard 标量与 ``D_train`` 一样使用
    ``global_step``，这样 g/d 曲线共享统一训练步数时间轴。
    """
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

    # Phase 5 #12:``seq_start_end.data`` 已弃用,改用 ``.tolist()``,
    # 既得到 Python int 元组,也避免对 requires_grad 张量的 in-place
    # 行为依赖。语义与原代码完全一致(只是把 start/end 从
    # 0-d 长整型张量转成 Python int,索引语义不变)。
    for start, end in seq_start_end.tolist():
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
            # Phase 3 #24: 不只依赖最后一帧，取最后 3 帧平均作为更鲁棒的监督目标。
            # cycle_feature_seq 形状 (T_obs, batch, 6)，在 8 帧观测窗 (~0.8s) 内
            # 灯态变化缓慢，取最后 3 帧平均可以减少单帧噪声，同时保持语义正确性。
            cycle_feature_tail = aux_info["cycle_feature_seq"][-3:]
            cycle_target_last = cycle_feature_tail.mean(dim=0)
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
    writer.add_scalar("g_l2_loss", losses.avg, global_step)
    writer.add_scalar("g_ad_loss", g_losses.avg * args.gan_weight, global_step)
    writer.add_scalar("g_queue_reg_loss", aux_queue_reg_losses.avg, global_step)
    writer.add_scalar("g_queue_cls_loss", aux_queue_cls_losses.avg, global_step)
    writer.add_scalar("g_queue_rollout_reg_loss", aux_queue_rollout_reg_losses.avg, global_step)
    writer.add_scalar("g_queue_rollout_cls_loss", aux_queue_rollout_cls_losses.avg, global_step)
    writer.add_scalar("g_cycle_phase_loss", aux_cycle_phase_losses.avg, global_step)
    writer.add_scalar("g_cycle_time_loss", aux_cycle_time_losses.avg, global_step)
    writer.add_scalar("g_cycle_change_loss", aux_cycle_change_losses.avg, global_step)
    writer.add_scalar(
        "state_decoder_init_residual_norm",
        stability_metrics["decoder_state_init_residual_norm"],
        global_step,
    )
    writer.add_scalar(
        "state_decoder_step_residual_norm",
        stability_metrics["decoder_state_step_residual_norm"],
        global_step,
    )
    writer.add_scalar(
        "state_queue_rollout_hidden_norm",
        stability_metrics["queue_rollout_hidden_norm"],
        global_step,
    )
    writer.add_scalar(
        "state_pred_offset_norm",
        stability_metrics["pred_offset_norm"],
        global_step,
    )
    writer.add_scalar(
        "g_grad_norm",
        float(grad_norm.item() if grad_norm is not None else 0.0),
        global_step,
    )

def D_train(args,lens, model,batch_idx,batch,Discriminator, optimizer, epoch, training_step, writer, global_step=0):
    """更新判别器。

    Phase 5 #13:``writer.add_scalar("d_train_loss", ...)`` 的步数从
    ``epoch`` 改为 ``global_step``(由调用方 ``main`` 维护的全局
    训练步数),否则在长程训练里多 epoch 共享同一 ``epoch`` 步数
    会让 d_train_loss 的时间轴完全不可读。
    """
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

    # Phase 5 #13:tensorboard 步数从 epoch 改为 global_step(见函数 docstring)
    writer.add_scalar("d_train_loss", D_losses.avg, global_step)

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
