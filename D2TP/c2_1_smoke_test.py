"""C2-1 第一变体 (``C2-1-MV1``) smoke test: 验证 2 层 stacked trajectory encoder。

不依赖真实数据集, 直接用 ``torch.randn`` 构造 ``obs_traj`` / ``obs_traj_rel`` /
``obs_state`` / ``pred_state`` / ``seq_start_end``, 走一次 forward + backward,
检查:

1. ``TrajectoryGenerator`` 与 ``CycleStateTrajectoryGenerator`` 在
   ``c2_1_trajectory_level_mode=True`` 时, 都正确创建
   ``traj_lstm_layer2`` (2 层 stacked LSTMCell);
2. ``c2_1_trajectory_level_mode=False`` 时, ``traj_lstm_layer2`` 为 None
   (保持与历史 checkpoint 兼容);
3. forward 输出 shape 与 ``(pred_len, batch, 2)`` 一致;
4. loss 能正常 backward, 梯度能流到 ``traj_lstm_model`` 与
   ``traj_lstm_layer2``, 证明两层都参与实际优化;
5. ``pred_lstm_hidden_size`` / ``pred_lstm_model`` 输入维度
   **不**变 — C2-1-MV1 是孤立改动, 不影响 decoder 接口;
6. c2_1 与 state injection 模式 (DE-3 / DE-1 / AR-1 / AR-2) 正交,
   ``CycleStateTrajectoryGenerator`` 在 c2_1 模式下不应被强制开启
   ``minimal_viable_mode``。
"""

import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from models import CycleStateTrajectoryGenerator, TrajectoryGenerator  # noqa: E402


def _build_inputs(batch=8, obs_len=8, pred_len=12, num_scene=2, device="cpu"):
    torch.manual_seed(0)
    obs_traj_pos = torch.randn(obs_len, batch, 10, device=device)
    obs_traj_pos[:, :, 4] = (obs_traj_pos[:, :, 4] * 0.0).long()  # lane id
    obs_traj_pos[:, :, 9] = obs_traj_pos[:, :, 9] * 0.5 + 1.0  # direction
    obs_traj_rel = torch.randn(obs_len + pred_len, batch, 4, device=device)
    obs_state = torch.zeros(obs_len, batch, 4, device=device)
    obs_state[:, :, 0] = 100.0
    obs_state[:, :, 1] = 100.0
    obs_state[:, :, 2] = torch.randint(0, 3, (obs_len, batch), device=device).float()
    obs_state[:, :, 3] = torch.rand(obs_len, batch, device=device) * 30.0
    pred_state = torch.zeros(pred_len, batch, 4, device=device)
    pred_state[:, :, 0] = 100.0
    pred_state[:, :, 1] = 100.0
    pred_state[:, :, 2] = torch.randint(0, 3, (pred_len, batch), device=device).float()
    pred_state[:, :, 3] = torch.rand(pred_len, batch, device=device) * 30.0
    # 把 batch 切成 num_scene 个 scene, 每个 scene 4 个 agent (batch=8, num_scene=2)
    sizes = [batch // num_scene] * num_scene
    if sum(sizes) < batch:
        sizes[-1] += batch - sum(sizes)
    ends = torch.cumsum(torch.tensor(sizes), dim=0)
    starts = torch.zeros_like(ends)
    starts[1:] = ends[:-1]
    seq_start_end = torch.stack([starts, ends], dim=1)
    return obs_traj_rel, obs_traj_pos, obs_state, pred_state, seq_start_end


def _build_d2tpred(c2_1_mode=False):
    return TrajectoryGenerator(
        obs_len=8,
        pred_len=12,
        traj_lstm_input_size=2,
        traj_lstm_hidden_size=32,
        n_units=[32, 32, 32],
        n_heads=[4, 1],
        graph_network_out_dims=32,
        dropout=0.0,
        alpha=0.2,
        graph_lstm_hidden_size=32,
        noise_dim=(8,),
        noise_type="gaussian",
        light_input_size=5,
        embedding_size=64,
        light_embedding_size=32,
        c2_1_trajectory_level_mode=c2_1_mode,
    )


def _build_cyclestate(c2_1_mode=False, minimal_viable_mode=False):
    return CycleStateTrajectoryGenerator(
        obs_len=8,
        pred_len=12,
        traj_lstm_input_size=2,
        traj_lstm_hidden_size=32,
        n_units=[32, 32, 32],
        n_heads=[4, 1],
        graph_network_out_dims=32,
        dropout=0.0,
        alpha=0.2,
        graph_lstm_hidden_size=32,
        noise_dim=(8,),
        noise_type="gaussian",
        light_input_size=5,
        embedding_size=64,
        light_embedding_size=32,
        queue_lstm_hidden_size=32,
        cycle_lstm_hidden_size=16,
        minimal_viable_mode=minimal_viable_mode,
        c2_1_trajectory_level_mode=c2_1_mode,
    )


def _assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def test_c2_1_disabled_d2tpred_layer2_is_none():
    """``c2_1_trajectory_level_mode=False`` 时, d2tpred 不创建 layer2。"""
    model = _build_d2tpred(c2_1_mode=False)
    _assert_true(
        not model.c2_1_trajectory_level_mode,
        "c2_1_trajectory_level_mode=False 时, c2_1 flag 应为 False",
    )
    _assert_true(
        model.traj_lstm_layer2 is None,
        "c2_1_trajectory_level_mode=False 时, traj_lstm_layer2 应为 None",
    )


def test_c2_1_enabled_d2tpred_layer2_exists():
    """``c2_1_trajectory_level_mode=True`` 时, d2tpred 创建 layer2。"""
    model = _build_d2tpred(c2_1_mode=True)
    _assert_true(
        model.c2_1_trajectory_level_mode,
        "c2_1_trajectory_level_mode=True 时, c2_1 flag 应为 True",
    )
    _assert_true(
        model.traj_lstm_layer2 is not None,
        "c2_1_trajectory_level_mode=True 时, traj_lstm_layer2 应被创建",
    )
    # layer2 是 LSTMCell(32 → 32), 与 layer1 hidden_size 对齐
    _assert_true(
        model.traj_lstm_layer2.input_size == 32
        and model.traj_lstm_layer2.hidden_size == 32,
        "traj_lstm_layer2 应当是 LSTMCell(32, 32), got ({}, {})".format(
            model.traj_lstm_layer2.input_size,
            model.traj_lstm_layer2.hidden_size,
        ),
    )
    # pred_lstm_hidden_size 应与原版完全一致 (C2-1 不动 decoder 接口)
    _assert_true(
        model.pred_lstm_hidden_size == 32 + 32 + 32 + 8,  # light+traj+graph+noise
        "c2_1 不应改变 pred_lstm_hidden_size, expected {}, got {}".format(
            32 + 32 + 32 + 8, model.pred_lstm_hidden_size
        ),
    )


def test_c2_1_d2tpred_forward_backward():
    """c2_1 d2tpred 走一次 forward+backward, 验证 layer2 有梯度。"""
    model = _build_d2tpred(c2_1_mode=True)
    model.train()
    inputs = _build_inputs()
    pred_traj_fake_rel = model(*inputs, teacher_forcing_ratio=0.5)
    _assert_true(
        pred_traj_fake_rel.shape == (12, 8, 2),
        "c2_1 d2tpred forward 输出 shape 应该是 (12, 8, 2), got {}".format(
            pred_traj_fake_rel.shape
        ),
    )
    loss = pred_traj_fake_rel.pow(2).sum()
    loss.backward()
    layer1_grad = model.traj_lstm_model.weight_hh.grad
    layer2_grad = model.traj_lstm_layer2.weight_hh.grad
    _assert_true(
        layer1_grad is not None and torch.isfinite(layer1_grad).all(),
        "traj_lstm_model 应该有有效梯度",
    )
    _assert_true(
        layer2_grad is not None and torch.isfinite(layer2_grad).all(),
        "traj_lstm_layer2 应该有有效梯度",
    )


def test_c2_1_d2tpred_disabled_forward_still_works():
    """c2_1 d2tpred 默认模式 forward+backward 也应能跑通, 保证向后兼容。"""
    model = _build_d2tpred(c2_1_mode=False)
    model.train()
    inputs = _build_inputs()
    pred_traj_fake_rel = model(*inputs, teacher_forcing_ratio=0.5)
    _assert_true(
        pred_traj_fake_rel.shape == (12, 8, 2),
        "默认 d2tpred forward 输出 shape 应该是 (12, 8, 2), got {}".format(
            pred_traj_fake_rel.shape
        ),
    )
    loss = pred_traj_fake_rel.pow(2).sum()
    loss.backward()


def test_c2_1_cyclestate_orthogonal_to_minimal_viable():
    """c2_1 与 minimal_viable_mode 正交 — c2_1 不强制开启 minimal_viable。"""
    model = _build_cyclestate(c2_1_mode=True, minimal_viable_mode=False)
    _assert_true(
        model.c2_1_trajectory_level_mode,
        "c2_1_trajectory_level_mode 应为 True",
    )
    _assert_true(
        not model.minimal_viable_mode,
        "c2_1 不应强制开启 minimal_viable_mode (与 state injection 模式正交)",
    )
    _assert_true(
        model.traj_lstm_layer2 is not None,
        "c2_1 模式 cyclestate 也应创建 traj_lstm_layer2",
    )
    # 不在 DE-3 模式时, pred_lstm_hidden_size 与原版一致
    _assert_true(
        model.pred_lstm_hidden_size == 32 + 32 + 32 + 8,
        "非 DE-3 模式 pred_lstm_hidden_size 应当为 {}, got {}".format(
            32 + 32 + 32 + 8, model.pred_lstm_hidden_size
        ),
    )


def test_c2_1_cyclestate_combined_with_minimal_viable():
    """c2_1 + minimal_viable_mode 组合: layer2 存在, 且 pred_lstm_hidden_size 扩大。"""
    model = _build_cyclestate(c2_1_mode=True, minimal_viable_mode=True)
    _assert_true(
        model.minimal_viable_mode,
        "explicit minimal_viable_mode=True 应被保留",
    )
    _assert_true(
        model.traj_lstm_layer2 is not None,
        "c2_1 + minimal_viable 组合也应创建 traj_lstm_layer2",
    )
    # minimal_viable_mode=True 时, pred_lstm_hidden_size 应加上 queue+cycle
    base = 32 + 32 + 32
    expected = base + 32 + 16 + 8  # +queue_lstm_hidden + cycle_lstm_hidden + noise
    _assert_true(
        model.pred_lstm_hidden_size == expected,
        "c2_1 + minimal_viable 组合 pred_lstm_hidden_size 应当为 {}, got {}".format(
            expected, model.pred_lstm_hidden_size
        ),
    )


def test_c2_1_cyclestate_forward_backward():
    """c2_1 cyclestate (无 minimal_viable) forward+backward 验证。"""
    model = _build_cyclestate(c2_1_mode=True, minimal_viable_mode=False)
    model.train()
    inputs = _build_inputs()
    pred_traj_fake_rel = model(*inputs, teacher_forcing_ratio=0.5)
    _assert_true(
        pred_traj_fake_rel.shape == (12, 8, 2),
        "c2_1 cyclestate forward 输出 shape 应该是 (12, 8, 2), got {}".format(
            pred_traj_fake_rel.shape
        ),
    )
    loss = pred_traj_fake_rel.pow(2).sum()
    loss.backward()
    layer2_grad = model.traj_lstm_layer2.weight_hh.grad
    _assert_true(
        layer2_grad is not None and torch.isfinite(layer2_grad).all(),
        "c2_1 cyclestate 模式 traj_lstm_layer2 应该有有效梯度",
    )


def main():
    test_c2_1_disabled_d2tpred_layer2_is_none()
    print("[PASS] test_c2_1_disabled_d2tpred_layer2_is_none")
    test_c2_1_enabled_d2tpred_layer2_exists()
    print("[PASS] test_c2_1_enabled_d2tpred_layer2_exists")
    test_c2_1_d2tpred_forward_backward()
    print("[PASS] test_c2_1_d2tpred_forward_backward")
    test_c2_1_d2tpred_disabled_forward_still_works()
    print("[PASS] test_c2_1_d2tpred_disabled_forward_still_works")
    test_c2_1_cyclestate_orthogonal_to_minimal_viable()
    print("[PASS] test_c2_1_cyclestate_orthogonal_to_minimal_viable")
    test_c2_1_cyclestate_combined_with_minimal_viable()
    print("[PASS] test_c2_1_cyclestate_combined_with_minimal_viable")
    test_c2_1_cyclestate_forward_backward()
    print("[PASS] test_c2_1_cyclestate_forward_backward")
    print("ALL C2-1 SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
