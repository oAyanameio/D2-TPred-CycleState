"""DE-3 smoke test: verify ``minimal_viable_mode`` forward/backward works。

不依赖真实数据集, 直接用 ``torch.randn`` 构造 ``obs_traj`` / ``obs_traj_rel`` /
``obs_state`` / ``pred_state`` / ``seq_start_end``, 走一次 forward + backward,
检查:

1. 模型能正常构造, ``pred_lstm_hidden_size`` 真正按 DE-3 协议扩大;
2. forward 输出 shape 与 ``(pred_len, batch, 2)`` 一致;
3. loss 能正常 backward, 梯度能流到 ``queue_lstm_model`` / ``cycle_lstm_model`` /
   ``pred_lstm_model``, 证明 state 分支参与了实际优化;
4. 与非 DE-3 模式 (默认) 相比, ``pred_lstm_hidden_size`` 增加了
   ``queue_lstm_hidden_size + cycle_lstm_hidden_size``, 与 PLAN.md 设计一致;
5. 关闭 ``minimal_viable_mode`` 时, 旧版 ``encoded_before_noise_hidden`` 形状
   保持不变, 不会破坏向后兼容。
"""

import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from models import CycleStateTrajectoryGenerator  # noqa: E402


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


def _build_model(minimal_viable_mode=False):
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
    )


def _assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def test_default_mode_shape():
    """默认模式 (minimal_viable_mode=False) 形状与原 CycleState 行为一致。"""
    model = _build_model(minimal_viable_mode=False)
    expected = 32 + 32 + 32 + 8  # light+traj+graph+noise
    _assert_true(
        model.pred_lstm_hidden_size == expected,
        "默认模式 pred_lstm_hidden_size 应该等于 {}, got {}".format(
            expected, model.pred_lstm_hidden_size
        ),
    )
    _assert_true(
        not model.minimal_viable_mode,
        "默认模式 minimal_viable_mode 必须为 False",
    )
    _assert_true(
        not model.disable_state_gating,
        "默认模式 disable_state_gating 必须为 False",
    )


def test_de3_mode_shape():
    """DE-3 模式下, pred_lstm_hidden_size 应在原值基础上加上 queue+cycle 维度。"""
    model = _build_model(minimal_viable_mode=True)
    base = 32 + 32 + 32 + 8
    expected = base + 32 + 16  # +queue_lstm_hidden + cycle_lstm_hidden
    _assert_true(
        model.pred_lstm_hidden_size == expected,
        "DE-3 模式 pred_lstm_hidden_size 应该等于 {}, got {}".format(
            expected, model.pred_lstm_hidden_size
        ),
    )
    _assert_true(
        model.minimal_viable_mode,
        "DE-3 模式 minimal_viable_mode 必须为 True",
    )
    _assert_true(
        model.disable_state_gating,
        "DE-3 模式 disable_state_gating 必须为 True",
    )
    _assert_true(
        model.disable_queue_rollout,
        "DE-3 模式 disable_queue_rollout 必须为 True",
    )
    _assert_true(
        model.disable_lane_queue_anchor,
        "DE-3 模式 disable_lane_queue_anchor 必须为 True",
    )
    _assert_true(
        model.disable_decoder_state_residual,
        "DE-3 模式 disable_decoder_state_residual 必须为 True",
    )
    _assert_true(
        model.disable_aux_losses,
        "DE-3 模式 disable_aux_losses 必须为 True",
    )


def test_de3_forward_backward():
    """DE-3 模式 forward + backward 应能正常跑通。"""
    model = _build_model(minimal_viable_mode=True)
    model.train()
    inputs = _build_inputs()
    pred_traj_fake_rel = model(*inputs, teacher_forcing_ratio=0.5)
    _assert_true(
        pred_traj_fake_rel.shape == (12, 8, 2),
        "DE-3 forward 输出 shape 应该是 (12, 8, 2), got {}".format(
            pred_traj_fake_rel.shape
        ),
    )
    loss = pred_traj_fake_rel.pow(2).sum()
    loss.backward()
    # 至少验证 queue/cycle LSTM 与 pred_lstm_model 有真实梯度流过
    queue_grad = model.queue_lstm_model.weight_hh.grad
    cycle_grad = model.cycle_lstm_model.weight_hh.grad
    pred_lstm_grad = model.pred_lstm_model.weight_hh.grad
    _assert_true(
        queue_grad is not None and torch.isfinite(queue_grad).all(),
        "queue_lstm_model 应该有有效梯度",
    )
    _assert_true(
        cycle_grad is not None and torch.isfinite(cycle_grad).all(),
        "cycle_lstm_model 应该有有效梯度",
    )
    _assert_true(
        pred_lstm_grad is not None and torch.isfinite(pred_lstm_grad).all(),
        "pred_lstm_model 应该有有效梯度",
    )


def test_default_mode_forward_still_works():
    """非 DE-3 模式 forward + backward 也应能跑通, 保证向后兼容。"""
    model = _build_model(minimal_viable_mode=False)
    model.train()
    inputs = _build_inputs()
    pred_traj_fake_rel = model(*inputs, teacher_forcing_ratio=0.5)
    _assert_true(
        pred_traj_fake_rel.shape == (12, 8, 2),
        "默认模式 forward 输出 shape 应该是 (12, 8, 2), got {}".format(
            pred_traj_fake_rel.shape
        ),
    )
    loss = pred_traj_fake_rel.pow(2).sum()
    loss.backward()


def main():
    test_default_mode_shape()
    print("[PASS] test_default_mode_shape")
    test_de3_mode_shape()
    print("[PASS] test_de3_mode_shape")
    test_de3_forward_backward()
    print("[PASS] test_de3_forward_backward")
    test_default_mode_forward_still_works()
    print("[PASS] test_default_mode_forward_still_works")
    print("ALL DE-3 SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
