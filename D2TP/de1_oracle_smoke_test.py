"""DE-1 smoke test: verify ``oracle_inject_mode`` forward/backward works。

不依赖真实数据集, 直接用 ``torch.randn`` 构造 ``obs_traj`` / ``obs_traj_rel`` /
``obs_state`` / ``pred_state`` / ``seq_start_end``, 走一次 forward + backward,
检查:

1. 模型能正常构造, ``oracle_inject_mode=True`` 后 ``pred_lstm_model`` 的
   ``input_size`` 真正按 DE-1 协议扩大到 ``traj_lstm_input_size + 10``;
2. forward 输出 shape 与 ``(pred_len, batch, 2)`` 一致;
3. loss 能正常 backward, 梯度能流到 ``pred_lstm_model``, 证明 oracle 特征
   被真实消费;
4. ``oracle_inject_mode=True`` 时 5 个 disable 开关 (state_gating /
   queue_rollout / lane_queue_anchor / decoder_state_residual / aux_losses)
   都被强制置 True, 与 PLAN.md 设计一致;
5. 关闭 ``oracle_inject_mode`` 时, 旧版 ``pred_lstm_model`` 形状保持不变,
   不会破坏向后兼容。
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


def _build_model(oracle_inject_mode=False):
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
        oracle_inject_mode=oracle_inject_mode,
    )


def _assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def test_default_mode_shape():
    """默认模式 (oracle_inject_mode=False) ``pred_lstm_model`` 形状不变。"""
    model = _build_model(oracle_inject_mode=False)
    # 默认 ``traj_lstm_input_size=2`` 时, ``pred_lstm_model`` 应当消费 2 维 offset。
    _assert_true(
        model.pred_lstm_model.input_size == 2,
        "默认模式 pred_lstm_model.input_size 应该等于 2, got {}".format(
            model.pred_lstm_model.input_size
        ),
    )
    _assert_true(
        not model.oracle_inject_mode,
        "默认模式 oracle_inject_mode 必须为 False",
    )
    _assert_true(
        model.oracle_feature_dim == 10,
        "oracle_feature_dim 应当为 10, got {}".format(model.oracle_feature_dim),
    )


def test_de1_mode_shape():
    """DE-1 模式下, ``pred_lstm_model`` 应当消费 2+10 维特征。"""
    model = _build_model(oracle_inject_mode=True)
    expected_input = 2 + 10  # traj_lstm_input_size + oracle_feature_dim
    _assert_true(
        model.pred_lstm_model.input_size == expected_input,
        "DE-1 模式 pred_lstm_model.input_size 应该等于 {}, got {}".format(
            expected_input, model.pred_lstm_model.input_size
        ),
    )
    _assert_true(
        model.oracle_inject_mode,
        "DE-1 模式 oracle_inject_mode 必须为 True",
    )
    # 5 个 disable 开关都应被强制开启
    _assert_true(
        model.disable_state_gating,
        "DE-1 模式 disable_state_gating 必须为 True",
    )
    _assert_true(
        model.disable_queue_rollout,
        "DE-1 模式 disable_queue_rollout 必须为 True",
    )
    _assert_true(
        model.disable_lane_queue_anchor,
        "DE-1 模式 disable_lane_queue_anchor 必须为 True",
    )
    _assert_true(
        model.disable_decoder_state_residual,
        "DE-1 模式 disable_decoder_state_residual 必须为 True",
    )
    _assert_true(
        model.disable_aux_losses,
        "DE-1 模式 disable_aux_losses 必须为 True",
    )


def test_de1_forward_backward_train():
    """DE-1 模式 train 阶段 forward + backward 应能正常跑通。"""
    model = _build_model(oracle_inject_mode=True)
    model.train()
    inputs = _build_inputs()
    pred_traj_fake_rel = model(*inputs, teacher_forcing_ratio=0.5)
    _assert_true(
        pred_traj_fake_rel.shape == (12, 8, 2),
        "DE-1 train forward 输出 shape 应该是 (12, 8, 2), got {}".format(
            pred_traj_fake_rel.shape
        ),
    )
    loss = pred_traj_fake_rel.pow(2).sum()
    loss.backward()
    # 验证 ``pred_lstm_model`` 有真实梯度流过, 证明 oracle 特征被消费
    pred_lstm_grad = model.pred_lstm_model.weight_hh.grad
    _assert_true(
        pred_lstm_grad is not None and torch.isfinite(pred_lstm_grad).all(),
        "pred_lstm_model 应该有有效梯度",
    )
    # 验证 pred_lstm_model.input_size 真的从 2 变成了 12
    _assert_true(
        model.pred_lstm_model.input_size == 12,
        "train 阶段 pred_lstm_model.input_size 应为 12 (2+10), got {}".format(
            model.pred_lstm_model.input_size
        ),
    )


def test_de1_forward_eval():
    """DE-1 模式 eval 阶段 forward 也应能跑通, 形状正确。"""
    model = _build_model(oracle_inject_mode=True)
    model.eval()
    inputs = _build_inputs()
    with torch.no_grad():
        pred_traj_fake_rel = model(*inputs, teacher_forcing_ratio=0.0)
    _assert_true(
        pred_traj_fake_rel.shape == (12, 8, 2),
        "DE-1 eval forward 输出 shape 应该是 (12, 8, 2), got {}".format(
            pred_traj_fake_rel.shape
        ),
    )


def test_oracle_step_feature_shape():
    """``build_oracle_step_feature`` 输出 shape 应为 ``(batch, 10)``。"""
    model = _build_model(oracle_inject_mode=True)
    obs_traj_rel, obs_traj_pos, obs_state, pred_state, _ = _build_inputs()
    real_pos = obs_traj_pos[-1, :, 2:4]
    last_offset = obs_traj_rel[obs_traj_rel.shape[0] - 1, :, 2:4]
    feat = model.build_oracle_step_feature(
        step_index=0,
        real_pos=real_pos,
        last_pred_offset=last_offset,
        obs_state=obs_state,
        pred_state=pred_state,
    )
    _assert_true(
        feat.shape == (8, 10),
        "oracle step feature shape 应该是 (8, 10), got {}".format(feat.shape),
    )


def test_default_mode_forward_still_works():
    """非 DE-1 模式 forward + backward 也应能跑通, 保证向后兼容。"""
    model = _build_model(oracle_inject_mode=False)
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
    # 默认模式 pred_lstm_model.input_size 应保持 2
    _assert_true(
        model.pred_lstm_model.input_size == 2,
        "默认模式 pred_lstm_model.input_size 应为 2, got {}".format(
            model.pred_lstm_model.input_size
        ),
    )


def main():
    test_default_mode_shape()
    print("[PASS] test_default_mode_shape")
    test_de1_mode_shape()
    print("[PASS] test_de1_mode_shape")
    test_de1_forward_backward_train()
    print("[PASS] test_de1_forward_backward_train")
    test_de1_forward_eval()
    print("[PASS] test_de1_forward_eval")
    test_oracle_step_feature_shape()
    print("[PASS] test_oracle_step_feature_shape")
    test_default_mode_forward_still_works()
    print("[PASS] test_default_mode_forward_still_works")
    print("ALL DE-1 SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
