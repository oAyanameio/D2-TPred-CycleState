"""AR-1 smoke test: verify ``ar1_direct_inject_mode`` forward/backward works。

不依赖真实数据集, 直接用 ``torch.randn`` 构造 ``obs_traj`` / ``obs_traj_rel`` /
``obs_state`` / ``pred_state`` / ``seq_start_end``, 走一次 forward + backward,
检查:

1. AR-1 模式下 ``pred_lstm_model`` 的 ``input_size`` 真正按 AR-1 协议扩大到
   ``traj_lstm_input_size + ar1_state_context_dim (= 48)``;
2. AR-1 模式下 ``pred_hidden2pos`` 的 ``in_features`` 真正按 AR-1 协议扩大到
   ``light_embedding_size + pred_lstm_hidden_size + ar1_state_context_dim``;
3. forward 输出 shape 与 ``(pred_len, batch, 2)`` 一致;
4. loss 能正常 backward, 梯度能流到 ``pred_lstm_model`` / ``pred_hidden2pos``,
   证明 state context 被真实消费;
5. AR-1 模式下 5 个 disable 开关 + ``minimal_viable_mode`` 都被强制开启;
6. AR-1 与 ``oracle_inject_mode`` 互斥校验: 同时开启会抛 ``ValueError``;
7. 关闭 ``ar1_direct_inject_mode`` 时, 旧版 ``pred_lstm_model`` 形状保持不变,
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


def _build_model(ar1_direct_inject_mode=False, oracle_inject_mode=False):
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
        ar1_direct_inject_mode=ar1_direct_inject_mode,
        oracle_inject_mode=oracle_inject_mode,
    )


def _assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def test_default_mode_shape():
    """默认模式 (ar1_direct_inject_mode=False) 形状不变。"""
    model = _build_model()
    # 默认 ``traj_lstm_input_size=2`` 时, ``pred_lstm_model`` 应当消费 2 维 offset。
    _assert_true(
        model.pred_lstm_model.input_size == 2,
        "默认模式 pred_lstm_model.input_size 应该等于 2, got {}".format(
            model.pred_lstm_model.input_size
        ),
    )
    _assert_true(
        not model.ar1_direct_inject_mode,
        "默认模式 ar1_direct_inject_mode 必须为 False",
    )
    _assert_true(
        not model.minimal_viable_mode,
        "默认模式 minimal_viable_mode 必须为 False (AR-1 未启用)",
    )


def test_ar1_mode_shape():
    """AR-1 模式下, ``pred_lstm_model`` / ``pred_hidden2pos`` 形状应扩大。"""
    model = _build_model(ar1_direct_inject_mode=True)
    # AR-1 state_context_dim = 32 + 16 = 48
    expected_state_ctx_dim = 48
    _assert_true(
        model.ar1_state_context_dim == expected_state_ctx_dim,
        "AR-1 ar1_state_context_dim 应为 {}, got {}".format(
            expected_state_ctx_dim, model.ar1_state_context_dim
        ),
    )
    expected_pred_lstm_input = 2 + expected_state_ctx_dim  # 50
    _assert_true(
        model.pred_lstm_model.input_size == expected_pred_lstm_input,
        "AR-1 pred_lstm_model.input_size 应该等于 {}, got {}".format(
            expected_pred_lstm_input, model.pred_lstm_model.input_size
        ),
    )
    # AR-1 隐含 minimal_viable_mode=True
    _assert_true(
        model.minimal_viable_mode,
        "AR-1 模式 minimal_viable_mode 必须为 True",
    )
    # 5 个 disable 开关都应被强制开启
    _assert_true(model.disable_state_gating, "AR-1 模式 disable_state_gating 必须为 True")
    _assert_true(model.disable_queue_rollout, "AR-1 模式 disable_queue_rollout 必须为 True")
    _assert_true(model.disable_lane_queue_anchor, "AR-1 模式 disable_lane_queue_anchor 必须为 True")
    _assert_true(
        model.disable_decoder_state_residual,
        "AR-1 模式 disable_decoder_state_residual 必须为 True",
    )
    _assert_true(model.disable_aux_losses, "AR-1 模式 disable_aux_losses 必须为 True")
    # pred_hidden2pos 的 in_features = light_embedding_size + pred_lstm_hidden_size + state_context_dim
    expected_pred_hidden2pos_in = (
        model.light_embedding_size + model.pred_lstm_hidden_size + expected_state_ctx_dim
    )
    _assert_true(
        model.pred_hidden2pos.in_features == expected_pred_hidden2pos_in,
        "AR-1 pred_hidden2pos.in_features 应为 {}, got {}".format(
            expected_pred_hidden2pos_in, model.pred_hidden2pos.in_features
        ),
    )


def test_ar1_forward_backward_train():
    """AR-1 模式 train 阶段 forward + backward 应能正常跑通。"""
    model = _build_model(ar1_direct_inject_mode=True)
    model.train()
    inputs = _build_inputs()
    pred_traj_fake_rel = model(*inputs, teacher_forcing_ratio=0.5)
    _assert_true(
        pred_traj_fake_rel.shape == (12, 8, 2),
        "AR-1 train forward 输出 shape 应该是 (12, 8, 2), got {}".format(
            pred_traj_fake_rel.shape
        ),
    )
    loss = pred_traj_fake_rel.pow(2).sum()
    loss.backward()
    pred_lstm_grad = model.pred_lstm_model.weight_hh.grad
    _assert_true(
        pred_lstm_grad is not None and torch.isfinite(pred_lstm_grad).all(),
        "AR-1 pred_lstm_model 应该有有效梯度",
    )
    pred_hidden2pos_grad = model.pred_hidden2pos.weight.grad
    _assert_true(
        pred_hidden2pos_grad is not None and torch.isfinite(pred_hidden2pos_grad).all(),
        "AR-1 pred_hidden2pos 应该有有效梯度 (state context 在输出投影被消费)",
    )


def test_ar1_forward_eval():
    """AR-1 模式 eval 阶段 forward 也应能跑通, 形状正确。"""
    model = _build_model(ar1_direct_inject_mode=True)
    model.eval()
    inputs = _build_inputs()
    with torch.no_grad():
        pred_traj_fake_rel = model(*inputs, teacher_forcing_ratio=0.0)
    _assert_true(
        pred_traj_fake_rel.shape == (12, 8, 2),
        "AR-1 eval forward 输出 shape 应该是 (12, 8, 2), got {}".format(
            pred_traj_fake_rel.shape
        ),
    )


def test_ar1_oracle_mutual_exclusion():
    """AR-1 与 ``oracle_inject_mode`` 互斥 — 同时开启会抛 ``ValueError``。"""
    raised = False
    try:
        _build_model(ar1_direct_inject_mode=True, oracle_inject_mode=True)
    except ValueError:
        raised = True
    _assert_true(raised, "AR-1 + oracle_inject_mode 同时开启应抛 ValueError")


def test_default_mode_forward_still_works():
    """非 AR-1 模式 forward + backward 也应能跑通, 保证向后兼容。"""
    model = _build_model()
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
    _assert_true(
        model.pred_lstm_model.input_size == 2,
        "默认模式 pred_lstm_model.input_size 应为 2, got {}".format(
            model.pred_lstm_model.input_size
        ),
    )


def main():
    test_default_mode_shape()
    print("[PASS] test_default_mode_shape")
    test_ar1_mode_shape()
    print("[PASS] test_ar1_mode_shape")
    test_ar1_forward_backward_train()
    print("[PASS] test_ar1_forward_backward_train")
    test_ar1_forward_eval()
    print("[PASS] test_ar1_forward_eval")
    test_ar1_oracle_mutual_exclusion()
    print("[PASS] test_ar1_oracle_mutual_exclusion")
    test_default_mode_forward_still_works()
    print("[PASS] test_default_mode_forward_still_works")
    print("ALL AR-1 SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
