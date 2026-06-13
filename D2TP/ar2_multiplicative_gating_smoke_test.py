"""AR-2 smoke test: verify ``ar2_multiplicative_gating_mode`` forward/backward works。

不依赖真实数据集, 直接用 ``torch.randn`` 构造 ``obs_traj`` / ``obs_traj_rel`` /
``obs_state`` / ``pred_state`` / ``seq_start_end``, 走一次 forward + backward,
检查:

1. AR-2 模式下 ``pred_lstm_model`` / ``pred_hidden2pos`` 的输入维度与 DE-3
   保持一致 (AR-2 不"加性"地扩大输入, 仅通过 sigmoid gate 调制 hidden);
2. AR-2 模式下 ``ar2_hidden_gate`` 模块被正确创建, 形状为
   ``(Linear -> ReLU -> Linear -> Sigmoid)``;
3. forward 输出 shape 与 ``(pred_len, batch, 2)`` 一致;
4. loss 能正常 backward, 梯度能流到 ``pred_lstm_model`` / ``ar2_hidden_gate`` /
   ``pred_hidden2pos``, 证明 state context 通过乘法被真实消费;
5. AR-2 模式下 5 个 disable 开关 + ``minimal_viable_mode`` 都被强制开启;
6. AR-2 与 ``oracle_inject_mode`` / ``ar1_direct_inject_mode`` 互斥校验:
   同时开启会抛 ``ValueError``;
7. 关闭 ``ar2_multiplicative_gating_mode`` 时, 旧版模型行为保持不变,
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


def _build_model(
    ar2_multiplicative_gating_mode=False,
    ar1_direct_inject_mode=False,
    oracle_inject_mode=False,
):
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
        ar2_multiplicative_gating_mode=ar2_multiplicative_gating_mode,
        ar1_direct_inject_mode=ar1_direct_inject_mode,
        oracle_inject_mode=oracle_inject_mode,
    )


def _assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def test_default_mode_shape():
    """默认模式 (ar2_multiplicative_gating_mode=False) 形状不变。"""
    model = _build_model()
    # 默认 ``traj_lstm_input_size=2`` 时, ``pred_lstm_model`` 应当消费 2 维 offset。
    _assert_true(
        model.pred_lstm_model.input_size == 2,
        "默认模式 pred_lstm_model.input_size 应该等于 2, got {}".format(
            model.pred_lstm_model.input_size
        ),
    )
    _assert_true(
        not model.ar2_multiplicative_gating_mode,
        "默认模式 ar2_multiplicative_gating_mode 必须为 False",
    )
    _assert_true(
        not hasattr(model, "ar2_hidden_gate"),
        "默认模式不应有 ar2_hidden_gate 模块",
    )


def test_ar2_mode_shape():
    """AR-2 模式下, ``pred_lstm_model`` 形状不变 (与 DE-3 一致), ``ar2_hidden_gate`` 存在。"""
    model = _build_model(ar2_multiplicative_gating_mode=True)
    # AR-2 state_context_dim = 32 + 16 = 48
    expected_state_ctx_dim = 48
    _assert_true(
        model.ar2_state_context_dim == expected_state_ctx_dim,
        "AR-2 ar2_state_context_dim 应为 {}, got {}".format(
            expected_state_ctx_dim, model.ar2_state_context_dim
        ),
    )
    # AR-2 不修改 pred_lstm_model / pred_hidden2pos 的输入维度 (与 DE-3 一致)
    _assert_true(
        model.pred_lstm_model.input_size == 2,
        "AR-2 pred_lstm_model.input_size 应该等于 2 (与 DE-3 一致), got {}".format(
            model.pred_lstm_model.input_size
        ),
    )
    expected_pred_hidden2pos_in = (
        model.light_embedding_size + model.pred_lstm_hidden_size
    )
    _assert_true(
        model.pred_hidden2pos.in_features == expected_pred_hidden2pos_in,
        "AR-2 pred_hidden2pos.in_features 应为 {} (与 DE-3 一致), got {}".format(
            expected_pred_hidden2pos_in, model.pred_hidden2pos.in_features
        ),
    )
    # ar2_hidden_gate 应当存在
    _assert_true(
        hasattr(model, "ar2_hidden_gate"),
        "AR-2 模式必须创建 ar2_hidden_gate 模块",
    )
    # ar2_hidden_gate 应该是 Sequential (Linear -> ReLU -> Linear -> Sigmoid)
    _assert_true(
        isinstance(model.ar2_hidden_gate, torch.nn.Sequential),
        "AR-2 ar2_hidden_gate 必须是 nn.Sequential, got {}".format(
            type(model.ar2_hidden_gate)
        ),
    )
    _assert_true(
        len(model.ar2_hidden_gate) == 4,
        "AR-2 ar2_hidden_gate 应当有 4 层 (Linear, ReLU, Linear, Sigmoid), got {}".format(
            len(model.ar2_hidden_gate)
        ),
    )
    # AR-2 隐含 minimal_viable_mode=True
    _assert_true(
        model.minimal_viable_mode,
        "AR-2 模式 minimal_viable_mode 必须为 True",
    )
    # 5 个 disable 开关都应被强制开启
    _assert_true(model.disable_state_gating, "AR-2 模式 disable_state_gating 必须为 True")
    _assert_true(model.disable_queue_rollout, "AR-2 模式 disable_queue_rollout 必须为 True")
    _assert_true(model.disable_lane_queue_anchor, "AR-2 模式 disable_lane_queue_anchor 必须为 True")
    _assert_true(
        model.disable_decoder_state_residual,
        "AR-2 模式 disable_decoder_state_residual 必须为 True",
    )
    _assert_true(model.disable_aux_losses, "AR-2 模式 disable_aux_losses 必须为 True")


def test_ar2_forward_backward_train():
    """AR-2 模式 train 阶段 forward + backward 应能正常跑通。"""
    model = _build_model(ar2_multiplicative_gating_mode=True)
    model.train()
    inputs = _build_inputs()
    pred_traj_fake_rel = model(*inputs, teacher_forcing_ratio=0.5)
    _assert_true(
        pred_traj_fake_rel.shape == (12, 8, 2),
        "AR-2 train forward 输出 shape 应该是 (12, 8, 2), got {}".format(
            pred_traj_fake_rel.shape
        ),
    )
    loss = pred_traj_fake_rel.pow(2).sum()
    loss.backward()
    pred_lstm_grad = model.pred_lstm_model.weight_hh.grad
    _assert_true(
        pred_lstm_grad is not None and torch.isfinite(pred_lstm_grad).all(),
        "AR-2 pred_lstm_model 应该有有效梯度",
    )
    ar2_gate_grad = model.ar2_hidden_gate[0].weight.grad
    _assert_true(
        ar2_gate_grad is not None and torch.isfinite(ar2_gate_grad).all(),
        "AR-2 ar2_hidden_gate 应该有有效梯度 (state context 通过乘法被消费)",
    )
    pred_hidden2pos_grad = model.pred_hidden2pos.weight.grad
    _assert_true(
        pred_hidden2pos_grad is not None and torch.isfinite(pred_hidden2pos_grad).all(),
        "AR-2 pred_hidden2pos 应该有有效梯度",
    )


def test_ar2_forward_eval():
    """AR-2 模式 eval 阶段 forward 也应能跑通, 形状正确。"""
    model = _build_model(ar2_multiplicative_gating_mode=True)
    model.eval()
    inputs = _build_inputs()
    with torch.no_grad():
        pred_traj_fake_rel = model(*inputs, teacher_forcing_ratio=0.0)
    _assert_true(
        pred_traj_fake_rel.shape == (12, 8, 2),
        "AR-2 eval forward 输出 shape 应该是 (12, 8, 2), got {}".format(
            pred_traj_fake_rel.shape
        ),
    )


def test_ar2_oracle_mutual_exclusion():
    """AR-2 与 ``oracle_inject_mode`` 互斥 — 同时开启会抛 ``ValueError``。"""
    raised = False
    try:
        _build_model(ar2_multiplicative_gating_mode=True, oracle_inject_mode=True)
    except ValueError:
        raised = True
    _assert_true(raised, "AR-2 + oracle_inject_mode 同时开启应抛 ValueError")


def test_ar2_ar1_mutual_exclusion():
    """AR-2 与 AR-1 互斥 — 同时开启会抛 ``ValueError``。"""
    raised = False
    try:
        _build_model(ar2_multiplicative_gating_mode=True, ar1_direct_inject_mode=True)
    except ValueError:
        raised = True
    _assert_true(raised, "AR-2 + ar1_direct_inject_mode 同时开启应抛 ValueError")


def test_default_mode_forward_still_works():
    """非 AR-2 模式 forward + backward 也应能跑通, 保证向后兼容。"""
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


def test_ar2_gate_modulates_hidden():
    """AR-2 模式 gate 输出应该限制在 (0, 1) 区间 (sigmoid 性质)。"""
    model = _build_model(ar2_multiplicative_gating_mode=True)
    model.eval()
    # 构造一个简单的输入测试 gate 的输出范围
    batch_size = 4
    pred_hidden = torch.randn(batch_size, model.pred_lstm_hidden_size)
    state_ctx = torch.randn(batch_size, model.ar2_state_context_dim)
    gate_input = torch.cat((pred_hidden, state_ctx), dim=1)
    with torch.no_grad():
        gate = model.ar2_hidden_gate(gate_input)
    _assert_true(
        torch.all(gate >= 0.0) and torch.all(gate <= 1.0),
        "AR-2 gate 输出必须限制在 (0, 1) 区间 (sigmoid), got min={}, max={}".format(
            gate.min().item(), gate.max().item()
        ),
    )
    _assert_true(
        gate.shape == (batch_size, model.pred_lstm_hidden_size),
        "AR-2 gate 输出 shape 应为 (batch, pred_lstm_hidden_size), got {}".format(
            gate.shape
        ),
    )


def main():
    test_default_mode_shape()
    print("[PASS] test_default_mode_shape")
    test_ar2_mode_shape()
    print("[PASS] test_ar2_mode_shape")
    test_ar2_forward_backward_train()
    print("[PASS] test_ar2_forward_backward_train")
    test_ar2_forward_eval()
    print("[PASS] test_ar2_forward_eval")
    test_ar2_oracle_mutual_exclusion()
    print("[PASS] test_ar2_oracle_mutual_exclusion")
    test_ar2_ar1_mutual_exclusion()
    print("[PASS] test_ar2_ar1_mutual_exclusion")
    test_default_mode_forward_still_works()
    print("[PASS] test_default_mode_forward_still_works")
    test_ar2_gate_modulates_hidden()
    print("[PASS] test_ar2_gate_modulates_hidden")
    print("ALL AR-2 SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
