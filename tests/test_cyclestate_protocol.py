import importlib.util
import pathlib
import sys
import types
import unittest

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_ROOT = REPO_ROOT / "D2TP"
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))


def load_module(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(
        module_name, REPO_ROOT / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


models = load_module("d2tp_models", "D2TP/models.py")
train = load_module("d2tp_train", "D2TP/train.py")


class CycleStateProtocolTest(unittest.TestCase):
    def setUp(self):
        self.model = models.CycleStateTrajectoryGenerator(
            obs_len=8,
            pred_len=12,
            traj_lstm_input_size=2,
            traj_lstm_hidden_size=32,
            n_units=[32, 16, 32],
            n_heads=[4, 1],
            graph_network_out_dims=32,
            dropout=0.0,
            alpha=0.2,
            graph_lstm_hidden_size=32,
            noise_dim=(16,),
            noise_type="gaussian",
        )

        batch = 3
        self.obs_traj = torch.zeros(8, batch, 10)
        self.obs_traj_rel = torch.zeros(20, batch, 9)
        self.pred_traj_gt = torch.zeros(12, batch, 10)
        self.pred_traj_gt_rel = torch.zeros(12, batch, 9)
        self.obs_state = torch.zeros(8, batch, 4)
        self.pred_state = torch.zeros(12, batch, 4)
        self.seq_start_end = torch.tensor([[0, batch]], dtype=torch.long)

        lane_ids = torch.tensor([0, 0, 1], dtype=torch.float32)
        stopline_x = torch.tensor([0.0, 0.0, 10.0])
        stopline_y = torch.tensor([0.0, 0.0, 0.0])
        phase_ids = torch.tensor([0.0, 0.0, 1.0])
        elapsed = torch.arange(1, 9, dtype=torch.float32)

        for t in range(8):
            self.obs_traj[t, :, 2] = torch.tensor([10.0 - t, 20.0 - t, 50.0 - t])
            self.obs_traj[t, :, 3] = torch.tensor([0.0, 0.0, 0.0])
            self.obs_traj[t, :, 4] = lane_ids
            self.obs_traj[t, :, 7] = phase_ids
            self.obs_traj[t, :, 8] = elapsed[t]
            self.obs_traj[t, :, 9] = torch.tensor([0.0, 0.0, 0.0])
            self.obs_state[t, :, 0] = stopline_x
            self.obs_state[t, :, 1] = stopline_y
            self.obs_state[t, :, 2] = phase_ids
            self.obs_state[t, :, 3] = elapsed[t]

        self.obs_traj_rel[:8, :, 2] = 1.0
        self.obs_traj_rel[:8, :, 3] = 0.0
        self.pred_traj_gt_rel[:, :, 2] = 1.0
        self.pred_traj_gt_rel[:, :, 3] = 0.0
        for t in range(12):
            self.pred_state[t, :, 0] = stopline_x
            self.pred_state[t, :, 1] = stopline_y
            self.pred_state[t, :, 2] = phase_ids
            self.pred_state[t, :, 3] = float(t + 1)

    def test_build_traffic_context_has_expected_keys(self):
        traffic_context = self.model.build_traffic_context(
            self.obs_traj_rel,
            self.obs_traj,
            self.obs_state,
            self.pred_state,
            self.seq_start_end,
        )
        self.assertEqual(
            {"agent", "signal", "scene", "meso"}, set(traffic_context.keys())
        )
        self.assertEqual(
            (8, 3, self.model.queue_feature_dim),
            tuple(traffic_context["meso"]["queue_feature_seq"].shape),
        )
        self.assertEqual(
            (8, 3, 6), tuple(traffic_context["meso"]["queue_targets"].shape)
        )
        self.assertEqual(
            (8, 3, 6),
            tuple(traffic_context["signal"]["cycle_feature_seq"].shape),
        )
        self.assertEqual(
            (8, 3, self.model.queue_feature_dim),
            tuple(traffic_context["meso"]["lane_queue_anchor_seq"].shape),
        )
        lane_anchor_t0 = traffic_context["meso"]["lane_queue_anchor_seq"][0]
        self.assertTrue(torch.allclose(lane_anchor_t0[0], lane_anchor_t0[1]))
        self.assertFalse(torch.allclose(lane_anchor_t0[0], lane_anchor_t0[2]))

    def test_cycle_forward_accepts_external_traffic_context(self):
        traffic_context = self.model.build_traffic_context(
            self.obs_traj_rel,
            self.obs_traj,
            self.obs_state,
            self.pred_state,
            self.seq_start_end,
        )
        outputs = self.model(
            self.obs_traj_rel,
            self.obs_traj,
            self.obs_state,
            self.pred_state,
            self.seq_start_end,
            traffic_context=traffic_context,
        )
        self.assertEqual((12, 3, 2), tuple(outputs.shape))
        self.assertIn("queue_rollout_hidden_seq", self.model.debug_last_aux)
        self.assertIn("queue_rollout_pred_seq", self.model.debug_last_aux)
        self.assertIn("lane_queue_rollout_anchor_seq", self.model.debug_last_aux)
        self.assertEqual(
            (12, 3, self.model.queue_lstm_hidden_size),
            tuple(self.model.debug_last_aux["queue_rollout_hidden_seq"].shape),
        )
        self.assertEqual(
            (12, 3, 6),
            tuple(self.model.debug_last_aux["queue_rollout_pred_seq"].shape),
        )
        self.assertEqual(
            (12, 3, self.model.queue_feature_dim),
            tuple(self.model.debug_last_aux["lane_queue_rollout_anchor_seq"].shape),
        )

    def test_queue_rollout_changes_over_prediction_steps(self):
        traffic_context = self.model.build_traffic_context(
            self.obs_traj_rel,
            self.obs_traj,
            self.obs_state,
            self.pred_state,
            self.seq_start_end,
        )
        self.model(
            self.obs_traj_rel,
            self.obs_traj,
            self.obs_state,
            self.pred_state,
            self.seq_start_end,
            traffic_context=traffic_context,
        )
        queue_rollout = self.model.debug_last_aux["queue_rollout_hidden_seq"]
        self.assertFalse(torch.allclose(queue_rollout[0], queue_rollout[-1]))
        lane_anchor_rollout = self.model.debug_last_aux["lane_queue_rollout_anchor_seq"]
        self.assertTrue(torch.allclose(lane_anchor_rollout[0, 0], lane_anchor_rollout[0, 1]))
        self.assertFalse(torch.allclose(lane_anchor_rollout[0, 0], lane_anchor_rollout[0, 2]))

    def test_cycle_forward_supports_disabled_queue_rollout(self):
        no_rollout_model = models.CycleStateTrajectoryGenerator(
            obs_len=8,
            pred_len=12,
            traj_lstm_input_size=2,
            traj_lstm_hidden_size=32,
            n_units=[32, 16, 32],
            n_heads=[4, 1],
            graph_network_out_dims=32,
            dropout=0.0,
            alpha=0.2,
            graph_lstm_hidden_size=32,
            noise_dim=(16,),
            noise_type="gaussian",
            disable_queue_rollout=True,
        )
        outputs = no_rollout_model(
            self.obs_traj_rel,
            self.obs_traj,
            self.obs_state,
            self.pred_state,
            self.seq_start_end,
        )
        self.assertEqual((12, 3, 2), tuple(outputs.shape))
        self.assertIsNone(no_rollout_model.debug_last_aux["queue_rollout_hidden_seq"])
        self.assertIsNone(no_rollout_model.debug_last_aux["queue_rollout_pred_seq"])
        self.assertIsNone(no_rollout_model.debug_last_aux["queue_rollout_target_seq"])

    def test_cycle_forward_supports_disabled_lane_queue_anchor(self):
        no_anchor_model = models.CycleStateTrajectoryGenerator(
            obs_len=8,
            pred_len=12,
            traj_lstm_input_size=2,
            traj_lstm_hidden_size=32,
            n_units=[32, 16, 32],
            n_heads=[4, 1],
            graph_network_out_dims=32,
            dropout=0.0,
            alpha=0.2,
            graph_lstm_hidden_size=32,
            noise_dim=(16,),
            noise_type="gaussian",
            disable_lane_queue_anchor=True,
        )
        outputs = no_anchor_model(
            self.obs_traj_rel,
            self.obs_traj,
            self.obs_state,
            self.pred_state,
            self.seq_start_end,
        )
        self.assertEqual((12, 3, 2), tuple(outputs.shape))
        self.assertIsNone(no_anchor_model.debug_last_aux["lane_queue_rollout_anchor_seq"])

    def test_cycle_forward_supports_disabled_state_gating(self):
        no_gate_model = models.CycleStateTrajectoryGenerator(
            obs_len=8,
            pred_len=12,
            traj_lstm_input_size=2,
            traj_lstm_hidden_size=32,
            n_units=[32, 16, 32],
            n_heads=[4, 1],
            graph_network_out_dims=32,
            dropout=0.0,
            alpha=0.2,
            graph_lstm_hidden_size=32,
            noise_dim=(16,),
            noise_type="gaussian",
            disable_state_gating=True,
        )
        outputs = no_gate_model(
            self.obs_traj_rel,
            self.obs_traj,
            self.obs_state,
            self.pred_state,
            self.seq_start_end,
        )
        self.assertEqual((12, 3, 2), tuple(outputs.shape))

    def test_cycle_forward_supports_disabled_decoder_state_residual(self):
        no_residual_model = models.CycleStateTrajectoryGenerator(
            obs_len=8,
            pred_len=12,
            traj_lstm_input_size=2,
            traj_lstm_hidden_size=32,
            n_units=[32, 16, 32],
            n_heads=[4, 1],
            graph_network_out_dims=32,
            dropout=0.0,
            alpha=0.2,
            graph_lstm_hidden_size=32,
            noise_dim=(16,),
            noise_type="gaussian",
            disable_decoder_state_residual=True,
        )
        outputs = no_residual_model(
            self.obs_traj_rel,
            self.obs_traj,
            self.obs_state,
            self.pred_state,
            self.seq_start_end,
        )
        self.assertEqual((12, 3, 2), tuple(outputs.shape))
        self.assertIsNone(
            no_residual_model.debug_last_aux["decoder_state_init_residual"]
        )
        self.assertIsNone(
            no_residual_model.debug_last_aux["decoder_state_step_residual_seq"]
        )

    def test_structured_auxiliary_losses_split_by_target_type(self):
        queue_pred = torch.tensor(
            [[0.9, 0.2, 0.7, 0.3, 1.2, -0.5]], dtype=torch.float32
        )
        queue_target = torch.tensor(
            [[1.0, 0.0, 0.5, 0.25, 1.0, 0.0]], dtype=torch.float32
        )
        cycle_pred = torch.tensor(
            [[0.1, 1.8, -0.4, 0.4, 0.6, -0.2]], dtype=torch.float32
        )
        cycle_target = torch.tensor(
            [[0.0, 1.0, 0.0, 0.25, 0.75, 1.0]], dtype=torch.float32
        )
        queue_rollout_pred = torch.tensor(
            [
                [[0.8, 0.1, 0.5, 0.2, 1.0, -0.2]],
                [[0.7, 0.1, 0.6, 0.3, 0.9, 0.3]],
            ],
            dtype=torch.float32,
        )
        queue_rollout_target = torch.tensor(
            [
                [[0.7, 0.0, 0.4, 0.3, 1.0, 0.0]],
                [[0.6, 0.0, 0.5, 0.2, 1.0, 1.0]],
            ],
            dtype=torch.float32,
        )

        losses = train.compute_structured_aux_losses(
            queue_pred,
            queue_target,
            cycle_pred,
            cycle_target,
            queue_rollout_pred_seq=queue_rollout_pred,
            queue_rollout_target_seq=queue_rollout_target,
        )

        expected_keys = {
            "queue_reg_loss",
            "queue_cls_loss",
            "queue_rollout_reg_loss",
            "queue_rollout_cls_loss",
            "cycle_phase_loss",
            "cycle_time_loss",
            "cycle_change_loss",
            "queue_total_loss",
            "cycle_total_loss",
        }
        self.assertEqual(expected_keys, set(losses.keys()))
        self.assertGreater(losses["queue_reg_loss"].item(), 0.0)
        self.assertGreater(losses["queue_cls_loss"].item(), 0.0)
        self.assertGreater(losses["queue_rollout_reg_loss"].item(), 0.0)
        self.assertGreater(losses["queue_rollout_cls_loss"].item(), 0.0)
        self.assertGreater(losses["cycle_phase_loss"].item(), 0.0)
        self.assertGreater(losses["cycle_time_loss"].item(), 0.0)
        self.assertGreater(losses["cycle_change_loss"].item(), 0.0)

    def test_structured_auxiliary_losses_can_create_zero_losses_on_target_device(self):
        losses = train.compute_structured_aux_losses(
            None,
            None,
            None,
            None,
            device=torch.device("cpu"),
        )
        for value in losses.values():
            self.assertEqual("cpu", value.device.type)
            self.assertEqual(0.0, value.item())

    def test_train_stage_defaults_follow_protocol(self):
        args = types.SimpleNamespace(
            train_stage="warmup",
            model_type="cyclestate",
            generator_only=None,
            gan_weight=None,
            aux_queue_weight=None,
            aux_cycle_weight=None,
        )
        train.apply_stage_defaults(args)
        self.assertTrue(args.generator_only)
        self.assertEqual(0.0, args.gan_weight)
        self.assertGreater(args.aux_queue_weight, 0.0)
        self.assertGreater(args.aux_cycle_weight, 0.0)

    def test_explicit_stage_overrides_are_preserved_for_ablation(self):
        args = types.SimpleNamespace(
            train_stage="warmup",
            model_type="cyclestate",
            generator_only=True,
            gan_weight=0.0,
            aux_queue_weight=0.0,
            aux_cycle_weight=0.0,
        )
        train.apply_stage_defaults(args)
        self.assertTrue(args.generator_only)
        self.assertEqual(0.0, args.gan_weight)
        self.assertEqual(0.0, args.aux_queue_weight)
        self.assertEqual(0.0, args.aux_cycle_weight)

    def test_cyclestate_keeps_baseline_decoder_shapes_for_warm_start(self):
        baseline_model = models.TrajectoryGenerator(
            obs_len=8,
            pred_len=12,
            traj_lstm_input_size=2,
            traj_lstm_hidden_size=32,
            n_units=[32, 16, 32],
            n_heads=[4, 1],
            graph_network_out_dims=32,
            dropout=0.0,
            alpha=0.2,
            graph_lstm_hidden_size=32,
            noise_dim=(16,),
            noise_type="gaussian",
        )
        self.assertEqual(
            baseline_model.pred_lstm_hidden_size,
            self.model.pred_lstm_hidden_size,
        )
        self.assertEqual(
            tuple(baseline_model.pred_lstm_model.weight_ih.shape),
            tuple(self.model.pred_lstm_model.weight_ih.shape),
        )
        self.assertEqual(
            tuple(baseline_model.pred_hidden2pos.weight.shape),
            tuple(self.model.pred_hidden2pos.weight.shape),
        )

    def test_cyclestate_can_fully_load_baseline_generator_weights(self):
        baseline_model = models.TrajectoryGenerator(
            obs_len=8,
            pred_len=12,
            traj_lstm_input_size=2,
            traj_lstm_hidden_size=32,
            n_units=[32, 16, 32],
            n_heads=[4, 1],
            graph_network_out_dims=32,
            dropout=0.0,
            alpha=0.2,
            graph_lstm_hidden_size=32,
            noise_dim=(16,),
            noise_type="gaussian",
        )
        skipped = train.maybe_load_compatible_weights(
            self.model, baseline_model.state_dict()
        )
        self.assertEqual([], skipped)


if __name__ == "__main__":
    unittest.main()
