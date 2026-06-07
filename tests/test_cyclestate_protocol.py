import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock
from types import SimpleNamespace

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
evaluate_model = load_module("d2tp_evaluate_model", "D2TP/evaluate_model.py")


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

    def test_seqgat_parameters_receive_gradients_in_baseline_generator(self):
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
        baseline_model.train()
        baseline_model.zero_grad()

        outputs = baseline_model(
            self.obs_traj_rel,
            self.obs_traj,
            self.obs_state,
            self.pred_state,
            self.seq_start_end,
            teacher_forcing_ratio=1.0,
        )
        loss = outputs.sum()
        loss.backward()

        grad_norm = 0.0
        for name, param in baseline_model.seqgatencoder.named_parameters():
            if param.grad is not None:
                grad_norm += float(param.grad.abs().sum().item())
        self.assertGreater(
            grad_norm,
            0.0,
            "seqGAT encoder parameters should receive gradients during baseline training",
        )

    def test_seqgat_parameters_receive_gradients_in_cyclestate_generator(self):
        self.model.train()
        self.model.zero_grad()

        outputs = self.model(
            self.obs_traj_rel,
            self.obs_traj,
            self.obs_state,
            self.pred_state,
            self.seq_start_end,
            teacher_forcing_ratio=1.0,
        )
        loss = outputs.sum()
        loss.backward()

        grad_norm = 0.0
        for name, param in self.model.seqgatencoder.named_parameters():
            if param.grad is not None:
                grad_norm += float(param.grad.abs().sum().item())
        self.assertGreater(
            grad_norm,
            0.0,
            "seqGAT encoder parameters should receive gradients during CycleState training",
        )

    def test_relation_matrix_respects_direction_sector_in_normal_range(self):
        encoder = models.GATEncoder(n_units=[32, 16, 32], n_heads=[4, 1], dropout=0.0, alpha=0.2)
        curr_dire = torch.zeros(1, 3, 6, dtype=torch.float32)
        # agent 0: heading 100 deg -> valid forward sector [38, 162]
        curr_dire[0, 0, 2:4] = torch.tensor([0.0, 0.0])
        curr_dire[0, 0, 5] = 100.0
        # neighbor 1: inside sector, distance small
        curr_dire[0, 1, 2:4] = torch.tensor([1.0, 1.0])  # 45 deg
        curr_dire[0, 1, 5] = 0.0
        # neighbor 2: outside sector, distance small
        curr_dire[0, 2, 2:4] = torch.tensor([1.0, -1.0])  # 315 deg
        curr_dire[0, 2, 5] = 0.0

        relation = encoder.relation_Matrix(curr_dire)

        self.assertEqual(1.0, relation[0, 0, 1].item())
        self.assertEqual(0.0, relation[0, 0, 2].item())

    def test_queue_rollout_uses_previous_step_state(self):
        traffic_context = self.model.build_traffic_context(
            self.obs_traj_rel,
            self.obs_traj,
            self.obs_state,
            self.pred_state,
            self.seq_start_end,
        )
        recorded_inputs = []
        original_rollout = self.model.rollout_queue_features

        def capture_rollout(prev_queue_feature, current_cycle_feature, last_pred_offset, step_index):
            recorded_inputs.append(prev_queue_feature.detach().clone())
            return original_rollout(
                prev_queue_feature,
                current_cycle_feature,
                last_pred_offset,
                step_index,
            )

        with mock.patch.object(
            self.model,
            "rollout_queue_features",
            side_effect=capture_rollout,
        ):
            self.model(
                self.obs_traj_rel,
                self.obs_traj,
                self.obs_state,
                self.pred_state,
                self.seq_start_end,
                traffic_context=traffic_context,
            )

        self.assertEqual(self.model.pred_len, len(recorded_inputs))
        base_queue_feature = traffic_context["meso"]["queue_feature_seq"][-1]
        self.assertTrue(torch.allclose(recorded_inputs[0], base_queue_feature))
        self.assertFalse(torch.allclose(recorded_inputs[1], base_queue_feature))
        self.assertFalse(torch.allclose(recorded_inputs[1], recorded_inputs[0]))

    def test_training_rollout_step_zero_uses_last_observed_offset(self):
        traffic_context = self.model.build_traffic_context(
            self.obs_traj_rel,
            self.obs_traj,
            self.obs_state,
            self.pred_state,
            self.seq_start_end,
        )
        rollout_offsets = []
        original_rollout = self.model.rollout_queue_features

        def capture_rollout(prev_queue_feature, current_cycle_feature, last_pred_offset, step_index):
            rollout_offsets.append((step_index, last_pred_offset.detach().clone()))
            return original_rollout(
                prev_queue_feature,
                current_cycle_feature,
                last_pred_offset,
                step_index,
            )

        with mock.patch.object(
            self.model,
            "rollout_queue_features",
            side_effect=capture_rollout,
        ):
            self.model.train()
            self.model(
                self.obs_traj_rel,
                self.obs_traj,
                self.obs_state,
                self.pred_state,
                self.seq_start_end,
                teacher_forcing_ratio=1.0,
                traffic_context=traffic_context,
            )

        self.assertGreaterEqual(len(rollout_offsets), 1)
        self.assertEqual(0, rollout_offsets[0][0])
        expected_last_observed_offset = self.obs_traj_rel[self.model.obs_len - 1, :, 2:4]
        self.assertTrue(
            torch.allclose(rollout_offsets[0][1], expected_last_observed_offset)
        )

    def test_rollout_decode_context_is_anchored_to_observed_queue_state(self):
        traffic_context = self.model.build_traffic_context(
            self.obs_traj_rel,
            self.obs_traj,
            self.obs_state,
            self.pred_state,
            self.seq_start_end,
        )
        recorded_queue_contexts = []
        original_build_residual = self.model.build_decoder_state_residual

        def capture_queue_context(light_state_embedding, queue_context, cycle_context):
            recorded_queue_contexts.append(queue_context.detach().clone())
            return original_build_residual(
                light_state_embedding,
                queue_context,
                cycle_context,
            )

        with mock.patch.object(
            self.model,
            "build_decoder_state_residual",
            side_effect=capture_queue_context,
        ):
            self.model(
                self.obs_traj_rel,
                self.obs_traj,
                self.obs_state,
                self.pred_state,
                self.seq_start_end,
                traffic_context=traffic_context,
            )

        observed_queue_context = self.model.debug_last_aux["queue_hidden_last"]
        rollout_queue_context = self.model.debug_last_aux["queue_rollout_hidden_seq"][0]
        step0_decode_context = recorded_queue_contexts[1]
        self.assertFalse(torch.allclose(step0_decode_context, rollout_queue_context))
        observed_distance = (step0_decode_context - observed_queue_context).norm(dim=1)
        rollout_distance = (rollout_queue_context - observed_queue_context).norm(dim=1)
        self.assertTrue(torch.all(observed_distance < rollout_distance))

    def test_rollout_residual_scale_zero_keeps_observed_queue_context(self):
        bounded_model = models.CycleStateTrajectoryGenerator(
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
            rollout_residual_scale=0.0,
        )
        observed = torch.randn(3, bounded_model.queue_lstm_hidden_size)
        rollout = observed + torch.randn_like(observed)
        light = torch.randn(3, bounded_model.light_embedding_size)

        decoded = bounded_model.build_rollout_decode_queue_context(
            observed, rollout, light
        )

        self.assertTrue(torch.allclose(decoded, observed))

    def test_detach_rollout_state_keeps_outputs_but_cuts_cross_step_hidden_grad(self):
        detach_model = models.CycleStateTrajectoryGenerator(
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
            detach_rollout_state=True,
        )
        traffic_context = detach_model.build_traffic_context(
            self.obs_traj_rel,
            self.obs_traj,
            self.obs_state,
            self.pred_state,
            self.seq_start_end,
        )
        hidden_requires_grad = []
        original_step = detach_model.rollout_queue_step

        def capture_rollout_step(*args, **kwargs):
            rollout_queue_h_t = args[9]
            hidden_requires_grad.append(rollout_queue_h_t.requires_grad)
            return original_step(*args, **kwargs)

        with mock.patch.object(
            detach_model,
            "rollout_queue_step",
            side_effect=capture_rollout_step,
        ):
            detach_model.train()
            detach_model(
                self.obs_traj_rel,
                self.obs_traj,
                self.obs_state,
                self.pred_state,
                self.seq_start_end,
                teacher_forcing_ratio=1.0,
                traffic_context=traffic_context,
            )

        self.assertIsNotNone(detach_model.debug_last_aux["queue_rollout_pred_seq"])
        self.assertGreaterEqual(len(hidden_requires_grad), 2)
        self.assertTrue(hidden_requires_grad[0])
        self.assertFalse(hidden_requires_grad[1])

    def test_lane_anchor_rollout_is_dynamic_after_first_step(self):
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
        lane_anchor_rollout = self.model.debug_last_aux["lane_queue_rollout_anchor_seq"]
        observed_anchor = traffic_context["meso"]["lane_queue_anchor_seq"][-1]
        self.assertTrue(torch.allclose(lane_anchor_rollout[0], observed_anchor))
        self.assertFalse(torch.allclose(lane_anchor_rollout[1], observed_anchor))

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
            "queue_main_loss",
            "queue_rollout_reg_loss",
            "queue_rollout_cls_loss",
            "queue_rollout_loss",
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
            aux_rollout_weight=None,
            aux_cycle_weight=None,
            grad_clip=None,
            rollout_residual_scale=None,
            detach_rollout_state=None,
        )
        train.apply_stage_defaults(args)
        self.assertTrue(args.generator_only)
        self.assertEqual(0.0, args.gan_weight)
        self.assertGreater(args.aux_queue_weight, 0.0)
        self.assertGreater(args.aux_cycle_weight, 0.0)
        self.assertGreater(args.grad_clip, 0.0)
        self.assertLess(args.rollout_residual_scale, 1.0)
        self.assertTrue(args.detach_rollout_state)

    def test_explicit_stage_overrides_are_preserved_for_ablation(self):
        args = types.SimpleNamespace(
            train_stage="warmup",
            model_type="cyclestate",
            generator_only=True,
            gan_weight=0.0,
            aux_queue_weight=0.0,
            aux_rollout_weight=2.5,
            aux_cycle_weight=0.0,
            grad_clip=0.0,
            rollout_residual_scale=0.75,
            detach_rollout_state=False,
        )
        train.apply_stage_defaults(args)
        self.assertTrue(args.generator_only)
        self.assertEqual(0.0, args.gan_weight)
        self.assertEqual(0.0, args.aux_queue_weight)
        self.assertEqual(2.5, args.aux_rollout_weight)
        self.assertEqual(0.0, args.aux_cycle_weight)
        self.assertEqual(0.0, args.grad_clip)
        self.assertEqual(0.75, args.rollout_residual_scale)
        self.assertFalse(args.detach_rollout_state)

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

    def test_evaluate_helper_matches_scene_level_best_of_k_aggregation(self):
        seq_start_end = torch.tensor([[0, 2]], dtype=torch.long)
        ade_raw = torch.tensor([3.0, 9.0], dtype=torch.float32)
        fde_raw = torch.tensor([1.0, 5.0], dtype=torch.float32)

        ade_sum = train.evaluate_helper([ade_raw], seq_start_end)
        fde_sum = train.evaluate_helper([fde_raw], seq_start_end)

        self.assertEqual(12.0, ade_sum.item())
        self.assertEqual(6.0, fde_sum.item())

    def test_compute_average_displacement_metrics_matches_manual_average(self):
        pred_traj_gt = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 2.0]],
                [[2.0, 0.0], [0.0, 4.0]],
            ],
            dtype=torch.float32,
        )
        pred_traj_fake = torch.tensor(
            [
                [[0.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 3.0]],
            ],
            dtype=torch.float32,
        )
        ade, fde = train.compute_average_displacement_metrics(
            pred_traj_gt, pred_traj_fake
        )
        self.assertAlmostEqual(1.0, ade.item(), places=6)
        self.assertAlmostEqual(1.0, fde.item(), places=6)

    def test_compute_best_of_k_metrics_chooses_lowest_scene_error(self):
        seq_start_end = torch.tensor([[0, 2]], dtype=torch.long)
        ade_candidates = [
            torch.tensor([3.0, 9.0], dtype=torch.float32),
            torch.tensor([1.0, 1.0], dtype=torch.float32),
        ]
        fde_candidates = [
            torch.tensor([4.0, 6.0], dtype=torch.float32),
            torch.tensor([1.0, 2.0], dtype=torch.float32),
        ]
        ade, fde = train.compute_best_of_k_metrics(
            ade_candidates,
            fde_candidates,
            seq_start_end,
            pred_len=12,
            total_traj=2,
        )
        self.assertAlmostEqual((1.0 + 1.0) / (2.0 * 12.0), ade.item(), places=6)
        self.assertAlmostEqual((1.0 + 2.0) / 2.0, fde.item(), places=6)

    def test_should_run_validation_uses_epoch_boundary_for_formal_training(self):
        args = types.SimpleNamespace(
            start_epoch=0,
            val_every=1,
            max_train_batches=0,
            num_epochs=10,
            print_every=1,
        )
        self.assertFalse(train.should_run_validation(args, epoch=0, batch_idx=0, num_batches=5))
        self.assertTrue(train.should_run_validation(args, epoch=0, batch_idx=4, num_batches=5))

    def test_should_run_validation_keeps_batch_level_feedback_for_smoke_runs(self):
        args = types.SimpleNamespace(
            start_epoch=0,
            val_every=1,
            max_train_batches=1,
            num_epochs=0,
            print_every=2,
        )
        self.assertFalse(train.should_run_validation(args, epoch=0, batch_idx=0, num_batches=5))
        self.assertTrue(train.should_run_validation(args, epoch=0, batch_idx=1, num_batches=5))
        self.assertFalse(train.should_run_validation(args, epoch=0, batch_idx=2, num_batches=5))
        self.assertTrue(train.should_run_validation(args, epoch=0, batch_idx=4, num_batches=5))

    def test_should_run_validation_still_triggers_on_last_smoke_batch(self):
        args = types.SimpleNamespace(
            start_epoch=0,
            val_every=1,
            max_train_batches=3,
            num_epochs=0,
            print_every=20,
        )
        self.assertTrue(train.should_run_validation(args, epoch=0, batch_idx=2, num_batches=3))

    def test_apply_stage_defaults_preserves_explicit_num_val_samples(self):
        args = types.SimpleNamespace(
            train_stage="warmup",
            model_type="cyclestate",
            generator_only=None,
            gan_weight=None,
            aux_queue_weight=None,
            aux_rollout_weight=None,
            aux_cycle_weight=None,
            num_val_samples=5,
        )
        train.apply_stage_defaults(args)
        self.assertEqual(5, args.num_val_samples)

    def test_apply_stage_defaults_defaults_rollout_weight_to_queue_weight(self):
        args = types.SimpleNamespace(
            train_stage="warmup",
            model_type="cyclestate",
            generator_only=None,
            gan_weight=None,
            aux_queue_weight=None,
            aux_rollout_weight=None,
            aux_cycle_weight=None,
            num_val_samples=4,
        )
        train.apply_stage_defaults(args)
        self.assertEqual(args.aux_queue_weight, args.aux_rollout_weight)

    def test_apply_stage_defaults_preserves_explicit_rollout_weight(self):
        args = types.SimpleNamespace(
            train_stage="warmup",
            model_type="cyclestate",
            generator_only=None,
            gan_weight=None,
            aux_queue_weight=None,
            aux_rollout_weight=2.5,
            aux_cycle_weight=None,
            num_val_samples=4,
        )
        train.apply_stage_defaults(args)
        self.assertEqual(2.5, args.aux_rollout_weight)

    def test_parser_defaults_train_validation_to_val_split(self):
        args = train.parser.parse_args([])
        self.assertEqual("val", args.val_dset_type)

    def test_parser_supports_explicit_rollout_detach_override(self):
        args = train.parser.parse_args(["--no_detach_rollout_state"])
        train.apply_stage_defaults(args)
        self.assertFalse(args.detach_rollout_state)

    def test_validation_path_uses_configured_split(self):
        args = types.SimpleNamespace(dataset_name="VTP_C", val_dset_type="val")
        self.assertTrue(train.get_validation_dset_path(args).endswith("VTP_C/val"))
        args.val_dset_type = "test"
        self.assertTrue(train.get_validation_dset_path(args).endswith("VTP_C/test"))

    def test_build_optimizers_uses_configured_learning_rate(self):
        args = types.SimpleNamespace(lr=3e-4)
        model = torch.nn.Linear(2, 2)
        discriminator = torch.nn.Linear(2, 1)
        optimizer, optimizer_d = train.build_optimizers(args, model, discriminator)
        self.assertEqual(3e-4, optimizer.param_groups[0]["lr"])
        self.assertEqual(3e-4, optimizer_d.param_groups[0]["lr"])

    def test_extract_state_stability_metrics_reads_debug_norms(self):
        debug = {
            "decoder_state_init_residual_norm": torch.tensor([1.0, 3.0]),
            "decoder_state_step_residual_norm_seq": torch.tensor(
                [[2.0, 4.0], [6.0, 8.0]]
            ),
            "queue_rollout_hidden_seq": torch.tensor(
                [[[3.0, 4.0]], [[0.0, 5.0]]]
            ),
        }
        pred_offsets = torch.tensor([[[3.0, 4.0]], [[6.0, 8.0]]])

        metrics = train.extract_state_stability_metrics(debug, pred_offsets)

        self.assertAlmostEqual(2.0, metrics["decoder_state_init_residual_norm"])
        self.assertAlmostEqual(5.0, metrics["decoder_state_step_residual_norm"])
        self.assertAlmostEqual(5.0, metrics["queue_rollout_hidden_norm"])
        self.assertAlmostEqual(7.5, metrics["pred_offset_norm"])

    def test_evaluate_parser_exposes_rollout_stability_knobs(self):
        args = evaluate_model.parser.parse_args(
            ["--model_type", "cyclestate", "--rollout_residual_scale", "0.35", "--detach_rollout_state"]
        )
        self.assertEqual(0.35, args.rollout_residual_scale)
        self.assertTrue(args.detach_rollout_state)

    def test_evaluate_model_supports_max_eval_batches_and_weighted_averaging(self):
        batch = [
            torch.zeros(8, 2, 10),
            torch.zeros(12, 2, 10),
            torch.zeros(8, 2, 9),
            torch.zeros(12, 2, 9),
            torch.zeros(8, 2, 4),
            torch.zeros(12, 2, 4),
            torch.zeros(2),
            torch.zeros(2, 20),
            torch.tensor([[0, 2]], dtype=torch.long),
        ]
        generator = mock.Mock()
        generator.return_value = torch.zeros(12, 2, 2)
        args = SimpleNamespace(
            device="cpu",
            model_type="d2tpred",
            num_samples=1,
            pred_len=12,
            max_eval_batches=1,
            eval_print_every=1,
        )
        with mock.patch.object(
            evaluate_model,
            "compute_raw_displacement_metrics",
            return_value=(
                torch.tensor([1.0, 3.0], dtype=torch.float32),
                torch.tensor([2.0, 4.0], dtype=torch.float32),
            ),
        ) as raw_metrics, mock.patch.object(
            evaluate_model,
            "relative_to_abs",
            return_value=torch.zeros(12, 2, 2),
        ):
            ade, fde = evaluate_model.evaluate(args, [batch, batch], generator)
        self.assertAlmostEqual((1.0 + 3.0) / (2.0 * 12.0), ade.item(), places=6)
        self.assertAlmostEqual((2.0 + 4.0) / 2.0, fde.item(), places=6)
        self.assertEqual(1, raw_metrics.call_count)


if __name__ == "__main__":
    unittest.main()
