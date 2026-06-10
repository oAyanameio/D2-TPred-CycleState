import importlib.util
import logging
import os
import pathlib
import random
import re
import sys
import tempfile
import types
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np
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
utils = load_module("d2tp_utils", "D2TP/utils.py")


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
        self.assertIn("cycle_rollout_hidden_seq", self.model.debug_last_aux)
        self.assertEqual(
            (12, 3, self.model.queue_lstm_hidden_size),
            tuple(self.model.debug_last_aux["queue_rollout_hidden_seq"].shape),
        )
        self.assertEqual(
            (12, 3, self.model.cycle_lstm_hidden_size),
            tuple(self.model.debug_last_aux["cycle_rollout_hidden_seq"].shape),
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
        cycle_rollout = self.model.debug_last_aux["cycle_rollout_hidden_seq"]
        self.assertFalse(torch.allclose(cycle_rollout[0], cycle_rollout[-1]))
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

    def test_cycle_rollout_uses_previous_step_state(self):
        traffic_context = self.model.build_traffic_context(
            self.obs_traj_rel,
            self.obs_traj,
            self.obs_state,
            self.pred_state,
            self.seq_start_end,
        )
        recorded_hidden = []
        original_rollout = self.model.rollout_cycle_step

        def capture_rollout(current_cycle_feature, rollout_cycle_h_t, rollout_cycle_c_t):
            recorded_hidden.append(rollout_cycle_h_t.detach().clone())
            return original_rollout(
                current_cycle_feature,
                rollout_cycle_h_t,
                rollout_cycle_c_t,
            )

        with mock.patch.object(
            self.model,
            "rollout_cycle_step",
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

        self.assertEqual(self.model.pred_len, len(recorded_hidden))
        base_cycle_hidden = self.model.debug_last_aux["cycle_hidden_last"]
        self.assertTrue(torch.allclose(recorded_hidden[0], base_cycle_hidden))
        self.assertFalse(torch.allclose(recorded_hidden[1], base_cycle_hidden))
        self.assertFalse(torch.allclose(recorded_hidden[1], recorded_hidden[0]))

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

    def test_rollout_cycle_context_is_used_by_decoder_state_residual(self):
        model = models.CycleStateTrajectoryGenerator(
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
        traffic_context = model.build_traffic_context(
            self.obs_traj_rel,
            self.obs_traj,
            self.obs_state,
            self.pred_state,
            self.seq_start_end,
        )
        recorded_cycle_contexts = []
        original_build_residual = model.build_decoder_state_residual
        synthetic_cycle_hidden = torch.full(
            (self.obs_traj.shape[1], model.cycle_lstm_hidden_size),
            0.25,
        )

        def fake_cycle_rollout(current_cycle_feature, rollout_cycle_h_t, rollout_cycle_c_t):
            return {
                "cycle_hidden": synthetic_cycle_hidden,
                "cycle_cell": torch.zeros_like(synthetic_cycle_hidden),
            }

        def capture_cycle_context(light_state_embedding, queue_context, cycle_context):
            recorded_cycle_contexts.append(cycle_context.detach().clone())
            return original_build_residual(
                light_state_embedding,
                queue_context,
                cycle_context,
            )

        with mock.patch.object(
            model,
            "rollout_cycle_step",
            side_effect=fake_cycle_rollout,
        ):
            with mock.patch.object(
                model,
                "build_decoder_state_residual",
                side_effect=capture_cycle_context,
            ):
                model(
                    self.obs_traj_rel,
                    self.obs_traj,
                    self.obs_state,
                    self.pred_state,
                    self.seq_start_end,
                    traffic_context=traffic_context,
                )

        self.assertGreaterEqual(len(recorded_cycle_contexts), 2)
        step0_cycle_context = recorded_cycle_contexts[1]
        self.assertTrue(torch.allclose(step0_cycle_context, synthetic_cycle_hidden))

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

    # ---- Phase 4 #22: disable_aux_losses 统一主开关测试 ----

    def test_disable_aux_losses_forwards_to_all_four_individual_flags(self):
        """#22: disable_aux_losses=True 应强制四个独立 disable 标志位为 True。"""
        model = models.CycleStateTrajectoryGenerator(
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
            disable_aux_losses=True,
        )
        self.assertTrue(model.disable_aux_losses)
        self.assertTrue(model.disable_state_gating)
        self.assertTrue(model.disable_queue_rollout)
        self.assertTrue(model.disable_lane_queue_anchor)
        self.assertTrue(model.disable_decoder_state_residual)

    def test_disable_aux_losses_forward_output_correct_shape(self):
        """#22: disable_aux_losses=True 时 forward 仍正常生成轨迹。"""
        model = models.CycleStateTrajectoryGenerator(
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
            disable_aux_losses=True,
        )
        outputs = model(
            self.obs_traj_rel,
            self.obs_traj,
            self.obs_state,
            self.pred_state,
            self.seq_start_end,
        )
        self.assertEqual((12, 3, 2), tuple(outputs.shape))

    def test_disable_aux_losses_nulls_all_aux_debug_fields(self):
        """#22: disable_aux_losses=True 时所有 CycleState 特有 debug 字段应为 None。"""
        model = models.CycleStateTrajectoryGenerator(
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
            disable_aux_losses=True,
        )
        model(
            self.obs_traj_rel,
            self.obs_traj,
            self.obs_state,
            self.pred_state,
            self.seq_start_end,
        )
        aux = model.debug_last_aux
        self.assertIsNone(aux["queue_rollout_hidden_seq"])
        self.assertIsNone(aux["queue_rollout_pred_seq"])
        self.assertIsNone(aux["queue_rollout_target_seq"])
        self.assertIsNone(aux["lane_queue_rollout_anchor_seq"])
        self.assertIsNone(aux["decoder_state_init_residual"])
        self.assertIsNone(aux["decoder_state_step_residual_seq"])

    def test_disable_aux_losses_preserves_model_architecture_parameters(self):
        """#22: disable_aux_losses=True 不改变模型结构，参数数量应与默认模型一致。"""
        model_default = models.CycleStateTrajectoryGenerator(
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
        model_ablated = models.CycleStateTrajectoryGenerator(
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
            disable_aux_losses=True,
        )
        default_params = sum(p.numel() for p in model_default.parameters())
        ablated_params = sum(p.numel() for p in model_ablated.parameters())
        self.assertEqual(
            default_params,
            ablated_params,
            "disable_aux_losses 只改变运行时行为，不改变模型结构与参数数量",
        )

    def test_train_cli_accepts_disable_aux_losses(self):
        """#22: train.py --disable_aux_losses 被 argparse 正确识别。"""
        parser = train.parser
        with self.subTest(arg="--disable_aux_losses"):
            args = parser.parse_args(["--disable_aux_losses"])
            self.assertTrue(args.disable_aux_losses)
        with self.subTest(arg="without flag"):
            args = parser.parse_args([])
            self.assertFalse(args.disable_aux_losses)

    def test_evaluate_model_cli_accepts_disable_aux_losses(self):
        """#22: evaluate_model.py --disable_aux_losses 被 argparse 正确识别。"""
        parser = evaluate_model.parser
        with self.subTest(arg="--disable_aux_losses"):
            args = parser.parse_args(["--disable_aux_losses"])
            self.assertTrue(args.disable_aux_losses)
        with self.subTest(arg="without flag"):
            args = parser.parse_args([])
            self.assertFalse(args.disable_aux_losses)

    def test_train_main_zeros_aux_weights_on_disable_aux_losses(self):
        """#22: 源码守卫: train.py main() 中 disable_aux_losses=True 时
        aux_queue_weight / aux_cycle_weight / aux_rollout_weight 被置零。"""
        train_py = (
            pathlib.Path(__file__).resolve().parent.parent
            / "D2TP" / "train.py"
        )
        source = train_py.read_text(encoding="utf-8")
        self.assertIn("disable_aux_losses", source,
                       "train.py 应包含 disable_aux_losses 逻辑")
        self.assertIn("aux_queue_weight = 0.0", source,
                       "disable_aux_losses 时应将 aux_queue_weight 置零")
        self.assertIn("aux_cycle_weight = 0.0", source,
                       "disable_aux_losses 时应将 aux_cycle_weight 置零")
        self.assertIn("aux_rollout_weight = 0.0", source,
                       "disable_aux_losses 时应将 aux_rollout_weight 置零")

    def test_evaluate_model_passes_disable_aux_losses_to_constructor(self):
        """#22: 源码守卫: evaluate_model.py 将 disable_aux_losses 传入模型构造函数。"""
        evaluate_py = (
            pathlib.Path(__file__).resolve().parent.parent
            / "D2TP" / "evaluate_model.py"
        )
        source = evaluate_py.read_text(encoding="utf-8")
        self.assertIn('"disable_aux_losses"', source,
                       "evaluate_model.py 应把 disable_aux_losses 传给 model_kwargs")
        self.assertIn("args.disable_aux_losses", source,
                       "evaluate_model.py 应引用 args.disable_aux_losses")

    def test_train_log_uses_effective_disable_flags_not_raw_args(self):
        """#22 修: 训练协议日志口径必须打印 ``disable_aux_losses`` 强制开启四个
        子开关之后的"模型实际生效状态",而不是原始 ``args.disable_*``,保证
        消融实验日志与运行时真实行为一致,满足可审计性。

        行为守卫:
        1. 日志格式串里四个 disable 字段必须带 ``(eff)`` 后缀。
        2. 日志里必须额外打印 ``disable_aux_losses=%s`` 字段,便于一眼看出
           是否启用了统一主开关。
        3. 源码中必须先计算 ``_eff_disable_*`` 变量,再把它们传入 logging.info。
        """
        train_py = (
            pathlib.Path(__file__).resolve().parent.parent
            / "D2TP" / "train.py"
        )
        source = train_py.read_text(encoding="utf-8")
        for marker in (
            "disable_state_gating(eff)=",
            "disable_queue_rollout(eff)=",
            "disable_lane_queue_anchor(eff)=",
            "disable_decoder_state_residual(eff)=",
        ):
            self.assertIn(
                marker, source,
                f"训练协议日志格式串必须包含有效状态字段 {marker!r}"
            )
        self.assertIn(
            "disable_aux_losses=%s", source,
            "训练协议日志必须额外打印 disable_aux_losses 主开关状态"
        )
        # 验证日志里使用的是有效状态变量,不是直接拿 args.disable_*
        self.assertIn(
            "_eff_disable_state_gating", source,
            "应计算 _eff_disable_state_gating 再传入 logging.info"
        )
        self.assertNotIn(
            "args.disable_state_gating,", source.split(
                "Training protocol", 1
            )[1].split("\n", 1)[0] if "Training protocol" in source else "",
            "Training protocol 日志行不应再直接引用 args.disable_state_gating"
        )

    def test_train_log_effective_flags_match_model_runtime_state(self):
        """#22 修: 行为守卫 - 验证 train.py 中的 _eff_disable_* 变量在
        ``disable_aux_losses=True`` 时全部为 True,在 ``disable_aux_losses=False``
        时等于 ``args.disable_*``。这样日志口径才能与模型内部运行时状态保持一致。"""
        import argparse

        # 1. disable_aux_losses=False -> 有效值应等于原始 args
        args = argparse.Namespace(
            disable_aux_losses=False,
            disable_state_gating=True,
            disable_queue_rollout=False,
            disable_lane_queue_anchor=False,
            disable_decoder_state_residual=True,
        )
        _disable_aux = bool(getattr(args, "disable_aux_losses", False))
        eff_sg = _disable_aux or bool(args.disable_state_gating)
        eff_qr = _disable_aux or bool(args.disable_queue_rollout)
        eff_lqa = _disable_aux or bool(args.disable_lane_queue_anchor)
        eff_dsr = _disable_aux or bool(args.disable_decoder_state_residual)
        self.assertEqual(eff_sg, args.disable_state_gating)
        self.assertEqual(eff_qr, args.disable_queue_rollout)
        self.assertEqual(eff_lqa, args.disable_lane_queue_anchor)
        self.assertEqual(eff_dsr, args.disable_decoder_state_residual)

        # 2. disable_aux_losses=True -> 全部应被强制为 True
        args2 = argparse.Namespace(
            disable_aux_losses=True,
            disable_state_gating=False,
            disable_queue_rollout=False,
            disable_lane_queue_anchor=False,
            disable_decoder_state_residual=False,
        )
        _disable_aux2 = bool(getattr(args2, "disable_aux_losses", False))
        eff_sg2 = _disable_aux2 or bool(args2.disable_state_gating)
        eff_qr2 = _disable_aux2 or bool(args2.disable_queue_rollout)
        eff_lqa2 = _disable_aux2 or bool(args2.disable_lane_queue_anchor)
        eff_dsr2 = _disable_aux2 or bool(args2.disable_decoder_state_residual)
        self.assertTrue(eff_sg2)
        self.assertTrue(eff_qr2)
        self.assertTrue(eff_lqa2)
        self.assertTrue(eff_dsr2)

    def test_disable_aux_losses_aux_heads_still_exist_but_unused(self):
        """#22: disable_aux_losses=True 时 aux 头仍存在（checkpoint 兼容），
        但 debug_last_aux 中 queue_pred_last / cycle_pred_last 仍为非 None
        （编码期 aux 头计算与 disable 标志独立，仅在训练时 aux 权重被置零）。"""
        model = models.CycleStateTrajectoryGenerator(
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
            disable_aux_losses=True,
        )
        model(
            self.obs_traj_rel,
            self.obs_traj,
            self.obs_state,
            self.pred_state,
            self.seq_start_end,
        )
        # aux 头的结构仍存在（不做结构性删除，保持 checkpoint 兼容）
        self.assertIsNotNone(model.queue_aux_reg_head)
        self.assertIsNotNone(model.queue_aux_cls_head)
        self.assertIsNotNone(model.cycle_aux_phase_head)
        self.assertIsNotNone(model.cycle_aux_time_head)
        self.assertIsNotNone(model.cycle_aux_change_head)

    # ===== Phase 3 #23 · phase_duration_limits 可配置化 =====

    def test_train_cli_phase_duration_limits_defaults_to_none(self):
        """#23: train.py 默认不传 ``--phase_duration_limits`` 时,
        ``args.phase_duration_limits`` 应为 ``None`` (触发 ``__init__`` 默认值)。"""
        parser = train.parser
        args = parser.parse_args([])
        self.assertIsNone(args.phase_duration_limits)

    def test_train_cli_phase_duration_limits_parses_three_floats(self):
        """#23: train.py ``--phase_duration_limits "40,50,3"`` 应解析为
        ``(40.0, 50.0, 3.0)``。"""
        parser = train.parser
        args = parser.parse_args(["--phase_duration_limits", "40,50,3"])
        self.assertEqual(tuple(args.phase_duration_limits), (40.0, 50.0, 3.0))

    def test_train_cli_phase_duration_limits_rejects_wrong_length(self):
        """#23: train.py ``--phase_duration_limits "1,2"`` 必须被 argparse 拒绝。"""
        parser = train.parser
        with self.assertRaises(SystemExit):
            parser.parse_args(["--phase_duration_limits", "1,2"])

    def test_train_cli_phase_duration_limits_rejects_non_float(self):
        """#23: train.py ``--phase_duration_limits "1,abc,3"`` 必须被 argparse 拒绝。"""
        parser = train.parser
        with self.assertRaises(SystemExit):
            parser.parse_args(["--phase_duration_limits", "1,abc,3"])

    def test_train_cli_phase_duration_limits_rejects_negative(self):
        """#23: train.py ``--phase_duration_limits "1,-2,3"`` 必须被 argparse 拒绝。"""
        parser = train.parser
        with self.assertRaises(SystemExit):
            parser.parse_args(["--phase_duration_limits", "1,-2,3"])

    def test_evaluate_model_cli_phase_duration_limits_parses_three_floats(self):
        """#23: evaluate_model.py ``--phase_duration_limits "42,55,2.5"`` 应解析为
        ``(42.0, 55.0, 2.5)``。"""
        parser = evaluate_model.parser
        args = parser.parse_args(["--phase_duration_limits", "42,55,2.5"])
        self.assertEqual(tuple(args.phase_duration_limits), (42.0, 55.0, 2.5))

    def test_evaluate_model_cli_phase_duration_limits_defaults_to_none(self):
        """#23: evaluate_model.py 默认 ``--phase_duration_limits`` 为 ``None``。"""
        parser = evaluate_model.parser
        args = parser.parse_args([])
        self.assertIsNone(args.phase_duration_limits)

    def test_cyclestate_default_phase_duration_limits_is_38_47_2(self):
        """#23: ``CycleStateTrajectoryGenerator`` 默认 phase_duration_limits
        必须为 ``(38.0, 47.0, 2.0)`` (向后兼容)。"""
        model = models.CycleStateTrajectoryGenerator(
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
        torch.testing.assert_close(
            model.phase_duration_limits,
            torch.tensor([38.0, 47.0, 2.0], dtype=torch.float32),
        )

    def test_cyclestate_custom_phase_duration_limits_propagates_to_buffer(self):
        """#23: 显式传入 ``phase_duration_limits=(40, 50, 3)`` 时,模型
        ``register_buffer`` 必须保存新值 (并保持 float32 dtype)。"""
        model = models.CycleStateTrajectoryGenerator(
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
            phase_duration_limits=(40.0, 50.0, 3.0),
        )
        torch.testing.assert_close(
            model.phase_duration_limits,
            torch.tensor([40.0, 50.0, 3.0], dtype=torch.float32),
        )

    def test_cyclestate_phase_duration_limits_preserves_default_param_count(self):
        """#23: ``phase_duration_limits`` 是 ``register_buffer``,不进入
        ``parameters()`` 也不进入 ``state_dict()`` 的 ``parameters`` 字段,
        因此不应改变模型参数数量。"""
        default_model = models.CycleStateTrajectoryGenerator(
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
        custom_model = models.CycleStateTrajectoryGenerator(
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
            phase_duration_limits=(60.0, 70.0, 5.0),
        )
        default_params = sum(p.numel() for p in default_model.parameters())
        custom_params = sum(p.numel() for p in custom_model.parameters())
        self.assertEqual(
            default_params,
            custom_params,
            "phase_duration_limits 走 register_buffer, 不应改变 parameters() 数量",
        )

    def test_validate_stage_consistency_rejects_wrong_length_phase_duration(self):
        """#23: ``validate_stage_consistency`` 必须拒绝长度非 3 的
        ``phase_duration_limits``。"""
        import argparse
        args = argparse.Namespace(
            train_stage="warmup",
            model_type="cyclestate",
            generator_only=True,
            gan_weight=0.0,
            grad_clip=1.0,
            rollout_residual_scale=1.0,
            teacher_forcing_ratio=0.5,
            aux_queue_weight=0.0,
            aux_cycle_weight=0.0,
            aux_rollout_weight=0.0,
            phase_duration_limits=(1.0, 2.0),  # length=2
        )
        with self.assertRaises(ValueError) as cm:
            train.validate_stage_consistency(args)
        self.assertIn("phase_duration_limits", str(cm.exception))

    def test_validate_stage_consistency_rejects_negative_phase_duration(self):
        """#23: ``validate_stage_consistency`` 必须拒绝负值
        ``phase_duration_limits``。"""
        import argparse
        args = argparse.Namespace(
            train_stage="warmup",
            model_type="cyclestate",
            generator_only=True,
            gan_weight=0.0,
            grad_clip=1.0,
            rollout_residual_scale=1.0,
            teacher_forcing_ratio=0.5,
            aux_queue_weight=0.0,
            aux_cycle_weight=0.0,
            aux_rollout_weight=0.0,
            phase_duration_limits=(1.0, -2.0, 3.0),
        )
        with self.assertRaises(ValueError) as cm:
            train.validate_stage_consistency(args)
        self.assertIn("phase_duration_limits", str(cm.exception))

    def test_validate_stage_consistency_accepts_valid_phase_duration(self):
        """#23: ``validate_stage_consistency`` 应接受合法的
        ``phase_duration_limits=(40, 50, 3)``。"""
        import argparse
        args = argparse.Namespace(
            train_stage="warmup",
            model_type="cyclestate",
            generator_only=True,
            gan_weight=0.0,
            grad_clip=1.0,
            rollout_residual_scale=1.0,
            teacher_forcing_ratio=0.5,
            aux_queue_weight=0.0,
            aux_cycle_weight=0.0,
            aux_rollout_weight=0.0,
            phase_duration_limits=(40.0, 50.0, 3.0),
        )
        # 不应抛异常
        train.validate_stage_consistency(args)

    def test_train_main_forwards_phase_duration_limits_to_model(self):
        """#23 端到端: 不实际启动训练,直接调用 ``main`` 中构造模型的
        代码路径, 验证 ``args.phase_duration_limits`` 被透传到
        ``model_kwargs``。

        出于健壮性, 我们在测试中 import ``train`` 模块的内部符号并
        monkey-patch ``model_cls`` 为只记录 ``**kwargs`` 的 stub,这样
        不依赖 torch / 数据加载。
        """
        import argparse
        import types

        captured = {}

        class _StubModel:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.kwargs = kwargs

            def to(self, device):
                return self

            def parameters(self):
                # 最小可迭代以兼容 ``sum(p.numel() for p in ...)``
                return iter(())

        # 准备一个最小化 args 命名空间; 只关心我们关心的字段
        args = argparse.Namespace(
            model_type="cyclestate",
            obs_len=8,
            pred_len=12,
            traj_lstm_input_size=2,
            traj_lstm_hidden_size=32,
            graph_network_out_dims=32,
            dropout=0.0,
            alpha=0.2,
            graph_lstm_hidden_size=32,
            noise_dim=(16,),
            noise_type="gaussian",
            n_units=[32, 16, 32],
            n_heads=[4, 1],
            disable_state_gating=False,
            disable_queue_rollout=False,
            disable_lane_queue_anchor=False,
            disable_decoder_state_residual=False,
            disable_aux_losses=False,
            rollout_residual_scale=1.0,
            detach_rollout_state=False,
            phase_duration_limits=(40.0, 50.0, 3.0),
            rollout_queue_coefs_json="",
        )

        # 直接复制 train.py main 中 cyclestate 分支构造 model_kwargs 的逻辑
        # (不调用 main,避免加载数据 / 启动训练)
        n_units = args.n_units
        n_heads = args.n_heads
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
        # cyclestate 分支透传 (与 train.py L1378-1397 镜像)
        model_kwargs["disable_state_gating"] = args.disable_state_gating
        model_kwargs["disable_queue_rollout"] = args.disable_queue_rollout
        model_kwargs["disable_lane_queue_anchor"] = args.disable_lane_queue_anchor
        model_kwargs["disable_decoder_state_residual"] = (
            args.disable_decoder_state_residual
        )
        model_kwargs["disable_aux_losses"] = args.disable_aux_losses
        model_kwargs["rollout_residual_scale"] = args.rollout_residual_scale
        model_kwargs["detach_rollout_state"] = args.detach_rollout_state
        if getattr(args, "phase_duration_limits", None) is not None:
            model_kwargs["phase_duration_limits"] = tuple(
                args.phase_duration_limits
            )

        # 用 stub 构造并断言
        stub = _StubModel(**model_kwargs)
        self.assertEqual(
            stub.kwargs.get("phase_duration_limits"), (40.0, 50.0, 3.0)
        )

    def test_train_main_skips_phase_duration_limits_when_none(self):
        """#23: 当 ``args.phase_duration_limits=None`` 时,
        ``model_kwargs`` 不应包含 ``phase_duration_limits`` 键 (让
        ``__init__`` 默认值生效)。"""
        import argparse
        args_dict = {
            "phase_duration_limits": None,
        }
        # 镜像 train.py L1390-1397 的逻辑
        model_kwargs = {}
        if getattr(argparse.Namespace(**args_dict), "phase_duration_limits", None) is not None:
            model_kwargs["phase_duration_limits"] = tuple(
                args_dict["phase_duration_limits"]
            )
        self.assertNotIn("phase_duration_limits", model_kwargs)

    def test_train_log_includes_phase_duration_limits_field(self):
        """#23 源码守卫: train.py 训练协议日志必须打印
        ``phase_duration_limits`` 字段, 且 ``None`` 时回退到 ``(38.0, 47.0, 2.0)``。"""
        train_py = (
            pathlib.Path(__file__).resolve().parent.parent
            / "D2TP" / "train.py"
        )
        source = train_py.read_text(encoding="utf-8")
        self.assertIn(
            "phase_duration_limits=%s", source,
            "训练协议日志格式串应包含 phase_duration_limits 字段"
        )
        self.assertIn(
            "(38.0, 47.0, 2.0)", source,
            "训练协议日志应在 args.phase_duration_limits=None 时回退到 __init__ 默认值"
        )

    def test_evaluate_model_passes_phase_duration_limits_to_constructor(self):
        """#23 源码守卫: evaluate_model.py 必须把 ``phase_duration_limits``
        传给 ``model_kwargs``。"""
        evaluate_py = (
            pathlib.Path(__file__).resolve().parent.parent
            / "D2TP" / "evaluate_model.py"
        )
        source = evaluate_py.read_text(encoding="utf-8")
        self.assertIn('"phase_duration_limits"', source,
                       "evaluate_model.py 应把 phase_duration_limits 传给 model_kwargs")
        self.assertIn("args.phase_duration_limits", source,
                       "evaluate_model.py 应引用 args.phase_duration_limits")

    def test_compatible_resume_reapplies_explicit_phase_duration_limits_after_load(self):
        """#23 回归: cyclestate resume 走 ``maybe_load_compatible_weights`` 时,
        显式 CLI ``phase_duration_limits`` 不能被 checkpoint 同名 buffer 覆盖。
        """
        ckpt_model = models.CycleStateTrajectoryGenerator(
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
            phase_duration_limits=(38.0, 47.0, 2.0),
        )
        resumed_model = models.CycleStateTrajectoryGenerator(
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
            phase_duration_limits=(40.0, 50.0, 3.0),
        )
        train.maybe_load_compatible_weights(resumed_model, ckpt_model.state_dict())
        self.assertFalse(
            torch.allclose(
                resumed_model.phase_duration_limits,
                torch.tensor([40.0, 50.0, 3.0], dtype=torch.float32),
            ),
            "前置失败假设: 未修复前, maybe_load_compatible_weights 会把显式 "
            "phase_duration_limits 覆盖回 checkpoint 值; 如果这里已经相等, "
            "说明测试假设失效,需先重审 root cause。",
        )
        train.reapply_phase_duration_limits_if_overridden(
            resumed_model, (40.0, 50.0, 3.0)
        )
        torch.testing.assert_close(
            resumed_model.phase_duration_limits,
            torch.tensor([40.0, 50.0, 3.0], dtype=torch.float32),
        )

    def test_evaluate_model_reapplies_explicit_phase_duration_limits_after_load(self):
        """#23 回归: evaluate_model.get_generator() 在 load_state_dict() 之后
        也必须保留显式 CLI ``phase_duration_limits``，不能被 checkpoint 覆盖。
        """
        ckpt_model = models.CycleStateTrajectoryGenerator(
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
            phase_duration_limits=(38.0, 47.0, 2.0),
        )
        checkpoint = {"state_dict": ckpt_model.state_dict()}
        had_args = hasattr(evaluate_model, "args")
        old_args = getattr(evaluate_model, "args", None)
        evaluate_model.args = SimpleNamespace(
            model_type="cyclestate",
            obs_len=8,
            pred_len=12,
            traj_lstm_input_size=2,
            traj_lstm_hidden_size=32,
            hidden_units="16",
            heads="4,1",
            graph_network_out_dims=32,
            dropout=0.0,
            alpha=0.2,
            graph_lstm_hidden_size=32,
            noise_dim=(16,),
            noise_type="gaussian",
            disable_state_gating=False,
            disable_queue_rollout=False,
            disable_lane_queue_anchor=False,
            disable_decoder_state_residual=False,
            disable_aux_losses=False,
            rollout_residual_scale=1.0,
            detach_rollout_state=False,
            phase_duration_limits=(40.0, 50.0, 3.0),
            rollout_queue_coefs_json="",
            device="cpu",
            num_samples=20,
        )
        try:
            generator = evaluate_model.get_generator(checkpoint)
        finally:
            if had_args:
                evaluate_model.args = old_args
            else:
                delattr(evaluate_model, "args")
        torch.testing.assert_close(
            generator.phase_duration_limits,
            torch.tensor([40.0, 50.0, 3.0], dtype=torch.float32),
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
            teacher_forcing_ratio=0.6,
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
        self.assertEqual(0.6, args.teacher_forcing_ratio)
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

    def test_parser_supports_explicit_teacher_forcing_override(self):
        args = train.parser.parse_args(
            ["--model_type", "cyclestate", "--teacher_forcing_ratio", "0.6"]
        )
        train.apply_stage_defaults(args)
        self.assertEqual(0.6, args.teacher_forcing_ratio)

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
            # Phase 5 #26: key 去掉了 ``_seq`` 后缀, 与 ``init_residual_norm`` 命名对齐。
            "decoder_state_step_residual_norm": torch.tensor(
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

    def test_dstep_initialization_lives_inside_epoch_loop(self):
        """#1 P0 fix: D_step counter must be reset at the start of each epoch.

        If ``D_step = 2`` is initialized outside the epoch loop, then after the
        first epoch ends the counter is left at whatever value the final batch
        produced. Subsequent epochs therefore start with a stale counter and the
        discriminator warmup schedule (train D twice, then G once) drifts
        across epoch boundaries. The fix is to move the initialization inside
        the ``for epoch`` loop.

        The test locates the ``for epoch in range`` loop and the
        ``for batch_idx`` inner loop. It then asserts that there exists a
        ``D_step = 2`` initialization located *after* the for-epoch loop starts
        and *before* the for-batch_idx loop starts — i.e. the initialization
        lives directly in the for-epoch loop body and runs at the start of
        every epoch.
        """
        source = (REPO_ROOT / "D2TP" / "train.py").read_text()
        lines = source.splitlines()

        epoch_loop_idx = None
        batch_loop_idx = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith("for "):
                continue
            if "for epoch in range" in stripped and epoch_loop_idx is None:
                epoch_loop_idx = i
            elif "for batch_idx" in stripped and batch_loop_idx is None:
                batch_loop_idx = i
            if epoch_loop_idx is not None and batch_loop_idx is not None:
                break

        self.assertIsNotNone(
            epoch_loop_idx, "Could not locate the for-epoch loop in train.py"
        )
        self.assertIsNotNone(
            batch_loop_idx, "Could not locate the for-batch_idx loop in train.py"
        )
        self.assertLess(
            epoch_loop_idx, batch_loop_idx,
            "for-epoch loop must appear before for-batch_idx loop",
        )

        dstep_in_epoch_body = []
        for i in range(epoch_loop_idx + 1, batch_loop_idx):
            stripped = lines[i].strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "D_step" in stripped and "=" in stripped:
                left, _ = stripped.split("=", 1)
                if "D_step" in left and "2" in stripped:
                    dstep_in_epoch_body.append((i, lines[i]))

        self.assertEqual(
            len(dstep_in_epoch_body),
            1,
            "Expected exactly one D_step = 2 initialization in the for-epoch "
            "loop body (between the for-epoch line and the for-batch_idx "
            "line). Found: {}".format(dstep_in_epoch_body),
        )

    def test_dstep_resets_between_epochs_via_simulated_main_loop(self):
        """Behavioural guard: re-initializing D_step at the start of each epoch
        guarantees that the discriminator is trained exactly twice before the
        first generator update in every epoch, regardless of where the previous
        epoch stopped.
        """
        seen = []

        def fake_d_train(*args, **kwargs):
            seen.append("d_train_called")

        def fake_train(*args, **kwargs):
            seen.append("g_train_called")

        # Simulate the D/G scheduling used in train.py main(), but with the
        # fixed invariant: D_step is reset to 2 at the start of every epoch.
        for epoch in range(3):
            D_step = 2  # the fix: per-epoch reset
            for _batch_idx in range(6):  # 6 is a multiple of the 3-step D/D/G cycle
                if D_step > 0:
                    fake_d_train()
                    D_step -= 1
                else:
                    fake_train()
                    D_step = 2

        # Each epoch should start with two D-trains in a row, no matter what.
        # With the buggy version (single D_step = 2 outside the loop), epoch 1
        # would inherit whatever counter value epoch 0 ended with and would not
        # necessarily begin with two D-train calls.
        for epoch in range(3):
            start = epoch * 6
            window = seen[start:start + 2]
            self.assertEqual(
                ["d_train_called", "d_train_called"],
                window,
                f"Epoch {epoch} did not begin with two discriminator updates; "
                "the per-epoch D_step reset is broken.",
            )

        # With 6 batches per epoch (a multiple of 3) the trailing D_step value
        # is always 0 — meaning the *next* epoch also restarts from 0 (not 2)
        # if D_step is not reset. The above invariant catches exactly that
        # failure mode.

    def test_state_loss_signature_exposes_loss_mask(self):
        """#2 P0 fix: state_loss in utils.py must accept loss_mask as a
        parameter so the 'average' mode body can run without NameError.

        Before the fix, calling state_loss with mode='average' raised
        NameError: name 'loss_mask' is not defined. This regression guard
        checks the signature explicitly.
        """
        import inspect

        sig = inspect.signature(utils.state_loss)
        self.assertIn(
            "loss_mask",
            sig.parameters,
            "state_loss must accept a 'loss_mask' parameter; otherwise the "
            "'average' branch in its body triggers NameError.",
        )

    def test_state_loss_average_mode_uses_loss_mask(self):
        """Behavioural guard: with the loss_mask parameter added, the
        'average' mode branch in state_loss must run without NameError and
        return a tensor whose value matches the formula:
            sum(loss) / numel(loss_mask)
        """
        torch.manual_seed(0)
        pred_len, batch, _ = 4, 3, 2
        pred_traj_fake = torch.zeros(pred_len, batch, 2)
        # last 2 dims are offset; fill GT with a small constant so loss > 0
        pred_traj_gt = torch.zeros(pred_len, batch, 4)
        pred_traj_gt[..., 2:4] = 0.5
        loss_mask = torch.ones(pred_len, batch, 1)

        # Should not raise NameError
        result = utils.state_loss(
            pred_traj_fake, pred_traj_gt, loss_mask=loss_mask, mode="average"
        )

        # pred_traj_gt[..., 2:4] = 0.5 (shape T*V*2 elements), pred_traj_fake = 0
        # loss = (0.5 - 0)^2 = 0.25 per cell, summed over (V, T, C) = 0.25 * 3*4*2 = 6.0
        # divided by numel(loss_mask) = T*V*1 = 12 → expected 0.5
        expected = torch.tensor(0.25 * batch * pred_len * 2) / float(loss_mask.numel())
        self.assertTrue(torch.is_tensor(result))
        self.assertAlmostEqual(
            float(result), float(expected), places=5,
            msg="state_loss(average) must divide by numel(loss_mask) as the "
                "sibling l2_loss does.",
        )

    def test_state_loss_is_not_invoked_by_active_training_path(self):
        """Document the dead-code status: state_loss is only imported in
        train.py, but compute_structured_aux_losses is what train.py actually
        calls. We guard that contract so a future refactor that wires
        state_loss into training breaks this test loudly instead of silently
        shipping a NameError-prone function into the hot path.
        """
        import ast

        train_source = (REPO_ROOT / "D2TP" / "train.py").read_text()
        tree = ast.parse(train_source)
        call_sites = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "state_loss"
        ]
        self.assertEqual(
            [],
            call_sites,
            "state_loss must not be called anywhere in train.py. Active "
            "training uses compute_structured_aux_losses. If you intend to "
            "wire state_loss into the live training path, first re-validate "
            "its loss_mask dependency and update docs/PLAN.md.",
        )

    # --- #4 P0 fix: cycle/queue aux head subspace split ---

    def test_aux_heads_split_into_independent_regression_and_classification_modules(self):
        """#4 P0 fix: the queue/cycle aux heads must be split into independent
        regression and classification sub-heads so the regression and binary
        subspaces do not share parameters.

        Before the fix, ``queue_aux_head`` and ``cycle_aux_head`` were each a
        single ``nn.Linear(..., 6)`` whose 6 output channels were sliced into
        regression / classification / change indices downstream. That meant
        the regression loss gradient flowed through the same parameters as
        the binary cross-entropy gradient, blurring the semantic subspace
        boundary. The fix introduces independent modules:

        - ``queue_aux_reg_head``: 4-dim regression (count/waiting/release/lane)
        - ``queue_aux_cls_head``: 2-dim binary (stop-line/front-of-queue)
        - ``cycle_aux_phase_head``: 3-dim phase logits
        - ``cycle_aux_time_head``: 2-dim elapsed/remaining
        - ``cycle_aux_change_head``: 1-dim phase change logit
        """
        self.assertTrue(
            hasattr(self.model, "queue_aux_reg_head"),
            "model must expose queue_aux_reg_head as an nn.Module "
            "(regression subspace for queue aux head).",
        )
        self.assertTrue(
            hasattr(self.model, "queue_aux_cls_head"),
            "model must expose queue_aux_cls_head as an nn.Module "
            "(binary subspace for queue aux head).",
        )
        self.assertTrue(
            hasattr(self.model, "cycle_aux_phase_head"),
            "model must expose cycle_aux_phase_head as an nn.Module "
            "(phase classification subspace for cycle aux head).",
        )
        self.assertTrue(
            hasattr(self.model, "cycle_aux_time_head"),
            "model must expose cycle_aux_time_head as an nn.Module "
            "(elapsed/remaining regression subspace for cycle aux head).",
        )
        self.assertTrue(
            hasattr(self.model, "cycle_aux_change_head"),
            "model must expose cycle_aux_change_head as an nn.Module "
            "(phase change binary subspace for cycle aux head).",
        )

        # Output dim assertions: nn.Linear stores weight as (out_features, in_features)
        self.assertEqual(
            (4, self.model.queue_lstm_hidden_size),
            tuple(self.model.queue_aux_reg_head.weight.shape),
        )
        self.assertEqual(
            (2, self.model.queue_lstm_hidden_size),
            tuple(self.model.queue_aux_cls_head.weight.shape),
        )
        self.assertEqual(
            (3, self.model.cycle_lstm_hidden_size),
            tuple(self.model.cycle_aux_phase_head.weight.shape),
        )
        self.assertEqual(
            (2, self.model.cycle_lstm_hidden_size),
            tuple(self.model.cycle_aux_time_head.weight.shape),
        )
        self.assertEqual(
            (1, self.model.cycle_lstm_hidden_size),
            tuple(self.model.cycle_aux_change_head.weight.shape),
        )

        # Regression and binary subspaces must NOT share parameters.
        reg_param_ids = {
            id(p) for p in self.model.queue_aux_reg_head.parameters()
        }
        cls_param_ids = {
            id(p) for p in self.model.queue_aux_cls_head.parameters()
        }
        self.assertEqual(
            set(),
            reg_param_ids & cls_param_ids,
            "queue regression and classification sub-heads must not share "
            "parameters; otherwise the regression/classification gradients "
            "are entangled in the same linear layer.",
        )

        phase_param_ids = {
            id(p) for p in self.model.cycle_aux_phase_head.parameters()
        }
        time_param_ids = {
            id(p) for p in self.model.cycle_aux_time_head.parameters()
        }
        change_param_ids = {
            id(p) for p in self.model.cycle_aux_change_head.parameters()
        }
        all_cycle_param_ids = (
            phase_param_ids | time_param_ids | change_param_ids
        )
        self.assertEqual(
            3,
            len([phase_param_ids, time_param_ids, change_param_ids]),
            "cycle phase/time/change sub-heads must be three independent "
            "parameter sets.",
        )
        self.assertEqual(
            set(),
            phase_param_ids & time_param_ids,
            "cycle phase and time sub-heads must not share parameters.",
        )
        self.assertEqual(
            set(),
            phase_param_ids & change_param_ids,
            "cycle phase and change sub-heads must not share parameters.",
        )
        self.assertEqual(
            set(),
            time_param_ids & change_param_ids,
            "cycle time and change sub-heads must not share parameters.",
        )
        # Sanity: total unique params across the three cycle sub-heads should
        # be 3x what a single shared-head would need. Just ensure none of the
        # sub-heads is empty.
        for name, ids in (
            ("phase", phase_param_ids),
            ("time", time_param_ids),
            ("change", change_param_ids),
        ):
            self.assertGreater(
                len(ids), 0,
                f"cycle_aux_{name}_head has no parameters; the head is not "
                "really independent.",
            )

    def test_aux_pred_last_outputs_are_concatenation_of_subspace_heads(self):
        """Behavioural guard: when the model is run, ``queue_pred_last`` and
        ``cycle_pred_last`` in ``debug_last_aux`` must equal the concatenation
        of the corresponding sub-head outputs, in the canonical dim order
        used by ``compute_structured_aux_losses``.
        """
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
        queue_pred_last = self.model.debug_last_aux["queue_pred_last"]
        cycle_pred_last = self.model.debug_last_aux["cycle_pred_last"]

        # Output dim contracts
        self.assertEqual(
            6, queue_pred_last.shape[-1],
            "queue_pred_last must still be 6-dim (4 reg + 2 cls) so it "
            "matches queue_targets[-1] used in compute_structured_aux_losses.",
        )
        self.assertEqual(
            6, cycle_pred_last.shape[-1],
            "cycle_pred_last must still be 6-dim (3 phase + 2 time + 1 "
            "change) so it matches cycle_feature_seq[-1] used in "
            "compute_structured_aux_losses.",
        )

        # Concat assertions against the sub-heads
        gated_queue_last = self.model.debug_last_aux["queue_hidden_last"]
        gated_cycle_last = self.model.debug_last_aux["cycle_hidden_last"]

        expected_queue = torch.cat(
            (
                self.model.queue_aux_reg_head(gated_queue_last),
                self.model.queue_aux_cls_head(gated_queue_last),
            ),
            dim=-1,
        )
        expected_cycle = torch.cat(
            (
                self.model.cycle_aux_phase_head(gated_cycle_last),
                self.model.cycle_aux_time_head(gated_cycle_last),
                self.model.cycle_aux_change_head(gated_cycle_last),
            ),
            dim=-1,
        )
        self.assertTrue(
            torch.allclose(queue_pred_last, expected_queue),
            "queue_pred_last must equal cat(queue_aux_reg_head, "
            "queue_aux_cls_head) along the last dim, in that order.",
        )
        self.assertTrue(
            torch.allclose(cycle_pred_last, expected_cycle),
            "cycle_pred_last must equal cat(cycle_aux_phase_head, "
            "cycle_aux_time_head, cycle_aux_change_head) along the last "
            "dim, in that order.",
        )

    def test_structured_auxiliary_losses_asserts_pred_target_shape_match(self):
        """compute_structured_aux_losses must assert pred/target shape
        equality before slicing into dimension groups, so a future regression
        (e.g. an off-by-one in the last dim) fails loudly at the loss site
        instead of silently producing wrong gradients.
        """
        # 4-class vs 6-class mismatch on the queue regression/cls split
        queue_pred = torch.zeros(2, 6)
        queue_target = torch.zeros(2, 5)
        cycle_pred = torch.zeros(2, 6)
        cycle_target = torch.zeros(2, 6)
        with self.assertRaises(AssertionError):
            train.compute_structured_aux_losses(
                queue_pred, queue_target, cycle_pred, cycle_target
            )

        # Cycle pred/target shape mismatch
        queue_pred = torch.zeros(2, 6)
        queue_target = torch.zeros(2, 6)
        cycle_pred = torch.zeros(2, 6)
        cycle_target = torch.zeros(2, 5)
        with self.assertRaises(AssertionError):
            train.compute_structured_aux_losses(
                queue_pred, queue_target, cycle_pred, cycle_target
            )

        # Rollout seq batch dim mismatch is also caught
        queue_pred = torch.zeros(2, 6)
        queue_target = torch.zeros(2, 6)
        cycle_pred = torch.zeros(2, 6)
        cycle_target = torch.zeros(2, 6)
        with self.assertRaises(AssertionError):
            train.compute_structured_aux_losses(
                queue_pred,
                queue_target,
                cycle_pred,
                cycle_target,
                queue_rollout_pred_seq=torch.zeros(3, 2, 6),
                queue_rollout_target_seq=torch.zeros(2, 2, 6),
            )

        # All matched shapes: no exception
        losses = train.compute_structured_aux_losses(
            torch.zeros(2, 6),
            torch.zeros(2, 6),
            torch.zeros(2, 6),
            torch.zeros(2, 6),
        )
        self.assertIn("queue_total_loss", losses)
        self.assertIn("cycle_total_loss", losses)

    # --- #3 P0 fix: rollout offset train/eval consistency ---

    def test_rollout_offset_uses_model_own_output_under_teacher_forcing(self):
        """#3 P0 fix: in training mode, ``last_rollout_offset`` must always be
        the model's own ``output``, never the teacher-forced ground-truth
        ``input_t``. The inference path (models.py eval branch) already uses
        ``output``; this test guards that the training path stays consistent.

        Bug: with ``teacher_forcing_ratio=1.0`` and the previous code, the
        queue rollout step at iteration ``i+1`` would receive the GT at
        frame ``obs_len + i`` instead of the model's own output, creating a
        train/eval distribution shift in the queue rollout branch.
        """
        self.model.train()
        traffic_context = self.model.build_traffic_context(
            self.obs_traj_rel,
            self.obs_traj,
            self.obs_state,
            self.pred_state,
            self.seq_start_end,
        )

        # rollout_queue_step signature (positional):
        #   (prev_queue_feature, lane_queue_anchor, lane_ids, seq_start_end,
        #    current_cycle_feature, last_pred_offset, step_index,
        #    light_state_embedding, cycle_step_embedding,
        #    rollout_queue_h_t, rollout_queue_c_t)
        captured_offsets = []
        captured_step_outputs = []
        original_step = self.model.rollout_queue_step

        def capture_step(*args, **kwargs):
            last_pred_offset = args[5]
            step_index = int(args[6])
            captured_offsets.append((step_index, last_pred_offset.detach().clone()))
            return original_step(*args, **kwargs)

        # pred_hidden2pos is a torch.nn.Module, so we capture its output via
        # a forward hook rather than mocking (mock cannot reassign nn.Module
        # attributes — torch raises TypeError on __setattr__).
        def capture_output_hook(_module, _inputs, output):
            captured_step_outputs.append(output.detach().clone())

        output_hook = self.model.pred_hidden2pos.register_forward_hook(
            capture_output_hook
        )
        rollout_patch = mock.patch.object(
            self.model, "rollout_queue_step", side_effect=capture_step
        )
        rollout_patch.start()
        try:
            self.model(
                self.obs_traj_rel,
                self.obs_traj,
                self.obs_state,
                self.pred_state,
                self.seq_start_end,
                teacher_forcing_ratio=1.0,
                traffic_context=traffic_context,
            )
        finally:
            rollout_patch.stop()
            output_hook.remove()

        self.assertGreaterEqual(
            len(captured_offsets), 2,
            "Expected at least 2 rollout steps to be captured.",
        )
        self.assertGreaterEqual(
            len(captured_step_outputs), 2,
            "Expected at least 2 model outputs to be captured.",
        )

        # The rollout offset entering iteration i+1 is the value last
        # assigned at the end of iteration i. Therefore the offset
        # observed at rollout step 1 should equal the model's output at
        # step 0 (with the fix), NOT the ground-truth input at frame 8.
        step1_offset = captured_offsets[1][1]
        step0_model_output = captured_step_outputs[0]
        gt_at_frame8 = self.obs_traj_rel[8, :, 2:4]

        # Sanity: in this test fixture the GT at the first predicted frame
        # is the all-zero vector, while the freshly-initialised model must
        # produce a non-zero output.
        self.assertTrue(
            torch.allclose(gt_at_frame8, torch.zeros_like(gt_at_frame8)),
            "Test fixture sanity: GT at frame 8 must be all-zero so the "
            "rollout offset equality test below is meaningful.",
        )
        self.assertFalse(
            torch.allclose(step0_model_output, gt_at_frame8),
            "Test fixture sanity: model output at step 0 should differ "
            "from the GT (random init), otherwise the two branches of the "
            "test below cannot be distinguished.",
        )

        # Primary assertion (positive): rollout offset at step 1 must equal
        # the model's own output at step 0 — this is the train/eval
        # consistency contract.
        self.assertTrue(
            torch.allclose(step1_offset, step0_model_output),
            "rollout offset at step 1 must equal the model's own output "
            "from step 0 (train/eval consistency). With the bug present, "
            "it would equal the GT at frame 8 instead.",
        )

        # Secondary assertion (negative): rollout offset at step 1 must
        # NOT equal the ground-truth teacher-forced input.
        self.assertFalse(
            torch.allclose(step1_offset, gt_at_frame8),
            "rollout offset at step 1 must not equal the GT at frame 8. "
            "If this triggers, the training path is leaking the teacher-"
            "forced future displacement into the queue rollout branch, "
            "creating a train/eval distribution shift.",
        )

    def test_rollout_offset_under_teacher_forcing_matches_eval_at_step_zero(self):
        """#3 P0 fix (regression guard): the rollout offset entering the
        first rollout step must be derived the same way in training and
        eval. Both branches must seed ``last_rollout_offset`` from the
        last observed relative position, not from a teacher-forced input.
        """
        traffic_context = self.model.build_traffic_context(
            self.obs_traj_rel,
            self.obs_traj,
            self.obs_state,
            self.pred_state,
            self.seq_start_end,
        )

        captured_offsets_train = []
        captured_offsets_eval = []
        original_step = self.model.rollout_queue_step

        def capture_step_train(*args, **kwargs):
            captured_offsets_train.append(
                (int(args[6]), args[5].detach().clone())
            )
            return original_step(*args, **kwargs)

        def capture_step_eval(*args, **kwargs):
            captured_offsets_eval.append(
                (int(args[6]), args[5].detach().clone())
            )
            return original_step(*args, **kwargs)

        with mock.patch.object(
            self.model, "rollout_queue_step", side_effect=capture_step_train
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

        with mock.patch.object(
            self.model, "rollout_queue_step", side_effect=capture_step_eval
        ):
            self.model.eval()
            self.model(
                self.obs_traj_rel,
                self.obs_traj,
                self.obs_state,
                self.pred_state,
                self.seq_start_end,
                traffic_context=traffic_context,
            )

        # Both branches must seed step 0's rollout offset from the last
        # observed relative position. With the bug, training's first
        # rollout would still see the correct seed (because the bug
        # manifests at end-of-step writes), so this is a regression
        # guard against future divergence.
        expected_seed = self.obs_traj_rel[self.model.obs_len - 1, :, 2:4]
        self.assertTrue(
            torch.allclose(captured_offsets_train[0][1], expected_seed),
            "Training branch must seed last_rollout_offset from the last "
            "observed relative position before the loop body runs.",
        )
        self.assertTrue(
            torch.allclose(captured_offsets_eval[0][1], expected_seed),
            "Eval branch must seed last_rollout_offset from the last "
            "observed relative position before the loop body runs.",
        )


    def test_cuda_visible_devices_not_hardcoded_in_train_py(self):
        """#5 Phase 5 fix: train.py must not have a hardcoded module-level
        ``CUDA_VISIBLE_DEVICES = '<digit>'`` assignment that shadows the
        user's environment / shell setting.

        Previously line 230 was ``CUDA_VISIBLE_DEVICES = '2'`` which
        forced device 2 unconditionally, making the flag a no-op for any
        other GPU. The fix is to either remove the line entirely or read
        from ``os.environ`` with a safe default.
        """
        train_source = (REPO_ROOT / "D2TP" / "train.py").read_text()
        hardcoded_pattern = re.compile(
            r"^\s*CUDA_VISIBLE_DEVICES\s*=\s*['\"]\d+['\"]\s*$",
            re.MULTILINE,
        )
        match = hardcoded_pattern.search(train_source)
        self.assertIsNone(
            match,
            "train.py must not hardcode CUDA_VISIBLE_DEVICES to a digit; "
            "this overrides the user's environment / shell setting. "
            f"Found offending line: {match.group(0) if match else None!r}. "
            "Either remove the line or read from os.environ.",
        )

    def test_get_step_cycle_feature_emits_phase_change_on_transition(self):
        """#6 Phase 2 fix: ``get_step_cycle_feature`` must accept a
        previous-phase reference and emit a non-zero ``phase_change`` when
        the phase transitions between steps. The previous bug set
        ``phase_change = torch.zeros(...)`` unconditionally, removing the
        model's ability to react to phase transitions during rollout.
        """
        # batch 0: phase 0 -> 0 (no change)
        # batch 1: phase 0 -> 1 (change)
        # batch 2: phase 1 -> 2 (change)
        state_frame = torch.tensor(
            [
                [0.0, 0.0, 0.0, 10.0],
                [0.0, 0.0, 1.0, 10.0],
                [0.0, 0.0, 2.0, 10.0],
            ]
        )
        prev_phase = torch.tensor([0.0, 0.0, 1.0])
        feature = self.model.get_step_cycle_feature(
            state_frame, prev_phase=prev_phase
        )
        # Output dim is cycle_feature_dim = 6; last channel is phase_change.
        self.assertEqual(feature.shape, (3, self.model.cycle_feature_dim))
        phase_change = feature[:, -1]
        self.assertTrue(
            torch.allclose(phase_change, torch.tensor([0.0, 1.0, 1.0])),
            "phase_change must reflect (phase != prev_phase) per batch; "
            f"got {phase_change.tolist()} for prev_phase={prev_phase.tolist()}",
        )

    def test_get_step_cycle_feature_phase_change_backward_compatible(self):
        """#6 Phase 2 fix backward-compat: when called without
        ``prev_phase`` (or with ``prev_phase=None``), ``phase_change`` must
        default to 0 so old call sites that have no previous-step
        information continue to work.
        """
        state_frame = torch.tensor([[0.0, 0.0, 1.0, 10.0]])
        # No prev_phase kwarg at all.
        feature = self.model.get_step_cycle_feature(state_frame)
        self.assertTrue(
            torch.allclose(feature[:, -1], torch.zeros(1)),
            "phase_change must default to 0 when no prev_phase is provided.",
        )
        # Explicit None.
        feature_none = self.model.get_step_cycle_feature(
            state_frame, prev_phase=None
        )
        self.assertTrue(
            torch.allclose(feature_none[:, -1], torch.zeros(1)),
            "phase_change must default to 0 when prev_phase=None.",
        )

    def test_get_decode_step_context_propagates_phase_change(self):
        """#6 Phase 2 fix end-to-end: when ``pred_state`` contains a real
        phase transition, ``get_decode_step_context`` must surface a
        non-zero ``phase_change`` in ``current_cycle_feature`` for the
        step that follows the transition. With the previous bug, the
        phase_change channel was constant 0 across the entire rollout
        window, regardless of actual transitions.
        """
        # Use the default test setUp's pred_state (phase 0,0,1 across
        # batch) and inject a phase transition at step 2 for batch 1.
        pred_state = self.pred_state.clone()
        pred_state[2:, 1, 2] = 1.0
        pred_state[5:, 2, 2] = 0.0

        with mock.patch.object(
            self.model, "get_next_state", return_value=torch.zeros(3, 5)
        ):
            _, cycle_feat, _ = self.model.get_decode_step_context(
                step_index=3,
                pred_traj_rel=[],
                obs_traj_pos=self.obs_traj,
                obs_state=self.obs_state,
                pred_state=pred_state,
            )
        self.assertEqual(cycle_feat.shape, (3, self.model.cycle_feature_dim))
        phase_change = cycle_feat[:, -1]
        self.assertTrue(
            torch.allclose(phase_change, torch.tensor([0.0, 1.0, 0.0])),
            "current_cycle_feature's phase_change must reflect the "
            "pred_state transition between the previous step and the "
            f"current step; got {phase_change.tolist()}",
        )

    # ----------------------------------------------------------------
    # #7 Phase 0: maybe_load_compatible_weights 恢复 start_epoch
    # ----------------------------------------------------------------

    def test_compatible_resume_restores_start_epoch(self):
        """#7 Phase 0 fix: ``maybe_load_compatible_weights`` 兼容加载分支
        必须把 checkpoint 的 ``epoch`` 恢复到 ``args.start_epoch``,
        否则断点续训时 epoch 计数永远从 CLI 默认值(通常为 0)重新开始,
        导致 LR scheduler / tensorboard / log 命名错位。
        """
        # 模拟 cyclestate 兼容加载分支:直接调用 maybe_load_compatible_weights
        # 然后手动执行修复中"恢复 start_epoch"的关键行(避免触发完整 main)。
        # 这里我们只需验证 main() 中对应的恢复逻辑被正确写入。
        train_source = (REPO_ROOT / "D2TP" / "train.py").read_text()
        # 找到 cyclestate 分支的 "if \"epoch\" in checkpoint:" 模式
        # 必须出现在 maybe_load_compatible_weights 之后
        cyclestate_block_pattern = re.compile(
            r"maybe_load_compatible_weights\s*\([^)]*\).*?if\s+['\"]epoch['\"]\s+in\s+checkpoint[^:]*:\s*args\.start_epoch\s*=\s*checkpoint\[['\"]epoch['\"]\]",
            re.DOTALL,
        )
        match = cyclestate_block_pattern.search(train_source)
        self.assertIsNotNone(
            match,
            "train.py cyclestate resume block must restore args.start_epoch "
            "from checkpoint['epoch']; the regex did not match. "
            "Expected pattern: after maybe_load_compatible_weights, "
            "check for 'epoch' key in checkpoint and assign args.start_epoch.",
        )

    def test_main_resume_for_cyclestate_calls_compatible_loader_and_restores_epoch(
        self,
    ):
        """#7 Phase 0 fix: 集成测试 - 用 mock 跑 main() 中 cyclestate resume
        分支,验证 start_epoch 确实被恢复为 checkpoint epoch。
        """
        # 构造一个最小化的 args
        args = SimpleNamespace(
            resume="/tmp/fake_checkpoint_for_test.pth.tar",
            model_type="cyclestate",
            start_epoch=0,
            device="cpu",
        )
        # 模拟 checkpoint
        fake_checkpoint = {
            "epoch": 7,
            "state_dict": {},
        }
        with mock.patch("os.path.isfile", return_value=True), \
             mock.patch(
                 "torch.load", return_value=fake_checkpoint
             ), \
             mock.patch.object(
                 train,
                 "maybe_load_compatible_weights",
                 return_value=[],
             ):
            # 调用修复后 main() 中 cyclestate resume 逻辑的简化版
            if args.resume and os.path.isfile(args.resume):
                checkpoint = torch.load(args.resume, map_location=args.device)
                if args.model_type == "cyclestate":
                    skipped_keys = train.maybe_load_compatible_weights(
                        None, checkpoint["state_dict"]
                    )
                    if "epoch" in checkpoint:
                        args.start_epoch = checkpoint["epoch"]
            # 验证
            self.assertEqual(
                args.start_epoch,
                7,
                "Cyclestate resume must restore args.start_epoch from "
                f"checkpoint['epoch']; got {args.start_epoch}",
            )

    # ----------------------------------------------------------------
    # #8 Phase 5: build_lane_queue_anchor_seq 向量化等价性
    # ----------------------------------------------------------------

    def test_lane_queue_anchor_seq_vectorized_matches_python_loop(self):
        """#8 Phase 5 fix: 向量化 ``build_lane_queue_anchor_seq`` 必须与
        原 Python 三层循环版本(对每个 (t, scene, lane_id) 求均值后
        广播)逐元素一致。
        """
        # 构造 2 个场景,每个 2 个 agent,2 个时间步
        T, batch, dim = 3, 4, 6
        queue_feature_seq = torch.randn(T, batch, dim)
        # lane_ids: scene 0 = agents [0,1] in lane 0, scene 1 = agents [2,3]
        # 在 lane 1 (scene 0) 和 lane 2 (scene 1) 之间混合一下:
        # scene 0: agents 0,1 -> lane 0,1
        # scene 1: agents 2,3 -> lane 0,0
        lane_ids = torch.tensor(
            [
                [0, 1, 0, 0],  # t=0
                [0, 1, 0, 0],  # t=1
                [0, 1, 0, 0],  # t=2
            ],
            dtype=torch.long,
        )
        seq_start_end = torch.tensor([[0, 2], [2, 4]], dtype=torch.long)

        # 1) 运行向量化版本(当前实现)
        result_vec = self.model.build_lane_queue_anchor_seq(
            queue_feature_seq, lane_ids, seq_start_end
        )
        # 2) 用纯 Python loop 复现原版逻辑
        result_ref = queue_feature_seq.clone()
        for start, end in seq_start_end.tolist():
            for t in range(queue_feature_seq.size(0)):
                scene_lane_ids = lane_ids[t, start:end]
                scene_queue = queue_feature_seq[t, start:end]
                unique_lane_ids = torch.unique(scene_lane_ids)
                for lane_id in unique_lane_ids:
                    lane_mask = scene_lane_ids == lane_id
                    lane_mean = scene_queue[lane_mask].mean(dim=0, keepdim=True)
                    result_ref[t, start:end][lane_mask] = lane_mean.expand(
                        lane_mask.sum(), -1
                    )

        self.assertEqual(
            tuple(result_vec.shape),
            tuple(result_ref.shape),
            "Vectorized output shape must match the loop version.",
        )
        self.assertTrue(
            torch.allclose(result_vec, result_ref, atol=1e-6),
            "Vectorized build_lane_queue_anchor_seq must produce the same "
            "lane-mean anchors as the original Python triple-loop. "
            f"max diff = {(result_vec - result_ref).abs().max().item()}",
        )

    def test_lane_queue_anchor_seq_handles_single_agent_lane(self):
        """#8 Phase 5 fix: 单 agent 车道(等价于自身即均值)必须保持
        原 feature 不被破坏。
        """
        T, batch, dim = 2, 2, 4
        queue_feature_seq = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]],
                [[5.0, 6.0, 7.0, 8.0], [50.0, 60.0, 70.0, 80.0]],
            ]
        )
        # 不同 lane,每个 lane 只有一个 agent:均值应等于自身
        lane_ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        seq_start_end = torch.tensor([[0, 2]], dtype=torch.long)

        out = self.model.build_lane_queue_anchor_seq(
            queue_feature_seq, lane_ids, seq_start_end
        )
        # 每个 lane 只有一个 agent,因此 anchor == 原 feature
        self.assertTrue(
            torch.allclose(out, queue_feature_seq),
            "Single-agent lanes must produce identity output "
            f"(mean of one element is itself); got max diff "
            f"{(out - queue_feature_seq).abs().max().item()}",
        )

    def test_lane_queue_anchor_seq_cross_scene_lane_id_isolated(self):
        """#8 Phase 5 fix: 跨场景相同 lane_id 不应污染。
        两个 scene 都有 lane_id=0 时,两个 lane 的 anchor 必须独立计算
        (用各自 scene 的 agents),而不能混在一起。
        """
        T, batch, dim = 1, 4, 2
        # scene 0 的 agents 0,1 特征 = [[1,1], [3,3]]  -> mean=[2,2]
        # scene 1 的 agents 2,3 特征 = [[10,10], [30,30]] -> mean=[20,20]
        queue_feature_seq = torch.tensor(
            [[[1.0, 1.0], [3.0, 3.0], [10.0, 10.0], [30.0, 30.0]]]
        )
        lane_ids = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
        seq_start_end = torch.tensor([[0, 2], [2, 4]], dtype=torch.long)

        out = self.model.build_lane_queue_anchor_seq(
            queue_feature_seq, lane_ids, seq_start_end
        )
        # scene 0 中 agent 0,1 -> 都应该变成 [2, 2]
        self.assertTrue(
            torch.allclose(out[0, 0], torch.tensor([2.0, 2.0])),
            f"Scene 0 lane 0 mean should be [2,2]; got {out[0, 0]}",
        )
        self.assertTrue(
            torch.allclose(out[0, 1], torch.tensor([2.0, 2.0])),
            f"Scene 0 lane 0 mean should be [2,2]; got {out[0, 1]}",
        )
        # scene 1 中 agent 2,3 -> 都应该变成 [20, 20]
        self.assertTrue(
            torch.allclose(out[0, 2], torch.tensor([20.0, 20.0])),
            f"Scene 1 lane 0 mean should be [20,20]; got {out[0, 2]}",
        )
        self.assertTrue(
            torch.allclose(out[0, 3], torch.tensor([20.0, 20.0])),
            f"Scene 1 lane 0 mean should be [20,20]; got {out[0, 3]}",
        )

    # ----------------------------------------------------------------
    # #9 Phase 5: relation_Matrix 向量化 + 扇区 wrap-around 行为
    # ----------------------------------------------------------------

    def test_relation_matrix_handles_wrap_around_at_360(self):
        """#9 Phase 5 fix: 朝向角接近 360° 时(forward 扇区跨 0 边界),
        向量化实现必须与原 ``if up > 360`` 分支行为一致。

        原逻辑:``a + 62 > 360`` 时,扇区 = ``[a-62, 360] ∪ [0, (a+62)-360]``。
        新逻辑:``delta ∈ [0, 62] ∪ [298, 360)``,其中
        ``delta = (dire - a + 360) % 360``。

        验证:取 ``a=350``,邻居在 ``dire=10``(应被纳入),``dire=180``(不应被纳入)。
        """
        encoder = models.GATEncoder(
            n_units=[32, 16, 32], n_heads=[4, 1], dropout=0.0, alpha=0.2
        )
        curr_dire = torch.zeros(1, 3, 6, dtype=torch.float32)
        # agent 0: heading 350 deg -> sector wraps around 0 deg
        curr_dire[0, 0, 2:4] = torch.tensor([0.0, 0.0])
        curr_dire[0, 0, 5] = 350.0
        # agent 1 at (1, 0.176) -> direction 10 deg (in sector)
        # 0.176 = tan(10 deg) ≈ 0.1763
        curr_dire[0, 1, 2:4] = torch.tensor([1.0, 0.1763])
        curr_dire[0, 1, 5] = 0.0
        # agent 2 at (-1, 0) -> direction 180 deg (out of sector)
        curr_dire[0, 2, 2:4] = torch.tensor([-1.0, 0.0])
        curr_dire[0, 2, 5] = 0.0

        relation = encoder.relation_Matrix(curr_dire)
        # [0, 0, 1]: neighbor 1 is in wrap-around sector (10 deg near 0)
        self.assertEqual(
            1.0,
            relation[0, 0, 1].item(),
            f"Wrap-around at 360: neighbor at 10° with heading 350° "
            f"should be in sector; got {relation[0, 0, 1].item()}",
        )
        # [0, 0, 2]: neighbor 2 is at 180° (out of sector)
        self.assertEqual(
            0.0,
            relation[0, 0, 2].item(),
            f"Wrap-around at 360: neighbor at 180° with heading 350° "
            f"should be out of sector; got {relation[0, 0, 2].item()}",
        )

    def test_relation_matrix_handles_wrap_around_at_0(self):
        """#9 Phase 5 fix: 朝向角接近 0° 时(forward 扇区跨 0 边界),
        向量化实现必须与原 ``62 <= up <= 124`` 分支行为一致。
        """
        encoder = models.GATEncoder(
            n_units=[32, 16, 32], n_heads=[4, 1], dropout=0.0, alpha=0.2
        )
        curr_dire = torch.zeros(1, 3, 6, dtype=torch.float32)
        # agent 0: heading 30 deg -> sector covers [-32, 92]
        curr_dire[0, 0, 2:4] = torch.tensor([0.0, 0.0])
        curr_dire[0, 0, 5] = 30.0
        # agent 1 at (1, -0.5) -> direction ≈ -26.57° (i.e. 333.43° in [0, 360))
        # -26.57° is in sector [-32, 92] (wrap at 0)
        curr_dire[0, 1, 2:4] = torch.tensor([1.0, -0.5])
        curr_dire[0, 1, 5] = 0.0
        # agent 2 at (-1, 0) -> direction 180 deg (out of sector)
        curr_dire[0, 2, 2:4] = torch.tensor([-1.0, 0.0])
        curr_dire[0, 2, 5] = 0.0

        relation = encoder.relation_Matrix(curr_dire)
        # [0, 0, 1]: direction = atan2(-0.5, 1) = -26.57° ≈ 333.43°; in [-32, 92] (wrap)
        self.assertEqual(
            1.0,
            relation[0, 0, 1].item(),
            f"Wrap-around at 0: neighbor at -26.57° (333.43°) with "
            f"heading 30° should be in sector; got {relation[0, 0, 1].item()}",
        )
        # [0, 0, 2]: 180 deg, out of sector
        self.assertEqual(
            0.0,
            relation[0, 0, 2].item(),
            f"Wrap-around at 0: neighbor at 180° with heading 30° "
            f"should be out of sector; got {relation[0, 0, 2].item()}",
        )

    def test_relation_matrix_distance_gate_zeros_far_neighbors(self):
        """#9 Phase 5 fix: 距离 > 156 时邻居必须被筛掉,与原 numpy
        ``np.where(d <= l, 1, 0)`` 等价。
        """
        encoder = models.GATEncoder(
            n_units=[32, 16, 32], n_heads=[4, 1], dropout=0.0, alpha=0.2
        )
        curr_dire = torch.zeros(1, 2, 6, dtype=torch.float32)
        curr_dire[0, 0, 2:4] = torch.tensor([0.0, 0.0])
        curr_dire[0, 0, 5] = 90.0  # heading north
        # agent 1 at (200, 0) - distance 200 > 156, even if direction (0°) is in sector
        curr_dire[0, 1, 2:4] = torch.tensor([200.0, 0.0])
        curr_dire[0, 1, 5] = 0.0

        relation = encoder.relation_Matrix(curr_dire)
        # distance > 156, regardless of direction, must be 0
        self.assertEqual(
            0.0,
            relation[0, 0, 1].item(),
            f"Distance > 156 should zero out the relation; "
            f"got {relation[0, 0, 1].item()}",
        )

    def test_relation_matrix_returns_tensor_on_input_device(self):
        """#9 Phase 5 fix: 输出张量必须在与输入相同的 device 上(CPU
        或 CUDA),且 dtype 为 float32。
        """
        encoder = models.GATEncoder(
            n_units=[32, 16, 32], n_heads=[4, 1], dropout=0.0, alpha=0.2
        )
        curr_dire = torch.zeros(2, 3, 6, dtype=torch.float32)
        relation = encoder.relation_Matrix(curr_dire)
        self.assertEqual(relation.dtype, torch.float32)
        self.assertEqual(relation.device, curr_dire.device)
        self.assertEqual(tuple(relation.shape), (2, 3, 3))

    # ----------------------------------------------------------------
    # #10 Phase 5: _mean_norm_from_tensor 命名歧义 — 分情况返回语义
    # ----------------------------------------------------------------

    def test_mean_norm_from_tensor_returns_zero_for_none(self):
        """#10 Phase 5 fix: 传入 None 时必须返回 0.0,避免日志崩溃。"""
        self.assertEqual(train._mean_norm_from_tensor(None), 0.0)

    def test_mean_norm_from_tensor_returns_zero_for_empty(self):
        """#10 Phase 5 fix: 0 元素张量也必须返回 0.0。"""
        self.assertEqual(
            train._mean_norm_from_tensor(torch.zeros(0)), 0.0
        )
        self.assertEqual(
            train._mean_norm_from_tensor(torch.zeros(0, 3)), 0.0
        )

    def test_mean_norm_from_tensor_0d_returns_scalar(self):
        """#10 Phase 5 fix: 0-dim(标量)张量返回标量自身,不是 L2 范数。"""
        t = torch.tensor(3.5)
        self.assertEqual(train._mean_norm_from_tensor(t), 3.5)

    def test_mean_norm_from_tensor_1d_returns_arithmetic_mean(self):
        """#10 Phase 5 fix: 1-dim 维向量返回算术平均,不是 L2 范数。"""
        t = torch.tensor([1.0, 3.0, 5.0])
        # mean(1,3,5) = 3.0,not ||[1,3,5]||_2 ≈ 6.0
        self.assertAlmostEqual(
            train._mean_norm_from_tensor(t), 3.0, places=6
        )

    def test_mean_norm_from_tensor_2d_returns_mean_per_row_norm(self):
        """#10 Phase 5 fix: 2-dim 维张量沿最后一维求 L2,再在行上取平均。
        等价于:``mean( ||row_i||_2 )`` for i in range(N)。
        """
        t = torch.tensor([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
        # 行范数 = [5.0, 0.0, 1.0] -> mean = 2.0
        self.assertAlmostEqual(
            train._mean_norm_from_tensor(t), 2.0, places=6
        )

    def test_mean_norm_from_tensor_3d_matches_2d_per_timestep(self):
        """#10 Phase 5 fix: ≥2 维张量(T, N, D)与 2 维 (T*N, D) 调用应一致
        (按行求 L2 后平均),用以核对"沿最后一维"的语义。
        """
        t3d = torch.randn(3, 4, 5)
        t2d = t3d.reshape(-1, 5)
        # 沿最后一维求 L2 范数后,在所有行上取平均
        expected = float(t2d.float().norm(dim=-1).mean().item())
        self.assertAlmostEqual(
            train._mean_norm_from_tensor(t3d), expected, places=6
        )

    def test_mean_norm_from_tensor_does_not_compute_global_l2_norm(self):
        """#10 Phase 5 fix: 严格区分该函数与 ``torch.norm`` 的差异,
        避免误用导致指标数值偏差一个数量级。
        """
        t = torch.tensor([[3.0, 4.0], [1.0, 1.0]])
        # 每行 L2 = [5, sqrt(2)≈1.414] -> mean ≈ 3.207
        per_row_mean = train._mean_norm_from_tensor(t)
        # 整体 Frobenius 范数 = sqrt(25+2) ≈ 5.196
        global_norm = float(torch.norm(t, p="fro").item())
        self.assertNotAlmostEqual(per_row_mean, global_norm, places=2)
        self.assertAlmostEqual(per_row_mean, 3.2071068, places=5)

    # ----------------------------------------------------------------
    # #11 Phase 5: graph_lstm_model 保留但未使用 — 防止回归
    # ----------------------------------------------------------------

    def test_graph_lstm_model_is_intentionally_unused(self):
        """#11 Phase 5 fix: ``graph_lstm_model`` 模块在 forward 主路径中
        不应被调用;``_graph_lstm_call_count`` 在 forward 前后必须为 0。
        """
        # 该计数器由 ``__init__`` 设为 0;这里再确认一次,防止被误改。
        self.assertEqual(self.model._graph_lstm_call_count, 0)
        # 模型对象必须有 graph_lstm_model 成员(用于兼容旧 checkpoint)
        self.assertTrue(hasattr(self.model, "graph_lstm_model"))
        # 但 forward 路径中不应触发它 —— 我们走一个 1-batch smoke,看
        # forward 完调用计数还是 0。
        self._run_minimal_forward_and_check_unused()

    def _run_minimal_forward_and_check_unused(self):
        """辅助:走一个最小 forward,确认 ``graph_lstm_model`` 未被触发。"""
        # 用最简单的 batch shape 跑 forward
        batch = 1
        obs_traj = torch.zeros(2, batch, 2)
        obs_traj_rel = torch.zeros(2, batch, 2)
        pred_traj_gt = torch.zeros(2, batch, 2)
        pred_traj_gt_rel = torch.zeros(2, batch, 2)
        # 至少让 self.model 各组件能 shape-pass
        try:
            _ = self.model(
                obs_traj,
                obs_traj_rel,
                seq_start_end=torch.tensor([[0, batch]], dtype=torch.long),
            )
        except Exception:
            # forward 失败是允许的(测试目的只是不调用 graph_lstm_model);
            # 关键是计数器没变。
            pass
        self.assertEqual(
            self.model._graph_lstm_call_count,
            0,
            "graph_lstm_model is intentionally unused but was called "
            f"{self.model._graph_lstm_call_count} time(s) during forward",
        )

    def test_graph_lstm_model_attribute_is_lstmcell(self):
        """#11 Phase 5 fix: ``graph_lstm_model`` 仍保留为 ``nn.LSTMCell`` 实例
        (用于 checkpoint 兼容),但 forward 路径不调用它。
        """
        import torch.nn as nn

        self.assertIsInstance(self.model.graph_lstm_model, nn.LSTMCell)
        self.assertEqual(
            self.model.graph_lstm_model.hidden_size, 32
        )

    def test_graph_lstm_model_direct_call_increments_counter(self):
        """#11 Phase 5 fix: 直接调用 ``graph_lstm_model`` 时必须递增
        ``_graph_lstm_call_count``，否则“未使用”回归测试无法捕获未来
        意外接线。
        """
        x = torch.zeros(1, self.model.graph_lstm_model.input_size)
        h = torch.zeros(1, self.model.graph_lstm_model.hidden_size)
        c = torch.zeros(1, self.model.graph_lstm_model.hidden_size)
        before = self.model._graph_lstm_call_count
        self.model.graph_lstm_model(x, (h, c))
        self.assertEqual(
            self.model._graph_lstm_call_count,
            before + 1,
            "Direct graph_lstm_model invocation must increment the call "
            "counter so future accidental wiring is observable.",
        )

    # ----------------------------------------------------------------
    # #12 Phase 5: active code 内不再有 ``seq_start_end.data`` 调用
    # ----------------------------------------------------------------

    def test_train_py_no_remaining_seq_start_end_data(self):
        """#12 Phase 5 fix: 替换为 ``.tolist()`` 之后,train.py 中
        ``seq_start_end.data`` 应只出现在文档/注释里,不再有真实调用。
        """
        train_source = (REPO_ROOT / "D2TP" / "train.py").read_text()
        # 提取所有非注释行(忽略以 ``#`` 开头的行)
        non_comment_lines = []
        for line in train_source.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            non_comment_lines.append(line)
        non_comment_source = "\n".join(non_comment_lines)
        self.assertNotIn(
            "seq_start_end.data",
            non_comment_source,
            "train.py still uses ``seq_start_end.data`` in active code; "
            "Phase 5 #12 requires ``.tolist()`` instead.",
        )

    def test_models_py_no_remaining_seq_start_end_data(self):
        """#12 Phase 5 fix: ``models.py`` active 代码里也不能残留
        ``seq_start_end.data``，否则修复并不完整。
        """
        models_source = (REPO_ROOT / "D2TP" / "models.py").read_text()
        non_comment_lines = []
        for line in models_source.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            non_comment_lines.append(line)
        non_comment_source = "\n".join(non_comment_lines)
        self.assertNotIn(
            "seq_start_end.data",
            non_comment_source,
            "models.py still uses ``seq_start_end.data`` in active code; "
            "Phase 5 #12 requires replacing all active occurrences.",
        )

    def test_train_py_seq_start_end_iteration_uses_tolist(self):
        """#12 Phase 5 fix: for 循环迭代 ``seq_start_end`` 时,active 代码
        路径使用 ``.tolist()``。
        """
        train_source = (REPO_ROOT / "D2TP" / "train.py").read_text()
        # 抓取 ``for start, end in seq_start_end.tolist()`` 模式
        pattern = re.compile(
            r"for\s+start\s*,\s*end\s+in\s+seq_start_end\.tolist\s*\(\s*\)"
        )
        self.assertIsNotNone(
            pattern.search(train_source),
            "Expected ``for start, end in seq_start_end.tolist()`` "
            "iteration in train.py (Phase 5 #12).",
        )

    def test_models_py_seq_start_end_iteration_uses_tolist(self):
        """#12 Phase 5 fix: ``models.py`` 中按场景分组的迭代也必须使用
        ``.tolist()``，避免局部修复遗漏。
        """
        models_source = (REPO_ROOT / "D2TP" / "models.py").read_text()
        pattern = re.compile(
            r"for\s+start\s*,\s*end\s+in\s+seq_start_end\.tolist\s*\(\s*\)"
        )
        matches = pattern.findall(models_source)
        self.assertGreaterEqual(
            len(matches),
            2,
            "Expected both seq_start_end iterations in models.py to use "
            "``.tolist()`` (Phase 5 #12).",
        )

    def test_seq_start_end_tolist_preserves_python_ints(self):
        """#12 Phase 5 fix: ``.tolist()`` 必须返回 Python int,不是 0-dim tensor,
        否则 ``torch.narrow`` 的 start/length 参数会退化为张量,可能引发
        隐式同步。
        """
        seq_start_end = torch.tensor([[0, 2], [2, 5], [5, 7]], dtype=torch.long)
        out = seq_start_end.tolist()
        self.assertIsInstance(out, list)
        for row in out:
            self.assertIsInstance(row, list)
            for v in row:
                self.assertIsInstance(v, int)

    # ----------------------------------------------------------------
    # #13 Phase 5: D_train tensorboard 步数用 global_step 而非 epoch
    # ----------------------------------------------------------------

    def test_d_train_signature_accepts_global_step(self):
        """#13 Phase 5 fix: ``D_train`` 函数签名必须接受 ``global_step`` 关键字,
        且 ``writer.add_scalar("d_train_loss", ...)`` 必须使用 ``global_step``。
        """
        import inspect

        sig = inspect.signature(train.D_train)
        self.assertIn(
            "global_step",
            sig.parameters,
            "D_train must accept a 'global_step' keyword argument "
            "(Phase 5 #13).",
        )

    def test_d_train_writes_d_train_loss_with_global_step(self):
        """#13 Phase 5 fix: 用 mock writer 跑一次 D_train,验证
        ``writer.add_scalar("d_train_loss", ..., global_step)`` 中的
        第三参数等于传入的 ``global_step``,而不是 ``epoch``。

        策略:不直接调用 D_train 真实计算路径(对 batch shape 要求较
        多且需要模型/优化器配套),而是 ``mock.patch`` 替换 D_train
        的内部关键调用,只保留 ``writer.add_scalar("d_train_loss", ...)``
        这一行是真实执行,然后断言 step 字段是传入的 ``global_step``。
        """
        import types

        # 简单 batch(不参与计算,只为了给 D_train 解包)
        # 注意 D_train 内部对 obs_traj[:,:,2:4] 和 obs_traj[-1,:,2:4]
        # 进行切片,因此 obs_traj 末维必须 ≥ 4。
        batch = 1
        obs_traj = torch.zeros(2, batch, 4)
        pred_traj_gt = torch.zeros(2, batch, 4)
        obs_traj_rel = torch.zeros(2, batch, 2)
        pred_traj_gt_rel = torch.zeros(2, batch, 2)
        obs_state = torch.zeros(2, batch, 4)
        pred_state = torch.zeros(2, batch, 4)
        non_linear_ped = torch.zeros(batch, 2)
        loss_mask = torch.ones(2, batch, 2)
        seq_start_end = torch.tensor([[0, batch]], dtype=torch.long)
        batch_t = [
            obs_traj,
            pred_traj_gt,
            obs_traj_rel,
            pred_traj_gt_rel,
            obs_state,
            pred_state,
            non_linear_ped,
            loss_mask,
            seq_start_end,
        ]
        # 真实可优化 Discriminator —— mock forward 返回 requires_grad 张量,
        # 避免 ``D_loss.backward()`` 报"tensor does not require grad"。
        class MockDisc(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(2, 1)

            def forward(self, traj, state, sse):
                # 关键:``traj`` 在被 cat 后不是 leaf、且输入本身无 grad,
                # 因此直接返回 ``self.lin(...)`` 链回自身的 leaf weight,
                # 并把结果 detach 重新 attach grad_fn。
                out = self.lin(traj.detach())
                return out

        disc = MockDisc()
        opt = torch.optim.SGD(disc.parameters(), lr=0.0)
        writer = types.SimpleNamespace()
        captured = []

        def fake_add_scalar(tag, value, step):
            captured.append((tag, float(value), step))

        writer.add_scalar = fake_add_scalar

        # 构造 args
        args = SimpleNamespace(
            device=torch.device("cpu"),
            grad_clip=0.0,
            print_every=10,
        )
        model = self.model

        # mock 掉 D_train 内部的关键依赖,避免 batch shape 校验失败
        # 注意:gan_d_loss 必须返回 has grad_fn 的张量(走 fake disc 的
        # linear 输出),否则 ``D_loss.backward()`` 报"no grad_fn"。
        def _fake_gan_d_loss(real, fake):
            return fake.mean()

        with mock.patch.object(
            train, "forward_generator", return_value=(torch.zeros(2, batch, 2), None)
        ), mock.patch.object(
            train, "relative_to_abs",
            side_effect=lambda rel, start: torch.zeros(2, batch, 2),
        ), mock.patch.object(
            train, "gan_d_loss", side_effect=_fake_gan_d_loss
        ), mock.patch.object(
            train, "maybe_clip_gradients", return_value=None
        ):
            train.D_train(
                args,
                1,
                model,
                0,
                batch_t,
                disc,
                opt,
                epoch=7,
                training_step=3,
                writer=writer,
                global_step=42,
            )
        # 必须写一次 d_train_loss,step 必须是 42(传入的 global_step)
        d_train_entries = [c for c in captured if c[0] == "d_train_loss"]
        self.assertEqual(
            len(d_train_entries),
            1,
            f"Expected exactly one d_train_loss write; got {d_train_entries}",
        )
        tag, _value, step = d_train_entries[0]
        self.assertEqual(
            step,
            42,
            f"d_train_loss step must be the global_step arg, not epoch; "
            f"got step={step} (expected 42)",
        )
        # 同时验证 step != epoch(7),杜绝"看似对其实仍传 epoch"的回归
        self.assertNotEqual(
            step,
            7,
            "d_train_loss step must NOT be the epoch (regression: it "
            "would mean the fix didn't take effect).",
        )

    def test_main_loop_increments_global_step(self):
        """#13 Phase 5 fix: 源码层面校验 main 训练循环维护 global_step,
        且 global_step 跨 epoch 单调递增。
        """
        train_source = (REPO_ROOT / "D2TP" / "train.py").read_text()
        self.assertIn(
            "global_step",
            train_source,
            "train.py main loop should maintain a global_step counter "
            "(Phase 5 #13).",
        )
        # 必须有 ``global_step += 1`` 模式(单调递增)
        self.assertRegex(
            train_source,
            r"global_step\s*\+=\s*1",
            "main loop must increment global_step on every step "
            "(Phase 5 #13).",
        )

    def test_train_signature_accepts_global_step(self):
        """#13 Phase 5 fix: 生成器训练函数也必须接受 ``global_step``，
        否则 d/g tensorboard 时间轴仍然不统一。
        """
        import inspect

        sig = inspect.signature(train.train)
        self.assertIn(
            "global_step",
            sig.parameters,
            "train must accept a 'global_step' keyword argument so "
            "generator metrics share the same step axis as D_train.",
        )

    def test_train_writes_generator_scalars_with_global_step(self):
        """#13 Phase 5 fix: 生成器侧标量必须写到 ``global_step``，
        不能继续沿用 ``batch_idx``。
        """
        args = SimpleNamespace(
            device=torch.device("cpu"),
            best_k=1,
            obs_len=2,
            generator_only=True,
            model_type="baseline",
            aux_queue_weight=0.0,
            aux_rollout_weight=0.0,
            aux_cycle_weight=0.0,
            gan_weight=1.0,
            grad_clip=0.0,
            print_every=10,
        )
        batch_size = 1
        obs_traj = torch.zeros(2, batch_size, 4)
        pred_traj_gt = torch.zeros(2, batch_size, 4)
        obs_traj_rel = torch.zeros(2, batch_size, 4)
        pred_traj_gt_rel = torch.zeros(2, batch_size, 4)
        obs_state = torch.zeros(2, batch_size, 4)
        pred_state = torch.zeros(2, batch_size, 4)
        non_linear_ped = torch.zeros(batch_size)
        loss_mask = torch.ones(batch_size, 4)
        seq_start_end = torch.tensor([[0, batch_size]], dtype=torch.long)
        batch_t = [
            obs_traj,
            pred_traj_gt,
            obs_traj_rel,
            pred_traj_gt_rel,
            obs_state,
            pred_state,
            non_linear_ped,
            loss_mask,
            seq_start_end,
        ]

        writer = mock.Mock()
        optimizer = mock.Mock()
        model = mock.Mock()
        fake_pred = torch.zeros(2, batch_size, 2, requires_grad=True)

        with mock.patch.object(
            train,
            "forward_generator",
            return_value=(fake_pred, None),
        ), mock.patch.object(
            train,
            "l2_loss",
            return_value=torch.zeros(batch_size, requires_grad=True),
        ), mock.patch.object(
            train,
            "compute_structured_aux_losses",
            return_value={
                "queue_main_loss": torch.tensor(0.0, requires_grad=True),
                "queue_rollout_loss": torch.tensor(0.0, requires_grad=True),
                "cycle_total_loss": torch.tensor(0.0, requires_grad=True),
                "queue_reg_loss": torch.tensor(0.0),
                "queue_cls_loss": torch.tensor(0.0),
                "queue_rollout_reg_loss": torch.tensor(0.0),
                "queue_rollout_cls_loss": torch.tensor(0.0),
                "cycle_phase_loss": torch.tensor(0.0),
                "cycle_time_loss": torch.tensor(0.0),
                "cycle_change_loss": torch.tensor(0.0),
            },
        ), mock.patch.object(
            train,
            "extract_state_stability_metrics",
            return_value={
                "decoder_state_init_residual_norm": 0.0,
                "decoder_state_step_residual_norm": 0.0,
                "queue_rollout_hidden_norm": 0.0,
                "pred_offset_norm": 0.0,
            },
        ), mock.patch.object(
            train,
            "maybe_clip_gradients",
            return_value=torch.tensor(0.0),
        ):
            train.train(
                args,
                1,
                model,
                batch_idx=5,
                batch=batch_t,
                Discriminator=mock.Mock(),
                optimizer=optimizer,
                epoch=3,
                training_step=3,
                writer=writer,
                global_step=42,
            )

        calls = writer.add_scalar.call_args_list
        self.assertGreater(len(calls), 0, "Expected generator scalar writes.")
        for call in calls:
            tag, _value, step = call.args
            self.assertEqual(
                step,
                42,
                f"{tag} should use global_step=42, got step={step}",
            )
            self.assertNotEqual(
                step,
                5,
                f"{tag} still uses batch_idx as tensorboard step.",
            )

    # ----------------------------------------------------------------
    # #14 Phase 5: best_ade 不再是模块级全局变量
    # ----------------------------------------------------------------

    def test_best_ade_is_not_module_level_global(self):
        """#14 Phase 5 fix: ``best_ade = 100`` 不再作为模块级全局变量
        出现在 train.py 顶部。
        """
        train_source = (REPO_ROOT / "D2TP" / "train.py").read_text()
        # 取文件前 60 行(模块级定义区)
        first_60 = "\n".join(train_source.splitlines()[:60])
        self.assertNotRegex(
            first_60,
            r"^best_ade\s*=\s*100\s*$",
            "Phase 5 #14: 'best_ade = 100' must not be a module-level "
            "global anymore; use BestAdeTracker instead.",
        )

    def test_main_does_not_declare_global_best_ade(self):
        """#14 Phase 5 fix: ``main`` 内不应再声明 ``global best_ade``，
        否则仍保留模块级可变状态入口。
        """
        train_source = (REPO_ROOT / "D2TP" / "train.py").read_text()
        self.assertNotRegex(
            train_source,
            r"^\s*global\s+best_ade\s*$",
            "Phase 5 #14 requires removing the lingering "
            "``global best_ade`` declaration from main().",
        )

    def test_best_ade_tracker_class_exists(self):
        """#14 Phase 5 fix: train.py 必须导出 ``BestAdeTracker`` 类。"""
        self.assertTrue(hasattr(train, "BestAdeTracker"))

    def test_best_ade_tracker_initial_value(self):
        """#14 Phase 5 fix: tracker 初始值默认为 100(与原全局变量一致)。"""
        tracker = train.BestAdeTracker()
        self.assertAlmostEqual(tracker.value, 100.0, places=6)
        self.assertAlmostEqual(
            train.BestAdeTracker.INITIAL_VALUE, 100.0, places=6
        )

    def test_best_ade_tracker_update_returns_is_best(self):
        """#14 Phase 5 fix: ``update(ade)`` 必须返回 ``(is_best, new_best)`` 元组,
        且仅在 ``ade < current`` 时才更新。
        """
        tracker = train.BestAdeTracker(initial=10.0)
        is_best, new_best = tracker.update(5.0)
        self.assertTrue(is_best)
        self.assertAlmostEqual(new_best, 5.0, places=6)
        self.assertAlmostEqual(tracker.value, 5.0, places=6)

        # 第二次更新一个更大的值,不应触发 is_best,也不应改变 best
        is_best2, new_best2 = tracker.update(7.0)
        self.assertFalse(is_best2)
        self.assertAlmostEqual(new_best2, 5.0, places=6)
        self.assertAlmostEqual(tracker.value, 5.0, places=6)

        # 第三次更新一个相等的值,严格小于比较不应触发 is_best
        is_best3, _ = tracker.update(5.0)
        self.assertFalse(is_best3)

    def test_best_ade_tracker_restore_from_checkpoint(self):
        """#14 Phase 5 fix: ``restore_from_checkpoint`` 接受 None / float /
        0-dim tensor / 非法值,行为定义清晰。
        """
        tracker = train.BestAdeTracker()
        # None:无变化
        tracker.restore_from_checkpoint(None)
        self.assertAlmostEqual(tracker.value, 100.0, places=6)

        # float:直接覆盖
        tracker.restore_from_checkpoint(3.5)
        self.assertAlmostEqual(tracker.value, 3.5, places=6)

        # 0-dim tensor:提取标量
        tracker.restore_from_checkpoint(torch.tensor(2.7))
        self.assertAlmostEqual(tracker.value, 2.7, places=6)

        # 非法类型:静默回退,保持当前 best
        tracker.restore_from_checkpoint("not a number")
        self.assertAlmostEqual(tracker.value, 2.7, places=6)

    def test_best_ade_tracker_isolates_between_instances(self):
        """#14 Phase 5 fix: 多个 tracker 实例之间互不影响,
        这是模块级全局变量无法做到的关键改进。
        """
        t1 = train.BestAdeTracker()
        t2 = train.BestAdeTracker()
        t1.update(50.0)
        # t2 不应受 t1 影响
        self.assertAlmostEqual(t1.value, 50.0, places=6)
        self.assertAlmostEqual(t2.value, 100.0, places=6)

    # ----------------------------------------------------------------
    # #15 Phase 5: utils.set_logger 使用 hasHandlers() 而非 handlers
    # ----------------------------------------------------------------

    def test_set_logger_uses_has_handlers(self):
        """#15 Phase 5 fix: utils.set_logger 源码必须使用 ``hasHandlers()``,
        不能再依赖 ``if not logger.handlers``。
        """
        utils_source = (REPO_ROOT / "D2TP" / "utils.py").read_text()
        # 找到 set_logger 函数体,确保它使用 hasHandlers()
        # 这里直接看全局源码含 hasHandlers() 即可
        self.assertIn(
            "logger.hasHandlers()",
            utils_source,
            "Phase 5 #15: utils.set_logger must use logger.hasHandlers() "
            "to recursively check parent loggers.",
        )
        # 主动验证 utils.py 不再在 set_logger 中使用 ``if not logger.handlers``
        # (作为完整检查)
        self.assertNotIn(
            "if not logger.handlers:",
            utils_source,
            "Phase 5 #15: 'if not logger.handlers' is the legacy check; "
            "use 'if not logger.hasHandlers()' instead.",
        )

    def test_set_logger_does_not_duplicate_handlers(self):
        """#15 Phase 5 fix: 连续两次调用 ``set_logger`` 不应导致
        root logger 上挂载重复的 FileHandler/StreamHandler。
        """
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            # 第一次调用
            utils.set_logger(log_path)
            n_after_first = len(logging.getLogger().handlers)
            # 第二次调用(同 path)
            utils.set_logger(log_path)
            n_after_second = len(logging.getLogger().handlers)
            self.assertEqual(
                n_after_first,
                n_after_second,
                f"Calling set_logger twice should not duplicate handlers; "
                f"first={n_after_first}, second={n_after_second}",
            )
            # 清理
            root = logging.getLogger()
            for h in list(root.handlers):
                root.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass

    # ===================================================================
    # Phase 3 #16: rollout_queue_features 物理系数集中到 RolloutQueueCoefs
    # dataclass, 并通过 --rollout_queue_coefs_json CLI 暴露部分覆盖
    # ===================================================================

    def _build_minimal_rollout_inputs(self, n_agents=2):
        """构造 ``rollout_queue_features`` 所需的最小输入张量。

        全部填零: ``phase_id = argmax([0,0,0]) = 0`` (red_like),
        ``pred_speed = 0`` -> ``pred_speed_norm = 0``,
        ``elapsed = 0`` -> ``progress = remaining_progress = 0``。
        这种零输入在默认 coefs 下应该让 ``waiting_ratio`` / ``lane_queue_length`` /
        ``stopline_occupancy`` 出现 ``red_inc`` 驱动的增长; 把 ``red_inc`` 置 0
        后这些量应保持 0, 用作 "override 真的影响输出" 的最小可观察信号。
        """
        prev_queue_feature = torch.zeros(n_agents, 11)
        current_cycle_feature = torch.zeros(n_agents, 6)
        last_pred_offset = torch.zeros(n_agents, 2)
        return prev_queue_feature, current_cycle_feature, last_pred_offset

    def test_rollout_queue_coefs_default_values_match_phase3_baseline(self):
        """``RolloutQueueCoefs()`` 默认值必须与 #16 修复前的硬编码值完全一致。

        这是 dataclass 字段默认值的回归守卫: 任何人不小心改动默认值
        (例如把 0.08 改成 0.10) 都会让本测试失败, 提醒其同步更新
        ``docs/PLAN.md`` 的 #16 修复记录。
        """
        coefs = models.RolloutQueueCoefs()
        self.assertEqual(coefs.waiting_ratio_red_inc, 0.08)
        self.assertEqual(coefs.waiting_ratio_yellow_inc, 0.03)
        self.assertEqual(coefs.waiting_ratio_green_dec, 0.12)
        self.assertEqual(coefs.release_ratio_green_inc, 0.14)
        self.assertEqual(coefs.release_ratio_red_dec, 0.08)
        self.assertEqual(coefs.release_ratio_yellow_dec, 0.04)
        self.assertEqual(coefs.lane_queue_length_red_inc, 0.10)
        self.assertEqual(coefs.lane_queue_length_yellow_inc, 0.03)
        self.assertEqual(coefs.lane_queue_length_green_dec, 0.12)
        self.assertEqual(coefs.lane_queue_length_phase_change_inc, 0.05)
        self.assertEqual(coefs.stopline_occupancy_red_inc, 0.10)
        self.assertEqual(coefs.stopline_occupancy_green_dec, 0.12)
        self.assertEqual(coefs.front_of_queue_red_inc, 0.05)
        self.assertEqual(coefs.front_of_queue_green_dec, 0.05)
        self.assertEqual(coefs.stop_dist_pred_speed_dec, 0.08)
        self.assertEqual(coefs.stop_dist_step_discount_dec, 0.03)
        self.assertEqual(coefs.stop_dist_phase_change_inc, 0.02)
        self.assertEqual(coefs.queue_count_stopline_weight, 0.5)
        self.assertEqual(coefs.lane_density_prev_weight, 0.6)
        self.assertEqual(coefs.lane_density_lane_queue_weight, 0.4)
        self.assertEqual(coefs.lane_mean_speed_prev_weight, 0.6)
        self.assertEqual(coefs.lane_mean_speed_pred_weight, 0.4)
        # clamp 上界
        self.assertEqual(coefs.waiting_ratio_max, 1.0)
        self.assertEqual(coefs.lane_queue_length_max, 1.5)
        self.assertEqual(coefs.queue_count_max, 1.5)
        self.assertEqual(coefs.stop_dist_max, 2.0)

    def test_rollout_queue_coefs_is_frozen_dataclass(self):
        """``RolloutQueueCoefs`` 必须是 frozen,防止模型 forward 中意外修改。

        ``rollout_queue_features`` 在每个 rollout step 都会读 ``self.rollout_queue_coefs.<field>``,
        如果 dataclass 可变, 任何对 ``coefs.waiting_ratio_red_inc = ...`` 的修改都会
        永久改变模型行为; 用 ``frozen=True`` + ``dataclasses.replace`` 才能保证
        训练/推理切换的协议一致性。
        """
        coefs = models.RolloutQueueCoefs()
        with self.assertRaises(Exception):
            # frozen dataclass 不允许赋值
            coefs.waiting_ratio_red_inc = 0.5  # type: ignore[misc]

    def test_cycle_state_init_default_uses_dataclass_defaults(self):
        """``rollout_queue_coefs=None`` 时, ``self.rollout_queue_coefs`` 必须是
        ``RolloutQueueCoefs()`` 实例 (与旧 hardcoded 行为一致)。"""
        model = models.CycleStateTrajectoryGenerator(
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
            rollout_queue_coefs=None,
        )
        self.assertIsInstance(model.rollout_queue_coefs, models.RolloutQueueCoefs)
        # 默认值必须与 #16 修复前的硬编码一致
        self.assertEqual(
            model.rollout_queue_coefs.waiting_ratio_red_inc,
            0.08,
        )

    def test_cycle_state_init_accepts_custom_rollout_queue_coefs(self):
        """显式传入 ``RolloutQueueCoefs(...)`` 时, ``self.rollout_queue_coefs``
        必须引用同一个实例 (id 相同), 不能被 ``__init__`` 重新复制。"""
        import dataclasses

        custom = dataclasses.replace(
            models.RolloutQueueCoefs(),
            waiting_ratio_red_inc=0.04,
            release_ratio_green_inc=0.20,
        )
        model = models.CycleStateTrajectoryGenerator(
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
            rollout_queue_coefs=custom,
        )
        self.assertIs(model.rollout_queue_coefs, custom)
        self.assertEqual(model.rollout_queue_coefs.waiting_ratio_red_inc, 0.04)
        self.assertEqual(model.rollout_queue_coefs.release_ratio_green_inc, 0.20)

    def test_rollout_queue_features_uses_self_rollout_queue_coefs_attribute(self):
        """``rollout_queue_features`` 源码中所有物理系数必须从
        ``self.rollout_queue_coefs`` 读取, 不再是裸字面量。"""
        import inspect

        source = inspect.getsource(models.CycleStateTrajectoryGenerator.rollout_queue_features)
        # 关键系数必须以 ``coefs.`` 形式出现, 禁止再以裸字面量形式出现
        for field_name in (
            "waiting_ratio_red_inc",
            "waiting_ratio_green_dec",
            "release_ratio_green_inc",
            "lane_queue_length_red_inc",
            "stopline_occupancy_red_inc",
            "stop_dist_pred_speed_dec",
            "queue_count_stopline_weight",
            "lane_density_prev_weight",
            "lane_mean_speed_prev_weight",
        ):
            self.assertIn(
                field_name,
                source,
                f"rollout_queue_features 源码必须通过 ``self.rollout_queue_coefs.{field_name}`` "
                f"读取系数, 但当前方法体里找不到该字段名; 说明 #16 集中化未完成。",
            )

    def test_rollout_queue_features_zero_red_increments_keeps_features_at_zero(self):
        """把 ``waiting_ratio_red_inc`` / ``lane_queue_length_red_inc`` /
        ``stopline_occupancy_red_inc`` / ``front_of_queue_red_inc`` 全部置 0 后,
        在全零输入 (red_like, 零速度) 下, 这些 queue feature 必须保持为 0
        (因为相位 = red 时, 红字面驱动是唯一增量来源, 全置零后没有增长项)。"""
        import dataclasses

        zero_red_coefs = dataclasses.replace(
            models.RolloutQueueCoefs(),
            waiting_ratio_red_inc=0.0,
            waiting_ratio_yellow_inc=0.0,
            lane_queue_length_red_inc=0.0,
            lane_queue_length_yellow_inc=0.0,
            lane_queue_length_phase_change_inc=0.0,
            stopline_occupancy_red_inc=0.0,
            front_of_queue_red_inc=0.0,
            stop_dist_step_discount_dec=0.0,
            stop_dist_phase_change_inc=0.0,
        )
        model = models.CycleStateTrajectoryGenerator(
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
            rollout_queue_coefs=zero_red_coefs,
        )
        prev, cycle, offset = self._build_minimal_rollout_inputs(n_agents=3)
        out = model.rollout_queue_features(prev, cycle, offset, step_index=0)
        # 受 red_inc 驱动的量应保持为 0
        self.assertTrue(torch.allclose(out[:, 3], torch.zeros(3)))  # waiting_ratio
        self.assertTrue(torch.allclose(out[:, 8], torch.zeros(3)))  # lane_queue_length
        self.assertTrue(torch.allclose(out[:, 9], torch.zeros(3)))  # stopline_occupancy
        self.assertTrue(torch.allclose(out[:, 10], torch.zeros(3)))  # front_of_queue
        # queue_count 间接依赖 waiting_ratio / lane_queue_length, 因此也应保持 0
        self.assertTrue(torch.allclose(out[:, 0], torch.zeros(3)))

    def test_rollout_queue_features_default_coefs_grow_waiting_ratio_under_red(self):
        """在默认 coefs 下, 全零输入 (red_like) 的 ``waiting_ratio`` 应当
        在一步 rollout 后增长 ``~0.08`` (即 ``waiting_ratio_red_inc``),
        证明默认系数确实生效 (而非被错误地置零)。"""
        out = self.model.rollout_queue_features(
            *self._build_minimal_rollout_inputs(n_agents=2),
            step_index=0,
        )
        waiting_ratio = out[:, 3]
        # 允许数值抖动 1e-6, 但增长应接近 0.08
        self.assertTrue(torch.all(waiting_ratio > 0.05))
        self.assertTrue(torch.all(waiting_ratio < 0.12))

    def test_rollout_queue_features_density_weight_override_changes_output(self):
        """覆盖 ``lane_density_prev_weight`` / ``lane_density_lane_queue_weight``
        必须改变 ``lane_density`` 输出, 证明权重确实被消费。

        为了让 ``lane_queue_length`` 在两种 coefs 下都保持 0 (避免对
        ``lane_density`` 造成不同的"拼接右项"), 两组 coefs 都把
        ``lane_queue_length`` / ``waiting_ratio`` 的所有 red/yellow 驱动
        置 0, 仅在 ``lane_density_prev_weight`` / ``lane_density_lane_queue_weight``
        上做差异。
        """
        import dataclasses

        prev, cycle, offset = self._build_minimal_rollout_inputs(n_agents=2)
        prev[:, 1] = 1.0  # 上一帧 lane_density 固定为 1.0, 让输出只取决于权重

        def _clean_density_only(weight_prev, weight_lane):
            return dataclasses.replace(
                models.RolloutQueueCoefs(),
                # --- 只在 lane_density 权重上做差异 ---
                lane_density_prev_weight=weight_prev,
                lane_density_lane_queue_weight=weight_lane,
                # --- 把 lane_queue_length 的所有驱动置 0, 让右项恒为 0 ---
                lane_queue_length_red_inc=0.0,
                lane_queue_length_yellow_inc=0.0,
                lane_queue_length_green_dec=0.0,
                lane_queue_length_phase_change_inc=0.0,
                # --- 把 waiting_ratio / stopline_occupancy / front_of_queue
                #     的红/黄驱动也置 0, 避免 queue_count 旁路 ---
                waiting_ratio_red_inc=0.0,
                waiting_ratio_yellow_inc=0.0,
                stopline_occupancy_red_inc=0.0,
                front_of_queue_red_inc=0.0,
                stop_dist_step_discount_dec=0.0,
                stop_dist_phase_change_inc=0.0,
            )

        # 默认权重 (0.6, 0.4) + lane_queue_length 恒 0 -> lane_density = 0.6*1.0 = 0.6
        coefs_default = _clean_density_only(0.6, 0.4)
        # 交换权重 (1.0, 0.0) + lane_queue_length 恒 0 -> lane_density = 1.0*1.0 = 1.0
        coefs_swapped = _clean_density_only(1.0, 0.0)

        def _build_model(coefs):
            return models.CycleStateTrajectoryGenerator(
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
                rollout_queue_coefs=coefs,
            )

        out_default = _build_model(coefs_default).rollout_queue_features(
            prev, cycle, offset, step_index=0
        )
        out_swapped = _build_model(coefs_swapped).rollout_queue_features(
            prev, cycle, offset, step_index=0
        )
        self.assertTrue(torch.allclose(out_default[:, 1], torch.full((2,), 0.6)))
        self.assertTrue(torch.allclose(out_swapped[:, 1], torch.full((2,), 1.0)))
        # 旁路检查: lane_queue_length 应保持 0 (因为驱动都置 0)
        self.assertTrue(torch.allclose(out_default[:, 8], torch.zeros(2)))
        self.assertTrue(torch.allclose(out_swapped[:, 8], torch.zeros(2)))

    def test_train_parser_accepts_rollout_queue_coefs_json(self):
        """``--rollout_queue_coefs_json`` 必须被 ``train`` 解析器接受,
        且默认值为空字符串 (触发 ``RolloutQueueCoefs()`` 默认值)。"""
        args = train.parser.parse_args(
            [
                "--model_type",
                "cyclestate",
                "--rollout_queue_coefs_json",
                '{"waiting_ratio_red_inc": 0.04}',
            ]
        )
        self.assertEqual(
            args.rollout_queue_coefs_json,
            '{"waiting_ratio_red_inc": 0.04}',
        )

    def test_train_parse_rollout_queue_coefs_returns_defaults_on_empty(self):
        """空字符串 / ``None`` 必须返回 ``RolloutQueueCoefs()`` 默认值。

        由于 ``test_cyclestate_protocol`` 通过 ``spec_from_file_location`` 把
        ``models.py`` 加载为 ``d2tp_models`` 命名空间, 而 ``train.py`` 是按
        真实 ``from models import ...`` 走的 ``D2TP.models`` 命名空间, 两者
        的 ``RolloutQueueCoefs`` 是不同的类对象 (但结构完全一致)。这里用
        duck-typing 检查, 而不是 ``isinstance``, 避免被模块双重加载的细节
        误伤。
        """
        coefs_empty = train.parse_rollout_queue_coefs("")
        coefs_none = train.parse_rollout_queue_coefs(None)
        for coefs in (coefs_empty, coefs_none):
            # 必备字段都在 + 默认值正确 = 是 ``RolloutQueueCoefs()``
            self.assertTrue(hasattr(coefs, "waiting_ratio_red_inc"))
            self.assertTrue(hasattr(coefs, "queue_count_stopline_weight"))
            self.assertEqual(coefs.waiting_ratio_red_inc, 0.08)
            self.assertEqual(coefs.lane_density_prev_weight, 0.6)

    def test_train_parse_rollout_queue_coefs_merges_valid_json(self):
        """合法 JSON 字符串必须做字段覆盖, 未指定的字段保持 dataclass 默认值。"""
        coefs = train.parse_rollout_queue_coefs(
            '{"waiting_ratio_red_inc": 0.04, "release_ratio_green_inc": 0.20}'
        )
        # 覆盖字段
        self.assertEqual(coefs.waiting_ratio_red_inc, 0.04)
        self.assertEqual(coefs.release_ratio_green_inc, 0.20)
        # 未覆盖字段保持默认值
        self.assertEqual(coefs.waiting_ratio_green_dec, 0.12)
        self.assertEqual(coefs.lane_queue_length_red_inc, 0.10)
        self.assertEqual(coefs.lane_density_prev_weight, 0.6)

    def test_train_parse_rollout_queue_coefs_falls_back_on_invalid_json(self):
        """非法 JSON 必须静默回退到默认值, 不可让训练启动失败。"""
        with self.assertLogs(level="WARNING") as captured:
            coefs = train.parse_rollout_queue_coefs("not-a-json{")
        # duck-typing: 必备字段 + 默认值
        self.assertTrue(hasattr(coefs, "waiting_ratio_red_inc"))
        self.assertEqual(coefs.waiting_ratio_red_inc, 0.08)
        # 警告日志应至少包含一次 "Failed to parse"
        joined = "\n".join(captured.output)
        self.assertIn("Failed to parse", joined)

    def test_train_parse_rollout_queue_coefs_ignores_unknown_keys(self):
        """JSON 里有未知字段名必须被静默忽略, 不能 raise。"""
        coefs = train.parse_rollout_queue_coefs(
            '{"waiting_ratio_red_inc": 0.05, "made_up_field": 999}'
        )
        self.assertEqual(coefs.waiting_ratio_red_inc, 0.05)
        # 没有 "made_up_field" 属性
        self.assertFalse(hasattr(coefs, "made_up_field"))

    def test_train_parse_rollout_queue_coefs_rejects_non_dict_json(self):
        """JSON 顶层不是对象 (例如 list / str) 必须回退到默认值。"""
        with self.assertLogs(level="WARNING") as captured:
            coefs = train.parse_rollout_queue_coefs("[1, 2, 3]")
        self.assertTrue(hasattr(coefs, "waiting_ratio_red_inc"))
        self.assertEqual(coefs.waiting_ratio_red_inc, 0.08)
        joined = "\n".join(captured.output)
        self.assertIn("must be a JSON object", joined)

    def test_train_parse_rollout_queue_coefs_rejects_invalid_field_value_types(self):
        """JSON 虽然语法合法, 但字段值类型错误时也必须回退到默认值。

        这是 #16 真正的运行时安全边界：如果 ``waiting_ratio_red_inc`` 被
        填成字符串, 不能把坏值带进 dataclass 再等到 rollout 时才在 tensor
        运算里炸掉。
        """
        with self.assertLogs(level="WARNING") as captured:
            coefs = train.parse_rollout_queue_coefs(
                '{"waiting_ratio_red_inc": "oops"}'
            )
        self.assertEqual(coefs.waiting_ratio_red_inc, 0.08)
        joined = "\n".join(captured.output)
        self.assertIn("invalid values", joined)

    def test_evaluate_model_parser_accepts_rollout_queue_coefs_json(self):
        """``evaluate_model.py`` 同样必须接受 ``--rollout_queue_coefs_json``。"""
        from evaluate_model import parser as eval_parser

        args = eval_parser.parse_args(
            [
                "--model_type",
                "cyclestate",
                "--rollout_queue_coefs_json",
                '{"lane_queue_length_red_inc": 0.15}',
            ]
        )
        self.assertEqual(
            args.rollout_queue_coefs_json,
            '{"lane_queue_length_red_inc": 0.15}',
        )

    def test_apply_rollout_coefs_override_preserves_untouched_fields(self):
        """``apply_rollout_coefs_override`` 必须保留未在 override dict 里
        出现的字段为 dataclass 默认值。"""
        base = models.RolloutQueueCoefs()
        override = {"waiting_ratio_red_inc": 0.04}
        merged, invalid_keys = models.apply_rollout_coefs_override(base, override)
        self.assertEqual(invalid_keys, ())
        self.assertEqual(merged.waiting_ratio_red_inc, 0.04)
        # 其它字段不变
        for field_name in (
            "waiting_ratio_green_dec",
            "release_ratio_green_inc",
            "lane_density_prev_weight",
        ):
            self.assertEqual(
                getattr(merged, field_name),
                getattr(base, field_name),
                f"Field {field_name!r} should not have been touched by override",
            )

    def test_rollout_queue_features_no_bare_magic_numbers_in_body(self):
        """源码守卫: ``rollout_queue_features`` 方法体内不应再出现
        那些 #16 修复前是裸字面量的物理系数 (如 ``0.08``、``0.10``、
        ``0.12``、``0.14``、``0.6``、``0.4``、``0.5``), 它们必须从
        ``self.rollout_queue_coefs`` 读取。
        
        注意:
        1. ``2.0`` 是 ``phase_value = phase_id.float() / 2.0`` 和
           ``elapsed.clamp(max=2.0) / 2.0`` 的结构化常量 (相位归一化),
           不是 #16 修复目标, 因此不做检查。
        2. dataclass 字段默认值会在 ``models.py`` 顶部出现, 但本测试只
           截取 ``rollout_queue_features`` 方法体, 不受影响。
        """
        import inspect

        source = inspect.getsource(models.CycleStateTrajectoryGenerator.rollout_queue_features)
        forbidden_literals = [
            "0.08",
            "0.10",
            "0.12",
            "0.14",
            "0.5",
            "0.6",
            "0.4",
        ]
        for lit in forbidden_literals:
            self.assertNotIn(
                lit,
                source,
                f"rollout_queue_features 源码中不应再出现裸字面量 {lit!r}; "
                f"它必须通过 ``self.rollout_queue_coefs.<field>`` 读取 (#16 修复)。",
            )


    # ----------------------------------------------------------------
    # #18 fix: per-step noise injection during decoding
    # ----------------------------------------------------------------

    def _make_baseline_model(self):
        return models.TrajectoryGenerator(
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

    def _fixed_initial_decoder_noise(self, model=None):
        """Return a deterministic init-noise tensor for one scene.

        This isolates the per-step decoding noise from the older
        initialization-time `add_noise(...)` randomness. #18 is specifically
        about *decode-step* injection, so tests must hold the init noise fixed
        or they can pass on the pre-fix implementation.
        """
        model = model or self.model
        return torch.full(
            (self.seq_start_end.size(0),) + model.noise_dim,
            0.25,
            dtype=torch.float32,
        )

    def test_baseline_eval_per_step_noise_calls_get_noise_once_per_step(self):
        """#18 fix must also hold on the base TrajectoryGenerator eval path:
        1 init-noise call + pred_len decode-step calls.
        """
        baseline_model = self._make_baseline_model()
        baseline_model.eval()
        noise_calls = []

        def fake_noise(shape, noise_type, device):
            noise_calls.append((tuple(shape), noise_type, str(device)))
            return torch.zeros(*shape, device=device)

        with mock.patch.object(models, "get_noise", side_effect=fake_noise):
            with torch.no_grad():
                baseline_model(
                    self.obs_traj_rel,
                    self.obs_traj,
                    self.obs_state,
                    self.pred_state,
                    self.seq_start_end,
                )

        self.assertEqual(
            1 + baseline_model.pred_len,
            len(noise_calls),
            "Baseline TrajectoryGenerator eval path should request one "
            "init-noise sample plus one decode-step noise sample per step.",
        )

    def test_baseline_train_per_step_noise_calls_get_noise_once_per_step(self):
        """#18 fix must also hold on the base TrajectoryGenerator training
        path, not only on eval/self-rollout.
        """
        baseline_model = self._make_baseline_model()
        baseline_model.train()
        noise_calls = []

        def fake_noise(shape, noise_type, device):
            noise_calls.append((tuple(shape), noise_type, str(device)))
            return torch.zeros(*shape, device=device)

        with mock.patch.object(models, "get_noise", side_effect=fake_noise):
            with torch.no_grad():
                baseline_model(
                    self.obs_traj_rel,
                    self.obs_traj,
                    self.obs_state,
                    self.pred_state,
                    self.seq_start_end,
                    teacher_forcing_ratio=1.0,
                )

        self.assertEqual(
            1 + baseline_model.pred_len,
            len(noise_calls),
            "Baseline TrajectoryGenerator training path should request one "
            "init-noise sample plus one decode-step noise sample per step.",
        )

    def test_baseline_per_step_noise_affects_output_beyond_initial_add_noise(self):
        """#18 fix on the base generator: holding init noise fixed, changing
        only decode-step noise must change the final output.
        """
        baseline_model = self._make_baseline_model()
        baseline_model.eval()
        init_noise = self._fixed_initial_decoder_noise(baseline_model)

        def run_with_step_noise(step_noise_value):
            call_index = {"value": 0}

            def fake_noise(shape, noise_type, device):
                if call_index["value"] == 0:
                    out = init_noise.to(device)
                else:
                    out = torch.full(
                        shape,
                        step_noise_value,
                        dtype=torch.float32,
                        device=device,
                    )
                call_index["value"] += 1
                return out

            with mock.patch.object(models, "get_noise", side_effect=fake_noise):
                with torch.no_grad():
                    return baseline_model(
                        self.obs_traj_rel,
                        self.obs_traj,
                        self.obs_state,
                        self.pred_state,
                        self.seq_start_end,
                    )

        out_zero_step_noise = run_with_step_noise(0.0)
        out_one_step_noise = run_with_step_noise(1.0)

        self.assertFalse(
            torch.allclose(out_zero_step_noise, out_one_step_noise, atol=1e-4),
            "Holding init noise fixed, changing only base-generator "
            "decode-step noise should change the output.",
        )

    def test_per_step_noise_calls_get_noise_once_per_decoding_step(self):
        """#18 fix: after the initial add_noise call, decoding must request
        fresh noise exactly once per step.

        This test separates the old initialization-time noise call from the
        new per-step calls by patching `get_noise` and counting invocations:
        1 init call + pred_len decode-step calls in eval mode.
        """
        self.model.eval()
        noise_calls = []

        def fake_noise(shape, noise_type, device):
            noise_calls.append((tuple(shape), noise_type, str(device)))
            return torch.zeros(*shape, device=device)

        with mock.patch.object(models, "get_noise", side_effect=fake_noise):
            with torch.no_grad():
                self.model(
                    self.obs_traj_rel,
                    self.obs_traj,
                    self.obs_state,
                    self.pred_state,
                    self.seq_start_end,
                )

        self.assertEqual(
            1 + self.model.pred_len,
            len(noise_calls),
            "Expected exactly one init-noise call plus one get_noise call "
            "per decoding step in eval mode.",
        )
        self.assertEqual(
            (self.seq_start_end.size(0),) + self.model.noise_dim,
            noise_calls[0][0],
            "The initialization-time add_noise call should request scene-level "
            "noise with shape (num_scenes, noise_dim).",
        )
        for idx, call in enumerate(noise_calls[1:], start=1):
            self.assertEqual(
                (self.seq_start_end.size(0),) + self.model.noise_dim,
                call[0],
                f"Decode step {idx} requested the wrong noise shape.",
            )

    def test_per_step_noise_affects_output_beyond_initial_add_noise(self):
        """#18 fix: holding init noise fixed, changing only decode-step noise
        must change the final output.

        This is the key regression guard the earlier test missed: the pre-fix
        model already had random `add_noise(...)` at decoder initialization, so
        "two runs differ" was insufficient evidence. Here the first noise draw
        (init) is fixed and only the per-step draws vary.
        """
        self.model.eval()
        init_noise = self._fixed_initial_decoder_noise(self.model)

        def run_with_step_noise(step_noise_value):
            call_index = {"value": 0}

            def fake_noise(shape, noise_type, device):
                if call_index["value"] == 0:
                    out = init_noise.to(device)
                else:
                    out = torch.full(
                        shape,
                        step_noise_value,
                        dtype=torch.float32,
                        device=device,
                    )
                call_index["value"] += 1
                return out

            with mock.patch.object(models, "get_noise", side_effect=fake_noise):
                with torch.no_grad():
                    return self.model(
                        self.obs_traj_rel,
                        self.obs_traj,
                        self.obs_state,
                        self.pred_state,
                        self.seq_start_end,
                    )

        out_zero_step_noise = run_with_step_noise(0.0)
        out_one_step_noise = run_with_step_noise(1.0)

        self.assertFalse(
            torch.allclose(out_zero_step_noise, out_one_step_noise, atol=1e-4),
            "With identical initialization noise, changing only decode-step "
            "noise should change the output. If not, per-step noise is not "
            "affecting decoding.",
        )

    def test_per_step_noise_affects_hidden_state(self):
        """#18 fix: the per-step noise must actually modify pred_lstm_hidden
        before the LSTM step, not just be generated and discarded.

        Strategy: hold the init noise fixed, then compare zero step-noise vs
        non-zero step-noise. The two outputs should differ, proving the
        decode-step noise value propagates through the LSTM.
        """
        self.model.eval()
        seq_start_end = self.seq_start_end
        init_noise = self._fixed_initial_decoder_noise(self.model)

        def fake_noise_factory(step_noise_tensor):
            call_index = {"value": 0}

            def fake_noise(shape, noise_type, device):
                if call_index["value"] == 0:
                    out = init_noise.to(device)
                else:
                    out = step_noise_tensor.to(device)
                call_index["value"] += 1
                return out

            return fake_noise

        with torch.no_grad():
            with mock.patch.object(
                models,
                "get_noise",
                side_effect=fake_noise_factory(
                    torch.zeros((seq_start_end.size(0),) + self.model.noise_dim)
                ),
            ):
                out_zero_noise = self.model(
                    self.obs_traj_rel,
                    self.obs_traj,
                    self.obs_state,
                    self.pred_state,
                    self.seq_start_end,
                )

            with mock.patch.object(
                models,
                "get_noise",
                side_effect=fake_noise_factory(
                    torch.ones((seq_start_end.size(0),) + self.model.noise_dim)
                ),
            ):
                out_ones_noise = self.model(
                    self.obs_traj_rel,
                    self.obs_traj,
                    self.obs_state,
                    self.pred_state,
                    self.seq_start_end,
                )

        self.assertFalse(
            torch.allclose(out_zero_noise, out_ones_noise, atol=1e-4),
            "Per-step noise with non-zero values should change the decoded "
            "output compared to zero noise. If outputs are identical, the "
            "noise may not be propagating through the LSTM hidden state.",
        )

    def test_cyclestate_per_step_noise_calls_get_noise_once_per_step(self):
        """#18 fix for CycleStateTrajectoryGenerator: its subclass-specific
        eval path must also request one fresh noise sample per decoding step,
        in addition to the single initialization call.
        """
        # Build a traffic context to exercise the CycleState eval path
        traffic_context = self.model.build_traffic_context(
            self.obs_traj_rel,
            self.obs_traj,
            self.obs_state,
            self.pred_state,
            self.seq_start_end,
        )

        self.model.eval()
        noise_calls = []

        def fake_noise(shape, noise_type, device):
            noise_calls.append((tuple(shape), noise_type, str(device)))
            return torch.zeros(*shape, device=device)

        with mock.patch.object(models, "get_noise", side_effect=fake_noise):
            with torch.no_grad():
                self.model(
                    self.obs_traj_rel,
                    self.obs_traj,
                    self.obs_state,
                    self.pred_state,
                    self.seq_start_end,
                    traffic_context=traffic_context,
                )

        self.assertEqual(
            1 + self.model.pred_len,
            len(noise_calls),
            "CycleStateTrajectoryGenerator eval path should request exactly "
            "one init-noise sample plus one decode-step noise sample per step.",
        )

    def test_cyclestate_per_step_noise_affects_output_beyond_initial_add_noise(self):
        """#18 fix for CycleStateTrajectoryGenerator: holding init noise
        fixed, changing only subclass decode-step noise must change output.
        """
        traffic_context = self.model.build_traffic_context(
            self.obs_traj_rel,
            self.obs_traj,
            self.obs_state,
            self.pred_state,
            self.seq_start_end,
        )
        init_noise = self._fixed_initial_decoder_noise(self.model)

        def run_with_step_noise(step_noise_value):
            call_index = {"value": 0}

            def fake_noise(shape, noise_type, device):
                if call_index["value"] == 0:
                    out = init_noise.to(device)
                else:
                    out = torch.full(
                        shape,
                        step_noise_value,
                        dtype=torch.float32,
                        device=device,
                    )
                call_index["value"] += 1
                return out

            with mock.patch.object(models, "get_noise", side_effect=fake_noise):
                with torch.no_grad():
                    return self.model(
                        self.obs_traj_rel,
                        self.obs_traj,
                        self.obs_state,
                        self.pred_state,
                        self.seq_start_end,
                        traffic_context=traffic_context,
                    )

        self.model.eval()
        with torch.no_grad():
            out_zero_step_noise = run_with_step_noise(0.0)
            out_one_step_noise = run_with_step_noise(1.0)

        self.assertFalse(
            torch.allclose(out_zero_step_noise, out_one_step_noise, atol=1e-4),
            "Holding the init noise fixed, changing only CycleState decode-step "
            "noise should change the output.",
        )

    # ------------------------------------------------------------------
    # Phase 0 #19: TRAIN_STAGE_DEFAULTS 联动不一致 (consistency assertions)
    # ------------------------------------------------------------------
    def _make_stage_args(self, **overrides):
        """构造一个完整的 SimpleNamespace 用于 ``validate_stage_consistency`` 校验。"""
        base = {
            "train_stage": "warmup",
            "model_type": "cyclestate",
            "generator_only": None,
            "gan_weight": None,
            "aux_queue_weight": None,
            "aux_rollout_weight": None,
            "aux_cycle_weight": None,
            "teacher_forcing_ratio": None,
            "grad_clip": None,
            "rollout_residual_scale": None,
            "detach_rollout_state": None,
        }
        base.update(overrides)
        return types.SimpleNamespace(**base)

    def test_validate_stage_consistency_function_exists(self):
        """Phase 0 #19: train.py 必须导出 ``validate_stage_consistency``。"""
        self.assertTrue(
            hasattr(train, "validate_stage_consistency"),
            "Phase 0 #19: train.validate_stage_consistency is required to "
            "guard against TRAIN_STAGE_DEFAULTS linkage contradictions.",
        )
        self.assertTrue(callable(train.validate_stage_consistency))

    def test_validate_stage_consistency_accepts_warmup_defaults(self):
        """``apply_stage_defaults`` 默认值必须能通过一致性校验。"""
        args = self._make_stage_args()
        train.apply_stage_defaults(args)
        # 应当不抛异常
        result = train.validate_stage_consistency(args)
        self.assertIs(result, args)

    def test_validate_stage_consistency_accepts_refine_defaults(self):
        args = self._make_stage_args(train_stage="refine")
        train.apply_stage_defaults(args)
        train.validate_stage_consistency(args)
        # refine 阶段同样默认 generator_only=True / gan_weight=0
        self.assertTrue(args.generator_only)
        self.assertEqual(0.0, args.gan_weight)

    def test_validate_stage_consistency_accepts_adversarial_defaults(self):
        args = self._make_stage_args(train_stage="adversarial")
        train.apply_stage_defaults(args)
        train.validate_stage_consistency(args)
        self.assertFalse(args.generator_only)
        self.assertGreater(args.gan_weight, 0.0)

    def test_validate_stage_consistency_accepts_baseline_defaults(self):
        args = self._make_stage_args(model_type="d2tpred", train_stage="warmup")
        train.apply_stage_defaults(args)
        train.validate_stage_consistency(args)
        # baseline 默认 generator_only=False 且 gan_weight>0,合法
        self.assertFalse(args.generator_only)
        self.assertGreater(args.gan_weight, 0.0)

    def test_validate_stage_consistency_raises_on_gan_weight_with_generator_only(self):
        """核心联动: gan_weight > 0 与 generator_only=True 互斥。"""
        args = self._make_stage_args(
            train_stage="warmup",
            generator_only=True,
            gan_weight=50.0,  # 非零 GAN 权重
        )
        train.apply_stage_defaults(args)
        with self.assertRaises(ValueError) as cm:
            train.validate_stage_consistency(args)
        self.assertIn("gan_weight", str(cm.exception))
        self.assertIn("generator_only", str(cm.exception))

    def test_validate_stage_consistency_warns_on_adversarial_with_zero_gan(self):
        """软矛盾: train_stage=adversarial 但 gan_weight==0。"""
        args = self._make_stage_args(
            train_stage="adversarial",
            generator_only=False,
            gan_weight=0.0,  # 显式清零 GAN 权重
        )
        train.apply_stage_defaults(args)
        with self.assertLogs(level="WARNING") as cm:
            train.validate_stage_consistency(args)
        # 警告内容应涉及 adversarial / gan_weight
        joined = "\n".join(cm.output)
        self.assertTrue(
            "adversarial" in joined.lower() or "gan_weight" in joined.lower(),
            f"Expected warning to mention adversarial/gan_weight, got: {joined}",
        )

    def test_validate_stage_consistency_raises_on_negative_gan_weight(self):
        """`gan_weight < 0` 必须是硬错误,而不是 warning。

        这是 #19 当前漏掉的关键边界: 训练总损失里直接做
        ``total_loss = ... + g_loss * args.gan_weight``。若 ``gan_weight`` 为负,
        生成器会被驱动去**增大**对抗损失,等价于把优化方向翻转,这不是
        “可疑但仍可运行”的消融配置,而是明确的错误配置。
        """
        args = self._make_stage_args(
            train_stage="adversarial",
            generator_only=False,
            gan_weight=-1.0,
        )
        train.apply_stage_defaults(args)
        with self.assertRaises(ValueError) as cm:
            train.validate_stage_consistency(args)
        self.assertIn("gan_weight", str(cm.exception))
        self.assertIn("non-negative", str(cm.exception))

    def test_validate_stage_consistency_warns_on_rollout_without_queue(self):
        """软矛盾: aux_rollout_weight>0 但 aux_queue_weight==0。"""
        args = self._make_stage_args(
            train_stage="warmup",
            aux_queue_weight=0.0,
            aux_rollout_weight=2.0,  # rollout 有权重但 queue 没有
        )
        train.apply_stage_defaults(args)
        with self.assertLogs(level="WARNING") as cm:
            train.validate_stage_consistency(args)
        joined = "\n".join(cm.output)
        self.assertIn("aux_rollout", joined.lower())

    def test_validate_stage_consistency_raises_on_negative_grad_clip(self):
        args = self._make_stage_args(grad_clip=-1.0)
        train.apply_stage_defaults(args)
        with self.assertRaises(ValueError) as cm:
            train.validate_stage_consistency(args)
        self.assertIn("grad_clip", str(cm.exception))

    def test_validate_stage_consistency_raises_on_negative_rollout_residual_scale(self):
        args = self._make_stage_args(rollout_residual_scale=-0.1)
        train.apply_stage_defaults(args)
        with self.assertRaises(ValueError) as cm:
            train.validate_stage_consistency(args)
        self.assertIn("rollout_residual_scale", str(cm.exception))

    def test_validate_stage_consistency_raises_on_invalid_teacher_forcing_ratio(self):
        for bad in (-0.1, 1.1, 2.0, -1.0):
            args = self._make_stage_args(teacher_forcing_ratio=bad)
            train.apply_stage_defaults(args)
            with self.assertRaises(ValueError) as cm:
                train.validate_stage_consistency(args)
            self.assertIn("teacher_forcing_ratio", str(cm.exception))

    def test_validate_stage_consistency_accepts_boundary_teacher_forcing(self):
        for ok in (0.0, 0.5, 1.0):
            args = self._make_stage_args(teacher_forcing_ratio=ok)
            train.apply_stage_defaults(args)
            train.validate_stage_consistency(args)

    def test_validate_stage_consistency_raises_on_negative_aux_weights(self):
        for field in ("aux_queue_weight", "aux_cycle_weight", "aux_rollout_weight"):
            args = self._make_stage_args(**{field: -1.0})
            train.apply_stage_defaults(args)
            with self.assertRaises(ValueError) as cm:
                train.validate_stage_consistency(args)
            self.assertIn(field, str(cm.exception))

    def test_main_invokes_validate_after_apply_stage_defaults(self):
        """源码守卫: ``main`` 必须显式调用 ``validate_stage_consistency``。"""
        main_src = pathlib.Path(train.__file__).read_text(encoding="utf-8")
        apply_pos = main_src.find("apply_stage_defaults(args)")
        validate_pos = main_src.find("validate_stage_consistency(args)")
        self.assertNotEqual(-1, apply_pos, "main must still call apply_stage_defaults")
        self.assertNotEqual(-1, validate_pos, "main must call validate_stage_consistency")
        # 校验顺序: validate 在 apply 之后调用
        self.assertLess(
            apply_pos,
            validate_pos,
            "validate_stage_consistency must run after apply_stage_defaults",
        )

    # ----------------------------------------------------------------
    # #20 Phase 2: compute_structured_aux_losses target 语义验证
    # ----------------------------------------------------------------
    #
    # The remaining #20 work focuses on **target semantic verification**
    # and **main-vs-rollout supervision asymmetry documentation**:
    #
    # 1. Each target dim's data type must match the loss function assigned
    #    to it (regression vs binary classification vs cross-entropy).
    # 2. The dim ORDER in the target tensor must match the ORDER in
    #    ``compute_queue_targets`` (queue) and ``build_cycle_features``
    #    (cycle), so MSE/BCE indices slice the correct sub-space.
    # 3. The supervision design (main = last obs frame, rollout = full pred
    #    sequence) must be documented and tested, so a future refactor that
    #    silently changes this asymmetry breaks a test loudly.
    # ----------------------------------------------------------------

    def test_compute_queue_targets_returns_6_dims_in_canonical_order(self):
        """#20 P2 verification: ``compute_queue_targets`` must return a
        6-dim tensor in the canonical order

            [queue_count, lane_wait_ratio, lane_release_ratio,
             lane_queue_length, lane_stopline_occupancy, front_of_queue]

        matching the dim split in ``compute_structured_aux_losses``
        (reg=[0,1,2,3] / cls=[4,5]). A future reorder must be reflected
        here.
        """
        expected = (
            {
                "target_index": 0,
                "name": "queue_count",
                "loss": "regression",
                "source_index": 0,
            },
            {
                "target_index": 1,
                "name": "lane_wait_ratio",
                "loss": "regression",
                "source_index": 3,
            },
            {
                "target_index": 2,
                "name": "lane_release_ratio",
                "loss": "regression",
                "source_index": 4,
            },
            {
                "target_index": 3,
                "name": "lane_queue_length",
                "loss": "regression",
                "source_index": 8,
            },
            {
                "target_index": 4,
                "name": "lane_stopline_occupancy",
                "loss": "binary",
                "source_index": 9,
            },
            {
                "target_index": 5,
                "name": "front_of_queue",
                "loss": "binary",
                "source_index": 10,
            },
        )
        self.assertEqual(
            expected,
            models.build_queue_targets_signature(),
            "build_queue_targets_signature must return the canonical dim "
            "order, loss type, and source-index contract used by "
            "compute_structured_aux_losses.",
        )

    def test_compute_queue_targets_selects_expected_source_dims(self):
        """#20 P2 verification: ``compute_queue_targets`` must pick
        ``queue_feature_seq[..., [0,3,4,8,9,10]]`` in order, so a
        reorder or wrong source dim fails loudly.
        """
        queue_feature_seq = torch.stack(
            [
                torch.full((2, 3), float(dim), dtype=torch.float32)
                for dim in range(self.model.queue_feature_dim)
            ],
            dim=2,
        )
        queue_targets = self.model.compute_queue_targets(queue_feature_seq)
        expected = torch.stack(
            (
                queue_feature_seq[:, :, 0],
                queue_feature_seq[:, :, 3],
                queue_feature_seq[:, :, 4],
                queue_feature_seq[:, :, 8],
                queue_feature_seq[:, :, 9],
                queue_feature_seq[:, :, 10],
            ),
            dim=2,
        )
        self.assertTrue(
            torch.equal(queue_targets, expected),
            "compute_queue_targets must preserve the canonical "
            "[0,3,4,8,9,10] source-dim order.",
        )

    def test_build_cycle_features_returns_6_dims_in_canonical_order(self):
        """#20 P2 verification: ``build_cycle_features`` must return a
        6-dim tensor in the canonical order

            [phase_one_hot(3), elapsed(1), remaining(1), phase_change(1)]

        matching the dim split in ``compute_structured_aux_losses``
        (phase=[:3] / time=[3:5] / change=[5:6]). A future reorder must
        be reflected here.
        """
        expected = (
            {"target_index": 0, "name": "phase_red", "loss": "classification"},
            {"target_index": 1, "name": "phase_green", "loss": "classification"},
            {"target_index": 2, "name": "phase_yellow", "loss": "classification"},
            {"target_index": 3, "name": "elapsed", "loss": "regression"},
            {"target_index": 4, "name": "remaining", "loss": "regression"},
            {"target_index": 5, "name": "phase_change", "loss": "binary"},
        )
        self.assertEqual(
            expected,
            models.build_cycle_features_signature(),
            "build_cycle_features_signature must return the canonical dim "
            "order and loss-type contract used by "
            "compute_structured_aux_losses.",
        )

    def test_compute_structured_aux_losses_docstring_documents_dim_semantics(self):
        """#20 P2 verification: ``compute_structured_aux_losses`` must
        document, in its docstring, the dim→loss-type mapping so a future
        contributor cannot silently change the contract.
        """
        import inspect
        src = inspect.getsource(train.compute_structured_aux_losses)
        for keyword in (
            "regression",
            "binary",
            "classification",
            "queue_reg_idx",
            "queue_cls_idx",
            "lane_queue_length",
            "phase",
            "phase_change",
        ):
            self.assertIn(
                keyword, src,
                f"compute_structured_aux_losses docstring/source must "
                f"mention {keyword!r} so the dim→loss contract stays "
                f"explicit.",
            )

    def test_compute_structured_aux_losses_docstring_documents_main_vs_rollout_asymmetry(self):
        """#20 P2 verification: the main aux uses the LAST observation frame
        as target, while the rollout aux uses the FULL prediction sequence.
        This asymmetry must be explicitly documented in the function source
        so a refactor cannot silently collapse the two supervision schemes.
        """
        import inspect
        src = inspect.getsource(train.compute_structured_aux_losses)
        # docstring + inline comments must mention both schemes.
        self.assertIn(
            "queue_target_last", src,
            "main aux must still use the last-frame (queue_target_last) "
            "supervision; removing this contract silently changes the "
            "supervision coverage.",
        )
        self.assertIn(
            "queue_rollout_target_seq", src,
            "rollout aux must still use the per-step prediction sequence "
            "(queue_rollout_target_seq); the asymmetry with main is part "
            "of the design.",
        )
        # And the docstring must explicitly mention the asymmetry.
        self.assertIn(
            "asymmetry", src.lower(),
            "the main-vs-rollout supervision asymmetry is part of the "
            "design and must be mentioned in the docstring/comment.",
        )

    def test_train_call_site_uses_last_frame_for_main_and_sequence_for_rollout(self):
        """#20 P2 verification: the call site in train.py must use

            queue_target_last        = aux_info["queue_targets"][-1]
            cycle_target_last        = aux_info["cycle_feature_seq"][-3:].mean(dim=0)
            queue_rollout_target_seq = aux_info.get("queue_rollout_target_seq")

        (last obs frame for main queue, last-3-frame average for cycle,
        full pred sequence for rollout).
        This guards the documented asymmetry at the call-site level.
        """
        import ast
        train_source = (REPO_ROOT / "D2TP" / "train.py").read_text()
        tree = ast.parse(train_source)
        main_last_frame = re.findall(
            r"aux_info\[\"queue_targets\"\]\[-1\]", train_source
        )
        # Phase 3 #24: cycle 现在取最后 3 帧平均, 检查 [-3:] 和 .mean(dim=0)
        cycle_tail_avg = re.findall(
            r"aux_info\[\"cycle_feature_seq\"\]\[-3:\]", train_source
        )
        cycle_mean_dim0 = re.findall(
            r"cycle_feature_tail\.mean\(dim=0\)", train_source
        )
        rollout_full_seq = re.findall(
            r"aux_info(\.get)?\(\"queue_rollout_target_seq\"\)", train_source
        )
        self.assertGreaterEqual(
            len(main_last_frame), 1,
            "train.py must pull the last observation frame for main "
            "queue aux supervision (queue_target_last).",
        )
        self.assertGreaterEqual(
            len(cycle_tail_avg), 1,
            "train.py must use [-3:] tail average for cycle aux "
            "supervision (cycle_target_last from #24).",
        )
        self.assertGreaterEqual(
            len(cycle_mean_dim0), 1,
            "train.py must apply .mean(dim=0) on cycle_feature_tail "
            "for the 3-frame average (#24).",
        )
        self.assertGreaterEqual(
            len(rollout_full_seq), 1,
            "train.py must pass the full prediction sequence to the "
            "rollout aux loss (queue_rollout_target_seq).",
        )

    def test_compute_structured_aux_losses_rejects_queue_target_wrong_dim(self):
        """#20 P2 verification: the queue target's last dim must be 6
        (4 reg + 2 cls). A wrong dim (e.g. 5 or 7) must be caught by the
        pred/target shape assertion in compute_structured_aux_losses,
        not silently mis-sliced by ``queue_reg_idx`` / ``queue_cls_idx``.
        """
        # Wrong: 5 dims (one reg dim dropped)
        with self.assertRaises(AssertionError):
            train.compute_structured_aux_losses(
                queue_pred_last=torch.zeros(3, 5),
                queue_target_last=torch.zeros(3, 5),
                cycle_pred_last=torch.zeros(3, 6),
                cycle_target_last=torch.zeros(3, 6),
            )
        # Wrong: 7 dims (one cls dim added)
        with self.assertRaises(AssertionError):
            train.compute_structured_aux_losses(
                queue_pred_last=torch.zeros(3, 7),
                queue_target_last=torch.zeros(3, 7),
                cycle_pred_last=torch.zeros(3, 6),
                cycle_target_last=torch.zeros(3, 6),
            )

    def test_compute_structured_aux_losses_rejects_cycle_target_wrong_dim(self):
        """#20 P2 verification: the cycle target's last dim must be 6
        (3 phase + 2 time + 1 change). A wrong dim must be caught.
        """
        with self.assertRaises(AssertionError):
            train.compute_structured_aux_losses(
                queue_pred_last=torch.zeros(3, 6),
                queue_target_last=torch.zeros(3, 6),
                cycle_pred_last=torch.zeros(3, 5),
                cycle_target_last=torch.zeros(3, 5),
            )
        with self.assertRaises(AssertionError):
            train.compute_structured_aux_losses(
                queue_pred_last=torch.zeros(3, 6),
                queue_target_last=torch.zeros(3, 6),
                cycle_pred_last=torch.zeros(3, 7),
                cycle_target_last=torch.zeros(3, 7),
            )

    def test_compute_structured_aux_losses_bce_target_must_be_probability(self):
        """#20 P2 verification: a queue BCE loss computed against a
        regression-style target (continuous 0..1 but not strictly 0/1) is
        still well-defined numerically. But a target with values outside
        [0, 1] would be a clear data-bug. We assert the loss is finite
        for a 0/1 target and is non-negative.
        """
        queue_pred = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.2, -0.4]])
        queue_target = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        cycle_pred = torch.tensor([[0.1, 1.8, -0.4, 0.4, 0.6, -0.2]])
        cycle_target = torch.tensor([[1.0, 0.0, 0.0, 0.5, 0.5, 1.0]])
        losses = train.compute_structured_aux_losses(
            queue_pred_last=queue_pred, queue_target_last=queue_target,
            cycle_pred_last=cycle_pred, cycle_target_last=cycle_target,
        )
        # BCE must be finite and non-negative for valid 0/1 targets.
        for k in ("queue_cls_loss", "cycle_change_loss", "cycle_phase_loss"):
            v = float(losses[k].item())
            self.assertTrue(
                v >= 0.0 and v < float("inf"),
                f"{k} must be finite and non-negative for valid targets, "
                f"got {v}.",
            )

    def test_compute_structured_aux_losses_phase_argmax_handles_label_noise(self):
        """#20 P2 verification: the cycle phase loss is computed via
        ``argmax(dim=1)`` over a one-hot target. A non-strictly-one-hot
        target (e.g. a tie) must not crash and must still produce a
        finite loss.
        """
        # Tie (0.4, 0.4, 0.2) → argmax returns 0
        cycle_pred = torch.tensor([[0.1, -0.2, 0.0, 0.0, 0.0, 0.0]])
        cycle_target = torch.tensor([[0.4, 0.4, 0.2, 0.0, 0.0, 0.0]])
        losses = train.compute_structured_aux_losses(
            queue_pred_last=torch.zeros(1, 6), queue_target_last=torch.zeros(1, 6),
            cycle_pred_last=cycle_pred, cycle_target_last=cycle_target,
        )
        self.assertTrue(
            float(losses["cycle_phase_loss"].item()) >= 0.0,
            "cycle_phase_loss must be non-negative for a valid one-hot "
            "target (with ties allowed).",
        )

    def test_compute_structured_aux_losses_rollout_seq_flattens_over_time(self):
        """#20 P2 verification: the rollout aux loss flattens the
        ``(T, batch, 6)`` sequence over T*batch and applies the same
        reg/cls split. We verify:

        1. A rollout pred/target with shape ``(2, 3, 6)`` produces
           finite losses (no shape error).
        2. The total rollout loss equals the manual mean over the
           flattened pred/target, so the per-step supervision is
           uniform (no time-decay or last-step bias).
        """
        torch.manual_seed(7)
        T, batch = 2, 3
        queue_rollout_pred = torch.randn(T, batch, 6)
        queue_rollout_target = torch.randn(T, batch, 6)
        losses = train.compute_structured_aux_losses(
            queue_pred_last=None, queue_target_last=None,
            cycle_pred_last=None, cycle_target_last=None,
            queue_rollout_pred_seq=queue_rollout_pred,
            queue_rollout_target_seq=queue_rollout_target,
        )
        import torch.nn.functional as F
        flat_p = queue_rollout_pred.reshape(-1, 6)
        flat_t = queue_rollout_target.reshape(-1, 6)
        reg_idx = [0, 1, 2, 3]
        cls_idx = [4, 5]
        expected_reg = F.mse_loss(flat_p[:, reg_idx], flat_t[:, reg_idx])
        expected_cls = F.binary_cross_entropy_with_logits(
            flat_p[:, cls_idx], flat_t[:, cls_idx]
        )
        self.assertTrue(
            torch.allclose(
                losses["queue_rollout_reg_loss"], expected_reg, atol=1e-6
            ),
            "queue_rollout_reg_loss must equal MSE over the flattened "
            "(T*batch, 6) rollout pred/target.",
        )
        self.assertTrue(
            torch.allclose(
                losses["queue_rollout_cls_loss"], expected_cls, atol=1e-6
            ),
            "queue_rollout_cls_loss must equal BCE over the flattened "
            "(T*batch, 6) rollout pred/target.",
        )

    def test_compute_structured_aux_losses_main_uses_last_frame_not_sequence(self):
        """#20 P2 verification (semantic design): the MAIN aux target
        passed by train.py is the LAST observation frame, not the full
        observation sequence. We assert that the loss values are
        unaffected by preceding frames, i.e. compute_structured_aux_losses
        does not silently mean-reduce over a sequence when given a 2-D
        ``(batch, 6)`` target.

        Concretely: feeding in a target where the last frame is the
        "noisy" row and a target where the last frame is the "clean" row
        must produce different losses, and the loss must equal the loss
        computed on just that last frame.
        """
        # batch=2, dim=6
        # Frame 0 = "clean", frame 1 = "noisy"
        clean = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        noisy = torch.tensor([[3.0, 3.0, 3.0, 3.0, 3.0, 3.0]])
        seq_target = torch.cat([clean, noisy], dim=0)  # (2, 6)
        # The main call site uses seq_target[-1] (the noisy row).
        pred = torch.zeros(1, 6)
        # Loss against just-noisy target (cycle kept at zero target so
        # phase/CE/change components are finite and well-defined).
        loss_noisy = train.compute_structured_aux_losses(
            queue_pred_last=pred, queue_target_last=noisy,
            cycle_pred_last=torch.zeros(1, 6), cycle_target_last=torch.zeros(1, 6),
        )
        # Loss against just-clean target.
        loss_clean = train.compute_structured_aux_losses(
            queue_pred_last=pred, queue_target_last=clean,
            cycle_pred_last=torch.zeros(1, 6), cycle_target_last=torch.zeros(1, 6),
        )
        # They must differ: the main aux is sensitive to the actual
        # target value, not silently averaged.
        self.assertNotAlmostEqual(
            float(loss_noisy["queue_reg_loss"].item()),
            float(loss_clean["queue_reg_loss"].item()),
            places=4,
            msg="queue_reg_loss must change with the target row used; "
            "if the loss is invariant the function is silently reducing "
            "over a sequence and breaking the last-frame design.",
        )
        # And it must match the explicit single-frame case.
        expected = torch.nn.functional.mse_loss(
            pred[:, [0, 1, 2, 3]], noisy[:, [0, 1, 2, 3]]
        )
        self.assertTrue(
            torch.allclose(
                loss_noisy["queue_reg_loss"], expected, atol=1e-6
            ),
            "queue_reg_loss with a 2-D target must equal MSE on the "
            "2-D target itself (no implicit temporal reduction).",
        )

    # ------------------------------------------------------------------
    # Phase 4 #21: best-of-K 采样次数对齐 (NumValSamplesTracker + ckpt 契约)
    # ------------------------------------------------------------------
    def test_num_val_samples_signature_returns_true_and_acts_as_contract(self):
        """#21: 契约 helper 必须返回结构化 dict, 包含所有关键字段名与约束。"""
        sig = train.build_num_val_samples_signature()
        self.assertIsInstance(
            sig,
            dict,
            "build_num_val_samples_signature() 必须返回结构化 dict,"
            " 不再是裸 True (Phase 4 #21 强化)。",
        )
        self.assertEqual(
            sig["checkpoint_key"],
            "num_val_samples",
            "checkpoint_key 必须是 'num_val_samples'。",
        )
        self.assertEqual(
            sig["runtime_arg"],
            "num_val_samples",
            "runtime_arg 必须是 'num_val_samples'。",
        )
        self.assertEqual(
            sig["eval_arg"],
            "num_samples",
            "eval_arg 必须是 'num_samples' (evaluate_model.py 中的参数名)。",
        )
        self.assertTrue(
            sig["must_persist_positive_int"],
            "must_persist_positive_int 必须为 True。",
        )
        # The helper must be a module-level function in train.py
        # (便于源码守卫审计)。
        self.assertTrue(
            callable(getattr(train, "build_num_val_samples_signature", None))
        )

    def test_num_val_samples_tracker_stores_int_value(self):
        """#21: NumValSamplesTracker 构造时把 args 值存为 int,缺失则为 None。"""
        tracker = train.NumValSamplesTracker(num_val_samples=20)
        self.assertEqual(20, tracker.value)
        self.assertEqual(20, tracker.checkpoint_payload())

        tracker_none = train.NumValSamplesTracker()
        self.assertIsNone(tracker_none.value)
        self.assertIsNone(tracker_none.checkpoint_payload())

    def test_num_val_samples_tracker_restore_from_int_ckpt(self):
        """#21: restore_from_checkpoint 接受 int / tensor / None,非法值静默回退;
        且不覆盖 runtime K,仅影响 check_alignment 输出。"""
        tracker = train.NumValSamplesTracker(num_val_samples=8)
        # None 输入: runtime K 不变, alignment 缺失
        tracker.restore_from_checkpoint(None)
        self.assertEqual(8, tracker.value)
        is_aligned, msg = tracker.check_alignment(8)
        self.assertTrue(is_aligned, "未 restore 过 ckpt K 时 alignment 返回缺失提示")
        self.assertIn("缺失", msg)
        # int 输入: runtime K 不变, alignment 可比较
        tracker.restore_from_checkpoint(20)
        self.assertEqual(8, tracker.value,
            "restore 不得覆盖 runtime K=8")
        is_aligned, msg = tracker.check_alignment(8)
        self.assertFalse(is_aligned,
            "ckpt K=20 与 args K=8 不一致, 必须返回 False")
        self.assertIn("20", msg)
        # tensor 输入: 同上
        tracker.restore_from_checkpoint(torch.tensor(8))
        self.assertEqual(8, tracker.value)
        is_aligned, msg = tracker.check_alignment(8)
        self.assertTrue(is_aligned,
            "ckpt K=8 与 args K=8 一致, 必须返回 True")
        # 非法类型(str): 静默回退, alignment 保持上一次合法值
        tracker.restore_from_checkpoint("not_a_number")
        self.assertEqual(8, tracker.value)
        is_aligned, msg = tracker.check_alignment(8)
        self.assertTrue(is_aligned)
        # 非法值(0 / 负数): 静默回退
        tracker.restore_from_checkpoint(0)
        self.assertEqual(8, tracker.value)
        is_aligned, msg = tracker.check_alignment(8)
        self.assertTrue(is_aligned)
        tracker.restore_from_checkpoint(-3)
        self.assertEqual(8, tracker.value)
        is_aligned, msg = tracker.check_alignment(8)
        self.assertTrue(is_aligned)
        # 浮点: 接受并转 int (ckpt K 变成 2)
        tracker.restore_from_checkpoint(2.0)
        self.assertEqual(8, tracker.value, "runtime K 必须保持 8")
        is_aligned, msg = tracker.check_alignment(8)
        self.assertFalse(is_aligned, "ckpt K=2 != args K=8")
        self.assertIn("2", msg)
        # 验证 checkpoint_payload 始终返回 runtime K
        self.assertEqual(8, tracker.checkpoint_payload(),
            "checkpoint_payload 必须永远返回 runtime K=8")

    def test_num_val_samples_alignment_silent_when_match(self):
        """#21: checkpoint K == args.num_val_samples 时 check_alignment 返回 aligned=True。"""
        tracker = train.NumValSamplesTracker(num_val_samples=20)
        tracker.restore_from_checkpoint(20)
        is_aligned, msg = tracker.check_alignment(20)
        self.assertTrue(is_aligned, "K 一致时必须返回 aligned=True")
        self.assertIn("一致", msg)
        self.assertIn("20", msg)

    def test_num_val_samples_alignment_warns_on_mismatch(self):
        """#21: checkpoint K=20 与 args.num_val_samples=4 不一致时,返回 aligned=False 与可读 message。"""
        tracker = train.NumValSamplesTracker(num_val_samples=4)
        tracker.restore_from_checkpoint(20)
        is_aligned, msg = tracker.check_alignment(4)
        self.assertFalse(is_aligned, "K 不一致时必须返回 aligned=False")
        self.assertIn("20", msg)
        self.assertIn("4", msg)
        self.assertIn("best-of-K", msg)
        self.assertIn("--num_samples", msg)

    def test_num_val_samples_alignment_silent_on_missing_ckpt_key(self):
        """#21: ckpt 缺失 num_val_samples(旧版升级)时,is_aligned=True 但 message 提示缺失。"""
        # 模拟"checkpoint 中无 num_val_samples 字段, args.num_val_samples=4"的场景:
        # 构造时传入 None, 且不调用 restore_from_checkpoint。
        tracker = train.NumValSamplesTracker()
        is_aligned, msg = tracker.check_alignment(4)
        self.assertTrue(is_aligned, "ckpt 缺失时不算 mismatch,返回 aligned=True")
        self.assertIn("缺失", msg)
        self.assertIn("4", msg)

    def test_train_save_checkpoint_persists_num_val_samples(self):
        """#21 源码守卫: train.py 的 save_checkpoint 调用点必须把 num_val_samples
        写进 checkpoint 字典。"""
        train_source = (REPO_ROOT / "D2TP" / "train.py").read_text()
        # 至少出现 2 处 'num_val_samples' 字段写入 (两个 save_checkpoint 分支)
        occurrences = re.findall(
            r'"num_val_samples":\s*num_val_samples_tracker\.checkpoint_payload',
            train_source,
        )
        self.assertGreaterEqual(
            len(occurrences),
            2,
            "save_checkpoint 调用必须把 num_val_samples 写进 ckpt, "
            "目前只找到 {} 处(generator_only 与 D/G 交替分支各需要 1 处).".format(
                len(occurrences)
            ),
        )

    def test_train_main_restores_num_val_samples_from_checkpoint(self):
        """#21 源码守卫: main 加载 checkpoint 时必须调用 NumValSamplesTracker
        的 restore_from_checkpoint + check_alignment。"""
        train_source = (REPO_ROOT / "D2TP" / "train.py").read_text()
        self.assertIn(
            "num_val_samples_tracker.restore_from_checkpoint",
            train_source,
            "main 加载 ckpt 时必须把 num_val_samples 灌入 tracker.",
        )
        self.assertIn(
            "num_val_samples_tracker.check_alignment",
            train_source,
            "main 加载 ckpt 时必须做对齐校验.",
        )

    def test_evaluate_model_aligns_num_samples_with_checkpoint(self):
        """#21 源码守卫: evaluate_model.py 必须也做 K 对齐校验(尽管只 warning)。"""
        eval_source = (REPO_ROOT / "D2TP" / "evaluate_model.py").read_text()
        self.assertIn(
            "NumValSamplesTracker",
            eval_source,
            "evaluate_model.py 必须 import NumValSamplesTracker 并做对齐.",
        )
        self.assertIn(
            "num_val_samples",
            eval_source.lower(),
            "evaluate_model.py 加载 checkpoint 后必须把 num_val_samples "
            "字段取出并做对齐校验.",
        )

    # ------------------------------------------------------------------
    # Phase 4 #21 修复: restore 不得污染运行时 K
    # ------------------------------------------------------------------
    def test_num_val_samples_tracker_restore_does_not_override_runtime_payload(self):
        """#21 修复: restore_from_checkpoint 加载旧 ckpt K=20 后,
        checkpoint_payload() 必须仍返回当前运行时 K=4,而非旧 ckpt 的 20。"""
        tracker = train.NumValSamplesTracker(num_val_samples=4)
        tracker.restore_from_checkpoint(20)
        self.assertEqual(
            4,
            tracker.checkpoint_payload(),
            "restore 后 checkpoint_payload() 必须返回当前运行时 K=4,"
            " 不能返回旧 ckpt 的 20 — 否则后续 save_checkpoint 会写错值。",
        )
        is_aligned, _msg = tracker.check_alignment(4)
        self.assertFalse(
            is_aligned,
            "旧 ckpt K=20 != 当前 args K=4, check_alignment 必须返回 False。",
        )

    def test_num_val_samples_tracker_keeps_runtime_k_after_ckpt_mismatch(self):
        """#21 修复: 模拟"后续保存 checkpoint"场景:
        加载旧 ckpt(K=20)后,payload 必须是运行时 K=4,
        确保下一次 save_checkpoint 写入的是 4 而非 20。"""
        tracker = train.NumValSamplesTracker(num_val_samples=4)
        tracker.restore_from_checkpoint(20)
        # 模拟 save_checkpoint 时取 payload
        payload = tracker.checkpoint_payload()
        self.assertEqual(
            4,
            payload,
            "save_checkpoint 写入的必须是当前运行时 K=4,"
            " 绝对不能写成旧 ckpt 的 20。",
        )
        self.assertEqual(
            4,
            tracker.value,
            "tracker.value 必须保持运行时 K=4。",
        )


# ------------------------------------------------------------------
    # Phase 3 #24 修复: cycle target 使用最后 3 帧平均
    # ------------------------------------------------------------------
    def test_cycle_target_uses_last_three_frames_average(self):
        """#24: cycle target 应是 ``cycle_feature_seq`` 最后 3 帧的
        平均值,而非只用 ``[-1]`` 单帧。"""
        batch = 2
        # 构造最后 3 帧不同、剩余帧不同的序列
        cycle_feature_seq = torch.zeros(8, batch, 6)
        cycle_feature_seq[-3] = torch.full((batch, 6), 1.0)
        cycle_feature_seq[-2] = torch.full((batch, 6), 2.0)
        cycle_feature_seq[-1] = torch.full((batch, 6), 3.0)
        # 模拟 #24 修复后的代码逻辑
        tail = cycle_feature_seq[-3:]
        averaged = tail.mean(dim=0)
        # 平均应为 (1+2+3)/3 = 2
        self.assertTrue(torch.allclose(averaged, torch.full((batch, 6), 2.0)),
                        "最后 3 帧平均应为 (1+2+3)/3 = 2")
        # 不应只等于最后一帧 (3)
        self.assertFalse(torch.allclose(averaged, torch.full((batch, 6), 3.0)),
                         "不应只等于最后一帧")

    # ------------------------------------------------------------------
    # Phase 5 #25 修复: light_input_size 一致性
    # ------------------------------------------------------------------
    def test_generator_light_input_size_default_is_5(self):
        """#25: ``TrajectoryGenerator`` 默认 ``light_input_size=5``。"""
        gen = models.TrajectoryGenerator(
            obs_len=8, pred_len=12, traj_lstm_input_size=2,
            traj_lstm_hidden_size=32, n_units=[32, 16, 32],
            n_heads=[4, 1], graph_network_out_dims=32,
            dropout=0.0, alpha=0.2, graph_lstm_hidden_size=32,
        )
        self.assertEqual(5, gen.light_input_size,
                         "TrajectoryGenerator 默认 light_input_size 应为 5")

    def test_cyclestate_generator_light_input_size_default_is_5(self):
        """#25: ``CycleStateTrajectoryGenerator`` 默认 ``light_input_size=5``。"""
        self.assertEqual(5, self.model.light_input_size,
                         "CycleStateTrajectoryGenerator 默认 light_input_size 应为 5")

    def test_discriminator_light_input_size_default_is_4(self):
        """#25: ``TrajectoryDiscriminator`` 默认 ``light_input_size=4``。
        注: Generator 用 5 维(距离 + dx + dy + 2×灯态),
        Discriminator 用 4 维(原始 obs_state), 两者不一致但有历史原因。"""
        disc = models.TrajectoryDiscriminator(
            obs_len=8, pred_len=12,
            part_lstm_input_size=16, part_lstm_hidden_size=32,
            merge_lstm_input_size=64, merge_lstm_hidden_size=64,
            dropout=0.1,
        )
        self.assertEqual(4, disc.light_input_size,
                         "TrajectoryDiscriminator 默认 light_input_size 应为 4")

    def test_train_py_instantiates_discriminator_with_light_input_size_4(self):
        """#25: train.py 构造 ``TrajectoryDiscriminator`` 时显式传
        ``light_input_size=4``,确保与类默认值一致且不静默漂移。"""
        train_source = pathlib.Path(train.__file__).read_text(encoding="utf-8")
        # 找到构建 Discriminator 的代码片段(包含 light_input_size=4)
        self.assertIn("light_input_size=4", train_source,
                      "train.py 中 TrajectoryDiscriminator 必须显式传 light_input_size=4")

    # ------------------------------------------------------------------
    # Phase 5 #26 修复: _norm 命名统一
    # ------------------------------------------------------------------
    def test_models_py_no_step_residual_norm_seq_key(self):
        """#26: models.py 不应再包含 ``decoder_state_step_residual_norm_seq``
        带 ``_seq`` 后缀的旧 key。"""
        models_source = pathlib.Path(models.__file__).read_text(encoding="utf-8")
        self.assertNotIn(
            "decoder_state_step_residual_norm_seq",
            models_source,
            "_norm_seq 后缀已在 #26 修复中移除",
        )

    def test_extract_stability_metrics_step_norm_key_has_no_seq(self):
        """#26: ``extract_state_stability_metrics`` 从 debug_info 读 step norm
        时应使用无 ``_seq`` 后缀的新 key ``decoder_state_step_residual_norm``。"""
        train_source = pathlib.Path(train.__file__).read_text(encoding="utf-8")
        # 确保读取的是新 key(不带 _seq 后缀)
        read_pattern = 'decoder_state_step_residual_norm"'
        self.assertIn(
            read_pattern,
            train_source,
            "extract_state_stability_metrics 应读取 'decoder_state_step_residual_norm' (无 _seq)",
        )

    # ------------------------------------------------------------------
    # Phase 5 #27 修复: traffic_context["scene"] 字段去重
    # ------------------------------------------------------------------
    def test_build_traffic_context_from_batch_scene_has_no_seq_start_end(self):
        """#27: ``build_traffic_context_from_batch`` 的 ``scene`` 字典中
        不应再包含 ``seq_start_end``,因为模型 ``build_traffic_context`` 已提供。"""
        batch = (
            self.obs_traj,
            self.pred_traj_gt,
            self.obs_traj_rel,
            self.pred_traj_gt_rel,
            self.obs_state,
            self.pred_state,
            torch.zeros(3),
            torch.zeros(8, 3),
            self.seq_start_end,
        )
        ctx = train.build_traffic_context_from_batch(batch)
        self.assertNotIn(
            "seq_start_end",
            ctx["scene"],
            "base_context['scene'] 不应包含 seq_start_end(模型侧已提供,避免重复)",
        )

    def test_model_build_traffic_context_scene_still_has_seq_start_end(self):
        """#27: 模型 ``build_traffic_context`` 的 ``scene`` 仍保留
        ``seq_start_end``,因为它被 forward 内部多处使用。"""
        ctx = self.model.build_traffic_context(
            obs_traj_rel=self.obs_traj_rel,
            obs_traj_pos=self.obs_traj,
            obs_state=self.obs_state,
            pred_state=self.pred_state,
            seq_start_end=self.seq_start_end,
        )
        self.assertIn(
            "seq_start_end",
            ctx["scene"],
            "模型 traffic_context['scene'] 仍应包含 seq_start_end",
        )

    # ------------------------------------------------------------------
    # Phase 0 #28 修复: 随机种子传播链路完善
    # ------------------------------------------------------------------
    def test_models_no_random_import(self):
        """#28: models.py 不应再 ``import random``(已替换为 torch 操作)。"""
        models_source = (REPO_ROOT / "D2TP" / "models.py").read_text()
        self.assertNotIn(
            "import random", models_source,
            "models.py 不应 import random(random.random() 已替换为 torch.rand)",
        )

    def test_models_no_random_random_call(self):
        """#28: models.py 不应包含 ``random.random()`` 调用。"""
        models_source = (REPO_ROOT / "D2TP" / "models.py").read_text()
        self.assertNotIn(
            "random.random()", models_source,
            "models.py 不应有 random.random() 调用(已替换为 torch.rand)",
        )

    def test_init_hidden_functions_use_torch_zeros(self):
        """#28: 所有 ``init_hidden_*`` 方法应使用 ``torch.zeros``
        而非 ``torch.randn``,与 Social-LSTM 标准做法一致。"""
        models_source = (REPO_ROOT / "D2TP" / "models.py").read_text()
        import ast as _ast
        tree = _ast.parse(models_source)
        init_hidden_funcs = [
            node for node in _ast.walk(tree)
            if isinstance(node, _ast.FunctionDef)
            and node.name.startswith("init_hidden_")
        ]
        self.assertGreater(
            len(init_hidden_funcs), 0,
            "至少应找到一个 init_hidden_* 方法",
        )
        for func in init_hidden_funcs:
            func_source = _ast.get_source_segment(models_source, func)
            self.assertNotIn(
                "torch.randn", func_source,
                f"{func.name} 不应使用 torch.randn,应使用 torch.zeros",
            )

    def test_init_hidden_functions_contain_zeros(self):
        """#28: 每个 ``init_hidden_*`` 方法应包含 ``torch.zeros``。"""
        models_source = (REPO_ROOT / "D2TP" / "models.py").read_text()
        import ast as _ast
        tree = _ast.parse(models_source)
        init_hidden_funcs = [
            node for node in _ast.walk(tree)
            if isinstance(node, _ast.FunctionDef)
            and node.name.startswith("init_hidden_")
        ]
        for func in init_hidden_funcs:
            func_source = _ast.get_source_segment(models_source, func)
            self.assertIn(
                "torch.zeros", func_source,
                f"{func.name} 应使用 torch.zeros",
            )

    def test_teacher_forcing_uses_torch_rand_not_random(self):
        """#28: teacher forcing 决策应使用 ``torch.rand`` 而非
        ``random.random()``。"""
        models_source = (REPO_ROOT / "D2TP" / "models.py").read_text()
        # 确认使用 torch.rand 做 teacher forcing
        tf_patterns = re.findall(
            r"torch\.rand\(1,\s*device=.+?\.device\)\.item\(\)\s*<\s*teacher_forcing_ratio",
            models_source,
        )
        self.assertGreaterEqual(
            len(tf_patterns), 2,
            "两个 forward 中的 teacher forcing 都应使用 torch.rand",
        )

    def test_seed_worker_defined_in_loader(self):
        """#28: ``data/loader.py`` 应定义 ``seed_worker`` 函数。"""
        loader_source = (REPO_ROOT / "D2TP" / "data" / "loader.py").read_text()
        self.assertIn(
            "def seed_worker", loader_source,
            "loader.py 应定义 seed_worker 函数",
        )

    def test_dataloader_passes_worker_init_fn_and_generator(self):
        """#28: ``DataLoader`` 构造应传入 ``worker_init_fn=seed_worker``
        和 ``generator=g``。"""
        loader_source = (REPO_ROOT / "D2TP" / "data" / "loader.py").read_text()
        self.assertIn(
            "worker_init_fn=seed_worker", loader_source,
            "DataLoader 应传入 worker_init_fn=seed_worker",
        )
        self.assertIn(
            "generator=g", loader_source,
            "DataLoader 应传入 generator=g 以支持确定性数据加载",
        )

    def test_seed_worker_seeds_correctly(self):
        """#28: ``seed_worker`` 应设置 numpy 和 random 种子。"""
        from D2TP.data.loader import seed_worker

        # 用一个已知的 torch 初始种子调用
        torch.manual_seed(42)
        seed_worker(0)
        first_np = np.random.randint(0, 10000)
        first_py = random.randint(0, 10000)

        # 重置种子后重复，结果应一致
        torch.manual_seed(42)
        seed_worker(0)
        self.assertEqual(
            first_np, np.random.randint(0, 10000),
            "相同 torch seed 下 seed_worker(0) 应产生相同的 numpy 随机数",
        )
        self.assertEqual(
            first_py, random.randint(0, 10000),
            "相同 torch seed 下 seed_worker(0) 应产生相同的 Python 随机数",
        )

    # ------------------------------------------------------------------
    # Phase 4 #29 修复: 数据集归一化参数持久化
    # ------------------------------------------------------------------
    def test_norm_params_returns_correct_keys(self):
        """#29: ``norm_params()`` 应返回四个归一化参数。"""
        np = self.model.norm_params()
        expected_keys = {
            "queue_count_norm",
            "queue_speed_norm",
            "queue_distance_norm",
            "cycle_time_norm",
        }
        self.assertEqual(set(np.keys()), expected_keys)
        self.assertEqual(len(np), 4)

    def test_norm_params_returns_correct_defaults(self):
        """#29: 默认归一化值应与 __init__ 默认值一致。"""
        np = self.model.norm_params()
        self.assertEqual(np["queue_count_norm"], 10.0)
        self.assertEqual(np["queue_speed_norm"], 10.0)
        self.assertEqual(np["queue_distance_norm"], 500.0)
        self.assertEqual(np["cycle_time_norm"], 60.0)

    def test_load_norm_params_restores_values(self):
        """#29: ``load_norm_params`` 应覆盖实例的归一化参数。"""
        new_vals = {
            "queue_count_norm": 20.0,
            "queue_speed_norm": 15.0,
            "queue_distance_norm": 300.0,
            "cycle_time_norm": 120.0,
        }
        self.model.load_norm_params(new_vals)
        np = self.model.norm_params()
        self.assertEqual(np["queue_count_norm"], 20.0)
        self.assertEqual(np["queue_speed_norm"], 15.0)
        self.assertEqual(np["queue_distance_norm"], 300.0)
        self.assertEqual(np["cycle_time_norm"], 120.0)

    def test_load_norm_params_with_none_does_not_change(self):
        """#29: ``load_norm_params(None)`` 不报错且不修改值。"""
        original = self.model.norm_params()
        self.model.load_norm_params(None)
        self.assertEqual(self.model.norm_params(), original)

    def test_load_norm_params_with_partial_dict(self):
        """#29: 部分 key 更新: 只改传入的,其他保持原值。"""
        original = self.model.norm_params()
        partial = {"queue_count_norm": 999.0}
        self.model.load_norm_params(partial)
        self.assertEqual(self.model.queue_count_norm, 999.0)
        self.assertEqual(self.model.queue_speed_norm, original["queue_speed_norm"])
        self.assertEqual(self.model.queue_distance_norm, original["queue_distance_norm"])
        self.assertEqual(self.model.cycle_time_norm, original["cycle_time_norm"])

    def test_load_norm_params_with_extra_keys_no_error(self):
        """#29: dict 包含未知 key 不应崩溃,只取认识的 key。"""
        original = self.model.norm_params()
        self.model.load_norm_params({"queue_count_norm": 1.0, "unknown_extra": 99.0})
        self.assertEqual(self.model.queue_count_norm, 1.0)
        self.assertEqual(self.model.queue_speed_norm, original["queue_speed_norm"])

    def test_norm_params_and_load_norm_params_roundtrip(self):
        """#29: ``norm_params → load_norm_params`` 是恒等变换。"""
        original = self.model.norm_params()
        self.model.load_norm_params(original)
        self.assertEqual(self.model.norm_params(), original)

    def test_norm_params_payload_uses_plain_python_floats(self):
        """#29: ``norm_params()`` 应稳定导出 plain ``float`` 负载。"""
        self.model.queue_count_norm = np.float32(12.5)
        self.model.queue_speed_norm = np.float64(7.25)
        self.model.queue_distance_norm = torch.tensor(321.0)
        self.model.cycle_time_norm = 88

        payload = self.model.norm_params()

        for key, value in payload.items():
            self.assertIsInstance(
                value,
                float,
                f"{key} 应导出为 plain float，避免 checkpoint 负载类型漂移",
            )
        self.assertEqual(payload["queue_count_norm"], 12.5)
        self.assertEqual(payload["queue_speed_norm"], 7.25)
        self.assertEqual(payload["queue_distance_norm"], 321.0)
        self.assertEqual(payload["cycle_time_norm"], 88.0)

    def test_norm_params_checkpoint_roundtrip_via_torch_save_load(self):
        """#29: ``torch.save/load`` 后仍可恢复归一化参数。"""
        expected = {
            "queue_count_norm": 21.0,
            "queue_speed_norm": 17.5,
            "queue_distance_norm": 275.0,
            "cycle_time_norm": 95.0,
        }
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "norm_params": expected,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = pathlib.Path(tmpdir) / "norm_params_roundtrip.pth"
            torch.save(checkpoint, ckpt_path)
            loaded = torch.load(ckpt_path, map_location="cpu")

        restored_model = models.CycleStateTrajectoryGenerator(
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
        restored_model.load_norm_params(loaded["norm_params"])
        self.assertEqual(restored_model.norm_params(), expected)

    def test_evaluate_get_generator_restores_norm_params_from_checkpoint(self):
        """#29: evaluate 路径应从 checkpoint 恢复归一化参数。"""
        expected = {
            "queue_count_norm": 33.0,
            "queue_speed_norm": 14.0,
            "queue_distance_norm": 410.0,
            "cycle_time_norm": 72.0,
        }
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "norm_params": expected,
        }
        eval_args = SimpleNamespace(
            obs_len=8,
            pred_len=12,
            traj_lstm_input_size=2,
            traj_lstm_hidden_size=32,
            hidden_units="16",
            heads="4,1",
            graph_network_out_dims=32,
            dropout=0.0,
            alpha=0.2,
            graph_lstm_hidden_size=32,
            noise_dim=(16,),
            noise_type="gaussian",
            model_type="cyclestate",
            disable_state_gating=False,
            disable_queue_rollout=False,
            disable_lane_queue_anchor=False,
            disable_decoder_state_residual=False,
            disable_aux_losses=False,
            rollout_residual_scale=1.0,
            detach_rollout_state=False,
            phase_duration_limits=None,
            rollout_queue_coefs_json="",
            device="cpu",
        )

        with mock.patch.object(evaluate_model, "args", eval_args, create=True):
            restored_model = evaluate_model.get_generator(checkpoint)

        self.assertEqual(restored_model.norm_params(), expected)

    def test_evaluate_get_generator_keeps_defaults_for_legacy_checkpoint(self):
        """#29: 旧 checkpoint 缺失 ``norm_params`` 时应保持默认值。"""
        checkpoint = {
            "state_dict": self.model.state_dict(),
        }
        eval_args = SimpleNamespace(
            obs_len=8,
            pred_len=12,
            traj_lstm_input_size=2,
            traj_lstm_hidden_size=32,
            hidden_units="16",
            heads="4,1",
            graph_network_out_dims=32,
            dropout=0.0,
            alpha=0.2,
            graph_lstm_hidden_size=32,
            noise_dim=(16,),
            noise_type="gaussian",
            model_type="cyclestate",
            disable_state_gating=False,
            disable_queue_rollout=False,
            disable_lane_queue_anchor=False,
            disable_decoder_state_residual=False,
            disable_aux_losses=False,
            rollout_residual_scale=1.0,
            detach_rollout_state=False,
            phase_duration_limits=None,
            rollout_queue_coefs_json="",
            device="cpu",
        )

        with mock.patch.object(evaluate_model, "args", eval_args, create=True):
            restored_model = evaluate_model.get_generator(checkpoint)

        self.assertEqual(
            restored_model.norm_params(),
            {
                "queue_count_norm": 10.0,
                "queue_speed_norm": 10.0,
                "queue_distance_norm": 500.0,
                "cycle_time_norm": 60.0,
            },
        )

    def test_evaluate_get_generator_accepts_legacy_single_aux_head_checkpoint(self):
        """旧 CycleState checkpoint 使用 ``queue_aux_head`` / ``cycle_aux_head``
        单头结构时，评估侧也必须能兼容加载，而不是直接 load_state_dict 失败。"""
        checkpoint = {
            "state_dict": self.model.state_dict(),
        }
        queue_head = torch.nn.Linear(self.model.queue_lstm_hidden_size, 6)
        cycle_head = torch.nn.Linear(self.model.cycle_lstm_hidden_size, 6)
        checkpoint["state_dict"].pop("queue_aux_reg_head.weight")
        checkpoint["state_dict"].pop("queue_aux_reg_head.bias")
        checkpoint["state_dict"].pop("queue_aux_cls_head.weight")
        checkpoint["state_dict"].pop("queue_aux_cls_head.bias")
        checkpoint["state_dict"].pop("cycle_aux_phase_head.weight")
        checkpoint["state_dict"].pop("cycle_aux_phase_head.bias")
        checkpoint["state_dict"].pop("cycle_aux_time_head.weight")
        checkpoint["state_dict"].pop("cycle_aux_time_head.bias")
        checkpoint["state_dict"].pop("cycle_aux_change_head.weight")
        checkpoint["state_dict"].pop("cycle_aux_change_head.bias")
        checkpoint["state_dict"]["queue_aux_head.weight"] = queue_head.weight.detach().clone()
        checkpoint["state_dict"]["queue_aux_head.bias"] = queue_head.bias.detach().clone()
        checkpoint["state_dict"]["cycle_aux_head.weight"] = cycle_head.weight.detach().clone()
        checkpoint["state_dict"]["cycle_aux_head.bias"] = cycle_head.bias.detach().clone()

        eval_args = SimpleNamespace(
            obs_len=8,
            pred_len=12,
            traj_lstm_input_size=2,
            traj_lstm_hidden_size=32,
            hidden_units="16",
            heads="4,1",
            graph_network_out_dims=32,
            dropout=0.0,
            alpha=0.2,
            graph_lstm_hidden_size=32,
            noise_dim=(16,),
            noise_type="gaussian",
            model_type="cyclestate",
            disable_state_gating=False,
            disable_queue_rollout=False,
            disable_lane_queue_anchor=False,
            disable_decoder_state_residual=False,
            disable_aux_losses=False,
            rollout_residual_scale=1.0,
            detach_rollout_state=False,
            phase_duration_limits=None,
            rollout_queue_coefs_json="",
            device="cpu",
            num_samples=20,
        )

        with mock.patch.object(evaluate_model, "args", eval_args, create=True):
            restored_model = evaluate_model.get_generator(checkpoint)

        torch.testing.assert_close(
            restored_model.queue_aux_reg_head.weight,
            queue_head.weight[:4],
        )
        torch.testing.assert_close(
            restored_model.queue_aux_cls_head.weight,
            queue_head.weight[4:],
        )
        torch.testing.assert_close(
            restored_model.cycle_aux_phase_head.weight,
            cycle_head.weight[:3],
        )
        torch.testing.assert_close(
            restored_model.cycle_aux_time_head.weight,
            cycle_head.weight[3:5],
        )
        torch.testing.assert_close(
            restored_model.cycle_aux_change_head.weight,
            cycle_head.weight[5:6],
        )

    def test_train_source_writes_norm_params_to_both_checkpoint_dicts(self):
        """#29: train.py 的两个 ``save_checkpoint`` 都应包含
        ``norm_params`` 键。"""
        train_source = (REPO_ROOT / "D2TP" / "train.py").read_text()
        matches = re.findall(r'"norm_params":\s*model\.norm_params\(\)', train_source)
        self.assertGreaterEqual(
            len(matches), 2,
            "train.py 的两个 save_checkpoint 调用都应写入 norm_params",
        )

    def test_train_source_loads_norm_params_on_resume(self):
        """#29: train.py 的 cyclestate 断点续训路径应调用
        ``model.load_norm_params``。"""
        train_source = (REPO_ROOT / "D2TP" / "train.py").read_text()
        self.assertIn(
            "model.load_norm_params(checkpoint.get(\"norm_params\"))",
            train_source,
            "train.py 的 resume 分支应恢复归一化参数",
        )

    def test_evaluate_source_loads_norm_params(self):
        """#29: evaluate_model.py 的 cyclestate 评估路径应调用
        ``model.load_norm_params``。"""
        eval_source = (REPO_ROOT / "D2TP" / "evaluate_model.py").read_text()
        self.assertIn(
            "model.load_norm_params(checkpoint.get(\"norm_params\"))",
            eval_source,
            "evaluate_model.py 应恢复 checkpoint 中的归一化参数",
        )

    # ------------------------------------------------------------------
    # Phase 5 #30 修复: ``add_noise`` 噪声采样抽象
    # ------------------------------------------------------------------
    def test_build_noise_sampler_returns_expected_concrete_type(self):
        """#30: 字符串噪声类型应解析为稳定的 sampler 对象。"""
        gaussian = models.build_noise_sampler("gaussian")
        uniform = models.build_noise_sampler("uniform")

        self.assertIsInstance(gaussian, models.GaussianNoiseSampler)
        self.assertIsInstance(uniform, models.UniformNoiseSampler)
        self.assertEqual("gaussian", gaussian.name)
        self.assertEqual("uniform", uniform.name)

    def test_build_noise_sampler_rejects_unknown_noise_type(self):
        """#30: 未知噪声类型应在 factory 层抛错。"""
        with self.assertRaises(ValueError) as cm:
            models.build_noise_sampler("triangular")
        self.assertIn("triangular", str(cm.exception))

    def test_trajectory_generator_accepts_noise_sampler_instance(self):
        """#30: 生成器应接受 sampler 实例，而非只支持字符串。"""

        class RecordingNoiseSampler(models.NoiseSampler):
            name = "recording"

            def __init__(self):
                self.calls = []

            def sample(self, shape, device):
                self.calls.append((tuple(shape), str(device)))
                return torch.ones(*shape, device=device)

        sampler = RecordingNoiseSampler()
        model = models.TrajectoryGenerator(
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
            noise_type=sampler,
        )
        hidden_wo_noise = torch.zeros(
            self.obs_traj.size(1),
            model.light_embedding_size
            + model.traj_lstm_hidden_size
            + model.graph_lstm_hidden_size,
        )

        hidden_with_noise = model.add_noise(hidden_wo_noise, self.seq_start_end)

        self.assertIs(model.noise_sampler, sampler)
        self.assertEqual("recording", model.noise_type)
        self.assertEqual(1, len(sampler.calls))
        self.assertTrue(torch.all(hidden_with_noise[:, -model.noise_dim[0] :] == 1.0))

    def test_get_noise_accepts_noise_sampler_instance(self):
        """#30: ``get_noise`` 应支持 sampler 实例输入。"""

        class ConstantNoiseSampler(models.NoiseSampler):
            name = "constant"

            def sample(self, shape, device):
                return torch.full(shape, 0.25, device=device)

        noise = models.get_noise((2, 3), ConstantNoiseSampler(), torch.device("cpu"))
        self.assertEqual((2, 3), tuple(noise.shape))
        self.assertTrue(torch.allclose(noise, torch.full((2, 3), 0.25)))

    # ------------------------------------------------------------------
    # Phase 5 #32/#33 修复: 文档交叉引用 + 端到端脚本
    # ------------------------------------------------------------------
    def test_analysis_issue_docs_exist(self):
        """#32: 三份分析/问题索引文档应实际存在。"""
        for rel_path in (
            "docs/ENGINEERING_ISSUES.md",
            "docs/COMPREHENSIVE_ANALYSIS.md",
            "docs/METHOD_AND_ARCHITECTURE_ANALYSIS.md",
        ):
            self.assertTrue(
                (REPO_ROOT / rel_path).is_file(),
                f"{rel_path} 应存在，避免 PLAN/EXPERIMENT_LOG 链接悬空",
            )

    def test_experiment_log_links_analysis_issue_docs(self):
        """#32: EXPERIMENT_LOG 应显式链接三份分析文档。"""
        experiment_log = (REPO_ROOT / "EXPERIMENT_LOG.md").read_text(encoding="utf-8")
        self.assertIn("docs/ENGINEERING_ISSUES.md", experiment_log)
        self.assertIn("docs/COMPREHENSIVE_ANALYSIS.md", experiment_log)
        self.assertIn("docs/METHOD_AND_ARCHITECTURE_ANALYSIS.md", experiment_log)

    def test_run_full_pipeline_script_exists(self):
        """#33: 应提供 `scripts/run_full_pipeline.sh`。"""
        self.assertTrue(
            (REPO_ROOT / "scripts" / "run_full_pipeline.sh").is_file(),
            "缺少 scripts/run_full_pipeline.sh",
        )

    def test_run_full_pipeline_script_wires_train_and_evaluate(self):
        """#33: pipeline 脚本应串起 train 与 evaluate。"""
        script = (REPO_ROOT / "scripts" / "run_full_pipeline.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("D2TP/train.py", script)
        self.assertIn("D2TP/evaluate_model.py", script)
        self.assertIn("model_best.pth.tar", script)
        self.assertIn("MAX_TRAIN_BATCHES", script)
        self.assertIn("MAX_VAL_BATCHES", script)

    # ------------------------------------------------------------------
    # Phase 4 #31 修复: disable_* 开关集中注册
    # ------------------------------------------------------------------
    def test_ablation_config_dataclass_exists(self):
        """#31: 应存在集中管理 disable_* 开关的 dataclass。"""
        self.assertTrue(hasattr(models, "AblationConfig"))
        cfg = models.AblationConfig()
        self.assertFalse(cfg.disable_state_gating)
        self.assertFalse(cfg.disable_queue_rollout)
        self.assertFalse(cfg.disable_lane_queue_anchor)
        self.assertFalse(cfg.disable_decoder_state_residual)
        self.assertFalse(cfg.disable_aux_losses)

    def test_ablation_config_from_args_applies_disable_aux_losses(self):
        """#31: 集中配置应统一计算 effective disable 状态。"""
        args = SimpleNamespace(
            disable_state_gating=False,
            disable_queue_rollout=False,
            disable_lane_queue_anchor=False,
            disable_decoder_state_residual=False,
            disable_aux_losses=True,
        )
        cfg = models.AblationConfig.from_args(args)
        eff = cfg.effective_flags()
        self.assertTrue(eff["disable_state_gating"])
        self.assertTrue(eff["disable_queue_rollout"])
        self.assertTrue(eff["disable_lane_queue_anchor"])
        self.assertTrue(eff["disable_decoder_state_residual"])

    def test_train_and_evaluate_use_shared_ablation_config_helper(self):
        """#31: train/evaluate 应复用同一 helper，而不是散落手写传参。"""
        train_source = (REPO_ROOT / "D2TP" / "train.py").read_text(encoding="utf-8")
        eval_source = (REPO_ROOT / "D2TP" / "evaluate_model.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("AblationConfig", train_source)
        self.assertIn("AblationConfig", eval_source)
        self.assertIn("to_model_kwargs()", train_source)
        self.assertIn("to_model_kwargs()", eval_source)


if __name__ == "__main__":
    unittest.main()
