#!/usr/bin/env python
"""调试脚本：单 batch 前向，打印模型从输入到输出各环节数据形状和统计。"""

import argparse
import os
import sys
import numpy as np
import torch

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "D2TP"))

from data.loader import data_loader
from models import CycleStateTrajectoryGenerator, RolloutQueueCoefs, apply_rollout_coefs_override
from utils import int_tuple, get_dset_path


def describe(name, tensor_or_list, max_elems=8):
    """打印 tensor 或 list[tensor] 的形状和统计信息。"""
    if isinstance(tensor_or_list, (list, tuple)):
        if len(tensor_or_list) == 0:
            print(f"  [EMPTY LIST] {name}")
            return
        shapes = []
        for i, t in enumerate(tensor_or_list):
            if t is None:
                shapes.append(f"[{i}]=None")
                continue
            s = tuple(t.shape)
            shapes.append(str(s))
            first_shown = i < 3 or i >= len(tensor_or_list) - 2
            if first_shown:
                flat = t.detach().float()
                print(f"  {name}[{i}/{len(tensor_or_list)}]  shape={s}  "
                      f"mean={flat.mean().item():.6f}  std={flat.std().item():.6f}  "
                      f"min={flat.min().item():.4f}  max={flat.max().item():.4f}")
            elif i == 3:
                print(f"  ... ({len(tensor_or_list) - 5} entries skipped) ...")
        if len(shapes) <= 6:
            print(f"  SHAPES summary for {name}: {shapes}")
    else:
        t = tensor_or_list
        if t is None:
            print(f"  [NONE] {name}")
            return
        s = tuple(t.shape)
        flat = t.detach().float()
        print(f"  {name}  shape={s}  "
              f"mean={flat.mean().item():.6f}  std={flat.std().item():.6f}  "
              f"min={flat.min().item():.4f}  max={flat.max().item():.4f}")
        if flat.numel() <= max_elems:
            print(f"    values={flat.tolist()}")
        else:
            print(f"    first_{max_elems}={flat.flatten()[:max_elems].tolist()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs_len", default=8, type=int)
    parser.add_argument("--pred_len", default=12, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--skip", default=1, type=int)
    parser.add_argument("--seed", default=72, type=int)
    parser.add_argument("--dataset_name", default="VTP_C", type=str)
    parser.add_argument("--loader_num_workers", default=0, type=int)
    parser.add_argument("--delim", default="\t")
    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--train_stage", default="warmup", type=str)
    parser.add_argument("--use_gpu", default=0, type=int)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. 加载数据
    train_path = get_dset_path(args.dataset_name, "train")
    _, loader = data_loader(args, train_path)
    batch = next(iter(loader))
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

    device = torch.device(args.device)
    obs_traj = obs_traj.to(device)
    pred_traj_gt = pred_traj_gt.to(device)
    obs_traj_rel = obs_traj_rel.to(device)
    pred_traj_gt_rel = pred_traj_gt_rel.to(device)
    obs_state = obs_state.to(device)
    pred_state = pred_state.to(device)

    # 训练时的 generator_input = cat(obs_traj_rel, pred_traj_gt_rel)
    generator_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)

    print("=" * 70)
    print("1. 输入数据概况")
    print("=" * 70)
    describe("obs_traj      (观测期绝对轨迹)", obs_traj)
    describe("pred_traj_gt  (预测期绝对轨迹 GT)", pred_traj_gt)
    describe("obs_traj_rel  (观测期相对轨迹)", obs_traj_rel)
    describe("pred_traj_gt_rel (预测期相对轨迹 GT)", pred_traj_gt_rel)
    describe("obs_state     (观测期信号灯状态)", obs_state)
    describe("pred_state    (预测期信号灯状态)", pred_state)
    describe("generator_input (cat obs+pred rel)", generator_input)
    print(f"  seq_start_end shape={seq_start_end.shape}")
    print(f"  seq_start_end[:5]={seq_start_end[:5].tolist()}")
    print(f"  共 {seq_start_end.shape[0]} 个场景")
    print()

    # 2. 构建模型
    print("=" * 70)
    print("2. 模型结构")
    print("=" * 70)
    model = CycleStateTrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        traj_lstm_input_size=2,
        traj_lstm_hidden_size=32,
        n_units=(16,),
        n_heads=(4, 1),
        graph_network_out_dims=32,
        dropout=0.0,
        alpha=0.2,
        graph_lstm_hidden_size=32,
        noise_dim=(16,),
        noise_type="gaussian",
        light_input_size=5,
        embedding_size=64,
        light_embedding_size=32,
        queue_lstm_hidden_size=32,
        cycle_lstm_hidden_size=16,
        disable_state_gating=False,
        disable_queue_rollout=False,
        disable_lane_queue_anchor=False,
        disable_decoder_state_residual=False,
        rollout_residual_scale=1.0,
        detach_rollout_state=False,
    ).to(device).eval()
    model.training = True  # 走训练分支（有 teacher forcing）

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    for name, param in model.named_parameters():
        s = tuple(param.shape)
        print(f"    {name:55s}  shape={str(s):20s}  requires_grad={param.requires_grad}")
    print()

    # 3. 构建 traffic_context（预处理）
    print("=" * 70)
    print("3. traffic_context（中观/信号特征）")
    print("=" * 70)
    traffic_context = model.build_traffic_context(
        generator_input, obs_traj, obs_state, pred_state, seq_start_end
    )
    print("  === meso (中观队列特征) ===")
    describe("queue_feature_seq (观测期: 观测步+预测步)", traffic_context["meso"]["queue_feature_seq"])
    describe("lane_queue_anchor_seq", traffic_context["meso"]["lane_queue_anchor_seq"])
    describe("queue_targets", traffic_context["meso"]["queue_targets"])
    print("  === signal (信号周期特征) ===")
    describe("cycle_feature_seq", traffic_context["signal"]["cycle_feature_seq"])
    print("  === agent (单车特征) ===")
    describe("lane_ids", traffic_context["agent"]["lane_ids"])
    describe("obs_traj (abs)", traffic_context["agent"]["obs_traj"])
    describe("direction", traffic_context["agent"]["direction"])
    print()

    # 4. 前向传播
    print("=" * 70)
    print("4. 前向传播各阶段")
    print("=" * 70)

    batch = generator_input.shape[1]
    traj_lstm_h_t, traj_lstm_c_t = model.init_hidden_traj_lstm(batch)
    queue_lstm_h_t, queue_lstm_c_t = model.init_hidden_queue_lstm(batch)
    cycle_lstm_h_t, cycle_lstm_c_t = model.init_hidden_cycle_lstm(batch)

    describe("init_traj_lstm_hidden", traj_lstm_h_t)
    describe("init_queue_lstm_hidden", queue_lstm_h_t)
    describe("init_cycle_lstm_hidden", cycle_lstm_h_t)
    print()

    # --- 观测期: traj LSTM ---
    print("--- 4a. 观测期 traj LSTM (逐步编码轨迹) ---")
    traj_lstm_hidden_states = []
    for input_t in generator_input[:args.obs_len].chunk(args.obs_len, dim=0):
        inputtraj = input_t[:, :, 2:4]
        traj_lstm_h_t, traj_lstm_c_t = model.traj_lstm_model(
            inputtraj.squeeze(0), (traj_lstm_h_t, traj_lstm_c_t)
        )
        traj_lstm_hidden_states.append(traj_lstm_h_t)
    describe("traj_lstm_hidden_states", traj_lstm_hidden_states)
    print()

    # --- 观测期: GAT + SeqGAT ---
    print("--- 4b. 观测期 GAT + SeqGAT (图编码) ---")
    obs_dire = obs_traj[:, :, 0:6]
    obs_dire[:, :, 5] = obs_traj[:, :, 9]
    graph_lstm_input = model.gatencoder(
        torch.stack(traj_lstm_hidden_states), seq_start_end, obs_dire
    )
    describe("gatencoder output", graph_lstm_input)

    graph_lstm_hidden_states = []
    staend = torch.zeros((1, 2), dtype=torch.int, device=device)
    for j in range(args.obs_len):
        if j <= 6:
            staend[0, 1] = j + 1
            graph_inter_input = model.seqgatencoder(
                graph_lstm_input[0:(j + 1)].permute(1, 0, 2), staend
            )
        else:
            staend[0, 1] = 7
            graph_inter_input = model.seqgatencoder(
                graph_lstm_input[(j - 6):(j + 1)].permute(1, 0, 2), staend,
            )
        graph_lstm_hidden_states.append(graph_inter_input[:, -1, :])
    describe("graph_lstm_hidden_states", graph_lstm_hidden_states)
    print()

    # --- 观测期: queue LSTM ---
    print("--- 4c. 观测期 queue LSTM (中观队列记忆) ---")
    queue_feature_seq = traffic_context["meso"]["queue_feature_seq"]
    queue_lstm_hidden_states = []
    for t in range(args.obs_len):
        queue_embed = model.queue_feature_embedding(queue_feature_seq[t])
        queue_lstm_h_t, queue_lstm_c_t = model.queue_lstm_model(
            queue_embed, (queue_lstm_h_t, queue_lstm_c_t)
        )
        queue_lstm_hidden_states.append(queue_lstm_h_t)
    describe("queue_lstm_hidden_states", queue_lstm_hidden_states)
    describe("queue_last (观测窗口最后一帧)", queue_lstm_hidden_states[-1])
    print()

    # --- 观测期: cycle LSTM ---
    print("--- 4d. 观测期 cycle LSTM (信号周期记忆) ---")
    cycle_feature_seq = traffic_context["signal"]["cycle_feature_seq"]
    cycle_lstm_hidden_states = []
    for t in range(args.obs_len):
        cycle_embed = model.cycle_feature_embedding(cycle_feature_seq[t])
        cycle_lstm_h_t, cycle_lstm_c_t = model.cycle_lstm_model(
            cycle_embed, (cycle_lstm_h_t, cycle_lstm_c_t)
        )
        cycle_lstm_hidden_states.append(cycle_lstm_h_t)
    describe("cycle_lstm_hidden_states", cycle_lstm_hidden_states)
    describe("cycle_last (观测窗口最后一帧)", cycle_lstm_hidden_states[-1])
    print()

    # --- 信号灯状态编码 & State Gating ---
    print("--- 4e. 信号灯编码 + State Gating ---")
    light_state = model.get_last_state(obs_traj, obs_state)
    light_state_embedding = model.light_embedding(light_state)
    describe("light_state (原始)", light_state)
    describe("light_state_embedding", light_state_embedding)

    queue_last = queue_lstm_hidden_states[-1]
    cycle_last = cycle_lstm_hidden_states[-1]

    if not model.disable_state_gating:
        phase_gate_input = torch.cat(
            (light_state_embedding, queue_last, cycle_last), dim=1
        )
        queue_gate = model.queue_context_gate(phase_gate_input)
        cycle_gate = model.cycle_context_gate(phase_gate_input)
        describe("queue_context_gate (sigmoid)", queue_gate)
        describe("cycle_context_gate (sigmoid)", cycle_gate)
        gated_queue_last = queue_last * queue_gate
        gated_cycle_last = cycle_last * cycle_gate
    else:
        gated_queue_last = queue_last
        gated_cycle_last = cycle_last
    describe("gated_queue_last", gated_queue_last)
    describe("gated_cycle_last", gated_cycle_last)
    print()

    # --- Aux Prediction Heads ---
    print("--- 4f. aux 预测头 (queue/cycle pred) ---")
    queue_pred_last = torch.cat(
        (model.queue_aux_reg_head(gated_queue_last),
         model.queue_aux_cls_head(gated_queue_last)),
        dim=-1,
    )
    cycle_pred_last = torch.cat(
        (model.cycle_aux_phase_head(gated_cycle_last),
         model.cycle_aux_time_head(gated_cycle_last),
         model.cycle_aux_change_head(gated_cycle_last)),
        dim=-1,
    )
    describe("queue_pred_last", queue_pred_last)
    describe("cycle_pred_last", cycle_pred_last)
    print()

    # --- 解码器初始化 ---
    print("--- 4g. 解码器初始化 (add_noise + state residual) ---")
    encoded_before_noise_hidden = torch.cat(
        (
            light_state_embedding,
            traj_lstm_hidden_states[-1],
            graph_lstm_hidden_states[-1],
        ),
        dim=1,
    )
    describe("encoded_before_noise_hidden", encoded_before_noise_hidden)

    pred_lstm_hidden = model.add_noise(encoded_before_noise_hidden, seq_start_end)
    describe("after add_noise", pred_lstm_hidden)

    init_state_residual = model.build_decoder_state_residual(
        light_state_embedding, gated_queue_last, gated_cycle_last
    )
    if init_state_residual is not None:
        describe("init_decoder_state_residual", init_state_residual)
        pred_lstm_hidden = pred_lstm_hidden + init_state_residual
        describe("decoder_hidden (after init residual)", pred_lstm_hidden)
    pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden)
    print()

    # --- 预测期 rollout ---
    print("--- 4h. 预测期 rollout (逐步解码) ---")
    # 模拟 model.forward 中 Line ~2055 的 `obs_traj_rel = obs_traj_rel[:, :, 2:4]`
    gen_input_2d = generator_input[:, :, 2:4]
    output = gen_input_2d[args.obs_len - 1]
    last_rollout_offset = output
    rollout_queue_h_t = gated_queue_last
    rollout_queue_c_t = torch.zeros_like(gated_queue_last)
    rollout_queue_feature = queue_feature_seq[-1]
    rollout_lane_queue_anchor = traffic_context["meso"]["lane_queue_anchor_seq"][-1]

    pred_traj_rel = []
    pred_lstm_hidden_per_step = []
    queue_rollout_hidden_per_step = []
    queue_rollout_feature_per_step = []
    cycle_step_embedding_per_step = []
    decoder_state_residual_per_step = []
    decode_step_context_per_step = []
    teacher_forcing_ratio = 0.5

    for i, input_t in enumerate(gen_input_2d[-args.pred_len:].chunk(args.pred_len, dim=0)):
        pred_lstm_hidden = model.inject_per_step_decoder_noise(
            pred_lstm_hidden, seq_start_end
        )
        teacher_force = torch.rand(1).item() < teacher_forcing_ratio
        input_t = input_t if teacher_force else output.unsqueeze(0)
        pred_lstm_hidden, pred_lstm_c_t = model.pred_lstm_model(
            input_t.squeeze(0), (pred_lstm_hidden, pred_lstm_c_t)
        )
        pred_lstm_hidden_per_step.append(pred_lstm_hidden)

        (
            light_state_embedding_step,
            current_cycle_feature,
            cycle_step_embedding,
        ) = model.get_decode_step_context(
            i, pred_traj_rel, obs_traj, obs_state, pred_state,
        )
        decode_step_context_per_step.append({
            "light_state_embedding": light_state_embedding_step,
            "current_cycle_feature": current_cycle_feature,
            "cycle_step_embedding": cycle_step_embedding,
        })
        cycle_step_embedding_per_step.append(cycle_step_embedding)

        if not model.disable_queue_rollout:
            rollout_info = model.rollout_queue_step(
                rollout_queue_feature,
                rollout_lane_queue_anchor,
                traffic_context["agent"]["lane_ids"][-1],
                seq_start_end,
                current_cycle_feature,
                last_rollout_offset,
                i,
                light_state_embedding_step,
                cycle_step_embedding,
                rollout_queue_h_t,
                rollout_queue_c_t,
            )
            rollout_queue_feature = rollout_info["queue_feature"]
            rollout_lane_queue_anchor = rollout_info["next_lane_queue_anchor"]
            rollout_queue_h_t = rollout_info["queue_hidden"]
            rollout_queue_c_t = rollout_info["queue_cell"]
            queue_rollout_hidden_per_step.append(rollout_queue_h_t)
            queue_rollout_feature_per_step.append(rollout_queue_feature)

            queue_context_for_decode = model.build_rollout_decode_queue_context(
                gated_queue_last, rollout_queue_h_t, light_state_embedding_step,
            )
        else:
            queue_context_for_decode = gated_queue_last

        if not model.disable_state_gating:
            cycle_step_embedding = cycle_step_embedding * model.decode_cycle_gate(
                torch.cat((light_state_embedding_step, cycle_step_embedding), dim=1)
            )

        step_state_residual = model.build_decoder_state_residual(
            light_state_embedding_step, queue_context_for_decode, cycle_step_embedding,
        )
        if step_state_residual is not None:
            pred_lstm_hidden = pred_lstm_hidden + step_state_residual
            decoder_state_residual_per_step.append(step_state_residual)

        pred_input = torch.cat((light_state_embedding_step, pred_lstm_hidden), dim=1)
        output = model.pred_hidden2pos(pred_input)
        last_rollout_offset = output
        pred_traj_rel.append(output)

    # 打印预测期逐步摘要
    for i in range(min(3, args.pred_len)):
        print(f"  --- prediction step {i} (teacher_force=TF) ---")
        describe(f"    pred_lstm_hidden  ", pred_lstm_hidden_per_step[i])
        describe(f"    cycle_step_embedding", cycle_step_embedding_per_step[i])
        describe(f"    decoder_state_residual", decoder_state_residual_per_step[i] if i < len(decoder_state_residual_per_step) else None)
        describe(f"    pred_traj_rel (output)", pred_traj_rel[i])
        if queue_rollout_hidden_per_step:
            describe(f"    rollout_queue_hidden", queue_rollout_hidden_per_step[i])
            describe(f"    rollout_queue_feature (raw)", queue_rollout_feature_per_step[i])

    # 打印中间跳过的步
    if args.pred_len > 6:
        print(f"  ... ({args.pred_len - 6} steps skipped) ...")
    for i in range(max(3, args.pred_len - 3), args.pred_len):
        print(f"  --- prediction step {i} (teacher_force=TF) ---")
        describe(f"    pred_traj_rel (output)", pred_traj_rel[i])

    print()

    # 5. 最终输出
    print("=" * 70)
    print("5. 最终输出")
    print("=" * 70)
    pred_traj_rel_tensor = torch.stack(pred_traj_rel)
    describe("pred_traj_rel (全部预测步)", pred_traj_rel_tensor)

    # 计算 traj loss
    gt_traj = gen_input_2d[-args.pred_len:, :, :2]
    traj_loss = torch.nn.functional.mse_loss(pred_traj_rel_tensor, gt_traj)
    print(f"  MSELoss(pred, gt) = {traj_loss.item():.6f}")
    print()

    # 6. aux losses
    print("=" * 70)
    print("6. aux losses")
    print("=" * 70)
    print("  queue_targets shape:", tuple(traffic_context["meso"]["queue_targets"].shape))
    print("  queue_pred_last shape:", tuple(queue_pred_last.shape))
    print("  cycle_targets shape:", tuple(traffic_context["signal"]["cycle_feature_seq"][-1].shape))
    print("  cycle_pred_last shape:", tuple(cycle_pred_last.shape))
    print()

    # 7. 从 built_context 直接调 forward 验证
    print("=" * 70)
    print("7. 使用 model.forward() 完整运行一遍（含 debug_last_aux）")
    print("=" * 70)
    with torch.no_grad():
        _ = model.forward(
            generator_input, obs_traj, obs_state, pred_state, seq_start_end,
            teacher_forcing_ratio=0.5, training_step=3,
        )
    aux = model.debug_last_aux
    for key in sorted(aux.keys()):
        val = aux[key]
        if val is None:
            print(f"  {key}: None")
        elif isinstance(val, dict):
            print(f"  {key}: dict with keys {list(val.keys())[:20]}")
        else:
            describe(f"  {key}", val)

    print("\nDone.")


if __name__ == "__main__":
    main()