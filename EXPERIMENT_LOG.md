# Experiment Log

## Naming Convention
- Formal short runs are stored under:
  - `experiments/cyclestate/warmup_v1`
  - `experiments/cyclestate/refine_v1`
  - `experiments/cyclestate/adversarial_v1`
- Quick smoke runs can still live under `quick_runs/`, but the preferred convention
  moving forward is to use `experiments/` so that logs, checkpoints, and protocol
  settings stay grouped by run.

## Configuration Template
- Experiment name:
- Date:
- Branch:
- Model type: `d2tpred` or `cyclestate`
- Train stage: `warmup` / `refine` / `adversarial`
- Warm-start checkpoint:
- `generator_only`:
- `gan_weight`:
- `aux_queue_weight`:
- `aux_cycle_weight`:
- `disable_state_gating`:
- `teacher_forcing_ratio`:
- `max_train_batches`:
- `max_val_batches`:
- Command:
- Best validation ADE:
- Best validation FDE:
- Notes:

## Current Protocol Summary
- `warmup`
  - target: stabilize queue/cycle state branches first
  - default: `generator_only=True`, `gan_weight=0`, higher aux weights
- `refine`
  - target: let structured state supervision help trajectory reconstruction rather than dominate it
  - default: `generator_only=True`, `gan_weight=0`, medium aux weights, decayed teacher forcing
- `adversarial`
  - target: reintroduce GAN only as a distribution refinement term
  - default: `generator_only=False`, smaller `gan_weight`, structured aux retained

## Known Smoke Results Before This Round
- Source: `quick_runs/cyclestate_aux_smoke/train.log`
- Observation:
  - generator-only + auxiliary supervision was more stable than early direct GAN mixing
  - queue/cycle auxiliary losses showed a downward trend
  - one short run reached approximately:
    - ADE `147.615`
    - FDE `278.867`
- Interpretation:
  - this is only a smoke signal, not a formal reproduced benchmark

## This Round: Protocol Upgrade Notes
- Added tuple-to-context adapter at training/evaluation entry.
- Added stage-based training defaults.
- Upgraded auxiliary supervision from unified MSE to structured queue/cycle losses:
  - `queue_reg_loss`
  - `queue_cls_loss`
  - `cycle_phase_loss`
  - `cycle_time_loss`
  - `cycle_change_loss`
- Added `disable_state_gating` for controllable `phase-conditioned state modulation` ablation.
- Standardized experiment outputs so each run can write:
  - `train.log`
  - per-run checkpoints under `checkpoint/`

## Next Structural Upgrade
- Added `Phase-Rolling Queue Memory`.
- New interpretation:
  queue-wave state is no longer treated as a static context token injected once at decode start.
  It is now rolled forward during prediction under:
  - phase progression
  - predicted agent motion
  - the last observed meso-state anchor
- New internal debug/analysis signals:
  - `queue_rollout_hidden_seq`
  - `queue_rollout_pred_seq`
  - `queue_rollout_target_seq`
- New queue supervision components already wired into the existing queue auxiliary objective:
  - `queue_rollout_reg_loss`
  - `queue_rollout_cls_loss`

## Planned Minimum Short Comparisons
1. `cyclestate + warmup + aux + gating`
2. `cyclestate + warmup + aux + no gating`
3. `cyclestate + warmup + no aux + gating`
4. `d2tpred baseline quick run`

## Smoke Run Result Template
### Run
- Name:
- Command:
- Best ADE:
- Best FDE:
- Stability notes:

### Comparative Observation
- Gating effect:
- Structured auxiliary loss effect:
- Compared with baseline:

## Smoke Runs Completed In This Round
### Run
- Name: `experiments/cyclestate/warmup_gating_aux_v1`
- Command:
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_gating_aux_v1 --model_type cyclestate --train_stage warmup --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --best_k 2 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 1 --max_train_batches 1 --max_val_batches 1`
- Best ADE: `155.554`
- Best FDE: `298.113`
- Stability notes:
  - forward/backward passed
  - validation completed
  - structured queue/cycle losses all produced non-zero values

### Run
- Name: `experiments/cyclestate/warmup_nogating_aux_v1`
- Command:
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_nogating_aux_v1 --model_type cyclestate --train_stage warmup --disable_state_gating --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --best_k 2 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 1 --max_train_batches 1 --max_val_batches 1`
- Best ADE: `155.569`
- Best FDE: `298.149`
- Stability notes:
  - gating ablation path runs correctly
  - metrics are slightly worse than the gated version in this tiny smoke setting

### Run
- Name: `experiments/cyclestate/warmup_gating_noaux_v1`
- Command:
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_gating_noaux_v1 --model_type cyclestate --train_stage warmup --aux_queue_weight 0 --aux_cycle_weight 0 --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --best_k 2 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 1 --max_train_batches 1 --max_val_batches 1`
- Best ADE: `155.555`
- Best FDE: `298.116`
- Stability notes:
  - this ablation initially exposed a device-placement bug for zero auxiliary losses
  - the bug was fixed and regression-tested
  - after the fix, the no-aux path ran correctly

### Run
- Name: `experiments/d2tpred/baseline_quick_v3`
- Command:
  `python D2TP/train.py --log_dir experiments/d2tpred/baseline_quick_v3 --model_type d2tpred --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --best_k 2 --resume D2TP/model_best.pth.tar --num_epochs 129 --print_every 1 --max_train_batches 3 --max_val_batches 1`
- Best ADE: `84.391`
- Best FDE: `172.836`
- Stability notes:
  - because original D2-TPred uses discriminator-first scheduling and resumes from epoch 129, the baseline quick run needs a slightly different smoke protocol to actually enter one generator update + one validation pass

## Current Short-Run Reading
- Gating effect:
  in the current 1-batch warmup smoke setting, `with gating` is slightly better than `without gating`
  (`155.554/298.113` vs `155.569/298.149`), which is directionally encouraging but too small to be a substantive claim.
- Structured auxiliary loss effect:
  in this extremely small setting, `with aux` and `without aux` are numerically very close
  (`155.554/298.113` vs `155.555/298.116`).
  The more important current evidence is that structured aux losses are trainable, logged, and do not break the warmup protocol.
- Compared with baseline:
  the original D2-TPred checkpoint remains much stronger in this quick comparison.
  This is expected at the current stage because `CycleState` is still a newly warm-started prototype with new random-initialized state branches.
- Main conclusion of this round:
  the protocol upgrade is successful as an infrastructure step:
  `CycleState` now supports staged training, structured state losses, controllable gating ablation, and experiment-local logging/checkpoints.

## Recommended Next Short Comparisons
1. `cyclestate + warmup + aux + gating + queue rollout`
2. `cyclestate + warmup + aux + gating + queue rollout disabled`:
   this would require a later ablation switch if we decide to expose one
3. `cyclestate + refine + aux + gating + queue rollout`
4. compare whether rollout losses decline more stably than the old static queue-only supervision

## Smoke Run: Queue Rollout Enabled
### Run
- Name: `experiments/cyclestate/warmup_rollout_aux_v1`
- Command:
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_rollout_aux_v1 --model_type cyclestate --train_stage warmup --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --best_k 2 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 1 --max_train_batches 1 --max_val_batches 1`
- Best ADE: `156.592`
- Best FDE: `300.038`
- Stability notes:
  - forward/backward passed
  - validation completed
  - new rollout losses appeared in logs:
    - `QRollReg 0.364289`
    - `QRollCls 0.688436`

## Current Reading After Queue Rollout
- This round does not yet prove metric gain.
- What it does prove:
  - the new `phase-rolling queue memory` is fully wired into training
  - rollout supervision is numerically active
  - the scientific story is now stronger than the earlier static queue injection version
- The next meaningful comparison should be:
  `rollout enabled` vs `rollout disabled` under a longer warmup or refine run,
  rather than treating this 1-batch smoke as a performance verdict.

## Smoke Run: Queue Rollout On/Off Ablation
### Run
- Name: `experiments/cyclestate/warmup_rollout_on_v1`
- Command:
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_rollout_on_v1 --model_type cyclestate --train_stage warmup --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --best_k 2 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 1 --max_train_batches 1 --max_val_batches 1`
- Best ADE: `156.592`
- Best FDE: `300.038`
- Stability notes:
  - rollout losses active:
    - `QRollReg 0.364289`
    - `QRollCls 0.688436`

### Run
- Name: `experiments/cyclestate/warmup_rollout_off_v1`
- Command:
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_rollout_off_v1 --model_type cyclestate --train_stage warmup --disable_queue_rollout --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --best_k 2 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 1 --max_train_batches 1 --max_val_batches 1`
- Best ADE: `156.663`
- Best FDE: `300.138`
- Stability notes:
  - rollout losses correctly dropped to zero:
    - `QRollReg 0.000000`
    - `QRollCls 0.000000`

## Current Reading After Rollout Ablation
- The ablation path is now clean and verifiable.
- In the current tiny smoke run:
  - rollout `on` is slightly better than rollout `off`
  - the difference is too small to claim substantive gain
- The stronger conclusion is methodological:
  `Phase-Rolling Queue Memory` is now a proper experimental factor, not just a default code path.

## Structural Upgrade: Lane-Consensus Meso Anchor
- Added `lane_queue_anchor_seq` into `traffic_context`.
- Interpretation:
  queue-wave evolution is now softly regularized by lane-level consensus rather than relying
  only on per-agent local meso-state.
- New controllable variable:
  - `--disable_lane_queue_anchor`

## Smoke Run: Lane Anchor Enabled
### Run
- Name: `experiments/cyclestate/warmup_lane_anchor_v1`
- Command:
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_lane_anchor_v1 --model_type cyclestate --train_stage warmup --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --best_k 2 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 1 --max_train_batches 1 --max_val_batches 1`
- Best ADE: `156.591`
- Best FDE: `300.036`
- Stability notes:
  - warmup path stayed stable
  - rollout losses remained active:
    - `QRollReg 0.254486`
    - `QRollCls 0.688171`

## Current Reading After Lane Anchor
- This does not yet prove a strong gain.
- What it proves:
  - the model can now express a cleaner meso-level scientific story
  - lane-level collective traffic state has been inserted into the rollout path without
    breaking training
- Additional progress:
  - predictive lane anchors are now traceable through `lane_queue_rollout_anchor_seq`
  - this gives us future visualization/analysis hooks for showing how meso consensus evolves
- Next best comparison:
  `lane anchor on` vs `lane anchor off` under the same rollout-enabled warmup/refine setting

## Structural Upgrade: Baseline-Compatible Decoder State Residual
- Added `decoder_state_residual` and `decoder_state_gate`.
- Interpretation:
  the state branches should modulate a strong baseline decoder, rather than forcing
  CycleState to relearn a wider decoder from scratch.
- New controllable variable:
  - `--disable_decoder_state_residual`
- Additional verification:
  - CycleState decoder shape is now baseline-compatible
  - original D2-TPred generator checkpoint can be loaded with `skipped 0 keys`

## Smoke Run: Decoder Residual On/Off
### Run
- Name: `experiments/cyclestate/warmup_residual_on_v1`
- Command:
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_residual_on_v1 --model_type cyclestate --train_stage warmup --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --best_k 2 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 1 --max_train_batches 1 --max_val_batches 1`
- Best ADE: `54.784`
- Best FDE: `114.138`
- Stability notes:
  - full decoder warm-start restored
  - checkpoint load reported `skipped 0 keys`
  - structured aux losses remained active

### Run
- Name: `experiments/cyclestate/warmup_residual_off_v1`
- Command:
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_residual_off_v1 --model_type cyclestate --train_stage warmup --disable_decoder_state_residual --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --best_k 2 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 1 --max_train_batches 1 --max_val_batches 1`
- Best ADE: `45.856`
- Best FDE: `93.743`
- Stability notes:
  - full decoder warm-start restored
  - checkpoint load reported `skipped 0 keys`
  - this run is currently stronger than the residual-on variant in the same tiny smoke setup

## Current Reading After Decoder Residual Upgrade
- The key result of this round is not only the on/off comparison itself.
- The more important conclusion is:
  the earlier severe metric collapse was strongly tied to decoder warm-start incompatibility.
- Once full decoder compatibility is restored, CycleState short-run behavior improves dramatically
  from the earlier `155+ / 298+` range to the current `54.784 / 114.138` and `45.856 / 93.743`.
- Current cautious interpretation:
  - the residual pathway is scientifically meaningful and should stay as a controllable variable
  - but the present 1-batch result does not yet prove that `residual on` is better than `residual off`
  - the next correct step is a longer warmup/refine comparison under the same protocol
