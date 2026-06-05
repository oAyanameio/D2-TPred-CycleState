# Conversation Context For `D2-TPred-CycleState`

This file is created to "copy" the practical context of the current long-running
research conversation into the `D2-TPred-CycleState` project workspace.

It does not copy the chat thread itself, but it records the project state,
research direction, implementation progress, and the next recommended actions.

## Current Working Project
- Project path: `/home/lbh/D2-TPred-CycleState`
- Current branch: `feature/cyclestate-prototype`
- Purpose:
  continue implementing and iterating on the new research idea:
  `CycleState: Full-Cycle Traffic-State Memory for Trajectory Prediction at Signalized Intersections`

## Why This Branch Exists
The original `D2-TPred` repository is preserved as the reproducible baseline.
This experimental clone is used to:
1. avoid polluting the baseline repository
2. keep a separate research branch for rapid iteration
3. turn the earlier paper-analysis discussion into a trainable prototype

## Main Research Story
The current idea is not framed as "just a multi-scale temporal module".
Instead, it is framed as:

`signalized intersection trajectory prediction should be modeled as a full-cycle traffic-state memory problem`

The intended hierarchy is:
1. micro level: agent motion and local interaction
2. meso level: lane-level queue-wave state
3. macro level: signal-cycle state

This framing is chosen to better match high-level publication goals:
- stronger scientific story
- less incremental than a generic temporal encoder upgrade
- easier to argue novelty against prior work

## What Has Already Been Implemented

### 1. Experimental generator
- Added `CycleStateTrajectoryGenerator` in `D2TP/models.py`
- It reuses the original D2-TPred micro trajectory and graph interaction branches
- It adds:
  - queue-state memory branch
  - cycle-state memory branch
  - hierarchical decoder initialization

### 2. Training entry integration
- Added `--model_type {d2tpred, cyclestate}` in training/evaluation
- Added partial warm-start from original D2-TPred checkpoint
- For CycleState:
  - compatible weights are reused
  - incompatible new layers are randomly initialized
  - training starts from epoch 0

### 3. Smoke tests that already passed
- forward pass works
- backward pass works
- partial checkpoint warm-start works

### 4. Training stabilization utilities
- `--generator_only`
- `--aux_queue_weight`
- `--aux_cycle_weight`
- `--gan_weight`
- `--max_train_batches`
- `--max_val_batches`

### 5. Auxiliary supervision upgrade
The earlier auxiliary supervision version directly supervised slices of hidden states.
This has already been improved:
- `queue_aux_head`
- `cycle_aux_head`

Now the queue/cycle branches explicitly predict auxiliary state targets.

## Quick Experimental Findings Already Observed

### Earlier prototype with GAN
CycleState could train, but the optimization was noisy.

### Generator-only + auxiliary supervision
This mode was more stable.

Observed trends from quick smoke runs:
- queue auxiliary loss decreased clearly
- cycle auxiliary loss also decreased
- validation ADE/FDE showed a downward trend in short training

This means:
- the CycleState branch is trainable
- state supervision is useful
- generator-only is currently a better early-stage mode than full GAN training

## Files Most Relevant To Continue From Here
- `README.md`
  contains the cumulative optimization log and the current research framing
- `D2TP/models.py`
  contains the current CycleState prototype
- `D2TP/train.py`
  contains training controls and auxiliary loss hooks
- `D2TP/evaluate_model.py`
  supports `--model_type cyclestate`

## Current High-Priority Next Steps
1. strengthen queue-state targets:
   queue length, release order, stop-line occupancy, front-of-queue status
2. run a more complete generator-only training experiment
3. add evaluation-time auxiliary-state analysis
4. after stable convergence, reintroduce GAN with smaller weight

## Current Development Shift
The development mainline has now explicitly shifted to:

`training-protocol strengthening + INT2 interface compatibility`

This means the immediate focus is no longer on rapidly adding more topology.
Instead, the priority is to make the current `CycleState` branch:
1. train under a staged protocol
2. supervise queue/cycle states with structured losses
3. support controllable gating ablation
4. expose a tuple-to-context adapter so future INT2 migration only needs a new adapter layer

## Newest Modeling Upgrade
The next concrete modeling step after protocol stabilization is:

`Phase-Rolling Queue Memory`

Meaning:
- queue-state should not stay frozen after decoder initialization
- instead, the model should roll meso queue-wave state forward during prediction
- this rollout is conditioned on:
  - cycle progression
  - predicted motion
  - last observed queue-state anchor

This keeps the research story coherent:
`signalized intersection forecasting as full-cycle traffic-state memory modeling`
now includes not only state encoding, but also state evolution during the future horizon.

## Latest Structural Correction
The newest important correction is:

`baseline-compatible decoder state residual`

Meaning:
- earlier CycleState versions expanded the decoder directly and broke full warm-start
  compatibility with the original D2-TPred decoder
- this was likely one major reason for the earlier strong metric degradation
- the current version keeps the original decoder shape intact and injects state memory
  through a gated residual pathway

Current implication:
- future optimization should treat `protect baseline decoder capability` as a first-order
  constraint, not a secondary engineering detail

## Important Constraint From The Ongoing Research Goal
Further optimization should keep following these requirements:
1. tell a strong scientific story
2. aim to exceed the original paper's practical performance
3. avoid simply copying existing common trajectory-prediction tricks

## Practical Note
From this point on, collaboration should be treated as happening inside
`D2-TPred-CycleState`, on branch:

`feature/cyclestate-prototype`
