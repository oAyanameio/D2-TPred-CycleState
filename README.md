# D2 TPred : Discontinuous Dependency for Trajectory Prediction under Traffic Lights

# CycleState Experiment Branch
This repository is also used as a working branch for a new research idea:
`CycleState: Full-Cycle Traffic-State Memory for Trajectory Prediction at Signalized Intersections`.

The core motivation is to move from the original short-history interaction modeling
in D2-TPred towards a hierarchical traffic-state view:
1. Micro level: short-term agent motion and interaction.
2. Meso level: lane-level queue-wave state.
3. Macro level: signal-cycle state.

At the current stage, the implementation is still an experimental prototype
(`CycleState v0/v1`) built on top of D2-TPred, but it is already trainable and
supports warm-start from the original checkpoint.

# Current Experimental Files
- `D2TP/models.py`
  Added `CycleStateTrajectoryGenerator`.
- `D2TP/train.py`
  Added `--model_type cyclestate`, `--generator_only`, auxiliary-loss options,
  and quick-smoke training controls.
- `D2TP/evaluate_model.py`
  Added `--model_type cyclestate`.

# CycleState Design Summary
Compared with the original D2-TPred generator, CycleState currently adds:
1. A queue-state branch:
   weak queue-wave statistics are extracted from lane-level neighbors,
   including queue count, lane density, waiting ratio, release ratio,
   mean lane speed, current light state, elapsed light time, and stop-line distance.
2. A cycle-state branch:
   phase one-hot, elapsed phase time, and phase-change indicators are encoded
   as a compact signal-cycle memory.
3. A hierarchical decoder:
   the decoder is initialized not only by motion and graph interaction features,
   but also by queue-state memory and cycle-state memory.
4. Explicit auxiliary heads:
   queue-state and cycle-state are now predicted by dedicated heads instead of
   directly supervising hidden-state slices.

# Optimization Log
Below is the optimization record from the first prototype to the current version.

## Stage 0: Original Baseline
- Original D2-TPred was kept as the baseline.
- PyTorch compatibility for evaluation was fixed in the base branch:
  device-agnostic execution, safer normalization for short temporal windows,
  and stable evaluation entry.

## Stage 1: Experimental Clone
- A separate experimental copy was created at:
  `/home/lbh/D2-TPred-CycleState`
- This avoids polluting the original reproducible D2-TPred repository.

## Stage 2: CycleState v0 Prototype
- Added `CycleStateTrajectoryGenerator`.
- Reused the original micro trajectory encoder and graph interaction branch.
- Added:
  - lane-level queue-state memory
  - signal-cycle memory
- Verified:
  - forward pass works
  - backward pass works
  - partial warm-start from original `model_best.pth.tar` works

## Stage 3: Quick Smoke Training Support
- Added quick-debug controls in `train.py`:
  - `--max_train_batches`
  - `--max_val_batches`
- This made it possible to rapidly test whether the new branch can learn at all.

## Stage 4: Warm-Start Strategy
- Added a compatible partial checkpoint loader.
- For `cyclestate`, the model now:
  - reuses compatible weights from original D2-TPred
  - keeps new queue/cycle modules randomly initialized
  - starts training from epoch 0 instead of inheriting the old epoch counter

## Stage 5: Generator-Only Stabilization
- Added `--generator_only`.
- Motivation:
  early CycleState experiments should first validate whether queue/cycle memory
  is useful, without GAN instability masking the effect.

## Stage 6: Auxiliary State Supervision
- Added:
  - `--aux_queue_weight`
  - `--aux_cycle_weight`
  - `--gan_weight`
- First implementation supervised queue/cycle hidden states indirectly.
- Current implementation upgrades this to explicit auxiliary prediction heads:
  - `queue_aux_head`
  - `cycle_aux_head`
- This makes the optimization target more interpretable and easier to analyze.

## Stage 7: Current Best Early Observation
- Pure generator + auxiliary supervision is more stable than directly mixing GAN.
- In short smoke runs, queue/cycle auxiliary losses decreased clearly.
- Validation ADE/FDE in quick experiments also showed a downward trend, indicating
  that the CycleState branch is trainable and already contributes positive signal.

# Recommended Current Training Mode
For early-stage CycleState experiments, the recommended mode is:
1. use `--model_type cyclestate`
2. use `--generator_only`
3. warm-start from original `model_best.pth.tar`
4. enable `--aux_queue_weight` and `--aux_cycle_weight`
5. keep GAN disabled until the queue/cycle branches become stable

Example:
```bash
CUDA_VISIBLE_DEVICES=2 python D2TP/train.py \
  --model_type cyclestate \
  --generator_only \
  --device cuda \
  --pin_memory \
  --resume ./model_best.pth.tar \
  --aux_queue_weight 10 \
  --aux_cycle_weight 5
```

# Next Planned Optimizations
The current CycleState branch is still an early prototype. The next planned
optimizations are:
1. strengthen queue-state targets:
   queue length, release order, stop-line occupancy, front-of-queue status.
2. add evaluation-time analysis for auxiliary state prediction quality.
3. after generator-only training becomes stable, reintroduce adversarial training
   with smaller GAN weight.
4. extend from weak queue-wave statistics to richer cycle-aware traffic-state tokens.
# How was the data collected?
The data in VTP-TL comes from at urban intersections with traffic lights is used to predict vehicles trajectory in different times of day and provides a broad range of real-world driving scenarios. We use drones to hover at 70 to 120 meters above the traffic intersections, as statically as possible, to record vehicle trajectories passing through the area with a bird’s-eye view in the daytime of the non-rush hours, rush hours, and the evening.


<div align=center>
<img src="https://github.com/VTP-TL/D2-TPred/blob/main/drone.png" width="780" height="312" alt=" "/><br/>
</div>

# Where was the data collected?
We choose 3 different traffic intersections, including crossroad, T-junction, and roundabout scenarios. In these scenario, they own the different number of roads and traffic lights, and cause to different movement behaviors for vehicles.

<div align=center>
<img src="https://github.com/VTP-TL/D2-TPred/blob/main/scenarios.png" width="762" height="628" alt=" "/><br/>
</div>

# Summary of the Dataset 
In the [VTP-TL](https://pan.baidu.com/s/1gAdWP58RCKl0RrsvtQotpw) dataset, we have collected data from 3 different categories of traffic scenarios using drones. The summary of the data is listed in the following table. 

<div align=center>
<img src="https://github.com/VTP-TL/D2-TPred/blob/main/summary.png" width="772" height="503" alt=" "/><br/>
</div>

# Included Materials
For the 3 recording scenarios, we include 2 files for each scenarios: 
1. The sample of video clips (xxx.mp4) 
2. Recorded vehicle trajectory file (xxx.txt) 
where, we provide trajectories information in pixel.

# Recorded Vehicle Trajectory files (xxx.txt)
**F_id:** column 1. For each agent (per Agent_id), frame_id represents the frames the agent appears in the video.    
**A_id:** column 2. For each xxx.txt file, the Agent_id starts from 0, and represent the ID of the agent.   
**x:** column 3, the x position of the agent at each frame. The unit is pixel.     
**y:** column 4, the y position of the agent at each frame. The unit is pixel.   
**Lane_id:** column 5, For each xxx.txt file, the Lane_id starts from 0, and represent the ID of the traffic lane.   
**pa:** column 6, For each xxx.txt file, the inperception is set as 0 or 1, and represent whether vehicle locates in the influencing area of traffic light.   
**f:** column 7, For each xxx.txt file, the isfirstobj is set as 0 or 1, and represent whether vehicle is the first agent in the influencing area of traffic light.   
**Lig_id:** column 8, For each xxx.txt file, the Lig_id starts from 0, and represent the ID of the traffic light.   
**ls:** column 9, For each xxx.txt file, the Ls is set as 0, 1 and 2, and represents the state of traffic light.   
**mb:** column 10, For each xxx.txt file, the Mb is set as 0, 1 and 2, and represents the movement behaviors of vehicle.   
**lt:** column 11, For each xxx.txt file, the Ldurtime represents the durtime of traffic light.   

**Example:**
<div align=center>
<img src="https://github.com/VTP-TL/D2-TPred/blob/main/smaple.png" alt=" "/><br/>
</div>
