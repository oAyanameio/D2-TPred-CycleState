# 实验日志

## 命名规范
- 正式短实验建议存放在：
  - `experiments/cyclestate/warmup_v1`
  - `experiments/cyclestate/refine_v1`
  - `experiments/cyclestate/adversarial_v1`
- quick smoke run 也可以继续放在 `quick_runs/` 下，
  但后续更推荐统一使用 `experiments/`，
  这样日志、checkpoint 和协议设置都能按实验分组保存。

## 配置模板
- Experiment name：
- Date：
- Branch：
- Model type：`d2tpred` 或 `cyclestate`
- Train stage：`warmup` / `refine` / `adversarial`
- Warm-start checkpoint：
- `generator_only`:
- `gan_weight`:
- `aux_queue_weight`:
- `aux_cycle_weight`:
- `disable_state_gating`:
- `teacher_forcing_ratio`:
- `max_train_batches`:
- `max_val_batches`:
- Command：
- Best validation ADE：
- Best validation FDE：
- Notes：

## 当前协议概览
- `warmup`
  - 目标：先稳定 queue/cycle 状态分支
  - 默认：`generator_only=True`、`gan_weight=0`、较高的 auxiliary weights
- `refine`
  - 目标：让 structured state supervision 帮助轨迹重建，而不是反过来主导训练
  - 默认：`generator_only=True`、`gan_weight=0`、中等 auxiliary weights、衰减式 teacher forcing
- `adversarial`
  - 目标：只把 GAN 作为分布精修项重新引入
  - 默认：`generator_only=False`、较小的 `gan_weight`、保留 structured aux

## 本轮之前已知的 Smoke 结果
- 来源：`quick_runs/cyclestate_aux_smoke/train.log`
- 观察：
  - `generator-only + auxiliary supervision` 比早期直接混入 GAN 更稳定
  - queue/cycle auxiliary losses 呈现下降趋势
  - 一次短实验大致达到：
    - ADE `147.615`
    - FDE `278.867`
- 解读：
  - 这只是一个 smoke signal，还不能视作正式复现实验结果

## 本轮：训练协议升级说明
- 在训练/评估入口新增了 `tuple-to-context adapter`
- 新增了基于阶段的训练默认配置
- 将 auxiliary supervision 从统一 MSE 升级为结构化 queue/cycle losses：
  - `queue_reg_loss`
  - `queue_cls_loss`
  - `cycle_phase_loss`
  - `cycle_time_loss`
  - `cycle_change_loss`
- 新增 `disable_state_gating`，用于可控的 `phase-conditioned state modulation` 消融
- 统一实验输出规范，使每个 run 都可以保存：
  - `train.log`
  - 位于 `checkpoint/` 下的实验专属 checkpoints

## 下一项结构升级
- 新增 `Phase-Rolling Queue Memory`
- 新的理解方式是：
  - queue-wave state 不再被视作一个只在解码开始时注入一次的静态 context token
  - 它现在会在预测期内持续向前滚动，依赖：
    - phase progression
    - predicted agent motion
    - 最后观测到的 `meso-state anchor`
- 新增内部 debug/analysis 信号：
  - `queue_rollout_hidden_seq`
  - `queue_rollout_pred_seq`
  - `queue_rollout_target_seq`
- 新增 queue supervision 分量，并已接入现有 queue auxiliary objective：
  - `queue_rollout_reg_loss`
  - `queue_rollout_cls_loss`

## 计划中的最小短实验比较
1. `cyclestate + warmup + aux + gating`
2. `cyclestate + warmup + aux + no gating`
3. `cyclestate + warmup + no aux + gating`
4. `d2tpred baseline quick run`

## Smoke Run 结果模板
### Run
- Name：
- Command：
- Best ADE：
- Best FDE：
- Stability notes：

### Comparative Observation
- Gating effect：
- Structured auxiliary loss effect：
- Compared with baseline：

## 本轮已完成的 Smoke Runs
### Run
- Name：`experiments/cyclestate/warmup_gating_aux_v1`
- Command：
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_gating_aux_v1 --model_type cyclestate --train_stage warmup --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --best_k 2 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 1 --max_train_batches 1 --max_val_batches 1`
- Best ADE：`155.554`
- Best FDE：`298.113`
- Stability notes：
  - forward/backward passed
  - validation completed
  - structured queue/cycle losses all produced non-zero values

### Run
- Name：`experiments/cyclestate/warmup_nogating_aux_v1`
- Command：
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_nogating_aux_v1 --model_type cyclestate --train_stage warmup --disable_state_gating --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --best_k 2 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 1 --max_train_batches 1 --max_val_batches 1`
- Best ADE：`155.569`
- Best FDE：`298.149`
- Stability notes：
  - gating 消融路径运行正常
  - 在当前极小 smoke 设置下，指标略差于带 gating 的版本

### Run
- Name：`experiments/cyclestate/warmup_gating_noaux_v1`
- Command：
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_gating_noaux_v1 --model_type cyclestate --train_stage warmup --aux_queue_weight 0 --aux_cycle_weight 0 --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --best_k 2 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 1 --max_train_batches 1 --max_val_batches 1`
- Best ADE：`155.555`
- Best FDE：`298.116`
- Stability notes：
  - 这个消融最初暴露了“零 auxiliary loss 的设备放置 bug”
  - 该问题已经修复，并且补了 regression test
  - 修复后，`no-aux` 路径运行正常

### Run
- Name：`experiments/d2tpred/baseline_quick_v3`
- Command：
  `python D2TP/train.py --log_dir experiments/d2tpred/baseline_quick_v3 --model_type d2tpred --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --best_k 2 --resume D2TP/model_best.pth.tar --num_epochs 129 --print_every 1 --max_train_batches 3 --max_val_batches 1`
- Best ADE：`84.391`
- Best FDE：`172.836`
- Stability notes：
  - 原始 `D2-TPred` 使用判别器优先调度，并且从 epoch 129 恢复，
    因此这条 baseline quick run 需要一个略有不同的 smoke protocol，
    才能真正进入“一次生成器更新 + 一次验证”的流程

## 当前短实验解读
- Gating effect：
  - 在当前单 batch `warmup` smoke 设置下，`with gating` 略优于 `without gating`
    （`155.554/298.113` 对 `155.569/298.149`），
    方向上是鼓舞性的，但差异还太小，不能当成实质性结论。
- Structured auxiliary loss effect：
  - 在这个极小设置下，`with aux` 和 `without aux` 数值非常接近
    （`155.554/298.113` 对 `155.555/298.116`）。
  - 更重要的证据是：structured aux losses 已经可训练、可记录，
    并且不会破坏 `warmup` 协议。
- Compared with baseline：
  - 在这次快速比较里，原始 `D2-TPred` checkpoint 仍明显更强。
  - 这是符合预期的，因为当前 `CycleState` 仍然是一个
    新分支 warm-start 原型，很多状态分支参数是随机初始化的。
- 本轮主要结论：
  - 这次协议升级已经在基础设施层面成功：
    `CycleState` 现在支持 staged training、structured state losses、
    controllable gating ablation，以及 experiment-local logging/checkpoints。

## 推荐的下一批短实验比较
1. `cyclestate + warmup + aux + gating + queue rollout`
2. `cyclestate + warmup + aux + gating + queue rollout disabled`:
   - 如果决定暴露这个变量，就需要一个后续消融开关
3. `cyclestate + refine + aux + gating + queue rollout`
4. 比较 rollout losses 是否比旧的静态 queue-only supervision 下降得更稳定

## Smoke Run：Queue Rollout Enabled
### Run
- Name：`experiments/cyclestate/warmup_rollout_aux_v1`
- Command：
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_rollout_aux_v1 --model_type cyclestate --train_stage warmup --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --best_k 2 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 1 --max_train_batches 1 --max_val_batches 1`
- Best ADE：`156.592`
- Best FDE：`300.038`
- Stability notes：
  - forward/backward passed
  - validation completed
  - 新的 rollout losses 已经出现在日志中：
    - `QRollReg 0.364289`
    - `QRollCls 0.688436`

## Queue Rollout 之后的当前解读
- 本轮还不能证明指标收益。
- 但它已经证明：
  - 新的 `phase-rolling queue memory` 已经完整接入训练流程
  - rollout supervision 在数值上是激活的
  - 当前科研叙事相较于早期静态 queue injection 版本更完整
- 下一步更有意义的比较应该是：
  - 在更长的 `warmup` 或 `refine` 实验下，对比
    `rollout enabled` 与 `rollout disabled`
  - 而不是把这次单 batch smoke 当成性能定论

## Smoke Run：Queue Rollout On/Off 消融
### Run
- Name：`experiments/cyclestate/warmup_rollout_on_v1`
- Command：
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_rollout_on_v1 --model_type cyclestate --train_stage warmup --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --best_k 2 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 1 --max_train_batches 1 --max_val_batches 1`
- Best ADE：`156.592`
- Best FDE：`300.038`
- Stability notes：
  - rollout losses 已激活：
    - `QRollReg 0.364289`
    - `QRollCls 0.688436`

### Run
- Name：`experiments/cyclestate/warmup_rollout_off_v1`
- Command：
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_rollout_off_v1 --model_type cyclestate --train_stage warmup --disable_queue_rollout --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --best_k 2 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 1 --max_train_batches 1 --max_val_batches 1`
- Best ADE：`156.663`
- Best FDE：`300.138`
- Stability notes：
  - rollout losses 已正确降为 0：
    - `QRollReg 0.000000`
    - `QRollCls 0.000000`

## Rollout 消融后的当前解读
- 当前这条消融路径已经足够干净、可验证。
- 在这个极小 smoke run 中：
  - rollout `on` 略优于 rollout `off`
  - 但差异仍然太小，不能据此宣称有实质收益
- 更强的结论是方法论层面的：
  - `Phase-Rolling Queue Memory` 现在已经成为一个真正的实验变量，
    而不是一个默认隐藏代码路径

## 结构升级：Lane-Consensus Meso Anchor
- 在 `traffic_context` 中新增了 `lane_queue_anchor_seq`
- 理解方式：
  - queue-wave evolution 现在会受到 lane-level consensus 的软约束，
    而不再只依赖每个 agent 的局部 `meso-state`
- 新的可控变量：
  - `--disable_lane_queue_anchor`

## Smoke Run：Lane Anchor Enabled
### Run
- Name：`experiments/cyclestate/warmup_lane_anchor_v1`
- Command：
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_lane_anchor_v1 --model_type cyclestate --train_stage warmup --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --best_k 2 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 1 --max_train_batches 1 --max_val_batches 1`
- Best ADE：`156.591`
- Best FDE：`300.036`
- Stability notes：
  - `warmup` 路径保持稳定
  - rollout losses 仍处于激活状态：
    - `QRollReg 0.254486`
    - `QRollCls 0.688171`

## Lane Anchor 之后的当前解读
- 当前还不能证明它带来了明显收益。
- 但它已经证明：
  - 模型现在可以表达一个更清晰的中观层科学故事
  - lane-level collective traffic state 已经被接入 rollout 路径，
    而且没有破坏训练稳定性
- 额外进展：
  - predictive lane anchors 现在可以通过 `lane_queue_rollout_anchor_seq` 被追踪
  - 这为后续可视化和分析 `meso consensus` 如何演化提供了接口
- 下一步最值得做的比较：
  - 在相同 rollout-enabled 的 `warmup/refine` 设置下，
    对比 `lane anchor on` 与 `lane anchor off`

## 结构升级：Baseline-Compatible Decoder State Residual
- 新增了 `decoder_state_residual` 和 `decoder_state_gate`
- 理解方式：
  - 状态分支应该去调制一个强 baseline decoder，
    而不是迫使 `CycleState` 从零开始重学一个更宽的 decoder
- 新的可控变量：
  - `--disable_decoder_state_residual`
- 额外验证结果：
  - `CycleState` decoder 形状现在与 baseline 兼容
  - 原始 `D2-TPred` 生成器 checkpoint 可以在 `skipped 0 keys` 的情况下加载

## Smoke Run：Decoder Residual On/Off
### Run
- Name：`experiments/cyclestate/warmup_residual_on_v1`
- Command：
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_residual_on_v1 --model_type cyclestate --train_stage warmup --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --best_k 2 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 1 --max_train_batches 1 --max_val_batches 1`
- Best ADE：`54.784`
- Best FDE：`114.138`
- Stability notes：
  - 已恢复完整 decoder warm-start
  - checkpoint 加载报告为 `skipped 0 keys`
  - structured aux losses 仍然保持激活

### Run
- Name：`experiments/cyclestate/warmup_residual_off_v1`
- Command：
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_residual_off_v1 --model_type cyclestate --train_stage warmup --disable_decoder_state_residual --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --best_k 2 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 1 --max_train_batches 1 --max_val_batches 1`
- Best ADE：`45.856`
- Best FDE：`93.743`
- Stability notes：
  - 已恢复完整 decoder warm-start
  - checkpoint 加载报告为 `skipped 0 keys`
  - 在当前这个极小 smoke 设置下，该 run 暂时优于 residual-on 版本

## Decoder Residual 升级后的当前解读
- 本轮最关键的结果，不仅仅是 on/off 比较本身。
- 更重要的结论是：
  - 早期那种严重的指标崩塌，与 decoder warm-start 不兼容有很强关联。
- 一旦恢复完整 decoder 兼容性，`CycleState` 的短实验行为就出现了明显改善：
  - 从早期 `155+ / 298+` 的量级，回到了当前
    `54.784 / 114.138` 和 `45.856 / 93.743`
- 当前更谨慎的理解是：
  - residual pathway 在科学上是有意义的，应该继续保留为可控变量
  - 但当前这次 1-batch 结果仍不能证明 `residual on` 一定优于 `residual off`
  - 下一步正确做法仍然是在相同协议下做更长的 `warmup/refine` 比较
