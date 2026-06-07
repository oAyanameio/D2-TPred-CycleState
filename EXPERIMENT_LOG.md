# 实验日志

## 命名规范
- 正式短实验建议存放在：
  - `experiments/cyclestate/warmup_v1`
  - `experiments/cyclestate/refine_v1`
  - `experiments/cyclestate/adversarial_v1`
- quick smoke run 也可以继续放在 `quick_runs/` 下，
  但后续更推荐统一使用 `experiments/`，
  这样日志、checkpoint 和协议设置都能按实验分组保存。

## 结果标签规范
- `smoke`
  - 只用于验证 forward/backward、日志、checkpoint、消融开关和损失项是否工作。
  - 典型特征：`max_train_batches=1`、`max_val_batches=1`、`num_epochs=0`。
- `protocol-check`
  - 用于验证训练协议、checkpoint 恢复、评估脚本和统计口径是否一致。
- `comparable`
  - 只有在 split、checkpoint 来源、ADE/FDE 聚合方式、采样次数和评估脚本口径
    都对齐后，才允许打这个标签。

## 当前口径审计状态
- `train.py` 与 `evaluate_model.py` 现在已经共用底层 raw/average displacement 计算函数。
- 新增 `--val_every`，用来区分：
  - 正式训练：按 epoch 间隔验证
  - smoke/protocol-check：按极小 batch 快速验证
- 当前日志里的大多数历史结果仍属于 `smoke`，
  不能直接与论文表格或正式 baseline 结论等价。

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
- `val_dset_type`:
- `num_val_samples`:
- `grad_clip`:
- `aux_rollout_weight`:
- `rollout_residual_scale`:
- `detach_rollout_state`:
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
- Tag：`smoke`
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
- Tag：`smoke`
- Command：
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_nogating_aux_v1 --model_type cyclestate --train_stage warmup --disable_state_gating --device cuda --pin_memory --loader_num_workers 0 --batch_size 16 --best_k 2 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 1 --max_train_batches 1 --max_val_batches 1`
- Best ADE：`155.569`
- Best FDE：`298.149`
- Stability notes：
  - gating 消融路径运行正常
  - 在当前极小 smoke 设置下，指标略差于带 gating 的版本

### Run
- Name：`experiments/cyclestate/warmup_gating_noaux_v1`
- Tag：`smoke`
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
- Tag：`smoke`
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
- Tag：`smoke`
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
- Tag：`smoke`
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
- Tag：`smoke`
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
- Tag：`smoke`
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
- Tag：`smoke`
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
- Tag：`smoke`
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

## 本轮实现更新：动态 rollout / 动态 lane anchor / 评测口径收口
- 状态：
  - `protocol-check`
- 已完成实现：
  - `queue rollout` 改为真正递推，每一步使用上一步 rolled meso-state
  - `lane-consensus anchor` 改为预测期动态重聚合
  - 新增调试信号：
    - `queue_rollout_feature_seq`
    - `decoder_state_init_residual_norm`
    - `decoder_state_step_residual_norm_seq`
  - 训练内 `validate` 与独立评估脚本现在共用同一套 raw/average displacement 计算函数
  - 新增 `--val_every`，显式区分正式训练验证与 smoke 验证
- 已完成验证：
  - `python tests/test_cyclestate_protocol.py`
  - `python -m unittest discover -s tests -p 'test_cyclestate_protocol.py'`
  - `python -m py_compile D2TP/models.py D2TP/train.py D2TP/evaluate_model.py tests/test_cyclestate_protocol.py`
- 当前意义：
  - 这一步主要是让实现与科研叙事、训练验证与独立评估重新对齐。
  - 它仍然不是新的 `comparable` 性能结论。

### Run
- Name：`experiments/cyclestate/warmup_dynamic_protocol_v1`
- Tag：`protocol-check`
- Command：
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_dynamic_protocol_v1 --model_type cyclestate --train_stage warmup --device cuda --pin_memory --loader_num_workers 0 --batch_size 8 --best_k 2 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 1 --max_train_batches 1 --max_val_batches 1 --val_every 1`
- Best ADE：`59.663`
- Best FDE：`126.717`
- Stability notes：
  - warm-start 正常，日志显示 `skipped 0 keys`
  - 单 batch 训练和验证均正常完成
  - 新路径上的 rollout losses 正常激活：
    - `QRollReg 0.358161`
    - `QRollCls 0.704186`
  - 该结果属于 `protocol-check`，不能直接与论文或正式 baseline 表格比较

### Run
- Name：`experiments/cyclestate/warmup_bestofk_protocol_v1`
- Tag：`protocol-check`
- Command：
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_bestofk_protocol_v1 --model_type cyclestate --train_stage warmup --device cuda --pin_memory --loader_num_workers 0 --batch_size 8 --best_k 2 --num_val_samples 4 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 1 --max_train_batches 1 --max_val_batches 1 --val_every 1`
- Best ADE：`52.322`
- Best FDE：`109.916`
- Stability notes：
  - warm-start 正常，日志显示 `skipped 0 keys`
  - 训练内验证已经切到多采样 best-of-K 口径（`num_val_samples=4`）
  - 与上一条单采样 `protocol-check` 相比，验证指标进一步下降：
    - 单采样验证：`59.663 / 126.717`
    - 多采样验证：`52.322 / 109.916`
  - 这说明“训练内验证是否采用多采样 best-of-K”会真实影响 checkpoint 选择
  - 该结果仍属于 `protocol-check`，不应直接拿去和论文表格比较

## 下一轮正式实验矩阵
1. `baseline_audit_v1`
   - 标签目标：`protocol-check -> comparable`
   - 作用：统一口径复核 `D2TP/model_best.pth.tar`
2. `cyclestate_warmup_main_v2`
   - 标签目标：`protocol-check`
   - 配置：`residual on + rollout on + lane anchor on + gating on`
3. `cyclestate_refine_main_v2`
   - 标签目标：`protocol-check`
   - 配置：从最佳 warmup checkpoint 续训
4. `cyclestate_refine_no_rollout_v2`
   - 标签目标：`protocol-check`
   - 配置：主配置仅关闭 rollout
5. `cyclestate_refine_no_anchor_v2`
   - 标签目标：`protocol-check`
   - 配置：主配置仅关闭 lane anchor

## Baseline Audit：统一口径可控复核（进行中）
### Run
- Name：`baseline_audit_v1_val_partial`
- Tag：`protocol-check`
- Command：
  `python D2TP/evaluate_model.py --model_type d2tpred --device cuda --loader_num_workers 0 --batch_size 16 --num_samples 4 --max_eval_batches 5 --eval_print_every 1 --resume D2TP/model_best.pth.tar --dset_type val`
- ADE：`40.673`
- FDE：`79.941`
- Notes：
  - 这是统一口径、可控批次数的 baseline audit 部分结果，不是完整 `val` split 最终结论。
  - 主要意义是确认：
    - `evaluate_model.py` 现在支持进度输出
    - 支持 `max_eval_batches`
    - scene-level best-of-K 聚合与总轨迹归一化逻辑已生效

### Run
- Name：`baseline_audit_v1_val_full_num_samples4`
- Tag：`comparable`
- Command：
  `python D2TP/evaluate_model.py --model_type d2tpred --device cuda --loader_num_workers 0 --batch_size 16 --num_samples 4 --eval_print_every 10 --resume D2TP/model_best.pth.tar --dset_type val`
- ADE：`38.493`
- FDE：`78.706`
- Notes：
  - 这是目前仓库内最完整的 `val` split baseline 统一口径复核结果。
  - 该结果与 `baseline_audit_v1_val_partial` 一致地表明：
    baseline audit 管线已经可以用于正式 split 复核。

### Run
- Name：`baseline_audit_v1_test_full_num_samples4`
- Tag：`comparable`
- Command：
  `python D2TP/evaluate_model.py --model_type d2tpred --device cuda --loader_num_workers 0 --batch_size 16 --num_samples 4 --eval_print_every 10 --resume D2TP/model_best.pth.tar --dset_type test`
- ADE：`17.812`
- FDE：`37.568`
- Notes：
  - 这是目前仓库内最完整的 `test` split baseline 统一口径复核结果。
  - 与完整 `val` 结果的明显落差提示：
    后续所有“超过 baseline”的说法必须明确 split 和采样口径，
    不能再把 `val/test` 混写。
  - `num_samples=20` 的更接近论文口径审计已启动过一次，
    但本轮未跑完全量，因此暂不记作正式结果。

## Warmup Main v2：短正式原型（进行中）
### Run
- Name：`experiments/cyclestate/warmup_main_v2_proto`
- Tag：`protocol-check`
- Command：
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_main_v2_proto --model_type cyclestate --train_stage warmup --device cuda --pin_memory --loader_num_workers 0 --batch_size 8 --best_k 4 --num_val_samples 4 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 5 --max_train_batches 20 --max_val_batches 5 --val_every 1`
- Best ADE：`60.857`
- Best FDE：`118.979`
- Stability notes：
  - 完整 warm-start 正常：`skipped 0 keys`
  - 训练内验证已使用多采样 best-of-K 口径：`num_val_samples=4`
  - 在短原型设置下，验证指标出现了明确下降轨迹：
    - 第一次验证：`103.942 / 206.707`
    - 中间最低：`92.723 / 189.754`
    - 当前最佳：`60.857 / 118.979`
  - queue/cycle 结构化辅助项与 rollout losses 同步下降，
    说明当前主配置不仅能跑通，而且已经开始出现更像正式训练的优化信号

### Run
- Name：`experiments/cyclestate/warmup_main_v2`
- Tag：`protocol-check`
- Command：
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_main_v2 --model_type cyclestate --train_stage warmup --device cuda --pin_memory --loader_num_workers 0 --batch_size 8 --best_k 4 --num_val_samples 4 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 20 --max_train_batches 100 --max_val_batches 20 --val_every 1`
- Best ADE：`56.827`
- Best FDE：`107.416`
- Stability notes：
  - 该 run 使用的是修复前的 `should_run_validation()` 逻辑，
    会在 `batch 0` 之后立刻做一次 20-batch 验证。
  - 因此它更适合证明“主配置确实在继续学习”，
    不适合当作最干净的 schedule 对照实验。
  - 关键信号：
    - 第一个验证点：`99.934 / 195.568`
    - 第二个验证点：`56.827 / 107.416`
  - 训练 batch 20 时，日志中的生成器重建项已经降到：
    - `L2_Loss 56.534`
    - `QRollReg 0.015945`
    - `QRollCls 0.417931`

## Stage 21：验证调度修复后的 Rollout 短对照
### Run
- Name：`experiments/cyclestate/warmup_main_v2_schedfix`
- Tag：`protocol-check`
- Command：
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_main_v2_schedfix --model_type cyclestate --train_stage warmup --device cuda --pin_memory --loader_num_workers 0 --batch_size 8 --best_k 4 --num_val_samples 4 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 20 --max_train_batches 20 --max_val_batches 20 --val_every 1`
- Best ADE：`78.227`
- Best FDE：`152.544`
- Notes：
  - 使用修复后的验证规则：
    `smoke / protocol-check` 模式不再在 `batch 0` 后立刻验证，
    而是在 `print_every` 区间末尾或最后一个 batch 验证。
  - 该结果代表“rollout on”的当前短协议基线。

### Run
- Name：`experiments/cyclestate/warmup_no_rollout_v2_schedfix`
- Tag：`protocol-check`
- Command：
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_no_rollout_v2_schedfix --model_type cyclestate --train_stage warmup --disable_queue_rollout --device cuda --pin_memory --loader_num_workers 0 --batch_size 8 --best_k 4 --num_val_samples 4 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 20 --max_train_batches 20 --max_val_batches 20 --val_every 1`
- Best ADE：`71.863`
- Best FDE：`140.974`
- Notes：
  - 这是与主配置完全匹配、仅关闭 rollout 的短协议对照。
  - 当前 `no_rollout` 反而优于 `rollout on`，
    说明真正递推的 meso-state memory 还没有被当前 warmup 配方训顺。
  - 下一步应优先修正 rollout 注入与监督策略，
    而不是直接推进 `refine` 或重新引入 GAN。

## Stage 22：Rollout 路径根因修复
### Diagnosis
- 根因 1：
  - 训练态 rollout 在 step 0 使用了 teacher-forced future offset，
    而推理态使用的是最后观测 offset。
  - 这意味着训练态和推理态并没有共享同一套单步 meso-state 演化逻辑。
- 根因 2：
  - rollout 开启后，decoder 会直接使用 `rollout_queue_h_t`
    作为 queue decode context。
  - 对观测期 `gated_queue_last` 没有锚定，等价于让 rollout hidden
    在 very-short warmup 下整块接管 meso context。

### Code Fix
- 训练态 step-0 rollout 现在改为：
  - 使用最后观测 offset 作为上一时刻已知运动
- decoder queue context 现在改为：
  - `observed queue context + gated rollout delta`
  - 不再直接用 `rollout_queue_h_t` 整块替换
- 已新增对应回归测试：
  - `test_training_rollout_step_zero_uses_last_observed_offset`
  - `test_rollout_decode_context_is_anchored_to_observed_queue_state`

### Run
- Name：`experiments/cyclestate/warmup_main_v2_schedfix_rollfix_v2`
- Tag：`protocol-check`
- Command：
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_main_v2_schedfix_rollfix_v2 --model_type cyclestate --train_stage warmup --device cuda --pin_memory --loader_num_workers 0 --batch_size 8 --best_k 4 --num_val_samples 4 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 20 --max_train_batches 20 --max_val_batches 20 --val_every 1`
- Best ADE：`66.793`
- Best FDE：`132.168`
- Notes：
  - 相比修复前的 `warmup_main_v2_schedfix`：
    - ADE：`78.227 -> 66.793`
    - FDE：`152.544 -> 132.168`
  - 当前已经重新优于短协议 `no_rollout` 对照：
    - `71.863 / 140.974`
  - 这说明 rollout 主线的 scientific story 仍然成立，
    之前的问题主要来自训练/注入路径，而不是 idea 本身失效。

## Stage 23：Matched Warmup Stability 与 Rollout-Aux 独立调节
### Run
- Name：`experiments/cyclestate/warmup_main_v2_schedfix_rollfix_100b`
- Tag：`protocol-check`
- Command：
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_main_v2_schedfix_rollfix_100b --model_type cyclestate --train_stage warmup --device cuda --pin_memory --loader_num_workers 0 --batch_size 8 --best_k 4 --num_val_samples 4 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 50 --max_train_batches 100 --max_val_batches 20 --val_every 1`
- Checkpoints：
  - `batch 50`: `ADE 71.978 / FDE 135.287`
  - `batch 100`: `ADE 185.583 / FDE 322.389`
- Notes：
  - 默认 rollout-on 在更长 matched warmup 下，
    到 `batch 50` 仍然略输 `no_rollout`
  - 到 `batch 100` 时已经明显崩坏

### Run
- Name：`experiments/cyclestate/warmup_no_rollout_v2_schedfix_100b`
- Tag：`protocol-check`
- Command：
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_no_rollout_v2_schedfix_100b --model_type cyclestate --train_stage warmup --disable_queue_rollout --device cuda --pin_memory --loader_num_workers 0 --batch_size 8 --best_k 4 --num_val_samples 4 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 50 --max_train_batches 100 --max_val_batches 20 --val_every 1`
- Checkpoints：
  - `batch 50`: `ADE 67.747 / FDE 124.741`
  - `batch 100`: `ADE 196.345 / FDE 331.583`
- Notes：
  - `no_rollout` 在 `batch 50` 仍优于默认 rollout-on
  - 但到 `batch 100` 同样崩坏，
    说明当前 warmup 协议在更长短程训练上整体不稳

### Code Change
- 新增 `--aux_rollout_weight`
- 默认行为：
  - 若未显式指定，则 `aux_rollout_weight = aux_queue_weight`
  - 即保持历史实验口径兼容
- 目的：
  - 允许单独削弱 rollout auxiliary supervision，
    而不同时削弱观测期 queue-state supervision

### Run
- Name：`experiments/cyclestate/warmup_main_v2_schedfix_rollfix_aux025_50b`
- Tag：`protocol-check`
- Command：
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_main_v2_schedfix_rollfix_aux025_50b --model_type cyclestate --train_stage warmup --device cuda --pin_memory --loader_num_workers 0 --batch_size 8 --best_k 4 --num_val_samples 4 --aux_rollout_weight 2.5 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 50 --max_train_batches 50 --max_val_batches 20 --val_every 1`
- Best ADE：`66.761`
- Best FDE：`122.728`
- Notes：
  - 这是当前最好的 `50-batch rollout-on` 结果
  - 相比默认 rollout-on：
    - `71.978 / 135.287 -> 66.761 / 122.728`
  - 也重新优于 `no_rollout@50b`：
    - `67.747 / 124.741`

### Run
- Name：`experiments/cyclestate/warmup_main_v2_schedfix_rollfix_aux010_100b`
- Tag：`protocol-check`
- Command：
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_main_v2_schedfix_rollfix_aux010_100b --model_type cyclestate --train_stage warmup --device cuda --pin_memory --loader_num_workers 0 --batch_size 8 --best_k 4 --num_val_samples 4 --aux_rollout_weight 1.0 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 50 --max_train_batches 100 --max_val_batches 20 --val_every 1`
- Checkpoints：
  - `batch 50`: `ADE 71.872 / FDE 134.018`
  - `batch 100`: `ADE 190.641 / FDE 320.417`
- Notes：
  - 把 rollout aux 直接降到 `1.0` 并没有改善 `batch 50` 结果，
    反而明显差于 `2.5`
  - 这说明 rollout aux 不能被简单粗暴砍得过低

### Stage 23 Conclusion
- 当前最合理的解释是：
  1. rollout aux 确实需要从 `aux_queue_weight` 中独立出来
  2. 当前短程最佳点更接近 `aux_rollout_weight=2.5`，而不是 `1.0`
  3. 当前更大的问题已经从“rollout 是否有效”
     转移到“warmup 在更长短程训练上整体不稳”

## Stage 24：协议优先稳定化实现
### Code Change
- 训练内验证 split 从历史上的隐式 `test` 改为可配置参数：
  - 新增 `--val_dset_type {val,test}`
  - 默认 `val`
  - `test` 仅用于最终复核或兼容旧协议
- 修复 `--lr` 被忽略的问题：
  - 新增 `build_optimizers()`
  - 生成器和判别器 RMSprop 均使用 `args.lr`
- 新增 warmup/refine 稳定化默认值：
  - `grad_clip=1.0`
  - warmup `rollout_residual_scale=0.35`
  - refine/adversarial `rollout_residual_scale=0.7`
  - warmup `detach_rollout_state=True`
- `--detach_rollout_state` 支持显式开关：
  - `--detach_rollout_state`
  - `--no_detach_rollout_state`
- `CycleStateTrajectoryGenerator` 新增 bounded rollout residual injection：
  - decoder queue context 仍为 `observed queue context + rollout delta`
  - rollout delta 先经过 `tanh` 限幅
  - 再乘以 `rollout_residual_scale`
  - 当 `rollout_residual_scale=0` 时，严格退化为纯 observed queue context
- 新增 warmup rollout-state detach：
  - 预测期每一步 rollout 仍产生 queue prediction 与 decoder context
  - 但 warmup 默认截断下一步 rollout state 的跨步反传
  - 目的：先学可预测的 meso-state，再学习长链状态耦合
- 新增训练日志稳定性指标：
  - `StateStability | DInitNorm ... DStepNorm ... QRollHNorm ... PredOffsetNorm ... GradNorm ...`
  - TensorBoard scalar：
    - `state_decoder_init_residual_norm`
    - `state_decoder_step_residual_norm`
    - `state_queue_rollout_hidden_norm`
    - `state_pred_offset_norm`
    - `g_grad_norm`
- `evaluate_model.py` 同步支持：
  - `--rollout_residual_scale`
  - `--detach_rollout_state`
  - 以保证新协议 checkpoint 能按同构模型离线评估

### Tests
- 新增/更新单元测试覆盖：
  - 训练内验证默认走 `val` split
  - `--val_dset_type test` 可显式恢复旧 split
  - `--lr` 能真实传入优化器
  - `warmup` 默认启用 `grad_clip`、低强度 rollout residual、rollout-state detach
  - `--no_detach_rollout_state` 能覆盖 warmup 默认
  - `rollout_residual_scale=0` 退化为 observed queue context
  - `detach_rollout_state=True` 时 rollout prediction 仍存在，但第二步 hidden 输入已 detach
  - 状态稳定性日志 helper 能正确汇总 debug norms
  - `evaluate_model.py` 暴露新协议参数

### Recommended Next Runs
1. `baseline_audit_v2_val_full_num_samples20`
   - Tag：`comparable`
   - Command：
     `python D2TP/evaluate_model.py --model_type d2tpred --device cuda --loader_num_workers 0 --batch_size 16 --num_samples 20 --eval_print_every 10 --resume D2TP/model_best.pth.tar --dset_type val`
2. `baseline_audit_v2_test_full_num_samples20`
   - Tag：`comparable`
   - Command：
     `python D2TP/evaluate_model.py --model_type d2tpred --device cuda --loader_num_workers 0 --batch_size 16 --num_samples 20 --eval_print_every 10 --resume D2TP/model_best.pth.tar --dset_type test`
3. `experiments/cyclestate/warmup_protocol_stable_v1_50b`
   - Tag：`protocol-check`
   - Command：
     `python D2TP/train.py --log_dir experiments/cyclestate/warmup_protocol_stable_v1_50b --model_type cyclestate --train_stage warmup --device cuda --pin_memory --loader_num_workers 0 --batch_size 8 --best_k 4 --num_val_samples 4 --aux_rollout_weight 2.5 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 50 --max_train_batches 50 --max_val_batches 20 --val_dset_type val`
4. `experiments/cyclestate/warmup_protocol_stable_v1_100b`
   - Tag：`protocol-check`
   - Command：
     `python D2TP/train.py --log_dir experiments/cyclestate/warmup_protocol_stable_v1_100b --model_type cyclestate --train_stage warmup --device cuda --pin_memory --loader_num_workers 0 --batch_size 8 --best_k 4 --num_val_samples 4 --aux_rollout_weight 2.5 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 50 --max_train_batches 100 --max_val_batches 20 --val_dset_type val`

### Current Interpretation
- Stage 24 不改变科研故事，也不引入外部通用结构。
- 本阶段专门解决此前 `100-batch` 后半程崩坏暴露出的协议问题：
  - split 混用
  - learning rate 参数无效
  - 缺少梯度约束
  - rollout delta 注入缺少幅值约束
  - warmup 阶段状态链路反传过长
- 下一步判断标准固定为：
  - `100-batch` 的 true-val ADE 不得比 `50-batch` 恶化超过 `15%`
  - rollout-on 仍需优于匹配 no-rollout
  - 通过后再进入 `refine`

### Run
- Name：`experiments/cyclestate/warmup_protocol_stable_v1_50b`
- Tag：`protocol-check`
- Command：
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_protocol_stable_v1_50b --model_type cyclestate --train_stage warmup --device cuda --pin_memory --loader_num_workers 0 --batch_size 8 --best_k 4 --num_val_samples 4 --aux_rollout_weight 2.5 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 50 --max_train_batches 50 --max_val_batches 20 --val_dset_type val`
- Best ADE：`87.082`
- Best FDE：`175.723`
- Stability notes：
  - true-val 协议、`lr=1e-3`、`grad_clip=1.0`、`rollout_residual_scale=0.35`
  - 初始 `StateStability`：
    - `QRollHNorm 0.357492`
    - `PredOffsetNorm 14.647298`
    - `GradNorm 631.850952`
  - 与此前使用 test split 的短结果不能直接比较；这是 Stage 24 的 true-val 起点。

### Run
- Name：`experiments/cyclestate/warmup_protocol_stable_v1_100b`
- Tag：`protocol-check`
- Command：
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_protocol_stable_v1_100b --model_type cyclestate --train_stage warmup --device cuda --pin_memory --loader_num_workers 0 --batch_size 8 --best_k 4 --num_val_samples 4 --aux_rollout_weight 2.5 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 50 --max_train_batches 100 --max_val_batches 20 --val_dset_type val`
- Checkpoints：
  - `batch 50`: `ADE 87.082 / FDE 175.723`
  - `batch 100`: `ADE 231.420 / FDE 420.862`
- Stability notes：
  - `batch 100` 相对 `batch 50` 恶化 `165.7%`
  - 15% 稳定性阈值对应 `ADE <= 100.144`，当前未通过
  - `batch 50` 稳定性指标：
    - `DInitNorm 0.033967`
    - `DStepNorm 0.037403`
    - `QRollHNorm 3.211943`
    - `PredOffsetNorm 8.711025`
    - `GradNorm 2165.728516`
  - 结论：Stage 24 默认稳定化没有解决 true-val 100b 崩坏。

### Run
- Name：`experiments/cyclestate/warmup_protocol_stable_v1_lr0003_100b`
- Tag：`protocol-check`
- Command：
  `python D2TP/train.py --log_dir experiments/cyclestate/warmup_protocol_stable_v1_lr0003_100b --model_type cyclestate --train_stage warmup --lr 0.0003 --device cuda --pin_memory --loader_num_workers 0 --batch_size 8 --best_k 4 --num_val_samples 4 --aux_rollout_weight 2.5 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 50 --max_train_batches 100 --max_val_batches 20 --val_dset_type val`
- Checkpoints：
  - `batch 50`: `ADE 88.598 / FDE 171.890`
  - `batch 100`: `ADE 226.302 / FDE 411.163`
- Stability notes：
  - 降低学习率后，`batch 50` 的状态范数更温和：
    - `QRollHNorm 0.927990`
    - `PredOffsetNorm 10.118897`
    - `GradNorm 1167.409912`
  - 但 `batch 100` 仍明显崩坏，说明单纯降低学习率不足以解决问题。

### Stage 24 Experimental Conclusion
- true-val 口径比历史 test-split protocol-check 更严格，当前 `50b` 指标回到 `ADE 87-89` 区间。
- 默认稳定化与 `lr=3e-4` 都未通过 `100b` 稳定性门槛。
- 进一步代码审查显示，当前不宜立刻把下一步收敛为 exposure-bias 调参；有两个更基础的高优先级问题需要先修：
  1. `seqGAT` 前向被 `torch.no_grad()` 包裹，局部时序图注意力参数实际不参与训练
  2. `relation_Matrix` 在正常方向区间下会把距离内邻居无条件连边，方向扇区约束失真

## Stage 25：基础交互建模 P0 修复
### Code Change
1. 恢复 `seqGAT` 可训练
   - 已移除 `TrajectoryGenerator` 与 `CycleStateTrajectoryGenerator`
     中 `seqgatencoder` 外层的 `torch.no_grad()`
2. 修复 `relation_Matrix` 方向扇区 bug
   - 已将正常区间 `else` 分支改为
     `down <= dire_n_neig <= up`
   - 不再对距离内邻居无条件设边

### Tests
- 新增梯度流测试：
  - `test_seqgat_parameters_receive_gradients_in_baseline_generator`
  - `test_seqgat_parameters_receive_gradients_in_cyclestate_generator`
- 新增方向扇区测试：
  - `test_relation_matrix_respects_direction_sector_in_normal_range`
- 当前协议测试总数：
  - `38`
  - 全部通过

### Smoke Verification
- Name：`quick_runs/stage25_p0_cpu_smoke`
- Tag：`smoke`
- Command：
  `python D2TP/train.py --log_dir quick_runs/stage25_p0_cpu_smoke --model_type cyclestate --train_stage warmup --device cpu --loader_num_workers 0 --batch_size 2 --best_k 1 --num_val_samples 1 --resume D2TP/model_best.pth.tar --num_epochs 0 --print_every 1 --max_train_batches 1 --max_val_batches 1`
- Result：
  - `ADE 39.382 / FDE 70.956`
  - 训练、验证、checkpoint、`StateStability` 日志均正常

### P1 Follow-ups
1. `randn init -> zero init` 受控实验
2. GAN label smoothing 修复

### Updated Next Runs
1. `experiments/cyclestate/warmup_p0_seqgat_relation_v1_50b`
2. `experiments/cyclestate/warmup_p0_seqgat_relation_v1_100b`

### Updated Decision Rule
- Stage 25 的 P0 修复已完成，下一步直接回到 true-val 稳定性复测。
- 若修复后 `100b` 仍明显崩坏，再进入：
  - 降低 `teacher_forcing_ratio`
  - 或 50b warmup 后提前切到 `refine`
