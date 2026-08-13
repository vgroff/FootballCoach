2026-08-13 16:12:24,528 INFO Checkpoint dir: checkpoints/phase1_run106
2026-08-13 16:12:25,136 INFO Starting training: phase=phase1_get_possession, total_steps=6,000,000
2026-08-13 16:12:25,137 INFO Phase description: 1v1 scenario: learn to get possession and bring the ball toward the opponent box.  Decision network frozen except Move/GetPossession and latent vector.
2026-08-13 16:12:25,152 INFO _from_ckpt: overriding bc_pretrain_epochs=18 → 2
2026-08-13 16:12:25,152 INFO _from_ckpt: overriding demo_value_pretrain_epochs=25 → 0
2026-08-13 16:12:25,152 INFO _from_ckpt: overriding value_pretrain_epochs=55 → 50
2026-08-13 16:12:25,152 INFO _from_ckpt: overriding aux_coeff_start=0.0 → 0.0
2026-08-13 16:12:25,152 INFO _from_ckpt: overriding aux_coeff_end=0.0 → 0.0
2026-08-13 16:12:25,152 INFO _from_ckpt: overriding aux_coeff_anneal_fraction=0.3 → 0.35
/home/vincent/Documents/not_work/repos/FootballCoach/.venv/lib/python3.12/site-packages/torch/_compile.py:54: UserWarning: optimizer contains a parameter group with duplicate parameters; in future, this will cause an error; see github.com/pytorch/pytorch/issues/40967 for more information
  return disable_fn(*args, **kwargs)
2026-08-13 16:12:38,507 INFO Logging to checkpoints/phase1_run106/training_log1.txt
2026-08-13 16:12:39,836 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vgood_immobile.pt (step 5796000)
2026-08-13 16:12:39,836 INFO Loaded checkpoint for re-pretraining: checkpoints/longterm/checkpoint_vgood_immobile.pt — will still run BC/value pre-training
2026-08-13 16:12:39,836 INFO --reset-dir-log-std: move_dir_log_std=-2.3  kick_dir_log_std=-1.6
2026-08-13 16:12:40,017 INFO Loading 2752 demonstration file(s) from demonstrations/phase1_long
2026-08-13 16:14:08,219 INFO Dataset: 1,316,196 steps loaded
2026-08-13 16:14:08,227 INFO Offline BC dataset: 1,316,196 steps from demonstrations/phase1_long/
2026-08-13 16:14:08,227 INFO Reward diagnostic: running 40 rules-vs-rules episodes...
2026-08-13 16:14:12,166 INFO Reward diagnostic (40 ep, rules vs rules): mean_ep_rew=3.30  per_episode: get_possession=+1.43  lose_possession=-0.56  box_possession=+2.40  speed_bonus=+0.88  opponent_box=-0.60  timeout=-0.15  stamina_penalty=-0.09
2026-08-13 16:14:12,318 INFO BC pos_weight (auto-computed from dataset): kick=1.35  tackle_attempt=1.35
2026-08-13 16:14:12,320 INFO Combined BC + value pre-training: 2 epoch(s), batch_size=2048, dataset=1,316,196 steps, rollout_steps=120000
2026-08-13 16:14:13,850 INFO   BC pretrain split: 571,340 train rows  |  100,459 val rows
2026-08-13 16:14:21,631 INFO   Downsample trivial rows (epoch 1): 112,252/671,799 (16.7%) rows classified trivial, excluding ~95,414 this epoch (frac=0.85)
/home/vincent/Documents/not_work/repos/FootballCoach/src/footballcoach/ai/ppo/bc.py:706: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /__w/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:822.)
  "decision":       float(dec_loss[valid].mean()),
