2026-08-10 15:59:13,781 INFO Checkpoint dir: checkpoints/phase1_run77
2026-08-10 15:59:13,846 INFO Starting training: phase=phase1_get_possession, total_steps=6,000,000
2026-08-10 15:59:13,846 INFO Phase description: 1v1 scenario: learn to get possession and bring the ball toward the opponent box.  Decision network frozen except Move/GetPossession and latent vector.
2026-08-10 15:59:13,848 INFO _from_ckpt: overriding bc_pretrain_epochs=18 → 18
2026-08-10 15:59:13,849 INFO _from_ckpt: overriding demo_value_pretrain_epochs=25 → 20
2026-08-10 15:59:13,849 INFO _from_ckpt: overriding value_pretrain_epochs=55 → 50
2026-08-10 15:59:13,849 INFO _from_ckpt: overriding aux_coeff_start=0.0 → 0.0
2026-08-10 15:59:13,849 INFO _from_ckpt: overriding aux_coeff_end=0.0 → 0.0
2026-08-10 15:59:13,849 INFO _from_ckpt: overriding aux_coeff_anneal_fraction=0.3 → 0.35
/home/vincent/Documents/not_work/repos/FootballCoach/.venv/lib/python3.12/site-packages/torch/_compile.py:54: UserWarning: optimizer contains a parameter group with duplicate parameters; in future, this will cause an error; see github.com/pytorch/pytorch/issues/40967 for more information
  return disable_fn(*args, **kwargs)
2026-08-10 15:59:15,061 INFO Logging to checkpoints/phase1_run77/training_log1.txt
2026-08-10 15:59:15,062 INFO --latest(-pretrain): resolved to checkpoints/phase1_run73/latest.pt
2026-08-10 15:59:15,245 INFO Loaded checkpoint: checkpoints/phase1_run73/latest.pt (step 6021000)
2026-08-10 15:59:15,245 INFO Loaded checkpoint for re-pretraining: checkpoints/phase1_run73/latest.pt — will still run BC/value pre-training
2026-08-10 15:59:15,245 INFO --reset-dir-log-std: move_dir_log_std=-1.8  kick_dir_log_std=-1.6
2026-08-10 15:59:15,270 INFO Loading 3128 demonstration file(s) from demonstrations/phase1_long
2026-08-10 15:59:39,305 INFO Dataset: 2,282,732 steps loaded
2026-08-10 15:59:39,309 INFO Offline BC dataset: 2,282,732 steps from demonstrations/phase1_long/
2026-08-10 15:59:39,309 INFO Reward diagnostic: running 40 rules-vs-rules episodes...
2026-08-10 15:59:40,146 INFO Reward diagnostic (40 ep, rules vs rules): mean_ep_rew=2.01  per_episode: get_possession=+1.00  lose_possession=-0.29  box_possession=+2.70  opponent_box=-1.20  step_penalty=-0.06  stamina_penalty=-0.14
2026-08-10 15:59:40,193 INFO BC pos_weight (auto-computed from dataset): kick=1.35  tackle_attempt=1.35
2026-08-10 15:59:40,193 INFO Combined BC + value pre-training: 18 epoch(s), batch_size=2048, dataset=2,282,732 steps, rollout_steps=170000
2026-08-10 15:59:40,760 INFO Phase 0 — decision-net warm-up (BC + self.value_net MSE; single value head convention): 20 epoch(s), gamma=0.995, returns mean=3.28  std=1.03  lr=0.002  phase0_value_coef=1.0  split: 985,640 train / 172,375 val rows
/home/vincent/Documents/not_work/repos/FootballCoach/src/footballcoach/ai/ppo/bc.py:706: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /__w/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:822.)
  "decision":       float(dec_loss[valid].mean()),
