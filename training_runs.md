2026-08-09 19:10:23,130 INFO Checkpoint dir: checkpoints/phase1_run56
2026-08-09 19:10:23,199 INFO Starting training: phase=phase1_get_possession, total_steps=6,000,000
2026-08-09 19:10:23,200 INFO Phase description: 1v1 scenario: learn to get possession and bring the ball toward the opponent box.  Decision network frozen except Move/GetPossession and latent vector.
/home/vincent/Documents/not_work/repos/FootballCoach/.venv/lib/python3.12/site-packages/torch/_compile.py:54: UserWarning: optimizer contains a parameter group with duplicate parameters; in future, this will cause an error; see github.com/pytorch/pytorch/issues/40967 for more information
  return disable_fn(*args, **kwargs)
2026-08-09 19:10:24,613 INFO Logging to checkpoints/phase1_run56/training_log1.txt
2026-08-09 19:10:24,807 INFO Loading 1625 demonstration file(s) from demonstrations/phase1
2026-08-09 19:18:22,170 INFO Checkpoint dir: checkpoints/phase1_run57
2026-08-09 19:18:22,232 INFO Starting training: phase=phase1_get_possession, total_steps=6,000,000
2026-08-09 19:18:22,233 INFO Phase description: 1v1 scenario: learn to get possession and bring the ball toward the opponent box.  Decision network frozen except Move/GetPossession and latent vector.
/home/vincent/Documents/not_work/repos/FootballCoach/.venv/lib/python3.12/site-packages/torch/_compile.py:54: UserWarning: optimizer contains a parameter group with duplicate parameters; in future, this will cause an error; see github.com/pytorch/pytorch/issues/40967 for more information
  return disable_fn(*args, **kwargs)
2026-08-09 19:18:23,540 INFO Logging to checkpoints/phase1_run57/training_log1.txt
2026-08-09 19:18:23,720 INFO Loading 1625 demonstration file(s) from demonstrations/phase1
2026-08-09 19:18:32,604 INFO Dataset: 731,992 steps loaded
2026-08-09 19:18:32,605 INFO Offline BC dataset: 731,992 steps from demonstrations/phase1/
2026-08-09 19:18:32,613 INFO Reward diagnostic: running 40 rules-vs-rules episodes...
2026-08-09 19:18:33,569 INFO Reward diagnostic (40 ep, rules vs rules): mean_ep_rew=2.00  per_episode: get_possession=+1.23  lose_possession=-0.49  box_possession=+2.70  opponent_box=-1.20  step_penalty=-0.08  stamina_penalty=-0.15
2026-08-09 19:18:33,586 INFO BC pos_weight (auto-computed from dataset): kick=1.50  tackle_attempt=1.50
2026-08-09 19:18:33,586 INFO Combined BC + value pre-training: 9 epoch(s), batch_size=1000, dataset=731,992 steps, rollout_steps=35000
2026-08-09 19:18:33,776 INFO Phase 0 — decision-net warm-up (BC + self.value_net MSE; single value head convention): 5 epoch(s), gamma=0.995, returns mean=3.59  std=1.17  lr=0.008  phase0_value_coef=1.0  split: 317,320 train / 57,054 val rows
/home/vincent/Documents/not_work/repos/FootballCoach/src/footballcoach/ai/ppo/bc.py:703: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /__w/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:822.)
  "decision":       float(dec_loss[valid].mean()),
2026-08-09 19:20:53,780 INFO   Phase 0 epoch 1/5: loss=2.9031  dec_bc=1.6998  bc_adj=0.3273(floor=1.3725)  val_mse=1.2033(x1.0)=1.2033
2026-08-09 19:21:03,864 INFO     val  p0_val_loss=2.2584
