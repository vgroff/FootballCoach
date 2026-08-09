2026-08-09 18:31:38,866 INFO Checkpoint dir: checkpoints/phase1_run54
2026-08-09 18:31:38,922 INFO Starting training: phase=phase1_get_possession, total_steps=6,000,000
2026-08-09 18:31:38,922 INFO Phase description: 1v1 scenario: learn to get possession and bring the ball toward the opponent box.  Decision network frozen except Move/GetPossession and latent vector.
/home/vincent/Documents/not_work/repos/FootballCoach/.venv/lib/python3.12/site-packages/torch/_compile.py:54: UserWarning: optimizer contains a parameter group with duplicate parameters; in future, this will cause an error; see github.com/pytorch/pytorch/issues/40967 for more information
  return disable_fn(*args, **kwargs)
2026-08-09 18:31:40,254 INFO Logging to checkpoints/phase1_run54/training_log1.txt
2026-08-09 18:31:40,445 INFO Loading 2500 demonstration file(s) from demonstrations/phase1
2026-08-09 18:31:53,348 INFO Dataset: 1,117,070 steps loaded
2026-08-09 18:31:53,350 INFO Offline BC dataset: 1,117,070 steps from demonstrations/phase1/
2026-08-09 18:31:53,350 INFO Reward diagnostic: running 40 rules-vs-rules episodes...
2026-08-09 18:31:54,456 INFO Reward diagnostic (40 ep, rules vs rules): mean_ep_rew=2.79  per_episode: get_possession=+1.18  lose_possession=-0.36  box_possession=+3.00  opponent_box=-0.80  step_penalty=-0.09  stamina_penalty=-0.14
2026-08-09 18:31:54,480 INFO BC pos_weight (auto-computed from dataset): kick=1.50  tackle_attempt=1.50
2026-08-09 18:31:54,480 INFO Combined BC + value pre-training: 9 epoch(s), batch_size=1000, dataset=1,117,070 steps, rollout_steps=35000
2026-08-09 18:31:54,774 INFO Phase 0 — decision-net warm-up (BC + self.value_net MSE; single value head convention): 6 epoch(s), gamma=0.995, returns mean=3.60  std=1.17  lr=0.005  phase0_value_coef=0.85  split: 485,522 train / 85,733 val rows
/home/vincent/Documents/not_work/repos/FootballCoach/src/footballcoach/ai/ppo/bc.py:703: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /__w/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:822.)
  "decision":       float(dec_loss[valid].mean()),
2026-08-09 18:34:01,261 INFO Checkpoint dir: checkpoints/phase1_run55
2026-08-09 18:34:01,321 INFO Starting training: phase=phase1_get_possession, total_steps=6,000,000
2026-08-09 18:34:01,321 INFO Phase description: 1v1 scenario: learn to get possession and bring the ball toward the opponent box.  Decision network frozen except Move/GetPossession and latent vector.
/home/vincent/Documents/not_work/repos/FootballCoach/.venv/lib/python3.12/site-packages/torch/_compile.py:54: UserWarning: optimizer contains a parameter group with duplicate parameters; in future, this will cause an error; see github.com/pytorch/pytorch/issues/40967 for more information
  return disable_fn(*args, **kwargs)
2026-08-09 18:34:02,728 INFO Logging to checkpoints/phase1_run55/training_log1.txt
2026-08-09 18:34:02,930 INFO Loading 2500 demonstration file(s) from demonstrations/phase1
2026-08-09 18:34:16,960 INFO Dataset: 1,117,070 steps loaded
2026-08-09 18:34:16,963 INFO Offline BC dataset: 1,117,070 steps from demonstrations/phase1/
2026-08-09 18:34:16,963 INFO Reward diagnostic: running 40 rules-vs-rules episodes...
2026-08-09 18:34:18,112 INFO Reward diagnostic (40 ep, rules vs rules): mean_ep_rew=2.22  per_episode: get_possession=+1.27  lose_possession=-0.52  box_possession=+2.80  opponent_box=-1.10  step_penalty=-0.08  stamina_penalty=-0.15
2026-08-09 18:34:18,135 INFO BC pos_weight (auto-computed from dataset): kick=1.50  tackle_attempt=1.50
2026-08-09 18:34:18,135 INFO Combined BC + value pre-training: 9 epoch(s), batch_size=1000, dataset=1,117,070 steps, rollout_steps=35000
2026-08-09 18:34:18,443 INFO Phase 0 — decision-net warm-up (BC + self.value_net MSE; single value head convention): 6 epoch(s), gamma=0.995, returns mean=3.60  std=1.17  lr=0.005  phase0_value_coef=0.85  split: 485,522 train / 85,733 val rows
/home/vincent/Documents/not_work/repos/FootballCoach/src/footballcoach/ai/ppo/bc.py:703: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /__w/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:822.)
  "decision":       float(dec_loss[valid].mean()),
2026-08-09 18:37:47,384 INFO   Phase 0 epoch 1/6: loss=3.7328  dec_bc=2.7051  bc_adj=1.3326(floor=1.3725)  val_mse=1.2090(x0.85)=1.0277
2026-08-09 18:38:01,355 INFO     val  p0_val_loss=3.3589
2026-08-09 18:41:27,113 INFO   Phase 0 epoch 2/6: loss=2.5919  dec_bc=1.9217  bc_adj=0.5491(floor=1.3725)  val_mse=0.7885(x0.85)=0.6703
2026-08-09 18:41:41,230 INFO     val  p0_val_loss=2.1216
2026-08-09 18:45:03,560 INFO   Phase 0 epoch 3/6: loss=2.0416  dec_bc=1.3961  bc_adj=0.0236(floor=1.3725)  val_mse=0.7594(x0.85)=0.6455
2026-08-09 18:45:17,664 INFO     val  p0_val_loss=2.0802
2026-08-09 18:48:42,074 INFO   Phase 0 epoch 4/6: loss=2.0187  dec_bc=1.3953  bc_adj=0.0228(floor=1.3725)  val_mse=0.7333(x0.85)=0.6233
2026-08-09 18:48:56,476 INFO     val  p0_val_loss=2.0797
