2026-08-09 23:47:43,610 INFO Dataset: 2,357,564 steps loaded
2026-08-09 23:47:43,616 INFO Offline BC dataset: 2,357,564 steps from demonstrations/phase1_long/
2026-08-09 23:47:43,616 INFO Reward diagnostic: running 40 rules-vs-rules episodes...
2026-08-09 23:47:44,530 INFO Reward diagnostic (40 ep, rules vs rules): mean_ep_rew=1.89  per_episode: get_possession=+1.07  lose_possession=-0.36  box_possession=+2.70  opponent_box=-1.30  step_penalty=-0.08  stamina_penalty=-0.14
2026-08-09 23:47:44,577 INFO BC pos_weight (auto-computed from dataset): kick=1.35  tackle_attempt=1.35
2026-08-09 23:47:44,577 INFO Combined BC + value pre-training: 9 epoch(s), batch_size=1500, dataset=2,357,564 steps, rollout_steps=130000
2026-08-09 23:47:45,172 INFO Phase 0 — decision-net warm-up (BC + self.value_net MSE; single value head convention): 5 epoch(s), gamma=0.995, returns mean=3.59  std=1.19  lr=0.003  phase0_value_coef=1.0  split: 1,025,556 train / 180,006 val rows
/home/vincent/Documents/not_work/repos/FootballCoach/src/footballcoach/ai/ppo/bc.py:703: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /__w/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:822.)
  "decision":       float(dec_loss[valid].mean()),
2026-08-09 23:50:51,769 INFO   Phase 0 epoch 1/5: loss=1.2021  dec_bc=0.4581  bc_adj=0.0174(floor=0.4407)  val_mse=0.7440(x1.0)=0.7440
2026-08-09 23:51:03,963 INFO     val  p0_val_loss=1.2303  best=1.2303  (improved)
2026-08-09 23:54:17,240 INFO   Phase 0 epoch 2/5: loss=1.0777  dec_bc=0.4504  bc_adj=0.0097(floor=0.4407)  val_mse=0.6273(x1.0)=0.6273
2026-08-09 23:54:29,682 INFO     val  p0_val_loss=1.1792  best=1.1792  (improved)
2026-08-09 23:58:06,299 INFO   Phase 0 epoch 3/5: loss=1.0446  dec_bc=0.4495  bc_adj=0.0088(floor=0.4407)  val_mse=0.5950(x1.0)=0.5950
2026-08-09 23:58:22,075 INFO     val  p0_val_loss=1.1954  best=1.1792  (patience 1/2)
2026-08-10 00:02:05,593 INFO   Phase 0 epoch 4/5: loss=1.0285  dec_bc=0.4496  bc_adj=0.0089(floor=0.4407)  val_mse=0.5790(x1.0)=0.5790
2026-08-10 00:02:18,198 INFO     val  p0_val_loss=1.1579  best=1.1579  (improved)
2026-08-10 00:05:35,951 INFO   Phase 0 epoch 5/5: loss=1.0188  dec_bc=0.4492  bc_adj=0.0085(floor=0.4407)  val_mse=0.5695(x1.0)=0.5695
2026-08-10 00:05:48,323 INFO     val  p0_val_loss=1.1824  best=1.1579  (patience 1/2)
2026-08-10 00:05:48,323 INFO Phase 0 done (decision-net BC + critic value_head warm-up, 5 epoch(s))
2026-08-10 00:05:48,365 INFO   BC pretrain split: 1,025,556 train rows  |  180,006 val rows
2026-08-10 00:18:19,555 INFO   BC epoch 1/9  (751.2s)
    loss       bc=1.0118  bc_adj=0.3437(floor=0.6682)
    heads      dir_cos=0.982  kick_dir_cos=0.992
               move_prob=0.932  sprint_prob=0.330  kick_prob=0.190  tackle_prob=0.011
    pr/rec     kick:   p=0.908  r=0.934  f1=0.921  (tp=677916 fp=68285 fn=47696)
               tackle: p=1.000  r=0.004  f1=0.008  (tp=5 fp=0 fn=1271)
    breakdown  decision=0.222  exec_bce=0.500  sprint=0.215  move=0.096  tackle_attempt=0.058  direction=0.052
               region=0.012  kick=0.13128  kick_direction=0.02260  kick_power=0.00193  kick_spin=0.00000
2026-08-10 00:18:31,302 INFO     val        bc_val_loss=0.9040  best=0.9040  (improved)
2026-08-10 00:31:21,897 INFO   BC epoch 2/9  (770.6s)
    loss       bc=0.9055  bc_adj=0.2373(floor=0.6682)
    heads      dir_cos=0.989  kick_dir_cos=0.996
               move_prob=0.932  sprint_prob=0.330  kick_prob=0.187  tackle_prob=0.010
    pr/rec     kick:   p=0.938  r=0.964  f1=0.951  (tp=699168 fp=46061 fn=26444)
               tackle: p=0.949  r=0.103  f1=0.185  (tp=131 fp=7 fn=1145)
    breakdown  decision=0.221  exec_bce=0.421  sprint=0.176  move=0.084  tackle_attempt=0.057  direction=0.033
               region=0.008  kick=0.10286  kick_direction=0.01165  kick_power=0.00152  kick_spin=0.00000
2026-08-10 00:31:33,681 INFO     val        bc_val_loss=0.8865  best=0.8865  (improved)
