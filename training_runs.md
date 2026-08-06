2026-08-06 13:33:44,318 INFO Checkpoint dir: checkpoints/phase1_run44
2026-08-06 13:33:44,330 INFO Starting training: phase=phase1_get_possession, total_steps=100,000
2026-08-06 13:33:44,330 INFO Phase description: 1v1 scenario: learn to get possession and bring the ball toward the opponent box.  Decision network frozen except Move/GetPossession and latent vector.
2026-08-06 13:33:44,371 INFO _from_ckpt: overriding bc_pretrain_epochs=6 → 3
2026-08-06 13:33:44,371 INFO _from_ckpt: overriding demo_value_pretrain_epochs=6 → 0
2026-08-06 13:33:44,371 INFO _from_ckpt: overriding value_pretrain_epochs=60 → 35
2026-08-06 13:33:44,371 INFO _from_ckpt: overriding aux_coeff_start=0.0 → 0.0
2026-08-06 13:33:44,372 INFO _from_ckpt: overriding aux_coeff_end=0.0 → 0.0
2026-08-06 13:33:44,372 INFO _from_ckpt: overriding aux_coeff_anneal_fraction=0.3 → 0.35
2026-08-06 13:33:45,452 INFO Logging to checkpoints/phase1_run44/training_log1.txt
2026-08-06 13:33:45,452 INFO --latest(-pretrain): resolved to checkpoints/phase1_run43/latest.pt
2026-08-06 13:33:45,612 INFO Loaded checkpoint: checkpoints/phase1_run43/latest.pt (step 100000)
2026-08-06 13:33:45,612 INFO Loaded checkpoint for re-pretraining: checkpoints/phase1_run43/latest.pt — will still run BC/value pre-training
2026-08-06 13:33:45,618 INFO Loading 750 demonstration file(s) from demonstrations/phase1
2026-08-06 13:33:48,381 INFO Dataset: 290,906 steps loaded
2026-08-06 13:33:48,382 INFO Offline BC dataset: 290,906 steps from demonstrations/phase1/
2026-08-06 13:33:48,382 INFO Reward diagnostic: running 40 rules-vs-rules episodes...
2026-08-06 13:33:48,976 INFO Reward diagnostic (40 ep, rules vs rules): mean_ep_rew=4.50  per_episode: approach_speed=+0.09  get_possession=+1.07  lose_possession=-0.71  box_possession=+3.15  speed_bonus=+1.29  opponent_box=-0.34  stamina_penalty=-0.06
2026-08-06 13:33:48,989 INFO BC pos_weight (auto-computed from dataset): kick=3.00  tackle_attempt=3.00
2026-08-06 13:33:48,990 INFO Combined BC + value pre-training: 3 epoch(s), batch_size=752, dataset=290,906 steps, rollout_steps=45000
2026-08-06 13:33:48,995 INFO   BC pretrain split: 218,155 train rows  |  39,416 val rows
2026-08-06 13:33:49,041 INFO   Downsample trivial rows (epoch 1): 103,748/257,571 (40.3%) rows classified trivial, excluding ~77,811 this epoch (frac=0.75)
/home/vincent/Documents/not_work/repos/FootballCoach/src/footballcoach/ai/ppo/bc.py:691: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /__w/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:822.)
  "decision":       float(dec_loss[valid].mean()),
2026-08-06 13:34:16,509 INFO   BC epoch 1/3  (27.5s)
    loss       bc=2.1334  bc_adj=0.1423(floor=1.9910)
    heads      dir_cos=0.982  kick_dir_cos=0.998
               move_prob=0.871  sprint_prob=0.783  kick_prob=0.069  tackle_prob=0.038
    pr/rec     kick:   p=0.987  r=0.997  f1=0.992  (tp=21851 fp=299 fn=57)
               tackle: p=0.935  r=0.866  f1=0.899  (tp=1244 fp=87 fn=192)
    breakdown  decision=0.687  exec_bce=0.700  sprint=0.199  move=0.181  tackle_attempt=0.155  direction=0.052
               region=0.008  kick=0.16498  kick_direction=0.00487  kick_power=0.00076  kick_spin=0.00000
2026-08-06 13:34:17,224 INFO     val        bc_val_loss=2.1093  best=2.1093  (improved)
2026-08-06 13:34:17,231 INFO   Downsample trivial rows (epoch 2): 103,748/257,571 (40.3%) rows classified trivial, excluding ~77,811 this epoch (frac=0.75)
2026-08-06 13:34:44,117 INFO   BC epoch 2/3  (26.9s)
    loss       bc=2.1280  bc_adj=0.1370(floor=1.9910)
    heads      dir_cos=0.982  kick_dir_cos=0.998
               move_prob=0.871  sprint_prob=0.783  kick_prob=0.069  tackle_prob=0.038
    pr/rec     kick:   p=0.988  r=0.998  f1=0.993  (tp=21860 fp=263 fn=48)
               tackle: p=0.938  r=0.890  f1=0.914  (tp=1278 fp=84 fn=158)
    breakdown  decision=0.686  exec_bce=0.696  sprint=0.197  move=0.180  tackle_attempt=0.155  direction=0.051
               region=0.008  kick=0.16479  kick_direction=0.00599  kick_power=0.00081  kick_spin=0.00000
2026-08-06 13:34:44,841 INFO     val        bc_val_loss=2.1249  best=2.1093  (patience 1/2)
2026-08-06 13:34:44,845 INFO   Downsample trivial rows (epoch 3): 103,748/257,571 (40.3%) rows classified trivial, excluding ~77,811 this epoch (frac=0.75)
2026-08-06 13:35:11,437 INFO   BC epoch 3/3  (26.6s)
    loss       bc=2.1259  bc_adj=0.1348(floor=1.9910)
    heads      dir_cos=0.982  kick_dir_cos=0.999
               move_prob=0.871  sprint_prob=0.782  kick_prob=0.069  tackle_prob=0.038
    pr/rec     kick:   p=0.988  r=0.998  f1=0.993  (tp=21865 fp=276 fn=43)
               tackle: p=0.934  r=0.896  f1=0.915  (tp=1287 fp=91 fn=149)
    breakdown  decision=0.686  exec_bce=0.695  sprint=0.197  move=0.179  tackle_attempt=0.154  direction=0.051
               region=0.008  kick=0.16465  kick_direction=0.00425  kick_power=0.00070  kick_spin=0.00000
2026-08-06 13:35:12,147 INFO     val        bc_val_loss=2.1052  best=2.1052  (improved)
2026-08-06 13:35:12,150 INFO BC pre-training done (3 epoch(s), final bc_loss=2.1259)
2026-08-06 13:35:12,150 INFO Value pre-training: 45000 steps, 35 epochs, lr=0.0002
2026-08-06 13:38:56,606 INFO   [value pretrain rollout] mean_return=6.00 (945 episode(s))  vs_rules(0): nan%  vs_immobile(945): 86%  vs_neural(0): nan%
2026-08-06 13:38:56,612 INFO   [value pretrain rollout] rew/ep (mean/std/min/max per episode, 945 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.004    0.003    +0.000    +0.038
  retreat           -0.003    0.002    -0.023    -0.000
  approach_speed    -0.082    0.248    -1.179    +1.000
  heading           -0.056    0.086    -0.469    +0.000
  get_possession    +0.927    0.276    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.008    0.123    -1.900    +0.000
  ball_out          -0.024    0.243    -2.500    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +3.871    1.560    +0.000    +4.500
  speed_bonus       +1.398    1.228    +0.000    +3.689
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           +0.000    0.000    +0.000    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.028    0.031    -0.159    +0.000
2026-08-06 13:38:58,640 INFO   Value pretrain split: 803 train eps (38397 steps)  |  142 val eps (6603 steps)
2026-08-06 13:39:03,996 INFO   Value epoch 1/35: train=0.3386 rmse=0.96  val=0.3347 val_rmse=0.96 (std=1.7)
2026-08-06 13:39:09,082 INFO   Value epoch 2/35: train=0.3231 rmse=0.94  val=0.3436 val_rmse=0.97 (std=1.7)
2026-08-06 13:39:14,076 INFO   Value epoch 3/35: train=0.3158 rmse=0.93  val=0.3261 val_rmse=0.94 (std=1.7)
2026-08-06 13:39:19,065 INFO   Value epoch 4/35: train=0.3113 rmse=0.92  val=0.3251 val_rmse=0.94 (std=1.7)
2026-08-06 13:39:24,131 INFO   Value epoch 5/35: train=0.3063 rmse=0.91  val=0.3294 val_rmse=0.95 (std=1.7)
2026-08-06 13:39:29,129 INFO   Value epoch 6/35: train=0.3020 rmse=0.91  val=0.3360 val_rmse=0.96 (std=1.7)
2026-08-06 13:39:34,171 INFO   Value epoch 7/35: train=0.2974 rmse=0.90  val=0.3439 val_rmse=0.97 (std=1.7)
2026-08-06 13:39:39,212 INFO   Value epoch 8/35: train=0.2948 rmse=0.90  val=0.3449 val_rmse=0.97 (std=1.7)
2026-08-06 13:39:44,242 INFO   Value epoch 9/35: train=0.2903 rmse=0.89  val=0.3378 val_rmse=0.96 (std=1.7)
2026-08-06 13:39:49,207 INFO   Value epoch 10/35: train=0.2877 rmse=0.89  val=0.3487 val_rmse=0.98 (std=1.7)
2026-08-06 13:39:54,177 INFO   Value epoch 11/35: train=0.2846 rmse=0.88  val=0.3337 val_rmse=0.95 (std=1.7)
2026-08-06 13:39:59,138 INFO   Value epoch 12/35: train=0.2827 rmse=0.88  val=0.3354 val_rmse=0.96 (std=1.7)
2026-08-06 13:40:04,257 INFO   Value epoch 13/35: train=0.2803 rmse=0.87  val=0.3396 val_rmse=0.96 (std=1.7)
2026-08-06 13:40:09,269 INFO   Value epoch 14/35: train=0.2781 rmse=0.87  val=0.3368 val_rmse=0.96 (std=1.7)
2026-08-06 13:40:14,203 INFO   Value epoch 15/35: train=0.2750 rmse=0.87  val=0.3433 val_rmse=0.97 (std=1.7)
2026-08-06 13:40:19,275 INFO   Value epoch 16/35: train=0.2734 rmse=0.86  val=0.3465 val_rmse=0.97 (std=1.7)
2026-08-06 13:40:19,275 INFO   [value pretrain] early stop at epoch 16 (val stagnant for 12 epochs, best=0.3251)
2026-08-06 13:40:19,276 INFO   [value pretrain] restored best-val weights (val_loss=0.3251)
2026-08-06 13:40:19,276 INFO Value pre-training done (16 epoch(s), final train_loss=0.2734)
2026-08-06 13:40:25,503 INFO BC check after value warm-up: bc_loss=2.1021 (before=2.1259, delta=-0.0237)  OK
2026-08-06 13:40:25,504 INFO Combined pre-training complete.
2026-08-06 13:40:25,524 INFO Pre-trained checkpoint saved: checkpoints/phase1_run44/checkpoint_pretrained.pt
2026-08-06 13:40:28,164 INFO   [neural] trial 10/40: outcome=other, reward=0.11
2026-08-06 13:40:29,709 INFO   [neural] trial 20/40: outcome=opponent_box_possession, reward=-2.45
2026-08-06 13:40:32,504 INFO   [neural] trial 30/40: outcome=opponent_box_possession, reward=-2.78
2026-08-06 13:40:34,868 INFO   [neural] trial 40/40: outcome=box_possession, reward=4.50
2026-08-06 13:40:34,869 INFO Pre-PPO eval (rules opp): win=17.5%  mean_rew=-0.226  mean_val=2.606  outcomes={'opponent_box_possession': 25, 'box_possession': 7, 'other': 7, 'timeout': 1}
2026-08-06 13:40:34,869 INFO   rew breakdown (rules, per ep): lose_possession=-1.28  get_possession=+1.05  opponent_box=-0.94  box_possession=+0.79  speed_bonus=+0.31  stamina_penalty=-0.08  approach_speed=-0.05  heading=-0.03
2026-08-06 13:40:38,076 INFO   [neural] trial 10/40: outcome=timeout, reward=0.03
2026-08-06 13:40:39,969 INFO   [neural] trial 20/40: outcome=box_possession, reward=5.68
2026-08-06 13:40:41,269 INFO   [neural] trial 30/40: outcome=box_possession, reward=5.51
2026-08-06 13:40:44,736 INFO   [neural] trial 40/40: outcome=box_possession, reward=5.30
2026-08-06 13:40:44,737 INFO Pre-PPO eval (immobile opp): win=90.0%  mean_rew=6.291  mean_val=3.133  outcomes={'box_possession': 36, 'timeout': 1, 'miss': 3}
2026-08-06 13:40:44,737 INFO   rew breakdown (immobile, per ep): box_possession=+4.05  speed_bonus=+1.39  get_possession=+0.97  approach_speed=-0.06  heading=-0.05  stamina_penalty=-0.02
2026-08-06 13:40:50,620 INFO   [neural] trial 10/40: outcome=box_possession, reward=3.46
2026-08-06 13:40:54,246 INFO   [neural] trial 20/40: outcome=opponent_box_possession, reward=-1.52
2026-08-06 13:40:57,168 INFO   [neural] trial 30/40: outcome=box_possession, reward=7.28
2026-08-06 13:41:00,178 INFO   [neural] trial 40/40: outcome=box_possession, reward=3.98
2026-08-06 13:41:00,179 INFO Pre-PPO eval (self-play):   win=70.0%  mean_rew=4.282  mean_val=2.971  outcomes={'box_possession': 28, 'opponent_box_possession': 9, 'miss': 3}
2026-08-06 13:41:00,179 INFO   rew breakdown (self-play, per ep): box_possession=+4.16  get_possession=+1.65  speed_bonus=+1.52  opponent_box=-1.39  lose_possession=-1.38  approach_speed=-0.25  stamina_penalty=-0.18  illegal=-0.13  heading=-0.07  retreat=-0.01  approach=+0.01
2026-08-06 13:41:00,320 INFO   [baseline] trial 10/12: outcome=box_possession
2026-08-06 13:41:00,352 INFO Baseline (rules vs rules, 12 trials): trainee_win=66.7%  outcomes={'box_possession': 8, 'opponent_box_possession': 4}
2026-08-06 13:41:00,352 INFO Frozen decision_net.shoot_logit
2026-08-06 13:41:00,352 INFO Frozen decision_net.pass_logit
2026-08-06 13:41:00,352 INFO Frozen decision_net.tackle_logit
2026-08-06 13:41:00,353 INFO Frozen decision_net.get_possession_raw
2026-08-06 13:41:00,353 INFO Frozen decision_net.mark_logit
2026-08-06 13:41:00,353 INFO Frozen decision_net.hold_position_logit
2026-08-06 13:41:00,353 INFO Frozen decision_net.pass_target_logits
2026-08-06 13:41:00,353 INFO Frozen decision_net.tackle_target_logits
2026-08-06 13:41:00,353 INFO Frozen decision_net.mark_target_logits
2026-08-06 13:41:00,353 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-06 13:41:00,354 INFO PPO training started: steps_so_far=0  target=100,000  (+100,000 this run)
2026-08-06 13:43:14,149 INFO   [advantage] mean=0.000  std=1.000  min=-6.507  max=5.226
2026-08-06 13:43:14,150 INFO   [ratio] mean=0.9917  std=0.3017  min=0.0034  max=16.4220  clipped=16.7%
2026-08-06 13:43:14,151 INFO   [exec head grad norm] move_direction=0.010  exec_move=0.070  sprint=0.015  kick=0.035  kick_direction=0.004  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.046
2026-08-06 13:43:14,151 INFO   [exec continuous log_std] move_direction: start=-1.6311 end=-1.6306   kick_direction: start=-1.6392 end=-1.6390
2026-08-06 13:43:14,151 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0002≈0.01°/step  epoch≈1.2°  dlog_std=0.00000  Δσ°=0.000/step)  kick_direction(dmean=0.0002≈0.01°/step  epoch≈1.4°  dlog_std=0.00000  Δσ°=0.000/step)
2026-08-06 13:43:14,151 INFO   [exec discrete Δlogit per opt step] exec_move=0.0018  sprint=0.0011  kick=0.0006  tackle_attempt=0.0009
2026-08-06 13:43:14,151 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0001  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0155  sprint=+0.0121  kick=+0.0058  tackle_attempt=+0.0008  move_dir=+0.0062  kick_dir=+0.0034
2026-08-06 13:43:14,152 INFO   [grad clip] main: 32/96 steps clipped (33%)  pre-clip norm mean=0.393 max=0.917  limit=0.4
              direction: 0/96 steps clipped (0%)  pre-clip norm mean=0.012 max=0.030  limit=0.04
2026-08-06 13:43:14,173 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=24,000  speed=236/s  reward=6.45
  loss     policy=0.0175  value=0.3399(x0.5)=0.1699
           entropy=1.3790  kl=0.0439
  value    V=2.93±1.39  R=2.83±1.65  adv=-0.09±0.99
  moves    mv_ls=[-1.6306] (σ≈0.20, ≈11°) g=1.74e-03
           kk_ls=[-1.6390] (σ≈0.19, ≈11°)
  heads    move= 30 get_poss= 70 exec_move= 89 sprint= 37 kick=  4 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0359 kick_prob=0.0405
  vs       vs_immobile(514): 84%
  reward   approach=+2.13  retreat=-1.54  approach_speed=-37.08  heading=-27.60  get_possession=+466.00
           lose_possession=-3.80  ball_out=-5.00  box_possession=+1939.50  speed_bonus=+683.31  stamina_penalty=-13.41
  rew/ep   (mean/std/min/max per episode, 514 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.004    0.003    +0.000    +0.026
  retreat           -0.003    0.003    -0.022    +0.000
  approach_speed    -0.072    0.243    -1.056    +1.000
  heading           -0.053    0.079    -0.444    +0.000
  get_possession    +0.907    0.304    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.007    0.118    -1.900    +0.000
  ball_out          -0.010    0.156    -2.500    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +3.773    1.656    +0.000    +4.500
  speed_bonus       +1.329    1.204    +0.000    +3.592
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           +0.000    0.000    +0.000    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.026    0.030    -0.143    +0.000
──────────────────────────────────────────────────────────────────────
2026-08-06 13:43:14,774 INFO Saved checkpoint: checkpoints/phase1_run44/checkpoint1.pt
2026-08-06 13:43:14,774 INFO Logging to checkpoints/phase1_run44/training_log2.txt
2026-08-06 13:43:17,267 INFO   [eval vs rules] step=24,000  win=6/20 (30%)  outcomes={'opponent_box_possession': 12, 'other': 1, 'miss': 1, 'box_possession': 6}
2026-08-06 13:45:22,014 INFO   [advantage] mean=-0.000  std=1.000  min=-7.033  max=4.742
2026-08-06 13:45:22,015 INFO   [ratio] mean=0.9895  std=0.3046  min=0.0065  max=36.9800  clipped=15.3%
2026-08-06 13:45:22,015 INFO   [exec head grad norm] move_direction=0.011  exec_move=0.067  sprint=0.016  kick=0.040  kick_direction=0.003  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.049
2026-08-06 13:45:22,015 INFO   [exec continuous log_std] move_direction: start=-1.6306 end=-1.6300   kick_direction: start=-1.6390 end=-1.6387
2026-08-06 13:45:22,015 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0002≈0.01°/step  epoch≈1.1°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0002≈0.01°/step  epoch≈1.0°  dlog_std=0.00000  Δσ°=0.000/step)
2026-08-06 13:45:22,015 INFO   [exec discrete Δlogit per opt step] exec_move=0.0017  sprint=0.0013  kick=0.0006  tackle_attempt=0.0009
2026-08-06 13:45:22,016 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0001  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0153  sprint=+0.0110  kick=+0.0059  tackle_attempt=+0.0007  move_dir=+0.0062  kick_dir=+0.0034
2026-08-06 13:45:22,016 INFO   [grad clip] main: 20/96 steps clipped (21%)  pre-clip norm mean=0.365 max=0.628  limit=0.4
              direction: 0/96 steps clipped (0%)  pre-clip norm mean=0.012 max=0.029  limit=0.04
2026-08-06 13:45:22,044 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=48,000  speed=253/s  reward=7.28
  loss     policy=0.0151  value=0.3059(x0.5)=0.1529
           entropy=1.4787  kl=0.0426
  value    V=2.87±1.40  R=2.89±1.69  adv=0.02±0.96
  moves    mv_ls=[-1.6300] (σ≈0.20, ≈11°) g=2.23e-03  d_move=[+0.0006] (Δσ≈0.007°)
           kk_ls=[-1.6387] (σ≈0.19, ≈11°)  d_kick=[+0.0004] (Δσ≈0.004°)
  heads    move= 32 get_poss= 68 exec_move= 89 sprint= 40 kick=  4 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0387 kick_prob=0.0417
  vs       vs_immobile(523): 86%
  reward   approach=+2.14  retreat=-1.57  approach_speed=-32.24  heading=-24.22  get_possession=+479.00
           ball_out=-12.50  box_possession=+2025.00  speed_bonus=+762.92  opponent_box=-1.50  stamina_penalty=-16.51
  rew/ep   (mean/std/min/max per episode, 523 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.004    0.003    +0.000    +0.021
  retreat           -0.003    0.002    -0.021    -0.000
  approach_speed    -0.062    0.219    -0.717    +1.000
  heading           -0.047    0.071    -0.462    +0.000
  get_possession    +0.914    0.280    +0.000    +1.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    +0.000    0.000    +0.000    +0.000
  ball_out          -0.024    0.243    -2.500    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +3.872    1.559    +0.000    +4.500
  speed_bonus       +1.459    1.217    +0.000    +3.699
  opponent_box      -0.003    0.066    -1.500    +0.000
  timeout           +0.000    0.000    +0.000    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.032    0.033    -0.181    +0.000
──────────────────────────────────────────────────────────────────────
2026-08-06 13:45:22,689 INFO Saved checkpoint: checkpoints/phase1_run44/checkpoint2.pt
2026-08-06 13:45:22,689 INFO Logging to checkpoints/phase1_run44/training_log3.txt
2026-08-06 13:45:25,155 INFO   [eval vs rules] step=48,000  win=6/20 (30%)  outcomes={'opponent_box_possession': 14, 'box_possession': 6}
2026-08-06 13:47:42,713 INFO   [advantage] mean=-0.000  std=1.000  min=-6.876  max=5.305
2026-08-06 13:47:42,714 INFO   [ratio] mean=0.9931  std=0.3138  min=0.0068  max=22.8035  clipped=16.0%
2026-08-06 13:47:42,714 INFO   [exec head grad norm] move_direction=0.009  exec_move=0.063  sprint=0.015  kick=0.034  kick_direction=0.003  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.050
2026-08-06 13:47:42,714 INFO   [exec continuous log_std] move_direction: start=-1.6300 end=-1.6293   kick_direction: start=-1.6387 end=-1.6382
2026-08-06 13:47:42,714 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0002≈0.01°/step  epoch≈0.8°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0002≈0.01°/step  epoch≈0.9°  dlog_std=0.00000  Δσ°=0.000/step)
2026-08-06 13:47:42,714 INFO   [exec discrete Δlogit per opt step] exec_move=0.0015  sprint=0.0008  kick=0.0005  tackle_attempt=0.0009
2026-08-06 13:47:42,715 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0001  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0115  sprint=+0.0095  kick=+0.0048  tackle_attempt=+0.0012  move_dir=+0.0057  kick_dir=+0.0035
2026-08-06 13:47:42,715 INFO   [grad clip] main: 12/96 steps clipped (12%)  pre-clip norm mean=0.355 max=0.504  limit=0.4
              direction: 0/96 steps clipped (0%)  pre-clip norm mean=0.010 max=0.029  limit=0.04
2026-08-06 13:47:42,735 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=72,000  speed=230/s  reward=4.88
  loss     policy=0.0157  value=0.2945(x0.5)=0.1473
           entropy=1.5962  kl=0.0363
  value    V=3.06±1.36  R=3.08±1.64  adv=0.03±0.92
  moves    mv_ls=[-1.6293] (σ≈0.20, ≈11°) g=2.60e-03  d_move=[+0.0007] (Δσ≈0.008°)
           kk_ls=[-1.6382] (σ≈0.19, ≈11°)  d_kick=[+0.0004] (Δσ≈0.005°)
  heads    move= 35 get_poss= 66 exec_move= 87 sprint= 42 kick=  4 tackle=  4 shoot=
           2 hold=  3 tackle_prob=0.0418 kick_prob=0.0431
  vs       vs_immobile(531): 90%
  reward   approach=+2.12  retreat=-1.49  approach_speed=-35.83  heading=-20.72  get_possession=+504.00
           lose_possession=-9.50  ball_out=-10.00  box_possession=+2151.00  speed_bonus=+838.21  stamina_penalty=-17.53
  rew/ep   (mean/std/min/max per episode, 531 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.004    0.003    +0.001    +0.029
  retreat           -0.003    0.002    -0.020    -0.000
  approach_speed    -0.067    0.224    -0.997    +1.000
  heading           -0.039    0.052    -0.384    +0.000
  get_possession    +0.951    0.248    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.018    0.184    -1.900    +0.000
  ball_out          -0.019    0.216    -2.500    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +4.051    1.349    +0.000    +4.500
  speed_bonus       +1.579    1.200    +0.000    +3.744
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           +0.000    0.000    +0.000    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.033    0.033    -0.145    +0.000
──────────────────────────────────────────────────────────────────────
2026-08-06 13:47:43,353 INFO Saved checkpoint: checkpoints/phase1_run44/checkpoint3.pt
2026-08-06 13:47:43,353 INFO Logging to checkpoints/phase1_run44/training_log4.txt
2026-08-06 13:47:46,164 INFO   [eval vs rules] step=72,000  win=5/20 (25%)  outcomes={'other': 1, 'opponent_box_possession': 13, 'box_possession': 5, 'miss': 1}
2026-08-06 13:50:05,199 INFO   [advantage] mean=0.000  std=1.000  min=-7.356  max=5.011
2026-08-06 13:50:05,200 INFO   [ratio] mean=0.9920  std=0.2564  min=0.0009  max=17.7322  clipped=15.6%
2026-08-06 13:50:05,200 INFO   [exec head grad norm] move_direction=0.008  exec_move=0.050  sprint=0.018  kick=0.038  kick_direction=0.003  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.053
2026-08-06 13:50:05,200 INFO   [exec continuous log_std] move_direction: start=-1.6293 end=-1.6286   kick_direction: start=-1.6382 end=-1.6377
2026-08-06 13:50:05,201 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0001≈0.01°/step  epoch≈0.6°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0001≈0.01°/step  epoch≈0.7°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-06 13:50:05,201 INFO   [exec discrete Δlogit per opt step] exec_move=0.0011  sprint=0.0011  kick=0.0005  tackle_attempt=0.0008
2026-08-06 13:50:05,201 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0002  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0117  sprint=+0.0078  kick=+0.0046  tackle_attempt=+0.0011  move_dir=+0.0054  kick_dir=+0.0035
2026-08-06 13:50:05,201 INFO   [grad clip] main: 21/96 steps clipped (22%)  pre-clip norm mean=0.372 max=0.582  limit=0.4
              direction: 0/96 steps clipped (0%)  pre-clip norm mean=0.010 max=0.025  limit=0.04
2026-08-06 13:50:05,225 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=96,000  speed=224/s  reward=6.41
  loss     policy=0.0152  value=0.2874(x0.5)=0.1437
           entropy=1.7159  kl=0.0344
  value    V=3.07±1.43  R=3.12±1.69  adv=0.04±0.94
  moves    mv_ls=[-1.6286] (σ≈0.20, ≈11°) g=2.13e-03  d_move=[+0.0006] (Δσ≈0.007°)
           kk_ls=[-1.6377] (σ≈0.19, ≈11°)  d_kick=[+0.0005] (Δσ≈0.006°)
  heads    move= 37 get_poss= 63 exec_move= 87 sprint= 44 kick=  4 tackle=  4 shoot=
           3 hold=  3 tackle_prob=0.0450 kick_prob=0.0442
  vs       vs_immobile(546): 87%
  reward   approach=+2.16  retreat=-1.54  approach_speed=-24.33  heading=-18.21  get_possession=+505.00
           lose_possession=-5.70  ball_out=-20.00  box_possession=+2142.00  speed_bonus=+884.36  stamina_penalty=-18.45
  rew/ep   (mean/std/min/max per episode, 546 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.004    0.003    +0.000    +0.035
  retreat           -0.003    0.002    -0.023    -0.000
  approach_speed    -0.045    0.237    -0.884    +1.000
  heading           -0.033    0.046    -0.338    +0.000
  get_possession    +0.925    0.284    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.010    0.140    -1.900    +0.000
  ball_out          -0.037    0.300    -2.500    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +3.923    1.504    +0.000    +4.500
  speed_bonus       +1.620    1.250    +0.000    +3.755
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           +0.000    0.000    +0.000    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.034    0.032    -0.160    +0.000
──────────────────────────────────────────────────────────────────────
2026-08-06 13:50:05,798 INFO Saved checkpoint: checkpoints/phase1_run44/checkpoint4.pt
2026-08-06 13:50:05,798 INFO Logging to checkpoints/phase1_run44/training_log5.txt
2026-08-06 13:50:08,829 INFO   [eval vs rules] step=96,000  win=7/20 (35%)  outcomes={'other': 3, 'opponent_box_possession': 10, 'box_possession': 7}
2026-08-06 13:50:28,369 INFO Saved checkpoint: checkpoints/phase1_run44/checkpoint5.pt
2026-08-06 13:50:28,370 INFO Logging to checkpoints/phase1_run44/training_log6.txt
2026-08-06 13:50:28,370 INFO Final checkpoint saved.
2026-08-06 13:50:28,370 INFO Training complete. Total steps: 100,000
2026-08-06 13:54:47,218 INFO Checkpoint dir: checkpoints/phase1_run45
2026-08-06 13:54:47,266 INFO Starting training: phase=phase1_get_possession, total_steps=200,000
2026-08-06 13:54:47,266 INFO Phase description: 1v1 scenario: learn to get possession and bring the ball toward the opponent box.  Decision network frozen except Move/GetPossession and latent vector.
2026-08-06 13:54:47,303 INFO _from_ckpt: overriding bc_pretrain_epochs=6 → 1
2026-08-06 13:54:47,303 INFO _from_ckpt: overriding demo_value_pretrain_epochs=6 → 0
2026-08-06 13:54:47,303 INFO _from_ckpt: overriding value_pretrain_epochs=60 → 35
2026-08-06 13:54:47,303 INFO _from_ckpt: overriding aux_coeff_start=0.0 → 0.0
2026-08-06 13:54:47,303 INFO _from_ckpt: overriding aux_coeff_end=0.0 → 0.0
2026-08-06 13:54:47,303 INFO _from_ckpt: overriding aux_coeff_anneal_fraction=0.3 → 0.35
2026-08-06 13:54:48,420 INFO Logging to checkpoints/phase1_run45/training_log1.txt
2026-08-06 13:54:48,421 INFO --latest(-pretrain): resolved to checkpoints/phase1_run44/latest.pt
2026-08-06 13:54:48,596 INFO Loaded checkpoint: checkpoints/phase1_run44/latest.pt (step 100000)
2026-08-06 13:54:48,597 INFO Loaded checkpoint for re-pretraining: checkpoints/phase1_run44/latest.pt — will still run BC/value pre-training
2026-08-06 13:54:48,603 INFO Loading 750 demonstration file(s) from demonstrations/phase1
2026-08-06 13:54:51,497 INFO Dataset: 290,906 steps loaded
2026-08-06 13:54:51,498 INFO Offline BC dataset: 290,906 steps from demonstrations/phase1/
2026-08-06 13:54:51,498 INFO Reward diagnostic: running 40 rules-vs-rules episodes...
2026-08-06 13:54:52,179 INFO Reward diagnostic (40 ep, rules vs rules): mean_ep_rew=3.31  per_episode: approach=+0.02  retreat=-0.01  approach_speed=+0.07  heading=-0.01  get_possession=+1.07  progress=-0.10  lose_possession=-0.90  ball_out=-0.06  box_possession=+2.70  speed_bonus=+1.16  opponent_box=-0.56  stamina_penalty=-0.07
2026-08-06 13:54:52,188 INFO BC pos_weight (auto-computed from dataset): kick=2.00  tackle_attempt=2.00
2026-08-06 13:54:52,188 INFO Combined BC + value pre-training: 1 epoch(s), batch_size=752, dataset=290,906 steps, rollout_steps=45000
2026-08-06 13:54:52,193 INFO   BC pretrain split: 218,155 train rows  |  39,416 val rows
2026-08-06 13:54:52,242 INFO   Downsample trivial rows (epoch 1): 95,073/257,571 (36.9%) rows classified trivial, excluding ~66,551 this epoch (frac=0.70)
/home/vincent/Documents/not_work/repos/FootballCoach/src/footballcoach/ai/ppo/bc.py:691: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /__w/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:822.)
  "decision":       float(dec_loss[valid].mean()),
2026-08-06 13:55:23,679 INFO   BC epoch 1/1  (31.5s)
    loss       bc=2.1044  bc_adj=0.1195(floor=1.9849)
    heads      dir_cos=0.983  kick_dir_cos=0.999
               move_prob=0.876  sprint_prob=0.787  kick_prob=0.067  tackle_prob=0.037
    pr/rec     kick:   p=0.993  r=0.998  f1=0.996  (tp=21865 fp=154 fn=43)
               tackle: p=0.935  r=0.907  f1=0.921  (tp=1303 fp=90 fn=133)
    breakdown  decision=0.687  exec_bce=0.677  sprint=0.190  move=0.175  tackle_attempt=0.153  direction=0.047
               region=0.007  kick=0.15808  kick_direction=0.00340  kick_power=0.00064  kick_spin=0.00000
2026-08-06 13:55:24,392 INFO     val        bc_val_loss=2.0934  best=2.0934  (improved)
2026-08-06 13:55:24,395 INFO BC pre-training done (1 epoch(s), final bc_loss=2.1044)
2026-08-06 13:55:24,395 INFO Value pre-training: 45000 steps, 35 epochs, lr=0.0001
