2026-08-06 01:04:51,584 INFO Checkpoint dir: checkpoints/phase1_run8
2026-08-06 01:04:51,712 INFO Starting training: phase=phase1_get_possession, total_steps=200,000
2026-08-06 01:04:51,712 INFO Phase description: 1v1 scenario: learn to get possession and bring the ball toward the opponent box.  Decision network frozen except Move/GetPossession and latent vector.
2026-08-06 01:04:51,796 INFO _from_ckpt: overriding bc_pretrain_epochs=10 → 1
2026-08-06 01:04:51,796 INFO _from_ckpt: overriding demo_value_pretrain_epochs=10 → 0
2026-08-06 01:04:51,797 INFO _from_ckpt: overriding value_pretrain_epochs=45 → 35
2026-08-06 01:04:51,797 INFO _from_ckpt: overriding aux_coeff_start=0.0 → 0.0
2026-08-06 01:04:51,797 INFO _from_ckpt: overriding aux_coeff_end=0.0 → 0.0
2026-08-06 01:04:51,797 INFO _from_ckpt: overriding aux_coeff_anneal_fraction=0.3 → 0.35
2026-08-06 01:04:54,190 INFO Logging to checkpoints/phase1_run8/training_log1.txt
2026-08-06 01:04:54,191 INFO --latest(-pretrain): resolved to checkpoints/phase1_run6/latest.pt
2026-08-06 01:04:54,390 INFO Loaded checkpoint: checkpoints/phase1_run6/latest.pt (step 60001)
2026-08-06 01:04:54,391 INFO Loaded checkpoint for re-pretraining: checkpoints/phase1_run6/latest.pt — will still run BC/value pre-training
2026-08-06 01:04:54,400 INFO Loading 1000 demonstration file(s) from demonstrations/phase1
2026-08-06 01:05:08,076 INFO Dataset: 385,963 steps loaded
2026-08-06 01:05:08,078 INFO Offline BC dataset: 385,963 steps from demonstrations/phase1
2026-08-06 01:05:08,078 INFO Reward diagnostic: running 40 rules-vs-rules episodes...
2026-08-06 01:05:11,562 INFO Reward diagnostic (40 ep, rules vs rules): mean_ep_rew=9.85  per_episode: get_possession=+2.02  progress=+0.01  lose_possession=-0.68  box_possession=+3.38  speed_bonus=+5.46  opponent_box=-0.35
2026-08-06 01:05:11,612 INFO BC pos_weight (auto-computed from dataset): kick=2.00  tackle_attempt=2.00
2026-08-06 01:05:11,613 INFO Combined BC + value pre-training: 1 epoch(s), batch_size=652, dataset=385,963 steps, rollout_steps=25000
2026-08-06 01:05:12,024 INFO   Downsample trivial rows (epoch 1): 152,060/341,916 (44.5%) rows classified trivial, excluding ~121,648 this epoch (frac=0.80)
/home/vincent/Documents/not_work/repos/FootballCoach/src/footballcoach/ai/ppo/bc.py:685: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /__w/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:822.)
  "decision":       float(dec_loss[valid].mean()),
2026-08-06 01:08:08,904 INFO   BC epoch 1/1  (177.3s)
    loss       bc=3.0828  bc_adj=0.3936(floor=2.6892)
    heads      dir_cos=0.981  kick_dir_cos=0.998
               move_prob=0.855  sprint_prob=0.776  kick_prob=0.129  tackle_prob=0.102
    pr/rec     kick:   p=0.987  r=0.997  f1=0.992  (tp=33509 fp=449 fn=95)
               tackle: p=0.720  r=0.645  f1=0.680  (tp=3389 fp=1316 fn=1867)
    breakdown  decision=0.944  exec_bce=1.133  sprint=0.241  move=0.224  tackle_attempt=0.339  direction=0.053
               region=0.008  kick=0.32829  kick_direction=0.00507  kick_power=0.00100  kick_spin=0.00000
2026-08-06 01:08:08,905 INFO BC pre-training done (1 epoch(s), final bc_loss=3.0828)
2026-08-06 01:08:08,905 INFO Value pre-training: 25000 steps, 35 epochs, lr=0.0002
2026-08-06 01:12:34,466 INFO   [value pretrain rollout] mean_return=7.78 (461 episode(s))  vs_rules(0): nan%  vs_immobile(204): 79%  vs_neural(257): 59%
2026-08-06 01:12:35,610 INFO   Value pretrain split: 392 train eps (21505 steps)  |  69 val eps (3495 steps)
2026-08-06 01:12:38,615 INFO   Value epoch 1/35: train=0.4081 rmse=1.81  val=0.4640 val_rmse=1.93 (std=2.8)
2026-08-06 01:12:41,484 INFO   Value epoch 2/35: train=0.3824 rmse=1.76  val=0.4629 val_rmse=1.93 (std=2.8)
2026-08-06 01:12:44,359 INFO   Value epoch 3/35: train=0.3705 rmse=1.73  val=0.4820 val_rmse=1.97 (std=2.8)
2026-08-06 01:12:47,260 INFO   Value epoch 4/35: train=0.3629 rmse=1.71  val=0.5147 val_rmse=2.04 (std=2.8)
2026-08-06 01:12:50,092 INFO   Value epoch 5/35: train=0.3525 rmse=1.69  val=0.4840 val_rmse=1.97 (std=2.8)
2026-08-06 01:12:53,045 INFO   Value epoch 6/35: train=0.3482 rmse=1.67  val=0.4992 val_rmse=2.01 (std=2.8)
2026-08-06 01:12:56,125 INFO   Value epoch 7/35: train=0.3417 rmse=1.66  val=0.4722 val_rmse=1.95 (std=2.8)
2026-08-06 01:12:59,281 INFO   Value epoch 8/35: train=0.3376 rmse=1.65  val=0.4962 val_rmse=2.00 (std=2.8)
2026-08-06 01:13:02,405 INFO   Value epoch 9/35: train=0.3319 rmse=1.64  val=0.4971 val_rmse=2.00 (std=2.8)
2026-08-06 01:13:05,239 INFO   Value epoch 10/35: train=0.3259 rmse=1.62  val=0.5229 val_rmse=2.05 (std=2.8)
2026-08-06 01:13:08,198 INFO   Value epoch 11/35: train=0.3252 rmse=1.62  val=0.5251 val_rmse=2.06 (std=2.8)
2026-08-06 01:13:11,039 INFO   Value epoch 12/35: train=0.3201 rmse=1.61  val=0.5535 val_rmse=2.11 (std=2.8)
2026-08-06 01:13:13,956 INFO   Value epoch 13/35: train=0.3152 rmse=1.59  val=0.5286 val_rmse=2.06 (std=2.8)
2026-08-06 01:13:16,804 INFO   Value epoch 14/35: train=0.3128 rmse=1.59  val=0.4900 val_rmse=1.99 (std=2.8)
2026-08-06 01:13:19,701 INFO   Value epoch 15/35: train=0.3089 rmse=1.58  val=0.5529 val_rmse=2.11 (std=2.8)
2026-08-06 01:13:22,604 INFO   Value epoch 16/35: train=0.3047 rmse=1.57  val=0.5131 val_rmse=2.03 (std=2.8)
2026-08-06 01:13:25,448 INFO   Value epoch 17/35: train=0.3027 rmse=1.56  val=0.5711 val_rmse=2.14 (std=2.8)
2026-08-06 01:13:25,448 INFO   [value pretrain] early stop at epoch 17 (val stagnant for 15 epochs, best=0.4629)
2026-08-06 01:13:25,449 INFO   [value pretrain] restored best-val weights (val_loss=0.4629)
2026-08-06 01:13:25,449 INFO Value pre-training done (17 epoch(s), final train_loss=0.3027)
2026-08-06 01:13:32,707 INFO BC check after value warm-up: bc_loss=3.0607 (before=3.0828, delta=-0.0221)  OK
2026-08-06 01:13:32,707 INFO Combined pre-training complete.
2026-08-06 01:13:32,726 INFO Pre-trained checkpoint saved: checkpoints/phase1_run8/checkpoint_pretrained.pt
2026-08-06 01:13:34,757 INFO   [neural] trial 10/40: outcome=other, reward=1.80
2026-08-06 01:13:36,490 INFO   [neural] trial 20/40: outcome=miss, reward=-0.00
2026-08-06 01:13:39,863 INFO   [neural] trial 30/40: outcome=opponent_box_possession, reward=-2.00
2026-08-06 01:13:42,995 INFO   [neural] trial 40/40: outcome=opponent_box_possession, reward=-1.99
2026-08-06 01:13:42,995 INFO Pre-PPO eval (rules opp): win=12.5%  mean_rew=0.382  mean_val=2.296  outcomes={'opponent_box_possession': 24, 'other': 6, 'miss': 5, 'box_possession': 5}
2026-08-06 01:13:42,995 INFO   rew breakdown (rules, per ep): get_possession=+1.39  opponent_box=-1.20  lose_possession=-0.81  box_possession=+0.56  speed_bonus=+0.44
2026-08-06 01:13:46,116 INFO   [neural] trial 10/40: outcome=box_possession, reward=11.42
2026-08-06 01:13:49,901 INFO   [neural] trial 20/40: outcome=box_possession, reward=11.05
2026-08-06 01:13:53,645 INFO   [neural] trial 30/40: outcome=miss, reward=1.78
2026-08-06 01:13:59,385 INFO   [neural] trial 40/40: outcome=miss, reward=1.80
2026-08-06 01:13:59,386 INFO Pre-PPO eval (immobile opp): win=70.0%  mean_rew=9.160  mean_val=3.968  outcomes={'box_possession': 28, 'miss': 12}
2026-08-06 01:13:59,386 INFO   rew breakdown (immobile, per ep): speed_bonus=+4.27  box_possession=+3.15  get_possession=+1.75  ball_out=-0.04  progress=+0.02
2026-08-06 01:14:04,952 INFO   [neural] trial 10/40: outcome=box_possession, reward=9.60
2026-08-06 01:14:12,693 INFO   [neural] trial 20/40: outcome=box_possession, reward=10.46
2026-08-06 01:14:17,033 INFO   [neural] trial 30/40: outcome=box_possession, reward=12.19
2026-08-06 01:14:24,803 INFO   [neural] trial 40/40: outcome=box_possession, reward=14.82
2026-08-06 01:14:24,803 INFO Pre-PPO eval (self-play):   win=62.5%  mean_rew=6.941  mean_val=2.329  outcomes={'box_possession': 25, 'opponent_box_possession': 6, 'miss': 9}
2026-08-06 01:14:24,803 INFO   rew breakdown (self-play, per ep): box_possession=+3.49  speed_bonus=+3.40  get_possession=+3.33  illegal=-1.67  lose_possession=-1.57  opponent_box=-1.55  progress=+0.09  ball_out=-0.04  retreat=-0.01
2026-08-06 01:14:25,000 INFO   [baseline] trial 10/12: outcome=miss
2026-08-06 01:14:25,045 INFO Baseline (rules vs rules, 12 trials): trainee_win=66.7%  outcomes={'box_possession': 8, 'opponent_box_possession': 3, 'miss': 1}
2026-08-06 01:14:25,045 INFO Frozen decision_net.shoot_logit
2026-08-06 01:14:25,045 INFO Frozen decision_net.pass_logit
2026-08-06 01:14:25,045 INFO Frozen decision_net.tackle_logit
2026-08-06 01:14:25,045 INFO Frozen decision_net.get_possession_raw
2026-08-06 01:14:25,045 INFO Frozen decision_net.mark_logit
2026-08-06 01:14:25,046 INFO Frozen decision_net.hold_position_logit
2026-08-06 01:14:25,046 INFO Frozen decision_net.pass_target_logits
2026-08-06 01:14:25,046 INFO Frozen decision_net.tackle_target_logits
2026-08-06 01:14:25,046 INFO Frozen decision_net.mark_target_logits
2026-08-06 01:14:25,046 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-06 01:14:25,047 INFO PPO training started: steps_so_far=0  target=200,000  (+200,000 this run)
2026-08-06 01:16:56,933 INFO   [advantage] mean=-0.000  std=1.000  min=-5.799  max=8.358
2026-08-06 01:16:56,934 INFO   [ratio] mean=0.9809  std=0.3933  min=0.0000  max=20.7457  clipped=19.9%
2026-08-06 01:16:56,935 INFO   [exec head grad norm] move_direction=0.009  exec_move=0.089  sprint=0.050  kick=0.145  kick_direction=0.004  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.126
2026-08-06 01:16:56,936 INFO   [exec continuous log_std] move_direction: start=-1.4154 end=-1.4130   kick_direction: start=-1.4136 end=-1.4121
2026-08-06 01:16:56,936 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0002≈0.01°/step  epoch≈0.7°  dlog_std=0.00004  Δσ°=0.001/step)  kick_direction(dmean=0.0002≈0.01°/step  epoch≈0.8°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-06 01:16:56,936 INFO   [exec discrete Δlogit per opt step] exec_move=0.0032  sprint=0.0032  kick=0.0017  tackle_attempt=0.0023
2026-08-06 01:16:56,937 INFO   [per-head KL] shoot=-0.0984  pass_=-0.0929  move=+0.0411  tackle=-0.0914  gp_extra=-0.0904  mark=-0.0922  hold=-0.0893  exec_move=+0.0901  sprint=+0.0880  kick=+0.1055  tackle_attempt=+0.1070  move_dir=+0.0165  kick_dir=+0.0148
2026-08-06 01:16:56,937 INFO   [grad clip] main: 60/60 steps clipped (100%)  pre-clip norm mean=0.459 max=0.572  limit=0.35
              direction: 0/60 steps clipped (0%)  pre-clip norm mean=0.029 max=0.034  limit=0.05
2026-08-06 01:16:56,957 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=18,000  speed=163/s  reward=7.98/-0.42
  loss     policy=0.0419  value=0.4325(x0.5)=0.2163
           entropy=2.1388  kl=0.0766
  value    V=2.64±2.15  R=2.39±2.47  adv=-0.26±1.88
  moves    mv_ls=[-1.4130] (σ≈0.24, ≈14°) g=2.68e-02
           kk_ls=[-1.4121] (σ≈0.24, ≈14°)
  heads    move= 18 get_poss= 82 exec_move= 86 sprint= 56 kick= 10 tackle=  9 shoot=
           3 hold=  3 tackle_prob=0.0960 kick_prob=0.1035
  vs       vs_immobile(108): 73%  vs_neural(103): 62%/14%
  reward   approach=+0.38  retreat=-1.52  get_possession=+558.00  progress=+8.71  lose_possession=-203.40
           ball_out=-4.50  illegal=-212.80  box_possession=+706.50  speed_bonus=+750.73  opponent_box=-156.00
  rew/ep   (mean/std/min/max per episode, 211 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.002    0.002    +0.000    +0.022
  retreat           -0.007    0.006    -0.037    -0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +2.636    2.461    +0.000   +16.200
  progress          +0.041    0.166    -0.229    +1.751
  lose_possession    -0.964    2.366   -14.400    +0.000
  ball_out          -0.021    0.230    -3.000    +0.000
  illegal           -1.009    1.480    -7.800    +0.000
  box_possession    +3.348    1.964    +0.000    +4.500
  speed_bonus       +3.558    3.803    +0.000   +10.636
  opponent_box      -0.739    0.965    -2.000    +0.000
  timeout           +0.000    0.000    +0.000    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    +0.000    0.000    +0.000    +0.000
──────────────────────────────────────────────────────────────────────
2026-08-06 01:16:57,452 INFO Saved checkpoint: checkpoints/phase1_run8/checkpoint1.pt
2026-08-06 01:16:57,453 INFO Logging to checkpoints/phase1_run8/training_log2.txt
2026-08-06 01:17:03,220 INFO   [eval vs rules] step=18,000  win=2/20 (10%)  outcomes={'opponent_box_possession': 15, 'other': 2, 'box_possession': 2, 'miss': 1}
2026-08-06 01:19:24,413 INFO   [advantage] mean=-0.000  std=1.000  min=-7.561  max=7.706
2026-08-06 01:19:24,413 INFO   [ratio] mean=0.9813  std=0.3912  min=0.0009  max=61.2529  clipped=20.2%
2026-08-06 01:19:24,414 INFO   [exec head grad norm] move_direction=0.009  exec_move=0.079  sprint=0.045  kick=0.117  kick_direction=0.005  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.122
2026-08-06 01:19:24,414 INFO   [exec continuous log_std] move_direction: start=-1.4130 end=-1.4108   kick_direction: start=-1.4121 end=-1.4105
2026-08-06 01:19:24,414 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0002≈0.01°/step  epoch≈0.8°  dlog_std=0.00004  Δσ°=0.001/step)  kick_direction(dmean=0.0003≈0.01°/step  epoch≈0.9°  dlog_std=0.00003  Δσ°=0.000/step)
2026-08-06 01:19:24,414 INFO   [exec discrete Δlogit per opt step] exec_move=0.0026  sprint=0.0029  kick=0.0014  tackle_attempt=0.0022
2026-08-06 01:19:24,414 INFO   [per-head KL] shoot=-0.1046  pass_=-0.0973  move=+0.0531  tackle=-0.1059  gp_extra=-0.1100  mark=-0.1118  hold=-0.1043  exec_move=+0.0970  sprint=+0.0876  kick=+0.1114  tackle_attempt=+0.1100  move_dir=+0.0143  kick_dir=+0.0150
2026-08-06 01:19:24,415 INFO   [grad clip] main: 60/60 steps clipped (100%)  pre-clip norm mean=0.456 max=0.529  limit=0.35
              direction: 0/60 steps clipped (0%)  pre-clip norm mean=0.026 max=0.032  limit=0.05
2026-08-06 01:19:24,437 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=36,000  speed=165/s  reward=6.78/-0.54
  loss     policy=0.0245  value=0.3050(x0.5)=0.1525
           entropy=2.3855  kl=0.0702
  value    V=2.43±2.07  R=2.23±2.62  adv=-0.20±1.52
  moves    mv_ls=[-1.4108] (σ≈0.24, ≈14°) g=2.33e-02  d_move=[+0.0022] (Δσ≈0.030°)
           kk_ls=[-1.4105] (σ≈0.24, ≈14°)  d_kick=[+0.0016] (Δσ≈0.023°)
  heads    move= 20 get_poss= 80 exec_move= 86 sprint= 56 kick= 11 tackle= 10 shoot=  4 hold=
           4 tackle_prob=0.1083 kick_prob=0.1125
  vs       vs_immobile(116): 72%  vs_neural(107): 50%/23%
  reward   approach=+0.36  retreat=-1.57  get_possession=+570.60  progress=+10.35  lose_possession=-196.20
           ball_out=-10.50  illegal=-224.40  box_possession=+729.00  speed_bonus=+789.04  opponent_box=-156.00
  rew/ep   (mean/std/min/max per episode, 223 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.002    0.001    +0.000    +0.010
  retreat           -0.007    0.005    -0.030    -0.001
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +2.518    2.134    +0.000   +16.200
  progress          +0.046    0.140    -0.075    +0.950
  lose_possession    -0.839    2.037   -14.400    +0.000
  ball_out          -0.047    0.298    -3.000    +0.000
  illegal           -0.984    1.518    -6.600    +0.000
  box_possession    +3.269    2.006    +0.000    +4.500
  speed_bonus       +3.538    3.724    +0.000   +10.448
  opponent_box      -0.700    0.954    -2.000    +0.000
  timeout           +0.000    0.000    +0.000    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    +0.000    0.000    +0.000    +0.000
──────────────────────────────────────────────────────────────────────
2026-08-06 01:19:24,909 INFO Saved checkpoint: checkpoints/phase1_run8/checkpoint2.pt
2026-08-06 01:19:24,909 INFO Logging to checkpoints/phase1_run8/training_log3.txt
2026-08-06 01:19:29,404 INFO   [eval vs rules] step=36,000  win=3/20 (15%)  outcomes={'box_possession': 3, 'opponent_box_possession': 14, 'other': 2, 'miss': 1}
2026-08-06 01:21:32,985 INFO   [advantage] mean=-0.000  std=1.000  min=-6.887  max=8.897
2026-08-06 01:21:32,986 INFO   [ratio] mean=0.9823  std=0.3416  min=0.0000  max=33.0546  clipped=19.5%
2026-08-06 01:21:32,987 INFO   [exec head grad norm] move_direction=0.008  exec_move=0.068  sprint=0.053  kick=0.103  kick_direction=0.005  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.115
2026-08-06 01:21:32,987 INFO   [exec continuous log_std] move_direction: start=-1.4108 end=-1.4085   kick_direction: start=-1.4105 end=-1.4086
2026-08-06 01:21:32,987 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0002≈0.01°/step  epoch≈0.7°  dlog_std=0.00004  Δσ°=0.001/step)  kick_direction(dmean=0.0002≈0.01°/step  epoch≈0.8°  dlog_std=0.00003  Δσ°=0.000/step)
2026-08-06 01:21:32,987 INFO   [exec discrete Δlogit per opt step] exec_move=0.0026  sprint=0.0030  kick=0.0013  tackle_attempt=0.0021
2026-08-06 01:21:32,987 INFO   [per-head KL] shoot=-0.1172  pass_=-0.1110  move=+0.0747  tackle=-0.1085  gp_extra=-0.1231  mark=-0.1086  hold=-0.1047  exec_move=+0.1136  sprint=+0.1009  kick=+0.1375  tackle_attempt=+0.1342  move_dir=+0.0118  kick_dir=+0.0166
2026-08-06 01:21:32,988 INFO   [grad clip] main: 60/60 steps clipped (100%)  pre-clip norm mean=0.461 max=0.596  limit=0.35
              direction: 0/60 steps clipped (0%)  pre-clip norm mean=0.028 max=0.033  limit=0.05
2026-08-06 01:21:33,009 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=54,000  speed=201/s  reward=8.19/-1.85
  loss     policy=0.0179  value=0.3482(x0.5)=0.1741
           entropy=2.6390  kl=0.0602
  value    V=1.98±2.09  R=1.86±2.56  adv=-0.12±1.58
  moves    mv_ls=[-1.4085] (σ≈0.24, ≈14°) g=2.49e-02  d_move=[+0.0023] (Δσ≈0.032°)
           kk_ls=[-1.4086] (σ≈0.24, ≈14°)  d_kick=[+0.0018] (Δσ≈0.026°)
  heads    move= 21 get_poss= 80 exec_move= 84 sprint= 62 kick= 11 tackle= 12 shoot=  5 hold=
           4 tackle_prob=0.1219 kick_prob=0.1196
  vs       vs_immobile(88): 74%  vs_neural(118): 56%/19%
  reward   approach=+0.36  retreat=-1.63  get_possession=+606.60  progress=+10.67  lose_possession=-255.60
           ball_out=-12.00  illegal=-284.20  box_possession=+693.00  speed_bonus=+701.10  opponent_box=-180.00
  rew/ep   (mean/std/min/max per episode, 206 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.002    0.001    +0.000    +0.007
  retreat           -0.008    0.007    -0.035    -0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +2.988    3.266    +0.000   +25.200
  progress          +0.052    0.180    -0.075    +1.580
  lose_possession    -1.284    3.203   -23.400    +0.000
  ball_out          -0.058    0.357    -3.000    +0.000
  illegal           -1.404    2.020   -11.600    +0.000
  box_possession    +3.364    1.955    +0.000    +4.500
  speed_bonus       +3.403    3.734    +0.000   +10.400
  opponent_box      -0.874    0.992    -2.000    +0.000
  timeout           +0.000    0.000    +0.000    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    +0.000    0.000    +0.000    +0.000
──────────────────────────────────────────────────────────────────────
2026-08-06 01:21:33,472 INFO Saved checkpoint: checkpoints/phase1_run8/checkpoint3.pt
2026-08-06 01:21:33,473 INFO Logging to checkpoints/phase1_run8/training_log4.txt
2026-08-06 01:21:37,435 INFO   [eval vs rules] step=54,000  win=2/20 (10%)  outcomes={'opponent_box_possession': 17, 'box_possession': 2, 'miss': 1}
2026-08-06 01:23:55,340 INFO   [advantage] mean=0.000  std=1.000  min=-7.140  max=8.608
2026-08-06 01:23:55,340 INFO   [ratio] mean=0.9801  std=0.2986  min=0.0029  max=22.0138  clipped=19.6%
2026-08-06 01:23:55,341 INFO   [exec head grad norm] move_direction=0.009  exec_move=0.071  sprint=0.058  kick=0.123  kick_direction=0.005  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.124
2026-08-06 01:23:55,341 INFO   [exec continuous log_std] move_direction: start=-1.4085 end=-1.4062   kick_direction: start=-1.4086 end=-1.4067
2026-08-06 01:23:55,341 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0002≈0.01°/step  epoch≈0.6°  dlog_std=0.00004  Δσ°=0.001/step)  kick_direction(dmean=0.0002≈0.01°/step  epoch≈0.7°  dlog_std=0.00003  Δσ°=0.000/step)
2026-08-06 01:23:55,341 INFO   [exec discrete Δlogit per opt step] exec_move=0.0025  sprint=0.0028  kick=0.0013  tackle_attempt=0.0019
2026-08-06 01:23:55,341 INFO   [per-head KL] shoot=-0.1362  pass_=-0.1330  move=+0.0816  tackle=-0.1272  gp_extra=-0.1404  mark=-0.1303  hold=-0.1303  exec_move=+0.1154  sprint=+0.0998  kick=+0.1305  tackle_attempt=+0.1268  move_dir=+0.0108  kick_dir=+0.0175
2026-08-06 01:23:55,342 INFO   [grad clip] main: 60/60 steps clipped (100%)  pre-clip norm mean=0.472 max=0.528  limit=0.35
              direction: 0/60 steps clipped (0%)  pre-clip norm mean=0.028 max=0.033  limit=0.05
2026-08-06 01:23:55,367 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=72,000  speed=176/s  reward=9.26/-1.36
  loss     policy=0.0145  value=0.3085(x0.5)=0.1542
           entropy=2.9084  kl=0.0564
  value    V=2.06±2.28  R=1.98±2.72  adv=-0.08±1.59
  moves    mv_ls=[-1.4062] (σ≈0.25, ≈14°) g=2.48e-02  d_move=[+0.0023] (Δσ≈0.032°)
           kk_ls=[-1.4067] (σ≈0.24, ≈14°)  d_kick=[+0.0019] (Δσ≈0.027°)
  heads    move= 20 get_poss= 80 exec_move= 84 sprint= 59 kick= 13 tackle= 12 shoot=  5 hold=
           5 tackle_prob=0.1349 kick_prob=0.1278
  vs       vs_immobile(112): 70%  vs_neural(109): 52%/21%
  reward   approach=+0.38  retreat=-1.64  get_possession=+581.40  progress=+7.28  lose_possession=-212.40
           ball_out=-10.50  illegal=-273.40  box_possession=+711.00  speed_bonus=+802.85  opponent_box=-160.00
  rew/ep   (mean/std/min/max per episode, 221 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.002    0.001    +0.000    +0.008
  retreat           -0.007    0.007    -0.043    -0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +2.631    2.594    +0.000   +19.800
  progress          +0.033    0.127    -0.232    +1.097
  lose_possession    -0.961    2.502   -18.000    +0.000
  ball_out          -0.048    0.331    -3.000    +0.000
  illegal           -1.230    1.973   -12.000    +0.000
  box_possession    +3.217    2.032    +0.000    +4.500
  speed_bonus       +3.633    3.796    +0.000   +10.200
  opponent_box      -0.724    0.961    -2.000    +0.000
  timeout           +0.000    0.000    +0.000    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    +0.000    0.000    +0.000    +0.000
──────────────────────────────────────────────────────────────────────
2026-08-06 01:23:55,850 INFO Saved checkpoint: checkpoints/phase1_run8/checkpoint4.pt
2026-08-06 01:23:55,850 INFO Logging to checkpoints/phase1_run8/training_log5.txt
2026-08-06 01:24:01,082 INFO   [eval vs rules] step=72,000  win=3/20 (15%)  outcomes={'box_possession': 3, 'opponent_box_possession': 15, 'miss': 1, 'other': 1}
2026-08-06 01:26:34,269 INFO   [advantage] mean=0.000  std=1.000  min=-7.093  max=9.059
2026-08-06 01:26:34,270 INFO   [ratio] mean=0.9816  std=0.3013  min=0.0012  max=23.4316  clipped=21.8%
2026-08-06 01:26:34,270 INFO   [exec head grad norm] move_direction=0.010  exec_move=0.091  sprint=0.058  kick=0.136  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.125
2026-08-06 01:26:34,270 INFO   [exec continuous log_std] move_direction: start=-1.4062 end=-1.4039   kick_direction: start=-1.4067 end=-1.4046
2026-08-06 01:26:34,271 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0002≈0.01°/step  epoch≈0.7°  dlog_std=0.00004  Δσ°=0.001/step)  kick_direction(dmean=0.0003≈0.02°/step  epoch≈1.0°  dlog_std=0.00003  Δσ°=0.000/step)
2026-08-06 01:26:34,271 INFO   [exec discrete Δlogit per opt step] exec_move=0.0026  sprint=0.0028  kick=0.0013  tackle_attempt=0.0019
2026-08-06 01:26:34,271 INFO   [per-head KL] shoot=-0.1447  pass_=-0.1495  move=+0.0914  tackle=-0.1420  gp_extra=-0.1703  mark=-0.1538  hold=-0.1470  exec_move=+0.1223  sprint=+0.1023  kick=+0.1298  tackle_attempt=+0.1368  move_dir=+0.0120  kick_dir=+0.0177
2026-08-06 01:26:34,271 INFO   [grad clip] main: 60/60 steps clipped (100%)  pre-clip norm mean=0.493 max=0.637  limit=0.35
              direction: 1/60 steps clipped (2%)  pre-clip norm mean=0.031 max=0.055  limit=0.05
2026-08-06 01:26:34,293 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=90,000  speed=151/s  reward=6.34/-1.76
  loss     policy=0.0169  value=0.2933(x0.5)=0.1467
           entropy=3.1907  kl=0.0583
  value    V=2.09±2.37  R=1.85±2.59  adv=-0.24±1.51
  moves    mv_ls=[-1.4039] (σ≈0.25, ≈14°) g=2.55e-02  d_move=[+0.0024] (Δσ≈0.033°)
           kk_ls=[-1.4046] (σ≈0.25, ≈14°)  d_kick=[+0.0021] (Δσ≈0.029°)
  heads    move= 20 get_poss= 80 exec_move= 83 sprint= 57 kick= 14 tackle= 14 shoot=  6 hold=
           6 tackle_prob=0.1508 kick_prob=0.1385
  vs       vs_immobile(102): 67%  vs_neural(100): 46%/15%
  reward   approach=+0.37  retreat=-1.53  get_possession=+522.00  progress=+4.91  lose_possession=-199.80
           ball_out=-6.00  illegal=-296.80  box_possession=+580.50  speed_bonus=+648.14  opponent_box=-122.00
  rew/ep   (mean/std/min/max per episode, 202 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.002    0.002    +0.000    +0.010
  retreat           -0.008    0.006    -0.034    -0.001
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +2.584    2.468    +0.000   +14.400
  progress          +0.024    0.076    -0.053    +0.551
  lose_possession    -0.989    2.315   -12.600    +0.000
  ball_out          -0.030    0.209    -1.500    +0.000
  illegal           -1.473    2.268   -13.800    +0.000
  box_possession    +2.874    2.162    +0.000    +4.500
  speed_bonus       +3.209    3.669    +0.000   +10.296
  opponent_box      -0.604    0.918    -2.000    +0.000
  timeout           +0.000    0.000    +0.000    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    +0.000    0.000    +0.000    +0.000
──────────────────────────────────────────────────────────────────────
2026-08-06 01:26:34,789 INFO Saved checkpoint: checkpoints/phase1_run8/checkpoint5.pt
2026-08-06 01:26:34,790 INFO Logging to checkpoints/phase1_run8/training_log6.txt
2026-08-06 01:26:39,172 INFO   [eval vs rules] step=90,000  win=4/20 (20%)  outcomes={'box_possession': 4, 'opponent_box_possession': 14, 'other': 2}
2026-08-06 01:29:01,607 INFO   [advantage] mean=0.000  std=1.000  min=-6.669  max=8.990
2026-08-06 01:29:01,607 INFO   [ratio] mean=0.9751  std=0.2551  min=0.0010  max=7.4414  clipped=22.8%
2026-08-06 01:29:01,608 INFO   [exec head grad norm] move_direction=0.009  exec_move=0.071  sprint=0.045  kick=0.108  kick_direction=0.006  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.116
2026-08-06 01:29:01,608 INFO   [exec continuous log_std] move_direction: start=-1.4039 end=-1.4014   kick_direction: start=-1.4046 end=-1.4026
2026-08-06 01:29:01,608 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0002≈0.01°/step  epoch≈0.8°  dlog_std=0.00004  Δσ°=0.001/step)  kick_direction(dmean=0.0003≈0.02°/step  epoch≈0.9°  dlog_std=0.00003  Δσ°=0.000/step)
2026-08-06 01:29:01,608 INFO   [exec discrete Δlogit per opt step] exec_move=0.0023  sprint=0.0021  kick=0.0011  tackle_attempt=0.0018
2026-08-06 01:29:01,608 INFO   [per-head KL] shoot=-0.1630  pass_=-0.1750  move=+0.1008  tackle=-0.1672  gp_extra=-0.1950  mark=-0.1684  hold=-0.1625  exec_move=+0.1274  sprint=+0.1054  kick=+0.1282  tackle_attempt=+0.1358  move_dir=+0.0117  kick_dir=+0.0213
2026-08-06 01:29:01,609 INFO   [grad clip] main: 60/60 steps clipped (100%)  pre-clip norm mean=0.472 max=0.575  limit=0.35
              direction: 0/60 steps clipped (0%)  pre-clip norm mean=0.030 max=0.041  limit=0.05
2026-08-06 01:29:01,636 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=108,000  speed=168/s  reward=7.59/-2.77
  loss     policy=0.0192  value=0.2824(x0.5)=0.1412
           entropy=3.4650  kl=0.0639
  value    V=1.98±2.25  R=1.91±2.69  adv=-0.07±1.47
  moves    mv_ls=[-1.4014] (σ≈0.25, ≈14°) g=2.69e-02  d_move=[+0.0025] (Δσ≈0.035°)
           kk_ls=[-1.4026] (σ≈0.25, ≈14°)  d_kick=[+0.0020] (Δσ≈0.029°)
  heads    move= 22 get_poss= 79 exec_move= 82 sprint= 57 kick= 14 tackle= 16 shoot=  6 hold=
           6 tackle_prob=0.1642 kick_prob=0.1470
  vs       vs_immobile(115): 70%  vs_neural(103): 52%/14%
  reward   approach=+0.36  retreat=-1.63  get_possession=+545.40  progress=+3.75  lose_possession=-192.60
           ball_out=-3.00  illegal=-295.40  box_possession=+666.00  speed_bonus=+714.14  opponent_box=-136.00
  rew/ep   (mean/std/min/max per episode, 218 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.002    0.001    +0.000    +0.009
  retreat           -0.008    0.007    -0.054    -0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +2.502    2.915    +0.000   +28.800
  progress          +0.017    0.056    -0.086    +0.639
  lose_possession    -0.883    2.808   -27.000    +0.000
  ball_out          -0.014    0.203    -3.000    +0.000
  illegal           -1.359    2.240   -13.400    +0.000
  box_possession    +3.055    2.101    +0.000    +4.500
  speed_bonus       +3.276    3.643    +0.000   +10.328
  opponent_box      -0.624    0.927    -2.000    +0.000
  timeout           +0.000    0.000    +0.000    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    +0.000    0.000    +0.000    +0.000
──────────────────────────────────────────────────────────────────────
2026-08-06 01:29:02,098 INFO Saved checkpoint: checkpoints/phase1_run8/checkpoint6.pt
2026-08-06 01:29:02,098 INFO Logging to checkpoints/phase1_run8/training_log7.txt
2026-08-06 01:29:07,962 INFO   [eval vs rules] step=108,000  win=1/20 (5%)  outcomes={'opponent_box_possession': 16, 'miss': 2, 'other': 1, 'box_possession': 1}
2026-08-06 01:30:54,851 INFO   [early stop e0 mb12]  KL=0.34598 > target=0.15  steps_this_update=13
    [per-head KL] shoot=-0.0649  pass_=-0.0678  move=+0.0337  tackle=-0.0671  gp_extra=-0.6433  mark=-0.0680  hold=-0.0650  exec_move=+0.0317  sprint=+0.0469  kick=+0.0476  tackle_attempt=+0.4582  move_dir=+0.2957
2026-08-06 01:31:07,062 INFO   [value-only continuation] 65 extra minibatch step(s)  after policy early-stop  final_val_loss=0.0235
2026-08-06 01:31:07,063 INFO   [advantage] mean=0.004  std=0.948  min=-6.115  max=10.469
2026-08-06 01:31:07,063 INFO   [ratio] mean=0.9798  std=0.2550  min=0.0001  max=21.8368  clipped=22.0%
2026-08-06 01:31:07,063 INFO   [exec head grad norm] move_direction=0.015  exec_move=0.093  sprint=0.075  kick=0.137  kick_direction=0.006  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.265
2026-08-06 01:31:07,063 INFO   [exec continuous log_std] move_direction: start=-1.4014 end=-1.4009   kick_direction: start=-1.4026 end=-1.4021
2026-08-06 01:31:07,064 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0003≈0.02°/step  epoch≈0.2°  dlog_std=0.00004  Δσ°=0.001/step)  kick_direction(dmean=0.0003≈0.02°/step  epoch≈0.2°  dlog_std=0.00004  Δσ°=0.001/step)
2026-08-06 01:31:07,064 INFO   [exec discrete Δlogit per opt step] exec_move=0.0024  sprint=0.0027  kick=0.0012  tackle_attempt=0.0018
2026-08-06 01:31:07,064 INFO   [per-head KL] shoot=-0.1630  pass_=-0.1728  move=+0.1196  tackle=-0.1715  gp_extra=-0.2391  mark=-0.1738  hold=-0.1686  exec_move=+0.1326  sprint=+0.1132  kick=+0.1322  tackle_attempt=+0.1817  move_dir=+0.0328  kick_dir=+0.0180
2026-08-06 01:31:07,064 INFO   [grad clip] main: 13/13 steps clipped (100%)  pre-clip norm mean=0.630 max=2.000  limit=0.35
              direction: 1/13 steps clipped (8%)  pre-clip norm mean=0.033 max=0.095  limit=0.05
2026-08-06 01:31:07,067 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=126,001  speed=172/s  reward=5.91/-2.47
  loss     policy=0.0078  value=0.3019(x0.5)=0.1509
           entropy=3.6499  kl=0.0766
  value    V=1.87±2.23  R=1.69±2.55  adv=-0.17±1.46
  moves    mv_ls=[-1.4009] (σ≈0.25, ≈14°) g=2.35e-02  d_move=[+0.0005] (Δσ≈0.007°)
           kk_ls=[-1.4021] (σ≈0.25, ≈14°)  d_kick=[+0.0005] (Δσ≈0.007°)
  heads    move= 23 get_poss= 79 exec_move= 81 sprint= 57 kick= 15 tackle= 18 shoot=  7 hold=
           8 tackle_prob=0.1753 kick_prob=0.1534
  vs       vs_immobile(96): 67%  vs_neural(106): 47%/9%
  reward   approach=+0.39  retreat=-1.64  get_possession=+511.20  progress=+4.35  lose_possession=-167.40
           ball_out=-9.00  illegal=-348.00  box_possession=+558.00  speed_bonus=+582.41  opponent_box=-120.00
  rew/ep   (mean/std/min/max per episode, 202 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.002    0.002    +0.000    +0.010
  retreat           -0.008    0.006    -0.034    -0.001
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +2.540    2.160    +0.000   +19.800
  progress          +0.022    0.095    -0.181    +1.172
  lose_possession    -0.829    2.089   -18.000    +0.000
  ball_out          -0.045    0.295    -3.000    +0.000
  illegal           -1.719    2.375   -11.800    +0.000
  box_possession    +2.762    2.191    +0.000    +4.500
  speed_bonus       +2.883    3.618    +0.000   +10.080
  opponent_box      -0.594    0.914    -2.000    +0.000
  timeout           +0.000    0.000    +0.000    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    +0.000    0.000    +0.000    +0.000
──────────────────────────────────────────────────────────────────────
2026-08-06 01:31:07,560 INFO Saved checkpoint: checkpoints/phase1_run8/checkpoint7.pt
2026-08-06 01:31:07,561 INFO Logging to checkpoints/phase1_run8/training_log8.txt
2026-08-06 01:31:12,791 INFO   [eval vs rules] step=126,001  win=2/20 (10%)  outcomes={'opponent_box_possession': 17, 'box_possession': 2, 'miss': 1}
2026-08-06 01:33:46,698 INFO   [advantage] mean=0.000  std=1.000  min=-6.595  max=10.536
2026-08-06 01:33:46,699 INFO   [ratio] mean=0.9793  std=0.2768  min=0.0007  max=15.1252  clipped=25.0%
2026-08-06 01:33:46,699 INFO   [exec head grad norm] move_direction=0.007  exec_move=0.056  sprint=0.049  kick=0.121  kick_direction=0.005  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.111
2026-08-06 01:33:46,699 INFO   [exec continuous log_std] move_direction: start=-1.4009 end=-1.3985   kick_direction: start=-1.4021 end=-1.3998
2026-08-06 01:33:46,700 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0003≈0.02°/step  epoch≈1.0°  dlog_std=0.00004  Δσ°=0.001/step)  kick_direction(dmean=0.0003≈0.02°/step  epoch≈1.1°  dlog_std=0.00004  Δσ°=0.001/step)
2026-08-06 01:33:46,700 INFO   [exec discrete Δlogit per opt step] exec_move=0.0017  sprint=0.0025  kick=0.0012  tackle_attempt=0.0017
2026-08-06 01:33:46,700 INFO   [per-head KL] shoot=-0.1820  pass_=-0.1899  move=+0.1160  tackle=-0.1889  gp_extra=-0.2149  mark=-0.1830  hold=-0.1833  exec_move=+0.1356  sprint=+0.1131  kick=+0.1349  tackle_attempt=+0.1438  move_dir=+0.0116  kick_dir=+0.0224
2026-08-06 01:33:46,700 INFO   [grad clip] main: 60/60 steps clipped (100%)  pre-clip norm mean=0.465 max=0.550  limit=0.35
              direction: 0/60 steps clipped (0%)  pre-clip norm mean=0.028 max=0.033  limit=0.05
2026-08-06 01:33:46,723 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=144,001  speed=152/s  reward=7.95/-4.31
  loss     policy=0.0196  value=0.2419(x0.5)=0.1209
           entropy=3.7699  kl=0.0598
  value    V=1.83±2.30  R=1.72±2.69  adv=-0.11±1.40
  moves    mv_ls=[-1.3985] (σ≈0.25, ≈14°) g=2.56e-02  d_move=[+0.0023] (Δσ≈0.033°)
           kk_ls=[-1.3998] (σ≈0.25, ≈14°)  d_kick=[+0.0023] (Δσ≈0.032°)
  heads    move= 24 get_poss= 77 exec_move= 80 sprint= 56 kick= 15 tackle= 17 shoot=  8 hold=
           8 tackle_prob=0.1813 kick_prob=0.1566
  vs       vs_immobile(113): 67%  vs_neural(87): 44%/21%
  reward   approach=+0.37  retreat=-1.58  get_possession=+522.00  progress=+5.89  lose_possession=-187.20
           ball_out=-4.50  illegal=-316.80  box_possession=+594.00  speed_bonus=+686.64  opponent_box=-112.00
  rew/ep   (mean/std/min/max per episode, 200 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.002    0.002    +0.000    +0.010
  retreat           -0.008    0.007    -0.043    -0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +2.601    2.734    +0.000   +23.400
  progress          +0.029    0.117    -0.073    +1.044
  lose_possession    -0.936    2.645   -21.600    +0.000
  ball_out          -0.022    0.182    -1.500    +0.000
  illegal           -1.569    2.657   -15.200    +0.000
  box_possession    +2.970    2.132    +0.000    +4.500
  speed_bonus       +3.433    3.694    +0.000   +10.256
  opponent_box      -0.560    0.898    -2.000    +0.000
  timeout           +0.000    0.000    +0.000    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    +0.000    0.000    +0.000    +0.000
──────────────────────────────────────────────────────────────────────
2026-08-06 01:33:47,213 INFO Saved checkpoint: checkpoints/phase1_run8/checkpoint8.pt
2026-08-06 01:33:47,213 INFO Logging to checkpoints/phase1_run8/training_log9.txt
2026-08-06 01:33:50,861 INFO   [eval vs rules] step=144,001  win=4/20 (20%)  outcomes={'opponent_box_possession': 12, 'box_possession': 4, 'other': 1, 'miss': 3}
2026-08-06 01:36:28,475 INFO   [advantage] mean=-0.000  std=1.000  min=-6.142  max=9.455
2026-08-06 01:36:28,476 INFO   [ratio] mean=0.9742  std=0.2659  min=0.0012  max=12.3628  clipped=27.7%
2026-08-06 01:36:28,476 INFO   [exec head grad norm] move_direction=0.006  exec_move=0.053  sprint=0.043  kick=0.107  kick_direction=0.005  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.116
2026-08-06 01:36:28,476 INFO   [exec continuous log_std] move_direction: start=-1.3985 end=-1.3961   kick_direction: start=-1.3998 end=-1.3973
2026-08-06 01:36:28,476 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0002≈0.01°/step  epoch≈0.8°  dlog_std=0.00004  Δσ°=0.001/step)  kick_direction(dmean=0.0003≈0.02°/step  epoch≈1.0°  dlog_std=0.00004  Δσ°=0.001/step)
2026-08-06 01:36:28,476 INFO   [exec discrete Δlogit per opt step] exec_move=0.0020  sprint=0.0021  kick=0.0011  tackle_attempt=0.0018
2026-08-06 01:36:28,477 INFO   [per-head KL] shoot=-0.1963  pass_=-0.1943  move=+0.1410  tackle=-0.1975  gp_extra=-0.2366  mark=-0.2039  hold=-0.1957  exec_move=+0.1543  sprint=+0.1246  kick=+0.1497  tackle_attempt=+0.1673  move_dir=+0.0122  kick_dir=+0.0230
2026-08-06 01:36:28,477 INFO   [grad clip] main: 60/60 steps clipped (100%)  pre-clip norm mean=0.447 max=0.519  limit=0.35
              direction: 0/60 steps clipped (0%)  pre-clip norm mean=0.029 max=0.036  limit=0.05
2026-08-06 01:36:28,515 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=162,001  speed=151/s  reward=6.84/-1.44
  loss     policy=0.0143  value=0.2981(x0.5)=0.1490
           entropy=4.0734  kl=0.0651
  value    V=1.60±2.13  R=1.51±2.57  adv=-0.10±1.45
  moves    mv_ls=[-1.3961] (σ≈0.25, ≈14°) g=2.61e-02  d_move=[+0.0024] (Δσ≈0.034°)
           kk_ls=[-1.3973] (σ≈0.25, ≈14°)  d_kick=[+0.0025] (Δσ≈0.036°)
  heads    move= 26 get_poss= 76 exec_move= 80 sprint= 58 kick= 16 tackle= 20 shoot=  9 hold=
           9 tackle_prob=0.1985 kick_prob=0.1664
  vs       vs_immobile(99): 62%  vs_neural(100): 46%/13%
  reward   approach=+0.35  retreat=-1.57  get_possession=+534.60  progress=+3.74  lose_possession=-205.20
           ball_out=-10.50  illegal=-378.80  box_possession=+540.00  speed_bonus=+619.93  opponent_box=-118.00
  rew/ep   (mean/std/min/max per episode, 199 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.002    0.001    +0.000    +0.008
  retreat           -0.008    0.006    -0.032    -0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +2.686    3.242    +0.000   +28.800
  progress          +0.019    0.089    -0.073    +1.049
  lose_possession    -1.031    3.158   -27.000    +0.000
  ball_out          -0.053    0.276    -1.500    +0.000
  illegal           -1.923    2.834   -15.000    +0.000
  box_possession    +2.714    2.202    +0.000    +4.500
  speed_bonus       +3.115    3.641    +0.000   +10.144
  opponent_box      -0.593    0.913    -2.000    +0.000
  timeout           +0.000    0.000    +0.000    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    +0.000    0.000    +0.000    +0.000
──────────────────────────────────────────────────────────────────────
2026-08-06 01:36:29,005 INFO Saved checkpoint: checkpoints/phase1_run8/checkpoint9.pt
2026-08-06 01:36:29,006 INFO Logging to checkpoints/phase1_run8/training_log10.txt
2026-08-06 01:36:33,770 INFO   [eval vs rules] step=162,001  win=3/20 (15%)  outcomes={'opponent_box_possession': 15, 'box_possession': 3, 'miss': 2}
2026-08-06 01:39:09,199 INFO   [advantage] mean=0.002  std=0.979  min=-7.027  max=10.099
2026-08-06 01:39:09,200 INFO   [ratio] mean=0.9731  std=0.2687  min=0.0012  max=9.9994  clipped=30.0%
2026-08-06 01:39:09,200 INFO   [exec head grad norm] move_direction=0.015  exec_move=0.125  sprint=0.078  kick=0.156  kick_direction=0.007  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.222
2026-08-06 01:39:09,200 INFO   [exec continuous log_std] move_direction: start=-1.3961 end=-1.3936   kick_direction: start=-1.3973 end=-1.3945
2026-08-06 01:39:09,200 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0003≈0.01°/step  epoch≈1.0°  dlog_std=0.00004  Δσ°=0.001/step)  kick_direction(dmean=0.0003≈0.02°/step  epoch≈1.3°  dlog_std=0.00004  Δσ°=0.001/step)
2026-08-06 01:39:09,200 INFO   [exec discrete Δlogit per opt step] exec_move=0.0022  sprint=0.0023  kick=0.0012  tackle_attempt=0.0016
2026-08-06 01:39:09,200 INFO   [per-head KL] shoot=-0.2040  pass_=-0.2068  move=+0.1687  tackle=-0.2002  gp_extra=-0.2440  mark=-0.2211  hold=-0.2056  exec_move=+0.1804  sprint=+0.1386  kick=+0.1550  tackle_attempt=+0.1819  move_dir=+0.0114  kick_dir=+0.0233
2026-08-06 01:39:09,201 INFO   [grad clip] main: 65/65 steps clipped (100%)  pre-clip norm mean=0.657 max=6.460  limit=0.35
              direction: 4/65 steps clipped (6%)  pre-clip norm mean=0.039 max=0.298  limit=0.05
2026-08-06 01:39:09,225 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=180,002  speed=147/s  reward=9.31/-4.61
  loss     policy=0.0177  value=0.2663(x0.5)=0.1331
           entropy=4.4239  kl=0.0605
  value    V=1.59±2.32  R=1.48±2.64  adv=-0.10±1.41
  moves    mv_ls=[-1.3936] (σ≈0.25, ≈14°) g=2.73e-02  d_move=[+0.0026] (Δσ≈0.036°)
           kk_ls=[-1.3945] (σ≈0.25, ≈14°)  d_kick=[+0.0028] (Δσ≈0.039°)
  heads    move= 27 get_poss= 76 exec_move= 78 sprint= 58 kick= 17 tackle= 21 shoot= 10 hold= 10 tackle_prob=0.2174 kick_prob=0.1768
  vs       vs_immobile(92): 68%  vs_neural(95): 55%/16%
  reward   approach=+0.38  retreat=-1.65  get_possession=+482.40  progress=+3.97  lose_possession=-176.40
           ball_out=-1.50  illegal=-413.60  box_possession=+585.00  speed_bonus=+638.26  opponent_box=-134.00
  rew/ep   (mean/std/min/max per episode, 187 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.002    0.002    +0.000    +0.011
  retreat           -0.009    0.008    -0.041    -0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +2.570    2.317    +0.000   +14.400
  progress          +0.021    0.097    -0.048    +1.083
  lose_possession    -0.934    2.189   -12.600    +0.000
  ball_out          -0.008    0.109    -1.500    +0.000
  illegal           -2.185    3.122   -16.200    +0.000
  box_possession    +3.128    2.071    +0.000    +4.500
  speed_bonus       +3.413    3.557    +0.000   +10.096
  opponent_box      -0.717    0.959    -2.000    +0.000
  timeout           +0.000    0.000    +0.000    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    +0.000    0.000    +0.000    +0.000
──────────────────────────────────────────────────────────────────────
2026-08-06 01:39:09,690 INFO Saved checkpoint: checkpoints/phase1_run8/checkpoint10.pt
2026-08-06 01:39:09,691 INFO Logging to checkpoints/phase1_run8/training_log11.txt
2026-08-06 01:39:14,331 INFO   [eval vs rules] step=180,002  win=3/20 (15%)  outcomes={'opponent_box_possession': 16, 'box_possession': 3, 'miss': 1}
2026-08-06 01:51:30,154 INFO Checkpoint dir: checkpoints/phase1_run9
2026-08-06 01:51:30,173 INFO Starting training: phase=phase1_get_possession, total_steps=200,000
2026-08-06 01:51:30,173 INFO Phase description: 1v1 scenario: learn to get possession and bring the ball toward the opponent box.  Decision network frozen except Move/GetPossession and latent vector.
2026-08-06 01:51:30,228 INFO _from_ckpt: overriding bc_pretrain_epochs=10 → 2
2026-08-06 01:51:30,228 INFO _from_ckpt: overriding demo_value_pretrain_epochs=10 → 0
2026-08-06 01:51:30,228 INFO _from_ckpt: overriding value_pretrain_epochs=45 → 35
2026-08-06 01:51:30,228 INFO _from_ckpt: overriding aux_coeff_start=0.0 → 0.8
2026-08-06 01:51:30,228 INFO _from_ckpt: overriding aux_coeff_end=0.0 → 0.0
2026-08-06 01:51:30,228 INFO _from_ckpt: overriding aux_coeff_anneal_fraction=0.3 → 0.35
2026-08-06 01:51:31,664 INFO Logging to checkpoints/phase1_run9/training_log1.txt
2026-08-06 01:51:31,665 INFO --latest(-pretrain): resolved to checkpoints/phase1_run8/latest.pt
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/vincent/Documents/not_work/repos/FootballCoach/src/footballcoach/ai/scripts/train.py", line 523, in <module>
    main()
  File "/home/vincent/Documents/not_work/repos/FootballCoach/src/footballcoach/ai/scripts/train.py", line 257, in main
    trainer.load_checkpoint(ptrain_path)
  File "/home/vincent/Documents/not_work/repos/FootballCoach/src/footballcoach/ai/ppo/ppo_trainer.py", line 3498, in load_checkpoint
    self.execution_net.load_state_dict(ckpt["execution_net"])
  File "/home/vincent/Documents/not_work/repos/FootballCoach/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 2638, in load_state_dict
    raise RuntimeError(
RuntimeError: Error(s) in loading state_dict for ExecutionNetwork:
	Missing key(s) in state_dict: "value_head.1.weight", "value_head.1.bias", "value_head.4.weight", "value_head.4.bias". 
	Unexpected key(s) in state_dict: "value_head.0.weight", "value_head.0.bias", "value_head.2.weight", "value_head.2.bias". 
2026-08-06 01:53:30,653 INFO Checkpoint dir: checkpoints/phase1_run10
2026-08-06 01:53:30,667 INFO Starting training: phase=phase1_get_possession, total_steps=200,000
2026-08-06 01:53:30,667 INFO Phase description: 1v1 scenario: learn to get possession and bring the ball toward the opponent box.  Decision network frozen except Move/GetPossession and latent vector.
2026-08-06 01:53:31,877 INFO Logging to checkpoints/phase1_run10/training_log1.txt
2026-08-06 01:53:32,040 INFO Loading 1000 demonstration file(s) from demonstrations/phase1
2026-08-06 01:53:35,907 INFO Dataset: 385,963 steps loaded
2026-08-06 01:53:35,908 INFO Offline BC dataset: 385,963 steps from demonstrations/phase1
2026-08-06 01:53:35,908 INFO Reward diagnostic: running 40 rules-vs-rules episodes...
2026-08-06 01:53:36,612 INFO Reward diagnostic (40 ep, rules vs rules): mean_ep_rew=7.41  per_episode: get_possession=+1.27  progress=+0.03  lose_possession=-0.95  ball_out=-0.04  box_possession=+3.26  speed_bonus=+4.17  opponent_box=-0.34
2026-08-06 01:53:36,623 INFO BC pos_weight (auto-computed from dataset): kick=2.00  tackle_attempt=2.00
2026-08-06 01:53:36,623 INFO Combined BC + value pre-training: 6 epoch(s), batch_size=752, dataset=385,963 steps, rollout_steps=26000
2026-08-06 01:53:36,714 INFO Phase 0 — decision-net warm-up (BC + execution_net.value_head MSE; single value head convention): 10 epoch(s), gamma=0.995, returns mean=8.09  std=6.09  lr=0.012  phase0_value_coef=0.85
/home/vincent/Documents/not_work/repos/FootballCoach/src/footballcoach/ai/ppo/bc.py:685: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /__w/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:822.)
  "decision":       float(dec_loss[valid].mean()),
2026-08-06 01:53:53,869 INFO   Phase 0 epoch 1/10: loss=2.8966  dec_bc=2.0811  val=0.9594(x0.85)=0.8155
2026-08-06 01:54:12,001 INFO   Phase 0 epoch 2/10: loss=2.5636  dec_bc=1.9077  val=0.7717(x0.85)=0.6559
2026-08-06 01:54:29,842 INFO   Phase 0 epoch 3/10: loss=2.5270  dec_bc=1.9043  val=0.7326(x0.85)=0.6227
2026-08-06 01:54:48,843 INFO   Phase 0 epoch 4/10: loss=2.5075  dec_bc=1.9030  val=0.7111(x0.85)=0.6045
2026-08-06 01:55:06,772 INFO   Phase 0 epoch 5/10: loss=2.4902  dec_bc=1.9016  val=0.6924(x0.85)=0.5886
2026-08-06 01:55:24,596 INFO   Phase 0 epoch 6/10: loss=2.4805  dec_bc=1.9012  val=0.6815(x0.85)=0.5793
