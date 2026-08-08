2026-08-07 19:43:04,526 INFO Loaded checkpoint: checkpoints/phase1_run35/latest.pt (step 480000)
2026-08-07 19:43:04,526 INFO Loaded checkpoint for re-pretraining: checkpoints/phase1_run35/latest.pt — will still run BC/value pre-training
2026-08-07 19:43:04,526 INFO --reset-dir-log-std: move/kick dir_log_std reset to -2.8
2026-08-07 19:43:04,537 INFO Loading 1250 demonstration file(s) from demonstrations/phase1
2026-08-07 19:43:09,987 INFO Dataset: 475,414 steps loaded
2026-08-07 19:43:09,989 INFO Offline BC dataset: 475,414 steps from demonstrations/phase1/
2026-08-07 19:43:09,989 INFO Reward diagnostic: running 40 rules-vs-rules episodes...
2026-08-07 19:43:10,922 INFO Reward diagnostic (40 ep, rules vs rules): mean_ep_rew=1.53  per_episode: get_possession=+0.90  lose_possession=-0.29  ball_out=-0.12  box_possession=+1.12  speed_bonus=+0.76  opponent_box=-0.82  stamina_penalty=-0.01
2026-08-07 19:43:10,935 INFO BC pos_weight (auto-computed from dataset): kick=1.50  tackle_attempt=1.50
2026-08-07 19:43:10,936 INFO Combined BC + value pre-training: 1 epoch(s), batch_size=800, dataset=475,414 steps, rollout_steps=18000
2026-08-07 19:43:10,942 INFO   BC pretrain split: 355,681 train rows  |  64,066 val rows
2026-08-07 19:43:11,168 INFO   Downsample trivial rows (epoch 1): 127,947/419,747 (30.5%) rows classified trivial, excluding ~102,358 this epoch (frac=0.80)
/home/vincent/Documents/not_work/repos/FootballCoach/src/footballcoach/ai/ppo/bc.py:683: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /__w/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:822.)
  "decision":       float(dec_loss[valid].mean()),
2026-08-07 19:45:05,346 INFO   BC epoch 1/1  (114.4s)
    loss       bc=2.2171  bc_adj=0.1507(floor=2.0664)
    heads      dir_cos=0.982  kick_dir_cos=0.999
               move_prob=0.887  sprint_prob=0.813  kick_prob=0.066  tackle_prob=0.257
    pr/rec     kick:   p=0.993  r=0.995  f1=0.994  (tp=30558 fp=210 fn=150)
               tackle: p=0.985  r=0.994  f1=0.990  (tp=250133 fp=3776 fn=1467)
    breakdown  decision=0.689  exec_bce=0.778  sprint=0.211  move=0.197  tackle_attempt=0.200  direction=0.052
               region=0.010  kick=0.17132  kick_direction=0.00442  kick_power=0.00119  kick_spin=0.00000
2026-08-07 19:45:07,649 INFO     val        bc_val_loss=2.1949  best=2.1949  (improved)
2026-08-07 19:45:07,654 INFO BC pre-training done (1 epoch(s), final bc_loss=2.2171)
2026-08-07 19:45:07,655 INFO Value pre-training: 18000 steps, 15 epochs, lr=3e-05
2026-08-07 19:45:07,656 INFO   [value pretrain rollout] parallel collection: 6 worker(s), ~3000 steps/worker
2026-08-07 19:45:09,554 INFO Frozen decision_net.shoot_logit
2026-08-07 19:45:09,554 INFO Frozen decision_net.pass_logit
2026-08-07 19:45:09,555 INFO Frozen decision_net.tackle_logit
2026-08-07 19:45:09,555 INFO Frozen decision_net.get_possession_raw
2026-08-07 19:45:09,555 INFO Frozen decision_net.mark_logit
2026-08-07 19:45:09,555 INFO Frozen decision_net.hold_position_logit
2026-08-07 19:45:09,555 INFO Frozen decision_net.pass_target_logits
2026-08-07 19:45:09,555 INFO Frozen decision_net.tackle_target_logits
2026-08-07 19:45:09,555 INFO Frozen decision_net.mark_target_logits
2026-08-07 19:45:09,555 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-07 19:45:09,611 INFO Frozen decision_net.shoot_logit
2026-08-07 19:45:09,611 INFO Frozen decision_net.pass_logit
2026-08-07 19:45:09,611 INFO Frozen decision_net.tackle_logit
2026-08-07 19:45:09,611 INFO Frozen decision_net.get_possession_raw
2026-08-07 19:45:09,611 INFO Frozen decision_net.mark_logit
2026-08-07 19:45:09,611 INFO Frozen decision_net.hold_position_logit
2026-08-07 19:45:09,611 INFO Frozen decision_net.pass_target_logits
2026-08-07 19:45:09,611 INFO Frozen decision_net.tackle_target_logits
2026-08-07 19:45:09,611 INFO Frozen decision_net.mark_target_logits
2026-08-07 19:45:09,611 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-07 19:45:09,682 INFO Frozen decision_net.shoot_logit
2026-08-07 19:45:09,682 INFO Frozen decision_net.pass_logit
2026-08-07 19:45:09,682 INFO Frozen decision_net.tackle_logit
2026-08-07 19:45:09,682 INFO Frozen decision_net.get_possession_raw
2026-08-07 19:45:09,682 INFO Frozen decision_net.mark_logit
2026-08-07 19:45:09,682 INFO Frozen decision_net.hold_position_logit
2026-08-07 19:45:09,682 INFO Frozen decision_net.pass_target_logits
2026-08-07 19:45:09,682 INFO Frozen decision_net.tackle_target_logits
2026-08-07 19:45:09,682 INFO Frozen decision_net.mark_target_logits
2026-08-07 19:45:09,682 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-07 19:45:09,709 INFO Frozen decision_net.shoot_logit
2026-08-07 19:45:09,709 INFO Frozen decision_net.pass_logit
2026-08-07 19:45:09,709 INFO Frozen decision_net.tackle_logit
2026-08-07 19:45:09,709 INFO Frozen decision_net.get_possession_raw
2026-08-07 19:45:09,709 INFO Frozen decision_net.mark_logit
2026-08-07 19:45:09,709 INFO Frozen decision_net.hold_position_logit
2026-08-07 19:45:09,709 INFO Frozen decision_net.pass_target_logits
2026-08-07 19:45:09,709 INFO Frozen decision_net.tackle_target_logits
2026-08-07 19:45:09,709 INFO Frozen decision_net.mark_target_logits
2026-08-07 19:45:09,710 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-07 19:45:09,719 INFO Frozen decision_net.shoot_logit
2026-08-07 19:45:09,719 INFO Frozen decision_net.pass_logit
2026-08-07 19:45:09,719 INFO Frozen decision_net.tackle_logit
2026-08-07 19:45:09,719 INFO Frozen decision_net.get_possession_raw
2026-08-07 19:45:09,719 INFO Frozen decision_net.mark_logit
2026-08-07 19:45:09,719 INFO Frozen decision_net.hold_position_logit
2026-08-07 19:45:09,719 INFO Frozen decision_net.pass_target_logits
2026-08-07 19:45:09,719 INFO Frozen decision_net.tackle_target_logits
2026-08-07 19:45:09,719 INFO Frozen decision_net.mark_target_logits
2026-08-07 19:45:09,719 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-07 19:45:09,733 INFO Frozen decision_net.shoot_logit
2026-08-07 19:45:09,733 INFO Frozen decision_net.pass_logit
2026-08-07 19:45:09,733 INFO Frozen decision_net.tackle_logit
2026-08-07 19:45:09,733 INFO Frozen decision_net.get_possession_raw
2026-08-07 19:45:09,733 INFO Frozen decision_net.mark_logit
2026-08-07 19:45:09,733 INFO Frozen decision_net.hold_position_logit
2026-08-07 19:45:09,733 INFO Frozen decision_net.pass_target_logits
2026-08-07 19:45:09,733 INFO Frozen decision_net.tackle_target_logits
2026-08-07 19:45:09,733 INFO Frozen decision_net.mark_target_logits
2026-08-07 19:45:09,733 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-07 19:45:37,059 INFO   [value pretrain rollout] dropped 224 trailing (incomplete-episode) step(s) across workers before MC-return fit
2026-08-07 19:45:37,065 INFO   [value pretrain rollout] mean_return=4.04 (402 episode(s))  vs[win/loss/tout/miss]  vs_rules(0): n/a  vs_immobile(402): 64.7%/0.0%/0.5%/9.2%/26%  vs_neural(0): n/a
2026-08-07 19:45:37,066 INFO   [value pretrain rollout] ep_len 13.1±8.1s  (n=402, min=0.6s, max=50.0s)
2026-08-07 19:45:37,067 INFO   [value pretrain rollout] rew/ep (mean/std/min/max per episode, 402 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.841    0.373    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.002    0.045    -0.900    +0.000
  ball_out          -0.050    0.496    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.617    1.195    +0.000    +2.500
  speed_bonus       +1.643    1.527    +0.000    +4.463
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.007    0.106    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.028    +0.000
2026-08-07 19:45:37,167 INFO   Value pretrain split: 342 train eps (15172 steps)  |  60 val eps (2604 steps)
2026-08-07 19:45:41,097 INFO   Value epoch 1/15: train=0.6153 rmse=1.90  val=0.6706 val_rmse=1.98 (std=2.4)
    V(train)=+3.204  R(train)=+3.232  |  V(val)=+3.439  R(val)=+3.337
2026-08-07 19:45:44,944 INFO   Value epoch 2/15: train=0.5456 rmse=1.79  val=0.6635 val_rmse=1.97 (std=2.4)
    V(train)=+3.215  R(train)=+3.232  |  V(val)=+3.492  R(val)=+3.337
2026-08-07 19:45:48,706 INFO   Value epoch 3/15: train=0.5131 rmse=1.73  val=0.6636 val_rmse=1.97 (std=2.4)
    V(train)=+3.229  R(train)=+3.232  |  V(val)=+3.433  R(val)=+3.337
2026-08-07 19:45:52,451 INFO   Value epoch 4/15: train=0.4960 rmse=1.70  val=0.6325 val_rmse=1.92 (std=2.4)
    V(train)=+3.239  R(train)=+3.232  |  V(val)=+3.229  R(val)=+3.337
2026-08-07 19:45:56,314 INFO   Value epoch 5/15: train=0.4826 rmse=1.68  val=0.6402 val_rmse=1.94 (std=2.4)
    V(train)=+3.228  R(train)=+3.232  |  V(val)=+3.246  R(val)=+3.337
2026-08-07 19:46:00,176 INFO   Value epoch 6/15: train=0.4727 rmse=1.66  val=0.7398 val_rmse=2.08 (std=2.4)
    V(train)=+3.246  R(train)=+3.232  |  V(val)=+3.832  R(val)=+3.337
2026-08-07 19:46:03,964 INFO   Value epoch 7/15: train=0.4625 rmse=1.65  val=0.6861 val_rmse=2.00 (std=2.4)
    V(train)=+3.219  R(train)=+3.232  |  V(val)=+3.307  R(val)=+3.337
2026-08-07 19:46:07,830 INFO   Value epoch 8/15: train=0.4565 rmse=1.63  val=0.6936 val_rmse=2.02 (std=2.4)
    V(train)=+3.224  R(train)=+3.232  |  V(val)=+3.642  R(val)=+3.337
2026-08-07 19:46:07,831 INFO   [value pretrain] early stop at epoch 8 (val stagnant for 4 epochs, best=0.6325)
2026-08-07 19:46:07,833 INFO   [value pretrain] restored best-val weights (val_loss=0.6325)
2026-08-07 19:46:07,833 INFO Value pre-training done (8 epoch(s), final train_loss=0.4565)
2026-08-07 19:46:20,170 INFO BC check after value warm-up: bc_loss=2.1942 (before=2.2171, delta=-0.0229)  OK
2026-08-07 19:46:20,170 INFO Combined pre-training complete.
2026-08-07 19:46:51,154 INFO Pre-PPO eval (rules opp): win=25.8%  mean_rew=0.416  V=2.539  R=0.014  gap=+2.525  outcomes={'other': 32, 'opponent_box_possession': 53, 'box_possession': 33, 'timeout': 2, 'miss': 8}
2026-08-07 19:46:51,155 INFO   rew breakdown (rules, per ep): opponent_box=-1.24  get_possession=+1.09  box_possession=+0.64  lose_possession=-0.58  speed_bonus=+0.55  timeout=-0.02  stamina_penalty=-0.01
2026-08-07 19:47:21,174 INFO Pre-PPO eval (immobile opp): win=56.2%  mean_rew=3.382  V=2.891  R=1.921  gap=+0.969  outcomes={'other': 35, 'box_possession': 72, 'miss': 18, 'timeout': 3}
2026-08-07 19:47:21,174 INFO   rew breakdown (immobile, per ep): box_possession=+1.41  speed_bonus=+1.33  get_possession=+0.71  timeout=-0.04  lose_possession=-0.02  stamina_penalty=-0.01
2026-08-07 19:48:25,302 INFO Pre-PPO eval (self-play):   win=53.9%  mean_rew=2.472  V=2.494  R=1.457  gap=+1.037  outcomes={'other': 23, 'box_possession': 69, 'opponent_box_possession': 24, 'timeout': 1, 'miss': 11}
2026-08-07 19:48:25,302 INFO   rew breakdown (self-play, per ep): opponent_box=-2.18  get_possession=+2.02  box_possession=+1.82  speed_bonus=+1.43  lose_possession=-1.10  ball_out=-0.08  stamina_penalty=-0.03  timeout=-0.02
2026-08-07 19:48:25,302 INFO   [seeded eval] running 12x8 episodes across 7 worker process(es)...
2026-08-07 19:48:28,533 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 19:48:28,533 INFO Baseline (rules vs rules, 12 trials): trainee_win=83.3%  outcomes={'box_possession': 80, 'other': 8, 'opponent_box_possession': 8}
2026-08-07 19:48:28,534 INFO Frozen decision_net.shoot_logit
2026-08-07 19:48:28,534 INFO Frozen decision_net.pass_logit
2026-08-07 19:48:28,534 INFO Frozen decision_net.tackle_logit
2026-08-07 19:48:28,534 INFO Frozen decision_net.get_possession_raw
2026-08-07 19:48:28,534 INFO Frozen decision_net.mark_logit
2026-08-07 19:48:28,534 INFO Frozen decision_net.hold_position_logit
2026-08-07 19:48:28,534 INFO Frozen decision_net.pass_target_logits
2026-08-07 19:48:28,534 INFO Frozen decision_net.tackle_target_logits
2026-08-07 19:48:28,534 INFO Frozen decision_net.mark_target_logits
2026-08-07 19:48:28,534 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-07 19:48:28,534 INFO PPO parallel training started: 6 worker(s), ~4000 steps/worker/rollout, steps_so_far=0  target=600,000
2026-08-07 19:48:30,406 INFO Frozen decision_net.shoot_logit
2026-08-07 19:48:30,407 INFO Frozen decision_net.pass_logit
2026-08-07 19:48:30,407 INFO Frozen decision_net.tackle_logit
2026-08-07 19:48:30,407 INFO Frozen decision_net.get_possession_raw
2026-08-07 19:48:30,407 INFO Frozen decision_net.mark_logit
2026-08-07 19:48:30,407 INFO Frozen decision_net.hold_position_logit
2026-08-07 19:48:30,407 INFO Frozen decision_net.pass_target_logits
2026-08-07 19:48:30,407 INFO Frozen decision_net.tackle_target_logits
2026-08-07 19:48:30,407 INFO Frozen decision_net.mark_target_logits
2026-08-07 19:48:30,407 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-07 19:48:30,416 INFO Frozen decision_net.shoot_logit
2026-08-07 19:48:30,416 INFO Frozen decision_net.pass_logit
2026-08-07 19:48:30,416 INFO Frozen decision_net.tackle_logit
2026-08-07 19:48:30,416 INFO Frozen decision_net.get_possession_raw
2026-08-07 19:48:30,416 INFO Frozen decision_net.mark_logit
2026-08-07 19:48:30,416 INFO Frozen decision_net.hold_position_logit
2026-08-07 19:48:30,416 INFO Frozen decision_net.pass_target_logits
2026-08-07 19:48:30,416 INFO Frozen decision_net.tackle_target_logits
2026-08-07 19:48:30,416 INFO Frozen decision_net.mark_target_logits
2026-08-07 19:48:30,416 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-07 19:48:30,425 INFO Frozen decision_net.shoot_logit
2026-08-07 19:48:30,426 INFO Frozen decision_net.pass_logit
2026-08-07 19:48:30,426 INFO Frozen decision_net.tackle_logit
2026-08-07 19:48:30,426 INFO Frozen decision_net.get_possession_raw
2026-08-07 19:48:30,426 INFO Frozen decision_net.mark_logit
2026-08-07 19:48:30,426 INFO Frozen decision_net.hold_position_logit
2026-08-07 19:48:30,426 INFO Frozen decision_net.pass_target_logits
2026-08-07 19:48:30,426 INFO Frozen decision_net.tackle_target_logits
2026-08-07 19:48:30,426 INFO Frozen decision_net.mark_target_logits
2026-08-07 19:48:30,426 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-07 19:48:30,436 INFO Frozen decision_net.shoot_logit
2026-08-07 19:48:30,436 INFO Frozen decision_net.pass_logit
2026-08-07 19:48:30,436 INFO Frozen decision_net.tackle_logit
2026-08-07 19:48:30,436 INFO Frozen decision_net.get_possession_raw
2026-08-07 19:48:30,436 INFO Frozen decision_net.mark_logit
2026-08-07 19:48:30,436 INFO Frozen decision_net.hold_position_logit
2026-08-07 19:48:30,436 INFO Frozen decision_net.pass_target_logits
2026-08-07 19:48:30,436 INFO Frozen decision_net.tackle_target_logits
2026-08-07 19:48:30,436 INFO Frozen decision_net.mark_target_logits
2026-08-07 19:48:30,436 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-07 19:48:30,457 INFO Frozen decision_net.shoot_logit
2026-08-07 19:48:30,458 INFO Frozen decision_net.pass_logit
2026-08-07 19:48:30,458 INFO Frozen decision_net.tackle_logit
2026-08-07 19:48:30,458 INFO Frozen decision_net.get_possession_raw
2026-08-07 19:48:30,458 INFO Frozen decision_net.mark_logit
2026-08-07 19:48:30,458 INFO Frozen decision_net.hold_position_logit
2026-08-07 19:48:30,458 INFO Frozen decision_net.pass_target_logits
2026-08-07 19:48:30,458 INFO Frozen decision_net.tackle_target_logits
2026-08-07 19:48:30,458 INFO Frozen decision_net.mark_target_logits
2026-08-07 19:48:30,458 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-07 19:48:30,488 INFO Frozen decision_net.shoot_logit
2026-08-07 19:48:30,489 INFO Frozen decision_net.pass_logit
2026-08-07 19:48:30,489 INFO Frozen decision_net.tackle_logit
2026-08-07 19:48:30,489 INFO Frozen decision_net.get_possession_raw
2026-08-07 19:48:30,489 INFO Frozen decision_net.mark_logit
2026-08-07 19:48:30,489 INFO Frozen decision_net.hold_position_logit
2026-08-07 19:48:30,489 INFO Frozen decision_net.pass_target_logits
2026-08-07 19:48:30,489 INFO Frozen decision_net.tackle_target_logits
2026-08-07 19:48:30,489 INFO Frozen decision_net.mark_target_logits
2026-08-07 19:48:30,489 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-07 19:49:04,195 INFO   [early stop e0 mb0]  KL=0.49991 > target=0.12  steps_this_update=1
    [per-head KL] exec_move=+0.0271  sprint=+0.0226  kick=+0.0054  tackle_attempt=+0.0010  move_dir=+0.3483  kick_dir=+0.0950
2026-08-07 19:49:04,197 INFO   [KL mean=0.4999 median=0.4999 > 0.05] ratio percentiles:  p5=0.100  p25=0.841  p50=0.991  p75=1.000  p95=1.093  max=17.036
  move_dir_log_std=[-2.7999818325042725]  kick_dir_log_std=[-2.7999770641326904]
2026-08-07 19:49:04,210 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.097  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.209  kick=-0.217  t_att=-0.143
    move_dir=2.969 (min=-0.683 max=3.762)  kick_dir=0.202 (min=0.000 max=3.762)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.17
  [worst sample] idx=81  ratio=27.376  adv=-0.588  old_lp=-3.408  new_lp=-0.098
    stored move_dir=167.6°  new_mean=165.5°  angular_diff=2.1°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx=  81  ratio=  27.376  adv=-0.588  lp: old=-3.408  new=-0.098
      rew=+0.0000  ret=+2.5745  val=+3.1621  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9634  sprint_p_new=0.0311  kick_p_new=0.0368  tackle_attempt_p_new=0.0384
    idx=  52  ratio=  26.483  adv=-4.948  lp: old=-3.389  new=-0.112
      rew=+0.0000  ret=+0.4356  val=+5.3832  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9622  sprint_p_new=0.0225  kick_p_new=0.0390  tackle_attempt_p_new=0.0499
  [best sample (highest new_lp)] idx=151  new_lp=0.074  adv=-0.493  stored move_dir=60.0°  new_mean=61.4°
    per-head contributions: move_dir:0.184  move:-0.021  tackle_attempt:-0.028  kick:-0.050
2026-08-07 19:49:04,210 INFO   [advantage] mean=0.008  std=0.994  min=-5.283  max=4.376
2026-08-07 19:49:04,211 INFO   [ratio] mean=0.8796  std=0.4314  min=0.0000  max=17.0361  clipped=29.3%
2026-08-07 19:49:04,211 INFO   [exec head grad norm] move_direction=0.073  exec_move=0.051  sprint=0.117  kick=0.078  kick_direction=0.016  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.096
2026-08-07 19:49:04,211 INFO   [exec continuous log_std] move_direction: start=-2.8000 end=-2.8000   kick_direction: start=-2.8000 end=-2.8000
2026-08-07 19:49:04,211 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0003≈0.02°/step  epoch≈0.0°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0004≈0.02°/step  epoch≈0.0°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-07 19:49:04,211 INFO   [exec discrete Δlogit per opt step] exec_move=0.0007  sprint=0.0013  kick=0.0006  tackle_attempt=0.0013
2026-08-07 19:49:04,211 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0006  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0271  sprint=+0.0226  kick=+0.0054  tackle_attempt=+0.0010  move_dir=+0.3483  kick_dir=+0.0950
2026-08-07 19:49:04,212 INFO   [grad clip] main: 1/1 steps clipped (100%)  pre-clip norm mean=1.038 max=1.038  limit=0.4
              direction: 1/1 steps clipped (100%)  pre-clip norm mean=0.084 max=0.084  limit=0.02
2026-08-07 19:49:04,242 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=24,000  speed=708/s  reward=4.46
  loss     policy=0.0489  value=0.6088(x0.5)=0.3044
           entropy=1.3249  kl=0.4999
  value    V=2.96±1.66  R=3.11±1.82  adv=0.15±1.39
  moves    mv_ls=[-2.8000] (σ≈0.06, ≈3°) g=3.86e-02
           kk_ls=[-2.8000] (σ≈0.06, ≈3°)
  heads    move= 43 get_poss= 57 exec_move= 91 sprint= 48 kick=  4 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0400 kick_prob=0.0433
  vs       vs[win/loss/tout/miss]  vs_immobile(517): 66.9%/0.0%/0.4%/10.8%/22%
  ep_len   13.7±8.2s  (n=517, min=0.5s, max=50.0s)
  reward   get_possession=+442.00  lose_possession=-1.80  ball_out=-30.00  box_possession=+865.00
           speed_bonus=+815.00  timeout=-3.00  stamina_penalty=-3.01
  rew/ep   (mean/std/min/max per episode, 517 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.855    0.363    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.003    0.056    -0.900    +0.000
  ball_out          -0.058    0.536    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.673    1.176    +0.000    +2.500
  speed_bonus       +1.576    1.507    +0.000    +4.326
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.006    0.093    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.024    +0.000
  rew/step (per-step stats, n=24000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     445    +0.019    0.135     +4.015     1.401     +0.379      2.9701      1.253     3.571
  lose_possession       2    -0.000    0.008     +3.426     0.117     +0.491      0.2577      0.491     0.606
  ball_out             6    -0.001    0.079     -4.833     0.373     -5.667     34.3012      5.667     7.463
  box_possession     346    +0.036    0.298     +4.850     1.241     +0.533      1.7660      1.014     2.597
  speed_bonus        330    +0.034    0.318     +4.964     1.154     +0.623      1.7461      0.999     2.662
  timeout              2    -0.000    0.014     -1.500     0.000     -0.460      0.2563      0.460     0.651
  stamina_penalty     326    -0.000    0.001     +4.906     1.222     +0.534      1.7108      0.992     2.516
  gae/td   mean_return=+3.108  std_return=1.824  mean_gae=+0.150  mean_sq_td=1.9683
──────────────────────────────────────────────────────────────────────
2026-08-07 19:49:04,270 INFO Saved checkpoint: checkpoints/phase1_run38/checkpoint1.pt
2026-08-07 19:49:04,270 INFO Logging to checkpoints/phase1_run38/training_log2.txt
2026-08-07 19:49:04,271 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 19:49:15,923 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 19:49:15,924 INFO   [eval vs immobile] step=24,000  seeds=16x8  win=53%  mean_rew=3.202±2.954  V=3.018  gap=-0.184  outcomes={'other': 41, 'box_possession': 68, 'timeout': 1, 'miss': 18}
2026-08-07 19:49:15,925 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 19:49:28,254 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 19:49:28,256 INFO   [eval vs rules] step=24,000  seeds=16x8  win=27%  mean_rew=0.353±3.567  V=2.766  gap=+2.412  outcomes={'other': 29, 'box_possession': 34, 'opponent_box_possession': 55, 'timeout': 1, 'miss': 9}
2026-08-07 19:50:00,775 INFO   [early stop e0 mb0]  KL=0.48049 > target=0.12  steps_this_update=1
    [per-head KL] exec_move=+0.0258  sprint=+0.0171  kick=+0.0061  move_dir=+0.3233  kick_dir=+0.1072
2026-08-07 19:50:00,778 INFO   [KL mean=0.4805 median=0.4805 > 0.05] ratio percentiles:  p5=0.098  p25=0.844  p50=0.990  p75=1.000  p95=1.092  max=8.478
  move_dir_log_std=[-2.7999627590179443]  kick_dir_log_std=[-2.7999560832977295]
2026-08-07 19:50:00,792 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.127  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.093  kick=-0.192  t_att=-0.174
    move_dir=2.844 (min=-1.845 max=3.762)  kick_dir=0.133 (min=-2.307 max=3.762)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.27
  [worst sample] idx=218  ratio=28.311  adv=+1.341  old_lp=-3.436  new_lp=-0.093
    stored move_dir=-20.9°  new_mean=-17.4°  angular_diff=3.5°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 218  ratio=  28.311  adv=+1.341  lp: old=-3.436  new=-0.093
      rew=+0.0000  ret=+5.0424  val=+3.7019  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9647  sprint_p_new=0.9635  kick_p_new=0.0372  tackle_attempt_p_new=0.0340
    idx= 105  ratio=  27.729  adv=+0.118  lp: old=-3.416  new=-0.093
      rew=+0.0000  ret=-0.0193  val=-0.1370  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9639  sprint_p_new=0.0234  kick_p_new=0.0352  tackle_attempt_p_new=0.0349
  [best sample (highest new_lp)] idx=165  new_lp=0.085  adv=-0.358  stored move_dir=25.7°  new_mean=25.2°
    per-head contributions: move_dir:0.188  move:-0.022  tackle_attempt:-0.024  kick:-0.039
2026-08-07 19:50:00,792 INFO   [advantage] mean=-0.015  std=1.016  min=-6.229  max=4.646
2026-08-07 19:50:00,793 INFO   [ratio] mean=0.8765  std=0.3696  min=0.0000  max=8.4782  clipped=29.1%
2026-08-07 19:50:00,793 INFO   [exec head grad norm] move_direction=0.044  exec_move=0.074  sprint=0.063  kick=0.074  kick_direction=0.012  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.052
2026-08-07 19:50:00,793 INFO   [exec continuous log_std] move_direction: start=-2.8000 end=-2.8000   kick_direction: start=-2.8000 end=-2.8000
2026-08-07 19:50:00,793 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0005≈0.03°/step  epoch≈0.0°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0005≈0.03°/step  epoch≈0.0°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-07 19:50:00,794 INFO   [exec discrete Δlogit per opt step] exec_move=0.0012  sprint=0.0016  kick=0.0006  tackle_attempt=0.0013
2026-08-07 19:50:00,794 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0002  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0258  sprint=+0.0171  kick=+0.0061  tackle_attempt=+0.0008  move_dir=+0.3233  kick_dir=+0.1072
2026-08-07 19:50:00,794 INFO   [grad clip] main: 1/1 steps clipped (100%)  pre-clip norm mean=0.715 max=0.715  limit=0.4
              direction: 1/1 steps clipped (100%)  pre-clip norm mean=0.059 max=0.059  limit=0.02
2026-08-07 19:50:00,823 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=48,000  speed=773/s  reward=3.92
  loss     policy=0.0686  value=0.5099(x0.5)=0.2549
           entropy=1.3252  kl=0.4805
  value    V=3.15±1.68  R=3.22±1.87  adv=0.08±1.29
  moves    mv_ls=[-2.8000] (σ≈0.06, ≈3°) g=3.64e-02  d_move=[+0.0000] (Δσ≈0.000°)
           kk_ls=[-2.8000] (σ≈0.06, ≈3°)  d_kick=[+0.0000] (Δσ≈0.000°)
  heads    move= 41 get_poss= 59 exec_move= 91 sprint= 47 kick=  4 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0404 kick_prob=0.0431
  vs       vs[win/loss/tout/miss]  vs_immobile(523): 65.8%/0.2%/0.4%/11.3%/22%
  ep_len   13.7±8.5s  (n=523, min=1.1s, max=50.0s)
  reward   get_possession=+422.00  lose_possession=-1.80  ball_out=-30.00  box_possession=+860.00
           speed_bonus=+842.77  opponent_box=-3.00  timeout=-3.00  stamina_penalty=-2.99
  rew/ep   (mean/std/min/max per episode, 523 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.807    0.404    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.003    0.056    -0.900    +0.000
  ball_out          -0.057    0.532    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.644    1.186    +0.000    +2.500
  speed_bonus       +1.611    1.514    +0.000    +4.305
  opponent_box      -0.006    0.131    -3.000    +0.000
  timeout           -0.006    0.093    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.028    +0.000
  rew/step (per-step stats, n=24000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     424    +0.018    0.132     +4.235     1.465     +0.361      2.5346      1.164     3.509
  lose_possession       2    -0.000    0.008     +2.782     1.017     -0.959      2.3147      1.181     2.044
  ball_out             6    -0.001    0.079     -4.667     0.471     -6.401     43.8101      6.401     8.281
  box_possession     344    +0.036    0.297     +4.947     1.184     +0.523      1.5923      0.986     2.465
  speed_bonus        335    +0.035    0.324     +5.010     1.134     +0.564      1.5293      0.967     2.437
  opponent_box         1    -0.000    0.019     -3.001     0.000     -2.536      6.4308      2.536     2.536
  timeout              2    -0.000    0.014     -1.500     0.000     -1.414      1.9990      1.414     1.436
  stamina_penalty     326    -0.000    0.001     +4.991     1.235     +0.515      1.4987      0.961     2.402
  gae/td   mean_return=+3.225  std_return=1.866  mean_gae=+0.076  mean_sq_td=1.6605
──────────────────────────────────────────────────────────────────────
2026-08-07 19:50:00,848 INFO Saved checkpoint: checkpoints/phase1_run38/checkpoint2.pt
2026-08-07 19:50:00,848 INFO Logging to checkpoints/phase1_run38/training_log3.txt
2026-08-07 19:50:00,849 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 19:50:11,920 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 19:50:11,922 INFO   [eval vs immobile] step=48,000  seeds=16x8  win=52%  mean_rew=3.260±2.926  V=3.098  gap=-0.161  outcomes={'other': 39, 'miss': 22, 'box_possession': 67}
2026-08-07 19:50:11,923 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 19:50:23,930 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 19:50:23,931 INFO   [eval vs rules] step=48,000  seeds=16x8  win=24%  mean_rew=0.370±3.442  V=2.838  gap=+2.468  outcomes={'other': 37, 'box_possession': 31, 'opponent_box_possession': 52, 'miss': 8}
2026-08-07 19:51:01,621 INFO   [early stop e0 mb0]  KL=0.52516 > target=0.12  steps_this_update=1
    [per-head KL] exec_move=+0.0293  sprint=+0.0171  kick=+0.0050  move_dir=+0.3710  kick_dir=+0.1022
2026-08-07 19:51:01,624 INFO   [KL mean=0.5252 median=0.5252 > 0.05] ratio percentiles:  p5=0.097  p25=0.847  p50=0.989  p75=1.000  p95=1.103  max=16.602
  move_dir_log_std=[-2.7999424934387207]  kick_dir_log_std=[-2.7999367713928223]
2026-08-07 19:51:01,640 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.081  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.169  kick=-0.141  t_att=-0.102
    move_dir=2.934 (min=0.000 max=3.762)  kick_dir=0.108 (min=0.000 max=3.747)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.25
  [worst sample] idx=13  ratio=32.043  adv=+0.049  old_lp=-3.567  new_lp=-0.100
    stored move_dir=171.5°  new_mean=171.0°  angular_diff=0.4°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx=  13  ratio=  32.043  adv=+0.049  lp: old=-3.567  new=-0.100
      rew=+0.0000  ret=-0.0205  val=-0.0692  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9688  sprint_p_new=0.0243  kick_p_new=0.0373  tackle_attempt_p_new=0.0389
    idx=  16  ratio=  29.821  adv=+0.059  lp: old=-3.491  new=-0.096
      rew=+0.0000  ret=-0.0158  val=-0.0751  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9664  sprint_p_new=0.0244  kick_p_new=0.0375  tackle_attempt_p_new=0.0353
  [best sample (highest new_lp)] idx=202  new_lp=0.077  adv=-0.427  stored move_dir=-141.3°  new_mean=-141.9°
    per-head contributions: move_dir:0.187  move:-0.022  tackle_attempt:-0.031  kick:-0.045
2026-08-07 19:51:01,640 INFO   [advantage] mean=0.012  std=0.988  min=-6.033  max=4.688
2026-08-07 19:51:01,641 INFO   [ratio] mean=0.8822  std=0.4622  min=0.0000  max=16.6016  clipped=29.0%
2026-08-07 19:51:01,641 INFO   [exec head grad norm] move_direction=0.028  exec_move=0.075  sprint=0.082  kick=0.050  kick_direction=0.007  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.105
2026-08-07 19:51:01,641 INFO   [exec continuous log_std] move_direction: start=-2.8000 end=-2.7999   kick_direction: start=-2.8000 end=-2.7999
2026-08-07 19:51:01,641 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0007≈0.04°/step  epoch≈0.0°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0007≈0.04°/step  epoch≈0.0°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-07 19:51:01,641 INFO   [exec discrete Δlogit per opt step] exec_move=0.0017  sprint=0.0021  kick=0.0006  tackle_attempt=0.0014
2026-08-07 19:51:01,641 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0005  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0293  sprint=+0.0171  kick=+0.0050  tackle_attempt=+0.0001  move_dir=+0.3710  kick_dir=+0.1022
2026-08-07 19:51:01,642 INFO   [grad clip] main: 1/1 steps clipped (100%)  pre-clip norm mean=0.515 max=0.515  limit=0.4
              direction: 1/1 steps clipped (100%)  pre-clip norm mean=0.044 max=0.044  limit=0.02
2026-08-07 19:51:01,647 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=72,000  speed=667/s  reward=4.04
  loss     policy=0.0343  value=0.4881(x0.5)=0.2441
           entropy=1.3249  kl=0.5252
  value    V=3.33±1.65  R=3.30±1.91  adv=-0.04±1.33
  moves    mv_ls=[-2.7999] (σ≈0.06, ≈3°) g=3.36e-02  d_move=[+0.0000] (Δσ≈0.000°)
           kk_ls=[-2.7999] (σ≈0.06, ≈3°)  d_kick=[+0.0000] (Δσ≈0.000°)
  heads    move= 42 get_poss= 58 exec_move= 91 sprint= 49 kick=  4 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0391 kick_prob=0.0428
  vs       vs[win/loss/tout/miss]  vs_immobile(533): 65.7%/0.4%/0.0%/14.4%/20%
  ep_len   13.3±8.1s  (n=533, min=1.2s, max=49.7s)
  reward   get_possession=+440.00  lose_possession=-1.80  ball_out=-55.00  box_possession=+875.00
           speed_bonus=+855.46  opponent_box=-6.00  stamina_penalty=-3.05
  rew/ep   (mean/std/min/max per episode, 533 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.826    0.384    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.003    0.055    -0.900    +0.000
  ball_out          -0.103    0.711    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.642    1.187    +0.000    +2.500
  speed_bonus       +1.605    1.547    +0.000    +4.357
  opponent_box      -0.011    0.183    -3.000    +0.000
  timeout           +0.000    0.000    +0.000    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.034    +0.000
  rew/step (per-step stats, n=24000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     446    +0.019    0.135     +4.259     1.714     +0.252      2.7336      1.143     3.511
  lose_possession       2    -0.000    0.008     +1.076     1.948     -0.614      0.5839      0.614     1.023
  ball_out            11    -0.002    0.107     -4.727     0.445     -7.370     57.4188      7.370     9.249
  box_possession     350    +0.036    0.300     +4.938     1.260     +0.375      1.4915      0.919     2.109
  speed_bonus        338    +0.036    0.330     +5.025     1.194     +0.429      1.4868      0.911     2.140
  opponent_box         2    -0.000    0.027     -3.005     0.000     -4.871     27.6567      4.871     6.655
  stamina_penalty     336    -0.000    0.001     +4.928     1.392     +0.334      1.5493      0.927     2.146
  gae/td   mean_return=+3.295  std_return=1.905  mean_gae=-0.038  mean_sq_td=1.7571
──────────────────────────────────────────────────────────────────────
2026-08-07 19:51:01,676 INFO Saved checkpoint: checkpoints/phase1_run38/checkpoint3.pt
2026-08-07 19:51:01,676 INFO Logging to checkpoints/phase1_run38/training_log4.txt
2026-08-07 19:51:01,677 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 19:51:14,400 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 19:51:14,401 INFO   [eval vs immobile] step=72,000  seeds=16x8  win=56%  mean_rew=3.422±2.918  V=3.233  gap=-0.188  outcomes={'other': 38, 'box_possession': 72, 'miss': 18}
2026-08-07 19:51:14,403 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 19:51:28,524 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 19:51:28,526 INFO   [eval vs rules] step=72,000  seeds=16x8  win=27%  mean_rew=0.603±3.460  V=2.857  gap=+2.254  outcomes={'other': 37, 'box_possession': 34, 'opponent_box_possession': 48, 'miss': 9}
2026-08-07 19:52:07,343 INFO   [early stop e0 mb0]  KL=0.45452 > target=0.12  steps_this_update=1
    [per-head KL] exec_move=+0.0263  sprint=+0.0173  kick=+0.0042  move_dir=+0.2777  kick_dir=+0.1288
2026-08-07 19:52:07,346 INFO   [KL mean=0.4545 median=0.4545 > 0.05] ratio percentiles:  p5=0.136  p25=0.846  p50=0.989  p75=1.000  p95=1.084  max=7.048
  move_dir_log_std=[-2.799922466278076]  kick_dir_log_std=[-2.7999191284179688]
2026-08-07 19:52:07,361 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.066  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.157  kick=-0.219  t_att=-0.154
    move_dir=2.914 (min=0.000 max=3.762)  kick_dir=0.184 (min=0.000 max=3.731)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.22
  [worst sample] idx=220  ratio=28.220  adv=+0.363  old_lp=-6.508  new_lp=-3.168
    stored move_dir=17.6°  new_mean=22.5°  angular_diff=4.9°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 220  ratio=  28.220  adv=+0.363  lp: old=-6.508  new=-3.168
      rew=+0.0000  ret=+4.4104  val=+4.0474  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9646  sprint_p_new=0.9645  kick_p_new=0.0380  tackle_attempt_p_new=0.0352
    idx= 164  ratio=  27.964  adv=-1.822  lp: old=-3.424  new=-0.093
      rew=+0.0000  ret=+2.4027  val=+4.2251  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9643  sprint_p_new=0.9657  kick_p_new=0.0356  tackle_attempt_p_new=0.0407
  [best sample (highest new_lp)] idx=67  new_lp=0.097  adv=+0.002  stored move_dir=129.7°  new_mean=129.2°
    per-head contributions: move_dir:0.188  move:-0.021  kick:-0.038
2026-08-07 19:52:07,362 INFO   [advantage] mean=-0.017  std=1.005  min=-7.730  max=3.841
2026-08-07 19:52:07,362 INFO   [ratio] mean=0.8785  std=0.3390  min=0.0000  max=7.0479  clipped=28.3%
2026-08-07 19:52:07,362 INFO   [exec head grad norm] move_direction=0.074  exec_move=0.031  sprint=0.089  kick=0.130  kick_direction=0.008  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.074
2026-08-07 19:52:07,362 INFO   [exec continuous log_std] move_direction: start=-2.7999 end=-2.7999   kick_direction: start=-2.7999 end=-2.7999
2026-08-07 19:52:07,362 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0007≈0.04°/step  epoch≈0.0°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0007≈0.04°/step  epoch≈0.0°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-07 19:52:07,362 INFO   [exec discrete Δlogit per opt step] exec_move=0.0014  sprint=0.0019  kick=0.0006  tackle_attempt=0.0014
2026-08-07 19:52:07,362 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0002  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0263  sprint=+0.0173  kick=+0.0042  tackle_attempt=+0.0001  move_dir=+0.2777  kick_dir=+0.1288
2026-08-07 19:52:07,363 INFO   [grad clip] main: 1/1 steps clipped (100%)  pre-clip norm mean=0.453 max=0.453  limit=0.4
              direction: 1/1 steps clipped (100%)  pre-clip norm mean=0.082 max=0.082  limit=0.02
2026-08-07 19:52:07,368 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=96,000  speed=651/s  reward=4.44
  loss     policy=0.0620  value=0.5025(x0.5)=0.2513
           entropy=1.3224  kl=0.4545
  value    V=3.45±1.63  R=3.34±1.86  adv=-0.11±1.30
  moves    mv_ls=[-2.7999] (σ≈0.06, ≈3°) g=3.44e-02  d_move=[+0.0000] (Δσ≈0.000°)
           kk_ls=[-2.7999] (σ≈0.06, ≈3°)  d_kick=[+0.0000] (Δσ≈0.000°)
  heads    move= 41 get_poss= 59 exec_move= 91 sprint= 46 kick=  5 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0396 kick_prob=0.0434
  vs       vs[win/loss/tout/miss]  vs_immobile(519): 65.5%/0.2%/0.4%/9.2%/25%
  ep_len   13.8±8.5s  (n=519, min=0.9s, max=50.0s)
  reward   get_possession=+435.00  lose_possession=-3.60  ball_out=-25.00  box_possession=+850.00
           speed_bonus=+822.10  opponent_box=-3.00  timeout=-3.00  stamina_penalty=-2.78
  rew/ep   (mean/std/min/max per episode, 519 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.838    0.389    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.007    0.079    -0.900    +0.000
  ball_out          -0.048    0.488    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.638    1.188    +0.000    +2.500
  speed_bonus       +1.584    1.532    +0.000    +4.274
  opponent_box      -0.006    0.132    -3.000    +0.000
  timeout           -0.006    0.093    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.005    0.006    -0.031    +0.000
  rew/step (per-step stats, n=24000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     437    +0.018    0.134     +4.355     1.383     +0.195      2.1974      1.071     3.346
  lose_possession       4    -0.000    0.012     +2.682     1.508     -1.270      3.0745      1.356     2.866
  ball_out             5    -0.001    0.072     -4.800     0.400     -6.231     44.1746      6.231     9.443
  box_possession     340    +0.035    0.295     +4.916     1.242     +0.422      1.2227      0.901     2.056
  speed_bonus        329    +0.034    0.322     +4.997     1.180     +0.477      1.2050      0.890     2.063
  opponent_box         1    -0.000    0.019     -3.006     0.000     -5.175     26.7779      5.175     5.175
  timeout              2    -0.000    0.014     -1.503     0.003     -0.812      0.7603      0.812     1.098
  stamina_penalty     321    -0.000    0.001     +4.972     1.317     +0.426      1.2480      0.892     2.010
  gae/td   mean_return=+3.336  std_return=1.863  mean_gae=-0.113  mean_sq_td=1.6983
──────────────────────────────────────────────────────────────────────
2026-08-07 19:52:07,400 INFO Saved checkpoint: checkpoints/phase1_run38/checkpoint4.pt
2026-08-07 19:52:07,400 INFO Logging to checkpoints/phase1_run38/training_log5.txt
2026-08-07 19:52:07,401 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 19:52:18,982 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 19:52:18,983 INFO   [eval vs immobile] step=96,000  seeds=16x8  win=58%  mean_rew=3.501±2.925  V=3.253  gap=-0.248  outcomes={'other': 36, 'box_possession': 74, 'miss': 18}
2026-08-07 19:52:18,985 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 19:52:30,013 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 19:52:30,014 INFO   [eval vs rules] step=96,000  seeds=16x8  win=27%  mean_rew=0.457±3.433  V=2.855  gap=+2.398  outcomes={'other': 35, 'box_possession': 34, 'opponent_box_possession': 51, 'miss': 8}
2026-08-07 19:53:01,611 INFO   [early stop e0 mb0]  KL=0.48854 > target=0.12  steps_this_update=1
    [per-head KL] exec_move=+0.0264  sprint=+0.0177  kick=+0.0043  move_dir=+0.3288  kick_dir=+0.1098
2026-08-07 19:53:01,613 INFO   [KL mean=0.4885 median=0.4885 > 0.05] ratio percentiles:  p5=0.117  p25=0.855  p50=0.992  p75=1.000  p95=1.084  max=15.678
  move_dir_log_std=[-2.7999019622802734]  kick_dir_log_std=[-2.79990291595459]
2026-08-07 19:53:01,625 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.050  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.163  kick=-0.173  t_att=-0.151
    move_dir=2.930 (min=0.000 max=3.762)  kick_dir=0.138 (min=0.000 max=3.762)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.20
  [worst sample] idx=21  ratio=26.736  adv=+0.150  old_lp=-3.380  new_lp=-0.094
    stored move_dir=-157.0°  new_mean=-162.1°  angular_diff=5.2°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx=  21  ratio=  26.736  adv=+0.150  lp: old=-3.380  new=-0.094
      rew=+0.0000  ret=+4.3263  val=+4.1768  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9626  sprint_p_new=0.9620  kick_p_new=0.0390  tackle_attempt_p_new=0.0333
    idx=  25  ratio=  26.104  adv=+0.015  lp: old=-3.356  new=-0.094
      rew=+0.0000  ret=+4.4213  val=+4.4061  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9617  sprint_p_new=0.9611  kick_p_new=0.0387  tackle_attempt_p_new=0.0324
  [best sample (highest new_lp)] idx=136  new_lp=0.071  adv=+0.109  stored move_dir=151.7°  new_mean=151.5°
    per-head contributions: move_dir:0.188  move:-0.022  sprint:-0.023  tackle_attempt:-0.034  kick:-0.038
2026-08-07 19:53:01,626 INFO   [advantage] mean=0.002  std=0.997  min=-6.234  max=4.486
2026-08-07 19:53:01,626 INFO   [ratio] mean=0.8838  std=0.4009  min=0.0000  max=15.6784  clipped=27.9%
2026-08-07 19:53:01,626 INFO   [exec head grad norm] move_direction=0.047  exec_move=0.109  sprint=0.112  kick=0.115  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.066
2026-08-07 19:53:01,626 INFO   [exec continuous log_std] move_direction: start=-2.7999 end=-2.7999   kick_direction: start=-2.7999 end=-2.7999
2026-08-07 19:53:01,626 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0008≈0.05°/step  epoch≈0.0°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0009≈0.05°/step  epoch≈0.1°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-07 19:53:01,626 INFO   [exec discrete Δlogit per opt step] exec_move=0.0018  sprint=0.0024  kick=0.0006  tackle_attempt=0.0013
2026-08-07 19:53:01,627 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0007  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0264  sprint=+0.0177  kick=+0.0043  tackle_attempt=+0.0008  move_dir=+0.3288  kick_dir=+0.1098
2026-08-07 19:53:01,627 INFO   [grad clip] main: 1/1 steps clipped (100%)  pre-clip norm mean=0.623 max=0.623  limit=0.4
              direction: 1/1 steps clipped (100%)  pre-clip norm mean=0.060 max=0.060  limit=0.02
2026-08-07 19:53:01,631 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=120,000  speed=795/s  reward=3.70
  loss     policy=0.0453  value=0.4682(x0.5)=0.2341
           entropy=1.3240  kl=0.4885
  value    V=3.48±1.50  R=3.42±1.84  adv=-0.06±1.25
  moves    mv_ls=[-2.7999] (σ≈0.06, ≈3°) g=3.58e-02  d_move=[+0.0000] (Δσ≈0.000°)
           kk_ls=[-2.7999] (σ≈0.06, ≈3°)  d_kick=[+0.0000] (Δσ≈0.000°)
  heads    move= 43 get_poss= 57 exec_move= 91 sprint= 49 kick=  4 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0390 kick_prob=0.0425
  vs       vs[win/loss/tout/miss]  vs_immobile(515): 68.0%/0.0%/1.0%/9.5%/22%
  ep_len   13.8±8.2s  (n=515, min=1.0s, max=50.0s)
  reward   get_possession=+447.00  lose_possession=-3.60  ball_out=-30.00  box_possession=+875.00
           speed_bonus=+881.11  timeout=-7.50  stamina_penalty=-3.11
  rew/ep   (mean/std/min/max per episode, 515 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.868    0.361    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.007    0.079    -0.900    +0.000
  ball_out          -0.058    0.537    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.699    1.167    +0.000    +2.500
  speed_bonus       +1.711    1.547    +0.000    +4.284
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.015    0.147    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.031    +0.000
  rew/step (per-step stats, n=24000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     452    +0.019    0.136     +4.339     1.435     +0.346      2.3537      1.090     3.033
  lose_possession       4    -0.000    0.012     +2.984     0.569     -0.762      0.9649      0.822     1.404
  ball_out             6    -0.001    0.079     -4.333     0.471     -5.796     39.2333      5.796     8.409
  box_possession     350    +0.036    0.300     +5.011     1.213     +0.492      1.1274      0.876     1.945
  speed_bonus        341    +0.037    0.336     +5.078     1.157     +0.540      1.1018      0.864     1.852
  timeout              5    -0.000    0.022     -1.501     0.001     -0.689      1.4885      1.068     1.874
  stamina_penalty     330    -0.000    0.001     +5.060     1.230     +0.530      1.1342      0.882     1.921
  gae/td   mean_return=+3.418  std_return=1.841  mean_gae=-0.059  mean_sq_td=1.5689
──────────────────────────────────────────────────────────────────────
2026-08-07 19:53:01,661 INFO Saved checkpoint: checkpoints/phase1_run38/checkpoint5.pt
2026-08-07 19:53:01,662 INFO Logging to checkpoints/phase1_run38/training_log6.txt
2026-08-07 19:53:01,663 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 19:53:12,415 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 19:53:12,417 INFO   [eval vs immobile] step=120,000  seeds=16x8  win=58%  mean_rew=3.493±2.913  V=3.186  gap=-0.307  outcomes={'other': 37, 'box_possession': 74, 'miss': 17}
2026-08-07 19:53:12,418 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 19:53:25,179 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 19:53:25,181 INFO   [eval vs rules] step=120,000  seeds=16x8  win=23%  mean_rew=0.027±3.261  V=2.628  gap=+2.601  outcomes={'other': 31, 'box_possession': 30, 'opponent_box_possession': 57, 'timeout': 1, 'miss': 9}
2026-08-07 19:53:56,802 INFO   [early stop e0 mb0]  KL=0.48254 > target=0.12  steps_this_update=1
    [per-head KL] exec_move=+0.0231  sprint=+0.0162  kick=+0.0041  move_dir=+0.3435  kick_dir=+0.0947
2026-08-07 19:53:56,804 INFO   [KL mean=0.4825 median=0.4825 > 0.05] ratio percentiles:  p5=0.117  p25=0.868  p50=0.994  p75=1.000  p95=1.086  max=15.310
  move_dir_log_std=[-2.799882411956787]  kick_dir_log_std=[-2.7998881340026855]
2026-08-07 19:53:56,818 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.081  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.128  kick=-0.150  t_att=-0.174
    move_dir=2.901 (min=-0.486 max=3.762)  kick_dir=0.136 (min=-1.112 max=3.761)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.25
  [worst sample] idx=201  ratio=30.215  adv=+0.307  old_lp=-3.507  new_lp=-0.098
    stored move_dir=-135.1°  new_mean=-134.5°  angular_diff=0.5°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 201  ratio=  30.215  adv=+0.307  lp: old=-3.507  new=-0.098
      rew=+0.0000  ret=+4.8367  val=+4.5299  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9669  sprint_p_new=0.0351  kick_p_new=0.0378  tackle_attempt_p_new=0.0379
    idx= 249  ratio=  26.651  adv=+0.499  lp: old=-3.379  new=-0.096
      rew=+0.0000  ret=+5.2799  val=+4.7806  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9625  sprint_p_new=0.0264  kick_p_new=0.0368  tackle_attempt_p_new=0.0371
  [best sample (highest new_lp)] idx=58  new_lp=0.074  adv=-0.726  stored move_dir=90.1°  new_mean=90.9°
    per-head contributions: move_dir:0.187  sprint:-0.021  move:-0.021  tackle_attempt:-0.030  kick:-0.041
2026-08-07 19:53:56,818 INFO   [advantage] mean=0.010  std=0.987  min=-6.378  max=3.984
2026-08-07 19:53:56,818 INFO   [ratio] mean=0.8916  std=0.4302  min=0.0000  max=15.3102  clipped=26.9%
2026-08-07 19:53:56,819 INFO   [exec head grad norm] move_direction=0.127  exec_move=0.084  sprint=0.039  kick=0.094  kick_direction=0.005  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.114
2026-08-07 19:53:56,819 INFO   [exec continuous log_std] move_direction: start=-2.7999 end=-2.7999   kick_direction: start=-2.7999 end=-2.7999
2026-08-07 19:53:56,819 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0010≈0.06°/step  epoch≈0.1°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0010≈0.06°/step  epoch≈0.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-07 19:53:56,819 INFO   [exec discrete Δlogit per opt step] exec_move=0.0020  sprint=0.0024  kick=0.0006  tackle_attempt=0.0014
2026-08-07 19:53:56,819 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0003  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0231  sprint=+0.0162  kick=+0.0041  tackle_attempt=+0.0007  move_dir=+0.3435  kick_dir=+0.0947
2026-08-07 19:53:56,820 INFO   [grad clip] main: 1/1 steps clipped (100%)  pre-clip norm mean=0.760 max=0.760  limit=0.4
              direction: 1/1 steps clipped (100%)  pre-clip norm mean=0.132 max=0.132  limit=0.02
2026-08-07 19:53:56,825 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=144,000  speed=794/s  reward=4.46
  loss     policy=0.0530  value=0.4470(x0.5)=0.2235
           entropy=1.3201  kl=0.4825
  value    V=3.37±1.39  R=3.34±1.78  adv=-0.03±1.18
  moves    mv_ls=[-2.7999] (σ≈0.06, ≈3°) g=3.30e-02  d_move=[+0.0000] (Δσ≈0.000°)
           kk_ls=[-2.7999] (σ≈0.06, ≈3°)  d_kick=[+0.0000] (Δσ≈0.000°)
  heads    move= 42 get_poss= 58 exec_move= 91 sprint= 48 kick=  4 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0390 kick_prob=0.0418
  vs       vs[win/loss/tout/miss]  vs_immobile(552): 66.3%/0.0%/0.2%/12.0%/22%
  ep_len   13.0±7.9s  (n=552, min=1.1s, max=50.0s)
  reward   get_possession=+450.00  lose_possession=-2.70  ball_out=-25.00  box_possession=+915.00
           speed_bonus=+873.01  timeout=-1.50  stamina_penalty=-2.98
  rew/ep   (mean/std/min/max per episode, 552 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.815    0.402    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.005    0.066    -0.900    +0.000
  ball_out          -0.045    0.474    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.658    1.182    +0.000    +2.500
  speed_bonus       +1.582    1.509    +0.000    +4.316
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.003    0.064    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.005    0.006    -0.027    +0.000
  rew/step (per-step stats, n=24000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     451    +0.019    0.136     +4.338     1.332     +0.331      1.8775      0.952     2.728
  lose_possession       3    -0.000    0.010     +3.345     0.576     -0.275      0.2130      0.397     0.680
  ball_out             5    -0.001    0.072     -4.800     0.400     -7.670     59.2365      7.670     8.458
  box_possession     366    +0.038    0.306     +4.883     1.221     +0.535      1.1178      0.895     1.852
  speed_bonus        355    +0.036    0.329     +4.954     1.169     +0.580      1.1063      0.892     1.817
  timeout              1    -0.000    0.010     -1.500     0.000     -1.273      1.6196      1.273     1.273
  stamina_penalty     347    -0.000    0.001     +4.947     1.192     +0.575      1.1258      0.901     1.852
  gae/td   mean_return=+3.343  std_return=1.783  mean_gae=-0.026  mean_sq_td=1.4027
──────────────────────────────────────────────────────────────────────
2026-08-07 19:53:56,850 INFO Saved checkpoint: checkpoints/phase1_run38/checkpoint6.pt
2026-08-07 19:53:56,850 INFO Logging to checkpoints/phase1_run38/training_log7.txt
2026-08-07 19:53:56,852 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 19:54:08,206 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 19:54:08,208 INFO   [eval vs immobile] step=144,000  seeds=16x8  win=52%  mean_rew=3.189±2.904  V=2.940  gap=-0.250  outcomes={'other': 42, 'box_possession': 66, 'miss': 20}
2026-08-07 19:54:08,209 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 19:54:19,500 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 19:54:19,501 INFO   [eval vs rules] step=144,000  seeds=16x8  win=23%  mean_rew=0.008±3.247  V=2.586  gap=+2.577  outcomes={'box_possession': 29, 'other': 32, 'opponent_box_possession': 59, 'miss': 8}
2026-08-07 19:54:52,016 INFO   [early stop e0 mb0]  KL=0.45688 > target=0.12  steps_this_update=1
    [per-head KL] exec_move=+0.0274  sprint=+0.0144  kick=+0.0023  move_dir=+0.3158  kick_dir=+0.0965
2026-08-07 19:54:52,018 INFO   [KL mean=0.4569 median=0.4569 > 0.05] ratio percentiles:  p5=0.133  p25=0.851  p50=0.991  p75=1.000  p95=1.090  max=9.277
  move_dir_log_std=[-2.7998626232147217]  kick_dir_log_std=[-2.7998743057250977]
2026-08-07 19:54:52,033 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.050  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.188  kick=-0.200  t_att=-0.116
    move_dir=3.000 (min=-0.344 max=3.762)  kick_dir=0.172 (min=0.000 max=3.749)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.18
  [worst sample] idx=158  ratio=28.284  adv=-0.043  old_lp=-6.568  new_lp=-3.226
    stored move_dir=-179.2°  new_mean=-179.8°  angular_diff=0.5°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 158  ratio=  28.284  adv=-0.043  lp: old=-6.568  new=-3.226
      rew=+0.0000  ret=+4.0149  val=+4.0576  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9647  sprint_p_new=0.9654  kick_p_new=0.0359  tackle_attempt_p_new=0.0355
    idx= 109  ratio=  23.709  adv=-0.576  lp: old=-3.266  new=-0.100
      rew=+0.0000  ret=+4.9570  val=+5.5332  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9578  sprint_p_new=0.0353  kick_p_new=0.0382  tackle_attempt_p_new=0.0401
  [best sample (highest new_lp)] idx=2  new_lp=0.076  adv=-0.070  stored move_dir=61.3°  new_mean=63.6°
    per-head contributions: move_dir:0.177  move:-0.020  tackle_attempt:-0.029  kick:-0.033
2026-08-07 19:54:52,033 INFO   [advantage] mean=0.010  std=1.012  min=-5.109  max=3.846
2026-08-07 19:54:52,033 INFO   [ratio] mean=0.8796  std=0.3714  min=0.0000  max=9.2770  clipped=28.2%
2026-08-07 19:54:52,034 INFO   [exec head grad norm] move_direction=0.058  exec_move=0.073  sprint=0.042  kick=0.100  kick_direction=0.032  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.151
2026-08-07 19:54:52,034 INFO   [exec continuous log_std] move_direction: start=-2.7999 end=-2.7999   kick_direction: start=-2.7999 end=-2.7999
2026-08-07 19:54:52,034 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0010≈0.06°/step  epoch≈0.1°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0011≈0.07°/step  epoch≈0.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-07 19:54:52,034 INFO   [exec discrete Δlogit per opt step] exec_move=0.0021  sprint=0.0026  kick=0.0006  tackle_attempt=0.0014
2026-08-07 19:54:52,034 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0001  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0274  sprint=+0.0144  kick=+0.0023  tackle_attempt=+0.0006  move_dir=+0.3158  kick_dir=+0.0965
2026-08-07 19:54:52,035 INFO   [grad clip] main: 1/1 steps clipped (100%)  pre-clip norm mean=2.309 max=2.309  limit=0.4
              direction: 1/1 steps clipped (100%)  pre-clip norm mean=0.078 max=0.078  limit=0.02
2026-08-07 19:54:52,040 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=168,000  speed=771/s  reward=3.35
  loss     policy=0.0479  value=0.5072(x0.5)=0.2536
           entropy=1.3278  kl=0.4569
  value    V=3.21±1.36  R=3.25±1.81  adv=0.04±1.26
  moves    mv_ls=[-2.7999] (σ≈0.06, ≈3°) g=4.06e-02  d_move=[+0.0000] (Δσ≈0.000°)
           kk_ls=[-2.7999] (σ≈0.06, ≈3°)  d_kick=[+0.0000] (Δσ≈0.000°)
  heads    move= 42 get_poss= 58 exec_move= 91 sprint= 47 kick=  4 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0403 kick_prob=0.0429
  vs       vs[win/loss/tout/miss]  vs_immobile(551): 63.3%/0.2%/0.0%/12.5%/24%
  ep_len   12.9±7.8s  (n=551, min=0.4s, max=47.1s)
  reward   get_possession=+448.00  lose_possession=-3.60  ball_out=-25.00  box_possession=+872.50
           speed_bonus=+871.67  opponent_box=-3.00  stamina_penalty=-2.96
  rew/ep   (mean/std/min/max per episode, 551 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.813    0.408    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.007    0.076    -0.900    +0.000
  ball_out          -0.045    0.474    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.583    1.205    +0.000    +2.500
  speed_bonus       +1.582    1.553    +0.000    +4.279
  opponent_box      -0.005    0.128    -3.000    +0.000
  timeout           +0.000    0.000    +0.000    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.005    0.006    -0.039    +0.000
  rew/step (per-step stats, n=24000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     449    +0.019    0.135     +4.243     1.258     +0.501      2.0373      1.092     2.947
  lose_possession       4    -0.000    0.012     +2.957     0.329     -0.419      0.3254      0.419     0.936
  ball_out             5    -0.001    0.072     -5.000     0.000     -5.133     29.4293      5.133     7.737
  box_possession     349    +0.036    0.299     +4.992     1.226     +0.712      1.3563      0.989     2.138
  speed_bonus        336    +0.036    0.334     +5.086     1.150     +0.775      1.3593      0.989     2.151
  opponent_box         1    -0.000    0.019     -3.001     0.000     -3.131      9.8048      3.131     3.131
  stamina_penalty     325    -0.000    0.001     +5.035     1.279     +0.731      1.3257      0.982     2.085
  gae/td   mean_return=+3.247  std_return=1.806  mean_gae=+0.042  mean_sq_td=1.5917
──────────────────────────────────────────────────────────────────────
2026-08-07 19:54:52,064 INFO Saved checkpoint: checkpoints/phase1_run38/checkpoint7.pt
2026-08-07 19:54:52,064 INFO Logging to checkpoints/phase1_run38/training_log8.txt
2026-08-07 19:54:52,065 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 19:55:02,896 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 19:55:02,898 INFO   [eval vs immobile] step=168,000  seeds=16x8  win=55%  mean_rew=3.387±2.919  V=2.918  gap=-0.469  outcomes={'other': 37, 'box_possession': 71, 'miss': 20}
2026-08-07 19:55:02,899 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 19:55:13,482 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 19:55:13,483 INFO   [eval vs rules] step=168,000  seeds=16x8  win=23%  mean_rew=0.040±3.452  V=2.627  gap=+2.587  outcomes={'other': 28, 'opponent_box_possession': 61, 'box_possession': 30, 'miss': 9}
2026-08-07 19:55:45,397 INFO   [early stop e0 mb0]  KL=0.46342 > target=0.12  steps_this_update=1
    [per-head KL] exec_move=+0.0244  sprint=+0.0124  kick=+0.0061  move_dir=+0.3107  kick_dir=+0.1100
2026-08-07 19:55:45,399 INFO   [KL mean=0.4634 median=0.4634 > 0.05] ratio percentiles:  p5=0.114  p25=0.865  p50=0.992  p75=1.000  p95=1.087  max=9.410
  move_dir_log_std=[-2.799842119216919]  kick_dir_log_std=[-2.799861431121826]
2026-08-07 19:55:45,410 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.095  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.213  kick=-0.215  t_att=-0.226
    move_dir=2.958 (min=-0.638 max=3.762)  kick_dir=0.163 (min=0.000 max=3.762)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.18
  [worst sample] idx=84  ratio=25.540  adv=+2.196  old_lp=-6.529  new_lp=-3.289
    stored move_dir=-13.8°  new_mean=-10.6°  angular_diff=3.3°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx=  84  ratio=  25.540  adv=+2.196  lp: old=-6.529  new=-3.289
      rew=+0.0000  ret=-0.1827  val=-2.3783  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9608  sprint_p_new=0.0305  kick_p_new=0.0364  tackle_attempt_p_new=0.0398
    idx= 152  ratio=  25.303  adv=+0.574  lp: old=-3.335  new=-0.104
      rew=+0.0000  ret=+4.8668  val=+4.2933  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9605  sprint_p_new=0.9224  kick_p_new=0.0398  tackle_attempt_p_new=0.0411
  [best sample (highest new_lp)] idx=134  new_lp=0.079  adv=+1.630  stored move_dir=97.4°  new_mean=98.0°
    per-head contributions: move_dir:0.187  move:-0.022  tackle_attempt:-0.023  sprint:-0.028  kick:-0.035
2026-08-07 19:55:45,411 INFO   [advantage] mean=0.006  std=0.996  min=-6.759  max=4.093
2026-08-07 19:55:45,411 INFO   [ratio] mean=0.8838  std=0.3632  min=0.0000  max=9.4103  clipped=26.9%
2026-08-07 19:55:45,411 INFO   [exec head grad norm] move_direction=0.047  exec_move=0.061  sprint=0.064  kick=0.121  kick_direction=0.005  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.096
2026-08-07 19:55:45,411 INFO   [exec continuous log_std] move_direction: start=-2.7999 end=-2.7998   kick_direction: start=-2.7999 end=-2.7999
2026-08-07 19:55:45,411 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0010≈0.06°/step  epoch≈0.1°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0011≈0.06°/step  epoch≈0.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-07 19:55:45,411 INFO   [exec discrete Δlogit per opt step] exec_move=0.0023  sprint=0.0027  kick=0.0006  tackle_attempt=0.0014
2026-08-07 19:55:45,411 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=-0.0000  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0244  sprint=+0.0124  kick=+0.0061  tackle_attempt=-0.0002  move_dir=+0.3107  kick_dir=+0.1100
2026-08-07 19:55:45,412 INFO   [grad clip] main: 1/1 steps clipped (100%)  pre-clip norm mean=0.550 max=0.550  limit=0.4
              direction: 1/1 steps clipped (100%)  pre-clip norm mean=0.060 max=0.060  limit=0.02
2026-08-07 19:55:45,417 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=192,000  speed=787/s  reward=4.72
  loss     policy=0.0415  value=0.5087(x0.5)=0.2544
           entropy=1.3251  kl=0.4634
  value    V=3.12±1.28  R=3.21±1.79  adv=0.09±1.25
  moves    mv_ls=[-2.7998] (σ≈0.06, ≈3°) g=3.71e-02  d_move=[+0.0000] (Δσ≈0.000°)
           kk_ls=[-2.7999] (σ≈0.06, ≈3°)  d_kick=[+0.0000] (Δσ≈0.000°)
  heads    move= 43 get_poss= 57 exec_move= 91 sprint= 47 kick=  4 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0397 kick_prob=0.0447
  vs       vs[win/loss/tout/miss]  vs_immobile(540): 65.6%/0.0%/0.2%/10.6%/24%
  ep_len   13.1±8.0s  (n=540, min=1.1s, max=50.0s)
  reward   get_possession=+457.00  lose_possession=-3.60  ball_out=-25.00  box_possession=+885.00
           speed_bonus=+877.70  timeout=-1.50  stamina_penalty=-2.86
  rew/ep   (mean/std/min/max per episode, 540 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.846    0.376    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.007    0.077    -0.900    +0.000
  ball_out          -0.046    0.479    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.639    1.188    +0.000    +2.500
  speed_bonus       +1.625    1.532    +0.000    +4.331
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.003    0.064    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.005    0.006    -0.030    +0.000
  rew/step (per-step stats, n=24000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     463    +0.019    0.138     +4.198     1.223     +0.605      1.9068      1.099     2.623
  lose_possession       4    -0.000    0.012     +2.479     1.912     +0.168      1.3755      0.939     1.793
  ball_out             5    -0.001    0.072     -5.000     0.000     -5.639     34.0616      5.639     7.890
  box_possession     354    +0.037    0.301     +4.974     1.202     +0.704      1.3547      0.975     1.948
  speed_bonus        342    +0.037    0.333     +5.061     1.128     +0.760      1.3608      0.975     1.977
  timeout              1    -0.000    0.010     -1.500     0.000     -1.666      2.7746      1.666     1.666
  stamina_penalty     335    -0.000    0.001     +5.037     1.170     +0.742      1.3756      0.981     1.982
  gae/td   mean_return=+3.214  std_return=1.787  mean_gae=+0.091  mean_sq_td=1.5713
──────────────────────────────────────────────────────────────────────
2026-08-07 19:55:45,444 INFO Saved checkpoint: checkpoints/phase1_run38/checkpoint8.pt
2026-08-07 19:55:45,444 INFO Logging to checkpoints/phase1_run38/training_log9.txt
2026-08-07 19:55:45,445 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 19:55:55,608 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 19:55:55,610 INFO   [eval vs immobile] step=192,000  seeds=16x8  win=57%  mean_rew=3.496±2.964  V=2.974  gap=-0.522  outcomes={'other': 36, 'box_possession': 73, 'miss': 19}
2026-08-07 19:55:55,611 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 19:56:09,667 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 19:56:09,669 INFO   [eval vs rules] step=192,000  seeds=16x8  win=23%  mean_rew=0.367±3.372  V=2.527  gap=+2.159  outcomes={'other': 38, 'opponent_box_possession': 49, 'box_possession': 30, 'timeout': 2, 'miss': 9}
2026-08-07 19:56:41,121 INFO   [early stop e0 mb0]  KL=0.44842 > target=0.12  steps_this_update=1
    [per-head KL] exec_move=+0.0238  sprint=+0.0165  kick=+0.0035  move_dir=+0.3132  kick_dir=+0.0907
2026-08-07 19:56:41,124 INFO   [KL mean=0.4484 median=0.4484 > 0.05] ratio percentiles:  p5=0.152  p25=0.856  p50=0.989  p75=1.000  p95=1.088  max=9.848
  move_dir_log_std=[-2.799821376800537]  kick_dir_log_std=[-2.799849510192871]
2026-08-07 19:56:41,136 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.142  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.157  kick=-0.223  t_att=-0.179
    move_dir=2.856 (min=0.000 max=3.762)  kick_dir=0.214 (min=0.000 max=3.759)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.34
  [worst sample] idx=57  ratio=29.185  adv=-1.378  old_lp=-3.462  new_lp=-0.088
    stored move_dir=-173.0°  new_mean=-176.9°  angular_diff=3.9°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx=  57  ratio=  29.185  adv=-1.378  lp: old=-3.462  new=-0.088
      rew=+0.0000  ret=+2.2300  val=+3.6084  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9657  sprint_p_new=0.9658  kick_p_new=0.0363  tackle_attempt_p_new=0.0334
    idx=  53  ratio=  25.220  adv=-1.309  lp: old=-3.320  new=-0.093
      rew=+0.0000  ret=+2.3439  val=+3.6532  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9603  sprint_p_new=0.9599  kick_p_new=0.0381  tackle_attempt_p_new=0.0345
  [best sample (highest new_lp)] idx=70  new_lp=0.071  adv=-1.323  stored move_dir=53.8°  new_mean=53.7°
    per-head contributions: move_dir:0.188  move:-0.022  tackle_attempt:-0.033  kick:-0.046
2026-08-07 19:56:41,136 INFO   [advantage] mean=-0.008  std=1.015  min=-7.737  max=3.824
2026-08-07 19:56:41,137 INFO   [ratio] mean=0.8863  std=0.3584  min=0.0000  max=9.8480  clipped=27.9%
2026-08-07 19:56:41,137 INFO   [exec head grad norm] move_direction=0.046  exec_move=0.083  sprint=0.063  kick=0.068  kick_direction=0.016  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.075
2026-08-07 19:56:41,137 INFO   [exec continuous log_std] move_direction: start=-2.7998 end=-2.7998   kick_direction: start=-2.7999 end=-2.7998
2026-08-07 19:56:41,137 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0012≈0.07°/step  epoch≈0.1°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0012≈0.07°/step  epoch≈0.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-07 19:56:41,137 INFO   [exec discrete Δlogit per opt step] exec_move=0.0024  sprint=0.0028  kick=0.0006  tackle_attempt=0.0014
2026-08-07 19:56:41,137 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0000  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0238  sprint=+0.0165  kick=+0.0035  tackle_attempt=+0.0007  move_dir=+0.3132  kick_dir=+0.0907
2026-08-07 19:56:41,138 INFO   [grad clip] main: 1/1 steps clipped (100%)  pre-clip norm mean=0.908 max=0.908  limit=0.4
              direction: 1/1 steps clipped (100%)  pre-clip norm mean=0.059 max=0.059  limit=0.02
2026-08-07 19:56:41,143 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=216,000  speed=802/s  reward=4.18
  loss     policy=0.0555  value=0.5191(x0.5)=0.2595
           entropy=1.3282  kl=0.4484
  value    V=3.08±1.29  R=3.18±1.80  adv=0.11±1.27
  moves    mv_ls=[-2.7998] (σ≈0.06, ≈3°) g=3.30e-02  d_move=[+0.0000] (Δσ≈0.000°)
           kk_ls=[-2.7998] (σ≈0.06, ≈3°)  d_kick=[+0.0000] (Δσ≈0.000°)
  heads    move= 41 get_poss= 59 exec_move= 91 sprint= 46 kick=  4 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0411 kick_prob=0.0422
  vs       vs[win/loss/tout/miss]  vs_immobile(535): 65.0%/0.0%/0.0%/14.2%/21%
  ep_len   13.4±8.3s  (n=535, min=0.9s, max=49.2s)
  reward   get_possession=+438.00  lose_possession=-3.60  ball_out=-50.00
           box_possession=+870.00  speed_bonus=+858.82  stamina_penalty=-2.81
  rew/ep   (mean/std/min/max per episode, 535 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.819    0.404    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.007    0.078    -0.900    +0.000
  ball_out          -0.093    0.677    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.626    1.192    +0.000    +2.500
  speed_bonus       +1.605    1.517    +0.000    +4.310
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           +0.000    0.000    +0.000    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.005    0.006    -0.025    +0.000
  rew/step (per-step stats, n=24000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     443    +0.018    0.135     +4.116     1.480     +0.662      2.3612      1.194     2.779
  lose_possession       4    -0.000    0.012     +3.581     0.461     +0.306      0.2687      0.466     0.750
  ball_out            10    -0.002    0.102     -4.800     0.400     -7.293     56.7097      7.293     9.428
  box_possession     348    +0.036    0.299     +4.963     1.182     +0.574      1.0800      0.879     1.760
  speed_bonus        336    +0.036    0.328     +5.051     1.106     +0.637      1.0563      0.867     1.751
  stamina_penalty     327    -0.000    0.001     +5.031     1.151     +0.595      1.0947      0.886     1.760
  gae/td   mean_return=+3.183  std_return=1.803  mean_gae=+0.106  mean_sq_td=1.6149
──────────────────────────────────────────────────────────────────────
2026-08-07 19:56:41,166 INFO Saved checkpoint: checkpoints/phase1_run38/checkpoint9.pt
2026-08-07 19:56:41,167 INFO Logging to checkpoints/phase1_run38/training_log10.txt
2026-08-07 19:56:41,167 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 19:56:53,227 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 19:56:53,229 INFO   [eval vs immobile] step=216,000  seeds=16x8  win=55%  mean_rew=3.374±2.919  V=2.952  gap=-0.423  outcomes={'other': 39, 'timeout': 1, 'box_possession': 71, 'miss': 17}
2026-08-07 19:56:53,230 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 19:57:04,804 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 19:57:04,805 INFO   [eval vs rules] step=216,000  seeds=16x8  win=23%  mean_rew=0.182±3.457  V=2.753  gap=+2.570  outcomes={'other': 33, 'box_possession': 30, 'opponent_box_possession': 54, 'timeout': 1, 'miss': 10}
2026-08-07 19:57:36,327 INFO   [early stop e0 mb0]  KL=0.44913 > target=0.12  steps_this_update=1
    [per-head KL] exec_move=+0.0191  sprint=+0.0124  kick=+0.0058  move_dir=+0.2888  kick_dir=+0.1217
2026-08-07 19:57:36,330 INFO   [KL mean=0.4491 median=0.4491 > 0.05] ratio percentiles:  p5=0.139  p25=0.861  p50=0.991  p75=1.000  p95=1.086  max=7.691
  move_dir_log_std=[-2.7998006343841553]  kick_dir_log_std=[-2.7998383045196533]
2026-08-07 19:57:36,343 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.110  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.261  kick=-0.139  t_att=-0.165
    move_dir=2.988 (min=-0.491 max=3.762)  kick_dir=0.105 (min=0.000 max=3.761)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.19
  [worst sample] idx=153  ratio=29.783  adv=-0.146  old_lp=-3.492  new_lp=-0.099
    stored move_dir=-127.0°  new_mean=-131.3°  angular_diff=4.3°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 153  ratio=  29.783  adv=-0.146  lp: old=-3.492  new=-0.099
      rew=+0.0000  ret=+0.0380  val=+0.1837  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9664  sprint_p_new=0.0427  kick_p_new=0.0389  tackle_attempt_p_new=0.0366
    idx= 116  ratio=  28.977  adv=-1.264  lp: old=-7.210  new=-3.844
      rew=+0.0000  ret=+0.4396  val=+1.7034  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9654  sprint_p_new=0.3973  kick_p_new=0.0376  tackle_attempt_p_new=0.0371
  [best sample (highest new_lp)] idx=201  new_lp=0.074  adv=+0.627  stored move_dir=-42.9°  new_mean=-41.7°
    per-head contributions: move_dir:0.185  move:-0.021  tackle_attempt:-0.035  kick:-0.037
2026-08-07 19:57:36,343 INFO   [advantage] mean=-0.003  std=1.012  min=-6.277  max=4.329
2026-08-07 19:57:36,343 INFO   [ratio] mean=0.8833  std=0.3208  min=0.0000  max=7.6909  clipped=27.4%
2026-08-07 19:57:36,344 INFO   [exec head grad norm] move_direction=0.056  exec_move=0.088  sprint=0.067  kick=0.114  kick_direction=0.032  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.090
2026-08-07 19:57:36,344 INFO   [exec continuous log_std] move_direction: start=-2.7998 end=-2.7998   kick_direction: start=-2.7998 end=-2.7998
2026-08-07 19:57:36,344 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0011≈0.06°/step  epoch≈0.1°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0012≈0.07°/step  epoch≈0.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-07 19:57:36,344 INFO   [exec discrete Δlogit per opt step] exec_move=0.0024  sprint=0.0028  kick=0.0006  tackle_attempt=0.0014
2026-08-07 19:57:36,344 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0005  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0191  sprint=+0.0124  kick=+0.0058  tackle_attempt=+0.0009  move_dir=+0.2888  kick_dir=+0.1217
2026-08-07 19:57:36,345 INFO   [grad clip] main: 1/1 steps clipped (100%)  pre-clip norm mean=0.626 max=0.626  limit=0.4
              direction: 1/1 steps clipped (100%)  pre-clip norm mean=0.074 max=0.074  limit=0.02
2026-08-07 19:57:36,349 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=240,000  speed=798/s  reward=3.59
  loss     policy=0.0454  value=0.5093(x0.5)=0.2547
           entropy=1.3271  kl=0.4491
  value    V=3.17±1.26  R=3.31±1.78  adv=0.14±1.25
  moves    mv_ls=[-2.7998] (σ≈0.06, ≈3°) g=3.62e-02  d_move=[+0.0000] (Δσ≈0.000°)
           kk_ls=[-2.7998] (σ≈0.06, ≈3°)  d_kick=[+0.0000] (Δσ≈0.000°)
  heads    move= 42 get_poss= 58 exec_move= 91 sprint= 49 kick=  4 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0408 kick_prob=0.0436
  vs       vs[win/loss/tout/miss]  vs_immobile(566): 67.3%/0.0%/0.4%/11.0%/21%
  ep_len   12.5±7.3s  (n=566, min=1.0s, max=50.0s)
  reward   get_possession=+479.00  lose_possession=-9.00  ball_out=-35.00  box_possession=+952.50
           speed_bonus=+912.45  timeout=-3.00  stamina_penalty=-3.28
  rew/ep   (mean/std/min/max per episode, 566 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.846    0.407    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.016    0.119    -0.900    +0.000
  ball_out          -0.062    0.553    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.683    1.173    +0.000    +2.500
  speed_bonus       +1.612    1.513    +0.000    +4.427
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.005    0.089    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.024    +0.000
  rew/step (per-step stats, n=24000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     483    +0.020    0.140     +4.182     1.367     +0.720      2.4626      1.225     2.909
  lose_possession      10    -0.000    0.018     +2.806     1.058     -0.357      1.4887      0.780     2.287
  ball_out             7    -0.001    0.085     -5.000     0.000     -6.807     50.0001      6.807     8.709
  box_possession     381    +0.040    0.312     +4.892     1.234     +0.356      1.1688      0.892     1.921
  speed_bonus        365    +0.038    0.337     +4.997     1.152     +0.416      1.1627      0.886     1.946
  timeout              2    -0.000    0.014     -1.500     0.000     -1.890      3.6386      1.890     2.123
  stamina_penalty     364    -0.000    0.001     +4.941     1.209     +0.359      1.1372      0.886     1.891
  gae/td   mean_return=+3.312  std_return=1.783  mean_gae=+0.142  mean_sq_td=1.5815
──────────────────────────────────────────────────────────────────────
2026-08-07 19:57:36,375 INFO Saved checkpoint: checkpoints/phase1_run38/checkpoint10.pt
2026-08-07 19:57:36,376 INFO Logging to checkpoints/phase1_run38/training_log11.txt
2026-08-07 19:57:36,377 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 19:57:47,363 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 19:57:47,365 INFO   [eval vs immobile] step=240,000  seeds=16x8  win=54%  mean_rew=3.288±2.990  V=3.096  gap=-0.193  outcomes={'other': 38, 'box_possession': 69, 'miss': 21}
2026-08-07 19:57:47,366 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 19:57:58,644 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 19:57:58,646 INFO   [eval vs rules] step=240,000  seeds=16x8  win=23%  mean_rew=-0.086±3.408  V=2.857  gap=+2.943  outcomes={'box_possession': 29, 'other': 26, 'opponent_box_possession': 63, 'miss': 10}
2026-08-07 19:58:30,436 INFO   [early stop e0 mb0]  KL=0.51908 > target=0.12  steps_this_update=1
    [per-head KL] exec_move=+0.0234  sprint=+0.0195  kick=+0.0060  move_dir=+0.3351  kick_dir=+0.1353
2026-08-07 19:58:30,438 INFO   [KL mean=0.5191 median=0.5191 > 0.05] ratio percentiles:  p5=0.119  p25=0.857  p50=0.989  p75=1.000  p95=1.090  max=13.461
  move_dir_log_std=[-2.7997798919677734]  kick_dir_log_std=[-2.799827814102173]
2026-08-07 19:58:30,457 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.096  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.116  kick=-0.139  t_att=-0.180
    move_dir=2.960 (min=-0.676 max=3.762)  kick_dir=0.104 (min=0.000 max=3.744)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.20
  [worst sample] idx=110  ratio=28.555  adv=+0.798  old_lp=-3.442  new_lp=-0.091
    stored move_dir=173.4°  new_mean=177.8°  angular_diff=4.5°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 110  ratio=  28.555  adv=+0.798  lp: old=-3.442  new=-0.091
      rew=+0.0000  ret=+4.3626  val=+3.5650  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9650  sprint_p_new=0.9658  kick_p_new=0.0357  tackle_attempt_p_new=0.0355
    idx= 171  ratio=  28.497  adv=-2.217  lp: old=-3.448  new=-0.098
      rew=+0.0000  ret=+1.4484  val=+3.6650  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9649  sprint_p_new=0.9634  kick_p_new=0.0376  tackle_attempt_p_new=0.0369
  [best sample (highest new_lp)] idx=49  new_lp=0.077  adv=-0.527  stored move_dir=65.0°  new_mean=67.1°
    per-head contributions: move_dir:0.180  move:-0.023  tackle_attempt:-0.028  kick:-0.034
2026-08-07 19:58:30,458 INFO   [advantage] mean=0.015  std=0.988  min=-7.084  max=4.034
2026-08-07 19:58:30,458 INFO   [ratio] mean=0.8807  std=0.3683  min=0.0000  max=13.4606  clipped=28.0%
2026-08-07 19:58:30,458 INFO   [exec head grad norm] move_direction=0.058  exec_move=0.035  sprint=0.065  kick=0.114  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.098
2026-08-07 19:58:30,458 INFO   [exec continuous log_std] move_direction: start=-2.7998 end=-2.7998   kick_direction: start=-2.7998 end=-2.7998
2026-08-07 19:58:30,458 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0015≈0.08°/step  epoch≈0.1°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0013≈0.08°/step  epoch≈0.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-07 19:58:30,459 INFO   [exec discrete Δlogit per opt step] exec_move=0.0026  sprint=0.0028  kick=0.0006  tackle_attempt=0.0013
2026-08-07 19:58:30,459 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=-0.0001  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0234  sprint=+0.0195  kick=+0.0060  tackle_attempt=-0.0001  move_dir=+0.3351  kick_dir=+0.1353
2026-08-07 19:58:30,459 INFO   [grad clip] main: 1/1 steps clipped (100%)  pre-clip norm mean=1.017 max=1.017  limit=0.4
              direction: 1/1 steps clipped (100%)  pre-clip norm mean=0.066 max=0.066  limit=0.02
2026-08-07 19:58:30,475 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=264,000  speed=790/s  reward=4.28
  loss     policy=0.0243  value=0.4822(x0.5)=0.2411
           entropy=1.3285  kl=0.5191
  value    V=3.29±1.27  R=3.39±1.74  adv=0.10±1.21
  moves    mv_ls=[-2.7998] (σ≈0.06, ≈3°) g=3.12e-02  d_move=[+0.0000] (Δσ≈0.000°)
           kk_ls=[-2.7998] (σ≈0.06, ≈3°)  d_kick=[+0.0000] (Δσ≈0.000°)
  heads    move= 44 get_poss= 57 exec_move= 91 sprint= 49 kick=  4 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0395 kick_prob=0.0425
  vs       vs[win/loss/tout/miss]  vs_immobile(536): 68.1%/0.2%/0.2%/10.3%/21%
  ep_len   13.4±7.6s  (n=536, min=0.6s, max=50.0s)
  reward   get_possession=+446.00  lose_possession=-2.70  ball_out=-20.00  box_possession=+912.50
           speed_bonus=+903.86  opponent_box=-3.00  timeout=-1.50  stamina_penalty=-3.21
  rew/ep   (mean/std/min/max per episode, 536 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.832    0.388    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.005    0.067    -0.900    +0.000
  ball_out          -0.037    0.430    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.702    1.165    +0.000    +2.500
  speed_bonus       +1.686    1.520    +0.000    +4.234
  opponent_box      -0.006    0.129    -3.000    +0.000
  timeout           -0.003    0.065    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.033    +0.000
  rew/step (per-step stats, n=24000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     451    +0.019    0.136     +4.316     1.252     +0.701      2.2610      1.145     2.670
  lose_possession       3    -0.000    0.010     +3.597     0.455     +0.267      0.2392      0.459     0.660
  ball_out             4    -0.001    0.065     -4.750     0.433     -7.121     54.0047      7.121     8.945
  box_possession     365    +0.038    0.306     +4.970     1.191     +0.239      0.9196      0.771     1.894
  speed_bonus        350    +0.038    0.337     +5.076     1.098     +0.296      0.8911      0.756     1.786
  opponent_box         1    -0.000    0.019     -3.005     0.000     -6.433     41.3827      6.433     6.433
  timeout              1    -0.000    0.010     -1.500     0.000     -1.870      3.4968      1.870     1.870
  stamina_penalty     343    -0.000    0.001     +5.024     1.234     +0.236      1.0477      0.792     1.982
  gae/td   mean_return=+3.394  std_return=1.736  mean_gae=+0.104  mean_sq_td=1.4680
──────────────────────────────────────────────────────────────────────
2026-08-07 19:58:30,499 INFO Saved checkpoint: checkpoints/phase1_run38/checkpoint11.pt
2026-08-07 19:58:30,499 INFO Logging to checkpoints/phase1_run38/training_log12.txt
2026-08-07 19:58:30,500 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 19:58:41,548 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 19:58:41,549 INFO   [eval vs immobile] step=264,000  seeds=16x8  win=55%  mean_rew=3.325±2.910  V=3.273  gap=-0.052  outcomes={'other': 38, 'box_possession': 70, 'timeout': 1, 'miss': 19}
2026-08-07 19:58:41,551 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 19:58:52,255 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 19:58:52,256 INFO   [eval vs rules] step=264,000  seeds=16x8  win=26%  mean_rew=0.355±3.485  V=3.057  gap=+2.702  outcomes={'other': 34, 'opponent_box_possession': 52, 'box_possession': 33, 'miss': 9}
2026-08-07 19:59:23,962 INFO   [early stop e0 mb0]  KL=0.43960 > target=0.12  steps_this_update=1
    [per-head KL] exec_move=+0.0183  sprint=+0.0139  kick=+0.0051  tackle_attempt=+0.0012  move_dir=+0.2923  kick_dir=+0.1084
2026-08-07 19:59:23,964 INFO   [KL mean=0.4396 median=0.4396 > 0.05] ratio percentiles:  p5=0.118  p25=0.867  p50=0.991  p75=1.000  p95=1.097  max=11.368
  move_dir_log_std=[-2.7997593879699707]  kick_dir_log_std=[-2.7998180389404297]
2026-08-07 19:59:23,976 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.139  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.170  kick=-0.157  t_att=-0.151
    move_dir=3.036 (min=-0.044 max=3.761)  kick_dir=0.144 (min=0.000 max=3.761)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.20
  [worst sample] idx=39  ratio=25.701  adv=-0.127  old_lp=-3.351  new_lp=-0.104
    stored move_dir=65.3°  new_mean=74.1°  angular_diff=8.8°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx=  39  ratio=  25.701  adv=-0.127  lp: old=-3.351  new=-0.104
      rew=+0.0000  ret=+0.1959  val=+0.3227  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9611  sprint_p_new=0.9621  kick_p_new=0.0365  tackle_attempt_p_new=0.0445
    idx=  96  ratio=  25.284  adv=-0.114  lp: old=-3.330  new=-0.100
      rew=+0.0000  ret=+4.3163  val=+4.4302  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9604  sprint_p_new=0.0418  kick_p_new=0.0398  tackle_attempt_p_new=0.0350
  [best sample (highest new_lp)] idx=1  new_lp=0.068  adv=+0.282  stored move_dir=-168.5°  new_mean=-168.9°
    per-head contributions: move_dir:0.188  sprint:-0.031  tackle_attempt:-0.037  kick:-0.037
2026-08-07 19:59:23,976 INFO   [advantage] mean=-0.000  std=1.007  min=-6.651  max=4.879
2026-08-07 19:59:23,976 INFO   [ratio] mean=0.8936  std=0.4104  min=0.0000  max=11.3678  clipped=27.0%
2026-08-07 19:59:23,976 INFO   [exec head grad norm] move_direction=0.055  exec_move=0.071  sprint=0.032  kick=0.083  kick_direction=0.007  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.107
2026-08-07 19:59:23,976 INFO   [exec continuous log_std] move_direction: start=-2.7998 end=-2.7998   kick_direction: start=-2.7998 end=-2.7998
2026-08-07 19:59:23,977 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0014≈0.08°/step  epoch≈0.1°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0013≈0.08°/step  epoch≈0.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-07 19:59:23,977 INFO   [exec discrete Δlogit per opt step] exec_move=0.0027  sprint=0.0029  kick=0.0006  tackle_attempt=0.0015
2026-08-07 19:59:23,977 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0003  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0183  sprint=+0.0139  kick=+0.0051  tackle_attempt=+0.0012  move_dir=+0.2923  kick_dir=+0.1084
2026-08-07 19:59:23,977 INFO   [grad clip] main: 1/1 steps clipped (100%)  pre-clip norm mean=0.486 max=0.486  limit=0.4
              direction: 1/1 steps clipped (100%)  pre-clip norm mean=0.062 max=0.062  limit=0.02
2026-08-07 19:59:23,981 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=288,000  speed=794/s  reward=5.16
  loss     policy=0.0335  value=0.4924(x0.5)=0.2462
           entropy=1.3284  kl=0.4396
  value    V=3.44±1.35  R=3.41±1.83  adv=-0.04±1.26
  moves    mv_ls=[-2.7998] (σ≈0.06, ≈3°) g=2.81e-02  d_move=[+0.0000] (Δσ≈0.000°)
           kk_ls=[-2.7998] (σ≈0.06, ≈3°)  d_kick=[+0.0000] (Δσ≈0.000°)
  heads    move= 42 get_poss= 58 exec_move= 91 sprint= 49 kick=  4 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0405 kick_prob=0.0427
  vs       vs[win/loss/tout/miss]  vs_immobile(544): 65.8%/0.0%/0.6%/9.6%/24%
  ep_len   13.2±7.8s  (n=544, min=0.4s, max=50.0s)
  reward   get_possession=+456.00  lose_possession=-2.70  ball_out=-35.00  box_possession=+895.00
           speed_bonus=+907.21  timeout=-4.50  stamina_penalty=-3.06
  rew/ep   (mean/std/min/max per episode, 544 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.838    0.383    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.005    0.067    -0.900    +0.000
  ball_out          -0.064    0.564    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.645    1.186    +0.000    +2.500
  speed_bonus       +1.668    1.539    +0.000    +4.489
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.008    0.111    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.032    +0.000
  rew/step (per-step stats, n=24000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     459    +0.019    0.137     +4.363     1.434     +0.572      2.1895      1.140     2.787
  lose_possession       3    -0.000    0.010     +3.360     0.656     -0.230      0.6888      0.675     1.257
  ball_out             7    -0.001    0.085     -4.714     0.452     -6.938     50.3064      6.938     8.858
  box_possession     358    +0.037    0.303     +5.026     1.182     +0.158      1.0994      0.795     1.810
  speed_bonus        348    +0.038    0.340     +5.098     1.117     +0.203      1.0679      0.778     1.774
  timeout              3    -0.000    0.017     -1.500     0.000     -1.429      2.1364      1.429     1.690
  stamina_penalty     345    -0.000    0.001     +5.038     1.223     +0.128      0.9573      0.767     1.762
  gae/td   mean_return=+3.407  std_return=1.827  mean_gae=-0.038  mean_sq_td=1.5966
──────────────────────────────────────────────────────────────────────
2026-08-07 19:59:24,003 INFO Saved checkpoint: checkpoints/phase1_run38/checkpoint12.pt
2026-08-07 19:59:24,004 INFO Logging to checkpoints/phase1_run38/training_log13.txt
2026-08-07 19:59:24,005 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 19:59:35,012 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 19:59:35,014 INFO   [eval vs immobile] step=288,000  seeds=16x8  win=54%  mean_rew=3.218±2.839  V=3.441  gap=+0.223  outcomes={'other': 42, 'box_possession': 69, 'miss': 17}
2026-08-07 19:59:35,015 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 19:59:48,550 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 19:59:48,551 INFO   [eval vs rules] step=288,000  seeds=16x8  win=31%  mean_rew=0.655±3.606  V=3.229  gap=+2.574  outcomes={'box_possession': 40, 'other': 28, 'opponent_box_possession': 51, 'timeout': 1, 'miss': 8}
2026-08-07 20:00:20,558 INFO   [early stop e0 mb0]  KL=0.43243 > target=0.12  steps_this_update=1
    [per-head KL] exec_move=+0.0273  sprint=+0.0176  kick=+0.0052  move_dir=+0.2828  kick_dir=+0.0996
2026-08-07 20:00:20,561 INFO   [KL mean=0.4324 median=0.4324 > 0.05] ratio percentiles:  p5=0.132  p25=0.867  p50=0.991  p75=1.000  p95=1.083  max=4.748
  move_dir_log_std=[-2.7997376918792725]  kick_dir_log_std=[-2.7998087406158447]
2026-08-07 20:00:20,573 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.081  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.120  kick=-0.206  t_att=-0.128
    move_dir=3.042 (min=-0.376 max=3.762)  kick_dir=0.189 (min=0.000 max=3.739)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.18
  [worst sample] idx=246  ratio=27.826  adv=-2.655  old_lp=-3.425  new_lp=-0.099
    stored move_dir=-153.9°  new_mean=-158.0°  angular_diff=4.2°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 246  ratio=  27.826  adv=-2.655  lp: old=-3.425  new=-0.099
      rew=+0.0000  ret=+2.1556  val=+4.8103  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9640  sprint_p_new=0.0762  kick_p_new=0.0379  tackle_attempt_p_new=0.0369
    idx=  66  ratio=  26.180  adv=-1.746  lp: old=-3.362  new=-0.097
      rew=+0.0000  ret=+2.4155  val=+4.1616  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9618  sprint_p_new=0.9603  kick_p_new=0.0389  tackle_attempt_p_new=0.0384
  [best sample (highest new_lp)] idx=157  new_lp=0.076  adv=+1.144  stored move_dir=-22.2°  new_mean=-22.1°
    per-head contributions: move_dir:0.188  move:-0.021  tackle_attempt:-0.037  kick:-0.037
2026-08-07 20:00:20,573 INFO   [advantage] mean=0.007  std=0.985  min=-5.991  max=4.792
2026-08-07 20:00:20,573 INFO   [ratio] mean=0.8808  std=0.3000  min=0.0000  max=4.7479  clipped=27.1%
2026-08-07 20:00:20,574 INFO   [exec head grad norm] move_direction=0.025  exec_move=0.123  sprint=0.136  kick=0.088  kick_direction=0.005  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.061
2026-08-07 20:00:20,574 INFO   [exec continuous log_std] move_direction: start=-2.7998 end=-2.7997   kick_direction: start=-2.7998 end=-2.7998
2026-08-07 20:00:20,574 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0012≈0.07°/step  epoch≈0.1°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0013≈0.07°/step  epoch≈0.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-07 20:00:20,574 INFO   [exec discrete Δlogit per opt step] exec_move=0.0024  sprint=0.0028  kick=0.0006  tackle_attempt=0.0014
2026-08-07 20:00:20,574 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0003  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0273  sprint=+0.0176  kick=+0.0052  tackle_attempt=-0.0003  move_dir=+0.2828  kick_dir=+0.0996
2026-08-07 20:00:20,574 INFO   [grad clip] main: 1/1 steps clipped (100%)  pre-clip norm mean=0.730 max=0.730  limit=0.4
              direction: 1/1 steps clipped (100%)  pre-clip norm mean=0.040 max=0.040  limit=0.02
2026-08-07 20:00:20,578 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=312,000  speed=788/s  reward=3.71
  loss     policy=0.0326  value=0.4726(x0.5)=0.2363
           entropy=1.3201  kl=0.4324
  value    V=3.53±1.38  R=3.40±1.81  adv=-0.13±1.23
  moves    mv_ls=[-2.7997] (σ≈0.06, ≈3°) g=2.98e-02  d_move=[+0.0000] (Δσ≈0.000°)
           kk_ls=[-2.7998] (σ≈0.06, ≈3°)  d_kick=[+0.0000] (Δσ≈0.000°)
  heads    move= 43 get_poss= 57 exec_move= 91 sprint= 49 kick=  4 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0420 kick_prob=0.0423
  vs       vs[win/loss/tout/miss]  vs_immobile(532): 64.3%/0.0%/0.2%/12.6%/23%
  ep_len   13.4±7.8s  (n=532, min=0.5s, max=50.0s)
  reward   get_possession=+442.00  lose_possession=-7.20  ball_out=-30.00  box_possession=+855.00
           speed_bonus=+846.24  timeout=-1.50  stamina_penalty=-3.10
  rew/ep   (mean/std/min/max per episode, 532 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.831    0.418    +0.000    +3.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.014    0.123    -1.800    +0.000
  ball_out          -0.056    0.528    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.607    1.198    +0.000    +2.500
  speed_bonus       +1.591    1.523    +0.000    +4.258
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.003    0.065    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.035    +0.000
  rew/step (per-step stats, n=24000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     447    +0.019    0.135     +4.347     1.426     +0.366      2.0465      1.030     2.793
  lose_possession       8    -0.000    0.016     +2.950     1.536     -0.753      2.2989      1.207     2.550
  ball_out             6    -0.001    0.079     -4.833     0.373     -7.944     64.4574      7.944     9.463
  box_possession     342    +0.036    0.296     +4.968     1.186     +0.049      1.0239      0.783     1.869
  speed_bonus        328    +0.035    0.326     +5.074     1.092     +0.113      0.9420      0.746     1.753
  timeout              1    -0.000    0.010     -1.500     0.000     -1.199      1.4374      1.199     1.199
  stamina_penalty     324    -0.000    0.001     +5.032     1.159     +0.056      1.0049      0.778     1.857
  gae/td   mean_return=+3.399  std_return=1.809  mean_gae=-0.126  mean_sq_td=1.5385
──────────────────────────────────────────────────────────────────────
2026-08-07 20:00:20,600 INFO Saved checkpoint: checkpoints/phase1_run38/checkpoint13.pt
2026-08-07 20:00:20,600 INFO Logging to checkpoints/phase1_run38/training_log14.txt
2026-08-07 20:00:20,601 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 20:00:31,754 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 20:00:31,755 INFO   [eval vs immobile] step=312,000  seeds=16x8  win=55%  mean_rew=3.402±2.972  V=3.432  gap=+0.030  outcomes={'other': 38, 'box_possession': 70, 'miss': 20}
2026-08-07 20:00:31,756 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 20:00:44,090 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 20:00:44,092 INFO   [eval vs rules] step=312,000  seeds=16x8  win=30%  mean_rew=0.637±3.721  V=3.340  gap=+2.704  outcomes={'other': 29, 'opponent_box_possession': 53, 'box_possession': 38, 'miss': 8}
2026-08-07 20:01:16,120 INFO   [early stop e0 mb0]  KL=0.44731 > target=0.12  steps_this_update=1
    [per-head KL] exec_move=+0.0282  sprint=+0.0152  kick=+0.0051  move_dir=+0.3005  kick_dir=+0.0974
2026-08-07 20:01:16,123 INFO   [KL mean=0.4473 median=0.4473 > 0.05] ratio percentiles:  p5=0.104  p25=0.850  p50=0.989  p75=1.000  p95=1.095  max=5.877
  move_dir_log_std=[-2.799715995788574]  kick_dir_log_std=[-2.799799919128418]
2026-08-07 20:01:16,133 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.141  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.133  kick=-0.188  t_att=-0.209
    move_dir=3.005 (min=-0.545 max=3.762)  kick_dir=0.127 (min=-3.183 max=3.716)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.20
  [worst sample] idx=235  ratio=27.099  adv=+0.695  old_lp=-3.404  new_lp=-0.105
    stored move_dir=7.0°  new_mean=2.5°  angular_diff=4.4°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 235  ratio=  27.099  adv=+0.695  lp: old=-3.404  new=-0.105
      rew=+0.0000  ret=+5.0374  val=+4.3428  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9631  sprint_p_new=0.0336  kick_p_new=0.0369  tackle_attempt_p_new=0.0456
    idx= 159  ratio=  26.254  adv=-1.028  lp: old=-3.362  new=-0.095
      rew=+0.0000  ret=+3.9920  val=+5.0197  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9619  sprint_p_new=0.0273  kick_p_new=0.0375  tackle_attempt_p_new=0.0334
  [best sample (highest new_lp)] idx=214  new_lp=0.069  adv=-4.373  stored move_dir=86.7°  new_mean=84.7°
    per-head contributions: move_dir:0.180  move:-0.021  sprint:-0.026  tackle_attempt:-0.030  kick:-0.035
2026-08-07 20:01:16,134 INFO   [advantage] mean=-0.008  std=1.011  min=-6.618  max=4.264
2026-08-07 20:01:16,134 INFO   [ratio] mean=0.8784  std=0.3550  min=0.0000  max=5.8768  clipped=28.3%
2026-08-07 20:01:16,134 INFO   [exec head grad norm] move_direction=0.050  exec_move=0.057  sprint=0.022  kick=0.061  kick_direction=0.006  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.121
2026-08-07 20:01:16,134 INFO   [exec continuous log_std] move_direction: start=-2.7997 end=-2.7997   kick_direction: start=-2.7998 end=-2.7998
2026-08-07 20:01:16,134 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0014≈0.08°/step  epoch≈0.1°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0016≈0.09°/step  epoch≈0.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-07 20:01:16,134 INFO   [exec discrete Δlogit per opt step] exec_move=0.0028  sprint=0.0032  kick=0.0006  tackle_attempt=0.0015
2026-08-07 20:01:16,135 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0001  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0282  sprint=+0.0152  kick=+0.0051  tackle_attempt=+0.0009  move_dir=+0.3005  kick_dir=+0.0974
2026-08-07 20:01:16,135 INFO   [grad clip] main: 1/1 steps clipped (100%)  pre-clip norm mean=0.818 max=0.818  limit=0.4
              direction: 1/1 steps clipped (100%)  pre-clip norm mean=0.061 max=0.061  limit=0.02
2026-08-07 20:01:16,138 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=336,000  speed=786/s  reward=5.16
  loss     policy=0.0527  value=0.4741(x0.5)=0.2370
           entropy=1.3324  kl=0.4473
  value    V=3.61±1.36  R=3.56±1.77  adv=-0.06±1.19
  moves    mv_ls=[-2.7997] (σ≈0.06, ≈3°) g=3.42e-02  d_move=[+0.0000] (Δσ≈0.000°)
           kk_ls=[-2.7998] (σ≈0.06, ≈3°)  d_kick=[+0.0000] (Δσ≈0.000°)
  heads    move= 44 get_poss= 57 exec_move= 91 sprint= 50 kick=  4 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0396 kick_prob=0.0431
  vs       vs[win/loss/tout/miss]  vs_immobile(546): 67.0%/0.0%/0.0%/10.6%/22%
  ep_len   13.0±7.5s  (n=546, min=1.5s, max=42.6s)
  reward   get_possession=+448.00  lose_possession=-3.60  ball_out=-20.00
           box_possession=+915.00  speed_bonus=+903.00  stamina_penalty=-3.16
  rew/ep   (mean/std/min/max per episode, 546 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.821    0.407    +0.000    +3.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.007    0.094    -1.800    +0.000
  ball_out          -0.037    0.426    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.676    1.175    +0.000    +2.500
  speed_bonus       +1.654    1.555    +0.000    +4.464
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           +0.000    0.000    +0.000    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.029    +0.000
  rew/step (per-step stats, n=24000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     454    +0.019    0.136     +4.510     1.253     +0.374      1.4780      0.897     2.471
  lose_possession       4    -0.000    0.012     +3.531     0.528     -0.281      0.5062      0.674     0.955
  ball_out             4    -0.001    0.065     -4.500     0.500     -5.657     34.0098      5.657     7.572
  box_possession     366    +0.038    0.306     +4.959     1.263     +0.145      1.2534      0.897     1.915
  speed_bonus        345    +0.038    0.340     +5.108     1.140     +0.243      1.1793      0.863     1.888
  stamina_penalty     350    -0.000    0.001     +5.018     1.232     +0.164      1.1349      0.860     1.871
  gae/td   mean_return=+3.556  std_return=1.767  mean_gae=-0.057  mean_sq_td=1.4095
──────────────────────────────────────────────────────────────────────
2026-08-07 20:01:16,161 INFO Saved checkpoint: checkpoints/phase1_run38/checkpoint14.pt
2026-08-07 20:01:16,162 INFO Logging to checkpoints/phase1_run38/training_log15.txt
2026-08-07 20:01:16,163 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 20:01:27,357 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 20:01:27,359 INFO   [eval vs immobile] step=336,000  seeds=16x8  win=55%  mean_rew=3.366±2.923  V=3.392  gap=+0.026  outcomes={'other': 37, 'box_possession': 71, 'miss': 19, 'timeout': 1}
2026-08-07 20:01:27,360 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 20:01:39,757 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 20:01:39,758 INFO   [eval vs rules] step=336,000  seeds=16x8  win=27%  mean_rew=0.441±3.698  V=3.302  gap=+2.860  outcomes={'other': 29, 'opponent_box_possession': 54, 'box_possession': 34, 'timeout': 3, 'miss': 8}
2026-08-07 20:02:10,456 INFO   [early stop e0 mb0]  KL=0.40602 > target=0.12  steps_this_update=1
    [per-head KL] exec_move=+0.0189  sprint=+0.0132  kick=+0.0057  tackle_attempt=+0.0010  move_dir=+0.2728  kick_dir=+0.0943
2026-08-07 20:02:10,458 INFO   [KL mean=0.4060 median=0.4060 > 0.05] ratio percentiles:  p5=0.147  p25=0.854  p50=0.989  p75=1.000  p95=1.087  max=9.437
  move_dir_log_std=[-2.799694299697876]  kick_dir_log_std=[-2.7997915744781494]
2026-08-07 20:02:10,469 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.141  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.119  kick=-0.147  t_att=-0.141
    move_dir=2.677 (min=-1.036 max=3.762)  kick_dir=0.125 (min=0.000 max=3.749)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.33
  [worst sample] idx=71  ratio=33.711  adv=-0.696  old_lp=-3.608  new_lp=-0.090
    stored move_dir=-154.2°  new_mean=-151.3°  angular_diff=2.9°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx=  71  ratio=  33.711  adv=-0.696  lp: old=-3.608  new=-0.090
      rew=+0.0000  ret=+3.4038  val=+4.0997  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9703  sprint_p_new=0.9704  kick_p_new=0.0373  tackle_attempt_p_new=0.0379
    idx= 164  ratio=  24.798  adv=+2.569  lp: old=-3.318  new=-0.107
      rew=+0.0000  ret=+2.8411  val=+0.2717  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9597  sprint_p_new=0.9563  kick_p_new=0.0402  tackle_attempt_p_new=0.0443
  [best sample (highest new_lp)] idx=155  new_lp=0.084  adv=+1.047  stored move_dir=-134.2°  new_mean=-131.9°
    per-head contributions: move_dir:0.178  tackle_attempt:-0.020  move:-0.021  kick:-0.046
2026-08-07 20:02:10,470 INFO   [advantage] mean=-0.005  std=1.009  min=-6.712  max=4.807
2026-08-07 20:02:10,470 INFO   [ratio] mean=0.8879  std=0.3706  min=0.0000  max=9.4367  clipped=27.7%
2026-08-07 20:02:10,470 INFO   [exec head grad norm] move_direction=0.056  exec_move=0.072  sprint=0.055  kick=0.092  kick_direction=0.010  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.095
2026-08-07 20:02:10,470 INFO   [exec continuous log_std] move_direction: start=-2.7997 end=-2.7997   kick_direction: start=-2.7998 end=-2.7998
2026-08-07 20:02:10,471 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0013≈0.08°/step  epoch≈0.1°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0015≈0.08°/step  epoch≈0.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-07 20:02:10,471 INFO   [exec discrete Δlogit per opt step] exec_move=0.0026  sprint=0.0029  kick=0.0006  tackle_attempt=0.0014
2026-08-07 20:02:10,471 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0000  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0189  sprint=+0.0132  kick=+0.0057  tackle_attempt=+0.0010  move_dir=+0.2728  kick_dir=+0.0943
2026-08-07 20:02:10,471 INFO   [grad clip] main: 1/1 steps clipped (100%)  pre-clip norm mean=0.845 max=0.845  limit=0.4
              direction: 1/1 steps clipped (100%)  pre-clip norm mean=0.065 max=0.065  limit=0.02
2026-08-07 20:02:10,475 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=360,000  speed=819/s  reward=3.02
  loss     policy=0.0530  value=0.4838(x0.5)=0.2419
           entropy=1.3268  kl=0.4060
  value    V=3.54±1.39  R=3.45±1.83  adv=-0.09±1.26
  moves    mv_ls=[-2.7997] (σ≈0.06, ≈3°) g=3.14e-02  d_move=[+0.0000] (Δσ≈0.000°)
           kk_ls=[-2.7998] (σ≈0.06, ≈3°)  d_kick=[+0.0000] (Δσ≈0.000°)
  heads    move= 42 get_poss= 58 exec_move= 91 sprint= 48 kick=  4 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0416 kick_prob=0.0434
  vs       vs[win/loss/tout/miss]  vs_immobile(528): 65.9%/0.2%/0.4%/12.1%/21%
  ep_len   13.6±8.0s  (n=528, min=0.6s, max=50.0s)
  reward   get_possession=+449.00  lose_possession=-6.30  ball_out=-35.00  box_possession=+870.00
           speed_bonus=+873.67  opponent_box=-3.00  timeout=-3.00  stamina_penalty=-3.05
  rew/ep   (mean/std/min/max per episode, 528 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.850    0.392    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.012    0.103    -0.900    +0.000
  ball_out          -0.066    0.572    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.648    1.185    +0.000    +2.500
  speed_bonus       +1.655    1.543    +0.000    +4.244
  opponent_box      -0.006    0.130    -3.000    +0.000
  timeout           -0.006    0.092    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.031    +0.000
  rew/step (per-step stats, n=24000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     452    +0.019    0.136     +4.364     1.395     +0.261      2.0746      0.998     2.834
  lose_possession       7    -0.000    0.015     +3.553     0.719     -0.172      0.5548      0.604     1.128
  ball_out             7    -0.001    0.085     -4.714     0.452     -5.869     37.4401      5.869     8.161
  box_possession     348    +0.036    0.299     +5.005     1.206     +0.303      1.3369      0.895     2.096
  speed_bonus        335    +0.036    0.334     +5.102     1.121     +0.371      1.2928      0.873     2.095
  opponent_box         1    -0.000    0.019     -3.002     0.000     -7.117     50.6449      7.117     7.117
  timeout              2    -0.000    0.014     -1.500     0.000     -1.227      1.5058      1.227     1.237
  stamina_penalty     329    -0.000    0.001     +5.041     1.270     +0.320      1.4515      0.906     2.096
  gae/td   mean_return=+3.450  std_return=1.827  mean_gae=-0.093  mean_sq_td=1.5931
──────────────────────────────────────────────────────────────────────
2026-08-07 20:02:10,502 INFO Saved checkpoint: checkpoints/phase1_run38/checkpoint15.pt
2026-08-07 20:02:10,502 INFO Logging to checkpoints/phase1_run38/training_log16.txt
2026-08-07 20:02:10,503 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 20:02:21,054 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 20:02:21,055 INFO   [eval vs immobile] step=360,000  seeds=16x8  win=59%  mean_rew=3.523±3.115  V=3.366  gap=-0.157  outcomes={'other': 32, 'box_possession': 75, 'timeout': 1, 'miss': 20}
2026-08-07 20:02:21,063 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 20:02:33,821 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 20:02:33,822 INFO   [eval vs rules] step=360,000  seeds=16x8  win=23%  mean_rew=0.165±3.511  V=3.247  gap=+3.082  outcomes={'other': 32, 'opponent_box_possession': 58, 'box_possession': 29, 'timeout': 1, 'miss': 8}
2026-08-07 20:03:04,499 INFO   [early stop e0 mb0]  KL=0.43295 > target=0.12  steps_this_update=1
    [per-head KL] exec_move=+0.0227  sprint=+0.0145  kick=+0.0061  move_dir=+0.2737  kick_dir=+0.1148
2026-08-07 20:03:04,501 INFO   [KL mean=0.4330 median=0.4330 > 0.05] ratio percentiles:  p5=0.155  p25=0.865  p50=0.992  p75=1.000  p95=1.081  max=11.043
  move_dir_log_std=[-2.799672842025757]  kick_dir_log_std=[-2.799783706665039]
2026-08-07 20:03:04,513 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.155  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.154  kick=-0.215  t_att=-0.174
    move_dir=2.980 (min=-3.659 max=3.761)  kick_dir=0.188 (min=0.000 max=3.762)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.22
  [worst sample] idx=174  ratio=30.794  adv=-0.047  old_lp=-3.515  new_lp=-0.088
    stored move_dir=57.4°  new_mean=59.5°  angular_diff=2.2°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 174  ratio=  30.794  adv=-0.047  lp: old=-3.515  new=-0.088
      rew=+0.0000  ret=+0.0058  val=+0.0530  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9675  sprint_p_new=0.0176  kick_p_new=0.0332  tackle_attempt_p_new=0.0304
    idx= 136  ratio=  29.544  adv=-1.716  lp: old=-3.479  new=-0.093
      rew=+0.0000  ret=+2.0521  val=+3.7682  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9662  sprint_p_new=0.9647  kick_p_new=0.0374  tackle_attempt_p_new=0.0349
  [best sample (highest new_lp)] idx=173  new_lp=0.083  adv=-0.046  stored move_dir=65.0°  new_mean=64.9°
    per-head contributions: move_dir:0.188  move:-0.023  tackle_attempt:-0.029  kick:-0.034
2026-08-07 20:03:04,514 INFO   [advantage] mean=0.001  std=0.984  min=-6.059  max=4.428
2026-08-07 20:03:04,514 INFO   [ratio] mean=0.8903  std=0.3656  min=0.0000  max=11.0427  clipped=27.0%
2026-08-07 20:03:04,514 INFO   [exec head grad norm] move_direction=0.059  exec_move=0.103  sprint=0.050  kick=0.123  kick_direction=0.010  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.043
2026-08-07 20:03:04,514 INFO   [exec continuous log_std] move_direction: start=-2.7997 end=-2.7997   kick_direction: start=-2.7998 end=-2.7998
2026-08-07 20:03:04,514 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0012≈0.07°/step  epoch≈0.1°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0014≈0.08°/step  epoch≈0.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-07 20:03:04,515 INFO   [exec discrete Δlogit per opt step] exec_move=0.0025  sprint=0.0029  kick=0.0006  tackle_attempt=0.0014
2026-08-07 20:03:04,515 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0002  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0227  sprint=+0.0145  kick=+0.0061  tackle_attempt=+0.0009  move_dir=+0.2737  kick_dir=+0.1148
2026-08-07 20:03:04,515 INFO   [grad clip] main: 1/1 steps clipped (100%)  pre-clip norm mean=0.645 max=0.645  limit=0.4
              direction: 1/1 steps clipped (100%)  pre-clip norm mean=0.068 max=0.068  limit=0.02
2026-08-07 20:03:04,519 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=384,000  speed=817/s  reward=3.47
  loss     policy=0.0485  value=0.4564(x0.5)=0.2282
           entropy=1.3240  kl=0.4330
  value    V=3.41±1.42  R=3.35±1.84  adv=-0.06±1.25
  moves    mv_ls=[-2.7997] (σ≈0.06, ≈3°) g=3.37e-02  d_move=[+0.0000] (Δσ≈0.000°)
           kk_ls=[-2.7998] (σ≈0.06, ≈3°)  d_kick=[+0.0000] (Δσ≈0.000°)
  heads    move= 42 get_poss= 58 exec_move= 92 sprint= 49 kick=  4 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0404 kick_prob=0.0436
  vs       vs[win/loss/tout/miss]  vs_immobile(567): 64.7%/0.0%/0.0%/13.2%/22%
  ep_len   12.6±7.4s  (n=567, min=0.4s, max=49.7s)
  reward   get_possession=+458.00  ball_out=-40.00  box_possession=+917.50
           speed_bonus=+855.05  stamina_penalty=-3.16
  rew/ep   (mean/std/min/max per episode, 567 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.808    0.394    +0.000    +1.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    +0.000    0.000    +0.000    +0.000
  ball_out          -0.071    0.590    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.618    1.195    +0.000    +2.500
  speed_bonus       +1.508    1.505    +0.000    +4.326
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           +0.000    0.000    +0.000    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.032    +0.000
  rew/step (per-step stats, n=24000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     461    +0.019    0.137     +4.307     1.342     +0.311      1.7137      0.929     2.769
  ball_out             8    -0.002    0.091     -4.750     0.433     -5.628     36.8511      5.628     9.350
  box_possession     367    +0.038    0.307     +4.827     1.257     +0.427      1.2886      0.932     1.942
  speed_bonus        353    +0.036    0.326     +4.919     1.191     +0.487      1.2841      0.926     1.897
  stamina_penalty     348    -0.000    0.001     +4.866     1.221     +0.446      1.1709      0.910     1.897
  gae/td   mean_return=+3.354  std_return=1.843  mean_gae=-0.059  mean_sq_td=1.5775
──────────────────────────────────────────────────────────────────────
2026-08-07 20:03:04,542 INFO Saved checkpoint: checkpoints/phase1_run38/checkpoint16.pt
2026-08-07 20:03:04,542 INFO Logging to checkpoints/phase1_run38/training_log17.txt
2026-08-07 20:03:04,543 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 20:03:15,223 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 20:03:15,225 INFO   [eval vs immobile] step=384,000  seeds=16x8  win=55%  mean_rew=3.364±2.893  V=3.175  gap=-0.190  outcomes={'other': 39, 'box_possession': 71, 'miss': 18}
2026-08-07 20:03:15,226 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 20:03:25,760 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 20:03:25,762 INFO   [eval vs rules] step=384,000  seeds=16x8  win=23%  mean_rew=0.053±3.389  V=3.268  gap=+3.215  outcomes={'other': 32, 'box_possession': 29, 'opponent_box_possession': 59, 'miss': 8}
2026-08-07 20:03:56,327 INFO   [early stop e0 mb0]  KL=0.43463 > target=0.12  steps_this_update=1
    [per-head KL] exec_move=+0.0192  sprint=+0.0138  kick=+0.0044  tackle_attempt=+0.0014  move_dir=+0.2931  kick_dir=+0.1020
2026-08-07 20:03:56,329 INFO   [KL mean=0.4346 median=0.4346 > 0.05] ratio percentiles:  p5=0.130  p25=0.851  p50=0.989  p75=1.000  p95=1.078  max=9.941
  move_dir_log_std=[-2.7996511459350586]  kick_dir_log_std=[-2.799776315689087]
2026-08-07 20:03:56,339 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.081  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.186  kick=-0.171  t_att=-0.153
    move_dir=2.765 (min=-0.150 max=3.761)  kick_dir=0.125 (min=0.000 max=3.687)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.26
  [worst sample] idx=218  ratio=33.035  adv=-0.104  old_lp=-3.584  new_lp=-0.086
    stored move_dir=3.3°  new_mean=1.1°  angular_diff=2.2°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 218  ratio=  33.035  adv=-0.104  lp: old=-3.584  new=-0.086
      rew=+0.0000  ret=+3.7854  val=+3.8898  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9697  sprint_p_new=0.9659  kick_p_new=0.0350  tackle_attempt_p_new=0.0346
    idx= 133  ratio=  25.641  adv=+1.177  lp: old=-3.338  new=-0.094
      rew=+0.0000  ret=+5.3467  val=+4.1697  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9611  sprint_p_new=0.9614  kick_p_new=0.0369  tackle_attempt_p_new=0.0362
  [best sample (highest new_lp)] idx=161  new_lp=0.068  adv=-1.017  stored move_dir=-121.7°  new_mean=-120.8°
    per-head contributions: move_dir:0.186  sprint:-0.021  move:-0.022  tackle_attempt:-0.035  kick:-0.040
2026-08-07 20:03:56,340 INFO   [advantage] mean=-0.017  std=1.025  min=-5.723  max=4.883
2026-08-07 20:03:56,340 INFO   [ratio] mean=0.8834  std=0.3916  min=0.0000  max=9.9415  clipped=27.9%
2026-08-07 20:03:56,340 INFO   [exec head grad norm] move_direction=0.054  exec_move=0.053  sprint=0.064  kick=0.132  kick_direction=0.010  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.047
2026-08-07 20:03:56,340 INFO   [exec continuous log_std] move_direction: start=-2.7997 end=-2.7997   kick_direction: start=-2.7998 end=-2.7998
2026-08-07 20:03:56,340 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0014≈0.08°/step  epoch≈0.1°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0015≈0.09°/step  epoch≈0.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-07 20:03:56,340 INFO   [exec discrete Δlogit per opt step] exec_move=0.0030  sprint=0.0032  kick=0.0006  tackle_attempt=0.0014
2026-08-07 20:03:56,340 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0008  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0192  sprint=+0.0138  kick=+0.0044  tackle_attempt=+0.0014  move_dir=+0.2931  kick_dir=+0.1020
2026-08-07 20:03:56,341 INFO   [grad clip] main: 1/1 steps clipped (100%)  pre-clip norm mean=0.838 max=0.838  limit=0.4
              direction: 1/1 steps clipped (100%)  pre-clip norm mean=0.067 max=0.067  limit=0.02
2026-08-07 20:03:56,344 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=408,000  speed=821/s  reward=2.92
  loss     policy=0.0767  value=0.4876(x0.5)=0.2438
           entropy=1.3308  kl=0.4346
  value    V=3.38±1.32  R=3.41±1.77  adv=0.03±1.19
  moves    mv_ls=[-2.7997] (σ≈0.06, ≈3°) g=3.83e-02  d_move=[+0.0000] (Δσ≈0.000°)
           kk_ls=[-2.7998] (σ≈0.06, ≈3°)  d_kick=[+0.0000] (Δσ≈0.000°)
  heads    move= 43 get_poss= 57 exec_move= 91 sprint= 48 kick=  4 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0402 kick_prob=0.0432
  vs       vs[win/loss/tout/miss]  vs_immobile(521): 65.6%/0.0%/0.6%/11.3%/22%
  ep_len   13.7±8.4s  (n=521, min=0.2s, max=50.0s)
  reward   get_possession=+422.00  lose_possession=-3.60  ball_out=-20.00  box_possession=+855.00
           speed_bonus=+881.22  timeout=-4.50  stamina_penalty=-3.05
  rew/ep   (mean/std/min/max per episode, 521 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.810    0.411    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.007    0.079    -0.900    +0.000
  ball_out          -0.038    0.436    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.641    1.187    +0.000    +2.500
  speed_bonus       +1.691    1.571    +0.000    +4.378
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.009    0.113    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.026    +0.000
  rew/step (per-step stats, n=24000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     426    +0.018    0.132     +4.415     1.204     +0.414      1.5663      0.884     2.520
  lose_possession       4    -0.000    0.012     +3.444     0.420     -0.039      0.1523      0.332     0.575
  ball_out             4    -0.001    0.065     -4.750     0.433     -7.994     64.3568      7.994     8.668
  box_possession     342    +0.036    0.296     +5.068     1.212     +0.820      1.7246      1.117     2.227
  speed_bonus        326    +0.037    0.338     +5.194     1.096     +0.903      1.7517      1.124     2.267
  timeout              3    -0.000    0.017     -1.500     0.000     -2.676      8.0290      2.676     3.545
  stamina_penalty     326    -0.000    0.001     +5.104     1.192     +0.821      1.6628      1.109     2.220
  gae/td   mean_return=+3.409  std_return=1.773  mean_gae=+0.028  mean_sq_td=1.4264
──────────────────────────────────────────────────────────────────────
2026-08-07 20:03:56,369 INFO Saved checkpoint: checkpoints/phase1_run38/checkpoint17.pt
2026-08-07 20:03:56,369 INFO Logging to checkpoints/phase1_run38/training_log18.txt
2026-08-07 20:03:56,371 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 20:04:06,688 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 20:04:06,689 INFO   [eval vs immobile] step=408,000  seeds=16x8  win=59%  mean_rew=3.584±2.957  V=3.164  gap=-0.420  outcomes={'other': 35, 'box_possession': 75, 'miss': 18}
2026-08-07 20:04:06,690 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 20:04:16,986 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 20:04:16,988 INFO   [eval vs rules] step=408,000  seeds=16x8  win=23%  mean_rew=0.244±3.355  V=3.230  gap=+2.986  outcomes={'other': 38, 'box_possession': 29, 'opponent_box_possession': 53, 'miss': 8}
2026-08-07 20:04:47,968 INFO   [early stop e0 mb0]  KL=0.43891 > target=0.12  steps_this_update=1
    [per-head KL] exec_move=+0.0188  sprint=+0.0182  kick=+0.0036  tackle_attempt=+0.0019  move_dir=+0.3065  kick_dir=+0.0898
2026-08-07 20:04:47,971 INFO   [KL mean=0.4389 median=0.4389 > 0.05] ratio percentiles:  p5=0.115  p25=0.856  p50=0.989  p75=1.000  p95=1.081  max=12.940
  move_dir_log_std=[-2.799628973007202]  kick_dir_log_std=[-2.799769163131714]
2026-08-07 20:04:47,982 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.066  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.152  kick=-0.165  t_att=-0.174
    move_dir=3.021 (min=-0.047 max=3.761)  kick_dir=0.117 (min=0.000 max=3.752)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.26
  [worst sample] idx=0  ratio=28.699  adv=-0.562  old_lp=-3.460  new_lp=-0.103
    stored move_dir=-141.5°  new_mean=-142.7°  angular_diff=1.2°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx=   0  ratio=  28.699  adv=-0.562  lp: old=-3.460  new=-0.103
      rew=+0.0000  ret=+2.8311  val=+3.3934  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9651  sprint_p_new=0.0329  kick_p_new=0.0384  tackle_attempt_p_new=0.0397
    idx= 146  ratio=  27.600  adv=+1.222  lp: old=-3.417  new=-0.100
      rew=+0.0000  ret=+4.1241  val=+2.9021  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9637  sprint_p_new=0.0475  kick_p_new=0.0391  tackle_attempt_p_new=0.0380
  [best sample (highest new_lp)] idx=205  new_lp=0.065  adv=-0.598  stored move_dir=130.3°  new_mean=128.4°
    per-head contributions: move_dir:0.181  move:-0.022  sprint:-0.023  tackle_attempt:-0.032  kick:-0.039
2026-08-07 20:04:47,983 INFO   [advantage] mean=0.009  std=0.996  min=-6.807  max=4.166
2026-08-07 20:04:47,983 INFO   [ratio] mean=0.8812  std=0.3887  min=0.0000  max=12.9399  clipped=27.6%
2026-08-07 20:04:47,983 INFO   [exec head grad norm] move_direction=0.055  exec_move=0.033  sprint=0.059  kick=0.132  kick_direction=0.016  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.093
2026-08-07 20:04:47,983 INFO   [exec continuous log_std] move_direction: start=-2.7997 end=-2.7996   kick_direction: start=-2.7998 end=-2.7998
2026-08-07 20:04:47,983 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0012≈0.07°/step  epoch≈0.1°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0016≈0.09°/step  epoch≈0.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-07 20:04:47,983 INFO   [exec discrete Δlogit per opt step] exec_move=0.0027  sprint=0.0030  kick=0.0006  tackle_attempt=0.0013
2026-08-07 20:04:47,984 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0001  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0188  sprint=+0.0182  kick=+0.0036  tackle_attempt=+0.0019  move_dir=+0.3065  kick_dir=+0.0898
2026-08-07 20:04:47,984 INFO   [grad clip] main: 1/1 steps clipped (100%)  pre-clip norm mean=0.664 max=0.664  limit=0.4
              direction: 1/1 steps clipped (100%)  pre-clip norm mean=0.071 max=0.071  limit=0.02
2026-08-07 20:04:48,027 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=432,000  speed=811/s  reward=2.73
  loss     policy=0.0501  value=0.4563(x0.5)=0.2282
           entropy=1.3317  kl=0.4389
  value    V=3.28±1.26  R=3.36±1.77  adv=0.08±1.20
  moves    mv_ls=[-2.7996] (σ≈0.06, ≈3°) g=4.20e-02  d_move=[+0.0000] (Δσ≈0.000°)
           kk_ls=[-2.7998] (σ≈0.06, ≈3°)  d_kick=[+0.0000] (Δσ≈0.000°)
  heads    move= 42 get_poss= 58 exec_move= 91 sprint= 48 kick=  4 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0398 kick_prob=0.0431
  vs       vs[win/loss/tout/miss]  vs_immobile(538): 68.0%/0.0%/0.2%/11.5%/20%
  ep_len   13.2±7.7s  (n=538, min=0.4s, max=50.0s)
  reward   get_possession=+449.00  lose_possession=-1.80  ball_out=-40.00  box_possession=+915.00
           speed_bonus=+891.62  timeout=-1.50  stamina_penalty=-3.08
  rew/ep   (mean/std/min/max per episode, 538 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.835    0.381    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.003    0.055    -0.900    +0.000
  ball_out          -0.074    0.605    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.701    1.166    +0.000    +2.500
  speed_bonus       +1.657    1.548    +0.000    +4.386
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.003    0.065    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.027    +0.000
  rew/step (per-step stats, n=24000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     453    +0.019    0.136     +4.238     1.466     +0.395      1.8887      1.004     2.635
  lose_possession       2    -0.000    0.008     +3.768     0.351     -0.105      0.2902      0.528     0.623
  ball_out             8    -0.002    0.091     -4.750     0.433     -7.472     57.1931      7.472     8.392
  box_possession     366    +0.038    0.306     +4.930     1.267     +0.808      1.9329      1.160     2.397
  speed_bonus        351    +0.037    0.337     +5.032     1.193     +0.887      1.9509      1.163     2.436
  timeout              1    -0.000    0.010     -1.500     0.000     -1.317      1.7332      1.317     1.317
  stamina_penalty     342    -0.000    0.001     +5.013     1.226     +0.867      1.9677      1.176     2.406
  gae/td   mean_return=+3.355  std_return=1.772  mean_gae=+0.080  mean_sq_td=1.4346
──────────────────────────────────────────────────────────────────────
2026-08-07 20:04:48,050 INFO Saved checkpoint: checkpoints/phase1_run38/checkpoint18.pt
2026-08-07 20:04:48,050 INFO Logging to checkpoints/phase1_run38/training_log19.txt
2026-08-07 20:04:48,052 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 20:04:58,457 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 20:04:58,459 INFO   [eval vs immobile] step=432,000  seeds=16x8  win=56%  mean_rew=3.413±2.979  V=3.086  gap=-0.327  outcomes={'other': 34, 'box_possession': 72, 'timeout': 2, 'miss': 20}
2026-08-07 20:04:58,460 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 20:05:08,902 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 20:05:08,903 INFO   [eval vs rules] step=432,000  seeds=16x8  win=25%  mean_rew=0.290±3.499  V=3.138  gap=+2.849  outcomes={'other': 32, 'opponent_box_possession': 55, 'box_possession': 32, 'timeout': 1, 'miss': 8}
2026-08-07 20:05:39,867 INFO   [early stop e0 mb0]  KL=0.40608 > target=0.12  steps_this_update=1
    [per-head KL] exec_move=+0.0174  sprint=+0.0159  kick=+0.0034  move_dir=+0.2640  kick_dir=+0.1053
2026-08-07 20:05:39,869 INFO   [KL mean=0.4061 median=0.4061 > 0.05] ratio percentiles:  p5=0.149  p25=0.874  p50=0.992  p75=1.000  p95=1.087  max=7.316
  move_dir_log_std=[-2.799607276916504]  kick_dir_log_std=[-2.799762487411499]
2026-08-07 20:05:39,882 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.035  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.174  kick=-0.117  t_att=-0.170
    move_dir=3.007 (min=-1.214 max=3.761)  kick_dir=0.052 (min=-0.165 max=3.753)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.18
  [worst sample] idx=254  ratio=29.061  adv=+0.512  old_lp=-3.463  new_lp=-0.093
    stored move_dir=5.0°  new_mean=10.7°  angular_diff=5.7°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 254  ratio=  29.061  adv=+0.512  lp: old=-3.463  new=-0.093
      rew=+0.0000  ret=+4.3893  val=+3.8775  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9656  sprint_p_new=0.9641  kick_p_new=0.0379  tackle_attempt_p_new=0.0360
    idx= 247  ratio=  27.155  adv=+0.489  lp: old=-3.399  new=-0.097
      rew=+0.0000  ret=+4.1304  val=+3.6411  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9632  sprint_p_new=0.9615  kick_p_new=0.0385  tackle_attempt_p_new=0.0373
  [best sample (highest new_lp)] idx=51  new_lp=0.070  adv=-3.347  stored move_dir=127.9°  new_mean=128.9°
    per-head contributions: move_dir:0.186  move:-0.020  sprint:-0.025  tackle_attempt:-0.031  kick:-0.040
2026-08-07 20:05:39,882 INFO   [advantage] mean=0.002  std=1.012  min=-6.998  max=4.353
2026-08-07 20:05:39,882 INFO   [ratio] mean=0.8926  std=0.3474  min=0.0000  max=7.3156  clipped=26.0%
2026-08-07 20:05:39,882 INFO   [exec head grad norm] move_direction=0.081  exec_move=0.050  sprint=0.100  kick=0.042  kick_direction=0.013  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.116
2026-08-07 20:05:39,883 INFO   [exec continuous log_std] move_direction: start=-2.7996 end=-2.7996   kick_direction: start=-2.7998 end=-2.7998
2026-08-07 20:05:39,883 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0014≈0.08°/step  epoch≈0.1°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0015≈0.09°/step  epoch≈0.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-07 20:05:39,883 INFO   [exec discrete Δlogit per opt step] exec_move=0.0026  sprint=0.0028  kick=0.0006  tackle_attempt=0.0013
2026-08-07 20:05:39,883 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0001  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0174  sprint=+0.0159  kick=+0.0034  tackle_attempt=-0.0000  move_dir=+0.2640  kick_dir=+0.1053
2026-08-07 20:05:39,883 INFO   [grad clip] main: 1/1 steps clipped (100%)  pre-clip norm mean=0.672 max=0.672  limit=0.4
              direction: 1/1 steps clipped (100%)  pre-clip norm mean=0.090 max=0.090  limit=0.02
2026-08-07 20:05:39,916 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=456,000  speed=812/s  reward=4.09
  loss     policy=0.0528  value=0.4523(x0.5)=0.2262
           entropy=1.3276  kl=0.4061
  value    V=3.29±1.27  R=3.35±1.77  adv=0.05±1.17
  moves    mv_ls=[-2.7996] (σ≈0.06, ≈3°) g=3.80e-02  d_move=[+0.0000] (Δσ≈0.000°)
           kk_ls=[-2.7998] (σ≈0.06, ≈3°)  d_kick=[+0.0000] (Δσ≈0.000°)
  heads    move= 43 get_poss= 57 exec_move= 91 sprint= 49 kick=  4 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0403 kick_prob=0.0430
  vs       vs[win/loss/tout/miss]  vs_immobile(539): 65.7%/0.0%/0.7%/12.2%/21%
  ep_len   13.3±8.2s  (n=539, min=0.9s, max=50.0s)
  reward   get_possession=+450.00  lose_possession=-2.70  ball_out=-15.00  box_possession=+885.00
           speed_bonus=+898.37  timeout=-6.00  stamina_penalty=-3.16
  rew/ep   (mean/std/min/max per episode, 539 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.835    0.386    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.005    0.067    -0.900    +0.000
  ball_out          -0.028    0.372    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.642    1.187    +0.000    +2.500
  speed_bonus       +1.667    1.553    +0.000    +4.293
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.011    0.129    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.029    +0.000
  rew/step (per-step stats, n=24000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     454    +0.019    0.136     +4.304     1.255     +0.446      1.5322      0.955     2.367
  lose_possession       3    -0.000    0.010     +3.880     0.665     +0.298      0.9372      0.767     1.481
  ball_out             3    -0.001    0.056     -4.333     0.471     -6.309     42.7576      6.309     8.020
  box_possession     354    +0.037    0.301     +5.029     1.206     +0.805      1.6385      1.107     2.198
  speed_bonus        340    +0.037    0.339     +5.133     1.113     +0.879      1.6541      1.111     2.210
  timeout              4    -0.000    0.019     -1.500     0.000     -2.965      8.8848      2.965     3.389
  stamina_penalty     338    -0.000    0.001     +5.069     1.185     +0.829      1.6512      1.111     2.211
  gae/td   mean_return=+3.346  std_return=1.772  mean_gae=+0.052  mean_sq_td=1.3798
──────────────────────────────────────────────────────────────────────
2026-08-07 20:05:39,947 INFO Saved checkpoint: checkpoints/phase1_run38/checkpoint19.pt
2026-08-07 20:05:39,948 INFO Logging to checkpoints/phase1_run38/training_log20.txt
2026-08-07 20:05:39,950 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 20:05:50,176 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 20:05:50,178 INFO   [eval vs immobile] step=456,000  seeds=16x8  win=59%  mean_rew=3.579±3.009  V=3.160  gap=-0.419  outcomes={'other': 33, 'box_possession': 76, 'miss': 19}
2026-08-07 20:05:50,180 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 20:06:01,302 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 20:06:01,303 INFO   [eval vs rules] step=456,000  seeds=16x8  win=26%  mean_rew=0.421±3.512  V=3.109  gap=+2.688  outcomes={'other': 35, 'box_possession': 33, 'opponent_box_possession': 50, 'miss': 9, 'timeout': 1}
2026-08-07 20:06:32,100 INFO   [early stop e0 mb0]  KL=0.38540 > target=0.12  steps_this_update=1
    [per-head KL] exec_move=+0.0215  sprint=+0.0097  kick=+0.0064  tackle_attempt=+0.0019  move_dir=+0.2459  kick_dir=+0.0997
2026-08-07 20:06:32,102 INFO   [KL mean=0.3854 median=0.3854 > 0.05] ratio percentiles:  p5=0.154  p25=0.872  p50=0.993  p75=1.000  p95=1.090  max=17.129
  move_dir_log_std=[-2.799586057662964]  kick_dir_log_std=[-2.7997565269470215]
2026-08-07 20:06:32,114 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.081  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.207  kick=-0.196  t_att=-0.201
    move_dir=2.957 (min=-3.506 max=3.761)  kick_dir=0.180 (min=0.000 max=3.751)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.25
  [worst sample] idx=164  ratio=23.230  adv=-0.273  old_lp=-3.242  new_lp=-0.096
    stored move_dir=178.7°  new_mean=-178.3°  angular_diff=3.0°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 164  ratio=  23.230  adv=-0.273  lp: old=-3.242  new=-0.096
      rew=+0.0000  ret=+3.6390  val=+3.9118  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9570  sprint_p_new=0.9562  kick_p_new=0.0385  tackle_attempt_p_new=0.0338
    idx= 187  ratio=  22.916  adv=+0.327  lp: old=-3.228  new=-0.097
      rew=+0.0000  ret=+4.0356  val=+3.7085  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9563  sprint_p_new=0.0261  kick_p_new=0.0386  tackle_attempt_p_new=0.0343
  [best sample (highest new_lp)] idx=93  new_lp=0.065  adv=-2.158  stored move_dir=90.0°  new_mean=89.7°
    per-head contributions: move_dir:0.188  move:-0.023  sprint:-0.031  tackle_attempt:-0.033  kick:-0.036
2026-08-07 20:06:32,114 INFO   [advantage] mean=-0.014  std=0.998  min=-5.078  max=5.144
2026-08-07 20:06:32,115 INFO   [ratio] mean=0.8941  std=0.3993  min=0.0000  max=17.1285  clipped=26.7%
2026-08-07 20:06:32,115 INFO   [exec head grad norm] move_direction=0.062  exec_move=0.090  sprint=0.079  kick=0.064  kick_direction=0.030  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.150
2026-08-07 20:06:32,115 INFO   [exec continuous log_std] move_direction: start=-2.7996 end=-2.7996   kick_direction: start=-2.7998 end=-2.7998
2026-08-07 20:06:32,115 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0013≈0.07°/step  epoch≈0.1°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0015≈0.09°/step  epoch≈0.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-07 20:06:32,115 INFO   [exec discrete Δlogit per opt step] exec_move=0.0024  sprint=0.0026  kick=0.0006  tackle_attempt=0.0013
2026-08-07 20:06:32,115 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0002  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0215  sprint=+0.0097  kick=+0.0064  tackle_attempt=+0.0019  move_dir=+0.2459  kick_dir=+0.0997
2026-08-07 20:06:32,116 INFO   [grad clip] main: 1/1 steps clipped (100%)  pre-clip norm mean=1.164 max=1.164  limit=0.4
              direction: 1/1 steps clipped (100%)  pre-clip norm mean=0.076 max=0.076  limit=0.02
2026-08-07 20:06:32,122 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=480,000  speed=817/s  reward=4.27
  loss     policy=0.0672  value=0.4658(x0.5)=0.2329
           entropy=1.3290  kl=0.3854
  value    V=3.36±1.23  R=3.41±1.74  adv=0.05±1.19
  moves    mv_ls=[-2.7996] (σ≈0.06, ≈3°) g=3.17e-02  d_move=[+0.0000] (Δσ≈0.000°)
           kk_ls=[-2.7998] (σ≈0.06, ≈3°)  d_kick=[+0.0000] (Δσ≈0.000°)
  heads    move= 42 get_poss= 58 exec_move= 91 sprint= 48 kick=  4 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0401 kick_prob=0.0444
  vs       vs[win/loss/tout/miss]  vs_immobile(530): 67.5%/0.0%/0.2%/11.7%/21%
  ep_len   13.4±7.7s  (n=530, min=0.5s, max=50.0s)
  reward   get_possession=+446.00  lose_possession=-1.80  ball_out=-30.00  box_possession=+895.00
           speed_bonus=+877.38  timeout=-1.50  stamina_penalty=-3.06
  rew/ep   (mean/std/min/max per episode, 530 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.842    0.375    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.003    0.055    -0.900    +0.000
  ball_out          -0.057    0.529    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.689    1.170    +0.000    +2.500
  speed_bonus       +1.655    1.546    +0.000    +4.268
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.003    0.065    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.028    +0.000
  rew/step (per-step stats, n=24000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     452    +0.019    0.136     +4.309     1.401     +0.504      1.9566      1.056     2.554
  lose_possession       2    -0.000    0.008     +3.875     0.760     +0.465      0.9578      0.861     1.280
  ball_out             6    -0.001    0.079     -4.833     0.373     -7.802     61.3261      7.802     8.375
  box_possession     358    +0.037    0.303     +4.942     1.259     +0.537      1.3805      1.006     2.013
  speed_bonus        342    +0.037    0.335     +5.057     1.169     +0.616      1.3711      0.999     2.013
  timeout              1    -0.000    0.010     -1.500     0.000     -1.367      1.8693      1.367     1.367
  stamina_penalty     340    -0.000    0.001     +4.993     1.241     +0.563      1.3801      1.005     2.013
  gae/td   mean_return=+3.409  std_return=1.738  mean_gae=+0.047  mean_sq_td=1.4093
──────────────────────────────────────────────────────────────────────
2026-08-07 20:06:32,145 INFO Saved checkpoint: checkpoints/phase1_run38/checkpoint20.pt
2026-08-07 20:06:32,146 INFO Logging to checkpoints/phase1_run38/training_log21.txt
2026-08-07 20:06:32,146 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 20:06:43,064 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 20:06:43,065 INFO   [eval vs immobile] step=480,000  seeds=16x8  win=55%  mean_rew=3.417±2.952  V=3.204  gap=-0.213  outcomes={'other': 42, 'box_possession': 70, 'miss': 16}
2026-08-07 20:06:43,066 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 20:06:54,629 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 20:06:54,631 INFO   [eval vs rules] step=480,000  seeds=16x8  win=23%  mean_rew=0.020±3.421  V=3.094  gap=+3.073  outcomes={'opponent_box_possession': 59, 'other': 27, 'timeout': 2, 'box_possession': 30, 'miss': 10}
2026-08-07 20:07:25,872 INFO   [early stop e0 mb0]  KL=0.42050 > target=0.12  steps_this_update=1
    [per-head KL] exec_move=+0.0200  sprint=+0.0147  kick=+0.0057  move_dir=+0.2649  kick_dir=+0.1144
2026-08-07 20:07:25,875 INFO   [KL mean=0.4205 median=0.4205 > 0.05] ratio percentiles:  p5=0.165  p25=0.870  p50=0.992  p75=1.000  p95=1.087  max=6.059
  move_dir_log_std=[-2.7995641231536865]  kick_dir_log_std=[-2.799750804901123]
2026-08-07 20:07:25,887 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.065  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.159  kick=-0.114  t_att=-0.163
    move_dir=2.980 (min=-2.184 max=3.761)  kick_dir=0.081 (min=0.000 max=3.761)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.19
  [worst sample] idx=231  ratio=28.689  adv=+1.017  old_lp=-3.446  new_lp=-0.089
    stored move_dir=-176.6°  new_mean=-176.2°  angular_diff=0.4°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 231  ratio=  28.689  adv=+1.017  lp: old=-3.446  new=-0.089
      rew=+0.0000  ret=+5.1422  val=+4.1252  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9652  sprint_p_new=0.9651  kick_p_new=0.0366  tackle_attempt_p_new=0.0349
    idx= 127  ratio=  26.784  adv=+0.031  lp: old=-3.380  new=-0.092
      rew=+0.0000  ret=+3.4654  val=+3.4348  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9626  sprint_p_new=0.9614  kick_p_new=0.0386  tackle_attempt_p_new=0.0307
  [best sample (highest new_lp)] idx=138  new_lp=0.070  adv=+0.012  stored move_dir=-159.6°  new_mean=-159.9°
    per-head contributions: move_dir:0.188  move:-0.021  sprint:-0.023  kick:-0.037  tackle_attempt:-0.037
2026-08-07 20:07:25,887 INFO   [advantage] mean=-0.005  std=1.008  min=-7.009  max=4.418
2026-08-07 20:07:25,887 INFO   [ratio] mean=0.8835  std=0.3055  min=0.0000  max=6.0590  clipped=26.9%
2026-08-07 20:07:25,887 INFO   [exec head grad norm] move_direction=0.026  exec_move=0.048  sprint=0.053  kick=0.077  kick_direction=0.018  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.054
2026-08-07 20:07:25,887 INFO   [exec continuous log_std] move_direction: start=-2.7996 end=-2.7996   kick_direction: start=-2.7998 end=-2.7998
2026-08-07 20:07:25,888 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0012≈0.07°/step  epoch≈0.1°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0016≈0.09°/step  epoch≈0.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-07 20:07:25,888 INFO   [exec discrete Δlogit per opt step] exec_move=0.0024  sprint=0.0027  kick=0.0006  tackle_attempt=0.0013
2026-08-07 20:07:25,888 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0001  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0200  sprint=+0.0147  kick=+0.0057  tackle_attempt=+0.0007  move_dir=+0.2649  kick_dir=+0.1144
2026-08-07 20:07:25,888 INFO   [grad clip] main: 1/1 steps clipped (100%)  pre-clip norm mean=0.508 max=0.508  limit=0.4
              direction: 1/1 steps clipped (100%)  pre-clip norm mean=0.043 max=0.043  limit=0.02
2026-08-07 20:07:25,892 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=504,000  speed=806/s  reward=4.16
  loss     policy=0.0402  value=0.4646(x0.5)=0.2323
           entropy=1.3294  kl=0.4205
  value    V=3.49±1.23  R=3.48±1.77  adv=-0.01±1.20
  moves    mv_ls=[-2.7996] (σ≈0.06, ≈3°) g=2.96e-02  d_move=[+0.0000] (Δσ≈0.000°)
           kk_ls=[-2.7998] (σ≈0.06, ≈3°)  d_kick=[+0.0000] (Δσ≈0.000°)
  heads    move= 43 get_poss= 57 exec_move= 91 sprint= 48 kick=  5 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0404 kick_prob=0.0448
  vs       vs[win/loss/tout/miss]  vs_immobile(531): 67.0%/0.2%/0.0%/11.9%/21%
  ep_len   13.5±7.8s  (n=531, min=0.2s, max=47.1s)
  reward   get_possession=+442.00  lose_possession=-4.50  ball_out=-45.00  box_possession=+890.00
           speed_bonus=+901.82  opponent_box=-3.00  stamina_penalty=-3.17
  rew/ep   (mean/std/min/max per episode, 531 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.832    0.393    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.008    0.087    -0.900    +0.000
  ball_out          -0.085    0.645    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.676    1.175    +0.000    +2.500
  speed_bonus       +1.698    1.517    +0.000    +4.279
  opponent_box      -0.006    0.130    -3.000    +0.000
  timeout           +0.000    0.000    +0.000    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.025    +0.000
  rew/step (per-step stats, n=24000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     446    +0.019    0.135     +4.328     1.407     +0.446      2.1002      1.059     2.228
  lose_possession       5    -0.000    0.013     +3.278     0.401     -0.232      0.2974      0.420     0.925
  ball_out             9    -0.002    0.097     -4.889     0.314     -6.445     46.7621      6.445     8.964
  box_possession     356    +0.037    0.302     +5.027     1.144     +0.397      1.0365      0.840     1.857
  speed_bonus        347    +0.038    0.337     +5.093     1.083     +0.442      1.0127      0.827     1.728
  opponent_box         1    -0.000    0.019     -3.002     0.000     -6.157     37.9073      6.157     6.157
  stamina_penalty     345    -0.000    0.001     +5.014     1.218     +0.369      1.0839      0.843     1.824
  gae/td   mean_return=+3.481  std_return=1.768  mean_gae=-0.012  mean_sq_td=1.4311
──────────────────────────────────────────────────────────────────────
2026-08-07 20:07:25,915 INFO Saved checkpoint: checkpoints/phase1_run38/checkpoint21.pt
2026-08-07 20:07:25,916 INFO Logging to checkpoints/phase1_run38/training_log22.txt
2026-08-07 20:07:25,917 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 20:07:36,724 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 20:07:36,726 INFO   [eval vs immobile] step=504,000  seeds=16x8  win=56%  mean_rew=3.438±2.967  V=3.259  gap=-0.179  outcomes={'other': 37, 'box_possession': 72, 'timeout': 2, 'miss': 17}
2026-08-07 20:07:36,727 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 20:07:47,495 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 20:07:47,497 INFO   [eval vs rules] step=504,000  seeds=16x8  win=26%  mean_rew=0.368±3.531  V=3.251  gap=+2.882  outcomes={'other': 33, 'box_possession': 33, 'opponent_box_possession': 53, 'miss': 9}
2026-08-07 20:08:18,423 INFO   [early stop e0 mb0]  KL=0.40510 > target=0.12  steps_this_update=1
    [per-head KL] exec_move=+0.0186  sprint=+0.0134  kick=+0.0040  move_dir=+0.2608  kick_dir=+0.1085
2026-08-07 20:08:18,425 INFO   [KL mean=0.4051 median=0.4051 > 0.05] ratio percentiles:  p5=0.174  p25=0.868  p50=0.994  p75=1.000  p95=1.086  max=6.957
  move_dir_log_std=[-2.799541711807251]  kick_dir_log_std=[-2.7997453212738037]
2026-08-07 20:08:18,438 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.051  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.216  kick=-0.139  t_att=-0.175
    move_dir=2.938 (min=-0.227 max=3.761)  kick_dir=0.102 (min=0.000 max=3.758)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.23
  [worst sample] idx=113  ratio=31.262  adv=+0.608  old_lp=-3.538  new_lp=-0.095
    stored move_dir=-130.4°  new_mean=-131.7°  angular_diff=1.2°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 113  ratio=  31.262  adv=+0.608  lp: old=-3.538  new=-0.095
      rew=+0.0000  ret=+5.1258  val=+4.5180  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9680  sprint_p_new=0.0305  kick_p_new=0.0379  tackle_attempt_p_new=0.0333
    idx= 186  ratio=  29.766  adv=-0.153  lp: old=-3.490  new=-0.097
      rew=+0.0000  ret=+3.4582  val=+3.6115  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9664  sprint_p_new=0.9649  kick_p_new=0.0373  tackle_attempt_p_new=0.0372
  [best sample (highest new_lp)] idx=16  new_lp=0.084  adv=-0.831  stored move_dir=126.0°  new_mean=127.0°
    per-head contributions: move_dir:0.186  tackle_attempt:-0.022  move:-0.023  kick:-0.042
2026-08-07 20:08:18,438 INFO   [advantage] mean=0.023  std=0.975  min=-5.074  max=4.056
2026-08-07 20:08:18,438 INFO   [ratio] mean=0.8909  std=0.3111  min=0.0000  max=6.9571  clipped=26.3%
2026-08-07 20:08:18,439 INFO   [exec head grad norm] move_direction=0.030  exec_move=0.046  sprint=0.054  kick=0.103  kick_direction=0.011  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.077
2026-08-07 20:08:18,439 INFO   [exec continuous log_std] move_direction: start=-2.7996 end=-2.7995   kick_direction: start=-2.7998 end=-2.7997
2026-08-07 20:08:18,439 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0012≈0.07°/step  epoch≈0.1°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0016≈0.09°/step  epoch≈0.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-07 20:08:18,439 INFO   [exec discrete Δlogit per opt step] exec_move=0.0025  sprint=0.0028  kick=0.0006  tackle_attempt=0.0013
2026-08-07 20:08:18,439 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=-0.0002  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0186  sprint=+0.0134  kick=+0.0040  tackle_attempt=-0.0000  move_dir=+0.2608  kick_dir=+0.1085
2026-08-07 20:08:18,440 INFO   [grad clip] main: 1/1 steps clipped (100%)  pre-clip norm mean=0.594 max=0.594  limit=0.4
              direction: 1/1 steps clipped (100%)  pre-clip norm mean=0.042 max=0.042  limit=0.02
2026-08-07 20:08:18,445 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=528,000  speed=812/s  reward=4.34
  loss     policy=0.0109  value=0.4801(x0.5)=0.2401
           entropy=1.3271  kl=0.4051
  value    V=3.54±1.27  R=3.44±1.75  adv=-0.10±1.23
  moves    mv_ls=[-2.7995] (σ≈0.06, ≈3°) g=2.70e-02  d_move=[+0.0000] (Δσ≈0.000°)
           kk_ls=[-2.7997] (σ≈0.06, ≈3°)  d_kick=[+0.0000] (Δσ≈0.000°)
  heads    move= 42 get_poss= 58 exec_move= 91 sprint= 48 kick=  4 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0407 kick_prob=0.0431
  vs       vs[win/loss/tout/miss]  vs_immobile(543): 66.9%/0.0%/0.2%/9.9%/23%
  ep_len   13.1±8.0s  (n=543, min=0.9s, max=50.0s)
  reward   get_possession=+457.00  lose_possession=-0.90  ball_out=-30.00  box_possession=+907.50
           speed_bonus=+863.80  timeout=-1.50  stamina_penalty=-3.01
  rew/ep   (mean/std/min/max per episode, 543 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.842    0.370    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.002    0.039    -0.900    +0.000
  ball_out          -0.055    0.523    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.671    1.177    +0.000    +2.500
  speed_bonus       +1.591    1.532    +0.000    +4.328
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.003    0.064    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.028    +0.000
  rew/step (per-step stats, n=24000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     460    +0.019    0.137     +4.362     1.395     +0.392      1.9328      1.015     2.456
  lose_possession       1    -0.000    0.006     +3.145     0.000     +0.041      0.0017      0.041     0.041
  ball_out             6    -0.001    0.079     -4.833     0.373     -7.892     62.6463      7.892     8.599
  box_possession     363    +0.038    0.305     +4.871     1.276     +0.117      0.9895      0.822     1.677
  speed_bonus        349    +0.036    0.330     +4.967     1.207     +0.171      0.9585      0.805     1.647
  timeout              1    -0.000    0.010     -1.500     0.000     -4.597     21.1309      4.597     4.597
  stamina_penalty     343    -0.000    0.001     +4.919     1.253     +0.139      0.9389      0.812     1.644
  gae/td   mean_return=+3.441  std_return=1.754  mean_gae=-0.099  mean_sq_td=1.5139
──────────────────────────────────────────────────────────────────────
2026-08-07 20:08:18,483 INFO Saved checkpoint: checkpoints/phase1_run38/checkpoint22.pt
2026-08-07 20:08:18,484 INFO Logging to checkpoints/phase1_run38/training_log23.txt
2026-08-07 20:08:18,486 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 20:08:30,078 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 20:08:30,079 INFO   [eval vs immobile] step=528,000  seeds=16x8  win=56%  mean_rew=3.481±2.974  V=3.277  gap=-0.204  outcomes={'other': 36, 'timeout': 1, 'box_possession': 72, 'miss': 19}
2026-08-07 20:08:30,081 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 20:08:40,863 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 20:08:40,865 INFO   [eval vs rules] step=528,000  seeds=16x8  win=31%  mean_rew=0.625±3.664  V=3.183  gap=+2.558  outcomes={'box_possession': 40, 'other': 26, 'opponent_box_possession': 52, 'miss': 9, 'timeout': 1}
2026-08-07 20:09:11,807 INFO   [early stop e0 mb0]  KL=0.33796 > target=0.12  steps_this_update=1
    [per-head KL] exec_move=+0.0146  sprint=+0.0143  kick=+0.0027  move_dir=+0.2210  kick_dir=+0.0855
2026-08-07 20:09:11,809 INFO   [KL mean=0.3380 median=0.3380 > 0.05] ratio percentiles:  p5=0.192  p25=0.880  p50=0.994  p75=1.000  p95=1.091  max=8.560
  move_dir_log_std=[-2.7995195388793945]  kick_dir_log_std=[-2.7997398376464844]
2026-08-07 20:09:11,820 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.097  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.220  kick=-0.116  t_att=-0.160
    move_dir=2.942 (min=0.000 max=3.761)  kick_dir=0.082 (min=0.000 max=3.752)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.25
  [worst sample] idx=85  ratio=24.241  adv=+0.385  old_lp=-3.297  new_lp=-0.109
    stored move_dir=16.0°  new_mean=15.7°  angular_diff=0.3°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx=  85  ratio=  24.241  adv=+0.385  lp: old=-3.297  new=-0.109
      rew=+0.0000  ret=+5.2470  val=+4.8620  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9587  sprint_p_new=0.0442  kick_p_new=0.0381  tackle_attempt_p_new=0.0484
    idx=  86  ratio=  24.232  adv=+0.430  lp: old=-6.263  new=-3.075
      rew=+0.0000  ret=+5.2863  val=+4.8567  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9587  sprint_p_new=0.0435  kick_p_new=0.0380  tackle_attempt_p_new=0.0490
  [best sample (highest new_lp)] idx=33  new_lp=0.078  adv=-0.237  stored move_dir=-58.2°  new_mean=-58.0°
    per-head contributions: move_dir:0.188  move:-0.022  tackle_attempt:-0.029  kick:-0.040
2026-08-07 20:09:11,821 INFO   [advantage] mean=-0.002  std=1.019  min=-6.962  max=4.310
2026-08-07 20:09:11,821 INFO   [ratio] mean=0.8991  std=0.3540  min=0.0000  max=8.5599  clipped=25.9%
2026-08-07 20:09:11,821 INFO   [exec head grad norm] move_direction=0.054  exec_move=0.103  sprint=0.061  kick=0.040  kick_direction=0.019  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.118
2026-08-07 20:09:11,821 INFO   [exec continuous log_std] move_direction: start=-2.7995 end=-2.7995   kick_direction: start=-2.7997 end=-2.7997
2026-08-07 20:09:11,821 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0011≈0.06°/step  epoch≈0.1°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0016≈0.09°/step  epoch≈0.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-07 20:09:11,822 INFO   [exec discrete Δlogit per opt step] exec_move=0.0021  sprint=0.0023  kick=0.0006  tackle_attempt=0.0014
2026-08-07 20:09:11,822 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0004  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0146  sprint=+0.0143  kick=+0.0027  tackle_attempt=-0.0004  move_dir=+0.2210  kick_dir=+0.0855
2026-08-07 20:09:11,822 INFO   [grad clip] main: 1/1 steps clipped (100%)  pre-clip norm mean=0.451 max=0.451  limit=0.4
              direction: 1/1 steps clipped (100%)  pre-clip norm mean=0.065 max=0.065  limit=0.02
2026-08-07 20:09:11,827 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=552,000  speed=813/s  reward=3.84
  loss     policy=0.0353  value=0.4500(x0.5)=0.2250
           entropy=1.3318  kl=0.3380
  value    V=3.49±1.38  R=3.48±1.76  adv=-0.01±1.16
  moves    mv_ls=[-2.7995] (σ≈0.06, ≈3°) g=3.03e-02  d_move=[+0.0000] (Δσ≈0.000°)
           kk_ls=[-2.7997] (σ≈0.06, ≈3°)  d_kick=[+0.0000] (Δσ≈0.000°)
  heads    move= 41 get_poss= 59 exec_move= 91 sprint= 48 kick=  4 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0409 kick_prob=0.0431
  vs       vs[win/loss/tout/miss]  vs_immobile(559): 69.9%/0.0%/0.4%/9.8%/20%
  ep_len   12.8±8.0s  (n=559, min=0.2s, max=50.0s)
  reward   get_possession=+472.00  lose_possession=-4.50  ball_out=-30.00  box_possession=+977.50
           speed_bonus=+872.68  timeout=-3.00  stamina_penalty=-3.07
  rew/ep   (mean/std/min/max per episode, 559 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.844    0.382    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.008    0.085    -0.900    +0.000
  ball_out          -0.054    0.515    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.749    1.146    +0.000    +2.500
  speed_bonus       +1.561    1.484    +0.000    +4.347
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.005    0.090    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.005    0.006    -0.027    +0.000
  rew/step (per-step stats, n=24000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     476    +0.020    0.139     +4.402     1.354     +0.514      1.6810      0.988     2.346
  lose_possession       5    -0.000    0.013     +2.386     1.692     -0.922      1.8738      1.100     2.347
  ball_out             6    -0.001    0.079     -4.833     0.373     -7.109     53.3129      7.109     8.522
  box_possession     391    +0.041    0.316     +4.732     1.277     -0.024      1.1355      0.862     1.952
  speed_bonus        374    +0.036    0.327     +4.828     1.219     +0.037      1.0549      0.828     1.780
  timeout              2    -0.000    0.014     -1.500     0.000     -0.723      0.5233      0.723     0.752
  stamina_penalty     363    -0.000    0.001     +4.808     1.250     -0.002      1.0826      0.844     1.770
  gae/td   mean_return=+3.476  std_return=1.764  mean_gae=-0.014  mean_sq_td=1.3403
──────────────────────────────────────────────────────────────────────
2026-08-07 20:09:11,854 INFO Saved checkpoint: checkpoints/phase1_run38/checkpoint23.pt
2026-08-07 20:09:11,854 INFO Logging to checkpoints/phase1_run38/training_log24.txt
2026-08-07 20:09:11,856 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 20:09:22,788 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 20:09:22,790 INFO   [eval vs immobile] step=552,000  seeds=16x8  win=55%  mean_rew=3.339±2.928  V=3.197  gap=-0.142  outcomes={'other': 37, 'box_possession': 71, 'timeout': 2, 'miss': 18}
2026-08-07 20:09:22,791 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 20:09:35,279 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 20:09:35,281 INFO   [eval vs rules] step=552,000  seeds=16x8  win=26%  mean_rew=0.271±3.547  V=3.151  gap=+2.881  outcomes={'other': 30, 'box_possession': 33, 'opponent_box_possession': 54, 'miss': 10, 'timeout': 1}
2026-08-07 20:10:06,066 INFO   [early stop e0 mb0]  KL=0.34585 > target=0.12  steps_this_update=1
    [per-head KL] exec_move=+0.0173  sprint=+0.0124  kick=+0.0050  move_dir=+0.2243  kick_dir=+0.0858
2026-08-07 20:10:06,069 INFO   [KL mean=0.3458 median=0.3458 > 0.05] ratio percentiles:  p5=0.181  p25=0.871  p50=0.991  p75=1.000  p95=1.086  max=14.690
  move_dir_log_std=[-2.799497365951538]  kick_dir_log_std=[-2.799734592437744]
2026-08-07 20:10:06,079 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.113  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.202  kick=-0.142  t_att=-0.181
    move_dir=2.927 (min=-1.041 max=3.761)  kick_dir=0.105 (min=0.000 max=3.760)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.21
  [worst sample] idx=146  ratio=28.305  adv=+0.533  old_lp=-3.441  new_lp=-0.098
    stored move_dir=127.6°  new_mean=131.7°  angular_diff=4.2°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 146  ratio=  28.305  adv=+0.533  lp: old=-3.441  new=-0.098
      rew=+0.0000  ret=+4.5040  val=+3.9709  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9647  sprint_p_new=0.9631  kick_p_new=0.0390  tackle_attempt_p_new=0.9627
    idx=  39  ratio=  26.682  adv=+0.351  lp: old=-3.383  new=-0.099
      rew=+0.0000  ret=+4.0944  val=+3.7436  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9625  sprint_p_new=0.0283  kick_p_new=0.0384  tackle_attempt_p_new=0.0371
  [best sample (highest new_lp)] idx=34  new_lp=0.070  adv=-0.061  stored move_dir=-152.7°  new_mean=-152.4°
    per-head contributions: move_dir:0.188  move:-0.021  sprint:-0.023  kick:-0.036  tackle_attempt:-0.038
2026-08-07 20:10:06,080 INFO   [advantage] mean=0.005  std=0.998  min=-5.747  max=4.310
2026-08-07 20:10:06,080 INFO   [ratio] mean=0.8922  std=0.3474  min=0.0000  max=14.6904  clipped=26.0%
2026-08-07 20:10:06,080 INFO   [exec head grad norm] move_direction=0.044  exec_move=0.150  sprint=0.139  kick=0.079  kick_direction=0.014  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.094
2026-08-07 20:10:06,080 INFO   [exec continuous log_std] move_direction: start=-2.7995 end=-2.7995   kick_direction: start=-2.7997 end=-2.7997
2026-08-07 20:10:06,080 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0011≈0.06°/step  epoch≈0.1°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0013≈0.08°/step  epoch≈0.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-07 20:10:06,080 INFO   [exec discrete Δlogit per opt step] exec_move=0.0020  sprint=0.0023  kick=0.0006  tackle_attempt=0.0014
2026-08-07 20:10:06,081 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0002  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0173  sprint=+0.0124  kick=+0.0050  tackle_attempt=+0.0008  move_dir=+0.2243  kick_dir=+0.0858
2026-08-07 20:10:06,081 INFO   [grad clip] main: 1/1 steps clipped (100%)  pre-clip norm mean=0.604 max=0.604  limit=0.4
              direction: 1/1 steps clipped (100%)  pre-clip norm mean=0.054 max=0.054  limit=0.02
2026-08-07 20:10:06,085 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=576,000  speed=817/s  reward=4.84
  loss     policy=0.0293  value=0.4568(x0.5)=0.2284
           entropy=1.3280  kl=0.3458
  value    V=3.43±1.45  R=3.41±1.82  adv=-0.02±1.22
  moves    mv_ls=[-2.7995] (σ≈0.06, ≈3°) g=2.84e-02  d_move=[+0.0000] (Δσ≈0.000°)
           kk_ls=[-2.7997] (σ≈0.06, ≈3°)  d_kick=[+0.0000] (Δσ≈0.000°)
  heads    move= 42 get_poss= 58 exec_move= 91 sprint= 49 kick=  4 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0405 kick_prob=0.0439
  vs       vs[win/loss/tout/miss]  vs_immobile(549): 65.6%/0.2%/0.4%/9.1%/25%
  ep_len   13.0±8.0s  (n=549, min=0.8s, max=50.0s)
  reward   get_possession=+455.00  lose_possession=-3.60  ball_out=-20.00  box_possession=+900.00
           speed_bonus=+883.54  opponent_box=-3.00  timeout=-3.00  stamina_penalty=-3.08
  rew/ep   (mean/std/min/max per episode, 549 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.829    0.396    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.007    0.077    -0.900    +0.000
  ball_out          -0.036    0.425    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.639    1.188    +0.000    +2.500
  speed_bonus       +1.609    1.534    +0.000    +4.284
  opponent_box      -0.005    0.128    -3.000    +0.000
  timeout           -0.005    0.090    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.027    +0.000
  rew/step (per-step stats, n=24000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     458    +0.019    0.137     +4.387     1.334     +0.428      1.9783      1.041     2.925
  lose_possession       4    -0.000    0.012     +3.559     0.752     -0.147      0.5201      0.623     1.010
  ball_out             4    -0.001    0.065     -4.500     0.500     -5.136     27.5312      5.136     6.215
  box_possession     360    +0.037    0.304     +4.951     1.222     +0.222      1.1046      0.854     1.806
  speed_bonus        344    +0.037    0.334     +5.065     1.127     +0.304      1.0323      0.822     1.677
  opponent_box         1    -0.000    0.019     -3.006     0.000     -7.049     49.6851      7.049     7.049
  timeout              2    -0.000    0.014     -1.500     0.000     -1.471      2.2331      1.471     1.708
  stamina_penalty     346    -0.000    0.001     +4.963     1.288     +0.203      1.2274      0.875     1.826
  gae/td   mean_return=+3.412  std_return=1.822  mean_gae=-0.017  mean_sq_td=1.4977
──────────────────────────────────────────────────────────────────────
2026-08-07 20:10:06,110 INFO Saved checkpoint: checkpoints/phase1_run38/checkpoint24.pt
2026-08-07 20:10:06,110 INFO Logging to checkpoints/phase1_run38/training_log25.txt
2026-08-07 20:10:06,111 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 20:10:18,473 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 20:10:18,474 INFO   [eval vs immobile] step=576,000  seeds=16x8  win=56%  mean_rew=3.351±2.844  V=3.240  gap=-0.111  outcomes={'other': 39, 'box_possession': 72, 'miss': 17}
2026-08-07 20:10:18,476 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 20:10:29,907 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 20:10:29,908 INFO   [eval vs rules] step=576,000  seeds=16x8  win=23%  mean_rew=0.081±3.273  V=3.149  gap=+3.068  outcomes={'other': 35, 'box_possession': 29, 'opponent_box_possession': 53, 'miss': 10, 'timeout': 1}
2026-08-07 20:11:00,742 INFO   [early stop e0 mb0]  KL=0.33978 > target=0.12  steps_this_update=1
    [per-head KL] exec_move=+0.0108  sprint=+0.0112  kick=+0.0048  move_dir=+0.2133  kick_dir=+0.0985
2026-08-07 20:11:00,745 INFO   [KL mean=0.3398 median=0.3398 > 0.05] ratio percentiles:  p5=0.201  p25=0.881  p50=0.993  p75=1.000  p95=1.082  max=10.087
  move_dir_log_std=[-2.799475908279419]  kick_dir_log_std=[-2.799729347229004]
2026-08-07 20:11:00,758 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.171  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.131  kick=-0.233  t_att=-0.140
    move_dir=3.008 (min=0.000 max=3.761)  kick_dir=0.188 (min=0.000 max=3.762)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.25
  [worst sample] idx=143  ratio=27.794  adv=-0.030  old_lp=-3.419  new_lp=-0.094
    stored move_dir=-62.0°  new_mean=-61.7°  angular_diff=0.3°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 143  ratio=  27.794  adv=-0.030  lp: old=-3.419  new=-0.094
      rew=+0.0000  ret=+4.3291  val=+4.3588  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9640  sprint_p_new=0.9605  kick_p_new=0.0253  tackle_attempt_p_new=0.0429
    idx= 250  ratio=  26.791  adv=-0.397  lp: old=-7.189  new=-3.901
      rew=+0.0000  ret=+4.2059  val=+4.6026  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9626  sprint_p_new=0.0288  kick_p_new=0.0388  tackle_attempt_p_new=0.0359
  [best sample (highest new_lp)] idx=48  new_lp=0.076  adv=-0.101  stored move_dir=-98.1°  new_mean=-98.3°
    per-head contributions: move_dir:0.188  tackle_attempt:-0.026  sprint:-0.027  kick:-0.041
2026-08-07 20:11:00,758 INFO   [advantage] mean=-0.014  std=1.008  min=-7.638  max=4.995
2026-08-07 20:11:00,758 INFO   [ratio] mean=0.8975  std=0.3611  min=0.0000  max=10.0870  clipped=25.7%
2026-08-07 20:11:00,759 INFO   [exec head grad norm] move_direction=0.073  exec_move=0.128  sprint=0.051  kick=0.122  kick_direction=0.005  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.057
2026-08-07 20:11:00,759 INFO   [exec continuous log_std] move_direction: start=-2.7995 end=-2.7995   kick_direction: start=-2.7997 end=-2.7997
2026-08-07 20:11:00,759 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0012≈0.07°/step  epoch≈0.1°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0016≈0.09°/step  epoch≈0.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-07 20:11:00,759 INFO   [exec discrete Δlogit per opt step] exec_move=0.0022  sprint=0.0024  kick=0.0006  tackle_attempt=0.0013
2026-08-07 20:11:00,759 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0002  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0108  sprint=+0.0112  kick=+0.0048  tackle_attempt=+0.0010  move_dir=+0.2133  kick_dir=+0.0985
2026-08-07 20:11:00,759 INFO   [grad clip] main: 1/1 steps clipped (100%)  pre-clip norm mean=0.719 max=0.719  limit=0.4
              direction: 1/1 steps clipped (100%)  pre-clip norm mean=0.080 max=0.080  limit=0.02
2026-08-07 20:11:00,763 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=600,000  speed=816/s  reward=5.27
  loss     policy=0.0511  value=0.4636(x0.5)=0.2318
           entropy=1.3287  kl=0.3398
  value    V=3.51±1.39  R=3.58±1.74  adv=0.06±1.17
  moves    mv_ls=[-2.7995] (σ≈0.06, ≈3°) g=3.19e-02  d_move=[+0.0000] (Δσ≈0.000°)
           kk_ls=[-2.7997] (σ≈0.06, ≈3°)  d_kick=[+0.0000] (Δσ≈0.000°)
  heads    move= 44 get_poss= 56 exec_move= 91 sprint= 49 kick=  4 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0413 kick_prob=0.0439
  vs       vs[win/loss/tout/miss]  vs_immobile(518): 71.6%/0.2%/0.2%/9.3%/19%
  ep_len   13.7±7.4s  (n=518, min=1.4s, max=50.0s)
  reward   get_possession=+455.00  lose_possession=-5.40  ball_out=-15.00  box_possession=+927.50
           speed_bonus=+953.74  opponent_box=-3.00  timeout=-1.50  stamina_penalty=-3.30
  rew/ep   (mean/std/min/max per episode, 518 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.878    0.361    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.010    0.096    -0.900    +0.000
  ball_out          -0.029    0.379    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.791    1.127    +0.000    +2.500
  speed_bonus       +1.841    1.528    +0.000    +4.410
  opponent_box      -0.006    0.132    -3.000    +0.000
  timeout           -0.003    0.066    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.035    +0.000
  rew/step (per-step stats, n=24000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     460    +0.019    0.137     +4.492     1.247     +0.420      1.7553      0.976     2.462
  lose_possession       6    -0.000    0.014     +3.603     0.582     +0.007      0.1588      0.337     0.630
  ball_out             3    -0.001    0.056     -5.000     0.000     -7.399     58.4661      7.399     8.843
  box_possession     371    +0.039    0.308     +5.062     1.174     +0.391      1.0969      0.893     1.794
  speed_bonus        355    +0.040    0.349     +5.177     1.063     +0.482      1.0207      0.859     1.703
  opponent_box         1    -0.000    0.019     -3.003     0.000     -6.524     42.5630      6.524     6.524
  timeout              1    -0.000    0.010     -1.500     0.000     -0.393      0.1546      0.393     0.393
  stamina_penalty     356    -0.000    0.001     +5.070     1.230     +0.394      1.2093      0.906     1.811
  gae/td   mean_return=+3.577  std_return=1.739  mean_gae=+0.064  mean_sq_td=1.3700
──────────────────────────────────────────────────────────────────────
2026-08-07 20:11:00,785 INFO Saved checkpoint: checkpoints/phase1_run38/checkpoint25.pt
2026-08-07 20:11:00,785 INFO Logging to checkpoints/phase1_run38/training_log26.txt
2026-08-07 20:11:00,786 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 20:11:11,655 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 20:11:11,657 INFO   [eval vs immobile] step=600,000  seeds=16x8  win=54%  mean_rew=3.334±2.916  V=3.288  gap=-0.045  outcomes={'other': 41, 'box_possession': 69, 'miss': 18}
2026-08-07 20:11:11,658 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-07 20:11:24,562 INFO   [seeded eval] all workers finished, merging results.
2026-08-07 20:11:24,563 INFO   [eval vs rules] step=600,000  seeds=16x8  win=29%  mean_rew=0.650±3.667  V=3.152  gap=+2.502  outcomes={'other': 29, 'opponent_box_possession': 51, 'box_possession': 37, 'timeout': 3, 'miss': 8}
2026-08-07 20:11:28,362 INFO Saved checkpoint: checkpoints/phase1_run38/checkpoint26.pt
2026-08-07 20:11:28,362 INFO Logging to checkpoints/phase1_run38/training_log27.txt
2026-08-07 20:11:28,362 INFO Final checkpoint saved.
2026-08-07 20:11:28,362 INFO Training complete. Total steps: 600,000
