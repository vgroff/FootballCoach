uv : 2026-09-01 16:28:23,912 INFO Device: cuda (explicitly requested via --device)
At line:1 char:1
+ uv run python -m footballcoach.ai.scripts.train `
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : NotSpecified: (2026-09-01 16:2...d via --device):String) [], RemoteException
    + FullyQualifiedErrorId : NativeCommandError
 
2026-09-01 16:28:23,912 INFO Checkpoint dir: checkpoints\phase1_run36
2026-09-01 16:28:24,035 INFO Starting training: phase=phase1_get_possession, total_steps=300,000
2026-09-01 16:28:24,035 INFO Phase description: 1v1 scenario: learn to get possession and bring the ball toward the opponent box.  Decision network 
frozen except Move/GetPossession and the shared latent vector -- Move and GetPossession are the only two decision-level actions the rules-AI 
actually exercises in this scenario, kept together deliberately (confirmed 2026-09-01): either both stay trainable or neither does.
C:\Users\vgrof\Documents\repos\FootballCoach\.venv\Lib\site-packages\torch\_compile.py:54: UserWarning: optimizer contains a parameter group with 
duplicate parameters; in future, this will cause an error; see github.com/pytorch/pytorch/issues/40967 for more information
  return disable_fn(*args, **kwargs)
2026-09-01 16:28:26,496 INFO Logging to checkpoints\phase1_run36\training_log1.txt
2026-09-01 16:28:26,603 INFO --reset-optimizer: skipping optimizer state restore for checkpoints\phase1_run33\checkpoint_pretrained.pt (network 
weights still loaded normally)
2026-09-01 16:28:26,603 INFO Loaded checkpoint: checkpoints\phase1_run33\checkpoint_pretrained.pt (step 0)
2026-09-01 16:28:26,603 INFO Loaded pre-trained checkpoint: checkpoints\phase1_run33\checkpoint_pretrained.pt — skipping BC/value pre-training
2026-09-01 16:28:26,603 INFO   [seeded eval] running 15x4 episodes across 6 worker process(es)...
2026-09-01 16:28:30,016 INFO Loaded checkpoint: checkpoints\phase1_run33\checkpoint_pretrained.pt (step 0)
2026-09-01 16:28:30,095 INFO Loaded checkpoint: checkpoints\phase1_run33\checkpoint_pretrained.pt (step 0)
2026-09-01 16:28:30,095 INFO Loaded checkpoint: checkpoints\phase1_run33\checkpoint_pretrained.pt (step 0)
2026-09-01 16:28:30,105 INFO Loaded checkpoint: checkpoints\phase1_run33\checkpoint_pretrained.pt (step 0)
2026-09-01 16:28:30,171 INFO Loaded checkpoint: checkpoints\phase1_run33\checkpoint_pretrained.pt (step 0)
2026-09-01 16:28:30,198 INFO Loaded checkpoint: checkpoints\phase1_run33\checkpoint_pretrained.pt (step 0)
2026-09-01 16:28:41,107 INFO   [seeded eval] all workers finished, merging results.
2026-09-01 16:28:41,107 INFO Pre-PPO eval (rules opp): win=8.3%  mean_rew=-1.249  V=-0.361  R=-0.774  gap=+0.413  outcomes={'timeout': 20, 
'box_possession': 5, 'opponent_box_possession': 30, 'invalid': 4, 'miss': 1}
2026-09-01 16:28:41,107 INFO   rew breakdown (rules, per ep): opponent_box=-1.25  timeout=-0.33  speed_bonus=+0.18  box_possession=+0.17  
get_possession=+0.12  ball_out=-0.07  stamina_penalty=-0.05  lose_possession=-0.02
2026-09-01 16:28:41,107 INFO   [seeded eval] running 15x4 episodes across 6 worker process(es)...
2026-09-01 16:28:44,736 INFO Loaded checkpoint: checkpoints\phase1_run33\checkpoint_pretrained.pt (step 0)
2026-09-01 16:28:44,777 INFO Loaded checkpoint: checkpoints\phase1_run33\checkpoint_pretrained.pt (step 0)
2026-09-01 16:28:44,799 INFO Loaded checkpoint: checkpoints\phase1_run33\checkpoint_pretrained.pt (step 0)
2026-09-01 16:28:44,840 INFO Loaded checkpoint: checkpoints\phase1_run33\checkpoint_pretrained.pt (step 0)
2026-09-01 16:28:44,851 INFO Loaded checkpoint: checkpoints\phase1_run33\checkpoint_pretrained.pt (step 0)
2026-09-01 16:28:44,887 INFO Loaded checkpoint: checkpoints\phase1_run33\checkpoint_pretrained.pt (step 0)
2026-09-01 16:28:56,682 INFO   [seeded eval] all workers finished, merging results.
2026-09-01 16:28:56,686 INFO Pre-PPO eval (immobile opp): win=36.7%  mean_rew=1.376  V=0.535  R=0.371  gap=+0.163  outcomes={'box_possession': 22, 
'timeout': 28, 'miss': 3, 'invalid': 7}
2026-09-01 16:28:56,686 INFO   rew breakdown (immobile, per ep): speed_bonus=+0.82  box_possession=+0.73  get_possession=+0.57  timeout=-0.47  
ball_out=-0.20  stamina_penalty=-0.06  lose_possession=-0.02
2026-09-01 16:28:56,686 INFO   [seeded eval] running 15x4 episodes across 6 worker process(es)...
2026-09-01 16:29:00,383 INFO Loaded checkpoint: checkpoints\phase1_run33\checkpoint_pretrained.pt (step 0)
2026-09-01 16:29:00,495 INFO Loaded checkpoint: checkpoints\phase1_run33\checkpoint_pretrained.pt (step 0)
2026-09-01 16:29:00,522 INFO Loaded checkpoint: checkpoints\phase1_run33\checkpoint_pretrained.pt (step 0)
2026-09-01 16:29:00,561 INFO Loaded checkpoint: checkpoints\phase1_run33\checkpoint_pretrained.pt (step 0)
2026-09-01 16:29:00,568 INFO Loaded checkpoint: checkpoints\phase1_run33\checkpoint_pretrained.pt (step 0)
2026-09-01 16:29:00,581 INFO Loaded checkpoint: checkpoints\phase1_run33\checkpoint_pretrained.pt (step 0)
2026-09-01 16:29:27,335 INFO   [seeded eval] all workers finished, merging results.
2026-09-01 16:29:27,335 INFO Pre-PPO eval (self-play):   win=16.7%  mean_rew=-0.170  V=0.185  R=-0.326  gap=+0.511  outcomes={'timeout': 34, 
'opponent_box_possession': 9, 'box_possession': 10, 'invalid': 7}
2026-09-01 16:29:27,335 INFO   rew breakdown (self-play, per ep): timeout=-0.57  opponent_box=-0.38  speed_bonus=+0.35  box_possession=+0.33  
get_possession=+0.22  ball_out=-0.07  stamina_penalty=-0.05  lose_possession=-0.02
2026-09-01 16:29:27,335 INFO   [seeded eval] running 12x4 episodes across 6 worker process(es)...
2026-09-01 16:29:31,140 INFO   [seeded eval] all workers finished, merging results.
2026-09-01 16:29:31,140 INFO Baseline (rules vs rules, 12 trials): trainee_win=83.3%  outcomes={'box_possession': 40, 'opponent_box_possession': 8}
2026-09-01 16:29:31,140 INFO Frozen decision_net.shoot_logit
2026-09-01 16:29:31,140 INFO Frozen decision_net.pass_logit
2026-09-01 16:29:31,140 INFO Frozen decision_net.tackle_logit
2026-09-01 16:29:31,140 INFO Frozen decision_net.mark_logit
2026-09-01 16:29:31,140 INFO Frozen decision_net.hold_position_logit
2026-09-01 16:29:31,140 INFO Frozen decision_net.pass_target_logits
2026-09-01 16:29:31,140 INFO Frozen decision_net.tackle_target_logits
2026-09-01 16:29:31,140 INFO Frozen decision_net.mark_target_logits
2026-09-01 16:29:31,140 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this 
curriculum phase, no reward signal): hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed 
normally.
2026-09-01 16:29:31,140 INFO PPO parallel training started: 6 worker(s), ~5833 steps/worker/rollout, steps_so_far=0  target=300,000
2026-09-01 16:29:34,870 INFO Frozen decision_net.shoot_logit
2026-09-01 16:29:34,870 INFO Frozen decision_net.pass_logit
2026-09-01 16:29:34,870 INFO Frozen decision_net.tackle_logit
2026-09-01 16:29:34,870 INFO Frozen decision_net.mark_logit
2026-09-01 16:29:34,870 INFO Frozen decision_net.hold_position_logit
2026-09-01 16:29:34,870 INFO Frozen decision_net.pass_target_logits
2026-09-01 16:29:34,870 INFO Frozen decision_net.tackle_target_logits
2026-09-01 16:29:34,870 INFO Frozen decision_net.mark_target_logits
2026-09-01 16:29:34,870 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this 
curriculum phase, no reward signal): hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed 
normally.
2026-09-01 16:29:34,876 INFO Frozen decision_net.shoot_logit
2026-09-01 16:29:34,876 INFO Frozen decision_net.pass_logit
2026-09-01 16:29:34,876 INFO Frozen decision_net.tackle_logit
2026-09-01 16:29:34,876 INFO Frozen decision_net.mark_logit
2026-09-01 16:29:34,878 INFO Frozen decision_net.hold_position_logit
2026-09-01 16:29:34,878 INFO Frozen decision_net.pass_target_logits
2026-09-01 16:29:34,878 INFO Frozen decision_net.tackle_target_logits
2026-09-01 16:29:34,878 INFO Frozen decision_net.mark_target_logits
2026-09-01 16:29:34,878 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this 
curriculum phase, no reward signal): hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed 
normally.
2026-09-01 16:29:34,900 INFO Frozen decision_net.shoot_logit
2026-09-01 16:29:34,900 INFO Frozen decision_net.pass_logit
2026-09-01 16:29:34,900 INFO Frozen decision_net.tackle_logit
2026-09-01 16:29:34,900 INFO Frozen decision_net.mark_logit
2026-09-01 16:29:34,900 INFO Frozen decision_net.hold_position_logit
2026-09-01 16:29:34,900 INFO Frozen decision_net.pass_target_logits
2026-09-01 16:29:34,900 INFO Frozen decision_net.tackle_target_logits
2026-09-01 16:29:34,900 INFO Frozen decision_net.mark_target_logits
2026-09-01 16:29:34,900 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this 
curriculum phase, no reward signal): hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed 
normally.
2026-09-01 16:29:34,926 INFO Frozen decision_net.shoot_logit
2026-09-01 16:29:34,926 INFO Frozen decision_net.pass_logit
2026-09-01 16:29:34,926 INFO Frozen decision_net.tackle_logit
2026-09-01 16:29:34,926 INFO Frozen decision_net.mark_logit
2026-09-01 16:29:34,926 INFO Frozen decision_net.hold_position_logit
2026-09-01 16:29:34,926 INFO Frozen decision_net.pass_target_logits
2026-09-01 16:29:34,926 INFO Frozen decision_net.tackle_target_logits
2026-09-01 16:29:34,926 INFO Frozen decision_net.mark_target_logits
2026-09-01 16:29:34,926 INFO Frozen decision_net.shoot_logit
2026-09-01 16:29:34,932 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this 
curriculum phase, no reward signal): hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed 
normally.
2026-09-01 16:29:34,932 INFO Frozen decision_net.pass_logit
2026-09-01 16:29:34,932 INFO Frozen decision_net.tackle_logit
2026-09-01 16:29:34,932 INFO Frozen decision_net.mark_logit
2026-09-01 16:29:34,932 INFO Frozen decision_net.hold_position_logit
2026-09-01 16:29:34,933 INFO Frozen decision_net.pass_target_logits
2026-09-01 16:29:34,933 INFO Frozen decision_net.tackle_target_logits
2026-09-01 16:29:34,933 INFO Frozen decision_net.mark_target_logits
2026-09-01 16:29:34,933 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this 
curriculum phase, no reward signal): hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed 
normally.
2026-09-01 16:29:34,957 INFO Frozen decision_net.shoot_logit
2026-09-01 16:29:34,957 INFO Frozen decision_net.pass_logit
2026-09-01 16:29:34,957 INFO Frozen decision_net.tackle_logit
2026-09-01 16:29:34,957 INFO Frozen decision_net.mark_logit
2026-09-01 16:29:34,957 INFO Frozen decision_net.hold_position_logit
2026-09-01 16:29:34,957 INFO Frozen decision_net.pass_target_logits
2026-09-01 16:29:34,957 INFO Frozen decision_net.tackle_target_logits
2026-09-01 16:29:34,957 INFO Frozen decision_net.mark_target_logits
2026-09-01 16:29:34,957 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this 
curriculum phase, no reward signal): hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed 
normally.
  [rollout] (6 workers): 0/34998 (  0.0%)     0.0 steps/s
  [rollout] (6 workers): 3547/34998 ( 10.1%)   346.0 steps/s
  [rollout] (6 workers): 7016/34998 ( 20.0%)   367.5 steps/s
  [rollout] (6 workers): 10524/34998 ( 30.1%)   376.9 steps/s
  [rollout] (6 workers): 14070/34998 ( 40.2%)   382.7 steps/s
  [rollout] (6 workers): 17524/34998 ( 50.1%)   386.3 steps/s
  [rollout] (6 workers): 21082/34998 ( 60.2%)   389.3 steps/s
  [rollout] (6 workers): 24516/34998 ( 70.0%)   393.1 steps/s
  [rollout] (6 workers): 28071/34998 ( 80.2%)   395.5 steps/s
  [rollout] (6 workers): 31563/34998 ( 90.2%)   397.6 steps/s
[worker 5] done: 87.2s total  (14.96 ms/step, 0.95 s/episode over 92 episode(s))
[worker 2] done: 87.6s total  (15.01 ms/step, 0.93 s/episode over 94 episode(s))
[worker 1] done: 87.7s total  (15.04 ms/step, 0.88 s/episode over 100 episode(s))
[worker 3] done: 87.9s total  (15.07 ms/step, 0.97 s/episode over 91 episode(s))
[worker 0] done: 88.0s total  (15.09 ms/step, 0.90 s/episode over 98 episode(s))
[worker 4] done: 88.1s total  (15.11 ms/step, 0.93 s/episode over 95 episode(s))
  [rollout] (6 workers): 34998/34998 (100.0%)   396.9 steps/s
  [ppo update] 1/280 (  0.4%)     2.8 steps/s  epoch=1/4  kl=0.5389
2026-09-01 16:31:10,841 INFO   [early stop e0 mb15]  KL=0.70477 > target=0.6  steps_this_update=16
    [per-head KL] exec_move=+0.0154  kick=+0.0047  move_dir=+0.6825  kick_dir=+0.0023
2026-09-01 16:31:10,853 INFO   [KL mean=0.4615 median=0.4650 > 0.05] ratio percentiles:  p5=0.231  p25=0.894  p50=0.995  p75=1.040  p95=1.428  
max=134.189
  move_dir_log_std=[-2.402556896209717]  kick_dir_log_std=[-1.7676204442977905]
2026-09-01 16:31:10,914 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.049  tackle=0.000  gp=-0.003  mark=0.000  hold=0.000
    sprint=-0.012  kick=-0.059  t_att=-0.025
    move_dir=2.300 (min=-1.278 max=2.967)  kick_dir=0.008 (min=-1.907 max=2.115)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.05
  [worst sample] idx=165  ratio=215.265  adv=+0.578  old_lp=-5.386  new_lp=-0.014
    stored move_dir=6.5°  new_mean=-0.2°  angular_diff=6.7°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 165  ratio= 215.265  adv=+0.578  lp: old=-5.386  new=-0.014
      rew=+0.0000  ret=+1.7452  val=+1.1669  outcome=terminal:box_possession
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9954  sprint_p_new=0.9959  kick_p_new=0.0044  tackle_attempt_p_new=0.0040
    idx= 139  ratio= 140.748  adv=+0.793  lp: old=-4.965  new=-0.018
      rew=+0.0000  ret=+1.3373  val=+0.5447  outcome=terminal:box_possession
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9931  sprint_p_new=0.9935  kick_p_new=0.0058  tackle_attempt_p_new=0.0068
  [best sample (highest new_lp)] idx=120  new_lp=2.975  adv=+2.285  stored move_dir=-67.8°  new_mean=-67.6°
    per-head contributions: move_dir:2.966  kick_dir:2.115  kick_power:0.434  kick:-2.521
2026-09-01 16:31:10,914 INFO   [advantage] mean=0.000  std=1.001  min=-6.262  max=4.358
2026-09-01 16:31:10,914 INFO   [ratio] mean=0.9922  std=1.2507  min=0.0000  max=134.1886  clipped=36.2%
2026-09-01 16:31:10,914 INFO   [exec head grad norm] move_direction=7.209  exec_move=0.030  sprint=0.038  kick=0.015  kick_direction=0.170  
kick_power=0.162  kick_spin=0.000  tackle_attempt=0.017
2026-09-01 16:31:10,914 INFO   [exec continuous log_std] move_direction: start=-2.4027 end=-2.4026   kick_direction: start=-1.7677 end=-1.7676
2026-09-01 16:31:10,914 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0026≈0.15°/step  epoch≈2.4°  dlog_std=0.00001  
Δσ°=0.000/step)  kick_direction(dmean=0.0006≈0.04°/step  epoch≈0.6°  dlog_std=0.00000  Δσ°=0.000/step)
2026-09-01 16:31:10,914 INFO   [exec discrete Δlogit per opt step] exec_move=0.0060  sprint=0.0069  kick=0.0047  tackle_attempt=0.0034
2026-09-01 16:31:10,914 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0000  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  
hold=+0.0000  exec_move=+0.0097  sprint=+0.0058  kick=+0.0033  tackle_attempt=+0.0007  move_dir=+0.4385  kick_dir=+0.0023  kick_power=+0.0011  
kick_spin=+0.0000
2026-09-01 16:31:10,914 INFO   [grad clip] main: 16/16 steps clipped (100%)  pre-clip norm mean=5.774 max=8.980  limit=0.5
              direction: 16/16 steps clipped (100%)  pre-clip norm mean=7.212 max=12.472  limit=0.02
2026-09-01 16:31:10,914 INFO   [value RMSE by outcome] box_possession=1.760(n=4677)  invalid=0.545(n=561)  miss=2.809(n=84)  timeout=0.874(n=10590) 
 unknown=0.895(n=88)
2026-09-01 16:31:10,968 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=34,998  speed=366/s  reward=0.85
  loss     policy=0.0481  value=0.4914(x0.5)=0.2457  val_pre=0.4981
           entropy=0.1916  kl=0.4615
  value    V=0.43±1.30  R=0.43±1.73  adv=-0.01±1.22  |adv|=0.84
  entropy  shoot=0.0000  pass_=0.0000  move=0.0000  tackle=0.0000  gp_extra=0.0000  mark=0.0000  hold=0.0000  exec_move=0.0532
           kick=0.0435  tackle_attempt=0.0347  sprint=0.0621  move_dir=-0.0019  kick_dir=-0.0000  kick_power=0.0000  kick_spin=0.0000
  moves    mv_ls=[-2.4026] (σ≈0.09, ≈5°) g=4.83e-02
           kk_ls=[-1.7676] (σ≈0.17, ≈10°)
  heads    move= 19 get_poss= 81 exec_move= 96 sprint= 87 kick=  2 tackle=  1 shoot=
           0 hold=  0 tackle_prob=0.0122 kick_prob=0.0197
  vs       vs[win/loss/tout/miss/inval]  vs_immobile(570): 37.9%/0.0%/52.6%/1.2%/8.2%
  ep_len   14.6±5.1s  (n=570, min=0.7s, max=18.5s)
  reward   get_possession=+315.00  lose_possession=-2.70  ball_out=-28.00  box_possession=+432.00
           speed_bonus=+336.43  timeout=-300.00  stamina_penalty=-44.60
  rew/ep   (mean/std/min/max per episode, 570 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.553    0.497    +0.000    +1.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.005    0.065    -0.900    +0.000
  ball_out          -0.049    0.441    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +0.758    0.970    +0.000    +2.000
  speed_bonus       +0.590    0.957    +0.000    +3.857
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.526    0.499    -1.000    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  step_penalty      +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.078    0.036    -0.142    +0.000
  rew/step (per-step stats, n=34998 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     318    +0.009    0.095     +2.745     2.130     +1.628      4.7287      1.891     3.668
  lose_possession       3    -0.000    0.008     -1.234     0.380     -5.208     27.8236      5.208     6.128
  ball_out             7    -0.001    0.057     -4.000     0.000     -4.342     19.5914      4.342     5.430
  box_possession     216    +0.012    0.157     +3.582     1.069     +0.420      0.9976      0.541     2.828
  speed_bonus        215    +0.010    0.143     +3.585     1.071     +0.408      0.9609      0.530     2.667
  timeout            300    -0.009    0.092     -1.083     0.086     -0.160      0.2608      0.287     1.366
  stamina_penalty     516    -0.001    0.011     +0.870     2.404     +0.083      0.5692      0.393     1.862
  gae/td   mean_return=+0.428  std_return=1.728  mean_gae=-0.005  mean_sq_td=1.4941
──────────────────────────────────────────────────────────────────────
2026-09-01 16:31:11,045 INFO Saved checkpoint: checkpoints\phase1_run36\checkpoint1.pt
2026-09-01 16:31:11,045 INFO Logging to checkpoints\phase1_run36\training_log2.txt
2026-09-01 16:31:11,062 INFO   [seeded eval] running 15x4 episodes across 6 worker process(es)...
2026-09-01 16:31:26,698 INFO   [seeded eval] all workers finished, merging results.
2026-09-01 16:31:26,698 INFO   [eval vs immobile] step=34,998  seeds=15x4  win=43%  mean_rew=1.748±3.001 (sem=0.387)  V=0.546  gap=-1.202  
outcomes={'box_possession': 26, 'timeout': 24, 'invalid': 8, 'miss': 2}
2026-09-01 16:31:26,698 INFO   [seeded eval] running 15x4 episodes across 6 worker process(es)...
2026-09-01 16:31:43,123 INFO   [seeded eval] all workers finished, merging results.
2026-09-01 16:31:43,123 INFO   [eval vs rules] step=34,998  seeds=15x4  win=2%  mean_rew=-1.660±1.262 (sem=0.163)  V=-0.370  gap=+1.290  
outcomes={'timeout': 24, 'miss': 2, 'opponent_box_possession': 29, 'invalid': 4, 'box_possession': 1}
  [rollout] (6 workers): 84/34998 (  0.2%)   413.8 steps/s
  [rollout] (6 workers): 3575/34998 ( 10.2%)   395.8 steps/s
  [rollout] (6 workers): 7013/34998 ( 20.0%)   383.9 steps/s
  [rollout] (6 workers): 10531/34998 ( 30.1%)   364.1 steps/s
  [rollout] (6 workers): 14014/34998 ( 40.0%)   364.9 steps/s
  [rollout] (6 workers): 17529/34998 ( 50.1%)   365.8 steps/s
  [rollout] (6 workers): 21074/34998 ( 60.2%)   372.2 steps/s
  [rollout] (6 workers): 24564/34998 ( 70.2%)   375.8 steps/s
  [rollout] (6 workers): 28025/34998 ( 80.1%)   378.4 steps/s
  [rollout] (6 workers): 31500/34998 ( 90.0%)   378.7 steps/s
[worker 5] done: 91.7s total  (15.73 ms/step, 0.99 s/episode over 93 episode(s))
[worker 3] done: 92.0s total  (15.77 ms/step, 0.89 s/episode over 103 episode(s))
[worker 2] done: 92.2s total  (15.81 ms/step, 1.04 s/episode over 89 episode(s))
[worker 0] done: 92.2s total  (15.81 ms/step, 0.93 s/episode over 99 episode(s))
[worker 1] done: 92.4s total  (15.84 ms/step, 0.96 s/episode over 96 episode(s))
[worker 4] done: 92.5s total  (15.86 ms/step, 0.94 s/episode over 98 episode(s))
  [rollout] (6 workers): 34998/34998 (100.0%)   377.7 steps/s
  [ppo update] 1/280 (  0.4%)     9.1 steps/s  epoch=1/4  kl=0.5997
2026-09-01 16:33:21,706 INFO   [early stop e0 mb1]  KL=0.82027 > target=0.6  steps_this_update=2
    [per-head KL] exec_move=+0.0129  sprint=+0.0181  kick=+0.0109  move_dir=+0.7783  kick_dir=-0.0010
2026-09-01 16:33:21,706 INFO   [KL mean=0.7100 median=0.7100 > 0.05] ratio percentiles:  p5=0.243  p25=0.947  p50=1.000  p75=1.004  p95=1.325  
max=12.262
  move_dir_log_std=[-2.4025375843048096]  kick_dir_log_std=[-1.7676156759262085]
2026-09-01 16:33:21,753 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.027  tackle=0.000  gp=-0.049  mark=0.000  hold=0.000
    sprint=-0.116  kick=-0.044  t_att=-0.069
    move_dir=2.048 (min=-3.568 max=2.967)  kick_dir=0.007 (min=0.000 max=0.913)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.17
  [worst sample] idx=206  ratio=201.508  adv=-0.753  old_lp=-5.325  new_lp=-0.019
    stored move_dir=-175.2°  new_mean=-178.5°  angular_diff=3.3°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 206  ratio= 201.508  adv=-0.753  lp: old=-5.325  new=-0.019
      rew=+0.0000  ret=-0.3565  val=+0.3966  outcome=terminal:timeout
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9951  sprint_p_new=0.9939  kick_p_new=0.0063  tackle_attempt_p_new=0.0066
    idx= 245  ratio= 183.101  adv=-0.314  lp: old=-5.225  new=-0.015
      rew=+0.0000  ret=-0.8171  val=-0.5032  outcome=terminal:timeout
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9946  sprint_p_new=0.9943  kick_p_new=0.0047  tackle_attempt_p_new=0.0043
  [best sample (highest new_lp)] idx=200  new_lp=2.948  adv=-0.908  stored move_dir=-175.5°  new_mean=-175.6°
    per-head contributions: move_dir:2.967
2026-09-01 16:33:21,753 INFO   [advantage] mean=0.015  std=1.053  min=-5.634  max=3.681
2026-09-01 16:33:21,754 INFO   [ratio] mean=0.9768  std=0.5397  min=0.0000  max=12.2625  clipped=29.1%
2026-09-01 16:33:21,754 INFO   [exec head grad norm] move_direction=6.011  exec_move=0.023  sprint=0.024  kick=0.064  kick_direction=0.174  
kick_power=0.289  kick_spin=0.000  tackle_attempt=0.011
2026-09-01 16:33:21,754 INFO   [exec continuous log_std] move_direction: start=-2.4026 end=-2.4025   kick_direction: start=-1.7676 end=-1.7676
2026-09-01 16:33:21,755 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0018≈0.11°/step  epoch≈0.2°  dlog_std=0.00001  
Δσ°=0.000/step)  kick_direction(dmean=0.0005≈0.03°/step  epoch≈0.1°  dlog_std=0.00000  Δσ°=0.000/step)
2026-09-01 16:33:21,755 INFO   [exec discrete Δlogit per opt step] exec_move=0.0023  sprint=0.0025  kick=0.0032  tackle_attempt=0.0024
2026-09-01 16:33:21,755 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=-0.0000  tackle=+0.0000  gp_extra=+0.0001  mark=+0.0000  
hold=+0.0000  exec_move=+0.0104  sprint=+0.0097  kick=+0.0075  tackle_attempt=-0.0001  move_dir=+0.6819  kick_dir=+0.0000  kick_power=+0.0005  
kick_spin=+0.0000
2026-09-01 16:33:21,756 INFO   [grad clip] main: 2/2 steps clipped (100%)  pre-clip norm mean=5.371 max=5.816  limit=0.5
              direction: 2/2 steps clipped (100%)  pre-clip norm mean=6.014 max=6.899  limit=0.02
2026-09-01 16:33:21,756 INFO   [value RMSE by outcome] box_possession=1.813(n=581)  invalid=0.470(n=54)  miss=4.028(n=10)  timeout=0.930(n=1347)  
unknown=0.669(n=8)
2026-09-01 16:33:21,817 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=69,996  speed=363/s  reward=0.75
  loss     policy=0.0258  value=0.5262(x0.5)=0.2631  val_pre=0.4735
           entropy=0.1922  kl=0.7100
  value    V=0.46±1.23  R=0.43±1.76  adv=-0.03±1.20  |adv|=0.85
  entropy  shoot=0.0000(+0.0000)  pass_=0.0000(+0.0000)  move=0.0000(+0.0000)  tackle=0.0000(+0.0000)  gp_extra=0.0000(+0.0000)  
mark=0.0000(+0.0000)  hold=0.0000(+0.0000)  exec_move=0.0524(-0.0008)
           kick=0.0434(-0.0001)  tackle_attempt=0.0356(+0.0009)  sprint=0.0627(+0.0007)  move_dir=-0.0019(-0.0000)  kick_dir=-0.0000(+0.0000)  
kick_power=0.0000(-0.0000)  kick_spin=0.0000(+0.0000)
  moves    mv_ls=[-2.4025] (σ≈0.09, ≈5°) g=2.84e-02  d_move=[+0.0000] (Δσ≈0.000°)
           kk_ls=[-1.7676] (σ≈0.17, ≈10°)  d_kick=[+0.0000] (Δσ≈0.000°)
  heads    move= 19 get_poss= 81 exec_move= 97 sprint= 89 kick=  2 tackle=  2 shoot=
           0 hold=  0 tackle_prob=0.0141 kick_prob=0.0183
  vs       vs[win/loss/tout/miss/inval]  vs_immobile(578): 38.8%/0.0%/54.0%/0.7%/6.6%
  ep_len   14.5±5.3s  (n=578, min=1.0s, max=18.5s)
  reward   get_possession=+319.00  lose_possession=-5.40  ball_out=-16.00  box_possession=+448.00
           speed_bonus=+387.55  timeout=-312.00  stamina_penalty=-47.05
  rew/ep   (mean/std/min/max per episode, 578 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.552    0.504    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.009    0.091    -0.900    +0.000
  ball_out          -0.028    0.332    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +0.775    0.974    +0.000    +2.000
  speed_bonus       +0.670    1.043    +0.000    +3.792
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.540    0.498    -1.000    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  step_penalty      +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.081    0.035    -0.142    +0.000
  rew/step (per-step stats, n=34998 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     319    +0.009    0.095     +2.922     2.178     +1.690      4.7496      1.865     3.714
  lose_possession       6    -0.000    0.012     -1.578     0.368     -1.426      2.4145      1.426     2.418
  ball_out             4    -0.000    0.043     -3.750     0.433     -4.080     19.1811      4.080     6.307
  box_possession     224    +0.013    0.159     +3.795     1.136     +0.725      1.8014      0.756     3.421
  speed_bonus        223    +0.011    0.159     +3.803     1.131     +0.725      1.8063      0.756     3.423
  timeout            312    -0.009    0.094     -1.088     0.064     -0.270      0.2420      0.303     1.332
  stamina_penalty     536    -0.001    0.011     +0.952     2.518     +0.146      0.8937      0.492     2.082
  gae/td   mean_return=+0.434  std_return=1.758  mean_gae=-0.026  mean_sq_td=1.4471
──────────────────────────────────────────────────────────────────────
2026-09-01 16:33:21,895 INFO Saved checkpoint: checkpoints\phase1_run36\checkpoint2.pt
2026-09-01 16:33:21,896 INFO Logging to checkpoints\phase1_run36\training_log3.txt
2026-09-01 16:33:21,905 INFO   [seeded eval] running 15x4 episodes across 6 worker process(es)...

