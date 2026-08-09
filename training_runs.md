2026-08-09 04:12:31,230 INFO Checkpoint dir: checkpoints/phase1_run49
2026-08-09 04:12:31,283 INFO Starting training: phase=phase1_get_possession, total_steps=6,000,000
2026-08-09 04:12:31,283 INFO Phase description: 1v1 scenario: learn to get possession and bring the ball toward the opponent box.  Decision network frozen except Move/GetPossession and latent vector.
/home/vincent/Documents/not_work/repos/FootballCoach/.venv/lib/python3.12/site-packages/torch/_compile.py:54: UserWarning: optimizer contains a parameter group with duplicate parameters; in future, this will cause an error; see github.com/pytorch/pytorch/issues/40967 for more information
  return disable_fn(*args, **kwargs)
2026-08-09 04:12:32,492 INFO Logging to checkpoints/phase1_run49/training_log1.txt
2026-08-09 04:12:32,674 INFO Loading 1000 demonstration file(s) from demonstrations/phase1
2026-08-09 04:12:37,241 INFO Dataset: 410,630 steps loaded
2026-08-09 04:12:37,242 INFO Offline BC dataset: 410,630 steps from demonstrations/phase1/
2026-08-09 04:12:37,242 INFO Reward diagnostic: running 40 rules-vs-rules episodes...
2026-08-09 04:12:38,326 INFO Reward diagnostic (40 ep, rules vs rules): mean_ep_rew=1.20  per_episode: get_possession=+1.20  lose_possession=-0.47  ball_out=-0.25  box_possession=+1.30  speed_bonus=+0.48  opponent_box=-1.05  stamina_penalty=-0.02
2026-08-09 04:12:38,339 INFO BC pos_weight (auto-computed from dataset): kick=1.50  tackle_attempt=1.50
2026-08-09 04:12:38,339 INFO Combined BC + value pre-training: 12 epoch(s), batch_size=1000, dataset=410,630 steps, rollout_steps=22000
2026-08-09 04:12:38,449 INFO Phase 0 — decision-net warm-up (BC + execution_net.value_head MSE; single value head convention): 6 epoch(s), gamma=0.995, returns mean=1.17  std=3.29  lr=0.015  phase0_value_coef=0.85
/home/vincent/Documents/not_work/repos/FootballCoach/src/footballcoach/ai/ppo/bc.py:688: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /__w/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:822.)
  "decision":       float(dec_loss[valid].mean()),
2026-08-09 04:13:12,885 INFO   Phase 0 epoch 1/6: loss=2.3631  dec_bc=1.5797  val=0.9216(x0.85)=0.7833
2026-08-09 04:13:48,244 INFO   Phase 0 epoch 2/6: loss=2.1301  dec_bc=1.3923  val=0.8680(x0.85)=0.7378
2026-08-09 04:14:24,413 INFO   Phase 0 epoch 3/6: loss=2.0291  dec_bc=1.3907  val=0.7511(x0.85)=0.6384
2026-08-09 04:15:00,254 INFO   Phase 0 epoch 4/6: loss=1.9946  dec_bc=1.3890  val=0.7124(x0.85)=0.6055
2026-08-09 04:15:33,905 INFO   Phase 0 epoch 5/6: loss=1.9759  dec_bc=1.3886  val=0.6910(x0.85)=0.5874
2026-08-09 04:16:07,637 INFO   Phase 0 epoch 6/6: loss=1.9618  dec_bc=1.3866  val=0.6767(x0.85)=0.5752
2026-08-09 04:16:07,638 INFO Phase 0 done (decision-net BC + critic value_head warm-up, 6 epoch(s))
2026-08-09 04:16:07,746 INFO Phase 1 BC epochs will include joint value loss (coef=2.0, gamma=0.995, returns std=3.29)
2026-08-09 04:16:07,760 INFO   BC pretrain split: 307,504 train rows  |  56,053 val rows
2026-08-09 04:16:07,994 INFO   Downsample trivial rows (epoch 1): 119,112/363,557 (32.8%) rows classified trivial, excluding ~95,290 this epoch (frac=0.80)
2026-08-09 04:18:15,305 INFO   BC epoch 1/12  (127.5s)
    loss       bc=3.5580  bc_adj=1.4934(floor=2.0646)  val_loss=0.7102(x2.0)=1.4203  rmse=2.77 (returns std=3.3)
    heads      dir_cos=0.839  kick_dir_cos=0.672
               move_prob=0.853  sprint_prob=0.786  kick_prob=0.106  tackle_prob=0.265
    pr/rec     kick:   p=0.000  r=0.000  f1=nan  (tp=0 fp=2 fn=44308)
               tackle: p=0.786  r=0.461  f1=0.581  (tp=80499 fp=21879 fn=94253)
    breakdown  decision=0.690  exec_bce=1.664  sprint=0.488  move=0.383  tackle_attempt=0.471  direction=0.450
               region=0.014  kick=0.32162  kick_direction=0.98500  kick_power=0.02382  kick_spin=0.00012
2026-08-09 04:18:17,091 INFO     val        bc_val_loss=2.7924  best=2.7924  (improved)
2026-08-09 04:18:17,103 INFO   Downsample trivial rows (epoch 2): 119,112/363,557 (32.8%) rows classified trivial, excluding ~95,290 this epoch (frac=0.80)
2026-08-09 04:20:17,947 INFO   BC epoch 2/12  (120.9s)
    loss       bc=2.8047  bc_adj=0.7401(floor=2.0646)  val_loss=0.6847(x2.0)=1.3694  rmse=2.72 (returns std=3.3)
    heads      dir_cos=0.945  kick_dir_cos=0.962
               move_prob=0.872  sprint_prob=0.828  kick_prob=0.091  tackle_prob=0.231
    pr/rec     kick:   p=0.802  r=0.443  f1=0.570  (tp=19619 fp=4858 fn=24689)
               tackle: p=0.879  r=0.979  f1=0.927  (tp=171136 fp=23528 fn=3616)
    breakdown  decision=0.688  exec_bce=1.254  sprint=0.420  move=0.329  tackle_attempt=0.256  direction=0.153
               region=0.015  kick=0.24786  kick_direction=0.11533  kick_power=0.00408  kick_spin=0.00005
2026-08-09 04:20:19,809 INFO     val        bc_val_loss=3.0158  best=2.7924  (patience 1/3)
2026-08-09 04:20:19,815 INFO   Downsample trivial rows (epoch 3): 119,112/363,557 (32.8%) rows classified trivial, excluding ~95,290 this epoch (frac=0.80)
2026-08-09 04:22:30,571 INFO   BC epoch 3/12  (130.8s)
    loss       bc=2.6460  bc_adj=0.5814(floor=2.0646)  val_loss=0.6772(x2.0)=1.3545  rmse=2.70 (returns std=3.3)
    heads      dir_cos=0.950  kick_dir_cos=0.967
               move_prob=0.864  sprint_prob=0.818  kick_prob=0.095  tackle_prob=0.225
    pr/rec     kick:   p=0.887  r=0.835  f1=0.860  (tp=36997 fp=4709 fn=7311)
               tackle: p=0.893  r=0.976  f1=0.933  (tp=170529 fp=20328 fn=4223)
    breakdown  decision=0.688  exec_bce=1.113  sprint=0.367  move=0.287  tackle_attempt=0.242  direction=0.138
               region=0.014  kick=0.21732  kick_direction=0.10041  kick_power=0.00241  kick_spin=0.00003
2026-08-09 04:22:32,607 INFO     val        bc_val_loss=2.5064  best=2.5064  (improved)
2026-08-09 04:22:32,617 INFO   Downsample trivial rows (epoch 4): 119,112/363,557 (32.8%) rows classified trivial, excluding ~95,290 this epoch (frac=0.80)
2026-08-09 04:24:41,665 INFO   BC epoch 4/12  (129.1s)
    loss       bc=2.5611  bc_adj=0.4965(floor=2.0646)  val_loss=0.6720(x2.0)=1.3440  rmse=2.69 (returns std=3.3)
    heads      dir_cos=0.954  kick_dir_cos=0.971
               move_prob=0.862  sprint_prob=0.812  kick_prob=0.092  tackle_prob=0.224
    pr/rec     kick:   p=0.937  r=0.858  f1=0.896  (tp=38009 fp=2541 fn=6299)
               tackle: p=0.902  r=0.975  f1=0.937  (tp=170366 fp=18542 fn=4386)
    breakdown  decision=0.687  exec_bce=1.042  sprint=0.331  move=0.267  tackle_attempt=0.238  direction=0.127
               region=0.013  kick=0.20580  kick_direction=0.08799  kick_power=0.00175  kick_spin=0.00002
2026-08-09 04:24:43,528 INFO     val        bc_val_loss=2.4458  best=2.4458  (improved)
2026-08-09 04:24:43,538 INFO   Downsample trivial rows (epoch 5): 119,112/363,557 (32.8%) rows classified trivial, excluding ~95,290 this epoch (frac=0.80)
2026-08-09 04:27:12,577 INFO   BC epoch 5/12  (149.0s)
    loss       bc=2.5052  bc_adj=0.4406(floor=2.0646)  val_loss=0.6653(x2.0)=1.3306  rmse=2.68 (returns std=3.3)
    heads      dir_cos=0.960  kick_dir_cos=0.974
               move_prob=0.862  sprint_prob=0.810  kick_prob=0.090  tackle_prob=0.224
    pr/rec     kick:   p=0.949  r=0.884  f1=0.916  (tp=39189 fp=2110 fn=5119)
               tackle: p=0.906  r=0.974  f1=0.939  (tp=170278 fp=17630 fn=4474)
    breakdown  decision=0.687  exec_bce=1.002  sprint=0.311  move=0.259  tackle_attempt=0.235  direction=0.113
               region=0.012  kick=0.19693  kick_direction=0.07841  kick_power=0.00173  kick_spin=0.00001
2026-08-09 04:27:14,528 INFO     val        bc_val_loss=2.4138  best=2.4138  (improved)
2026-08-09 04:27:14,540 INFO   Downsample trivial rows (epoch 6): 119,112/363,557 (32.8%) rows classified trivial, excluding ~95,290 this epoch (frac=0.80)
2026-08-09 04:29:34,886 INFO   BC epoch 6/12  (140.4s)
    loss       bc=2.4722  bc_adj=0.4076(floor=2.0646)  val_loss=0.6620(x2.0)=1.3240  rmse=2.67 (returns std=3.3)
    heads      dir_cos=0.962  kick_dir_cos=0.976
               move_prob=0.862  sprint_prob=0.809  kick_prob=0.088  tackle_prob=0.224
    pr/rec     kick:   p=0.950  r=0.907  f1=0.928  (tp=40191 fp=2134 fn=4117)
               tackle: p=0.910  r=0.975  f1=0.942  (tp=170470 fp=16804 fn=4282)
    breakdown  decision=0.687  exec_bce=0.976  sprint=0.298  move=0.253  tackle_attempt=0.233  direction=0.107
               region=0.011  kick=0.19144  kick_direction=0.07147  kick_power=0.00182  kick_spin=0.00001
2026-08-09 04:29:36,677 INFO     val        bc_val_loss=2.4296  best=2.4138  (patience 1/3)
2026-08-09 04:29:36,683 INFO   Downsample trivial rows (epoch 7): 119,112/363,557 (32.8%) rows classified trivial, excluding ~95,290 this epoch (frac=0.80)
2026-08-09 04:31:56,891 INFO   BC epoch 7/12  (140.2s)
    loss       bc=2.4495  bc_adj=0.3849(floor=2.0646)  val_loss=0.6605(x2.0)=1.3210  rmse=2.67 (returns std=3.3)
    heads      dir_cos=0.963  kick_dir_cos=0.979
               move_prob=0.862  sprint_prob=0.809  kick_prob=0.088  tackle_prob=0.223
    pr/rec     kick:   p=0.951  r=0.921  f1=0.936  (tp=40806 fp=2119 fn=3502)
               tackle: p=0.914  r=0.976  f1=0.944  (tp=170593 fp=16007 fn=4159)
    breakdown  decision=0.687  exec_bce=0.957  sprint=0.289  move=0.249  tackle_attempt=0.231  direction=0.104
               region=0.011  kick=0.18858  kick_direction=0.06340  kick_power=0.00184  kick_spin=0.00001
2026-08-09 04:31:58,877 INFO     val        bc_val_loss=2.3750  best=2.3750  (improved)
2026-08-09 04:31:58,889 INFO   Downsample trivial rows (epoch 8): 119,112/363,557 (32.8%) rows classified trivial, excluding ~95,290 this epoch (frac=0.80)
2026-08-09 04:34:28,047 INFO   BC epoch 8/12  (149.2s)
    loss       bc=2.4325  bc_adj=0.3678(floor=2.0646)  val_loss=0.6572(x2.0)=1.3143  rmse=2.66 (returns std=3.3)
    heads      dir_cos=0.965  kick_dir_cos=0.983
               move_prob=0.862  sprint_prob=0.809  kick_prob=0.087  tackle_prob=0.223
    pr/rec     kick:   p=0.952  r=0.930  f1=0.941  (tp=41193 fp=2091 fn=3115)
               tackle: p=0.919  r=0.977  f1=0.947  (tp=170817 fp=15120 fn=3935)
    breakdown  decision=0.687  exec_bce=0.946  sprint=0.284  move=0.247  tackle_attempt=0.229  direction=0.099
               region=0.011  kick=0.18695  kick_direction=0.05165  kick_power=0.00175  kick_spin=0.00001
2026-08-09 04:34:29,883 INFO     val        bc_val_loss=2.3592  best=2.3592  (improved)
2026-08-09 04:34:29,893 INFO   Downsample trivial rows (epoch 9): 119,112/363,557 (32.8%) rows classified trivial, excluding ~95,290 this epoch (frac=0.80)
2026-08-09 04:36:27,762 INFO   BC epoch 9/12  (117.9s)
    loss       bc=2.4122  bc_adj=0.3476(floor=2.0646)  val_loss=0.6523(x2.0)=1.3046  rmse=2.65 (returns std=3.3)
    heads      dir_cos=0.967  kick_dir_cos=0.986
               move_prob=0.862  sprint_prob=0.808  kick_prob=0.087  tackle_prob=0.223
    pr/rec     kick:   p=0.958  r=0.937  f1=0.947  (tp=41513 fp=1836 fn=2795)
               tackle: p=0.922  r=0.978  f1=0.949  (tp=170964 fp=14496 fn=3788)
    breakdown  decision=0.687  exec_bce=0.932  sprint=0.277  move=0.243  tackle_attempt=0.227  direction=0.093
               region=0.011  kick=0.18523  kick_direction=0.04148  kick_power=0.00175  kick_spin=0.00001
2026-08-09 04:36:29,607 INFO     val        bc_val_loss=2.3576  best=2.3576  (improved)
2026-08-09 04:36:29,618 INFO   Downsample trivial rows (epoch 10): 119,112/363,557 (32.8%) rows classified trivial, excluding ~95,290 this epoch (frac=0.80)
2026-08-09 04:38:35,333 INFO   BC epoch 10/12  (125.7s)
    loss       bc=2.3997  bc_adj=0.3351(floor=2.0646)  val_loss=0.6518(x2.0)=1.3036  rmse=2.65 (returns std=3.3)
    heads      dir_cos=0.967  kick_dir_cos=0.987
               move_prob=0.862  sprint_prob=0.809  kick_prob=0.087  tackle_prob=0.223
    pr/rec     kick:   p=0.957  r=0.944  f1=0.950  (tp=41811 fp=1864 fn=2497)
               tackle: p=0.924  r=0.979  f1=0.951  (tp=171026 fp=13988 fn=3726)
    breakdown  decision=0.687  exec_bce=0.922  sprint=0.273  move=0.239  tackle_attempt=0.225  direction=0.091
               region=0.011  kick=0.18431  kick_direction=0.03949  kick_power=0.00175  kick_spin=0.00001
2026-08-09 04:38:37,500 INFO     val        bc_val_loss=2.3497  best=2.3497  (improved)
2026-08-09 04:38:37,511 INFO   Downsample trivial rows (epoch 11): 119,112/363,557 (32.8%) rows classified trivial, excluding ~95,290 this epoch (frac=0.80)
2026-08-09 04:40:54,304 INFO   BC epoch 11/12  (136.8s)
    loss       bc=2.3882  bc_adj=0.3236(floor=2.0646)  val_loss=0.6490(x2.0)=1.2980  rmse=2.65 (returns std=3.3)
    heads      dir_cos=0.968  kick_dir_cos=0.987
               move_prob=0.862  sprint_prob=0.809  kick_prob=0.086  tackle_prob=0.223
    pr/rec     kick:   p=0.962  r=0.949  f1=0.956  (tp=42039 fp=1643 fn=2269)
               tackle: p=0.927  r=0.979  f1=0.952  (tp=171083 fp=13545 fn=3669)
    breakdown  decision=0.687  exec_bce=0.912  sprint=0.268  move=0.237  tackle_attempt=0.224  direction=0.090
               region=0.010  kick=0.18310  kick_direction=0.03898  kick_power=0.00181  kick_spin=0.00000
2026-08-09 04:40:56,091 INFO     val        bc_val_loss=2.3219  best=2.3219  (improved)
2026-08-09 04:40:56,101 INFO   Downsample trivial rows (epoch 12): 119,112/363,557 (32.8%) rows classified trivial, excluding ~95,290 this epoch (frac=0.80)
2026-08-09 04:43:13,773 INFO   BC epoch 12/12  (137.7s)
    loss       bc=2.3791  bc_adj=0.3145(floor=2.0646)  val_loss=0.6482(x2.0)=1.2964  rmse=2.65 (returns std=3.3)
    heads      dir_cos=0.968  kick_dir_cos=0.988
               move_prob=0.862  sprint_prob=0.809  kick_prob=0.086  tackle_prob=0.222
    pr/rec     kick:   p=0.964  r=0.952  f1=0.958  (tp=42202 fp=1584 fn=2106)
               tackle: p=0.928  r=0.979  f1=0.953  (tp=171051 fp=13228 fn=3701)
    breakdown  decision=0.687  exec_bce=0.904  sprint=0.265  move=0.234  tackle_attempt=0.223  direction=0.089
               region=0.010  kick=0.18247  kick_direction=0.03716  kick_power=0.00177  kick_spin=0.00000
2026-08-09 04:43:16,004 INFO     val        bc_val_loss=2.3148  best=2.3148  (improved)
2026-08-09 04:43:16,009 INFO BC pre-training done (12 epoch(s), final bc_loss=2.3791)
2026-08-09 04:43:16,009 INFO Value pre-training: 22000 steps, 35 epochs, lr=3e-05
2026-08-09 04:43:16,011 INFO   [value pretrain rollout] parallel collection: 6 worker(s), ~3666 steps/worker
2026-08-09 04:43:18,054 INFO Frozen decision_net.shoot_logit
2026-08-09 04:43:18,054 INFO Frozen decision_net.pass_logit
2026-08-09 04:43:18,054 INFO Frozen decision_net.tackle_logit
2026-08-09 04:43:18,054 INFO Frozen decision_net.get_possession_raw
2026-08-09 04:43:18,054 INFO Frozen decision_net.mark_logit
2026-08-09 04:43:18,054 INFO Frozen decision_net.hold_position_logit
2026-08-09 04:43:18,054 INFO Frozen decision_net.pass_target_logits
2026-08-09 04:43:18,054 INFO Frozen decision_net.tackle_target_logits
2026-08-09 04:43:18,054 INFO Frozen decision_net.mark_target_logits
2026-08-09 04:43:18,054 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 04:43:18,079 INFO Frozen decision_net.shoot_logit
2026-08-09 04:43:18,079 INFO Frozen decision_net.pass_logit
2026-08-09 04:43:18,079 INFO Frozen decision_net.tackle_logit
2026-08-09 04:43:18,079 INFO Frozen decision_net.get_possession_raw
2026-08-09 04:43:18,079 INFO Frozen decision_net.mark_logit
2026-08-09 04:43:18,079 INFO Frozen decision_net.hold_position_logit
2026-08-09 04:43:18,079 INFO Frozen decision_net.pass_target_logits
2026-08-09 04:43:18,079 INFO Frozen decision_net.tackle_target_logits
2026-08-09 04:43:18,079 INFO Frozen decision_net.mark_target_logits
2026-08-09 04:43:18,079 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 04:43:18,096 INFO Frozen decision_net.shoot_logit
2026-08-09 04:43:18,096 INFO Frozen decision_net.pass_logit
2026-08-09 04:43:18,096 INFO Frozen decision_net.tackle_logit
2026-08-09 04:43:18,096 INFO Frozen decision_net.get_possession_raw
2026-08-09 04:43:18,096 INFO Frozen decision_net.mark_logit
2026-08-09 04:43:18,096 INFO Frozen decision_net.hold_position_logit
2026-08-09 04:43:18,096 INFO Frozen decision_net.pass_target_logits
2026-08-09 04:43:18,096 INFO Frozen decision_net.tackle_target_logits
2026-08-09 04:43:18,096 INFO Frozen decision_net.mark_target_logits
2026-08-09 04:43:18,096 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 04:43:18,100 INFO Frozen decision_net.shoot_logit
2026-08-09 04:43:18,101 INFO Frozen decision_net.pass_logit
2026-08-09 04:43:18,101 INFO Frozen decision_net.tackle_logit
2026-08-09 04:43:18,101 INFO Frozen decision_net.get_possession_raw
2026-08-09 04:43:18,101 INFO Frozen decision_net.mark_logit
2026-08-09 04:43:18,101 INFO Frozen decision_net.hold_position_logit
2026-08-09 04:43:18,101 INFO Frozen decision_net.pass_target_logits
2026-08-09 04:43:18,101 INFO Frozen decision_net.tackle_target_logits
2026-08-09 04:43:18,101 INFO Frozen decision_net.mark_target_logits
2026-08-09 04:43:18,101 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 04:43:18,170 INFO Frozen decision_net.shoot_logit
2026-08-09 04:43:18,170 INFO Frozen decision_net.pass_logit
2026-08-09 04:43:18,170 INFO Frozen decision_net.tackle_logit
2026-08-09 04:43:18,170 INFO Frozen decision_net.get_possession_raw
2026-08-09 04:43:18,170 INFO Frozen decision_net.mark_logit
2026-08-09 04:43:18,170 INFO Frozen decision_net.hold_position_logit
2026-08-09 04:43:18,170 INFO Frozen decision_net.pass_target_logits
2026-08-09 04:43:18,170 INFO Frozen decision_net.tackle_target_logits
2026-08-09 04:43:18,170 INFO Frozen decision_net.mark_target_logits
2026-08-09 04:43:18,170 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 04:43:18,179 INFO Frozen decision_net.shoot_logit
2026-08-09 04:43:18,179 INFO Frozen decision_net.pass_logit
2026-08-09 04:43:18,179 INFO Frozen decision_net.tackle_logit
2026-08-09 04:43:18,179 INFO Frozen decision_net.get_possession_raw
2026-08-09 04:43:18,179 INFO Frozen decision_net.mark_logit
2026-08-09 04:43:18,179 INFO Frozen decision_net.hold_position_logit
2026-08-09 04:43:18,179 INFO Frozen decision_net.pass_target_logits
2026-08-09 04:43:18,179 INFO Frozen decision_net.tackle_target_logits
2026-08-09 04:43:18,179 INFO Frozen decision_net.mark_target_logits
2026-08-09 04:43:18,179 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 04:43:50,642 INFO   [value pretrain rollout] dropped 461 trailing (incomplete-episode) step(s) across workers before MC-return fit
2026-08-09 04:43:50,650 INFO   [value pretrain rollout] mean_return=1.33 (300 episode(s))  vs[win/loss/tout/miss]  vs_rules(0): n/a  vs_immobile(300): 71.0%/0.0%/8.7%/20.3%  vs_neural(0): n/a
2026-08-09 04:43:50,650 INFO   [value pretrain rollout] ep_len 21.4±14.5s  (n=300, min=1.7s, max=50.1s)
2026-08-09 04:43:50,651 INFO   [value pretrain rollout] rew/ep (mean/std/min/max per episode, 300 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.923    0.278    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.003    0.052    -0.900    +0.000
  ball_out          -1.167    3.210   -10.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.420    0.908    +0.000    +2.000
  speed_bonus       +0.338    0.368    +0.000    +1.224
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.173    0.563    -2.000    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.008    0.009    -0.059    +0.000
2026-08-09 04:43:50,798 INFO   Value pretrain split: 255 train eps (18089 steps)  |  45 val eps (3446 steps)
2026-08-09 04:43:54,244 INFO   Value epoch 1/35: train=1.2417 rmse=3.60  val=0.9352 val_rmse=3.12 (std=3.2)
    V(train)=+1.988  R(train)=+0.526  |  V(val)=+2.016  R(val)=+0.898
2026-08-09 04:43:57,538 INFO   Value epoch 2/35: train=1.1966 rmse=3.53  val=0.8995 val_rmse=3.06 (std=3.2)
    V(train)=+1.831  R(train)=+0.524  |  V(val)=+1.862  R(val)=+0.898
2026-08-09 04:44:00,825 INFO   Value epoch 3/35: train=1.1530 rmse=3.47  val=0.8678 val_rmse=3.01 (std=3.2)
    V(train)=+1.672  R(train)=+0.526  |  V(val)=+1.706  R(val)=+0.898
2026-08-09 04:44:04,060 INFO   Value epoch 4/35: train=1.1131 rmse=3.41  val=0.8403 val_rmse=2.96 (std=3.2)
    V(train)=+1.512  R(train)=+0.527  |  V(val)=+1.548  R(val)=+0.898
2026-08-09 04:44:07,294 INFO   Value epoch 5/35: train=1.0791 rmse=3.36  val=0.8169 val_rmse=2.92 (std=3.2)
    V(train)=+1.347  R(train)=+0.527  |  V(val)=+1.385  R(val)=+0.898
2026-08-09 04:44:10,592 INFO   Value epoch 6/35: train=1.0509 rmse=3.31  val=0.7984 val_rmse=2.89 (std=3.2)
    V(train)=+1.181  R(train)=+0.523  |  V(val)=+1.221  R(val)=+0.898
2026-08-09 04:44:14,107 INFO   Value epoch 7/35: train=1.0230 rmse=3.27  val=0.7853 val_rmse=2.86 (std=3.2)
    V(train)=+1.017  R(train)=+0.527  |  V(val)=+1.062  R(val)=+0.898
2026-08-09 04:44:17,583 INFO   Value epoch 8/35: train=1.0046 rmse=3.24  val=0.7775 val_rmse=2.85 (std=3.2)
    V(train)=+0.861  R(train)=+0.526  |  V(val)=+0.913  R(val)=+0.898
2026-08-09 04:44:20,805 INFO   Value epoch 9/35: train=0.9888 rmse=3.21  val=0.7732 val_rmse=2.84 (std=3.2)
    V(train)=+0.723  R(train)=+0.527  |  V(val)=+0.802  R(val)=+0.898
2026-08-09 04:44:24,073 INFO   Value epoch 10/35: train=0.9810 rmse=3.20  val=0.7716 val_rmse=2.84 (std=3.2)
    V(train)=+0.627  R(train)=+0.525  |  V(val)=+0.734  R(val)=+0.898
2026-08-09 04:44:27,420 INFO   Value epoch 11/35: train=0.9742 rmse=3.19  val=0.7714 val_rmse=2.84 (std=3.2)
    V(train)=+0.572  R(train)=+0.526  |  V(val)=+0.704  R(val)=+0.898
2026-08-09 04:44:30,731 INFO   Value epoch 12/35: train=0.9726 rmse=3.19  val=0.7715 val_rmse=2.84 (std=3.2)
    V(train)=+0.556  R(train)=+0.523  |  V(val)=+0.698  R(val)=+0.898
2026-08-09 04:44:34,003 INFO   Value epoch 13/35: train=0.9696 rmse=3.18  val=0.7721 val_rmse=2.84 (std=3.2)
    V(train)=+0.547  R(train)=+0.524  |  V(val)=+0.691  R(val)=+0.898
2026-08-09 04:44:37,205 INFO   Value epoch 14/35: train=0.9659 rmse=3.18  val=0.7725 val_rmse=2.84 (std=3.2)
    V(train)=+0.547  R(train)=+0.526  |  V(val)=+0.692  R(val)=+0.898
2026-08-09 04:44:40,426 INFO   Value epoch 15/35: train=0.9634 rmse=3.17  val=0.7733 val_rmse=2.84 (std=3.2)
    V(train)=+0.540  R(train)=+0.526  |  V(val)=+0.689  R(val)=+0.898
2026-08-09 04:44:40,427 INFO   [value pretrain] early stop at epoch 15 (val stagnant for 4 epochs, best=0.7714)
2026-08-09 04:44:40,427 INFO   [value pretrain] restored best-val weights (val_loss=0.7714)
2026-08-09 04:44:40,427 INFO Value pre-training done (15 epoch(s), final train_loss=0.9634)
2026-08-09 04:44:52,391 INFO BC check after value warm-up: bc_loss=2.3144 (before=2.3791, delta=-0.0646)  OK
2026-08-09 04:44:52,391 INFO Combined pre-training complete.
2026-08-09 04:45:27,605 INFO Pre-PPO eval (rules opp): win=27.3%  mean_rew=-1.354  V=-0.075  R=-1.025  gap=+0.951  outcomes={'box_possession': 35, 'opponent_box_possession': 76, 'miss': 15, 'timeout': 2}
2026-08-09 04:45:27,605 INFO   rew breakdown (rules, per ep): opponent_box=-2.08  get_possession=+0.73  box_possession=+0.55  lose_possession=-0.37  ball_out=-0.23  speed_bonus=+0.10  timeout=-0.03  stamina_penalty=-0.02
2026-08-09 04:46:22,193 INFO Pre-PPO eval (immobile opp): win=67.2%  mean_rew=1.241  V=0.546  R=0.443  gap=+0.103  outcomes={'box_possession': 86, 'miss': 35, 'timeout': 7}
2026-08-09 04:46:22,194 INFO   rew breakdown (immobile, per ep): box_possession=+1.34  ball_out=-1.09  get_possession=+0.87  speed_bonus=+0.26  timeout=-0.11  lose_possession=-0.02  stamina_penalty=-0.01
2026-08-09 04:48:08,147 INFO Pre-PPO eval (self-play):   win=49.2%  mean_rew=0.351  V=0.272  R=-0.110  gap=+0.382  outcomes={'miss': 34, 'box_possession': 63, 'opponent_box_possession': 20, 'timeout': 11}
2026-08-09 04:48:08,147 INFO   rew breakdown (self-play, per ep): opponent_box=-2.27  ball_out=-1.64  get_possession=+1.59  box_possession=+1.30  lose_possession=-0.64  timeout=-0.34  speed_bonus=+0.31  stamina_penalty=-0.04
2026-08-09 04:48:08,147 INFO   [seeded eval] running 12x8 episodes across 7 worker process(es)...
2026-08-09 04:48:22,298 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 04:48:22,299 INFO Baseline (rules vs rules, 12 trials): trainee_win=0.0%  outcomes={'miss': 56, 'timeout': 40}
2026-08-09 04:48:22,299 INFO Frozen decision_net.shoot_logit
2026-08-09 04:48:22,299 INFO Frozen decision_net.pass_logit
2026-08-09 04:48:22,299 INFO Frozen decision_net.tackle_logit
2026-08-09 04:48:22,299 INFO Frozen decision_net.get_possession_raw
2026-08-09 04:48:22,299 INFO Frozen decision_net.mark_logit
2026-08-09 04:48:22,299 INFO Frozen decision_net.hold_position_logit
2026-08-09 04:48:22,299 INFO Frozen decision_net.pass_target_logits
2026-08-09 04:48:22,300 INFO Frozen decision_net.tackle_target_logits
2026-08-09 04:48:22,300 INFO Frozen decision_net.mark_target_logits
2026-08-09 04:48:22,300 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 04:48:22,300 INFO PPO parallel training started: 6 worker(s), ~6000 steps/worker/rollout, steps_so_far=0  target=6,000,000
2026-08-09 04:48:24,202 INFO Frozen decision_net.shoot_logit
2026-08-09 04:48:24,203 INFO Frozen decision_net.pass_logit
2026-08-09 04:48:24,203 INFO Frozen decision_net.tackle_logit
2026-08-09 04:48:24,203 INFO Frozen decision_net.get_possession_raw
2026-08-09 04:48:24,203 INFO Frozen decision_net.mark_logit
2026-08-09 04:48:24,203 INFO Frozen decision_net.hold_position_logit
2026-08-09 04:48:24,203 INFO Frozen decision_net.pass_target_logits
2026-08-09 04:48:24,203 INFO Frozen decision_net.tackle_target_logits
2026-08-09 04:48:24,203 INFO Frozen decision_net.mark_target_logits
2026-08-09 04:48:24,203 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 04:48:24,256 INFO Frozen decision_net.shoot_logit
2026-08-09 04:48:24,256 INFO Frozen decision_net.pass_logit
2026-08-09 04:48:24,256 INFO Frozen decision_net.tackle_logit
2026-08-09 04:48:24,256 INFO Frozen decision_net.get_possession_raw
2026-08-09 04:48:24,256 INFO Frozen decision_net.mark_logit
2026-08-09 04:48:24,256 INFO Frozen decision_net.hold_position_logit
2026-08-09 04:48:24,256 INFO Frozen decision_net.pass_target_logits
2026-08-09 04:48:24,256 INFO Frozen decision_net.tackle_target_logits
2026-08-09 04:48:24,256 INFO Frozen decision_net.mark_target_logits
2026-08-09 04:48:24,256 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 04:48:24,275 INFO Frozen decision_net.shoot_logit
2026-08-09 04:48:24,276 INFO Frozen decision_net.pass_logit
2026-08-09 04:48:24,276 INFO Frozen decision_net.tackle_logit
2026-08-09 04:48:24,276 INFO Frozen decision_net.get_possession_raw
2026-08-09 04:48:24,276 INFO Frozen decision_net.mark_logit
2026-08-09 04:48:24,276 INFO Frozen decision_net.hold_position_logit
2026-08-09 04:48:24,276 INFO Frozen decision_net.pass_target_logits
2026-08-09 04:48:24,276 INFO Frozen decision_net.tackle_target_logits
2026-08-09 04:48:24,276 INFO Frozen decision_net.mark_target_logits
2026-08-09 04:48:24,276 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 04:48:24,304 INFO Frozen decision_net.shoot_logit
2026-08-09 04:48:24,304 INFO Frozen decision_net.pass_logit
2026-08-09 04:48:24,304 INFO Frozen decision_net.tackle_logit
2026-08-09 04:48:24,304 INFO Frozen decision_net.get_possession_raw
2026-08-09 04:48:24,305 INFO Frozen decision_net.mark_logit
2026-08-09 04:48:24,305 INFO Frozen decision_net.hold_position_logit
2026-08-09 04:48:24,305 INFO Frozen decision_net.pass_target_logits
2026-08-09 04:48:24,305 INFO Frozen decision_net.tackle_target_logits
2026-08-09 04:48:24,305 INFO Frozen decision_net.mark_target_logits
2026-08-09 04:48:24,305 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 04:48:24,311 INFO Frozen decision_net.shoot_logit
2026-08-09 04:48:24,311 INFO Frozen decision_net.pass_logit
2026-08-09 04:48:24,311 INFO Frozen decision_net.tackle_logit
2026-08-09 04:48:24,311 INFO Frozen decision_net.get_possession_raw
2026-08-09 04:48:24,311 INFO Frozen decision_net.mark_logit
2026-08-09 04:48:24,312 INFO Frozen decision_net.hold_position_logit
2026-08-09 04:48:24,312 INFO Frozen decision_net.pass_target_logits
2026-08-09 04:48:24,312 INFO Frozen decision_net.tackle_target_logits
2026-08-09 04:48:24,312 INFO Frozen decision_net.mark_target_logits
2026-08-09 04:48:24,312 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 04:48:24,371 INFO Frozen decision_net.shoot_logit
2026-08-09 04:48:24,372 INFO Frozen decision_net.pass_logit
2026-08-09 04:48:24,372 INFO Frozen decision_net.tackle_logit
2026-08-09 04:48:24,372 INFO Frozen decision_net.get_possession_raw
2026-08-09 04:48:24,372 INFO Frozen decision_net.mark_logit
2026-08-09 04:48:24,372 INFO Frozen decision_net.hold_position_logit
2026-08-09 04:48:24,372 INFO Frozen decision_net.pass_target_logits
2026-08-09 04:48:24,372 INFO Frozen decision_net.tackle_target_logits
2026-08-09 04:48:24,372 INFO Frozen decision_net.mark_target_logits
2026-08-09 04:48:24,372 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 04:52:57,842 INFO   [KL mean=0.1351 median=0.1353 > 0.05] ratio percentiles:  p5=0.391  p25=0.886  p50=0.974  p75=1.013  p95=1.289  max=99.198
  move_dir_log_std=[-1.6491615772247314]  kick_dir_log_std=[-1.649237871170044]
2026-08-09 04:52:57,859 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.157  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.196  kick=-0.124  t_att=-0.152
    move_dir=0.812 (min=-3.346 max=1.460)  kick_dir=0.037 (min=-0.000 max=2.041)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.29
  [worst sample] idx=134  ratio=24.325  adv=+1.155  old_lp=-3.295  new_lp=-0.103
    stored move_dir=11.8°  new_mean=2.8°  angular_diff=9.0°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 134  ratio=  24.325  adv=+1.155  lp: old=-3.295  new=-0.103
      rew=+0.0000  ret=+2.0580  val=+0.9032  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9533  sprint_p_new=0.9516  kick_p_new=0.0479  tackle_attempt_p_new=0.0379
    idx=  84  ratio=  20.308  adv=+0.403  lp: old=-3.114  new=-0.103
      rew=+0.0000  ret=+2.1923  val=+1.7891  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9494  sprint_p_new=0.9399  kick_p_new=0.0358  tackle_attempt_p_new=0.0429
  [best sample (highest new_lp)] idx=172  new_lp=-0.035  adv=+6.244  stored move_dir=6.9°  new_mean=1.1°
    per-head contributions: move_dir:0.066  tackle_attempt:-0.022  kick:-0.025  sprint:-0.043
2026-08-09 04:52:57,860 INFO   [advantage] mean=0.000  std=1.000  min=-6.582  max=5.335
2026-08-09 04:52:57,861 INFO   [ratio] mean=0.9517  std=0.5390  min=0.0000  max=99.1978  clipped=30.0%
2026-08-09 04:52:57,861 INFO   [exec head grad norm] move_direction=0.200  exec_move=0.037  sprint=0.029  kick=0.034  kick_direction=0.040  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.031
2026-08-09 04:52:57,861 INFO   [exec continuous log_std] move_direction: start=-1.6500 end=-1.6492   kick_direction: start=-1.6500 end=-1.6492
2026-08-09 04:52:57,861 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0034≈0.20°/step  epoch≈11.8°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0037≈0.21°/step  epoch≈12.6°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 04:52:57,861 INFO   [exec discrete Δlogit per opt step] exec_move=0.0043  sprint=0.0064  kick=0.0037  tackle_attempt=0.0034
2026-08-09 04:52:57,862 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0004  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0212  sprint=+0.0293  kick=+0.0062  tackle_attempt=+0.0038  move_dir=+0.0515  kick_dir=+0.0227
2026-08-09 04:52:57,862 INFO   [grad clip] main: 60/60 steps clipped (100%)  pre-clip norm mean=0.798 max=2.141  limit=0.4
              direction: 60/60 steps clipped (100%)  pre-clip norm mean=0.206 max=0.889  limit=0.02
2026-08-09 04:52:57,927 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=36,000  speed=814/s  reward=1.30
  loss     policy=0.0352  value=0.8121(x0.5)=0.4061
           entropy=1.5709  kl=0.1351
  value    V=0.57±0.99  R=0.73±1.87  adv=0.16±1.69
  moves    mv_ls=[-1.6492] (σ≈0.19, ≈11°) g=1.16e-02
           kk_ls=[-1.6492] (σ≈0.19, ≈11°)
  heads    move= 27 get_poss= 73 exec_move= 87 sprint= 45 kick=  4 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0411 kick_prob=0.0480
  vs       vs[win/loss/tout/miss]  vs_immobile(508): 69.7%/0.0%/8.1%/22.2%
  ep_len   21.1±13.6s  (n=508, min=1.7s, max=50.1s)
  reward   get_possession=+452.00  ball_out=-490.00  box_possession=+708.00
           speed_bonus=+176.53  timeout=-82.00  stamina_penalty=-3.91
  rew/ep   (mean/std/min/max per episode, 508 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.890    0.313    +0.000    +1.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    +0.000    0.000    +0.000    +0.000
  ball_out          -0.965    2.952   -10.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.394    0.919    +0.000    +2.000
  speed_bonus       +0.347    0.379    +0.000    +1.237
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.161    0.545    -2.000    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.008    0.008    -0.041    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     453    +0.013    0.111     +1.663     1.891     +0.866      3.9954      1.625     3.161
  ball_out            49    -0.014    0.369    -10.000     0.000     -7.745     67.9037      7.745    10.751
  box_possession     354    +0.020    0.197     +2.582     0.368     +1.302      2.0838      1.302     2.297
  speed_bonus        319    +0.005    0.061     +2.577     0.352     +1.289      2.0477      1.289     2.299
  timeout             41    -0.002    0.067     -2.010     0.010     -2.801      8.3394      2.801     3.722
  stamina_penalty     330    -0.000    0.001     +2.230     1.288     +0.954      2.5574      1.403     2.851
  gae/td   mean_return=+0.733  std_return=1.874  mean_gae=+0.158  mean_sq_td=2.8920
──────────────────────────────────────────────────────────────────────
2026-08-09 04:52:57,951 INFO Saved checkpoint: checkpoints/phase1_run49/checkpoint1.pt
2026-08-09 04:52:57,951 INFO Logging to checkpoints/phase1_run49/training_log2.txt
2026-08-09 04:52:57,952 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 04:53:21,346 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 04:53:21,347 INFO   [eval vs immobile] step=36,000  seeds=16x8  win=69%  mean_rew=1.701±3.222  V=0.717  gap=-0.984  outcomes={'box_possession': 88, 'timeout': 11, 'miss': 29}
2026-08-09 04:53:21,349 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 04:53:51,923 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 04:53:51,925 INFO   [eval vs rules] step=36,000  seeds=16x8  win=20%  mean_rew=-1.135±3.369  V=-0.353  gap=+0.782  outcomes={'box_possession': 25, 'miss': 49, 'opponent_box_possession': 35, 'timeout': 19}
