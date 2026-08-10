2026-08-10 01:42:13,133 INFO Checkpoint dir: checkpoints/phase1_run70
2026-08-10 01:42:13,293 INFO Starting training: phase=phase1_get_possession, total_steps=6,000,000
2026-08-10 01:42:13,294 INFO Phase description: 1v1 scenario: learn to get possession and bring the ball toward the opponent box.  Decision network frozen except Move/GetPossession and latent vector.
/home/vincent/Documents/not_work/repos/FootballCoach/.venv/lib/python3.12/site-packages/torch/_compile.py:54: UserWarning: optimizer contains a parameter group with duplicate parameters; in future, this will cause an error; see github.com/pytorch/pytorch/issues/40967 for more information
  return disable_fn(*args, **kwargs)
2026-08-10 01:42:15,765 INFO Logging to checkpoints/phase1_run70/training_log1.txt
2026-08-10 01:42:16,454 INFO Loading 3000 demonstration file(s) from demonstrations/phase1_long
2026-08-10 01:42:40,305 INFO Dataset: 1,338,665 steps loaded
2026-08-10 01:42:40,309 INFO Offline BC dataset: 1,338,665 steps from demonstrations/phase1_long/
2026-08-10 01:42:40,309 INFO Reward diagnostic: running 40 rules-vs-rules episodes...
2026-08-10 01:42:42,166 INFO Reward diagnostic (40 ep, rules vs rules): mean_ep_rew=2.44  per_episode: get_possession=+1.10  lose_possession=-0.34  box_possession=+2.90  opponent_box=-1.00  step_penalty=-0.08  stamina_penalty=-0.14
2026-08-10 01:42:42,205 INFO BC pos_weight (auto-computed from dataset): kick=1.35  tackle_attempt=1.35
2026-08-10 01:42:42,205 INFO Combined BC + value pre-training: 18 epoch(s), batch_size=2048, dataset=1,338,665 steps, rollout_steps=120000
2026-08-10 01:42:43,056 INFO Phase 0 — decision-net warm-up (BC + self.value_net MSE; single value head convention): 12 epoch(s), gamma=0.995, returns mean=3.59  std=1.18  lr=0.005  phase0_value_coef=1.0  split: 582,196 train / 102,167 val rows
/home/vincent/Documents/not_work/repos/FootballCoach/src/footballcoach/ai/ppo/bc.py:710: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /__w/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:822.)
  "decision":       float(dec_loss[valid].mean()),
2026-08-10 01:45:05,610 INFO   Phase 0 epoch 1/12: loss=4.5767  dec_bc=2.5184  bc_adj=2.0777(floor=0.4407)  val_mse=2.0584(x1.0)=2.0584
2026-08-10 01:45:12,304 INFO     val  p0_val_loss=3.2398  bc_adj=2.0192  val_mse=0.7799  best=3.2398  (improved)
2026-08-10 01:48:32,914 INFO   Phase 0 epoch 2/12: loss=3.2601  dec_bc=2.4619  bc_adj=2.0212(floor=0.4407)  val_mse=0.7982(x1.0)=0.7982
2026-08-10 01:48:39,095 INFO     val  p0_val_loss=3.1837  bc_adj=2.0145  val_mse=0.7285  best=3.1837  (improved)
2026-08-10 01:50:14,589 INFO   Phase 0 epoch 3/12: loss=3.2322  dec_bc=2.4602  bc_adj=2.0195(floor=0.4407)  val_mse=0.7720(x1.0)=0.7720
2026-08-10 01:50:20,741 INFO     val  p0_val_loss=3.2544  bc_adj=2.0278  val_mse=0.7859  best=3.1837  (patience 1/3)
2026-08-10 01:51:56,809 INFO   Phase 0 epoch 4/12: loss=3.2167  dec_bc=2.4587  bc_adj=2.0180(floor=0.4407)  val_mse=0.7581(x1.0)=0.7581
2026-08-10 01:52:02,855 INFO     val  p0_val_loss=3.2338  bc_adj=2.0225  val_mse=0.7706  best=3.1837  (patience 2/3)
2026-08-10 01:53:43,767 INFO   Phase 0 epoch 5/12: loss=3.2218  dec_bc=2.4637  bc_adj=2.0230(floor=0.4407)  val_mse=0.7580(x1.0)=0.7580
2026-08-10 01:53:50,256 INFO     val  p0_val_loss=3.1841  bc_adj=2.0130  val_mse=0.7305  best=3.1837  (patience 3/3)
2026-08-10 01:53:50,257 INFO   [Phase 0] early stop at epoch 5 (val stagnant for 3 epochs, best=3.1837)
2026-08-10 01:53:50,258 INFO   [Phase 0] restored best-val weights (p0_val_loss=3.1837)
2026-08-10 01:53:50,259 INFO Phase 0 done (decision-net BC + critic value_head warm-up, 12 epoch(s))
2026-08-10 01:53:50,272 INFO   BC pretrain split: 582,196 train rows  |  102,167 val rows
2026-08-10 01:53:50,664 INFO   Downsample trivial rows (epoch 1): 268,796/684,363 (39.3%) rows classified trivial, excluding ~177,405 this epoch (frac=0.66)
2026-08-10 01:56:28,735 INFO   BC epoch 1/18  (158.5s)
    loss       bc=2.6219  bc_adj=1.9526(floor=0.6693)
    heads      dir_cos=0.751  kick_dir_cos=0.956
               move_prob=0.910  sprint_prob=0.370  kick_prob=0.271  tackle_prob=0.018
    pr/rec     kick:   p=0.657  r=0.473  f1=0.550  (tp=95971 fp=50095 fn=106771)
               tackle: p=0.003  r=0.015  f1=0.005  (tp=6 fp=2162 fn=404)
    breakdown  decision=0.241  exec_bce=1.334  sprint=0.521  move=0.246  tackle_attempt=0.081  direction=0.700
               region=0.074  kick=0.48582  kick_direction=0.13115  kick_power=0.00698  kick_spin=0.00062
2026-08-10 01:56:34,975 INFO     val        bc_val_loss=1.3388  best=1.3388  (improved)
2026-08-10 01:56:34,991 INFO   Downsample trivial rows (epoch 2): 268,796/684,363 (39.3%) rows classified trivial, excluding ~177,405 this epoch (frac=0.66)
2026-08-10 01:59:09,541 INFO   BC epoch 2/18  (154.6s)
    loss       bc=1.2364  bc_adj=0.5670(floor=0.6693)
    heads      dir_cos=0.968  kick_dir_cos=0.987
               move_prob=0.911  sprint_prob=0.360  kick_prob=0.251  tackle_prob=0.011
    pr/rec     kick:   p=0.887  r=0.917  f1=0.902  (tp=185926 fp=23655 fn=16816)
               tackle: p=nan  r=0.000  f1=nan  (tp=0 fp=0 fn=410)
    breakdown  decision=0.224  exec_bce=0.674  sprint=0.310  move=0.120  tackle_attempt=0.060  direction=0.090
               region=0.015  kick=0.18442  kick_direction=0.04023  kick_power=0.00211  kick_spin=0.00007
2026-08-10 01:59:15,710 INFO     val        bc_val_loss=1.2899  best=1.2899  (improved)
2026-08-10 01:59:15,724 INFO   Downsample trivial rows (epoch 3): 268,796/684,363 (39.3%) rows classified trivial, excluding ~177,405 this epoch (frac=0.66)
2026-08-10 02:01:46,437 INFO   BC epoch 3/18  (150.7s)
    loss       bc=1.0961  bc_adj=0.4267(floor=0.6693)
    heads      dir_cos=0.977  kick_dir_cos=0.992
               move_prob=0.911  sprint_prob=0.359  kick_prob=0.249  tackle_prob=0.011
    pr/rec     kick:   p=0.902  r=0.936  f1=0.919  (tp=189693 fp=20506 fn=13049)
               tackle: p=nan  r=0.000  f1=nan  (tp=0 fp=0 fn=410)
    breakdown  decision=0.223  exec_bce=0.566  sprint=0.241  move=0.107  tackle_attempt=0.059  direction=0.064
               region=0.015  kick=0.15850  kick_direction=0.02501  kick_power=0.00178  kick_spin=0.00004
2026-08-10 02:01:52,420 INFO     val        bc_val_loss=0.9943  best=0.9943  (improved)
2026-08-10 02:01:52,441 INFO   Downsample trivial rows (epoch 4): 268,796/684,363 (39.3%) rows classified trivial, excluding ~177,405 this epoch (frac=0.66)
2026-08-10 02:04:25,108 INFO   BC epoch 4/18  (152.7s)
    loss       bc=1.0106  bc_adj=0.3413(floor=0.6693)
    heads      dir_cos=0.981  kick_dir_cos=0.994
               move_prob=0.912  sprint_prob=0.359  kick_prob=0.247  tackle_prob=0.011
    pr/rec     kick:   p=0.928  r=0.954  f1=0.941  (tp=193466 fp=15047 fn=9276)
               tackle: p=1.000  r=0.005  f1=0.010  (tp=2 fp=0 fn=408)
    breakdown  decision=0.222  exec_bce=0.496  sprint=0.207  move=0.098  tackle_attempt=0.059  direction=0.054
               region=0.012  kick=0.13197  kick_direction=0.01681  kick_power=0.00140  kick_spin=0.00002
2026-08-10 02:04:31,239 INFO     val        bc_val_loss=0.9557  best=0.9557  (improved)
2026-08-10 02:04:31,260 INFO   Downsample trivial rows (epoch 5): 268,796/684,363 (39.3%) rows classified trivial, excluding ~177,405 this epoch (frac=0.66)
2026-08-10 02:07:02,285 INFO   BC epoch 5/18  (151.0s)
    loss       bc=0.9808  bc_adj=0.3115(floor=0.6693)
    heads      dir_cos=0.982  kick_dir_cos=0.996
               move_prob=0.912  sprint_prob=0.360  kick_prob=0.246  tackle_prob=0.011
    pr/rec     kick:   p=0.933  r=0.960  f1=0.946  (tp=194623 fp=13982 fn=8119)
               tackle: p=1.000  r=0.037  f1=0.071  (tp=15 fp=0 fn=395)
    breakdown  decision=0.222  exec_bce=0.472  sprint=0.193  move=0.097  tackle_attempt=0.058  direction=0.051
               region=0.011  kick=0.12426  kick_direction=0.01229  kick_power=0.00140  kick_spin=0.00001
2026-08-10 02:07:08,446 INFO     val        bc_val_loss=0.9231  best=0.9231  (improved)
2026-08-10 02:07:08,461 INFO   Downsample trivial rows (epoch 6): 268,796/684,363 (39.3%) rows classified trivial, excluding ~177,405 this epoch (frac=0.66)
2026-08-10 02:09:42,869 INFO   BC epoch 6/18  (154.4s)
    loss       bc=0.9541  bc_adj=0.2848(floor=0.6693)
    heads      dir_cos=0.983  kick_dir_cos=0.996
               move_prob=0.911  sprint_prob=0.360  kick_prob=0.246  tackle_prob=0.011
    pr/rec     kick:   p=0.940  r=0.964  f1=0.952  (tp=195522 fp=12460 fn=7220)
               tackle: p=0.938  r=0.146  f1=0.253  (tp=60 fp=4 fn=350)
    breakdown  decision=0.222  exec_bce=0.451  sprint=0.180  move=0.096  tackle_attempt=0.058  direction=0.047
               region=0.011  kick=0.11720  kick_direction=0.01110  kick_power=0.00135  kick_spin=0.00001
2026-08-10 02:09:49,020 INFO     val        bc_val_loss=0.9842  best=0.9231  (patience 1/3)
2026-08-10 02:09:49,033 INFO   Downsample trivial rows (epoch 7): 268,796/684,363 (39.3%) rows classified trivial, excluding ~177,405 this epoch (frac=0.66)
2026-08-10 02:12:21,301 INFO   BC epoch 7/18  (152.3s)
    loss       bc=0.9391  bc_adj=0.2697(floor=0.6693)
    heads      dir_cos=0.984  kick_dir_cos=0.997
               move_prob=0.912  sprint_prob=0.360  kick_prob=0.245  tackle_prob=0.011
    pr/rec     kick:   p=0.944  r=0.968  f1=0.956  (tp=196164 fp=11536 fn=6578)
               tackle: p=0.797  r=0.363  f1=0.499  (tp=149 fp=38 fn=261)
    breakdown  decision=0.221  exec_bce=0.438  sprint=0.174  move=0.094  tackle_attempt=0.058  direction=0.045
               region=0.010  kick=0.11221  kick_direction=0.00958  kick_power=0.00124  kick_spin=0.00001
2026-08-10 02:12:27,447 INFO     val        bc_val_loss=0.9075  best=0.9075  (improved)
2026-08-10 02:12:27,465 INFO   Downsample trivial rows (epoch 8): 268,796/684,363 (39.3%) rows classified trivial, excluding ~177,405 this epoch (frac=0.66)
2026-08-10 02:15:00,932 INFO   BC epoch 8/18  (153.5s)
    loss       bc=0.9353  bc_adj=0.2660(floor=0.6693)
    heads      dir_cos=0.984  kick_dir_cos=0.997
               move_prob=0.911  sprint_prob=0.360  kick_prob=0.245  tackle_prob=0.011
    pr/rec     kick:   p=0.945  r=0.968  f1=0.956  (tp=196218 fp=11502 fn=6524)
               tackle: p=0.840  r=0.590  f1=0.693  (tp=242 fp=46 fn=168)
    breakdown  decision=0.221  exec_bce=0.434  sprint=0.171  move=0.094  tackle_attempt=0.057  direction=0.044
               region=0.011  kick=0.11171  kick_direction=0.00921  kick_power=0.00130  kick_spin=0.00001
2026-08-10 02:15:07,057 INFO     val        bc_val_loss=0.9110  best=0.9075  (patience 1/3)
2026-08-10 02:15:07,074 INFO   Downsample trivial rows (epoch 9): 268,796/684,363 (39.3%) rows classified trivial, excluding ~177,405 this epoch (frac=0.66)
2026-08-10 02:17:40,320 INFO   BC epoch 9/18  (153.3s)
    loss       bc=0.9214  bc_adj=0.2520(floor=0.6693)
    heads      dir_cos=0.985  kick_dir_cos=0.997
               move_prob=0.911  sprint_prob=0.360  kick_prob=0.245  tackle_prob=0.011
    pr/rec     kick:   p=0.949  r=0.970  f1=0.959  (tp=196616 fp=10664 fn=6126)
               tackle: p=0.839  r=0.634  f1=0.722  (tp=260 fp=50 fn=150)
    breakdown  decision=0.221  exec_bce=0.424  sprint=0.165  move=0.093  tackle_attempt=0.057  direction=0.042
               region=0.010  kick=0.10815  kick_direction=0.00812  kick_power=0.00127  kick_spin=0.00000
2026-08-10 02:17:46,380 INFO     val        bc_val_loss=0.8708  best=0.8708  (improved)
2026-08-10 02:17:46,400 INFO   Downsample trivial rows (epoch 10): 268,796/684,363 (39.3%) rows classified trivial, excluding ~177,405 this epoch (frac=0.66)
2026-08-10 02:20:14,437 INFO   BC epoch 10/18  (148.1s)
    loss       bc=0.9166  bc_adj=0.2473(floor=0.6693)
    heads      dir_cos=0.985  kick_dir_cos=0.997
               move_prob=0.912  sprint_prob=0.360  kick_prob=0.245  tackle_prob=0.011
    pr/rec     kick:   p=0.948  r=0.970  f1=0.959  (tp=196635 fp=10703 fn=6107)
               tackle: p=0.834  r=0.724  f1=0.775  (tp=297 fp=59 fn=113)
    breakdown  decision=0.221  exec_bce=0.419  sprint=0.162  move=0.093  tackle_attempt=0.057  direction=0.041
               region=0.012  kick=0.10742  kick_direction=0.00811  kick_power=0.00120  kick_spin=0.00000
2026-08-10 02:20:20,465 INFO     val        bc_val_loss=0.8730  best=0.8708  (patience 1/3)
2026-08-10 02:20:20,476 INFO   Downsample trivial rows (epoch 11): 268,796/684,363 (39.3%) rows classified trivial, excluding ~177,405 this epoch (frac=0.66)
2026-08-10 02:22:46,629 INFO   BC epoch 11/18  (146.2s)
    loss       bc=0.9038  bc_adj=0.2344(floor=0.6693)
    heads      dir_cos=0.986  kick_dir_cos=0.998
               move_prob=0.912  sprint_prob=0.360  kick_prob=0.244  tackle_prob=0.011
    pr/rec     kick:   p=0.955  r=0.974  f1=0.964  (tp=197406 fp=9298 fn=5336)
               tackle: p=0.878  r=0.861  f1=0.869  (tp=353 fp=49 fn=57)
    breakdown  decision=0.221  exec_bce=0.409  sprint=0.160  move=0.091  tackle_attempt=0.057  direction=0.040
               region=0.010  kick=0.10173  kick_direction=0.00699  kick_power=0.00116  kick_spin=0.00000
2026-08-10 02:22:52,580 INFO     val        bc_val_loss=0.8824  best=0.8708  (patience 2/3)
2026-08-10 02:22:52,598 INFO   Downsample trivial rows (epoch 12): 268,796/684,363 (39.3%) rows classified trivial, excluding ~177,405 this epoch (frac=0.66)
2026-08-10 02:25:18,993 INFO   BC epoch 12/18  (146.4s)
    loss       bc=0.8983  bc_adj=0.2290(floor=0.6693)
    heads      dir_cos=0.986  kick_dir_cos=0.998
               move_prob=0.912  sprint_prob=0.360  kick_prob=0.244  tackle_prob=0.011
    pr/rec     kick:   p=0.955  r=0.974  f1=0.965  (tp=197540 fp=9271 fn=5202)
               tackle: p=0.911  r=0.878  f1=0.894  (tp=360 fp=35 fn=50)
    breakdown  decision=0.221  exec_bce=0.405  sprint=0.157  move=0.091  tackle_attempt=0.057  direction=0.039
               region=0.010  kick=0.10098  kick_direction=0.00668  kick_power=0.00119  kick_spin=0.00000
2026-08-10 02:25:24,914 INFO     val        bc_val_loss=0.8745  best=0.8708  (patience 3/3)
2026-08-10 02:25:24,914 INFO   [BC pretrain] early stop at epoch 12 (val stagnant for 3 epochs, best=0.8708)
2026-08-10 02:25:24,916 INFO   [BC pretrain] restored best-val weights (bc_val_loss=0.8708)
2026-08-10 02:25:24,916 INFO BC pre-training done (18 epoch(s), final bc_loss=0.8983)
2026-08-10 02:25:24,916 INFO Value pre-training: 120000 steps, 45 epochs, lr=0.005, batch_size=2048
2026-08-10 02:25:24,917 INFO   [value pretrain rollout] parallel collection: 6 worker(s), ~20000 steps/worker
2026-08-10 02:25:26,762 INFO Frozen decision_net.shoot_logit
2026-08-10 02:25:26,763 INFO Frozen decision_net.pass_logit
2026-08-10 02:25:26,763 INFO Frozen decision_net.tackle_logit
2026-08-10 02:25:26,763 INFO Frozen decision_net.shoot_logit
2026-08-10 02:25:26,763 INFO Frozen decision_net.get_possession_raw
2026-08-10 02:25:26,763 INFO Frozen decision_net.mark_logit
2026-08-10 02:25:26,763 INFO Frozen decision_net.pass_logit
2026-08-10 02:25:26,763 INFO Frozen decision_net.hold_position_logit
2026-08-10 02:25:26,763 INFO Frozen decision_net.tackle_logit
2026-08-10 02:25:26,763 INFO Frozen decision_net.pass_target_logits
2026-08-10 02:25:26,763 INFO Frozen decision_net.get_possession_raw
2026-08-10 02:25:26,763 INFO Frozen decision_net.tackle_target_logits
2026-08-10 02:25:26,763 INFO Frozen decision_net.mark_logit
2026-08-10 02:25:26,763 INFO Frozen decision_net.mark_target_logits
2026-08-10 02:25:26,763 INFO Frozen decision_net.hold_position_logit
2026-08-10 02:25:26,763 INFO Frozen decision_net.pass_target_logits
2026-08-10 02:25:26,763 INFO Frozen decision_net.tackle_target_logits
2026-08-10 02:25:26,763 INFO Frozen decision_net.mark_target_logits
2026-08-10 02:25:26,763 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-10 02:25:26,763 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-10 02:25:26,802 INFO Frozen decision_net.shoot_logit
2026-08-10 02:25:26,802 INFO Frozen decision_net.pass_logit
2026-08-10 02:25:26,802 INFO Frozen decision_net.tackle_logit
2026-08-10 02:25:26,802 INFO Frozen decision_net.get_possession_raw
2026-08-10 02:25:26,802 INFO Frozen decision_net.mark_logit
2026-08-10 02:25:26,802 INFO Frozen decision_net.hold_position_logit
2026-08-10 02:25:26,802 INFO Frozen decision_net.pass_target_logits
2026-08-10 02:25:26,802 INFO Frozen decision_net.tackle_target_logits
2026-08-10 02:25:26,802 INFO Frozen decision_net.mark_target_logits
2026-08-10 02:25:26,802 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-10 02:25:26,803 INFO Frozen decision_net.shoot_logit
2026-08-10 02:25:26,803 INFO Frozen decision_net.pass_logit
2026-08-10 02:25:26,803 INFO Frozen decision_net.tackle_logit
2026-08-10 02:25:26,803 INFO Frozen decision_net.get_possession_raw
2026-08-10 02:25:26,803 INFO Frozen decision_net.mark_logit
2026-08-10 02:25:26,803 INFO Frozen decision_net.hold_position_logit
2026-08-10 02:25:26,803 INFO Frozen decision_net.pass_target_logits
2026-08-10 02:25:26,803 INFO Frozen decision_net.tackle_target_logits
2026-08-10 02:25:26,803 INFO Frozen decision_net.mark_target_logits
2026-08-10 02:25:26,803 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-10 02:25:26,805 INFO Frozen decision_net.shoot_logit
2026-08-10 02:25:26,805 INFO Frozen decision_net.pass_logit
2026-08-10 02:25:26,805 INFO Frozen decision_net.tackle_logit
2026-08-10 02:25:26,805 INFO Frozen decision_net.get_possession_raw
2026-08-10 02:25:26,805 INFO Frozen decision_net.mark_logit
2026-08-10 02:25:26,805 INFO Frozen decision_net.hold_position_logit
2026-08-10 02:25:26,805 INFO Frozen decision_net.pass_target_logits
2026-08-10 02:25:26,805 INFO Frozen decision_net.tackle_target_logits
2026-08-10 02:25:26,805 INFO Frozen decision_net.mark_target_logits
2026-08-10 02:25:26,805 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-10 02:25:26,823 INFO Frozen decision_net.shoot_logit
2026-08-10 02:25:26,824 INFO Frozen decision_net.pass_logit
2026-08-10 02:25:26,824 INFO Frozen decision_net.tackle_logit
2026-08-10 02:25:26,824 INFO Frozen decision_net.get_possession_raw
2026-08-10 02:25:26,824 INFO Frozen decision_net.mark_logit
2026-08-10 02:25:26,824 INFO Frozen decision_net.hold_position_logit
2026-08-10 02:25:26,824 INFO Frozen decision_net.pass_target_logits
2026-08-10 02:25:26,824 INFO Frozen decision_net.tackle_target_logits
2026-08-10 02:25:26,824 INFO Frozen decision_net.mark_target_logits
2026-08-10 02:25:26,824 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-10 02:28:05,632 INFO   [value pretrain rollout] dropped 670 trailing (incomplete-episode) step(s) across workers before MC-return fit
2026-08-10 02:28:05,662 INFO   [value pretrain rollout] mean_return=2.19 (955 episode(s))  vs[win/loss/tout/miss/inval]  vs_rules(0): n/a  vs_immobile(955): 62.7%/0.0%/27.2%/2.6%/7.4%  vs_neural(0): n/a
2026-08-10 02:28:05,662 INFO   [value pretrain rollout] ep_len 37.4±30.4s  (n=955, min=0.2s, max=80.1s)
2026-08-10 02:28:05,663 INFO   [value pretrain rollout] rew/ep (mean/std/min/max per episode, 955 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.724    0.468    +0.000    +3.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.008    0.096    -1.800    +0.000
  ball_out          -0.105    0.716    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +2.509    1.934    +0.000    +4.000
  speed_bonus       +0.000    0.000    +0.000    +0.000
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.545    0.890    -2.000    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  step_penalty      -0.250    0.202    -0.534    -0.002
  stamina_penalty    -0.132    0.099    -0.458    +0.000
2026-08-10 02:28:06,372 INFO   Value pretrain split: 812 train eps (103968 steps)  |  143 val eps (15362 steps)
2026-08-10 02:28:31,959 INFO   Value epoch 1/45: train=0.9944 rmse=2.36  val=1.1305 val_rmse=2.52 (std=2.4)
    V(train)=+0.046  R(train)=+0.410  |  V(val)=+0.064  R(val)=+1.008
2026-08-10 02:28:57,168 INFO   Value epoch 2/45: train=0.9252 rmse=2.28  val=0.9734 val_rmse=2.34 (std=2.4)
    V(train)=+0.099  R(train)=+0.409  |  V(val)=+0.210  R(val)=+1.008
2026-08-10 02:29:22,568 INFO   Value epoch 3/45: train=0.8070 rmse=2.13  val=0.8348 val_rmse=2.16 (std=2.4)
    V(train)=+0.300  R(train)=+0.409  |  V(val)=+0.434  R(val)=+1.008
2026-08-10 02:29:47,784 INFO   Value epoch 4/45: train=0.7505 rmse=2.05  val=0.7604 val_rmse=2.06 (std=2.4)
    V(train)=+0.390  R(train)=+0.409  |  V(val)=+0.510  R(val)=+1.008
2026-08-10 02:30:13,157 INFO   Value epoch 5/45: train=0.7039 rmse=1.99  val=0.7201 val_rmse=2.01 (std=2.4)
    V(train)=+0.394  R(train)=+0.409  |  V(val)=+0.541  R(val)=+1.008
2026-08-10 02:30:38,584 INFO   Value epoch 6/45: train=0.6713 rmse=1.94  val=0.7459 val_rmse=2.04 (std=2.4)
    V(train)=+0.393  R(train)=+0.409  |  V(val)=+0.441  R(val)=+1.008
2026-08-10 02:31:03,981 INFO   Value epoch 7/45: train=0.6538 rmse=1.91  val=0.7348 val_rmse=2.03 (std=2.4)
    V(train)=+0.396  R(train)=+0.409  |  V(val)=+0.576  R(val)=+1.008
2026-08-10 02:31:29,262 INFO   Value epoch 8/45: train=0.6411 rmse=1.90  val=0.7201 val_rmse=2.01 (std=2.4)
    V(train)=+0.397  R(train)=+0.409  |  V(val)=+0.682  R(val)=+1.008
2026-08-10 02:31:53,774 INFO   Value epoch 9/45: train=0.6310 rmse=1.88  val=0.7396 val_rmse=2.04 (std=2.4)
    V(train)=+0.400  R(train)=+0.409  |  V(val)=+0.569  R(val)=+1.008
2026-08-10 02:32:18,328 INFO   Value epoch 10/45: train=0.6222 rmse=1.87  val=0.7441 val_rmse=2.04 (std=2.4)
    V(train)=+0.397  R(train)=+0.410  |  V(val)=+0.603  R(val)=+1.008
2026-08-10 02:32:42,859 INFO   Value epoch 11/45: train=0.6149 rmse=1.86  val=0.7393 val_rmse=2.04 (std=2.4)
    V(train)=+0.398  R(train)=+0.409  |  V(val)=+0.643  R(val)=+1.008
2026-08-10 02:33:08,971 INFO   Value epoch 12/45: train=0.6078 rmse=1.85  val=0.7402 val_rmse=2.04 (std=2.4)
    V(train)=+0.399  R(train)=+0.409  |  V(val)=+0.647  R(val)=+1.008
2026-08-10 02:33:35,943 INFO   Value epoch 13/45: train=0.6009 rmse=1.83  val=0.7363 val_rmse=2.03 (std=2.4)
    V(train)=+0.401  R(train)=+0.409  |  V(val)=+0.648  R(val)=+1.008
2026-08-10 02:34:02,306 INFO   Value epoch 14/45: train=0.5944 rmse=1.82  val=0.7532 val_rmse=2.05 (std=2.4)
    V(train)=+0.396  R(train)=+0.409  |  V(val)=+0.629  R(val)=+1.008
2026-08-10 02:34:28,334 INFO   Value epoch 15/45: train=0.5883 rmse=1.82  val=0.7493 val_rmse=2.05 (std=2.4)
    V(train)=+0.400  R(train)=+0.409  |  V(val)=+0.663  R(val)=+1.008
2026-08-10 02:34:28,334 INFO   [value pretrain] early stop at epoch 15 (val stagnant for 10 epochs, best=0.7201)
2026-08-10 02:34:28,335 INFO   [value pretrain] restored best-val weights (val_loss=0.7201)
2026-08-10 02:34:28,335 INFO Value pre-training done (15 epoch(s), final train_loss=0.5883)
2026-08-10 02:35:09,699 INFO BC check after value warm-up: bc_loss=0.8681 (before=0.8983, delta=-0.0302)  OK
2026-08-10 02:35:09,699 INFO Combined pre-training complete.
2026-08-10 02:35:09,715 INFO Pre-trained checkpoint saved: checkpoints/phase1_run70/checkpoint_pretrained.pt
2026-08-10 02:35:09,718 INFO   [seeded eval] running 16x10 episodes across 7 worker process(es)...
2026-08-10 02:35:11,818 INFO Loaded checkpoint: checkpoints/phase1_run70/checkpoint_pretrained.pt (step 0)
2026-08-10 02:35:11,824 INFO Loaded checkpoint: checkpoints/phase1_run70/checkpoint_pretrained.pt (step 0)
2026-08-10 02:35:11,833 INFO Loaded checkpoint: checkpoints/phase1_run70/checkpoint_pretrained.pt (step 0)
2026-08-10 02:35:11,838 INFO Loaded checkpoint: checkpoints/phase1_run70/checkpoint_pretrained.pt (step 0)
2026-08-10 02:35:11,932 INFO Loaded checkpoint: checkpoints/phase1_run70/checkpoint_pretrained.pt (step 0)
2026-08-10 02:35:11,943 INFO Loaded checkpoint: checkpoints/phase1_run70/checkpoint_pretrained.pt (step 0)
2026-08-10 02:35:12,006 INFO Loaded checkpoint: checkpoints/phase1_run70/checkpoint_pretrained.pt (step 0)
2026-08-10 02:35:22,601 INFO   [seeded eval] all workers finished, merging results.
2026-08-10 02:35:22,603 INFO Pre-PPO eval (rules opp): win=22.5%  mean_rew=-1.715  V=0.870  R=-1.381  gap=+2.250  outcomes={'opponent_box_possession': 106, 'box_possession': 36, 'miss': 1, 'invalid': 17}
2026-08-10 02:35:22,603 INFO   rew breakdown (rules, per ep): opponent_box=-2.65  box_possession=+0.90  get_possession=+0.40  lose_possession=-0.15  stamina_penalty=-0.10  step_penalty=-0.08  ball_out=-0.03
2026-08-10 02:35:22,603 INFO   [seeded eval] running 16x10 episodes across 7 worker process(es)...
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/vincent/Documents/not_work/repos/FootballCoach/src/footballcoach/ai/scripts/train.py", line 566, in <module>
    main()
  File "/home/vincent/Documents/not_work/repos/FootballCoach/src/footballcoach/ai/scripts/train.py", line 490, in main
    immobile_stats = _run_evaluation(trainer, immobile_env, _pre_ppo_n_seeds, checkpoint_path=_eval_ckpt_path)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vincent/Documents/not_work/repos/FootballCoach/src/footballcoach/ai/scripts/evaluate.py", line 231, in _run_evaluation
    result = run_seeded_evaluation_parallel(worker_factory, seeds, repeats, n_workers=n_workers)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vincent/Documents/not_work/repos/FootballCoach/src/footballcoach/ai/eval/seeded_eval.py", line 288, in run_seeded_evaluation_parallel
    results = pool.starmap(
              ^^^^^^^^^^^^^
  File "/home/vincent/.pyenv/versions/3.12.7/lib/python3.12/multiprocessing/pool.py", line 375, in starmap
    return self._map_async(func, iterable, starmapstar, chunksize).get()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vincent/.pyenv/versions/3.12.7/lib/python3.12/multiprocessing/pool.py", line 774, in get
    raise self._value
  File "/home/vincent/.pyenv/versions/3.12.7/lib/python3.12/multiprocessing/pool.py", line 540, in _handle_tasks
    put(task)
  File "/home/vincent/.pyenv/versions/3.12.7/lib/python3.12/multiprocessing/connection.py", line 206, in send
    self._send_bytes(_ForkingPickler.dumps(obj))
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vincent/.pyenv/versions/3.12.7/lib/python3.12/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
AttributeError: Can't get local object 'main.<locals>._immobile_build'
2026-08-10 02:37:43,795 INFO Checkpoint dir: checkpoints/phase1_run71
2026-08-10 02:37:43,859 INFO Starting training: phase=phase1_get_possession, total_steps=6,000,000
2026-08-10 02:37:43,859 INFO Phase description: 1v1 scenario: learn to get possession and bring the ball toward the opponent box.  Decision network frozen except Move/GetPossession and latent vector.
2026-08-10 02:37:43,861 INFO _from_ckpt: overriding bc_pretrain_epochs=18 → 15
2026-08-10 02:37:43,861 INFO _from_ckpt: overriding demo_value_pretrain_epochs=18 → 15
2026-08-10 02:37:43,861 INFO _from_ckpt: overriding value_pretrain_epochs=45 → 45
2026-08-10 02:37:43,861 INFO _from_ckpt: overriding aux_coeff_start=0.0 → 0.0
2026-08-10 02:37:43,861 INFO _from_ckpt: overriding aux_coeff_end=0.0 → 0.0
2026-08-10 02:37:43,861 INFO _from_ckpt: overriding aux_coeff_anneal_fraction=0.3 → 0.35
/home/vincent/Documents/not_work/repos/FootballCoach/.venv/lib/python3.12/site-packages/torch/_compile.py:54: UserWarning: optimizer contains a parameter group with duplicate parameters; in future, this will cause an error; see github.com/pytorch/pytorch/issues/40967 for more information
  return disable_fn(*args, **kwargs)
2026-08-10 02:37:45,079 INFO Logging to checkpoints/phase1_run71/training_log1.txt
2026-08-10 02:37:45,079 ERROR --latest: no checkpoints found under checkpoints/phase1_run*/
2026-08-10 02:38:46,009 INFO Checkpoint dir: checkpoints/phase1_run72
2026-08-10 02:38:46,062 INFO Starting training: phase=phase1_get_possession, total_steps=6,000,000
2026-08-10 02:38:46,062 INFO Phase description: 1v1 scenario: learn to get possession and bring the ball toward the opponent box.  Decision network frozen except Move/GetPossession and latent vector.
2026-08-10 02:38:46,064 INFO _from_ckpt: overriding bc_pretrain_epochs=18 → 15
2026-08-10 02:38:46,064 INFO _from_ckpt: overriding demo_value_pretrain_epochs=18 → 15
2026-08-10 02:38:46,064 INFO _from_ckpt: overriding value_pretrain_epochs=45 → 45
2026-08-10 02:38:46,064 INFO _from_ckpt: overriding aux_coeff_start=0.0 → 0.0
2026-08-10 02:38:46,064 INFO _from_ckpt: overriding aux_coeff_end=0.0 → 0.0
2026-08-10 02:38:46,064 INFO _from_ckpt: overriding aux_coeff_anneal_fraction=0.3 → 0.35
/home/vincent/Documents/not_work/repos/FootballCoach/.venv/lib/python3.12/site-packages/torch/_compile.py:54: UserWarning: optimizer contains a parameter group with duplicate parameters; in future, this will cause an error; see github.com/pytorch/pytorch/issues/40967 for more information
  return disable_fn(*args, **kwargs)
2026-08-10 02:38:47,186 INFO Logging to checkpoints/phase1_run72/training_log1.txt
2026-08-10 02:38:47,186 ERROR --pretrain-from-checkpoint: file not found: checkpoints/phase1_run71/checkpoint_pretrained.pt
2026-08-10 02:39:27,355 INFO Checkpoint dir: checkpoints/phase1_run73
2026-08-10 02:39:27,412 INFO Starting training: phase=phase1_get_possession, total_steps=6,000,000
2026-08-10 02:39:27,412 INFO Phase description: 1v1 scenario: learn to get possession and bring the ball toward the opponent box.  Decision network frozen except Move/GetPossession and latent vector.
2026-08-10 02:39:27,414 INFO _from_ckpt: overriding bc_pretrain_epochs=18 → 15
2026-08-10 02:39:27,414 INFO _from_ckpt: overriding demo_value_pretrain_epochs=18 → 15
2026-08-10 02:39:27,414 INFO _from_ckpt: overriding value_pretrain_epochs=45 → 45
2026-08-10 02:39:27,414 INFO _from_ckpt: overriding aux_coeff_start=0.0 → 0.0
2026-08-10 02:39:27,414 INFO _from_ckpt: overriding aux_coeff_end=0.0 → 0.0
2026-08-10 02:39:27,414 INFO _from_ckpt: overriding aux_coeff_anneal_fraction=0.3 → 0.35
/home/vincent/Documents/not_work/repos/FootballCoach/.venv/lib/python3.12/site-packages/torch/_compile.py:54: UserWarning: optimizer contains a parameter group with duplicate parameters; in future, this will cause an error; see github.com/pytorch/pytorch/issues/40967 for more information
  return disable_fn(*args, **kwargs)
2026-08-10 02:39:28,537 INFO Logging to checkpoints/phase1_run73/training_log1.txt
2026-08-10 02:39:28,714 INFO Loaded checkpoint: checkpoints/phase1_run70/checkpoint_pretrained.pt (step 0)
2026-08-10 02:39:28,714 INFO Loaded checkpoint for re-pretraining: checkpoints/phase1_run70/checkpoint_pretrained.pt — will still run BC/value pre-training
2026-08-10 02:39:28,722 INFO Loading 1000 demonstration file(s) from demonstrations/phase1_long
2026-08-10 02:39:33,716 INFO Dataset: 448,439 steps loaded
2026-08-10 02:39:33,717 INFO Offline BC dataset: 448,439 steps from demonstrations/phase1_long/
2026-08-10 02:39:33,717 INFO Reward diagnostic: running 40 rules-vs-rules episodes...
2026-08-10 02:39:34,595 INFO Reward diagnostic (40 ep, rules vs rules): mean_ep_rew=2.57  per_episode: get_possession=+1.15  lose_possession=-0.36  box_possession=+3.00  opponent_box=-1.00  step_penalty=-0.08  stamina_penalty=-0.14
2026-08-10 02:39:34,610 INFO BC pos_weight (auto-computed from dataset): kick=1.35  tackle_attempt=1.35
2026-08-10 02:39:34,610 INFO Combined BC + value pre-training: 15 epoch(s), batch_size=2048, dataset=448,439 steps, rollout_steps=120000
2026-08-10 02:39:34,720 INFO Phase 0 — decision-net warm-up (BC + self.value_net MSE; single value head convention): 15 epoch(s), gamma=0.995, returns mean=3.59  std=1.19  lr=0.004  phase0_value_coef=1.0  split: 194,484 train / 34,713 val rows
/home/vincent/Documents/not_work/repos/FootballCoach/src/footballcoach/ai/ppo/bc.py:710: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /__w/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:822.)
  "decision":       float(dec_loss[valid].mean()),
2026-08-10 02:40:07,600 INFO   Phase 0 epoch 1/15: loss=1.4484  dec_bc=0.4778  bc_adj=0.0370(floor=0.4407)  val_mse=0.9707(x1.0)=0.9707
2026-08-10 02:40:09,709 INFO     val  p0_val_loss=1.1209  bc_adj=0.0145  val_mse=0.6656  best=1.1209  (improved)
2026-08-10 02:40:42,140 INFO   Phase 0 epoch 2/15: loss=1.1842  dec_bc=0.4563  bc_adj=0.0156(floor=0.4407)  val_mse=0.7279(x1.0)=0.7279
2026-08-10 02:40:44,242 INFO     val  p0_val_loss=1.1148  bc_adj=0.0149  val_mse=0.6592  best=1.1148  (improved)
2026-08-10 02:41:16,520 INFO   Phase 0 epoch 3/15: loss=1.1283  dec_bc=0.4536  bc_adj=0.0129(floor=0.4407)  val_mse=0.6748(x1.0)=0.6748
2026-08-10 02:41:18,627 INFO     val  p0_val_loss=1.0394  bc_adj=0.0109  val_mse=0.5879  best=1.0394  (improved)
2026-08-10 02:41:50,746 INFO   Phase 0 epoch 4/15: loss=1.1208  dec_bc=0.4524  bc_adj=0.0117(floor=0.4407)  val_mse=0.6684(x1.0)=0.6684
2026-08-10 02:41:52,842 INFO     val  p0_val_loss=1.0463  bc_adj=0.0104  val_mse=0.5952  best=1.0394  (patience 1/5)
2026-08-10 02:42:25,001 INFO   Phase 0 epoch 5/15: loss=1.0934  dec_bc=0.4515  bc_adj=0.0108(floor=0.4407)  val_mse=0.6419(x1.0)=0.6419
2026-08-10 02:42:27,113 INFO     val  p0_val_loss=0.9792  bc_adj=0.0086  val_mse=0.5299  best=0.9792  (improved)
2026-08-10 02:42:58,876 INFO   Phase 0 epoch 6/15: loss=1.0615  dec_bc=0.4509  bc_adj=0.0102(floor=0.4407)  val_mse=0.6106(x1.0)=0.6106
2026-08-10 02:43:01,002 INFO     val  p0_val_loss=1.1164  bc_adj=0.0133  val_mse=0.6624  best=0.9792  (patience 1/5)
2026-08-10 02:43:32,911 INFO   Phase 0 epoch 7/15: loss=1.0877  dec_bc=0.4514  bc_adj=0.0107(floor=0.4407)  val_mse=0.6362(x1.0)=0.6362
2026-08-10 02:43:34,999 INFO     val  p0_val_loss=1.0468  bc_adj=0.0118  val_mse=0.5943  best=0.9792  (patience 2/5)
2026-08-10 02:44:06,812 INFO   Phase 0 epoch 8/15: loss=1.0444  dec_bc=0.4514  bc_adj=0.0107(floor=0.4407)  val_mse=0.5930(x1.0)=0.5930
2026-08-10 02:44:08,914 INFO     val  p0_val_loss=0.9774  bc_adj=0.0093  val_mse=0.5274  best=0.9774  (improved)
2026-08-10 02:44:41,771 INFO   Phase 0 epoch 9/15: loss=1.0846  dec_bc=0.4511  bc_adj=0.0104(floor=0.4407)  val_mse=0.6335(x1.0)=0.6335
2026-08-10 02:44:44,021 INFO     val  p0_val_loss=1.0907  bc_adj=0.0109  val_mse=0.6391  best=0.9774  (patience 1/5)
2026-08-10 02:45:17,917 INFO   Phase 0 epoch 10/15: loss=1.0511  dec_bc=0.4508  bc_adj=0.0101(floor=0.4407)  val_mse=0.6003(x1.0)=0.6003
2026-08-10 02:45:20,063 INFO     val  p0_val_loss=1.0001  bc_adj=0.0129  val_mse=0.5464  best=0.9774  (patience 2/5)
2026-08-10 02:45:53,607 INFO   Phase 0 epoch 11/15: loss=1.0518  dec_bc=0.4511  bc_adj=0.0104(floor=0.4407)  val_mse=0.6007(x1.0)=0.6007
2026-08-10 02:45:55,774 INFO     val  p0_val_loss=1.0547  bc_adj=0.0121  val_mse=0.6018  best=0.9774  (patience 3/5)
2026-08-10 02:46:28,472 INFO   Phase 0 epoch 12/15: loss=1.0475  dec_bc=0.4530  bc_adj=0.0123(floor=0.4407)  val_mse=0.5946(x1.0)=0.5946
2026-08-10 02:46:30,561 INFO     val  p0_val_loss=0.9767  bc_adj=0.0096  val_mse=0.5264  best=0.9767  (improved)
2026-08-10 02:47:04,306 INFO   Phase 0 epoch 13/15: loss=1.0165  dec_bc=0.4504  bc_adj=0.0097(floor=0.4407)  val_mse=0.5661(x1.0)=0.5661
2026-08-10 02:47:06,525 INFO     val  p0_val_loss=0.9658  bc_adj=0.0090  val_mse=0.5160  best=0.9658  (improved)
2026-08-10 02:47:39,986 INFO   Phase 0 epoch 14/15: loss=0.9950  dec_bc=0.4499  bc_adj=0.0092(floor=0.4407)  val_mse=0.5451(x1.0)=0.5451
2026-08-10 02:47:42,140 INFO     val  p0_val_loss=0.9550  bc_adj=0.0092  val_mse=0.5052  best=0.9550  (improved)
2026-08-10 02:48:16,239 INFO   Phase 0 epoch 15/15: loss=1.0366  dec_bc=0.4509  bc_adj=0.0102(floor=0.4407)  val_mse=0.5857(x1.0)=0.5857
2026-08-10 02:48:18,539 INFO     val  p0_val_loss=1.0027  bc_adj=0.0088  val_mse=0.5531  best=0.9550  (patience 1/5)
2026-08-10 02:48:18,539 INFO Phase 0 done (decision-net BC + critic value_head warm-up, 15 epoch(s))
2026-08-10 02:48:18,544 INFO   BC pretrain split: 194,484 train rows  |  34,713 val rows
2026-08-10 02:48:18,680 INFO   Downsample trivial rows (epoch 1): 90,161/229,197 (39.3%) rows classified trivial, excluding ~59,506 this epoch (frac=0.66)
2026-08-10 02:49:11,778 INFO   BC epoch 1/15  (53.2s)
    loss       bc=1.1524  bc_adj=0.4831(floor=0.6693)
    heads      dir_cos=0.975  kick_dir_cos=0.990
               move_prob=0.911  sprint_prob=0.360  kick_prob=0.252  tackle_prob=0.012
    pr/rec     kick:   p=0.874  r=0.915  f1=0.894  (tp=61126 fp=8812 fn=5666)
               tackle: p=0.933  r=0.082  f1=0.151  (tp=14 fp=1 fn=156)
    breakdown  decision=0.224  exec_bce=0.606  sprint=0.230  move=0.122  tackle_attempt=0.061  direction=0.071
               region=0.019  kick=0.19311  kick_direction=0.02874  kick_power=0.00211  kick_spin=0.00003
2026-08-10 02:49:13,844 INFO     val        bc_val_loss=1.0013  best=1.0013  (improved)
2026-08-10 02:49:13,854 INFO   Downsample trivial rows (epoch 2): 90,161/229,197 (39.3%) rows classified trivial, excluding ~59,506 this epoch (frac=0.66)
