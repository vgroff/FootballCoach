usage: train.py [-h] [--phase {1,2,3,4}] [--total-steps TOTAL_STEPS]
                [--checkpoint CHECKPOINT] [--checkpoint-dir CHECKPOINT_DIR]
                [--device DEVICE] [--seed SEED]
                [--bc-pretrain-steps BC_PRETRAIN_STEPS] [--no-bc-aux]
                [--experiment-separate-value-net] [--separate-value-net]
                [--bc-dataset BC_DATASET]
                [--bc-pretrain-epochs BC_PRETRAIN_EPOCHS]
                [--bc-pretrain-batch-size BC_PRETRAIN_BATCH_SIZE] [--verbose]
                [--from-pretrained PATH] [--pretrain-from-checkpoint PATH]
                [--pre-ppo-eval-trials PRE_PPO_EVAL_TRIALS] [--no-head-freeze]
                [--latest] [--latest-pretrain] [--reset-dir-log-std]
train.py: error: argument --bc-dataset: expected one argument
2026-08-08 18:00:25,669 INFO Checkpoint dir: checkpoints/phase1_run44
2026-08-08 18:00:25,728 INFO Starting training: phase=phase1_get_possession, total_steps=600,000
2026-08-08 18:00:25,728 INFO Phase description: 1v1 scenario: learn to get possession and bring the ball toward the opponent box.  Decision network frozen except Move/GetPossession and latent vector.
2026-08-08 18:00:25,731 INFO _from_ckpt: overriding bc_pretrain_epochs=8 → 1
2026-08-08 18:00:25,731 INFO _from_ckpt: overriding demo_value_pretrain_epochs=6 → 0
2026-08-08 18:00:25,731 INFO _from_ckpt: overriding value_pretrain_epochs=8 → 15
2026-08-08 18:00:25,731 INFO _from_ckpt: overriding aux_coeff_start=0.0 → 0.0
2026-08-08 18:00:25,731 INFO _from_ckpt: overriding aux_coeff_end=0.0 → 0.0
2026-08-08 18:00:25,731 INFO _from_ckpt: overriding aux_coeff_anneal_fraction=0.3 → 0.35
/home/vincent/Documents/not_work/repos/FootballCoach/.venv/lib/python3.12/site-packages/torch/_compile.py:54: UserWarning: optimizer contains a parameter group with duplicate parameters; in future, this will cause an error; see github.com/pytorch/pytorch/issues/40967 for more information
  return disable_fn(*args, **kwargs)
2026-08-08 18:00:27,020 INFO Logging to checkpoints/phase1_run44/training_log1.txt
2026-08-08 18:00:27,021 INFO --latest(-pretrain): resolved to checkpoints/phase1_run43/latest.pt
2026-08-08 18:00:27,233 WARNING Optimizer param group count changed (2 -> 3); skipping optimizer state restore for checkpoints/phase1_run43/latest.pt (network weights still loaded normally).
2026-08-08 18:00:27,233 INFO Loaded checkpoint: checkpoints/phase1_run43/latest.pt (step 264000)
2026-08-08 18:00:27,234 INFO Loaded checkpoint for re-pretraining: checkpoints/phase1_run43/latest.pt — will still run BC/value pre-training
2026-08-08 18:00:27,234 INFO --reset-dir-log-std: move/kick dir_log_std reset to -1.65
2026-08-08 18:00:27,250 INFO Loading 1250 demonstration file(s) from demonstrations/phase1
2026-08-08 18:00:32,776 INFO Dataset: 471,301 steps loaded
2026-08-08 18:00:32,778 INFO Offline BC dataset: 471,301 steps from demonstrations/phase1/
2026-08-08 18:00:32,778 INFO Reward diagnostic: running 40 rules-vs-rules episodes...
2026-08-08 18:00:33,561 INFO Reward diagnostic (40 ep, rules vs rules): mean_ep_rew=2.30  per_episode: get_possession=+0.88  lose_possession=-0.27  box_possession=+1.31  speed_bonus=+1.22  opponent_box=-0.82  stamina_penalty=-0.01
2026-08-08 18:00:33,577 INFO BC pos_weight (auto-computed from dataset): kick=1.50  tackle_attempt=1.50
2026-08-08 18:00:33,577 INFO Combined BC + value pre-training: 1 epoch(s), batch_size=800, dataset=471,301 steps, rollout_steps=21000
2026-08-08 18:00:33,585 INFO   BC pretrain split: 355,096 train rows  |  61,570 val rows
2026-08-08 18:00:33,795 INFO   Downsample trivial rows (epoch 1): 125,791/416,666 (30.2%) rows classified trivial, excluding ~100,633 this epoch (frac=0.80)
/home/vincent/Documents/not_work/repos/FootballCoach/src/footballcoach/ai/ppo/bc.py:688: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /__w/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:822.)
  "decision":       float(dec_loss[valid].mean()),
2026-08-08 18:02:43,394 INFO   BC epoch 1/1  (129.8s)
    loss       bc=2.3191  bc_adj=0.2527(floor=2.0664)
    heads      dir_cos=0.974  kick_dir_cos=0.990
               move_prob=0.888  sprint_prob=0.814  kick_prob=0.067  tackle_prob=0.259
    pr/rec     kick:   p=0.968  r=0.963  f1=0.966  (tp=29719 fp=986 fn=1129)
               tackle: p=0.967  r=0.987  f1=0.977  (tp=249099 fp=8434 fn=3317)
    breakdown  decision=0.687  exec_bce=0.859  sprint=0.249  move=0.220  tackle_attempt=0.213  direction=0.075
               region=0.011  kick=0.17619  kick_direction=0.02969  kick_power=0.00203  kick_spin=0.00001
2026-08-08 18:02:45,641 INFO     val        bc_val_loss=2.2581  best=2.2581  (improved)
2026-08-08 18:02:45,648 INFO BC pre-training done (1 epoch(s), final bc_loss=2.3191)
2026-08-08 18:02:45,648 INFO Value pre-training: 21000 steps, 15 epochs, lr=3e-05
2026-08-08 18:02:45,649 INFO   [value pretrain rollout] parallel collection: 6 worker(s), ~3500 steps/worker
2026-08-08 18:02:47,468 INFO Frozen decision_net.shoot_logit
2026-08-08 18:02:47,469 INFO Frozen decision_net.pass_logit
2026-08-08 18:02:47,469 INFO Frozen decision_net.tackle_logit
2026-08-08 18:02:47,469 INFO Frozen decision_net.get_possession_raw
2026-08-08 18:02:47,469 INFO Frozen decision_net.mark_logit
2026-08-08 18:02:47,469 INFO Frozen decision_net.hold_position_logit
2026-08-08 18:02:47,469 INFO Frozen decision_net.pass_target_logits
2026-08-08 18:02:47,469 INFO Frozen decision_net.tackle_target_logits
2026-08-08 18:02:47,469 INFO Frozen decision_net.mark_target_logits
2026-08-08 18:02:47,469 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-08 18:02:47,489 INFO Frozen decision_net.shoot_logit
2026-08-08 18:02:47,489 INFO Frozen decision_net.pass_logit
2026-08-08 18:02:47,489 INFO Frozen decision_net.tackle_logit
2026-08-08 18:02:47,489 INFO Frozen decision_net.get_possession_raw
2026-08-08 18:02:47,489 INFO Frozen decision_net.mark_logit
2026-08-08 18:02:47,489 INFO Frozen decision_net.hold_position_logit
2026-08-08 18:02:47,489 INFO Frozen decision_net.pass_target_logits
2026-08-08 18:02:47,489 INFO Frozen decision_net.tackle_target_logits
2026-08-08 18:02:47,489 INFO Frozen decision_net.mark_target_logits
2026-08-08 18:02:47,489 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-08 18:02:47,510 INFO Frozen decision_net.shoot_logit
2026-08-08 18:02:47,510 INFO Frozen decision_net.pass_logit
2026-08-08 18:02:47,510 INFO Frozen decision_net.tackle_logit
2026-08-08 18:02:47,510 INFO Frozen decision_net.get_possession_raw
2026-08-08 18:02:47,510 INFO Frozen decision_net.mark_logit
2026-08-08 18:02:47,510 INFO Frozen decision_net.hold_position_logit
2026-08-08 18:02:47,510 INFO Frozen decision_net.pass_target_logits
2026-08-08 18:02:47,510 INFO Frozen decision_net.tackle_target_logits
2026-08-08 18:02:47,510 INFO Frozen decision_net.mark_target_logits
2026-08-08 18:02:47,510 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-08 18:02:47,511 INFO Frozen decision_net.shoot_logit
2026-08-08 18:02:47,511 INFO Frozen decision_net.pass_logit
2026-08-08 18:02:47,511 INFO Frozen decision_net.tackle_logit
2026-08-08 18:02:47,511 INFO Frozen decision_net.get_possession_raw
2026-08-08 18:02:47,511 INFO Frozen decision_net.mark_logit
2026-08-08 18:02:47,511 INFO Frozen decision_net.hold_position_logit
2026-08-08 18:02:47,511 INFO Frozen decision_net.pass_target_logits
2026-08-08 18:02:47,511 INFO Frozen decision_net.tackle_target_logits
2026-08-08 18:02:47,511 INFO Frozen decision_net.mark_target_logits
2026-08-08 18:02:47,511 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-08 18:02:47,572 INFO Frozen decision_net.shoot_logit
2026-08-08 18:02:47,572 INFO Frozen decision_net.pass_logit
2026-08-08 18:02:47,572 INFO Frozen decision_net.tackle_logit
2026-08-08 18:02:47,572 INFO Frozen decision_net.get_possession_raw
2026-08-08 18:02:47,573 INFO Frozen decision_net.mark_logit
2026-08-08 18:02:47,573 INFO Frozen decision_net.hold_position_logit
2026-08-08 18:02:47,573 INFO Frozen decision_net.pass_target_logits
2026-08-08 18:02:47,573 INFO Frozen decision_net.tackle_target_logits
2026-08-08 18:02:47,573 INFO Frozen decision_net.mark_target_logits
2026-08-08 18:02:47,573 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-08 18:02:47,576 INFO Frozen decision_net.shoot_logit
2026-08-08 18:02:47,576 INFO Frozen decision_net.pass_logit
2026-08-08 18:02:47,576 INFO Frozen decision_net.tackle_logit
2026-08-08 18:02:47,576 INFO Frozen decision_net.get_possession_raw
2026-08-08 18:02:47,576 INFO Frozen decision_net.mark_logit
2026-08-08 18:02:47,576 INFO Frozen decision_net.hold_position_logit
2026-08-08 18:02:47,576 INFO Frozen decision_net.pass_target_logits
2026-08-08 18:02:47,576 INFO Frozen decision_net.tackle_target_logits
2026-08-08 18:02:47,576 INFO Frozen decision_net.mark_target_logits
2026-08-08 18:02:47,576 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-08 18:03:18,745 INFO   [value pretrain rollout] dropped 358 trailing (incomplete-episode) step(s) across workers before MC-return fit
2026-08-08 18:03:18,752 INFO   [value pretrain rollout] mean_return=2.58 (346 episode(s))  vs[win/loss/tout/miss]  vs_rules(0): n/a  vs_immobile(346): 52.0%/0.0%/5.5%/20.8%/22%  vs_neural(0): n/a
2026-08-08 18:03:18,752 INFO   [value pretrain rollout] ep_len 17.8±13.3s  (n=346, min=0.8s, max=50.0s)
2026-08-08 18:03:18,753 INFO   [value pretrain rollout] rew/ep (mean/std/min/max per episode, 346 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.780    0.441    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.010    0.096    -0.900    +0.000
  ball_out          -0.419    1.386    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.301    1.249    +0.000    +2.500
  speed_bonus       +1.012    1.337    +0.000    +4.058
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.082    0.342    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.004    0.006    -0.026    +0.000
2026-08-08 18:03:18,877 INFO   Value pretrain split: 294 train eps (17528 steps)  |  52 val eps (3114 steps)
2026-08-08 18:03:23,429 INFO   Value epoch 1/15: train=0.8460 rmse=2.49  val=0.9505 val_rmse=2.64 (std=2.7)
    V(train)=+1.743  R(train)=+1.769  |  V(val)=+1.649  R(val)=+1.815
2026-08-08 18:03:27,924 INFO   Value epoch 2/15: train=0.8047 rmse=2.43  val=0.9588 val_rmse=2.65 (std=2.7)
    V(train)=+1.773  R(train)=+1.769  |  V(val)=+1.675  R(val)=+1.815
2026-08-08 18:03:32,177 INFO   Value epoch 3/15: train=0.7797 rmse=2.39  val=0.9597 val_rmse=2.65 (std=2.7)
    V(train)=+1.772  R(train)=+1.768  |  V(val)=+1.711  R(val)=+1.815
2026-08-08 18:03:36,417 INFO   Value epoch 4/15: train=0.7660 rmse=2.37  val=0.9634 val_rmse=2.66 (std=2.7)
    V(train)=+1.782  R(train)=+1.768  |  V(val)=+1.773  R(val)=+1.815
2026-08-08 18:03:40,913 INFO   Value epoch 5/15: train=0.7566 rmse=2.36  val=0.9649 val_rmse=2.66 (std=2.7)
    V(train)=+1.780  R(train)=+1.768  |  V(val)=+1.637  R(val)=+1.815
2026-08-08 18:03:40,913 INFO   [value pretrain] early stop at epoch 5 (val stagnant for 4 epochs, best=0.9505)
2026-08-08 18:03:40,914 INFO   [value pretrain] restored best-val weights (val_loss=0.9505)
2026-08-08 18:03:40,915 INFO Value pre-training done (5 epoch(s), final train_loss=0.7566)
2026-08-08 18:03:53,150 INFO BC check after value warm-up: bc_loss=2.2614 (before=2.3191, delta=-0.0577)  OK
2026-08-08 18:03:53,150 INFO Combined pre-training complete.
2026-08-08 18:04:26,186 INFO Pre-PPO eval (rules opp): win=23.4%  mean_rew=-0.155  V=1.643  R=-0.167  gap=+1.810  outcomes={'other': 22, 'opponent_box_possession': 64, 'miss': 12, 'box_possession': 30}
2026-08-08 18:04:26,186 INFO   rew breakdown (rules, per ep): opponent_box=-1.50  box_possession=+0.59  get_possession=+0.57  speed_bonus=+0.51  lose_possession=-0.23  ball_out=-0.08  stamina_penalty=-0.01
2026-08-08 18:05:09,092 INFO Pre-PPO eval (immobile opp): win=40.6%  mean_rew=2.016  V=1.715  R=0.949  gap=+0.766  outcomes={'other': 34, 'timeout': 8, 'box_possession': 52, 'miss': 34}
2026-08-08 18:05:09,092 INFO   rew breakdown (immobile, per ep): box_possession=+1.02  speed_bonus=+0.83  get_possession=+0.67  ball_out=-0.39  timeout=-0.09  lose_possession=-0.01
2026-08-08 18:06:09,902 INFO Pre-PPO eval (self-play):   win=43.0%  mean_rew=1.872  V=1.821  R=0.991  gap=+0.829  outcomes={'other': 26, 'box_possession': 55, 'opponent_box_possession': 29, 'miss': 18}
2026-08-08 18:06:09,903 INFO   rew breakdown (self-play, per ep): opponent_box=-1.97  box_possession=+1.64  speed_bonus=+1.53  get_possession=+1.26  lose_possession=-0.41  ball_out=-0.27  stamina_penalty=-0.02
2026-08-08 18:06:09,903 INFO   [seeded eval] running 12x8 episodes across 7 worker process(es)...
2026-08-08 18:06:12,930 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 18:06:12,931 INFO Baseline (rules vs rules, 12 trials): trainee_win=83.3%  outcomes={'box_possession': 80, 'other': 8, 'opponent_box_possession': 8}
2026-08-08 18:06:12,931 INFO Frozen decision_net.shoot_logit
2026-08-08 18:06:12,931 INFO Frozen decision_net.pass_logit
2026-08-08 18:06:12,931 INFO Frozen decision_net.tackle_logit
2026-08-08 18:06:12,931 INFO Frozen decision_net.get_possession_raw
2026-08-08 18:06:12,931 INFO Frozen decision_net.mark_logit
2026-08-08 18:06:12,931 INFO Frozen decision_net.hold_position_logit
2026-08-08 18:06:12,931 INFO Frozen decision_net.pass_target_logits
2026-08-08 18:06:12,931 INFO Frozen decision_net.tackle_target_logits
2026-08-08 18:06:12,931 INFO Frozen decision_net.mark_target_logits
2026-08-08 18:06:12,932 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-08 18:06:12,932 INFO PPO parallel training started: 6 worker(s), ~6000 steps/worker/rollout, steps_so_far=0  target=600,000
2026-08-08 18:06:14,650 INFO Frozen decision_net.shoot_logit
2026-08-08 18:06:14,650 INFO Frozen decision_net.pass_logit
2026-08-08 18:06:14,650 INFO Frozen decision_net.tackle_logit
2026-08-08 18:06:14,650 INFO Frozen decision_net.get_possession_raw
2026-08-08 18:06:14,651 INFO Frozen decision_net.mark_logit
2026-08-08 18:06:14,651 INFO Frozen decision_net.hold_position_logit
2026-08-08 18:06:14,651 INFO Frozen decision_net.pass_target_logits
2026-08-08 18:06:14,651 INFO Frozen decision_net.tackle_target_logits
2026-08-08 18:06:14,651 INFO Frozen decision_net.mark_target_logits
2026-08-08 18:06:14,651 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-08 18:06:14,656 INFO Frozen decision_net.shoot_logit
2026-08-08 18:06:14,656 INFO Frozen decision_net.pass_logit
2026-08-08 18:06:14,656 INFO Frozen decision_net.tackle_logit
2026-08-08 18:06:14,656 INFO Frozen decision_net.get_possession_raw
2026-08-08 18:06:14,656 INFO Frozen decision_net.mark_logit
2026-08-08 18:06:14,656 INFO Frozen decision_net.hold_position_logit
2026-08-08 18:06:14,656 INFO Frozen decision_net.pass_target_logits
2026-08-08 18:06:14,656 INFO Frozen decision_net.tackle_target_logits
2026-08-08 18:06:14,656 INFO Frozen decision_net.mark_target_logits
2026-08-08 18:06:14,656 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-08 18:06:14,663 INFO Frozen decision_net.shoot_logit
2026-08-08 18:06:14,663 INFO Frozen decision_net.pass_logit
2026-08-08 18:06:14,663 INFO Frozen decision_net.tackle_logit
2026-08-08 18:06:14,663 INFO Frozen decision_net.get_possession_raw
2026-08-08 18:06:14,663 INFO Frozen decision_net.mark_logit
2026-08-08 18:06:14,663 INFO Frozen decision_net.hold_position_logit
2026-08-08 18:06:14,664 INFO Frozen decision_net.pass_target_logits
2026-08-08 18:06:14,664 INFO Frozen decision_net.tackle_target_logits
2026-08-08 18:06:14,664 INFO Frozen decision_net.mark_target_logits
2026-08-08 18:06:14,664 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-08 18:06:14,704 INFO Frozen decision_net.shoot_logit
2026-08-08 18:06:14,704 INFO Frozen decision_net.pass_logit
2026-08-08 18:06:14,704 INFO Frozen decision_net.tackle_logit
2026-08-08 18:06:14,704 INFO Frozen decision_net.get_possession_raw
2026-08-08 18:06:14,704 INFO Frozen decision_net.mark_logit
2026-08-08 18:06:14,704 INFO Frozen decision_net.hold_position_logit
2026-08-08 18:06:14,704 INFO Frozen decision_net.pass_target_logits
2026-08-08 18:06:14,704 INFO Frozen decision_net.tackle_target_logits
2026-08-08 18:06:14,704 INFO Frozen decision_net.mark_target_logits
2026-08-08 18:06:14,704 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-08 18:06:14,704 INFO Frozen decision_net.shoot_logit
2026-08-08 18:06:14,704 INFO Frozen decision_net.pass_logit
2026-08-08 18:06:14,704 INFO Frozen decision_net.tackle_logit
2026-08-08 18:06:14,704 INFO Frozen decision_net.get_possession_raw
2026-08-08 18:06:14,704 INFO Frozen decision_net.mark_logit
2026-08-08 18:06:14,704 INFO Frozen decision_net.hold_position_logit
2026-08-08 18:06:14,704 INFO Frozen decision_net.pass_target_logits
2026-08-08 18:06:14,704 INFO Frozen decision_net.tackle_target_logits
2026-08-08 18:06:14,705 INFO Frozen decision_net.mark_target_logits
2026-08-08 18:06:14,705 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-08 18:06:14,727 INFO Frozen decision_net.shoot_logit
2026-08-08 18:06:14,727 INFO Frozen decision_net.pass_logit
2026-08-08 18:06:14,727 INFO Frozen decision_net.tackle_logit
2026-08-08 18:06:14,727 INFO Frozen decision_net.get_possession_raw
2026-08-08 18:06:14,728 INFO Frozen decision_net.mark_logit
2026-08-08 18:06:14,728 INFO Frozen decision_net.hold_position_logit
2026-08-08 18:06:14,728 INFO Frozen decision_net.pass_target_logits
2026-08-08 18:06:14,728 INFO Frozen decision_net.tackle_target_logits
2026-08-08 18:06:14,728 INFO Frozen decision_net.mark_target_logits
2026-08-08 18:06:14,728 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-08 18:11:06,718 INFO   [KL mean=0.0874 median=0.0872 > 0.05] ratio percentiles:  p5=0.559  p25=0.912  p50=0.977  p75=1.014  p95=1.228  max=25.290
  move_dir_log_std=[-1.6491327285766602]  kick_dir_log_std=[-1.6491737365722656]
2026-08-08 18:11:06,733 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.082  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.190  kick=-0.204  t_att=-0.154
    move_dir=0.721 (min=-9.281 max=1.460)  kick_dir=0.075 (min=-0.319 max=2.170)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.22
  [worst sample] idx=89  ratio=32.983  adv=+3.056  old_lp=-3.614  new_lp=-0.118
    stored move_dir=-1.2°  new_mean=-4.5°  angular_diff=3.4°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx=  89  ratio=  32.983  adv=+3.056  lp: old=-3.614  new=-0.118
      rew=+0.0000  ret=+5.1142  val=+2.0579  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9589  sprint_p_new=0.0164  kick_p_new=0.0478  tackle_attempt_p_new=0.0405
    idx= 166  ratio=  30.374  adv=-1.353  lp: old=-3.538  new=-0.124
      rew=+0.0000  ret=+0.9722  val=+2.3249  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9586  sprint_p_new=0.0562  kick_p_new=0.0537  tackle_attempt_p_new=0.0416
  [best sample (highest new_lp)] idx=205  new_lp=-0.046  adv=+0.074  stored move_dir=-156.8°  new_mean=-163.2°
    per-head contributions: move_dir:0.065  move:-0.026  tackle_attempt:-0.039  kick:-0.039
2026-08-08 18:11:06,734 INFO   [advantage] mean=-0.000  std=1.000  min=-5.368  max=4.014
2026-08-08 18:11:06,735 INFO   [ratio] mean=0.9606  std=0.2814  min=0.0004  max=25.2901  clipped=24.1%
2026-08-08 18:11:06,736 INFO   [exec head grad norm] move_direction=0.048  exec_move=0.073  sprint=0.047  kick=0.084  kick_direction=0.010  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.061
2026-08-08 18:11:06,736 INFO   [exec continuous log_std] move_direction: start=-1.6500 end=-1.6491   kick_direction: start=-1.6500 end=-1.6492
2026-08-08 18:11:06,736 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0021≈0.12°/step  epoch≈7.1°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0018≈0.10°/step  epoch≈6.2°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-08 18:11:06,736 INFO   [exec discrete Δlogit per opt step] exec_move=0.0034  sprint=0.0032  kick=0.0025  tackle_attempt=0.0030
2026-08-08 18:11:06,737 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0001  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0098  sprint=+0.0115  kick=+0.0110  tackle_attempt=+0.0007  move_dir=+0.0424  kick_dir=+0.0119
2026-08-08 18:11:06,737 INFO   [grad clip] main: 11/60 steps clipped (18%)  pre-clip norm mean=0.336 max=0.609  limit=0.4
              direction: 60/60 steps clipped (100%)  pre-clip norm mean=0.052 max=0.200  limit=0.015
2026-08-08 18:11:06,789 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=36,000  speed=763/s  reward=2.47
  loss     policy=0.0216  value=0.7271(x0.5)=0.3636
           entropy=1.5252  kl=0.0874
  value    V=1.72±0.92  R=1.84±1.69  adv=0.12±1.44
  moves    mv_ls=[-1.6491] (σ≈0.19, ≈11°) g=1.21e-02
           kk_ls=[-1.6492] (σ≈0.19, ≈11°)
  heads    move= 25 get_poss= 75 exec_move= 91 sprint= 33 kick=  5 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0431 kick_prob=0.0534
  vs       vs[win/loss/tout/miss]  vs_immobile(576): 55.0%/0.0%/7.1%/14.1%/24%
  ep_len   18.6±13.8s  (n=576, min=0.8s, max=50.0s)
  reward   get_possession=+470.00  lose_possession=-2.70  ball_out=-130.00  box_possession=+792.50
           speed_bonus=+640.99  timeout=-61.50  stamina_penalty=-2.17
  rew/ep   (mean/std/min/max per episode, 576 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.816    0.401    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.005    0.065    -0.900    +0.000
  ball_out          -0.226    1.038    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.376    1.244    +0.000    +2.500
  speed_bonus       +1.113    1.399    +0.000    +4.256
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.107    0.386    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.004    0.006    -0.037    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     471    +0.013    0.114     +3.411     1.422     +2.116      6.3947      2.269     3.924
  lose_possession       3    -0.000    0.008     +0.736     0.421     -1.966      3.8794      1.966     2.114
  ball_out            26    -0.004    0.134     -4.962     0.192     -5.912     35.4614      5.912     7.311
  box_possession     317    +0.022    0.234     +4.525     1.299     +1.790      4.8885      1.837     3.643
  speed_bonus        294    +0.018    0.225     +4.683     1.214     +1.942      5.2528      1.954     3.661
  timeout             41    -0.002    0.051     -1.500     0.000     -2.629      7.7803      2.629     4.577
  stamina_penalty     254    -0.000    0.001     +4.747     1.331     +1.947      5.5360      2.014     3.668
  gae/td   mean_return=+1.845  std_return=1.690  mean_gae=+0.121  mean_sq_td=2.0866
──────────────────────────────────────────────────────────────────────
2026-08-08 18:11:06,821 INFO Saved checkpoint: checkpoints/phase1_run44/checkpoint1.pt
2026-08-08 18:11:06,822 INFO Logging to checkpoints/phase1_run44/training_log2.txt
2026-08-08 18:11:06,823 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 18:11:18,728 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 18:11:18,730 INFO   [eval vs immobile] step=36,000  seeds=16x8  win=42%  mean_rew=2.343±3.277  V=1.954  gap=-0.390  outcomes={'other': 40, 'box_possession': 54, 'timeout': 3, 'miss': 31}
2026-08-08 18:11:18,731 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 18:11:30,116 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 18:11:30,117 INFO   [eval vs rules] step=36,000  seeds=16x8  win=24%  mean_rew=-0.111±3.481  V=1.744  gap=+1.855  outcomes={'other': 24, 'box_possession': 31, 'opponent_box_possession': 55, 'miss': 18}
2026-08-08 18:16:27,785 INFO   [KL mean=0.0592 median=0.0592 > 0.05] ratio percentiles:  p5=0.647  p25=0.933  p50=0.980  p75=1.013  p95=1.222  max=31.633
  move_dir_log_std=[-1.6483176946640015]  kick_dir_log_std=[-1.6483196020126343]
2026-08-08 18:16:27,802 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.137  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.194  kick=-0.152  t_att=-0.199
    move_dir=0.840 (min=-5.094 max=1.459)  kick_dir=0.045 (min=-0.354 max=2.136)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.28
  [worst sample] idx=198  ratio=26.954  adv=+0.416  old_lp=-3.395  new_lp=-0.101
    stored move_dir=176.8°  new_mean=178.2°  angular_diff=1.5°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 198  ratio=  26.954  adv=+0.416  lp: old=-3.395  new=-0.101
      rew=+0.0000  ret=+3.7852  val=+3.3692  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9553  sprint_p_new=0.9571  kick_p_new=0.0295  tackle_attempt_p_new=0.0437
    idx=  57  ratio=  25.172  adv=-1.124  lp: old=-3.368  new=-0.143
      rew=+0.0000  ret=+0.1610  val=+1.2845  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9506  sprint_p_new=0.0156  kick_p_new=0.0579  tackle_attempt_p_new=0.0485
  [best sample (highest new_lp)] idx=60  new_lp=-0.065  adv=-0.912  stored move_dir=7.0°  new_mean=7.6°
    per-head contributions: move_dir:0.073  move:-0.034  tackle_attempt:-0.046  kick:-0.053
2026-08-08 18:16:27,802 INFO   [advantage] mean=-0.000  std=1.000  min=-5.201  max=3.416
2026-08-08 18:16:27,803 INFO   [ratio] mean=0.9745  std=0.2523  min=0.0009  max=31.6328  clipped=20.2%
2026-08-08 18:16:27,803 INFO   [exec head grad norm] move_direction=0.030  exec_move=0.088  sprint=0.046  kick=0.087  kick_direction=0.008  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.066
2026-08-08 18:16:27,803 INFO   [exec continuous log_std] move_direction: start=-1.6491 end=-1.6483   kick_direction: start=-1.6492 end=-1.6483
2026-08-08 18:16:27,804 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0008≈0.05°/step  epoch≈2.7°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0009≈0.05°/step  epoch≈3.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-08 18:16:27,804 INFO   [exec discrete Δlogit per opt step] exec_move=0.0032  sprint=0.0023  kick=0.0024  tackle_attempt=0.0029
2026-08-08 18:16:27,804 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0002  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0070  sprint=+0.0106  kick=+0.0066  tackle_attempt=+0.0009  move_dir=+0.0245  kick_dir=+0.0094
2026-08-08 18:16:27,804 INFO   [grad clip] main: 8/60 steps clipped (13%)  pre-clip norm mean=0.316 max=0.490  limit=0.4
              direction: 60/60 steps clipped (100%)  pre-clip norm mean=0.033 max=0.065  limit=0.015
2026-08-08 18:16:27,855 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=72,000  speed=746/s  reward=4.17
  loss     policy=0.0157  value=0.7359(x0.5)=0.3679
           entropy=1.6479  kl=0.0592
  value    V=1.93±0.89  R=2.03±1.81  adv=0.10±1.56
  moves    mv_ls=[-1.6483] (σ≈0.19, ≈11°) g=9.44e-03  d_move=[+0.0008] (Δσ≈0.009°)
           kk_ls=[-1.6483] (σ≈0.19, ≈11°)  d_kick=[+0.0009] (Δσ≈0.009°)
  heads    move= 29 get_poss= 71 exec_move= 91 sprint= 38 kick=  5 tackle=  4 shoot=
           2 hold=  2 tackle_prob=0.0496 kick_prob=0.0529
  vs       vs[win/loss/tout/miss]  vs_immobile(629): 54.5%/0.2%/4.0%/18.0%/23%
  ep_len   17.0±12.5s  (n=629, min=0.5s, max=50.0s)
  reward   get_possession=+515.00  lose_possession=-2.70  ball_out=-245.00  box_possession=+857.50
           speed_bonus=+745.06  opponent_box=-3.00  timeout=-37.50  stamina_penalty=-2.62
  rew/ep   (mean/std/min/max per episode, 629 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.819    0.397    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.004    0.062    -0.900    +0.000
  ball_out          -0.390    1.340    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.363    1.245    +0.000    +2.500
  speed_bonus       +1.185    1.442    +0.000    +4.316
  opponent_box      -0.005    0.120    -3.000    +0.000
  timeout           -0.060    0.293    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.004    0.006    -0.023    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     519    +0.014    0.119     +3.279     1.740     +1.788      6.0481      2.186     4.068
  lose_possession       3    -0.000    0.008     +2.702     1.108     -0.428      1.4696      0.901     1.860
  ball_out            49    -0.007    0.184     -4.959     0.198     -6.116     37.9094      6.116     7.300
  box_possession     343    +0.024    0.243     +4.665     1.289     +1.781      4.9418      1.865     3.656
  speed_bonus        334    +0.021    0.246     +4.723     1.255     +1.841      5.0678      1.902     3.658
  opponent_box         1    -0.000    0.016     -3.003     0.000     -4.896     23.9674      4.896     4.896
  timeout             25    -0.001    0.040     -1.501     0.003     -2.663      7.8719      2.663     4.329
  stamina_penalty     296    -0.000    0.001     +4.799     1.395     +1.913      5.5885      2.038     3.672
  gae/td   mean_return=+2.028  std_return=1.805  mean_gae=+0.101  mean_sq_td=2.4481
──────────────────────────────────────────────────────────────────────
2026-08-08 18:16:27,889 INFO Saved checkpoint: checkpoints/phase1_run44/checkpoint2.pt
2026-08-08 18:16:27,889 INFO Logging to checkpoints/phase1_run44/training_log3.txt
2026-08-08 18:16:27,890 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 18:16:40,938 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 18:16:40,939 INFO   [eval vs immobile] step=72,000  seeds=16x8  win=48%  mean_rew=2.604±3.328  V=2.030  gap=-0.575  outcomes={'other': 35, 'opponent_box_possession': 1, 'box_possession': 62, 'miss': 28, 'timeout': 2}
2026-08-08 18:16:40,940 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 18:16:52,720 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 18:16:52,722 INFO   [eval vs rules] step=72,000  seeds=16x8  win=30%  mean_rew=0.536±3.737  V=1.682  gap=+1.146  outcomes={'box_possession': 39, 'other': 23, 'opponent_box_possession': 53, 'miss': 13}
2026-08-08 18:22:25,676 INFO Checkpoint dir: checkpoints/phase1_run45
2026-08-08 18:22:25,729 INFO Starting training: phase=phase1_get_possession, total_steps=6,000,000
2026-08-08 18:22:25,729 INFO Phase description: 1v1 scenario: learn to get possession and bring the ball toward the opponent box.  Decision network frozen except Move/GetPossession and latent vector.
/home/vincent/Documents/not_work/repos/FootballCoach/.venv/lib/python3.12/site-packages/torch/_compile.py:54: UserWarning: optimizer contains a parameter group with duplicate parameters; in future, this will cause an error; see github.com/pytorch/pytorch/issues/40967 for more information
  return disable_fn(*args, **kwargs)
2026-08-08 18:22:26,795 INFO Logging to checkpoints/phase1_run45/training_log1.txt
2026-08-08 18:22:26,796 INFO --latest: resolved to checkpoints/phase1_run44/latest.pt
2026-08-08 18:22:26,984 INFO Loaded checkpoint: checkpoints/phase1_run44/latest.pt (step 72000)
2026-08-08 18:22:26,985 INFO --reset-dir-log-std: move_dir_log_std=-1.65  kick_dir_log_std=-1.65
2026-08-08 18:22:50,940 INFO Pre-PPO eval (rules opp): win=32.8%  mean_rew=0.579  V=1.720  R=0.314  gap=+1.405  outcomes={'other': 20, 'box_possession': 42, 'opponent_box_possession': 55, 'timeout': 1, 'miss': 10}
2026-08-08 18:22:50,940 INFO   rew breakdown (rules, per ep): opponent_box=-1.29  box_possession=+0.82  speed_bonus=+0.66  get_possession=+0.65  lose_possession=-0.20  ball_out=-0.03  timeout=-0.01  stamina_penalty=-0.01
2026-08-08 18:23:23,503 INFO Pre-PPO eval (immobile opp): win=49.2%  mean_rew=2.780  V=2.066  R=1.472  gap=+0.594  outcomes={'other': 35, 'box_possession': 63, 'miss': 27, 'timeout': 3}
2026-08-08 18:23:23,503 INFO   rew breakdown (immobile, per ep): box_possession=+1.23  speed_bonus=+1.13  get_possession=+0.68  ball_out=-0.22  timeout=-0.04
2026-08-08 18:24:16,739 INFO Pre-PPO eval (self-play):   win=34.4%  mean_rew=0.997  V=1.803  R=0.467  gap=+1.336  outcomes={'other': 28, 'box_possession': 44, 'opponent_box_possession': 33, 'miss': 23}
2026-08-08 18:24:16,739 INFO   rew breakdown (self-play, per ep): opponent_box=-1.80  box_possession=+1.50  get_possession=+1.28  speed_bonus=+1.24  ball_out=-0.47  lose_possession=-0.46  stamina_penalty=-0.02
2026-08-08 18:24:16,739 INFO   [seeded eval] running 12x8 episodes across 7 worker process(es)...
2026-08-08 18:24:19,621 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 18:24:19,622 INFO Baseline (rules vs rules, 12 trials): trainee_win=83.3%  outcomes={'box_possession': 80, 'other': 8, 'opponent_box_possession': 8}
2026-08-08 18:24:19,622 INFO Frozen decision_net.shoot_logit
2026-08-08 18:24:19,622 INFO Frozen decision_net.pass_logit
2026-08-08 18:24:19,622 INFO Frozen decision_net.tackle_logit
2026-08-08 18:24:19,622 INFO Frozen decision_net.get_possession_raw
2026-08-08 18:24:19,622 INFO Frozen decision_net.mark_logit
2026-08-08 18:24:19,622 INFO Frozen decision_net.hold_position_logit
2026-08-08 18:24:19,622 INFO Frozen decision_net.pass_target_logits
2026-08-08 18:24:19,622 INFO Frozen decision_net.tackle_target_logits
2026-08-08 18:24:19,622 INFO Frozen decision_net.mark_target_logits
2026-08-08 18:24:19,622 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-08 18:24:19,623 INFO PPO parallel training started: 6 worker(s), ~6000 steps/worker/rollout, steps_so_far=72,000  target=6,072,000
2026-08-08 18:24:21,316 INFO Frozen decision_net.shoot_logit
2026-08-08 18:24:21,317 INFO Frozen decision_net.pass_logit
2026-08-08 18:24:21,317 INFO Frozen decision_net.tackle_logit
2026-08-08 18:24:21,317 INFO Frozen decision_net.get_possession_raw
2026-08-08 18:24:21,317 INFO Frozen decision_net.mark_logit
2026-08-08 18:24:21,317 INFO Frozen decision_net.hold_position_logit
2026-08-08 18:24:21,317 INFO Frozen decision_net.pass_target_logits
2026-08-08 18:24:21,317 INFO Frozen decision_net.tackle_target_logits
2026-08-08 18:24:21,317 INFO Frozen decision_net.mark_target_logits
2026-08-08 18:24:21,317 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-08 18:24:21,326 INFO Frozen decision_net.shoot_logit
2026-08-08 18:24:21,327 INFO Frozen decision_net.pass_logit
2026-08-08 18:24:21,327 INFO Frozen decision_net.tackle_logit
2026-08-08 18:24:21,327 INFO Frozen decision_net.get_possession_raw
2026-08-08 18:24:21,327 INFO Frozen decision_net.mark_logit
2026-08-08 18:24:21,327 INFO Frozen decision_net.hold_position_logit
2026-08-08 18:24:21,327 INFO Frozen decision_net.pass_target_logits
2026-08-08 18:24:21,327 INFO Frozen decision_net.tackle_target_logits
2026-08-08 18:24:21,327 INFO Frozen decision_net.mark_target_logits
2026-08-08 18:24:21,327 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-08 18:24:21,330 INFO Frozen decision_net.shoot_logit
2026-08-08 18:24:21,331 INFO Frozen decision_net.pass_logit
2026-08-08 18:24:21,331 INFO Frozen decision_net.tackle_logit
2026-08-08 18:24:21,331 INFO Frozen decision_net.get_possession_raw
2026-08-08 18:24:21,331 INFO Frozen decision_net.mark_logit
2026-08-08 18:24:21,331 INFO Frozen decision_net.hold_position_logit
2026-08-08 18:24:21,331 INFO Frozen decision_net.pass_target_logits
2026-08-08 18:24:21,331 INFO Frozen decision_net.tackle_target_logits
2026-08-08 18:24:21,331 INFO Frozen decision_net.mark_target_logits
2026-08-08 18:24:21,331 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-08 18:24:21,337 INFO Frozen decision_net.shoot_logit
2026-08-08 18:24:21,337 INFO Frozen decision_net.pass_logit
2026-08-08 18:24:21,337 INFO Frozen decision_net.tackle_logit
2026-08-08 18:24:21,337 INFO Frozen decision_net.get_possession_raw
2026-08-08 18:24:21,337 INFO Frozen decision_net.mark_logit
2026-08-08 18:24:21,337 INFO Frozen decision_net.hold_position_logit
2026-08-08 18:24:21,337 INFO Frozen decision_net.pass_target_logits
2026-08-08 18:24:21,337 INFO Frozen decision_net.tackle_target_logits
2026-08-08 18:24:21,337 INFO Frozen decision_net.mark_target_logits
2026-08-08 18:24:21,337 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-08 18:24:21,356 INFO Frozen decision_net.shoot_logit
2026-08-08 18:24:21,356 INFO Frozen decision_net.pass_logit
2026-08-08 18:24:21,356 INFO Frozen decision_net.tackle_logit
2026-08-08 18:24:21,356 INFO Frozen decision_net.get_possession_raw
2026-08-08 18:24:21,356 INFO Frozen decision_net.mark_logit
2026-08-08 18:24:21,356 INFO Frozen decision_net.hold_position_logit
2026-08-08 18:24:21,356 INFO Frozen decision_net.pass_target_logits
2026-08-08 18:24:21,356 INFO Frozen decision_net.tackle_target_logits
2026-08-08 18:24:21,356 INFO Frozen decision_net.mark_target_logits
2026-08-08 18:24:21,356 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-08 18:24:21,395 INFO Frozen decision_net.shoot_logit
2026-08-08 18:24:21,395 INFO Frozen decision_net.pass_logit
2026-08-08 18:24:21,395 INFO Frozen decision_net.tackle_logit
2026-08-08 18:24:21,395 INFO Frozen decision_net.get_possession_raw
2026-08-08 18:24:21,395 INFO Frozen decision_net.mark_logit
2026-08-08 18:24:21,395 INFO Frozen decision_net.hold_position_logit
2026-08-08 18:24:21,395 INFO Frozen decision_net.pass_target_logits
2026-08-08 18:24:21,395 INFO Frozen decision_net.tackle_target_logits
2026-08-08 18:24:21,395 INFO Frozen decision_net.mark_target_logits
2026-08-08 18:24:21,395 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-08 18:29:17,995 INFO   [advantage] mean=0.000  std=1.000  min=-4.869  max=3.441
2026-08-08 18:29:17,996 INFO   [ratio] mean=0.9781  std=0.2303  min=0.0006  max=20.3217  clipped=18.0%
2026-08-08 18:29:17,996 INFO   [exec head grad norm] move_direction=0.038  exec_move=0.077  sprint=0.046  kick=0.082  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.066
2026-08-08 18:29:17,996 INFO   [exec continuous log_std] move_direction: start=-1.6500 end=-1.6491   kick_direction: start=-1.6500 end=-1.6490
2026-08-08 18:29:17,996 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0010≈0.06°/step  epoch≈3.4°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0011≈0.07°/step  epoch≈3.9°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 18:29:17,997 INFO   [exec discrete Δlogit per opt step] exec_move=0.0022  sprint=0.0018  kick=0.0017  tackle_attempt=0.0022
2026-08-08 18:29:17,997 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0002  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0050  sprint=+0.0097  kick=+0.0042  tackle_attempt=+0.0008  move_dir=+0.0214  kick_dir=+0.0078
2026-08-08 18:29:17,997 INFO   [grad clip] main: 3/60 steps clipped (5%)  pre-clip norm mean=0.305 max=0.552  limit=0.4
              direction: 55/60 steps clipped (92%)  pre-clip norm mean=0.041 max=0.159  limit=0.02
2026-08-08 18:29:18,052 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=108,000  speed=777/s  reward=2.81
  loss     policy=0.0141  value=0.7312(x0.5)=0.3656
           entropy=1.7701  kl=0.0492
  value    V=2.07±0.88  R=2.32±1.75  adv=0.25±1.49
  moves    mv_ls=[-1.6491] (σ≈0.19, ≈11°) g=8.55e-03
           kk_ls=[-1.6490] (σ≈0.19, ≈11°)
  heads    move= 32 get_poss= 68 exec_move= 91 sprint= 41 kick=  5 tackle=  5 shoot=
           3 hold=  3 tackle_prob=0.0589 kick_prob=0.0520
  vs       vs[win/loss/tout/miss]  vs_immobile(638): 57.4%/0.0%/3.0%/17.4%/22%
  ep_len   16.6±11.0s  (n=638, min=0.2s, max=50.0s)
  reward   get_possession=+533.00  lose_possession=-8.10  ball_out=-168.00  box_possession=+915.00
           speed_bonus=+872.34  timeout=-28.50  stamina_penalty=-3.25
  rew/ep   (mean/std/min/max per episode, 638 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.835    0.411    +0.000    +3.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.013    0.117    -1.800    +0.000
  ball_out          -0.263    0.992    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.434    1.236    +0.000    +2.500
  speed_bonus       +1.367    1.515    +0.000    +4.368
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.045    0.255    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.005    0.007    -0.033    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     537    +0.015    0.121     +3.434     1.575     +1.718      5.3232      2.035     3.660
  lose_possession       9    -0.000    0.014     +1.621     0.977     -1.457      2.9102      1.457     2.601
  ball_out            42    -0.005    0.137     -3.976     0.152     -5.232     28.0618      5.232     6.799
  box_possession     366    +0.025    0.251     +4.883     1.248     +2.006      5.7096      2.078     3.813
  speed_bonus        354    +0.024    0.271     +4.964     1.188     +2.084      5.8919      2.132     3.823
  timeout             19    -0.001    0.034     -1.501     0.004     -2.831      8.9058      2.831     4.172
  stamina_penalty     334    -0.000    0.001     +4.987     1.291     +2.085      6.0584      2.178     3.823
  gae/td   mean_return=+2.316  std_return=1.745  mean_gae=+0.249  mean_sq_td=2.2724
──────────────────────────────────────────────────────────────────────
2026-08-08 18:29:18,074 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint1.pt
2026-08-08 18:29:18,074 INFO Logging to checkpoints/phase1_run45/training_log2.txt
2026-08-08 18:29:18,076 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 18:29:31,634 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 18:29:31,635 INFO   [eval vs immobile] step=108,000  seeds=16x8  win=48%  mean_rew=2.746±3.113  V=2.261  gap=-0.485  outcomes={'other': 37, 'box_possession': 62, 'timeout': 2, 'miss': 27}
2026-08-08 18:29:31,637 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 18:29:41,270 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 18:29:41,271 INFO   [eval vs rules] step=108,000  seeds=16x8  win=27%  mean_rew=0.337±3.640  V=1.776  gap=+1.439  outcomes={'opponent_box_possession': 56, 'other': 25, 'box_possession': 35, 'miss': 12}
2026-08-08 18:34:39,671 INFO   [advantage] mean=0.000  std=1.000  min=-4.842  max=4.067
2026-08-08 18:34:39,672 INFO   [ratio] mean=0.9810  std=0.3385  min=0.0025  max=102.4136  clipped=18.0%
2026-08-08 18:34:39,672 INFO   [exec head grad norm] move_direction=0.029  exec_move=0.062  sprint=0.045  kick=0.070  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.069
2026-08-08 18:34:39,672 INFO   [exec continuous log_std] move_direction: start=-1.6491 end=-1.6482   kick_direction: start=-1.6490 end=-1.6481
2026-08-08 18:34:39,672 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0006≈0.03°/step  epoch≈2.0°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0006≈0.03°/step  epoch≈2.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-08 18:34:39,672 INFO   [exec discrete Δlogit per opt step] exec_move=0.0020  sprint=0.0019  kick=0.0014  tackle_attempt=0.0022
2026-08-08 18:34:39,672 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0002  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0049  sprint=+0.0095  kick=+0.0036  tackle_attempt=+0.0009  move_dir=+0.0188  kick_dir=+0.0069
2026-08-08 18:34:39,673 INFO   [grad clip] main: 1/60 steps clipped (2%)  pre-clip norm mean=0.275 max=0.417  limit=0.4
              direction: 52/60 steps clipped (87%)  pre-clip norm mean=0.032 max=0.063  limit=0.02
2026-08-08 18:34:39,724 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=144,000  speed=805/s  reward=1.94
  loss     policy=0.0137  value=0.7064(x0.5)=0.3532
           entropy=1.9013  kl=0.0449
  value    V=2.39±0.86  R=2.65±1.77  adv=0.27±1.48
  moves    mv_ls=[-1.6482] (σ≈0.19, ≈11°) g=7.68e-03  d_move=[+0.0009] (Δσ≈0.010°)
           kk_ls=[-1.6481] (σ≈0.19, ≈11°)  d_kick=[+0.0009] (Δσ≈0.010°)
  heads    move= 35 get_poss= 65 exec_move= 90 sprint= 45 kick=  5 tackle=  6 shoot=
           3 hold=  3 tackle_prob=0.0641 kick_prob=0.0521
  vs       vs[win/loss/tout/miss]  vs_immobile(715): 61.5%/0.0%/1.0%/16.4%/21%
  ep_len   15.0±9.9s  (n=715, min=1.1s, max=50.0s)
  reward   get_possession=+581.00  lose_possession=-1.80  ball_out=-172.00  box_possession=+1100.00
           speed_bonus=+1033.34  timeout=-10.50  stamina_penalty=-3.72
  rew/ep   (mean/std/min/max per episode, 715 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.813    0.397    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.003    0.048    -0.900    +0.000
  ball_out          -0.241    0.951    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.538    1.216    +0.000    +2.500
  speed_bonus       +1.445    1.510    +0.000    +4.389
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.015    0.148    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.005    0.006    -0.042    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     585    +0.016    0.126     +3.760     1.469     +1.780      5.1903      2.037     3.501
  lose_possession       2    -0.000    0.007     +4.135     0.116     +1.212      1.6633      1.212     1.609
  ball_out            43    -0.005    0.138     -3.977     0.151     -5.042     25.8920      5.042     6.605
  box_possession     440    +0.031    0.275     +4.847     1.257     +1.805      4.8466      1.904     3.537
  speed_bonus        417    +0.029    0.293     +4.974     1.164     +1.914      5.0884      1.983     3.546
  timeout              7    -0.000    0.021     -1.501     0.002     -3.709     14.5867      3.709     4.918
  stamina_penalty     404    -0.000    0.001     +4.947     1.265     +1.888      5.1356      1.982     3.572
  gae/td   mean_return=+2.654  std_return=1.773  mean_gae=+0.269  mean_sq_td=2.2758
──────────────────────────────────────────────────────────────────────
2026-08-08 18:34:39,745 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint2.pt
2026-08-08 18:34:39,746 INFO Logging to checkpoints/phase1_run45/training_log3.txt
2026-08-08 18:34:39,747 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 18:34:51,812 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 18:34:51,814 INFO   [eval vs immobile] step=144,000  seeds=16x8  win=47%  mean_rew=2.558±3.264  V=2.513  gap=-0.046  outcomes={'other': 33, 'box_possession': 60, 'miss': 32, 'timeout': 3}
2026-08-08 18:34:51,815 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 18:35:01,856 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 18:35:01,858 INFO   [eval vs rules] step=144,000  seeds=16x8  win=27%  mean_rew=0.469±3.751  V=1.992  gap=+1.523  outcomes={'other': 29, 'box_possession': 35, 'opponent_box_possession': 53, 'miss': 11}
2026-08-08 18:40:01,579 INFO   [advantage] mean=0.000  std=1.000  min=-5.062  max=4.054
2026-08-08 18:40:01,581 INFO   [ratio] mean=0.9793  std=0.2132  min=0.0025  max=14.8590  clipped=18.8%
2026-08-08 18:40:01,581 INFO   [exec head grad norm] move_direction=0.033  exec_move=0.057  sprint=0.050  kick=0.078  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.070
2026-08-08 18:40:01,581 INFO   [exec continuous log_std] move_direction: start=-1.6482 end=-1.6473   kick_direction: start=-1.6481 end=-1.6471
2026-08-08 18:40:01,581 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0007≈0.04°/step  epoch≈2.3°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0007≈0.04°/step  epoch≈2.4°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 18:40:01,581 INFO   [exec discrete Δlogit per opt step] exec_move=0.0019  sprint=0.0019  kick=0.0021  tackle_attempt=0.0025
2026-08-08 18:40:01,582 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0008  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0043  sprint=+0.0084  kick=+0.0019  tackle_attempt=+0.0010  move_dir=+0.0201  kick_dir=+0.0077
2026-08-08 18:40:01,582 INFO   [grad clip] main: 4/60 steps clipped (7%)  pre-clip norm mean=0.301 max=0.429  limit=0.4
              direction: 50/60 steps clipped (83%)  pre-clip norm mean=0.036 max=0.167  limit=0.02
2026-08-08 18:40:01,626 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=180,000  speed=801/s  reward=3.54
  loss     policy=0.0142  value=0.6804(x0.5)=0.3402
           entropy=2.0606  kl=0.0442
  value    V=2.60±0.91  R=2.79±1.73  adv=0.19±1.43
  moves    mv_ls=[-1.6473] (σ≈0.19, ≈11°) g=8.05e-03  d_move=[+0.0009] (Δσ≈0.010°)
           kk_ls=[-1.6471] (σ≈0.19, ≈11°)  d_kick=[+0.0010] (Δσ≈0.011°)
  heads    move= 35 get_poss= 66 exec_move= 89 sprint= 44 kick=  5 tackle=  7 shoot=
           3 hold=  3 tackle_prob=0.0728 kick_prob=0.0538
  vs       vs[win/loss/tout/miss]  vs_immobile(717): 65.4%/0.0%/1.1%/15.5%/18%
  ep_len   14.9±10.4s  (n=717, min=0.2s, max=50.0s)
  reward   get_possession=+585.00  lose_possession=-1.80  ball_out=-152.00  box_possession=+1172.50
           speed_bonus=+1023.18  timeout=-12.00  stamina_penalty=-3.72
  rew/ep   (mean/std/min/max per episode, 717 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.816    0.395    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.003    0.047    -0.900    +0.000
  ball_out          -0.212    0.896    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.635    1.189    +0.000    +2.500
  speed_bonus       +1.427    1.473    +0.000    +4.347
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.017    0.158    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.005    0.006    -0.025    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     590    +0.016    0.127     +3.946     1.457     +1.691      4.8222      1.956     3.494
  lose_possession       2    -0.000    0.007     +1.395     1.369     -2.068      5.9187      2.068     3.221
  ball_out            38    -0.004    0.130     -3.947     0.223     -5.264     28.2693      5.264     6.844
  box_possession     469    +0.033    0.283     +4.676     1.286     +1.430      3.5796      1.578     3.306
  speed_bonus        443    +0.028    0.288     +4.804     1.207     +1.555      3.7495      1.629     3.312
  timeout              8    -0.000    0.022     -1.501     0.002     -3.293     11.3037      3.293     4.377
  stamina_penalty     426    -0.000    0.001     +4.791     1.314     +1.524      3.8887      1.678     3.334
  gae/td   mean_return=+2.789  std_return=1.734  mean_gae=+0.189  mean_sq_td=2.0677
──────────────────────────────────────────────────────────────────────
2026-08-08 18:40:01,649 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint3.pt
2026-08-08 18:40:01,650 INFO Logging to checkpoints/phase1_run45/training_log4.txt
2026-08-08 18:40:01,650 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 18:40:12,982 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 18:40:12,984 INFO   [eval vs immobile] step=180,000  seeds=16x8  win=48%  mean_rew=2.651±3.215  V=2.742  gap=+0.090  outcomes={'other': 36, 'box_possession': 61, 'miss': 31}
2026-08-08 18:40:12,985 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 18:40:23,247 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 18:40:23,248 INFO   [eval vs rules] step=180,000  seeds=16x8  win=23%  mean_rew=-0.126±3.497  V=2.006  gap=+2.132  outcomes={'other': 24, 'box_possession': 30, 'opponent_box_possession': 66, 'miss': 8}
2026-08-08 18:45:23,658 INFO   [advantage] mean=-0.000  std=1.000  min=-4.990  max=3.944
2026-08-08 18:45:23,659 INFO   [ratio] mean=0.9804  std=0.2467  min=0.0011  max=43.4194  clipped=19.2%
2026-08-08 18:45:23,659 INFO   [exec head grad norm] move_direction=0.028  exec_move=0.059  sprint=0.046  kick=0.070  kick_direction=0.008  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.071
2026-08-08 18:45:23,659 INFO   [exec continuous log_std] move_direction: start=-1.6473 end=-1.6464   kick_direction: start=-1.6471 end=-1.6460
2026-08-08 18:45:23,659 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0007≈0.04°/step  epoch≈2.4°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0008≈0.05°/step  epoch≈2.8°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 18:45:23,660 INFO   [exec discrete Δlogit per opt step] exec_move=0.0021  sprint=0.0024  kick=0.0020  tackle_attempt=0.0026
2026-08-08 18:45:23,660 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0009  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0043  sprint=+0.0084  kick=+0.0019  tackle_attempt=+0.0013  move_dir=+0.0194  kick_dir=+0.0067
2026-08-08 18:45:23,660 INFO   [grad clip] main: 2/60 steps clipped (3%)  pre-clip norm mean=0.287 max=0.417  limit=0.4
              direction: 50/60 steps clipped (83%)  pre-clip norm mean=0.032 max=0.099  limit=0.02
2026-08-08 18:45:23,707 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=216,000  speed=810/s  reward=4.18
  loss     policy=0.0149  value=0.7142(x0.5)=0.3571
           entropy=2.2309  kl=0.0429
  value    V=2.88±0.91  R=3.01±1.75  adv=0.14±1.48
  moves    mv_ls=[-1.6464] (σ≈0.19, ≈11°) g=8.67e-03  d_move=[+0.0010] (Δσ≈0.011°)
           kk_ls=[-1.6460] (σ≈0.19, ≈11°)  d_kick=[+0.0010] (Δσ≈0.012°)
  heads    move= 38 get_poss= 63 exec_move= 89 sprint= 47 kick=  5 tackle=  8 shoot=
           4 hold=  4 tackle_prob=0.0842 kick_prob=0.0553
  vs       vs[win/loss/tout/miss]  vs_immobile(712): 65.7%/0.0%/1.0%/15.7%/18%
  ep_len   15.1±9.1s  (n=712, min=0.6s, max=50.0s)
  reward   get_possession=+600.00  lose_possession=-7.20  ball_out=-164.00  box_possession=+1170.00
           speed_bonus=+1115.15  timeout=-10.50  stamina_penalty=-4.04
  rew/ep   (mean/std/min/max per episode, 712 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.843    0.394    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.010    0.095    -0.900    +0.000
  ball_out          -0.230    0.932    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.643    1.187    +0.000    +2.500
  speed_bonus       +1.566    1.501    +0.000    +4.328
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.015    0.148    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.027    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     603    +0.017    0.128     +4.049     1.461     +1.494      4.2858      1.828     3.298
  lose_possession       8    -0.000    0.013     +2.387     0.829     -0.976      1.4253      1.037     1.823
  ball_out            41    -0.005    0.135     -3.878     0.327     -4.844     24.3779      4.844     7.032
  box_possession     468    +0.033    0.283     +4.878     1.208     +1.375      3.3037      1.543     3.070
  speed_bonus        451    +0.031    0.303     +4.966     1.141     +1.450      3.3874      1.567     3.088
  timeout              7    -0.000    0.021     -1.500     0.000     -3.681     14.6191      3.681     4.855
  stamina_penalty     436    -0.000    0.001     +4.961     1.178     +1.432      3.3990      1.574     3.081
  gae/td   mean_return=+3.013  std_return=1.748  mean_gae=+0.136  mean_sq_td=2.1995
──────────────────────────────────────────────────────────────────────
2026-08-08 18:45:23,732 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint4.pt
2026-08-08 18:45:23,733 INFO Logging to checkpoints/phase1_run45/training_log5.txt
2026-08-08 18:45:23,734 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 18:45:35,207 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 18:45:35,209 INFO   [eval vs immobile] step=216,000  seeds=16x8  win=49%  mean_rew=2.807±3.141  V=2.925  gap=+0.118  outcomes={'other': 36, 'box_possession': 63, 'miss': 27, 'timeout': 2}
2026-08-08 18:45:35,210 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 18:45:44,999 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 18:45:45,000 INFO   [eval vs rules] step=216,000  seeds=16x8  win=29%  mean_rew=0.417±3.673  V=2.135  gap=+1.718  outcomes={'box_possession': 37, 'other': 26, 'opponent_box_possession': 56, 'miss': 9}
2026-08-08 18:50:48,660 INFO   [advantage] mean=0.000  std=1.000  min=-5.268  max=3.986
2026-08-08 18:50:48,661 INFO   [ratio] mean=0.9796  std=0.2203  min=0.0036  max=25.3112  clipped=20.1%
2026-08-08 18:50:48,661 INFO   [exec head grad norm] move_direction=0.027  exec_move=0.057  sprint=0.043  kick=0.070  kick_direction=0.010  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.062
2026-08-08 18:50:48,661 INFO   [exec continuous log_std] move_direction: start=-1.6464 end=-1.6454   kick_direction: start=-1.6460 end=-1.6450
2026-08-08 18:50:48,662 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0007≈0.04°/step  epoch≈2.3°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0009≈0.05°/step  epoch≈3.3°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 18:50:48,662 INFO   [exec discrete Δlogit per opt step] exec_move=0.0018  sprint=0.0018  kick=0.0018  tackle_attempt=0.0020
2026-08-08 18:50:48,662 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0017  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0041  sprint=+0.0080  kick=+0.0017  tackle_attempt=+0.0009  move_dir=+0.0195  kick_dir=+0.0078
2026-08-08 18:50:48,663 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.291 max=0.399  limit=0.4
              direction: 48/60 steps clipped (80%)  pre-clip norm mean=0.031 max=0.125  limit=0.02
2026-08-08 18:50:48,704 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=252,000  speed=804/s  reward=3.36
  loss     policy=0.0146  value=0.6753(x0.5)=0.3376
           entropy=2.4233  kl=0.0437
  value    V=3.00±0.96  R=3.08±1.76  adv=0.08±1.46
  moves    mv_ls=[-1.6454] (σ≈0.19, ≈11°) g=8.30e-03  d_move=[+0.0010] (Δσ≈0.011°)
           kk_ls=[-1.6450] (σ≈0.19, ≈11°)  d_kick=[+0.0011] (Δσ≈0.012°)
  heads    move= 39 get_poss= 62 exec_move= 88 sprint= 49 kick=  6 tackle=  9 shoot=
           4 hold=  4 tackle_prob=0.0917 kick_prob=0.0568
  vs       vs[win/loss/tout/miss]  vs_immobile(719): 65.5%/0.1%/0.3%/14.7%/19%
  ep_len   14.9±8.8s  (n=719, min=0.2s, max=50.0s)
  reward   get_possession=+602.00  lose_possession=-1.80  ball_out=-188.00  box_possession=+1177.50
           speed_bonus=+1097.43  opponent_box=-3.00  timeout=-3.00  stamina_penalty=-4.23
  rew/ep   (mean/std/min/max per episode, 719 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.837    0.377    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.003    0.047    -0.900    +0.000
  ball_out          -0.261    0.989    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.638    1.188    +0.000    +2.500
  speed_bonus       +1.526    1.488    +0.000    +4.342
  opponent_box      -0.004    0.112    -3.000    +0.000
  timeout           -0.004    0.079    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.007    -0.031    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     607    +0.017    0.129     +4.024     1.613     +1.209      4.1441      1.724     3.636
  lose_possession       2    -0.000    0.007     +2.978     0.468     -0.475      0.3778      0.475     0.826
  ball_out            47    -0.005    0.144     -3.936     0.244     -5.453     31.1537      5.453     7.377
  box_possession     471    +0.033    0.284     +4.823     1.223     +1.369      3.3029      1.561     3.133
  speed_bonus        450    +0.030    0.300     +4.930     1.144     +1.463      3.4114      1.593     3.170
  opponent_box         1    -0.000    0.016     -3.003     0.000     -5.411     29.2751      5.411     5.411
  timeout              2    -0.000    0.011     -1.500     0.000     -4.711     22.2124      4.711     4.821
  stamina_penalty     437    -0.000    0.001     +4.885     1.248     +1.410      3.4793      1.612     3.176
  gae/td   mean_return=+3.082  std_return=1.758  mean_gae=+0.083  mean_sq_td=2.1287
──────────────────────────────────────────────────────────────────────
2026-08-08 18:50:48,727 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint5.pt
2026-08-08 18:50:48,727 INFO Logging to checkpoints/phase1_run45/training_log6.txt
2026-08-08 18:50:48,728 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 18:51:00,951 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 18:51:00,952 INFO   [eval vs immobile] step=252,000  seeds=16x8  win=54%  mean_rew=3.099±2.980  V=3.100  gap=+0.001  outcomes={'other': 37, 'box_possession': 69, 'miss': 22}
2026-08-08 18:51:00,953 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 18:51:11,007 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 18:51:11,008 INFO   [eval vs rules] step=252,000  seeds=16x8  win=20%  mean_rew=-0.159±3.456  V=2.082  gap=+2.241  outcomes={'other': 30, 'opponent_box_possession': 63, 'box_possession': 26, 'miss': 9}
2026-08-08 18:56:17,418 INFO   [KL mean=0.0503 median=0.0501 > 0.05] ratio percentiles:  p5=0.696  p25=0.916  p50=0.970  p75=1.025  p95=1.247  max=17.179
  move_dir_log_std=[-1.6445019245147705]  kick_dir_log_std=[-1.6441073417663574]
2026-08-08 18:56:17,435 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.262  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.263  kick=-0.221  t_att=-0.265
    move_dir=0.899 (min=-1.623 max=1.451)  kick_dir=0.086 (min=-0.308 max=2.086)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.36
  [worst sample] idx=226  ratio=22.598  adv=-3.808  old_lp=-5.651  new_lp=-2.533
    stored move_dir=148.4°  new_mean=173.6°  angular_diff=25.2°
    [worst sample per-head delta, sorted by |delta|] move:+0.169
  [top-2 highest-ratio samples]
    idx= 226  ratio=  22.598  adv=-3.808  lp: old=-5.651  new=-2.533
      rew=+0.0000  ret=-1.4903  val=+2.3173  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: move:+0.169
      saturation: exec_move_p_new=0.9405  sprint_p_new=0.2282  kick_p_new=0.1055  tackle_attempt_p_new=0.0987
    idx=  99  ratio=  18.273  adv=-0.018  lp: old=-5.225  new=-2.319
      rew=+0.0000  ret=+4.3214  val=+4.3395  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: move:+0.401
      saturation: exec_move_p_new=0.9060  sprint_p_new=0.9025  kick_p_new=0.0184  tackle_attempt_p_new=0.0744
  [best sample (highest new_lp)] idx=19  new_lp=-0.160  adv=-0.066  stored move_dir=18.5°  new_mean=24.9°
    per-head contributions: kick:-0.031  move:-0.048  tackle_attempt:-0.081
2026-08-08 18:56:17,436 INFO   [advantage] mean=0.000  std=1.000  min=-5.688  max=3.547
2026-08-08 18:56:17,437 INFO   [ratio] mean=0.9749  std=0.2118  min=0.0074  max=17.1791  clipped=21.9%
2026-08-08 18:56:17,437 INFO   [exec head grad norm] move_direction=0.025  exec_move=0.054  sprint=0.045  kick=0.063  kick_direction=0.013  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.070
2026-08-08 18:56:17,437 INFO   [exec continuous log_std] move_direction: start=-1.6454 end=-1.6445   kick_direction: start=-1.6450 end=-1.6441
2026-08-08 18:56:17,437 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0008≈0.04°/step  epoch≈2.7°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0008≈0.05°/step  epoch≈2.9°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-08 18:56:17,437 INFO   [exec discrete Δlogit per opt step] exec_move=0.0018  sprint=0.0020  kick=0.0019  tackle_attempt=0.0022
2026-08-08 18:56:17,438 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0019  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0043  sprint=+0.0081  kick=+0.0016  tackle_attempt=+0.0014  move_dir=+0.0231  kick_dir=+0.0097
2026-08-08 18:56:17,438 INFO   [grad clip] main: 3/60 steps clipped (5%)  pre-clip norm mean=0.283 max=0.453  limit=0.4
              direction: 55/60 steps clipped (92%)  pre-clip norm mean=0.030 max=0.069  limit=0.02
2026-08-08 18:56:17,476 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=288,000  speed=795/s  reward=4.70
  loss     policy=0.0144  value=0.6265(x0.5)=0.3133
           entropy=2.6405  kl=0.0503
  value    V=3.12±0.95  R=3.29±1.70  adv=0.17±1.35
  moves    mv_ls=[-1.6445] (σ≈0.19, ≈11°) g=8.12e-03  d_move=[+0.0009] (Δσ≈0.010°)
           kk_ls=[-1.6441] (σ≈0.19, ≈11°)  d_kick=[+0.0008] (Δσ≈0.009°)
  heads    move= 39 get_poss= 62 exec_move= 88 sprint= 49 kick=  6 tackle=  9 shoot=
           5 hold=  5 tackle_prob=0.1013 kick_prob=0.0600
  vs       vs[win/loss/tout/miss]  vs_immobile(732): 69.3%/0.3%/0.3%/13.5%/17%
  ep_len   14.7±9.0s  (n=732, min=0.6s, max=50.0s)
  reward   get_possession=+609.00  lose_possession=-4.50  ball_out=-144.00  box_possession=+1267.50
           speed_bonus=+1201.74  opponent_box=-6.00  timeout=-3.00  stamina_penalty=-4.50
  rew/ep   (mean/std/min/max per episode, 732 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.832    0.392    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.006    0.074    -0.900    +0.000
  ball_out          -0.197    0.865    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.732    1.154    +0.000    +2.500
  speed_bonus       +1.642    1.497    +0.000    +4.305
  opponent_box      -0.008    0.157    -3.000    +0.000
  timeout           -0.004    0.078    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.028    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     612    +0.017    0.129     +4.180     1.610     +1.149      3.8375      1.658     3.365
  lose_possession       5    -0.000    0.011     +2.899     1.216     -1.177      3.0314      1.243     3.068
  ball_out            36    -0.004    0.126     -4.000     0.000     -5.915     36.4598      5.915     7.377
  box_possession     507    +0.035    0.295     +4.869     1.214     +1.299      3.0162      1.472     3.048
  speed_bonus        490    +0.033    0.315     +4.950     1.154     +1.359      3.0835      1.496     3.097
  opponent_box         2    -0.000    0.022     -3.002     0.001     -5.752     33.1842      5.752     6.037
  timeout              2    -0.000    0.011     -1.500     0.000     -4.097     16.8185      4.097     4.265
  stamina_penalty     472    -0.000    0.001     +4.922     1.316     +1.318      3.2651      1.534     3.121
  gae/td   mean_return=+3.292  std_return=1.704  mean_gae=+0.170  mean_sq_td=1.8444
──────────────────────────────────────────────────────────────────────
2026-08-08 18:56:17,500 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint6.pt
2026-08-08 18:56:17,500 INFO Logging to checkpoints/phase1_run45/training_log7.txt
2026-08-08 18:56:17,501 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 18:56:28,476 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 18:56:28,478 INFO   [eval vs immobile] step=288,000  seeds=16x8  win=59%  mean_rew=3.447±2.993  V=3.294  gap=-0.153  outcomes={'other': 32, 'box_possession': 76, 'miss': 19, 'timeout': 1}
2026-08-08 18:56:28,479 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 18:56:40,117 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 18:56:40,118 INFO   [eval vs rules] step=288,000  seeds=16x8  win=26%  mean_rew=0.226±3.661  V=2.180  gap=+1.954  outcomes={'box_possession': 33, 'other': 26, 'opponent_box_possession': 60, 'miss': 9}
2026-08-08 19:01:31,328 INFO   [KL mean=0.0514 median=0.0516 > 0.05] ratio percentiles:  p5=0.683  p25=0.908  p50=0.968  p75=1.030  p95=1.268  max=6.505
  move_dir_log_std=[-1.6435720920562744]  kick_dir_log_std=[-1.6430743932724]
2026-08-08 19:01:31,339 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.382  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.338  kick=-0.167  t_att=-0.314
    move_dir=0.758 (min=-9.031 max=1.449)  kick_dir=0.054 (min=0.000 max=2.164)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.28
  [worst sample] idx=33  ratio=10.924  adv=+0.525  old_lp=-2.563  new_lp=-0.172
    stored move_dir=-160.2°  new_mean=-170.8°  angular_diff=10.6°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx=  33  ratio=  10.924  adv=+0.525  lp: old=-2.563  new=-0.172
      rew=+0.0000  ret=+5.1404  val=+4.6158  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9020  sprint_p_new=0.9038  kick_p_new=0.0131  tackle_attempt_p_new=0.0719
    idx=  52  ratio=  10.497  adv=+2.267  lp: old=-2.598  new=-0.247
      rew=+0.0000  ret=+6.3531  val=+4.0861  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: move:-0.045
      saturation: exec_move_p_new=0.8999  sprint_p_new=0.8904  kick_p_new=0.0171  tackle_attempt_p_new=0.0800
  [best sample (highest new_lp)] idx=252  new_lp=-0.162  adv=-0.208  stored move_dir=-15.4°  new_mean=-6.6°
    per-head contributions: kick:-0.023  move:-0.058  tackle_attempt:-0.080
2026-08-08 19:01:31,339 INFO   [advantage] mean=-0.000  std=1.000  min=-5.487  max=3.136
2026-08-08 19:01:31,340 INFO   [ratio] mean=0.9752  std=0.2153  min=0.0019  max=6.5049  clipped=24.1%
2026-08-08 19:01:31,341 INFO   [exec head grad norm] move_direction=0.030  exec_move=0.056  sprint=0.050  kick=0.059  kick_direction=0.010  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.079
2026-08-08 19:01:31,341 INFO   [exec continuous log_std] move_direction: start=-1.6445 end=-1.6436   kick_direction: start=-1.6441 end=-1.6431
2026-08-08 19:01:31,341 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0008≈0.05°/step  epoch≈2.8°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0009≈0.05°/step  epoch≈3.0°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 19:01:31,341 INFO   [exec discrete Δlogit per opt step] exec_move=0.0016  sprint=0.0022  kick=0.0022  tackle_attempt=0.0023
2026-08-08 19:01:31,341 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0025  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0039  sprint=+0.0072  kick=+0.0021  tackle_attempt=+0.0016  move_dir=+0.0240  kick_dir=+0.0100
2026-08-08 19:01:31,342 INFO   [grad clip] main: 5/60 steps clipped (8%)  pre-clip norm mean=0.302 max=0.466  limit=0.4
              direction: 49/60 steps clipped (82%)  pre-clip norm mean=0.034 max=0.100  limit=0.02
2026-08-08 19:01:31,392 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=324,000  speed=812/s  reward=4.28
  loss     policy=0.0156  value=0.6126(x0.5)=0.3063
           entropy=2.8875  kl=0.0514
  value    V=3.25±1.02  R=3.22±1.78  adv=-0.03±1.38
  moves    mv_ls=[-1.6436] (σ≈0.19, ≈11°) g=8.73e-03  d_move=[+0.0009] (Δσ≈0.010°)
           kk_ls=[-1.6431] (σ≈0.19, ≈11°)  d_kick=[+0.0010] (Δσ≈0.011°)
  heads    move= 38 get_poss= 63 exec_move= 87 sprint= 47 kick=  6 tackle= 11 shoot=
           5 hold=  5 tackle_prob=0.1147 kick_prob=0.0655
  vs       vs[win/loss/tout/miss]  vs_immobile(680): 67.1%/0.1%/0.9%/16.0%/16%
  ep_len   15.8±9.2s  (n=680, min=0.8s, max=50.0s)
  reward   get_possession=+576.00  lose_possession=-5.40  ball_out=-188.00  box_possession=+1140.00
           speed_bonus=+1066.41  opponent_box=-3.00  timeout=-9.00  stamina_penalty=-3.96
  rew/ep   (mean/std/min/max per episode, 680 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.847    0.384    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.008    0.084    -0.900    +0.000
  ball_out          -0.276    1.015    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.676    1.175    +0.000    +2.500
  speed_bonus       +1.568    1.466    +0.000    +4.200
  opponent_box      -0.004    0.115    -3.000    +0.000
  timeout           -0.013    0.140    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.026    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     578    +0.016    0.126     +4.137     1.710     +0.930      3.5177      1.557     3.515
  lose_possession       6    -0.000    0.012     +3.129     0.659     -0.717      1.6213      1.221     1.586
  ball_out            47    -0.005    0.144     -3.894     0.308     -5.389     30.6558      5.389     7.383
  box_possession     456    +0.032    0.280     +4.830     1.182     +1.115      2.5027      1.333     2.757
  speed_bonus        439    +0.030    0.294     +4.920     1.110     +1.188      2.5668      1.354     2.779
  opponent_box         1    -0.000    0.016     -3.002     0.000     -5.876     34.5265      5.876     5.876
  timeout              6    -0.000    0.019     -1.501     0.001     -4.307     20.7165      4.307     6.067
  stamina_penalty     414    -0.000    0.001     +4.888     1.311     +1.150      2.8641      1.417     2.846
  gae/td   mean_return=+3.222  std_return=1.777  mean_gae=-0.032  mean_sq_td=1.9148
──────────────────────────────────────────────────────────────────────
2026-08-08 19:01:31,423 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint7.pt
2026-08-08 19:01:31,424 INFO Logging to checkpoints/phase1_run45/training_log8.txt
2026-08-08 19:01:31,425 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 19:01:42,423 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 19:01:42,424 INFO   [eval vs immobile] step=324,000  seeds=16x8  win=52%  mean_rew=2.918±3.017  V=3.123  gap=+0.205  outcomes={'other': 36, 'box_possession': 66, 'miss': 26}
2026-08-08 19:01:42,426 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 19:01:51,128 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 19:01:51,129 INFO   [eval vs rules] step=324,000  seeds=16x8  win=27%  mean_rew=0.273±3.730  V=2.220  gap=+1.947  outcomes={'box_possession': 35, 'other': 24, 'opponent_box_possession': 60, 'miss': 9}
2026-08-08 19:06:33,224 INFO   [advantage] mean=0.000  std=1.000  min=-5.664  max=3.634
2026-08-08 19:06:33,225 INFO   [ratio] mean=0.9759  std=0.2166  min=0.0039  max=11.8034  clipped=25.0%
2026-08-08 19:06:33,226 INFO   [exec head grad norm] move_direction=0.028  exec_move=0.055  sprint=0.051  kick=0.055  kick_direction=0.011  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.073
2026-08-08 19:06:33,226 INFO   [exec continuous log_std] move_direction: start=-1.6436 end=-1.6427   kick_direction: start=-1.6431 end=-1.6421
2026-08-08 19:06:33,226 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0009≈0.05°/step  epoch≈2.9°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0010≈0.06°/step  epoch≈3.6°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 19:06:33,226 INFO   [exec discrete Δlogit per opt step] exec_move=0.0017  sprint=0.0020  kick=0.0022  tackle_attempt=0.0022
2026-08-08 19:06:33,226 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0030  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0042  sprint=+0.0074  kick=+0.0019  tackle_attempt=+0.0014  move_dir=+0.0232  kick_dir=+0.0087
2026-08-08 19:06:33,227 INFO   [grad clip] main: 7/60 steps clipped (12%)  pre-clip norm mean=0.319 max=0.515  limit=0.4
              direction: 56/60 steps clipped (93%)  pre-clip norm mean=0.033 max=0.102  limit=0.02
2026-08-08 19:06:33,272 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=360,000  speed=872/s  reward=2.84
  loss     policy=0.0160  value=0.6277(x0.5)=0.3138
           entropy=3.1275  kl=0.0498
  value    V=3.25±1.06  R=3.31±1.72  adv=0.06±1.37
  moves    mv_ls=[-1.6427] (σ≈0.19, ≈11°) g=8.60e-03  d_move=[+0.0009] (Δσ≈0.010°)
           kk_ls=[-1.6421] (σ≈0.19, ≈11°)  d_kick=[+0.0009] (Δσ≈0.010°)
  heads    move= 40 get_poss= 62 exec_move= 86 sprint= 49 kick=  6 tackle= 12 shoot=
           6 hold=  6 tackle_prob=0.1245 kick_prob=0.0662
  vs       vs[win/loss/tout/miss]  vs_immobile(711): 69.2%/0.3%/0.3%/14.3%/16%
  ep_len   15.0±8.6s  (n=711, min=1.2s, max=50.0s)
  reward   get_possession=+603.00  lose_possession=-3.60  ball_out=-160.00  box_possession=+1230.00
           speed_bonus=+1145.95  opponent_box=-6.00  timeout=-3.00  stamina_penalty=-4.21
  rew/ep   (mean/std/min/max per episode, 711 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.848    0.374    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.005    0.067    -0.900    +0.000
  ball_out          -0.225    0.922    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.730    1.154    +0.000    +2.500
  speed_bonus       +1.612    1.490    +0.000    +4.357
  opponent_box      -0.008    0.159    -3.000    +0.000
  timeout           -0.004    0.079    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.038    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     608    +0.017    0.129     +4.149     1.707     +0.948      3.6486      1.577     3.546
  lose_possession       4    -0.000    0.009     +3.050     0.403     -0.877      0.9512      0.877     1.433
  ball_out            40    -0.004    0.133     -3.950     0.218     -5.770     34.7613      5.770     7.279
  box_possession     492    +0.034    0.290     +4.825     1.236     +1.278      3.0258      1.478     3.043
  speed_bonus        466    +0.032    0.307     +4.952     1.141     +1.380      3.1396      1.514     3.078
  opponent_box         2    -0.000    0.022     -3.002     0.001     -6.092     37.1242      6.092     6.161
  timeout              2    -0.000    0.011     -1.500     0.000     -5.242     27.9911      5.242     5.886
  stamina_penalty     455    -0.000    0.001     +4.885     1.349     +1.316      3.3943      1.554     3.157
  gae/td   mean_return=+3.308  std_return=1.725  mean_gae=+0.056  mean_sq_td=1.8841
──────────────────────────────────────────────────────────────────────
2026-08-08 19:06:33,299 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint8.pt
2026-08-08 19:06:33,299 INFO Logging to checkpoints/phase1_run45/training_log9.txt
2026-08-08 19:06:33,300 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 19:06:44,230 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 19:06:44,231 INFO   [eval vs immobile] step=360,000  seeds=16x8  win=56%  mean_rew=3.135±3.083  V=3.268  gap=+0.133  outcomes={'other': 30, 'box_possession': 72, 'timeout': 1, 'miss': 25}
2026-08-08 19:06:44,232 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 19:06:53,496 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 19:06:53,498 INFO   [eval vs rules] step=360,000  seeds=16x8  win=19%  mean_rew=-0.202±3.372  V=2.284  gap=+2.486  outcomes={'other': 32, 'opponent_box_possession': 64, 'box_possession': 24, 'miss': 8}
2026-08-08 19:11:37,887 INFO   [KL mean=0.0584 median=0.0584 > 0.05] ratio percentiles:  p5=0.660  p25=0.892  p50=0.961  p75=1.036  p95=1.290  max=31.987
  move_dir_log_std=[-1.641772747039795]  kick_dir_log_std=[-1.6413261890411377]
2026-08-08 19:11:37,900 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.421  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.306  kick=-0.239  t_att=-0.362
    move_dir=0.731 (min=-6.794 max=1.446)  kick_dir=0.048 (min=-3.873 max=2.111)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.39
  [worst sample] idx=24  ratio=14.187  adv=-0.212  old_lp=-4.567  new_lp=-1.915
    stored move_dir=118.8°  new_mean=109.0°  angular_diff=9.8°
    [worst sample per-head delta, sorted by |delta|] move:-0.044  tackle_attempt:+0.102
  [top-2 highest-ratio samples]
    idx=  24  ratio=  14.187  adv=-0.212  lp: old=-4.567  new=-1.915
      rew=+0.0000  ret=+3.1400  val=+3.3524  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: tackle_attempt:+0.102  move:-0.044
      saturation: exec_move_p_new=0.9195  sprint_p_new=0.3119  kick_p_new=0.1425  tackle_attempt_p_new=0.2064
    idx=  30  ratio=  13.011  adv=-0.179  lp: old=-6.382  new=-3.817
      rew=+0.0000  ret=+3.1919  val=+3.3712  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: kick:+0.098  tackle_attempt:+0.091  move:-0.043
      saturation: exec_move_p_new=0.9027  sprint_p_new=0.1131  kick_p_new=0.1282  tackle_attempt_p_new=0.1911
  [best sample (highest new_lp)] idx=237  new_lp=-0.198  adv=-0.565  stored move_dir=-174.4°  new_mean=-175.9°
    per-head contributions: move:-0.089  tackle_attempt:-0.093
2026-08-08 19:11:37,901 INFO   [advantage] mean=0.000  std=1.000  min=-5.532  max=3.359
2026-08-08 19:11:37,902 INFO   [ratio] mean=0.9711  std=0.2364  min=0.0007  max=31.9874  clipped=28.2%
2026-08-08 19:11:37,903 INFO   [exec head grad norm] move_direction=0.046  exec_move=0.052  sprint=0.054  kick=0.053  kick_direction=0.011  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.070
2026-08-08 19:11:37,903 INFO   [exec continuous log_std] move_direction: start=-1.6427 end=-1.6418   kick_direction: start=-1.6421 end=-1.6413
2026-08-08 19:11:37,903 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0011≈0.06°/step  epoch≈3.7°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0012≈0.07°/step  epoch≈4.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-08 19:11:37,903 INFO   [exec discrete Δlogit per opt step] exec_move=0.0017  sprint=0.0024  kick=0.0025  tackle_attempt=0.0022
2026-08-08 19:11:37,903 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0038  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0045  sprint=+0.0069  kick=+0.0022  tackle_attempt=+0.0023  move_dir=+0.0285  kick_dir=+0.0103
2026-08-08 19:11:37,904 INFO   [grad clip] main: 10/60 steps clipped (17%)  pre-clip norm mean=0.329 max=1.382  limit=0.4
              direction: 59/60 steps clipped (98%)  pre-clip norm mean=0.050 max=0.770  limit=0.02
2026-08-08 19:11:37,950 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=396,000  speed=873/s  reward=2.16
  loss     policy=0.0188  value=0.6143(x0.5)=0.3072
           entropy=3.4107  kl=0.0584
  value    V=3.25±1.07  R=3.20±1.77  adv=-0.05±1.38
  moves    mv_ls=[-1.6418] (σ≈0.19, ≈11°) g=9.91e-03  d_move=[+0.0009] (Δσ≈0.010°)
           kk_ls=[-1.6413] (σ≈0.19, ≈11°)  d_kick=[+0.0008] (Δσ≈0.009°)
  heads    move= 39 get_poss= 64 exec_move= 85 sprint= 48 kick=  7 tackle= 13 shoot=
           7 hold=  7 tackle_prob=0.1399 kick_prob=0.0739
  vs       vs[win/loss/tout/miss]  vs_immobile(703): 64.2%/0.0%/0.6%/16.8%/18%
  ep_len   15.2±9.3s  (n=703, min=0.9s, max=50.0s)
  reward   get_possession=+567.00  lose_possession=-8.10  ball_out=-176.00  box_possession=+1127.50
           speed_bonus=+1029.38  timeout=-6.00  stamina_penalty=-3.83
  rew/ep   (mean/std/min/max per episode, 703 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.807    0.426    +0.000    +3.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.012    0.112    -1.800    +0.000
  ball_out          -0.250    0.969    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.604    1.199    +0.000    +2.500
  speed_bonus       +1.464    1.467    +0.000    +4.190
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.009    0.113    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.005    0.006    -0.033    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     570    +0.016    0.125     +4.187     1.606     +0.893      3.1189      1.436     3.098
  lose_possession       9    -0.000    0.014     +2.693     1.250     -1.248      1.9726      1.248     2.199
  ball_out            44    -0.005    0.140     -3.955     0.208     -5.317     30.0554      5.317     7.409
  box_possession     451    +0.031    0.278     +4.783     1.216     +1.168      2.7041      1.394     2.872
  speed_bonus        431    +0.029    0.288     +4.889     1.138     +1.259      2.7879      1.419     2.883
  timeout              4    -0.000    0.016     -1.503     0.005     -4.614     22.1007      4.614     5.680
  stamina_penalty     423    -0.000    0.001     +4.833     1.264     +1.184      2.8646      1.436     2.911
  gae/td   mean_return=+3.204  std_return=1.766  mean_gae=-0.049  mean_sq_td=1.9152
──────────────────────────────────────────────────────────────────────
2026-08-08 19:11:37,975 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint9.pt
2026-08-08 19:11:37,976 INFO Logging to checkpoints/phase1_run45/training_log10.txt
2026-08-08 19:11:37,977 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 19:11:48,340 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 19:11:48,342 INFO   [eval vs immobile] step=396,000  seeds=16x8  win=59%  mean_rew=3.364±3.052  V=3.199  gap=-0.164  outcomes={'other': 28, 'box_possession': 75, 'miss': 25}
2026-08-08 19:11:48,343 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 19:11:57,798 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 19:11:57,800 INFO   [eval vs rules] step=396,000  seeds=16x8  win=18%  mean_rew=-0.383±3.427  V=2.130  gap=+2.513  outcomes={'box_possession': 23, 'other': 27, 'opponent_box_possession': 67, 'miss': 11}
2026-08-08 19:16:44,818 INFO   [KL mean=0.0603 median=0.0603 > 0.05] ratio percentiles:  p5=0.654  p25=0.886  p50=0.962  p75=1.036  p95=1.291  max=8.212
  move_dir_log_std=[-1.6408287286758423]  kick_dir_log_std=[-1.6402333974838257]
2026-08-08 19:16:44,830 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.463  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.299  kick=-0.351  t_att=-0.457
    move_dir=0.784 (min=-3.410 max=1.444)  kick_dir=0.156 (min=-0.010 max=2.060)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.45
  [worst sample] idx=185  ratio=17.062  adv=-0.635  old_lp=-7.174  new_lp=-4.337
    stored move_dir=-30.3°  new_mean=-19.2°  angular_diff=11.0°
    [worst sample per-head delta, sorted by |delta|] move:+0.248  tackle_attempt:+0.096
  [top-2 highest-ratio samples]
    idx= 185  ratio=  17.062  adv=-0.635  lp: old=-7.174  new=-4.337
      rew=+0.0000  ret=+3.2807  val=+3.9158  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: move:+0.248  tackle_attempt:+0.096
      saturation: exec_move_p_new=0.9089  sprint_p_new=0.9086  kick_p_new=0.0201  tackle_attempt_p_new=0.0954
    idx= 118  ratio=  12.758  adv=+1.632  lp: old=-2.860  new=-0.314
      rew=+0.0000  ret=+2.6018  val=+0.9697  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9149  sprint_p_new=0.7534  kick_p_new=0.0844  tackle_attempt_p_new=0.1341
  [best sample (highest new_lp)] idx=97  new_lp=-0.185  adv=-0.019  stored move_dir=31.9°  new_mean=10.4°
    per-head contributions: move:-0.058  tackle_attempt:-0.107
2026-08-08 19:16:44,830 INFO   [advantage] mean=-0.000  std=1.000  min=-5.671  max=3.313
2026-08-08 19:16:44,831 INFO   [ratio] mean=0.9691  std=0.2237  min=0.0011  max=8.2119  clipped=30.0%
2026-08-08 19:16:44,832 INFO   [exec head grad norm] move_direction=0.026  exec_move=0.063  sprint=0.048  kick=0.058  kick_direction=0.011  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.075
2026-08-08 19:16:44,832 INFO   [exec continuous log_std] move_direction: start=-1.6418 end=-1.6408   kick_direction: start=-1.6413 end=-1.6402
2026-08-08 19:16:44,832 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0009≈0.05°/step  epoch≈3.1°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0010≈0.06°/step  epoch≈3.6°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 19:16:44,832 INFO   [exec discrete Δlogit per opt step] exec_move=0.0017  sprint=0.0024  kick=0.0021  tackle_attempt=0.0019
2026-08-08 19:16:44,832 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0056  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0041  sprint=+0.0071  kick=+0.0020  tackle_attempt=+0.0018  move_dir=+0.0280  kick_dir=+0.0117
2026-08-08 19:16:44,833 INFO   [grad clip] main: 10/60 steps clipped (17%)  pre-clip norm mean=0.336 max=0.575  limit=0.4
              direction: 53/60 steps clipped (88%)  pre-clip norm mean=0.031 max=0.068  limit=0.02
2026-08-08 19:16:44,883 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=432,000  speed=861/s  reward=2.48
  loss     policy=0.0172  value=0.6231(x0.5)=0.3115
           entropy=3.6900  kl=0.0603
  value    V=3.23±1.05  R=3.23±1.69  adv=0.00±1.33
  moves    mv_ls=[-1.6408] (σ≈0.19, ≈11°) g=9.00e-03  d_move=[+0.0009] (Δσ≈0.010°)
           kk_ls=[-1.6402] (σ≈0.19, ≈11°)  d_kick=[+0.0011] (Δσ≈0.012°)
  heads    move= 40 get_poss= 64 exec_move= 84 sprint= 47 kick=  8 tackle= 14 shoot=
           8 hold=  8 tackle_prob=0.1523 kick_prob=0.0808
  vs       vs[win/loss/tout/miss]  vs_immobile(681): 67.0%/0.1%/0.4%/15.1%/17%
  ep_len   15.7±9.2s  (n=681, min=1.4s, max=50.0s)
  reward   get_possession=+568.00  lose_possession=-4.50  ball_out=-140.00  box_possession=+1140.00
           speed_bonus=+1030.25  opponent_box=-3.00  timeout=-4.50  stamina_penalty=-3.75
  rew/ep   (mean/std/min/max per episode, 681 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.834    0.391    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.007    0.077    -0.900    +0.000
  ball_out          -0.206    0.883    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.674    1.176    +0.000    +2.500
  speed_bonus       +1.513    1.440    +0.000    +4.321
  opponent_box      -0.004    0.115    -3.000    +0.000
  timeout           -0.007    0.099    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.025    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     572    +0.016    0.125     +4.172     1.570     +0.859      3.0044      1.403     3.178
  lose_possession       5    -0.000    0.011     +3.284     0.477     -0.753      0.7871      0.795     1.168
  ball_out            35    -0.004    0.125     -3.943     0.232     -5.418     31.3180      5.418     7.302
  box_possession     456    +0.032    0.280     +4.753     1.182     +1.121      2.4141      1.309     2.776
  speed_bonus        436    +0.029    0.286     +4.857     1.103     +1.196      2.4933      1.337     2.847
  opponent_box         1    -0.000    0.016     -3.001     0.000     -6.755     45.6258      6.755     6.755
  timeout              3    -0.000    0.014     -1.511     0.004     -5.466     30.1237      5.466     5.951
  stamina_penalty     425    -0.000    0.001     +4.779     1.319     +1.100      2.7898      1.386     2.910
  gae/td   mean_return=+3.230  std_return=1.690  mean_gae=+0.001  mean_sq_td=1.7674
──────────────────────────────────────────────────────────────────────
2026-08-08 19:16:44,907 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint10.pt
2026-08-08 19:16:44,907 INFO Logging to checkpoints/phase1_run45/training_log11.txt
2026-08-08 19:16:44,908 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 19:16:56,495 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 19:16:56,496 INFO   [eval vs immobile] step=432,000  seeds=16x8  win=55%  mean_rew=3.002±3.018  V=3.041  gap=+0.040  outcomes={'other': 32, 'box_possession': 70, 'miss': 26}
2026-08-08 19:16:56,497 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 19:17:06,562 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 19:17:06,563 INFO   [eval vs rules] step=432,000  seeds=16x8  win=20%  mean_rew=-0.130±3.371  V=2.248  gap=+2.378  outcomes={'opponent_box_possession': 61, 'other': 34, 'box_possession': 25, 'miss': 8}
2026-08-08 19:21:49,340 INFO   [KL mean=0.0633 median=0.0632 > 0.05] ratio percentiles:  p5=0.641  p25=0.880  p50=0.962  p75=1.039  p95=1.296  max=11.338
  move_dir_log_std=[-1.639859676361084]  kick_dir_log_std=[-1.639216423034668]
2026-08-08 19:21:49,351 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.449  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.299  kick=-0.212  t_att=-0.416
    move_dir=0.783 (min=-5.226 max=1.442)  kick_dir=0.062 (min=-1.522 max=2.156)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.49
  [worst sample] idx=75  ratio=10.362  adv=+1.011  old_lp=-2.811  new_lp=-0.473
    stored move_dir=60.4°  new_mean=39.9°  angular_diff=20.5°
    [worst sample per-head delta, sorted by |delta|] move:-0.024
  [top-2 highest-ratio samples]
    idx=  75  ratio=  10.362  adv=+1.011  lp: old=-2.811  new=-0.473
      rew=+0.0000  ret=+4.0221  val=+3.0115  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: move:-0.024
      saturation: exec_move_p_new=0.9002  sprint_p_new=0.5227  kick_p_new=0.1165  tackle_attempt_p_new=0.1708
    idx=  99  ratio=  10.070  adv=-0.192  lp: old=-4.157  new=-1.848
      rew=+0.0000  ret=+4.0526  val=+4.2442  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: move:+0.262
      saturation: exec_move_p_new=0.8668  sprint_p_new=0.8565  kick_p_new=0.0166  tackle_attempt_p_new=0.1121
  [best sample (highest new_lp)] idx=209  new_lp=-0.214  adv=-0.277  stored move_dir=173.4°  new_mean=-175.6°
    per-head contributions: move:-0.087  tackle_attempt:-0.111
2026-08-08 19:21:49,351 INFO   [advantage] mean=0.000  std=1.000  min=-5.564  max=4.564
2026-08-08 19:21:49,352 INFO   [ratio] mean=0.9677  std=0.2309  min=0.0059  max=11.3384  clipped=31.4%
2026-08-08 19:21:49,353 INFO   [exec head grad norm] move_direction=0.031  exec_move=0.057  sprint=0.041  kick=0.052  kick_direction=0.012  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.071
2026-08-08 19:21:49,353 INFO   [exec continuous log_std] move_direction: start=-1.6408 end=-1.6399   kick_direction: start=-1.6402 end=-1.6392
2026-08-08 19:21:49,353 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0009≈0.05°/step  epoch≈3.3°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0011≈0.06°/step  epoch≈3.6°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 19:21:49,353 INFO   [exec discrete Δlogit per opt step] exec_move=0.0014  sprint=0.0016  kick=0.0020  tackle_attempt=0.0018
2026-08-08 19:21:49,353 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0067  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0039  sprint=+0.0063  kick=+0.0020  tackle_attempt=+0.0014  move_dir=+0.0311  kick_dir=+0.0120
2026-08-08 19:21:49,354 INFO   [grad clip] main: 9/60 steps clipped (15%)  pre-clip norm mean=0.322 max=0.500  limit=0.4
              direction: 57/60 steps clipped (95%)  pre-clip norm mean=0.036 max=0.073  limit=0.02
2026-08-08 19:21:49,402 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=468,000  speed=877/s  reward=4.85
  loss     policy=0.0199  value=0.6387(x0.5)=0.3193
           entropy=3.9886  kl=0.0633
  value    V=3.21±1.09  R=3.16±1.77  adv=-0.05±1.41
  moves    mv_ls=[-1.6399] (σ≈0.19, ≈11°) g=1.07e-02  d_move=[+0.0010] (Δσ≈0.011°)
           kk_ls=[-1.6392] (σ≈0.19, ≈11°)  d_kick=[+0.0010] (Δσ≈0.011°)
  heads    move= 40 get_poss= 64 exec_move= 83 sprint= 49 kick=  8 tackle= 15 shoot=
           9 hold= 10 tackle_prob=0.1632 kick_prob=0.0827
  vs       vs[win/loss/tout/miss]  vs_immobile(664): 67.9%/0.2%/0.2%/14.5%/17%
  ep_len   16.1±9.2s  (n=664, min=1.4s, max=50.0s)
  reward   get_possession=+573.00  lose_possession=-5.40  ball_out=-212.00  box_possession=+1127.50
           speed_bonus=+1006.11  opponent_box=-3.00  timeout=-1.50  stamina_penalty=-3.86
  rew/ep   (mean/std/min/max per episode, 664 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.863    0.369    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.008    0.085    -0.900    +0.000
  ball_out          -0.319    1.084    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.698    1.167    +0.000    +2.500
  speed_bonus       +1.515    1.416    +0.000    +4.373
  opponent_box      -0.005    0.116    -3.000    +0.000
  timeout           -0.002    0.058    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.028    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     575    +0.016    0.125     +4.035     1.671     +0.764      3.3673      1.459     3.507
  lose_possession       6    -0.000    0.012     +3.114     0.408     -0.486      0.8522      0.817     1.326
  ball_out            53    -0.006    0.153     -3.962     0.191     -5.226     29.1892      5.226     7.380
  box_possession     451    +0.031    0.278     +4.725     1.158     +1.183      2.5442      1.364     2.750
  speed_bonus        431    +0.028    0.280     +4.828     1.078     +1.265      2.6327      1.394     2.769
  opponent_box         1    -0.000    0.016     -3.005     0.000     -6.085     37.0260      6.085     6.085
  timeout              1    -0.000    0.008     -1.500     0.000     -5.955     35.4572      5.955     5.955
  stamina_penalty     424    -0.000    0.001     +4.771     1.203     +1.203      2.7093      1.404     2.786
  gae/td   mean_return=+3.161  std_return=1.766  mean_gae=-0.052  mean_sq_td=1.9916
──────────────────────────────────────────────────────────────────────
2026-08-08 19:21:49,428 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint11.pt
2026-08-08 19:21:49,429 INFO Logging to checkpoints/phase1_run45/training_log12.txt
2026-08-08 19:21:49,430 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 19:22:01,889 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 19:22:01,891 INFO   [eval vs immobile] step=468,000  seeds=16x8  win=56%  mean_rew=3.142±3.058  V=3.043  gap=-0.098  outcomes={'other': 30, 'box_possession': 72, 'miss': 26}
2026-08-08 19:22:01,892 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 19:22:11,279 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 19:22:11,281 INFO   [eval vs rules] step=468,000  seeds=16x8  win=16%  mean_rew=-0.592±3.336  V=2.181  gap=+2.772  outcomes={'other': 26, 'opponent_box_possession': 73, 'box_possession': 21, 'miss': 8}
2026-08-08 19:26:58,851 INFO   [KL mean=0.0681 median=0.0686 > 0.05] ratio percentiles:  p5=0.625  p25=0.873  p50=0.962  p75=1.045  p95=1.306  max=8.335
  move_dir_log_std=[-1.6389325857162476]  kick_dir_log_std=[-1.6382471323013306]
2026-08-08 19:26:58,862 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.473  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.320  kick=-0.159  t_att=-0.493
    move_dir=0.613 (min=-14.711 max=1.440)  kick_dir=0.049 (min=-2.036 max=2.097)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.50
  [worst sample] idx=122  ratio=9.724  adv=-1.179  old_lp=-2.471  new_lp=-0.196
    stored move_dir=144.9°  new_mean=142.8°  angular_diff=2.1°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 122  ratio=   9.724  adv=-1.179  lp: old=-2.471  new=-0.196
      rew=+0.0000  ret=+2.9685  val=+4.1471  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9020  sprint_p_new=0.8965  kick_p_new=0.0104  tackle_attempt_p_new=0.0972
    idx= 131  ratio=   9.368  adv=-0.807  lp: old=-6.828  new=-4.590
      rew=+0.0000  ret=+0.3917  val=+1.1992  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: kick:+0.101  tackle_attempt:+0.063
      saturation: exec_move_p_new=0.8692  sprint_p_new=0.8586  kick_p_new=0.0633  tackle_attempt_p_new=0.1685
  [best sample (highest new_lp)] idx=122  new_lp=-0.196  adv=-1.179  stored move_dir=144.9°  new_mean=142.8°
    per-head contributions: move:-0.084  tackle_attempt:-0.102
2026-08-08 19:26:58,862 INFO   [advantage] mean=-0.000  std=1.000  min=-5.588  max=3.492
2026-08-08 19:26:58,863 INFO   [ratio] mean=0.9652  std=0.2315  min=0.0063  max=8.3346  clipped=33.2%
2026-08-08 19:26:58,863 INFO   [exec head grad norm] move_direction=0.029  exec_move=0.044  sprint=0.044  kick=0.045  kick_direction=0.012  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.062
2026-08-08 19:26:58,863 INFO   [exec continuous log_std] move_direction: start=-1.6399 end=-1.6389   kick_direction: start=-1.6392 end=-1.6382
2026-08-08 19:26:58,864 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0012≈0.07°/step  epoch≈4.1°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0014≈0.08°/step  epoch≈4.7°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 19:26:58,864 INFO   [exec discrete Δlogit per opt step] exec_move=0.0011  sprint=0.0020  kick=0.0019  tackle_attempt=0.0015
2026-08-08 19:26:58,864 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0060  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0035  sprint=+0.0067  kick=+0.0023  tackle_attempt=+0.0013  move_dir=+0.0348  kick_dir=+0.0134
2026-08-08 19:26:58,864 INFO   [grad clip] main: 8/60 steps clipped (13%)  pre-clip norm mean=0.331 max=0.476  limit=0.4
              direction: 56/60 steps clipped (93%)  pre-clip norm mean=0.034 max=0.077  limit=0.02
2026-08-08 19:26:58,918 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=504,000  speed=873/s  reward=3.88
  loss     policy=0.0200  value=0.6154(x0.5)=0.3077
           entropy=4.2801  kl=0.0681
  value    V=3.15±1.05  R=3.07±1.75  adv=-0.08±1.38
  moves    mv_ls=[-1.6389] (σ≈0.19, ≈11°) g=9.84e-03  d_move=[+0.0009] (Δσ≈0.010°)
           kk_ls=[-1.6382] (σ≈0.19, ≈11°)  d_kick=[+0.0010] (Δσ≈0.011°)
  heads    move= 42 get_poss= 63 exec_move= 83 sprint= 48 kick=
           9 tackle= 17 shoot= 11 hold= 11 tackle_prob=0.1748 kick_prob=0.0893
  vs       vs[win/loss/tout/miss]  vs_immobile(640): 65.9%/0.0%/0.9%/16.6%/17%
  ep_len   16.8±9.7s  (n=640, min=1.4s, max=50.0s)
  reward   get_possession=+550.00  lose_possession=-8.10  ball_out=-192.00  box_possession=+1055.00
           speed_bonus=+937.72  timeout=-9.00  stamina_penalty=-3.64
  rew/ep   (mean/std/min/max per episode, 640 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.859    0.390    +0.000    +3.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.013    0.117    -1.800    +0.000
  ball_out          -0.300    1.054    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.648    1.185    +0.000    +2.500
  speed_bonus       +1.465    1.428    +0.000    +4.266
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.014    0.145    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.024    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     552    +0.015    0.123     +4.010     1.600     +0.676      2.9054      1.337     3.328
  lose_possession       9    -0.000    0.014     +3.239     0.509     -0.570      0.5348      0.570     1.338
  ball_out            48    -0.005    0.146     -3.958     0.200     -5.200     28.8459      5.200     7.662
  box_possession     422    +0.029    0.269     +4.718     1.184     +1.191      2.5386      1.358     2.696
  speed_bonus        402    +0.026    0.272     +4.826     1.106     +1.274      2.6169      1.385     2.696
  timeout              6    -0.000    0.019     -1.506     0.004     -4.008     19.0526      4.008     5.911
  stamina_penalty     392    -0.000    0.001     +4.784     1.292     +1.216      2.9225      1.459     2.745
  gae/td   mean_return=+3.073  std_return=1.753  mean_gae=-0.076  mean_sq_td=1.9024
──────────────────────────────────────────────────────────────────────
2026-08-08 19:26:58,945 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint12.pt
2026-08-08 19:26:58,945 INFO Logging to checkpoints/phase1_run45/training_log13.txt
2026-08-08 19:26:58,947 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 19:27:10,643 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 19:27:10,645 INFO   [eval vs immobile] step=504,000  seeds=16x8  win=54%  mean_rew=2.822±3.119  V=2.964  gap=+0.141  outcomes={'other': 28, 'opponent_box_possession': 1, 'box_possession': 69, 'miss': 30}
2026-08-08 19:27:10,646 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 19:27:20,218 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 19:27:20,219 INFO   [eval vs rules] step=504,000  seeds=16x8  win=15%  mean_rew=-0.837±3.259  V=2.098  gap=+2.935  outcomes={'opponent_box_possession': 78, 'other': 22, 'box_possession': 19, 'miss': 9}
2026-08-08 19:32:02,146 INFO   [KL mean=0.0761 median=0.0767 > 0.05] ratio percentiles:  p5=0.601  p25=0.858  p50=0.958  p75=1.049  p95=1.322  max=6.660
  move_dir_log_std=[-1.637978434562683]  kick_dir_log_std=[-1.6370940208435059]
2026-08-08 19:32:02,160 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.591  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.306  kick=-0.287  t_att=-0.407
    move_dir=0.736 (min=-5.464 max=1.438)  kick_dir=0.088 (min=-2.132 max=2.133)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.45
  [worst sample] idx=51  ratio=10.394  adv=-0.399  old_lp=-4.707  new_lp=-2.366
    stored move_dir=-154.6°  new_mean=-150.1°  angular_diff=4.5°
    [worst sample per-head delta, sorted by |delta|] move:+0.068
  [top-2 highest-ratio samples]
    idx=  51  ratio=  10.394  adv=-0.399  lp: old=-4.707  new=-2.366
      rew=+0.0000  ret=+3.5022  val=+3.9014  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: move:+0.068
      saturation: exec_move_p_new=0.8997  sprint_p_new=0.8997  kick_p_new=0.0147  tackle_attempt_p_new=0.1198
    idx=   0  ratio=   9.034  adv=-0.547  lp: old=-2.695  new=-0.494
      rew=+0.0000  ret=+3.1108  val=+3.6574  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: move:-0.062
      saturation: exec_move_p_new=0.8935  sprint_p_new=0.8910  kick_p_new=0.0220  tackle_attempt_p_new=0.1430
  [best sample (highest new_lp)] idx=130  new_lp=-0.307  adv=+2.557  stored move_dir=-54.8°  new_mean=-50.7°
    per-head contributions: kick:-0.028  move:-0.122  tackle_attempt:-0.157
2026-08-08 19:32:02,161 INFO   [advantage] mean=0.000  std=1.000  min=-5.511  max=3.744
2026-08-08 19:32:02,163 INFO   [ratio] mean=0.9615  std=0.2451  min=0.0050  max=6.6597  clipped=36.5%
2026-08-08 19:32:02,163 INFO   [exec head grad norm] move_direction=0.030  exec_move=0.045  sprint=0.041  kick=0.043  kick_direction=0.008  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.065
2026-08-08 19:32:02,163 INFO   [exec continuous log_std] move_direction: start=-1.6389 end=-1.6380   kick_direction: start=-1.6382 end=-1.6371
2026-08-08 19:32:02,164 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0011≈0.07°/step  epoch≈3.9°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0012≈0.07°/step  epoch≈4.2°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 19:32:02,164 INFO   [exec discrete Δlogit per opt step] exec_move=0.0010  sprint=0.0019  kick=0.0020  tackle_attempt=0.0016
2026-08-08 19:32:02,164 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0074  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0041  sprint=+0.0074  kick=+0.0026  tackle_attempt=+0.0020  move_dir=+0.0390  kick_dir=+0.0135
2026-08-08 19:32:02,165 INFO   [grad clip] main: 8/60 steps clipped (13%)  pre-clip norm mean=0.334 max=0.545  limit=0.4
              direction: 55/60 steps clipped (92%)  pre-clip norm mean=0.034 max=0.084  limit=0.02
2026-08-08 19:32:02,213 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=540,000  speed=875/s  reward=3.66
  loss     policy=0.0216  value=0.6265(x0.5)=0.3132
           entropy=4.6029  kl=0.0761
  value    V=3.08±1.00  R=3.08±1.64  adv=0.00±1.30
  moves    mv_ls=[-1.6380] (σ≈0.19, ≈11°) g=9.98e-03  d_move=[+0.0010] (Δσ≈0.011°)
           kk_ls=[-1.6371] (σ≈0.19, ≈11°)  d_kick=[+0.0012] (Δσ≈0.013°)
  heads    move= 42 get_poss= 62 exec_move= 83 sprint= 50 kick=
           9 tackle= 18 shoot= 13 hold= 14 tackle_prob=0.1852 kick_prob=0.0939
  vs       vs[win/loss/tout/miss]  vs_immobile(634): 66.9%/0.0%/0.6%/15.8%/17%
  ep_len   16.9±10.0s  (n=634, min=1.3s, max=50.0s)
  reward   get_possession=+525.00  lose_possession=-2.70  ball_out=-144.00  box_possession=+1060.00
           speed_bonus=+899.81  timeout=-6.00  stamina_penalty=-3.86
  rew/ep   (mean/std/min/max per episode, 634 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.828    0.390    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.004    0.062    -0.900    +0.000
  ball_out          -0.227    0.926    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.672    1.177    +0.000    +2.500
  speed_bonus       +1.419    1.404    +0.000    +4.205
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.009    0.119    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.024    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     529    +0.015    0.120     +4.066     1.455     +0.805      2.6501      1.298     3.166
  lose_possession       3    -0.000    0.008     +2.548     0.460     -1.136      1.4822      1.136     1.663
  ball_out            36    -0.004    0.126     -3.944     0.229     -5.154     28.0664      5.154     6.949
  box_possession     424    +0.029    0.270     +4.618     1.208     +1.173      2.6539      1.355     2.921
  speed_bonus        403    +0.025    0.264     +4.729     1.135     +1.269      2.7578      1.390     2.929
  timeout              4    -0.000    0.016     -1.508     0.005     -5.155     26.6097      5.155     5.433
  stamina_penalty     409    -0.000    0.001     +4.638     1.291     +1.182      2.9182      1.415     2.948
  gae/td   mean_return=+3.080  std_return=1.642  mean_gae=+0.001  mean_sq_td=1.7005
──────────────────────────────────────────────────────────────────────
2026-08-08 19:32:02,237 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint13.pt
2026-08-08 19:32:02,238 INFO Logging to checkpoints/phase1_run45/training_log14.txt
2026-08-08 19:32:02,239 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 19:32:15,096 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 19:32:15,097 INFO   [eval vs immobile] step=540,000  seeds=16x8  win=46%  mean_rew=2.373±3.032  V=2.965  gap=+0.592  outcomes={'other': 35, 'box_possession': 59, 'miss': 32, 'timeout': 2}
2026-08-08 19:32:15,099 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 19:32:24,526 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 19:32:24,528 INFO   [eval vs rules] step=540,000  seeds=16x8  win=17%  mean_rew=-0.371±3.306  V=2.175  gap=+2.546  outcomes={'other': 30, 'opponent_box_possession': 66, 'box_possession': 22, 'miss': 10}
2026-08-08 19:37:04,987 INFO   [KL mean=0.0836 median=0.0836 > 0.05] ratio percentiles:  p5=0.582  p25=0.850  p50=0.958  p75=1.049  p95=1.316  max=8.894
  move_dir_log_std=[-1.6368499994277954]  kick_dir_log_std=[-1.6359630823135376]
2026-08-08 19:37:05,004 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.555  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.333  kick=-0.285  t_att=-0.424
    move_dir=0.771 (min=-6.257 max=1.436)  kick_dir=0.137 (min=-0.466 max=2.150)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.41
  [worst sample] idx=197  ratio=9.942  adv=-0.409  old_lp=-8.329  new_lp=-6.032
    stored move_dir=-131.6°  new_mean=-137.9°  angular_diff=6.3°
    [worst sample per-head delta, sorted by |delta|] kick:+0.042  tackle_attempt:+0.094
  [top-2 highest-ratio samples]
    idx= 197  ratio=   9.942  adv=-0.409  lp: old=-8.329  new=-6.032
      rew=+0.0000  ret=+2.9574  val=+3.3668  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: tackle_attempt:+0.094  kick:+0.042
      saturation: exec_move_p_new=0.8855  sprint_p_new=0.8706  kick_p_new=0.0235  tackle_attempt_p_new=0.1601
    idx=  35  ratio=   9.037  adv=+0.261  lp: old=-5.214  new=-3.013
      rew=+0.0000  ret=+4.3321  val=+4.0711  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: tackle_attempt:+0.086  move:+0.033
      saturation: exec_move_p_new=0.8870  sprint_p_new=0.8756  kick_p_new=0.0242  tackle_attempt_p_new=0.1405
  [best sample (highest new_lp)] idx=18  new_lp=-0.276  adv=+0.070  stored move_dir=-8.3°  new_mean=-16.2°
    per-head contributions: move_dir:0.059  move:-0.091  sprint:-0.101  tackle_attempt:-0.130
2026-08-08 19:37:05,004 INFO   [advantage] mean=0.000  std=1.000  min=-5.810  max=3.554
2026-08-08 19:37:05,005 INFO   [ratio] mean=0.9557  std=0.2454  min=0.0068  max=8.8936  clipped=37.5%
2026-08-08 19:37:05,005 INFO   [exec head grad norm] move_direction=0.025  exec_move=0.050  sprint=0.044  kick=0.046  kick_direction=0.010  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.074
2026-08-08 19:37:05,005 INFO   [exec continuous log_std] move_direction: start=-1.6380 end=-1.6368   kick_direction: start=-1.6371 end=-1.6360
2026-08-08 19:37:05,005 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0013≈0.07°/step  epoch≈4.5°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0013≈0.07°/step  epoch≈4.5°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 19:37:05,005 INFO   [exec discrete Δlogit per opt step] exec_move=0.0011  sprint=0.0019  kick=0.0022  tackle_attempt=0.0018
2026-08-08 19:37:05,006 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0081  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0041  sprint=+0.0065  kick=+0.0022  tackle_attempt=+0.0024  move_dir=+0.0453  kick_dir=+0.0150
2026-08-08 19:37:05,006 INFO   [grad clip] main: 8/60 steps clipped (13%)  pre-clip norm mean=0.333 max=0.526  limit=0.4
              direction: 53/60 steps clipped (88%)  pre-clip norm mean=0.030 max=0.055  limit=0.02
2026-08-08 19:37:05,054 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=576,000  speed=857/s  reward=3.38
  loss     policy=0.0208  value=0.6248(x0.5)=0.3124
           entropy=4.9033  kl=0.0836
  value    V=3.11±1.00  R=3.07±1.61  adv=-0.04±1.28
  moves    mv_ls=[-1.6368] (σ≈0.19, ≈11°) g=1.17e-02  d_move=[+0.0011] (Δσ≈0.013°)
           kk_ls=[-1.6360] (σ≈0.19, ≈11°)  d_kick=[+0.0011] (Δσ≈0.013°)
  heads    move= 43 get_poss= 61 exec_move= 82 sprint= 49 kick= 10 tackle= 18 shoot= 16 hold= 16 tackle_prob=0.1996 kick_prob=0.1011
  vs       vs[win/loss/tout/miss]  vs_immobile(611): 68.2%/0.0%/1.0%/15.5%/15%
  ep_len   17.5±10.5s  (n=611, min=0.8s, max=50.0s)
  reward   get_possession=+525.00  lose_possession=-6.30  ball_out=-176.00  box_possession=+1042.50
           speed_bonus=+858.86  timeout=-9.00  stamina_penalty=-3.87
  rew/ep   (mean/std/min/max per episode, 611 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.859    0.379    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.010    0.096    -0.900    +0.000
  ball_out          -0.288    1.034    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.706    1.164    +0.000    +2.500
  speed_bonus       +1.406    1.325    +0.000    +4.163
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.015    0.148    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.007    -0.027    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     529    +0.015    0.120     +3.880     1.667     +0.529      3.0558      1.297     3.774
  lose_possession       7    -0.000    0.013     +3.170     0.489     -0.548      0.5757      0.641     1.166
  ball_out            44    -0.005    0.140     -3.955     0.208     -5.551     32.2863      5.551     7.183
  box_possession     417    +0.029    0.268     +4.553     1.102     +1.208      2.5462      1.333     2.772
  speed_bonus        400    +0.024    0.251     +4.640     1.038     +1.285      2.6328      1.362     2.777
  timeout              6    -0.000    0.019     -1.506     0.004     -4.306     19.7241      4.306     5.331
  stamina_penalty     402    -0.000    0.001     +4.539     1.250     +1.189      2.8117      1.397     2.856
  gae/td   mean_return=+3.068  std_return=1.605  mean_gae=-0.040  mean_sq_td=1.6297
──────────────────────────────────────────────────────────────────────
2026-08-08 19:37:05,078 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint14.pt
2026-08-08 19:37:05,079 INFO Logging to checkpoints/phase1_run45/training_log15.txt
2026-08-08 19:37:05,080 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 19:37:18,968 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 19:37:18,969 INFO   [eval vs immobile] step=576,000  seeds=16x8  win=49%  mean_rew=2.432±3.115  V=2.892  gap=+0.460  outcomes={'other': 30, 'miss': 34, 'box_possession': 63, 'timeout': 1}
2026-08-08 19:37:18,971 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 19:37:28,286 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 19:37:28,288 INFO   [eval vs rules] step=576,000  seeds=16x8  win=13%  mean_rew=-1.010±2.963  V=2.164  gap=+3.173  outcomes={'other': 24, 'opponent_box_possession': 77, 'box_possession': 17, 'miss': 10}
2026-08-08 19:42:13,747 INFO   [KL mean=0.0913 median=0.0908 > 0.05] ratio percentiles:  p5=0.551  p25=0.843  p50=0.958  p75=1.054  p95=1.331  max=11.426
  move_dir_log_std=[-1.6359139680862427]  kick_dir_log_std=[-1.634881854057312]
2026-08-08 19:42:13,759 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.625  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.351  kick=-0.217  t_att=-0.511
    move_dir=0.556 (min=-50.753 max=1.434)  kick_dir=0.073 (min=-0.228 max=2.115)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.47
  [worst sample] idx=72  ratio=10.689  adv=+0.540  old_lp=-5.448  new_lp=-3.078
    stored move_dir=165.8°  new_mean=160.4°  angular_diff=5.4°
    [worst sample per-head delta, sorted by |delta|] move:+0.247  tackle_attempt:+0.058
  [top-2 highest-ratio samples]
    idx=  72  ratio=  10.689  adv=+0.540  lp: old=-5.448  new=-3.078
      rew=+0.0000  ret=+4.4718  val=+3.9318  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: move:+0.247  tackle_attempt:+0.058
      saturation: exec_move_p_new=0.8744  sprint_p_new=0.8477  kick_p_new=0.0183  tackle_attempt_p_new=0.1530
    idx= 220  ratio=  10.242  adv=-2.984  lp: old=-4.566  new=-2.240
      rew=+0.0000  ret=+0.3685  val=+3.3530  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: tackle_attempt:+0.068  move:-0.021
      saturation: exec_move_p_new=0.8988  sprint_p_new=0.8734  kick_p_new=0.0107  tackle_attempt_p_new=0.1272
  [best sample (highest new_lp)] idx=218  new_lp=-0.416  adv=-2.822  stored move_dir=164.1°  new_mean=167.5°
    per-head contributions: move_dir:0.069  tackle_attempt:-0.141  sprint:-0.143  move:-0.190
2026-08-08 19:42:13,759 INFO   [advantage] mean=0.000  std=1.000  min=-5.451  max=3.487
2026-08-08 19:42:13,760 INFO   [ratio] mean=0.9541  std=0.2625  min=0.0027  max=11.4258  clipped=39.5%
2026-08-08 19:42:13,760 INFO   [exec head grad norm] move_direction=0.030  exec_move=0.041  sprint=0.042  kick=0.048  kick_direction=0.010  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.054
2026-08-08 19:42:13,760 INFO   [exec continuous log_std] move_direction: start=-1.6368 end=-1.6359   kick_direction: start=-1.6360 end=-1.6349
2026-08-08 19:42:13,761 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0014≈0.08°/step  epoch≈4.9°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0015≈0.08°/step  epoch≈5.0°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 19:42:13,761 INFO   [exec discrete Δlogit per opt step] exec_move=0.0011  sprint=0.0020  kick=0.0021  tackle_attempt=0.0014
2026-08-08 19:42:13,761 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0069  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0038  sprint=+0.0086  kick=+0.0028  tackle_attempt=+0.0015  move_dir=+0.0510  kick_dir=+0.0168
2026-08-08 19:42:13,761 INFO   [grad clip] main: 6/60 steps clipped (10%)  pre-clip norm mean=0.323 max=0.456  limit=0.4
              direction: 58/60 steps clipped (97%)  pre-clip norm mean=0.034 max=0.097  limit=0.02
2026-08-08 19:42:13,810 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=612,000  speed=878/s  reward=3.32
  loss     policy=0.0227  value=0.6794(x0.5)=0.3397
           entropy=5.2150  kl=0.0913
  value    V=2.99±1.02  R=2.90±1.62  adv=-0.10±1.34
  moves    mv_ls=[-1.6359] (σ≈0.19, ≈11°) g=1.03e-02  d_move=[+0.0009] (Δσ≈0.010°)
           kk_ls=[-1.6349] (σ≈0.19, ≈11°)  d_kick=[+0.0011] (Δσ≈0.012°)
  heads    move= 45 get_poss= 61 exec_move= 82 sprint= 48 kick= 10 tackle= 20 shoot= 19 hold= 19 tackle_prob=0.2130 kick_prob=0.1086
  vs       vs[win/loss/tout/miss]  vs_immobile(583): 65.0%/0.0%/1.0%/16.1%/18%
  ep_len   18.4±10.7s  (n=583, min=0.2s, max=50.0s)
  reward   get_possession=+491.00  lose_possession=-4.50  ball_out=-176.00  box_possession=+947.50
           speed_bonus=+750.24  timeout=-9.00  stamina_penalty=-3.59
  rew/ep   (mean/std/min/max per episode, 583 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.842    0.387    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.008    0.083    -0.900    +0.000
  ball_out          -0.302    1.057    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.625    1.192    +0.000    +2.500
  speed_bonus       +1.287    1.348    +0.000    +4.247
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.015    0.151    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.007    -0.037    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     495    +0.014    0.116     +3.845     1.525     +0.634      2.7886      1.273     3.522
  lose_possession       5    -0.000    0.011     +2.450     0.670     -1.198      2.5526      1.359     2.559
  ball_out            44    -0.005    0.140     -3.955     0.208     -5.411     31.0310      5.411     7.006
  box_possession     379    +0.026    0.255     +4.473     1.189     +1.228      2.8025      1.400     2.867
  speed_bonus        362    +0.021    0.236     +4.566     1.135     +1.308      2.9169      1.441     2.878
  timeout              6    -0.000    0.019     -1.508     0.006     -4.631     22.6386      4.631     5.459
  stamina_penalty     364    -0.000    0.001     +4.436     1.371     +1.183      3.1547      1.474     2.934
  gae/td   mean_return=+2.895  std_return=1.616  mean_gae=-0.095  mean_sq_td=1.7997
──────────────────────────────────────────────────────────────────────
2026-08-08 19:42:13,835 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint15.pt
2026-08-08 19:42:13,835 INFO Logging to checkpoints/phase1_run45/training_log16.txt
2026-08-08 19:42:13,836 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 19:42:27,125 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 19:42:27,127 INFO   [eval vs immobile] step=612,000  seeds=16x8  win=52%  mean_rew=2.677±3.021  V=2.811  gap=+0.134  outcomes={'other': 31, 'box_possession': 67, 'miss': 29, 'timeout': 1}
2026-08-08 19:42:27,128 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 19:42:36,854 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 19:42:36,855 INFO   [eval vs rules] step=612,000  seeds=16x8  win=16%  mean_rew=-0.608±3.428  V=2.210  gap=+2.819  outcomes={'other': 26, 'opponent_box_possession': 74, 'box_possession': 20, 'miss': 8}
2026-08-08 19:47:21,604 INFO   [KL mean=0.0970 median=0.0973 > 0.05] ratio percentiles:  p5=0.545  p25=0.839  p50=0.958  p75=1.050  p95=1.328  max=14.657
  move_dir_log_std=[-1.6347147226333618]  kick_dir_log_std=[-1.633769154548645]
2026-08-08 19:47:21,618 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.671  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.403  kick=-0.320  t_att=-0.522
    move_dir=0.769 (min=-3.546 max=1.431)  kick_dir=0.121 (min=-1.539 max=2.038)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.49
  [worst sample] idx=189  ratio=10.337  adv=+0.903  old_lp=-4.324  new_lp=-1.988
    stored move_dir=132.3°  new_mean=144.1°  angular_diff=11.8°
    [worst sample per-head delta, sorted by |delta|] move:-0.060  tackle_attempt:+0.040
  [top-2 highest-ratio samples]
    idx= 189  ratio=  10.337  adv=+0.903  lp: old=-4.324  new=-1.988
      rew=+0.0000  ret=+3.2023  val=+2.2989  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: move:-0.060  tackle_attempt:+0.040
      saturation: exec_move_p_new=0.9056  sprint_p_new=0.3059  kick_p_new=0.1626  tackle_attempt_p_new=0.2691
    idx=  94  ratio=   9.977  adv=-2.046  lp: old=-3.644  new=-1.343
      rew=+0.0000  ret=+1.6774  val=+3.7230  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: move:+0.227
      saturation: exec_move_p_new=0.8717  sprint_p_new=0.8433  kick_p_new=0.0169  tackle_attempt_p_new=0.1472
  [best sample (highest new_lp)] idx=91  new_lp=-0.567  adv=-1.651  stored move_dir=-145.2°  new_mean=-131.5°
    per-head contributions: move_dir:0.034  kick:-0.026  sprint:-0.148  tackle_attempt:-0.172  move:-0.255
2026-08-08 19:47:21,619 INFO   [advantage] mean=0.000  std=1.000  min=-5.632  max=3.521
2026-08-08 19:47:21,619 INFO   [ratio] mean=0.9503  std=0.2605  min=0.0039  max=14.6573  clipped=39.4%
2026-08-08 19:47:21,620 INFO   [exec head grad norm] move_direction=0.022  exec_move=0.048  sprint=0.043  kick=0.041  kick_direction=0.010  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.063
2026-08-08 19:47:21,620 INFO   [exec continuous log_std] move_direction: start=-1.6359 end=-1.6347   kick_direction: start=-1.6349 end=-1.6338
2026-08-08 19:47:21,620 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0013≈0.08°/step  epoch≈4.5°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0014≈0.08°/step  epoch≈4.8°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 19:47:21,620 INFO   [exec discrete Δlogit per opt step] exec_move=0.0012  sprint=0.0021  kick=0.0021  tackle_attempt=0.0014
2026-08-08 19:47:21,620 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0080  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0036  sprint=+0.0088  kick=+0.0027  tackle_attempt=+0.0023  move_dir=+0.0541  kick_dir=+0.0175
2026-08-08 19:47:21,621 INFO   [grad clip] main: 4/60 steps clipped (7%)  pre-clip norm mean=0.319 max=0.469  limit=0.4
              direction: 52/60 steps clipped (87%)  pre-clip norm mean=0.028 max=0.048  limit=0.02
2026-08-08 19:47:21,672 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=648,000  speed=874/s  reward=2.05
  loss     policy=0.0247  value=0.6586(x0.5)=0.3293
           entropy=5.4762  kl=0.0970
  value    V=2.88±0.95  R=2.72±1.61  adv=-0.16±1.30
  moves    mv_ls=[-1.6347] (σ≈0.20, ≈11°) g=1.26e-02  d_move=[+0.0012] (Δσ≈0.013°)
           kk_ls=[-1.6338] (σ≈0.20, ≈11°)  d_kick=[+0.0011] (Δσ≈0.012°)
  heads    move= 46 get_poss= 60 exec_move= 82 sprint= 48 kick= 11 tackle= 21 shoot= 22 hold= 22 tackle_prob=0.2223 kick_prob=0.1159
  vs       vs[win/loss/tout/miss]  vs_immobile(565): 59.8%/0.2%/3.0%/17.5%/19%
  ep_len   19.0±11.9s  (n=565, min=1.0s, max=50.0s)
  reward   get_possession=+460.00  lose_possession=-5.40  ball_out=-164.00  box_possession=+845.00
           speed_bonus=+662.96  opponent_box=-3.00  timeout=-25.50  stamina_penalty=-3.49
  rew/ep   (mean/std/min/max per episode, 565 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.814    0.415    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.010    0.092    -0.900    +0.000
  ball_out          -0.290    1.038    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.496    1.226    +0.000    +2.500
  speed_bonus       +1.173    1.300    +0.000    +4.079
  opponent_box      -0.005    0.126    -3.000    +0.000
  timeout           -0.045    0.256    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.007    -0.028    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     462    +0.013    0.113     +3.737     1.464     +0.607      2.5540      1.209     3.025
  lose_possession       6    -0.000    0.012     +3.035     0.415     -0.679      0.5947      0.679     1.174
  ball_out            41    -0.005    0.135     -3.927     0.260     -5.190     28.1365      5.190     6.566
  box_possession     338    +0.023    0.241     +4.452     1.130     +1.310      2.8055      1.389     2.910
  speed_bonus        324    +0.018    0.219     +4.536     1.077     +1.379      2.9168      1.431     2.924
  opponent_box         1    -0.000    0.016     -3.002     0.000     -5.898     34.7908      5.898     5.898
  timeout             17    -0.001    0.033     -1.508     0.007     -4.048     18.1021      4.048     5.467
  stamina_penalty     340    -0.000    0.001     +4.263     1.609     +1.121      3.6106      1.540     3.202
  gae/td   mean_return=+2.719  std_return=1.606  mean_gae=-0.157  mean_sq_td=1.7177
──────────────────────────────────────────────────────────────────────
2026-08-08 19:47:21,697 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint16.pt
2026-08-08 19:47:21,698 INFO Logging to checkpoints/phase1_run45/training_log17.txt
2026-08-08 19:47:21,699 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 19:47:37,200 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 19:47:37,201 INFO   [eval vs immobile] step=648,000  seeds=16x8  win=52%  mean_rew=2.534±2.965  V=2.663  gap=+0.128  outcomes={'other': 21, 'box_possession': 67, 'timeout': 4, 'miss': 36}
2026-08-08 19:47:37,203 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 19:47:46,100 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 19:47:46,101 INFO   [eval vs rules] step=648,000  seeds=16x8  win=15%  mean_rew=-0.982±3.092  V=2.162  gap=+3.144  outcomes={'other': 20, 'opponent_box_possession': 78, 'box_possession': 19, 'miss': 11}
2026-08-08 19:52:31,498 INFO   [KL mean=0.1060 median=0.1057 > 0.05] ratio percentiles:  p5=0.531  p25=0.832  p50=0.962  p75=1.047  p95=1.313  max=7.514
  move_dir_log_std=[-1.6335912942886353]  kick_dir_log_std=[-1.6326904296875]
2026-08-08 19:52:31,512 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.653  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.375  kick=-0.277  t_att=-0.508
    move_dir=0.675 (min=-3.301 max=1.429)  kick_dir=0.080 (min=-4.986 max=2.103)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.43
  [worst sample] idx=4  ratio=9.653  adv=+0.510  old_lp=-4.685  new_lp=-2.418
    stored move_dir=18.6°  new_mean=18.1°  angular_diff=0.5°
    [worst sample per-head delta, sorted by |delta|] move:-0.059  tackle_attempt:+0.050
  [top-2 highest-ratio samples]
    idx=   4  ratio=   9.653  adv=+0.510  lp: old=-4.685  new=-2.418
      rew=+0.0000  ret=+4.1565  val=+3.6468  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: move:-0.059  tackle_attempt:+0.050
      saturation: exec_move_p_new=0.9020  sprint_p_new=0.8632  kick_p_new=0.0217  tackle_attempt_p_new=0.1574
    idx= 186  ratio=   9.343  adv=+0.129  lp: old=-4.949  new=-2.715
      rew=+0.0000  ret=+3.7594  val=+3.6303  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: move:+0.088  tackle_attempt:+0.052
      saturation: exec_move_p_new=0.8743  sprint_p_new=0.8145  kick_p_new=0.0157  tackle_attempt_p_new=0.1626
  [best sample (highest new_lp)] idx=175  new_lp=-0.440  adv=+0.000  stored move_dir=-177.4°  new_mean=-175.9°
    per-head contributions: move_dir:0.071  sprint:-0.151  tackle_attempt:-0.151  move:-0.198
2026-08-08 19:52:31,512 INFO   [advantage] mean=0.000  std=1.000  min=-5.206  max=3.629
2026-08-08 19:52:31,513 INFO   [ratio] mean=0.9440  std=0.2559  min=0.0044  max=7.5135  clipped=39.9%
2026-08-08 19:52:31,513 INFO   [exec head grad norm] move_direction=0.030  exec_move=0.044  sprint=0.051  kick=0.047  kick_direction=0.012  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.067
2026-08-08 19:52:31,513 INFO   [exec continuous log_std] move_direction: start=-1.6347 end=-1.6336   kick_direction: start=-1.6338 end=-1.6327
2026-08-08 19:52:31,514 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0013≈0.07°/step  epoch≈4.5°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0013≈0.08°/step  epoch≈4.6°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 19:52:31,514 INFO   [exec discrete Δlogit per opt step] exec_move=0.0013  sprint=0.0021  kick=0.0023  tackle_attempt=0.0015
2026-08-08 19:52:31,514 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0072  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0041  sprint=+0.0078  kick=+0.0025  tackle_attempt=+0.0023  move_dir=+0.0621  kick_dir=+0.0198
2026-08-08 19:52:31,514 INFO   [grad clip] main: 4/60 steps clipped (7%)  pre-clip norm mean=0.308 max=0.459  limit=0.4
              direction: 60/60 steps clipped (100%)  pre-clip norm mean=0.036 max=0.079  limit=0.02
2026-08-08 19:52:31,560 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=684,000  speed=874/s  reward=3.32
  loss     policy=0.0252  value=0.7331(x0.5)=0.3666
           entropy=5.7487  kl=0.1060
  value    V=2.72±0.92  R=2.44±1.64  adv=-0.28±1.40
  moves    mv_ls=[-1.6336] (σ≈0.20, ≈11°) g=1.39e-02  d_move=[+0.0011] (Δσ≈0.013°)
           kk_ls=[-1.6327] (σ≈0.20, ≈11°)  d_kick=[+0.0011] (Δσ≈0.012°)
  heads    move= 47 get_poss= 58 exec_move= 82 sprint= 46 kick= 12 tackle= 23 shoot= 25 hold= 25 tackle_prob=0.2349 kick_prob=0.1260
  vs       vs[win/loss/tout/miss]  vs_immobile(515): 54.8%/0.0%/4.9%/21.7%/19%
  ep_len   20.9±13.0s  (n=515, min=1.1s, max=50.0s)
  reward   get_possession=+433.00  lose_possession=-3.60  ball_out=-248.00  box_possession=+705.00
           speed_bonus=+550.70  timeout=-37.50  stamina_penalty=-2.98
  rew/ep   (mean/std/min/max per episode, 515 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.841    0.387    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.007    0.079    -0.900    +0.000
  ball_out          -0.482    1.302    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.369    1.244    +0.000    +2.500
  speed_bonus       +1.069    1.289    +0.000    +4.342
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.073    0.322    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.007    -0.035    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     437    +0.012    0.110     +3.440     1.603     +0.453      2.5719      1.192     3.481
  lose_possession       4    -0.000    0.009     +2.643     1.166     -0.301      1.5092      1.063     1.693
  ball_out            62    -0.007    0.166     -4.000     0.000     -5.197     28.5194      5.197     6.888
  box_possession     282    +0.020    0.220     +4.443     1.142     +1.442      3.2699      1.524     2.976
  speed_bonus        272    +0.015    0.200     +4.515     1.099     +1.497      3.3822      1.566     2.979
  timeout             25    -0.001    0.040     -1.510     0.008     -4.209     18.6856      4.209     5.045
  stamina_penalty     289    -0.000    0.001     +4.085     1.874     +1.089      4.5387      1.768     4.171
  gae/td   mean_return=+2.442  std_return=1.636  mean_gae=-0.282  mean_sq_td=2.0435
──────────────────────────────────────────────────────────────────────
2026-08-08 19:52:31,585 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint17.pt
2026-08-08 19:52:31,586 INFO Logging to checkpoints/phase1_run45/training_log18.txt
2026-08-08 19:52:31,587 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 19:52:45,216 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 19:52:45,218 INFO   [eval vs immobile] step=684,000  seeds=16x8  win=49%  mean_rew=2.381±3.116  V=2.470  gap=+0.088  outcomes={'other': 28, 'box_possession': 63, 'miss': 33, 'timeout': 4}
2026-08-08 19:52:45,219 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 19:52:53,907 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 19:52:53,909 INFO   [eval vs rules] step=684,000  seeds=16x8  win=20%  mean_rew=-0.380±3.515  V=2.093  gap=+2.473  outcomes={'other': 24, 'opponent_box_possession': 71, 'box_possession': 25, 'miss': 8}
2026-08-08 19:57:38,174 INFO   [KL mean=0.0951 median=0.0952 > 0.05] ratio percentiles:  p5=0.554  p25=0.839  p50=0.965  p75=1.049  p95=1.310  max=8.853
  move_dir_log_std=[-1.63259756565094]  kick_dir_log_std=[-1.6316800117492676]
2026-08-08 19:57:38,185 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.663  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.350  kick=-0.425  t_att=-0.530
    move_dir=0.671 (min=-4.329 max=1.427)  kick_dir=0.184 (min=-5.826 max=2.137)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.50
  [worst sample] idx=225  ratio=8.791  adv=-0.884  old_lp=-3.261  new_lp=-1.087
    stored move_dir=61.4°  new_mean=59.0°  angular_diff=2.4°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 225  ratio=   8.791  adv=-0.884  lp: old=-3.261  new=-1.087
      rew=+0.0000  ret=+2.2921  val=+3.1760  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8951  sprint_p_new=0.8661  kick_p_new=0.0291  tackle_attempt_p_new=0.1811
    idx=  81  ratio=   8.406  adv=-0.331  lp: old=-2.510  new=-0.381
      rew=+0.0000  ret=+2.5076  val=+2.8390  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: move:-0.043
      saturation: exec_move_p_new=0.8922  sprint_p_new=0.8715  kick_p_new=0.0134  tackle_attempt_p_new=0.1605
  [best sample (highest new_lp)] idx=78  new_lp=-0.326  adv=-0.388  stored move_dir=119.3°  new_mean=128.0°
    per-head contributions: move:-0.128  tackle_attempt:-0.181
2026-08-08 19:57:38,185 INFO   [advantage] mean=0.000  std=1.000  min=-5.591  max=3.675
2026-08-08 19:57:38,186 INFO   [ratio] mean=0.9497  std=0.2526  min=0.0043  max=8.8533  clipped=38.9%
2026-08-08 19:57:38,186 INFO   [exec head grad norm] move_direction=0.033  exec_move=0.043  sprint=0.046  kick=0.045  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.067
2026-08-08 19:57:38,186 INFO   [exec continuous log_std] move_direction: start=-1.6336 end=-1.6326   kick_direction: start=-1.6327 end=-1.6317
2026-08-08 19:57:38,186 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0014≈0.08°/step  epoch≈4.8°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0016≈0.09°/step  epoch≈5.4°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 19:57:38,186 INFO   [exec discrete Δlogit per opt step] exec_move=0.0011  sprint=0.0018  kick=0.0020  tackle_attempt=0.0015
2026-08-08 19:57:38,187 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0068  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0040  sprint=+0.0068  kick=+0.0029  tackle_attempt=+0.0021  move_dir=+0.0533  kick_dir=+0.0192
2026-08-08 19:57:38,187 INFO   [grad clip] main: 1/60 steps clipped (2%)  pre-clip norm mean=0.300 max=0.434  limit=0.4
              direction: 60/60 steps clipped (100%)  pre-clip norm mean=0.037 max=0.104  limit=0.02
2026-08-08 19:57:38,222 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=720,000  speed=869/s  reward=3.44
  loss     policy=0.0226  value=0.6875(x0.5)=0.3437
           entropy=5.9200  kl=0.0951
  value    V=2.45±0.83  R=2.39±1.58  adv=-0.05±1.32
  moves    mv_ls=[-1.6326] (σ≈0.20, ≈11°) g=1.28e-02  d_move=[+0.0010] (Δσ≈0.011°)
           kk_ls=[-1.6317] (σ≈0.20, ≈11°)  d_kick=[+0.0010] (Δσ≈0.011°)
  heads    move= 48 get_poss= 55 exec_move= 81 sprint= 45 kick= 13 tackle= 24 shoot= 28 hold= 27 tackle_prob=0.2492 kick_prob=0.1357
  vs       vs[win/loss/tout/miss]  vs_immobile(539): 59.4%/0.0%/2.4%/19.1%/19%
  ep_len   19.8±12.2s  (n=539, min=0.5s, max=50.0s)
  reward   get_possession=+446.00  lose_possession=-5.40  ball_out=-216.00  box_possession=+800.00
           speed_bonus=+609.62  timeout=-19.50  stamina_penalty=-2.96
  rew/ep   (mean/std/min/max per episode, 539 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.827    0.411    +0.000    +3.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.010    0.109    -1.800    +0.000
  ball_out          -0.401    1.201    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.484    1.228    +0.000    +2.500
  speed_bonus       +1.131    1.264    +0.000    +4.100
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.036    0.230    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.005    0.006    -0.026    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     452    +0.013    0.111     +3.330     1.605     +0.559      2.8332      1.282     3.807
  lose_possession       7    -0.000    0.013     +2.248     0.509     -0.941      1.1971      1.014     1.517
  ball_out            54    -0.006    0.155     -3.963     0.189     -5.173     28.0037      5.173     6.885
  box_possession     320    +0.022    0.235     +4.402     1.093     +1.700      4.0490      1.727     3.468
  speed_bonus        308    +0.017    0.207     +4.470     1.055     +1.748      4.1581      1.770     3.478
  timeout             13    -0.001    0.028     -1.508     0.009     -3.512     13.3506      3.512     4.614
  stamina_penalty     310    -0.000    0.001     +4.320     1.424     +1.625      4.4534      1.827     3.603
  gae/td   mean_return=+2.394  std_return=1.582  mean_gae=-0.054  mean_sq_td=1.7330
──────────────────────────────────────────────────────────────────────
2026-08-08 19:57:38,247 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint18.pt
2026-08-08 19:57:38,247 INFO Logging to checkpoints/phase1_run45/training_log19.txt
2026-08-08 19:57:38,248 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 19:57:53,140 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 19:57:53,142 INFO   [eval vs immobile] step=720,000  seeds=16x8  win=48%  mean_rew=2.458±2.853  V=2.382  gap=-0.076  outcomes={'other': 36, 'box_possession': 61, 'timeout': 6, 'miss': 25}
2026-08-08 19:57:53,143 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 19:58:03,057 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 19:58:03,058 INFO   [eval vs rules] step=720,000  seeds=16x8  win=12%  mean_rew=-1.070±2.831  V=1.950  gap=+3.020  outcomes={'other': 27, 'opponent_box_possession': 77, 'box_possession': 15, 'miss': 9}
2026-08-08 20:02:49,723 INFO   [KL mean=0.1009 median=0.1009 > 0.05] ratio percentiles:  p5=0.544  p25=0.840  p50=0.969  p75=1.045  p95=1.295  max=7.260
  move_dir_log_std=[-1.6315388679504395]  kick_dir_log_std=[-1.6304993629455566]
2026-08-08 20:02:49,733 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.673  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.394  kick=-0.305  t_att=-0.587
    move_dir=0.686 (min=-5.069 max=1.425)  kick_dir=0.105 (min=-4.676 max=2.113)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.49
  [worst sample] idx=169  ratio=9.642  adv=+0.441  old_lp=-3.273  new_lp=-1.007
    stored move_dir=2.2°  new_mean=-11.2°  angular_diff=13.4°
    [worst sample per-head delta, sorted by |delta|] move:+0.044
  [top-2 highest-ratio samples]
    idx= 169  ratio=   9.642  adv=+0.441  lp: old=-3.273  new=-1.007
      rew=+0.0000  ret=+3.2810  val=+2.8401  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: move:+0.044
      saturation: exec_move_p_new=0.8978  sprint_p_new=0.8374  kick_p_new=0.0257  tackle_attempt_p_new=0.1780
    idx=  31  ratio=   9.119  adv=-0.918  lp: old=-3.068  new=-0.858
      rew=+0.0000  ret=+2.1731  val=+3.0906  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: move:+0.062
      saturation: exec_move_p_new=0.8850  sprint_p_new=0.8391  kick_p_new=0.0299  tackle_attempt_p_new=0.1808
  [best sample (highest new_lp)] idx=187  new_lp=-0.741  adv=+0.604  stored move_dir=-2.0°  new_mean=11.9°
    per-head contributions: kick:-0.024  tackle_attempt:-0.204  move:-0.513
2026-08-08 20:02:49,734 INFO   [advantage] mean=-0.000  std=1.000  min=-5.666  max=4.659
2026-08-08 20:02:49,735 INFO   [ratio] mean=0.9465  std=0.2465  min=0.0055  max=7.2603  clipped=38.2%
2026-08-08 20:02:49,735 INFO   [exec head grad norm] move_direction=0.027  exec_move=0.056  sprint=0.047  kick=0.046  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.058
2026-08-08 20:02:49,735 INFO   [exec continuous log_std] move_direction: start=-1.6326 end=-1.6315   kick_direction: start=-1.6317 end=-1.6305
2026-08-08 20:02:49,735 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0017≈0.10°/step  epoch≈5.8°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0018≈0.10°/step  epoch≈6.0°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 20:02:49,735 INFO   [exec discrete Δlogit per opt step] exec_move=0.0014  sprint=0.0020  kick=0.0020  tackle_attempt=0.0013
2026-08-08 20:02:49,735 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0056  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0042  sprint=+0.0066  kick=+0.0034  tackle_attempt=+0.0018  move_dir=+0.0572  kick_dir=+0.0221
2026-08-08 20:02:49,736 INFO   [grad clip] main: 2/60 steps clipped (3%)  pre-clip norm mean=0.306 max=0.636  limit=0.4
              direction: 54/60 steps clipped (90%)  pre-clip norm mean=0.032 max=0.156  limit=0.02
2026-08-08 20:02:49,787 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=756,000  speed=877/s  reward=3.80
  loss     policy=0.0213  value=0.6769(x0.5)=0.3385
           entropy=6.0694  kl=0.1009
  value    V=2.44±0.84  R=2.47±1.52  adv=0.03±1.24
  moves    mv_ls=[-1.6315] (σ≈0.20, ≈11°) g=1.24e-02  d_move=[+0.0011] (Δσ≈0.012°)
           kk_ls=[-1.6305] (σ≈0.20, ≈11°)  d_kick=[+0.0012] (Δσ≈0.013°)
  heads    move= 49 get_poss= 54 exec_move= 81 sprint= 43 kick= 14 tackle= 26 shoot= 30 hold= 30 tackle_prob=0.2628 kick_prob=0.1482
  vs       vs[win/loss/tout/miss]  vs_immobile(532): 60.5%/0.2%/4.1%/16.4%/19%
  ep_len   20.2±12.9s  (n=532, min=1.5s, max=50.0s)
  reward   get_possession=+439.00  lose_possession=-4.50  ball_out=-144.00  box_possession=+805.00
           speed_bonus=+622.94  opponent_box=-3.00  timeout=-33.00  stamina_penalty=-2.77
  rew/ep   (mean/std/min/max per episode, 532 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.825    0.404    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.008    0.087    -0.900    +0.000
  ball_out          -0.271    1.005    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.513    1.222    +0.000    +2.500
  speed_bonus       +1.171    1.300    +0.000    +4.121
  opponent_box      -0.006    0.130    -3.000    +0.000
  timeout           -0.062    0.299    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.005    0.006    -0.027    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     444    +0.012    0.110     +3.453     1.523     +0.735      2.8273      1.308     3.678
  lose_possession       5    -0.000    0.011     +2.668     0.674     -0.483      0.5419      0.658     1.148
  ball_out            36    -0.004    0.126     -4.000     0.000     -4.954     25.7458      4.954     6.521
  box_possession     322    +0.022    0.235     +4.429     1.143     +1.544      3.6093      1.602     3.301
  speed_bonus        308    +0.017    0.212     +4.517     1.090     +1.626      3.7664      1.659     3.305
  opponent_box         1    -0.000    0.016     -3.006     0.000     -5.757     33.1481      5.757     5.757
  timeout             22    -0.001    0.037     -1.505     0.005     -4.048     17.2990      4.048     4.618
  stamina_penalty     315    -0.000    0.001     +4.228     1.701     +1.357      4.4575      1.792     3.521
  gae/td   mean_return=+2.468  std_return=1.520  mean_gae=+0.025  mean_sq_td=1.5423
──────────────────────────────────────────────────────────────────────
2026-08-08 20:02:49,813 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint19.pt
2026-08-08 20:02:49,814 INFO Logging to checkpoints/phase1_run45/training_log20.txt
2026-08-08 20:02:49,815 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 20:03:05,618 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 20:03:05,619 INFO   [eval vs immobile] step=756,000  seeds=16x8  win=48%  mean_rew=2.320±3.079  V=2.441  gap=+0.120  outcomes={'other': 26, 'box_possession': 62, 'miss': 32, 'timeout': 8}
2026-08-08 20:03:05,620 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 20:03:15,608 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 20:03:15,610 INFO   [eval vs rules] step=756,000  seeds=16x8  win=11%  mean_rew=-1.327±2.784  V=2.034  gap=+3.361  outcomes={'box_possession': 14, 'opponent_box_possession': 82, 'other': 20, 'miss': 12}
2026-08-08 20:08:02,634 INFO   [KL mean=0.1054 median=0.1057 > 0.05] ratio percentiles:  p5=0.531  p25=0.837  p50=0.968  p75=1.041  p95=1.301  max=8.751
  move_dir_log_std=[-1.6304242610931396]  kick_dir_log_std=[-1.6292219161987305]
2026-08-08 20:08:02,650 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.660  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.311  kick=-0.428  t_att=-0.575
    move_dir=0.722 (min=-4.619 max=1.423)  kick_dir=0.201 (min=-2.190 max=2.124)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.46
  [worst sample] idx=244  ratio=9.995  adv=+0.769  old_lp=-4.820  new_lp=-2.518
    stored move_dir=168.2°  new_mean=152.5°  angular_diff=15.7°
    [worst sample per-head delta, sorted by |delta|] move:+0.055  tackle_attempt:+0.028
  [top-2 highest-ratio samples]
    idx= 244  ratio=   9.995  adv=+0.769  lp: old=-4.820  new=-2.518
      rew=+0.0000  ret=+3.9391  val=+3.1698  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: move:+0.055  tackle_attempt:+0.028
      saturation: exec_move_p_new=0.8897  sprint_p_new=0.8109  kick_p_new=0.0140  tackle_attempt_p_new=0.1852
    idx=  22  ratio=   8.148  adv=-1.418  lp: old=-3.236  new=-1.139
      rew=+0.0000  ret=-0.0322  val=+1.3858  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: move:+0.033
      saturation: exec_move_p_new=0.8745  sprint_p_new=0.0943  kick_p_new=0.1815  tackle_attempt_p_new=0.3030
  [best sample (highest new_lp)] idx=50  new_lp=-0.696  adv=-2.532  stored move_dir=25.1°  new_mean=20.2°
    per-head contributions: kick:-0.095  tackle_attempt:-0.284  move:-0.317
2026-08-08 20:08:02,651 INFO   [advantage] mean=-0.000  std=1.000  min=-5.604  max=4.410
2026-08-08 20:08:02,651 INFO   [ratio] mean=0.9442  std=0.2558  min=0.0021  max=8.7511  clipped=38.2%
2026-08-08 20:08:02,652 INFO   [exec head grad norm] move_direction=0.027  exec_move=0.048  sprint=0.052  kick=0.044  kick_direction=0.011  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.055
2026-08-08 20:08:02,652 INFO   [exec continuous log_std] move_direction: start=-1.6315 end=-1.6304   kick_direction: start=-1.6305 end=-1.6292
2026-08-08 20:08:02,652 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0015≈0.09°/step  epoch≈5.2°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0017≈0.10°/step  epoch≈5.7°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 20:08:02,652 INFO   [exec discrete Δlogit per opt step] exec_move=0.0013  sprint=0.0022  kick=0.0021  tackle_attempt=0.0011
2026-08-08 20:08:02,652 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0047  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0050  sprint=+0.0077  kick=+0.0037  tackle_attempt=+0.0022  move_dir=+0.0593  kick_dir=+0.0227
2026-08-08 20:08:02,653 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.287 max=0.399  limit=0.4
              direction: 59/60 steps clipped (98%)  pre-clip norm mean=0.034 max=0.067  limit=0.02
2026-08-08 20:08:02,706 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=792,000  speed=874/s  reward=2.98
  loss     policy=0.0239  value=0.6667(x0.5)=0.3333
           entropy=6.1846  kl=0.1054
  value    V=2.44±0.86  R=2.43±1.51  adv=-0.01±1.24
  moves    mv_ls=[-1.6304] (σ≈0.20, ≈11°) g=1.46e-02  d_move=[+0.0011] (Δσ≈0.013°)
           kk_ls=[-1.6292] (σ≈0.20, ≈11°)  d_kick=[+0.0013] (Δσ≈0.014°)
  heads    move= 50 get_poss= 52 exec_move= 82 sprint= 44 kick= 15 tackle= 27 shoot= 32 hold= 33 tackle_prob=0.2759 kick_prob=0.1565
  vs       vs[win/loss/tout/miss]  vs_immobile(537): 58.5%/0.0%/2.6%/19.2%/20%
  ep_len   20.1±12.1s  (n=537, min=1.0s, max=50.0s)
  reward   get_possession=+441.00  lose_possession=-9.00  ball_out=-152.00  box_possession=+785.00
           speed_bonus=+575.58  timeout=-21.00  stamina_penalty=-2.91
  rew/ep   (mean/std/min/max per episode, 537 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.821    0.429    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.017    0.122    -0.900    +0.000
  ball_out          -0.283    1.026    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.462    1.232    +0.000    +2.500
  speed_bonus       +1.072    1.260    +0.000    +4.075
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.039    0.239    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.005    0.006    -0.030    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     444    +0.012    0.110     +3.388     1.518     +0.732      2.8290      1.334     3.284
  lose_possession      10    -0.000    0.015     +2.587     0.640     -0.319      0.7885      0.713     1.481
  ball_out            38    -0.004    0.130     -3.974     0.160     -5.196     28.2586      5.196     6.792
  box_possession     314    +0.022    0.232     +4.334     1.151     +1.283      2.8217      1.388     2.942
  speed_bonus        302    +0.016    0.201     +4.407     1.112     +1.346      2.9221      1.424     2.944
  timeout             14    -0.001    0.030     -1.505     0.008     -3.593     14.4526      3.593     5.038
  stamina_penalty     304    -0.000    0.001     +4.225     1.475     +1.191      3.1806      1.466     2.997
  gae/td   mean_return=+2.430  std_return=1.515  mean_gae=-0.005  mean_sq_td=1.5485
──────────────────────────────────────────────────────────────────────
2026-08-08 20:08:02,733 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint20.pt
2026-08-08 20:08:02,733 INFO Logging to checkpoints/phase1_run45/training_log21.txt
2026-08-08 20:08:02,734 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 20:08:17,887 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 20:08:17,889 INFO   [eval vs immobile] step=792,000  seeds=16x8  win=48%  mean_rew=2.432±2.888  V=2.382  gap=-0.050  outcomes={'other': 31, 'box_possession': 62, 'miss': 30, 'timeout': 5}
2026-08-08 20:08:17,891 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 20:08:27,409 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 20:08:27,411 INFO   [eval vs rules] step=792,000  seeds=16x8  win=16%  mean_rew=-0.853±3.169  V=1.955  gap=+2.808  outcomes={'other': 20, 'opponent_box_possession': 77, 'miss': 10, 'box_possession': 21}
2026-08-08 20:13:12,063 INFO   [KL mean=0.1084 median=0.1083 > 0.05] ratio percentiles:  p5=0.528  p25=0.843  p50=0.972  p75=1.038  p95=1.276  max=8.423
  move_dir_log_std=[-1.6292458772659302]  kick_dir_log_std=[-1.6280065774917603]
2026-08-08 20:13:12,074 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.696  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.365  kick=-0.404  t_att=-0.589
    move_dir=0.583 (min=-23.790 max=1.421)  kick_dir=0.164 (min=-3.727 max=2.101)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.47
  [worst sample] idx=33  ratio=11.599  adv=+1.378  old_lp=-5.035  new_lp=-2.584
    stored move_dir=171.8°  new_mean=156.3°  angular_diff=15.5°
    [worst sample per-head delta, sorted by |delta|] move:+0.153  tackle_attempt:+0.025
  [top-2 highest-ratio samples]
    idx=  33  ratio=  11.599  adv=+1.378  lp: old=-5.035  new=-2.584
      rew=+4.3332  ret=+4.3332  val=+2.9549  outcome=terminal:box_possession
      rew_breakdown: box=+2.500  spd=+1.838  stam=-0.005
      head_deltas: move:+0.153  tackle_attempt:+0.025
      saturation: exec_move_p_new=0.8946  sprint_p_new=0.8221  kick_p_new=0.0107  tackle_attempt_p_new=0.1797
    idx= 196  ratio=   9.151  adv=-0.925  lp: old=-3.179  new=-0.965
      rew=+0.0000  ret=+2.3064  val=+3.2315  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8904  sprint_p_new=0.8308  kick_p_new=0.0216  tackle_attempt_p_new=0.1866
  [best sample (highest new_lp)] idx=180  new_lp=-0.787  adv=-0.171  stored move_dir=-45.7°  new_mean=-32.1°
    per-head contributions: move_dir:0.035  kick:-0.021  sprint:-0.140  tackle_attempt:-0.194  move:-0.467
2026-08-08 20:13:12,075 INFO   [advantage] mean=0.000  std=1.000  min=-5.522  max=3.630
2026-08-08 20:13:12,076 INFO   [ratio] mean=0.9412  std=0.2445  min=0.0041  max=8.4229  clipped=36.8%
2026-08-08 20:13:12,076 INFO   [exec head grad norm] move_direction=0.026  exec_move=0.039  sprint=0.048  kick=0.043  kick_direction=0.011  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.052
2026-08-08 20:13:12,076 INFO   [exec continuous log_std] move_direction: start=-1.6304 end=-1.6292   kick_direction: start=-1.6292 end=-1.6280
2026-08-08 20:13:12,077 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0014≈0.08°/step  epoch≈4.6°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0014≈0.08°/step  epoch≈4.9°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 20:13:12,077 INFO   [exec discrete Δlogit per opt step] exec_move=0.0010  sprint=0.0014  kick=0.0020  tackle_attempt=0.0010
2026-08-08 20:13:12,077 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0050  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0031  sprint=+0.0061  kick=+0.0035  tackle_attempt=+0.0019  move_dir=+0.0639  kick_dir=+0.0249
2026-08-08 20:13:12,078 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.271 max=0.376  limit=0.4
              direction: 60/60 steps clipped (100%)  pre-clip norm mean=0.033 max=0.053  limit=0.02
2026-08-08 20:13:12,127 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=828,000  speed=874/s  reward=3.04
  loss     policy=0.0220  value=0.6849(x0.5)=0.3425
           entropy=6.3076  kl=0.1084
  value    V=2.46±0.84  R=2.39±1.54  adv=-0.06±1.27
  moves    mv_ls=[-1.6292] (σ≈0.20, ≈11°) g=1.57e-02  d_move=[+0.0012] (Δσ≈0.013°)
           kk_ls=[-1.6280] (σ≈0.20, ≈11°)  d_kick=[+0.0012] (Δσ≈0.014°)
  heads    move= 50 get_poss= 50 exec_move= 81 sprint= 44 kick= 16 tackle= 28 shoot= 35 hold= 35 tackle_prob=0.2825 kick_prob=0.1675
  vs       vs[win/loss/tout/miss]  vs_immobile(518): 60.4%/0.0%/4.2%/18.5%/17%
  ep_len   20.7±12.7s  (n=518, min=0.5s, max=50.0s)
  reward   get_possession=+435.00  lose_possession=-5.40  ball_out=-216.00  box_possession=+782.50
           speed_bonus=+528.42  timeout=-33.00  stamina_penalty=-2.80
  rew/ep   (mean/std/min/max per episode, 518 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.840    0.397    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.010    0.096    -0.900    +0.000
  ball_out          -0.417    1.222    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.511    1.223    +0.000    +2.500
  speed_bonus       +1.020    1.204    +0.000    +4.115
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.064    0.302    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.005    0.006    -0.035    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     438    +0.012    0.110     +3.263     1.683     +0.628      3.2542      1.426     4.006
  lose_possession       6    -0.000    0.012     +2.939     0.739     -0.327      0.6388      0.721     1.272
  ball_out            54    -0.006    0.155     -3.981     0.135     -4.805     24.2589      4.805     6.513
  box_possession     313    +0.022    0.232     +4.183     1.123     +1.204      2.6062      1.319     2.966
  speed_bonus        301    +0.015    0.189     +4.250     1.092     +1.264      2.6948      1.352     2.983
  timeout             22    -0.001    0.037     -1.505     0.007     -4.051     16.7301      4.051     4.821
  stamina_penalty     304    -0.000    0.001     +4.018     1.599     +1.060      3.3946      1.503     3.346
  gae/td   mean_return=+2.392  std_return=1.536  mean_gae=-0.064  mean_sq_td=1.6221
──────────────────────────────────────────────────────────────────────
2026-08-08 20:13:12,153 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint21.pt
2026-08-08 20:13:12,153 INFO Logging to checkpoints/phase1_run45/training_log22.txt
2026-08-08 20:13:12,154 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 20:13:27,308 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 20:13:27,309 INFO   [eval vs immobile] step=828,000  seeds=16x8  win=48%  mean_rew=2.437±2.965  V=2.316  gap=-0.121  outcomes={'other': 29, 'box_possession': 62, 'miss': 30, 'timeout': 7}
2026-08-08 20:13:27,311 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 20:13:36,647 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 20:13:36,649 INFO   [eval vs rules] step=828,000  seeds=16x8  win=10%  mean_rew=-1.230±2.835  V=1.962  gap=+3.192  outcomes={'other': 24, 'opponent_box_possession': 82, 'box_possession': 13, 'miss': 9}
2026-08-08 20:18:18,993 INFO   [KL mean=0.1112 median=0.1112 > 0.05] ratio percentiles:  p5=0.532  p25=0.840  p50=0.970  p75=1.034  p95=1.268  max=6.274
  move_dir_log_std=[-1.6282570362091064]  kick_dir_log_std=[-1.6267629861831665]
2026-08-08 20:18:19,008 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.684  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.383  kick=-0.375  t_att=-0.527
    move_dir=0.636 (min=-3.124 max=1.419)  kick_dir=0.085 (min=-4.115 max=2.119)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.50
  [worst sample] idx=89  ratio=10.654  adv=+0.536  old_lp=-3.207  new_lp=-0.841
    stored move_dir=64.1°  new_mean=66.0°  angular_diff=1.9°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx=  89  ratio=  10.654  adv=+0.536  lp: old=-3.207  new=-0.841
      rew=+0.0000  ret=+3.5142  val=+2.9787  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9072  sprint_p_new=0.8562  kick_p_new=0.0229  tackle_attempt_p_new=0.1997
    idx=  51  ratio=  10.231  adv=-0.186  lp: old=-3.374  new=-1.048
      rew=+0.0000  ret=+2.5720  val=+2.7577  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: move:+0.047
      saturation: exec_move_p_new=0.9004  sprint_p_new=0.8496  kick_p_new=0.0220  tackle_attempt_p_new=0.1725
  [best sample (highest new_lp)] idx=157  new_lp=-0.691  adv=+1.075  stored move_dir=-27.3°  new_mean=-13.9°
    per-head contributions: kick:-0.031  tackle_attempt:-0.265  move:-0.396
2026-08-08 20:18:19,009 INFO   [advantage] mean=-0.000  std=1.000  min=-5.760  max=3.515
2026-08-08 20:18:19,010 INFO   [ratio] mean=0.9387  std=0.2436  min=0.0023  max=6.2738  clipped=36.6%
2026-08-08 20:18:19,010 INFO   [exec head grad norm] move_direction=0.027  exec_move=0.042  sprint=0.047  kick=0.050  kick_direction=0.011  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.049
2026-08-08 20:18:19,010 INFO   [exec continuous log_std] move_direction: start=-1.6292 end=-1.6283   kick_direction: start=-1.6280 end=-1.6268
2026-08-08 20:18:19,010 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0016≈0.09°/step  epoch≈5.4°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0015≈0.09°/step  epoch≈5.2°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 20:18:19,010 INFO   [exec discrete Δlogit per opt step] exec_move=0.0011  sprint=0.0015  kick=0.0021  tackle_attempt=0.0009
2026-08-08 20:18:19,010 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0051  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0042  sprint=+0.0066  kick=+0.0041  tackle_attempt=+0.0025  move_dir=+0.0626  kick_dir=+0.0261
2026-08-08 20:18:19,011 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.256 max=0.359  limit=0.4
              direction: 58/60 steps clipped (97%)  pre-clip norm mean=0.033 max=0.062  limit=0.02
2026-08-08 20:18:19,071 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=864,000  speed=874/s  reward=3.32
  loss     policy=0.0201  value=0.6438(x0.5)=0.3219
           entropy=6.4125  kl=0.1112
  value    V=2.40±0.82  R=2.37±1.55  adv=-0.03±1.24
  moves    mv_ls=[-1.6283] (σ≈0.20, ≈11°) g=1.25e-02  d_move=[+0.0010] (Δσ≈0.011°)
           kk_ls=[-1.6268] (σ≈0.20, ≈11°)  d_kick=[+0.0012] (Δσ≈0.014°)
  heads    move= 52 get_poss= 48 exec_move= 81 sprint= 43 kick= 17 tackle= 28 shoot= 37 hold= 37 tackle_prob=0.2927 kick_prob=0.1816
  vs       vs[win/loss/tout/miss]  vs_immobile(497): 58.6%/0.2%/7.2%/18.7%/15%
  ep_len   21.6±13.4s  (n=497, min=0.2s, max=50.0s)
  reward   get_possession=+403.00  lose_possession=-0.90  ball_out=-172.00  box_possession=+727.50
           speed_bonus=+566.25  opponent_box=-3.00  timeout=-54.00  stamina_penalty=-2.91
  rew/ep   (mean/std/min/max per episode, 497 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.811    0.397    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.002    0.040    -0.900    +0.000
  ball_out          -0.346    1.125    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.464    1.232    +0.000    +2.500
  speed_bonus       +1.139    1.290    +0.000    +4.116
  opponent_box      -0.006    0.134    -3.000    +0.000
  timeout           -0.109    0.389    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.026    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     407    +0.011    0.106     +3.262     1.685     +0.656      2.9965      1.405     3.662
  lose_possession       2    -0.000    0.007     +2.877     0.502     -0.341      0.2679      0.390     0.696
  ball_out            43    -0.005    0.138     -4.000     0.000     -4.807     24.4492      4.807     6.959
  box_possession     291    +0.020    0.224     +4.440     1.123     +1.429      3.2698      1.508     3.206
  speed_bonus        286    +0.016    0.202     +4.474     1.102     +1.460      3.3233      1.528     3.208
  opponent_box         1    -0.000    0.016     -3.006     0.000     -6.307     39.7810      6.307     6.307
  timeout             36    -0.002    0.047     -1.504     0.006     -3.953     16.0887      3.953     4.847
  stamina_penalty     293    -0.000    0.001     +4.101     1.847     +1.100      4.3641      1.722     3.977
  gae/td   mean_return=+2.368  std_return=1.547  mean_gae=-0.031  mean_sq_td=1.5423
──────────────────────────────────────────────────────────────────────
2026-08-08 20:18:19,098 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint22.pt
2026-08-08 20:18:19,098 INFO Logging to checkpoints/phase1_run45/training_log23.txt
2026-08-08 20:18:19,099 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 20:18:34,272 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 20:18:34,273 INFO   [eval vs immobile] step=864,000  seeds=16x8  win=48%  mean_rew=2.473±2.745  V=2.272  gap=-0.201  outcomes={'other': 35, 'box_possession': 61, 'timeout': 4, 'miss': 28}
2026-08-08 20:18:34,282 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 20:18:45,174 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 20:18:45,176 INFO   [eval vs rules] step=864,000  seeds=16x8  win=12%  mean_rew=-1.286±2.910  V=1.927  gap=+3.212  outcomes={'opponent_box_possession': 87, 'other': 18, 'box_possession': 15, 'miss': 8}
2026-08-08 20:23:28,826 INFO   [KL mean=0.1048 median=0.1046 > 0.05] ratio percentiles:  p5=0.541  p25=0.844  p50=0.972  p75=1.036  p95=1.269  max=6.654
  move_dir_log_std=[-1.6273185014724731]  kick_dir_log_std=[-1.6256561279296875]
2026-08-08 20:23:28,837 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.695  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.362  kick=-0.444  t_att=-0.581
    move_dir=0.713 (min=-13.130 max=1.417)  kick_dir=0.171 (min=-2.736 max=2.119)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.46
  [worst sample] idx=103  ratio=8.613  adv=-2.058  old_lp=-2.910  new_lp=-0.757
    stored move_dir=0.4°  new_mean=6.1°  angular_diff=5.6°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 103  ratio=   8.613  adv=-2.058  lp: old=-2.910  new=-0.757
      rew=+0.0000  ret=+1.5526  val=+3.6106  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8904  sprint_p_new=0.7833  kick_p_new=0.0124  tackle_attempt_p_new=0.1946
    idx= 104  ratio=   7.542  adv=-2.052  lp: old=-2.824  new=-0.804
      rew=+0.0000  ret=+1.4988  val=+3.5505  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8736  sprint_p_new=0.7675  kick_p_new=0.0161  tackle_attempt_p_new=0.2026
  [best sample (highest new_lp)] idx=103  new_lp=-0.757  adv=-2.058  stored move_dir=0.4°  new_mean=6.1°
    per-head contributions: tackle_attempt:-0.216  move:-0.528
2026-08-08 20:23:28,837 INFO   [advantage] mean=0.000  std=1.000  min=-5.409  max=3.218
2026-08-08 20:23:28,838 INFO   [ratio] mean=0.9418  std=0.2394  min=0.0029  max=6.6537  clipped=36.0%
2026-08-08 20:23:28,839 INFO   [exec head grad norm] move_direction=0.031  exec_move=0.044  sprint=0.043  kick=0.043  kick_direction=0.012  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.067
2026-08-08 20:23:28,839 INFO   [exec continuous log_std] move_direction: start=-1.6283 end=-1.6273   kick_direction: start=-1.6268 end=-1.6257
2026-08-08 20:23:28,839 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0017≈0.10°/step  epoch≈6.0°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0018≈0.11°/step  epoch≈6.3°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 20:23:28,839 INFO   [exec discrete Δlogit per opt step] exec_move=0.0010  sprint=0.0014  kick=0.0019  tackle_attempt=0.0013
2026-08-08 20:23:28,839 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0041  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0036  sprint=+0.0062  kick=+0.0046  tackle_attempt=+0.0022  move_dir=+0.0597  kick_dir=+0.0244
2026-08-08 20:23:28,840 INFO   [grad clip] main: 2/60 steps clipped (3%)  pre-clip norm mean=0.274 max=0.411  limit=0.4
              direction: 60/60 steps clipped (100%)  pre-clip norm mean=0.038 max=0.093  limit=0.02
2026-08-08 20:23:28,876 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=900,000  speed=874/s  reward=2.46
  loss     policy=0.0211  value=0.6079(x0.5)=0.3039
           entropy=6.4728  kl=0.1048
  value    V=2.36±0.93  R=2.41±1.58  adv=0.05±1.22
  moves    mv_ls=[-1.6273] (σ≈0.20, ≈11°) g=1.45e-02  d_move=[+0.0009] (Δσ≈0.011°)
           kk_ls=[-1.6257] (σ≈0.20, ≈11°)  d_kick=[+0.0011] (Δσ≈0.012°)
  heads    move= 52 get_poss= 46 exec_move= 81 sprint= 44 kick= 18 tackle= 29 shoot= 38 hold= 39 tackle_prob=0.3010 kick_prob=0.1891
  vs       vs[win/loss/tout/miss]  vs_immobile(533): 59.1%/0.2%/4.9%/18.8%/17%
  ep_len   20.2±12.9s  (n=533, min=0.5s, max=50.0s)
  reward   get_possession=+426.00  lose_possession=-1.80  ball_out=-164.00  box_possession=+787.50
           speed_bonus=+608.23  opponent_box=-3.00  timeout=-39.00  stamina_penalty=-2.94
  rew/ep   (mean/std/min/max per episode, 533 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.799    0.405    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.003    0.055    -0.900    +0.000
  ball_out          -0.308    1.066    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.477    1.229    +0.000    +2.500
  speed_bonus       +1.141    1.288    +0.000    +4.037
  opponent_box      -0.006    0.130    -3.000    +0.000
  timeout           -0.073    0.323    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.027    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     430    +0.012    0.109     +3.311     1.784     +0.812      3.5078      1.556     3.828
  lose_possession       2    -0.000    0.007     +0.767     1.999     -2.314      9.8781      2.314     4.228
  ball_out            41    -0.005    0.135     -3.951     0.215     -4.388     20.0595      4.388     5.758
  box_possession     315    +0.022    0.233     +4.428     1.125     +1.236      2.6277      1.350     2.818
  speed_bonus        303    +0.017    0.209     +4.505     1.078     +1.296      2.7207      1.386     2.828
  opponent_box         1    -0.000    0.016     -3.001     0.000     -5.298     28.0662      5.298     5.298
  timeout             26    -0.001    0.040     -1.504     0.005     -3.808     15.2969      3.808     5.029
  stamina_penalty     308    -0.000    0.001     +4.239     1.650     +1.083      3.2270      1.484     3.163
  gae/td   mean_return=+2.412  std_return=1.577  mean_gae=+0.053  mean_sq_td=1.5034
──────────────────────────────────────────────────────────────────────
2026-08-08 20:23:28,898 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint23.pt
2026-08-08 20:23:28,899 INFO Logging to checkpoints/phase1_run45/training_log24.txt
2026-08-08 20:23:28,900 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 20:23:44,363 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 20:23:44,364 INFO   [eval vs immobile] step=900,000  seeds=16x8  win=47%  mean_rew=2.311±3.067  V=2.291  gap=-0.020  outcomes={'other': 26, 'box_possession': 60, 'timeout': 9, 'miss': 33}
2026-08-08 20:23:44,365 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 20:23:53,297 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 20:23:53,298 INFO   [eval vs rules] step=900,000  seeds=16x8  win=12%  mean_rew=-1.151±2.803  V=1.739  gap=+2.890  outcomes={'opponent_box_possession': 79, 'other': 24, 'box_possession': 15, 'miss': 10}
2026-08-08 20:28:37,816 INFO   [KL mean=0.1023 median=0.1024 > 0.05] ratio percentiles:  p5=0.553  p25=0.846  p50=0.971  p75=1.036  p95=1.275  max=7.128
  move_dir_log_std=[-1.6263428926467896]  kick_dir_log_std=[-1.6243501901626587]
2026-08-08 20:28:37,829 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.693  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.434  kick=-0.359  t_att=-0.617
    move_dir=0.763 (min=-3.466 max=1.415)  kick_dir=0.116 (min=-2.261 max=2.116)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.47
  [worst sample] idx=12  ratio=10.167  adv=+1.988  old_lp=-3.298  new_lp=-0.979
    stored move_dir=5.6°  new_mean=13.3°  angular_diff=7.8°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx=  12  ratio=  10.167  adv=+1.988  lp: old=-3.298  new=-0.979
      rew=+0.0000  ret=+5.4019  val=+3.4142  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9056  sprint_p_new=0.8093  kick_p_new=0.0166  tackle_attempt_p_new=0.2006
    idx=  13  ratio=   9.737  adv=+1.980  lp: old=-3.293  new=-1.017
      rew=+0.0000  ret=+5.4884  val=+3.5086  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8904  sprint_p_new=0.7896  kick_p_new=0.0193  tackle_attempt_p_new=0.2150
  [best sample (highest new_lp)] idx=4  new_lp=-0.763  adv=+1.399  stored move_dir=26.4°  new_mean=21.9°
    per-head contributions: move_dir:0.067  sprint:-0.167  tackle_attempt:-0.189  move:-0.461
2026-08-08 20:28:37,830 INFO   [advantage] mean=-0.000  std=1.000  min=-6.255  max=4.097
2026-08-08 20:28:37,831 INFO   [ratio] mean=0.9440  std=0.2409  min=0.0058  max=7.1282  clipped=36.0%
2026-08-08 20:28:37,831 INFO   [exec head grad norm] move_direction=0.029  exec_move=0.043  sprint=0.042  kick=0.045  kick_direction=0.011  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.054
2026-08-08 20:28:37,831 INFO   [exec continuous log_std] move_direction: start=-1.6273 end=-1.6263   kick_direction: start=-1.6257 end=-1.6244
2026-08-08 20:28:37,831 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0019≈0.11°/step  epoch≈6.6°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0018≈0.10°/step  epoch≈6.1°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 20:28:37,831 INFO   [exec discrete Δlogit per opt step] exec_move=0.0012  sprint=0.0018  kick=0.0017  tackle_attempt=0.0010
2026-08-08 20:28:37,832 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0039  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0037  sprint=+0.0058  kick=+0.0039  tackle_attempt=+0.0019  move_dir=+0.0573  kick_dir=+0.0259
2026-08-08 20:28:37,832 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.255 max=0.357  limit=0.4
              direction: 58/60 steps clipped (97%)  pre-clip norm mean=0.036 max=0.073  limit=0.02
2026-08-08 20:28:37,886 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=936,000  speed=873/s  reward=3.13
  loss     policy=0.0217  value=0.6314(x0.5)=0.3157
           entropy=6.5209  kl=0.1023
  value    V=2.43±0.97  R=2.47±1.48  adv=0.04±1.17
  moves    mv_ls=[-1.6263] (σ≈0.20, ≈11°) g=1.38e-02  d_move=[+0.0010] (Δσ≈0.011°)
           kk_ls=[-1.6244] (σ≈0.20, ≈11°)  d_kick=[+0.0013] (Δσ≈0.015°)
  heads    move= 52 get_poss= 45 exec_move= 81 sprint= 45 kick= 19 tackle= 30 shoot= 40 hold= 40 tackle_prob=0.3105 kick_prob=0.1964
  vs       vs[win/loss/tout/miss]  vs_immobile(498): 61.0%/0.0%/3.2%/19.1%/17%
  ep_len   21.5±13.0s  (n=498, min=0.5s, max=50.0s)
  reward   get_possession=+400.00  lose_possession=-6.30  ball_out=-132.00  box_possession=+760.00
           speed_bonus=+539.10  timeout=-24.00  stamina_penalty=-3.06
  rew/ep   (mean/std/min/max per episode, 498 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.803    0.431    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.013    0.106    -0.900    +0.000
  ball_out          -0.265    0.995    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.526    1.219    +0.000    +2.500
  speed_bonus       +1.083    1.241    +0.000    +4.127
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.048    0.265    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.007    -0.026    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     403    +0.011    0.105     +3.432     1.489     +0.968      2.8931      1.425     2.946
  lose_possession       7    -0.000    0.013     +2.601     0.635     -0.905      1.3773      0.941     1.985
  ball_out            33    -0.004    0.121     -3.970     0.171     -5.009     26.7438      5.009     7.190
  box_possession     304    +0.021    0.229     +4.267     1.134     +1.046      2.2460      1.233     2.684
  speed_bonus        288    +0.015    0.193     +4.362     1.088     +1.121      2.3290      1.260     2.772
  timeout             16    -0.001    0.032     -1.504     0.006     -4.168     18.0151      4.168     5.113
  stamina_penalty     298    -0.000    0.001     +4.164     1.457     +0.947      2.7750      1.344     2.905
  gae/td   mean_return=+2.466  std_return=1.476  mean_gae=+0.041  mean_sq_td=1.3653
──────────────────────────────────────────────────────────────────────
2026-08-08 20:28:37,910 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint24.pt
2026-08-08 20:28:37,911 INFO Logging to checkpoints/phase1_run45/training_log25.txt
2026-08-08 20:28:37,912 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 20:28:51,808 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 20:28:51,809 INFO   [eval vs immobile] step=936,000  seeds=16x8  win=48%  mean_rew=2.200±3.104  V=2.399  gap=+0.199  outcomes={'other': 26, 'box_possession': 61, 'timeout': 6, 'miss': 35}
2026-08-08 20:28:51,811 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 20:29:01,315 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 20:29:01,317 INFO   [eval vs rules] step=936,000  seeds=16x8  win=12%  mean_rew=-1.320±2.920  V=1.908  gap=+3.228  outcomes={'other': 17, 'opponent_box_possession': 88, 'box_possession': 15, 'miss': 8}
2026-08-08 20:33:46,469 INFO   [KL mean=0.1046 median=0.1050 > 0.05] ratio percentiles:  p5=0.550  p25=0.845  p50=0.970  p75=1.034  p95=1.262  max=7.004
  move_dir_log_std=[-1.6254122257232666]  kick_dir_log_std=[-1.6231579780578613]
2026-08-08 20:33:46,481 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.687  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.393  kick=-0.467  t_att=-0.630
    move_dir=0.674 (min=-4.026 max=1.413)  kick_dir=0.226 (min=-5.098 max=2.098)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.49
  [worst sample] idx=223  ratio=12.861  adv=-3.433  old_lp=-3.631  new_lp=-1.077
    stored move_dir=-15.2°  new_mean=-17.0°  angular_diff=1.7°
    [worst sample per-head delta, sorted by |delta|] move:+0.064
  [top-2 highest-ratio samples]
    idx= 223  ratio=  12.861  adv=-3.433  lp: old=-3.631  new=-1.077
      rew=+0.0000  ret=-0.0476  val=+3.3857  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: move:+0.064
      saturation: exec_move_p_new=0.9143  sprint_p_new=0.8349  kick_p_new=0.0147  tackle_attempt_p_new=0.2301
    idx=  13  ratio=  10.037  adv=-1.682  lp: old=-3.176  new=-0.870
      rew=+0.0000  ret=+1.8113  val=+3.4930  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9005  sprint_p_new=0.8027  kick_p_new=0.0168  tackle_attempt_p_new=0.2263
  [best sample (highest new_lp)] idx=231  new_lp=-0.831  adv=-4.775  stored move_dir=8.2°  new_mean=2.4°
    per-head contributions: tackle_attempt:-0.258  move:-0.558
2026-08-08 20:33:46,482 INFO   [advantage] mean=0.000  std=1.000  min=-5.464  max=3.696
2026-08-08 20:33:46,482 INFO   [ratio] mean=0.9412  std=0.2359  min=0.0059  max=7.0042  clipped=35.5%
2026-08-08 20:33:46,483 INFO   [exec head grad norm] move_direction=0.030  exec_move=0.044  sprint=0.038  kick=0.039  kick_direction=0.010  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.057
2026-08-08 20:33:46,483 INFO   [exec continuous log_std] move_direction: start=-1.6263 end=-1.6254   kick_direction: start=-1.6244 end=-1.6232
2026-08-08 20:33:46,483 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0019≈0.11°/step  epoch≈6.7°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0019≈0.11°/step  epoch≈6.7°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 20:33:46,483 INFO   [exec discrete Δlogit per opt step] exec_move=0.0011  sprint=0.0013  kick=0.0020  tackle_attempt=0.0011
2026-08-08 20:33:46,483 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0036  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0035  sprint=+0.0056  kick=+0.0042  tackle_attempt=+0.0020  move_dir=+0.0594  kick_dir=+0.0263
2026-08-08 20:33:46,484 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.241 max=0.384  limit=0.4
              direction: 59/60 steps clipped (98%)  pre-clip norm mean=0.035 max=0.073  limit=0.02
2026-08-08 20:33:46,550 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=972,000  speed=872/s  reward=3.08
  loss     policy=0.0196  value=0.7081(x0.5)=0.3540
           entropy=6.5812  kl=0.1046
  value    V=2.52±0.87  R=2.41±1.56  adv=-0.11±1.32
  moves    mv_ls=[-1.6254] (σ≈0.20, ≈11°) g=1.31e-02  d_move=[+0.0009] (Δσ≈0.010°)
           kk_ls=[-1.6232] (σ≈0.20, ≈11°)  d_kick=[+0.0012] (Δσ≈0.013°)
  heads    move= 53 get_poss= 44 exec_move= 80 sprint= 44 kick= 20 tackle= 31 shoot= 41 hold= 42 tackle_prob=0.3211 kick_prob=0.2068
  vs       vs[win/loss/tout/miss]  vs_immobile(497): 59.8%/0.0%/6.4%/20.7%/13%
  ep_len   21.7±13.1s  (n=497, min=0.8s, max=50.0s)
  reward   get_possession=+421.00  lose_possession=-3.60  ball_out=-240.00  box_possession=+742.50
           speed_bonus=+514.06  timeout=-48.00  stamina_penalty=-2.91
  rew/ep   (mean/std/min/max per episode, 497 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.847    0.382    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.007    0.080    -0.900    +0.000
  ball_out          -0.483    1.303    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.494    1.226    +0.000    +2.500
  speed_bonus       +1.034    1.229    +0.000    +4.079
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.097    0.368    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.034    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     424    +0.012    0.108     +3.183     1.760     +0.569      3.0986      1.424     3.837
  lose_possession       4    -0.000    0.009     +2.979     0.329     +0.099      0.2711      0.434     0.752
  ball_out            60    -0.007    0.163     -3.983     0.128     -5.329     29.7467      5.329     6.908
  box_possession     297    +0.021    0.226     +4.235     1.143     +1.115      2.3514      1.252     2.677
  speed_bonus        279    +0.014    0.188     +4.340     1.096     +1.183      2.4754      1.298     2.678
  timeout             32    -0.001    0.045     -1.505     0.007     -4.138     17.9123      4.138     5.155
  stamina_penalty     295    -0.000    0.001     +3.928     1.800     +0.810      3.3621      1.459     3.446
  gae/td   mean_return=+2.407  std_return=1.560  mean_gae=-0.108  mean_sq_td=1.7505
──────────────────────────────────────────────────────────────────────
2026-08-08 20:33:46,575 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint25.pt
2026-08-08 20:33:46,575 INFO Logging to checkpoints/phase1_run45/training_log26.txt
2026-08-08 20:33:46,576 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 20:33:59,456 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 20:33:59,457 INFO   [eval vs immobile] step=972,000  seeds=16x8  win=50%  mean_rew=2.386±3.167  V=2.317  gap=-0.069  outcomes={'other': 25, 'box_possession': 64, 'miss': 38, 'timeout': 1}
2026-08-08 20:33:59,458 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 20:34:08,733 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 20:34:08,734 INFO   [eval vs rules] step=972,000  seeds=16x8  win=9%  mean_rew=-1.528±2.734  V=1.925  gap=+3.454  outcomes={'other': 15, 'opponent_box_possession': 89, 'box_possession': 12, 'miss': 12}
2026-08-08 20:38:50,983 INFO   [KL mean=0.1037 median=0.1038 > 0.05] ratio percentiles:  p5=0.550  p25=0.846  p50=0.974  p75=1.031  p95=1.258  max=10.464
  move_dir_log_std=[-1.6244525909423828]  kick_dir_log_std=[-1.6217941045761108]
2026-08-08 20:38:50,994 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.669  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.478  kick=-0.541  t_att=-0.642
    move_dir=0.644 (min=-3.232 max=1.411)  kick_dir=0.308 (min=-2.533 max=2.101)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.47
  [worst sample] idx=196  ratio=7.149  adv=-1.309  old_lp=-5.081  new_lp=-3.114
    stored move_dir=77.0°  new_mean=46.0°  angular_diff=31.0°
    [worst sample per-head delta, sorted by |delta|] kick:+0.038  tackle_attempt:+0.027
  [top-2 highest-ratio samples]
    idx= 196  ratio=   7.149  adv=-1.309  lp: old=-5.081  new=-3.114
      rew=+0.0000  ret=+0.6925  val=+2.0012  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: kick:+0.038  tackle_attempt:+0.027
      saturation: exec_move_p_new=0.8485  sprint_p_new=0.4869  kick_p_new=0.2358  tackle_attempt_p_new=0.3136
    idx=  95  ratio=   6.764  adv=-2.002  lp: old=-3.153  new=-1.242
      rew=+0.0000  ret=+0.2187  val=+2.2209  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8545  sprint_p_new=0.5726  kick_p_new=0.2144  tackle_attempt_p_new=0.3467
  [best sample (highest new_lp)] idx=28  new_lp=-0.670  adv=-1.750  stored move_dir=-114.4°  new_mean=-131.9°
    per-head contributions: kick:-0.060  tackle_attempt:-0.302  move:-0.309
2026-08-08 20:38:50,995 INFO   [advantage] mean=-0.000  std=1.000  min=-5.466  max=3.506
2026-08-08 20:38:50,995 INFO   [ratio] mean=0.9415  std=0.2379  min=0.0051  max=10.4645  clipped=35.4%
2026-08-08 20:38:50,996 INFO   [exec head grad norm] move_direction=0.022  exec_move=0.034  sprint=0.045  kick=0.034  kick_direction=0.011  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.066
2026-08-08 20:38:50,996 INFO   [exec continuous log_std] move_direction: start=-1.6254 end=-1.6245   kick_direction: start=-1.6232 end=-1.6218
2026-08-08 20:38:50,996 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0014≈0.08°/step  epoch≈4.8°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0014≈0.08°/step  epoch≈4.8°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 20:38:50,996 INFO   [exec discrete Δlogit per opt step] exec_move=0.0009  sprint=0.0012  kick=0.0015  tackle_attempt=0.0011
2026-08-08 20:38:50,996 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0036  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0039  sprint=+0.0048  kick=+0.0043  tackle_attempt=+0.0024  move_dir=+0.0579  kick_dir=+0.0267
2026-08-08 20:38:50,997 INFO   [grad clip] main: 2/60 steps clipped (3%)  pre-clip norm mean=0.245 max=0.439  limit=0.4
              direction: 56/60 steps clipped (93%)  pre-clip norm mean=0.029 max=0.051  limit=0.02
2026-08-08 20:38:51,047 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=1,008,000  speed=875/s  reward=3.05
  loss     policy=0.0158  value=0.7030(x0.5)=0.3515
           entropy=6.6132  kl=0.1037
  value    V=2.41±0.83  R=2.29±1.57  adv=-0.12±1.32
  moves    mv_ls=[-1.6245] (σ≈0.20, ≈11°) g=1.19e-02  d_move=[+0.0010] (Δσ≈0.011°)
           kk_ls=[-1.6218] (σ≈0.20, ≈11°)  d_kick=[+0.0014] (Δσ≈0.015°)
  heads    move= 52 get_poss= 44 exec_move= 81 sprint= 44 kick= 21 tackle= 33 shoot= 43 hold= 43 tackle_prob=0.3289 kick_prob=0.2109
  vs       vs[win/loss/tout/miss]  vs_immobile(513): 56.7%/0.0%/4.7%/22.6%/16%
  ep_len   20.9±12.9s  (n=513, min=1.2s, max=50.0s)
  reward   get_possession=+428.00  lose_possession=-3.60  ball_out=-260.00  box_possession=+727.50
           speed_bonus=+509.31  timeout=-36.00  stamina_penalty=-2.77
  rew/ep   (mean/std/min/max per episode, 513 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.834    0.392    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.007    0.079    -0.900    +0.000
  ball_out          -0.507    1.331    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.418    1.239    +0.000    +2.500
  speed_bonus       +0.993    1.198    +0.000    +4.069
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.070    0.317    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.005    0.006    -0.026    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     431    +0.012    0.109     +3.083     1.867     +0.527      3.4264      1.459     4.315
  lose_possession       4    -0.000    0.009     +2.942     0.955     +0.316      1.1492      0.705     1.832
  ball_out            65    -0.007    0.170     -3.969     0.173     -4.936     25.6041      4.936     6.302
  box_possession     291    +0.020    0.224     +4.241     1.096     +1.434      3.1588      1.489     3.183
  speed_bonus        282    +0.014    0.185     +4.297     1.068     +1.492      3.2507      1.523     3.184
  timeout             24    -0.001    0.039     -1.506     0.006     -3.863     15.7296      3.863     4.772
  stamina_penalty     302    -0.000    0.001     +3.933     1.700     +1.139      3.9578      1.634     3.587
  gae/td   mean_return=+2.288  std_return=1.574  mean_gae=-0.122  mean_sq_td=1.7539
──────────────────────────────────────────────────────────────────────
2026-08-08 20:38:51,074 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint26.pt
2026-08-08 20:38:51,074 INFO Logging to checkpoints/phase1_run45/training_log27.txt
2026-08-08 20:38:51,076 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 20:39:05,557 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 20:39:05,559 INFO   [eval vs immobile] step=1,008,000  seeds=16x8  win=49%  mean_rew=2.397±2.972  V=2.179  gap=-0.218  outcomes={'other': 30, 'box_possession': 63, 'miss': 30, 'timeout': 5}
2026-08-08 20:39:05,560 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 20:39:15,432 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 20:39:15,433 INFO   [eval vs rules] step=1,008,000  seeds=16x8  win=7%  mean_rew=-1.723±2.420  V=1.831  gap=+3.554  outcomes={'box_possession': 9, 'opponent_box_possession': 90, 'other': 17, 'miss': 12}
2026-08-08 20:43:58,737 INFO   [KL mean=0.1011 median=0.1008 > 0.05] ratio percentiles:  p5=0.561  p25=0.852  p50=0.976  p75=1.031  p95=1.249  max=6.358
  move_dir_log_std=[-1.623467206954956]  kick_dir_log_std=[-1.6206307411193848]
2026-08-08 20:43:58,750 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.692  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.433  kick=-0.259  t_att=-0.605
    move_dir=0.688 (min=-3.654 max=1.409)  kick_dir=0.116 (min=-0.666 max=2.036)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.43
  [worst sample] idx=38  ratio=9.305  adv=+2.800  old_lp=-3.304  new_lp=-1.073
    stored move_dir=-28.0°  new_mean=-7.7°  angular_diff=20.2°
    [worst sample per-head delta, sorted by |delta|] move:+0.025
  [top-2 highest-ratio samples]
    idx=  38  ratio=   9.305  adv=+2.800  lp: old=-3.304  new=-1.073
      rew=+0.0000  ret=+5.6973  val=+2.8971  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: move:+0.025
      saturation: exec_move_p_new=0.9002  sprint_p_new=0.7791  kick_p_new=0.0115  tackle_attempt_p_new=0.2404
    idx=  81  ratio=   8.779  adv=+0.892  lp: old=-3.304  new=-1.132
      rew=+0.0000  ret=+4.1400  val=+3.2477  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: move:+0.024
      saturation: exec_move_p_new=0.8942  sprint_p_new=0.7608  kick_p_new=0.0110  tackle_attempt_p_new=0.2379
  [best sample (highest new_lp)] idx=183  new_lp=-0.850  adv=-0.223  stored move_dir=168.1°  new_mean=156.0°
    per-head contributions: tackle_attempt:-0.283  move:-0.551
2026-08-08 20:43:58,750 INFO   [advantage] mean=-0.000  std=1.000  min=-5.142  max=3.349
2026-08-08 20:43:58,751 INFO   [ratio] mean=0.9424  std=0.2290  min=0.0064  max=6.3578  clipped=34.3%
2026-08-08 20:43:58,751 INFO   [exec head grad norm] move_direction=0.027  exec_move=0.039  sprint=0.042  kick=0.039  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.049
2026-08-08 20:43:58,751 INFO   [exec continuous log_std] move_direction: start=-1.6245 end=-1.6235   kick_direction: start=-1.6218 end=-1.6206
2026-08-08 20:43:58,751 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0019≈0.11°/step  epoch≈6.7°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0019≈0.11°/step  epoch≈6.4°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 20:43:58,751 INFO   [exec discrete Δlogit per opt step] exec_move=0.0010  sprint=0.0012  kick=0.0016  tackle_attempt=0.0008
2026-08-08 20:43:58,752 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0032  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0029  sprint=+0.0048  kick=+0.0048  tackle_attempt=+0.0022  move_dir=+0.0562  kick_dir=+0.0269
2026-08-08 20:43:58,752 INFO   [grad clip] main: 1/60 steps clipped (2%)  pre-clip norm mean=0.244 max=0.581  limit=0.4
              direction: 54/60 steps clipped (90%)  pre-clip norm mean=0.033 max=0.111  limit=0.02
2026-08-08 20:43:58,813 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=1,044,000  speed=875/s  reward=2.43
  loss     policy=0.0163  value=0.6374(x0.5)=0.3187
           entropy=6.6600  kl=0.1011
  value    V=2.27±0.88  R=2.14±1.62  adv=-0.13±1.30
  moves    mv_ls=[-1.6235] (σ≈0.20, ≈11°) g=1.30e-02  d_move=[+0.0010] (Δσ≈0.011°)
           kk_ls=[-1.6206] (σ≈0.20, ≈11°)  d_kick=[+0.0012] (Δσ≈0.013°)
  heads    move= 53 get_poss= 43 exec_move= 80 sprint= 44 kick= 21 tackle= 34 shoot= 43 hold= 44 tackle_prob=0.3397 kick_prob=0.2217
  vs       vs[win/loss/tout/miss]  vs_immobile(481): 54.5%/0.4%/6.9%/22.9%/15%
  ep_len   22.3±13.7s  (n=481, min=0.8s, max=50.0s)
  reward   get_possession=+400.00  lose_possession=-3.60  ball_out=-264.00  box_possession=+655.00
           speed_bonus=+477.99  opponent_box=-6.00  timeout=-49.50  stamina_penalty=-2.78
  rew/ep   (mean/std/min/max per episode, 481 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.832    0.396    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.007    0.082    -0.900    +0.000
  ball_out          -0.549    1.376    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.362    1.245    +0.000    +2.500
  speed_bonus       +0.994    1.265    +0.000    +4.153
  opponent_box      -0.012    0.193    -3.000    +0.000
  timeout           -0.103    0.379    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.025    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     403    +0.011    0.105     +2.922     1.927     +0.489      3.3443      1.456     3.997
  lose_possession       4    -0.000    0.009     +2.788     0.691     +0.181      0.1880      0.320     0.728
  ball_out            66    -0.007    0.171     -3.970     0.171     -4.641     22.4890      4.641     6.200
  box_possession     262    +0.018    0.212     +4.322     1.185     +1.662      4.0609      1.700     3.328
  speed_bonus        248    +0.013    0.185     +4.422     1.139     +1.736      4.2168      1.767     3.323
  opponent_box         2    -0.000    0.022     -3.003     0.000     -5.426     29.6183      5.426     5.804
  timeout             33    -0.001    0.045     -1.508     0.007     -3.793     15.2037      3.793     4.657
  stamina_penalty     280    -0.000    0.001     +3.787     2.097     +1.153      5.3054      1.929     4.168
  gae/td   mean_return=+2.144  std_return=1.621  mean_gae=-0.126  mean_sq_td=1.6991
──────────────────────────────────────────────────────────────────────
2026-08-08 20:43:58,840 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint27.pt
2026-08-08 20:43:58,840 INFO Logging to checkpoints/phase1_run45/training_log28.txt
2026-08-08 20:43:58,841 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 20:44:14,194 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 20:44:14,196 INFO   [eval vs immobile] step=1,044,000  seeds=16x8  win=40%  mean_rew=1.618±3.055  V=1.957  gap=+0.339  outcomes={'other': 27, 'box_possession': 51, 'miss': 43, 'timeout': 7}
2026-08-08 20:44:14,197 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 20:44:24,402 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 20:44:24,403 INFO   [eval vs rules] step=1,044,000  seeds=16x8  win=9%  mean_rew=-1.560±2.731  V=1.687  gap=+3.247  outcomes={'opponent_box_possession': 92, 'other': 14, 'box_possession': 12, 'miss': 10}
2026-08-08 20:49:08,591 INFO   [KL mean=0.1038 median=0.1038 > 0.05] ratio percentiles:  p5=0.553  p25=0.853  p50=0.977  p75=1.030  p95=1.248  max=6.409
  move_dir_log_std=[-1.6226314306259155]  kick_dir_log_std=[-1.6196166276931763]
2026-08-08 20:49:08,608 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.681  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.394  kick=-0.596  t_att=-0.646
    move_dir=0.623 (min=-3.760 max=1.407)  kick_dir=0.277 (min=-3.527 max=2.100)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.57
  [worst sample] idx=136  ratio=7.809  adv=-3.695  old_lp=-4.152  new_lp=-2.096
    stored move_dir=11.4°  new_mean=7.4°  angular_diff=4.1°
    [worst sample per-head delta, sorted by |delta|] tackle_attempt:+0.038
  [top-2 highest-ratio samples]
    idx= 136  ratio=   7.809  adv=-3.695  lp: old=-4.152  new=-2.096
      rew=+0.0000  ret=-2.7987  val=+0.8967  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: tackle_attempt:+0.038
      saturation: exec_move_p_new=0.8655  sprint_p_new=0.2301  kick_p_new=0.2277  tackle_attempt_p_new=0.3541
    idx=  49  ratio=   6.464  adv=-0.129  lp: old=-3.347  new=-1.480
      rew=+0.0000  ret=+1.4075  val=+1.5362  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: tackle_attempt:-0.027
      saturation: exec_move_p_new=0.8480  sprint_p_new=0.2954  kick_p_new=0.2447  tackle_attempt_p_new=0.3342
  [best sample (highest new_lp)] idx=7  new_lp=-1.203  adv=-1.011  stored move_dir=-96.9°  new_mean=-88.0°
    per-head contributions: move_dir:0.055  kick:-0.209  sprint:-0.259  move:-0.355  tackle_attempt:-0.436
2026-08-08 20:49:08,608 INFO   [advantage] mean=-0.000  std=1.000  min=-5.411  max=4.428
2026-08-08 20:49:08,609 INFO   [ratio] mean=0.9421  std=0.2280  min=0.0052  max=6.4086  clipped=34.0%
2026-08-08 20:49:08,609 INFO   [exec head grad norm] move_direction=0.031  exec_move=0.037  sprint=0.048  kick=0.035  kick_direction=0.013  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.057
2026-08-08 20:49:08,609 INFO   [exec continuous log_std] move_direction: start=-1.6235 end=-1.6226   kick_direction: start=-1.6206 end=-1.6196
2026-08-08 20:49:08,610 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0021≈0.12°/step  epoch≈7.2°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0020≈0.11°/step  epoch≈6.9°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 20:49:08,610 INFO   [exec discrete Δlogit per opt step] exec_move=0.0010  sprint=0.0012  kick=0.0014  tackle_attempt=0.0009
2026-08-08 20:49:08,610 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0030  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0032  sprint=+0.0046  kick=+0.0038  tackle_attempt=+0.0020  move_dir=+0.0578  kick_dir=+0.0294
2026-08-08 20:49:08,610 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.251 max=0.397  limit=0.4
              direction: 60/60 steps clipped (100%)  pre-clip norm mean=0.037 max=0.071  limit=0.02
2026-08-08 20:49:08,659 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=1,080,000  speed=871/s  reward=2.22
  loss     policy=0.0174  value=0.6195(x0.5)=0.3098
           entropy=6.6988  kl=0.1038
  value    V=2.14±0.92  R=2.07±1.62  adv=-0.07±1.27
  moves    mv_ls=[-1.6226] (σ≈0.20, ≈11°) g=1.28e-02  d_move=[+0.0008] (Δσ≈0.009°)
           kk_ls=[-1.6196] (σ≈0.20, ≈11°)  d_kick=[+0.0010] (Δσ≈0.011°)
  heads    move= 52 get_poss= 41 exec_move= 80 sprint= 43 kick= 23 tackle= 34 shoot= 44 hold= 45 tackle_prob=0.3489 kick_prob=0.2358
  vs       vs[win/loss/tout/miss]  vs_immobile(485): 53.6%/0.2%/6.8%/22.9%/16%
  ep_len   22.2±13.6s  (n=485, min=1.0s, max=50.0s)
  reward   get_possession=+395.00  lose_possession=-5.40  ball_out=-244.00  box_possession=+650.00
           speed_bonus=+477.05  opponent_box=-3.00  timeout=-49.50  stamina_penalty=-2.55
  rew/ep   (mean/std/min/max per episode, 485 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.814    0.419    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.011    0.099    -0.900    +0.000
  ball_out          -0.503    1.326    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.340    1.247    +0.000    +2.500
  speed_bonus       +0.984    1.226    +0.000    +4.079
  opponent_box      -0.006    0.136    -3.000    +0.000
  timeout           -0.102    0.378    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.005    0.006    -0.027    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     397    +0.011    0.104     +2.930     1.837     +0.618      3.4499      1.507     3.914
  lose_possession       6    -0.000    0.012     +2.281     0.791     -0.428      0.4908      0.512     1.268
  ball_out            61    -0.007    0.165     -3.967     0.178     -4.849     24.7103      4.849     6.659
  box_possession     260    +0.018    0.212     +4.326     1.113     +1.587      3.6148      1.619     3.225
  speed_bonus        250    +0.013    0.182     +4.399     1.072     +1.648      3.7511      1.668     3.228
  opponent_box         1    -0.000    0.016     -3.003     0.000     -5.368     28.8154      5.368     5.368
  timeout             33    -0.001    0.045     -1.506     0.006     -3.621     13.9031      3.621     4.776
  stamina_penalty     271    -0.000    0.001     +3.890     1.918     +1.201      4.5399      1.800     3.533
  gae/td   mean_return=+2.068  std_return=1.615  mean_gae=-0.071  mean_sq_td=1.6202
──────────────────────────────────────────────────────────────────────
2026-08-08 20:49:08,686 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint28.pt
2026-08-08 20:49:08,686 INFO Logging to checkpoints/phase1_run45/training_log29.txt
2026-08-08 20:49:08,687 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 20:49:27,025 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 20:49:27,027 INFO   [eval vs immobile] step=1,080,000  seeds=16x8  win=38%  mean_rew=1.657±3.003  V=1.871  gap=+0.214  outcomes={'other': 25, 'box_possession': 48, 'miss': 40, 'timeout': 15}
2026-08-08 20:49:27,029 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 20:49:37,699 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 20:49:37,700 INFO   [eval vs rules] step=1,080,000  seeds=16x8  win=6%  mean_rew=-1.722±2.407  V=1.533  gap=+3.256  outcomes={'other': 19, 'box_possession': 8, 'opponent_box_possession': 91, 'miss': 10}
2026-08-08 20:54:23,020 INFO   [KL mean=0.0985 median=0.0986 > 0.05] ratio percentiles:  p5=0.573  p25=0.860  p50=0.979  p75=1.031  p95=1.239  max=6.514
  move_dir_log_std=[-1.6217433214187622]  kick_dir_log_std=[-1.6185104846954346]
2026-08-08 20:54:23,032 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.681  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.432  kick=-0.333  t_att=-0.601
    move_dir=0.730 (min=-2.453 max=1.406)  kick_dir=0.151 (min=-3.059 max=2.093)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.42
  [worst sample] idx=249  ratio=11.464  adv=+0.997  old_lp=-3.412  new_lp=-0.973
    stored move_dir=49.6°  new_mean=52.3°  angular_diff=2.8°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 249  ratio=  11.464  adv=+0.997  lp: old=-3.412  new=-0.973
      rew=+0.0000  ret=+3.4353  val=+2.4383  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9139  sprint_p_new=0.8327  kick_p_new=0.0249  tackle_attempt_p_new=0.2395
    idx= 219  ratio=  11.139  adv=-4.656  lp: old=-3.469  new=-1.058
      rew=+0.0000  ret=-1.9403  val=+2.7156  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9109  sprint_p_new=0.7787  kick_p_new=0.0077  tackle_attempt_p_new=0.2399
  [best sample (highest new_lp)] idx=141  new_lp=-0.762  adv=+0.767  stored move_dir=146.3°  new_mean=156.6°
    per-head contributions: tackle_attempt:-0.261  move:-0.494
2026-08-08 20:54:23,032 INFO   [advantage] mean=0.000  std=1.000  min=-4.925  max=3.783
2026-08-08 20:54:23,033 INFO   [ratio] mean=0.9447  std=0.2231  min=0.0060  max=6.5140  clipped=32.8%
2026-08-08 20:54:23,034 INFO   [exec head grad norm] move_direction=0.027  exec_move=0.043  sprint=0.051  kick=0.041  kick_direction=0.011  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.050
2026-08-08 20:54:23,034 INFO   [exec continuous log_std] move_direction: start=-1.6226 end=-1.6217   kick_direction: start=-1.6196 end=-1.6185
2026-08-08 20:54:23,034 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0017≈0.10°/step  epoch≈5.7°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0017≈0.09°/step  epoch≈5.7°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 20:54:23,034 INFO   [exec discrete Δlogit per opt step] exec_move=0.0009  sprint=0.0014  kick=0.0014  tackle_attempt=0.0007
2026-08-08 20:54:23,034 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0034  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0030  sprint=+0.0044  kick=+0.0045  tackle_attempt=+0.0017  move_dir=+0.0544  kick_dir=+0.0272
2026-08-08 20:54:23,035 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.242 max=0.335  limit=0.4
              direction: 60/60 steps clipped (100%)  pre-clip norm mean=0.033 max=0.056  limit=0.02
2026-08-08 20:54:23,080 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=1,116,000  speed=870/s  reward=4.71
  loss     policy=0.0170  value=0.6085(x0.5)=0.3042
           entropy=6.7176  kl=0.0985
  value    V=2.14±0.95  R=2.14±1.61  adv=0.00±1.26
  moves    mv_ls=[-1.6217] (σ≈0.20, ≈11°) g=1.25e-02  d_move=[+0.0009] (Δσ≈0.010°)
           kk_ls=[-1.6185] (σ≈0.20, ≈11°)  d_kick=[+0.0011] (Δσ≈0.013°)
  heads    move= 53 get_poss= 41 exec_move= 79 sprint= 43 kick= 23 tackle= 35 shoot= 44 hold= 45 tackle_prob=0.3538 kick_prob=0.2390
  vs       vs[win/loss/tout/miss]  vs_immobile(479): 57.6%/0.0%/6.3%/21.7%/14%
  ep_len   22.3±13.6s  (n=479, min=0.9s, max=50.0s)
  reward   get_possession=+400.00  lose_possession=-7.20  ball_out=-224.00  box_possession=+690.00
           speed_bonus=+492.32  timeout=-45.00  stamina_penalty=-2.55
  rew/ep   (mean/std/min/max per episode, 479 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.835    0.414    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.015    0.115    -0.900    +0.000
  ball_out          -0.468    1.285    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.441    1.235    +0.000    +2.500
  speed_bonus       +1.028    1.216    +0.000    +4.006
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.094    0.363    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.005    0.006    -0.028    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     406    +0.011    0.106     +3.088     1.723     +0.880      3.4545      1.539     3.747
  lose_possession       8    -0.000    0.013     +2.597     0.421     -0.351      0.2850      0.418     0.889
  ball_out            56    -0.006    0.158     -3.946     0.225     -4.126     18.1199      4.126     5.918
  box_possession     276    +0.019    0.218     +4.282     1.094     +1.508      3.2784      1.553     3.017
  speed_bonus        269    +0.014    0.183     +4.329     1.069     +1.539      3.3386      1.573     3.019
  timeout             30    -0.001    0.043     -1.504     0.006     -3.707     14.1643      3.707     4.751
  stamina_penalty     277    -0.000    0.001     +3.939     1.809     +1.185      4.2469      1.761     3.727
  gae/td   mean_return=+2.141  std_return=1.607  mean_gae=+0.004  mean_sq_td=1.5759
──────────────────────────────────────────────────────────────────────
2026-08-08 20:54:23,108 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint29.pt
2026-08-08 20:54:23,108 INFO Logging to checkpoints/phase1_run45/training_log30.txt
2026-08-08 20:54:23,110 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 20:54:38,592 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 20:54:38,593 INFO   [eval vs immobile] step=1,116,000  seeds=16x8  win=45%  mean_rew=2.012±3.196  V=1.985  gap=-0.027  outcomes={'other': 24, 'box_possession': 57, 'timeout': 5, 'opponent_box_possession': 1, 'miss': 41}
2026-08-08 20:54:38,595 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 20:54:48,741 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 20:54:48,743 INFO   [eval vs rules] step=1,116,000  seeds=16x8  win=5%  mean_rew=-1.945±2.143  V=1.591  gap=+3.536  outcomes={'other': 16, 'opponent_box_possession': 95, 'miss': 11, 'box_possession': 6}
2026-08-08 20:59:34,774 INFO   [KL mean=0.0997 median=0.0992 > 0.05] ratio percentiles:  p5=0.570  p25=0.858  p50=0.983  p75=1.025  p95=1.240  max=16.670
  move_dir_log_std=[-1.6208491325378418]  kick_dir_log_std=[-1.6173475980758667]
2026-08-08 20:59:34,788 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.699  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.426  kick=-0.519  t_att=-0.634
    move_dir=0.694 (min=-6.701 max=1.404)  kick_dir=0.296 (min=-4.686 max=2.090)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.49
  [worst sample] idx=58  ratio=8.800  adv=+1.442  old_lp=-3.080  new_lp=-0.905
    stored move_dir=173.7°  new_mean=156.0°  angular_diff=17.6°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx=  58  ratio=   8.800  adv=+1.442  lp: old=-3.080  new=-0.905
      rew=+0.0000  ret=+4.8641  val=+3.4217  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8823  sprint_p_new=0.7411  kick_p_new=0.0102  tackle_attempt_p_new=0.2612
    idx=  54  ratio=   8.662  adv=+1.179  lp: old=-4.154  new=-1.995
      rew=+0.0000  ret=+4.6119  val=+3.4331  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8841  sprint_p_new=0.7349  kick_p_new=0.0087  tackle_attempt_p_new=0.2531
  [best sample (highest new_lp)] idx=58  new_lp=-0.905  adv=+1.442  stored move_dir=173.7°  new_mean=156.0°
    per-head contributions: tackle_attempt:-0.303  move:-0.592
2026-08-08 20:59:34,788 INFO   [advantage] mean=0.000  std=1.000  min=-5.111  max=3.862
2026-08-08 20:59:34,789 INFO   [ratio] mean=0.9442  std=0.2308  min=0.0031  max=16.6698  clipped=33.0%
2026-08-08 20:59:34,789 INFO   [exec head grad norm] move_direction=0.026  exec_move=0.036  sprint=0.042  kick=0.039  kick_direction=0.011  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.051
2026-08-08 20:59:34,789 INFO   [exec continuous log_std] move_direction: start=-1.6217 end=-1.6208   kick_direction: start=-1.6185 end=-1.6173
2026-08-08 20:59:34,789 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0020≈0.12°/step  epoch≈7.0°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0021≈0.12°/step  epoch≈7.1°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 20:59:34,790 INFO   [exec discrete Δlogit per opt step] exec_move=0.0009  sprint=0.0012  kick=0.0010  tackle_attempt=0.0005
2026-08-08 20:59:34,790 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0031  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0036  sprint=+0.0040  kick=+0.0038  tackle_attempt=+0.0018  move_dir=+0.0545  kick_dir=+0.0289
2026-08-08 20:59:34,790 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.233 max=0.324  limit=0.4
              direction: 58/60 steps clipped (97%)  pre-clip norm mean=0.033 max=0.061  limit=0.02
2026-08-08 20:59:34,839 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=1,152,000  speed=871/s  reward=2.24
  loss     policy=0.0155  value=0.6244(x0.5)=0.3122
           entropy=6.7534  kl=0.0997
  value    V=2.08±0.98  R=1.98±1.65  adv=-0.10±1.29
  moves    mv_ls=[-1.6208] (σ≈0.20, ≈11°) g=1.26e-02  d_move=[+0.0009] (Δσ≈0.010°)
           kk_ls=[-1.6173] (σ≈0.20, ≈11°)  d_kick=[+0.0012] (Δσ≈0.013°)
  heads    move= 53 get_poss= 41 exec_move= 79 sprint= 43 kick= 25 tackle= 36 shoot= 45 hold= 46 tackle_prob=0.3593 kick_prob=0.2510
  vs       vs[win/loss/tout/miss]  vs_immobile(484): 52.1%/0.0%/8.3%/26.9%/13%
  ep_len   22.3±14.5s  (n=484, min=1.1s, max=50.0s)
  reward   get_possession=+393.00  lose_possession=-5.40  ball_out=-296.00  box_possession=+630.00
           speed_bonus=+423.56  timeout=-60.00  stamina_penalty=-2.40
  rew/ep   (mean/std/min/max per episode, 484 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.812    0.426    +0.000    +3.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.011    0.115    -1.800    +0.000
  ball_out          -0.612    1.440    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.302    1.249    +0.000    +2.500
  speed_bonus       +0.875    1.179    +0.000    +3.943
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.124    0.413    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.005    0.006    -0.026    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     395    +0.011    0.104     +2.776     2.022     +0.600      3.8471      1.610     3.998
  lose_possession       6    -0.000    0.012     +2.184     0.340     -0.277      0.6524      0.636     1.279
  ball_out            74    -0.008    0.181     -3.986     0.115     -4.389     20.3947      4.389     6.017
  box_possession     252    +0.018    0.208     +4.176     1.142     +1.430      3.2901      1.500     3.235
  speed_bonus        240    +0.012    0.170     +4.260     1.106     +1.508      3.4476      1.559     3.270
  timeout             40    -0.002    0.050     -1.504     0.004     -3.474     13.3174      3.474     4.608
  stamina_penalty     267    -0.000    0.001     +3.661     2.014     +0.981      4.3080      1.714     3.984
  gae/td   mean_return=+1.982  std_return=1.650  mean_gae=-0.100  mean_sq_td=1.6778
──────────────────────────────────────────────────────────────────────
2026-08-08 20:59:34,867 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint30.pt
2026-08-08 20:59:34,867 INFO Logging to checkpoints/phase1_run45/training_log31.txt
2026-08-08 20:59:34,868 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 20:59:49,924 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 20:59:49,925 INFO   [eval vs immobile] step=1,152,000  seeds=16x8  win=41%  mean_rew=1.959±3.087  V=1.873  gap=-0.087  outcomes={'other': 26, 'box_possession': 53, 'timeout': 7, 'miss': 42}
2026-08-08 20:59:49,926 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 21:00:00,349 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 21:00:00,350 INFO   [eval vs rules] step=1,152,000  seeds=16x8  win=3%  mean_rew=-2.168±1.862  V=1.570  gap=+3.738  outcomes={'other': 11, 'opponent_box_possession': 101, 'box_possession': 4, 'miss': 12}
2026-08-08 21:04:43,451 INFO   [KL mean=0.0941 median=0.0945 > 0.05] ratio percentiles:  p5=0.591  p25=0.865  p50=0.978  p75=1.033  p95=1.235  max=4.673
  move_dir_log_std=[-1.6199641227722168]  kick_dir_log_std=[-1.6160948276519775]
2026-08-08 21:04:43,462 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.677  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.461  kick=-0.603  t_att=-0.622
    move_dir=0.765 (min=-3.513 max=1.402)  kick_dir=0.324 (min=-4.375 max=2.086)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.49
  [worst sample] idx=143  ratio=7.686  adv=-0.035  old_lp=-3.032  new_lp=-0.993
    stored move_dir=128.0°  new_mean=142.3°  angular_diff=14.2°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 143  ratio=   7.686  adv=-0.035  lp: old=-3.032  new=-0.993
      rew=+0.0000  ret=+2.5503  val=+2.5848  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8678  sprint_p_new=0.7588  kick_p_new=0.0123  tackle_attempt_p_new=0.2589
    idx= 123  ratio=   7.588  adv=-0.339  lp: old=-2.894  new=-0.867
      rew=+0.0000  ret=+2.5573  val=+2.8960  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: move:-0.021  tackle_attempt:-0.021
      saturation: exec_move_p_new=0.8680  sprint_p_new=0.7281  kick_p_new=0.0089  tackle_attempt_p_new=0.2597
  [best sample (highest new_lp)] idx=123  new_lp=-0.867  adv=-0.339  stored move_dir=152.0°  new_mean=159.8°
    per-head contributions: tackle_attempt:-0.301  move:-0.557
2026-08-08 21:04:43,462 INFO   [advantage] mean=-0.000  std=1.000  min=-5.302  max=4.433
2026-08-08 21:04:43,463 INFO   [ratio] mean=0.9470  std=0.2140  min=0.0053  max=4.6732  clipped=32.0%
2026-08-08 21:04:43,463 INFO   [exec head grad norm] move_direction=0.027  exec_move=0.045  sprint=0.045  kick=0.036  kick_direction=0.010  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.058
2026-08-08 21:04:43,463 INFO   [exec continuous log_std] move_direction: start=-1.6208 end=-1.6200   kick_direction: start=-1.6173 end=-1.6161
2026-08-08 21:04:43,463 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0017≈0.10°/step  epoch≈5.8°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0018≈0.10°/step  epoch≈6.1°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 21:04:43,463 INFO   [exec discrete Δlogit per opt step] exec_move=0.0010  sprint=0.0013  kick=0.0012  tackle_attempt=0.0008
2026-08-08 21:04:43,464 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0027  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0031  sprint=+0.0033  kick=+0.0039  tackle_attempt=+0.0016  move_dir=+0.0519  kick_dir=+0.0276
2026-08-08 21:04:43,464 INFO   [grad clip] main: 3/60 steps clipped (5%)  pre-clip norm mean=0.266 max=0.819  limit=0.4
              direction: 57/60 steps clipped (95%)  pre-clip norm mean=0.033 max=0.152  limit=0.02
2026-08-08 21:04:43,517 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=1,188,000  speed=859/s  reward=2.84
  loss     policy=0.0161  value=0.6090(x0.5)=0.3045
           entropy=6.7782  kl=0.0941
  value    V=2.02±0.99  R=2.08±1.57  adv=0.06±1.22
  moves    mv_ls=[-1.6200] (σ≈0.20, ≈11°) g=1.14e-02  d_move=[+0.0009] (Δσ≈0.010°)
           kk_ls=[-1.6161] (σ≈0.20, ≈11°)  d_kick=[+0.0013] (Δσ≈0.014°)
  heads    move= 52 get_poss= 41 exec_move= 80 sprint= 43 kick= 25 tackle= 36 shoot= 46 hold= 46 tackle_prob=0.3635 kick_prob=0.2543
  vs       vs[win/loss/tout/miss]  vs_immobile(471): 57.7%/0.0%/7.4%/20.0%/15%
  ep_len   22.7±14.2s  (n=471, min=0.8s, max=50.0s)
  reward   get_possession=+384.00  lose_possession=-1.80  ball_out=-200.00  box_possession=+680.00
           speed_bonus=+473.21  timeout=-52.50  stamina_penalty=-2.58
  rew/ep   (mean/std/min/max per episode, 471 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.815    0.399    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.004    0.059    -0.900    +0.000
  ball_out          -0.425    1.232    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.444    1.235    +0.000    +2.500
  speed_bonus       +1.005    1.234    +0.000    +4.258
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.111    0.393    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.005    0.006    -0.024    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     387    +0.011    0.103     +3.015     1.811     +0.958      3.6741      1.615     3.579
  lose_possession       2    -0.000    0.007     +2.372     0.125     +0.858      0.7953      0.858     1.076
  ball_out            50    -0.006    0.149     -3.980     0.140     -4.108     17.8871      4.108     5.647
  box_possession     272    +0.019    0.216     +4.235     1.166     +1.559      3.7597      1.617     3.390
  speed_bonus        258    +0.013    0.182     +4.329     1.123     +1.648      3.9548      1.686     3.394
  timeout             35    -0.001    0.047     -1.505     0.006     -3.693     14.6248      3.693     4.902
  stamina_penalty     280    -0.000    0.001     +3.814     1.966     +1.166      4.9835      1.864     4.090
  gae/td   mean_return=+2.077  std_return=1.570  mean_gae=+0.055  mean_sq_td=1.4874
──────────────────────────────────────────────────────────────────────
2026-08-08 21:04:43,545 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint31.pt
2026-08-08 21:04:43,546 INFO Logging to checkpoints/phase1_run45/training_log32.txt
2026-08-08 21:04:43,547 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 21:04:59,010 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 21:04:59,012 INFO   [eval vs immobile] step=1,188,000  seeds=16x8  win=44%  mean_rew=1.860±3.159  V=1.998  gap=+0.138  outcomes={'other': 20, 'miss': 44, 'box_possession': 56, 'timeout': 8}
2026-08-08 21:04:59,013 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 21:05:09,915 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 21:05:09,917 INFO   [eval vs rules] step=1,188,000  seeds=16x8  win=6%  mean_rew=-1.866±2.319  V=1.597  gap=+3.463  outcomes={'other': 13, 'opponent_box_possession': 95, 'miss': 12, 'box_possession': 8}
2026-08-08 21:09:59,911 INFO   [KL mean=0.0875 median=0.0872 > 0.05] ratio percentiles:  p5=0.601  p25=0.868  p50=0.981  p75=1.029  p95=1.233  max=8.920
  move_dir_log_std=[-1.6191335916519165]  kick_dir_log_std=[-1.614888310432434]
2026-08-08 21:09:59,923 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.667  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.429  kick=-0.442  t_att=-0.604
    move_dir=0.620 (min=-5.084 max=1.400)  kick_dir=0.246 (min=-1.738 max=2.083)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.53
  [worst sample] idx=66  ratio=10.656  adv=+1.748  old_lp=-3.347  new_lp=-0.980
    stored move_dir=172.9°  new_mean=173.6°  angular_diff=0.7°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx=  66  ratio=  10.656  adv=+1.748  lp: old=-3.347  new=-0.980
      rew=+0.0000  ret=+4.5157  val=+2.7675  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9036  sprint_p_new=0.7736  kick_p_new=0.0061  tackle_attempt_p_new=0.2559
    idx=  63  ratio=  10.558  adv=+1.197  lp: old=-3.344  new=-0.987
      rew=+0.0000  ret=+4.3171  val=+3.1205  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9029  sprint_p_new=0.7722  kick_p_new=0.0062  tackle_attempt_p_new=0.2572
  [best sample (highest new_lp)] idx=198  new_lp=-0.934  adv=-0.382  stored move_dir=-18.7°  new_mean=3.3°
    per-head contributions: tackle_attempt:-0.309  move:-0.616
2026-08-08 21:09:59,923 INFO   [advantage] mean=0.000  std=1.000  min=-5.165  max=3.181
2026-08-08 21:09:59,924 INFO   [ratio] mean=0.9500  std=0.2147  min=0.0027  max=8.9198  clipped=31.3%
2026-08-08 21:09:59,924 INFO   [exec head grad norm] move_direction=0.023  exec_move=0.039  sprint=0.045  kick=0.030  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.050
2026-08-08 21:09:59,924 INFO   [exec continuous log_std] move_direction: start=-1.6200 end=-1.6191   kick_direction: start=-1.6161 end=-1.6149
2026-08-08 21:09:59,925 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0014≈0.08°/step  epoch≈4.8°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0015≈0.08°/step  epoch≈5.1°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 21:09:59,925 INFO   [exec discrete Δlogit per opt step] exec_move=0.0009  sprint=0.0011  kick=0.0011  tackle_attempt=0.0008
2026-08-08 21:09:59,925 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0028  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0030  sprint=+0.0034  kick=+0.0045  tackle_attempt=+0.0014  move_dir=+0.0475  kick_dir=+0.0250
2026-08-08 21:09:59,925 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.232 max=0.392  limit=0.4
              direction: 48/60 steps clipped (80%)  pre-clip norm mean=0.028 max=0.089  limit=0.02
2026-08-08 21:09:59,961 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=1,224,000  speed=863/s  reward=3.78
  loss     policy=0.0139  value=0.5791(x0.5)=0.2896
           entropy=6.7646  kl=0.0875
  value    V=2.15±0.95  R=2.18±1.65  adv=0.04±1.27
  moves    mv_ls=[-1.6191] (σ≈0.20, ≈11°) g=9.56e-03  d_move=[+0.0008] (Δσ≈0.009°)
           kk_ls=[-1.6149] (σ≈0.20, ≈11°)  d_kick=[+0.0012] (Δσ≈0.014°)
  heads    move= 51 get_poss= 42 exec_move= 79 sprint= 45 kick= 24 tackle= 36 shoot= 45 hold= 45 tackle_prob=0.3646 kick_prob=0.2441
  vs       vs[win/loss/tout/miss]  vs_immobile(508): 57.1%/0.0%/4.9%/24.0%/14%
  ep_len   21.1±13.1s  (n=508, min=0.5s, max=50.0s)
  reward   get_possession=+401.00  lose_possession=-1.80  ball_out=-244.00  box_possession=+725.00
           speed_bonus=+535.74  timeout=-37.50  stamina_penalty=-2.78
  rew/ep   (mean/std/min/max per episode, 508 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.789    0.417    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.004    0.056    -0.900    +0.000
  ball_out          -0.480    1.300    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.427    1.237    +0.000    +2.500
  speed_bonus       +1.055    1.244    +0.000    +4.058
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.074    0.324    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.005    0.006    -0.024    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     406    +0.011    0.106     +3.050     1.782     +0.888      3.5415      1.606     3.489
  lose_possession       2    -0.000    0.007     +0.160     0.211     -2.896      8.3967      2.896     2.984
  ball_out            61    -0.007    0.165     -3.967     0.178     -4.203     18.5161      4.203     5.663
  box_possession     290    +0.020    0.223     +4.345     1.111     +1.487      3.3330      1.532     3.101
  speed_bonus        282    +0.015    0.193     +4.394     1.085     +1.522      3.4108      1.559     3.103
  timeout             25    -0.001    0.040     -1.506     0.006     -3.701     14.6746      3.701     4.884
  stamina_penalty     297    -0.000    0.001     +4.038     1.764     +1.192      4.1627      1.694     3.515
  gae/td   mean_return=+2.184  std_return=1.647  mean_gae=+0.037  mean_sq_td=1.6050
──────────────────────────────────────────────────────────────────────
2026-08-08 21:09:59,985 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint32.pt
2026-08-08 21:09:59,985 INFO Logging to checkpoints/phase1_run45/training_log33.txt
2026-08-08 21:09:59,986 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 21:10:14,835 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 21:10:14,836 INFO   [eval vs immobile] step=1,224,000  seeds=16x8  win=48%  mean_rew=2.312±3.111  V=2.107  gap=-0.204  outcomes={'other': 22, 'box_possession': 62, 'timeout': 10, 'opponent_box_possession': 1, 'miss': 33}
2026-08-08 21:10:14,838 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 21:10:25,318 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 21:10:25,319 INFO   [eval vs rules] step=1,224,000  seeds=16x8  win=3%  mean_rew=-2.053±1.948  V=1.546  gap=+3.599  outcomes={'other': 16, 'opponent_box_possession': 95, 'box_possession': 4, 'miss': 13}
2026-08-08 21:15:10,222 INFO   [KL mean=0.0967 median=0.0969 > 0.05] ratio percentiles:  p5=0.590  p25=0.864  p50=0.973  p75=1.030  p95=1.218  max=6.382
  move_dir_log_std=[-1.61817467212677]  kick_dir_log_std=[-1.6136665344238281]
2026-08-08 21:15:10,235 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.694  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.435  kick=-0.493  t_att=-0.664
    move_dir=0.558 (min=-11.972 max=1.398)  kick_dir=0.226 (min=-2.916 max=2.083)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.51
  [worst sample] idx=205  ratio=7.148  adv=+1.012  old_lp=-3.048  new_lp=-1.081
    stored move_dir=-8.7°  new_mean=0.2°  angular_diff=8.9°
    [worst sample per-head delta, sorted by |delta|] tackle_attempt:-0.025
  [top-2 highest-ratio samples]
    idx= 205  ratio=   7.148  adv=+1.012  lp: old=-3.048  new=-1.081
      rew=+0.0000  ret=+4.8676  val=+3.8553  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: tackle_attempt:-0.025
      saturation: exec_move_p_new=0.8649  sprint_p_new=0.6880  kick_p_new=0.0096  tackle_attempt_p_new=0.2695
    idx= 201  ratio=   7.071  adv=+0.883  lp: old=-3.904  new=-1.948
      rew=+0.0000  ret=+4.6557  val=+3.7725  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: tackle_attempt:+0.065
      saturation: exec_move_p_new=0.8499  sprint_p_new=0.6625  kick_p_new=0.0094  tackle_attempt_p_new=0.2699
  [best sample (highest new_lp)] idx=172  new_lp=-0.999  adv=+0.691  stored move_dir=-22.0°  new_mean=-14.3°
    per-head contributions: kick:-0.035  tackle_attempt:-0.350  move:-0.614
2026-08-08 21:15:10,236 INFO   [advantage] mean=0.000  std=1.000  min=-5.519  max=3.141
2026-08-08 21:15:10,237 INFO   [ratio] mean=0.9426  std=0.2088  min=0.0017  max=6.3816  clipped=31.4%
2026-08-08 21:15:10,237 INFO   [exec head grad norm] move_direction=0.023  exec_move=0.035  sprint=0.042  kick=0.037  kick_direction=0.010  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.061
2026-08-08 21:15:10,237 INFO   [exec continuous log_std] move_direction: start=-1.6191 end=-1.6182   kick_direction: start=-1.6149 end=-1.6137
2026-08-08 21:15:10,237 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0019≈0.11°/step  epoch≈6.4°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0020≈0.11°/step  epoch≈6.7°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 21:15:10,237 INFO   [exec discrete Δlogit per opt step] exec_move=0.0009  sprint=0.0011  kick=0.0015  tackle_attempt=0.0011
2026-08-08 21:15:10,237 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0025  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0031  sprint=+0.0033  kick=+0.0045  tackle_attempt=+0.0019  move_dir=+0.0534  kick_dir=+0.0281
2026-08-08 21:15:10,238 INFO   [grad clip] main: 1/60 steps clipped (2%)  pre-clip norm mean=0.237 max=0.427  limit=0.4
              direction: 60/60 steps clipped (100%)  pre-clip norm mean=0.029 max=0.051  limit=0.02
2026-08-08 21:15:10,293 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=1,260,000  speed=866/s  reward=1.59
  loss     policy=0.0137  value=0.5874(x0.5)=0.2937
           entropy=6.8152  kl=0.0967
  value    V=2.18±1.01  R=2.12±1.67  adv=-0.06±1.29
  moves    mv_ls=[-1.6182] (σ≈0.20, ≈11°) g=1.26e-02  d_move=[+0.0010] (Δσ≈0.011°)
           kk_ls=[-1.6137] (σ≈0.20, ≈11°)  d_kick=[+0.0012] (Δσ≈0.014°)
  heads    move= 51 get_poss= 41 exec_move= 79 sprint= 44 kick= 26 tackle= 37 shoot= 46 hold= 46 tackle_prob=0.3782 kick_prob=0.2624
  vs       vs[win/loss/tout/miss]  vs_immobile(499): 54.3%/0.2%/6.8%/24.2%/14%
  ep_len   21.5±13.5s  (n=499, min=0.7s, max=50.0s)
  reward   get_possession=+406.00  lose_possession=-3.60  ball_out=-280.00  box_possession=+677.50
           speed_bonus=+497.62  opponent_box=-3.00  timeout=-51.00  stamina_penalty=-2.75
  rew/ep   (mean/std/min/max per episode, 499 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.814    0.409    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.007    0.080    -0.900    +0.000
  ball_out          -0.561    1.389    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.358    1.245    +0.000    +2.500
  speed_bonus       +0.997    1.242    +0.000    +4.111
  opponent_box      -0.006    0.134    -3.000    +0.000
  timeout           -0.102    0.378    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.025    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     408    +0.011    0.106     +2.855     2.042     +0.603      3.8832      1.637     3.842
  lose_possession       4    -0.000    0.009     +2.941     0.493     +0.014      0.6005      0.702     1.113
  ball_out            70    -0.008    0.176     -3.971     0.167     -4.237     19.2956      4.237     5.741
  box_possession     271    +0.019    0.216     +4.327     1.138     +1.421      3.2475      1.505     3.316
  speed_bonus        260    +0.014    0.187     +4.404     1.096     +1.481      3.3721      1.553     3.350
  opponent_box         1    -0.000    0.016     -3.000     0.000     -4.579     20.9690      4.579     4.579
  timeout             34    -0.001    0.046     -1.506     0.005     -4.027     17.1273      4.027     5.146
  stamina_penalty     295    -0.000    0.001     +3.742     2.107     +0.867      4.7415      1.786     4.207
  gae/td   mean_return=+2.123  std_return=1.673  mean_gae=-0.058  mean_sq_td=1.6582
──────────────────────────────────────────────────────────────────────
2026-08-08 21:15:10,321 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint33.pt
2026-08-08 21:15:10,322 INFO Logging to checkpoints/phase1_run45/training_log34.txt
2026-08-08 21:15:10,323 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 21:15:26,409 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 21:15:26,410 INFO   [eval vs immobile] step=1,260,000  seeds=16x8  win=42%  mean_rew=1.745±3.121  V=1.904  gap=+0.159  outcomes={'other': 22, 'box_possession': 54, 'miss': 43, 'timeout': 9}
2026-08-08 21:15:26,412 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 21:15:37,818 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 21:15:37,820 INFO   [eval vs rules] step=1,260,000  seeds=16x8  win=2%  mean_rew=-2.150±1.793  V=1.457  gap=+3.607  outcomes={'opponent_box_possession': 99, 'other': 12, 'miss': 14, 'box_possession': 3}
2026-08-08 21:20:23,017 INFO   [KL mean=0.0932 median=0.0936 > 0.05] ratio percentiles:  p5=0.590  p25=0.867  p50=0.981  p75=1.024  p95=1.217  max=8.896
  move_dir_log_std=[-1.6173509359359741]  kick_dir_log_std=[-1.6124716997146606]
2026-08-08 21:20:23,030 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.689  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.438  kick=-0.555  t_att=-0.644
    move_dir=0.654 (min=-3.118 max=1.397)  kick_dir=0.286 (min=-3.597 max=2.055)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.53
  [worst sample] idx=78  ratio=7.877  adv=-3.503  old_lp=-3.100  new_lp=-1.036
    stored move_dir=156.2°  new_mean=154.8°  angular_diff=1.5°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx=  78  ratio=   7.877  adv=-3.503  lp: old=-3.100  new=-1.036
      rew=+0.0000  ret=-0.3000  val=+3.2032  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8617  sprint_p_new=0.7254  kick_p_new=0.0127  tackle_attempt_p_new=0.3138
    idx=  59  ratio=   7.750  adv=-1.722  lp: old=-3.093  new=-1.045
      rew=+0.0000  ret=+1.1462  val=+2.8684  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8615  sprint_p_new=0.7577  kick_p_new=0.0195  tackle_attempt_p_new=0.3355
  [best sample (highest new_lp)] idx=69  new_lp=-1.001  adv=-2.611  stored move_dir=169.9°  new_mean=158.1°
    per-head contributions: tackle_attempt:-0.354  move:-0.630
2026-08-08 21:20:23,031 INFO   [advantage] mean=0.000  std=1.000  min=-4.314  max=3.834
2026-08-08 21:20:23,032 INFO   [ratio] mean=0.9445  std=0.2088  min=0.0047  max=8.8962  clipped=30.9%
2026-08-08 21:20:23,032 INFO   [exec head grad norm] move_direction=0.024  exec_move=0.039  sprint=0.042  kick=0.037  kick_direction=0.013  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.056
2026-08-08 21:20:23,032 INFO   [exec continuous log_std] move_direction: start=-1.6182 end=-1.6174   kick_direction: start=-1.6137 end=-1.6125
2026-08-08 21:20:23,032 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0020≈0.12°/step  epoch≈6.9°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0019≈0.11°/step  epoch≈6.6°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 21:20:23,033 INFO   [exec discrete Δlogit per opt step] exec_move=0.0008  sprint=0.0011  kick=0.0009  tackle_attempt=0.0005
2026-08-08 21:20:23,033 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0026  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0028  sprint=+0.0031  kick=+0.0035  tackle_attempt=+0.0017  move_dir=+0.0510  kick_dir=+0.0286
2026-08-08 21:20:23,034 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.240 max=0.348  limit=0.4
              direction: 57/60 steps clipped (95%)  pre-clip norm mean=0.031 max=0.053  limit=0.02
2026-08-08 21:20:23,087 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=1,296,000  speed=866/s  reward=2.29
  loss     policy=0.0126  value=0.6336(x0.5)=0.3168
           entropy=6.8301  kl=0.0932
  value    V=2.12±1.04  R=2.06±1.65  adv=-0.07±1.31
  moves    mv_ls=[-1.6174] (σ≈0.20, ≈11°) g=1.09e-02  d_move=[+0.0008] (Δσ≈0.009°)
           kk_ls=[-1.6125] (σ≈0.20, ≈11°)  d_kick=[+0.0012] (Δσ≈0.014°)
  heads    move= 51 get_poss= 40 exec_move= 79 sprint= 44 kick= 27 tackle= 38 shoot= 47 hold= 47 tackle_prob=0.3842 kick_prob=0.2651
  vs       vs[win/loss/tout/miss]  vs_immobile(482): 52.5%/0.2%/9.8%/23.0%/15%
  ep_len   22.4±14.5s  (n=482, min=1.0s, max=50.0s)
  reward   get_possession=+380.00  lose_possession=-0.90  ball_out=-236.00  box_possession=+632.50
           speed_bonus=+467.42  opponent_box=-3.00  timeout=-70.50  stamina_penalty=-2.57
  rew/ep   (mean/std/min/max per episode, 482 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.788    0.414    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.002    0.041    -0.900    +0.000
  ball_out          -0.490    1.311    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.312    1.248    +0.000    +2.500
  speed_bonus       +0.970    1.229    +0.000    +3.990
  opponent_box      -0.006    0.137    -3.000    +0.000
  timeout           -0.146    0.445    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.005    0.006    -0.022    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     384    +0.011    0.103     +2.978     1.935     +0.966      4.0319      1.702     3.812
  lose_possession       1    -0.000    0.005     +2.554     0.000     +0.888      0.7888      0.888     0.888
  ball_out            59    -0.007    0.162     -3.983     0.129     -3.921     16.1376      3.921     5.066
  box_possession     253    +0.018    0.209     +4.338     1.119     +1.485      3.3437      1.541     3.197
  speed_bonus        241    +0.013    0.181     +4.430     1.066     +1.538      3.4729      1.583     3.219
  opponent_box         1    -0.000    0.016     -3.003     0.000     -5.307     28.1597      5.307     5.307
  timeout             47    -0.002    0.054     -1.506     0.005     -3.994     16.7072      3.994     5.144
  stamina_penalty     289    -0.000    0.001     +3.548     2.287     +0.729      5.3802      1.911     4.664
  gae/td   mean_return=+2.056  std_return=1.653  mean_gae=-0.068  mean_sq_td=1.7211
──────────────────────────────────────────────────────────────────────
2026-08-08 21:20:23,113 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint34.pt
2026-08-08 21:20:23,114 INFO Logging to checkpoints/phase1_run45/training_log35.txt
2026-08-08 21:20:23,115 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 21:20:37,660 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 21:20:37,661 INFO   [eval vs immobile] step=1,296,000  seeds=16x8  win=43%  mean_rew=1.961±3.087  V=1.876  gap=-0.085  outcomes={'other': 21, 'miss': 43, 'box_possession': 55, 'timeout': 9}
2026-08-08 21:20:37,663 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 21:20:47,329 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 21:20:47,330 INFO   [eval vs rules] step=1,296,000  seeds=16x8  win=5%  mean_rew=-1.906±2.300  V=1.311  gap=+3.217  outcomes={'other': 14, 'opponent_box_possession': 94, 'miss': 13, 'box_possession': 7}
2026-08-08 21:25:32,025 INFO   [KL mean=0.0857 median=0.0855 > 0.05] ratio percentiles:  p5=0.621  p25=0.873  p50=0.981  p75=1.029  p95=1.210  max=4.932
  move_dir_log_std=[-1.6165132522583008]  kick_dir_log_std=[-1.611326813697815]
2026-08-08 21:25:32,036 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.679  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.476  kick=-0.429  t_att=-0.661
    move_dir=0.580 (min=-4.221 max=1.395)  kick_dir=0.106 (min=-6.704 max=2.027)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.55
  [worst sample] idx=22  ratio=9.716  adv=+1.736  old_lp=-4.211  new_lp=-1.938
    stored move_dir=-1.6°  new_mean=-3.2°  angular_diff=1.6°
    [worst sample per-head delta, sorted by |delta|] tackle_attempt:+0.031
  [top-2 highest-ratio samples]
    idx=  22  ratio=   9.716  adv=+1.736  lp: old=-4.211  new=-1.938
      rew=+0.0000  ret=+5.4406  val=+3.7049  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: tackle_attempt:+0.031
      saturation: exec_move_p_new=0.8957  sprint_p_new=0.7403  kick_p_new=0.0079  tackle_attempt_p_new=0.3160
    idx=  26  ratio=   7.667  adv=+2.162  lp: old=-3.054  new=-1.017
      rew=+0.0000  ret=+5.7769  val=+3.6153  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8749  sprint_p_new=0.6983  kick_p_new=0.0080  tackle_attempt_p_new=0.3138
  [best sample (highest new_lp)] idx=7  new_lp=-0.964  adv=+0.833  stored move_dir=10.1°  new_mean=32.0°
    per-head contributions: tackle_attempt:-0.349  move:-0.607
2026-08-08 21:25:32,037 INFO   [advantage] mean=-0.000  std=1.000  min=-5.128  max=3.742
2026-08-08 21:25:32,037 INFO   [ratio] mean=0.9490  std=0.1991  min=0.0060  max=4.9315  clipped=29.7%
2026-08-08 21:25:32,038 INFO   [exec head grad norm] move_direction=0.027  exec_move=0.039  sprint=0.043  kick=0.031  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.051
2026-08-08 21:25:32,038 INFO   [exec continuous log_std] move_direction: start=-1.6174 end=-1.6165   kick_direction: start=-1.6125 end=-1.6113
2026-08-08 21:25:32,038 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0018≈0.10°/step  epoch≈6.2°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0019≈0.11°/step  epoch≈6.5°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 21:25:32,038 INFO   [exec discrete Δlogit per opt step] exec_move=0.0009  sprint=0.0014  kick=0.0010  tackle_attempt=0.0007
2026-08-08 21:25:32,038 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0017  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0030  sprint=+0.0036  kick=+0.0028  tackle_attempt=+0.0016  move_dir=+0.0478  kick_dir=+0.0252
2026-08-08 21:25:32,039 INFO   [grad clip] main: 1/60 steps clipped (2%)  pre-clip norm mean=0.247 max=0.515  limit=0.4
              direction: 55/60 steps clipped (92%)  pre-clip norm mean=0.032 max=0.092  limit=0.02
2026-08-08 21:25:32,079 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=1,332,000  speed=870/s  reward=3.80
  loss     policy=0.0124  value=0.5859(x0.5)=0.2929
           entropy=6.8238  kl=0.0857
  value    V=2.12±0.97  R=2.21±1.63  adv=0.09±1.25
  moves    mv_ls=[-1.6165] (σ≈0.20, ≈11°) g=1.10e-02  d_move=[+0.0008] (Δσ≈0.010°)
           kk_ls=[-1.6113] (σ≈0.20, ≈11°)  d_kick=[+0.0011] (Δσ≈0.013°)
  heads    move= 51 get_poss= 40 exec_move= 80 sprint= 46 kick= 25 tackle= 38 shoot= 46 hold= 46 tackle_prob=0.3833 kick_prob=0.2562
  vs       vs[win/loss/tout/miss]  vs_immobile(496): 60.5%/0.0%/5.6%/22.2%/12%
  ep_len   21.5±13.5s  (n=496, min=0.7s, max=50.0s)
  reward   get_possession=+406.00  lose_possession=-0.90  ball_out=-244.00  box_possession=+750.00
           speed_bonus=+562.60  timeout=-42.00  stamina_penalty=-2.92
  rew/ep   (mean/std/min/max per episode, 496 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.819    0.391    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.002    0.040    -0.900    +0.000
  ball_out          -0.492    1.314    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.512    1.222    +0.000    +2.500
  speed_bonus       +1.134    1.262    +0.000    +4.226
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.085    0.346    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.027    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     411    +0.011    0.106     +2.962     1.922     +0.981      4.0843      1.736     3.691
  lose_possession       1    -0.000    0.005     +2.636     0.000     +0.129      0.0167      0.129     0.129
  ball_out            61    -0.007    0.165     -4.000     0.000     -3.910     16.0111      3.910     5.159
  box_possession     300    +0.021    0.227     +4.369     1.109     +1.590      3.8747      1.648     3.561
  speed_bonus        290    +0.016    0.199     +4.430     1.076     +1.638      3.9883      1.685     3.561
  timeout             28    -0.001    0.042     -1.505     0.004     -3.694     14.5612      3.694     4.911
  stamina_penalty     316    -0.000    0.001     +4.001     1.814     +1.254      4.6688      1.804     3.959
  gae/td   mean_return=+2.211  std_return=1.632  mean_gae=+0.091  mean_sq_td=1.5723
──────────────────────────────────────────────────────────────────────
2026-08-08 21:25:32,103 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint35.pt
2026-08-08 21:25:32,104 INFO Logging to checkpoints/phase1_run45/training_log36.txt
2026-08-08 21:25:32,105 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 21:25:47,987 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 21:25:47,988 INFO   [eval vs immobile] step=1,332,000  seeds=16x8  win=47%  mean_rew=2.181±3.071  V=2.009  gap=-0.173  outcomes={'other': 26, 'timeout': 7, 'box_possession': 60, 'miss': 35}
2026-08-08 21:25:47,990 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 21:25:57,751 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 21:25:57,752 INFO   [eval vs rules] step=1,332,000  seeds=16x8  win=5%  mean_rew=-2.043±2.131  V=1.260  gap=+3.303  outcomes={'opponent_box_possession': 98, 'other': 11, 'box_possession': 6, 'miss': 13}
2026-08-08 21:30:40,993 INFO   [KL mean=0.0851 median=0.0854 > 0.05] ratio percentiles:  p5=0.617  p25=0.872  p50=0.981  p75=1.029  p95=1.215  max=4.970
  move_dir_log_std=[-1.6156306266784668]  kick_dir_log_std=[-1.6101292371749878]
2026-08-08 21:30:41,007 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.693  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.451  kick=-0.443  t_att=-0.651
    move_dir=0.696 (min=-2.473 max=1.393)  kick_dir=0.212 (min=-3.382 max=2.070)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.52
  [worst sample] idx=164  ratio=9.730  adv=+3.254  old_lp=-4.147  new_lp=-1.872
    stored move_dir=25.7°  new_mean=7.8°  angular_diff=18.0°
    [worst sample per-head delta, sorted by |delta|] tackle_attempt:+0.063
  [top-2 highest-ratio samples]
    idx= 164  ratio=   9.730  adv=+3.254  lp: old=-4.147  new=-1.872
      rew=+6.0463  ret=+6.0463  val=+2.7922  outcome=terminal:box_possession
      rew_breakdown: box=+2.500  spd=+3.559  stam=-0.013
      head_deltas: tackle_attempt:+0.063
      saturation: exec_move_p_new=0.8907  sprint_p_new=0.7197  kick_p_new=0.0065  tackle_attempt_p_new=0.3290
    idx= 250  ratio=   8.687  adv=-1.693  lp: old=-3.205  new=-1.043
      rew=+0.0000  ret=+1.3629  val=+3.0562  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: tackle_attempt:-0.028
      saturation: exec_move_p_new=0.8908  sprint_p_new=0.7613  kick_p_new=0.0157  tackle_attempt_p_new=0.3454
  [best sample (highest new_lp)] idx=21  new_lp=-0.981  adv=+0.102  stored move_dir=169.2°  new_mean=165.2°
    per-head contributions: tackle_attempt:-0.378  move:-0.593
2026-08-08 21:30:41,008 INFO   [advantage] mean=-0.000  std=1.000  min=-5.224  max=4.114
2026-08-08 21:30:41,009 INFO   [ratio] mean=0.9490  std=0.1986  min=0.0066  max=4.9700  clipped=30.0%
2026-08-08 21:30:41,009 INFO   [exec head grad norm] move_direction=0.025  exec_move=0.034  sprint=0.044  kick=0.030  kick_direction=0.010  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.052
2026-08-08 21:30:41,009 INFO   [exec continuous log_std] move_direction: start=-1.6165 end=-1.6156   kick_direction: start=-1.6113 end=-1.6101
2026-08-08 21:30:41,009 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0016≈0.09°/step  epoch≈5.5°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0018≈0.10°/step  epoch≈6.1°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 21:30:41,009 INFO   [exec discrete Δlogit per opt step] exec_move=0.0007  sprint=0.0010  kick=0.0009  tackle_attempt=0.0008
2026-08-08 21:30:41,009 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0025  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0028  sprint=+0.0029  kick=+0.0031  tackle_attempt=+0.0018  move_dir=+0.0469  kick_dir=+0.0251
2026-08-08 21:30:41,010 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.236 max=0.389  limit=0.4
              direction: 52/60 steps clipped (87%)  pre-clip norm mean=0.031 max=0.079  limit=0.02
2026-08-08 21:30:41,061 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=1,368,000  speed=873/s  reward=2.97
  loss     policy=0.0131  value=0.5707(x0.5)=0.2853
           entropy=6.8485  kl=0.0851
  value    V=2.14±1.06  R=2.18±1.69  adv=0.04±1.27
  moves    mv_ls=[-1.6156] (σ≈0.20, ≈11°) g=1.11e-02  d_move=[+0.0009] (Δσ≈0.010°)
           kk_ls=[-1.6101] (σ≈0.20, ≈11°)  d_kick=[+0.0012] (Δσ≈0.014°)
  heads    move= 51 get_poss= 40 exec_move= 79 sprint= 45 kick= 26 tackle= 39 shoot= 47 hold= 47 tackle_prob=0.3921 kick_prob=0.2624
  vs       vs[win/loss/tout/miss]  vs_immobile(503): 58.8%/0.0%/5.6%/22.9%/13%
  ep_len   21.3±13.2s  (n=503, min=1.2s, max=50.0s)
  reward   get_possession=+413.00  lose_possession=-4.50  ball_out=-252.00  box_possession=+740.00
           speed_bonus=+528.85  timeout=-42.00  stamina_penalty=-2.79
  rew/ep   (mean/std/min/max per episode, 503 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.821    0.408    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.009    0.089    -0.900    +0.000
  ball_out          -0.501    1.324    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.471    1.230    +0.000    +2.500
  speed_bonus       +1.051    1.253    +0.000    +4.242
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.083    0.344    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.025    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     418    +0.012    0.107     +3.056     1.980     +0.932      4.0948      1.740     3.488
  lose_possession       5    -0.000    0.011     +2.311     0.488     -0.060      1.1271      0.931     1.567
  ball_out            63    -0.007    0.167     -3.968     0.175     -4.088     17.8108      4.088     5.915
  box_possession     296    +0.021    0.226     +4.278     1.163     +1.228      2.6684      1.346     3.014
  speed_bonus        285    +0.015    0.193     +4.347     1.130     +1.283      2.7626      1.384     3.016
  timeout             28    -0.001    0.042     -1.507     0.005     -3.572     13.8592      3.572     4.929
  stamina_penalty     305    -0.000    0.001     +3.892     1.902     +0.906      3.6740      1.559     3.749
  gae/td   mean_return=+2.183  std_return=1.690  mean_gae=+0.045  mean_sq_td=1.6167
──────────────────────────────────────────────────────────────────────
2026-08-08 21:30:41,087 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint36.pt
2026-08-08 21:30:41,088 INFO Logging to checkpoints/phase1_run45/training_log37.txt
2026-08-08 21:30:41,089 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 21:30:56,214 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 21:30:56,215 INFO   [eval vs immobile] step=1,368,000  seeds=16x8  win=51%  mean_rew=2.415±3.041  V=2.053  gap=-0.362  outcomes={'other': 24, 'box_possession': 65, 'miss': 32, 'timeout': 7}
2026-08-08 21:30:56,216 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 21:31:06,530 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 21:31:06,532 INFO   [eval vs rules] step=1,368,000  seeds=16x8  win=2%  mean_rew=-2.083±1.816  V=1.190  gap=+3.274  outcomes={'opponent_box_possession': 96, 'other': 15, 'miss': 14, 'box_possession': 3}
2026-08-08 21:35:50,441 INFO   [KL mean=0.0791 median=0.0795 > 0.05] ratio percentiles:  p5=0.627  p25=0.879  p50=0.985  p75=1.026  p95=1.212  max=23.344
  move_dir_log_std=[-1.6147048473358154]  kick_dir_log_std=[-1.6090459823608398]
2026-08-08 21:35:50,455 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.694  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.484  kick=-0.492  t_att=-0.655
    move_dir=0.668 (min=-6.692 max=1.392)  kick_dir=0.278 (min=-2.403 max=2.061)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.50
  [worst sample] idx=97  ratio=14.252  adv=+2.700  old_lp=-3.615  new_lp=-0.958
    stored move_dir=101.2°  new_mean=95.5°  angular_diff=5.7°
    [worst sample per-head delta, sorted by |delta|] tackle_attempt:-0.024
  [top-2 highest-ratio samples]
    idx=  97  ratio=  14.252  adv=+2.700  lp: old=-3.615  new=-0.958
      rew=+4.2934  ret=+4.2934  val=+1.5935  outcome=terminal:box_possession
      rew_breakdown: box=+2.500  spd=+1.803  stam=-0.010
      head_deltas: tackle_attempt:-0.024
      saturation: exec_move_p_new=0.9243  sprint_p_new=0.7725  kick_p_new=0.0056  tackle_attempt_p_new=0.3374
    idx= 187  ratio=   9.188  adv=-1.914  lp: old=-3.219  new=-1.001
      rew=+0.0000  ret=+1.4824  val=+3.3967  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: tackle_attempt:-0.023
      saturation: exec_move_p_new=0.8948  sprint_p_new=0.7305  kick_p_new=0.0094  tackle_attempt_p_new=0.3451
  [best sample (highest new_lp)] idx=97  new_lp=-0.958  adv=+2.700  stored move_dir=101.2°  new_mean=95.5°
    per-head contributions: tackle_attempt:-0.412  move:-0.541
2026-08-08 21:35:50,455 INFO   [advantage] mean=0.000  std=1.000  min=-5.717  max=4.260
2026-08-08 21:35:50,456 INFO   [ratio] mean=0.9527  std=0.2053  min=0.0044  max=23.3439  clipped=28.9%
2026-08-08 21:35:50,456 INFO   [exec head grad norm] move_direction=0.021  exec_move=0.035  sprint=0.045  kick=0.033  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.050
2026-08-08 21:35:50,456 INFO   [exec continuous log_std] move_direction: start=-1.6156 end=-1.6147   kick_direction: start=-1.6101 end=-1.6090
2026-08-08 21:35:50,456 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0013≈0.07°/step  epoch≈4.3°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0014≈0.08°/step  epoch≈4.6°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 21:35:50,456 INFO   [exec discrete Δlogit per opt step] exec_move=0.0006  sprint=0.0010  kick=0.0008  tackle_attempt=0.0007
2026-08-08 21:35:50,456 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0017  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0031  sprint=+0.0029  kick=+0.0036  tackle_attempt=+0.0017  move_dir=+0.0438  kick_dir=+0.0223
2026-08-08 21:35:50,457 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.224 max=0.372  limit=0.4
              direction: 53/60 steps clipped (88%)  pre-clip norm mean=0.027 max=0.049  limit=0.02
2026-08-08 21:35:50,509 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=1,404,000  speed=870/s  reward=3.34
  loss     policy=0.0126  value=0.5538(x0.5)=0.2769
           entropy=6.8352  kl=0.0791
  value    V=2.26±1.06  R=2.30±1.67  adv=0.04±1.23
  moves    mv_ls=[-1.6147] (σ≈0.20, ≈11°) g=1.10e-02  d_move=[+0.0009] (Δσ≈0.011°)
           kk_ls=[-1.6090] (σ≈0.20, ≈11°)  d_kick=[+0.0011] (Δσ≈0.012°)
  heads    move= 51 get_poss= 40 exec_move= 79 sprint= 47 kick= 25 tackle= 39 shoot= 46 hold= 46 tackle_prob=0.3923 kick_prob=0.2511
  vs       vs[win/loss/tout/miss]  vs_immobile(500): 61.2%/0.2%/6.0%/21.6%/11%
  ep_len   21.6±13.1s  (n=500, min=1.1s, max=50.0s)
  reward   get_possession=+422.00  lose_possession=-7.20  ball_out=-244.00  box_possession=+765.00
           speed_bonus=+548.47  opponent_box=-3.00  timeout=-45.00  stamina_penalty=-3.10
  rew/ep   (mean/std/min/max per episode, 500 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.844    0.405    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.014    0.113    -0.900    +0.000
  ball_out          -0.488    1.309    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.530    1.218    +0.000    +2.500
  speed_bonus       +1.097    1.222    +0.000    +4.121
  opponent_box      -0.006    0.134    -3.000    +0.000
  timeout           -0.090    0.356    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.027    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     424    +0.012    0.108     +3.089     1.959     +0.806      3.8467      1.667     3.725
  lose_possession       8    -0.000    0.013     +2.343     1.065     -0.488      1.7540      0.877     2.574
  ball_out            61    -0.007    0.165     -3.951     0.216     -4.024     17.1787      4.024     5.453
  box_possession     306    +0.021    0.230     +4.286     1.088     +1.359      2.9638      1.430     3.087
  speed_bonus        289    +0.015    0.193     +4.388     1.031     +1.405      3.0677      1.473     3.096
  opponent_box         1    -0.000    0.016     -3.001     0.000     -4.953     24.5359      4.953     4.953
  timeout             30    -0.001    0.043     -1.507     0.004     -3.661     15.1265      3.661     5.343
  stamina_penalty     326    -0.000    0.001     +3.812     1.946     +0.924      4.1039      1.640     4.147
  gae/td   mean_return=+2.302  std_return=1.666  mean_gae=+0.043  mean_sq_td=1.5134
──────────────────────────────────────────────────────────────────────
2026-08-08 21:35:50,535 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint37.pt
2026-08-08 21:35:50,536 INFO Logging to checkpoints/phase1_run45/training_log38.txt
2026-08-08 21:35:50,537 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 21:36:04,668 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 21:36:04,669 INFO   [eval vs immobile] step=1,404,000  seeds=16x8  win=47%  mean_rew=2.125±3.242  V=2.144  gap=+0.019  outcomes={'other': 21, 'box_possession': 60, 'miss': 44, 'timeout': 3}
2026-08-08 21:36:04,670 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 21:36:14,795 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 21:36:14,796 INFO   [eval vs rules] step=1,404,000  seeds=16x8  win=5%  mean_rew=-1.928±2.232  V=1.156  gap=+3.084  outcomes={'opponent_box_possession': 94, 'other': 12, 'miss': 15, 'box_possession': 6, 'timeout': 1}
2026-08-08 21:40:57,701 INFO   [KL mean=0.0754 median=0.0755 > 0.05] ratio percentiles:  p5=0.643  p25=0.883  p50=0.983  p75=1.028  p95=1.208  max=10.756
  move_dir_log_std=[-1.6137926578521729]  kick_dir_log_std=[-1.6079727411270142]
2026-08-08 21:40:57,713 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.679  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.489  kick=-0.558  t_att=-0.648
    move_dir=0.693 (min=-3.124 max=1.390)  kick_dir=0.276 (min=-4.476 max=2.062)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.52
  [worst sample] idx=159  ratio=7.645  adv=+0.405  old_lp=-6.974  new_lp=-4.940
    stored move_dir=49.2°  new_mean=23.1°  angular_diff=26.1°
    [worst sample per-head delta, sorted by |delta|] kick:-0.063
  [top-2 highest-ratio samples]
    idx= 159  ratio=   7.645  adv=+0.405  lp: old=-6.974  new=-4.940
      rew=+0.0000  ret=+3.6242  val=+3.2196  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: kick:-0.063
      saturation: exec_move_p_new=0.8721  sprint_p_new=0.7345  kick_p_new=0.0218  tackle_attempt_p_new=0.3415
    idx= 151  ratio=   6.820  adv=+0.475  lp: old=-2.904  new=-0.984
      rew=+0.0000  ret=+3.3961  val=+2.9209  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8511  sprint_p_new=0.7454  kick_p_new=0.0315  tackle_attempt_p_new=0.2933
  [best sample (highest new_lp)] idx=151  new_lp=-0.984  adv=+0.475  stored move_dir=44.4°  new_mean=34.3°
    per-head contributions: kick:-0.032  tackle_attempt:-0.347  move:-0.605
2026-08-08 21:40:57,713 INFO   [advantage] mean=-0.000  std=1.000  min=-4.793  max=3.857
2026-08-08 21:40:57,714 INFO   [ratio] mean=0.9550  std=0.1935  min=0.0066  max=10.7560  clipped=28.1%
2026-08-08 21:40:57,714 INFO   [exec head grad norm] move_direction=0.026  exec_move=0.040  sprint=0.043  kick=0.032  kick_direction=0.010  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.054
2026-08-08 21:40:57,714 INFO   [exec continuous log_std] move_direction: start=-1.6147 end=-1.6138   kick_direction: start=-1.6090 end=-1.6080
2026-08-08 21:40:57,715 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0017≈0.10°/step  epoch≈5.7°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0018≈0.10°/step  epoch≈6.1°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 21:40:57,715 INFO   [exec discrete Δlogit per opt step] exec_move=0.0008  sprint=0.0011  kick=0.0009  tackle_attempt=0.0009
2026-08-08 21:40:57,715 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0021  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0028  sprint=+0.0028  kick=+0.0030  tackle_attempt=+0.0015  move_dir=+0.0410  kick_dir=+0.0221
2026-08-08 21:40:57,715 INFO   [grad clip] main: 1/60 steps clipped (2%)  pre-clip norm mean=0.252 max=0.834  limit=0.4
              direction: 54/60 steps clipped (90%)  pre-clip norm mean=0.032 max=0.209  limit=0.02
2026-08-08 21:40:57,763 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=1,440,000  speed=870/s  reward=4.37
  loss     policy=0.0117  value=0.5435(x0.5)=0.2717
           entropy=6.8564  kl=0.0754
  value    V=2.27±1.11  R=2.35±1.64  adv=0.08±1.22
  moves    mv_ls=[-1.6138] (σ≈0.20, ≈11°) g=1.11e-02  d_move=[+0.0009] (Δσ≈0.010°)
           kk_ls=[-1.6080] (σ≈0.20, ≈11°)  d_kick=[+0.0011] (Δσ≈0.012°)
  heads    move= 51 get_poss= 40 exec_move= 79 sprint= 47 kick= 25 tackle= 39 shoot= 46 hold= 47 tackle_prob=0.3975 kick_prob=0.2551
  vs       vs[win/loss/tout/miss]  vs_immobile(482): 63.3%/0.4%/7.1%/19.9%/9%
  ep_len   22.2±13.3s  (n=482, min=1.2s, max=50.0s)
  reward   get_possession=+411.00  lose_possession=-5.40  ball_out=-192.00  box_possession=+762.50
           speed_bonus=+557.81  opponent_box=-6.00  timeout=-51.00  stamina_penalty=-3.13
  rew/ep   (mean/std/min/max per episode, 482 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.853    0.383    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.011    0.100    -0.900    +0.000
  ball_out          -0.398    1.198    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.582    1.205    +0.000    +2.500
  speed_bonus       +1.157    1.248    +0.000    +4.184
  opponent_box      -0.012    0.193    -3.000    +0.000
  timeout           -0.106    0.384    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.024    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     414    +0.011    0.107     +3.238     1.827     +0.942      3.6435      1.620     3.632
  lose_possession       6    -0.000    0.012     +2.442     0.779     -0.713      1.4460      0.713     2.275
  ball_out            48    -0.005    0.146     -3.958     0.200     -3.696     14.3585      3.696     5.078
  box_possession     305    +0.021    0.229     +4.326     1.102     +1.278      2.6811      1.353     3.070
  speed_bonus        294    +0.015    0.196     +4.391     1.067     +1.310      2.7463      1.380     3.072
  opponent_box         2    -0.000    0.022     -3.001     0.000     -5.129     26.6879      5.129     5.682
  timeout             34    -0.001    0.046     -1.510     0.006     -4.006     17.2922      4.006     5.452
  stamina_penalty     335    -0.000    0.001     +3.767     2.038     +0.760      4.2251      1.626     4.408
  gae/td   mean_return=+2.349  std_return=1.643  mean_gae=+0.076  mean_sq_td=1.4834
──────────────────────────────────────────────────────────────────────
2026-08-08 21:40:57,789 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint38.pt
2026-08-08 21:40:57,789 INFO Logging to checkpoints/phase1_run45/training_log39.txt
2026-08-08 21:40:57,791 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 21:41:11,081 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 21:41:11,083 INFO   [eval vs immobile] step=1,440,000  seeds=16x8  win=43%  mean_rew=1.740±3.198  V=2.230  gap=+0.490  outcomes={'other': 23, 'box_possession': 55, 'opponent_box_possession': 1, 'miss': 46, 'timeout': 3}
2026-08-08 21:41:11,084 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 21:41:20,815 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 21:41:20,817 INFO   [eval vs rules] step=1,440,000  seeds=16x8  win=2%  mean_rew=-2.360±1.466  V=1.261  gap=+3.621  outcomes={'opponent_box_possession': 102, 'other': 11, 'miss': 13, 'box_possession': 2}
2026-08-08 21:46:04,618 INFO   [KL mean=0.0709 median=0.0709 > 0.05] ratio percentiles:  p5=0.656  p25=0.888  p50=0.987  p75=1.025  p95=1.200  max=9.020
  move_dir_log_std=[-1.6129645109176636]  kick_dir_log_std=[-1.606973648071289]
2026-08-08 21:46:04,629 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.680  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.519  kick=-0.400  t_att=-0.661
    move_dir=0.679 (min=-3.958 max=1.388)  kick_dir=0.226 (min=-5.539 max=2.024)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.47
  [worst sample] idx=214  ratio=10.610  adv=+0.125  old_lp=-4.045  new_lp=-1.683
    stored move_dir=-24.0°  new_mean=-16.5°  angular_diff=7.4°
    [worst sample per-head delta, sorted by |delta|] tackle_attempt:+0.027
  [top-2 highest-ratio samples]
    idx= 214  ratio=  10.610  adv=+0.125  lp: old=-4.045  new=-1.683
      rew=+0.0000  ret=+2.9826  val=+2.8577  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: tackle_attempt:+0.027
      saturation: exec_move_p_new=0.9026  sprint_p_new=0.7353  kick_p_new=0.0069  tackle_attempt_p_new=0.3535
    idx= 142  ratio=   9.936  adv=-0.714  lp: old=-3.986  new=-1.690
      rew=+0.0000  ret=+2.9622  val=+3.6761  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8959  sprint_p_new=0.7156  kick_p_new=0.0059  tackle_attempt_p_new=0.3549
  [best sample (highest new_lp)] idx=132  new_lp=-1.066  adv=-0.168  stored move_dir=-169.9°  new_mean=-172.8°
    per-head contributions: move_dir:0.068  sprint:-0.226  tackle_attempt:-0.302  move:-0.601
2026-08-08 21:46:04,629 INFO   [advantage] mean=-0.000  std=1.000  min=-5.012  max=3.974
2026-08-08 21:46:04,630 INFO   [ratio] mean=0.9568  std=0.1861  min=0.0070  max=9.0204  clipped=26.9%
2026-08-08 21:46:04,630 INFO   [exec head grad norm] move_direction=0.021  exec_move=0.038  sprint=0.047  kick=0.031  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.053
2026-08-08 21:46:04,630 INFO   [exec continuous log_std] move_direction: start=-1.6138 end=-1.6130   kick_direction: start=-1.6080 end=-1.6070
2026-08-08 21:46:04,631 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0013≈0.07°/step  epoch≈4.4°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0014≈0.08°/step  epoch≈4.9°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 21:46:04,631 INFO   [exec discrete Δlogit per opt step] exec_move=0.0007  sprint=0.0009  kick=0.0008  tackle_attempt=0.0005
2026-08-08 21:46:04,631 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0023  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0025  sprint=+0.0026  kick=+0.0027  tackle_attempt=+0.0014  move_dir=+0.0388  kick_dir=+0.0205
2026-08-08 21:46:04,631 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.234 max=0.344  limit=0.4
              direction: 49/60 steps clipped (82%)  pre-clip norm mean=0.026 max=0.043  limit=0.02
2026-08-08 21:46:04,677 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=1,476,000  speed=872/s  reward=2.21
  loss     policy=0.0096  value=0.5903(x0.5)=0.2951
           entropy=6.8556  kl=0.0709
  value    V=2.43±1.06  R=2.42±1.63  adv=-0.01±1.25
  moves    mv_ls=[-1.6130] (σ≈0.20, ≈11°) g=9.33e-03  d_move=[+0.0008] (Δσ≈0.009°)
           kk_ls=[-1.6070] (σ≈0.20, ≈11°)  d_kick=[+0.0010] (Δσ≈0.011°)
  heads    move= 51 get_poss= 40 exec_move= 79 sprint= 47 kick= 25 tackle= 40 shoot= 47 hold= 47 tackle_prob=0.4006 kick_prob=0.2515
  vs       vs[win/loss/tout/miss]  vs_immobile(507): 60.2%/0.0%/5.1%/21.9%/13%
  ep_len   21.2±13.3s  (n=507, min=0.9s, max=50.0s)
  reward   get_possession=+417.00  lose_possession=-4.50  ball_out=-228.00  box_possession=+762.50
           speed_bonus=+550.22  timeout=-39.00  stamina_penalty=-3.06
  rew/ep   (mean/std/min/max per episode, 507 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.822    0.407    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.009    0.089    -0.900    +0.000
  ball_out          -0.450    1.264    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.504    1.224    +0.000    +2.500
  speed_bonus       +1.085    1.231    +0.000    +4.095
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.077    0.331    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.033    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     421    +0.012    0.108     +3.171     1.980     +0.726      3.8365      1.610     4.046
  lose_possession       5    -0.000    0.011     +2.491     0.411     -0.416      0.2652      0.416     0.823
  ball_out            57    -0.006    0.159     -3.947     0.223     -4.129     18.0892      4.129     5.673
  box_possession     305    +0.021    0.229     +4.295     1.106     +1.143      2.5724      1.312     2.910
  speed_bonus        296    +0.015    0.194     +4.349     1.076     +1.187      2.6332      1.332     2.937
  timeout             26    -0.001    0.040     -1.509     0.006     -4.279     19.0351      4.279     5.183
  stamina_penalty     325    -0.000    0.001     +3.869     1.878     +0.743      3.8548      1.548     4.314
  gae/td   mean_return=+2.420  std_return=1.634  mean_gae=-0.009  mean_sq_td=1.5727
──────────────────────────────────────────────────────────────────────
2026-08-08 21:46:04,702 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint39.pt
2026-08-08 21:46:04,703 INFO Logging to checkpoints/phase1_run45/training_log40.txt
2026-08-08 21:46:04,704 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 21:46:19,286 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 21:46:19,288 INFO   [eval vs immobile] step=1,476,000  seeds=16x8  win=46%  mean_rew=2.048±3.207  V=2.104  gap=+0.055  outcomes={'other': 22, 'box_possession': 59, 'miss': 40, 'timeout': 7}
2026-08-08 21:46:19,289 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 21:46:29,106 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 21:46:29,107 INFO   [eval vs rules] step=1,476,000  seeds=16x8  win=7%  mean_rew=-1.815±2.310  V=1.220  gap=+3.035  outcomes={'other': 13, 'opponent_box_possession': 91, 'box_possession': 9, 'miss': 15}
2026-08-08 21:51:13,633 INFO   [KL mean=0.0712 median=0.0714 > 0.05] ratio percentiles:  p5=0.656  p25=0.888  p50=0.983  p75=1.029  p95=1.203  max=4.358
  move_dir_log_std=[-1.6121108531951904]  kick_dir_log_std=[-1.6061550378799438]
2026-08-08 21:51:13,646 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.692  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.520  kick=-0.364  t_att=-0.681
    move_dir=0.718 (min=-3.651 max=1.386)  kick_dir=0.192 (min=-7.410 max=2.025)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.38
  [worst sample] idx=180  ratio=8.599  adv=+0.544  old_lp=-3.294  new_lp=-1.142
    stored move_dir=-22.8°  new_mean=-18.5°  angular_diff=4.3°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 180  ratio=   8.599  adv=+0.544  lp: old=-3.294  new=-1.142
      rew=+0.0000  ret=+3.3393  val=+2.7955  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8782  sprint_p_new=0.7195  kick_p_new=0.0123  tackle_attempt_p_new=0.3780
    idx= 210  ratio=   8.545  adv=+0.791  lp: old=-3.962  new=-1.816
      rew=+0.0000  ret=+4.4974  val=+3.7068  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8773  sprint_p_new=0.6695  kick_p_new=0.0063  tackle_attempt_p_new=0.3630
  [best sample (highest new_lp)] idx=162  new_lp=-0.926  adv=-0.214  stored move_dir=73.4°  new_mean=77.6°
    per-head contributions: tackle_attempt:-0.376  move:-0.544
2026-08-08 21:51:13,647 INFO   [advantage] mean=0.000  std=1.000  min=-5.010  max=4.264
2026-08-08 21:51:13,648 INFO   [ratio] mean=0.9571  std=0.1853  min=0.0037  max=4.3575  clipped=26.9%
2026-08-08 21:51:13,648 INFO   [exec head grad norm] move_direction=0.027  exec_move=0.041  sprint=0.049  kick=0.032  kick_direction=0.008  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.051
2026-08-08 21:51:13,648 INFO   [exec continuous log_std] move_direction: start=-1.6130 end=-1.6121   kick_direction: start=-1.6070 end=-1.6062
2026-08-08 21:51:13,648 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0014≈0.08°/step  epoch≈4.9°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0014≈0.08°/step  epoch≈4.8°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-08 21:51:13,648 INFO   [exec discrete Δlogit per opt step] exec_move=0.0011  sprint=0.0011  kick=0.0013  tackle_attempt=0.0005
2026-08-08 21:51:13,649 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0019  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0031  sprint=+0.0025  kick=+0.0027  tackle_attempt=+0.0012  move_dir=+0.0401  kick_dir=+0.0199
2026-08-08 21:51:13,649 INFO   [grad clip] main: 1/60 steps clipped (2%)  pre-clip norm mean=0.254 max=1.713  limit=0.4
              direction: 48/60 steps clipped (80%)  pre-clip norm mean=0.031 max=0.320  limit=0.02
2026-08-08 21:51:13,704 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=1,512,000  speed=871/s  reward=3.44
  loss     policy=0.0093  value=0.5754(x0.5)=0.2877
           entropy=6.8433  kl=0.0712
  value    V=2.44±1.04  R=2.49±1.72  adv=0.05±1.30
  moves    mv_ls=[-1.6121] (σ≈0.20, ≈11°) g=9.86e-03  d_move=[+0.0009] (Δσ≈0.010°)
           kk_ls=[-1.6062] (σ≈0.20, ≈11°)  d_kick=[+0.0008] (Δσ≈0.009°)
  heads    move= 50 get_poss= 40 exec_move= 80 sprint= 49 kick= 24 tackle= 40 shoot= 46 hold= 47 tackle_prob=0.4002 kick_prob=0.2398
  vs       vs[win/loss/tout/miss]  vs_immobile(534): 63.1%/0.4%/3.0%/21.7%/12%
  ep_len   20.1±12.2s  (n=534, min=1.4s, max=50.0s)
  reward   get_possession=+450.00  lose_possession=-3.60  ball_out=-280.00  box_possession=+842.50
           speed_bonus=+651.10  opponent_box=-6.00  timeout=-24.00  stamina_penalty=-3.36
  rew/ep   (mean/std/min/max per episode, 534 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.843    0.384    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.007    0.078    -0.900    +0.000
  ball_out          -0.524    1.350    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.578    1.206    +0.000    +2.500
  speed_bonus       +1.219    1.306    +0.000    +4.127
  opponent_box      -0.011    0.183    -3.000    +0.000
  timeout           -0.045    0.256    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.026    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     454    +0.013    0.112     +3.176     2.035     +0.746      4.0703      1.673     4.088
  lose_possession       4    -0.000    0.009     +2.709     0.368     -0.412      0.3803      0.468     1.026
  ball_out            70    -0.008    0.176     -3.986     0.119     -4.024     17.0103      4.024     5.505
  box_possession     337    +0.023    0.241     +4.426     1.153     +1.590      3.7982      1.650     3.343
  speed_bonus        328    +0.018    0.217     +4.479     1.123     +1.636      3.8923      1.682     3.346
  opponent_box         2    -0.000    0.022     -3.005     0.001     -5.005     25.3796      5.005     5.520
  timeout             16    -0.001    0.032     -1.511     0.007     -4.280     19.3363      4.280     5.406
  stamina_penalty     346    -0.000    0.001     +4.159     1.716     +1.315      4.6257      1.787     4.033
  gae/td   mean_return=+2.489  std_return=1.719  mean_gae=+0.046  mean_sq_td=1.6828
──────────────────────────────────────────────────────────────────────
2026-08-08 21:51:13,729 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint40.pt
2026-08-08 21:51:13,729 INFO Logging to checkpoints/phase1_run45/training_log41.txt
2026-08-08 21:51:13,730 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 21:51:27,831 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 21:51:27,833 INFO   [eval vs immobile] step=1,512,000  seeds=16x8  win=49%  mean_rew=2.297±3.206  V=2.084  gap=-0.213  outcomes={'other': 21, 'box_possession': 63, 'timeout': 5, 'miss': 39}
2026-08-08 21:51:27,835 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 21:51:38,187 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 21:51:38,189 INFO   [eval vs rules] step=1,512,000  seeds=16x8  win=4%  mean_rew=-1.969±2.118  V=1.258  gap=+3.227  outcomes={'other': 13, 'opponent_box_possession': 93, 'box_possession': 5, 'timeout': 1, 'miss': 16}
2026-08-08 21:56:25,336 INFO   [KL mean=0.0718 median=0.0718 > 0.05] ratio percentiles:  p5=0.659  p25=0.887  p50=0.985  p75=1.028  p95=1.200  max=7.842
  move_dir_log_std=[-1.6112412214279175]  kick_dir_log_std=[-1.6052346229553223]
2026-08-08 21:56:25,347 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.691  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.454  kick=-0.473  t_att=-0.676
    move_dir=0.661 (min=-3.107 max=1.385)  kick_dir=0.311 (min=-1.534 max=1.980)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.54
  [worst sample] idx=144  ratio=8.469  adv=-1.290  old_lp=-3.715  new_lp=-1.578
    stored move_dir=161.2°  new_mean=168.9°  angular_diff=7.7°
    [worst sample per-head delta, sorted by |delta|] tackle_attempt:+0.049
  [top-2 highest-ratio samples]
    idx= 144  ratio=   8.469  adv=-1.290  lp: old=-3.715  new=-1.578
      rew=+0.0000  ret=+2.8397  val=+4.1300  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: tackle_attempt:+0.049
      saturation: exec_move_p_new=0.8746  sprint_p_new=0.6820  kick_p_new=0.0061  tackle_attempt_p_new=0.3853
    idx= 150  ratio=   8.079  adv=-1.333  lp: old=-3.685  new=-1.595
      rew=+0.0000  ret=+2.6917  val=+4.0250  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: tackle_attempt:+0.043
      saturation: exec_move_p_new=0.8697  sprint_p_new=0.6578  kick_p_new=0.0065  tackle_attempt_p_new=0.3778
  [best sample (highest new_lp)] idx=134  new_lp=-1.125  adv=-0.844  stored move_dir=-145.1°  new_mean=-173.0°
    per-head contributions: tackle_attempt:-0.473  move:-0.644
2026-08-08 21:56:25,347 INFO   [advantage] mean=-0.000  std=1.000  min=-5.796  max=3.702
2026-08-08 21:56:25,348 INFO   [ratio] mean=0.9564  std=0.1865  min=0.0026  max=7.8423  clipped=26.9%
2026-08-08 21:56:25,348 INFO   [exec head grad norm] move_direction=0.025  exec_move=0.038  sprint=0.047  kick=0.033  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.053
2026-08-08 21:56:25,348 INFO   [exec continuous log_std] move_direction: start=-1.6121 end=-1.6112   kick_direction: start=-1.6062 end=-1.6052
2026-08-08 21:56:25,348 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0017≈0.09°/step  epoch≈5.7°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0017≈0.10°/step  epoch≈6.0°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 21:56:25,349 INFO   [exec discrete Δlogit per opt step] exec_move=0.0008  sprint=0.0011  kick=0.0010  tackle_attempt=0.0008
2026-08-08 21:56:25,349 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0018  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0023  sprint=+0.0026  kick=+0.0031  tackle_attempt=+0.0018  move_dir=+0.0400  kick_dir=+0.0201
2026-08-08 21:56:25,349 INFO   [grad clip] main: 1/60 steps clipped (2%)  pre-clip norm mean=0.239 max=0.436  limit=0.4
              direction: 58/60 steps clipped (97%)  pre-clip norm mean=0.030 max=0.071  limit=0.02
2026-08-08 21:56:25,403 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=1,548,000  speed=870/s  reward=2.61
  loss     policy=0.0102  value=0.5492(x0.5)=0.2746
           entropy=6.8691  kl=0.0718
  value    V=2.44±1.09  R=2.46±1.64  adv=0.02±1.22
  moves    mv_ls=[-1.6112] (σ≈0.20, ≈11°) g=1.09e-02  d_move=[+0.0009] (Δσ≈0.010°)
           kk_ls=[-1.6052] (σ≈0.20, ≈12°)  d_kick=[+0.0009] (Δσ≈0.011°)
  heads    move= 51 get_poss= 39 exec_move= 79 sprint= 47 kick= 25 tackle= 40 shoot= 47 hold= 47 tackle_prob=0.4096 kick_prob=0.2501
  vs       vs[win/loss/tout/miss]  vs_immobile(502): 64.1%/0.0%/5.2%/20.1%/11%
  ep_len   21.3±12.7s  (n=502, min=1.2s, max=50.0s)
  reward   get_possession=+424.00  lose_possession=-3.60  ball_out=-224.00  box_possession=+805.00
           speed_bonus=+579.35  timeout=-39.00  stamina_penalty=-3.17
  rew/ep   (mean/std/min/max per episode, 502 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.845    0.384    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.007    0.080    -0.900    +0.000
  ball_out          -0.446    1.259    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.604    1.199    +0.000    +2.500
  speed_bonus       +1.154    1.232    +0.000    +3.896
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.078    0.332    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.035    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     428    +0.012    0.108     +3.214     1.921     +0.665      3.5087      1.548     3.931
  lose_possession       4    -0.000    0.009     +2.500     0.341     -0.882      0.9089      0.882     1.267
  ball_out            56    -0.006    0.158     -4.000     0.000     -3.862     16.3063      3.862     6.406
  box_possession     322    +0.022    0.235     +4.290     1.096     +1.397      3.0369      1.468     3.134
  speed_bonus        305    +0.016    0.199     +4.390     1.038     +1.460      3.1706      1.516     3.181
  timeout             26    -0.001    0.040     -1.510     0.006     -4.117     18.3120      4.117     5.421
  stamina_penalty     337    -0.000    0.001     +3.901     1.852     +1.011      4.2347      1.683     4.009
  gae/td   mean_return=+2.459  std_return=1.639  mean_gae=+0.018  mean_sq_td=1.4828
──────────────────────────────────────────────────────────────────────
2026-08-08 21:56:25,430 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint41.pt
2026-08-08 21:56:25,431 INFO Logging to checkpoints/phase1_run45/training_log42.txt
2026-08-08 21:56:25,432 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 21:56:39,523 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 21:56:39,524 INFO   [eval vs immobile] step=1,548,000  seeds=16x8  win=55%  mean_rew=2.760±3.061  V=2.240  gap=-0.520  outcomes={'other': 23, 'box_possession': 71, 'timeout': 2, 'miss': 32}
2026-08-08 21:56:39,526 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 21:56:49,714 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 21:56:49,716 INFO   [eval vs rules] step=1,548,000  seeds=16x8  win=5%  mean_rew=-1.888±2.240  V=1.239  gap=+3.127  outcomes={'opponent_box_possession': 94, 'other': 12, 'box_possession': 7, 'miss': 15}
2026-08-08 22:01:32,693 INFO   [KL mean=0.0708 median=0.0706 > 0.05] ratio percentiles:  p5=0.663  p25=0.889  p50=0.987  p75=1.024  p95=1.192  max=5.738
  move_dir_log_std=[-1.6104376316070557]  kick_dir_log_std=[-1.604235291481018]
2026-08-08 22:01:32,705 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.689  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.481  kick=-0.371  t_att=-0.663
    move_dir=0.627 (min=-7.808 max=1.383)  kick_dir=0.202 (min=-4.135 max=2.044)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.53
  [worst sample] idx=191  ratio=9.367  adv=+1.389  old_lp=-3.457  new_lp=-1.220
    stored move_dir=171.4°  new_mean=179.1°  angular_diff=7.6°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 191  ratio=   9.367  adv=+1.389  lp: old=-3.457  new=-1.220
      rew=+0.0000  ret=+5.4967  val=+4.1081  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8985  sprint_p_new=0.7151  kick_p_new=0.0042  tackle_attempt_p_new=0.3915
    idx= 186  ratio=   8.825  adv=+1.061  lp: old=-3.758  new=-1.581
      rew=+0.0000  ret=+5.1722  val=+4.1114  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8911  sprint_p_new=0.6617  kick_p_new=0.0048  tackle_attempt_p_new=0.3928
  [best sample (highest new_lp)] idx=128  new_lp=-1.080  adv=+0.377  stored move_dir=-29.7°  new_mean=-41.6°
    per-head contributions: tackle_attempt:-0.430  move:-0.642
2026-08-08 22:01:32,705 INFO   [advantage] mean=-0.000  std=1.000  min=-5.569  max=4.047
2026-08-08 22:01:32,706 INFO   [ratio] mean=0.9560  std=0.1779  min=0.0045  max=5.7376  clipped=26.1%
2026-08-08 22:01:32,706 INFO   [exec head grad norm] move_direction=0.023  exec_move=0.038  sprint=0.047  kick=0.034  kick_direction=0.010  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.047
2026-08-08 22:01:32,706 INFO   [exec continuous log_std] move_direction: start=-1.6112 end=-1.6104   kick_direction: start=-1.6052 end=-1.6042
2026-08-08 22:01:32,707 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0019≈0.11°/step  epoch≈6.5°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0019≈0.11°/step  epoch≈6.6°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 22:01:32,707 INFO   [exec discrete Δlogit per opt step] exec_move=0.0008  sprint=0.0011  kick=0.0008  tackle_attempt=0.0005
2026-08-08 22:01:32,707 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0017  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0029  sprint=+0.0025  kick=+0.0035  tackle_attempt=+0.0015  move_dir=+0.0379  kick_dir=+0.0208
2026-08-08 22:01:32,707 INFO   [grad clip] main: 1/60 steps clipped (2%)  pre-clip norm mean=0.238 max=0.477  limit=0.4
              direction: 52/60 steps clipped (87%)  pre-clip norm mean=0.028 max=0.053  limit=0.02
2026-08-08 22:01:32,760 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=1,584,000  speed=873/s  reward=0.78
  loss     policy=0.0102  value=0.5662(x0.5)=0.2831
           entropy=6.8738  kl=0.0708
  value    V=2.43±1.06  R=2.42±1.69  adv=-0.01±1.28
  moves    mv_ls=[-1.6104] (σ≈0.20, ≈11°) g=9.89e-03  d_move=[+0.0008] (Δσ≈0.009°)
           kk_ls=[-1.6042] (σ≈0.20, ≈12°)  d_kick=[+0.0010] (Δσ≈0.012°)
  heads    move= 50 get_poss= 39 exec_move= 79 sprint= 48 kick= 25 tackle= 41 shoot= 47 hold= 47 tackle_prob=0.4119 kick_prob=0.2505
  vs       vs[win/loss/tout/miss]  vs_immobile(512): 61.9%/0.0%/5.5%/22.3%/10%
  ep_len   21.0±12.7s  (n=512, min=1.6s, max=50.0s)
  reward   get_possession=+429.00  lose_possession=-3.60  ball_out=-272.00  box_possession=+792.50
           speed_bonus=+579.11  timeout=-42.00  stamina_penalty=-3.16
  rew/ep   (mean/std/min/max per episode, 512 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.838    0.389    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.007    0.079    -0.900    +0.000
  ball_out          -0.531    1.357    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.548    1.214    +0.000    +2.500
  speed_bonus       +1.131    1.220    +0.000    +4.136
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.082    0.341    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.024    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     431    +0.012    0.109     +3.076     2.052     +0.645      3.9179      1.630     3.968
  lose_possession       4    -0.000    0.009     +2.622     0.493     -0.187      0.1467      0.313     0.598
  ball_out            68    -0.008    0.174     -3.941     0.235     -4.068     17.7591      4.068     6.042
  box_possession     317    +0.022    0.234     +4.321     1.061     +1.166      2.4127      1.277     2.910
  speed_bonus        303    +0.016    0.198     +4.405     1.008     +1.218      2.5124      1.318     2.933
  timeout             28    -0.001    0.042     -1.510     0.006     -4.264     19.0009      4.264     5.392
  stamina_penalty     339    -0.000    0.001     +3.885     1.860     +0.754      3.7377      1.519     4.087
  gae/td   mean_return=+2.419  std_return=1.689  mean_gae=-0.007  mean_sq_td=1.6352
──────────────────────────────────────────────────────────────────────
2026-08-08 22:01:32,783 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint42.pt
2026-08-08 22:01:32,783 INFO Logging to checkpoints/phase1_run45/training_log43.txt
2026-08-08 22:01:32,784 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 22:01:48,113 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 22:01:48,114 INFO   [eval vs immobile] step=1,584,000  seeds=16x8  win=50%  mean_rew=2.377±3.162  V=2.180  gap=-0.197  outcomes={'other': 20, 'box_possession': 64, 'miss': 35, 'timeout': 9}
2026-08-08 22:01:48,116 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 22:01:57,803 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 22:01:57,805 INFO   [eval vs rules] step=1,584,000  seeds=16x8  win=5%  mean_rew=-1.983±2.121  V=1.202  gap=+3.184  outcomes={'box_possession': 6, 'opponent_box_possession': 97, 'other': 13, 'miss': 12}
2026-08-08 22:06:40,922 INFO   [KL mean=0.0667 median=0.0667 > 0.05] ratio percentiles:  p5=0.675  p25=0.892  p50=0.986  p75=1.024  p95=1.189  max=8.739
  move_dir_log_std=[-1.609567403793335]  kick_dir_log_std=[-1.6032229661941528]
2026-08-08 22:06:40,937 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.688  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.492  kick=-0.453  t_att=-0.688
    move_dir=0.663 (min=-4.438 max=1.381)  kick_dir=0.246 (min=-1.873 max=2.013)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.52
  [worst sample] idx=79  ratio=7.773  adv=+0.001  old_lp=-3.720  new_lp=-1.669
    stored move_dir=-153.4°  new_mean=-164.7°  angular_diff=11.3°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx=  79  ratio=   7.773  adv=+0.001  lp: old=-3.720  new=-1.669
      rew=+0.0000  ret=+4.1332  val=+4.1318  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8683  sprint_p_new=0.6648  kick_p_new=0.0069  tackle_attempt_p_new=0.3719
    idx=  69  ratio=   7.730  adv=-0.172  lp: old=-3.233  new=-1.188
      rew=+0.0000  ret=+3.8831  val=+4.0554  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8659  sprint_p_new=0.6647  kick_p_new=0.0075  tackle_attempt_p_new=0.3771
  [best sample (highest new_lp)] idx=186  new_lp=-0.981  adv=+0.108  stored move_dir=-31.9°  new_mean=-41.6°
    per-head contributions: kick:-0.038  tackle_attempt:-0.388  move:-0.554
2026-08-08 22:06:40,937 INFO   [advantage] mean=-0.000  std=1.000  min=-5.959  max=4.094
2026-08-08 22:06:40,938 INFO   [ratio] mean=0.9579  std=0.1757  min=0.0050  max=8.7395  clipped=25.5%
2026-08-08 22:06:40,938 INFO   [exec head grad norm] move_direction=0.023  exec_move=0.041  sprint=0.046  kick=0.033  kick_direction=0.010  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.051
2026-08-08 22:06:40,938 INFO   [exec continuous log_std] move_direction: start=-1.6104 end=-1.6096   kick_direction: start=-1.6042 end=-1.6032
2026-08-08 22:06:40,939 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0020≈0.12°/step  epoch≈7.0°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0019≈0.11°/step  epoch≈6.6°  dlog_std=0.00002  Δσ°=0.000/step)
2026-08-08 22:06:40,939 INFO   [exec discrete Δlogit per opt step] exec_move=0.0009  sprint=0.0009  kick=0.0009  tackle_attempt=0.0005
2026-08-08 22:06:40,939 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0017  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0023  sprint=+0.0021  kick=+0.0030  tackle_attempt=+0.0013  move_dir=+0.0376  kick_dir=+0.0188
2026-08-08 22:06:40,939 INFO   [grad clip] main: 1/60 steps clipped (2%)  pre-clip norm mean=0.249 max=0.671  limit=0.4
              direction: 57/60 steps clipped (95%)  pre-clip norm mean=0.029 max=0.095  limit=0.02
2026-08-08 22:06:40,971 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=1,620,000  speed=875/s  reward=3.15
  loss     policy=0.0104  value=0.5793(x0.5)=0.2896
           entropy=6.8940  kl=0.0667
  value    V=2.42±1.07  R=2.34±1.67  adv=-0.08±1.27
  moves    mv_ls=[-1.6096] (σ≈0.20, ≈11°) g=1.06e-02  d_move=[+0.0009] (Δσ≈0.010°)
           kk_ls=[-1.6032] (σ≈0.20, ≈12°)  d_kick=[+0.0010] (Δσ≈0.012°)
  heads    move= 50 get_poss= 39 exec_move= 79 sprint= 46 kick= 26 tackle= 42 shoot= 47 hold= 47 tackle_prob=0.4164 kick_prob=0.2586
  vs       vs[win/loss/tout/miss]  vs_immobile(482): 59.1%/0.0%/6.6%/23.9%/10%
  ep_len   22.2±13.5s  (n=482, min=1.5s, max=50.0s)
  reward   get_possession=+413.00  lose_possession=-5.40  ball_out=-280.00  box_possession=+712.50
           speed_bonus=+498.85  timeout=-48.00  stamina_penalty=-2.86
  rew/ep   (mean/std/min/max per episode, 482 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.857    0.384    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.011    0.100    -0.900    +0.000
  ball_out          -0.581    1.409    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.478    1.229    +0.000    +2.500
  speed_bonus       +1.035    1.210    +0.000    +3.922
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.100    0.373    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.024    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     417    +0.012    0.107     +2.968     2.154     +0.608      4.0341      1.664     3.912
  lose_possession       6    -0.000    0.012     +2.588     0.493     -0.624      0.6177      0.632     1.295
  ball_out            70    -0.008    0.176     -3.986     0.119     -3.875     16.1326      3.875     5.321
  box_possession     285    +0.020    0.222     +4.248     1.101     +0.993      2.1734      1.207     2.628
  speed_bonus        270    +0.014    0.184     +4.346     1.048     +1.051      2.2793      1.249     2.725
  timeout             32    -0.001    0.045     -1.506     0.005     -3.993     17.2851      3.993     5.660
  stamina_penalty     310    -0.000    0.001     +3.724     1.980     +0.527      3.6717      1.482     4.190
  gae/td   mean_return=+2.343  std_return=1.667  mean_gae=-0.079  mean_sq_td=1.6113
──────────────────────────────────────────────────────────────────────
2026-08-08 22:06:40,997 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint43.pt
2026-08-08 22:06:40,997 INFO Logging to checkpoints/phase1_run45/training_log44.txt
2026-08-08 22:06:40,998 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 22:06:55,777 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 22:06:55,779 INFO   [eval vs immobile] step=1,620,000  seeds=16x8  win=52%  mean_rew=2.599±3.216  V=2.111  gap=-0.488  outcomes={'other': 20, 'box_possession': 67, 'timeout': 7, 'miss': 34}
2026-08-08 22:06:55,781 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 22:07:06,649 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 22:07:06,650 INFO   [eval vs rules] step=1,620,000  seeds=16x8  win=3%  mean_rew=-2.002±1.953  V=1.007  gap=+3.009  outcomes={'opponent_box_possession': 93, 'other': 11, 'box_possession': 4, 'timeout': 1, 'miss': 19}
2026-08-08 22:11:56,953 INFO   [KL mean=0.0657 median=0.0658 > 0.05] ratio percentiles:  p5=0.681  p25=0.896  p50=0.987  p75=1.025  p95=1.188  max=2.964
  move_dir_log_std=[-1.608818769454956]  kick_dir_log_std=[-1.6023802757263184]
2026-08-08 22:11:56,964 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.694  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.431  kick=-0.489  t_att=-0.664
    move_dir=0.544 (min=-5.745 max=1.380)  kick_dir=0.201 (min=-3.186 max=1.976)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.55
  [worst sample] idx=109  ratio=7.461  adv=-0.822  old_lp=-3.699  new_lp=-1.689
    stored move_dir=146.4°  new_mean=161.6°  angular_diff=15.3°
    [worst sample per-head delta, sorted by |delta|] tackle_attempt:+0.026
  [top-2 highest-ratio samples]
    idx= 109  ratio=   7.461  adv=-0.822  lp: old=-3.699  new=-1.689
      rew=+0.0000  ret=+2.7265  val=+3.5485  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: tackle_attempt:+0.026
      saturation: exec_move_p_new=0.8595  sprint_p_new=0.6815  kick_p_new=0.0069  tackle_attempt_p_new=0.3963
    idx=  89  ratio=   7.008  adv=-0.570  lp: old=-3.540  new=-1.593
      rew=+0.0000  ret=+2.7301  val=+3.2997  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: tackle_attempt:+0.030
      saturation: exec_move_p_new=0.8520  sprint_p_new=0.6890  kick_p_new=0.0072  tackle_attempt_p_new=0.3718
  [best sample (highest new_lp)] idx=136  new_lp=-1.045  adv=-0.782  stored move_dir=142.9°  new_mean=131.8°
    per-head contributions: tackle_attempt:-0.454  move:-0.583
2026-08-08 22:11:56,964 INFO   [advantage] mean=0.000  std=1.000  min=-4.918  max=4.515
2026-08-08 22:11:56,965 INFO   [ratio] mean=0.9590  std=0.1712  min=0.0066  max=2.9639  clipped=24.9%
2026-08-08 22:11:56,965 INFO   [exec head grad norm] move_direction=0.024  exec_move=0.041  sprint=0.046  kick=0.030  kick_direction=0.008  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.057
2026-08-08 22:11:56,965 INFO   [exec continuous log_std] move_direction: start=-1.6096 end=-1.6088   kick_direction: start=-1.6032 end=-1.6024
2026-08-08 22:11:56,965 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0012≈0.07°/step  epoch≈4.3°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0014≈0.08°/step  epoch≈5.0°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-08 22:11:56,965 INFO   [exec discrete Δlogit per opt step] exec_move=0.0007  sprint=0.0009  kick=0.0010  tackle_attempt=0.0008
2026-08-08 22:11:56,966 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0017  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0029  sprint=+0.0022  kick=+0.0030  tackle_attempt=+0.0017  move_dir=+0.0359  kick_dir=+0.0183
2026-08-08 22:11:56,966 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.231 max=0.355  limit=0.4
              direction: 46/60 steps clipped (77%)  pre-clip norm mean=0.028 max=0.067  limit=0.02
2026-08-08 22:11:57,019 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=1,656,000  speed=867/s  reward=2.50
  loss     policy=0.0079  value=0.5445(x0.5)=0.2723
           entropy=6.8835  kl=0.0657
  value    V=2.40±1.10  R=2.42±1.73  adv=0.02±1.27
  moves    mv_ls=[-1.6088] (σ≈0.20, ≈11°) g=8.38e-03  d_move=[+0.0007] (Δσ≈0.009°)
           kk_ls=[-1.6024] (σ≈0.20, ≈12°)  d_kick=[+0.0008] (Δσ≈0.010°)
  heads    move= 50 get_poss= 39 exec_move= 79 sprint= 48 kick= 24 tackle= 41 shoot= 47 hold= 48 tackle_prob=0.4203 kick_prob=0.2459
  vs       vs[win/loss/tout/miss]  vs_immobile(524): 59.9%/0.0%/5.2%/23.5%/11%
  ep_len   20.4±12.7s  (n=524, min=0.2s, max=50.0s)
  reward   get_possession=+427.00  lose_possession=-0.90  ball_out=-284.00  box_possession=+785.00
           speed_bonus=+611.04  timeout=-40.50  stamina_penalty=-3.08
  rew/ep   (mean/std/min/max per episode, 524 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.815    0.393    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.002    0.039    -0.900    +0.000
  ball_out          -0.542    1.369    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.498    1.225    +0.000    +2.500
  speed_bonus       +1.166    1.276    +0.000    +4.000
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.077    0.332    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.023    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     433    +0.012    0.109     +3.054     2.165     +0.731      4.1909      1.730     3.958
  lose_possession       1    -0.000    0.005     +3.452     0.000     +0.308      0.0952      0.308     0.308
  ball_out            71    -0.008    0.177     -3.986     0.118     -3.568     13.5302      3.568     4.745
  box_possession     314    +0.022    0.232     +4.437     1.095     +1.292      2.6698      1.369     2.994
  speed_bonus        306    +0.017    0.208     +4.488     1.062     +1.324      2.7308      1.391     3.058
  timeout             27    -0.001    0.041     -1.508     0.005     -4.226     18.8635      4.226     5.440
  stamina_penalty     336    -0.000    0.001     +3.990     1.907     +0.868      3.9522      1.596     3.989
  gae/td   mean_return=+2.421  std_return=1.729  mean_gae=+0.021  mean_sq_td=1.6070
──────────────────────────────────────────────────────────────────────
2026-08-08 22:11:57,045 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint44.pt
2026-08-08 22:11:57,046 INFO Logging to checkpoints/phase1_run45/training_log45.txt
2026-08-08 22:11:57,047 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 22:12:13,192 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 22:12:13,194 INFO   [eval vs immobile] step=1,656,000  seeds=16x8  win=52%  mean_rew=2.397±3.145  V=2.075  gap=-0.323  outcomes={'other': 20, 'box_possession': 66, 'miss': 37, 'timeout': 5}
2026-08-08 22:12:13,195 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 22:12:22,975 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 22:12:22,977 INFO   [eval vs rules] step=1,656,000  seeds=16x8  win=4%  mean_rew=-2.013±2.030  V=1.082  gap=+3.094  outcomes={'other': 14, 'box_possession': 5, 'opponent_box_possession': 94, 'miss': 15}
2026-08-08 22:17:10,069 INFO   [KL mean=0.0650 median=0.0646 > 0.05] ratio percentiles:  p5=0.676  p25=0.895  p50=0.987  p75=1.027  p95=1.190  max=5.624
  move_dir_log_std=[-1.6080477237701416]  kick_dir_log_std=[-1.6015578508377075]
2026-08-08 22:17:10,081 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.686  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.514  kick=-0.594  t_att=-0.674
    move_dir=0.749 (min=-2.794 max=1.378)  kick_dir=0.359 (min=-2.108 max=2.034)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.48
  [worst sample] idx=9  ratio=7.824  adv=-1.007  old_lp=-3.384  new_lp=-1.326
    stored move_dir=5.1°  new_mean=12.0°  angular_diff=6.9°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx=   9  ratio=   7.824  adv=-1.007  lp: old=-3.384  new=-1.326
      rew=+0.0000  ret=+2.2992  val=+3.3062  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8790  sprint_p_new=0.6420  kick_p_new=0.0041  tackle_attempt_p_new=0.4101
    idx=  20  ratio=   7.571  adv=-1.695  lp: old=-3.307  new=-1.282
      rew=+0.0000  ret=+1.9758  val=+3.6713  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8737  sprint_p_new=0.6493  kick_p_new=0.0051  tackle_attempt_p_new=0.3990
  [best sample (highest new_lp)] idx=25  new_lp=-1.135  adv=-2.090  stored move_dir=10.0°  new_mean=10.4°
    per-head contributions: tackle_attempt:-0.497  move:-0.631
2026-08-08 22:17:10,082 INFO   [advantage] mean=0.000  std=1.000  min=-4.993  max=3.636
2026-08-08 22:17:10,083 INFO   [ratio] mean=0.9596  std=0.1755  min=0.0037  max=5.6241  clipped=25.3%
2026-08-08 22:17:10,083 INFO   [exec head grad norm] move_direction=0.025  exec_move=0.038  sprint=0.050  kick=0.035  kick_direction=0.008  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.054
2026-08-08 22:17:10,083 INFO   [exec continuous log_std] move_direction: start=-1.6088 end=-1.6080   kick_direction: start=-1.6024 end=-1.6016
2026-08-08 22:17:10,083 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0014≈0.08°/step  epoch≈4.8°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0013≈0.07°/step  epoch≈4.3°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-08 22:17:10,083 INFO   [exec discrete Δlogit per opt step] exec_move=0.0006  sprint=0.0009  kick=0.0010  tackle_attempt=0.0006
2026-08-08 22:17:10,084 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0017  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0021  sprint=+0.0027  kick=+0.0026  tackle_attempt=+0.0012  move_dir=+0.0364  kick_dir=+0.0183
2026-08-08 22:17:10,084 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.235 max=0.384  limit=0.4
              direction: 52/60 steps clipped (87%)  pre-clip norm mean=0.029 max=0.052  limit=0.02
2026-08-08 22:17:10,126 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=1,692,000  speed=866/s  reward=1.96
  loss     policy=0.0083  value=0.5125(x0.5)=0.2562
           entropy=6.8854  kl=0.0650
  value    V=2.34±1.19  R=2.29±1.77  adv=-0.04±1.27
  moves    mv_ls=[-1.6080] (σ≈0.20, ≈11°) g=9.54e-03  d_move=[+0.0008] (Δσ≈0.009°)
           kk_ls=[-1.6016] (σ≈0.20, ≈12°)  d_kick=[+0.0008] (Δσ≈0.009°)
  heads    move= 50 get_poss= 39 exec_move= 78 sprint= 48 kick= 25 tackle= 42 shoot= 47 hold= 48 tackle_prob=0.4261 kick_prob=0.2491
  vs       vs[win/loss/tout/miss]  vs_immobile(522): 57.7%/0.0%/3.1%/25.7%/14%
  ep_len   20.6±12.6s  (n=522, min=0.8s, max=50.0s)
  reward   get_possession=+429.00  lose_possession=-9.00  ball_out=-320.00  box_possession=+752.50
           speed_bonus=+541.47  timeout=-24.00  stamina_penalty=-2.99
  rew/ep   (mean/std/min/max per episode, 522 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.822    0.421    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.017    0.135    -1.800    +0.000
  ball_out          -0.613    1.441    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.442    1.235    +0.000    +2.500
  speed_bonus       +1.037    1.211    +0.000    +4.032
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.046    0.259    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.022    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     433    +0.012    0.109     +2.863     2.231     +0.491      4.1059      1.675     4.162
  lose_possession      10    -0.000    0.015     +2.131     1.494     -0.873      3.1245      1.043     3.518
  ball_out            80    -0.009    0.188     -3.950     0.218     -3.511     13.3770      3.511     5.274
  box_possession     301    +0.021    0.228     +4.290     1.083     +1.290      2.9232      1.410     3.142
  speed_bonus        292    +0.015    0.191     +4.345     1.051     +1.330      3.0058      1.439     3.160
  timeout             16    -0.001    0.032     -1.509     0.004     -4.385     20.4558      4.385     5.703
  stamina_penalty     310    -0.000    0.001     +4.034     1.632     +1.022      3.8756      1.578     4.088
  gae/td   mean_return=+2.294  std_return=1.770  mean_gae=-0.044  mean_sq_td=1.6097
──────────────────────────────────────────────────────────────────────
2026-08-08 22:17:10,150 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint45.pt
2026-08-08 22:17:10,151 INFO Logging to checkpoints/phase1_run45/training_log46.txt
2026-08-08 22:17:10,152 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 22:17:25,160 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 22:17:25,161 INFO   [eval vs immobile] step=1,692,000  seeds=16x8  win=55%  mean_rew=2.699±3.163  V=2.062  gap=-0.638  outcomes={'other': 19, 'box_possession': 70, 'miss': 34, 'timeout': 5}
2026-08-08 22:17:25,163 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 22:17:34,883 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 22:17:34,885 INFO   [eval vs rules] step=1,692,000  seeds=16x8  win=2%  mean_rew=-2.194±1.640  V=0.899  gap=+3.093  outcomes={'other': 15, 'opponent_box_possession': 91, 'miss': 20, 'box_possession': 2}
2026-08-08 22:22:20,940 INFO   [KL mean=0.0673 median=0.0672 > 0.05] ratio percentiles:  p5=0.675  p25=0.893  p50=0.987  p75=1.026  p95=1.185  max=4.334
  move_dir_log_std=[-1.6071560382843018]  kick_dir_log_std=[-1.6007486581802368]
2026-08-08 22:22:20,952 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.685  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.488  kick=-0.534  t_att=-0.682
    move_dir=0.645 (min=-9.981 max=1.376)  kick_dir=0.384 (min=-1.981 max=2.000)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.49
  [worst sample] idx=212  ratio=8.267  adv=-1.106  old_lp=-3.721  new_lp=-1.609
    stored move_dir=-18.1°  new_mean=-1.7°  angular_diff=16.4°
    [worst sample per-head delta, sorted by |delta|] tackle_attempt:+0.033
  [top-2 highest-ratio samples]
    idx= 212  ratio=   8.267  adv=-1.106  lp: old=-3.721  new=-1.609
      rew=+0.0000  ret=+2.9873  val=+4.0928  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: tackle_attempt:+0.033
      saturation: exec_move_p_new=0.8749  sprint_p_new=0.6499  kick_p_new=0.0046  tackle_attempt_p_new=0.4241
    idx= 216  ratio=   7.038  adv=-1.216  lp: old=-3.142  new=-1.191
      rew=+0.0000  ret=+2.9030  val=+4.1187  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: tackle_attempt:-0.023
      saturation: exec_move_p_new=0.8584  sprint_p_new=0.6367  kick_p_new=0.0062  tackle_attempt_p_new=0.4183
  [best sample (highest new_lp)] idx=132  new_lp=-1.176  adv=-0.965  stored move_dir=165.9°  new_mean=155.8°
    per-head contributions: kick:-0.027  tackle_attempt:-0.470  move:-0.679
2026-08-08 22:22:20,953 INFO   [advantage] mean=0.000  std=1.000  min=-5.634  max=4.093
2026-08-08 22:22:20,954 INFO   [ratio] mean=0.9577  std=0.1722  min=0.0084  max=4.3339  clipped=25.2%
2026-08-08 22:22:20,954 INFO   [exec head grad norm] move_direction=0.024  exec_move=0.041  sprint=0.042  kick=0.034  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.060
2026-08-08 22:22:20,954 INFO   [exec continuous log_std] move_direction: start=-1.6080 end=-1.6072   kick_direction: start=-1.6016 end=-1.6007
2026-08-08 22:22:20,954 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0014≈0.08°/step  epoch≈4.8°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0014≈0.08°/step  epoch≈4.9°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-08 22:22:20,954 INFO   [exec discrete Δlogit per opt step] exec_move=0.0008  sprint=0.0008  kick=0.0011  tackle_attempt=0.0007
2026-08-08 22:22:20,954 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0014  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0016  sprint=+0.0025  kick=+0.0017  tackle_attempt=+0.0011  move_dir=+0.0406  kick_dir=+0.0183
2026-08-08 22:22:20,955 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.237 max=0.355  limit=0.4
              direction: 52/60 steps clipped (87%)  pre-clip norm mean=0.029 max=0.067  limit=0.02
2026-08-08 22:22:21,007 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=1,728,000  speed=868/s  reward=3.95
  loss     policy=0.0093  value=0.5351(x0.5)=0.2676
           entropy=6.8980  kl=0.0673
  value    V=2.37±1.16  R=2.42±1.64  adv=0.05±1.20
  moves    mv_ls=[-1.6072] (σ≈0.20, ≈11°) g=1.11e-02  d_move=[+0.0009] (Δσ≈0.010°)
           kk_ls=[-1.6007] (σ≈0.20, ≈12°)  d_kick=[+0.0008] (Δσ≈0.009°)
  heads    move= 50 get_poss= 38 exec_move= 79 sprint= 47 kick= 25 tackle= 42 shoot= 48 hold= 48 tackle_prob=0.4298 kick_prob=0.2461
  vs       vs[win/loss/tout/miss]  vs_immobile(519): 61.5%/0.0%/5.2%/20.6%/13%
  ep_len   20.7±12.9s  (n=519, min=0.1s, max=50.0s)
  reward   get_possession=+431.00  lose_possession=-3.60  ball_out=-224.00  box_possession=+797.50
           speed_bonus=+563.91  timeout=-40.50  stamina_penalty=-3.10
  rew/ep   (mean/std/min/max per episode, 519 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.830    0.395    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.007    0.079    -0.900    +0.000
  ball_out          -0.432    1.241    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.537    1.217    +0.000    +2.500
  speed_bonus       +1.087    1.220    +0.000    +4.190
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.078    0.333    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.025    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     433    +0.012    0.109     +3.045     2.044     +0.749      4.0358      1.687     3.800
  lose_possession       4    -0.000    0.009     +2.952     1.275     -0.407      1.8722      1.040     2.226
  ball_out            56    -0.006    0.158     -4.000     0.000     -3.502     13.7125      3.502     5.701
  box_possession     319    +0.022    0.234     +4.262     1.105     +1.408      3.1442      1.475     3.315
  speed_bonus        301    +0.016    0.196     +4.368     1.046     +1.474      3.2754      1.524     3.345
  timeout             27    -0.001    0.041     -1.509     0.006     -4.127     18.0545      4.127     5.497
  stamina_penalty     336    -0.000    0.001     +3.872     1.833     +1.009      4.3042      1.681     4.147
  gae/td   mean_return=+2.420  std_return=1.640  mean_gae=+0.046  mean_sq_td=1.4540
──────────────────────────────────────────────────────────────────────
2026-08-08 22:22:21,031 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint46.pt
2026-08-08 22:22:21,031 INFO Logging to checkpoints/phase1_run45/training_log47.txt
2026-08-08 22:22:21,032 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 22:22:35,577 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 22:22:35,579 INFO   [eval vs immobile] step=1,728,000  seeds=16x8  win=52%  mean_rew=2.588±3.144  V=2.114  gap=-0.474  outcomes={'other': 23, 'box_possession': 66, 'opponent_box_possession': 1, 'timeout': 5, 'miss': 33}
2026-08-08 22:22:35,580 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 22:22:45,327 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 22:22:45,329 INFO   [eval vs rules] step=1,728,000  seeds=16x8  win=0%  mean_rew=-2.327±1.316  V=0.961  gap=+3.288  outcomes={'other': 13, 'opponent_box_possession': 96, 'miss': 19}
2026-08-08 22:27:30,042 INFO   [KL mean=0.0581 median=0.0581 > 0.05] ratio percentiles:  p5=0.700  p25=0.898  p50=0.986  p75=1.029  p95=1.185  max=4.916
  move_dir_log_std=[-1.6063343286514282]  kick_dir_log_std=[-1.5999616384506226]
2026-08-08 22:27:30,055 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.691  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.550  kick=-0.327  t_att=-0.685
    move_dir=0.681 (min=-4.426 max=1.375)  kick_dir=0.215 (min=-4.031 max=2.038)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.43
  [worst sample] idx=91  ratio=8.404  adv=+0.250  old_lp=-3.280  new_lp=-1.152
    stored move_dir=-32.4°  new_mean=-23.8°  angular_diff=8.6°
    [worst sample per-head delta, sorted by |delta|] move:+0.031  tackle_attempt:-0.029
  [top-2 highest-ratio samples]
    idx=  91  ratio=   8.404  adv=+0.250  lp: old=-3.280  new=-1.152
      rew=+0.0000  ret=+4.4540  val=+4.2039  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: move:+0.031  tackle_attempt:-0.029
      saturation: exec_move_p_new=0.8817  sprint_p_new=0.6794  kick_p_new=0.0050  tackle_attempt_p_new=0.4123
    idx= 208  ratio=   7.685  adv=+0.097  lp: old=-3.461  new=-1.421
      rew=+0.0000  ret=+3.9287  val=+3.8316  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: tackle_attempt:+0.042  move:+0.029
      saturation: exec_move_p_new=0.8601  sprint_p_new=0.6757  kick_p_new=0.0055  tackle_attempt_p_new=0.4346
  [best sample (highest new_lp)] idx=71  new_lp=-1.056  adv=+0.900  stored move_dir=-37.5°  new_mean=-34.5°
    per-head contributions: kick:-0.024  tackle_attempt:-0.456  move:-0.575
2026-08-08 22:27:30,056 INFO   [advantage] mean=-0.000  std=1.000  min=-5.761  max=4.369
2026-08-08 22:27:30,056 INFO   [ratio] mean=0.9627  std=0.1660  min=0.0056  max=4.9160  clipped=24.1%
2026-08-08 22:27:30,056 INFO   [exec head grad norm] move_direction=0.026  exec_move=0.040  sprint=0.048  kick=0.035  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.054
2026-08-08 22:27:30,057 INFO   [exec continuous log_std] move_direction: start=-1.6072 end=-1.6063   kick_direction: start=-1.6007 end=-1.6000
2026-08-08 22:27:30,057 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0016≈0.09°/step  epoch≈5.4°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0018≈0.10°/step  epoch≈6.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-08 22:27:30,057 INFO   [exec discrete Δlogit per opt step] exec_move=0.0008  sprint=0.0010  kick=0.0009  tackle_attempt=0.0008
2026-08-08 22:27:30,057 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0013  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0021  sprint=+0.0019  kick=+0.0020  tackle_attempt=+0.0014  move_dir=+0.0332  kick_dir=+0.0162
2026-08-08 22:27:30,057 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.246 max=0.367  limit=0.4
              direction: 53/60 steps clipped (88%)  pre-clip norm mean=0.030 max=0.061  limit=0.02
2026-08-08 22:27:30,099 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=1,764,000  speed=873/s  reward=2.09
  loss     policy=0.0083  value=0.5081(x0.5)=0.2540
           entropy=6.8894  kl=0.0581
  value    V=2.44±1.13  R=2.51±1.74  adv=0.07±1.25
  moves    mv_ls=[-1.6063] (σ≈0.20, ≈11°) g=1.00e-02  d_move=[+0.0008] (Δσ≈0.009°)
           kk_ls=[-1.6000] (σ≈0.20, ≈12°)  d_kick=[+0.0008] (Δσ≈0.009°)
  heads    move= 50 get_poss= 39 exec_move= 79 sprint= 47 kick= 24 tackle= 43 shoot= 48 hold= 48 tackle_prob=0.4370 kick_prob=0.2401
  vs       vs[win/loss/tout/miss]  vs_immobile(540): 61.9%/0.0%/3.1%/22.4%/13%
  ep_len   19.9±11.9s  (n=540, min=0.2s, max=50.0s)
  reward   get_possession=+440.00  lose_possession=-4.50  ball_out=-272.00  box_possession=+835.00
           speed_bonus=+648.82  timeout=-25.50  stamina_penalty=-3.05
  rew/ep   (mean/std/min/max per episode, 540 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.815    0.412    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.008    0.086    -0.900    +0.000
  ball_out          -0.504    1.327    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.546    1.214    +0.000    +2.500
  speed_bonus       +1.202    1.288    +0.000    +4.011
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.047    0.262    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.005    -0.023    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     441    +0.012    0.110     +3.158     2.120     +0.747      4.2208      1.748     4.068
  lose_possession       5    -0.000    0.011     +2.833     0.702     -0.052      0.5349      0.573     1.157
  ball_out            68    -0.008    0.174     -3.956     0.205     -3.503     13.8979      3.503     5.540
  box_possession     334    +0.023    0.240     +4.437     1.110     +1.109      2.0416      1.195     2.530
  speed_bonus        325    +0.018    0.215     +4.491     1.076     +1.149      2.0913      1.218     2.531
  timeout             17    -0.001    0.033     -1.510     0.005     -4.308     20.6520      4.331     5.521
  stamina_penalty     344    -0.000    0.001     +4.181     1.656     +0.858      2.9244      1.347     2.858
  gae/td   mean_return=+2.511  std_return=1.741  mean_gae=+0.072  mean_sq_td=1.5556
──────────────────────────────────────────────────────────────────────
2026-08-08 22:27:30,122 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint47.pt
2026-08-08 22:27:30,123 INFO Logging to checkpoints/phase1_run45/training_log48.txt
2026-08-08 22:27:30,124 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 22:27:45,164 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 22:27:45,166 INFO   [eval vs immobile] step=1,764,000  seeds=16x8  win=55%  mean_rew=2.689±3.049  V=2.293  gap=-0.396  outcomes={'other': 23, 'opponent_box_possession': 1, 'box_possession': 71, 'miss': 31, 'timeout': 2}
2026-08-08 22:27:45,167 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 22:27:55,195 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 22:27:55,197 INFO   [eval vs rules] step=1,764,000  seeds=16x8  win=4%  mean_rew=-2.047±2.033  V=0.928  gap=+2.975  outcomes={'opponent_box_possession': 92, 'other': 9, 'miss': 22, 'box_possession': 5}
2026-08-08 22:32:42,260 INFO   [KL mean=0.0574 median=0.0571 > 0.05] ratio percentiles:  p5=0.700  p25=0.903  p50=0.989  p75=1.026  p95=1.182  max=4.956
  move_dir_log_std=[-1.6055607795715332]  kick_dir_log_std=[-1.5991747379302979]
2026-08-08 22:32:42,272 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.686  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.557  kick=-0.352  t_att=-0.677
    move_dir=0.660 (min=-3.330 max=1.373)  kick_dir=0.192 (min=-1.613 max=1.983)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.45
  [worst sample] idx=205  ratio=8.987  adv=+0.131  old_lp=-3.607  new_lp=-1.412
    stored move_dir=26.4°  new_mean=-1.8°  angular_diff=28.2°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 205  ratio=   8.987  adv=+0.131  lp: old=-3.607  new=-1.412
      rew=+0.0000  ret=+4.4987  val=+4.3675  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8850  sprint_p_new=0.6485  kick_p_new=0.0036  tackle_attempt_p_new=0.4483
    idx= 125  ratio=   6.995  adv=-0.062  lp: old=-3.375  new=-1.430
      rew=+0.0000  ret=+3.2558  val=+3.3181  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8489  sprint_p_new=0.6167  kick_p_new=0.0055  tackle_attempt_p_new=0.4304
  [best sample (highest new_lp)] idx=209  new_lp=-1.161  adv=+0.118  stored move_dir=-13.4°  new_mean=-6.7°
    per-head contributions: tackle_attempt:-0.527  move:-0.625
2026-08-08 22:32:42,272 INFO   [advantage] mean=0.000  std=1.000  min=-5.319  max=3.156
2026-08-08 22:32:42,273 INFO   [ratio] mean=0.9636  std=0.1632  min=0.0102  max=4.9563  clipped=23.1%
2026-08-08 22:32:42,273 INFO   [exec head grad norm] move_direction=0.025  exec_move=0.037  sprint=0.046  kick=0.028  kick_direction=0.008  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.050
2026-08-08 22:32:42,273 INFO   [exec continuous log_std] move_direction: start=-1.6063 end=-1.6056   kick_direction: start=-1.6000 end=-1.5992
2026-08-08 22:32:42,273 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0012≈0.07°/step  epoch≈4.0°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0012≈0.07°/step  epoch≈4.2°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-08 22:32:42,274 INFO   [exec discrete Δlogit per opt step] exec_move=0.0007  sprint=0.0009  kick=0.0008  tackle_attempt=0.0005
2026-08-08 22:32:42,274 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0011  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0018  sprint=+0.0019  kick=+0.0018  tackle_attempt=+0.0014  move_dir=+0.0336  kick_dir=+0.0157
2026-08-08 22:32:42,274 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.228 max=0.334  limit=0.4
              direction: 52/60 steps clipped (87%)  pre-clip norm mean=0.029 max=0.062  limit=0.02
2026-08-08 22:32:42,321 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=1,800,000  speed=873/s  reward=3.26
  loss     policy=0.0061  value=0.5082(x0.5)=0.2541
           entropy=6.8779  kl=0.0574
  value    V=2.51±1.20  R=2.55±1.82  adv=0.04±1.31
  moves    mv_ls=[-1.6056] (σ≈0.20, ≈12°) g=9.26e-03  d_move=[+0.0008] (Δσ≈0.009°)
           kk_ls=[-1.5992] (σ≈0.20, ≈12°)  d_kick=[+0.0008] (Δσ≈0.009°)
  heads    move= 51 get_poss= 39 exec_move= 79 sprint= 48 kick= 23 tackle= 44 shoot= 48 hold= 48 tackle_prob=0.4399 kick_prob=0.2328
  vs       vs[win/loss/tout/miss]  vs_immobile(552): 62.3%/0.0%/3.1%/23.9%/11%
  ep_len   19.4±11.9s  (n=552, min=0.1s, max=50.0s)
  reward   get_possession=+461.00  lose_possession=-3.60  ball_out=-336.00  box_possession=+860.00
           speed_bonus=+656.72  timeout=-25.50  stamina_penalty=-3.29
  rew/ep   (mean/std/min/max per episode, 552 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.835    0.390    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.007    0.076    -0.900    +0.000
  ball_out          -0.609    1.437    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.558    1.211    +0.000    +2.500
  speed_bonus       +1.190    1.256    +0.000    +4.153
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.046    0.259    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.025    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     466    +0.013    0.113     +3.126     2.188     +0.790      4.4961      1.828     4.010
  lose_possession       4    -0.000    0.009     +2.576     0.348     -0.286      0.1711      0.396     0.515
  ball_out            84    -0.009    0.193     -3.988     0.108     -3.682     15.1933      3.682     5.548
  box_possession     344    +0.024    0.243     +4.409     1.064     +0.998      1.9709      1.133     2.772
  speed_bonus        333    +0.018    0.213     +4.469     1.027     +1.022      2.0218      1.153     2.805
  timeout             17    -0.001    0.033     -1.511     0.006     -4.536     21.6278      4.536     6.029
  stamina_penalty     358    -0.000    0.001     +4.134     1.634     +0.734      2.8996      1.294     3.557
  gae/td   mean_return=+2.550  std_return=1.817  mean_gae=+0.036  mean_sq_td=1.7090
──────────────────────────────────────────────────────────────────────
2026-08-08 22:32:42,344 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint48.pt
2026-08-08 22:32:42,345 INFO Logging to checkpoints/phase1_run45/training_log49.txt
2026-08-08 22:32:42,346 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 22:32:56,682 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 22:32:56,683 INFO   [eval vs immobile] step=1,800,000  seeds=16x8  win=53%  mean_rew=2.580±3.214  V=2.270  gap=-0.310  outcomes={'other': 23, 'miss': 34, 'box_possession': 68, 'timeout': 3}
2026-08-08 22:32:56,685 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 22:33:06,857 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 22:33:06,858 INFO   [eval vs rules] step=1,800,000  seeds=16x8  win=6%  mean_rew=-2.026±2.297  V=0.943  gap=+2.968  outcomes={'opponent_box_possession': 100, 'other': 8, 'box_possession': 8, 'miss': 12}
2026-08-08 22:37:53,313 INFO   [KL mean=0.0576 median=0.0574 > 0.05] ratio percentiles:  p5=0.702  p25=0.901  p50=0.986  p75=1.030  p95=1.184  max=5.290
  move_dir_log_std=[-1.6047335863113403]  kick_dir_log_std=[-1.5983896255493164]
2026-08-08 22:37:53,325 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.685  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.484  kick=-0.411  t_att=-0.667
    move_dir=0.575 (min=-3.258 max=1.372)  kick_dir=0.192 (min=-1.059 max=2.014)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.50
  [worst sample] idx=88  ratio=8.720  adv=-2.973  old_lp=-3.732  new_lp=-1.566
    stored move_dir=-12.8°  new_mean=-7.0°  angular_diff=5.8°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx=  88  ratio=   8.720  adv=-2.973  lp: old=-3.732  new=-1.566
      rew=+0.0000  ret=+1.3061  val=+4.2791  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8885  sprint_p_new=0.6602  kick_p_new=0.0043  tackle_attempt_p_new=0.4565
    idx= 195  ratio=   7.570  adv=-0.299  lp: old=-3.684  new=-1.660
      rew=+0.0000  ret=+4.0401  val=+4.3392  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8729  sprint_p_new=0.6055  kick_p_new=0.0037  tackle_attempt_p_new=0.4064
  [best sample (highest new_lp)] idx=65  new_lp=-1.107  adv=-1.251  stored move_dir=6.4°  new_mean=28.2°
    per-head contributions: tackle_attempt:-0.548  move:-0.554
2026-08-08 22:37:53,325 INFO   [advantage] mean=0.000  std=1.000  min=-6.054  max=4.263
2026-08-08 22:37:53,326 INFO   [ratio] mean=0.9637  std=0.1659  min=0.0022  max=5.2901  clipped=23.4%
2026-08-08 22:37:53,326 INFO   [exec head grad norm] move_direction=0.024  exec_move=0.041  sprint=0.046  kick=0.030  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.059
2026-08-08 22:37:53,326 INFO   [exec continuous log_std] move_direction: start=-1.6056 end=-1.6047   kick_direction: start=-1.5992 end=-1.5984
2026-08-08 22:37:53,326 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0013≈0.08°/step  epoch≈4.5°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0014≈0.08°/step  epoch≈4.8°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-08 22:37:53,327 INFO   [exec discrete Δlogit per opt step] exec_move=0.0008  sprint=0.0011  kick=0.0009  tackle_attempt=0.0007
2026-08-08 22:37:53,327 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0012  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0023  sprint=+0.0020  kick=+0.0020  tackle_attempt=+0.0015  move_dir=+0.0336  kick_dir=+0.0150
2026-08-08 22:37:53,327 INFO   [grad clip] main: 1/60 steps clipped (2%)  pre-clip norm mean=0.230 max=0.432  limit=0.4
              direction: 54/60 steps clipped (90%)  pre-clip norm mean=0.028 max=0.068  limit=0.02
2026-08-08 22:37:53,379 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=1,836,000  speed=868/s  reward=3.83
  loss     policy=0.0074  value=0.5226(x0.5)=0.2613
           entropy=6.8772  kl=0.0576
  value    V=2.60±1.22  R=2.69±1.76  adv=0.09±1.27
  moves    mv_ls=[-1.6047] (σ≈0.20, ≈12°) g=9.96e-03  d_move=[+0.0008] (Δσ≈0.010°)
           kk_ls=[-1.5984] (σ≈0.20, ≈12°)  d_kick=[+0.0008] (Δσ≈0.009°)
  heads    move= 51 get_poss= 39 exec_move= 79 sprint= 49 kick= 23 tackle= 44 shoot= 47 hold= 48 tackle_prob=0.4459 kick_prob=0.2305
  vs       vs[win/loss/tout/miss]  vs_immobile(553): 66.2%/0.0%/2.9%/22.2%/9%
  ep_len   19.5±11.8s  (n=553, min=0.9s, max=50.0s)
  reward   get_possession=+465.00  lose_possession=-3.60  ball_out=-280.00  box_possession=+915.00
           speed_bonus=+695.59  timeout=-24.00  stamina_penalty=-3.36
  rew/ep   (mean/std/min/max per episode, 553 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.841    0.385    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.007    0.076    -0.900    +0.000
  ball_out          -0.506    1.330    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.655    1.183    +0.000    +2.500
  speed_bonus       +1.258    1.253    +0.000    +4.152
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.043    0.251    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.025    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     466    +0.013    0.113     +3.357     2.090     +0.927      4.1885      1.764     3.794
  lose_possession       4    -0.000    0.009     +2.928     0.685     -0.384      0.8701      0.849     1.275
  ball_out            70    -0.008    0.176     -3.957     0.203     -3.676     15.2504      3.676     6.068
  box_possession     366    +0.025    0.251     +4.395     1.069     +1.028      2.0952      1.199     2.730
  speed_bonus        352    +0.019    0.219     +4.470     1.019     +1.066      2.1362      1.211     2.737
  timeout             16    -0.001    0.032     -1.510     0.006     -4.668     22.6529      4.668     5.696
  stamina_penalty     378    -0.000    0.001     +4.157     1.583     +0.803      2.9834      1.353     3.133
  gae/td   mean_return=+2.687  std_return=1.758  mean_gae=+0.085  mean_sq_td=1.6255
──────────────────────────────────────────────────────────────────────
2026-08-08 22:37:53,406 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint49.pt
2026-08-08 22:37:53,407 INFO Logging to checkpoints/phase1_run45/training_log50.txt
2026-08-08 22:37:53,408 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 22:38:08,107 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 22:38:08,109 INFO   [eval vs immobile] step=1,836,000  seeds=16x8  win=53%  mean_rew=2.505±3.203  V=2.383  gap=-0.122  outcomes={'other': 20, 'box_possession': 68, 'opponent_box_possession': 1, 'timeout': 4, 'miss': 35}
2026-08-08 22:38:08,110 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 22:38:18,896 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 22:38:18,897 INFO   [eval vs rules] step=1,836,000  seeds=16x8  win=2%  mean_rew=-2.203±1.851  V=1.103  gap=+3.306  outcomes={'other': 9, 'opponent_box_possession': 99, 'miss': 17, 'box_possession': 3}
2026-08-08 22:42:58,920 INFO   [KL mean=0.0563 median=0.0561 > 0.05] ratio percentiles:  p5=0.705  p25=0.903  p50=0.988  p75=1.024  p95=1.178  max=6.192
  move_dir_log_std=[-1.6038392782211304]  kick_dir_log_std=[-1.5975326299667358]
2026-08-08 22:42:58,933 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.687  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.519  kick=-0.591  t_att=-0.687
    move_dir=0.653 (min=-4.169 max=1.370)  kick_dir=0.417 (min=-2.012 max=2.035)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.50
  [worst sample] idx=17  ratio=7.070  adv=-0.955  old_lp=-3.332  new_lp=-1.376
    stored move_dir=42.3°  new_mean=47.8°  angular_diff=5.5°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx=  17  ratio=   7.070  adv=-0.955  lp: old=-3.332  new=-1.376
      rew=+0.0000  ret=+2.9240  val=+3.8789  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8550  sprint_p_new=0.5780  kick_p_new=0.0034  tackle_attempt_p_new=0.4304
    idx= 246  ratio=   6.256  adv=-0.535  lp: old=-3.394  new=-1.561
      rew=+0.0000  ret=+2.5890  val=+3.1245  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8350  sprint_p_new=0.5605  kick_p_new=0.0058  tackle_attempt_p_new=0.4695
  [best sample (highest new_lp)] idx=243  new_lp=-1.233  adv=-0.528  stored move_dir=18.2°  new_mean=32.6°
    per-head contributions: move:-0.607  tackle_attempt:-0.620
2026-08-08 22:42:58,933 INFO   [advantage] mean=0.000  std=1.000  min=-5.887  max=3.939
2026-08-08 22:42:58,934 INFO   [ratio] mean=0.9634  std=0.1602  min=0.0025  max=6.1925  clipped=22.8%
2026-08-08 22:42:58,934 INFO   [exec head grad norm] move_direction=0.021  exec_move=0.044  sprint=0.045  kick=0.026  kick_direction=0.008  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.062
2026-08-08 22:42:58,934 INFO   [exec continuous log_std] move_direction: start=-1.6047 end=-1.6038   kick_direction: start=-1.5984 end=-1.5975
2026-08-08 22:42:58,934 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0010≈0.06°/step  epoch≈3.3°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0010≈0.06°/step  epoch≈3.5°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-08 22:42:58,934 INFO   [exec discrete Δlogit per opt step] exec_move=0.0009  sprint=0.0008  kick=0.0009  tackle_attempt=0.0006
2026-08-08 22:42:58,934 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0014  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0023  sprint=+0.0019  kick=+0.0023  tackle_attempt=+0.0011  move_dir=+0.0322  kick_dir=+0.0151
2026-08-08 22:42:58,935 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.240 max=0.356  limit=0.4
              direction: 49/60 steps clipped (82%)  pre-clip norm mean=0.025 max=0.049  limit=0.02
2026-08-08 22:42:58,992 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=1,872,000  speed=873/s  reward=4.47
  loss     policy=0.0084  value=0.4921(x0.5)=0.2461
           entropy=6.8897  kl=0.0563
  value    V=2.72±1.19  R=2.75±1.70  adv=0.03±1.19
  moves    mv_ls=[-1.6038] (σ≈0.20, ≈12°) g=9.59e-03  d_move=[+0.0009] (Δσ≈0.010°)
           kk_ls=[-1.5975] (σ≈0.20, ≈12°)  d_kick=[+0.0009] (Δσ≈0.010°)
  heads    move= 52 get_poss= 38 exec_move= 79 sprint= 48 kick= 23 tackle= 45 shoot= 48 hold= 48 tackle_prob=0.4506 kick_prob=0.2325
  vs       vs[win/loss/tout/miss]  vs_immobile(544): 65.3%/0.0%/2.9%/20.4%/11%
  ep_len   19.6±11.7s  (n=544, min=1.4s, max=50.0s)
  reward   get_possession=+449.00  lose_possession=-6.30  ball_out=-228.00  box_possession=+887.50
           speed_bonus=+678.98  timeout=-24.00  stamina_penalty=-3.29
  rew/ep   (mean/std/min/max per episode, 544 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.825    0.412    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.012    0.101    -0.900    +0.000
  ball_out          -0.419    1.225    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.631    1.190    +0.000    +2.500
  speed_bonus       +1.248    1.277    +0.000    +4.000
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.044    0.253    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.025    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     453    +0.013    0.111     +3.408     2.077     +0.790      3.8017      1.644     3.910
  lose_possession       7    -0.000    0.013     +2.780     0.638     -0.301      0.8451      0.782     1.424
  ball_out            57    -0.006    0.159     -3.947     0.223     -3.344     12.2292      3.344     4.784
  box_possession     355    +0.025    0.247     +4.410     1.103     +0.946      2.0299      1.145     2.763
  speed_bonus        340    +0.019    0.219     +4.494     1.050     +0.993      2.1005      1.172     2.790
  timeout             16    -0.001    0.032     -1.513     0.006     -4.519     21.0971      4.519     5.632
  stamina_penalty     369    -0.000    0.001     +4.159     1.619     +0.713      2.8641      1.293     3.693
  gae/td   mean_return=+2.745  std_return=1.700  mean_gae=+0.029  mean_sq_td=1.4158
──────────────────────────────────────────────────────────────────────
2026-08-08 22:42:59,018 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint50.pt
2026-08-08 22:42:59,018 INFO Logging to checkpoints/phase1_run45/training_log51.txt
2026-08-08 22:42:59,019 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 22:43:14,124 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 22:43:14,125 INFO   [eval vs immobile] step=1,872,000  seeds=16x8  win=48%  mean_rew=2.005±3.230  V=2.302  gap=+0.297  outcomes={'other': 19, 'box_possession': 62, 'miss': 43, 'timeout': 4}
2026-08-08 22:43:14,126 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 22:43:24,835 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 22:43:24,837 INFO   [eval vs rules] step=1,872,000  seeds=16x8  win=2%  mean_rew=-2.099±1.795  V=1.145  gap=+3.244  outcomes={'opponent_box_possession': 94, 'other': 14, 'miss': 17, 'box_possession': 3}
2026-08-08 22:48:04,404 INFO   [KL mean=0.0575 median=0.0574 > 0.05] ratio percentiles:  p5=0.706  p25=0.903  p50=0.988  p75=1.024  p95=1.177  max=5.406
  move_dir_log_std=[-1.6028982400894165]  kick_dir_log_std=[-1.5966863632202148]
2026-08-08 22:48:04,418 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.690  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.499  kick=-0.369  t_att=-0.681
    move_dir=0.703 (min=-2.050 max=1.368)  kick_dir=0.228 (min=-1.981 max=2.017)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.51
  [worst sample] idx=206  ratio=8.026  adv=+1.631  old_lp=-3.658  new_lp=-1.575
    stored move_dir=-179.0°  new_mean=-174.3°  angular_diff=4.7°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 206  ratio=   8.026  adv=+1.631  lp: old=-3.658  new=-1.575
      rew=+0.0000  ret=+5.3939  val=+3.7627  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8680  sprint_p_new=0.6651  kick_p_new=0.0049  tackle_attempt_p_new=0.4410
    idx= 208  ratio=   7.571  adv=+2.090  lp: old=-3.494  new=-1.470
      rew=+0.0000  ret=+5.5626  val=+3.4721  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8610  sprint_p_new=0.6569  kick_p_new=0.0058  tackle_attempt_p_new=0.4396
  [best sample (highest new_lp)] idx=179  new_lp=-1.216  adv=+0.635  stored move_dir=154.9°  new_mean=161.7°
    per-head contributions: move:-0.587  tackle_attempt:-0.623
2026-08-08 22:48:04,418 INFO   [advantage] mean=-0.000  std=1.000  min=-5.865  max=4.003
2026-08-08 22:48:04,419 INFO   [ratio] mean=0.9627  std=0.1596  min=0.0027  max=5.4059  clipped=22.6%
2026-08-08 22:48:04,419 INFO   [exec head grad norm] move_direction=0.022  exec_move=0.039  sprint=0.049  kick=0.035  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.049
2026-08-08 22:48:04,419 INFO   [exec continuous log_std] move_direction: start=-1.6038 end=-1.6029   kick_direction: start=-1.5975 end=-1.5967
2026-08-08 22:48:04,419 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0014≈0.08°/step  epoch≈5.0°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0014≈0.08°/step  epoch≈4.8°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-08 22:48:04,419 INFO   [exec discrete Δlogit per opt step] exec_move=0.0009  sprint=0.0008  kick=0.0014  tackle_attempt=0.0006
2026-08-08 22:48:04,420 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0014  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0021  sprint=+0.0022  kick=+0.0020  tackle_attempt=+0.0010  move_dir=+0.0335  kick_dir=+0.0153
2026-08-08 22:48:04,420 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.235 max=0.373  limit=0.4
              direction: 49/60 steps clipped (82%)  pre-clip norm mean=0.027 max=0.072  limit=0.02
2026-08-08 22:48:04,493 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=1,908,000  speed=875/s  reward=2.40
  loss     policy=0.0087  value=0.5118(x0.5)=0.2559
           entropy=6.9055  kl=0.0575
  value    V=2.72±1.22  R=2.71±1.73  adv=-0.01±1.24
  moves    mv_ls=[-1.6029] (σ≈0.20, ≈12°) g=1.04e-02  d_move=[+0.0009] (Δσ≈0.011°)
           kk_ls=[-1.5967] (σ≈0.20, ≈12°)  d_kick=[+0.0008] (Δσ≈0.010°)
  heads    move= 51 get_poss= 39 exec_move= 79 sprint= 48 kick= 24 tackle= 45 shoot= 48 hold= 48 tackle_prob=0.4550 kick_prob=0.2393
  vs       vs[win/loss/tout/miss]  vs_immobile(537): 64.4%/0.0%/3.0%/22.3%/10%
  ep_len   20.1±11.9s  (n=537, min=0.8s, max=50.0s)
  reward   get_possession=+451.00  lose_possession=-8.10  ball_out=-260.00  box_possession=+865.00
           speed_bonus=+657.18  timeout=-24.00  stamina_penalty=-3.16
  rew/ep   (mean/std/min/max per episode, 537 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.840    0.410    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.015    0.116    -0.900    +0.000
  ball_out          -0.484    1.305    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.611    1.197    +0.000    +2.500
  speed_bonus       +1.224    1.294    +0.000    +4.174
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.045    0.255    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.023    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     453    +0.013    0.111     +3.254     2.190     +0.580      4.3707      1.715     4.339
  lose_possession       9    -0.000    0.014     +2.868     0.640     -0.471      1.1559      0.873     1.977
  ball_out            65    -0.007    0.170     -3.969     0.173     -3.219     11.8157      3.219     5.185
  box_possession     346    +0.024    0.244     +4.396     1.148     +1.009      2.2340      1.207     2.748
  speed_bonus        333    +0.018    0.217     +4.471     1.106     +1.052      2.2931      1.227     2.789
  timeout             16    -0.001    0.032     -1.509     0.007     -4.722     23.7283      4.722     6.059
  stamina_penalty     353    -0.000    0.001     +4.229     1.525     +0.823      3.0839      1.344     3.429
  gae/td   mean_return=+2.713  std_return=1.728  mean_gae=-0.007  mean_sq_td=1.5368
──────────────────────────────────────────────────────────────────────
2026-08-08 22:48:04,520 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint51.pt
2026-08-08 22:48:04,521 INFO Logging to checkpoints/phase1_run45/training_log52.txt
2026-08-08 22:48:04,522 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 22:48:19,416 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 22:48:19,418 INFO   [eval vs immobile] step=1,908,000  seeds=16x8  win=55%  mean_rew=2.835±3.044  V=2.511  gap=-0.324  outcomes={'other': 19, 'timeout': 6, 'box_possession': 71, 'miss': 32}
2026-08-08 22:48:19,419 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 22:48:29,309 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 22:48:29,310 INFO   [eval vs rules] step=1,908,000  seeds=16x8  win=1%  mean_rew=-2.310±1.508  V=1.168  gap=+3.478  outcomes={'other': 12, 'opponent_box_possession': 98, 'miss': 17, 'box_possession': 1}
2026-08-08 22:53:12,761 INFO   [KL mean=0.0572 median=0.0568 > 0.05] ratio percentiles:  p5=0.713  p25=0.905  p50=0.989  p75=1.025  p95=1.171  max=3.827
  move_dir_log_std=[-1.6020808219909668]  kick_dir_log_std=[-1.595947265625]
2026-08-08 22:53:12,775 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.695  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.516  kick=-0.335  t_att=-0.695
    move_dir=0.713 (min=-1.784 max=1.366)  kick_dir=0.178 (min=-3.057 max=2.023)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.48
  [worst sample] idx=223  ratio=9.571  adv=+0.165  old_lp=-3.681  new_lp=-1.422
    stored move_dir=-149.8°  new_mean=-164.7°  angular_diff=14.9°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 223  ratio=   9.571  adv=+0.165  lp: old=-3.681  new=-1.422
      rew=+0.0000  ret=+3.0270  val=+2.8625  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8966  sprint_p_new=0.6575  kick_p_new=0.0031  tackle_attempt_p_new=0.4756
    idx= 231  ratio=   8.670  adv=-0.306  lp: old=-3.646  new=-1.486
      rew=+0.0000  ret=+3.1361  val=+3.4421  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8863  sprint_p_new=0.6208  kick_p_new=0.0036  tackle_attempt_p_new=0.4715
  [best sample (highest new_lp)] idx=103  new_lp=-1.224  adv=+0.359  stored move_dir=-156.6°  new_mean=-146.5°
    per-head contributions: tackle_attempt:-0.586  move:-0.627
2026-08-08 22:53:12,775 INFO   [advantage] mean=-0.000  std=1.000  min=-5.724  max=4.261
2026-08-08 22:53:12,777 INFO   [ratio] mean=0.9634  std=0.1570  min=0.0079  max=3.8267  clipped=22.0%
2026-08-08 22:53:12,777 INFO   [exec head grad norm] move_direction=0.024  exec_move=0.035  sprint=0.046  kick=0.033  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.049
2026-08-08 22:53:12,777 INFO   [exec continuous log_std] move_direction: start=-1.6029 end=-1.6021   kick_direction: start=-1.5967 end=-1.5959
2026-08-08 22:53:12,777 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0016≈0.09°/step  epoch≈5.4°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0015≈0.09°/step  epoch≈5.3°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-08 22:53:12,778 INFO   [exec discrete Δlogit per opt step] exec_move=0.0008  sprint=0.0008  kick=0.0007  tackle_attempt=0.0005
2026-08-08 22:53:12,778 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0011  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0017  sprint=+0.0014  kick=+0.0020  tackle_attempt=+0.0011  move_dir=+0.0342  kick_dir=+0.0157
2026-08-08 22:53:12,779 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.231 max=0.399  limit=0.4
              direction: 52/60 steps clipped (87%)  pre-clip norm mean=0.029 max=0.048  limit=0.02
2026-08-08 22:53:12,823 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=1,944,000  speed=869/s  reward=2.36
  loss     policy=0.0071  value=0.5337(x0.5)=0.2668
           entropy=6.9100  kl=0.0572
  value    V=2.74±1.16  R=2.78±1.65  adv=0.04±1.21
  moves    mv_ls=[-1.6021] (σ≈0.20, ≈12°) g=9.83e-03  d_move=[+0.0008] (Δσ≈0.009°)
           kk_ls=[-1.5959] (σ≈0.20, ≈12°)  d_kick=[+0.0007] (Δσ≈0.009°)
  heads    move= 50 get_poss= 38 exec_move= 78 sprint= 48 kick= 24 tackle= 46 shoot= 48 hold= 49 tackle_prob=0.4566 kick_prob=0.2398
  vs       vs[win/loss/tout/miss]  vs_immobile(528): 68.4%/0.0%/4.5%/18.4%/9%
  ep_len   20.3±12.0s  (n=528, min=0.6s, max=50.0s)
  reward   get_possession=+449.00  lose_possession=-1.80  ball_out=-200.00  box_possession=+902.50
           speed_bonus=+676.86  timeout=-36.00  stamina_penalty=-3.20
  rew/ep   (mean/std/min/max per episode, 528 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.850    0.367    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.003    0.055    -0.900    +0.000
  ball_out          -0.379    1.171    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.709    1.163    +0.000    +2.500
  speed_bonus       +1.282    1.264    +0.000    +4.074
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.068    0.312    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.005    -0.025    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     450    +0.013    0.111     +3.491     1.964     +0.790      3.6316      1.591     3.768
  lose_possession       2    -0.000    0.007     +2.961     0.537     -0.250      0.2502      0.433     0.658
  ball_out            50    -0.006    0.149     -3.920     0.271     -3.737     15.4917      3.737     6.206
  box_possession     361    +0.025    0.249     +4.370     1.110     +0.993      1.9741      1.118     2.752
  speed_bonus        342    +0.019    0.217     +4.474     1.047     +1.052      2.0641      1.152     2.766
  timeout             24    -0.001    0.039     -1.512     0.006     -4.443     21.1845      4.443     5.692
  stamina_penalty     376    -0.000    0.001     +4.040     1.771     +0.680      3.1716      1.330     3.900
  gae/td   mean_return=+2.780  std_return=1.648  mean_gae=+0.044  mean_sq_td=1.4574
──────────────────────────────────────────────────────────────────────
2026-08-08 22:53:12,846 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint52.pt
2026-08-08 22:53:12,847 INFO Logging to checkpoints/phase1_run45/training_log53.txt
2026-08-08 22:53:12,848 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 22:53:26,714 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 22:53:26,715 INFO   [eval vs immobile] step=1,944,000  seeds=16x8  win=52%  mean_rew=2.454±3.165  V=2.506  gap=+0.052  outcomes={'other': 18, 'box_possession': 66, 'miss': 40, 'timeout': 4}
2026-08-08 22:53:26,716 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 22:53:36,658 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 22:53:36,660 INFO   [eval vs rules] step=1,944,000  seeds=16x8  win=1%  mean_rew=-2.324±1.544  V=1.223  gap=+3.547  outcomes={'opponent_box_possession': 99, 'other': 11, 'miss': 17, 'box_possession': 1}
2026-08-08 22:58:21,644 INFO   [KL mean=0.0583 median=0.0582 > 0.05] ratio percentiles:  p5=0.708  p25=0.904  p50=0.985  p75=1.028  p95=1.168  max=3.580
  move_dir_log_std=[-1.6012948751449585]  kick_dir_log_std=[-1.5951122045516968]
2026-08-08 22:58:21,657 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.691  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.469  kick=-0.374  t_att=-0.684
    move_dir=0.631 (min=-3.003 max=1.365)  kick_dir=0.193 (min=-2.486 max=2.000)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.58
  [worst sample] idx=211  ratio=9.286  adv=-1.061  old_lp=-3.735  new_lp=-1.506
    stored move_dir=14.6°  new_mean=8.8°  angular_diff=5.8°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 211  ratio=   9.286  adv=-1.061  lp: old=-3.735  new=-1.506
      rew=+0.0000  ret=+3.1725  val=+4.2332  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8933  sprint_p_new=0.6444  kick_p_new=0.0027  tackle_attempt_p_new=0.4744
    idx= 201  ratio=   9.073  adv=-0.604  lp: old=-3.646  new=-1.440
      rew=+0.0000  ret=+3.2560  val=+3.8600  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8926  sprint_p_new=0.6417  kick_p_new=0.0029  tackle_attempt_p_new=0.4752
  [best sample (highest new_lp)] idx=134  new_lp=-1.206  adv=-0.606  stored move_dir=-41.8°  new_mean=-31.1°
    per-head contributions: kick:-0.046  move:-0.576  tackle_attempt:-0.584
2026-08-08 22:58:21,657 INFO   [advantage] mean=-0.000  std=1.000  min=-6.038  max=4.275
2026-08-08 22:58:21,658 INFO   [ratio] mean=0.9619  std=0.1569  min=0.0082  max=3.5798  clipped=21.9%
2026-08-08 22:58:21,658 INFO   [exec head grad norm] move_direction=0.025  exec_move=0.045  sprint=0.045  kick=0.034  kick_direction=0.008  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.063
2026-08-08 22:58:21,658 INFO   [exec continuous log_std] move_direction: start=-1.6021 end=-1.6013   kick_direction: start=-1.5959 end=-1.5951
2026-08-08 22:58:21,658 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0017≈0.10°/step  epoch≈5.9°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0017≈0.10°/step  epoch≈5.8°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-08 22:58:21,659 INFO   [exec discrete Δlogit per opt step] exec_move=0.0008  sprint=0.0009  kick=0.0012  tackle_attempt=0.0008
2026-08-08 22:58:21,659 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0013  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0020  sprint=+0.0021  kick=+0.0018  tackle_attempt=+0.0010  move_dir=+0.0349  kick_dir=+0.0153
2026-08-08 22:58:21,659 INFO   [grad clip] main: 1/60 steps clipped (2%)  pre-clip norm mean=0.246 max=0.413  limit=0.4
              direction: 53/60 steps clipped (88%)  pre-clip norm mean=0.029 max=0.049  limit=0.02
2026-08-08 22:58:21,711 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=1,980,000  speed=869/s  reward=2.27
  loss     policy=0.0074  value=0.5113(x0.5)=0.2556
           entropy=6.9094  kl=0.0583
  value    V=2.80±1.09  R=2.75±1.64  adv=-0.04±1.18
  moves    mv_ls=[-1.6013] (σ≈0.20, ≈12°) g=9.40e-03  d_move=[+0.0008] (Δσ≈0.009°)
           kk_ls=[-1.5951] (σ≈0.20, ≈12°)  d_kick=[+0.0008] (Δσ≈0.010°)
  heads    move= 50 get_poss= 38 exec_move= 79 sprint= 47 kick= 24 tackle= 45 shoot= 48 hold= 49 tackle_prob=0.4629 kick_prob=0.2400
  vs       vs[win/loss/tout/miss]  vs_immobile(528): 65.3%/0.0%/4.0%/23.1%/8%
  ep_len   20.3±12.8s  (n=528, min=1.2s, max=50.0s)
  reward   get_possession=+445.00  lose_possession=-3.60  ball_out=-244.00  box_possession=+862.50
           speed_bonus=+625.97  timeout=-31.50  stamina_penalty=-3.06
  rew/ep   (mean/std/min/max per episode, 528 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.843    0.384    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.007    0.078    -0.900    +0.000
  ball_out          -0.462    1.279    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.634    1.190    +0.000    +2.500
  speed_bonus       +1.186    1.229    +0.000    +4.058
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.060    0.293    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.005    -0.021    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     449    +0.012    0.111     +3.311     2.113     +0.567      3.8433      1.595     4.205
  lose_possession       4    -0.000    0.009     +2.215     0.808     -0.964      1.3879      0.964     1.728
  ball_out            61    -0.007    0.165     -3.951     0.216     -3.518     13.8682      3.518     5.493
  box_possession     345    +0.024    0.244     +4.306     1.081     +0.836      1.6517      1.039     2.328
  speed_bonus        333    +0.017    0.206     +4.371     1.043     +0.872      1.6955      1.060     2.333
  timeout             21    -0.001    0.036     -1.510     0.006     -4.841     23.8542      4.841     5.669
  stamina_penalty     363    -0.000    0.001     +3.991     1.694     +0.524      2.8885      1.252     3.900
  gae/td   mean_return=+2.754  std_return=1.641  mean_gae=-0.044  mean_sq_td=1.3946
──────────────────────────────────────────────────────────────────────
2026-08-08 22:58:21,735 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint53.pt
2026-08-08 22:58:21,735 INFO Logging to checkpoints/phase1_run45/training_log54.txt
2026-08-08 22:58:21,736 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 22:58:35,618 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 22:58:35,619 INFO   [eval vs immobile] step=1,980,000  seeds=16x8  win=57%  mean_rew=2.945±3.033  V=2.548  gap=-0.397  outcomes={'other': 19, 'box_possession': 73, 'timeout': 4, 'miss': 32}
2026-08-08 22:58:35,620 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 22:58:46,544 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 22:58:46,546 INFO   [eval vs rules] step=1,980,000  seeds=16x8  win=2%  mean_rew=-2.101±1.818  V=1.100  gap=+3.201  outcomes={'opponent_box_possession': 95, 'other': 13, 'miss': 17, 'box_possession': 3}
2026-08-08 23:03:27,274 INFO   [KL mean=0.0530 median=0.0532 > 0.05] ratio percentiles:  p5=0.726  p25=0.911  p50=0.988  p75=1.026  p95=1.168  max=6.831
  move_dir_log_std=[-1.6004489660263062]  kick_dir_log_std=[-1.59424889087677]
2026-08-08 23:03:27,287 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.679  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.515  kick=-0.424  t_att=-0.687
    move_dir=0.648 (min=-8.348 max=1.363)  kick_dir=0.252 (min=-2.273 max=2.009)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.52
  [worst sample] idx=212  ratio=8.243  adv=+0.776  old_lp=-3.565  new_lp=-1.456
    stored move_dir=-176.6°  new_mean=167.4°  angular_diff=16.0°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 212  ratio=   8.243  adv=+0.776  lp: old=-3.565  new=-1.456
      rew=+0.0000  ret=+5.5951  val=+4.8186  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8756  sprint_p_new=0.6241  kick_p_new=0.0036  tackle_attempt_p_new=0.4877
    idx= 195  ratio=   7.445  adv=+0.399  lp: old=-3.510  new=-1.503
      rew=+0.0000  ret=+4.7493  val=+4.3507  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8669  sprint_p_new=0.5764  kick_p_new=0.0044  tackle_attempt_p_new=0.4901
  [best sample (highest new_lp)] idx=172  new_lp=-1.242  adv=+0.438  stored move_dir=-174.7°  new_mean=-173.3°
    per-head contributions: tackle_attempt:-0.604  move:-0.628
2026-08-08 23:03:27,288 INFO   [advantage] mean=0.000  std=1.000  min=-5.404  max=3.887
2026-08-08 23:03:27,288 INFO   [ratio] mean=0.9663  std=0.1544  min=0.0001  max=6.8306  clipped=20.6%
2026-08-08 23:03:27,289 INFO   [exec head grad norm] move_direction=0.023  exec_move=0.036  sprint=0.044  kick=0.035  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.047
2026-08-08 23:03:27,289 INFO   [exec continuous log_std] move_direction: start=-1.6013 end=-1.6004   kick_direction: start=-1.5951 end=-1.5942
2026-08-08 23:03:27,289 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0017≈0.09°/step  epoch≈5.7°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0016≈0.09°/step  epoch≈5.5°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-08 23:03:27,289 INFO   [exec discrete Δlogit per opt step] exec_move=0.0007  sprint=0.0011  kick=0.0008  tackle_attempt=0.0005
2026-08-08 23:03:27,289 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0010  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0015  sprint=+0.0019  kick=+0.0023  tackle_attempt=+0.0013  move_dir=+0.0302  kick_dir=+0.0148
2026-08-08 23:03:27,290 INFO   [grad clip] main: 2/60 steps clipped (3%)  pre-clip norm mean=0.244 max=0.426  limit=0.4
              direction: 53/60 steps clipped (88%)  pre-clip norm mean=0.027 max=0.074  limit=0.02
2026-08-08 23:03:27,334 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=2,016,000  speed=877/s  reward=3.54
  loss     policy=0.0079  value=0.4872(x0.5)=0.2436
           entropy=6.9186  kl=0.0530
  value    V=2.74±1.17  R=2.68±1.71  adv=-0.06±1.19
  moves    mv_ls=[-1.6004] (σ≈0.20, ≈12°) g=9.61e-03  d_move=[+0.0008] (Δσ≈0.010°)
           kk_ls=[-1.5942] (σ≈0.20, ≈12°)  d_kick=[+0.0009] (Δσ≈0.010°)
  heads    move= 51 get_poss= 38 exec_move= 79 sprint= 47 kick= 25 tackle= 46 shoot= 49 hold= 49 tackle_prob=0.4671 kick_prob=0.2442
  vs       vs[win/loss/tout/miss]  vs_immobile(529): 63.7%/0.0%/3.2%/23.1%/10%
  ep_len   20.4±12.4s  (n=529, min=0.2s, max=50.0s)
  reward   get_possession=+443.00  lose_possession=-3.60  ball_out=-284.00  box_possession=+842.50
           speed_bonus=+604.68  timeout=-25.50  stamina_penalty=-3.09
  rew/ep   (mean/std/min/max per episode, 529 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.837    0.394    +0.000    +3.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.007    0.096    -1.800    +0.000
  ball_out          -0.537    1.364    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.593    1.202    +0.000    +2.500
  speed_bonus       +1.143    1.219    +0.000    +4.195
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.048    0.265    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.023    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     446    +0.012    0.111     +3.172     2.243     +0.426      4.3622      1.692     4.524
  lose_possession       4    -0.000    0.009     +2.546     0.968     -0.530      0.8487      0.751     1.328
  ball_out            71    -0.008    0.177     -3.958     0.201     -3.654     14.7132      3.654     5.695
  box_possession     337    +0.023    0.241     +4.286     1.079     +0.572      1.2651      0.905     2.234
  speed_bonus        325    +0.017    0.202     +4.352     1.041     +0.597      1.2890      0.913     2.241
  timeout             17    -0.001    0.033     -1.509     0.006     -4.249     20.1170      4.249     5.691
  stamina_penalty     346    -0.000    0.001     +4.051     1.584     +0.370      2.0868      1.055     2.792
  gae/td   mean_return=+2.679  std_return=1.712  mean_gae=-0.056  mean_sq_td=1.4273
──────────────────────────────────────────────────────────────────────
2026-08-08 23:03:27,359 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint54.pt
2026-08-08 23:03:27,360 INFO Logging to checkpoints/phase1_run45/training_log55.txt
2026-08-08 23:03:27,361 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 23:03:41,730 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 23:03:41,732 INFO   [eval vs immobile] step=2,016,000  seeds=16x8  win=43%  mean_rew=1.557±3.335  V=2.414  gap=+0.857  outcomes={'other': 19, 'box_possession': 55, 'timeout': 5, 'miss': 49}
2026-08-08 23:03:41,733 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 23:03:50,995 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 23:03:50,996 INFO   [eval vs rules] step=2,016,000  seeds=16x8  win=3%  mean_rew=-2.009±1.930  V=0.874  gap=+2.883  outcomes={'opponent_box_possession': 84, 'other': 14, 'miss': 26, 'box_possession': 4}
2026-08-08 23:08:35,875 INFO   [KL mean=0.0535 median=0.0534 > 0.05] ratio percentiles:  p5=0.723  p25=0.910  p50=0.989  p75=1.025  p95=1.164  max=5.246
  move_dir_log_std=[-1.5997015237808228]  kick_dir_log_std=[-1.5933986902236938]
2026-08-08 23:08:35,887 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.691  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.495  kick=-0.509  t_att=-0.689
    move_dir=0.673 (min=-2.221 max=1.362)  kick_dir=0.242 (min=-6.574 max=2.015)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.52
  [worst sample] idx=187  ratio=7.536  adv=-0.083  old_lp=-3.386  new_lp=-1.366
    stored move_dir=-161.0°  new_mean=-168.6°  angular_diff=7.6°
    [worst sample per-head delta, sorted by |delta|] tackle_attempt:-0.029
  [top-2 highest-ratio samples]
    idx= 187  ratio=   7.536  adv=-0.083  lp: old=-3.386  new=-1.366
      rew=+0.0000  ret=+3.1972  val=+3.2799  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: tackle_attempt:-0.029
      saturation: exec_move_p_new=0.8767  sprint_p_new=0.6235  kick_p_new=0.0045  tackle_attempt_p_new=0.4723
    idx= 178  ratio=   7.191  adv=-0.205  lp: old=-3.499  new=-1.526
      rew=+0.0000  ret=+3.0979  val=+3.3025  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: tackle_attempt:-0.027
      saturation: exec_move_p_new=0.8720  sprint_p_new=0.6045  kick_p_new=0.0052  tackle_attempt_p_new=0.4891
  [best sample (highest new_lp)] idx=215  new_lp=-1.250  adv=-0.634  stored move_dir=-103.4°  new_mean=-122.2°
    per-head contributions: move:-0.611  tackle_attempt:-0.633
2026-08-08 23:08:35,887 INFO   [advantage] mean=0.000  std=1.000  min=-5.796  max=3.445
2026-08-08 23:08:35,888 INFO   [ratio] mean=0.9653  std=0.1537  min=0.0005  max=5.2456  clipped=20.5%
2026-08-08 23:08:35,888 INFO   [exec head grad norm] move_direction=0.022  exec_move=0.038  sprint=0.052  kick=0.030  kick_direction=0.008  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.047
2026-08-08 23:08:35,888 INFO   [exec continuous log_std] move_direction: start=-1.6004 end=-1.5997   kick_direction: start=-1.5942 end=-1.5934
2026-08-08 23:08:35,889 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0017≈0.10°/step  epoch≈5.9°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0017≈0.10°/step  epoch≈5.7°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-08 23:08:35,889 INFO   [exec discrete Δlogit per opt step] exec_move=0.0009  sprint=0.0010  kick=0.0012  tackle_attempt=0.0006
2026-08-08 23:08:35,889 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0009  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0020  sprint=+0.0019  kick=+0.0018  tackle_attempt=+0.0012  move_dir=+0.0314  kick_dir=+0.0144
2026-08-08 23:08:35,889 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.230 max=0.335  limit=0.4
              direction: 47/60 steps clipped (78%)  pre-clip norm mean=0.026 max=0.050  limit=0.02
2026-08-08 23:08:35,939 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=2,052,000  speed=877/s  reward=3.48
  loss     policy=0.0038  value=0.5064(x0.5)=0.2532
           entropy=6.9162  kl=0.0535
  value    V=2.72±1.19  R=2.68±1.78  adv=-0.04±1.27
  moves    mv_ls=[-1.5997] (σ≈0.20, ≈12°) g=8.05e-03  d_move=[+0.0007] (Δσ≈0.009°)
           kk_ls=[-1.5934] (σ≈0.20, ≈12°)  d_kick=[+0.0009] (Δσ≈0.010°)
  heads    move= 51 get_poss= 38 exec_move= 79 sprint= 47 kick= 25 tackle= 47 shoot= 49 hold= 49 tackle_prob=0.4682 kick_prob=0.2465
  vs       vs[win/loss/tout/miss]  vs_immobile(545): 63.7%/0.0%/3.7%/23.3%/9%
  ep_len   19.7±11.9s  (n=545, min=1.6s, max=50.0s)
  reward   get_possession=+456.00  lose_possession=-6.30  ball_out=-288.00  box_possession=+867.50
           speed_bonus=+656.83  timeout=-30.00  stamina_penalty=-2.99
  rew/ep   (mean/std/min/max per episode, 545 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.837    0.407    +0.000    +3.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.012    0.115    -1.800    +0.000
  ball_out          -0.528    1.354    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.592    1.202    +0.000    +2.500
  speed_bonus       +1.205    1.252    +0.000    +4.200
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.055    0.282    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.005    0.005    -0.023    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     459    +0.013    0.112     +3.357     2.104     +0.737      4.0461      1.681     4.034
  lose_possession       7    -0.000    0.013     +2.915     0.277     +0.488      1.3223      0.849     2.131
  ball_out            72    -0.008    0.179     -3.986     0.117     -3.718     15.1025      3.718     5.573
  box_possession     347    +0.024    0.244     +4.385     1.077     +0.919      1.7747      1.066     2.647
  speed_bonus        337    +0.018    0.213     +4.441     1.042     +0.932      1.7937      1.076     2.666
  timeout             20    -0.001    0.035     -1.508     0.006     -4.464     20.7616      4.464     5.795
  stamina_penalty     362    -0.000    0.001     +4.118     1.629     +0.665      2.7026      1.232     3.286
  gae/td   mean_return=+2.679  std_return=1.779  mean_gae=-0.041  mean_sq_td=1.6135
──────────────────────────────────────────────────────────────────────
2026-08-08 23:08:35,965 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint55.pt
2026-08-08 23:08:35,965 INFO Logging to checkpoints/phase1_run45/training_log56.txt
2026-08-08 23:08:35,966 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 23:08:50,989 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 23:08:50,990 INFO   [eval vs immobile] step=2,052,000  seeds=16x8  win=52%  mean_rew=2.508±3.219  V=2.456  gap=-0.052  outcomes={'other': 21, 'box_possession': 67, 'miss': 36, 'timeout': 4}
2026-08-08 23:08:50,991 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 23:09:02,454 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 23:09:02,455 INFO   [eval vs rules] step=2,052,000  seeds=16x8  win=2%  mean_rew=-2.213±1.661  V=0.971  gap=+3.184  outcomes={'opponent_box_possession': 95, 'other': 11, 'miss': 20, 'box_possession': 2}
2026-08-08 23:13:47,373 INFO   [KL mean=0.0535 median=0.0535 > 0.05] ratio percentiles:  p5=0.729  p25=0.912  p50=0.988  p75=1.027  p95=1.161  max=8.383
  move_dir_log_std=[-1.5990161895751953]  kick_dir_log_std=[-1.5926754474639893]
2026-08-08 23:13:47,385 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.689  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.541  kick=-0.466  t_att=-0.689
    move_dir=0.645 (min=-3.080 max=1.360)  kick_dir=0.275 (min=-1.344 max=2.016)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.46
  [worst sample] idx=197  ratio=8.288  adv=-0.839  old_lp=-3.600  new_lp=-1.486
    stored move_dir=12.4°  new_mean=5.5°  angular_diff=6.8°
    [worst sample per-head delta, sorted by |delta|] tackle_attempt:+0.025
  [top-2 highest-ratio samples]
    idx= 197  ratio=   8.288  adv=-0.839  lp: old=-3.600  new=-1.486
      rew=+0.0000  ret=+3.5529  val=+4.3923  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: tackle_attempt:+0.025
      saturation: exec_move_p_new=0.8755  sprint_p_new=0.6182  kick_p_new=0.0035  tackle_attempt_p_new=0.4843
    idx= 176  ratio=   7.403  adv=-0.322  lp: old=-3.379  new=-1.377
      rew=+0.0000  ret=+3.4903  val=+3.8123  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: tackle_attempt:+0.031
      saturation: exec_move_p_new=0.8659  sprint_p_new=0.6412  kick_p_new=0.0061  tackle_attempt_p_new=0.4706
  [best sample (highest new_lp)] idx=36  new_lp=-1.234  adv=-0.143  stored move_dir=154.2°  new_mean=150.3°
    per-head contributions: move:-0.585  tackle_attempt:-0.645
2026-08-08 23:13:47,385 INFO   [advantage] mean=0.000  std=1.000  min=-5.652  max=4.365
2026-08-08 23:13:47,386 INFO   [ratio] mean=0.9657  std=0.1569  min=0.0019  max=8.3832  clipped=20.0%
2026-08-08 23:13:47,386 INFO   [exec head grad norm] move_direction=0.030  exec_move=0.036  sprint=0.052  kick=0.034  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.058
2026-08-08 23:13:47,386 INFO   [exec continuous log_std] move_direction: start=-1.5997 end=-1.5990   kick_direction: start=-1.5934 end=-1.5927
2026-08-08 23:13:47,387 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0020≈0.11°/step  epoch≈6.7°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0018≈0.10°/step  epoch≈6.3°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-08 23:13:47,387 INFO   [exec discrete Δlogit per opt step] exec_move=0.0007  sprint=0.0013  kick=0.0010  tackle_attempt=0.0006
2026-08-08 23:13:47,387 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0012  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0020  sprint=+0.0021  kick=+0.0018  tackle_attempt=+0.0011  move_dir=+0.0312  kick_dir=+0.0140
2026-08-08 23:13:47,387 INFO   [grad clip] main: 1/60 steps clipped (2%)  pre-clip norm mean=0.267 max=0.690  limit=0.4
              direction: 53/60 steps clipped (88%)  pre-clip norm mean=0.034 max=0.236  limit=0.02
2026-08-08 23:13:47,443 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=2,088,000  speed=863/s  reward=2.50
  loss     policy=0.0055  value=0.4949(x0.5)=0.2475
           entropy=6.8831  kl=0.0535
  value    V=2.81±1.18  R=2.83±1.64  adv=0.02±1.17
  moves    mv_ls=[-1.5990] (σ≈0.20, ≈12°) g=8.21e-03  d_move=[+0.0007] (Δσ≈0.008°)
           kk_ls=[-1.5927] (σ≈0.20, ≈12°)  d_kick=[+0.0007] (Δσ≈0.008°)
  heads    move= 50 get_poss= 38 exec_move= 79 sprint= 48 kick= 23 tackle= 47 shoot= 48 hold= 48 tackle_prob=0.4725 kick_prob=0.2308
  vs       vs[win/loss/tout/miss]  vs_immobile(524): 68.1%/0.0%/1.9%/20.6%/9%
  ep_len   20.4±11.9s  (n=524, min=0.8s, max=50.0s)
  reward   get_possession=+436.00  lose_possession=-6.30  ball_out=-220.00  box_possession=+892.50
           speed_bonus=+661.33  timeout=-15.00  stamina_penalty=-3.28
  rew/ep   (mean/std/min/max per episode, 524 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.832    0.413    +0.000    +3.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.012    0.117    -1.800    +0.000
  ball_out          -0.420    1.226    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.703    1.165    +0.000    +2.500
  speed_bonus       +1.262    1.235    +0.000    +4.127
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.029    0.205    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.028    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     439    +0.012    0.110     +3.463     1.879     +0.749      3.6508      1.572     4.019
  lose_possession       7    -0.000    0.013     +2.078     1.292     -1.147      2.4943      1.216     2.841
  ball_out            55    -0.006    0.156     -3.982     0.134     -3.691     14.8712      3.691     5.649
  box_possession     357    +0.025    0.248     +4.349     1.073     +1.004      2.3746      1.191     3.169
  speed_bonus        338    +0.018    0.212     +4.453     1.006     +1.059      2.4740      1.226     3.198
  timeout             10    -0.000    0.025     -1.510     0.006     -4.598     21.2935      4.598     5.155
  stamina_penalty     360    -0.000    0.001     +4.211     1.427     +0.849      2.9065      1.289     3.588
  gae/td   mean_return=+2.835  std_return=1.637  mean_gae=+0.020  mean_sq_td=1.3711
──────────────────────────────────────────────────────────────────────
2026-08-08 23:13:47,470 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint56.pt
2026-08-08 23:13:47,470 INFO Logging to checkpoints/phase1_run45/training_log57.txt
2026-08-08 23:13:47,471 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 23:14:00,574 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 23:14:00,576 INFO   [eval vs immobile] step=2,088,000  seeds=16x8  win=56%  mean_rew=2.861±3.168  V=2.513  gap=-0.348  outcomes={'other': 22, 'box_possession': 72, 'miss': 33, 'timeout': 1}
2026-08-08 23:14:00,577 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 23:14:11,768 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 23:14:11,769 INFO   [eval vs rules] step=2,088,000  seeds=16x8  win=1%  mean_rew=-2.171±1.485  V=0.774  gap=+2.945  outcomes={'opponent_box_possession': 95, 'other': 12, 'miss': 20, 'box_possession': 1}
2026-08-08 23:18:58,096 INFO   [KL mean=0.0549 median=0.0551 > 0.05] ratio percentiles:  p5=0.722  p25=0.908  p50=0.986  p75=1.028  p95=1.163  max=7.744
  move_dir_log_std=[-1.5981873273849487]  kick_dir_log_std=[-1.5919040441513062]
2026-08-08 23:18:58,109 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.690  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.530  kick=-0.275  t_att=-0.696
    move_dir=0.603 (min=-5.228 max=1.358)  kick_dir=0.159 (min=-1.977 max=2.011)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.46
  [worst sample] idx=129  ratio=9.277  adv=+0.472  old_lp=-3.729  new_lp=-1.502
    stored move_dir=-7.1°  new_mean=-8.2°  angular_diff=1.0°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 129  ratio=   9.277  adv=+0.472  lp: old=-3.729  new=-1.502
      rew=+0.0000  ret=+4.7428  val=+4.2711  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8971  sprint_p_new=0.6525  kick_p_new=0.0031  tackle_attempt_p_new=0.4820
    idx=  82  ratio=   8.002  adv=+0.059  lp: old=-3.443  new=-1.363
      rew=+0.0000  ret=+4.6883  val=+4.6289  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.8846  sprint_p_new=0.6414  kick_p_new=0.0036  tackle_attempt_p_new=0.4907
  [best sample (highest new_lp)] idx=243  new_lp=-1.208  adv=+0.267  stored move_dir=-42.3°  new_mean=-40.3°
    per-head contributions: tackle_attempt:-0.593  move:-0.610
2026-08-08 23:18:58,110 INFO   [advantage] mean=-0.000  std=1.000  min=-6.226  max=4.146
2026-08-08 23:18:58,110 INFO   [ratio] mean=0.9643  std=0.1549  min=0.0005  max=7.7439  clipped=20.6%
2026-08-08 23:18:58,111 INFO   [exec head grad norm] move_direction=0.025  exec_move=0.043  sprint=0.048  kick=0.033  kick_direction=0.008  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.052
2026-08-08 23:18:58,111 INFO   [exec continuous log_std] move_direction: start=-1.5990 end=-1.5982   kick_direction: start=-1.5927 end=-1.5919
2026-08-08 23:18:58,111 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0019≈0.11°/step  epoch≈6.4°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0016≈0.09°/step  epoch≈5.5°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-08 23:18:58,111 INFO   [exec discrete Δlogit per opt step] exec_move=0.0010  sprint=0.0009  kick=0.0015  tackle_attempt=0.0005
2026-08-08 23:18:58,111 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0011  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0015  sprint=+0.0018  kick=+0.0021  tackle_attempt=+0.0012  move_dir=+0.0342  kick_dir=+0.0130
2026-08-08 23:18:58,112 INFO   [grad clip] main: 1/60 steps clipped (2%)  pre-clip norm mean=0.241 max=0.731  limit=0.4
              direction: 48/60 steps clipped (80%)  pre-clip norm mean=0.029 max=0.117  limit=0.02
2026-08-08 23:18:58,162 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=2,124,000  speed=867/s  reward=3.60
  loss     policy=0.0056  value=0.5035(x0.5)=0.2517
           entropy=6.8837  kl=0.0549
  value    V=2.82±1.19  R=2.88±1.61  adv=0.06±1.15
  moves    mv_ls=[-1.5982] (σ≈0.20, ≈12°) g=9.78e-03  d_move=[+0.0008] (Δσ≈0.010°)
           kk_ls=[-1.5919] (σ≈0.20, ≈12°)  d_kick=[+0.0008] (Δσ≈0.009°)
  heads    move= 50 get_poss= 39 exec_move= 79 sprint= 49 kick= 23 tackle= 47 shoot= 48 hold= 48 tackle_prob=0.4709 kick_prob=0.2265
  vs       vs[win/loss/tout/miss]  vs_immobile(526): 68.6%/0.0%/3.2%/19.4%/9%
  ep_len   20.5±12.5s  (n=526, min=0.3s, max=50.0s)
  reward   get_possession=+442.00  lose_possession=-1.80  ball_out=-196.00  box_possession=+902.50
           speed_bonus=+676.78  timeout=-25.50  stamina_penalty=-3.54
  rew/ep   (mean/std/min/max per episode, 526 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.840    0.377    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.003    0.055    -0.900    +0.000
  ball_out          -0.373    1.163    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.716    1.160    +0.000    +2.500
  speed_bonus       +1.287    1.262    +0.000    +4.072
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.048    0.265    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.007    0.006    -0.025    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     444    +0.012    0.110     +3.485     2.056     +0.629      3.9040      1.589     4.522
  lose_possession       2    -0.000    0.007     +2.973     0.603     -0.734      1.1019      0.750     1.411
  ball_out            49    -0.005    0.147     -4.000     0.000     -3.404     13.1146      3.404     5.514
  box_possession     361    +0.025    0.249     +4.365     1.104     +0.911      1.9413      1.131     2.750
  speed_bonus        349    +0.019    0.217     +4.430     1.066     +0.950      2.0030      1.160     2.759
  timeout             17    -0.001    0.033     -1.511     0.006     -4.550     22.8637      4.601     6.069
  stamina_penalty     371    -0.000    0.001     +4.116     1.636     +0.670      2.9268      1.300     3.335
  gae/td   mean_return=+2.877  std_return=1.611  mean_gae=+0.059  mean_sq_td=1.3192
──────────────────────────────────────────────────────────────────────
2026-08-08 23:18:58,187 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint57.pt
2026-08-08 23:18:58,187 INFO Logging to checkpoints/phase1_run45/training_log58.txt
2026-08-08 23:18:58,189 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 23:19:12,814 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 23:19:12,815 INFO   [eval vs immobile] step=2,124,000  seeds=16x8  win=55%  mean_rew=2.692±3.025  V=2.667  gap=-0.024  outcomes={'other': 20, 'box_possession': 70, 'timeout': 6, 'miss': 32}
2026-08-08 23:19:12,816 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 23:19:23,751 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 23:19:23,753 INFO   [eval vs rules] step=2,124,000  seeds=16x8  win=2%  mean_rew=-2.152±1.757  V=1.024  gap=+3.176  outcomes={'other': 11, 'opponent_box_possession': 95, 'miss': 19, 'box_possession': 3}
2026-08-08 23:24:06,033 INFO   [KL mean=0.0558 median=0.0556 > 0.05] ratio percentiles:  p5=0.720  p25=0.907  p50=0.984  p75=1.025  p95=1.160  max=5.738
  move_dir_log_std=[-1.597443699836731]  kick_dir_log_std=[-1.5911673307418823]
2026-08-08 23:24:06,046 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.691  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.504  kick=-0.544  t_att=-0.692
    move_dir=0.490 (min=-3.821 max=1.357)  kick_dir=0.342 (min=-4.421 max=2.017)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.55
  [worst sample] idx=217  ratio=9.893  adv=+1.923  old_lp=-3.626  new_lp=-1.334
    stored move_dir=-23.8°  new_mean=0.3°  angular_diff=24.1°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx= 217  ratio=   9.893  adv=+1.923  lp: old=-3.626  new=-1.334
      rew=+0.0000  ret=+6.0813  val=+4.1583  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9018  sprint_p_new=0.6475  kick_p_new=0.0026  tackle_attempt_p_new=0.4915
    idx= 220  ratio=   8.756  adv=+2.616  lp: old=-3.574  new=-1.404
      rew=+6.3830  ret=+6.3830  val=+3.7670  outcome=terminal:box_possession
      rew_breakdown: box=+2.500  spd=+3.890  stam=-0.007
      head_deltas: 
      saturation: exec_move_p_new=0.8905  sprint_p_new=0.6211  kick_p_new=0.0025  tackle_attempt_p_new=0.4790
  [best sample (highest new_lp)] idx=135  new_lp=-1.261  adv=-4.601  stored move_dir=173.0°  new_mean=143.1°
    per-head contributions: move:-0.621  tackle_attempt:-0.636
2026-08-08 23:24:06,046 INFO   [advantage] mean=0.000  std=1.000  min=-6.384  max=3.940
2026-08-08 23:24:06,048 INFO   [ratio] mean=0.9621  std=0.1506  min=0.0088  max=5.7375  clipped=20.6%
2026-08-08 23:24:06,048 INFO   [exec head grad norm] move_direction=0.031  exec_move=0.039  sprint=0.045  kick=0.031  kick_direction=0.008  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.052
2026-08-08 23:24:06,048 INFO   [exec continuous log_std] move_direction: start=-1.5982 end=-1.5974   kick_direction: start=-1.5919 end=-1.5912
2026-08-08 23:24:06,049 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0025≈0.14°/step  epoch≈8.5°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0021≈0.12°/step  epoch≈7.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-08 23:24:06,049 INFO   [exec discrete Δlogit per opt step] exec_move=0.0008  sprint=0.0010  kick=0.0011  tackle_attempt=0.0006
2026-08-08 23:24:06,049 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0008  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0018  sprint=+0.0012  kick=+0.0019  tackle_attempt=+0.0012  move_dir=+0.0360  kick_dir=+0.0131
2026-08-08 23:24:06,050 INFO   [grad clip] main: 1/60 steps clipped (2%)  pre-clip norm mean=0.245 max=0.450  limit=0.4
              direction: 55/60 steps clipped (92%)  pre-clip norm mean=0.035 max=0.087  limit=0.02
2026-08-08 23:24:06,100 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=2,160,000  speed=870/s  reward=3.48
  loss     policy=0.0060  value=0.5175(x0.5)=0.2588
           entropy=6.8992  kl=0.0558
  value    V=2.90±1.03  R=2.91±1.58  adv=0.01±1.14
  moves    mv_ls=[-1.5974] (σ≈0.20, ≈12°) g=9.53e-03  d_move=[+0.0007] (Δσ≈0.009°)
           kk_ls=[-1.5912] (σ≈0.20, ≈12°)  d_kick=[+0.0007] (Δσ≈0.009°)
  heads    move= 49 get_poss= 38 exec_move= 80 sprint= 50 kick= 23 tackle= 47 shoot= 48 hold= 48 tackle_prob=0.4721 kick_prob=0.2351
  vs       vs[win/loss/tout/miss]  vs_immobile(530): 67.7%/0.0%/2.6%/18.9%/11%
  ep_len   20.2±12.0s  (n=530, min=0.6s, max=50.0s)
  reward   get_possession=+433.00  lose_possession=-4.50  ball_out=-188.00  box_possession=+897.50
           speed_bonus=+688.13  timeout=-21.00  stamina_penalty=-3.72
  rew/ep   (mean/std/min/max per episode, 530 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.817    0.406    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.008    0.087    -0.900    +0.000
  ball_out          -0.355    1.137    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.693    1.169    +0.000    +2.500
  speed_bonus       +1.298    1.284    +0.000    +3.985
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.040    0.241    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.007    0.006    -0.029    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     436    +0.012    0.109     +3.626     1.895     +0.586      3.4331      1.460     4.510
  lose_possession       5    -0.000    0.011     +2.446     1.551     -0.580      2.7049      0.945     2.984
  ball_out            47    -0.005    0.144     -3.979     0.144     -4.438     21.7647      4.438     6.798
  box_possession     359    +0.025    0.248     +4.418     1.118     +0.918      1.8269      1.090     2.504
  speed_bonus        349    +0.019    0.221     +4.473     1.085     +0.955      1.8740      1.112     2.515
  timeout             14    -0.001    0.030     -1.510     0.004     -4.350     19.6084      4.350     5.479
  stamina_penalty     371    -0.000    0.001     +4.203     1.573     +0.726      2.5069      1.218     3.049
  gae/td   mean_return=+2.907  std_return=1.578  mean_gae=+0.008  mean_sq_td=1.3036
──────────────────────────────────────────────────────────────────────
2026-08-08 23:24:06,128 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint58.pt
2026-08-08 23:24:06,128 INFO Logging to checkpoints/phase1_run45/training_log59.txt
2026-08-08 23:24:06,129 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 23:24:19,956 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 23:24:19,959 INFO   [eval vs immobile] step=2,160,000  seeds=16x8  win=49%  mean_rew=2.264±3.111  V=2.664  gap=+0.400  outcomes={'other': 20, 'box_possession': 63, 'timeout': 2, 'miss': 43}
2026-08-08 23:24:19,960 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 23:24:30,715 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 23:24:30,716 INFO   [eval vs rules] step=2,160,000  seeds=16x8  win=4%  mean_rew=-1.953±2.193  V=1.055  gap=+3.008  outcomes={'other': 12, 'opponent_box_possession': 96, 'box_possession': 5, 'miss': 15}
2026-08-08 23:29:13,287 INFO   [advantage] mean=-0.000  std=1.000  min=-5.940  max=3.155
2026-08-08 23:29:13,288 INFO   [ratio] mean=0.9669  std=0.1447  min=0.0026  max=3.5394  clipped=19.3%
2026-08-08 23:29:13,288 INFO   [exec head grad norm] move_direction=0.027  exec_move=0.045  sprint=0.050  kick=0.031  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.057
2026-08-08 23:29:13,288 INFO   [exec continuous log_std] move_direction: start=-1.5974 end=-1.5967   kick_direction: start=-1.5912 end=-1.5905
2026-08-08 23:29:13,288 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0017≈0.10°/step  epoch≈5.8°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0017≈0.10°/step  epoch≈5.8°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-08 23:29:13,289 INFO   [exec discrete Δlogit per opt step] exec_move=0.0010  sprint=0.0009  kick=0.0011  tackle_attempt=0.0006
2026-08-08 23:29:13,289 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0008  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0014  sprint=+0.0014  kick=+0.0013  tackle_attempt=+0.0009  move_dir=+0.0312  kick_dir=+0.0125
2026-08-08 23:29:13,289 INFO   [grad clip] main: 2/60 steps clipped (3%)  pre-clip norm mean=0.249 max=0.430  limit=0.4
              direction: 52/60 steps clipped (87%)  pre-clip norm mean=0.030 max=0.056  limit=0.02
2026-08-08 23:29:13,340 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=2,196,000  speed=876/s  reward=3.79
  loss     policy=0.0037  value=0.5116(x0.5)=0.2558
           entropy=6.9056  kl=0.0494
  value    V=2.91±1.14  R=2.83±1.74  adv=-0.08±1.26
  moves    mv_ls=[-1.5967] (σ≈0.20, ≈12°) g=8.31e-03  d_move=[+0.0007] (Δσ≈0.008°)
           kk_ls=[-1.5905] (σ≈0.20, ≈12°)  d_kick=[+0.0007] (Δσ≈0.008°)
  heads    move= 50 get_poss= 38 exec_move= 80 sprint= 49 kick= 24 tackle= 47 shoot= 48 hold= 49 tackle_prob=0.4734 kick_prob=0.2367
  vs       vs[win/loss/tout/miss]  vs_immobile(547): 64.0%/0.4%/3.1%/20.8%/12%
  ep_len   19.6±11.7s  (n=547, min=0.7s, max=50.0s)
  reward   get_possession=+450.00  lose_possession=-3.60  ball_out=-260.00  box_possession=+875.00
           speed_bonus=+684.35  opponent_box=-6.00  timeout=-25.50  stamina_penalty=-3.40
  rew/ep   (mean/std/min/max per episode, 547 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.823    0.401    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.007    0.077    -0.900    +0.000
  ball_out          -0.475    1.294    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.600    1.200    +0.000    +2.500
  speed_bonus       +1.251    1.292    +0.000    +4.048
  opponent_box      -0.011    0.181    -3.000    +0.000
  timeout           -0.047    0.260    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.027    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     452    +0.013    0.111     +3.469     2.073     +0.600      3.7169      1.578     3.924
  lose_possession       4    -0.000    0.009     +1.852     1.771     -0.056      3.4155      1.541     2.778
  ball_out            65    -0.007    0.170     -3.985     0.123     -4.134     18.9030      4.134     6.180
  box_possession     350    +0.024    0.245     +4.449     1.106     +0.739      1.5379      1.013     2.415
  speed_bonus        337    +0.019    0.221     +4.521     1.061     +0.773      1.5776      1.031     2.430
  opponent_box         2    -0.000    0.022     -3.002     0.001     -5.508     30.3557      5.508     5.617
  timeout             17    -0.001    0.033     -1.510     0.005     -4.518     21.3846      4.518     5.732
  stamina_penalty     364    -0.000    0.001     +4.163     1.713     +0.475      2.5882      1.192     2.918
  gae/td   mean_return=+2.827  std_return=1.740  mean_gae=-0.081  mean_sq_td=1.5910
──────────────────────────────────────────────────────────────────────
2026-08-08 23:29:13,368 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint59.pt
2026-08-08 23:29:13,369 INFO Logging to checkpoints/phase1_run45/training_log60.txt
2026-08-08 23:29:13,370 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 23:29:26,573 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 23:29:26,575 INFO   [eval vs immobile] step=2,196,000  seeds=16x8  win=51%  mean_rew=2.552±3.232  V=2.542  gap=-0.010  outcomes={'other': 22, 'box_possession': 65, 'opponent_box_possession': 1, 'miss': 34, 'timeout': 6}
2026-08-08 23:29:26,576 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 23:29:38,068 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 23:29:38,069 INFO   [eval vs rules] step=2,196,000  seeds=16x8  win=5%  mean_rew=-1.758±2.384  V=0.880  gap=+2.638  outcomes={'miss': 19, 'opponent_box_possession': 91, 'other': 11, 'box_possession': 7}
2026-08-08 23:34:29,276 INFO   [advantage] mean=0.000  std=1.000  min=-6.146  max=3.551
2026-08-08 23:34:29,277 INFO   [ratio] mean=0.9681  std=0.1402  min=0.0084  max=6.3332  clipped=18.0%
2026-08-08 23:34:29,278 INFO   [exec head grad norm] move_direction=0.025  exec_move=0.039  sprint=0.048  kick=0.032  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.052
2026-08-08 23:34:29,278 INFO   [exec continuous log_std] move_direction: start=-1.5967 end=-1.5959   kick_direction: start=-1.5905 end=-1.5897
2026-08-08 23:34:29,278 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0015≈0.09°/step  epoch≈5.3°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0014≈0.08°/step  epoch≈5.0°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-08 23:34:29,278 INFO   [exec discrete Δlogit per opt step] exec_move=0.0009  sprint=0.0009  kick=0.0011  tackle_attempt=0.0005
2026-08-08 23:34:29,278 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0011  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0015  sprint=+0.0014  kick=+0.0020  tackle_attempt=+0.0010  move_dir=+0.0275  kick_dir=+0.0124
2026-08-08 23:34:29,279 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.238 max=0.336  limit=0.4
              direction: 55/60 steps clipped (92%)  pre-clip norm mean=0.029 max=0.048  limit=0.02
2026-08-08 23:34:29,329 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=2,232,000  speed=863/s  reward=4.07
  loss     policy=0.0049  value=0.5137(x0.5)=0.2568
           entropy=6.9040  kl=0.0470
  value    V=2.86±1.16  R=2.86±1.66  adv=0.01±1.19
  moves    mv_ls=[-1.5959] (σ≈0.20, ≈12°) g=9.59e-03  d_move=[+0.0008] (Δσ≈0.010°)
           kk_ls=[-1.5897] (σ≈0.20, ≈12°)  d_kick=[+0.0008] (Δσ≈0.009°)
  heads    move= 50 get_poss= 38 exec_move= 79 sprint= 49 kick= 24 tackle= 47 shoot= 49 hold= 49 tackle_prob=0.4768 kick_prob=0.2363
  vs       vs[win/loss/tout/miss]  vs_immobile(539): 68.1%/0.4%/2.4%/19.3%/10%
  ep_len   19.8±11.5s  (n=539, min=1.1s, max=50.0s)
  reward   get_possession=+455.00  lose_possession=-3.60  ball_out=-224.00  box_possession=+917.50
           speed_bonus=+685.15  opponent_box=-6.00  timeout=-19.50  stamina_penalty=-3.48
  rew/ep   (mean/std/min/max per episode, 539 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.844    0.378    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.007    0.095    -1.800    +0.000
  ball_out          -0.416    1.221    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.702    1.165    +0.000    +2.500
  speed_bonus       +1.271    1.250    +0.000    +4.326
  opponent_box      -0.011    0.182    -3.000    +0.000
  timeout           -0.036    0.230    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.022    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     458    +0.013    0.112     +3.511     1.970     +0.696      3.5444      1.555     3.794
  lose_possession       4    -0.000    0.009     +1.798     1.978     -0.965      5.7148      1.575     4.101
  ball_out            56    -0.006    0.158     -3.982     0.132     -3.467     13.7415      3.467     5.840
  box_possession     367    +0.025    0.251     +4.361     1.086     +1.063      2.4068      1.245     3.067
  speed_bonus        352    +0.019    0.217     +4.440     1.036     +1.094      2.4958      1.279     3.074
  opponent_box         2    -0.000    0.022     -3.002     0.001     -5.065     25.6718      5.065     5.173
  timeout             13    -0.001    0.028     -1.513     0.006     -4.772     23.3547      4.772     5.722
  stamina_penalty     376    -0.000    0.001     +4.138     1.598     +0.812      3.2098      1.374     3.813
  gae/td   mean_return=+2.861  std_return=1.664  mean_gae=+0.006  mean_sq_td=1.4278
──────────────────────────────────────────────────────────────────────
2026-08-08 23:34:29,358 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint60.pt
2026-08-08 23:34:29,358 INFO Logging to checkpoints/phase1_run45/training_log61.txt
2026-08-08 23:34:29,359 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 23:34:43,040 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 23:34:43,042 INFO   [eval vs immobile] step=2,232,000  seeds=16x8  win=53%  mean_rew=2.564±3.247  V=2.501  gap=-0.063  outcomes={'other': 24, 'box_possession': 68, 'miss': 34, 'timeout': 2}
2026-08-08 23:34:43,043 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 23:34:54,529 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 23:34:54,531 INFO   [eval vs rules] step=2,232,000  seeds=16x8  win=4%  mean_rew=-1.868±2.197  V=0.875  gap=+2.744  outcomes={'opponent_box_possession': 91, 'other': 14, 'box_possession': 5, 'miss': 18}
2026-08-08 23:39:40,626 INFO   [advantage] mean=-0.000  std=1.000  min=-5.439  max=4.320
2026-08-08 23:39:40,627 INFO   [ratio] mean=0.9690  std=0.1410  min=0.0068  max=5.8780  clipped=18.6%
2026-08-08 23:39:40,628 INFO   [exec head grad norm] move_direction=0.025  exec_move=0.042  sprint=0.051  kick=0.029  kick_direction=0.008  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.063
2026-08-08 23:39:40,628 INFO   [exec continuous log_std] move_direction: start=-1.5959 end=-1.5951   kick_direction: start=-1.5897 end=-1.5890
2026-08-08 23:39:40,628 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0014≈0.08°/step  epoch≈4.7°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0014≈0.08°/step  epoch≈4.8°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-08 23:39:40,628 INFO   [exec discrete Δlogit per opt step] exec_move=0.0011  sprint=0.0009  kick=0.0013  tackle_attempt=0.0007
2026-08-08 23:39:40,628 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0008  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0011  sprint=+0.0016  kick=+0.0015  tackle_attempt=+0.0007  move_dir=+0.0282  kick_dir=+0.0124
2026-08-08 23:39:40,629 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.248 max=0.388  limit=0.4
              direction: 47/60 steps clipped (78%)  pre-clip norm mean=0.029 max=0.052  limit=0.02
2026-08-08 23:39:40,684 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=2,268,000  speed=861/s  reward=4.23
  loss     policy=0.0055  value=0.5007(x0.5)=0.2504
           entropy=6.9152  kl=0.0464
  value    V=2.79±1.18  R=2.81±1.69  adv=0.02±1.19
  moves    mv_ls=[-1.5951] (σ≈0.20, ≈12°) g=9.12e-03  d_move=[+0.0008] (Δσ≈0.009°)
           kk_ls=[-1.5890] (σ≈0.20, ≈12°)  d_kick=[+0.0007] (Δσ≈0.008°)
  heads    move= 50 get_poss= 38 exec_move= 80 sprint= 48 kick= 24 tackle= 48 shoot= 48 hold= 48 tackle_prob=0.4826 kick_prob=0.2398
  vs       vs[win/loss/tout/miss]  vs_immobile(542): 66.8%/0.4%/2.6%/21.8%/8%
  ep_len   19.8±11.7s  (n=542, min=0.9s, max=50.0s)
  reward   get_possession=+450.00  lose_possession=-4.50  ball_out=-240.00  box_possession=+905.00
           speed_bonus=+667.39  opponent_box=-6.00  timeout=-21.00  stamina_penalty=-3.31
  rew/ep   (mean/std/min/max per episode, 542 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.830    0.399    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.008    0.086    -0.900    +0.000
  ball_out          -0.443    1.255    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.670    1.177    +0.000    +2.500
  speed_bonus       +1.231    1.237    +0.000    +3.932
  opponent_box      -0.011    0.182    -3.000    +0.000
  timeout           -0.039    0.238    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.024    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     453    +0.013    0.111     +3.378     2.087     +0.564      3.8077      1.597     4.293
  lose_possession       5    -0.000    0.011     +2.744     1.043     -0.280      0.7657      0.703     1.454
  ball_out            60    -0.007    0.163     -4.000     0.000     -3.560     14.0607      3.560     5.808
  box_possession     362    +0.025    0.249     +4.338     1.080     +1.025      2.2319      1.226     2.890
  speed_bonus        347    +0.019    0.213     +4.417     1.032     +1.082      2.3189      1.261     2.904
  opponent_box         2    -0.000    0.022     -3.002     0.001     -5.045     26.4669      5.045     5.950
  timeout             14    -0.001    0.030     -1.511     0.006     -3.941     18.0837      3.941     5.271
  stamina_penalty     376    -0.000    0.001     +4.088     1.616     +0.811      2.9617      1.351     3.322
  gae/td   mean_return=+2.810  std_return=1.692  mean_gae=+0.022  mean_sq_td=1.4077
──────────────────────────────────────────────────────────────────────
2026-08-08 23:39:40,711 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint61.pt
2026-08-08 23:39:40,712 INFO Logging to checkpoints/phase1_run45/training_log62.txt
2026-08-08 23:39:40,713 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 23:39:55,679 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 23:39:55,681 INFO   [eval vs immobile] step=2,268,000  seeds=16x8  win=50%  mean_rew=2.435±3.072  V=2.494  gap=+0.059  outcomes={'other': 23, 'box_possession': 64, 'timeout': 9, 'miss': 32}
2026-08-08 23:39:55,682 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 23:40:06,670 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 23:40:06,671 INFO   [eval vs rules] step=2,268,000  seeds=16x8  win=3%  mean_rew=-2.087±1.941  V=0.891  gap=+2.978  outcomes={'opponent_box_possession': 96, 'other': 14, 'box_possession': 4, 'miss': 14}
2026-08-08 23:44:56,781 INFO   [advantage] mean=-0.000  std=1.000  min=-6.015  max=4.727
2026-08-08 23:44:56,782 INFO   [ratio] mean=0.9692  std=0.1424  min=0.0081  max=6.3174  clipped=18.1%
2026-08-08 23:44:56,782 INFO   [exec head grad norm] move_direction=0.027  exec_move=0.038  sprint=0.051  kick=0.034  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.055
2026-08-08 23:44:56,782 INFO   [exec continuous log_std] move_direction: start=-1.5951 end=-1.5944   kick_direction: start=-1.5890 end=-1.5883
2026-08-08 23:44:56,782 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0013≈0.07°/step  epoch≈4.5°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0012≈0.07°/step  epoch≈4.2°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-08 23:44:56,782 INFO   [exec discrete Δlogit per opt step] exec_move=0.0006  sprint=0.0009  kick=0.0009  tackle_attempt=0.0005
2026-08-08 23:44:56,783 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0010  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0014  sprint=+0.0017  kick=+0.0015  tackle_attempt=+0.0007  move_dir=+0.0278  kick_dir=+0.0119
2026-08-08 23:44:56,783 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.253 max=0.378  limit=0.4
              direction: 50/60 steps clipped (83%)  pre-clip norm mean=0.030 max=0.076  limit=0.02
2026-08-08 23:44:56,828 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=2,304,000  speed=854/s  reward=2.51
  loss     policy=0.0058  value=0.4678(x0.5)=0.2339
           entropy=6.9031  kl=0.0461
  value    V=2.78±1.21  R=2.89±1.59  adv=0.11±1.09
  moves    mv_ls=[-1.5944] (σ≈0.20, ≈12°) g=8.21e-03  d_move=[+0.0007] (Δσ≈0.009°)
           kk_ls=[-1.5883] (σ≈0.20, ≈12°)  d_kick=[+0.0007] (Δσ≈0.008°)
  heads    move= 49 get_poss= 39 exec_move= 79 sprint= 48 kick= 23 tackle= 49 shoot= 48 hold= 49 tackle_prob=0.4878 kick_prob=0.2335
  vs       vs[win/loss/tout/miss]  vs_immobile(554): 67.3%/0.2%/2.5%/18.2%/12%
  ep_len   19.4±12.0s  (n=554, min=0.5s, max=50.0s)
  reward   get_possession=+441.00  lose_possession=-4.50  ball_out=-168.00  box_possession=+932.50
           speed_bonus=+694.10  opponent_box=-3.00  timeout=-21.00  stamina_penalty=-3.42
  rew/ep   (mean/std/min/max per episode, 554 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.796    0.425    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.008    0.085    -0.900    +0.000
  ball_out          -0.303    1.059    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.683    1.173    +0.000    +2.500
  speed_bonus       +1.253    1.224    +0.000    +3.938
  opponent_box      -0.005    0.127    -3.000    +0.000
  timeout           -0.038    0.235    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.026    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     444    +0.012    0.110     +3.561     1.952     +0.726      3.7531      1.575     4.310
  lose_possession       5    -0.000    0.011     +2.672     0.542     +0.132      0.4069      0.545     0.968
  ball_out            42    -0.005    0.137     -3.976     0.152     -3.545     14.3906      3.545     5.729
  box_possession     373    +0.026    0.253     +4.355     1.042     +0.922      1.8881      1.085     2.788
  speed_bonus        362    +0.019    0.216     +4.411     1.005     +0.953      1.9283      1.101     2.788
  opponent_box         1    -0.000    0.016     -3.003     0.000     -5.239     27.4504      5.239     5.239
  timeout             14    -0.001    0.030     -1.513     0.005     -4.445     20.5836      4.445     5.614
  stamina_penalty     383    -0.000    0.001     +4.139     1.544     +0.710      2.6475      1.220     3.306
  gae/td   mean_return=+2.889  std_return=1.590  mean_gae=+0.111  mean_sq_td=1.2022
──────────────────────────────────────────────────────────────────────
2026-08-08 23:44:56,854 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint62.pt
2026-08-08 23:44:56,855 INFO Logging to checkpoints/phase1_run45/training_log63.txt
2026-08-08 23:44:56,856 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 23:45:10,296 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 23:45:10,298 INFO   [eval vs immobile] step=2,304,000  seeds=16x8  win=55%  mean_rew=2.681±3.138  V=2.668  gap=-0.013  outcomes={'other': 22, 'miss': 32, 'box_possession': 71, 'timeout': 3}
2026-08-08 23:45:10,299 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 23:45:20,102 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 23:45:20,103 INFO   [eval vs rules] step=2,304,000  seeds=16x8  win=2%  mean_rew=-2.181±1.666  V=0.837  gap=+3.018  outcomes={'other': 13, 'opponent_box_possession': 97, 'miss': 16, 'box_possession': 2}
2026-08-08 23:50:09,984 INFO   [advantage] mean=-0.000  std=1.000  min=-6.265  max=5.115
2026-08-08 23:50:09,985 INFO   [ratio] mean=0.9675  std=0.1397  min=0.0059  max=3.6066  clipped=18.1%
2026-08-08 23:50:09,985 INFO   [exec head grad norm] move_direction=0.031  exec_move=0.036  sprint=0.043  kick=0.033  kick_direction=0.008  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.052
2026-08-08 23:50:09,985 INFO   [exec continuous log_std] move_direction: start=-1.5944 end=-1.5937   kick_direction: start=-1.5883 end=-1.5877
2026-08-08 23:50:09,985 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0021≈0.12°/step  epoch≈7.4°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0019≈0.11°/step  epoch≈6.5°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-08 23:50:09,986 INFO   [exec discrete Δlogit per opt step] exec_move=0.0008  sprint=0.0010  kick=0.0010  tackle_attempt=0.0006
2026-08-08 23:50:09,986 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0009  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0015  sprint=+0.0013  kick=+0.0018  tackle_attempt=+0.0011  move_dir=+0.0290  kick_dir=+0.0121
2026-08-08 23:50:09,986 INFO   [grad clip] main: 2/60 steps clipped (3%)  pre-clip norm mean=0.244 max=0.452  limit=0.4
              direction: 52/60 steps clipped (87%)  pre-clip norm mean=0.034 max=0.104  limit=0.02
2026-08-08 23:50:10,044 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=2,340,000  speed=871/s  reward=2.76
  loss     policy=0.0049  value=0.4720(x0.5)=0.2360
           entropy=6.9186  kl=0.0478
  value    V=2.85±1.20  R=2.85±1.73  adv=-0.00±1.20
  moves    mv_ls=[-1.5937] (σ≈0.20, ≈12°) g=7.87e-03  d_move=[+0.0006] (Δσ≈0.007°)
           kk_ls=[-1.5877] (σ≈0.20, ≈12°)  d_kick=[+0.0007] (Δσ≈0.008°)
  heads    move= 50 get_poss= 38 exec_move= 80 sprint= 48 kick= 24 tackle= 48 shoot= 49 hold= 49 tackle_prob=0.4876 kick_prob=0.2413
  vs       vs[win/loss/tout/miss]  vs_immobile(570): 65.3%/0.0%/1.4%/21.2%/12%
  ep_len   18.8±11.3s  (n=570, min=0.6s, max=50.0s)
  reward   get_possession=+459.00  lose_possession=-1.80  ball_out=-256.00  box_possession=+930.00
           speed_bonus=+713.90  timeout=-12.00  stamina_penalty=-3.28
  rew/ep   (mean/std/min/max per episode, 570 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.805    0.405    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.003    0.053    -0.900    +0.000
  ball_out          -0.449    1.263    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.632    1.190    +0.000    +2.500
  speed_bonus       +1.252    1.287    +0.000    +4.168
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.021    0.176    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.025    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     463    +0.013    0.113     +3.438     2.169     +0.476      4.1991      1.626     4.619
  lose_possession       2    -0.000    0.007     +2.922     0.025     -0.082      0.0081      0.082     0.115
  ball_out            64    -0.007    0.169     -3.953     0.211     -3.962     17.4885      3.962     6.195
  box_possession     372    +0.026    0.253     +4.416     1.114     +0.790      1.7601      1.084     2.378
  speed_bonus        353    +0.020    0.225     +4.514     1.056     +0.838      1.7831      1.095     2.332
  timeout              8    -0.000    0.022     -1.513     0.005     -4.896     24.3340      4.896     5.828
  stamina_penalty     370    -0.000    0.001     +4.314     1.400     +0.676      2.2729      1.176     2.731
  gae/td   mean_return=+2.849  std_return=1.730  mean_gae=-0.002  mean_sq_td=1.4480
──────────────────────────────────────────────────────────────────────
2026-08-08 23:50:10,073 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint63.pt
2026-08-08 23:50:10,073 INFO Logging to checkpoints/phase1_run45/training_log64.txt
2026-08-08 23:50:10,075 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 23:50:25,130 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 23:50:25,132 INFO   [eval vs immobile] step=2,340,000  seeds=16x8  win=45%  mean_rew=1.890±3.371  V=2.503  gap=+0.612  outcomes={'other': 25, 'box_possession': 57, 'miss': 43, 'timeout': 3}
2026-08-08 23:50:25,133 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 23:50:35,401 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 23:50:35,402 INFO   [eval vs rules] step=2,340,000  seeds=16x8  win=5%  mean_rew=-1.901±2.177  V=0.949  gap=+2.850  outcomes={'opponent_box_possession': 93, 'miss': 16, 'other': 13, 'box_possession': 6}
2026-08-08 23:55:20,505 INFO   [advantage] mean=0.000  std=1.000  min=-4.879  max=3.503
2026-08-08 23:55:20,506 INFO   [ratio] mean=0.9686  std=0.1388  min=0.0082  max=3.7198  clipped=17.8%
2026-08-08 23:55:20,506 INFO   [exec head grad norm] move_direction=0.034  exec_move=0.047  sprint=0.060  kick=0.028  kick_direction=0.008  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.052
2026-08-08 23:55:20,506 INFO   [exec continuous log_std] move_direction: start=-1.5937 end=-1.5931   kick_direction: start=-1.5877 end=-1.5871
2026-08-08 23:55:20,506 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0020≈0.12°/step  epoch≈7.0°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0019≈0.11°/step  epoch≈6.5°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-08 23:55:20,506 INFO   [exec discrete Δlogit per opt step] exec_move=0.0011  sprint=0.0011  kick=0.0012  tackle_attempt=0.0005
2026-08-08 23:55:20,507 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0009  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0013  sprint=+0.0013  kick=+0.0012  tackle_attempt=+0.0011  move_dir=+0.0291  kick_dir=+0.0122
2026-08-08 23:55:20,507 INFO   [grad clip] main: 2/60 steps clipped (3%)  pre-clip norm mean=0.277 max=0.512  limit=0.4
              direction: 54/60 steps clipped (90%)  pre-clip norm mean=0.036 max=0.153  limit=0.02
2026-08-08 23:55:20,555 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=2,376,000  speed=866/s  reward=3.73
  loss     policy=0.0034  value=0.5109(x0.5)=0.2554
           entropy=6.9216  kl=0.0471
  value    V=2.87±1.19  R=2.78±1.80  adv=-0.09±1.31
  moves    mv_ls=[-1.5931] (σ≈0.20, ≈12°) g=7.85e-03  d_move=[+0.0006] (Δσ≈0.007°)
           kk_ls=[-1.5871] (σ≈0.20, ≈12°)  d_kick=[+0.0006] (Δσ≈0.007°)
  heads    move= 50 get_poss= 38 exec_move= 80 sprint= 48 kick= 24 tackle= 49 shoot= 49 hold= 49 tackle_prob=0.4889 kick_prob=0.2382
  vs       vs[win/loss/tout/miss]  vs_immobile(558): 63.1%/0.0%/3.2%/23.1%/11%
  ep_len   19.3±11.9s  (n=558, min=0.9s, max=50.0s)
  reward   get_possession=+454.00  lose_possession=-2.70  ball_out=-308.00  box_possession=+880.00
           speed_bonus=+693.66  timeout=-27.00  stamina_penalty=-3.18
  rew/ep   (mean/std/min/max per episode, 558 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.814    0.403    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.005    0.066    -0.900    +0.000
  ball_out          -0.552    1.380    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.577    1.206    +0.000    +2.500
  speed_bonus       +1.243    1.282    +0.000    +4.111
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.048    0.265    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.024    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     456    +0.013    0.112     +3.394     2.180     +0.551      4.2054      1.659     4.491
  lose_possession       3    -0.000    0.008     +3.190     0.633     -0.131      0.4117      0.636     0.739
  ball_out            77    -0.009    0.185     -3.987     0.113     -3.600     14.3187      3.600     5.173
  box_possession     352    +0.024    0.246     +4.465     1.077     +0.629      1.2796      0.927     2.178
  speed_bonus        338    +0.019    0.221     +4.547     1.020     +0.674      1.3112      0.938     2.208
  timeout             18    -0.001    0.034     -1.512     0.006     -5.026     25.8350      5.026     5.951
  stamina_penalty     363    -0.000    0.001     +4.210     1.640     +0.374      2.4528      1.119     2.901
  gae/td   mean_return=+2.781  std_return=1.804  mean_gae=-0.090  mean_sq_td=1.7229
──────────────────────────────────────────────────────────────────────
2026-08-08 23:55:20,579 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint64.pt
2026-08-08 23:55:20,579 INFO Logging to checkpoints/phase1_run45/training_log65.txt
2026-08-08 23:55:20,580 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 23:55:34,356 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 23:55:34,357 INFO   [eval vs immobile] step=2,376,000  seeds=16x8  win=52%  mean_rew=2.510±3.276  V=2.404  gap=-0.107  outcomes={'other': 23, 'box_possession': 67, 'miss': 34, 'timeout': 4}
2026-08-08 23:55:34,358 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-08 23:55:44,167 INFO   [seeded eval] all workers finished, merging results.
2026-08-08 23:55:44,169 INFO   [eval vs rules] step=2,376,000  seeds=16x8  win=2%  mean_rew=-2.036±1.899  V=0.860  gap=+2.896  outcomes={'opponent_box_possession': 95, 'other': 10, 'box_possession': 3, 'miss': 20}
2026-08-09 00:00:27,068 INFO   [advantage] mean=0.000  std=1.000  min=-5.670  max=4.906
2026-08-09 00:00:27,069 INFO   [ratio] mean=0.9684  std=0.1392  min=0.0029  max=4.3197  clipped=17.6%
2026-08-09 00:00:27,070 INFO   [exec head grad norm] move_direction=0.029  exec_move=0.046  sprint=0.054  kick=0.034  kick_direction=0.008  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.057
2026-08-09 00:00:27,070 INFO   [exec continuous log_std] move_direction: start=-1.5931 end=-1.5925   kick_direction: start=-1.5871 end=-1.5865
2026-08-09 00:00:27,070 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0019≈0.11°/step  epoch≈6.6°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0017≈0.10°/step  epoch≈5.7°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 00:00:27,070 INFO   [exec discrete Δlogit per opt step] exec_move=0.0008  sprint=0.0011  kick=0.0010  tackle_attempt=0.0005
2026-08-09 00:00:27,070 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0008  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0014  sprint=+0.0013  kick=+0.0016  tackle_attempt=+0.0011  move_dir=+0.0286  kick_dir=+0.0120
2026-08-09 00:00:27,071 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.256 max=0.374  limit=0.4
              direction: 56/60 steps clipped (93%)  pre-clip norm mean=0.032 max=0.059  limit=0.02
2026-08-09 00:00:27,119 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=2,412,000  speed=873/s  reward=3.51
  loss     policy=0.0035  value=0.5300(x0.5)=0.2650
           entropy=6.9189  kl=0.0468
  value    V=2.80±1.21  R=2.74±1.70  adv=-0.06±1.24
  moves    mv_ls=[-1.5925] (σ≈0.20, ≈12°) g=7.57e-03  d_move=[+0.0006] (Δσ≈0.007°)
           kk_ls=[-1.5865] (σ≈0.20, ≈12°)  d_kick=[+0.0006] (Δσ≈0.007°)
  heads    move= 49 get_poss= 39 exec_move= 79 sprint= 48 kick= 24 tackle= 49 shoot= 49 hold= 49 tackle_prob=0.4873 kick_prob=0.2340
  vs       vs[win/loss/tout/miss]  vs_immobile(518): 64.9%/0.0%/4.8%/19.7%/11%
  ep_len   20.6±12.7s  (n=518, min=1.2s, max=50.0s)
  reward   get_possession=+429.00  lose_possession=-4.50  ball_out=-216.00  box_possession=+840.00
           speed_bonus=+642.61  timeout=-37.50  stamina_penalty=-3.16
  rew/ep   (mean/std/min/max per episode, 518 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.828    0.407    +0.000    +3.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.009    0.104    -1.800    +0.000
  ball_out          -0.417    1.222    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.622    1.193    +0.000    +2.500
  speed_bonus       +1.241    1.276    +0.000    +4.226
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.072    0.321    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.021    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     431    +0.012    0.109     +3.506     1.890     +0.655      3.4653      1.510     4.162
  lose_possession       5    -0.000    0.011     +1.531     1.325     -1.493      4.7124      1.515     3.855
  ball_out            54    -0.006    0.155     -3.963     0.189     -3.515     14.3278      3.515     5.946
  box_possession     336    +0.023    0.240     +4.404     1.106     +1.048      2.3176      1.209     2.883
  speed_bonus        325    +0.018    0.213     +4.469     1.066     +1.070      2.3768      1.230     2.907
  timeout             25    -0.001    0.040     -1.510     0.005     -4.576     22.1199      4.576     5.830
  stamina_penalty     355    -0.000    0.001     +4.033     1.808     +0.677      3.6762      1.437     4.507
  gae/td   mean_return=+2.739  std_return=1.697  mean_gae=-0.059  mean_sq_td=1.5325
──────────────────────────────────────────────────────────────────────
2026-08-09 00:00:27,145 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint65.pt
2026-08-09 00:00:27,146 INFO Logging to checkpoints/phase1_run45/training_log66.txt
2026-08-09 00:00:27,147 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 00:00:41,089 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 00:00:41,091 INFO   [eval vs immobile] step=2,412,000  seeds=16x8  win=52%  mean_rew=2.551±3.065  V=2.528  gap=-0.023  outcomes={'other': 22, 'box_possession': 67, 'timeout': 5, 'miss': 34}
2026-08-09 00:00:41,092 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 00:00:51,070 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 00:00:51,072 INFO   [eval vs rules] step=2,412,000  seeds=16x8  win=1%  mean_rew=-2.243±1.545  V=0.987  gap=+3.231  outcomes={'other': 12, 'opponent_box_possession': 96, 'miss': 19, 'box_possession': 1}
2026-08-09 00:05:35,624 INFO   [advantage] mean=0.000  std=1.000  min=-6.071  max=3.878
2026-08-09 00:05:35,624 INFO   [ratio] mean=0.9673  std=0.1430  min=0.0034  max=12.0681  clipped=17.9%
2026-08-09 00:05:35,625 INFO   [exec head grad norm] move_direction=0.040  exec_move=0.044  sprint=0.052  kick=0.034  kick_direction=0.008  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.058
2026-08-09 00:05:35,625 INFO   [exec continuous log_std] move_direction: start=-1.5925 end=-1.5917   kick_direction: start=-1.5865 end=-1.5859
2026-08-09 00:05:35,625 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0022≈0.13°/step  epoch≈7.5°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0021≈0.12°/step  epoch≈7.3°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 00:05:35,625 INFO   [exec discrete Δlogit per opt step] exec_move=0.0009  sprint=0.0009  kick=0.0013  tackle_attempt=0.0005
2026-08-09 00:05:35,625 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0008  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0015  sprint=+0.0017  kick=+0.0012  tackle_attempt=+0.0009  move_dir=+0.0302  kick_dir=+0.0118
2026-08-09 00:05:35,626 INFO   [grad clip] main: 4/60 steps clipped (7%)  pre-clip norm mean=0.295 max=1.563  limit=0.4
              direction: 55/60 steps clipped (92%)  pre-clip norm mean=0.043 max=0.413  limit=0.02
2026-08-09 00:05:35,683 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=2,448,000  speed=873/s  reward=2.88
  loss     policy=0.0044  value=0.5470(x0.5)=0.2735
           entropy=6.9177  kl=0.0480
  value    V=2.80±1.09  R=2.79±1.59  adv=-0.01±1.20
  moves    mv_ls=[-1.5917] (σ≈0.20, ≈12°) g=8.94e-03  d_move=[+0.0007] (Δσ≈0.008°)
           kk_ls=[-1.5859] (σ≈0.20, ≈12°)  d_kick=[+0.0006] (Δσ≈0.007°)
  heads    move= 50 get_poss= 38 exec_move= 79 sprint= 48 kick= 23 tackle= 49 shoot= 49 hold= 49 tackle_prob=0.4909 kick_prob=0.2351
  vs       vs[win/loss/tout/miss]  vs_immobile(524): 67.6%/0.0%/5.7%/17.7%/9%
  ep_len   20.5±12.9s  (n=524, min=1.2s, max=50.0s)
  reward   get_possession=+446.00  lose_possession=-4.50  ball_out=-200.00  box_possession=+885.00
           speed_bonus=+643.63  timeout=-45.00  stamina_penalty=-3.43
  rew/ep   (mean/std/min/max per episode, 524 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.851    0.382    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.009    0.087    -0.900    +0.000
  ball_out          -0.382    1.175    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.689    1.170    +0.000    +2.500
  speed_bonus       +1.228    1.222    +0.000    +3.990
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.086    0.348    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.007    0.006    -0.022    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     450    +0.013    0.111     +3.373     2.011     +0.563      3.9453      1.575     4.247
  lose_possession       5    -0.000    0.011     +2.723     0.576     +0.183      1.5955      0.869     2.284
  ball_out            50    -0.006    0.149     -3.960     0.196     -3.668     15.3626      3.668     5.584
  box_possession     354    +0.025    0.247     +4.312     1.062     +1.075      2.3598      1.250     2.949
  speed_bonus        341    +0.018    0.208     +4.382     1.020     +1.101      2.4095      1.269     2.957
  timeout             30    -0.001    0.043     -1.511     0.005     -4.390     20.5442      4.390     5.979
  stamina_penalty     379    -0.000    0.001     +3.868     1.876     +0.655      3.8228      1.506     4.450
  gae/td   mean_return=+2.794  std_return=1.586  mean_gae=-0.010  mean_sq_td=1.4308
──────────────────────────────────────────────────────────────────────
2026-08-09 00:05:35,709 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint66.pt
2026-08-09 00:05:35,709 INFO Logging to checkpoints/phase1_run45/training_log67.txt
2026-08-09 00:05:35,710 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 00:05:49,144 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 00:05:49,145 INFO   [eval vs immobile] step=2,448,000  seeds=16x8  win=52%  mean_rew=2.579±3.135  V=2.498  gap=-0.081  outcomes={'other': 22, 'timeout': 4, 'box_possession': 67, 'miss': 35}
2026-08-09 00:05:49,146 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 00:05:59,669 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 00:05:59,670 INFO   [eval vs rules] step=2,448,000  seeds=16x8  win=4%  mean_rew=-1.966±2.083  V=0.974  gap=+2.939  outcomes={'opponent_box_possession': 92, 'other': 10, 'box_possession': 5, 'miss': 21}
2026-08-09 00:10:44,132 INFO   [advantage] mean=0.000  std=1.000  min=-5.553  max=3.191
2026-08-09 00:10:44,133 INFO   [ratio] mean=0.9692  std=0.1417  min=0.0018  max=6.8864  clipped=17.7%
2026-08-09 00:10:44,133 INFO   [exec head grad norm] move_direction=0.029  exec_move=0.038  sprint=0.048  kick=0.033  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.052
2026-08-09 00:10:44,133 INFO   [exec continuous log_std] move_direction: start=-1.5917 end=-1.5909   kick_direction: start=-1.5859 end=-1.5852
2026-08-09 00:10:44,133 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0019≈0.11°/step  epoch≈6.5°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0018≈0.10°/step  epoch≈6.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 00:10:44,133 INFO   [exec discrete Δlogit per opt step] exec_move=0.0008  sprint=0.0009  kick=0.0016  tackle_attempt=0.0005
2026-08-09 00:10:44,133 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0009  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0014  sprint=+0.0011  kick=+0.0019  tackle_attempt=+0.0012  move_dir=+0.0273  kick_dir=+0.0122
2026-08-09 00:10:44,134 INFO   [grad clip] main: 1/60 steps clipped (2%)  pre-clip norm mean=0.251 max=0.457  limit=0.4
              direction: 53/60 steps clipped (88%)  pre-clip norm mean=0.033 max=0.068  limit=0.02
2026-08-09 00:10:44,184 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=2,484,000  speed=872/s  reward=2.08
  loss     policy=0.0053  value=0.5018(x0.5)=0.2509
           entropy=6.9200  kl=0.0461
  value    V=2.73±1.11  R=2.72±1.66  adv=-0.01±1.17
  moves    mv_ls=[-1.5909] (σ≈0.20, ≈12°) g=9.70e-03  d_move=[+0.0008] (Δσ≈0.009°)
           kk_ls=[-1.5852] (σ≈0.20, ≈12°)  d_kick=[+0.0007] (Δσ≈0.009°)
  heads    move= 50 get_poss= 38 exec_move= 79 sprint= 49 kick= 24 tackle= 49 shoot= 48 hold= 48 tackle_prob=0.4905 kick_prob=0.2347
  vs       vs[win/loss/tout/miss]  vs_immobile(545): 63.7%/0.2%/3.7%/22.6%/10%
  ep_len   19.7±12.7s  (n=545, min=0.2s, max=50.0s)
  reward   get_possession=+444.00  lose_possession=-6.30  ball_out=-220.00  box_possession=+867.50
           speed_bonus=+653.70  opponent_box=-3.00  timeout=-30.00  stamina_penalty=-3.36
  rew/ep   (mean/std/min/max per episode, 545 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.815    0.420    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.012    0.101    -0.900    +0.000
  ball_out          -0.404    1.205    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.592    1.202    +0.000    +2.500
  speed_bonus       +1.199    1.269    +0.000    +4.172
  opponent_box      -0.006    0.128    -3.000    +0.000
  timeout           -0.055    0.282    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.026    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     447    +0.012    0.111     +3.340     2.061     +0.556      3.8048      1.570     4.537
  lose_possession       7    -0.000    0.013     +1.965     0.831     -1.007      1.8875      1.028     2.361
  ball_out            55    -0.006    0.156     -3.964     0.187     -3.484     13.7508      3.484     6.240
  box_possession     347    +0.024    0.244     +4.375     1.112     +0.715      1.4796      0.992     2.246
  speed_bonus        333    +0.018    0.214     +4.454     1.065     +0.762      1.5236      1.009     2.276
  opponent_box         1    -0.000    0.016     -3.001     0.000     -5.552     30.8299      5.552     5.552
  timeout             20    -0.001    0.035     -1.513     0.005     -4.881     24.0698      4.881     5.597
  stamina_penalty     365    -0.000    0.001     +4.041     1.762     +0.399      2.8094      1.224     4.226
  gae/td   mean_return=+2.720  std_return=1.660  mean_gae=-0.015  mean_sq_td=1.3709
──────────────────────────────────────────────────────────────────────
2026-08-09 00:10:44,209 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint67.pt
2026-08-09 00:10:44,210 INFO Logging to checkpoints/phase1_run45/training_log68.txt
2026-08-09 00:10:44,211 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 00:10:57,313 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 00:10:57,314 INFO   [eval vs immobile] step=2,484,000  seeds=16x8  win=54%  mean_rew=2.801±3.007  V=2.513  gap=-0.288  outcomes={'other': 26, 'box_possession': 69, 'miss': 29, 'timeout': 4}
2026-08-09 00:10:57,315 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 00:11:07,058 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 00:11:07,059 INFO   [eval vs rules] step=2,484,000  seeds=16x8  win=3%  mean_rew=-1.895±2.142  V=0.967  gap=+2.863  outcomes={'opponent_box_possession': 87, 'other': 16, 'box_possession': 4, 'miss': 21}
2026-08-09 00:15:54,775 INFO   [advantage] mean=-0.000  std=1.000  min=-5.403  max=4.568
2026-08-09 00:15:54,776 INFO   [ratio] mean=0.9713  std=0.1398  min=0.0086  max=4.5422  clipped=16.7%
2026-08-09 00:15:54,776 INFO   [exec head grad norm] move_direction=0.025  exec_move=0.042  sprint=0.048  kick=0.030  kick_direction=0.008  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.054
2026-08-09 00:15:54,776 INFO   [exec continuous log_std] move_direction: start=-1.5909 end=-1.5901   kick_direction: start=-1.5852 end=-1.5845
2026-08-09 00:15:54,777 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0015≈0.08°/step  epoch≈5.1°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0015≈0.08°/step  epoch≈5.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 00:15:54,777 INFO   [exec discrete Δlogit per opt step] exec_move=0.0008  sprint=0.0008  kick=0.0010  tackle_attempt=0.0005
2026-08-09 00:15:54,777 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0007  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0013  sprint=+0.0012  kick=+0.0013  tackle_attempt=+0.0009  move_dir=+0.0271  kick_dir=+0.0112
2026-08-09 00:15:54,777 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.252 max=0.393  limit=0.4
              direction: 50/60 steps clipped (83%)  pre-clip norm mean=0.028 max=0.097  limit=0.02
2026-08-09 00:15:54,837 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=2,520,000  speed=875/s  reward=4.15
  loss     policy=0.0046  value=0.4845(x0.5)=0.2423
           entropy=6.9154  kl=0.0437
  value    V=2.73±1.14  R=2.72±1.75  adv=-0.01±1.23
  moves    mv_ls=[-1.5901] (σ≈0.20, ≈12°) g=8.17e-03  d_move=[+0.0008] (Δσ≈0.009°)
           kk_ls=[-1.5845] (σ≈0.21, ≈12°)  d_kick=[+0.0007] (Δσ≈0.008°)
  heads    move= 50 get_poss= 39 exec_move= 80 sprint= 49 kick= 23 tackle= 49 shoot= 49 hold= 48 tackle_prob=0.4901 kick_prob=0.2341
  vs       vs[win/loss/tout/miss]  vs_immobile(559): 64.0%/0.0%/3.0%/21.6%/11%
  ep_len   19.2±11.9s  (n=559, min=0.9s, max=50.0s)
  reward   get_possession=+453.00  lose_possession=-4.50  ball_out=-272.00  box_possession=+895.00
           speed_bonus=+685.38  timeout=-25.50  stamina_penalty=-3.47
  rew/ep   (mean/std/min/max per episode, 559 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.810    0.414    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.008    0.085    -0.900    +0.000
  ball_out          -0.487    1.308    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.601    1.200    +0.000    +2.500
  speed_bonus       +1.226    1.301    +0.000    +4.053
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.046    0.258    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.027    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     457    +0.013    0.112     +3.305     2.182     +0.558      4.3137      1.699     4.469
  lose_possession       5    -0.000    0.011     +2.918     0.521     -0.161      0.6690      0.700     1.246
  ball_out            68    -0.008    0.174     -3.956     0.205     -3.780     16.1753      3.780     5.841
  box_possession     358    +0.025    0.248     +4.411     1.144     +0.652      1.4534      0.997     2.188
  speed_bonus        337    +0.019    0.222     +4.524     1.079     +0.716      1.4849      1.009     2.189
  timeout             17    -0.001    0.033     -1.512     0.003     -4.819     24.3481      4.819     6.033
  stamina_penalty     369    -0.000    0.001     +4.153     1.674     +0.408      2.5184      1.176     2.732
  gae/td   mean_return=+2.722  std_return=1.748  mean_gae=-0.012  mean_sq_td=1.5162
──────────────────────────────────────────────────────────────────────
2026-08-09 00:15:54,866 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint68.pt
2026-08-09 00:15:54,867 INFO Logging to checkpoints/phase1_run45/training_log69.txt
2026-08-09 00:15:54,868 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 00:16:08,385 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 00:16:08,386 INFO   [eval vs immobile] step=2,520,000  seeds=16x8  win=55%  mean_rew=2.740±3.071  V=2.419  gap=-0.320  outcomes={'other': 21, 'box_possession': 70, 'timeout': 4, 'miss': 33}
2026-08-09 00:16:08,387 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 00:16:18,385 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 00:16:18,386 INFO   [eval vs rules] step=2,520,000  seeds=16x8  win=1%  mean_rew=-2.127±1.546  V=0.765  gap=+2.892  outcomes={'other': 14, 'opponent_box_possession': 95, 'miss': 18, 'box_possession': 1}
2026-08-09 00:21:04,428 INFO   [advantage] mean=0.000  std=1.000  min=-6.297  max=4.482
2026-08-09 00:21:04,429 INFO   [ratio] mean=0.9711  std=0.1318  min=0.0009  max=3.2868  clipped=16.1%
2026-08-09 00:21:04,429 INFO   [exec head grad norm] move_direction=0.025  exec_move=0.042  sprint=0.047  kick=0.033  kick_direction=0.010  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.057
2026-08-09 00:21:04,430 INFO   [exec continuous log_std] move_direction: start=-1.5901 end=-1.5894   kick_direction: start=-1.5845 end=-1.5837
2026-08-09 00:21:04,430 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0016≈0.09°/step  epoch≈5.4°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0020≈0.11°/step  epoch≈6.8°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 00:21:04,430 INFO   [exec discrete Δlogit per opt step] exec_move=0.0010  sprint=0.0008  kick=0.0012  tackle_attempt=0.0008
2026-08-09 00:21:04,430 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0009  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0009  sprint=+0.0014  kick=+0.0012  tackle_attempt=+0.0010  move_dir=+0.0262  kick_dir=+0.0111
2026-08-09 00:21:04,430 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.250 max=0.384  limit=0.4
              direction: 51/60 steps clipped (85%)  pre-clip norm mean=0.029 max=0.056  limit=0.02
2026-08-09 00:21:04,479 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=2,556,000  speed=871/s  reward=3.14
  loss     policy=0.0034  value=0.4995(x0.5)=0.2497
           entropy=6.9145  kl=0.0427
  value    V=2.79±1.16  R=2.86±1.65  adv=0.07±1.16
  moves    mv_ls=[-1.5894] (σ≈0.20, ≈12°) g=7.76e-03  d_move=[+0.0007] (Δσ≈0.009°)
           kk_ls=[-1.5837] (σ≈0.21, ≈12°)  d_kick=[+0.0008] (Δσ≈0.009°)
  heads    move= 51 get_poss= 38 exec_move= 79 sprint= 49 kick= 23 tackle= 49 shoot= 49 hold= 49 tackle_prob=0.4942 kick_prob=0.2346
  vs       vs[win/loss/tout/miss]  vs_immobile(538): 68.8%/0.2%/2.8%/17.8%/10%
  ep_len   20.0±11.8s  (n=538, min=0.8s, max=50.0s)
  reward   get_possession=+449.00  lose_possession=-2.70  ball_out=-216.00  box_possession=+925.00
           speed_bonus=+701.55  opponent_box=-3.00  timeout=-22.50  stamina_penalty=-3.48
  rew/ep   (mean/std/min/max per episode, 538 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.835    0.386    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.005    0.067    -0.900    +0.000
  ball_out          -0.401    1.202    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.719    1.159    +0.000    +2.500
  speed_bonus       +1.304    1.270    +0.000    +4.111
  opponent_box      -0.006    0.129    -3.000    +0.000
  timeout           -0.042    0.247    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.024    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     452    +0.013    0.111     +3.458     2.013     +0.632      3.7323      1.574     4.272
  lose_possession       5    -0.000    0.011     +2.393     0.292     -0.561      0.3282      0.561     0.720
  ball_out            54    -0.006    0.155     -4.000     0.000     -3.673     15.3530      3.673     6.479
  box_possession     370    +0.026    0.252     +4.390     1.106     +0.801      1.7000      1.051     2.555
  speed_bonus        352    +0.019    0.222     +4.487     1.045     +0.844      1.7560      1.077     2.616
  opponent_box         1    -0.000    0.016     -3.006     0.000     -5.656     31.9896      5.656     5.656
  timeout             15    -0.001    0.031     -1.509     0.005     -4.277     19.4974      4.277     5.965
  stamina_penalty     383    -0.000    0.001     +4.161     1.592     +0.600      2.4356      1.183     3.128
  gae/td   mean_return=+2.860  std_return=1.649  mean_gae=+0.068  mean_sq_td=1.3562
──────────────────────────────────────────────────────────────────────
2026-08-09 00:21:04,504 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint69.pt
2026-08-09 00:21:04,505 INFO Logging to checkpoints/phase1_run45/training_log70.txt
2026-08-09 00:21:04,506 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 00:21:18,005 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 00:21:18,007 INFO   [eval vs immobile] step=2,556,000  seeds=16x8  win=55%  mean_rew=2.716±3.144  V=2.493  gap=-0.224  outcomes={'other': 20, 'box_possession': 71, 'miss': 32, 'timeout': 5}
2026-08-09 00:21:18,015 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 00:21:28,899 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 00:21:28,900 INFO   [eval vs rules] step=2,556,000  seeds=16x8  win=1%  mean_rew=-2.318±1.494  V=0.763  gap=+3.082  outcomes={'opponent_box_possession': 101, 'other': 12, 'miss': 14, 'box_possession': 1}
2026-08-09 00:26:15,431 INFO   [advantage] mean=0.000  std=1.000  min=-6.255  max=4.877
2026-08-09 00:26:15,432 INFO   [ratio] mean=0.9711  std=0.1363  min=0.0028  max=3.2257  clipped=16.5%
2026-08-09 00:26:15,432 INFO   [exec head grad norm] move_direction=0.021  exec_move=0.040  sprint=0.049  kick=0.034  kick_direction=0.008  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.051
2026-08-09 00:26:15,432 INFO   [exec continuous log_std] move_direction: start=-1.5894 end=-1.5885   kick_direction: start=-1.5837 end=-1.5830
2026-08-09 00:26:15,433 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0018≈0.10°/step  epoch≈6.2°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0019≈0.11°/step  epoch≈6.4°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 00:26:15,433 INFO   [exec discrete Δlogit per opt step] exec_move=0.0007  sprint=0.0012  kick=0.0011  tackle_attempt=0.0006
2026-08-09 00:26:15,433 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0007  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0010  sprint=+0.0015  kick=+0.0014  tackle_attempt=+0.0009  move_dir=+0.0275  kick_dir=+0.0107
2026-08-09 00:26:15,433 INFO   [grad clip] main: 1/60 steps clipped (2%)  pre-clip norm mean=0.238 max=0.433  limit=0.4
              direction: 44/60 steps clipped (73%)  pre-clip norm mean=0.025 max=0.055  limit=0.02
2026-08-09 00:26:15,498 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=2,592,000  speed=870/s  reward=3.35
  loss     policy=0.0046  value=0.4632(x0.5)=0.2316
           entropy=6.9082  kl=0.0437
  value    V=2.76±1.20  R=2.93±1.60  adv=0.18±1.08
  moves    mv_ls=[-1.5885] (σ≈0.20, ≈12°) g=8.66e-03  d_move=[+0.0009] (Δσ≈0.010°)
           kk_ls=[-1.5830] (σ≈0.21, ≈12°)  d_kick=[+0.0007] (Δσ≈0.009°)
  heads    move= 50 get_poss= 39 exec_move= 80 sprint= 49 kick= 23 tackle= 50 shoot= 49 hold= 49 tackle_prob=0.4917 kick_prob=0.2314
  vs       vs[win/loss/tout/miss]  vs_immobile(553): 72.2%/0.2%/1.1%/16.6%/10%
  ep_len   19.4±11.4s  (n=553, min=1.2s, max=50.0s)
  reward   get_possession=+465.00  lose_possession=-7.20  ball_out=-160.00  box_possession=+997.50
           speed_bonus=+743.69  opponent_box=-3.00  timeout=-9.00  stamina_penalty=-3.69
  rew/ep   (mean/std/min/max per episode, 553 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.841    0.403    +0.000    +3.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.013    0.120    -1.800    +0.000
  ball_out          -0.289    1.036    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.804    1.121    +0.000    +2.500
  speed_bonus       +1.345    1.256    +0.000    +4.106
  opponent_box      -0.005    0.127    -3.000    +0.000
  timeout           -0.016    0.155    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.007    0.006    -0.025    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     468    +0.013    0.113     +3.660     1.774     +0.862      3.5209      1.557     3.660
  lose_possession       8    -0.000    0.013     +2.274     1.557     -0.507      3.0414      1.007     3.384
  ball_out            40    -0.004    0.133     -4.000     0.000     -3.457     13.3367      3.457     5.375
  box_possession     399    +0.028    0.262     +4.357     1.101     +0.802      1.8015      1.073     2.643
  speed_bonus        383    +0.021    0.227     +4.433     1.058     +0.842      1.8470      1.092     2.694
  opponent_box         1    -0.000    0.016     -3.009     0.000     -5.977     35.7189      5.977     5.977
  timeout              6    -0.000    0.019     -1.512     0.006     -4.448     20.3145      4.448     5.412
  stamina_penalty     400    -0.000    0.001     +4.269     1.352     +0.718      2.1840      1.143     2.981
  gae/td   mean_return=+2.933  std_return=1.596  mean_gae=+0.176  mean_sq_td=1.2035
──────────────────────────────────────────────────────────────────────
2026-08-09 00:26:15,525 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint70.pt
2026-08-09 00:26:15,525 INFO Logging to checkpoints/phase1_run45/training_log71.txt
2026-08-09 00:26:15,526 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 00:26:28,656 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 00:26:28,658 INFO   [eval vs immobile] step=2,592,000  seeds=16x8  win=57%  mean_rew=2.795±3.123  V=2.699  gap=-0.096  outcomes={'other': 20, 'box_possession': 73, 'timeout': 3, 'miss': 32}
2026-08-09 00:26:28,659 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 00:26:39,107 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 00:26:39,109 INFO   [eval vs rules] step=2,592,000  seeds=16x8  win=3%  mean_rew=-2.029±1.957  V=0.882  gap=+2.910  outcomes={'opponent_box_possession': 90, 'other': 13, 'miss': 21, 'box_possession': 4}
2026-08-09 00:31:29,375 INFO   [advantage] mean=-0.000  std=1.000  min=-5.675  max=4.556
2026-08-09 00:31:29,376 INFO   [ratio] mean=0.9699  std=0.1414  min=0.0096  max=9.7992  clipped=16.2%
2026-08-09 00:31:29,376 INFO   [exec head grad norm] move_direction=0.029  exec_move=0.039  sprint=0.042  kick=0.027  kick_direction=0.008  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.051
2026-08-09 00:31:29,376 INFO   [exec continuous log_std] move_direction: start=-1.5885 end=-1.5877   kick_direction: start=-1.5830 end=-1.5823
2026-08-09 00:31:29,377 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0018≈0.11°/step  epoch≈6.3°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0017≈0.10°/step  epoch≈6.0°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 00:31:29,377 INFO   [exec discrete Δlogit per opt step] exec_move=0.0008  sprint=0.0009  kick=0.0011  tackle_attempt=0.0005
2026-08-09 00:31:29,377 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0007  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0010  sprint=+0.0016  kick=+0.0017  tackle_attempt=+0.0008  move_dir=+0.0276  kick_dir=+0.0107
2026-08-09 00:31:29,377 INFO   [grad clip] main: 1/60 steps clipped (2%)  pre-clip norm mean=0.237 max=0.434  limit=0.4
              direction: 53/60 steps clipped (88%)  pre-clip norm mean=0.032 max=0.085  limit=0.02
2026-08-09 00:31:29,451 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=2,628,000  speed=870/s  reward=4.35
  loss     policy=0.0044  value=0.4756(x0.5)=0.2378
           entropy=6.9174  kl=0.0440
  value    V=2.93±1.13  R=2.96±1.64  adv=0.02±1.13
  moves    mv_ls=[-1.5877] (σ≈0.20, ≈12°) g=8.69e-03  d_move=[+0.0008] (Δσ≈0.009°)
           kk_ls=[-1.5823] (σ≈0.21, ≈12°)  d_kick=[+0.0006] (Δσ≈0.008°)
  heads    move= 50 get_poss= 38 exec_move= 80 sprint= 50 kick= 24 tackle= 49 shoot= 49 hold= 49 tackle_prob=0.4955 kick_prob=0.2373
  vs       vs[win/loss/tout/miss]  vs_immobile(554): 69.0%/0.0%/2.7%/20.0%/8%
  ep_len   19.3±11.7s  (n=554, min=1.1s, max=50.0s)
  reward   get_possession=+463.00  lose_possession=-7.20  ball_out=-200.00  box_possession=+955.00
           speed_bonus=+725.70  timeout=-22.50  stamina_penalty=-3.80
  rew/ep   (mean/std/min/max per episode, 554 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.836    0.408    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.013    0.107    -0.900    +0.000
  ball_out          -0.361    1.146    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.724    1.157    +0.000    +2.500
  speed_bonus       +1.310    1.278    +0.000    +4.116
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.041    0.243    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.007    0.006    -0.029    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     466    +0.013    0.113     +3.599     2.001     +0.557      3.6256      1.521     4.612
  lose_possession       8    -0.000    0.013     +2.704     0.769     -0.380      0.6781      0.650     1.551
  ball_out            50    -0.006    0.149     -3.940     0.237     -3.947     17.3771      3.947     5.789
  box_possession     382    +0.027    0.256     +4.393     1.114     +0.594      1.3282      0.949     2.043
  speed_bonus        363    +0.020    0.226     +4.492     1.052     +0.640      1.3751      0.970     2.068
  timeout             15    -0.001    0.031     -1.516     0.005     -4.962     25.1345      4.962     5.922
  stamina_penalty     391    -0.000    0.001     +4.187     1.574     +0.386      2.2503      1.106     2.744
  gae/td   mean_return=+2.956  std_return=1.639  mean_gae=+0.024  mean_sq_td=1.2689
──────────────────────────────────────────────────────────────────────
2026-08-09 00:31:29,475 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint71.pt
2026-08-09 00:31:29,476 INFO Logging to checkpoints/phase1_run45/training_log72.txt
2026-08-09 00:31:29,477 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 00:31:42,468 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 00:31:42,470 INFO   [eval vs immobile] step=2,628,000  seeds=16x8  win=52%  mean_rew=2.573±3.376  V=2.629  gap=+0.056  outcomes={'other': 23, 'box_possession': 66, 'timeout': 6, 'miss': 33}
2026-08-09 00:31:42,472 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 00:31:52,064 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 00:31:52,065 INFO   [eval vs rules] step=2,628,000  seeds=16x8  win=0%  mean_rew=-2.308±1.332  V=0.793  gap=+3.102  outcomes={'other': 15, 'opponent_box_possession': 96, 'miss': 17}
2026-08-09 00:36:38,386 INFO   [advantage] mean=-0.000  std=1.000  min=-6.350  max=3.548
2026-08-09 00:36:38,387 INFO   [ratio] mean=0.9742  std=0.1257  min=0.0081  max=3.5164  clipped=14.9%
2026-08-09 00:36:38,387 INFO   [exec head grad norm] move_direction=0.027  exec_move=0.043  sprint=0.051  kick=0.031  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.056
2026-08-09 00:36:38,387 INFO   [exec continuous log_std] move_direction: start=-1.5877 end=-1.5870   kick_direction: start=-1.5823 end=-1.5816
2026-08-09 00:36:38,388 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0015≈0.09°/step  epoch≈5.2°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0017≈0.10°/step  epoch≈5.7°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 00:36:38,388 INFO   [exec discrete Δlogit per opt step] exec_move=0.0009  sprint=0.0009  kick=0.0011  tackle_attempt=0.0006
2026-08-09 00:36:38,388 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0008  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0009  sprint=+0.0011  kick=+0.0011  tackle_attempt=+0.0005  move_dir=+0.0236  kick_dir=+0.0096
2026-08-09 00:36:38,388 INFO   [grad clip] main: 1/60 steps clipped (2%)  pre-clip norm mean=0.264 max=0.428  limit=0.4
              direction: 47/60 steps clipped (78%)  pre-clip norm mean=0.030 max=0.067  limit=0.02
2026-08-09 00:36:38,438 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=2,664,000  speed=872/s  reward=4.70
  loss     policy=0.0042  value=0.4866(x0.5)=0.2433
           entropy=6.9138  kl=0.0376
  value    V=3.00±1.18  R=3.05±1.68  adv=0.04±1.17
  moves    mv_ls=[-1.5870] (σ≈0.20, ≈12°) g=7.63e-03  d_move=[+0.0007] (Δσ≈0.009°)
           kk_ls=[-1.5816] (σ≈0.21, ≈12°)  d_kick=[+0.0007] (Δσ≈0.009°)
  heads    move= 50 get_poss= 38 exec_move= 80 sprint= 50 kick= 23 tackle= 49 shoot= 49 hold= 50 tackle_prob=0.4950 kick_prob=0.2319
  vs       vs[win/loss/tout/miss]  vs_immobile(561): 70.2%/0.2%/2.3%/18.7%/9%
  ep_len   19.2±11.3s  (n=561, min=1.8s, max=50.0s)
  reward   get_possession=+472.00  lose_possession=-0.90  ball_out=-216.00  box_possession=+985.00
           speed_bonus=+786.49  opponent_box=-3.00  timeout=-19.50  stamina_penalty=-3.72
  rew/ep   (mean/std/min/max per episode, 561 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.841    0.365    +0.000    +1.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.002    0.038    -0.900    +0.000
  ball_out          -0.385    1.180    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.756    1.143    +0.000    +2.500
  speed_bonus       +1.402    1.313    +0.000    +4.116
  opponent_box      -0.005    0.127    -3.000    +0.000
  timeout           -0.035    0.226    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.007    0.006    -0.025    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     474    +0.013    0.114     +3.630     2.101     +0.617      3.7103      1.549     4.380
  lose_possession       1    -0.000    0.005     +3.090     0.000     -0.073      0.0053      0.073     0.073
  ball_out            54    -0.006    0.155     -3.981     0.135     -3.410     13.8653      3.410     5.743
  box_possession     394    +0.027    0.260     +4.490     1.123     +0.587      1.3823      0.941     2.407
  speed_bonus        379    +0.022    0.239     +4.566     1.075     +0.619      1.4180      0.955     2.413
  opponent_box         1    -0.000    0.016     -3.003     0.000     -5.163     26.6571      5.163     5.163
  timeout             13    -0.001    0.028     -1.511     0.006     -4.765     23.5064      4.765     6.110
  stamina_penalty     402    -0.000    0.001     +4.311     1.544     +0.406      2.0989      1.063     2.796
  gae/td   mean_return=+3.045  std_return=1.676  mean_gae=+0.041  mean_sq_td=1.3621
──────────────────────────────────────────────────────────────────────
2026-08-09 00:36:38,464 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint72.pt
2026-08-09 00:36:38,465 INFO Logging to checkpoints/phase1_run45/training_log73.txt
2026-08-09 00:36:38,466 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 00:36:51,812 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 00:36:51,814 INFO   [eval vs immobile] step=2,664,000  seeds=16x8  win=55%  mean_rew=2.664±3.153  V=2.684  gap=+0.020  outcomes={'other': 22, 'box_possession': 71, 'miss': 34, 'timeout': 1}
2026-08-09 00:36:51,815 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 00:37:02,337 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 00:37:02,338 INFO   [eval vs rules] step=2,664,000  seeds=16x8  win=2%  mean_rew=-2.096±1.754  V=0.988  gap=+3.084  outcomes={'opponent_box_possession': 95, 'other': 15, 'box_possession': 2, 'miss': 16}
2026-08-09 00:41:51,697 INFO   [advantage] mean=0.000  std=1.000  min=-7.136  max=4.363
2026-08-09 00:41:51,698 INFO   [ratio] mean=0.9741  std=0.1342  min=0.0042  max=8.5872  clipped=14.8%
2026-08-09 00:41:51,698 INFO   [exec head grad norm] move_direction=0.029  exec_move=0.041  sprint=0.048  kick=0.032  kick_direction=0.008  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.065
2026-08-09 00:41:51,699 INFO   [exec continuous log_std] move_direction: start=-1.5870 end=-1.5863   kick_direction: start=-1.5816 end=-1.5810
2026-08-09 00:41:51,699 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0017≈0.10°/step  epoch≈5.8°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0017≈0.10°/step  epoch≈5.8°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 00:41:51,699 INFO   [exec discrete Δlogit per opt step] exec_move=0.0007  sprint=0.0009  kick=0.0013  tackle_attempt=0.0006
2026-08-09 00:41:51,699 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0006  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0011  sprint=+0.0016  kick=+0.0014  tackle_attempt=+0.0009  move_dir=+0.0232  kick_dir=+0.0108
2026-08-09 00:41:51,700 INFO   [grad clip] main: 1/60 steps clipped (2%)  pre-clip norm mean=0.264 max=0.483  limit=0.4
              direction: 50/60 steps clipped (83%)  pre-clip norm mean=0.031 max=0.066  limit=0.02
2026-08-09 00:41:51,754 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=2,700,000  speed=872/s  reward=3.91
  loss     policy=0.0038  value=0.4736(x0.5)=0.2368
           entropy=6.9153  kl=0.0396
  value    V=3.00±1.18  R=3.00±1.65  adv=-0.00±1.13
  moves    mv_ls=[-1.5863] (σ≈0.20, ≈12°) g=6.74e-03  d_move=[+0.0006] (Δσ≈0.008°)
           kk_ls=[-1.5810] (σ≈0.21, ≈12°)  d_kick=[+0.0006] (Δσ≈0.007°)
  heads    move= 50 get_poss= 40 exec_move= 80 sprint= 49 kick= 23 tackle= 50 shoot= 49 hold= 49 tackle_prob=0.4934 kick_prob=0.2338
  vs       vs[win/loss/tout/miss]  vs_immobile(539): 69.8%/0.0%/2.4%/18.2%/10%
  ep_len   19.9±11.7s  (n=539, min=1.1s, max=50.0s)
  reward   get_possession=+458.00  lose_possession=-9.00  ball_out=-184.00  box_possession=+940.00
           speed_bonus=+727.85  timeout=-19.50  stamina_penalty=-3.53
  rew/ep   (mean/std/min/max per episode, 539 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.850    0.401    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.017    0.121    -0.900    +0.000
  ball_out          -0.341    1.118    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.744    1.148    +0.000    +2.500
  speed_bonus       +1.350    1.295    +0.000    +4.116
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.036    0.230    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.007    0.006    -0.022    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     461    +0.013    0.112     +3.708     1.876     +0.657      3.3590      1.486     3.690
  lose_possession      10    -0.000    0.015     +1.630     1.768     -1.294      5.2897      1.687     4.609
  ball_out            46    -0.005    0.143     -3.957     0.204     -3.591     14.8649      3.591     5.954
  box_possession     376    +0.026    0.254     +4.429     1.124     +0.606      1.3768      0.942     2.224
  speed_bonus        361    +0.020    0.228     +4.510     1.074     +0.648      1.4102      0.955     2.306
  timeout             13    -0.001    0.028     -1.512     0.004     -4.393     20.6419      4.393     5.967
  stamina_penalty     382    -0.000    0.001     +4.254     1.540     +0.448      2.0465      1.066     2.787
  gae/td   mean_return=+2.997  std_return=1.653  mean_gae=-0.003  mean_sq_td=1.2830
──────────────────────────────────────────────────────────────────────
2026-08-09 00:41:51,779 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint73.pt
2026-08-09 00:41:51,779 INFO Logging to checkpoints/phase1_run45/training_log74.txt
2026-08-09 00:41:51,781 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 00:42:06,061 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 00:42:06,063 INFO   [eval vs immobile] step=2,700,000  seeds=16x8  win=53%  mean_rew=2.601±3.271  V=2.723  gap=+0.122  outcomes={'other': 22, 'box_possession': 68, 'miss': 35, 'timeout': 3}
2026-08-09 00:42:06,064 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 00:42:16,369 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 00:42:16,371 INFO   [eval vs rules] step=2,700,000  seeds=16x8  win=4%  mean_rew=-2.051±1.991  V=0.934  gap=+2.984  outcomes={'other': 12, 'opponent_box_possession': 94, 'box_possession': 5, 'miss': 17}
2026-08-09 00:47:03,820 INFO   [advantage] mean=-0.000  std=1.000  min=-5.665  max=4.347
2026-08-09 00:47:03,822 INFO   [ratio] mean=0.9750  std=0.1310  min=0.0057  max=4.6922  clipped=14.7%
2026-08-09 00:47:03,822 INFO   [exec head grad norm] move_direction=0.030  exec_move=0.040  sprint=0.048  kick=0.036  kick_direction=0.010  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.055
2026-08-09 00:47:03,822 INFO   [exec continuous log_std] move_direction: start=-1.5863 end=-1.5856   kick_direction: start=-1.5810 end=-1.5803
2026-08-09 00:47:03,822 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0015≈0.08°/step  epoch≈5.0°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0014≈0.08°/step  epoch≈4.8°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 00:47:03,822 INFO   [exec discrete Δlogit per opt step] exec_move=0.0008  sprint=0.0008  kick=0.0012  tackle_attempt=0.0006
2026-08-09 00:47:03,822 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0008  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0012  sprint=+0.0012  kick=+0.0017  tackle_attempt=+0.0011  move_dir=+0.0222  kick_dir=+0.0092
2026-08-09 00:47:03,823 INFO   [grad clip] main: 4/60 steps clipped (7%)  pre-clip norm mean=0.268 max=0.590  limit=0.4
              direction: 53/60 steps clipped (88%)  pre-clip norm mean=0.034 max=0.109  limit=0.02
2026-08-09 00:47:03,880 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=2,736,000  speed=873/s  reward=3.34
  loss     policy=0.0045  value=0.4734(x0.5)=0.2367
           entropy=6.9127  kl=0.0374
  value    V=2.99±1.21  R=3.01±1.68  adv=0.02±1.16
  moves    mv_ls=[-1.5856] (σ≈0.20, ≈12°) g=7.89e-03  d_move=[+0.0007] (Δσ≈0.008°)
           kk_ls=[-1.5803] (σ≈0.21, ≈12°)  d_kick=[+0.0006] (Δσ≈0.008°)
  heads    move= 50 get_poss= 39 exec_move= 80 sprint= 50 kick= 23 tackle= 50 shoot= 49 hold= 49 tackle_prob=0.4901 kick_prob=0.2331
  vs       vs[win/loss/tout/miss]  vs_immobile(556): 70.9%/0.0%/1.3%/17.6%/10%
  ep_len   19.3±11.3s  (n=556, min=1.4s, max=50.0s)
  reward   get_possession=+464.00  lose_possession=-2.70  ball_out=-200.00  box_possession=+985.00
           speed_bonus=+762.79  timeout=-10.50  stamina_penalty=-3.63
  rew/ep   (mean/std/min/max per episode, 556 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.835    0.386    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.005    0.066    -0.900    +0.000
  ball_out          -0.360    1.144    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.772    1.136    +0.000    +2.500
  speed_bonus       +1.372    1.285    +0.000    +4.058
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.019    0.167    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.007    0.006    -0.023    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     465    +0.013    0.113     +3.751     1.940     +0.693      3.7172      1.544     4.117
  lose_possession       3    -0.000    0.008     +2.839     0.116     -0.462      0.4068      0.548     0.892
  ball_out            50    -0.006    0.149     -3.980     0.140     -3.506     13.9735      3.506     5.854
  box_possession     394    +0.027    0.260     +4.430     1.109     +0.840      1.8570      1.091     2.612
  speed_bonus        376    +0.021    0.233     +4.522     1.049     +0.880      1.8933      1.108     2.611
  timeout              7    -0.000    0.021     -1.510     0.005     -5.170     26.9916      5.170     5.822
  stamina_penalty     396    -0.000    0.001     +4.352     1.318     +0.753      2.2711      1.161     2.836
  gae/td   mean_return=+3.014  std_return=1.679  mean_gae=+0.024  mean_sq_td=1.3553
──────────────────────────────────────────────────────────────────────
2026-08-09 00:47:03,906 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint74.pt
2026-08-09 00:47:03,906 INFO Logging to checkpoints/phase1_run45/training_log75.txt
2026-08-09 00:47:03,907 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 00:47:17,531 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 00:47:17,532 INFO   [eval vs immobile] step=2,736,000  seeds=16x8  win=59%  mean_rew=3.077±2.979  V=2.707  gap=-0.370  outcomes={'other': 22, 'box_possession': 75, 'miss': 30, 'timeout': 1}
2026-08-09 00:47:17,534 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 00:47:28,296 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 00:47:28,297 INFO   [eval vs rules] step=2,736,000  seeds=16x8  win=1%  mean_rew=-2.177±1.617  V=0.942  gap=+3.119  outcomes={'other': 14, 'opponent_box_possession': 94, 'miss': 19, 'box_possession': 1}
2026-08-09 00:52:18,547 INFO   [advantage] mean=-0.000  std=1.000  min=-6.097  max=3.738
2026-08-09 00:52:18,548 INFO   [ratio] mean=0.9750  std=0.1251  min=0.0089  max=3.7467  clipped=14.5%
2026-08-09 00:52:18,548 INFO   [exec head grad norm] move_direction=0.025  exec_move=0.041  sprint=0.050  kick=0.034  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.053
2026-08-09 00:52:18,548 INFO   [exec continuous log_std] move_direction: start=-1.5856 end=-1.5850   kick_direction: start=-1.5803 end=-1.5796
2026-08-09 00:52:18,549 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0014≈0.08°/step  epoch≈4.9°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0015≈0.09°/step  epoch≈5.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 00:52:18,549 INFO   [exec discrete Δlogit per opt step] exec_move=0.0009  sprint=0.0010  kick=0.0010  tackle_attempt=0.0005
2026-08-09 00:52:18,549 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0006  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0009  sprint=+0.0012  kick=+0.0012  tackle_attempt=+0.0007  move_dir=+0.0234  kick_dir=+0.0085
2026-08-09 00:52:18,549 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.243 max=0.395  limit=0.4
              direction: 48/60 steps clipped (80%)  pre-clip norm mean=0.029 max=0.057  limit=0.02
2026-08-09 00:52:18,606 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=2,772,000  speed=877/s  reward=2.72
  loss     policy=0.0022  value=0.4635(x0.5)=0.2318
           entropy=6.9092  kl=0.0364
  value    V=3.10±1.16  R=3.12±1.66  adv=0.02±1.14
  moves    mv_ls=[-1.5850] (σ≈0.20, ≈12°) g=6.68e-03  d_move=[+0.0007] (Δσ≈0.008°)
           kk_ls=[-1.5796] (σ≈0.21, ≈12°)  d_kick=[+0.0007] (Δσ≈0.008°)
  heads    move= 50 get_poss= 39 exec_move= 80 sprint= 49 kick= 23 tackle= 49 shoot= 49 hold= 49 tackle_prob=0.4868 kick_prob=0.2294
  vs       vs[win/loss/tout/miss]  vs_immobile(551): 71.0%/0.0%/3.3%/17.6%/8%
  ep_len   19.4±11.5s  (n=551, min=0.1s, max=50.0s)
  reward   get_possession=+464.00  lose_possession=-2.70  ball_out=-196.00  box_possession=+977.50
           speed_bonus=+789.85  timeout=-27.00  stamina_penalty=-3.77
  rew/ep   (mean/std/min/max per episode, 551 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.842    0.379    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.005    0.066    -0.900    +0.000
  ball_out          -0.356    1.139    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.774    1.135    +0.000    +2.500
  speed_bonus       +1.433    1.317    +0.000    +4.200
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.049    0.267    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.007    0.006    -0.025    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     468    +0.013    0.113     +3.762     2.102     +0.667      3.8012      1.562     4.334
  lose_possession       3    -0.000    0.008     +2.908     1.288     -0.406      1.9704      1.107     2.128
  ball_out            49    -0.005    0.147     -3.959     0.198     -3.642     16.0098      3.642     6.411
  box_possession     391    +0.027    0.259     +4.511     1.122     +0.575      1.2837      0.923     2.133
  speed_bonus        373    +0.022    0.240     +4.608     1.055     +0.636      1.3146      0.934     2.194
  timeout             18    -0.001    0.034     -1.513     0.005     -5.028     25.7883      5.028     6.023
  stamina_penalty     401    -0.000    0.001     +4.262     1.663     +0.328      2.4028      1.116     2.975
  gae/td   mean_return=+3.119  std_return=1.659  mean_gae=+0.022  mean_sq_td=1.2910
──────────────────────────────────────────────────────────────────────
2026-08-09 00:52:18,632 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint75.pt
2026-08-09 00:52:18,632 INFO Logging to checkpoints/phase1_run45/training_log76.txt
2026-08-09 00:52:18,634 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 00:52:31,412 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 00:52:31,413 INFO   [eval vs immobile] step=2,772,000  seeds=16x8  win=56%  mean_rew=2.849±3.140  V=2.643  gap=-0.206  outcomes={'other': 22, 'box_possession': 72, 'miss': 32, 'timeout': 2}
2026-08-09 00:52:31,414 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 00:52:42,471 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 00:52:42,472 INFO   [eval vs rules] step=2,772,000  seeds=16x8  win=3%  mean_rew=-2.132±1.990  V=0.944  gap=+3.076  outcomes={'opponent_box_possession': 98, 'other': 10, 'box_possession': 4, 'miss': 16}
2026-08-09 00:57:33,119 INFO   [advantage] mean=0.000  std=1.000  min=-6.365  max=4.013
2026-08-09 00:57:33,121 INFO   [ratio] mean=0.9757  std=0.1284  min=0.0048  max=6.1928  clipped=14.2%
2026-08-09 00:57:33,121 INFO   [exec head grad norm] move_direction=0.025  exec_move=0.037  sprint=0.045  kick=0.035  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.056
2026-08-09 00:57:33,121 INFO   [exec continuous log_std] move_direction: start=-1.5850 end=-1.5842   kick_direction: start=-1.5796 end=-1.5790
2026-08-09 00:57:33,122 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0015≈0.09°/step  epoch≈5.2°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0016≈0.09°/step  epoch≈5.6°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 00:57:33,122 INFO   [exec discrete Δlogit per opt step] exec_move=0.0010  sprint=0.0008  kick=0.0020  tackle_attempt=0.0006
2026-08-09 00:57:33,122 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0004  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0011  sprint=+0.0009  kick=+0.0014  tackle_attempt=+0.0007  move_dir=+0.0230  kick_dir=+0.0100
2026-08-09 00:57:33,123 INFO   [grad clip] main: 1/60 steps clipped (2%)  pre-clip norm mean=0.243 max=0.414  limit=0.4
              direction: 44/60 steps clipped (73%)  pre-clip norm mean=0.028 max=0.059  limit=0.02
2026-08-09 00:57:33,178 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=2,808,000  speed=869/s  reward=3.96
  loss     policy=0.0030  value=0.4725(x0.5)=0.2363
           entropy=6.9009  kl=0.0375
  value    V=3.11±1.19  R=3.16±1.62  adv=0.05±1.12
  moves    mv_ls=[-1.5842] (σ≈0.21, ≈12°) g=7.13e-03  d_move=[+0.0007] (Δσ≈0.008°)
           kk_ls=[-1.5790] (σ≈0.21, ≈12°)  d_kick=[+0.0006] (Δσ≈0.008°)
  heads    move= 50 get_poss= 39 exec_move= 80 sprint= 50 kick= 23 tackle= 49 shoot= 49 hold= 49 tackle_prob=0.4894 kick_prob=0.2281
  vs       vs[win/loss/tout/miss]  vs_immobile(563): 72.3%/0.0%/2.8%/15.8%/9%
  ep_len   18.9±11.3s  (n=563, min=1.0s, max=50.0s)
  reward   get_possession=+475.00  lose_possession=-8.10  ball_out=-160.00  box_possession=+1017.50
           speed_bonus=+820.79  timeout=-24.00  stamina_penalty=-3.87
  rew/ep   (mean/std/min/max per episode, 563 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.844    0.400    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.014    0.113    -0.900    +0.000
  ball_out          -0.284    1.028    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.807    1.119    +0.000    +2.500
  speed_bonus       +1.458    1.312    +0.000    +4.027
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.043    0.249    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.007    0.006    -0.022    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     479    +0.013    0.115     +3.921     1.811     +0.665      3.3134      1.431     3.846
  lose_possession       9    -0.000    0.014     +2.523     1.396     -0.588      3.1132      1.261     3.501
  ball_out            40    -0.004    0.133     -3.975     0.156     -4.354     20.5921      4.354     6.507
  box_possession     407    +0.028    0.264     +4.508     1.119     +0.579      1.3745      0.930     2.392
  speed_bonus        383    +0.023    0.244     +4.634     1.030     +0.629      1.4149      0.951     2.397
  timeout             16    -0.001    0.032     -1.512     0.004     -4.599     21.9278      4.599     5.975
  stamina_penalty     420    -0.000    0.001     +4.290     1.590     +0.384      2.1623      1.071     2.843
  gae/td   mean_return=+3.158  std_return=1.624  mean_gae=+0.049  mean_sq_td=1.2521
──────────────────────────────────────────────────────────────────────
2026-08-09 00:57:33,207 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint76.pt
2026-08-09 00:57:33,208 INFO Logging to checkpoints/phase1_run45/training_log77.txt
2026-08-09 00:57:33,209 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 00:57:46,430 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 00:57:46,432 INFO   [eval vs immobile] step=2,808,000  seeds=16x8  win=55%  mean_rew=2.797±3.220  V=2.764  gap=-0.033  outcomes={'other': 23, 'box_possession': 71, 'miss': 32, 'timeout': 2}
2026-08-09 00:57:46,433 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 00:57:56,997 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 00:57:56,999 INFO   [eval vs rules] step=2,808,000  seeds=16x8  win=5%  mean_rew=-1.820±2.310  V=0.958  gap=+2.778  outcomes={'opponent_box_possession': 92, 'miss': 17, 'other': 13, 'box_possession': 6}
2026-08-09 01:02:40,857 INFO   [advantage] mean=0.000  std=1.000  min=-6.589  max=4.382
2026-08-09 01:02:40,858 INFO   [ratio] mean=0.9766  std=0.1215  min=0.0073  max=4.1417  clipped=13.7%
2026-08-09 01:02:40,858 INFO   [exec head grad norm] move_direction=0.030  exec_move=0.042  sprint=0.046  kick=0.036  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.056
2026-08-09 01:02:40,858 INFO   [exec continuous log_std] move_direction: start=-1.5842 end=-1.5836   kick_direction: start=-1.5790 end=-1.5784
2026-08-09 01:02:40,858 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0017≈0.10°/step  epoch≈5.8°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0017≈0.10°/step  epoch≈5.7°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 01:02:40,858 INFO   [exec discrete Δlogit per opt step] exec_move=0.0008  sprint=0.0008  kick=0.0012  tackle_attempt=0.0007
2026-08-09 01:02:40,859 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0006  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0011  sprint=+0.0011  kick=+0.0008  tackle_attempt=+0.0011  move_dir=+0.0211  kick_dir=+0.0081
2026-08-09 01:02:40,859 INFO   [grad clip] main: 3/60 steps clipped (5%)  pre-clip norm mean=0.258 max=0.513  limit=0.4
              direction: 51/60 steps clipped (85%)  pre-clip norm mean=0.033 max=0.086  limit=0.02
2026-08-09 01:02:40,933 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=2,844,000  speed=874/s  reward=3.43
  loss     policy=0.0032  value=0.4348(x0.5)=0.2174
           entropy=6.8926  kl=0.0340
  value    V=3.15±1.22  R=3.18±1.68  adv=0.03±1.12
  moves    mv_ls=[-1.5836] (σ≈0.21, ≈12°) g=7.08e-03  d_move=[+0.0007] (Δσ≈0.008°)
           kk_ls=[-1.5784] (σ≈0.21, ≈12°)  d_kick=[+0.0006] (Δσ≈0.007°)
  heads    move= 49 get_poss= 39 exec_move= 80 sprint= 50 kick= 22 tackle= 49 shoot= 49 hold= 49 tackle_prob=0.4858 kick_prob=0.2195
  vs       vs[win/loss/tout/miss]  vs_immobile(587): 70.7%/0.2%/1.9%/17.5%/10%
  ep_len   18.3±10.4s  (n=587, min=1.3s, max=50.0s)
  reward   get_possession=+486.00  lose_possession=-3.60  ball_out=-188.00  box_possession=+1037.50
           speed_bonus=+833.68  opponent_box=-3.00  timeout=-16.50  stamina_penalty=-3.77
  rew/ep   (mean/std/min/max per episode, 587 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.828    0.391    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.006    0.074    -0.900    +0.000
  ball_out          -0.320    1.086    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.767    1.138    +0.000    +2.500
  speed_bonus       +1.420    1.290    +0.000    +4.084
  opponent_box      -0.005    0.124    -3.000    +0.000
  timeout           -0.028    0.203    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.023    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     490    +0.014    0.116     +3.871     1.960     +0.652      3.3032      1.440     4.138
  lose_possession       4    -0.000    0.009     +2.596     1.522     -0.361      0.4511      0.498     1.115
  ball_out            47    -0.005    0.144     -3.957     0.202     -3.797     16.3758      3.797     6.154
  box_possession     415    +0.029    0.267     +4.503     1.078     +0.505      1.1623      0.866     2.147
  speed_bonus        395    +0.023    0.244     +4.604     1.003     +0.565      1.1806      0.869     2.178
  opponent_box         1    -0.000    0.016     -3.003     0.000     -5.114     26.1550      5.114     5.114
  timeout             11    -0.000    0.026     -1.511     0.004     -5.258     29.0041      5.258     6.184
  stamina_penalty     422    -0.000    0.001     +4.337     1.474     +0.341      1.9497      0.991     2.462
  gae/td   mean_return=+3.179  std_return=1.681  mean_gae=+0.025  mean_sq_td=1.2577
──────────────────────────────────────────────────────────────────────
2026-08-09 01:02:40,959 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint77.pt
2026-08-09 01:02:40,960 INFO Logging to checkpoints/phase1_run45/training_log78.txt
2026-08-09 01:02:40,961 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 01:02:55,312 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 01:02:55,314 INFO   [eval vs immobile] step=2,844,000  seeds=16x8  win=58%  mean_rew=2.908±3.204  V=2.843  gap=-0.066  outcomes={'other': 21, 'box_possession': 74, 'timeout': 4, 'miss': 29}
2026-08-09 01:02:55,315 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 01:03:06,714 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 01:03:06,715 INFO   [eval vs rules] step=2,844,000  seeds=16x8  win=5%  mean_rew=-1.812±2.311  V=0.818  gap=+2.631  outcomes={'opponent_box_possession': 89, 'other': 12, 'box_possession': 7, 'miss': 20}
2026-08-09 01:07:56,170 INFO   [advantage] mean=0.000  std=1.000  min=-5.885  max=3.150
2026-08-09 01:07:56,171 INFO   [ratio] mean=0.9770  std=0.1263  min=0.0045  max=6.1321  clipped=14.2%
2026-08-09 01:07:56,171 INFO   [exec head grad norm] move_direction=0.036  exec_move=0.044  sprint=0.051  kick=0.033  kick_direction=0.008  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.060
2026-08-09 01:07:56,171 INFO   [exec continuous log_std] move_direction: start=-1.5836 end=-1.5828   kick_direction: start=-1.5784 end=-1.5777
2026-08-09 01:07:56,172 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0015≈0.09°/step  epoch≈5.3°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0016≈0.09°/step  epoch≈5.4°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 01:07:56,172 INFO   [exec discrete Δlogit per opt step] exec_move=0.0010  sprint=0.0010  kick=0.0012  tackle_attempt=0.0006
2026-08-09 01:07:56,172 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0007  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0012  sprint=+0.0011  kick=+0.0010  tackle_attempt=+0.0009  move_dir=+0.0203  kick_dir=+0.0093
2026-08-09 01:07:56,172 INFO   [grad clip] main: 2/60 steps clipped (3%)  pre-clip norm mean=0.297 max=2.715  limit=0.4
              direction: 43/60 steps clipped (72%)  pre-clip norm mean=0.039 max=0.572  limit=0.02
2026-08-09 01:07:56,229 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=2,880,000  speed=869/s  reward=3.80
  loss     policy=0.0034  value=0.4827(x0.5)=0.2414
           entropy=6.9013  kl=0.0345
  value    V=3.20±1.27  R=3.15±1.77  adv=-0.05±1.25
  moves    mv_ls=[-1.5828] (σ≈0.21, ≈12°) g=6.80e-03  d_move=[+0.0007] (Δσ≈0.008°)
           kk_ls=[-1.5777] (σ≈0.21, ≈12°)  d_kick=[+0.0007] (Δσ≈0.008°)
  heads    move= 49 get_poss= 39 exec_move= 80 sprint= 50 kick= 22 tackle= 49 shoot= 49 hold= 49 tackle_prob=0.4897 kick_prob=0.2215
  vs       vs[win/loss/tout/miss]  vs_immobile(580): 68.8%/0.0%/1.7%/21.6%/8%
  ep_len   18.5±10.8s  (n=580, min=2.0s, max=50.0s)
  reward   get_possession=+482.00  lose_possession=-5.40  ball_out=-248.00  box_possession=+997.50
           speed_bonus=+823.34  timeout=-15.00  stamina_penalty=-3.85
  rew/ep   (mean/std/min/max per episode, 580 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.831    0.401    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.009    0.091    -0.900    +0.000
  ball_out          -0.428    1.236    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.720    1.158    +0.000    +2.500
  speed_bonus       +1.420    1.328    +0.000    +4.278
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.026    0.195    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.007    0.006    -0.024    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     485    +0.013    0.115     +3.779     2.061     +0.367      3.7618      1.487     4.702
  lose_possession       6    -0.000    0.012     +2.924     0.910     -0.510      0.7293      0.609     1.471
  ball_out            62    -0.007    0.166     -4.000     0.000     -3.952     18.1312      3.952     6.653
  box_possession     399    +0.028    0.262     +4.554     1.110     +0.517      1.2535      0.900     2.127
  speed_bonus        382    +0.023    0.246     +4.646     1.044     +0.556      1.2799      0.912     2.134
  timeout             10    -0.000    0.025     -1.513     0.004     -5.329     30.8359      5.329     6.368
  stamina_penalty     404    -0.000    0.001     +4.416     1.445     +0.375      1.9935      1.012     2.420
  gae/td   mean_return=+3.151  std_return=1.773  mean_gae=-0.053  mean_sq_td=1.5567
──────────────────────────────────────────────────────────────────────
2026-08-09 01:07:56,257 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint78.pt
2026-08-09 01:07:56,257 INFO Logging to checkpoints/phase1_run45/training_log79.txt
2026-08-09 01:07:56,258 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 01:08:10,519 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 01:08:10,521 INFO   [eval vs immobile] step=2,880,000  seeds=16x8  win=54%  mean_rew=2.631±3.168  V=2.682  gap=+0.051  outcomes={'other': 18, 'box_possession': 69, 'miss': 38, 'timeout': 3}
2026-08-09 01:08:10,522 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 01:08:22,498 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 01:08:22,499 INFO   [eval vs rules] step=2,880,000  seeds=16x8  win=5%  mean_rew=-1.995±2.109  V=0.911  gap=+2.906  outcomes={'opponent_box_possession': 93, 'miss': 18, 'box_possession': 6, 'other': 11}
2026-08-09 01:13:11,253 INFO   [advantage] mean=-0.000  std=1.000  min=-6.467  max=6.122
2026-08-09 01:13:11,254 INFO   [ratio] mean=0.9765  std=0.1264  min=0.0118  max=5.2217  clipped=14.0%
2026-08-09 01:13:11,254 INFO   [exec head grad norm] move_direction=0.028  exec_move=0.050  sprint=0.057  kick=0.030  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.066
2026-08-09 01:13:11,254 INFO   [exec continuous log_std] move_direction: start=-1.5828 end=-1.5822   kick_direction: start=-1.5777 end=-1.5770
2026-08-09 01:13:11,254 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0018≈0.10°/step  epoch≈6.2°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0018≈0.10°/step  epoch≈6.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 01:13:11,255 INFO   [exec discrete Δlogit per opt step] exec_move=0.0008  sprint=0.0011  kick=0.0010  tackle_attempt=0.0007
2026-08-09 01:13:11,255 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0005  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0016  sprint=+0.0008  kick=+0.0012  tackle_attempt=+0.0007  move_dir=+0.0217  kick_dir=+0.0084
2026-08-09 01:13:11,255 INFO   [grad clip] main: 1/60 steps clipped (2%)  pre-clip norm mean=0.267 max=0.463  limit=0.4
              direction: 49/60 steps clipped (82%)  pre-clip norm mean=0.031 max=0.081  limit=0.02
2026-08-09 01:13:11,311 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=2,916,000  speed=873/s  reward=3.12
  loss     policy=0.0025  value=0.4668(x0.5)=0.2334
           entropy=6.9022  kl=0.0349
  value    V=3.08±1.26  R=3.10±1.74  adv=0.03±1.19
  moves    mv_ls=[-1.5822] (σ≈0.21, ≈12°) g=6.26e-03  d_move=[+0.0006] (Δσ≈0.007°)
           kk_ls=[-1.5770] (σ≈0.21, ≈12°)  d_kick=[+0.0007] (Δσ≈0.008°)
  heads    move= 50 get_poss= 39 exec_move= 80 sprint= 51 kick= 22 tackle= 49 shoot= 49 hold= 50 tackle_prob=0.4975 kick_prob=0.2190
  vs       vs[win/loss/tout/miss]  vs_immobile(594): 68.7%/0.0%/1.7%/19.5%/10%
  ep_len   18.1±10.5s  (n=594, min=1.2s, max=50.0s)
  reward   get_possession=+489.00  lose_possession=-4.50  ball_out=-236.00  box_possession=+1020.00
           speed_bonus=+819.52  timeout=-15.00  stamina_penalty=-3.78
  rew/ep   (mean/std/min/max per episode, 594 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.823    0.399    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.008    0.082    -0.900    +0.000
  ball_out          -0.397    1.196    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.717    1.159    +0.000    +2.500
  speed_bonus       +1.380    1.318    +0.000    +4.127
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.025    0.193    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.028    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     491    +0.014    0.116     +3.730     2.112     +0.594      3.8818      1.577     4.512
  lose_possession       5    -0.000    0.011     +2.902     1.168     +0.132      0.3197      0.494     0.809
  ball_out            59    -0.007    0.162     -3.966     0.181     -3.440     14.7016      3.460     6.020
  box_possession     408    +0.028    0.265     +4.500     1.124     +0.527      1.1839      0.880     2.135
  speed_bonus        388    +0.023    0.244     +4.603     1.054     +0.575      1.2229      0.896     2.191
  timeout             10    -0.000    0.025     -1.513     0.006     -4.950     26.0710      4.950     6.091
  stamina_penalty     415    -0.000    0.001     +4.368     1.439     +0.397      1.7849      0.978     2.357
  gae/td   mean_return=+3.105  std_return=1.743  mean_gae=+0.028  mean_sq_td=1.4252
──────────────────────────────────────────────────────────────────────
2026-08-09 01:13:11,341 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint79.pt
2026-08-09 01:13:11,341 INFO Logging to checkpoints/phase1_run45/training_log80.txt
2026-08-09 01:13:11,342 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 01:13:24,406 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 01:13:24,408 INFO   [eval vs immobile] step=2,916,000  seeds=16x8  win=50%  mean_rew=2.298±3.364  V=2.710  gap=+0.412  outcomes={'other': 20, 'box_possession': 64, 'miss': 44}
2026-08-09 01:13:24,409 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 01:13:35,016 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 01:13:35,018 INFO   [eval vs rules] step=2,916,000  seeds=16x8  win=2%  mean_rew=-1.999±1.881  V=0.881  gap=+2.880  outcomes={'other': 15, 'opponent_box_possession': 93, 'box_possession': 3, 'miss': 17}
2026-08-09 01:18:22,793 INFO   [advantage] mean=0.000  std=1.000  min=-6.640  max=4.161
2026-08-09 01:18:22,793 INFO   [ratio] mean=0.9750  std=0.1243  min=0.0119  max=5.3298  clipped=14.4%
2026-08-09 01:18:22,794 INFO   [exec head grad norm] move_direction=0.023  exec_move=0.047  sprint=0.051  kick=0.034  kick_direction=0.008  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.060
2026-08-09 01:18:22,794 INFO   [exec continuous log_std] move_direction: start=-1.5822 end=-1.5815   kick_direction: start=-1.5770 end=-1.5762
2026-08-09 01:18:22,794 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0018≈0.10°/step  epoch≈6.2°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0016≈0.09°/step  epoch≈5.4°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 01:18:22,794 INFO   [exec discrete Δlogit per opt step] exec_move=0.0009  sprint=0.0011  kick=0.0015  tackle_attempt=0.0005
2026-08-09 01:18:22,794 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0005  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0012  sprint=+0.0012  kick=+0.0008  tackle_attempt=+0.0008  move_dir=+0.0233  kick_dir=+0.0087
2026-08-09 01:18:22,795 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.240 max=0.330  limit=0.4
              direction: 43/60 steps clipped (72%)  pre-clip norm mean=0.026 max=0.043  limit=0.02
2026-08-09 01:18:22,851 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=2,952,000  speed=873/s  reward=3.00
  loss     policy=0.0032  value=0.4858(x0.5)=0.2429
           entropy=6.9045  kl=0.0365
  value    V=3.18±1.25  R=3.18±1.69  adv=0.00±1.17
  moves    mv_ls=[-1.5815] (σ≈0.21, ≈12°) g=6.89e-03  d_move=[+0.0007] (Δσ≈0.009°)
           kk_ls=[-1.5762] (σ≈0.21, ≈12°)  d_kick=[+0.0008] (Δσ≈0.010°)
  heads    move= 50 get_poss= 39 exec_move= 80 sprint= 50 kick= 22 tackle= 50 shoot= 49 hold= 49 tackle_prob=0.5008 kick_prob=0.2165
  vs       vs[win/loss/tout/miss]  vs_immobile(565): 73.8%/0.0%/0.5%/17.5%/8%
  ep_len   19.0±10.7s  (n=565, min=1.8s, max=50.0s)
  reward   get_possession=+491.00  lose_possession=-5.40  ball_out=-236.00  box_possession=+1042.50
           speed_bonus=+793.90  timeout=-4.50  stamina_penalty=-3.73
  rew/ep   (mean/std/min/max per episode, 565 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.869    0.368    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.010    0.092    -0.900    +0.000
  ball_out          -0.418    1.223    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.845    1.099    +0.000    +2.500
  speed_bonus       +1.405    1.278    +0.000    +4.079
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.008    0.109    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.007    0.006    -0.024    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     495    +0.014    0.116     +3.743     2.029     +0.421      3.5661      1.429     4.811
  lose_possession       6    -0.000    0.012     +2.237     1.809     -1.192      4.4784      1.353     4.035
  ball_out            59    -0.007    0.162     -3.983     0.129     -3.370     13.4473      3.378     5.717
  box_possession     417    +0.029    0.268     +4.395     1.123     +0.687      1.6536      1.016     2.414
  speed_bonus        395    +0.022    0.237     +4.501     1.058     +0.721      1.6956      1.035     2.416
  timeout              3    -0.000    0.014     -1.512     0.005     -5.364     28.8361      5.364     5.665
  stamina_penalty     418    -0.000    0.001     +4.355     1.225     +0.645      1.8562      1.051     2.494
  gae/td   mean_return=+3.182  std_return=1.686  mean_gae=+0.000  mean_sq_td=1.3740
──────────────────────────────────────────────────────────────────────
2026-08-09 01:18:22,877 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint80.pt
2026-08-09 01:18:22,878 INFO Logging to checkpoints/phase1_run45/training_log81.txt
2026-08-09 01:18:22,879 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 01:18:36,076 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 01:18:36,078 INFO   [eval vs immobile] step=2,952,000  seeds=16x8  win=55%  mean_rew=2.703±3.053  V=2.724  gap=+0.020  outcomes={'other': 20, 'box_possession': 71, 'miss': 34, 'timeout': 3}
2026-08-09 01:18:36,079 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 01:18:45,903 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 01:18:45,904 INFO   [eval vs rules] step=2,952,000  seeds=16x8  win=2%  mean_rew=-2.195±1.695  V=0.960  gap=+3.156  outcomes={'other': 11, 'opponent_box_possession': 95, 'miss': 20, 'box_possession': 2}
2026-08-09 01:23:30,094 INFO   [advantage] mean=-0.000  std=1.000  min=-6.979  max=3.696
2026-08-09 01:23:30,095 INFO   [ratio] mean=0.9733  std=0.1250  min=0.0075  max=3.7388  clipped=14.9%
2026-08-09 01:23:30,095 INFO   [exec head grad norm] move_direction=0.028  exec_move=0.045  sprint=0.050  kick=0.036  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.061
2026-08-09 01:23:30,096 INFO   [exec continuous log_std] move_direction: start=-1.5815 end=-1.5807   kick_direction: start=-1.5762 end=-1.5754
2026-08-09 01:23:30,096 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0017≈0.10°/step  epoch≈5.9°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0017≈0.10°/step  epoch≈5.7°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 01:23:30,096 INFO   [exec discrete Δlogit per opt step] exec_move=0.0010  sprint=0.0011  kick=0.0014  tackle_attempt=0.0006
2026-08-09 01:23:30,096 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0005  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0011  sprint=+0.0007  kick=+0.0010  tackle_attempt=+0.0008  move_dir=+0.0250  kick_dir=+0.0092
2026-08-09 01:23:30,096 INFO   [grad clip] main: 2/60 steps clipped (3%)  pre-clip norm mean=0.269 max=0.671  limit=0.4
              direction: 45/60 steps clipped (75%)  pre-clip norm mean=0.031 max=0.143  limit=0.02
2026-08-09 01:23:30,139 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=2,988,000  speed=869/s  reward=3.59
  loss     policy=0.0029  value=0.4920(x0.5)=0.2460
           entropy=6.9001  kl=0.0383
  value    V=3.11±1.20  R=3.14±1.52  adv=0.04±1.08
  moves    mv_ls=[-1.5807] (σ≈0.21, ≈12°) g=7.23e-03  d_move=[+0.0008] (Δσ≈0.009°)
           kk_ls=[-1.5754] (σ≈0.21, ≈12°)  d_kick=[+0.0007] (Δσ≈0.009°)
  heads    move= 50 get_poss= 39 exec_move= 79 sprint= 51 kick= 21 tackle= 50 shoot= 49 hold= 49 tackle_prob=0.5040 kick_prob=0.2164
  vs       vs[win/loss/tout/miss]  vs_immobile(538): 73.4%/0.0%/2.2%/16.5%/8%
  ep_len   19.9±11.4s  (n=538, min=0.2s, max=50.0s)
  reward   get_possession=+453.00  lose_possession=-9.00  ball_out=-136.00  box_possession=+987.50
           speed_bonus=+730.63  timeout=-18.00  stamina_penalty=-3.90
  rew/ep   (mean/std/min/max per episode, 538 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.842    0.417    +0.000    +3.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.017    0.133    -1.800    +0.000
  ball_out          -0.253    0.973    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.836    1.104    +0.000    +2.500
  speed_bonus       +1.358    1.257    +0.000    +4.079
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.033    0.222    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.007    0.006    -0.024    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     457    +0.013    0.112     +3.879     1.734     +0.635      3.0523      1.356     3.644
  lose_possession      10    -0.000    0.015     +2.937     0.493     -0.002      0.6965      0.677     1.518
  ball_out            34    -0.004    0.123     -3.853     0.354     -3.523     14.9805      3.523     6.054
  box_possession     395    +0.027    0.260     +4.345     1.113     +0.746      1.5860      1.007     2.407
  speed_bonus        377    +0.020    0.225     +4.434     1.061     +0.781      1.6367      1.027     2.443
  timeout             12    -0.001    0.027     -1.515     0.003     -5.861     34.4045      5.861     6.222
  stamina_penalty     405    -0.000    0.001     +4.177     1.480     +0.549      2.5609      1.151     3.060
  gae/td   mean_return=+3.145  std_return=1.522  mean_gae=+0.037  mean_sq_td=1.1771
──────────────────────────────────────────────────────────────────────
2026-08-09 01:23:30,164 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint81.pt
2026-08-09 01:23:30,165 INFO Logging to checkpoints/phase1_run45/training_log82.txt
2026-08-09 01:23:30,166 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 01:23:43,910 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 01:23:43,912 INFO   [eval vs immobile] step=2,988,000  seeds=16x8  win=59%  mean_rew=3.035±3.086  V=2.933  gap=-0.103  outcomes={'other': 20, 'box_possession': 76, 'miss': 30, 'timeout': 2}
2026-08-09 01:23:43,913 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 01:23:54,780 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 01:23:54,781 INFO   [eval vs rules] step=2,988,000  seeds=16x8  win=4%  mean_rew=-1.918±2.125  V=1.013  gap=+2.931  outcomes={'opponent_box_possession': 89, 'other': 12, 'box_possession': 5, 'miss': 22}
2026-08-09 01:28:36,789 INFO   [advantage] mean=-0.000  std=1.000  min=-6.864  max=4.458
2026-08-09 01:28:36,790 INFO   [ratio] mean=0.9769  std=0.1208  min=0.0139  max=3.2167  clipped=13.7%
2026-08-09 01:28:36,790 INFO   [exec head grad norm] move_direction=0.029  exec_move=0.041  sprint=0.041  kick=0.033  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.054
2026-08-09 01:28:36,791 INFO   [exec continuous log_std] move_direction: start=-1.5807 end=-1.5800   kick_direction: start=-1.5754 end=-1.5748
2026-08-09 01:28:36,791 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0021≈0.12°/step  epoch≈7.4°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0019≈0.11°/step  epoch≈6.5°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 01:28:36,791 INFO   [exec discrete Δlogit per opt step] exec_move=0.0009  sprint=0.0010  kick=0.0012  tackle_attempt=0.0005
2026-08-09 01:28:36,791 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0008  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0010  sprint=+0.0008  kick=+0.0009  tackle_attempt=+0.0009  move_dir=+0.0212  kick_dir=+0.0082
2026-08-09 01:28:36,791 INFO   [grad clip] main: 1/60 steps clipped (2%)  pre-clip norm mean=0.246 max=0.431  limit=0.4
              direction: 52/60 steps clipped (87%)  pre-clip norm mean=0.032 max=0.069  limit=0.02
2026-08-09 01:28:36,841 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=3,024,000  speed=871/s  reward=3.98
  loss     policy=0.0036  value=0.4902(x0.5)=0.2451
           entropy=6.9147  kl=0.0338
  value    V=3.23±1.06  R=3.21±1.57  adv=-0.02±1.10
  moves    mv_ls=[-1.5800] (σ≈0.21, ≈12°) g=7.48e-03  d_move=[+0.0007] (Δσ≈0.009°)
           kk_ls=[-1.5748] (σ≈0.21, ≈12°)  d_kick=[+0.0007] (Δσ≈0.008°)
  heads    move= 49 get_poss= 39 exec_move= 79 sprint= 51 kick= 22 tackle= 50 shoot= 49 hold= 49 tackle_prob=0.5063 kick_prob=0.2248
  vs       vs[win/loss/tout/miss]  vs_immobile(585): 70.6%/0.3%/2.6%/16.9%/10%
  ep_len   18.4±11.5s  (n=585, min=0.1s, max=50.0s)
  reward   get_possession=+473.00  lose_possession=-3.60  ball_out=-164.00  box_possession=+1032.50
           speed_bonus=+802.75  opponent_box=-6.00  timeout=-22.50  stamina_penalty=-3.96
  rew/ep   (mean/std/min/max per episode, 585 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.809    0.410    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.006    0.074    -0.900    +0.000
  ball_out          -0.280    1.021    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.765    1.139    +0.000    +2.500
  speed_bonus       +1.372    1.260    +0.000    +4.037
  opponent_box      -0.010    0.175    -3.000    +0.000
  timeout           -0.038    0.237    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.007    0.006    -0.023    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     475    +0.013    0.114     +3.919     1.844     +0.559      3.2800      1.367     4.512
  lose_possession       4    -0.000    0.009     +2.757     0.253     -0.621      0.5339      0.621     1.097
  ball_out            41    -0.005    0.135     -3.976     0.154     -4.451     22.8417      4.451     6.944
  box_possession     413    +0.029    0.266     +4.435     1.066     +0.419      0.9719      0.821     1.788
  speed_bonus        397    +0.022    0.236     +4.513     1.012     +0.459      0.9925      0.830     1.797
  opponent_box         2    -0.000    0.022     -3.002     0.002     -5.016     25.1995      5.016     5.193
  timeout             15    -0.001    0.031     -1.513     0.006     -4.734     22.8017      4.734     5.621
  stamina_penalty     426    -0.000    0.001     +4.218     1.554     +0.230      1.8067      0.972     2.422
  gae/td   mean_return=+3.206  std_return=1.568  mean_gae=-0.024  mean_sq_td=1.2178
──────────────────────────────────────────────────────────────────────
2026-08-09 01:28:36,868 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint82.pt
2026-08-09 01:28:36,869 INFO Logging to checkpoints/phase1_run45/training_log83.txt
2026-08-09 01:28:36,870 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 01:28:50,999 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 01:28:51,001 INFO   [eval vs immobile] step=3,024,000  seeds=16x8  win=61%  mean_rew=3.215±2.918  V=2.878  gap=-0.337  outcomes={'other': 23, 'box_possession': 78, 'miss': 26, 'timeout': 1}
2026-08-09 01:28:51,003 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 01:29:01,877 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 01:29:01,878 INFO   [eval vs rules] step=3,024,000  seeds=16x8  win=3%  mean_rew=-1.977±2.011  V=1.080  gap=+3.057  outcomes={'miss': 22, 'other': 12, 'opponent_box_possession': 90, 'box_possession': 4}
2026-08-09 01:33:48,689 INFO   [advantage] mean=0.000  std=1.000  min=-5.985  max=3.095
2026-08-09 01:33:48,690 INFO   [ratio] mean=0.9785  std=0.1231  min=0.0141  max=6.9387  clipped=13.3%
2026-08-09 01:33:48,690 INFO   [exec head grad norm] move_direction=0.028  exec_move=0.040  sprint=0.052  kick=0.030  kick_direction=0.008  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.055
2026-08-09 01:33:48,690 INFO   [exec continuous log_std] move_direction: start=-1.5800 end=-1.5793   kick_direction: start=-1.5748 end=-1.5741
2026-08-09 01:33:48,691 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0016≈0.09°/step  epoch≈5.6°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0015≈0.09°/step  epoch≈5.3°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 01:33:48,691 INFO   [exec discrete Δlogit per opt step] exec_move=0.0008  sprint=0.0010  kick=0.0009  tackle_attempt=0.0005
2026-08-09 01:33:48,691 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0006  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0007  sprint=+0.0008  kick=+0.0010  tackle_attempt=+0.0009  move_dir=+0.0200  kick_dir=+0.0086
2026-08-09 01:33:48,692 INFO   [grad clip] main: 1/60 steps clipped (2%)  pre-clip norm mean=0.261 max=0.430  limit=0.4
              direction: 50/60 steps clipped (83%)  pre-clip norm mean=0.030 max=0.061  limit=0.02
2026-08-09 01:33:48,735 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=3,060,000  speed=881/s  reward=3.63
  loss     policy=0.0025  value=0.4607(x0.5)=0.2303
           entropy=6.9153  kl=0.0326
  value    V=3.15±1.16  R=3.13±1.67  adv=-0.01±1.13
  moves    mv_ls=[-1.5793] (σ≈0.21, ≈12°) g=6.03e-03  d_move=[+0.0007] (Δσ≈0.008°)
           kk_ls=[-1.5741] (σ≈0.21, ≈12°)  d_kick=[+0.0007] (Δσ≈0.008°)
  heads    move= 50 get_poss= 39 exec_move= 79 sprint= 51 kick= 23 tackle= 51 shoot= 49 hold= 50 tackle_prob=0.5089 kick_prob=0.2280
  vs       vs[win/loss/tout/miss]  vs_immobile(569): 68.5%/0.0%/2.1%/20.9%/8%
  ep_len   18.9±11.3s  (n=569, min=0.7s, max=50.0s)
  reward   get_possession=+465.00  lose_possession=-6.30  ball_out=-208.00  box_possession=+975.00
           speed_bonus=+783.53  timeout=-18.00  stamina_penalty=-3.82
  rew/ep   (mean/std/min/max per episode, 569 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.817    0.417    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.011    0.099    -0.900    +0.000
  ball_out          -0.366    1.153    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.714    1.161    +0.000    +2.500
  speed_bonus       +1.377    1.316    +0.000    +4.006
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.032    0.216    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.007    0.006    -0.026    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     469    +0.013    0.113     +3.737     2.063     +0.459      3.7990      1.499     4.804
  lose_possession       7    -0.000    0.013     +2.871     0.552     -0.198      0.5326      0.646     1.017
  ball_out            52    -0.006    0.152     -3.942     0.233     -3.336     13.9100      3.336     6.006
  box_possession     390    +0.027    0.259     +4.500     1.121     +0.372      0.9920      0.827     1.774
  speed_bonus        377    +0.022    0.238     +4.569     1.075     +0.401      0.9984      0.827     1.787
  timeout             12    -0.001    0.027     -1.512     0.004     -4.923     24.5285      4.923     5.795
  stamina_penalty     398    -0.000    0.001     +4.333     1.506     +0.218      1.7042      0.952     1.991
  gae/td   mean_return=+3.135  std_return=1.669  mean_gae=-0.012  mean_sq_td=1.2881
──────────────────────────────────────────────────────────────────────
2026-08-09 01:33:48,759 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint83.pt
2026-08-09 01:33:48,760 INFO Logging to checkpoints/phase1_run45/training_log84.txt
2026-08-09 01:33:48,761 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 01:34:01,330 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 01:34:01,331 INFO   [eval vs immobile] step=3,060,000  seeds=16x8  win=58%  mean_rew=2.914±3.129  V=2.782  gap=-0.132  outcomes={'other': 20, 'box_possession': 74, 'timeout': 2, 'miss': 32}
2026-08-09 01:34:01,333 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 01:34:12,451 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 01:34:12,453 INFO   [eval vs rules] step=3,060,000  seeds=16x8  win=2%  mean_rew=-2.039±1.841  V=1.027  gap=+3.066  outcomes={'opponent_box_possession': 94, 'other': 14, 'miss': 17, 'box_possession': 3}
2026-08-09 01:38:50,678 INFO   [advantage] mean=-0.000  std=1.000  min=-6.923  max=4.687
2026-08-09 01:38:50,679 INFO   [ratio] mean=0.9768  std=0.1216  min=0.0140  max=3.2063  clipped=12.8%
2026-08-09 01:38:50,679 INFO   [exec head grad norm] move_direction=0.028  exec_move=0.045  sprint=0.053  kick=0.036  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.063
2026-08-09 01:38:50,679 INFO   [exec continuous log_std] move_direction: start=-1.5793 end=-1.5786   kick_direction: start=-1.5741 end=-1.5733
2026-08-09 01:38:50,679 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0019≈0.11°/step  epoch≈6.5°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0018≈0.10°/step  epoch≈6.2°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 01:38:50,679 INFO   [exec discrete Δlogit per opt step] exec_move=0.0010  sprint=0.0009  kick=0.0013  tackle_attempt=0.0006
2026-08-09 01:38:50,680 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0007  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0008  sprint=+0.0009  kick=+0.0010  tackle_attempt=+0.0010  move_dir=+0.0205  kick_dir=+0.0096
2026-08-09 01:38:50,680 INFO   [grad clip] main: 4/60 steps clipped (7%)  pre-clip norm mean=0.272 max=0.584  limit=0.4
              direction: 46/60 steps clipped (77%)  pre-clip norm mean=0.031 max=0.119  limit=0.02
2026-08-09 01:38:50,726 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=3,096,000  speed=874/s  reward=3.93
  loss     policy=0.0034  value=0.4072(x0.5)=0.2036
           entropy=6.8998  kl=0.0345
  value    V=3.10±1.24  R=3.21±1.65  adv=0.11±1.06
  moves    mv_ls=[-1.5786] (σ≈0.21, ≈12°) g=6.26e-03  d_move=[+0.0007] (Δσ≈0.008°)
           kk_ls=[-1.5733] (σ≈0.21, ≈12°)  d_kick=[+0.0007] (Δσ≈0.009°)
  heads    move= 49 get_poss= 40 exec_move= 79 sprint= 52 kick= 22 tackle= 51 shoot= 49 hold= 49 tackle_prob=0.5094 kick_prob=0.2215
  vs       vs[win/loss/tout/miss]  vs_immobile(585): 70.6%/0.0%/1.5%/14.9%/13%
  ep_len   18.3±10.7s  (n=585, min=0.2s, max=50.0s)
  reward   get_possession=+477.00  lose_possession=-5.40  ball_out=-148.00  box_possession=+1032.50
           speed_bonus=+869.65  timeout=-13.50  stamina_penalty=-3.95
  rew/ep   (mean/std/min/max per episode, 585 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.815    0.414    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.009    0.091    -0.900    +0.000
  ball_out          -0.253    0.974    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.765    1.139    +0.000    +2.500
  speed_bonus       +1.487    1.347    +0.000    +4.127
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.023    0.185    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.007    0.006    -0.025    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     480    +0.013    0.115     +3.924     1.830     +0.746      2.9957      1.417     3.423
  lose_possession       6    -0.000    0.012     +2.984     0.446     -0.420      0.7364      0.686     1.420
  ball_out            37    -0.004    0.128     -4.000     0.000     -3.645     16.1360      3.645     6.149
  box_possession     413    +0.029    0.266     +4.596     1.125     +0.776      1.4930      1.003     2.308
  speed_bonus        400    +0.024    0.255     +4.665     1.077     +0.806      1.5280      1.019     2.321
  timeout              9    -0.000    0.024     -1.516     0.008     -5.245     28.4158      5.245     6.317
  stamina_penalty     418    -0.000    0.001     +4.492     1.390     +0.667      2.0299      1.089     2.442
  gae/td   mean_return=+3.208  std_return=1.648  mean_gae=+0.110  mean_sq_td=1.1260
──────────────────────────────────────────────────────────────────────
2026-08-09 01:38:50,757 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint84.pt
2026-08-09 01:38:50,757 INFO Logging to checkpoints/phase1_run45/training_log85.txt
2026-08-09 01:38:50,758 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 01:39:02,787 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 01:39:02,788 INFO   [eval vs immobile] step=3,096,000  seeds=16x8  win=63%  mean_rew=3.329±2.971  V=2.919  gap=-0.410  outcomes={'other': 21, 'miss': 26, 'box_possession': 81}
2026-08-09 01:39:02,789 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 01:39:13,522 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 01:39:13,524 INFO   [eval vs rules] step=3,096,000  seeds=16x8  win=1%  mean_rew=-2.394±1.319  V=1.059  gap=+3.452  outcomes={'opponent_box_possession': 97, 'other': 9, 'box_possession': 1, 'miss': 21}
2026-08-09 01:43:56,692 INFO   [advantage] mean=0.000  std=1.000  min=-6.580  max=4.764
2026-08-09 01:43:56,693 INFO   [ratio] mean=0.9782  std=0.1230  min=0.0164  max=4.4062  clipped=12.7%
2026-08-09 01:43:56,693 INFO   [exec head grad norm] move_direction=0.022  exec_move=0.041  sprint=0.049  kick=0.037  kick_direction=0.010  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.052
2026-08-09 01:43:56,693 INFO   [exec continuous log_std] move_direction: start=-1.5786 end=-1.5780   kick_direction: start=-1.5733 end=-1.5726
2026-08-09 01:43:56,693 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0012≈0.07°/step  epoch≈4.0°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0011≈0.06°/step  epoch≈3.7°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 01:43:56,693 INFO   [exec discrete Δlogit per opt step] exec_move=0.0006  sprint=0.0008  kick=0.0011  tackle_attempt=0.0006
2026-08-09 01:43:56,694 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0004  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0011  sprint=+0.0009  kick=+0.0013  tackle_attempt=+0.0006  move_dir=+0.0202  kick_dir=+0.0085
2026-08-09 01:43:56,694 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.234 max=0.353  limit=0.4
              direction: 43/60 steps clipped (72%)  pre-clip norm mean=0.026 max=0.066  limit=0.02
2026-08-09 01:43:56,750 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=3,132,000  speed=871/s  reward=3.44
  loss     policy=0.0036  value=0.3970(x0.5)=0.1985
           entropy=6.9010  kl=0.0330
  value    V=3.23±1.29  R=3.30±1.67  adv=0.07±1.06
  moves    mv_ls=[-1.5780] (σ≈0.21, ≈12°) g=5.78e-03  d_move=[+0.0007] (Δσ≈0.008°)
           kk_ls=[-1.5726] (σ≈0.21, ≈12°)  d_kick=[+0.0007] (Δσ≈0.008°)
  heads    move= 49 get_poss= 39 exec_move= 79 sprint= 52 kick= 23 tackle= 50 shoot= 49 hold= 49 tackle_prob=0.5052 kick_prob=0.2242
  vs       vs[win/loss/tout/miss]  vs_immobile(606): 71.1%/0.0%/0.7%/17.7%/11%
  ep_len   17.8±10.4s  (n=606, min=0.8s, max=50.0s)
  reward   get_possession=+491.00  lose_possession=-5.40  ball_out=-172.00  box_possession=+1077.50
           speed_bonus=+895.23  timeout=-6.00  stamina_penalty=-4.08
  rew/ep   (mean/std/min/max per episode, 606 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.810    0.417    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.009    0.089    -0.900    +0.000
  ball_out          -0.284    1.027    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.778    1.133    +0.000    +2.500
  speed_bonus       +1.477    1.333    +0.000    +4.132
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.010    0.121    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.007    0.006    -0.024    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     492    +0.014    0.116     +3.981     1.986     +0.471      3.6176      1.390     4.787
  lose_possession       6    -0.000    0.012     +2.629     0.588     -0.556      0.5114      0.611     1.154
  ball_out            43    -0.005    0.138     -3.977     0.151     -3.928     17.6646      3.928     6.239
  box_possession     431    +0.030    0.272     +4.568     1.119     +0.528      1.1019      0.873     1.957
  speed_bonus        415    +0.025    0.257     +4.648     1.062     +0.571      1.1082      0.876     1.949
  timeout              4    -0.000    0.016     -1.510     0.008     -4.645     22.2742      4.645     5.640
  stamina_penalty     430    -0.000    0.001     +4.529     1.252     +0.487      1.2948      0.907     1.987
  gae/td   mean_return=+3.297  std_return=1.672  mean_gae=+0.071  mean_sq_td=1.1219
──────────────────────────────────────────────────────────────────────
2026-08-09 01:43:56,775 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint85.pt
2026-08-09 01:43:56,775 INFO Logging to checkpoints/phase1_run45/training_log86.txt
2026-08-09 01:43:56,776 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 01:44:10,154 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 01:44:10,155 INFO   [eval vs immobile] step=3,132,000  seeds=16x8  win=52%  mean_rew=2.589±3.153  V=2.926  gap=+0.337  outcomes={'other': 23, 'box_possession': 67, 'timeout': 4, 'opponent_box_possession': 1, 'miss': 33}
2026-08-09 01:44:10,157 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 01:44:20,855 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 01:44:20,856 INFO   [eval vs rules] step=3,132,000  seeds=16x8  win=3%  mean_rew=-1.979±1.994  V=0.979  gap=+2.958  outcomes={'opponent_box_possession': 91, 'box_possession': 4, 'other': 8, 'miss': 25}
2026-08-09 01:49:05,560 INFO   [advantage] mean=0.000  std=1.000  min=-6.906  max=3.655
2026-08-09 01:49:05,561 INFO   [ratio] mean=0.9783  std=0.1190  min=0.0139  max=4.5625  clipped=12.6%
2026-08-09 01:49:05,568 INFO   [exec head grad norm] move_direction=0.026  exec_move=0.041  sprint=0.046  kick=0.036  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.061
2026-08-09 01:49:05,568 INFO   [exec continuous log_std] move_direction: start=-1.5780 end=-1.5773   kick_direction: start=-1.5726 end=-1.5719
2026-08-09 01:49:05,568 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0016≈0.09°/step  epoch≈5.5°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0016≈0.09°/step  epoch≈5.4°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 01:49:05,568 INFO   [exec discrete Δlogit per opt step] exec_move=0.0009  sprint=0.0009  kick=0.0012  tackle_attempt=0.0006
2026-08-09 01:49:05,568 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0005  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0014  sprint=+0.0010  kick=+0.0013  tackle_attempt=+0.0005  move_dir=+0.0191  kick_dir=+0.0083
2026-08-09 01:49:05,569 INFO   [grad clip] main: 1/60 steps clipped (2%)  pre-clip norm mean=0.259 max=0.454  limit=0.4
              direction: 42/60 steps clipped (70%)  pre-clip norm mean=0.029 max=0.067  limit=0.02
2026-08-09 01:49:05,623 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=3,168,000  speed=873/s  reward=3.79
  loss     policy=0.0016  value=0.3940(x0.5)=0.1970
           entropy=6.9070  kl=0.0321
  value    V=3.31±1.24  R=3.29±1.64  adv=-0.02±1.04
  moves    mv_ls=[-1.5773] (σ≈0.21, ≈12°) g=5.88e-03  d_move=[+0.0007] (Δσ≈0.008°)
           kk_ls=[-1.5719] (σ≈0.21, ≈12°)  d_kick=[+0.0007] (Δσ≈0.008°)
  heads    move= 49 get_poss= 39 exec_move= 78 sprint= 52 kick= 23 tackle= 50 shoot= 49 hold= 50 tackle_prob=0.5010 kick_prob=0.2230
  vs       vs[win/loss/tout/miss]  vs_immobile(561): 70.4%/0.0%/1.8%/17.5%/10%
  ep_len   19.1±10.8s  (n=561, min=0.7s, max=50.0s)
  reward   get_possession=+458.00  lose_possession=-4.50  ball_out=-152.00  box_possession=+987.50
           speed_bonus=+826.22  timeout=-15.00  stamina_penalty=-3.90
  rew/ep   (mean/std/min/max per episode, 561 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.816    0.410    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.008    0.085    -0.900    +0.000
  ball_out          -0.271    1.005    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.760    1.141    +0.000    +2.500
  speed_bonus       +1.473    1.331    +0.000    +4.142
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.027    0.198    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.007    0.006    -0.024    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     459    +0.013    0.112     +4.008     1.913     +0.455      2.8493      1.273     3.584
  lose_possession       5    -0.000    0.011     +3.037     0.377     -0.610      0.5500      0.610     1.106
  ball_out            38    -0.004    0.130     -3.974     0.160     -3.906     17.2884      3.906     6.065
  box_possession     395    +0.027    0.260     +4.582     1.105     +0.415      1.0814      0.811     2.083
  speed_bonus        382    +0.023    0.247     +4.653     1.053     +0.432      1.1005      0.822     2.092
  timeout             10    -0.000    0.025     -1.412     0.300     -4.948     25.4596      4.948     6.173
  stamina_penalty     403    -0.000    0.001     +4.441     1.435     +0.287      1.6898      0.915     2.464
  gae/td   mean_return=+3.293  std_return=1.643  mean_gae=-0.021  mean_sq_td=1.0720
──────────────────────────────────────────────────────────────────────
2026-08-09 01:49:05,649 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint86.pt
2026-08-09 01:49:05,650 INFO Logging to checkpoints/phase1_run45/training_log87.txt
2026-08-09 01:49:05,651 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 01:49:18,280 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 01:49:18,282 INFO   [eval vs immobile] step=3,168,000  seeds=16x8  win=61%  mean_rew=3.128±3.037  V=2.964  gap=-0.164  outcomes={'other': 20, 'box_possession': 78, 'opponent_box_possession': 2, 'miss': 28}
2026-08-09 01:49:18,283 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 01:49:29,801 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 01:49:29,802 INFO   [eval vs rules] step=3,168,000  seeds=16x8  win=0%  mean_rew=-2.330±1.318  V=1.086  gap=+3.416  outcomes={'opponent_box_possession': 97, 'other': 11, 'miss': 20}
2026-08-09 01:54:16,266 INFO   [advantage] mean=-0.000  std=1.000  min=-6.534  max=4.080
2026-08-09 01:54:16,267 INFO   [ratio] mean=0.9778  std=0.1213  min=0.0143  max=4.3258  clipped=13.4%
2026-08-09 01:54:16,268 INFO   [exec head grad norm] move_direction=0.035  exec_move=0.042  sprint=0.052  kick=0.037  kick_direction=0.010  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.051
2026-08-09 01:54:16,268 INFO   [exec continuous log_std] move_direction: start=-1.5773 end=-1.5767   kick_direction: start=-1.5719 end=-1.5713
2026-08-09 01:54:16,268 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0023≈0.13°/step  epoch≈7.8°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0022≈0.13°/step  epoch≈7.5°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 01:54:16,268 INFO   [exec discrete Δlogit per opt step] exec_move=0.0011  sprint=0.0015  kick=0.0013  tackle_attempt=0.0007
2026-08-09 01:54:16,268 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0007  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0010  sprint=+0.0011  kick=+0.0013  tackle_attempt=+0.0007  move_dir=+0.0195  kick_dir=+0.0085
2026-08-09 01:54:16,269 INFO   [grad clip] main: 5/60 steps clipped (8%)  pre-clip norm mean=0.287 max=0.704  limit=0.4
              direction: 52/60 steps clipped (87%)  pre-clip norm mean=0.038 max=0.130  limit=0.02
2026-08-09 01:54:16,331 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=3,204,000  speed=869/s  reward=3.89
  loss     policy=0.0029  value=0.4560(x0.5)=0.2280
           entropy=6.9063  kl=0.0327
  value    V=3.25±1.28  R=3.24±1.68  adv=-0.01±1.14
  moves    mv_ls=[-1.5767] (σ≈0.21, ≈12°) g=6.17e-03  d_move=[+0.0006] (Δσ≈0.007°)
           kk_ls=[-1.5713] (σ≈0.21, ≈12°)  d_kick=[+0.0006] (Δσ≈0.007°)
  heads    move= 49 get_poss= 39 exec_move= 79 sprint= 51 kick= 23 tackle= 50 shoot= 49 hold= 50 tackle_prob=0.4958 kick_prob=0.2253
  vs       vs[win/loss/tout/miss]  vs_immobile(592): 70.8%/0.2%/1.5%/18.2%/9%
  ep_len   18.1±10.7s  (n=592, min=0.7s, max=50.0s)
  reward   get_possession=+490.00  lose_possession=-3.60  ball_out=-192.00  box_possession=+1047.50
           speed_bonus=+846.91  opponent_box=-3.00  timeout=-13.50  stamina_penalty=-3.82
  rew/ep   (mean/std/min/max per episode, 592 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.828    0.395    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.006    0.074    -0.900    +0.000
  ball_out          -0.324    1.092    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.769    1.137    +0.000    +2.500
  speed_bonus       +1.431    1.292    +0.000    +4.163
  opponent_box      -0.005    0.123    -3.000    +0.000
  timeout           -0.023    0.184    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.025    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     492    +0.014    0.116     +3.878     2.069     +0.370      3.8123      1.421     5.054
  lose_possession       4    -0.000    0.009     +3.427     0.294     -0.206      0.2273      0.429     0.660
  ball_out            48    -0.005    0.146     -3.938     0.242     -4.028     19.1319      4.047     6.561
  box_possession     419    +0.029    0.268     +4.515     1.076     +0.478      1.2064      0.891     2.091
  speed_bonus        401    +0.024    0.246     +4.606     1.009     +0.518      1.2265      0.899     2.088
  opponent_box         1    -0.000    0.016     -3.004     0.000     -4.275     18.2788      4.275     4.275
  timeout              9    -0.000    0.024     -1.514     0.005     -5.015     25.5548      5.015     5.969
  stamina_penalty     426    -0.000    0.001     +4.379     1.416     +0.354      1.7669      0.988     2.336
  gae/td   mean_return=+3.244  std_return=1.682  mean_gae=-0.008  mean_sq_td=1.2975
──────────────────────────────────────────────────────────────────────
2026-08-09 01:54:16,357 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint87.pt
2026-08-09 01:54:16,357 INFO Logging to checkpoints/phase1_run45/training_log88.txt
2026-08-09 01:54:16,358 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 01:54:34,155 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 01:54:34,157 INFO   [eval vs immobile] step=3,204,000  seeds=16x8  win=59%  mean_rew=3.025±3.057  V=2.958  gap=-0.067  outcomes={'other': 20, 'box_possession': 76, 'miss': 31, 'timeout': 1}
2026-08-09 01:54:34,158 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 01:54:46,155 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 01:54:46,157 INFO   [eval vs rules] step=3,204,000  seeds=16x8  win=5%  mean_rew=-2.048±2.110  V=1.015  gap=+3.063  outcomes={'opponent_box_possession': 94, 'other': 7, 'box_possession': 6, 'miss': 21}
2026-08-09 02:00:10,832 INFO   [advantage] mean=0.000  std=1.000  min=-6.640  max=4.275
2026-08-09 02:00:10,833 INFO   [ratio] mean=0.9783  std=0.1295  min=0.0102  max=12.3454  clipped=13.1%
2026-08-09 02:00:10,833 INFO   [exec head grad norm] move_direction=0.025  exec_move=0.046  sprint=0.048  kick=0.038  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.056
2026-08-09 02:00:10,833 INFO   [exec continuous log_std] move_direction: start=-1.5767 end=-1.5759   kick_direction: start=-1.5713 end=-1.5705
2026-08-09 02:00:10,833 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0019≈0.11°/step  epoch≈6.6°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0019≈0.11°/step  epoch≈6.4°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 02:00:10,834 INFO   [exec discrete Δlogit per opt step] exec_move=0.0010  sprint=0.0010  kick=0.0011  tackle_attempt=0.0006
2026-08-09 02:00:10,834 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0007  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0010  sprint=+0.0008  kick=+0.0012  tackle_attempt=+0.0010  move_dir=+0.0197  kick_dir=+0.0085
2026-08-09 02:00:10,834 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.249 max=0.357  limit=0.4
              direction: 40/60 steps clipped (67%)  pre-clip norm mean=0.028 max=0.059  limit=0.02
2026-08-09 02:00:10,897 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=3,240,000  speed=735/s  reward=3.56
  loss     policy=0.0033  value=0.4325(x0.5)=0.2162
           entropy=6.9065  kl=0.0330
  value    V=3.31±1.21  R=3.23±1.72  adv=-0.07±1.13
  moves    mv_ls=[-1.5759] (σ≈0.21, ≈12°) g=6.95e-03  d_move=[+0.0008] (Δσ≈0.009°)
           kk_ls=[-1.5705] (σ≈0.21, ≈12°)  d_kick=[+0.0008] (Δσ≈0.010°)
  heads    move= 48 get_poss= 40 exec_move= 79 sprint= 52 kick= 22 tackle= 50 shoot= 49 hold= 49 tackle_prob=0.4933 kick_prob=0.2240
  vs       vs[win/loss/tout/miss]  vs_immobile(595): 68.6%/0.2%/2.0%/20.2%/9%
  ep_len   18.0±10.8s  (n=595, min=1.2s, max=50.0s)
  reward   get_possession=+492.00  lose_possession=-6.30  ball_out=-224.00  box_possession=+1020.00
           speed_bonus=+826.45  opponent_box=-3.00  timeout=-18.00  stamina_penalty=-4.02
  rew/ep   (mean/std/min/max per episode, 595 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.827    0.408    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.011    0.097    -0.900    +0.000
  ball_out          -0.376    1.168    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.714    1.161    +0.000    +2.500
  speed_bonus       +1.389    1.310    +0.000    +4.069
  opponent_box      -0.005    0.123    -3.000    +0.000
  timeout           -0.030    0.211    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.007    0.006    -0.024    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     495    +0.014    0.116     +3.732     2.238     +0.293      3.9232      1.446     5.076
  lose_possession       7    -0.000    0.013     +3.097     0.458     -0.436      0.4174      0.535     1.059
  ball_out            56    -0.006    0.158     -3.946     0.225     -3.745     16.9379      3.745     6.377
  box_possession     408    +0.028    0.265     +4.516     1.100     +0.317      0.9799      0.800     1.800
  speed_bonus        394    +0.023    0.244     +4.588     1.050     +0.348      0.9975      0.807     1.808
  opponent_box         1    -0.000    0.016     -3.005     0.000     -7.035     49.4933      7.035     7.035
  timeout             12    -0.001    0.027     -1.515     0.005     -5.162     27.2476      5.162     6.402
  stamina_penalty     417    -0.000    0.001     +4.333     1.523     +0.141      1.8560      0.942     2.263
  gae/td   mean_return=+3.232  std_return=1.720  mean_gae=-0.075  mean_sq_td=1.2886
──────────────────────────────────────────────────────────────────────
2026-08-09 02:00:10,934 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint88.pt
2026-08-09 02:00:10,934 INFO Logging to checkpoints/phase1_run45/training_log89.txt
2026-08-09 02:00:10,936 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 02:00:24,723 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 02:00:24,724 INFO   [eval vs immobile] step=3,240,000  seeds=16x8  win=60%  mean_rew=3.109±3.168  V=2.858  gap=-0.250  outcomes={'other': 20, 'box_possession': 77, 'timeout': 3, 'miss': 28}
2026-08-09 02:00:24,726 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 02:00:36,019 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 02:00:36,021 INFO   [eval vs rules] step=3,240,000  seeds=16x8  win=2%  mean_rew=-2.196±1.857  V=1.036  gap=+3.232  outcomes={'opponent_box_possession': 99, 'other': 13, 'miss': 13, 'box_possession': 3}
2026-08-09 02:05:58,470 INFO   [advantage] mean=0.000  std=1.000  min=-6.311  max=4.009
2026-08-09 02:05:58,471 INFO   [ratio] mean=0.9785  std=0.1244  min=0.0090  max=7.7376  clipped=12.9%
2026-08-09 02:05:58,471 INFO   [exec head grad norm] move_direction=0.026  exec_move=0.050  sprint=0.051  kick=0.029  kick_direction=0.009  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.064
2026-08-09 02:05:58,471 INFO   [exec continuous log_std] move_direction: start=-1.5759 end=-1.5752   kick_direction: start=-1.5705 end=-1.5698
2026-08-09 02:05:58,472 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0017≈0.10°/step  epoch≈5.8°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0016≈0.09°/step  epoch≈5.5°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 02:05:58,472 INFO   [exec discrete Δlogit per opt step] exec_move=0.0009  sprint=0.0011  kick=0.0013  tackle_attempt=0.0005
2026-08-09 02:05:58,472 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0004  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0009  sprint=+0.0011  kick=+0.0013  tackle_attempt=+0.0008  move_dir=+0.0195  kick_dir=+0.0082
2026-08-09 02:05:58,473 INFO   [grad clip] main: 1/60 steps clipped (2%)  pre-clip norm mean=0.263 max=0.441  limit=0.4
              direction: 49/60 steps clipped (82%)  pre-clip norm mean=0.029 max=0.076  limit=0.02
2026-08-09 02:05:58,533 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=3,276,000  speed=815/s  reward=4.60
  loss     policy=0.0025  value=0.4458(x0.5)=0.2229
           entropy=6.9123  kl=0.0323
  value    V=3.13±1.32  R=3.12±1.77  adv=-0.01±1.20
  moves    mv_ls=[-1.5752] (σ≈0.21, ≈12°) g=5.77e-03  d_move=[+0.0007] (Δσ≈0.008°)
           kk_ls=[-1.5698] (σ≈0.21, ≈12°)  d_kick=[+0.0007] (Δσ≈0.009°)
  heads    move= 48 get_poss= 39 exec_move= 79 sprint= 52 kick= 22 tackle= 49 shoot= 50 hold= 50 tackle_prob=0.4925 kick_prob=0.2225
  vs       vs[win/loss/tout/miss]  vs_immobile(574): 70.7%/0.0%/1.6%/16.9%/11%
  ep_len   18.6±10.4s  (n=574, min=1.4s, max=50.0s)
  reward   get_possession=+483.00  lose_possession=-5.40  ball_out=-228.00  box_possession=+1015.00
           speed_bonus=+821.89  timeout=-13.50  stamina_penalty=-3.92
  rew/ep   (mean/std/min/max per episode, 574 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.841    0.388    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.009    0.092    -0.900    +0.000
  ball_out          -0.397    1.196    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.768    1.137    +0.000    +2.500
  speed_bonus       +1.432    1.329    +0.000    +4.103
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.024    0.186    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.007    0.006    -0.027    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     486    +0.013    0.115     +3.864     2.005     +0.509      3.6232      1.483     4.501
  lose_possession       6    -0.000    0.012     +1.954     1.774     -0.944      2.6801      1.062     3.107
  ball_out            57    -0.006    0.159     -3.982     0.131     -3.872     18.3457      3.896     6.705
  box_possession     406    +0.028    0.264     +4.518     1.136     +0.502      1.0470      0.839     1.957
  speed_bonus        391    +0.023    0.246     +4.595     1.085     +0.552      1.0547      0.841     1.969
  timeout              9    -0.000    0.024     -1.518     0.004     -5.537     31.0553      5.537     6.321
  stamina_penalty     412    -0.000    0.001     +4.397     1.426     +0.381      1.6998      0.941     2.173
  gae/td   mean_return=+3.120  std_return=1.773  mean_gae=-0.010  mean_sq_td=1.4331
──────────────────────────────────────────────────────────────────────
2026-08-09 02:05:58,558 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint89.pt
2026-08-09 02:05:58,559 INFO Logging to checkpoints/phase1_run45/training_log90.txt
2026-08-09 02:05:58,561 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 02:06:14,308 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 02:06:14,310 INFO   [eval vs immobile] step=3,276,000  seeds=16x8  win=55%  mean_rew=2.674±3.236  V=2.875  gap=+0.202  outcomes={'other': 20, 'box_possession': 71, 'miss': 36, 'timeout': 1}
2026-08-09 02:06:14,312 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 02:06:26,624 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 02:06:26,626 INFO   [eval vs rules] step=3,276,000  seeds=16x8  win=2%  mean_rew=-2.061±1.762  V=1.052  gap=+3.113  outcomes={'opponent_box_possession': 95, 'other': 17, 'box_possession': 2, 'miss': 14}
2026-08-09 02:11:59,190 INFO   [advantage] mean=-0.000  std=1.000  min=-7.326  max=4.914
2026-08-09 02:11:59,191 INFO   [ratio] mean=0.9770  std=0.1238  min=0.0046  max=5.0608  clipped=13.5%
2026-08-09 02:11:59,191 INFO   [exec head grad norm] move_direction=0.035  exec_move=0.045  sprint=0.051  kick=0.031  kick_direction=0.008  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.069
2026-08-09 02:11:59,192 INFO   [exec continuous log_std] move_direction: start=-1.5752 end=-1.5746   kick_direction: start=-1.5698 end=-1.5693
2026-08-09 02:11:59,192 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0028≈0.16°/step  epoch≈9.5°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0021≈0.12°/step  epoch≈7.2°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 02:11:59,192 INFO   [exec discrete Δlogit per opt step] exec_move=0.0009  sprint=0.0012  kick=0.0018  tackle_attempt=0.0012
2026-08-09 02:11:59,192 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0004  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0009  sprint=+0.0008  kick=+0.0015  tackle_attempt=+0.0013  move_dir=+0.0215  kick_dir=+0.0074
2026-08-09 02:11:59,193 INFO   [grad clip] main: 2/60 steps clipped (3%)  pre-clip norm mean=0.264 max=0.482  limit=0.4
              direction: 56/60 steps clipped (93%)  pre-clip norm mean=0.037 max=0.072  limit=0.02
2026-08-09 02:11:59,237 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=3,312,000  speed=749/s  reward=4.15
  loss     policy=0.0035  value=0.3995(x0.5)=0.1998
           entropy=6.8988  kl=0.0338
  value    V=3.19±1.28  R=3.29±1.52  adv=0.10±0.97
  moves    mv_ls=[-1.5746] (σ≈0.21, ≈12°) g=6.82e-03  d_move=[+0.0006] (Δσ≈0.008°)
           kk_ls=[-1.5693] (σ≈0.21, ≈12°)  d_kick=[+0.0005] (Δσ≈0.006°)
  heads    move= 48 get_poss= 40 exec_move= 79 sprint= 51 kick= 21 tackle= 49 shoot= 50 hold= 50 tackle_prob=0.4995 kick_prob=0.2118
  vs       vs[win/loss/tout/miss]  vs_immobile(572): 76.7%/0.0%/1.0%/14.5%/8%
  ep_len   18.8±10.8s  (n=572, min=1.8s, max=50.0s)
  reward   get_possession=+477.00  lose_possession=-1.80  ball_out=-112.00  box_possession=+1097.50
           speed_bonus=+821.23  timeout=-9.00  stamina_penalty=-3.97
  rew/ep   (mean/std/min/max per episode, 572 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.834    0.381    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.003    0.053    -0.900    +0.000
  ball_out          -0.196    0.863    -4.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.919    1.056    +0.000    +2.500
  speed_bonus       +1.436    1.257    +0.000    +4.237
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.016    0.153    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.007    0.006    -0.024    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     480    +0.013    0.115     +4.055     1.660     +0.611      2.4105      1.191     3.278
  lose_possession       2    -0.000    0.007     +3.248     0.144     -0.319      0.1792      0.319     0.570
  ball_out            28    -0.003    0.112     -3.964     0.186     -3.667     15.2786      3.667     5.817
  box_possession     439    +0.030    0.274     +4.364     1.112     +0.581      1.4485      0.930     2.332
  speed_bonus        421    +0.023    0.239     +4.444     1.065     +0.625      1.4889      0.948     2.354
  timeout              6    -0.000    0.019     -1.513     0.008     -4.790     23.9550      4.790     6.213
  stamina_penalty     439    -0.000    0.001     +4.315     1.264     +0.518      1.7273      0.977     2.472
  gae/td   mean_return=+3.289  std_return=1.522  mean_gae=+0.103  mean_sq_td=0.9473
──────────────────────────────────────────────────────────────────────
2026-08-09 02:11:59,262 INFO Saved checkpoint: checkpoints/phase1_run45/checkpoint90.pt
2026-08-09 02:11:59,262 INFO Logging to checkpoints/phase1_run45/training_log91.txt
2026-08-09 02:11:59,263 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 02:12:14,047 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 02:12:14,049 INFO   [eval vs immobile] step=3,312,000  seeds=16x8  win=56%  mean_rew=2.085±2.673  V=3.009  gap=+0.924  outcomes={'other': 21, 'box_possession': 72, 'miss': 34, 'timeout': 1}
2026-08-09 02:12:14,050 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 02:12:27,909 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 02:12:27,911 INFO   [eval vs rules] step=3,312,000  seeds=16x8  win=2%  mean_rew=-2.112±1.723  V=1.130  gap=+3.242  outcomes={'opponent_box_possession': 96, 'other': 13, 'miss': 16, 'box_possession': 3}
2026-08-09 02:12:48,399 INFO Checkpoint dir: checkpoints/phase1_run46
2026-08-09 02:12:48,460 INFO Starting training: phase=phase1_get_possession, total_steps=6,000,000
2026-08-09 02:12:48,461 INFO Phase description: 1v1 scenario: learn to get possession and bring the ball toward the opponent box.  Decision network frozen except Move/GetPossession and latent vector.
/home/vincent/Documents/not_work/repos/FootballCoach/.venv/lib/python3.12/site-packages/torch/_compile.py:54: UserWarning: optimizer contains a parameter group with duplicate parameters; in future, this will cause an error; see github.com/pytorch/pytorch/issues/40967 for more information
  return disable_fn(*args, **kwargs)
2026-08-09 02:12:49,702 INFO Logging to checkpoints/phase1_run46/training_log1.txt
2026-08-09 02:12:49,703 INFO --latest: resolved to checkpoints/phase1_run45/latest.pt
2026-08-09 02:12:49,906 INFO Loaded checkpoint: checkpoints/phase1_run45/latest.pt (step 3312000)
2026-08-09 02:13:29,293 INFO Pre-PPO eval (rules opp): win=0.8%  mean_rew=-2.312  V=1.084  R=-1.553  gap=+2.637  outcomes={'opponent_box_possession': 98, 'other': 11, 'miss': 18, 'box_possession': 1}
2026-08-09 02:13:29,294 INFO   rew breakdown (rules, per ep): opponent_box=-2.30  get_possession=+0.23  lose_possession=-0.15  ball_out=-0.12  box_possession=+0.02  speed_bonus=+0.01  stamina_penalty=-0.01
2026-08-09 02:14:19,159 INFO Pre-PPO eval (immobile opp): win=57.0%  mean_rew=1.612  V=2.956  R=0.872  gap=+2.084  outcomes={'other': 21, 'box_possession': 73, 'miss': 34}
2026-08-09 02:14:19,160 INFO   rew breakdown (immobile, per ep): box_possession=+1.14  get_possession=+0.68  ball_out=-0.47  speed_bonus=+0.27  lose_possession=-0.01  stamina_penalty=-0.01
2026-08-09 02:15:37,405 INFO Pre-PPO eval (self-play):   win=39.8%  mean_rew=0.514  V=1.146  R=0.185  gap=+0.961  outcomes={'opponent_box_possession': 29, 'other': 15, 'miss': 29, 'box_possession': 51, 'timeout': 4}
2026-08-09 02:15:37,405 INFO   rew breakdown (self-play, per ep): opponent_box=-1.88  box_possession=+1.25  get_possession=+1.13  ball_out=-0.43  lose_possession=-0.34  speed_bonus=+0.34  timeout=-0.09  stamina_penalty=-0.01
2026-08-09 02:15:37,405 INFO   [seeded eval] running 12x8 episodes across 7 worker process(es)...
2026-08-09 02:15:40,621 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 02:15:40,622 INFO Baseline (rules vs rules, 12 trials): trainee_win=83.3%  outcomes={'box_possession': 80, 'other': 8, 'opponent_box_possession': 8}
2026-08-09 02:15:40,622 INFO Frozen decision_net.shoot_logit
2026-08-09 02:15:40,622 INFO Frozen decision_net.pass_logit
2026-08-09 02:15:40,622 INFO Frozen decision_net.tackle_logit
2026-08-09 02:15:40,622 INFO Frozen decision_net.get_possession_raw
2026-08-09 02:15:40,622 INFO Frozen decision_net.mark_logit
2026-08-09 02:15:40,622 INFO Frozen decision_net.hold_position_logit
2026-08-09 02:15:40,622 INFO Frozen decision_net.pass_target_logits
2026-08-09 02:15:40,622 INFO Frozen decision_net.tackle_target_logits
2026-08-09 02:15:40,622 INFO Frozen decision_net.mark_target_logits
2026-08-09 02:15:40,622 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 02:15:40,623 INFO PPO parallel training started: 6 worker(s), ~6000 steps/worker/rollout, steps_so_far=3,312,000  target=9,312,000
2026-08-09 02:15:42,546 INFO Frozen decision_net.shoot_logit
2026-08-09 02:15:42,546 INFO Frozen decision_net.pass_logit
2026-08-09 02:15:42,546 INFO Frozen decision_net.tackle_logit
2026-08-09 02:15:42,546 INFO Frozen decision_net.get_possession_raw
2026-08-09 02:15:42,546 INFO Frozen decision_net.mark_logit
2026-08-09 02:15:42,546 INFO Frozen decision_net.hold_position_logit
2026-08-09 02:15:42,546 INFO Frozen decision_net.pass_target_logits
2026-08-09 02:15:42,546 INFO Frozen decision_net.tackle_target_logits
2026-08-09 02:15:42,547 INFO Frozen decision_net.mark_target_logits
2026-08-09 02:15:42,547 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 02:15:42,559 INFO Frozen decision_net.shoot_logit
2026-08-09 02:15:42,559 INFO Frozen decision_net.pass_logit
2026-08-09 02:15:42,559 INFO Frozen decision_net.tackle_logit
2026-08-09 02:15:42,559 INFO Frozen decision_net.get_possession_raw
2026-08-09 02:15:42,559 INFO Frozen decision_net.mark_logit
2026-08-09 02:15:42,560 INFO Frozen decision_net.hold_position_logit
2026-08-09 02:15:42,560 INFO Frozen decision_net.pass_target_logits
2026-08-09 02:15:42,560 INFO Frozen decision_net.tackle_target_logits
2026-08-09 02:15:42,560 INFO Frozen decision_net.mark_target_logits
2026-08-09 02:15:42,560 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 02:15:42,567 INFO Frozen decision_net.shoot_logit
2026-08-09 02:15:42,567 INFO Frozen decision_net.pass_logit
2026-08-09 02:15:42,567 INFO Frozen decision_net.tackle_logit
2026-08-09 02:15:42,567 INFO Frozen decision_net.get_possession_raw
2026-08-09 02:15:42,567 INFO Frozen decision_net.mark_logit
2026-08-09 02:15:42,567 INFO Frozen decision_net.hold_position_logit
2026-08-09 02:15:42,568 INFO Frozen decision_net.pass_target_logits
2026-08-09 02:15:42,568 INFO Frozen decision_net.tackle_target_logits
2026-08-09 02:15:42,568 INFO Frozen decision_net.mark_target_logits
2026-08-09 02:15:42,568 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 02:15:42,578 INFO Frozen decision_net.shoot_logit
2026-08-09 02:15:42,578 INFO Frozen decision_net.pass_logit
2026-08-09 02:15:42,579 INFO Frozen decision_net.tackle_logit
2026-08-09 02:15:42,579 INFO Frozen decision_net.get_possession_raw
2026-08-09 02:15:42,579 INFO Frozen decision_net.mark_logit
2026-08-09 02:15:42,579 INFO Frozen decision_net.hold_position_logit
2026-08-09 02:15:42,579 INFO Frozen decision_net.pass_target_logits
2026-08-09 02:15:42,579 INFO Frozen decision_net.tackle_target_logits
2026-08-09 02:15:42,579 INFO Frozen decision_net.mark_target_logits
2026-08-09 02:15:42,579 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 02:15:42,603 INFO Frozen decision_net.shoot_logit
2026-08-09 02:15:42,604 INFO Frozen decision_net.pass_logit
2026-08-09 02:15:42,604 INFO Frozen decision_net.tackle_logit
2026-08-09 02:15:42,604 INFO Frozen decision_net.get_possession_raw
2026-08-09 02:15:42,604 INFO Frozen decision_net.mark_logit
2026-08-09 02:15:42,604 INFO Frozen decision_net.hold_position_logit
2026-08-09 02:15:42,605 INFO Frozen decision_net.pass_target_logits
2026-08-09 02:15:42,605 INFO Frozen decision_net.tackle_target_logits
2026-08-09 02:15:42,605 INFO Frozen decision_net.mark_target_logits
2026-08-09 02:15:42,605 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 02:15:42,621 INFO Frozen decision_net.shoot_logit
2026-08-09 02:15:42,622 INFO Frozen decision_net.pass_logit
2026-08-09 02:15:42,622 INFO Frozen decision_net.tackle_logit
2026-08-09 02:15:42,622 INFO Frozen decision_net.get_possession_raw
2026-08-09 02:15:42,622 INFO Frozen decision_net.mark_logit
2026-08-09 02:15:42,622 INFO Frozen decision_net.hold_position_logit
2026-08-09 02:15:42,622 INFO Frozen decision_net.pass_target_logits
2026-08-09 02:15:42,622 INFO Frozen decision_net.tackle_target_logits
2026-08-09 02:15:42,622 INFO Frozen decision_net.mark_target_logits
2026-08-09 02:15:42,622 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 02:29:13,125 INFO Checkpoint dir: checkpoints/phase1_run47
2026-08-09 02:29:13,175 INFO Starting training: phase=phase1_get_possession, total_steps=6,000,000
2026-08-09 02:29:13,175 INFO Phase description: 1v1 scenario: learn to get possession and bring the ball toward the opponent box.  Decision network frozen except Move/GetPossession and latent vector.
2026-08-09 02:29:13,177 INFO _from_ckpt: overriding bc_pretrain_epochs=8 → 3
2026-08-09 02:29:13,177 INFO _from_ckpt: overriding demo_value_pretrain_epochs=6 → 0
2026-08-09 02:29:13,177 INFO _from_ckpt: overriding value_pretrain_epochs=8 → 15
2026-08-09 02:29:13,177 INFO _from_ckpt: overriding aux_coeff_start=0.0 → 0.0
2026-08-09 02:29:13,177 INFO _from_ckpt: overriding aux_coeff_end=0.0 → 0.0
2026-08-09 02:29:13,177 INFO _from_ckpt: overriding aux_coeff_anneal_fraction=0.3 → 0.35
/home/vincent/Documents/not_work/repos/FootballCoach/.venv/lib/python3.12/site-packages/torch/_compile.py:54: UserWarning: optimizer contains a parameter group with duplicate parameters; in future, this will cause an error; see github.com/pytorch/pytorch/issues/40967 for more information
  return disable_fn(*args, **kwargs)
2026-08-09 02:29:14,423 INFO Logging to checkpoints/phase1_run47/training_log1.txt
2026-08-09 02:29:14,423 INFO --latest(-pretrain): resolved to checkpoints/phase1_run45/latest.pt
2026-08-09 02:29:14,610 INFO Loaded checkpoint: checkpoints/phase1_run45/latest.pt (step 3312000)
2026-08-09 02:29:14,611 INFO Loaded checkpoint for re-pretraining: checkpoints/phase1_run45/latest.pt — will still run BC/value pre-training
2026-08-09 02:29:14,623 INFO Loading 1625 demonstration file(s) from demonstrations/phase1
2026-08-09 02:29:21,566 INFO Dataset: 618,198 steps loaded
2026-08-09 02:29:21,567 INFO Offline BC dataset: 618,198 steps from demonstrations/phase1/
2026-08-09 02:29:21,568 INFO Reward diagnostic: running 40 rules-vs-rules episodes...
2026-08-09 02:29:22,518 INFO Reward diagnostic (40 ep, rules vs rules): mean_ep_rew=2.01  per_episode: get_possession=+0.90  lose_possession=-0.16  box_possession=+1.25  speed_bonus=+0.48  opponent_box=-0.45  stamina_penalty=-0.01
2026-08-09 02:29:22,536 INFO BC pos_weight (auto-computed from dataset): kick=1.50  tackle_attempt=1.50
2026-08-09 02:29:22,536 INFO Combined BC + value pre-training: 3 epoch(s), batch_size=800, dataset=618,198 steps, rollout_steps=21000
2026-08-09 02:29:22,544 INFO   BC pretrain split: 465,064 train rows  |  80,612 val rows
2026-08-09 02:29:22,841 INFO   Downsample trivial rows (epoch 1): 162,874/545,676 (29.8%) rows classified trivial, excluding ~130,299 this epoch (frac=0.80)
/home/vincent/Documents/not_work/repos/FootballCoach/src/footballcoach/ai/ppo/bc.py:688: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /__w/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:822.)
  "decision":       float(dec_loss[valid].mean()),
2026-08-09 02:44:33,216 INFO Checkpoint dir: checkpoints/phase1_run48
2026-08-09 02:44:33,272 INFO Starting training: phase=phase1_get_possession, total_steps=6,000,000
2026-08-09 02:44:33,272 INFO Phase description: 1v1 scenario: learn to get possession and bring the ball toward the opponent box.  Decision network frozen except Move/GetPossession and latent vector.
2026-08-09 02:44:33,273 INFO _from_ckpt: overriding bc_pretrain_epochs=8 → 3
2026-08-09 02:44:33,273 INFO _from_ckpt: overriding demo_value_pretrain_epochs=6 → 0
2026-08-09 02:44:33,273 INFO _from_ckpt: overriding value_pretrain_epochs=8 → 15
2026-08-09 02:44:33,273 INFO _from_ckpt: overriding aux_coeff_start=0.0 → 0.0
2026-08-09 02:44:33,273 INFO _from_ckpt: overriding aux_coeff_end=0.0 → 0.0
2026-08-09 02:44:33,273 INFO _from_ckpt: overriding aux_coeff_anneal_fraction=0.3 → 0.35
/home/vincent/Documents/not_work/repos/FootballCoach/.venv/lib/python3.12/site-packages/torch/_compile.py:54: UserWarning: optimizer contains a parameter group with duplicate parameters; in future, this will cause an error; see github.com/pytorch/pytorch/issues/40967 for more information
  return disable_fn(*args, **kwargs)
2026-08-09 02:44:34,411 INFO Logging to checkpoints/phase1_run48/training_log1.txt
2026-08-09 02:44:34,412 INFO --latest(-pretrain): resolved to checkpoints/phase1_run45/latest.pt
2026-08-09 02:44:34,610 INFO Loaded checkpoint: checkpoints/phase1_run45/latest.pt (step 3312000)
2026-08-09 02:44:34,610 INFO Loaded checkpoint for re-pretraining: checkpoints/phase1_run45/latest.pt — will still run BC/value pre-training
2026-08-09 02:44:34,630 INFO Loading 1625 demonstration file(s) from demonstrations/phase1
2026-08-09 02:44:41,197 INFO Dataset: 618,198 steps loaded
2026-08-09 02:44:41,199 INFO Offline BC dataset: 618,198 steps from demonstrations/phase1/
2026-08-09 02:44:41,200 INFO Reward diagnostic: running 40 rules-vs-rules episodes...
2026-08-09 02:44:42,050 INFO Reward diagnostic (40 ep, rules vs rules): mean_ep_rew=2.25  per_episode: get_possession=+1.07  lose_possession=-0.27  ball_out=-0.12  box_possession=+1.50  speed_bonus=+0.50  opponent_box=-0.38  timeout=-0.04  stamina_penalty=-0.01
2026-08-09 02:44:42,068 INFO BC pos_weight (auto-computed from dataset): kick=1.50  tackle_attempt=1.50
2026-08-09 02:44:42,068 INFO Combined BC + value pre-training: 3 epoch(s), batch_size=800, dataset=618,198 steps, rollout_steps=21000
2026-08-09 02:44:42,076 INFO   BC pretrain split: 465,064 train rows  |  80,612 val rows
2026-08-09 02:44:42,364 INFO   Downsample trivial rows (epoch 1): 162,874/545,676 (29.8%) rows classified trivial, excluding ~130,299 this epoch (frac=0.80)
/home/vincent/Documents/not_work/repos/FootballCoach/src/footballcoach/ai/ppo/bc.py:688: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /__w/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:822.)
  "decision":       float(dec_loss[valid].mean()),
2026-08-09 02:47:27,471 INFO   BC epoch 1/3  (165.4s)
    loss       bc=2.3475  bc_adj=0.2809(floor=2.0665)
    heads      dir_cos=0.974  kick_dir_cos=0.990
               move_prob=0.887  sprint_prob=0.814  kick_prob=0.068  tackle_prob=0.261
    pr/rec     kick:   p=0.970  r=0.961  f1=0.965  (tp=39091 fp=1212 fn=1585)
               tackle: p=0.966  r=0.988  f1=0.977  (tp=330452 fp=11549 fn=4100)
    breakdown  decision=0.701  exec_bce=0.856  sprint=0.246  move=0.219  tackle_attempt=0.213  direction=0.075
               region=0.014  kick=0.17731  kick_direction=0.02873  kick_power=0.00240  kick_spin=0.00000
2026-08-09 02:47:29,869 INFO     val        bc_val_loss=2.3091  best=2.3091  (improved)
2026-08-09 02:47:29,881 INFO   Downsample trivial rows (epoch 2): 162,874/545,676 (29.8%) rows classified trivial, excluding ~130,299 this epoch (frac=0.80)
2026-08-09 02:50:03,373 INFO   BC epoch 2/3  (153.5s)
    loss       bc=2.2950  bc_adj=0.2285(floor=2.0665)
    heads      dir_cos=0.975  kick_dir_cos=0.993
               move_prob=0.887  sprint_prob=0.814  kick_prob=0.067  tackle_prob=0.260
    pr/rec     kick:   p=0.975  r=0.970  f1=0.972  (tp=39447 fp=1020 fn=1229)
               tackle: p=0.972  r=0.988  f1=0.980  (tp=330677 fp=9578 fn=3875)
    breakdown  decision=0.687  exec_bce=0.841  sprint=0.241  move=0.215  tackle_attempt=0.210  direction=0.070
               region=0.010  kick=0.17484  kick_direction=0.02091  kick_power=0.00197  kick_spin=0.00000
2026-08-09 02:50:06,079 INFO     val        bc_val_loss=2.2437  best=2.2437  (improved)
2026-08-09 02:50:06,090 INFO   Downsample trivial rows (epoch 3): 162,874/545,676 (29.8%) rows classified trivial, excluding ~130,299 this epoch (frac=0.80)
2026-08-09 02:52:41,119 INFO   BC epoch 3/3  (155.0s)
    loss       bc=2.2850  bc_adj=0.2184(floor=2.0666)
    heads      dir_cos=0.976  kick_dir_cos=0.994
               move_prob=0.887  sprint_prob=0.814  kick_prob=0.067  tackle_prob=0.260
    pr/rec     kick:   p=0.978  r=0.976  f1=0.977  (tp=39684 fp=884 fn=992)
               tackle: p=0.973  r=0.989  f1=0.981  (tp=330906 fp=9044 fn=3646)
    breakdown  decision=0.687  exec_bce=0.833  sprint=0.238  move=0.213  tackle_attempt=0.208  direction=0.069
               region=0.010  kick=0.17402  kick_direction=0.01798  kick_power=0.00186  kick_spin=0.00000
2026-08-09 02:52:44,171 INFO     val        bc_val_loss=2.2605  best=2.2437  (patience 1/2)
2026-08-09 02:52:44,172 INFO BC pre-training done (3 epoch(s), final bc_loss=2.2850)
2026-08-09 02:52:44,172 INFO Value pre-training: 21000 steps, 15 epochs, lr=3e-05
2026-08-09 02:52:44,173 INFO   [value pretrain rollout] parallel collection: 6 worker(s), ~3500 steps/worker
2026-08-09 02:52:46,209 INFO Frozen decision_net.shoot_logit
2026-08-09 02:52:46,209 INFO Frozen decision_net.pass_logit
2026-08-09 02:52:46,210 INFO Frozen decision_net.tackle_logit
2026-08-09 02:52:46,210 INFO Frozen decision_net.get_possession_raw
2026-08-09 02:52:46,210 INFO Frozen decision_net.mark_logit
2026-08-09 02:52:46,210 INFO Frozen decision_net.hold_position_logit
2026-08-09 02:52:46,210 INFO Frozen decision_net.pass_target_logits
2026-08-09 02:52:46,210 INFO Frozen decision_net.tackle_target_logits
2026-08-09 02:52:46,210 INFO Frozen decision_net.mark_target_logits
2026-08-09 02:52:46,210 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 02:52:46,227 INFO Frozen decision_net.shoot_logit
2026-08-09 02:52:46,228 INFO Frozen decision_net.pass_logit
2026-08-09 02:52:46,228 INFO Frozen decision_net.tackle_logit
2026-08-09 02:52:46,228 INFO Frozen decision_net.get_possession_raw
2026-08-09 02:52:46,228 INFO Frozen decision_net.mark_logit
2026-08-09 02:52:46,228 INFO Frozen decision_net.hold_position_logit
2026-08-09 02:52:46,228 INFO Frozen decision_net.pass_target_logits
2026-08-09 02:52:46,228 INFO Frozen decision_net.tackle_target_logits
2026-08-09 02:52:46,228 INFO Frozen decision_net.mark_target_logits
2026-08-09 02:52:46,228 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 02:52:46,229 INFO Frozen decision_net.shoot_logit
2026-08-09 02:52:46,230 INFO Frozen decision_net.pass_logit
2026-08-09 02:52:46,230 INFO Frozen decision_net.tackle_logit
2026-08-09 02:52:46,230 INFO Frozen decision_net.get_possession_raw
2026-08-09 02:52:46,230 INFO Frozen decision_net.mark_logit
2026-08-09 02:52:46,230 INFO Frozen decision_net.hold_position_logit
2026-08-09 02:52:46,230 INFO Frozen decision_net.pass_target_logits
2026-08-09 02:52:46,230 INFO Frozen decision_net.tackle_target_logits
2026-08-09 02:52:46,230 INFO Frozen decision_net.mark_target_logits
2026-08-09 02:52:46,230 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 02:52:46,324 INFO Frozen decision_net.shoot_logit
2026-08-09 02:52:46,324 INFO Frozen decision_net.pass_logit
2026-08-09 02:52:46,324 INFO Frozen decision_net.tackle_logit
2026-08-09 02:52:46,324 INFO Frozen decision_net.get_possession_raw
2026-08-09 02:52:46,324 INFO Frozen decision_net.mark_logit
2026-08-09 02:52:46,324 INFO Frozen decision_net.hold_position_logit
2026-08-09 02:52:46,324 INFO Frozen decision_net.pass_target_logits
2026-08-09 02:52:46,324 INFO Frozen decision_net.tackle_target_logits
2026-08-09 02:52:46,324 INFO Frozen decision_net.mark_target_logits
2026-08-09 02:52:46,324 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 02:52:46,344 INFO Frozen decision_net.shoot_logit
2026-08-09 02:52:46,344 INFO Frozen decision_net.pass_logit
2026-08-09 02:52:46,344 INFO Frozen decision_net.tackle_logit
2026-08-09 02:52:46,344 INFO Frozen decision_net.get_possession_raw
2026-08-09 02:52:46,344 INFO Frozen decision_net.mark_logit
2026-08-09 02:52:46,344 INFO Frozen decision_net.hold_position_logit
2026-08-09 02:52:46,344 INFO Frozen decision_net.pass_target_logits
2026-08-09 02:52:46,344 INFO Frozen decision_net.tackle_target_logits
2026-08-09 02:52:46,344 INFO Frozen decision_net.mark_target_logits
2026-08-09 02:52:46,344 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 02:52:46,384 INFO Frozen decision_net.shoot_logit
2026-08-09 02:52:46,384 INFO Frozen decision_net.pass_logit
2026-08-09 02:52:46,384 INFO Frozen decision_net.tackle_logit
2026-08-09 02:52:46,384 INFO Frozen decision_net.get_possession_raw
2026-08-09 02:52:46,384 INFO Frozen decision_net.mark_logit
2026-08-09 02:52:46,384 INFO Frozen decision_net.hold_position_logit
2026-08-09 02:52:46,384 INFO Frozen decision_net.pass_target_logits
2026-08-09 02:52:46,384 INFO Frozen decision_net.tackle_target_logits
2026-08-09 02:52:46,384 INFO Frozen decision_net.mark_target_logits
2026-08-09 02:52:46,384 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 02:53:18,613 INFO   [value pretrain rollout] dropped 154 trailing (incomplete-episode) step(s) across workers before MC-return fit
2026-08-09 02:53:18,629 INFO   [value pretrain rollout] mean_return=2.76 (335 episode(s))  vs[win/loss/tout/miss]  vs_rules(0): n/a  vs_immobile(335): 82.1%/0.0%/6.3%/11.6%  vs_neural(0): n/a
2026-08-09 02:53:18,630 INFO   [value pretrain rollout] ep_len 18.6±13.0s  (n=335, min=1.7s, max=50.1s)
2026-08-09 02:53:18,630 INFO   [value pretrain rollout] rew/ep (mean/std/min/max per episode, 335 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.928    0.301    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.011    0.098    -0.900    +0.000
  ball_out          -0.194    0.966    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.642    0.767    +0.000    +2.000
  speed_bonus       +0.497    0.396    +0.000    +1.213
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.094    0.364    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.025    +0.000
2026-08-09 02:53:18,742 INFO   Value pretrain split: 285 train eps (17248 steps)  |  50 val eps (3598 steps)
2026-08-09 02:53:22,977 INFO   Value epoch 1/15: train=1.3058 rmse=1.93  val=0.7192 val_rmse=1.43 (std=1.7)
    V(train)=+1.673  R(train)=+1.697  |  V(val)=+1.702  R(val)=+1.718
2026-08-09 02:53:27,180 INFO   Value epoch 2/15: train=0.9445 rmse=1.64  val=0.6998 val_rmse=1.41 (std=1.7)
    V(train)=+1.710  R(train)=+1.697  |  V(val)=+1.738  R(val)=+1.718
2026-08-09 02:53:31,343 INFO   Value epoch 3/15: train=0.9167 rmse=1.62  val=0.6967 val_rmse=1.41 (std=1.7)
    V(train)=+1.711  R(train)=+1.696  |  V(val)=+1.684  R(val)=+1.718
2026-08-09 02:53:35,764 INFO   Value epoch 4/15: train=0.8985 rmse=1.60  val=0.6974 val_rmse=1.41 (std=1.7)
    V(train)=+1.701  R(train)=+1.696  |  V(val)=+1.716  R(val)=+1.718
2026-08-09 02:53:40,041 INFO   Value epoch 5/15: train=0.8861 rmse=1.59  val=0.7002 val_rmse=1.41 (std=1.7)
    V(train)=+1.710  R(train)=+1.699  |  V(val)=+1.730  R(val)=+1.718
2026-08-09 02:53:44,337 INFO   Value epoch 6/15: train=0.8766 rmse=1.58  val=0.7021 val_rmse=1.41 (std=1.7)
    V(train)=+1.708  R(train)=+1.698  |  V(val)=+1.677  R(val)=+1.718
2026-08-09 02:53:48,528 INFO   Value epoch 7/15: train=0.8672 rmse=1.57  val=0.7080 val_rmse=1.42 (std=1.7)
    V(train)=+1.711  R(train)=+1.696  |  V(val)=+1.691  R(val)=+1.718
2026-08-09 02:53:48,528 INFO   [value pretrain] early stop at epoch 7 (val stagnant for 4 epochs, best=0.6967)
2026-08-09 02:53:48,529 INFO   [value pretrain] restored best-val weights (val_loss=0.6967)
2026-08-09 02:53:48,530 INFO Value pre-training done (7 epoch(s), final train_loss=0.8672)
2026-08-09 02:54:05,188 INFO BC check after value warm-up: bc_loss=2.2635 (before=2.2850, delta=-0.0215)  OK
2026-08-09 02:54:05,188 INFO Combined pre-training complete.
2026-08-09 02:54:39,885 INFO Pre-PPO eval (rules opp): win=41.4%  mean_rew=-0.126  V=1.253  R=-0.277  gap=+1.530  outcomes={'box_possession': 53, 'opponent_box_possession': 65, 'miss': 10}
2026-08-09 02:54:39,886 INFO   rew breakdown (rules, per ep): opponent_box=-1.52  get_possession=+0.88  box_possession=+0.83  lose_possession=-0.41  speed_bonus=+0.19  ball_out=-0.08  stamina_penalty=-0.02
2026-08-09 02:55:16,059 INFO Pre-PPO eval (immobile opp): win=75.8%  mean_rew=2.401  V=1.587  R=1.176  gap=+0.411  outcomes={'box_possession': 97, 'miss': 28, 'timeout': 3}
2026-08-09 02:55:16,059 INFO   rew breakdown (immobile, per ep): box_possession=+1.52  get_possession=+0.85  speed_bonus=+0.36  ball_out=-0.27  timeout=-0.04  lose_possession=-0.01
2026-08-09 02:56:35,176 INFO Pre-PPO eval (self-play):   win=71.1%  mean_rew=1.941  V=1.025  R=0.992  gap=+0.033  outcomes={'box_possession': 91, 'miss': 18, 'opponent_box_possession': 18, 'timeout': 1}
2026-08-09 02:56:35,176 INFO   rew breakdown (self-play, per ep): opponent_box=-2.55  box_possession=+1.70  get_possession=+1.61  lose_possession=-0.65  speed_bonus=+0.38  ball_out=-0.20  stamina_penalty=-0.03  timeout=-0.02
2026-08-09 02:56:35,176 INFO   [seeded eval] running 12x8 episodes across 7 worker process(es)...
2026-08-09 02:56:38,331 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 02:56:38,332 INFO Baseline (rules vs rules, 12 trials): trainee_win=91.7%  outcomes={'box_possession': 88, 'opponent_box_possession': 8}
2026-08-09 02:56:38,333 INFO Frozen decision_net.shoot_logit
2026-08-09 02:56:38,333 INFO Frozen decision_net.pass_logit
2026-08-09 02:56:38,333 INFO Frozen decision_net.tackle_logit
2026-08-09 02:56:38,333 INFO Frozen decision_net.get_possession_raw
2026-08-09 02:56:38,333 INFO Frozen decision_net.mark_logit
2026-08-09 02:56:38,333 INFO Frozen decision_net.hold_position_logit
2026-08-09 02:56:38,333 INFO Frozen decision_net.pass_target_logits
2026-08-09 02:56:38,333 INFO Frozen decision_net.tackle_target_logits
2026-08-09 02:56:38,333 INFO Frozen decision_net.mark_target_logits
2026-08-09 02:56:38,333 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 02:56:38,333 INFO PPO parallel training started: 6 worker(s), ~6000 steps/worker/rollout, steps_so_far=0  target=6,000,000
2026-08-09 02:56:40,168 INFO Frozen decision_net.shoot_logit
2026-08-09 02:56:40,168 INFO Frozen decision_net.pass_logit
2026-08-09 02:56:40,168 INFO Frozen decision_net.tackle_logit
2026-08-09 02:56:40,168 INFO Frozen decision_net.get_possession_raw
2026-08-09 02:56:40,168 INFO Frozen decision_net.mark_logit
2026-08-09 02:56:40,168 INFO Frozen decision_net.hold_position_logit
2026-08-09 02:56:40,168 INFO Frozen decision_net.pass_target_logits
2026-08-09 02:56:40,168 INFO Frozen decision_net.tackle_target_logits
2026-08-09 02:56:40,168 INFO Frozen decision_net.mark_target_logits
2026-08-09 02:56:40,168 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 02:56:40,174 INFO Frozen decision_net.shoot_logit
2026-08-09 02:56:40,174 INFO Frozen decision_net.pass_logit
2026-08-09 02:56:40,174 INFO Frozen decision_net.tackle_logit
2026-08-09 02:56:40,174 INFO Frozen decision_net.get_possession_raw
2026-08-09 02:56:40,174 INFO Frozen decision_net.mark_logit
2026-08-09 02:56:40,174 INFO Frozen decision_net.hold_position_logit
2026-08-09 02:56:40,174 INFO Frozen decision_net.pass_target_logits
2026-08-09 02:56:40,174 INFO Frozen decision_net.tackle_target_logits
2026-08-09 02:56:40,175 INFO Frozen decision_net.mark_target_logits
2026-08-09 02:56:40,175 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 02:56:40,215 INFO Frozen decision_net.shoot_logit
2026-08-09 02:56:40,215 INFO Frozen decision_net.pass_logit
2026-08-09 02:56:40,215 INFO Frozen decision_net.tackle_logit
2026-08-09 02:56:40,215 INFO Frozen decision_net.get_possession_raw
2026-08-09 02:56:40,215 INFO Frozen decision_net.mark_logit
2026-08-09 02:56:40,215 INFO Frozen decision_net.hold_position_logit
2026-08-09 02:56:40,215 INFO Frozen decision_net.pass_target_logits
2026-08-09 02:56:40,215 INFO Frozen decision_net.tackle_target_logits
2026-08-09 02:56:40,215 INFO Frozen decision_net.mark_target_logits
2026-08-09 02:56:40,215 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 02:56:40,250 INFO Frozen decision_net.shoot_logit
2026-08-09 02:56:40,250 INFO Frozen decision_net.pass_logit
2026-08-09 02:56:40,250 INFO Frozen decision_net.tackle_logit
2026-08-09 02:56:40,250 INFO Frozen decision_net.get_possession_raw
2026-08-09 02:56:40,251 INFO Frozen decision_net.mark_logit
2026-08-09 02:56:40,251 INFO Frozen decision_net.hold_position_logit
2026-08-09 02:56:40,251 INFO Frozen decision_net.pass_target_logits
2026-08-09 02:56:40,251 INFO Frozen decision_net.tackle_target_logits
2026-08-09 02:56:40,251 INFO Frozen decision_net.mark_target_logits
2026-08-09 02:56:40,251 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 02:56:40,259 INFO Frozen decision_net.shoot_logit
2026-08-09 02:56:40,259 INFO Frozen decision_net.pass_logit
2026-08-09 02:56:40,259 INFO Frozen decision_net.tackle_logit
2026-08-09 02:56:40,259 INFO Frozen decision_net.get_possession_raw
2026-08-09 02:56:40,259 INFO Frozen decision_net.mark_logit
2026-08-09 02:56:40,259 INFO Frozen decision_net.hold_position_logit
2026-08-09 02:56:40,259 INFO Frozen decision_net.pass_target_logits
2026-08-09 02:56:40,259 INFO Frozen decision_net.tackle_target_logits
2026-08-09 02:56:40,259 INFO Frozen decision_net.mark_target_logits
2026-08-09 02:56:40,259 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 02:56:40,290 INFO Frozen decision_net.shoot_logit
2026-08-09 02:56:40,290 INFO Frozen decision_net.pass_logit
2026-08-09 02:56:40,290 INFO Frozen decision_net.tackle_logit
2026-08-09 02:56:40,290 INFO Frozen decision_net.get_possession_raw
2026-08-09 02:56:40,290 INFO Frozen decision_net.mark_logit
2026-08-09 02:56:40,290 INFO Frozen decision_net.hold_position_logit
2026-08-09 02:56:40,290 INFO Frozen decision_net.pass_target_logits
2026-08-09 02:56:40,290 INFO Frozen decision_net.tackle_target_logits
2026-08-09 02:56:40,290 INFO Frozen decision_net.mark_target_logits
2026-08-09 02:56:40,290 WARNING PPO log_prob masking ACTIVE — the following decision heads are excluded from the importance ratio (frozen for this curriculum phase, no reward signal): get_possession_raw, hold_position_logit, mark_logit, pass_logit, shoot_logit, tackle_logit.  Their BC aux loss is still computed normally.
2026-08-09 03:02:26,166 INFO   [KL mean=0.0656 median=0.0651 > 0.05] ratio percentiles:  p5=0.669  p25=0.913  p50=0.966  p75=1.003  p95=1.247  max=16.315
  move_dir_log_std=[-1.573224425315857]  kick_dir_log_std=[-1.5690257549285889]
2026-08-09 03:02:26,180 INFO   [per-head new lp means, n=256]
    shoot=0.000  pass=0.000  move=-0.070  tackle=0.000  gp=0.000  mark=0.000  hold=0.000
    sprint=-0.205  kick=-0.205  t_att=-0.160
    move_dir=0.562 (min=-11.644 max=1.309)  kick_dir=0.049 (min=-1.355 max=1.854)
  [head lp deltas (new-old, |d|>0.05)] exec_move:+0.15
  [worst sample] idx=48  ratio=39.753  adv=-3.842  old_lp=-3.823  new_lp=-0.140
    stored move_dir=115.3°  new_mean=115.1°  angular_diff=0.2°
    [worst sample per-head delta, sorted by |delta|] 
  [top-2 highest-ratio samples]
    idx=  48  ratio=  39.753  adv=-3.842  lp: old=-3.823  new=-0.140
      rew=+0.0000  ret=-4.7057  val=-0.8640  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9677  sprint_p_new=0.0639  kick_p_new=0.0544  tackle_attempt_p_new=0.0568
    idx= 132  ratio=  24.968  adv=+1.185  lp: old=-3.349  new=-0.131
      rew=+0.0000  ret=+3.1502  val=+1.9650  outcome=mid-ep
      rew_breakdown: n/a
      head_deltas: 
      saturation: exec_move_p_new=0.9476  sprint_p_new=0.0149  kick_p_new=0.0376  tackle_attempt_p_new=0.0594
  [best sample (highest new_lp)] idx=168  new_lp=-0.052  adv=+0.688  stored move_dir=2.3°  new_mean=2.3°
    per-head contributions: move_dir:0.065  move:-0.029  kick:-0.036  tackle_attempt:-0.041
2026-08-09 03:02:26,181 INFO   [advantage] mean=-0.000  std=1.000  min=-8.095  max=3.459
2026-08-09 03:02:26,182 INFO   [ratio] mean=0.9674  std=0.2470  min=0.0008  max=16.3145  clipped=21.9%
2026-08-09 03:02:26,182 INFO   [exec head grad norm] move_direction=0.054  exec_move=0.089  sprint=0.043  kick=0.049  kick_direction=0.007  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.059
2026-08-09 03:02:26,182 INFO   [exec continuous log_std] move_direction: start=-1.5746 end=-1.5732   kick_direction: start=-1.5693 end=-1.5690
2026-08-09 03:02:26,182 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0029≈0.17°/step  epoch≈9.9°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0030≈0.17°/step  epoch≈10.2°  dlog_std=0.00000  Δσ°=0.000/step)
2026-08-09 03:02:26,182 INFO   [exec discrete Δlogit per opt step] exec_move=0.0048  sprint=0.0045  kick=0.0030  tackle_attempt=0.0033
2026-08-09 03:02:26,182 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0006  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0079  sprint=+0.0079  kick=+0.0040  tackle_attempt=+0.0009  move_dir=+0.0380  kick_dir=+0.0064
2026-08-09 03:02:26,183 INFO   [grad clip] main: 10/60 steps clipped (17%)  pre-clip norm mean=0.296 max=0.718  limit=0.4
              direction: 60/60 steps clipped (100%)  pre-clip norm mean=0.060 max=0.211  limit=0.02
2026-08-09 03:02:26,225 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=36,000  speed=673/s  reward=2.38
  loss     policy=0.0216  value=0.7868(x0.5)=0.3934
           entropy=1.5413  kl=0.0656
  value    V=1.66±0.62  R=1.76±1.05  adv=0.09±0.95
  moves    mv_ls=[-1.5732] (σ≈0.21, ≈12°) g=2.04e-02
           kk_ls=[-1.5690] (σ≈0.21, ≈12°)
  heads    move= 30 get_poss= 70 exec_move= 93 sprint= 37 kick=  4 tackle=  5 shoot=
           2 hold=  2 tackle_prob=0.0512 kick_prob=0.0434
  vs       vs[win/loss/tout/miss]  vs_immobile(592): 80.6%/0.0%/5.4%/14.0%
  ep_len   18.1±12.8s  (n=592, min=1.8s, max=50.1s)
  reward   get_possession=+535.00  lose_possession=-0.90  ball_out=-120.00  box_possession=+954.00
           speed_bonus=+264.96  timeout=-48.00  stamina_penalty=-3.24
  rew/ep   (mean/std/min/max per episode, 592 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.904    0.301    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.002    0.037    -0.900    +0.000
  ball_out          -0.203    0.986    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.611    0.791    +0.000    +2.000
  speed_bonus       +0.448    0.403    +0.000    +1.220
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.081    0.339    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.005    0.006    -0.025    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     539    +0.015    0.121     +2.772     1.042     +1.366      2.9491      1.584     2.663
  lose_possession       1    -0.000    0.005     +1.729     0.000     -0.981      0.9622      0.981     0.981
  ball_out            24    -0.003    0.129     -4.958     0.200     -5.288     28.5364      5.288     6.557
  box_possession     477    +0.026    0.229     +2.618     0.387     +0.342      0.7310      0.629     1.958
  speed_bonus        437    +0.007    0.077     +2.640     0.369     +0.327      0.6866      0.606     1.876
  timeout             32    -0.001    0.045     -1.500     0.002     -3.027      9.5133      3.027     4.280
  stamina_penalty     396    -0.000    0.001     +2.614     0.517     +0.291      0.7682      0.619     1.927
  gae/td   mean_return=+1.756  std_return=1.052  mean_gae=+0.093  mean_sq_td=0.9026
──────────────────────────────────────────────────────────────────────
2026-08-09 03:02:26,246 INFO Saved checkpoint: checkpoints/phase1_run48/checkpoint1.pt
2026-08-09 03:02:26,246 INFO Logging to checkpoints/phase1_run48/training_log2.txt
2026-08-09 03:02:26,248 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 03:02:40,654 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 03:02:40,655 INFO   [eval vs immobile] step=36,000  seeds=16x8  win=78%  mean_rew=2.482±2.115  V=1.727  gap=-0.755  outcomes={'box_possession': 100, 'miss': 27, 'timeout': 1}
2026-08-09 03:02:40,657 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 03:02:51,199 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 03:02:51,200 INFO   [eval vs rules] step=36,000  seeds=16x8  win=37%  mean_rew=-0.434±3.018  V=1.278  gap=+1.712  outcomes={'box_possession': 47, 'opponent_box_possession': 71, 'miss': 9, 'timeout': 1}
2026-08-09 03:08:10,818 INFO   [advantage] mean=-0.000  std=1.000  min=-6.942  max=3.266
2026-08-09 03:08:10,820 INFO   [ratio] mean=0.9823  std=0.2147  min=0.0074  max=18.7313  clipped=19.0%
2026-08-09 03:08:10,820 INFO   [exec head grad norm] move_direction=0.022  exec_move=0.072  sprint=0.044  kick=0.048  kick_direction=0.006  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.069
2026-08-09 03:08:10,820 INFO   [exec continuous log_std] move_direction: start=-1.5732 end=-1.5721   kick_direction: start=-1.5690 end=-1.5687
2026-08-09 03:08:10,820 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0010≈0.05°/step  epoch≈3.3°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0011≈0.06°/step  epoch≈3.8°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 03:08:10,820 INFO   [exec discrete Δlogit per opt step] exec_move=0.0038  sprint=0.0039  kick=0.0027  tackle_attempt=0.0036
2026-08-09 03:08:10,821 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0009  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0050  sprint=+0.0077  kick=+0.0024  tackle_attempt=+0.0014  move_dir=+0.0167  kick_dir=+0.0053
2026-08-09 03:08:10,821 INFO   [grad clip] main: 3/60 steps clipped (5%)  pre-clip norm mean=0.262 max=0.449  limit=0.4
              direction: 41/60 steps clipped (68%)  pre-clip norm mean=0.025 max=0.048  limit=0.02
2026-08-09 03:08:10,876 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=72,000  speed=739/s  reward=2.48
  loss     policy=0.0132  value=0.7465(x0.5)=0.3733
           entropy=1.8193  kl=0.0395
  value    V=1.83±0.49  R=1.91±1.11  adv=0.08±0.98
  moves    mv_ls=[-1.5721] (σ≈0.21, ≈12°) g=8.72e-03  d_move=[+0.0011] (Δσ≈0.013°)
           kk_ls=[-1.5687] (σ≈0.21, ≈12°)  d_kick=[+0.0003] (Δσ≈0.004°)
  heads    move= 33 get_poss= 67 exec_move= 91 sprint= 41 kick=  5 tackle=  5 shoot=
           2 hold=  3 tackle_prob=0.0615 kick_prob=0.0482
  vs       vs[win/loss/tout/miss]  vs_immobile(659): 82.1%/0.0%/1.8%/16.1%
  ep_len   16.2±11.2s  (n=659, min=1.5s, max=50.1s)
  reward   get_possession=+590.00  lose_possession=-4.50  ball_out=-175.00  box_possession=+1082.00
           speed_bonus=+325.55  timeout=-18.00  stamina_penalty=-3.72
  rew/ep   (mean/std/min/max per episode, 659 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.895    0.330    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.007    0.078    -0.900    +0.000
  ball_out          -0.266    1.121    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.642    0.767    +0.000    +2.000
  speed_bonus       +0.494    0.414    +0.000    +1.272
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.027    0.201    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.026    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     591    +0.016    0.127     +2.862     0.895     +1.108      2.0325      1.301     2.143
  lose_possession       5    -0.000    0.011     +1.879     0.216     -0.125      0.2338      0.476     0.600
  ball_out            35    -0.005    0.156     -4.971     0.167     -5.448     30.1446      5.448     6.517
  box_possession     541    +0.030    0.243     +2.684     0.373     +0.438      0.5822      0.582     1.511
  speed_bonus        492    +0.009    0.087     +2.697     0.355     +0.413      0.5275      0.553     1.450
  timeout             12    -0.001    0.027     -1.501     0.002     -3.096      9.7157      3.096     3.527
  stamina_penalty     462    -0.000    0.001     +2.699     0.440     +0.414      0.5678      0.561     1.434
  gae/td   mean_return=+1.908  std_return=1.106  mean_gae=+0.079  mean_sq_td=0.9620
──────────────────────────────────────────────────────────────────────
2026-08-09 03:08:10,899 INFO Saved checkpoint: checkpoints/phase1_run48/checkpoint2.pt
2026-08-09 03:08:10,900 INFO Logging to checkpoints/phase1_run48/training_log3.txt
2026-08-09 03:08:10,901 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 03:08:24,443 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 03:08:24,445 INFO   [eval vs immobile] step=72,000  seeds=16x8  win=76%  mean_rew=2.293±2.312  V=1.805  gap=-0.488  outcomes={'box_possession': 97, 'timeout': 2, 'miss': 29}
2026-08-09 03:08:24,446 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 03:08:34,811 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 03:08:34,813 INFO   [eval vs rules] step=72,000  seeds=16x8  win=34%  mean_rew=-0.617±3.035  V=1.230  gap=+1.846  outcomes={'box_possession': 44, 'opponent_box_possession': 72, 'miss': 12}
2026-08-09 03:14:12,528 INFO   [advantage] mean=-0.000  std=1.000  min=-6.935  max=3.189
2026-08-09 03:14:12,529 INFO   [ratio] mean=0.9850  std=0.2251  min=0.0079  max=17.0198  clipped=18.7%
2026-08-09 03:14:12,529 INFO   [exec head grad norm] move_direction=0.020  exec_move=0.061  sprint=0.046  kick=0.057  kick_direction=0.007  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.068
2026-08-09 03:14:12,530 INFO   [exec continuous log_std] move_direction: start=-1.5721 end=-1.5711   kick_direction: start=-1.5687 end=-1.5683
2026-08-09 03:14:12,530 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0005≈0.03°/step  epoch≈1.9°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0008≈0.05°/step  epoch≈2.8°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 03:14:12,530 INFO   [exec discrete Δlogit per opt step] exec_move=0.0030  sprint=0.0034  kick=0.0028  tackle_attempt=0.0030
2026-08-09 03:14:12,530 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0007  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0046  sprint=+0.0073  kick=+0.0017  tackle_attempt=+0.0014  move_dir=+0.0150  kick_dir=+0.0053
2026-08-09 03:14:12,531 INFO   [grad clip] main: 3/60 steps clipped (5%)  pre-clip norm mean=0.258 max=0.434  limit=0.4
              direction: 39/60 steps clipped (65%)  pre-clip norm mean=0.023 max=0.046  limit=0.02
2026-08-09 03:14:12,586 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=108,000  speed=763/s  reward=2.92
  loss     policy=0.0126  value=0.7215(x0.5)=0.3607
           entropy=2.0952  kl=0.0360
  value    V=1.88±0.59  R=1.93±1.12  adv=0.05±0.96
  moves    mv_ls=[-1.5711] (σ≈0.21, ≈12°) g=8.12e-03  d_move=[+0.0010] (Δσ≈0.012°)
           kk_ls=[-1.5683] (σ≈0.21, ≈12°)  d_kick=[+0.0004] (Δσ≈0.005°)
  heads    move= 33 get_poss= 67 exec_move= 90 sprint= 40 kick=  5 tackle=  7 shoot=
           3 hold=  3 tackle_prob=0.0714 kick_prob=0.0552
  vs       vs[win/loss/tout/miss]  vs_immobile(645): 82.6%/0.0%/2.2%/15.2%
  ep_len   16.7±10.8s  (n=645, min=1.4s, max=50.1s)
  reward   get_possession=+593.00  lose_possession=-7.20  ball_out=-175.00  box_possession=+1066.00
           speed_bonus=+314.46  timeout=-21.00  stamina_penalty=-3.39
  rew/ep   (mean/std/min/max per episode, 645 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.919    0.315    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.011    0.100    -0.900    +0.000
  ball_out          -0.271    1.133    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.653    0.758    +0.000    +2.000
  speed_bonus       +0.488    0.392    -0.001    +1.190
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.033    0.219    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.005    0.006    -0.023    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     595    +0.017    0.127     +2.811     0.984     +0.865      1.7616      1.148     2.178
  lose_possession       8    -0.000    0.013     +2.076     0.247     +0.027      0.1356      0.315     0.631
  ball_out            35    -0.005    0.156     -4.914     0.280     -4.354     20.1143      4.354     6.044
  box_possession     533    +0.030    0.242     +2.683     0.361     +0.607      0.8906      0.714     2.030
  speed_bonus        481    +0.009    0.083     +2.689     0.345     +0.596      0.8580      0.691     2.037
  timeout             14    -0.001    0.030     -1.502     0.006     -2.889      9.2715      2.940     3.656
  stamina_penalty     447    -0.000    0.001     +2.693     0.389     +0.577      0.8463      0.696     2.022
  gae/td   mean_return=+1.933  std_return=1.124  mean_gae=+0.050  mean_sq_td=0.9194
──────────────────────────────────────────────────────────────────────
2026-08-09 03:14:12,610 INFO Saved checkpoint: checkpoints/phase1_run48/checkpoint3.pt
2026-08-09 03:14:12,610 INFO Logging to checkpoints/phase1_run48/training_log4.txt
2026-08-09 03:14:12,611 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 03:14:26,634 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 03:14:26,636 INFO   [eval vs immobile] step=108,000  seeds=16x8  win=75%  mean_rew=2.228±2.391  V=1.857  gap=-0.371  outcomes={'box_possession': 96, 'miss': 30, 'timeout': 2}
2026-08-09 03:14:26,637 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 03:14:37,914 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 03:14:37,915 INFO   [eval vs rules] step=108,000  seeds=16x8  win=31%  mean_rew=-0.727±2.948  V=1.239  gap=+1.966  outcomes={'box_possession': 40, 'opponent_box_possession': 77, 'miss': 11}
2026-08-09 03:20:29,858 INFO   [advantage] mean=-0.000  std=1.000  min=-7.097  max=3.748
2026-08-09 03:20:29,859 INFO   [ratio] mean=0.9846  std=0.2044  min=0.0045  max=24.2225  clipped=19.4%
2026-08-09 03:20:29,859 INFO   [exec head grad norm] move_direction=0.018  exec_move=0.061  sprint=0.042  kick=0.065  kick_direction=0.006  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.068
2026-08-09 03:20:29,860 INFO   [exec continuous log_std] move_direction: start=-1.5711 end=-1.5701   kick_direction: start=-1.5683 end=-1.5678
2026-08-09 03:20:29,860 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0006≈0.03°/step  epoch≈2.0°  dlog_std=0.00002  Δσ°=0.000/step)  kick_direction(dmean=0.0008≈0.05°/step  epoch≈2.8°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 03:20:29,860 INFO   [exec discrete Δlogit per opt step] exec_move=0.0025  sprint=0.0031  kick=0.0029  tackle_attempt=0.0030
2026-08-09 03:20:29,860 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0007  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0042  sprint=+0.0072  kick=+0.0017  tackle_attempt=+0.0016  move_dir=+0.0129  kick_dir=+0.0063
2026-08-09 03:20:29,860 INFO   [grad clip] main: 0/60 steps clipped (0%)  pre-clip norm mean=0.264 max=0.391  limit=0.4
              direction: 30/60 steps clipped (50%)  pre-clip norm mean=0.021 max=0.033  limit=0.02
2026-08-09 03:20:29,915 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=144,000  speed=755/s  reward=2.76
  loss     policy=0.0120  value=0.6921(x0.5)=0.3460
           entropy=2.3701  kl=0.0347
  value    V=1.94±0.56  R=1.99±1.18  adv=0.04±0.99
  moves    mv_ls=[-1.5701] (σ≈0.21, ≈12°) g=7.83e-03  d_move=[+0.0010] (Δσ≈0.012°)
           kk_ls=[-1.5678] (σ≈0.21, ≈12°)  d_kick=[+0.0005] (Δσ≈0.006°)
  heads    move= 35 get_poss= 65 exec_move= 89 sprint= 44 kick=  6 tackle=  8 shoot=
           4 hold=  4 tackle_prob=0.0830 kick_prob=0.0628
  vs       vs[win/loss/tout/miss]  vs_immobile(671): 83.5%/0.0%/0.4%/16.1%
  ep_len   15.9±9.5s  (n=671, min=1.1s, max=50.1s)
  reward   get_possession=+625.00  lose_possession=-6.30  ball_out=-230.00  box_possession=+1120.00
           speed_bonus=+329.63  timeout=-4.50  stamina_penalty=-3.76
  rew/ep   (mean/std/min/max per episode, 671 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.931    0.291    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.009    0.091    -0.900    +0.000
  ball_out          -0.343    1.263    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.669    0.743    +0.000    +2.000
  speed_bonus       +0.491    0.401    +0.000    +1.240
  opponent_box      +0.000    0.000    +0.000    +0.000
  timeout           -0.007    0.100    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.006    0.006    -0.025    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     628    +0.017    0.131     +2.764     1.111     +0.646      1.5545      1.014     1.810
  lose_possession       7    -0.000    0.013     +1.561     0.847     -0.601      0.9814      0.601     1.902
  ball_out            46    -0.006    0.179     -4.957     0.204     -4.706     24.0491      4.706     6.836
  box_possession     560    +0.031    0.247     +2.685     0.363     +0.488      0.5919      0.589     1.557
  speed_bonus        506    +0.009    0.086     +2.686     0.350     +0.487      0.5822      0.575     1.555
  timeout              3    -0.000    0.014     -1.500     0.000     -2.705      7.8977      2.705     3.306
  stamina_penalty     480    -0.000    0.001     +2.693     0.350     +0.481      0.5822      0.580     1.558
  gae/td   mean_return=+1.987  std_return=1.184  mean_gae=+0.043  mean_sq_td=0.9813
──────────────────────────────────────────────────────────────────────
2026-08-09 03:20:29,940 INFO Saved checkpoint: checkpoints/phase1_run48/checkpoint4.pt
2026-08-09 03:20:29,940 INFO Logging to checkpoints/phase1_run48/training_log5.txt
2026-08-09 03:20:29,941 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 03:20:43,998 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 03:20:44,000 INFO   [eval vs immobile] step=144,000  seeds=16x8  win=73%  mean_rew=2.098±2.486  V=1.888  gap=-0.210  outcomes={'box_possession': 93, 'opponent_box_possession': 1, 'miss': 32, 'timeout': 2}
2026-08-09 03:20:44,002 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 03:20:56,466 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 03:20:56,469 INFO   [eval vs rules] step=144,000  seeds=16x8  win=31%  mean_rew=-0.773±2.923  V=1.247  gap=+2.019  outcomes={'opponent_box_possession': 76, 'box_possession': 40, 'timeout': 1, 'miss': 11}
2026-08-09 03:26:50,474 INFO   [advantage] mean=-0.000  std=1.000  min=-7.186  max=4.183
2026-08-09 03:26:50,475 INFO   [ratio] mean=0.9837  std=0.2004  min=0.0045  max=14.9220  clipped=21.8%
2026-08-09 03:26:50,475 INFO   [exec head grad norm] move_direction=0.019  exec_move=0.071  sprint=0.049  kick=0.064  kick_direction=0.008  kick_power=0.000  kick_spin=0.000  tackle_attempt=0.079
2026-08-09 03:26:50,475 INFO   [exec continuous log_std] move_direction: start=-1.5701 end=-1.5693   kick_direction: start=-1.5678 end=-1.5672
2026-08-09 03:26:50,475 INFO   [exec continuous Δ per opt step] move_direction(dmean=0.0007≈0.04°/step  epoch≈2.5°  dlog_std=0.00001  Δσ°=0.000/step)  kick_direction(dmean=0.0009≈0.05°/step  epoch≈3.1°  dlog_std=0.00001  Δσ°=0.000/step)
2026-08-09 03:26:50,475 INFO   [exec discrete Δlogit per opt step] exec_move=0.0031  sprint=0.0030  kick=0.0027  tackle_attempt=0.0030
2026-08-09 03:26:50,476 INFO   [per-head KL] shoot=+0.0000  pass_=+0.0000  move=+0.0012  tackle=+0.0000  gp_extra=+0.0000  mark=+0.0000  hold=+0.0000  exec_move=+0.0047  sprint=+0.0075  kick=+0.0012  tackle_attempt=+0.0015  move_dir=+0.0128  kick_dir=+0.0072
2026-08-09 03:26:50,476 INFO   [grad clip] main: 4/60 steps clipped (7%)  pre-clip norm mean=0.283 max=0.468  limit=0.4
              direction: 34/60 steps clipped (57%)  pre-clip norm mean=0.022 max=0.058  limit=0.02
2026-08-09 03:26:50,535 INFO ──────────────────────────────────────────────────────────────────────
[PPO] step=180,000  speed=753/s  reward=3.18
  loss     policy=0.0111  value=0.6966(x0.5)=0.3483
           entropy=2.6616  kl=0.0360
  value    V=2.00±0.62  R=1.97±1.18  adv=-0.02±0.99
  moves    mv_ls=[-1.5693] (σ≈0.21, ≈12°) g=6.48e-03  d_move=[+0.0008] (Δσ≈0.010°)
           kk_ls=[-1.5672] (σ≈0.21, ≈12°)  d_kick=[+0.0006] (Δσ≈0.007°)
  heads    move= 33 get_poss= 68 exec_move= 88 sprint= 40 kick=  7 tackle=  9 shoot=
           5 hold=  4 tackle_prob=0.0956 kick_prob=0.0731
  vs       vs[win/loss/tout/miss]  vs_immobile(638): 83.2%/0.2%/1.3%/15.4%
  ep_len   16.8±10.4s  (n=638, min=0.6s, max=50.1s)
  reward   get_possession=+591.00  lose_possession=-1.80  ball_out=-230.00  box_possession=+1062.00
           speed_bonus=+305.35  opponent_box=-3.00  timeout=-12.00  stamina_penalty=-3.14
  rew/ep   (mean/std/min/max per episode, 638 ep)
  component           mean      std       min       max
  --------------  --------  -------  --------  --------
  approach          +0.000    0.000    +0.000    +0.000
  retreat           +0.000    0.000    +0.000    +0.000
  approach_speed    +0.000    0.000    +0.000    +0.000
  heading           +0.000    0.000    +0.000    +0.000
  get_possession    +0.926    0.273    +0.000    +2.000
  progress          +0.000    0.000    +0.000    +0.000
  lose_possession    -0.003    0.050    -0.900    +0.000
  ball_out          -0.361    1.293    -5.000    +0.000
  illegal           +0.000    0.000    +0.000    +0.000
  box_possession    +1.665    0.747    +0.000    +2.000
  speed_bonus       +0.479    0.386    +0.000    +1.227
  opponent_box      -0.005    0.119    -3.000    +0.000
  timeout           -0.019    0.167    -1.500    +0.000
  proximity_bonus    +0.000    0.000    +0.000    +0.000
  stamina_penalty    -0.005    0.005    -0.028    +0.000
  rew/step (per-step stats, n=36000 steps; ret/gae/td at steps where component fired)
  component        count      mean      std   mean_ret   std_ret   mean_gae  mean_sq_td   mean|td|   p95|td|
  --------------  ------  --------  -------  ---------  --------  ---------  ----------  ---------  --------
  get_possession     595    +0.017    0.127     +2.754     1.139     +0.476      1.4598      0.906     1.878
  lose_possession       2    -0.000    0.007     +1.745     0.172     -0.352      0.1277      0.352     0.406
  ball_out            46    -0.006    0.179     -5.000     0.000     -4.269     20.2316      4.269     6.670
  box_possession     531    +0.029    0.241     +2.663     0.370     +0.422      0.5186      0.534     1.392
  speed_bonus        487    +0.008    0.081     +2.674     0.354     +0.436      0.5259      0.533     1.394
  opponent_box         1    -0.000    0.016     -3.002     0.000     -5.129     26.3034      5.129     5.129
  timeout              8    -0.000    0.022     -1.500     0.000     -3.584     12.9422      3.584     3.905
  stamina_penalty     439    -0.000    0.001     +2.670     0.446     +0.417      0.5854      0.552     1.387
  gae/td   mean_return=+1.974  std_return=1.183  mean_gae=-0.021  mean_sq_td=0.9731
──────────────────────────────────────────────────────────────────────
2026-08-09 03:26:50,565 INFO Saved checkpoint: checkpoints/phase1_run48/checkpoint5.pt
2026-08-09 03:26:50,566 INFO Logging to checkpoints/phase1_run48/training_log6.txt
2026-08-09 03:26:50,567 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 03:27:05,086 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 03:27:05,087 INFO   [eval vs immobile] step=180,000  seeds=16x8  win=80%  mean_rew=2.647±1.838  V=1.908  gap=-0.739  outcomes={'box_possession': 103, 'miss': 25}
2026-08-09 03:27:05,089 INFO   [seeded eval] running 16x8 episodes across 7 worker process(es)...
2026-08-09 03:27:21,009 INFO   [seeded eval] all workers finished, merging results.
2026-08-09 03:27:21,011 INFO   [eval vs rules] step=180,000  seeds=16x8  win=25%  mean_rew=-1.161±2.781  V=1.145  gap=+2.307  outcomes={'box_possession': 32, 'opponent_box_possession': 85, 'timeout': 1, 'miss': 10}
