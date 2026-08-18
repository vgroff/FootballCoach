2026-08-18 11:18:42,540 WARNING --checkpoint and --data both given: using --data as the dataset (no live rollout will be collected) and --checkpoint only to initialize decision_net/value_net's weights -- i.e. warm-starting value fitting from a trained checkpoint against a pre-recorded dataset instead of a fresh rollout.
2026-08-18 11:18:42,655 INFO Checkpoint checkpoints/longterm/checkpoint_vvgood_immobile.pt: separate_value_net=True (auto-detected)
2026-08-18 11:18:42,685 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 11:18:42,687 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 11:18:42,687 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 11:18:42,690 INFO --reset-dir-log-std: reset move_dir_log_std/kick_dir_log_std to config init values before rollout collection.
2026-08-18 11:18:43,049 INFO Loading 25041 demonstration file(s) from demonstrations/phase1_neural_vvgood
2026-08-18 11:50:46,663 WARNING --checkpoint and --data both given: using --data as the dataset (no live rollout will be collected) and --checkpoint only to initialize decision_net/value_net's weights -- i.e. warm-starting value fitting from a trained checkpoint against a pre-recorded dataset instead of a fresh rollout.
2026-08-18 11:50:46,818 INFO Checkpoint checkpoints/longterm/checkpoint_vvgood_immobile.pt: separate_value_net=True (auto-detected)
2026-08-18 11:50:46,851 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 11:50:46,852 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 11:50:46,852 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 11:50:46,855 INFO --reset-dir-log-std: reset move_dir_log_std/kick_dir_log_std to config init values before rollout collection.
2026-08-18 11:50:47,267 INFO Loading 25041 demonstration file(s) from demonstrations/phase1_neural_vvgood
2026-08-18 11:53:18,500 INFO Dataset: 11,124,006 steps loaded
2026-08-18 11:53:18,536 INFO Loaded 11,124,006 rows total
2026-08-18 11:53:18,540 INFO has_rewards=True
2026-08-18 11:53:18,735 INFO valid_indices(): 5,564,872 rows (50.0% of total)
2026-08-18 11:53:21,778 INFO Returns over ALL rows: mean=3.784 std=1.388 min=-4.000 max=34.657
2026-08-18 11:53:21,794 INFO Returns over valid_indices(): mean=3.785 std=1.394
2026-08-18 11:53:25,667 INFO --- Dataset distribution (11,124,006 rows, 200304 episodes) ---
2026-08-18 11:53:25,713 INFO   self.ai_type == rules: 0.0%
2026-08-18 11:53:25,750 INFO   self.ai_type == immobile: 50.0%
2026-08-18 11:53:25,786 INFO   self.ai_type == neural: 50.0%
2026-08-18 11:53:25,820 INFO   opponent.ai_type == rules: 0.0%
2026-08-18 11:53:25,860 INFO   opponent.ai_type == immobile: 50.0%
2026-08-18 11:53:25,898 INFO   opponent.ai_type == neural: 50.0%
2026-08-18 11:53:26,622 INFO   valid rows: kick_this_tick rate=0.00%  tackle_attempt rate=0.75%
2026-08-18 11:53:26,628 INFO   dones=1 rows: 400,608  |  zero-reward rows: 10,410,340 (93.6%)
2026-08-18 11:53:26,891 INFO   return percentiles (all rows): p10=2.40  p50=3.83  p90=5.45
2026-08-18 11:53:34,456 INFO --- Reward component breakdown (all episodes, 400608 episode(s)) ---
2026-08-18 11:53:34,456 INFO   component           mean      std       min       max
2026-08-18 11:53:34,524 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:34,594 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:34,667 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:34,739 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:34,821 INFO   get_possession    +0.932    0.974    +0.000    +6.000
2026-08-18 11:53:34,895 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:34,955 INFO   lose_possession    -0.008    0.121    -3.600    +0.000
2026-08-18 11:53:35,013 INFO   ball_out          -0.027    0.327    -4.000    +0.000
2026-08-18 11:53:35,079 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:35,145 INFO   box_possession    +1.828    0.560    +0.000    +2.000
2026-08-18 11:53:35,214 INFO   speed_bonus       +2.095    0.927    -0.023    +3.985
2026-08-18 11:53:35,275 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:35,345 INFO   timeout           -0.003    0.052    -1.000    +0.000
2026-08-18 11:53:35,408 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:35,466 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:35,543 INFO   stamina_penalty    -0.072    0.034    -0.157    +0.000
2026-08-18 11:53:38,263 INFO --- Reward component breakdown (outcome=ball_out, 2698 episode(s)) ---
2026-08-18 11:53:38,263 INFO   component           mean      std       min       max
2026-08-18 11:53:38,264 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:38,264 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:38,264 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:38,265 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:38,265 INFO   get_possession    +0.950    0.747    +0.000    +4.000
2026-08-18 11:53:38,265 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:38,265 INFO   lose_possession    -0.002    0.060    -1.800    +0.000
2026-08-18 11:53:38,265 INFO   ball_out          -4.000    0.000    -4.000    -4.000
2026-08-18 11:53:38,266 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:38,266 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:38,266 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:38,266 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:38,267 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:38,267 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:38,267 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:38,267 INFO   stamina_penalty    +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:41,058 INFO --- Reward component breakdown (outcome=invalid, 30620 episode(s)) ---
2026-08-18 11:53:41,058 INFO   component           mean      std       min       max
2026-08-18 11:53:41,066 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:41,073 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:41,080 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:41,086 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:41,092 INFO   get_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:41,101 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:41,110 INFO   lose_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:41,116 INFO   ball_out          +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:41,127 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:41,136 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:41,145 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:41,152 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:41,164 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:41,172 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:41,180 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:41,189 INFO   stamina_penalty    +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:43,590 INFO --- Reward component breakdown (outcome=loss, 0 episode(s)) ---
2026-08-18 11:53:45,960 INFO --- Reward component breakdown (outcome=timeout, 1074 episode(s)) ---
2026-08-18 11:53:45,960 INFO   component           mean      std       min       max
2026-08-18 11:53:45,961 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:45,961 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:45,962 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:45,962 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:45,962 INFO   get_possession    +1.063    1.181    +0.000    +6.000
2026-08-18 11:53:45,962 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:45,962 INFO   lose_possession    -0.124    0.462    -3.600    +0.000
2026-08-18 11:53:45,963 INFO   ball_out          +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:45,963 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:45,963 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:45,963 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:45,963 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:45,963 INFO   timeout           -1.000    0.000    -1.000    -1.000
2026-08-18 11:53:45,963 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:45,963 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 11:53:45,964 INFO   stamina_penalty    -0.088    0.029    -0.154    -0.023
2026-08-18 11:53:51,236 INFO --- MC returns by outcome (all rows, 11,124,006 rows) ---
2026-08-18 11:53:51,447 INFO   ball_out     n= 52,878  mean=-2.240  std=0.728  min=-4.000  max=-0.226
2026-08-18 11:53:51,571 INFO   invalid      n=417,414  mean=+0.000  std=0.000  min=+0.000  max=+0.000
2026-08-18 11:53:51,696 INFO   timeout      n= 72,212  mean=+0.233  std=0.848  min=-2.754  max=+1.547
2026-08-18 11:53:51,881 INFO   win          n=10,581,502  mean=+3.988  std=1.069  min=-0.316  max=+34.657
2026-08-18 11:53:58,372 INFO --- Episode total reward by outcome (all rows, 200304 episode(s)) ---
2026-08-18 11:53:58,408 INFO   ball_out     n= 1,349  mean=-2.551  std=0.593  min=-4.000  max=-1.800
2026-08-18 11:53:58,409 INFO   invalid      n=15,310  mean=+0.000  std=0.000  min=+0.000  max=+0.000
2026-08-18 11:53:58,410 INFO   timeout      n=   537  mean=+0.783  std=0.510  min=-1.119  max=+1.335
2026-08-18 11:53:58,411 INFO   win          n=183,108  mean=+6.128  std=0.663  min=+2.963  max=+35.431
2026-08-18 11:54:01,465 INFO Train/val split (valid_only=True): 4,452,750 train rows across 160243 episodes  |  1,112,122 val rows across 40061 episodes
2026-08-18 11:54:06,124 INFO --- Linear regression baseline (top_speed, ball_dist, ball_to_box, time_remaining) ---
2026-08-18 11:54:06,943 INFO   [all outcomes] n_train=4,452,750  n_val=1,112,122  coef(top_speed, ball_dist, ball_to_box, time_rem, intercept)=[0.323, -4.054, -1.877, 4.45, 1.464]
    train_rmse=1.1243 (norm=0.8073)  val_rmse=1.1369 (norm=0.8164)
2026-08-18 11:54:11,246 INFO   [win outcomes only] n_train=4,235,005  n_val=1,058,460  coef(top_speed, ball_dist, ball_to_box, time_rem, intercept)=[0.417, -2.092, -1.95, 4.526, 1.432]
    train_rmse=0.6633 (norm=0.6156)  val_rmse=0.6650 (norm=0.6172)
2026-08-18 11:54:12,833 INFO value_net: fresh separate value_net (--reset-separate-value-net)  total_params=281,340  trainable_params=281,340
2026-08-18 11:54:12,844 INFO Fitting fresh separate value network: 100 epochs, lr=0.0002, weight_decay=1e-06, batch_size=1096, train_ret_std=1.393, outcome_reweight=False
2026-08-18 12:06:01,608 INFO epoch   0/100 (baseline, no training yet)  train_rmse=3.9395 (norm=2.8290)  val_rmse=3.9435 (norm=2.8319)
2026-08-18 12:06:01,608 INFO     opponent=immobile   train_rmse=3.9395 (n=4452750)  val_rmse=3.9435 (n=1112122)
2026-08-18 12:06:01,608 INFO     outcome=ball_out      train_rmse=2.4203 (n=20546)  val_rmse=2.4196 (n=5921)
2026-08-18 12:06:01,608 INFO     outcome=invalid       train_rmse=0.0889 (n=167260)  val_rmse=0.0888 (n=41447)
2026-08-18 12:06:01,608 INFO     outcome=timeout       train_rmse=0.8697 (n=29939)  val_rmse=0.8914 (n=6294)
2026-08-18 12:06:01,608 INFO     outcome=win           train_rmse=4.0353 (n=4235005)  val_rmse=4.0376 (n=1058460)
2026-08-18 15:38:56,103 WARNING --checkpoint and --data both given: using --data as the dataset (no live rollout will be collected) and --checkpoint only to initialize decision_net/value_net's weights -- i.e. warm-starting value fitting from a trained checkpoint against a pre-recorded dataset instead of a fresh rollout.
2026-08-18 15:38:56,211 INFO Checkpoint checkpoints/longterm/checkpoint_vvgood_immobile.pt: separate_value_net=True (auto-detected)
2026-08-18 15:38:56,245 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 15:38:56,248 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 15:38:56,248 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 15:38:56,253 INFO --reset-dir-log-std: reset move_dir_log_std/kick_dir_log_std to config init values before rollout collection.
2026-08-18 15:38:56,256 INFO Loading 240 demonstration file(s) from demonstrations/phase1_neural_vvgood
2026-08-18 15:39:58,876 INFO Dataset: 6,667,368 steps loaded
2026-08-18 15:39:58,876 INFO Loaded 6,667,368 rows total
2026-08-18 15:39:58,878 INFO has_rewards=True
2026-08-18 15:39:58,946 INFO valid_indices(): 3,335,345 rows (50.0% of total)
2026-08-18 15:40:01,266 INFO Returns over ALL rows: mean=0.823 std=3.238 min=-5.140 max=198.470
2026-08-18 15:40:01,272 INFO Returns over valid_indices(): mean=3.743 std=1.308
2026-08-18 15:40:03,563 INFO --- Dataset distribution (6,667,368 rows, 120000 episodes) ---
2026-08-18 15:40:03,600 INFO   self.ai_type == rules: 0.0%
2026-08-18 15:40:03,625 INFO   self.ai_type == immobile: 50.0%
2026-08-18 15:40:03,651 INFO   self.ai_type == neural: 50.0%
2026-08-18 15:40:03,687 INFO   opponent.ai_type == rules: 0.0%
2026-08-18 15:40:03,716 INFO   opponent.ai_type == immobile: 50.0%
2026-08-18 15:40:03,746 INFO   opponent.ai_type == neural: 50.0%
2026-08-18 15:40:04,105 INFO   valid rows: kick_this_tick rate=0.00%  tackle_attempt rate=0.74%
2026-08-18 15:40:04,109 INFO   dones=1 rows: 240,000  |  zero-reward rows: 6,337,944 (95.1%)
2026-08-18 15:40:04,213 INFO   return percentiles (all rows): p10=-2.44  p50=0.00  p90=4.69
2026-08-18 15:40:08,916 INFO --- Reward component breakdown (all episodes, 240000 episode(s)) ---
2026-08-18 15:40:08,917 INFO   component           mean      std       min       max
2026-08-18 15:40:08,953 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:09,013 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:09,082 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:09,153 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:09,218 INFO   get_possession    +0.950    1.024    +0.000   +10.000
2026-08-18 15:40:09,271 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:09,320 INFO   lose_possession    -0.022    0.263    -7.200    +0.000
2026-08-18 15:40:09,361 INFO   ball_out          -0.028    0.331    -4.000    +0.000
2026-08-18 15:40:09,400 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:09,447 INFO   box_possession    +1.856    1.497    +0.000  +254.000
2026-08-18 15:40:09,508 INFO   speed_bonus       +2.137    2.197    -0.023  +239.212
2026-08-18 15:40:09,558 INFO   opponent_box      -2.289    0.695    -2.500    +0.000
2026-08-18 15:40:09,601 INFO   timeout           -0.005    0.104    -2.000    +0.000
2026-08-18 15:40:09,652 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:09,701 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:09,763 INFO   stamina_penalty    -0.122    0.067    -9.830    +0.000
2026-08-18 15:40:11,422 INFO --- Reward component breakdown (outcome=ball_out, 1650 episode(s)) ---
2026-08-18 15:40:11,422 INFO   component           mean      std       min       max
2026-08-18 15:40:11,423 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:11,423 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:11,424 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:11,424 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:11,425 INFO   get_possession    +0.960    0.759    +0.000    +6.000
2026-08-18 15:40:11,425 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:11,425 INFO   lose_possession    -0.009    0.177    -3.600    +0.000
2026-08-18 15:40:11,425 INFO   ball_out          -4.000    0.000    -4.000    -4.000
2026-08-18 15:40:11,425 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:11,426 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:11,426 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:11,426 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:11,426 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:11,426 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:11,427 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:11,427 INFO   stamina_penalty    +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:13,127 INFO --- Reward component breakdown (outcome=invalid, 17940 episode(s)) ---
2026-08-18 15:40:13,127 INFO   component           mean      std       min       max
2026-08-18 15:40:13,132 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:13,139 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:13,147 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:13,152 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:13,161 INFO   get_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:13,167 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:13,175 INFO   lose_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:13,179 INFO   ball_out          +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:13,183 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:13,187 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:13,195 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:13,202 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:13,208 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:13,214 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:13,218 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:13,226 INFO   stamina_penalty    +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:14,651 INFO --- Reward component breakdown (outcome=loss, 0 episode(s)) ---
2026-08-18 15:40:16,045 INFO --- Reward component breakdown (outcome=timeout, 654 episode(s)) ---
2026-08-18 15:40:16,045 INFO   component           mean      std       min       max
2026-08-18 15:40:16,045 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:16,045 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:16,045 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:16,045 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:16,046 INFO   get_possession    +1.232    1.575    +0.000   +10.000
2026-08-18 15:40:16,046 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:16,046 INFO   lose_possession    -0.234    0.883    -7.200    +0.000
2026-08-18 15:40:16,046 INFO   ball_out          +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:16,046 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:16,046 INFO   box_possession    +1.394   15.731    +0.000  +254.000
2026-08-18 15:40:16,046 INFO   speed_bonus       +1.634   18.437    -0.023  +239.212
2026-08-18 15:40:16,046 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:16,046 INFO   timeout           -1.997    0.055    -2.000    -1.000
2026-08-18 15:40:16,046 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:16,047 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 15:40:16,047 INFO   stamina_penalty    -0.170    0.529    -9.830    -0.024
2026-08-18 15:40:19,113 INFO --- MC returns by outcome (all rows, 6,667,368 rows) ---
2026-08-18 15:40:19,248 INFO   ball_out     n= 32,987  mean=-1.413  std=1.452  min=-4.774  max=+1.000
2026-08-18 15:40:19,331 INFO   invalid      n=246,666  mean=+0.000  std=0.069  min=-2.514  max=+6.836
2026-08-18 15:40:19,405 INFO   timeout      n= 43,945  mean=-0.079  std=7.249  min=-4.307  max=+198.470
2026-08-18 15:40:19,517 INFO   win          n=6,343,770  mean=+0.873  std=3.254  min=-5.140  max=+139.577
2026-08-18 15:40:23,984 INFO --- Episode total reward by outcome (all rows, 120000 episode(s)) ---
2026-08-18 15:40:24,007 INFO   ball_out     n=   825  mean=-3.063  std=0.288  min=-4.000  max=+1.000
2026-08-18 15:40:24,008 INFO   invalid      n= 8,970  mean=+0.002  std=0.103  min=+0.000  max=+6.892
2026-08-18 15:40:24,008 INFO   timeout      n=   327  mean=-0.360  std=0.698  min=-3.761  max=+2.773
2026-08-18 15:40:24,009 INFO   win          n=109,878  mean=+5.187  std=0.799  min=-3.493  max=+36.081
2026-08-18 15:40:25,939 INFO Train/val split (valid_only=True): 2,670,501 train rows across 96000 episodes  |  664,844 val rows across 24000 episodes
2026-08-18 15:40:28,716 INFO --- Linear regression baseline (top_speed, ball_dist, ball_to_box, time_remaining) ---
2026-08-18 15:40:29,418 INFO   [all outcomes] n_train=2,670,501  n_val=664,844  coef(top_speed, ball_dist, ball_to_box, time_rem, intercept)=[0.361, -4.124, -1.79, 4.009, 1.689]
    train_rmse=1.0575 (norm=0.8102)  val_rmse=1.0700 (norm=0.8198)
2026-08-18 15:40:32,069 INFO   [win outcomes only] n_train=2,542,251  n_val=631,216  coef(top_speed, ball_dist, ball_to_box, time_rem, intercept)=[0.379, -2.201, -1.842, 4.021, 1.757]
    train_rmse=0.5165 (norm=0.5571)  val_rmse=0.5170 (norm=0.5576)
2026-08-18 15:40:33,200 INFO value_net: fresh separate value_net (--reset-separate-value-net)  total_params=281,340  trainable_params=281,340
2026-08-18 15:40:33,205 INFO Fitting fresh separate value network: 100 epochs, lr=0.0002, weight_decay=1e-06, batch_size=3096, train_ret_std=1.305, outcome_reweight=False
2026-08-18 15:51:38,006 WARNING --checkpoint and --data both given: using --data as the dataset (no live rollout will be collected) and --checkpoint only to initialize decision_net/value_net's weights -- i.e. warm-starting value fitting from a trained checkpoint against a pre-recorded dataset instead of a fresh rollout.
2026-08-18 15:51:38,128 INFO Checkpoint checkpoints/longterm/checkpoint_vvgood_immobile.pt: separate_value_net=True (auto-detected)
2026-08-18 15:51:38,160 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 15:51:38,162 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 15:51:38,162 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 15:51:38,165 INFO --reset-dir-log-std: reset move_dir_log_std/kick_dir_log_std to config init values before rollout collection.
2026-08-18 15:51:38,167 INFO Loading 240 demonstration file(s) from demonstrations/phase1_neural_vvgood
2026-08-18 15:52:31,732 INFO Dataset: 6,667,368 steps loaded
2026-08-18 15:52:31,732 INFO Loaded 6,667,368 rows total
2026-08-18 15:52:31,734 INFO has_rewards=True
2026-08-18 15:52:31,880 INFO valid_indices(): 3,335,345 rows (50.0% of total)
2026-08-18 15:52:34,167 INFO Returns over ALL rows: mean=0.823 std=3.238 min=-5.140 max=198.470
2026-08-18 15:52:34,173 INFO Returns over valid_indices(): mean=3.743 std=1.308
2026-08-18 15:52:36,458 INFO --- Dataset distribution (6,667,368 rows, 120000 episodes) ---
2026-08-18 15:52:36,487 INFO   self.ai_type == rules: 0.0%
2026-08-18 15:52:36,517 INFO   self.ai_type == immobile: 50.0%
2026-08-18 15:52:36,544 INFO   self.ai_type == neural: 50.0%
2026-08-18 15:52:36,582 INFO   opponent.ai_type == rules: 0.0%
2026-08-18 15:52:36,606 INFO   opponent.ai_type == immobile: 50.0%
2026-08-18 15:52:36,631 INFO   opponent.ai_type == neural: 50.0%
2026-08-18 15:52:37,032 INFO   valid rows: kick_this_tick rate=0.00%  tackle_attempt rate=0.74%
2026-08-18 15:52:37,036 INFO   dones=1 rows: 240,000  |  zero-reward rows: 6,337,944 (95.1%)
2026-08-18 15:52:37,135 INFO   return percentiles (all rows): p10=-2.44  p50=0.00  p90=4.69
2026-08-18 15:52:40,699 INFO --- Reward component breakdown (all episodes, 120000 episode(s)) ---
2026-08-18 15:52:40,699 INFO   component           mean      std       min       max
2026-08-18 15:52:40,718 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:40,747 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:40,786 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:40,816 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:40,845 INFO   get_possession    +1.899    0.680    +0.000   +10.000
2026-08-18 15:52:40,865 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:40,887 INFO   lose_possession    -0.045    0.371    -7.200    +0.000
2026-08-18 15:52:40,907 INFO   ball_out          -0.055    0.661    -8.000    +0.000
2026-08-18 15:52:40,928 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:40,951 INFO   box_possession    +3.711    2.256    +0.000  +256.000
2026-08-18 15:52:40,972 INFO   speed_bonus       +4.274    3.347    -0.045  +239.189
2026-08-18 15:52:41,013 INFO   opponent_box      -4.578    1.390    -5.000    +0.000
2026-08-18 15:52:41,034 INFO   timeout           -0.011    0.208    -4.000    +0.000
2026-08-18 15:52:41,065 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:41,088 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:41,112 INFO   stamina_penalty    -0.244    0.120    -9.988    +0.000
2026-08-18 15:52:43,125 INFO --- Reward component breakdown (outcome=ball_out, 825 episode(s)) ---
2026-08-18 15:52:43,125 INFO   component           mean      std       min       max
2026-08-18 15:52:43,125 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:43,125 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:43,125 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:43,125 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:43,126 INFO   get_possession    +1.920    0.520    +0.000    +6.000
2026-08-18 15:52:43,126 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:43,126 INFO   lose_possession    -0.017    0.250    -3.600    +0.000
2026-08-18 15:52:43,126 INFO   ball_out          -8.000    0.000    -8.000    -8.000
2026-08-18 15:52:43,126 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:43,126 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:43,126 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:43,126 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:43,126 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:43,126 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:43,127 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:43,127 INFO   stamina_penalty    +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:45,130 INFO --- Reward component breakdown (outcome=invalid, 8970 episode(s)) ---
2026-08-18 15:52:45,130 INFO   component           mean      std       min       max
2026-08-18 15:52:45,132 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:45,133 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:45,134 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:45,135 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:45,136 INFO   get_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:45,137 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:45,138 INFO   lose_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:45,139 INFO   ball_out          +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:45,140 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:45,141 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:45,142 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:45,145 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:45,147 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:45,149 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:45,151 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:45,152 INFO   stamina_penalty    +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:46,995 INFO --- Reward component breakdown (outcome=loss, 0 episode(s)) ---
2026-08-18 15:52:48,825 INFO --- Reward component breakdown (outcome=timeout, 327 episode(s)) ---
2026-08-18 15:52:48,825 INFO   component           mean      std       min       max
2026-08-18 15:52:48,826 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:48,826 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:48,826 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:48,826 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:48,826 INFO   get_possession    +2.465    1.400    +0.000   +10.000
2026-08-18 15:52:48,826 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:48,826 INFO   lose_possession    -0.468    1.206    -7.200    +0.000
2026-08-18 15:52:48,826 INFO   ball_out          +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:48,826 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:48,826 INFO   box_possession    +2.789   22.229    +0.000  +256.000
2026-08-18 15:52:48,826 INFO   speed_bonus       +3.268   25.971    +0.000  +239.189
2026-08-18 15:52:48,826 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:48,827 INFO   timeout           -3.994    0.110    -4.000    -2.000
2026-08-18 15:52:48,827 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:48,827 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 15:52:48,827 INFO   stamina_penalty    -0.341    0.752    -9.988    -0.048
2026-08-18 15:52:51,893 INFO --- MC returns by outcome (all rows, 6,667,368 rows) ---
2026-08-18 15:52:52,019 INFO   ball_out     n= 32,987  mean=-1.413  std=1.452  min=-4.774  max=+1.000
2026-08-18 15:52:52,095 INFO   invalid      n=246,666  mean=+0.000  std=0.069  min=-2.514  max=+6.836
2026-08-18 15:52:52,171 INFO   timeout      n= 43,945  mean=-0.079  std=7.249  min=-4.307  max=+198.470
2026-08-18 15:52:52,313 INFO   win          n=6,343,770  mean=+0.873  std=3.254  min=-5.140  max=+139.577
2026-08-18 15:52:56,756 INFO --- Episode total reward by outcome (all rows, 120000 episode(s)) ---
2026-08-18 15:52:56,778 INFO   ball_out     n=   825  mean=-3.063  std=0.288  min=-4.000  max=+1.000
2026-08-18 15:52:56,778 INFO   invalid      n= 8,970  mean=+0.002  std=0.103  min=+0.000  max=+6.892
2026-08-18 15:52:56,779 INFO   timeout      n=   327  mean=-0.360  std=0.698  min=-3.761  max=+2.773
2026-08-18 15:52:56,780 INFO   win          n=109,878  mean=+5.187  std=0.799  min=-3.493  max=+36.081
2026-08-18 15:52:58,623 INFO Train/val split (valid_only=True): 2,670,501 train rows across 96000 episodes  |  664,844 val rows across 24000 episodes
2026-08-18 15:53:01,456 INFO --- Linear regression baseline (top_speed, ball_dist, ball_to_box, time_remaining) ---
2026-08-18 15:53:02,293 INFO   [all outcomes] n_train=2,670,501  n_val=664,844  coef(top_speed, ball_dist, ball_to_box, time_rem, intercept)=[0.361, -4.124, -1.79, 4.009, 1.689]
    train_rmse=1.0575 (norm=0.8102)  val_rmse=1.0700 (norm=0.8198)
2026-08-18 15:53:05,133 INFO   [win outcomes only] n_train=2,542,251  n_val=631,216  coef(top_speed, ball_dist, ball_to_box, time_rem, intercept)=[0.379, -2.201, -1.842, 4.021, 1.757]
    train_rmse=0.5165 (norm=0.5571)  val_rmse=0.5170 (norm=0.5576)
2026-08-18 15:53:06,204 INFO value_net: fresh separate value_net (--reset-separate-value-net)  total_params=281,340  trainable_params=281,340
2026-08-18 15:53:06,209 INFO Fitting fresh separate value network: 100 epochs, lr=0.0002, weight_decay=1e-06, batch_size=3096, train_ret_std=1.305, outcome_reweight=False
2026-08-18 16:19:38,055 WARNING --checkpoint and --data both given: using --data as the dataset (no live rollout will be collected) and --checkpoint only to initialize decision_net/value_net's weights -- i.e. warm-starting value fitting from a trained checkpoint against a pre-recorded dataset instead of a fresh rollout.
2026-08-18 16:19:38,156 INFO Checkpoint checkpoints/longterm/checkpoint_vvgood_immobile.pt: separate_value_net=True (auto-detected)
2026-08-18 16:19:38,187 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 16:19:38,188 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 16:19:38,188 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 16:19:38,191 INFO --reset-dir-log-std: reset move_dir_log_std/kick_dir_log_std to config init values before rollout collection.
2026-08-18 16:19:38,191 INFO Loading 5 demonstration file(s) from demonstrations/phase1_neural_vvgood
2026-08-18 16:19:38,700 INFO Dataset: 54,793 steps loaded
2026-08-18 16:19:38,700 INFO Loaded 54,793 rows total
2026-08-18 16:19:38,700 INFO has_rewards=True
2026-08-18 16:19:38,701 INFO valid_indices(): 27,406 rows (50.0% of total)
2026-08-18 16:19:38,721 INFO Returns over ALL rows: mean=0.802 std=3.097 min=-4.000 max=6.783
2026-08-18 16:19:38,721 INFO Returns over valid_indices(): mean=3.717 std=1.368
2026-08-18 16:19:38,750 INFO --- Dataset distribution (54,793 rows, 1000 episodes) ---
2026-08-18 16:19:38,752 INFO   self.ai_type == rules: 0.0%
2026-08-18 16:19:38,752 INFO   self.ai_type == immobile: 50.0%
2026-08-18 16:19:38,753 INFO   self.ai_type == neural: 50.0%
2026-08-18 16:19:38,753 INFO   opponent.ai_type == rules: 0.0%
2026-08-18 16:19:38,753 INFO   opponent.ai_type == immobile: 50.0%
2026-08-18 16:19:38,753 INFO   opponent.ai_type == neural: 50.0%
2026-08-18 16:19:38,754 INFO   valid rows: kick_this_tick rate=0.00%  tackle_attempt rate=0.76%
2026-08-18 16:19:38,754 INFO   dones=1 rows: 2,000  |  zero-reward rows: 52,081 (95.1%)
2026-08-18 16:19:38,766 INFO   return percentiles (valid rows): p10=2.49  p50=3.81  p90=5.24
2026-08-18 16:19:38,808 INFO --- Reward component breakdown (all episodes, 1000 episode(s)) ---
2026-08-18 16:19:38,809 INFO   component           mean      std       min       max
2026-08-18 16:19:38,809 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,810 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,810 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,810 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,810 INFO   get_possession    +0.942    0.344    +0.000    +3.000
2026-08-18 16:19:38,810 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,811 INFO   lose_possession    -0.022    0.176    -1.800    +0.000
2026-08-18 16:19:38,811 INFO   ball_out          -0.032    0.356    -4.000    +0.000
2026-08-18 16:19:38,811 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,811 INFO   box_possession    +1.814    0.581    +0.000    +2.000
2026-08-18 16:19:38,811 INFO   speed_bonus       +2.099    0.963    +0.000    +3.850
2026-08-18 16:19:38,811 INFO   opponent_box      -2.268    0.726    -2.500    +0.000
2026-08-18 16:19:38,812 INFO   timeout           -0.006    0.109    -2.000    +0.000
2026-08-18 16:19:38,812 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,812 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,812 INFO   stamina_penalty    -0.118    0.051    -0.221    +0.000
2026-08-18 16:19:38,828 INFO --- Reward component breakdown (outcome=ball_out, 8 episode(s)) ---
2026-08-18 16:19:38,829 INFO   component           mean      std       min       max
2026-08-18 16:19:38,829 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,829 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,829 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,829 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,829 INFO   get_possession    +0.875    0.331    +0.000    +1.000
2026-08-18 16:19:38,829 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,829 INFO   lose_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,829 INFO   ball_out          -4.000    0.000    -4.000    -4.000
2026-08-18 16:19:38,829 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,829 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,829 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,830 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,830 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,830 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,830 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,830 INFO   stamina_penalty    +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,843 INFO --- Reward component breakdown (outcome=invalid, 82 episode(s)) ---
2026-08-18 16:19:38,843 INFO   component           mean      std       min       max
2026-08-18 16:19:38,843 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,843 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,844 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,844 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,844 INFO   get_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,844 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,844 INFO   lose_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,844 INFO   ball_out          +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,844 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,844 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,844 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,844 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,844 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,844 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,844 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,845 INFO   stamina_penalty    +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,857 INFO --- Reward component breakdown (outcome=loss, 0 episode(s)) ---
2026-08-18 16:19:38,869 INFO --- Reward component breakdown (outcome=timeout, 3 episode(s)) ---
2026-08-18 16:19:38,869 INFO   component           mean      std       min       max
2026-08-18 16:19:38,869 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,869 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,869 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,869 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,869 INFO   get_possession    +1.000    0.000    +1.000    +1.000
2026-08-18 16:19:38,870 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,870 INFO   lose_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,870 INFO   ball_out          +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,870 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,870 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,870 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,870 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,870 INFO   timeout           -2.000    0.000    -2.000    -2.000
2026-08-18 16:19:38,870 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,870 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 16:19:38,870 INFO   stamina_penalty    -0.131    0.014    -0.146    -0.113
2026-08-18 16:19:38,888 INFO --- MC returns by outcome (valid rows, 27,406 rows) ---
2026-08-18 16:19:38,889 INFO   ball_out     n=    181  mean=-2.834  std=0.484  min=-4.000  max=-2.023
2026-08-18 16:19:38,889 INFO   invalid      n=  1,159  mean=+0.000  std=0.000  min=+0.000  max=+0.000
2026-08-18 16:19:38,890 INFO   timeout      n=    201  mean=-0.245  std=0.482  min=-1.104  max=+0.081
2026-08-18 16:19:38,891 INFO   win          n= 25,865  mean=+3.960  std=0.937  min=+0.405  max=+6.783
2026-08-18 16:19:38,925 INFO --- Episode total reward by outcome (valid rows, 1000 episode(s)) ---
2026-08-18 16:19:38,926 INFO   ball_out     n=     8  mean=-3.125  std=0.331  min=-4.000  max=-3.000
2026-08-18 16:19:38,926 INFO   invalid      n=    82  mean=+0.000  std=0.000  min=+0.000  max=+0.000
2026-08-18 16:19:38,926 INFO   timeout      n=     3  mean=-0.084  std=0.014  min=-0.104  max=-0.071
2026-08-18 16:19:38,926 INFO   win          n=   907  mean=+5.209  std=0.777  min=+1.822  max=+6.783
2026-08-18 16:19:38,941 INFO Train/val split (valid_only=True): 21,991 train rows across 800 episodes  |  5,415 val rows across 200 episodes
2026-08-18 16:19:38,966 INFO --- Linear regression baseline (top_speed, ball_dist, ball_to_box, time_remaining) ---
2026-08-18 16:19:38,970 INFO   [all outcomes] n_train=21,991  n_val=5,415  coef(top_speed, ball_dist, ball_to_box, time_rem, intercept)=[-0.053, -4.377, -1.766, 4.061, 1.94]
    train_rmse=1.1188 (norm=0.8107)  val_rmse=1.1086 (norm=0.8033)
2026-08-18 16:19:38,992 INFO   [win outcomes only] n_train=20,720  n_val=5,145  coef(top_speed, ball_dist, ball_to_box, time_rem, intercept)=[-0.281, -2.005, -1.838, 3.938, 2.315]
    train_rmse=0.4953 (norm=0.5256)  val_rmse=0.4604 (norm=0.4886)
2026-08-18 16:19:40,215 INFO value_net: fresh separate value_net (--reset-separate-value-net)  total_params=281,340  trainable_params=281,340
2026-08-18 16:19:40,215 INFO Fitting fresh separate value network: 100 epochs, lr=0.0002, weight_decay=1e-06, batch_size=3096, train_ret_std=1.380, outcome_reweight=False
2026-08-18 16:19:45,014 INFO epoch   0/100 (baseline, no training yet)  train_rmse=3.8688 (norm=2.8034)  val_rmse=3.8587 (norm=2.7960)
2026-08-18 16:19:45,014 INFO     opponent=immobile   train_rmse=3.8688 (n=21991)  val_rmse=3.8587 (n=5415)
2026-08-18 16:19:45,014 INFO     outcome=ball_out      train_rmse=2.9948 (n=143)  val_rmse=2.8548 (n=38)
2026-08-18 16:19:45,014 INFO     outcome=invalid       train_rmse=0.0893 (n=927)  val_rmse=0.0875 (n=232)
2026-08-18 16:19:45,014 INFO     outcome=timeout       train_rmse=0.6018 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:19:45,014 INFO     outcome=win           train_rmse=3.9775 (n=20720)  val_rmse=3.9510 (n=5145)
2026-08-18 16:19:53,471 INFO epoch   1/100  train_rmse=3.7658 (norm=2.7288)  val_rmse=3.5615 (norm=2.5807)
2026-08-18 16:19:53,471 INFO     opponent=immobile   train_rmse=3.7658 (n=21991)  val_rmse=3.5615 (n=5415)
2026-08-18 16:19:53,471 INFO     outcome=ball_out      train_rmse=3.0959 (n=143)  val_rmse=3.1199 (n=38)
2026-08-18 16:19:53,472 INFO     outcome=invalid       train_rmse=0.2107 (n=927)  val_rmse=0.3902 (n=232)
2026-08-18 16:19:53,472 INFO     outcome=timeout       train_rmse=0.6919 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:19:53,472 INFO     outcome=win           train_rmse=3.8702 (n=20720)  val_rmse=3.6430 (n=5145)
2026-08-18 16:20:01,755 INFO epoch   2/100  train_rmse=3.3850 (norm=2.4528)  val_rmse=2.9263 (norm=2.1204)
2026-08-18 16:20:01,755 INFO     opponent=immobile   train_rmse=3.3850 (n=21991)  val_rmse=2.9263 (n=5415)
2026-08-18 16:20:01,755 INFO     outcome=ball_out      train_rmse=3.4573 (n=143)  val_rmse=3.7158 (n=38)
2026-08-18 16:20:01,756 INFO     outcome=invalid       train_rmse=0.6156 (n=927)  val_rmse=1.0430 (n=232)
2026-08-18 16:20:01,756 INFO     outcome=timeout       train_rmse=1.0452 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:20:01,756 INFO     outcome=win           train_rmse=3.4715 (n=20720)  val_rmse=2.9768 (n=5145)
2026-08-18 16:20:09,493 INFO epoch   3/100  train_rmse=2.5488 (norm=1.8468)  val_rmse=1.7453 (norm=1.2647)
2026-08-18 16:20:09,493 INFO     opponent=immobile   train_rmse=2.5488 (n=21991)  val_rmse=1.7453 (n=5415)
2026-08-18 16:20:09,493 INFO     outcome=ball_out      train_rmse=4.3552 (n=143)  val_rmse=5.2132 (n=38)
2026-08-18 16:20:09,493 INFO     outcome=invalid       train_rmse=1.5988 (n=927)  val_rmse=2.6128 (n=232)
2026-08-18 16:20:09,493 INFO     outcome=timeout       train_rmse=2.0421 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:20:09,493 INFO     outcome=win           train_rmse=2.5708 (n=20720)  val_rmse=1.6424 (n=5145)
2026-08-18 16:20:17,499 INFO epoch   4/100  train_rmse=1.6896 (norm=1.2243)  val_rmse=1.7070 (norm=1.2369)
2026-08-18 16:20:17,499 INFO     opponent=immobile   train_rmse=1.6896 (n=21991)  val_rmse=1.7070 (n=5415)
2026-08-18 16:20:17,499 INFO     outcome=ball_out      train_rmse=6.4679 (n=143)  val_rmse=7.1453 (n=38)
2026-08-18 16:20:17,499 INFO     outcome=invalid       train_rmse=3.6781 (n=927)  val_rmse=4.6392 (n=232)
2026-08-18 16:20:17,499 INFO     outcome=timeout       train_rmse=4.1140 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:20:17,499 INFO     outcome=win           train_rmse=1.4042 (n=20720)  val_rmse=1.3112 (n=5145)
2026-08-18 16:20:25,086 INFO epoch   5/100  train_rmse=1.6155 (norm=1.1706)  val_rmse=1.4048 (norm=1.0179)
2026-08-18 16:20:25,086 INFO     opponent=immobile   train_rmse=1.6155 (n=21991)  val_rmse=1.4048 (n=5415)
2026-08-18 16:20:25,086 INFO     outcome=ball_out      train_rmse=7.2278 (n=143)  val_rmse=6.3828 (n=38)
2026-08-18 16:20:25,086 INFO     outcome=invalid       train_rmse=4.5136 (n=927)  val_rmse=3.7508 (n=232)
2026-08-18 16:20:25,087 INFO     outcome=timeout       train_rmse=4.6282 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:20:25,087 INFO     outcome=win           train_rmse=1.1358 (n=20720)  val_rmse=1.0685 (n=5145)
2026-08-18 16:20:32,788 INFO epoch   6/100  train_rmse=1.4427 (norm=1.0454)  val_rmse=1.3223 (norm=0.9582)
2026-08-18 16:20:32,788 INFO     opponent=immobile   train_rmse=1.4427 (n=21991)  val_rmse=1.3223 (n=5415)
2026-08-18 16:20:32,789 INFO     outcome=ball_out      train_rmse=6.5015 (n=143)  val_rmse=6.7982 (n=38)
2026-08-18 16:20:32,789 INFO     outcome=invalid       train_rmse=3.7508 (n=927)  val_rmse=3.8879 (n=232)
2026-08-18 16:20:32,789 INFO     outcome=timeout       train_rmse=3.8122 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:20:32,789 INFO     outcome=win           train_rmse=1.0709 (n=20720)  val_rmse=0.9041 (n=5145)
2026-08-18 16:20:41,012 INFO epoch   7/100  train_rmse=1.3646 (norm=0.9888)  val_rmse=1.2597 (norm=0.9128)
2026-08-18 16:20:41,012 INFO     opponent=immobile   train_rmse=1.3646 (n=21991)  val_rmse=1.2597 (n=5415)
2026-08-18 16:20:41,012 INFO     outcome=ball_out      train_rmse=6.6161 (n=143)  val_rmse=6.5680 (n=38)
2026-08-18 16:20:41,012 INFO     outcome=invalid       train_rmse=3.7967 (n=927)  val_rmse=3.3620 (n=232)
2026-08-18 16:20:41,012 INFO     outcome=timeout       train_rmse=3.9624 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:20:41,012 INFO     outcome=win           train_rmse=0.9365 (n=20720)  val_rmse=0.9175 (n=5145)
2026-08-18 16:20:48,994 INFO epoch   8/100  train_rmse=1.2886 (norm=0.9337)  val_rmse=1.1726 (norm=0.8497)
2026-08-18 16:20:48,994 INFO     opponent=immobile   train_rmse=1.2886 (n=21991)  val_rmse=1.1726 (n=5415)
2026-08-18 16:20:48,994 INFO     outcome=ball_out      train_rmse=6.4162 (n=143)  val_rmse=7.0267 (n=38)
2026-08-18 16:20:48,994 INFO     outcome=invalid       train_rmse=3.5809 (n=927)  val_rmse=3.2880 (n=232)
2026-08-18 16:20:48,994 INFO     outcome=timeout       train_rmse=3.6396 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:20:48,994 INFO     outcome=win           train_rmse=0.8809 (n=20720)  val_rmse=0.7714 (n=5145)
2026-08-18 16:20:57,042 INFO epoch   9/100  train_rmse=1.2054 (norm=0.8734)  val_rmse=1.1507 (norm=0.8338)
2026-08-18 16:20:57,042 INFO     opponent=immobile   train_rmse=1.2054 (n=21991)  val_rmse=1.1507 (n=5415)
2026-08-18 16:20:57,042 INFO     outcome=ball_out      train_rmse=6.1791 (n=143)  val_rmse=7.5111 (n=38)
2026-08-18 16:20:57,042 INFO     outcome=invalid       train_rmse=3.4912 (n=927)  val_rmse=3.0121 (n=232)
2026-08-18 16:20:57,042 INFO     outcome=timeout       train_rmse=3.1950 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:20:57,042 INFO     outcome=win           train_rmse=0.7964 (n=20720)  val_rmse=0.7535 (n=5145)
2026-08-18 16:21:05,195 INFO epoch  10/100  train_rmse=1.1674 (norm=0.8459)  val_rmse=1.1302 (norm=0.8190)
2026-08-18 16:21:05,195 INFO     opponent=immobile   train_rmse=1.1674 (n=21991)  val_rmse=1.1302 (n=5415)
2026-08-18 16:21:05,195 INFO     outcome=ball_out      train_rmse=5.9318 (n=143)  val_rmse=7.1412 (n=38)
2026-08-18 16:21:05,195 INFO     outcome=invalid       train_rmse=3.4700 (n=927)  val_rmse=2.9119 (n=232)
2026-08-18 16:21:05,195 INFO     outcome=timeout       train_rmse=2.7745 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:21:05,195 INFO     outcome=win           train_rmse=0.7683 (n=20720)  val_rmse=0.7652 (n=5145)
2026-08-18 16:21:13,197 INFO epoch  11/100  train_rmse=1.1384 (norm=0.8249)  val_rmse=1.1199 (norm=0.8115)
2026-08-18 16:21:13,197 INFO     opponent=immobile   train_rmse=1.1384 (n=21991)  val_rmse=1.1199 (n=5415)
2026-08-18 16:21:13,197 INFO     outcome=ball_out      train_rmse=5.8045 (n=143)  val_rmse=7.0635 (n=38)
2026-08-18 16:21:13,197 INFO     outcome=invalid       train_rmse=3.4109 (n=927)  val_rmse=2.8795 (n=232)
2026-08-18 16:21:13,197 INFO     outcome=timeout       train_rmse=2.7324 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:21:13,197 INFO     outcome=win           train_rmse=0.7416 (n=20720)  val_rmse=0.7600 (n=5145)
2026-08-18 16:21:21,186 INFO epoch  12/100  train_rmse=1.1186 (norm=0.8106)  val_rmse=1.1070 (norm=0.8022)
2026-08-18 16:21:21,187 INFO     opponent=immobile   train_rmse=1.1186 (n=21991)  val_rmse=1.1070 (n=5415)
2026-08-18 16:21:21,187 INFO     outcome=ball_out      train_rmse=5.6853 (n=143)  val_rmse=7.1912 (n=38)
2026-08-18 16:21:21,187 INFO     outcome=invalid       train_rmse=3.3196 (n=927)  val_rmse=2.9791 (n=232)
2026-08-18 16:21:21,187 INFO     outcome=timeout       train_rmse=2.6674 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:21:21,187 INFO     outcome=win           train_rmse=0.7369 (n=20720)  val_rmse=0.7125 (n=5145)
2026-08-18 16:21:29,279 INFO epoch  13/100  train_rmse=1.1084 (norm=0.8031)  val_rmse=1.1043 (norm=0.8002)
2026-08-18 16:21:29,279 INFO     opponent=immobile   train_rmse=1.1084 (n=21991)  val_rmse=1.1043 (n=5415)
2026-08-18 16:21:29,279 INFO     outcome=ball_out      train_rmse=5.5782 (n=143)  val_rmse=7.1448 (n=38)
2026-08-18 16:21:29,279 INFO     outcome=invalid       train_rmse=3.2516 (n=927)  val_rmse=2.9020 (n=232)
2026-08-18 16:21:29,279 INFO     outcome=timeout       train_rmse=2.6033 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:21:29,279 INFO     outcome=win           train_rmse=0.7418 (n=20720)  val_rmse=0.7257 (n=5145)
2026-08-18 16:21:37,020 INFO epoch  14/100  train_rmse=1.0955 (norm=0.7938)  val_rmse=1.0999 (norm=0.7970)
2026-08-18 16:21:37,020 INFO     opponent=immobile   train_rmse=1.0955 (n=21991)  val_rmse=1.0999 (n=5415)
2026-08-18 16:21:37,020 INFO     outcome=ball_out      train_rmse=5.4991 (n=143)  val_rmse=7.1487 (n=38)
2026-08-18 16:21:37,020 INFO     outcome=invalid       train_rmse=3.1967 (n=927)  val_rmse=2.9643 (n=232)
2026-08-18 16:21:37,020 INFO     outcome=timeout       train_rmse=2.5162 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:21:37,020 INFO     outcome=win           train_rmse=0.7392 (n=20720)  val_rmse=0.7068 (n=5145)
2026-08-18 16:21:44,913 INFO epoch  15/100  train_rmse=1.0872 (norm=0.7878)  val_rmse=1.1046 (norm=0.8004)
2026-08-18 16:21:44,913 INFO     opponent=immobile   train_rmse=1.0872 (n=21991)  val_rmse=1.1046 (n=5415)
2026-08-18 16:21:44,913 INFO     outcome=ball_out      train_rmse=5.5225 (n=143)  val_rmse=6.9548 (n=38)
2026-08-18 16:21:44,913 INFO     outcome=invalid       train_rmse=3.2330 (n=927)  val_rmse=2.7165 (n=232)
2026-08-18 16:21:44,914 INFO     outcome=timeout       train_rmse=2.5226 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:21:44,914 INFO     outcome=win           train_rmse=0.7175 (n=20720)  val_rmse=0.7709 (n=5145)
2026-08-18 16:21:52,884 INFO epoch  16/100  train_rmse=1.0779 (norm=0.7811)  val_rmse=1.0854 (norm=0.7865)
2026-08-18 16:21:52,884 INFO     opponent=immobile   train_rmse=1.0779 (n=21991)  val_rmse=1.0854 (n=5415)
2026-08-18 16:21:52,884 INFO     outcome=ball_out      train_rmse=5.3263 (n=143)  val_rmse=6.9393 (n=38)
2026-08-18 16:21:52,884 INFO     outcome=invalid       train_rmse=3.1480 (n=927)  val_rmse=2.8087 (n=232)
2026-08-18 16:21:52,884 INFO     outcome=timeout       train_rmse=2.4501 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:21:52,884 INFO     outcome=win           train_rmse=0.7320 (n=20720)  val_rmse=0.7270 (n=5145)
2026-08-18 16:22:01,050 INFO epoch  17/100  train_rmse=1.0675 (norm=0.7735)  val_rmse=1.0983 (norm=0.7959)
2026-08-18 16:22:01,050 INFO     opponent=immobile   train_rmse=1.0675 (n=21991)  val_rmse=1.0983 (n=5415)
2026-08-18 16:22:01,050 INFO     outcome=ball_out      train_rmse=5.3710 (n=143)  val_rmse=6.8424 (n=38)
2026-08-18 16:22:01,050 INFO     outcome=invalid       train_rmse=3.1379 (n=927)  val_rmse=2.6735 (n=232)
2026-08-18 16:22:01,050 INFO     outcome=timeout       train_rmse=2.4577 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:22:01,050 INFO     outcome=win           train_rmse=0.7150 (n=20720)  val_rmse=0.7756 (n=5145)
2026-08-18 16:22:09,248 INFO epoch  18/100  train_rmse=1.0563 (norm=0.7654)  val_rmse=1.0918 (norm=0.7911)
2026-08-18 16:22:09,248 INFO     opponent=immobile   train_rmse=1.0563 (n=21991)  val_rmse=1.0918 (n=5415)
2026-08-18 16:22:09,248 INFO     outcome=ball_out      train_rmse=5.2463 (n=143)  val_rmse=6.8350 (n=38)
2026-08-18 16:22:09,248 INFO     outcome=invalid       train_rmse=3.1302 (n=927)  val_rmse=2.6161 (n=232)
2026-08-18 16:22:09,248 INFO     outcome=timeout       train_rmse=2.3217 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:22:09,248 INFO     outcome=win           train_rmse=0.7097 (n=20720)  val_rmse=0.7752 (n=5145)
2026-08-18 16:22:17,376 INFO epoch  19/100  train_rmse=1.0468 (norm=0.7585)  val_rmse=1.0769 (norm=0.7803)
2026-08-18 16:22:17,376 INFO     opponent=immobile   train_rmse=1.0468 (n=21991)  val_rmse=1.0769 (n=5415)
2026-08-18 16:22:17,376 INFO     outcome=ball_out      train_rmse=5.1364 (n=143)  val_rmse=6.8474 (n=38)
2026-08-18 16:22:17,376 INFO     outcome=invalid       train_rmse=3.0148 (n=927)  val_rmse=2.7427 (n=232)
2026-08-18 16:22:17,376 INFO     outcome=timeout       train_rmse=2.3267 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:22:17,376 INFO     outcome=win           train_rmse=0.7223 (n=20720)  val_rmse=0.7315 (n=5145)
2026-08-18 16:22:25,690 INFO epoch  20/100  train_rmse=1.0339 (norm=0.7492)  val_rmse=1.0796 (norm=0.7823)
2026-08-18 16:22:25,690 INFO     opponent=immobile   train_rmse=1.0339 (n=21991)  val_rmse=1.0796 (n=5415)
2026-08-18 16:22:25,690 INFO     outcome=ball_out      train_rmse=5.0771 (n=143)  val_rmse=6.9141 (n=38)
2026-08-18 16:22:25,690 INFO     outcome=invalid       train_rmse=3.0040 (n=927)  val_rmse=2.7599 (n=232)
2026-08-18 16:22:25,690 INFO     outcome=timeout       train_rmse=2.2783 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:22:25,690 INFO     outcome=win           train_rmse=0.7090 (n=20720)  val_rmse=0.7280 (n=5145)
2026-08-18 16:22:33,632 INFO epoch  21/100  train_rmse=1.0343 (norm=0.7495)  val_rmse=1.0730 (norm=0.7775)
2026-08-18 16:22:33,632 INFO     opponent=immobile   train_rmse=1.0343 (n=21991)  val_rmse=1.0730 (n=5415)
2026-08-18 16:22:33,632 INFO     outcome=ball_out      train_rmse=5.0500 (n=143)  val_rmse=6.8502 (n=38)
2026-08-18 16:22:33,632 INFO     outcome=invalid       train_rmse=3.0538 (n=927)  val_rmse=2.7315 (n=232)
2026-08-18 16:22:33,632 INFO     outcome=timeout       train_rmse=2.2117 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:22:33,632 INFO     outcome=win           train_rmse=0.7034 (n=20720)  val_rmse=0.7271 (n=5145)
2026-08-18 16:22:41,694 INFO epoch  22/100  train_rmse=1.0184 (norm=0.7379)  val_rmse=1.0778 (norm=0.7810)
2026-08-18 16:22:41,694 INFO     opponent=immobile   train_rmse=1.0184 (n=21991)  val_rmse=1.0778 (n=5415)
2026-08-18 16:22:41,694 INFO     outcome=ball_out      train_rmse=4.8699 (n=143)  val_rmse=6.6545 (n=38)
2026-08-18 16:22:41,694 INFO     outcome=invalid       train_rmse=2.8761 (n=927)  val_rmse=2.5309 (n=232)
2026-08-18 16:22:41,694 INFO     outcome=timeout       train_rmse=2.1769 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:22:41,694 INFO     outcome=win           train_rmse=0.7218 (n=20720)  val_rmse=0.7790 (n=5145)
2026-08-18 16:22:49,826 INFO epoch  23/100  train_rmse=1.0097 (norm=0.7316)  val_rmse=1.0686 (norm=0.7743)
2026-08-18 16:22:49,826 INFO     opponent=immobile   train_rmse=1.0097 (n=21991)  val_rmse=1.0686 (n=5415)
2026-08-18 16:22:49,826 INFO     outcome=ball_out      train_rmse=4.9278 (n=143)  val_rmse=6.6194 (n=38)
2026-08-18 16:22:49,826 INFO     outcome=invalid       train_rmse=2.9490 (n=927)  val_rmse=2.5651 (n=232)
2026-08-18 16:22:49,826 INFO     outcome=timeout       train_rmse=2.1691 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:22:49,826 INFO     outcome=win           train_rmse=0.6926 (n=20720)  val_rmse=0.7625 (n=5145)
2026-08-18 16:22:57,848 INFO epoch  24/100  train_rmse=1.0008 (norm=0.7252)  val_rmse=1.0687 (norm=0.7744)
2026-08-18 16:22:57,848 INFO     opponent=immobile   train_rmse=1.0008 (n=21991)  val_rmse=1.0687 (n=5415)
2026-08-18 16:22:57,848 INFO     outcome=ball_out      train_rmse=4.8272 (n=143)  val_rmse=6.7369 (n=38)
2026-08-18 16:22:57,848 INFO     outcome=invalid       train_rmse=2.8661 (n=927)  val_rmse=2.6293 (n=232)
2026-08-18 16:22:57,848 INFO     outcome=timeout       train_rmse=2.1316 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:22:57,848 INFO     outcome=win           train_rmse=0.7005 (n=20720)  val_rmse=0.7451 (n=5145)
2026-08-18 16:23:05,216 INFO epoch  25/100  train_rmse=0.9954 (norm=0.7213)  val_rmse=1.0685 (norm=0.7742)
2026-08-18 16:23:05,216 INFO     opponent=immobile   train_rmse=0.9954 (n=21991)  val_rmse=1.0685 (n=5415)
2026-08-18 16:23:05,216 INFO     outcome=ball_out      train_rmse=4.7167 (n=143)  val_rmse=6.6086 (n=38)
2026-08-18 16:23:05,216 INFO     outcome=invalid       train_rmse=2.8395 (n=927)  val_rmse=2.5156 (n=232)
2026-08-18 16:23:05,216 INFO     outcome=timeout       train_rmse=2.0705 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:23:05,216 INFO     outcome=win           train_rmse=0.7041 (n=20720)  val_rmse=0.7705 (n=5145)
2026-08-18 16:23:12,827 INFO epoch  26/100  train_rmse=0.9862 (norm=0.7146)  val_rmse=1.0721 (norm=0.7769)
2026-08-18 16:23:12,827 INFO     opponent=immobile   train_rmse=0.9862 (n=21991)  val_rmse=1.0721 (n=5415)
2026-08-18 16:23:12,827 INFO     outcome=ball_out      train_rmse=4.7114 (n=143)  val_rmse=6.6474 (n=38)
2026-08-18 16:23:12,827 INFO     outcome=invalid       train_rmse=2.8043 (n=927)  val_rmse=2.5529 (n=232)
2026-08-18 16:23:12,827 INFO     outcome=timeout       train_rmse=2.0589 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:23:12,827 INFO     outcome=win           train_rmse=0.6973 (n=20720)  val_rmse=0.7678 (n=5145)
2026-08-18 16:23:20,717 INFO epoch  27/100  train_rmse=0.9770 (norm=0.7079)  val_rmse=1.0647 (norm=0.7715)
2026-08-18 16:23:20,717 INFO     opponent=immobile   train_rmse=0.9770 (n=21991)  val_rmse=1.0647 (n=5415)
2026-08-18 16:23:20,717 INFO     outcome=ball_out      train_rmse=4.6868 (n=143)  val_rmse=6.7689 (n=38)
2026-08-18 16:23:20,717 INFO     outcome=invalid       train_rmse=2.7865 (n=927)  val_rmse=2.6298 (n=232)
2026-08-18 16:23:20,717 INFO     outcome=timeout       train_rmse=2.0734 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:23:20,717 INFO     outcome=win           train_rmse=0.6873 (n=20720)  val_rmse=0.7367 (n=5145)
2026-08-18 16:23:28,650 INFO epoch  28/100  train_rmse=0.9688 (norm=0.7020)  val_rmse=1.0837 (norm=0.7852)
2026-08-18 16:23:28,650 INFO     opponent=immobile   train_rmse=0.9688 (n=21991)  val_rmse=1.0837 (n=5415)
2026-08-18 16:23:28,650 INFO     outcome=ball_out      train_rmse=4.6475 (n=143)  val_rmse=6.6534 (n=38)
2026-08-18 16:23:28,650 INFO     outcome=invalid       train_rmse=2.7723 (n=927)  val_rmse=2.4655 (n=232)
2026-08-18 16:23:28,650 INFO     outcome=timeout       train_rmse=2.0330 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:23:28,650 INFO     outcome=win           train_rmse=0.6805 (n=20720)  val_rmse=0.7968 (n=5145)
2026-08-18 16:23:36,685 INFO epoch  29/100  train_rmse=0.9691 (norm=0.7022)  val_rmse=1.0606 (norm=0.7685)
2026-08-18 16:23:36,685 INFO     opponent=immobile   train_rmse=0.9691 (n=21991)  val_rmse=1.0606 (n=5415)
2026-08-18 16:23:36,685 INFO     outcome=ball_out      train_rmse=4.4575 (n=143)  val_rmse=6.5419 (n=38)
2026-08-18 16:23:36,685 INFO     outcome=invalid       train_rmse=2.6583 (n=927)  val_rmse=2.5586 (n=232)
2026-08-18 16:23:36,685 INFO     outcome=timeout       train_rmse=1.9783 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:23:36,685 INFO     outcome=win           train_rmse=0.7110 (n=20720)  val_rmse=0.7567 (n=5145)
2026-08-18 16:23:44,387 INFO epoch  30/100  train_rmse=0.9575 (norm=0.6938)  val_rmse=1.0850 (norm=0.7862)
2026-08-18 16:23:44,387 INFO     opponent=immobile   train_rmse=0.9575 (n=21991)  val_rmse=1.0850 (n=5415)
2026-08-18 16:23:44,387 INFO     outcome=ball_out      train_rmse=4.4872 (n=143)  val_rmse=6.5537 (n=38)
2026-08-18 16:23:44,387 INFO     outcome=invalid       train_rmse=2.6937 (n=927)  val_rmse=2.4188 (n=232)
2026-08-18 16:23:44,387 INFO     outcome=timeout       train_rmse=2.0214 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:23:44,387 INFO     outcome=win           train_rmse=0.6854 (n=20720)  val_rmse=0.8112 (n=5145)
2026-08-18 16:23:52,462 INFO epoch  31/100  train_rmse=0.9572 (norm=0.6936)  val_rmse=1.0731 (norm=0.7776)
2026-08-18 16:23:52,462 INFO     opponent=immobile   train_rmse=0.9572 (n=21991)  val_rmse=1.0731 (n=5415)
2026-08-18 16:23:52,462 INFO     outcome=ball_out      train_rmse=4.3916 (n=143)  val_rmse=6.3934 (n=38)
2026-08-18 16:23:52,462 INFO     outcome=invalid       train_rmse=2.6014 (n=927)  val_rmse=2.4564 (n=232)
2026-08-18 16:23:52,462 INFO     outcome=timeout       train_rmse=1.9722 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:23:52,462 INFO     outcome=win           train_rmse=0.7062 (n=20720)  val_rmse=0.7988 (n=5145)
2026-08-18 16:24:00,264 INFO epoch  32/100  train_rmse=0.9569 (norm=0.6934)  val_rmse=1.0572 (norm=0.7661)
2026-08-18 16:24:00,264 INFO     opponent=immobile   train_rmse=0.9569 (n=21991)  val_rmse=1.0572 (n=5415)
2026-08-18 16:24:00,264 INFO     outcome=ball_out      train_rmse=4.4739 (n=143)  val_rmse=6.5267 (n=38)
2026-08-18 16:24:00,264 INFO     outcome=invalid       train_rmse=2.6995 (n=927)  val_rmse=2.4576 (n=232)
2026-08-18 16:24:00,264 INFO     outcome=timeout       train_rmse=1.9943 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:24:00,264 INFO     outcome=win           train_rmse=0.6849 (n=20720)  val_rmse=0.7677 (n=5145)
2026-08-18 16:24:08,058 INFO epoch  33/100  train_rmse=0.9488 (norm=0.6875)  val_rmse=1.0731 (norm=0.7776)
2026-08-18 16:24:08,058 INFO     opponent=immobile   train_rmse=0.9488 (n=21991)  val_rmse=1.0731 (n=5415)
2026-08-18 16:24:08,058 INFO     outcome=ball_out      train_rmse=4.3055 (n=143)  val_rmse=6.6568 (n=38)
2026-08-18 16:24:08,058 INFO     outcome=invalid       train_rmse=2.5952 (n=927)  val_rmse=2.5622 (n=232)
2026-08-18 16:24:08,058 INFO     outcome=timeout       train_rmse=1.9471 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:24:08,058 INFO     outcome=win           train_rmse=0.6996 (n=20720)  val_rmse=0.7672 (n=5145)
2026-08-18 16:24:16,029 INFO epoch  34/100  train_rmse=0.9419 (norm=0.6825)  val_rmse=1.0666 (norm=0.7729)
2026-08-18 16:24:16,029 INFO     opponent=immobile   train_rmse=0.9419 (n=21991)  val_rmse=1.0666 (n=5415)
2026-08-18 16:24:16,030 INFO     outcome=ball_out      train_rmse=4.3038 (n=143)  val_rmse=6.7008 (n=38)
2026-08-18 16:24:16,030 INFO     outcome=invalid       train_rmse=2.5792 (n=927)  val_rmse=2.6430 (n=232)
2026-08-18 16:24:16,030 INFO     outcome=timeout       train_rmse=1.9404 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:24:16,030 INFO     outcome=win           train_rmse=0.6926 (n=20720)  val_rmse=0.7421 (n=5145)
2026-08-18 16:24:23,891 INFO epoch  35/100  train_rmse=0.9473 (norm=0.6864)  val_rmse=1.0598 (norm=0.7680)
2026-08-18 16:24:23,891 INFO     opponent=immobile   train_rmse=0.9473 (n=21991)  val_rmse=1.0598 (n=5415)
2026-08-18 16:24:23,891 INFO     outcome=ball_out      train_rmse=4.3190 (n=143)  val_rmse=6.6567 (n=38)
2026-08-18 16:24:23,891 INFO     outcome=invalid       train_rmse=2.5573 (n=927)  val_rmse=2.5824 (n=232)
2026-08-18 16:24:23,891 INFO     outcome=timeout       train_rmse=2.0615 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:24:23,891 INFO     outcome=win           train_rmse=0.6999 (n=20720)  val_rmse=0.7444 (n=5145)
2026-08-18 16:24:31,802 INFO epoch  36/100  train_rmse=0.9338 (norm=0.6766)  val_rmse=1.0599 (norm=0.7680)
2026-08-18 16:24:31,802 INFO     opponent=immobile   train_rmse=0.9338 (n=21991)  val_rmse=1.0599 (n=5415)
2026-08-18 16:24:31,802 INFO     outcome=ball_out      train_rmse=4.2783 (n=143)  val_rmse=6.6770 (n=38)
2026-08-18 16:24:31,802 INFO     outcome=invalid       train_rmse=2.5589 (n=927)  val_rmse=2.5752 (n=232)
2026-08-18 16:24:31,802 INFO     outcome=timeout       train_rmse=1.9529 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:24:31,802 INFO     outcome=win           train_rmse=0.6850 (n=20720)  val_rmse=0.7443 (n=5145)
2026-08-18 16:24:39,852 INFO epoch  37/100  train_rmse=0.9214 (norm=0.6676)  val_rmse=1.0667 (norm=0.7729)
2026-08-18 16:24:39,852 INFO     opponent=immobile   train_rmse=0.9214 (n=21991)  val_rmse=1.0667 (n=5415)
2026-08-18 16:24:39,852 INFO     outcome=ball_out      train_rmse=4.1819 (n=143)  val_rmse=6.5038 (n=38)
2026-08-18 16:24:39,852 INFO     outcome=invalid       train_rmse=2.5095 (n=927)  val_rmse=2.3903 (n=232)
2026-08-18 16:24:39,852 INFO     outcome=timeout       train_rmse=1.9154 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:24:39,852 INFO     outcome=win           train_rmse=0.6804 (n=20720)  val_rmse=0.7921 (n=5145)
2026-08-18 16:24:47,824 INFO epoch  38/100  train_rmse=0.9164 (norm=0.6641)  val_rmse=1.0813 (norm=0.7835)
2026-08-18 16:24:47,824 INFO     opponent=immobile   train_rmse=0.9164 (n=21991)  val_rmse=1.0813 (n=5415)
2026-08-18 16:24:47,824 INFO     outcome=ball_out      train_rmse=4.1927 (n=143)  val_rmse=6.5833 (n=38)
2026-08-18 16:24:47,824 INFO     outcome=invalid       train_rmse=2.4584 (n=927)  val_rmse=2.5449 (n=232)
2026-08-18 16:24:47,824 INFO     outcome=timeout       train_rmse=1.9543 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:24:47,824 INFO     outcome=win           train_rmse=0.6802 (n=20720)  val_rmse=0.7864 (n=5145)
2026-08-18 16:24:55,959 INFO epoch  39/100  train_rmse=0.9137 (norm=0.6621)  val_rmse=1.0549 (norm=0.7644)
2026-08-18 16:24:55,959 INFO     opponent=immobile   train_rmse=0.9137 (n=21991)  val_rmse=1.0549 (n=5415)
2026-08-18 16:24:55,959 INFO     outcome=ball_out      train_rmse=4.0825 (n=143)  val_rmse=6.3153 (n=38)
2026-08-18 16:24:55,959 INFO     outcome=invalid       train_rmse=2.4644 (n=927)  val_rmse=2.4613 (n=232)
2026-08-18 16:24:55,959 INFO     outcome=timeout       train_rmse=1.9101 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:24:55,959 INFO     outcome=win           train_rmse=0.6811 (n=20720)  val_rmse=0.7769 (n=5145)
2026-08-18 16:25:04,084 INFO epoch  40/100  train_rmse=0.9091 (norm=0.6587)  val_rmse=1.0800 (norm=0.7826)
2026-08-18 16:25:04,084 INFO     opponent=immobile   train_rmse=0.9091 (n=21991)  val_rmse=1.0800 (n=5415)
2026-08-18 16:25:04,084 INFO     outcome=ball_out      train_rmse=4.1154 (n=143)  val_rmse=6.6425 (n=38)
2026-08-18 16:25:04,084 INFO     outcome=invalid       train_rmse=2.4174 (n=927)  val_rmse=2.5347 (n=232)
2026-08-18 16:25:04,084 INFO     outcome=timeout       train_rmse=1.9727 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:25:04,084 INFO     outcome=win           train_rmse=0.6790 (n=20720)  val_rmse=0.7824 (n=5145)
2026-08-18 16:25:12,492 INFO epoch  41/100  train_rmse=0.8988 (norm=0.6513)  val_rmse=1.0606 (norm=0.7685)
2026-08-18 16:25:12,492 INFO     opponent=immobile   train_rmse=0.8988 (n=21991)  val_rmse=1.0606 (n=5415)
2026-08-18 16:25:12,492 INFO     outcome=ball_out      train_rmse=3.9676 (n=143)  val_rmse=6.6441 (n=38)
2026-08-18 16:25:12,492 INFO     outcome=invalid       train_rmse=2.3941 (n=927)  val_rmse=2.5751 (n=232)
2026-08-18 16:25:12,492 INFO     outcome=timeout       train_rmse=1.8818 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:25:12,492 INFO     outcome=win           train_rmse=0.6768 (n=20720)  val_rmse=0.7476 (n=5145)
2026-08-18 16:25:20,466 INFO epoch  42/100  train_rmse=0.8897 (norm=0.6447)  val_rmse=1.0719 (norm=0.7767)
2026-08-18 16:25:20,467 INFO     opponent=immobile   train_rmse=0.8897 (n=21991)  val_rmse=1.0719 (n=5415)
2026-08-18 16:25:20,467 INFO     outcome=ball_out      train_rmse=3.9294 (n=143)  val_rmse=6.4530 (n=38)
2026-08-18 16:25:20,467 INFO     outcome=invalid       train_rmse=2.3533 (n=927)  val_rmse=2.4344 (n=232)
2026-08-18 16:25:20,467 INFO     outcome=timeout       train_rmse=1.9107 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:25:20,467 INFO     outcome=win           train_rmse=0.6712 (n=20720)  val_rmse=0.7965 (n=5145)
2026-08-18 16:25:28,416 INFO epoch  43/100  train_rmse=0.8829 (norm=0.6398)  val_rmse=1.0841 (norm=0.7856)
2026-08-18 16:25:28,416 INFO     opponent=immobile   train_rmse=0.8829 (n=21991)  val_rmse=1.0841 (n=5415)
2026-08-18 16:25:28,416 INFO     outcome=ball_out      train_rmse=3.9324 (n=143)  val_rmse=6.3213 (n=38)
2026-08-18 16:25:28,416 INFO     outcome=invalid       train_rmse=2.3493 (n=927)  val_rmse=2.3525 (n=232)
2026-08-18 16:25:28,416 INFO     outcome=timeout       train_rmse=1.9010 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:25:28,416 INFO     outcome=win           train_rmse=0.6623 (n=20720)  val_rmse=0.8321 (n=5145)
2026-08-18 16:25:36,381 INFO epoch  44/100  train_rmse=0.8838 (norm=0.6404)  val_rmse=1.0675 (norm=0.7735)
2026-08-18 16:25:36,381 INFO     opponent=immobile   train_rmse=0.8838 (n=21991)  val_rmse=1.0675 (n=5415)
2026-08-18 16:25:36,381 INFO     outcome=ball_out      train_rmse=3.7735 (n=143)  val_rmse=6.5410 (n=38)
2026-08-18 16:25:36,381 INFO     outcome=invalid       train_rmse=2.2821 (n=927)  val_rmse=2.5839 (n=232)
2026-08-18 16:25:36,381 INFO     outcome=timeout       train_rmse=1.8605 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:25:36,381 INFO     outcome=win           train_rmse=0.6813 (n=20720)  val_rmse=0.7630 (n=5145)
2026-08-18 16:25:44,292 INFO epoch  45/100  train_rmse=0.8780 (norm=0.6362)  val_rmse=1.0769 (norm=0.7804)
2026-08-18 16:25:44,292 INFO     opponent=immobile   train_rmse=0.8780 (n=21991)  val_rmse=1.0769 (n=5415)
2026-08-18 16:25:44,292 INFO     outcome=ball_out      train_rmse=3.7648 (n=143)  val_rmse=6.6707 (n=38)
2026-08-18 16:25:44,292 INFO     outcome=invalid       train_rmse=2.2635 (n=927)  val_rmse=2.6316 (n=232)
2026-08-18 16:25:44,292 INFO     outcome=timeout       train_rmse=1.8328 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:25:44,292 INFO     outcome=win           train_rmse=0.6771 (n=20720)  val_rmse=0.7614 (n=5145)
2026-08-18 16:25:52,283 INFO epoch  46/100  train_rmse=0.8767 (norm=0.6353)  val_rmse=1.0791 (norm=0.7819)
2026-08-18 16:25:52,283 INFO     opponent=immobile   train_rmse=0.8767 (n=21991)  val_rmse=1.0791 (n=5415)
2026-08-18 16:25:52,283 INFO     outcome=ball_out      train_rmse=3.7638 (n=143)  val_rmse=6.4750 (n=38)
2026-08-18 16:25:52,283 INFO     outcome=invalid       train_rmse=2.3415 (n=927)  val_rmse=2.4072 (n=232)
2026-08-18 16:25:52,283 INFO     outcome=timeout       train_rmse=1.8523 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:25:52,283 INFO     outcome=win           train_rmse=0.6629 (n=20720)  val_rmse=0.8091 (n=5145)
2026-08-18 16:26:00,390 INFO epoch  47/100  train_rmse=0.8679 (norm=0.6289)  val_rmse=1.0808 (norm=0.7832)
2026-08-18 16:26:00,390 INFO     opponent=immobile   train_rmse=0.8679 (n=21991)  val_rmse=1.0808 (n=5415)
2026-08-18 16:26:00,390 INFO     outcome=ball_out      train_rmse=3.6697 (n=143)  val_rmse=6.4837 (n=38)
2026-08-18 16:26:00,390 INFO     outcome=invalid       train_rmse=2.2590 (n=927)  val_rmse=2.4454 (n=232)
2026-08-18 16:26:00,390 INFO     outcome=timeout       train_rmse=1.8032 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:26:00,390 INFO     outcome=win           train_rmse=0.6684 (n=20720)  val_rmse=0.8058 (n=5145)
2026-08-18 16:26:07,960 INFO epoch  48/100  train_rmse=0.8612 (norm=0.6240)  val_rmse=1.0798 (norm=0.7824)
2026-08-18 16:26:07,960 INFO     opponent=immobile   train_rmse=0.8612 (n=21991)  val_rmse=1.0798 (n=5415)
2026-08-18 16:26:07,960 INFO     outcome=ball_out      train_rmse=3.6470 (n=143)  val_rmse=6.6670 (n=38)
2026-08-18 16:26:07,961 INFO     outcome=invalid       train_rmse=2.1712 (n=927)  val_rmse=2.6817 (n=232)
2026-08-18 16:26:07,961 INFO     outcome=timeout       train_rmse=1.8610 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:26:07,961 INFO     outcome=win           train_rmse=0.6714 (n=20720)  val_rmse=0.7580 (n=5145)
2026-08-18 16:26:15,569 INFO epoch  49/100  train_rmse=0.8582 (norm=0.6219)  val_rmse=1.0947 (norm=0.7932)
2026-08-18 16:26:15,570 INFO     opponent=immobile   train_rmse=0.8582 (n=21991)  val_rmse=1.0947 (n=5415)
2026-08-18 16:26:15,570 INFO     outcome=ball_out      train_rmse=3.6516 (n=143)  val_rmse=6.4687 (n=38)
2026-08-18 16:26:15,570 INFO     outcome=invalid       train_rmse=2.2599 (n=927)  val_rmse=2.4152 (n=232)
2026-08-18 16:26:15,570 INFO     outcome=timeout       train_rmse=1.8973 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:26:15,570 INFO     outcome=win           train_rmse=0.6529 (n=20720)  val_rmse=0.8302 (n=5145)
2026-08-18 16:26:23,312 INFO epoch  50/100  train_rmse=0.8527 (norm=0.6179)  val_rmse=1.0849 (norm=0.7861)
2026-08-18 16:26:23,312 INFO     opponent=immobile   train_rmse=0.8527 (n=21991)  val_rmse=1.0849 (n=5415)
2026-08-18 16:26:23,312 INFO     outcome=ball_out      train_rmse=3.4604 (n=143)  val_rmse=6.5200 (n=38)
2026-08-18 16:26:23,312 INFO     outcome=invalid       train_rmse=2.1124 (n=927)  val_rmse=2.3976 (n=232)
2026-08-18 16:26:23,312 INFO     outcome=timeout       train_rmse=1.7293 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:26:23,312 INFO     outcome=win           train_rmse=0.6785 (n=20720)  val_rmse=0.8158 (n=5145)
2026-08-18 16:26:30,959 INFO epoch  51/100  train_rmse=0.8493 (norm=0.6154)  val_rmse=1.0807 (norm=0.7831)
2026-08-18 16:26:30,959 INFO     opponent=immobile   train_rmse=0.8493 (n=21991)  val_rmse=1.0807 (n=5415)
2026-08-18 16:26:30,959 INFO     outcome=ball_out      train_rmse=3.5895 (n=143)  val_rmse=6.8104 (n=38)
2026-08-18 16:26:30,959 INFO     outcome=invalid       train_rmse=2.1725 (n=927)  val_rmse=2.7568 (n=232)
2026-08-18 16:26:30,959 INFO     outcome=timeout       train_rmse=1.8875 (n=201)  val_rmse=nan (n=0)
2026-08-18 16:26:30,959 INFO     outcome=win           train_rmse=0.6565 (n=20720)  val_rmse=0.7376 (n=5145)
2026-08-18 16:26:30,959 INFO Early stopping at epoch 51/100 (val normalized MSE did not improve for 12 epochs).
2026-08-18 16:26:30,959 INFO Best val normalized MSE achieved: 0.5843 (RMSE=0.7644; <1.0 = better than predicting the mean; <0.5 = useful critic)
2026-08-18 16:26:31,880 INFO --- Per-component MC-return magnitude (val rows) ---
2026-08-18 16:26:31,880 INFO   get_possession    mean=+0.4692  std=0.4761
2026-08-18 16:26:31,880 INFO   lose_possession   mean=-0.0058  std=0.0696
2026-08-18 16:26:31,880 INFO   ball_out          mean=-0.0259  std=0.3085
2026-08-18 16:26:31,880 INFO   box_possession    mean=+1.6729  std=0.4149
2026-08-18 16:26:31,880 INFO   speed_bonus       mean=+1.6935  std=0.8001
2026-08-18 16:26:31,880 INFO   opponent_box      mean=-2.0912  std=0.5186
2026-08-18 16:26:31,880 INFO   stamina_penalty   mean=-0.1107  std=0.0405
2026-08-18 16:26:32,161 INFO --- Reward-component vs. value-residual correlation (200 val episodes) ---
2026-08-18 16:26:32,162 INFO   component            corr   comp_std
2026-08-18 16:26:32,163 INFO   lose_possession    +0.066     0.0777
2026-08-18 16:26:32,163 INFO   ball_out           +0.466     0.3455
2026-08-18 16:26:32,163 INFO   speed_bonus        +0.607     0.9274
2026-08-18 16:26:32,163 INFO   get_possession     +0.641     0.2786
2026-08-18 16:26:32,163 INFO   stamina_penalty    -0.660     0.0417
2026-08-18 16:26:32,163 INFO   box_possession     +0.791     0.5054
2026-08-18 16:26:32,163 INFO   opponent_box       -0.791     0.6318
2026-08-18 16:26:32,163 INFO   (components near the top -- low |corr| despite real variance -- are the ones the value net's errors track least; read alongside the per-component MC-return magnitude above.)
2026-08-18 16:26:32,174 INFO --- Worst val episode for outcome=ball_out (2 episode(s)): rows [46751, 46801], residual=-7.392 -- saved match log to results/debug_value_worst_episode_ball_out.json ---
2026-08-18 16:26:32,176 INFO --- Worst val episode for outcome=invalid (18 episode(s)): rows [45010, 45034], residual=-5.035 -- saved match log to results/debug_value_worst_episode_invalid.json ---
2026-08-18 16:26:32,178 INFO --- Worst val episode for outcome=win (180 episode(s)): rows [52963, 53083], residual=-2.462 -- saved match log to results/debug_value_worst_episode_win.json ---
2026-08-18 17:09:04,457 WARNING --checkpoint and --data both given: using --data as the dataset (no live rollout will be collected) and --checkpoint only to initialize decision_net/value_net's weights -- i.e. warm-starting value fitting from a trained checkpoint against a pre-recorded dataset instead of a fresh rollout.
2026-08-18 17:09:04,602 INFO Checkpoint checkpoints/longterm/checkpoint_vvgood_immobile.pt: separate_value_net=True (auto-detected)
2026-08-18 17:09:04,642 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 17:09:04,644 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 17:09:04,644 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 17:09:04,648 INFO --reset-dir-log-std: reset move_dir_log_std/kick_dir_log_std to config init values before rollout collection.
2026-08-18 17:09:04,649 INFO Loading 46 demonstration file(s) from demonstrations/phase1_neural_vvgood
2026-08-18 17:09:17,789 INFO Dataset: 1,194,878 steps loaded
2026-08-18 17:09:17,789 INFO Loaded 1,194,878 rows total
2026-08-18 17:09:17,790 INFO has_rewards=True
2026-08-18 17:09:17,818 INFO valid_indices(): 597,733 rows (50.0% of total)
2026-08-18 17:09:18,451 INFO Returns over ALL rows: mean=0.807 std=3.099 min=-5.129 max=11.631
2026-08-18 17:09:18,453 INFO Returns over valid_indices(): mean=3.739 std=1.306
2026-08-18 17:09:19,249 INFO --- Dataset distribution (1,194,878 rows, 21500 episodes) ---
2026-08-18 17:09:19,259 INFO   self.ai_type == rules: 0.0%
2026-08-18 17:09:19,272 INFO   self.ai_type == immobile: 50.0%
2026-08-18 17:09:19,285 INFO   self.ai_type == neural: 50.0%
2026-08-18 17:09:19,293 INFO   opponent.ai_type == rules: 0.0%
2026-08-18 17:09:19,303 INFO   opponent.ai_type == immobile: 50.0%
2026-08-18 17:09:19,310 INFO   opponent.ai_type == neural: 50.0%
2026-08-18 17:09:19,334 INFO   valid rows: kick_this_tick rate=0.00%  tackle_attempt rate=0.71%
2026-08-18 17:09:19,336 INFO   dones=1 rows: 43,000  |  zero-reward rows: 1,136,210 (95.1%)
2026-08-18 17:09:19,358 INFO   return percentiles (valid rows): p10=2.57  p50=3.83  p90=5.15
2026-08-18 17:09:20,341 INFO --- Reward component breakdown (all episodes, 21500 episode(s)) ---
2026-08-18 17:09:20,341 INFO   component           mean      std       min       max
2026-08-18 17:09:20,347 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:20,351 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:20,357 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:20,363 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:20,372 INFO   get_possession    +0.948    0.338    +0.000    +5.000
2026-08-18 17:09:20,378 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:20,384 INFO   lose_possession    -0.022    0.181    -3.600    +0.000
2026-08-18 17:09:20,392 INFO   ball_out          -0.029    0.342    -4.000    +0.000
2026-08-18 17:09:20,397 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:20,405 INFO   box_possession    +1.830    0.558    +0.000    +2.000
2026-08-18 17:09:20,411 INFO   speed_bonus       +2.095    0.929    -0.015    +3.985
2026-08-18 17:09:20,418 INFO   opponent_box      -2.287    0.698    -2.500    +0.000
2026-08-18 17:09:20,423 INFO   timeout           -0.005    0.098    -2.000    +0.000
2026-08-18 17:09:20,429 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:20,434 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:20,441 INFO   stamina_penalty    -0.121    0.051    -0.229    +0.000
2026-08-18 17:09:20,820 INFO --- Reward component breakdown (outcome=ball_out, 158 episode(s)) ---
2026-08-18 17:09:20,820 INFO   component           mean      std       min       max
2026-08-18 17:09:20,820 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:20,821 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:20,821 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:20,821 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:20,821 INFO   get_possession    +0.918    0.275    +0.000    +1.000
2026-08-18 17:09:20,821 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:20,821 INFO   lose_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:20,821 INFO   ball_out          -4.000    0.000    -4.000    -4.000
2026-08-18 17:09:20,821 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:20,822 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:20,822 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:20,822 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:20,822 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:20,822 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:20,822 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:20,822 INFO   stamina_penalty    +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,216 INFO --- Reward component breakdown (outcome=invalid, 1620 episode(s)) ---
2026-08-18 17:09:21,216 INFO   component           mean      std       min       max
2026-08-18 17:09:21,217 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,218 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,220 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,221 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,222 INFO   get_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,222 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,223 INFO   lose_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,223 INFO   ball_out          +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,223 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,224 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,224 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,224 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,225 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,225 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,226 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,226 INFO   stamina_penalty    +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,617 INFO --- Reward component breakdown (outcome=loss, 0 episode(s)) ---
2026-08-18 17:09:21,935 INFO --- Reward component breakdown (outcome=timeout, 52 episode(s)) ---
2026-08-18 17:09:21,935 INFO   component           mean      std       min       max
2026-08-18 17:09:21,935 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,935 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,935 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,935 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,936 INFO   get_possession    +1.212    0.599    +0.000    +3.000
2026-08-18 17:09:21,936 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,936 INFO   lose_possession    -0.208    0.518    -1.800    +0.000
2026-08-18 17:09:21,936 INFO   ball_out          +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,936 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,936 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,936 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,936 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,936 INFO   timeout           -2.000    0.000    -2.000    -2.000
2026-08-18 17:09:21,936 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,936 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 17:09:21,936 INFO   stamina_penalty    -0.132    0.036    -0.199    -0.047
2026-08-18 17:09:22,420 INFO --- MC returns by outcome (valid rows, 597,733 rows) ---
2026-08-18 17:09:22,444 INFO   ball_out     n=  3,220  mean=-2.844  std=0.405  min=-4.000  max=-1.986
2026-08-18 17:09:22,458 INFO   invalid      n= 21,959  mean=+0.004  std=0.163  min=+0.000  max=+6.809
2026-08-18 17:09:22,472 INFO   timeout      n=  3,498  mean=-0.402  std=0.598  min=-2.730  max=+0.122
2026-08-18 17:09:22,491 INFO   win          n=569,056  mean=+3.946  std=0.925  min=-2.600  max=+11.631
2026-08-18 17:09:23,464 INFO --- Episode total reward by outcome (valid rows, 21500 episode(s)) ---
2026-08-18 17:09:23,470 INFO   ball_out     n=   158  mean=-3.095  std=0.293  min=-4.000  max=-3.000
2026-08-18 17:09:23,470 INFO   invalid      n= 1,620  mean=+0.004  std=0.170  min=+0.000  max=+6.864
2026-08-18 17:09:23,470 INFO   timeout      n=    52  mean=-0.334  std=0.569  min=-1.996  max=-0.031
2026-08-18 17:09:23,471 INFO   win          n=19,670  mean=+5.182  std=0.788  min=-2.600  max=+12.686
2026-08-18 17:09:23,931 INFO Train/val split (valid_only=True): 478,663 train rows across 17200 episodes  |  119,070 val rows across 4300 episodes
2026-08-18 17:09:24,651 INFO --- Linear regression baseline (top_speed, ball_dist, ball_to_box, time_remaining) ---
2026-08-18 17:09:24,759 INFO   [all outcomes] n_train=478,663  n_val=119,070  coef(top_speed, ball_dist, ball_to_box, time_rem, intercept)=[0.536, -4.087, -1.748, 3.913, 1.604]
    train_rmse=1.0555 (norm=0.8159)  val_rmse=1.0914 (norm=0.8437)
2026-08-18 17:09:25,445 INFO   [win outcomes only] n_train=456,193  n_val=112,863  coef(top_speed, ball_dist, ball_to_box, time_rem, intercept)=[0.391, -2.216, -1.859, 4.004, 1.771]
    train_rmse=0.4958 (norm=0.5388)  val_rmse=0.5367 (norm=0.5832)
2026-08-18 17:09:27,087 INFO value_net: fresh separate value_net (--reset-separate-value-net)  total_params=281,340  trainable_params=281,340
2026-08-18 17:09:27,090 INFO Fitting fresh separate value network: 100 epochs, lr=0.0002, weight_decay=1e-06, batch_size=3096, train_ret_std=1.294, outcome_reweight=False
2026-08-18 17:11:59,607 INFO epoch   0/100 (baseline, no training yet)  train_rmse=3.8662 (norm=2.9888)  val_rmse=3.8665 (norm=2.9890)
2026-08-18 17:11:59,607 INFO     opponent=immobile   train_rmse=3.8662 (n=478663)  val_rmse=3.8665 (n=119070)
2026-08-18 17:11:59,607 INFO     outcome=ball_out      train_rmse=2.9518 (n=2532)  val_rmse=2.9958 (n=688)
2026-08-18 17:11:59,607 INFO     outcome=invalid       train_rmse=0.2006 (n=17653)  val_rmse=0.0877 (n=4306)
2026-08-18 17:11:59,607 INFO     outcome=timeout       train_rmse=0.7038 (n=2285)  val_rmse=0.9174 (n=1213)
2026-08-18 17:11:59,607 INFO     outcome=win           train_rmse=3.9537 (n=456193)  val_rmse=3.9633 (n=112863)
2026-08-18 17:15:58,824 INFO epoch   1/100  train_rmse=1.6780 (norm=1.2971)  val_rmse=1.0772 (norm=0.8327)
2026-08-18 17:15:58,825 INFO     opponent=immobile   train_rmse=1.6780 (n=478663)  val_rmse=1.0772 (n=119070)
2026-08-18 17:15:58,825 INFO     outcome=ball_out      train_rmse=6.1201 (n=2532)  val_rmse=6.3470 (n=688)
2026-08-18 17:15:58,825 INFO     outcome=invalid       train_rmse=3.2200 (n=17653)  val_rmse=3.0246 (n=4306)
2026-08-18 17:15:58,825 INFO     outcome=timeout       train_rmse=3.4779 (n=2285)  val_rmse=3.6976 (n=1213)
2026-08-18 17:15:58,825 INFO     outcome=win           train_rmse=1.5115 (n=456193)  val_rmse=0.6946 (n=112863)
2026-08-18 17:19:59,940 INFO epoch   2/100  train_rmse=0.9845 (norm=0.7610)  val_rmse=1.0358 (norm=0.8007)
2026-08-18 17:19:59,940 INFO     opponent=immobile   train_rmse=0.9845 (n=478663)  val_rmse=1.0358 (n=119070)
2026-08-18 17:19:59,940 INFO     outcome=ball_out      train_rmse=6.0120 (n=2532)  val_rmse=6.1807 (n=688)
2026-08-18 17:19:59,940 INFO     outcome=invalid       train_rmse=2.7713 (n=17653)  val_rmse=2.6615 (n=4306)
2026-08-18 17:19:59,940 INFO     outcome=timeout       train_rmse=3.3017 (n=2285)  val_rmse=3.7086 (n=1213)
2026-08-18 17:19:59,940 INFO     outcome=win           train_rmse=0.6815 (n=456193)  val_rmse=0.6935 (n=112863)
2026-08-18 17:23:44,291 INFO epoch   3/100  train_rmse=0.9393 (norm=0.7261)  val_rmse=0.9990 (norm=0.7723)
2026-08-18 17:23:44,291 INFO     opponent=immobile   train_rmse=0.9393 (n=478663)  val_rmse=0.9990 (n=119070)
2026-08-18 17:23:44,291 INFO     outcome=ball_out      train_rmse=5.7564 (n=2532)  val_rmse=6.0361 (n=688)
2026-08-18 17:23:44,291 INFO     outcome=invalid       train_rmse=2.4478 (n=17653)  val_rmse=2.3136 (n=4306)
2026-08-18 17:23:44,291 INFO     outcome=timeout       train_rmse=3.2520 (n=2285)  val_rmse=3.6244 (n=1213)
2026-08-18 17:23:44,291 INFO     outcome=win           train_rmse=0.6759 (n=456193)  val_rmse=0.6968 (n=112863)
2026-08-18 17:26:57,659 INFO epoch   4/100  train_rmse=0.9059 (norm=0.7003)  val_rmse=0.9919 (norm=0.7668)
2026-08-18 17:26:57,659 INFO     opponent=immobile   train_rmse=0.9059 (n=478663)  val_rmse=0.9919 (n=119070)
2026-08-18 17:26:57,659 INFO     outcome=ball_out      train_rmse=5.5783 (n=2532)  val_rmse=5.5792 (n=688)
2026-08-18 17:26:57,659 INFO     outcome=invalid       train_rmse=2.2276 (n=17653)  val_rmse=1.8644 (n=4306)
2026-08-18 17:26:57,659 INFO     outcome=timeout       train_rmse=3.2110 (n=2285)  val_rmse=3.5229 (n=1213)
2026-08-18 17:26:57,659 INFO     outcome=win           train_rmse=0.6668 (n=456193)  val_rmse=0.7630 (n=112863)
2026-08-18 17:29:55,952 INFO epoch   5/100  train_rmse=0.8838 (norm=0.6832)  val_rmse=0.9635 (norm=0.7448)
2026-08-18 17:29:55,952 INFO     opponent=immobile   train_rmse=0.8838 (n=478663)  val_rmse=0.9635 (n=119070)
2026-08-18 17:29:55,952 INFO     outcome=ball_out      train_rmse=5.4205 (n=2532)  val_rmse=5.9720 (n=688)
2026-08-18 17:29:55,952 INFO     outcome=invalid       train_rmse=2.0761 (n=17653)  val_rmse=2.1367 (n=4306)
2026-08-18 17:29:55,952 INFO     outcome=timeout       train_rmse=3.1824 (n=2285)  val_rmse=3.5956 (n=1213)
2026-08-18 17:29:55,952 INFO     outcome=win           train_rmse=0.6626 (n=456193)  val_rmse=0.6699 (n=112863)
2026-08-18 17:33:01,727 INFO epoch   6/100  train_rmse=0.8642 (norm=0.6680)  val_rmse=0.9512 (norm=0.7353)
2026-08-18 17:33:01,727 INFO     opponent=immobile   train_rmse=0.8642 (n=478663)  val_rmse=0.9512 (n=119070)
2026-08-18 17:33:01,727 INFO     outcome=ball_out      train_rmse=5.3033 (n=2532)  val_rmse=5.8148 (n=688)
2026-08-18 17:33:01,727 INFO     outcome=invalid       train_rmse=1.9435 (n=17653)  val_rmse=1.8570 (n=4306)
2026-08-18 17:33:01,727 INFO     outcome=timeout       train_rmse=3.1345 (n=2285)  val_rmse=3.5898 (n=1213)
2026-08-18 17:33:01,727 INFO     outcome=win           train_rmse=0.6573 (n=456193)  val_rmse=0.6917 (n=112863)
2026-08-18 17:50:26,784 WARNING --checkpoint and --data both given: using --data as the dataset (no live rollout will be collected) and --checkpoint only to initialize decision_net/value_net's weights -- i.e. warm-starting value fitting from a trained checkpoint against a pre-recorded dataset instead of a fresh rollout.
2026-08-18 17:50:26,929 INFO Checkpoint checkpoints/longterm/checkpoint_vvgood_immobile.pt: separate_value_net=True (auto-detected)
2026-08-18 17:50:26,969 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 17:50:26,970 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 17:50:26,970 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 17:50:26,973 INFO --reset-dir-log-std: reset move_dir_log_std/kick_dir_log_std to config init values before rollout collection.
2026-08-18 17:50:26,974 INFO Loading 10 demonstration file(s) from demonstrations/phase1_neural_vvgood
2026-08-18 17:50:29,990 INFO Dataset: 279,866 steps loaded
2026-08-18 17:50:29,991 INFO Loaded 279,866 rows total
2026-08-18 17:50:29,991 INFO has_rewards=True
2026-08-18 17:50:29,995 INFO valid_indices(): 139,989 rows (50.0% of total)
2026-08-18 17:50:30,117 INFO Returns over ALL rows: mean=0.807 std=3.098 min=-5.024 max=10.616
2026-08-18 17:50:30,118 INFO Returns over valid_indices(): mean=3.741 std=1.293
2026-08-18 17:50:30,235 INFO --- Dataset distribution (279,866 rows, 5000 episodes) ---
2026-08-18 17:50:30,238 INFO   self.ai_type == rules: 0.0%
2026-08-18 17:50:30,239 INFO   self.ai_type == immobile: 50.0%
2026-08-18 17:50:30,241 INFO   self.ai_type == neural: 50.0%
2026-08-18 17:50:30,243 INFO   opponent.ai_type == rules: 0.0%
2026-08-18 17:50:30,246 INFO   opponent.ai_type == immobile: 50.0%
2026-08-18 17:50:30,248 INFO   opponent.ai_type == neural: 50.0%
2026-08-18 17:50:30,255 INFO   valid rows: kick_this_tick rate=0.00%  tackle_attempt rate=0.54%
2026-08-18 17:50:30,255 INFO   dones=1 rows: 10,000  |  zero-reward rows: 266,243 (95.1%)
2026-08-18 17:50:30,268 INFO   return percentiles (valid rows): p10=2.59  p50=3.83  p90=5.15
2026-08-18 17:50:30,495 INFO --- Reward component breakdown (all episodes, 5000 episode(s)) ---
2026-08-18 17:50:30,495 INFO   component           mean      std       min       max
2026-08-18 17:50:30,496 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,497 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,498 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,498 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,499 INFO   get_possession    +0.943    0.324    +0.000    +3.000
2026-08-18 17:50:30,500 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,500 INFO   lose_possession    -0.017    0.160    -1.800    +0.000
2026-08-18 17:50:30,501 INFO   ball_out          -0.024    0.309    -4.000    +0.000
2026-08-18 17:50:30,502 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,502 INFO   box_possession    +1.829    0.560    +0.000    +2.000
2026-08-18 17:50:30,503 INFO   speed_bonus       +2.085    0.918    -0.015    +3.947
2026-08-18 17:50:30,503 INFO   opponent_box      -2.286    0.699    -2.500    +0.000
2026-08-18 17:50:30,504 INFO   timeout           -0.007    0.116    -2.000    +0.000
2026-08-18 17:50:30,504 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,505 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,505 INFO   stamina_penalty    -0.122    0.051    -0.229    +0.000
2026-08-18 17:50:30,668 INFO --- Reward component breakdown (outcome=ball_out, 30 episode(s)) ---
2026-08-18 17:50:30,668 INFO   component           mean      std       min       max
2026-08-18 17:50:30,668 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,668 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,668 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,668 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,668 INFO   get_possession    +0.967    0.180    +0.000    +1.000
2026-08-18 17:50:30,669 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,669 INFO   lose_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,669 INFO   ball_out          -4.000    0.000    -4.000    -4.000
2026-08-18 17:50:30,669 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,669 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,669 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,669 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,669 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,669 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,669 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,669 INFO   stamina_penalty    +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,745 INFO --- Reward component breakdown (outcome=invalid, 381 episode(s)) ---
2026-08-18 17:50:30,745 INFO   component           mean      std       min       max
2026-08-18 17:50:30,745 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,745 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,745 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,746 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,746 INFO   get_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,746 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,746 INFO   lose_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,746 INFO   ball_out          +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,746 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,746 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,746 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,747 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,747 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,747 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,747 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,747 INFO   stamina_penalty    +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,810 INFO --- Reward component breakdown (outcome=loss, 0 episode(s)) ---
2026-08-18 17:50:30,872 INFO --- Reward component breakdown (outcome=timeout, 17 episode(s)) ---
2026-08-18 17:50:30,872 INFO   component           mean      std       min       max
2026-08-18 17:50:30,872 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,872 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,872 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,872 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,872 INFO   get_possession    +1.176    0.513    +1.000    +3.000
2026-08-18 17:50:30,872 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,872 INFO   lose_possession    -0.159    0.462    -1.800    +0.000
2026-08-18 17:50:30,872 INFO   ball_out          +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,872 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,872 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,872 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,873 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,873 INFO   timeout           -2.000    0.000    -2.000    -2.000
2026-08-18 17:50:30,873 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,873 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 17:50:30,873 INFO   stamina_penalty    -0.145    0.035    -0.201    -0.081
2026-08-18 17:50:30,967 INFO --- MC returns by outcome (valid rows, 139,989 rows) ---
2026-08-18 17:50:30,971 INFO   ball_out     n=    634  mean=-2.842  std=0.360  min=-4.000  max=-2.205
2026-08-18 17:50:30,973 INFO   invalid      n=  5,135  mean=+0.000  std=0.000  min=+0.000  max=+0.000
2026-08-18 17:50:30,975 INFO   timeout      n=  1,142  mean=-0.388  std=0.566  min=-2.583  max=+0.141
2026-08-18 17:50:30,978 INFO   win          n=133,078  mean=+3.952  std=0.905  min=-2.528  max=+10.616
2026-08-18 17:50:31,164 INFO --- Episode total reward by outcome (valid rows, 5000 episode(s)) ---
2026-08-18 17:50:31,166 INFO   ball_out     n=    30  mean=-3.067  std=0.249  min=-4.000  max=-3.000
2026-08-18 17:50:31,166 INFO   invalid      n=   381  mean=+0.000  std=0.000  min=+0.000  max=+0.000
2026-08-18 17:50:31,166 INFO   timeout      n=    17  mean=-0.257  std=0.466  min=-1.912  max=-0.045
2026-08-18 17:50:31,166 INFO   win          n= 4,572  mean=+5.174  std=0.762  min=-2.528  max=+11.671
2026-08-18 17:50:31,258 INFO Train/val split (valid_only=True): 112,352 train rows across 4000 episodes  |  27,637 val rows across 1000 episodes
2026-08-18 17:50:31,411 INFO --- Linear regression baseline (top_speed, ball_dist, ball_to_box, time_remaining) ---
2026-08-18 17:50:31,436 INFO   [all outcomes] n_train=112,352  n_val=27,637  coef(top_speed, ball_dist, ball_to_box, time_rem, intercept)=[0.464, -3.936, -1.804, 4.006, 1.614]
    train_rmse=1.0192 (norm=0.8025)  val_rmse=1.1295 (norm=0.8893)
2026-08-18 17:50:31,563 INFO   [win outcomes only] n_train=107,078  n_val=26,000  coef(top_speed, ball_dist, ball_to_box, time_rem, intercept)=[0.375, -2.19, -1.842, 3.998, 1.779]
    train_rmse=0.4841 (norm=0.5393)  val_rmse=0.5038 (norm=0.5613)
2026-08-18 17:50:33,027 INFO value_net: fresh separate value_net (--reset-separate-value-net)  total_params=281,340  trainable_params=281,340
2026-08-18 17:50:33,027 INFO Fitting fresh separate value network: 100 epochs, lr=0.0002, weight_decay=1e-06, batch_size=3096, train_ret_std=1.270, outcome_reweight=False
2026-08-18 18:23:38,157 WARNING --checkpoint and --data both given: using --data as the dataset (no live rollout will be collected) and --checkpoint only to initialize decision_net/value_net's weights -- i.e. warm-starting value fitting from a trained checkpoint against a pre-recorded dataset instead of a fresh rollout.
2026-08-18 18:23:38,676 INFO Checkpoint checkpoints/longterm/checkpoint_vvgood_immobile.pt: separate_value_net=True (auto-detected)
2026-08-18 18:23:38,844 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 18:23:38,850 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 18:23:38,850 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 18:23:38,864 INFO --reset-dir-log-std: reset move_dir_log_std/kick_dir_log_std to config init values before rollout collection.
Traceback (most recent call last):
  File "/home/vincent/Documents/not_work/repos/FootballCoach/debug_value_network.py", line 2208, in <module>
    main()
  File "/home/vincent/Documents/not_work/repos/FootballCoach/debug_value_network.py", line 1771, in main
    ds = DemonstrationDataset.from_directory(args.data)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vincent/Documents/not_work/repos/FootballCoach/src/footballcoach/ai/bc/dataset.py", line 596, in from_directory
    raise FileNotFoundError(f"No .npz files found in {directory}")
FileNotFoundError: No .npz files found in demonstrations/phase1_neural_vvgood
