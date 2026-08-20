2026-08-18 21:21:46,651 INFO value_net: fresh separate value_net (--reset-separate-value-net)  total_params=281,340  trainable_params=281,340
2026-08-18 21:21:46,652 INFO Fitting fresh separate value network: 100 epochs, lr=0.0002, weight_decay=1e-06, batch_size=3096, train_ret_std=1.297, outcome_reweight=False
2026-08-18 21:21:53,329 INFO epoch   0/100 (baseline, no training yet)  train_rmse=3.8721 (norm=2.9865)  val_rmse=3.9345 (norm=3.0346)
2026-08-18 21:21:53,329 INFO     opponent=immobile   train_rmse=3.8721 (n=16835)  val_rmse=3.9345 (n=4162)
2026-08-18 21:21:53,329 INFO     outcome=ball_out      train_rmse=2.8950 (n=106)  val_rmse=3.9513 (n=10)
2026-08-18 21:21:53,329 INFO     outcome=invalid       train_rmse=0.0893 (n=608)  val_rmse=0.0887 (n=113)
2026-08-18 21:21:53,329 INFO     outcome=timeout       train_rmse=0.6149 (n=136)  val_rmse=nan (n=0)
2026-08-18 21:21:53,329 INFO     outcome=win           train_rmse=3.9663 (n=15985)  val_rmse=3.9891 (n=4039)
2026-08-18 23:38:13,973 WARNING --checkpoint and --data both given: using --data as the dataset (no live rollout will be collected) and --checkpoint only to initialize decision_net/value_net's weights -- i.e. warm-starting value fitting from a trained checkpoint against a pre-recorded dataset instead of a fresh rollout.
2026-08-18 23:38:14,223 INFO Checkpoint checkpoints/longterm/checkpoint_vvgood_immobile.pt: separate_value_net=True (auto-detected)
2026-08-18 23:38:14,296 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 23:38:14,299 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 23:38:14,299 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 23:38:14,309 INFO --reset-dir-log-std: reset move_dir_log_std/kick_dir_log_std to config init values before rollout collection.
2026-08-18 23:38:14,315 INFO Loading 252 demonstration file(s) from demonstrations/phase1_neural_vvgood
2026-08-18 23:39:01,308 INFO Dataset: 2,785,004 steps loaded
2026-08-18 23:39:01,310 INFO Loaded 2,785,004 rows total
2026-08-18 23:39:01,311 INFO has_rewards=True
2026-08-18 23:39:01,374 INFO valid_indices(): 1,393,215 rows (50.0% of total)
2026-08-18 23:39:03,873 INFO Returns over ALL rows: mean=0.806 std=3.111 min=-5.124 max=13.009
2026-08-18 23:39:03,878 INFO Returns over valid_indices(): mean=3.756 std=1.290
2026-08-18 23:39:05,512 INFO --- Dataset distribution (2,785,004 rows, 50000 episodes) ---
2026-08-18 23:39:05,541 INFO   self.ai_type == rules: 0.0%
2026-08-18 23:39:05,570 INFO   self.ai_type == immobile: 50.0%
2026-08-18 23:39:05,602 INFO   self.ai_type == neural: 50.0%
2026-08-18 23:39:05,637 INFO   opponent.ai_type == rules: 0.0%
2026-08-18 23:39:05,666 INFO   opponent.ai_type == immobile: 50.0%
2026-08-18 23:39:05,691 INFO   opponent.ai_type == neural: 50.0%
2026-08-18 23:39:06,345 INFO   valid rows: kick_this_tick rate=0.00%  tackle_attempt rate=0.78%
2026-08-18 23:39:06,348 INFO   dones=1 rows: 100,000  |  zero-reward rows: 2,647,771 (95.1%)
2026-08-18 23:39:06,387 INFO   return percentiles (trainee's own valid rows): p10=2.63  p50=3.84  p90=5.16
2026-08-18 23:39:09,352 INFO --- Reward component breakdown (all episodes, 50000 episode(s)) ---
2026-08-18 23:39:09,352 INFO   component           mean      std       min       max
2026-08-18 23:39:09,366 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:09,385 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:09,408 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:09,429 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:09,450 INFO   get_possession    +0.934    0.285    +0.000    +3.000
2026-08-18 23:39:09,475 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:09,494 INFO   lose_possession    -0.009    0.088    -1.800    +0.000
2026-08-18 23:39:09,511 INFO   ball_out          -0.029    0.338    -4.000    +0.000
2026-08-18 23:39:09,532 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:09,551 INFO   box_possession    +1.828    0.560    +0.000    +2.000
2026-08-18 23:39:09,574 INFO   speed_bonus       +2.092    0.929    -0.023    +3.985
2026-08-18 23:39:09,591 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:09,613 INFO   timeout           -0.003    0.056    -1.000    +0.000
2026-08-18 23:39:09,628 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:09,645 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:09,662 INFO   stamina_penalty    -0.072    0.034    -0.158    +0.000
2026-08-18 23:39:13,109 INFO --- Reward component breakdown (outcome=win, 45708 episode(s)) ---
2026-08-18 23:39:13,109 INFO   component           mean      std       min       max
2026-08-18 23:39:13,122 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:13,139 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:13,153 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:13,167 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:13,182 INFO   get_possession    +1.010    0.100    +1.000    +3.000
2026-08-18 23:39:13,191 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:13,199 INFO   lose_possession    -0.009    0.090    -1.800    +0.000
2026-08-18 23:39:13,210 INFO   ball_out          +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:13,225 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:13,241 INFO   box_possession    +2.000    0.000    +2.000    +2.000
2026-08-18 23:39:13,257 INFO   speed_bonus       +2.289    0.703    -0.023    +3.985
2026-08-18 23:39:13,272 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:13,292 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:13,307 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:13,325 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:13,343 INFO   stamina_penalty    -0.078    0.028    -0.158    -0.015
2026-08-18 23:39:14,649 INFO --- Reward component breakdown (outcome=loss, 0 episode(s)) ---
2026-08-18 23:39:15,895 INFO --- Reward component breakdown (outcome=ball_out, 359 episode(s)) ---
2026-08-18 23:39:15,895 INFO   component           mean      std       min       max
2026-08-18 23:39:15,895 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:15,896 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:15,896 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:15,896 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:15,896 INFO   get_possession    +0.953    0.237    +0.000    +2.000
2026-08-18 23:39:15,897 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:15,897 INFO   lose_possession    -0.005    0.067    -0.900    +0.000
2026-08-18 23:39:15,897 INFO   ball_out          -4.000    0.000    -4.000    -4.000
2026-08-18 23:39:15,897 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:15,897 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:15,897 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:15,897 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:15,898 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:15,898 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:15,898 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:15,898 INFO   stamina_penalty    +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:17,340 INFO --- Reward component breakdown (outcome=invalid, 3773 episode(s)) ---
2026-08-18 23:39:17,341 INFO   component           mean      std       min       max
2026-08-18 23:39:17,344 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:17,346 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:17,348 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:17,350 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:17,353 INFO   get_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:17,355 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:17,359 INFO   lose_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:17,360 INFO   ball_out          +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:17,362 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:17,365 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:17,368 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:17,370 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:17,372 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:17,374 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:17,375 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:17,377 INFO   stamina_penalty    +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:18,611 INFO --- Reward component breakdown (outcome=timeout, 160 episode(s)) ---
2026-08-18 23:39:18,612 INFO   component           mean      std       min       max
2026-08-18 23:39:18,612 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:18,612 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:18,613 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:18,613 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:18,613 INFO   get_possession    +1.062    0.348    +0.000    +2.000
2026-08-18 23:39:18,614 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:18,614 INFO   lose_possession    -0.101    0.284    -0.900    +0.000
2026-08-18 23:39:18,614 INFO   ball_out          +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:18,614 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:18,614 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:18,615 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:18,615 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:18,615 INFO   timeout           -1.000    0.000    -1.000    -1.000
2026-08-18 23:39:18,615 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:18,615 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 23:39:18,615 INFO   stamina_penalty    -0.087    0.029    -0.152    -0.012
2026-08-18 23:39:20,579 INFO --- MC returns by outcome (trainee's own valid rows, 1,393,215 rows) ---
2026-08-18 23:39:20,645 INFO   ball_out     n=  7,089  mean=-2.822  std=0.396  min=-4.000  max=-1.372
2026-08-18 23:39:20,683 INFO   invalid      n= 51,507  mean=+0.000  std=0.000  min=+0.000  max=+0.000
2026-08-18 23:39:20,716 INFO   timeout      n= 10,789  mean=-0.279  std=0.492  min=-1.615  max=+0.395
2026-08-18 23:39:20,767 INFO   win          n=1,323,830  mean=+3.970  std=0.889  min=+1.098  max=+13.009
2026-08-18 23:39:24,210 INFO --- Episode total reward by outcome (trainee's own valid rows, 50000 episode(s)) ---
2026-08-18 23:39:24,226 INFO   ball_out     n=   359  mean=-3.043  std=0.244  min=-4.000  max=-2.000
2026-08-18 23:39:24,226 INFO   invalid      n= 3,773  mean=+0.000  std=0.000  min=+0.000  max=+0.000
2026-08-18 23:39:24,227 INFO   timeout      n=   160  mean=-0.115  std=0.210  min=-1.086  max=+0.150
2026-08-18 23:39:24,227 INFO   win          n=45,708  mean=+5.214  std=0.717  min=+2.847  max=+13.064
2026-08-18 23:39:25,574 INFO Train/val split (valid_only=True): 1,114,035 train rows across 40000 episodes  |  279,180 val rows across 10000 episodes
2026-08-18 23:39:27,575 INFO --- Linear regression baseline (top_speed, ball_dist, ball_to_box, time_remaining) ---
2026-08-18 23:39:28,305 INFO   [all outcomes] n_train=1,114,035  n_val=279,180  coef(top_speed, ball_dist, ball_to_box, time_rem, intercept)=[0.42, -4.204, -1.791, 4.037, 1.642]
    train_rmse=1.0366 (norm=0.8013)  val_rmse=1.0135 (norm=0.7834)
2026-08-18 23:39:31,173 INFO   [win outcomes only] n_train=1,058,219  n_val=265,611  coef(top_speed, ball_dist, ball_to_box, time_rem, intercept)=[0.435, -2.211, -1.839, 4.03, 1.727]
    train_rmse=0.4375 (norm=0.4923)  val_rmse=0.4449 (norm=0.5007)
2026-08-18 23:39:33,103 INFO value_net: fresh separate value_net (--reset-separate-value-net)  total_params=281,340  trainable_params=281,340
2026-08-18 23:39:33,106 INFO Fitting fresh separate value network: 100 epochs, lr=0.0002, weight_decay=1e-06, batch_size=3096, train_ret_std=1.294, outcome_reweight=False
2026-08-18 23:46:12,965 INFO epoch   0/100 (baseline, no training yet)  train_rmse=3.8768 (norm=2.9967)  val_rmse=3.8761 (norm=2.9962)
2026-08-18 23:46:12,965 INFO     opponent=immobile   train_rmse=3.8768 (n=1114035)  val_rmse=3.8761 (n=279180)
2026-08-18 23:46:12,965 INFO     outcome=ball_out      train_rmse=2.9434 (n=5810)  val_rmse=2.9184 (n=1279)
2026-08-18 23:46:12,965 INFO     outcome=invalid       train_rmse=0.0892 (n=41102)  val_rmse=0.0891 (n=10405)
2026-08-18 23:46:12,965 INFO     outcome=timeout       train_rmse=0.6332 (n=8904)  val_rmse=0.6047 (n=1885)
2026-08-18 23:46:12,965 INFO     outcome=win           train_rmse=3.9713 (n=1058219)  val_rmse=3.9684 (n=265611)
2026-08-18 23:55:47,934 INFO epoch   1/100  train_rmse=1.3212 (norm=1.0213)  val_rmse=0.9262 (norm=0.7159)
2026-08-18 23:55:47,934 INFO     opponent=immobile   train_rmse=1.3212 (n=1114035)  val_rmse=0.9262 (n=279180)
2026-08-18 23:55:47,934 INFO     outcome=ball_out      train_rmse=6.0206 (n=5810)  val_rmse=5.7829 (n=1279)
2026-08-18 23:55:47,934 INFO     outcome=invalid       train_rmse=2.9497 (n=41102)  val_rmse=2.5686 (n=10405)
2026-08-18 23:55:47,934 INFO     outcome=timeout       train_rmse=3.4502 (n=8904)  val_rmse=3.3618 (n=1885)
2026-08-18 23:55:47,934 INFO     outcome=win           train_rmse=1.0957 (n=1058219)  val_rmse=0.6340 (n=265611)
2026-08-19 00:06:06,190 INFO epoch   2/100  train_rmse=0.9113 (norm=0.7044)  val_rmse=0.8753 (norm=0.6766)
2026-08-19 00:06:06,190 INFO     opponent=immobile   train_rmse=0.9113 (n=1114035)  val_rmse=0.8753 (n=279180)
2026-08-19 00:06:06,190 INFO     outcome=ball_out      train_rmse=5.6212 (n=5810)  val_rmse=5.6395 (n=1279)
2026-08-19 00:06:06,190 INFO     outcome=invalid       train_rmse=2.2435 (n=41102)  val_rmse=2.2794 (n=10405)
2026-08-19 00:06:06,190 INFO     outcome=timeout       train_rmse=3.3635 (n=8904)  val_rmse=3.3215 (n=1885)
2026-08-19 00:06:06,190 INFO     outcome=win           train_rmse=0.6404 (n=1058219)  val_rmse=0.6086 (n=265611)
2026-08-19 00:13:26,419 INFO epoch   3/100  train_rmse=0.8706 (norm=0.6729)  val_rmse=0.8405 (norm=0.6497)
2026-08-19 00:13:26,420 INFO     opponent=immobile   train_rmse=0.8706 (n=1114035)  val_rmse=0.8405 (n=279180)
2026-08-19 00:13:26,420 INFO     outcome=ball_out      train_rmse=5.4001 (n=5810)  val_rmse=5.3946 (n=1279)
2026-08-19 00:13:26,420 INFO     outcome=invalid       train_rmse=1.9724 (n=41102)  val_rmse=1.9110 (n=10405)
2026-08-19 00:13:26,420 INFO     outcome=timeout       train_rmse=3.3181 (n=8904)  val_rmse=3.2604 (n=1885)
2026-08-19 00:13:26,420 INFO     outcome=win           train_rmse=0.6277 (n=1058219)  val_rmse=0.6197 (n=265611)
2026-08-19 00:23:27,252 INFO epoch   4/100  train_rmse=0.8331 (norm=0.6440)  val_rmse=0.8157 (norm=0.6305)
2026-08-19 00:23:27,252 INFO     opponent=immobile   train_rmse=0.8331 (n=1114035)  val_rmse=0.8157 (n=279180)
2026-08-19 00:23:27,252 INFO     outcome=ball_out      train_rmse=5.2318 (n=5810)  val_rmse=5.2126 (n=1279)
2026-08-19 00:23:27,253 INFO     outcome=invalid       train_rmse=1.7865 (n=41102)  val_rmse=1.7093 (n=10405)
2026-08-19 00:23:27,253 INFO     outcome=timeout       train_rmse=3.2018 (n=8904)  val_rmse=3.0257 (n=1885)
2026-08-19 00:23:27,253 INFO     outcome=win           train_rmse=0.6084 (n=1058219)  val_rmse=0.6237 (n=265611)
2026-08-19 00:33:18,965 INFO epoch   5/100  train_rmse=0.8042 (norm=0.6217)  val_rmse=0.7854 (norm=0.6071)
2026-08-19 00:33:18,965 INFO     opponent=immobile   train_rmse=0.8042 (n=1114035)  val_rmse=0.7854 (n=279180)
2026-08-19 00:33:18,965 INFO     outcome=ball_out      train_rmse=5.1087 (n=5810)  val_rmse=5.1857 (n=1279)
2026-08-19 00:33:18,965 INFO     outcome=invalid       train_rmse=1.6713 (n=41102)  val_rmse=1.6475 (n=10405)
2026-08-19 00:33:18,965 INFO     outcome=timeout       train_rmse=3.0263 (n=8904)  val_rmse=2.9778 (n=1885)
2026-08-19 00:33:18,965 INFO     outcome=win           train_rmse=0.5933 (n=1058219)  val_rmse=0.5912 (n=265611)
2026-08-19 00:42:29,791 INFO epoch   6/100  train_rmse=0.7708 (norm=0.5958)  val_rmse=0.7717 (norm=0.5965)
2026-08-19 00:42:29,791 INFO     opponent=immobile   train_rmse=0.7708 (n=1114035)  val_rmse=0.7717 (n=279180)
2026-08-19 00:42:29,791 INFO     outcome=ball_out      train_rmse=5.0099 (n=5810)  val_rmse=5.0946 (n=1279)
2026-08-19 00:42:29,791 INFO     outcome=invalid       train_rmse=1.5732 (n=41102)  val_rmse=1.5851 (n=10405)
2026-08-19 00:42:29,791 INFO     outcome=timeout       train_rmse=2.8443 (n=8904)  val_rmse=2.6895 (n=1885)
2026-08-19 00:42:29,791 INFO     outcome=win           train_rmse=0.5687 (n=1058219)  val_rmse=0.5927 (n=265611)
2026-08-19 00:49:35,554 INFO epoch   7/100  train_rmse=0.7527 (norm=0.5819)  val_rmse=0.7855 (norm=0.6072)
2026-08-19 00:49:35,554 INFO     opponent=immobile   train_rmse=0.7527 (n=1114035)  val_rmse=0.7855 (n=279180)
2026-08-19 00:49:35,554 INFO     outcome=ball_out      train_rmse=4.9171 (n=5810)  val_rmse=5.0775 (n=1279)
2026-08-19 00:49:35,554 INFO     outcome=invalid       train_rmse=1.5090 (n=41102)  val_rmse=1.6204 (n=10405)
2026-08-19 00:49:35,554 INFO     outcome=timeout       train_rmse=2.6873 (n=8904)  val_rmse=2.8038 (n=1885)
2026-08-19 00:49:35,554 INFO     outcome=win           train_rmse=0.5608 (n=1058219)  val_rmse=0.6048 (n=265611)
2026-08-19 00:58:55,793 INFO epoch   8/100  train_rmse=0.7460 (norm=0.5766)  val_rmse=0.7412 (norm=0.5729)
2026-08-19 00:58:55,793 INFO     opponent=immobile   train_rmse=0.7460 (n=1114035)  val_rmse=0.7412 (n=279180)
2026-08-19 00:58:55,793 INFO     outcome=ball_out      train_rmse=4.8791 (n=5810)  val_rmse=5.3648 (n=1279)
2026-08-19 00:58:55,793 INFO     outcome=invalid       train_rmse=1.4631 (n=41102)  val_rmse=1.7850 (n=10405)
2026-08-19 00:58:55,793 INFO     outcome=timeout       train_rmse=2.6291 (n=8904)  val_rmse=2.6826 (n=1885)
2026-08-19 00:58:55,793 INFO     outcome=win           train_rmse=0.5602 (n=1058219)  val_rmse=0.5128 (n=265611)
2026-08-19 01:09:19,185 INFO epoch   9/100  train_rmse=0.7227 (norm=0.5586)  val_rmse=0.7186 (norm=0.5555)
2026-08-19 01:09:19,185 INFO     opponent=immobile   train_rmse=0.7227 (n=1114035)  val_rmse=0.7186 (n=279180)
2026-08-19 01:09:19,186 INFO     outcome=ball_out      train_rmse=4.8447 (n=5810)  val_rmse=4.9203 (n=1279)
2026-08-19 01:09:19,186 INFO     outcome=invalid       train_rmse=1.4241 (n=41102)  val_rmse=1.4376 (n=10405)
2026-08-19 01:09:19,186 INFO     outcome=timeout       train_rmse=2.5211 (n=8904)  val_rmse=2.4643 (n=1885)
2026-08-19 01:09:19,186 INFO     outcome=win           train_rmse=0.5374 (n=1058219)  val_rmse=0.5496 (n=265611)
2026-08-19 01:19:33,348 INFO epoch  10/100  train_rmse=0.7140 (norm=0.5519)  val_rmse=0.7218 (norm=0.5579)
2026-08-19 01:19:33,348 INFO     opponent=immobile   train_rmse=0.7140 (n=1114035)  val_rmse=0.7218 (n=279180)
2026-08-19 01:19:33,348 INFO     outcome=ball_out      train_rmse=4.7963 (n=5810)  val_rmse=4.9744 (n=1279)
2026-08-19 01:19:33,348 INFO     outcome=invalid       train_rmse=1.4014 (n=41102)  val_rmse=1.5274 (n=10405)
2026-08-19 01:19:33,348 INFO     outcome=timeout       train_rmse=2.4647 (n=8904)  val_rmse=2.4412 (n=1885)
2026-08-19 01:19:33,348 INFO     outcome=win           train_rmse=0.5320 (n=1058219)  val_rmse=0.5429 (n=265611)
2026-08-19 01:28:14,175 INFO epoch  11/100  train_rmse=0.7057 (norm=0.5455)  val_rmse=0.7430 (norm=0.5743)
2026-08-19 01:28:14,175 INFO     opponent=immobile   train_rmse=0.7057 (n=1114035)  val_rmse=0.7430 (n=279180)
2026-08-19 01:28:14,175 INFO     outcome=ball_out      train_rmse=4.7601 (n=5810)  val_rmse=5.4910 (n=1279)
2026-08-19 01:28:14,175 INFO     outcome=invalid       train_rmse=1.3626 (n=41102)  val_rmse=1.8191 (n=10405)
2026-08-19 01:28:14,175 INFO     outcome=timeout       train_rmse=2.4061 (n=8904)  val_rmse=2.6422 (n=1885)
2026-08-19 01:28:14,175 INFO     outcome=win           train_rmse=0.5282 (n=1058219)  val_rmse=0.5058 (n=265611)
2026-08-19 01:34:43,369 INFO epoch  12/100  train_rmse=0.7066 (norm=0.5462)  val_rmse=0.7090 (norm=0.5481)
2026-08-19 01:34:43,369 INFO     opponent=immobile   train_rmse=0.7066 (n=1114035)  val_rmse=0.7090 (n=279180)
2026-08-19 01:34:43,369 INFO     outcome=ball_out      train_rmse=4.7105 (n=5810)  val_rmse=4.8218 (n=1279)
2026-08-19 01:34:43,369 INFO     outcome=invalid       train_rmse=1.3298 (n=41102)  val_rmse=1.3243 (n=10405)
2026-08-19 01:34:43,369 INFO     outcome=timeout       train_rmse=2.3965 (n=8904)  val_rmse=2.4108 (n=1885)
2026-08-19 01:34:43,369 INFO     outcome=win           train_rmse=0.5355 (n=1058219)  val_rmse=0.5536 (n=265611)
2026-08-19 01:43:09,246 INFO epoch  13/100  train_rmse=0.6939 (norm=0.5364)  val_rmse=0.7279 (norm=0.5627)
2026-08-19 01:43:09,246 INFO     opponent=immobile   train_rmse=0.6939 (n=1114035)  val_rmse=0.7279 (n=279180)
2026-08-19 01:43:09,246 INFO     outcome=ball_out      train_rmse=4.6946 (n=5810)  val_rmse=4.5811 (n=1279)
2026-08-19 01:43:09,246 INFO     outcome=invalid       train_rmse=1.3115 (n=41102)  val_rmse=1.1466 (n=10405)
2026-08-19 01:43:09,246 INFO     outcome=timeout       train_rmse=2.3346 (n=8904)  val_rmse=2.0646 (n=1885)
2026-08-19 01:43:09,246 INFO     outcome=win           train_rmse=0.5227 (n=1058219)  val_rmse=0.6117 (n=265611)
2026-08-19 01:52:24,917 INFO epoch  14/100  train_rmse=0.6850 (norm=0.5295)  val_rmse=0.6844 (norm=0.5290)
2026-08-19 01:52:24,917 INFO     opponent=immobile   train_rmse=0.6850 (n=1114035)  val_rmse=0.6844 (n=279180)
2026-08-19 01:52:24,917 INFO     outcome=ball_out      train_rmse=4.6657 (n=5810)  val_rmse=4.9280 (n=1279)
2026-08-19 01:52:24,917 INFO     outcome=invalid       train_rmse=1.2760 (n=41102)  val_rmse=1.3393 (n=10405)
2026-08-19 01:52:24,917 INFO     outcome=timeout       train_rmse=2.2927 (n=8904)  val_rmse=2.2447 (n=1885)
2026-08-19 01:52:24,917 INFO     outcome=win           train_rmse=0.5167 (n=1058219)  val_rmse=0.5190 (n=265611)
2026-08-19 02:01:21,106 INFO epoch  15/100  train_rmse=0.6857 (norm=0.5300)  val_rmse=0.6964 (norm=0.5383)
2026-08-19 02:01:21,106 INFO     opponent=immobile   train_rmse=0.6857 (n=1114035)  val_rmse=0.6964 (n=279180)
2026-08-19 02:01:21,106 INFO     outcome=ball_out      train_rmse=4.6145 (n=5810)  val_rmse=4.7602 (n=1279)
2026-08-19 02:01:21,106 INFO     outcome=invalid       train_rmse=1.2562 (n=41102)  val_rmse=1.2655 (n=10405)
2026-08-19 02:01:21,106 INFO     outcome=timeout       train_rmse=2.2731 (n=8904)  val_rmse=2.0836 (n=1885)
2026-08-19 02:01:21,106 INFO     outcome=win           train_rmse=0.5228 (n=1058219)  val_rmse=0.5541 (n=265611)
2026-08-19 02:10:22,578 INFO epoch  16/100  train_rmse=0.6814 (norm=0.5267)  val_rmse=0.6915 (norm=0.5345)
2026-08-19 02:10:22,578 INFO     opponent=immobile   train_rmse=0.6814 (n=1114035)  val_rmse=0.6915 (n=279180)
2026-08-19 02:10:22,578 INFO     outcome=ball_out      train_rmse=4.6041 (n=5810)  val_rmse=4.6903 (n=1279)
2026-08-19 02:10:22,578 INFO     outcome=invalid       train_rmse=1.2311 (n=41102)  val_rmse=1.2792 (n=10405)
2026-08-19 02:10:22,578 INFO     outcome=timeout       train_rmse=2.2398 (n=8904)  val_rmse=2.1789 (n=1885)
2026-08-19 02:10:22,578 INFO     outcome=win           train_rmse=0.5209 (n=1058219)  val_rmse=0.5467 (n=265611)
2026-08-19 02:21:37,939 INFO epoch  17/100  train_rmse=0.6710 (norm=0.5187)  val_rmse=0.6930 (norm=0.5357)
2026-08-19 02:21:37,939 INFO     opponent=immobile   train_rmse=0.6710 (n=1114035)  val_rmse=0.6930 (n=279180)
2026-08-19 02:21:37,939 INFO     outcome=ball_out      train_rmse=4.5908 (n=5810)  val_rmse=5.1808 (n=1279)
2026-08-19 02:21:37,939 INFO     outcome=invalid       train_rmse=1.2327 (n=41102)  val_rmse=1.5684 (n=10405)
2026-08-19 02:21:37,940 INFO     outcome=timeout       train_rmse=2.1974 (n=8904)  val_rmse=2.3604 (n=1885)
2026-08-19 02:21:37,940 INFO     outcome=win           train_rmse=0.5085 (n=1058219)  val_rmse=0.4895 (n=265611)
2026-08-19 02:31:29,017 INFO epoch  18/100  train_rmse=0.6726 (norm=0.5199)  val_rmse=0.6749 (norm=0.5217)
2026-08-19 02:31:29,017 INFO     opponent=immobile   train_rmse=0.6726 (n=1114035)  val_rmse=0.6749 (n=279180)
2026-08-19 02:31:29,017 INFO     outcome=ball_out      train_rmse=4.5672 (n=5810)  val_rmse=4.9491 (n=1279)
2026-08-19 02:31:29,017 INFO     outcome=invalid       train_rmse=1.2133 (n=41102)  val_rmse=1.4064 (n=10405)
2026-08-19 02:31:29,017 INFO     outcome=timeout       train_rmse=2.1828 (n=8904)  val_rmse=2.1508 (n=1885)
2026-08-19 02:31:29,017 INFO     outcome=win           train_rmse=0.5142 (n=1058219)  val_rmse=0.5005 (n=265611)
2026-08-19 02:40:23,806 INFO epoch  19/100  train_rmse=0.6675 (norm=0.5159)  val_rmse=0.6882 (norm=0.5320)
2026-08-19 02:40:23,807 INFO     opponent=immobile   train_rmse=0.6675 (n=1114035)  val_rmse=0.6882 (n=279180)
2026-08-19 02:40:23,807 INFO     outcome=ball_out      train_rmse=4.5474 (n=5810)  val_rmse=4.9563 (n=1279)
2026-08-19 02:40:23,807 INFO     outcome=invalid       train_rmse=1.2040 (n=41102)  val_rmse=1.2625 (n=10405)
2026-08-19 02:40:23,807 INFO     outcome=timeout       train_rmse=2.1477 (n=8904)  val_rmse=2.2008 (n=1885)
2026-08-19 02:40:23,807 INFO     outcome=win           train_rmse=0.5102 (n=1058219)  val_rmse=0.5318 (n=265611)
2026-08-19 02:49:18,228 INFO epoch  20/100  train_rmse=0.6607 (norm=0.5107)  val_rmse=0.6688 (norm=0.5169)
2026-08-19 02:49:18,228 INFO     opponent=immobile   train_rmse=0.6607 (n=1114035)  val_rmse=0.6688 (n=279180)
2026-08-19 02:49:18,228 INFO     outcome=ball_out      train_rmse=4.5024 (n=5810)  val_rmse=4.8116 (n=1279)
2026-08-19 02:49:18,228 INFO     outcome=invalid       train_rmse=1.1754 (n=41102)  val_rmse=1.1927 (n=10405)
2026-08-19 02:49:18,228 INFO     outcome=timeout       train_rmse=2.1212 (n=8904)  val_rmse=2.1195 (n=1885)
2026-08-19 02:49:18,228 INFO     outcome=win           train_rmse=0.5067 (n=1058219)  val_rmse=0.5206 (n=265611)
2026-08-19 02:58:11,521 INFO epoch  21/100  train_rmse=0.6615 (norm=0.5113)  val_rmse=0.6787 (norm=0.5246)
2026-08-19 02:58:11,521 INFO     opponent=immobile   train_rmse=0.6615 (n=1114035)  val_rmse=0.6787 (n=279180)
2026-08-19 02:58:11,521 INFO     outcome=ball_out      train_rmse=4.4900 (n=5810)  val_rmse=5.0361 (n=1279)
2026-08-19 02:58:11,521 INFO     outcome=invalid       train_rmse=1.1772 (n=41102)  val_rmse=1.3840 (n=10405)
2026-08-19 02:58:11,521 INFO     outcome=timeout       train_rmse=2.1131 (n=8904)  val_rmse=2.3025 (n=1885)
2026-08-19 02:58:11,521 INFO     outcome=win           train_rmse=0.5085 (n=1058219)  val_rmse=0.4994 (n=265611)
2026-08-19 03:07:08,761 INFO epoch  22/100  train_rmse=0.6599 (norm=0.5101)  val_rmse=0.6854 (norm=0.5298)
2026-08-19 03:07:08,761 INFO     opponent=immobile   train_rmse=0.6599 (n=1114035)  val_rmse=0.6854 (n=279180)
2026-08-19 03:07:08,761 INFO     outcome=ball_out      train_rmse=4.4671 (n=5810)  val_rmse=4.9482 (n=1279)
2026-08-19 03:07:08,761 INFO     outcome=invalid       train_rmse=1.1665 (n=41102)  val_rmse=1.3558 (n=10405)
2026-08-19 03:07:08,761 INFO     outcome=timeout       train_rmse=2.1037 (n=8904)  val_rmse=2.1444 (n=1885)
2026-08-19 03:07:08,761 INFO     outcome=win           train_rmse=0.5088 (n=1058219)  val_rmse=0.5207 (n=265611)
2026-08-19 03:16:13,039 INFO epoch  23/100  train_rmse=0.6509 (norm=0.5032)  val_rmse=0.6806 (norm=0.5261)
2026-08-19 03:16:13,039 INFO     opponent=immobile   train_rmse=0.6509 (n=1114035)  val_rmse=0.6806 (n=279180)
2026-08-19 03:16:13,039 INFO     outcome=ball_out      train_rmse=4.4487 (n=5810)  val_rmse=4.7655 (n=1279)
2026-08-19 03:16:13,039 INFO     outcome=invalid       train_rmse=1.1535 (n=41102)  val_rmse=1.1712 (n=10405)
2026-08-19 03:16:13,039 INFO     outcome=timeout       train_rmse=2.0822 (n=8904)  val_rmse=2.1132 (n=1885)
2026-08-19 03:16:13,039 INFO     outcome=win           train_rmse=0.4992 (n=1058219)  val_rmse=0.5405 (n=265611)
2026-08-19 03:25:09,139 INFO epoch  24/100  train_rmse=0.6523 (norm=0.5042)  val_rmse=0.6896 (norm=0.5331)
2026-08-19 03:25:09,140 INFO     opponent=immobile   train_rmse=0.6523 (n=1114035)  val_rmse=0.6896 (n=279180)
2026-08-19 03:25:09,140 INFO     outcome=ball_out      train_rmse=4.4098 (n=5810)  val_rmse=4.5971 (n=1279)
2026-08-19 03:25:09,140 INFO     outcome=invalid       train_rmse=1.1474 (n=41102)  val_rmse=1.0263 (n=10405)
2026-08-19 03:25:09,140 INFO     outcome=timeout       train_rmse=2.0790 (n=8904)  val_rmse=1.8515 (n=1885)
2026-08-19 03:25:09,140 INFO     outcome=win           train_rmse=0.5037 (n=1058219)  val_rmse=0.5767 (n=265611)
2026-08-19 03:34:01,638 INFO epoch  25/100  train_rmse=0.6469 (norm=0.5001)  val_rmse=0.6900 (norm=0.5334)
2026-08-19 03:34:01,638 INFO     opponent=immobile   train_rmse=0.6469 (n=1114035)  val_rmse=0.6900 (n=279180)
2026-08-19 03:34:01,638 INFO     outcome=ball_out      train_rmse=4.4014 (n=5810)  val_rmse=5.0873 (n=1279)
2026-08-19 03:34:01,638 INFO     outcome=invalid       train_rmse=1.1224 (n=41102)  val_rmse=1.5972 (n=10405)
2026-08-19 03:34:01,638 INFO     outcome=timeout       train_rmse=2.0670 (n=8904)  val_rmse=2.0631 (n=1885)
2026-08-19 03:34:01,638 INFO     outcome=win           train_rmse=0.4993 (n=1058219)  val_rmse=0.4956 (n=265611)
2026-08-19 03:42:47,002 INFO epoch  26/100  train_rmse=0.6422 (norm=0.4964)  val_rmse=0.6857 (norm=0.5300)
2026-08-19 03:42:47,002 INFO     opponent=immobile   train_rmse=0.6422 (n=1114035)  val_rmse=0.6857 (n=279180)
2026-08-19 03:42:47,003 INFO     outcome=ball_out      train_rmse=4.3809 (n=5810)  val_rmse=4.5725 (n=1279)
2026-08-19 03:42:47,003 INFO     outcome=invalid       train_rmse=1.1234 (n=41102)  val_rmse=1.0806 (n=10405)
2026-08-19 03:42:47,003 INFO     outcome=timeout       train_rmse=2.0527 (n=8904)  val_rmse=1.9888 (n=1885)
2026-08-19 03:42:47,003 INFO     outcome=win           train_rmse=0.4942 (n=1058219)  val_rmse=0.5654 (n=265611)
2026-08-19 03:51:33,439 INFO epoch  27/100  train_rmse=0.6389 (norm=0.4939)  val_rmse=0.6683 (norm=0.5166)
2026-08-19 03:51:33,440 INFO     opponent=immobile   train_rmse=0.6389 (n=1114035)  val_rmse=0.6683 (n=279180)
2026-08-19 03:51:33,440 INFO     outcome=ball_out      train_rmse=4.3479 (n=5810)  val_rmse=4.7999 (n=1279)
2026-08-19 03:51:33,440 INFO     outcome=invalid       train_rmse=1.1020 (n=41102)  val_rmse=1.2008 (n=10405)
2026-08-19 03:51:33,440 INFO     outcome=timeout       train_rmse=2.0465 (n=8904)  val_rmse=1.9447 (n=1885)
2026-08-19 03:51:33,440 INFO     outcome=win           train_rmse=0.4935 (n=1058219)  val_rmse=0.5246 (n=265611)
2026-08-19 04:00:23,746 INFO epoch  28/100  train_rmse=0.6404 (norm=0.4950)  val_rmse=0.6884 (norm=0.5321)
2026-08-19 04:00:23,746 INFO     opponent=immobile   train_rmse=0.6404 (n=1114035)  val_rmse=0.6884 (n=279180)
2026-08-19 04:00:23,746 INFO     outcome=ball_out      train_rmse=4.3371 (n=5810)  val_rmse=5.0726 (n=1279)
2026-08-19 04:00:23,746 INFO     outcome=invalid       train_rmse=1.1046 (n=41102)  val_rmse=1.4146 (n=10405)
2026-08-19 04:00:23,746 INFO     outcome=timeout       train_rmse=2.0478 (n=8904)  val_rmse=2.1523 (n=1885)
2026-08-19 04:00:23,746 INFO     outcome=win           train_rmse=0.4958 (n=1058219)  val_rmse=0.5127 (n=265611)
2026-08-19 04:09:09,457 INFO epoch  29/100  train_rmse=0.6366 (norm=0.4921)  val_rmse=0.6747 (norm=0.5215)
2026-08-19 04:09:09,457 INFO     opponent=immobile   train_rmse=0.6366 (n=1114035)  val_rmse=0.6747 (n=279180)
2026-08-19 04:09:09,457 INFO     outcome=ball_out      train_rmse=4.3147 (n=5810)  val_rmse=4.9094 (n=1279)
2026-08-19 04:09:09,457 INFO     outcome=invalid       train_rmse=1.0940 (n=41102)  val_rmse=1.3035 (n=10405)
2026-08-19 04:09:09,457 INFO     outcome=timeout       train_rmse=2.0333 (n=8904)  val_rmse=1.9876 (n=1885)
2026-08-19 04:09:09,457 INFO     outcome=win           train_rmse=0.4931 (n=1058219)  val_rmse=0.5175 (n=265611)
2026-08-19 04:17:58,192 INFO epoch  30/100  train_rmse=0.6341 (norm=0.4902)  val_rmse=0.6697 (norm=0.5177)
2026-08-19 04:17:58,192 INFO     opponent=immobile   train_rmse=0.6341 (n=1114035)  val_rmse=0.6697 (n=279180)
2026-08-19 04:17:58,192 INFO     outcome=ball_out      train_rmse=4.2969 (n=5810)  val_rmse=4.9788 (n=1279)
2026-08-19 04:17:58,192 INFO     outcome=invalid       train_rmse=1.0978 (n=41102)  val_rmse=1.3979 (n=10405)
2026-08-19 04:17:58,192 INFO     outcome=timeout       train_rmse=2.0299 (n=8904)  val_rmse=2.0663 (n=1885)
2026-08-19 04:17:58,192 INFO     outcome=win           train_rmse=0.4904 (n=1058219)  val_rmse=0.4952 (n=265611)
2026-08-19 04:26:41,534 INFO epoch  31/100  train_rmse=0.6322 (norm=0.4887)  val_rmse=0.6731 (norm=0.5203)
2026-08-19 04:26:41,535 INFO     opponent=immobile   train_rmse=0.6322 (n=1114035)  val_rmse=0.6731 (n=279180)
2026-08-19 04:26:41,535 INFO     outcome=ball_out      train_rmse=4.2809 (n=5810)  val_rmse=4.5951 (n=1279)
2026-08-19 04:26:41,535 INFO     outcome=invalid       train_rmse=1.0976 (n=41102)  val_rmse=0.9858 (n=10405)
2026-08-19 04:26:41,535 INFO     outcome=timeout       train_rmse=2.0296 (n=8904)  val_rmse=2.0838 (n=1885)
2026-08-19 04:26:41,535 INFO     outcome=win           train_rmse=0.4886 (n=1058219)  val_rmse=0.5529 (n=265611)
2026-08-19 04:35:30,668 INFO epoch  32/100  train_rmse=0.6299 (norm=0.4869)  val_rmse=0.6637 (norm=0.5130)
2026-08-19 04:35:30,669 INFO     opponent=immobile   train_rmse=0.6299 (n=1114035)  val_rmse=0.6637 (n=279180)
2026-08-19 04:35:30,669 INFO     outcome=ball_out      train_rmse=4.2630 (n=5810)  val_rmse=4.6912 (n=1279)
2026-08-19 04:35:30,669 INFO     outcome=invalid       train_rmse=1.0673 (n=41102)  val_rmse=1.2537 (n=10405)
2026-08-19 04:35:30,669 INFO     outcome=timeout       train_rmse=2.0222 (n=8904)  val_rmse=1.9739 (n=1885)
2026-08-19 04:35:30,669 INFO     outcome=win           train_rmse=0.4891 (n=1058219)  val_rmse=0.5175 (n=265611)
2026-08-19 04:44:22,900 INFO epoch  33/100  train_rmse=0.6273 (norm=0.4849)  val_rmse=0.6798 (norm=0.5255)
2026-08-19 04:44:22,900 INFO     opponent=immobile   train_rmse=0.6273 (n=1114035)  val_rmse=0.6798 (n=279180)
2026-08-19 04:44:22,900 INFO     outcome=ball_out      train_rmse=4.2441 (n=5810)  val_rmse=4.6963 (n=1279)
2026-08-19 04:44:22,900 INFO     outcome=invalid       train_rmse=1.0687 (n=41102)  val_rmse=1.0718 (n=10405)
2026-08-19 04:44:22,900 INFO     outcome=timeout       train_rmse=2.0146 (n=8904)  val_rmse=2.1712 (n=1885)
2026-08-19 04:44:22,900 INFO     outcome=win           train_rmse=0.4867 (n=1058219)  val_rmse=0.5487 (n=265611)
2026-08-19 04:53:17,609 INFO epoch  34/100  train_rmse=0.6268 (norm=0.4845)  val_rmse=0.6711 (norm=0.5188)
2026-08-19 04:53:17,609 INFO     opponent=immobile   train_rmse=0.6268 (n=1114035)  val_rmse=0.6711 (n=279180)
2026-08-19 04:53:17,609 INFO     outcome=ball_out      train_rmse=4.2179 (n=5810)  val_rmse=4.7634 (n=1279)
2026-08-19 04:53:17,609 INFO     outcome=invalid       train_rmse=1.0616 (n=41102)  val_rmse=1.2617 (n=10405)
2026-08-19 04:53:17,609 INFO     outcome=timeout       train_rmse=2.0196 (n=8904)  val_rmse=1.9539 (n=1885)
2026-08-19 04:53:17,609 INFO     outcome=win           train_rmse=0.4876 (n=1058219)  val_rmse=0.5241 (n=265611)
2026-08-19 05:02:09,484 INFO epoch  35/100  train_rmse=0.6245 (norm=0.4828)  val_rmse=0.6692 (norm=0.5173)
2026-08-19 05:02:09,484 INFO     opponent=immobile   train_rmse=0.6245 (n=1114035)  val_rmse=0.6692 (n=279180)
2026-08-19 05:02:09,484 INFO     outcome=ball_out      train_rmse=4.1607 (n=5810)  val_rmse=4.9309 (n=1279)
2026-08-19 05:02:09,484 INFO     outcome=invalid       train_rmse=1.0403 (n=41102)  val_rmse=1.3911 (n=10405)
2026-08-19 05:02:09,484 INFO     outcome=timeout       train_rmse=2.0163 (n=8904)  val_rmse=1.9292 (n=1885)
2026-08-19 05:02:09,484 INFO     outcome=win           train_rmse=0.4892 (n=1058219)  val_rmse=0.5014 (n=265611)
2026-08-19 05:11:13,249 INFO epoch  36/100  train_rmse=0.6208 (norm=0.4798)  val_rmse=0.6798 (norm=0.5255)
2026-08-19 05:11:13,249 INFO     opponent=immobile   train_rmse=0.6208 (n=1114035)  val_rmse=0.6798 (n=279180)
2026-08-19 05:11:13,249 INFO     outcome=ball_out      train_rmse=4.1746 (n=5810)  val_rmse=4.7292 (n=1279)
2026-08-19 05:11:13,249 INFO     outcome=invalid       train_rmse=1.0631 (n=41102)  val_rmse=1.0507 (n=10405)
2026-08-19 05:11:13,249 INFO     outcome=timeout       train_rmse=2.0019 (n=8904)  val_rmse=2.0380 (n=1885)
2026-08-19 05:11:13,249 INFO     outcome=win           train_rmse=0.4820 (n=1058219)  val_rmse=0.5526 (n=265611)
2026-08-19 05:20:13,233 INFO epoch  37/100  train_rmse=0.6196 (norm=0.4790)  val_rmse=0.6599 (norm=0.5101)
2026-08-19 05:20:13,234 INFO     opponent=immobile   train_rmse=0.6196 (n=1114035)  val_rmse=0.6599 (n=279180)
2026-08-19 05:20:13,234 INFO     outcome=ball_out      train_rmse=4.1446 (n=5810)  val_rmse=5.0057 (n=1279)
2026-08-19 05:20:13,234 INFO     outcome=invalid       train_rmse=1.0365 (n=41102)  val_rmse=1.4470 (n=10405)
2026-08-19 05:20:13,234 INFO     outcome=timeout       train_rmse=2.0016 (n=8904)  val_rmse=2.0466 (n=1885)
2026-08-19 05:20:13,234 INFO     outcome=win           train_rmse=0.4842 (n=1058219)  val_rmse=0.4747 (n=265611)
2026-08-19 05:29:09,518 INFO epoch  38/100  train_rmse=0.6154 (norm=0.4757)  val_rmse=0.6705 (norm=0.5183)
2026-08-19 05:29:09,518 INFO     opponent=immobile   train_rmse=0.6154 (n=1114035)  val_rmse=0.6705 (n=279180)
2026-08-19 05:29:09,518 INFO     outcome=ball_out      train_rmse=4.1352 (n=5810)  val_rmse=4.6753 (n=1279)
2026-08-19 05:29:09,518 INFO     outcome=invalid       train_rmse=1.0407 (n=41102)  val_rmse=1.0955 (n=10405)
2026-08-19 05:29:09,518 INFO     outcome=timeout       train_rmse=1.9967 (n=8904)  val_rmse=1.9057 (n=1885)
2026-08-19 05:29:09,518 INFO     outcome=win           train_rmse=0.4788 (n=1058219)  val_rmse=0.5427 (n=265611)
2026-08-19 05:38:04,926 INFO epoch  39/100  train_rmse=0.6181 (norm=0.4778)  val_rmse=0.6605 (norm=0.5105)
2026-08-19 05:38:04,926 INFO     opponent=immobile   train_rmse=0.6181 (n=1114035)  val_rmse=0.6605 (n=279180)
2026-08-19 05:38:04,926 INFO     outcome=ball_out      train_rmse=4.1433 (n=5810)  val_rmse=4.8605 (n=1279)
2026-08-19 05:38:04,926 INFO     outcome=invalid       train_rmse=1.0596 (n=41102)  val_rmse=1.2605 (n=10405)
2026-08-19 05:38:04,926 INFO     outcome=timeout       train_rmse=1.9894 (n=8904)  val_rmse=2.0438 (n=1885)
2026-08-19 05:38:04,926 INFO     outcome=win           train_rmse=0.4807 (n=1058219)  val_rmse=0.5028 (n=265611)
2026-08-19 05:46:58,056 INFO epoch  40/100  train_rmse=0.6138 (norm=0.4744)  val_rmse=0.6754 (norm=0.5221)
2026-08-19 05:46:58,057 INFO     opponent=immobile   train_rmse=0.6138 (n=1114035)  val_rmse=0.6754 (n=279180)
2026-08-19 05:46:58,057 INFO     outcome=ball_out      train_rmse=4.1064 (n=5810)  val_rmse=4.7267 (n=1279)
2026-08-19 05:46:58,057 INFO     outcome=invalid       train_rmse=1.0403 (n=41102)  val_rmse=1.1568 (n=10405)
2026-08-19 05:46:58,057 INFO     outcome=timeout       train_rmse=1.9820 (n=8904)  val_rmse=1.8927 (n=1885)
2026-08-19 05:46:58,057 INFO     outcome=win           train_rmse=0.4784 (n=1058219)  val_rmse=0.5423 (n=265611)
2026-08-19 05:55:45,966 INFO epoch  41/100  train_rmse=0.6100 (norm=0.4715)  val_rmse=0.6628 (norm=0.5123)
2026-08-19 05:55:45,967 INFO     opponent=immobile   train_rmse=0.6100 (n=1114035)  val_rmse=0.6628 (n=279180)
2026-08-19 05:55:45,967 INFO     outcome=ball_out      train_rmse=4.0669 (n=5810)  val_rmse=4.5420 (n=1279)
2026-08-19 05:55:45,967 INFO     outcome=invalid       train_rmse=1.0162 (n=41102)  val_rmse=1.1747 (n=10405)
2026-08-19 05:55:45,967 INFO     outcome=timeout       train_rmse=1.9837 (n=8904)  val_rmse=2.0046 (n=1885)
2026-08-19 05:55:45,967 INFO     outcome=win           train_rmse=0.4772 (n=1058219)  val_rmse=0.5290 (n=265611)
2026-08-19 06:04:34,317 INFO epoch  42/100  train_rmse=0.6092 (norm=0.4709)  val_rmse=0.6682 (norm=0.5165)
2026-08-19 06:04:34,317 INFO     opponent=immobile   train_rmse=0.6092 (n=1114035)  val_rmse=0.6682 (n=279180)
2026-08-19 06:04:34,317 INFO     outcome=ball_out      train_rmse=4.0523 (n=5810)  val_rmse=4.6336 (n=1279)
2026-08-19 06:04:34,317 INFO     outcome=invalid       train_rmse=1.0192 (n=41102)  val_rmse=1.1452 (n=10405)
2026-08-19 06:04:34,317 INFO     outcome=timeout       train_rmse=1.9831 (n=8904)  val_rmse=1.9258 (n=1885)
2026-08-19 06:04:34,317 INFO     outcome=win           train_rmse=0.4766 (n=1058219)  val_rmse=0.5368 (n=265611)
2026-08-19 06:13:27,327 INFO epoch  43/100  train_rmse=0.6065 (norm=0.4688)  val_rmse=0.6578 (norm=0.5085)
2026-08-19 06:13:27,327 INFO     opponent=immobile   train_rmse=0.6065 (n=1114035)  val_rmse=0.6578 (n=279180)
2026-08-19 06:13:27,327 INFO     outcome=ball_out      train_rmse=4.0257 (n=5810)  val_rmse=4.9300 (n=1279)
2026-08-19 06:13:27,327 INFO     outcome=invalid       train_rmse=1.0125 (n=41102)  val_rmse=1.2553 (n=10405)
2026-08-19 06:13:27,327 INFO     outcome=timeout       train_rmse=1.9711 (n=8904)  val_rmse=1.9347 (n=1885)
2026-08-19 06:13:27,327 INFO     outcome=win           train_rmse=0.4751 (n=1058219)  val_rmse=0.4995 (n=265611)
2026-08-19 06:22:19,119 INFO epoch  44/100  train_rmse=0.6046 (norm=0.4674)  val_rmse=0.6824 (norm=0.5275)
2026-08-19 06:22:19,119 INFO     opponent=immobile   train_rmse=0.6046 (n=1114035)  val_rmse=0.6824 (n=279180)
2026-08-19 06:22:19,119 INFO     outcome=ball_out      train_rmse=4.0183 (n=5810)  val_rmse=4.5181 (n=1279)
2026-08-19 06:22:19,119 INFO     outcome=invalid       train_rmse=1.0057 (n=41102)  val_rmse=1.0428 (n=10405)
2026-08-19 06:22:19,119 INFO     outcome=timeout       train_rmse=1.9696 (n=8904)  val_rmse=1.9839 (n=1885)
2026-08-19 06:22:19,119 INFO     outcome=win           train_rmse=0.4736 (n=1058219)  val_rmse=0.5663 (n=265611)
2026-08-19 06:31:03,899 INFO epoch  45/100  train_rmse=0.6032 (norm=0.4663)  val_rmse=0.6733 (norm=0.5204)
2026-08-19 06:31:03,899 INFO     opponent=immobile   train_rmse=0.6032 (n=1114035)  val_rmse=0.6733 (n=279180)
2026-08-19 06:31:03,899 INFO     outcome=ball_out      train_rmse=3.9904 (n=5810)  val_rmse=4.6595 (n=1279)
2026-08-19 06:31:03,899 INFO     outcome=invalid       train_rmse=1.0015 (n=41102)  val_rmse=1.0786 (n=10405)
2026-08-19 06:31:03,899 INFO     outcome=timeout       train_rmse=1.9659 (n=8904)  val_rmse=1.9358 (n=1885)
2026-08-19 06:31:03,899 INFO     outcome=win           train_rmse=0.4735 (n=1058219)  val_rmse=0.5475 (n=265611)
2026-08-19 06:39:49,010 INFO epoch  46/100  train_rmse=0.6037 (norm=0.4666)  val_rmse=0.6638 (norm=0.5131)
2026-08-19 06:39:49,011 INFO     opponent=immobile   train_rmse=0.6037 (n=1114035)  val_rmse=0.6638 (n=279180)
2026-08-19 06:39:49,011 INFO     outcome=ball_out      train_rmse=4.0036 (n=5810)  val_rmse=4.9015 (n=1279)
2026-08-19 06:39:49,011 INFO     outcome=invalid       train_rmse=1.0262 (n=41102)  val_rmse=1.2355 (n=10405)
2026-08-19 06:39:49,011 INFO     outcome=timeout       train_rmse=1.9638 (n=8904)  val_rmse=2.0069 (n=1885)
2026-08-19 06:39:49,011 INFO     outcome=win           train_rmse=0.4715 (n=1058219)  val_rmse=0.5091 (n=265611)
2026-08-19 06:48:43,615 INFO epoch  47/100  train_rmse=0.6035 (norm=0.4665)  val_rmse=0.6648 (norm=0.5139)
2026-08-19 06:48:43,615 INFO     opponent=immobile   train_rmse=0.6035 (n=1114035)  val_rmse=0.6648 (n=279180)
2026-08-19 06:48:43,615 INFO     outcome=ball_out      train_rmse=3.9831 (n=5810)  val_rmse=4.7577 (n=1279)
2026-08-19 06:48:43,615 INFO     outcome=invalid       train_rmse=1.0185 (n=41102)  val_rmse=1.2228 (n=10405)
2026-08-19 06:48:43,615 INFO     outcome=timeout       train_rmse=1.9580 (n=8904)  val_rmse=2.0250 (n=1885)
2026-08-19 06:48:43,615 INFO     outcome=win           train_rmse=0.4730 (n=1058219)  val_rmse=0.5176 (n=265611)
2026-08-19 06:57:41,284 INFO epoch  48/100  train_rmse=0.5982 (norm=0.4624)  val_rmse=0.6726 (norm=0.5199)
2026-08-19 06:57:41,284 INFO     opponent=immobile   train_rmse=0.5982 (n=1114035)  val_rmse=0.6726 (n=279180)
2026-08-19 06:57:41,284 INFO     outcome=ball_out      train_rmse=3.9351 (n=5810)  val_rmse=4.6522 (n=1279)
2026-08-19 06:57:41,284 INFO     outcome=invalid       train_rmse=1.0018 (n=41102)  val_rmse=1.2191 (n=10405)
2026-08-19 06:57:41,284 INFO     outcome=timeout       train_rmse=1.9611 (n=8904)  val_rmse=2.0498 (n=1885)
2026-08-19 06:57:41,284 INFO     outcome=win           train_rmse=0.4695 (n=1058219)  val_rmse=0.5322 (n=265611)
2026-08-19 07:06:44,687 INFO epoch  49/100  train_rmse=0.5987 (norm=0.4628)  val_rmse=0.6945 (norm=0.5368)
2026-08-19 07:06:44,687 INFO     opponent=immobile   train_rmse=0.5987 (n=1114035)  val_rmse=0.6945 (n=279180)
2026-08-19 07:06:44,687 INFO     outcome=ball_out      train_rmse=3.9252 (n=5810)  val_rmse=4.7662 (n=1279)
2026-08-19 07:06:44,687 INFO     outcome=invalid       train_rmse=1.0074 (n=41102)  val_rmse=1.1294 (n=10405)
2026-08-19 07:06:44,687 INFO     outcome=timeout       train_rmse=1.9552 (n=8904)  val_rmse=2.2160 (n=1885)
2026-08-19 07:06:44,687 INFO     outcome=win           train_rmse=0.4703 (n=1058219)  val_rmse=0.5592 (n=265611)
2026-08-19 07:15:31,445 INFO epoch  50/100  train_rmse=0.5961 (norm=0.4608)  val_rmse=0.6599 (norm=0.5101)
2026-08-19 07:15:31,445 INFO     opponent=immobile   train_rmse=0.5961 (n=1114035)  val_rmse=0.6599 (n=279180)
2026-08-19 07:15:31,445 INFO     outcome=ball_out      train_rmse=3.9052 (n=5810)  val_rmse=4.8835 (n=1279)
2026-08-19 07:15:31,445 INFO     outcome=invalid       train_rmse=1.0076 (n=41102)  val_rmse=1.3962 (n=10405)
2026-08-19 07:15:31,445 INFO     outcome=timeout       train_rmse=1.9476 (n=8904)  val_rmse=1.9531 (n=1885)
2026-08-19 07:15:31,445 INFO     outcome=win           train_rmse=0.4680 (n=1058219)  val_rmse=0.4894 (n=265611)
2026-08-19 07:24:27,168 INFO epoch  51/100  train_rmse=0.5963 (norm=0.4609)  val_rmse=0.6756 (norm=0.5222)
2026-08-19 07:24:27,168 INFO     opponent=immobile   train_rmse=0.5963 (n=1114035)  val_rmse=0.6756 (n=279180)
2026-08-19 07:24:27,168 INFO     outcome=ball_out      train_rmse=3.9103 (n=5810)  val_rmse=4.9251 (n=1279)
2026-08-19 07:24:27,168 INFO     outcome=invalid       train_rmse=1.0040 (n=41102)  val_rmse=1.3307 (n=10405)
2026-08-19 07:24:27,168 INFO     outcome=timeout       train_rmse=1.9464 (n=8904)  val_rmse=2.0725 (n=1885)
2026-08-19 07:24:27,168 INFO     outcome=win           train_rmse=0.4683 (n=1058219)  val_rmse=0.5130 (n=265611)
2026-08-19 07:33:15,921 INFO epoch  52/100  train_rmse=0.5941 (norm=0.4592)  val_rmse=0.6739 (norm=0.5209)
2026-08-19 07:33:15,921 INFO     opponent=immobile   train_rmse=0.5941 (n=1114035)  val_rmse=0.6739 (n=279180)
2026-08-19 07:33:15,921 INFO     outcome=ball_out      train_rmse=3.8858 (n=5810)  val_rmse=5.1950 (n=1279)
2026-08-19 07:33:15,921 INFO     outcome=invalid       train_rmse=0.9977 (n=41102)  val_rmse=1.5386 (n=10405)
2026-08-19 07:33:15,921 INFO     outcome=timeout       train_rmse=1.9445 (n=8904)  val_rmse=2.0677 (n=1885)
2026-08-19 07:33:15,921 INFO     outcome=win           train_rmse=0.4671 (n=1058219)  val_rmse=0.4737 (n=265611)
2026-08-19 07:42:03,637 INFO epoch  53/100  train_rmse=0.5924 (norm=0.4579)  val_rmse=0.6795 (norm=0.5252)
2026-08-19 07:42:03,637 INFO     opponent=immobile   train_rmse=0.5924 (n=1114035)  val_rmse=0.6795 (n=279180)
2026-08-19 07:42:03,637 INFO     outcome=ball_out      train_rmse=3.8771 (n=5810)  val_rmse=4.7264 (n=1279)
2026-08-19 07:42:03,637 INFO     outcome=invalid       train_rmse=0.9959 (n=41102)  val_rmse=1.0394 (n=10405)
2026-08-19 07:42:03,637 INFO     outcome=timeout       train_rmse=1.9351 (n=8904)  val_rmse=2.0288 (n=1885)
2026-08-19 07:42:03,637 INFO     outcome=win           train_rmse=0.4657 (n=1058219)  val_rmse=0.5533 (n=265611)
2026-08-19 07:50:53,073 INFO epoch  54/100  train_rmse=0.5881 (norm=0.4546)  val_rmse=0.6702 (norm=0.5181)
2026-08-19 07:50:53,073 INFO     opponent=immobile   train_rmse=0.5881 (n=1114035)  val_rmse=0.6702 (n=279180)
2026-08-19 07:50:53,073 INFO     outcome=ball_out      train_rmse=3.8245 (n=5810)  val_rmse=4.5522 (n=1279)
2026-08-19 07:50:53,073 INFO     outcome=invalid       train_rmse=0.9785 (n=41102)  val_rmse=1.0782 (n=10405)
2026-08-19 07:50:53,073 INFO     outcome=timeout       train_rmse=1.9392 (n=8904)  val_rmse=1.9066 (n=1885)
2026-08-19 07:50:53,073 INFO     outcome=win           train_rmse=0.4637 (n=1058219)  val_rmse=0.5487 (n=265611)
2026-08-19 07:59:39,631 INFO epoch  55/100  train_rmse=0.5877 (norm=0.4542)  val_rmse=0.6679 (norm=0.5163)
2026-08-19 07:59:39,631 INFO     opponent=immobile   train_rmse=0.5877 (n=1114035)  val_rmse=0.6679 (n=279180)
2026-08-19 07:59:39,631 INFO     outcome=ball_out      train_rmse=3.7869 (n=5810)  val_rmse=4.8100 (n=1279)
2026-08-19 07:59:39,631 INFO     outcome=invalid       train_rmse=0.9793 (n=41102)  val_rmse=1.3261 (n=10405)
2026-08-19 07:59:39,631 INFO     outcome=timeout       train_rmse=1.9357 (n=8904)  val_rmse=1.9225 (n=1885)
2026-08-19 07:59:39,631 INFO     outcome=win           train_rmse=0.4648 (n=1058219)  val_rmse=0.5122 (n=265611)
2026-08-19 07:59:39,631 INFO Early stopping at epoch 55/100 (val normalized MSE did not improve for 12 epochs).
2026-08-19 07:59:39,631 INFO Best val normalized MSE achieved: 0.2586 (RMSE=0.5085; <1.0 = better than predicting the mean; <0.5 = useful critic)
2026-08-19 08:00:46,340 INFO --- Per-component MC-return magnitude (val rows) ---
2026-08-19 08:00:46,342 INFO   get_possession    mean=+0.4756  std=0.4737
2026-08-19 08:00:46,344 INFO   lose_possession   mean=-0.0045  std=0.0617
2026-08-19 08:00:46,345 INFO   ball_out          mean=-0.0167  std=0.2465
2026-08-19 08:00:46,346 INFO   box_possession    mean=+1.6761  std=0.4114
2026-08-19 08:00:46,347 INFO   speed_bonus       mean=+1.7026  std=0.7758
2026-08-19 08:00:46,349 INFO   timeout           mean=-0.0052  std=0.0643
2026-08-19 08:00:46,350 INFO   stamina_penalty   mean=-0.0687  std=0.0280
2026-08-19 08:01:05,843 INFO --- Reward-component vs. value-residual correlation (10000 val episodes) ---
2026-08-19 08:01:05,843 INFO   component            corr   comp_std
2026-08-19 08:01:05,893 INFO   lose_possession    +0.030     0.0664
2026-08-19 08:01:05,894 INFO   timeout            +0.158     0.0310
2026-08-19 08:01:05,894 INFO   stamina_penalty    -0.324     0.0265
2026-08-19 08:01:05,894 INFO   ball_out           +0.382     0.2668
2026-08-19 08:01:05,894 INFO   get_possession     +0.403     0.2550
2026-08-19 08:01:05,894 INFO   speed_bonus        +0.493     0.8725
2026-08-19 08:01:05,894 INFO   box_possession     +0.515     0.4710
2026-08-19 08:01:05,894 INFO   (components near the top -- low |corr| despite real variance -- are the ones the value net's errors track least; read alongside the per-component MC-return magnitude above.)
2026-08-19 08:01:06,023 INFO --- Worst val episode for outcome=ball_out (61 episode(s)): rows [2322237, 2322285], residual=-8.103 -- saved match log to results/debug_value_worst_episode_ball_out.json ---
2026-08-19 08:01:06,039 INFO --- Worst val episode for outcome=invalid (764 episode(s)): rows [2320045, 2320071], residual=-5.391 -- saved match log to results/debug_value_worst_episode_invalid.json ---
2026-08-19 08:01:06,056 INFO --- Worst val episode for outcome=timeout (28 episode(s)): rows [2598290, 2598423], residual=-3.928 -- saved match log to results/debug_value_worst_episode_timeout.json ---
2026-08-19 08:01:06,076 INFO --- Worst val episode for outcome=win (9147 episode(s)): rows [2758698, 2758736], residual=+6.815 -- saved match log to results/debug_value_worst_episode_win.json ---
