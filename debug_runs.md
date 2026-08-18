(footballcoach) 130 repos/FootballCoach - uv run python debug_value_network.py \
    --checkpoint checkpoints/longterm/checkpoint_vvgood_immobile.pt \
    --rollout-steps 95000 \
    --n-parallel-envs 8 \
    --worker-torch-threads 1 \
    --epochs 100 \
    --gamma 0.992 \
    --lr 7e-4 \
    --val-frac 0.2 \
    --seed 0 --patience 5 --batch-size 1096 --reset-dir-log-std --weight-decay 2e-6 --reset-value-weights 2>&1 | tee -a debug_runs.md
2026-08-17 23:48:40,299 INFO Checkpoint checkpoints/longterm/checkpoint_vvgood_immobile.pt: separate_value_net=True (auto-detected)
2026-08-17 23:48:40,327 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-17 23:48:40,328 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-17 23:48:40,328 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-17 23:48:40,331 INFO --reset-dir-log-std: reset move_dir_log_std/kick_dir_log_std to config init values before rollout collection.
2026-08-17 23:48:40,331 INFO Collecting rollout: 8 parallel worker(s), ~11875 steps/worker, worker_torch_threads=1
2026-08-17 23:48:56,508 INFO Checkpoint checkpoints/longterm/checkpoint_vvgood_immobile.pt: separate_value_net=True (auto-detected)
Rollout (8 workers): 9506/95000 ( 10.0%)   640.7 steps/s
Rollout (8 workers): 19087/95000 ( 20.1%)   705.3 steps/s
Rollout (8 workers): 28609/95000 ( 30.1%)   720.8 steps/s
Rollout (8 workers): 38016/95000 ( 40.0%)   729.4 steps/s
Rollout (8 workers): 47586/95000 ( 50.1%)   730.4 steps/s
Rollout (8 workers): 57157/95000 ( 60.2%)   733.0 steps/s
Rollout (8 workers): 66643/95000 ( 70.2%)   729.1 steps/s
Rollout (8 workers): 76001/95000 ( 80.0%)   727.7 steps/s
[worker 5] done: 116.1s total  (9.78 ms/step, 0.45 s/episode over 259 episode(s))
Rollout (8 workers): 85615/95000 ( 90.1%)   725.1 steps/s
[worker 6] done: 116.9s total  (9.84 ms/step, 0.45 s/episode over 262 episode(s))
[worker 4] done: 117.6s total  (9.91 ms/step, 0.45 s/episode over 262 episode(s))
[worker 0] done: 118.8s total  (10.01 ms/step, 0.45 s/episode over 263 episode(s))
[worker 2] done: 119.5s total  (10.07 ms/step, 0.45 s/episode over 263 episode(s))
[worker 7] done: 119.8s total  (10.09 ms/step, 0.46 s/episode over 263 episode(s))
[worker 1] done: 120.1s total  (10.11 ms/step, 0.46 s/episode over 259 episode(s))
[worker 3] done: 121.1s total  (10.20 ms/step, 0.46 s/episode over 265 episode(s))
2026-08-17 23:51:01,938 INFO   [parallel rollout] total: 125.4s wall  (1.32 ms/step aggregate, 755.7 steps/s aggregate across 8 worker(s))
2026-08-17 23:51:01,938 INFO Dropped 241 trailing (incomplete-episode) step(s) across workers
2026-08-17 23:51:02,033 INFO Rollout dataset: 94,759 steps, 2096 complete episode(s)
2026-08-17 23:51:02,499 INFO Loaded 94,759 rows total
2026-08-17 23:51:02,499 INFO has_rewards=True
2026-08-17 23:51:02,500 INFO valid_indices(): 94,759 rows (100.0% of total)
2026-08-17 23:51:02,528 INFO Returns over ALL rows: mean=3.930 std=1.069 min=-4.000 max=6.818
2026-08-17 23:51:02,528 INFO Returns over valid_indices(): mean=3.930 std=1.069
2026-08-17 23:51:02,551 INFO --- Dataset distribution (94,759 rows, 2096 episodes) ---
2026-08-17 23:51:02,552 INFO   self.ai_type == rules: 0.0%
2026-08-17 23:51:02,553 INFO   self.ai_type == immobile: 0.0%
2026-08-17 23:51:02,553 INFO   self.ai_type == neural: 100.0%
2026-08-17 23:51:02,554 INFO   opponent.ai_type == rules: 0.0%
2026-08-17 23:51:02,554 INFO   opponent.ai_type == immobile: 100.0%
2026-08-17 23:51:02,554 INFO   opponent.ai_type == neural: 0.0%
2026-08-17 23:51:02,558 INFO   valid rows: kick_this_tick rate=0.00%  tackle_attempt rate=0.00%
2026-08-17 23:51:02,558 INFO   dones=1 rows: 2,096  |  zero-reward rows: 90,867 (95.9%)
2026-08-17 23:51:02,566 INFO   return percentiles (all rows): p10=2.95  p50=3.94  p90=5.13
2026-08-17 23:51:02,611 INFO --- Reward component breakdown (all episodes, 2096 episode(s)) ---
2026-08-17 23:51:02,611 INFO   component           mean      std       min       max
2026-08-17 23:51:02,611 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,612 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,612 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,612 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,612 INFO   get_possession    +0.965    0.215    +0.000    +2.000
2026-08-17 23:51:02,612 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,613 INFO   lose_possession    -0.006    0.071    -0.900    +0.000
2026-08-17 23:51:02,613 INFO   ball_out          -0.013    0.231    -4.000    +0.000
2026-08-17 23:51:02,613 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,613 INFO   box_possession    +1.911    0.412    +0.000    +2.000
2026-08-17 23:51:02,614 INFO   speed_bonus       +2.624    0.768    +0.000    +3.890
2026-08-17 23:51:02,614 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,614 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,614 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,614 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,615 INFO   stamina_penalty    -0.074    0.031    -0.143    +0.000
2026-08-17 23:51:02,635 INFO --- Reward component breakdown (outcome=ball_out, 7 episode(s)) ---
2026-08-17 23:51:02,635 INFO   component           mean      std       min       max
2026-08-17 23:51:02,635 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,635 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,635 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,635 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,636 INFO   get_possession    +1.000    0.000    +1.000    +1.000
2026-08-17 23:51:02,636 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,636 INFO   lose_possession    +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,636 INFO   ball_out          -4.000    0.000    -4.000    -4.000
2026-08-17 23:51:02,636 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,636 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,636 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,636 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,636 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,636 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,636 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,636 INFO   stamina_penalty    +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,655 INFO --- Reward component breakdown (outcome=invalid, 86 episode(s)) ---
2026-08-17 23:51:02,655 INFO   component           mean      std       min       max
2026-08-17 23:51:02,655 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,655 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,655 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,655 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,655 INFO   get_possession    +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,655 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,655 INFO   lose_possession    +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,655 INFO   ball_out          +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,655 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,655 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,655 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,655 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,656 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,656 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,656 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,656 INFO   stamina_penalty    +0.000    0.000    +0.000    +0.000
2026-08-17 23:51:02,673 INFO --- Reward component breakdown (outcome=loss, 0 episode(s)) ---
2026-08-17 23:51:02,691 INFO --- Reward component breakdown (outcome=timeout, 0 episode(s)) ---
2026-08-17 23:51:02,741 INFO --- MC returns by outcome (all rows, 94,759 rows) ---
2026-08-17 23:51:02,743 INFO   ball_out     n=    231  mean=-2.618  std=0.302  min=-4.000  max=-1.913
2026-08-17 23:51:02,744 INFO   invalid      n=  2,237  mean=+0.000  std=0.000  min=+0.000  max=+0.000
2026-08-17 23:51:02,745 INFO   win          n= 92,291  mean=+4.042  std=0.824  min=+1.875  max=+6.818
2026-08-17 23:51:02,798 INFO --- Episode total reward by outcome (all rows, 2096 episode(s)) ---
2026-08-17 23:51:02,798 INFO   ball_out     n=     7  mean=-3.000  std=0.000  min=-3.000  max=-3.000
2026-08-17 23:51:02,798 INFO   invalid      n=    86  mean=+0.000  std=0.000  min=+0.000  max=+0.000
2026-08-17 23:51:02,798 INFO   win          n= 2,003  mean=+5.669  std=0.540  min=+3.978  max=+6.818
2026-08-17 23:51:02,846 INFO Train/val split (valid_only=True): 75,846 train rows across 1677 episodes  |  18,913 val rows across 419 episodes
2026-08-17 23:51:02,887 INFO --- Linear regression baseline (top_speed, ball_dist, ball_to_box, time_remaining) ---
2026-08-17 23:51:02,901 INFO   [all outcomes] n_train=75,846  n_val=18,913  coef(top_speed, ball_dist, ball_to_box, time_rem, intercept)=[-0.248, -2.219, -2.044, 6.669, -0.637]
    train_rmse=0.8120 (norm=0.7659)  val_rmse=0.8833 (norm=0.8331)
2026-08-17 23:51:02,955 INFO   [win outcomes only] n_train=73,965  n_val=18,326  coef(top_speed, ball_dist, ball_to_box, time_rem, intercept)=[0.227, -1.705, -2.197, 7.515, -1.6]
    train_rmse=0.3740 (norm=0.4526)  val_rmse=0.3559 (norm=0.4307)
2026-08-17 23:51:04,044 INFO value_net: checkpoint's separate value_net (value_head reset)  total_params=63,600  trainable_params=63,600
2026-08-17 23:51:04,045 INFO Fitting fresh separate value network: 50 epochs, lr=0.0007, weight_decay=2e-06, batch_size=1096, train_ret_std=1.060, outcome_reweight=False
2026-08-17 23:51:13,784 INFO epoch   0/50 (baseline, no training yet)  train_rmse=4.1073 (norm=3.8741)  val_rmse=4.0951 (norm=3.8626)
2026-08-17 23:51:13,785 INFO     opponent=immobile   train_rmse=4.1073 (n=75846)  val_rmse=4.0951 (n=18913)
2026-08-17 23:51:13,785 INFO     outcome=ball_out      train_rmse=2.6414 (n=176)  val_rmse=2.7357 (n=55)
2026-08-17 23:51:13,785 INFO     outcome=invalid       train_rmse=0.0939 (n=1705)  val_rmse=0.0940 (n=532)
2026-08-17 23:51:13,785 INFO     outcome=win           train_rmse=4.1572 (n=73965)  val_rmse=4.1575 (n=18326)
2026-08-17 23:51:30,278 INFO epoch   1/50  train_rmse=1.5659 (norm=1.4770)  val_rmse=0.7653 (norm=0.7218)
2026-08-17 23:51:30,278 INFO     opponent=immobile   train_rmse=1.5659 (n=75846)  val_rmse=0.7653 (n=18913)
2026-08-17 23:51:30,278 INFO     outcome=ball_out      train_rmse=6.2510 (n=176)  val_rmse=5.6393 (n=55)
2026-08-17 23:51:30,278 INFO     outcome=invalid       train_rmse=3.4812 (n=1705)  val_rmse=2.7522 (n=532)
2026-08-17 23:51:30,278 INFO     outcome=win           train_rmse=1.4636 (n=73965)  val_rmse=0.5377 (n=18326)
2026-08-17 23:51:46,785 INFO epoch   2/50  train_rmse=0.6848 (norm=0.6460)  val_rmse=0.6489 (norm=0.6120)
2026-08-17 23:51:46,785 INFO     opponent=immobile   train_rmse=0.6848 (n=75846)  val_rmse=0.6489 (n=18913)
2026-08-17 23:51:46,785 INFO     outcome=ball_out      train_rmse=5.9936 (n=176)  val_rmse=5.1344 (n=55)
2026-08-17 23:51:46,785 INFO     outcome=invalid       train_rmse=2.6647 (n=1705)  val_rmse=2.5902 (n=532)
2026-08-17 23:51:46,785 INFO     outcome=win           train_rmse=0.4814 (n=73965)  val_rmse=0.4008 (n=18326)
2026-08-17 23:52:03,044 INFO epoch   3/50  train_rmse=0.6158 (norm=0.5808)  val_rmse=0.6242 (norm=0.5888)
2026-08-17 23:52:03,044 INFO     opponent=immobile   train_rmse=0.6158 (n=75846)  val_rmse=0.6242 (n=18913)
2026-08-17 23:52:03,044 INFO     outcome=ball_out      train_rmse=5.8639 (n=176)  val_rmse=4.6801 (n=55)
2026-08-17 23:52:03,044 INFO     outcome=invalid       train_rmse=2.0601 (n=1705)  val_rmse=2.4000 (n=532)
2026-08-17 23:52:03,044 INFO     outcome=win           train_rmse=0.4574 (n=73965)  val_rmse=0.4114 (n=18326)
2026-08-17 23:52:19,379 INFO epoch   4/50  train_rmse=0.5963 (norm=0.5624)  val_rmse=0.6052 (norm=0.5708)
2026-08-17 23:52:19,379 INFO     opponent=immobile   train_rmse=0.5963 (n=75846)  val_rmse=0.6052 (n=18913)
2026-08-17 23:52:19,379 INFO     outcome=ball_out      train_rmse=5.7132 (n=176)  val_rmse=4.0600 (n=55)
2026-08-17 23:52:19,379 INFO     outcome=invalid       train_rmse=1.8725 (n=1705)  val_rmse=2.2243 (n=532)
2026-08-17 23:52:19,379 INFO     outcome=win           train_rmse=0.4540 (n=73965)  val_rmse=0.4300 (n=18326)
2026-08-17 23:52:35,751 INFO epoch   5/50  train_rmse=0.5702 (norm=0.5378)  val_rmse=0.6457 (norm=0.6091)
2026-08-17 23:52:35,751 INFO     opponent=immobile   train_rmse=0.5702 (n=75846)  val_rmse=0.6457 (n=18913)
2026-08-17 23:52:35,751 INFO     outcome=ball_out      train_rmse=5.5722 (n=176)  val_rmse=5.2198 (n=55)
2026-08-17 23:52:35,751 INFO     outcome=invalid       train_rmse=1.5950 (n=1705)  val_rmse=2.7457 (n=532)
2026-08-17 23:52:35,751 INFO     outcome=win           train_rmse=0.4482 (n=73965)  val_rmse=0.3601 (n=18326)
2026-08-17 23:52:51,906 INFO epoch   6/50  train_rmse=0.5592 (norm=0.5275)  val_rmse=0.6352 (norm=0.5991)
2026-08-17 23:52:51,906 INFO     opponent=immobile   train_rmse=0.5592 (n=75846)  val_rmse=0.6352 (n=18913)
2026-08-17 23:52:51,906 INFO     outcome=ball_out      train_rmse=5.4792 (n=176)  val_rmse=4.9954 (n=55)
2026-08-17 23:52:51,906 INFO     outcome=invalid       train_rmse=1.5922 (n=1705)  val_rmse=2.6783 (n=532)
2026-08-17 23:52:51,906 INFO     outcome=win           train_rmse=0.4368 (n=73965)  val_rmse=0.3651 (n=18326)
2026-08-17 23:53:08,163 INFO epoch   7/50  train_rmse=0.5661 (norm=0.5340)  val_rmse=0.5960 (norm=0.5622)
2026-08-17 23:53:08,163 INFO     opponent=immobile   train_rmse=0.5661 (n=75846)  val_rmse=0.5960 (n=18913)
2026-08-17 23:53:08,164 INFO     outcome=ball_out      train_rmse=5.3100 (n=176)  val_rmse=4.1783 (n=55)
2026-08-17 23:53:08,164 INFO     outcome=invalid       train_rmse=1.5178 (n=1705)  val_rmse=2.3223 (n=532)
2026-08-17 23:53:08,164 INFO     outcome=win           train_rmse=0.4565 (n=73965)  val_rmse=0.3970 (n=18326)
2026-08-17 23:53:24,454 INFO epoch   8/50  train_rmse=0.5522 (norm=0.5209)  val_rmse=0.6091 (norm=0.5745)
2026-08-17 23:53:24,454 INFO     opponent=immobile   train_rmse=0.5522 (n=75846)  val_rmse=0.6091 (n=18913)
2026-08-17 23:53:24,454 INFO     outcome=ball_out      train_rmse=5.3672 (n=176)  val_rmse=3.8623 (n=55)
2026-08-17 23:53:24,454 INFO     outcome=invalid       train_rmse=1.5025 (n=1705)  val_rmse=2.1906 (n=532)
2026-08-17 23:53:24,454 INFO     outcome=win           train_rmse=0.4383 (n=73965)  val_rmse=0.4459 (n=18326)
2026-08-17 23:53:40,587 INFO epoch   9/50  train_rmse=0.5420 (norm=0.5112)  val_rmse=0.6196 (norm=0.5844)
2026-08-17 23:53:40,587 INFO     opponent=immobile   train_rmse=0.5420 (n=75846)  val_rmse=0.6196 (n=18913)
2026-08-17 23:53:40,587 INFO     outcome=ball_out      train_rmse=5.2126 (n=176)  val_rmse=4.7323 (n=55)
2026-08-17 23:53:40,587 INFO     outcome=invalid       train_rmse=1.4445 (n=1705)  val_rmse=2.4932 (n=532)
2026-08-17 23:53:40,587 INFO     outcome=win           train_rmse=0.4341 (n=73965)  val_rmse=0.3854 (n=18326)
2026-08-17 23:53:56,799 INFO epoch  10/50  train_rmse=0.5278 (norm=0.4978)  val_rmse=0.5878 (norm=0.5544)
2026-08-17 23:53:56,799 INFO     opponent=immobile   train_rmse=0.5278 (n=75846)  val_rmse=0.5878 (n=18913)
2026-08-17 23:53:56,799 INFO     outcome=ball_out      train_rmse=5.1432 (n=176)  val_rmse=3.5296 (n=55)
2026-08-17 23:53:56,799 INFO     outcome=invalid       train_rmse=1.4147 (n=1705)  val_rmse=2.1506 (n=532)
2026-08-17 23:53:56,799 INFO     outcome=win           train_rmse=0.4202 (n=73965)  val_rmse=0.4300 (n=18326)
2026-08-17 23:54:12,911 INFO epoch  11/50  train_rmse=0.5280 (norm=0.4980)  val_rmse=0.5784 (norm=0.5456)
2026-08-17 23:54:12,911 INFO     opponent=immobile   train_rmse=0.5280 (n=75846)  val_rmse=0.5784 (n=18913)
2026-08-17 23:54:12,911 INFO     outcome=ball_out      train_rmse=4.9764 (n=176)  val_rmse=3.7956 (n=55)
2026-08-17 23:54:12,911 INFO     outcome=invalid       train_rmse=1.4045 (n=1705)  val_rmse=2.3048 (n=532)
2026-08-17 23:54:12,911 INFO     outcome=win           train_rmse=0.4260 (n=73965)  val_rmse=0.3845 (n=18326)
2026-08-17 23:54:28,885 INFO epoch  12/50  train_rmse=0.5186 (norm=0.4891)  val_rmse=0.5643 (norm=0.5322)
2026-08-17 23:54:28,885 INFO     opponent=immobile   train_rmse=0.5186 (n=75846)  val_rmse=0.5643 (n=18913)
2026-08-17 23:54:28,885 INFO     outcome=ball_out      train_rmse=4.9697 (n=176)  val_rmse=3.9367 (n=55)
2026-08-17 23:54:28,885 INFO     outcome=invalid       train_rmse=1.3719 (n=1705)  val_rmse=2.1729 (n=532)
2026-08-17 23:54:28,885 INFO     outcome=win           train_rmse=0.4166 (n=73965)  val_rmse=0.3808 (n=18326)
2026-08-17 23:54:45,573 INFO epoch  13/50  train_rmse=0.5159 (norm=0.4866)  val_rmse=0.5803 (norm=0.5473)
2026-08-17 23:54:45,573 INFO     opponent=immobile   train_rmse=0.5159 (n=75846)  val_rmse=0.5803 (n=18913)
2026-08-17 23:54:45,573 INFO     outcome=ball_out      train_rmse=4.8390 (n=176)  val_rmse=4.0484 (n=55)
2026-08-17 23:54:45,573 INFO     outcome=invalid       train_rmse=1.3321 (n=1705)  val_rmse=2.2247 (n=532)
2026-08-17 23:54:45,574 INFO     outcome=win           train_rmse=0.4198 (n=73965)  val_rmse=0.3932 (n=18326)
2026-08-17 23:55:02,554 INFO epoch  14/50  train_rmse=0.5112 (norm=0.4821)  val_rmse=0.5539 (norm=0.5224)
2026-08-17 23:55:02,554 INFO     opponent=immobile   train_rmse=0.5112 (n=75846)  val_rmse=0.5539 (n=18913)
2026-08-17 23:55:02,554 INFO     outcome=ball_out      train_rmse=4.8855 (n=176)  val_rmse=3.6374 (n=55)
2026-08-17 23:55:02,554 INFO     outcome=invalid       train_rmse=1.3199 (n=1705)  val_rmse=1.8504 (n=532)
2026-08-17 23:55:02,554 INFO     outcome=win           train_rmse=0.4135 (n=73965)  val_rmse=0.4213 (n=18326)
2026-08-17 23:55:18,886 INFO epoch  15/50  train_rmse=0.5097 (norm=0.4808)  val_rmse=0.5823 (norm=0.5493)
2026-08-17 23:55:18,886 INFO     opponent=immobile   train_rmse=0.5097 (n=75846)  val_rmse=0.5823 (n=18913)
2026-08-17 23:55:18,886 INFO     outcome=ball_out      train_rmse=5.0064 (n=176)  val_rmse=4.1591 (n=55)
2026-08-17 23:55:18,886 INFO     outcome=invalid       train_rmse=1.3532 (n=1705)  val_rmse=2.2629 (n=532)
2026-08-17 23:55:18,886 INFO     outcome=win           train_rmse=0.4057 (n=73965)  val_rmse=0.3866 (n=18326)
2026-08-17 23:55:35,657 INFO epoch  16/50  train_rmse=0.5060 (norm=0.4773)  val_rmse=0.5757 (norm=0.5430)
2026-08-17 23:55:35,657 INFO     opponent=immobile   train_rmse=0.5060 (n=75846)  val_rmse=0.5757 (n=18913)
2026-08-17 23:55:35,657 INFO     outcome=ball_out      train_rmse=4.8676 (n=176)  val_rmse=4.2641 (n=55)
2026-08-17 23:55:35,657 INFO     outcome=invalid       train_rmse=1.3325 (n=1705)  val_rmse=2.4503 (n=532)
2026-08-17 23:55:35,657 INFO     outcome=win           train_rmse=0.4065 (n=73965)  val_rmse=0.3363 (n=18326)
2026-08-17 23:55:52,542 INFO epoch  17/50  train_rmse=0.4992 (norm=0.4709)  val_rmse=0.6449 (norm=0.6083)
2026-08-17 23:55:52,542 INFO     opponent=immobile   train_rmse=0.4992 (n=75846)  val_rmse=0.6449 (n=18913)
2026-08-17 23:55:52,542 INFO     outcome=ball_out      train_rmse=4.7952 (n=176)  val_rmse=4.1115 (n=55)
2026-08-17 23:55:52,542 INFO     outcome=invalid       train_rmse=1.3040 (n=1705)  val_rmse=1.9885 (n=532)
2026-08-17 23:55:52,542 INFO     outcome=win           train_rmse=0.4021 (n=73965)  val_rmse=0.5135 (n=18326)
2026-08-17 23:56:09,742 INFO epoch  18/50  train_rmse=0.5100 (norm=0.4811)  val_rmse=0.5812 (norm=0.5482)
2026-08-17 23:56:09,742 INFO     opponent=immobile   train_rmse=0.5100 (n=75846)  val_rmse=0.5812 (n=18913)
2026-08-17 23:56:09,742 INFO     outcome=ball_out      train_rmse=4.9559 (n=176)  val_rmse=3.7993 (n=55)
2026-08-17 23:56:09,742 INFO     outcome=invalid       train_rmse=1.3168 (n=1705)  val_rmse=2.1239 (n=532)
2026-08-17 23:56:09,742 INFO     outcome=win           train_rmse=0.4103 (n=73965)  val_rmse=0.4175 (n=18326)
2026-08-17 23:56:27,030 INFO epoch  19/50  train_rmse=0.5096 (norm=0.4806)  val_rmse=0.5852 (norm=0.5520)
2026-08-17 23:56:27,030 INFO     opponent=immobile   train_rmse=0.5096 (n=75846)  val_rmse=0.5852 (n=18913)
2026-08-17 23:56:27,030 INFO     outcome=ball_out      train_rmse=4.9123 (n=176)  val_rmse=3.4707 (n=55)
2026-08-17 23:56:27,030 INFO     outcome=invalid       train_rmse=1.3490 (n=1705)  val_rmse=2.3712 (n=532)
2026-08-17 23:56:27,030 INFO     outcome=win           train_rmse=0.4085 (n=73965)  val_rmse=0.3925 (n=18326)
2026-08-17 23:56:27,030 INFO Early stopping at epoch 19/50 (val normalized MSE did not improve for 5 epochs).
2026-08-17 23:56:27,030 INFO Best val normalized MSE achieved: 0.2729 (RMSE=0.5224; <1.0 = better than predicting the mean; <0.5 = useful critic)
2026-08-17 23:56:29,311 INFO --- Per-component MC-return magnitude (val rows) ---
2026-08-17 23:56:29,311 INFO   get_possession    mean=+0.3755  std=0.4565
2026-08-17 23:56:29,312 INFO   lose_possession   mean=-0.0040  std=0.0560
2026-08-17 23:56:29,312 INFO   ball_out          mean=-0.0105  std=0.1940
2026-08-17 23:56:29,312 INFO   box_possession    mean=+1.5835  std=0.3653
2026-08-17 23:56:29,312 INFO   speed_bonus       mean=+2.0273  std=0.6889
2026-08-17 23:56:29,312 INFO   stamina_penalty   mean=-0.0625  std=0.0252
2026-08-17 23:56:29,650 INFO --- Reward-component vs. value-residual correlation (419 val episodes) ---
2026-08-17 23:56:29,650 INFO   component            corr   comp_std
2026-08-17 23:56:29,661 INFO   lose_possession    +0.038     0.0597
2026-08-17 23:56:29,661 INFO   ball_out           +0.304     0.2231
2026-08-17 23:56:29,661 INFO   stamina_penalty    -0.354     0.0211
2026-08-17 23:56:29,661 INFO   get_possession     +0.488     0.2012
2026-08-17 23:56:29,661 INFO   speed_bonus        +0.489     0.7848
2026-08-17 23:56:29,661 INFO   box_possession     +0.565     0.3723
2026-08-17 23:56:29,661 INFO   (components near the top -- low |corr| despite real variance -- are the ones the value net's errors track least; read alongside the per-component MC-return magnitude above.)
2026-08-17 23:56:29,665 INFO --- Worst val episode for outcome=ball_out (2 episode(s)): rows [79421, 79443], residual=-3.614 -- saved match log to results/debug_value_worst_episode_ball_out.json ---
2026-08-17 23:56:29,666 INFO --- Worst val episode for outcome=invalid (19 episode(s)): rows [78936, 78964], residual=-4.009 -- saved match log to results/debug_value_worst_episode_invalid.json ---
2026-08-17 23:56:29,667 INFO --- Worst val episode for outcome=win (398 episode(s)): rows [86426, 86434], residual=+5.078 -- saved match log to results/debug_value_worst_episode_win.json ---
2026-08-17 23:59:57,879 INFO Checkpoint checkpoints/longterm/checkpoint_vvgood_immobile.pt: separate_value_net=True (auto-detected)
2026-08-17 23:59:57,904 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-17 23:59:57,905 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-17 23:59:57,905 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-17 23:59:57,908 INFO --reset-dir-log-std: reset move_dir_log_std/kick_dir_log_std to config init values before rollout collection.
2026-08-17 23:59:57,909 INFO Collecting rollout: 8 parallel worker(s), ~11875 steps/worker, worker_torch_threads=1
Rollout (8 workers): 0/95000 (  0.0%)     0.0 steps/s
2026-08-18 00:00:00,785 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 00:00:00,787 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 00:00:00,787 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 00:00:00,795 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 00:00:00,796 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 00:00:00,796 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 00:00:00,824 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 00:00:00,826 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 00:00:00,826 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 00:00:00,834 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 00:00:00,836 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 00:00:00,836 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 00:00:00,845 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 00:00:00,847 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 00:00:00,847 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 00:00:00,888 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 00:00:00,890 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 00:00:00,890 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 00:00:00,895 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 00:00:00,898 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 00:00:00,898 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 00:00:00,930 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 00:00:00,933 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 00:00:00,933 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
Rollout (8 workers): 9650/95000 ( 10.2%)   668.6 steps/s
Rollout (8 workers): 19123/95000 ( 20.1%)   722.7 steps/s
Rollout (8 workers): 28660/95000 ( 30.2%)   737.0 steps/s
Rollout (8 workers): 38005/95000 ( 40.0%)   740.6 steps/s
Rollout (8 workers): 47556/95000 ( 50.1%)   743.7 steps/s
Rollout (8 workers): 57060/95000 ( 60.1%)   747.1 steps/s
Rollout (8 workers): 66562/95000 ( 70.1%)   749.6 steps/s
Rollout (8 workers): 76062/95000 ( 80.1%)   749.9 steps/s
Rollout (8 workers): 85658/95000 ( 90.2%)   749.7 steps/s
[worker 4] done: 113.1s total  (9.53 ms/step, 0.42 s/episode over 267 episode(s))
[worker 6] done: 113.3s total  (9.54 ms/step, 0.42 s/episode over 270 episode(s))
[worker 7] done: 113.4s total  (9.55 ms/step, 0.42 s/episode over 273 episode(s))
[worker 1] done: 114.9s total  (9.68 ms/step, 0.42 s/episode over 273 episode(s))
[worker 2] done: 115.9s total  (9.76 ms/step, 0.43 s/episode over 272 episode(s))
[worker 0] done: 116.3s total  (9.79 ms/step, 0.44 s/episode over 265 episode(s))
[worker 3] done: 116.8s total  (9.84 ms/step, 0.44 s/episode over 267 episode(s))
[worker 5] done: 117.1s total  (9.86 ms/step, 0.43 s/episode over 271 episode(s))
2026-08-18 00:01:59,321 INFO   [parallel rollout] total: 121.4s wall  (1.28 ms/step aggregate, 780.4 steps/s aggregate across 8 worker(s))
2026-08-18 00:01:59,321 INFO Dropped 252 trailing (incomplete-episode) step(s) across workers
2026-08-18 00:01:59,404 INFO Rollout dataset: 94,748 steps, 2158 complete episode(s)
2026-08-18 00:01:59,829 INFO Loaded 94,748 rows total
2026-08-18 00:01:59,829 INFO has_rewards=True
2026-08-18 00:01:59,831 INFO valid_indices(): 94,748 rows (100.0% of total)
2026-08-18 00:01:59,856 INFO Returns over ALL rows: mean=4.000 std=1.038 min=-4.000 max=6.818
2026-08-18 00:01:59,857 INFO Returns over valid_indices(): mean=4.000 std=1.038
2026-08-18 00:01:59,877 INFO --- Dataset distribution (94,748 rows, 2158 episodes) ---
2026-08-18 00:01:59,878 INFO   self.ai_type == rules: 0.0%
2026-08-18 00:01:59,880 INFO   self.ai_type == immobile: 0.0%
2026-08-18 00:01:59,881 INFO   self.ai_type == neural: 100.0%
2026-08-18 00:01:59,882 INFO   opponent.ai_type == rules: 0.0%
2026-08-18 00:01:59,882 INFO   opponent.ai_type == immobile: 100.0%
2026-08-18 00:01:59,882 INFO   opponent.ai_type == neural: 0.0%
2026-08-18 00:01:59,886 INFO   valid rows: kick_this_tick rate=0.00%  tackle_attempt rate=0.00%
2026-08-18 00:01:59,886 INFO   dones=1 rows: 2,158  |  zero-reward rows: 90,677 (95.7%)
2026-08-18 00:01:59,894 INFO   return percentiles (all rows): p10=3.03  p50=4.00  p90=5.17
2026-08-18 00:01:59,937 INFO --- Reward component breakdown (all episodes, 2158 episode(s)) ---
2026-08-18 00:01:59,937 INFO   component           mean      std       min       max
2026-08-18 00:01:59,938 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,938 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,938 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,938 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,938 INFO   get_possession    +0.978    0.221    +0.000    +2.000
2026-08-18 00:01:59,939 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,939 INFO   lose_possession    -0.012    0.104    -0.900    +0.000
2026-08-18 00:01:59,939 INFO   ball_out          -0.009    0.192    -4.000    +0.000
2026-08-18 00:01:59,939 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,939 INFO   box_possession    +1.924    0.382    +0.000    +2.000
2026-08-18 00:01:59,939 INFO   speed_bonus       +2.681    0.732    +0.000    +3.890
2026-08-18 00:01:59,940 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,940 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,940 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,940 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,940 INFO   stamina_penalty    -0.074    0.030    -0.141    +0.000
2026-08-18 00:01:59,960 INFO --- Reward component breakdown (outcome=ball_out, 5 episode(s)) ---
2026-08-18 00:01:59,960 INFO   component           mean      std       min       max
2026-08-18 00:01:59,960 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,960 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,960 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,961 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,961 INFO   get_possession    +1.000    0.000    +1.000    +1.000
2026-08-18 00:01:59,961 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,961 INFO   lose_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,961 INFO   ball_out          -4.000    0.000    -4.000    -4.000
2026-08-18 00:01:59,961 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,961 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,961 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,961 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,961 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,961 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,961 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,961 INFO   stamina_penalty    +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,980 INFO --- Reward component breakdown (outcome=invalid, 77 episode(s)) ---
2026-08-18 00:01:59,980 INFO   component           mean      std       min       max
2026-08-18 00:01:59,980 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,980 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,980 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,980 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,980 INFO   get_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,980 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,980 INFO   lose_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,980 INFO   ball_out          +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,980 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,980 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,980 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,980 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,980 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,980 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,980 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,980 INFO   stamina_penalty    +0.000    0.000    +0.000    +0.000
2026-08-18 00:01:59,997 INFO --- Reward component breakdown (outcome=loss, 0 episode(s)) ---
2026-08-18 00:02:00,014 INFO --- Reward component breakdown (outcome=timeout, 0 episode(s)) ---
2026-08-18 00:02:00,057 INFO --- MC returns by outcome (all rows, 94,748 rows) ---
2026-08-18 00:02:00,059 INFO   ball_out     n=    184  mean=-2.578  std=0.321  min=-4.000  max=-1.838
2026-08-18 00:02:00,060 INFO   invalid      n=  2,029  mean=+0.000  std=0.000  min=+0.000  max=+0.000
2026-08-18 00:02:00,062 INFO   win          n= 92,535  mean=+4.100  std=0.811  min=+1.796  max=+6.818
2026-08-18 00:02:00,114 INFO --- Episode total reward by outcome (all rows, 2158 episode(s)) ---
2026-08-18 00:02:00,115 INFO   ball_out     n=     5  mean=-3.000  std=0.000  min=-3.000  max=-3.000
2026-08-18 00:02:00,115 INFO   invalid      n=    77  mean=+0.000  std=0.000  min=+0.000  max=+0.000
2026-08-18 00:02:00,115 INFO   win          n= 2,076  mean=+5.711  std=0.519  min=+3.909  max=+6.818
2026-08-18 00:02:00,162 INFO Train/val split (valid_only=True): 75,856 train rows across 1726 episodes  |  18,892 val rows across 432 episodes
2026-08-18 00:02:00,202 INFO --- Linear regression baseline (top_speed, ball_dist, ball_to_box, time_remaining) ---
2026-08-18 00:02:00,212 INFO   [all outcomes] n_train=75,856  n_val=18,892  coef(top_speed, ball_dist, ball_to_box, time_rem, intercept)=[0.383, -1.65, -2.075, 6.662, -1.094]
    train_rmse=0.7830 (norm=0.7536)  val_rmse=0.7840 (norm=0.7545)
2026-08-18 00:02:00,264 INFO   [win outcomes only] n_train=74,110  n_val=18,425  coef(top_speed, ball_dist, ball_to_box, time_rem, intercept)=[0.183, -1.338, -2.185, 7.484, -1.542]
    train_rmse=0.3724 (norm=0.4576)  val_rmse=0.3430 (norm=0.4215)
2026-08-18 00:02:01,539 INFO value_net: checkpoint's separate value_net (value_head reset)  total_params=63,600  trainable_params=63,600
2026-08-18 00:02:01,539 INFO Fitting fresh separate value network: 50 epochs, lr=0.0007, weight_decay=2e-06, batch_size=1096, train_ret_std=1.039, outcome_reweight=False
2026-08-18 00:02:11,231 INFO epoch   0/50 (baseline, no training yet)  train_rmse=4.1635 (norm=4.0071)  val_rmse=4.1695 (norm=4.0129)
2026-08-18 00:02:11,231 INFO     opponent=immobile   train_rmse=4.1635 (n=75856)  val_rmse=4.1695 (n=18892)
2026-08-18 00:02:11,231 INFO     outcome=ball_out      train_rmse=2.6035 (n=153)  val_rmse=2.7178 (n=31)
2026-08-18 00:02:11,231 INFO     outcome=invalid       train_rmse=0.0941 (n=1593)  val_rmse=0.0880 (n=436)
2026-08-18 00:02:11,231 INFO     outcome=win           train_rmse=4.2105 (n=74110)  val_rmse=4.2205 (n=18425)
2026-08-18 00:02:27,611 INFO epoch   1/50  train_rmse=1.5591 (norm=1.5005)  val_rmse=0.6603 (norm=0.6355)
2026-08-18 00:02:27,611 INFO     opponent=immobile   train_rmse=1.5591 (n=75856)  val_rmse=0.6603 (n=18892)
2026-08-18 00:02:27,611 INFO     outcome=ball_out      train_rmse=5.8620 (n=153)  val_rmse=5.7709 (n=31)
2026-08-18 00:02:27,611 INFO     outcome=invalid       train_rmse=3.5219 (n=1593)  val_rmse=2.9660 (n=436)
2026-08-18 00:02:27,611 INFO     outcome=win           train_rmse=1.4664 (n=74110)  val_rmse=0.4276 (n=18425)
2026-08-18 00:02:43,930 INFO epoch   2/50  train_rmse=0.6250 (norm=0.6015)  val_rmse=0.5876 (norm=0.5655)
2026-08-18 00:02:43,930 INFO     opponent=immobile   train_rmse=0.6250 (n=75856)  val_rmse=0.5876 (n=18892)
2026-08-18 00:02:43,930 INFO     outcome=ball_out      train_rmse=5.4527 (n=153)  val_rmse=5.2660 (n=31)
2026-08-18 00:02:43,930 INFO     outcome=invalid       train_rmse=2.5560 (n=1593)  val_rmse=2.1969 (n=436)
2026-08-18 00:02:43,930 INFO     outcome=win           train_rmse=0.4450 (n=74110)  val_rmse=0.4395 (n=18425)
2026-08-18 00:03:00,386 INFO epoch   3/50  train_rmse=0.5418 (norm=0.5214)  val_rmse=0.5244 (norm=0.5047)
2026-08-18 00:03:00,386 INFO     opponent=immobile   train_rmse=0.5418 (n=75856)  val_rmse=0.5244 (n=18892)
2026-08-18 00:03:00,386 INFO     outcome=ball_out      train_rmse=4.9248 (n=153)  val_rmse=3.9819 (n=31)
2026-08-18 00:03:00,386 INFO     outcome=invalid       train_rmse=1.7787 (n=1593)  val_rmse=1.8075 (n=436)
2026-08-18 00:03:00,386 INFO     outcome=win           train_rmse=0.4270 (n=74110)  val_rmse=0.4218 (n=18425)
2026-08-18 00:03:16,862 INFO epoch   4/50  train_rmse=0.5082 (norm=0.4891)  val_rmse=0.5355 (norm=0.5154)
2026-08-18 00:03:16,863 INFO     opponent=immobile   train_rmse=0.5082 (n=75856)  val_rmse=0.5355 (n=18892)
2026-08-18 00:03:16,863 INFO     outcome=ball_out      train_rmse=4.0726 (n=153)  val_rmse=4.8224 (n=31)
2026-08-18 00:03:16,863 INFO     outcome=invalid       train_rmse=1.5426 (n=1593)  val_rmse=1.7403 (n=436)
2026-08-18 00:03:16,863 INFO     outcome=win           train_rmse=0.4230 (n=74110)  val_rmse=0.4281 (n=18425)
2026-08-18 00:03:33,093 INFO epoch   5/50  train_rmse=0.4990 (norm=0.4803)  val_rmse=0.5409 (norm=0.5206)
2026-08-18 00:03:33,093 INFO     opponent=immobile   train_rmse=0.4990 (n=75856)  val_rmse=0.5409 (n=18892)
2026-08-18 00:03:33,093 INFO     outcome=ball_out      train_rmse=3.7476 (n=153)  val_rmse=5.2521 (n=31)
2026-08-18 00:03:33,093 INFO     outcome=invalid       train_rmse=1.4852 (n=1593)  val_rmse=1.8248 (n=436)
2026-08-18 00:03:33,093 INFO     outcome=win           train_rmse=0.4224 (n=74110)  val_rmse=0.4180 (n=18425)
2026-08-18 00:03:49,298 INFO epoch   6/50  train_rmse=0.4842 (norm=0.4660)  val_rmse=0.4631 (norm=0.4457)
2026-08-18 00:03:49,298 INFO     opponent=immobile   train_rmse=0.4842 (n=75856)  val_rmse=0.4631 (n=18892)
2026-08-18 00:03:49,298 INFO     outcome=ball_out      train_rmse=3.7967 (n=153)  val_rmse=2.8340 (n=31)
2026-08-18 00:03:49,298 INFO     outcome=invalid       train_rmse=1.3580 (n=1593)  val_rmse=1.2700 (n=436)
2026-08-18 00:03:49,298 INFO     outcome=win           train_rmse=0.4130 (n=74110)  val_rmse=0.4102 (n=18425)
2026-08-18 00:04:05,600 INFO epoch   7/50  train_rmse=0.4690 (norm=0.4514)  val_rmse=0.5269 (norm=0.5071)
2026-08-18 00:04:05,600 INFO     opponent=immobile   train_rmse=0.4690 (n=75856)  val_rmse=0.5269 (n=18892)
2026-08-18 00:04:05,600 INFO     outcome=ball_out      train_rmse=3.4195 (n=153)  val_rmse=2.7157 (n=31)
2026-08-18 00:04:05,600 INFO     outcome=invalid       train_rmse=1.3012 (n=1593)  val_rmse=1.7457 (n=436)
2026-08-18 00:04:05,600 INFO     outcome=win           train_rmse=0.4058 (n=74110)  val_rmse=0.4474 (n=18425)
2026-08-18 00:04:21,939 INFO epoch   8/50  train_rmse=0.4605 (norm=0.4432)  val_rmse=0.4621 (norm=0.4448)
2026-08-18 00:04:21,939 INFO     opponent=immobile   train_rmse=0.4605 (n=75856)  val_rmse=0.4621 (n=18892)
2026-08-18 00:04:21,939 INFO     outcome=ball_out      train_rmse=3.0555 (n=153)  val_rmse=3.6677 (n=31)
2026-08-18 00:04:21,939 INFO     outcome=invalid       train_rmse=1.2741 (n=1593)  val_rmse=1.6600 (n=436)
2026-08-18 00:04:21,939 INFO     outcome=win           train_rmse=0.4036 (n=74110)  val_rmse=0.3621 (n=18425)
2026-08-18 00:04:38,140 INFO epoch   9/50  train_rmse=0.4580 (norm=0.4408)  val_rmse=0.4568 (norm=0.4396)
2026-08-18 00:04:38,140 INFO     opponent=immobile   train_rmse=0.4580 (n=75856)  val_rmse=0.4568 (n=18892)
2026-08-18 00:04:38,140 INFO     outcome=ball_out      train_rmse=3.1498 (n=153)  val_rmse=2.8818 (n=31)
2026-08-18 00:04:38,140 INFO     outcome=invalid       train_rmse=1.2274 (n=1593)  val_rmse=1.2790 (n=436)
2026-08-18 00:04:38,140 INFO     outcome=win           train_rmse=0.4023 (n=74110)  val_rmse=0.4016 (n=18425)
2026-08-18 00:04:54,388 INFO epoch  10/50  train_rmse=0.4525 (norm=0.4355)  val_rmse=0.4646 (norm=0.4472)
2026-08-18 00:04:54,388 INFO     opponent=immobile   train_rmse=0.4525 (n=75856)  val_rmse=0.4646 (n=18892)
2026-08-18 00:04:54,389 INFO     outcome=ball_out      train_rmse=2.9815 (n=153)  val_rmse=2.7035 (n=31)
2026-08-18 00:04:54,389 INFO     outcome=invalid       train_rmse=1.2510 (n=1593)  val_rmse=1.3923 (n=436)
2026-08-18 00:04:54,389 INFO     outcome=win           train_rmse=0.3970 (n=74110)  val_rmse=0.4039 (n=18425)
2026-08-18 00:05:10,975 INFO epoch  11/50  train_rmse=0.4395 (norm=0.4229)  val_rmse=0.4723 (norm=0.4545)
2026-08-18 00:05:10,975 INFO     opponent=immobile   train_rmse=0.4395 (n=75856)  val_rmse=0.4723 (n=18892)
2026-08-18 00:05:10,975 INFO     outcome=ball_out      train_rmse=2.8944 (n=153)  val_rmse=4.2180 (n=31)
2026-08-18 00:05:10,975 INFO     outcome=invalid       train_rmse=1.1568 (n=1593)  val_rmse=1.8671 (n=436)
2026-08-18 00:05:10,975 INFO     outcome=win           train_rmse=0.3894 (n=74110)  val_rmse=0.3410 (n=18425)
2026-08-18 00:05:27,221 INFO epoch  12/50  train_rmse=0.4326 (norm=0.4163)  val_rmse=0.4630 (norm=0.4456)
2026-08-18 00:05:27,221 INFO     opponent=immobile   train_rmse=0.4326 (n=75856)  val_rmse=0.4630 (n=18892)
2026-08-18 00:05:27,221 INFO     outcome=ball_out      train_rmse=2.9215 (n=153)  val_rmse=2.2350 (n=31)
2026-08-18 00:05:27,221 INFO     outcome=invalid       train_rmse=1.1205 (n=1593)  val_rmse=1.2500 (n=436)
2026-08-18 00:05:27,221 INFO     outcome=win           train_rmse=0.3833 (n=74110)  val_rmse=0.4176 (n=18425)
2026-08-18 00:05:43,454 INFO epoch  13/50  train_rmse=0.4417 (norm=0.4251)  val_rmse=0.4563 (norm=0.4392)
2026-08-18 00:05:43,455 INFO     opponent=immobile   train_rmse=0.4417 (n=75856)  val_rmse=0.4563 (n=18892)
2026-08-18 00:05:43,455 INFO     outcome=ball_out      train_rmse=3.3459 (n=153)  val_rmse=3.0839 (n=31)
2026-08-18 00:05:43,455 INFO     outcome=invalid       train_rmse=1.1428 (n=1593)  val_rmse=1.9237 (n=436)
2026-08-18 00:05:43,455 INFO     outcome=win           train_rmse=0.3853 (n=74110)  val_rmse=0.3315 (n=18425)
2026-08-18 00:05:59,399 INFO epoch  14/50  train_rmse=0.4320 (norm=0.4157)  val_rmse=0.4703 (norm=0.4526)
2026-08-18 00:05:59,399 INFO     opponent=immobile   train_rmse=0.4320 (n=75856)  val_rmse=0.4703 (n=18892)
2026-08-18 00:05:59,399 INFO     outcome=ball_out      train_rmse=2.7821 (n=153)  val_rmse=2.9471 (n=31)
2026-08-18 00:05:59,399 INFO     outcome=invalid       train_rmse=1.1327 (n=1593)  val_rmse=1.5662 (n=436)
2026-08-18 00:05:59,399 INFO     outcome=win           train_rmse=0.3840 (n=74110)  val_rmse=0.3925 (n=18425)
2026-08-18 00:06:15,578 INFO epoch  15/50  train_rmse=0.4316 (norm=0.4154)  val_rmse=0.4672 (norm=0.4497)
2026-08-18 00:06:15,578 INFO     opponent=immobile   train_rmse=0.4316 (n=75856)  val_rmse=0.4672 (n=18892)
2026-08-18 00:06:15,578 INFO     outcome=ball_out      train_rmse=2.7095 (n=153)  val_rmse=2.3376 (n=31)
2026-08-18 00:06:15,578 INFO     outcome=invalid       train_rmse=1.1088 (n=1593)  val_rmse=1.7922 (n=436)
2026-08-18 00:06:15,578 INFO     outcome=win           train_rmse=0.3861 (n=74110)  val_rmse=0.3723 (n=18425)
2026-08-18 00:06:31,700 INFO epoch  16/50  train_rmse=0.4239 (norm=0.4080)  val_rmse=0.4482 (norm=0.4314)
2026-08-18 00:06:31,700 INFO     opponent=immobile   train_rmse=0.4239 (n=75856)  val_rmse=0.4482 (n=18892)
2026-08-18 00:06:31,700 INFO     outcome=ball_out      train_rmse=2.7912 (n=153)  val_rmse=2.5283 (n=31)
2026-08-18 00:06:31,700 INFO     outcome=invalid       train_rmse=1.1296 (n=1593)  val_rmse=1.6200 (n=436)
2026-08-18 00:06:31,700 INFO     outcome=win           train_rmse=0.3748 (n=74110)  val_rmse=0.3649 (n=18425)
2026-08-18 00:06:47,655 INFO epoch  17/50  train_rmse=0.4187 (norm=0.4030)  val_rmse=0.4924 (norm=0.4739)
2026-08-18 00:06:47,655 INFO     opponent=immobile   train_rmse=0.4187 (n=75856)  val_rmse=0.4924 (n=18892)
2026-08-18 00:06:47,655 INFO     outcome=ball_out      train_rmse=2.6010 (n=153)  val_rmse=5.2953 (n=31)
2026-08-18 00:06:47,655 INFO     outcome=invalid       train_rmse=1.1114 (n=1593)  val_rmse=1.7451 (n=436)
2026-08-18 00:06:47,656 INFO     outcome=win           train_rmse=0.3728 (n=74110)  val_rmse=0.3596 (n=18425)
2026-08-18 00:07:03,631 INFO epoch  18/50  train_rmse=0.4374 (norm=0.4210)  val_rmse=0.4389 (norm=0.4224)
2026-08-18 00:07:03,631 INFO     opponent=immobile   train_rmse=0.4374 (n=75856)  val_rmse=0.4389 (n=18892)
2026-08-18 00:07:03,631 INFO     outcome=ball_out      train_rmse=3.1627 (n=153)  val_rmse=3.0918 (n=31)
2026-08-18 00:07:03,631 INFO     outcome=invalid       train_rmse=1.2543 (n=1593)  val_rmse=1.3981 (n=436)
2026-08-18 00:07:03,631 INFO     outcome=win           train_rmse=0.3760 (n=74110)  val_rmse=0.3677 (n=18425)
2026-08-18 00:07:19,532 INFO epoch  19/50  train_rmse=0.4154 (norm=0.3998)  val_rmse=0.4288 (norm=0.4127)
2026-08-18 00:07:19,532 INFO     opponent=immobile   train_rmse=0.4154 (n=75856)  val_rmse=0.4288 (n=18892)
2026-08-18 00:07:19,532 INFO     outcome=ball_out      train_rmse=2.7094 (n=153)  val_rmse=3.1229 (n=31)
2026-08-18 00:07:19,532 INFO     outcome=invalid       train_rmse=1.0925 (n=1593)  val_rmse=1.4777 (n=436)
2026-08-18 00:07:19,532 INFO     outcome=win           train_rmse=0.3685 (n=74110)  val_rmse=0.3470 (n=18425)
2026-08-18 00:07:35,845 INFO epoch  20/50  train_rmse=0.4181 (norm=0.4024)  val_rmse=0.4614 (norm=0.4441)
2026-08-18 00:07:35,845 INFO     opponent=immobile   train_rmse=0.4181 (n=75856)  val_rmse=0.4614 (n=18892)
2026-08-18 00:07:35,845 INFO     outcome=ball_out      train_rmse=2.8260 (n=153)  val_rmse=3.2111 (n=31)
2026-08-18 00:07:35,845 INFO     outcome=invalid       train_rmse=1.0718 (n=1593)  val_rmse=1.6036 (n=436)
2026-08-18 00:07:35,845 INFO     outcome=win           train_rmse=0.3711 (n=74110)  val_rmse=0.3743 (n=18425)
2026-08-18 00:07:51,913 INFO epoch  21/50  train_rmse=0.4156 (norm=0.4000)  val_rmse=0.4244 (norm=0.4085)
2026-08-18 00:07:51,913 INFO     opponent=immobile   train_rmse=0.4156 (n=75856)  val_rmse=0.4244 (n=18892)
2026-08-18 00:07:51,913 INFO     outcome=ball_out      train_rmse=2.6918 (n=153)  val_rmse=2.7054 (n=31)
2026-08-18 00:07:51,913 INFO     outcome=invalid       train_rmse=1.0785 (n=1593)  val_rmse=1.3791 (n=436)
2026-08-18 00:07:51,913 INFO     outcome=win           train_rmse=0.3699 (n=74110)  val_rmse=0.3569 (n=18425)
2026-08-18 00:08:08,069 INFO epoch  22/50  train_rmse=0.4099 (norm=0.3945)  val_rmse=0.4429 (norm=0.4262)
2026-08-18 00:08:08,069 INFO     opponent=immobile   train_rmse=0.4099 (n=75856)  val_rmse=0.4429 (n=18892)
2026-08-18 00:08:08,069 INFO     outcome=ball_out      train_rmse=2.7182 (n=153)  val_rmse=2.7636 (n=31)
2026-08-18 00:08:08,069 INFO     outcome=invalid       train_rmse=1.0713 (n=1593)  val_rmse=1.8139 (n=436)
2026-08-18 00:08:08,069 INFO     outcome=win           train_rmse=0.3634 (n=74110)  val_rmse=0.3323 (n=18425)
2026-08-18 00:08:24,303 INFO epoch  23/50  train_rmse=0.4144 (norm=0.3988)  val_rmse=0.4205 (norm=0.4047)
2026-08-18 00:08:24,303 INFO     opponent=immobile   train_rmse=0.4144 (n=75856)  val_rmse=0.4205 (n=18892)
2026-08-18 00:08:24,303 INFO     outcome=ball_out      train_rmse=2.6293 (n=153)  val_rmse=3.0202 (n=31)
2026-08-18 00:08:24,303 INFO     outcome=invalid       train_rmse=1.1730 (n=1593)  val_rmse=1.1098 (n=436)
2026-08-18 00:08:24,303 INFO     outcome=win           train_rmse=0.3632 (n=74110)  val_rmse=0.3699 (n=18425)
2026-08-18 00:08:40,372 INFO epoch  24/50  train_rmse=0.4183 (norm=0.4026)  val_rmse=0.4411 (norm=0.4245)
2026-08-18 00:08:40,372 INFO     opponent=immobile   train_rmse=0.4183 (n=75856)  val_rmse=0.4411 (n=18892)
2026-08-18 00:08:40,372 INFO     outcome=ball_out      train_rmse=2.9358 (n=153)  val_rmse=3.2598 (n=31)
2026-08-18 00:08:40,372 INFO     outcome=invalid       train_rmse=1.0679 (n=1593)  val_rmse=1.6330 (n=436)
2026-08-18 00:08:40,373 INFO     outcome=win           train_rmse=0.3698 (n=74110)  val_rmse=0.3442 (n=18425)
2026-08-18 00:08:56,525 INFO epoch  25/50  train_rmse=0.4098 (norm=0.3944)  val_rmse=0.4331 (norm=0.4168)
2026-08-18 00:08:56,526 INFO     opponent=immobile   train_rmse=0.4098 (n=75856)  val_rmse=0.4331 (n=18892)
2026-08-18 00:08:56,526 INFO     outcome=ball_out      train_rmse=2.7399 (n=153)  val_rmse=2.7451 (n=31)
2026-08-18 00:08:56,526 INFO     outcome=invalid       train_rmse=1.0872 (n=1593)  val_rmse=1.5750 (n=436)
2026-08-18 00:08:56,526 INFO     outcome=win           train_rmse=0.3619 (n=74110)  val_rmse=0.3478 (n=18425)
2026-08-18 00:09:12,797 INFO epoch  26/50  train_rmse=0.4045 (norm=0.3893)  val_rmse=0.4306 (norm=0.4144)
2026-08-18 00:09:12,797 INFO     opponent=immobile   train_rmse=0.4045 (n=75856)  val_rmse=0.4306 (n=18892)
2026-08-18 00:09:12,797 INFO     outcome=ball_out      train_rmse=2.5842 (n=153)  val_rmse=3.3445 (n=31)
2026-08-18 00:09:12,797 INFO     outcome=invalid       train_rmse=1.0566 (n=1593)  val_rmse=1.4611 (n=436)
2026-08-18 00:09:12,797 INFO     outcome=win           train_rmse=0.3601 (n=74110)  val_rmse=0.3475 (n=18425)
2026-08-18 00:09:28,829 INFO epoch  27/50  train_rmse=0.4005 (norm=0.3854)  val_rmse=0.4805 (norm=0.4624)
2026-08-18 00:09:28,829 INFO     opponent=immobile   train_rmse=0.4005 (n=75856)  val_rmse=0.4805 (n=18892)
2026-08-18 00:09:28,829 INFO     outcome=ball_out      train_rmse=2.6750 (n=153)  val_rmse=2.9223 (n=31)
2026-08-18 00:09:28,829 INFO     outcome=invalid       train_rmse=1.0855 (n=1593)  val_rmse=1.4522 (n=436)
2026-08-18 00:09:28,829 INFO     outcome=win           train_rmse=0.3522 (n=74110)  val_rmse=0.4152 (n=18425)
2026-08-18 00:09:44,768 INFO epoch  28/50  train_rmse=0.3960 (norm=0.3811)  val_rmse=0.4346 (norm=0.4183)
2026-08-18 00:09:44,768 INFO     opponent=immobile   train_rmse=0.3960 (n=75856)  val_rmse=0.4346 (n=18892)
2026-08-18 00:09:44,768 INFO     outcome=ball_out      train_rmse=2.5116 (n=153)  val_rmse=2.6003 (n=31)
2026-08-18 00:09:44,768 INFO     outcome=invalid       train_rmse=1.0439 (n=1593)  val_rmse=1.3831 (n=436)
2026-08-18 00:09:44,768 INFO     outcome=win           train_rmse=0.3522 (n=74110)  val_rmse=0.3702 (n=18425)
2026-08-18 00:09:44,768 INFO Early stopping at epoch 28/50 (val normalized MSE did not improve for 5 epochs).
2026-08-18 00:09:44,768 INFO Best val normalized MSE achieved: 0.1638 (RMSE=0.4047; <1.0 = better than predicting the mean; <0.5 = useful critic)
2026-08-18 00:09:46,922 INFO --- Per-component MC-return magnitude (val rows) ---
2026-08-18 00:09:46,923 INFO   get_possession    mean=+0.3736  std=0.4636
2026-08-18 00:09:46,923 INFO   lose_possession   mean=-0.0070  std=0.0749
2026-08-18 00:09:46,923 INFO   ball_out          mean=-0.0058  std=0.1443
2026-08-18 00:09:46,923 INFO   box_possession    mean=+1.6062  std=0.3392
2026-08-18 00:09:46,923 INFO   speed_bonus       mean=+2.1012  std=0.6561
2026-08-18 00:09:46,923 INFO   stamina_penalty   mean=-0.0637  std=0.0250
2026-08-18 00:09:47,255 INFO --- Reward-component vs. value-residual correlation (432 val episodes) ---
2026-08-18 00:09:47,255 INFO   component            corr   comp_std
2026-08-18 00:09:47,256 INFO   lose_possession    +0.095     0.0856
2026-08-18 00:09:47,256 INFO   ball_out           +0.237     0.1511
2026-08-18 00:09:47,256 INFO   speed_bonus        +0.251     0.7532
2026-08-18 00:09:47,256 INFO   get_possession     +0.296     0.1978
2026-08-18 00:09:47,256 INFO   stamina_penalty    -0.308     0.0210
2026-08-18 00:09:47,256 INFO   box_possession     +0.340     0.3442
2026-08-18 00:09:47,256 INFO   (components near the top -- low |corr| despite real variance -- are the ones the value net's errors track least; read alongside the per-component MC-return magnitude above.)
2026-08-18 00:09:47,260 INFO --- Worst val episode for outcome=ball_out (1 episode(s)): rows [90024, 90054], residual=-2.930 -- saved match log to results/debug_value_worst_episode_ball_out.json ---
2026-08-18 00:09:47,261 INFO --- Worst val episode for outcome=invalid (16 episode(s)): rows [87417, 87441], residual=-3.060 -- saved match log to results/debug_value_worst_episode_invalid.json ---
2026-08-18 00:09:47,261 INFO --- Worst val episode for outcome=win (415 episode(s)): rows [78345, 78386], residual=+3.778 -- saved match log to results/debug_value_worst_episode_win.json ---
2026-08-18 01:42:20,270 INFO Checkpoint checkpoints/longterm/checkpoint_vvgood_immobile.pt: separate_value_net=True (auto-detected)
2026-08-18 01:42:20,300 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 01:42:20,301 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 01:42:20,301 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 01:42:20,303 INFO --reset-dir-log-std: reset move_dir_log_std/kick_dir_log_std to config init values before rollout collection.
2026-08-18 01:42:20,303 INFO Collecting rollout: 8 parallel worker(s), ~14375 steps/worker, worker_torch_threads=1
Rollout (8 workers): 0/115000 (  0.0%)     0.0 steps/s
2026-08-18 01:42:23,286 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 01:42:23,287 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 01:42:23,287 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 01:42:23,297 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 01:42:23,298 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 01:42:23,299 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 01:42:23,318 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 01:42:23,320 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 01:42:23,321 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 01:42:23,327 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 01:42:23,330 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 01:42:23,331 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 01:42:23,333 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 01:42:23,335 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 01:42:23,336 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 01:42:23,340 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 01:42:23,343 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 01:42:23,343 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 01:42:23,428 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 01:42:23,430 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 01:42:23,430 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 01:42:23,487 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 01:42:23,489 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 01:42:23,490 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
Rollout (8 workers): 11555/115000 ( 10.0%)   711.1 steps/s
Rollout (8 workers): 23150/115000 ( 20.1%)   759.5 steps/s
Rollout (8 workers): 34568/115000 ( 30.1%)   769.6 steps/s
Rollout (8 workers): 46010/115000 ( 40.0%)   770.1 steps/s
Rollout (8 workers): 57548/115000 ( 50.0%)   755.4 steps/s
Rollout (8 workers): 69033/115000 ( 60.0%)   753.4 steps/s
Rollout (8 workers): 80565/115000 ( 70.1%)   751.1 steps/s
Rollout (8 workers): 92087/115000 ( 80.1%)   749.3 steps/s
Rollout (8 workers): 103578/115000 ( 90.1%)   749.8 steps/s
[worker 0] done: 136.9s total  (9.53 ms/step, 0.42 s/episode over 325 episode(s))
[worker 4] done: 136.9s total  (9.53 ms/step, 0.42 s/episode over 325 episode(s))
[worker 6] done: 137.4s total  (9.56 ms/step, 0.43 s/episode over 323 episode(s))
[worker 5] done: 138.5s total  (9.64 ms/step, 0.42 s/episode over 328 episode(s))
[worker 7] done: 139.2s total  (9.69 ms/step, 0.42 s/episode over 331 episode(s))
[worker 1] done: 140.4s total  (9.76 ms/step, 0.43 s/episode over 328 episode(s))
[worker 2] done: 140.9s total  (9.80 ms/step, 0.43 s/episode over 328 episode(s))
[worker 3] done: 141.3s total  (9.83 ms/step, 0.43 s/episode over 325 episode(s))
2026-08-18 01:44:46,122 INFO   [parallel rollout] total: 145.8s wall  (1.27 ms/step aggregate, 787.5 steps/s aggregate across 8 worker(s))
2026-08-18 01:44:46,123 INFO Dropped 161 trailing (incomplete-episode) step(s) across workers
2026-08-18 01:44:46,266 INFO Rollout dataset: 114,839 steps, 2613 complete episode(s)
2026-08-18 01:44:46,780 INFO Loaded 114,839 rows total
2026-08-18 01:44:46,781 INFO has_rewards=True
2026-08-18 01:44:46,783 INFO valid_indices(): 114,839 rows (100.0% of total)
2026-08-18 01:44:46,816 INFO Returns over ALL rows: mean=3.982 std=1.046 min=-4.000 max=6.818
2026-08-18 01:44:46,817 INFO Returns over valid_indices(): mean=3.982 std=1.046
2026-08-18 01:44:46,845 INFO --- Dataset distribution (114,839 rows, 2613 episodes) ---
2026-08-18 01:44:46,847 INFO   self.ai_type == rules: 0.0%
2026-08-18 01:44:46,847 INFO   self.ai_type == immobile: 0.0%
2026-08-18 01:44:46,848 INFO   self.ai_type == neural: 100.0%
2026-08-18 01:44:46,849 INFO   opponent.ai_type == rules: 0.0%
2026-08-18 01:44:46,849 INFO   opponent.ai_type == immobile: 100.0%
2026-08-18 01:44:46,850 INFO   opponent.ai_type == neural: 0.0%
2026-08-18 01:44:46,854 INFO   valid rows: kick_this_tick rate=0.00%  tackle_attempt rate=0.00%
2026-08-18 01:44:46,854 INFO   dones=1 rows: 2,613  |  zero-reward rows: 109,910 (95.7%)
2026-08-18 01:44:46,863 INFO   return percentiles (all rows): p10=3.02  p50=3.97  p90=5.17
2026-08-18 01:44:46,921 INFO --- Reward component breakdown (all episodes, 2613 episode(s)) ---
2026-08-18 01:44:46,922 INFO   component           mean      std       min       max
2026-08-18 01:44:46,922 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,922 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,923 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,923 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,923 INFO   get_possession    +0.976    0.218    +0.000    +2.000
2026-08-18 01:44:46,923 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,924 INFO   lose_possession    -0.011    0.097    -0.900    +0.000
2026-08-18 01:44:46,924 INFO   ball_out          -0.014    0.234    -4.000    +0.000
2026-08-18 01:44:46,924 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,924 INFO   box_possession    +1.920    0.391    +0.000    +2.000
2026-08-18 01:44:46,924 INFO   speed_bonus       +2.672    0.745    +0.000    +3.890
2026-08-18 01:44:46,925 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,925 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,925 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,925 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,926 INFO   stamina_penalty    -0.075    0.031    -0.141    +0.000
2026-08-18 01:44:46,949 INFO --- Reward component breakdown (outcome=ball_out, 9 episode(s)) ---
2026-08-18 01:44:46,949 INFO   component           mean      std       min       max
2026-08-18 01:44:46,949 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,949 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,949 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,949 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,949 INFO   get_possession    +1.000    0.000    +1.000    +1.000
2026-08-18 01:44:46,949 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,949 INFO   lose_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,949 INFO   ball_out          -4.000    0.000    -4.000    -4.000
2026-08-18 01:44:46,949 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,949 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,949 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,949 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,949 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,949 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,949 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,950 INFO   stamina_penalty    +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,974 INFO --- Reward component breakdown (outcome=invalid, 95 episode(s)) ---
2026-08-18 01:44:46,974 INFO   component           mean      std       min       max
2026-08-18 01:44:46,974 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,974 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,974 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,974 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,974 INFO   get_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,974 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,974 INFO   lose_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,974 INFO   ball_out          +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,974 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,974 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,974 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,974 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,974 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,974 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,975 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,975 INFO   stamina_penalty    +0.000    0.000    +0.000    +0.000
2026-08-18 01:44:46,997 INFO --- Reward component breakdown (outcome=loss, 0 episode(s)) ---
2026-08-18 01:44:47,019 INFO --- Reward component breakdown (outcome=timeout, 0 episode(s)) ---
2026-08-18 01:44:47,072 INFO --- MC returns by outcome (all rows, 114,839 rows) ---
2026-08-18 01:44:47,074 INFO   ball_out     n=    268  mean=-2.661  std=0.357  min=-4.000  max=-1.838
2026-08-18 01:44:47,076 INFO   invalid      n=  2,409  mean=+0.000  std=0.000  min=+0.000  max=+0.000
2026-08-18 01:44:47,078 INFO   win          n=112,162  mean=+4.084  std=0.815  min=+1.796  max=+6.818
2026-08-18 01:44:47,146 INFO --- Episode total reward by outcome (all rows, 2613 episode(s)) ---
2026-08-18 01:44:47,147 INFO   ball_out     n=     9  mean=-3.000  std=0.000  min=-3.000  max=-3.000
2026-08-18 01:44:47,147 INFO   invalid      n=    95  mean=+0.000  std=0.000  min=+0.000  max=+0.000
2026-08-18 01:44:47,147 INFO   win          n= 2,509  mean=+5.706  std=0.526  min=+3.909  max=+6.818
2026-08-18 01:44:47,207 INFO Train/val split (valid_only=True): 91,800 train rows across 2090 episodes  |  23,039 val rows across 523 episodes
2026-08-18 01:44:47,256 INFO --- Linear regression baseline (top_speed, ball_dist, ball_to_box, time_remaining) ---
2026-08-18 01:44:47,272 INFO   [all outcomes] n_train=91,800  n_val=23,039  coef(top_speed, ball_dist, ball_to_box, time_rem, intercept)=[0.339, -1.561, -2.137, 6.65, -1.025]
    train_rmse=0.7772 (norm=0.7420)  val_rmse=0.7738 (norm=0.7386)
2026-08-18 01:44:47,340 INFO   [win outcomes only] n_train=89,698  n_val=22,464  coef(top_speed, ball_dist, ball_to_box, time_rem, intercept)=[0.331, -1.261, -2.182, 7.432, -1.618]
    train_rmse=0.3733 (norm=0.4575)  val_rmse=0.3547 (norm=0.4347)
2026-08-18 01:44:48,386 INFO value_net: fresh separate value_net (--reset-separate-value-net)  total_params=185,440  trainable_params=185,440
2026-08-18 01:44:48,387 INFO Fitting fresh separate value network: 50 epochs, lr=0.0002, weight_decay=1e-06, batch_size=1096, train_ret_std=1.048, outcome_reweight=False
2026-08-18 01:45:01,452 INFO epoch   0/50 (baseline, no training yet)  train_rmse=4.0562 (norm=3.8720)  val_rmse=4.0407 (norm=3.8572)
2026-08-18 01:45:01,452 INFO     opponent=immobile   train_rmse=4.0562 (n=91800)  val_rmse=4.0407 (n=23039)
2026-08-18 01:45:01,452 INFO     outcome=ball_out      train_rmse=2.7488 (n=237)  val_rmse=2.7333 (n=31)
2026-08-18 01:45:01,452 INFO     outcome=invalid       train_rmse=0.0577 (n=1865)  val_rmse=0.0606 (n=544)
2026-08-18 01:45:01,452 INFO     outcome=win           train_rmse=4.1010 (n=89698)  val_rmse=4.0908 (n=22464)
2026-08-18 01:45:23,909 INFO epoch   1/50  train_rmse=2.2259 (norm=2.1248)  val_rmse=0.8066 (norm=0.7700)
2026-08-18 01:45:23,909 INFO     opponent=immobile   train_rmse=2.2259 (n=91800)  val_rmse=0.8066 (n=23039)
2026-08-18 01:45:23,909 INFO     outcome=ball_out      train_rmse=5.6451 (n=237)  val_rmse=5.7405 (n=31)
2026-08-18 01:45:23,909 INFO     outcome=invalid       train_rmse=3.2814 (n=1865)  val_rmse=3.5043 (n=544)
2026-08-18 01:45:23,910 INFO     outcome=win           train_rmse=2.1823 (n=89698)  val_rmse=0.5695 (n=22464)
2026-08-18 01:45:46,357 INFO epoch   2/50  train_rmse=0.7289 (norm=0.6958)  val_rmse=0.6585 (norm=0.6286)
2026-08-18 01:45:46,357 INFO     opponent=immobile   train_rmse=0.7289 (n=91800)  val_rmse=0.6585 (n=23039)
2026-08-18 01:45:46,357 INFO     outcome=ball_out      train_rmse=5.5718 (n=237)  val_rmse=5.5388 (n=31)
2026-08-18 01:45:46,357 INFO     outcome=invalid       train_rmse=3.2385 (n=1865)  val_rmse=2.8905 (n=544)
2026-08-18 01:45:46,357 INFO     outcome=win           train_rmse=0.4936 (n=89698)  val_rmse=0.4473 (n=22464)
2026-08-18 01:46:09,736 INFO epoch   3/50  train_rmse=0.6562 (norm=0.6264)  val_rmse=0.6076 (norm=0.5800)
2026-08-18 01:46:09,736 INFO     opponent=immobile   train_rmse=0.6562 (n=91800)  val_rmse=0.6076 (n=23039)
2026-08-18 01:46:09,736 INFO     outcome=ball_out      train_rmse=5.1830 (n=237)  val_rmse=5.2578 (n=31)
2026-08-18 01:46:09,736 INFO     outcome=invalid       train_rmse=2.7358 (n=1865)  val_rmse=2.5740 (n=544)
2026-08-18 01:46:09,736 INFO     outcome=win           train_rmse=0.4627 (n=89698)  val_rmse=0.4242 (n=22464)
2026-08-18 01:46:32,720 INFO epoch   4/50  train_rmse=0.6043 (norm=0.5768)  val_rmse=0.5655 (norm=0.5398)
2026-08-18 01:46:32,720 INFO     opponent=immobile   train_rmse=0.6043 (n=91800)  val_rmse=0.5655 (n=23039)
2026-08-18 01:46:32,720 INFO     outcome=ball_out      train_rmse=4.9087 (n=237)  val_rmse=5.1779 (n=31)
2026-08-18 01:46:32,720 INFO     outcome=invalid       train_rmse=2.3323 (n=1865)  val_rmse=2.2507 (n=544)
2026-08-18 01:46:32,720 INFO     outcome=win           train_rmse=0.4437 (n=89698)  val_rmse=0.4103 (n=22464)
2026-08-18 01:46:55,054 INFO epoch   5/50  train_rmse=0.5674 (norm=0.5416)  val_rmse=0.5410 (norm=0.5164)
2026-08-18 01:46:55,054 INFO     opponent=immobile   train_rmse=0.5674 (n=91800)  val_rmse=0.5410 (n=23039)
2026-08-18 01:46:55,054 INFO     outcome=ball_out      train_rmse=4.7221 (n=237)  val_rmse=4.8921 (n=31)
2026-08-18 01:46:55,054 INFO     outcome=invalid       train_rmse=2.0300 (n=1865)  val_rmse=1.9215 (n=544)
2026-08-18 01:46:55,054 INFO     outcome=win           train_rmse=0.4299 (n=89698)  val_rmse=0.4215 (n=22464)
2026-08-18 01:47:16,151 INFO epoch   6/50  train_rmse=0.5402 (norm=0.5157)  val_rmse=0.5356 (norm=0.5113)
2026-08-18 01:47:16,152 INFO     opponent=immobile   train_rmse=0.5402 (n=91800)  val_rmse=0.5356 (n=23039)
2026-08-18 01:47:16,152 INFO     outcome=ball_out      train_rmse=4.4783 (n=237)  val_rmse=4.5422 (n=31)
2026-08-18 01:47:16,152 INFO     outcome=invalid       train_rmse=1.7805 (n=1865)  val_rmse=1.7956 (n=544)
2026-08-18 01:47:16,152 INFO     outcome=win           train_rmse=0.4240 (n=89698)  val_rmse=0.4332 (n=22464)
2026-08-18 01:47:37,527 INFO epoch   7/50  train_rmse=0.5210 (norm=0.4973)  val_rmse=0.5160 (norm=0.4926)
2026-08-18 01:47:37,528 INFO     opponent=immobile   train_rmse=0.5210 (n=91800)  val_rmse=0.5160 (n=23039)
2026-08-18 01:47:37,528 INFO     outcome=ball_out      train_rmse=4.3333 (n=237)  val_rmse=4.3185 (n=31)
2026-08-18 01:47:37,528 INFO     outcome=invalid       train_rmse=1.6266 (n=1865)  val_rmse=1.7498 (n=544)
2026-08-18 01:47:37,528 INFO     outcome=win           train_rmse=0.4161 (n=89698)  val_rmse=0.4161 (n=22464)
2026-08-18 01:47:58,673 INFO epoch   8/50  train_rmse=0.5029 (norm=0.4800)  val_rmse=0.5181 (norm=0.4945)
2026-08-18 01:47:58,673 INFO     opponent=immobile   train_rmse=0.5029 (n=91800)  val_rmse=0.5181 (n=23039)
2026-08-18 01:47:58,673 INFO     outcome=ball_out      train_rmse=3.9816 (n=237)  val_rmse=4.4236 (n=31)
2026-08-18 01:47:58,673 INFO     outcome=invalid       train_rmse=1.5425 (n=1865)  val_rmse=1.7559 (n=544)
2026-08-18 01:47:58,673 INFO     outcome=win           train_rmse=0.4092 (n=89698)  val_rmse=0.4166 (n=22464)
2026-08-18 01:48:19,510 INFO epoch   9/50  train_rmse=0.4886 (norm=0.4664)  val_rmse=0.5014 (norm=0.4786)
2026-08-18 01:48:19,510 INFO     opponent=immobile   train_rmse=0.4886 (n=91800)  val_rmse=0.5014 (n=23039)
2026-08-18 01:48:19,510 INFO     outcome=ball_out      train_rmse=3.8733 (n=237)  val_rmse=3.8864 (n=31)
2026-08-18 01:48:19,510 INFO     outcome=invalid       train_rmse=1.5183 (n=1865)  val_rmse=1.7374 (n=544)
2026-08-18 01:48:19,510 INFO     outcome=win           train_rmse=0.3960 (n=89698)  val_rmse=0.4049 (n=22464)
2026-08-18 01:48:40,413 INFO epoch  10/50  train_rmse=0.4779 (norm=0.4562)  val_rmse=0.5182 (norm=0.4947)
2026-08-18 01:48:40,413 INFO     opponent=immobile   train_rmse=0.4779 (n=91800)  val_rmse=0.5182 (n=23039)
2026-08-18 01:48:40,413 INFO     outcome=ball_out      train_rmse=3.4501 (n=237)  val_rmse=4.2548 (n=31)
2026-08-18 01:48:40,413 INFO     outcome=invalid       train_rmse=1.4450 (n=1865)  val_rmse=2.2021 (n=544)
2026-08-18 01:48:40,413 INFO     outcome=win           train_rmse=0.3986 (n=89698)  val_rmse=0.3648 (n=22464)
2026-08-18 01:49:01,270 INFO epoch  11/50  train_rmse=0.4729 (norm=0.4514)  val_rmse=0.4790 (norm=0.4573)
2026-08-18 01:49:01,270 INFO     opponent=immobile   train_rmse=0.4729 (n=91800)  val_rmse=0.4790 (n=23039)
2026-08-18 01:49:01,270 INFO     outcome=ball_out      train_rmse=3.3548 (n=237)  val_rmse=3.2195 (n=31)
2026-08-18 01:49:01,270 INFO     outcome=invalid       train_rmse=1.4493 (n=1865)  val_rmse=1.6327 (n=544)
2026-08-18 01:49:01,270 INFO     outcome=win           train_rmse=0.3943 (n=89698)  val_rmse=0.3956 (n=22464)
2026-08-18 01:49:22,571 INFO epoch  12/50  train_rmse=0.4631 (norm=0.4421)  val_rmse=0.4809 (norm=0.4591)
2026-08-18 01:49:22,571 INFO     opponent=immobile   train_rmse=0.4631 (n=91800)  val_rmse=0.4809 (n=23039)
2026-08-18 01:49:22,571 INFO     outcome=ball_out      train_rmse=3.1176 (n=237)  val_rmse=3.5785 (n=31)
2026-08-18 01:49:22,571 INFO     outcome=invalid       train_rmse=1.3898 (n=1865)  val_rmse=1.7800 (n=544)
2026-08-18 01:49:22,571 INFO     outcome=win           train_rmse=0.3920 (n=89698)  val_rmse=0.3779 (n=22464)
2026-08-18 01:49:43,510 INFO epoch  13/50  train_rmse=0.4529 (norm=0.4323)  val_rmse=0.4674 (norm=0.4462)
2026-08-18 01:49:43,510 INFO     opponent=immobile   train_rmse=0.4529 (n=91800)  val_rmse=0.4674 (n=23039)
2026-08-18 01:49:43,510 INFO     outcome=ball_out      train_rmse=3.0755 (n=237)  val_rmse=2.9972 (n=31)
2026-08-18 01:49:43,510 INFO     outcome=invalid       train_rmse=1.3795 (n=1865)  val_rmse=1.6137 (n=544)
2026-08-18 01:49:43,510 INFO     outcome=win           train_rmse=0.3813 (n=89698)  val_rmse=0.3855 (n=22464)
2026-08-18 01:50:04,557 INFO epoch  14/50  train_rmse=0.4473 (norm=0.4270)  val_rmse=0.4881 (norm=0.4660)
2026-08-18 01:50:04,557 INFO     opponent=immobile   train_rmse=0.4473 (n=91800)  val_rmse=0.4881 (n=23039)
2026-08-18 01:50:04,557 INFO     outcome=ball_out      train_rmse=3.0961 (n=237)  val_rmse=3.3040 (n=31)
2026-08-18 01:50:04,557 INFO     outcome=invalid       train_rmse=1.3388 (n=1865)  val_rmse=2.0167 (n=544)
2026-08-18 01:50:04,557 INFO     outcome=win           train_rmse=0.3771 (n=89698)  val_rmse=0.3617 (n=22464)
2026-08-18 01:50:25,523 INFO epoch  15/50  train_rmse=0.4433 (norm=0.4232)  val_rmse=0.4682 (norm=0.4469)
2026-08-18 01:50:25,523 INFO     opponent=immobile   train_rmse=0.4433 (n=91800)  val_rmse=0.4682 (n=23039)
2026-08-18 01:50:25,523 INFO     outcome=ball_out      train_rmse=2.9834 (n=237)  val_rmse=3.8658 (n=31)
2026-08-18 01:50:25,524 INFO     outcome=invalid       train_rmse=1.3361 (n=1865)  val_rmse=1.5296 (n=544)
2026-08-18 01:50:25,524 INFO     outcome=win           train_rmse=0.3748 (n=89698)  val_rmse=0.3841 (n=22464)
2026-08-18 01:50:46,566 INFO epoch  16/50  train_rmse=0.4361 (norm=0.4163)  val_rmse=0.4544 (norm=0.4338)
2026-08-18 01:50:46,566 INFO     opponent=immobile   train_rmse=0.4361 (n=91800)  val_rmse=0.4544 (n=23039)
2026-08-18 01:50:46,566 INFO     outcome=ball_out      train_rmse=2.8533 (n=237)  val_rmse=3.1112 (n=31)
2026-08-18 01:50:46,566 INFO     outcome=invalid       train_rmse=1.3251 (n=1865)  val_rmse=1.4619 (n=544)
2026-08-18 01:50:46,566 INFO     outcome=win           train_rmse=0.3696 (n=89698)  val_rmse=0.3830 (n=22464)
2026-08-18 01:51:08,261 INFO epoch  17/50  train_rmse=0.4328 (norm=0.4132)  val_rmse=0.4535 (norm=0.4329)
2026-08-18 01:51:08,261 INFO     opponent=immobile   train_rmse=0.4328 (n=91800)  val_rmse=0.4535 (n=23039)
2026-08-18 01:51:08,261 INFO     outcome=ball_out      train_rmse=2.8600 (n=237)  val_rmse=2.1953 (n=31)
2026-08-18 01:51:08,261 INFO     outcome=invalid       train_rmse=1.2855 (n=1865)  val_rmse=1.5923 (n=544)
2026-08-18 01:51:08,261 INFO     outcome=win           train_rmse=0.3684 (n=89698)  val_rmse=0.3780 (n=22464)
2026-08-18 01:51:29,968 INFO epoch  18/50  train_rmse=0.4306 (norm=0.4110)  val_rmse=0.4652 (norm=0.4441)
2026-08-18 01:51:29,969 INFO     opponent=immobile   train_rmse=0.4306 (n=91800)  val_rmse=0.4652 (n=23039)
2026-08-18 01:51:29,969 INFO     outcome=ball_out      train_rmse=2.7611 (n=237)  val_rmse=4.2016 (n=31)
2026-08-18 01:51:29,969 INFO     outcome=invalid       train_rmse=1.3078 (n=1865)  val_rmse=1.7946 (n=544)
2026-08-18 01:51:29,969 INFO     outcome=win           train_rmse=0.3661 (n=89698)  val_rmse=0.3459 (n=22464)
2026-08-18 01:51:51,387 INFO epoch  19/50  train_rmse=0.4303 (norm=0.4108)  val_rmse=0.4876 (norm=0.4655)
2026-08-18 01:51:51,387 INFO     opponent=immobile   train_rmse=0.4303 (n=91800)  val_rmse=0.4876 (n=23039)
2026-08-18 01:51:51,387 INFO     outcome=ball_out      train_rmse=2.6804 (n=237)  val_rmse=3.1377 (n=31)
2026-08-18 01:51:51,387 INFO     outcome=invalid       train_rmse=1.3107 (n=1865)  val_rmse=1.4510 (n=544)
2026-08-18 01:51:51,387 INFO     outcome=win           train_rmse=0.3672 (n=89698)  val_rmse=0.4234 (n=22464)
2026-08-18 01:52:12,975 INFO epoch  20/50  train_rmse=0.4248 (norm=0.4055)  val_rmse=0.4647 (norm=0.4436)
2026-08-18 01:52:12,976 INFO     opponent=immobile   train_rmse=0.4248 (n=91800)  val_rmse=0.4647 (n=23039)
2026-08-18 01:52:12,976 INFO     outcome=ball_out      train_rmse=2.7245 (n=237)  val_rmse=2.4919 (n=31)
2026-08-18 01:52:12,976 INFO     outcome=invalid       train_rmse=1.2716 (n=1865)  val_rmse=1.7160 (n=544)
2026-08-18 01:52:12,976 INFO     outcome=win           train_rmse=0.3625 (n=89698)  val_rmse=0.3763 (n=22464)
2026-08-18 01:52:34,033 INFO epoch  21/50  train_rmse=0.4208 (norm=0.4017)  val_rmse=0.4572 (norm=0.4364)
2026-08-18 01:52:34,033 INFO     opponent=immobile   train_rmse=0.4208 (n=91800)  val_rmse=0.4572 (n=23039)
2026-08-18 01:52:34,033 INFO     outcome=ball_out      train_rmse=2.5452 (n=237)  val_rmse=3.5804 (n=31)
2026-08-18 01:52:34,033 INFO     outcome=invalid       train_rmse=1.2692 (n=1865)  val_rmse=1.4901 (n=544)
2026-08-18 01:52:34,033 INFO     outcome=win           train_rmse=0.3614 (n=89698)  val_rmse=0.3780 (n=22464)
2026-08-18 01:52:55,960 INFO epoch  22/50  train_rmse=0.4203 (norm=0.4012)  val_rmse=0.4742 (norm=0.4527)
2026-08-18 01:52:55,960 INFO     opponent=immobile   train_rmse=0.4203 (n=91800)  val_rmse=0.4742 (n=23039)
2026-08-18 01:52:55,960 INFO     outcome=ball_out      train_rmse=2.8226 (n=237)  val_rmse=4.4555 (n=31)
2026-08-18 01:52:55,960 INFO     outcome=invalid       train_rmse=1.2500 (n=1865)  val_rmse=1.8384 (n=544)
2026-08-18 01:52:55,960 INFO     outcome=win           train_rmse=0.3567 (n=89698)  val_rmse=0.3484 (n=22464)
2026-08-18 01:53:17,941 INFO epoch  23/50  train_rmse=0.4207 (norm=0.4016)  val_rmse=0.4458 (norm=0.4255)
2026-08-18 01:53:17,941 INFO     opponent=immobile   train_rmse=0.4207 (n=91800)  val_rmse=0.4458 (n=23039)
2026-08-18 01:53:17,941 INFO     outcome=ball_out      train_rmse=2.8327 (n=237)  val_rmse=2.9598 (n=31)
2026-08-18 01:53:17,941 INFO     outcome=invalid       train_rmse=1.2608 (n=1865)  val_rmse=1.6162 (n=544)
2026-08-18 01:53:17,941 INFO     outcome=win           train_rmse=0.3562 (n=89698)  val_rmse=0.3584 (n=22464)
2026-08-18 01:53:40,012 INFO epoch  24/50  train_rmse=0.4121 (norm=0.3934)  val_rmse=0.4552 (norm=0.4345)
2026-08-18 01:53:40,012 INFO     opponent=immobile   train_rmse=0.4121 (n=91800)  val_rmse=0.4552 (n=23039)
2026-08-18 01:53:40,012 INFO     outcome=ball_out      train_rmse=2.6335 (n=237)  val_rmse=2.1787 (n=31)
2026-08-18 01:53:40,012 INFO     outcome=invalid       train_rmse=1.2607 (n=1865)  val_rmse=1.3106 (n=544)
2026-08-18 01:53:40,012 INFO     outcome=win           train_rmse=0.3499 (n=89698)  val_rmse=0.4054 (n=22464)
2026-08-18 01:54:01,375 INFO epoch  25/50  train_rmse=0.4123 (norm=0.3936)  val_rmse=0.4473 (norm=0.4270)
2026-08-18 01:54:01,375 INFO     opponent=immobile   train_rmse=0.4123 (n=91800)  val_rmse=0.4473 (n=23039)
2026-08-18 01:54:01,375 INFO     outcome=ball_out      train_rmse=2.6241 (n=237)  val_rmse=3.4731 (n=31)
2026-08-18 01:54:01,375 INFO     outcome=invalid       train_rmse=1.2224 (n=1865)  val_rmse=1.5750 (n=544)
2026-08-18 01:54:01,375 INFO     outcome=win           train_rmse=0.3532 (n=89698)  val_rmse=0.3584 (n=22464)
2026-08-18 01:54:23,027 INFO epoch  26/50  train_rmse=0.4121 (norm=0.3933)  val_rmse=0.4353 (norm=0.4155)
2026-08-18 01:54:23,027 INFO     opponent=immobile   train_rmse=0.4121 (n=91800)  val_rmse=0.4353 (n=23039)
2026-08-18 01:54:23,027 INFO     outcome=ball_out      train_rmse=2.6111 (n=237)  val_rmse=2.6073 (n=31)
2026-08-18 01:54:23,027 INFO     outcome=invalid       train_rmse=1.2493 (n=1865)  val_rmse=1.4584 (n=544)
2026-08-18 01:54:23,028 INFO     outcome=win           train_rmse=0.3511 (n=89698)  val_rmse=0.3653 (n=22464)
2026-08-18 01:54:44,283 INFO epoch  27/50  train_rmse=0.4113 (norm=0.3926)  val_rmse=0.4441 (norm=0.4239)
2026-08-18 01:54:44,283 INFO     opponent=immobile   train_rmse=0.4113 (n=91800)  val_rmse=0.4441 (n=23039)
2026-08-18 01:54:44,283 INFO     outcome=ball_out      train_rmse=2.5389 (n=237)  val_rmse=2.8526 (n=31)
2026-08-18 01:54:44,283 INFO     outcome=invalid       train_rmse=1.2188 (n=1865)  val_rmse=1.5349 (n=544)
2026-08-18 01:54:44,283 INFO     outcome=win           train_rmse=0.3538 (n=89698)  val_rmse=0.3660 (n=22464)
2026-08-18 01:55:05,446 INFO epoch  28/50  train_rmse=0.4092 (norm=0.3906)  val_rmse=0.4377 (norm=0.4179)
2026-08-18 01:55:05,446 INFO     opponent=immobile   train_rmse=0.4092 (n=91800)  val_rmse=0.4377 (n=23039)
2026-08-18 01:55:05,446 INFO     outcome=ball_out      train_rmse=2.6366 (n=237)  val_rmse=2.2434 (n=31)
2026-08-18 01:55:05,446 INFO     outcome=invalid       train_rmse=1.2468 (n=1865)  val_rmse=1.3701 (n=544)
2026-08-18 01:55:05,446 INFO     outcome=win           train_rmse=0.3474 (n=89698)  val_rmse=0.3796 (n=22464)
2026-08-18 01:55:26,542 INFO epoch  29/50  train_rmse=0.4054 (norm=0.3870)  val_rmse=0.4616 (norm=0.4406)
2026-08-18 01:55:26,542 INFO     opponent=immobile   train_rmse=0.4054 (n=91800)  val_rmse=0.4616 (n=23039)
2026-08-18 01:55:26,542 INFO     outcome=ball_out      train_rmse=2.5865 (n=237)  val_rmse=2.4588 (n=31)
2026-08-18 01:55:26,542 INFO     outcome=invalid       train_rmse=1.1880 (n=1865)  val_rmse=1.9114 (n=544)
2026-08-18 01:55:26,542 INFO     outcome=win           train_rmse=0.3481 (n=89698)  val_rmse=0.3488 (n=22464)
2026-08-18 01:55:48,222 INFO epoch  30/50  train_rmse=0.4028 (norm=0.3845)  val_rmse=0.4468 (norm=0.4265)
2026-08-18 01:55:48,222 INFO     opponent=immobile   train_rmse=0.4028 (n=91800)  val_rmse=0.4468 (n=23039)
2026-08-18 01:55:48,222 INFO     outcome=ball_out      train_rmse=2.5089 (n=237)  val_rmse=4.0709 (n=31)
2026-08-18 01:55:48,222 INFO     outcome=invalid       train_rmse=1.2242 (n=1865)  val_rmse=1.5964 (n=544)
2026-08-18 01:55:48,222 INFO     outcome=win           train_rmse=0.3438 (n=89698)  val_rmse=0.3466 (n=22464)
2026-08-18 01:56:10,356 INFO epoch  31/50  train_rmse=0.4023 (norm=0.3840)  val_rmse=0.4405 (norm=0.4205)
2026-08-18 01:56:10,356 INFO     opponent=immobile   train_rmse=0.4023 (n=91800)  val_rmse=0.4405 (n=23039)
2026-08-18 01:56:10,356 INFO     outcome=ball_out      train_rmse=2.6417 (n=237)  val_rmse=3.0814 (n=31)
2026-08-18 01:56:10,356 INFO     outcome=invalid       train_rmse=1.2241 (n=1865)  val_rmse=1.3907 (n=544)
2026-08-18 01:56:10,356 INFO     outcome=win           train_rmse=0.3407 (n=89698)  val_rmse=0.3730 (n=22464)
2026-08-18 01:56:32,384 INFO epoch  32/50  train_rmse=0.4062 (norm=0.3878)  val_rmse=0.4341 (norm=0.4144)
2026-08-18 01:56:32,384 INFO     opponent=immobile   train_rmse=0.4062 (n=91800)  val_rmse=0.4341 (n=23039)
2026-08-18 01:56:32,384 INFO     outcome=ball_out      train_rmse=2.6113 (n=237)  val_rmse=2.3674 (n=31)
2026-08-18 01:56:32,384 INFO     outcome=invalid       train_rmse=1.2015 (n=1865)  val_rmse=1.4498 (n=544)
2026-08-18 01:56:32,384 INFO     outcome=win           train_rmse=0.3476 (n=89698)  val_rmse=0.3669 (n=22464)
2026-08-18 01:56:54,008 INFO epoch  33/50  train_rmse=0.4000 (norm=0.3818)  val_rmse=0.4391 (norm=0.4192)
2026-08-18 01:56:54,008 INFO     opponent=immobile   train_rmse=0.4000 (n=91800)  val_rmse=0.4391 (n=23039)
2026-08-18 01:56:54,008 INFO     outcome=ball_out      train_rmse=2.6641 (n=237)  val_rmse=3.0291 (n=31)
2026-08-18 01:56:54,008 INFO     outcome=invalid       train_rmse=1.1955 (n=1865)  val_rmse=1.7401 (n=544)
2026-08-18 01:56:54,008 INFO     outcome=win           train_rmse=0.3395 (n=89698)  val_rmse=0.3343 (n=22464)
2026-08-18 01:57:15,131 INFO epoch  34/50  train_rmse=0.3994 (norm=0.3812)  val_rmse=0.4415 (norm=0.4214)
2026-08-18 01:57:15,131 INFO     opponent=immobile   train_rmse=0.3994 (n=91800)  val_rmse=0.4415 (n=23039)
2026-08-18 01:57:15,131 INFO     outcome=ball_out      train_rmse=2.5757 (n=237)  val_rmse=3.6893 (n=31)
2026-08-18 01:57:15,131 INFO     outcome=invalid       train_rmse=1.2020 (n=1865)  val_rmse=1.5264 (n=544)
2026-08-18 01:57:15,131 INFO     outcome=win           train_rmse=0.3401 (n=89698)  val_rmse=0.3531 (n=22464)
2026-08-18 01:57:35,995 INFO epoch  35/50  train_rmse=0.3959 (norm=0.3779)  val_rmse=0.4463 (norm=0.4260)
2026-08-18 01:57:35,995 INFO     opponent=immobile   train_rmse=0.3959 (n=91800)  val_rmse=0.4463 (n=23039)
2026-08-18 01:57:35,995 INFO     outcome=ball_out      train_rmse=2.5480 (n=237)  val_rmse=2.0260 (n=31)
2026-08-18 01:57:35,995 INFO     outcome=invalid       train_rmse=1.2044 (n=1865)  val_rmse=1.3913 (n=544)
2026-08-18 01:57:35,995 INFO     outcome=win           train_rmse=0.3363 (n=89698)  val_rmse=0.3895 (n=22464)
2026-08-18 01:57:57,361 INFO epoch  36/50  train_rmse=0.3960 (norm=0.3780)  val_rmse=0.4615 (norm=0.4405)
2026-08-18 01:57:57,361 INFO     opponent=immobile   train_rmse=0.3960 (n=91800)  val_rmse=0.4615 (n=23039)
2026-08-18 01:57:57,361 INFO     outcome=ball_out      train_rmse=2.5224 (n=237)  val_rmse=4.4928 (n=31)
2026-08-18 01:57:57,361 INFO     outcome=invalid       train_rmse=1.2097 (n=1865)  val_rmse=1.8443 (n=544)
2026-08-18 01:57:57,361 INFO     outcome=win           train_rmse=0.3365 (n=89698)  val_rmse=0.3289 (n=22464)
2026-08-18 01:58:18,516 INFO epoch  37/50  train_rmse=0.3924 (norm=0.3746)  val_rmse=0.4311 (norm=0.4116)
2026-08-18 01:58:18,516 INFO     opponent=immobile   train_rmse=0.3924 (n=91800)  val_rmse=0.4311 (n=23039)
2026-08-18 01:58:18,516 INFO     outcome=ball_out      train_rmse=2.5231 (n=237)  val_rmse=2.3992 (n=31)
2026-08-18 01:58:18,516 INFO     outcome=invalid       train_rmse=1.1739 (n=1865)  val_rmse=1.3702 (n=544)
2026-08-18 01:58:18,516 INFO     outcome=win           train_rmse=0.3349 (n=89698)  val_rmse=0.3705 (n=22464)
2026-08-18 01:58:39,540 INFO epoch  38/50  train_rmse=0.3941 (norm=0.3762)  val_rmse=0.4274 (norm=0.4080)
2026-08-18 01:58:39,540 INFO     opponent=immobile   train_rmse=0.3941 (n=91800)  val_rmse=0.4274 (n=23039)
2026-08-18 01:58:39,540 INFO     outcome=ball_out      train_rmse=2.4538 (n=237)  val_rmse=3.0623 (n=31)
2026-08-18 01:58:39,540 INFO     outcome=invalid       train_rmse=1.1673 (n=1865)  val_rmse=1.4068 (n=544)
2026-08-18 01:58:39,540 INFO     outcome=win           train_rmse=0.3387 (n=89698)  val_rmse=0.3556 (n=22464)
2026-08-18 01:59:01,667 INFO epoch  39/50  train_rmse=0.3915 (norm=0.3737)  val_rmse=0.4357 (norm=0.4159)
2026-08-18 01:59:01,667 INFO     opponent=immobile   train_rmse=0.3915 (n=91800)  val_rmse=0.4357 (n=23039)
2026-08-18 01:59:01,667 INFO     outcome=ball_out      train_rmse=2.5181 (n=237)  val_rmse=3.2250 (n=31)
2026-08-18 01:59:01,667 INFO     outcome=invalid       train_rmse=1.1806 (n=1865)  val_rmse=1.6652 (n=544)
2026-08-18 01:59:01,667 INFO     outcome=win           train_rmse=0.3334 (n=89698)  val_rmse=0.3365 (n=22464)
2026-08-18 01:59:24,487 INFO epoch  40/50  train_rmse=0.3915 (norm=0.3737)  val_rmse=0.4450 (norm=0.4248)
2026-08-18 01:59:24,487 INFO     opponent=immobile   train_rmse=0.3915 (n=91800)  val_rmse=0.4450 (n=23039)
2026-08-18 01:59:24,487 INFO     outcome=ball_out      train_rmse=2.5959 (n=237)  val_rmse=2.7137 (n=31)
2026-08-18 01:59:24,487 INFO     outcome=invalid       train_rmse=1.1796 (n=1865)  val_rmse=1.6276 (n=544)
2026-08-18 01:59:24,487 INFO     outcome=win           train_rmse=0.3319 (n=89698)  val_rmse=0.3588 (n=22464)
2026-08-18 01:59:46,352 INFO epoch  41/50  train_rmse=0.3899 (norm=0.3722)  val_rmse=0.4420 (norm=0.4220)
2026-08-18 01:59:46,352 INFO     opponent=immobile   train_rmse=0.3899 (n=91800)  val_rmse=0.4420 (n=23039)
2026-08-18 01:59:46,352 INFO     outcome=ball_out      train_rmse=2.4963 (n=237)  val_rmse=2.0529 (n=31)
2026-08-18 01:59:46,352 INFO     outcome=invalid       train_rmse=1.1855 (n=1865)  val_rmse=1.7847 (n=544)
2026-08-18 01:59:46,352 INFO     outcome=win           train_rmse=0.3315 (n=89698)  val_rmse=0.3427 (n=22464)
2026-08-18 02:00:08,092 INFO epoch  42/50  train_rmse=0.3877 (norm=0.3701)  val_rmse=0.4314 (norm=0.4118)
2026-08-18 02:00:08,092 INFO     opponent=immobile   train_rmse=0.3877 (n=91800)  val_rmse=0.4314 (n=23039)
2026-08-18 02:00:08,092 INFO     outcome=ball_out      train_rmse=2.3837 (n=237)  val_rmse=2.5941 (n=31)
2026-08-18 02:00:08,092 INFO     outcome=invalid       train_rmse=1.2034 (n=1865)  val_rmse=1.5099 (n=544)
2026-08-18 02:00:08,092 INFO     outcome=win           train_rmse=0.3297 (n=89698)  val_rmse=0.3554 (n=22464)
2026-08-18 02:00:29,978 INFO epoch  43/50  train_rmse=0.3875 (norm=0.3699)  val_rmse=0.4283 (norm=0.4088)
2026-08-18 02:00:29,978 INFO     opponent=immobile   train_rmse=0.3875 (n=91800)  val_rmse=0.4283 (n=23039)
2026-08-18 02:00:29,978 INFO     outcome=ball_out      train_rmse=2.4112 (n=237)  val_rmse=3.2098 (n=31)
2026-08-18 02:00:29,978 INFO     outcome=invalid       train_rmse=1.2158 (n=1865)  val_rmse=1.6011 (n=544)
2026-08-18 02:00:29,978 INFO     outcome=win           train_rmse=0.3280 (n=89698)  val_rmse=0.3344 (n=22464)
2026-08-18 02:00:51,930 INFO epoch  44/50  train_rmse=0.3884 (norm=0.3708)  val_rmse=0.4444 (norm=0.4242)
2026-08-18 02:00:51,930 INFO     opponent=immobile   train_rmse=0.3884 (n=91800)  val_rmse=0.4444 (n=23039)
2026-08-18 02:00:51,930 INFO     outcome=ball_out      train_rmse=2.4888 (n=237)  val_rmse=2.5508 (n=31)
2026-08-18 02:00:51,930 INFO     outcome=invalid       train_rmse=1.1687 (n=1865)  val_rmse=1.6361 (n=544)
2026-08-18 02:00:51,930 INFO     outcome=win           train_rmse=0.3311 (n=89698)  val_rmse=0.3588 (n=22464)
2026-08-18 02:01:14,204 INFO epoch  45/50  train_rmse=0.3859 (norm=0.3684)  val_rmse=0.4528 (norm=0.4323)
2026-08-18 02:01:14,204 INFO     opponent=immobile   train_rmse=0.3859 (n=91800)  val_rmse=0.4528 (n=23039)
2026-08-18 02:01:14,204 INFO     outcome=ball_out      train_rmse=2.4937 (n=237)  val_rmse=3.5482 (n=31)
2026-08-18 02:01:14,204 INFO     outcome=invalid       train_rmse=1.1661 (n=1865)  val_rmse=1.7803 (n=544)
2026-08-18 02:01:14,204 INFO     outcome=win           train_rmse=0.3282 (n=89698)  val_rmse=0.3409 (n=22464)
2026-08-18 02:01:14,204 INFO Early stopping at epoch 45/50 (val normalized MSE did not improve for 7 epochs).
2026-08-18 02:01:14,204 INFO Best val normalized MSE achieved: 0.1665 (RMSE=0.4080; <1.0 = better than predicting the mean; <0.5 = useful critic)
2026-08-18 02:01:16,967 INFO --- Per-component MC-return magnitude (val rows) ---
2026-08-18 02:01:16,967 INFO   get_possession    mean=+0.3692  std=0.4636
2026-08-18 02:01:16,967 INFO   lose_possession   mean=-0.0078  std=0.0787
2026-08-18 02:01:16,967 INFO   ball_out          mean=-0.0048  std=0.1307
2026-08-18 02:01:16,967 INFO   box_possession    mean=+1.6011  std=0.3419
2026-08-18 02:01:16,967 INFO   speed_bonus       mean=+2.0778  std=0.6656
2026-08-18 02:01:16,968 INFO   stamina_penalty   mean=-0.0645  std=0.0251
2026-08-18 02:01:17,367 INFO --- Reward-component vs. value-residual correlation (523 val episodes) ---
2026-08-18 02:01:17,367 INFO   component            corr   comp_std
2026-08-18 02:01:17,370 INFO   lose_possession    +0.059     0.0895
2026-08-18 02:01:17,370 INFO   ball_out           +0.241     0.1373
2026-08-18 02:01:17,370 INFO   stamina_penalty    -0.311     0.0214
2026-08-18 02:01:17,370 INFO   speed_bonus        +0.358     0.7710
2026-08-18 02:01:17,371 INFO   get_possession     +0.422     0.2055
2026-08-18 02:01:17,371 INFO   box_possession     +0.456     0.3538
2026-08-18 02:01:17,371 INFO   (components near the top -- low |corr| despite real variance -- are the ones the value net's errors track least; read alongside the per-component MC-return magnitude above.)
2026-08-18 02:01:17,375 INFO --- Worst val episode for outcome=ball_out (1 episode(s)): rows [107550, 107580], residual=-3.187 -- saved match log to results/debug_value_worst_episode_ball_out.json ---
2026-08-18 02:01:17,376 INFO --- Worst val episode for outcome=invalid (21 episode(s)): rows [94457, 94480], residual=-3.329 -- saved match log to results/debug_value_worst_episode_invalid.json ---
2026-08-18 02:01:17,377 INFO --- Worst val episode for outcome=win (501 episode(s)): rows [100125, 100195], residual=+3.798 -- saved match log to results/debug_value_worst_episode_win.json ---
2026-08-18 02:07:57,334 INFO Checkpoint checkpoints/longterm/checkpoint_vvgood_immobile.pt: separate_value_net=True (auto-detected)
2026-08-18 02:07:57,363 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:07:57,365 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:07:57,365 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 02:07:57,368 INFO --reset-dir-log-std: reset move_dir_log_std/kick_dir_log_std to config init values before rollout collection.
2026-08-18 02:07:57,368 INFO Collecting rollout: 8 parallel worker(s), ~14375 steps/worker, worker_torch_threads=1
Rollout (8 workers): 0/115000 (  0.0%)     0.0 steps/s
2026-08-18 02:08:00,137 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:08:00,138 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:08:00,138 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 02:08:00,170 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:08:00,172 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:08:00,172 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 02:08:00,177 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:08:00,179 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:08:00,179 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:08:00,179 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 02:08:00,180 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:08:00,181 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 02:08:00,196 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:08:00,198 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:08:00,198 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 02:08:00,203 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:08:00,205 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:08:00,205 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 02:08:00,219 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:08:00,222 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:08:00,222 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 02:08:00,231 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:08:00,233 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:08:00,234 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
Rollout (8 workers): 11520/115000 ( 10.0%)   700.7 steps/s
Rollout (8 workers): 23011/115000 ( 20.0%)   745.4 steps/s
Rollout (8 workers): 34622/115000 ( 30.1%)   747.6 steps/s
Rollout (8 workers): 46068/115000 ( 40.1%)   748.5 steps/s
Rollout (8 workers): 57572/115000 ( 50.1%)   753.8 steps/s
Rollout (8 workers): 69120/115000 ( 60.1%)   746.3 steps/s
Rollout (8 workers): 80593/115000 ( 70.1%)   740.4 steps/s
Rollout (8 workers): 92052/115000 ( 80.0%)   719.5 steps/s
[worker 4] done: 143.9s total  (10.01 ms/step, 0.44 s/episode over 325 episode(s))
[worker 3] done: 144.7s total  (10.07 ms/step, 0.44 s/episode over 326 episode(s))
Rollout (8 workers): 103559/115000 ( 90.1%)   692.0 steps/s
[worker 7] done: 148.3s total  (10.32 ms/step, 0.45 s/episode over 327 episode(s))
[worker 0] done: 150.4s total  (10.46 ms/step, 0.46 s/episode over 325 episode(s))
[worker 1] done: 150.7s total  (10.48 ms/step, 0.46 s/episode over 329 episode(s))
[worker 6] done: 151.4s total  (10.53 ms/step, 0.46 s/episode over 329 episode(s))
[worker 5] done: 153.2s total  (10.66 ms/step, 0.47 s/episode over 328 episode(s))
[worker 2] done: 154.2s total  (10.73 ms/step, 0.47 s/episode over 328 episode(s))
2026-08-18 02:10:36,360 INFO   [parallel rollout] total: 159.0s wall  (1.38 ms/step aggregate, 722.5 steps/s aggregate across 8 worker(s))
2026-08-18 02:10:36,360 INFO Dropped 127 trailing (incomplete-episode) step(s) across workers
2026-08-18 02:10:36,566 INFO Rollout dataset: 114,873 steps, 2617 complete episode(s)
2026-08-18 02:10:37,140 INFO Loaded 114,873 rows total
2026-08-18 02:10:37,140 INFO has_rewards=True
2026-08-18 02:10:37,142 INFO valid_indices(): 114,873 rows (100.0% of total)
2026-08-18 02:10:37,171 INFO Returns over ALL rows: mean=3.879 std=1.053 min=-4.000 max=6.808
2026-08-18 02:10:37,172 INFO Returns over valid_indices(): mean=3.879 std=1.053
2026-08-18 02:10:37,200 INFO --- Dataset distribution (114,873 rows, 2617 episodes) ---
2026-08-18 02:10:37,202 INFO   self.ai_type == rules: 0.0%
2026-08-18 02:10:37,203 INFO   self.ai_type == immobile: 0.0%
2026-08-18 02:10:37,204 INFO   self.ai_type == neural: 100.0%
2026-08-18 02:10:37,207 INFO   opponent.ai_type == rules: 0.0%
2026-08-18 02:10:37,208 INFO   opponent.ai_type == immobile: 100.0%
2026-08-18 02:10:37,209 INFO   opponent.ai_type == neural: 0.0%
2026-08-18 02:10:37,213 INFO   valid rows: kick_this_tick rate=0.00%  tackle_attempt rate=0.00%
2026-08-18 02:10:37,214 INFO   dones=1 rows: 2,617  |  zero-reward rows: 109,950 (95.7%)
2026-08-18 02:10:37,224 INFO   return percentiles (all rows): p10=2.89  p50=3.84  p90=5.12
2026-08-18 02:10:37,284 INFO --- Reward component breakdown (all episodes, 2617 episode(s)) ---
2026-08-18 02:10:37,284 INFO   component           mean      std       min       max
2026-08-18 02:10:37,285 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,285 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,286 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,286 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,286 INFO   get_possession    +0.972    0.206    +0.000    +2.000
2026-08-18 02:10:37,287 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,287 INFO   lose_possession    -0.007    0.078    -0.900    +0.000
2026-08-18 02:10:37,287 INFO   ball_out          -0.015    0.247    -4.000    +0.000
2026-08-18 02:10:37,288 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,288 INFO   box_possession    +1.923    0.385    +0.000    +2.000
2026-08-18 02:10:37,288 INFO   speed_bonus       +2.568    0.759    +0.000    +3.880
2026-08-18 02:10:37,288 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,289 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,289 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,289 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,289 INFO   stamina_penalty    -0.075    0.030    -0.141    +0.000
2026-08-18 02:10:37,314 INFO --- Reward component breakdown (outcome=ball_out, 10 episode(s)) ---
2026-08-18 02:10:37,314 INFO   component           mean      std       min       max
2026-08-18 02:10:37,314 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,314 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,314 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,314 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,314 INFO   get_possession    +0.800    0.400    +0.000    +1.000
2026-08-18 02:10:37,314 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,314 INFO   lose_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,314 INFO   ball_out          -4.000    0.000    -4.000    -4.000
2026-08-18 02:10:37,314 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,314 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,314 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,315 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,315 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,315 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,315 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,315 INFO   stamina_penalty    +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,340 INFO --- Reward component breakdown (outcome=invalid, 91 episode(s)) ---
2026-08-18 02:10:37,341 INFO   component           mean      std       min       max
2026-08-18 02:10:37,341 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,341 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,341 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,341 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,341 INFO   get_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,341 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,341 INFO   lose_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,341 INFO   ball_out          +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,341 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,341 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,341 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,341 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,342 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,342 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,342 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,342 INFO   stamina_penalty    +0.000    0.000    +0.000    +0.000
2026-08-18 02:10:37,365 INFO --- Reward component breakdown (outcome=loss, 0 episode(s)) ---
2026-08-18 02:10:37,386 INFO --- Reward component breakdown (outcome=timeout, 0 episode(s)) ---
2026-08-18 02:10:37,445 INFO --- MC returns by outcome (all rows, 114,873 rows) ---
2026-08-18 02:10:37,447 INFO   ball_out     n=    264  mean=-2.922  std=0.524  min=-4.000  max=-1.976
2026-08-18 02:10:37,449 INFO   invalid      n=  2,312  mean=+0.000  std=0.000  min=+0.000  max=+0.000
2026-08-18 02:10:37,452 INFO   win          n=112,297  mean=+3.975  std=0.841  min=+1.799  max=+6.808
2026-08-18 02:10:37,539 INFO --- Episode total reward by outcome (all rows, 2617 episode(s)) ---
2026-08-18 02:10:37,540 INFO   ball_out     n=    10  mean=-3.200  std=0.400  min=-4.000  max=-3.000
2026-08-18 02:10:37,540 INFO   invalid      n=    91  mean=+0.000  std=0.000  min=+0.000  max=+0.000
2026-08-18 02:10:37,540 INFO   win          n= 2,516  mean=+5.594  std=0.576  min=+3.769  max=+6.808
2026-08-18 02:10:37,604 INFO Train/val split (valid_only=True): 91,718 train rows across 2094 episodes  |  23,155 val rows across 523 episodes
2026-08-18 02:10:37,655 INFO --- Linear regression baseline (top_speed, ball_dist, ball_to_box, time_remaining) ---
2026-08-18 02:10:37,674 INFO   [all outcomes] n_train=91,718  n_val=23,155  coef(top_speed, ball_dist, ball_to_box, time_rem, intercept)=[0.367, -1.572, -2.145, 6.493, -0.94]
    train_rmse=0.7731 (norm=0.7350)  val_rmse=0.7723 (norm=0.7342)
2026-08-18 02:10:37,744 INFO   [win outcomes only] n_train=89,717  n_val=22,580  coef(top_speed, ball_dist, ball_to_box, time_rem, intercept)=[0.345, -1.343, -2.234, 7.342, -1.552]
    train_rmse=0.3775 (norm=0.4487)  val_rmse=0.3568 (norm=0.4241)
2026-08-18 02:10:38,912 INFO value_net: fresh separate value_net (--reset-separate-value-net)  total_params=241,312  trainable_params=241,312
2026-08-18 02:10:38,913 INFO Fitting fresh separate value network: 50 epochs, lr=0.0002, weight_decay=1e-06, batch_size=1096, train_ret_std=1.052, outcome_reweight=False
2026-08-18 02:10:51,319 INFO epoch   0/50 (baseline, no training yet)  train_rmse=4.0736 (norm=3.8726)  val_rmse=4.0473 (norm=3.8476)
2026-08-18 02:10:51,319 INFO     opponent=immobile   train_rmse=4.0736 (n=91718)  val_rmse=4.0473 (n=23155)
2026-08-18 02:10:51,319 INFO     outcome=ball_out      train_rmse=3.0295 (n=211)  val_rmse=2.4171 (n=53)
2026-08-18 02:10:51,319 INFO     outcome=invalid       train_rmse=0.0507 (n=1790)  val_rmse=0.0525 (n=522)
2026-08-18 02:10:51,319 INFO     outcome=win           train_rmse=4.1162 (n=89717)  val_rmse=4.0969 (n=22580)
2026-08-18 02:11:14,292 INFO epoch   1/50  train_rmse=2.0421 (norm=1.9413)  val_rmse=0.7913 (norm=0.7523)
2026-08-18 02:11:14,293 INFO     opponent=immobile   train_rmse=2.0421 (n=91718)  val_rmse=0.7913 (n=23155)
2026-08-18 02:11:14,293 INFO     outcome=ball_out      train_rmse=6.2464 (n=211)  val_rmse=5.7357 (n=53)
2026-08-18 02:11:14,293 INFO     outcome=invalid       train_rmse=3.2510 (n=1790)  val_rmse=3.4201 (n=522)
2026-08-18 02:11:14,293 INFO     outcome=win           train_rmse=1.9901 (n=89717)  val_rmse=0.5427 (n=22580)
2026-08-18 02:11:38,489 INFO epoch   2/50  train_rmse=0.7080 (norm=0.6731)  val_rmse=0.6537 (norm=0.6214)
2026-08-18 02:11:38,489 INFO     opponent=immobile   train_rmse=0.7080 (n=91718)  val_rmse=0.6537 (n=23155)
2026-08-18 02:11:38,489 INFO     outcome=ball_out      train_rmse=6.1878 (n=211)  val_rmse=5.7812 (n=53)
2026-08-18 02:11:38,489 INFO     outcome=invalid       train_rmse=2.9639 (n=1790)  val_rmse=2.4772 (n=522)
2026-08-18 02:11:38,489 INFO     outcome=win           train_rmse=0.4972 (n=89717)  val_rmse=0.4668 (n=22580)
2026-08-18 02:12:01,095 INFO epoch   3/50  train_rmse=0.6282 (norm=0.5972)  val_rmse=0.6125 (norm=0.5822)
2026-08-18 02:12:01,095 INFO     opponent=immobile   train_rmse=0.6282 (n=91718)  val_rmse=0.6125 (n=23155)
2026-08-18 02:12:01,095 INFO     outcome=ball_out      train_rmse=5.6707 (n=211)  val_rmse=5.7785 (n=53)
2026-08-18 02:12:01,095 INFO     outcome=invalid       train_rmse=2.3714 (n=1790)  val_rmse=2.1280 (n=522)
2026-08-18 02:12:01,095 INFO     outcome=win           train_rmse=0.4644 (n=89717)  val_rmse=0.4490 (n=22580)
2026-08-18 02:12:23,463 INFO epoch   4/50  train_rmse=0.5805 (norm=0.5518)  val_rmse=0.5882 (norm=0.5592)
2026-08-18 02:12:23,463 INFO     opponent=immobile   train_rmse=0.5805 (n=91718)  val_rmse=0.5882 (n=23155)
2026-08-18 02:12:23,463 INFO     outcome=ball_out      train_rmse=5.4266 (n=211)  val_rmse=5.9115 (n=53)
2026-08-18 02:12:23,463 INFO     outcome=invalid       train_rmse=1.9720 (n=1790)  val_rmse=2.0385 (n=522)
2026-08-18 02:12:23,463 INFO     outcome=win           train_rmse=0.4445 (n=89717)  val_rmse=0.4204 (n=22580)
2026-08-18 02:12:44,694 INFO epoch   5/50  train_rmse=0.5428 (norm=0.5160)  val_rmse=0.5711 (norm=0.5429)
2026-08-18 02:12:44,695 INFO     opponent=immobile   train_rmse=0.5428 (n=91718)  val_rmse=0.5711 (n=23155)
2026-08-18 02:12:44,695 INFO     outcome=ball_out      train_rmse=5.2074 (n=211)  val_rmse=5.9575 (n=53)
2026-08-18 02:12:44,695 INFO     outcome=invalid       train_rmse=1.6752 (n=1790)  val_rmse=1.7412 (n=522)
2026-08-18 02:12:44,695 INFO     outcome=win           train_rmse=0.4259 (n=89717)  val_rmse=0.4255 (n=22580)
2026-08-18 02:13:06,402 INFO epoch   6/50  train_rmse=0.5211 (norm=0.4954)  val_rmse=0.5631 (norm=0.5354)
2026-08-18 02:13:06,402 INFO     opponent=immobile   train_rmse=0.5211 (n=91718)  val_rmse=0.5631 (n=23155)
2026-08-18 02:13:06,402 INFO     outcome=ball_out      train_rmse=5.0362 (n=211)  val_rmse=5.7701 (n=53)
2026-08-18 02:13:06,402 INFO     outcome=invalid       train_rmse=1.5053 (n=1790)  val_rmse=1.3329 (n=522)
2026-08-18 02:13:06,402 INFO     outcome=win           train_rmse=0.4156 (n=89717)  val_rmse=0.4538 (n=22580)
2026-08-18 02:13:28,139 INFO epoch   7/50  train_rmse=0.5068 (norm=0.4818)  val_rmse=0.5587 (norm=0.5312)
2026-08-18 02:13:28,139 INFO     opponent=immobile   train_rmse=0.5068 (n=91718)  val_rmse=0.5587 (n=23155)
2026-08-18 02:13:28,139 INFO     outcome=ball_out      train_rmse=5.0038 (n=211)  val_rmse=5.9640 (n=53)
2026-08-18 02:13:28,139 INFO     outcome=invalid       train_rmse=1.4258 (n=1790)  val_rmse=1.4671 (n=522)
2026-08-18 02:13:28,139 INFO     outcome=win           train_rmse=0.4039 (n=89717)  val_rmse=0.4323 (n=22580)
2026-08-18 02:13:49,868 INFO epoch   8/50  train_rmse=0.4997 (norm=0.4750)  val_rmse=0.5631 (norm=0.5353)
2026-08-18 02:13:49,868 INFO     opponent=immobile   train_rmse=0.4997 (n=91718)  val_rmse=0.5631 (n=23155)
2026-08-18 02:13:49,868 INFO     outcome=ball_out      train_rmse=4.8488 (n=211)  val_rmse=6.0412 (n=53)
2026-08-18 02:13:49,868 INFO     outcome=invalid       train_rmse=1.3669 (n=1790)  val_rmse=1.8802 (n=522)
2026-08-18 02:13:49,868 INFO     outcome=win           train_rmse=0.4033 (n=89717)  val_rmse=0.3972 (n=22580)
2026-08-18 02:14:11,654 INFO epoch   9/50  train_rmse=0.4857 (norm=0.4617)  val_rmse=0.5569 (norm=0.5295)
2026-08-18 02:14:11,654 INFO     opponent=immobile   train_rmse=0.4857 (n=91718)  val_rmse=0.5569 (n=23155)
2026-08-18 02:14:11,654 INFO     outcome=ball_out      train_rmse=4.8646 (n=211)  val_rmse=5.9742 (n=53)
2026-08-18 02:14:11,654 INFO     outcome=invalid       train_rmse=1.3226 (n=1790)  val_rmse=1.6534 (n=522)
2026-08-18 02:14:11,654 INFO     outcome=win           train_rmse=0.3881 (n=89717)  val_rmse=0.4136 (n=22580)
2026-08-18 02:14:34,673 INFO epoch  10/50  train_rmse=0.4801 (norm=0.4564)  val_rmse=0.5379 (norm=0.5114)
2026-08-18 02:14:34,673 INFO     opponent=immobile   train_rmse=0.4801 (n=91718)  val_rmse=0.5379 (n=23155)
2026-08-18 02:14:34,673 INFO     outcome=ball_out      train_rmse=4.8057 (n=211)  val_rmse=5.7500 (n=53)
2026-08-18 02:14:34,673 INFO     outcome=invalid       train_rmse=1.2638 (n=1790)  val_rmse=1.7495 (n=522)
2026-08-18 02:14:34,673 INFO     outcome=win           train_rmse=0.3866 (n=89717)  val_rmse=0.3852 (n=22580)
2026-08-18 02:14:57,342 INFO epoch  11/50  train_rmse=0.4703 (norm=0.4471)  val_rmse=0.5303 (norm=0.5041)
2026-08-18 02:14:57,342 INFO     opponent=immobile   train_rmse=0.4703 (n=91718)  val_rmse=0.5303 (n=23155)
2026-08-18 02:14:57,342 INFO     outcome=ball_out      train_rmse=4.8034 (n=211)  val_rmse=5.8083 (n=53)
2026-08-18 02:14:57,342 INFO     outcome=invalid       train_rmse=1.2231 (n=1790)  val_rmse=1.4873 (n=522)
2026-08-18 02:14:57,342 INFO     outcome=win           train_rmse=0.3768 (n=89717)  val_rmse=0.3975 (n=22580)
2026-08-18 02:15:22,110 INFO epoch  12/50  train_rmse=0.4630 (norm=0.4401)  val_rmse=0.5286 (norm=0.5025)
2026-08-18 02:15:22,110 INFO     opponent=immobile   train_rmse=0.4630 (n=91718)  val_rmse=0.5286 (n=23155)
2026-08-18 02:15:22,110 INFO     outcome=ball_out      train_rmse=4.7313 (n=211)  val_rmse=5.9243 (n=53)
2026-08-18 02:15:22,110 INFO     outcome=invalid       train_rmse=1.1720 (n=1790)  val_rmse=1.5623 (n=522)
2026-08-18 02:15:22,110 INFO     outcome=win           train_rmse=0.3729 (n=89717)  val_rmse=0.3844 (n=22580)
2026-08-18 02:15:45,423 INFO epoch  13/50  train_rmse=0.4593 (norm=0.4367)  val_rmse=0.5466 (norm=0.5196)
2026-08-18 02:15:45,424 INFO     opponent=immobile   train_rmse=0.4593 (n=91718)  val_rmse=0.5466 (n=23155)
2026-08-18 02:15:45,424 INFO     outcome=ball_out      train_rmse=4.6826 (n=211)  val_rmse=5.5505 (n=53)
2026-08-18 02:15:45,424 INFO     outcome=invalid       train_rmse=1.1799 (n=1790)  val_rmse=1.4334 (n=522)
2026-08-18 02:15:45,424 INFO     outcome=win           train_rmse=0.3692 (n=89717)  val_rmse=0.4319 (n=22580)
2026-08-18 02:16:08,370 INFO epoch  14/50  train_rmse=0.4617 (norm=0.4389)  val_rmse=0.5162 (norm=0.4907)
2026-08-18 02:16:08,370 INFO     opponent=immobile   train_rmse=0.4617 (n=91718)  val_rmse=0.5162 (n=23155)
2026-08-18 02:16:08,370 INFO     outcome=ball_out      train_rmse=4.6847 (n=211)  val_rmse=5.7599 (n=53)
2026-08-18 02:16:08,370 INFO     outcome=invalid       train_rmse=1.1501 (n=1790)  val_rmse=1.4577 (n=522)
2026-08-18 02:16:08,370 INFO     outcome=win           train_rmse=0.3740 (n=89717)  val_rmse=0.3824 (n=22580)
2026-08-18 02:16:31,273 INFO epoch  15/50  train_rmse=0.4507 (norm=0.4284)  val_rmse=0.5197 (norm=0.4941)
2026-08-18 02:16:31,274 INFO     opponent=immobile   train_rmse=0.4507 (n=91718)  val_rmse=0.5197 (n=23155)
2026-08-18 02:16:31,274 INFO     outcome=ball_out      train_rmse=4.5793 (n=211)  val_rmse=5.6213 (n=53)
2026-08-18 02:16:31,274 INFO     outcome=invalid       train_rmse=1.1011 (n=1790)  val_rmse=1.7265 (n=522)
2026-08-18 02:16:31,274 INFO     outcome=win           train_rmse=0.3662 (n=89717)  val_rmse=0.3659 (n=22580)
2026-08-18 02:16:56,040 INFO epoch  16/50  train_rmse=0.4491 (norm=0.4270)  val_rmse=0.5232 (norm=0.4974)
2026-08-18 02:16:56,040 INFO     opponent=immobile   train_rmse=0.4491 (n=91718)  val_rmse=0.5232 (n=23155)
2026-08-18 02:16:56,040 INFO     outcome=ball_out      train_rmse=4.7285 (n=211)  val_rmse=5.8149 (n=53)
2026-08-18 02:16:56,040 INFO     outcome=invalid       train_rmse=1.1208 (n=1790)  val_rmse=1.4012 (n=522)
2026-08-18 02:16:56,040 INFO     outcome=win           train_rmse=0.3586 (n=89717)  val_rmse=0.3949 (n=22580)
2026-08-18 02:17:21,597 INFO epoch  17/50  train_rmse=0.4436 (norm=0.4217)  val_rmse=0.5118 (norm=0.4866)
2026-08-18 02:17:21,597 INFO     opponent=immobile   train_rmse=0.4436 (n=91718)  val_rmse=0.5118 (n=23155)
2026-08-18 02:17:21,597 INFO     outcome=ball_out      train_rmse=4.6804 (n=211)  val_rmse=5.8254 (n=53)
2026-08-18 02:17:21,597 INFO     outcome=invalid       train_rmse=1.0827 (n=1790)  val_rmse=1.4713 (n=522)
2026-08-18 02:17:21,597 INFO     outcome=win           train_rmse=0.3553 (n=89717)  val_rmse=0.3727 (n=22580)
2026-08-18 02:17:44,154 INFO epoch  18/50  train_rmse=0.4386 (norm=0.4169)  val_rmse=0.5117 (norm=0.4865)
2026-08-18 02:17:44,154 INFO     opponent=immobile   train_rmse=0.4386 (n=91718)  val_rmse=0.5117 (n=23155)
2026-08-18 02:17:44,154 INFO     outcome=ball_out      train_rmse=4.6595 (n=211)  val_rmse=5.7210 (n=53)
2026-08-18 02:17:44,154 INFO     outcome=invalid       train_rmse=1.0652 (n=1790)  val_rmse=1.4874 (n=522)
2026-08-18 02:17:44,154 INFO     outcome=win           train_rmse=0.3506 (n=89717)  val_rmse=0.3749 (n=22580)
2026-08-18 02:18:06,785 INFO epoch  19/50  train_rmse=0.4374 (norm=0.4158)  val_rmse=0.5080 (norm=0.4829)
2026-08-18 02:18:06,785 INFO     opponent=immobile   train_rmse=0.4374 (n=91718)  val_rmse=0.5080 (n=23155)
2026-08-18 02:18:06,785 INFO     outcome=ball_out      train_rmse=4.6132 (n=211)  val_rmse=5.7675 (n=53)
2026-08-18 02:18:06,785 INFO     outcome=invalid       train_rmse=1.0754 (n=1790)  val_rmse=1.3868 (n=522)
2026-08-18 02:18:06,785 INFO     outcome=win           train_rmse=0.3499 (n=89717)  val_rmse=0.3769 (n=22580)
2026-08-18 02:18:29,436 INFO epoch  20/50  train_rmse=0.4370 (norm=0.4154)  val_rmse=0.5181 (norm=0.4925)
2026-08-18 02:18:29,437 INFO     opponent=immobile   train_rmse=0.4370 (n=91718)  val_rmse=0.5181 (n=23155)
2026-08-18 02:18:29,437 INFO     outcome=ball_out      train_rmse=4.5720 (n=211)  val_rmse=5.7056 (n=53)
2026-08-18 02:18:29,437 INFO     outcome=invalid       train_rmse=1.0478 (n=1790)  val_rmse=1.2660 (n=522)
2026-08-18 02:18:29,437 INFO     outcome=win           train_rmse=0.3523 (n=89717)  val_rmse=0.4022 (n=22580)
2026-08-18 02:18:52,463 INFO epoch  21/50  train_rmse=0.4329 (norm=0.4115)  val_rmse=0.5028 (norm=0.4780)
2026-08-18 02:18:52,463 INFO     opponent=immobile   train_rmse=0.4329 (n=91718)  val_rmse=0.5028 (n=23155)
2026-08-18 02:18:52,463 INFO     outcome=ball_out      train_rmse=4.6514 (n=211)  val_rmse=5.5386 (n=53)
2026-08-18 02:18:52,463 INFO     outcome=invalid       train_rmse=1.0581 (n=1790)  val_rmse=1.3466 (n=522)
2026-08-18 02:18:52,463 INFO     outcome=win           train_rmse=0.3440 (n=89717)  val_rmse=0.3812 (n=22580)
2026-08-18 02:19:15,174 INFO epoch  22/50  train_rmse=0.4282 (norm=0.4071)  val_rmse=0.5217 (norm=0.4960)
2026-08-18 02:19:15,174 INFO     opponent=immobile   train_rmse=0.4282 (n=91718)  val_rmse=0.5217 (n=23155)
2026-08-18 02:19:15,174 INFO     outcome=ball_out      train_rmse=4.5699 (n=211)  val_rmse=5.7309 (n=53)
2026-08-18 02:19:15,174 INFO     outcome=invalid       train_rmse=1.0197 (n=1790)  val_rmse=1.6096 (n=522)
2026-08-18 02:19:15,174 INFO     outcome=win           train_rmse=0.3429 (n=89717)  val_rmse=0.3770 (n=22580)
2026-08-18 02:19:37,604 INFO epoch  23/50  train_rmse=0.4293 (norm=0.4082)  val_rmse=0.5157 (norm=0.4902)
2026-08-18 02:19:37,604 INFO     opponent=immobile   train_rmse=0.4293 (n=91718)  val_rmse=0.5157 (n=23155)
2026-08-18 02:19:37,604 INFO     outcome=ball_out      train_rmse=4.6066 (n=211)  val_rmse=5.6712 (n=53)
2026-08-18 02:19:37,604 INFO     outcome=invalid       train_rmse=1.0207 (n=1790)  val_rmse=1.5294 (n=522)
2026-08-18 02:19:37,604 INFO     outcome=win           train_rmse=0.3432 (n=89717)  val_rmse=0.3783 (n=22580)
2026-08-18 02:20:03,222 INFO epoch  24/50  train_rmse=0.4246 (norm=0.4037)  val_rmse=0.5105 (norm=0.4853)
2026-08-18 02:20:03,222 INFO     opponent=immobile   train_rmse=0.4246 (n=91718)  val_rmse=0.5105 (n=23155)
2026-08-18 02:20:03,222 INFO     outcome=ball_out      train_rmse=4.5784 (n=211)  val_rmse=5.6763 (n=53)
2026-08-18 02:20:03,222 INFO     outcome=invalid       train_rmse=1.0179 (n=1790)  val_rmse=1.6051 (n=522)
2026-08-18 02:20:03,223 INFO     outcome=win           train_rmse=0.3382 (n=89717)  val_rmse=0.3634 (n=22580)
2026-08-18 02:20:27,063 INFO epoch  25/50  train_rmse=0.4238 (norm=0.4029)  val_rmse=0.5042 (norm=0.4793)
2026-08-18 02:20:27,063 INFO     opponent=immobile   train_rmse=0.4238 (n=91718)  val_rmse=0.5042 (n=23155)
2026-08-18 02:20:27,063 INFO     outcome=ball_out      train_rmse=4.6028 (n=211)  val_rmse=5.5161 (n=53)
2026-08-18 02:20:27,063 INFO     outcome=invalid       train_rmse=1.0040 (n=1790)  val_rmse=1.4299 (n=522)
2026-08-18 02:20:27,063 INFO     outcome=win           train_rmse=0.3371 (n=89717)  val_rmse=0.3768 (n=22580)
2026-08-18 02:20:50,892 INFO epoch  26/50  train_rmse=0.4226 (norm=0.4018)  val_rmse=0.5026 (norm=0.4778)
2026-08-18 02:20:50,892 INFO     opponent=immobile   train_rmse=0.4226 (n=91718)  val_rmse=0.5026 (n=23155)
2026-08-18 02:20:50,892 INFO     outcome=ball_out      train_rmse=4.5161 (n=211)  val_rmse=5.7035 (n=53)
2026-08-18 02:20:50,892 INFO     outcome=invalid       train_rmse=1.0254 (n=1790)  val_rmse=1.2775 (n=522)
2026-08-18 02:20:50,892 INFO     outcome=win           train_rmse=0.3371 (n=89717)  val_rmse=0.3807 (n=22580)
2026-08-18 02:21:18,343 INFO epoch  27/50  train_rmse=0.4234 (norm=0.4025)  val_rmse=0.5152 (norm=0.4898)
2026-08-18 02:21:18,344 INFO     opponent=immobile   train_rmse=0.4234 (n=91718)  val_rmse=0.5152 (n=23155)
2026-08-18 02:21:18,344 INFO     outcome=ball_out      train_rmse=4.4987 (n=211)  val_rmse=5.7728 (n=53)
2026-08-18 02:21:18,344 INFO     outcome=invalid       train_rmse=0.9632 (n=1790)  val_rmse=1.6579 (n=522)
2026-08-18 02:21:18,344 INFO     outcome=win           train_rmse=0.3422 (n=89717)  val_rmse=0.3612 (n=22580)
2026-08-18 02:22:28,256 INFO epoch  28/50  train_rmse=0.4201 (norm=0.3994)  val_rmse=0.5047 (norm=0.4798)
2026-08-18 02:22:28,256 INFO     opponent=immobile   train_rmse=0.4201 (n=91718)  val_rmse=0.5047 (n=23155)
2026-08-18 02:22:28,256 INFO     outcome=ball_out      train_rmse=4.5257 (n=211)  val_rmse=5.7389 (n=53)
2026-08-18 02:22:28,256 INFO     outcome=invalid       train_rmse=1.0031 (n=1790)  val_rmse=1.4298 (n=522)
2026-08-18 02:22:28,256 INFO     outcome=win           train_rmse=0.3349 (n=89717)  val_rmse=0.3697 (n=22580)
2026-08-18 02:23:16,768 INFO epoch  29/50  train_rmse=0.4169 (norm=0.3964)  val_rmse=0.4908 (norm=0.4666)
2026-08-18 02:23:16,769 INFO     opponent=immobile   train_rmse=0.4169 (n=91718)  val_rmse=0.4908 (n=23155)
2026-08-18 02:23:16,769 INFO     outcome=ball_out      train_rmse=4.5319 (n=211)  val_rmse=5.6489 (n=53)
2026-08-18 02:23:16,769 INFO     outcome=invalid       train_rmse=0.9791 (n=1790)  val_rmse=1.3289 (n=522)
2026-08-18 02:23:16,769 INFO     outcome=win           train_rmse=0.3321 (n=89717)  val_rmse=0.3624 (n=22580)
2026-08-18 02:24:00,682 INFO epoch  30/50  train_rmse=0.4164 (norm=0.3959)  val_rmse=0.5101 (norm=0.4849)
2026-08-18 02:24:00,683 INFO     opponent=immobile   train_rmse=0.4164 (n=91718)  val_rmse=0.5101 (n=23155)
2026-08-18 02:24:00,683 INFO     outcome=ball_out      train_rmse=4.4843 (n=211)  val_rmse=5.6192 (n=53)
2026-08-18 02:24:00,683 INFO     outcome=invalid       train_rmse=0.9712 (n=1790)  val_rmse=1.6782 (n=522)
2026-08-18 02:24:00,683 INFO     outcome=win           train_rmse=0.3334 (n=89717)  val_rmse=0.3572 (n=22580)
2026-08-18 02:24:38,992 INFO epoch  31/50  train_rmse=0.4144 (norm=0.3939)  val_rmse=0.4991 (norm=0.4744)
2026-08-18 02:24:38,992 INFO     opponent=immobile   train_rmse=0.4144 (n=91718)  val_rmse=0.4991 (n=23155)
2026-08-18 02:24:38,992 INFO     outcome=ball_out      train_rmse=4.5243 (n=211)  val_rmse=5.6955 (n=53)
2026-08-18 02:24:38,992 INFO     outcome=invalid       train_rmse=0.9741 (n=1790)  val_rmse=1.4395 (n=522)
2026-08-18 02:24:38,992 INFO     outcome=win           train_rmse=0.3293 (n=89717)  val_rmse=0.3624 (n=22580)
2026-08-18 02:25:07,169 INFO epoch  32/50  train_rmse=0.4129 (norm=0.3925)  val_rmse=0.4929 (norm=0.4686)
2026-08-18 02:25:07,169 INFO     opponent=immobile   train_rmse=0.4129 (n=91718)  val_rmse=0.4929 (n=23155)
2026-08-18 02:25:07,169 INFO     outcome=ball_out      train_rmse=4.4863 (n=211)  val_rmse=5.5788 (n=53)
2026-08-18 02:25:07,169 INFO     outcome=invalid       train_rmse=0.9827 (n=1790)  val_rmse=1.3742 (n=522)
2026-08-18 02:25:07,169 INFO     outcome=win           train_rmse=0.3282 (n=89717)  val_rmse=0.3639 (n=22580)
2026-08-18 02:25:29,973 INFO epoch  33/50  train_rmse=0.4161 (norm=0.3956)  val_rmse=0.4958 (norm=0.4713)
2026-08-18 02:25:29,973 INFO     opponent=immobile   train_rmse=0.4161 (n=91718)  val_rmse=0.4958 (n=23155)
2026-08-18 02:25:29,973 INFO     outcome=ball_out      train_rmse=4.5090 (n=211)  val_rmse=5.6002 (n=53)
2026-08-18 02:25:29,973 INFO     outcome=invalid       train_rmse=0.9698 (n=1790)  val_rmse=1.5130 (n=522)
2026-08-18 02:25:29,973 INFO     outcome=win           train_rmse=0.3323 (n=89717)  val_rmse=0.3543 (n=22580)
2026-08-18 02:25:53,515 INFO epoch  34/50  train_rmse=0.4162 (norm=0.3957)  val_rmse=0.5018 (norm=0.4771)
2026-08-18 02:25:53,515 INFO     opponent=immobile   train_rmse=0.4162 (n=91718)  val_rmse=0.5018 (n=23155)
2026-08-18 02:25:53,515 INFO     outcome=ball_out      train_rmse=4.4378 (n=211)  val_rmse=5.5797 (n=53)
2026-08-18 02:25:53,515 INFO     outcome=invalid       train_rmse=0.9314 (n=1790)  val_rmse=1.6357 (n=522)
2026-08-18 02:25:53,515 INFO     outcome=win           train_rmse=0.3369 (n=89717)  val_rmse=0.3512 (n=22580)
2026-08-18 02:26:16,283 INFO epoch  35/50  train_rmse=0.4085 (norm=0.3884)  val_rmse=0.4926 (norm=0.4683)
2026-08-18 02:26:16,283 INFO     opponent=immobile   train_rmse=0.4085 (n=91718)  val_rmse=0.4926 (n=23155)
2026-08-18 02:26:16,283 INFO     outcome=ball_out      train_rmse=4.4288 (n=211)  val_rmse=5.5717 (n=53)
2026-08-18 02:26:16,283 INFO     outcome=invalid       train_rmse=0.9606 (n=1790)  val_rmse=1.3989 (n=522)
2026-08-18 02:26:16,283 INFO     outcome=win           train_rmse=0.3257 (n=89717)  val_rmse=0.3616 (n=22580)
2026-08-18 02:26:38,787 INFO epoch  36/50  train_rmse=0.4076 (norm=0.3875)  val_rmse=0.5067 (norm=0.4817)
2026-08-18 02:26:38,787 INFO     opponent=immobile   train_rmse=0.4076 (n=91718)  val_rmse=0.5067 (n=23155)
2026-08-18 02:26:38,787 INFO     outcome=ball_out      train_rmse=4.4877 (n=211)  val_rmse=5.6431 (n=53)
2026-08-18 02:26:38,787 INFO     outcome=invalid       train_rmse=0.9568 (n=1790)  val_rmse=1.4842 (n=522)
2026-08-18 02:26:38,787 INFO     outcome=win           train_rmse=0.3228 (n=89717)  val_rmse=0.3709 (n=22580)
2026-08-18 02:26:38,787 INFO Early stopping at epoch 36/50 (val normalized MSE did not improve for 7 epochs).
2026-08-18 02:26:38,787 INFO Best val normalized MSE achieved: 0.2177 (RMSE=0.4666; <1.0 = better than predicting the mean; <0.5 = useful critic)
2026-08-18 02:26:41,721 INFO --- Per-component MC-return magnitude (val rows) ---
2026-08-18 02:26:41,722 INFO   get_possession    mean=+0.3668  std=0.4583
2026-08-18 02:26:41,722 INFO   lose_possession   mean=-0.0056  std=0.0658
2026-08-18 02:26:41,722 INFO   ball_out          mean=-0.0075  std=0.1575
2026-08-18 02:26:41,722 INFO   box_possession    mean=+1.6001  std=0.3421
2026-08-18 02:26:41,722 INFO   speed_bonus       mean=+1.9666  std=0.6804
2026-08-18 02:26:41,723 INFO   stamina_penalty   mean=-0.0648  std=0.0251
2026-08-18 02:26:42,174 INFO --- Reward-component vs. value-residual correlation (523 val episodes) ---
2026-08-18 02:26:42,174 INFO   component            corr   comp_std
2026-08-18 02:26:42,177 INFO   lose_possession    +0.051     0.0685
2026-08-18 02:26:42,177 INFO   stamina_penalty    -0.315     0.0212
2026-08-18 02:26:42,177 INFO   ball_out           +0.378     0.1151
2026-08-18 02:26:42,177 INFO   speed_bonus        +0.466     0.7790
2026-08-18 02:26:42,177 INFO   get_possession     +0.500     0.1918
2026-08-18 02:26:42,177 INFO   box_possession     +0.576     0.3491
2026-08-18 02:26:42,177 INFO   (components near the top -- low |corr| despite real variance -- are the ones the value net's errors track least; read alongside the per-component MC-return magnitude above.)
2026-08-18 02:26:42,190 INFO --- Worst val episode for outcome=ball_out (1 episode(s)): rows [107589, 107641], residual=-5.010 -- saved match log to results/debug_value_worst_episode_ball_out.json ---
2026-08-18 02:26:42,192 INFO --- Worst val episode for outcome=invalid (20 episode(s)): rows [96556, 96582], residual=-3.322 -- saved match log to results/debug_value_worst_episode_invalid.json ---
2026-08-18 02:26:42,193 INFO --- Worst val episode for outcome=win (502 episode(s)): rows [97141, 97175], residual=+4.178 -- saved match log to results/debug_value_worst_episode_win.json ---
2026-08-18 02:28:39,451 INFO Checkpoint checkpoints/longterm/checkpoint_vvgood_immobile.pt: separate_value_net=True (auto-detected)
2026-08-18 02:28:39,479 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:28:39,480 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:28:39,480 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 02:28:39,483 INFO --reset-dir-log-std: reset move_dir_log_std/kick_dir_log_std to config init values before rollout collection.
2026-08-18 02:28:39,483 INFO Collecting rollout: 8 parallel worker(s), ~14375 steps/worker, worker_torch_threads=1
Rollout (8 workers): 0/115000 (  0.0%)     0.0 steps/s
2026-08-18 02:28:42,430 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:28:42,432 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:28:42,432 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 02:28:42,448 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:28:42,450 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:28:42,450 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 02:28:42,457 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:28:42,458 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:28:42,459 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 02:28:42,481 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:28:42,481 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:28:42,483 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:28:42,483 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 02:28:42,484 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:28:42,484 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 02:28:42,484 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:28:42,486 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:28:42,487 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 02:28:42,487 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:28:42,489 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:28:42,489 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
2026-08-18 02:28:42,553 WARNING execution_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:28:42,555 WARNING value_net: skipped 1 shape-mismatched param(s) (network architecture changed since checkpoint was saved), keeping fresh init for: ['value_head.1.weight']
2026-08-18 02:28:42,555 INFO Loaded checkpoint: checkpoints/longterm/checkpoint_vvgood_immobile.pt (step 6015060)
Rollout (8 workers): 11537/115000 ( 10.0%)   632.5 steps/s
Rollout (8 workers): 23017/115000 ( 20.0%)   671.4 steps/s
Rollout (8 workers): 34653/115000 ( 30.1%)   688.7 steps/s
Rollout (8 workers): 46126/115000 ( 40.1%)   703.7 steps/s
Rollout (8 workers): 57515/115000 ( 50.0%)   713.7 steps/s
Rollout (8 workers): 69101/115000 ( 60.1%)   721.2 steps/s
Rollout (8 workers): 80543/115000 ( 70.0%)   722.6 steps/s
Rollout (8 workers): 92083/115000 ( 80.1%)   726.8 steps/s
Rollout (8 workers): 103652/115000 ( 90.1%)   729.3 steps/s
[worker 7] done: 140.3s total  (9.76 ms/step, 0.43 s/episode over 327 episode(s))
[worker 6] done: 142.2s total  (9.89 ms/step, 0.43 s/episode over 329 episode(s))
[worker 0] done: 143.4s total  (9.98 ms/step, 0.44 s/episode over 325 episode(s))
[worker 2] done: 143.8s total  (10.01 ms/step, 0.44 s/episode over 328 episode(s))
[worker 1] done: 143.8s total  (10.01 ms/step, 0.44 s/episode over 329 episode(s))
[worker 4] done: 145.2s total  (10.10 ms/step, 0.45 s/episode over 325 episode(s))
[worker 3] done: 145.5s total  (10.12 ms/step, 0.45 s/episode over 326 episode(s))
[worker 5] done: 145.7s total  (10.14 ms/step, 0.44 s/episode over 328 episode(s))
2026-08-18 02:31:09,797 INFO   [parallel rollout] total: 150.3s wall  (1.31 ms/step aggregate, 764.2 steps/s aggregate across 8 worker(s))
2026-08-18 02:31:09,797 INFO Dropped 127 trailing (incomplete-episode) step(s) across workers
2026-08-18 02:31:09,907 INFO Rollout dataset: 114,873 steps, 2617 complete episode(s)
2026-08-18 02:31:10,360 INFO Loaded 114,873 rows total
2026-08-18 02:31:10,361 INFO has_rewards=True
2026-08-18 02:31:10,362 INFO valid_indices(): 114,873 rows (100.0% of total)
2026-08-18 02:31:10,398 INFO Returns over ALL rows: mean=3.879 std=1.053 min=-4.000 max=6.808
2026-08-18 02:31:10,399 INFO Returns over valid_indices(): mean=3.879 std=1.053
2026-08-18 02:31:10,427 INFO --- Dataset distribution (114,873 rows, 2617 episodes) ---
2026-08-18 02:31:10,428 INFO   self.ai_type == rules: 0.0%
2026-08-18 02:31:10,429 INFO   self.ai_type == immobile: 0.0%
2026-08-18 02:31:10,429 INFO   self.ai_type == neural: 100.0%
2026-08-18 02:31:10,430 INFO   opponent.ai_type == rules: 0.0%
2026-08-18 02:31:10,430 INFO   opponent.ai_type == immobile: 100.0%
2026-08-18 02:31:10,430 INFO   opponent.ai_type == neural: 0.0%
2026-08-18 02:31:10,433 INFO   valid rows: kick_this_tick rate=0.00%  tackle_attempt rate=0.00%
2026-08-18 02:31:10,434 INFO   dones=1 rows: 2,617  |  zero-reward rows: 109,950 (95.7%)
2026-08-18 02:31:10,442 INFO   return percentiles (all rows): p10=2.89  p50=3.84  p90=5.12
2026-08-18 02:31:10,505 INFO --- Reward component breakdown (all episodes, 2617 episode(s)) ---
2026-08-18 02:31:10,505 INFO   component           mean      std       min       max
2026-08-18 02:31:10,505 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,506 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,506 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,506 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,506 INFO   get_possession    +0.972    0.206    +0.000    +2.000
2026-08-18 02:31:10,507 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,507 INFO   lose_possession    -0.007    0.078    -0.900    +0.000
2026-08-18 02:31:10,507 INFO   ball_out          -0.015    0.247    -4.000    +0.000
2026-08-18 02:31:10,507 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,508 INFO   box_possession    +1.923    0.385    +0.000    +2.000
2026-08-18 02:31:10,508 INFO   speed_bonus       +2.568    0.759    +0.000    +3.880
2026-08-18 02:31:10,508 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,508 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,508 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,509 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,509 INFO   stamina_penalty    -0.075    0.030    -0.141    +0.000
2026-08-18 02:31:10,533 INFO --- Reward component breakdown (outcome=ball_out, 10 episode(s)) ---
2026-08-18 02:31:10,533 INFO   component           mean      std       min       max
2026-08-18 02:31:10,533 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,533 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,534 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,534 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,534 INFO   get_possession    +0.800    0.400    +0.000    +1.000
2026-08-18 02:31:10,534 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,534 INFO   lose_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,534 INFO   ball_out          -4.000    0.000    -4.000    -4.000
2026-08-18 02:31:10,534 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,534 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,534 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,534 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,534 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,534 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,534 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,534 INFO   stamina_penalty    +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,558 INFO --- Reward component breakdown (outcome=invalid, 91 episode(s)) ---
2026-08-18 02:31:10,558 INFO   component           mean      std       min       max
2026-08-18 02:31:10,558 INFO   approach          +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,558 INFO   retreat           +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,558 INFO   approach_speed    +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,558 INFO   heading           +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,558 INFO   get_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,558 INFO   progress          +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,558 INFO   lose_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,558 INFO   ball_out          +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,558 INFO   illegal           +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,558 INFO   box_possession    +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,558 INFO   speed_bonus       +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,559 INFO   opponent_box      +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,559 INFO   timeout           +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,559 INFO   proximity_bonus    +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,559 INFO   step_penalty      +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,559 INFO   stamina_penalty    +0.000    0.000    +0.000    +0.000
2026-08-18 02:31:10,581 INFO --- Reward component breakdown (outcome=loss, 0 episode(s)) ---
2026-08-18 02:31:10,602 INFO --- Reward component breakdown (outcome=timeout, 0 episode(s)) ---
2026-08-18 02:31:10,657 INFO --- MC returns by outcome (all rows, 114,873 rows) ---
2026-08-18 02:31:10,660 INFO   ball_out     n=    264  mean=-2.922  std=0.524  min=-4.000  max=-1.976
2026-08-18 02:31:10,661 INFO   invalid      n=  2,312  mean=+0.000  std=0.000  min=+0.000  max=+0.000
2026-08-18 02:31:10,663 INFO   win          n=112,297  mean=+3.975  std=0.841  min=+1.799  max=+6.808
2026-08-18 02:31:10,732 INFO --- Episode total reward by outcome (all rows, 2617 episode(s)) ---
2026-08-18 02:31:10,733 INFO   ball_out     n=    10  mean=-3.200  std=0.400  min=-4.000  max=-3.000
2026-08-18 02:31:10,733 INFO   invalid      n=    91  mean=+0.000  std=0.000  min=+0.000  max=+0.000
2026-08-18 02:31:10,733 INFO   win          n= 2,516  mean=+5.594  std=0.576  min=+3.769  max=+6.808
2026-08-18 02:31:10,792 INFO Train/val split (valid_only=True): 91,718 train rows across 2094 episodes  |  23,155 val rows across 523 episodes
2026-08-18 02:31:10,842 INFO --- Linear regression baseline (top_speed, ball_dist, ball_to_box, time_remaining) ---
2026-08-18 02:31:10,860 INFO   [all outcomes] n_train=91,718  n_val=23,155  coef(top_speed, ball_dist, ball_to_box, time_rem, intercept)=[0.367, -1.572, -2.145, 6.493, -0.94]
    train_rmse=0.7731 (norm=0.7350)  val_rmse=0.7723 (norm=0.7342)
2026-08-18 02:31:10,931 INFO   [win outcomes only] n_train=89,717  n_val=22,580  coef(top_speed, ball_dist, ball_to_box, time_rem, intercept)=[0.345, -1.343, -2.234, 7.342, -1.552]
    train_rmse=0.3775 (norm=0.4487)  val_rmse=0.3568 (norm=0.4241)
2026-08-18 02:31:12,047 INFO value_net: fresh separate value_net (--reset-separate-value-net)  total_params=265,892  trainable_params=265,892
2026-08-18 02:31:12,047 INFO Fitting fresh separate value network: 50 epochs, lr=0.0002, weight_decay=1e-06, batch_size=1096, train_ret_std=1.052, outcome_reweight=False
2026-08-18 02:31:25,897 INFO epoch   0/50 (baseline, no training yet)  train_rmse=3.9696 (norm=3.7738)  val_rmse=3.9428 (norm=3.7482)
2026-08-18 02:31:25,897 INFO     opponent=immobile   train_rmse=3.9696 (n=91718)  val_rmse=3.9428 (n=23155)
2026-08-18 02:31:25,897 INFO     outcome=ball_out      train_rmse=3.1197 (n=211)  val_rmse=2.5011 (n=53)
2026-08-18 02:31:25,897 INFO     outcome=invalid       train_rmse=0.0369 (n=1790)  val_rmse=0.0390 (n=522)
2026-08-18 02:31:25,897 INFO     outcome=win           train_rmse=4.0108 (n=89717)  val_rmse=3.9908 (n=22580)
2026-08-18 02:31:52,483 INFO epoch   1/50  train_rmse=2.0257 (norm=1.9257)  val_rmse=0.7763 (norm=0.7380)
2026-08-18 02:31:52,484 INFO     opponent=immobile   train_rmse=2.0257 (n=91718)  val_rmse=0.7763 (n=23155)
2026-08-18 02:31:52,484 INFO     outcome=ball_out      train_rmse=6.3094 (n=211)  val_rmse=5.5128 (n=53)
2026-08-18 02:31:52,484 INFO     outcome=invalid       train_rmse=3.2831 (n=1790)  val_rmse=3.2625 (n=522)
2026-08-18 02:31:52,484 INFO     outcome=win           train_rmse=1.9713 (n=89717)  val_rmse=0.5483 (n=22580)
2026-08-18 02:32:19,795 INFO epoch   2/50  train_rmse=0.7061 (norm=0.6712)  val_rmse=0.6617 (norm=0.6290)
2026-08-18 02:32:19,795 INFO     opponent=immobile   train_rmse=0.7061 (n=91718)  val_rmse=0.6617 (n=23155)
2026-08-18 02:32:19,795 INFO     outcome=ball_out      train_rmse=6.3138 (n=211)  val_rmse=5.9348 (n=53)
2026-08-18 02:32:19,795 INFO     outcome=invalid       train_rmse=2.9641 (n=1790)  val_rmse=2.6753 (n=522)
2026-08-18 02:32:19,795 INFO     outcome=win           train_rmse=0.4905 (n=89717)  val_rmse=0.4481 (n=22580)
2026-08-18 02:32:47,245 INFO epoch   3/50  train_rmse=0.6331 (norm=0.6018)  val_rmse=0.6188 (norm=0.5882)
2026-08-18 02:32:47,245 INFO     opponent=immobile   train_rmse=0.6331 (n=91718)  val_rmse=0.6188 (n=23155)
2026-08-18 02:32:47,245 INFO     outcome=ball_out      train_rmse=5.8664 (n=211)  val_rmse=5.8990 (n=53)
2026-08-18 02:32:47,245 INFO     outcome=invalid       train_rmse=2.4442 (n=1790)  val_rmse=2.3279 (n=522)
2026-08-18 02:32:47,245 INFO     outcome=win           train_rmse=0.4578 (n=89717)  val_rmse=0.4309 (n=22580)
2026-08-18 02:33:13,740 INFO epoch   4/50  train_rmse=0.5802 (norm=0.5515)  val_rmse=0.5920 (norm=0.5628)
2026-08-18 02:33:13,740 INFO     opponent=immobile   train_rmse=0.5802 (n=91718)  val_rmse=0.5920 (n=23155)
2026-08-18 02:33:13,740 INFO     outcome=ball_out      train_rmse=5.4827 (n=211)  val_rmse=5.7676 (n=53)
2026-08-18 02:33:13,740 INFO     outcome=invalid       train_rmse=1.9875 (n=1790)  val_rmse=2.0898 (n=522)
2026-08-18 02:33:13,740 INFO     outcome=win           train_rmse=0.4411 (n=89717)  val_rmse=0.4247 (n=22580)
2026-08-18 02:33:40,076 INFO epoch   5/50  train_rmse=0.5486 (norm=0.5215)  val_rmse=0.5889 (norm=0.5599)
2026-08-18 02:33:40,077 INFO     opponent=immobile   train_rmse=0.5486 (n=91718)  val_rmse=0.5889 (n=23155)
2026-08-18 02:33:40,077 INFO     outcome=ball_out      train_rmse=5.3006 (n=211)  val_rmse=5.7240 (n=53)
2026-08-18 02:33:40,077 INFO     outcome=invalid       train_rmse=1.7417 (n=1790)  val_rmse=1.6473 (n=522)
2026-08-18 02:33:40,077 INFO     outcome=win           train_rmse=0.4255 (n=89717)  val_rmse=0.4648 (n=22580)
2026-08-18 02:34:06,183 INFO epoch   6/50  train_rmse=0.5256 (norm=0.4996)  val_rmse=0.5664 (norm=0.5385)
2026-08-18 02:34:06,183 INFO     opponent=immobile   train_rmse=0.5256 (n=91718)  val_rmse=0.5664 (n=23155)
2026-08-18 02:34:06,183 INFO     outcome=ball_out      train_rmse=5.1085 (n=211)  val_rmse=5.7498 (n=53)
2026-08-18 02:34:06,183 INFO     outcome=invalid       train_rmse=1.5696 (n=1790)  val_rmse=1.4472 (n=522)
2026-08-18 02:34:06,183 INFO     outcome=win           train_rmse=0.4146 (n=89717)  val_rmse=0.4506 (n=22580)
