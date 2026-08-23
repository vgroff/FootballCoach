2026-08-23 11:48:07,285 INFO Loaded 8760 shard(s) from physics_pretrain_data/ball/
2026-08-23 11:48:07,542 INFO Dataset: 2,190,000 episodes (1,861,500 train / 328,500 val)
2026-08-23 11:48:08,050 INFO pos_weight (max cap: 1.5):
2026-08-23 11:48:08,050 INFO     t= 0.2s  out_of_bounds=1.50  goal_scored=1.50
2026-08-23 11:48:08,050 INFO     t= 0.5s  out_of_bounds=1.50  goal_scored=1.50
2026-08-23 11:48:08,050 INFO     t= 1.0s  out_of_bounds=1.50  goal_scored=1.50
2026-08-23 11:48:08,050 INFO     t= 2.0s  out_of_bounds=1.50  goal_scored=1.50
2026-08-23 11:48:08,050 INFO     t= 3.5s  out_of_bounds=1.50  goal_scored=1.50
2026-08-23 11:48:08,050 INFO     t= 5.0s  out_of_bounds=1.36  goal_scored=1.50
2026-08-23 11:48:08,050 INFO     t= 7.0s  out_of_bounds=1.18  goal_scored=1.50
2026-08-23 11:48:08,050 INFO     t=10.0s  out_of_bounds=1.09  goal_scored=1.50
2026-08-23 11:48:14,295 INFO Main-loop optimizer: Adam (lr=5.00e-05, weight_decay=0.0)
2026-08-23 11:48:16,276 INFO Widened checkpoint from checkpoints/physics_pretrain/ball_encoder_37.midtrain_latest.pt to current config dims (hidden_dim: 512->1024); resumed (phase=midtrain_latest)
2026-08-23 11:48:16,953 INFO Latent diagnostics (50,000 rows, latent_dim=41):
    per-dim std: mean=0.2919  pooled=0.3842  |  latent norm: mean=2.3656 std=0.6858
    dead dims (std < threshold): 0/41
    off-diagonal |corr|: mean=0.2938  max=0.8553 (dims (33, 36))  redundant pairs (|corr|>threshold): 0
    effective rank: 5.39/41 (participation ratio)  95%-variance components: 10/41  condition number: 2.49e+04
    smallest-std dims: 39(std=0.1090,mean=0.0141), 40(std=0.1157,mean=-0.0070), 2(std=0.1364,mean=0.0036), 38(std=0.1408,mean=0.0520), 22(std=0.1721,mean=-0.0058)
    largest-std dims:  0(std=0.7209,mean=-0.2195), 5(std=0.6021,mean=0.4192), 36(std=0.5705,mean=0.3474), 1(std=0.5030,mean=0.0202), 37(std=0.4498,mean=-0.2356)
    most-correlated pairs: (33,36)=0.855, (33,21)=-0.830, (36,21)=-0.813, (24,14)=-0.785, (24,20)=0.776
2026-08-23 11:48:31,401 INFO Adjacent-pair training enabled: 7 start-horizon(s), max_skip=2 (13 (start, skip) combos total), min_start_speed=0.50m/s, 9,478,100/13,030,500 (horizon, row) combos mask-eligible (shares rows/batches with autoencode/t0 -- no separate rows of its own anymore)
2026-08-23 11:48:31,421 INFO Training row-count summary (train split):
    main (per-horizon heads)        : 1,861,500 rows -- own batches
    autoencode/t0 (bottleneck recon): 14,892,000 rows -- own batches (1,861,500 rows x 8 horizons)
    adjacent-pair (dynamics)        : 9,478,100/14,892,000 horizon-pass rows mask-eligible -- shares the horizon pass's own latent, no extra rows/batches
    crossing_head (at each horizon) : 2,910,706/14,892,000 horizon-pass rows mask-eligible -- shares the horizon pass's own latent, no extra rows/batches
    resting_head (at each horizon)  : 2,208,032/14,892,000 horizon-pass rows mask-eligible -- shares the horizon pass's own latent, no extra rows/batches
    crossing_head (at t=0, in main) : 846,156/1,861,500 main rows masked-valid (position term only; delta_t trains on all 1,861,500) -- NO extra rows/batches, shares main's own latent
    resting_head (at t=0, in main)  : 412,334/1,861,500 main rows masked-valid -- NO extra rows/batches, shares main's own latent
2026-08-23 11:48:31,421 INFO LR schedule: cosine warm restarts, T_0=40 epochs, T_mult=1, eta_min=1.00e-05, peak_decay=0.5
2026-08-23 12:03:44,914 INFO Saved 'midtrain_latest' checkpoint to checkpoints/physics_pretrain/ball_encoder_38.midtrain_latest.pt
2026-08-23 12:03:44,914 INFO epoch 1/1000: train_loss=0.0059  val_loss=0.0060  best=0.0060  (improved by inf > min_delta=1.0e-07)  lr=5.00e-05  train_pair_loss=0.0006  val_pair_loss=0.0006  train_t0_loss=0.0000  val_t0_loss=0.0000  train_crossing_loss=0.1877 (pos_dist=0.0209, dt_mae=0.1095)  val_crossing_loss=0.2727 (pos_dist=0.0206, dt_mae=0.1189)  train_resting_loss=0.0005 (pos_dist=0.0131)  train_position_loss=0.0000 (pos_dist=0.0005)  train_event_loss=0.0897 (oob_acc=0.9915, goal_acc=0.9734)  train_backprop_loss=0.0079  val_resting_loss=0.0005 (pos_dist=0.0135)  val_position_loss=0.0000 (pos_dist=0.0003)  val_event_loss=0.0987 (oob_acc=0.9876, goal_acc=0.9727)  val_backprop_loss=0.0077
2026-08-23 12:03:44,915 INFO     grad_norm: mean=0.180596 std=0.096306 min=0.069324 max=0.871781
2026-08-23 12:03:44,915 INFO     train_loss_delta (batch-to-batch): mean=-0.000003 std=0.000244 min=-0.001510 max=0.001428
2026-08-23 12:03:44,915 INFO     val_loss_delta (epoch-over-epoch): nan
2026-08-23 12:03:44,915 INFO     train pos_rmse  by horizon (m): [0.9017 0.8429 1.0064 1.2187 1.231  1.1266 1.1073 1.2099], mean: 1.0806 m
2026-08-23 12:03:44,915 INFO     train pos_dist  by horizon (m): [1.3834 1.3284 1.4916 1.7071 1.6388 1.5216 1.4754 1.532 ], mean: 1.5098 m
2026-08-23 12:03:44,915 INFO     train vel_rmse  by horizon (m/s): [0.8137 0.6419 0.7776 1.4698 1.721  1.6022 1.4951 1.4499], mean: 1.2464 m/s
2026-08-23 12:03:44,915 INFO     train vel_dist  by horizon (m/s): [1.0274 0.8639 1.0352 2.0205 2.2653 2.0441 1.9397 1.9602], mean: 1.6445 m/s
2026-08-23 12:03:44,916 INFO     train pos_r2    by horizon: [0.9982 0.9985 0.998  0.9976 0.998  0.9986 0.9988 0.9987], mean: 0.9983
2026-08-23 12:03:44,916 INFO     train vel_r2    by horizon: [ 0.9917  0.9931  0.9857  0.9184  0.8123  0.6271  0.3586 -0.4613], mean: 0.6532
2026-08-23 12:03:44,916 INFO     train pos_err_pct_disp by horizon: [47.6396 19.4469 12.8915  9.388   6.8357  5.3005  4.6616  4.763 ], mean: 13.8658
2026-08-23 12:03:44,916 INFO     train vel_err_pct_disp by horizon: [11.069   8.7784 11.194  18.4325 18.1853 16.6871 14.5755 13.7477], mean: 14.0837
2026-08-23 12:03:44,916 INFO     train pos_err_pct_ballistic by horizon: [251.4566  67.4945  39.0955  21.6467   9.2737   4.8545   2.8896   1.9231], mean: 49.8293
2026-08-23 12:03:44,916 INFO     train vel_err_pct_ballistic by horizon: [10.456   7.5521  8.4023 11.0668  7.9329  5.3603  3.6556  2.5165], mean: 7.1178
2026-08-23 12:03:44,917 INFO     train t0 pos_rmse  by horizon (m): [0.6113 0.3785 0.2395 0.1906 0.1639 0.1148 0.0795 0.0729], mean: 0.2314 m
2026-08-23 12:03:44,917 INFO     val   pos_rmse  by horizon (m): [0.9137 0.8362 1.0086 1.2154 1.2235 1.1391 1.1194 1.1944], mean: 1.0813 m
2026-08-23 12:03:44,917 INFO     val   pos_dist  by horizon (m): [1.4081 1.3147 1.5007 1.6979 1.6238 1.5225 1.4673 1.4848], mean: 1.5025 m
2026-08-23 12:03:44,917 INFO     val   vel_rmse  by horizon (m/s): [0.8309 0.6519 0.7791 1.4726 1.7279 1.6073 1.4957 1.4523], mean: 1.2522 m/s
2026-08-23 12:03:44,917 INFO     val   vel_dist  by horizon (m/s): [1.0412 0.859  1.023  2.023  2.2662 2.0492 1.9404 1.9651], mean: 1.6459 m/s
2026-08-23 12:03:44,917 INFO     val   pos_r2    by horizon: [0.9981 0.9985 0.998  0.9976 0.998  0.9985 0.9988 0.9987], mean: 0.9983
2026-08-23 12:03:44,917 INFO     val   vel_r2    by horizon: [ 0.9914  0.9929  0.9857  0.9181  0.8108  0.6246  0.3579 -0.466 ], mean: 0.6519
2026-08-23 12:03:44,917 INFO     val   pos_err_pct_disp by horizon: [48.2576 19.2862 12.918   9.3626  6.7939  5.3581  4.7096  4.6977], mean: 13.9230
2026-08-23 12:03:44,918 INFO     val   vel_err_pct_disp by horizon: [11.2782  8.9164 11.2119 18.4642 18.2587 16.7412 14.5826 13.7698], mean: 14.1529
2026-08-23 12:03:44,918 INFO     val   t0 pos_rmse  by horizon (m): [0.6156 0.3786 0.2395 0.1931 0.1653 0.1166 0.087  0.08  ], mean: 0.2345 m
2026-08-23 12:03:44,918 INFO     val   pos_err_pct_ballistic by horizon: [254.7188  66.9368  39.1757  21.5881   9.217    4.9073   2.9194   1.8967], mean: 50.1700
2026-08-23 12:03:44,918 INFO     val   vel_err_pct_ballistic by horizon: [10.6536  7.6708  8.4158 11.0859  7.9649  5.3777  3.6574  2.5205], mean: 7.1683
