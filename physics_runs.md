2026-08-22 17:30:48,128 INFO epoch 8/10: train_loss=0.0071  val_loss=0.0071  best=0.0071  (patience 1/15)  lr=9.14e-06  train_pair_loss=0.0006  val_pair_loss=0.0006  train_t0_loss=0.0001  val_t0_loss=0.0001  train_crossing_loss=0.2316 (pos_dist=0.0282, dt_mae=0.1137)  val_crossing_loss=0.2624 (pos_dist=0.0281, dt_mae=0.1175)  train_resting_loss=0.0005 (pos_dist=0.0127)  train_backprop_loss=0.0093  val_resting_loss=0.0005 (pos_dist=0.0128)  val_backprop_loss=0.0085
2026-08-22 17:30:48,129 INFO     grad_norm: mean=0.087376 std=0.019750 min=0.051386 max=0.173336
2026-08-22 17:30:48,129 INFO     train_loss_delta (batch-to-batch): mean=-0.000001 std=0.000169 min=-0.000462 max=0.000481
2026-08-22 17:30:48,129 INFO     val_loss_delta (epoch-over-epoch): 0.000009
2026-08-22 17:30:48,129 INFO     train pos_rmse  by horizon (m): [0.9702 0.9196 1.1054 1.4174 1.3247 1.2112 1.2116 1.3311], mean: 1.1864 m
2026-08-22 17:30:48,129 INFO     train pos_dist  by horizon (m): [1.5011 1.4555 1.6437 1.9977 1.8188 1.6476 1.6245 1.7201], mean: 1.6761 m
2026-08-22 17:30:48,129 INFO     train vel_rmse  by horizon (m/s): [0.977  0.7451 0.8443 1.5939 1.8911 1.6876 1.6182 1.5509], mean: 1.3635 m/s
2026-08-22 17:30:48,129 INFO     train vel_dist  by horizon (m/s): [1.3114 1.0078 1.1582 2.24   2.4501 2.2095 2.1515 2.1081], mean: 1.8296 m/s
2026-08-22 17:30:48,130 INFO     train pos_r2    by horizon: [0.9979 0.9982 0.9976 0.9967 0.9977 0.9984 0.9985 0.9984], mean: 0.9979
2026-08-22 17:30:48,130 INFO     train vel_r2    by horizon: [ 0.9881  0.9907  0.9832  0.9041  0.7734  0.5863  0.2486 -0.6716], mean: 0.6003
2026-08-22 17:30:48,130 INFO     train pos_err_pct_disp by horizon: [51.2387 21.2097 14.157  10.9167  7.3548  5.6957  5.0964  5.2331], mean: 15.1128
2026-08-22 17:30:48,130 INFO     train vel_err_pct_disp by horizon: [13.2555 10.1869 12.1486 19.984  19.9827 17.5754 15.7761 14.7039], mean: 15.4516
2026-08-22 17:30:48,130 INFO     train pos_err_pct_ballistic by horizon: [270.4538  73.6127  42.9332  25.1716   9.978    5.2165   3.1592   2.1129], mean: 54.0797
2026-08-22 17:30:48,130 INFO     train vel_err_pct_ballistic by horizon: [12.5214  8.7638  9.1188 11.9984  8.7169  5.6457  3.9567  2.6915], mean: 7.9267
2026-08-22 17:30:48,130 INFO     train t0 pos_rmse  by horizon (m): [0.7466 0.4656 0.3117 0.2619 0.1857 0.1174 0.0626 0.0412], mean: 0.2741 m
2026-08-22 17:30:48,131 INFO     val   pos_rmse  by horizon (m): [0.9775 0.915  1.1077 1.4215 1.3291 1.2182 1.2183 1.3354], mean: 1.1903 m
2026-08-22 17:30:48,131 INFO     val   pos_dist  by horizon (m): [1.5223 1.4494 1.6495 2.0029 1.8237 1.6591 1.6319 1.7229], mean: 1.6827 m
2026-08-22 17:30:48,131 INFO     val   vel_rmse  by horizon (m/s): [0.9797 0.7452 0.8441 1.5982 1.8963 1.6889 1.6199 1.5551], mean: 1.3659 m/s
2026-08-22 17:30:48,131 INFO     val   vel_dist  by horizon (m/s): [1.3178 1.0062 1.1566 2.2459 2.4514 2.2073 2.1513 2.109 ], mean: 1.8307 m/s
2026-08-22 17:30:48,131 INFO     val   pos_r2    by horizon: [0.9979 0.9982 0.9976 0.9967 0.9977 0.9983 0.9985 0.9984], mean: 0.9979
2026-08-22 17:30:48,131 INFO     val   vel_r2    by horizon: [ 0.988   0.9907  0.9832  0.9036  0.7722  0.5856  0.2469 -0.6807], mean: 0.5987
2026-08-22 17:30:48,131 INFO     val   pos_err_pct_disp by horizon: [51.6233 21.1049 14.1868 10.9496  7.38    5.7293  5.1251  5.25  ], mean: 15.1686
2026-08-22 17:30:48,131 INFO     val   vel_err_pct_disp by horizon: [13.2923 10.1887 12.1462 20.0367 20.0377 17.5908 15.7931 14.7436], mean: 15.4786
2026-08-22 17:30:48,132 INFO     val   t0 pos_rmse  by horizon (m): [0.7373 0.4596 0.308  0.261  0.1871 0.1192 0.0662 0.046 ], mean: 0.2731 m
2026-08-22 17:30:48,132 INFO     val   pos_err_pct_ballistic by horizon: [272.484   73.2491  43.0235  25.2474  10.0121   5.2473   3.177    2.1197], mean: 54.3200
2026-08-22 17:30:48,132 INFO     val   vel_err_pct_ballistic by horizon: [12.5562  8.7654  9.117  12.03    8.7409  5.6506  3.961   2.6988], mean: 7.9400
2026-08-22 17:32:45,451 INFO epoch 9/10: train_loss=0.0071  val_loss=0.0071  best=0.0071  (patience 2/15)  lr=8.89e-06  train_pair_loss=0.0006  val_pair_loss=0.0006  train_t0_loss=0.0001  val_t0_loss=0.0001  train_crossing_loss=0.2314 (pos_dist=0.0282, dt_mae=0.1136)  val_crossing_loss=0.2623 (pos_dist=0.0281, dt_mae=0.1174)  train_resting_loss=0.0005 (pos_dist=0.0127)  train_backprop_loss=0.0093  val_resting_loss=0.0005 (pos_dist=0.0128)  val_backprop_loss=0.0085
2026-08-22 17:32:45,451 INFO     grad_norm: mean=0.090584 std=0.023166 min=0.041273 max=0.209468
2026-08-22 17:32:45,451 INFO     train_loss_delta (batch-to-batch): mean=-0.000001 std=0.000169 min=-0.000592 max=0.000556
2026-08-22 17:32:45,452 INFO     val_loss_delta (epoch-over-epoch): 0.000022
2026-08-22 17:32:45,452 INFO     train pos_rmse  by horizon (m): [0.9703 0.9201 1.1056 1.4173 1.3245 1.2115 1.2115 1.331 ], mean: 1.1865 m
2026-08-22 17:32:45,452 INFO     train pos_dist  by horizon (m): [1.5013 1.4564 1.6438 1.9972 1.8187 1.6483 1.6244 1.7201], mean: 1.6763 m
2026-08-22 17:32:45,452 INFO     train vel_rmse  by horizon (m/s): [0.9768 0.7454 0.8443 1.5938 1.891  1.6872 1.6184 1.5511], mean: 1.3635 m/s
2026-08-22 17:32:45,452 INFO     train vel_dist  by horizon (m/s): [1.311  1.008  1.1582 2.2396 2.4497 2.2092 2.1519 2.1084], mean: 1.8295 m/s
2026-08-22 17:32:45,452 INFO     train pos_r2    by horizon: [0.9979 0.9982 0.9976 0.9967 0.9977 0.9984 0.9985 0.9984], mean: 0.9979
2026-08-22 17:32:45,453 INFO     train vel_r2    by horizon: [ 0.9881  0.9907  0.9832  0.9041  0.7735  0.5864  0.2481 -0.6722], mean: 0.6002
2026-08-22 17:32:45,453 INFO     train pos_err_pct_disp by horizon: [51.2428 21.2221 14.161  10.9157  7.353   5.6975  5.0965  5.2335], mean: 15.1153
2026-08-22 17:32:45,453 INFO     train vel_err_pct_disp by horizon: [13.252  10.1908 12.1499 19.9815 19.9774 17.5732 15.7803 14.7064], mean: 15.4514
2026-08-22 17:32:45,453 INFO     train pos_err_pct_ballistic by horizon: [270.4756  73.6556  42.9452  25.1694   9.9755   5.2181   3.1592   2.113 ], mean: 54.0890
2026-08-22 17:32:45,453 INFO     train vel_err_pct_ballistic by horizon: [12.5181  8.7671  9.1198 11.9969  8.7146  5.645   3.9578  2.692 ], mean: 7.9264
2026-08-22 17:32:45,453 INFO     train t0 pos_rmse  by horizon (m): [0.7457 0.4649 0.311  0.2612 0.1856 0.1172 0.0628 0.041 ], mean: 0.2737 m
2026-08-22 17:32:45,453 INFO     val   pos_rmse  by horizon (m): [0.9756 0.9192 1.1045 1.417  1.3307 1.2189 1.2191 1.3446], mean: 1.1912 m
2026-08-22 17:32:45,454 INFO     val   pos_dist  by horizon (m): [1.5158 1.4548 1.641  1.9982 1.8232 1.6566 1.6369 1.7423], mean: 1.6836 m
2026-08-22 17:32:45,454 INFO     val   vel_rmse  by horizon (m/s): [0.9837 0.7433 0.843  1.6013 1.8991 1.6919 1.6252 1.5612], mean: 1.3686 m/s
2026-08-22 17:32:45,454 INFO     val   vel_dist  by horizon (m/s): [1.3233 1.0061 1.156  2.2495 2.4529 2.2139 2.1701 2.1292], mean: 1.8376 m/s
2026-08-22 17:32:45,454 INFO     val   pos_r2    by horizon: [0.9979 0.9982 0.9976 0.9967 0.9977 0.9983 0.9985 0.9983], mean: 0.9979
2026-08-22 17:32:45,454 INFO     val   vel_r2    by horizon: [ 0.9879  0.9907  0.9833  0.9032  0.7715  0.5841  0.2419 -0.694 ], mean: 0.5961
2026-08-22 17:32:45,454 INFO     val   pos_err_pct_disp by horizon: [51.5248 21.2019 14.1464 10.9148  7.3888  5.7328  5.1282  5.2858], mean: 15.1654
2026-08-22 17:32:45,454 INFO     val   vel_err_pct_disp by horizon: [13.3465 10.163  12.1308 20.0755 20.0672 17.6216 15.8453 14.8018], mean: 15.5065
2026-08-22 17:32:45,455 INFO     val   t0 pos_rmse  by horizon (m): [0.7348 0.4578 0.3051 0.2586 0.1851 0.1166 0.0643 0.0438], mean: 0.2708 m
2026-08-22 17:32:45,455 INFO     val   pos_err_pct_ballistic by horizon: [271.9643  73.5855  42.901   25.1672  10.0241   5.2504   3.1789   2.1342], mean: 54.2757
2026-08-22 17:32:45,455 INFO     val   vel_err_pct_ballistic by horizon: [12.6073  8.7433  9.1054 12.0533  8.7538  5.6605  3.9741  2.7094], mean: 7.9509
2026-08-22 17:34:11,639 INFO Loaded 8760 shard(s) from physics_pretrain_data/ball/
2026-08-22 17:34:11,882 INFO Dataset: 2,190,000 episodes (1,861,500 train / 328,500 val)
2026-08-22 17:34:12,392 INFO pos_weight (max cap: 1.5):
2026-08-22 17:34:12,392 INFO     t= 0.2s  out_of_bounds=1.50  goal_scored=1.50
2026-08-22 17:34:12,392 INFO     t= 0.5s  out_of_bounds=1.50  goal_scored=1.50
2026-08-22 17:34:12,392 INFO     t= 1.0s  out_of_bounds=1.50  goal_scored=1.50
2026-08-22 17:34:12,392 INFO     t= 2.0s  out_of_bounds=1.50  goal_scored=1.50
2026-08-22 17:34:12,392 INFO     t= 3.5s  out_of_bounds=1.50  goal_scored=1.50
2026-08-22 17:34:12,392 INFO     t= 5.0s  out_of_bounds=1.36  goal_scored=1.50
2026-08-22 17:34:12,392 INFO     t= 7.0s  out_of_bounds=1.18  goal_scored=1.50
2026-08-22 17:34:12,392 INFO     t=10.0s  out_of_bounds=1.09  goal_scored=1.50
2026-08-22 17:34:18,902 INFO Main-loop optimizer: Adam (lr=1.00e-05, weight_decay=0.0)
2026-08-22 17:34:21,402 INFO Widened checkpoint from checkpoints/physics_pretrain/ball_encoder_34.midtrain_latest.pt to current config dims (latent_dim: 36->38, decoder_hidden_dim: 20->21); resumed (phase=midtrain_latest)
2026-08-22 17:34:39,859 INFO Adjacent-pair training enabled: 7 start-horizon(s), max_skip=2 (13 (start, skip) combos total), min_start_speed=0.50m/s, 9,478,100/13,030,500 (horizon, row) combos mask-eligible (shares rows/batches with autoencode/t0 -- no separate rows of its own anymore)
2026-08-22 17:34:39,882 INFO Training row-count summary (train split):
    main (per-horizon heads)        : 1,861,500 rows -- own batches
    autoencode/t0 (bottleneck recon): 14,892,000 rows -- own batches (1,861,500 rows x 8 horizons)
    adjacent-pair (dynamics)        : 9,478,100/14,892,000 horizon-pass rows mask-eligible -- shares the horizon pass's own latent, no extra rows/batches
    crossing_head (at each horizon) : 2,910,706/14,892,000 horizon-pass rows mask-eligible -- shares the horizon pass's own latent, no extra rows/batches
    resting_head (at each horizon)  : 2,208,032/14,892,000 horizon-pass rows mask-eligible -- shares the horizon pass's own latent, no extra rows/batches
    crossing_head (at t=0, in main) : 846,156/1,861,500 main rows masked-valid (position term only; delta_t trains on all 1,861,500) -- NO extra rows/batches, shares main's own latent
    resting_head (at t=0, in main)  : 412,334/1,861,500 main rows masked-valid -- NO extra rows/batches, shares main's own latent
2026-08-22 17:34:39,882 INFO LR schedule: cosine warm restarts, T_0=30 epochs, T_mult=1, eta_min=3.30e-06, peak_decay=0.7
2026-08-22 17:34:39,882 INFO Decoder-only-pretrain optimizer: Adam (lr=3.00e-06, weight_decay=0.0)
2026-08-22 17:34:39,882 INFO Decoder-only pretraining: 15 epoch(s), encoder TRUNK frozen (encoder.out latent layer trainable, identity rows gradient-masked), lr=3.00e-06
2026-08-22 17:36:20,682 INFO decoder-only pretrain epoch 1/15: train_loss=0.0073  val_loss=0.0076  best=0.0076  (improved)  train_pair_loss=0.0007  val_pair_loss=0.0007  train_t0_loss=0.0001  val_t0_loss=0.0001  train_crossing_loss=0.3233 (pos_dist=0.0679, dt_mae=0.3352)  val_crossing_loss=0.3983 (pos_dist=0.1080, dt_mae=0.3966)  train_resting_loss=0.0006 (pos_dist=0.0150)  train_position_loss=0.5837 (pos_dist=0.6741)  train_event_loss=1.3657 (oob_acc=0.5263, goal_acc=0.6420)  train_backprop_loss=1.7883  val_resting_loss=0.0006 (pos_dist=0.0153)  val_position_loss=0.5166 (pos_dist=0.6359)  val_event_loss=1.3652 (oob_acc=0.5091, goal_acc=0.6362)  val_backprop_loss=1.8915
2026-08-22 17:36:20,682 INFO     train pos_rmse  by horizon (m): [1.1276 0.9394 1.1241 1.4347 1.3362 1.244  1.2316 1.3681], mean: 1.2257 m
2026-08-22 17:36:20,682 INFO     train pos_dist  by horizon (m): [1.8091 1.4883 1.6551 2.0249 1.8382 1.7055 1.6695 1.8187], mean: 1.7512 m
2026-08-22 17:36:20,682 INFO     train vel_rmse  by horizon (m/s): [1.0003 0.7736 0.8719 1.6065 1.897  1.689  1.6207 1.5651], mean: 1.3780 m/s
2026-08-22 17:36:20,682 INFO     train vel_dist  by horizon (m/s): [1.3728 1.0529 1.2176 2.2664 2.4592 2.2131 2.1613 2.1413], mean: 1.8606 m/s
2026-08-22 17:36:20,683 INFO     train pos_r2    by horizon: [0.9971 0.9981 0.9975 0.9966 0.9977 0.9983 0.9985 0.9983], mean: 0.9978
2026-08-22 17:36:20,683 INFO     train vel_r2    by horizon: [ 0.9875  0.99    0.9821  0.9026  0.7721  0.5856  0.2463 -0.7026], mean: 0.5955
2026-08-22 17:36:20,683 INFO     train pos_err_pct_disp by horizon: [59.7079 21.6727 14.3971 11.0496  7.4184  5.8501  5.1796  5.3779], mean: 16.3317
2026-08-22 17:36:20,683 INFO     train vel_err_pct_disp by horizon: [13.5717 10.5777 12.5475 20.1415 20.0411 17.5903 15.7993 14.8393], mean: 15.6385
2026-08-22 17:36:20,683 INFO     train pos_err_pct_ballistic by horizon: [315.1571  75.2197  43.6613  25.4782  10.0643   5.3578   3.2108   2.1713], mean: 60.0401
2026-08-22 17:36:20,683 INFO     train vel_err_pct_ballistic by horizon: [12.8201  9.1     9.4182 12.0929  8.7424  5.6505  3.9625  2.7163], mean: 8.0629
2026-08-22 17:36:20,684 INFO     train t0 pos_rmse  by horizon (m): [0.7635 0.502  0.3414 0.2844 0.2136 0.1435 0.0949 0.0755], mean: 0.3024 m
2026-08-22 17:36:20,684 INFO     val   pos_rmse  by horizon (m): [1.2657 0.975  1.1404 1.444  1.3556 1.2823 1.2568 1.4045], mean: 1.2655 m
2026-08-22 17:36:20,684 INFO     val   pos_dist  by horizon (m): [2.05   1.5434 1.6851 2.0474 1.8759 1.7708 1.7112 1.9011], mean: 1.8231 m
2026-08-22 17:36:20,684 INFO     val   vel_rmse  by horizon (m/s): [1.0149 0.7881 0.8876 1.618  1.9088 1.6937 1.6235 1.5876], mean: 1.3903 m/s
2026-08-22 17:36:20,684 INFO     val   vel_dist  by horizon (m/s): [1.3982 1.0798 1.254  2.2895 2.4698 2.2182 2.17   2.1884], mean: 1.8835 m/s
2026-08-22 17:36:20,685 INFO     val   pos_r2    by horizon: [0.9964 0.9979 0.9974 0.9966 0.9976 0.9982 0.9984 0.9982], mean: 0.9976
2026-08-22 17:36:20,685 INFO     val   vel_r2    by horizon: [ 0.9872  0.9896  0.9814  0.9012  0.7691  0.5833  0.2435 -0.7516], mean: 0.5880
2026-08-22 17:36:20,685 INFO     val   pos_err_pct_disp by horizon: [66.8459 22.4883 14.606  11.1228  7.5266  6.0306  5.2865  5.5208], mean: 17.4284
2026-08-22 17:36:20,685 INFO     val   vel_err_pct_disp by horizon: [13.7696 10.7751 12.7723 20.2853 20.1701 17.6399 15.8288 15.0514], mean: 15.7866
2026-08-22 17:36:20,686 INFO     val   pos_err_pct_ballistic by horizon: [352.8334  78.0502  44.2948  25.6468  10.211    5.5232   3.277    2.229 ], mean: 65.2582
2026-08-22 17:36:20,686 INFO     val   vel_err_pct_ballistic by horizon: [13.007   9.2699  9.587  12.1793  8.7987  5.6664  3.9699  2.7551], mean: 8.1542
2026-08-22 17:36:20,686 INFO     val   t0 pos_rmse  by horizon (m): [0.8131 0.5651 0.3916 0.3254 0.2546 0.1836 0.1406 0.119 ], mean: 0.3491 m
2026-08-22 17:37:54,304 INFO decoder-only pretrain epoch 2/15: train_loss=0.0078  val_loss=0.0081  best=0.0076  (patience 1/3)  train_pair_loss=0.0007  val_pair_loss=0.0008  train_t0_loss=0.0001  val_t0_loss=0.0001  train_crossing_loss=0.3435 (pos_dist=0.1207, dt_mae=0.3481)  val_crossing_loss=0.3539 (pos_dist=0.1091, dt_mae=0.3172)  train_resting_loss=0.0006 (pos_dist=0.0153)  train_position_loss=0.4656 (pos_dist=0.6043)  train_event_loss=1.3593 (oob_acc=0.4960, goal_acc=0.6728)  train_backprop_loss=1.6958  val_resting_loss=0.0006 (pos_dist=0.0157)  val_position_loss=0.4218 (pos_dist=0.5760)  val_event_loss=1.3507 (oob_acc=0.4859, goal_acc=0.7264)  val_backprop_loss=1.7824
2026-08-22 17:37:54,304 INFO     train pos_rmse  by horizon (m): [1.382  1.0157 1.1589 1.4502 1.3703 1.3185 1.2827 1.4198], mean: 1.2998 m
2026-08-22 17:37:54,304 INFO     train pos_dist  by horizon (m): [2.2367 1.6052 1.727  2.0716 1.9129 1.8318 1.7526 1.9343], mean: 1.8840 m
2026-08-22 17:37:54,304 INFO     train vel_rmse  by horizon (m/s): [1.0162 0.7855 0.8925 1.6221 1.909  1.6977 1.6308 1.6037], mean: 1.3947 m/s
2026-08-22 17:37:54,304 INFO     train vel_dist  by horizon (m/s): [1.4155 1.0838 1.2643 2.2998 2.4846 2.2341 2.1937 2.2277], mean: 1.9004 m/s
2026-08-22 17:37:54,305 INFO     train pos_r2    by horizon: [0.9957 0.9978 0.9973 0.9966 0.9976 0.9981 0.9984 0.9981], mean: 0.9974
2026-08-22 17:37:54,305 INFO     train vel_r2    by horizon: [ 0.9871  0.9897  0.9812  0.9007  0.7691  0.5813  0.2367 -0.7873], mean: 0.5823
2026-08-22 17:37:54,305 INFO     train pos_err_pct_disp by horizon: [73.0553 23.4349 14.8425 11.1697  7.6076  6.2012  5.3953  5.5808], mean: 18.4109
2026-08-22 17:37:54,305 INFO     train vel_err_pct_disp by horizon: [13.7863 10.7395 12.8428 20.3375 20.1703 17.6822 15.9001 15.2043], mean: 15.8329
2026-08-22 17:37:54,305 INFO     train pos_err_pct_ballistic by horizon: [385.609   81.3358  45.012   25.7549  10.3209   5.6794   3.3445   2.2533], mean: 69.9137
2026-08-22 17:37:54,305 INFO     train vel_err_pct_ballistic by horizon: [13.0228  9.2392  9.6399 12.2106  8.7988  5.68    3.9878  2.7831], mean: 8.1703
2026-08-22 17:37:54,305 INFO     train t0 pos_rmse  by horizon (m): [0.875  0.6284 0.4559 0.3814 0.3087 0.2433 0.198  0.1714], mean: 0.4078 m
2026-08-22 17:37:54,306 INFO     val   pos_rmse  by horizon (m): [1.4634 1.0471 1.1744 1.4622 1.4006 1.3666 1.3245 1.4439], mean: 1.3353 m
2026-08-22 17:37:54,306 INFO     val   pos_dist  by horizon (m): [2.3569 1.6525 1.7622 2.1009 1.9655 1.9027 1.8084 1.9714], mean: 1.9401 m
2026-08-22 17:37:54,306 INFO     val   vel_rmse  by horizon (m/s): [1.0383 0.7876 0.9008 1.6278 1.9243 1.71   1.6398 1.6267], mean: 1.4069 m/s
2026-08-22 17:37:54,306 INFO     val   vel_dist  by horizon (m/s): [1.4487 1.09   1.2718 2.307  2.5046 2.2562 2.2166 2.2702], mean: 1.9206 m/s
2026-08-22 17:37:54,306 INFO     val   pos_r2    by horizon: [0.9952 0.9976 0.9973 0.9965 0.9974 0.9979 0.9983 0.9981], mean: 0.9973
2026-08-22 17:37:54,306 INFO     val   vel_r2    by horizon: [ 0.9866  0.9896  0.9809  0.9     0.7654  0.5752  0.2283 -0.839 ], mean: 0.5734
2026-08-22 17:37:54,306 INFO     val   pos_err_pct_disp by horizon: [77.2847 24.1526 15.0413 11.2631  7.7763  6.4267  5.571   5.6755], mean: 19.1489
2026-08-22 17:37:54,306 INFO     val   vel_err_pct_disp by horizon: [14.0873 10.7689 12.9616 20.408  20.3344 17.8103 15.9867 15.4222], mean: 15.9724
2026-08-22 17:37:54,307 INFO     val   pos_err_pct_ballistic by horizon: [407.9328  83.8265  45.6151  25.9703  10.5497   5.886    3.4533   2.2915], mean: 73.1907
2026-08-22 17:37:54,307 INFO     val   vel_err_pct_ballistic by horizon: [13.3072  9.2645  9.7291 12.253   8.8703  5.7211  4.0095  2.823 ], mean: 8.2472
2026-08-22 17:37:54,307 INFO     val   t0 pos_rmse  by horizon (m): [0.9379 0.6839 0.5168 0.4383 0.3734 0.2996 0.2585 0.2247], mean: 0.4666 m
2026-08-22 17:39:21,018 INFO decoder-only pretrain epoch 3/15: train_loss=0.0083  val_loss=0.0086  best=0.0076  (patience 2/3)  train_pair_loss=0.0008  val_pair_loss=0.0008  train_t0_loss=0.0001  val_t0_loss=0.0002  train_crossing_loss=0.3200 (pos_dist=0.0912, dt_mae=0.3059)  val_crossing_loss=0.3588 (pos_dist=0.0776, dt_mae=0.3158)  train_resting_loss=0.0006 (pos_dist=0.0156)  train_position_loss=0.3839 (pos_dist=0.5497)  train_event_loss=1.3379 (oob_acc=0.4818, goal_acc=0.7958)  train_backprop_loss=1.6265  val_resting_loss=0.0006 (pos_dist=0.0155)  val_position_loss=0.3495 (pos_dist=0.5247)  val_event_loss=1.3243 (oob_acc=0.4802, goal_acc=0.8583)  val_backprop_loss=1.6842
2026-08-22 17:39:21,019 INFO     train pos_rmse  by horizon (m): [1.4836 1.0821 1.1888 1.4795 1.4405 1.4159 1.3693 1.4559], mean: 1.3645 m
2026-08-22 17:39:21,019 INFO     train pos_dist  by horizon (m): [2.3564 1.7068 1.7863 2.1312 2.0271 1.9675 1.8664 1.9899], mean: 1.9789 m
2026-08-22 17:39:21,019 INFO     train vel_rmse  by horizon (m/s): [1.0771 0.7988 0.9149 1.6324 1.9308 1.7232 1.65   1.6259], mean: 1.4191 m/s
2026-08-22 17:39:21,020 INFO     train vel_dist  by horizon (m/s): [1.4889 1.099  1.2798 2.3158 2.5334 2.2886 2.2369 2.2601], mean: 1.9378 m/s
2026-08-22 17:39:21,020 INFO     train pos_r2    by horizon: [0.9951 0.9975 0.9972 0.9964 0.9973 0.9978 0.9981 0.9981], mean: 0.9972
2026-08-22 17:39:21,020 INFO     train vel_r2    by horizon: [ 0.9855  0.9893  0.9803  0.8994  0.7638  0.5686  0.2187 -0.8369], mean: 0.5711
2026-08-22 17:39:21,020 INFO     train pos_err_pct_disp by horizon: [78.3499 24.9634 15.2259 11.396   7.9984  6.6594  5.7599  5.7228], mean: 19.5094
2026-08-22 17:39:21,020 INFO     train vel_err_pct_disp by horizon: [14.616  10.922  13.1651 20.469  20.4008 17.9482 16.0862 15.4138], mean: 16.1276
2026-08-22 17:39:21,021 INFO     train pos_err_pct_ballistic by horizon: [413.5552  86.6406  46.1747  26.2768  10.8511   6.0991   3.5704   2.3106], mean: 74.4348
2026-08-22 17:39:21,021 INFO     train vel_err_pct_ballistic by horizon: [13.8065  9.3962  9.8818 12.2896  8.8993  5.7654  4.0345  2.8214], mean: 8.3619
2026-08-22 17:39:21,021 INFO     train t0 pos_rmse  by horizon (m): [1.011  0.748  0.5764 0.4868 0.4133 0.3375 0.2879 0.2485], mean: 0.5137 m
2026-08-22 17:39:21,021 INFO     val   pos_rmse  by horizon (m): [1.4814 1.121  1.2122 1.5108 1.4991 1.4769 1.4268 1.4874], mean: 1.4020 m
2026-08-22 17:39:21,021 INFO     val   pos_dist  by horizon (m): [2.3138 1.7679 1.816  2.1771 2.1083 2.0416 1.9377 2.0436], mean: 2.0258 m
2026-08-22 17:39:21,021 INFO     val   vel_rmse  by horizon (m/s): [1.1198 0.8101 0.9267 1.6373 1.9503 1.7411 1.6627 1.6334], mean: 1.4352 m/s
2026-08-22 17:39:21,022 INFO     val   vel_dist  by horizon (m/s): [1.5236 1.1084 1.2895 2.3219 2.561  2.3187 2.2583 2.2604], mean: 1.9552 m/s
2026-08-22 17:39:21,022 INFO     val   pos_r2    by horizon: [0.9951 0.9973 0.9971 0.9963 0.9971 0.9976 0.998  0.998 ], mean: 0.9970
2026-08-22 17:39:21,022 INFO     val   vel_r2    by horizon: [ 0.9844  0.989   0.9798  0.8988  0.759   0.5596  0.2066 -0.8541], mean: 0.5654
2026-08-22 17:39:21,022 INFO     val   pos_err_pct_disp by horizon: [78.2386 25.8577 15.5255 11.6375  8.323   6.9452  6.001   5.8462], mean: 19.7969
2026-08-22 17:39:21,022 INFO     val   vel_err_pct_disp by horizon: [15.1931 11.0772 13.3343 20.5269 20.6085 18.1344 16.21   15.4855], mean: 16.3213
2026-08-22 17:39:21,022 INFO     val   pos_err_pct_ballistic by horizon: [412.968   89.7446  47.0833  26.8337  11.2915   6.3609   3.7199   2.3604], mean: 75.0453
2026-08-22 17:39:21,022 INFO     val   vel_err_pct_ballistic by horizon: [14.3517  9.5298 10.0088 12.3244  8.99    5.8253  4.0655  2.8346], mean: 8.4912
2026-08-22 17:39:21,023 INFO     val   t0 pos_rmse  by horizon (m): [1.0749 0.8041 0.6317 0.5324 0.4423 0.3561 0.305  0.2603], mean: 0.5508 m
2026-08-22 17:40:49,373 INFO decoder-only pretrain epoch 4/15: train_loss=0.0088  val_loss=0.0092  best=0.0076  (patience 3/3)  train_pair_loss=0.0008  val_pair_loss=0.0008  train_t0_loss=0.0002  val_t0_loss=0.0002  train_crossing_loss=0.3482 (pos_dist=0.0780, dt_mae=0.3345)  val_crossing_loss=0.4043 (pos_dist=0.0792, dt_mae=0.3644)  train_resting_loss=0.0006 (pos_dist=0.0154)  train_position_loss=0.3167 (pos_dist=0.4996)  train_event_loss=1.3100 (oob_acc=0.4826, goal_acc=0.9011)  train_backprop_loss=1.5612  val_resting_loss=0.0006 (pos_dist=0.0155)  val_position_loss=0.2860 (pos_dist=0.4749)  val_event_loss=1.2961 (oob_acc=0.4856, goal_acc=0.9312)  val_backprop_loss=1.5933
2026-08-22 17:40:49,374 INFO     train pos_rmse  by horizon (m): [1.4769 1.1689 1.2485 1.5473 1.5527 1.5216 1.4635 1.512 ], mean: 1.4364 m
2026-08-22 17:40:49,374 INFO     train pos_dist  by horizon (m): [2.2659 1.8419 1.8623 2.2271 2.183  2.0981 1.9896 2.1022], mean: 2.0713 m
2026-08-22 17:40:49,374 INFO     train vel_rmse  by horizon (m/s): [1.1505 0.8216 0.9346 1.6428 1.9535 1.7514 1.6735 1.6297], mean: 1.4447 m/s
2026-08-22 17:40:49,374 INFO     train vel_dist  by horizon (m/s): [1.5543 1.1218 1.2978 2.3346 2.582  2.3421 2.277  2.2558], mean: 1.9707 m/s
2026-08-22 17:40:49,374 INFO     train pos_r2    by horizon: [0.9951 0.997  0.9969 0.9961 0.9969 0.9974 0.9979 0.9979], mean: 0.9969
2026-08-22 17:40:49,374 INFO     train vel_r2    by horizon: [ 0.9835  0.9887  0.9794  0.8981  0.7582  0.5543  0.1962 -0.8456], mean: 0.5641
2026-08-22 17:40:49,375 INFO     train pos_err_pct_disp by horizon: [77.996  26.9684 15.9924 11.9193  8.6219  7.1559  6.1556  5.9431], mean: 20.0941
2026-08-22 17:40:49,375 INFO     train vel_err_pct_disp by horizon: [15.61   11.2336 13.4484 20.5977 20.6409 18.2422 16.3159 15.4502], mean: 16.4424
2026-08-22 17:40:49,375 INFO     train pos_err_pct_ballistic by horizon: [411.6872  93.5993  48.4994  27.4835  11.6969   6.5538   3.8157   2.3996], mean: 75.7169
2026-08-22 17:40:49,375 INFO     train vel_err_pct_ballistic by horizon: [14.7455  9.6643 10.0944 12.3668  9.0041  5.8599  4.0921  2.8281], mean: 8.5819
2026-08-22 17:40:49,375 INFO     train t0 pos_rmse  by horizon (m): [1.1435 0.8589 0.6788 0.5787 0.4836 0.3914 0.3369 0.2883], mean: 0.5950 m
2026-08-22 17:40:49,375 INFO     val   pos_rmse  by horizon (m): [1.4818 1.2177 1.2916 1.5967 1.6223 1.5828 1.5168 1.5557], mean: 1.4832 m
2026-08-22 17:40:49,375 INFO     val   pos_dist  by horizon (m): [2.2385 1.9178 1.9157 2.2959 2.2778 2.1746 2.062  2.1874], mean: 2.1337 m
2026-08-22 17:40:49,375 INFO     val   vel_rmse  by horizon (m/s): [1.1856 0.8315 0.9432 1.6516 1.9698 1.7658 1.6856 1.64  ], mean: 1.4591 m/s
2026-08-22 17:40:49,375 INFO     val   vel_dist  by horizon (m/s): [1.5843 1.135  1.3114 2.3488 2.6033 2.3639 2.2951 2.2701], mean: 1.9890 m/s
2026-08-22 17:40:49,375 INFO     val   pos_r2    by horizon: [0.9951 0.9968 0.9967 0.9958 0.9966 0.9972 0.9977 0.9978], mean: 0.9967
2026-08-22 17:40:49,376 INFO     val   vel_r2    by horizon: [ 0.9825  0.9884  0.979   0.897   0.7542  0.547   0.1846 -0.869 ], mean: 0.5580
2026-08-22 17:40:49,376 INFO     val   pos_err_pct_disp by horizon: [78.26   28.0891 16.5419 12.2988  9.0071  7.443   6.3792  6.1138], mean: 20.5166
2026-08-22 17:40:49,376 INFO     val   vel_err_pct_disp by horizon: [16.0857 11.3706 13.5714 20.7072 20.8141 18.3922 16.4334 15.5479], mean: 16.6153
2026-08-22 17:40:49,376 INFO     val   pos_err_pct_ballistic by horizon: [413.0811  97.489   50.1656  28.3584  12.2195   6.8167   3.9543   2.4685], mean: 76.8191
2026-08-22 17:40:49,376 INFO     val   vel_err_pct_ballistic by horizon: [15.1948  9.7821 10.1868 12.4326  9.0796  5.9081  4.1216  2.846 ], mean: 8.6939
2026-08-22 17:40:49,376 INFO     val   t0 pos_rmse  by horizon (m): [1.1996 0.9104 0.7222 0.6283 0.5132 0.4129 0.357  0.3047], mean: 0.6310 m
2026-08-22 17:40:49,376 INFO Decoder-only pretrain early stop at epoch 4/15 (val stagnant for 3 epochs, best=0.0076)
2026-08-22 17:40:49,379 INFO Restored best decoder-only-pretrain weights (val_loss=0.0076)
2026-08-22 17:40:49,384 INFO Saved 'after_decoder_pretrain' checkpoint to checkpoints/physics_pretrain/ball_encoder_35.after_decoder_pretrain.pt
2026-08-22 17:42:44,227 INFO Loaded 8760 shard(s) from physics_pretrain_data/ball/
2026-08-22 17:42:44,552 INFO Dataset: 2,190,000 episodes (1,861,500 train / 328,500 val)
2026-08-22 17:42:45,123 INFO pos_weight (max cap: 1.5):
2026-08-22 17:42:45,124 INFO     t= 0.2s  out_of_bounds=1.50  goal_scored=1.50
2026-08-22 17:42:45,124 INFO     t= 0.5s  out_of_bounds=1.50  goal_scored=1.50
2026-08-22 17:42:45,124 INFO     t= 1.0s  out_of_bounds=1.50  goal_scored=1.50
2026-08-22 17:42:45,124 INFO     t= 2.0s  out_of_bounds=1.50  goal_scored=1.50
2026-08-22 17:42:45,124 INFO     t= 3.5s  out_of_bounds=1.50  goal_scored=1.50
2026-08-22 17:42:45,124 INFO     t= 5.0s  out_of_bounds=1.36  goal_scored=1.50
2026-08-22 17:42:45,124 INFO     t= 7.0s  out_of_bounds=1.18  goal_scored=1.50
2026-08-22 17:42:45,124 INFO     t=10.0s  out_of_bounds=1.09  goal_scored=1.50
2026-08-22 17:42:52,395 INFO Main-loop optimizer: Adam (lr=1.00e-05, weight_decay=0.0)
2026-08-22 17:42:54,974 INFO Widened checkpoint from checkpoints/physics_pretrain/ball_encoder_34.midtrain_latest.pt to current config dims (latent_dim: 36->38, decoder_hidden_dim: 20->21); resumed (phase=midtrain_latest)
2026-08-22 17:43:14,168 INFO Adjacent-pair training enabled: 7 start-horizon(s), max_skip=2 (13 (start, skip) combos total), min_start_speed=0.50m/s, 9,478,100/13,030,500 (horizon, row) combos mask-eligible (shares rows/batches with autoencode/t0 -- no separate rows of its own anymore)
2026-08-22 17:43:14,196 INFO Training row-count summary (train split):
    main (per-horizon heads)        : 1,861,500 rows -- own batches
    autoencode/t0 (bottleneck recon): 14,892,000 rows -- own batches (1,861,500 rows x 8 horizons)
    adjacent-pair (dynamics)        : 9,478,100/14,892,000 horizon-pass rows mask-eligible -- shares the horizon pass's own latent, no extra rows/batches
    crossing_head (at each horizon) : 2,910,706/14,892,000 horizon-pass rows mask-eligible -- shares the horizon pass's own latent, no extra rows/batches
    resting_head (at each horizon)  : 2,208,032/14,892,000 horizon-pass rows mask-eligible -- shares the horizon pass's own latent, no extra rows/batches
    crossing_head (at t=0, in main) : 846,156/1,861,500 main rows masked-valid (position term only; delta_t trains on all 1,861,500) -- NO extra rows/batches, shares main's own latent
    resting_head (at t=0, in main)  : 412,334/1,861,500 main rows masked-valid -- NO extra rows/batches, shares main's own latent
2026-08-22 17:43:14,196 INFO LR schedule: cosine warm restarts, T_0=30 epochs, T_mult=1, eta_min=3.30e-06, peak_decay=0.7
2026-08-22 17:43:14,196 INFO Decoder-only-pretrain optimizer: Adam (lr=5.00e-06, weight_decay=0.0)
2026-08-22 17:43:14,197 INFO Decoder-only pretraining: 25 epoch(s), encoder TRUNK frozen (encoder.out latent layer trainable, identity rows gradient-masked), lr=5.00e-06
2026-08-22 17:44:45,947 INFO decoder-only pretrain epoch 1/25: train_loss=0.0083  val_loss=0.0095  best=0.0095  train_pair_loss=0.0008  val_pair_loss=0.0009  train_t0_loss=0.0001  val_t0_loss=0.0002  train_crossing_loss=0.4216 (pos_dist=0.2996, dt_mae=0.3098)  val_crossing_loss=0.4891 (pos_dist=0.3998, dt_mae=0.2450)  train_resting_loss=0.0006 (pos_dist=0.0146)  train_position_loss=0.4056 (pos_dist=0.5566)  train_event_loss=1.3680 (oob_acc=0.5618, goal_acc=0.6251)  train_backprop_loss=1.6804  val_resting_loss=0.0007 (pos_dist=0.0166)  val_position_loss=0.3297 (pos_dist=0.5049)  val_event_loss=1.3477 (oob_acc=0.5794, goal_acc=0.7095)  val_backprop_loss=1.6893
2026-08-22 17:44:45,947 INFO     train pos_rmse  by horizon (m): [1.0456 1.0326 1.2274 1.5331 1.4146 1.308  1.3411 1.6491], mean: 1.3189 m
2026-08-22 17:44:45,948 INFO     train pos_dist  by horizon (m): [1.5621 1.6299 1.8384 2.2021 2.0023 1.8482 1.8605 2.2597], mean: 1.9004 m
2026-08-22 17:44:45,948 INFO     train vel_rmse  by horizon (m/s): [1.0947 0.8569 0.9719 1.6443 1.9016 1.7079 1.6734 1.7532], mean: 1.4505 m/s
2026-08-22 17:44:45,948 INFO     train vel_dist  by horizon (m/s): [1.3816 1.1711 1.3474 2.3421 2.4673 2.279  2.2398 2.4245], mean: 1.9566 m/s
2026-08-22 17:44:45,948 INFO     train pos_r2    by horizon: [0.9975 0.9977 0.997  0.9961 0.9974 0.9981 0.9982 0.9975], mean: 0.9974
2026-08-22 17:44:45,948 INFO     train vel_r2    by horizon: [ 0.985   0.9876  0.9776  0.8979  0.771   0.5762  0.1958 -1.1579], mean: 0.5292
2026-08-22 17:44:45,948 INFO     train pos_err_pct_disp by horizon: [55.2952 23.8436 15.7329 11.8133  7.8572  6.1561  5.6498  6.5298], mean: 16.6097
2026-08-22 17:44:45,948 INFO     train vel_err_pct_disp by horizon: [14.8869 11.7459 14.0161 20.6175 20.0903 17.7878 16.3208 16.7061], mean: 16.5215
2026-08-22 17:44:45,949 INFO     train pos_err_pct_ballistic by horizon: [291.8655  82.754   47.7123  27.2389  10.6596   5.6381   3.5022   2.6364], mean: 59.0009
2026-08-22 17:44:45,949 INFO     train vel_err_pct_ballistic by horizon: [14.0625 10.1051 10.5206 12.3787  8.7639  5.7139  4.0933  3.058 ], mean: 8.5870
2026-08-22 17:44:45,949 INFO     train t0 pos_rmse  by horizon (m): [0.9038 0.6029 0.4829 0.4136 0.3169 0.2584 0.2323 0.2248], mean: 0.4294 m
2026-08-22 17:44:45,949 INFO     val   pos_rmse  by horizon (m): [1.134  1.0728 1.2688 1.5676 1.4567 1.3842 1.4788 1.9927], mean: 1.4194 m
2026-08-22 17:44:45,949 INFO     val   pos_dist  by horizon (m): [1.6998 1.6799 1.8767 2.2507 2.071  1.9774 2.1032 2.83  ], mean: 2.0611 m
2026-08-22 17:44:45,949 INFO     val   vel_rmse  by horizon (m/s): [1.1987 0.9372 1.0417 1.6719 1.9287 1.7458 1.7867 2.0655], mean: 1.5470 m/s
2026-08-22 17:44:45,949 INFO     val   vel_dist  by horizon (m/s): [1.5201 1.2969 1.4535 2.3901 2.5036 2.3567 2.4162 2.8843], mean: 2.1027 m/s
2026-08-22 17:44:45,949 INFO     val   pos_r2    by horizon: [0.9971 0.9975 0.9968 0.996  0.9972 0.9979 0.9978 0.9964], mean: 0.9971
2026-08-22 17:44:45,950 INFO     val   vel_r2    by horizon: [ 0.9821  0.9853  0.9744  0.8945  0.7643  0.5572  0.0838 -1.9649], mean: 0.4096
2026-08-22 17:44:45,950 INFO     val   pos_err_pct_disp by horizon: [59.8941 24.7454 16.2519 12.0748  8.0881  6.5097  6.2191  7.8293], mean: 17.7016
2026-08-22 17:44:45,950 INFO     val   vel_err_pct_disp by horizon: [16.2638 12.8143 14.9912 20.9606 20.3807 18.1826 17.4198 19.5823], mean: 17.5744
2026-08-22 17:44:45,950 INFO     val   pos_err_pct_ballistic by horizon: [316.1397  85.884   49.2863  27.8419  10.9728   5.962    3.8551   3.1611], mean: 62.8879
2026-08-22 17:44:45,950 INFO     val   vel_err_pct_ballistic by horizon: [15.3631 11.0241 11.2525 12.5847  8.8905  5.8407  4.369   3.5845], mean: 9.1136
2026-08-22 17:44:45,950 INFO     val   t0 pos_rmse  by horizon (m): [1.0692 0.7546 0.6081 0.5463 0.4632 0.418  0.3989 0.3805], mean: 0.5799 m
2026-08-22 17:46:13,190 INFO decoder-only pretrain epoch 2/25: train_loss=0.0094  val_loss=0.0092  best=0.0092  train_pair_loss=0.0008  val_pair_loss=0.0008  train_t0_loss=0.0002  val_t0_loss=0.0002  train_crossing_loss=0.4070 (pos_dist=0.3092, dt_mae=0.2573)  val_crossing_loss=0.3959 (pos_dist=0.2034, dt_mae=0.2866)  train_resting_loss=0.0007 (pos_dist=0.0168)  train_position_loss=0.2741 (pos_dist=0.4601)  train_event_loss=1.3250 (oob_acc=0.5984, goal_acc=0.7885)  train_backprop_loss=1.5455  val_resting_loss=0.0007 (pos_dist=0.0167)  val_position_loss=0.2248 (pos_dist=0.4172)  val_event_loss=1.3020 (oob_acc=0.6184, goal_acc=0.8595)  val_backprop_loss=1.5381
2026-08-22 17:46:13,191 INFO     train pos_rmse  by horizon (m): [1.1688 1.0563 1.2438 1.5399 1.4504 1.4005 1.5017 1.9658], mean: 1.4159 m
2026-08-22 17:46:13,191 INFO     train pos_dist  by horizon (m): [1.7737 1.6535 1.8352 2.1964 2.0519 1.9804 2.1361 2.7966], mean: 2.0530 m
2026-08-22 17:46:13,191 INFO     train vel_rmse  by horizon (m/s): [1.1614 0.91   1.0118 1.6666 1.9414 1.7682 1.8285 2.0318], mean: 1.5400 m/s
2026-08-22 17:46:13,191 INFO     train vel_dist  by horizon (m/s): [1.5199 1.2656 1.416  2.3783 2.5332 2.3867 2.474  2.8383], mean: 2.1015 m/s
2026-08-22 17:46:13,192 INFO     train pos_r2    by horizon: [0.9969 0.9976 0.9969 0.9961 0.9973 0.9978 0.9978 0.9965], mean: 0.9971
2026-08-22 17:46:13,192 INFO     train vel_r2    by horizon: [ 0.9832  0.9861  0.9759  0.8951  0.7612  0.5458  0.0405 -1.8706], mean: 0.4146
2026-08-22 17:46:13,192 INFO     train pos_err_pct_disp by horizon: [61.7293 24.3644 15.931  11.8607  8.052   6.5855  6.3152  7.7261], mean: 17.8205
2026-08-22 17:46:13,192 INFO     train vel_err_pct_disp by horizon: [15.7576 12.4435 14.563  20.895  20.5129 18.4159 17.8271 19.2686], mean: 17.4604
2026-08-22 17:46:13,192 INFO     train pos_err_pct_ballistic by horizon: [325.8269  84.5616  48.3132  27.3483  10.9238   6.0314   3.9146   3.1194], mean: 63.7549
2026-08-22 17:46:13,192 INFO     train vel_err_pct_ballistic by horizon: [14.8849 10.7052 10.9311 12.5453  8.9482  5.9157  4.4711  3.5271], mean: 8.9911
2026-08-22 17:46:13,192 INFO     train t0 pos_rmse  by horizon (m): [1.07   0.7402 0.5886 0.539  0.4634 0.4219 0.4025 0.3856], mean: 0.5764 m
2026-08-22 17:46:13,192 INFO     val   pos_rmse  by horizon (m): [1.1976 1.048  1.2213 1.5188 1.4552 1.4119 1.4941 1.8741], mean: 1.4026 m
2026-08-22 17:46:13,193 INFO     val   pos_dist  by horizon (m): [1.8273 1.6441 1.8054 2.1556 2.0537 1.9802 2.123  2.6549], mean: 2.0305 m
2026-08-22 17:46:13,193 INFO     val   vel_rmse  by horizon (m/s): [1.151  0.8891 0.9806 1.6635 1.9597 1.7776 1.8219 1.9249], mean: 1.5210 m/s
2026-08-22 17:46:13,193 INFO     val   vel_dist  by horizon (m/s): [1.5332 1.2347 1.3717 2.3685 2.5594 2.3958 2.4672 2.6858], mean: 2.0770 m/s
2026-08-22 17:46:13,193 INFO     val   pos_r2    by horizon: [0.9968 0.9976 0.997  0.9962 0.9972 0.9978 0.9978 0.9968], mean: 0.9972
2026-08-22 17:46:13,193 INFO     val   vel_r2    by horizon: [ 0.9835  0.9868  0.9773  0.8955  0.7567  0.541   0.0474 -1.5751], mean: 0.4516
2026-08-22 17:46:13,193 INFO     val   pos_err_pct_disp by horizon: [63.2494 24.175  15.6424 11.6993  8.0801  6.6398  6.2833  7.3641], mean: 17.8917
2026-08-22 17:46:13,193 INFO     val   vel_err_pct_disp by horizon: [15.6183 12.1566 14.1113 20.8563 20.708  18.5137 17.7625 18.2498], mean: 17.2471
2026-08-22 17:46:13,193 INFO     val   pos_err_pct_ballistic by horizon: [333.8504  83.9045  47.4378  26.9762  10.9619   6.0811   3.8949   2.9733], mean: 64.5100
2026-08-22 17:46:13,194 INFO     val   vel_err_pct_ballistic by horizon: [14.7533 10.4583 10.592  12.5221  9.0333  5.9471  4.4549  3.3406], mean: 8.8877
2026-08-22 17:46:13,194 INFO     val   t0 pos_rmse  by horizon (m): [1.0362 0.7099 0.5554 0.5164 0.4443 0.4061 0.3867 0.3721], mean: 0.5534 m
2026-08-22 17:47:42,827 INFO decoder-only pretrain epoch 3/25: train_loss=0.0092  val_loss=0.0092  best=0.0092  train_pair_loss=0.0008  val_pair_loss=0.0008  train_t0_loss=0.0002  val_t0_loss=0.0002  train_crossing_loss=0.3663 (pos_dist=0.1797, dt_mae=0.3034)  val_crossing_loss=0.4092 (pos_dist=0.1752, dt_mae=0.3260)  train_resting_loss=0.0007 (pos_dist=0.0169)  train_position_loss=0.1816 (pos_dist=0.3739)  train_event_loss=1.2786 (oob_acc=0.6407, goal_acc=0.9033)  train_backprop_loss=1.4417  val_resting_loss=0.0007 (pos_dist=0.0173)  val_position_loss=0.1428 (pos_dist=0.3315)  val_event_loss=1.2557 (oob_acc=0.6606, goal_acc=0.9340)  val_backprop_loss=1.4098
2026-08-22 17:47:42,828 INFO     train pos_rmse  by horizon (m): [1.2376 1.0639 1.2258 1.5187 1.4632 1.416  1.4711 1.7946], mean: 1.3989 m
2026-08-22 17:47:42,828 INFO     train pos_dist  by horizon (m): [1.8938 1.6682 1.8188 2.1602 2.0611 1.9759 2.0821 2.515 ], mean: 2.0219 m
2026-08-22 17:47:42,828 INFO     train vel_rmse  by horizon (m/s): [1.1985 0.9128 0.9992 1.6746 1.97   1.7819 1.8081 1.8391], mean: 1.5230 m/s
2026-08-22 17:47:42,828 INFO     train vel_dist  by horizon (m/s): [1.5939 1.2524 1.3884 2.3871 2.5849 2.4053 2.4473 2.5508], mean: 2.0763 m/s
2026-08-22 17:47:42,829 INFO     train pos_r2    by horizon: [0.9966 0.9975 0.997  0.9962 0.9972 0.9978 0.9979 0.997 ], mean: 0.9972
2026-08-22 17:47:42,829 INFO     train vel_r2    by horizon: [ 0.9821  0.986   0.9765  0.8941  0.7541  0.5387  0.0617 -1.3521], mean: 0.4802
2026-08-22 17:47:42,829 INFO     train pos_err_pct_disp by horizon: [65.3697 24.5409 15.6992 11.6973  8.1232  6.6583  6.1865  7.054 ], mean: 18.1661
2026-08-22 17:47:42,829 INFO     train vel_err_pct_disp by horizon: [16.2676 12.4821 14.3788 20.9969 20.8152 18.559  17.6286 17.4416], mean: 17.3212
2026-08-22 17:47:42,829 INFO     train pos_err_pct_ballistic by horizon: [345.0416  85.1744  47.6102  26.9715  11.0203   6.0981   3.8349   2.848 ], mean: 66.0749
2026-08-22 17:47:42,829 INFO     train vel_err_pct_ballistic by horizon: [15.3667 10.7383 10.7929 12.6065  9.0801  5.9616  4.4213  3.1926], mean: 9.0200
2026-08-22 17:47:42,830 INFO     train t0 pos_rmse  by horizon (m): [1.0214 0.7062 0.5426 0.5135 0.434  0.3851 0.3621 0.3473], mean: 0.5390 m
2026-08-22 17:47:42,830 INFO     val   pos_rmse  by horizon (m): [1.2781 1.0876 1.2318 1.5247 1.4825 1.4293 1.4623 1.7509], mean: 1.4059 m
2026-08-22 17:47:42,830 INFO     val   pos_dist  by horizon (m): [1.9502 1.706  1.832  2.1722 2.0843 1.9875 2.0647 2.4366], mean: 2.0292 m
2026-08-22 17:47:42,830 INFO     val   vel_rmse  by horizon (m/s): [1.2763 0.9572 1.0298 1.6892 1.9896 1.7904 1.7945 1.7823], mean: 1.5386 m/s
2026-08-22 17:47:42,830 INFO     val   vel_dist  by horizon (m/s): [1.676  1.2992 1.4267 2.4094 2.609  2.42   2.439  2.4718], mean: 2.0939 m/s
2026-08-22 17:47:42,830 INFO     val   pos_r2    by horizon: [0.9963 0.9974 0.997  0.9962 0.9971 0.9977 0.9979 0.9972], mean: 0.9971
2026-08-22 17:47:42,830 INFO     val   vel_r2    by horizon: [ 0.9797  0.9847  0.975   0.8923  0.7492  0.5343  0.0759 -1.2076], mean: 0.4979
2026-08-22 17:47:42,830 INFO     val   pos_err_pct_disp by horizon: [67.5033 25.0865 15.7762 11.7446  8.2316  6.7216  6.1498  6.8805], mean: 18.5118
2026-08-22 17:47:42,831 INFO     val   vel_err_pct_disp by horizon: [17.3192 13.0895 14.8197 21.1778 21.0232 18.647  17.4948 16.8976], mean: 17.5586
2026-08-22 17:47:42,831 INFO     val   pos_err_pct_ballistic by horizon: [356.3036  87.0681  47.8437  27.0807  11.1674   6.156    3.8121   2.778 ], mean: 67.7762
2026-08-22 17:47:42,831 INFO     val   vel_err_pct_ballistic by horizon: [16.36   11.261  11.1238 12.7151  9.1708  5.9899  4.3878  3.0931], mean: 9.2627
2026-08-22 17:47:42,831 INFO     val   t0 pos_rmse  by horizon (m): [1.0082 0.7039 0.5341 0.512  0.4294 0.3726 0.3462 0.3293], mean: 0.5295 m
2026-08-22 17:49:13,908 INFO decoder-only pretrain epoch 4/25: train_loss=0.0093  val_loss=0.0096  best=0.0092  train_pair_loss=0.0008  val_pair_loss=0.0008  train_t0_loss=0.0002  val_t0_loss=0.0002  train_crossing_loss=0.3970 (pos_dist=0.1857, dt_mae=0.3380)  val_crossing_loss=0.4453 (pos_dist=0.1935, dt_mae=0.3552)  train_resting_loss=0.0007 (pos_dist=0.0176)  train_position_loss=0.1099 (pos_dist=0.2896)  train_event_loss=1.2333 (oob_acc=0.6775, goal_acc=0.9449)  train_backprop_loss=1.3500  val_resting_loss=0.0008 (pos_dist=0.0181)  val_position_loss=0.0808 (pos_dist=0.2483)  val_event_loss=1.2113 (oob_acc=0.6909, goal_acc=0.9511)  val_backprop_loss=1.3040
2026-08-22 17:49:13,909 INFO     train pos_rmse  by horizon (m): [1.316  1.1094 1.2455 1.5312 1.4894 1.437  1.4534 1.724 ], mean: 1.4132 m
2026-08-22 17:49:13,909 INFO     train pos_dist  by horizon (m): [2.0031 1.7381 1.8547 2.1842 2.0902 1.9933 2.0493 2.3894], mean: 2.0378 m
2026-08-22 17:49:13,909 INFO     train vel_rmse  by horizon (m/s): [1.3306 0.9973 1.0632 1.7102 2.0007 1.7971 1.7894 1.7505], mean: 1.5549 m/s
2026-08-22 17:49:13,910 INFO     train vel_dist  by horizon (m/s): [1.7378 1.3416 1.4678 2.4426 2.6341 2.4368 2.4328 2.4256], mean: 2.1149 m/s
2026-08-22 17:49:13,910 INFO     train pos_r2    by horizon: [0.9961 0.9973 0.9969 0.9962 0.9971 0.9977 0.9979 0.9973], mean: 0.9971
2026-08-22 17:49:13,910 INFO     train vel_r2    by horizon: [ 0.9779  0.9833  0.9734  0.8896  0.7464  0.5308  0.081  -1.1295], mean: 0.5066
2026-08-22 17:49:13,910 INFO     train pos_err_pct_disp by horizon: [69.5038 25.5912 15.9514 11.7935  8.2682  6.7567  6.1119  6.7743], mean: 18.8439
2026-08-22 17:49:13,910 INFO     train vel_err_pct_disp by horizon: [18.0568 13.6397 15.303  21.4429 21.1407 18.7173 17.4463 16.5958], mean: 17.7928
2026-08-22 17:49:13,910 INFO     train pos_err_pct_ballistic by horizon: [366.8628  88.8194  48.375   27.1934  11.2171   6.1882   3.7887   2.7351], mean: 69.3975
2026-08-22 17:49:13,910 INFO     train vel_err_pct_ballistic by horizon: [17.0568 11.7343 11.4866 12.8743  9.2221  6.0125  4.3756  3.0378], mean: 9.4750
2026-08-22 17:49:13,910 INFO     train t0 pos_rmse  by horizon (m): [1.0105 0.7092 0.534  0.5217 0.4364 0.3743 0.3441 0.3267], mean: 0.5321 m
2026-08-22 17:49:13,911 INFO     val   pos_rmse  by horizon (m): [1.3471 1.1343 1.2628 1.5475 1.5101 1.4557 1.4553 1.7166], mean: 1.4287 m
2026-08-22 17:49:13,911 INFO     val   pos_dist  by horizon (m): [2.0398 1.7755 1.8811 2.2082 2.1133 2.0114 2.0474 2.3737], mean: 2.0563 m
2026-08-22 17:49:13,911 INFO     val   vel_rmse  by horizon (m/s): [1.3968 1.0426 1.1016 1.7325 2.0231 1.8134 1.7879 1.7339], mean: 1.5790 m/s
2026-08-22 17:49:13,911 INFO     val   vel_dist  by horizon (m/s): [1.8056 1.3948 1.5187 2.4754 2.6599 2.4605 2.4361 2.3988], mean: 2.1437 m/s
2026-08-22 17:49:13,911 INFO     val   pos_r2    by horizon: [0.9959 0.9972 0.9968 0.9961 0.997  0.9976 0.9979 0.9973], mean: 0.9970
2026-08-22 17:49:13,911 INFO     val   vel_r2    by horizon: [ 0.9757  0.9818  0.9714  0.8867  0.7407  0.5223  0.0827 -1.0895], mean: 0.5090
2026-08-22 17:49:13,912 INFO     val   pos_err_pct_disp by horizon: [71.1452 26.1636 16.1739 11.9198  8.3846  6.8457  6.1206  6.7456], mean: 19.1874
2026-08-22 17:49:13,913 INFO     val   vel_err_pct_disp by horizon: [18.9536 14.2582 15.8538 21.7211 21.377  18.8865 17.4307 16.4392], mean: 18.1150
2026-08-22 17:49:13,913 INFO     val   pos_err_pct_ballistic by horizon: [375.5267  90.8062  49.0498  27.4847  11.375    6.2697   3.794    2.7235], mean: 70.8787
2026-08-22 17:49:13,913 INFO     val   vel_err_pct_ballistic by horizon: [17.9039 12.2664 11.8999 13.0413  9.3251  6.0669  4.3717  3.0091], mean: 9.7355
2026-08-22 17:49:13,913 INFO     val   t0 pos_rmse  by horizon (m): [1.0086 0.7111 0.5352 0.5303 0.4408 0.3723 0.3408 0.3204], mean: 0.5324 m
2026-08-22 17:50:45,533 INFO decoder-only pretrain epoch 5/25: train_loss=0.0097  val_loss=0.0100  best=0.0092  train_pair_loss=0.0009  val_pair_loss=0.0009  train_t0_loss=0.0002  val_t0_loss=0.0002  train_crossing_loss=0.4211 (pos_dist=0.2023, dt_mae=0.3523)  val_crossing_loss=0.4505 (pos_dist=0.2067, dt_mae=0.3491)  train_resting_loss=0.0008 (pos_dist=0.0185)  train_position_loss=0.0581 (pos_dist=0.2086)  train_event_loss=1.1900 (oob_acc=0.7012, goal_acc=0.9520)  train_backprop_loss=1.2680  val_resting_loss=0.0008 (pos_dist=0.0188)  val_position_loss=0.0386 (pos_dist=0.1700)  val_event_loss=1.1689 (oob_acc=0.7089, goal_acc=0.9525)  val_backprop_loss=1.2198
2026-08-22 17:50:45,533 INFO     train pos_rmse  by horizon (m): [1.37   1.1543 1.2792 1.5603 1.5204 1.4672 1.449  1.7021], mean: 1.4378 m
2026-08-22 17:50:45,533 INFO     train pos_dist  by horizon (m): [2.0649 1.8038 1.9058 2.227  2.1272 2.0232 2.0333 2.3486], mean: 2.0667 m
2026-08-22 17:50:45,533 INFO     train vel_rmse  by horizon (m/s): [1.4322 1.0809 1.1334 1.7572 2.036  1.8246 1.7892 1.725 ], mean: 1.5973 m/s
2026-08-22 17:50:45,534 INFO     train vel_dist  by horizon (m/s): [1.8413 1.4386 1.5616 2.5102 2.6839 2.4829 2.4417 2.394 ], mean: 2.1693 m/s
2026-08-22 17:50:45,534 INFO     train pos_r2    by horizon: [0.9958 0.9971 0.9967 0.996  0.997  0.9976 0.9979 0.9973], mean: 0.9969
2026-08-22 17:50:45,534 INFO     train vel_r2    by horizon: [ 0.9744  0.9804  0.9697  0.8834  0.7374  0.5164  0.0812 -1.0681], mean: 0.5094
2026-08-22 17:50:45,534 INFO     train pos_err_pct_disp by horizon: [72.3539 26.6277 16.3847 12.0178  8.4407  6.8992  6.0937  6.6891], mean: 19.4384
2026-08-22 17:50:45,534 INFO     train vel_err_pct_disp by horizon: [19.4329 14.7817 16.3121 22.0331 21.5105 19.0021 17.4441 16.3548], mean: 18.3589
2026-08-22 17:50:45,534 INFO     train pos_err_pct_ballistic by horizon: [381.9067  92.4169  49.689   27.7106  11.4511   6.3187   3.7774   2.7007], mean: 71.9964
2026-08-22 17:50:45,534 INFO     train vel_err_pct_ballistic by horizon: [18.3567 12.7167 12.244  13.2286  9.3834  6.104   4.3751  2.9937], mean: 9.9253
2026-08-22 17:50:45,535 INFO     train t0 pos_rmse  by horizon (m): [1.0222 0.7225 0.5398 0.54   0.4435 0.3754 0.3407 0.3202], mean: 0.5380 m
2026-08-22 17:50:45,535 INFO     val   pos_rmse  by horizon (m): [1.3876 1.1772 1.2984 1.5828 1.5465 1.492  1.4551 1.7042], mean: 1.4555 m
2026-08-22 17:50:45,535 INFO     val   pos_dist  by horizon (m): [2.0771 1.8381 1.933  2.258  2.159  2.0508 2.0356 2.3444], mean: 2.0870 m
2026-08-22 17:50:45,535 INFO     val   vel_rmse  by horizon (m/s): [1.4755 1.1126 1.1647 1.7808 2.0602 1.8447 1.7923 1.7258], mean: 1.6196 m/s
2026-08-22 17:50:45,535 INFO     val   vel_dist  by horizon (m/s): [1.885  1.4798 1.6023 2.5424 2.7115 2.5092 2.4481 2.3897], mean: 2.1960 m/s
2026-08-22 17:50:45,535 INFO     val   pos_r2    by horizon: [0.9957 0.997  0.9966 0.9959 0.9969 0.9975 0.9979 0.9973], mean: 0.9969
2026-08-22 17:50:45,535 INFO     val   vel_r2    by horizon: [ 0.9728  0.9793  0.968   0.8803  0.7311  0.5056  0.0782 -1.0699], mean: 0.5057
2026-08-22 17:50:45,535 INFO     val   pos_err_pct_disp by horizon: [73.2823 27.1522 16.6295 12.1912  8.5865  7.0161  6.1197  6.6971], mean: 19.7093
2026-08-22 17:50:45,536 INFO     val   vel_err_pct_disp by horizon: [20.0224 15.2161 16.7618 22.3263 21.769  19.2126 17.4734 16.3619], mean: 18.6429
2026-08-22 17:50:45,536 INFO     val   pos_err_pct_ballistic by horizon: [386.8072  94.2373  50.4314  28.1103  11.6489   6.4257   3.7935   2.704 ], mean: 73.0198
2026-08-22 17:50:45,536 INFO     val   vel_err_pct_ballistic by horizon: [18.9135 13.0905 12.5815 13.4047  9.4962  6.1716  4.3824  2.995 ], mean: 10.1294
2026-08-22 17:50:45,536 INFO     val   t0 pos_rmse  by horizon (m): [1.0286 0.7287 0.5454 0.5448 0.4418 0.3724 0.3389 0.3182], mean: 0.5399 m
2026-08-22 17:52:44,439 INFO Loaded 8760 shard(s) from physics_pretrain_data/ball/
2026-08-22 17:52:44,752 INFO Dataset: 2,190,000 episodes (1,861,500 train / 328,500 val)
2026-08-22 17:52:45,330 INFO pos_weight (max cap: 1.5):
2026-08-22 17:52:45,330 INFO     t= 0.2s  out_of_bounds=1.50  goal_scored=1.50
2026-08-22 17:52:45,330 INFO     t= 0.5s  out_of_bounds=1.50  goal_scored=1.50
2026-08-22 17:52:45,330 INFO     t= 1.0s  out_of_bounds=1.50  goal_scored=1.50
2026-08-22 17:52:45,330 INFO     t= 2.0s  out_of_bounds=1.50  goal_scored=1.50
2026-08-22 17:52:45,330 INFO     t= 3.5s  out_of_bounds=1.50  goal_scored=1.50
2026-08-22 17:52:45,330 INFO     t= 5.0s  out_of_bounds=1.36  goal_scored=1.50
2026-08-22 17:52:45,330 INFO     t= 7.0s  out_of_bounds=1.18  goal_scored=1.50
2026-08-22 17:52:45,330 INFO     t=10.0s  out_of_bounds=1.09  goal_scored=1.50
2026-08-22 17:52:52,990 INFO Main-loop optimizer: Adam (lr=1.00e-05, weight_decay=0.0)
2026-08-22 17:52:55,470 INFO Widened checkpoint from checkpoints/physics_pretrain/ball_encoder_34.midtrain_latest.pt to current config dims (latent_dim: 36->38, decoder_hidden_dim: 20->21); resumed (phase=midtrain_latest)
2026-08-22 17:53:16,285 INFO Adjacent-pair training enabled: 7 start-horizon(s), max_skip=2 (13 (start, skip) combos total), min_start_speed=0.50m/s, 9,478,100/13,030,500 (horizon, row) combos mask-eligible (shares rows/batches with autoencode/t0 -- no separate rows of its own anymore)
2026-08-22 17:53:16,310 INFO Training row-count summary (train split):
    main (per-horizon heads)        : 1,861,500 rows -- own batches
    autoencode/t0 (bottleneck recon): 14,892,000 rows -- own batches (1,861,500 rows x 8 horizons)
    adjacent-pair (dynamics)        : 9,478,100/14,892,000 horizon-pass rows mask-eligible -- shares the horizon pass's own latent, no extra rows/batches
    crossing_head (at each horizon) : 2,910,706/14,892,000 horizon-pass rows mask-eligible -- shares the horizon pass's own latent, no extra rows/batches
    resting_head (at each horizon)  : 2,208,032/14,892,000 horizon-pass rows mask-eligible -- shares the horizon pass's own latent, no extra rows/batches
    crossing_head (at t=0, in main) : 846,156/1,861,500 main rows masked-valid (position term only; delta_t trains on all 1,861,500) -- NO extra rows/batches, shares main's own latent
    resting_head (at t=0, in main)  : 412,334/1,861,500 main rows masked-valid -- NO extra rows/batches, shares main's own latent
2026-08-22 17:53:16,310 INFO LR schedule: cosine warm restarts, T_0=30 epochs, T_mult=1, eta_min=3.30e-06, peak_decay=0.7
2026-08-22 17:53:16,311 INFO Decoder-only-pretrain optimizer: Adam (lr=5.00e-06, weight_decay=0.0)
2026-08-22 17:53:16,311 INFO Decoder-only pretraining: 25 epoch(s), encoder TRUNK frozen (encoder.out latent layer trainable, identity rows gradient-masked), lr=5.00e-06
2026-08-22 17:55:23,405 INFO decoder-only pretrain epoch 1/25: train_loss=0.0074  val_loss=0.0075  best=0.0075  train_pair_loss=0.0007  val_pair_loss=0.0007  train_t0_loss=0.0001  val_t0_loss=0.0001  train_crossing_loss=0.2667 (pos_dist=0.1478, dt_mae=0.1609)  val_crossing_loss=0.2983 (pos_dist=0.1002, dt_mae=0.2037)  train_resting_loss=0.0008 (pos_dist=0.0201)  train_position_loss=0.5736 (pos_dist=0.6733)  train_event_loss=1.3972 (oob_acc=0.5855, goal_acc=0.3884)  train_backprop_loss=0.3859  val_resting_loss=0.0008 (pos_dist=0.0188)  val_position_loss=0.4966 (pos_dist=0.6278)  val_event_loss=1.3981 (oob_acc=0.5882, goal_acc=0.3740)  val_backprop_loss=0.5099
2026-08-22 17:55:23,406 INFO     train pos_rmse  by horizon (m): [1.0248 0.9649 1.1706 1.4623 1.3489 1.2513 1.2592 1.407 ], mean: 1.2361 m
2026-08-22 17:55:23,406 INFO     train pos_dist  by horizon (m): [1.5938 1.5323 1.7516 2.0757 1.8749 1.7112 1.7125 1.8597], mean: 1.7639 m
2026-08-22 17:55:23,406 INFO     train vel_rmse  by horizon (m/s): [1.0149 0.7737 0.8732 1.6087 1.8996 1.6993 1.6353 1.5691], mean: 1.3842 m/s
2026-08-22 17:55:23,406 INFO     train vel_dist  by horizon (m/s): [1.3771 1.0453 1.1934 2.274  2.4712 2.2384 2.1874 2.1541], mean: 1.8676 m/s
2026-08-22 17:55:23,406 INFO     train pos_r2    by horizon: [0.9976 0.998  0.9973 0.9965 0.9976 0.9983 0.9984 0.9982], mean: 0.9977
2026-08-22 17:55:23,406 INFO     train vel_r2    by horizon: [ 0.9872  0.99    0.982   0.9023  0.7715  0.5805  0.2327 -0.7112], mean: 0.5919
2026-08-22 17:55:23,406 INFO     train pos_err_pct_disp by horizon: [54.1399 22.2592 14.9934 11.2627  7.4892  5.8844  5.2964  5.5308], mean: 15.8570
2026-08-22 17:55:23,407 INFO     train vel_err_pct_disp by horizon: [13.7691 10.578  12.5662 20.1692 20.0688 17.6977 15.9418 14.877 ], mean: 15.7085
2026-08-22 17:55:23,407 INFO     train pos_err_pct_ballistic by horizon: [285.7672  77.2551  45.4697  25.9694  10.1603   5.3893   3.2831   2.2331], mean: 56.9409
2026-08-22 17:55:23,407 INFO     train vel_err_pct_ballistic by horizon: [13.0066  9.1003  9.4323 12.1095  8.7545  5.685   3.9983  2.7232], mean: 8.1012
2026-08-22 17:55:23,407 INFO     train t0 pos_rmse  by horizon (m): [0.7489 0.48   0.3396 0.2929 0.2287 0.1588 0.1228 0.1183], mean: 0.3112 m
2026-08-22 17:55:23,407 INFO     val   pos_rmse  by horizon (m): [1.0514 0.9654 1.1661 1.4713 1.3758 1.2831 1.2945 1.4314], mean: 1.2549 m
2026-08-22 17:55:23,407 INFO     val   pos_dist  by horizon (m): [1.6363 1.5306 1.7402 2.0909 1.9128 1.7528 1.7643 1.8959], mean: 1.7905 m
2026-08-22 17:55:23,407 INFO     val   vel_rmse  by horizon (m/s): [1.0067 0.7739 0.8698 1.6073 1.9026 1.6947 1.6329 1.5782], mean: 1.3833 m/s
2026-08-22 17:55:23,407 INFO     val   vel_dist  by horizon (m/s): [1.3812 1.0498 1.194  2.2666 2.4593 2.2251 2.1865 2.1693], mean: 1.8665 m/s
2026-08-22 17:55:23,408 INFO     val   pos_r2    by horizon: [0.9975 0.998  0.9973 0.9965 0.9975 0.9982 0.9983 0.9981], mean: 0.9977
2026-08-22 17:55:23,408 INFO     val   vel_r2    by horizon: [ 0.9874  0.99    0.9822  0.9025  0.7706  0.5827  0.2348 -0.7309], mean: 0.5899
2026-08-22 17:55:23,408 INFO     val   pos_err_pct_disp by horizon: [55.527  22.2679 14.934  11.3336  7.6392  6.0346  5.4448  5.6265], mean: 16.1010
2026-08-22 17:55:23,408 INFO     val   vel_err_pct_disp by horizon: [13.6589 10.5811 12.5158 20.1509 20.105  17.6512 15.9202 14.9625], mean: 15.6932
2026-08-22 17:55:23,409 INFO     val   pos_err_pct_ballistic by horizon: [293.0887  77.2854  45.2896  26.1329  10.3638   5.5268   3.3751   2.2717], mean: 57.9168
2026-08-22 17:55:23,409 INFO     val   vel_err_pct_ballistic by horizon: [12.9024  9.1029  9.3944 12.0986  8.7703  5.67    3.9928  2.7388], mean: 8.0838
2026-08-22 17:55:23,409 INFO     val   t0 pos_rmse  by horizon (m): [0.7752 0.5091 0.3852 0.3267 0.2565 0.1745 0.1309 0.1163], mean: 0.3343 m
2026-08-22 17:58:55,048 INFO decoder-only pretrain epoch 2/25: train_loss=0.0075  val_loss=0.0077  best=0.0075  train_pair_loss=0.0007  val_pair_loss=0.0007  train_t0_loss=0.0001  val_t0_loss=0.0001  train_crossing_loss=0.2829 (pos_dist=0.0957, dt_mae=0.2319)  val_crossing_loss=0.3292 (pos_dist=0.0988, dt_mae=0.2593)  train_resting_loss=0.0008 (pos_dist=0.0194)  train_position_loss=0.4289 (pos_dist=0.5832)  train_event_loss=1.3965 (oob_acc=0.5913, goal_acc=0.3764)  train_backprop_loss=0.3036  val_resting_loss=0.0009 (pos_dist=0.0200)  val_position_loss=0.3669 (pos_dist=0.5402)  val_event_loss=1.3945 (oob_acc=0.5936, goal_acc=0.3847)  val_backprop_loss=0.3804
2026-08-22 17:58:55,048 INFO     train pos_rmse  by horizon (m): [1.0449 0.9626 1.1728 1.4769 1.3845 1.3027 1.3092 1.4381], mean: 1.2615 m
2026-08-22 17:58:55,048 INFO     train pos_dist  by horizon (m): [1.6162 1.5238 1.747  2.0947 1.9297 1.7825 1.7871 1.9112], mean: 1.7990 m
2026-08-22 17:58:55,049 INFO     train vel_rmse  by horizon (m/s): [1.0072 0.7786 0.8716 1.6075 1.8988 1.6932 1.6289 1.5697], mean: 1.3819 m/s
2026-08-22 17:58:55,049 INFO     train vel_dist  by horizon (m/s): [1.384  1.0569 1.1956 2.2656 2.4593 2.2231 2.1803 2.1546], mean: 1.8649 m/s
2026-08-22 17:58:55,049 INFO     train pos_r2    by horizon: [0.9976 0.998  0.9973 0.9964 0.9975 0.9981 0.9983 0.9981], mean: 0.9977
2026-08-22 17:58:55,050 INFO     train vel_r2    by horizon: [ 0.9874  0.9898  0.9821  0.9024  0.7716  0.5835  0.2385 -0.7123], mean: 0.5929
2026-08-22 17:58:55,050 INFO     train pos_err_pct_disp by horizon: [55.182  22.2037 15.0202 11.3755  7.6864  6.1261  5.506   5.6527], mean: 16.0941
2026-08-22 17:58:55,050 INFO     train vel_err_pct_disp by horizon: [13.6645 10.6451 12.5423 20.154  20.0625 17.6355 15.8808 14.8816], mean: 15.6833
2026-08-22 17:58:55,050 INFO     train pos_err_pct_ballistic by horizon: [291.2679  77.0626  45.551   26.2295  10.4278   5.6107   3.4131   2.2823], mean: 57.7306
2026-08-22 17:58:55,051 INFO     train vel_err_pct_ballistic by horizon: [12.9077  9.158   9.4143 12.1004  8.7518  5.665   3.983   2.724 ], mean: 8.0880
2026-08-22 17:58:55,051 INFO     train t0 pos_rmse  by horizon (m): [0.7963 0.5328 0.4283 0.3619 0.282  0.2002 0.1493 0.1286], mean: 0.3599 m
2026-08-22 17:58:55,051 INFO     val   pos_rmse  by horizon (m): [1.043  0.9655 1.1861 1.4912 1.4013 1.3246 1.3287 1.4736], mean: 1.2768 m
2026-08-22 17:58:55,051 INFO     val   pos_dist  by horizon (m): [1.6091 1.5278 1.7632 2.1129 1.9544 1.8116 1.8151 1.9655], mean: 1.8199 m
2026-08-22 17:58:55,051 INFO     val   vel_rmse  by horizon (m/s): [1.0115 0.7807 0.8747 1.6152 1.9094 1.6996 1.629  1.5726], mean: 1.3866 m/s
2026-08-22 17:58:55,051 INFO     val   vel_dist  by horizon (m/s): [1.3877 1.0615 1.1986 2.2767 2.4682 2.2288 2.1766 2.1531], mean: 1.8689 m/s
2026-08-22 17:58:55,052 INFO     val   pos_r2    by horizon: [0.9976 0.998  0.9972 0.9964 0.9974 0.998  0.9982 0.998 ], mean: 0.9976
2026-08-22 17:58:55,052 INFO     val   vel_r2    by horizon: [ 0.9872  0.9898  0.982   0.9015  0.769   0.5803  0.2385 -0.7187], mean: 0.5912
2026-08-22 17:58:55,052 INFO     val   pos_err_pct_disp by horizon: [55.0835 22.2695 15.1907 11.4868  7.7807  6.2294  5.5883  5.7919], mean: 16.1776
2026-08-22 17:58:55,052 INFO     val   vel_err_pct_disp by horizon: [13.7247 10.6743 12.5856 20.2498 20.1761 17.7024 15.8816 14.9095], mean: 15.7380
2026-08-22 17:58:55,053 INFO     val   pos_err_pct_ballistic by horizon: [290.7482  77.2909  46.068   26.4862  10.5557   5.7053   3.4641   2.3385], mean: 57.8321
2026-08-22 17:58:55,053 INFO     val   vel_err_pct_ballistic by horizon: [12.9646  9.1831  9.4469 12.158   8.8013  5.6865  3.9832  2.7291], mean: 8.1191
2026-08-22 17:58:55,053 INFO     val   t0 pos_rmse  by horizon (m): [0.8106 0.5471 0.4612 0.3959 0.31   0.2221 0.172  0.1476], mean: 0.3833 m
2026-08-22 18:00:40,430 INFO Loaded 8760 shard(s) from physics_pretrain_data/ball/
2026-08-22 18:00:40,821 INFO Dataset: 2,190,000 episodes (1,861,500 train / 328,500 val)
2026-08-22 18:00:41,603 INFO pos_weight (max cap: 1.5):
2026-08-22 18:00:41,603 INFO     t= 0.2s  out_of_bounds=1.50  goal_scored=1.50
2026-08-22 18:00:41,603 INFO     t= 0.5s  out_of_bounds=1.50  goal_scored=1.50
2026-08-22 18:00:41,603 INFO     t= 1.0s  out_of_bounds=1.50  goal_scored=1.50
2026-08-22 18:00:41,603 INFO     t= 2.0s  out_of_bounds=1.50  goal_scored=1.50
2026-08-22 18:00:41,603 INFO     t= 3.5s  out_of_bounds=1.50  goal_scored=1.50
2026-08-22 18:00:41,603 INFO     t= 5.0s  out_of_bounds=1.36  goal_scored=1.50
2026-08-22 18:00:41,603 INFO     t= 7.0s  out_of_bounds=1.18  goal_scored=1.50
2026-08-22 18:00:41,603 INFO     t=10.0s  out_of_bounds=1.09  goal_scored=1.50
2026-08-22 18:00:49,480 INFO Main-loop optimizer: Adam (lr=1.00e-05, weight_decay=0.0)
2026-08-22 18:00:52,056 INFO Widened checkpoint from checkpoints/physics_pretrain/ball_encoder_34.midtrain_latest.pt to current config dims (latent_dim: 36->38, decoder_hidden_dim: 20->21); resumed (phase=midtrain_latest)
2026-08-22 18:01:11,861 INFO Adjacent-pair training enabled: 7 start-horizon(s), max_skip=2 (13 (start, skip) combos total), min_start_speed=0.50m/s, 9,478,100/13,030,500 (horizon, row) combos mask-eligible (shares rows/batches with autoencode/t0 -- no separate rows of its own anymore)
2026-08-22 18:01:11,885 INFO Training row-count summary (train split):
    main (per-horizon heads)        : 1,861,500 rows -- own batches
    autoencode/t0 (bottleneck recon): 14,892,000 rows -- own batches (1,861,500 rows x 8 horizons)
    adjacent-pair (dynamics)        : 9,478,100/14,892,000 horizon-pass rows mask-eligible -- shares the horizon pass's own latent, no extra rows/batches
    crossing_head (at each horizon) : 2,910,706/14,892,000 horizon-pass rows mask-eligible -- shares the horizon pass's own latent, no extra rows/batches
    resting_head (at each horizon)  : 2,208,032/14,892,000 horizon-pass rows mask-eligible -- shares the horizon pass's own latent, no extra rows/batches
    crossing_head (at t=0, in main) : 846,156/1,861,500 main rows masked-valid (position term only; delta_t trains on all 1,861,500) -- NO extra rows/batches, shares main's own latent
    resting_head (at t=0, in main)  : 412,334/1,861,500 main rows masked-valid -- NO extra rows/batches, shares main's own latent
2026-08-22 18:01:11,885 INFO LR schedule: cosine warm restarts, T_0=30 epochs, T_mult=1, eta_min=3.30e-06, peak_decay=0.7
2026-08-22 18:01:11,886 INFO Decoder-only-pretrain optimizer: Adam (lr=5.00e-05, weight_decay=0.0)
2026-08-22 18:01:11,886 INFO Decoder-only pretraining: 25 epoch(s), encoder TRUNK frozen (encoder.out latent layer trainable, identity rows gradient-masked), lr=5.00e-05
2026-08-22 18:02:37,921 INFO decoder-only pretrain epoch 1/25: train_loss=0.0081  val_loss=0.0075  best=0.0075  train_pair_loss=0.0008  val_pair_loss=0.0007  train_t0_loss=0.0001  val_t0_loss=0.0001  train_crossing_loss=0.2397 (pos_dist=0.0496, dt_mae=0.1484)  val_crossing_loss=0.2634 (pos_dist=0.0366, dt_mae=0.1215)  train_resting_loss=0.0005 (pos_dist=0.0135)  train_position_loss=0.0932 (pos_dist=0.2147)  train_event_loss=1.3287 (oob_acc=0.4420, goal_acc=0.8196)  train_backprop_loss=0.0353  val_resting_loss=0.0006 (pos_dist=0.0135)  val_position_loss=0.0019 (pos_dist=0.0350)  val_event_loss=1.2952 (oob_acc=0.4651, goal_acc=0.8975)  val_backprop_loss=0.0133
2026-08-22 18:02:37,922 INFO     train pos_rmse  by horizon (m): [1.1148 1.0573 1.2698 1.5226 1.4046 1.3133 1.3023 1.5055], mean: 1.3113 m
2026-08-22 18:02:37,922 INFO     train pos_dist  by horizon (m): [1.7232 1.6672 1.9132 2.169  1.946  1.8187 1.7878 2.0475], mean: 1.8841 m
2026-08-22 18:02:37,922 INFO     train vel_rmse  by horizon (m/s): [1.1405 0.8752 0.9495 1.639  1.9268 1.7258 1.6584 1.6065], mean: 1.4402 m/s
2026-08-22 18:02:37,922 INFO     train vel_dist  by horizon (m/s): [1.5187 1.1928 1.2976 2.3217 2.509  2.2828 2.2229 2.2083], mean: 1.9442 m/s
2026-08-22 18:02:37,922 INFO     train pos_r2    by horizon: [0.9972 0.9976 0.9968 0.9962 0.9974 0.9981 0.9983 0.9979], mean: 0.9974
2026-08-22 18:02:37,922 INFO     train vel_r2    by horizon: [ 0.9837  0.9871  0.9787  0.8986  0.7649  0.5673  0.2108 -0.7961], mean: 0.5744
2026-08-22 18:02:37,923 INFO     train pos_err_pct_disp by horizon: [59.3525 24.4368 16.3207 11.7378  7.8051  6.1787  5.4805  5.9369], mean: 17.1561
2026-08-22 18:02:37,923 INFO     train vel_err_pct_disp by horizon: [15.4986 11.9972 13.6825 20.5505 20.3563 17.9738 16.1675 15.2414], mean: 16.4335
2026-08-22 18:02:37,923 INFO     train pos_err_pct_ballistic by horizon: [313.2813  84.8131  49.495   27.0649  10.5888   5.6588   3.3972   2.397 ], mean: 62.0870
2026-08-22 18:02:37,923 INFO     train vel_err_pct_ballistic by horizon: [14.6403 10.3213 10.2702 12.3385  8.8799  5.7737  4.0549  2.7899], mean: 8.6336
2026-08-22 18:02:37,923 INFO     train t0 pos_rmse  by horizon (m): [0.9159 0.6472 0.4903 0.4187 0.3303 0.2793 0.2587 0.2539], mean: 0.4493 m
2026-08-22 18:02:37,924 INFO     val   pos_rmse  by horizon (m): [1.0059 0.9923 1.1598 1.4399 1.3394 1.2671 1.2493 1.3937], mean: 1.2309 m
2026-08-22 18:02:37,924 INFO     val   pos_dist  by horizon (m): [1.5441 1.5657 1.7215 2.0349 1.8478 1.7418 1.6906 1.83  ], mean: 1.7470 m
2026-08-22 18:02:37,924 INFO     val   vel_rmse  by horizon (m/s): [1.0349 0.7901 0.8795 1.6171 1.9203 1.7201 1.6427 1.5628], mean: 1.3959 m/s
2026-08-22 18:02:37,924 INFO     val   vel_dist  by horizon (m/s): [1.4056 1.0716 1.2053 2.282  2.4884 2.2636 2.2003 2.1307], mean: 1.8809 m/s
2026-08-22 18:02:37,925 INFO     val   pos_r2    by horizon: [0.9977 0.9979 0.9973 0.9966 0.9977 0.9982 0.9985 0.9982], mean: 0.9978
2026-08-22 18:02:37,925 INFO     val   vel_r2    by horizon: [ 0.9866  0.9895  0.9818  0.9013  0.7664  0.5701  0.2255 -0.6974], mean: 0.5905
2026-08-22 18:02:37,925 INFO     val   pos_err_pct_disp by horizon: [53.1262 22.8875 14.8543 11.0913  7.4371  5.9593  5.2552  5.4786], mean: 15.7612
2026-08-22 18:02:37,925 INFO     val   vel_err_pct_disp by horizon: [14.041  10.803  12.6563 20.2733 20.2915 17.9154 16.0163 14.8166], mean: 15.8517
2026-08-22 18:02:37,925 INFO     val   pos_err_pct_ballistic by horizon: [280.4167  79.4357  45.0478  25.5741  10.0896   5.4579   3.2576   2.212 ], mean: 56.4364
2026-08-22 18:02:37,925 INFO     val   vel_err_pct_ballistic by horizon: [13.2634  9.2939  9.4999 12.1721  8.8516  5.7549  4.0169  2.7121], mean: 8.1956
2026-08-22 18:02:37,926 INFO     val   t0 pos_rmse  by horizon (m): [0.7994 0.5343 0.3999 0.3487 0.268  0.202  0.1755 0.1668], mean: 0.3618 m
2026-08-22 18:04:15,752 INFO decoder-only pretrain epoch 2/25: train_loss=0.0073  val_loss=0.0072  best=0.0072  train_pair_loss=0.0007  val_pair_loss=0.0007  train_t0_loss=0.0001  val_t0_loss=0.0001  train_crossing_loss=0.2317 (pos_dist=0.0331, dt_mae=0.1161)  val_crossing_loss=0.2625 (pos_dist=0.0306, dt_mae=0.1190)  train_resting_loss=0.0005 (pos_dist=0.0133)  train_position_loss=0.0008 (pos_dist=0.0224)  train_event_loss=1.2596 (oob_acc=0.5073, goal_acc=0.9289)  train_backprop_loss=0.0137  val_resting_loss=0.0005 (pos_dist=0.0131)  val_position_loss=0.0004 (pos_dist=0.0151)  val_event_loss=1.2254 (oob_acc=0.5500, goal_acc=0.9469)  val_backprop_loss=0.0124
2026-08-22 18:04:15,752 INFO     train pos_rmse  by horizon (m): [0.9871 0.964  1.1421 1.4288 1.3277 1.2412 1.233  1.367 ], mean: 1.2114 m
2026-08-22 18:04:15,753 INFO     train pos_dist  by horizon (m): [1.5286 1.5236 1.6929 2.0132 1.8301 1.7024 1.6665 1.7931], mean: 1.7188 m
2026-08-22 18:04:15,753 INFO     train vel_rmse  by horizon (m/s): [1.0026 0.7725 0.8646 1.6077 1.9044 1.7023 1.624  1.549 ], mean: 1.3784 m/s
2026-08-22 18:04:15,753 INFO     train vel_dist  by horizon (m/s): [1.3538 1.0521 1.1895 2.265  2.4734 2.2348 2.1636 2.1062], mean: 1.8548 m/s
2026-08-22 18:04:15,753 INFO     train pos_r2    by horizon: [0.9978 0.998  0.9974 0.9967 0.9977 0.9983 0.9985 0.9983], mean: 0.9978
2026-08-22 18:04:15,753 INFO     train vel_r2    by horizon: [ 0.9875  0.99    0.9824  0.9024  0.7702  0.579   0.2431 -0.6675], mean: 0.5984
2026-08-22 18:04:15,754 INFO     train pos_err_pct_disp by horizon: [52.1317 22.237  14.6276 11.0047  7.3708  5.8367  5.1859  5.3737], mean: 15.4710
2026-08-22 18:04:15,754 INFO     train vel_err_pct_disp by horizon: [13.603  10.5623 12.4422 20.1568 20.122  17.7299 15.8331 14.6858], mean: 15.6419
2026-08-22 18:04:15,754 INFO     train pos_err_pct_ballistic by horizon: [275.1673  77.178   44.3603  25.3745   9.9996   5.3456   3.2146   2.1697], mean: 55.3512
2026-08-22 18:04:15,755 INFO     train vel_err_pct_ballistic by horizon: [12.8497  9.0868  9.3392 12.1021  8.7777  5.6953  3.971   2.6882], mean: 8.0638
2026-08-22 18:04:15,755 INFO     train t0 pos_rmse  by horizon (m): [0.7839 0.5141 0.3693 0.317  0.2438 0.1773 0.1416 0.1282], mean: 0.3344 m
2026-08-22 18:04:15,755 INFO     val   pos_rmse  by horizon (m): [0.9763 0.957  1.1252 1.422  1.331  1.2291 1.2241 1.3611], mean: 1.2032 m
2026-08-22 18:04:15,755 INFO     val   pos_dist  by horizon (m): [1.5076 1.5122 1.6667 2.0027 1.8305 1.6772 1.6385 1.7694], mean: 1.7006 m
2026-08-22 18:04:15,756 INFO     val   vel_rmse  by horizon (m/s): [0.988  0.7624 0.858  1.6062 1.9037 1.6949 1.6174 1.5519], mean: 1.3728 m/s
2026-08-22 18:04:15,756 INFO     val   vel_dist  by horizon (m/s): [1.3274 1.0382 1.1839 2.2593 2.463  2.216  2.1479 2.1024], mean: 1.8423 m/s
2026-08-22 18:04:15,756 INFO     val   pos_r2    by horizon: [0.9979 0.998  0.9975 0.9967 0.9977 0.9983 0.9985 0.9983], mean: 0.9979
2026-08-22 18:04:15,756 INFO     val   vel_r2    by horizon: [ 0.9878  0.9903  0.9827  0.9026  0.7704  0.5827  0.2493 -0.6739], mean: 0.5990
2026-08-22 18:04:15,757 INFO     val   pos_err_pct_disp by horizon: [51.5621 22.0743 14.4106 10.9533  7.39    5.7808  5.1496  5.3508], mean: 15.3339
2026-08-22 18:04:15,757 INFO     val   vel_err_pct_disp by horizon: [13.4039 10.4236 12.346  20.1371 20.1161 17.6526 15.7686 14.7141], mean: 15.5703
2026-08-22 18:04:15,757 INFO     val   pos_err_pct_ballistic by horizon: [272.1608  76.6136  43.7022  25.2561  10.0257   5.2944   3.1921   2.1604], mean: 54.8007
2026-08-22 18:04:15,758 INFO     val   vel_err_pct_ballistic by horizon: [12.6615  8.9674  9.267  12.0903  8.7751  5.6705  3.9548  2.6934], mean: 8.0100
2026-08-22 18:04:15,758 INFO     val   t0 pos_rmse  by horizon (m): [0.7684 0.4902 0.3423 0.2904 0.2245 0.1584 0.1186 0.1024], mean: 0.3119 m
2026-08-22 18:05:47,530 INFO decoder-only pretrain epoch 3/25: train_loss=0.0072  val_loss=0.0072  best=0.0072  train_pair_loss=0.0006  val_pair_loss=0.0006  train_t0_loss=0.0001  val_t0_loss=0.0001  train_crossing_loss=0.2312 (pos_dist=0.0301, dt_mae=0.1151)  val_crossing_loss=0.2624 (pos_dist=0.0294, dt_mae=0.1205)  train_resting_loss=0.0005 (pos_dist=0.0131)  train_position_loss=0.0002 (pos_dist=0.0115)  train_event_loss=1.1938 (oob_acc=0.5904, goal_acc=0.9509)  train_backprop_loss=0.0130  val_resting_loss=0.0005 (pos_dist=0.0130)  val_position_loss=0.0001 (pos_dist=0.0086)  val_event_loss=1.1628 (oob_acc=0.6293, goal_acc=0.9524)  val_backprop_loss=0.0121
2026-08-22 18:05:47,530 INFO     train pos_rmse  by horizon (m): [0.976  0.9452 1.1232 1.4186 1.3234 1.2224 1.2237 1.347 ], mean: 1.1974 m
2026-08-22 18:05:47,530 INFO     train pos_dist  by horizon (m): [1.511  1.4948 1.6686 1.9983 1.8196 1.6693 1.6489 1.7529], mean: 1.6954 m
2026-08-22 18:05:47,531 INFO     train vel_rmse  by horizon (m/s): [0.9853 0.7543 0.8515 1.6027 1.896  1.692  1.6176 1.5436], mean: 1.3679 m/s
2026-08-22 18:05:47,531 INFO     train vel_dist  by horizon (m/s): [1.3287 1.0251 1.171  2.2556 2.4595 2.2159 2.1515 2.0928], mean: 1.8375 m/s
2026-08-22 18:05:47,531 INFO     train pos_r2    by horizon: [0.9979 0.9981 0.9975 0.9967 0.9977 0.9983 0.9985 0.9983], mean: 0.9979
2026-08-22 18:05:47,531 INFO     train vel_r2    by horizon: [ 0.9879  0.9905  0.9829  0.903   0.7723  0.584   0.249  -0.6557], mean: 0.6017
2026-08-22 18:05:47,531 INFO     train pos_err_pct_disp by horizon: [51.5467 21.8024 14.3849 10.9269  7.3472  5.7478  5.1466  5.2955], mean: 15.2748
2026-08-22 18:05:47,532 INFO     train vel_err_pct_disp by horizon: [13.3679 10.3131 12.2524 20.0962 20.0333 17.6235 15.7709 14.6338], mean: 15.5114
2026-08-22 18:05:47,532 INFO     train pos_err_pct_ballistic by horizon: [272.0798  75.6698  43.6244  25.1952   9.9676   5.2642   3.1902   2.1381], mean: 54.6412
2026-08-22 18:05:47,532 INFO     train vel_err_pct_ballistic by horizon: [12.6275  8.8724  9.1967 12.0657  8.739   5.6611  3.9554  2.6787], mean: 7.9746
2026-08-22 18:05:47,532 INFO     train t0 pos_rmse  by horizon (m): [0.7687 0.4908 0.3414 0.286  0.2144 0.1483 0.1052 0.0883], mean: 0.3054 m
2026-08-22 18:05:47,532 INFO     val   pos_rmse  by horizon (m): [0.9767 0.9387 1.1225 1.4165 1.323  1.2264 1.2331 1.3453], mean: 1.1978 m
2026-08-22 18:05:47,532 INFO     val   pos_dist  by horizon (m): [1.5178 1.487  1.6662 1.9867 1.823  1.6737 1.6678 1.7407], mean: 1.6954 m
2026-08-22 18:05:47,532 INFO     val   vel_rmse  by horizon (m/s): [0.9809 0.7507 0.8498 1.6042 1.9037 1.6955 1.6194 1.5497], mean: 1.3692 m/s
2026-08-22 18:05:47,532 INFO     val   vel_dist  by horizon (m/s): [1.3169 1.0232 1.1672 2.2564 2.4699 2.2168 2.1472 2.0931], mean: 1.8363 m/s
2026-08-22 18:05:47,533 INFO     val   pos_r2    by horizon: [0.9979 0.9981 0.9975 0.9967 0.9977 0.9983 0.9985 0.9983], mean: 0.9979
2026-08-22 18:05:47,533 INFO     val   vel_r2    by horizon: [ 0.988   0.9906  0.983   0.9029  0.7704  0.5824  0.2474 -0.6691], mean: 0.5994
2026-08-22 18:05:47,533 INFO     val   pos_err_pct_disp by horizon: [51.5855 21.6523 14.3759 10.9114  7.3457  5.768   5.1873  5.2886], mean: 15.2643
2026-08-22 18:05:47,533 INFO     val   vel_err_pct_disp by horizon: [13.3082 10.2641 12.2284 20.1119 20.1166 17.6588 15.7881 14.6928], mean: 15.5211
2026-08-22 18:05:47,533 INFO     val   pos_err_pct_ballistic by horizon: [272.2845  75.1488  43.597   25.1595   9.9655   5.2827   3.2155   2.1353], mean: 54.5986
2026-08-22 18:05:47,533 INFO     val   vel_err_pct_ballistic by horizon: [12.5712  8.8302  9.1787 12.0751  8.7754  5.6725  3.9597  2.6895], mean: 7.9690
2026-08-22 18:05:47,533 INFO     val   t0 pos_rmse  by horizon (m): [0.7569 0.4797 0.331  0.2752 0.2057 0.1386 0.095  0.0776], mean: 0.2950 m
2026-08-22 18:07:20,696 INFO decoder-only pretrain epoch 4/25: train_loss=0.0071  val_loss=0.0071  best=0.0071  train_pair_loss=0.0006  val_pair_loss=0.0006  train_t0_loss=0.0001  val_t0_loss=0.0001  train_crossing_loss=0.2311 (pos_dist=0.0294, dt_mae=0.1150)  val_crossing_loss=0.2622 (pos_dist=0.0290, dt_mae=0.1183)  train_resting_loss=0.0005 (pos_dist=0.0130)  train_position_loss=0.0001 (pos_dist=0.0067)  train_event_loss=1.1323 (oob_acc=0.6652, goal_acc=0.9524)  train_backprop_loss=0.0127  val_resting_loss=0.0005 (pos_dist=0.0130)  val_position_loss=0.0000 (pos_dist=0.0052)  val_event_loss=1.1020 (oob_acc=0.6988, goal_acc=0.9525)  val_backprop_loss=0.0118
2026-08-22 18:07:20,696 INFO     train pos_rmse  by horizon (m): [0.9727 0.9373 1.1129 1.4138 1.3199 1.2152 1.2184 1.3356], mean: 1.1907 m
2026-08-22 18:07:20,697 INFO     train pos_dist  by horizon (m): [1.5054 1.4832 1.6558 1.9916 1.811  1.657  1.6394 1.7327], mean: 1.6845 m
2026-08-22 18:07:20,697 INFO     train vel_rmse  by horizon (m/s): [0.9789 0.7485 0.8478 1.599  1.8913 1.6895 1.6168 1.5402], mean: 1.3640 m/s
2026-08-22 18:07:20,697 INFO     train vel_dist  by horizon (m/s): [1.3169 1.0174 1.1656 2.2496 2.4513 2.2122 2.1482 2.0833], mean: 1.8306 m/s
2026-08-22 18:07:20,697 INFO     train pos_r2    by horizon: [0.9979 0.9981 0.9975 0.9967 0.9977 0.9984 0.9985 0.9984], mean: 0.9979
2026-08-22 18:07:20,698 INFO     train vel_r2    by horizon: [ 0.9881  0.9906  0.9831  0.9035  0.7734  0.5853  0.2498 -0.6484], mean: 0.6032
2026-08-22 18:07:20,698 INFO     train pos_err_pct_disp by horizon: [51.3693 21.6196 14.2531 10.8894  7.3273  5.714   5.1247  5.2507], mean: 15.1935
2026-08-22 18:07:20,699 INFO     train vel_err_pct_disp by horizon: [13.281  10.2348 12.1986 20.0484 19.9849 17.5976 15.7632 14.6014], mean: 15.4637
2026-08-22 18:07:20,699 INFO     train pos_err_pct_ballistic by horizon: [271.1434  75.0355  43.2247  25.1086   9.9407   5.2332   3.1767   2.12  ], mean: 54.3728
2026-08-22 18:07:20,699 INFO     train vel_err_pct_ballistic by horizon: [12.5455  8.805   9.1564 12.037   8.7179  5.6528  3.9535  2.6727], mean: 7.9426
2026-08-22 18:07:20,699 INFO     train t0 pos_rmse  by horizon (m): [0.7583 0.4801 0.3293 0.2749 0.2028 0.1366 0.0896 0.0711], mean: 0.2928 m
2026-08-22 18:07:20,699 INFO     val   pos_rmse  by horizon (m): [0.9513 0.9474 1.1067 1.4145 1.3281 1.218  1.2245 1.3438], mean: 1.1918 m
2026-08-22 18:07:20,699 INFO     val   pos_dist  by horizon (m): [1.4586 1.4998 1.6402 1.9893 1.8179 1.657  1.6474 1.7418], mean: 1.6815 m
2026-08-22 18:07:20,699 INFO     val   vel_rmse  by horizon (m/s): [0.9791 0.7447 0.8478 1.6024 1.8967 1.6923 1.6198 1.5439], mean: 1.3658 m/s
2026-08-22 18:07:20,699 INFO     val   vel_dist  by horizon (m/s): [1.3077 1.0126 1.1701 2.2553 2.4473 2.213  2.1543 2.0872], mean: 1.8309 m/s
2026-08-22 18:07:20,699 INFO     val   pos_r2    by horizon: [0.998  0.9981 0.9976 0.9967 0.9977 0.9983 0.9985 0.9983], mean: 0.9979
2026-08-22 18:07:20,700 INFO     val   vel_r2    by horizon: [ 0.988   0.9907  0.9831  0.9031  0.7721  0.5839  0.247  -0.6565], mean: 0.6014
2026-08-22 18:07:20,700 INFO     val   pos_err_pct_disp by horizon: [50.2404 21.8534 14.174  10.8954  7.3745  5.7286  5.1511  5.2829], mean: 15.0876
2026-08-22 18:07:20,700 INFO     val   vel_err_pct_disp by horizon: [13.2841 10.1818 12.1994 20.0902 20.0417 17.6263 15.792  14.6373], mean: 15.4816
2026-08-22 18:07:20,700 INFO     val   pos_err_pct_ballistic by horizon: [265.1847  75.8469  42.9847  25.1226  10.0046   5.2466   3.1931   2.133 ], mean: 53.7145
2026-08-22 18:07:20,700 INFO     val   vel_err_pct_ballistic by horizon: [12.5485  8.7594  9.1569 12.0621  8.7427  5.662   3.9607  2.6793], mean: 7.9465
2026-08-22 18:07:20,700 INFO     val   t0 pos_rmse  by horizon (m): [0.7547 0.4783 0.3256 0.2738 0.204  0.1383 0.0967 0.0809], mean: 0.2941 m
