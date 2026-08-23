2026-08-23 11:51:32,073 INFO Loaded 1340 shard(s) from physics_pretrain_data/player
2026-08-23 11:51:32,208 INFO Dataset: 1,340,000 episodes (1,139,000 train / 201,000 val)
2026-08-23 11:51:32,447 INFO pos_weight (max cap: 1.0):
2026-08-23 11:51:32,447 INFO     t= 0.2s  out_of_bounds=1.00  goal_scored=1.00
2026-08-23 11:51:32,447 INFO     t= 1.0s  out_of_bounds=1.00  goal_scored=1.00
2026-08-23 11:51:32,447 INFO     t= 3.0s  out_of_bounds=1.00  goal_scored=1.00
2026-08-23 11:51:32,447 INFO     t= 5.0s  out_of_bounds=1.00  goal_scored=1.00
2026-08-23 11:51:32,447 INFO     t=10.0s  out_of_bounds=1.00  goal_scored=1.00
2026-08-23 11:51:36,730 INFO Widened checkpoint from checkpoints/physics_pretrain/player_encoder_13.midtrain_latest.pt to current config dims (hidden_dim: 192->256, encoder_bottleneck_dim: 96->128); resumed (phase=midtrain_latest)
2026-08-23 11:51:36,972 INFO Latent diagnostics (50,000 rows, latent_dim=36):
    per-dim std: mean=0.3684  pooled=0.6442  |  latent norm: mean=3.8604 std=0.4021
    dead dims (std < threshold): 0/36
    off-diagonal |corr|: mean=0.2134  max=0.6875 (dims (11, 13))  redundant pairs (|corr|>threshold): 0
    effective rank: 7.75/36 (participation ratio)  95%-variance components: 13/36  condition number: 4.29e+03
    smallest-std dims: 26(std=0.1549,mean=0.0995), 32(std=0.1648,mean=0.0372), 12(std=0.1952,mean=0.4795), 33(std=0.2084,mean=0.0855), 30(std=0.2187,mean=-0.1328)
    largest-std dims:  1(std=0.8223,mean=-0.0930), 2(std=0.7615,mean=-0.0600), 11(std=0.7458,mean=-1.6498), 10(std=0.6206,mean=0.2053), 0(std=0.5952,mean=0.4554)
    most-correlated pairs: (11,13)=-0.687, (0,13)=-0.678, (7,24)=0.658, (5,20)=-0.642, (22,15)=0.624
2026-08-23 11:51:40,567 INFO Training row-count summary (train split):
    main (per-horizon heads)        : 1,139,000 rows -- own batches
    autoencode/t0 (bottleneck recon): 5,695,000 rows -- own batches (1,139,000 rows x 5 horizons)
    adjacent-pair (dynamics)        : 4,561,805/5,695,000 horizon-pass rows mask-eligible -- shares the horizon pass's own latent, no extra rows/batches
    crossing_head (at each horizon) : 1,258,236/5,695,000 horizon-pass rows mask-eligible -- shares the horizon pass's own latent, no extra rows/batches
    crossing_head (at t=0, in main) : 393,586/1,139,000 main rows masked-valid (position term only; delta_t trains unmasked on the -1 sentinel) -- shares main's own latent
    goal_dist_delta_head (main only): 1,139,000 main rows, unmasked -- shares main's own latent
    short-horizon probes (main only): 1,139,000 main rows x 2 heads, unmasked -- shares main's own latent
2026-08-23 11:51:40,567 INFO Decoder-only pretraining: 10 epoch(s), lr=5.00e-05, optimizer=adam, freeze_latent=False
2026-08-23 11:52:48,690 INFO   decoder-only pretrain epoch 1/10: train_loss=0.0050  val_loss=0.0051  (improved by inf > min_delta=1.0e-06)
2026-08-23 11:52:48,690 INFO     crossing_head: train loss=0.0015 pos_dist=3.616m dt_mae=0.922s | val loss=0.0014 pos_dist=3.309m dt_mae=0.909s
2026-08-23 11:52:48,690 INFO     goal_dist_delta_head: train loss=0.00179 mae=(left 1.714m, right 1.717m) | val loss=0.00180 mae=(left 1.714m, right 1.701m)
2026-08-23 11:52:48,690 INFO     short_horizon_probes: train loss=0.00011 rmse_norm=(0.2s 0.0065, 1.0s 0.0084) | val loss=0.00012 rmse_norm=(0.2s 0.0067, 1.0s 0.0086)
2026-08-23 11:53:58,947 INFO   decoder-only pretrain epoch 2/10: train_loss=0.0049  val_loss=0.0049  (improved by 0.000137 > min_delta=1.0e-06)
2026-08-23 11:53:58,947 INFO     crossing_head: train loss=0.0015 pos_dist=3.626m dt_mae=0.920s | val loss=0.0014 pos_dist=3.300m dt_mae=0.909s
2026-08-23 11:53:58,947 INFO     goal_dist_delta_head: train loss=0.00179 mae=(left 1.713m, right 1.716m) | val loss=0.00179 mae=(left 1.731m, right 1.702m)
2026-08-23 11:53:58,947 INFO     short_horizon_probes: train loss=0.00011 rmse_norm=(0.2s 0.0065, 1.0s 0.0084) | val loss=0.00011 rmse_norm=(0.2s 0.0065, 1.0s 0.0084)
2026-08-23 11:55:02,297 INFO   decoder-only pretrain epoch 3/10: train_loss=0.0049  val_loss=0.0050  (patience 1/0, raw_drop=-0.000056 <= min_delta=1.0e-06)
2026-08-23 11:55:02,297 INFO     crossing_head: train loss=0.0015 pos_dist=3.616m dt_mae=0.920s | val loss=0.0014 pos_dist=3.297m dt_mae=0.909s
2026-08-23 11:55:02,298 INFO     goal_dist_delta_head: train loss=0.00179 mae=(left 1.713m, right 1.716m) | val loss=0.00178 mae=(left 1.719m, right 1.702m)
2026-08-23 11:55:02,298 INFO     short_horizon_probes: train loss=0.00011 rmse_norm=(0.2s 0.0064, 1.0s 0.0084) | val loss=0.00011 rmse_norm=(0.2s 0.0064, 1.0s 0.0084)
2026-08-23 11:56:01,838 INFO   decoder-only pretrain epoch 4/10: train_loss=0.0049  val_loss=0.0049  (patience 2/0, raw_drop=-0.000003 <= min_delta=1.0e-06)
2026-08-23 11:56:01,838 INFO     crossing_head: train loss=0.0015 pos_dist=3.616m dt_mae=0.919s | val loss=0.0014 pos_dist=3.305m dt_mae=0.908s
2026-08-23 11:56:01,838 INFO     goal_dist_delta_head: train loss=0.00178 mae=(left 1.712m, right 1.714m) | val loss=0.00178 mae=(left 1.705m, right 1.711m)
2026-08-23 11:56:01,838 INFO     short_horizon_probes: train loss=0.00011 rmse_norm=(0.2s 0.0064, 1.0s 0.0084) | val loss=0.00011 rmse_norm=(0.2s 0.0064, 1.0s 0.0084)
2026-08-23 11:57:04,973 INFO   decoder-only pretrain epoch 5/10: train_loss=0.0049  val_loss=0.0049  (improved by 0.000002 > min_delta=1.0e-06)
2026-08-23 11:57:04,974 INFO     crossing_head: train loss=0.0015 pos_dist=3.615m dt_mae=0.919s | val loss=0.0014 pos_dist=3.319m dt_mae=0.908s
2026-08-23 11:57:04,974 INFO     goal_dist_delta_head: train loss=0.00178 mae=(left 1.712m, right 1.713m) | val loss=0.00178 mae=(left 1.719m, right 1.722m)
2026-08-23 11:57:04,974 INFO     short_horizon_probes: train loss=0.00011 rmse_norm=(0.2s 0.0064, 1.0s 0.0084) | val loss=0.00011 rmse_norm=(0.2s 0.0064, 1.0s 0.0084)
2026-08-23 11:58:05,062 INFO   decoder-only pretrain epoch 6/10: train_loss=0.0049  val_loss=0.0049  (improved by 0.000010 > min_delta=1.0e-06)
2026-08-23 11:58:05,062 INFO     crossing_head: train loss=0.0015 pos_dist=3.613m dt_mae=0.919s | val loss=0.0014 pos_dist=3.287m dt_mae=0.909s
2026-08-23 11:58:05,062 INFO     goal_dist_delta_head: train loss=0.00178 mae=(left 1.711m, right 1.712m) | val loss=0.00178 mae=(left 1.716m, right 1.705m)
2026-08-23 11:58:05,062 INFO     short_horizon_probes: train loss=0.00011 rmse_norm=(0.2s 0.0064, 1.0s 0.0083) | val loss=0.00011 rmse_norm=(0.2s 0.0064, 1.0s 0.0083)
2026-08-23 11:59:09,475 INFO   decoder-only pretrain epoch 7/10: train_loss=0.0049  val_loss=0.0050  (patience 1/0, raw_drop=-0.000077 <= min_delta=1.0e-06)
2026-08-23 11:59:09,476 INFO     crossing_head: train loss=0.0015 pos_dist=3.605m dt_mae=0.918s | val loss=0.0014 pos_dist=3.304m dt_mae=0.910s
2026-08-23 11:59:09,476 INFO     goal_dist_delta_head: train loss=0.00178 mae=(left 1.711m, right 1.711m) | val loss=0.00178 mae=(left 1.705m, right 1.702m)
2026-08-23 11:59:09,476 INFO     short_horizon_probes: train loss=0.00011 rmse_norm=(0.2s 0.0064, 1.0s 0.0083) | val loss=0.00011 rmse_norm=(0.2s 0.0064, 1.0s 0.0084)
2026-08-23 12:00:12,227 INFO   decoder-only pretrain epoch 8/10: train_loss=0.0049  val_loss=0.0050  (patience 2/0, raw_drop=-0.000066 <= min_delta=1.0e-06)
2026-08-23 12:00:12,227 INFO     crossing_head: train loss=0.0015 pos_dist=3.618m dt_mae=0.918s | val loss=0.0014 pos_dist=3.331m dt_mae=0.908s
2026-08-23 12:00:12,227 INFO     goal_dist_delta_head: train loss=0.00178 mae=(left 1.710m, right 1.710m) | val loss=0.00178 mae=(left 1.710m, right 1.707m)
2026-08-23 12:00:12,227 INFO     short_horizon_probes: train loss=0.00011 rmse_norm=(0.2s 0.0063, 1.0s 0.0083) | val loss=0.00011 rmse_norm=(0.2s 0.0063, 1.0s 0.0083)
2026-08-23 12:01:15,449 INFO   decoder-only pretrain epoch 9/10: train_loss=0.0049  val_loss=0.0049  (improved by 0.000022 > min_delta=1.0e-06)
2026-08-23 12:01:15,449 INFO     crossing_head: train loss=0.0015 pos_dist=3.606m dt_mae=0.918s | val loss=0.0014 pos_dist=3.288m dt_mae=0.914s
2026-08-23 12:01:15,449 INFO     goal_dist_delta_head: train loss=0.00178 mae=(left 1.710m, right 1.710m) | val loss=0.00177 mae=(left 1.705m, right 1.699m)
2026-08-23 12:01:15,449 INFO     short_horizon_probes: train loss=0.00011 rmse_norm=(0.2s 0.0063, 1.0s 0.0083) | val loss=0.00011 rmse_norm=(0.2s 0.0063, 1.0s 0.0083)
2026-08-23 12:02:25,192 INFO   decoder-only pretrain epoch 10/10: train_loss=0.0049  val_loss=0.0049  (improved by 0.000019 > min_delta=1.0e-06)
2026-08-23 12:02:25,192 INFO     crossing_head: train loss=0.0015 pos_dist=3.610m dt_mae=0.918s | val loss=0.0014 pos_dist=3.347m dt_mae=0.909s
2026-08-23 12:02:25,192 INFO     goal_dist_delta_head: train loss=0.00178 mae=(left 1.709m, right 1.709m) | val loss=0.00177 mae=(left 1.718m, right 1.700m)
2026-08-23 12:02:25,192 INFO     short_horizon_probes: train loss=0.00011 rmse_norm=(0.2s 0.0063, 1.0s 0.0083) | val loss=0.00011 rmse_norm=(0.2s 0.0063, 1.0s 0.0083)
2026-08-23 12:02:25,194 INFO Decoder-only pretraining: restored best-val weights (val_loss=0.0049)
2026-08-23 12:02:25,197 INFO Saved 'after_decoder_pretrain' checkpoint to checkpoints/physics_pretrain/player_encoder_14.after_decoder_pretrain.pt
2026-08-23 12:03:57,493 INFO epoch 1/500: train_loss=0.0049  pair_loss=0.0021  t0_loss=0.0007  val_loss=0.0049  best=0.0049  (improved by inf > min_delta=1.0e-07)
2026-08-23 12:03:57,493 INFO     grad_norm: mean=0.350798 std=0.193689 min=0.085535 max=2.183054
2026-08-23 12:03:57,493 INFO     train_loss_delta (batch-to-batch): mean=0.000000 std=0.000251 min=-0.001872 max=0.001996
2026-08-23 12:03:57,493 INFO     crossing_head: train loss=0.0015 pos_dist=3.614m dt_mae=0.917s | val loss=0.0014 pos_dist=3.285m dt_mae=0.902s
2026-08-23 12:03:57,493 INFO     goal_dist_delta_head: train loss=0.00178 mae=(left 1.706m, right 1.705m) | val loss=0.00177 mae=(left 1.708m, right 1.706m)
2026-08-23 12:03:57,493 INFO     short_horizon_probes: train loss=0.00011 rmse_norm=(0.2s 0.0062, 1.0s 0.0083) | val loss=0.00011 rmse_norm=(0.2s 0.0062, 1.0s 0.0083)
2026-08-23 12:03:57,493 INFO     val_loss_delta (epoch-over-epoch): nan
2026-08-23 12:03:57,494 INFO         train pos_rmse     by horizon (m): [0.8954 0.9279 1.6261 1.4525 1.3274], mean: 1.2459 m
2026-08-23 12:03:57,494 INFO         val   pos_rmse     by horizon (m): [0.8812 0.9322 1.6197 1.4667 1.3485], mean: 1.2497 m
2026-08-23 12:03:57,494 INFO         train pos_dist     by horizon (m): [1.0302 1.0995 1.9389 1.635  1.3075], mean: 1.4022 m
2026-08-23 12:03:57,494 INFO         val   pos_dist     by horizon (m): [1.0226 1.1096 1.9403 1.6399 1.3515], mean: 1.4128 m
2026-08-23 12:03:57,495 INFO         train vel_rmse     by horizon (m/s): [1.1446 0.5736 0.5232 0.4131 0.3906], mean: 0.6090 m/s
2026-08-23 12:03:57,495 INFO         val   vel_rmse     by horizon (m/s): [1.1403 0.5768 0.5281 0.4166 0.3867], mean: 0.6097 m/s
2026-08-23 12:03:57,495 INFO         train vel_dist     by horizon (m/s): [1.3136 0.6401 0.5778 0.4647 0.4594], mean: 0.6911 m/s
2026-08-23 12:03:57,495 INFO         val   vel_dist     by horizon (m/s): [1.3089 0.6454 0.5787 0.4688 0.4536], mean: 0.6911 m/s
2026-08-23 12:03:57,495 INFO         train heading_rmse by horizon: [0.0332 0.0286 0.0126 0.0041 0.0026], mean: 0.0162
2026-08-23 12:03:57,495 INFO         val   heading_rmse by horizon: [0.0331 0.0286 0.0131 0.0037 0.0024], mean: 0.0162
2026-08-23 12:03:57,495 INFO         train heading_dist by horizon: [0.0236 0.0126 0.0026 0.0017 0.0017], mean: 0.0084
2026-08-23 12:03:57,496 INFO         val   heading_dist by horizon: [0.0235 0.0124 0.0025 0.0015 0.0015], mean: 0.0083
2026-08-23 12:03:57,496 INFO         train stamina_rmse by horizon: [0.0043 0.0025 0.0045 0.0042 0.0086], mean: 0.0048
2026-08-23 12:03:57,496 INFO         val   stamina_rmse by horizon: [0.0044 0.0026 0.0044 0.0042 0.0086], mean: 0.0048
2026-08-23 12:03:57,496 INFO     train pos       R2 by horizon: [0.999 0.999 0.997 0.998 0.999], mean: 0.998
2026-08-23 12:03:57,496 INFO     val   pos       R2 by horizon: [0.999 0.999 0.997 0.997 0.999], mean: 0.998
2026-08-23 12:03:57,496 INFO     train pos       %-of-persistence by horizon: [218.   47.9  24.7  12.5   5.4]
2026-08-23 12:03:57,497 INFO     val   pos       %-of-persistence by horizon: [214.5  48.1  24.6  12.6   5.5]
2026-08-23 12:03:57,497 INFO     train vel       R2 by horizon: [0.679 0.939 0.961 0.975 0.977], mean: 0.906
2026-08-23 12:03:57,497 INFO     val   vel       R2 by horizon: [0.681 0.939 0.96  0.975 0.977], mean: 0.906
2026-08-23 12:03:57,497 INFO     train vel       %-of-persistence by horizon: [136.2  20.9  15.3  12.1  11.6]
2026-08-23 12:03:57,497 INFO     val   vel       %-of-persistence by horizon: [135.7  21.1  15.5  12.2  11.5]
2026-08-23 12:03:57,498 INFO     train heading   R2 by horizon: [0.998 0.998 1.    1.    1.   ], mean: 0.999
2026-08-23 12:03:57,498 INFO     val   heading   R2 by horizon: [0.998 0.998 1.    1.    1.   ], mean: 0.999
2026-08-23 12:03:57,498 INFO     train heading   %-of-persistence by horizon: [6.8 3.1 1.3 0.4 0.3]
2026-08-23 12:03:57,498 INFO     val   heading   %-of-persistence by horizon: [6.8 3.1 1.3 0.4 0.2]
2026-08-23 12:03:57,498 INFO     train stamina   R2 by horizon: [1.    1.    1.    1.    0.999], mean: 1.000
2026-08-23 12:03:57,498 INFO     val   stamina   R2 by horizon: [1.    1.    1.    1.    0.999], mean: 1.000
2026-08-23 12:03:57,498 INFO     train stamina   %-of-persistence by horizon: [489.7  57.6  33.9  19.   19.9]
2026-08-23 12:03:57,498 INFO     val   stamina   %-of-persistence by horizon: [492.3  58.2  33.5  19.   19.8]
2026-08-23 12:03:57,505 INFO Saved 'midtrain_latest' checkpoint to checkpoints/physics_pretrain/player_encoder_14.midtrain_latest.pt
2026-08-23 12:05:33,070 INFO epoch 2/500: train_loss=0.0049  pair_loss=0.0021  t0_loss=0.0007  val_loss=0.0049  best=0.0049  (improved by 0.000054 > min_delta=1.0e-07)
2026-08-23 12:05:33,070 INFO     grad_norm: mean=0.323999 std=0.146488 min=0.090856 max=1.629905
2026-08-23 12:05:33,070 INFO     train_loss_delta (batch-to-batch): mean=0.000000 std=0.000232 min=-0.001117 max=0.001513
2026-08-23 12:05:33,070 INFO     crossing_head: train loss=0.0015 pos_dist=3.601m dt_mae=0.915s | val loss=0.0014 pos_dist=3.276m dt_mae=0.900s
2026-08-23 12:05:33,070 INFO     goal_dist_delta_head: train loss=0.00177 mae=(left 1.706m, right 1.704m) | val loss=0.00177 mae=(left 1.711m, right 1.695m)
2026-08-23 12:05:33,070 INFO     short_horizon_probes: train loss=0.00011 rmse_norm=(0.2s 0.0062, 1.0s 0.0083) | val loss=0.00011 rmse_norm=(0.2s 0.0062, 1.0s 0.0083)
2026-08-23 12:05:33,070 INFO     val_loss_delta (epoch-over-epoch): -0.000054
2026-08-23 12:05:33,071 INFO         train pos_rmse     by horizon (m): [0.8942 0.9277 1.6212 1.4505 1.3195], mean: 1.2426 m
2026-08-23 12:05:33,071 INFO         val   pos_rmse     by horizon (m): [0.8976 0.9293 1.6218 1.4411 1.3556], mean: 1.2491 m
2026-08-23 12:05:33,071 INFO         train pos_dist     by horizon (m): [1.0288 1.1    1.9309 1.6331 1.2949], mean: 1.3975 m
2026-08-23 12:05:33,072 INFO         val   pos_dist     by horizon (m): [1.0271 1.103  1.9584 1.5935 1.3245], mean: 1.4013 m
2026-08-23 12:05:33,072 INFO         train vel_rmse     by horizon (m/s): [1.1446 0.5732 0.5227 0.4129 0.3902], mean: 0.6087 m/s
2026-08-23 12:05:33,072 INFO         val   vel_rmse     by horizon (m/s): [1.1451 0.5711 0.5261 0.4101 0.392 ], mean: 0.6089 m/s
2026-08-23 12:05:33,072 INFO         train vel_dist     by horizon (m/s): [1.3132 0.6394 0.5774 0.4646 0.4589], mean: 0.6907 m/s
2026-08-23 12:05:33,072 INFO         val   vel_dist     by horizon (m/s): [1.3143 0.6358 0.5852 0.4606 0.461 ], mean: 0.6914 m/s
2026-08-23 12:05:33,072 INFO         train heading_rmse by horizon: [0.033  0.0286 0.0126 0.0041 0.0025], mean: 0.0162
2026-08-23 12:05:33,073 INFO         val   heading_rmse by horizon: [0.0326 0.0283 0.013  0.0039 0.0027], mean: 0.0161
2026-08-23 12:05:33,073 INFO         train heading_dist by horizon: [0.0234 0.0125 0.0026 0.0017 0.0017], mean: 0.0084
2026-08-23 12:05:33,073 INFO         val   heading_dist by horizon: [0.0231 0.0121 0.0028 0.0017 0.0019], mean: 0.0083
2026-08-23 12:05:33,073 INFO         train stamina_rmse by horizon: [0.0044 0.0025 0.0044 0.0041 0.0086], mean: 0.0048
2026-08-23 12:05:33,073 INFO         val   stamina_rmse by horizon: [0.0044 0.0025 0.0044 0.0041 0.0086], mean: 0.0048
2026-08-23 12:05:33,073 INFO     train pos       R2 by horizon: [0.999 0.999 0.997 0.998 0.999], mean: 0.998
2026-08-23 12:05:33,073 INFO     val   pos       R2 by horizon: [0.999 0.999 0.997 0.998 0.999], mean: 0.998
2026-08-23 12:05:33,073 INFO     train pos       %-of-persistence by horizon: [217.7  47.9  24.7  12.4   5.4]
2026-08-23 12:05:33,074 INFO     val   pos       %-of-persistence by horizon: [218.5  48.   24.7  12.4   5.5]
2026-08-23 12:05:33,074 INFO     train vel       R2 by horizon: [0.679 0.939 0.961 0.975 0.977], mean: 0.906
2026-08-23 12:05:33,074 INFO     val   vel       R2 by horizon: [0.679 0.94  0.96  0.976 0.977], mean: 0.906
2026-08-23 12:05:33,074 INFO     train vel       %-of-persistence by horizon: [136.2  20.9  15.3  12.1  11.6]
2026-08-23 12:05:33,074 INFO     val   vel       %-of-persistence by horizon: [136.2  20.9  15.4  12.   11.6]
2026-08-23 12:05:33,074 INFO     train heading   R2 by horizon: [0.998 0.998 1.    1.    1.   ], mean: 0.999
2026-08-23 12:05:33,074 INFO     val   heading   R2 by horizon: [0.998 0.998 1.    1.    1.   ], mean: 0.999
2026-08-23 12:05:33,075 INFO     train heading   %-of-persistence by horizon: [6.8 3.1 1.3 0.4 0.3]
2026-08-23 12:05:33,075 INFO     val   heading   %-of-persistence by horizon: [6.7 3.1 1.3 0.4 0.3]
2026-08-23 12:05:33,075 INFO     train stamina   R2 by horizon: [1.    1.    1.    1.    0.999], mean: 1.000
2026-08-23 12:05:33,075 INFO     val   stamina   R2 by horizon: [1.    1.    1.    1.    0.999], mean: 1.000
2026-08-23 12:05:33,075 INFO     train stamina   %-of-persistence by horizon: [491.8  57.7  33.7  18.9  20. ]
2026-08-23 12:05:33,075 INFO     val   stamina   %-of-persistence by horizon: [495.7  55.7  33.1  18.9  19.9]
2026-08-23 12:05:33,081 INFO Saved 'midtrain_latest' checkpoint to checkpoints/physics_pretrain/player_encoder_14.midtrain_latest.pt
2026-08-23 12:07:12,217 INFO epoch 3/500: train_loss=0.0049  pair_loss=0.0021  t0_loss=0.0007  val_loss=0.0048  best=0.0048  (improved by 0.000034 > min_delta=1.0e-07)
2026-08-23 12:07:12,217 INFO     grad_norm: mean=0.341787 std=0.173183 min=0.089103 max=1.371239
2026-08-23 12:07:12,217 INFO     train_loss_delta (batch-to-batch): mean=-0.000000 std=0.000226 min=-0.000918 max=0.000840
2026-08-23 12:07:12,217 INFO     crossing_head: train loss=0.0015 pos_dist=3.588m dt_mae=0.913s | val loss=0.0014 pos_dist=3.304m dt_mae=0.908s
2026-08-23 12:07:12,217 INFO     goal_dist_delta_head: train loss=0.00177 mae=(left 1.707m, right 1.704m) | val loss=0.00177 mae=(left 1.707m, right 1.694m)
2026-08-23 12:07:12,217 INFO     short_horizon_probes: train loss=0.00011 rmse_norm=(0.2s 0.0062, 1.0s 0.0083) | val loss=0.00011 rmse_norm=(0.2s 0.0062, 1.0s 0.0083)
2026-08-23 12:07:12,217 INFO     val_loss_delta (epoch-over-epoch): -0.000034
2026-08-23 12:07:12,218 INFO         train pos_rmse     by horizon (m): [0.8937 0.9266 1.6183 1.4502 1.3183], mean: 1.2414 m
2026-08-23 12:07:12,218 INFO         val   pos_rmse     by horizon (m): [0.8991 0.9261 1.6099 1.459  1.2803], mean: 1.2349 m
2026-08-23 12:07:12,218 INFO         train pos_dist     by horizon (m): [1.028  1.0989 1.9274 1.6333 1.2978], mean: 1.3971 m
2026-08-23 12:07:12,218 INFO         val   pos_dist     by horizon (m): [1.033  1.0985 1.8996 1.6594 1.2616], mean: 1.3904 m
2026-08-23 12:07:12,218 INFO         train vel_rmse     by horizon (m/s): [1.1451 0.5731 0.5221 0.412  0.3893], mean: 0.6083 m/s
2026-08-23 12:07:12,218 INFO         val   vel_rmse     by horizon (m/s): [1.1442 0.5708 0.5184 0.4107 0.3877], mean: 0.6063 m/s
2026-08-23 12:07:12,218 INFO         train vel_dist     by horizon (m/s): [1.3135 0.6388 0.577  0.4638 0.4579], mean: 0.6902 m/s
2026-08-23 12:07:12,219 INFO         val   vel_dist     by horizon (m/s): [1.3132 0.6343 0.5715 0.4626 0.4559], mean: 0.6875 m/s
2026-08-23 12:07:12,219 INFO         train heading_rmse by horizon: [0.0329 0.0285 0.0126 0.0041 0.0025], mean: 0.0161
2026-08-23 12:07:12,219 INFO         val   heading_rmse by horizon: [0.0327 0.0285 0.013  0.0038 0.0025], mean: 0.0161
2026-08-23 12:07:12,219 INFO         train heading_dist by horizon: [0.0234 0.0125 0.0026 0.0017 0.0017], mean: 0.0084
2026-08-23 12:07:12,219 INFO         val   heading_dist by horizon: [0.0231 0.0123 0.0026 0.0016 0.0015], mean: 0.0082
2026-08-23 12:07:12,219 INFO         train stamina_rmse by horizon: [0.0044 0.0026 0.0044 0.0041 0.0087], mean: 0.0048
2026-08-23 12:07:12,219 INFO         val   stamina_rmse by horizon: [0.0044 0.0026 0.0044 0.0041 0.0086], mean: 0.0048
2026-08-23 12:07:12,219 INFO     train pos       R2 by horizon: [0.999 0.999 0.997 0.998 0.999], mean: 0.998
2026-08-23 12:07:12,220 INFO     val   pos       R2 by horizon: [0.999 0.999 0.997 0.998 0.999], mean: 0.998
2026-08-23 12:07:12,220 INFO     train pos       %-of-persistence by horizon: [217.6  47.8  24.6  12.4   5.4]
2026-08-23 12:07:12,220 INFO     val   pos       %-of-persistence by horizon: [218.9  47.8  24.5  12.5   5.2]
2026-08-23 12:07:12,220 INFO     train vel       R2 by horizon: [0.679 0.939 0.961 0.975 0.977], mean: 0.906
2026-08-23 12:07:12,220 INFO     val   vel       R2 by horizon: [0.679 0.94  0.961 0.975 0.977], mean: 0.907
2026-08-23 12:07:12,220 INFO     train vel       %-of-persistence by horizon: [136.2  20.9  15.3  12.1  11.5]
2026-08-23 12:07:12,220 INFO     val   vel       %-of-persistence by horizon: [136.1  20.8  15.2  12.   11.5]
2026-08-23 12:07:12,220 INFO     train heading   R2 by horizon: [0.998 0.998 1.    1.    1.   ], mean: 0.999
2026-08-23 12:07:12,221 INFO     val   heading   R2 by horizon: [0.998 0.998 1.    1.    1.   ], mean: 0.999
2026-08-23 12:07:12,221 INFO     train heading   %-of-persistence by horizon: [6.8 3.1 1.3 0.4 0.3]
2026-08-23 12:07:12,221 INFO     val   heading   %-of-persistence by horizon: [6.7 3.1 1.3 0.4 0.2]
2026-08-23 12:07:12,221 INFO     train stamina   R2 by horizon: [1.    1.    1.    1.    0.999], mean: 1.000
2026-08-23 12:07:12,221 INFO     val   stamina   R2 by horizon: [1.    1.    1.    1.    0.999], mean: 1.000
2026-08-23 12:07:12,221 INFO     train stamina   %-of-persistence by horizon: [495.3  57.9  33.4  18.9  20. ]
2026-08-23 12:07:12,222 INFO     val   stamina   %-of-persistence by horizon: [495.3  58.2  33.   18.5  19.9]
2026-08-23 12:07:12,227 INFO Saved 'midtrain_latest' checkpoint to checkpoints/physics_pretrain/player_encoder_14.midtrain_latest.pt
2026-08-23 12:08:51,924 INFO epoch 4/500: train_loss=0.0049  pair_loss=0.0021  t0_loss=0.0007  val_loss=0.0049  best=0.0048  (patience 1/25, raw_drop=-0.000011 <= min_delta=1.0e-07)
2026-08-23 12:08:51,924 INFO     grad_norm: mean=0.325402 std=0.152797 min=0.080215 max=1.849443
2026-08-23 12:08:51,924 INFO     train_loss_delta (batch-to-batch): mean=-0.000000 std=0.000234 min=-0.001710 max=0.001745
2026-08-23 12:08:51,924 INFO     crossing_head: train loss=0.0015 pos_dist=3.591m dt_mae=0.911s | val loss=0.0014 pos_dist=3.271m dt_mae=0.908s
2026-08-23 12:08:51,924 INFO     goal_dist_delta_head: train loss=0.00177 mae=(left 1.707m, right 1.702m) | val loss=0.00177 mae=(left 1.708m, right 1.688m)
2026-08-23 12:08:51,924 INFO     short_horizon_probes: train loss=0.00011 rmse_norm=(0.2s 0.0062, 1.0s 0.0083) | val loss=0.00011 rmse_norm=(0.2s 0.0062, 1.0s 0.0083)
2026-08-23 12:08:51,924 INFO     val_loss_delta (epoch-over-epoch): 0.000011
2026-08-23 12:08:51,924 INFO         train pos_rmse     by horizon (m): [0.8935 0.9259 1.6142 1.4479 1.3107], mean: 1.2385 m
2026-08-23 12:08:51,924 INFO         val   pos_rmse     by horizon (m): [0.8903 0.9263 1.6201 1.4354 1.2564], mean: 1.2257 m
2026-08-23 12:08:51,924 INFO         train pos_dist     by horizon (m): [1.0274 1.0982 1.9213 1.6304 1.2863], mean: 1.3927 m
2026-08-23 12:08:51,925 INFO         val   pos_dist     by horizon (m): [1.0197 1.0985 1.9352 1.5927 1.1988], mean: 1.3690 m
2026-08-23 12:08:51,925 INFO         train vel_rmse     by horizon (m/s): [1.1445 0.5731 0.522  0.4117 0.3889], mean: 0.6080 m/s
2026-08-23 12:08:51,925 INFO         val   vel_rmse     by horizon (m/s): [1.143  0.5727 0.5215 0.4109 0.3878], mean: 0.6072 m/s
2026-08-23 12:08:51,925 INFO         train vel_dist     by horizon (m/s): [1.3125 0.6387 0.5769 0.4635 0.4573], mean: 0.6898 m/s
2026-08-23 12:08:51,925 INFO         val   vel_dist     by horizon (m/s): [1.3107 0.636  0.5762 0.4627 0.4567], mean: 0.6885 m/s
2026-08-23 12:08:51,925 INFO         train heading_rmse by horizon: [0.0329 0.0284 0.0126 0.0041 0.0025], mean: 0.0161
2026-08-23 12:08:51,925 INFO         val   heading_rmse by horizon: [0.0328 0.029  0.0131 0.0039 0.0028], mean: 0.0163
2026-08-23 12:08:51,925 INFO         train heading_dist by horizon: [0.0233 0.0124 0.0026 0.0017 0.0017], mean: 0.0083
2026-08-23 12:08:51,925 INFO         val   heading_dist by horizon: [0.0231 0.0126 0.0028 0.0018 0.0021], mean: 0.0085
2026-08-23 12:08:51,925 INFO         train stamina_rmse by horizon: [0.0044 0.0026 0.0044 0.0041 0.0087], mean: 0.0048
2026-08-23 12:08:51,925 INFO         val   stamina_rmse by horizon: [0.0044 0.0025 0.0043 0.0041 0.0086], mean: 0.0048
2026-08-23 12:08:51,926 INFO     train pos       R2 by horizon: [0.999 0.999 0.997 0.998 0.999], mean: 0.998
2026-08-23 12:08:51,926 INFO     val   pos       R2 by horizon: [0.999 0.999 0.997 0.998 0.999], mean: 0.998
2026-08-23 12:08:51,926 INFO     train pos       %-of-persistence by horizon: [217.5  47.8  24.5  12.4   5.4]
2026-08-23 12:08:51,926 INFO     val   pos       %-of-persistence by horizon: [216.7  47.8  24.6  12.3   5.1]
2026-08-23 12:08:51,926 INFO     train vel       R2 by horizon: [0.679 0.939 0.961 0.975 0.977], mean: 0.906
2026-08-23 12:08:51,926 INFO     val   vel       R2 by horizon: [0.68  0.939 0.961 0.975 0.977], mean: 0.907
2026-08-23 12:08:51,926 INFO     train vel       %-of-persistence by horizon: [136.2  20.9  15.3  12.1  11.5]
2026-08-23 12:08:51,926 INFO     val   vel       %-of-persistence by horizon: [136.   20.9  15.3  12.   11.5]
2026-08-23 12:08:51,926 INFO     train heading   R2 by horizon: [0.998 0.998 1.    1.    1.   ], mean: 0.999
2026-08-23 12:08:51,927 INFO     val   heading   R2 by horizon: [0.998 0.998 1.    1.    1.   ], mean: 0.999
2026-08-23 12:08:51,927 INFO     train heading   %-of-persistence by horizon: [6.7 3.1 1.3 0.4 0.3]
2026-08-23 12:08:51,927 INFO     val   heading   %-of-persistence by horizon: [6.7 3.1 1.3 0.4 0.3]
2026-08-23 12:08:51,927 INFO     train stamina   R2 by horizon: [1.    1.    1.    1.    0.999], mean: 1.000
2026-08-23 12:08:51,927 INFO     val   stamina   R2 by horizon: [1.    1.    1.    1.    0.999], mean: 1.000
2026-08-23 12:08:51,927 INFO     train stamina   %-of-persistence by horizon: [498.2  58.   33.2  18.8  20. ]
2026-08-23 12:08:51,927 INFO     val   stamina   %-of-persistence by horizon: [499.6  56.8  32.5  18.6  19.8]
2026-08-23 12:10:27,232 INFO epoch 5/500: train_loss=0.0048  pair_loss=0.0021  t0_loss=0.0007  val_loss=0.0049  best=0.0048  (patience 2/25, raw_drop=-0.000042 <= min_delta=1.0e-07)
2026-08-23 12:10:27,232 INFO     grad_norm: mean=0.332451 std=0.147067 min=0.100173 max=1.380747
2026-08-23 12:10:27,232 INFO     train_loss_delta (batch-to-batch): mean=-0.000000 std=0.000231 min=-0.000858 max=0.001044
2026-08-23 12:10:27,232 INFO     crossing_head: train loss=0.0015 pos_dist=3.573m dt_mae=0.910s | val loss=0.0014 pos_dist=3.284m dt_mae=0.892s
2026-08-23 12:10:27,232 INFO     goal_dist_delta_head: train loss=0.00177 mae=(left 1.707m, right 1.700m) | val loss=0.00177 mae=(left 1.719m, right 1.688m)
2026-08-23 12:10:27,232 INFO     short_horizon_probes: train loss=0.00011 rmse_norm=(0.2s 0.0062, 1.0s 0.0083) | val loss=0.00011 rmse_norm=(0.2s 0.0062, 1.0s 0.0083)
2026-08-23 12:10:27,232 INFO     val_loss_delta (epoch-over-epoch): 0.000031
2026-08-23 12:10:27,232 INFO         train pos_rmse     by horizon (m): [0.8935 0.9256 1.6117 1.4458 1.3087], mean: 1.2370 m
2026-08-23 12:10:27,232 INFO         val   pos_rmse     by horizon (m): [0.8863 0.9253 1.6005 1.4374 1.2898], mean: 1.2279 m
2026-08-23 12:10:27,233 INFO         train pos_dist     by horizon (m): [1.0272 1.0977 1.9177 1.6286 1.2864], mean: 1.3915 m
2026-08-23 12:10:27,233 INFO         val   pos_dist     by horizon (m): [1.0187 1.0973 1.9044 1.613  1.21  ], mean: 1.3687 m
2026-08-23 12:10:27,233 INFO         train vel_rmse     by horizon (m/s): [1.1448 0.5729 0.5216 0.4113 0.3884], mean: 0.6078 m/s
2026-08-23 12:10:27,233 INFO         val   vel_rmse     by horizon (m/s): [1.145  0.5751 0.5235 0.4091 0.3865], mean: 0.6078 m/s
2026-08-23 12:10:27,233 INFO         train vel_dist     by horizon (m/s): [1.3128 0.6382 0.5766 0.4632 0.4566], mean: 0.6895 m/s
2026-08-23 12:10:27,233 INFO         val   vel_dist     by horizon (m/s): [1.314  0.6399 0.5798 0.4596 0.4537], mean: 0.6894 m/s
2026-08-23 12:10:27,233 INFO         train heading_rmse by horizon: [0.0328 0.0284 0.0125 0.0041 0.0026], mean: 0.0161
2026-08-23 12:10:27,233 INFO         val   heading_rmse by horizon: [0.0333 0.0289 0.0131 0.0038 0.0024], mean: 0.0163
2026-08-23 12:10:27,233 INFO         train heading_dist by horizon: [0.0232 0.0124 0.0026 0.0017 0.0017], mean: 0.0083
2026-08-23 12:10:27,233 INFO         val   heading_dist by horizon: [0.0236 0.0128 0.0026 0.0016 0.0015], mean: 0.0084
2026-08-23 12:10:27,233 INFO         train stamina_rmse by horizon: [0.0044 0.0026 0.0043 0.0041 0.0087], mean: 0.0048
2026-08-23 12:10:27,234 INFO         val   stamina_rmse by horizon: [0.0044 0.0025 0.0043 0.0041 0.0086], mean: 0.0048
2026-08-23 12:10:27,234 INFO     train pos       R2 by horizon: [0.999 0.999 0.997 0.998 0.999], mean: 0.998
2026-08-23 12:10:27,234 INFO     val   pos       R2 by horizon: [0.999 0.999 0.997 0.998 0.999], mean: 0.998
2026-08-23 12:10:27,234 INFO     train pos       %-of-persistence by horizon: [217.5  47.8  24.5  12.4   5.3]
2026-08-23 12:10:27,234 INFO     val   pos       %-of-persistence by horizon: [215.8  47.8  24.3  12.3   5.3]
2026-08-23 12:10:27,234 INFO     train vel       R2 by horizon: [0.679 0.939 0.961 0.975 0.977], mean: 0.906
2026-08-23 12:10:27,234 INFO     val   vel       R2 by horizon: [0.679 0.939 0.961 0.976 0.977], mean: 0.906
2026-08-23 12:10:27,235 INFO     train vel       %-of-persistence by horizon: [136.2  20.9  15.3  12.   11.5]
2026-08-23 12:10:27,235 INFO     val   vel       %-of-persistence by horizon: [136.2  21.   15.3  12.   11.5]
2026-08-23 12:10:27,235 INFO     train heading   R2 by horizon: [0.998 0.998 1.    1.    1.   ], mean: 0.999
2026-08-23 12:10:27,235 INFO     val   heading   R2 by horizon: [0.998 0.998 1.    1.    1.   ], mean: 0.999
2026-08-23 12:10:27,235 INFO     train heading   %-of-persistence by horizon: [6.7 3.1 1.3 0.4 0.3]
2026-08-23 12:10:27,235 INFO     val   heading   %-of-persistence by horizon: [6.8 3.1 1.3 0.4 0.2]
2026-08-23 12:10:27,235 INFO     train stamina   R2 by horizon: [1.    1.    1.    1.    0.999], mean: 1.000
2026-08-23 12:10:27,235 INFO     val   stamina   R2 by horizon: [1.    1.    1.    1.    0.999], mean: 1.000
2026-08-23 12:10:27,236 INFO     train stamina   %-of-persistence by horizon: [500.5  58.1  32.9  18.7  20. ]
2026-08-23 12:10:27,236 INFO     val   stamina   %-of-persistence by horizon: [500.7  56.   32.9  18.7  19.9]
2026-08-23 12:12:08,498 INFO Loaded 1340 shard(s) from physics_pretrain_data/player
2026-08-23 12:12:08,634 INFO Dataset: 1,340,000 episodes (1,139,000 train / 201,000 val)
2026-08-23 12:12:08,886 INFO pos_weight (max cap: 1.0):
2026-08-23 12:12:08,887 INFO     t= 0.2s  out_of_bounds=1.00  goal_scored=1.00
2026-08-23 12:12:08,887 INFO     t= 1.0s  out_of_bounds=1.00  goal_scored=1.00
2026-08-23 12:12:08,887 INFO     t= 3.0s  out_of_bounds=1.00  goal_scored=1.00
2026-08-23 12:12:08,887 INFO     t= 5.0s  out_of_bounds=1.00  goal_scored=1.00
2026-08-23 12:12:08,887 INFO     t=10.0s  out_of_bounds=1.00  goal_scored=1.00
2026-08-23 12:12:13,222 INFO Latent diagnostics (50,000 rows, latent_dim=36):
    per-dim std: mean=0.0725  pooled=0.2174  |  latent norm: mean=1.2978 std=0.1535
    dead dims (std < threshold): 29/36  [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    off-diagonal |corr|: mean=0.0033  max=0.8220 (dims (5, 2))  redundant pairs (|corr|>threshold): 0
    effective rank: 3.57/36 (participation ratio)  95%-variance components: 5/36  condition number: 5.09e+11
    smallest-std dims: 7(std=0.0000,mean=0.0000), 14(std=0.0000,mean=0.0000), 13(std=0.0000,mean=0.0000), 12(std=0.0000,mean=0.0000), 11(std=0.0000,mean=0.0000)
    largest-std dims:  5(std=0.7120,mean=0.0045), 4(std=0.7077,mean=0.0005), 0(std=0.5182,mean=-0.0015), 1(std=0.3141,mean=0.0015), 6(std=0.2889,mean=0.4970)
    most-correlated pairs: (5,2)=0.822, (4,3)=0.800, (2,3)=0.126, (3,5)=0.103, (4,2)=0.055
2026-08-23 12:12:16,573 INFO Training row-count summary (train split):
    main (per-horizon heads)        : 1,139,000 rows -- own batches
    autoencode/t0 (bottleneck recon): 5,695,000 rows -- own batches (1,139,000 rows x 5 horizons)
    adjacent-pair (dynamics)        : 4,561,805/5,695,000 horizon-pass rows mask-eligible -- shares the horizon pass's own latent, no extra rows/batches
    crossing_head (at each horizon) : 1,258,236/5,695,000 horizon-pass rows mask-eligible -- shares the horizon pass's own latent, no extra rows/batches
    crossing_head (at t=0, in main) : 393,586/1,139,000 main rows masked-valid (position term only; delta_t trains unmasked on the -1 sentinel) -- shares main's own latent
    goal_dist_delta_head (main only): 1,139,000 main rows, unmasked -- shares main's own latent
    short-horizon probes (main only): 1,139,000 main rows x 2 heads, unmasked -- shares main's own latent
2026-08-23 12:12:16,573 INFO Autoencode pretraining: 5 epoch(s), lr=3.00e-04, optimizer=adam
2026-08-23 12:12:59,228 INFO   autoencode pretrain epoch 1/5: train_loss=0.0178  val_loss=0.0001
2026-08-23 12:12:59,229 INFO         train pos_rmse     by horizon (m): [1.5234 1.5859 1.599  1.4752 1.9055], mean: 1.6178 m
2026-08-23 12:12:59,229 INFO         train pos_dist     by horizon (m): [1.8664 1.9787 2.0233 1.8616 2.3241], mean: 2.0108 m
2026-08-23 12:12:59,229 INFO         train vel_rmse     by horizon (m/s): [1.168  1.0857 1.2373 1.136  1.1781], mean: 1.1610 m/s
2026-08-23 12:12:59,229 INFO         train vel_dist     by horizon (m/s): [1.4257 1.3151 1.4757 1.3475 1.422 ], mean: 1.3972 m/s
2026-08-23 12:12:59,230 INFO         train heading_rmse by horizon: [0.0238 0.0228 0.0253 0.0218 0.0252], mean: 0.0238
2026-08-23 12:12:59,230 INFO         train heading_dist by horizon: [0.0239 0.0254 0.0303 0.0229 0.0315], mean: 0.0268
2026-08-23 12:12:59,230 INFO         train stamina_rmse by horizon: [0.0189 0.0187 0.0196 0.0171 0.0206], mean: 0.0190
2026-08-23 12:12:59,230 INFO         val   pos_rmse     by horizon (m): [0.4154 0.2834 0.2576 0.2554 0.2946], mean: 0.3013 m
2026-08-23 12:12:59,230 INFO         val   pos_dist     by horizon (m): [0.4829 0.3419 0.3171 0.3156 0.3388], mean: 0.3593 m
2026-08-23 12:12:59,230 INFO         val   vel_rmse     by horizon (m/s): [0.7495 0.4012 0.4328 0.4182 0.4026], mean: 0.4809 m/s
2026-08-23 12:12:59,230 INFO         val   vel_dist     by horizon (m/s): [0.8751 0.4678 0.4829 0.4711 0.4588], mean: 0.5512 m/s
2026-08-23 12:12:59,230 INFO         val   heading_rmse by horizon: [0.0073 0.0047 0.0043 0.0043 0.0047], mean: 0.0051
2026-08-23 12:12:59,231 INFO         val   heading_dist by horizon: [0.0051 0.0035 0.0032 0.0033 0.0034], mean: 0.0037
2026-08-23 12:12:59,231 INFO         val   stamina_rmse by horizon: [0.0056 0.0037 0.0034 0.0035 0.0038], mean: 0.0040
2026-08-23 12:13:40,284 INFO   autoencode pretrain epoch 2/5: train_loss=0.0002  val_loss=0.0000
2026-08-23 12:13:40,285 INFO         train pos_rmse     by horizon (m): [0.4192 0.3018 0.2923 0.2926 0.336 ], mean: 0.3284 m
2026-08-23 12:13:40,285 INFO         train pos_dist     by horizon (m): [0.5065 0.3833 0.3785 0.3814 0.4127], mean: 0.4125 m
2026-08-23 12:13:40,285 INFO         train vel_rmse     by horizon (m/s): [0.4529 0.2377 0.2369 0.2227 0.2243], mean: 0.2749 m/s
2026-08-23 12:13:40,285 INFO         train vel_dist     by horizon (m/s): [0.5235 0.2801 0.2763 0.2629 0.2644], mean: 0.3214 m/s
2026-08-23 12:13:40,285 INFO         train heading_rmse by horizon: [0.0066 0.0048 0.0045 0.0044 0.0049], mean: 0.0050
2026-08-23 12:13:40,286 INFO         train heading_dist by horizon: [0.0048 0.0038 0.0036 0.0036 0.0038], mean: 0.0039
2026-08-23 12:13:40,286 INFO         train stamina_rmse by horizon: [0.0069 0.0056 0.0055 0.0053 0.0057], mean: 0.0058
2026-08-23 12:13:40,286 INFO         val   pos_rmse     by horizon (m): [0.2803 0.1802 0.1616 0.1596 0.1945], mean: 0.1952 m
2026-08-23 12:13:40,286 INFO         val   pos_dist     by horizon (m): [0.3197 0.2187 0.2028 0.2015 0.2162], mean: 0.2318 m
2026-08-23 12:13:40,286 INFO         val   vel_rmse     by horizon (m/s): [0.2211 0.1249 0.115  0.1129 0.1173], mean: 0.1382 m/s
2026-08-23 12:13:40,286 INFO         val   vel_dist     by horizon (m/s): [0.2515 0.1451 0.1358 0.1345 0.1368], mean: 0.1607 m/s
2026-08-23 12:13:40,286 INFO         val   heading_rmse by horizon: [0.0057 0.0036 0.0033 0.0033 0.0034], mean: 0.0039
2026-08-23 12:13:40,286 INFO         val   heading_dist by horizon: [0.0041 0.0028 0.0027 0.0026 0.0026], mean: 0.0030
2026-08-23 12:13:40,286 INFO         val   stamina_rmse by horizon: [0.0057 0.0026 0.0024 0.0024 0.0027], mean: 0.0032
2026-08-23 12:14:22,922 INFO   autoencode pretrain epoch 3/5: train_loss=0.0001  val_loss=0.0003
2026-08-23 12:14:22,922 INFO         train pos_rmse     by horizon (m): [0.2834 0.2439 0.2211 0.2199 0.2524], mean: 0.2441 m
2026-08-23 12:14:22,922 INFO         train pos_dist     by horizon (m): [0.3432 0.3195 0.2934 0.2928 0.3138], mean: 0.3125 m
2026-08-23 12:14:22,923 INFO         train vel_rmse     by horizon (m/s): [0.1546 0.1005 0.0919 0.0906 0.0999], mean: 0.1075 m/s
2026-08-23 12:14:22,923 INFO         train vel_dist     by horizon (m/s): [0.1788 0.1195 0.1114 0.1104 0.1168], mean: 0.1274 m/s
2026-08-23 12:14:22,923 INFO         train heading_rmse by horizon: [0.0047 0.0037 0.0034 0.0034 0.0037], mean: 0.0038
2026-08-23 12:14:22,923 INFO         train heading_dist by horizon: [0.0035 0.003  0.0029 0.0028 0.0029], mean: 0.0030
2026-08-23 12:14:22,923 INFO         train stamina_rmse by horizon: [0.0048 0.0042 0.0037 0.0038 0.0039], mean: 0.0041
2026-08-23 12:14:22,923 INFO         val   pos_rmse     by horizon (m): [0.5483 0.4851 0.475  0.4746 0.4823], mean: 0.4931 m
2026-08-23 12:14:22,923 INFO         val   pos_dist     by horizon (m): [0.735  0.6607 0.6496 0.6487 0.6521], mean: 0.6692 m
2026-08-23 12:14:22,923 INFO         val   vel_rmse     by horizon (m/s): [0.1091 0.0805 0.0776 0.0764 0.0816], mean: 0.0850 m/s
2026-08-23 12:14:22,924 INFO         val   vel_dist     by horizon (m/s): [0.1299 0.1001 0.0982 0.0969 0.0983], mean: 0.1047 m/s
2026-08-23 12:14:22,924 INFO         val   heading_rmse by horizon: [0.0064 0.0072 0.0074 0.0074 0.0074], mean: 0.0072
2026-08-23 12:14:22,924 INFO         val   heading_dist by horizon: [0.0053 0.0064 0.0066 0.0066 0.0066], mean: 0.0063
2026-08-23 12:14:22,924 INFO         val   stamina_rmse by horizon: [0.0147 0.0137 0.0135 0.0135 0.0135], mean: 0.0138
2026-08-23 12:15:05,767 INFO   autoencode pretrain epoch 4/5: train_loss=0.0001  val_loss=0.0000
2026-08-23 12:15:05,767 INFO         train pos_rmse     by horizon (m): [0.2313 0.1937 0.1803 0.1724 0.1967], mean: 0.1949 m
2026-08-23 12:15:05,768 INFO         train pos_dist     by horizon (m): [0.2862 0.2576 0.2432 0.2319 0.2441], mean: 0.2526 m
2026-08-23 12:15:05,768 INFO         train vel_rmse     by horizon (m/s): [0.0986 0.0721 0.066  0.065  0.0776], mean: 0.0759 m/s
2026-08-23 12:15:05,768 INFO         train vel_dist     by horizon (m/s): [0.116  0.0869 0.0807 0.0797 0.0873], mean: 0.0901 m/s
2026-08-23 12:15:05,768 INFO         train heading_rmse by horizon: [0.0041 0.0035 0.0031 0.003  0.0034], mean: 0.0034
2026-08-23 12:15:05,768 INFO         train heading_dist by horizon: [0.0031 0.0029 0.0026 0.0025 0.0027], mean: 0.0028
2026-08-23 12:15:05,768 INFO         train stamina_rmse by horizon: [0.004  0.0035 0.0032 0.0034 0.0033], mean: 0.0035
2026-08-23 12:15:05,769 INFO         val   pos_rmse     by horizon (m): [0.1092 0.0543 0.045  0.0443 0.0791], mean: 0.0664 m
2026-08-23 12:15:05,769 INFO         val   pos_dist     by horizon (m): [0.1049 0.0547 0.0478 0.0477 0.0595], mean: 0.0629 m
2026-08-23 12:15:05,769 INFO         val   vel_rmse     by horizon (m/s): [0.0746 0.0489 0.044  0.0433 0.0566], mean: 0.0535 m/s
2026-08-23 12:15:05,769 INFO         val   vel_dist     by horizon (m/s): [0.0848 0.056  0.0517 0.0512 0.058 ], mean: 0.0604 m/s
2026-08-23 12:15:05,769 INFO         val   heading_rmse by horizon: [0.005  0.0042 0.0041 0.0041 0.0042], mean: 0.0043
2026-08-23 12:15:05,769 INFO         val   heading_dist by horizon: [0.0042 0.0037 0.0036 0.0036 0.0036], mean: 0.0037
2026-08-23 12:15:05,769 INFO         val   stamina_rmse by horizon: [0.0054 0.0047 0.0046 0.0046 0.0047], mean: 0.0048
2026-08-23 12:15:49,978 INFO   autoencode pretrain epoch 5/5: train_loss=0.0000  val_loss=0.0000
2026-08-23 12:15:49,979 INFO         train pos_rmse     by horizon (m): [0.1946 0.156  0.1564 0.1551 0.1691], mean: 0.1662 m
2026-08-23 12:15:49,979 INFO         train pos_dist     by horizon (m): [0.2433 0.2082 0.212  0.2103 0.2135], mean: 0.2175 m
2026-08-23 12:15:49,979 INFO         train vel_rmse     by horizon (m/s): [0.078  0.0598 0.0566 0.0562 0.0678], mean: 0.0637 m/s
2026-08-23 12:15:49,979 INFO         train vel_dist     by horizon (m/s): [0.0931 0.0726 0.0697 0.0698 0.0766], mean: 0.0764 m/s
2026-08-23 12:15:49,980 INFO         train heading_rmse by horizon: [0.0036 0.0029 0.0027 0.0028 0.0029], mean: 0.0030
2026-08-23 12:15:49,980 INFO         train heading_dist by horizon: [0.0027 0.0024 0.0023 0.0024 0.0023], mean: 0.0024
2026-08-23 12:15:49,980 INFO         train stamina_rmse by horizon: [0.0038 0.0031 0.003  0.0031 0.0032], mean: 0.0032
2026-08-23 12:15:49,980 INFO         val   pos_rmse     by horizon (m): [0.1042 0.0785 0.0762 0.0761 0.0876], mean: 0.0845 m
2026-08-23 12:15:49,980 INFO         val   pos_dist     by horizon (m): [0.1229 0.1035 0.1023 0.1022 0.1073], mean: 0.1076 m
2026-08-23 12:15:49,980 INFO         val   vel_rmse     by horizon (m/s): [0.052  0.0376 0.036  0.0362 0.0445], mean: 0.0413 m/s
2026-08-23 12:15:49,980 INFO         val   vel_dist     by horizon (m/s): [0.0608 0.0457 0.0443 0.0446 0.0496], mean: 0.0490 m/s
2026-08-23 12:15:49,980 INFO         val   heading_rmse by horizon: [0.0018 0.0012 0.0012 0.0012 0.0013], mean: 0.0013
2026-08-23 12:15:49,981 INFO         val   heading_dist by horizon: [0.0012 0.0009 0.0009 0.0009 0.001 ], mean: 0.0010
2026-08-23 12:15:49,981 INFO         val   stamina_rmse by horizon: [0.0016 0.001  0.0009 0.0009 0.001 ], mean: 0.0011
2026-08-23 12:15:49,983 INFO Autoencode pretraining: restored best-val weights (val_loss=0.0000)
2026-08-23 12:15:49,991 INFO Saved 'after_autoencode' checkpoint to checkpoints/physics_pretrain/player_encoder_14.after_autoencode.pt
2026-08-23 12:15:49,992 INFO Decoder-only pretraining: 20 epoch(s), lr=3.00e-04, optimizer=adam, freeze_latent=False
2026-08-23 12:17:46,498 INFO   decoder-only pretrain epoch 1/20: train_loss=0.4624  val_loss=0.2847  (improved by inf > min_delta=1.0e-06)
2026-08-23 12:17:46,498 INFO     crossing_head: train loss=0.0058 pos_dist=11.582m dt_mae=2.181s | val loss=0.0041 pos_dist=9.814m dt_mae=1.827s
2026-08-23 12:17:46,498 INFO     goal_dist_delta_head: train loss=0.01230 mae=(left 4.345m, right 4.704m) | val loss=0.00574 mae=(left 3.609m, right 3.616m)
2026-08-23 12:17:46,498 INFO     short_horizon_probes: train loss=0.01626 rmse_norm=(0.2s 0.0434, 1.0s 0.0506) | val loss=0.00088 rmse_norm=(0.2s 0.0155, 1.0s 0.0253)
2026-08-23 12:19:35,052 INFO   decoder-only pretrain epoch 2/20: train_loss=0.2613  val_loss=0.2473  (improved by 0.037387 > min_delta=1.0e-06)
2026-08-23 12:19:35,052 INFO     crossing_head: train loss=0.0050 pos_dist=10.370m dt_mae=2.065s | val loss=0.0041 pos_dist=9.616m dt_mae=1.883s
2026-08-23 12:19:35,052 INFO     goal_dist_delta_head: train loss=0.00554 mae=(left 3.552m, right 3.511m) | val loss=0.00549 mae=(left 3.521m, right 3.496m)
2026-08-23 12:19:35,052 INFO     short_horizon_probes: train loss=0.00061 rmse_norm=(0.2s 0.0121, 1.0s 0.0213) | val loss=0.00046 rmse_norm=(0.2s 0.0090, 1.0s 0.0195)
2026-08-23 12:21:27,551 INFO   decoder-only pretrain epoch 3/20: train_loss=0.2296  val_loss=0.2055  (improved by 0.041826 > min_delta=1.0e-06)
2026-08-23 12:21:27,551 INFO     crossing_head: train loss=0.0050 pos_dist=10.269m dt_mae=2.074s | val loss=0.0041 pos_dist=9.618m dt_mae=1.865s
2026-08-23 12:21:27,552 INFO     goal_dist_delta_head: train loss=0.00547 mae=(left 3.512m, right 3.493m) | val loss=0.00542 mae=(left 3.498m, right 3.454m)
2026-08-23 12:21:27,552 INFO     short_horizon_probes: train loss=0.00049 rmse_norm=(0.2s 0.0097, 1.0s 0.0197) | val loss=0.00051 rmse_norm=(0.2s 0.0103, 1.0s 0.0201)
2026-08-23 12:23:19,820 INFO   decoder-only pretrain epoch 4/20: train_loss=0.1690  val_loss=0.1343  (improved by 0.071235 > min_delta=1.0e-06)
2026-08-23 12:23:19,820 INFO     crossing_head: train loss=0.0050 pos_dist=10.262m dt_mae=2.060s | val loss=0.0041 pos_dist=9.913m dt_mae=1.860s
2026-08-23 12:23:19,820 INFO     goal_dist_delta_head: train loss=0.00540 mae=(left 3.498m, right 3.457m) | val loss=0.00537 mae=(left 3.489m, right 3.426m)
2026-08-23 12:23:19,820 INFO     short_horizon_probes: train loss=0.00046 rmse_norm=(0.2s 0.0089, 1.0s 0.0196) | val loss=0.00042 rmse_norm=(0.2s 0.0079, 1.0s 0.0190)
2026-08-23 12:24:40,995 INFO Loaded 1340 shard(s) from physics_pretrain_data/player
2026-08-23 12:24:41,119 INFO Dataset: 1,340,000 episodes (1,139,000 train / 201,000 val)
2026-08-23 12:24:41,377 INFO pos_weight (max cap: 1.0):
2026-08-23 12:24:41,377 INFO     t= 0.2s  out_of_bounds=1.00  goal_scored=1.00
2026-08-23 12:24:41,377 INFO     t= 1.0s  out_of_bounds=1.00  goal_scored=1.00
2026-08-23 12:24:41,377 INFO     t= 3.0s  out_of_bounds=1.00  goal_scored=1.00
2026-08-23 12:24:41,377 INFO     t= 5.0s  out_of_bounds=1.00  goal_scored=1.00
2026-08-23 12:24:41,377 INFO     t=10.0s  out_of_bounds=1.00  goal_scored=1.00
2026-08-23 12:24:45,783 INFO Latent diagnostics (50,000 rows, latent_dim=36):
    per-dim std: mean=0.0722  pooled=0.2165  |  latent norm: mean=1.2929 std=0.1541
    dead dims (std < threshold): 29/36  [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    off-diagonal |corr|: mean=0.0030  max=0.8161 (dims (2, 5))  redundant pairs (|corr|>threshold): 0
    effective rank: 3.58/36 (participation ratio)  95%-variance components: 5/36  condition number: 5e+11
    smallest-std dims: 7(std=0.0000,mean=0.0000), 14(std=0.0000,mean=0.0000), 13(std=0.0000,mean=0.0000), 12(std=0.0000,mean=0.0000), 11(std=0.0000,mean=0.0000)
    largest-std dims:  5(std=0.7061,mean=0.0080), 4(std=0.7047,mean=0.0007), 0(std=0.5183,mean=0.0025), 1(std=0.3134,mean=0.0040), 6(std=0.2891,mean=0.4973)
    most-correlated pairs: (5,2)=0.816, (3,4)=0.806, (0,2)=-0.064, (0,3)=0.061, (6,3)=0.022
2026-08-23 12:24:49,129 INFO Training row-count summary (train split):
    main (per-horizon heads)        : 1,139,000 rows -- own batches
    autoencode/t0 (bottleneck recon): 5,695,000 rows -- own batches (1,139,000 rows x 5 horizons)
    adjacent-pair (dynamics)        : 4,561,805/5,695,000 horizon-pass rows mask-eligible -- shares the horizon pass's own latent, no extra rows/batches
    crossing_head (at each horizon) : 1,258,236/5,695,000 horizon-pass rows mask-eligible -- shares the horizon pass's own latent, no extra rows/batches
    crossing_head (at t=0, in main) : 393,586/1,139,000 main rows masked-valid (position term only; delta_t trains unmasked on the -1 sentinel) -- shares main's own latent
    goal_dist_delta_head (main only): 1,139,000 main rows, unmasked -- shares main's own latent
    short-horizon probes (main only): 1,139,000 main rows x 2 heads, unmasked -- shares main's own latent
2026-08-23 12:24:49,129 INFO Autoencode pretraining: 5 epoch(s), lr=3.00e-04, optimizer=adam
2026-08-23 12:25:29,608 INFO   autoencode pretrain epoch 1/5: train_loss=0.0128  val_loss=0.0001
2026-08-23 12:25:29,609 INFO         train pos_rmse     by horizon (m): [1.2345 1.2439 1.3284 1.2093 1.5496], mean: 1.3131 m
2026-08-23 12:25:29,609 INFO         train pos_dist     by horizon (m): [1.5099 1.5617 1.6877 1.5299 1.8995], mean: 1.6377 m
2026-08-23 12:25:29,609 INFO         train vel_rmse     by horizon (m/s): [1.1142 1.0164 1.1224 1.0531 1.1027], mean: 1.0817 m/s
2026-08-23 12:25:29,610 INFO         train vel_dist     by horizon (m/s): [1.3597 1.2361 1.3465 1.2597 1.3375], mean: 1.3079 m/s
2026-08-23 12:25:29,610 INFO         train heading_rmse by horizon: [0.0209 0.0191 0.021  0.0172 0.0213], mean: 0.0199
2026-08-23 12:25:29,610 INFO         train heading_dist by horizon: [0.0185 0.02   0.0205 0.0139 0.0214], mean: 0.0189
2026-08-23 12:25:29,610 INFO         train stamina_rmse by horizon: [0.0171 0.0161 0.0168 0.0147 0.017 ], mean: 0.0163
2026-08-23 12:25:29,611 INFO         val   pos_rmse     by horizon (m): [0.469  0.4047 0.3944 0.3937 0.4156], mean: 0.4155 m
2026-08-23 12:25:29,611 INFO         val   pos_dist     by horizon (m): [0.582  0.531  0.5234 0.5226 0.5364], mean: 0.5391 m
2026-08-23 12:25:29,611 INFO         val   vel_rmse     by horizon (m/s): [0.5676 0.3021 0.2972 0.2916 0.2896], mean: 0.3496 m/s
2026-08-23 12:25:29,612 INFO         val   vel_dist     by horizon (m/s): [0.6667 0.3635 0.3522 0.3477 0.3465], mean: 0.4153 m/s
2026-08-23 12:25:29,612 INFO         val   heading_rmse by horizon: [0.0074 0.0059 0.0057 0.0056 0.0058], mean: 0.0061
2026-08-23 12:25:29,612 INFO         val   heading_dist by horizon: [0.0056 0.0048 0.0046 0.0046 0.0046], mean: 0.0048
2026-08-23 12:25:29,612 INFO         val   stamina_rmse by horizon: [0.0053 0.0036 0.0033 0.0033 0.0034], mean: 0.0038
2026-08-23 12:26:12,204 INFO   autoencode pretrain epoch 2/5: train_loss=0.0001  val_loss=0.0000
2026-08-23 12:26:12,204 INFO         train pos_rmse     by horizon (m): [0.3864 0.2833 0.2646 0.2599 0.297 ], mean: 0.2982 m
2026-08-23 12:26:12,205 INFO         train pos_dist     by horizon (m): [0.4628 0.3609 0.3424 0.3369 0.363 ], mean: 0.3732 m
2026-08-23 12:26:12,205 INFO         train vel_rmse     by horizon (m/s): [0.3332 0.1938 0.1814 0.1777 0.1813], mean: 0.2135 m/s
2026-08-23 12:26:12,205 INFO         train vel_dist     by horizon (m/s): [0.3966 0.2357 0.2221 0.2189 0.221 ], mean: 0.2589 m/s
2026-08-23 12:26:12,205 INFO         train heading_rmse by horizon: [0.007  0.0052 0.0053 0.005  0.0053], mean: 0.0056
2026-08-23 12:26:12,205 INFO         train heading_dist by horizon: [0.0052 0.0042 0.0043 0.0041 0.0042], mean: 0.0044
2026-08-23 12:26:12,205 INFO         train stamina_rmse by horizon: [0.0055 0.004  0.0039 0.0038 0.004 ], mean: 0.0043
2026-08-23 12:26:12,205 INFO         val   pos_rmse     by horizon (m): [0.2295 0.1667 0.1592 0.1596 0.183 ], mean: 0.1796 m
2026-08-23 12:26:12,206 INFO         val   pos_dist     by horizon (m): [0.2639 0.2103 0.2052 0.2064 0.2206], mean: 0.2213 m
2026-08-23 12:26:12,206 INFO         val   vel_rmse     by horizon (m/s): [0.1705 0.1072 0.0955 0.0947 0.1002], mean: 0.1136 m/s
2026-08-23 12:26:12,206 INFO         val   vel_dist     by horizon (m/s): [0.204  0.1282 0.1153 0.1145 0.119 ], mean: 0.1362 m/s
2026-08-23 12:26:12,206 INFO         val   heading_rmse by horizon: [0.0035 0.0021 0.0019 0.0018 0.002 ], mean: 0.0023
2026-08-23 12:26:12,206 INFO         val   heading_dist by horizon: [0.0022 0.0014 0.0013 0.0013 0.0014], mean: 0.0015
2026-08-23 12:26:12,206 INFO         val   stamina_rmse by horizon: [0.0031 0.0018 0.0017 0.0017 0.0018], mean: 0.0020
2026-08-23 12:26:54,901 INFO   autoencode pretrain epoch 3/5: train_loss=0.0001  val_loss=0.0000
2026-08-23 12:26:54,902 INFO         train pos_rmse     by horizon (m): [0.2727 0.2105 0.208  0.2016 0.2092], mean: 0.2204 m
2026-08-23 12:26:54,902 INFO         train pos_dist     by horizon (m): [0.3284 0.2726 0.2754 0.2665 0.2587], mean: 0.2803 m
2026-08-23 12:26:54,902 INFO         train vel_rmse     by horizon (m/s): [0.1455 0.1057 0.1021 0.1009 0.1044], mean: 0.1117 m/s
2026-08-23 12:26:54,902 INFO         train vel_dist     by horizon (m/s): [0.1784 0.1312 0.1289 0.1274 0.1272], mean: 0.1386 m/s
2026-08-23 12:26:54,902 INFO         train heading_rmse by horizon: [0.005  0.0039 0.004  0.0038 0.0038], mean: 0.0041
2026-08-23 12:26:54,902 INFO         train heading_dist by horizon: [0.0037 0.0032 0.0034 0.0032 0.0031], mean: 0.0033
2026-08-23 12:26:54,902 INFO         train stamina_rmse by horizon: [0.0043 0.003  0.0032 0.0032 0.0031], mean: 0.0033
2026-08-23 12:26:54,903 INFO         val   pos_rmse     by horizon (m): [0.168  0.1345 0.1339 0.1328 0.1433], mean: 0.1425 m
2026-08-23 12:26:54,903 INFO         val   pos_dist     by horizon (m): [0.1976 0.1774 0.1801 0.1793 0.1829], mean: 0.1835 m
2026-08-23 12:26:54,903 INFO         val   vel_rmse     by horizon (m/s): [0.0997 0.0672 0.0601 0.0593 0.064 ], mean: 0.0701 m/s
2026-08-23 12:26:54,903 INFO         val   vel_dist     by horizon (m/s): [0.1188 0.0799 0.0726 0.0719 0.0733], mean: 0.0833 m/s
2026-08-23 12:26:54,903 INFO         val   heading_rmse by horizon: [0.0036 0.0032 0.0032 0.0032 0.0033], mean: 0.0033
2026-08-23 12:26:54,903 INFO         val   heading_dist by horizon: [0.0027 0.0027 0.0028 0.0028 0.0028], mean: 0.0028
2026-08-23 12:26:54,903 INFO         val   stamina_rmse by horizon: [0.0034 0.0017 0.0014 0.0014 0.0015], mean: 0.0019
2026-08-23 12:27:37,972 INFO   autoencode pretrain epoch 4/5: train_loss=0.0001  val_loss=0.0000
2026-08-23 12:27:37,972 INFO         train pos_rmse     by horizon (m): [0.2038 0.1722 0.164  0.1661 0.1796], mean: 0.1772 m
2026-08-23 12:27:37,973 INFO         train pos_dist     by horizon (m): [0.2474 0.2258 0.2188 0.2224 0.2263], mean: 0.2281 m
2026-08-23 12:27:37,973 INFO         train vel_rmse     by horizon (m/s): [0.1004 0.0815 0.0756 0.0775 0.0819], mean: 0.0834 m/s
2026-08-23 12:27:37,973 INFO         train vel_dist     by horizon (m/s): [0.1227 0.1017 0.0956 0.0985 0.0982], mean: 0.1033 m/s
2026-08-23 12:27:37,973 INFO         train heading_rmse by horizon: [0.004  0.0032 0.0032 0.0032 0.0033], mean: 0.0034
2026-08-23 12:27:37,973 INFO         train heading_dist by horizon: [0.0029 0.0027 0.0027 0.0027 0.0026], mean: 0.0027
2026-08-23 12:27:37,974 INFO         train stamina_rmse by horizon: [0.0034 0.0028 0.0027 0.0028 0.0027], mean: 0.0029
2026-08-23 12:27:37,974 INFO         val   pos_rmse     by horizon (m): [0.1258 0.0848 0.0785 0.0778 0.0956], mean: 0.0925 m
2026-08-23 12:27:37,974 INFO         val   pos_dist     by horizon (m): [0.1445 0.1058 0.1007 0.1006 0.1087], mean: 0.1121 m
2026-08-23 12:27:37,974 INFO         val   vel_rmse     by horizon (m/s): [0.0727 0.0546 0.0503 0.0501 0.0576], mean: 0.0571 m/s
2026-08-23 12:27:37,974 INFO         val   vel_dist     by horizon (m/s): [0.0855 0.0642 0.0599 0.0598 0.0641], mean: 0.0667 m/s
2026-08-23 12:27:37,974 INFO         val   heading_rmse by horizon: [0.002  0.001  0.0009 0.0009 0.0011], mean: 0.0012
2026-08-23 12:27:37,974 INFO         val   heading_dist by horizon: [0.001  0.0006 0.0006 0.0006 0.0006], mean: 0.0007
2026-08-23 12:27:37,974 INFO         val   stamina_rmse by horizon: [0.0018 0.0009 0.0007 0.0007 0.0009], mean: 0.0010
2026-08-23 12:28:20,964 INFO   autoencode pretrain epoch 5/5: train_loss=0.0000  val_loss=0.0000
2026-08-23 12:28:20,964 INFO         train pos_rmse     by horizon (m): [0.173  0.1379 0.1366 0.1389 0.1451], mean: 0.1463 m
2026-08-23 12:28:20,965 INFO         train pos_dist     by horizon (m): [0.2155 0.1824 0.1837 0.187  0.1849], mean: 0.1907 m
2026-08-23 12:28:20,965 INFO         train vel_rmse     by horizon (m/s): [0.0792 0.0624 0.0596 0.061  0.0669], mean: 0.0658 m/s
2026-08-23 12:28:20,965 INFO         train vel_dist     by horizon (m/s): [0.0971 0.0781 0.0757 0.0778 0.08  ], mean: 0.0817 m/s
2026-08-23 12:28:20,965 INFO         train heading_rmse by horizon: [0.0033 0.0026 0.0027 0.0025 0.0026], mean: 0.0027
2026-08-23 12:28:20,965 INFO         train heading_dist by horizon: [0.0025 0.0021 0.0023 0.0021 0.0021], mean: 0.0022
2026-08-23 12:28:20,965 INFO         train stamina_rmse by horizon: [0.0032 0.0024 0.0023 0.0024 0.0024], mean: 0.0025
2026-08-23 12:28:20,965 INFO         val   pos_rmse     by horizon (m): [0.11   0.1004 0.0998 0.099  0.1129], mean: 0.1044 m
2026-08-23 12:28:20,965 INFO         val   pos_dist     by horizon (m): [0.1332 0.1312 0.1333 0.1326 0.1385], mean: 0.1338 m
2026-08-23 12:28:20,966 INFO         val   vel_rmse     by horizon (m/s): [0.0574 0.0451 0.0421 0.0416 0.0527], mean: 0.0478 m/s
2026-08-23 12:28:20,966 INFO         val   vel_dist     by horizon (m/s): [0.0677 0.0539 0.0516 0.0511 0.0541], mean: 0.0557 m/s
2026-08-23 12:28:20,966 INFO         val   heading_rmse by horizon: [0.0029 0.003  0.003  0.003  0.0031], mean: 0.0030
2026-08-23 12:28:20,966 INFO         val   heading_dist by horizon: [0.0022 0.0026 0.0027 0.0027 0.0027], mean: 0.0026
2026-08-23 12:28:20,966 INFO         val   stamina_rmse by horizon: [0.0022 0.0022 0.0023 0.0023 0.0024], mean: 0.0023
2026-08-23 12:28:20,967 INFO Autoencode pretraining: restored best-val weights (val_loss=0.0000)
2026-08-23 12:28:20,973 INFO Saved 'after_autoencode' checkpoint to checkpoints/physics_pretrain/player_encoder_14.after_autoencode.pt
2026-08-23 12:30:43,673 INFO epoch 1/500: train_loss=0.6571  pair_loss=0.0704  t0_loss=0.0856  val_loss=0.4352  best=0.4352  (improved by inf > min_delta=1.0e-07)
2026-08-23 12:30:43,673 INFO     grad_norm: mean=0.946167 std=0.957665 min=0.251867 max=16.503901
2026-08-23 12:30:43,673 INFO     train_loss_delta (batch-to-batch): mean=-0.000681 std=0.027691 min=-0.252603 max=0.299407
2026-08-23 12:30:43,673 INFO     crossing_head: train loss=0.0068 pos_dist=13.413m dt_mae=2.293s | val loss=0.0044 pos_dist=9.056m dt_mae=2.135s
2026-08-23 12:30:43,673 INFO     goal_dist_delta_head: train loss=0.01636 mae=(left 5.519m, right 5.476m) | val loss=0.00540 mae=(left 3.054m, right 3.752m)
2026-08-23 12:30:43,673 INFO     short_horizon_probes: train loss=0.04749 rmse_norm=(0.2s 0.1035, 1.0s 0.0972) | val loss=0.00342 rmse_norm=(0.2s 0.0443, 1.0s 0.0382)
2026-08-23 12:30:43,673 INFO     val_loss_delta (epoch-over-epoch): nan
2026-08-23 12:30:43,674 INFO         train pos_rmse     by horizon (m): [ 5.5739  4.8493  3.2663  5.0477 15.3452], mean: 6.8165 m
2026-08-23 12:30:43,674 INFO         val   pos_rmse     by horizon (m): [ 5.7115  4.77    3.3685  3.8549 12.2761], mean: 5.9962 m
2026-08-23 12:30:43,674 INFO         train pos_dist     by horizon (m): [ 7.2456  6.2523  4.0065  5.9293 17.7881], mean: 8.2444 m
2026-08-23 12:30:43,674 INFO         val   pos_dist     by horizon (m): [ 7.3489  6.0287  4.2175  4.8241 14.4348], mean: 7.3708 m
2026-08-23 12:30:43,674 INFO         train vel_rmse     by horizon (m/s): [1.6561 1.625  1.8619 1.8458 2.0725], mean: 1.8122 m/s
2026-08-23 12:30:43,674 INFO         val   vel_rmse     by horizon (m/s): [1.3196 1.2188 1.3167 1.214  1.2743], mean: 1.2687 m/s
2026-08-23 12:30:43,674 INFO         train vel_dist     by horizon (m/s): [2.0288 1.9969 2.1607 2.1752 2.5644], mean: 2.1852 m/s
2026-08-23 12:30:43,674 INFO         val   vel_dist     by horizon (m/s): [1.5006 1.4761 1.4822 1.3904 1.5385], mean: 1.4775 m/s
2026-08-23 12:30:43,675 INFO         train heading_rmse by horizon: [0.552  0.2779 0.2144 0.194  0.1717], mean: 0.2820
2026-08-23 12:30:43,675 INFO         val   heading_rmse by horizon: [0.4922 0.2622 0.1804 0.1333 0.1054], mean: 0.2347
2026-08-23 12:30:43,675 INFO         train heading_dist by horizon: [0.7899 0.293  0.2244 0.2039 0.1847], mean: 0.3392
2026-08-23 12:30:43,675 INFO         val   heading_dist by horizon: [0.7109 0.2563 0.1534 0.1033 0.0815], mean: 0.2611
2026-08-23 12:30:43,675 INFO         train stamina_rmse by horizon: [0.0174 0.0139 0.0087 0.0114 0.0314], mean: 0.0166
2026-08-23 12:30:43,675 INFO         val   stamina_rmse by horizon: [0.0142 0.0111 0.0051 0.0077 0.0266], mean: 0.0129
2026-08-23 12:30:43,675 INFO     train pos       R2 by horizon: [0.956 0.967 0.985 0.967 0.815], mean: 0.938
2026-08-23 12:30:43,675 INFO     val   pos       R2 by horizon: [0.955 0.969 0.985 0.983 0.886], mean: 0.955
2026-08-23 12:30:43,676 INFO     train pos       %-of-persistence by horizon: [1364.9  252.    50.9   45.8   63.8]
2026-08-23 12:30:43,676 INFO     val   pos       %-of-persistence by horizon: [1390.2  246.3   51.2   33.1   50.1]
2026-08-23 12:30:43,676 INFO     train vel       R2 by horizon: [0.241 0.457 0.468 0.456 0.186], mean: 0.362
2026-08-23 12:30:43,676 INFO     val   vel       R2 by horizon: [0.573 0.726 0.751 0.786 0.754], mean: 0.718
2026-08-23 12:30:43,676 INFO     train vel       %-of-persistence by horizon: [209.5  62.6  56.3  56.6  68.7]
2026-08-23 12:30:43,676 INFO     val   vel       %-of-persistence by horizon: [157.   44.5  38.5  35.5  37.8]
2026-08-23 12:30:43,676 INFO     train heading   R2 by horizon: [0.386 0.825 0.874 0.888 0.9  ], mean: 0.775
2026-08-23 12:30:43,676 INFO     val   heading   R2 by horizon: [0.515 0.862 0.935 0.964 0.978], mean: 0.851
2026-08-23 12:30:43,677 INFO     train heading   %-of-persistence by horizon: [113.5  31.9  25.1  23.7  22.3]
2026-08-23 12:30:43,677 INFO     val   heading   %-of-persistence by horizon: [100.8  28.2  18.1  13.3  10.5]
2026-08-23 12:30:43,677 INFO     train stamina   R2 by horizon: [0.996 0.997 0.999 0.998 0.987], mean: 0.995
2026-08-23 12:30:43,677 INFO     val   stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.997
2026-08-23 12:30:43,677 INFO     train stamina   %-of-persistence by horizon: [2047.3  331.5   76.8   56.2   76.2]
2026-08-23 12:30:43,677 INFO     val   stamina   %-of-persistence by horizon: [1606.5  250.9   38.3   35.2   61.5]
2026-08-23 12:30:43,682 INFO Saved 'midtrain_latest' checkpoint to checkpoints/physics_pretrain/player_encoder_14.midtrain_latest.pt
2026-08-23 12:33:13,613 INFO epoch 2/500: train_loss=0.3716  pair_loss=0.0423  t0_loss=0.0560  val_loss=0.3185  best=0.3185  (improved by 0.116698 > min_delta=1.0e-07)
2026-08-23 12:33:13,613 INFO     grad_norm: mean=1.067919 std=0.522740 min=0.303842 max=5.379900
2026-08-23 12:33:13,613 INFO     train_loss_delta (batch-to-batch): mean=-0.000015 std=0.014573 min=-0.059591 max=0.051801
2026-08-23 12:33:13,613 INFO     crossing_head: train loss=0.0052 pos_dist=9.618m dt_mae=2.268s | val loss=0.0043 pos_dist=8.783m dt_mae=2.123s
2026-08-23 12:33:13,613 INFO     goal_dist_delta_head: train loss=0.00481 mae=(left 2.975m, right 3.348m) | val loss=0.00452 mae=(left 2.934m, right 3.148m)
2026-08-23 12:33:13,613 INFO     short_horizon_probes: train loss=0.00222 rmse_norm=(0.2s 0.0356, 1.0s 0.0304) | val loss=0.00148 rmse_norm=(0.2s 0.0293, 1.0s 0.0248)
2026-08-23 12:33:13,613 INFO     val_loss_delta (epoch-over-epoch): -0.116698
2026-08-23 12:33:13,613 INFO         train pos_rmse     by horizon (m): [ 5.8646  5.1083  3.9471  3.9198 12.9113], mean: 6.3502 m
2026-08-23 12:33:13,613 INFO         val   pos_rmse     by horizon (m): [ 5.9812  5.3539  4.2179  3.9222 13.8603], mean: 6.6671 m
2026-08-23 12:33:13,614 INFO         train pos_dist     by horizon (m): [ 7.4968  6.3942  5.0091  4.9607 15.0802], mean: 7.7882 m
2026-08-23 12:33:13,614 INFO         val   pos_dist     by horizon (m): [ 7.591   6.6866  5.3645  4.9504 16.1174], mean: 8.1420 m
2026-08-23 12:33:13,614 INFO         train vel_rmse     by horizon (m/s): [1.2267 1.1745 1.1479 1.0238 0.9967], mean: 1.1139 m/s
2026-08-23 12:33:13,614 INFO         val   vel_rmse     by horizon (m/s): [1.1099 1.1384 1.0605 0.925  0.893 ], mean: 1.0254 m/s
2026-08-23 12:33:13,614 INFO         train vel_dist     by horizon (m/s): [1.3807 1.4158 1.2968 1.1853 1.181 ], mean: 1.2919 m/s
2026-08-23 12:33:13,614 INFO         val   vel_dist     by horizon (m/s): [1.2708 1.3633 1.2021 1.0853 1.0522], mean: 1.1947 m/s
2026-08-23 12:33:13,614 INFO         train heading_rmse by horizon: [0.4319 0.267  0.1573 0.1025 0.0883], mean: 0.2094
2026-08-23 12:33:13,614 INFO         val   heading_rmse by horizon: [0.3723 0.273  0.1349 0.076  0.0694], mean: 0.1851
2026-08-23 12:33:13,614 INFO         train heading_dist by horizon: [0.5829 0.2565 0.1212 0.0758 0.0641], mean: 0.2201
2026-08-23 12:33:13,615 INFO         val   heading_dist by horizon: [0.4531 0.253  0.0907 0.0522 0.0458], mean: 0.1790
2026-08-23 12:33:13,615 INFO         train stamina_rmse by horizon: [0.0139 0.0108 0.005  0.0076 0.0262], mean: 0.0127
2026-08-23 12:33:13,615 INFO         val   stamina_rmse by horizon: [0.0136 0.0105 0.0051 0.0078 0.0263], mean: 0.0126
2026-08-23 12:33:13,615 INFO     train pos       R2 by horizon: [0.952 0.964 0.979 0.982 0.874], mean: 0.950
2026-08-23 12:33:13,615 INFO     val   pos       R2 by horizon: [0.95  0.96  0.977 0.982 0.855], mean: 0.945
2026-08-23 12:33:13,615 INFO     train pos       %-of-persistence by horizon: [1428.1  264.1   60.2   33.6   52.7]
2026-08-23 12:33:13,615 INFO     val   pos       %-of-persistence by horizon: [1455.9  276.4   64.1   33.6   56.5]
2026-08-23 12:33:13,616 INFO     train vel       R2 by horizon: [0.63  0.745 0.81  0.847 0.848], mean: 0.776
2026-08-23 12:33:13,616 INFO     val   vel       R2 by horizon: [0.698 0.761 0.839 0.876 0.879], mean: 0.810
2026-08-23 12:33:13,616 INFO     train vel       %-of-persistence by horizon: [146.1  42.9  33.7  30.1  29.7]
2026-08-23 12:33:13,616 INFO     val   vel       %-of-persistence by horizon: [132.1  41.6  31.   27.1  26.5]
2026-08-23 12:33:13,616 INFO     train heading   R2 by horizon: [0.625 0.857 0.95  0.978 0.984], mean: 0.879
2026-08-23 12:33:13,616 INFO     val   heading   R2 by horizon: [0.723 0.851 0.964 0.988 0.99 ], mean: 0.903
2026-08-23 12:33:13,616 INFO     train heading   %-of-persistence by horizon: [88.7 28.7 15.8 10.4  8.9]
2026-08-23 12:33:13,616 INFO     val   heading   %-of-persistence by horizon: [76.3 29.4 13.5  7.6  6.9]
2026-08-23 12:33:13,616 INFO     train stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.997
2026-08-23 12:33:13,617 INFO     val   stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.997
2026-08-23 12:33:13,617 INFO     train stamina   %-of-persistence by horizon: [1566.7  244.3   38.2   34.7   60.4]
2026-08-23 12:33:13,617 INFO     val   stamina   %-of-persistence by horizon: [1534.3  237.3   38.6   35.5   60.6]
2026-08-23 12:33:13,622 INFO Saved 'midtrain_latest' checkpoint to checkpoints/physics_pretrain/player_encoder_14.midtrain_latest.pt
