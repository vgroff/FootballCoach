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
