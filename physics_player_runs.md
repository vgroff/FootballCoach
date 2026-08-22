2026-08-22 17:34:15,017 INFO Loaded 610 shard(s) from physics_pretrain_data/player
2026-08-22 17:34:15,064 INFO Dataset: 610,000 episodes (518,500 train / 91,500 val)
2026-08-22 17:34:15,154 INFO pos_weight (max cap: 1.0):
2026-08-22 17:34:15,154 INFO     t= 0.2s  out_of_bounds=1.00  goal_scored=1.00
2026-08-22 17:34:15,155 INFO     t= 1.0s  out_of_bounds=1.00  goal_scored=1.00
2026-08-22 17:34:15,155 INFO     t= 3.0s  out_of_bounds=1.00  goal_scored=1.00
2026-08-22 17:34:15,155 INFO     t= 5.0s  out_of_bounds=1.00  goal_scored=1.00
2026-08-22 17:34:15,155 INFO     t=10.0s  out_of_bounds=1.00  goal_scored=1.00
2026-08-22 17:34:17,400 INFO Resumed full model (encoder+decoder) from checkpoints/physics_pretrain/player_encoder_10.midtrain_latest.pt (phase=midtrain_latest)
2026-08-22 17:34:18,625 INFO Training row-count summary (train split):
    main (per-horizon heads)        : 518,500 rows -- own batches
    autoencode/t0 (bottleneck recon): 2,592,500 rows -- own batches (518,500 rows x 5 horizons)
    adjacent-pair (dynamics)        : 2,077,175/2,592,500 horizon-pass rows mask-eligible -- shares the horizon pass's own latent, no extra rows/batches
    crossing_head (at each horizon) : 573,299/2,592,500 horizon-pass rows mask-eligible -- shares the horizon pass's own latent, no extra rows/batches
    crossing_head (at t=0, in main) : 179,133/518,500 main rows masked-valid (position term only; delta_t trains unmasked on the -1 sentinel) -- shares main's own latent
    goal_dist_delta_head (main only): 518,500 main rows, unmasked -- shares main's own latent
    short-horizon probes (main only): 518,500 main rows x 2 heads, unmasked -- shares main's own latent
2026-08-22 17:34:18,625 INFO Autoencode pretraining: 1 epoch(s), lr=3.00e-06, optimizer=adam
2026-08-22 17:34:24,180 INFO   autoencode pretrain epoch 1/1: train_loss=0.0012  val_loss=0.0008
2026-08-22 17:34:24,181 INFO         train pos_rmse     by horizon (m): [0.9056 0.6696 0.4429 0.4209 0.4327], mean: 0.5743 m
2026-08-22 17:34:24,181 INFO         train pos_dist     by horizon (m): [0.9944 0.6641 0.4953 0.4891 0.4929], mean: 0.6272 m
2026-08-22 17:34:24,181 INFO         train vel_rmse     by horizon (m/s): [1.2769 0.5835 0.5051 0.485  0.4428], mean: 0.6587 m/s
2026-08-22 17:34:24,182 INFO         train vel_dist     by horizon (m/s): [1.4953 0.5897 0.535  0.521  0.4882], mean: 0.7258 m/s
2026-08-22 17:34:24,182 INFO         train heading_rmse by horizon: [0.0623 0.0213 0.0095 0.0084 0.0087], mean: 0.0221
2026-08-22 17:34:24,182 INFO         train heading_dist by horizon: [0.0419 0.0107 0.0057 0.0056 0.0058], mean: 0.0139
2026-08-22 17:34:24,182 INFO         train stamina_rmse by horizon: [0.0082 0.0059 0.0053 0.0055 0.0059], mean: 0.0062
2026-08-22 17:34:24,183 INFO         val   pos_rmse     by horizon (m): [0.8386 0.5906 0.3799 0.3566 0.3716], mean: 0.5075 m
2026-08-22 17:34:24,183 INFO         val   pos_dist     by horizon (m): [0.9313 0.5887 0.4334 0.4243 0.4298], mean: 0.5615 m
2026-08-22 17:34:24,183 INFO         val   vel_rmse     by horizon (m/s): [1.2746 0.5648 0.4812 0.4555 0.4058], mean: 0.6364 m/s
2026-08-22 17:34:24,183 INFO         val   vel_dist     by horizon (m/s): [1.4914 0.5634 0.5019 0.4823 0.4448], mean: 0.6968 m/s
2026-08-22 17:34:24,183 INFO         val   heading_rmse by horizon: [0.0499 0.0186 0.0087 0.0077 0.008 ], mean: 0.0186
2026-08-22 17:34:24,184 INFO         val   heading_dist by horizon: [0.0335 0.0097 0.0053 0.0052 0.0055], mean: 0.0118
2026-08-22 17:34:24,184 INFO         val   stamina_rmse by horizon: [0.006  0.0038 0.0033 0.0034 0.0039], mean: 0.0041
2026-08-22 17:34:24,185 INFO Autoencode pretraining: restored best-val weights (val_loss=0.0008)
2026-08-22 17:34:24,188 INFO Saved 'after_autoencode' checkpoint to checkpoints/physics_pretrain/player_encoder_11.after_autoencode.pt
2026-08-22 17:34:24,189 INFO Decoder-only pretraining: 55 epoch(s), lr=2.00e-05, optimizer=adam, freeze_latent=False
2026-08-22 17:34:49,595 INFO   decoder-only pretrain epoch 1/55: train_loss=0.0159  val_loss=0.0136
2026-08-22 17:34:49,595 INFO     crossing_head: train loss=8.5066 pos_dist=23.463m dt_mae=2.279s | val loss=6.5602 pos_dist=18.293m dt_mae=2.021s
2026-08-22 17:34:49,595 INFO     goal_dist_delta_head: train loss=0.01195 mae=(left 4.664m, right 6.092m) | val loss=0.00987 mae=(left 4.308m, right 5.436m)
2026-08-22 17:34:49,595 INFO     short_horizon_probes: train loss=0.10654 rmse_norm=(0.2s 0.2367, 1.0s 0.2240) | val loss=0.08735 rmse_norm=(0.2s 0.2160, 1.0s 0.2017)
2026-08-22 17:35:22,099 INFO   decoder-only pretrain epoch 2/55: train_loss=0.0131  val_loss=0.0128
2026-08-22 17:35:22,099 INFO     crossing_head: train loss=8.3870 pos_dist=17.981m dt_mae=2.262s | val loss=6.4673 pos_dist=14.090m dt_mae=2.003s
2026-08-22 17:35:22,099 INFO     goal_dist_delta_head: train loss=0.00855 mae=(left 4.105m, right 4.902m) | val loss=0.00745 mae=(left 3.926m, right 4.443m)
2026-08-22 17:35:22,099 INFO     short_horizon_probes: train loss=0.07270 rmse_norm=(0.2s 0.1982, 1.0s 0.1821) | val loss=0.05894 rmse_norm=(0.2s 0.1805, 1.0s 0.1624)
2026-08-22 17:35:58,407 INFO   decoder-only pretrain epoch 3/55: train_loss=0.0126  val_loss=0.0126
2026-08-22 17:35:58,408 INFO     crossing_head: train loss=8.3087 pos_dist=14.753m dt_mae=2.254s | val loss=6.4023 pos_dist=12.342m dt_mae=1.982s
2026-08-22 17:35:58,408 INFO     goal_dist_delta_head: train loss=0.00667 mae=(left 3.778m, right 4.060m) | val loss=0.00601 mae=(left 3.639m, right 3.748m)
2026-08-22 17:35:58,408 INFO     short_horizon_probes: train loss=0.04865 rmse_norm=(0.2s 0.1652, 1.0s 0.1455) | val loss=0.03915 rmse_norm=(0.2s 0.1502, 1.0s 0.1288)
2026-08-22 17:36:33,970 INFO   decoder-only pretrain epoch 4/55: train_loss=0.0124  val_loss=0.0124
2026-08-22 17:36:33,970 INFO     crossing_head: train loss=8.2529 pos_dist=13.273m dt_mae=2.242s | val loss=6.3554 pos_dist=11.659m dt_mae=1.965s
2026-08-22 17:36:33,970 INFO     goal_dist_delta_head: train loss=0.00558 mae=(left 3.525m, right 3.501m) | val loss=0.00519 mae=(left 3.414m, right 3.315m)
2026-08-22 17:36:33,970 INFO     short_horizon_probes: train loss=0.03227 rmse_norm=(0.2s 0.1374, 1.0s 0.1152) | val loss=0.02607 rmse_norm=(0.2s 0.1250, 1.0s 0.1022)
2026-08-22 17:37:06,871 INFO   decoder-only pretrain epoch 5/55: train_loss=0.0123  val_loss=0.0123
2026-08-22 17:37:06,872 INFO     crossing_head: train loss=8.2071 pos_dist=12.521m dt_mae=2.232s | val loss=6.3146 pos_dist=11.285m dt_mae=1.950s
2026-08-22 17:37:06,872 INFO     goal_dist_delta_head: train loss=0.00497 mae=(left 3.332m, right 3.175m) | val loss=0.00474 mae=(left 3.245m, right 3.079m)
2026-08-22 17:37:06,872 INFO     short_horizon_probes: train loss=0.02170 rmse_norm=(0.2s 0.1143, 1.0s 0.0925) | val loss=0.01782 rmse_norm=(0.2s 0.1040, 1.0s 0.0837)
2026-08-22 17:37:38,482 INFO   decoder-only pretrain epoch 6/55: train_loss=0.0122  val_loss=0.0123
2026-08-22 17:37:38,482 INFO     crossing_head: train loss=8.1691 pos_dist=12.086m dt_mae=2.221s | val loss=6.2787 pos_dist=11.043m dt_mae=1.937s
2026-08-22 17:37:38,482 INFO     goal_dist_delta_head: train loss=0.00463 mae=(left 3.186m, right 3.001m) | val loss=0.00448 mae=(left 3.117m, right 2.951m)
2026-08-22 17:37:38,482 INFO     short_horizon_probes: train loss=0.01517 rmse_norm=(0.2s 0.0953, 1.0s 0.0778) | val loss=0.01281 rmse_norm=(0.2s 0.0870, 1.0s 0.0724)
2026-08-22 17:38:10,156 INFO   decoder-only pretrain epoch 7/55: train_loss=0.0122  val_loss=0.0122
2026-08-22 17:38:10,157 INFO     crossing_head: train loss=8.1317 pos_dist=11.845m dt_mae=2.211s | val loss=6.2476 pos_dist=10.872m dt_mae=1.927s
2026-08-22 17:38:10,157 INFO     goal_dist_delta_head: train loss=0.00442 mae=(left 3.073m, right 2.900m) | val loss=0.00431 mae=(left 3.017m, right 2.871m)
2026-08-22 17:38:10,157 INFO     short_horizon_probes: train loss=0.01109 rmse_norm=(0.2s 0.0798, 1.0s 0.0685) | val loss=0.00947 rmse_norm=(0.2s 0.0728, 1.0s 0.0646)
2026-08-22 17:38:40,798 INFO   decoder-only pretrain epoch 8/55: train_loss=0.0122  val_loss=0.0122
2026-08-22 17:38:40,798 INFO     crossing_head: train loss=8.0998 pos_dist=11.589m dt_mae=2.206s | val loss=6.2104 pos_dist=10.742m dt_mae=1.906s
2026-08-22 17:38:40,798 INFO     goal_dist_delta_head: train loss=0.00426 mae=(left 2.982m, right 2.832m) | val loss=0.00417 mae=(left 2.935m, right 2.811m)
2026-08-22 17:38:40,798 INFO     short_horizon_probes: train loss=0.00824 rmse_norm=(0.2s 0.0670, 1.0s 0.0612) | val loss=0.00708 rmse_norm=(0.2s 0.0612, 1.0s 0.0577)
2026-08-22 17:39:12,509 INFO   decoder-only pretrain epoch 9/55: train_loss=0.0121  val_loss=0.0121
2026-08-22 17:39:12,509 INFO     crossing_head: train loss=8.0692 pos_dist=11.504m dt_mae=2.195s | val loss=6.1810 pos_dist=10.649m dt_mae=1.893s
2026-08-22 17:39:12,510 INFO     goal_dist_delta_head: train loss=0.00414 mae=(left 2.907m, right 2.778m) | val loss=0.00406 mae=(left 2.867m, right 2.761m)
2026-08-22 17:39:12,510 INFO     short_horizon_probes: train loss=0.00616 rmse_norm=(0.2s 0.0564, 1.0s 0.0544) | val loss=0.00530 rmse_norm=(0.2s 0.0518, 1.0s 0.0511)
2026-08-22 17:39:44,180 INFO   decoder-only pretrain epoch 10/55: train_loss=0.0121  val_loss=0.0122
2026-08-22 17:39:44,180 INFO     crossing_head: train loss=8.0390 pos_dist=11.426m dt_mae=2.187s | val loss=6.1506 pos_dist=10.583m dt_mae=1.876s
2026-08-22 17:39:44,180 INFO     goal_dist_delta_head: train loss=0.00404 mae=(left 2.844m, right 2.732m) | val loss=0.00397 mae=(left 2.807m, right 2.721m)
2026-08-22 17:39:44,180 INFO     short_horizon_probes: train loss=0.00463 rmse_norm=(0.2s 0.0480, 1.0s 0.0482) | val loss=0.00402 rmse_norm=(0.2s 0.0444, 1.0s 0.0452)
2026-08-22 17:40:16,107 INFO   decoder-only pretrain epoch 11/55: train_loss=0.0121  val_loss=0.0121
2026-08-22 17:40:16,107 INFO     crossing_head: train loss=8.0092 pos_dist=11.257m dt_mae=2.175s | val loss=6.1265 pos_dist=10.543m dt_mae=1.867s
2026-08-22 17:40:16,107 INFO     goal_dist_delta_head: train loss=0.00396 mae=(left 2.790m, right 2.694m) | val loss=0.00390 mae=(left 2.757m, right 2.684m)
2026-08-22 17:40:16,107 INFO     short_horizon_probes: train loss=0.00353 rmse_norm=(0.2s 0.0415, 1.0s 0.0424) | val loss=0.00308 rmse_norm=(0.2s 0.0388, 1.0s 0.0397)
2026-08-22 17:40:46,511 INFO   decoder-only pretrain epoch 12/55: train_loss=0.0121  val_loss=0.0121
2026-08-22 17:40:46,511 INFO     crossing_head: train loss=7.9792 pos_dist=11.230m dt_mae=2.168s | val loss=6.0996 pos_dist=10.500m dt_mae=1.852s
2026-08-22 17:40:46,512 INFO     goal_dist_delta_head: train loss=0.00389 mae=(left 2.743m, right 2.660m) | val loss=0.00383 mae=(left 2.714m, right 2.652m)
2026-08-22 17:40:46,512 INFO     short_horizon_probes: train loss=0.00272 rmse_norm=(0.2s 0.0364, 1.0s 0.0373) | val loss=0.00238 rmse_norm=(0.2s 0.0342, 1.0s 0.0348)
2026-08-22 17:41:20,538 INFO   decoder-only pretrain epoch 13/55: train_loss=0.0121  val_loss=0.0121
2026-08-22 17:41:20,538 INFO     crossing_head: train loss=7.9511 pos_dist=11.167m dt_mae=2.159s | val loss=6.0759 pos_dist=10.476m dt_mae=1.841s
2026-08-22 17:41:20,538 INFO     goal_dist_delta_head: train loss=0.00384 mae=(left 2.702m, right 2.631m) | val loss=0.00378 mae=(left 2.677m, right 2.623m)
2026-08-22 17:41:20,538 INFO     short_horizon_probes: train loss=0.00211 rmse_norm=(0.2s 0.0322, 1.0s 0.0327) | val loss=0.00186 rmse_norm=(0.2s 0.0303, 1.0s 0.0306)
2026-08-22 17:41:57,156 INFO   decoder-only pretrain epoch 14/55: train_loss=0.0121  val_loss=0.0121
2026-08-22 17:41:57,157 INFO     crossing_head: train loss=7.9208 pos_dist=11.083m dt_mae=2.151s | val loss=6.0505 pos_dist=10.434m dt_mae=1.826s
2026-08-22 17:41:57,158 INFO     goal_dist_delta_head: train loss=0.00379 mae=(left 2.666m, right 2.604m) | val loss=0.00374 mae=(left 2.642m, right 2.599m)
2026-08-22 17:41:57,158 INFO     short_horizon_probes: train loss=0.00166 rmse_norm=(0.2s 0.0288, 1.0s 0.0287) | val loss=0.00147 rmse_norm=(0.2s 0.0273, 1.0s 0.0269)
2026-08-22 17:42:20,659 INFO   decoder-only pretrain epoch 15/55: train_loss=0.0121  val_loss=0.0121
2026-08-22 17:42:20,659 INFO     crossing_head: train loss=7.8909 pos_dist=11.035m dt_mae=2.138s | val loss=6.0319 pos_dist=10.405m dt_mae=1.824s
2026-08-22 17:42:20,659 INFO     goal_dist_delta_head: train loss=0.00375 mae=(left 2.633m, right 2.581m) | val loss=0.00370 mae=(left 2.611m, right 2.579m)
2026-08-22 17:42:20,659 INFO     short_horizon_probes: train loss=0.00133 rmse_norm=(0.2s 0.0260, 1.0s 0.0254) | val loss=0.00119 rmse_norm=(0.2s 0.0248, 1.0s 0.0239)
2026-08-22 17:42:39,053 INFO   decoder-only pretrain epoch 16/55: train_loss=0.0121  val_loss=0.0121
2026-08-22 17:42:39,053 INFO     crossing_head: train loss=7.8597 pos_dist=11.035m dt_mae=2.130s | val loss=6.0104 pos_dist=10.370m dt_mae=1.814s
2026-08-22 17:42:39,053 INFO     goal_dist_delta_head: train loss=0.00372 mae=(left 2.605m, right 2.560m) | val loss=0.00367 mae=(left 2.585m, right 2.557m)
2026-08-22 17:42:39,053 INFO     short_horizon_probes: train loss=0.00109 rmse_norm=(0.2s 0.0239, 1.0s 0.0227) | val loss=0.00099 rmse_norm=(0.2s 0.0229, 1.0s 0.0215)
2026-08-22 17:42:59,691 INFO   decoder-only pretrain epoch 17/55: train_loss=0.0121  val_loss=0.0121
2026-08-22 17:42:59,691 INFO     crossing_head: train loss=7.8284 pos_dist=10.976m dt_mae=2.123s | val loss=5.9856 pos_dist=10.342m dt_mae=1.797s
2026-08-22 17:42:59,691 INFO     goal_dist_delta_head: train loss=0.00368 mae=(left 2.580m, right 2.542m) | val loss=0.00364 mae=(left 2.557m, right 2.542m)
2026-08-22 17:42:59,691 INFO     short_horizon_probes: train loss=0.00092 rmse_norm=(0.2s 0.0222, 1.0s 0.0205) | val loss=0.00084 rmse_norm=(0.2s 0.0214, 1.0s 0.0196)
2026-08-22 17:43:22,645 INFO   decoder-only pretrain epoch 18/55: train_loss=0.0121  val_loss=0.0121
2026-08-22 17:43:22,645 INFO     crossing_head: train loss=7.7967 pos_dist=10.952m dt_mae=2.111s | val loss=5.9675 pos_dist=10.307m dt_mae=1.793s
2026-08-22 17:43:22,645 INFO     goal_dist_delta_head: train loss=0.00366 mae=(left 2.557m, right 2.527m) | val loss=0.00361 mae=(left 2.539m, right 2.526m)
2026-08-22 17:43:22,645 INFO     short_horizon_probes: train loss=0.00079 rmse_norm=(0.2s 0.0209, 1.0s 0.0189) | val loss=0.00074 rmse_norm=(0.2s 0.0203, 1.0s 0.0181)
2026-08-22 17:43:56,381 INFO   decoder-only pretrain epoch 19/55: train_loss=0.0121  val_loss=0.0121
2026-08-22 17:43:56,381 INFO     crossing_head: train loss=7.7653 pos_dist=10.911m dt_mae=2.105s | val loss=5.9450 pos_dist=10.252m dt_mae=1.777s
2026-08-22 17:43:56,381 INFO     goal_dist_delta_head: train loss=0.00364 mae=(left 2.537m, right 2.512m) | val loss=0.00359 mae=(left 2.520m, right 2.511m)
2026-08-22 17:43:56,381 INFO     short_horizon_probes: train loss=0.00071 rmse_norm=(0.2s 0.0199, 1.0s 0.0176) | val loss=0.00066 rmse_norm=(0.2s 0.0194, 1.0s 0.0170)
2026-08-22 17:44:29,599 INFO   decoder-only pretrain epoch 20/55: train_loss=0.0121  val_loss=0.0121
2026-08-22 17:44:29,599 INFO     crossing_head: train loss=7.7340 pos_dist=10.883m dt_mae=2.095s | val loss=5.9272 pos_dist=10.221m dt_mae=1.769s
2026-08-22 17:44:29,599 INFO     goal_dist_delta_head: train loss=0.00362 mae=(left 2.519m, right 2.499m) | val loss=0.00357 mae=(left 2.502m, right 2.499m)
2026-08-22 17:44:29,599 INFO     short_horizon_probes: train loss=0.00064 rmse_norm=(0.2s 0.0191, 1.0s 0.0166) | val loss=0.00061 rmse_norm=(0.2s 0.0186, 1.0s 0.0161)
2026-08-22 17:45:02,232 INFO   decoder-only pretrain epoch 21/55: train_loss=0.0121  val_loss=0.0122
2026-08-22 17:45:02,233 INFO     crossing_head: train loss=7.7037 pos_dist=10.761m dt_mae=2.086s | val loss=5.9132 pos_dist=10.200m dt_mae=1.767s
2026-08-22 17:45:02,233 INFO     goal_dist_delta_head: train loss=0.00360 mae=(left 2.503m, right 2.488m) | val loss=0.00356 mae=(left 2.487m, right 2.489m)
2026-08-22 17:45:02,233 INFO     short_horizon_probes: train loss=0.00059 rmse_norm=(0.2s 0.0184, 1.0s 0.0158) | val loss=0.00057 rmse_norm=(0.2s 0.0181, 1.0s 0.0155)
2026-08-22 17:45:33,842 INFO   decoder-only pretrain epoch 22/55: train_loss=0.0121  val_loss=0.0121
2026-08-22 17:45:33,842 INFO     crossing_head: train loss=7.6749 pos_dist=10.816m dt_mae=2.078s | val loss=5.8990 pos_dist=10.157m dt_mae=1.762s
2026-08-22 17:45:33,842 INFO     goal_dist_delta_head: train loss=0.00359 mae=(left 2.490m, right 2.478m) | val loss=0.00354 mae=(left 2.473m, right 2.477m)
2026-08-22 17:45:33,842 INFO     short_horizon_probes: train loss=0.00055 rmse_norm=(0.2s 0.0178, 1.0s 0.0152) | val loss=0.00053 rmse_norm=(0.2s 0.0175, 1.0s 0.0149)
2026-08-22 17:45:33,842 INFO   decoder-only pretrain: early stopping after 22 epochs (patience=5)
2026-08-22 17:45:33,843 INFO Decoder-only pretraining: restored best-val weights (val_loss=0.0121)
2026-08-22 17:45:33,846 INFO Saved 'after_decoder_pretrain' checkpoint to checkpoints/physics_pretrain/player_encoder_11.after_decoder_pretrain.pt
2026-08-22 17:46:10,090 INFO epoch 1/500: train_loss=0.0101  pair_loss=0.0016  t0_loss=0.0021  val_loss=0.0101  best=0.0101  (improved)
2026-08-22 17:46:10,090 INFO     grad_norm: mean=0.265436 std=0.100750 min=0.095360 max=1.139233
2026-08-22 17:46:10,090 INFO     train_loss_delta (batch-to-batch): mean=-0.000001 std=0.000613 min=-0.003259 max=0.002940
2026-08-22 17:46:10,090 INFO     crossing_head: train loss=7.7097 pos_dist=10.941m dt_mae=2.090s | val loss=5.8842 pos_dist=10.226m dt_mae=1.762s
2026-08-22 17:46:10,090 INFO     goal_dist_delta_head: train loss=0.00367 mae=(left 2.555m, right 2.535m) | val loss=0.00361 mae=(left 2.533m, right 2.530m)
2026-08-22 17:46:10,090 INFO     short_horizon_probes: train loss=0.00081 rmse_norm=(0.2s 0.0211, 1.0s 0.0191) | val loss=0.00074 rmse_norm=(0.2s 0.0204, 1.0s 0.0180)
2026-08-22 17:46:10,090 INFO     val_loss_delta (epoch-over-epoch): nan
2026-08-22 17:46:10,091 INFO         train pos_rmse     by horizon (m): [1.0757 1.0122 1.848  1.739  1.5771], mean: 1.4504 m
2026-08-22 17:46:10,091 INFO         val   pos_rmse     by horizon (m): [1.0607 1.0064 1.8442 1.7048 1.5388], mean: 1.4310 m
2026-08-22 17:46:10,091 INFO         train pos_dist     by horizon (m): [1.2561 1.2108 2.2576 2.0441 1.8352], mean: 1.7208 m
2026-08-22 17:46:10,091 INFO         val   pos_dist     by horizon (m): [1.2403 1.2048 2.2537 1.9976 1.7854], mean: 1.6964 m
2026-08-22 17:46:10,091 INFO         train vel_rmse     by horizon (m/s): [1.312  0.6547 0.5208 0.4727 0.5808], mean: 0.7082 m/s
2026-08-22 17:46:10,091 INFO         val   vel_rmse     by horizon (m/s): [1.3146 0.6499 0.5135 0.4704 0.5721], mean: 0.7041 m/s
2026-08-22 17:46:10,091 INFO         train vel_dist     by horizon (m/s): [1.5351 0.7202 0.5893 0.5526 0.6525], mean: 0.8099 m/s
2026-08-22 17:46:10,091 INFO         val   vel_dist     by horizon (m/s): [1.5382 0.7125 0.5797 0.5509 0.6415], mean: 0.8046 m/s
2026-08-22 17:46:10,091 INFO         train heading_rmse by horizon: [0.0537 0.0416 0.0275 0.0075 0.0067], mean: 0.0274
2026-08-22 17:46:10,091 INFO         val   heading_rmse by horizon: [0.0539 0.0424 0.0271 0.0077 0.0063], mean: 0.0275
2026-08-22 17:46:10,092 INFO         train heading_dist by horizon: [0.039  0.0193 0.0077 0.0049 0.0049], mean: 0.0152
2026-08-22 17:46:10,092 INFO         val   heading_dist by horizon: [0.0389 0.0193 0.0074 0.0046 0.0045], mean: 0.0150
2026-08-22 17:46:10,092 INFO         train stamina_rmse by horizon: [0.0109 0.0099 0.0051 0.0079 0.0259], mean: 0.0119
2026-08-22 17:46:10,092 INFO         val   stamina_rmse by horizon: [0.0102 0.0092 0.0047 0.0081 0.0264], mean: 0.0117
2026-08-22 17:46:10,092 INFO     train pos       R2 by horizon: [0.998 0.999 0.996 0.996 0.998], mean: 0.997
2026-08-22 17:46:10,092 INFO     val   pos       R2 by horizon: [0.998 0.999 0.996 0.997 0.998], mean: 0.997
2026-08-22 17:46:10,092 INFO     train pos       %-of-persistence by horizon: [261.7  52.3  28.1  14.9   6.4]
2026-08-22 17:46:10,092 INFO     val   pos       %-of-persistence by horizon: [258.   52.   28.1  14.6   6.3]
2026-08-22 17:46:10,092 INFO     train vel       R2 by horizon: [0.579 0.921 0.961 0.967 0.949], mean: 0.875
2026-08-22 17:46:10,093 INFO     val   vel       R2 by horizon: [0.577 0.922 0.962 0.968 0.95 ], mean: 0.876
2026-08-22 17:46:10,093 INFO     train vel       %-of-persistence by horizon: [156.1  23.9  15.2  13.8  17.2]
2026-08-22 17:46:10,093 INFO     val   vel       %-of-persistence by horizon: [156.4  23.7  15.   13.8  16.9]
2026-08-22 17:46:10,093 INFO     train heading   R2 by horizon: [0.994 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:46:10,093 INFO     val   heading   R2 by horizon: [0.994 0.996 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:46:10,094 INFO     train heading   %-of-persistence by horizon: [11.   4.5  2.8  0.8  0.7]
2026-08-22 17:46:10,094 INFO     val   heading   %-of-persistence by horizon: [11.   4.6  2.8  0.8  0.6]
2026-08-22 17:46:10,094 INFO     train stamina   R2 by horizon: [0.999 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:46:10,094 INFO     val   stamina   R2 by horizon: [0.999 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:46:10,094 INFO     train stamina   %-of-persistence by horizon: [1231.2  223.3   38.5   35.9   59.9]
2026-08-22 17:46:10,095 INFO     val   stamina   %-of-persistence by horizon: [1155.5  209.3   35.5   37.2   61. ]
2026-08-22 17:46:10,098 INFO Saved 'midtrain_latest' checkpoint to checkpoints/physics_pretrain/player_encoder_11.midtrain_latest.pt
2026-08-22 17:46:47,293 INFO epoch 2/500: train_loss=0.0097  pair_loss=0.0016  t0_loss=0.0021  val_loss=0.0099  best=0.0099  (improved)
2026-08-22 17:46:47,293 INFO     grad_norm: mean=0.253777 std=0.093601 min=0.099228 max=0.986003
2026-08-22 17:46:47,293 INFO     train_loss_delta (batch-to-batch): mean=0.000000 std=0.000597 min=-0.002459 max=0.002331
2026-08-22 17:46:47,293 INFO     crossing_head: train loss=7.5890 pos_dist=10.861m dt_mae=2.061s | val loss=5.8292 pos_dist=10.161m dt_mae=1.753s
2026-08-22 17:46:47,293 INFO     goal_dist_delta_head: train loss=0.00364 mae=(left 2.530m, right 2.518m) | val loss=0.00359 mae=(left 2.509m, right 2.513m)
2026-08-22 17:46:47,293 INFO     short_horizon_probes: train loss=0.00070 rmse_norm=(0.2s 0.0199, 1.0s 0.0174) | val loss=0.00065 rmse_norm=(0.2s 0.0194, 1.0s 0.0167)
2026-08-22 17:46:47,293 INFO     val_loss_delta (epoch-over-epoch): -0.000189
2026-08-22 17:46:47,293 INFO         train pos_rmse     by horizon (m): [1.0647 1.0048 1.8325 1.6985 1.5331], mean: 1.4267 m
2026-08-22 17:46:47,293 INFO         val   pos_rmse     by horizon (m): [1.053  0.9969 1.8353 1.6769 1.5163], mean: 1.4157 m
2026-08-22 17:46:47,293 INFO         train pos_dist     by horizon (m): [1.2396 1.2052 2.2354 1.9927 1.7624], mean: 1.6871 m
2026-08-22 17:46:47,293 INFO         val   pos_dist     by horizon (m): [1.2318 1.1966 2.2406 1.9553 1.7367], mean: 1.6722 m
2026-08-22 17:46:47,294 INFO         train vel_rmse     by horizon (m/s): [1.317  0.6456 0.5119 0.4731 0.5734], mean: 0.7042 m/s
2026-08-22 17:46:47,294 INFO         val   vel_rmse     by horizon (m/s): [1.3132 0.6404 0.5082 0.476  0.5696], mean: 0.7015 m/s
2026-08-22 17:46:47,294 INFO         train vel_dist     by horizon (m/s): [1.541  0.7074 0.5785 0.5514 0.6394], mean: 0.8035 m/s
2026-08-22 17:46:47,294 INFO         val   vel_dist     by horizon (m/s): [1.5349 0.7013 0.5728 0.5565 0.6361], mean: 0.8003 m/s
2026-08-22 17:46:47,295 INFO         train heading_rmse by horizon: [0.0521 0.0408 0.0274 0.0071 0.0063], mean: 0.0267
2026-08-22 17:46:47,295 INFO         val   heading_rmse by horizon: [0.0535 0.0416 0.027  0.0075 0.0061], mean: 0.0271
2026-08-22 17:46:47,295 INFO         train heading_dist by horizon: [0.0376 0.0187 0.0074 0.0045 0.0045], mean: 0.0145
2026-08-22 17:46:47,295 INFO         val   heading_dist by horizon: [0.0385 0.0188 0.0071 0.0044 0.0043], mean: 0.0146
2026-08-22 17:46:47,295 INFO         train stamina_rmse by horizon: [0.0107 0.0097 0.0047 0.0077 0.0259], mean: 0.0117
2026-08-22 17:46:47,295 INFO         val   stamina_rmse by horizon: [0.0103 0.0093 0.0044 0.008  0.0264], mean: 0.0117
2026-08-22 17:46:47,296 INFO     train pos       R2 by horizon: [0.998 0.999 0.996 0.997 0.998], mean: 0.997
2026-08-22 17:46:47,296 INFO     val   pos       R2 by horizon: [0.998 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:46:47,296 INFO     train pos       %-of-persistence by horizon: [259.   51.9  27.9  14.6   6.3]
2026-08-22 17:46:47,296 INFO     val   pos       %-of-persistence by horizon: [256.1  51.5  27.9  14.4   6.2]
2026-08-22 17:46:47,296 INFO     train vel       R2 by horizon: [0.576 0.923 0.962 0.967 0.95 ], mean: 0.876
2026-08-22 17:46:47,296 INFO     val   vel       R2 by horizon: [0.578 0.924 0.963 0.967 0.951], mean: 0.877
2026-08-22 17:46:47,296 INFO     train vel       %-of-persistence by horizon: [156.7  23.6  15.   13.8  17. ]
2026-08-22 17:46:47,296 INFO     val   vel       %-of-persistence by horizon: [156.3  23.4  14.9  13.9  16.9]
2026-08-22 17:46:47,296 INFO     train heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:46:47,297 INFO     val   heading   R2 by horizon: [0.994 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:46:47,297 INFO     train heading   %-of-persistence by horizon: [10.7  4.4  2.8  0.7  0.6]
2026-08-22 17:46:47,297 INFO     val   heading   %-of-persistence by horizon: [11.   4.5  2.8  0.8  0.6]
2026-08-22 17:46:47,297 INFO     train stamina   R2 by horizon: [0.999 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:46:47,297 INFO     val   stamina   R2 by horizon: [0.999 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:46:47,297 INFO     train stamina   %-of-persistence by horizon: [1208.4  219.6   35.5   35.2   59.8]
2026-08-22 17:46:47,297 INFO     val   stamina   %-of-persistence by horizon: [1164.5  209.9   33.5   36.8   60.9]
2026-08-22 17:46:47,300 INFO Saved 'midtrain_latest' checkpoint to checkpoints/physics_pretrain/player_encoder_11.midtrain_latest.pt
2026-08-22 17:47:24,827 INFO epoch 3/500: train_loss=0.0096  pair_loss=0.0016  t0_loss=0.0020  val_loss=0.0096  best=0.0096  (improved)
2026-08-22 17:47:24,828 INFO     grad_norm: mean=0.262574 std=0.107469 min=0.083905 max=0.921271
2026-08-22 17:47:24,828 INFO     train_loss_delta (batch-to-batch): mean=0.000000 std=0.000608 min=-0.002035 max=0.001995
2026-08-22 17:47:24,828 INFO     crossing_head: train loss=7.4890 pos_dist=10.763m dt_mae=2.040s | val loss=5.7815 pos_dist=10.089m dt_mae=1.741s
2026-08-22 17:47:24,828 INFO     goal_dist_delta_head: train loss=0.00361 mae=(left 2.508m, right 2.504m) | val loss=0.00357 mae=(left 2.490m, right 2.502m)
2026-08-22 17:47:24,828 INFO     short_horizon_probes: train loss=0.00063 rmse_norm=(0.2s 0.0190, 1.0s 0.0163) | val loss=0.00060 rmse_norm=(0.2s 0.0186, 1.0s 0.0158)
2026-08-22 17:47:24,829 INFO     val_loss_delta (epoch-over-epoch): -0.000272
2026-08-22 17:47:24,829 INFO         train pos_rmse     by horizon (m): [1.058  0.9974 1.8285 1.6734 1.5296], mean: 1.4174 m
2026-08-22 17:47:24,829 INFO         val   pos_rmse     by horizon (m): [1.0473 0.9951 1.8174 1.6625 1.5148], mean: 1.4075 m
2026-08-22 17:47:24,829 INFO         train pos_dist     by horizon (m): [1.2293 1.1976 2.2297 1.959  1.7417], mean: 1.6715 m
2026-08-22 17:47:24,830 INFO         val   pos_dist     by horizon (m): [1.217  1.1959 2.216  1.9416 1.7206], mean: 1.6582 m
2026-08-22 17:47:24,830 INFO         train vel_rmse     by horizon (m/s): [1.3176 0.6409 0.5072 0.4748 0.5697], mean: 0.7020 m/s
2026-08-22 17:47:24,830 INFO         val   vel_rmse     by horizon (m/s): [1.3154 0.6367 0.5022 0.4726 0.5639], mean: 0.6982 m/s
2026-08-22 17:47:24,830 INFO         train vel_dist     by horizon (m/s): [1.541  0.6992 0.5718 0.5507 0.6316], mean: 0.7988 m/s
2026-08-22 17:47:24,830 INFO         val   vel_dist     by horizon (m/s): [1.5363 0.6928 0.5641 0.5485 0.6242], mean: 0.7932 m/s
2026-08-22 17:47:24,830 INFO         train heading_rmse by horizon: [0.0519 0.0405 0.0273 0.0069 0.0061], mean: 0.0266
2026-08-22 17:47:24,830 INFO         val   heading_rmse by horizon: [0.0518 0.041  0.027  0.0074 0.006 ], mean: 0.0266
2026-08-22 17:47:24,830 INFO         train heading_dist by horizon: [0.0374 0.0185 0.0072 0.0044 0.0044], mean: 0.0144
2026-08-22 17:47:24,831 INFO         val   heading_dist by horizon: [0.037  0.0183 0.0074 0.0045 0.0045], mean: 0.0144
2026-08-22 17:47:24,831 INFO         train stamina_rmse by horizon: [0.0107 0.0097 0.0046 0.0077 0.0259], mean: 0.0117
2026-08-22 17:47:24,831 INFO         val   stamina_rmse by horizon: [0.0106 0.0096 0.0044 0.0078 0.0261], mean: 0.0117
2026-08-22 17:47:24,831 INFO     train pos       R2 by horizon: [0.998 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:47:24,831 INFO     val   pos       R2 by horizon: [0.998 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:47:24,831 INFO     train pos       %-of-persistence by horizon: [257.4  51.5  27.8  14.4   6.2]
2026-08-22 17:47:24,832 INFO     val   pos       %-of-persistence by horizon: [254.7  51.4  27.6  14.3   6.2]
2026-08-22 17:47:24,832 INFO     train vel       R2 by horizon: [0.575 0.924 0.963 0.967 0.951], mean: 0.876
2026-08-22 17:47:24,832 INFO     val   vel       R2 by horizon: [0.577 0.925 0.964 0.968 0.952], mean: 0.877
2026-08-22 17:47:24,832 INFO     train vel       %-of-persistence by horizon: [156.8  23.4  14.8  13.9  16.9]
2026-08-22 17:47:24,832 INFO     val   vel       %-of-persistence by horizon: [156.5  23.2  14.7  13.8  16.7]
2026-08-22 17:47:24,832 INFO     train heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:47:24,832 INFO     val   heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:47:24,832 INFO     train heading   %-of-persistence by horizon: [10.6  4.4  2.8  0.7  0.6]
2026-08-22 17:47:24,833 INFO     val   heading   %-of-persistence by horizon: [10.6  4.4  2.8  0.8  0.6]
2026-08-22 17:47:24,833 INFO     train stamina   R2 by horizon: [0.999 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:47:24,833 INFO     val   stamina   R2 by horizon: [0.999 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:47:24,833 INFO     train stamina   %-of-persistence by horizon: [1215.   219.5   34.6   35.    59.9]
2026-08-22 17:47:24,833 INFO     val   stamina   %-of-persistence by horizon: [1201.2  217.3   33.4   35.8   60.3]
2026-08-22 17:47:24,836 INFO Saved 'midtrain_latest' checkpoint to checkpoints/physics_pretrain/player_encoder_11.midtrain_latest.pt
2026-08-22 17:48:02,647 INFO epoch 4/500: train_loss=0.0096  pair_loss=0.0016  t0_loss=0.0021  val_loss=0.0096  best=0.0096  (improved)
2026-08-22 17:48:02,648 INFO     grad_norm: mean=0.252670 std=0.101753 min=0.087613 max=1.043193
2026-08-22 17:48:02,648 INFO     train_loss_delta (batch-to-batch): mean=0.000000 std=0.000637 min=-0.002129 max=0.002342
2026-08-22 17:48:02,648 INFO     crossing_head: train loss=7.4057 pos_dist=10.633m dt_mae=2.022s | val loss=5.7522 pos_dist=10.028m dt_mae=1.745s
2026-08-22 17:48:02,648 INFO     goal_dist_delta_head: train loss=0.00360 mae=(left 2.492m, right 2.492m) | val loss=0.00355 mae=(left 2.474m, right 2.492m)
2026-08-22 17:48:02,648 INFO     short_horizon_probes: train loss=0.00058 rmse_norm=(0.2s 0.0183, 1.0s 0.0156) | val loss=0.00055 rmse_norm=(0.2s 0.0178, 1.0s 0.0152)
2026-08-22 17:48:02,648 INFO     val_loss_delta (epoch-over-epoch): -0.000048
2026-08-22 17:48:02,648 INFO         train pos_rmse     by horizon (m): [1.054  0.9917 1.8259 1.655  1.5308], mean: 1.4115 m
2026-08-22 17:48:02,648 INFO         val   pos_rmse     by horizon (m): [1.0506 0.9897 1.8161 1.6521 1.5172], mean: 1.4051 m
2026-08-22 17:48:02,648 INFO         train pos_dist     by horizon (m): [1.2218 1.1913 2.226  1.9344 1.7302], mean: 1.6608 m
2026-08-22 17:48:02,648 INFO         val   pos_dist     by horizon (m): [1.213  1.1836 2.217  1.9315 1.7135], mean: 1.6517 m
2026-08-22 17:48:02,648 INFO         train vel_rmse     by horizon (m/s): [1.3183 0.6377 0.5023 0.4737 0.5661], mean: 0.6996 m/s
2026-08-22 17:48:02,649 INFO         val   vel_rmse     by horizon (m/s): [1.3166 0.6347 0.4999 0.4693 0.561 ], mean: 0.6963 m/s
2026-08-22 17:48:02,649 INFO         train vel_dist     by horizon (m/s): [1.5409 0.6922 0.5639 0.5461 0.623 ], mean: 0.7932 m/s
2026-08-22 17:48:02,649 INFO         val   vel_dist     by horizon (m/s): [1.5375 0.6871 0.5595 0.5401 0.6149], mean: 0.7878 m/s
2026-08-22 17:48:02,650 INFO         train heading_rmse by horizon: [0.0516 0.0404 0.0273 0.0068 0.006 ], mean: 0.0264
2026-08-22 17:48:02,650 INFO         val   heading_rmse by horizon: [0.0514 0.0412 0.0269 0.0072 0.0057], mean: 0.0265
2026-08-22 17:48:02,650 INFO         train heading_dist by horizon: [0.037  0.0183 0.0071 0.0043 0.0043], mean: 0.0142
2026-08-22 17:48:02,651 INFO         val   heading_dist by horizon: [0.0367 0.0184 0.0068 0.004  0.0039], mean: 0.0140
2026-08-22 17:48:02,651 INFO         train stamina_rmse by horizon: [0.0109 0.0097 0.0045 0.0076 0.0259], mean: 0.0117
2026-08-22 17:48:02,651 INFO         val   stamina_rmse by horizon: [0.0107 0.0095 0.0043 0.0078 0.0262], mean: 0.0117
2026-08-22 17:48:02,652 INFO     train pos       R2 by horizon: [0.998 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:48:02,652 INFO     val   pos       R2 by horizon: [0.998 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:48:02,652 INFO     train pos       %-of-persistence by horizon: [256.4  51.2  27.8  14.2   6.3]
2026-08-22 17:48:02,652 INFO     val   pos       %-of-persistence by horizon: [255.5  51.1  27.6  14.2   6.2]
2026-08-22 17:48:02,652 INFO     train vel       R2 by horizon: [0.575 0.925 0.964 0.967 0.951], mean: 0.876
2026-08-22 17:48:02,652 INFO     val   vel       R2 by horizon: [0.576 0.925 0.964 0.968 0.952], mean: 0.877
2026-08-22 17:48:02,653 INFO     train vel       %-of-persistence by horizon: [156.9  23.3  14.7  13.9  16.8]
2026-08-22 17:48:02,653 INFO     val   vel       %-of-persistence by horizon: [156.7  23.2  14.6  13.7  16.6]
2026-08-22 17:48:02,653 INFO     train heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:48:02,653 INFO     val   heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:48:02,653 INFO     train heading   %-of-persistence by horizon: [10.6  4.3  2.8  0.7  0.6]
2026-08-22 17:48:02,653 INFO     val   heading   %-of-persistence by horizon: [10.5  4.4  2.7  0.8  0.6]
2026-08-22 17:48:02,653 INFO     train stamina   R2 by horizon: [0.999 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:48:02,653 INFO     val   stamina   R2 by horizon: [0.999 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:48:02,653 INFO     train stamina   %-of-persistence by horizon: [1226.9  220.2   34.1   34.9   59.9]
2026-08-22 17:48:02,653 INFO     val   stamina   %-of-persistence by horizon: [1208.6  215.2   32.4   35.7   60.5]
2026-08-22 17:48:02,658 INFO Saved 'midtrain_latest' checkpoint to checkpoints/physics_pretrain/player_encoder_11.midtrain_latest.pt
2026-08-22 17:48:38,695 INFO epoch 5/500: train_loss=0.0095  pair_loss=0.0016  t0_loss=0.0021  val_loss=0.0095  best=0.0095  (improved)
2026-08-22 17:48:38,695 INFO     grad_norm: mean=0.254027 std=0.106729 min=0.105899 max=1.050331
2026-08-22 17:48:38,695 INFO     train_loss_delta (batch-to-batch): mean=-0.000000 std=0.000620 min=-0.001858 max=0.002372
2026-08-22 17:48:38,696 INFO     crossing_head: train loss=7.3426 pos_dist=10.646m dt_mae=2.010s | val loss=5.7200 pos_dist=9.941m dt_mae=1.736s
2026-08-22 17:48:38,696 INFO     goal_dist_delta_head: train loss=0.00358 mae=(left 2.478m, right 2.481m) | val loss=0.00354 mae=(left 2.461m, right 2.483m)
2026-08-22 17:48:38,696 INFO     short_horizon_probes: train loss=0.00053 rmse_norm=(0.2s 0.0176, 1.0s 0.0150) | val loss=0.00051 rmse_norm=(0.2s 0.0172, 1.0s 0.0147)
2026-08-22 17:48:38,696 INFO     val_loss_delta (epoch-over-epoch): -0.000095
2026-08-22 17:48:38,696 INFO         train pos_rmse     by horizon (m): [1.0494 0.9878 1.8221 1.644  1.5309], mean: 1.4069 m
2026-08-22 17:48:38,696 INFO         val   pos_rmse     by horizon (m): [1.0492 0.9898 1.8146 1.6372 1.5193], mean: 1.4020 m
2026-08-22 17:48:38,696 INFO         train pos_dist     by horizon (m): [1.2138 1.187  2.2198 1.9204 1.7219], mean: 1.6526 m
2026-08-22 17:48:38,696 INFO         val   pos_dist     by horizon (m): [1.2106 1.1902 2.2117 1.9099 1.71  ], mean: 1.6465 m
2026-08-22 17:48:38,696 INFO         train vel_rmse     by horizon (m/s): [1.3186 0.6351 0.4978 0.471  0.5634], mean: 0.6972 m/s
2026-08-22 17:48:38,696 INFO         val   vel_rmse     by horizon (m/s): [1.3169 0.63   0.4931 0.4699 0.5599], mean: 0.6940 m/s
2026-08-22 17:48:38,696 INFO         train vel_dist     by horizon (m/s): [1.5404 0.6858 0.5563 0.5402 0.616 ], mean: 0.7877 m/s
2026-08-22 17:48:38,697 INFO         val   vel_dist     by horizon (m/s): [1.5372 0.6808 0.5487 0.5421 0.6138], mean: 0.7845 m/s
2026-08-22 17:48:38,697 INFO         train heading_rmse by horizon: [0.0512 0.0402 0.0273 0.0068 0.0059], mean: 0.0263
2026-08-22 17:48:38,697 INFO         val   heading_rmse by horizon: [0.051  0.0407 0.0269 0.0072 0.0057], mean: 0.0263
2026-08-22 17:48:38,697 INFO         train heading_dist by horizon: [0.0367 0.0181 0.007  0.0042 0.0042], mean: 0.0141
2026-08-22 17:48:38,697 INFO         val   heading_dist by horizon: [0.0365 0.0181 0.0067 0.0041 0.0039], mean: 0.0139
2026-08-22 17:48:38,697 INFO         train stamina_rmse by horizon: [0.011  0.0097 0.0044 0.0076 0.0259], mean: 0.0117
2026-08-22 17:48:38,698 INFO         val   stamina_rmse by horizon: [0.0108 0.0096 0.0043 0.0078 0.0261], mean: 0.0117
2026-08-22 17:48:38,698 INFO     train pos       R2 by horizon: [0.998 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:48:38,698 INFO     val   pos       R2 by horizon: [0.998 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:48:38,698 INFO     train pos       %-of-persistence by horizon: [255.3  51.   27.7  14.1   6.3]
2026-08-22 17:48:38,699 INFO     val   pos       %-of-persistence by horizon: [255.2  51.1  27.6  14.1   6.2]
2026-08-22 17:48:38,699 INFO     train vel       R2 by horizon: [0.575 0.925 0.964 0.968 0.952], mean: 0.877
2026-08-22 17:48:38,699 INFO     val   vel       R2 by horizon: [0.576 0.927 0.965 0.968 0.952], mean: 0.878
2026-08-22 17:48:38,699 INFO     train vel       %-of-persistence by horizon: [156.9  23.2  14.6  13.8  16.7]
2026-08-22 17:48:38,699 INFO     val   vel       %-of-persistence by horizon: [156.7  23.   14.4  13.7  16.6]
2026-08-22 17:48:38,699 INFO     train heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:48:38,699 INFO     val   heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:48:38,699 INFO     train heading   %-of-persistence by horizon: [10.5  4.3  2.8  0.7  0.6]
2026-08-22 17:48:38,699 INFO     val   heading   %-of-persistence by horizon: [10.4  4.4  2.7  0.8  0.6]
2026-08-22 17:48:38,699 INFO     train stamina   R2 by horizon: [0.999 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:48:38,700 INFO     val   stamina   R2 by horizon: [0.999 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:48:38,700 INFO     train stamina   %-of-persistence by horizon: [1238.4  220.5   33.5   34.7   59.9]
2026-08-22 17:48:38,700 INFO     val   stamina   %-of-persistence by horizon: [1224.5  218.2   32.6   35.8   60.3]
2026-08-22 17:48:38,704 INFO Saved 'midtrain_latest' checkpoint to checkpoints/physics_pretrain/player_encoder_11.midtrain_latest.pt
2026-08-22 17:49:15,996 INFO epoch 6/500: train_loss=0.0094  pair_loss=0.0016  t0_loss=0.0020  val_loss=0.0094  best=0.0094  (improved)
2026-08-22 17:49:15,997 INFO     grad_norm: mean=0.260446 std=0.094736 min=0.093107 max=0.813703
2026-08-22 17:49:15,997 INFO     train_loss_delta (batch-to-batch): mean=-0.000000 std=0.000609 min=-0.001992 max=0.002360
2026-08-22 17:49:15,997 INFO     crossing_head: train loss=7.2960 pos_dist=10.506m dt_mae=1.999s | val loss=5.6978 pos_dist=9.910m dt_mae=1.733s
2026-08-22 17:49:15,997 INFO     goal_dist_delta_head: train loss=0.00357 mae=(left 2.467m, right 2.472m) | val loss=0.00352 mae=(left 2.445m, right 2.466m)
2026-08-22 17:49:15,997 INFO     short_horizon_probes: train loss=0.00050 rmse_norm=(0.2s 0.0170, 1.0s 0.0145) | val loss=0.00048 rmse_norm=(0.2s 0.0167, 1.0s 0.0143)
2026-08-22 17:49:15,997 INFO     val_loss_delta (epoch-over-epoch): -0.000031
2026-08-22 17:49:15,997 INFO         train pos_rmse     by horizon (m): [1.0422 0.9861 1.8187 1.6382 1.5277], mean: 1.4026 m
2026-08-22 17:49:15,997 INFO         val   pos_rmse     by horizon (m): [1.0457 0.9843 1.8102 1.6311 1.5221], mean: 1.3987 m
2026-08-22 17:49:15,998 INFO         train pos_dist     by horizon (m): [1.2027 1.1855 2.2135 1.9116 1.7126], mean: 1.6452 m
2026-08-22 17:49:15,998 INFO         val   pos_dist     by horizon (m): [1.2039 1.1772 2.2091 1.8888 1.7034], mean: 1.6365 m
2026-08-22 17:49:15,998 INFO         train vel_rmse     by horizon (m/s): [1.319  0.6331 0.4944 0.4687 0.5622], mean: 0.6955 m/s
2026-08-22 17:49:15,998 INFO         val   vel_rmse     by horizon (m/s): [1.3193 0.6311 0.4907 0.4663 0.5571], mean: 0.6929 m/s
2026-08-22 17:49:15,998 INFO         train vel_dist     by horizon (m/s): [1.5404 0.6812 0.5508 0.5357 0.6114], mean: 0.7839 m/s
2026-08-22 17:49:15,998 INFO         val   vel_dist     by horizon (m/s): [1.5401 0.6773 0.546  0.5356 0.6074], mean: 0.7812 m/s
2026-08-22 17:49:15,998 INFO         train heading_rmse by horizon: [0.051  0.0402 0.0272 0.0067 0.0059], mean: 0.0262
2026-08-22 17:49:15,998 INFO         val   heading_rmse by horizon: [0.051  0.0405 0.0269 0.0071 0.0056], mean: 0.0262
2026-08-22 17:49:15,998 INFO         train heading_dist by horizon: [0.0365 0.0181 0.007  0.0042 0.0042], mean: 0.0140
2026-08-22 17:49:15,998 INFO         val   heading_dist by horizon: [0.0364 0.0178 0.0068 0.004  0.004 ], mean: 0.0138
2026-08-22 17:49:15,999 INFO         train stamina_rmse by horizon: [0.0111 0.0098 0.0044 0.0076 0.0259], mean: 0.0117
2026-08-22 17:49:15,999 INFO         val   stamina_rmse by horizon: [0.011  0.0096 0.0043 0.0077 0.0261], mean: 0.0117
2026-08-22 17:49:15,999 INFO     train pos       R2 by horizon: [0.998 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:49:15,999 INFO     val   pos       R2 by horizon: [0.998 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:49:16,000 INFO     train pos       %-of-persistence by horizon: [253.5  50.9  27.7  14.1   6.2]
2026-08-22 17:49:16,000 INFO     val   pos       %-of-persistence by horizon: [254.3  50.8  27.5  14.    6.2]
2026-08-22 17:49:16,000 INFO     train vel       R2 by horizon: [0.574 0.926 0.965 0.968 0.952], mean: 0.877
2026-08-22 17:49:16,000 INFO     val   vel       R2 by horizon: [0.574 0.926 0.965 0.968 0.953], mean: 0.877
2026-08-22 17:49:16,001 INFO     train vel       %-of-persistence by horizon: [157.   23.1  14.5  13.7  16.7]
2026-08-22 17:49:16,001 INFO     val   vel       %-of-persistence by horizon: [157.   23.   14.4  13.6  16.5]
2026-08-22 17:49:16,001 INFO     train heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:49:16,001 INFO     val   heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:49:16,001 INFO     train heading   %-of-persistence by horizon: [10.5  4.3  2.8  0.7  0.6]
2026-08-22 17:49:16,001 INFO     val   heading   %-of-persistence by horizon: [10.4  4.4  2.7  0.7  0.6]
2026-08-22 17:49:16,001 INFO     train stamina   R2 by horizon: [0.999 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:49:16,001 INFO     val   stamina   R2 by horizon: [0.999 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:49:16,002 INFO     train stamina   %-of-persistence by horizon: [1249.3  221.    33.3   34.6   60. ]
2026-08-22 17:49:16,002 INFO     val   stamina   %-of-persistence by horizon: [1237.9  218.1   32.4   35.1   60.3]
2026-08-22 17:49:16,005 INFO Saved 'midtrain_latest' checkpoint to checkpoints/physics_pretrain/player_encoder_11.midtrain_latest.pt
2026-08-22 17:49:51,675 INFO epoch 7/500: train_loss=0.0094  pair_loss=0.0016  t0_loss=0.0020  val_loss=0.0096  best=0.0094  (patience 1/20)
2026-08-22 17:49:51,675 INFO     grad_norm: mean=0.258136 std=0.094192 min=0.089279 max=0.758217
2026-08-22 17:49:51,675 INFO     train_loss_delta (batch-to-batch): mean=0.000000 std=0.000608 min=-0.001942 max=0.002033
2026-08-22 17:49:51,675 INFO     crossing_head: train loss=7.2602 pos_dist=10.417m dt_mae=1.991s | val loss=5.6675 pos_dist=9.867m dt_mae=1.713s
2026-08-22 17:49:51,675 INFO     goal_dist_delta_head: train loss=0.00355 mae=(left 2.455m, right 2.463m) | val loss=0.00351 mae=(left 2.443m, right 2.461m)
2026-08-22 17:49:51,675 INFO     short_horizon_probes: train loss=0.00048 rmse_norm=(0.2s 0.0165, 1.0s 0.0142) | val loss=0.00046 rmse_norm=(0.2s 0.0163, 1.0s 0.0140)
2026-08-22 17:49:51,675 INFO     val_loss_delta (epoch-over-epoch): 0.000192
2026-08-22 17:49:51,675 INFO         train pos_rmse     by horizon (m): [1.0364 0.9852 1.8162 1.6338 1.5255], mean: 1.3994 m
2026-08-22 17:49:51,675 INFO         val   pos_rmse     by horizon (m): [1.0279 0.9826 1.8125 1.6333 1.5157], mean: 1.3944 m
2026-08-22 17:49:51,675 INFO         train pos_dist     by horizon (m): [1.1935 1.1849 2.2083 1.9057 1.7059], mean: 1.6396 m
2026-08-22 17:49:51,676 INFO         val   pos_dist     by horizon (m): [1.186  1.1794 2.1971 1.9168 1.6927], mean: 1.6344 m
2026-08-22 17:49:51,676 INFO         train vel_rmse     by horizon (m/s): [1.3195 0.6315 0.4921 0.4665 0.5619], mean: 0.6943 m/s
2026-08-22 17:49:51,676 INFO         val   vel_rmse     by horizon (m/s): [1.3198 0.6325 0.4904 0.4653 0.5622], mean: 0.6940 m/s
2026-08-22 17:49:51,677 INFO         train vel_dist     by horizon (m/s): [1.5407 0.6773 0.547  0.5319 0.6085], mean: 0.7811 m/s
2026-08-22 17:49:51,677 INFO         val   vel_dist     by horizon (m/s): [1.5416 0.6784 0.5426 0.5328 0.6146], mean: 0.7820 m/s
2026-08-22 17:49:51,677 INFO         train heading_rmse by horizon: [0.0509 0.0402 0.0272 0.0067 0.0058], mean: 0.0262
2026-08-22 17:49:51,678 INFO         val   heading_rmse by horizon: [0.0521 0.0412 0.027  0.0073 0.006 ], mean: 0.0267
2026-08-22 17:49:51,678 INFO         train heading_dist by horizon: [0.0364 0.018  0.007  0.0042 0.0042], mean: 0.0139
2026-08-22 17:49:51,678 INFO         val   heading_dist by horizon: [0.0372 0.0182 0.0076 0.0046 0.0046], mean: 0.0144
2026-08-22 17:49:51,678 INFO         train stamina_rmse by horizon: [0.0111 0.0098 0.0043 0.0075 0.0259], mean: 0.0117
2026-08-22 17:49:51,679 INFO         val   stamina_rmse by horizon: [0.0107 0.0092 0.0044 0.008  0.0265], mean: 0.0118
2026-08-22 17:49:51,679 INFO     train pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:49:51,679 INFO     val   pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:49:51,679 INFO     train pos       %-of-persistence by horizon: [252.1  50.9  27.6  14.    6.2]
2026-08-22 17:49:51,679 INFO     val   pos       %-of-persistence by horizon: [250.   50.7  27.6  14.    6.2]
2026-08-22 17:49:51,680 INFO     train vel       R2 by horizon: [0.574 0.926 0.965 0.968 0.952], mean: 0.877
2026-08-22 17:49:51,680 INFO     val   vel       R2 by horizon: [0.574 0.926 0.965 0.969 0.952], mean: 0.877
2026-08-22 17:49:51,680 INFO     train vel       %-of-persistence by horizon: [157.   23.   14.4  13.7  16.6]
2026-08-22 17:49:51,680 INFO     val   vel       %-of-persistence by horizon: [157.   23.1  14.3  13.6  16.7]
2026-08-22 17:49:51,680 INFO     train heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:49:51,681 INFO     val   heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:49:51,681 INFO     train heading   %-of-persistence by horizon: [10.4  4.3  2.8  0.7  0.6]
2026-08-22 17:49:51,681 INFO     val   heading   %-of-persistence by horizon: [10.7  4.4  2.8  0.8  0.6]
2026-08-22 17:49:51,681 INFO     train stamina   R2 by horizon: [0.999 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:49:51,681 INFO     val   stamina   R2 by horizon: [0.999 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:49:51,682 INFO     train stamina   %-of-persistence by horizon: [1260.1  221.3   33.    34.5   59.9]
2026-08-22 17:49:51,682 INFO     val   stamina   %-of-persistence by horizon: [1209.3  209.5   33.4   36.5   61.3]
2026-08-22 17:50:28,429 INFO epoch 8/500: train_loss=0.0094  pair_loss=0.0016  t0_loss=0.0020  val_loss=0.0093  best=0.0093  (improved)
2026-08-22 17:50:28,429 INFO     grad_norm: mean=0.254105 std=0.094501 min=0.093604 max=0.787866
2026-08-22 17:50:28,429 INFO     train_loss_delta (batch-to-batch): mean=0.000000 std=0.000598 min=-0.002007 max=0.001957
2026-08-22 17:50:28,429 INFO     crossing_head: train loss=7.2271 pos_dist=10.475m dt_mae=1.982s | val loss=5.6438 pos_dist=9.850m dt_mae=1.704s
2026-08-22 17:50:28,429 INFO     goal_dist_delta_head: train loss=0.00354 mae=(left 2.445m, right 2.455m) | val loss=0.00350 mae=(left 2.429m, right 2.455m)
2026-08-22 17:50:28,429 INFO     short_horizon_probes: train loss=0.00046 rmse_norm=(0.2s 0.0162, 1.0s 0.0140) | val loss=0.00045 rmse_norm=(0.2s 0.0160, 1.0s 0.0139)
2026-08-22 17:50:28,429 INFO     val_loss_delta (epoch-over-epoch): -0.000296
2026-08-22 17:50:28,430 INFO         train pos_rmse     by horizon (m): [1.0287 0.9884 1.8136 1.6313 1.5228], mean: 1.3970 m
2026-08-22 17:50:28,430 INFO         val   pos_rmse     by horizon (m): [1.0536 0.9889 1.8081 1.6235 1.5122], mean: 1.3973 m
2026-08-22 17:50:28,430 INFO         train pos_dist     by horizon (m): [1.1831 1.1887 2.2024 1.901  1.6989], mean: 1.6348 m
2026-08-22 17:50:28,430 INFO         val   pos_dist     by horizon (m): [1.2133 1.1866 2.2021 1.8858 1.6928], mean: 1.6361 m
2026-08-22 17:50:28,430 INFO         train vel_rmse     by horizon (m/s): [1.3205 0.63   0.4903 0.4653 0.5625], mean: 0.6937 m/s
2026-08-22 17:50:28,430 INFO         val   vel_rmse     by horizon (m/s): [1.3226 0.6306 0.4886 0.461  0.5575], mean: 0.6920 m/s
2026-08-22 17:50:28,430 INFO         train vel_dist     by horizon (m/s): [1.5418 0.6739 0.5443 0.5301 0.6076], mean: 0.7796 m/s
2026-08-22 17:50:28,430 INFO         val   vel_dist     by horizon (m/s): [1.5444 0.6734 0.5436 0.5272 0.6036], mean: 0.7784 m/s
2026-08-22 17:50:28,430 INFO         train heading_rmse by horizon: [0.0509 0.0402 0.0272 0.0067 0.0058], mean: 0.0262
2026-08-22 17:50:28,430 INFO         val   heading_rmse by horizon: [0.0499 0.0407 0.027  0.0072 0.0058], mean: 0.0261
2026-08-22 17:50:28,431 INFO         train heading_dist by horizon: [0.0363 0.018  0.007  0.0042 0.0042], mean: 0.0139
2026-08-22 17:50:28,431 INFO         val   heading_dist by horizon: [0.0357 0.018  0.0071 0.0043 0.0043], mean: 0.0139
2026-08-22 17:50:28,431 INFO         train stamina_rmse by horizon: [0.0112 0.0098 0.0043 0.0076 0.026 ], mean: 0.0118
2026-08-22 17:50:28,431 INFO         val   stamina_rmse by horizon: [0.0116 0.0101 0.0044 0.0072 0.0256], mean: 0.0118
2026-08-22 17:50:28,431 INFO     train pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:50:28,432 INFO     val   pos       R2 by horizon: [0.998 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:50:28,432 INFO     train pos       %-of-persistence by horizon: [250.2  51.   27.6  14.    6.2]
2026-08-22 17:50:28,432 INFO     val   pos       %-of-persistence by horizon: [256.2  51.1  27.5  13.9   6.2]
2026-08-22 17:50:28,432 INFO     train vel       R2 by horizon: [0.574 0.927 0.965 0.968 0.952], mean: 0.877
2026-08-22 17:50:28,432 INFO     val   vel       R2 by horizon: [0.572 0.926 0.966 0.969 0.953], mean: 0.877
2026-08-22 17:50:28,432 INFO     train vel       %-of-persistence by horizon: [157.1  23.   14.3  13.6  16.7]
2026-08-22 17:50:28,432 INFO     val   vel       %-of-persistence by horizon: [157.4  23.   14.3  13.5  16.5]
2026-08-22 17:50:28,432 INFO     train heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:50:28,432 INFO     val   heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:50:28,433 INFO     train heading   %-of-persistence by horizon: [10.4  4.3  2.8  0.7  0.6]
2026-08-22 17:50:28,433 INFO     val   heading   %-of-persistence by horizon: [10.2  4.4  2.8  0.7  0.6]
2026-08-22 17:50:28,433 INFO     train stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:50:28,433 INFO     val   stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:50:28,433 INFO     train stamina   %-of-persistence by horizon: [1267.4  221.1   32.8   34.5   60. ]
2026-08-22 17:50:28,433 INFO     val   stamina   %-of-persistence by horizon: [1312.   227.9   33.1   33.1   59.2]
2026-08-22 17:50:28,439 INFO Saved 'midtrain_latest' checkpoint to checkpoints/physics_pretrain/player_encoder_11.midtrain_latest.pt
2026-08-22 17:51:11,987 INFO epoch 9/500: train_loss=0.0094  pair_loss=0.0016  t0_loss=0.0020  val_loss=0.0094  best=0.0093  (patience 1/20)
2026-08-22 17:51:11,987 INFO     grad_norm: mean=0.259392 std=0.102306 min=0.089912 max=0.915837
2026-08-22 17:51:11,987 INFO     train_loss_delta (batch-to-batch): mean=0.000001 std=0.000617 min=-0.002580 max=0.003140
2026-08-22 17:51:11,987 INFO     crossing_head: train loss=7.2014 pos_dist=10.477m dt_mae=1.976s | val loss=5.6154 pos_dist=9.840m dt_mae=1.681s
2026-08-22 17:51:11,987 INFO     goal_dist_delta_head: train loss=0.00353 mae=(left 2.435m, right 2.448m) | val loss=0.00349 mae=(left 2.420m, right 2.442m)
2026-08-22 17:51:11,987 INFO     short_horizon_probes: train loss=0.00044 rmse_norm=(0.2s 0.0159, 1.0s 0.0138) | val loss=0.00044 rmse_norm=(0.2s 0.0157, 1.0s 0.0137)
2026-08-22 17:51:11,988 INFO     val_loss_delta (epoch-over-epoch): 0.000081
2026-08-22 17:51:11,988 INFO         train pos_rmse     by horizon (m): [1.0207 0.9927 1.8136 1.6282 1.5198], mean: 1.3950 m
2026-08-22 17:51:11,988 INFO         val   pos_rmse     by horizon (m): [1.0303 1.002  1.809  1.6271 1.5103], mean: 1.3957 m
2026-08-22 17:51:11,988 INFO         train pos_dist     by horizon (m): [1.1721 1.1942 2.201  1.8951 1.6928], mean: 1.6310 m
2026-08-22 17:51:11,988 INFO         val   pos_dist     by horizon (m): [1.1808 1.1932 2.2058 1.8802 1.6823], mean: 1.6285 m
2026-08-22 17:51:11,988 INFO         train vel_rmse     by horizon (m/s): [1.3213 0.6289 0.4898 0.4645 0.5631], mean: 0.6935 m/s
2026-08-22 17:51:11,989 INFO         val   vel_rmse     by horizon (m/s): [1.327  0.6301 0.4871 0.4623 0.5597], mean: 0.6932 m/s
2026-08-22 17:51:11,989 INFO         train vel_dist     by horizon (m/s): [1.5428 0.6715 0.5433 0.5293 0.6074], mean: 0.7789 m/s
2026-08-22 17:51:11,989 INFO         val   vel_dist     by horizon (m/s): [1.5516 0.6733 0.5403 0.5295 0.6041], mean: 0.7798 m/s
2026-08-22 17:51:11,989 INFO         train heading_rmse by horizon: [0.0505 0.0402 0.0272 0.0067 0.0058], mean: 0.0261
2026-08-22 17:51:11,989 INFO         val   heading_rmse by horizon: [0.0507 0.0406 0.0269 0.0074 0.0059], mean: 0.0263
2026-08-22 17:51:11,990 INFO         train heading_dist by horizon: [0.036  0.018  0.007  0.0042 0.0042], mean: 0.0139
2026-08-22 17:51:11,990 INFO         val   heading_dist by horizon: [0.036  0.0183 0.0068 0.0044 0.0043], mean: 0.0139
2026-08-22 17:51:11,990 INFO         train stamina_rmse by horizon: [0.0113 0.0098 0.0043 0.0075 0.0259], mean: 0.0118
2026-08-22 17:51:11,990 INFO         val   stamina_rmse by horizon: [0.0114 0.0098 0.0043 0.0075 0.0259], mean: 0.0118
2026-08-22 17:51:11,991 INFO     train pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:51:11,991 INFO     val   pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:51:11,991 INFO     train pos       %-of-persistence by horizon: [248.3  51.3  27.6  14.    6.2]
2026-08-22 17:51:11,991 INFO     val   pos       %-of-persistence by horizon: [250.6  51.7  27.5  14.    6.2]
2026-08-22 17:51:11,991 INFO     train vel       R2 by horizon: [0.573 0.927 0.966 0.969 0.952], mean: 0.877
2026-08-22 17:51:11,992 INFO     val   vel       R2 by horizon: [0.569 0.927 0.966 0.969 0.952], mean: 0.877
2026-08-22 17:51:11,992 INFO     train vel       %-of-persistence by horizon: [157.2  22.9  14.3  13.6  16.7]
2026-08-22 17:51:11,992 INFO     val   vel       %-of-persistence by horizon: [157.9  23.   14.2  13.5  16.6]
2026-08-22 17:51:11,992 INFO     train heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:51:11,992 INFO     val   heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:51:11,992 INFO     train heading   %-of-persistence by horizon: [10.3  4.3  2.8  0.7  0.6]
2026-08-22 17:51:11,992 INFO     val   heading   %-of-persistence by horizon: [10.4  4.4  2.7  0.8  0.6]
2026-08-22 17:51:11,992 INFO     train stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:51:11,993 INFO     val   stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:51:11,993 INFO     train stamina   %-of-persistence by horizon: [1282.6  222.    32.5   34.2   59.9]
2026-08-22 17:51:11,993 INFO     val   stamina   %-of-persistence by horizon: [1285.2  222.3   32.3   34.2   59.9]
2026-08-22 17:51:59,956 INFO epoch 10/500: train_loss=0.0094  pair_loss=0.0016  t0_loss=0.0020  val_loss=0.0097  best=0.0093  (patience 2/20)
2026-08-22 17:51:59,957 INFO     grad_norm: mean=0.259344 std=0.095480 min=0.096597 max=0.833095
2026-08-22 17:51:59,957 INFO     train_loss_delta (batch-to-batch): mean=-0.000000 std=0.000596 min=-0.003675 max=0.004228
2026-08-22 17:51:59,957 INFO     crossing_head: train loss=7.1771 pos_dist=10.440m dt_mae=1.968s | val loss=5.6105 pos_dist=9.828m dt_mae=1.697s
2026-08-22 17:51:59,957 INFO     goal_dist_delta_head: train loss=0.00352 mae=(left 2.426m, right 2.440m) | val loss=0.00347 mae=(left 2.409m, right 2.441m)
2026-08-22 17:51:59,957 INFO     short_horizon_probes: train loss=0.00043 rmse_norm=(0.2s 0.0157, 1.0s 0.0137) | val loss=0.00043 rmse_norm=(0.2s 0.0155, 1.0s 0.0136)
2026-08-22 17:51:59,957 INFO     val_loss_delta (epoch-over-epoch): 0.000240
2026-08-22 17:51:59,957 INFO         train pos_rmse     by horizon (m): [1.0091 1.0009 1.8141 1.6278 1.52  ], mean: 1.3944 m
2026-08-22 17:51:59,958 INFO         val   pos_rmse     by horizon (m): [1.0115 1.0041 1.8013 1.6351 1.5187], mean: 1.3941 m
2026-08-22 17:51:59,958 INFO         train pos_dist     by horizon (m): [1.1591 1.2035 2.1991 1.8905 1.6902], mean: 1.6285 m
2026-08-22 17:51:59,958 INFO         val   pos_dist     by horizon (m): [1.1572 1.202  2.1906 1.8799 1.6952], mean: 1.6250 m
2026-08-22 17:51:59,958 INFO         train vel_rmse     by horizon (m/s): [1.3223 0.6271 0.4892 0.4648 0.5652], mean: 0.6937 m/s
2026-08-22 17:51:59,958 INFO         val   vel_rmse     by horizon (m/s): [1.3241 0.6275 0.4893 0.4617 0.563 ], mean: 0.6931 m/s
2026-08-22 17:51:59,959 INFO         train vel_dist     by horizon (m/s): [1.5437 0.6682 0.5428 0.5296 0.6085], mean: 0.7786 m/s
2026-08-22 17:51:59,959 INFO         val   vel_dist     by horizon (m/s): [1.5466 0.6678 0.5442 0.528  0.6065], mean: 0.7786 m/s
2026-08-22 17:51:59,959 INFO         train heading_rmse by horizon: [0.0505 0.0403 0.0272 0.0067 0.0058], mean: 0.0261
2026-08-22 17:51:59,959 INFO         val   heading_rmse by horizon: [0.051  0.0429 0.0269 0.0078 0.0068], mean: 0.0271
2026-08-22 17:51:59,959 INFO         train heading_dist by horizon: [0.036  0.018  0.0069 0.0042 0.0042], mean: 0.0139
2026-08-22 17:51:59,959 INFO         val   heading_dist by horizon: [0.0362 0.0197 0.007  0.0048 0.0051], mean: 0.0146
2026-08-22 17:51:59,959 INFO         train stamina_rmse by horizon: [0.0114 0.0098 0.0043 0.0075 0.0259], mean: 0.0118
2026-08-22 17:51:59,959 INFO         val   stamina_rmse by horizon: [0.0111 0.0095 0.004  0.0078 0.0263], mean: 0.0117
2026-08-22 17:51:59,959 INFO     train pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:51:59,960 INFO     val   pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:51:59,960 INFO     train pos       %-of-persistence by horizon: [245.5  51.7  27.6  14.    6.2]
2026-08-22 17:51:59,960 INFO     val   pos       %-of-persistence by horizon: [246.   51.8  27.4  14.    6.2]
2026-08-22 17:51:59,960 INFO     train vel       R2 by horizon: [0.572 0.927 0.966 0.969 0.952], mean: 0.877
2026-08-22 17:51:59,960 INFO     val   vel       R2 by horizon: [0.571 0.927 0.966 0.969 0.952], mean: 0.877
2026-08-22 17:51:59,961 INFO     train vel       %-of-persistence by horizon: [157.3  22.9  14.3  13.6  16.7]
2026-08-22 17:51:59,961 INFO     val   vel       %-of-persistence by horizon: [157.6  22.9  14.3  13.5  16.7]
2026-08-22 17:51:59,961 INFO     train heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:51:59,961 INFO     val   heading   R2 by horizon: [0.995 0.996 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:51:59,961 INFO     train heading   %-of-persistence by horizon: [10.4  4.3  2.8  0.7  0.6]
2026-08-22 17:51:59,961 INFO     val   heading   %-of-persistence by horizon: [10.4  4.6  2.7  0.8  0.7]
2026-08-22 17:51:59,961 INFO     train stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:51:59,962 INFO     val   stamina   R2 by horizon: [0.999 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:51:59,962 INFO     train stamina   %-of-persistence by horizon: [1288.9  222.1   32.3   34.1   59.9]
2026-08-22 17:51:59,962 INFO     val   stamina   %-of-persistence by horizon: [1251.7  214.3   30.5   35.4   60.8]
2026-08-22 17:52:37,127 INFO epoch 11/500: train_loss=0.0094  pair_loss=0.0016  t0_loss=0.0020  val_loss=0.0093  best=0.0093  (improved)
2026-08-22 17:52:37,127 INFO     grad_norm: mean=0.251416 std=0.091153 min=0.094530 max=0.747379
2026-08-22 17:52:37,127 INFO     train_loss_delta (batch-to-batch): mean=-0.000001 std=0.000601 min=-0.002357 max=0.001994
2026-08-22 17:52:37,127 INFO     crossing_head: train loss=7.1559 pos_dist=10.379m dt_mae=1.962s | val loss=5.6252 pos_dist=9.829m dt_mae=1.727s
2026-08-22 17:52:37,127 INFO     goal_dist_delta_head: train loss=0.00351 mae=(left 2.418m, right 2.434m) | val loss=0.00347 mae=(left 2.396m, right 2.437m)
2026-08-22 17:52:37,127 INFO     short_horizon_probes: train loss=0.00042 rmse_norm=(0.2s 0.0154, 1.0s 0.0136) | val loss=0.00042 rmse_norm=(0.2s 0.0153, 1.0s 0.0136)
2026-08-22 17:52:37,127 INFO     val_loss_delta (epoch-over-epoch): -0.000379
2026-08-22 17:52:37,127 INFO         train pos_rmse     by horizon (m): [1.0017 1.0041 1.8131 1.6275 1.5186], mean: 1.3930 m
2026-08-22 17:52:37,128 INFO         val   pos_rmse     by horizon (m): [0.9864 0.9973 1.8012 1.6317 1.5067], mean: 1.3847 m
2026-08-22 17:52:37,128 INFO         train pos_dist     by horizon (m): [1.1501 1.2074 2.1959 1.887  1.6869], mean: 1.6255 m
2026-08-22 17:52:37,128 INFO         val   pos_dist     by horizon (m): [1.1327 1.1987 2.1807 1.8939 1.6773], mean: 1.6167 m
2026-08-22 17:52:37,128 INFO         train vel_rmse     by horizon (m/s): [1.3223 0.6256 0.4894 0.4652 0.5667], mean: 0.6938 m/s
2026-08-22 17:52:37,128 INFO         val   vel_rmse     by horizon (m/s): [1.315  0.6216 0.4877 0.4598 0.5663], mean: 0.6901 m/s
2026-08-22 17:52:37,128 INFO         train vel_dist     by horizon (m/s): [1.5435 0.666  0.5432 0.5301 0.6091], mean: 0.7784 m/s
2026-08-22 17:52:37,128 INFO         val   vel_dist     by horizon (m/s): [1.5314 0.661  0.5374 0.5218 0.6105], mean: 0.7724 m/s
2026-08-22 17:52:37,128 INFO         train heading_rmse by horizon: [0.0504 0.0402 0.0273 0.0067 0.0058], mean: 0.0261
2026-08-22 17:52:37,128 INFO         val   heading_rmse by horizon: [0.0498 0.0407 0.0269 0.0071 0.0056], mean: 0.0260
2026-08-22 17:52:37,128 INFO         train heading_dist by horizon: [0.0359 0.018  0.0069 0.0042 0.0042], mean: 0.0138
2026-08-22 17:52:37,128 INFO         val   heading_dist by horizon: [0.0353 0.0181 0.0068 0.0041 0.0039], mean: 0.0136
2026-08-22 17:52:37,128 INFO         train stamina_rmse by horizon: [0.0114 0.0098 0.0042 0.0075 0.0259], mean: 0.0118
2026-08-22 17:52:37,129 INFO         val   stamina_rmse by horizon: [0.0117 0.01   0.0042 0.0071 0.0256], mean: 0.0118
2026-08-22 17:52:37,129 INFO     train pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:52:37,129 INFO     val   pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:52:37,129 INFO     train pos       %-of-persistence by horizon: [243.7  51.9  27.6  14.    6.2]
2026-08-22 17:52:37,129 INFO     val   pos       %-of-persistence by horizon: [239.9  51.5  27.4  14.    6.2]
2026-08-22 17:52:37,129 INFO     train vel       R2 by horizon: [0.572 0.928 0.966 0.969 0.951], mean: 0.877
2026-08-22 17:52:37,129 INFO     val   vel       R2 by horizon: [0.577 0.929 0.966 0.969 0.951], mean: 0.878
2026-08-22 17:52:37,129 INFO     train vel       %-of-persistence by horizon: [157.3  22.8  14.3  13.6  16.8]
2026-08-22 17:52:37,129 INFO     val   vel       %-of-persistence by horizon: [156.5  22.7  14.3  13.4  16.8]
2026-08-22 17:52:37,129 INFO     train heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:52:37,130 INFO     val   heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:52:37,130 INFO     train heading   %-of-persistence by horizon: [10.3  4.3  2.8  0.7  0.6]
2026-08-22 17:52:37,130 INFO     val   heading   %-of-persistence by horizon: [10.2  4.4  2.7  0.7  0.6]
2026-08-22 17:52:37,130 INFO     train stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:52:37,130 INFO     val   stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:52:37,130 INFO     train stamina   %-of-persistence by horizon: [1294.4  222.    32.1   34.1   59.9]
2026-08-22 17:52:37,130 INFO     val   stamina   %-of-persistence by horizon: [1326.3  227.6   31.8   32.6   59.2]
2026-08-22 17:52:37,133 INFO Saved 'midtrain_latest' checkpoint to checkpoints/physics_pretrain/player_encoder_11.midtrain_latest.pt
2026-08-22 17:53:02,678 INFO epoch 12/500: train_loss=0.0094  pair_loss=0.0016  t0_loss=0.0020  val_loss=0.0093  best=0.0093  (improved)
2026-08-22 17:53:02,678 INFO     grad_norm: mean=0.251316 std=0.095539 min=0.098973 max=0.765630
2026-08-22 17:53:02,678 INFO     train_loss_delta (batch-to-batch): mean=0.000001 std=0.000622 min=-0.002052 max=0.001963
2026-08-22 17:53:02,678 INFO     crossing_head: train loss=7.1380 pos_dist=10.344m dt_mae=1.959s | val loss=5.6071 pos_dist=9.820m dt_mae=1.717s
2026-08-22 17:53:02,678 INFO     goal_dist_delta_head: train loss=0.00350 mae=(left 2.409m, right 2.428m) | val loss=0.00346 mae=(left 2.394m, right 2.431m)
2026-08-22 17:53:02,678 INFO     short_horizon_probes: train loss=0.00042 rmse_norm=(0.2s 0.0153, 1.0s 0.0136) | val loss=0.00041 rmse_norm=(0.2s 0.0152, 1.0s 0.0135)
2026-08-22 17:53:02,678 INFO     val_loss_delta (epoch-over-epoch): -0.000005
2026-08-22 17:53:02,679 INFO         train pos_rmse     by horizon (m): [0.9957 1.006  1.8121 1.6279 1.5186], mean: 1.3921 m
2026-08-22 17:53:02,679 INFO         val   pos_rmse     by horizon (m): [0.9876 1.0008 1.793  1.6342 1.5119], mean: 1.3855 m
2026-08-22 17:53:02,679 INFO         train pos_dist     by horizon (m): [1.143  1.2093 2.1933 1.886  1.6849], mean: 1.6233 m
2026-08-22 17:53:02,679 INFO         val   pos_dist     by horizon (m): [1.1295 1.1997 2.1744 1.8922 1.6808], mean: 1.6153 m
2026-08-22 17:53:02,679 INFO         train vel_rmse     by horizon (m/s): [1.3212 0.6246 0.4901 0.4654 0.568 ], mean: 0.6939 m/s
2026-08-22 17:53:02,679 INFO         val   vel_rmse     by horizon (m/s): [1.3164 0.6215 0.4881 0.459  0.5627], mean: 0.6895 m/s
2026-08-22 17:53:02,679 INFO         train vel_dist     by horizon (m/s): [1.542  0.6645 0.5438 0.5299 0.6094], mean: 0.7779 m/s
2026-08-22 17:53:02,679 INFO         val   vel_dist     by horizon (m/s): [1.5337 0.6602 0.5399 0.5199 0.5985], mean: 0.7705 m/s
2026-08-22 17:53:02,679 INFO         train heading_rmse by horizon: [0.0504 0.0402 0.0272 0.0067 0.0058], mean: 0.0261
2026-08-22 17:53:02,679 INFO         val   heading_rmse by horizon: [0.0497 0.0407 0.0269 0.0071 0.0058], mean: 0.0260
2026-08-22 17:53:02,679 INFO         train heading_dist by horizon: [0.0359 0.018  0.007  0.0042 0.0042], mean: 0.0138
2026-08-22 17:53:02,679 INFO         val   heading_dist by horizon: [0.0352 0.0179 0.007  0.0042 0.0043], mean: 0.0137
2026-08-22 17:53:02,680 INFO         train stamina_rmse by horizon: [0.0115 0.0098 0.0042 0.0074 0.0259], mean: 0.0118
2026-08-22 17:53:02,680 INFO         val   stamina_rmse by horizon: [0.0119 0.0101 0.0044 0.0071 0.0255], mean: 0.0118
2026-08-22 17:53:02,680 INFO     train pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:53:02,680 INFO     val   pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:53:02,680 INFO     train pos       %-of-persistence by horizon: [242.2  52.   27.6  14.    6.2]
2026-08-22 17:53:02,680 INFO     val   pos       %-of-persistence by horizon: [240.2  51.7  27.3  14.    6.2]
2026-08-22 17:53:02,680 INFO     train vel       R2 by horizon: [0.573 0.928 0.965 0.968 0.951], mean: 0.877
2026-08-22 17:53:02,680 INFO     val   vel       R2 by horizon: [0.576 0.929 0.966 0.969 0.952], mean: 0.878
2026-08-22 17:53:02,680 INFO     train vel       %-of-persistence by horizon: [157.2  22.8  14.3  13.6  16.8]
2026-08-22 17:53:02,680 INFO     val   vel       %-of-persistence by horizon: [156.6  22.7  14.3  13.4  16.7]
2026-08-22 17:53:02,680 INFO     train heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:53:02,681 INFO     val   heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:53:02,681 INFO     train heading   %-of-persistence by horizon: [10.3  4.3  2.8  0.7  0.6]
2026-08-22 17:53:02,681 INFO     val   heading   %-of-persistence by horizon: [10.2  4.4  2.7  0.7  0.6]
2026-08-22 17:53:02,681 INFO     train stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:53:02,681 INFO     val   stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:53:02,681 INFO     train stamina   %-of-persistence by horizon: [1300.4  221.9   31.8   34.    59.9]
2026-08-22 17:53:02,681 INFO     val   stamina   %-of-persistence by horizon: [1341.1  229.4   33.3   32.5   59. ]
2026-08-22 17:53:02,684 INFO Saved 'midtrain_latest' checkpoint to checkpoints/physics_pretrain/player_encoder_11.midtrain_latest.pt
2026-08-22 17:53:36,453 INFO epoch 13/500: train_loss=0.0093  pair_loss=0.0016  t0_loss=0.0020  val_loss=0.0096  best=0.0093  (patience 1/20)
2026-08-22 17:53:36,453 INFO     grad_norm: mean=0.250160 std=0.092459 min=0.103581 max=0.857582
2026-08-22 17:53:36,453 INFO     train_loss_delta (batch-to-batch): mean=0.000000 std=0.000614 min=-0.001850 max=0.002264
2026-08-22 17:53:36,453 INFO     crossing_head: train loss=7.1212 pos_dist=10.346m dt_mae=1.953s | val loss=5.5744 pos_dist=9.819m dt_mae=1.690s
2026-08-22 17:53:36,453 INFO     goal_dist_delta_head: train loss=0.00349 mae=(left 2.402m, right 2.422m) | val loss=0.00344 mae=(left 2.386m, right 2.423m)
2026-08-22 17:53:36,453 INFO     short_horizon_probes: train loss=0.00041 rmse_norm=(0.2s 0.0152, 1.0s 0.0135) | val loss=0.00041 rmse_norm=(0.2s 0.0151, 1.0s 0.0134)
2026-08-22 17:53:36,453 INFO     val_loss_delta (epoch-over-epoch): 0.000285
2026-08-22 17:53:36,453 INFO         train pos_rmse     by horizon (m): [0.9897 1.0081 1.811  1.628  1.5185], mean: 1.3910 m
2026-08-22 17:53:36,453 INFO         val   pos_rmse     by horizon (m): [0.9728 1.0113 1.8101 1.6276 1.5275], mean: 1.3899 m
2026-08-22 17:53:36,453 INFO         train pos_dist     by horizon (m): [1.1352 1.212  2.1911 1.8847 1.6831], mean: 1.6212 m
2026-08-22 17:53:36,453 INFO         val   pos_dist     by horizon (m): [1.1196 1.2164 2.1879 1.8883 1.7046], mean: 1.6234 m
2026-08-22 17:53:36,453 INFO         train vel_rmse     by horizon (m/s): [1.3215 0.6237 0.49   0.4651 0.5682], mean: 0.6937 m/s
2026-08-22 17:53:36,453 INFO         val   vel_rmse     by horizon (m/s): [1.3195 0.6154 0.4908 0.4662 0.5658], mean: 0.6915 m/s
2026-08-22 17:53:36,454 INFO         train vel_dist     by horizon (m/s): [1.5422 0.6631 0.5438 0.5295 0.6088], mean: 0.7775 m/s
2026-08-22 17:53:36,454 INFO         val   vel_dist     by horizon (m/s): [1.5388 0.6555 0.5458 0.5335 0.6057], mean: 0.7759 m/s
2026-08-22 17:53:36,454 INFO         train heading_rmse by horizon: [0.0502 0.0403 0.0272 0.0067 0.0058], mean: 0.0260
2026-08-22 17:53:36,454 INFO         val   heading_rmse by horizon: [0.0517 0.0412 0.0268 0.0073 0.0058], mean: 0.0266
2026-08-22 17:53:36,454 INFO         train heading_dist by horizon: [0.0358 0.018  0.0069 0.0042 0.0042], mean: 0.0138
2026-08-22 17:53:36,454 INFO         val   heading_dist by horizon: [0.0372 0.0182 0.0068 0.0042 0.004 ], mean: 0.0141
2026-08-22 17:53:36,455 INFO         train stamina_rmse by horizon: [0.0116 0.0098 0.0042 0.0074 0.0259], mean: 0.0118
2026-08-22 17:53:36,455 INFO         val   stamina_rmse by horizon: [0.0112 0.0095 0.0041 0.0076 0.0262], mean: 0.0117
2026-08-22 17:53:36,455 INFO     train pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:53:36,455 INFO     val   pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:53:36,456 INFO     train pos       %-of-persistence by horizon: [240.7  52.1  27.6  14.    6.2]
2026-08-22 17:53:36,456 INFO     val   pos       %-of-persistence by horizon: [236.6  52.2  27.5  14.    6.2]
2026-08-22 17:53:36,456 INFO     train vel       R2 by horizon: [0.573 0.928 0.965 0.969 0.951], mean: 0.877
2026-08-22 17:53:36,456 INFO     val   vel       R2 by horizon: [0.574 0.93  0.965 0.968 0.951], mean: 0.878
2026-08-22 17:53:36,456 INFO     train vel       %-of-persistence by horizon: [157.2  22.8  14.3  13.6  16.8]
2026-08-22 17:53:36,456 INFO     val   vel       %-of-persistence by horizon: [157.   22.4  14.4  13.6  16.8]
2026-08-22 17:53:36,456 INFO     train heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:53:36,456 INFO     val   heading   R2 by horizon: [0.995 0.997 0.999 1.    1.   ], mean: 0.998
2026-08-22 17:53:36,456 INFO     train heading   %-of-persistence by horizon: [10.3  4.3  2.8  0.7  0.6]
2026-08-22 17:53:36,457 INFO     val   heading   %-of-persistence by horizon: [10.6  4.4  2.7  0.8  0.6]
2026-08-22 17:53:36,457 INFO     train stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:53:36,457 INFO     val   stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:53:36,457 INFO     train stamina   %-of-persistence by horizon: [1307.2  222.5   31.8   33.9   59.8]
2026-08-22 17:53:36,458 INFO     val   stamina   %-of-persistence by horizon: [1271.   215.    31.    34.9   60.6]
2026-08-22 17:54:23,019 INFO epoch 14/500: train_loss=0.0093  pair_loss=0.0016  t0_loss=0.0020  val_loss=0.0094  best=0.0093  (patience 2/20)
2026-08-22 17:54:23,019 INFO     grad_norm: mean=0.246721 std=0.089466 min=0.105103 max=0.701236
2026-08-22 17:54:23,019 INFO     train_loss_delta (batch-to-batch): mean=0.000000 std=0.000620 min=-0.002144 max=0.003045
2026-08-22 17:54:23,019 INFO     crossing_head: train loss=7.1032 pos_dist=10.391m dt_mae=1.947s | val loss=5.6162 pos_dist=9.806m dt_mae=1.741s
2026-08-22 17:54:23,019 INFO     goal_dist_delta_head: train loss=0.00348 mae=(left 2.395m, right 2.417m) | val loss=0.00344 mae=(left 2.378m, right 2.419m)
2026-08-22 17:54:23,019 INFO     short_horizon_probes: train loss=0.00041 rmse_norm=(0.2s 0.0151, 1.0s 0.0135) | val loss=0.00040 rmse_norm=(0.2s 0.0150, 1.0s 0.0135)
2026-08-22 17:54:23,019 INFO     val_loss_delta (epoch-over-epoch): -0.000155
2026-08-22 17:54:23,019 INFO         train pos_rmse     by horizon (m): [0.9846 1.0106 1.8105 1.6296 1.522 ], mean: 1.3915 m
2026-08-22 17:54:23,019 INFO         val   pos_rmse     by horizon (m): [0.9812 1.0053 1.797  1.6315 1.5235], mean: 1.3877 m
2026-08-22 17:54:23,019 INFO         train pos_dist     by horizon (m): [1.1293 1.2148 2.1889 1.8841 1.6865], mean: 1.6207 m
2026-08-22 17:54:23,019 INFO         val   pos_dist     by horizon (m): [1.1205 1.2031 2.1761 1.8893 1.693 ], mean: 1.6164 m
2026-08-22 17:54:23,020 INFO         train vel_rmse     by horizon (m/s): [1.322  0.6228 0.4904 0.466  0.5701], mean: 0.6943 m/s
2026-08-22 17:54:23,020 INFO         val   vel_rmse     by horizon (m/s): [1.3199 0.6159 0.4887 0.4629 0.5683], mean: 0.6911 m/s
2026-08-22 17:54:23,020 INFO         train vel_dist     by horizon (m/s): [1.5427 0.6619 0.5444 0.5304 0.6099], mean: 0.7779 m/s
2026-08-22 17:54:23,021 INFO         val   vel_dist     by horizon (m/s): [1.539  0.6536 0.5418 0.5283 0.6114], mean: 0.7748 m/s
2026-08-22 17:54:23,021 INFO         train heading_rmse by horizon: [0.0502 0.0403 0.0272 0.0067 0.0058], mean: 0.0260
2026-08-22 17:54:23,021 INFO         val   heading_rmse by horizon: [0.0507 0.0408 0.0268 0.0072 0.0056], mean: 0.0262
2026-08-22 17:54:23,022 INFO         train heading_dist by horizon: [0.0358 0.018  0.0069 0.0042 0.0042], mean: 0.0138
2026-08-22 17:54:23,022 INFO         val   heading_dist by horizon: [0.0362 0.0179 0.0067 0.0041 0.0039], mean: 0.0138
2026-08-22 17:54:23,022 INFO         train stamina_rmse by horizon: [0.0116 0.0098 0.0042 0.0074 0.0259], mean: 0.0118
2026-08-22 17:54:23,022 INFO         val   stamina_rmse by horizon: [0.0113 0.0094 0.0042 0.0077 0.0262], mean: 0.0118
2026-08-22 17:54:23,022 INFO     train pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:54:23,022 INFO     val   pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:54:23,023 INFO     train pos       %-of-persistence by horizon: [239.5  52.2  27.6  14.    6.2]
2026-08-22 17:54:23,023 INFO     val   pos       %-of-persistence by horizon: [238.6  51.9  27.3  14.    6.2]
2026-08-22 17:54:23,023 INFO     train vel       R2 by horizon: [0.573 0.928 0.965 0.968 0.951], mean: 0.877
2026-08-22 17:54:23,023 INFO     val   vel       R2 by horizon: [0.574 0.93  0.966 0.969 0.951], mean: 0.878
2026-08-22 17:54:23,023 INFO     train vel       %-of-persistence by horizon: [157.3  22.7  14.3  13.6  16.9]
2026-08-22 17:54:23,023 INFO     val   vel       %-of-persistence by horizon: [157.   22.5  14.3  13.5  16.8]
2026-08-22 17:54:23,023 INFO     train heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:54:23,024 INFO     val   heading   R2 by horizon: [0.995 0.997 0.999 1.    1.   ], mean: 0.998
2026-08-22 17:54:23,024 INFO     train heading   %-of-persistence by horizon: [10.3  4.3  2.8  0.7  0.6]
2026-08-22 17:54:23,024 INFO     val   heading   %-of-persistence by horizon: [10.4  4.4  2.7  0.7  0.6]
2026-08-22 17:54:23,024 INFO     train stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:54:23,024 INFO     val   stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:54:23,024 INFO     train stamina   %-of-persistence by horizon: [1307.7  221.9   31.6   33.9   59.9]
2026-08-22 17:54:23,024 INFO     val   stamina   %-of-persistence by horizon: [1274.   213.9   31.6   35.    60.6]
2026-08-22 17:55:13,211 INFO epoch 15/500: train_loss=0.0094  pair_loss=0.0016  t0_loss=0.0020  val_loss=0.0094  best=0.0093  (patience 3/20)
2026-08-22 17:55:13,211 INFO     grad_norm: mean=0.260502 std=0.120351 min=0.101838 max=1.176915
2026-08-22 17:55:13,211 INFO     train_loss_delta (batch-to-batch): mean=-0.000000 std=0.000614 min=-0.002589 max=0.002743
2026-08-22 17:55:13,211 INFO     crossing_head: train loss=7.0883 pos_dist=10.365m dt_mae=1.945s | val loss=5.5327 pos_dist=9.790m dt_mae=1.635s
2026-08-22 17:55:13,211 INFO     goal_dist_delta_head: train loss=0.00348 mae=(left 2.388m, right 2.413m) | val loss=0.00343 mae=(left 2.371m, right 2.413m)
2026-08-22 17:55:13,211 INFO     short_horizon_probes: train loss=0.00041 rmse_norm=(0.2s 0.0150, 1.0s 0.0134) | val loss=0.00040 rmse_norm=(0.2s 0.0149, 1.0s 0.0134)
2026-08-22 17:55:13,211 INFO     val_loss_delta (epoch-over-epoch): -0.000003
2026-08-22 17:55:13,211 INFO         train pos_rmse     by horizon (m): [0.981  1.0109 1.8098 1.6324 1.5271], mean: 1.3922 m
2026-08-22 17:55:13,212 INFO         val   pos_rmse     by horizon (m): [0.9807 1.0216 1.8109 1.6215 1.5167], mean: 1.3903 m
2026-08-22 17:55:13,212 INFO         train pos_dist     by horizon (m): [1.1253 1.2152 2.1872 1.8873 1.6928], mean: 1.6215 m
2026-08-22 17:55:13,212 INFO         val   pos_dist     by horizon (m): [1.1281 1.2265 2.1927 1.8683 1.6846], mean: 1.6201 m
2026-08-22 17:55:13,212 INFO         train vel_rmse     by horizon (m/s): [1.3202 0.6216 0.4913 0.4666 0.571 ], mean: 0.6941 m/s
2026-08-22 17:55:13,212 INFO         val   vel_rmse     by horizon (m/s): [1.3295 0.621  0.4901 0.4663 0.5696], mean: 0.6953 m/s
2026-08-22 17:55:13,212 INFO         train vel_dist     by horizon (m/s): [1.5403 0.6604 0.5455 0.5309 0.6103], mean: 0.7775 m/s
2026-08-22 17:55:13,212 INFO         val   vel_dist     by horizon (m/s): [1.5552 0.6615 0.5469 0.536  0.6147], mean: 0.7829 m/s
2026-08-22 17:55:13,212 INFO         train heading_rmse by horizon: [0.0503 0.0403 0.0272 0.0067 0.0058], mean: 0.0261
2026-08-22 17:55:13,212 INFO         val   heading_rmse by horizon: [0.0505 0.0409 0.0268 0.0071 0.0056], mean: 0.0262
2026-08-22 17:55:13,212 INFO         train heading_dist by horizon: [0.0358 0.0181 0.007  0.0042 0.0042], mean: 0.0139
2026-08-22 17:55:13,212 INFO         val   heading_dist by horizon: [0.036  0.0179 0.0067 0.0041 0.0039], mean: 0.0137
2026-08-22 17:55:13,213 INFO         train stamina_rmse by horizon: [0.0116 0.0098 0.0042 0.0074 0.0259], mean: 0.0118
2026-08-22 17:55:13,213 INFO         val   stamina_rmse by horizon: [0.0115 0.0098 0.0041 0.0075 0.026 ], mean: 0.0118
2026-08-22 17:55:13,213 INFO     train pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:55:13,213 INFO     val   pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:55:13,213 INFO     train pos       %-of-persistence by horizon: [238.6  52.2  27.5  14.    6.2]
2026-08-22 17:55:13,213 INFO     val   pos       %-of-persistence by horizon: [238.5  52.8  27.6  13.9   6.2]
2026-08-22 17:55:13,213 INFO     train vel       R2 by horizon: [0.574 0.929 0.965 0.968 0.951], mean: 0.877
2026-08-22 17:55:13,213 INFO     val   vel       R2 by horizon: [0.568 0.929 0.965 0.968 0.951], mean: 0.876
2026-08-22 17:55:13,213 INFO     train vel       %-of-persistence by horizon: [157.1  22.7  14.4  13.7  16.9]
2026-08-22 17:55:13,214 INFO     val   vel       %-of-persistence by horizon: [158.2  22.7  14.3  13.6  16.9]
2026-08-22 17:55:13,214 INFO     train heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:55:13,214 INFO     val   heading   R2 by horizon: [0.995 0.997 0.999 1.    1.   ], mean: 0.998
2026-08-22 17:55:13,214 INFO     train heading   %-of-persistence by horizon: [10.3  4.3  2.8  0.7  0.6]
2026-08-22 17:55:13,214 INFO     val   heading   %-of-persistence by horizon: [10.3  4.4  2.7  0.7  0.6]
2026-08-22 17:55:13,214 INFO     train stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:55:13,215 INFO     val   stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:55:13,215 INFO     train stamina   %-of-persistence by horizon: [1310.7  221.7   31.5   33.9   59.9]
2026-08-22 17:55:13,215 INFO     val   stamina   %-of-persistence by horizon: [1302.9  221.3   30.9   34.4   60. ]
2026-08-22 17:56:09,334 INFO epoch 16/500: train_loss=0.0094  pair_loss=0.0016  t0_loss=0.0020  val_loss=0.0093  best=0.0093  (improved)
2026-08-22 17:56:09,334 INFO     grad_norm: mean=0.238994 std=0.086869 min=0.092021 max=0.938188
2026-08-22 17:56:09,334 INFO     train_loss_delta (batch-to-batch): mean=0.000000 std=0.000602 min=-0.003724 max=0.003730
2026-08-22 17:56:09,334 INFO     crossing_head: train loss=7.0705 pos_dist=10.323m dt_mae=1.938s | val loss=5.5415 pos_dist=9.780m dt_mae=1.680s
2026-08-22 17:56:09,334 INFO     goal_dist_delta_head: train loss=0.00347 mae=(left 2.381m, right 2.409m) | val loss=0.00343 mae=(left 2.367m, right 2.412m)
2026-08-22 17:56:09,334 INFO     short_horizon_probes: train loss=0.00040 rmse_norm=(0.2s 0.0149, 1.0s 0.0134) | val loss=0.00040 rmse_norm=(0.2s 0.0148, 1.0s 0.0134)
2026-08-22 17:56:09,334 INFO     val_loss_delta (epoch-over-epoch): -0.000139
2026-08-22 17:56:09,334 INFO         train pos_rmse     by horizon (m): [0.976  1.0127 1.8107 1.6328 1.5298], mean: 1.3924 m
2026-08-22 17:56:09,334 INFO         val   pos_rmse     by horizon (m): [0.9671 1.0072 1.8026 1.6335 1.5211], mean: 1.3863 m
2026-08-22 17:56:09,334 INFO         train pos_dist     by horizon (m): [1.1191 1.2176 2.187  1.8854 1.6953], mean: 1.6209 m
2026-08-22 17:56:09,335 INFO         val   pos_dist     by horizon (m): [1.1102 1.2124 2.1775 1.8824 1.6927], mean: 1.6150 m
2026-08-22 17:56:09,335 INFO         train vel_rmse     by horizon (m/s): [1.321  0.6209 0.4915 0.4672 0.5719], mean: 0.6945 m/s
2026-08-22 17:56:09,335 INFO         val   vel_rmse     by horizon (m/s): [1.3154 0.6157 0.49   0.4639 0.5668], mean: 0.6904 m/s
2026-08-22 17:56:09,335 INFO         train vel_dist     by horizon (m/s): [1.5414 0.6598 0.546  0.532  0.6108], mean: 0.7780 m/s
2026-08-22 17:56:09,335 INFO         val   vel_dist     by horizon (m/s): [1.5321 0.6557 0.5426 0.5291 0.6022], mean: 0.7724 m/s
2026-08-22 17:56:09,335 INFO         train heading_rmse by horizon: [0.0503 0.0403 0.0272 0.0067 0.0058], mean: 0.0260
2026-08-22 17:56:09,336 INFO         val   heading_rmse by horizon: [0.0496 0.0406 0.0268 0.0071 0.0056], mean: 0.0259
2026-08-22 17:56:09,336 INFO         train heading_dist by horizon: [0.0358 0.018  0.0069 0.0042 0.0041], mean: 0.0138
2026-08-22 17:56:09,336 INFO         val   heading_dist by horizon: [0.0353 0.0179 0.0067 0.004  0.0039], mean: 0.0136
2026-08-22 17:56:09,336 INFO         train stamina_rmse by horizon: [0.0116 0.0098 0.0041 0.0074 0.0259], mean: 0.0118
2026-08-22 17:56:09,336 INFO         val   stamina_rmse by horizon: [0.0119 0.0102 0.0042 0.0072 0.0255], mean: 0.0118
2026-08-22 17:56:09,337 INFO     train pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:56:09,337 INFO     val   pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:56:09,337 INFO     train pos       %-of-persistence by horizon: [237.4  52.3  27.6  14.    6.2]
2026-08-22 17:56:09,337 INFO     val   pos       %-of-persistence by horizon: [235.2  52.   27.4  14.    6.2]
2026-08-22 17:56:09,338 INFO     train vel       R2 by horizon: [0.573 0.929 0.965 0.968 0.95 ], mean: 0.877
2026-08-22 17:56:09,338 INFO     val   vel       R2 by horizon: [0.577 0.93  0.966 0.969 0.951], mean: 0.878
2026-08-22 17:56:09,338 INFO     train vel       %-of-persistence by horizon: [157.2  22.7  14.4  13.7  16.9]
2026-08-22 17:56:09,338 INFO     val   vel       %-of-persistence by horizon: [156.5  22.5  14.3  13.6  16.8]
2026-08-22 17:56:09,338 INFO     train heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:56:09,339 INFO     val   heading   R2 by horizon: [0.995 0.997 0.999 1.    1.   ], mean: 0.998
2026-08-22 17:56:09,339 INFO     train heading   %-of-persistence by horizon: [10.3  4.3  2.8  0.7  0.6]
2026-08-22 17:56:09,339 INFO     val   heading   %-of-persistence by horizon: [10.2  4.4  2.7  0.7  0.6]
2026-08-22 17:56:09,339 INFO     train stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:56:09,339 INFO     val   stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:56:09,339 INFO     train stamina   %-of-persistence by horizon: [1311.3  221.2   31.3   33.9   59.9]
2026-08-22 17:56:09,339 INFO     val   stamina   %-of-persistence by horizon: [1347.7  230.2   31.8   32.7   59. ]
2026-08-22 17:56:09,343 INFO Saved 'midtrain_latest' checkpoint to checkpoints/physics_pretrain/player_encoder_11.midtrain_latest.pt
2026-08-22 17:57:41,797 INFO epoch 17/500: train_loss=0.0094  pair_loss=0.0016  t0_loss=0.0019  val_loss=0.0092  best=0.0092  (improved)
2026-08-22 17:57:41,797 INFO     grad_norm: mean=0.246334 std=0.086666 min=0.095619 max=1.002150
2026-08-22 17:57:41,797 INFO     train_loss_delta (batch-to-batch): mean=0.000000 std=0.000616 min=-0.002258 max=0.002256
2026-08-22 17:57:41,797 INFO     crossing_head: train loss=7.0562 pos_dist=10.314m dt_mae=1.935s | val loss=5.5150 pos_dist=9.764m dt_mae=1.641s
2026-08-22 17:57:41,797 INFO     goal_dist_delta_head: train loss=0.00346 mae=(left 2.374m, right 2.406m) | val loss=0.00342 mae=(left 2.360m, right 2.406m)
2026-08-22 17:57:41,797 INFO     short_horizon_probes: train loss=0.00040 rmse_norm=(0.2s 0.0149, 1.0s 0.0134) | val loss=0.00040 rmse_norm=(0.2s 0.0148, 1.0s 0.0133)
2026-08-22 17:57:41,797 INFO     val_loss_delta (epoch-over-epoch): -0.000059
2026-08-22 17:57:41,797 INFO         train pos_rmse     by horizon (m): [0.9725 1.0125 1.8113 1.6346 1.5374], mean: 1.3937 m
2026-08-22 17:57:41,797 INFO         val   pos_rmse     by horizon (m): [0.9786 1.0097 1.7948 1.6357 1.5203], mean: 1.3878 m
2026-08-22 17:57:41,797 INFO         train pos_dist     by horizon (m): [1.115  1.2176 2.1871 1.888  1.7044], mean: 1.6224 m
2026-08-22 17:57:41,797 INFO         val   pos_dist     by horizon (m): [1.1194 1.2088 2.1748 1.8789 1.6877], mean: 1.6139 m
2026-08-22 17:57:41,798 INFO         train vel_rmse     by horizon (m/s): [1.3198 0.6201 0.4924 0.4676 0.573 ], mean: 0.6946 m/s
2026-08-22 17:57:41,798 INFO         val   vel_rmse     by horizon (m/s): [1.321  0.6143 0.4893 0.4638 0.5663], mean: 0.6909 m/s
2026-08-22 17:57:41,798 INFO         train vel_dist     by horizon (m/s): [1.5397 0.6589 0.5469 0.5321 0.6111], mean: 0.7777 m/s
2026-08-22 17:57:41,798 INFO         val   vel_dist     by horizon (m/s): [1.5409 0.6532 0.5426 0.5288 0.602 ], mean: 0.7735 m/s
2026-08-22 17:57:41,798 INFO         train heading_rmse by horizon: [0.0504 0.0404 0.0271 0.0067 0.0058], mean: 0.0261
2026-08-22 17:57:41,798 INFO         val   heading_rmse by horizon: [0.049  0.0406 0.0267 0.0071 0.0056], mean: 0.0258
2026-08-22 17:57:41,798 INFO         train heading_dist by horizon: [0.0359 0.0181 0.0069 0.0042 0.0042], mean: 0.0139
2026-08-22 17:57:41,798 INFO         val   heading_dist by horizon: [0.035  0.018  0.0066 0.004  0.004 ], mean: 0.0135
2026-08-22 17:57:41,798 INFO         train stamina_rmse by horizon: [0.0116 0.0098 0.0041 0.0074 0.0259], mean: 0.0118
2026-08-22 17:57:41,798 INFO         val   stamina_rmse by horizon: [0.0121 0.0103 0.0043 0.007  0.0254], mean: 0.0118
2026-08-22 17:57:41,798 INFO     train pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:57:41,799 INFO     val   pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:57:41,799 INFO     train pos       %-of-persistence by horizon: [236.6  52.3  27.6  14.    6.3]
2026-08-22 17:57:41,799 INFO     val   pos       %-of-persistence by horizon: [238.   52.1  27.3  14.    6.2]
2026-08-22 17:57:41,799 INFO     train vel       R2 by horizon: [0.574 0.929 0.965 0.968 0.95 ], mean: 0.877
2026-08-22 17:57:41,799 INFO     val   vel       R2 by horizon: [0.573 0.93  0.966 0.969 0.951], mean: 0.878
2026-08-22 17:57:41,799 INFO     train vel       %-of-persistence by horizon: [157.1  22.6  14.4  13.7  17. ]
2026-08-22 17:57:41,799 INFO     val   vel       %-of-persistence by horizon: [157.2  22.4  14.3  13.6  16.8]
2026-08-22 17:57:41,799 INFO     train heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:57:41,799 INFO     val   heading   R2 by horizon: [0.995 0.997 0.999 1.    1.   ], mean: 0.998
2026-08-22 17:57:41,799 INFO     train heading   %-of-persistence by horizon: [10.3  4.4  2.8  0.7  0.6]
2026-08-22 17:57:41,800 INFO     val   heading   %-of-persistence by horizon: [10.   4.4  2.7  0.7  0.6]
2026-08-22 17:57:41,800 INFO     train stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:57:41,800 INFO     val   stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:57:41,800 INFO     train stamina   %-of-persistence by horizon: [1314.9  221.3   31.3   33.9   59.9]
2026-08-22 17:57:41,800 INFO     val   stamina   %-of-persistence by horizon: [1372.3  232.5   32.3   31.8   58.7]
2026-08-22 17:57:41,803 INFO Saved 'midtrain_latest' checkpoint to checkpoints/physics_pretrain/player_encoder_11.midtrain_latest.pt
2026-08-22 17:59:14,959 INFO epoch 18/500: train_loss=0.0094  pair_loss=0.0016  t0_loss=0.0020  val_loss=0.0093  best=0.0092  (patience 1/20)
2026-08-22 17:59:14,959 INFO     grad_norm: mean=0.250447 std=0.089506 min=0.101148 max=0.789112
2026-08-22 17:59:14,959 INFO     train_loss_delta (batch-to-batch): mean=0.000001 std=0.000617 min=-0.001963 max=0.002335
2026-08-22 17:59:14,959 INFO     crossing_head: train loss=7.0387 pos_dist=10.272m dt_mae=1.930s | val loss=5.5064 pos_dist=9.770m dt_mae=1.645s
2026-08-22 17:59:14,959 INFO     goal_dist_delta_head: train loss=0.00345 mae=(left 2.369m, right 2.402m) | val loss=0.00341 mae=(left 2.349m, right 2.401m)
2026-08-22 17:59:14,959 INFO     short_horizon_probes: train loss=0.00040 rmse_norm=(0.2s 0.0148, 1.0s 0.0134) | val loss=0.00040 rmse_norm=(0.2s 0.0148, 1.0s 0.0133)
2026-08-22 17:59:14,959 INFO     val_loss_delta (epoch-over-epoch): 0.000120
2026-08-22 17:59:14,959 INFO         train pos_rmse     by horizon (m): [0.9711 1.0121 1.8099 1.6366 1.5419], mean: 1.3943 m
2026-08-22 17:59:14,959 INFO         val   pos_rmse     by horizon (m): [0.9683 1.0201 1.7972 1.6374 1.5407], mean: 1.3927 m
2026-08-22 17:59:14,959 INFO         train pos_dist     by horizon (m): [1.1125 1.2176 2.1844 1.8893 1.7093], mean: 1.6226 m
2026-08-22 17:59:14,960 INFO         val   pos_dist     by horizon (m): [1.109  1.2299 2.1691 1.8989 1.712 ], mean: 1.6238 m
2026-08-22 17:59:14,960 INFO         train vel_rmse     by horizon (m/s): [1.3203 0.6195 0.4922 0.4672 0.5723], mean: 0.6943 m/s
2026-08-22 17:59:14,960 INFO         val   vel_rmse     by horizon (m/s): [1.3281 0.6244 0.4913 0.4614 0.5687], mean: 0.6948 m/s
2026-08-22 17:59:14,960 INFO         train vel_dist     by horizon (m/s): [1.5403 0.6582 0.5467 0.5317 0.6101], mean: 0.7774 m/s
2026-08-22 17:59:14,960 INFO         val   vel_dist     by horizon (m/s): [1.5525 0.6641 0.5459 0.5248 0.6066], mean: 0.7788 m/s
2026-08-22 17:59:14,960 INFO         train heading_rmse by horizon: [0.0502 0.0404 0.0271 0.0067 0.0058], mean: 0.0260
2026-08-22 17:59:14,960 INFO         val   heading_rmse by horizon: [0.0496 0.0409 0.0268 0.007  0.0056], mean: 0.0260
2026-08-22 17:59:14,960 INFO         train heading_dist by horizon: [0.0357 0.0181 0.0069 0.0042 0.0042], mean: 0.0138
2026-08-22 17:59:14,960 INFO         val   heading_dist by horizon: [0.0353 0.018  0.0067 0.004  0.0039], mean: 0.0136
2026-08-22 17:59:14,960 INFO         train stamina_rmse by horizon: [0.0117 0.0098 0.0041 0.0074 0.0259], mean: 0.0118
2026-08-22 17:59:14,960 INFO         val   stamina_rmse by horizon: [0.0118 0.0098 0.0043 0.0074 0.0258], mean: 0.0118
2026-08-22 17:59:14,961 INFO     train pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:59:14,961 INFO     val   pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:59:14,961 INFO     train pos       %-of-persistence by horizon: [236.2  52.3  27.5  14.    6.3]
2026-08-22 17:59:14,961 INFO     val   pos       %-of-persistence by horizon: [235.5  52.7  27.3  14.1   6.3]
2026-08-22 17:59:14,961 INFO     train vel       R2 by horizon: [0.574 0.929 0.965 0.968 0.95 ], mean: 0.877
2026-08-22 17:59:14,962 INFO     val   vel       R2 by horizon: [0.569 0.928 0.965 0.969 0.951], mean: 0.876
2026-08-22 17:59:14,962 INFO     train vel       %-of-persistence by horizon: [157.1  22.6  14.4  13.7  17. ]
2026-08-22 17:59:14,962 INFO     val   vel       %-of-persistence by horizon: [158.   22.8  14.4  13.5  16.8]
2026-08-22 17:59:14,962 INFO     train heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:59:14,962 INFO     val   heading   R2 by horizon: [0.995 0.997 0.999 1.    1.   ], mean: 0.998
2026-08-22 17:59:14,963 INFO     train heading   %-of-persistence by horizon: [10.3  4.4  2.8  0.7  0.6]
2026-08-22 17:59:14,963 INFO     val   heading   %-of-persistence by horizon: [10.2  4.4  2.7  0.7  0.6]
2026-08-22 17:59:14,963 INFO     train stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:59:14,963 INFO     val   stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:59:14,963 INFO     train stamina   %-of-persistence by horizon: [1318.9  221.5   31.3   33.9   59.8]
2026-08-22 17:59:14,963 INFO     val   stamina   %-of-persistence by horizon: [1332.6  222.8   32.5   33.8   59.7]
2026-08-22 17:59:56,696 INFO epoch 19/500: train_loss=0.0094  pair_loss=0.0016  t0_loss=0.0020  val_loss=0.0093  best=0.0092  (patience 2/20)
2026-08-22 17:59:56,696 INFO     grad_norm: mean=0.249865 std=0.092511 min=0.101459 max=0.764175
2026-08-22 17:59:56,696 INFO     train_loss_delta (batch-to-batch): mean=0.000000 std=0.000605 min=-0.002084 max=0.002047
2026-08-22 17:59:56,696 INFO     crossing_head: train loss=7.0235 pos_dist=10.260m dt_mae=1.925s | val loss=5.4934 pos_dist=9.766m dt_mae=1.633s
2026-08-22 17:59:56,696 INFO     goal_dist_delta_head: train loss=0.00345 mae=(left 2.362m, right 2.399m) | val loss=0.00340 mae=(left 2.346m, right 2.399m)
2026-08-22 17:59:56,696 INFO     short_horizon_probes: train loss=0.00040 rmse_norm=(0.2s 0.0148, 1.0s 0.0133) | val loss=0.00039 rmse_norm=(0.2s 0.0147, 1.0s 0.0133)
2026-08-22 17:59:56,696 INFO     val_loss_delta (epoch-over-epoch): -0.000036
2026-08-22 17:59:56,696 INFO         train pos_rmse     by horizon (m): [0.9696 1.0132 1.8102 1.6389 1.5525], mean: 1.3969 m
2026-08-22 17:59:56,697 INFO         val   pos_rmse     by horizon (m): [0.9706 1.0113 1.8023 1.6367 1.5438], mean: 1.3929 m
2026-08-22 17:59:56,697 INFO         train pos_dist     by horizon (m): [1.111  1.2188 2.1838 1.8913 1.7218], mean: 1.6254 m
2026-08-22 17:59:56,697 INFO         val   pos_dist     by horizon (m): [1.1126 1.2121 2.1837 1.8819 1.7194], mean: 1.6219 m
2026-08-22 17:59:56,697 INFO         train vel_rmse     by horizon (m/s): [1.321  0.6187 0.4921 0.4675 0.573 ], mean: 0.6944 m/s
2026-08-22 17:59:56,697 INFO         val   vel_rmse     by horizon (m/s): [1.319  0.6159 0.4902 0.4637 0.5683], mean: 0.6914 m/s
2026-08-22 17:59:56,697 INFO         train vel_dist     by horizon (m/s): [1.5411 0.657  0.5465 0.532  0.6104], mean: 0.7774 m/s
2026-08-22 17:59:56,697 INFO         val   vel_dist     by horizon (m/s): [1.5387 0.654  0.5425 0.528  0.6059], mean: 0.7738 m/s
2026-08-22 17:59:56,697 INFO         train heading_rmse by horizon: [0.0503 0.0405 0.0271 0.0067 0.0059], mean: 0.0261
2026-08-22 17:59:56,697 INFO         val   heading_rmse by horizon: [0.0495 0.0408 0.0268 0.0071 0.0056], mean: 0.0259
2026-08-22 17:59:56,697 INFO         train heading_dist by horizon: [0.0358 0.0181 0.0069 0.0042 0.0042], mean: 0.0139
2026-08-22 17:59:56,698 INFO         val   heading_dist by horizon: [0.0352 0.018  0.0066 0.0041 0.004 ], mean: 0.0136
2026-08-22 17:59:56,698 INFO         train stamina_rmse by horizon: [0.0116 0.0097 0.0041 0.0074 0.0259], mean: 0.0118
2026-08-22 17:59:56,698 INFO         val   stamina_rmse by horizon: [0.012  0.0102 0.0042 0.0071 0.0255], mean: 0.0118
2026-08-22 17:59:56,699 INFO     train pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:59:56,699 INFO     val   pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 17:59:56,699 INFO     train pos       %-of-persistence by horizon: [235.8  52.3  27.5  14.1   6.3]
2026-08-22 17:59:56,699 INFO     val   pos       %-of-persistence by horizon: [236.   52.2  27.4  14.    6.3]
2026-08-22 17:59:56,699 INFO     train vel       R2 by horizon: [0.573 0.929 0.965 0.968 0.95 ], mean: 0.877
2026-08-22 17:59:56,700 INFO     val   vel       R2 by horizon: [0.574 0.93  0.965 0.969 0.951], mean: 0.878
2026-08-22 17:59:56,700 INFO     train vel       %-of-persistence by horizon: [157.2  22.6  14.4  13.7  17. ]
2026-08-22 17:59:56,700 INFO     val   vel       %-of-persistence by horizon: [156.9  22.5  14.3  13.6  16.8]
2026-08-22 17:59:56,700 INFO     train heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 17:59:56,700 INFO     val   heading   R2 by horizon: [0.995 0.997 0.999 1.    1.   ], mean: 0.998
2026-08-22 17:59:56,701 INFO     train heading   %-of-persistence by horizon: [10.3  4.4  2.8  0.7  0.6]
2026-08-22 17:59:56,701 INFO     val   heading   %-of-persistence by horizon: [10.1  4.4  2.7  0.7  0.6]
2026-08-22 17:59:56,701 INFO     train stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:59:56,701 INFO     val   stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 17:59:56,701 INFO     train stamina   %-of-persistence by horizon: [1317.1  220.8   31.2   34.    59.9]
2026-08-22 17:59:56,701 INFO     val   stamina   %-of-persistence by horizon: [1360.9  230.1   32.1   32.4   58.9]
2026-08-22 18:00:23,159 INFO epoch 20/500: train_loss=0.0094  pair_loss=0.0016  t0_loss=0.0020  val_loss=0.0095  best=0.0092  (patience 3/20)
2026-08-22 18:00:23,159 INFO     grad_norm: mean=0.249480 std=0.091975 min=0.092114 max=0.718573
2026-08-22 18:00:23,159 INFO     train_loss_delta (batch-to-batch): mean=0.000000 std=0.000616 min=-0.002409 max=0.002957
2026-08-22 18:00:23,159 INFO     crossing_head: train loss=7.0060 pos_dist=10.299m dt_mae=1.921s | val loss=5.4790 pos_dist=9.759m dt_mae=1.607s
2026-08-22 18:00:23,159 INFO     goal_dist_delta_head: train loss=0.00344 mae=(left 2.357m, right 2.397m) | val loss=0.00340 mae=(left 2.338m, right 2.393m)
2026-08-22 18:00:23,159 INFO     short_horizon_probes: train loss=0.00040 rmse_norm=(0.2s 0.0148, 1.0s 0.0133) | val loss=0.00039 rmse_norm=(0.2s 0.0147, 1.0s 0.0133)
2026-08-22 18:00:23,159 INFO     val_loss_delta (epoch-over-epoch): 0.000203
2026-08-22 18:00:23,160 INFO         train pos_rmse     by horizon (m): [0.9683 1.0124 1.8103 1.6409 1.5628], mean: 1.3989 m
2026-08-22 18:00:23,160 INFO         val   pos_rmse     by horizon (m): [0.9674 1.0248 1.8106 1.6335 1.5625], mean: 1.3998 m
2026-08-22 18:00:23,160 INFO         train pos_dist     by horizon (m): [1.1093 1.2183 2.1834 1.8942 1.7349], mean: 1.6280 m
2026-08-22 18:00:23,160 INFO         val   pos_dist     by horizon (m): [1.1104 1.2298 2.1939 1.8772 1.7401], mean: 1.6303 m
2026-08-22 18:00:23,160 INFO         train vel_rmse     by horizon (m/s): [1.32   0.6184 0.4928 0.4675 0.5727], mean: 0.6943 m/s
2026-08-22 18:00:23,160 INFO         val   vel_rmse     by horizon (m/s): [1.3377 0.6295 0.4929 0.4649 0.5754], mean: 0.7001 m/s
2026-08-22 18:00:23,160 INFO         train vel_dist     by horizon (m/s): [1.5399 0.6567 0.5473 0.5319 0.61  ], mean: 0.7772 m/s
2026-08-22 18:00:23,160 INFO         val   vel_dist     by horizon (m/s): [1.5684 0.672  0.5486 0.5318 0.6236], mean: 0.7889 m/s
2026-08-22 18:00:23,160 INFO         train heading_rmse by horizon: [0.0503 0.0405 0.0271 0.0067 0.0059], mean: 0.0261
2026-08-22 18:00:23,160 INFO         val   heading_rmse by horizon: [0.0506 0.0411 0.0268 0.0071 0.0057], mean: 0.0263
2026-08-22 18:00:23,160 INFO         train heading_dist by horizon: [0.0359 0.0182 0.0069 0.0042 0.0042], mean: 0.0139
2026-08-22 18:00:23,161 INFO         val   heading_dist by horizon: [0.036  0.0181 0.007  0.0042 0.0041], mean: 0.0139
2026-08-22 18:00:23,161 INFO         train stamina_rmse by horizon: [0.0117 0.0097 0.0041 0.0074 0.0259], mean: 0.0118
2026-08-22 18:00:23,161 INFO         val   stamina_rmse by horizon: [0.0117 0.0097 0.0042 0.0076 0.026 ], mean: 0.0118
2026-08-22 18:00:23,161 INFO     train pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 18:00:23,161 INFO     val   pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 18:00:23,161 INFO     train pos       %-of-persistence by horizon: [235.5  52.3  27.5  14.1   6.4]
2026-08-22 18:00:23,161 INFO     val   pos       %-of-persistence by horizon: [235.3  52.9  27.5  14.    6.4]
2026-08-22 18:00:23,161 INFO     train vel       R2 by horizon: [0.574 0.929 0.965 0.968 0.95 ], mean: 0.877
2026-08-22 18:00:23,161 INFO     val   vel       R2 by horizon: [0.562 0.927 0.965 0.969 0.95 ], mean: 0.875
2026-08-22 18:00:23,161 INFO     train vel       %-of-persistence by horizon: [157.1  22.6  14.4  13.7  17. ]
2026-08-22 18:00:23,162 INFO     val   vel       %-of-persistence by horizon: [159.2  23.   14.4  13.6  17. ]
2026-08-22 18:00:23,162 INFO     train heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 18:00:23,162 INFO     val   heading   R2 by horizon: [0.995 0.997 0.999 1.    1.   ], mean: 0.998
2026-08-22 18:00:23,162 INFO     train heading   %-of-persistence by horizon: [10.3  4.4  2.8  0.7  0.6]
2026-08-22 18:00:23,162 INFO     val   heading   %-of-persistence by horizon: [10.4  4.4  2.7  0.7  0.6]
2026-08-22 18:00:23,162 INFO     train stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 18:00:23,162 INFO     val   stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 18:00:23,162 INFO     train stamina   %-of-persistence by horizon: [1320.3  220.8   31.1   34.    59.9]
2026-08-22 18:00:23,162 INFO     val   stamina   %-of-persistence by horizon: [1317.7  219.5   31.5   34.5   60.1]
2026-08-22 18:00:46,926 INFO epoch 21/500: train_loss=0.0094  pair_loss=0.0016  t0_loss=0.0020  val_loss=0.0095  best=0.0092  (patience 4/20)
2026-08-22 18:00:46,927 INFO     grad_norm: mean=0.239568 std=0.082054 min=0.095837 max=0.882268
2026-08-22 18:00:46,927 INFO     train_loss_delta (batch-to-batch): mean=-0.000000 std=0.000612 min=-0.002130 max=0.001932
2026-08-22 18:00:46,927 INFO     crossing_head: train loss=6.9850 pos_dist=10.323m dt_mae=1.915s | val loss=5.4665 pos_dist=9.745m dt_mae=1.615s
2026-08-22 18:00:46,927 INFO     goal_dist_delta_head: train loss=0.00343 mae=(left 2.350m, right 2.394m) | val loss=0.00339 mae=(left 2.330m, right 2.398m)
2026-08-22 18:00:46,927 INFO     short_horizon_probes: train loss=0.00040 rmse_norm=(0.2s 0.0148, 1.0s 0.0133) | val loss=0.00039 rmse_norm=(0.2s 0.0147, 1.0s 0.0133)
2026-08-22 18:00:46,927 INFO     val_loss_delta (epoch-over-epoch): 0.000009
2026-08-22 18:00:46,927 INFO         train pos_rmse     by horizon (m): [0.9657 1.013  1.8126 1.643  1.5757], mean: 1.4020 m
2026-08-22 18:00:46,927 INFO         val   pos_rmse     by horizon (m): [0.9497 1.0244 1.8189 1.6421 1.582 ], mean: 1.4034 m
2026-08-22 18:00:46,928 INFO         train pos_dist     by horizon (m): [1.1064 1.2195 2.1851 1.8966 1.751 ], mean: 1.6317 m
2026-08-22 18:00:46,928 INFO         val   pos_dist     by horizon (m): [1.0951 1.2435 2.1847 1.9153 1.7723], mean: 1.6422 m
2026-08-22 18:00:46,928 INFO         train vel_rmse     by horizon (m/s): [1.3201 0.6178 0.4933 0.4683 0.5735], mean: 0.6946 m/s
2026-08-22 18:00:46,928 INFO         val   vel_rmse     by horizon (m/s): [1.3193 0.6169 0.491  0.4696 0.5719], mean: 0.6937 m/s
2026-08-22 18:00:46,928 INFO         train vel_dist     by horizon (m/s): [1.54   0.6563 0.548  0.5329 0.6106], mean: 0.7775 m/s
2026-08-22 18:00:46,928 INFO         val   vel_dist     by horizon (m/s): [1.5392 0.6571 0.5416 0.5374 0.6127], mean: 0.7776 m/s
2026-08-22 18:00:46,928 INFO         train heading_rmse by horizon: [0.0504 0.0406 0.0271 0.0067 0.0058], mean: 0.0261
2026-08-22 18:00:46,928 INFO         val   heading_rmse by horizon: [0.0506 0.0412 0.0268 0.0072 0.0057], mean: 0.0263
2026-08-22 18:00:46,928 INFO         train heading_dist by horizon: [0.0359 0.0182 0.0069 0.0042 0.0042], mean: 0.0139
2026-08-22 18:00:46,928 INFO         val   heading_dist by horizon: [0.0361 0.0183 0.0069 0.0041 0.004 ], mean: 0.0139
2026-08-22 18:00:46,928 INFO         train stamina_rmse by horizon: [0.0117 0.0097 0.0041 0.0075 0.0259], mean: 0.0118
2026-08-22 18:00:46,929 INFO         val   stamina_rmse by horizon: [0.0116 0.0097 0.004  0.0076 0.026 ], mean: 0.0118
2026-08-22 18:00:46,929 INFO     train pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 18:00:46,929 INFO     val   pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 18:00:46,929 INFO     train pos       %-of-persistence by horizon: [234.9  52.3  27.6  14.1   6.4]
2026-08-22 18:00:46,929 INFO     val   pos       %-of-persistence by horizon: [231.   52.9  27.7  14.1   6.5]
2026-08-22 18:00:46,929 INFO     train vel       R2 by horizon: [0.574 0.929 0.965 0.968 0.95 ], mean: 0.877
2026-08-22 18:00:46,929 INFO     val   vel       R2 by horizon: [0.574 0.93  0.965 0.968 0.95 ], mean: 0.878
2026-08-22 18:00:46,929 INFO     train vel       %-of-persistence by horizon: [157.1  22.5  14.4  13.7  17. ]
2026-08-22 18:00:46,930 INFO     val   vel       %-of-persistence by horizon: [157.   22.5  14.4  13.7  16.9]
2026-08-22 18:00:46,930 INFO     train heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 18:00:46,930 INFO     val   heading   R2 by horizon: [0.995 0.997 0.999 1.    1.   ], mean: 0.998
2026-08-22 18:00:46,930 INFO     train heading   %-of-persistence by horizon: [10.3  4.4  2.8  0.7  0.6]
2026-08-22 18:00:46,930 INFO     val   heading   %-of-persistence by horizon: [10.4  4.4  2.7  0.7  0.6]
2026-08-22 18:00:46,930 INFO     train stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 18:00:46,930 INFO     val   stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 18:00:46,930 INFO     train stamina   %-of-persistence by horizon: [1320.6  220.4   31.1   34.1   59.9]
2026-08-22 18:00:46,930 INFO     val   stamina   %-of-persistence by horizon: [1309.8  220.1   30.1   34.6   60.1]
2026-08-22 18:01:16,584 INFO epoch 22/500: train_loss=0.0095  pair_loss=0.0016  t0_loss=0.0020  val_loss=0.0095  best=0.0092  (patience 5/20)
2026-08-22 18:01:16,585 INFO     grad_norm: mean=0.248834 std=0.099956 min=0.090785 max=1.468911
2026-08-22 18:01:16,585 INFO     train_loss_delta (batch-to-batch): mean=0.000001 std=0.000643 min=-0.002591 max=0.001983
2026-08-22 18:01:16,585 INFO     crossing_head: train loss=6.9620 pos_dist=10.263m dt_mae=1.909s | val loss=5.4700 pos_dist=9.748m dt_mae=1.660s
2026-08-22 18:01:16,585 INFO     goal_dist_delta_head: train loss=0.00342 mae=(left 2.344m, right 2.393m) | val loss=0.00338 mae=(left 2.324m, right 2.395m)
2026-08-22 18:01:16,585 INFO     short_horizon_probes: train loss=0.00039 rmse_norm=(0.2s 0.0148, 1.0s 0.0133) | val loss=0.00039 rmse_norm=(0.2s 0.0147, 1.0s 0.0133)
2026-08-22 18:01:16,585 INFO     val_loss_delta (epoch-over-epoch): -0.000048
2026-08-22 18:01:16,585 INFO         train pos_rmse     by horizon (m): [0.9656 1.0128 1.8146 1.6452 1.5924], mean: 1.4061 m
2026-08-22 18:01:16,585 INFO         val   pos_rmse     by horizon (m): [0.9625 1.0139 1.8063 1.6437 1.5894], mean: 1.4031 m
2026-08-22 18:01:16,585 INFO         train pos_dist     by horizon (m): [1.1062 1.2199 2.1869 1.8999 1.7728], mean: 1.6371 m
2026-08-22 18:01:16,586 INFO         val   pos_dist     by horizon (m): [1.1053 1.2277 2.1748 1.9041 1.7743], mean: 1.6372 m
2026-08-22 18:01:16,586 INFO         train vel_rmse     by horizon (m/s): [1.3198 0.6176 0.4939 0.4687 0.5741], mean: 0.6948 m/s
2026-08-22 18:01:16,586 INFO         val   vel_rmse     by horizon (m/s): [1.3205 0.6125 0.4915 0.4662 0.5685], mean: 0.6918 m/s
2026-08-22 18:01:16,586 INFO         train vel_dist     by horizon (m/s): [1.5396 0.6562 0.5487 0.5333 0.6112], mean: 0.7778 m/s
2026-08-22 18:01:16,586 INFO         val   vel_dist     by horizon (m/s): [1.5403 0.651  0.5454 0.5319 0.6039], mean: 0.7745 m/s
2026-08-22 18:01:16,586 INFO         train heading_rmse by horizon: [0.0506 0.0407 0.0271 0.0068 0.0059], mean: 0.0262
2026-08-22 18:01:16,586 INFO         val   heading_rmse by horizon: [0.0501 0.0413 0.0267 0.0072 0.0058], mean: 0.0262
2026-08-22 18:01:16,586 INFO         train heading_dist by horizon: [0.0361 0.0183 0.0069 0.0043 0.0043], mean: 0.0140
2026-08-22 18:01:16,586 INFO         val   heading_dist by horizon: [0.0355 0.0181 0.0069 0.0042 0.0041], mean: 0.0138
2026-08-22 18:01:16,586 INFO         train stamina_rmse by horizon: [0.0117 0.0097 0.0041 0.0075 0.0259], mean: 0.0118
2026-08-22 18:01:16,586 INFO         val   stamina_rmse by horizon: [0.0116 0.0097 0.004  0.0075 0.026 ], mean: 0.0117
2026-08-22 18:01:16,587 INFO     train pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 18:01:16,587 INFO     val   pos       R2 by horizon: [0.999 0.999 0.996 0.997 0.998], mean: 0.998
2026-08-22 18:01:16,587 INFO     train pos       %-of-persistence by horizon: [234.9  52.3  27.6  14.1   6.5]
2026-08-22 18:01:16,588 INFO     val   pos       %-of-persistence by horizon: [234.1  52.4  27.5  14.1   6.5]
2026-08-22 18:01:16,588 INFO     train vel       R2 by horizon: [0.574 0.929 0.965 0.968 0.95 ], mean: 0.877
2026-08-22 18:01:16,588 INFO     val   vel       R2 by horizon: [0.574 0.931 0.965 0.968 0.951], mean: 0.878
2026-08-22 18:01:16,589 INFO     train vel       %-of-persistence by horizon: [157.   22.5  14.4  13.7  17. ]
2026-08-22 18:01:16,589 INFO     val   vel       %-of-persistence by horizon: [157.1  22.3  14.4  13.6  16.8]
2026-08-22 18:01:16,589 INFO     train heading   R2 by horizon: [0.995 0.997 0.998 1.    1.   ], mean: 0.998
2026-08-22 18:01:16,589 INFO     val   heading   R2 by horizon: [0.995 0.997 0.999 1.    1.   ], mean: 0.998
2026-08-22 18:01:16,589 INFO     train heading   %-of-persistence by horizon: [10.4  4.4  2.8  0.7  0.6]
2026-08-22 18:01:16,589 INFO     val   heading   %-of-persistence by horizon: [10.3  4.4  2.7  0.8  0.6]
2026-08-22 18:01:16,589 INFO     train stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 18:01:16,589 INFO     val   stamina   R2 by horizon: [0.998 0.999 1.    0.999 0.992], mean: 0.998
2026-08-22 18:01:16,589 INFO     train stamina   %-of-persistence by horizon: [1320.8  219.9   31.    34.1   60. ]
2026-08-22 18:01:16,589 INFO     val   stamina   %-of-persistence by horizon: [1311.5  218.6   30.    34.3   60.1]
2026-08-22 18:02:11,803 INFO Loaded 610 shard(s) from physics_pretrain_data/player
2026-08-22 18:02:11,862 INFO Dataset: 610,000 episodes (518,500 train / 91,500 val)
2026-08-22 18:02:11,980 INFO pos_weight (max cap: 1.0):
2026-08-22 18:02:11,981 INFO     t= 0.2s  out_of_bounds=1.00  goal_scored=1.00
2026-08-22 18:02:11,981 INFO     t= 1.0s  out_of_bounds=1.00  goal_scored=1.00
2026-08-22 18:02:11,981 INFO     t= 3.0s  out_of_bounds=1.00  goal_scored=1.00
2026-08-22 18:02:11,981 INFO     t= 5.0s  out_of_bounds=1.00  goal_scored=1.00
2026-08-22 18:02:11,981 INFO     t=10.0s  out_of_bounds=1.00  goal_scored=1.00
2026-08-22 18:02:14,734 INFO Widened checkpoint from checkpoints/physics_pretrain/player_encoder_10.midtrain_latest.pt to current config dims (hidden_dim: 72->96, encoder_bottleneck_dim: 42->64, latent_dim: 16->26); resumed (phase=midtrain_latest)
2026-08-22 18:02:16,029 INFO Training row-count summary (train split):
    main (per-horizon heads)        : 518,500 rows -- own batches
    autoencode/t0 (bottleneck recon): 2,592,500 rows -- own batches (518,500 rows x 5 horizons)
    adjacent-pair (dynamics)        : 2,077,175/2,592,500 horizon-pass rows mask-eligible -- shares the horizon pass's own latent, no extra rows/batches
    crossing_head (at each horizon) : 573,299/2,592,500 horizon-pass rows mask-eligible -- shares the horizon pass's own latent, no extra rows/batches
    crossing_head (at t=0, in main) : 179,133/518,500 main rows masked-valid (position term only; delta_t trains unmasked on the -1 sentinel) -- shares main's own latent
    goal_dist_delta_head (main only): 518,500 main rows, unmasked -- shares main's own latent
    short-horizon probes (main only): 518,500 main rows x 2 heads, unmasked -- shares main's own latent
2026-08-22 18:02:16,030 INFO Autoencode pretraining: 1 epoch(s), lr=3.00e-06, optimizer=adam
2026-08-22 18:02:25,315 INFO   autoencode pretrain epoch 1/1: train_loss=0.0012  val_loss=0.0008
2026-08-22 18:02:25,316 INFO         train pos_rmse     by horizon (m): [0.8928 0.654  0.4298 0.4086 0.4205], mean: 0.5611 m
2026-08-22 18:02:25,316 INFO         train pos_dist     by horizon (m): [0.9805 0.6461 0.4798 0.474  0.4777], mean: 0.6116 m
2026-08-22 18:02:25,316 INFO         train vel_rmse     by horizon (m/s): [1.2766 0.5819 0.5032 0.4829 0.4403], mean: 0.6570 m/s
2026-08-22 18:02:25,316 INFO         train vel_dist     by horizon (m/s): [1.4945 0.5866 0.5316 0.5175 0.4844], mean: 0.7229 m/s
2026-08-22 18:02:25,316 INFO         train heading_rmse by horizon: [0.0612 0.021  0.0094 0.0084 0.0086], mean: 0.0217
2026-08-22 18:02:25,316 INFO         train heading_dist by horizon: [0.0411 0.0106 0.0057 0.0056 0.0058], mean: 0.0137
2026-08-22 18:02:25,316 INFO         train stamina_rmse by horizon: [0.0079 0.0055 0.005  0.0051 0.0056], mean: 0.0058
2026-08-22 18:02:25,316 INFO         val   pos_rmse     by horizon (m): [0.8189 0.5687 0.3655 0.3433 0.3575], mean: 0.4908 m
2026-08-22 18:02:25,316 INFO         val   pos_dist     by horizon (m): [0.9111 0.5676 0.4178 0.409  0.4139], mean: 0.5439 m
2026-08-22 18:02:25,316 INFO         val   vel_rmse     by horizon (m/s): [1.2738 0.5637 0.4794 0.4536 0.4035], mean: 0.6348 m/s
2026-08-22 18:02:25,317 INFO         val   vel_dist     by horizon (m/s): [1.4903 0.5616 0.4993 0.4795 0.4418], mean: 0.6945 m/s
2026-08-22 18:02:25,317 INFO         val   heading_rmse by horizon: [0.0486 0.0182 0.0086 0.0076 0.0078], mean: 0.0181
2026-08-22 18:02:25,317 INFO         val   heading_dist by horizon: [0.0326 0.0096 0.0052 0.0051 0.0054], mean: 0.0116
2026-08-22 18:02:25,317 INFO         val   stamina_rmse by horizon: [0.0056 0.0034 0.003  0.003  0.0036], mean: 0.0037
2026-08-22 18:02:25,320 INFO Autoencode pretraining: restored best-val weights (val_loss=0.0008)
2026-08-22 18:02:25,324 INFO Saved 'after_autoencode' checkpoint to checkpoints/physics_pretrain/player_encoder_11.after_autoencode.pt
2026-08-22 18:02:25,324 INFO Decoder-only pretraining: 55 epoch(s), lr=4.00e-05, optimizer=adam, freeze_latent=False
2026-08-22 18:03:05,346 INFO   decoder-only pretrain epoch 1/55: train_loss=0.0143  val_loss=0.0126
2026-08-22 18:03:05,346 INFO     crossing_head: train loss=0.0066 pos_dist=18.952m dt_mae=2.262s | val loss=0.0042 pos_dist=12.359m dt_mae=1.975s
2026-08-22 18:03:05,346 INFO     goal_dist_delta_head: train loss=0.00959 mae=(left 4.295m, right 5.241m) | val loss=0.00654 mae=(left 3.791m, right 4.017m)
2026-08-22 18:03:05,346 INFO     short_horizon_probes: train loss=0.08134 rmse_norm=(0.2s 0.2072, 1.0s 0.1909) | val loss=0.04255 rmse_norm=(0.2s 0.1550, 1.0s 0.1361)
2026-08-22 18:03:40,186 INFO   decoder-only pretrain epoch 2/55: train_loss=0.0122  val_loss=0.0121
2026-08-22 18:03:40,186 INFO     crossing_head: train loss=0.0051 pos_dist=12.094m dt_mae=2.209s | val loss=0.0039 pos_dist=10.577m dt_mae=1.895s
2026-08-22 18:03:40,187 INFO     goal_dist_delta_head: train loss=0.00527 mae=(left 3.455m, right 3.320m) | val loss=0.00451 mae=(left 3.198m, right 2.918m)
2026-08-22 18:03:40,187 INFO     short_horizon_probes: train loss=0.02448 rmse_norm=(0.2s 0.1178, 1.0s 0.0995) | val loss=0.01337 rmse_norm=(0.2s 0.0899, 1.0s 0.0726)
2026-08-22 18:04:17,405 INFO   decoder-only pretrain epoch 3/55: train_loss=0.0119  val_loss=0.0119
2026-08-22 18:04:17,406 INFO     crossing_head: train loss=0.0048 pos_dist=10.915m dt_mae=2.153s | val loss=0.0037 pos_dist=10.081m dt_mae=1.805s
2026-08-22 18:04:17,406 INFO     goal_dist_delta_head: train loss=0.00429 mae=(left 3.047m, right 2.811m) | val loss=0.00408 mae=(left 2.916m, right 2.741m)
2026-08-22 18:04:17,406 INFO     short_horizon_probes: train loss=0.00920 rmse_norm=(0.2s 0.0748, 1.0s 0.0591) | val loss=0.00626 rmse_norm=(0.2s 0.0625, 1.0s 0.0485)
2026-08-22 18:04:52,209 INFO   decoder-only pretrain epoch 4/55: train_loss=0.0118  val_loss=0.0118
2026-08-22 18:04:52,209 INFO     crossing_head: train loss=0.0047 pos_dist=10.512m dt_mae=2.105s | val loss=0.0037 pos_dist=9.836m dt_mae=1.770s
2026-08-22 18:04:52,209 INFO     goal_dist_delta_head: train loss=0.00398 mae=(left 2.826m, right 2.680m) | val loss=0.00384 mae=(left 2.735m, right 2.635m)
2026-08-22 18:04:52,209 INFO     short_horizon_probes: train loss=0.00469 rmse_norm=(0.2s 0.0539, 1.0s 0.0418) | val loss=0.00345 rmse_norm=(0.2s 0.0462, 1.0s 0.0362)
2026-08-22 18:05:26,888 INFO   decoder-only pretrain epoch 5/55: train_loss=0.0117  val_loss=0.0116
2026-08-22 18:05:26,888 INFO     crossing_head: train loss=0.0046 pos_dist=10.285m dt_mae=2.084s | val loss=0.0036 pos_dist=9.670m dt_mae=1.781s
2026-08-22 18:05:26,888 INFO     goal_dist_delta_head: train loss=0.00380 mae=(left 2.683m, right 2.593m) | val loss=0.00371 mae=(left 2.621m, right 2.570m)
2026-08-22 18:05:26,889 INFO     short_horizon_probes: train loss=0.00274 rmse_norm=(0.2s 0.0407, 1.0s 0.0327) | val loss=0.00214 rmse_norm=(0.2s 0.0357, 1.0s 0.0295)
2026-08-22 18:06:01,265 INFO   decoder-only pretrain epoch 6/55: train_loss=0.0116  val_loss=0.0117
2026-08-22 18:06:01,266 INFO     crossing_head: train loss=0.0045 pos_dist=10.155m dt_mae=2.073s | val loss=0.0036 pos_dist=9.541m dt_mae=1.794s
2026-08-22 18:06:01,266 INFO     goal_dist_delta_head: train loss=0.00370 mae=(left 2.593m, right 2.541m) | val loss=0.00363 mae=(left 2.551m, right 2.523m)
2026-08-22 18:06:01,266 INFO     short_horizon_probes: train loss=0.00178 rmse_norm=(0.2s 0.0321, 1.0s 0.0273) | val loss=0.00148 rmse_norm=(0.2s 0.0289, 1.0s 0.0254)
2026-08-22 18:06:37,257 INFO   decoder-only pretrain epoch 7/55: train_loss=0.0116  val_loss=0.0115
2026-08-22 18:06:37,257 INFO     crossing_head: train loss=0.0045 pos_dist=10.074m dt_mae=2.063s | val loss=0.0036 pos_dist=9.455m dt_mae=1.793s
2026-08-22 18:06:37,257 INFO     goal_dist_delta_head: train loss=0.00364 mae=(left 2.535m, right 2.507m) | val loss=0.00358 mae=(left 2.505m, right 2.502m)
2026-08-22 18:06:37,257 INFO     short_horizon_probes: train loss=0.00129 rmse_norm=(0.2s 0.0267, 1.0s 0.0239) | val loss=0.00111 rmse_norm=(0.2s 0.0246, 1.0s 0.0225)
2026-08-22 18:07:11,811 INFO   decoder-only pretrain epoch 8/55: train_loss=0.0115  val_loss=0.0115
2026-08-22 18:07:11,812 INFO     crossing_head: train loss=0.0044 pos_dist=9.918m dt_mae=2.051s | val loss=0.0036 pos_dist=9.353m dt_mae=1.791s
2026-08-22 18:07:11,812 INFO     goal_dist_delta_head: train loss=0.00360 mae=(left 2.496m, right 2.486m) | val loss=0.00355 mae=(left 2.470m, right 2.479m)
2026-08-22 18:07:11,812 INFO     short_horizon_probes: train loss=0.00100 rmse_norm=(0.2s 0.0232, 1.0s 0.0215) | val loss=0.00089 rmse_norm=(0.2s 0.0218, 1.0s 0.0204)
