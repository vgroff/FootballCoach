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
