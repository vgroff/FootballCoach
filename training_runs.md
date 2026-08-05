2026-08-06 00:44:54,725 INFO Checkpoint dir: checkpoints/phase1_run6
2026-08-06 00:44:54,741 INFO Starting training: phase=phase1_get_possession, total_steps=200,000
2026-08-06 00:44:54,741 INFO Phase description: 1v1 scenario: learn to get possession and bring the ball toward the opponent box.  Decision network frozen except Move/GetPossession and latent vector.
2026-08-06 00:44:54,788 INFO _from_ckpt: overriding bc_pretrain_epochs=10 → 3
2026-08-06 00:44:54,788 INFO _from_ckpt: overriding demo_value_pretrain_epochs=10 → 0
2026-08-06 00:44:54,788 INFO _from_ckpt: overriding value_pretrain_epochs=45 → 35
2026-08-06 00:44:54,788 INFO _from_ckpt: overriding aux_coeff_start=0.0 → 0.0
2026-08-06 00:44:54,788 INFO _from_ckpt: overriding aux_coeff_end=0.0 → 0.0
2026-08-06 00:44:54,788 INFO _from_ckpt: overriding aux_coeff_anneal_fraction=0.3 → 0.35
2026-08-06 00:44:56,047 INFO --latest(-pretrain): resolved to checkpoints/phase1_run5/latest.pt
2026-08-06 00:44:56,065 INFO Loaded checkpoint: checkpoints/phase1_run5/latest.pt (step 30000)
2026-08-06 00:44:56,065 INFO Loaded checkpoint for re-pretraining: checkpoints/phase1_run5/latest.pt — will still run BC/value pre-training
2026-08-06 00:44:56,226 INFO Loading 1000 demonstration file(s) from demonstrations/phase1
2026-08-06 00:45:00,199 INFO Dataset: 385,963 steps loaded
2026-08-06 00:45:00,200 INFO Offline BC dataset: 385,963 steps from demonstrations/phase1
2026-08-06 00:45:00,200 INFO Reward diagnostic: running 40 rules-vs-rules episodes...
