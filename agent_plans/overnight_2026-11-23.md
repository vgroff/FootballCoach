# Overnight 2026-11-23: PPO run82 (from the archived distill refit)

Start: checkpoints/archive/phase12_refit_run81_cycle12.pt (see the 2026-11-22
log in agent_plans/overnight_2026-11-20.md and the .md next to the archive).
Settings: phase mix 35/65; n_epochs 6, lr 3e-5, target_kl 2.0, 500k;
ent_kick_power_only_weight 4; direction reg 0.1; no distill anchor.

- 01:45, checkpoints 1-9 (~4.5M steps): stable, no errors.
  - Phase 1 vs rules, mean reward / win: 1.185 / 64, 1.149, 1.141, 1.139,
    1.195, 1.177, 1.163, 1.175 / 63% (flat, no run66-style slide).
  - Attacker vs rules: 76.1% goals -> 78-80% as kick kappa sharpens
    300 -> 310 (run66 79.3%).
  - Keeper vs rules: 13.5-14.3 goals / ~38 saves; defender 6.5-8.5 goals;
    attacker vs ai keeper 58-61 / 23-26 saves; all-neural ~52 / 24 / 19.
  - Power sigma 0.200 (no drift under ent weight 4), move kappa 264.
  - KL 0.011-0.014 per update.
  - Direction group 100% grad-clipped every update (pre-clip mean 35-40 vs
    limit 4; run66 was ~31 at norm ~1).
  - Ratio spikes 37-311 on single rows, both signs, bounded by clip 0.3 /
    dual-clip 10.
  - No action taken.
