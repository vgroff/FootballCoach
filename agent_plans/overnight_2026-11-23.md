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
- 02:35, checkpoints 10-20 (~10M steps): stable, no errors, no action.
  - Phase 1 vs rules mean reward 1.13-1.23 (best 1.234 / 65% at
    checkpoint 17); no slide.
  - Attacker vs rules 78-83% (best 82.8 at checkpoint 19).
  - Keeper vs rules: a slight drift, goals 13.5-14.3 over checkpoints 1-9
    -> 14.3-15.7 over checkpoints 10-20 (latest 15.2 / 36.3 saved). About +1
    pt, within the per-eval SE (~1.6) but persistent on the same seeds. It
    stopped growing; watch (attacker outpacing the keeper, as in earlier
    cross-play). Act if it reaches ~16.5 or rises 3 checkpoints in a row.
  - Defender 6.3-7.6; attacker vs ai keeper 58-63; all-neural ~50 / 25 / 19.
  - Spreads: kick kappa 300 -> 322 (slow, ~+1 per update), power sigma
    0.200 -> 0.199, move kappa 263-264, z sigma flat.
  - KL 0.010-0.013.
  - Direction group still 100% clipped (pre-clip mean 35-38).
  - Largest ratio spike 1489 (checkpoint 14, negative advantage, bounded
    by dual-clip; KL normal).
