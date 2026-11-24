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
- 03:40, checkpoints 21-30 (~15M steps): stable, no errors, no action.
  - Phase 1 vs rules 1.11-1.24 (new best 1.244 / 65% at checkpoint 30);
    the 1.113 at checkpoint 21 was a single low.
  - Attacker vs rules 78.5-82.0%.
  - Keeper vs rules 14.6-15.4 goals / 36-37 saves (the drift stopped and
    eased back).
  - Defender 6.3-7.4; attacker vs ai keeper 58-63; all-neural goals
    47.8-56.5 (self-play balance swinging, no trend).
  - Kick kappa 323 -> 334 (~+1.1 / update), power sigma 0.199, move kappa 264.
  - KL 0.010-0.014; direction group still 100% clipped (pre-clip mean 33-38).
  - Ratio spikes up to 571 (checkpoint 25, positive advantage).
- 04:35, checkpoints 31-48 (~24M steps): no errors. (Missed re-arming the
  watch for ~30 min, so this was reviewed in one go.)
  - KEEPER vs rules attacker, a real mild regression: goals 13.5-14.3
    (checkpoints 1-9) -> 14.6-15.2 (29-34) -> 15.4-15.7 (35-38) -> 15.9-16.5
    (39-47), back to 15.4 at 48; saves ~37-38 -> 35.0-35.9.
  - Same seeds, ~10 checkpoints shifted ~+2 pts (per-eval SE ~1.6) -> real.
  - Likely specialisation to self-play: only ~10% of neural-keeper episodes
    face the rules attacker; saves vs its own attacker are flat / up
    (attacker_vs_ai_keeper 22-26, self-play rollouts 23-27); attacker vs
    rules up 76 -> 80-82.
  - Decision: HOLD, user to decide in the morning (the remedy, more keeper
    episodes vs the rules attacker, changes the training distribution and
    needs a restart). Act alone only if the keeper passes ~17.5 conceded or
    saves drop below 34.
  - Phase 1 vs rules 5-checkpoint means: 1.169 (29-33) -> 1.139 (39-43)
    -> 1.129 (44-48); slight drift, above the 1.10 act threshold (new rule:
    5-checkpoint mean < 1.10 or a single reading < 1.05).
  - Kick kappa 334 -> 357 (~+1.2 / update); power sigma 0.200; move kappa
    263; KL 0.011-0.012.
- 05:05, checkpoints 49-54 (~27M steps): no errors. Two slow drifts, both
  inside the act-alone thresholds, so held for the user's morning call.
  - Phase 1 vs rules: 1.153 1.139 1.140 1.120 1.107 1.121 (5-checkpoint
    mean ~1.13 vs 1.17 early in the night; threshold mean < 1.10).
  - Keeper vs rules attacker: 15.4 15.4 16.1 16.3 15.9 15.9 goals / ~36
    saves (start 14.3 / 37.2; thresholds > 17.5 or saves < 34).
  - All-neural saves 25.7 -> 19.8-20.9 over the last 5. That RETRACTS the
    pure "keeper specialises to the self-play attacker" explanation: it is
    slipping vs its own attacker too, i.e. the attacker is outpacing it.
  - Attacker vs rules 80-82%.
  - Kick kappa 365, power sigma 0.200, KL 0.011-0.013.
  OPTIONS FOR THE USER (each a restart in a fresh dir from a recent
  checkpoint):
  - A) phase mix 35/65 -> 50/50 (fixed phase 1 in the refit);
  - B) more keeper episodes vs the rules attacker: curriculum.phase2
    controllers attacker rules:neural is 1:9 (user's 2026-11-20 call for
    more neural training) -> e.g. 1:4;
  - C) both.
  If a threshold trips before then: phase 1 -> A, keeper -> B.

## 2026-11-24 20:55: catch-up review (monitoring gap)

- MONITORING GAP: after the 05:05 (11-23) check nothing was processed; the
  hourly check-ins queued. The run trained on from checkpoint 54 to 296
  (~148M steps) until 03:25 on 11-24, then the laptop slept until 20:52
  (training paused). No errors in the whole log.
- Evals, checkpoint 1 -> 61 -> 296:
  - phase 1 vs rules 1.185 / 64% -> 1.100 / 62% -> 1.225 / 65% (held
    1.10-1.23 all run: NO run66-style slide under 35/65);
  - keeper vs rules 14.3 / 37.2 -> 16.3 / 35.7 -> 14.1 / 39.6 (12.2-14.1
    / 39-41.5 over checkpoints 200-296: the overnight drift REVERSED on its
    own, now better than the start and run66's 13.5 / 38.5);
  - attacker vs rules 76.1 -> 81 -> 84.3; defender ~7-8;
  - attacker vs ai keeper 59 -> 61-69 (goals up);
  - all-neural saves 22 -> 24-30.
- FLAG, kick kappa sharpening and accelerating: 300 (1), 372 (60), 438
  (120), 512 (180), 610 (240), 724 (296); +1.2 -> ~+2 per update, heading
  back toward run66's 1630.
  - Power sigma 0.200 -> 0.188 (-6%); move kappa flat 265.
  - KL 0.017 -> 0.009; direction clip still 100% (pre-clip mean 39 -> 32).
  - Not hurting play (attacker goals rose with it, consistent with the
    kappa sweep: precision is rewarded).
  - User's call: (a) ent_kick_dir_azimuth_only_weight 5 -> ~10, (b) a
    kick-kappa cap ~1000, (c) leave it.
