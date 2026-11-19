# 2026-11-19: harder phase-2 start, keeper tweaks (run31)

User: the attacker has an easy time -- start further back, random starting
speed/direction for all players (as phase 1), and see whether the rules
keeper can save more (close the angle, close down).

## Measurements (scratchpad keeper_sweep*.py; run30/checkpoint53 as the AI
attacker; defender + keeper rules; 115 phase-2 eval seeds; the rules attacker
is deterministic per seed, so its rows are 115 independent episodes)

Scenario, 6 m keeper (goals scored, rules / AI attacker): current 15.7 /
87.4%; start 25-45 m 5.2 / 75.9%; random velocity 10.4 / 61.7%; both 6.1 /
57.4% (defender's ball wins vs the AI 10.2 -> 38.5%).

Keeper angle positioning (new, StagedGoalkeeperAI._guard_target): stand on
the goal-centre -> ball line, fraction f of the ball distance off the line
(cap m). Current scenario, rules attacker conceded: off 15.7%, f0.05/2 m
13.0%, f0.10/3 m 27.0%, f0.15/4 m 29.6%, f0.20/6 m 30.4% (saves 31 -> 18%);
AI attacker 86.5-87.3% for all. Worse -> left OFF (code kept, default off).

Close-down entry distance (angle off), conceded rules / AI attacker (AI shots
off target): 6 m 15.7 / 86.7 (1.0%); 8 m 15.7 / 79.6 (7.9%); 10 m 15.7 /
45.3 (39.6%); 12 m 16.5 / 23.4 (58.8%); 14 m 21.7 / 15.7 (59.7%). The AI
attacker (trained only vs the 6 m keeper) shoots wide against a keeper
rushing out; expected to recover with training (untested).

## User's calls -> run31
- Keeper close-down 10 m (exit 12 m), global (orders.json).
- Phase 2: attacker 18-40 m from goal, random_initial_velocity on.
- Run30's mix kept (50:50 phases; attacker 20:80, defender 30:70 rules:neural).
Run31 from run30/checkpoint53, checkpoints/phase2_run31, log
scratchpad/ppo_run31.log; watchdog moved. Phase-2 evals not comparable with
earlier runs (new start + keeper).

## Run31 first hour (13 evals)

Rules baselines (harder start + 10 m keeper): rules attacker 7.0% (keeper) /
54.8% (no keeper); rules defender concedes 7.0% to the rules attacker.

The sweep's 10 m-keeper collapse was real but short-lived: run31's FIRST
rollout (run30/53 weights, before any update) had the AI attacker at 32.3%
goals with 31.5% shots off target vs the rules defence (sweep: 26.3% / 32.0%),
and after ONE 10-epoch update the eval was 52.0% with 4.1% off target. My
"will take time to adapt" guess was wrong -- one update.

| | evals 1-4 | evals 10-13 | rules |
|---|---|---|---|
| phase 1 vs rules | 63-65% | 64% | -- |
| attacker vs rules (keeper) | 52-55% | 55-58% | 7.0% |
| attacker vs rules (no keeper) | 55-58% | 57-62% | 54.8% |
| defender_vs_rules conceded | 8-10% | 9-11% | 7.0% |

Health: log current, CPU rising, no errors.

## 12:14 run31 crashed (my fault) -> run32

I ran a 24-worker fixed-seed NvN sweep next to the live run; run31 died with
`OSError: [WinError 1450] Insufficient system resources` (Windows resource
limit, not RAM: 25.7 GB free). Restarted as run32 from run31/checkpoint16
(saved complete at 12:11:50, before the crash), config unchanged; watchdog
moved. Memory note updated: no multi-process probes next to a live run.

Fixed-seed NvN on run31/checkpoint16 (AI attacker of the same checkpoint;
115 eval seeds x 8 = 920 eps; run31 scenario, 10 m keeper):

| | AI defender | rules defender |
|---|---|---|
| keeper in: AI attacker scores | 60.0% (ball won 26.4%) | 57.8% (ball won 35.7%) |
| no keeper: AI attacker scores | 67.8% (ball won 19.9%) | 60.4% (ball won 30.0%) |

Under the new start the AI defender is now WORSE than the rules one against
the AI attacker (it was better before: ~77% vs ~83% under the old start).

### 12:53 run32 check-in (9 evals) + defender test

| | run32 evals | run31 rules baselines |
|---|---|---|
| phase 1 vs rules | 64-65% | -- |
| attacker vs rules (keeper) | 57-61% | 7.0% |
| attacker vs rules (no keeper) | 59-67% | 54.8% |
| defender_vs_rules conceded | 8.5-10.4% (slight drift down) | 7.0% |

Fixed-attacker defender test (single process; attacker = run31/checkpoint1,
460 eps each, run31 scenario + 10 m keeper): defender run30/53 conceded 53.7%
(ball won 23.9%), run31/16 53.0% (27.6%), run32/4 52.6% (28.3%). Goals
conceded flat over ~20 updates; ball wins +4.4 pts (~1.6 SE, hint only;
keeper saves fell 16.3 -> 13.3%, i.e. the extra ball wins replace keeper
saves). Re-test at ~30-40 updates. Free RAM only 4.4-5.6 GB during
training now -- keep probes single-process and light.
Health: log current, CPU rising, no errors.

## 13:2x Reward audit -> PPG value refit (run33)

Run32 stopped at checkpoint16 (attacker ~55-58% / no keeper ~63%, defender
7-10% vs rules). New phase-2 terminal rewards (user's calls, goal stays +-4,
no shaping; ai_config.json reward.phase2 _comment_terminal_20261119):

| outcome | attacker | defender | keeper |
|---|---|---|---|
| goal | +4 | -4 | -4 |
| saved | -1 (was +1) | +2 (was +1) | +3 (was +2) |
| defence_possession | -2 | +3 | +1 |
| shot_off_target | -2 | +2 | +2 |
| attacker_out | -3 (was -4) | +2 | +1 (was +2) |
| timeout | -3 (was -1) | +1 | +1 |

PPG value refit before PPO (user's idea, avoids noisy first checkpoints):
`--ppg-refit-only --ppg-num-rollouts 6` from run32/checkpoint16, SAME mix as
PPO (`--phase-mix "1:0.5,2:0.5"`, confirmed 288/288 env slots; phase-2
controllers / tiers / start / keeper from the same config). MC returns, KL
anchor 600, 770k steps x 6. Output checkpoints/phase2_run33/checkpoint_pretrained.pt,
log scratchpad/ppg_refit_run33.log. PPO run34 from it next.

Refit done in 22 min (6 cycles x ~3.7 min). Value fit vs the original value
net on the same fresh rows: +0.0% (cycle 1 baseline), then +2.3 / +1.2 /
+0.9 / +1.2 / +1.2% better; overall bias -0.035 -> +0.007; KL to the old
policy <= 0.007. Per-outcome biases (e.g. saved|defender -2.5) are hindsight
grouping, not fit errors. The new rewards touch rare outcomes (saved ~5%,
timeout / out ~0-2%), so the old value net was already nearly right. R2
0.72 -> 0.63 reflects each cycle replaying the hardest 25% of seeds.
PPO run34 started from checkpoints/phase2_run33/checkpoint_pretrained.pt
(same 50:50 mix), log scratchpad/ppo_run34.log.

### 14:48 run34 check-in (16 evals, new rewards)

| | evals 1-4 | evals 13-16 | run31 rules baseline |
|---|---|---|---|
| phase 1 vs rules | 63-65% | 64-65% | -- |
| attacker vs rules (keeper) | 57-60% | 59-60% | 7.0% |
| attacker vs rules (no keeper) | 62-65% | 61-65% | 54.8% |
| defender_vs_rules conceded | 8.0-8.7% | 7.2-8.0% | 7.0% |
| attacker shots saved (keeper in) | 3.5-4.3% | 2.6-3.5% | -- |

Flat overall; defender_vs_rules drifting toward the rules defender (7.2% last
two); saves slightly down (consistent with saved -1, weak signal). Health:
log current, CPU rising, no errors.

### 15:48 run34 check-in (32 checkpoints)

| | last 8 evals | run31 rules baseline |
|---|---|---|
| phase 1 vs rules | 64-65% | -- |
| attacker vs rules (keeper) | 60-64% (up from 57-60%) | 7.0% |
| attacker vs rules (no keeper) | 59-65% | 54.8% |
| defender_vs_rules conceded | 7.6-9.3% | 7.0% |

Fixed-attacker defender test on run34's latest started (single process).
Health: log current, CPU rising, no errors.

Fixed-attacker defender test (attacker run31/checkpoint1, 460 eps each, run31
scenario + 10 m keeper): run30/53 53.7% conceded (ball won 23.9%) -> run31/16
53.0% (27.6%) -> run32/4 52.6% (28.3%) -> **run34/32 47.0% (31.7%)**. -6.7 pts
vs untrained (~2 SE, likely real); improvement came after ~20 updates, during
the new-reward stretch (cannot separate rewards from more training).

### 16:48 run34 check-in

| | last 8 evals | run31 rules baseline |
|---|---|---|
| phase 1 vs rules | 63-65% | -- |
| attacker vs rules (keeper) | 59-64% (best 64.3%) | 7.0% |
| attacker vs rules (no keeper) | 58-65% | 54.8% |
| defender_vs_rules conceded | 8.0-9.3% | 7.0% |
| defender_vs_rules ball won | 43-47% (was 41-43% earlier in run34) | 48.7% |

Health: log current, CPU 45k -> 67k s, no errors.

### 17:48 run34 check-in

| | last 8 evals | run31 rules baseline |
|---|---|---|
| phase 1 vs rules | 64-65% | -- |
| attacker vs rules (keeper) | 62-64% | 7.0% |
| attacker vs rules (no keeper) | 61-65% | 54.8% |
| defender_vs_rules conceded | 7.6-8.5% | 7.0% |
| defender_vs_rules ball won | 43-46% | 48.7% |

Plateau-ish: all within noise of the last hour. Health: log current, CPU
67k -> 88k s, no errors.

### 18:48 run34 check-in

| | last 8 evals | run31 rules baseline |
|---|---|---|
| phase 1 vs rules | 63-65% | -- |
| attacker vs rules (keeper) | 62-65% (best 64.8%) | 7.0% |
| attacker vs rules (no keeper) | 62-65% | 54.8% |
| defender_vs_rules conceded | 6.5-8.3% (two evals at 6.5%, below rules) | 7.0% |
| defender_vs_rules ball won | 43-47% | 48.7% |

The AI defender now matches the rules defender on goals conceded vs the rules
attacker (it trained mostly vs the AI attacker). Health: log current, CPU
88k -> 110k s, no errors.

## 19:0x Run34 stopped (user's call, "seems like it's working") -- final checkpoint82

Confirmatory fixed-attacker test, machine free so parallel (scratchpad
defender_fixed_attacker_mp.py; attacker run31/checkpoint1; 115 eval seeds x 8
= 920 eps each): untrained defender (run30/53) 54.5% conceded, ball won 24.7%
-> run34/82 **44.9%**, ball won **36.0%**. -9.6 pts, ~4 SE: the defender IS
improving. Final evals: attacker vs rules 61-65% (keeper) / 62-65% (no
keeper), defender_vs_rules conceded 6.5-7.8% (rules defender 7.0%), phase 1
63-65%. Next (user): neural goalkeeper + keeper physics changes.

## Keeper mechanics (user's proposal), implemented behind physics.json switches, all OFF

- movement.goalkeeper_dive_gating (+ goalkeeper_dive_duration_s 1.5 /
  goalkeeper_dive_recovery_s 1.5): sprint IS the dive; the keeper speed x1.23
  / accel x3.3 apply only while diving; after a dive (time up or stopped) a
  recovery countdown during which a sprint request is forced to a jog (the
  countdown keeps running). Match._gate_goalkeeper_dive, Player.gk_dive_*.
- tackling.goalkeeper_tackle_cooldown_multiplier: keeper TACKLER's inactive
  time x this (apply_tackle_result).
- ball_pickup.goalkeeper_pickup_radius_multiplier: keeper in own box
  (Match._pickup_params_for).
Tests: tests/unit/test_goalkeeper_dive_and_hands.py; full suite passes with
all off (2781).

Sweep (scratchpad keeper_phys_sweep.py; run31 scenario, 10 m rules keeper,
rules defender; AI attacker = run34/82; 920 eps, rules rows = 115 unique):

| keeper | rules attacker scores | AI attacker scores | keeper saves vs AI |
|---|---|---|---|
| off | 7.0% | 62.8% | 2.3% |
| dive gating | 9.6% | 61.8% | 2.8% |
| tackle cooldown x2 | 7.0% | 62.8% | 2.3% (identical) |
| pickup radius x1.5 | 2.6% | 60.4% | 4.7% |
| all three | 5.2% | 60.1% | 4.6% |

Activity check (gk_activity.py, 115 eps each, gating on): dives in 37 (rules
attacker) / 73 (AI) episodes, forced recovery jogs in 26 / 31, keeper tackles
in 0 / 0 -- the tackle nerf cannot matter in phase 2. The keeper saves only
2-5% of the AI attacker's attempts: vs the AI attacker he is not the
bottleneck (the defender's ball wins are). Pickup x1.5 is a buff, the
largest effect; all three together = slightly stronger keeper.

## Neural goalkeeper: steps 1-4 (user's plan, 2026-11-19 evening)

1. Dive timers as player inputs (e448f63): PlayerFeatures.gk_dive_remaining /
   gk_dive_recovery appended (39 -> 41); older checkpoints get zero columns on
   load (checkpoint_io.pad_player_feature_columns). Verified: run34/82 gives
   identical outcomes on 40 seeded episodes, old code vs new.
2. distill_pretrain role_bc (400139f): a role taught by its rules AI (BC,
   label-smoothed) instead of a teacher network.
3. Keeper evals (0380f51): goalkeeper_vs_rules (vs rules_baseline 7.0%),
   attacker_vs_ai_keeper (vs attacker_vs_rules ~62%); specs now run only when
   all their neural roles train.
4. Keeper pretrain started (config c3c08d8): student run34/82, frozen self as
   teacher for both phases, keeper rows taught by the rules keeper,
   bc_label_smoothing 0.01 (targets 0.995/0.005), counterfactual off,
   goalkeeper neural 50%, phase mix 50:50, 6 cycles x 370k x 12 epochs, lr
   3e-5. Log scratchpad/keeper_pretrain_run35.log. Next (user): PPG value
   refit, then PPO -- not started unattended.

### Keeper pretrain: two crashes fixed, boost + shaping added, restarted as run37

- run35 crash: RoleScenarioEnv's end-of-episode obs was encoded on the NEXT
  trial's match (unseeded episodes draw the keeper per trial) -> KeyError with
  the keeper as trainee. Fixed: encode on the episode's own final tick
  (8ef49be).
- run36 crash: _bootstrap_last_values re-encoded every secondary track in the
  buffer against the current episode; a keeper track from an earlier episode
  -> KeyError. Fixed: only current neural secondaries (a26ffa5). New smoke
  tests (neural keeper + no-keeper trials through the real worker pool, for
  the pretrain and a PPO cycle) reproduce both KeyErrors on the old code; the
  session fixture that strips no_keeper_prob is why the existing ones missed it.
- User's calls (f8fd71e): keeper dive boost accel 3.3 -> 3.5, speed 1.23 ->
  1.4 (only during a gated dive); keeper save-target shaping 0.1/m while the
  ball is loose and heading for his goal (pure potential; target = the rules
  keeper's SaveOrder target; never while held).
- Restarted: checkpoints/phase2_run37, log scratchpad/keeper_pretrain_run37.log.
  run35/run36 dirs (crashed attempts) left in place. Next: PPG refit, then PPO.

### Keeper pretrain run39 (role BC 0.005, 20 cycles, keeper start +-2 m / +-0.5 m)

Cycles 1-3 after: frozen-teacher KL phase 1 0.004 / 0.011 / 0.012, phase 2
0.006 / 0.018 / 0.022 (user's stop threshold 0.35); keeper BC (x0.005)
0.476 / 0.341 / 0.226. Per-cycle checkpoints copied to
checkpoints/phase2_run39/checkpoint_cycleN.pt (pretrain overwrites one file).

CORRECTION: I called ~28 of the unweighted keeper BC (decision heads) an
unlearnable floor (~0.14 logged). Wrong: bc.bc_trains_frozen_heads is true,
so the frozen heads' BC backpropagates through them into the shared trunk;
measured on 156 keeper rows the decision part fell 28.8 -> 13.6 by cycle 3
(total 130 -> 56). Those heads never act in phases 1-2, so that trunk push is
interference and a plausible part of the slow KL drift (share not measured).
Fix available (detach frozen heads in role BC); user chose to let run39
finish as is.
