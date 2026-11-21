# Overnight 2026-11-20: getting the neural keeper to learn (option 1)

User's call (before bed): "Do option 1 until tomorrow morning ... just keep
trying to get the neural gk to learn". Option 1 = PPG value refit, then PPO,
starting from the keeper pretrain run40 (role-BC label-frame fix 6dcbec1).

Starting point (run40 final, see overnight_2026-11-19.md): keeper gates
learned from the rules keeper (exec_move / dive / tackle / move / get
possession agreement 61-95%), move direction not learned. Keeper evals
(115 x 4): vs rules attacker 44.1% conceded (rules keeper 7%), vs AI
attacker 25.2% (rules keeper 62%).

## Log

- ~01:5x: PPG value refit started: --ppg-refit-only --ppg-num-rollouts 6,
  --phase-mix "1:0.5,2:0.5", from checkpoints/phase2_run40/checkpoint_pretrained.pt
  (log scratchpad/ppg_refit_run41.log). Config unchanged: phase-2 controllers
  attacker rules:neural 1:4, defender 3:7, keeper 1:1; keeper save-target
  shaping 0.1; PPO n_epochs 10, rollout 500k, target_kl 0.5, replay 0.35
  with fresh engine dice; evals fully seeded.
- 02:34: PPG refit done -> checkpoints/phase2_run41/checkpoint_pretrained.pt
  (40 min, 6 cycles). Value MSE/var per cycle 0.40 -> 0.50 is the replayed
  hardest episodes (the ORIGINAL net's MSE on the same rows rose too:
  0.40 -> 0.51); the refit beats the original by a little each cycle
  (0.500 vs 0.509 on the last). Policy KL <= 0.007.
- 02:35: PPO started -> checkpoints/phase2_run42 (log scratchpad/ppo_run42.log),
  --checkpoint checkpoints/phase2_run41/checkpoint_pretrained.pt, config as above.
- 02:39: run42 stopped after ONE rollout ("Training complete"): launched
  without --total-steps (earlier overnight runs pass --total-steps
  300000000). Its checkpoint-1 evals = the refit's start point: keeper vs
  rules attacker 44.1% conceded (saved 2.4%), vs AI attacker 24.3%
  (saved 10.4%, shot off target 30.9%).
- 02:40: relaunched as run43 (fresh dir), same start (run41 refit), with
  --total-steps 300000000 (log scratchpad/ppo_run43.log).
- 03:09 watchdog: run43 healthy, 6 checkpoints (~5 min/round), no errors.
  Policy KL ~0.02/update, ratio clipped 4.9%, grad clip 0%.
  Keeper evals ckpt 1-5 (conceded / saved): vs rules attacker 44.1/2.6,
  44.3/3.3, 44.3/2.8, 44.3/3.0, 44.6/3.0 (rules keeper 2.6/47.8); vs AI
  attacker 23.7/11.7, 23.7/12.2, 24.1/12.6, 24.1/14.1, 24.8/15.7 (rules
  keeper 62). No regressions: attacker_vs_rules 60.7-62.4%,
  defender_vs_rules 6.3-7.4% conceded.
  Checked (single process): the keeper's save-target shaping reaches its
  SECONDARY rows (it is trainee only when attacker+defender are rules) --
  13 non-zero rows / 20 all-neural episodes, sum +0.02: present but tiny.
  Too early to call; no changes.
- 03:15 probe (single process, scratchpad/keeper_shot_state.py, run43
  ckpt 6, 60 eval seeds, rules attacker + defender): keeper state at the
  first tick a shot heads for goal.
  neural keeper: 27 goals / 1 save; median 2.78 m OFF his line, 5.3 m from
  the save target; diving 11%, in dive recovery 16%.
  rules keeper: 1 goal / 28 saves; 0.06 m off his line, 3.1 m from the
  save target; diving 30%, recovery 5%.
  -> the failure is POSITIONING (comes ~3 m out and gets beaten), not
  the dive gate; matches the unlearned move direction from the pretrain.
  Untested guess: generalizes "step at the attacker" from the defender
  (same team / canonical frame). Ckpt 6 saves vs AI attacker 17.6% (up
  from 11.7%). Watching before changing anything.
- 03:17 (user's calls, still awake): phase mix 50:50 -> 25:75 (phase 2 is
  more game-like / newer) and less rules-AI involvement: phase-2
  controllers attacker rules:neural 1:4 -> 1:9, defender 3:7 -> 1:4,
  keeper 1:1 -> 1:3 (some rules keeper kept so the attacker still meets
  real saves). Run43 stopped right after ckpt 8 (keeper ckpt 6 evals: vs
  rules attacker 44.6/2.6, vs AI attacker 23.7/17.6). Run44 (fresh dir)
  from checkpoints/phase2_run43/checkpoint8.pt, --phase-mix "1:0.25,2:0.75"
  --total-steps 300000000 (log scratchpad/ppo_run44.log). Guard: if
  phase-1 [eval vs rules] (64%) drops > ~3 points, go back to 50:50.
- 03:58 watchdog: run44 healthy, 8 checkpoints (~5 min/round), no errors;
  KL ~0.02/update, ratio clipped 4.8%, grad clip 0%, entropy -1.51.
  Run44 ckpts 1-8: phase-1 vs rules 62.3-64.2% (guard OK); attacker_vs_rules
  61.5-63.7%; defender_vs_rules 6.3-6.7%; keeper vs rules attacker 43.9-44.8%
  conceded, saved 2.4-3.3% (flat); keeper vs AI attacker 25.2-28.0%
  conceded, saved 18.5-23.3% (rising) with shot_off_target 22 -> 16%:
  mostly the attacker putting more on target vs the neural keeper
  (co-adaptation), so vs-rules-attacker is the cleaner keeper measure.
  No changes (slow BC->PPO is expected).

## Morning (13:50) -- the laptop slept from ~04:37 to ~13:45

Run44 paused with the machine (no crash, no errors) and resumed on wake;
it reached checkpoint 17 by 04:37, i.e. only ~80 min of PPO overnight.
Run44 ckpts 1-17 (conceded/saved %):
- keeper vs rules attacker: 44.3/2.8 ... 43.7/3.3 -- flat (rules keeper 2.6/47.8)
- keeper vs AI attacker: 25.9/19.1 -> 32.4/25.7 (attacker putting more on
  target vs the neural keeper; co-adaptation)
- attacker_vs_rules 63.0 -> 61.5 (saved 3.5 -> 6.5); defender_vs_rules 6.3 -> 7.8
- phase-1 vs rules 62.1-64.2% (guard OK)
Positioning probe on ckpt 17 (60 seeds, rules attacker): 26 goals / 2 saves;
median 3.08 m off his line at shot time (03:15: 2.78 m), 5.25 m from the
save target. Positioning has not improved.
- 15:23: run44 ckpt 37, healthy (KL ~0.018, ratio clip 4.9%, grad clip 0%).
  Keeper vs rules attacker 43.7 -> 43.3% conceded (ckpts 26-37), saved
  ~3%: still flat. Keeper vs AI attacker 34 -> 40% conceded, saved ~26%:
  the attacker is learning to beat the neural keeper faster than the
  keeper improves. attacker_vs_rules ~62%, defender_vs_rules ~7.5% (was
  6.3), phase-1 63-64%. Positioning probe ckpt 37: 25 goals / 3 saves,
  still 2.93 m off his line at shot time. PPO alone is not fixing it.

## Afternoon: keeper BC refit (user's call, ~15:35)

User: stop the run, refit with the BC first; "it should be able to learn
standing still from them". Run44 stopped right after checkpoint 40.
- Sides: phase 2 never swaps (attacker LEFT, defender + keeper RIGHT); the
  canonical frame (+x attack, y-mirror) makes the network side-agnostic.
- Standing labels checked (139 standing rules-keeper rows, run34/82 play):
  the direction label is the arrival point he already stands on -- median
  166 deg away from the ball, towards his own goal in 98%, while his heading
  faces the ball (label vs heading 164 deg). The network already faces the
  ball (13 deg). So those labels would teach him to turn his back; standing
  itself is taught by exec_move (label 0). They were 74% of the direction
  rows -> why run40's direction term never fell.
- New opt-in role-BC options (distill_pretrain): role_bc_direction_moving_only
  (bc_loss_from_tensor direction_moving_only) and role_bc_direction_loss_weight
  (3 -> 12 for role-BC rows only). Test added (test_role_bc_label_frame.py).
- Refit run45 from checkpoints/phase2_run44/checkpoint40.pt, teachers = that
  checkpoint frozen (both phases), role_bc 0.005, 12 cycles, phase mix 25:75
  (log scratchpad/keeper_refit_run45.log). Stop if KL > 0.35.
- 15:4x CORRECTION (user: "does the rules AI turn its back to play? players
  still turn when stationary"): yes they turn (step_player_towards faces a
  STANDSTILL player towards desired_direction), and the rules keeper does
  NOT turn his back. My 166-deg measurement was in the neural keeper's
  off-line states and compared against the NEURAL keeper's heading. With
  the rules keeper in control (60 seeds, 160 standing+direction decisions):
  label = his real desired_direction (median 14 deg), roughly along the goal
  line (87 deg from the ball, 55% towards own goal), he is on his line
  (0.09 m). Standing labels are genuine facing commands -> moving_only OFF
  again (kept as an opt-in switch, docstring corrected).
  run45 (moving_only on) stopped after 2 cycles; run46 started with the old
  setting by mistake (my config edit failed JSON validation and was not
  written) and was stopped within a minute (no checkpoint). Refit relaunched
  as run47: same as run45 but moving_only off, direction weight 12.
- 16:15: run47 (role BC 0.005, direction weight 12) cycles 1-7: KL 0.003-0.019;
  keeper BC unweighted 26.4 -> 16.8 (slow, ~0.4/cycle late). Mid-run probe
  cycle 3 vs start (run44/40, own play, 30 seeds): direction term 3.88 ->
  3.71, angle moving 128 -> 120 deg, standing 140 -> 129: improving slowly.
  User: "falling pretty slow, the KL is low, juice the coefficients" ->
  role_bc goalkeeper 0.005 -> 0.02; continued from run47 cycle 7 as run48
  (fresh dir, same frozen teacher run44/40), 10 cycles.
- 16:2x run48 (role BC 0.02) cycles 1-5: KL 0.003-0.024; keeper BC 16.9 ->
  8.6 unweighted. Breakdown (30 seeds, own play): gates fitted (exec_bce
  8.97 -> 2.5 by cycle 1), direction 12.76 -> 9.59 by cycle 2 (mean cos
  ~0.04 -> 0.2): the remaining loss is direction. In the keeper's own
  states the rules label is simply "towards the centre of my own goal"
  (median 0 deg; 88% / 96% within 30 deg, moving / standing; keeper ~2 m
  off his line) -- easy target, yet unlearned.
- Cause found: raw move-direction norm median ~28 (keeper rows), ~42 all
  roles (kick ~54). bc.direction_mag_reg_coef had been 0.0 since it was
  added (2026-09-02) and distill_pretrain never applied it; the policy
  ignores the norm (atan2 mean, separate kappa) but every angle loss's
  gradient on the raw vector shrinks ~1/||raw|| (and each Adam step rotates
  it ~1/||raw||). Probably slows ALL roles' direction learning in PPO too
  (untested).
- User: enable it in BC too, rescaling sensible. Done:
  src/footballcoach/ai/scripts/rescale_direction_heads.py rescaled run48
  cycle 5 -> checkpoints/phase2_run48/checkpoint_cycle5_dirscaled.pt (move
  / kick final layers / 41.6 / 54.3; verified on 1717 rows: normalized
  directions equal to 2.6e-5, 0 action / log-prob mismatches over 300 rows
  x deterministic + seeded); bc.direction_mag_reg_coef 0 -> 0.001
  (PPO, BC, DAgger, and now distill_pretrain); role-BC direction weight
  back to the default 3 (rescaled: ~10x run48's direction gradient).
  Refit continued as run49 (5 cycles). Only two unit-vector outputs exist
  (move_direction, kick_direction); the reg covers both.
- 16:4x run49 (rescaled heads, BC 0.02, direction weight 3, mag reg 0.001)
  cycles 1-2: KL 0.029/0.010 -> 0.034/0.041 (rising: rescaled direction
  heads move freely, other players' move_dir err vs teacher 0.2 -> 0.4 deg);
  keeper BC flat (4.76 -> 4.70). Cycle 1 breakdown: direction 1.48 -> 1.50
  at weight 3 (mean cos ~0.5) -- the rescale gave no visible speed-up in one
  cycle; most direction progress came in run48 cycles 3-5 (cos 0.2 -> 0.5)
  at BC 0.02. Keeper raw norm after rescale 0.67 (all-roles median used).
  Stopped after cycle 2 (plateau + rising KL).
  run49 cycle 2 probes: positioning FIXED-ish -- at shot time 0.62 m off
  his line (run44/40: 2.93; rules 0.06), 3.9 m from the save target (5.2;
  rules 3.1); direction vs rules on own states 50 deg moving / 51 standing
  (run44/40: 84 / 97). But 29 goals / 0 saves vs rules attacker (60 seeds):
  never diving at shot time (0%; rules 30%). Untested: dive-at-shot rows
  are too rare for BC; PPO should now be able to learn it since he is in
  position. Waiting for the user on PPG refit + PPO from run49 cycle 2.
- 16:5x shot-in-flight probe (run49 c2, rules attacker, 80 seeds): shots in
  flight are 34% of keeper decisions (281 rows). There the rules keeper
  moves 75% / dives 48%; the network P(move) 0.25, P(dive) 0.57 when it
  should vs 0.41 when not, direction cos -0.01, 28% of those rows already in
  dive recovery. Rest of play: direction cos 0.70. User: the physics
  pretrain already encodes the ball trajectory, so just refit harder.
  role_bc goalkeeper 0.02 -> 0.1; run50 from run49 cycle 2, 10 cycles; KL
  may reach 0.3 if it comes back down. Probe: scratchpad/shot_rows_probe.py.
- 17:5x run50 (BC 0.1) cycles 1-6: KL 0.015-0.039 (well under 0.3); keeper
  BC unweighted 4.59 -> 3.33. Shot-in-flight probe (rules attacker, 80
  seeds), run49 c2 -> run50 c3 -> c6: P(move) 0.25 -> 0.43 -> 0.48 (rules
  ~65-75%); direction cos -0.01 -> 0.10 -> 0.18; P(dive) should / shouldn't
  0.57/0.41 -> 0.51/0.35 -> 0.52/0.33; in dive recovery 28 -> 31 -> 33%
  (dives earlier, spent when the shot comes). Open-play direction cos
  steady ~0.72. Improving ~4x slower than open play did.
- 18:10 run50 done (10 cycles, KL 0.015-0.039; keeper BC 4.59 -> 3.06).
  Final shot-in-flight probe: P(move) 0.50 (agree 65%), dive 0.55 should /
  0.28 shouldn't (gap 0.27, up from 0.16), direction cos 0.14 (c6 0.18:
  noise-level, flat), in recovery 30%. Open play direction cos 0.76.
  User: KL tiny, juice harder; no dedicated test needed. role_bc 0.1 ->
  0.3, run51 from run50 final, 10 cycles.
- 18:50 run51 (BC 0.3) done: KL 0.015-0.046; keeper BC 2.84 -> 2.12.
  Shot-in-flight (run50 final -> run51 c3 -> c6 -> final): direction cos
  0.14 -> 0.33 -> 0.61 -> 0.53; P(move) 0.50 -> 0.53 -> 0.65 -> 0.65 (agree
  74%; rules moves 77%); dive should/shouldn't 0.55/0.28 -> 0.48/0.22;
  recovery 30 -> 26%. Open play direction cos 0.84.
  Outcomes vs rules attacker (60 seeds): run51 final sampled 28 goals / 1
  save; deterministic (mode) 23 / 5; rules keeper 1 / 28.
  Shot trace (run51 c6): neural 3.8 -> 2.3 m from the save target over the
  ~4-decision flight, 3.0 m/s; rules 3.1 -> 0.15 m, 3.9 m/s.
  Decision-interval test: rules keeper with its command held 6 ticks
  (0.24 s = the network's decision interval): 9 goals / 21 saves (vs 1/28
  per tick; still gets SaveOrder's <=0.15 m position snap, so slightly
  optimistic). -> the interval is a partial ceiling (~7 saves in 28), the
  neural keeper's shot-time policy (5 deterministic) is the main gap.
- 19:0x user: no shorter decision interval (the network should learn to
  lead the target -- via PPO, since BC copies the rules keeper's myopic
  per-tick commands); more BC first. role_bc 0.3 -> 0.6, run52 from run51
  final, 10 cycles; then PPG refit + PPO.
- 19:2x run52 (BC 0.6, lr 3e-5) cycles 1-5: KL 0.016-0.061; keeper BC
  2.05 -> 1.84. Cycle-3 shot probe flat vs run51 final (P(move) 0.65,
  dir cos 0.56, dive 0.48/0.20). User asked whether epochs / LR need
  tweaking: YES -- within each cycle the BC loss still falls linearly at
  epoch 12 (cycle 4: 1.182 -> 1.150) and the next cycle's fresh rollouts
  start at 1.130 (no generalisation gap): optimisation-limited. (Retracts
  my "close to what BC can extract".) Hidden rules-keeper state:
  _closing_down hysteresis (entry 10 m / exit 12 m) and
  _close_down_ended_s cooldown, plus RulesExpert's shadow order.
  distill lr 3e-5 -> 1e-4; run53 from run52 cycle 5, 10 cycles.
- 19:3x run53 (lr 1e-4, BC 0.6) cycles 1-3: KL 0.063/0.059 -> 0.075/0.055
  -> 0.087/0.069 (rising); keeper BC 1.87 -> 1.85 -> 1.83. Cycle 3 fits
  faster within the cycle (1.120 -> 1.087) but the next fresh rollout
  started above the previous end (1.143 vs 1.111; epoch-1 train KL 0.19):
  a generalisation gap at the higher LR. NB every relaunch restarts the
  1000-step LR warmup (~3 cycles at ~360 steps/cycle) -- user: halve it
  next time. Shot probe c3: P(move) 0.66 (agree 76%), dir cos 0.58, dive
  0.45/0.17, open play dir 0.90. Saves vs rules attacker (60 seeds): 1
  sampled / 6 deterministic (run51: 1 / 5) -- no change in outcome yet.
- Passing plan updated with the user's offside rules (pass only, never a
  player kicking to himself; ball/passer as the line if beyond the last
  defender) and terminal rewards (defender a bit more for offside than for
  a timeout).
- 19:4x KEY: the rules keeper's saves rely on SaveOrder's position SNAP
  (within max(0.15 m, top_speed*dt) of the save point it sets
  player.position = target and stops -- a teleport the network cannot do).
  60 seeds vs rules attacker: rules keeper as is 1 goal / 28 saves; no snap,
  per tick 10 / 19; no snap + command held 0.24 s (= the network's
  decision interval, i.e. a perfect label replay through the network's
  action path) 21 / 9. Neural keeper run53 c3: deterministic 22 / 6,
  sampled 28 / 1. So the fair ceiling is ~9 saves, not 28; the neural
  keeper (deterministic) is ~2/3 of the way; the rest is mostly sampling
  noise. The snap also inflates every eval against the rules keeper.
  Proposed (user's call): opt-in remove the snap from the rules keeper.
- Passing plan: offside as a general Match rule for every player and
  scenario, full law (user).
- 19:5x User: remove the snap ("always a bit of a backstop"); asked for
  rules-AI ideas. Added SaveOrder options (orders.json goalkeeper_ai
  save_snap / save_arrival_brake / save_settle_radius_m; defaults = old
  keeper, verified 1 goal / 28 saves unchanged; tests
  tests/test_unit/test_save_order_options.py). Sweep (60 seeds, rules
  attacker; per tick | held 0.24 s): no snap 10/19 | 21/9; + brake 4/25 |
  17/13; + settle 0.3 m 5/24 | 16/14; + settle 0.5 m 3/26 | 16/14; + both
  3/26 | 17/13. Proposal: save_snap false + save_settle_radius_m 0.5
  (within the 0.6 m keeper pickup radius) -- to be switched on after run53
  (changing the rules keeper mid-run would change its labels).
- 20:0x User's idea: brake for the save point + reach/2. Added
  save_arrival_brake_offset_m (+ = brake target short of the save point,
  - = beyond). Sweep (no snap, 60 seeds, per tick | held): brake 0.3 m
  short 6/23 | 17/13; 0.3 beyond 9/20 | 19/11 (overshoots); with settle 0.5
  every offset (-1.0..+0.5) 3/26 | 16/14 = settle alone. -> settle 0.5 alone
  is best; offset kept, default 0. "Lead the target": the save point is
  already a prediction of the crossing / intercept point, nearly static for
  a straight shot, so not implemented.
- 20:1x run53 (lr 1e-4) cycles 1-6: KL 0.055-0.087; keeper BC 1.87 -> 1.70.
  Cycle-6 shot probe: P(move) 0.68 (agree 77%), dir cos 0.65, dive
  0.44/0.18; open play dir cos 0.89. User: cut it and continue with the new
  keeper. orders.json goalkeeper_ai: save_snap false, save_settle_radius_m
  0.5. distill lr_warmup_steps 1000 -> 500 (user). lr stays 1e-4, epochs 12.
  Run54 from run53 cycle 6, role_bc 0.6, 10 cycles. NB the rules keeper
  changes everywhere it is used (phase-2 opponent, evals, labels).
- 20:2x 0.24 s keeper ideas (held rules keeper, 60 seeds, no-snap settle-0.5
  keeper; goals/saves): extra decision at shot start 17/13 (no help); hold
  the save point + per-tick steering 11/19 (+5) -- user rejected (the
  network must steer directly, no engine help). Interval-aware rules tweaks
  (per tick | held): settle 1.0 6/23 | 15/15; settle 1.5 15/14 | 17/13;
  momentum lead 0.12 / 0.24 s 3/26 | 16/14 and 3/26 | 15/15; settle 1.0 +
  lead 0.12 / 0.24 6/23 | 14/16 and 5/24 | 14/16 -- within noise, so kept
  settle 0.5; added opt-in SaveOrder lead_velocity_s
  (save_lead_velocity_s, default 0). The rest of the interval gap is for
  the network to learn (PPO).
- run54 cycle 1-2: KL 0.177/0.171 (new labels + shorter warmup) -> 0.076/
  0.057; keeper BC 1.73 -> 1.69 -> 1.78 (fresh) -> 1.70.
- 20:3x WHY the 0.24 s keeper concedes (user's question): it OVERSHOOTS
  (retracting "not overshooting"). On the 13 shots saved per tick but
  conceded when held: closest approach to the save point 0.08 vs 0.09 m
  (both reach it), at ball arrival 0.21 vs 1.67 m, speed 2.45 vs 4.48 m/s;
  the save point does not move (0.00 m). The settle check only runs at a
  decision tick, a sprinting keeper covers ~1.4 m per decision and needs
  ~1-1.5 m to stop. New opt-in SaveOrder settle_speed_s (settle radius =
  max(settle_radius_m, speed * k)). Sweep (60 seeds; held | per tick):
  k=0 16/14 | 3/26; 0.08-0.24 no change; 0.3 13/17 | 3/26; 0.4 13/17 |
  11/17; 0.5 14/16 | 16/12; 0.7 18/11 | 21/7. Held-out 55 seeds: k=0 held
  10/16, per tick 4/24; k=0.3 held 7/19, per tick 5/22 -> +3 saves at the
  network's rate on both sets. Proposed save_settle_speed_s 0.3 (labels
  are computed at decision ticks = the held setting).
- 20:4x run54 (no-snap keeper) cycles 1-3: KL 0.177/0.171 -> 0.076/0.057 ->
  0.083/0.072; keeper BC 1.69 / 1.70 / 1.70. User: enable the speed-scaled
  settle and restart from the latest -> orders.json save_settle_speed_s
  0.3; run55 from run54 cycle 3, same settings (BC 0.6, lr 1e-4, warmup
  500, 10 cycles).
- 20:2x run55 (rules keeper: no snap, settle max(0.5 m, speed*0.3 s)):
  cycles 1-6 KL 0.072/0.095, 0.062/0.074, 0.110/0.196, 0.090/0.107,
  0.079/0.103, 0.076/0.061; keeper BC 1.73 -> 1.62. Breakdown (c1, own
  play): exec_move 0.55, direction 0.45 (cos ~0.85), sprint 0.10, tackle
  0.075, kick 0.045, region 0.02 (+ frozen decision heads). Shot probe c3
  -> c6: dive 0.47/0.15 -> 0.52/0.14, move agree 73 -> 78%, dir cos 0.60,
  recovery 22 -> 20%. Saves vs rules attacker (60 seeds): c3 and c6 both
  3 sampled / 8 deterministic (run51: 1 / 5; fair ceiling ~17). Cycle 6
  epoch curve 0.980 -> 0.969 (slowing), no fresh-rollout gap: ~flat.
- 20:4x run55 done (10 cycles): KL c7-c10 0.073/0.076, 0.111/0.246 (peak),
  0.080/0.097, 0.084/0.071; keeper BC 1.60 / 1.58 / 1.63 / 1.61 (flat).
  Final: shot-in-flight P(move) 0.66 (agree 79%), dive 0.51/0.13, dir cos
  0.63, recovery 18%; open play dir 0.93. Saves vs rules attacker (60
  seeds): 6 sampled (24 goals) / 6 deterministic (23 goals); c6 was 3 / 8.
  Fair ceiling ~17. Waiting for the user's go on PPG refit + PPO.
- 21:xx Why ~6 saves (fair ceiling ~17)? run55 final, rules attacker, 80
  seeds: shot-in-flight imitation vs AI attacker is WORSE (38 rows, dir cos
  0.30, move agree 74%) than vs rules attacker (256 rows, 0.63, 79%) -- so
  not "too few rules-attacker rows". Labels during shots: MOVE labels point
  exactly at keeper_save_target (median 0.0 deg, 100% within 10 deg) =
  a function of observable state; 42/256 STAND labels 3 m from the save
  point all come from SaveOrder itself (closing_down not involved), mostly
  in the first 1-2 decision steps of the shot -- likely brake-to-turn in
  _compute_movement_intent (use_brake_to_turn=True), which depends on the
  keeper's velocity (observed). -> not hidden state; the network
  under-fits the intercept geometry (shot cos 0.63 vs open play 0.93).
  Options: PPO; upweight shot-in-flight rows in role BC (opt-in); a
  dedicated shot-row fit to test capacity.
- User: juice the BC coefficients in general. role_bc goalkeeper 0.6 -> 1.2
  (run55 KL mostly 0.06-0.11, spikes 0.196 / 0.246, ended 0.084/0.071;
  keeper BC now ~1.6 vs ~107 when 1.0 wrecked run37). Run56 from run55
  final, 10 cycles, lr 1e-4, warmup 500.
- 23:2x run56 (role BC 1.2) cycles 1-6: KL 0.137/0.103, 0.080/0.092,
  0.088/0.068, 0.087/0.058, 0.117/0.105, 0.107/0.101; keeper BC 1.57 ->
  1.50. Shot probe (run55 final -> c3 -> c6): dir cos 0.63 -> 0.68 ->
  0.75, move agree 79 -> 81 -> 78%, dive 0.51/0.13 -> 0.49/0.13 ->
  0.48/0.12. Saves vs rules attacker (sampled / deterministic, 60 seeds):
  6/6 -> 9/8 -> 7/6 (noise band ~6-9; ceiling ~17). Open-play dir 0.94.
- 23:38 run56 done: KL c7-c10 0.099/0.066, 0.130/0.146, 0.093/0.065,
  0.098/0.108; keeper BC 1.51 / 1.44 / 1.44 / 1.49. Final: shot dir cos
  0.76 (from 0.63 at run55 final), move agree 79%, dive 0.52/0.14,
  recovery 18%; open play dir 0.95. Saves vs rules attacker (60 seeds):
  6 sampled / 7 deterministic (c3 9/8, c6 7/6; run55 final 6/6) -- within
  the ~6-9 noise band: decisions keep improving, saves do not follow yet.
- 23:4x User: more BC, no PPO until reasonably close to the rules keeper.
  role_bc goalkeeper 1.2 -> 2.0; run57 from run56 final, 10 cycles. Test:
  shot-time move agreement (~79%) and saves (6-9 / 60) must rise above the
  band; otherwise BC is done.
- 23:5x User: juice it even higher ("we can always cut back"). run57 (2.0)
  stopped before its first cycle; role_bc goalkeeper 4.0; run58 from run56
  final, 10 cycles, same test (move agreement / saves above the band).
- 00:1x run58 (role BC 4.0) cycles 1-6: KL 0.123/0.105, 0.157/0.089,
  0.126/0.158, 0.154/0.084, 0.147/0.101, 0.223/0.102; keeper BC 1.45 ->
  1.31. c3: dir 0.79, dive 0.50/0.10, move agree 76%, saves 6/8. c6: dir
  0.81, dive 0.61/0.11, move agree 80%, recovery 17%, saves 10 sampled /
  9 deterministic (goals 18 / 20) -- first result above the 6-9 band
  (borderline on 60 seeds; confirm at c9/c10).
- 00:3x run58 done (role BC 4.0, 10 cycles): KL c7-c10 0.164/0.152,
  0.180/0.109, 0.236/0.271 (peak), 0.205/0.140; keeper BC 1.45 -> 1.24.
  c9: saves 10 / 10, move agree 82%. FINAL: saves vs rules attacker (60
  seeds) 11 sampled (18 goals) / 13 deterministic (16 goals) -- out of the
  old 6-9 band, ~65-75% of the fair ceiling (~17). Shot-in-flight: move
  agree 81%, dir cos 0.78, dive 0.51/0.11, recovery 16%; open play dir
  0.97, move agree 83%.
- 01:3x User: that'll do -- short PPG then PPO. Chain script
  (scratchpad/ppg_then_ppo.sh): PPG value refit --ppg-num-rollouts 3 from
  checkpoints/phase2_run58/checkpoint_pretrained.pt -> checkpoints/phase2_run59,
  then PPO from its checkpoint_pretrained.pt, --total-steps 300000000,
  --phase-mix "1:0.25,2:0.75" (logs scratchpad/ppg_refit_after_run58.log,
  ppo_after_run58.log). Watch the keeper's goalkeeper_vs_rules saves (note:
  the eval uses the new no-snap rules attacker/defender; fair ceiling
  ~17/60 in the 60-seed probe).
- 01:43 PPG refit run59 done (3 rollouts; final value loss 0.4335, policy
  KL ~0.022) -> checkpoints/phase2_run59/checkpoint_pretrained.pt. My chain
  script's path parsing failed (reported "no checkpoint"), so PPO was
  launched by hand: run60 (checkpoints/phase2_run60), --checkpoint
  checkpoints/phase2_run59/checkpoint_pretrained.pt --total-steps 300000000
  --phase-mix "1:0.25,2:0.75" (log scratchpad/ppo_after_run58.log).
- 02:1x run60 PPO healthy (6+ checkpoints, no errors). Evals ckpt 1-6
  (goals/saved %): keeper vs rules attacker 31.1/17.2 -> 32.0/17.2 (rules
  keeper baseline now 7.0/41.7 without the snap); AI attacker vs neural
  keeper 62.8/4.8 -> 64.1/1.7 (saves falling); AI attacker vs rules keeper
  61.5/6.3 -> 59.8/4.8; defender_vs_rules 8.7/43; phase-1 64-65%.
- 02:1x Side effect of the direction-head rescale on PPO (run60): direction
  param group pre-clip grad norm mean 44 (run43: 2.2), clipped 141/141 steps
  (limit 4.0); policy KL per update 0.07-0.11 (run43 ~0.02); updates
  early-stop at epoch 3/10 (KL 0.644 > target 0.5) with move_dir = 0.643 of
  it. Cause: ppo.direction_learning_rate 3e-5 unchanged, but with raw norm
  ~1 (was ~40) each Adam step rotates the direction ~30x more. Proposed
  (waiting for the user): direction_learning_rate 3e-5 -> 3e-6, restart from
  the latest run60 checkpoint into a fresh dir.
- 02:1x User: most interesting overall = scored / saved / defended share
  (excluding off target, out, timeout) in the all-neural game, plus the
  keeper-specific evals. All-neural from run60's TRAINING rollouts (not
  seeded): scored 65.3 -> 64.5%, saved 12.6 -> 10.4%, defended 22.1 ->
  25.2% (rollouts 1-7). Added a seeded eval spec all_neural (phases.py;
  test updated) -- active from the next restart. Keeper vs rules attacker
  ckpt 7: 36.3/12.8 (worse than 31/17 at ckpts 1-4); AI attacker vs neural
  keeper 64.1/1.5.
- 02:2x User asked whether the high direction gradients are temporary and
  whether to relax kappa: move_dir kappa ~943 (noise ~1.9 deg; KL for a
  mean shift ~ kappa*dtheta^2/2, so 0.64 KL = ~2.1 deg average rotation).
  Not temporary (norm held ~1 by the reg; direction weights ~16x smaller,
  so the same LR rotates them much more). Both unit-vector heads (move /
  kick direction) carry the magnitude reg in PPO (coef 0.001, in
  total_loss). User: suspicious of specialised LRs, probe first. run60
  stopped at ckpt 8 (kept). Probes (scratchpad/dirlr_probes.sh, sequential,
  config restored after): 2-round PPO from run60 ckpt 8 with
  ppo.direction_learning_rate 3e-5 / 1e-5 / 3e-6; compare early-stop epoch,
  per-head KL, keeper evals (incl. new all_neural).
- 02:1x Direction-LR probes abandoned before any result (user moved to
  target_kl 2.0, kappa reset, +30% entropy -- the probes' question no
  longer applies). NB: TaskStop on the probe script did not stop its
  training process or its bash loop (it advanced to the next probe twice);
  had to taskkill the process trees -- partial dirs phase2_run61/62/63 from
  those (config + log only, no PPO checkpoints). Config restored from the
  backup before the new changes.
- 02:16 User: kappa back to ~350 ("it'll tighten again in no time, easier
  for the keeper to explore"), entropy +30%, target_kl up (it was 2.0 for
  most of phase 1; can go entirely if useless). New start checkpoint
  checkpoints/phase2_run60_kappa350/checkpoint_start.pt = run60 ckpt 8 with
  move_dir log kappa set to ln 350 (was 941), optimizer state dropped.
  ppo.ent_coef_start/end 0.003 -> 0.0039, ppo.target_kl 0.5 -> 2.0,
  direction LR unchanged (3e-5). PPO run64 (checkpoints/phase2_run64),
  --total-steps 300000000, phase mix 25:75, with the new all_neural eval
  (log scratchpad/ppo_kappa350.log).
- 02:18 User: cut direction LR to 1e-5 ("not any lower"). run64 stopped
  after ~1 min (no checkpoint; process tree killed and verified), relaunched
  as run65 from the same kappa-350 start checkpoint
  (log scratchpad/ppo_kappa350_dlr1e5.log).
- 02:2x User: report every other checkpoint; going to bed at some point --
  train as far as sensible by tomorrow, log, nothing destructive, choices
  well justified and not hacky, keep timers running. And: don't read much
  into trends over the first ~0-20 evals of a run (episode-replay list still
  building up, noisy) -- no regression calls / plan changes on those alone.
- 02:2x User: after the kappa reset the base numbers differ -- judge drops
  relative to run65's own first eval, not run60's. run65 ckpt-1 baseline:
  phase-1 vs rules 62%; all_neural scored/saved/defended share (excl. off
  target/out/timeout) 65.9/7.7/26.4% (raw goal 61.5, saved 7.2, defence
  24.6); goalkeeper_vs_rules 36.7/11.5; attacker_vs_ai_keeper 60.7/4.8;
  attacker_vs_rules 62.0/4.1; defender_vs_rules 8.7/43.0. Phase-1 guard:
  sustained < ~59% after ckpt ~20.
- 03:07 watchdog: run65 healthy, 9 checkpoints (~4.5 min/round), no errors,
  no early stops; approx KL 0.175 (update 1) -> 0.027-0.037, move_dir KL
  0.025-0.034 (no longer dominant; kick_dir 0.03-0.06); move_dir log kappa
  5.858 -> 5.842 (kappa ~345, slowly loosening, not re-tightening yet).
  Evals ckpt 1-9: phase-1 62-64%; attacker_vs_rules 58.3-62.0%;
  defender_vs_rules 7.6-9.3%; keeper vs rules attacker (every 2nd) 37.2/10.9,
  37.0/11.7, 36.1/12.0, 34.3/13.7; AI attacker vs neural keeper 60.9/3.9,
  61.1/3.0, 60.0/3.0, 59.6/3.9; all_neural share 68.1/6.1/25.8,
  66.6/6.8/26.6, 67.5/7.3/25.2, 65.8/6.3/27.9. Within the first-20 noise
  window -- no trend calls, no changes.
- 04:07 watchdog: run65 healthy, 21 checkpoints, no errors, 0 early stops
  over the whole run; approx KL ~0.025, move_dir KL ~0.023, kick_dir
  ~0.035; move_dir log kappa 5.858 -> 5.828 (kappa ~340, drifting slowly
  down). Evals (every 2nd ckpt, 10-19): keeper vs rules attacker 35.2/13.3,
  36.1/12.2 (13), 35.4/12.8, 35.0/13.5, 33.9/14.3; AI attacker vs neural
  keeper 60.0/2.0, 61.1/3.7, 60.2/4.1, 62.2/3.5, 60.7/4.1, 63.0/5.0;
  all_neural share (scored/saved/defended) 63.4/7.5/29.1, 62.6/10.8/26.6,
  64.9/9.3/25.9, 61.1/12.2/26.7, 62.8/10.2/27.0, 63.1/11.0/25.9.
  attacker_vs_rules 58.5-62.0%, defender_vs_rules 8.0-9.8%, phase-1 62-64%
  (guard ok). First ~20 checkpoints done; early signs vs ckpt-1 baseline:
  all-neural saves 7.7 -> ~10-12%, scored 65.9 -> ~62-63%, keeper vs rules
  attacker 36.7/11.5 -> ~34-35/13-14. No changes.
- 05:08 watchdog: run65 healthy, 32 checkpoints, no errors, 0 early stops;
  policy kl ~0.019-0.03, per-head move_dir ~0.030, kick_dir ~0.050; move_dir
  log kappa 5.81 (kappa ~333, still loosening slowly). Evals (every 2nd ckpt,
  23-31): keeper vs rules attacker 32.4/15.9, 32.8/15.4, 35.2/13.3,
  34.1/14.3, 31.5/16.5; AI attacker vs neural keeper 60.4/8.0, 62.2/7.2,
  60.9/6.7, 59.3/7.6, 61.1/6.7; all_neural share 60.9/11.9/27.2,
  61.1/12.2/26.7, 63.5/10.8/25.7, 61.9/12.3/25.8, 62.5/11.9/25.6.
  attacker_vs_rules 61-62%, defender_vs_rules 7.4-7.8% conceded / 45% saved,
  phase-1 vs rules 62-63% (guard ok). Sustained keeper gain vs ckpt-1
  baseline (vs rules 36.7/11.5 -> ~33/15; AI-attacker saves 4.8 -> ~7;
  all-neural saves 7.7 -> ~12). No regressions; no changes.
- 06:07 watchdog: run65 healthy, 44 checkpoints, no errors, 0 early stops;
  policy kl ~0.023-0.026; per-head move_dir 0.017-0.08 (one-update spike
  to 0.08, fine at target_kl 2.0), kick_dir ~0.044; move_dir log kappa 5.79
  (kappa ~327, slow loosening). Evals (every 2nd ckpt, 33-41): keeper vs
  rules attacker 31.5/16.7, 31.5/16.7, 31.1/17.2, 33.0/15.2, 32.2/16.1; AI
  attacker vs neural keeper 62.2/6.3, 58.7/8.9, 59.1/7.8, 60.2/10.2,
  60.2/7.8; all_neural share 62.0/12.1/25.9, 60.8/12.9/26.3,
  62.7/11.4/25.9, 59.4/14.6/26.0, 60.4/12.6/27.0. attacker_vs_rules
  59.6-61.3%, defender_vs_rules 8.7-8.9% conceded / 43-44% saved, phase-1
  vs rules 62-63% (guard ok). Keeper gains vs ckpt-1 holding / creeping up
  (vs rules attacker ~32/16 vs 36.7/11.5; all-neural saves ~12-13 vs 7.7).
  Checked the identical 31.5/16.7 at ckpt 33 and 35: ckpt 34 scored
  32.6/15.7, so the eval is not stale (defence share is fixed by the rules
  defender on seeded episodes). No changes.
- 07:07 watchdog: run65 healthy, 55 checkpoints, no errors, 0 early stops;
  policy kl 0.026-0.055, move_dir KL ~0.02-0.025, kick_dir ~0.045-0.054;
  move_dir log kappa 5.78 (kappa ~324). Evals (every 2nd ckpt, 43-53):
  keeper vs rules attacker 32.6/15.7, 31.7/16.5, 31.7/16.5, 32.2/16.3,
  32.4/16.1, 32.4/16.1 (plateau ~32/16 since ckpt 31); AI attacker vs
  neural keeper saves 8.9, 8.5, 11.5, 10.2, 11.1, 9.3 (slowly rising);
  all_neural share 61.4/9.5/29.2, 64.2/9.2/26.6, 62.0/9.4/28.6,
  63.1/10.6/26.3, 62.9/12.5/24.6, 62.5/12.3/25.3 (dip at 43-47 recovered;
  keeper-only evals did not dip, so read as three-way-game noise).
  attacker_vs_rules 62.4-63.0%, defender_vs_rules 8.5-9.6% / 41-43% saved.
  Phase-1 vs rules 60-62% over the last 6 (was 62-63%): a slight drift,
  above the ~59% guard; watching, no change.
- 08:07 watchdog: run65 healthy, 66 checkpoints, no errors, 0 early stops;
  policy kl ~0.006-0.024, move_dir KL 0.014-0.020, kick_dir ~0.045; move_dir
  log kappa 5.775 (kappa ~322). Evals (every 2nd ckpt, 55-65): keeper vs
  rules attacker 30.7/17.6, 30.4/17.8, 30.7/17.4, 30.4/17.8, 30.7/18.0,
  31.7/16.7 (new best band ~30.5/17.5); AI attacker vs neural keeper
  59.1/8.0, 57.6/11.7, 56.5/10.0, 57.4/11.3, 56.3/10.9, 56.7/10.7;
  all_neural share 62.4/12.0/25.7, 63.3/12.1/24.7, 61.2/12.6/26.1,
  61.2/13.5/25.3, 64.7/12.4/22.9, 62.3/13.9/23.8. Checked the lower
  all-neural defended share (22.9, 23.8 at 63/65 vs 26.4 baseline):
  defender_vs_rules goals conceded flat (8.7-9.8%, baseline 8.7) but its
  defence_possession drifted 44 -> 41% while the rules keeper's saves rose
  43 -> 46% -- the neural defender wins the ball slightly less often and
  more shots reach the keeper; no change in goals. Mild, watching, not a
  regression trigger. attacker_vs_rules 60.9-63.3% (saves 5-7%), phase-1 vs
  rules 60-62% (guard ~59% ok). No changes.
- 09:07 watchdog: run65 healthy, 78 checkpoints, no errors, 0 early stops;
  policy kl ~0.02, move_dir KL 0.015-0.037, kick_dir ~0.045; move_dir log
  kappa 5.765 (kappa ~319). Evals (every 2nd ckpt, 67-75): keeper vs rules
  attacker 30.4/17.6, 29.6/18.5, 29.8/18.3, 26.5/21.5, 28.0/20.4 (a step
  up at 73-75); AI attacker vs neural keeper 57.6/11.3, 56.7/12.0,
  57.4/9.1, 56.5/12.4, 55.0/13.9 (best); all_neural share 61.8/14.0/24.2,
  62.7/13.2/24.1, 64.4/11.9/23.7, 63.3/14.1/22.6, 62.0/13.7/24.3.
  Defender recheck: defender_vs_rules goals conceded 7.8-8.9% (baseline
  8.7, flat / slightly better), defence_possession steady ~42% over the
  last 8 (the 44 -> 41 drift stopped), rules keeper saves ~45%. So the
  lower all-neural defended share (~23-24 vs 26.4) is not a defender
  regression in goals; more likely the neural attacker/keeper interplay in
  the three-way game. attacker_vs_rules 61.3-62.2% (saves 6.7-8.5%);
  phase-1 vs rules 60-62% (guard ok). No changes.
- 10:07 watchdog: run65 healthy, ~89 checkpoints, no errors, 0 early stops;
  policy kl 0.022-0.028, move_dir KL 0.015-0.018, kick_dir ~0.045; move_dir
  log kappa 5.755 (kappa ~316). Evals (every 2nd ckpt, 77-87): keeper vs
  rules attacker 27.2/21.1, 28.7/19.6, 28.5/19.6, 28.0/20.2, 28.7/19.8,
  29.1/19.6 (holding the 73+ step, ~28/20); AI attacker vs neural keeper
  saves 10.7, 10.7, 10.9, 13.3, 15.0, 15.9 (rising; 54.8% conceded at 87,
  best); all_neural share 63.7/14.2/22.1, 63.4/12.4/24.2, 61.7/13.3/25.0,
  61.1/14.5/24.4, 57.0/20.6/22.4, 60.0/16.5/23.5 (saves stepping up at
  85-87). attacker_vs_rules now 64.1-67.4% (was 61-63; baseline 62.0) --
  the attacker improving too. defender_vs_rules 8.0-8.5% conceded,
  possession ~42.5% (flat). phase-1 vs rules 60-62% (guard ok). No changes.
- 11:07 watchdog: run65 healthy, 100 checkpoints, no errors, 0 early stops;
  policy kl 0.022-0.030, move_dir KL ~0.018, kick_dir ~0.045; move_dir log
  kappa 5.746 (kappa ~313). Evals (every 2nd ckpt, 89-99): keeper vs rules
  attacker 27.8/20.4, 28.0/20.7, 28.5/20.4, 27.6/21.1, 27.4/20.7,
  28.9/19.6 (holding ~28/20); AI attacker vs neural keeper 52.4/15.2,
  52.4/18.3, 50.9/20.9, 49.6/23.0, 48.3/24.3, 51.5/22.6 (big rise from
  ~11% saves at ckpt 81); all_neural share 58.5/16.0/25.4, 58.7/20.6/20.6,
  57.3/21.5/21.1, 52.8/24.6/22.5, 55.4/22.4/22.2, 53.5/24.4/22.1.
  Phase-1 vs rules 59-62% (one 59 in the last 8; guard is SUSTAINED <59).
  Tested why the all-neural defended share fell (26.4 -> ~21-22):
  hypothesis "the neural attacker now beats the defender more, not the
  neural defender getting worse". Prediction that could have refuted it:
  with a RULES defender against the neural attacker (attacker_vs_rules),
  defence_possession should fall too, and the neural defender against the
  rules attacker should stay flat. Result: attacker_vs_rules
  defence_possession 32.4 -> 20.7% over the run (goals 62.0 -> 64-66%,
  rules keeper saves 4.1 -> ~10%); defender_vs_rules defence_possession
  44.8 -> 42.4% with goals conceded flat 8-9%. So the drop is the attacker
  improving (dribbling past the defender), not a defender regression. No
  changes.
- 11:55 event: ckpt 107 keeper vs rules attacker dropped to 37.6/10.4 (from
  27.6/21.3 at 105, 28.0/20.4 at 106). Cause found in the log: the 11:45
  update's policy KL jumped 0.019 -> 0.50 within one epoch (one minibatch
  grad-norm max 27 vs ~5-11 normal), per-head move_dir KL 0.276 (was
  0.015-0.048 for the previous hour); target_kl 2.0 did not stop it. Same
  ckpt: attacker_vs_rules 72.6% (from ~65), attacker_vs_ai_keeper saves
  18.5 (from 21-24), all_neural share 58.1/19.6/22.3 (from ~52/26/22). The
  11:50 update was again large (KL to 0.78, move_dir 0.351), yet ckpt 108
  keeper vs rules = 25.4/23.3 (best of the run) and attacker_vs_rules
  70.2%. move_dir kappa unchanged (log 5.737), mean per-step angle change
  unchanged (~0.18 deg), so the KL is likely concentrated in a few rows
  (untested). Reading: a sudden larger policy move in move direction, not
  (yet) a regression -- 107 bad, 108 best. Not acting on two updates;
  watching move_dir KL and the keeper evals. Last checkpoint before the
  jumps: checkpoint106.pt (keep as the rollback point if it turns out to
  be damaging).
- 12:07 watchdog: run65 healthy, 112 checkpoints, no errors, 0 early stops.
  The move_dir KL jump was two updates only (11:45 0.276, 11:50 0.351);
  since then 0.019, 0.023, 0.021 (normal); kappa unchanged (log 5.736,
  ~310); policy kl ~0.023-0.030. Evals: keeper vs rules attacker 107
  37.6/10.4 (the dip), 108 25.4/23.3, 109 25.7/22.8 (best of run); AI
  attacker vs neural keeper 101-109 saves 21.7, 20.9, 21.3, 18.5, 19.6;
  all_neural share 101 57.8/21.4/20.8, 103 54.3/25.2/20.5, 105
  52.1/25.8/22.1, 107 58.1/19.6/22.3, 109 53.8/24.9/21.3. After the jump
  the attacker stepped up: attacker_vs_rules 70.2-72.6% (from ~65; base
  62.0), rules defender possession ~18% -- which plausibly explains the
  slightly lower keeper saves vs the AI attacker (harder attacks) while the
  keeper vs the rules attacker is at its best. defender_vs_rules 8.5-8.7%
  conceded, possession ~43% (flat). Phase-1 vs rules 59-62% (one 59;
  guard is sustained <59). Verdict on the jump: transient, net positive;
  no rollback, no changes.
