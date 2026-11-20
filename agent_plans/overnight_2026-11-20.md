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
