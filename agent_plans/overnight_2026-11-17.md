# Overnight run log, 2026-11-17

Running log of the unattended overnight work (user asleep ~10 h). Every run,
its settings, results and the reason for each decision. Newest last.

## Starting point (before the night)

- **Goal:** one network that plays both phase 1 (1v1 box possession) and
  phase 2 (attacker vs rules defender + keeper), then PPO to improve on it.
- **Teachers** (`ai_config.json` `distill_pretrain.teachers`):
  - phase 1: `checkpoints/migrated/phase1_run321_checkpoint763_migrated.pt`
    (65% wins vs rules at 22 s, 0% misses);
  - phase 2: `checkpoints/phase2_run6/checkpoint3.pt` (75.1% goals vs rules,
    1840 episodes; all-rules baseline 35.7%).
- **Distillation pretrain** (`train.py --distill-pretrain-only`,
  `ai/ppo/distill_pretrain.py`): student-driven rollouts over a 50/50 mix;
  each row twice (real: own teacher + MC value; counterfactual: task id
  swapped, the other teacher, weight `counterfactual_weight`); forward KL on
  the discrete heads, means only on the continuous ones, kick / move sub-heads
  weighted by the teacher's probabilities; label smoothing 1e-5.
- **phase2_run20** (from 763, counterfactual weight 0.1, relaxed spreads
  kappa 500 / 20, sigma 0.25 / 0.5, 4 cycles x 370k x 12 epochs, warmup 1000):
  after-fit KL phase 1 real 0.099, phase 2 real 0.076, counterfactuals 0.49 /
  2.2. Offline evals of its checkpoint:
  - relaxed spreads: phase 1 53-55% wins (misses ~10%), phase 2 70.5-73.7%;
  - 763's tight spreads: phase 1 **62%** (misses 3.7%), phase 2 **83.3%**
    (shots off target 1.4%).
- **phase2_run21** (from run20, user's spreads move kappa 500, kick_dir kappa
  30, elevation sigma 0.15, power sigma 0.25; counterfactual weight 0.3):
  running at the start of the night. Phase-1 real after-fit KL ~0.21 per
  cycle, fresh-rollout 0.34 -> 0.38 (readable numbers flat: move 1.0 deg,
  power diff 0.025); phase 2 real 0.10 -> 0.069; counterfactuals improving.

## Instructions for the night (user, before bed)

1. Keep running distillation continuations until the KLs really stop
   improving; want them low, counterfactuals included.
2. "Looks good" = per-group KL under 0.1 (ideally 0.05), and both phases
   within 10% of their teachers when evaluated at the same spreads (phase 1
   >= 58.5% vs 763's 65%, phase 2 >= 67.6% vs run6's 75.1%). If the KL is
   low the evals will follow; don't over-worry.
3. Then PPO from the best distilled checkpoint: 70/30 phase-1/phase-2 mix,
   NO BC and NO teacher distillation term (watch it learn freely), the
   current slightly relaxed spreads, whatever else seems sensible. Go big:
   ~300M steps, still running in the morning to look at graphs / logs
   together.
4. Nothing destructive; document every change (this file, commit messages).

## Log

### 00:41 phase2_run21 done (continuation of run20, counterfactual weight 0.3, user's spreads)

After-fit KL by cycle (real phase 1 / real phase 2 / cf phase 1 / cf phase 2):
c1 0.227 / 0.100 / 0.362 / 1.84; c2 0.213 / 0.069 / 0.253 / 1.45;
c3 0.231 / 0.080 / 0.214 / 1.24; c4 0.231 / 0.066 / 0.210 / 1.11.
Phase-1 real is flat at ~0.22 (readable: move 0.9 deg, power diff 0.017-0.025,
P(kick) matched); phase-2 real and both counterfactuals still falling.

Offline evals of `phase2_run21/checkpoint_pretrained.pt`:
- phase 1 vs rules (920 eps, 22 s): **61%** at the user's spreads (misses
  2.1%), **66%** at 763's spreads (misses 0.4%) -- 763 itself 65%;
- phase 2 attacker (1840 eps): **76.1%** at the user's spreads, **80.6%** at
  763's -- run6 (teacher) 75.1%.
So the eval half of "looks good" is already met; the KL half is not (phase-1
real 0.23, cf phase 2 1.1 vs the < 0.1 target).

Label smoothing (user asked to check): it cannot explain the phase-1
plateau -- at the start of run20 the student WAS 763 and its KL to the
smoothed 763 targets was 0.0005, the whole floor smoothing adds. Kept on.

Change: group report now also logs the top-5 per-head KL per group (to see
what the phase-1 0.23 is made of). Next: phase2_run22 = continuation from
run21 with lr 1e-4 -> 3e-5 (one change: is the plateau an LR noise floor?).

### 00:45 phase2_run22 cycle 1 "before" -- per-head KL breakdown

Move direction is most of every group's KL (top heads):
- phase 1 real 0.288: move_dir 0.208, kick_power 0.023, kick 0.021, sprint 0.011, exec_move 0.011;
- phase 2 cf 1.329: move_dir 0.982, kick_power 0.104, kick 0.096, sprint 0.050, exec_move 0.040;
- phase 1 cf 0.198: move_dir 0.131; phase 2 real 0.068: move_dir 0.044.

Note on the "KL < 0.1" target: at move kappa 500 the von Mises KL is about
kappa (1 - cos dtheta) ~ 250 dtheta^2, so a ~1 deg mean error already costs
~0.1, and the quadratic makes a few larger-error rows dominate. The KL target
is effectively a move-direction precision target (rms error under ~1 deg for
phase 1), and its scale depends on kappa (the same match reads ~5x lower at
kappa 100). The behavioural evals already meet the teachers (run21).

### 00:52 Finding: the phase-1 plateau is a generalisation gap, not (only) the LR

run22 cycle 1 (lr 3e-5) fitted phase 1 real down to 0.199 (move_dir 0.136),
but cycle 2's fresh rollout read 0.291 again -- the same as run21's fresh
0.29. Each cycle fits its own 370k-step rollout well, and the next rollout's
new states show the error again; the data of each cycle was discarded.

Change (code, opt-in): `distill_pretrain.aggregate_cycles` (default 1 = old
behaviour) -- proper DAgger aggregation: fit on this cycle's rows plus the last
(N - 1) cycles' rows. Old rows keep their teacher targets (frozen teachers)
but train the KL only, not the value head (their MC returns came from an
earlier student). Tests: concat / history, and the smoke test runs 2 cycles
with aggregation. To be used from the next continuation (N = 3).

### 00:51 phase2_run22 stopped after cycle 2 (checkpoint saved), continuing with aggregation

run22 after-fit (real p1 / real p2 / cf p1 / cf p2): c1 0.199 / 0.054 / 0.176 / 1.21;
c2 0.191 / 0.054 / 0.199 / 1.20. Small steps; fresh rollouts keep resetting
phase 1 to ~0.29. Next: phase2_run23 from run22's checkpoint, aggregate_cycles
1 -> 3 (only change; lr 3e-5, counterfactual weight 0.3 kept).

### 01:10 phase2_run23 (aggregation 3, lr 3e-5, cf 0.3), cycles 1-3

After-fit (real p1 / real p2 / cf p1 / cf p2): c1 0.184 / 0.053 / 0.188 / 1.25;
c2 0.173 / 0.053 / 0.162 / 1.09; c3 0.171 / 0.046 / 0.154 / 1.11.
Fresh-rollout phase-1 real: 0.25, 0.219, 0.216 (was resetting to ~0.29
without aggregation) -- the fresh-data gap shrank from ~0.1 to ~0.04.
Phase 2 real is under 0.05. Flattening again; cf phase 2 flat at ~1.1.

Hypothesis for cf phase 2 (to test, not assumed): its target is run6 on 1v1
states labelled task id 2, an input run6 never saw (its own mix had phase-1
states at task id 1), so run6's outputs there may be erratic / partly
unlearnable. 763's counterfactual is also OOD, but 763's task-id columns are
zeroed, so it ignores the label and fits fine.

**01:12 Hypothesis refuted (tested).** On 2218 phase-1 states: run6's move
direction with task id 1 vs task id 2 differs by 0.11 deg mean (p90 0.26);
its sensitivity to a 1% input nudge is the same with either label (move angle
2.37 vs 2.36 deg mean, |dP(kick)| 0.024 vs 0.024) and the same as 763's. So
run6 on "1v1 + task id 2" is just run6's own smooth 1v1 behaviour (P(kick)
0.25 vs 763's 0.12) -- run6 itself barely uses the task id. cf phase 2's
plateau is the weighting trade-off: on the SAME 1v1 states the student must
produce 763's behaviour (real, weight 1) and run6's (cf, weight 0.3) split
only by the task id; the gradient favours 763 there. Lever: a higher
counterfactual weight, watching phase-1 real.

### 01:20 User's idea (woke briefly): does the student switch behaviour on the keeper instead of the task id?

Probe (`scratchpad/no_keeper_probe.py`, not committed): phase 2, task id 2,
150 seeds, keeper present vs removed from the match (role env without the
goalkeeper role):

| | keeper | goals | kicks/ep | power | notable |
|---|---|---|---|---|---|
| 763 | yes | 8.7% | 0.13 | 0.05 | |
| 763 | no | 16.0% | 0.35 | 0.19 | 112/150 dribbled out |
| run6 | yes | 74.7% | 0.89 | 0.83 | |
| run6 | no | 24.0% | 0.66 | 0.89 | 65 shots off target, 35 out |
| student (run23 c3) | yes | 82.0% | 0.93 | 0.84 | |
| student | no | 20.0% | 0.53 | 0.89 | 60 off target, 55 out |

- The student does NOT revert to phase-1 behaviour without the keeper: it
  keeps shooting hard (power 0.89, 0.53 kicks/ep vs 763's 0.13). So the
  phase-2 "mode" isn't keyed on the keeper.
- But run6 and the student both collapse at an EMPTY goal (75-82% -> 20-24%,
  mostly wide): their shot aim / timing is keeper-relative. The student
  inherited it from run6. Partly just out-of-distribution (one opponent fewer
  than ever seen).
- As a distillation augmentation: no good teacher for it (run6 is 24% there).
  As a phase-2 setting for PPO (~10% episodes without the keeper): sensible
  for robust shooting (and 5v5, keepers out of position), needs per-episode
  role sets in RoleScenarioEnv. NOT added to tonight's PPO run (the agreed
  setup; changing the scenario unattended would muddy it). Candidate for the
  morning.

### 01:22 Added (off): `distill_pretrain.counterfactual_kick_only_phases`

User's idea (then withdrawn once the "erratic" hypothesis was refuted):
counterfactual rows targeting a listed phase take only the KICK heads from
that phase's teacher, every other head from the row's real teacher (e.g. 1v1
states as task id 2: shoot like run6, move like 763). Implemented + tested,
default () = off; NOT used tonight (the user's design -- counterfactuals match
the other teacher's whole output -- is kept). Possible later use: the cf phase
2 KL is mostly move direction (0.98 of 1.33).

Checked (user asked): phase 2 can't overfit to a side / team -- the attacker
is always Team.LEFT, but observations are canonicalised (x mirrored by the
player's attacking direction, incl. the absolute last_touch_team_direction;
y-canonical), so inside the network the attacker always attacks +x and the
absolute team is unrecoverable; the scenario is randomised per seed.

### 01:24 Hourly check-in scheduled (session cron job 3946e5bc, :17 past each hour, first ~02:24)

In case of usage limits / stalls: each check-in re-reads this log and the
latest run logs, checks what is running before launching anything (no
duplicate runs, no resuming into an existing dir), then continues the plan.
Session-only: dies if the Claude session or the machine closes.

### 01:30 Phase-2 player variety (user asked: do attributes vary? avoid overfitting, esp. for PPO)

They did NOT: build_1v2_scenario always used attacker 0.9 / defender 0.55 /
keeper 0.55, all eight attributes equal (PlayerAttributes.average), full
stamina, every episode. Phase 1 randomises attributes (generate_attributes by
tier), stamina and headings.

Added (off by default; code + tests): build_1v2_scenario `attribute_tiers`
({role: [tier, ...]}, attributes drawn per trial with generate_attributes from
a uniformly chosen tier) and `stamina_range`; drawn from their own seeded
stream so a seed keeps exactly the same positions; wired from
`curriculum.phase2.scenario_kwargs` through build_role_env.

Plan: keep it OFF for the distillation (run6, the phase-2 teacher, only ever
saw the fixed matchup, so copying it on random attributes would teach its
out-of-distribution guesses) and switch it ON for the PPO run, which learns
from reward: attacker tiers premier_league / boosted (means 0.72 / 0.71),
defender and keeper generic / championship / league_one (0.55 / 0.56 /
0.44), stamina U(0.5, 1.0). The attacker is weaker on average than the old
0.9, so phase-2 goal rates and the rules baseline under PPO are NOT
comparable with the numbers above (the run recomputes its rules baseline).

### 01:40 phase2_run23 done (6 cycles, aggregation 3, lr 3e-5, cf 0.3)

After-fit (real p1 / real p2 / cf p1 / cf p2): c4 0.172 / 0.045 / 0.137 / 1.00;
c5 0.176 / 0.044 / 0.132 / 1.03; c6 **0.160 / 0.041 / 0.129 / 0.925** (all best).
Phase-1 real flattening at ~0.16-0.18.

Offline evals of `phase2_run23/checkpoint_pretrained.pt` -- the best student
so far; at the user's spreads it now matches or beats both teachers:
- phase 1 vs rules (920 eps, 22 s): **64%** at the user's spreads (misses
  1.7%), 66% at 763's -- 763 itself 65%;
- phase 2 (1840 eps): **79.7%** at the user's spreads, 82.7% at 763's -- run6 75.1%.
(run21 at the user's spreads: 61% / 76%; aggregation helped play too.)

Next: phase2_run24 from run23, counterfactual weight 0.3 -> 0.5 (the lever
left for the counterfactual KLs), aggregation kept; fall back to run23 for
PPO if phase 1 slips on the eval.

### 01:50 phase2_run24 (cf 0.5) cycle 1

After-fit: real p1 0.212, real p2 0.040, cf p1 0.119, **cf p2 0.834** (run23
ended 0.925). Fresh-rollout phase-1 real rose 0.194 -> 0.264 (move 1.0 deg):
the higher weight pulls phase 1 away from 763, slowly. Behaviour check on the
cycle-1 checkpoint: phase 1 vs rules **62.5%** (misses 2.4%) at the user's
spreads vs run23's 64% (within the +-1.6 noise). Rule: re-check after cycle
3; continue while phase 1 >= ~61%, else stop and take the best-by-eval
checkpoint (run23 so far) into PPO.

### 02:45 Distillation stopped; PPO start point = phase2_run23

phase2_run24 (cf weight 0.5) after-fit c2 / c3: real p1 0.219 / 0.227, real p2
0.037 / 0.040, cf p1 0.106 / **0.0996**, cf p2 0.752 / 0.747 (flattening);
fresh-rollout phase-1 real rising 0.194 -> 0.264 -> 0.269 -> 0.305.
Phase-1 eval slid run23 64% -> run24 c1 62.5% -> c3 61%; phase 2 (user's
spreads) run24 c3 80.6% / 85.0% tight -- same as run23 within noise. The
higher weight was trading phase-1 play for counterfactual KL that had stopped
improving: stopped. PPO starts from **phase2_run23** (better phase 1, same
phase 2); run24 kept as the low-counterfactual-KL alternative.

Final distillation KLs (run23 / run24 c3): real p1 0.160 / 0.227, real p2
0.041 / 0.040, cf p1 0.129 / 0.0996, cf p2 0.925 / 0.747. The < 0.1 target
is met for phase-2 real and (run24) cf phase 1, not for phase-1 real (move
direction precision at kappa 500, see 00:45) nor cf phase 2.

Incident: a phase-2 eval of run24 hung at 02:03 (processes alive, ~2 s CPU
each in 38 min; it started while run24's fit was spawning workers) and the
02:24 check-in reported it as running. Killed and re-run (normal, ~5 min). The
hourly check-in was rewritten to check real progress (CPU / log time) and its
own schedule (job bf7c02cd); saved to memory as feedback.

PPO config (committed with this entry): curriculum.phase2.bc = {},
curriculum.distill = {}, curriculum.phase2.scenario_kwargs = player variety
(attacker premier_league / boosted; defender + keeper generic / championship
/ league_one; stamina U(0.5, 1)), eval.neural_snapshot_eval_every_n_rollouts
1 -> 4. Unchanged: n_epochs 1, rollout_steps 250000, target_kl 0.15, PPG off,
attacker-only neural, the student's own spreads. Full suite: 2764 passed.
Launch: --phase 2 --phase-mix "1:0.7,2:0.3" --checkpoint
checkpoints/phase2_run23/checkpoint_pretrained.pt --total-steps 300000000.

### 02:53 PPO run phase2_run25 started (300M steps, ~1.5 min / round -> ~30 h)

Round 1 (rows trained once): phase 1 vs rules (22 s) **63%** (misses 1.3%),
consistent with run23's offline 64% (and confirms the in-run eval-length
fix); phase 2 attacker **33.3%** goals WITH player variety; the all-rules
phase-2 baseline with variety is **25.2%** (35.7% on the old fixed matchup) --
the varied matchup is much harder, so the student is ~1.3x rules there (was
~2.2x on the fixed matchup). Phase-1 rules-vs-rules baseline 46% (unchanged).
Logs: scratchpad/ppo_overnight.log and checkpoints/phase2_run25/training_log*.txt.

### 03:36 PPO phase2_run25 progress (~41 rounds, ~10M steps)

| time | ~round | phase 1 vs rules | phase 2 goals (varied matchup) |
|---|---|---|---|
| 02:52 | 1 | 63% | 33.3% |
| 02:54 | 3 | 63% | 39.6% |
| 03:05 | 13 | 63% | 49.3% |
| 03:15 | 23 | 61% | 56.5% |
| 03:35 | 41 | 63% | 63.9% |

Phase 2 climbing fast (rules baseline with variety 25.2%), phase 1 holding at
the teacher's level (763: 65%) with no BC / teacher term: the separation test
is passing so far.

### 04:40 check-in: PPO phase2_run25 healthy (~100 rounds, ~25M steps)

| time | ~round | phase 1 vs rules | phase 2 goals |
|---|---|---|---|
| 03:45 | 51 | 63% | 59.8% |
| 04:05 | 69 | 63% | 63.5% |
| 04:16 | 79 | 64% | 58.7% |
| 04:35 | 98 | 63% | **66.7%** |

Learning signals (~87 rounds): KL per update ~0.001 (target 0.15 never hit,
0 early stops) -- 1 epoch over ~23 minibatches moves the policy very little;
entropy steady (~-1.38), kick entropy slowly falling, value loss 0.25 ->
0.21. Phase 2 creeping up slowly rather than plateaued; phase 1 rock steady.
Not changed unattended (the user chose the small-step setup for smoothness);
morning option if it stays flat: 2 epochs or a higher LR.
Health: log current, python CPU 30047 s -> 67276 s since 03:39, 100
checkpoints, 623 GB disk free.

### 05:40 check-in: PPO healthy (~156 rounds, ~39M steps)

| time | ~round | phase 1 | phase 2 |
|---|---|---|---|
| 04:46 | 108 | 63% | 65.4% |
| 05:06 | 126 | 63% | 63.5% |
| 05:16 | 136 | 64% | 64.8% |
| 05:36 | 154 | 64% | **68.3%** |

Phase 2 still creeping up, phase 1 steady at 63-64%. Health: log current,
CPU 67k -> 105k s since 04:39, 156 checkpoints (51 python processes =
rollout + eval pools + the every-4th-round snapshot pool).

### 06:40 check-in: PPO healthy (~211 rounds, ~53M steps)

| time | ~round | phase 1 | phase 2 |
|---|---|---|---|
| 05:47 | 164 | 64% | 65.4% |
| 06:06 | 181 | 64% | 63.7% |
| 06:16 | 191 | 64% | **70.2%** |
| 06:36 | 209 | 64% | 68.7% |

Phase 2 high 70.2%, phase 1 steady at 64% (763: 65%). Health: log current,
CPU 105k -> 142k s, 211 checkpoints. (Some rounds' phase-2 eval lands ~39 s
after the phase-1 one instead of ~6 s: the every-4th-round snapshot pool
sharing the CPU; rounds are not slowing overall.)

### 07:40 check-in: PPO healthy (~265 rounds, ~66M steps)

| time | ~round | phase 1 | phase 2 |
|---|---|---|---|
| 06:47 | 219 | 64% | **71.5%** |
| 07:05 | 236 | **65%** | 70.2% |
| 07:17 | 246 | 65% | 71.3% |
| 07:36 | 264 | 64% | 69.1% |

Phase 1 now level with 763 (65%), phase 2 at ~70% on the varied matchup
(rules 25.2%; started at 33.3%). Health: log current, CPU 142k -> 179k s,
265 checkpoints.

### 08:39 check-in

| time | phase-1 vs rules | phase-2 attacker vs rules |
|---|---|---|
| 07:57 | — | **74.3%** |
| 08:20 | — | 74.3% |
| 08:39 | 65% | 74.3% (shot_off_target 3.9%) |

Phase 2 now plateauing around 72-74% (rules 25.2%); phase 1 steady at 65%
(= 763), 45-46% vs its own past snapshots. Health: log current (08:39), CPU
179k -> 216k s, no errors.

### 09:39 check-in

| time | phase-1 vs rules | phase-2 attacker vs rules |
|---|---|---|
| 08:50 | — | 71.7% |
| 09:10 | — | **74.8%** |
| 09:20 | — | 73.7% |
| 09:39 | 64% | 71.1% |

Plateau: phase 2 71-75% for ~2 h, phase 1 64-65%. Health: log current,
CPU 216k -> 253k s, no errors. No changes made.

### 10:39 check-in

| time | phase-1 vs rules | phase-2 attacker vs rules |
|---|---|---|
| 09:50 | — | 70.7% |
| 10:08 | — | 74.6% |
| 10:20 | — | **74.8%** |
| 10:39 | 63% | 74.1% |

Plateau continues: phase 2 71-75% (~3 h), phase 1 63-65%. Health: log
current, CPU 253k -> 290k s, no errors. No changes made.

### 11:39 check-in

| time | phase-1 vs rules | phase-2 attacker vs rules |
|---|---|---|
| 10:50 | — | **78.5%** |
| 11:09 | — | 76.3% |
| 11:19 | — | **78.5%** |
| 11:39 | 63% | 74.8% |

Phase 2 stepped up after ~10:45: three evals at 76-78.5%, now above the run6
teacher (75.1%). Phase 1 checked over 08:23-11:34: flat 62-66% (mostly 64%),
no drift. Health: log current, CPU 290k -> 327k s, no errors. No changes.

### 12:39 check-in

| time | phase-1 vs rules | phase-2 attacker vs rules |
|---|---|---|
| 11:40 | — | 77.0% |
| 11:50 | — | 77.0% |
| 12:09 | — | 77.8% |
| 12:20 | — | 75.7% |
| 12:38 | 64% | 77.0% |

New level holding: phase 2 75.7-78.5% since ~10:45 (above run6 teacher
75.1%); phase 1 64%. Health: log current, CPU 327k -> 364k s, no errors.
No changes.

### 13:39 check-in (user back)

| time | phase-1 vs rules | phase-2 attacker vs rules |
|---|---|---|
| 12:39 | — | 77.2% |
| 12:49 | — | **80.4%** |
| 13:09 | — | 78.7% |
| 13:19 | — | 78.9% |
| 13:39 | 63% | 80.0% |

Phase 2 now ~78-80%; phase 1 63-64%. ~149M of 300M steps. Health: log
current, CPU 364k -> 402k s, no errors. No changes.

### 14:39 check-in

Phase 2 attacker vs rules: 79.8% (14:30), then 81.7 / 82.8 / 81.5 / 81.3%
(14:35-14:38) -- new highs, now ~81-83%. Phase 1 vs rules 63-64%. ~163M of
300M steps. Health: log current, CPU 402k -> 440k s, no errors. No changes.

## Morning (user back): restore training settings, add no-keeper episodes

Config audit vs before the phase-2 test runs (154f14d^): n_epochs 13 -> 1,
rollout_steps 750k -> 250k, target_kl 2.0 -> 0.15, snapshot evals every 1 -> 4,
PPG + interleave on -> off, ppg_mc_returns off -> on were all test-run
settings. User's call:

- `ppo.n_epochs` 10, `ppo.rollout_steps` 500000, `ppo.target_kl` 0.5,
  `eval.neural_snapshot_eval_every_n_rollouts` 2. PPG stays off.
- New: `curriculum.phase2.scenario_kwargs.no_keeper_prob` 0.2 -- 20% of
  phase-2 episodes have no goalkeeper (overnight probe: empty-goal scoring
  only 20-24%).

Implementation: `build_1v2_scenario(no_keeper_prob=)` (own seeded stream; the
keeper is still built so a seed's positions/attributes/stamina are identical
either way); `CurriculumPhase.optional_roles` (phase 2: goalkeeper);
`RoleScenarioEnv.episode_roles` = configured roles present in the match
(a missing non-optional role still raises); eval specs may carry
`scenario_kwargs`: rules_baseline / attacker_vs_rules / defender_vs_rules pin
the keeper IN (comparable with run25), new rules_baseline_no_keeper /
attacker_vs_rules_no_keeper pin it OUT. No-keeper training episodes log under
their own tag `phase2_1v2[attacker=neural,defender=rules]`.
Smoke: 44/200 seeds drop the keeper; rules attacker 34/60 goals at an empty
goal vs 14/60 with a keeper. Tests: tests/ai_scenario/test_phase2_no_keeper.py,
conftest fixture zeroes no_keeper_prob for other tests (run-config leak),
spec-list expectations updated.

### 15:39 run26 check-in

Run26 = `checkpoints/phase2_run26`, from run25 checkpoint684 (stopped 15:12),
log scratchpad/ppo_run26.log. 10-epoch updates run all epochs (no early
stop), KL ~0.095 per update (run25: ~0.001). No-keeper share of phase-2
rollout episodes 20.2% (1369/6768 in round 1).

| time | phase-1 vs rules | phase-2 with keeper | phase-2 no keeper |
|---|---|---|---|
| 15:23 (start) | 65% | 79.8% | 27.6% (rules attacker 56.5%) |
| 15:27 | 63% | 82.0% | 30.0% |
| 15:31 | 63% | 83.3% | 29.3% |
| 15:35 | 64% | 83.5% | 29.8% |
| 15:39 | 64% | 82.4% | 33.5% |

Hourly check-in job replaced: the old prompt's fallback (restart from run23
with 1 epoch / 250k / target_kl 0.15) was stale; the new one restarts from
run26's latest checkpoint with the current config.

### 16:46 run26 check-in

| time | phase-1 vs rules | phase-2 with keeper | phase-2 no keeper |
|---|---|---|---|
| 15:43 | 64% | 83.0% | 34.6% |
| 15:51 | ~64% | 83.7% | 40.7% |
| 16:02 | 63% | 84.3% | 44.1% |
| 16:13 | 64% | 84.3% | 54.6% |
| 16:32 | ~63% | 83.9% | 70.2% |
| 16:44 | 65% | **85.0%** | **75.2%** |

No-keeper 27.6% -> 75.2% in ~80 min (rules attacker 56.5%): dribbling out
27.8% -> 0.2% first, then shots wide 38% -> 15.7%. With-keeper also up
(79.8 -> 85.0%), phase 1 flat 62-65%. Health: log current, CPU 7k -> 32k s,
no errors.

### 17:46 run26 check-in

| time | phase-1 vs rules | phase-2 with keeper | phase-2 no keeper |
|---|---|---|---|
| 17:02 | ~64% | 85.9% | 76.3% |
| 17:14 | ~64% | **87.0%** | 78.5% |
| 17:29 | 64% | **87.0%** | **81.7%** |
| 17:40 | 64% | 85.9% | 81.5% |

No-keeper levelling off ~81-82% (rules attacker 56.5%), with keeper ~86-87%,
phase 1 flat 63-66%. Health: log current, CPU 32k -> 54k s, no errors.
Also added (bddfd1a): UI scenario `phase2_neural_attacker` to watch phase-2
checkpoints (attacker checkpoint picker, no-keeper chance, player variety).
