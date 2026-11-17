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
