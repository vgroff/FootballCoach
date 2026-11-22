# Phase 2 add-on: a second attacker to pass to (+ offside)

Status: **idea / plan, not started** (user's proposal, 2026-11-20 ~19:20),
to do once the neural keeper is learning.

## Goal

Teach passing (and timing it against offside) inside the existing phase-2
1v2 scenario: sometimes add a second attacker in space that the first
attacker can pass to, like the 2v2 UI scenario. Then refit on it (rules-AI
labels via the phase-2 episode system), and PPO -- "then we'd have all the
basics for football".

## Scenario (user's spec)

- In a share of phase-2 episodes (config), add attacker B:
  - roughly level with the defender along the pitch length, offset in width;
  - 80% start onside, otherwise slightly offside (must be onside again, or
    be caught offside, if passed to).
- Attacker A (on the ball) as now.
- All knobs in config (opt-in, existing defaults unchanged).

## Rules AI for the labels (new / improved)

- Attacker A: carries for < 1 s, then passes to B just before B would go
  offside (or shoots if B is not a better option).
- Attacker B: makes the run for the ball, receives, then the usual shoot
  order.
- Possibly sweep the timing / thresholds (single process, or only when
  nothing is training).

## Offside -- findings (2026-11-20), must be fixed first

1. **Not enforced at all.** `engine/offside.py` has `is_offside_position` /
   `check_offside_on_pass` and `Match` has an `offside_enabled` flag, but
   nothing in the match loop calls them (grep: no call sites outside
   offside.py).
2. **Simplified line = deepest defender INCLUDING the keeper** (module
   docstring: "a deliberate simplification of the real second-last-defender
   rule"). With one defender + one keeper the line would be the keeper, so B
   could stand beyond the defender and never be offside -- no offside to
   time against. Needs the real second-last-defender line (opt-in), so the
   defender sets it.
3. Consequence of an offside call in a phase-2 episode: end the trial with
   a new terminal outcome `offside` (not a free kick), with its own
   reward-table row. User's call (2026-11-20): the DEFENDER gets a bit MORE
   for an offside than for a timeout (his positioning can cause it -- he
   sets the line); attacker negative (a wasted attack, like losing the
   ball); keeper small positive. Exact numbers to agree.
4. **General rule, not a scenario special** (user, 2026-11-20): offside is
   enabled for EVERY player in every scenario / phase (Match-level, on by
   default once correct; phase-1 1v1 simply never triggers it -- no
   teammate), with all the relevant law implemented correctly and tested:
   second-last defender (the keeper counts as a defender), level = onside,
   never offside in his own half, only when the ball is played by a
   teammate, judged at the moment it is played, ball/passer as the line
   when beyond the defenders. Restarts that exempt offside in the real law
   (goal kick, throw-in, corner) only matter once those restarts exist --
   note them as exemptions when they do.
5. Law details (user's calls, 2026-11-20):
   - **Only a PASS can trigger offside** -- the ball played by one attacker
     to a teammate. An attacker kicking the ball ahead to himself
     (dribbling / knock-on) or shooting never makes HIM offside.
   - **The offside line includes the ball / passer**: if the attacker
     playing the ball is himself beyond the last defender, the line is the
     ball's position -- a teammate level with or behind the ball when it is
     played is onside (already true in is_offside_position's "beyond the
     ball carrier" check; keep it, and make the line max(second-last
     defender, ball) in the attacking direction).
   - Judged at the moment the pass is played (not on receipt); level =
     onside; never offside in his own half (already).

## Prerequisite: action-head refactor (spin on + bounded scalar heads)

Decided 2026-11-22 (user): enable kick spin BEFORE the passing work, and do
it as one refactor together with the bounded-scalar action heads, so the
new head type and its checkpoint migration are built once (passing will add
its own power-like head).

**Why.** `kick_power` is the only bounded, sampled continuous head today: a
sigmoid-squashed Gaussian on [floor, 1] (`SquashedNormalHead`, one global
`kick_power_log_std`). Measured 2026-11-22: the phase-2 teacher's raw power
mean on kick rows is about -11, deep in the sigmoid's flat tail. The raw-space
KL is mathematically exact (KL is invariant under the sigmoid), but it
spends about 0.31 KL per kick row there on an actual power difference of
0.009. And in PPO a saturated mean gets almost no gradient on actual power.
`spin_implementation_plan.md` §2.3 would add a second head of the same
kind (spin magnitude), plus a normalised spin axis.

**Candidate: Beta distribution for bounded scalars** (Chou et al. 2017, the
reference fix for boundary bias). Applies to kick power, spin magnitude and
any pass power. Direction heads stay von Mises. Region centres are mean-only
and unaffected.
- Pros: native support on [0, 1] (affine to [floor, 1]); no squash or
  Jacobian; log-prob / entropy / KL in action space. With alpha = 1 +
  softplus(a), beta = 1 + softplus(b) from the network, no saturation: a mean
  near a bound needs a large alpha/beta ratio, and softplus is linear there.
- Cons:
  - The spread becomes state-dependent, so we lose the single global
    log_std knob and the distill refit's frozen-spread trick. The variant
    with mean = sigmoid(raw) and a global concentration brings the
    saturation back.
  - Near-deterministic targets (always full power) push alpha/beta very
    large, and the log-prob diverges at exactly 0 / 1, so actions must be
    clamped to (eps, 1 - eps).
  - No exact policy-preserving migration from the squashed Gaussian, only
    moment matching. The KL between a Beta student and a squashed-Gaussian
    teacher has no closed form (sample-based KL in the migration refit).
- Cheaper alternative if the Beta proves awkward: keep the squashed Gaussian
  and add a penalty on |raw mean| beyond the useful range (the same idea as
  the direction-norm regulariser).

**What changes:**
- a new head class in `action/distributions.py`, and `_kick_power_head` /
  spin heads in `ppo/action_sampling.py`;
- log-prob / entropy / KL paths (`ppg._ppg_kl_penalty`, the per-head
  entropy weights such as `ent_kick_power_only_weight`, the PPO update);
- the BC power loss (BCE on the mean can stay);
- the apply path (`to_physical`, `kick_power_floor`);
- a checkpoint migration script plus a distill refit to migrate;
- tests.

The spin axis is a normalised vector, so it needs the direction-norm
regulariser and a rescale like move / kick direction (`bc.direction_mag_reg_coef`,
`scripts/rescale_direction_heads.py`). Lesson from 2026-11-22: 0.001 did not
hold the norm, 0.1 is in use.

**Related, PPO-time:** `kick_power_log_std` grew 0.25 -> 0.425 during the
phase-2 runs under `ent_kick_power_only_weight` 12 (763, the phase-1
teacher, has 0.10). The weight is to be lowered at the next PPO launch;
see the power-sigma eval sweep of 2026-11-22 in
agent_plans/overnight_2026-11-20.md.

## Rough order

0. Action-head refactor above: spin physics on per
   `spin_implementation_plan.md` (its §0.4 re-enable checklist), bounded
   scalar heads (Beta, or squashed Gaussian plus a mean penalty) for kick
   power / spin magnitude, migration refit.
1. Offside: wire into `Match` (at kick time), second-last-defender line,
   `offside` terminal outcome + reward rows; tests (incl. level / own half /
   behind the ball).
2. Scenario option for attacker B (+ onside/offside start split) in
   `build_1v2_scenario`; obs already handles extra players (21 slots).
3. Rules attacker A pass timing + B run/shoot; check rules-vs-rules
   outcomes (pass rate, offside rate, goals).
4. New role for B? (attacker B shares task id / attacker reward table, or a
   new role name -- decide.) Controllers config, BC labels.
5. Distill refit with role BC on the attackers' passing, then PPO.
