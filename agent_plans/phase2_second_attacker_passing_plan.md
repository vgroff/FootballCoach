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

## Rough order

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
