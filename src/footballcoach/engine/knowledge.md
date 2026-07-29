# engine/

The simulation itself. Each module is a (mostly) pure-function library
operating on `entities/` objects; `match.py` ties them together into a
steppable `Match` with a fixed timestep. All constants come from
`config/physics.json` via each module's `*Params.from_config()`.

## Tick order (`Match.step()`)

```
1. _update_state_timers   - regen stamina for ACTIVE players; count down
                             INACTIVE_TACKLED / CONTROLLING_BALL timers,
                             completing control (granting possession) when
                             a CONTROLLING_BALL timer expires
2. _process_orders        - execute each player's current order (Move/Kick/
                             Shoot/Tackle/Pass/ChaseTackle/Save) for this tick
3. _sync_possessed_ball   - snap the ball to whoever currently has
                             possession (it's "stuck to them" per Idea.md)
4. step_ball (if loose)   - advance free-flight physics for a loose ball,
                             UNLESS a player is currently CONTROLLING_BALL
5. _update_loose_ball_pickup - check if any ACTIVE player is close enough to
                             a loose ball to start a control-time countdown;
                             freezes the ball's velocity the instant contact
                             is made
6. release-grace countdown - decrement ball.release_grace_s if active
7. resolve_all_overlaps   - push apart any overlapping players
8. goal_linger countdown / _check_goal - if a goal linger is active, count
                             it down and call _reset_after_goal() when it
                             expires; otherwise detect goals normally
```

**Ordering subtlety #1 - kick-then-pickup:** step 4 (advance loose-ball
physics) runs *before* step 5 (check pickup), not after. If a player kicks
the ball in step 2, the ball is released at the kicker's feet with a new
velocity. If we checked pickup before moving the ball, the kicker would
find the ball at distance 0 from himself and immediately start
"controlling" it again, silently cancelling every kick. Moving physics
first lets the ball travel away from the kicker within the same tick before
pickup-eligibility is checked.

**Ordering subtlety #2 - release grace period:** even with subtlety #1
handled, a *slow* pass/kick (a few m/s - realistic for a short pass; see
`kicking.pass_speed_mps`) doesn't necessarily clear the passer's own
`pickup_radius_m` (0.4m) within a single 1/30s tick. Without a fix, the
passer would still instantly re-acquire their own pass a few ticks later.
`Ball.last_released_by` / `Ball.release_grace_s` (set by
`Match._start_release_grace`, called right after `kick_ball`/`pass_ball` in
`_process_orders`) exclude the releasing player specifically from
`_update_loose_ball_pickup` for `release_grace_duration_s` (0.3s by
default) - long enough for even a minimum-pace pass to put a couple of
metres between itself and the passer, but short enough not to meaningfully
delay a teammate's *own* attempt to receive a fast return pass.

**Ordering subtlety #3 - the ball freezes on contact, not on control
completion:** `_update_loose_ball_pickup` sets `ball.velocity = Vector3.zero()`
the *instant* a player is close enough to begin the control-time countdown
- not when that countdown finishes. This was a real bug fix: previously the
ball kept flying at full speed for the *entire* control-time window (which
can be several tenths of a second for a fast/high/awkward ball), so a hard
shot could sail straight through a goalkeeper who was technically
"catching" it and still cross the goal line before the timer completed,
making saves against anything but a slow ball essentially impossible. The
ball's height/position is preserved when frozen (important for high balls
caught mid-air) - only velocity is zeroed. Correspondingly,
`Match.step()` also skips `step_ball()` entirely while any player is in
`CONTROLLING_BALL` state (`_any_player_controlling_ball()`), not just while
the ball is possessed - a frozen ball must stay frozen for the whole
control-time window, not just for the first tick.

If you reorder `Match.step()`, keep all three of the above dependencies in
mind.

## `movement.py` - movement, stamina, turning

- **Velocity invariant**: `step_player_towards` is the **only** function
  permitted to write `player.velocity`.  All callers pass a `SpeedMode` enum
  value (`SPRINT`, `JOG`, or `STANDSTILL`); the function owns all kinematics.
  A near-zero snap (`_STOP_SNAP_THRESHOLD_MPS = 0.02 m/s`, applied only in
  `STANDSTILL` mode) clears floating-point drift when the target is 0 — this
  is the only velocity snap in the engine.  `STANDSTILL` additionally uses
  a `standstill_decel_multiplier` (1.5× by default, config-tunable) on top
  of `a_max` so stopping is snappier than accelerating from rest.

- **Goalkeeper movement boosts**: goalkeepers get two flat multipliers from
  `physics.json` applied on top of their attribute-driven movement:
  - `goalkeeper_accel_multiplier` (currently 3.1 in physics.json): applied
    to straight-line acceleration (`effective_acceleration`), lateral/turning
    acceleration (`lateral_accel_capability`), and thus turn rate
    (`max_turn_rate_rad_s`). **Must be applied to all three** - an earlier
    version only boosted straight-line acceleration, which caused overshoot
    and oscillation as the keeper built speed faster than they could correct
    direction when the predicted crossing point shifted each tick.
  - `goalkeeper_speed_multiplier` (currently 1.45): applied to top speed
    in `effective_top_speed`. Effective GK top speed is therefore
    `(5.0 + 4.5*attr) * 1.45`, ranging from ~7.25 m/s (attr=0) to
    ~13.8 m/s (attr=1.0). Simulates explosive diving reach. Applied after
    stamina/ball-carry penalties so it stacks multiplicatively.
  Both multipliers are applied automatically via `player.is_goalkeeper`
  inside `step_player_towards`, so no call-site changes are needed for new
  order types.
- **SaveOrder snap threshold**: the arrival check uses
  `max(0.15, gk_top_speed * dt)` rather than a fixed 0.15m.  Without the
  per-tick term a fast GK (speed 13+ m/s at 30 Hz \u2248 0.46 m/tick) sails
  through the fixed 0.15m window and overshoots the target every tick,
  meaning a fast keeper saved *less* than a slow one.  The dynamic threshold
  ensures the keeper snaps to the intercept point rather than oscillating
  past it.  See `tests/scenario/test_save_order.py` for regression tests
  covering overshoot, tunneling, and drift.

- **Top speed / acceleration**: linear in the attribute,
  `v_max = 5.0 + 4.5*top_speed` m/s (5.0-9.5 m/s), `a_max = 2.5 + 5.0*accel`
  m/s² (2.5-7.5 m/s²). 9.5 m/s is a very fast but real football sprint
  speed; 5.0 m/s is a brisk jog, deliberately never "slow" per Idea.md's
  "League Three should still be competent" requirement.
- **Stamina multiplier**: `1 - 0.65*(1 - stamina_fraction)`, i.e. exactly
  the "reduces speed/acceleration by up to 65%" from Idea.md. At full
  stamina the multiplier is 1 (no penalty); at 0 stamina it's 0.35.
- **Ball-carry speed multiplier**: `0.75 + 0.22*ball_control`, capped at
  0.97 even at `ball_control=1.0` - Idea.md explicitly requires dribbling to
  never be as fast as running free, "even at 1.0 dribbling".
- **Turning**: modelled via a max *lateral acceleration* budget
  `a_lat = 4.0 + 4.0*accel_attr` m/s², from which a max turn rate is derived
  as `omega_max = a_lat / max(speed, min_speed)`. This is the key modelling
  choice: turn rate is *inversely proportional to current speed*, matching
  the real-world fact that sharp turns are much more costly at high speed
  than when jogging or standing still (a stationary player can pivot almost
  freely). Carrying the ball scales `a_lat` down further unless
  `ball_control` is high (`a_lat *= 1 - 0.6*(1-ball_control)`, so at
  `ball_control=1.0` there's no turning penalty at all, matching the same
  "at 1.0 it's the same as without the ball" rule Idea.md states for the
  speed penalty). A large heading change also caps the *target* speed for
  that tick (`turn_speed_penalty`), so you can't "moonwalk" instantly from
  full speed forward to full speed sideways.
- **Stamina drain/regen**: `drain_rate = base * (1.6 - 1.2*stamina_attr)`,
  `regen_rate = base * (0.6 + 0.8*stamina_attr)`. Chosen so a continuous
  sprint drains a mid-attribute (0.5) player to near-zero over roughly
  60-90 seconds - fast enough to matter within a single passage of play,
  slow enough that short sprints aren't punished. See
  `tests/balance/test_stamina_balance.py` for the actual measured curve;
  tune `stamina_drain_sprint_base_per_s` / `stamina_regen_idle_base_per_s`
  in `physics.json` if this feels too fast/slow.

## `ball_physics.py` - free-flight ball physics

Standard projectile physics with drag and Magnus effect, all in SI units:

- **Gravity**: `F = -m*g`, straightforward.
- **Drag**: `F = -0.5 * rho_air * C_d * A * |v| * v` (quadratic drag, the
  standard model for a sphere at football speeds/Reynolds numbers).
  `C_d = 0.25` is a commonly-cited value for a football.
- **Magnus effect**: `F = rho_air * A * r_ball * C_L * (omega x v)`. This is
  a simplified/tuned proxy for the real (much messier) aerodynamic Magnus
  force, per Idea.md's explicit instruction to "approximate" this rather
  than model it exactly. `C_L = 0.25` was chosen to produce a visible but
  not absurd curve on a hard, heavily-spun shot over ~20-30m - tune if shots
  curve too much or too little.
- **Ground bounce**: vertical restitution `e_v = 0.6`, horizontal retention
  `e_h = 0.8` (bounces lose more vertical energy than horizontal, matching
  real ball behaviour), spin halves on each bounce. `e_v = 0.6` is in the
  right ballpark for a football's official rebound-height spec (a ball
  dropped from 2m should rebound roughly 1.0-1.5m, and
  `sqrt(rebound_height/drop_height) ≈ 0.6` for the middle of that range).
- **Rolling friction**: once the ball is settled on the ground
  (`|v_z| < 0.05`), horizontal speed decays via `a = mu_roll * g`,
  `mu_roll = 0.06`, chosen so a ball rolled at ~5 m/s travels roughly 20m
  before stopping - a plausible distance for a firm pass along real grass.
- **Possessed ball**: `step_ball()` is a no-op if `ball.possessed_by` is
  set - `Match._sync_possessed_ball()` handles that ball's motion instead
  (it's glued to the carrying player, not simulated freely).

**Bug fix - "bounce" vs "resting contact" (BOUNCE_THRESHOLD_MPS):** the
ground-collision code distinguishes a genuine bounce from ordinary
resting/rolling ground contact by checking whether the incoming vertical
velocity exceeds `BOUNCE_THRESHOLD_MPS` (0.5 m/s), applying restitution
(which scales horizontal velocity too) only in the former case. This
matters because gravity's per-tick integration nudges a *resting* ball's
next-tick z-position slightly below `ball_radius_m` every single tick,
giving `new_velocity.z` a small negative value purely as an artefact of
that integration - not a real bounce. An earlier version of this code
treated *any* negative `new_velocity.z` while grounded as a full bounce,
applying `bounce_restitution_horizontal` (0.8) to horizontal speed on
*every* tick (~30x/second) instead of the intended, much gentler
`rolling_friction_coefficient` - this decayed any grounded/rolling ball's
speed almost instantly (a ball passed at 5 m/s would stop within about a
metre instead of the intended ~20m). See
`tests/unit/test_ball_physics.py`'s
`test_rolling_ball_decelerates_at_the_analytically_correct_rate` and
`test_rolling_ball_travels_plausible_distance_before_stopping` for
regression tests against this. If you ever see a rolling ball stopping
suspiciously fast again, check this threshold first.

## `kicking.py` - power, direction, spin, and error

**This module takes an absolute 3D `aim_point`, not a direction vector.**
The kicker's intended target (e.g. "1.1m high, dead centre of the goal, 11m
away") is a real point in space; the module *solves* the launch angle
needed to actually reach that point under gravity (see
`solve_launch_pitch_rad`), rather than trusting a hand-picked direction
vector's z-component. This is important: a naive "aim a bit upward" vector
does not reliably arrive at a specific height at a specific distance once
gravity is accounted for, and an earlier version of this code that used raw
direction vectors was breaking penalty-scoring balance tests (shots dropped
into the ground metres short of goal) until this was fixed.

`solve_launch_pitch_rad(horizontal_distance, height_diff, speed, gravity)`
solves the classic projectile range equation
`R = v² sin(2*theta) / g` (extended for a height difference) for `theta`, as
a quadratic in `tan(theta)`. Of the two real roots (a flat, fast trajectory
and a high, lofted one), the flatter is chosen - matching how a real player
drives a ball rather than looping it, unless the target is literally
unreachable at the given speed (falls back to a direct straight-line angle,
which only happens for very weak kicks over long distances).

Error model:
- `sigma_angle(precision) = 0.0107 + 0.0893*(1 - precision)` radians,
  applied independently to both yaw and pitch after the ballistic solve.
  Never zero even at `precision = 1.0` (0.0107 rad ≈ 0.6°), per Idea.md's
  "should never be exact even with 1.0 precision".
  - These constants were derived by working backwards from the penalty
    balance targets in Idea.md (see `tests/balance/test_penalty_balance.py`)
    - i.e. picking sigma such that a 0.5-precision player scores ~50-80% at
    a tight corner and >95% aiming centrally, etc. If you change pitch
    dimensions or add new balance targets, expect to re-tune these two
    constants and re-check the balance tests.
- **Kick power**: `v_ball_max = 15 + 20*kick_power_attr` m/s (15-35 m/s,
  i.e. roughly 54-126 km/h, spanning a firm pass to an elite strike).
  Actual launch speed is `v_ball_max * power_fraction` (the order's
  requested power, 0-1).
- **First-time-shot difficulty**: when a player shoots directly off a
  difficult ball (see `possession.py`'s `compute_difficulty`) instead of
  first controlling it, `firsttime_difficulty_multiplier` inflates
  `sigma_angle` further, less so for high `kick_precision`
  (`1 + (1 - 0.8*precision)*difficulty`).
- Per the `rng_reduction` game option, `sigma` is scaled by
  `reduced_sigma()` before being applied - at `rng_reduction=1.0`, `sigma=0`
  and kicks land exactly on the ballistic solution (used by scenario tests
  for deterministic pass/fail).

### `pass_ball` - dedicated "Pass" action

A grounded pass to a target position is a different technical skill from
curling a shot into a corner, so it gets its own, more forgiving error
model (`PassingParams` in `config/physics.json["passing"]`) rather than
reusing `KickingParams`:

- `sigma_angle(precision) = 0.009 + 0.014*(1-precision)` rad - much tighter
  than the shooting model's `0.0107 + 0.0893*(1-precision)`, reflecting that
  rolling a ball along the ground to a nearby teammate is inherently easier
  to be accurate at than curling a shot into a specific corner from range.
  These constants were tuned against the user's explicit pass-accuracy
  targets (see `tests/balance/test_pass_balance.py`): >80% success at 10m
  for any player (~99% for a precision-0.9 player), >50% at 30m (~90% for
  precision-0.9).
- **Auto-paced power** (`pass_speed_mps`): rather than the caller manually
  picking a `power_fraction` like a shot, a pass's pace is auto-computed
  from the target distance, modelled as the initial speed a rolling ball
  under rolling friction alone would need to *just* reach that distance:
  `v = sqrt(2 * mu_roll * g * distance)`. Since the full ball physics also
  applies (smaller, but non-negligible) aerodynamic drag on top of rolling
  friction, this formula alone undershoots - a tunable
  `power_overshoot_factor` (1.35) compensates, found by matching the
  auto-paced ball's actual arrival behaviour against the distance targets
  above. A caller can still override with an explicit `power_fraction` (as
  `orders.PassOrder` allows) if manual power control is wanted instead.
- Shares the same ballistic-aim machinery as `kick_ball` via the extracted
  `_launch_ball` helper (both solve a launch angle via
  `solve_launch_pitch_rad`, then perturb yaw/pitch by Gaussian noise) - a
  pass just always aims at `target_position.with_z(0)` (ground level) at
  its auto-computed (or overridden) pace.

## `possession.py` - first-touch control-time model

Models how long it takes a player to bring a loose ball under control on
first touch, as a function of the ball's height, the relative velocity
between ball and player, the player's own speed, and their `ball_control`
attribute. There was no explicit numeric target for this from the design
brief (unlike penalties/tackles) - the formula was derived to be
*intuitively* football-realistic, then validated with monotonicity/ordering
balance tests (see `tests/balance/test_control_time_balance.py`) rather than
hard percentage targets.

- **Height difficulty** (`height_difficulty_factor`): a piecewise, quadratic
  ramp with control points anchored at real body landmarks:
  - flat at `f=1.0` for anything at or below knee height (0.49m) - a
    rolling or bouncing low ball is roughly as easy to control regardless
    of exact height.
  - rises quadratically to `f=2.0` at waist height (0.95m) - a knee-to-waist
    ball requires real technique, and the *quadratic* shape (rather than
    linear) means it stays easy near the knee and gets meaningfully harder
    only as it approaches the waist.
  - rises quadratically again to `f=4.0` at head height (player height,
    1.8m) - chest/head control is a distinctly harder skill.
  - beyond head height, `f` increases linearly and is capped at 6.0 (very
    high balls are hard to control but not infinitely so - a player will
    still eventually bring it down).
- **Velocity difficulty**: linear penalties,
  `k1 * relative_speed + k2 * player_own_speed`, with
  `k1 = 0.15 s/m` and `k2 = 0.05 s/m` - the relative speed between ball and
  player (how "hot" the pass/shot arrives) is weighted 3x more than the
  player's own running speed, since receiving a fast, driven ball is a
  bigger technical challenge than merely jogging while the ball trickles
  towards you.
- **Ball control scaling**: `extra = (1 - 0.85*ball_control) * difficulty`
  - even at `ball_control = 1.0`, only 85% of the difficulty is negated
    (never fully to zero), consistent with the "never perfect" philosophy
    applied everywhere else in this codebase (kicks, tackles).
- **Final formula**: `t_control = t_base + t_scale * extra`, with
  `t_base = 0.1s` (an irreducible minimum reaction/first-touch time) and
  `t_scale = 0.3s`.
- **Goalkeeper-in-box special case**: per Idea.md, goalkeepers in their own
  box control the ball much more easily (they can use their hands) - modeled
  with a lower `t_base_gk = 0.08s`, a height-factor scaled down to 40% of
  its normal effect (`gk_height_factor_scale = 0.4`), and a higher
  `ball_control_alpha = 0.9`. Outside the box, goalkeepers use the normal
  outfield-player formula (checked via `pitch.is_in_either_box()` in
  `Match._update_loose_ball_pickup`).
- **Jump penalty (GK and outfield)**: above head height (1.8m), control
  time increases faster than the base height-difficulty curve — modeled as
  a continuous scaling of the height-factor term that ramps up toward each
  player type's maximum reach height. GK max reach is higher than outfield
  and the per-metre penalty is lower (GK advantage). Below head height,
  outfield players are completely unaffected by this extension (regression
  safe). Config keys in `physics.json["control_time"]`.
- A small proportional Gaussian noise term (`noise_sigma_fraction = 0.1`,
  scaled by `rng_reduction`) is added on top of the deterministic
  `t_control` in `Match._update_loose_ball_pickup`, so touches aren't
  perfectly predictable even for a fixed situation - this wasn't explicitly
  requested in Idea.md but was added to stay consistent with the rest of
  the game's "nothing is ever perfectly deterministic" philosophy. Flag to
  the project owner if this should be removed/reconsidered.

## `tackling.py` - tackle skill checks

A single RNG skill check per Idea.md's spec:
`tackler_roll = skill_roll(tackling_attr * 1.2, rng_reduction)` vs
`dribbler_roll = skill_roll(dribbling_attr, rng_reduction)`, tackler wins if
`tackler_roll > dribbler_roll`. The `1.2` tackler boost is Idea.md's
explicit instruction ("tackling always gets a 20% boost... to favour the
defender... they can then roll higher than 1"). Analytically, at the
default `rng_reduction=0.3`, tackling=0.8 vs dribbling=0.6 gives the tackler
a ~82.5% win rate, matching the design target of 70-90%
(`tests/balance/test_tackling_balance.py` verifies this empirically over
5000 trials and also reports a full win-rate grid across many attribute
pairs for balance inspection).

On a successful tackle in `Match._process_orders`, the tackled player
enters `PlayerState.INACTIVE_TACKLED` for `inactive_duration_s` (0.6s by
default), during which they can't tackle and (per Idea.md) should have
reduced speed - the reduced-speed-while-inactive multiplier
(`inactive_speed_penalty`) is defined in config but not yet wired into
`movement.py`'s speed calculation; this is a known gap to close before this
mechanic is fully complete (see "Known gaps" below).

A **failed** tackle attempt also briefly incapacitates the *tackler* (not
just a successful one dispossessing the victim): `player.state` is set to
`INACTIVE_TACKLED` for `tackler_miss_inactive_duration_s` (shorter than the
victim's `inactive_duration_s`, since a mistimed lunge leaves you
momentarily off-balance but not as badly as actually being dispossessed).
Applies in the `ChaseTackleOrder` branch of `Match._process_orders`.

### Phase B tackle modifiers (added after initial implementation)

Two additional modifiers are applied at the `attempt_tackle()` call sites in
`Match._process_orders` — not inside `attempt_tackle()` itself, keeping the
function signature clean:

**1. GK outside-box penalty** (`gk_outside_box: bool = False`):
- If the tackler is a goalkeeper and is **outside** their own penalty box,
  their effective `tackling_attr` is multiplied by
  `(1 - goalkeeper_outside_box_tackle_penalty)` (currently `0.4` → 40%
  penalty, i.e. GK tackles at 60% effectiveness outside the box).
- Convention: `Team.LEFT` GK defends the box at the left end of the pitch
  (x ≤ `pitch.left_box_max_x`); `Team.RIGHT` GK defends the right end.
- Call sites check `player.is_goalkeeper and not pitch.is_in_own_box(player)`.
- Config: `physics.json["tackling"]["goalkeeper_outside_box_tackle_penalty"]`.
- Balance test: `tests/balance/test_gk_tackle_balance.py`.

**2. GK in own box with ball — untackleable**:
- If the *target* is a goalkeeper currently in their own box with possession,
  the tackle attempt is skipped entirely (returns early at all four call
  sites: `ChaseTackleOrder`, `GetPossessionOrder`, and
  `_check_head_on_tackles`). This models the goalkeeper's protected status
  inside the box.

**3. CONTROLLING_BALL dribble penalty**:
- If the *target* is in `PlayerState.CONTROLLING_BALL` (mid first-touch),
  their effective `dribbling_attr` is penalised based on how long they've
  been in that state: `penalty_frac = min(1.0, state_timer_s /
  control_time_penalty_reference_s)`, effective dribbling =
  `dribbling_attr * (1 - 0.25 * penalty_frac)`.
- Config: `physics.json["tackling"]["control_time_penalty_reference_s"]`
  (currently `0.3s`).
- If the tackler wins against a `CONTROLLING_BALL` target, the tackler is
  given the ball (not just the target losing possession).

### `ChaseTackleOrder` - the "Tackle" high-level action

`ChaseTackleOrder` persists across ticks: the
tackler runs straight at the target's *current* position (re-aiming every
tick, so a moving target is actually chased, not just run at their
starting spot) until `are_touching()`, at which point exactly one tackle
attempt is resolved and the order completes. This is what `actions.tackle()`
issues. See `tests/balance/test_tackle_action_balance.py` for the
end-to-end (chase + tackle) balance validation, as opposed to
`tests/balance/test_tackling_balance.py` which tests the underlying
`attempt_tackle()` skill check in isolation.

## `goalkeeping.py` - the "Save" action

Implements the goalkeeper-only "Save" behaviour from the design brief:
"calculate where the shot is going to cross the goal line and run there".

- **`predict_goal_line_crossing`**: given the ball's current
  position/velocity, solves for where it will cross a given x-plane under
  gravity alone (`z = z0 + vz*t - 0.5*g*t²`, `y = y0 + vy*t`, with
  `t = dx/vx`). Like `kicking.solve_launch_pitch_rad`'s aiming solve, this
  deliberately ignores drag and Magnus for the *prediction* - the keeper
  "reads" the shot with straightforward physics judgement, so heavily
  curved/backspun shots are naturally (and only slightly) harder to predict
  correctly, which is a reasonable, tunable source of difficulty rather
  than a limitation to fix. Returns `None` if the ball isn't currently
  moving towards that plane (stationary, or moving away) - i.e. there's
  nothing meaningful to react to yet.
- **`save_target_position`**: the keeper's actual movement target for this
  tick. If a shot is heading toward their own goal, it's the predicted
  crossing point, clamped to the goal frame (`+/- goal_width/2`,
  `[0, goal_height]`) plus a small margin (`goal_frame_margin_m`). If no
  shot is incoming, defaults to a sensible standing position just off the
  goal centre (`default_position_fraction_of_half_length`).
  - **Important: the target plane is placed *in front of* the true goal
    line, not on it** (`target_plane_x = goal_x + sign * goal_frame_margin_m`).
    This is needed because of the ball-freeze-on-contact behaviour in
    `Match` (see the tick-order notes above): if the keeper's target plane
    were the true goal line itself, a fast shot could cross that exact line
    in the same tick the keeper begins their pickup/control-time countdown,
    turning a well-read save into "too little, too late" - the ball would
    already be past the goal line (and thus already a goal, per
    `scoring.check_goal`) by the time contact registers. Targeting a plane
    slightly in front of the line gives the keeper a chance to make contact
    *before* the ball would have crossed.
- **Bug fix - snap to target on arrival, don't just freeze velocity:**
  `Match._process_orders`'s `SaveOrder` branch, once the keeper is within a
  small dead-zone (0.15m) of `target_position`, sets
  `player.position = target_position` (not just `player.velocity = zero`).
  An earlier version only zeroed velocity, leaving the keeper's actual
  position wherever they happened to be inside the dead-zone. This was
  fine for a normal-speed keeper (who typically "arrives" right around when
  the predicted crossing point stabilizes), but once goalkeepers got the
  `goalkeeper_accel_multiplier` diving boost (see `movement.py` above), a
  fast keeper would reach and freeze near the dead-zone well *before* the
  target position finished shifting - leaving a residual gap that could
  push the keeper's final resting spot just outside `pickup_radius_m`,
  turning what should have been an easy save into a miss. Snapping position
  (not just velocity) to the target on arrival closes that gap. See
  `tests/balance/test_save_balance.py` for the regression this fixed.
- `own_goal_x` mirrors the attacking-direction convention from
  `offside.py` (`Team.LEFT` attacks +x and defends the goal at -x, and vice
  versa) - keep these two modules' conventions in sync if either changes.
- Good vs bad goalkeepers are differentiated purely through the existing
  movement (`top_speed`/`acceleration`) and control-time (`ball_control`,
  plus the existing GK-in-box control-time bonus from `possession.py`)
  attributes - there's no goalkeeping-specific attribute. A fast keeper
  reaches the predicted crossing point in time; a good-ball-control keeper
  is less likely to spill/fumble it once there. See
  `tests/balance/test_save_balance.py` for validation that save rate
  responds sensibly to both.
- **Early intercept**: `early_intercept_target()` computes a candidate
  intercept point along the ball's flight path and moves the GK there
  instead of waiting for the goal-line crossing point, when the GK can
  reach it faster. Falls back to the goal-line target if the ball is too
  far or the intercept wouldn't save meaningful time. Config keys are in
  `physics.json["goalkeeping"]`.

## `offside.py` - simplified offside rule

Deliberately simplified per Idea.md's explicit instruction to capture "the
spirit" of offside rather than the full law:

- The offside line is the **single deepest defender** (`last_defender_x`),
  which includes the goalkeeper - this is a simplification of the real law
  (which uses the *second*-deepest outfield defender), chosen because it's
  much simpler to compute and still captures the core "don't camp in behind
  the last man" intent.
- A teammate is offside if they are simultaneously *beyond* that last
  defender **and** *beyond* the ball carrier **and** *beyond the halfway
  line*, in the attacking direction. The halfway-line condition means a
  player can never be offside in their own half, matching the real law -
  without it, a defender pushed high up their own half could be flagged
  offside against a long ball, which isn't how the actual rule works. Per
  Idea.md: any attacker meeting all three conditions is flagged "whether or
  not the ball was actually intended for them" - i.e. this function should
  be called for any attacking teammate near the ball's path, not just the
  pass's intended receiver.
- `offside.enabled_by_default` in `physics.json` exists so training
  scenarios can disable the rule entirely, per Idea.md's requirement, though
  wiring that flag into `Match`'s pass-handling isn't done yet (there is no
  pass-completion/whistle logic in `Match` yet at all - see "Known gaps").

## `collision.py` - player-player overlap resolution

Players are circles (radius 0.3m) viewed from above. Per Idea.md: "the
distance between the centre of the circles of any 2 players needs to be
[at least the sum of their radii]... if this distance is violated, both
[players] are re-adjusted... weighted by velocity... which enables players
to push each other while running alongside."

`resolve_player_overlap` implements exactly this: on overlap, both players
are pushed apart along the line connecting their centres, with the push
*weighted by each player's velocity component along that axis* - a player
charging hard into another gets pushed back less than the other, because
the tackler's momentum "wins" more of the separation. There's no explicit
mass system (Idea.md doesn't specify player mass), so velocity magnitude is
used as a direct proxy for how much a player should be able to shove another
- reasonable at football running speeds where players have broadly similar
masses. `resolve_all_overlaps` runs a few iterations over all pairs so
chains of 3+ overlapping players settle towards a mutually-valid
configuration.

**Inactive players are excluded from push-apart collision entirely** -
`resolve_all_overlaps` skips any pair where either player's
`Player.is_inactive` (true while `PlayerState.INACTIVE_TACKLED`) is true, so
active players can run straight through a just-tackled player lying/off-
balance on the ground rather than bumping into them like a solid obstacle.

**Velocity damping on collision**: when two overlapping players have a
  closing velocity above a minimum floor, the component of each player's
  velocity directed toward the other is damped. The floor prevents
  continuous damping of gentle jostling; the retention factor and floor are
  tunable in `physics.json["collision"]`. Damping applies even to inactive
  pairs (unlike position push-apart) so a just-tackled player coasting at
  full speed still slows on contact. Because overlap can persist across
  multiple ticks, the damping compounds — this is intentional but the floor
  prevents it from driving velocity to zero.
  inactive player's cylinder from *outside* it - a ball already inside the
  cylinder (e.g. one that was there when the player became inactive) does
  NOT get blocked, only a ball crossing in from outside, per the explicit
  design spec ("they can still block the ball... if it is shot from outside
  their cylinder and crosses in (but not if it is shot from inside the
  cylinder)"). Uses a ray-circle intersection test in the XY plane; on a
  block, the ball is stopped at the entry point and its velocity is damped/
  reflected using `block_restitution` (0.35 - a fairly dead deflection,
  since an inactive player is an unintentional obstacle, not an active
  block/save). Wired into `Match.step()` right after `step_ball()`, using
  the ball's pre-flight position captured before that call.

## `scoring.py` - goals

Thin wrapper: `check_goal(ball, pitch)` returns `"left"`/`"right"`/`None`
based on `Pitch.is_goal()`; `Scoreboard.score_for(side)` increments the
*opposing* team's tally (a ball entering the left goal is a goal *for* the
right-side team, matching real football's goal-naming convention). `Match`
calls this every tick and resets the ball to the centre spot on a goal
(no kickoff-formation/whistle logic yet - see "Known gaps").

## `match.py` - the top-level `Match`

Owns the list of players, the ball, the pitch, the scoreboard, and all the
`*Params` config objects (loaded once at construction, not re-read from
`config/` every tick, for performance). `Match.step()` is the single entry
point a future UI or RL training loop should call in a loop; `Match.paused`
lets a UI freeze the simulation while still allowing orders to be queued
(per Idea.md's "issue orders... while paused" requirement - though note
`step()` currently returns immediately when paused, so orders queued while
paused won't execute until `paused=False` and `step()` is called again;
there's no "single order resolution while otherwise paused" mode yet).

## Known gaps / explicitly deferred (not oversights - flagging for future work)

- `inactive_speed_penalty` (reduced speed while `INACTIVE_TACKLED`) is
  defined in config but not yet applied in `movement.py`.
- `PassOrder` now exists distinct from a generic `KickOrder` (see
  `pass_ball` above), but offside checks (`check_offside_on_pass`) still
  aren't wired into `Match` - there's no possession-change/whistle handling
  for fouls/offside/out-of-bounds in the match loop yet.
- No out-of-bounds (touchline/goal-line-but-not-goal) handling - a ball that
  leaves the pitch currently just keeps flying/rolling under free physics.
- No kickoff/restart formations - `Match._reset_after_goal` just resets the
  ball to the centre spot; players are not repositioned.
- `SaveOrder` never auto-completes (a goalkeeper is always "on duty"); there
  is currently no way to tell a goalkeeper to stop reacting to shots other
  than assigning them a different order.
- These are all reasonable future milestone additions once a UI or training
  loop actually needs them - the current engine is deliberately scoped to
  what the design brief and balance-test suites ask for (movement, kicking,
  passing, tackling, saving, ball physics, offside *detection*, goal
  detection).

## Running-while-kicking power and precision modifiers (`kicking.py`)

Both `kick_ball` and `pass_ball` accept optional `kicker_velocity` /
`kicker_top_speed_mps` parameters, used to apply two independent modifiers:

**Power** (`running_power_multiplier`): cosine projection of the kicker's
velocity onto the aim direction, scaled by fraction of top speed. Running
fully toward the aim direction at top speed adds up to
`running_power_coefficient` (+30%) extra power; running away reduces it
by the same amount. No-ops at zero velocity.

**Precision** (`running_direction_precision_multiplier`): reduces effective
`kick_precision` when kicking against the run direction — no penalty within
a forward cone (~70°), grading to a meaningful penalty at square-on, and a
steeper penalty when kicking backward relative to momentum. Applied to all
kicks (shots, passes, generic kicks) via the shared `_launch_ball` helper,
not shots only. Config breakpoints and penalty magnitudes are in
`physics.json["kicking"]`. Defaults to 1.0 (no penalty) when the kicker is
nearly stationary.

## `MarkOrder` and `GetPossessionOrder` (Phase F)

**`GetPossessionOrder`**: instructs a player to acquire the ball, however
necessary. Each tick: if a carrier exists (and isn't the player themselves),
chase them and attempt a tackle on contact; if the ball is loose, sprint to
the predicted intercept point. Completes once the player has possession.
The chase/tackle logic is factored into `_run_get_possession_behaviour()`
and shared with `MarkOrder`'s fallback.

**`MarkOrder(target_player_id)`**: instructs a player to mark a specific
opponent. Two modes per tick, selected automatically:
- **Intercept/tackle mode**: when the target has ball possession (or is
  `CONTROLLING_BALL`), or the ball is within `mark_intercept_radius_m`
  (config, default 4.0m) of the marker — delegates to
  `_run_get_possession_behaviour()`, identical to `GetPossessionOrder`.
- **Standoff mode**: otherwise, moves to a point between the target and the
  ball at `mark_standoff_m` (config, default 1.5m) offset from the target
  toward the ball, decelerating to a standstill there. Uses
  `_braking_speed_mode()` — no velocity snaps.

`MarkOrder` **never auto-completes** (analogous to `SaveOrder`); it must be
explicitly replaced with a different order. Config:
`physics.json["marking"]`. Balance tests: `tests/balance/test_mark_balance.py`.

**`StopOrder`**: decelerates the player to a standstill using
`SpeedMode.STANDSTILL` and completes once `speed_mps == 0.0` (driven by
the physics-level snap at `_STOP_SNAP_THRESHOLD_MPS`).

## `../actions.py` - high-level one-shot action helpers

`src/footballcoach/actions.py` (one level up from `engine/`) provides the
simple, literally-named functions requested for player control:
`move_to`, `shoot`, `pass_to`, `tackle`, `save`, `mark`. Each is a thin
wrapper that just constructs and assigns the appropriate order — all the
actual behaviour lives in the order types and `Match._process_orders`
described above. `opponent_goal_centre` resolves "which goal is this team
attacking" using the same `Team.LEFT` attacks +x / `Team.RIGHT` attacks -x
convention as `offside.py` and `goalkeeping.py`.
## Goal linger (`Match.goal_linger_s`, Phase G/H)

`Match.goal_linger_s: float = 0.0` — how many sim-seconds the ball stays in
the net after a goal before `_reset_after_goal()` is called. With the
default of 0.0 (headless/test mode), the existing immediate-reset behaviour
is fully preserved. The UI sets this from `physics.json["ui"]["goal_linger_s"]`
(default 3.0 s) when constructing a match.

When a goal is detected by `_check_goal`:
1. `scoreboard.score_for(side)` is called immediately (score updates on the
   detection tick, not when the linger expires).
2. `Match._goal_linger_remaining_s` is set to `goal_linger_s`; `_reset_after_goal`
   is **deferred**.
3. On each subsequent `Match.step()`, `_goal_linger_remaining_s` decrements by
   `dt`. `_check_goal` is skipped entirely during the countdown (no double-
   goal detection while the ball is still sitting in the net). When the
   countdown reaches 0, `_reset_after_goal` runs.

Tests: `tests/unit/test_goal_linger.py` covers immediate-reset regression,
linger duration, no double-goals during linger, and countdown-rate assertion.

## Game log (`Match.log_callback`, Phase G)

`Match.log_callback: Callable[[LogLevel, str], None] | None = None` is an
optional hook the UI attaches to receive real-time narration of match events.
Default is `None` (headless / test use = zero cost, since the helper methods
`_log_info` / `_log_debug` check for `None` before importing `gamelog.py`).

Call sites (all in `match.py`):
- Tackle outcomes (every `attempt_tackle` call site): INFO one-liner + DEBUG
  breakdown with `result.tackler_roll`, `result.dribbler_roll`, and any
  modifiers (GK box penalty, control-time penalty, head-on tag).
- GK-in-box auto-fail short-circuit: distinct INFO message (no roll values).
- Kick/shoot execution: DEBUG for kicks, INFO for shots.
- `_complete_control`: INFO when a player finishes their first-touch.
- Goal: INFO with scorer side and updated scoreboard.

`TackleResult` (in `tackling.py`) was extended with `tackler_roll: float`
and `dribbler_roll: float` fields so the log can report the exact draws
without re-deriving them from internal state.

## `ball_physics.py` — `just_bounced_timer_s` (Phase G)

`step_ball()` sets `ball.just_bounced_timer_s = params.just_bounced_display_duration_s`
(0.3 s, config) on the tick of each **real bounce** (incoming vertical speed
exceeds `BOUNCE_THRESHOLD_MPS` and the outgoing restituted speed also
exceeds it). It decrements by `dt` each tick and floors at 0. The renderer
uses it to draw an amber ring briefly after each bounce; the engine itself
never reads it back.

## AI / engine boundary (relevant when wiring up the AI training loop)

The AI package (`src/footballcoach/ai/`) drives the engine via the standard
`orders.py` / `actions.py` interface — same as the UI.  `ai/action/to_orders.py`
translates neural-network gating output into `MoveOrder`, `KickOrder`,
`PassOrder`, `ChaseTackleOrder`, `GetPossessionOrder`, `MarkOrder` objects
and assigns them to `player.current_order`, then `Match.step()` handles
execution identically to any other caller.

**Illegal-action guardrail audit** (ai_design_doc.md section 11 checklist;
items confirmed vs engine behaviour as of the time the AI package was added):

- `KickOrder`/`ShootOrder`: `Match._process_orders` requires `ball.possessed_by
  == player.player_id` before calling `kick_ball`/`pass_ball`; an AI
  attempting to shoot without possession is safely a no-op at the engine
  level. `to_orders.py` detects this independently and sets `illegal_action=True`
  for the reward function.
- `ChaseTackleOrder`/`GetPossessionOrder`: `attempt_tackle()` call sites
  already check `player.state != INACTIVE_TACKLED` via `player.is_available_to_tackle()`.
  `to_orders.py` additionally refuses to assign these orders to inactive players.
- `PassOrder`: same possession precondition as `KickOrder` (checked in
  `to_orders.py`; engine-level guard is the same `kick_ball` path).
- `SaveOrder`: documented as "goalkeeper-only" in orders.py; the engine
  does not enforce this with a hard guard (it would just chase the ball as
  an outfield player); `to_orders.py` does not currently guard non-GK save
  attempts — add a guard if non-GK save orders prove to be a training issue.
- AI must be punished for illegal attempts AND the engine must be a safe
  no-op — both protections coexist (see design doc 9.7).