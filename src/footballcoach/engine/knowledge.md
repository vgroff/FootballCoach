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
                             Shoot/Tackle/Pass/ChaseTackle/Save) for this tick.
                             CONTROLLING_BALL players are NOT skipped — their
                             order and AI run normally so they keep moving.
                             `_apply_movement` treats them as has_ball=True
                             (ball-carry speed penalty applies during control).
3. _sync_possessed_ball   - snap the ball to whoever currently has
                             possession (it's "stuck to them" per Idea.md)
4. step_ball (if loose)   - advance free-flight physics for a loose ball,
                             UNLESS a player is currently CONTROLLING_BALL
5. _update_loose_ball_pickup - check if any ACTIVE player is close enough to
                             AND closing on (see subtlety #2) a loose ball to
                             start a control-time countdown; freezes the
                             ball's velocity the instant contact is made
6. resolve_all_overlaps   - push apart any overlapping players
7. goal_linger countdown / _check_goal - if a goal linger is active, count
                             it down and call _reset_after_goal() when it
                             expires; otherwise detect goals normally
```

**Ordering subtlety #1 - kick-then-pickup:** step 4 (advance loose-ball
physics) runs *before* step 5 (check pickup), not after. If a player kicks
the ball in step 2, the ball is released at the kicker's feet with a new
velocity. Moving physics first lets the ball travel away from the kicker
within the same tick, so its post-kick velocity (moving away from the
kicker) is what subtlety #2's closing-velocity check actually sees.

**Ordering subtlety #2 - pickup requires closing velocity, not just
proximity:** `_update_loose_ball_pickup` calls `possession.can_pick_up_ball()`,
which requires a player be within `ball_pickup.pickup_radius_m` (0.4m) AND
either (a) closing on the ball — the relative velocity `ball.velocity -
player.velocity` has a component reducing the separation — or (b) the
relative speed is below `ball_pickup.closing_speed_deadzone_mps` (0.3m/s by
default), i.e. ball and player are both roughly stationary relative to each
other so "closing" isn't a meaningful requirement. This replaced an earlier
purely time-based "release grace period" (a short window during which the
releasing player specifically was excluded from pickup) — the old approach
special-cased "who kicked it" rather than modelling the actual physical
fact: a player can't catch a ball that's moving away from them faster than
they're closing on it, regardless of who released it or when. The
closing-velocity rule handles the kicker/passer case for free (their own
kick is moving away from them, so it fails the closing check) without any
per-player bookkeeping, and generalises to any loose-ball scenario
(deflections, rebounds) with the same rule. See `possession.py`'s
`can_pick_up_ball()` docstring and `tests/unit/test_ball_pickup.py`.

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

**Ordering subtlety #4 - pickup swept check covers BOTH entities, and
contention is resolved by current distance, not list order:**
`possession.can_pick_up_ball()` accepts both `ball_pre_tick_position` and
`player_pre_tick_position` and checks the swept path of ball-relative-to-
player, so a fast PLAYER grazing past a slow/stationary ball within one tick
is caught by the same tunneling exception as a fast ball grazing past a slow
player (see `possession._swept_min_separation`) — also correctly handles
both moving simultaneously. `Match.step()` snapshots every player's position
at the very top of the tick, before `_process_orders`/`_apply_movement` can
move anyone, into `pre_tick_player_positions`, and threads it through to
`_update_loose_ball_pickup`. When multiple ACTIVE players are eligible in
the same tick (including via the swept exception), the one with the
smallest CURRENT distance to the ball wins — not the first eligible player
in `self.players` iteration order (a fragile, order-dependent tie-break the
contention logic used to have). A within-radius candidate always beats a
swept-only one, since a swept-only candidate is by definition currently
outside `pickup_radius_m`.

If you reorder `Match.step()`, keep all four of the above dependencies in
mind.

## `movement.py` - movement, stamina, turning

**Numeric constants in this section were found to have drifted from
`config/physics.json["movement"]` (2026-09-05 audit) and were corrected —
treat `physics.json` as authoritative over this prose if they disagree again.**

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
  - `goalkeeper_accel_multiplier` (currently **3.3** in physics.json): applied
    to straight-line acceleration (`effective_acceleration`), lateral/turning
    acceleration (`lateral_accel_capability`), and thus turn rate
    (`max_turn_rate_rad_s`). **Must be applied to all three** - an earlier
    version only boosted straight-line acceleration, which caused overshoot
    and oscillation as the keeper built speed faster than they could correct
    direction when the predicted crossing point shifted each tick.
  - `goalkeeper_speed_multiplier` (currently **1.23**): applied to top speed
    in `effective_top_speed`. Effective GK top speed is therefore
    `(5.4 + 4.7*attr) * 1.23`, ranging from ~6.6 m/s (attr=0) to
    ~12.4 m/s (attr=1.0). Simulates explosive diving reach. Applied after
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
  `v_max = 5.4 + 4.7*top_speed` m/s (5.4-10.1 m/s), `a_max = 3.3 + 2.4*accel`
  m/s² (3.3-5.7 m/s²). ~10 m/s is a very fast but real football sprint
  speed; 5.4 m/s is a brisk jog, deliberately never "slow" per Idea.md's
  "League Three should still be competent" requirement.
- **Stamina multiplier**: `1 - stamina_speed_penalty_max*(1 - stamina_fraction)`,
  `stamina_speed_penalty_max = 0.63`, i.e. roughly the "reduces speed/
  acceleration by up to 65%" from Idea.md. At full stamina the multiplier is
  1 (no penalty); at 0 stamina it's 0.37.
- **Ball-carry speed multiplier**: `ball_carry_speed_mult_base (0.77) +
  ball_carry_speed_mult_scale (0.19) * ball_control`, capped at 0.96 even at
  `ball_control=1.0` - Idea.md explicitly requires dribbling to never be as
  fast as running free, "even at 1.0 dribbling". This is also snapshotted
  once more at the very *start* of a first touch — see `control_speed_
  multiplier` in the `possession.py` section below, a separate, one-time
  cut applied on top of (not instead of) this ongoing multiplier.
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
- **Stamina drain/regen**: `drain_rate = base * lerp(stamina_drain_attr_lo
  (1.6), stamina_drain_attr_hi (1.1), stamina_attr)` = `base * (1.6 -
  0.5*stamina_attr)`, `regen_rate = base * lerp(stamina_regen_attr_lo (0.7),
  stamina_regen_attr_hi (0.9), stamina_attr)` = `base * (0.7 +
  0.2*stamina_attr)`. Chosen so a continuous
  sprint drains a mid-attribute (0.5) player to near-zero over roughly
  60-90 seconds - fast enough to matter within a single passage of play,
  slow enough that short sprints aren't punished. See
  `tests/balance/test_stamina_balance.py` for the actual measured curve;
  tune `stamina_drain_sprint_base_per_s` / `stamina_regen_idle_base_per_s`
  in `physics.json` if this feels too fast/slow.

## `ball_physics.py` - free-flight ball physics

Standard projectile physics with drag and Magnus effect, all in SI units.
**These values live in `config/physics.json`'s `ball_physics` section --
treat that as authoritative over the prose below if they ever disagree; this
section was found to have drifted from the live config once already
(2026-09-03 audit, see `agent_plans/physics_update.md`) and was corrected
then.**

- **Gravity**: `F = -m*g`, straightforward.
- **Drag**: `F = -0.5 * rho_air * C_d * A * |v| * v` (quadratic drag, the
  standard model for a sphere at football speeds/Reynolds numbers).
  `C_d = 0.2` sits at the low end of the literature range for a
  hard-struck ball above the aerodynamic "drag crisis" (real footballs have
  speed-dependent drag, higher at slow rolling/passing speeds -- this sim
  uses one fixed value, deliberately on the low-energy-loss end).
- **Magnus effect**: `F = rho_air * A * r_ball * C_L * (omega x v)`. This is
  a simplified/tuned proxy for the real (much messier) aerodynamic Magnus
  force, per Idea.md's explicit instruction to "approximate" this rather
  than model it exactly. `C_L = 0.25` was chosen to produce a visible but
  not absurd curve on a hard, heavily-spun shot over ~20-30m - tune if shots
  curve too much or too little.
- **Ground bounce - vertical**: restitution `e_v = 0.75`. This is already
  at/above the realistic ceiling for a grass pitch (artificial turf is
  typically measured around 0.65; a rigid-surface FIFA ball-quality test
  wants ~0.79-0.88) -- i.e. already tuned toward the low-energy-loss end.
- **Ground bounce - horizontal/spin (friction-coupled, not a flat
  constant)**: unlike vertical restitution, real tangential bounce behaviour
  is governed by friction and spin at the contact point, not a fixed
  retention fraction -- a ball with topspin loses little horizontal speed on
  a bounce, one with backspin or no spin loses much more (Cross, "Bounce of
  an oval shaped football," Physics Dept., Univ. of Sydney). `step_ball`'s
  `_resolve_bounce_friction` implements this directly: the ball's
  ground-contact-point slip velocity is computed from its horizontal
  velocity and horizontal-axis spin (`spin.x`/`spin.y` -- see
  `ui/kick_trajectory.py`'s topspin/backspin/sidespin convention, unchanged
  by this), then a Coulomb friction impulse (capped by
  `bounce_friction_coefficient * normal_impulse`, or the smaller impulse
  needed to fully cancel the slip if the ball "grips" mid-bounce -- the
  common case per Cross's force-plate measurements) simultaneously reduces
  horizontal speed and changes horizontal-axis spin, coupled through the
  ball's assumed moment of inertia (`ball_inertia_shell_factor`, k in
  `I = k*m*r^2`). A gripped, no-spin bounce retains exactly `1/(1+k)` of its
  horizontal speed -- 2/3 (0.667) is the idealized thin-pressurized-shell
  value (retention 0.6); tuned down to **0.5** (retention 0.667) as a
  deliberate softening, still short of 2/5 (a solid sphere, which the ball
  demonstrably isn't). `spin.z`
  (vertical-axis spin) doesn't couple to this planar model and keeps the old
  flat `bounce_spin_retention` decay. Added 2026-09-03, replacing a flat
  `bounce_restitution_horizontal` constant that ignored spin entirely --
  see `agent_plans/physics_update.md` for the derivation and the literature
  this is based on.
- **Ground-contact spin-up** (`_resolve_ground_friction`, added
  2026-09-03): a grounded ball that ISN'T yet rolling without slipping
  (spin doesn't match `v = r*omega` at the contact point -- e.g. any ball
  launched or kicked with less spin than true rolling needs, including a
  plain zero-spin roll) pays a real, physically-derived cost spinning up
  to true rolling, via the same Coulomb friction/moment-of-inertia physics
  as a bounce (`_apply_ground_friction_impulse`, shared by both). A ball
  that fully grips retains exactly `1/(1+k)` of its pre-slip speed
  (`k = ball_inertia_shell_factor`) -- the same relationship behind the
  textbook "5/7 v0" result for a solid sphere sliding-to-rolling under
  friction. This was previously unmodeled entirely: ground contact never
  touched spin at all, so "rolling" was really "sliding with a flat
  friction constant," missing the real (and larger) transition cost that
  precedes true rolling. Once the ball IS rolling without slipping, this
  is a no-op and the much gentler `rolling_friction_coefficient` below
  takes over -- critically, that ongoing decay now also scales spin down
  in lockstep with velocity (both by the same factor each tick), keeping
  `v = r*omega` true as the ball slows; without that, the next tick would
  reopen slip and re-trigger the much stronger ground-friction correction
  forever, instead of ever actually settling into gentle rolling decay
  (see `step_ball`'s `remaining_slip` check). See
  `agent_plans/physics_update.md` for the derivation, and the follow-on
  fix this required in `kicking.py`'s `pass_ball`/`orders.py`'s
  `_try_push_kick` (both now launch with matching rolling spin, since
  their calibration assumed no spin-up cost existed).
- **Rolling friction**: once the ball is ALREADY rolling without slipping
  (`|v_z| < bounce_threshold_mps` AND no meaningful contact-point slip),
  horizontal speed decays via `a = mu_roll * g`, `mu_roll = 0.05`, chosen
  so a ball rolled at ~5 m/s travels roughly 20m before stopping - a
  plausible distance for a firm pass along real grass, and already at the
  low-friction end of grass's realistic range (~0.03-0.10).
- **Possessed ball**: `step_ball()` is a no-op if `ball.possessed_by` is
  set - `Match._sync_possessed_ball()` handles that ball's motion instead
  (it's glued to the carrying player, not simulated freely).

**Bug fix - "bounce" vs "resting contact" (`was_grounded_before_tick` +
`BOUNCE_THRESHOLD_MPS`):** the ground-collision code distinguishes a genuine
bounce from ordinary resting/rolling ground contact. This matters because
gravity's per-tick integration nudges a *resting* ball's next-tick
z-position slightly below `ball_radius_m` every single tick, giving
`new_velocity.z` a small negative value purely as an artefact of that
integration - not a real bounce. An earlier version of this code treated
*any* negative `new_velocity.z` while grounded as a full bounce, applying
horizontal restitution on *every* tick (~30x/second) instead of the
intended, much gentler `rolling_friction_coefficient` - this decayed any
grounded/rolling ball's speed almost instantly (a ball passed at 5 m/s would
stop within about a metre instead of the intended ~20m).

The first fix for this was a velocity check, `BOUNCE_THRESHOLD_MPS`
(0.5 m/s): only treat a tick as a real bounce if the incoming vertical
velocity exceeds it. **This alone is dt_s-dependent** -- the per-tick
gravity-integration artefact is ~`gravity_mps2 * dt_s`, so the check is only
safe below `dt_s ≈ bounce_threshold_mps / (gravity_mps2 *
bounce_restitution_vertical) ≈ 0.068s (~14.7Hz)`. The UI's 30Hz tick has a
comfortable margin; training's `sim_dt_s` (0.06 as of this writing, ~16.7Hz)
was sitting at only a ~12% margin from that boundary, and the "faster
training" `sim_dt_s` value `ai_config.json`'s own comment used to suggest
(0.067s) was at 98.6% of it -- one nudge from silently reintroducing this
exact bug (confirmed 2026-09-03: a resting ball misfired as a full bounce at
`dt_s=0.07` and above). **Fixed (2026-09-03) by gating on
`ball.is_grounded()` checked BEFORE this tick's integration**
(`was_grounded_before_tick` in `step_ball`): an already-grounded ball can
never misfire as a bounce regardless of `dt_s`, since the classification no
longer depends on a single tick's gravity-integration artefact at all. A
ball that was genuinely airborne and only now reaches the ground still
carries real, multi-tick-accumulated fall velocity, so
`BOUNCE_THRESHOLD_MPS` remains meaningful (and now dt_s-independent) for
that case. See `tests/unit/test_ball_physics.py`'s
`test_rolling_ball_decelerates_at_the_analytically_correct_rate` and
`test_rolling_ball_travels_plausible_distance_before_stopping` for
regression tests against this. If you ever see a rolling ball stopping
suspiciously fast again, check this gate first.

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
- **Still launched spin-free, compensated via extra speed instead**
  (2026-09-03): `ball_physics.py`'s ground-contact spin-up cost
  (`_resolve_ground_friction`) means a spin-free pass now pays a real
  transition cost before it's genuinely rolling, which broke essentially
  every pass-balance/behaviour test. The fix tried first -- giving passes
  matching rolling spin at launch, via a `rolling_spin_for_direction`
  helper -- worked physically but broke something more important: the
  neural network's own kick path is hardcoded spin-free
  (`ai/action/apply_nn_action.py`, see `agent_plans/spin_implementation_plan.md`),
  so it could no longer replay a spin-imparting rules-AI kick exactly,
  breaking `test_rules_ai_nn_replay_equivalence.py`. Reverted in favour of
  `spinup_speed_boost_base`/`spinup_speed_boost_per_m`
  (`PassingParams`/`pass_speed_mps`) -- launch harder instead of avoiding
  the cost, keeping every kick path spin-free and NN-replayable. Rules-AI
  push-kicks (`orders.py`'s `_try_push_kick` /
  `_push_kick_power_fraction`) get the equivalent treatment via
  `orders.json`'s `push_kick.spinup_speed_boost`. See
  `agent_plans/physics_update.md` section 8 for the full history,
  including why the compensation can't fully restore the original
  long-pass (60m+) accuracy -- more launch speed also means more angular
  error (`kick_sigma_rad`'s power coupling), so past a point boosting
  speed further stops helping distance and only hurts accuracy.

## `possession.py` - first-touch control-time model

**The height/timing constants below were found to have drifted from
`config/physics.json["control_time"]` (2026-09-05 audit) and were
corrected.**

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
  - rises quadratically to `f=1.5` at waist height (0.95m) - a knee-to-waist
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
  `t_base = 0.04s` (an irreducible minimum reaction/first-touch time) and
  `t_scale = 0.17s`.
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
- **First-touch speed snap** (`control_speed_multiplier`, not previously
  documented here): the instant a loose ball is picked up and
  `player.state` becomes `CONTROLLING_BALL`, `Match._update_loose_ball_
  pickup` immediately multiplies the player's *current* velocity by
  `movement_params.control_speed_multiplier` (**0.6**, i.e. a 40% cut) —
  a one-time snap, separate from (and on top of) the ongoing ball-carry
  top-speed multiplier in `movement.py`. The player then coasts at that
  reduced speed for the rest of the control-time window (`t_control`
  above) before normal movement resumes.
  - **Exception — armed one-touch redirect**: if the player had already
    committed to a push-kick via `kick_armed`/`kick_armed_direction` (see
    `orders.py`'s `_try_push_kick` and `apply_nn_action.py`), the pickup
    skips `CONTROLLING_BALL` (and this speed snap) entirely and redirects
    the ball immediately with no control-time delay at all — the actual
    one-touch path. Gated on `ball_settled` (`|ball.velocity.z| <
    ball_pickup_params.armed_redirect_settle_vz_mps`) so an armed redirect
    can't fire while the ball still has real vertical velocity (e.g.
    mid-bounce from the same player's own prior kick) — see the long
    comment at `Match._update_loose_ball_pickup`'s armed-kick branch for
    the bug history this guards against.
- A small proportional Gaussian noise term (`noise_sigma_fraction = 0.1`,
  scaled by `rng_reduction`) is added on top of the deterministic
  `t_control` in `Match._update_loose_ball_pickup`, so touches aren't
  perfectly predictable even for a fixed situation - this wasn't explicitly
  requested in Idea.md but was added to stay consistent with the rest of
  the game's "nothing is ever perfectly deterministic" philosophy. Flag to
  the project owner if this should be removed/reconsidered.

## `tackling.py` - tackle skill checks

**This section was found to have drifted significantly from the live code
(2026-09-05 audit) and was rewritten from scratch against
`tackling.py`/`match.py`/`physics.json["tackling"]` — treat the numbers below
as authoritative over any other doc/comment that disagrees.**

`attempt_tackle()` runs a skill check with an angle-dependent boost:

```
effective_boost = base_boost * (1 + angle_modifier)
tackler_roll  = skill_roll(tackling_attr * effective_boost, rng_reduction)
dribbler_roll = skill_roll(effective_dribbling_attr, rng_reduction)
tackler wins iff tackler_roll >= dribbler_roll
```

- `base_boost` is `tackler_boost` (**1.25**, +25%) for an outfield tackler,
  or `goalkeeper_tackle_boost` (**2.0**, +100%) if the tackler is a
  goalkeeper — a keeper coming to punch/collect is a much stronger
  challenge than an outfield tackle.
- `angle_modifier` (`tackle_angle_modifier`) depends on which direction the
  tackle comes from relative to the dribbler's *heading* (not the tackler's
  own heading): **+0.10** tackling from directly in front
  (`angle_modifier_frontal`), **-0.05** side-on (`angle_modifier_side`),
  **-0.5** from directly behind (`angle_modifier_behind`) — piecewise-linear
  in `cos(angle)`. A tackler blindsiding a dribbler from behind is
  meaningfully weaker than one standing square in their path.
- `effective_dribbling_attr` is the target's `dribbling` attribute, reduced
  by the CONTROLLING_BALL penalty below where applicable.

At the default `rng_reduction=0.3`, tackling=0.8 vs dribbling=0.6 (no angle
modifier) still lands the tackler a win rate in the 70-90% design-target
band — see `tests/balance/test_tackling_balance.py`, which verifies this
empirically over 5000 trials and reports a full win-rate grid across
attribute pairs. That test predates the angle modifier and doesn't vary it,
so it's exercising the side-on-equivalent (`angle_modifier≈0`) case.

### Speed consequences — one-time snaps at the resolution instant

`apply_tackle_result()` is the only place velocity is written for a tackle
outcome; it applies **once**, not as an ongoing debuff:

- **Always, to both players regardless of outcome**: tackler
  `velocity *= tackle_attempt_tackler_speed_mult` (**0.5**), tacklee
  `velocity *= tackle_attempt_tacklee_speed_mult` (**0.8**) — contact costs
  both players pace even before considering who won.
- **The loser** (whichever roll was lower) additionally loses
  `min(|tackler_roll - dribbler_roll| * loser_speed_penalty_scale (1.2),
  loser_speed_penalty_max (0.8))` on top of their base multiplier — a
  convincingly-lost challenge costs real pace, a close one barely more than
  the base contact cost.
- **If the dribbler wins**, how convincingly they won also matters for
  *their own* speed (independent of the loser-penalty above, which in this
  branch applies to the tackler instead): margin `>= dribble_beaten_speed_
  threshold` (**0.35**, i.e. the dribbler's winning roll beat the tackler's
  by ≥35%) → dribbler keeps full speed; a narrower win scales down linearly
  to `1 - dribble_beaten_max_penalty` (**0.8**) at a near-zero margin — even
  "winning" a tackle attempt can cost a dribbler most of their speed if it
  was close.

### Inactivity — unified for both players, every attempt

`apply_tackle_result()` sets **both** the tackler and the tacklee to
`PlayerState.INACTIVE_TACKLED` for `tackle_cooldown_s` (**1.2s**) on *every*
resolved attempt, win or lose — there is no separate shorter "failed lunge"
duration any more; both parties get the same cooldown regardless of outcome.
While inactive a player: can't tackle or be tackled
(`Player.is_available_to_tackle()`), is excluded from push-apart collision
resolution *and* from velocity damping (see `collision.py` below), and
does not regen stamina (`Match._update_state_timers` only regens `ACTIVE`
players). There is **no ongoing speed penalty** for being inactive itself —
`inactive_speed_penalty` doesn't exist in `physics.json` (removed at some
point after being originally planned; the only speed effect of a tackle is
the one-time snap above, after which normal deceleration/physics take over).

### Phase B tackle modifiers (added after initial implementation)

Two additional modifiers are applied at the `attempt_tackle()` call sites in
`Match._process_orders` — not inside `attempt_tackle()` itself, keeping the
function signature clean:

**1. GK outside-box penalty** (`gk_outside_box: bool = False`):
- If the tackler is a goalkeeper and is **outside** their own penalty box,
  their `effective_boost` (already the 2.0 GK boost, see above) is
  multiplied by `(1 - goalkeeper_outside_box_tackle_penalty)` (currently
  `0.4` → 40% penalty), bringing the boost down to ~1.2 — roughly outfield
  level, so a roaming keeper isn't a supertackler everywhere outside their box.
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
  `dribbling_attr * (1 - 0.25 * penalty_frac)` (`Match._effective_dribbling`).
- Config: `physics.json["tackling"]["control_time_penalty_reference_s"]`
  (currently `0.3s`).
- If the tackler wins against a `CONTROLLING_BALL` target, the tackler is
  given the ball (not just the target losing possession).
- **Asymmetry (currently real, not obviously intentional)**: this penalty is
  only applied via `Match._attempt_tackle_contact` — the armed-tackle path
  used by `ChaseTackleOrder`/`GetPossessionOrder`. `Match._check_head_on_
  tackles` (the collision-based auto-tackle fallback) calls `attempt_tackle`
  with `carrier.attributes.dribbling` raw, never routing through
  `_effective_dribbling()`. So a player mid-first-touch is easier to
  dispossess via a deliberate chase-tackle than via an incidental head-on
  collision, for no documented reason — flag to the project owner if this
  should be unified.

**4. Aerial-ball tackle immunity — currently DEAD CODE, deferred on
purpose**:
- `physics.json["control_time"]["control_tackle_immune_height_m"]` (0.95,
  waist height) and the checks that read it
  (`Match._attempt_tackle_contact`, `Match._check_head_on_tackles`'s inline
  check) are real, reachable code that's SUPPOSED to block a tackle
  attempt against a `CONTROLLING_BALL` target whose ball is still above
  that height (a first-touch on an aerial ball, not yet brought to the
  ground). See `tests/scenario/test_control_behaviour.py`'s module
  docstring for the original intent.
- In practice this can only ever fire on the exact tick control starts.
  `Match._sync_possessed_ball()` (tick-order step 3) unconditionally snaps
  the ball to `radius_m` height for ANY carrier, every tick, regardless of
  `CONTROLLING_BALL` state — and it runs BEFORE the tackle-resolution steps
  (`_check_armed_tackles`/`_check_head_on_tackles`, missing from this file's
  own tick-order list above — TODO). So by the first tick a tackle attempt
  is actually resolved against an already-controlling target, the ball's
  height has already been flattened, and the immune-height check always
  reads a ground-level ball no matter how high the ball was when control
  began.
- Grounding the ball immediately on control is intentional for now (simpler
  physics) — NOT something to silently "fix" by making
  `_sync_possessed_ball` preserve height during `CONTROLLING_BALL`, without
  a deliberate decision to actually support aerial control properly.
  `tests/scenario/test_control_behaviour.py::_disabled_test_
  controlling_aerial_ball_immune_to_regular_tackle` is disabled (renamed
  with a leading underscore so pytest skips it) pending that decision — its
  head-on-tackle sibling test currently still passes by coincidence (a
  hand-set-up single `match.step()` that never exercises the multi-tick
  glue-then-check ordering), not because the underlying issue is fixed.

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
  tunable in `physics.json["collision"]`. Like position push-apart, damping
  is skipped entirely for pairs where either player is inactive - a player
  can push straight past someone they just tackled (or who just failed to
  tackle them) instead of getting stuck gliding against them at reduced
  speed. Because overlap can persist across multiple ticks (for active
  pairs), the damping compounds — this is intentional but the floor
  prevents it from driving velocity to zero.

**Ball-blocking by inactive players** (`resolve_ball_block_by_inactive_players`):
  a loose, in-flight ball can still be blocked by an
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
  `orders.braking_speed_mode()` — no velocity snaps.

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

## Player action callbacks (`Player.on_kick`, `Player.on_tackle`)

Two optional per-player callbacks fire at the **exact engine tick** an action executes:

```python
player.on_kick    = lambda player: ...   # KickOrder, ShootOrder, PassOrder
player.on_tackle  = lambda player: ...   # ChaseTackleOrder (when contact is made)
```

- `on_kick` fires inside `Match._process_orders()` immediately after `kick_ball()`
  or `pass_ball()` is called.
- `on_tackle` fires when `are_touching(player, target)` is True and
  `target.is_available_to_tackle()` — i.e. at the moment the tackle attempt
  executes, before the tackle outcome is resolved.
- Both default to `None` (no-op, zero cost).
- Useful for: BC recording, UI action icons, statistics, logging.

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

**The AI package does NOT drive the engine via Orders.** An earlier design
routed neural-network output through the standard `orders.py`/`actions.py`
interface (constructing `MoveOrder`/`KickOrder`/etc. and assigning them to
`player.current_order`, same as the UI/rules AI); that has since been
replaced by `ai/action/apply_nn_action.py::apply_action_to_player()`, which
writes execution-network output directly onto `Player` fields — no Order of
any kind is ever constructed for a neural-controlled player. See
`knowledge.md`'s "Neural network / Orders boundary" section and
`ai/knowledge.md`'s "THE NEURAL NETWORK NEVER ISSUES ORDERS" section for the
full current picture; only the rules-based AI, BC label generation (reading,
never issuing), and `HybridPlayerAI`'s explicit order-override channel ever
touch `player.current_order`.

**Illegal-action guardrail audit** (current state, not the original
ai_design_doc.md section 11 checklist — that described the since-replaced
Order-based path):

- **Tackle**: `apply_action_to_player()` explicitly flags
  `illegal_action=True` for `tackle_while_inactive` (player not
  `is_available_to_tackle()`) and `tackle_no_carrier` (no opposition ball
  carrier to tackle) before ever setting `player.tackle_armed`. This is the
  only case currently wired into the reward function's
  `illegal_action_penalty` (see `ai/env/reward.py`).
- **Kick**: `player.kick_with_direction()` silently no-ops if
  `ball.possessed_by != player.player_id` (mirrors the engine-level safety
  net `KickOrder`/`kick_direct()` also rely on), but `apply_action_to_player()`
  does not currently flag this as `illegal_action` for the reward function —
  a kick attempted without possession is a safe no-op, just not a penalised
  one at present.
- **Pass**: there is no separate pass-execution action in the current
  execution network (`ExecutionHeadsRaw` has no dedicated pass head) —
  passing is Phase 3 (not yet implemented, see `ai_trainer_knowledge.md`
  curriculum section), so `PassOrder` is not part of this boundary at all
  today.
- **Save**: `SaveOrder` is rules-AI/GK-only in every current phase; the
  execution network has no save action, so there is nothing to guard here
  either.
- AI must be punished for illegal attempts AND the engine must be a safe
  no-op — both protections coexist where implemented (see design doc 9.7);
  currently that's tackle only, per above.