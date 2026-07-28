# Plan: FootballCoach — batch of engine/UI features (Idea2.md backlog)

TL;DR: 15 features grouped into 8 phases, engine-first then UI. All new
constants go in physics.json (new sections: `repulsion`, `collision`,
`marking`, plus additions to `tackling`, `goalkeeping`, `control_time`,
`kicking`). Reuse existing patterns: `GetPossessionOrder`'s intercept/tackle
logic, the control-time model in possession.py, and
`ScenarioDefinition`/`ScenarioLoop` in ui/scenarios.py. **Note on Phase A**:
player repulsion is explicitly an AI/order-layer steering mechanic (new
top-level `steering.py`, used only by `MoveOrder` handling), not an engine
physics mechanic — `engine/movement.py` and `engine/collision.py` remain
untouched/unaware of it, matching the existing engine-vs-AI-layer boundary
(`actions.py`/orders decide *what* to ask for, `engine/` just executes it).
**Note on Phase H**: now includes a generic, reusable `ScenarioParam`
mechanism so every UI scenario (not just shooting) exposes adjustable
playtesting parameters, plus a new 1v2 scenario (elite attacker vs. average
defender+GK) alongside the existing 2v2 plan.

## Phase A — Movement: player repulsion during Move orders
**Important scoping note (per user, explicit correction)**: repulsion is
NOT an engine/physics mechanic — it must not live in `engine/` or be baked
into `movement.py`'s `step_player_towards` (which stays pure "go this
direction at this speed" kinematics, unaware of other players). Repulsion
is an **AI/order-layer decision**: it's steering logic that only the
`MoveOrder` handling uses to *decide what direction/speed to ask the engine
for*, analogous to how `Match._intercept_target`/`_leading_pass_target`
already compute AI-decided target points before handing them to engine
movement/kicking functions — those are also order-layer decisions, not
engine physics. Concretely: the new module must NOT touch
`engine/collision.py` (that remains the actual physics overlap/push-apart
resolution, unrelated to this) and must not be reachable from any order
other than `MoveOrder`.

1. New top-level `src/footballcoach/steering.py` (sibling to `actions.py`,
   NOT under `engine/`, to keep the engine/AI boundary explicit):
   `compute_repulsion(player, other_players) -> (adjusted_direction,
   speed_multiplier)`. For each other player within `radius_m=4.0` that
   does NOT currently have the ball: repulsion vector
   `r_i = normalize(self.pos - other.pos) * strength(d)`, linear falloff
   `strength(d) = repulsion_strength_base * (1 - d/radius_m)`. Sum to
   `net_repulsion`. If `self` has ball, multiply by
   `ball_carrier_repulsion_mult` (~1.8) and reduce `target_speed` by
   `min(ball_carrier_speed_penalty_max, |net_repulsion| * speed_penalty_scale)`.
   Collision-course detection: `rel_vel = self.velocity - nearest_other.velocity`;
   if `normalize(rel_vel).dot(normalize(net_repulsion)) < alignment_dot_threshold`
   (~-0.7, i.e. heading nearly straight at the obstacle), add an orthogonal
   nudge: `ortho = perpendicular(net_repulsion)` with sign chosen via 2D cross
   product `cross_z(net_repulsion, desired_dir)` (per user: simple, predictable;
   write a balance test checking for direction oscillation/flip-flopping across
   ticks near the threshold boundary and add small hysteresis if it shows up),
   magnitude `min_orthogonal_adjust_mps` (tunable, start ~1.5, tune via balance
   test). Final direction = `normalize(desired_dir + net_repulsion*w + ortho)`.
   This module takes `Player`/positions as plain inputs and returns plain
   vectors/scalars — it has no dependency on `engine/movement.py` or
   `engine/collision.py` internals, only on `mathutils`/`entities`, so it
   could in principle be swapped out or overridden by a future NN policy
   without touching engine code at all (same spirit as `actions.py` being a
   thin AI-facing layer over the engine).
2. Wire into `match.py::_process_orders` MoveOrder branch only (per literal
   ask) — `_process_orders` calls `steering.compute_repulsion(...)` to get
   an adjusted direction/speed, THEN passes that into the unmodified engine
   `step_player_towards` (which remains a pure physics/kinematics function
   that only ever sees a final direction+sprint flag, with no awareness that
   repulsion/AI steering exists upstream of it). No other order type
   (ChaseTackleOrder, GetPossessionOrder's chase phase, SaveOrder, MarkOrder
   from Phase F, etc.) calls into `steering.py` — confirmed explicitly
   out of scope, per "Move order only".
3. New config section `physics.json["repulsion"]`: `radius_m`(4.0),
   `strength_base`(3.5), `ball_carrier_repulsion_mult`(1.8),
   `ball_carrier_speed_penalty_max`(0.4), `speed_penalty_scale`(0.0847),
   `alignment_dot_threshold`(-0.7), `min_orthogonal_adjust_mps`(1.5, main
   tuning target — see derivation below).

   **Numeric derivation (first-guess, tunable via balance test, same
   convention as the existing angle_error constants elsewhere in the repo)**:
   - `strength_base = 3.5`: this is a dimensionless "m/s-equivalent"
     directional weight blended into the movement direction vector (not a
     literal physical force — repulsion here biases *direction*, not
     acceleration). Chosen as roughly half of a mid-tier player's top speed
     (`v_top(attr=0.5) = 5.0+4.5*0.5 = 7.25` m/s) at d=0 (touching), so the
     repulsion is a strong-but-not-total influence on heading — a player can
     still push through if their own desired-direction weight dominates.
     Linear falloff `strength(d) = 3.5*(1-d/4.0)`: 2.625 at d=1m, 1.75 at
     d=2m, 0.875 at d=3m, 0 at d=4m.
   - `ball_carrier_speed_penalty_max = 0.4`: sits between the existing
     `ball_carry_speed_multiplier` cap (max ~23% reduction just from
     dribbling, [movement.py](src/footballcoach/engine/movement.py)) and the
     existing mid-turn `turn_speed_penalty` cap (up to 70%,
     [movement.py](src/footballcoach/engine/movement.py)) — dodging a nearby
     opponent while dribbling is a comparably disruptive action to a hard
     turn, so a 40% cap is a reasonable middle point between "just
     dribbling" and "sharp turn".
   - `speed_penalty_scale = 0.0847`: derived so the 40% cap is reached at a
     realistic "about to collide" distance rather than requiring literal
     d=0 contact. At d=1m with `ball_carrier_repulsion_mult=1.8`:
     `|net_repulsion| = 3.5*(1-1/4)*1.8 = 4.725`. Setting
     `speed_penalty_scale = 0.4/4.725 ≈ 0.0847` means
     `min(0.4, 4.725*0.0847) = 0.4` is already saturated at d=1m, and stays
     saturated (capped, not increasing further) for any closer approach.
   - `min_orthogonal_adjust_mps = 1.5`: deliberately the main tuning knob —
     the user's own framing was "we have to find this value" via balance
     testing, so it's seeded at a brisk-walk-pace guess (1.5 m/s) rather
     than derived analytically like the others.
4. Balance tests (`tests/balance/test_repulsion_balance.py`, report full
   stats per repo convention, not just pass/fail):
   - **Baseline (regression)**: single player, no neighbours within 4m —
     path/arrival time identical to pre-Phase-A behaviour.
   - **Off (repulsion=0, min_orthogonal_adjust=0)**: two players on a direct
     collision course (same start-line offset by width, opposite headings)
     must overlap/run through each other — assert minimum observed
     separation over the run is `< radius_a + radius_b` (0.6m), confirming
     the *lack* of avoidance is correctly reproducible as a control case.
   - **On (tuned values)**: same collision-course setup — assert minimum
     separation stays `>= radius_a + radius_b + 0.1m` margin at every tick,
     and report arrival-time delta vs the unobstructed straight-line time
     (sanity bound, e.g. flag if avoidance costs more than ~2x the direct
     travel time as likely over-tuned).
   - **Ball-carrier stronger repulsion + slowdown**: repeat the collision
     course with the moving player carrying the ball — assert (a) avoidance
     still succeeds (min separation respected) and (b) report mean speed
     during the close-approach window vs the non-carrying case, confirming
     it's measurably slower (the `ball_carrier_repulsion_mult`/
     `ball_carrier_speed_penalty_max` effect is actually visible in the
     numbers, not just theoretically wired).
   - **No repulsion from ball carrier**: a player running near a *stationary*
     ball-carrying opponent (not on a collision course, just passing close)
     must NOT be deflected or slowed — asserts the "no repulsion if the
     nearby player has the ball" exclusion is honoured for the *other*
     player's perspective (only the carrier gets bonus repulsion/slowdown
     toward others; others are not repelled by a stationary carrier who
     isn't in their way — clarify this distinction in code comments since
     the literal ask ("no repulsion if the nearby player in question has the
     ball") is about skipping *that neighbour* as a repulsion source, which
     the base `compute_repulsion` loop already does by construction).
   - **Oscillation/hysteresis check**: construct a near-boundary scenario
     where `alignment_dot_threshold` is repeatedly crossed tick-to-tick
     (e.g. slowly converging paths) and assert the orthogonal-nudge sign
     doesn't flip back and forth more than once every N ticks (e.g. N=5);
     if it does, add small hysteresis (e.g. require crossing the threshold
     by a margin, or hold the previous sign for a minimum duration) as
     scoped with the user.
   - **Three-plus neighbours**: a player boxed in by 2-3 nearby players
     from different angles — assert `net_repulsion` sums sensibly (no
     crash/NaN from cancelling vectors) and the player still makes forward
     progress rather than freezing (report displacement over a fixed
     window).

**Relevant files**: new `src/footballcoach/steering.py` (AI-layer, new file —
NOT under `engine/`), [src/footballcoach/engine/movement.py](src/footballcoach/engine/movement.py)
(`step_player_towards` — read-only reference, stays unmodified/engine-pure),
[src/footballcoach/engine/match.py](src/footballcoach/engine/match.py)
(`_process_orders` MoveOrder branch — the only call site for `steering.py`),
[src/footballcoach/entities/player.py](src/footballcoach/entities/player.py),
[src/footballcoach/config/physics.json](src/footballcoach/config/physics.json)
(new `repulsion` section — placed alongside other tunables even though the
code lives outside `engine/`, consistent with the repo's existing
config-driven-constants convention for AI/action defaults like
`actions.py`'s `DEFAULT_SHOOT_POWER_FRACTION`).

## Phase B — Tackling & possession fixes (depends on nothing, can run parallel with A)
1. **GK tackle penalty outside own box (-40%)**: in
   `engine/tackling.py::attempt_tackle`, when `is_goalkeeper_tackle=True` and
   tackler is outside their own box (`pitch.is_in_box(tackler.position,
   left=team_defends_left)`), multiply `effective_boost` by
   `(1 - goalkeeper_outside_box_tackle_penalty)` (new config value 0.4).
   Caller in match.py must pass pitch + tackler position through.
2. **GK cannot be tackled in own box while in possession**: at the tackle
   call site(s) in match.py (wherever `attempt_tackle` is invoked against a
   ball-carrying goalkeeper), short-circuit to an automatic dribbler-win
   (tackle always fails) if `dribbler.is_goalkeeper and
   pitch.is_in_box(dribbler.position, left=dribbler_team_defends_left)`.
   Applies to `TackleOrder`, `ChaseTackleOrder`, `GetPossessionOrder`, and
   `_check_head_on_tackles`.
3. **Dribble-roll penalty while still controlling ball (CONTROLLING_BALL
   state) and tackled**: currently a player mid first-touch
   (`PlayerState.CONTROLLING_BALL`, ball frozen at their feet but
   `ball.possessed_by` not yet set) can't be targeted by tackles at all —
   confirm this during implementation (check what `ball_carrier()`/tackle
   target resolution in match.py considers). Need to: (a) allow tackle
   targeting of a `CONTROLLING_BALL` player as if they have the ball; (b) NO
   new field needed — `Player.state_timer_s` already holds remaining control
   time directly while `state == CONTROLLING_BALL` (confirmed: it's the
   single generic transient-state timer, decremented every tick in
   `_update_state_timers`; user correctly pointed out a second field would
   be redundant); (c) at tackle resolution against such a player, penalize
   their effective dribbling attribute off the *absolute* remaining seconds
   (not a fraction of an original total, which isn't stored) against a new
   reference constant: `penalty_frac = min(1.0, state_timer_s /
   control_time_penalty_reference_s)`; `dribbling_eff = dribbling_attr *
   (1 - 0.25*penalty_frac)` (0% penalty as timer nears 0, up to 25% at/above
   `control_time_penalty_reference_s` remaining). New config
   `tackling.control_time_penalty_reference_s` ≈ 0.3 (roughly the typical
   `t_control` magnitude from possession.py, so a "just picked up" touch is
   near max penalty and an "about to finish" touch is near zero).
4. New config: `tackling.goalkeeper_outside_box_tackle_penalty`(0.4).

   **Numeric derivation / expected impact**: the existing
   `goalkeeper_tackle_boost` (2.0) is already far above the outfield
   `tackler_boost` (1.25) — goalkeepers are modeled as much stronger
   tacklers by default. Outside the box, `effective_boost = 2.0*(1-0.4) =
   1.2*(1+angle_modifier)`, i.e. still comparable to or slightly better than
   an outfield tackler's baseline 1.25, which fits the intent: a GK
   wandering out of their box to make a tackle should end up "no better than
   an average outfield defender", not disproportionately dominant, without
   being made *worse* than outfield play (that would be an odd asymmetry).
   Worked example — GK tackling=0.7 vs dribbler dribbling=0.6,
   `rng_reduction=0.3`, no angle modifier (head-on): recall
   `skill_roll(skill) = (0.3 + 0.7*U)*skill`, `U~Uniform(0,1)`, so both rolls
   are `skill * X` with `X~Uniform(0.3,1)` i.i.d. across the two rolls.
   `P(tackler wins) = P(X_t/X_d >= dribbling/(tackling*effective_boost))`.
   - Inside box: `effective_boost=2.0` → ratio threshold `r =
     0.6/(0.7*2.0) = 0.429`. Since `X_t,X_d` are i.i.d., `P(X_t/X_d>=1)=0.5`
     by symmetry, and `r=0.429 < 1` pushes the win probability comfortably
     above 50% — expect roughly 80-85% (exact figure needs the balance
     test's numerical/Monte-Carlo estimate, same as the existing tackling
     balance test's 82.8% figure for tackling=0.8/dribbling=0.6/boost=1.2).
   - Outside box: `effective_boost=1.2` → ratio threshold `r =
     0.6/(0.7*1.2) = 0.714`, noticeably closer to the symmetric r=1 point,
     so win probability should land closer to 60-70% — a clear, measurable
     drop vs the inside-box figure, which is exactly the "discourage it"
     effect requested. Balance test should assert
     `win_rate_outside < win_rate_inside - some_margin` (e.g. at least 10
     percentage points) rather than hardcoding the above estimates, since
     they're derived approximately by hand, not computed exactly.
5. Tests:
   - **GK tackle win-rate outside vs inside box** (balance, N>=2000 per the
     repo's existing convention for 2-outcome probability estimates,
     `rng_reduction=0.3`): report both win rates and assert outside is
     meaningfully lower (see derivation above for expected direction/rough
     magnitude).
   - **GK-in-box-with-ball untackleable** (scenario test,
     `rng_reduction=1.0` for determinism): any tackle attempt (all four
     order types) against a GK possessing the ball inside their own box
     always results in `TackleResult.tackler_won=False` and no state change
     to the GK — run N=100 trials at `rng_reduction=0.3` too, to confirm
     it's *always* false even under full randomness (this must not be
     merely statistically likely — it's an absolute rule, so assert 0/N
     tackler wins, not just a low rate).
   - **Boundary regression**: same GK, ball, same position, but just outside
     the box line (e.g. 0.5m over `box_length_m`/2) — normal outside-box
     tackle rules (Phase B item 1) apply, confirming the box-boundary check
     itself is correct and the two new rules (outside-box penalty,
     in-box-untackleable) don't overlap/conflict at the boundary.
   - **Dribble roll penalty during CONTROLLING_BALL**: (a) unit test — table
     of `state_timer_s` values `[0.0, 0.075, 0.15, 0.225, 0.3, 0.4]` against
     `control_time_penalty_reference_s=0.3` confirming
     `dribbling_eff = dribbling_attr*(1-0.25*min(1, state_timer_s/0.3))`
     produces `[1.0, 0.9375, 0.875, 0.8125, 0.75, 0.75]x dribbling_attr`
     (capped at 25% once `state_timer_s>=0.3`); (b) balance test — fixed
     tackler (e.g. tackling=0.6) vs fixed dribbler (dribbling=0.6) tackled
     at `state_timer_s` near `control_time_penalty_reference_s` (early,
     near-max penalty) vs near 0 (late, near-zero penalty): assert win rate
     is measurably higher in the early case, and report both rates; (c)
     regression — a dribbler tackled while `state==ACTIVE` (already in full
     possession, not mid-control) is completely unaffected by this penalty
     (existing tackle balance tests, e.g. tackling=0.8 vs dribbling=0.6
     70-90% band, must still pass unmodified).

**Relevant files**: [src/footballcoach/engine/tackling.py](src/footballcoach/engine/tackling.py),
[src/footballcoach/engine/match.py](src/footballcoach/engine/match.py) (tackle call
sites, `_update_loose_ball_pickup`, `_check_head_on_tackles`),
[src/footballcoach/entities/player.py](src/footballcoach/entities/player.py),
[src/footballcoach/entities/pitch.py](src/footballcoach/entities/pitch.py)
(`is_in_box`/`is_in_either_box`).

## Phase C — Goalkeeper save & jumping (depends on nothing, parallel with A/B)
1. **Early interception on save**: extend `engine/goalkeeping.py` with
   `early_intercept_target(...)` reusing the same lead-prediction approach as
   `Match._intercept_target` (used by `GetPossessionOrder`). Compute
   candidate intercept point on the ball's flight path; compute
   `t_reach_intercept = distance(gk, intercept_point)/gk_effective_top_speed`
   and `t_reach_goal_line` (time for ball to reach the existing
   `save_target_position` plane, from `predict_goal_line_crossing`). If a
   shot is incoming (`predict_goal_line_crossing` returns non-None), ball is
   within `early_intercept_max_distance_m` (10.0) of the GK, and
   `t_reach_intercept < t_reach_goal_line * early_intercept_safety_margin`
   (0.85 — must be clearly reachable earlier, guards false positives),
   target the intercept point instead of the goal-line plane; else fall back
   to unmodified existing behavior.
2. **Jumping height penalty — GK AND outfield players** (extended per user
   request beyond original Idea2.md ask, which was GK-only): in
   `engine/possession.py::control_time_s`, both the GK-in-box path and a
   *new* outfield path get a "jump zone" above head height, with jumping
   modeled as strictly harder than the flat "receive it wherever" case, and
   GK favoured over outfield (can jump higher, penalised less per metre):
   - GK-in-box: keep existing `gk_height_factor_scale`(0.4) flattening for
     `h <= player_height_m` (1.8m, no jump needed up to head height). Above
     1.8m: `scale(h) = gk_height_factor_scale + (gk_jump_scale_at_max_reach -
     gk_height_factor_scale) * min(1, (h - player_height_m) /
     (gk_max_reach_height_m - player_height_m))`. New config
     `gk_jump_scale_at_max_reach`(1.2), `gk_max_reach_height_m`(2.2 — keeper
     height + arm/jump reach).
   - Outfield (new): below head height, difficulty uses the existing
     unmodified `height_difficulty_factor` curve (no special-casing, as
     today). Above `player_height_m` (1.8m — an outfield "jump zone" up to a
     *lower* max reach than GK, reflecting no gloves/less specialised
     technique): `scale(h) = 1.0 + (outfield_jump_scale_at_max_reach - 1.0) *
     min(1, (h - player_height_m) / (outfield_max_reach_height_m -
     player_height_m))` applied as a multiplier on the `(height_factor-1)`
     term above 1.8m (so below 1.8m nothing changes — pure regression
     safety). New config `outfield_jump_scale_at_max_reach`(2.0 — steeper
     than GK's 1.2, i.e. outfield jump control is penalised more per metre),
     `outfield_max_reach_height_m`(2.0 — lower ceiling than GK's 2.2, i.e.
     outfield players can't credibly control balls as high).
   - Existing `height_factor_max` cap (6.0) still applies on top in both
     cases.
3. New config: `goalkeeping.early_intercept_max_distance_m`(10.0),
   `goalkeeping.early_intercept_safety_margin`(0.85);
   `control_time.goalkeeper.jump_scale_at_max_reach`(1.2, renamed
   `gk_jump_scale_at_max_reach` for clarity vs the new outfield pair),
   `control_time.goalkeeper.max_reach_height_m`(2.2, renamed
   `gk_max_reach_height_m`); `control_time.outfield_jump_scale_at_max_reach`
   (2.0), `control_time.outfield_max_reach_height_m`(2.0).
   **Worked numeric example (jump penalty)**: outfield player, ball_control
   attr 0.5, at `h=2.0m` (0.2m above head, within outfield's 2.0m max reach
   — i.e. at the ceiling): `height_factor(2.0)` from the existing curve
   (`h>H`, capped at 6.0): `min(6.0, 4.0+1.0*((2.0-1.8)/1.8)) = 4.111`.
   Outfield jump scale at this height: since `h` is exactly at
   `outfield_max_reach_height_m`, `scale=outfield_jump_scale_at_max_reach=
   2.0` (full effect, fraction=1.0). Scaled term:
   `1.0 + (4.111-1.0)*2.0 = 7.222` used in place of `(height_factor-1)` in
   `compute_difficulty`, i.e. difficulty roughly doubles vs not applying the
   jump scaling at all (`4.111-1=3.111` unscaled) — concretely, with
   `k1=k2` terms held at 0 for isolation, `t_control = 0.1 + 0.3*(1-0.85*
   0.5)*7.222 ≈ 1.34s` (jump-scaled) vs `0.1+0.3*0.575*3.111≈0.64s`
   (unscaled height-factor-only) — roughly 2x slower, i.e. genuinely
   difficult, consistent with "a player leaping to control a head-high-plus
   ball should take noticeably longer". For GK-in-box at the same `h=2.0m`
   (within GK's 2.2m max reach, fraction=(2.0-1.8)/(2.2-1.8)=0.5):
   `gk_height_factor_scale`-flattened factor at `h=1.8` would be
   `1+(height_factor(1.8)-1)*0.4`; at h=2.0 the scale interpolates from 0.4
   toward 1.2: `scale=0.4+(1.2-0.4)*0.5=0.8`; giving a clearly easier time
   than the outfield player at the identical height — confirms the GK
   jump-advantage ordering holds at this sample point (exact balance-test
   numbers to be generated by running the code, this is a hand-check for
   plausibility only, flagged as such).
3. New config: `goalkeeping.early_intercept_max_distance_m`(10.0),
   `goalkeeping.early_intercept_safety_margin`(0.85);
   `control_time.goalkeeper.jump_scale_at_max_reach`(1.2, renamed
   `gk_jump_scale_at_max_reach` for clarity vs the new outfield pair),
   `control_time.goalkeeper.max_reach_height_m`(2.2, renamed
   `gk_max_reach_height_m`); `control_time.outfield_jump_scale_at_max_reach`
   (2.0), `control_time.outfield_max_reach_height_m`(2.0).
4. Tests:
   - **Early-intercept targeting**: close/slow shot (e.g. struck from just
     outside the box at moderate pace) → assert the computed save target is
     an intercept point strictly closer to the GK's current position (and
     ahead of the existing goal-line-margin plane) than
     `save_target_position`'s unmodified result; far/fast shot (e.g. a
     powerful strike from distance) → assert it falls back to the existing
     goal-line-plane target unchanged (regression — reuse existing save
     tests/balance tests unmodified as a guard against perf/behavioural
     regression on the common case).
   - **Early-intercept boundary**: a shot exactly at
     `early_intercept_max_distance_m` (10.0m) and one just beyond it, to
     confirm the distance gate itself is correctly enforced (not just the
     timing-margin gate).
   - **Early-intercept false-positive guard**: a shot where
     `t_reach_intercept` is only marginally less than
     `t_reach_goal_line*0.85` (i.e. right at the safety-margin boundary) —
     confirm behaviour doesn't flicker between intercept-target and
     goal-line-target across consecutive ticks as the ball moves (a
     regression/stability check analogous to Phase A's oscillation check).
   - **Control-time height tables**: table of GK-in-box control time vs ball
     height at `[0, 0.49, 0.95, 1.8, 2.0, 2.2, 2.4]`m AND a parallel table
     for outfield control time vs ball height at
     `[0, 0.49, 0.95, 1.8, 1.9, 2.0]`m (outfield's lower ceiling); assert
     both strictly monotonic increasing, with a visibly steeper slope above
     1.8m than below for both curves (confirms the "jump zone" is actually
     harder, not just continuing the old curve); assert GK time is lower
     than outfield time at every shared height sample above 1.8m (GK jump
     advantage, matching the worked example above); assert outfield values
     at/below 1.8m are byte-for-byte identical to the pre-Phase-C
     `height_difficulty_factor` output (hard regression check, not just
     "reasonable").

**Relevant files**: [src/footballcoach/engine/goalkeeping.py](src/footballcoach/engine/goalkeeping.py),
[src/footballcoach/engine/possession.py](src/footballcoach/engine/possession.py),
[src/footballcoach/engine/match.py](src/footballcoach/engine/match.py) (`_intercept_target`,
`SaveOrder` branch), config additions above.

## Phase D — Collision physics: velocity damping (depends on nothing)
1. In `engine/collision.py::resolve_player_overlap`, after computing
   `direction` (collision normal) and doing the existing position push-apart,
   also damp the closing velocity component for BOTH players: for each
   player, `closing_component = velocity_xy.dot(direction_towards_other)`.
   **Revised per user feedback (85% was too aggressive, and this runs every
   tick while overlap persists at 30Hz so effects compound)**:
   - Only damp if `closing_component > collision_damping_min_closing_speed_mps`
     (new config, floor ~0.3 m/s) — below this, leave velocity untouched
     entirely, so players can still gently nudge/jostle each other at low
     relative speed without being fought by the damping every tick.
   - Above the floor, reduce the closing component to
     `closing_component * collision_velocity_retention` with
     `collision_velocity_retention = 0.5` (50% reduction, down from the
     original 85% proposal), leaving the tangential component untouched, and
     reconstruct velocity (preserve z).
   - Because this reapplies every tick of sustained overlap, note in code
     comments that the effective per-tick damping compounds (e.g. two ticks
     of overlap ≈ 0.5*0.5=25% of original closing speed retained) — flagged
     as intentional (represents continuous "pushing against" contact, not a
     one-off bounce) but something to sanity-check visually/in balance tests
     for over-aggressiveness, and easy to soften further via the config
     value alone if it still feels too strong once tested.
   - Applies to ALL overlapping pairs including ones where a player
     `is_inactive` (the existing position-push-apart skip for inactive pairs
     is unchanged — inactive players still positionally "run through" per
     current design — but the velocity damping is NOT skipped for inactive
     pairs, otherwise a just-tackled player would keep gliding at full speed
     into the opponent; flagged as a deliberate deviation from the existing
     inactive-skip pattern, confirm visually in UI once built).
2. New config section `physics.json["collision"]`:
   `collision_velocity_retention`(0.5),
   `collision_damping_min_closing_speed_mps`(0.3).

   **Worked numeric example (compounding check)**: two players closing at
   3.0 m/s (a brisk jog-into-jog collision, well above top-speed sprint
   closing speeds being unrealistic for this check since players would
   normally start avoiding via Phase A well before contact — 3.0 m/s
   represents a case where avoidance failed/wasn't active, e.g. a scripted
   test or an off-ball scramble). At 30Hz (`dt=1/30s`), if the pair remains
   in overlap for `n` consecutive ticks before `resolve_all_overlaps`'
   position push-apart separates them again: `closing_component_after_n =
   3.0 * 0.5^n` (once above the 0.3 m/s floor each time) → after 1 tick:
   1.5 m/s; after 2 ticks: 0.75 m/s; after 3 ticks: 0.375 m/s; after 4
   ticks: 0.1875 m/s (now below the 0.3 floor — damping stops applying,
   remaining velocity left as-is). So in practice the compounding
   self-limits within about 3-4 ticks (~100-130ms) purely because the floor
   catches it — it will NOT asymptote all the way to zero, it bottoms out
   at whatever's below `collision_damping_min_closing_speed_mps` and stays
   there. This is the key non-degeneracy property the tests must verify
   numerically (see below), and is worth stating explicitly in code
   comments since it wasn't obvious before doing the arithmetic — the floor
   isn't just a "skip gentle jostling" feature, it's also what prevents the
   compounding from being unbounded.
3. Tests:
   - **Head-on closing-velocity reduction**: two players moving directly at
     each other at a fixed closing speed (e.g. 3.0 m/s per the worked
     example) — assert first-tick post-collision closing component is
     reduced to `~0.5x` (within numeric tolerance) of the pre-collision
     value.
   - **Non-degenerate multi-tick floor behaviour**: same setup, run for 5+
     ticks of sustained overlap — assert closing velocity follows the
     `3.0*0.5^n` pattern until it drops below
     `collision_damping_min_closing_speed_mps` (0.3), then assert it stops
     decreasing further that tick (floor reached, matching the worked
     example's ~4-tick bottom-out) — this directly tests the compounding
     concern raised in review isn't degenerate/unbounded.
   - **Floor boundary**: closing speed just above (e.g. 0.35 m/s) and just
     below (e.g. 0.25 m/s) the 0.3 m/s floor — confirm damping applies only
     in the former case.
   - **Tangential-only control (regression)**: two players moving
     side-by-side in parallel (zero closing component, pure tangential
     motion) — assert velocities are completely unaffected by the new
     damping code path.
   - **Inactive-pair damping applies**: a just-tackled (`is_inactive`)
     player still moving toward another player — assert the new velocity
     damping still applies (per the deliberate deviation from the
     position-push-apart skip) while confirming position push-apart is
     still skipped for that pair (both behaviours coexist correctly, not
     accidentally reusing one code path for both).
   - **Head-on-tackle-then-overlap stability**: reuse/extend an existing
     head-on-tackle scenario test — assert no infinite bounce/jitter over
     an extended run (e.g. check velocity magnitude settles/decays rather
     than oscillating indefinitely across 60+ ticks).
   - **Kick/pass unaffected**: a player kicking or passing while overlapping
     another player — confirm ball velocity itself is untouched by this
     player-velocity damping (scope check, since it'd be easy to
     accidentally reuse the same helper for ball-vs-player checks which are
     handled separately by `resolve_ball_block_by_inactive_players`).

**Relevant files**: [src/footballcoach/engine/collision.py](src/footballcoach/engine/collision.py),
[src/footballcoach/engine/match.py](src/footballcoach/engine/match.py) (`_check_head_on_tackles`
ordering relative to `resolve_all_overlaps`).

## Phase E — Running-direction precision penalty on ALL kicks (shots, passes,
generic kicks — user confirmed passes are just kicks too, no shots-only
restriction) (depends on nothing)
1. New function in `engine/kicking.py`:
   `running_direction_precision_multiplier(kicker_velocity_xy, aim_direction_xy,
   min_speed_mps) -> float`. If kicker's ground speed < `min_speed_mps` (1.0,
   config `running_direction_min_speed_mps`), return 1.0 (no effect — not
   meaningfully "running"). Else `cos_sim =
   normalize(kicker_velocity_xy).dot(normalize(aim_direction_xy))`:
   - `cos_sim >= 0.35` → multiplier `1.0`
   - `-0.2 <= cos_sim < 0.35` → linear from 1.0 to 0.75:
     `t=(0.35-cos_sim)/0.55; multiplier=1.0-0.25*t`
   - `cos_sim < -0.2` → linear from 0.75 to 0.25 (i.e. **-75%** at cos_sim=-1,
     resolved from the earlier -60%/-90% discussion — see worked numbers
     below): `t=(-0.2-cos_sim)/0.8; multiplier=0.75-0.50*min(t,1)`
2. Apply as `effective_kick_precision = kick_precision_attr * multiplier`
   uniformly wherever `angle_error_sigma_rad` (or the pass-equivalent sigma
   in `PassingParams`) is computed — i.e. inside the shared `_launch_ball`
   helper used by both `kick_ball` and `pass_ball`, so `ShootOrder`,
   `KickOrder`, and `PassOrder` all get it for free with one change (no
   shots-only restriction, per user correction that passes are kicks too).
   Requires threading kicker velocity + aim direction through to `pass_ball`
   call sites in match.py (kicker_velocity is already a `pass_ball` param;
   confirm aim direction is derivable there — for passes, aim direction is
   `target_position - kicker_position` normalized, already computed
   elsewhere in `_leading_pass_target`/pass handling).
3. New config `kicking` additions: `running_direction_precision_cos_high`
   (0.35), `running_direction_precision_cos_low`(-0.2),
   `running_direction_precision_penalty_mid`(0.25),
   `running_direction_precision_penalty_max`(0.75, **decided value** — see
   worked comparison below, settled between the earlier 60%/90% options),
   `running_direction_min_speed_mps`(1.0).

   **Why 0.35 / -0.2 as breakpoints**: `cos_sim=0.35` corresponds to
   `arccos(0.35)≈69.5°` off the run direction — i.e. up to roughly a
   70°-wide cone in front of the runner incurs no penalty at all (shooting
   "mostly forwards" relative to your run is fine, matching how a striker
   running through onto a pass and shooting in roughly the same direction
   shouldn't be punished). `cos_sim=-0.2` is `arccos(-0.2)≈101.5°` — just
   past square-on (90°) — so the steep -25%→-75% penalty zone only kicks in
   once you're shooting *backwards relative to your run* (e.g. cutting back
   against your own momentum), which is the genuinely awkward
   body-mechanics case (these are the user-facing angle thresholds directly
   from Idea2.md's spec, restated here in degrees for intuition since the
   spec was given as cosine values).

   **Combined effect on final sigma (corrected — an earlier draft of this
   doc miscalculated `0.28^0.87` and understated the effect; recomputed
   properly below across the three max-penalty values discussed, settling
   on -75% as the decided default)**: recall
   `angle_error_sigma_rad(precision) = base(0.0055) + scale(0.04)*
   (1-precision^0.87)`. With the new multiplier folded in as
   `effective_precision = precision * running_multiplier`, worked example
   at `precision=0.7` (a good but not elite kicker), baseline unaffected
   case (`cos_sim>=0.35`, multiplier=1.0): `0.7^0.87 = 0.7332` →
   `sigma = 0.0055 + 0.04*(1-0.7332) = 0.0055+0.04*0.2668 = 0.01617` rad.
   At a 15m shot, `positional_std ≈ distance*sigma_angle`
   ([engine/knowledge.md](src/footballcoach/engine/knowledge.md)) gives a
   baseline std of `15*0.01617 ≈ 0.243`m.

   | max penalty | `cos_sim=-1` multiplier | effective precision | sigma (rad) | std @15m (m) | vs baseline |
   |---|---|---|---|---|---|
   | (baseline, `cos_sim≥0.35`) | 1.00 | 0.700 | 0.01617 | 0.243 | 1.00x |
   | -60% (rejected — first draft, understated) | 0.40 | 0.280 | 0.03228 | 0.484 | 2.00x |
   | **-75% (DECIDED default)** | **0.25** | **0.175** | **0.03672** | **0.551** | **2.27x** |
   | -90% (considered, not chosen) | 0.10 | 0.070 | 0.04154 | 0.623 | 2.57x |

   Full breakpoint table for the decided **-75%** default (same
   `precision=0.7`, 15m shot; mid-point at `cos_sim=-0.2` stays -25% per the
   original ask, unaffected by which max-penalty value is chosen since only
   the `-0.2→-1.0` segment's steepness changes):

   | `cos_sim` | multiplier | effective precision | sigma (rad) | std @15m (m) | vs baseline |
   |---|---|---|---|---|---|
   | ≥0.35 | 1.00 (no penalty) | 0.700 | 0.01617 | 0.243 | 1.00x |
   | -0.2 (mid) | 0.75 (-25%) | 0.525 | 0.02267 | 0.340 | 1.40x |
   | -0.6 | 0.50 (interpolated) | 0.350 | 0.02945 | 0.442 | 1.82x |
   | -1.0 (max) | 0.25 (-75%) | 0.175 | 0.03672 | 0.551 | 2.27x |

   The relationship is nonlinear throughout because of the `precision^0.87`
   power law in `angle_error_sigma_rad`: as `effective_precision` drops
   toward 0, `sigma` accelerates toward its ceiling of `base+scale =
   0.0455` rad (the worst-case sigma at precision=0, i.e. an unskilled
   kicker), so cutting precision by a fixed percentage matters more/less
   depending on where you start on that curve — this is why 60%→75%→90%
   (each a further 15-point cut) produces shrinking sigma gains
   (2.00x→2.27x→2.57x, i.e. +0.27x then +0.30x) rather than equal steps.
   **-75% was chosen** as a middle ground: meaningfully more punishing than
   the original (rejected) -60% draft (2.27x vs 2.00x baseline dispersion
   at worst case) while leaving more headroom below the ~0.0455 rad
   ceiling than -90% does, so the `kick_precision` attribute still has some
   room left to matter even in the worst-case running-backwards scenario.
4. Tests:
   - **Multiplier table (unit)**: `cos_sim -> multiplier` at sample points
     `[1.0, 0.5, 0.35, 0.2, 0.0, -0.2, -0.6, -1.0]` matching the piecewise
     formula exactly (including the two breakpoints landing exactly on
     1.0 and 0.75 respectively), exercised through both `kick_ball` and
     `pass_ball` call paths (same helper, two callers — confirm both
     actually invoke it, not just one).
   - **Below min-speed gate (regression)**: kicker ground speed below
     `running_direction_min_speed_mps` (1.0) with an extreme aim angle
     (directly backwards relative to a near-zero velocity vector, an
     edge case where `normalize(kicker_velocity_xy)` is degenerate/
     ill-defined at v≈0) — assert multiplier is always 1.0 and no
     division-by-zero/NaN occurs when velocity is exactly zero.
   - **Sigma composition (unit)**: reproduce the worked example above
     numerically (precision=0.7, cos_sim unaffected vs cos_sim=-1) and
     assert `angle_error_sigma_rad` output matches within tight tolerance.
   - **Balance — shooting**: shooter running toward goal (cos_sim>0.35) vs
     square-across (cos_sim≈0) vs directly away (cos_sim=-1), same shot
     distance/power/precision otherwise — report and assert strict goal-rate
     ordering (toward > across > away) and report dispersion stats
     (std-dev of miss distance) confirming the ordering holds there too, not
     just in the pass/fail goal count (report full stats per repo
     convention).
   - **Balance — passing**: same three-condition setup for a passer aiming
     at a stationary teammate, using the existing pass-accuracy/completion
     metric already used by `tests/balance/test_pass_balance.py` — confirm
     the same toward > across > away ordering holds for passes too (this is
     the main test proving the "no shots-only restriction" requirement is
     actually honoured end-to-end, not just wired in code).
   - **Interaction with existing running-power effect (regression)**:
     confirm `running_power_multiplier` (the pre-existing, separate
     power-only effect) and the new precision-only multiplier don't
     interfere — e.g. a fast run directly toward goal should still get the
     existing power boost with no precision penalty (both at their
     "favourable" ends simultaneously), verified via an existing
     kicking-power unit test extended to also assert precision is
     untouched in that specific case."}]

**Relevant files**: [src/footballcoach/engine/kicking.py](src/footballcoach/engine/kicking.py)
(`angle_error_sigma_rad`, `kick_ball`), [src/footballcoach/actions.py](src/footballcoach/actions.py)
(`shoot`), [src/footballcoach/engine/match.py](src/footballcoach/engine/match.py) (Shoot/Kick
order branches — need kicker velocity + aim direction available there).

## Phase F — Mark order (depends on Phase B's shared tackle-fallback helper being clean;
otherwise independent — can run parallel with A-E)
1. Factor existing `GetPossessionOrder` per-tick chase/tackle logic in
   `match.py` into a reusable helper, e.g. `_run_get_possession_behaviour(
   player, dt) -> None`, called by both `GetPossessionOrder` handling and the
   new `MarkOrder` fallback (avoid duplicating chase/intercept/tackle code).
2. New `MarkOrder(target_player_id: str, status=OrderStatus.PENDING)` in
   [orders.py](src/footballcoach/orders.py).
3. New `_process_orders` branch: resolve `target_player` by id. If
   `target_player` currently has ball possession OR the ball is within
   `mark_intercept_radius_m` (4.0m, per user — updated from initial 3.0m
   proposal) of the marker → run `_run_get_possession_behaviour` (chase
   ball/tackle target). Otherwise, compute defensive standoff position:
   `mark_pos = target.position + normalize(ball.position - target.position)
   * mark_standoff_m` (1.5, config) — i.e. a point between the target and
   the ball, offset from the target toward the ball — and move there via
   `step_player_towards` (sprinting).
4. New config section `physics.json["marking"]`: `mark_intercept_radius_m`
   (4.0), `mark_standoff_m`(1.5).
5. New action helper in `actions.py`: `mark(player, target: Player) -> None`
   sets `player.current_order = MarkOrder(target_player_id=target.player_id)`.
6. Tests: marker stays near/between target-and-ball over several ticks with
   no possession change (position within some tolerance of the ideal
   between-point). Marker switches to chase/tackle when target gains
   possession. Marker switches to intercept when ball passes within
   `mark_intercept_radius_m` (4.0m) even without target possession (e.g. a
   stray loose ball rolling near the marker) — include a boundary case just
   inside/outside 4.0m. Balance tests (per user, explicitly requested): (a)
   marking reduces the marked attacker's effective time-on-ball / pass
   completion rate in a simple 2-player scenario vs an unmarked control,
   reporting full stats per repo convention; (b) report marker's average
   distance-to-ideal-standoff-point over a sustained passage of play (sanity
   check that the standoff behaviour is stable, not oscillating/chasing its
   own tail); (c) tackle/interception success rate for the marker once it
   switches to `_run_get_possession_behaviour`, compared against the
   existing `GetPossessionOrder` balance baseline (should be consistent,
   since it's the same shared logic).

**Relevant files**: [src/footballcoach/orders.py](src/footballcoach/orders.py),
[src/footballcoach/engine/match.py](src/footballcoach/engine/match.py) (`_process_orders`,
existing `GetPossessionOrder` branch, `_intercept_target`), `actions.py`.

## Phase G — Visual indicators + Game log (UI-only, depends on nothing engine-side
except reading existing Player/Ball state; do after A-F land so states like
CONTROLLING_BALL-with-penalty exist, but not a hard blocker)
1. **Inactivity / control-delay outlines**: in
   [ui/renderer.py](src/footballcoach/ui/renderer.py)`::draw_player`, add a
   distinct outline ring for `PlayerState.CONTROLLING_BALL` (new style colour
   `CONTROL_DELAY_OUTLINE`) separate from the existing white
   `POSSESSION_OUTLINE`, and add an explicit outline ring (not just
   translucency) for `PlayerState.INACTIVE_TACKLED` (new `INACTIVE_OUTLINE`
   colour) so the two penalty states read clearly at a glance. Keep it a
   simple full ring (no progress arc/countdown — avoid over-engineering).
   New constants in [ui/style.py](src/footballcoach/ui/style.py).
2. **Ball state indicator**: add `Ball.just_bounced_timer_s: float = 0.0` in
   [entities/ball.py](src/footballcoach/entities/ball.py); in
   `engine/ball_physics.py::step_ball`'s real-bounce branch, set it to
   `just_bounced_display_duration_s` (0.3, new config
   `ball_physics.just_bounced_display_duration_s`); decay by `dt` each tick
   (in `step_ball` itself, floored at 0). Rendering classification (derived,
   no new stored enum needed): "flying" if `position.z > 0.05` and not
   possessed; "rolling" if `position.z <= 0.05` and `velocity.length_xy() >
   0.05` and not possessed; "just bounced" if `just_bounced_timer_s > 0`
   (drawn as an overlay regardless of the other two). Add thin outline rings
   in `draw_ball` with new style colours `BALL_STATE_FLYING_OUTLINE`,
   `BALL_STATE_ROLLING_OUTLINE`, `BALL_STATE_BOUNCED_OUTLINE`.
3. **Game log**: new `ui/gamelog.py`: `LogLevel` enum (`INFO`, `DEBUG`);
   `GameLog` class wrapping a `collections.deque(maxlen=50)` of
   `(time_s, level, message)`; `.add(level, msg)`. Engine hook: add
   `Match.log_callback: Callable[[LogLevel, str], None] | None = None` field
   (default `None` → zero cost in tests/headless use), call it from
   `match.py` at: `_complete_control` (possession taken), kick/pass/shoot
   execution (include aim/power params at DEBUG, short summary at INFO),
   `_check_goal` (goal scored, INFO). Order-given logging happens in
   `ui/input.py` at the point orders are assigned (not in `actions.py`,
   so tests never pay for it). `App` wires `match.log_callback =
   self.game_log.add` when starting a match/scenario. New renderer function
   `draw_game_log(surface, entries, min_level, rect)` bottom-corner scrolling
   box (fixed rect, newest at bottom, simple text stack — no interactive
   scrollbar). New hotkey `L` in `app.py` cycles `min_level` (INFO/DEBUG).
   **Tackle logging (new, explicitly requested)**: every call site that
   resolves `attempt_tackle` (`TackleOrder`, `ChaseTackleOrder`,
   `GetPossessionOrder`/`_run_get_possession_behaviour`, `MarkOrder`'s
   fallback, `_check_head_on_tackles`) must emit a log entry via
   `log_callback` immediately after the `TackleResult` is known, capturing:
   tackler id/team, dribbler id/team, both effective attribute values fed
   into the roll (i.e. `tackling_attr * effective_boost` and
   `dribbling_eff` post any Phase B control-time penalty), the actual rolled
   values (`tackler_roll`, `dribbler_roll` from `TackleResult` — extend
   `TackleResult` with these two fields if not already exposed, since
   `skill_roll`'s internal draw is otherwise invisible outside
   `attempt_tackle`), any modifiers applied (angle modifier value,
   GK-outside-box penalty, GK-in-box-untackleable short-circuit, dribble
   roll's control-time penalty fraction) and the outcome (won/lost) plus
   the resulting state transition (dispossessed + inactive, or tackler
   missed + inactive). DEBUG level gets the full numeric breakdown (all
   rolls/modifiers), INFO level gets a one-line summary (e.g. "Player A
   tackled Player B" / "Player A's tackle on Player B failed"). This is the
   single richest log line type in the system and doubles as a manual
   sanity-check tool for the Phase B tackle changes (GK box penalty,
   untackleable-GK-in-box, control-time dribble penalty) without needing to
   re-run balance tests — encourages checking `TackleResult` shape now so
   it carries everything needed instead of being re-derived ad hoc at each
   log call site. Also log: a miss/near-miss shot going out of play or wide
   (INFO), and stamina hitting empty/near-empty for a sprinting player
   (DEBUG only — chatty, not interesting at INFO level).
4. Tests: unit test `just_bounced_timer_s` set on real bounce (above
   `BOUNCE_THRESHOLD_MPS`) and not on settling contact; decays to 0 over
   expected ticks. Unit test `GameLog.add`/deque eviction at maxlen. Unit
   test that a tackle (both a win and a loss case, deterministic via
   `rng_reduction=1.0`) produces exactly one log entry containing the
   tackler/dribbler ids, both rolled values, and the correct outcome text;
   a GK-outside-box tackle log entry includes the penalty modifier value;
   a GK-in-box-with-ball tackle attempt logs the auto-fail short-circuit
   distinctly (not a normal roll). No balance tests needed for the
   presentational parts (purely presentational), but the tackle-logging
   plumbing is exercised for free by Phase B's existing balance tests once
   `log_callback` is wired (assert no exceptions when a callback is
   attached, since balance tests run thousands of trials and would surface
   any per-call overhead or bugs immediately).

**Relevant files**: [src/footballcoach/ui/renderer.py](src/footballcoach/ui/renderer.py),
[src/footballcoach/ui/style.py](src/footballcoach/ui/style.py),
[src/footballcoach/entities/ball.py](src/footballcoach/entities/ball.py),
[src/footballcoach/engine/ball_physics.py](src/footballcoach/engine/ball_physics.py),
[src/footballcoach/engine/match.py](src/footballcoach/engine/match.py),
[src/footballcoach/ui/app.py](src/footballcoach/ui/app.py),
[src/footballcoach/ui/input.py](src/footballcoach/ui/input.py) (new `ui/gamelog.py`).

## Phase H — UI Scenarios: trim to 4, add randomization + params, add 2v2 and
1v2, and expose adjustable parameters for playtesting **on every scenario**
(depends on Phase F only for the optional Mark-based defender variant, which
is out of scope for the base 2v2/1v2 per literal spec — GetPossessionOrder is
used for the defender in both; otherwise independent of A-G, though visual
indicators from Phase G will show up naturally in these scenarios once done)

0. **Generic per-scenario adjustable-parameters system (new scaffolding,
   needed before any individual scenario's params can be exposed in the
   UI)**: today `build_close_range_save_scenario`'s randomization is
   hardcoded inside the function with no way for a user to change ranges
   without editing code — per user request this needs to become a general,
   reusable mechanism used by **all** scenarios (shoot, tackle, sprint,
   pass, 2v2, 1v2), not just shoot.
   - New `ScenarioParam` frozen dataclass in
     [ui/scenarios.py](src/footballcoach/ui/scenarios.py): `name: str` (kwarg
     name passed to the `build_*` function), `label: str` (UI display text),
     `min_value: float`, `max_value: float`, `step: float`, `default: float`.
   - Extend `ScenarioDefinition` with `params: list[ScenarioParam] =
     field(default_factory=list)` and change `build`'s type to
     `Callable[..., Match]` (accepting `rng_reduction` plus `**kwargs`
     matching each `ScenarioParam.name`, all with defaults equal to today's
     hardcoded values — so calling `build(rng_reduction)` with zero extra
     kwargs is unchanged/backward-compatible, satisfying the existing
     `ScenarioLoop` auto-rebuild-on-trial-end call site with no changes
     needed there beyond passing through a params dict).
   - New `Screen.SCENARIO_PARAMS` state in
     [ui/app.py](src/footballcoach/ui/app.py): selecting a scenario with a
     non-empty `params` list from the menu goes here first (not straight to
     MATCH); renders one row per `ScenarioParam` (label, current value,
     `-`/`+` buttons stepping by `step`, clamped to `[min_value,
     max_value]`), plus "Start" (builds the scenario with
     `**current_values` and proceeds to MATCH) and "Back" (returns to MENU)
     buttons. `App` stores `self._pending_scenario_params: dict[str,
     float]`, seeded from each `ScenarioParam.default` when the screen is
     entered, mutated by `+`/`-` clicks. Scenarios with an empty `params`
     list (none, after this phase, but kept as an escape hatch) skip
     straight to MATCH as today.
   - New renderer function `draw_scenario_params(surface, params, values,
     rects)` — simple vertical list, reusing existing HUD text/button
     drawing helpers rather than inventing new widgets (no drag-sliders,
     just discrete +/- steps — avoids over-engineering a slider widget for
     a playtesting tool).
   - This item is a prerequisite for items 2-7 below, which each define
     their own scenario-specific `ScenarioParam` list.
1. In [ui/scenarios.py](src/footballcoach/ui/scenarios.py), remove
   `build_penalty_scenario`, `build_save_scenario` (far-post fixed) and
   `build_shoot_scenario` (no-keeper fixed) from `SCENARIOS`; keep only
   `save_close` (the "Close range mixed results" shooting scenario), `pass`,
   `tackle`, `sprint`, and add the two new `2v2` and `1v2` entries (item 6, 7).
2. **Shoot (`save_close`)** — params: `distance_min_m`/`distance_max_m`
   (default 8/16, matching current hardcoded range), `shooter_y_offset_m`
   (max abs offset, default 5), `shooter_precision_min`/`_max` (default
   0.65/0.85), `shooter_power_min`/`_max` (default matches current
   distance-scaled formula's bounds), `gk_skill_min`/`_max` (default
   0.65/0.85) — the ~5-10 params already scoped in the original plan,
   formalized here using the new `ScenarioParam` mechanism from item 0.
3. **Tackle** — randomize both player positions with separation uniformly in
   `[separation_min_m, separation_max_m]` (new params, default `[1, 10]`)
   instead of fixed 0.5m; give the attacker a random jog direction — assign
   a `MoveOrder` with `sprint=False` toward a far point in a random heading
   (so they're "jogging", not sprinting or standing still). Additional
   params: `tackler_tackling_min`/`_max` (default fixed at 0.8/0.8, i.e. a
   playtester can widen this to see a range of tacklers) and
   `dribbler_dribbling_min`/`_max` (default 0.6/0.6) — reusing the existing
   fixed balance-test reference values (0.8 vs 0.6) as the *default point*
   but exposing them as a widenable range for playtesting variety.
4. **Sprint** — replace the fixed straight 100m course with a random
   5-point waypoint course: each leg length `U(leg_min_m, leg_max_m)`
   (params, default `[5, 25]`) in a random 2D heading, rejecting/clamping
   waypoints that would leave the pitch (`pitch.is_in_bounds`) — resample
   the heading (bounded retries) if out of bounds. Runner attribute level
   exposed as a single `runner_skill_min`/`_max` param (default 0.7/0.8,
   applied uniformly to top_speed/acceleration, matching the "decent
   runner" spec) rather than 16 separate per-attribute sliders — deliberate
   simplification since the literal ask only cares about overall pace tier.
   Needs waypoint-sequencing: extend `ScenarioLoop` (or a small
   scenario-local controller) to issue the next `MoveOrder` when the current
   one completes, reusing the same `on_tick` hook mechanism as 2v2/1v2 (see
   #6/#7).
5. **Pass** — randomize distance/position keeping the two players within
   `max_distance_m` (param, default 30); sample attributes normally via
   `generate_attributes`, then clamp `kick_precision`/`ball_control` (and
   any other relevant passing attrs) into `[attr_clamp_min,
   attr_clamp_max]` (params, default `[0.70, 0.80]`) only if outside that
   band (preserve existing values if already in range, per literal ask "if
   they aren't already").
6. **2v2 scenario** (new `build_2v2_scenario`): extend `ScenarioDefinition`
   with an optional `on_tick: Callable[[Match, int], None] | None = None`
   hook, invoked by `ScenarioLoop.step()` before `match.step()` each tick
   (all other scenarios pass `None`, zero behavior change). Setup: attacker A
   (ball) in the box aligned with left post, attacker B slightly behind in x
   (onside) aligned with right post, defender (`GetPossessionOrder`) and GK
   (`SaveOrder`). Scripted via a small `TwoVTwoController` (new class in
   scenarios.py) driving via `on_tick`: A starts with
   `PassOrder(target_player_id=B.id)` (leading pass); when `ball.possessed_by
   == B.player_id` is first observed, either issue `ShootOrder` immediately
   or a short forward `MoveOrder` + control + `ShootOrder` afterward
   (50/50 random choice, or a config flag) — B given a forward `MoveOrder`
   toward goal throughout so "running forwards" per spec. Params:
   `attacker_skill_min`/`_max` (default 0.7/0.85), `defender_skill_min`/
   `_max` (default 0.55/0.7), `gk_skill_min`/`_max` (default 0.55/0.7),
   `shoot_immediately_probability` (default 0.5).
7. **1v2 scenario (new, per user request)** — `build_1v2_scenario`: one
   attacker vs. one defender + one goalkeeper.
   - **Attributes (literal, not sampled)**: attacker gets all 8
     `PlayerAttributes` fields set to a single scalar (default **0.9** —
     "elite ... everything"); defender and GK each get all 8 fields set to
     a single scalar (default **0.55** — "average ... everything"). Exposed
     as three `ScenarioParam`s — `attacker_skill` (default 0.9),
     `defender_skill` (default 0.55), `gk_skill` (default 0.55) — each a
     single slider driving all 8 attributes uniformly (deliberate
     simplification vs. 24 individual sliders; consistent with the
     literal "X everything" phrasing in the request, and easy to widen
     later into per-attribute control if the user wants finer-grained
     playtesting).
   - **Random but sensible starting positions**: attacker start position
     sampled at distance `d0 ~ U(attacker_start_min_m, attacker_start_max_m)`
     (params, default `[18, 32]` — a "should credibly be able to attack and
     shoot" range, comparable to the existing shoot scenario's 8-16m but
     pulled back further since this scenario includes an approach run) from
     the goal along the pitch's long axis, with lateral offset `y0 ~
     U(-attacker_y_offset_m, +attacker_y_offset_m)` (param, default 12m,
     clamped so the position stays `pitch.is_in_bounds`). Defender placed
     along the straight line between the attacker's start and the goal
     centre, at a fraction `defender_line_fraction ~
     U(defender_fraction_min, defender_fraction_max)` (params, default
     `[0.3, 0.7]` — somewhere meaningfully between attacker and goal, not
     glued to either), with a small perpendicular jitter
     `U(-defender_jitter_m, +defender_jitter_m)` (param, default 2.0m) so
     the defender isn't always exactly on-line (more realistic/varied). GK
     placed at goal centre with a small lateral jitter (reuse the existing
     `goalkeeping.default_position_fraction_of_half_length`-style default
     positioning already used elsewhere, plus a small random y jitter,
     param `gk_start_jitter_m`, default 1.0m). **Rejection check**: if
     initial defender-attacker or defender-GK separation would be below a
     minimum sensible gap (e.g. `2*player.radius_m + 0.5`), resample the
     random draw (bounded retries, same pattern as Phase H item 4's
     waypoint rejection) rather than allowing a degenerate immediate-tackle
     start.
   - **Attacker behaviour**: on scenario start, compute
     `move_fraction ~ U(move_fraction_min, move_fraction_max)` (params,
     default `[0.10, 0.50]` — directly from the user's literal spec) of
     the *current* straight-line distance to goal; issue a `MoveOrder`
     (sprinting) to the point that fraction of the way from the attacker's
     start position toward the goal centre. Via the shared `on_tick` hook
     (same mechanism as 2v2/sprint waypointing), once that `MoveOrder`
     completes, issue a `ShootOrder` at the goal. No control-then-shoot
     branch (unlike 2v2) — per the literal ask this scenario is just
     move-then-shoot.
   - **Defender/GK behaviour**: defender's `current_order =
     GetPossessionOrder()` from tick 0 (chases the ball/attacker
     immediately, per literal ask); GK's `current_order = SaveOrder()` from
     tick 0.
   - **Outcome tracking note**: the existing `ScenarioLoop.outcomes` dict
     (`{"goal","saved","miss","other"}`) has no category for "defender won
     the ball before a shot was ever taken", which is a plausible and
     interesting outcome for both 1v2 and 2v2 given `GetPossessionOrder`
     chases proactively — recommend adding a new `"dispossessed"` outcome
     bucket (ball repossessed by the non-attacking side while no shot is
     in flight) to `ScenarioLoop`'s outcome detection, benefiting both new
     scenarios and giving richer stats for playtesting than lumping it into
     `"other"`.
8. New/updated tests: `tests/scenario` or a light UI-adjacent test for
   waypoint sequencing (5-point course completes all waypoints, stays
   inbounds); a similar test for 1v2's random-position rejection logic
   (asserts no degenerate immediate-tackle start across many seeds, and
   that positions always stay in-bounds); a test that
   `ScenarioParam`-driven `build_*` calls with explicit kwargs produce
   different results than the defaults (proves the params actually flow
   through, not just cosmetically defined) for at least shoot, tackle, and
   1v2; existing balance tests referencing removed scenarios (if any)
   updated to call the underlying engine functions directly instead of the
   removed `build_*` UI helpers (balance tests in `tests/balance/` already
   test the underlying mechanics directly per repo convention, not via UI
   scenarios, so should be unaffected — verify during implementation).

**Relevant files**: [src/footballcoach/ui/scenarios.py](src/footballcoach/ui/scenarios.py),
[src/footballcoach/ui/app.py](src/footballcoach/ui/app.py) (menu list, `_menu_items`,
new `Screen.SCENARIO_PARAMS`), [src/footballcoach/ui/renderer.py](src/footballcoach/ui/renderer.py)
(new `draw_scenario_params`), [src/footballcoach/generation/attributes.py](src/footballcoach/generation/attributes.py)
(`generate_attributes`), `orders.py`, `actions.py`.

## Cross-cutting decisions (confirmed with user, incl. 2nd round of feedback)
- **Running-direction precision max penalty settled at -75%** (Phase E),
  after a round of "60% doesn't feel punishing enough" → corrected-math
  review → considering 90% → landing on 75% as a deliberate middle ground.
  At `precision=0.7`, worst case (`cos_sim=-1`) now gives ~2.27x baseline
  angular dispersion (vs 2.00x at the rejected -60% draft, vs 2.57x at the
  considered-but-not-chosen -90%) — see full worked tables in Phase E.
- **Scenario playtesting parameters generalized to ALL scenarios, not just
  shooting** (per explicit user request): new reusable `ScenarioParam`/
  `Screen.SCENARIO_PARAMS` mechanism (Phase H item 0) used by shoot, tackle,
  sprint, pass, 2v2, and the new 1v2 scenario — each scenario defines its
  own small set of the "main 5-10 interesting" params (not every random
  number), consistent with the original shoot-only ask's spirit extended
  repo-wide.
- **New 1v2 scenario added** (per explicit user request, not from
  Idea2.md): elite attacker (all 8 attributes = 0.9) vs. average defender +
  GK (all 8 attributes = 0.55 each), random-but-sensible starting positions
  (rejection-sampled to avoid degenerate immediate-tackle starts), attacker
  moves `10-50%` of the way to goal then shoots, defender on
  `GetPossessionOrder`, GK on `SaveOrder` — see Phase H item 7 for full
  derivation of position-sampling ranges and rationale for single-scalar
  skill sliders (vs. 24 individual attribute sliders).
- **Repulsion is an AI/order-layer mechanic, not an engine mechanic** (per
  explicit user correction): lives in a new top-level `steering.py`, called
  only from `MoveOrder` handling in `match.py::_process_orders`;
  `engine/movement.py` (`step_player_towards`) and `engine/collision.py`
  remain unmodified and unaware repulsion exists. This mirrors the existing
  engine/AI boundary elsewhere in the repo (e.g. `_intercept_target`/
  `_leading_pass_target` are also order-layer prediction logic that feed
  plain targets into unmodified engine functions).
- Repulsion orthogonal-nudge side: simple 2D cross-product sign (not
  distance-to-neighbors); add a balance test checking for oscillation near
  the threshold and add small hysteresis only if it shows up in practice.
- Mark order falls back to GetPossession/tackle logic when target has
  possession OR ball comes within `mark_intercept_radius_m` = **4.0m**
  (revised from initial 3.0m proposal) of the marker; balance tests
  explicitly requested for the marking behaviour (see Phase F #6).
- GK jump penalty reuses the existing `control_time_s` height-factor curve,
  extending the scaling continuously above head height so higher shots are
  strictly harder (not just re-flattened). **Extended (per user, beyond
  original GK-only ask) to outfield players too**: outfield players can
  also "jump" for high balls, with a lower max reach height (2.0m vs GK's
  2.2m) and a steeper per-metre penalty (scale 2.0 vs GK's 1.2) — GK still
  strictly favoured. Tests added for both GK and outfield jump curves incl.
  a below-1.8m regression check for outfield.
- `Player.control_time_total_s` was proposed then **dropped** — user
  correctly pointed out `Player.state_timer_s` already holds remaining
  control time directly while `CONTROLLING_BALL`; the dribble-tackle penalty
  now scales off that absolute remaining-seconds value against a new
  reference constant (`control_time_penalty_reference_s`≈0.3) instead of a
  stored fraction-of-total.
- New, previously-unlisted rule folded into Phase B: goalkeepers cannot be
  tackled at all while in possession inside their own box (all tackle
  attempts against them auto-fail).
- Running-direction precision penalty (Phase E) applies to **all kicks**
  (shots, passes, generic kicks) — user corrected the initial "shots only"
  scoping since passes use the same underlying kick mechanics; implemented
  once in the shared `_launch_ball` helper so `ShootOrder`/`KickOrder`/
  `PassOrder` all get it together, with balance tests for both shooting and
  passing accuracy-vs-running-direction.
- Collision velocity damping (Phase D) **revised down significantly** after
  user feedback that 85% reduction was too aggressive given it reapplies
  every tick of sustained overlap at 30Hz: now 50% reduction
  (`collision_velocity_retention`=0.5) and only above a minimum closing-speed
  floor (`collision_damping_min_closing_speed_mps`=0.3) so gentle
  side-by-side jostling/nudging isn't fought every tick. Still applies even
  to pairs involving an `is_inactive` player (deviates from the existing
  position-push-apart inactive skip) — flagged as a deliberate design choice
  to confirm once visually tested. Tests explicitly check the damping isn't
  degenerate over multiple ticks (doesn't asymptote to a near-total freeze).

## Further considerations
1. Several new numeric constants (repulsion strength/nudge magnitude, GK
   jump scale, early-intercept margins) are first-guess placeholders
   explicitly meant to be tuned via the balance-test stats-reporting
   convention already used throughout the repo — expect a follow-up tuning
   pass after initial implementation, same as the existing penalty-corner
   angle_error constants that are "at the edge of their target band".
2. Phase H's 2v2 `on_tick` hook is new scaffolding not present today —
   recommend implementing it minimally (single optional callback) rather
   than a full scripting DSL, since only 2v2 needs it for now (sprint's
   waypoint sequencing can likely reuse the same hook rather than inventing
   a second mechanism).
