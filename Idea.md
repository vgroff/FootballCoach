# Football Coach

I want to do a personal project. The idea is as follows: a football coaching game, where the footballers are trained with neural networks to act idependently, but which also allows the coach (the human player) to influence their decisions and "coach" them (and also set the formation etc...). 

Do ask questions, don't assume design decisions without my input

# The plan for now

Build the football engine. Neural networks trained with PPO will eventually control the players, we will go into the details later, but obviously we should build it with that in mind. It should be a 2D top down view like the old football manager games, though it should actually be simulating in 3D. The graphics should obviously be separated from the football engine, so that the training can happen without the UI.

# Language and tools

I'm thinking Python with pygame for the 2D stuff. I know there's a bunch of libraries in python that can do PPO, I've only used Keras, not sure if it can do it, but happy to go with whatever. Using uv as the package manager. 


# The engine

As I say, though I want it displayed top-down 2D, I do want the whole simulation in 3D. It will be heavily simplified. The players are represented as cylinders, and the as a sphere. When a player has possession, he only loses it by being tackled or by kicking it away, otherwise the ball is stuck to him. Make the ball change size with height (but maybe exagerate the effect) and have a small number on it showing it's heigh in metres

Player attributes:
- top speed
- acceleration
- stamina
- kick precision
- kick power
- dribbling
- ball control
- tackling

These are all set between 0 and 1. I expect to generate the values on some kind of Gaussian distribution centered at 0.5, with a sigma of 0.2 or so - it'll be more intelligent than that because certain things will be correlated. But for the purpose of intuition, I want skills above 0.9 to be very rare, and I want first-team premier league type players to be represented in the 0.6-0.85 sort of area.

Actions the players can take:
- Move
    - in a direction
    - jog or sprint (with some acceleration/deceleration)
        - Changing directions - the bigger the change the more it should reduce speed/acceleration
        - losing stamina reduces top speed and acceleration by up to 65%
        - moving with the ball is slower than without - this is modulated by ball control skill but should never be the same even at 1.0 dribbling
        - changing directions with the ball is even more penalising that usual - but this is modulated by ball control, and at 1.0 the changing directions penalty is the same as without the ball
- Kick
    - power
    - direction
    - spin - do a simple approx to the Magnus effect, in all 3 dimensions
    - direction, spin and power are all gaussian-errored depending on how high kick precision is (they should never be exact even with 1.0 precision)
- Tackle
    - When a player is "touching" another player, they can do a tackle, at which point there is an RNG skill check between the dribbling and the tackling. Tackling always gets a 20% boost to it's attribute value in order to favour the defender (they can then roll higher than 1) - e.g. if a player has 0.5 tackling then their skill check roll is random.random()*1.2*0.5
- Delays
    - When a player is tackled, they are "inactive" for a bit, meaning they can't tackle and their speed and acceleration is reduced
    - Similarly, when a play comes into contact with the ball, the relative velocity between the ball and him, the speed at which he is moving, as well as how high the ball is (maybe piecewise this, above the knee is penalised (non-linearly) more than below it, and above the waist even more so), as well as his ball control skill, should determine how long it takes him to take control of the ball and be able to move
        - Goalkeepers behave differently - the penalty is much lower and the time to control is much lower in general, and the penalty for high balls is also lower, but only if they are in the box
    - The only exception to this is if the player immediately takes a shot instead of controlling the ball, in which case he shoots, but the relative velocity, his speed and the heigher of the ball all affect the gaussian errors on the shot (do it proportionally so that better shooters are affected less)

I'd like as many things as possible (e.g. pitch size, ball velocity etc...) to be stored in metres and seconds etc... SI Units wherever possible

Physics:
- Bouncing should reduce speed and spin
- The grass should have friction
- approximate things like air resistance and spin for now if we think they will either be complicated or computationally expensive
- Player circles can "overlap", but not too much, let's say the distance between the centre of the circles of any 2 players needs to be the radius. I wonder about the following - if this distance is violated, both spheres are re-adjusted to account for it, maybe weighted by velocity in that direction, which enables players to "push" each other while running alongside

Some assumptions:
- We always have 2 teams
- Always track goals
- By default have standard pitch size and goal size, but have it so that these can be changed


I'd also like a game option named "RNG reduction" - a number between 0 and 1. When this is enabled, two things happen:
- All skill checks are less random as follows, e.g. (rng_reduction + (1 - rng_reduction)*random.random()) * dribbling_skill
- All gaussian errors are reduced as follows sigma = sigma * (1 - rng_reduction)
- By default the rng_reduction is 0.3, so do things with that in mind

Another game option would be the offside rule, we will want it eventually but some training scenarios might not use it. Maybe we should just have a simplified rule where if the ball is passed and then goes near any player on your team that is past the last defender (and not behind the player with the ball), it's immediate offside whether or not the ball is actually passed to him. Something like that, the full rule can get quite complicated I think, we only really want the spirit of it



Computationally:
- We want this to be efficient because we want to train the neural networks on it by running many simulations. Don't compromise on the simulation too much, or at least parameterise this behaviour when possible so that it can be cranked up during actual play/fine-tuning, but where reasonably possible, lean on efficiency. Ask if unsure

Testing:
- Write unit tests as you go
- Also, I'd like some scenario "tests", where we try some basic stuff, e.g. stuff like:
    - players follow a move order correctly
    - player can kick the ball in the goal from the penalty spot (no goalkeeper) and the goal is recorded correctly
    - a player can tackle another player
    - for these you might want to set rng_reduction to 1.0 so that the tests pass reliably
- Balance scenarios:
    - For these have rng_reduction on 0.3
    - Test that a player with 0.5 kick accuracy and precision can score >95% of penalties without a goalie there kicking down the middle
    - Test that a player with 0.5 kick accuracy and precision scores <80% but >50% of penalties without a goalie there kicking bottom corner
    - Test that a player with 0.8 kick accuracy and precision can score >85% but <95% of penalties without a goalie there kicking bottom corner
    - Test that a player with 0.8 tackle can win a tackling challenge against a guy with 0.6 dribbling > 70% of the time but less than <90% of the time
    - Test that players with top speed and accel run the length of the pitch in an expected time, both with and without the ball
        - Test the same with bottom speed and accel 
    - Test that the changing direction mechanic also works sensibly in terms of time taken etc... (maybe have two players start from the same point running opposite directions and see how far away the first player gets by the time the second is moving in the right direction)
    - Test that stamina behaves reasonably
    - Basically test that all the players attributes behave somewhat reasonably in a similar way, using real-life scenarios
    - Remember that I want first-team premier league type players to be represented in the 0.6-0.85 sort of area. Players in League 3 would be in the 0.2-0.3 sort of area, so while there should be a meaningful difference between those scales, the League 3 players should still be competent footballers
    - Make it easy for me to tweak these in the code and ideally as to tweak the values in the code that are likely to affect them (e.g. the size of the gaussian errors, the max top speed etc...). Maybe expose them in a .json file or something? Whatever you think would be best
    - Expose these balance tests in the UI so that I can see them

UI/Graphics:
- Display the pitch with the goals on the left and right of the screen, draw the lines on the pitch for the box etc...
- The user should be able to play/pause the simulation, and also issue orders to players for move/kick/tackle, while paused. The players should follow that instruction until it's complete and return to their previous instructions
- On the UI I want the option to either play one of balance scenarios or to enter a training mode which has one player and the ball



### Note to AI

This stuff is implemented now, however I keep this file as useful for future reference. I'll also drop your plan.md as of 27th July 17:55, and there are READMEs.md and knowledge.md files around too

# Football Coach — Engine Milestone Plan

## Scope (Milestone 1)
Build the headless football simulation engine only: physics, movement, kicking,
tackling, possession/control, offside, unit tests, and statistical balance-test
suite. NO pygame UI, NO PPO/NN, NO career/training modes yet. Config-driven
constants in JSON so values are tunable without code changes.

## Confirmed decisions (from user Q&A)
- Milestone 1 = engine + physics + unit tests + balance tests only, no UI.
- Any number of players supported (not fixed 11v11); most tests use 1-3 players,
  but full XI tests welcome where useful.
- Fixed timestep dt = 1/30s, configurable.
- Stack: Python + uv, numpy for vectors, pytest for tests, dataclasses for
  entities. pygame deferred to milestone 2 (UI).
- Overlap rule = distance between player centers >= sum of radii (r1+r2).
- Offside = literal "last defender" (deepest opponent of any kind, GK included),
  not standard "second-last" rule — user's explicit simplification.
- Player-player push resolution weighted by current speed magnitude as a proxy
  for momentum (no explicit mass system exists).
- Turning balance-test scenario adjusted from "same start point opposite
  directions" (trivial since turning at v=0 is free) to "same initial top-speed
  velocity, one player reverses 180°, compare displacement over time" — needed
  to actually stress the turn-rate-limited-speed model.

## Core physics constants (derived/justified — see chat writeup for full derivation)
- Pitch 105m x 68m, goal 7.32m x 2.44m, box 16.5m x 40.32m, penalty spot 11m.
- Player: cylinder radius 0.3m, height 1.8m (visual only).
- Ball: radius 0.11m, mass 0.43kg.
- g = 9.81 m/s².
- Top speed: v_max(attr) = 5.0 + 4.5*attr (m/s), range 5.0–9.5.
- Acceleration: a_max(attr) = 2.5 + 5.0*attr (m/s²), range 2.5–7.5.
- Stamina multiplier: mult(S) = 1 - 0.65*(1-S), S in [0,1].
- Stamina drain (sprint, effort=1): base 1/90 s^-1 * (1.6 - 1.2*stamina_attr).
- Stamina regen (idle): base 1/60 s^-1 * (0.6 + 0.8*stamina_attr).
- Ball-carry top speed multiplier: 0.75 + 0.22*ball_control (never reaches 1.0).
- Turning: lateral accel capability a_lat = 4 + 4*accel_attr (m/s²);
  ω_max = a_lat / max(speed, ε); heading rotates toward target at ω_max*dt;
  with ball, a_lat_with_ball = a_lat*(1 - 0.6*(1-ball_control)).
- Kick direction/pitch error: σ_angle(precision) [rad] = 0.0107 + 0.0893*(1-precision)
  (isotropic, applied at both yaw and pitch; positional std at distance d ≈ d*σ_angle).
  Verified analytically against all 4 penalty/tackle balance targets (see chat).
- Kick power: v_ball_max(attr) = 15 + 20*attr (m/s), actual = v_ball_max*power_input.
- Tackle roll: (rng_reduction + (1-rng_reduction)*U(0,1)) * 1.2 * tackling_attr
  vs dribble roll: (rng_reduction + (1-rng_reduction)*U(0,1)) * dribbling_attr.
  At rng_reduction=0.3, tackling=0.8 vs dribbling=0.6 → P(win) ≈ 82.8% analytically
  (target range 70–90%, comfortable margin). Recommend N>=1000 trials per test.
- Ball drag: F = -0.5*ρ_air*C_d*A*|v|*v, ρ_air=1.225, C_d=0.25, A=π*0.11²=0.038m².
- Magnus: F = ρ_air*A*r_ball*C_L*(ω×v), C_L≈0.25 (tuned, informed by football
  aerodynamics literature).
- Ground bounce: vertical restitution e_v=0.6 (informed by FIFA ball-rebound
  lab spec sqrt(rebound/drop height)), horizontal retention e_h=0.8, spin decays
  *0.5 per bounce.
- Rolling friction: a_roll = μ_roll*g, μ_roll=0.06 (derived from ~5m/s ball
  rolling ~20m before stopping on grass).
- Attribute generation: correlated multivariate Gaussian (Cholesky of
  correlation matrix), base pop N(0.5, 0.2) clipped [0,1] defines "rarity"
  intuition (>0.9 rare, P(Z>2)=2.3%). Per-tier presets (mean,sigma) e.g.
  League3 mean 0.25, Premier League mean 0.72 sigma 0.08 truncated, so PL
  squads land ~0.6-0.85 with rare outliers. Correlations: top_speed~accel
  +0.6, kick_precision~ball_control +0.4, tackling~dribbling -0.2.
- Penalty scoring math (11m, no keeper, goal 7.32x2.44):
  - Center aim (h=1.1m, lateral=0): precision 0.5 → σ=0.55m at goal line →
    P(score)≈97% (target >95% ✓).
  - Corner aim (0.475m margin to post & ground): precision 0.5 → P≈65%
    (target 50-80% ✓); precision 0.8 → σ≈0.29m → P≈90% (target 85-95% ✓).

## File/module layout (to create)
- pyproject.toml (uv, deps: numpy, pytest)
- src/footballcoach/config/{attributes.json, physics.json}
- src/footballcoach/math/vector3.py
- src/footballcoach/entities/{player.py, ball.py, pitch.py}
- src/footballcoach/engine/{match.py, movement.py, kicking.py, tackling.py,
  ball_physics.py, possession.py, collision.py, offside.py}
- src/footballcoach/generation/attributes.py
- src/footballcoach/orders.py (Move/Kick/Tackle order state machine)
- tests/unit/..., tests/scenario/... (rng_reduction=1.0), tests/balance/...
  (rng_reduction=0.3, statistical assertions with N>=1000 trials)

## Implementation phases
1. Scaffold (pyproject/uv, package skeleton, config loading, vector math)
2. Entities + attribute generation/correlation
3. Movement & stamina (deterministic) + tests (sprint length, turning, stamina)
4. Ball physics (gravity/drag/magnus/bounce/rolling) + player-player collision
5. Kicking (power/direction/spin gaussian error + chaos factor under pressure)
6. Possession/control-time model (incl. goalkeeper + box special-casing) + tackling
7. Offside + goal detection/scoring
8. Balance scenario test suite matching derived probabilities above
(UI, PPO/NN training loop explicitly deferred to later milestones)

## Control-time-to-first-touch model (finalized, user approved deriving via intuition)
Height breakpoints (fractions of player height H=1.8m): knee_h=0.49m, waist_h=0.95m.
Height difficulty factor f(h), piecewise, convex/quadratic ramps (control points
f(0..knee_h)=1.0, f(waist_h)=2.0, f(H)=4.0, linear+capped beyond head):
- h <= knee_h: f=1.0
- knee_h<h<=waist_h: f = 1 + 1*((h-knee_h)/(waist_h-knee_h))^2
- waist_h<h<=H: f = 2 + 2*((h-waist_h)/(H-waist_h))^2
- h>H: f = 4 + 1*((h-H)/H), capped at 6

Velocity difficulty: linear in relative ball speed and player's own speed,
k1=0.15 s/m (relative vel), k2=0.05 s/m (own speed) — relative vel penalized
3x own speed since receiving a hard pass is harder than just jogging.

Ball control (bc) proportionally reduces difficulty (never fully to zero,
consistent w/ "never perfect" philosophy elsewhere): alpha=0.85.
  extra = (1 - alpha*bc) * [ (f(h)-1) + k1*v_rel + k2*v_player ]
  t_control = t_base + t_scale*extra,  t_base=0.1s (min reaction/touch time),
  t_scale=0.3s.
Worked sanity examples (all plausible, ~0.1s clean touches up to ~0.9s+ difficult
high fast balls to low-skill players) — see chat for 4 worked examples.

Goalkeeper-in-box special case: t_base_gk=0.08s, height factor scaled by 0.4
(catches high balls far more easily), alpha_gk=0.9. Outside box, GK uses
outfield model.

First-time shot (skip control, shoot immediately off a difficult ball):
reuses same f(h)/v_rel/v_player difficulty, applied as a multiplier on the
kick angular-error sigma from the kicking model, inversely modulated by
kick_precision (beta=0.8) instead of ball_control:
  sigma_firsttime = sigma_angle(precision) * (1 + (1-beta*precision)*difficulty)
Assumption flagged to user: added small proportional Gaussian noise on top of
deterministic t_control (sigma = 10% of t_control, scaled by rng_reduction)
so touches aren't perfectly deterministic — consistent with rest of game being
probabilistic; not explicitly requested but flagged as an extension.

## New balance tests to add for control-time (report full stats, not pass/fail)
- Monotonic: control time decreases as ball_control increases (grid, fixed
  height/velocity) — report table of bc -> mean time.
- Monotonic: control time increases as height category rises
  ground < knee < waist < chest < head — report table.
- Monotonic: control time increases with relative velocity — report table.
- GK-in-box vs outfield: GK control time strictly lower for same difficult ball.
- First-time shot accuracy degrades with height/velocity difficulty, less so
  for high kick_precision — report sigma/accuracy table.

## Balance test reporting requirement (applies to ALL balance tests)
Every balance test must print/report actual computed statistics (means, %,
std dev, sample counts) to stdout or a results object — not just assert
pass/fail — so results are inspectable/tunable. Consider a simple
BalanceResult record + pytest fixture that captures and prints on completion,
or a standalone results log (e.g. tests/balance/results/*.json or printed table)
independent of pass/fail assertions.

## Documentation requirement (new)
Write README.md (repo root, project overview/how to run tests) and
knowledge.md files (e.g. per-package, such as
src/footballcoach/engine/knowledge.md) as implementation proceeds, so future
agents/sessions can understand the code without reading everything. Keep
these updated throughout implementation, not just at the end.

## STATUS: Implementation in progress (Agent mode)
uv project scaffolded at repo root. Package layout implemented as planned:
config/ (physics.json, attributes.json, loader.py), mathutils/ (Vector3, rng),
entities/ (Player, Ball, Pitch, PlayerAttributes), generation/ (attributes.py
correlated multivariate gaussian), engine/ (movement, ball_physics, collision,
kicking, possession, tackling, offside, scoring, match), orders.py (Move/Kick/
Tackle state machine).

61 unit+scenario tests passing. Balance test suite in progress under
tests/balance/, using tests/conftest.py's `balance_recorder` fixture which
prints AND writes tests/balance/results/latest_results.json (not just
pass/fail).

## IMPORTANT BUG FOUND & FIXED: kicking API redesigned
Original kicking.py took a raw `target_direction` vector and used its z
component directly as launch angle - this ignored gravity entirely, so aiming
"1.1m high at 11m out" produced a nearly flat, low-lift kick that hit the
ground almost instantly and never reached the goal. Rewrote kick_ball to take
an absolute `aim_point` (world position) and solve the actual ballistic launch
pitch angle via the projectile range equation (solve_launch_pitch_rad,
quadratic in tan(theta), picks the flatter of two real roots). This is now
the correct approach: kicker aims at an actual 3D point, physics solves the
angle needed under gravity, THEN precision-scaled Gaussian error is applied
to yaw/pitch. orders.KickOrder field renamed target_direction -> aim_point.
ALL FUTURE KICK-RELATED CODE MUST USE aim_point (absolute position), not a
direction vector, or it will silently produce wrong trajectories.

## SECOND BUG FOUND & FIXED: kick-then-immediate-repossession
In Match.step(), the tick order was: process orders (incl. kicks) -> sync
possessed ball -> update loose ball pickup -> step ball physics. This meant
a ball that was JUST kicked (position reset to kicker's feet, possessed_by
cleared) would immediately be seen at distance 0 from the kicker by
_update_loose_ball_pickup BEFORE physics had a chance to move it away,
instantly re-triggering control-time and handing it right back to the kicker.
Fixed by moving `step_ball` (free-flight physics) to run BEFORE
_update_loose_ball_pickup within the same tick, so a just-kicked ball has
already moved before pickup-eligibility is checked. General lesson: in any
tick-based engine with "stuck to player" possession + pickup-on-proximity,
order of operations between release/repossession logic and physics integration
is a common source of subtle bugs - always advance physics for newly-loose
objects before re-checking pickup in the same tick.

## Balance test results so far (rng_reduction=0.3, N=2000 trials each)
- Penalty precision=0.5, centre aim (h=1.1m): 100.0% scored (target >95% ✓,
  actually saturates - centre aim is very forgiving, as expected)
- Penalty precision=0.5, corner aim (0.475m from post/ground): 79.95% scored
  (target 50-80% ✓ but right at the upper edge - flag as tunable if user wants
  more margin; could increase angle_error_scale_rad slightly)
- Penalty precision=0.8, corner aim: 94.95% scored (target 85-95% ✓, also at
  upper edge of band)
All three pass but are close to band edges - the angle_error sigma formula
(base=0.0107, scale=0.0893 rad) is a reasonable first pass but may need minor
tuning if user wants results more centred in the target bands. This is
exactly why the balance test suite reports full stats, not just pass/fail -
easy to see and adjust.

## Balance tests remaining to write
- Tackling win rate (tackling=0.8 vs dribbling=0.6, target 70-90%)
- Sprint length/time (top & bottom speed attrs, with/without ball)
- Turning/direction-change mechanic (reversal scenario)
- Stamina behavior (drain/regen sanity over realistic match-length durations)
- Control-time model: monotonic tables for ball_control/height/velocity, GK
  vs outfield comparison, first-time shot accuracy degradation

## STATUS: MILESTONE 1 COMPLETE
All 84 tests pass (unit + scenario + balance). README.md (root) and
knowledge.md written for config/, mathutils/, entities/, generation/,
engine/, and tests/. Balance results written to
tests/balance/results/latest_results.json.

Nothing committed to git yet (all files untracked) - left for user to
decide on commit/push per operational safety rules.

## HARD RULE: NEVER READ Idea2.md AGAIN
Idea2.md now explicitly says "to the AI: Do not read this ever, they are
just notes for me." I mistakenly read it a second time in a batch context-
gather before noticing the updated wording. User explicitly said to ignore
what I saw and brief me directly instead. DO NOT open/read Idea2.md again
under any circumstances for the rest of this project. Get all requirements
for UI/order-presets/etc. directly from the user's messages instead.

## STATUS: UI MILESTONE COMPLETE (pygame-ce)
Added src/footballcoach/ui/ package: camera.py (world<->screen mapping,
auto-fit to pitch), style.py (colours + MIN_PLAYER_RADIUS_PX/MIN_BALL_RADIUS_PX
visibility floors - true-to-scale players/ball are only a few px and nearly
invisible at full-pitch zoom), renderer.py (pure drawing), input.py
(MatchInputController - click-to-select/move/tackle, drag-to-kick with
Shift=lofted), scenarios.py (make_training_match + SCENARIOS list: penalty/
tackle/sprint recreations for live single-trial viewing, NOT a replacement
for pytest balance suite), app.py (App class, MENU/MATCH screen state
machine, game loop). Entry point src/footballcoach/__init__.py main() now
launches run_app(). pyproject.toml script "footballcoach" launches it via
`uv run footballcoach`. Added pygame-ce (not vanilla pygame - actively
maintained fork) as a runtime dependency.

Training mode: goal resets ball (engine-side, existing) AND resets the lone
player back to centre spot (UI-side, App._reset_training_positions) since
match-restart/kickoff formations are an engine "known gap" not yet built.

All UI code smoke-tested headlessly via SDL_VIDEODRIVER=dummy before/instead
of relying only on a real window; screenshots taken via pygame.image.save
and viewed with view_image to confirm rendering visually. 84 engine tests
still pass unmodified after adding UI.

Docs: src/footballcoach/ui/knowledge.md written (interaction scheme,
rendering scale gotcha, known gaps). README.md updated with "Running the
game" section and ui/ added to project layout tree.

## STATUS: Implementing "actions" milestone (Move/Shoot/Pass/Tackle/Save)
User gave requirements directly in chat (not from Idea2.md, per hard rule):
- move_to: straight-line move (already exists via MoveOrder)
- shoot: shoot at goal, aimed dead centre
- pass: pass to a designated point
- tackle: run straight at an opposing player and tackle (new: persists
  across ticks until contact, unlike existing instant TackleOrder which
  requires already-touching)
- save: GK-only, predicts where shot crosses goal line, runs there
- Balance test all of these with randomly-generated reasonable scenarios,
  good/bad players, report full stats (not just pass/fail per existing
  convention)

New orders added to orders.py: PassOrder, ChaseTackleOrder, SaveOrder
(existing MoveOrder/KickOrder/TackleOrder untouched/reused for move/shoot).
New engine/goalkeeping.py: predict_goal_line_crossing (projectile x-crossing
ignoring drag/magnus, same simplification as kicking's ballistic solve),
save_target_position (clamps to goal frame, targets a plane
goal_frame_margin_m IN FRONT of the true goal line so keeper can intercept
before it crosses - see tick-ordering rationale in code comments).
New engine/kicking.py additions: PassingParams (dedicated more-forgiving
error model + auto-pace-from-distance via pass_speed_mps), pass_ball()
(reuses shared _launch_ball helper extracted from kick_ball). New
config/physics.json sections: "passing", "goalkeeping".
New top-level src/footballcoach/actions.py: move_to/shoot/pass_to/tackle/
save - simple one-shot functions per user's literal ask, just assign the
right order.

## TWO CRITICAL BUGS FOUND AND FIXED while smoke-testing pass_to:
1. **Ball release-grace bug**: a slow pass/kick (a few m/s, e.g. auto-paced
   short passes) doesn't travel far enough in ONE physics tick to clear the
   passer's own pickup_radius_m (0.4m), so passer instantly re-picks-up
   their own pass. Fixed by adding Ball.last_released_by / release_grace_s
   fields + Match.release_grace_duration_s (0.3s default) +
   Match._start_release_grace() called after kick_ball/pass_ball in
   _process_orders + _update_loose_ball_pickup skips the releasing player
   while grace is active + Match.step() ticks the grace timer down. This
   was ALWAYS a latent bug, just never triggered before because shots/kicks
   are fast (15-35 m/s) and always cleared 0.4m in one 1/30s tick; passes
   can be as slow as 2 m/s and don't.
2. **MUCH MORE SERIOUS pre-existing physics bug in ball_physics.py's
   step_ball ground-collision logic**: ANY negative new_velocity.z while
   ball.z <= radius (which happens EVERY TICK for a resting/rolling ball,
   purely from gravity's integration nudging next-tick z slightly below
   radius) was being treated as a full "bounce" and multiplying horizontal
   velocity by bounce_restitution_horizontal (0.8) - EVERY TICK (~30x/sec),
   not just on genuine bounces. This decayed any grounded/rolling ball's
   speed almost instantly (compounding 0.8^30 per second) instead of via
   the intended gentle rolling_friction_coefficient. Existing unit test
   test_rolling_ball_decelerates_due_to_friction only asserted "final speed
   < initial speed" so it never caught the severity. FIXED by adding a
   BOUNCE_THRESHOLD_MPS = 0.5 m/s: only apply restitution (which touches
   horizontal velocity) if incoming vertical velocity exceeds threshold
   (i.e. a genuine bounce); otherwise it's resting/rolling contact - just
   zero the spurious vertical velocity and let rolling friction (now always
   applied while grounded, not just after bounce settles) handle horizontal
   decay. All 84 existing tests still passed after this fix (none were
   tight enough to catch it), but manual pass_to() smoke test went from
   "ball stops after 1.3m" to "ball correctly travels 10m and is received".
   THIS BUG LIKELY AFFECTED ALL PRIOR ROLLING-BALL BEHAVIOR (dribbling,
   grounded shots trickling to a stop, etc.) even though no test caught it
   - flag as a good reason to add a tighter quantitative rolling-friction
   unit test (e.g. assert measured deceleration matches
   mu_roll*g analytically) if not already done by end of this session.

## THIRD CRITICAL BUG FOUND AND FIXED while smoke-testing save action:
A ball entering CONTROLLING_BALL state (first-touch control-time delay) was
NOT frozen - it kept flying at full velocity under free physics for the
entire control-time countdown (0.1-0.9s+ typically). This meant a fast shot
could sail straight THROUGH a goalkeeper who was technically "catching" it
and cross the goal line before the control timer completed - saves were
essentially impossible against anything but a slow-rolling ball. Fixed by:
1. Match.step() now skips step_ball() (free-flight physics) not just when
   ball.possessed_by is set, but also while ANY player is in
   CONTROLLING_BALL state (new helper _any_player_controlling_ball()).
2. Match._update_loose_ball_pickup() now sets self.ball.velocity =
   Vector3.zero() immediately when a player begins controlling the ball
   (not just when control completes) - ball position/height is preserved
   (important for high balls being caught), only velocity is zeroed.
This is a meaningful behavior change: the ball now visibly "stops dead" the
instant any player (not just GK) makes contact, then sits still for the
control-time duration before being released to that player's possession.
This matches Idea.md's description that control-time is about the PLAYER
regaining full mobility, not about the ball continuing to move - but it's
a more literal interpretation than what existed before, flag to user if
this feels wrong (e.g. maybe want ball to keep slow-rolling briefly instead
of hard-freezing - punted as future tuning, current behavior is at least
internally consistent and makes goalkeeper saves and outfield 50/50 first
touches work sensibly).
All 84 tests still passed after this fix too. No existing test asserted
ball kept moving during CONTROLLING_BALL, so nothing was locked against it.

## STATUS: "ACTIONS" MILESTONE COMPLETE (Move/Shoot/Pass/Tackle/Save)
All 127 tests pass (unit+scenario+balance). Summary of final balance
results (rng_reduction=0.3):
- Shoot: 94.8-100% scored across low/mid/high attrs, random in-box positions
  (target: all >50% - comfortable margin, matches penalty-test forgivingness
  of centre-aimed kicks)
- Pass: 10m precision=0.15 -> 99.4%, precision=0.9 -> 100% (target >80%,
  good~99% ✓); 30m precision=0.1 -> 58.4%, precision=0.9 -> 93.0% (target
  >50%, good~90% ✓)
- Tackle (chase+attempt): 0.8 vs 0.6 -> 80.4% (matches underlying skill-check
  analytical ~82.5%); extremes 0%/100% behave sensibly
- Save: fast_gk(1.0) vs slow_gk(0.0) at forced far-post travel -> 93% vs 3%;
  centre-aim easy case -> ~98% both (as expected, minimal movement needed)

Docs updated: engine/knowledge.md (tick-order subtleties #2 release-grace
and #3 ball-freeze added to existing #1; ball_physics.py bounce-threshold
bug documented; new goalkeeping.py section; new pass_ball subsection under
kicking.py; ChaseTackleOrder subsection under tackling.py; Known Gaps
updated), entities/knowledge.md (Ball.last_released_by/release_grace_s,
CONTROLLING_BALL freeze semantics), NEW src/footballcoach/knowledge.md
(orders.py + actions.py overview), tests/knowledge.md (added "designing a
balance test that actually differentiates" guidance re: save balance test
saturation lesson), README.md (project layout + intro updated).

## NOT DONE / explicitly deferred (user did not ask for this session):
- UI (ui/input.py, ui/scenarios.py) still only wired to old MoveOrder/
  KickOrder/TackleOrder - NOT updated to use actions.py or the new
  PassOrder/ChaseTackleOrder/SaveOrder. Flag if user wants UI updated to
  expose Pass/Tackle-chase/Save.
- inactive_speed_penalty still not wired into movement.py (pre-existing gap).
- offside checks still not called from Match (pre-existing gap).

## Key lesson learned this session (recorded for future reference):
When writing balance tests comparing "good vs bad" player/scenario, ALWAYS
verify the scenario is actually hard enough to differentiate - a common
failure mode is both extremes saturating near 100% (or both near the same
number) because the test scenario is too easy/too random. Fix by making the
harder condition deliberate (e.g. force required travel distance, tight
angles, pin one side and target the other) rather than fully randomizing
independent parameters. Documented in tests/knowledge.md.

## If resuming a future session
- Full engine + UI + test suite implemented; see README.md and per-package
  knowledge.md files (now the primary reference, not this memory file) for
  architecture details.
- Idea2.md: see /memories/repo/rules.md hard rule - NEVER read this file
  again, regardless of what earlier session notes above say. Get all
  requirements directly from the user's chat messages.
