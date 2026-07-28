# src/footballcoach/ (top-level modules)

Two package-root modules sit above the `engine/`/`entities/`/`ui/`
sub-packages: `orders.py` and `actions.py`. Both are part of the
"instruction layer" between a caller (UI, tests, or a future RL policy) and
the engine's per-tick simulation.

## `orders.py` - the order data types

Defines the six order types a player can hold in `player.current_order`:
`MoveOrder`, `KickOrder`, `TackleOrder`, `PassOrder`, `ChaseTackleOrder`,
`SaveOrder`. These are plain dataclasses with an `OrderStatus`
(PENDING/IN_PROGRESS/COMPLETE) - the actual per-tick execution logic for
each lives in `engine/match.py`'s `Match._process_orders`, not here. See
`engine/knowledge.md` for how each order type behaves tick-to-tick
(especially the distinction between `TackleOrder`, which requires the two
players to already be touching, and `ChaseTackleOrder`, which persists
across ticks and closes the distance itself).

## `actions.py` - simple, literally-named action helpers

Per the project's requirement for straightforward functions rather than
hand-built order objects: `move_to`, `shoot`, `pass_to`, `tackle`, `save`.
Each just assigns the corresponding order to `player.current_order` - none
of them drive the match loop themselves, so a caller still needs to call
`Match.step()` in a loop afterwards for anything to actually happen.

- `move_to(player, target_position)` -> `MoveOrder`
- `shoot(player, pitch)` -> `KickOrder` aimed at
  `opponent_goal_centre(pitch, player.team)` (dead centre of the goal the
  player's team is attacking, at `DEFAULT_SHOOT_HEIGHT_M` = 1.1m,
  `DEFAULT_SHOOT_POWER_FRACTION` = 0.85). Only has an effect if the player
  currently has the ball.
- `pass_to(player, target_position)` -> `PassOrder` (auto-paced by
  distance unless `power_fraction` is given explicitly).
- `tackle(player, target)` -> `ChaseTackleOrder` (chase + one tackle
  attempt on contact).
- `save(goalkeeper)` -> `SaveOrder` (goalkeeper-only; continuously tracks
  the incoming shot).

`opponent_goal_centre(pitch, team)` resolves which goal a team is attacking
using the same convention as `engine/offside.py` and
`engine/goalkeeping.py`: `Team.LEFT` attacks +x (the right goal),
`Team.RIGHT` attacks -x (the left goal). If you ever flip this convention,
update it in all three places.

## Where the balance tests for these live

- Shoot: `tests/balance/test_shoot_balance.py`
- Pass: `tests/balance/test_pass_balance.py`
- Tackle (the full chase+tackle action): `tests/balance/test_tackle_action_balance.py`
  (as opposed to `test_tackling_balance.py`, which tests the underlying
  `attempt_tackle()` skill check directly, without any movement/chasing)
- Save: `tests/balance/test_save_balance.py`

Each reports full statistics (not just pass/fail) via the `balance_recorder`
fixture - see `tests/knowledge.md`.
## `steering.py` - player repulsion during Move orders

A thin AI/order-layer module (sibling to `actions.py`, deliberately NOT
under `engine/`) that computes a repulsion-adjusted movement direction and
speed multiplier for `MoveOrder` handling in `Match._process_orders`. The
engine modules (`movement.py`, `collision.py`) are completely unaware of it.

**Design boundary**: `steering.py` decides *what direction and speed to
request*; `engine/movement.py`'s `step_player_towards` then executes that
request as pure kinematics. This mirrors the existing `actions.py` /
`engine/` separation: "AI layer decides, engine executes".

**Only `MoveOrder` uses repulsion.** ChaseTackleOrder, GetPossessionOrder,
SaveOrder, etc. do not — confirmed out of scope.

### `compute_repulsion(player, desired_dir, other_players, ball_carrier_id, params)`

Returns `(adjusted_direction: Vector3, speed_multiplier: float)`.

- For each other player within `params.radius_m` that is **not** the ball
  carrier: add a repulsion vector away from them, with linear falloff
  `strength(d) = strength_base * (1 - d/radius_m)`.
- Ball carrier (if `player` has the ball) gets repulsion multiplied by
  `ball_carrier_repulsion_mult` and a speed penalty
  `min(ball_carrier_speed_penalty_max, |net_rep| * speed_penalty_scale)`.
- **Orthogonal nudge**: if `rel_vel · net_repulsion < alignment_dot_threshold`
  (player heading nearly straight into an obstacle), adds a perpendicular
  component. Sign is chosen via 2D cross product of net_repulsion and
  desired_dir to pick the side closest to the intended destination.
- **Deflection cap** (`max_deflection_deg`): the final blended direction is
  clamped so the steering angle relative to `desired_dir` never exceeds this
  value. Prevents players from moving backwards. Currently 90°.

### Config (`physics.json["repulsion"]`)

| Key | Value | Notes |
|-----|-------|-------|
| `radius_m` | 3.8 | distance at which repulsion begins |
| `strength_base` | 2.8 | peak directional weight at d=0 (m/s-equivalent) |
| `ball_carrier_repulsion_mult` | 2.2 | extra multiplier when player has ball |
| `ball_carrier_speed_penalty_max` | 0.4 | max fractional speed penalty (40%) |
| `speed_penalty_scale` | 0.14 | converts \|net_rep\| to speed penalty |
| `alignment_dot_threshold` | -0.7 | trigger orthogonal nudge threshold |
| `min_orthogonal_adjust_mps` | 0.75 | sideways nudge magnitude |
| `max_deflection_deg` | 90.0 | hard cap on steering angle change per tick |

Values were tuned via `scripts/grid_search_repulsion_params.py` across 8
scenarios (head-on collision, ball carrier vs jogger, cluster, corridor, etc.)
with a multi-objective scorer (avoidance%, min_sep, avg_deflection, bc
catchability, corridor clearance). See `scripts/repulsion_sandbox.py` for
per-tick visualisation. Balance tests: `tests/balance/test_repulsion_balance.py`.