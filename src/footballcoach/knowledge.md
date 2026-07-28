# src/footballcoach/ (top-level modules)

Two package-root modules sit above the `engine/`/`entities/`/`ui/`
sub-packages: `orders.py` and `actions.py`. Both are part of the
"instruction layer" between a caller (UI, tests, or a future RL policy) and
the engine's per-tick simulation.

## `orders.py` - the order data types

Defines the order types a player can hold in `player.current_order`:
`MoveOrder`, `KickOrder`, `ShootOrder`, `TackleOrder`, `PassOrder`,
`ChaseTackleOrder`, `GetPossessionOrder`, `MarkOrder`, `SaveOrder`,
`StopOrder`. These are plain dataclasses with an `OrderStatus`
(PENDING/IN_PROGRESS/COMPLETE) — the actual per-tick execution logic for
each lives in `engine/match.py`'s `Match._process_orders`, not here. See
`engine/knowledge.md` for how each order type behaves tick-to-tick
(especially the distinction between `TackleOrder`, which requires the two
players to already be touching, and `ChaseTackleOrder`, which persists
across ticks and closes the distance itself).

`MoveOrder` has an optional `max_speed_on_arrival_mps` field (`None` =
resolve to jog speed, `0.0` = full standstill). The engine uses
`_braking_speed_mode()` to switch from SPRINT→JOG→STANDSTILL early enough
to arrive at the requested speed — see the **Engine/AI boundary** section
below. No order is permitted to assign `player.velocity` directly.

## `actions.py` - simple, literally-named action helpers

Per the project's requirement for straightforward functions rather than
hand-built order objects: `move_to`, `shoot`, `pass_to`, `tackle`, `save`,
`mark`. Each just assigns the corresponding order to `player.current_order`
- none of them drive the match loop themselves, so a caller still needs to
call `Match.step()` in a loop afterwards for anything to actually happen.

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
- `mark(player, target)` -> `MarkOrder` (see below). Never auto-completes.

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
- Mark: `tests/balance/test_mark_balance.py`

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

## Engine/AI boundary — velocity invariant

**Orders and actions MUST NOT assign `player.velocity` directly.**
The only permitted interface between the order/AI layer and the engine is:

```python
step_player_towards(player, direction, SpeedMode.SPRINT | JOG | STANDSTILL, dt, params, has_ball)
```

The engine (`engine/movement.py`) owns `player.velocity` and
`player.heading_rad` exclusively. The sole velocity snap in the entire
codebase is a near-zero guard inside `step_player_towards` itself
(`_STOP_SNAP_THRESHOLD_MPS = 0.02 m/s`, used only in `STANDSTILL` mode to
prevent floating-point drift at rest). The GK save position teleport in
`SaveOrder` also calls `step_player_towards` rather than snapping, so even
that is covered.

### `SpeedMode` values (`engine/movement.py`)

| Mode | Target speed | Notes |
|------|-------------|-------|
| `SPRINT` | `v_top` | full attribute + stamina speed |
| `JOG` | `v_top * 0.5` | half pace |
| `STANDSTILL` | `0 m/s` | uses `standstill_decel_multiplier` (1.5×) accel boost — stopping feels snappier than accelerating |

### Arrival logic (`MoveOrder`, `MarkOrder` standoff)

`_braking_speed_mode(dist, speed, arrival_speed, ...)` in `match.py`
looks ahead each tick and returns the appropriate `SpeedMode` so the player
naturally decelerates to `arrival_speed` by the time they reach the target:

- `MoveOrder.max_speed_on_arrival_mps`:
  - `None` (default) → resolved to jog speed at execution; order completes
    as soon as the player is within `arrival_tolerance_m` at jog speed or below.
  - `0.0` → full standstill; the tolerance window is widened by 1.5× to
    give the braking physics room to settle; order does not complete until
    `speed_mps <= 0.05`.
  - Any explicit `float` → treated as the speed threshold at arrival.
- `MarkOrder` standoff position always uses `arrival_speed=0.0` (the marker
  holds the standoff point still).
- `_braking_speed_mode` also has a 0.5 m close-range guard that forces
  `STANDSTILL` regardless of the braking-distance calculation, preventing
  low-speed re-acceleration oscillation near the target.