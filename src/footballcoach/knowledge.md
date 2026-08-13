# src/footballcoach/ (top-level modules)

> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

Two package-root modules sit above the `engine/`/`entities/`/`ui/`
sub-packages: `orders.py` and `actions.py`. Both are part of the
"instruction layer" between a caller (UI, tests, or a future RL policy) and
the engine's per-tick simulation.

## `orders.py` - the order data types

Defines the order types a player can hold in `player.current_order`:
`MoveOrder`, `KickOrder`, `ShootOrder`, `PassOrder`,
`ChaseTackleOrder`, `GetPossessionOrder`, `MarkOrder`, `SaveOrder`,
`StopOrder`. These are plain dataclasses with an `OrderStatus`
(PENDING/IN_PROGRESS/COMPLETE) — the actual per-tick execution logic for
each lives in `engine/match.py`'s `Match._process_orders`, not here. See
`engine/knowledge.md` for how each order type behaves tick-to-tick.
`TackleOrder` was removed — use `ChaseTackleOrder` for all tackle actions
(it persists across ticks and closes the distance itself).

`MoveOrder` has an optional `max_speed_on_arrival_mps` field (`None` =
resolve to jog speed, `0.0` = full standstill). The engine uses
`orders.braking_speed_mode()` to switch from SPRINT→JOG→STANDSTILL early enough
to arrive at the requested speed — see the **Engine/AI boundary** section
below. No order is permitted to assign `player.velocity` directly.

## Player action methods — the canonical way to issue Orders

`Player` has named action methods that are the **only** correct way to set
`player.current_order`.  Never assign `player.current_order = SomeOrder(...)`
from outside the `Player` class.

| Method | Order issued | Notes |
|--------|-------------|-------|
| `player.kick(aim_point, power_fraction, spin)` | `KickOrder` | Only effective if player has ball |
| `player.pass_ball(target_position, target_player_id, power_fraction)` | `PassOrder` | Led pass if `target_player_id` set |
| `player.get_possession()` | `GetPossessionOrder` | Chase ball / dispossess carrier |
| `player.tackle_player(target_player_id)` | `ChaseTackleOrder` | Chase + tackle on contact |
| `player.mark_player(target_player_id)` | `MarkOrder` | Never auto-completes |
| `player.stop()` | `StopOrder` | Decelerate to standstill |
| `player.save_goal()` | `SaveOrder` | Goalkeeper only |
| `player.move_to(target_position, sprint, max_speed_on_arrival_mps)` | `MoveOrder` | Rules-based AI and BC labels only — see neural boundary below |

## Neural network / Orders boundary (IMPORTANT)

`MoveOrder` and all Order types are used by:
- The **rules-based AI** (assign `player.ai = Phase1RulesAI()` etc.; `Match.step()` calls `player.ai.act(player, match, tick)` automatically)
- **BC label generation** (supervised teacher signal — reads what order the
  rules AI *would* issue on a scratch player snapshot and translates it into
  equivalent physical targets for imitation; never issues a real Order on a
  live player)
- **`HybridPlayerAI`'s order-override channel** (`rules_ai.py`) — a sanctioned
  second case: a human (via the UI) or a rules-based caller can assign a real
  `Order` directly to a normally neural-controlled player, bypassing the
  neural network entirely for as long as that order is in progress. See
  `ai/knowledge.md`'s "HybridPlayerAI" section — this is orthogonal to (and
  takes priority over) that same class's decision-neuron override channel,
  which instead forces decision-head *probabilities* before gating while
  keeping the execution network in control of physical motor output.

**The neural network itself never issues an Order — not even when a decision
head (shoot/pass/tackle/get_possession/mark) fires.** Those decision heads are
strategic-context *input* to the execution network only.
`ai/action/apply_nn_action.py::apply_action_to_player()` is the sole place
execution-network outputs touch engine state, and it writes directly onto the
`Player` object — no Order of any kind is constructed:

- **Movement**: `gating.exec_move` (an **execution**-network Bernoulli,
  distinct from the **decision**-network's own `move` head) selects
  STANDSTILL vs moving. When moving, `player.desired_direction` is set
  straight from `gating.move_direction` (a unit vector) and
  `player.desired_speed_mode` to SPRINT or JOG per `gating.sprint`. No
  `MoveOrder`, no `player.move_to()` call — the engine's `_apply_movement()`
  loop reads `desired_direction`/`desired_speed_mode` directly every tick.
- **Kick**: `player.kick_with_direction(match, direction_3d, power, spin)`
  when `gating.kick_this_tick` fires — a parallel chokepoint to
  `kick_direct()` (used by `KickOrder`/rules AI/`MoveOrder`'s push-kick) that
  takes an explicit 3D direction instead of an aim point + auto-solved
  trajectory. Both set `kicked_this_tick`/`last_kick_*` and fire `on_kick` —
  see `entities/knowledge.md`.
- **Tackle**: `player.tackle_armed = True` when `gating.tackle_attempt` fires
  (preconditions permitting) — the engine resolves it on contact in
  `Match._check_armed_tackles()`, the same mechanism a rules-AI
  `ChaseTackleOrder`/`GetPossessionOrder` ultimately arms. No "attempt now"
  method call and no `ChaseTackleOrder` is created.

The decision network's `move_region_center` / `move_arrival_speed` are
**strategic context** for reward shaping and BC label generation — they are
NOT used as motor control inputs.

**Corollary — BC label generation must respect the same boundary in
reverse:** code that derives BC *labels* for the execution network (see
`ai/ppo/bc.py`'s `phase1_labels()`) must source execution-level fields
(`move_direction`, `sprint`, `exec_move`, kick vector, `tackle_attempt`) from
`player.desired_direction`/`player.desired_speed_mode`/`player.kicked_this_tick`/
`player.last_kick_*` — i.e. what the ORDER MACHINERY ACTUALLY PRODUCED on the
player that tick — never by re-deriving geometry from an Order's own fields
(that bypasses braking/repulsion/turning/push-kick logic). See
`ai/knowledge.md`'s "Orders vs execution-network labels boundary" section for
the full rule and `agent_plans/bc_execution_label_boundary_and_followups.md`
for the bug history.

## `actions.py` — thin shim (deprecated, kept for compatibility)

`actions.py` functions (`move_to`, `shoot`, `pass_to`, `tackle`, `save`,
`mark`) each delegate to the matching `Player` method in one line.  Existing
call sites in the rules-based AI, UI, and tests are preserved.  **New code
should call player methods directly.**

`opponent_goal_centre(pitch, team)` in `actions.py` still resolves goal
direction per the `Team.LEFT` attacks +x / `Team.RIGHT` attacks -x convention
from `engine/offside.py` and `engine/goalkeeping.py`.  If you ever flip this
convention update it in all three places.

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
## `ai/` - neural network / PPO player AI

See [`ai/knowledge.md`](ai/knowledge.md) for the full operational notes.
In brief: the `ai/` package is a pure consumer of the engine — it never
modifies `engine/`, `entities/`, `orders.py`, or `actions.py`.  Install with
`uv sync --group ai` (torch is a separate optional dependency group so the
base game stays lightweight).

The two MVP training experiments (phase 1: get-possession/move, phase 2:
shoot) are driven via:
```bash
uv run python -m footballcoach.ai.scripts.train --phase 1
uv run python -m footballcoach.ai.scripts.train --phase 2
```

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

`braking_speed_mode(dist, speed, arrival_speed, ...)` in `orders.py` (called
from `_compute_movement_intent()` — order-layer logic, NOT engine physics;
the neural network's direct-drive path bypasses it entirely) looks ahead each
tick and returns the appropriate `SpeedMode` so the player naturally
decelerates to `arrival_speed` by the time they reach the target:

- `MoveOrder.max_speed_on_arrival_mps`:
  - `None` (default) → resolved to jog speed at execution; order completes
    as soon as the player is within `arrival_tolerance_m` at jog speed or below.
  - `0.0` → full standstill; the tolerance window is widened by 1.5× to
    give the braking physics room to settle; order does not complete until
    `speed_mps <= 0.05`.
  - Any explicit `float` → treated as the speed threshold at arrival.
- `MarkOrder` standoff position always uses `arrival_speed=0.0` (the marker
  holds the standoff point still).
- `braking_speed_mode` also has a 0.5 m close-range guard that forces
  `STANDSTILL` regardless of the braking-distance calculation, preventing
  low-speed re-acceleration oscillation near the target.