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
