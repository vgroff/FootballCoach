# entities/

Plain data classes for the things that exist in a match: players, the ball,
and the pitch. No physics or rules logic lives here (that's `engine/`) -
these are just state containers plus a few purely-geometric queries.

## `PlayerAttributes` (`attributes.py`)

The 8 core skills from Idea.md, each a float in `[0, 1]`:
`top_speed`, `acceleration`, `stamina`, `kick_precision`, `kick_power`,
`dribbling`, `ball_control`, `tackling`. Validated on construction (raises
`ValueError` if any value is out of range). Don't construct these directly
in game code for "real" players - use `generation.generate_attributes()` so
correlations and tier presets are applied; `PlayerAttributes(...)` /
`PlayerAttributes.average(...)` are mainly for tests.

## `Player` (`player.py`)

A player's position/velocity/heading/stamina/state, plus a reference to its
attributes and a `current_order` (see `orders.py` at the package root - kept
as `object | None` here to avoid a circular import between `entities` and
`orders`). `Player.create(...)` is the normal constructor - it pulls
`radius_m`/`height_m` from `physics.json` so they stay in sync with the rest
of the engine.

### Direct-physics methods for the neural network

`Player.kick_direct()` and `Player.kick_with_direction()` bypass the Orders
system entirely and execute kick physics immediately — no `KickOrder` is
created either way. Only `kick_with_direction()` is called by the neural
network (via `ai/action/apply_nn_action.py::apply_action_to_player()`);
`kick_direct()` is used by `KickOrder.execute()`/rules AI and by
`MoveOrder`'s push-kick behaviour (`orders.py::_do_push_kick()`):

- `player.kick_direct(match, aim_point, power_fraction, spin)` — aims at an
  explicit target point; the ball's launch direction is solved from that aim
  point (same logic as `KickOrder.execute()`, which delegates here). Only
  fires if `ball.possessed_by == player.player_id`.
- `player.kick_with_direction(match, direction_3d, power_fraction, spin)` —
  takes an explicit 3D unit direction directly, no aim-point ballistic solve.
  Neural network only.

Both are otherwise equivalent chokepoints: each independently sets
`player.kicked_this_tick = True`, `last_kick_direction`/`last_kick_power_fraction`/
`last_kick_spin`, and fires `player.on_kick` if set — see `ai/knowledge.md`'s
BC label table for why this matters (BC supervision reads these fields, not
order types).

There is no `tackle_direct()` method. The neural network arms a tackle by
setting `player.tackle_armed = True` (via `apply_action_to_player()`) — the
same flag a rules-AI `ChaseTackleOrder`/`GetPossessionOrder` sets when it
closes to contact range. Either way, `Match._check_armed_tackles()` is what
actually resolves the tackle on contact; no order object is required.

All other action methods (`kick()`, `move_to()`, `get_possession()`, etc.)
set `current_order` and are for the **rules-based AI and human input only**.

`PlayerState` is a small state machine:
- `ACTIVE` - normal play, follows orders, regenerates stamina.
- `INACTIVE_TACKLED` - just been tackled (or just missed a tackle attempt
  themselves - see `engine/tackling.py`); can't tackle again and has reduced
  speed until `state_timer_s` runs out.
- `CONTROLLING_BALL` - mid first-touch control-time delay (see
  `engine/possession.py`); can't act until the timer completes, at which
  point `Match._complete_control` grants possession. The ball's velocity is
  frozen to zero the instant this state begins (see
  `Match._update_loose_ball_pickup`) and stays frozen (no free-flight
  physics runs) for the whole delay - it does NOT keep flying at the speed
  it arrived at while "being controlled".

`Player.is_inactive` (`True` iff `state == PlayerState.INACTIVE_TACKLED`) is
used by `engine/collision.py` to exclude inactive players from
player-player push-apart collision (you can run straight through a player
who's just been tackled/mistimed a tackle) while still allowing their
cylinder to block a loose ball crossing into it from outside - see
`engine/knowledge.md`'s `collision.py` section.

## `Ball` (`ball.py`)

Position/velocity/spin plus `possessed_by: str | None` (a player id, or
`None` if loose). When `possessed_by` is set, `engine/ball_physics.py`'s
`step_ball()` is a no-op for that ball - the `Match` loop instead glues the
ball to the carrying player's position every tick (see
`Match._sync_possessed_ball`). Free-flight physics only resumes once the
ball is released (by a kick, a successful tackle, going out of play, etc.).

Loose-ball pickup eligibility (including stopping a player from instantly
re-picking-up a ball they just kicked/passed) is governed by a
closing-velocity check in `possession.can_pick_up_ball()`, not by any field
on `Ball` itself - see `engine/knowledge.md`'s tick-order notes ("ordering
subtlety #2").

`just_bounced_timer_s: float = 0.0` is a **display-only** countdown set by
`engine/ball_physics.step_ball()` to `just_bounced_display_duration_s` (0.3 s,
from `physics.json["ball_physics"]`) whenever the ball makes a real bounce
(incoming vertical speed above `BOUNCE_THRESHOLD_MPS` and outgoing speed also
above it, i.e. the ball actually bounces rather than settling). It decays by
`dt` each tick and floors at 0. The renderer uses it to draw an amber ring
around the ball briefly after each bounce. It has no effect on simulation
logic — treat it as a visualisation hint only; the engine itself never reads
it.

## `Pitch` (`pitch.py`)

Static geometry only - dimensions, goal mouths, box boundaries - centred at
the origin with goals at `x = ±half_length`. `Pitch.standard()` builds a
regulation pitch from `physics.json["pitch"]`; pass explicit dimensions to
the dataclass constructor for non-standard pitches (per Idea.md's "these
can be changed" requirement). `is_goal()` returns which goal ("left"/
"right") a position is inside, used by `engine/scoring.py`.
