# ui/

A pygame-ce-based renderer and input layer sitting on top of the headless
engine (`engine/match.py`). Nothing in here touches simulation logic - it
only reads `Match`/`Player`/`Ball` state to draw, and writes
`player.current_order` to issue Move/Kick/Tackle orders. This keeps the UI
swappable/removable without touching the engine, per the project's
"engine is UI-agnostic" design principle.

## Files

- `camera.py` - `Camera` maps between world metres (engine convention: `x` =
  pitch length axis, `y` = pitch width axis, origin at pitch centre) and
  screen pixels (origin top-left, `y` grows downward). The window is
  auto-sized to fit the whole pitch plus a margin at a fixed
  `pixels_per_metre` (`Camera.fit_to_pitch`).
- `style.py` - colour palette and a couple of rendering constants.
- `renderer.py` - pure drawing functions (`draw_pitch`, `draw_player`,
  `draw_ball`, `draw_hud_text`, `draw_drag_indicator`). Takes a `pygame.Surface`
  and entities, has no game-loop or input awareness.
- `input.py` - `MatchInputController` translates raw mouse events into
  orders on the engine's `Player.current_order`. See its docstring for the
  full click/drag interaction scheme (click player to select, click ground
  to move, click opponent to tackle, drag from the ball-carrier to kick).
- `scenarios.py` - builds `Match` instances for the two non-freeplay modes:
  `make_training_match()` (1 player + ball, full pitch, both goals live) and
  `SCENARIOS` (a handful of hand-picked recreations of the pytest balance
  scenarios - penalty, tackle, sprint - for *watching* a single live trial,
  not for statistical validation; the pytest suite in `tests/balance/`
  remains the source of truth for balance numbers).
- `app.py` - `App` owns the pygame window, main loop, a simple two-screen
  state machine (`MENU` / `MATCH`), and wires input events to
  `MatchInputController` + `Match.step()` + `Renderer`.

## Rendering scale gotcha

Players (radius 0.3m) and the ball (radius 0.11m) are only a few pixels
across at a realistic `pixels_per_metre` zoom level fitting the whole pitch
on screen - true-to-scale rendering makes them nearly invisible. `style.py`
defines `MIN_PLAYER_RADIUS_PX` / `MIN_BALL_RADIUS_PX` floors that the
renderer applies on top of the physically-scaled radius; **positions remain
physically accurate**, only the *drawn* circle size is boosted for
visibility. Don't use the rendered circle size for any gameplay logic (e.g.
click-to-select uses `SELECT_TOLERANCE_PX` in `input.py`, not the drawn
radius, though in practice they're similar).

## Ball height display

Per the original design spec ("make the ball change size with height (but
maybe exaggerate the effect) and have a small number on it showing its
height in metres"): `draw_ball` boosts the ball's radius by
`1 + min(height, 5)*0.35` and renders a `"{height:.1f}m"` label next to it
whenever height exceeds 0.15m.

## Interaction scheme (`input.py`)

- **Click a player** -> select them. Click the same player again to
  deselect. Click a different same-team player to switch selection.
- **Click an opposing player** (while one of your players is selected) ->
  issues a `TackleOrder` targeting the clicked player. No proximity check is
  done client-side - the engine's `are_touching()` check in `Match` decides
  whether the tackle actually resolves this tick or just sits pending.
- **Click empty ground** (while a player is selected) -> issues a
  `MoveOrder` to that world position (always `sprint=True` currently - no UI
  toggle yet for jog vs sprint).
- **Click-and-drag starting on the selected player** -> issues a
  `KickOrder`. Drag direction sets aim direction; drag length (capped at
  `MAX_KICK_DRAG_M`) sets `power_fraction`; the aim point is projected out
  along the drag direction at `2x` the drag length (capped at 60m) at a
  fixed height (`GROUND_AIM_HEIGHT_M` normally, `LOFTED_AIM_HEIGHT_M` if
  Shift is held while dragging, for a chip/lob). This only does anything
  useful if the selected player currently has the ball - `Match` silently
  no-ops a `KickOrder` for a player without possession.
- A short click (drag distance below `CLICK_DRAG_THRESHOLD_PX`) is always
  treated as a click, not a drag, even if it started on the selected player
  - this lets you re-click your own player to deselect without accidentally
  triggering a tiny, useless kick.

## Training mode goal reset (`app.py`)

`Match._reset_after_goal` (engine-side) only resets the ball to the centre
spot - it deliberately doesn't touch player positions, since that's a
match-restart/kickoff concern out of scope for the current engine milestone
(see engine/knowledge.md's "Known gaps"). Training mode is single-player
free play, though, so `App._reset_training_positions` (UI-side) additionally
resets the lone player back to the centre spot on every goal, so the
practice loop doesn't require the trainee to trek back from wherever they
ended up after a shot. This is intentionally training-mode-specific logic
living in the UI layer, not a general engine behaviour.

## Known gaps / not yet implemented

- No jog/sprint toggle for Move orders (always sprint).
- No visual indicator for ball possession state beyond player/ball
  proximity (no highlight ring showing "this player has the ball").
- Balance scenarios always use `rng_reduction=0.3` and don't loop/repeat
  automatically - selecting one plays a single live trial; go back to the
  menu (Esc) and re-select to see another random outcome.
- No sound, no game clock/timer, no formations/kickoff - out of scope for
  this milestone (mirrors the engine's own documented gaps).
