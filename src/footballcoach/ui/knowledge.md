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
  `SCENARIOS` (hand-picked recreations of the pytest balance scenarios -
  penalty, save/goalkeeper dive, shoot from box, 20m pass, tackle challenge,
  sprint race - for *watching* a single live trial, not for statistical
  validation; the pytest suite in `tests/balance/` remains the source of
  truth for balance numbers).
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

## Player visual indicators (`style.py` / `renderer.draw_player`)

- **Goalkeepers** are drawn in `GOALKEEPER_COLOUR` (a distinct orange)
  instead of their team colour, so the keeper is identifiable at a glance.
- **Ball possession**: whichever player currently has the ball
  (`has_ball`, passed in by `App._draw_match`) gets a white
  `POSSESSION_OUTLINE` ring drawn around their circle.
- **Inactive players** (`PlayerState.INACTIVE_TACKLED`) are drawn onto a
  per-pixel-alpha `pygame.Surface` at `INACTIVE_ALPHA` transparency (rather
  than a flat colour substitute), so a tackled/off-balance player visibly
  fades rather than just changing colour.
- **Top layer for ball carrier**: `App._draw_match` sorts the player list so
  whichever player has the ball is drawn last, i.e. on top of every other
  player - avoids the possession ring/player circle being partially
  obscured by a nearby defender drawn afterwards.

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
- **`P` key** -> enters one-shot "Pass mode" (`MatchInputController.
  enter_pass_mode()`, tracked via the `OrderMode` enum). The next click on a
  same-team player or empty ground issues a `PassOrder` at that
  player/position instead of the normal select/move click handling, then
  automatically reverts to `OrderMode.MOVE`. `Esc` cancels any transient
  mode (`cancel_order_mode()`).
- **`K` key** -> enters one-shot "Shoot mode" (`enter_shoot_mode()`). The
  next click on any pitch point issues a `ShootOrder` aimed at that point
  (z=1.0m, full power), then automatically reverts to `OrderMode.MOVE`.
  `Esc` also cancels shoot mode.
- **`S` key** -> issues a `SaveOrder` to the currently-selected player via
  `MatchInputController.issue_save_order()`, but only if that player
  `.is_goalkeeper` (a no-op otherwise).

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

## Help overlay (`app.py`)

`H` (or clicking the help button in the top-right corner of the match
screen, drawn by `_draw_help_button`) toggles `App.show_help`. While shown,
`_draw_help_overlay` renders a full-screen semi-transparent panel listing
every control (click/drag/tackle/pass/save/pause/menu) and what each visual
indicator means (goalkeeper colour, possession outline, inactive
translucency). Match input events are suppressed while the overlay is open
(only the help button/`H`/`Esc` are handled) so you can't accidentally
issue orders while reading it; `Esc` closes the overlay first before
falling back to its normal pass-mode-cancel / return-to-menu behaviour.

## Hotkey bar (`renderer.draw_hotkey_bar` / `App._hotkey_entries`)

A permanent strip at the bottom of the screen shows every hotkey at all
times.  Each entry is rendered in one of three states:
- **Active** (accent colour): the key is the current transient mode, e.g.
  `[P] Pass` while PASS mode is engaged.
- **Enabled** (bright): the action is currently valid for the selected
  player (e.g. `[K] Shoot` lights up only if the selected player has the
  ball).
- **Disabled** (dim but readable): action is not valid right now (no
  selection, wrong player type, etc.).

This replaces the old inline key-hint text in the HUD, which only appeared
when a player was selected. `App._hotkey_entries()` computes the seven
entries (`[Spc]`, `[P]`, `[K]`, `[S]`, `[X]`, `[H]`, `[Esc]`) and their
states from the current selection/ball/mode.

## Balance scenario looping (`ScenarioLoop` in `scenarios.py`)

`ScenarioLoop` wraps a `ScenarioDefinition` and replays it for
`max_trials` (default 20) consecutive trials.  The UI calls
`loop.step()` once per frame; each call advances the current trial's
`Match` by one physics tick.  A trial ends when:
1. Ball crosses the touchline or goal line (out of bounds).
2. All player orders have resolved *and* the ball is possessed or
   stationary (≥ 30 ticks in, to let the initial kick/tackle execute).
3. `timeout_ticks` failsafe (default 500, ≈ 16.7 s at 30 Hz).

`loop.step()` returns `True` the tick a trial ends, at which point the
loop has already built a fresh `Match` (via `definition.build()`) for
the next trial, or set `loop.complete = True` if `max_trials` reached.
`App._step_match` handles the three outcomes (mid-trial, new trial
started, loop complete → return to menu).

The HUD shows `Trial N/max_trials` instead of the score during scenario
mode.

## Training mode auto-select

`App._start_match` now sets `input_controller.selected_player_id` to
the lone trainee's ID immediately when `is_training_mode=True`, so the
player is controllable from the first frame without an initial click.

## Known gaps / not yet implemented

- No jog/sprint toggle for Move orders (always sprint).
- Shoot mode (`K`) always fires at full power (`power_fraction=1.0`) with a
  fixed aim height of 1.0m; no UI control for power or aim height yet.
- No sound, no game clock/timer, no formations/kickoff - out of scope for
  this milestone (mirrors the engine's own documented gaps).
