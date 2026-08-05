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
- `style.py` - colour palette and rendering constants, including Phase G
  additions: `CONTROL_DELAY_OUTLINE` (cyan ring for `CONTROLLING_BALL`),
  `INACTIVE_OUTLINE` (red ring for `INACTIVE_TACKLED`), and three ball-state
  ring colours (`BALL_STATE_FLYING_OUTLINE`, `BALL_STATE_ROLLING_OUTLINE`,
  `BALL_STATE_BOUNCED_OUTLINE`).
- `renderer.py` - pure drawing functions. Phase G additions: player state
  rings (CONTROLLING_BALL / INACTIVE_TACKLED), ball state rings (flying /
  rolling / just-bounced via `ball.just_bounced_timer_s`), `draw_game_log`
  (scrolling bottom-right log box), `draw_scenario_params` (param editing
  screen with +/− buttons).
- `gamelog.py` - `LogLevel` enum (`INFO`, `DEBUG`) and `GameLog` (a
  `collections.deque` ring buffer of `LogEntry` objects). Zero cost in
  headless use — `Match.log_callback` is `None` by default and checked
  before any import or allocation. The UI attaches a callback in
  `App._wire_match_log()` and wires a new one each time a new trial is built.
- `input.py` - `MatchInputController` translates raw mouse events into
  orders on the engine's `Player.current_order`. See its docstring for the
  full click/drag interaction scheme (click player to select, click ground
  to move, click opponent to tackle, drag from the ball-carrier to kick).
- `scenarios.py` - builds `Match` instances for the two non-freeplay modes:
  `make_training_match()` (1 player + ball, full pitch, both goals live) and
  `SCENARIOS` (six parameterized balance scenarios — see "Scenarios" section
  below). Also houses `ScenarioParam`, extended `ScenarioDefinition`, and
  `ScenarioLoop` with linger support.
- `app.py` - `App` owns the pygame window, main loop, a three-screen state
  machine (`MENU` / `SCENARIO_PARAMS` / `MATCH`), and wires input events to
  `MatchInputController` + `Match.step()` + `Renderer` + `GameLog`.

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
- **CONTROLLING_BALL** (`PlayerState.CONTROLLING_BALL`): a cyan
  `CONTROL_DELAY_OUTLINE` ring, visually distinct from the possession ring.
  Indicates the player is mid first-touch control delay and can't yet move
  or be given orders. (Phase G)
- **Inactive players** (`PlayerState.INACTIVE_TACKLED`): translucent fill
  (`INACTIVE_ALPHA`) **plus** a red `INACTIVE_OUTLINE` ring, so the state
  is visible even when the player blends into the pitch. (Phase G)
- **Top layer for ball carrier**: `App._draw_match` sorts the player list so
  whichever player has the ball is drawn last, i.e. on top of every other
  player - avoids the possession ring/player circle being partially
  obscured by a nearby defender drawn afterwards.

## Ball state indicator rings (Phase G)

`renderer.draw_ball` draws a thin outline ring on top of the ball circle
indicating its current state.  Priority (only one ring shown at a time):
- **Amber** (`BALL_STATE_BOUNCED_OUTLINE`): `ball.just_bounced_timer_s > 0` —
  ball made a real bounce recently (decays after 0.3 s).
- **Blue** (`BALL_STATE_FLYING_OUTLINE`): ball is airborne (`z > 0.05 m`) and
  not possessed.
- **Green** (`BALL_STATE_ROLLING_OUTLINE`): ball is on the ground and rolling
  (`speed_xy > 0.05 m/s`) and not possessed.
No ring is drawn when the ball is possessed or stationary on the ground.

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

## Scenarios (Phase H)

`SCENARIOS` lists six `ScenarioDefinition` objects, all fully parameterized:

| key | Description |
|---|---|
| `save_close` | Shot vs GK, randomised distance/attributes |
| `pass` | Ground pass, randomised distance/angle |
| `tackle` | Defender chases jogging attacker, randomised separation |
| `sprint` | Random 5-waypoint course across the pitch |
| `2v2` | Attacker A passes to B, then B shoots; one defender + GK |
| `1v2` | Elite attacker runs then shoots vs. average defender + GK |

The older fixed scenarios (penalty, far-post save, no-keeper shoot) were
removed from `SCENARIOS`; they still exist as private `build_*` helpers if
needed for reference.

### `ScenarioParam` and `ScenarioDefinition`

```python
@dataclass(frozen=True)
class ScenarioParam:
    name: str        # kwarg name passed to build_*(rng_reduction, **kwargs)
    label: str       # UI display text
    min_value: float
    max_value: float
    step: float
    default: float   # used as the kwarg default AND as the UI seed value
```

`ScenarioDefinition` now carries:
- `params: list[ScenarioParam]` — the adjustable knobs for this scenario.
- `on_tick: Callable[[Match, int], None] | None` — called by `ScenarioLoop`
  **before** `match.step()` each tick. Used by `sprint` (waypoint sequencing
  via `SprintController`), `2v2` (`TwoVTwoController`), and `1v2`
  (`OneVTwoController`) to drive scripted multi-step behaviour.

Selecting a scenario with a non-empty `params` list from the menu goes to
`Screen.SCENARIO_PARAMS` first, not straight to `Screen.MATCH`. Scenarios
with an empty list would skip straight to MATCH (reserved escape hatch; all
current scenarios have params).

### `ScenarioLoop` linger (Phase H)

After an outcome is detected `ScenarioLoop` continues stepping the match for
`linger_s` sim-seconds before rebuilding (so players/ball keep moving and the
goal stays visible). During the linger `step()` returns `False`; it returns
`True` only once the linger expires and the new trial is ready.

- **Goal / saved / dispossessed / other**: full `linger_s` (default
  `physics.json["ui"]["scenario_linger_s"]` = 3.0 s).
- **Out-of-bounds miss** (ball crosses touchline or far end): **half**
  `linger_s` (1.5 s by default) — brief pause to see the ball leave, but
  not a full celebration wait.
- **Settled-ball miss** (ball stopped while loose after a shot/pass): half
  `linger_s` too, same path.

Tests pass `linger_s=0.0` explicitly to skip the wait and keep test runs
fast; see `tests/scenario/test_scenario_loop.py` for linger-specific tests.

`ScenarioLoop.outcomes` accumulates
`{'goal', 'saved', 'miss', 'dispossessed', 'other'}` counts. `dispossessed`
is new: ball repossessed by the non-attacking team before any shot is taken
(e.g. defender wins a tackle or intercepts the ball). HUD shows these as
`Goals: N  Saved: N  Miss: N  Disp: N`.

## Game log (`gamelog.py` / `App._wire_match_log`, Phase G)

`GameLog` is a `collections.deque(maxlen=50)` of `(time_s, level, message)`
entries. `App._wire_match_log(match)` attaches a closure to
`match.log_callback` that calls `game_log.add(level, msg, match.time_s)`.
A new closure is attached each time a new match is built (new trial, training
mode start) so old match references don't leak.

`renderer.draw_game_log(surface, game_log, min_level)` draws the most recent
entries in a semi-transparent box in the bottom-right corner, newest at the
bottom. `L` hotkey in-match cycles `App.log_min_level` between `INFO` and
`DEBUG`. DEBUG entries include full numeric roll breakdowns from tackles.

## `Screen.SCENARIO_PARAMS` (Phase H)

New `App` state between `MENU` and `MATCH` for parameterized scenarios:
- `App._pending_scenario_definition` and `_pending_scenario_params` hold the
  selected scenario and current knob values.
- `_draw_params_screen()` delegates to `renderer.draw_scenario_params()`,
  which renders a vertical list of param rows with `−`/`+` buttons and stores
  the button rects in `App._params_button_rects`.
- `_handle_params_click()` applies `±step` clamped to `[min, max]` for each
  param; **Start** calls `_start_scenario_with_params()`; **Back** returns to
  MENU.
- `Esc` also returns to MENU from this screen.

## Balance scenario looping (`ScenarioLoop` in `scenarios.py`)

`ScenarioLoop` wraps a `ScenarioDefinition` and replays it indefinitely
(default `max_trials=0`, meaning run forever; any positive value stops after
that many trials and returns to the menu).  The UI calls `loop.step()` once
per frame; each call (a) calls `definition.on_tick(match, tick)` if set,
then (b) advances the current trial's `Match` by one physics tick.

A trial ends when:
1. Ball crosses the touchline or goal line (OOB/goal).
2. Scoreboard changed (goal scored).
3. The initial ball carrier has released the ball AND the ball is since
   possessed by someone else, gone dead, or a goal was recorded.
   This correctly ends GK-save trials the instant the keeper controls the
   ball, without waiting for `SaveOrder` to complete (it never does).
4. Defender repossesses ball while no shot is in flight (`dispossessed`).
5. All non-persistent orders resolved and ball stationary (covers orderless
   scenarios like sprint).
6. `timeout_ticks` failsafe (default 500, ≈ 16.7 s at 30 Hz).

The HUD shows `Trial N  |  Playing` (no `/max`) when running indefinitely,
or `Trial N/max_trials` when a finite count was specified.

## Training mode auto-select

`App._start_match` now sets `input_controller.selected_player_id` to
the lone trainee's ID immediately when `is_training_mode=True`, so the
player is controllable from the first frame without an initial click.

## Training mode neural control (`N` hotkey)

Training mode's trainee can be switched between human control and
neural-network control at any time via the `N` hotkey
(`App._toggle_training_ai_mode`), which cycles
`human -> neural(checkpoint 1) -> neural(checkpoint 2) -> ... -> human`.
Checkpoints are discovered via `scenarios.discover_all_phase1_checkpoints()`
(scans every `checkpoints/phase1_run*/` dir, same discovery/sort logic as
the Phase 1 scenario's checkpoint dropdown) and loaded on demand via
`scenarios.load_trainer_for_ui()` (thin wrapper around the private
`_load_trainer()`, itself a thin wrapper around
`PPOTrainer.load_for_inference()`), cached per-path in
`App._training_trainer_cache` so repeated toggles don't reload the network.

Neural control is implemented by assigning `rules_ai.HybridPlayerAI` to
`player.ai` — nothing else in the UI's per-frame loop needs to know the
mode; `Match.step()` calls `player.ai.act()` automatically like any other
`PlayerAI`, same as the rules-based scenarios. `HybridPlayerAI` extends
`NeuralPlayerAI` with two override channels (order override + decision-
neuron override) so the human click/kick-UI input path keeps working
identically in either mode: `MatchInputController._issue_order()` and the
kick UI (`ui/input.py`) both detect `isinstance(player.ai, HybridPlayerAI)`
and route clicks through `HybridPlayerAI.issue_order()` (the order-override
channel) instead of writing `player.current_order` directly — a click on a
neural-controlled trainee "takes over" for exactly one order (Move/Shoot/
Pass/Kick/Save/Stop), after which control reverts to the network
automatically once the engine clears `player.current_order` back to `None`.
See `ai/knowledge.md`'s "HybridPlayerAI" section for the full design
(including the decision-neuron override channel, not yet wired to any UI
control but usable programmatically/from tests).

The hotkey bar shows `[N] Neural AI` / `[N] Human` (highlighted when
neural) only while in training mode (`App._hotkey_entries`); the help
overlay (`H`) documents the hotkey too. `App._start_match` resets
`_training_checkpoint_idx = -1` whenever a new training match is built, so
every fresh training session always starts in human control.

## Coordinate convention — critical pitfall

All x-coordinates are measured from the **pitch centre** (origin), not from
a goal line.  The left goal line is at `x = -pitch.half_length` (≈ -52.5m
on a standard 105m pitch).  A value like `x = -22` places the player 22m
from the *centre*, which is 30.5m from the left goal — **not** 22m from
goal.  Use `-(pitch.half_length - distance_from_goal)` whenever you mean a
specific distance from a goal line, e.g. `-(pitch.half_length - 25.0)` for
25m out.  This mistake has bitten the save-balance tests and scenarios
multiple times.

## Training mode goal linger

`App._step_match` watches for a goal by comparing the scoreboard tally each
tick. It only calls `_reset_training_positions()` (repositions the player)
once `match._goal_linger_remaining_s <= 0.0` — i.e. after the engine's own
goal-linger countdown has expired and `_reset_after_goal()` has already run.
This keeps the UI reset in sync with the engine reset rather than racing it.
`make_training_match()` sets `goal_linger_s` from `physics.json["ui"]["goal_linger_s"]`
so the ball stays in the net for the same duration as scenario trials.

## Known gaps / not yet implemented

- No jog/sprint toggle for Move orders (always sprint).
- Shoot mode (`K`) always fires at full power (`power_fraction=1.0`) with a
  fixed aim height of 1.0m; no UI control for power or aim height yet.
- No sound, no game clock/timer, no formations/kickoff - out of scope for
  this milestone (mirrors the engine's own documented gaps).
