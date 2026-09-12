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

## Windows DPI awareness (`app.py::_make_process_dpi_aware`)

Called once, before `pygame.init()`, in `App.__init__`. On Windows, a
process that hasn't declared itself DPI-aware gets its whole window
bitmap-rescaled by the desktop compositor to match the display's scaling
factor (125%/150% are the common non-100% defaults on laptops and many
external monitors) — the app itself still renders at "logical" pixel
coordinates and never sees this. That post-hoc resampling is what looked
like players "having a chunk eaten out of them": a user-supplied screenshot
inspected pixel-by-pixel showed a multi-pixel *gradual* colour blend across
the edge of a player's circle, not the crisp 1-2px transition
`pygame.gfxdraw.aacircle` actually produces — a gradual blend spanning
several pixels is the signature of a resampling filter, not anything drawn
by this renderer. It was worse on one side (bottom-right, matching the
original bug report) because the resampling kernel's alignment relative to
the source pixel grid is consistent across the whole window for a given
scale factor, so every circle gets the same directional bias. `SetProcess
DpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)` (falling back to the older
`SetProcessDPIAware` on failure, and silently no-op-ing on non-Windows or
if neither API is available) tells Windows to hand our window its own
native pixel buffer instead of rescaling it after the fact. This can't be
verified by rendering to an offscreen/dummy-driver surface the way the
rest of this file's rendering bugs were — there's no compositor involved
in that path — it has to be checked by actually running the app on a
scaled display.

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

Both floors are scaled by `Camera.zoom_scale` (`pixels_per_metre` as a
multiple of the fit-to-pitch baseline — 1.0 normally, `zoom_factor` under
the `[Z]` ball-follow zoom, see below) rather than used as flat pixel
constants. Without this, whichever entity is smaller in world units (the
ball, 0.11m, vs a player's 0.3m) stops growing under zoom as soon as its
true-to-scale size overtakes its *unscaled* floor — which happens much
sooner for the ball than for players — leaving the ball looking
disproportionately tiny next to zoomed-in players even though it visually
grew the least of anything on screen. Scaling both floors together keeps
their relative sizes roughly constant across zoom levels.

## Pitch dressing (`draw_pitch` — nets, defending-side markers, corner flags, benches)

Purely cosmetic additions with no gameplay meaning, all drawn as part of
`draw_pitch` (so they redraw fresh every frame along with the pitch lines,
same as everything else in that method):

- **Goal netting** (`_draw_goal_net`): fills each goal frame's rectangular
  footprint (`goal_depth_m` × `goal_width_m`) with a translucent diagonal
  X-hatch, drawn on an isolated SRCALPHA surface sized to the box so the
  diagonal lines clip cleanly at its edges. Colour/density/opacity are
  configurable via `graphics.json["goal_net"]` (`spacing_m`, `alpha`,
  `color`) rather than flat `style.py` constants, since "barely visible" was
  the first complaint about it — `style.GOAL_NET_COLOUR`/`GOAL_NET_ALPHA`
  now only serve as that config's fallback defaults. In this top-down view
  the goal's back panel and its two side panels all project onto the same
  rectangle, so one hatch fill reads as netting on all three sides at once
  — there's no separate "side netting" to add; before this there was no net
  graphic at all (front, back, or side) — the goal was a bare white-line
  outline, and the ball simply disappearing there on a goal is what read as
  "the net working".
- **Defending-side markers** (`_draw_defending_marker`): a translucent bar
  in the defending team's colour, set back a little (`gap_m`) behind each
  goal net rather than flush against it. Fixed per the engine's
  LEFT-attacks-+x / RIGHT-attacks-−x convention (see `engine/knowledge.md`
  and `actions.py::opponent_goal_centre`) — the left goal (x=−half_length)
  is Team.LEFT's own goal and gets `style.TEAM_LEFT_COLOUR` (blue); the
  right goal gets `style.TEAM_RIGHT_COLOUR` (red). If that attack-direction
  convention is ever flipped, this needs to flip too.
- **Corner flags** (`_draw_corner_flags`): a small pole dot + pennant
  triangle at each of the 4 pitch corners, leaning inward over the pitch so
  they're never clipped by the window edge at small margins.
- **Sideline benches** (`_draw_sideline_benches`): a row of benches (`x`
  positions in `Renderer._BENCH_X_OFFSETS_M`, spread across the middle
  third of the pitch, clear of the boxes/corners regardless of pitch size)
  just outside each touchline. Decorative only — no technical-area gameplay
  concept exists.

## Ball height display

Per the original design spec ("make the ball change size with height (but
maybe exaggerate the effect) and have a small number on it showing its
height in metres"): `draw_ball` boosts the ball's radius by
`1 + min(height, 5)*0.35` and renders a `"{height:.1f}m"` label next to it
whenever height exceeds 0.15m.

## Ring drawing (`Renderer._draw_ring`) — supersampled, not `pygame.draw.circle(..., width=N)`

Every circular outline (selection, possession, control-delay/inactive,
stamina-flash, and the ball's outline + state rings) is drawn via the shared
`Renderer._draw_ring()` helper, **not** a direct `pygame.draw.circle(...,
width=N)` + `gfxdraw.aacircle` pair. At the small radii these rings use
(~8-20px), pygame-ce's stroked-circle rasteriser is visibly ragged: sampling
real rendered frames pixel-by-pixel found genuine isolated single-pixel gaps
of pure background punched through an otherwise-solid ring (worse on the SE
side, lesser on NW) — this is what read as players "having a chunk eaten out
of them" at normal zoom, since the ring sits right up against the player's
own fill circle. Building the ring from two separately-rasterised filled
circles (opaque outer disc + transparent inner punch) does not fix it either
— their staircase edges don't line up in phase, turning one stray hole into
a visibly dashed/toothed ring instead. The fix that actually works: render
the ring at `_RING_SUPERSAMPLE` (4x) resolution and downscale with
`pygame.transform.smoothscale`, which area-averages instead of sampling one
point per pixel, so leftover rasteriser noise blends into a soft
anti-aliased edge instead of surviving as a hole. Any *new* circular outline
added to the renderer should go through this helper rather than a fresh
`draw.circle(..., width=N)` call.

## Ball spin dots (`draw_ball`'s dot-projection block)

The ball's surface dots (fixed points on a Fibonacci-lattice unit sphere,
rotated each frame by `_ball_orientation` and projected top-down) are drawn
as **projected tangent-plane patches, not flat discs of constant size**.
Each dot's outline is built from `_DOT_POLY_SEGMENTS` points arranged in a
small circle in the dot's own tangent plane (`u`/`v`, both perpendicular to
its centre direction `(wx,wy,wz)`), pushed through the *same* rotation +
orthographic projection as the centre point, then filled/antialiased with
`gfxdraw.filled_polygon` + `gfxdraw.aapolygon`. This is what makes dots near
the ball's silhouette edge correctly foreshorten into ellipses (a real
football's panels do the same under orthographic projection), while dots
near the visible "pole" stay circular — the shape falls out of the
projection automatically, with no separate squish-factor/rotation-angle
math needed. Before this, dots were plain `pygame.draw.circle` calls of a
single fixed radius with no antialiasing at all (visibly jagged, and flat
rather than sphere-like) — if you need to touch this code again, keep both
properties (foreshortening + AA) rather than reverting to a flat circle.

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
- **Heading indicator**: a broad, thin "V" — two unfilled lines touching
  the rim at wide-spread points and meeting at a point just ahead in the
  facing direction (`draw_player`'s "Heading indicator" block, colour
  `style.HEADING_INDICATOR_COLOUR`, geometry tunable via
  `graphics.json["heading_indicator"]`: `length_px` = how far the meeting
  point sits beyond the rim, `base_half_width_px` = how wide the two rim
  contact points are spread (bigger = broader V), `base_inset_px` = how far
  inside the rim those contact points sit (0 = exactly on the rim),
  `alpha`). Went through two earlier designs: a thin line from the player's
  *centre* out past the rim (read as a stray line poking through the fill),
  then a filled triangle badge (read as too heavy/blocky) — this is
  deliberately just two thin strokes, no fill, no outline.

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

### Choice-param dropdowns (`ScenarioChoiceParam` / `ScenarioGroupedChoiceParam`)

Both live in `scenarios.py` alongside `ScenarioParam`/`ScenarioBoolParam` (see
`AnyScenarioParam`). Both render as a `value ▼` box + `[>]` cycle button and
open a dropdown list below the row when clicked; `renderer.draw_scenario_params`
now returns `(button_rects, clamped_dropdown_scroll)` rather than just
`button_rects` so state can persist the clamp.

- **`ScenarioChoiceParam`** — flat list of string options (e.g. the tier
  dropdowns). Dropdown option rects: `f"{name}__option__{value}"`.
- **`ScenarioGroupedChoiceParam`** — options pre-partitioned into
  `groups: tuple[(group_label, (value, ...))]`; the build function still
  receives a single flat value string (grouping is UI-only). Used for the
  Phase 1 scenario's `trainee_checkpoint`/`opponent_checkpoint` pickers,
  grouped by checkpoint directory (`phase1_runN` / `longterm`), since a flat
  list across every run quickly exceeds what fits on screen. Clicking the
  value box opens the **group list** first (`f"{name}__folder__{group}"`
  rects); clicking a group opens that group's **value list** below a
  `f"{name}__grpback__"` back row (`f"{name}__option__{value}"` rects, same
  key format as the flat variant so `App._handle_params_click`'s selection
  branch is shared). `[>]` still cycles the fully-flattened option space
  (`param.flat_choices()`) without opening the dropdown.
- Both dropdown kinds are **scrollable**: at most 10 items render at once
  (`MAX_VISIBLE_ITEMS` / `ITEM_H` in `renderer.draw_scenario_params`), with a
  thumb-style scrollbar drawn when the list is truncated. `ScenarioParamsUIState.dropdown_scroll`
  (`app.py`) holds the offset, advanced by `pygame.MOUSEWHEEL` while a
  dropdown is open and reset to 0 whenever a dropdown opens or the expanded
  group changes; `renderer` re-clamps it every frame against the
  currently-visible list's length (list length can shrink when switching from
  a folder list to a shorter value list).
- `ScenarioParamsUIState.open_choice_folder` tracks which group is expanded
  for a `ScenarioGroupedChoiceParam` (`None` = showing the group list).

## Phase 1 UI scenario vs. actual PPO training conditions (`_make_phase1_scenario_pair`)

This scenario lets a human load a checkpoint and watch it play, so its
defaults should put the network in conditions matching what it was
*trained* under — otherwise you're evaluating it out-of-distribution
without realising it. Two of its numeric defaults used to be independently
hardcoded instead of sourced from `ai_config.json`, and had drifted from
the real training values:

- **`decision_interval_ms`** (default of the same-named `ScenarioParam`):
  now `_phase1_training_cfg()["decision_interval_s"] * 1000` (currently
  249.9ms) instead of a hardcoded `500.0`. Training's own decision cadence
  comes from `ai_config.json["observation"]["decision_interval_s"]`
  (`0.2499s`) combined with its own `sim_dt_s` (`0.05s`/20Hz) →
  `ScenarioEnv` computes `ticks_per_decision = round(0.2499/0.05) = 5`
  ticks, i.e. a real-world cadence of `5 * 0.05 = 0.25s`. The UI
  deliberately ticks the engine at a fixed 30Hz regardless (`UI_TICK_HZ`,
  for smooth human-visible playback — see `ai_config.json`'s own
  `_comment_sim_dt_s`: "UI ignores this and always uses 30Hz"), so matching
  training means matching the real-seconds cadence, not the tick count —
  `decision_interval_ticks = round(ms/1000 * 30)` is computed fresh from
  whatever `decision_interval_ms` the slider holds, so keeping the
  **default** in real seconds correct is what matters. The old hardcoded
  500ms was exactly 2x training's real cadence.
- **`max_episode_s`** passed to both players' `NeuralPlayerAI`: now
  `_phase1_training_cfg()["max_episode_s"]` (currently `18.5`, from
  `ai_config.json["curriculum"]["phase1_max_episode_s"]`) instead of a
  hardcoded `1e9`. `NeuralPlayerAI` derives a `time_remaining` observation
  feature from this (`rules_ai.py`: `max(0.0, max_episode_s -
  episode_ticks/30.0)`, clamped to 0 rather than going negative, so a UI
  trial running longer than 18.5s just saturates at "no time left" the same
  way a training episode nearing its cap would — no special-casing needed
  if a trial overruns it). Leaving this at `1e9` effectively froze that
  feature at a huge constant the whole time, so the network never saw the
  same time-pressure signal in the UI that it saw throughout training.

Both are sourced via `_phase1_training_cfg()`, which reads
`ai_config.json`'s `observation`/`curriculum` sections directly (distinct
from `_phase1_scenario_cfg()`, which only covers the *randomised-scenario*
knobs — tiers, ball speed/distance, stamina range, restitution — that
`build_1v1_scenario` already shares between the UI and
`ai/curriculum/envs.py`'s actual training env construction, so those were
never actually mismatched).

**`opponent_rules` default is now `False`** (neural, driven by
`opponent_checkpoint`'s dropdown default — the same latest-checkpoint
default `trainee_checkpoint` already used) rather than `True`
(rules-based). Training's curriculum ratios
(`ai_config.json["curriculum"]`: `phase1_opponent_rules_ratio=1` /
`_immobile_ratio=0` / `_neural_ratio=3`, normalized in
`ai/curriculum/envs.py`'s `opponent_type_probs()`) put the trainee against
a live neural self-play opponent 75% of the time during training,
rules-based only 25%, never immobile — so neural is the closer default to
training's dominant condition, even though no single boolean default can
reproduce the full 75/25/0 mix. This was a deliberate choice (flagged to
and confirmed by the user, favouring "closer to training" over "a fixed,
interpretable benchmark opponent") — `opponent_rules`/`opponent_immobile`
checkboxes are both still there for anyone who wants the old deterministic
rules-based benchmark back.

**Known remaining divergence, left as-is deliberately (not a bug)**:
Physics tick rate — UI always 30Hz vs. training's 20Hz (`sim_dt_s=0.05`).
Explicitly documented as intentional in `ai_config.json`'s own comment,
kept for smooth human playback; confirmed with the user rather than
changed.

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

## Ball-follow zoom (`Z` hotkey / `[-]`/value/`[+]` zoom control)

`Camera` supports a second mode besides the default "whole pitch always
fits the window" fit-to-pitch view: `zoom_factor > 1.0` (`zoomed` is a
derived bool, `zoom_factor > 1.0`) boosts `pixels_per_metre` by
`zoom_factor` and recentres the view on a world point every frame via
`Camera.follow()`, instead of centring on the pitch. This is a
**continuous, steppable level, not a toggle** — `Camera.set_zoom_level(level)`
sets it directly (`level <= 1.0` snaps back to the normal fit-to-pitch
view). `App._ZOOM_LEVELS` is a linear sequence from 1.0 up to
`graphics.json["camera"]["max_zoom"]` (default 5.0) in `zoom_step`
increments (default 0.5), built once in `__init__` the same way
`_SIM_SPEED_OPTIONS` is; `App._cycle_zoom(direction)` snaps to the nearest
level and steps (wrapping at either end, same as `_cycle_sim_speed`),
shared by the `Z` hotkey (always steps forward) and the on-screen
`Renderer.draw_zoom_control`'s `[-]`/`[+]` buttons (drawn immediately to
the *left* of the sim-speed control so neither it nor the value box
collides with the player-inspector panel that appears just below that
row — the value box itself highlights in the accent colour whenever
zoom is actually engaged). `App._draw_match` calls
`self.camera.follow(ball.x, ball.y)` once per frame whenever
`camera.zoomed` — so the view always tracks the ball, not any particular
player. `resize()` keeps `zoom_factor` applied consistently across
window-size changes while zoomed (only the *offset* is left for the next
`follow()` call to fix, since it's about to be overwritten anyway). Zoom
state persists across matches/trials (not reset by `_start_match`/
`_start_scenario`), same as `_sim_speed`.

`_SIM_SPEED_OPTIONS` itself is linear 0.25 steps from 0.25x to 8x
(`tuple(round(i * 0.25, 2) for i in range(1, 33))`), not the old doubling
sequence (0.2/0.5/1/2/4/8) — finer control around normal speed, at the
cost of more `]`/`[`/click steps to reach the extremes.

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
