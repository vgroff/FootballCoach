# ui/

A pygame-ce-based renderer and input layer sitting on top of the headless
engine (`engine/match.py`). Nothing in here touches simulation logic - it
only reads `Match`/`Player`/`Ball` state to draw, and issues orders
(Move/Pass/Shoot/GetPossession/Save/Stop) via `player.current_order` (or
`HybridPlayerAI.issue_order()` for neural-controlled players) plus direct
`player.kick_with_direction()` calls from the kick UI. This keeps the UI
swappable/removable without touching the engine, per the project's
"engine is UI-agnostic" design principle.

## Files

- `camera.py` - `Camera` maps between world metres (engine convention: `x` =
  pitch length axis, `y` = pitch width axis, origin at pitch centre) and
  screen pixels (origin top-left, `y` grows downward). The window is
  auto-sized to fit the whole pitch plus a margin at a fixed
  `pixels_per_metre` (`Camera.fit_to_pitch`). `world_to_screen` returns
  truncated ints; `world_to_screen_f` returns the same mapping un-truncated,
  for drawing smooth curves.
- `style.py` - colour palette and rendering constants, including Phase G
  additions: `CONTROL_DELAY_OUTLINE` (cyan ring for `CONTROLLING_BALL`),
  `INACTIVE_OUTLINE` (red ring for `INACTIVE_TACKLED`), and three ball-state
  ring colours (`BALL_STATE_FLYING_OUTLINE`, `BALL_STATE_ROLLING_OUTLINE`,
  `BALL_STATE_BOUNCED_OUTLINE`).
- `renderer.py` - pure drawing functions. Phase G additions: player state
  rings (CONTROLLING_BALL / INACTIVE_TACKLED), ball state rings (flying /
  rolling / just-bounced via `ball.just_bounced_timer_s`), `draw_game_log`
  (scrolling bottom-right log box), `draw_scenario_params` (param editing
  screen with +/− buttons). Also: `draw_kick_ui` (trajectory + error cone
  preview for the multi-phase kick UI), `draw_player_inspector` (side panel,
  see "Player inspector panel"), `draw_pause_notification` (banner above
  the hotkey bar), `draw_speed_control`/`draw_zoom_control`.
- `player_sprites.py` - procedurally-drawn top-down player sprites (head,
  shoulders, torso, animated legs) used by `Renderer.draw_player` in place
  of the plain circle; see "Player visual indicators". Its module docstring
  holds the design rationale (supersample-then-`smoothscale`, 9 cached poses
  per shirt colour, speed-driven stride animation).
- `sim_clock.py` - `SimTimeDelta`: sim seconds elapsed between rendered frames,
  from the match's own `time_s` (resyncing across a new match/trial or a clock that
  ran backwards). Drives the cosmetic animations; see "Animation clock".
- `kick_trajectory.py` - pure-math helpers for the kick UI preview:
  forward-simulates the ball with the engine's own `ball_physics.step_ball`
  (deterministic, no random kick error), builds the 1-sigma error cone
  (`compute_error_sigma`/`build_cone_boundaries`), maps height to colour
  (`height_to_colour`), and maps mouse position to a spin vector
  (`spin_from_mouse`). Its module docstring has the spin-axis convention.
- `gamelog.py` - `LogLevel` enum (`INFO`, `DEBUG`) and `GameLog` (a
  `collections.deque` ring buffer of `LogEntry` objects). Zero cost in
  headless use — `Match.log_callback` is `None` by default and checked
  before any import or allocation. The UI attaches a callback in
  `App._wire_match_log()` and wires a new one each time a new trial is built.
- `input.py` - `MatchInputController` translates raw mouse events into
  orders on the engine's `Player.current_order`. See "Interaction scheme"
  below (left-click selects/moves, right-click issues Get Possession, click
  the selected ball-carrier to enter the multi-phase kick UI).
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
grew the least of anything on screen.

**The floors don't scale with a flat `zoom_scale` any more, though —
`graphics.json["size_floor"]["zoom_decay"]` (0.45): `floor_px = min_radius_px *
zoom_scale ** zoom_decay`.** A flat `* zoom_scale` (`zoom_decay = 1.0`, the old
behaviour) keeps the floor growing at *exactly* the same rate as true-to-scale
size — so once the floor wins the `max()` at 1x (it always does: measured
~4.6x oversized for the ball, ~2.0x for a player, both floor-dominated), it
keeps winning by that SAME ratio forever, at every zoom, since neither side of
the `max()` ever catches up to the other. The ball/player never look properly
life-sized-relative-to-the-accurately-scaled-pitch no matter how far you zoom
in — confirmed by computing the ratio at zoom 1/2/3/4/5 before touching
anything (constant 4.64x / 2.00x throughout). A `zoom_decay < 1` floor still
equals `min_radius_px` at zoom 1 (`1 ** anything == 1`, so nothing changes at
the default view) but grows SLOWER than true size as you zoom in, so true
size increasingly overtakes it — the drawn size converges toward actually
life-sized (measured with `zoom_decay=0.45`: ball 4.64x → 2.07x oversized by
zoom 5, player 2.00x → exactly 1.00x, i.e. true-to-scale). Both entities share
the SAME exponent (rather than, say, only decaying the ball's) so they
approach true scale together instead of one catching up much sooner than the
other and briefly looking wrong-sized next to each other along the way — the
original rationale above (why the floor scales with zoom AT ALL, rather than
being a flat pixel count) still holds, just with the growth rate slowed
rather than removed. `_ball_base_radius_px` / `_player_radius_px` are the two
(and only two) places this floor is computed — every shadow, ring, sprite
scale and the goal-frame's own shading all derive their radius from one of
these, so the fix applies everywhere automatically. Doesn't touch hit-testing
(`input.py`'s click tolerance uses the player's true `radius_m` directly, no
floor at all, already independent of the drawn radius).

## Pitch markings (`draw_pitch`)

The standard Law 1 line markings, all drawn every frame in `draw_pitch` at
the shared line width (`0.12m * pixels_per_metre`, min 1px): boundary,
halfway line, centre circle + **centre spot**, both penalty boxes and
six-yard boxes, both **penalty spots**, both **penalty arcs** (the "D": the
part of the 9.15m circle round each penalty spot that lies *outside* the
box, meeting the box's front edge at y = ±√(r² − dx²) with dx = box length −
spot distance = 5.5m), and the four **corner arcs** (1m quarter circles
inside each corner). Tunables are in `graphics.json["pitch_markings"]`
(`spot_radius_m`, `penalty_arc_radius_m`, `corner_arc_radius_m`) — cosmetic
only, deliberately *not* on the engine's `Pitch` dataclass, which has many
positional constructors that a new required field would break.

- Spots (`_draw_pitch_spot`) are filled + anti-aliased circles, true to scale
  with a 2px floor (`spot_radius_m` is 0.18 rather than the real ~0.11m so a
  spot reads as distinct from the line width).
- Arcs are dense world-space polylines from the pure functions
  `penalty_arc_world_points` / `corner_arc_world_points` (3° steps), drawn by
  `_draw_world_polyline` with sub-pixel vertices (`Camera.world_to_screen_f`).
  This deliberately avoids `pygame.draw.arc`, whose thick strokes tear into
  gaps at high zoom. 1px lines use `aalines`, thicker ones `draw.lines`.
- **Corner arcs are clipped to the pitch.** The boundary is drawn INSIDE its rectangle
  (`rect_world`, `pygame.draw.rect(width=line_w)`), but each arc is a stroke centred on world
  points that lie on the corner lines' OUTER edge, so the stroke's thickness and anti-aliased
  edge poked past the boundary lines — measured 1 px outside at zoom 1 and 3-6 px at zoom 2-3
  (none at 5x) at some corners. `draw_pitch` sets the surface clip to exactly the boundary
  rectangle while drawing the four arcs. Tests: `test_corner_arcs_never_poke_out_past_the_boundary_lines`
  (6 zooms x 4 corners), `test_corner_arcs_are_still_drawn_inside_the_pitch` and
  `test_clipping_the_corner_arcs_removes_only_what_is_outside_the_pitch` (an over-tight clip
  would detach the arc from its lines).
- The geometry generators are unit-tested in
  `tests/unit/test_pitch_markings.py` (points on the circle, arc entirely
  outside the box and ending exactly on its edge, left/right mirror,
  corner arcs inside the pitch). The drawing was checked by rendering to an
  offscreen surface at full view and 3–4x zoom.

## Pitch dressing (`draw_pitch` — nets, defending-side markers, corner flags, benches)

Purely cosmetic additions with no gameplay meaning, all drawn as part of
`draw_pitch` (so they redraw fresh every frame along with the pitch lines,
same as everything else in that method):

- **Goal depth is the engine's**: `draw_pitch` uses `pitch.goal_depth_m`
  (2.45m, `physics.json`) for the net footprint, net hatch and defending
  marker — the same value `engine/ball_physics.py::resolve_goal_boundary`
  puts the back wall at. It used to be a hardcoded 2.0m, so the ball visibly
  passed through the drawn back of the net before bouncing off an invisible
  wall 0.45m further on. Tested by `test_drawn_net_depth_matches_the_engines_back_wall`
  (checks the outline pixel column at engine depth, and *no* line at 2.0m).
  The engine models each goal as a box: back wall at that depth, side posts
  at ±goal_width/2, and crossbar/roof at `goal_height_m` (2.44m) over the
  whole depth.
- **3D goal frame** (`_draw_goal_footprint` + `draw_goal_tops`): a slight
  parallax, since the view is top-down. The crossbar (2.44m up) is drawn
  displaced `graphics.json["goal_frame"]["crossbar_lean_m"]` (0.72m; was 0.9, cut ~20%
  to tone the effect down) from the
  goal line **away from the pitch** (over the net) instead of hidden exactly
  on top of it, which reveals the goal mouth — the opening between the goal
  line on the ground and the crossbar — with the posts drawn as the lines joining their feet
  (small dots on the goal line) to the crossbar ends, all at `post_width_m`
  (0.20m; thinned from 0.30 then 0.25 on request) — but never less than 2px
  thicker than the net's own 1px lines (`_goal_post_px`), so the frame always
  reads heavier than the net at any zoom (3px vs 1px at the default window —
  the floor, so unchanged by further thinning `post_width_m`; 5px at 3x, 7px at
  4x). A 2px post at 1x would be only 1px heavier than the net. "Away from the pitch" is what a real overhead/broadcast view
  does to tall objects (they lean away from the point under the camera); the
  opposite lean would put the frame sticking out over the pitch. The offset is
  **purely apparent** — clamped to the net depth, and the ball's ground truth
  is still the goal line (`Pitch.is_goal`); `crossbar_lean_m: 0` restores the
  flat look (net over the whole footprint).
  - **The posts and crossbar are shaded to look round** (`goal_frame.shade_strength`,
    0.7; 0 = the old flat white). The scene light is `style.LIGHT_DIR_XY`
    (toward the light, screen space: the up-left diagonal, exactly 45° so a post
    and the crossbar shade identically) at `style.LIGHT_ELEVATION_DEG` (55°) —
    shared constants so any later shading (shadows, ball, sprites) stays
    consistent. `surface_shade` is Lambert plus an
    ambient floor (`_SHADE_AMBIENT`) with a small gain (`_SHADE_GAIN`, so a
    cylinder's best-lit strip actually reaches full brightness), blended toward
    flat by the strength. `cylinder_shades(n_px, across, strength)` gives one
    brightness per pixel across a bar (each averaging 8 sub-positions of the
    cross-section, so a 3px bar at 1x gets three sensible tones); posts run
    along x so their width is across y ("y"), the crossbar the reverse.
    **The frame is lit as if from a point above the middle of the goal/pitch**:
    `flip_x`/`flip_y` mirror the light per part so each is highlighted on the
    side facing the centre and shaded on its OUTSIDE — top post shaded on top,
    bottom post on the bottom, crossbar on the side away from the pitch, feet
    likewise (`_draw_goal_top` passes `flip=top_post` / `flip=left`). Both posts
    are exact mirror images (tested), at every zoom (3 tones at 1x, a smooth
    gradient at 4-5x). Bars are opaque per-row/column
    fills, so the frame is still exactly pixel-aligned (symmetry/thickness
    tests unchanged), but its pixels are now a neutral **grey ramp**, not line
    white — tests detect the frame as `r == g == b >= 120` (nothing else near
    it is neutral: net/tints are blended over green). The **feet** are shaded
    domes (`_shaded_foot`: a sphere under the same light, cached sprite,
    coverage/shade computed per pixel from an 8x8 sub-grid so the rim is
    anti-aliased without smoothscale's edge darkening). It is drawn **under** the
    post (only the half beyond the post's end and any bulge wider than the post
    shows) and its radius is `_goal_foot_radius()`: the post's half-width plus a
    proud margin that is 0.25px on a 3px post (1x — it used to be `half + 1`, a
    5px blob on a 3px post) growing 0.25px per extra post pixel to a full pixel
    (unchanged from 4x-5x up). **The two crossbar/post corners are
    mitred** so the shadow wraps round the corner from post to bar: each pixel of
    the corner square takes the shade at its distance from the NEARER outer edge
    (`outer_first[min(a, b)]`; the seam is the diagonal from the outer to the
    inner corner). That is only continuous because post and bar share one
    profile — the reason the light is exactly 45° — asserted in
    `test_post_and_crossbar_shade_identically_so_their_corners_can_mitre`.
  - **The roof net sags** (`goal_net.sag_m`, 0.3m; 0 = flat). A roof hanging
    `sag_m` below crossbar height leans less than the crossbar does, so its mesh
    is displaced toward the pitch by `sag_m / goal_height × lean × ppm` px at its
    deepest point (~0.9px at 1x, ~4px at 5x), scaled by `(1-u²)(1-v²)` (u across
    the goal, v along the roof) so the crossbar, the posts' side edges and the
    top-back edge stay put and the diamonds bow in between
    (`_draw_goal_net(..., sag_dx_px)`, drawn as sampled polylines). Because the
    displacement is along x, only a **top-bottom** mirror keeps the two diagonal
    families identical, so the sagged roof layer (cache key `"roof-sag"`) is
    mirrored vertically, whereas the flat one (`"roof"`, used below ~0.05px) is
    mirrored left-right. A shading variant (mesh fading toward the middle + a soft
    dark blotch) was tried and dropped as unnecessary.
  - **The mouth is open: the net is only the roof.** The net hatch is drawn
    only behind the crossbar (roof: crossbar -> top-back edge, drawn over the
    back wall's own mesh, below), never in the mouth.
  - **The back of the net has the same parallax** (`_GoalPx.back_top_x`): the
    engine's goal is a box (vertical back wall, roof at crossbar height), so
    the wall's *top* edge is displaced outward by the same lean as the crossbar
    while its *foot* stays put. Drawn as: a 1px line where the net meets the
    ground (`back_x`), a separate 1px top-back edge (`back_top_x`), and an
    optional tint over the strip between them.
  - **The mouth and the back-wall strip share ONE tint, `goal_frame.footprint_tint_alpha`**
    (default 0 — no tint, same plain pitch green as everywhere else). They used to be two
    separate values (`mouth_alpha` 40, `back_wall_alpha` 30), which — measured with the net
    mesh removed so the flat tints alone were compared — came out as genuinely different
    colours ((73,156,89) vs (66,152,83) against a (43,141,62) pitch): a real, if subtle,
    inconsistency, not something imagined. Consolidated into one value on request ("I don't
    think we even need different values at all"), defaulted to off. The back-wall strip still
    reads visually denser than the roof net even with the tint off, purely because its own
    steep mesh and the roof mesh are both drawn there (see below) — nothing to do with tint.
    Tests: `test_the_mouth_and_back_wall_tints_are_always_the_same`,
    `test_the_goal_footprint_has_no_tint_by_default`.
  - **The back wall has its own mesh, at the right angle**
    (`_draw_goal_back_net`, geometry in the pure `back_wall_net_segments`).
    The wall is vertical, and the parallax maps a wall point's *height* to
    screen x (the short strip, `lean` wide for the full crossbar height) and its
    position *along the goal* to screen y. A 45° diamond mesh on that wall
    (horizontal run == height) therefore projects to two mirrored families of
    **steep** lines that rise `goal_height × ppm` px in y (22px at the default
    window) over a strip only ~6-26px wide in x — squashed, tall diamonds,
    visibly different from the roof's 45° lattice. Same supersampled,
    anti-aliased, halo-free drawing as the roof mesh. **The roof net is drawn
    over the wall mesh**: the roof is at crossbar height (nearer the camera) and
    physically extends to the top-back edge, so its see-through mesh overlaps the
    wall strip and both layers show there (wall drawn first, then the roof from
    the crossbar to the top-back edge — a lace-like double lattice, busier than
    either alone). An earlier version stopped the roof at the ground line so the
    strip showed only the wall mesh; that was dropped as inaccurate. Two cosmetic
    knobs: the wall mesh spacing is 2× the roof's (`goal_net.back_spacing_scale`,
    formerly a constant;
    at 1× the steep lines in a ~6px strip alias into a moiré, and 2× reads as
    clean tall diamonds at both zoom 1 and 4 — compared 1×/1.5×/2×/3× side by
    side) and its line weight is 0.75× (`_GOAL_BACK_NET_LINE_WEIGHT`;
    pygame thickens steeper lines more). Tested by
    `test_back_wall_net_segments_are_steep_mirrored_diagonals` (geometry) and
    `test_back_wall_net_is_steep_and_the_roof_net_is_drawn_over_it`, which
    renders the wall mesh and roof mesh separately (via `no_roof`/`no_wall` on the
    test helper) and measures orientation as the ratio of brightness variation
    along x vs y: roof ≈ 0.98 (45°), wall alone ≈ 2.6, and the combined strip
    ≈ 1.5 (a mix), with more ink than the wall alone; a 45° wall mesh would give
    1.0 and fail. All the
    goal's horizontal edges (ground and top) fall on the same two screen rows
    (the parallax is purely along x), so the side lines run continuously from
    the goal line to the top-back edge. The defending-side marker sits
    beyond that top-back edge (`marker_back_m = depth + lean`) so it doesn't
    overlap the net. Tests: `test_back_of_net_*`,
    `test_back_wall_strip_is_tinted_*`, `test_defending_marker_sits_beyond_*`.
  - **Two layers.** *Footprint* (drawn in `draw_pitch`, under everything):
    side/back lines + mouth tint. *Top* (`draw_goal_tops`): roof net, posts,
    crossbar. `draw_pitch(goal_tops=False)` skips the top layer so a caller
    can draw it later.
  - **Ball depth** (`draw_pitch_and_ball`, which `App._draw_match` calls
    instead of `draw_pitch` + `draw_ball`): if the ball is *inside* a goal —
    `ball_under_goal_frame`: past the goal line, no further than the back
    wall, between the posts (±ball radius), and `z < goal_height_m`, mirroring
    the engine's own inside-the-goal test in
    `ball_physics.resolve_goal_boundary` — the top layer is drawn OVER the
    ball (the bar cuts across it, the roof net shows faintly over it). Any
    other ball, including one above the bar going over, is drawn over the
    frame. The switch happens at the goal line, where the bar (displaced 0.72m
    behind it) doesn't yet overlap the ball, so there's no visible pop.
    Players are always drawn over the frame (not modelled — a keeper 1.8m
    tall is always under the bar anyway, and only rarely inside the net).
  - **Post alignment**: every goal part is placed from one set of integer
    coordinates (`Renderer._goal_px` -> `_GoalPx`), with the two post rows
    *symmetric about the pitch's centre row* (`c ± round(half_goal_width *
    ppm)`) and every stroke drawn as a filled rect centred on its row. This
    replaced independently-truncated world->screen coordinates plus
    `pygame.draw.line` (which extends a width-2 line *downward*) next to
    `pygame.draw.rect(width=...)` (which draws *inward*): together those put
    the top post 1px inside the mouth tint and the bottom post 1px outside it.
    Any post width works (even leaves the pair half a pixel off the centre row,
    invisible).
  - **Net border = mesh thickness**: the net's side and back lines (the
    "border"/outline) are drawn at `_GOAL_NET_LINE_PX` (1px), the same constant
    the mesh lines use, so the two can't drift. They used to be drawn at the
    zoom-scaled *pitch line width* (`0.12m × ppm` = 3px at 3x, 4px at 4x) while
    the mesh stayed 1px — measured against the committed renderer: 1/3/4px
    border vs a 1px mesh at zoom 1x/3x/4x. Posts and crossbar are the heavy
    parts, then the goal line/pitch lines, then the net (border + mesh).
  - Tests (`tests/unit/test_pitch_markings.py`): symmetric posts on the mouth
    edges, no net hatch in the mouth, `ball_under_goal_frame` cases, and a
    pixel comparison that a ball at z=1.0 leaves the crossbar pixels exactly
    as in a ball-less render while z=3.2 changes them.
- **The net is always a diamond lattice, never a checkerboard**
  (`_goal_net_gap_px`): the mesh gap is the physical `goal_net.spacing_m`
  (0.35m) at the current zoom but never below `goal_net.min_spacing_px` (**7**).
  Without the floor the gap was `int(0.35m × ppm)` = 3px at the default
  window, and two families of 1px diagonals 3-4px apart alias into a dense
  checker (measured: 58% of net pixels lit at 3px, 50% at 4px), while at 3-4x
  zoom the same formula gave a proper lattice — so the net looked different at
  every zoom level. Now: 7px at zoom 1x/2x, 9px at 3x, 12px at 4x (the floor
  only bites below ~2.5x, so high-zoom looks are unchanged). The floor went 8 →
  6 → 7 across sweeps at 1x: 5px is still blobby; 6px looked fine until both
  diagonals were drawn identically (see the mirroring pitfall), after which it
  read a little checker-like; 8px is cleaner but leaves only ~2 diamonds across
  the 15px-wide roof. At the default window the mesh is coarser than 0.35m
  (~0.7m), purely cosmetic. The back-wall mesh is 2× this (14px at 1x). Lines
  stay 1px. `test_goal_net_is_a_sparse_diamond_lattice_at_every_zoom` asserts the
  gap is ≥7 and the net's *ink* (mean lightening over grass) is <0.30 at
  zoom 1-5 (measured ~0.17-0.20 at 7px vs ~0.45 for a 3px checker), and a
  companion test shows the same metric flags the old behaviour with the floor
  removed. (When comparing screenshots across zooms, don't upscale some and
  not others.)
- **Goal netting** (`_draw_goal_net`, called from `_draw_goal_top` for the
  roof region only — crossbar to back of net; see above): fills a
  screen-space rectangle (the roof, `goal_depth_m` deep minus the crossbar
  lean, × `goal_width_m`) with a translucent, **anti-aliased** diagonal
  X-hatch (drawn supersampled and `smoothscale`d down — see "Anti-aliasing
  policy"; opacity applied once to the whole layer with `set_alpha`, so line
  crossings aren't double-blended brighter) on an isolated SRCALPHA surface
  sized to the box so the diagonal lines clip cleanly at its edges.
  Colour/density/opacity are
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
- **Corner flags** (`_draw_corner_flags`, geometry in the pure
  `corner_flag_world_points`): drawn with the same top-down parallax as the
  goal frame. The pole is vertical, so its top is displaced **away from the
  pitch centre**, along the centre->corner line, by `pole_height_m ×
  (crossbar_lean_m / goal_height_m)` — the crossbar's parallax rate, so one
  setting drives both (with the defaults 4.0 × 0.72/2.44 ≈ 1.18m, ~12px at the
  default window, ~59px at 5x zoom; it was 2.4m/≈0.71m/~7px until the flags were
  judged a little too small — see "Flag size" below). The pole is **yellow**
  (`style.CORNER_FLAG_POLE_COLOUR`); the pennant stays orange
  (`CORNER_FLAG_COLOUR`). The pole is at least 1px wide (a 2px minimum was
  tried and looked chunky/rectangular at the default zoom), and the foot dot
  marking the ground contact appears **only once the pole is ≥3px wide**
  (radius `pole_px//2 + 1`): a fixed 2px-radius dot is a 5px blob on the ~6px
  pole at 1x zoom and hid it entirely (compared five foot/pole variants at
  zoom 1/2/4 before choosing). The pennant is attached along the top `flag_length_m` (0.32m apparent) of the pole with
  the rest bare. **The cloth's free end is aimed perpendicular to
  the pole** (`flag_width_m` off it, on the side toward the pitch's middle
  along the touchline). Aiming it along the touchline instead ("inward" — the
  old design's direction) makes the pennant a needle: the pole leans along the
  corner diagonal and the cloth's base edge lies along the pole, so the tip
  ends up only ~30° off that edge and the triangle has almost no width
  (seen when first tried; reproduced across four parameter sets before
  changing the geometry). `pole_height_m` is *apparent* (real poles are
  ~1.5-1.8m) — bolder on purpose so the lean reads at the default zoom.
  **Flag size**: eight variants were compared side by side (taller pole, bigger pennant, both,
  and those plus an outline / white stripe / ground shadow). Chosen: the taller pole
  (4.0m apparent) **without** a bigger pennant, plus the shadow. The cloth's length along the
  pole is now an ABSOLUTE size, `flag_length_m` (the renderer derives the drop fraction
  `flag_length_m / lean` — `_corner_flag_drop`), because the earlier fractional
  `flag_drop_frac` (55% of the pole) made the pennant grow with the pole; the new default
  0.32m is a little shorter than the old 0.39m. An outline muddied the ~6px pennant at 1x
  and a white stripe turned into a noisy blob, so neither was kept.
  **Flag shadow** (`_draw_corner_flag_shadow`, `shadow_alpha` 75, `shadow_height_m` 1.2):
  cast by the scene light (see "Scene light and player lighting"): a point at height z above
  ground point P shadows at `P × H / (H − z)`, so the pole throws a line from its foot
  straight away from the pitch centre, `d × h / (H − h)` long, and the cloth a triangle. It
  lies along the same line as the pole itself, so it reads as the pole's extension.
  **`shadow_height_m` is deliberately NOT the physical pole height**: a corner is ~62m from the
  point light, i.e. the light is only ~33° above the horizon there, so a 2.4m pole threw a ~4m
  shadow (verified pixel-for-pixel against the formula: starts at the foot, ends at 4.0m) —
  3.4x the drawn pole, which is only a ~1.2m stub (the drawn pole's lean is the fixed crossbar
  parallax, not tied to the light, so the two models disagree in proportion). Set to 1.2m ->
  ~1.9m (1.6x the drawn pole). Drawn first (under the pole/cloth) 2x supersampled on a small black transparent
  layer and averaged down (soft edges, no halo), blitted at `shadow_alpha`.
  Config: `graphics.json["corner_flag"]`. Tests:
  `test_corner_flag_pole_leans_away_from_the_centre_and_the_cloth_is_a_real_triangle`,
  `test_corner_flag_pole_and_cloth_are_drawn`,
  `test_the_pennant_keeps_its_size_however_tall_the_pole_is`,
  `test_a_corner_flag_casts_a_shadow_straight_away_from_the_pitch_centre`,
  `test_flag_shadow_can_be_turned_off_and_shrinks_as_the_light_rises`.
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

Every circular outline (selection, control-delay/inactive,
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

## Anti-aliasing policy

Anything curved or diagonal is anti-aliased; axis-aligned rects/lines (pitch
lines, posts, crossbar, tint rects, the help-overlay rules) are pixel-aligned
and have nothing to smooth. Which technique depends on the destination:

- **Opaque destination** (the pitch surface): `gfxdraw.aacircle` /
  `filled_circle` for discs (spots, corner-flag pole, ball body; the post feet are
  shaded per-pixel sprites, see the 3D goal frame),
  `pygame.draw.aalines` for 1px curves (arcs at the default zoom, speed lines,
  heading V), and `gfxdraw.filled_polygon` + `aapolygon` for filled shapes
  (flag pennant, kick UI). **Thick arcs** (`_draw_world_polyline`, width > 1px
  once zoomed) are a polygon strip — the polyline offset ±half the width along
  its normals — drawn the same way; `pygame.draw.lines` at that width is
  aliased, and `pygame.draw.arc` tears.
- **Transparent / translucent layer** (net mesh, rings, ball dots, sprites,
  ball-trail ghosts, inactive-player fallback disc): draw at
  `_RING_SUPERSAMPLE`× (4×; the cached net meshes use `_NET_SUPERSAMPLE`, 8×)
  with plain `pygame.draw.*` and `smoothscale` down.
  `draw.aaline` and `gfxdraw.*` do **not** blend correctly onto a transparent
  layer (measured on the ball dots: near-white notches inside dark dots, alpha
  double-applied). Helper for small discs: `Renderer._aa_disc(rgb, radius,
  alpha)` (cached per colour/radius; the surface is shared, blit immediately).
  Apply a layer's translucency once, with `set_alpha` on the downscaled layer,
  rather than per-line alpha.

**Pitfall — fill the transparent background with the drawn colour at alpha 0
(`big.fill((*rgb, 0))`), not the default transparent black.** `smoothscale`
averages RGB and alpha *independently*, so a black background pulls every
edge pixel's RGB toward black; composited, those edge pixels come out darker
than both the shape and whatever it sits on — a dark halo. Measured: 161 of
610 net pixels (26%) were darker than the grass under them at 1x zoom; the
pre-existing `_draw_ring` had 72 ring-edge pixels dimmed below both ring and
grass. Fixed in `_draw_goal_net`, `_aa_disc` and `_draw_ring` (which also
brightens the selection/possession/control rings' edges). `_ball_dots_layer`
doesn't need it (its dots are already near-black).

**Pitfall — `pygame.draw.line` is not symmetric between the two diagonals.**
It rasterises through pixel centres, i.e. offset (+0.5, +0.5) from the
coordinates you give it — *along* the line for a down-right diagonal but
*perpendicular* to it for an up-right one. Drawn directly, mirrored lines come
out as different pixel profiles: measured on a single mirrored pair, `\` was a
narrow core over 3 pixels (alpha 15/191/47) while `/` was split over 2 pixels
half a pixel off (159/95) — same total ink (ratio 0.998-1.002), so an ink check
can't see it, but the `/` lines look wider and dimmer, and the net looked
uneven with a "shadow" on one side. **The net mesh therefore rasterises ONE
diagonal family and produces the other as its exact mirror**
(`Renderer._net_layer`: flip + `BLEND_RGBA_MAX`, so crossings aren't
double-brightened): the roof layer is left-right symmetric and the back-wall
layer top-bottom symmetric to the pixel (asserted in
`test_net_mesh_diagonal_families_are_exact_mirror_images`; the old two-family
layers were 94k-242k alpha-units asymmetric). Because the layers depend only on
their size/spacing/colour they are **built once and cached**
(`_net_layer_cache`, bounded at 64 entries — zoom changes make new sizes), which
is why the nets can use a finer `_NET_SUPERSAMPLE` (8×) than the per-frame
rings/dots (4×) at no per-frame cost. Any other mesh/hatch of mirrored diagonals
should do the same.

Not anti-aliased, deliberately or not yet: the ball's sub-pixel (<1px)
outline path (a 1px `draw.circle` at reduced alpha, only if
`ball.outline_width_px < 1`), and the legacy unused `draw_drag_indicator`.
Tests: `tests/unit/test_pitch_markings.py` (`test_goal_net_mesh_is_anti_aliased
_without_a_dark_halo`, `test_thick_pitch_arcs_are_anti_aliased`,
`test_aa_disc_and_rings_have_no_dark_fringe`); the density test measures *ink*
(mean lightening over grass), not a lit-pixel count, because anti-aliasing
spreads a line over more pixels at lower intensity but preserves the total.

## Ball shadow and shading (`draw_ball`)

- **Sphere shading** (`ball_shading` in `graphics.json`, `_ball_shade_sprite`): a
  black overlay with per-pixel alpha, drawn last over the ball and its dots, so the
  ball is dark on the side facing away from the light and clear where it faces it. The
  light is **the scene light** (`scene_light.height_m`, 40m above the pitch's middle;
  see "Scene light and player lighting"), via `_ball_light_angles`: horizontally from the ball
  toward the centre, elevation `atan((H - z) / distance)` — so the lit side always faces
  the middle of the pitch, the light is overhead at the centre (bright middle, even dark
  rim) and lower/more sideways toward the edges (about 37° at the far goal-line ends);
  a raised ball is nearer the light's height so it sees it lower still, not higher. (It was first a fixed top-left light,
  "variant C"; it now shares the shadow's light.) The sprite cache is keyed by
  (radius, azimuth, elevation) with the angles quantised to 10°/5° (bounded at 512); a
  Blinn-Phong highlight cancels the shading where it peaks (a white ball cannot get
  whiter, so "highlight" = no darkening; an earlier attempt that added white
  produced a grey spot). `strength` 0.85 ("variant C" of the options compared
  side by side). The sprite is computed with numpy at 6x sub-sampling, cached per
  radius; the dots still read clearly through it (tested).
- **Ground shadow** (`ball_shadow`, `_draw_ball_shadow`): cast by the scene light — a **point
  40m above the middle of the pitch (`scene_light.height_m`)**. A sphere's shadow is an
  **ellipse**: semi-minor axis its radius `r` (the DRAWN, enlarged ball's ground radius, so it
  keeps that ball's proportions), semi-major `r / sin(e)` for light elevation
  `e = atan((H − zc) / d)` (`zc` the drawn sphere's centre height, `d` its distance from the
  middle), elongated radially away from the centre and centred where the light through the ball's
  centre lands, `d × zc / (H − zc)` beyond the ball (`shadow of a point at height z sits at
  P × H / (H − z)`). So a grounded ball's shadow **depends on the light**: a circle hidden right
  under it at the centre (overhead light) that stretches and slides out with distance from it
  (~1.5 radii semi-major at the far edge, where the light is ~40° up at the ball's height); a
  raised ball's whole shadow lies further out. (It was a capsule — a constant-width bar with round
  ends, "cylinders", which looked wrong — first with a constant 0.10m contact offset that was ~1px,
  hidden at 1x and identical at every spot, measured before changing it.)
  **Minimum contact shadow**: where the physical ellipse would be too little to see, it is slid
  outward until its far tip clears the ball's edge by `contact_radius_frac` (0.33) of the DRAWN
  radius, at least `contact_min_px` (2px) — it fades out as the ball rises (gone by 1.5m), and points
  radially away from the centre, with a fixed lower-right bias within ~3m of it (where the overhead
  light has no true direction; keeps the direction continuous as the ball crosses the middle). It
  is a fraction of the drawn radius, NOT metres or a fixed pixel count, so it scales with zoom
  exactly like the ball: a fixed 5px minimum was a third of the ball's radius at zoom 3 but a whole
  radius at zoom 1 ("too big at zoom 1"). Fainter (alpha 100 → ~55 at 6m) and softer the higher the
  ball is; drawn in the shared shadow pass, under everything. Sprites are numpy soft ellipses
  (`_shadow_ellipse_sprite`: alpha fades over the elliptical distance from `1 − softness` to
  `1 + softness`; shared with the players' shadows), quantised and cached, bounded at 512.
  Tests: `test_a_grounded_balls_shadow_depends_on_the_light_it_grows_with_distance_from_the_centre`,
  `test_the_grounded_ball_shadow_is_the_ellipse_the_light_casts` (far tip within 1.5px of a formula
  that now accounts for the fade below, at zoom 1, 3, 5), `test_the_ball_shadow_is_an_ellipse_not_a_constant_width_bar`
  (half-width at 0.8 of the semi-major axis is 0.6 of the minor axis; a capsule would still be full width),
  `test_the_minimum_contact_shadow_scales_with_the_ball_across_zoom_levels`,
  `test_the_minimum_contact_shadow_still_points_radially_away_from_the_centre`.
  - **The ellipse itself fades near→far, like the players' capsules** (`tip_fade`, 0.3): the uniform
    ellipse above looked "too perfect" next to a player's shadow (which is sharp/dark at the feet and
    soft/light at the tip). `_shadow_ellipse_sprite` gained an along-its-own-major-axis fade: `t=0` at
    the end nearest the ball (`along = -a`, i.e. toward the side actually closest to the object —
    NOT necessarily toward the pitch centre, since the ellipse sits beyond the ball), `t=1` at the far
    tip (`along = +a`). Local softness is interpolated between `softness × _SHADOW_FADE_SOFT_NEAR_FRAC`
    (0.2) and `softness × _SHADOW_FADE_SOFT_FAR_FRAC` (0.8) — the same 0.2/0.8 split the players use —
    and the alpha is scaled by `1 - tip_fade × t`. `tip_fade=0` (the default for the parameter, used
    by `_ball_shadow_sprite`'s plain disc and any other non-directional caller) reproduces the old
    symmetric ellipse exactly; the ball's own draw call always passes `tip_fade=0.3`. A closed-form
    check: at the geometric edge (`rho=1`) the raw falloff is always exactly 0.5 (independent of
    softness), so near/far alpha there is exactly `1 : 1 − tip_fade`.
  - **The fade deepens and softens further the higher the ball is** (an airborne shadow has no sharp
    contact edge anywhere, so it should look uniformly hazier, not just fainter — it already faded in
    alpha and grew softer with height, but had no directional component of its own before this).
    `height_frac` (0 grounded, ramping linearly to 1 over `height_fade_ramp_m`) adds up to
    `height_extra_soft` to BOTH near/far softness and up to `height_extra_fade` to the fade. Sized by
    eye against a grid of options (labelled "C"/"D" during review, each a `(ramp_m, extra_soft,
    extra_fade)` triple; an even stronger first attempt, "too strong", was rejected) and landed 20% of
    the way from "C" (3.0m / +0.7 / +0.18) toward "D" (4.5m / +0.5 / +0.12): **3.3m / +0.66 / +0.168**.
  Tests: `test_ball_shadow_fade_defaults_come_from_the_config`,
  `test_the_ball_shadow_is_darker_and_sharper_near_the_ball_than_on_its_far_tip`,
  `test_zero_tip_fade_is_the_old_symmetric_uniform_ellipse`,
  `test_the_ball_shadow_fade_deepens_and_softens_further_the_higher_the_ball_is`,
  `test_a_grounded_ball_at_the_centre_has_no_height_fade_and_a_high_one_has_the_full_amount`.
- The state rings (see "Ball state indicator rings") were tightened to
  `offset_px` 1 / `width_px` 1 (from 2 / 2) so they hug the ball and read as a thin
  status outline rather than a halo.

## Scene light and player lighting

*Shadow strengths (`alpha`s: ball 100, players 90, flags 75): the first set was 120 / 107 / 90, raised ~25% to
150 / 135 / 115 (too dark), then set lighter than the original after a side-by-side of four options.*

One light drives every shadow and shading effect: a **point `scene_light.height_m`
(40m) above the middle of the pitch** (`Renderer._light_height_m`,
`_point_light_at(x, y, z)` -> azimuth toward the centre, elevation `atan((H - z) / d)`,
distance `d`). Higher = shorter shadows, more from-above shading; lower = long sideways
shadows and hard side-lighting toward the edges. It replaced the earlier
`ball_shadow.light_height_m`. The goal frame keeps its own fixed centre-facing
approximation (`style.LIGHT_*`), which is not tied to this height.

- **Ground shadows are one pass under everything.** `draw_pitch_and_ball(surface,
  pitch, ball, players=...)` draws the pitch's ground layer, then EVERY player's shadow
  (`draw_player_shadows`) and the ball's shadow, then the goal frame / ball, and the app
  draws the sprites after that: a shadow never falls on another player, the ball or the
  net (drawing them per player, inside `draw_player`, put later players' shadows over
  earlier sprites). `draw_ball(..., shadow=False)` is what `draw_pitch_and_ball` uses; the
  default still draws the ball's shadow itself.
- **Player shadow** (`player_shadow`: `alpha` 90, `tip_fade` 0.3, `contact_m` 0.18): a **capsule**
  — a constant-width bar with a round end ("cylinder") — from the feet straight away from the pitch
  centre, **sharpest and darkest at the feet and softer-edged and lighter toward the tip**
  (`_player_shadow_sprite`: the penumbra widens from 0.2 to 0.8 of the half-width along the
  shadow and the strength falls to `alpha × (1 − tip_fade)` — a real shadow's contact hardening;
  the 50%-alpha edge stays a constant-width bar). Its length is `d × h / (H − h)` (h =
  `player.height_m`, 1.8m; a 45m-out player gets ~2m, a player at the centre only the contact
  length), half-width 0.85 × the drawn radius. Direction has a lower-right bias that dominates within
  ~1.5m of the centre so it is continuous when a player crosses it. Options compared side by side:
  a plain capsule, this fading/softening one (**chosen**, with the fade cut from a first 0.55 to 0.3
  as "a bit strong"), a body-shaped width profile (legs / wide torso / head — rejected: it would have
  to rotate with the player's heading), and the player's own silhouette swept along the shadow
  (rejected: blocky and per-heading, so hard to cache). **An ellipse was tried for players too and
  rejected — the bar read better on a figure**; the ball, a sphere, keeps its exact ellipse. Sprites
  are numpy capsules cached per (length, 5° angle, radius, alpha, fade) in `_shadow_cache` (shared
  with the ball's ellipses, bounded at 512). Tests:
  `test_a_player_shadow_fades_and_softens_toward_the_tip_but_keeps_its_width`,
  `test_the_player_shadow_tip_fade_is_configurable_and_zero_means_uniform`,
  `test_a_player_shadow_still_reaches_the_length_the_light_gives`.
- **Player shading** (`player_shading.strength` 0.9; "option D" of the sheet compared with
  a whole-sprite dome and a shadow-only option): the flat art is lit **per body part**.
  `player_sprites.sprite_normals` treats the sprite's alpha, blurred at two scales, as a
  height map, so shoulders, torso, head, legs and arms are each a rounded lump with the
  tilt at their edges and flat tops; `shade_factors` is Lambert + ambient (0.45), scaled so
  the best-lit part is exactly 1 (only far sides darken). It is applied to the BASE
  80px pose **before** `rotozoom`, so the light is rotated into the sprite's frame:
  `local = (c·lx − s·ly, s·lx + c·ly)` for rotation φ = `rotate_deg` (the inverse of
  rotozoom's counter-clockwise-on-screen rotation), tested at 5 headings × 3 spots.
  `PlayerSpriteSet.shaded(side, level, az, el, strength)` caches lit variants per (pose,
  direction quantised to 15° / 5°, strength), bounded at 1024. The effect is strongest
  under a low light (near the pitch edges) and gentle when it is overhead (near the middle)
  — a higher `scene_light.height_m` softens it; under a 40m light the dark side of a
  figure near the edge is ~0.85 of the lit side. Cost: ~3ms/frame more with all 22 players
  turning (cache misses as headings change), ~0.5ms for shadows alone.
- **Tests** (`tests/unit/test_player_sprites.py`): normals/factors maths; lit variants
  keep alpha, never brighten, are cached; the near half of the figure (facing the centre)
  is the bright one at every heading and spot (low light so the contrast is measurable);
  shadow direction and `d·h/(H−h)` length; grows with distance and shrinks as the light
  rises; contact shadow at the centre; **shadows under everything** (a player drawn earlier
  and the ball are unchanged by a later player's shadow); the light height is one shared
  setting. An autouse fixture forces player shading/shadows off for the flat-art tests.

## Turf texture (`_draw_turf`, `graphics.json["turf"]`)

`draw_pitch` no longer fills one flat green. The background is the pitch green times
(a) **soft, world-anchored patches** — two octaves of smoothstepped value noise
(`patch_cell_m` 6.0m / 2.2m, `patch_amp` 0.02, slightly warmer where lighter), a
colour texture at 4px/m covering the pitch ± 15m, scaled/cropped to the camera each
time it moves so the patches stay glued to the pitch under pan and zoom — and (b) a
**screen-space multiply layer** of fine per-pixel grain (`noise_amp` 0.02) and a soft
vignette (`vignette` 0.14, darker toward the corners; applied to the grass only, so
lines, players and the HUD are untouched). The overlay has mean `_TURF_GAIN` (0.95)
and the base colour is pre-divided by it, so the average is exactly `PITCH_GREEN`.
The composed background is cached and rebuilt only when the camera's mapping changes
(a static camera = one blit, ~0.8ms/frame; following the ball at zoom 3-5 adds
~2ms/frame for the smoothscale). Patch strength was set from a side-by-side of
9 variants (fine noise alone looks grainy; patches at 6% were "a bit too strong" →
4% with slightly larger cells → 3% → 2% with cells 4.5/1.6m → 6.0/2.2m). `PITCH_GREEN` was lightened from (34, 120, 50) to
(43, 141, 62) after the textured pitch (with its vignette) read as a bit dark; several
tests use `style.PITCH_GREEN` rather than a literal so it can be tuned freely. Mowing stripes were rejected. **Tests** run with
`turf.enabled`, `ball_shadow.enabled` and `ball_shading.enabled` forced off by an
autouse fixture (they measure lines/nets against exactly flat grass and a plain ball);
the tests for those features switch them on explicitly.

## Ball spin dots (`draw_ball`'s dot-projection block)

The ball's surface dots (fixed points on a unit sphere — see "dot placement"
below for which lattice — rotated each frame by `_ball_orientation` and
projected top-down) are drawn
as **projected tangent-plane patches, not flat discs of constant size**.
Each dot's outline is built from `_DOT_POLY_SEGMENTS` points arranged in a
small circle in the dot's own tangent plane (`u`/`v`, both perpendicular to
its centre direction `(wx,wy,wz)`), pushed through the *same* rotation +
orthographic projection as the centre point, then rasterised by
`Renderer._ball_dots_layer` (see below). This is what makes dots near
the ball's silhouette edge correctly foreshorten into ellipses (a real
football's panels do the same under orthographic projection), while dots
near the visible "pole" stay circular — the shape falls out of the
projection automatically, with no separate squish-factor/rotation-angle
math needed. Before this, dots were plain `pygame.draw.circle` calls of a
single fixed radius with no antialiasing at all (visibly jagged, and flat
rather than sphere-like) — if you need to touch this code again, keep both
properties (foreshortening + AA) rather than reverting to a flat circle.

**Rasterisation: supersample, don't `gfxdraw`.** `_ball_dots_layer` draws the
dots with plain `pygame.draw.polygon` at `_RING_SUPERSAMPLE`x (4x) resolution
onto a transparent layer, clips them to the ball disc (also at 4x), and
`smoothscale`s down — the same trick as `_draw_ring`. The earlier
`gfxdraw.filled_polygon` + `gfxdraw.aapolygon` pair (both alpha-blending onto
the transparent layer) left visible defects on these ~8px dots: the fill stops
short of the polygon's four extreme vertices while the AA pass only covers
10-30% there, so near-white pixels (measured 210-248 against a ~85 interior)
sat inside the dark dot as cross-shaped notches, and the double-blended alpha
made the interior ~85 instead of the configured (30,30,30)@220 ≈ 62 over a
white ball. Reproduced on a synthetic polygon and confirmed on the real ball
at 4x zoom before changing it. The layer builder takes `orientation` /
`dot_positions` so a *single* dot can be tested:
`tests/unit/test_pitch_markings.py::test_ball_spin_dot_is_solid_without_light_holes`
asserts each row and column of one dot's dark pixels is a single contiguous run
(a convex dot has no light pixel between two dark ones) across radii and
foreshortening; that same check flags the old drawing at every radius tried.
Note the net drawn OVER a ball inside the goal (see the 3D goal frame) still
lightens dark dots where its hatch lines cross them — that is the net in front
of the ball, not this bug.

**Dot placement: icosahedron for the classic 12, Fibonacci lattice otherwise.**
`Renderer._make_ball_dot_positions(n)` dispatches to
`_make_icosahedron_vertices()` when `n == 12` (the config default,
`ball_spin_dots.count`), and to `_make_fibonacci_sphere(n)` for any other
count. The old code always used the Fibonacci lattice, at `n=15`: measured,
its closest pair of dots was only ~31° apart even though its *average*
coverage was even — visibly "two dots too close together", which is what the
user flagged. A real football is a truncated icosahedron (12 pentagons + 20
hexagons); each pentagon's centre sits exactly on the untruncated
icosahedron's 12 vertices, so using those 12 vertices both matches the real
panel layout and is the Tammes-optimal (provably most-even) placement for 12
points on a sphere — every point's nearest neighbours (5 of them, one per
icosahedron edge) sit at a uniform 63.4°, confirmed for all 30 nearest-
neighbour edges, not just the closest pair. The Fibonacci lattice is kept as
the fallback for any other configured count, since there's no equivalent
regular solid for arbitrary n.

**Dot size no longer swings with zoom.** `_ball_dots_layer` used to compute
`dot_r = max(1, int(radius_px * frac))` — but `radius_px` (the ball's own
drawn radius) is *already* an int, so truncating `dot_r` a second time on top
of that compounded: measured, the actual dot/ball ratio swung between ~0.167
and ~0.25 across zoom 1–5 for a configured `dot_radius_fraction` of 0.25, i.e.
visibly changing relative dot size purely from zoom, not a deliberate design
choice. The formula now lives in its own method, `_spin_dot_radius_px`,
returns a float, and is only rounded once — at the final per-vertex pixel
snap, same as every other coordinate in `_ball_dots_layer` already was. Tests:
`tests/unit/test_pitch_markings.py`'s
`test_the_default_12_dots_are_the_icosahedrons_vertices_evenly_spaced`,
`test_the_old_fibonacci_lattice_is_kept_for_non_classic_dot_counts`,
`test_a_real_footballs_15_point_fibonacci_lattice_really_did_have_a_close_pair`,
`test_the_dot_to_ball_size_ratio_no_longer_swings_with_zoom`,
`test_the_spin_dot_radius_formula_itself_is_not_truncated_to_an_int`,
`test_the_code_level_default_dot_count_is_12_even_if_config_omits_it`,
`test_dot_layer_pad_still_fits_the_dot_at_every_radius`.

## Player visual indicators (`style.py` / `renderer.draw_player`)

- **Body**: by default (`graphics.json["player_sprites"]["enabled"]`, default
  true) each player is a rotated `player_sprites` sprite (see that module's
  docstring), lit per body part with a ground shadow (see "Scene light and player
  lighting"), animated per-frame via `Renderer.update_player_animations`
  (only advanced while the match isn't paused). With sprites disabled it
  falls back to the plain filled circle described in the bullets below —
  the **heading V** is drawn *only* in that fallback mode (the sprite's own
  head/limb asymmetry already shows facing), and the translucent-inactive
  look is likewise done per-path.
- **Legs drawn under the ball, upper body over it** (`draw_player_legs`, `draw_player(..., legs=False)`):
  every sprite pose exists as TWO layers as well as the combined image --
  `player_sprites._render_pose_layers` (legs: shafts/feet/shoes/shorts; upper: arms/shoulders/
  torso/head) -- built independently of `_render_pose` (deliberately: compositing two separately
  downscaled layers isn't pixel-identical to one combined downscale at their seam, where the shorts
  tuck under the shirt hem, so `_render_pose` -- and everything that used to depend on it -- is left
  completely untouched; the split layers instead share the drawing code via `_pose_geometry` /
  `_draw_pose_legs` / `_draw_pose_upper`). `draw_pitch_and_ball` calls `draw_player_legs` for every
  player right before drawing the ball (in the same "under everything" pass as the shadows/rings);
  `App._draw_match` then draws each player's upper body only (`draw_player(..., legs=False)`) in its
  usual per-player loop, so the ball ends up sandwiched between the two -- a player standing over or
  near the ball shows it resting at their feet (the ball covers an outstretched foot where they
  overlap) with the torso still in front of it, rather than the whole figure sitting flatly on one
  side. `draw_player`'s default (`legs=True`) still draws one combined sprite in one call, unchanged,
  for standalone use (every existing test). No-op (both `draw_player_legs` and the `legs` flag) with
  sprites disabled -- the flat-circle fallback has nothing to split. Costs roughly another
  2-4ms/frame with all 22 players moving (two rotozooms per player instead of one) -- measured, not
  yet a problem at the 16ms/frame budget.
  **Known trade-off**: legs are drawn for every player in one pass before ANY player's upper body, so
  if two different players' sprites overlap (e.g. a tackle) the later-drawn one's upper body can cover
  the earlier one's legs regardless of intended stacking (normally the ball carrier is drawn last, on
  top) -- rare, and only affects the pixels where two players' sprites directly overlap.
  Tests: `test_legs_and_upper_layers_recomposite_close_to_the_combined_pose`,
  `test_the_legs_layer_has_no_shirt_or_hair_colour_and_the_upper_layer_has_no_shorts_or_shoe_colour`,
  `test_legs_plus_upper_bounding_box_matches_the_combined_pose_bounding_box`,
  `test_the_ball_is_drawn_over_an_outstretched_foot_but_under_the_torso` (4 headings, a full-stride
  front foot; compares against the old draw-ball-then-whole-sprite order),
  `test_a_ball_at_the_players_own_centre_still_sits_under_the_upper_body`,
  `test_app_draws_each_player_with_legs_false_since_draw_pitch_and_ball_already_drew_them`.
- **Shoulder shape** (`_pose_geometry`'s `shoulder_size` / `shoulder_corner_r`, and the matching
  inline constants near the top of `_render_pose` — kept in sync by hand since `_render_pose` stays
  independent, see the bullet above): `0.95x` size on top of the existing `0.9x`, corner radius
  `0.4x` of that size (up from `0.35x`) — less square/"too strong"-looking and a touch smaller,
  chosen from a 9-way grid (size in {1.0, 0.9, 0.8} × corner in {0.35, 0.40, 0.45}) shown to the
  user. Shrinking the shoulders very slightly reduced how far they poke out past the sprite's own
  drawn radius, which dropped two ring-placement tests' `covered` sample count from `>10` to exactly
  `10` (`test_rings_are_drawn_under_the_player_sprite`,
  `test_the_pipeline_draws_player_rings_under_every_sprite`) — real, measured, not a masked bug (the
  actual no-overdraw assertion still holds for every covered point); their bound was adjusted to
  `>= 10` with a comment, same pattern as the stamina-ring threshold above.
- **Animation clock** (`sim_clock.SimTimeDelta`, `App._animation_clock`): the
  stride gait (`update_player_animations` -> `advance_gait_phase`) and the ball's
  spin/roll (`update_ball_effects`) are advanced by the **simulation** time since
  the last rendered frame — `match.time_s` deltas — NOT by `1 / target_fps`. The sim
  runs `sim_speed` match-seconds per real second (and steps a whole number of
  physics ticks per frame), so the old fixed step made strides and ball spin play
  at `1 / sim_speed` of their true rate at any sim speed but 1x (and the ball's
  spin drifted from its rolling, which was already right because it is derived
  from position deltas). Paused matches don't animate (and, since the clock doesn't
  move, don't bank time); a new match or scenario trial resyncs to a 0 delta; a
  zero delta leaves the ball's orientation untouched. `advance_gait_phase` caps one
  update at `_MAX_CYCLES_PER_UPDATE` (0.4) stride cycles: the legs alternate every
  half cycle, so >= 0.5 cycles per frame would alias (stall / run backwards), which
  a very high sim speed (up to 8x at 60fps) would otherwise reach — the drawn
  stride then slows slightly instead of misleading.
- **Tests**: `tests/unit/test_player_sprites.py` — gait maths (`advance_gait_phase`,
  `pick_pose`), the 9-pose sprite sets (sizes, transparency, shirt/skin/hair colours,
  hair at the back, standing symmetry, the two leading sides being mirror images,
  stride length growing), drawing (rotated to the heading, sized from the radius,
  team/keeper colours, translucent when inactive, longer sprite when striding),
  `SimTimeDelta`, and the App wiring (strides and spin follow `match.time_s`
  whatever `_sim_speed` / fps are; paused = no animation).
- **Always-on extras** drawn around every player: a `player_id` label under
  them, two tiny stat bars beneath the label (stamina, then speed — the stamina
  bar is green / yellow / red by level, the speed bar light blue), motion "speed lines" trailing behind a
  player above a speed threshold, a pulsing outermost ring when stamina is
  low (`STAMINA_FLASH_OUTLINE`), and a `SELECTED_OUTLINE` ring on whoever is
  selected. All tunables live in `graphics.json`.
- **Action icons**: `App._wire_player_icon_callbacks` hooks each player's
  `on_kick`/`on_tackle`/`on_possession_gained` to set `player.action_icon`
  (⚽ kick, 🦵 tackle, 🧤 for a goalkeeper's tackle/possession);
  `App._poll_action_icons` harvests and clears it each frame into
  `ActionIconState`, which keeps the icon on screen for a wall-clock
  `graphics.json["action_icons"]["linger_s"]` (independent of sim speed or
  pausing) and `draw_player` floats it above the player. Emoji font lookup
  is `_EMOJI_FONT_CANDIDATES` in `renderer.py` (matched by registered family
  name, e.g. `segoeuiemoji`, not filename — see the comment there).
- **Goalkeepers** are drawn in `GOALKEEPER_COLOUR` (a distinct orange)
  instead of their team colour, so the keeper is identifiable at a glance.
- **No possession outline.** The player with the ball used to get a white ring
  (`has_ball` argument, `POSSESSION_OUTLINE`, `player.possession_outline_thickness`);
  removed — the action icons say enough. `draw_player` no longer takes `has_ball`.
- **All rings are one translucent layer UNDER the players and the ball**
  (`Renderer.draw_player_rings` + `_draw_ball_state_ring`, called by
  `draw_pitch_and_ball(..., players=, selected_id=)` right after the shadow pass and before
  the goal frame, the ball and every sprite — `draw_player` no longer draws any ring and no
  longer takes `selected`). They used to be drawn on top of everything. Their opacity is
  **`graphics.json["rings"]["alpha"]`** (0-255, default 140 — raised from an initial 110 on
  request; 0 hides every ring): it applies to
  the selection, control-delay, tackled and low-stamina rings and the ball's state ring alike
  (`_draw_ring(..., alpha)` sets the alpha on the supersampled ring layer). `draw_ball(ring=True)`,
  the default for standalone use, still draws the ball's ring on top; `draw_pitch_and_ball` passes
  `ring=False` and draws it in the layer. Radii: stamina flash +11px, selection +7px, state ring
  +4px. Tests: `test_the_state_rings_are_translucent_and_the_tackled_red_ring_is_kept`,
  `test_the_ring_alpha_is_configurable`, `test_rings_are_drawn_under_the_player_sprite`,
  `test_the_pipeline_draws_player_rings_under_every_sprite`,
  `test_the_pipeline_draws_the_ball_ring_under_the_ball`.
- **CONTROLLING_BALL** (`PlayerState.CONTROLLING_BALL`): a cyan
  `CONTROL_DELAY_OUTLINE` ring.
  Indicates the player is mid first-touch control delay and can't yet move
  or be given orders. (Phase G)
- **Inactive players** (`PlayerState.INACTIVE_TACKLED`): translucent fill
  (`INACTIVE_ALPHA`) **plus** a red `INACTIVE_OUTLINE` ring (removed for a while, then restored
  on request), so the state is visible even when the player blends into the pitch. (Phase G)
- **Top layer for ball carrier**: `App._draw_match` sorts the player list so
  whichever player has the ball is drawn last, i.e. on top of every other
  player - so the ball carrier is never partly covered by a nearby defender drawn
  afterwards.
- **Heading indicator** (sprites-disabled fallback only): a broad, thin "V" — two unfilled lines touching
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

`Renderer._draw_ball_state_ring` draws a thin translucent (`rings.alpha`) outline ring around the
ball, UNDER it (see "All rings are one translucent layer"), indicating its current state.  Priority (only one ring shown at a time):
- **Amber** (`BALL_STATE_BOUNCED_OUTLINE`): `ball.just_bounced_timer_s > 0` —
  ball made a real bounce recently (decays after 0.3 s).
- **Blue** (`BALL_STATE_FLYING_OUTLINE`): ball is airborne (`z > 0.05 m`) and
  not possessed.
- **Green** (`BALL_STATE_ROLLING_OUTLINE`): ball is on the ground and rolling
  (`speed_xy > 0.05 m/s`) and not possessed.
No ring is drawn when the ball is possessed or stationary on the ground.

## Interaction scheme (`input.py`)

Mouse events are dispatched by `App._handle_match_mouse_event`. Left button:
`MOUSEBUTTONDOWN` -> `handle_mouse_down`, `MOUSEBUTTONUP` ->
`handle_mouse_up` (a click is resolved on *release*). Right button:
`MOUSEBUTTONDOWN` only, routed to the kick UI's `regress_kick_ui()` if it's
open, otherwise `handle_right_click()`.

**Left-click — select / move / kick UI** (`_handle_click`):
- **Click any player, either team** -> select them. Selecting an opponent is
  *inspection-only* (the side panel, see "Player inspector panel"): every
  order-issuing path below only ever targets whoever is currently selected,
  so an opponent can be looked at but never commanded.
- **Click the already-selected player** -> if they have the ball, enter the
  multi-phase kick UI (below); otherwise **nothing happens**. There is no
  click-again-to-deselect (an older version of this doc, and the in-app help
  overlay, claim there is — the code never deselects; selection only ever
  changes to another player).
- **Click empty ground** (while a player is selected) -> `MoveOrder` to that
  world position (always `sprint=True` - no UI toggle yet for jog vs
  sprint). With nobody selected, a ground click does nothing.
- A mouse-down/up pair further apart than `CLICK_DRAG_THRESHOLD_PX` is
  treated as a drag and does **nothing** at all (the old click-drag-to-kick
  scheme is gone; `DragState` and `drag_indicator()` survive only as legacy
  shims).

**Right-click — Get Possession** (`handle_right_click`): right-clicking an
*opposing-team* player while any player is selected issues
`GetPossessionOrder()` to the *selected* player. Two things to know:
- It is **not** a targeted tackle. `GetPossessionOrder` has no
  target-player field: it sprints at the ball if loose, or at whoever
  *currently carries* it and attempts one tackle on contact
  (`orders.py::GetPossessionOrder`). The clicked opponent only *gates* the
  action (must be opposing-team) — right-clicking an opponent who isn't the
  carrier still chases the ball/carrier, not them. If UI-driven targeted
  tackles are wanted, that means wiring `Player.tackle_player()` /
  `ChaseTackleOrder`, which the UI does not currently use anywhere.
- No-op for a same-team click, empty ground, or nothing selected.
It's shown in the log as "Get Possession". `TackleOrder` no longer exists
(see the top-level `knowledge.md`); left-clicking an opponent used to issue
one and now only selects.

**Multi-phase kick UI** (`KickUIState`/`KickPhase`; replaces the old
click-drag kick). Entered by left-clicking the selected ball-carrier, or by
pressing `K` when the selected player has the ball (`try_enter_kick_ui`;
`K` falls back to Shoot mode otherwise — see below). Entering it immediately
issues a `StopOrder` (via `HybridPlayerAI.issue_order()` if neural) and
**pauses the match** (`App._on_kick_ui_entered`), so all three phases happen
frozen. Each phase is committed by a left-*click* (on mouse-up), previewed
live by `Renderer.draw_kick_ui` (trajectory from `kick_trajectory.py`,
coloured by height, plus a 1-sigma error cone):
1. **`AIM_XY`** — mouse position relative to the player sets aim direction;
   distance sets `power_fraction`, scaled against the distance to the nearest
   goal clamped to [10, 20] m.
2. **`AIM_Z`** — mouse *distance* from the player sets elevation (close = max
   loft, far = flat; cap `graphics.json["kick_ui"]["max_loft_angle_deg"]`).
3. **`SPIN`** — mouse angle around the player sets the spin axis, distance
   sets magnitude (capped by `max_spin_rad_s` for the player's
   `kick_precision`). Left-click fires `player.kick_with_direction()`
   directly — no `KickOrder` — and the match stays paused
   (`App._on_kick_issued`; press Space to watch it fly).

`Esc` cancels at any phase and resumes play; **right-click regresses one
phase** (and cancels from `AIM_XY`); the mouse wheel fine-tunes the current
phase (`handle_mouse_wheel`: power ±0.00225/notch, elevation ±0.125°/notch,
spin magnitude ±0.5/notch — a no-op on spin if it's currently zero). The
UI aborts itself if the player loses the ball mid-phase. Right-click in the
kick UI takes priority over Get Possession.

**Keyboard-driven orders** (all act on the *selected* player):
- **`P`** -> one-shot "Pass mode" (`enter_pass_mode()`, `OrderMode`). The
  next click on a same-team player or empty ground issues a `PassOrder` at
  that player/position, then auto-reverts to `OrderMode.MOVE`
  (`_issue_transient_order`). Enters regardless of who has the ball, even
  though the hotkey bar dims `[P]` when the selected player lacks it.
- **`K`** -> if the selected player has the ball, enters the kick UI (above).
  Otherwise enters one-shot "Shoot mode": the next click on any pitch point
  issues a `ShootOrder` (z=1.0m, full power). Effectively Shoot mode is only
  reachable by a player *without* the ball, and note the hotkey bar's `[K]`
  entry is lit precisely in the case where `K` opens the kick UI.
- **`S`** -> `SaveOrder`, goalkeepers only (`issue_save_order()`).
- **`X`** -> `StopOrder` (`issue_stop_order()`).
- `Esc` cancels, in order of priority: help overlay -> kick UI -> params
  screen -> transient order mode -> back to menu (and finally quit).

**Pause interplay** (`App._on_new_order`, `_on_human_order_complete`): issuing
any new order resumes play automatically (no separate Space press), and when
a human-issued order *completes* the match auto-pauses with a banner
(`_pause_notification`, drawn by `draw_pause_notification`: "<id>:
<order> complete — Space to resume"). Orders that never complete on their
own (e.g. Save) therefore never trigger the auto-pause.

## Player inspector panel (`renderer.draw_player_inspector`)

Drawn top-right (below the help/speed/zoom row) for `panel_player()` — which
is just `selected_player()`, so it appears for **either team's** player as
soon as they're left-clicked. Read-only. Shows: `player_id (LEFT|RIGHT)`;
what drives them (`_describe_player_ai`: "No AI (order-driven only)" for a
human-controlled player, "Neural net", "Neural net (hybrid[, override
active])" for `HybridPlayerAI`, else "Rules (<ClassName>)"); the active
order's type plus one `field: value` line per dataclass field
(`_format_order_lines`, generic via `dataclasses.fields()` so new Order types
need no UI work; `on_complete` and `_private` fields skipped) or "(no active
order)"; and the eight attribute ratings as red->green bars
(`_ATTRIBUTE_LABELS`, HSV-interpolated by `_attribute_bar_colour`). Lines are
truncated to 60 chars. When the match HUD has neural players it also shows
a `Value:` line of each `NeuralPlayerAI`'s latest value-head prediction
(`App._value_prediction_line`).

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
every control and what each visual indicator means (goalkeeper colour,
inactive translucency and its red ring, ball rings). Match input events
are suppressed while the overlay is open (only the help button/`H`/`Esc`
are handled) so you can't accidentally issue orders while reading it; `Esc`
closes the overlay first before falling back to its normal kick-UI-cancel /
pass-mode-cancel / return-to-menu behaviour.

The overlay content is the module-level `_HELP_SECTIONS` list in `app.py`
(sections of `(key, description)` rows: Mouse, Kick aiming, Keys, Match
controls, Indicators). It is **hand-maintained, so update it whenever
`input.py` or `App._handle_keydown` change** — it had drifted badly before
(still describing click-drag kicking / Shift-lob, click-to-deselect, and
right-click "tackling them"). `_draw_help_overlay` word-wraps each
description (`_wrap_help_text`) to its column and flows whole sections down
the columns, starting a new column when the next section won't fit; the
column count is however many fit at a 440px minimum width (2 at the default
~1150px window). At the 640x480 minimum window there is only one column and
the last sections are clipped off the bottom — there is no scrolling.

## Hotkey bar (`renderer.draw_hotkey_bar` / `App._hotkey_entries`)

A permanent strip at the bottom of the screen shows every hotkey at all
times.  Each entry is rendered in one of three states:
- **Active** (accent colour): the key is the current transient mode, e.g.
  `[P] Pass` while PASS mode is engaged.
- **Enabled** (bright): the action is currently valid for the selected
  player (e.g. `[K] Shoot` lights up only if the selected player has the
  ball — which, per `input.py`, is exactly when `K` opens the kick UI
  rather than Shoot mode).
- **Disabled** (dim but readable): action is not valid right now (no
  selection, wrong player type, etc.).

This replaces the old inline key-hint text in the HUD, which only appeared
when a player was selected. `App._hotkey_entries()` computes the nine
entries (`[Spc]`, `[P]`, `[K]`, `[S]`, `[X]`, `[RClk] Get Possession`,
`[Z]`, `[H]`, `[Esc]`) and their states from the current
selection/ball/mode; training mode inserts a tenth, `[N]`, before `[H]`.
(`[RClk]` is lit whenever *any* player is selected, though it only acts on
an opposing-team right-click.)

## Scenarios (Phase H)

`SCENARIOS` originally held six `ScenarioDefinition` objects, all fully
parameterized (table below). It has since grown — the full key list in
`scenarios.py` is `phase1_neural_ai`, `1v1_phase1` (both shown in the menu's
left "AI Scenarios" column, everything else in the right "Balance Scenarios"
column), `save_close`, `pass`, `tackle`, `goal_to_goal_sprint`,
`sprint_shuttle`, `sprint`, `2v2`, `1v2`, `repulsion_obstacle`,
`repulsion_obstacle_no_ball`, `mark_standoff`, `penalty_corner_accuracy`,
`gk_far_post` — the newer ones aren't described in this file yet; read their
`ScenarioDefinition` in `scenarios.py`. The original six:

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

`GameLog` is a `collections.deque(maxlen=50)` of `LogEntry(time_s, level,
message, detail)`. `App._wire_match_log(match)` attaches a closure to
`match.log_callback` that calls `game_log.add(level, msg, match.time_s, detail)`.
A new closure is attached each time a new match is built (new trial, training
mode start) so old match references don't leak.

`renderer.draw_game_log(surface, game_log, min_level)` draws the most recent
8 rows in a semi-transparent box in the bottom-right corner, newest at the
bottom, each prefixed with its match clock (`MM:SS.mmm`). `L` hotkey in-match
cycles `App.log_min_level` between `INFO` and `DEBUG`. A tackle's numeric
roll/modifier breakdown is attached to its (INFO) entry as `LogEntry.detail`
— shown as a hover tooltip on rows with a dim `[Explain]` suffix, not as
separate DEBUG lines.

**Spam collapsing** (`GameLog.collapsed_entries`, what `draw_game_log`
draws): a run of *consecutive* entries with an identical `message` (and
level) is merged into one row, shown as a clock range plus a count, e.g.
`00:12.000-00:13.000  p1 tackled p2 — won the ball (2x)`. Details:
- Merging is done at *display time* over the level-filtered list, not in
  `add()`, so raw entries (`all_entries`, `entries_above`) are never mutated
  and an INFO message with DEBUG lines interleaved still collapses in INFO
  view (in DEBUG view the interleaved line breaks the run, correctly).
- Only exact message matches merge, and never across a different message
  (A, B, A stays three rows).
- The merged row keeps the *latest* occurrence's `detail` for the hover
  tooltip; earlier occurrences' breakdowns aren't reachable from it.
- Counts only cover entries still in the 50-entry ring buffer, so a run
  longer than that reports at most 50 (and its start time drifts forward as
  the oldest entries are evicted). Raise `GameLog(max_entries=...)` in
  `App.__init__` if that ever matters.
- Rows trim the *message* by pixel width (not a fixed character count) so
  the `(Nx)` count and `[Explain]` suffix are never what gets cut off.
Tests: `tests/unit/test_gamelog.py` (`test_collapse_*`).

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
- Shoot mode (`K` without the ball) always fires at full power
  (`power_fraction=1.0`) with a fixed aim height of 1.0m; power/height are
  only controllable through the kick UI (which needs the ball).
- Right-click Get Possession can't target a specific opponent (see
  "Interaction scheme"); no UI path issues `ChaseTackleOrder`.
- `MatchInputController.on_kick_ui_entered` is **not declared as a
  dataclass field** — `App._wire_fresh_match` assigns it dynamically, but
  `_enter_kick_ui` reads it unconditionally, so driving the controller
  without an `App` (e.g. a unit test that calls `try_enter_kick_ui()`)
  raises `AttributeError`. Confirmed: the attribute is absent from
  `__dataclass_fields__`. There are currently no tests for
  `MatchInputController` at all.
- No sound, no game clock/timer, no formations/kickoff - out of scope for
  this milestone (mirrors the engine's own documented gaps).
