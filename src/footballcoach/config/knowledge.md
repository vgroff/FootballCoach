# config/

Holds every tunable balance constant for the game, as JSON, plus a small
cached loader.

## Files

- `physics.json` - world/pitch/player/ball geometry, movement, kicking,
  tackling, ball physics, control-time, and collision constants. Organised
  into sections matching the engine modules that consume them (e.g.
  `physics.json["movement"]` → `engine/movement.py`). Pure engine physics
  only — no AI, UI, or game-rule constants live here.
- `attributes.json` - player attribute generation: the base Gaussian
  distribution, inter-attribute correlations, and per-league "tier" presets
  (mean/sigma overrides) used by `generation/attributes.py`.
- `gameplay.json` - game rules and UI timing: `offside` toggle,
  `ui.scenario_linger_s`, `ui.goal_linger_s`. Read via `load_gameplay_config()`.
- `graphics.json` - all visual display constants: player/ball rendering,
  action icons, speed lines, stamina flash, heading indicator, and `kick_ui`
  (trajectory preview and spin input params). Read via `load_graphics_config()`.
- `loader.py` - loads and `functools.lru_cache`s all four JSON files. Use
  `load_physics_config()` / `load_attributes_config()` / `load_gameplay_config()` /
  `load_graphics_config()` rather than reading files directly, so config is
  only parsed once per process. Call `clear_config_cache()` in tests if you
  need to reload after editing a file on disk mid-test-run.
- AI steering and marking constants (`repulsion`, `marking`) live in
  `ai/config/ai_config.json`, loaded via `load_ai_config()` from
  `footballcoach.ai.config`.

## Why JSON instead of Python constants?

Per the project's design goal: the user should be able to tune game balance
(speeds, kick error, tackle odds, attribute distributions, etc.) by editing
values in a file, without touching code or understanding the surrounding
formulas. Every engine module reads its constants from here via a small
`*Params` dataclass with a `from_config()` static constructor (e.g.
`MovementParams.from_config()`), so there's exactly one place to look for
"what does this module depend on".

## Notable `physics.json` sections added in Phase G/H

- `ball_physics.just_bounced_display_duration_s` (0.3 s): how long the
  "just bounced" amber ring is shown on the ball after each real bounce.
  Pure display hint; no engine logic reads it.
- `ui.scenario_linger_s` (3.0 s): how long a `ScenarioLoop` trial keeps
  running after an outcome is detected before the next trial is built.
  Out-of-bounds events use **half** this value. Read at `ScenarioLoop`
  construction time; tests pass `linger_s=0.0` explicitly to skip the wait.
- `ui.goal_linger_s` (3.0 s): how long `Match` waits after a goal before
  calling `_reset_after_goal()` (i.e. how long the ball stays in the net).
  `Match.goal_linger_s` defaults to 0.0 (immediate reset) so headless tests
  are unaffected; the UI sets it from this config value when building matches.

## Adding a new tunable constant

1. Add the key under the relevant section of `physics.json` (or
   `attributes.json`).
2. Add a matching field to that module's `*Params` dataclass and its
   `from_config()` method.
3. Reference `params.your_new_field` in the module's logic - never hardcode
   a number that should be tunable.

## Gotcha

`attributes.json["correlations"]` includes a `_comment` key alongside the
real `"attrA:attrB": rho` entries - `generation/attributes.py` explicitly
skips keys starting with `_` when building the correlation matrix. If you
add more `_comment`-style keys elsewhere, make sure any code iterating over
that dict does the same.
