# config/

Holds every tunable balance constant for the game, as JSON, plus a small
cached loader.

## Files

- `physics.json` - world/pitch/player/ball geometry, movement, kicking,
  tackling, ball physics, and control-time constants. Organized into
  sections matching the engine modules that consume them (e.g.
  `physics.json["movement"]` is read by `engine/movement.py`).
- `attributes.json` - player attribute generation: the base Gaussian
  distribution, inter-attribute correlations, and per-league "tier" presets
  (mean/sigma overrides) used by `generation/attributes.py`.
- `loader.py` - loads and `functools.lru_cache`s both JSON files. Use
  `load_physics_config()` / `load_attributes_config()` rather than reading
  the files directly, so config is only parsed once per process. Call
  `clear_config_cache()` in tests if you need to reload after editing a file
  on disk mid-test-run (not needed in normal test runs).

## Why JSON instead of Python constants?

Per the project's design goal: the user should be able to tune game balance
(speeds, kick error, tackle odds, attribute distributions, etc.) by editing
values in a file, without touching code or understanding the surrounding
formulas. Every engine module reads its constants from here via a small
`*Params` dataclass with a `from_config()` static constructor (e.g.
`MovementParams.from_config()`), so there's exactly one place to look for
"what does this module depend on".

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
