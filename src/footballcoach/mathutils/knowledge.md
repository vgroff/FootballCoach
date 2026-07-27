# mathutils/

Small, dependency-light math helpers shared across the engine.

## `vector3.py`

`Vector3` is an immutable (`frozen`, `slots`) 3D vector with the usual
arithmetic operators, plus a few pitch-specific helpers:

- Axis convention: `x` = pitch length axis (goal-to-goal), `y` = pitch width
  axis, `z` = height above ground. This matches `entities/pitch.py`, where
  the pitch is centred at the origin and goals sit at `x = ±half_length`.
- `length_xy()` / `xy()` - ignore height, useful for "ground speed" and
  "ground position" since most movement/collision logic only cares about the
  horizontal plane.
- `angle_xy()` / `from_angle_xy()` - convert to/from a heading angle
  (radians, measured from +x), used by `engine/movement.py` for player
  heading and turn-rate integration.

It's backed by plain floats + a couple of `math` calls rather than numpy,
since per-vector operations on individual entities (one player, one ball)
don't benefit from numpy's vectorization and the overhead would hurt
per-tick performance in a tight simulation loop. `as_array()` /
`from_array()` exist for the rare spot that wants a numpy array (e.g. the
correlated attribute sampling in `generation/`, which operates on batches).

## `rng.py`

Implements the `rng_reduction` game option (see root README and Idea.md):

- `skill_roll(skill, rng_reduction, rng)` - dampens a uniform random roll
  towards the deterministic `skill` value as `rng_reduction -> 1.0`. Used by
  `engine/tackling.py` for tackle rolls.
- `reduced_sigma(sigma, rng_reduction)` - shrinks a Gaussian error's standard
  deviation towards 0 as `rng_reduction -> 1.0`. Used by `engine/kicking.py`
  for kick accuracy error.

Both formulas come directly from the user's design spec in Idea.md and
should not be changed without re-deriving the balance-test target
probabilities in `tests/balance/`.
