# Football Coach

A personal project: a football (soccer) coaching game where players are
eventually controlled by independently-acting neural networks (trained with
PPO), while a human coach can set tactics and issue live orders.

**Current milestone: headless simulation engine + a pygame UI + high-level
player actions.** The core physics/rules engine (movement, ball physics,
kicking, passing, tackling, goalkeeper saves, possession, offside, scoring)
is complete and covered by an extensive test suite. On top of the engine,
`actions.py` provides simple, named functions (`move_to`, `shoot`,
`pass_to`, `tackle`, `save`) for controlling players. A pygame-ce UI sits on
top of the engine for manual play/testing: a training mode (1 player +
ball, free play) and a picker for a few illustrative balance scenarios
(penalty, tackle, sprint). There are no neural networks yet - the engine is
built so an RL training loop can be layered on top later without touching
either the simulation or the UI.

See [Idea.md](Idea.md) for the original design brief.

## Project layout

```
src/footballcoach/
  config/       JSON-driven tunable constants (physics.json, attributes.json) + loader
  mathutils/    Vector3 helper + RNG-reduction utilities
  entities/     Player, Ball, Pitch, PlayerAttributes data classes
  generation/   Correlated attribute generation (player skill sampling)
  engine/       The simulation itself: movement, ball physics, collision,
                kicking, passing, possession/control-time, tackling,
                goalkeeping (Save action), offside, scoring, and the
                top-level Match loop that ties it all together
  orders.py     Move/Kick/Tackle/Pass/ChaseTackle/Save order types (the
                "instruction" layer the UI or a future NN policy issues to
                players)
  actions.py    Simple, literally-named action helpers: move_to, shoot,
                pass_to, tackle, save - each just assigns the corresponding
                order to a player
  ui/           pygame-ce renderer, mouse input handling, training mode, and
                a picker for illustrative balance scenarios
tests/
  unit/         Fast, deterministic unit tests per module
  scenario/     End-to-end scenarios at rng_reduction=1.0 (fully deterministic)
  balance/      Statistical balance tests at rng_reduction=0.3 (the default
                game setting), run over many trials, reporting full stats
                (not just pass/fail) so the numbers can be tuned
```

Each package under `src/footballcoach/` has its own `knowledge.md` explaining
its design and the reasoning behind its constants in more depth than this
top-level README - read those before making changes to a given area.

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency
management.

```bash
uv sync
```

## Running the game

```bash
uv run footballcoach
```

This opens a pygame window with a menu: pick **Training mode** (one player,
free play, goals counted and reset automatically) or one of the **balance
scenario** entries (a single live playthrough of a scenario also covered
statistically by `tests/balance/`). In a match:

- Click a player to select them.
- Click empty ground to send the selected player there (sprinting).
- Click-and-drag starting on the selected player to kick - drag direction
  and length set the kick's aim direction and power; hold Shift while
  dragging to loft/chip instead of driving it low. Only does something if
  that player currently has the ball.
- Click an opposing player to attempt a tackle.
- `P` then click a teammate or ground to issue a pass; `S` to send the
  selected goalkeeper to make a save.
- `Space` pauses/resumes the simulation; `Esc` returns to the menu (or quits
  from the menu); `H` or the help button (top-right) shows a full control
  reference in-game.

See [src/footballcoach/ui/knowledge.md](src/footballcoach/ui/knowledge.md)
for the full interaction scheme and rendering notes.

## Running tests

```bash
# Everything
uv run pytest

# Just fast unit tests
uv run pytest tests/unit

# Deterministic end-to-end scenarios
uv run pytest tests/scenario

# Statistical balance tests (prints full stats tables with -s)
uv run pytest tests/balance -s
```

Balance test results are also written to
[tests/balance/results/latest_results.json](tests/balance/results/latest_results.json)
after each run, so they can be inspected or diffed without re-running pytest.

## Tuning the game

Almost every constant that affects balance (speeds, accelerations, kick
error, tackle odds, control-time difficulty, attribute distributions, etc.)
lives in two JSON files, not in code:

- [src/footballcoach/config/physics.json](src/footballcoach/config/physics.json)
- [src/footballcoach/config/attributes.json](src/footballcoach/config/attributes.json)

Edit a value, re-run `uv run pytest tests/balance -s`, and check whether the
reported percentages/times land where you want them. The balance tests
encode the target ranges (e.g. "a 0.8 tackling player should beat a 0.6
dribbling player 70-90% of the time") as assertions, so you'll get a clear
pass/fail plus the actual measured number.

## Design principles carried through the code

- **SI units everywhere.** Metres, seconds, m/s, m/s², kg, radians.
- **Nothing is ever perfectly deterministic by skill alone** (kicks, tackles,
  ball control) - there's always some Gaussian/uniform noise, dampened but
  never fully removed by the `rng_reduction` game option (0=full randomness,
  1=fully deterministic).
- **The engine is UI-agnostic.** `Match` in
  [src/footballcoach/engine/match.py](src/footballcoach/engine/match.py) is
  the only thing that needs to be driven; a pygame renderer or an RL
  training loop are both future consumers of the same `Match.step()` API.
