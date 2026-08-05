# PPO scenario seed-pool plan

**Status:** Not yet implemented — planning only. Pick this up by reading this
file in full, then implementing section 3 in order.

## 1. Context / decision history

Discussed in an agent session on 2026-08-05 (see `training_runs.log` around
that date for the training runs that prompted this). Summary of the
reasoning, in case it needs to be re-litigated:

- **Problem observed**: `build_1v1_scenario()` in `src/footballcoach/ui/scenarios.py`
  calls `rng = random.Random()` — fully OS-entropy-seeded, a fresh unseeded
  draw every single episode. This makes:
  - BC/PPO training noisy in ways that are hard to separate from genuine
    policy-quality signal (is a bad rollout a bad policy, or a hard/unlucky
    scenario draw?).
  - Any specific rollout impossible to reproduce for debugging.
  - No way to hold out a fixed eval set of scenarios distinct from train.

- **Idea raised**: seed every trial's scenario RNG with a number drawn from
  `{1, ..., N}` instead of a fresh unseeded draw, so training only ever sees
  N unique base scenarios (positions, attributes, stamina, headings, ball
  state, opponent-type roll — everything that flows through the scenario's
  `rng`).

- **Objection raised (by the assistant) and resolved**: small N risks PPO
  memorizing/overfitting to specific scenario instances rather than learning
  general policy. **Resolution: this is an N-vs-training-bud589et ratio
  problem, not a fundamental flaw.** With N large enough (and growing across
  training, mirroring the existing but currently-unwired
  `curriculum.rng_reduction_start/end` annealing pattern in `ai_config.json`),
  memorization risk is negligible — this is exactly how Procgen/Sokoban-style
  RL benchmarks use large finite train seed sets for reproducibility without
  hurting generalization.

- **Scope decided (final)**:
  - **PPO only.** Do NOT touch demonstration recording
    (`record_demonstrations.py`) — BC training already benefits from full
    randomness per the existing `ai_trainer_knowledge.md` guidance ("have
    better/more varied move/kick/tackle data... run the same game scenario
    multiple times in the BC to show the difference" — that idea is about
    deliberately *repeating* specific scenarios in BC, which is a different,
    separate future task, not seed-pool-during-PPO).
  - **One seed per whole scenario** — not decomposed into independent
    position/attribute/ball-state sub-seeds (that was considered and
    explicitly rejected for simplicity; can revisit later if seed-pool alone
    doesn't reduce noise enough).
  - **Pool size (N) must vary across the curriculum** — start small-ish,
    grow as training progresses (same shape as `rng_reduction_start/end`
    linear annealing already in `TrainingSchedules`).
  - Held-out eval seed range was suggested as "cheap insurance" against
    memorization but is NOT required for the first implementation — flagged
    as a nice-to-have in section 6.

## 2. Relevant files (read these first)

| File | Role |
|---|---|
| `src/footballcoach/ui/scenarios.py` | `build_1v1_scenario()` (~line 1105) — the Phase 1 scenario builder. Currently `rng = random.Random()` unseeded. This is where the `seed` kwarg gets added. |
| `src/footballcoach/ai/env/scenario_env.py` | `ScenarioEnv.__init__`/`reset()` (~line 80-220) — owns `scenario_kwargs` forwarded to `definition.build()` on every `reset()`. This is where the seed-pool draw happens each episode. |
| `src/footballcoach/ai/ppo/schedules.py` | `TrainingSchedules` — existing schedule pattern (`rng_reduction_schedule`, `linear_anneal`). Add a new `seed_pool_size` schedule here, same shape. |
| `src/footballcoach/ai/ppo/ppo_trainer.py` | `PPOTrainer.train()` — the main loop where `progress = self._total_steps / total_steps` is already computed each step (search for that exact line). This is where `env.seed_pool_size` gets updated from the schedule each rollout (or each step — pool size changes slowly, updating once per rollout is enough). |
| `src/footballcoach/ai/config/ai_config.json` | Add `ppo.seed_pool_size_start` / `ppo.seed_pool_size_end` (or under `curriculum`, matching `rng_reduction_start/end`'s home — **prefer `curriculum`** for consistency, see section 3.3). |
| `src/footballcoach/ai/scripts/record_demonstrations.py` | **Do NOT touch.** Confirmed out of scope — demo recording keeps its own unseeded `random.Random()` per scenario, unaffected by this plan. |
| `src/footballcoach/ai/curriculum/envs.py` | `build_env(phase)` — where `ScenarioEnv` gets constructed for phase 1 in `train.py`'s path. Check whether `seed_pool_size` needs to be threaded through here as a CLI-configurable value, or just read from config directly inside `ScenarioEnv`. Recommend: config-driven default, no new CLI flag needed for v1. |
| `tests/ai_scenario/test_smoke.py` | Existing smoke tests for `ScenarioEnv`/`PPOTrainer` — add a new test here (or a new file `test_seed_pool.py`) per section 4. |
| `tests/ai_unit/test_reward.py` | Reference for existing unit test conventions/style in this codebase (config-driven test fixtures, `pytest.approx`, etc.) — not directly touched but useful as a style example. |

## 3. Implementation steps (in order)

### 3.1 `build_1v1_scenario()` — add `seed` kwarg

File: `src/footballcoach/ui/scenarios.py`, function starts at (as of this
writing) line 1105.

**Current signature (abbreviated):**
```python
def build_1v1_scenario(
    rng_reduction: float = 0.3,
    *,
    trainee_tier: str | None = None,
    opponent_tier: str | None = None,
    ball_max_speed_mps: float | None = None,
    restitution_sigma: float | None = None,
    ball_max_dist_from_trainee_m: float | None = None,
    stamina_min: float | None = None,
    stamina_max: float | None = None,
    trainee_team: "Team | None" = None,
    sim_dt_s: float = 1.0 / 30.0,
    opponent_rules_prob: float = 0.0,
    opponent_immobile_prob: float = 1.0,
    opponent_min_dist_m: float | None = None,
    opponent_max_dist_m: float | None = None,
) -> Match:
    """Phase 1 curriculum: 1v1 get-possession/move-toward-goal.
    ...
    """
    _cfg = _phase1_scenario_cfg()
    ...
    rng = random.Random()
    pitch = Pitch.standard()
```

**Change:** add `seed: int | None = None` to the keyword-only args, and pass
it straight into `random.Random(seed)`:

```python
def build_1v1_scenario(
    rng_reduction: float = 0.3,
    *,
    trainee_tier: str | None = None,
    opponent_tier: str | None = None,
    ball_max_speed_mps: float | None = None,
    restitution_sigma: float | None = None,
    ball_max_dist_from_trainee_m: float | None = None,
    stamina_min: float | None = None,
    stamina_max: float | None = None,
    trainee_team: "Team | None" = None,
    sim_dt_s: float = 1.0 / 30.0,
    opponent_rules_prob: float = 0.0,
    opponent_immobile_prob: float = 1.0,
    opponent_min_dist_m: float | None = None,
    opponent_max_dist_m: float | None = None,
    seed: int | None = None,
) -> Match:
    """Phase 1 curriculum: 1v1 get-possession/move-toward-goal.

    ``seed``: when given, the ENTIRE scenario (trainee/opponent team choice,
    positions, attributes, stamina, headings, ball placement/velocity/spin/
    restitution, and the opponent-type roll [rules/immobile/neural]) is
    fully deterministic for that seed value — every draw in this function
    goes through the single `rng` instance below. Used by PPO training's
    seed-pool mechanism (see ScenarioEnv.seed_pool_size / ai/knowledge.md)
    to replay a finite, growing set of scenarios instead of a fresh
    unseeded draw every episode, reducing rollout-to-rollout noise while
    keeping the pool large enough to avoid memorization (see
    agent_plans/ppo_scenario_seed_pool.md for the full design rationale).
    ``None`` (default) = fully random, matching prior behaviour exactly.
    NOT used by demonstration recording (record_demonstrations.py) — BC
    training intentionally keeps full randomness; this is a PPO-only change.

    Both players and the ball are placed randomly across the full pitch with
    ...
    """
    _cfg = _phase1_scenario_cfg()
    ...
    rng = random.Random(seed)   # was: random.Random()
    pitch = Pitch.standard()
```

**That is the ONLY line that changes inside the function body.** Every
downstream draw (`_rand_pos()`, `generate_attributes(rng=rng)`,
`rng.uniform(...)` for stamina/heading, ball velocity/spin/restitution, and
the final `_r = rng.random()` opponent-type roll) already goes through this
one `rng` instance, so seeding it fully determines the entire scenario.
`random.Random(None)` is equivalent to `random.Random()` (both fall back to
OS entropy / system time), so passing `seed=None` is a no-op change from
current behaviour — **verify this specific claim with a quick REPL check
before relying on it**, since it's the crux of backward compatibility:

```python
import random
a = random.Random(None)
b = random.Random()
# Both should differ from each other and be unseeded/non-reproducible.
# This confirms None behaves like the no-arg constructor.
```

(This is documented Python stdlib behavior — `random.Random(a=None)` seeds
from `os.urandom` or system time — but confirm in this repo's Python version
since it's a load-bearing assumption for backward compat.)

### 3.2 `ScenarioEnv` — add `seed_pool_size` + per-reset seed draw

File: `src/footballcoach/ai/env/scenario_env.py`.

**Add to `__init__`** (near the other simple int/float config attrs, e.g.
right after `self.secondary_player_ids`):

```python
def __init__(
    self,
    definition: ScenarioDefinition,
    trainee_player_id: str,
    phase: int = 1,
    rng_reduction: float = 0.3,
    max_episode_s: float = 120.0,
    linger_s: float = 0.0,
    rng: Optional[random.Random] = None,
    secondary_player_ids: Optional[list] = None,
    seed_pool_size: int = 0,
    **scenario_kwargs,
):
    ...
    self.secondary_player_ids: list = secondary_player_ids or []

    # --- Scenario seed pool (PPO-only noise reduction; see
    # agent_plans/ppo_scenario_seed_pool.md) ---
    # 0 (default) = disabled, fully random scenario every reset() (prior
    # behaviour, unchanged). >0 = each reset() draws a seed uniformly from
    # {1, ..., seed_pool_size} and forwards it as build_kwargs["seed"], so
    # only `seed_pool_size` unique base scenarios are ever seen. Mutated
    # externally by PPOTrainer.train() each rollout via a schedule (see
    # ai/ppo/schedules.py's seed_pool_size schedule) — NOT read from config
    # directly here, so ScenarioEnv stays agnostic to *how* the caller
    # decides to grow the pool over training.
    self.seed_pool_size: int = seed_pool_size
    # Separate, deliberately UNSEEDED rng — this one only decides WHICH pool
    # seed to draw each reset(), it must not itself be part of the
    # seed-pool determinism (otherwise the sequence of pool-seed draws would
    # itself be reproducible in a way nobody asked for, and more importantly
    # would couple with self.rng used for obs slot shuffling below).
    self._seed_pool_rng = random.Random()
```

**Modify `reset()`** — currently:
```python
def reset(self) -> ObservationBatch:
    """Start a new trial and return the initial observation."""
    from footballcoach.rules_ai import NeuralPlayerAI
    # Inject sim_dt_s so build functions can pass it to Match.
    # This has no effect on builds that don't accept it (e.g. phase 2).
    build_kwargs = {**self.scenario_kwargs, "sim_dt_s": self._dt_s}
    self._loop = ScenarioLoop(
        definition=self.definition,
        max_trials=0,
        rng_reduction=self.rng_reduction,
        linger_s=self.linger_s,
        kwargs=build_kwargs,
        timeout_ticks=int(self.max_episode_s / self._dt_s),
    )
```

**New:**
```python
def reset(self) -> ObservationBatch:
    """Start a new trial and return the initial observation."""
    from footballcoach.rules_ai import NeuralPlayerAI
    # Inject sim_dt_s so build functions can pass it to Match.
    # This has no effect on builds that don't accept it (e.g. phase 2).
    build_kwargs = {**self.scenario_kwargs, "sim_dt_s": self._dt_s}
    # Seed-pool draw (see agent_plans/ppo_scenario_seed_pool.md). Silently a
    # no-op for scenario builders that don't accept a `seed` kwarg (e.g.
    # build_penalty_scenario for phase 2) IF seed_pool_size stays 0 for those
    # phases — do NOT enable seed_pool_size for phase 2 until its builder
    # also accepts `seed`, or this will raise TypeError.
    if self.seed_pool_size > 0:
        build_kwargs["seed"] = self._seed_pool_rng.randint(1, self.seed_pool_size)
    self._loop = ScenarioLoop(
        definition=self.definition,
        max_trials=0,
        rng_reduction=self.rng_reduction,
        linger_s=self.linger_s,
        kwargs=build_kwargs,
        timeout_ticks=int(self.max_episode_s / self._dt_s),
    )
```

**Important gotcha to flag in code review**: `scenario_kwargs` is a dict
built once at `ScenarioEnv.__init__` time and merged fresh into
`build_kwargs` every `reset()` call — confirm `seed` doesn't collide with
anything already in `self.scenario_kwargs` (it shouldn't, since no current
caller passes `seed=...` through `ScenarioEnv(**scenario_kwargs)`, but grep
for `ScenarioEnv(` call sites in `train.py`/`curriculum/envs.py` to be sure
before merging).

### 3.3 `ai_config.json` — pool-size schedule config

Add next to `rng_reduction_start`/`rng_reduction_end` in the `curriculum`
section (keeping the two annealing concepts co-located, since they're
philosophically the same kind of curriculum knob — "how much variety does
the trainee see, and does that grow over training"):

```json
"curriculum": {
    "...": "...",
    "rng_reduction_start": 0.55,
    "rng_reduction_end": 0.3,
    "seed_pool_size_start": 200,
    "seed_pool_size_end": 20000,
    "_comment_seed_pool": "PPO-only scenario seed pool (see agent_plans/ppo_scenario_seed_pool.md). Each PPO episode's build_1v1_scenario() call is seeded with a value drawn uniformly from {1, ..., N} where N linearly grows from seed_pool_size_start to seed_pool_size_end over training progress (same shape as rng_reduction_start/end). 0/0 = fully disabled (unseeded every episode, prior behaviour). Larger N = less memorization risk but less noise reduction; smaller N = more noise reduction but more overfitting risk if N is small relative to how many times PPO revisits the same seed within training. NOT used by record_demonstrations.py (BC keeps full randomness)."
}
```

**Starting values (200 → 20000) are a rough guess, not tuned** — pick
something and iterate empirically by watching whether `val=`/`kl=`
noise in the training log actually drops without evidence of the policy
memorizing (e.g. suspiciously perfect performance on train scenarios that
doesn't transfer to eval — see section 6's held-out eval seed range for how
to actually check this).

### 3.4 `TrainingSchedules` — new schedule function

File: `src/footballcoach/ai/ppo/schedules.py`.

Mirror the existing `rng_reduction_schedule` exactly:

```python
def seed_pool_size_schedule(cfg: dict):
    """Build a seed_pool_size schedule from ai_config.json['curriculum'].

    Linearly anneals from ``seed_pool_size_start`` to ``seed_pool_size_end``
    over the course of training. Returns 0 (disabled) if either config key
    is absent, matching the "fully random, no seed pool" default behaviour.
    See agent_plans/ppo_scenario_seed_pool.md for full rationale.
    """
    start = float(cfg.get("seed_pool_size_start", 0))
    end = float(cfg.get("seed_pool_size_end", 0))
    return linear_anneal(start, end)
```

Add to `TrainingSchedules.__init__`:
```python
self.seed_pool_size = seed_pool_size_schedule(curriculum_cfg)
```

Add accessor method (mirrors `.rng()`):
```python
def seed_pool(self, progress: float) -> int:
    """Scenario seed-pool size (PPO-only) — see .rng() for the analogous
    rng_reduction schedule. Rounded to nearest int; 0 = disabled."""
    return max(0, round(self.seed_pool_size(progress)))
```

### 3.5 `PPOTrainer.train()` — wire the schedule into `env.seed_pool_size`

File: `src/footballcoach/ai/ppo/ppo_trainer.py`, inside the `while
self._total_steps < total_steps:` loop in `train()`. The loop already
computes `progress = self._total_steps / total_steps` as its very first
line every step — the cheapest hook point is right there, but updating
`env.seed_pool_size` doesn't need to happen every single step (pool size
changes slowly); doing it once per rollout (right after the PPO update, or
right at the top of the rollout, before `env.step()` is called for the
first step of that rollout) is enough and avoids a per-step attribute write
for no benefit. Recommended: update it right where `steps_this_rollout`
resets to 0 after a completed update (search for `steps_this_rollout = 0` in
`train()`), OR simpler — update every step anyway since it's just an int
assignment and the schedule call is cheap (a single linear interpolation):

```python
while self._total_steps < total_steps:
    progress = self._total_steps / total_steps
    env.seed_pool_size = self.schedules.seed_pool(progress)

    # --- Collect one decision step ---
    next_obs, reward, done, info = env.step()
    ...
```

**Caveat**: `env.seed_pool_size` only takes effect on the NEXT `env.reset()`
call (mid-episode changes don't retroactively reseed an in-progress
episode) — this is fine and expected, just document it in a comment at the
call site so nobody expects it to affect the currently-running episode.

**Also update the per-rollout log line** (optional but recommended for
visibility) — in the multi-line `_lines` block built for the `[PPO]` log
(search for `_lines = [` in `train()`), consider adding the current pool
size somewhere, e.g. appended to the existing step/speed/reward line:
```python
f"[PPO] step={self._total_steps:,}  speed={steps_per_sec:.0f}/s  "
f"reward={mean_ep_reward:.2f}{opp_rew_str}  seed_pool={env.seed_pool_size}",
```
This makes it trivial to eyeball in `training_runs.log` whether the pool is
actually growing as expected during a real run.

### 3.6 `curriculum/envs.py` / `train.py` — confirm no extra wiring needed

Check `src/footballcoach/ai/curriculum/envs.py`'s `build_env(phase)` for
phase 1 — it should already just construct `ScenarioEnv(...)` with whatever
kwargs it currently uses; **no changes needed there** for v1 since
`seed_pool_size` defaults to `0` (disabled) at `ScenarioEnv.__init__` and
gets mutated externally by the trainer post-construction (`env.seed_pool_size
= ...` in `train()`, not a constructor kwarg threaded through `build_env`).
This keeps the CLI/`train.py` surface unchanged for v1 — no new `--seed-pool`
flag needed. (A future v2 could add one to override the config defaults per
run, but is not required for the first cut.)

## 4. Tests to add

### 4.1 Unit test: `build_1v1_scenario(seed=N)` determinism

New test, could live in a new `tests/ai_unit/test_scenario_seeding.py` or
appended to an existing scenarios-adjacent test file if one already covers
`build_1v1_scenario` (grep for `build_1v1_scenario` under `tests/` first to
avoid creating a stray duplicate file — check `tests/ai_scenario/` and
`tests/scenario/` too, not just `tests/ai_unit/`).

```python
"""Tests for build_1v1_scenario's optional `seed` kwarg (PPO seed-pool
support). See agent_plans/ppo_scenario_seed_pool.md.
"""
import pytest

from footballcoach.ui.scenarios import build_1v1_scenario


class TestBuild1v1ScenarioSeeding:
    def test_same_seed_gives_identical_scenario(self):
        """Same seed -> identical trainee/opponent positions, attributes,
        ball state, and opponent-type roll (bitwise-reproducible)."""
        m1 = build_1v1_scenario(seed=42)
        m2 = build_1v1_scenario(seed=42)

        t1, t2 = m1.player_by_id("trainee"), m2.player_by_id("trainee")
        o1, o2 = m1.player_by_id("opponent"), m2.player_by_id("opponent")

        assert t1.position.x == pytest.approx(t2.position.x)
        assert t1.position.y == pytest.approx(t2.position.y)
        assert t1.stamina == pytest.approx(t2.stamina)
        assert t1.heading_rad == pytest.approx(t2.heading_rad)
        assert o1.position.x == pytest.approx(o2.position.x)
        assert o1.position.y == pytest.approx(o2.position.y)

        assert m1.ball.position.x == pytest.approx(m2.ball.position.x)
        assert m1.ball.position.y == pytest.approx(m2.ball.position.y)
        assert m1.ball.velocity.x == pytest.approx(m2.ball.velocity.x)
        assert m1.ball.velocity.y == pytest.approx(m2.ball.velocity.y)
        assert m1.ball.spin.x == pytest.approx(m2.ball.spin.x)

        # Opponent-type roll (rules/immobile/neural) must also match.
        assert getattr(m1, "_opponent_use_rules_ai", None) == getattr(m2, "_opponent_use_rules_ai", None)
        assert getattr(m1, "_opponent_is_immobile", None) == getattr(m2, "_opponent_is_immobile", None)

    def test_different_seeds_give_different_scenarios(self):
        """Different seeds should (almost always) produce different
        positions — this is a statistical sanity check, not a strict
        guarantee, but collision probability is astronomically low for
        continuous position draws."""
        m1 = build_1v1_scenario(seed=1)
        m2 = build_1v1_scenario(seed=2)
        t1, t2 = m1.player_by_id("trainee"), m2.player_by_id("trainee")
        assert (t1.position.x, t1.position.y) != (t2.position.x, t2.position.y)

    def test_seed_none_matches_prior_unseeded_behaviour(self):
        """seed=None (default) must remain fully random — two calls should
        essentially never produce the same trainee position (statistical
        check, not a hard guarantee)."""
        m1 = build_1v1_scenario(seed=None)
        m2 = build_1v1_scenario(seed=None)
        t1, t2 = m1.player_by_id("trainee"), m2.player_by_id("trainee")
        # Extremely unlikely to collide by chance on a continuous pitch coord.
        assert (t1.position.x, t1.position.y) != (t2.position.x, t2.position.y)

    def test_seed_pool_bounded_variety(self):
        """Sampling seeds from a small pool {1..5} should produce at most 5
        distinct trainee starting positions across many draws."""
        import random
        rng = random.Random(0)
        positions = set()
        for _ in range(50):
            seed = rng.randint(1, 5)
            m = build_1v1_scenario(seed=seed)
            t = m.player_by_id("trainee")
            positions.add((round(t.position.x, 6), round(t.position.y, 6)))
        assert len(positions) <= 5
```

### 4.2 Unit/integration test: `ScenarioEnv.seed_pool_size` behaviour

Could live in `tests/ai_scenario/test_smoke.py` (append) or a new
`tests/ai_scenario/test_seed_pool.py`. Needs a real `ScenarioEnv` around
`build_1v1_scenario`.

```python
"""Tests for ScenarioEnv's seed_pool_size mechanism (PPO-only scenario
determinism). See agent_plans/ppo_scenario_seed_pool.md.
"""
import functools

from footballcoach.ai.env.scenario_env import ScenarioEnv
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario


def _make_env(seed_pool_size: int = 0) -> ScenarioEnv:
    defn = ScenarioDefinition(
        key="_test_seed_pool",
        label="seed pool test",
        description="test",
        build=build_1v1_scenario,
    )
    return ScenarioEnv(
        definition=defn,
        trainee_player_id="trainee",
        phase=1,
        max_episode_s=30.0,
        seed_pool_size=seed_pool_size,
    )


class TestScenarioEnvSeedPool:
    def test_disabled_by_default(self):
        env = _make_env()
        assert env.seed_pool_size == 0

    def test_pool_bounded_variety_across_many_resets(self):
        """With seed_pool_size=3, repeated reset() calls should only ever
        produce 3 distinct trainee starting positions."""
        env = _make_env(seed_pool_size=3)
        positions = set()
        for _ in range(30):
            env.reset()
            t = env._loop.match.player_by_id("trainee")
            positions.add((round(t.position.x, 6), round(t.position.y, 6)))
        assert len(positions) <= 3

    def test_disabled_pool_gives_full_variety(self):
        """seed_pool_size=0 (disabled) should behave like before: many
        distinct positions across resets, not bounded to a tiny set."""
        env = _make_env(seed_pool_size=0)
        positions = set()
        for _ in range(30):
            env.reset()
            t = env._loop.match.player_by_id("trainee")
            positions.add((round(t.position.x, 6), round(t.position.y, 6)))
        # Overwhelmingly likely to see > 3 distinct positions in 30 unseeded draws.
        assert len(positions) > 3

    def test_mutating_seed_pool_size_after_construction(self):
        """PPOTrainer.train() mutates env.seed_pool_size directly post-hoc
        (not via constructor) each rollout — confirm this actually changes
        behaviour on the NEXT reset(), matching PPOTrainer's usage pattern."""
        env = _make_env(seed_pool_size=0)
        env.seed_pool_size = 2
        positions = set()
        for _ in range(20):
            env.reset()
            t = env._loop.match.player_by_id("trainee")
            positions.add((round(t.position.x, 6), round(t.position.y, 6)))
        assert len(positions) <= 2
```

### 4.3 Unit test: `TrainingSchedules.seed_pool()`

Append to wherever `TrainingSchedules`/schedule functions are already
tested, or add a small standalone test if none exists yet (grep
`tests/ai_unit` for `rng_reduction_schedule`/`TrainingSchedules` first — if
there's an existing `test_schedules.py`, add there instead of creating a
new file):

```python
from footballcoach.ai.ppo.schedules import TrainingSchedules


class TestSeedPoolSchedule:
    def test_disabled_when_config_absent(self):
        sched = TrainingSchedules(ppo_cfg={}, curriculum_cfg={})
        assert sched.seed_pool(0.0) == 0
        assert sched.seed_pool(1.0) == 0

    def test_linear_anneal_matches_start_end(self):
        cfg = {"seed_pool_size_start": 100, "seed_pool_size_end": 1000}
        sched = TrainingSchedules(ppo_cfg={}, curriculum_cfg=cfg)
        assert sched.seed_pool(0.0) == 100
        assert sched.seed_pool(1.0) == 1000
        assert sched.seed_pool(0.5) == pytest.approx(550, abs=1)
```

### 4.4 Regression check: existing smoke tests must still pass unmodified

`tests/ai_scenario/test_smoke.py` and `tests/ai_scenario/test_pretrain_combined_smoke.py`
construct `ScenarioEnv` without `seed_pool_size` — confirm they still pass
as-is after this change (they should, since the default is `0` = disabled,
fully backward compatible). Run:
```bash
uv run pytest tests/ai_scenario/ tests/ai_unit/ -q
```
after implementing, before considering this done.

## 5. Manual verification checklist (after implementation)

1. Run a short training burst and confirm the new `seed_pool=` field appears
   in the `[PPO]` log line and visibly grows over time:
   ```bash
   uv run python -m footballcoach.ai.scripts.train \
       --phase 1 --seed 42 \
       --bc-dataset demonstrations/phase1 \
       --total-steps 200000 \
       --checkpoint-dir checkpoints/_seed_pool_test/ 2>&1 | tee /tmp/seed_pool_test.log
   grep -o "seed_pool=[0-9]*" /tmp/seed_pool_test.log
   ```
2. Confirm `record_demonstrations.py` is completely unaffected — its
   scenarios should still show full positional variety (spot check a few
   `.npz` files' recorded starting positions, or just re-read the script to
   confirm it never passes `seed=` to `build_1v1_scenario`).
3. Confirm the `_from_ckpt` resume path (`--latest-pretrain` /
   `--from-pretrained`) doesn't break — since `seed_pool_size` is set
   post-construction by the trainer, resuming from any checkpoint should
   "just work" without needing to persist/restore pool size in the
   checkpoint dict (the schedule recomputes pool size from `progress`,
   which is derived from `self._total_steps` — already restored by
   `load_checkpoint()` — so no new checkpoint field is needed).

## 6. Follow-up / nice-to-haves (explicitly NOT required for v1)

- **Held-out eval seed range.** Reserve a disjoint seed range for
  `--eval`-style scripts (e.g. train pool draws from `{1..N}`, eval always
  uses `{1_000_000..1_000_500}` or similar) so memorization is actually
  *detectable* (compare win-rate on train-pool seeds vs eval-pool seeds) —
  this was flagged as "cheap insurance" in the original discussion but
  deliberately deferred. Would touch `evaluate.py` and the periodic
  `[eval vs rules]` block already inside `PPOTrainer.train()` (search for
  `_eval_rules_build` in `ppo_trainer.py` — that closure could gain a
  `seed=` kwarg pointing at the eval-only range).
- **Independent position/attribute/ball-state sub-seed pools.** Explicitly
  rejected for v1 (see section 1) in favor of simplicity — "one seed for the
  whole scenario". Revisit only if the single-seed-pool approach doesn't
  meaningfully reduce training noise in practice, since decomposing into
  multiple independent pools multiplies the effective variety at a given
  memorization-risk budget (e.g. 100 position seeds × 100 attribute seeds =
  10,000 effective combinations from just 200 total seed draws).
- **CLI override flag** (e.ed g. `--seed-pool-size N` on `train.py`) instead of
  config-only. Not needed for v1 since config edits are sufficient for
  experimentation; add only if this becomes a frequently-tuned knob.
- **Phase 2+ support.** `build_penalty_scenario` (phase 2) does not yet
  accept a `seed` kwarg — only wire up `seed_pool_size` for phase 2's
  `ScenarioEnv` once that builder is updated to match section 3.1's pattern.
  Until then, leave `curriculum.seed_pool_size_start/end` at `0` for any
  phase-2-specific config path if one gets split out later (currently phases
  share the same `curriculum` config block, so this isn't an issue yet, but
  will be if phase-specific curriculum sections get introduced).
- **Per-rollout eval win-rate split by train-vs-eval seed range** — a
  cheap diagnostic once section 6's held-out range exists: log
  `vs_rules_trainseed(N): W%` vs `vs_rules_evalseed(N): W%` side by side to
  directly see memorization if/when it happens, rather than inferring it
  indirectly from training curves.

## 7. Summary for whoever picks this up

Three files change (`scenarios.py`, `scenario_env.py`, `schedules.py`), one
config addition (`ai_config.json`), one small hook added to
`ppo_trainer.py`'s `train()` loop. Everything defaults to fully-disabled
(`seed_pool_size=0` / unseeded `random.Random()`), so this is a strictly
additive, backward-compatible change — no existing behavior changes unless
`curriculum.seed_pool_size_start/end` are explicitly set to nonzero values in
`ai_config.json`. Total estimated effort: under an hour of implementation,
plus test-writing per section 4. No demonstration-recording changes, no
CLI flag changes required for v1.
