# tests/

Three tiers of tests, per the project's design requirements:

## `unit/`

Fast, deterministic tests of individual functions/modules in isolation
(movement formulas, ball physics steps, kicking error math, etc.). No RNG
dependency where avoidable; where RNG is involved (e.g. attribute
generation), a seeded generator is used so results are reproducible.

## `scenario/`

End-to-end tests through the full `Match` loop, always at
`rng_reduction=1.0` so outcomes are fully deterministic (no flaky tests).
These validate that the pieces wire together correctly - e.g. "a player
given a Move order actually arrives", "a penalty kicked dead centre with
zero error scores and the goal is recorded", "a tackle with a big skill gap
wins deterministically".

`test_scenario_loop.py` tests `ScenarioLoop` headlessly without pygame,
driving `loop.step()` directly.  It parametrises over all `SCENARIOS` so new
scenarios are automatically covered.  All parametrised tests pass
`linger_s=0.0` so they don't have to budget for the 3-second UI linger;
dedicated linger tests (`test_linger_delays_trial_end`, `test_oob_linger_is_half_of_full_linger`,
etc.) use an explicit non-zero `linger_s` and assert the half-linger for OOB
events and full linger for goals/saves.

`test_save_order.py` tests the goalkeeper `SaveOrder` logic via specific
edge-cases at `rng_reduction=1.0` (deterministic): no overshoot past the
far post, no tunneling on fast close-range shots, no drift when already
on the crossing point, and confirmation that a slow keeper genuinely cannot
cover 7m in time for an 8m shot (so "always saved" tests aren't vacuously
true).  These complement `tests/balance/test_save_balance.py`'s statistical
coverage.

**Coordinate convention in tests**: x-coordinates are always from pitch
centre (origin), not from a goal line.  Use `-(pitch.half_length - dist_m)`
to place a shooter `dist_m` from the goal.  Hardcoding values like `-22.0`
has repeatedly caused mismatched expectations about shot distance — always
derive from `pitch.half_length`.

## `balance/`

Statistical tests at the *default* game setting (`rng_reduction=0.3`), run
over many trials (typically 1000-5000) and asserting the *aggregate*
outcome falls within a designer-specified band - e.g. "a 0.8 tackling player
beats a 0.6 dribbling player 70-90% of the time". These exist to validate
and tune game balance, not just correctness.

Balance tests that validate a new order type (e.g. `ShootOrder`) must reach
the same statistical targets as the equivalent `KickOrder` test, since both
call the same `kick_ball` code path. The penalty tests do exactly this:
`test_penalty_balance.py` runs each scenario with both `KickOrder` and
`ShootOrder`, asserting identical score-rate bands for both.

**Every balance test must report full statistics, not just pass/fail.** Use
the `balance_recorder` fixture (defined in `tests/conftest.py`):

```python
def test_something(balance_recorder):
    stats = {"n_trials": 1000, "wins": 812, "win_rate_pct": 81.2}
    balance_recorder.report("my_test_name", stats)
    assert 70.0 < stats["win_rate_pct"] < 90.0
```

`report()` prints the stats immediately (visible with `pytest -s`) and
accumulates them; at the end of the test session, everything is written to
`tests/balance/results/latest_results.json` for later inspection/diffing
without needing to re-run pytest. Many balance tests aren't pass/fail
against a single hard target at all - some just report a *table* (e.g.
control-time vs. ball height) with a loose monotonicity assertion, since the
design brief didn't specify exact numbers for every mechanic. Use your
judgement: if the user gave an explicit percentage/range target, assert it
tightly; if not, assert only that the *shape* of the behaviour is sane
(monotonic, right sign, etc.) and let the printed/JSON table be the primary
tuning aid.

**Designing a balance test that actually differentiates:** when comparing a
"good" vs "bad" player/scenario, make sure the scenario is hard enough that
a bad player/goalkeeper can actually fail some of the time and a good one
can actually succeed most of the time. A too-easy scenario (e.g. a
goalkeeper already standing where the shot is aimed) saturates near 100%
for both ends of the attribute range, making the comparison meaningless -
`tests/balance/test_save_balance.py` hit exactly this while being written
(a naively-random shot placement + GK start position both saved ~97%
regardless of GK attributes) before being tightened to force genuine
travel/reaction requirements (GK pinned to one post, shot aimed at the
other). If a comparison test is failing because both sides score
suspiciously similarly, that's usually the scenario being too easy, not a
balance bug.

### Phase G unit test files

- `test_ball_physics.py` — extended with four `just_bounced_timer_s` tests:
  timer set on real bounce, not set on settling contact, decays to zero over
  the expected number of ticks, and resets to the full duration on each
  subsequent bounce.
- `test_gamelog.py` — `GameLog` ring buffer (add, filter by level, deque
  eviction, timestamps) and tackle-logging plumbing (win/loss INFO entries,
  DEBUG roll breakdown, GK-box auto-fail distinct message, `log_callback=None`
  no-error regression).
- `test_goal_linger.py` — `Match.goal_linger_s`: immediate-reset regression,
  linger delays reset, no double-goals during linger, countdown rate.

## `conftest.py`

Shared fixtures:
- `pitch`, `seeded_rng` - trivial fixtures for common test setup.
- `make_player(...)` - builds a `Player` with all attributes defaulted to a
  single value, with individual overrides (e.g.
  `make_player(tackling=0.8, dribbling=0.5)`). Prefer this over constructing
  `PlayerAttributes` by hand in tests.
- `BalanceResultRecorder` / `balance_recorder` fixture - see above.

## `ai_unit/`

Fast, deterministic tests for the AI package (`src/footballcoach/ai/`).
These require `torch` installed (`uv sync --group ai`), but do NOT run any
training - just forward passes with random inputs and arithmetic checks.

**Key test files:**
- `test_obs_schema.py` — dimension constants (PLAYER_FEATURE_DIM=25,
  BALL_FEATURE_DIM=12, GLOBAL_FEATURE_DIM=11, MAX_OTHER_PLAYERS=21),
  to_array shapes, dtypes, field ordering.
- `test_obs_encoder.py` — position normalization, flag correctness,
  slot shuffling permutation invariance, padded-slot all-zero invariant,
  no-NaN guarantee, score-diff team perspective, log1p time normalization.
- `test_gae.py` — GAE(lambda) hand-computed reference cases (three-step
  episode with explicit expected values), Monte Carlo equivalence,
  episode-boundary isolation (done=1 must zero cross-boundary carry),
  last_value bootstrapping, RolloutBuffer housekeeping.
- `test_distributions.py` — MaskedCategorical masked slots get EXACTLY zero
  probability; unmasked probs sum to 1; SquashedNormalHead stays within
  physical bounds; DirectionHead output is a unit vector; no NaN anywhere.
- `test_gating.py` — winner-take-all rule, 0.5 does not fire (strictly >),
  highest-prob head wins, target slot propagation, execution pass-through.
- `test_to_orders.py` — correct order type assigned for each action, illegal
  preconditions detected (shoot without possession, tackle while inactive,
  tackle own teammate, no valid target), NONE with/without active order.
- `test_reward.py` — per-component arithmetic for phase1_reward and
  phase2_reward, EMAFilter slow/fast alpha, post-goal window expiry, reset.
- `test_networks.py` — DecisionNetwork and ExecutionNetwork output shapes for
  every head, no NaN/Inf, get_possession >= tackle constraint,
  flatten_decision_heads dim consistency with execution_net.decision_mlp.

**conftest.py** in `tests/ai_unit/` provides:
- `solo_match` — single player, loose ball
- `duel_match` — 1v1, p1 (LEFT) has ball, p2 (RIGHT) has none
- `gk_match` — GK (LEFT) vs attacker (RIGHT) with ball
- `standard_pitch` — standard Pitch fixture

## Running a subset

```bash
uv run pytest tests/unit                 # fast engine unit tests
uv run pytest tests/scenario             # deterministic end-to-end
uv run pytest tests/balance -s           # statistical, prints tables
uv run pytest tests/balance -s -k tackling  # just one balance area
uv run pytest tests/ai_unit              # AI unit tests (requires torch)
uv run pytest tests/ -q                  # everything
```
