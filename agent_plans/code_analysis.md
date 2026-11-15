> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

# Code Analysis — FootballCoach (non-AI code)

Scope: engine, entities, orders, actions, steering, rules_ai, ui, mathutils,
generation, config, tests and scripts. `src/footballcoach/ai/` is covered by
`ai_code_analysis.md`.

**Last reconciled 2026-11-15** against the code. Everything still listed under
"Open" was re-checked that day. Closed items are kept as one line each; the
long-form history of every finding is in git (this file before commit
`64b2644` and earlier).

**For future agents:** do mechanical, behaviour-neutral fixes without asking;
flag anything that changes behaviour or needs a judgment call. Verify engine
changes with `scripts/engine_trace_snapshot.py` and `scripts/golden_snapshot.py`
(compare against a run of the previous commit; a pure refactor gives 0
differences). Move items from Open to Closed as you go.

---

## Open

All of these were looked at on 2026-11-15 and deliberately deferred by the
user ("ignore for now"). None is a bug.

### Documentation drift risk
- **Worked numeric examples are written out in several places.**
  `kicking.py`'s `kick_sigma_rad` docstring, `possession.py`'s control-time
  docstrings, `tackling.py`'s `attempt_tackle` docstring and
  `match_tackles.py`'s `_effective_dribbling` docstring each repeat numbers
  that also appear in `physics.json` comments. After a retune, treat every
  docstring number as suspect until checked against `physics.json`. Fix would
  be picking one canonical place and pointing the other at it.
- **Balance target percentages are restated in about four places** (design
  docs, `physics.json` comments, the balance test assertions,
  `scripts/grid_search_kick_params.py`'s docstring). Same fix: make the test
  the source of truth and reference it.
- **The tick-order "ordering subtlety" comment in `engine/match.py` is
  duplicated near-verbatim in `engine/knowledge.md`.**

### Unvalidated approximations and close calls
- **`goalkeeping.predict_goal_line_crossing` ignores drag/Magnus** (same
  simplification as `kicking.solve_launch_pitch_rad`). No balance test
  measures how wrong this gets for heavily spun or curved shots.
- **Some balance bands sit right at the pass/fail edge** (e.g. precision 0.5
  corner aim 79.95% against a 50–80% band). A dedicated tuning pass, not a
  mechanical fix.
- **Saturation audit not done.** `tests/knowledge.md` records one balance
  test that saturated near 100% regardless of GK attributes. Nobody has
  checked whether any other balance test has the same "too easy" problem.

### Naming
- **`PlayerAttributes.average(value=0.5)`** (`entities/attributes.py`) has a
  test-sounding name and no docstring but is used in production
  (`ui/scenarios.py`). Rename it, or accept it as the production API and
  document it.

### UI
- **No tests for `ui/renderer.py` or `ui/camera.py`.** The emoji-font
  detection (probing render height) is platform-fragile and unchecked under
  `SDL_VIDEODRIVER=dummy`. Needs a decision on where UI tests live
  (`tests/ui/` does not exist).

### Known-failing tests (tracked as strict xfail since 2026-11-15)
Four tests fail on purpose until a retune or a gameplay decision. They are
marked `xfail(strict=True)`, so the suite stays green, they stay visible as
`x`, and they turn into a failure the moment one starts passing:
`test_pass_balance.py::test_good_player_succeeds_over_60_percent_at_40m`
(53.3% vs 60%), `::test_good_player_succeeds_over_30_percent_at_60m`
(29.1% vs 30%), `test_control_time_balance.py::test_goalkeeper_in_box_control_time_faster_for_high_balls`,
and `test_scenario_bugs.py::test_1v2_controller_issues_shoot_after_move`
(the defender tackles before the MoveOrder finishes). When one starts
passing, remove its marker.

---

## Closed

- **`engine/match.py` split (2026-11-15):** 1,412 → about 660 lines. Tackle
  resolution and GK tackle immunity moved to `match_tackles.py`;
  get-possession behaviour and intercept/lead-pass maths to
  `match_get_possession.py`, as stateless mixins (see `engine/knowledge.md`,
  "Module layout"). `_process_orders` and `_update_loose_ball_pickup` (about
  120 lines each) stay in `match.py` as tick-order glue.
- **Epsilon constants (2026-11-15, rename only):** `mathutils/tolerances.py`
  defines `GEOM_ZERO_EPS = 1e-9`, used by the 40 zero-length guards in
  engine/, entities/, orders.py and steering.py; `tests/unit/test_tolerances.py`
  rejects new bare `1e-9` literals there. A single canonical epsilon would
  change behaviour (1e-6 and 1e-9 both mean "zero length" at different sites;
  1e-12 is used on both lengths and squared lengths), so the 1e-6/1e-12 sites
  stay as they are on purpose.
- **`Vector3` docstring (2026-11-15):** no longer claims to wrap numpy. It is
  plain floats, which measured faster than numpy for single 3-vectors (add
  366 vs 310 ns, scale 363 vs 500, length 115 vs 927, dot 93 vs 347). The
  numbers are in the docstring.
- **`MoveOrder` defaults (2026-11-15):** `arrival_tolerance_m` (0.3) and
  `overshoot_timeout_s` (0.5) now come from `orders.json["move_order"]`, read
  at construction; explicit arguments still win and a missing key raises.
  Values unchanged; engine trace and golden identical.
  `tests/unit/test_move_order_config.py`.
- **`ball.possessed_by` is a player-id string:** kept as a string (avoids a
  circular reference). The cost was the lookup, now a dict built once at
  construction (`Match.player_by_id`), plus `Match.player_by_id_or_none`.
- **`resolve_ball_block_by_inactive_players` untested:** it now has
  `tests/unit/test_ball_block_by_inactive_players.py`.
- **Balance `N_TRIALS` drift:** every hard-target balance file now uses at
  least 1000 trials (1000–5000). The only lower count is the deliberate
  `n_trials=50` visual table in `test_pass_balance.py`, which asserts nothing.
- **`tests/ai_scenario/` single file:** it now has 30+ test files.
- **`tests/balance/results/latest_results.json` diff noise:** it is rewritten
  on every balance run, but the runs are seeded, so the file only changes
  when a balance number really changes. That diff is a useful signal, not
  noise.
