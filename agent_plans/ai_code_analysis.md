> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

# AI Code Analysis — FootballCoach (`src/footballcoach/ai/`)

Scope: `src/footballcoach/ai/**` plus the closely coupled `orders.py`,
`actions.py`, `rules_ai.py`, `steering.py` and the three knowledge docs. The
non-AI code is covered by `code_analysis.md`.

**Last reconciled 2026-11-15** against the code. Everything still listed under
"Open" was re-checked that day. Closed items are one line each, numbered as in
the original review (written 2026-08-13) so old references still resolve. The
long-form findings, code snippets and earlier status passes are in git (this
file before commit `64b2644` and earlier).

**For future agents:** verify refactors with `scripts/golden_snapshot.py`,
`scripts/env_trace_snapshot.py` and, for `_ppo_update`,
`scripts/ppo_update_trace_snapshot.py` (compare against a run of the previous
commit; a pure refactor gives 0 differences). Move items from Open to Closed
as you go.

---

## Open

- **BC-only continuation never runs once aux_coeff reaches 0 (found
  2026-11-15, not fixed).** `bc.bc_only_continuation_coeff`'s config comment
  says a fixed value keeps the BC-only loop going after `aux_coeff` has
  annealed to 0, but `_ppo_update` gates that loop on `has_bc`, which requires
  one of the annealed in-epoch coefficients to be > 0. Measured: with
  coefficient 0.7 and valid labels, the loop runs at aux 1e-9 and is skipped
  at aux 0.0. No effect on current runs (`bc_only_continuation_epochs` is 0).
  Fix would gate the continuation on `"bc_labels" in batch` and its own
  coefficient; that changes training whenever the feature is enabled.
- **`ai/physics_pretrain/` has never been reviewed.** About 9,000 lines,
  including `train_ball_dynamics.py` (4,329) and `train_player_dynamics.py`
  (2,977), added after the original review. Deferred by the user
  (2026-11-15): give it its own pass before refactoring it.
- **#2/#3 Large mixed-purpose modules.** `env/scenario_env.py` (about 1,020
  lines) still combines the step/reset API, reward glue, observation
  orchestration and BC-label wiring; `ppo/bc.py` (about 1,480 lines) combines
  the label schema, the loss functions and `phase1_labels()`'s counterfactual
  rules-AI query. Real but low urgency; no split planned.
- **#24 `ai_trainer_knowledge.md` not cross-checked.** It is a running notes
  file, so its "current status" claims should be checked against
  `ai_config.json` and recent `training_runs.md` entries before being
  trusted.

---

## Closed

### Structure
- **#1 `ppo_trainer.py` god object:** split into mixin modules (see
  `ai/knowledge.md`, "PPOTrainer module layout"), checked bit-identical with
  the golden snapshot. 11,982 → about 2,080 lines.
- **#1 `_ppo_update` as one ~2,000-line method (2026-11-15):** now a short
  orchestrator over named stages, with an `_UpdateContext`/`_UpdateStats`
  pair instead of ~60 locals, and the logging-only diagnostics moved to
  `ppo_update_diagnostics.py`. Verified bit-identical with
  `scripts/ppo_update_trace_snapshot.py` (0 differences over 15 variant runs;
  8 injected mutations all detected, including a reordered network call caught
  only through value dropout's RNG use) and the golden snapshot. Pieces
  unit-tested in `tests/ai_unit/test_ppo_update_parts.py`.
- **#4 Inconsistent package granularity:** new code has been added as
  focused modules, and the trainer split removed the main outlier.
- **#7 Hand-copied trainee/secondary state in `scenario_env.py`
  (2026-11-15):** almost all of it was write-only and was deleted. The one
  live piece, the trainee's previous-tick possession flag for the box early
  exit, is a plain bool pinned by `tests/ai_unit/test_scenario_env_early_exit.py`.
  Only the EMA pair (`_ema`/`_sec_ema`) remains paired.
- **Sampling-function "merge" (2026-11-15):** nothing to merge. Production
  uses one implementation (`_sample_action` is a batch of 1 through
  `_sample_action_from_heads_batch`); `_sample_action_from_heads` survives
  only as the per-row test oracle, and its docstring says so.

### Duplicate logic
- **#5 `bc.py`'s `_floor_bce` vs `_bce` (2026-11-15, not a duplicate):**
  `_floor_bce` is the minimum achievable loss for a smoothed target
  (subtracted to report excess loss); `_bce` is the actual loss. They share
  only the one-line hard-label `pos_weight` weighting.
- **#6/#25 Silent `except (KeyError, AttributeError)` lookups (2026-11-15):**
  the pure single-lookup sites use `Match.player_by_id_or_none`. The four
  remaining live sites (`ScenarioEnv._ball_dist_to_opponent_box` returning
  1.0, `_ball_dist_for_player` returning 0.0, and both `bc.py` label helpers
  returning an invalid label) now raise instead. These values feed the reward
  and BC labels. A counting probe over the env trace and the full AI test
  suite found no caller that ever hit the fallbacks, so behaviour is
  unchanged (env trace 0 differences, golden identical).
  `tests/ai_unit/test_fail_loud_lookups.py`. The last one was in the dead
  `curriculum/opponent_pool.py`, deleted 2026-11-15 (nothing imported it;
  `OpponentPool` and `apply_rules_based_opponent` had no callers).
- **#8/#12/#15 Head-name tables:** gating has one table (`_HEAD_TO_ACTION`,
  whose order is the tie-break order), `gating.DECISION_HEAD_NAMES` is the
  public list of `decision_probs` keys, and `set_decision_override` rejects
  unknown names. `tests/ai_unit/test_decision_head_names.py` writes down the
  correspondence with the other three spellings (network attribute,
  `HEAD_LP_KEYS`, `DecisionAction` field) and checks every table against it.
- **#10 `augment.py` column constants:** imported from `bc.py`'s `_I_*`.

### Magic numbers
- **#9 Pitch constants in sampling:** first moved to `physics.json`, then removed entirely with the `decision_physical` chain (#11), their only user.
- **#11 `decision_physical` squashing constants (2026-11-15):** the whole
  `decision_physical` chain had no production reader and was removed with its
  `1.0`/`3.0`/`9.5` literals.

### Risky logic
- **#13 Kick power fallback to 0.85:** now raises `ValueError`; tested.
- **#14 Kick direction fallback to +x:** now raises `ValueError` (NaN used
  to reach it silently); tested.
- **#16 Tackle illegal reasons:** `tackle_no_carrier` and
  `tackle_own_team_carrier` are distinct; tested.
- **#17 `encode_slot_player_ids`:** had no callers; deleted with
  `_resolve_target_player`.
- **#19 `NeuralPlayerAI` role:** confirmed accurate; `act()` really does
  drive action selection every decision interval.
- **#26 Broad `except Exception` in the trainer:** every site logs the
  exception; not a bug.

### Documentation
- **#18/#22 BC label layout drift:** `bc.py`'s docstring, `ai/knowledge.md`
  and `BC_LABEL_DIM = 27` agree, and `tests/ai_unit/test_bc.py::TestLabelLayout`
  now checks the `_I_*`/`BC_LABEL_DIM` consistency.
- **#20/#21 `to_orders.py` references:** `knowledge.md` describes
  `apply_nn_action.apply_action_to_player` setting physics fields directly,
  with no Orders for neural players.
- **#23 "Only `MoveOrder` uses repulsion":** verified accurate.

### Tests
- **#27 `apply_nn_action.py` edge cases:** kick power, kick direction
  (None/zero/NaN) and both tackle illegal reasons are tested.
- **#28 Curriculum:** `test_curriculum_opponent_ratios.py` and
  `test_curriculum_phases.py` (including `PHASES_BY_ID`).
- **#29 Seeded eval determinism:** `tests/ai_scenario/test_seeded_eval_determinism.py`
  (same seed reproduces exactly; side swap; different seeds differ; stochastic
  policy only after reseeding torch).
- **#30 Entity encoder permutation invariance:**
  `test_entity_encoder_permutation.py`, committed.
