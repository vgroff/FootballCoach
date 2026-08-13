> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

# AI Code Analysis — Critique & Improvement Plan

Scope: `src/footballcoach/ai/**`, plus the closely-coupled sibling files
`orders.py`, `actions.py`, `rules_ai.py`, `steering.py`, and the three
knowledge docs (`src/footballcoach/knowledge.md`, `src/footballcoach/ai/knowledge.md`,
`ai_trainer_knowledge.md`). This is a critique/planning document — it does
not itself change code.

---

## Part 1 — Summary of issues (short form)

### Structural / size
1. **`ai/ppo/ppo_trainer.py` is 4263 lines** — by far the largest file in the
   repo (3x the next largest, `scenario_env.py` at 920). It mixes: optimizer/
   param-group setup, the full PPO update math, BC pretraining (Phase 0/1/2/3),
   value-only pretraining, parallel rollout orchestration, evaluation harness
   glue (`_eval_vs_rules`/`_eval_vs_immobile`/`_eval_vs_opponent_type`),
   checkpoint save/load, and extensive per-rollout logging/formatting helpers.
   This one class (`PPOTrainer`) has ~30 methods and does the job of 5-6
   separate collaborators. See Part 2 for a proposed split.
2. **`ai/env/scenario_env.py` (920 lines)** similarly conflates: Gym-like
   step/reset API, reward computation glue, possession-transition state
   machine, observation encoding orchestration, and BC-label wiring into one
   class.
3. **`ai/ppo/bc.py` (953 lines)** mixes label schema/packing (`BCLabel`),
   two different loss-computation code paths (`bc_loss_from_tensor` operating
   on pre-decided tensors, and a second near-duplicate path around line 490
   operating on live decision/execution heads), and `phase1_labels()`
   generation logic that reaches into engine internals. These are three
   fairly separable concerns.
4. Several very small one-purpose files exist alongside a few "does
   everything" files — package granularity is inconsistent (`models/
   value_side_channel.py` is 99 lines and single-purpose; `ppo_trainer.py`
   is everything-purpose). A more even split would improve navigability.

### Duplicate / near-duplicate logic
5. `bc.py` has two BCE-loss helper functions (`_floor_bce` used inside
   `bc_loss_from_tensor`, and a second closure `_bce` inside the "live heads"
   loss function) that do almost the same job — computing a smoothed BCE term
   with optional `pos_weight` — but are declared independently and could
   silently drift apart (they already differ slightly in signature). Same
   `dec_label_smoothing`/`exec_label_smoothing` calling pattern is duplicated
   across the two.
6. `apply_nn_action.py::_resolve_target_player` and `opponent_pool.py`
   both catch `(KeyError, AttributeError)` around `match.player_by_id(...)`
   lookups — a `Match.player_by_id_or_none()` convenience method (or a
   shared helper) would remove this repeated try/except boilerplate from at
   least 3 call sites.
7. `ai/knowledge.md` documents that the trainee and secondary-player reward
   code paths were unified into `_compute_phase1_reward_for_player()` after a
   documented drift bug (see `/memories/repo/rules.md` "Reward parity"
   section) — good, but this pattern (hand-copied trainee/secondary blocks)
   recurs elsewhere in `scenario_env.py` (e.g. possession-transition state:
   `_trainee_pending_loss` vs `_sec_pending_loss` dicts, `_trainee_had_
   possession_last_step` vs `_sec_had_possession_last_step`). These are
   currently *not* unified into one generic per-player state object — worth
   auditing for the same class of drift risk the reward code already hit.
8. `_HEAD_ORDER` (gating.py) and `_HEAD_TO_ACTION` (gating.py) are two
   separate hand-maintained lists/dicts covering the same 7 head names —
   easy to add a head to one and forget the other silently (no runtime
   assertion that their key-sets match).

### Magic numbers / string literals
9. `pitch_hl = 52.5` / `pitch_hw = 34.0` hardcoded directly inside
   `PPOTrainer._sample_action()` (ppo_trainer.py:2724-2725) with an explicit
   `# TODO: get from obs if pitch varies` comment — this is a known,
   self-acknowledged wart. These constants should come from the pitch/global
   feature encoding or `config/` rather than being duplicated inline; if the
   pitch dimensions in `physics.json`/pitch config ever change, this silently
   goes stale with no error.
10. `augment.py` explicitly documents (in its own comment) that `_BC_DIR_X_COL`
    etc. are **hardcoded integers duplicating `bc.py`'s `_I_*` constants**,
    with a comment admitting "If a FUTURE bc.py change ever inserts/reorders
    fields... these constants will silently go stale." This is a real,
    acknowledged landmine — `bc.py`'s `_I_*` constants are already exported
    at module level and should simply be imported into `augment.py` instead
    of re-declared. There is no technical reason for the duplication (no
    circular import risk visible — `bc.py` doesn't import from `obs/`).
11. Magic tuning constants scattered through `ppo_trainer.py`'s physical
    squashing logic — e.g. `mv_size_phys = 1.0 + 3.0 * torch.sigmoid(...)`
    (implicit `[1,4]` metre range) and `mv_speed_phys = ... * 9.5` (implicit
    v_top) are inline literals rather than named config constants, unlike
    almost every other tunable in this codebase which lives in
    `ai_config.json`. `9.5` in particular duplicates whatever `v_top` actually
    resolves to elsewhere (attribute-derived), so a change to top-speed
    scaling would silently desync this squashing range from the actual
    physical arrival-speed range used elsewhere.
12. Head name literals (`"shoot"`, `"pass_"`, `"move"`, `"tackle"`,
    `"get_possession"`, `"mark"`, `"hold_position"`) are repeated as raw
    strings across `gating.py`, `ppo_trainer.py` (`decision_probs` dict
    construction), and presumably BC/obs code — an `Enum`/`Literal` type or a
    single shared tuple constant would remove this repeated stringly-typed
    surface and catch typos at import time instead of silently returning
    `0.0` from `.get(head_name, 0.0)`.

### Suspicious / risky logic
13. `apply_nn_action.py::apply_action_to_player` — the kick power fallback:
    ```python
    player.kick_with_direction(
        match,
        direction_3d,
        float(gating.kick_power_fraction) if gating.kick_power_fraction > 0 else 0.85,
        ...
    )
    ```
    silently substitutes a hardcoded `0.85` power fraction whenever the
    network outputs `kick_power_fraction <= 0`. This means a legitimately
    low-power kick output (e.g. `0.0` for a very soft touch, or any negative
    value from squashing edge cases) is silently overridden to near-max
    power rather than clamped to a small positive epsilon. This could bias
    training data / early-PPO kicks toward artificially hard kicks whenever
    the policy is still exploring near 0. Worth clamping to `max(eps, x)`
    instead of using the boolean threshold + fixed fallback, or asserting
    the squashing function can never legitimately return 0.
14. Same function's kick direction fallback (`direction_3d = Vector3(1.0, 0.0,
    0.0)` labeled "safe fallback, should not occur") — if this ever *does*
    occur (e.g. NaN direction from a numerically unstable policy early in
    training), it kicks the ball at a fixed absolute-pitch-x direction
    regardless of which goal the player is attacking, which is actually the
    *wrong* direction roughly half the time. Given the codebase's own
    documented history of squashing-edge-case bugs (see NaN/degenerate
    vectors called out elsewhere), this deserves at minimum a log/counter so
    silent occurrences during training are visible, not just a comment
    asserting it shouldn't happen.
15. `gating.py::select_action`'s docstring/comment says "tie-breaking order:
    shoot > pass > move > tackle > get_possession > mark > hold_position —
    the simplest resolution; adjust if needed" but the actual comparison
    (`if p > best_prob`) is a **strict greater-than** scan through
    `_HEAD_ORDER` — this does NOT implement "highest probability wins with
    ties broken by priority order" as the module docstring claims further up
    ("select the single highest-probability head"). It actually implements
    "first head in priority order whose prob exceeds the current best,
    scanning in priority order" — which for genuinely non-monotonic
    probabilities (e.g. shoot=0.6, pass=0.9) correctly finds pass (0.9 >
    0.6 > 0.5), so in the current code it does coincide with "true max" *only
    because it's a plain running-max scan*, not because of any tie-breaking
    logic. But if two heads are EXACTLY tied (e.g. both 0.7), whichever comes
    first in `_HEAD_ORDER` wins, which is fine — the real problem is the
    comment says "tie-breaking order" for something that is actually just
    "iteration order of a max-scan", i.e. the comment overclaims a
    tie-breaking *feature* that only exists as an accidental side effect of
    iteration order. Low severity, but worth simplifying the comment to be
    accurate, since a future reader could reasonably assume there's an
    explicit equal-probability tie-break policy being enforced when there
    isn't (e.g. two heads at literally 0.7000000 vs 0.7000001 due to float
    noise would non-deterministically flip on tiny model differences, not on
    anything semantically about priority).
16. `apply_nn_action.py::apply_action_to_player`'s tackle-legality check:
    ```python
    if match.ball_carrier() is None or match.ball_carrier().team == player.team:
        return OrderTranslationResult(illegal_action=True, illegal_reason="tackle_no_carrier")
    ```
    Two genuinely distinct illegal conditions ("no ball carrier at all" vs
    "carrier is a teammate") are collapsed into one `illegal_reason` string
    ("tackle_no_carrier") that's misleading for the second case (there IS a
    carrier, it's just on your own team). If the reward function or
    diagnostics ever branch on `illegal_reason` strings, this will
    misclassify "own-team-tackle-attempt" events as "no carrier" events. Easy
    win: split into two distinct reason strings.
17. `apply_nn_action.py::encode_slot_player_ids` silently returns a list with
    `None` for any player not present in `other_players` but does not
    validate that `slot_indices` and `other_players` are the same length
    (`zip()` silently truncates to the shorter of the two on a length
    mismatch) — a caller passing mismatched lists gets silently wrong (but
    not crashing) slot mappings rather than an assertion error.
18. `bc.py`'s BC_LABEL_DIM history (documented in `ai/knowledge.md` as
    "15→16→17", but the code comment at the top of `bc.py` says "24" while
    `BC_LABEL_DIM = 25` in the actual code, and adds a further `_I_KICK_DIR_Z
    = 24` field noted as "non-contiguous with x/y"). This is a live
    **documentation/code mismatch** — `ai/knowledge.md`'s BC label table
    (reproduced in Part 1's own doc section above) stops at index 23 and
    says "BC_LABEL_DIM = 24", but the actual `bc.py` module docstring says
    "17 floats" in one place and defines 25 slots with `BC_LABEL_DIM = 25`
    elsewhere, and index 24 (`kick_dir_z`) is not mentioned in
    `ai/knowledge.md`'s table at all. This is exactly the kind of drift the
    file's own top-of-doc warning says has already happened twice — it's
    now happened a third time and wasn't caught. See Part 2 for exact quotes
    and a proposed fix (single source of truth / auto-generated table).
19. `ScenarioEnv`'s docstring says "For neural players, ScenarioEnv assigns
    NeuralPlayerAI on reset()" but the actual per-tick action application
    goes through `apply_action_to_player` (a free function) called from
    `PPOTrainer`/rollout code, not obviously through a `NeuralPlayerAI.act()`
    method analogous to how `Phase1RulesAI.act()` works — worth verifying
    the `NeuralPlayerAI` class actually exists and is a thin no-op wrapper
    used only for engine bookkeeping (e.g. does `Match.step()` still call
    `player.ai.act()` for neural players, and if so what does that do, given
    the real action selection happens externally via `_sample_action` /
    `apply_action_to_player`?). If `NeuralPlayerAI.act()` is a no-op stub,
    the docstring is misleading about "who" drives the player each tick;
    this should be clarified explicitly in the docstring instead of implying
    parity with the rules-AI pattern.

### Documentation inconsistencies between the three knowledge docs
20. `src/footballcoach/knowledge.md` states, in the "Neural network / Orders
    boundary" section: *"The **execution neural network drives movement**
    via `move_direction`... `to_orders.py::apply_movement_to_player` converts
    this to a far-target `MoveOrder`..."* — but the actual file implementing
    this is `ai/action/apply_nn_action.py::apply_action_to_player`, which
    (per its own docstring and the code read above) does **NOT** construct a
    `MoveOrder` at all — it sets `player.desired_direction`/
    `player.desired_speed_mode` directly, exactly matching the "fix" that
    `/memories/repo/rules.md`'s CRITICAL ARCHITECTURE RULE demands. There is
    no `to_orders.py` file in the current tree (confirmed via `list_dir` —
    only `apply_nn_action.py`, `gating.py`, `schema.py`, `distributions.py`
    exist under `ai/action/`). So `src/footballcoach/knowledge.md` is
    **stale in two ways at once**: (a) it references a module
    (`to_orders.py`/`apply_movement_to_player`) that no longer exists under
    that name, and (b) it describes the OLD, admittedly-buggy behaviour
    (wrapping execution output in a `MoveOrder`) that the repo's own memory
    file says was a bug that "MUST BE FIXED" — and based on the actual
    `apply_nn_action.py` code, it evidently HAS been fixed, but
    `knowledge.md` was never updated to reflect the fix. This is a
    first-class doc/code mismatch that will actively mislead any future
    reader (or agent) who trusts `knowledge.md` over the actual code, and is
    exactly the failure mode the "Corollary" paragraph immediately below it
    warns against for OTHER code. Fix: rewrite that whole section to
    reference `ai/action/apply_nn_action.py::apply_action_to_player` and
    describe the direct-physics-field-setting behaviour, not a `MoveOrder`
    wrapper.
21. Relatedly, `src/footballcoach/knowledge.md`'s "Neural network / Orders
    boundary" section bullet list still says: *"**High-level decision head
    fires** — when the neural network's shoot/pass/tackle/get_possession/
    mark Bernoulli heads fire, `to_orders.py` calls the corresponding player
    method, which sets the Order"* — this is contradicted by
    `apply_nn_action.py`'s own module docstring: *"No Orders are created
    here. Orders exist for the rules-based AI only."* If decision heads
    firing (e.g. `shoot`) genuinely never construct a `ShootOrder` for a
    neural player (matching `apply_nn_action.py`'s stated design), then
    `knowledge.md` describes behaviour that contradicts the actual,
    already-fixed implementation entirely — not just a file-rename issue.
    This needs a decision: is `knowledge.md`'s "decision heads still issue
    real Orders on fire" description ever partially true anywhere (e.g. do
    `shoot`/`pass_`/`tackle` gating results reach real Order construction
    somewhere else that this review didn't examine, e.g. inside
    `ppo_trainer.py`'s rollout loop) or is `knowledge.md` simply wrong here
    too? Recommend explicitly re-auditing `_sample_action`/`train()`'s
    rollout step in `ppo_trainer.py` for any remaining
    `player.shoot(...)`/`player.tackle_player(...)` style calls driven by
    gating output — if none exist, delete this bullet from `knowledge.md`
    entirely rather than patch it, since per the "Neural network / Orders
    boundary (IMPORTANT)" heading itself and `ai/knowledge.md`'s own
    "!!!! CRITICAL" banner, Orders should ONLY ever be rules-AI/BC-teacher
    constructs for a neural-controlled player.
22. `ai/knowledge.md`'s BC label table (index 0-23) does not mention index 24
    (`kick_dir_z`, per `bc.py`'s `_I_KICK_DIR_Z = 24`) at all, and states
    `BC_LABEL_DIM = 24` in prose ("do not let this count drift out of sync,
    it has already changed twice: 15→16 ... 16→17") while the actual code
    constant is `BC_LABEL_DIM = 25`. Combined with finding #18 above, this
    means BOTH knowledge docs that discuss the BC label layout are stale
    against the current code by (at least) one field. This is a good
    candidate for either (a) a doc comment that says "count = N, see bc.py's
    own module docstring, do not duplicate the table" instead of maintaining
    two independent copies of the same table in two files, or (b) generating
    the table from `_I_*` constants directly (e.g. a small script/test that
    asserts the doc table's ordinal count matches `BC_LABEL_DIM`).
23. `steering.py`'s knowledge-doc section (`src/footballcoach/knowledge.md`)
    states repulsion is "Only `MoveOrder` uses repulsion... confirmed out of
    scope" for ChaseTackleOrder/GetPossessionOrder/SaveOrder — this claim is
    stated with confidence but this review did not verify it against
    `engine/match.py`'s actual `_process_orders` dispatch; given how many
    other "confirmed"/"deliberate" design claims in these docs have already
    been found stale above, this is flagged as a claim worth spot-checking
    rather than trusting outright (not a confirmed bug, just a
    "verify-before-trusting" flag per the task's instruction to flag
    unconvincing-even-if-documented claims).
24. `ai_trainer_knowledge.md` (repo root) was read as part of this review's
    required background, but this review focused code-reading effort on
    `ai/` itself per the user's request; any reward-shaping/PPO-hyperparameter
    narrative claims in `ai_trainer_knowledge.md` should be cross-checked
    against the current `ai_config.json` values and `reward.py` docstring in
    a follow-up pass — the sheer number of already-found stale claims in the
    other two docs (see #18-#22) suggests `ai_trainer_knowledge.md` (which is
    a running training log/notes file, likely to go stale fastest, per its
    own nature as a log) is also likely to contain at least a few outdated
    "current status" claims by now and shouldn't be taken as ground truth
    without cross-checking against `ai_config.json` and recent
    `training_runs.md` entries.

### Error handling / edge cases
25. Several `except (KeyError, AttributeError):` blocks (apply_nn_action.py:117,
    opponent_pool.py:83, scenario_env.py:823/835, bc.py:210) silently return
    `None`/default values on lookup failure with **no logging at all** — if
    `match.player_by_id()` starts raising a different exception type (e.g.
    from a future refactor), these will let it propagate uncaught (acceptable
    fail-fast), but if it silently returns wrong-but-valid data due to a
    stale `player_id` reference, there is no telemetry to notice a systematic
    problem (e.g. a slot-mapping bug causing every lookup in an episode to
    silently fail and always fall back to `None`). At minimum a
    `log.debug()` on the except branch during training (not eval, to avoid
    log spam) would make this diagnosable.
26. `ppo_trainer.py:1338` and `train.py:378,526` catch bare `except Exception
    as _e:` — broad catches around what look like diagnostic/logging code
    paths (based on variable naming `_diag_exc`) are probably intentional
    ("never let a diagnostic crash training") but should be confirmed to at
    minimum log the exception (worth a quick check that `log.exception(...)`
    or similar is actually called in each of these three blocks, not just
    swallowed silently — not verified in this pass).

### Testing gaps
27. `tests/ai_unit/` covers: BC label/tensor loss (`test_bc.py`), dataset
    loading (`test_demonstration_dataset.py`), distributions, GAE, gating,
    networks, obs encoder/schema, reward, ai-type side channel, frozen-head
    KL masking, and scenario-env reward wiring — a solid unit-level spread.
    However, there is **no dedicated unit test file for
    `apply_nn_action.py`** (`apply_action_to_player`, `encode_slot_player_ids`,
    `_resolve_target_player`) despite it being explicitly called out (in its
    own docstring) as *"the ONLY point where neural network outputs touch
    engine state"* — arguably the single highest-value module to have direct
    unit coverage for (kick fallback behaviour from finding #13, tackle
    legality branching from finding #16, slot-id mapping from finding #17
    are all untested edge cases as far as this review's `list_dir` could
    tell). `test_apply_nn_action.py` DOES exist under `tests/ai_unit/` —
    worth double-checking it actually covers the fallback-power/fallback-
    direction/illegal-reason-string edge cases flagged above rather than
    only the happy path, since the file's existence doesn't guarantee those
    specific edge cases are exercised.
28. `ai/curriculum/` (phases.py, envs.py, opponent_pool.py) has **no
    corresponding `tests/ai_unit/test_curriculum*.py` or
    `test_opponent_pool.py`** file visible in the `tests/ai_unit/` listing —
    curriculum-phase construction and opponent-pool sampling logic (both of
    which directly gate what data the network trains on) appear to have zero
    dedicated unit tests, only indirect coverage via
    `tests/ai_scenario/test_pretrain_combined_smoke.py`-style
    integration/smoke tests (if those even exercise curriculum selection
    directly).
29. `ai/eval/seeded_eval.py` (286 lines) similarly has no obviously
    corresponding unit test file in either `tests/ai_unit/` or
    `tests/ai_scenario/` listings — seeded-evaluation determinism (a
    documented open task item — "The evaluation steps should also use seeds
    so that they're always the same!" per the user's own notes) is exactly
    the kind of logic that most benefits from a unit test asserting "same
    seed twice → identical episode outcome", and none appears to exist yet.
30. No test file targets `ai/models/entity_encoder.py`'s permutation-
    invariance property directly at the *encoder* level (only
    `test_ai_type_side_channel.py`'s
    `TestValueSideChannelPermutationInvariance` tests the value-only side
    channel's invariance, per `ai/knowledge.md`). Given how much of this
    codebase's design rests on permutation invariance being correct (it's
    called out as a core architectural guarantee multiple times in the
    docs), a direct test asserting `entity_encoder(...)` output is invariant
    under a slot permutation (not just the side-channel wrapper) would be a
    valuable, cheap addition and a natural regression guard for any future
    encoder refactor.

---

## Part 2 — Detailed discussion, code snippets, and proposed fixes

### 2.1 `ppo_trainer.py` — the 4263-line god-object

**Evidence.** A `grep` for top-level method definitions shows `PPOTrainer`
owns (non-exhaustively): `__init__`, `_value_heads`, `_get_value_pretrain_freeze_params`,
`set_frozen_heads`, `train`, `_log_rollout_summary`, `_train_parallel`,
`_eval_vs_rules`, `_eval_vs_immobile`, `_eval_vs_opponent_type`,
`pretrain_combined`, `_collect_value_pretrain_rollout`, `pretrain_value`,
`_move_dir_head`, `_kick_dir_head`, `_per_head_new_log_probs`,
`_sample_action`, `_get_value`, `_compute_log_prob`, `_ppo_update`,
`_recompute_log_prob`, `_compute_entropy`, `_rotate_log_file`,
`_save_checkpoint`, `_save_checkpoint_to`, `load_checkpoint`, `from_config`,
`load_for_inference`. That's roughly 8 orthogonal responsibility clusters
living in one class:

1. **Optimizer/param-group construction & freezing** (`__init__`,
   `_get_value_pretrain_freeze_params`, `set_frozen_heads`).
2. **Action sampling / log-prob math** (`_sample_action`, `_get_value`,
   `_compute_log_prob`, `_recompute_log_prob`, `_compute_entropy`,
   `_move_dir_head`, `_kick_dir_head`, `_per_head_new_log_probs`).
3. **The PPO update step itself** (`_ppo_update`).
4. **BC/value pretraining orchestration** (`pretrain_combined`,
   `pretrain_value`, `_collect_value_pretrain_rollout`).
5. **Rollout collection / training loop / parallel dispatch** (`train`,
   `_train_parallel`).
6. **Evaluation harness glue** (`_eval_vs_rules`, `_eval_vs_immobile`,
   `_eval_vs_opponent_type`) — this arguably belongs entirely in `ai/eval/`,
   next to `seeded_eval.py`, not inside the trainer.
7. **Logging/formatting** (`_log_rollout_summary`, plus several nested
   formatting closures like `_fmt`/`_fmtu`/`_sigma_deg`/`_angular_dmean_deg`).
8. **Checkpoint I/O** (`_save_checkpoint`, `_save_checkpoint_to`,
   `load_checkpoint`, `from_config`, `load_for_inference`).

**Why this matters beyond aesthetics:** this file's size makes it very easy
for exactly the kind of drift bugs already documented in
`/memories/repo/rules.md` to keep happening (e.g. the trainee/secondary
reward-parity bug, the `pretrain_value()`-vs-`pretrain_combined()` duplication
bug that was already fixed once by consolidating into a shared function).
A single 4000+ line file makes "did I update every call site" much harder to
verify by eye, and makes code review/diffs noisier than necessary — most
diffs to this file necessarily show huge unrelated context.

**Proposed refactor (large, but well-contained given the existing internal
seams already visible from the method groupings above):**

```
ai/ppo/
  ppo_trainer.py       # PPOTrainer: __init__, train(), _train_parallel(),
                        # orchestration only — delegates to the below
  action_sampling.py   # _sample_action, _get_value, _compute_log_prob,
                        # _recompute_log_prob, _compute_entropy,
                        # _move_dir_head, _kick_dir_head, _per_head_new_log_probs
                        # (pure functions taking networks+obs, no self state
                        # beyond frozen-head config — good candidates to be
                        # free functions or a small stateless helper class)
  update.py            # _ppo_update — the actual clipped-objective step
  pretrain.py           # pretrain_combined, pretrain_value,
                        # _collect_value_pretrain_rollout
  checkpoint_io.py      # _save_checkpoint(_to), load_checkpoint,
                        # from_config, load_for_inference
  logging_utils.py      # _log_rollout_summary + its formatting helpers
ai/eval/
  trainer_eval.py       # _eval_vs_rules, _eval_vs_immobile,
                        # _eval_vs_opponent_type  (moved OUT of ppo_trainer.py
                        # entirely — these are pure "run env, tally outcomes"
                        # helpers, arguably belonging next to seeded_eval.py)
```

This is a genuinely large refactor and shouldn't be done in one PR — but even
splitting off `checkpoint_io.py` and `logging_utils.py` first (the two most
mechanically separable pieces, since they mostly just read/write
`self.decision_net`/`self.execution_net`/`self.optimizer` and have few
cross-dependencies on the rest of the class) would meaningfully shrink the
file without touching any training-math code at all, and is a low-risk
first step. `_eval_vs_*` methods migrating to `ai/eval/` is the next
lowest-risk step (they call `self`-owned networks + an env factory, could
take the trainer as a plain argument instead of being bound methods).

### 2.2 `bc.py`'s duplicated BCE-smoothing helpers

Two structurally similar smoothed-BCE closures exist in the same file:

```python
# inside bc_loss_from_tensor() (used by pretrain paths on stored labels)
_floor_bce(_I_SHOOT,    smoothing=dec_label_smoothing)
```
and later, inside the "live decision/execution heads" loss function:
```python
_bce(decision_heads.shoot_logit, _I_SHOOT, smoothing=dec_label_smoothing)
```

Both apply the same `target = target*(1-eps) + 0.5*eps` smoothing rule
documented in `/memories/repo/rules.md`'s "BC label smoothing" section, and
both take an optional `pos_weight`. Since they operate on slightly different
inputs (raw label tensor column vs. a live logit tensor from a forward pass),
full unification isn't trivial, but the **smoothing arithmetic itself**
(`target*(1-eps) + 0.5*eps`) is identical and could be factored into one
tiny shared helper (`_smooth_target(target, eps)`) that both `_floor_bce`
and `_bce` call, removing the risk of the two independently drifting on
exactly how smoothing is applied (e.g. one gets updated to clamp `eps` to
`[0, 1]` and the other doesn't).

### 2.3 The BC_LABEL_DIM / index-24 documentation drift (findings #18, #22)

Direct quotes for the record:

- `ai/knowledge.md` (top-level BC section header): *"`bc.py` stores 17
  floats per step (see the module docstring in `bc.py` for the authoritative
  up-to-date layout table — do not let this count drift out of sync, it has
  already changed twice: 15→16 (added `exec_move`), 16→17 (added
  `ai_type`))."* — This "17" is itself now three generations stale; the
  live code has grown to 25 (`kick_this_tick` supervision fields
  18-24 were added later per the "Critical" callout further down the same
  doc, which DOES correctly describe fields up to index 23, but never
  updates the "17" figure in the opening sentence, nor the `BC_LABEL_DIM`
  value itself).
- `ai/ppo/bc.py`'s own module docstring: *"Flat tensor layout for stored BC
  labels (17 floats per step):"* followed by a table that stops at index 16
  (`ai_type`) — but the code below it defines fields through index 24 and
  sets `BC_LABEL_DIM = 25`. So **the file's own docstring disagrees with its
  own code** — this is the most direct, unambiguous documentation bug found
  in this review, since it's not even a cross-file inconsistency, it's
  the same file's comment vs. its own executable constant.

**Root cause pattern:** every time a new field was appended (`kick_direction`,
`kick_power`, `kick_spin`, and now apparently `kick_dir_z` inserted
non-contiguously at the end as index 24), the code was updated but the
docstring's opening "N floats" sentence and/or its own table were not. The
project's own top-of-file convention note ("do not let this count drift out
of sync... it has already changed twice") shows the team is aware of this
exact risk class, but the mitigation ("write a comment telling future
editors not to forget") has already failed three times in the same file.

**Recommended fix:** stop hand-maintaining the count/table as prose in TWO
separate knowledge files AND the code docstring (three copies total). Options,
roughly in order of preference:
1. Add a cheap unit test (`tests/ai_unit/test_bc.py`, likely already exists —
   verify) that asserts `BC_LABEL_DIM == <the highest _I_* constant> + 1` and
   that every `_I_*` constant is unique and contiguous except where
   explicitly documented as non-contiguous (index 24) — this makes any
   FUTURE index/count mismatch a hard test failure instead of a silent doc
   drift.
2. Generate the markdown table in `ai/knowledge.md` from the `_I_*` constants
   plus a short human-written description string per field (e.g. a small
   dict `{_I_SHOOT: "decision Bernoulli: shoot"}` co-located with the `_I_*`
   definitions), and have a doc-generation script or test print/diff it — this
   removes the two-copies-of-the-same-table problem entirely, at the cost of
   a small amount of tooling.
3. At minimum, immediately fix the three known-wrong numbers (`"17"` in
   `bc.py`'s docstring header and in `ai/knowledge.md`'s opening sentence;
   `"BC_LABEL_DIM = 24"` implied by `ai/knowledge.md`'s table stopping at
   23) and add `_I_KICK_DIR_Z`/index 24 to both docs' tables.

### 2.4 `apply_nn_action.py` kick-power/kick-direction fallbacks (findings #13, #14)

```python
if gating.kick_this_tick:
    kick_dir = gating.kick_direction
    if kick_dir is not None and np.linalg.norm(kick_dir) > 1e-6:
        direction_3d = Vector3(float(kick_dir[0]), float(kick_dir[1]), float(kick_dir[2]) if len(kick_dir) > 2 else 0.0)
    else:
        direction_3d = Vector3(1.0, 0.0, 0.0)  # safe fallback, should not occur
    player.kick_with_direction(
        match,
        direction_3d,
        float(gating.kick_power_fraction) if gating.kick_power_fraction > 0 else 0.85,
        Vector3(*gating.kick_spin) if gating.kick_spin is not None else Vector3.zero(),
    )
```

Two related issues:

- **Power fallback conflates "no meaningful power output" with "kick at
  0.85 power."** If the execution network's power-squashing function (not
  reviewed in this pass, but presumably a `sigmoid`/similar in
  `execution_network.py`) can legitimately output values arbitrarily close
  to (but not exactly) 0, this branch is dead in practice and harmless. But
  if it CAN output exactly `0.0` (e.g. from a `relu`-like squash, or from
  float underflow at very negative logits), every such kick silently becomes
  an 85%-power kick instead of a near-zero "tap". During PPO exploration
  early in training, when a poorly-initialized network could plausibly
  output extreme logit values, this systematically distorts the actual
  physical outcome the agent experiences vs. what its own output implies,
  which could bias the resulting gradient/credit-assignment for the kick
  power head. Recommend clamping (`max(power, epsilon)`) instead of a
  boolean-gated large constant substitution, so the *relationship* between
  network output and physical effect stays monotonic even at the low end.
- **Direction fallback picks an absolute-pitch-frame `+x` direction with a
  comment claiming it "should not occur."** Given this codebase's own
  documented history of NaN/degenerate-vector bugs elsewhere (implied by how
  carefully e.g. `augment.py` and `bc.py` handle near-zero vector guards
  everywhere else with the same `1e-6` epsilon pattern used here), an
  assumption that a genuinely pathological case "should not occur" without
  any logging/counter to verify that assumption in production training runs
  is a soft spot. Recommend: `log.warning(...)` (rate-limited, or a
  `Counter`-style running stat surfaced in the per-rollout log) the first
  time this fallback fires per run, so a systematic policy-collapse issue
  (e.g. the direction head producing NaNs) is immediately visible instead of
  silently kicking the ball toward a fixed absolute-frame point for the rest
  of the run.

### 2.5 `to_orders.py` naming mismatch across `knowledge.md` vs. reality (findings #20, #21)

`src/footballcoach/knowledge.md`'s "Neural network / Orders boundary
(IMPORTANT)" section (quoted fully in Part 1 finding #20) describes a module
called `to_orders.py` with a function `apply_movement_to_player` that
constructs a 50m-away `MoveOrder` from the execution network's
`move_direction` output. Neither that filename nor that function exists in
the current tree — `list_dir` on `ai/action/` shows only `__init__.py`,
`apply_nn_action.py`, `distributions.py`, `gating.py`, `schema.py`. The
actual movement-application code (`apply_nn_action.py::apply_action_to_player`,
quoted in full in the tool trace above) sets `player.desired_direction` /
`player.desired_speed_mode` directly — no `MoveOrder` is constructed
anywhere in this function.

This matters for two reasons:

1. It's directly the bug `/memories/repo/rules.md`'s CRITICAL ARCHITECTURE
   RULE describes as needing a fix ("Currently to_orders.py WRONGLY wraps
   neural output in MoveOrder... THIS MUST BE FIXED"). Based on what this
   review found in the actual code, **the fix has already been applied** —
   the module was seemingly renamed/rewritten from `to_orders.py` to
   `apply_nn_action.py` and now does the correct direct-field-setting thing.
   But neither `src/footballcoach/knowledge.md` NOR (as far as this review's
   grep found) the memory rules file itself was updated to reflect that the
   fix landed — the memory file still describes this as an outstanding "MUST
   BE FIXED" item as of this conversation. **This should be verified with
   the user/git history and then both `knowledge.md` and the memory file
   updated** — either the fix genuinely landed (update both docs to stop
   describing it as broken/pending) or there's a SECOND code path this
   review didn't find that still does the old wrapping behaviour (in which
   case the memory file's warning remains valid and accurate, and
   `knowledge.md`'s description was simply always describing that other,
   still-broken path — needs disambiguation either way).
2. Even setting aside which description is "true," having a knowledge doc
   reference a filename that doesn't exist in the tree is a pure
   navigability bug — anyone (human or agent) grep-ing for `to_orders.py`
   to understand this boundary will get zero hits and have to discover
   `apply_nn_action.py` by other means (as this review had to).

**Recommended fix:** Once confirmed which behaviour is current, rewrite the
entire "Neural network / Orders boundary (IMPORTANT)" section of
`src/footballcoach/knowledge.md` to name `apply_nn_action.py` and
`apply_action_to_player` explicitly, drop the `MoveOrder`-construction
description entirely (replacing it with the direct
`desired_direction`/`desired_speed_mode` description that matches
`apply_nn_action.py`'s actual code and its own accurate docstring: *"No
Orders are created here. Orders exist for the rules-based AI only."*), and
add a one-line note in `/memories/repo/rules.md` that the previously-tracked
`to_orders.py` bug appears fixed as of this review (pending confirmation) so
future sessions don't keep re-flagging already-fixed code as broken.

**RESOLVED (follow-up session, confirmed via direct code read of
`apply_nn_action.py`):** the fix genuinely landed — `apply_action_to_player()`
sets `desired_direction`/`desired_speed_mode` directly, calls
`kick_with_direction()`, and sets `tackle_armed`; no Order/MoveOrder is ever
constructed. All `to_orders.py` filename references and the stale
`kick_direct()`/`tackle_direct()` claims have now been corrected across
`src/footballcoach/knowledge.md`, `src/footballcoach/entities/knowledge.md`,
`src/footballcoach/engine/knowledge.md`, `src/footballcoach/ai/knowledge.md`,
`ai_trainer_knowledge.md`, `tests/knowledge.md`, and the handful of inline
code comments that repeated the same stale filename/mechanism
(`entities/player.py`, `ai/action/schema.py`, `ai/models/decision_network.py`,
`ai/models/execution_network.py`, `ai/env/reward.py`, `ai/ppo/bc.py`).

### 2.6 `gating.py`'s `_HEAD_ORDER`/`_HEAD_TO_ACTION` duplication (finding #8)

```python
_HEAD_ORDER: list[str] = [
    "shoot", "pass_", "move", "tackle", "get_possession", "mark", "hold_position"
]
...
_HEAD_TO_ACTION: dict[str, SelectedAction] = {
    "shoot": SelectedAction.SHOOT,
    "pass_": SelectedAction.PASS,
    "move": SelectedAction.MOVE,
    "tackle": SelectedAction.TACKLE,
    "get_possession": SelectedAction.GET_POSSESSION,
    "mark": SelectedAction.MARK,
    "hold_position": SelectedAction.HOLD_POSITION,
}
```

Two structures, hand-kept in sync, both enumerating the same 7 head names. A
cheap, high-value addition:

```python
assert set(_HEAD_ORDER) == set(_HEAD_TO_ACTION), "gating head-name tables drifted apart"
```

placed at module import time (so it fails immediately and loudly on import
if someone adds a head to one list and not the other, rather than only
surfacing as a silent `.get(head_name, 0.0)` fallback to `0.0` probability
at runtime — the latter is exactly what would happen today if a new head
were added to `_HEAD_ORDER` but not `_HEAD_TO_ACTION`, since
`_HEAD_TO_ACTION[best_head]` would then raise `KeyError` deep inside
`select_action`, with an unhelpful stack trace far from the actual root
cause).

### 2.7 Test coverage gaps — concrete recommendations

- **`apply_nn_action.py`** (finding #13/#14/#16/#17): confirm
  `tests/ai_unit/test_apply_nn_action.py` covers:
  - `kick_power_fraction == 0.0` exactly → asserts NOT silently forced to
    0.85 (or, if the fallback is intentionally kept, at least asserts the
    behaviour explicitly rather than leaving it implicit).
  - `kick_direction` is `None` / a zero vector → asserts fallback direction
    and (recommended) a logged warning/counter increment.
  - Tackle attempt with `ball_carrier() is None` vs. `ball_carrier().team ==
    player.team` → asserts these produce **distinct** `illegal_reason`
    strings (currently they don't — see finding #16 — so this test would
    currently need to assert the (arguably wrong) shared string, or should
    be written now to lock in the fix once applied).
  - `encode_slot_player_ids` with mismatched `slot_indices`/`other_players`
    lengths → should assert either an explicit `ValueError` (recommended new
    behaviour) or explicitly document+test the current silent-truncation
    behaviour.
- **`ai/curriculum/`**: add `tests/ai_unit/test_curriculum_phases.py`
  (phase construction, `PHASES_BY_ID` completeness/uniqueness) and
  `tests/ai_unit/test_opponent_pool.py` (sampling distribution given a
  config, `apply_rules_based_opponent()` behaviour, the `except (KeyError,
  AttributeError)` branch at opponent_pool.py:83).
- **`ai/eval/seeded_eval.py`**: add a determinism test — run the same seed
  twice through the seeded eval path and assert identical outcome
  sequences/rewards. This directly targets an item already called out as an
  open task in the user's own notes ("The evaluation steps should also use
  seeds so that they're always the same!").
- **`ai/models/entity_encoder.py`**: add a direct permutation-invariance
  test at the encoder level (permute the "other players" slot order in a
  synthetic batch, assert `entity_encoder(...)`'s pooled output is identical
  up to floating-point tolerance) — currently this guarantee is only tested
  indirectly via the value-side-channel wrapper test, not the core encoder
  itself, despite this being one of the most architecturally load-bearing
  properties described repeatedly across the knowledge docs.
- **`gating.py`**: add the `_HEAD_ORDER`/`_HEAD_TO_ACTION` consistency
  assertion from 2.6 as an explicit unit test (in addition to, or instead
  of, the module-import-time assert) so a future CI run catches a drift
  immediately with a clear failure message and file location.

### 2.8 Steering/repulsion doc claim worth spot-checking (finding #23)

`src/footballcoach/knowledge.md` states plainly: *"**Only `MoveOrder` uses
repulsion.** ChaseTackleOrder, GetPossessionOrder, SaveOrder, etc. do not —
confirmed out of scope."* This review did not open `engine/match.py`'s
`_process_orders` dispatch to verify this claim directly (it's outside
`ai/` proper, and this task's stated scope is the `ai/` critique). Given how
many other "confirmed"/"deliberate" claims elsewhere in these same docs
turned out to be stale (findings #18/#20/#21/#22), this specific claim is
flagged as worth a deliberate one-off verification pass (grep `compute_repulsion`
call sites in `match.py`) rather than being re-asserted here as fact, purely
on the strength of it being written down confidently in a knowledge file
that has otherwise shown a real drift pattern.

### 2.9 Miscellaneous small items / easy wins

- `ppo_trainer.py:2724`'s `pitch_hl = 52.5` / `pitch_hw = 34.0` (finding #9):
  these values almost certainly already exist somewhere in `config/` (pitch
  dimensions are a core physics constant used throughout `engine/`). Easy
  win: thread the actual pitch half-length/half-width through from
  `GlobalFeatures`/config rather than re-hardcoding standard FIFA pitch
  dimensions inline, closing out the file's own `# TODO` comment.
- `augment.py`'s `_BC_DIR_X_COL`/etc. (finding #10): straightforward fix —
  `from footballcoach.ai.ppo.bc import _I_DIR_X, _I_DIR_Y, _I_REGION_X,
  _I_REGION_Y, _I_KICK_DIR_X, _I_KICK_DIR_Y, _I_KICK_DIR_Z, _I_KICK_SPIN_X,
  _I_KICK_SPIN_Y, _I_KICK_SPIN_Z` and delete the six hand-duplicated
  constants. No circular-import risk was found (`bc.py` does not import
  from `obs/`), so this is a pure mechanical de-duplication with no
  architectural cost — one of the cheapest, highest-value fixes in this
  whole review given it directly removes a landmine the code's own comment
  admits exists.
- `mv_size_phys = 1.0 + 3.0 * torch.sigmoid(mv_size_raw)` and `mv_speed_phys
  = ... * 9.5` (finding #11): move the `1.0`/`3.0`/`9.5` (and the pitch
  constants above) into `ai_config.json` alongside every other tunable in
  this codebase, for consistency with the project's own stated convention
  ("all coefficients are loaded from ai_config.json so they can be tuned
  without touching code" per `reward.py`'s own module docstring) — right now
  these squashing-range constants are the exception to that rule for no
  apparent reason.
- Head-name string literals (finding #12): introduce a single shared
  constant, e.g. `DECISION_HEAD_NAMES: tuple[str, ...] = ("shoot", "pass_",
  "move", "tackle", "get_possession", "mark", "hold_position")` in
  `action/schema.py` (or wherever `DecisionHeadsRaw` already lives), and have
  both `gating.py` and `ppo_trainer.py`'s `decision_probs` dict construction
  reference it instead of independently re-typing the same 7 strings.
