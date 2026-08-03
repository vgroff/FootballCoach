# AI Design Plan v2 — Observation/Value Architecture, Pre-training Refactor, BC Balancing

Status: design/plan only, not yet implemented. This is a work order for a
follow-up implementation session (or agent). Read this whole document before
touching code — later sections depend on decisions made in earlier ones.

Read first: `ai_trainer_knowledge.md`, `src/footballcoach/ai/knowledge.md`,
`src/footballcoach/ai/obs/schema.py`, `src/footballcoach/ai/models/*.py`,
`src/footballcoach/ai/ppo/bc.py`, `src/footballcoach/ai/ppo/ppo_trainer.py`.

**Tests are mandatory alongside every workstream below, not an afterthought.**
Several of the modules touched here (`bc.py`, `ppo_trainer.py`,
`DemonstrationDataset`) currently have **zero** dedicated test coverage.
Add baseline regression tests for current behaviour *before* changing it
where a workstream says so.

---

## 0. Ranking by difficulty / effort

Ordered easiest → hardest. Do them in roughly this order — later workstreams
build on earlier ones (in particular, W1 and W6 are prerequisites for W3 and
W5's opponent-type feature).

| # | Workstream | Effort | Depends on |
|---|---|---|---|
| W1 | Baseline tests for currently-untested code (`bc.py`, `DemonstrationDataset`, smoke test for `pretrain_combined`) | S | — |
| W2 | Doc fixes in `schema.py` (dimension comments) + `pos_x`/`pos_y` flip-sign test | S | — |
| W3 | Task-id field in `GlobalFeatures` (scaffold only, not wired to mixed-task training) | S | — |
| W4 | BC dataset class-balancing: `pos_weight` at load time + gentle per-epoch downsampling | M | W1 |
| W5 | Demo recording: both players + `ai_type` label | M | W1 |
| W6 | Value-only opponent-AI-type side-channel (`value_extra_mlp`, permutation-aware) | M/L | W1, W5 |
| W7 | Pre-training refactor: Phase 0 unfreeze + combined decision+value loss, `pretrain_combined` calls `pretrain_value`, reward/win-rate logging | M | W1 |
| W8 | Config exposure: opponent-immobile-prob for demo recording, new coefficients | S | W4, W6, W7 |

Total is a substantial multi-day effort if done carefully with tests. Do not
parallelize W6/W7 against each other — both touch `ppo_trainer.py` and will
conflict.

---

## 1. Decisions already made (do not re-litigate these)

These were settled across the design conversation. Restating them here so
implementation doesn't have to re-derive them from chat history.

1. **Ball absolute position**: NOT added. Relative-to-self is sufficient
   since the ball doesn't go through attention (single global entity, its
   own `ball_mlp` branch). Absolute ball position is trivially recoverable
   as `self.pos + ball.rel` if the trunk ever needed it.
2. **`distance_m` redundant feature**: KEEP as-is in both `PlayerFeatures`
   and `BallFeatures`. Deliberate — reduces function-approximation burden
   on early layers. Do not remove.
3. **Task-id**: visible to BOTH policy and value (unlike opponent-AI-type).
   Lives in `GlobalFeatures`, feeds the shared trunk normally.
4. **Opponent-AI-type**: visible to VALUE ONLY, never to policy heads.
   Implemented as a flat side-channel that bypasses the entity
   encoder/attention entirely (see W6) — NOT as a `PlayerFeatures` field.
   This was a reversal from an earlier idea (putting it in `PlayerFeatures`)
   — do not put opponent-AI-type in `PlayerFeatures`.
5. **Self AI-type during live PPO/eval**: always the "neural" one-hot
   (self is always the network being trained/evaluated in that context).
   During BC/value-demo-pretrain, self's AI-type reflects whatever produced
   the demo (currently always "rules", see W5).
6. **Demo dataset AI-type values**: for now, only ever `rules` or
   `immobile` (never `neural`) — `record_demonstrations.py` has no neural
   recording mode and we are explicitly deferring that (see W5 note).
   Code the one-hot / label plumbing generically so adding a `neural` value
   later is a config/data change, not an architecture change.
7. **`value_extra_mlp` warm-up**: no special-casing. It trains jointly with
   the rest of the value head from the very first pretraining gradient step
   (Phase 0). Its early data (Phase 0, demo-only) is a narrower distribution
   (never "neural") than later stages (Phase 2/3 live rollout, PPO) — this
   is expected, not a bug, and improves over training as broader-mix data
   arrives.
8. **DAgger**: NOT adding textbook dataset-aggregation DAgger. The existing
   BC-aux-during-PPO mechanism (querying the rules-AI expert on **live
   on-policy states** each rollout step via `label_fn(env)`) is already
   DAgger-*flavored* and sufufficient.
9. **Class-weight reweighting**: apply `pos_weight`-style reweighting to
   **BC losses only** (both pretrain BC and the BC-aux-during-PPO loss,
   including its annealing coefficient — the anneal schedule is unaffected
   and orthogonal). Do **NOT** apply frequency-based reweighting to the raw
   PPO policy-gradient/clipped-surrogate loss — that's a correctness risk,
   not just a variance one.
10. **Downsampling of "trivial" BC rows**: gentle, exposed via config,
    resampled freshly **every epoch** (never a fixed one-time filter at
    load time — that would permanently discard variance). Default: cut
    50% of trivial rows per epoch; if `n_epochs > 5`, cut 65% instead
    (both values exposed as config, not hardcoded).
11. **`pretrain_value()` vs `pretrain_combined()` Phase 2/3 duplication**:
    confirmed both already use the identical opponent-mix config
    (`phase1_opponent_rules_prob` / `phase1_opponent_immobile_prob` via
    `build_1v1_scenario`) — this was verified by reading
    `ui/scenarios.py` lines ~1124-1140 directly, not assumed. Refactor so
    `pretrain_combined()` calls `pretrain_value()` internally instead of
    duplicating rollout-collection + GAE + epoch-loop logic. Keep
    `pretrain_value()` usable standalone.
12. **BC repair** (`bc_repair_epochs`): leave alone. Already 0 in config
    (effectively disabled). Do not remove the code path, do not spend
    effort on it now.
13. **Phase 0 freezing**: REMOVE. Do not freeze trunk/encoder params during
    the demo value pretrain phase. Use a single optimizer over
    `decision_net.parameters()` (value_head included) with a combined loss
    `decision_bc_loss + phase0_value_coef * value_loss`, one backward pass.
    `phase0_value_coef` is a new exposed config key (naming: see W8).

---

## 2. W1 — Baseline tests for currently untested code

**Why first**: every later workstream edits `bc.py` and `ppo_trainer.py`.
Both currently have **no dedicated test file** (confirmed via directory
listing of `tests/ai_unit/`: only `test_apply_nn_action.py`,
`test_distributions.py`, `test_gae.py`, `test_gating.py`, `test_networks.py`,
`test_obs_encoder.py`, `test_obs_schema.py`, `test_reward.py` exist).
Changing delicate loss/pretraining code with zero regression coverage is
how silent bugs like the `pretrain_value()`-vs-`pretrain_combined()`
duplication drift happen.

### 2.1 New file: `tests/ai_unit/test_bc.py`

Cover **current** behaviour of `src/footballcoach/ai/ppo/bc.py` before any
of W4/W5/W6 changes land:

- `BCLabel.to_array()` / `BCLabel.invalid()`: round-trip a hand-built
  `BCLabel`, assert the flat array matches expected indices exactly
  (use the `_I_*` index constants from `bc.py`, don't hardcode magic
  numbers in the test — import them).
- `bc_loss_from_tensor()`:
  - Construct a minibatch of 4 synthetic rows with known logits (e.g. all
    zeros → sigmoid 0.5) and known labels; hand-compute expected BCE loss
    for at least one row and assert `abs(loss - expected) < 1e-5`.
  - `valid=False` rows must contribute exactly zero to the mean (test by
    comparing a batch with one invalid row of garbage labels vs the same
    batch with that row removed — losses must match, since the "removed"
    version excludes it entirely from denominator too, i.e. mean over valid
    only).
  - `move_direction` cosine loss: feed a perfectly-aligned predicted
    direction and assert loss ≈ 0; feed an exactly opposite direction and
    assert loss ≈ `direction_loss_weight * 2.0`.
  - `move_region_center` MSE: assert zero loss for a perfect match, nonzero
    scaling by `region_loss_weight`.
  - `return_breakdown=True`: assert the breakdown dict entries sum
    approximately to the fields they claim to cover (e.g.
    `decision + exec_bce + direction + region` accounts for `total`, modulo
    the double-counting note that `tackle_attempt` is a sub-component of
    `exec_bce` already, so don't naively sum every dict value — check
    `bc_loss_from_tensor`'s exact composition before writing this assertion,
    the dict is diagnostic, not strictly partitioned).
- `phase1_labels(env, player_id)`:
  - Build a minimal real `ScenarioEnv` (reuse the pattern from
    `debug_kl.py` / existing `test_apply_nn_action.py` for how tests already
    construct a `ScenarioEnv` + `build_1v1_scenario`).
  - Force a state where `Phase1RulesAI` will set a `MoveOrder` (e.g. no
    ball possession by anyone within reach, trainee not adjacent to ball,
    opponent has possession) — assert returned `BCLabel.move == 1.0` and
    `move_direction` points from player toward the actual `MoveOrder`
    target.
  - Force a state producing `GetPossessionOrder` — assert
    `get_possession_extra == 1.0` and direction points toward the ball.
  - Force degenerate zero-length direction (player already exactly at
    target) — assert `BCLabel.invalid()` is returned (see the
    `length < 1e-6` guard in `phase1_labels`).

### 2.2 New file: `tests/ai_unit/test_demonstration_dataset.py`

No test file currently covers `src/footballcoach/ai/bc/dataset.py` at all.
Before W4/W5 change its loading/iteration logic, add:

- `DemonstrationDataset.from_directory()` loads a small hand-written `.npz`
  fixture (write one in a `tmp_path` fixture, don't rely on the real
  `demonstrations/phase1/` data — that would make the test slow and
  non-deterministic against real recordings).
- `iterate_minibatches()`: batch shapes correct, `shuffle=True` actually
  permutes (compare two calls with different seeds produce different
  orders), `valid_only=True` filters rows where label validity flag is 0.
- `compute_returns(gamma)`: hand-compute expected discounted return for a
  short synthetic episode (3-4 steps, known rewards, known `dones`) and
  assert exact match.
- `has_rewards`: True/False correctly reflects whether the loaded `.npz`
  files contain a rewards array.

### 2.3 Smoke test: extend `tests/ai_scenario/test_smoke.py`

Add one coarse end-to-end smoke test: build a tiny synthetic
`DemonstrationDataset` (reuse the W1 fixture), call
`PPOTrainer.pretrain_combined(env, dataset, n_epochs=1, batch_size=8,
bc_lr=1e-3, value_lr=1e-3, rollout_steps=64, value_epochs=1)` with a real
but tiny `ScenarioEnv`, and assert:
- no `NaN`/`Inf` in any resulting parameter tensor after the call
  (iterate `decision_net.parameters()` + `execution_net.parameters()`,
  assert `torch.isfinite(p).all()`).
- it completes without raising.

This test would have caught the `pretrain_value()`-dead-code-path situation
and the Phase-0-gradient-discard non-bug earlier had it existed — treat it
as the regression guard for the W7 refactor.

**Do not proceed to W4–W7 until W1's new tests pass against the *current*,
unmodified code.** This validates the tests themselves are correct before
they're used to catch regressions.

---

## 3. W2 — Doc fixes + pos_x/pos_y flip test

Small, low-risk, do any time.

### 3.1 `schema.py` docstring fixes

Current inaccuracies to fix:
- `PlayerFeatures` docstring says `"(26 floats)"` — actual current count is
  27 (or will be 28 after W3's task-id... no, task-id goes in
  `GlobalFeatures`, not `PlayerFeatures` — `PlayerFeatures` stays at 27
  unless W6 changes it, which it does NOT, per decision #4 in section 1).
  **Fix the docstring number to reflect the true `len(fields(PlayerFeatures))`
  at time of edit** — don't hardcode a number that will drift again; prefer
  wording like *"(see `PLAYER_FEATURE_DIM` for the authoritative count)"*.
- The comment on `PLAYER_FEATURE_DIM: int = len(fields(PlayerFeatures))
  # 27 (was 25 pre pos_x/pos_y)` has a stale historical note. Replace with
  just deriving the number, dropping the misleading "was 25" (it was
  actually 26 pre pos_x/pos_y per the docstring elsewhere — the two
  comments contradict each other; fix both to agree, or better, remove the
  historical aside entirely since dataclass field count is self-evident
  from the class body).

### 3.2 New/confirmed test: `pos_x`/`pos_y` flip-sign in `test_obs_encoder.py`

Before this design conversation, it was **claimed but not directly
confirmed** that `augment.py` correctly negates `pos_x`/`pos_y` under
`flip_x`/`flip_y`. Grep of `test_obs_encoder.py` for `pos_x|flip_x|flip_y`
returned **no matches** — there is currently no test for this. Add one:

```python
def test_pos_x_pos_y_negated_under_flip(...):
    # Build an observation with a known non-zero pos_x/pos_y for self and
    # one other player, run augment_batch() with a flip_x-only variant,
    # assert self_feat[...pos_x index...] is exactly negated and
    # pos_y is unchanged; then check flip_y negates pos_y and leaves pos_x.
```

**Confirmed by direct read of `augment.py`**: `pos_x`/`pos_y` negation IS
already correctly wired — `PLAYER_FLIP_X_IDX` includes
`_field_index(PlayerFeatures, "pos_x")` (with comment "absolute x negated
under flip_x") and `PLAYER_FLIP_Y_IDX` includes `pos_y` analogously. So
`schema.py`'s claim was accurate, not stale — **this is documentation
confirmation, not a live bug**. The test in this subsection is still worth
writing (it's a real regression guard for something currently correct but
untested), just downgrade the framing from "possible bug" to "confirmed
correct, add regression coverage."

While reading `augment.py` for this, note the mechanism precisely for
future reference (relevant again in W6): `_field_index(dataclass_type,
name)` looks up a field's position via `fields(dataclass_type)` by name —
this is a clean, index-drift-proof mechanism **for `PlayerFeatures`/
`BallFeatures` fields specifically**. It does **not** generalize
automatically to arbitrary side-channel arrays like W6's
`self_ai_type`/`other_ai_type` (which are not `PlayerFeatures` fields at
all) — W6 will need its own, separate, explicit permutation-application
code, not a `_field_index()` lookup. Confirmed also: BC label columns are
handled completely differently in this same file — see the hardcoded
`_BC_DIR_X_COL = 7` / `_BC_REGION_X_COL = 10` etc. constants a few lines
below the `PLAYER_FLIP_X_IDX` block. **These are magic numbers, not
derived from `bc.py`'s `_I_*` constants** — a real, pre-existing
fragility: if `BC_LABEL_DIM`'s layout ever shifts (which W5 does, by
appending `ai_type` at index 16, after `valid` at index 14... wait, check
ordering: current layout per `bc.py`'s docstring is indices 0-15 with
`valid` at 14 and `exec_move` at 15 — so appending `ai_type` at a NEW
index 16, after both, does NOT shift any existing column, so `augment.py`'s
hardcoded `_BC_DIR_X_COL=7` etc. remain correct after W5's change
specifically. **This is a fortunate ordering, not a guarantee** — flag in
a code comment right next to these constants in `augment.py` that any
FUTURE `bc.py` label layout change must cross-check these hardcoded
indices too, since they are not automatically derived and will silently
go stale otherwise. Fixing them to import from `bc.py`'s `_I_*` constants
instead of hardcoding is a good small drive-by improvement while W5 is
already touching both files, though not strictly required for W5 to work.

---

## 4. W3 — Task-id scaffold in `GlobalFeatures`

**Scope for this workstream: add the field, populate it correctly from the
active `--phase` CLI arg, default to a valid one-hot. Do NOT implement
actual mixed-multi-phase rollout orchestration — that's a separate, larger
follow-on item (note it below but leave unimplemented).**

### 4.1 Schema change

In `ai_config.json`, add under a suitable section (`observation` or a new
top-level `curriculum` entry — prefer `observation` since it's a dimension
constant akin to `max_other_players`):
```json
"max_task_ids": 20
```
In `schema.py`, add to `GlobalFeatures`:
```python
    # --- Task identifier (one-hot) ---
    # Which curriculum phase/task is currently active. Fixed-width
    # (MAX_TASK_IDS from ai_config.json, default 20) so the network
    # architecture doesn't change as new phases are added — unused task
    # slots stay zero. Task 0 = phase 1, task 1 = phase 2, etc. (index =
    # phase_id - 1). NOT YET WIRED for mixed multi-phase training — see
    # ai/knowledge.md "Task-id: scaffolded, not yet load-bearing" note.
    task_id: tuple[float, ...] = field(default_factory=lambda: (0.0,) * MAX_TASK_IDS)
```
Careful: `dataclass` + `astuple()` will flatten a tuple-typed field
correctly into `to_array()`'s output only if you handle it — **verify**:
`astuple()` on a dataclass containing a tuple field produces a **nested**
tuple (the tuple field becomes a sub-tuple, not flattened), which will
break `np.array(astuple(self), dtype=np.float32)` for the object as a whole
(mixed scalar + tuple elements won't produce a clean flat float32 array).
**Do not use a tuple field.** Instead, either:
(a) add `MAX_TASK_IDS` individual scalar fields (`task_id_0` .. `task_id_19`)
    — verbose but keeps `astuple()`/`to_array()` working unchanged, matching
    the existing pattern of every other field in this file, or
(b) change `to_array()`'s implementation for `GlobalFeatures` to manually
    concatenate `astuple(self)[:-1]` (scalars) with the task-id array
    (requires reordering the field so the vector field is last, and a
    custom `to_array()` override instead of the generic `astuple()` call).

**Recommendation: (a)**, for consistency with the rest of the file and
because `MAX_TASK_IDS=20` is a small, fixed, config-driven constant — 20
extra float fields is not a maintenance burden, and avoids the flattening
pitfall entirely. If you find `GLOBAL_FEATURE_DIM` used anywhere assuming a
specific literal current value (11), grep for it before landing this change
— `decision_network.py`'s `global_dim: int = GLOBAL_FEATURE_DIM` default
argument reads the constant, not a literal, so should adapt automatically,
but any test asserting `GLOBAL_FEATURE_DIM == 11` literally will need
updating (check `test_obs_schema.py`).

### 4.2 Encoder change

`encode_observation()` needs a new parameter (`phase: int` or similar,
likely already available via the caller — check `ScenarioEnv`/training
scripts for where `phase` is already tracked) to populate the one-hot.
Default to all-zeros (or task 0) if `phase` isn't passed, but prefer making
it a required parameter once wired, so a missing task-id is a loud error,
not a silent all-zero encoding that could mask a real problem later once
this becomes load-bearing.

### 4.3 Knowledge-file note (required)

Add to `src/footballcoach/ai/knowledge.md` under a new short section:

> **Task-id (GlobalFeatures)**: a `MAX_TASK_IDS`-wide one-hot field exists
> and is correctly populated per the active `--phase`, but there is
> currently no mixed-multi-phase training loop that would ever populate it
> with more than one non-zero pattern within a single training run. It is
> scaffolding for future multi-task training; treat any gradient signal
> through it as currently uninformative (constant within a run). Wiring
> real multi-phase rollout mixing is a separate, larger workstream — not
> yet planned in detail.

### 4.4 Tests

- `test_obs_schema.py`: `GLOBAL_FEATURE_DIM` reflects the new field count;
  `GlobalFeatures().to_array()` shape includes the task-id block.
- `test_obs_encoder.py`: encoding with `phase=1` produces a one-hot at index
  0 (or whatever the agreed indexing convention is), `phase=2` at index 1,
  out-of-range phase raises or clamps (decide and test the decided
  behaviour explicitly, don't leave it implicit).

---

## 5. W4 — BC dataset class balancing

Two independent mechanisms, both configurable, both only affecting BC
losses (never raw PPO policy gradient, per decision #9).

### 5.1 `pos_weight` computed at dataset load time

Add to `DemonstrationDataset` (in `ai/bc/dataset.py`) a method computed
once in `from_directory()` (or lazily cached on first access — either is
fine, prefer eager-at-load since the dataset is already fully materialized
in memory at that point):

```python
def compute_pos_weights(self) -> dict[str, float]:
    """Inverse-frequency weights for rare Bernoulli BC targets.
    weight = n_negative / max(n_positive, 1), matching
    torch.nn.functional.binary_cross_entropy_with_logits's pos_weight arg
    semantics (weight applied to the positive-class term).
    """
```
Compute for at minimum: `kick_this_tick` (`_I_KICK_THIS_TICK`),
`tackle_attempt` (`_I_TACKLE_ATTEMPT`), and (optional, check actual
imbalance first via the W1-informed logging in section 7 before deciding
this is needed) `tackle`/`mark`/`hold_position` decision heads if they
also turn out to be rare.

**Exact implementation, using the real `DemonstrationDataset` internals**
(confirmed by direct read of `ai/bc/dataset.py` — labels are stored as a
private `self._labels` numpy array of shape `(N, BC_LABEL_DIM)`, and
`valid_indices()` already exists as `np.where(self._labels[:, -1] > 0.5)[0]`
— note it indexes `-1`, i.e. "last column", not the named `_I_VALID`
constant; this coincidentally still works after W5 bumps `BC_LABEL_DIM`
16→17 **only because `valid` happens to remain the last field** — if a
future change ever inserts a new field after `valid` instead of before it,
this `-1` indexing silently breaks. Worth proactively hardening while
you're in this file: change `valid_indices()` to
`np.where(self._labels[:, _I_VALID] > 0.5)[0]` with `_I_VALID` imported
from `bc.py`, removing the magic `-1` entirely.):
```python
def compute_pos_weights(self) -> dict[str, float]:
    from footballcoach.ai.ppo.bc import _I_KICK_THIS_TICK, _I_TACKLE_ATTEMPT
    valid = self.valid_indices()
    labels = self._labels[valid]
    out = {}
    for name, col in [("kick", _I_KICK_THIS_TICK), ("tackle_attempt", _I_TACKLE_ATTEMPT)]:
        n_pos = float((labels[:, col] > 0.5).sum())
        n_neg = float(len(labels)) - n_pos
        out[name] = n_neg / max(n_pos, 1.0)
    return out
```
Note `bc.py`'s `_I_*` constants are currently module-private (no `__all__`
export, leading underscore) — either drop the underscore prefix on the
ones needed cross-module (`_I_KICK_THIS_TICK` → `I_KICK_THIS_TICK`, small
rename, check for existing internal usages before renaming so you don't
break `bc.py` itself) or keep the underscore and just import it anyway
(Python doesn't enforce privacy, but it's a code-smell to import
underscore-prefixed names across module boundaries) — **prefer the
rename**, it's cheap and this workstream already touches `bc.py`.

Thread this into `bc_loss_from_tensor()` as new optional parameters:
```python
def bc_loss_from_tensor(
    labels, decision_heads, exec_heads,
    direction_loss_weight=3.0, region_loss_weight=1.0,
    pos_weight_kick: float = 1.0,
    pos_weight_tackle_attempt: float = 1.0,
    return_breakdown=False,
):
```
Apply via `F.binary_cross_entropy_with_logits(..., pos_weight=torch.tensor(pos_weight_kick, device=...))`
only on the two (or more) affected `_bce(...)` calls — leave all other
heads' `pos_weight` at the implicit default of 1.0 (no reweighting) unless
audit logging (section 7) shows another head needs it too.

**Pitfall**: `pos_weight` must be a scalar tensor on the correct device,
recomputed once per training run (not per-minibatch — it's a property of
the whole dataset, recomputing per-batch would be both wasteful and noisy
for small minibatches with zero positives by chance).

**Do NOT apply this to PPO's own aux-BC-loss call unless you also want
class balancing during PPO's annealed BC term** — actually decision #9
explicitly says apply to BOTH BC pretrain AND the BC-aux-during-PPO call
(same `bc_loss_from_tensor` function, same weights, just don't touch the
separate PPO policy-gradient loss). Make sure the `pos_weight_*` values are
computed once (at dataset-load, from the pretrain dataset) and threaded
through to *every* call site of `bc_loss_from_tensor`, including inside
`_ppo_update()`'s aux-loss computation — grep for all call sites before
starting, there are at least 3 in `bc.py`/`ppo_trainer.py`.

### 5.2 Gentle per-epoch downsampling of "trivial" movement rows

New config keys under `ai_config.json`'s `bc` section (final names — see
W8 for naming convention confirmation, draft names below):
```json
"downsample_trivial_enabled": true,
"downsample_trivial_frac_default": 0.5,
"downsample_trivial_frac_high_epoch": 0.65,
"downsample_trivial_epoch_threshold": 5,
"downsample_trivial_cos_threshold": 0.98
```
"Trivial" definition: a movement row where the BC label's `move_direction`
is closely aligned with the **previous timestep's** stored movement
direction for the same trajectory (cosine similarity above
`downsample_trivial_cos_threshold`) — i.e., "moving the same way as before,
nothing interesting changed." This requires access to consecutive-in-time
rows.

**Confirmed by direct read**: `DemonstrationDataset` arrays are populated
via `np.concatenate([p._self_feat for p in parts])` in `from_files()`, and
each part (`from_file()`) loads one `.npz` written by
`record_episodes()` in strict recording order — `record_episodes()`
appends samples to plain Python lists (`self_feats.append(...)` etc.) in
the exact order `_record_now()` is called, i.e. **original episode/time
order is preserved on disk and on load**; nothing shuffles it until
`iterate_minibatches(shuffle=True)` explicitly does `np.random.shuffle(indices)`
at iteration time. So: **yes, "previous timestep" is well-defined and
recoverable directly from row `i-1`** for any `i` that isn't the first row
of an episode (`self._dones[i-1] > 0.5` marks an episode boundary — treat
row 0 of every episode as automatically non-trivial / ineligible for
downsampling, since it has no valid "previous" within the same episode).
Compute the "trivial" boolean mask **once**, at load time (a new method
`_compute_trivial_mask()` called from `__init__` or lazily cached on first
access), using this exact adjacency — **do not** recompute it inside
`iterate_minibatches()` per epoch (only the *exclusion subset* should be
re-rolled per epoch, per the correctness point below; the underlying
"which rows are eligible to be excluded" classification is static and
should be computed exactly once).

**Important gap to close while implementing this**: `iterate_minibatches_with_returns()`
(used by `pretrain_combined()`'s Phase 0/joint-value-loss path) currently
has **no `valid_only` parameter at all** (confirmed by direct read — its
signature is `iterate_minibatches_with_returns(self, batch_size, returns,
shuffle=True, device=None)`, unlike `iterate_minibatches()` which has
`valid_only=True`). This means Phase 0's demo-value-pretrain loop
currently iterates over **all** rows including invalid ones (label
`valid=0`) for the *value* loss — which is arguably fine for value fitting
(the return target doesn't depend on BC label validity) but means the two
methods have inconsistent filtering semantics that will make downsampling
(which should also apply consistently, or explicitly not, to this path)
confusing to reason about. **Decide explicitly**: I'd recommend adding
`valid_only: bool = False` to `iterate_minibatches_with_returns()` too
(default False to preserve current behaviour exactly), and **not**
applying the trivial-downsampling logic to this returns-only iterator at
all (downsampling is specifically about reducing redundant *BC* signal;
value-target fitting benefits from seeing the full return distribution,
including "boring" states — those are informative for the critic even
when uninformative for BC). Document this asymmetry explicitly in the
docstring once implemented, since it's a deliberate, easy-to-forget
design choice.

**Critical correctness point (agreed in conversation, restate here so it's
not lost)**: the downsample selection must be **freshly re-rolled every
epoch** (independent random subset excluded each epoch), never a single
fixed mask decided once at load time. Implementation sketch:

```python
def iterate_minibatches(self, batch_size, shuffle, device, valid_only=True,
                         downsample_trivial_frac=0.0, rng=None):
    # 1. Identify "trivial" row indices ONCE (cached, computed from
    #    static per-row features, does not depend on epoch).
    # 2. Each call (i.e. each epoch): rng.choice() a FRESH random subset
    #    of the trivial-row-index set to EXCLUDE this epoch, sized by
    #    downsample_trivial_frac. Non-trivial rows are never excluded.
    # 3. Proceed with normal shuffle/batch over the remaining row indices.
```
Pass a `numpy.random.Generator` (not the global `np.random` state) so the
per-epoch re-rolling is reproducible given a seed, consistent with the
rest of this codebase's RNG-handling conventions (`rng_reduction`,
`RNG` utilities in `mathutils`).

**Pitfall flagged in conversation**: don't let the "trivial" criterion
accidentally strip context immediately preceding a rare event (e.g. the
straight-line run-up right before a tackle). Mitigate by additionally
requiring "not within K steps of a `kick_this_tick=1` or
`tackle_attempt=1` row in the same episode" for a row to be eligible for
downsampling at all (K itself exposed as config, e.g.
`downsample_trivial_exclude_radius_steps: 5`). This means the trivial-row
detection pass needs per-episode boundaries (available from the `.npz`
file structure — each file is one batch of episodes; verify episode
boundaries are recoverable, e.g. via a `dones` array, before implementing).

### 5.3 Tests

- Hand-construct a tiny dataset with a known trivial/non-trivial/near-rare-event
  row composition; assert:
  - non-trivial and near-rare-event rows are **never** excluded regardless
    of `frac`.
  - two consecutive epochs' iteration produce **different** excluded sets
    (proving fresh re-roll, not a fixed mask) — but both epochs still see
    100% of non-trivial rows.
  - `pos_weight` computed from a hand-built label array matches a
    hand-computed expected value exactly.
  - `bc_loss_from_tensor(..., pos_weight_kick=W)` produces a loss that,
    for a single positive-labeled kick row, scales linearly with `W`
    (contrast against `W=1.0` baseline).

---

## 6. W5 — Demo recording: both players + `ai_type` label

### 6.1 Both-players recording — exact current code, confirmed by direct read

Confirmed by reading `record_demonstrations.py::record_episodes()` directly
(current file, ~lines 60-190). Key facts, no longer speculative:

- The opponent player's id is the literal string `"opponent"`
  (`match.player_by_id("opponent")` — hardcoded, matches
  `build_1v1_scenario`'s player naming).
- The **trainee** is explicitly driven by `Phase1RulesAI()` during
  recording too (`player.ai = Phase1RulesAI()` right after `env.reset()`
  each episode) — so yes, confirmed: trainee's `ai_type` label during
  demo recording is always `rules`, never anything else. No guessing
  needed for W6.2's "is the trainee always rules during recording" open
  question — **it is, always, unconditionally, in the current script.**
- The opponent is `Phase1RulesAI()` with probability `opponent_rules_prob`
  (a function parameter, sourced from `--opponent-rules-prob` CLI flag /
  `phase1_demo_opponent_rules_prob` config default), else `opp.ai = None`
  (immobile). **There is no neural-opponent branch at all** in this
  function — confirms decision #6 exactly; nothing to change here for
  that.
- **Only `player` (the trainee) gets `on_kick`/`on_tackle` callbacks
  wired** (`player.on_kick = lambda p: _record_now()`,
  `player.on_tackle = lambda p: _record_now()`, immediately after
  `player.ai = Phase1RulesAI()`). **The opponent object (`opp`) never has
  these callbacks set, anywhere in this function.** This is a confirmed,
  real gap — not speculative. Every opponent kick/tackle event during
  recording is currently **silently missed** as an extra high-value
  sample; the opponent only gets the regular time-based
  `_record_now(reward=0.0, done=False)` samples inside the `while not
  done:` loop, at `sample_interval_s` cadence (default 0.2s from
  `record_episodes`, though `ai_config.json`'s `bc.demo_sample_interval_s`
  is documented as 1.0s default — **these two defaults disagree**; the
  function parameter default (0.2s) is only used if no explicit value is
  passed by the caller — check the actual CLI/`__main__` invocation to see
  which one wins in practice, and consider consolidating to read from
  `ai_config.json` only, removing the separate hardcoded `0.2` default in
  the function signature, to avoid future drift between the two).

**Concrete edit plan for `record_episodes()`:**

1. `_record_now()` currently calls `label_fn(env)` unconditionally for
   whichever single player was implicitly baked into the closure via
   `label_fn` (check `bc_label_fn_for_phase(1)` — it likely returns
   `phase1_labels` with `player_id=None`, which internally defaults to
   `env.trainee_player_id` per `phase1_labels`' own signature/body seen
   earlier). **Change `_record_now()` to accept a `player_id` argument**
   and internally build **two** samples, one per player, each time it's
   called for a *timed* sample:
   ```python
   def _record_now(reward: float = 0.0, done: bool = False, player_id: str | None = None):
       """player_id=None means 'record both trainee and opponent' (used for
       timed samples). A specific player_id means 'record only this player'
       (used for kick/tackle callback samples, which fire per-player)."""
       ids = [env.trainee_player_id, "opponent"] if player_id is None else [player_id]
       for pid in ids:
           obs = env._get_obs(player_id=pid)  # NOTE: verify _get_obs() accepts
                                                # a player_id override — check
                                                # ScenarioEnv._get_obs()'s actual
                                                # signature before assuming this;
                                                # it may currently be hardcoded to
                                                # the trainee and need extending.
           label = label_fn(env, player_id=pid)
           ...  # append to the *_feats/bc_labels/rewards/dones lists as before,
                 # plus a new ai_type_labels list (see 6.2)
   ```
2. The two `on_kick`/`on_tackle` callback registrations need a matching
   pair added for `opp`, right where `opp.ai` is assigned:
   ```python
   opp.on_kick = lambda p: _record_now(player_id="opponent")
   opp.on_tackle = lambda p: _record_now(player_id="opponent")
   ```
   Only meaningful when `opp.ai = Phase1RulesAI()` (the immobile branch
   never kicks/tackles, so wiring the callback there is harmless but
   inert — fine to set unconditionally for code simplicity rather than
   branching).
3. **Confirmed by direct read: `ScenarioEnv._get_obs(self) -> ObservationBatch`
   takes NO `player_id` parameter** (`ai/env/scenario_env.py` ~line 453) —
   it's hardcoded to call `encode_observation(..., player_id=self.trainee_player_id, ...)`.
   This **must** be extended:
   ```python
   def _get_obs(self, player_id: str | None = None) -> ObservationBatch:
       pid = player_id if player_id is not None else self.trainee_player_id
       time_remaining = max(0.0, self.max_episode_s - self._episode_ticks * self._dt_s)
       return encode_observation(
           match=self._loop.match,
           player_id=pid,
           time_remaining_s=time_remaining,
           attack_defence_smoothed=self._ema.smoothed,
           rng=self.rng,
       )
   ```
   Default `None` preserves every existing call site's behaviour
   unchanged (there are other internal callers of `self._get_obs()` inside
   `ScenarioEnv.step()`/`reset()` that must keep working exactly as before
   — grep all call sites of `_get_obs(` inside `scenario_env.py` itself
   before editing, not just the recording script, since this is a shared
   internal method).

   Also confirmed: `bc_label_fn_for_phase(1)` returns `phase1_labels`
   directly (from `ai/curriculum/envs.py` line ~28), and `phase1_labels`
   already accepts `player_id: str = None` per the signature read earlier
   in this design conversation — so **no change needed to
   `bc_label_fn_for_phase`/`phase1_labels` themselves**, only to the
   recording script's call site (`label_fn(env, player_id=pid)` instead of
   the current implicit `label_fn(env)`).
4. `np.stack(self_feats)` etc. at the end of `record_episodes()` need no
   change — they'll simply be ~2× longer once both players are appended
   per timed sample. Kick/tackle callback samples are naturally 1×
   per-event, per-player (unchanged in kind, just now possible for the
   opponent too).

This is a genuinely mechanical but delicate change — get `_get_obs`'s
signature right first, then the rest follows directly from the pattern
above.

### 6.2 `ai_type` label field

Extend `BCLabel`/the flat array format (`BC_LABEL_DIM`, currently 16) with
a new field. Since AI-type is categorical (not yet a probability), store
it as a small integer-coded field, e.g. one new float at index 16:
`0.0 = rules`, `1.0 = immobile`, `2.0 = neural` (reserved, unused for now
per decision #6). Bump `BC_LABEL_DIM` to 17, add `_I_AI_TYPE = 16` in
`bc.py` (alongside the existing `_I_SHOOT` .. `_I_EXEC_MOVE` constants —
keep the same naming convention). Update `BCLabel` dataclass with a new
`ai_type: float = 0.0` field, and `to_array()`/the module docstring's
layout table to include it.

**This is a breaking schema change to recorded `.npz` files** — same
category of change as the earlier `PLAYER_FEATURE_DIM` 26→28 break
mentioned in `ai_trainer_knowledge.md`. Follow the same convention:
document it prominently, and **all existing `demonstrations/phase1/*.npz`
files must be re-recorded** after this change. Concretely, add a guard in
`DemonstrationDataset.from_file()` (in `ai/bc/dataset.py`):
```python
labels = data["bc_labels"]
if labels.shape[1] != BC_LABEL_DIM:
    raise ValueError(
        f"{path}: bc_labels width {labels.shape[1]} != expected "
        f"BC_LABEL_DIM={BC_LABEL_DIM}. Re-record demonstrations "
        f"(see human_trainer_cheatsheet.md 're-record demonstrations')."
    )
```
`DemonstrationDataset` currently has **no such guard at all** (confirmed —
`from_file()` just does `data["bc_labels"]` directly with no shape check),
so right now a stale-schema `.npz` would load successfully with silently
misaligned columns — this is a real, pre-existing latent bug, not
hypothetical. Fix it as part of this workstream regardless of whether you
land the `ai_type` field in the same PR — it protects every future schema
change, not just this one.

**Where the label comes from — confirmed, not speculative** (see 6.1):
during recording, the **trainee is always `Phase1RulesAI`**, so its
`ai_type` label is always `0.0` (rules). The **opponent** is
`Phase1RulesAI` with probability `opponent_rules_prob`, else immobile
(`opp.ai = None`) — read this directly off the same branch in
`record_episodes()` that sets `match._opponent_use_rules_ai` /
`match._opponent_is_immobile` (lines ~150-163 in the current file), e.g.:
```python
trainee_ai_type = 0.0  # always rules, per the unconditional player.ai = Phase1RulesAI() above
opponent_ai_type = 0.0 if match._opponent_use_rules_ai else 1.0  # rules : immobile
```
Pass the correct value into `_record_now(..., ai_type=...)` per-player (or
have `_record_now` look it up itself from a small `{player_id: ai_type}`
dict built once per episode, cleaner than threading it through every call
site).

### 6.3 Is recording from a neural AI already possible?

**No** — deferred per your instruction ("forget the neural AI thing").
Do not implement. For future reference (not part of this workstream): a
neural AI produces no `Order` object `phase1_labels()` can translate into
a `BCLabel` — recording from one would require a different mechanism
(self-distillation/pseudo-labeling from a frozen checkpoint's sampled
actions, or reward-only trajectory recording without BC labels). Leave a
short note in `ai/knowledge.md` under the demo-recording section flagging
this as future scope, so it isn't silently forgotten.

### 6.4 Tests

- Extend/add to a recording-focused test (check if one exists under
  `tests/ai_unit` or `tests/ai_scenario` currently covering
  `record_demonstrations.py` — if none exists, that's another W1-style gap,
  add a minimal one): run a tiny recording session (few episodes), assert
  the resulting `.npz` contains **2× the row count** relative to a
  trainee-only baseline of the same episode count, and that `ai_type`
  values are present and take only the expected `{0.0, 1.0}` values (never
  `2.0`, since neural recording is deferred).
- `DemonstrationDataset` loading: assert loading a `.npz` with the old
  16-wide label format fails with a clear error (not a silent shape
  mismatch) once `BC_LABEL_DIM` becomes 17 — this directly tests the
  "guard against schema breaks" note in 6.2.

---

## 7. W6 — Value-only opponent-AI-type side-channel

This is the highest-effort, highest-risk workstream. Do it only after W1
(tests exist to catch regressions) and W5 (dataset actually has the labels
to train against).

### 7.1 Architecture (per decision #4, final agreed design)

- **Not** a `PlayerFeatures` field. **Not** routed through the entity
  encoder or attention at all. A genuinely separate flat input that
  bypasses the shared trunk's policy-facing path entirely.
- Self gets its own one-hot (`self_ai_type`, 3 floats: `is_immobile`,
  `is_neural`, `is_rules`) since self can be rules-based during BC/value
  pretrain (per your requirement — 22 slots total: self + 21 others).
- Others get a `(21, 3)` array, `other_ai_type`, permuted with the **exact
  same permutation index array** already computed for shuffling
  `other_feat`/`exists_mask` in `encode_observation()` — reuse the existing
  `perm` variable, do not compute a second independent permutation (that
  would desynchronize which AI-type entry corresponds to which actual
  player slot — a correctness-critical detail, get this line-level right).
- Flatten `self_ai_type (3,) ++ other_ai_type (21,3)→(63,)` = 66 floats,
  fed through a new small `value_extra_mlp` (e.g. `Linear(66, 16) → ReLU`),
  concatenated **only** into `value_head`'s input:
  ```python
  value_input = torch.cat([h, self.value_extra_mlp(ai_type_flat)], dim=-1)
  self.value_head = nn.Linear(trunk_hidden + 16, 1)
  ```
  All other heads (`shoot_logit`, `pass_logit`, ..., `latent_vector`)
  continue reading from `h` alone, completely unchanged, never touching
  `ai_type_flat`.

### 7.2 Where this needs plumbing

- `ObservationBatch` (schema.py): add `self_ai_type: np.ndarray` (shape
  `(3,)`) and `other_ai_type: np.ndarray` (shape `(21, 3)`), plus
  `to_torch_dict()` entries.
- `encode_observation()`: needs to know each player's actual controlling AI
  type (immobile/neural/rules) at encode time — this info must already be
  derivable from `match`/`player.ai` (check `player.ai is None` →
  immobile-ish, `isinstance(player.ai, Phase1RulesAI)` → rules,
  `isinstance(player.ai, NeuralPlayerAI)` → neural — verify these are the
  actual class names/import paths before writing this, don't assume).
  **This must reuse the exact `perm` array** already built for
  `other_feat`'s shuffle — read the current shuffling code path carefully
  and extend it to also permute `other_ai_type`, in the same function, in
  the same statement block, so it's structurally impossible for them to
  desync in a future refactor (e.g. compute both permuted arrays from the
  same `perm` variable in adjacent lines, ideally via a shared helper
  rather than copy-pasted indexing logic).
- `DecisionNetwork.forward()`: new required parameters
  `self_ai_type: torch.Tensor, other_ai_type: torch.Tensor`. Every call
  site in `ppo_trainer.py` (confirmed ~15+ call sites via grep) needs
  updating to pass these through — this is the largest mechanical part of
  this workstream, budget real time for it, and do it via a careful
  find-all-call-sites pass, not by eye.
- `augment.py`: `self_ai_type`/`other_ai_type` need to travel through the
  **same slot-permutation augmentation** as `other_feat` (exact analog to
  the existing MAX_OTHER_PLAYERS-shaped fields) — but need **NO**
  geometric-flip sign handling (not a positional/directional quantity,
  belongs in the "pass-through unchanged under flip_x/flip_y" bucket).
  Read `augment.py`'s exact mechanism (field-index derivation from
  dataclass fields at import time) — since `other_ai_type` is NOT a
  `PlayerFeatures` field, it won't be auto-discovered by that mechanism;
  it needs **explicit, manual handling** in `augment_batch()` wherever
  `other_feat`'s permutation is applied, adding one more array to the
  "permute alongside" list. This is a real code change to `augment.py`,
  not automatic — do not assume it "just works" because it looks similar
  to existing per-entity fields.
- `pretrain_combined()` / `pretrain_value()` / `_ppo_update()` / anywhere
  else in `ppo_trainer.py` that constructs observations for live rollout:
  self is **always** the "neural" one-hot in these contexts (decision #5).
  The opponent's AI type must be read from the **real, live**
  `match._opponent_use_rules_ai` / `_opponent_is_immobile` flags at each
  step, not assumed constant — this varies per-episode based on the
  existing curriculum config.
- BC pretraining path (`pretrain_combined()`'s Phase 0/1, reading from
  `DemonstrationDataset`): `self_ai_type`/`other_ai_type` come from the
  **recorded** `ai_type` label (W5) for whichever perspective that row was
  recorded from — this is why W5 must land first.

### 7.3 Tests (mandatory, this is the highest-risk workstream)

- **Gradient-isolation test (the most important new test in this entire
  plan)**: build a tiny forward pass, set `other_ai_type`/`self_ai_type`
  to `requires_grad_(True)`, compute `shoot_logit.sum()` (or any pure
  policy head), call `.backward()`, and assert
  `self_ai_type.grad is None or (self_ai_type.grad == 0).all()` and same
  for `other_ai_type.grad`. Then repeat for `value_head`'s output and
  assert the gradient IS non-zero there. This is a hard architectural
  guarantee — write it once, keep it forever, treat any future failure of
  this test as a release-blocking regression.
- Permutation-consistency test: encode an observation with 2+ distinct
  other-player AI types (e.g. one rules, one immobile), run the encoder
  twice with different internal shuffle RNG seeds (or directly call the
  permutation logic with two different `perm` arrays), and assert that
  wherever a given real player lands in `other_feat`, that exact player's
  AI-type one-hot is found at the same slot index in `other_ai_type` —
  i.e. verify by checking a *correlated* feature (e.g. `is_own_team` or
  `pos_x`, which already exist and are known-correctly-permuted) lines up
  with the AI-type slot for the same underlying player across multiple
  shuffles.
- Forward-pass shape test in `test_networks.py`: assert `value_head`'s
  input dimension is `trunk_hidden + value_extra_mlp_output_dim` and the
  network builds/runs without shape errors end to end.
- Augmentation test: assert `other_ai_type` is permuted identically to a
  known-correct existing field (e.g. `is_own_team` from `other_feat`)
  under `augment_batch()`'s slot-shuffle variants, and is **unchanged**
  (not sign-flipped) under `flip_x`/`flip_y` variants.

### 7.4 Knowledge-file note (required)

Add to `ai/knowledge.md`:

> **Opponent-AI-type (value-only)**: `self_ai_type`/`other_ai_type` are
> flat one-hot side-channels feeding ONLY `value_head`, bypassing the
> entity encoder/attention and all policy heads entirely — this is a
> deliberate architectural choice (the policy must never condition on
> opponent identity; only the critic needs it for calibration). Any change
> that routes this data through the shared trunk or `PlayerFeatures`
> reintroduces exactly the problem this design avoided — do not "simplify"
> this into a `PlayerFeatures` field without re-reading
> `ai_design_plan_v2.md` section 7's gradient-isolation test rationale
> first. During live PPO/eval, self's AI-type is always "neural"; demo/BC
> pretrain data supplies genuine self AI-type per the recorded episode
> (see W5 note on the recording format).

---

## 8. W7 — Pre-training refactor

### 8.1 Phase 0 (demo value pretrain) — unfreeze + combine with decision-head BC

Current (`pretrain_combined()`, roughly lines 515-535 in `ppo_trainer.py`):
separate `_demo_val_opt` scoped to `value_head.parameters()` only, backward
pass computes (and discards, since not stepped) gradients on trunk params
too.

New behaviour (per decisions #7/#13):
```python
demo_opt = torch.optim.Adam(
    list(self.decision_net.parameters()),  # ALL decision_net params, incl value_head
    lr=self._demo_value_pretrain_lr, eps=1e-5,
)
# NOTE: execution_net is intentionally NOT included here — Phase 0 is a
# decision-network-only high-level warm-up per your request; execution-net
# training still happens later in the main BC epoch loop (Phase 1).
```
Each Phase-0 minibatch:
```python
d_heads = self.decision_net(sf, of, em, bf, gf, self_ai_type, other_ai_type)
v_dec = d_heads.value.squeeze(-1)
value_loss = F.mse_loss(v_dec, ret_norm)   # note: execution_net's v_exc term REMOVED
                                             # from this stage since exec_net isn't
                                             # trained here at all now
dec_bc_loss, _ = bc_loss_from_tensor_decision_only(labels, d_heads)  # new helper —
                                             # see note below
combined = dec_bc_loss + self._phase0_value_coef * value_loss
demo_opt.zero_grad()
combined.backward()
nn.utils.clip_grad_norm_(list(self.decision_net.parameters()), self.max_grad_norm)
demo_opt.step()
```

**New helper needed**: `bc_loss_from_tensor()` currently always requires
`exec_heads` (it unconditionally computes `exec_bce_loss`,
`dir_loss_per`). Add a decision-heads-only variant — either a new
parameter `exec_heads: Optional[...] = None` that skips all
exec-dependent terms when `None`, or a separate smaller function
`decision_bc_loss_from_tensor()` that only computes the `dec_loss`
component. **Prefer the `Optional` parameter approach** — keeps one
function as the source of truth for the decision-head loss math instead of
duplicating the `_bce(...)` calls for the 7 Bernoulli heads in two places.

**Since `value.squeeze(-1)` previously averaged `decision_net.value` and
`execution_net.value`, and Phase 0 no longer runs `execution_net` at all,
the value target/prediction here is `decision_net.value` alone.** This is
a real, intentional change to what Phase 0 fits against — document this
clearly in the log line and in a code comment, since later stages
(Phase 2/3 `pretrain_value`) *do* average both networks' value heads, so
there's an inherent inconsistency in what "the value prediction" means
between Phase 0 and later phases. This is acceptable (Phase 0 is explicitly
scoped as "decision-network-only high-level warm-up" per your framing) but
must not be silently forgotten — a future reader could reasonably expect
both phases to fit the same target.

New config key: `phase0_value_coef` (final name via W8 convention check),
default suggestion `1.0` (equal weighting to start; expose and let you tune
empirically per the project's existing pattern of exposing everything).

### 8.2 Refactor: `pretrain_combined()` calls `pretrain_value()`

Currently `pretrain_combined()`'s "Phase 2/3" (collect rollout → GAE →
epoch loop fitting value heads) duplicates most of what
`pretrain_value()` already does standalone. Extract the shared logic into
`pretrain_value()` itself so `pretrain_combined()` just calls it:

```python
# Inside pretrain_combined(), replacing the current inline
# rollout-collection + GAE + value-epoch-loop block:
self.pretrain_value(
    env,
    n_steps=rollout_steps,
    n_epochs=value_epochs,
    lr=value_lr,
)
```

Requirements for this to be a clean drop-in:
- `pretrain_value()` must apply the same augmentation (`augment_batch`)
  currently applied inline in `pretrain_combined()`'s Phase 2/3 — check
  whether the **standalone** `pretrain_value()` currently applies
  augmentation at all (a quick read of its body earlier in this
  conversation did not show an `augment_batch` call) — if it doesn't, add
  it there, so behaviour is preserved for `pretrain_combined()` callers and
  gains augmentation for direct standalone callers too (strictly an
  improvement, not a behaviour regression, but confirm no direct caller
  relies on unaugmented behaviour first — check `train.py`'s fallback
  `else` branch usage).
- The BC-degradation check (comparing BC loss before/after value warm-up)
  currently sits inline in `pretrain_combined()` right after the value
  epoch loop — **keep this check in `pretrain_combined()`**, called right
  after the `self.pretrain_value(...)` call, since it's specifically about
  verifying BC didn't regress *in the combined-pretraining context* and
  doesn't belong in the standalone value-only function.
- `pretrain_value()`'s existing trunk-freezing behavior
  (`_get_value_pretrain_freeze_params()`) is **retained** for this call —
  it's a distinct, separate stage from Phase 0 (which per decision #13 has
  freezing removed). Freezing here is unaffected by the Phase 0 change;
  only Phase 0 itself loses freezing. Do not accidentally remove freezing
  from `pretrain_value()` while implementing the Phase 0 change — these
  are two different call sites with two different freezing decisions, keep
  them straight.

### 8.3 Reward/win-rate logging during `pretrain_value()`'s rollout

Currently the rollout-collection loop inside `pretrain_value()` (and the
soon-to-be-removed duplicate in `pretrain_combined()`) only accumulates
`buffer` for GAE — `reward`/`done`/`info` are read but never aggregated
for logging. Add the same style of tracking the main PPO rollout loop
already does (per-rollout log line mentions `vs_rules(N): win%/opp%` /
`vs_neural(N): win%/opp%` — reuse that exact logic/formatting, don't invent
a new format):

```python
episode_returns: list[float] = []
outcomes_vs_rules: list[str] = []
outcomes_vs_neural: list[str] = []
episode_accum = 0.0
for _ in range(n_steps):
    next_obs, reward, done, info = env.step()
    episode_accum += reward
    if done:
        episode_returns.append(episode_accum)
        episode_accum = 0.0
        (outcomes_vs_rules if info.is_rules_episode else outcomes_vs_neural).append(info.trial_outcome)
        env.reset()
...
log.info(
    f"  [value pretrain rollout] mean_return={np.mean(episode_returns):.2f} "
    f"({len(episode_returns)} episodes)  "
    f"vs_rules({len(outcomes_vs_rules)}): {win_frac(outcomes_vs_rules):.0%}  "
    f"vs_neural({len(outcomes_vs_neural)}): {win_frac(outcomes_vs_neural):.0%}"
)
```
Find the exact `win_frac`-equivalent helper already used by the main PPO
rollout log line in `ppo_trainer.py` and reuse it rather than
reimplementing win-rate computation a third time in this file.

### 8.4 Tests

- Extend the W1 smoke test (section 2.3) to additionally assert the new
  log-worthy statistics are computed without error (call the refactored
  functions, don't just check they don't crash — actually assert
  `len(episode_returns) > 0` for a rollout long enough to complete at least
  one episode).
- Unit test for the decision-only BC loss path: verify
  `bc_loss_from_tensor(labels, d_heads, exec_heads=None)` (or whichever
  API shape is chosen) returns a loss computed **only** from the 7
  Bernoulli decision heads + categorical/continuous decision heads,
  excluding all `exec_*` terms — compare against manually zeroing out the
  exec contribution in the full-signature call and assert equality.
- Regression test confirming Phase 0's optimizer now includes trunk/encoder
  parameters (inspect `demo_opt.param_groups` and assert e.g.
  `self.decision_net.entity_encoder.parameters()` are present — this
  directly guards against the freezing regression being silently
  reintroduced).

---

## 9. W8 — Config exposure / naming

Match existing `ai_config.json` naming conventions exactly (e.g.
`phase1_demo_opponent_rules_prob`, `demo_value_pretrain_lr`,
`value_pretrain_frozen_layers` — snake_case, phase-prefixed where
phase-specific, `demo_`-prefixed where specific to the demo-data path).

New keys to add (final list, consolidating everything above):

```json
"observation": {
  "max_task_ids": 20
},
"curriculum": {
  "phase1_demo_opponent_immobile_prob": 0.0
},
"bc": {
  "phase0_value_coef": 1.0,
  "pos_weight_kick": null,
  "pos_weight_tackle_attempt": null,
  "_comment_pos_weight": "null = auto-compute from dataset at load time (recommended). Set a float to override.",
  "downsample_trivial_enabled": true,
  "downsample_trivial_frac_default": 0.5,
  "downsample_trivial_frac_high_epoch": 0.65,
  "downsample_trivial_epoch_threshold": 5,
  "downsample_trivial_cos_threshold": 0.98,
  "downsample_trivial_exclude_radius_steps": 5
}
```

Note `phase1_demo_opponent_immobile_prob` mirrors the existing
`phase1_demo_opponent_rules_prob: 1.0` key exactly in naming style — read
by `record_demonstrations.py`'s `--opponent-rules-prob`-equivalent CLI
plumbing; verify whether an `--opponent-immobile-prob` CLI flag needs
adding alongside the existing `--opponent-rules-prob` flag mentioned in
`ai_trainer_knowledge.md`'s quick-start section, or whether config alone
is sufficient (check current CLI arg parsing in `record_demonstrations.py`
before deciding).

`pos_weight_kick`/`pos_weight_tackle_attempt` default to `null` meaning
"auto-compute from the training dataset" (per W4's `compute_pos_weights()`)
— only override manually for debugging/ablation. Document this null-means-auto
convention clearly since it differs from every other numeric key in this
file (which all have concrete numeric defaults, not `null`).

---

## 10. Cross-cutting pitfalls (read before starting any workstream)

1. **Schema breaks are not new to this project** — `PLAYER_FEATURE_DIM`
   already broke once (26→28, documented in `ai_trainer_knowledge.md`).
   W5's `BC_LABEL_DIM` bump (16→17) is another one. Every schema break
   requires: (a) a loud failure mode if old data is loaded against new
   code (don't let it silently misalign columns), (b) an explicit
   re-recording step for `demonstrations/phase1/`, (c) invalidating old
   `.pt` checkpoints that assumed the old input dimensions (network
   architecture changes in W6/W3 also invalidate checkpoints — this is
   expected, not a bug, but should be called out in any run-log or
   commit message when it happens).
2. **Permutation synchronization** is the single most error-prone part of
   this entire plan (W6 especially). Any new per-other-player array
   (`other_ai_type`) must be permuted using the exact same `perm` variable
   as `other_feat`/`exists_mask`, computed once, applied to all of them in
   the same code block. Do not recompute a "new" permutation anywhere for
   a related array — that's exactly the bug class already documented as
   previously fixed for `pass_target`/`tackle_target`/`mark_target` (see
   `ai/knowledge.md`'s existing warning about inverse-permutation
   remapping) — this plan introduces a new instance of the same risk
   category.
3. **`ppo_trainer.py` has 15+ call sites** constructing `decision_net(...)`
   /`execution_net(...)` forward passes. Any signature change (W6 adding
   `self_ai_type`/`other_ai_type` params) touches all of them. Do this as
   one focused, mechanical pass with a full-file re-read afterward, not
   scattered edits — a missed call site is a silent `TypeError` at best,
   or (if a default value is added to avoid the `TypeError`) a silently
   wrong default at worst.
4. **Don't conflate Phase 0's freezing removal with `pretrain_value()`'s
   freezing** — they are different functions with different intended
   freezing behavior (Phase 0: no freezing per decision #13;
   `pretrain_value()` standalone: freezing retained, per section 8.2's
   explicit call-out). Keep these decisions from bleeding into each other
   during the refactor.
5. **Test-first discipline for `bc.py`/`ppo_trainer.py`/`DemonstrationDataset`
   specifically** — these three files have zero pre-existing coverage.
   W1 must land and pass against unmodified code before any other
   workstream edits them, so regressions are attributable to the actual
   change under test, not ambiguous "was this always broken?" uncertainty.

---

## 11. Open items explicitly deferred (not part of this plan)

- Mixed multi-phase rollout training (task-id is scaffolded per W3, but
  actual orchestration is future work).
- Recording demonstrations from a neural AI (self-distillation /
  pseudo-labeling) — explicitly deferred per your instruction.
- BC repair (`bc_repair_epochs`) — left alone, already disabled.
- 2x-forward-pass hard isolation for opponent-AI-type — superseded by the
  cheaper side-channel design in W6; not needed.
