# src/footballcoach/ai/

Neural-network / PPO player AI.  The engine, entities, orders, actions, and
config packages are all complete and unchanged - this package is a pure
*consumer* of them.  See `ai_design_doc.md` in the repo root for the full
architecture specification; this file is the operational knowledge note.

## Install

```bash
uv sync --group ai   # pulls torch; base game install stays lightweight
```

## Package layout

```
ai/
  config/
    ai_config.json    # all network sizes, PPO hyperparams, reward coefficients
    __init__.py       # load_ai_config() with lru_cache (mirrors config/loader.py)
  obs/
    schema.py         # PlayerFeatures, BallFeatures, GlobalFeatures dataclasses
                      # + PLAYER_FEATURE_DIM / BALL_FEATURE_DIM / GLOBAL_FEATURE_DIM
                      # + MAX_OTHER_PLAYERS (21) + ObservationBatch
    encoder.py        # encode_observation(match, player_id, time_remaining_s) -> ObservationBatch
  action/
    schema.py         # DecisionHeadsRaw, DecisionAction, ExecutionHeadsRaw, ExecutionAction
    distributions.py  # IndependentBernoulli, MaskedCategorical, SquashedNormalHead, DirectionHead
    gating.py         # select_action() - pure Python winner-take-all, NEVER in gradient graph
    to_orders.py      # Execution outputs -> DIRECT player physics (NO ORDERS - see below)
  models/
    entity_encoder.py    # shared per-entity MLP + nn.MultiheadAttention
    decision_network.py  # DecisionNetwork.from_config() + derive_get_possession_prob()
    execution_network.py # ExecutionNetwork.from_config() + flatten_decision_heads()
  ppo/
    rollout_buffer.py  # RolloutBuffer.add() / compute_gae() / as_tensors() / clear()
    schedules.py       # LR, clip-range, rng_reduction schedules (progress 0→1)
    ppo_trainer.py     # PPOTrainer.from_config() + .train(env, total_steps)
  env/
    reward.py          # phase1_reward(), phase2_reward(), EMAFilter (attack/defence)
    scenario_env.py    # ScenarioEnv: Gym-like wrapper over ScenarioDefinition + ScenarioLoop
  bc/
    __init__.py
    dataset.py         # DemonstrationDataset: load .npz files, iterate_minibatches(), sample_batch()
  curriculum/
    phases.py          # CurriculumPhase dataclasses + PHASES_BY_ID dict
    envs.py            # build_env(phase) + bc_label_fn_for_phase(id) — ONE source of truth,
                       # imported by both train.py and record_demonstrations.py
    opponent_pool.py   # OpponentPool + apply_rules_based_opponent()
  scripts/
    train.py           # CLI: uv run python -m footballcoach.ai.scripts.train --phase 1
                       #   --bc-dataset demonstrations/phase1/  (offline BC pre-training)
                       #   --bc-pretrain-epochs N  --bc-pretrain-batch-size N
    evaluate.py        # CLI: ... evaluate --checkpoint path.pt --n-trials 100
    record_demonstrations.py  # CLI: record rules-based episodes as .npz BC datasets
```

## BC label vector (BC_LABEL_DIM = 17)

`bc.py` stores 17 floats per step (see the module docstring in `bc.py` for
the authoritative up-to-date layout table — do not let this count drift out
of sync, it has already changed twice: 15→16 (added `exec_move`), 16→17
(added `ai_type`)):

| idx | field | source |
|-----|-------|--------|
| 0–6 | decision Bernoullis (shoot, pass, move, tackle, gp_extra, mark, hold) | rules AI decision |
| 7–8 | move_dir_x/y | direction toward target |
| 9 | sprint | rules AI order |
| 10–11 | move_region_x/y (metres) | MoveOrder target position |
| 12 | **kick_this_tick** | `player.on_kick` callback (fired by engine) |
| 13 | **tackle_attempt** | `player.on_tackle` callback (fired by engine) |
| 14 | valid | 1.0 = use this label |
| 15 | exec_move | 1.0 = player is moving, 0.0 = standstill |
| 16 | ai_type | 0.0=rules, 1.0=immobile, 2.0=neural (reserved, unused) |

**Critical:** indices 12–13 come from **engine callbacks** (`Player.on_kick`,
`Player.on_kick`), not from inspecting the current order type.  They reflect
what the engine physically executed this tick.  Indices 0–11 come from asking
the rules AI what it *decides* next — these are input to the decision network
and movement-related execution heads.

### BC class balancing (pos_weight + trivial-row downsampling)

`DemonstrationDataset.compute_pos_weights()` computes inverse-frequency
`pos_weight` values (matching
`F.binary_cross_entropy_with_logits`'s `pos_weight` semantics) for the rare
`kick_this_tick`/`tackle_attempt` Bernoulli targets, over the dataset's
valid rows. `PPOTrainer` auto-computes these from the training dataset at
the start of `pretrain_combined()` unless overridden via
`ai_config.json['bc']['pos_weight_kick']` /
`['pos_weight_tackle_attempt']` (non-null = explicit override). Threaded
into **every** `bc_loss_from_tensor()` call site (pretrain BC epochs, BC
repair, the post-value-warmup BC degradation check, and the annealed
BC-aux-during-PPO loss) — never applied to the raw PPO policy-gradient
loss (reweighting that would be a correctness risk, not just variance).

`DemonstrationDataset.iterate_minibatches(..., downsample_trivial_frac=...)`
gently excludes a fraction of "trivial" movement rows (rows whose
`move_direction` label is nearly identical to the *previous row in the same
episode*, cosine similarity above `downsample_trivial_cos_threshold`) each
epoch. The trivial classification is cached once at load time
(`_compute_trivial_mask()`), but the actual excluded subset is a **fresh
random draw every call** (i.e. every epoch) — never a fixed one-time
filter. Rows within `downsample_trivial_exclude_radius_steps` of a
`kick_this_tick`/`tackle_attempt` event (same episode) are never eligible
for exclusion, so run-ups immediately preceding a rare event are preserved.
Controlled by `ai_config.json['bc']`: `downsample_trivial_enabled`,
`downsample_trivial_frac_default` (used for early epochs),
`downsample_trivial_frac_high_epoch` (used once
`epoch >= downsample_trivial_epoch_threshold`).
`DemonstrationDataset.downsample_trivial_stats()` reports the trivial-row
count/fraction and the expected exclusion count at a given `frac` (reuses
the same cached `_trivial_mask_cache`); `pretrain_combined()` logs this once
per BC epoch (`Downsample trivial rows (epoch N): X/Y (Z%) ... excluding
~W this epoch`) when `downsample_trivial_enabled` is true. Deliberately **not**
applied to `iterate_minibatches_with_returns()` — this iterator is now only
used standalone (kept for any future direct callers); `pretrain_combined()`'s
Phase 0 uses `iterate_minibatches(..., returns=...)` instead (see "Phase 0"
note below). `iterate_minibatches_with_returns()` also gained a `valid_only`
parameter (default `False`, preserving old behaviour) — value-target
fitting benefits from seeing the full return distribution including
"boring"/invalid-BC-label states, so it is deliberately NOT combined with
trivial-row downsampling (which is specifically about reducing redundant
*BC* signal).

### Pre-training phases (`pretrain_combined()` / `pretrain_value()`)

`PPOTrainer.pretrain_combined()` runs, in order:

- **Phase 0** — decision-network warm-up on demo returns. ONE combined
  backward pass per minibatch: `decision_bc_loss + phase0_value_coef *
  value_loss`. The optimizer covers ALL of `decision_net`'s parameters
  (encoders + trunk; `decision_net.value_head` itself stays frozen — single
  value head convention, see "Single value head convention" below) PLUS
  `execution_net.value_head` ONLY. `execution_net` still runs a forward pass
  every minibatch (needed to produce `e_heads.value` from `d_heads`), but no
  other `execution_net` output (move/sprint/kick/tackle heads etc.) is used
  or optimized here — those get their BC training in Phase 1. Uses
  `bc_loss_from_tensor(bc_labels, d_heads, exec_heads=None, ...)` for the
  decision side — the decision-heads-only path (skips exec_move/sprint/
  kick/tackle_attempt BCE and the move_direction cosine loss; see the
  `bc_loss_from_tensor()` docstring in `bc.py`) — and a plain
  `F.mse_loss(e_heads.value.squeeze(-1), ret_batch)` (variance-normalized)
  for the value side, i.e. `execution_net.value_head` is the single live
  critic trained here, consistent with the rest of the codebase (Phase 1,
  `pretrain_value()`, PPO). Per-epoch log line reports `loss=`, `dec_bc=`,
  and `val=` separately. Config: `demo_value_pretrain_epochs`,
  `demo_value_pretrain_lr`, `demo_value_pretrain_gamma`,
  `phase0_value_coef` (default 1.0). Skipped if the dataset has no reward
  data or `demo_value_pretrain_epochs=0`.
- **Phase 1** — BC epochs over the full dataset (all params of both
  networks), optionally with a joint value-MSE term if
  `bc_value_coef > 0` (config key `bc.bc_value_coef`; falls back to
  `demo_value_bc_coef` for backward compat).
- **Phase 2/3** — delegates to `self.pretrain_value(env, n_steps=rollout_steps,
  n_epochs=value_epochs, lr=value_lr, batch_size=batch_size)` instead of
  duplicating rollout-collection + GAE + value-epoch-loop logic inline (this
  used to be duplicated — the duplication was the root cause of a past
  `pretrain_value()`-vs-`pretrain_combined()` drift bug). `pretrain_value()`
  freezes trunk/encoder layers during this call (via
  `_get_value_pretrain_freeze_params()`) — a **different, deliberate**
  freezing decision from Phase 0 above. Do not conflate the two: Phase 0
  freezing was removed on purpose; `pretrain_value()`'s freezing was kept on
  purpose. `pretrain_value()` also applies the same `augment_batch()`
  augmentation as before (now inside the shared function, so standalone
  callers get it too) and returns a diagnostics dict
  (`episode_returns`, `outcomes_vs_rules`, `outcomes_vs_immobile`,
  `outcomes_vs_neural`) logged as a `vs_rules(N): win%` line matching the
  main PPO rollout log format.
- BC degradation check, then optional BC repair epochs (unchanged).

### Single value head convention

There are two `value_head` modules in the codebase (`decision_net.value_head`
and `execution_net.value_head`) for historical/checkpoint-compat reasons, but
only ONE is ever trained or read: **`execution_net.value_head`**.
`decision_net.value_head`'s parameters are permanently frozen in
`PPOTrainer.__init__` (`requires_grad_(False)`) and excluded from every value
loss and from `_get_value`/`_sample_action`. Previously the two heads were
fit independently (or averaged) depending on call site while inference used
`(d_val + e_val) / 2` — a train/inference mismatch that let the heads
silently diverge. All call sites now use `e_heads.value.squeeze(-1)` /
`e_heads.value.mean()`: Phase 0 (BC + `execution_net.value_head` MSE — see
"Pre-training phases" below), Phase 1 joint BC+value, `pretrain_value()`,
the PPO update (`_ppo_update`), `_sample_action`, `_get_value`, and the
BC-repair diagnostic.

Per-rollout PPO logging now also prints `[V=mean±std R=mean±std
adv=mean±std]` (value/return/advantage stats), with a DEBUG-level
per-minibatch block showing the d_val/e_val split (d_val is static now that
it's frozen) — useful for spotting value/return miscalibration at a glance.

### Possession gain/loss reward: real turnovers only

`ScenarioEnv.step()` scans every engine tick within a decision interval (not
just before/after the whole interval) to count possession transitions, via
the shared `_possession_transition_step()` static method — this catches
gain+loss pairs that both happen inside one interval (e.g. tackle then
immediately re-tackled), which a simple before/after bool comparison would
miss. `reward.py`'s `phase1_reward()` takes `gained_possession_this_step` /
`lost_possession_this_step` as `bool | int` COUNTS and multiplies the reward
coefficient by the count rather than gating on truthiness.

Critically, a tick-level "loss" is only counted as a **real turnover** if
possession actually settles onto a DIFFERENT player — not simply because
`ball.possessed_by` transiently reads `None` (loose ball in flight, e.g.
during a push-kick dribble touch) or because the SAME player re-collects it
themselves. `_possession_transition_step()` implements this as a small state
machine per tracked player (trainee, and each secondary player):
- `poss_prev` — do I currently have the ball?
- `pending_loss` — I just lost it, but no one else has grabbed it YET (ball
  loose/in-flight); counting is deferred.
- Only transitions `pending_loss` → a counted loss once a DIFFERENT player
  (not me, not `None`) is confirmed to hold the ball. If I re-gain it directly
  out of `pending_loss` with nobody else ever touching it, the pending loss is
  cancelled silently — no lost-count AND no fresh gained-count (the trainee
  never really lost it in the adversarial sense).
State (`_trainee_pending_loss` / `_sec_pending_loss` dicts) persists across
`step()` calls, not just within one, mirroring `_trainee_had_possession_last_step`
/ `_sec_had_possession_last_step`.

### Player event callbacks

`Player` has two optional callbacks set on the instance:
```python
player.on_kick    = lambda player: ...   # fired when KickOrder/ShootOrder/PassOrder executes
player.on_tackle  = lambda player: ...   # fired when ChaseTackleOrder makes contact
```
The engine fires these in `match.py` at the exact tick the action executes —
not when the order is set.  Useful for: BC recording, UI effects, logging,
statistics.  Both default to `None` (no-op).

### Demonstration recording (`record_demonstrations.py`)

Sampling strategy:
- `env.step()` is called for each 0.5s decision interval — episodes terminate
  correctly (box-possession terminal, timeout) because `ScenarioEnv.step()`
  handles those checks.
- Inside each `env.step()` call, the engine fires `player.on_kick` /
  `player.on_tackle` callbacks at the exact physics tick the action executes.
  These callbacks record an extra (obs, label) sample immediately.
- Net result: one regular sample per 0.5s + one extra sample per kick/tackle
  event. ~7k steps for 200 phase-1 episodes (~7s to record).
- Reward wiring: kick/tackle callback samples used to hardcode `reward=0.0`,
  silently dropping real reward (e.g. `gain_possession_bonus`) that fired on
  exactly that tick. Fixed via a mutable `_pending_reward` cell:
  `_record_now(reward=None, ...)` (the callback default) consumes and clears
  `_pending_reward[0]`; the main loop accrues `_pending_reward[0] +=
  float(_reward)` after every `env.step()`, so reward is never double-counted
  or dropped regardless of when a kick/tackle callback fires relative to a
  timed sample.
- Periodic logging (every 10 episodes, or at the final episode) now also
  prints a full reward-component breakdown line (`REWARD_COMP_LABELS` from
  `ppo_trainer.py`, accumulated from `env.last_reward_components` after every
  `env.step()` and reset after each log line) — mirrors `train.py`'s
  pre-training reward diagnostic (`_comp_acc` pattern) so demo-recording
  reward shaping can be sanity-checked the same way.

## Critical design rules

### !!!! CRITICAL: THE NEURAL NETWORK NEVER ISSUES ORDERS !!!!

The neural network does NOT set `player.current_order` to anything, ever.
Orders (`MoveOrder`, `GetPossessionOrder`, `MarkOrder`, `ChaseTackleOrder`,
etc.) are used **only** by:
- The rules-based AI (`Phase1RulesAI`, `StagedGoalkeeper`, etc.)
- Human input in the UI

The **only** things `apply_action_to_player()` in `to_orders.py` does:
1. **Movement**: sets `player.desired_direction` (Vector3) and
   `player.desired_speed_mode` (SpeedMode) directly from `gating.move_direction`
   and `gating.sprint`. The engine's `_apply_movement()` loop reads these fields.
2. **Kick**: calls `player.kick_direct(match, aim_pt, power, spin)` if
   `gating.kick_this_tick` is True. This executes kick physics immediately with
   no KickOrder.
3. **Tackle**: calls `player.tackle_direct(match, target_id)` if
   `gating.tackle_attempt` is True and preconditions are met.

The decision network heads (`shoot`, `pass_`, `move`, `get_possession`, `mark`,
`hold_position`) are **inputs to the execution network** — they provide
strategic context. They do NOT trigger any Orders.

### Neural players use `NeuralPlayerAI` — `_sample_action` is called inside it
`PPOTrainer.train()` sets `env.sample_action_fn = self._sample_action`.
`ScenarioEnv.reset()` then assigns `NeuralPlayerAI(sample_action_fn, ...)`
to the trainee (and secondary players).  `Match.step()` calls
`player.ai.act()` each physics tick; every 15 ticks `NeuralPlayerAI` calls
`_sample_action`, applies the action via `apply_action_to_player()`, and
stores the result in `player.ai.last_transition`.  `env.step()` reads this
into `env.last_trainee_transition` for the rollout buffer.

`_sample_action(obs_dict)` still returns an 8-tuple internally:
```
(action, log_prob, value, decision_probs, execution_physical,
 decision_physical, target_slots, raw_exec_samples)
```
`raw_exec_samples` must be forwarded to `_action_to_numpy(action, raw_exec_samples)`
so the rollout buffer holds the correct values for the PPO importance ratio.

### Two concerns that must NEVER be conflated

1. **PPO log_prob / training** – computed from raw logits / raw sampled values,
   entirely inside PyTorch autograd.  This is the ONLY place gradients flow.

2. **Action gating** – `select_action()` in `gating.py` is pure Python,
   `@torch.no_grad()`, called AFTER sampling.  It applies the winner-take-all
   rule (> 50% → selected action, all others suppressed) to decide what order
   the engine executes.  It has ZERO effect on the gradient graph.

See design doc section 2.6.  This separation is what makes the mixed
Bernoulli/Categorical/Normal action space work without Gumbel-softmax or
straight-through estimators.

### get-possession >= tackle constraint

`derive_get_possession_prob(tackle_logit, get_possession_raw)` in
`decision_network.py` encodes this as a structural guarantee:
```
gp_prob = tackle_prob + sigmoid(gp_raw) * (1 - tackle_prob)
```
Always in [tackle_prob, 1.0].  PPO log_prob is on the two raw logits
separately as independent Bernoullis, NOT on the derived gp_prob.

### Observation slot shuffling and geometric augmentation

`encode_observation()` randomly shuffles which of the 21 other-player slots
each real player lands in, every call.  This teaches the network permutation
invariance.

**Additional augmentation** (`obs/augment.py`) is applied inside `PPOTrainer._ppo_update()`
**for ALL training phases in this repo**.  Each rollout batch is expanded by
4 × `ppo.augment_n_slot_shuffles` (default 12×):
- 4 geometric flips: identity, flip_x, flip_y, flip_xy (exact pitch symmetries)
- n slot permutations per flip (exact for permutation-invariant attention)
Field indices for each flip are derived from `fields(PlayerFeatures)` /
`fields(BallFeatures)` at import time — see `obs/augment.py` for the full
derivation including pseudovector (spin) transforms.

**IMPORTANT — target slot index remapping**: when a slot permutation is
applied, the stored `pass_target`, `tackle_target`, `mark_target` action
indices must be remapped through the **inverse permutation** so they still
refer to the correct player in the permuted `other_feat`. `augment_batch()`
does this via `inv_perm = argsort(perm)`. Forgetting this causes
`MaskedCategorical.log_prob()` to return `-inf` (target index points to a
now-masked slot), which blows up `approx_kl` to `inf`. This was a bug that
was fixed — do not revert this remapping.

### time_remaining_s is caller-managed

The engine only tracks `match.time_s` (elapsed time).  The env wrapper
(`ScenarioEnv`) subtracts from its `max_episode_s` budget and passes the
remainder to `encode_observation()`.  Time is log1p-normalized so the
"urgent endgame" scenarios (1–20s remaining, 10% of curriculum) are
distinguishable from "2 minutes remaining" after normalization - see design
doc section 7.5 for why plain linear /7200 fails here.

### Restitution coefficient in observations

`match.ball_physics_params.bounce_restitution_vertical` is used as the
ball restitution coefficient in GlobalFeatures (not a field on `Ball` itself).

## ai_config.json structure

Mirrors `physics.json` / `attributes.json`:
- `observation`: MAX_OTHER_PLAYERS=21, decision_interval_s=0.5, normalization constants
- `network`: entity_embed_dim=64, trunk_hidden=256, latent_dim=32, etc.
- `ppo`: gamma=0.99, lam=0.95, clip_range=0.2, learning_rate=3e-4, etc.
- `curriculum`: rng_reduction_start=0.55, rng_reduction_end=0.3, pitch/goal scale limits
- `ema`: attack/defence alpha values and post-goal window duration
- `reward`: per-phase coefficient dicts (phase1, phase2)

## Tests

All AI unit tests live in `tests/ai_unit/`.  They run without a GPU and
without needing a real training loop - just forward passes with random inputs.

```bash
uv run pytest tests/ai_unit -v     # AI unit tests only
uv run pytest tests/ -q            # full suite (includes engine + balance tests)
```

Key test files:
- `test_obs_schema.py` – dimension constants, to_array shapes and dtypes
- `test_obs_encoder.py` – position normalization, flags, slot shuffling,
  padded-slot invariants, no-NaN guarantee
- `test_gae.py` – hand-computed GAE reference cases, episode boundary, bootstrapping
- `test_distributions.py` – masked slots exactly zero, squashed bounds, unit vectors
- `test_gating.py` – winner-take-all selection, threshold edge cases, pass-through
- `test_to_orders.py` – legal/illegal action detection, correct order types
- `test_reward.py` – per-component arithmetic, EMA latency, convergence
- `test_networks.py` – forward pass shapes, no-NaN, get_possession constraint

## Task-id (GlobalFeatures): scaffolded, not yet load-bearing

`GlobalFeatures` has a `MAX_TASK_IDS`-wide (20) one-hot field
(`task_id_0`..`task_id_19`) identifying the active curriculum phase/task.
It is correctly populated by `encode_observation(..., phase=N)` (1-based;
phase 1 → index 0) via `ScenarioEnv._get_obs()`/`_encode_obs_for_player()`,
which pass `self.phase` through automatically. `phase=None` or an
out-of-range value yields an all-zero one-hot (no error).

There is currently **no mixed-multi-phase training loop** that would ever
populate this with more than one non-zero pattern within a single training
run — treat any gradient signal through it as currently uninformative
(constant within a run). Wiring real multi-phase rollout mixing is a
separate, larger workstream, not yet planned in detail.

## Curriculum phases (MVP)

- **Phase 1** (`--phase 1`): 1v1 get-possession/move.  Rewards: closing
  ball distance, gaining possession, progressing toward opponent box.
  Episode ends when trainee reaches opponent box with ball, or 2-minute timeout.
- **Phase 2** (`--phase 2`): Shooting (penalty / keeper / static defender).
  Rewards: time-to-shoot decay, on-target bonus, goal terminal (+10).
  GK remains rules-based throughout.

Both use existing `ui/scenarios.py` scenario builders - no separate
training-only scenario code.  Add new training scenarios directly to
`ui/scenarios.py` so they're also available in the UI for visual inspection.
