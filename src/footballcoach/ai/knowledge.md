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

## BC label vector (BC_LABEL_DIM = 15)

`bc.py` stores 15 floats per step:

| idx | field | source |
|-----|-------|--------|
| 0–6 | decision Bernoullis (shoot, pass, move, tackle, gp_extra, mark, hold) | rules AI decision |
| 7–8 | move_dir_x/y | direction toward target |
| 9 | sprint | rules AI order |
| 10–11 | move_region_x/y (metres) | MoveOrder target position |
| 12 | **kick_this_tick** | `player.on_kick` callback (fired by engine) |
| 13 | **tackle_attempt** | `player.on_tackle` callback (fired by engine) |
| 14 | valid | 1.0 = use this label |

**Critical:** indices 12–13 come from **engine callbacks** (`Player.on_kick`,
`Player.on_kick`), not from inspecting the current order type.  They reflect
what the engine physically executed this tick.  Indices 0–11 come from asking
the rules AI what it *decides* next — these are input to the decision network
and movement-related execution heads.

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
