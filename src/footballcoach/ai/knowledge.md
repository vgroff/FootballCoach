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
    to_orders.py      # GatingResult -> engine orders + illegal-action detection
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
  curriculum/
    phases.py          # CurriculumPhase dataclasses + PHASES_BY_ID dict
    opponent_pool.py   # OpponentPool + apply_rules_based_opponent()
  scripts/
    train.py           # CLI: uv run python -m footballcoach.ai.scripts.train --phase 1
    evaluate.py        # CLI: ... evaluate --checkpoint path.pt --n-trials 100
```

## Critical design rules

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

### Observation slot shuffling

`encode_observation()` randomly shuffles which of the 21 other-player slots
each real player lands in, every call.  This teaches the network permutation
invariance.  Tests in `tests/ai_unit/test_obs_encoder.py` verify that (a)
features are identical across different shuffle seeds and (b) different seeds
actually produce different slot assignments.

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
