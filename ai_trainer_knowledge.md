# AI Trainer Knowledge

This file is the operational guide for an AI agent that needs to run,
modify, or create training experiments for the FootballCoach PPO player AI.
Read this file first and in full before touching any other file.
For deep-dive architecture rationale read `ai_design_doc.md` (the canonical
design document); this file is the actionable distillation of it.

---

## 1. Quick-start

```bash
# One-time: install torch group
uv sync --group ai

# Train phase 1 with default BC settings (pre-train 2000 steps, aux loss annealed)
uv run python -m footballcoach.ai.scripts.train --phase 1 --total-steps 500000

# Train without any BC (pure PPO from a random init)
uv run python -m footballcoach.ai.scripts.train \
    --phase 1 --total-steps 500000 --bc-pretrain-steps 0 --no-bc-aux

# More BC pre-training, then PPO with aux loss
uv run python -m footballcoach.ai.scripts.train \
    --phase 1 --total-steps 500000 --bc-pretrain-steps 5000

# Resume from a checkpoint (BC pre-training is skipped on resume)
uv run python -m footballcoach.ai.scripts.train \
    --phase 1 --total-steps 500000 \
    --checkpoint checkpoints/checkpoint_00002048.pt

# Evaluate a saved checkpoint (100 trials, writes JSON)
uv run python -m footballcoach.ai.scripts.evaluate \
    --checkpoint checkpoints/checkpoint_00500000.pt \
    --phase 1 --n-trials 100 --output results/eval_phase1.json

# Run all tests (engine + AI unit + smoke)
uv run pytest tests/ -q

# Run AI-only tests (faster)
uv run pytest tests/ai_unit tests/ai_scenario -v
```

**Checkpoint saving:** checkpoints are saved automatically after every PPO
rollout update (default every 2048 steps) AND at the very end of training,
so the final model is never lost.  Format: `checkpoints/checkpoint_NNNNNNNN.pt`.
Each `.pt` file contains `{step, decision_net, execution_net, optimizer}`.
Override the save directory with `--checkpoint-dir path/`.

---

## 2. File importance ranking — what to read and why

### Tier 1: MUST read before changing anything

| File | Lines | Why it matters |
|------|-------|----------------|
| `ai_trainer_knowledge.md` (this file) | ~350 | Operational guide |
| `src/footballcoach/ai/config/ai_config.json` | ~80 | All tunable hyperparameters incl. BC |
| `src/footballcoach/ai/scripts/train.py` | ~160 | CLI entry point; how envs and BC are built |
| `src/footballcoach/ai/curriculum/phases.py` | ~100 | Which scenario each phase uses |
| `src/footballcoach/ui/scenarios.py` | ~1100 | All scenario builders incl. `build_1v1_scenario`; **add new training scenarios here** |
| `src/footballcoach/ai/ppo/bc.py` | ~310 | BC label generation, pre-trainer, and aux loss — read when tuning BC |

### Tier 2: Read when modifying the training loop or reward

| File | Why it matters |
|------|----------------|
| `src/footballcoach/ai/ppo/ppo_trainer.py` | Full PPO loop: `train()`, `_sample_action()`, `_ppo_update()`, checkpointing |
| `src/footballcoach/ai/env/scenario_env.py` | Gym-like env wrapper: `reset()`, `step()`, reward integration |
| `src/footballcoach/ai/env/reward.py` | `phase1_reward()`, `phase2_reward()` — all coefficients from `ai_config.json` |
| `src/footballcoach/ai/ppo/rollout_buffer.py` | Buffer storage + `compute_gae()` |

### Tier 3: Read when debugging observations, network architecture, or actions

| File | Why it matters |
|------|----------------|
| `src/footballcoach/ai/obs/schema.py` | Feature vector dataclasses; defines `PLAYER_FEATURE_DIM=26`, `BALL_FEATURE_DIM=12`, `GLOBAL_FEATURE_DIM=12` |
| `src/footballcoach/ai/obs/encoder.py` | `encode_observation(match, player_id, time_remaining_s, ...)` → `ObservationBatch` |
| `src/footballcoach/ai/models/decision_network.py` | `DecisionNetwork.from_config()` |
| `src/footballcoach/ai/models/execution_network.py` | `ExecutionNetwork.from_config()` |
| `src/footballcoach/ai/action/gating.py` | `select_action()` — winner-take-all, NO gradients |
| `src/footballcoach/ai/action/to_orders.py` | Maps `GatingResult` → engine `Order` objects + illegal-action detection |
| `src/footballcoach/ai/action/distributions.py` | `IndependentBernoulli`, `MaskedCategorical`, `SquashedNormalHead`, `DirectionHead` |

### Tier 4: Read only if you need to understand the underlying engine

| File | Why it matters |
|------|----------------|
| `src/footballcoach/engine/match.py` | `Match.step()` — the sim tick; AI is a pure consumer, not a modifier |
| `src/footballcoach/orders.py` | `MoveOrder`, `ShootOrder`, `GetPossessionOrder`, etc. — what `to_orders.py` produces |
| `src/footballcoach/entities/player.py` | `Player`, `PlayerState`, `Team` |
| `src/footballcoach/entities/ball.py` | `Ball`, `Ball.at_rest()` |
| `src/footballcoach/entities/pitch.py` | `Pitch.standard()`, `pitch.half_length`, `pitch.is_in_box()` |
| `src/footballcoach/ai/knowledge.md` | Operational notes for the full `ai/` package |
| `ai_design_doc.md` | Full architecture spec and rationale (1500+ lines); reference, not required reading |

### Do NOT read / modify

| File | Why |
|------|-----|
| `Idea2.md` | Personal notes, not for AI agents |
| `src/footballcoach/ui/app.py`, `renderer.py`, `input.py` | Pygame UI, irrelevant to training |
| `tests/balance/` | Statistical tests of the engine physics, not AI |
| `tests/unit/`, `tests/scenario/` | Engine tests, not AI |

---

## 3. Architecture summary (enough to reason about training)

### 3.1 Two networks per player, shared weights

All players share **one** `DecisionNetwork` and **one** `ExecutionNetwork`
(not one per player — weight sharing is intentional).

**Decision network** runs first each decision tick:
- **Inputs**: self features, up to 21 other-player features (entity encoder +
  attention), ball features, global/match context.
- **Outputs** (all heads present from day one, even in early curriculum):
  - Independent sigmoid: `shoot`, `pass_`, `move`, `tackle`, `mark`, `hold_position`
  - Derived: `get_possession` ≥ `tackle` (structural guarantee)
  - Masked categorical: `pass_target`, `tackle_target`, `mark_target` (slots 0–20)
  - Continuous: `move_region_center`, `move_region_size`, `move_arrival_speed`
  - Scalar: `attack_defence_raw` (EMA-smoothed outside the network)
  - Vector: `latent_vector` (32-d, passed to execution network)
  - Scalar: `value` (shared-trunk critic)

**Execution network** runs second:
- **Inputs**: same observation + full decision-network output (all heads)
- **Outputs**: `move_direction` (unit vec), `sprint` (Bernoulli),
  `kick_this_tick` (Bernoulli), `kick_direction`, `kick_power`, `kick_spin`,
  `tackle_attempt` (Bernoulli), `value`

**Decision interval**: networks run every **0.5s** of sim time (~15 engine
ticks at 1/30s).  Between decisions, the last assigned `Order` persists.

### 3.2 Action gating (CRITICAL — do not confuse with training)

Two completely separate concerns:

1. **In-game gating** (`gating.py::select_action()`): pure Python,
   `@no_grad`, post-sampling.  If any sigmoid head > 0.5 → the highest
   one "wins" (all others treated as 0) → drives the engine via `to_orders.py`.
   This is *not* differentiable.

2. **PPO log_prob**: computed on raw logits/samples *before* gating, inside
   PyTorch autograd.  Uses `IndependentBernoulli.log_prob()` per head,
   summed.  The argmax in gating is never in the gradient graph.

**Never apply the gating rule inside a loss computation.**

### 3.3 PPO loop flow

```
env.reset() → ObservationBatch
loop:
    _sample_action(obs) → (action, log_prob, value, ...)
    env.step(action) → (next_obs, reward, done, info)
    buffer.add(obs, action, log_prob, value, reward, done)
    if buffer full (2048 steps):
        compute_gae(gamma=0.99, lam=0.95, last_value)
        _ppo_update(batch) → 4 epochs, minibatches of 64
        _save_checkpoint(step)
        buffer.clear()
    obs = next_obs if not done else env.reset()
```

One rollout = 2048 decision steps = ~1024 sim-seconds = ~8.5 min of sim time.

### 3.4 Observation encoding

`encode_observation(match, player_id, time_remaining_s, ...)`:
- Builds `PlayerFeatures` for self (rel_dx=0, is_self=1) and up to 21 others.
- **Randomly shuffles** real players into random slots each call (permutation
  invariance — do not "fix" this).
- Pads unused slots to zero; `exists=0.0` distinguishes empty from real.
- Normalises positions by pitch half-dims, velocities by per-player top speed,
  time by `log1p(t) / log1p(7200)`.
- Returns `ObservationBatch` with `.to_torch_dict()` → `{self_feat, other_feat,
  ball_feat, global_feat, exists_mask}`.

### 3.5 Model saving

Checkpoints save `decision_net`, `execution_net`, and `optimizer` state.
Load with:
```python
trainer = PPOTrainer.from_config()
trainer.load_checkpoint(Path("checkpoints/checkpoint_00500000.pt"))
```
To export just the network weights (e.g. for inference without the optimizer):
```python
import torch
ckpt = torch.load("checkpoints/checkpoint_00500000.pt", map_location="cpu")
torch.save(ckpt["decision_net"], "models/decision_net_500k.pt")
torch.save(ckpt["execution_net"], "models/execution_net_500k.pt")
```

---

## 4. Curriculum phases

### Phase 1: Get Possession / Move (current focus)

```
--phase 1  |  scenario: build_1v1_scenario  |  trainee: "trainee"
```

- Both players random placement, random attributes (`generic` tier), random
  stamina (0.3–1.0), random headings.
- Ball: random placement, random velocity (≤8 m/s, resampled to stay in
  bounds 3s), random spin, random restitution (Gaussian σ=0.08 around default).
- Opponent: **immobile** (no order) — the first sub-phase.
- Episode ends: trainee reaches opponent box with ball (+5.0 terminal), or
  120s timeout.
- Reward:
  ```
  +0.05 × (prev_ball_dist - curr_ball_dist)   # closing distance
  +1.0 if gained possession this step
  +0.1 × ball_progress_m (toward opponent goal, only when in possession)
  -1.0 if ball went out after trainee touched it
  -0.2 if illegal action attempted
  +5.0 if reached opponent box with possession (terminal)
  ```
- Progression (not yet automated): immobile opponent → rules-based opponent
  (`GetPossessionOrder`/`MoveOrder`) → frozen older-generation checkpoint.

**BC bootstrapping for Phase 1** (`bc.py::phase1_labels`):
- Ball loose or opponent has it → `get_possession_extra=1`, `move_dir` = toward ball, `sprint=1`
- Trainee has ball → `move=1`, `move_dir` = toward opponent goal (+x), `sprint=1`
- Trains both decision network Bernoulli heads **and** execution network `move_direction` + `sprint`

### Phase 2: Shooting

```
--phase 2  |  scenario: build_penalty_scenario  |  trainee: "kicker"
```

- Single penalty kicker vs. (optionally) rules-based GK.
- Reward: time-decay bonus for speed to shoot, +2 on target, +10 goal.
- GK remains rules-based (`SaveOrder`) throughout.

### Phases 3, 4 (stubs only — not yet implemented)

Phase 3 (`--phase 3`): passing. Phase 4 (`--phase 4`): tackling.
Both defined in `curriculum/phases.py` with `scenario_key` pointing at
existing UI scenarios, but `_build_env()` in `train.py` raises
`NotImplementedError` for phases > 2.  Implement by adding cases to
`train.py::_build_env()` and `train.py::_build_phase3_env()` etc.,
following the same pattern as `_build_phase1_env()`.

---

## 5. Hyperparameters — where and how to change them

All tunable constants live in **`src/footballcoach/ai/config/ai_config.json`**.
Never hardcode them.  Key sections:

```json
{
  "ppo": {
    "gamma": 0.99,          // discount factor
    "lam": 0.95,            // GAE lambda
    "clip_range": 0.2,      // PPO clipping epsilon
    "learning_rate": 3e-4,
    "n_epochs": 4,          // PPO epochs per rollout
    "minibatch_size": 64,
    "rollout_steps": 2048,  // steps before each update
    "target_kl": 0.02       // early-stop threshold per epoch
  },
  "bc": {
    "pretrain_steps": 2000,     // supervised BC steps before PPO (0 = skip)
    "pretrain_lr": 1e-3,        // learning rate for the BC pre-training phase
    "aux_coeff_start": 0.2,     // BC aux loss weight at step 0 of PPO
    "aux_coeff_end": 0.0        // BC aux loss weight at final step (linear anneal)
  },
  "reward": {
    "phase1": {
      "ball_distance_shaping": 0.05,
      "gain_possession_bonus": 1.0,
      "ball_progress_scale": 0.1,
      "ball_out_penalty": -1.0,
      "illegal_action_penalty": -0.2,
      "box_possession_terminal": 5.0
    }
  },
  "curriculum": {
    "rng_reduction_start": 0.55,  // physics randomness at step 0
    "rng_reduction_end": 0.3      // physics randomness at end
  }
}
```

Network architecture changes (layer sizes, heads) require modifying both
`ai_config.json` and potentially the network module files. Reward coefficient
changes only require editing the JSON.

---

## 6. Behavioural cloning (BC) from rules-based AI

BC lets the rules-based `orders` AI act as a teacher for the neural networks.
**It does not and cannot use rules-based actions as PPO rollout data** — PPO
requires `log π_old(a|s)` from the policy that took each action, which the
rules-based AI does not have.  Instead, BC is a *separate, additive* loss.

### Two modes (both configurable, composable)

**Mode 1 — Pre-training (before PPO):**
`BCPretrainer.pretrain(env, n_steps, label_fn)` in `bc.py`.
Rolls the *network* in the env but supervises it with BC labels from the
rules-based logic each step.  Network weights update via pure BCE/cosine
loss.  This gives the network a warm-start before PPO exploration begins,
dramatically reducing the number of PPO steps to reach competent behaviour.

**Mode 2 — Auxiliary loss during PPO:**
At each PPO rollout step, `label_fn(env)` is called and a `BCLabel` is
stored in the rollout buffer.  During `_ppo_update()`, a BC loss term is
added: `total_loss += bc_coeff * bc_loss`.  `bc_coeff` anneals linearly
from `aux_coeff_start` to `aux_coeff_end` (default 0.2 → 0.0) so BC
influence fades as the RL signal takes over.

### What is supervised

`bc_loss_from_tensor()` in `bc.py` supervises:
- **Decision network**: all 7 Bernoulli heads (shoot, pass, move, tackle,
  get_possession_extra, mark, hold_position) via `binary_cross_entropy_with_logits`.
- **Execution network**: `move_direction` (cosine loss on raw pre-normalised
  2D vector output), `sprint` (BCE from logit).
- Kick, tackle_attempt, and kick_direction are not supervised
  (rules-based AI doesn't kick in Phase 1; extend `BCLabel` if needed later).

### Adding BC for a new phase

1. Write `phase_N_labels(env) -> BCLabel` in `bc.py` following `phase1_labels`.
2. Add it to `_bc_label_fn_for_phase(phase_id)` in `train.py`.
3. Optionally tune `bc.pretrain_steps` / `bc.aux_coeff_start` in
   `ai_config.json` (or override per run with `--bc-pretrain-steps`).

### CLI flags

| Flag | Effect |
|------|--------|
| `--bc-pretrain-steps N` | Override `bc.pretrain_steps` from config |
| `--bc-pretrain-steps 0` | Skip pre-training entirely |
| `--no-bc-aux` | Disable BC auxiliary loss during PPO |

---

## 7. Adding a new training scenario

All scenarios live in **`src/footballcoach/ui/scenarios.py`** — one source of
truth for both the training loop and the UI scenario picker.

**Pattern** (copy from `build_1v1_scenario` as the template):

1. Write a `build_my_scenario(rng_reduction=0.3, *, param1=...) -> Match` function.
   - Use `random.Random()` internally (not global `random`) so it's re-seedable.
   - Use `generate_attributes(tier=..., rng=rng)` for player attributes.
   - Set `player.stamina`, `player.heading_rad` manually as needed.
   - Return a `Match` with orders pre-assigned to all non-trainee players.
   - Trainee player id should be a stable string like `"trainee"`.

2. Optionally write an `_my_scenario_on_tick(match, trial_tick)` hook for
   per-tick AI logic (only needed if rules-based players need updating each tick).

3. Add a `ScenarioDefinition(key="my_key", ..., build=build_my_scenario,
   on_tick=..., params=[...])` entry to the `SCENARIOS` list.

4. Add a `CurriculumPhase(phase_id=N, scenario_key="my_key", ...)` to
   `curriculum/phases.py` and `ALL_PHASES` / `PHASES_BY_ID`.

5. Add a `_build_phaseN_env(phase)` function in `train.py` that:
   - Creates a `ScenarioDefinition` wrapping the new builder.
   - Creates `ScenarioEnv(definition=defn, trainee_player_id="trainee", phase=N, **phase.env_kwargs)`.

6. Add the new phase to `_build_env(phase)` in `train.py`.

---

## 8. Environment wrapper — ScenarioEnv

```
src/footballcoach/ai/env/scenario_env.py
```

Key behaviour:
- `reset()` calls `ScenarioLoop._start_trial()` which calls `definition.build(rng_reduction)`.
- `step(action)` applies gating + `apply_action_to_player()`, then runs
  `ScenarioLoop.step()` in a sub-loop for `DECISION_INTERVAL_S / dt_s = 15` ticks.
- Returns `(obs, reward, done, StepInfo)`.
- `done` is `True` when `ScenarioLoop.step()` returns `True` (trial ended)
  OR the episode tick limit is reached.
- The `timeout_ticks` passed to `ScenarioLoop` is `max_episode_s / dt_s`
  (not the ScenarioLoop default of 500); this ensures 120s episodes work.

**Trainee player id**: must match the `player_id` string used in the
scenario's `build_*` function (e.g. `"trainee"` for phase 1,
`"kicker"` for phase 2).

---

## 8. Tests

### AI unit tests (no training, fast)
```bash
uv run pytest tests/ai_unit/ -v
```
Files: `test_obs_schema`, `test_obs_encoder`, `test_gae`, `test_distributions`,
`test_gating`, `test_to_orders`, `test_reward`, `test_networks`.

### Smoke tests (actual training loop, ~10s)
```bash
uv run pytest tests/ai_scenario/ -v
```
Five tests covering: obs finite, `_sample_action` finite log_prob, `step()`
finite reward, one PPO update finite losses, two updates change policy.

Run the smoke tests after any change to the env wrapper, reward function,
or training loop before starting a real training run.

### Full suite
```bash
uv run pytest tests/ -q   # all engine + AI + balance tests
```

---

## 9. Common gotchas

### `trainee_player_id` must match the scenario
If `ScenarioEnv` cannot find the player id in the `Match`, every call to
`_find_trainee()` raises `KeyError`.  Phase 1 uses `"trainee"`, phase 2 uses
`"kicker"`.  Check the `build_*` function's `Player.create(...)` call.

### `ScenarioLoop.timeout_ticks` must be set explicitly
`ScenarioLoop`'s default `timeout_ticks=500` (~16.7s).  `ScenarioEnv.reset()`
passes `timeout_ticks=int(max_episode_s / dt_s)` to override this.  If you
create `ScenarioLoop` directly, always pass this.

### Two-update test catches broken `loss.backward()`
`tests/ai_scenario/test_smoke.py::test_two_ppo_updates_change_policy` checks
that two sequential PPO updates produce *different* losses.  If they're
identical, `optimizer.step()` is not modifying weights (e.g. zero gradients).

### get_possession_prob is derived, not direct
`gp_prob = tackle_prob + sigmoid(gp_raw) * (1 - tackle_prob)`.
The PPO log_prob for get_possession is on `gp_raw` (raw Bernoulli), not on
`gp_prob`.  Do not try to add `gp_prob` as an independent Bernoulli.

### Observation slot order is random — this is intentional
`encode_observation()` shuffles real players into random slots each call.
Do not cache slot-to-player mappings between steps.  Use `slot_player_ids`
(returned by `_sample_action`) to convert a target slot index to a player id
when building engine orders.

### Categorical log_prob is gated by intent
`pass_target` log_prob is only added when `pass_` (Bernoulli) fired (=1).
Same for `tackle_target` and `mark_target`.  See `_compute_log_prob()` and
`_recompute_log_prob()` in `ppo_trainer.py`.  Violating this adds spurious
gradient noise.

### Checkpoint saves optimizer state — resume is exact
Resuming with `--checkpoint` restores the optimizer state (not just weights)
so the learning-rate schedule, momentum, etc. continue exactly from where
training stopped.  This is intentional.

---

## 10. Current state and next steps

### What is done
- Full architecture implemented and tested: entity encoder + attention,
  decision + execution networks, custom PPO loop with GAE, rollout buffer,
  masking, all action distributions (Bernoulli/Categorical/Normal/Direction),
  phase1 and phase2 reward functions, curriculum phase definitions, EMA
  smoothing for attack/defence weighting.
- Phase 1 scenario (`build_1v1_scenario`): random placement, attributes,
  stamina, heading; random ball velocity/spin/restitution.
- End-to-end smoke test suite in `tests/ai_scenario/`.
- Training loop confirmed to run: first PPO update at step=2048 produces
  finite losses and a positive mean episode reward from ball-chasing behaviour.
- Checkpoint saving after every rollout and at end of run.
- **Behavioural cloning (BC) bootstrapping** (`src/footballcoach/ai/ppo/bc.py`):
  - Pre-training phase: pure supervised BCE on rules-based labels for N steps before PPO.
  - Auxiliary loss during PPO: BC cross-entropy added to PPO loss with linearly annealed coefficient.
  - Both modes train **decision network** (all 7 Bernoulli heads) AND **execution network** (`move_direction`, `sprint`).
  - Phase 1 label fn: `bc.phase1_labels(env)` — toward ball when loose, toward goal when in possession.
  - Configurable via `ai_config.json["bc"]` and CLI flags `--bc-pretrain-steps` / `--no-bc-aux`.

### Immediate next step: first real Phase 1 experiment
```bash
uv run python -m footballcoach.ai.scripts.train \
    --phase 1 --total-steps 100000 --seed 42 \
    --checkpoint-dir checkpoints/phase1_run1/
```
Watch `ep_reward` trend in logs.  If positive trend visible after 50k steps,
the reward signal and observation encoding are connected end-to-end.  If
flatlines, check reward shaping coefficients in `ai_config.json`.

After that, evaluate:
```bash
uv run python -m footballcoach.ai.scripts.evaluate \
    --checkpoint checkpoints/phase1_run1/checkpoint_00100000.pt \
    --phase 1 --n-trials 200 --output results/phase1_100k.json
```

### Phase 1 opponent progression (not yet automated)
The design calls for: immobile → sometimes rules-based → sometimes older
frozen AI.  Currently the opponent is always immobile.  To add rules-based
opponent: in `build_1v1_scenario`, assign `opponent.current_order =
GetPossessionOrder()` when the opponent doesn't have the ball, and a `MoveOrder`
toward the trainee's goal when it does.  This can be done via an `on_tick` hook
on the `ScenarioDefinition` (see `_1v1_on_tick` as a template).

### Phase 2 (shooting) — not yet run
Ready to train.  Scenario is `build_penalty_scenario`.  Run with `--phase 2`.

### Phases 3 & 4 (passing, tackling) — stubs only
`NotImplementedError` in `train.py::_build_env()`.  Scenarios exist in
`ui/scenarios.py` (`build_pass_scenario`, `build_tackle_scenario`).
Needs: curriculum phase reward functions, `_build_phase3_env()` etc. in
`train.py`, and potentially new `phase3_reward()` / `phase4_reward()` in
`env/reward.py`.
