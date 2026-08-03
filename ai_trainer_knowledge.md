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

# --- Demonstrations (do this first; used for offline BC pre-training) ---

# Record 200 phase-1 episodes of rules-based AI play (~7k steps, ~7s)
# Sampling: env.step() every 0.5s + on_kick/on_tackle callbacks fire extra
# samples at the exact tick kicks/tackles execute. Demos already in demonstrations/phase1/.
uv run python -m footballcoach.ai.scripts.record_demonstrations \
    --phase 1 --n-episodes 200 --episodes-per-file 8 \
    --output demonstrations/phase1/

# Inspect what's already recorded
uv run python -m footballcoach.ai.scripts.record_demonstrations \
    --phase 1 --n-episodes 0 --output demonstrations/phase1/ --info

# --- Training ---

# Train phase 1 using offline BC dataset (recommended — stable pre-training)
uv run python -m footballcoach.ai.scripts.train \
    --phase 1 --total-steps 200000 \
    --bc-dataset demonstrations/phase1/ \
    --bc-pretrain-epochs 3 --bc-pretrain-batch-size 256 \
    --checkpoint-dir checkpoints/phase1_run4/

# Train without any BC (pure PPO from random init)
uv run python -m footballcoach.ai.scripts.train \
    --phase 1 --total-steps 200000 --bc-pretrain-steps 0 --no-bc-aux

# BC aux loss anneals to zero by 30% of training (aux_coeff_anneal_fraction=0.3 in config)
# — after that the network is fully free to explore via PPO rewards.

# Skip pre-training by reusing an existing pretrained checkpoint — fastest iteration.
# --from-pretrained accepts a directory (auto-finds checkpoint_pretrained.pt) or a file path.
# BC pre-training is saved as checkpoint_pretrained.pt in the checkpoint-dir automatically.
uv run python -m footballcoach.ai.scripts.train \
    --phase 1 --total-steps 50000 --seed 42 \
    --checkpoint-dir checkpoints/phase1_run24/ \
    --from-pretrained checkpoints/phase1_run23/

# Resume from a checkpoint (BC pre-training is skipped on resume)
uv run python -m footballcoach.ai.scripts.train \
    --phase 1 --total-steps 200000 \
    --checkpoint checkpoints/phase1_run4/checkpoint_00100000.pt

# Evaluate a saved checkpoint (100 trials, writes JSON)
uv run python -m footballcoach.ai.scripts.evaluate \
    --checkpoint checkpoints/phase1_run4/checkpoint_00200000.pt \
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
| `ai_trainer_knowledge.md` (this file) | ~400 | Operational guide |
| `src/footballcoach/ai/config/ai_config.json` | ~90 | All tunable hyperparameters incl. BC |
| `src/footballcoach/rules_ai.py` | ~150 | **Source of truth for BC labels** — `Phase1RulesAI.act()` is what `phase1_labels()` calls; change behaviour here, not in `bc.py` |
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
| `src/footballcoach/ai/obs/schema.py` | Feature vector dataclasses; defines `PLAYER_FEATURE_DIM=27`, `BALL_FEATURE_DIM=12`, `GLOBAL_FEATURE_DIM=31` (11 match-context fields + 20 `task_id_N` one-hot fields, see "Task-id" note in `ai/knowledge.md`) |
| `src/footballcoach/ai/obs/encoder.py` | `encode_observation(match, player_id, time_remaining_s, ...)` → `ObservationBatch` |
| `src/footballcoach/ai/models/decision_network.py` | `DecisionNetwork.from_config()` |
| `src/footballcoach/ai/models/execution_network.py` | `ExecutionNetwork.from_config()` |
| `src/footballcoach/ai/action/gating.py` | `select_action()` — winner-take-all, NO gradients |
| `src/footballcoach/ai/action/to_orders.py` | Applies execution outputs DIRECTLY to player — **no Orders**. Sets `desired_direction`, `desired_speed_mode`, calls `kick_direct()`, `tackle_direct()` |
| `src/footballcoach/ai/action/distributions.py` | `IndependentBernoulli`, `MaskedCategorical`, `SquashedNormalHead`, `DirectionHead` |
| `src/footballcoach/ai/obs/augment.py` | Geometric + slot-permutation augmentation. **CRITICAL**: target slot indices (pass/tackle/mark) are remapped through the inverse permutation — do not remove this |

### !!!! CRITICAL ARCHITECTURE RULE — THE NETWORK NEVER ISSUES ORDERS !!!!

The neural network NEVER sets `player.current_order`. It ONLY:
1. Sets `player.desired_direction` (Vector3) + `player.desired_speed_mode` (SpeedMode) directly
2. Calls `player.kick_direct(match, ...)` when `kick_this_tick` is True
3. Calls `player.tackle_direct(match, ...)` when `tackle_attempt` is True

The decision heads (`shoot`, `pass_`, `move`, `get_possession`, etc.) are
**inputs to the execution network** providing strategic context. They do NOT
trigger any Orders at inference time.

Orders are for the rules-based AI and human input only.

### Tier 4: Read only if you need to understand the underlying engine

| File | Why it matters |
|------|----------------|
| `src/footballcoach/engine/match.py` | `Match.step()` — the sim tick; AI is a pure consumer, not a modifier |
| `src/footballcoach/orders.py` | `MoveOrder`, `GetPossessionOrder`, etc. — used by rules-based AI only, NOT by neural network |
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
- **Outputs**: `exec_move` (Bernoulli: move vs standstill → sets `desired_speed_mode`),
  `move_direction` (unit vec → sets `desired_direction`), `sprint` (Bernoulli: SPRINT vs JOG),
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
env.sample_action_fn = trainer._sample_action  # wires NeuralPlayerAI
env.reset()  # assigns NeuralPlayerAI to trainee (+ secondary players)
loop:
    env.step() → (next_obs, reward, done, info)
    # NeuralPlayerAI sampled inside Match.step(); transition in env.last_trainee_transition
    tr = env.last_trainee_transition
    buffer.add(obs=tr["obs"], action=..., log_prob=tr["log_prob"],
               head_log_probs=tr["head_log_probs"], ...)  # per-head lps for DEBUG KL
    if buffer full (4096 steps):
        augment_batch (12× with flips + slot permutations)
        compute_gae(gamma=0.99, lam=0.95, last_value)
        _ppo_update(batch) → up to 4 epochs × minibatches of 64
            # early stop fires per-MINIBATCH when KL > target_kl (0.05)
            # typically stops after 1-3 minibatch steps
        _save_checkpoint(step)
        buffer.clear()
    if done: env.reset()
```

One rollout = 4096 decision steps (12× augmented = 49,152 effective samples).
With target_kl=0.05 and per-minibatch early stop, each rollout typically
generates 1–10 gradient steps rather than 768+ (which caused policy collapse).

**Per-head log_probs in buffer**: `tr["head_log_probs"]` stores the 13 individual
head log_probs (shoot, pass, move, tackle, gp_extra, mark, hold, exec_move, sprint,
kick, tackle_attempt, move_dir, kick_dir) computed at sample time.  These are
packed into `batch["head_log_probs"]` and used in the `[KL > threshold]` diagnostic
block to show which heads changed most since the rollout was collected.

### 3.4 Observation encoding

`encode_observation(match, player_id, time_remaining_s, ...)`:
- Builds `PlayerFeatures` (28 floats) for self (rel_dx=0, is_self=1) and up to 21 others.
- **Randomly shuffles** real players into random slots each call (permutation
  invariance — do not "fix" this).
- Pads unused slots to zero; `exists=0.0` distinguishes empty from real.
- Normalises positions by pitch half-dims, velocities by per-player top speed,
  time by `log1p(t) / log1p(7200)`.
- Returns `ObservationBatch` with `.to_torch_dict()` → `{self_feat, other_feat,
  ball_feat, global_feat, exists_mask}`.

**`PlayerFeatures` layout (27 floats)** — last two fields are new absolute position:
- `pos_x = player.position.x / 52.5` — world-frame x, ≈[-1,1] on standard pitch
- `pos_y = player.position.y / 34.0` — world-frame y, ≈[-1,1] on standard pitch
- Both are negated by the augmenter under `flip_x` / `flip_y` respectively.
- Self slot has non-zero `pos_x`/`pos_y` (unlike `rel_dx`/`rel_dy` which are always 0 for self).

**`GlobalFeatures` pitch/goal/box dims** are now normalised by standard values
(105m, 68m, 7.32m, 2.44m, 16.5m, 40.32m) so the network sees ≈1.0 on a standard
pitch and a fraction on smaller training pitches. Fields renamed from `*_m` to `*_norm`.

**⚠ Schema break**: `PLAYER_FEATURE_DIM` changed from 26 → 28. All existing
`.pt` checkpoints and `.npz` demonstration files are **incompatible** and must
be regenerated before training.

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
- **Trainee team is randomised each episode** (50% Team.LEFT attacks +x, 50%
  Team.RIGHT attacks -x).  Pass `trainee_team=Team.LEFT/RIGHT` to pin it
  (used in tests for direction-cosine assertions).
- Ball: random placement, random velocity (≤**10 m/s**, resampled to stay in
  bounds 3s), random spin, random restitution (Gaussian σ=0.08 around default).
- **Opponent: 50/50 per episode** — either `opponent.ai = Phase1RulesAI()`
  (chases ball / sprints to box; driven automatically by `Match.step()`) or
  `opponent.ai = None` (standing still while the neural network controls it).
  Flag `match._opponent_use_rules_ai` is set randomly in `build_1v1_scenario`.
- **Episode ends** when **either player** reaches the other's box with
  possession, **or** 120s timeout.
  - Trainee reaches opponent box: +5.0 terminal reward (trainee wins)
  - Opponent reaches trainee's box: episode ends, no terminal bonus (trainee
    just loses; the standard step rewards already penalise this path)
- Reward (same coefficients for both trainee and opponent when both are neural):
  ```
  +0.05 × (prev_ball_dist - curr_ball_dist)   # closing distance
  +1.0 if gained possession this step
  +0.1 × ball_progress_m (toward OWN attacking goal, only when in possession)
  -1.0 if ball went out after this player touched it
  -0.2 if illegal action attempted
  +5.0 if reached opponent box with possession (terminal)
  ```

**Dual-player training (secondary_player_ids)**:
- In non-rules-based episodes (50%), the opponent is also driven by the
  **shared neural network** and its transitions are added to the same rollout
  buffer as the trainee's.
- `ScenarioEnv` accepts `secondary_player_ids=["opponent"]`.  The trainer
  injects `env.sample_action_fn = self._sample_action` at the start of
  `train()`.  After each `env.step()` the trainer drains
  `env.last_secondary_results` and calls `buffer.add()` for each.
- In rules-based episodes (50%), the opponent is driven by `player.ai`
  (`Phase1RulesAI`) — no neural sampling is done for it (`ScenarioEnv` skips
  secondary players when `match._opponent_use_rules_ai` is True).
- Net effect: ~1.5× rollout fill rate vs. trainee-only; shared weights see
  both the attacker and defender perspectives within the same training run.
- **Episode type is tracked separately in the training log.**  `StepInfo.is_rules_episode`
  is set from `match._opponent_use_rules_ai` and propagated to the trainer so
  outcomes are split into `vs_rules(N): win%/opp%` and `vs_neural(N): win%/opp%`
  in the rollout log line.  This lets you distinguish whether the trainee is
  improving against the rules AI specifically or just beating the neural copy of
  itself.

**BC bootstrapping for Phase 1** (`bc.py::phase1_labels(env, player_id=None)`):
- **Does NOT duplicate logic** — calls `Phase1RulesAI().act()` on a temporary
  player state, reads back the `MoveOrder` or `GetPossessionOrder` it sets,
  and translates that into a `BCLabel`.  If the rules AI behaviour changes,
  the BC labels automatically follow.
- `rules_ai.py::Phase1RulesAI` is the single source of truth for what the
  rules-based player does.  Edit that, not `bc.py`.

**Offline BC dataset** (`ai/bc/dataset.py`):
- Pre-recorded .npz files of (obs, bc_label) pairs from rules-based play.
- Recorded with: `uv run python -m footballcoach.ai.scripts.record_demonstrations`
- Loaded with: `DemonstrationDataset.from_directory("demonstrations/phase1/")`
- `BCPretrainer.pretrain(..., dataset=ds, n_epochs=3, batch_size=256)` uses
  stable minibatch gradient descent instead of noisy single-sample online updates.
- **Always use the offline dataset when it exists.** Online pre-training
  (dataset=None) is noisy and oscillates on episode resets.

**BC aux loss annealing** (`ai_config.json::bc.aux_coeff_anneal_fraction`):
- `aux_coeff_start=0.95` → `aux_coeff_end=0.0`, reaching zero by
  `aux_coeff_anneal_fraction × total_steps` (default 0.35 = 35% of training).
- After that the network is purely RL-driven.  This is intentional — early
  guidance, then full freedom to explore.

**Per-epoch BC breakdown logged**: `pretrain_combined()` logs per-epoch:
`bc_loss`, `dir_cos`, `mv_p` (exec_move prob), `spr_p` (sprint prob), plus a
`[decision=X  exec_bce=X  sprint=X  move=X  direction=X  region=X]` breakdown
showing where the loss is coming from.  Sprint and region are the common floors.

**BC loss weights** — two separate config keys, both in `bc` section:
- `direction_loss_weight` (default 3.0): multiplier on move_direction cosine loss.
  Needed because one continuous head competes with ~11 Bernoulli BCE heads.
- `region_loss_weight` (default 1.0): multiplier on move_region_center MSE.
  **Deliberately lower than direction** — in Phase 1, the region target is a
  noisy proxy for ball position (the rules AI uses `GetPossessionOrder`, not
  `MoveOrder`) and inflating it raises the BC loss floor without benefit.
  The old code used `dir_w=3.0` for both — region is now decoupled.

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
existing UI scenarios, but `build_env()` in `curriculum/envs.py` raises
`NotImplementedError` for phases > 2.  Implement by adding `_build_phaseN_env()`
to `curriculum/envs.py` — `train.py` and `record_demonstrations.py` both
use `build_env()` automatically.

---

## 5. Hyperparameters — where and how to change them

All tunable constants live in **`src/footballcoach/ai/config/ai_config.json`**.
Never hardcode them.  Key sections:

```json
{
  "ppo": {
    "gamma": 0.99,              // discount factor
    "lam": 0.95,                // GAE lambda
    "clip_range": 0.1,          // PPO clipping epsilon
    "learning_rate": 1e-5,      // conservative — augmentation multiplies effective batch 12×
    "n_epochs": 4,              // PPO epochs per rollout (often early-stops after 1)
    "minibatch_size": 64,
    "rollout_steps": 4096,      // steps before each update
    "target_kl": 0.05,          // per-MINIBATCH early-stop threshold (not per-epoch)
    "ent_coef": 0.1,
    "ent_dir_weight": 0.05,     // direction head weight in log_prob AND entropy
    "augment_n_slot_shuffles": 3  // 12× effective batch; reduces needed gradient steps
  },
  "bc": {
    "pretrain_steps": 6000,               // online pre-training steps (fallback, no dataset)
    "pretrain_lr": 5e-3,
    "pretrain_online_batch_size": 16,
    "direction_loss_weight": 3.0,         // weight on cosine direction loss vs BCE heads
                                          // NOTE: keep ≤0.5 for short online pre-training
    "region_loss_weight": 1.0,            // weight on move_region_center MSE (separate from
                                          // direction — in Phase 1 this is a noisy proxy)
    "bc_pretrain_epochs": 15,             // offline BC epochs (used with --bc-dataset)
    "bc_pretrain_batch_size": 1024,
    "aux_coeff_start": 0.95,              // BC aux loss weight at PPO step 0
    "aux_coeff_end": 0.0,
    "aux_coeff_anneal_fraction": 0.35,    // fraction of total_steps over which coeff anneals
    "value_pretrain_steps": 4096,
    "value_pretrain_epochs": 15,
    "value_pretrain_lr": 6e-3,
    "bc_repair_epochs": 2,
    "bc_repair_lr": 2e-3
  },
  "reward": {
    "phase1": {
      "ball_distance_shaping": 0.05,
      "gain_possession_bonus": 1.0,
      "ball_progress_scale": 0.5,
      "ball_out_penalty": -1.0,
      "illegal_action_penalty": -0.2,
      "box_possession_terminal": 5.0
    }
  },
  "curriculum": {
    "rng_reduction_start": 0.55,
    "rng_reduction_end": 0.3
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

**Mode 1 — Combined offline pre-training (recommended):**
When `--bc-dataset` is provided, `PPOTrainer.pretrain_combined()` runs:
  0. **Phase 0** — decision-network-only warm-up on demo returns: combined
     `decision_bc_loss + phase0_value_coef * value_loss` in ONE backward pass,
     over ALL `decision_net` parameters (encoders + trunk + value_head —
     **no frozen layers**, unlike Phase 3 below). `execution_net` is NOT
     trained here. Uses the decision-heads-only `bc_loss_from_tensor(...,
     exec_heads=None)` path (skips exec_move/sprint/kick/tackle_attempt BCE
     and the move_direction cosine loss). Skipped if the dataset has no
     reward data or `demo_value_pretrain_epochs=0`. Config:
     `demo_value_pretrain_epochs`, `demo_value_pretrain_lr`,
     `demo_value_pretrain_gamma`, `phase0_value_coef` (default 1.0).
  1. **Phase 1** — N BC epochs over the dataset (all params, stable minibatch
     SGD), optionally with a joint value loss term if `demo_value_bc_coef > 0`.
  2. **Phase 2/3** — delegates to `PPOTrainer.pretrain_value()` (collect one
     on-policy rollout with the BC-warmed policy, apply augmentation, fit
     value heads for M epochs — value heads only, trunk **frozen** here, a
     different freezing decision than Phase 0). `pretrain_value()` is also
     usable standalone (used by Mode 2's fallback path) and now returns a
     dict of rollout diagnostics (`episode_returns`,
     `outcomes_vs_rules/immobile/neural`) which are logged as a
     `vs_rules(N): win%` style line, matching the main PPO rollout log format.
  3. BC degradation check (bc_loss before vs after value warm-up).
  4. Optional BC repair epochs (`bc_repair_epochs`, default 0/disabled).

This is better than online pre-training because gradients are low-variance
(minibatch over diverse dataset) and value targets are on-policy.

**Do not conflate Phase 0's freezing (removed) with Phase 2/3's freezing
(retained via `pretrain_value()`'s `_get_value_pretrain_freeze_params()`)** —
these are two different call sites with two deliberately different freezing
decisions.

**Mode 2 — Online pre-training (fallback, no dataset):**
`BCPretrainer._pretrain_online()` steps the env, accumulates
`pretrain_online_batch_size` (obs, label) pairs, then does one gradient step.
Still noisy vs. offline mode, but much better than single-sample updates.
`direction_loss_weight` must be kept low (≤0.5) for short online runs — high
values cause the direction cosine loss to corrupt the shared trunk and invert the
Bernoulli heads.  Use only when no dataset is available.

**Mode 3 — Auxiliary loss during PPO:**
At each PPO rollout step, `label_fn(env)` is called and a `BCLabel` is
stored in the rollout buffer.  During `_ppo_update()`, a BC loss term is
added: `total_loss += bc_coeff * bc_loss`.  `bc_coeff` anneals linearly
from `aux_coeff_start` to `aux_coeff_end` over `aux_coeff_anneal_fraction`
of total training steps (default 0.2 → 0.0 by 30%), then stays 0.0.

### What is supervised

`bc_loss_from_tensor()` in `bc.py` supervises:
- **Decision network**: all 7 Bernoulli heads (shoot, pass, move, tackle,
  get_possession_extra, mark, hold_position) via `binary_cross_entropy_with_logits`.
- **Execution network**: `move_direction` (cosine loss; `bc.py` normalizes
  both prediction and label before computing the loss), `sprint` (BCE from logit).
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
| `--no-head-freeze` | Skip `phase.frozen_heads` — train all decision heads during PPO |

**Head freezing**: `PPOTrainer.set_frozen_heads(names)` sets `requires_grad=False` on
named `nn.Module` attributes of `decision_net`. Applied automatically from
`phase.frozen_heads` in `phases.py` before PPO starts (after pre-training, so BC
still updates all heads). Phase 1 freezes shoot/pass/tackle/mark/hold heads, leaving
only move, get_possession, latent_vector, and trunk active during PPO.

---

## 7. Data augmentation (geometric flips + slot permutations)

**Source:** `src/footballcoach/ai/obs/augment.py`

**Applies to ALL AI training phases in this repo** — wired into `_ppo_update()`
in `PPOTrainer` so every phase automatically gets augmentation.

### What it does

Each real rollout batch is expanded by **4 × `augment_n_slot_shuffles`** before
any gradient step.  Default `augment_n_slot_shuffles=3` → **12× augmentation**.

**Geometric flips** (4 variants: identity, flip_x, flip_y, flip_xy):
- flip_x: negate all x-direction quantities (positions, velocities, heading_cos,
  attacking_direction, action vectors, BC label direction)
- flip_y: negate all y-direction quantities (heading_sin, etc.)
- Angular velocity (spin) is a pseudovector: under flip_x, spin_y and spin_z
  negate; under flip_y, spin_x and spin_z negate.
- These are **exact symmetries** of the football environment — reward and
  terminal conditions are identical under pitch reflections.

**Slot permutations** (`n_slot_shuffles` per geometric variant):
- Randomly permute the opponent-player slot ordering in `other_feat`/`exists_mask`.
- Exact for permutation-invariant attention networks.
- Speeds up learning of permutation invariance.

### Correctness of reusing old_log_probs

PPO requires `π_old(a|s)`.  For augmented samples, we reuse the original
log_prob.  This is:
- **Exact** for slot permutations (attention is permutation-invariant)
- **Approximate** for flips early in training; becomes exact as the network
  learns equivariance; provides a gradient signal *toward* equivariance

### Tuning

| Config key | Default | Effect |
|---|---|---|
| `ppo.augment_n_slot_shuffles` | `3` | Slot permutations per flip. `0` = off, `1` = 4×, `3` = 12× |

Note: 12× augmentation makes each PPO update ~12× more compute. Reduce
`rollout_steps` if training wall-clock time is too long, since each step
contributes 12 gradient-worthy samples.

### Adding augmentation to a new phase/scenario

No action needed.  `augment_batch()` operates on the encoded observation
arrays, not the scenario.  New scenarios are automatically augmented as long
as:
1. Positions are encoded as relative offsets from the observing player
2. Velocities are in world frame (both are true for all current scenarios)
3. If the scenario has asymmetric structure (e.g. goal only on one side),
   verify flip symmetry holds for your reward function before enabling flips.

---

## 8. Reading the training log

Each rollout (every 2048 steps) prints one line:

```
step=28,679 | rew=8.76 | pol=0.02 val=1.00 ent=0.25 kl=0.16  bc=2.84(x0.17) | 283sps  mv_ls=[0.01,0.00]  vs_rules(18): 65%/12%  vs_neural(16): 44%/25%
```

| Field | Meaning |
|-------|---------|
| `rew` | Mean episode return over last 20 episodes |
| `val` | **Normalised MSE** of the value head: `MSE(predicted, GAE_return) / Var(returns)`.  ~1.0 = predicting the mean (no better than constant).  <0.5 = critic is useful.  0.85 after warmup is expected — it improves as returns stabilise |
| `ent` | Policy entropy (higher = more exploration) |
| `kl` | Approximate KL divergence from old policy.  >0.1 = large update (KL diagnostics printed separately). Repeated >1.0 = policy diverging |
| `bc=X(xY)` | BC auxiliary loss value × current annealing coefficient.  Disappears once coeff reaches 0 |
| `sps` | Decision steps per second |
| `mv_ls` | `move_direction` log-std for both output dimensions (tracks direction head confidence) |
| `vs_rules(N): W%/L%` | Trainee win% / opponent win% in the N **rules-based opponent** episodes this rollout |
| `vs_neural(N): W%/L%` | Same for **neural opponent** episodes (shared-weight self-play).  Compare to `vs_rules` to see if improvement is vs the rules AI or just self-play |
| `act: mv=XX gp=XX spr=XX ...` | Per-head mean activation rate (0–100%) from stored buffer actions. Values near 0 or 100 = saturated head (collapse warning). Zero extra compute — reads from buffer directly |

Offline BC epoch lines (during `pretrain_combined`):
```
  BC epoch 5/50: bc_loss=0.52  dir_cos=0.34
```
`dir_cos` is the mean cosine similarity between predicted and label `move_direction` for valid steps in that epoch.  Should rise toward 1.0 across epochs.

---

## 9. Adding a new training scenario

All scenarios live in **`src/footballcoach/ui/scenarios.py`** — one source of
truth for both the training loop and the UI scenario picker.

**Pattern** (copy from `build_1v1_scenario` as the template):

1. Write a `build_my_scenario(rng_reduction=0.3, *, param1=...) -> Match` function.
   - Use `random.Random()` internally (not global `random`) so it's re-seedable.
   - Use `generate_attributes(tier=..., rng=rng)` for player attributes.
   - Set `player.stamina`, `player.heading_rad` manually as needed.
   - Assign `player.ai = Phase1RulesAI()` (or another `PlayerAI` subclass from
     `footballcoach.rules_ai`) to every non-trainee player that needs per-tick
     logic.  `Match.step()` calls `ai.act(player, match, tick)` automatically.
   - Return the `Match`; trainee player id should be a stable string like `"trainee"`.

2. Add a `ScenarioDefinition(key="my_key", ..., build=build_my_scenario,
   on_tick=None, params=[...])` entry to the `SCENARIOS` list.
   (`on_tick` is still available for rare cross-player coordination that can't
   live on a single player's AI, but almost all scenarios should not need it.)

4. Add a `CurriculumPhase(phase_id=N, scenario_key="my_key", ...)` to
   `curriculum/phases.py` and `ALL_PHASES` / `PHASES_BY_ID`.

5. Add a `_build_phaseN_env(phase)` function in `train.py` that:
   - Creates a `ScenarioDefinition` wrapping the new builder.
   - Creates `ScenarioEnv(definition=defn, trainee_player_id="trainee", phase=N, **phase.env_kwargs)`.

6. Add the new phase to `_build_env(phase)` in `train.py`.

---

## 10. Environment wrapper — ScenarioEnv

```
src/footballcoach/ai/env/scenario_env.py
```

Key behaviour:
- `reset()` calls `ScenarioLoop._start_trial()`, then assigns `NeuralPlayerAI`
  to the trainee (and secondary players) when `sample_action_fn` is set.
- `step()` (no args) ticks the sim 15 times; all player AI fires inside
  `Match.step()`.  After the tick loop, reads `player.ai.last_transition` to
  populate `env.last_trainee_transition` and `env.last_secondary_results`.
- Returns `(obs, reward, done, StepInfo)`.
- `done` is `True` when `ScenarioLoop.step()` returns `True` (trial ended),
  OR the episode tick limit is reached, OR **any** player reaches the opponent
  box with possession (phase 1).
- The `timeout_ticks` passed to `ScenarioLoop` is `max_episode_s / dt_s`
  (not the ScenarioLoop default of 500); this ensures 120s episodes work.

**Trainee player id**: must match the `player_id` string used in the
scenario's `build_*` function (e.g. `"trainee"` for phase 1,
`"kicker"` for phase 2).

**Secondary player training** (`secondary_player_ids`, `sample_action_fn`):
- Pass `secondary_player_ids=["opponent", ...]` to also drive those players
  with the shared neural network.
- Set `env.sample_action_fn = trainer._sample_action` (done automatically by
  `PPOTrainer.train()`).  `ScenarioEnv.reset()` then assigns `NeuralPlayerAI`
  to the trainee and all secondary players.
- After each `step()`, `env.last_trainee_transition` holds the PPO data for
  the trainee; `env.last_secondary_results` holds the same for secondary
  players (list of dicts `{obs, action, log_prob, value, raw_exec, reward, done}`).
  Both are populated from `player.ai.last_transition` inside `step()`.
- Secondary players are **skipped** (no neural sampling) when
  `match._opponent_use_rules_ai` is True — their orders come from `player.ai`
  instead.  Mixing neural + `player.ai` orders on the same player corrupts the
  episode.

---

## 10. Tests

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

## 11. Common gotchas

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

### `_sample_action` returns an 8-tuple (not 7)
As of the bug-fix commit (2026-07-29), `_sample_action` returns:
```
(action, log_prob, value, decision_probs, execution_physical,
 decision_physical, target_slots, raw_exec_samples)
```
`raw_exec_samples` is a dict of numpy arrays (`sprint`, `kick`,
`tackle_attempt`, `move_dir_raw`, `kick_dir_raw`) that must be passed to
`_action_to_numpy(action, raw_exec_samples)` so the rollout buffer stores
the real sampled values.  Any test or downstream code that unpacks the tuple
must expect 8 elements.

### Execution samples MUST be stored from `_sample_action`, not re-derived
The PPO importance ratio `exp(new_lp - old_lp)` requires that `old_lp` and
`new_lp` are computed over **identical** action components.  Three early
bugs caused `old_lp` to include terms that `new_lp` never matched:
1. `sprint/kick/tackle_attempt` were stored as `0.0` (fixed: stored from samples)
2. `move_dir_raw/kick_dir_raw` were not stored at all (fixed: stored + recomputed
   via `DirectionHead` in `_recompute_log_prob`; direction heads are now
   included in the PPO ratio — see design doc 8.6)
3. `bc_loss_val` was not detached before `.item()` (fixed: `.detach().item()`)

Symptoms of a recurrence: `approx_kl` > 10 at step 1, `value_loss` doubling
every rollout.

### Observation slot order is random — this is intentional
`encode_observation()` shuffles real players into random slots each call.
Do not cache slot-to-player mappings between steps.  Use `slot_player_ids`
(returned by `_sample_action`) to convert a target slot index to a player id
when building engine orders.

### Direction heads are unit-normalized in `forward()` — included in PPO ratio
`ExecutionNetwork.forward()` L2-normalizes `move_direction` and
`kick_direction` to unit vectors before returning them in `ExecutionHeadsRaw`.
This constrains the distribution mean to the unit circle (max mean-shift = 2),
bounding the KL contribution to O(1) per step. Consequently **both direction
heads are included in `_compute_log_prob` / `_recompute_log_prob`** like all
other heads — the old exclusion and the `dir_l2` loss penalty are gone.
The stored rollout values `move_dir_raw`/`kick_dir_raw` are still the noisy
samples (drawn from Normal(unit_mean, std)) and are *not* on the unit circle —
they're normalized separately when computing the physical direction for the engine.

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

## 12. Current state and next steps

### What is done
- Full architecture implemented and tested: entity encoder + attention,
  decision + execution networks, custom PPO loop with GAE, rollout buffer,
  masking, all action distributions (Bernoulli/Categorical/Normal/Direction),
  phase1 and phase2 reward functions, curriculum phase definitions, EMA
  smoothing for attack/defence weighting.
- Phase 1 scenario (`build_1v1_scenario`): random placement, attributes,
  stamina, heading; random ball velocity/spin/restitution.
  - **Trainee team randomised** each episode (attacks either end).
  - **Ball max speed increased to 10 m/s**.
  - **Episode ends when either player reaches the opponent's box** with
    possession (not just the trainee).
  - **Opponent 50/50**: rules-based AI or neural network per episode.
- End-to-end smoke test suite in `tests/ai_scenario/`.
- Training loop confirmed to run: first PPO update at step=2048 produces
  finite losses and a positive mean episode reward from ball-chasing behaviour.
- Checkpoint saving after every rollout and at end of run.
- **Dual-player training**: `ScenarioEnv` accepts `secondary_player_ids` and
  drives those players with the shared neural network (when not in rules-based
  mode).  Their transitions are drained into the same rollout buffer by
  `PPOTrainer.train()`.  Phase 1 training uses `secondary_player_ids=["opponent"]`.
- **Behavioural cloning (BC) bootstrapping** (`src/footballcoach/ai/ppo/bc.py`):
  - Pre-training phase: pure supervised BCE on rules-based labels for N steps before PPO.
  - Auxiliary loss during PPO: BC cross-entropy added to PPO loss with linearly annealed coefficient.
  - Both modes train **decision network** (all 7 Bernoulli heads) AND **execution network** (`move_direction`, `sprint`).
  - Phase 1 label fn: `bc.phase1_labels(env, player_id=None)` — team-aware,
    works for any player; toward ball when loose, toward own attacking goal when in possession.
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

### Phase 1 opponent progression
Current state: **50% rules-based / 50% neural each episode** (automated).
The `match._opponent_use_rules_ai` flag (set in `build_1v1_scenario`) controls
which mode applies.  `Phase1RulesAI` (via `opponent.ai`) drives the opponent in
rules-based episodes; `ScenarioEnv` drives it with the shared network in
neural episodes.
Next step: add frozen older-generation checkpoints as a third opponent mode.

### Phase 2 (shooting) — not yet run
Ready to train.  Scenario is `build_penalty_scenario`.  Run with `--phase 2`.

### Phases 3 & 4 (passing, tackling) — stubs only
`NotImplementedError` in `train.py::_build_env()`.  Scenarios exist in
`ui/scenarios.py` (`build_pass_scenario`, `build_tackle_scenario`).
Needs: curriculum phase reward functions, `_build_phase3_env()` etc. in
`train.py`, and potentially new `phase3_reward()` / `phase4_reward()` in
`env/reward.py`.
