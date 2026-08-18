# Ball Physics Pretraining Plan — Frozen Dynamics Latent for BallFeatures

> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

## 0. Status

**Not started.** This document is the full design; no code exists yet.
Ball-only for this pass — the analogous player-dynamics network is a
deliberately deferred follow-up (see §11) once this one is validated.

---

## 1. Motivation (why this exists)

Recap of the design discussion that produced this plan (see chat history if
more context is needed, but this section should be self-sufficient):

- The RL/BC networks currently see only the *instantaneous* ball state
  (`BallFeatures`: position, velocity, spin, possession — 11 floats) each
  decision tick. Reward-relevant outcomes like "will this ball go out of
  bounds" or "will this cross the goal line" are highly nonlinear functions
  of that state (bounces, spin/Magnus curve, drag) that the network
  currently has to learn purely from a comparatively sparse RL/BC signal.
- A separate self-supervised network, pretrained to predict the ball's own
  future state under the real physics engine, can pre-learn that nonlinear
  structure once, cheaply, and hand the live policy a compact summary
  (a bottleneck latent) instead of making it re-derive "ball physics
  intuition" from scratch via reward gradients.
- **Why a learned net instead of literally forward-simulating the real
  engine as a feature**: physics stepping is already the documented
  bottleneck of training (`ai_trainer_knowledge.md` §3.7 — CPU-bound,
  GIL-serialized, the whole reason `ppo.n_parallel_envs` exists). Recomputing
  a 10s ball rollout every decision tick, for every parallel env, would add
  real cost to the tightest part of the loop. A frozen MLP forward pass is a
  single cheap matmul, trained once (retrained only if `physics.json`'s ball
  params change, which is rare).
- **Why ball first, not ball+player together**: the ball's future state is a
  pure function of physics (fully determined by position/velocity/spin/
  restitution/pitch geometry — no unknown "intent" term). A player's future
  state additionally depends on unresolved future decisions (though giving
  `desired_direction`/`desired_speed_mode` as inputs, held fixed, turns it
  into an analogous pure-physics problem — see §11). Validating the
  approach on the strictly simpler, unambiguous case first de-risks the
  player version.

---

## 2. Scope boundary

**In scope**: a standalone ball-dynamics episode generator, a small
autoencoder-style network (encoder + 5-horizon decoder), a training script,
and wiring the *frozen encoder only* into `DecisionNetwork`/
`ExecutionNetwork`'s existing `ball_mlp` input.

**Explicitly out of scope for this pass** (do not implement speculatively):
- The player-dynamics network (§11 sketches it for a future plan doc).
- Any change to `BallFeatures`, `BALL_FEATURE_DIM`, `encode_observation()`,
  or recorded `.npz` BC datasets — see §8, this design deliberately avoids
  touching the observation schema at all.
- An "at rest" / "time to stop" auxiliary output, or any output beyond
  what's specified in §5. Easy to add later; not needed to validate the
  approach.
- Making the latent horizon-queryable at arbitrary continuous t (a
  horizon-conditioned decoder). Five fixed decoder heads is enough and
  simpler to train; revisit only if a concrete need for other horizons
  shows up.

---

## 3. Inputs and outputs (final spec)

### 3.1 Encoder input — 14 floats, one ball state at t=0

All fields read from a `Ball` + `BallPhysicsParams` + `Pitch` triple. Same
normalization conventions as `BallFeatures`/`GlobalFeatures`
([obs/schema.py](../src/footballcoach/ai/obs/schema.py)) so the encoder's
input distribution matches what it'll see at inference time inside the live
network (see §8).

| # | Field | Source | Normalization |
|---|---|---|---|
| 0 | `pos_x` | `ball.position.x` | `/ pitch.half_length` (see §4.2 — NOT the fixed 52.5 `BallFeatures` uses, since pitch size is randomized during pretraining) |
| 1 | `pos_y` | `ball.position.y` | `/ pitch.half_width` |
| 2 | `height_m` | `ball.position.z` | `/ height_norm_m` (same constant as `ai_config.json`'s `observation.height_norm_m`) |
| 3 | `velocity_x` | `ball.velocity.x` | `/ pitch_half_diagonal` (computed from the episode's own randomized pitch dims) |
| 4 | `velocity_y` | `ball.velocity.y` | `/ pitch_half_diagonal` |
| 5 | `velocity_z` | `ball.velocity.z` | `/ pitch_half_diagonal` |
| 6 | `spin_x` | `ball.spin.x` | `/ ball_spin_norm_max_rad_s` (`ai_config.json`) |
| 7 | `spin_y` | `ball.spin.y` | `/ ball_spin_norm_max_rad_s` |
| 8 | `spin_z` | `ball.spin.z` | `/ ball_spin_norm_max_rad_s` |
| 9 | `restitution` | `params.bounce_restitution_vertical` | raw value, already ∈ [0,1]-ish range (physics.json default ~0.6) |
| 10 | `pitch_length_norm` | `pitch.length_m` | `/ 105.0` |
| 11 | `pitch_width_norm` | `pitch.width_m` | `/ 68.0` |
| 12 | `goal_width_norm` | `pitch.goal_width_m` | `/ 7.32` |
| 13 | `goal_height_norm` | `pitch.goal_height_m` | `/ 2.44` |

Box dims (`box_length`/`box_width`) are deliberately excluded — irrelevant
to ball flight or the out/goal boundary checks, GK-box-specific only.

### 3.2 Decoder output — 11 floats × 5 horizons = 55 floats

Horizons: **t = 0.2s, 1s, 3s, 5s, 10s** (fixed, one decoder head per
horizon — see §6.2).

Per horizon:

| Field | Same normalization as input row |
|---|---|
| `pos_x, pos_y, height_m` | as §3.1 |
| `velocity_x, velocity_y, velocity_z` | as §3.1 |
| `spin_x, spin_y, spin_z` | as §3.1 |
| `out_of_bounds` | binary target, see §4.3 freeze semantics |
| `goal_scored` | binary target, see §4.3 |

### 3.3 Bottleneck latent — 16 floats (config default)

This is the **only** part that survives into the live policy network (see
§8). Configurable via `ai_config.json["physics_pretrain"]["ball"]
["latent_dim"]`; default `16` — well below the existing `network.latent_dim`
(48, the *decision*-network's own strategic latent) since this is a much
lower-dimensional problem (14 inputs, no attention/entity structure).

---

## 4. Episode generation

New module:
[`src/footballcoach/ai/physics_pretrain/ball_episode_gen.py`](../src/footballcoach/ai/physics_pretrain/ball_episode_gen.py)
(new package — see §9 for full file layout).

### 4.1 Why this can be standalone (no `Match` needed)

`step_ball(ball, dt_s, params)` and `resolve_goal_boundary(ball, pitch,
params)` in
[engine/ball_physics.py](../src/footballcoach/engine/ball_physics.py) only
touch a `Ball` + `BallPhysicsParams` + `Pitch` — no player, no `Match`, no
engine tick loop. One episode = construct a `Ball` with a random state,
step it in a tight loop, no per-tick AI/order/collision machinery to pay
for. This is what makes bulk generation cheap.

### 4.2 Randomization ranges

One episode = one random draw, run once, not a multi-decision rollout like
`ScenarioEnv`. Per episode, independently sample:

| Quantity | Distribution | Notes |
|---|---|---|
| `pitch.length_m` | `Uniform(105 * 0.67, 105 * 1.33)` | ±33% around standard, per user request |
| `pitch.width_m` | `Uniform(68 * 0.67, 68 * 1.33)` | independent of length |
| `pitch.goal_width_m` | `Uniform(7.32 * 0.67, 7.32 * 1.33)` | independent of pitch dims |
| `pitch.goal_height_m` | `Uniform(2.44 * 0.67, 2.44 * 1.33)` | independent |
| `restitution` (`bounce_restitution_vertical`) | `Uniform(default * 0.67, default * 1.33)`, clamped to `[0.2, 0.95]` (same clamp `ui/scenarios.py` already uses) | wider than live-game's episode σ=0.08 Gaussian — deliberate, see §1 |
| `ball.position` | uniform within pitch bounds (with margin), `height_m` uniform `[0, ~3m]` — include some already-airborne starts (post-kick states), not just grounded | |
| `ball.velocity` | random direction, speed `Uniform(0, 30 m/s)` — wider than live Phase-1's 10 m/s cap so the net generalizes to hard shots/clearances, not just loose-ball speeds | |
| `ball.spin` | random axis, magnitude `Uniform(0, ball_spin_norm_max_rad_s)` | |

All ranges are `ai_config.json["physics_pretrain"]["ball"]` keys — **never
hardcode them in the generator**, matching every other tunable constant in
this codebase.

Roughly **10–20% of episodes should start already out of pitch bounds or
already past the goal line** (e.g. a rebound scenario) so the net also
learns the boundary-adjacent region well, not just "typical in-play" states.

### 4.3 Simulation and the freeze-on-event rule

- Step at `dt_s = ai_config.json["observation"]["sim_dt_s"]` (reuse the
  existing training-time physics dt, currently 0.05s — 200 ticks to cover
  the full 10s horizon) inside a tight loop calling `step_ball()` then
  `resolve_goal_boundary()` each tick (same order `Match.step()` uses for a
  loose ball).
- After each tick, check `pitch.is_in_bounds(ball.position)` (→
  `out_of_bounds` if false) and `check_goal(ball, pitch)` (→ `goal_scored`
  if not `None`) — same functions the real engine uses
  ([entities/pitch.py](../src/footballcoach/entities/pitch.py),
  [engine/scoring.py](../src/footballcoach/engine/scoring.py)).
- **The instant either fires, freeze**: record that event tick's
  `(pos, velocity, spin)` and hold it for every later horizon still to be
  recorded, and latch both flags true from then on (a ball that went out at
  t=2s is still "gone out" at t=10s; if it *also* would have re-entered
  bounds under further hypothetical physics, we don't care — the real match
  would have already stopped play). This was confirmed with the user
  in the design discussion — do not implement free-running physics past an
  event.
- If neither event fires within 10s, all 5 horizons get real simulated
  state and both flags `0.0` throughout (a ball that stayed in play,
  presumably coming to rest via rolling friction well before 10s in most
  draws).
- Record the state snapshot at each of the 5 horizon ticks (nearest tick to
  0.2/1/3/5/10s at the chosen dt — exact if `dt_s` evenly divides all five,
  which `0.05` does).

### 4.4 Dataset format and volume

- Output: `.npz` shards, one row per episode: `input[14]`, `target[55]`
  (5×11, flattened, horizon order matches §3.2). Mirrors the existing
  `DemonstrationDataset` shard convention
  ([ai/bc/dataset.py](../src/footballcoach/ai/bc/dataset.py)) enough to
  reuse its shard-loading/train-val-split plumbing where sensible, but this
  is a fresh, simpler dataset class (no episode/BC-label structure to
  carry) — see §9.
- Target volume: start with **200k episodes** (config default,
  `n_episodes`). At 200 ticks/episode this is 4×10⁷ `step_ball()` calls —
  expect this to take a while in single-process pure Python. Parallelize
  the same way `rollout_worker.py` already does for PPO rollouts (plain
  `multiprocessing`, each worker generates an independent shard, no shared
  state needed since episodes are fully independent draws) rather than
  inventing a new parallelism pattern.
- Hold out 15% of episodes as a validation split (same
  `split_train_val_indices()`-style episode-level split used elsewhere, not
  a row-level split — trivial here since each "episode" is already one row,
  but keep the same naming convention for consistency).

---

## 5. Loss function

Per horizon `h`, per row:

- **Continuous fields** (`pos`, `velocity`, `spin` — 9 of the 11 outputs):
  plain MSE, summed then averaged over the batch. No cosine/direction-style
  loss needed here (unlike BC's `move_direction`) — these are absolute
  quantities, not primarily-directional ones.
- **Event flags** (`out_of_bounds`, `goal_scored`): `binary_cross_entropy_
  with_logits`, with `pos_weight` computed from the dataset's own class
  balance (expect `goal_scored` especially to be a small minority class —
  reuse the same inverse-frequency `pos_weight` pattern as
  `DemonstrationDataset.compute_pos_weights()` in `ai/bc/dataset.py` rather
  than inventing a new balancing scheme).

Total loss = `sum over horizons of (continuous_mse + event_bce)`, no
cross-horizon weighting initially (flag in the training log if t=10s clearly
dominates/is starved relative to t=0.2s — the freeze-on-event rule means
later horizons have a much higher positive rate for the event flags, which
should make them *easier*, not harder, so no a priori reason to expect an
imbalance here, but verify empirically before adding a weighting scheme).

---

## 6. Network architecture

New module:
[`src/footballcoach/ai/physics_pretrain/ball_dynamics_net.py`](../src/footballcoach/ai/physics_pretrain/ball_dynamics_net.py).

### 6.1 `BallDynamicsEncoder` (the piece that gets frozen)

Plain MLP, no attention/entity structure needed (single fixed-size input,
no variable-length player set):

```
input (14) -> Linear(hidden) -> ReLU -> Linear(hidden) -> ReLU -> Linear(latent_dim)
```

`hidden` config default `64`. No dropout, no batchnorm — this network is
frozen and always run in the same (implicit) eval mode, so keep it a
deterministic pure function of its input (see §8's note on why the existing
`value_dropout` caveat about `.eval()` not being called during rollout is
worth avoiding entirely here rather than working around).

### 6.2 `BallDynamicsDecoder` (training-only, discarded after freezing)

Five independent small heads reading the same latent (not a shared
horizon-conditioned decoder — see §2's scope note):

```
for each of 5 horizons:
    Linear(latent_dim -> 32) -> ReLU -> Linear(32 -> 11)
```

Output layout per head: `[pos_x, pos_y, height_m, vel_x, vel_y, vel_z,
spin_x, spin_y, spin_z, out_of_bounds_logit, goal_scored_logit]` — first 9
are direct regression targets, last 2 are BCE logits.

### 6.3 `BallDynamicsAutoencoder`

Thin wrapper composing encoder + all 5 decoder heads for training
convenience (`forward(x) -> (latent, [head_0..head_4])`). Not itself saved
anywhere permanent — only `encoder.state_dict()` becomes a real artifact
(§7).

---

## 7. Training script and config

New script:
[`src/footballcoach/ai/physics_pretrain/train_ball_dynamics.py`](../src/footballcoach/ai/physics_pretrain/train_ball_dynamics.py)
(or under `ai/scripts/` if you want it discoverable alongside `train.py`/
`record_demonstrations.py` — pick whichever matches how the rest of the
plan's file layout lands, no strong reason to prefer one over the other).

```bash
uv run python -m footballcoach.ai.physics_pretrain.train_ball_dynamics \
    --dataset physics_pretrain_data/ball/ \
    --output checkpoints/physics_pretrain/ball_encoder.pt \
    --epochs 50 --batch-size 1024
```

Plain supervised minibatch SGD (Adam), episode-level 85/15 train/val split,
early stop + best-weights restore on val loss — same pattern already
established by `bc_pretrain_early_stop_patience`/`value_pretrain_early_stop_
patience` in `ppo_trainer.py`, reused here rather than reinvented. Log
per-epoch: total loss, per-horizon continuous MSE, per-horizon event BCE
(both flags), matching the "floor-adjusted breakdown" spirit of the existing
BC logging (a raw event-BCE number is close to meaningless without knowing
the label's own entropy floor under whatever smoothing is used, if any).

New `ai_config.json` section:

```json
"physics_pretrain": {
  "ball": {
    "hidden_dim": 64,
    "latent_dim": 16,
    "horizons_s": [0.2, 1.0, 3.0, 5.0, 10.0],
    "n_episodes": 200000,
    "pitch_scale_range": [0.67, 1.33],
    "restitution_scale_range": [0.67, 1.33],
    "ball_speed_max_mps": 30.0,
    "out_of_bounds_start_frac": 0.15,
    "epochs": 50,
    "batch_size": 1024,
    "lr": 1e-3,
    "early_stop_patience": 5,
    "early_stop_min_delta": 1e-4
  }
}
```

**Output artifact**: `checkpoints/physics_pretrain/ball_encoder.pt`
containing `{encoder_state_dict, config_snapshot, dataset_hash_or_stats}` —
the config snapshot and dataset stats are worth saving alongside the weights
so a later "did the physics change since this was trained" check is
possible without re-deriving it from scratch (a plain warning at load time
if `physics.json`'s ball section hash doesn't match what's recorded here is
enough — see §8).

---

## 8. Integration into the live policy network

This is the part most worth getting right — see the design discussion that
led here for the reasoning; summarized:

### 8.1 Rejected approach: adding fields to `BallFeatures`

The obvious-looking approach — append the 16 latent floats as new trailing
fields on the `BallFeatures` dataclass, bump `BALL_FEATURE_DIM` — was
considered and **rejected**. Two problems:

1. **Checkpoint/dataset invalidation**: same class of break as the prior
   `PLAYER_FEATURE_DIM` 26→28 change (`ai/knowledge.md` §"3.4") — every
   existing checkpoint AND every recorded `.npz` BC dataset would need
   regenerating, since `encode_observation()` and the recorded arrays are
   the same code path.
2. **Canonical-frame mirroring is undefined for an opaque latent**:
   `obs/augment.py` derives `BALL_FLIP_X_IDX` (and the parallel `flip_y`
   set) by iterating `fields(BallFeatures)` at import time, and
   `obs/canonical.py`'s `CanonicalNetworkWrapper` negates exactly those
   indices for a `Team.RIGHT` observer. A learned latent vector has no
   inherent per-dimension sign — naively including it in that
   auto-derivation would be wrong (negating components that were never
   designed to be an equivariant geometric quantity), and naively
   *excluding* it would mean the latent stays in raw world-frame while
   every other ball/player field the network sees has been mirrored into
   "my team attacks +x" canonical frame — reintroducing exactly the kind of
   raw-frame asymmetry the canonical wrapper exists to eliminate, just for
   this one feature block.

### 8.2 Chosen approach: compute the latent inside the network `forward()`, not in `encode_observation()`

Insert the frozen encoder call **inside `DecisionNetwork.forward()` and
`ExecutionNetwork.forward()`**, near the top, before `ball_feat` reaches
either of its **two** existing consumers
([models/decision_network.py](../src/footballcoach/ai/models/decision_network.py),
[models/execution_network.py](../src/footballcoach/ai/models/execution_network.py),
[models/entity_encoder.py](../src/footballcoach/ai/models/entity_encoder.py)):

1. `self.ball_mlp` — `Linear(ball_feat_dim, ball_mlp_hidden) → ReLU`, feeds
   the trunk directly.
2. `entity_encoder.ball_query_proj` — `Linear(ball_feat_dim, embed_dim)`,
   added as a bias onto the self-attention query (`entity_encoder.py:73,
   128-129`) so player-attention can be ball-state-conditioned.

Both are separate, independently-weighted modules that happen to consume
the same raw `ball_feat` tensor — so build the widened tensor once and feed
it to both, rather than patching either module in isolation:

```python
physics_latent = self.ball_physics_encoder(ball_feat, global_feat)  # frozen, no_grad, (batch, 16)
physics_latent = physics_latent * loose_mask  # zero out when possessed, see 8.3
ball_feat_aug = torch.cat([ball_feat, physics_latent], dim=-1)      # (batch, 27)

ball_embed = self.ball_mlp(ball_feat_aug)                            # Linear(27, ball_mlp_hidden) now
context = self.entity_encoder(..., ball_feat=ball_feat_aug, ...)     # ball_query_proj: Linear(27, embed_dim) now
```

`ball_feat_dim` — the single constant threaded through both `ball_mlp`'s
constructor and `EntityEncoder.__init__(ball_feat_dim=...)` — goes from
`BALL_FEATURE_DIM` (11) to `11 + physics_latent_dim` (27 at the default
`latent_dim=16`). This is a `network` config change, not an
`observation`/schema change.

**Why this is correct with zero special-casing for mirroring/augmentation**:
by the time `forward()` runs, `ball_feat` has *already* been mirrored by
`CanonicalNetworkWrapper` (for `Team.RIGHT` observers) and already reflects
whatever `flip_y` augmentation variant `augment_batch()` produced for this
particular sample. Recomputing the physics latent from that same, already-
transformed `ball_feat` means it is automatically consistent with whatever
frame the rest of the network's input is in for that forward call — no
separate flip-index bookkeeping needed, and it exactly follows the existing
precedent in `canonical.py`: *"There is exactly ONE implementation of the
mirror... used automatically by every existing call site with zero changes
to those call sites."* We're extending that same principle one step further
rather than fighting it.

Tradeoff: the frozen encoder now runs on every forward pass (both networks,
all 12× augmented samples per PPO batch) instead of once per raw
observation. This is deliberately accepted — it's a tiny frozen MLP
(14→64→64→16, no attention), the cost is negligible next to the rest of
each network's forward pass, and it avoids a much worse alternative
(threading a precomputed-but-frame-dependent latent through `augment_batch()`
and the wrapper's mirror step by hand).

**`global_feat` slicing**: the encoder needs `restitution`,
`pitch_length_norm`, `pitch_width_norm`, `goal_width_norm`,
`goal_height_norm` from `global_feat`, and the 9 raw physics fields from
`ball_feat`. Slice by field name via `[f.name for f in
fields(BallFeatures)].index(...)` /
`[f.name for f in fields(GlobalFeatures)].index(...)` (same
robustness-to-reordering pattern `canonical.py`'s `X_SIGN_FIELD_IDX` already
uses) rather than hardcoded integer offsets — write this once as a small
shared helper (e.g. `ball_physics_encoder_inputs(ball_feat, global_feat) ->
Tensor[14]`) used by both `DecisionNetwork` and `ExecutionNetwork` so the
slicing logic exists in exactly one place.

### 8.3 Masking — only meaningful when the ball is loose

`BallFeatures.is_loose` is already the last field (index 10). Use it
directly as the multiplicative mask on the latent (`physics_latent *
ball_feat[..., IS_LOOSE_IDX:IS_LOOSE_IDX+1]`) — no new mask tensor, no
schema change, mirrors the existing `exists_mask` convention in spirit
(zero-fill + an existing flag the downstream trunk can condition on) without
needing a literal new mask field.

### 8.4 Freezing, checkpointing, and loading

- `DecisionNetwork.from_config()`/`ExecutionNetwork.from_config()` gain a
  `ball_physics_encoder_path` argument (config-driven, e.g.
  `network.ball_physics_encoder_checkpoint`), always loaded from the
  standalone pretrain artifact (§7's `ball_encoder.pt`) — **never** from
  the main PPO checkpoint. `requires_grad_(False)` on load; excluded from
  the optimizer entirely (not even in a param group, same treatment as
  `decision_net.value_head` gets under the "single value head convention").
- **Deliberately excluded from the main PPO checkpoint's `state_dict()`**
  (skip its keys in `_save_checkpoint()`/`_save_checkpoint_to()`, matching
  how `separate_value_net` mode already special-cases which submodules get
  saved where) — the frozen encoder is an external, versioned artifact, not
  training state. This also sidesteps any risk of a stale/frozen-at-a-worse-
  checkpoint version silently traveling inside old PPO checkpoints.
- `load_checkpoint()` should log a `WARNING` (not raise) if the configured
  `ball_encoder.pt`'s recorded config/dataset hash (§7) doesn't match what a
  fresh `from_config()` would load — cheap staleness detection, doesn't
  block loading (mirrors the existing tolerant-load pattern for shape-
  mismatched params elsewhere in `load_checkpoint()`).
- This is a genuine **new architecture change** — old checkpoints predating
  this feature will fail to load `ball_mlp`'s widened input layer. Expect
  and document this the same way `--reset-value-weights` documents its own
  precondition: existing checkpoints need `ball_mlp`'s first-layer weights
  reset (or the whole run restarted) once this lands. **Recorded `.npz` BC
  datasets are NOT affected** — this is the one major advantage of the §8.2
  approach over §8.1: the observation schema never changed, so the existing
  `demonstrations/phase1/` data stays valid.

---

## 9. File layout summary

```
src/footballcoach/ai/physics_pretrain/         # new package
  __init__.py
  ball_episode_gen.py      # random ball-episode generation, §4
  ball_dataset.py           # .npz shard load/save, train/val split, §4.4
  ball_dynamics_net.py      # BallDynamicsEncoder / Decoder / Autoencoder, §6
  train_ball_dynamics.py    # CLI training script, §7

src/footballcoach/ai/config/ai_config.json      # + "physics_pretrain.ball" section, §7
src/footballcoach/ai/models/decision_network.py  # ball_mlp input widened, encoder call added, §8.2
src/footballcoach/ai/models/execution_network.py # same, §8.2
src/footballcoach/ai/obs/canonical.py            # NOT modified (see §8.1's rejected-approach note — confirm no change is needed here, that's the point)

checkpoints/physics_pretrain/ball_encoder.pt     # frozen artifact, §7
physics_pretrain_data/ball/                       # generated .npz shards, §4.4 (gitignored, same as demonstrations/)

tests/ai_unit/test_ball_physics_pretrain.py       # §10
```

---

## 10. Testing plan

- **`ball_episode_gen`**: determinism given a seeded RNG; freeze-on-event
  correctness (construct a hand-picked state that goes out at a known tick,
  assert all later horizons hold the frozen value and latched flags);
  distribution sanity (a large sample's out-of-bounds/goal rates are
  nonzero and not saturated at 0%/100%).
- **`BallDynamicsEncoder`/`Decoder`**: forward-pass shape tests, no-NaN
  guarantee (same convention as `test_obs_encoder.py`/`test_networks.py`).
- **Loss function**: hand-computed reference case for the MSE+BCE
  combination on a tiny synthetic batch.
- **Training script smoke test**: a few epochs on a tiny synthetic dataset
  (~100 rows), assert loss decreases and the saved artifact round-trips
  (`load` → same weights).
- **Integration**: `DecisionNetwork`/`ExecutionNetwork` forward pass with
  the frozen encoder wired in — assert (a) output shapes unchanged from the
  caller's perspective, (b) `ball_physics_encoder`'s parameters never
  receive gradients after a backward pass, (c) the latent is exactly zero
  for a `is_loose=0` row and nonzero for `is_loose=1` (masking works), (d)
  running the SAME raw ball state through the wrapper for a `Team.LEFT` vs
  `Team.RIGHT` observer (with x mirrored) produces the mirrored-consistent
  latent difference expected — i.e. confirm §8.2's "no special-casing
  needed" claim actually holds rather than just asserting it in prose.

---

## 11. Follow-up: player dynamics network (not implemented here)

Sketch only, for whoever picks this up next, once the ball version is
validated end-to-end (trained, wired in, and shows some measurable effect —
e.g. faster BC convergence on kick/out-of-play-adjacent labels, or better
value-loss calibration near box/boundary states — before investing further):

- **Input (18 floats)**: `pos_x, pos_y, heading_rad(sin/cos), velocity_x,
  velocity_y, stamina, top_speed, acceleration, stamina_attr, ball_control,
  has_possession, desired_direction(sin/cos), desired_speed_mode(one-hot,
  3), pitch_length_norm, pitch_width_norm, goal_width_norm,
  goal_height_norm`. `heading_rad` is required, not redundant with
  velocity — confirmed via `engine/movement.py::step_player_towards`:
  `player.velocity` is reconstructed from `heading_rad` every tick
  (`Vector3.from_angle_xy(new_heading, new_speed)`), so heading is exactly
  recoverable from velocity when speed > 0, but at STANDSTILL (speed = 0)
  `heading_rad` is read directly and still drives next-tick
  `turn_speed_penalty` — two standstill players at the same position with
  different stored headings diverge onto different trajectories the moment
  either starts moving, which velocity alone (all-zero) can't distinguish.
- **Output per horizon**: `pos_x, pos_y, velocity_x, velocity_y,
  heading_rad(sin/cos), stamina, out_of_bounds, goal_scored` (the latter
  only meaningfully supervised on `has_possession=1` rows, treating the
  ball as glued to the player's position — dribble-carry assumption).
- Same freeze-on-event, same 5 horizons, same encoder-forward-time
  injection principle as §8.2 — but the concrete placement is **different**
  from the ball's, and worth spelling out precisely rather than waving at
  "the entity encoder":

  **Why it's a different layer, concretely.** The ball is a single global
  entity — one latent per observation, concatenated onto `ball_feat` once,
  fed to the ball's two independently-weighted top-level consumers
  (`ball_mlp`, `ball_query_proj`; see §8.2). A player is not a single
  entity — there are up to 22 per observation (self + `MAX_OTHER_PLAYERS`),
  and `self_features`/`other_features` are consumed by exactly **one**
  shared-weight module, `EntityEncoder.per_entity_mlp`
  (`entity_encoder.py:57-62`) — the *same* weights applied to the self slot
  and every other-player slot, which is what makes the network permutation-
  invariant over which physical player occupies which slot. There is no
  second, separately-weighted "player_mlp" the way `ball_query_proj` sits
  alongside `ball_mlp`. So there is exactly one place to widen, and it is
  one level deeper than the ball case — inside `EntityEncoder.__init__`
  (the `nn.Linear(entity_feature_dim, embed_dim)` at line 58) and wherever
  `per_entity_mlp` is called in `EntityEncoder.forward()` — not in
  `DecisionNetwork.forward()`'s top-level code at all.

  **Batching — no special code needed.** `other_features` is already shape
  `(batch, MAX_OTHER_PLAYERS=21, entity_feature_dim)`, and
  `per_entity_mlp(other_features)` is already a single call (line 137,
  no explicit loop over the 21 slots) — `nn.Linear`/`nn.Sequential`
  broadcast over any number of leading dimensions, operating only on the
  last one. The frozen player-physics-encoder follows the exact same
  pattern for free: call it once on the sliced-out physics-input columns of
  `self_features` (`(batch, player_phys_input_dim) → (batch,
  player_latent_dim)`) and once on the same columns of `other_features`
  (`(batch, 21, player_phys_input_dim) → (batch, 21, player_latent_dim)`),
  concatenate each onto its own slot's raw features, and pass the widened
  `self_features`/`other_features` into `per_entity_mlp` as normal. Padded
  slots (`exists=0`) get whatever the encoder produces on all-zero input —
  harmless, `exists_mask` already zeroes their contribution downstream
  regardless of what's sitting in a padded slot's embedding.

  **Additional open question this raises, not present in the ball case**:
  `desired_direction`/`desired_speed_mode` are needed as inputs for *every*
  player slot, not just self — but nothing in `PlayerFeatures` currently
  exposes another player's desired direction/speed mode (only their actual
  `velocity`, which is the *current*, not *target*, quantity). The engine
  does track this on every `Player` object regardless of who controls them,
  and the rest of the observation space already gives full state access to
  every player (no partial observability in this game), so exposing it is
  probably fine — but it's a genuinely new field to source for other-player
  slots, unlike the ball case where every needed input was already sitting
  in `BallFeatures`/`GlobalFeatures`. Resolve this alongside the masking
  question below before implementation starts.
- **Open question flagged but not resolved**: unlike the ball's
  `is_loose` gate, a player always has *some* future trajectory — there's
  no natural "don't bother" mask. Whether to always inject it, or gate on
  something else (e.g. only for players not currently under direct
  human/rules order-override via `HybridPlayerAI`), needs a decision before
  implementation starts.
- There is also a live-network blind spot adjacent to this (not part of
  this plan): `PlayerFeatures` itself has no `heading_rad` field at all
  (dropped per its docstring's "recoverable when moving, irrelevant at
  rest" reasoning) — which the analysis above shows is only half true. This
  is a separate, smaller potential fix (bump `PLAYER_FEATURE_DIM`,
  invalidate checkpoints/BC data again) tracked here as a pointer, not
  scoped into either this plan or the player-dynamics follow-up.
