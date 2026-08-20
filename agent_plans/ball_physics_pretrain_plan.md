# Ball Physics Pretraining Plan — Frozen Dynamics Latent for BallFeatures

> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

## 0. Status

**Standalone pipeline implemented and in active use** (episode generation,
dataset, network, training script, auto-generated HTML report — §4-7, §9,
§10). **§8 (wiring the frozen encoder into `DecisionNetwork`/
`ExecutionNetwork`) is explicitly ON HOLD** pending the user validating the
standalone pipeline (training data quality, loss curves, classification
metrics) — do not start §8 without an explicit go-ahead; it's a real
architecture change that invalidates existing checkpoints (see §8.4).

See §12 for a full account of what was actually built, including a set of
quality-of-life additions beyond the original design here (random seeds,
append-mode dataset generation, a live progress bar, fine-grained per-head
loss/classification-metric reporting, and an auto-opening HTML training
report) — §11's player-dynamics follow-up should get the same treatment,
not just the network/pipeline itself (flagged again at the top of §11).

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
| 14 | `speed_norm` | `\|velocity_norm\|` | `sqrt(velocity_x^2+velocity_y^2+velocity_z^2)` (in the already-normalized units of fields 3-5) |
| 15 | `speed_norm_sq` | `speed_norm^2` | engineered feature, see below |
| 16 | `spin_norm` | `\|spin_norm\|` | `sqrt(spin_x^2+spin_y^2+spin_z^2)` (in the already-normalized units of fields 6-8); no squared counterpart, see below |
| 17-19 | `magnus_cross_{x,y,z}` | `spin_norm × velocity_norm` | cross product of fields 6-8 and 3-5 (already-normalized units); see below |

Box dims (`box_length`/`box_width`) are deliberately excluded — irrelevant
to ball flight or the out/goal boundary checks, GK-box-specific only.

**Engineered features (fields 14-19, added post-implementation, see
§12.6):** `N_INPUT_FIELDS` is 20, not 14 — added after the initial build,
in response to real training results. Air drag is exactly proportional to
speed² in the real physics model (`ball_physics.py`: `drag_force =
-0.5*rho*Cd*A*|v|*v`), a nonlinear combination of the 3 raw velocity
components (fields 3-5) that a small encoder would otherwise have to learn
indirectly — fields 14-15 hand it over directly. Spin magnitude (field 16)
has NO squared counterpart deliberately: checked the physics model and
nothing there is proportional to spin² (spin decay is a plain exponential,
linear in spin; the Magnus force below is also linear in spin), unlike
drag's exact spin²-analogue in speed. The Magnus force is exactly `F ∝ spin
× velocity` (`ball_physics.py`'s `magnus_force`) — a bilinear combination
across 6 raw inputs, harder to learn than a simple square, so fields 17-19
hand over the full 3-component cross product (not just its magnitude) so
the network keeps the *direction* information that determines which way
Magnus curves the ball. All of these are deterministic functions of
already-present inputs (no new information) and free to compute — this is
meant to be an ongoing category (hand over any known nonlinear physics
combination that's cheap to compute), see §12.6.

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

> **As-built note**: the actual `ai_config.json["physics_pretrain"]["ball"]`
> section, CLI flags, and logged metrics have since diverged from what's
> written below (more config keys, richer per-epoch logging, an
> auto-generated HTML report) — see §12 for the current, accurate picture.
> This section is kept as the original design record.

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

> **Implemented — see `agent_plans/player_physics_pretrain_plan.md`.** The
> standalone player-dynamics pretraining pipeline (episode generator,
> dataset, network, training script, HTML report) described there was built
> as a parallel pipeline to this one, per the sketch below. Live-network
> integration (this plan's §8, and the analogous section for the player net)
> remains unimplemented/out of scope for that pass too. The sketch below is
> kept as the original design record — the "as-built" doc explains where
> the final implementation deviated and why.

Sketch only, for whoever picks this up next, once the ball version is
validated end-to-end (trained, wired in, and shows some measurable effect —
e.g. faster BC convergence on kick/out-of-play-adjacent labels, or better
value-loss calibration near box/boundary states — before investing further).

> **Bring the QOL layer over too, not just the network/pipeline shape.**
> §12 documents a set of workflow improvements that turned out to matter a
> lot in practice while iterating on the ball version (random seeds by
> default, append-mode data generation, a live progress bar, fine-grained
> per-head loss AND classification-metric reporting with per-horizon
> sums/means, and an auto-generated, auto-opening HTML training report).
> None of that is ball-specific — build the player version's dataset/train
> scripts with the same shape from the start (e.g. reuse/extend
> `ai/progress.py`'s `ProgressReporter`, the `report.py`/
> `report_template.html` pattern, the `pos_weight`-logging +
> `pos_weight_max`-capping convention) rather than retrofitting it later.

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

---

## 12. As-built: the standalone pipeline, and the QOL layer around it

Written up after actually building and iterating on §4-7/§9/§10 (the
episode generator, dataset, network, and training script), so future work
(especially §11's player-dynamics follow-up) starts from what's real rather
than the original design sketch above. **§8 (live-network integration)
remains untouched/on hold** — everything below is scoped to the standalone
pretraining pipeline only.

### 12.1 File layout (matches §9, confirmed as-built)

```
src/footballcoach/ai/physics_pretrain/
  __init__.py
  ball_episode_gen.py       # §4 — BallEpisodeGenParams, generate_episode(), generate_shard()
  ball_dataset.py            # §4.4 — generate_dataset(), BallDynamicsDataset, __main__ CLI
  ball_dynamics_net.py       # §6 — BallDynamicsEncoder / Decoder / Autoencoder
  train_ball_dynamics.py     # §7 — compute_loss(), compute_confusion_counts(), train(), __main__ CLI
  report.py                  # NEW, not in original plan — write_report()/open_in_browser()
  report_template.html       # NEW — self-contained HTML report template (see 12.4)

checkpoints/physics_pretrain/
  ball_encoder.pt             # {encoder_state_dict, config_snapshot, normalization, dataset_stats, physics_config_hash}
  ball_encoder.history.npz    # full per-epoch history, every metric below
  ball_encoder.report.html    # auto-generated, auto-opened HTML report (gitignored, like the checkpoint)

physics_pretrain_data/ball/   # generated .npz shards (gitignored)
tests/ai_unit/test_ball_physics_pretrain.py
```

`ai_config.json["physics_pretrain"]["ball"]` gained keys beyond §7's
original sketch: `pos_weight_max` (null = uncapped, same convention as
`bc.pos_weight_max`) and `spin_active_frac` (see 12.2). Current defaults
drifted from §7's sketch too (e.g. `hidden_dim: 128`, `latent_dim: 24`) —
read the live config, not the numbers in §7.

### 12.2 Episode generation QOL

- **Random seed by default**: `generate_dataset(seed=None, ...)` (and the
  CLI's `--seed`, also defaulting to `None`) draws a fresh random base seed
  via `secrets.randbelow()` and LOGS it (`"...pass --seed <N> to reproduce
  this run"`) — omitting `--seed` used to silently mean "always seed 0",
  which made every fresh-directory generation call produce byte-identical
  episodes. Now it's genuinely random unless you pin it.
- **Appends, never overwrites**: repeated `generate_dataset()` calls
  against the same `output_dir` continue shard numbering (and seeding) from
  whatever's already there (`shard_00000.npz`, `shard_00001.npz`, ...
  scanned via glob at call time) instead of starting back at
  `shard_00000.npz` and clobbering it. To start over, delete the directory
  yourself — there's no `--overwrite` flag, by design (an explicit `rm` is
  safer than a flag that's easy to pass by habit).
- **Live progress, shards written as they finish**: uses the project's
  existing dependency-free [`ai/progress.py`](../src/footballcoach/ai/progress.py)
  `ProgressReporter` (no new dependency — already used by rollout
  collection elsewhere) instead of blocking silently until every shard is
  done. Switched from `pool.map` to `pool.imap_unordered` so each shard
  gets written to disk (and progress updated) the moment IT finishes,
  rather than all arriving at once at the end. Auto-downgrades from a live
  `\r`-updating bar to milestone lines when stdout isn't a real terminal
  (piped/logged output), so it's safe in both interactive and non-
  interactive use.
- **`spin_active_frac`** (config key, default in `ai_config.json`): most
  episodes now get ZERO ball spin, matching real play (mostly low/no-spin),
  with only a configurable fraction getting a random-axis/random-magnitude
  spin — previously every episode got some nonzero spin, over-representing
  it relative to what the live game actually produces.

### 12.3 Training-loop reporting QOL

`compute_loss()` was restructured from the original §5 sketch (one blended
"continuous MSE" + one blended "event BCE" per horizon) into much
finer-grained, separately-reported components — the blended numbers were
actively hiding information (e.g. `goal_scored`'s BCE, averaged together
with the much-easier `out_of_bounds` BCE, looked far better than it
actually was):

- **RMSE, not MSE**, for the three continuous groups — `pos_rmse`,
  `vel_rmse`, `spin_rmse`, each reported separately (position/velocity/spin
  are different physical quantities on different scales, so even splitting
  them from one one another wasn't enough — RMSE is also directly
  interpretable/convertible to real units, unlike squared-unit MSE). The
  *optimized* loss is still plain summed MSE — only the *reported* numbers
  are square-rooted, so this is purely a readability change, not a training
  change.
- **`out_of_bounds` BCE and `goal_scored` BCE reported separately**, never
  merged into one averaged "event BCE" number.
- **`pos_weight` is printed at the start of training**, per horizon, per
  flag (e.g. `t=0.2s  out_of_bounds=13.51  goal_scored=35.32`) — makes the
  class-imbalance severity visible up front instead of an invisible input
  to the loss. Cappable via the new `pos_weight_max` config key (or
  `--pos-weight-max` CLI override) if a horizon's weight gets large enough
  to destabilise training.
- **Classification metrics, not just BCE**: `compute_confusion_counts()`
  computes per-horizon `(tp, fp, fn, tn)` at a `logit > 0` (i.e. `p > 0.5`)
  threshold for both event flags; counts are SUMMED across every batch in
  an epoch (never averaged per-batch) before computing accuracy/precision/
  recall — averaging small-batch ratios directly would be wrong here, since
  a batch can easily have zero positive predictions (or zero actual
  positives) for the rarer `goal_scored` flag, making that batch's
  precision/recall undefined. `_classification_metrics()` reports `NaN`
  (not a misleading `0`) wherever a denominator is genuinely zero for the
  whole epoch.
- **Every per-horizon log line ends with an aggregate**: `sum:` for the
  five loss components (additive — matches what's actually being summed
  into the total loss), `mean:` (via `_safe_nanmean`, which skips the
  "Mean of empty slice" warning for an all-`NaN` row) for the three
  classification ratios, since summing precision/recall across horizons
  isn't a meaningful quantity the way summing losses is.
- **Full per-epoch history**, not just the log tail: every metric above,
  every epoch, saved to `<checkpoint>.history.npz` — `train_pos_rmse`,
  `val_goal_recall`, etc., shape `(n_epochs, n_horizons)` (or `(n_epochs,)`
  for scalars like `train_loss`).

### 12.4 Auto-generated HTML training report

`report.py` (`write_report()` + `open_in_browser()`) renders
`report_template.html` (a real `<!doctype html>...</html>` document, not an
Artifact-tool body fragment) with the full history + config + normalization
data embedded as inline JSON, and writes it to
`<checkpoint>.report.html`. `train()` does this automatically at the end of
every run and opens it in the OS default browser unless
`--no-open-report`/`open_browser=False` is passed (`open_in_browser()`
never raises — a headless environment with no display is a common, benign
case, not a training failure, so it logs a warning and returns `False`
instead).

This is a **local file**, deliberately distinct from publishing to a
shareable claude.ai artifact — that step requires a person-driven tool call
inside a Claude Code conversation, which a training script has no way to
trigger itself. The local report is the fully-automatic equivalent: same
page, same charts, opened straight from disk with zero manual steps.

Report contents (all computed client-side from the embedded JSON, no
server):

- Header stat strip: episode/row counts, epochs run vs. configured, best
  epoch, best val loss, latent dim.
- One chart panel per metric (position/velocity/spin RMSE, out-of-bounds/
  goal-scored BCE, and — new — accuracy/precision/recall for both event
  flags): per-horizon lines, color-coded on a cool→warm gradient by horizon
  length (short=cool, long=warm), train dashed, val solid, a vertical
  marker at the restored best-val epoch, and — for the three RMSE panels —
  a secondary right-hand axis in real units (metres/m/s/rad/s), using
  `pitch_half_diag_m`/`ball_spin_norm_max_rad_s` scale factors saved into
  the checkpoint at training time (position's conversion is explicitly
  labeled "approx" — it blends three differently-scaled fields, so there's
  no single exact metres conversion, only a reasonable stand-in). The BCE
  panels get a `ln(2)` "coin-flip baseline" reference line instead.
  Accuracy/precision/recall panels are fixed to a `[0, 1]` y-range rather
  than auto-scaled.
- Best-epoch table: every metric above at the best (restored) epoch, per
  horizon — RMSE rows get both a `normalized` row and a real-unit row (two
  full-width `<td>`s per row rather than a `rowspan` cell, specifically
  because an earlier `rowspan` version silently broke `:nth-child`-based
  column styling — the unit row had one fewer cell than the normalized row,
  which shifted every value left by one column and made the shortest
  horizon's value render styled like a label. Worth remembering if this
  table is extended again: keep every row in a given metric group the same
  cell count).

Colors/typography/layout follow the `artifact-design` skill's "utilitarian
dashboard" treatment (dark/light theme-aware CSS tokens, monospace for
data, a cool→warm horizon palette chosen because it encodes something real
— short vs. long prediction horizon — rather than being decorative).

### 12.5 What to replicate for the player-dynamics follow-up (§11)

Everything in 12.2-12.4 is dataset/training-loop/reporting infrastructure,
not ball-specific — when §11 gets built, carry it over rather than
re-inventing a thinner version:

- `ai/progress.py`'s `ProgressReporter` for episode-generation progress.
- Append-by-default, randomly-seeded-by-default dataset generation.
- Per-quantity-group (not blended) loss reporting, with per-horizon
  sums logged — for the player net that likely means separating position/
  velocity/heading/stamina RMSE rather than one blended continuous MSE
  (mirroring the ball net's pos/vel/spin split), plus the same
  confusion-count-based accuracy/precision/recall treatment for
  `out_of_bounds`/`goal_scored` (§11's player output already includes
  both).
- `pos_weight` printed at training start, cappable via a
  `physics_pretrain.player.pos_weight_max` config key.
- A `report.py`/`report_template.html`-style auto-opening local HTML
  report — likely close enough to reuse most of the template directly
  (swap the metric key names, add a fourth continuous-RMSE panel for
  heading), rather than writing a second bespoke report generator.
  Specifically worth carrying over from the ball version's later additions
  (§12.6): the `pos_dist`/`vel_dist`-style mean-distance metrics leading
  the stat row (more literally answers "how far off, typically" than RMSE
  does — see §12.6's Jensen's-inequality gotcha before assuming RMSE ≥
  distance for whatever axis-averaging convention gets used), and the
  sample-predictions table at the bottom (random input/pred/target rows —
  cheap, and catches things aggregate metrics can hide).

**Lessons learned the hard way in §12.6, worth applying from the START
this time rather than discovering again after a bad training run:**

- **Interleave every same-epoch auxiliary pass from day one, don't bolt it
  on later.** The ball net ran main/adjacent-pair/t0 passes as 3 sequential
  phases for a long time before the ordering bias (whichever pass runs
  LAST is always measured against the most-updated model that epoch) was
  even suspected, let alone fixed (`_run_interleaved_train_epoch`). If the
  player net ends up with more than one same-epoch training pass sharing
  an optimizer (e.g. an equivalent adjacent-pair or t0 term), build the
  combined/interleaved loop structure in from the start rather than
  incurring the same silent per-horizon bias for however long it takes to
  notice.
- **Audit EVERY multi-epoch phase for its own best-val tracking and
  (optional) early stopping, not just the main loop.** Both autoencode
  pretraining and decoder-only pretraining initially had neither, and both
  needed it added after real, observed problems (a bad final epoch handed
  to the next phase; a fixed epoch budget with no way to stop once
  converged). If the player net gets an analogous pretraining phase, give
  it its own best-val tracking and its own (separately-configurable, phase-
  scoped) early-stopping knobs immediately, rather than treating them as
  the main loop's exclusive privilege.
- **Look for provably-constrained output fields and encode the constraint
  architecturally, not just hope the network learns it.** Height (ball_z)
  turned out to be provably non-negative, worth a single-ReLU special case
  in the identity-shortcut decoder init instead of the general bidirectional
  treatment. Player state likely has several analogous constraints worth
  auditing for up front: stamina/speed magnitude (non-negative), heading
  (circular — a raw angle is the WRONG representation entirely; encode as
  `(sin, cos)` pairs rather than trying to make a linear layer respect
  wraparound), field position (bounded by pitch dimensions, though that's
  a soft/probabilistic bound unlike height's hard physical floor). Cheap to
  identify early, expensive to retrofit once an identity-shortcut init
  already assumes the general bidirectional case everywhere.
- **When unifying normalization scales across axes of one quantity
  (position x/y/height sharing one divisor instead of 3 different ones),
  check what that does to EVERY axis's relative weight in the loss, not
  just whether it "sounds more consistent."** Naively unifying position's
  3 divisors to whichever is LARGEST (matching velocity's convention)
  would have made height's error term ~20x smaller in normalized space
  than it should be — nearly invisible in `pos_mse`/`pos_rmse` — if height
  had kept the smaller, more natural `height_norm_m` scale while x/y moved
  to the much larger pitch-derived one. For the player net, if any
  analogous per-axis-normalized quantity gets touched (position, velocity),
  explicitly check the RATIO between the old and new scale for every axis
  being unified before assuming "one shared divisor" is strictly better —
  it can silently zero out an axis's contribution to the loss instead.
- **Audit initial-condition SAMPLING code for physical-plausibility
  violations separately from the ONGOING simulation code, even when the
  simulation itself is already correct.** `step_ball`'s ground clamp
  (`z >= ball_radius_m`) was correct the whole time; the bug was in the
  INITIAL position sampler, which had never been checked against that same
  constraint and let ~3.7% of episodes start with the ball's center
  literally below/inside the ground. The two code paths (seed a state,
  then simulate it forward) can each independently respect or violate a
  physical constraint — checking one doesn't imply the other is fine. For
  the player net, whatever seeds player starting positions/velocities/
  stamina should get the same explicit check against whatever the ongoing
  simulation already (presumably correctly) enforces.
- **Gradient-masking hooks are attached at model CONSTRUCTION time and
  persist independent of whatever weights get loaded via `load_state_dict`
  afterward.** This means a checkpoint trained under an OLDER version of an
  init/masking scheme, resumed under NEWER code, silently gets the NEW
  code's masking behaviour applied to the OLD code's (possibly very
  different, possibly load-bearing) weight values — see §12.6's
  checkpoint-resume caveat for the concrete case this bit the ball net. If
  the player net's identity-shortcut/masking scheme ever changes after
  checkpoints already exist, check this interaction explicitly rather than
  assuming "the weights are what matter, hooks don't need re-checking."

### 12.6 Post-hoc architecture/training tweaks and engineered features

Changes made after the initial as-built pipeline (12.1-12.5) landed, in
response to real training results (oscillating loss late in training,
flat-looking velocity/spin RMSE across horizons):

- **Decoder**: switched from 6 independent per-horizon heads to a single
  shared decoder conditioned on 3 horizon features concatenated onto the
  latent (`decoder_hidden_dim` in config): `t_norm` (`horizon_s /
  max(horizons_s)`), `t_norm^2`, and `log(horizon_s)`. Log-only was the
  original design; switched to all 3 because most of the underlying
  kinematics is close to linear/quadratic in RAW time for short intervals
  (position ~ velocity*t, gravity's t^2 term, spin decay ~ linear-per-tick)
  — log(t) alone forces the network to implicitly apply exp() to recover
  that, which a linear layer could represent exactly from raw/squared t.
  Feeding all 3 (cheap, 2 extra scalars) lets the network use whichever
  combination fits a given output best. Only the decoder sees any of this
  — the encoder never does, so the latent stays a general dynamics-state
  representation rather than horizon-specific.
- **Encoder**: added a gradual-bottleneck layer
  (`hidden_dim -> encoder_bottleneck_dim -> latent_dim` instead of jumping
  straight from `hidden_dim` to `latent_dim` in one step). BatchNorm was
  considered and rejected — it needs `.eval()` reliably called at every
  inference site to use running stats instead of batch stats, and this
  encoder is meant to eventually run frozen at batch-of-1 inside the live
  rollout hot loop (§8, still on hold) — the same class of bug already seen
  once in this codebase (`network.value_dropout`'s "PPO rollout doesn't
  call `.eval()`" caveat). LayerNorm (no batch/train-eval dependency) is
  the safer alternative if normalization is revisited later — not yet
  added, no immediate need identified.
- **LR schedule**: added cosine annealing with warm restarts (SGDR,
  `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`), config-gated via
  `lr_cosine_restart_epochs` (0/null = flat LR, prior behaviour). Motivated
  by train loss oscillating-but-occasionally-improving late in training — a
  sign the LR was too coarse for the local curvature near the minimum.
  Current LR is logged per epoch and saved to `.history.npz`, but
  deliberately NOT added as a report panel — the chart machinery assumes
  per-horizon train/val series pairs, and LR is a single scalar with no val
  counterpart, so it'd need its own plotting path.
- **R² sanity metric**: added per-horizon, per-quantity-group (pos/vel/
  spin) R² (`1 - MSE/Var(target)`, baseline variance always from
  `train_idx`) alongside RMSE, specifically to distinguish "genuinely hard
  to predict that far out" from "collapsed to predicting the mean" when
  RMSE looks flat across horizons — RMSE magnitude alone can't tell those
  apart. Same sum-then-divide-once accumulation pattern as the confusion
  counts (never average per-batch ratios).
- **Displacement-normalized error** (`{pos,vel,spin}_err_pct_disp`):
  reuses the same per-horizon accumulated squared-error sums as R², but
  divides against a "persistence" baseline (predict the episode's OWN
  initial state, i.e. assume nothing moved -- `BallDynamicsDataset.
  compute_persistence_baseline_mse()`) instead of the train-set mean, and
  reports it as `100 * sqrt(MSE_model/MSE_baseline)` -- the model's typical
  error as a percentage of how far the ball typically moved over that
  horizon. Motivated by two issues with a plain mean-baseline R²: (1) the
  mean baseline is weak at short horizons (the ball hasn't moved much, so
  even "echo the input" scores deceptively high against it), whereas
  persistence is a much tougher bar there; (2) R² compresses via squaring,
  so a 0.9 there is only really a ~68% RMSE reduction, not 90% -- reporting
  the un-squared ratio directly (as a percentage) reads more honestly. An
  earlier version of this reported `1 - MSE/MSE_persistence` (R² form, same
  underlying computation) before being replaced with this percentage form
  for readability. A literal per-sample MAPE-style version (percent error
  per episode, then averaged) was considered and rejected -- individual
  episodes can have a near-zero true delta in a given dimension, and
  dividing by that per sample blows up/dominates the average; summing
  numerator and denominator separately across the whole population first
  (as done here) avoids that instability entirely.
- **Ballistic-baseline error** (`{pos,vel,spin}_err_pct_ballistic`, `Ball
  DynamicsDataset.compute_ballistic_baseline_mse()`): a third, stronger
  baseline than persistence for pos/vel specifically -- straight-line
  physics using ONLY constant velocity + gravity (no drag, no Magnus/spin
  effects, no bounce), computed in real units per episode (using that
  episode's own randomized pitch scale) then renormalized to compare
  against the model's actual (normalized) MSE, same
  `100 * sqrt(MSE_model/MSE_baseline)` reporting form as the persistence
  metric. Motivated by: position and velocity are handed to the encoder
  directly as raw inputs, and `pos0 + v0*t` (+ gravity's t^2 term) is close
  to exact for short horizons, so it's a much more telling bar than "assume
  nothing moved" for catching cases where the model isn't even matching
  free, nearly-linear extrapolation of information it already has. Spin has
  no separate ballistic model (nothing in this baseline acts on spin), so
  `spin_err_pct_ballistic` is always identical to `spin_err_pct_disp`.
  - **Post-launch fix: z is floored at 0 (can't go underground); x/y stay
    fully unclamped.** An x/y pitch-edge freeze (matching real episodes'
    freeze-on-out-of-bounds semantics) was tried first and then explicitly
    reverted at the user's request -- the intent is a "how good is naive
    straight-line kinematics" baseline, and x/y should keep extrapolating
    for the FULL horizon regardless of pitch boundaries; only z gets a
    physical-plausibility floor, since a ball can't be below the ground.
    Implementation: `z_pred = max(0, pos0.z + vz*t - 0.5*g*t^2)`, computed
    independently at each horizon from the full (unclamped) `t` -- NOT a
    "freeze forever after first ground contact" state carried between
    horizons, just "can't be below the ground at this instant" applied to
    that horizon's own prediction (a trajectory arcing back above `z=0`
    later, e.g. thrown up again, is still allowed to be positive at a
    later horizon even if an earlier horizon clamped it to 0). Velocity is
    NOT clamped at all, only position's z. Verified in
    `test_compute_ballistic_baseline_mse_hand_computed` (x_norm allowed
    past 1.0, proving x/y really is unclamped) and
    `test_compute_ballistic_baseline_mse_clamps_z_floor_but_not_xy` (z
    floored at exactly 0 while x keeps extrapolating past the pitch edge
    in the SAME row/horizon).
- **Freeze-on-event semantics moved from generation-time to training-time
  (2026-08-19).** `generate_episode` no longer stops/freezes physics on
  out_of_bounds OR goal_scored -- the ball is simulated with the SAME
  regular physics (gravity/drag/Magnus/ground bounce, and goal-net/post/
  crossbar bounce physics via `resolve_goal_boundary`) for the FULL
  horizon regardless of event status, so it can genuinely bounce back onto
  the pitch or back out of a goal mouth. Motivated by the hypothesis
  (discussed at length re: why oob classification lags goal classification
  and is hardest at short/mid horizons) that the hard discontinuity the
  freeze introduces into the pos/vel/spin regression targets is making the
  network's job harder than necessary, not easier. `out_of_bounds`/
  `goal_scored` at each recorded horizon are now the INSTANTANEOUS state
  at that exact moment (no longer latched/sticky) -- a ball out at t=1s
  but back in bounds at t=2s now has `out_of_bounds=0` at the t=2s horizon.
  - `generate_episode`/`generate_shard` now additionally return
    `crossing`/`crossing_time` -- an 11-field row (same layout as one
    per-horizon target block) capturing the FULL state at the moment
    out_of_bounds/goal_scored FIRST became true, and when. Persisted per
    episode in each `.npz` shard (`crossings`/`crossing_times` arrays)
    alongside the always-continuous `targets`.
  - `physics_pretrain.ball.freeze_semantics` (default `true`, prior
    behaviour) is a **training-time**, not generation-time, flag:
    `BallDynamicsDataset.targets_with_freeze_semantics()` reconstructs the
    OLD freeze-and-latch targets from the stored continuous trajectory +
    `crossings`/`crossing_times` (substitutes `crossings` into any horizon
    at or after that episode's `crossing_time`) right after `train()`
    loads the dataset. `false` trains against the raw stored data as-is.
    Both conventions live in the SAME generated dataset, so comparing them
    is just flipping this flag and re-running -- no regeneration needed.
  - Verified with `test_generate_episode_simulates_to_exact_horizon_time_
    not_nearest_tick` (unaffected by this change) plus new coverage for
    the always-continuous/instantaneous-flags/crossing-capture behavior
    and `targets_with_freeze_semantics`'s reconstruction (see the test
    file's `ball_episode_gen`/`BallDynamicsDataset` sections).
- **BCE loss weight (2026-08-19).** `physics_pretrain.ball.
  bce_loss_weight` (default 1.0, no behaviour change) multiplies the
  out_of_bounds/goal_scored BCE terms' contribution to every backpropagated
  loss in a run (main per-horizon heads, adjacent-pair, autoencode-pretrain,
  the t0 term) -- threaded through `compute_loss`/`compute_per_episode_
  loss`/`_single_target_loss(_with_breakdown)`/`_single_target_per_episode_
  loss`. Reported `oob_bce`/`goal_bce` metrics are always the raw,
  UNweighted value regardless of this setting, so runs at different
  weights stay directly comparable. `0.0` fully disables the classification
  heads' gradient (regression trains exactly as if out_of_bounds/
  goal_scored didn't exist as targets), for testing whether the
  classification heads are competing with regression quality. Verified
  with `test_compute_loss_bce_weight_scales_total_but_not_reported_
  breakdown` (breakdown unaffected at any weight; `total` scales linearly;
  `0.0` gives exactly zero gradient at the oob/goal output columns while
  leaving pos/vel/spin gradient untouched).
- **Spin loss weight (2026-08-19).** `physics_pretrain.ball.
  spin_loss_weight` (default 1.0, no behaviour change) -- same idea as
  `bce_loss_weight` immediately above, but for the spin MSE term instead of
  the BCE terms, same call sites, same "reported `spin_rmse` is always the
  raw unweighted value" convention. `0.0` fully disables the spin head's
  gradient, for testing whether spin regression is competing with (dragging
  down) pos/vel quality -- motivated directly by the post-training t=0
  sanity check showing spin reconstruction error far worse (proportionally)
  than pos/vel's, raising the question of whether spin was worth its share
  of gradient budget.
  - **Diagnostic-noise suppression, applies to `bce_loss_weight`/
    `spin_loss_weight` alike.** Once a weight is 0.0, its component isn't
    receiving gradient, so printing its raw (frozen, unchanging) value
    every epoch is noise, not signal -- `spin_rmse` (when `spin_loss_weight
    ==0`) and `oob_bce`/`goal_bce`/the oob/goal accuracy/precision/recall/R²
    /err_pct rows (when `bce_loss_weight==0`) are now skipped from the
    per-epoch console printout specifically (`_LOG_COMPONENTS`/`_LOG_CLS_
    KEYS`/`_LOG_R2_KEYS`/`_LOG_PCTD_KEYS`/`_LOG_PCTB_KEYS`, filtered
    variants of the unfiltered `_COMPONENTS`/`_CLS_KEYS`/etc. used
    everywhere else) -- `history`/`.history.npz`/the HTML report still
    record every column regardless, so no data is lost, only the console
    spam is trimmed.
- **Mean-distance metrics, `pos_dist`/`vel_dist` (2026-08-19).** Added
  alongside `pos_rmse`/`vel_rmse` (`LossBreakdown`, `compute_loss`,
  `_single_target_loss_with_breakdown`, `_COMPONENTS`/`_RMSE_UNIT_SCALE`) --
  purely a REPORTING metric, never contributes to the backpropagated loss.
  `pos_dist` is the mean PER-SAMPLE Euclidean distance
  `||pred_pos - target_pos||` (mean taken AFTER the sqrt), motivated by
  wanting a more literal answer to "how far off is the model typically"
  than RMSE gives. **Non-obvious gotcha, worth remembering for any future
  per-axis-averaged-then-square-rooted metric**: `pos_rmse` here is
  `sqrt(F.mse_loss(...))`, which averages squared error over the 3 position
  axes AND the batch TOGETHER (dividing by 3), making it closer to a
  "typical per-AXIS error" than the RMS of the 3D vector distance -- the
  TRUE RMS-of-distance is `sqrt(3) * pos_rmse`, and Jensen's inequality
  bounds `pos_dist` against THAT (`pos_dist <= sqrt(3) * pos_rmse`), not
  against `pos_rmse` directly. First-draft documentation (and verbal
  explanation to the user) claimed `pos_rmse >= pos_dist` always, which is
  WRONG for this specific `pos_rmse` definition -- `pos_dist` can and does
  come out numerically LARGER than `pos_rmse`, verified empirically (real
  run: `pos_rmse=1.47m`, `pos_dist=1.99m`). Corrected in `LossBreakdown`'s
  docstring. `vel_dist` mirrors `pos_dist` for velocity, same relationship
  to `vel_rmse`. Surfaced prominently in the HTML report
  (`report_template.html`): both lead the top stat row (accent-colored,
  ahead of episode/epoch counts), get their own chart panels placed FIRST
  (before the RMSE panels), and their own rows in the best-epoch table.
- **Sample-predictions table in the HTML report (2026-08-19).** 10 random
  `(episode, horizon)` pairs from the val set, saved as `val_examples` in
  the training artifact (`train_ball_dynamics.py`'s `train()`, reusing
  `describe_input_row`/`describe_target_row` for human-readable strings,
  reproducible per `--seed` since it draws from the run's own seeded
  `rng`), rendered as a 3-row-per-example (input/pred/target) table at the
  bottom of the report. Motivated by wanting to eyeball a handful of
  TYPICAL predictions directly, as a check against aggregate metrics
  alone -- complements (doesn't replace) the existing median/worst-episode
  diagnostic, which is deliberately NOT representative (picked for being
  extreme).
- **Combined interleaved training loop, main loop AND decoder-only
  pretraining (2026-08-19).** Previously each epoch ran 3 SEQUENTIAL
  passes -- the main per-horizon pass fully start-to-finish, then the
  adjacent-pair pass fully, then the t0 pass fully (each its own loop, own
  `optimizer.step()` calls). This meant whichever pass ran LAST each epoch
  was always measured against (and trained against) a model every OTHER
  pass had already updated that epoch, while whichever ran FIRST never
  benefited from the others' updates within that epoch -- the exact same
  systematic (not noise-averaging-out) bias `_interleaved_horizon_batches`
  already existed to fix ACROSS HORIZONS within one pass, just not yet
  applied ACROSS PASS-TYPE. Fixed by a new shared closure,
  `_run_interleaved_train_epoch(optimizer)`, that builds ONE combined,
  SHUFFLED sequence of minibatches drawn from all 3 sources (tagged by
  source so each batch still runs its own correct forward/loss/backward/
  step), and iterates that single sequence instead of 3 separate loops.
  Metrics stay fully separate per source in the log/history (`train_loss`
  vs `train_pair_loss` vs `train_t0_loss`) -- only EXECUTION ORDER changed,
  not what gets measured or reported. Parameterized by `optimizer` (not a
  fixed module-level one) specifically so the SAME function serves both the
  main loop and decoder-only pretraining, which use different optimizer
  instances over different trainable parameter subsets -- replaced both
  phases' previously-duplicated 3-block structure with one call each.
- **Position normalization unified to one shared divisor
  (2026-08-19).** Under `normalize_kinematics_by_base_pitch=true` (see the
  original entry above), position's 3 axes (x/y/height) now ALL divide by
  the base pitch's `half_diag` -- the same divisor velocity already used --
  instead of x/half_length, y/half_width, height/height_norm_m (3 DIFFERENT
  divisors, still the behaviour when the flag is `false`). Motivated by:
  `pos_mse` (`F.mse_loss`) sums squared error across all 3 axes before
  reporting, so mixing 3 different real-world-to-normalized scales into one
  combined number muddies what that number means, and specifically makes it
  hard to reason about which axis dominates the loss. Considered unifying
  ONLY x/y (leaving height on its own, much smaller, `height_norm_m` scale)
  but rejected -- height would then be ~20x smaller in normalized space
  than x/y, meaning height error would barely register in `pos_mse`/
  `pos_rmse` at all, defeating the point of tracking it. `_kinematics_
  divisors`/`_kinematics_denorm_scales` (encode/decode sides, `ball_
  episode_gen.py`/`train_ball_dynamics.py`) and `compute_ballistic_
  baseline_mse` all updated together to keep the encode/decode/baseline
  paths consistent. Dataset-generation-time change (like the flag itself)
  -- requires regenerating any dataset built before this landed, since old
  shards encode height under the OLD per-axis convention even where the
  flag itself is unchanged.
- **`goal_net_collisions_enabled` (2026-08-19).** `physics_pretrain.ball.
  goal_net_collisions_enabled` (default `true`, no behaviour change) --
  dataset-generation-time flag gating whether `resolve_goal_boundary()`
  (goal-net/post/crossbar bounce physics) runs at all once the ball has
  passed the goal line. `false` skips it entirely -- the ball just keeps
  flying through the goal mouth under regular gravity/drag/Magnus/ground-
  bounce physics, as if the frame weren't there. `goal_scored` detection
  (`check_goal`) is independent of this and unaffected either way. Added to
  isolate whether the goal-frame's bounce geometry (small, high-curvature
  region of state space, similar bounce-timing sensitivity to pitch-
  boundary bounces) is a meaningful contributor to the mid-horizon
  (t=1-3.5s) R² dip observed in real runs, independent of ordinary
  pitch-boundary bounces.
- **Initial-position ball-radius floor (2026-08-19).** Found while
  investigating why the trained model sometimes predicts a negative height:
  the dataset itself never contains a negative height ANYWHERE (0 negative
  values across a full scan of both inputs and targets), so that's a pure
  model-extrapolation artifact, not a label problem -- but a closer check
  (comparing against the ball's actual physical floor, `ball_radius_m`
  =0.11m, not 0, since `position.z` is the ball's CENTER and `step_ball`'s
  ground clamp holds a resting ball's center at exactly `ball_radius_m`)
  found that `_sample_position_in_play`/`_sample_position_already_special`
  were sampling initial `z` from `[0.0, ...)`, letting ~3.7% of episodes
  start with the ball's center below its own radius (physically, partially
  buried). Traced the actual physics consequence: `step_ball`'s ground
  clamp (`if new_position.z <= ball_radius_m: new_position.z = ball_
  radius_m`) triggers on the very FIRST simulated tick for these episodes,
  snapping the center straight up to `ball_radius_m` -- and if the sampled
  initial vertical velocity happened to be downward and fast enough, this
  reads as a genuine BOUNCE (restitution applied to vertical AND horizontal
  velocity, spin decayed) despite the ball never having actually fallen
  from height, an artificial one-frame discontinuity with no real-world
  analogue. Fixed by flooring all `z` sampling at `ball_radius_m` instead
  of `0.0` (`BallEpisodeGenParams` gained a `ball_radius_m` field, sourced
  from `physics.json`'s `ball.radius_m`, threaded into both sampling
  functions). Dataset-generation-time change -- requires regeneration.
- **Decoder identity-shortcut: single-unit special case for height
  (2026-08-19).** `_init_identity_shortcut_decoder`'s `dim` (=9) identity
  fields each get 2 dedicated hidden units (`ReLU(x) - ReLU(-x) == x`,
  needed because a single `ReLU` alone clips negative values) -- EXCEPT
  `Z_FIELD_INDEX` (height's position within those 9 fields, =2), which
  provably never needs to represent a negative value (see the initial-
  position fix above -- confirmed empirically, 0 negative values anywhere
  in the dataset). Height now gets only ONE dedicated unit (`ReLU(x) == x`
  is already exact for `x>=0`); its would-be second unit (`z_free_idx =
  2*Z_FIELD_INDEX+1`) is freed up as a genuine SPARE unit instead (same
  random-init/no-mask treatment as the other spare units, including write
  access to oob/goal logits) rather than wasting it on a `-1` weight
  connection nothing in the real data ever needs. Bonus effect: height's
  dedicated reconstruction path can now never itself contribute a negative
  value (its one live unit is a bare ReLU, clipped at 0) -- a soft
  architectural nudge against negative height predictions, though NOT a
  hard guarantee on the network's overall output (nothing stops OTHER
  units, e.g. this freed one, from contributing to that output column too,
  in principle, if the encoder ever routes signal that way).
  - **Reverted approach, kept for the record: making the freed unit
    permanently dead instead of a live spare.** Tried first, as a more
    conservative "definitely can't destabilize anything" option -- zero
    weight/bias, kept inside the permanently-masked dedicated block, same
    as if the field simply had 1 fewer unit. Reverted at the user's
    explicit request in favour of keeping it live: wasting a whole hidden
    unit for a single field is a bigger cost than fixing the thing that
    actually broke, which was the REGRESSION TEST's synthetic data (it fed
    a genuinely negative synthetic "height" target, something the real
    premise this whole change rests on -- height is never negative --
    explicitly rules out; the single ReLU unit can never resolve that
    target no matter how weights move, so real, unmaskable, ever-present
    gradient destabilized the newly-live shared-optimizer-state unit within
    a handful of adversarial steps). Fixed by correcting the test's
    synthetic data (`.abs()` on the synthetic height column) instead of
    weakening the architecture -- see `test_identity_shortcut_survives_
    adversarial_classification_gradient`'s updated docstring.
  - **Checkpoint-resume caveat.** Gradient-masking hooks (`register_hook`)
    attach to the `Parameter` object at model CONSTRUCTION time, which
    always happens BEFORE `load_state_dict()` when resuming from a
    checkpoint -- so hooks always reflect CURRENT code, regardless of what
    checkpoint is loaded on top, while the WEIGHT VALUES loaded in reflect
    whatever code trained that checkpoint. For a checkpoint trained under
    the OLD (pre-this-fix) scheme specifically, `z_free_idx` holds a real,
    meaningfully-trained negative-half reconstruction weight (not a benign
    near-zero spare unit) -- loading it under the NEW code's mask (which
    now treats that column as free to write to oob/goal) re-exposes an
    already-important weight to exactly the classification-gradient
    corruption this file's masking exists to prevent, just for this one
    unit. Not currently guarded against -- worth a warning or an explicit
    re-mask-on-resume fixup if old checkpoints need to keep training under
    the new code.
- **Exact-time horizon simulation (2026-08-19).** `generate_episode` used
  to snap each `horizons_s` value to the nearest whole physics tick
  (`round(h / sim_dt_s)`), then record whatever state existed at THAT
  tick as if it were the literal `horizons_s` value -- a systematic (not
  random) timing error bounded by `sim_dt_s/2`, worst in RELATIVE terms
  at the shortest horizons (e.g. with `sim_dt_s=0.06`, `horizons_s[0]=
  0.2` was actually recorded at t=0.18s, a 10% relative error, vs. ~0.2%
  by the longest horizon) while the decoder's horizon-conditioning
  features still used the literal nominal value -- a real, previously
  unaccounted-for source of label noise concentrated exactly where the
  horizon-ordering and oob-class-imbalance issues above already made
  short horizons hardest. Fixed by simulating chronologically to each
  horizon's EXACT time instead: regular `sim_dt_s`-sized steps until one
  more would overshoot, then a final partial step of exactly the
  remainder (`step_ball` takes `dt_s` as a plain parameter with no
  fixed-tick assumptions baked in, so an arbitrary partial step is exactly
  as valid as a full one). The out-of-bounds/goal check still runs after
  EVERY step regardless of size, so freeze detection is unchanged (if
  anything finer-grained, right at a horizon boundary). Verified with
  `test_generate_episode_simulates_to_exact_horizon_time_not_nearest_tick`,
  which spies on `step_ball`'s `dt_s` arguments and checks their
  cumulative sum lands exactly (not just close) on each horizon time.
- **Engineered input features** (fields 14-19, see §3's table for the full
  per-field breakdown): `speed_norm`/`speed_norm_sq` were added because air
  drag is exactly proportional to speed² in the real physics model
  (`ball_physics.py`'s drag force). `spin_norm` (magnitude only, no squared
  counterpart -- checked the physics model and nothing there is
  proportional to spin²) and the 3-component `magnus_cross_{x,y,z}` (the
  real `spin × velocity` cross product from `ball_physics.py`'s
  `magnus_force`, kept as a full vector rather than just its magnitude so
  the network keeps the direction information that determines which way
  Magnus curves the ball) were added at the same time. All are
  deterministic functions of already-present inputs (no new information)
  and cheap to compute. This is deliberately a *pattern*, not a one-off —
  the general principle is: when a known physics relationship depends on a
  nonlinear combination of values already in the input, and that
  combination is cheap/deterministic to compute, hand it over directly
  rather than making the network rediscover it. Considered and rejected:
  distance-to-boundary features (nearest touchline/goal-line) -- the
  network already gets normalized position directly, and edge distance is
  a simple LINEAR function of that (`1 - |pos_norm|`), not the kind of
  nonlinear combination this pattern is meant to address.
- **Horizon feature: `log1p(t)` instead of `log(t)`**: the decoder's third
  horizon feature (alongside `t_norm`/`t_norm^2`) is `log1p(horizon_s)`,
  not plain `log`. `log1p(t) ≈ t` for small t (derivative at t=0 is exactly
  1), so unlike an arbitrarily-shifted `log(t+eps)`, it doesn't fight the
  near-linear-kinematics reasoning above near t=0 -- it's close to a
  redundant near-linear term there, while still flattening/compressing at
  large t the way plain log(t) did (the actually useful part -- "7s vs
  10s" closer together than "0.2s vs 0.5s"). Well-defined and unremarkable
  at t=0 (`log1p(0)=0`), no epsilon to justify. Same pattern already used
  for `time_remaining` normalization elsewhere in this codebase
  (`ai/obs/encoder.py`).
- **`BallDynamicsDecoder.forward_at(latent, horizon_s)`**: a second
  forward path alongside `forward()`, taking an explicit horizon instead
  of reading the fixed per-instance trained-grid buffers -- lets the
  (already-trained) decoder be queried at ANY horizon, not just the ones
  it saw during training. No decoder architecture change was needed to
  support this (same 3-feature formula, same weights) -- what it unlocks
  is described below.
- **t=0 autoencoding sanity check** (post-training diagnostic, always
  logged when there's a val split): encode every val episode, decode at
  `t=0` via `forward_at()` (never a training target under normal
  per-horizon training), and compare directly against that episode's own
  input -- the correct answer is exact by construction (nothing has
  happened yet), which isolates "can the encoder+decoder round-trip
  through the latent bottleneck at all" from "can it also predict real
  dynamics". Motivated by: is a ~5-10m position RMSE at short horizons a
  genuine dynamics-prediction difficulty, or is the latent bottleneck
  itself (`latent_dim`) losing information it didn't need to lose? A bad
  t=0 result specifically implicates the bottleneck, since there's no
  actual prediction difficulty at t=0 to blame instead.
- **Autoencode pretraining** (`physics_pretrain.ball.
  autoencode_pretrain_epochs`, default 0/disabled): an optional phase run
  BEFORE the main training loop that actively trains (not just diagnoses)
  the t=0 round-trip, across every RECORDED horizon's state (not only the
  original t=0 samples) via `BallDynamicsDataset.build_autoencoding_data()`
  + `forward_at(latent, 0.0)`. Primes the bottleneck to preserve
  information before also asking it to learn dynamics. Flat LR, its own
  (`autoencode_lr`, separate Adam instance from the main loop's
  `optimizer`) -- autoencoding is a much easier task than multi-horizon
  dynamics prediction, so the right step size isn't necessarily the same;
  falls back to `lr` if unset. Always flat (no cosine schedule -- that's
  tuned for the main loop specifically). Also logs an "epoch 0 (before
  training)" baseline pass (forward-only, no gradient step) before the
  first real epoch, so init quality (e.g. `identity_shortcut`'s effect, see
  below) is directly visible rather than only inferrable from epoch 1's
  already-partially-trained numbers.
- **Adjacent-horizon-pair training**
  (`physics_pretrain.ball.adjacent_pair_training_enabled`, default true):
  additional (input, target) training examples derived from data already
  fully recorded, no new physics simulation needed
  (`BallDynamicsDataset.build_adjacent_pair_data()`) -- treats each
  recorded horizon's state as a NEW pseudo-initial-state, predicting the
  next recorded horizon from it via a fixed, statically-known delta
  (`horizons_s[i+1] - horizons_s[i]`, same for every row in that
  pair-type, so `forward_at()` -- one scalar delta per whole batch -- is
  exactly the right tool; NO decoder change was needed for this, unlike an
  initial (mistaken) assumption that a batched varying-delta path would be
  required). Run as an additional training pass within each epoch (same
  optimizer, same epoch, tracked/logged separately as `train_pair_loss`/
  `val_pair_loss`). Two distinct benefits, not equally strong: (1) more
  training examples "for free" (pure reshaping of already-recorded data,
  no new simulation); (2) more importantly, exposes the encoder to a
  realistic "mid-trajectory" input distribution -- the ORIGINAL training
  only ever encodes states drawn from uniform-random t=0 sampling, but a
  live-deployed encoder (§8, still on hold) will see whatever state the
  ball happens to be in mid-match, which looks far more like "partway
  through a trajectory" than "freshly uniform-random sampled". Rows where
  the episode was ALREADY resolved (out_of_bounds/goal_scored) by the
  pair's start horizon are excluded -- predicting "stays frozen" from an
  already-frozen state is trivially correct and would dilute the more
  informative signal with easy examples (same rationale as
  `bc.downsample_trivial` elsewhere in this codebase). Deliberately scoped
  to ADJACENT pairs only (not the full `n_horizons choose 2` combinatorial
  expansion of every start/end pair) to limit both the trivial-pair risk
  and the added per-epoch compute cost.
- **t=0 term during the main loop** (`physics_pretrain.ball.
  autoencode_during_main_loop_enabled`, default true): the same
  `build_autoencoding_data()` + `forward_at(latent, 0.0)` mechanism used by
  autoencode pretraining, but also run as an extra pass EVERY epoch of the
  MAIN loop, not just before it (mirrors adjacent-pair training's
  same-epoch-same-optimizer pattern; logged/saved as `train_t0_loss`/
  `val_t0_loss`). Motivated by a real gap: `horizons_s` never includes a
  literal `0.0`, so without this, nothing supervises t=0 once the main
  loop starts -- main-loop gradients, driven purely by the `0.2s`-`10.0s`
  targets, are free to erode whatever t=0 quality autoencode pretraining
  established, with zero corrective signal until the post-training t=0
  diagnostic reports the damage after the fact (too late to do anything
  about it). Data is derived for free regardless (pure reshaping of
  already-recorded data, no new simulation) so there's no real cost to
  keeping this on.
  - **Full diagnostics, not just the scalar loss** (`_run_t0_pass`/
    `_log_t0_diagnostics` helpers, shared with decoder-only pretraining
    below): per-horizon pos/vel/spin RMSE + oob/goal BCE breakdown (same
    format as the main per-horizon block, prefixed `train t0`/`val   t0`),
    R² against the train-mean baseline (valid here since the t=0 target IS
    exactly the recorded state at that horizon -- the same quantity
    `target_var` is computed over), and (val only) confusion-based
    accuracy/precision/recall, reusing `compute_group_sq_err`/
    `compute_confusion_counts` by wrapping the single `forward_at`
    prediction in a 1-element list. Deliberately does NOT compute
    `err_pct_disp`/`err_pct_ballistic` for this pass -- those baselines
    assume the ORIGINAL episode's t=0 state as the starting point, but
    this task's "input" is horizon h's OWN recorded state reused as a
    pseudo-initial-state, so reusing those baselines would silently
    compare against the wrong reference point.
- **Decoder-only pretraining** (`physics_pretrain.ball.
  decoder_only_pretrain_epochs`, default 0/disabled;
  `decoder_only_pretrain_lr`, falls back to `lr`): an optional phase, AFTER
  autoencode pretraining and BEFORE the main joint loop, that trains the
  decoder plus `encoder.out` (the latent-producing layer) on the REAL
  per-horizon dynamics task (main per-horizon loss + adjacent-pair + t=0
  term, all three, same as a regular main-loop epoch), with `encoder.trunk`
  (everything before `encoder.out`) frozen. Motivated by the same failure
  class as the classification leak fixed via `BallDynamicsDecoder`'s
  identity-shortcut masks, one level further out: on epoch 1 of the main
  loop, the decoder's dynamics-prediction capacity is still essentially
  untrained (autoencode pretraining only covers t=0 reconstruction +
  classification), so its initially large, far-from-converged loss would
  otherwise drag the encoder's already-good identity mapping around, the
  same way the classification loss did before that was masked off. The
  encoder having already learned to represent the initial state accurately
  (autoencode pretraining + identity shortcut) is a reasonable starting
  point for a ROUGH dynamics guess even before any decoder-only training
  happens -- this phase just lets the decoder get most of the way there
  against a STABLE target before the encoder's trunk is allowed to start
  moving too. Full diagnostic logging, reusing the exact same helpers/
  format as a regular main-loop epoch (per-component breakdown, R²,
  `err_pct_disp`, `err_pct_ballistic`, val confusion metrics) -- but NOT
  persisted to `history`/`.history.npz`/the HTML report, matching
  autoencode pretraining's own precedent, and keeping `history`'s "epoch"
  indices unambiguous relative to the `epochs` config value. No LR
  schedule here (same convention as autoencode pretraining) -- deliberately
  a short warm-up, not a training run in its own right. Optional early
  stopping IS available (see the post-launch fix below), separate from the
  main loop's own `early_stop_patience`/`early_stop_min_delta`.
  - **Post-launch fix (2026-08-19): best-val tracking + restore.** This
    phase originally had no best-state tracking at all -- whatever the
    model looked like after the LITERAL LAST epoch got handed to the main
    loop, even if that epoch was a bad one. Observed directly in a real
    run: epoch 20/20's val_loss spiked to ~4.5x the typical level (this
    phase's LR is flat/unscheduled and can be noisy), and the main loop's
    very first epoch showed a correspondingly elevated loss consistent
    with starting from that bad state rather than an earlier, better one
    (epoch 18's val_loss, in that run). Fixed by tracking the best-val
    epoch's weights and restoring them unconditionally at the end of this
    phase, regardless of whether early stopping (below) is enabled.
  - **Post-launch fix (2026-08-19): optional early stopping for this
    phase.** Originally always ran the full configured
    `decoder_only_pretrain_epochs` count no matter what, unlike the main
    loop. New `decoder_only_early_stop_patience`/`decoder_only_early_stop_
    min_delta` config keys (0/disabled by default, no behaviour change) add
    an early BREAK out of this phase's epoch loop once val hasn't improved
    for that many consecutive epochs -- separate knobs from the main loop's
    `early_stop_patience`/`early_stop_min_delta`, since this is typically a
    much shorter phase with its own LR and dynamics, so the same patience
    count doesn't mean the same thing here. Best-val weight tracking/
    restore (above) always happens regardless of this setting; the patience
    counter only controls whether the loop can end early.
  - **Post-launch fix (2026-08-19): `encoder.out` trainable, with its
    identity rows gradient-masked.** Originally the ENTIRE encoder was
    frozen during this phase (`requires_grad_(False)` on every encoder
    param). Left that way, the latent's non-identity "spare" dims sit
    completely untouched through this whole phase, never getting a chance
    to shape themselves toward anything dynamics-useful before the main
    loop starts. Now only `encoder.trunk` (input->hidden->hidden->
    bottleneck, everything upstream of the identity-shortcut concat, which
    bypasses it entirely -- see `BallDynamicsEncoder`'s docstring) stays
    frozen; `encoder.out` (the final Linear producing the latent, added to
    `decoder_only_optimizer`'s param list alongside `model.decoder.
    parameters()`) is trainable. Its IDENTITY rows (`weight[:N_IDENTITY_
    SHORTCUT_FIELDS, :]`/`bias[:N_IDENTITY_SHORTCUT_FIELDS]`, producing
    `latent[0:N_IDENTITY_SHORTCUT_FIELDS] ~= raw pos/vel/spin input`) are
    still gradient-masked for the duration of this phase specifically
    (`register_hook`, added right before the phase and `.remove()`'d right
    after) -- this phase's decoder is at its freshest/noisiest (the whole
    reason this phase exists), and unlike the main loop (where the decoder
    has already calmed down by the time the encoder is exposed to it
    again), leaving those specific rows unmasked here would reopen the
    same corruption failure mode this session already found once for the
    decoder's own dedicated units, just one layer further upstream. Same
    "no soft version of this" precedent as that fix -- scoped to ONLY this
    phase, since autoencode pretraining and the main loop already train
    `encoder.out` fully unmasked and that continues to work as intended
    there (the decoder's gradient is much gentler by the time the main
    loop starts).
  - Verified with `test_decoder_only_training_trunk_frozen_but_out_
    trains_with_identity_rows_masked`: `encoder.trunk` and `encoder.out`'s
    identity rows stay byte-identical across several training steps, while
    `encoder.out`'s spare rows and the decoder both measurably move. (The
    older `test_decoder_only_training_freezes_encoder` still passes too --
    it exercises the same general freeze-then-optimize mechanism
    standalone, just no longer describes this phase's actual current
    scoping.)
- **Shared engineered-feature helper**: `compute_engineered_features()` in
  `ball_episode_gen.py` factors out the speed/spin/Magnus formulas (fields
  14-19) so `_encode_input` (single-row, freshly-sampled state) and the
  adjacent-pair/autoencode reconstruction code (batched, RECORDED state
  reused as a pseudo-input) can't drift apart -- both need the identical
  formulas applied to different kinds of pos/vel/spin data.
- **Identity-shortcut init** (`physics_pretrain.ball.
  identity_shortcut_enabled`, default false; `identity_shortcut_noise_std`,
  default 0.05): motivated by the t=0 autoencoding sanity check target
  being a much lower bar than general dynamics prediction -- reconstructing
  a 9-dim (pos/vel/spin) state through an overcomplete latent (e.g.
  `latent_dim=64`) shouldn't need to be learned from scratch by a randomly
  initialized network, since a near-lossless copy-through solution is known
  to exist and is cheap to hand-construct as an init.
  - **Encoder**: when enabled, the raw input's first
    `N_IDENTITY_SHORTCUT_FIELDS` (=9) fields are concatenated onto the
    bottleneck output right before the final `Linear` to `latent_dim`, and
    that `Linear`'s weights are hand-initialized so `latent[0:9] ~= x[0:9]`
    at init (`_init_identity_shortcut_linear`). Safe because that final
    layer has no activation after it (no ReLU-clipping concern).
  - **Decoder**: 2 of the hidden layer's ReLU units per value (18 of
    `decoder_hidden_dim`, for the 9 values) are hand-initialized to
    implement `latent_i` exactly via `ReLU(x) - ReLU(-x) == x` (needed
    because a naive one-unit identity init would get clipped by the ReLU
    for negative latent values, which are common here), reading ONLY that
    one latent dim (horizon-feature weights zeroed) and writing straight to
    output field `i`, for `i` in `0..9`
    (`_init_identity_shortcut_decoder`). Because the horizon features are
    ignored, this makes the decoder predict the PERSISTENCE baseline
    ("nothing changed") at every horizon at init, not just t=0 -- training
    then only has to learn the delta away from persistence, rather than
    the copy-through behaviour AND the dynamics simultaneously from a
    random start. This lines up with why the persistence baseline
    (`err_pct_disp`, above) was already established as an informative
    reference elsewhere in this file.
  - Both paths route entirely through the latent -- the decoder never sees
    raw input directly -- so unlike the decoder-skip idea considered (and
    rejected) earlier in this section, this does NOT undermine the "what
    capacity does the latent actually need" probe; it only changes the
    STARTING POINT of a round trip the network still has to actually
    perform every forward pass.
  - `identity_shortcut_noise_std` adds Gaussian noise to the otherwise
    exact ~1/-1 identity weights on both sides, so training doesn't start
    already sitting exactly on the target (breaks init symmetry between the
    paired decoder ReLU units too) -- 0.0 gives an exact identity/
    persistence round trip at init (verified in
    `test_identity_shortcut_zero_noise_gives_exact_round_trip`).
  - Requires `latent_dim >= N_IDENTITY_SHORTCUT_FIELDS` (9); raises
    `ValueError` at construction otherwise.
  - **Post-launch fix (2026-08-19): the shortcut didn't survive real
    training.** Observed directly in production (`physics_runs.md`):
    `pos_rmse` went from ~0.3m at the pre-training baseline to 5+m within
    1-2 epochs while `oob_bce`/`goal_bce` dropped sharply over the same
    epochs -- not instability, gradient descent correctly exploiting a
    shortcut the init accidentally left open. Two distinct problems in
    `_init_identity_shortcut_decoder`, both in the "spare" hidden units
    (everything past the 18 dedicated ones):
    1. The DEDICATED units are alive (real latent signal flows through
       them), and `second.weight`'s oob/goal output rows started zero but
       UNMASKED against the dedicated units' columns -- gradient descent
       cheaply routed classification through those already-informative
       activations, which pulled gradient back through the dedicated
       units' OWN weights too (now shared between their "home"
       reconstruction row and classification), corrupting the identity
       mapping directly. This is the dominant leak. Fixed with a permanent
       backward-hook gradient mask on `second.weight[dim:, :2*dim]` --
       classification can never read the dedicated units, full stop, for
       the lifetime of training. Kept as a hard, unconditional block (no
       "soft" variant considered): this isn't about using information,
       it's about a second, unrelated task getting to retune the specific
       weights that are supposed to guarantee an exact reconstruction.
    2. Separately: the spare units start at weight=0, bias=0 on BOTH
       layers, which is a genuine mathematical fixed point under gradient
       descent (pre-activation exactly 0 always, `ReLU'(0)=0` by
       convention, so zero gradient reaches either layer, forever,
       regardless of loss or learning rate) -- verified directly: after 50
       adversarial training steps, every example produced the IDENTICAL
       oob logit, a pure bias-only classifier with zero per-example
       discrimination. Fixed by re-random-initializing the paths meant to
       be genuinely free: spare units' read access to the non-identity
       latent dims + 3 horizon features, and their write access to the
       oob/goal logits. Their write access to pos/vel/spin stays zero (so
       the epoch-0 baseline is preserved) but unmasked/trainable, since
       the main loop still needs spare units to learn real per-horizon
       dynamics deltas later.
    - A third question came up in review, not a bug: should spare units
      also be masked from ever READING the identity latent dims (the
      mirror image of fix #1)? Initially masked that way too, but
      reconsidered -- oob/goal genuinely ARE predictable from raw
      position, so forbidding the classifier from ever using that signal
      just forces it to redundantly re-derive position-like features
      elsewhere in the latent, wasting capacity. Unlike leak #1 (which
      threatens the dedicated units' OWN weights directly), a spare unit
      reading the identity dims only threatens the encoder's identity
      weights, indirectly, and only in proportion to how strongly it's
      wired in -- so instead of a hard mask, `first.weight[2*dim:, :dim]`
      gets a small (not zero, not frozen) random init: quiet at init, but
      genuinely free to grow if gradient descent finds it worthwhile.
      Verified this stays gentle (no collapse) under an 8-step adversarial
      full encoder+decoder test: `pos_rmse` oscillated in the 0.11-0.16m
      range rather than blowing up.
    - Regression test: `test_identity_shortcut_survives_adversarial_classification_gradient`.

  - **Per-phase checkpoints + resume (2026-08-19).** `train()` now writes a
    full model (encoder+decoder) checkpoint at `<output>.after_autoencode.pt`
    when autoencode pretraining finishes, `<output>.after_decoder_pretrain.pt`
    when decoder-only pretraining finishes, and `<output>.after_training.pt`
    after the main loop (and any best-val-weights restore) finishes -- on top
    of, not replacing, the existing final encoder-only artifact saved at
    `output_path` itself. A new `init_checkpoint` param (`--init-checkpoint`
    on the CLI) loads one of these (or an older encoder-only artifact, via a
    fallback path that only restores `model.encoder`) BEFORE any phase runs,
    so a later run can resume from any of the three points. Which phases
    still execute on top of the restored weights is controlled entirely by
    the existing `*_pretrain_epochs` config knobs -- e.g. resuming from
    `after_autoencode` with `autoencode_pretrain_epochs=0` skips straight to
    decoder-only pretrain/the main loop, while a nonzero value continues
    autoencode pretraining further first on top of the restored weights.
