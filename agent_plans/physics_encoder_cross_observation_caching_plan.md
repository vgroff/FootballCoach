# Physics-Encoder Cross-Observation Caching — True N-Player Invocation Bound

> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

## 0. Status

**Not started — planning only.** Follow-up to the physics-encoder
integration described in `agent_plans/ball_physics_pretrain_plan.md` §8 and
`ai/knowledge.md`'s "Frozen physics-dynamics encoders" section (both
implemented). This plan covers a real gap in that implementation, found and
explicitly scoped out at the time — see §1.

## 1. Context: the gap

The current implementation (`ai/models/decision_network.py`/
`execution_network.py`/`physics_encoders.py`) computes each frozen physics
block **once per observation**, shared between that observation's
`DecisionNetwork` and `ExecutionNetwork` calls (via new `DecisionHeadsRaw`
passthrough fields). In the current 1v1 curriculum this happens to look
like "the ball encoder runs twice per tick, each player runs twice per
tick" — but that's a coincidence of there being at most 2 observations
(trainee + one secondary) per tick today, not real caching.

**The actual target**, for a future N-player curriculum (e.g. full 11v11,
22 players): each entity's physics block should be computed **at most twice
per tick** — once per canonical team-frame (`Team.LEFT` observer / `Team.RIGHT`
observer, the only two frames `obs/canonical.py`'s `CanonicalNetworkWrapper`
ever produces) — **regardless of how many players are simultaneously making
a neural decision that tick**. For 22 players that's `22 × 2 (player) + 2
(ball) = 46` total encoder invocations, not up to `22 × 2 (per-observation,
self+other batched) = 44` **calls** that internally redundantly re-encode
the same physical entities up to 21 times over inside each observation's
`other_feat` batch.

Today's implementation does **not** hit that bound for N>2 players: every
`DecisionNetwork.forward()` call (one per observation, i.e. one per
player's decision this tick) independently recomputes the ball's physics
block and every other-player's physics block from scratch, even though many
of those recomputations are numerically identical to ones already done for
a different observation earlier the same tick (same entity, same team-frame).

This was a known, deliberate scope cut at implementation time — see the
"Known, explicitly out-of-scope limitation" note in that session's plan and
in `ai/knowledge.md`. Not a bug in what shipped; a documented gap for
whenever it matters (Phase 1 is 1v1-only today, so it doesn't yet).

## 2. Key design insight: physics-encoder inputs are entity-intrinsic

`ball_live_to_physics_input`/`player_obs_to_physics_input`
(`ai/physics_pretrain/live_encoder_features.py`) only ever read
**absolute/self-referential** fields — `pos_x`/`pos_y` (absolute position),
`velocity_x`/`velocity_y` (world-frame), `heading_sin`/`heading_cos`,
`desired_dir_x`/`desired_dir_y`, own attributes, `has_possession`,
`stamina`. **Never** an observer-relative field (`rel_dx`/`rel_dy`,
`ball_rel_dx`/`ball_rel_dy`, etc.). This means a given physical entity's
physics-encoder input does **not** depend on *who* is observing it — only
on **which team's canonical mirror** gets applied to it, and that mirror
has exactly 2 possible values (`x_sign = +1` for a `Team.LEFT` observer,
`-1` for `Team.RIGHT` — see `obs/canonical.py`'s `x_sign_of()`).

Consequence: the cache key is `(entity_id | "ball", observer_team)` — a
`(N_entities × 2)`-sized table, not `(N_observations × N_entities)`. And
critically, **populating it does not require running the full
`encode_observation()` pipeline once per frame** — the physics-encoder's
input fields can be read directly off `Player`/`Ball` attributes and mirrored
by a simple per-field sign flip (mirroring `canonical.py`'s own
`_mirror_columns()` logic, but applied to raw entity state instead of an
already-batched `PlayerFeatures`/`BallFeatures` row). This sidesteps
needing an "observer" at all for cache population — the frame determines
the mirror, not a specific observing player.

## 3. Design

### 3.1 Cache object and ownership

A new `PhysicsFeatureCache` (working name), holding:
```python
{(entity_id_or_BALL, team): physics_full_tensor}
```
computed lazily, invalidated by tick number (`Player.ai.act()` already
receives `trial_tick: int` — reuse this as the staleness key, no new engine
hook needed: "if cached_tick != trial_tick: recompute for the whole
match").

**Where it lives is the important architectural decision.** `engine/match.py`
must **not** import anything from `ai/models/` — this repo enforces a strict
one-way dependency (`ai/` is a pure consumer of `engine/`/`entities/`, see
the first lines of `ai/knowledge.md`). A cache holding references to
`BallPhysicsFeatureBlock`/`PlayerPhysicsFeatureBlock` (both `ai/models/`
types) therefore **cannot live on `Match`**, even though `Match` is the
natural per-tick-lifecycle object and already owns `self.players`/`self.ball`.

Recommended instead: own it on `PPOTrainer` (or whatever plays that role at
inference time — `PPOTrainer.load_for_inference()`'s trainer instance).
Every `NeuralPlayerAI` in a match already shares the *same*
`trainer._sample_action` callable (`rules_ai.py`'s `NeuralPlayerAI.__init__`
takes `sample_action_fn`, always `trainer._sample_action` in this codebase)
— so `PPOTrainer` is already the one object visible to every player's
decision this tick, without the engine needing to know the cache exists at
all. `_sample_action()` becomes the natural place to check-and-populate the
cache before building the widened tensors DecisionNetwork used to compute
internally.

### 3.2 Populating a tick's cache

Once per tick (lazily, on first access), for `match.ball` and each
`match.players` entry:
1. Read the raw absolute/self-referential fields directly off the entity
   (position, velocity, heading, attributes, stamina, possession,
   `last_desired_speed_mode` — see `ai/knowledge.md`'s "Heading and
   previous-decision movement intent" for where these live on `Player`).
2. Build the physics-encoder's native input tensor **twice** — once mirrored
   for `x_sign=+1`, once for `x_sign=-1` — reusing the same field-sign
   convention `obs/canonical.py`/`obs/augment.py`'s `PLAYER_FLIP_X_IDX`
   already encode (just applied to raw scalars instead of a batched tensor
   column).
3. Run `BallPhysicsFeatureBlock`/`PlayerPhysicsFeatureBlock` on each,
   batched across all entities at once per frame (one `(N,)`-batch forward
   call per frame per entity type, not N separate calls) — `2` ball calls
   total (trivial, always batch size 1) and `2` player calls total (each
   batched over all `N` players at once), matching the target invocation
   count from §1.

This step can reuse `BallPhysicsFeatureBlock`/`PlayerPhysicsFeatureBlock`
unchanged (`ai/models/physics_encoders.py`) — only the *input reconstruction*
needs a new raw-entity-to-tensor path, parallel to but distinct from
`ball_live_to_physics_input`/`player_obs_to_physics_input` (which read
already-encoded `PlayerFeatures`/`BallFeatures` columns, not raw `Player`/
`Ball` attributes directly). Likely a new pair of functions in
`live_encoder_features.py`, e.g. `ball_entity_to_physics_input(ball, mirror_x)
-> Tensor[20]` / `player_entity_to_physics_input(player, mirror_x) ->
Tensor[24]`.

### 3.3 Consuming the cache — DecisionNetwork's forward() needs a bypass

Today `DecisionNetwork.forward()` always computes the physics blocks
internally from `ball_feat`/`self_feat`/`other_feat`. To consume a
precomputed cache instead, it needs an optional bypass: accept
already-computed `ball_physics_full`/`self_physics_full`/`other_physics_full`
tensors as new optional `forward()` arguments, skipping the internal
`self.ball_physics_encoder(...)`/`self.player_physics_encoder(...)` calls
when provided. The existing internal-compute path stays as the default/
fallback (needed for standalone use — tests, `physics_value_net.py`-style
diagnostics, or any caller that doesn't have a `PhysicsFeatureCache` handy).

**`other_physics_full` needs the same slot order as `other_feat`.**
`encode_observation()`'s random other-player slot shuffle already produces
`slot_player_ids` (see `ai/action/apply_nn_action.py`'s
`encode_slot_player_ids()`) — the caller (`_sample_action()`, now holding
both the cache and `slot_player_ids`) assembles
`other_physics_full[slot] = cache[(slot_player_ids[slot], observer_team)]`
per slot before calling `decision_net.forward(...)`.

### 3.4 Scope boundary: live rollout/inference only, not `_ppo_update()`'s augmented batches

This cache only pays off where multiple observations share the same
underlying tick/entities — true during **live rollout** (multiple
`NeuralPlayerAI`s deciding within the same `Match.step()` tick) and
**BC/pretrain dataset construction** (`record_demonstrations.py` recording
multiple players' rows from the same tick). It does **not** extend to
`_ppo_update()`'s already-collected, `augment_batch()`-expanded minibatches:
`obs/augment.py`'s `flip_y` random augmentation is a *further* transform
applied on top of team-canonicalization, changing `ball_feat`/`self_feat`'s
numeric content per augmented copy — an opaque cached latent has no defined
transform under that further reflection (the exact problem
`ball_physics_pretrain_plan.md` §8.1 already rejected solving by baking a
latent into `BallFeatures` for this same reason). PPO gradient-step forward
passes keep computing physics features per-augmented-sample as they do
today; this plan only targets the rollout-collection/inference path.

## 4. Open questions (not resolved here)

- **Cache invalidation granularity**: per-tick (every physics tick) vs.
  per-decision-interval (every `decision_interval_ticks`, since physics
  blocks are only ever consumed on a decision tick anyway) — the latter is
  cheaper but needs the cache to know the decision cadence, which today is
  a property of each `NeuralPlayerAI` instance, not global. Needs a single
  source of truth if pursued (probably `PPOTrainer`/`ScenarioEnv`, not
  per-player).
- **Multi-env/parallel-rollout interaction**: `ppo.n_parallel_envs` (see
  `ai_trainer_knowledge.md` §3.7) runs N worker subprocesses, each with its
  own `Match`/`PPOTrainer`-equivalent — the cache would need to be
  per-worker, not shared across processes. Should fall out naturally if the
  cache lives on the trainer object each worker already has its own copy
  of, but worth confirming when implemented.
- **Immobile/rules-AI players**: do they need cache entries at all? They
  never drive a `DecisionNetwork.forward()` call themselves, but they DO
  appear in *other* players' `other_feat` — so yes, the cache must cover
  every `match.players` entry regardless of AI type, not just neural ones.
- **Testing strategy**: needs a test proving the invocation-count bound
  holds as N scales (e.g. spy-count assertions like
  `tests/ai_unit/test_physics_encoder_wiring.py`'s existing
  `test_execution_network_never_recomputes_physics_encoders`, extended to a
  synthetic multi-player tick) — not just correctness of the cached values.

## 5. Files likely touched (not exhaustive — plan only)

- `ai/physics_pretrain/live_encoder_features.py` — new raw-entity input
  reconstruction functions (§3.2).
- `ai/models/physics_encoders.py` or a new module — the `PhysicsFeatureCache`
  class itself.
- `ai/models/decision_network.py` — optional precomputed-block bypass args
  on `forward()`.
- `ai/ppo/ppo_trainer.py` — cache ownership + population, wired into
  `_sample_action()`.
- `ai/action/apply_nn_action.py` / wherever `slot_player_ids` is already
  produced — reused for per-slot cache lookup ordering (§3.3).
- `tests/ai_unit/test_physics_encoder_wiring.py` — extended coverage for
  the invocation-count bound at N>2 players.
