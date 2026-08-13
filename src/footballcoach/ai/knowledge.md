# src/footballcoach/ai/

Neural-network / PPO player AI.  The engine, entities, orders, actions, and
config packages are all complete and unchanged - this package is a pure
*consumer* of them.  See `ai_design_doc.md` in the repo root for the full
architecture specification; this file is the operational knowledge note.

## Install

```bash
uv sync --group ai   # pulls torch; base game install stays lightweight
```

## !!!! CRITICAL: Orders vs execution-network labels boundary !!!!

**Orders are INPUT to the execution network and OUTPUT of the decision
network. They are NEVER a source for deriving execution-network labels.**

- **Decision network** (`shoot`, `pass_`, `move`, `tackle`, `get_possession_extra`,
  `mark`, `hold_position` Bernoulli heads, plus `move_region_center`): these
  ARE supervised by "what order would the rules AI issue right now" — that
  is legitimately an order-level/intent-level decision, and reading order
  *type* and order *fields* (e.g. `MoveOrder.target_position`) for THIS
  purpose is correct.
- **Execution network** (`move_direction`, `sprint`, `exec_move`, `kick_*`,
  `tackle_attempt`): these must ONLY be derived from what actually lands on
  the `Player` object after the engine/order machinery runs — i.e.
  `player.desired_direction`, `player.desired_speed_mode`,
  `player.kicked_this_tick`/`last_kick_direction`/`last_kick_power_fraction`/
  `last_kick_spin`. **Never** by re-deriving geometry from an order's fields
  (e.g. `normalize(order.target_position - player.position)`) or reading
  `order.sprint` directly — that bypasses the real physics/turning/braking/
  repulsion/push-kick logic in `_compute_movement_intent()`/`step_player_towards()`
  and produces execution labels that don't match what the rules AI actually
  physically does that tick.
- The current order's *type* IS legitimate INPUT CONTEXT to the execution
  network (e.g. `ai_type`/context features) — reading order type for context
  is fine; reading order *fields* to derive execution *labels* is not.

**This bug has recurred multiple times** — always audit any BC-label-
generation code that reads an Order's fields and ask: "is this deriving a
decision-level label (OK) or an execution-level label (NOT OK, must come
from `player.desired_direction`/`desired_speed_mode`/`kick_direct` output
instead)?" See `agent_plans/bc_execution_label_boundary_and_followups.md`
for the concrete fix history and rationale. Implementation: `phase1_labels()`
in `ai/ppo/bc.py` snapshots player/ball state, runs the decided order's
`execute()` once, reads back `desired_direction`/`desired_speed_mode`, then
restores everything — this makes the exploratory call invisible to the real
simulation.

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
    apply_nn_action.py # Execution outputs -> DIRECT player physics (NO ORDERS - see below)
  models/
    entity_encoder.py    # shared per-entity MLP + nn.MultiheadAttention
    decision_network.py  # DecisionNetwork.from_config() + derive_get_possession_prob()
    execution_network.py # ExecutionNetwork.from_config() + flatten_decision_heads()
  ppo/
    rollout_buffer.py  # RolloutBuffer.add() / compute_gae() / as_tensors() / clear()
    schedules.py       # LR, clip-range, rng_reduction schedules (progress 0→1)
    ppo_trainer.py     # PPOTrainer.from_config() + .train(env, total_steps, phase_id=...)
    rollout_worker.py  # subprocess worker for ppo.n_parallel_envs > 1 -- see
                       # ai_trainer_knowledge.md "Parallel rollout collection"
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

## BC label vector (BC_LABEL_DIM = 24)

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
| 12 | **kick_this_tick** | `Player.kicked_this_tick` flag (set unconditionally by `kick_direct()`) |
| 13 | **tackle_attempt** | `ChaseTackleOrder` / `GetPossessionOrder` contact-tackle (order-type check, see `phase1_labels()`) |
| 14 | valid | 1.0 = use this label |
| 15 | exec_move | 1.0 = player is moving, 0.0 = standstill |
| 16 | ai_type | 0.0=rules, 1.0=immobile, 2.0=neural (reserved, unused) |
| 17 | opponent_ai_type | same coding as [16], for the OTHER player in the match |
| 18-19 | kick_direction | unit vector (dx, dy), read from `Player.last_kick_direction` |
| 20 | kick_power | power_fraction actually used, [0,1], from `Player.last_kick_power_fraction` |
| 21-23 | kick_spin | raw spin vector, from `Player.last_kick_spin` |

`kick_direction`/`kick_power`/`kick_spin` (indices 18-23) are captured at the
same `kick_direct()` chokepoint via `Player.last_kick_direction`/
`last_kick_power_fraction`/`last_kick_spin` (set unconditionally whenever a
kick actually executes, reset alongside `kicked_this_tick` in
`Match._process_orders()`), so BC supervision for the kick execution heads
works automatically for any AI that kicks — no per-AI wiring needed. See
`agent_plans/bc_kick_supervision_plan.md`.

**Critical:** index 12 (`kick_this_tick`) is read directly from
`player.kicked_this_tick` — an unconditional per-tick flag set inside
`Player.kick_direct()` every time kick physics actually executes, reset to
`False` for every player at the start of `Match._process_orders()`. This
flag is set **regardless of which Order (if any) triggered the kick** —
previously `phase1_labels()` used `isinstance(current_exec, (ShootOrder,
KickOrder, PassOrder))`, which silently missed kicks fired by `MoveOrder`'s
push-kick behaviour (`rules_ai.py`'s box-run — the ball carrier kicks ahead
and sprints to it while `current_order` stays a `MoveOrder`, never becoming
a `KickOrder`). That bug meant offline demonstration datasets recorded from
box-run episodes had **zero `kick_this_tick=1` rows** despite kicks being
clearly visible in the UI. Fixed by making `kick_direct()` the single
source of truth for "did this player kick THIS tick", independent of order
bookkeeping — see `Player.kicked_this_tick` docstring in `entities/player.py`.
Index 13 (`tackle_attempt`) still comes from inspecting the current order
type (`ChaseTackleOrder`, or `GetPossessionOrder` while touching the
carrier) — this one was not affected by the bug since tackles are always
issued via an explicit order. Indices 0–11 come from asking the rules AI
what it *decides* next — these are input to the decision network and
movement-related execution heads.

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
  and `val=` separately. Uses the same `bc.downsample_trivial_*` config as
  Phase 1's train loop (train rows only, never the held-out val split) —
  same per-epoch frac schedule (`downsample_trivial_frac_default` /
  `_frac_high_epoch` / `_epoch_threshold`), same trivial-row cache. Config:
  `demo_value_pretrain_epochs`,
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

Two separate config keys size the on-policy rollout collected before PPO
starts — they are **not** the same knob and are never meant to share a
value, even though they play a similar role:
- `bc.combined_pretrain_rollout_steps` — sizes the rollout used by
  `pretrain_combined()`'s Phase 2/3 value warm-up (the path used when
  `--bc-dataset` is supplied — i.e. normal offline-BC training).
- `bc.value_pretrain_steps` — sizes the rollout used by the standalone
  `pretrain_value()` **fallback** path (only used when no `--bc-dataset` is
  given, i.e. online-BC training). Also independent of `ppo.rollout_steps`
  (the main PPO rollout buffer size once training is underway).

### Decoupled policy/value learning rates and PPO-time value training

The shared Adam optimizer historically used ONE learning rate for both the
policy trunk/heads and `execution_net.value_head`/`value_ai_type_channel`. Two
problems compounded because of this:
1. `ppo.learning_rate` is deliberately tiny (single-digit `1e-6`s) to protect
   the BC-primed policy from large destructive steps — but the value head
   needs a much larger LR to track the returns distribution.
2. PPO's per-minibatch KL early-stop (`target_kl`, see below) frequently
   cuts a rollout's gradient steps down to 1–5 minibatches. Since the shared
   optimizer only steps while the policy loop is running, the value head was
   getting starved of updates on top of using an LR far too small for it —
   together these left normalized value loss stuck well above 1.0
   (worse than "always predict the mean") for the whole PPO phase, despite a
   well-converged pretrain.

Fixes (`PPOTrainer.__init__` / `_ppo_update()`):
- The optimizer is now built with **two named param groups** — `"policy"`
  (everything else) and `"value"` (`execution_net.value_head` +
  `execution_net.value_ai_type_channel`), identified by parameter name prefix.
  `ppo.value_learning_rate` sets the value group's LR independently (falls
  back to `ppo.learning_rate` if absent). `schedules.py`'s
  `TrainingSchedules.value_lr(progress)` returns a constant schedule reading
  this key (mirrors `lr()`'s pattern but currently non-annealing).
  **Not built at all when `separate_value_net=True`** (see "Separate value
  network" below) — in that mode the "value" param group would be dead
  weight (`execution_net.value_head`/`value_ai_type_channel` are frozen and
  unused), so the main optimizer only has `"policy"`/`"direction"` groups
  and a fully separate `value_net_optimizer` trains `trainer.value_net`
  instead.
- `load_checkpoint()` now checks the saved optimizer's param-group count
  against the live optimizer before calling `load_state_dict()`; on a
  mismatch (e.g. resuming an old single-group checkpoint after this change)
  it skips the optimizer-state restore with a `WARNING` log instead of
  raising `ValueError: loaded state dict has a different number of
  parameter groups` — network weights still load normally either way.
- **Value-only continuation**: after the policy epoch loop exits via KL
  early-stop, `_ppo_update()` runs an additional loop (up to
  `ppo.value_only_continuation_epochs`, default = `ppo.n_epochs` for
  backward compat — was previously hardcoded to always reuse `n_epochs`)
  that trains ONLY the value param group (no policy forward/backward, so no
  further KL risk) over fresh random minibatch permutations of the **same**
  rollout batch. Logged as `[value-only continuation] N extra minibatch
  step(s) after policy early-stop final_val_loss=X`. This is a deliberate,
  non-standard technique (the value function has no trust-region /
  importance-ratio constraint, so it's safe to keep training past the
  policy's early-stop point) but note it does reuse the same stale rollout
  data rather than fresh on-policy samples, so there's a real (if usually
  small) risk of overfitting the critic to that batch — raise
  `value_only_continuation_epochs` cautiously and watch for `val=`
  oscillating rather than trending down across rollouts.

If `target_kl` (`ppo.target_kl`) is too tight relative to the natural
per-minibatch KL noise floor, the early-stop fires almost every rollout
after just 1 minibatch, starving the *policy* itself of gradient signal
(the value-only continuation above only compensates for the critic, not the
actor). Symptoms: `steps_this_update=1` in nearly every `[early stop ...]`
log line. Raise `target_kl` (and/or `minibatch_size`, and/or lower
`learning_rate` further) if this happens — `clip_range` already bounds the
per-sample effect of any one step, so a looser KL gate is usually safe.

- **BC-only continuation** (`bc.bc_only_continuation_epochs`, default `0` =
  disabled/opt-in): mirrors the value-only continuation above, but for the
  BC auxiliary loss. Runs AFTER the value-only continuation, still gated on
  `if _early_stopped:`. Rationale: the policy's KL early-stop cuts the
  *combined* policy+value+BC loop short, which was silently truncating BC's
  intended per-rollout gradient budget too (only the value head had a
  dedicated continuation before this). Unlike the value-only continuation,
  BC updates the SAME `decision_net`/`execution_net` params the policy uses
  (there's no isolated "BC param group" to safely train in isolation the way
  `value_head`/`value_ai_type_channel` can be) — but the loop itself still can't
  trigger further early-stops because it never computes a policy
  forward/ratio/KL at all, it's a pure supervised step using
  `bc_loss_from_tensor()` with the SAME annealed `bc_coeff` already computed
  for the rollout (see `bc.aux_coeff_start/end/aux_coeff_anneal_fraction`).
  Logged as `[bc-only continuation] N extra minibatch step(s) after policy
  early-stop final_bc_loss=X`. Raise from `0` (e.g. `2`–`6`) if you suspect
  BC is contributing less than `aux_coeff` implies because of frequent early
  stops; leave at `0` (default) otherwise since it's a newer, opt-in
  mechanism and doubles as an easy on/off switch independent of
  `value_only_continuation_epochs`.

### Single value head convention

There are two `value_head` modules in the codebase (`decision_net.value_head`
and `execution_net.value_head`) for historical/checkpoint-compat reasons, but
only ONE is ever trained or read: **`execution_net.value_head`** (or
`trainer.value_net.value_head` when `separate_value_net=True` — see
"Separate value network" above; in that mode `execution_net.value_head` is
ALSO frozen/unused, same as `decision_net.value_head`).
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

### Separate value network (`separate_value_net` / CLI `--separate-value-net`)

A **permanent** architecture switch (distinct from the throwaway
`experiment_separate_value_net` diagnostic flag inside `pretrain_value()`,
which only runs in the `--bc-dataset=None` fallback path and is discarded
afterward). When `PPOTrainer(separate_value_net=True)`, a second, fully
independent `ExecutionNetwork` (`trainer.value_net` — own trunk/encoders,
zero weight sharing with `execution_net`) becomes the sole critic for the
**entire** run:

- `execution_net.value_head` and `execution_net.value_ai_type_channel` are
  permanently frozen (`requires_grad_(False)`) and excluded from the main
  optimizer entirely — they are structurally unused in this mode.
- `pretrain_combined()`'s Phase 0 (`demo_epochs`) and Phase 1 joint
  value-loss term (`_use_joint_val`) are both force-disabled
  (`self.separate_value_net` short-circuits them to `0`/`False`) — the
  entire point of this mode is a critic that never receives a BC gradient,
  so both of BC's value-loss injection points are skipped rather than
  redirected.
- `pretrain_value()` trains `trainer.value_net` (fully unfrozen — no
  `_get_value_pretrain_freeze_params()` freezing, since there's no BC-primed
  trunk to protect) via `trainer.value_net_optimizer` instead of the usual
  `execution_net.value_head`-only Adam instance.
- `_ppo_update()`'s main value loss forwards a **detached** copy of
  `d_heads` through `value_net` (`dataclasses.replace(d_heads, ...)` with
  every field `.detach()`-ed) so its gradient reaches only `value_net`'s own
  params, never `decision_net` — then does a second, independent
  `.backward()`/`value_net_optimizer.step()` right after the main
  `total_loss.backward()`/`self.optimizer.step()`. The value-only
  continuation loop (see above) also branches to train `value_net` instead
  of `execution_net.value_head` when this mode is on.
- `_get_value()`/`_sample_action()` both route their value read through
  `value_net` when enabled (via the `_value_heads()` helper for the former;
  `_sample_action` calls `self.value_net(...)` directly under `no_grad()`
  since `e_heads` there is still needed for the actual action sampling).
- Checkpoints gain two extra keys, `"value_net"`/`"value_net_optimizer"`,
  written by `_save_checkpoint()`/`_save_checkpoint_to()` and restored by
  `load_checkpoint()` (with a `WARNING` log — not a crash — if an older
  checkpoint lacks them). `PPOTrainer.load_for_inference()` auto-detects
  `separate_value_net` by peeking at the checkpoint file for a `"value_net"`
  key before constructing the trainer, so no CLI flag is needed at
  evaluation/inference time.

See `tests/ai_scenario/test_separate_value_net.py` for coverage (construction,
value routing, gradient isolation between `value_net` and
`execution_net.value_head`, checkpoint round-trip, `load_for_inference()`
auto-detection).

### Value-only opponent-AI-type side channel: permutation invariance

The value-only opponent-ai-type side channel (`value_ai_type_channel`,
instance of `ValueAiTypeSideChannel` in `ai/models/value_side_channel.py`, on
both `DecisionNetwork` and `ExecutionNetwork`) used to be a flatten+`Linear`
over `other_ai_type` (shape `(batch, MAX_OTHER_PLAYERS, AI_TYPE_ONE_HOT_DIM)`
flattened to `(batch, N*dim)`). That gives each slot position its own
learned weight block — **not** permutation-invariant, unlike the main
entity encoder (shared per-slot MLP + attention pooling), meaning swapping
which physical player occupies slot 3 vs slot 17 changed the value output.
Slot-shuffle augmentation was the only thing papering over this gap by
showing many shuffles per real transition.

Fixed by replacing it with `ValueAiTypeSideChannel`: a shared per-slot MLP
(own weights, zero sharing with `EntityEncoder`) + a dedicated masked
attention pool, exactly permutation-invariant by construction (same
guarantee as the main entity encoder). It's also enriched with a **detached**
copy of the main entity encoder's per-slot embeddings (`entity_encoder(...,
return_embeds=True)` returns `(context, self_embed, other_embed)` — the
latter two are the pre-attention per-slot embeddings, still attached to the
policy's autograd graph until the caller detaches them) so the critic sees
real spatial/attribute context per opponent, not just a bare AI-type
one-hot. The `.detach()` call in `DecisionNetwork.forward()`/
`ExecutionNetwork.forward()` is the load-bearing line keeping this value-only
— see the "Opponent-AI-type (value-only)" design constraint above (policy
must never condition on opponent identity).

See `tests/ai_unit/test_ai_type_side_channel.py::TestValueSideChannelPermutationInvariance`
for the regression test (moving the one real other-player from slot 0 to
slot 17 must not change `value` at all).

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
player.on_kick    = lambda player: ...   # fired when kick_direct() executes kick physics
player.on_tackle  = lambda player: ...   # fired when ChaseTackleOrder makes contact
```
The engine fires these in `match.py`/`player.py` at the exact tick the action
executes — not when the order is set.  Useful for: BC recording, UI effects,
logging, statistics.  Both default to `None` (no-op).

`on_kick` fires from **any** code path that calls `Player.kick_direct()` —
`KickOrder`/`ShootOrder`/`PassOrder.execute()` all delegate to it, and so does
`MoveOrder`'s push-kick behaviour (`_do_push_kick()` in `orders.py`) and the
neural network's direct-drive kick action. Alongside the optional callback,
`kick_direct()` also unconditionally sets `player.kicked_this_tick = True`
(reset to `False` for every player at the top of
`Match._process_orders()`) — this flag exists specifically so code that runs
*after* order processing (e.g. `bc.py`'s `phase1_labels()`) can check "did
this player kick this tick" without needing `on_kick` wired up and without
inspecting order types (which missed the `MoveOrder` push-kick case — see
the BC label table above).

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

The **only** things `apply_action_to_player()` in `ai/action/apply_nn_action.py` does:
1. **Movement**: `gating.exec_move` selects STANDSTILL vs moving; sets
   `player.desired_direction` (Vector3) and `player.desired_speed_mode`
   (SpeedMode) directly from `gating.move_direction` and `gating.sprint`. The
   engine's `_apply_movement()` loop reads these fields.
2. **Kick**: calls `player.kick_with_direction(match, direction_3d, power, spin)`
   if `gating.kick_this_tick` is True. This executes kick physics immediately
   with no KickOrder. (`kick_with_direction` is a parallel chokepoint to
   `kick_direct` — used by `KickOrder`/rules AI/`MoveOrder`'s push-kick — that
   takes an explicit 3D direction instead of an aim point; both set
   `kicked_this_tick`/`last_kick_*` and fire `on_kick`.)
3. **Tackle**: sets `player.tackle_armed = True` if `gating.tackle_attempt` is
   True and preconditions are met (else returns `illegal_action=True`); there
   is no `tackle_direct()` method. `Match._check_armed_tackles()` resolves the
   armed tackle on contact — the same mechanism a rules-AI `ChaseTackleOrder`
   ultimately arms.

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

### `HybridPlayerAI` — human/rules override on top of `NeuralPlayerAI` (UI training mode)

`rules_ai.HybridPlayerAI(sample_action_fn, ...)` subclasses `NeuralPlayerAI`
and adds two independent, opt-in override channels so a single player can be
driven by a mix of neural network control and direct human/rules
intervention — the intended general pattern for "some players neural, some
human, some rules-based, mixed per-player" going forward (see the UI's
training-mode `N` hotkey in `ui/app.py::_toggle_training_ai_mode` for the
current live consumer).

- **Channel 1 — order override** (`issue_order(order)` /
  `clear_order_override()`): assigns a real `Order` (`MoveOrder`,
  `ShootOrder`, `KickOrder`, ...) to `player.current_order` and skips the
  neural network entirely — no sampling, no `last_transition` — for as long
  as that order is in progress, exactly like a rules-based `PlayerAI` would.
  Control reverts to the network automatically the tick after the engine
  clears `player.current_order` back to `None` (order completed). This is
  "take direct control," bypassing the execution network's learned motor
  skill for that action — used by `MatchInputController._issue_order()` /
  the kick UI (`ui/input.py`), which detect `isinstance(player.ai,
  HybridPlayerAI)` and route every click/kick through this channel instead
  of writing `player.current_order` directly, so a human click on a
  neural-controlled trainee "takes over" for exactly one order.
- **Channel 2 — decision-neuron override**
  (`set_decision_override(head_name, value)` / `clear_decision_overrides()`):
  patches `decision_probs[head_name] = value` **after** the network samples
  but **before** `select_action()`'s winner-take-all gating runs — i.e. "give
  the neural net an order via its own decision neurons" (e.g. force
  `move`'s probability to `1.0` to guarantee the MOVE head wins gating this
  decision tick) while the execution network still supplies all the
  physical motor output (`move_direction`, `sprint`, kick physics, tackle).
  Valid head names match `ai/action/gating.py`'s `_HEAD_ORDER`: `shoot`,
  `pass_`, `move`, `tackle`, `get_possession`, `mark`, `hold_position`. Only
  takes effect on ticks where the network actually samples a fresh decision
  (every `decision_interval_ticks` ticks) and only when channel 1 isn't
  active.

Channel 1 always takes priority over channel 2 and over the network's own
sampling. Both channels are independent of *who* drives them — a human click,
a rules-based `PlayerAI`, or a scripted test can all call
`issue_order`/`set_decision_override` on the same `HybridPlayerAI` instance,
which is what makes this pattern reusable beyond the training-mode UI.

### `_sample_action(obs_dict)` still returns an 8-tuple internally:
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
`N_FLIP_VARIANTS` (2) × `ppo.augment_n_slot_shuffles` (default 6× total):
- 2 geometric flips: identity, flip_y (the one remaining exact pitch symmetry
  once the attacking axis is fixed — see "Canonical AI frame" below for why
  flip_x is no longer part of this random augmentation)
- n slot permutations per flip (exact for permutation-invariant attention)
Field indices for each flip are derived from `fields(PlayerFeatures)` /
`fields(BallFeatures)` at import time — see `obs/augment.py` for the full
derivation including pseudovector (spin) transforms. Always use
`augment.N_FLIP_VARIANTS` (never a hardcoded `4`) when tiling a parallel
array (e.g. `ret_batch.repeat(...)`) to match the augmented batch size —
a stale hardcoded `4` here (left over from when flip_x was still a random
augmentation) broke `pretrain_combined()`'s Phase 1 value-loss batch-size
match once flip_x was removed from `_FLIP_VARIANTS`.

### Canonical AI frame (`obs/canonical.py`)

Both networks are permanently wrapped in `PPOTrainer.__init__`
(`CanonicalNetworkWrapper`, see `ai/obs/canonical.py`): `self.decision_net`,
`self.execution_net`, and `self.value_net` (when `--separate-value-net`) are
all `CanonicalNetworkWrapper` instances, not the raw `DecisionNetwork`/
`ExecutionNetwork` modules.

**What it does**: on every `forward()` call, the wrapper negates every
x-signed field (`obs/augment.py`'s `PLAYER_FLIP_X_IDX`/`BALL_FLIP_X_IDX`) in
`self_feat`/`other_feat`/`ball_feat` for a `Team.RIGHT` observer (derived
from that row's `attacking_direction` field — see
`canonical.x_sign_of()`/`X_SIGN_FIELD_IDX`), before delegating to the real
network. So every network input is transformed so "my own team always
attacks +x" — the network never has to learn to condition on which raw
engine team it is; `is_own_team` + this fixed convention is all it needs.

**Why a wrapper and not baked into `encoder.py`/`bc.py`**: `obs/encoder.py`
and recorded BC `.npz` files stay in plain, unmirrored world-frame
coordinates — matching match logs and UI replays, and never needing
re-recording if the convention ever changes. There is exactly ONE
implementation of the mirror (the wrapper), used automatically by every
existing `self.decision_net(...)`/`self.execution_net(...)`/
`self.value_net(...)` call site in `ppo_trainer.py` with **zero changes**
to those call sites — this was a deliberate redesign after an earlier
attempt hand-inserted the mirror at ~15 individual call sites, which was
exactly the kind of hand-duplicated-logic-drift risk this codebase already
has scars from (see "Orders vs execution-network labels boundary" above).

**What the wrapper does NOT do**: it never touches network *outputs*
(`DecisionHeadsRaw`/`ExecutionHeadsRaw`) — those stay in canonical frame.
This is intentional: log_prob/BC-loss computations need the network output
compared against other canonical-frame quantities (BC labels via
`canonicalize_bc_labels()`, the rollout buffer's stored raw action samples
used to recompute PPO's importance ratio) — decanonicalizing here would
just require re-canonicalizing one line later. The ONE place decanonicalize
happens is `PPOTrainer._sample_action()`, right before the sampled
`move_direction`/`kick_direction`/`move_region_center_m` are handed back to
the caller as the actual physical action applied to engine state (via
`mirror_x()`, using the same `x_sign` derived once at the top of that
method) — everything downstream of that point (`apply_nn_action.py`, the
engine) is plain world-frame, same as it always was.

`state_dict()`/`load_state_dict()` are transparently delegated straight to
the wrapped module (bypassing `nn.Module`'s default submodule-prefixed
behaviour) so checkpoint keys are byte-identical to pre-wrapper checkpoints
— no migration needed for old `.pt` files.

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
- `test_apply_nn_action.py` – legal/illegal action detection, correct direct-field application
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
