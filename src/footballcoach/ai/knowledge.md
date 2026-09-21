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
  the `Player` object after running the decided order's `execute()` **as a
  counterfactual** — i.e. `player.desired_direction`, `player.desired_speed_mode`,
  `player.kicked_this_tick`/`last_kick_direction`/`last_kick_power_fraction`/
  `last_kick_spin`/`kick_armed`/`kick_armed_direction`/`kick_armed_power_fraction`/
  `tackle_armed`, all read from the SNAPSHOT/RESET/EXECUTE/RESTORE sandbox
  below, never from the real, already-executed state of whichever AI is
  actually driving the player. **Never** by re-deriving geometry from an
  order's fields (e.g. `normalize(order.target_position - player.position)`)
  or reading `order.sprint` directly — that bypasses the real physics/
  turning/braking/repulsion/push-kick logic in
  `_compute_movement_intent()`/`step_player_towards()`/`_try_push_kick()` and
  produces execution labels that don't match what the rules AI actually
  physically does that tick.
  - **A second, subtler version of this same bug (fixed 2026-09)**: reading
    the RIGHT fields (`player.kicked_this_tick`/`kick_armed`/`tackle_armed`)
    but from the REAL, already-executed state instead of the counterfactual
    one. These are Player-level flags, equally real regardless of which AI
    set them — during recorded demonstrations `Phase1RulesAI` really is
    driving, so reading them "for real" happened to be correct; during
    on-policy PPO/DAgger training the trainee's own `NeuralPlayerAI` is
    driving instead, so the exact same code silently echoed the STUDENT's
    own kick/tackle behaviour back at it, with zero corrective signal. Watch
    for this pattern specifically: an execution field being read from a
    Player/engine-level flag is not enough by itself to prove it's a genuine
    rules-AI counterfactual — confirm it's being read from *inside* the
    snapshot/execute/restore sandbox, not from real pre-existing state.
  - **A third bug (fixed 2026-09), about WHEN the counterfactual runs, not
    WHAT it reads**: `phase1_labels()`'s counterfactual reads whatever
    state `player`/`match` are in AT THE MOMENT it's called. `Match.step()`
    runs `_process_orders(dt)` (real decisions + the observation
    `NeuralPlayerAI` encodes, using state as of the START of the tick)
    BEFORE `_apply_movement(dt)` (which actually advances position/
    velocity/heading). Calling the label function from a rollout-loop
    caller AFTER a full `env.step()` had already returned — the pattern
    `PPOTrainer.train()`/`rollout_worker.py`/`ai/ppo/dagger.py` all used —
    meant it saw POST-movement state, one physics tick later than the
    observation it was meant to accompany. Small for move_direction, large
    for kick (`_try_push_kick`'s geometric gates and run-compensated power
    are sensitive to instantaneous position/heading/velocity, and can flip
    a fire/no-fire boolean right at a threshold). Fixed by moving WHERE the
    label is computed, not by anything inside `phase1_labels()` itself:
    `NeuralPlayerAI` gained an optional `bc_label_fn` (see its own
    docstring in `rules_ai.py`), called from INSIDE `act()` at the exact
    instant the observation is encoded — callers now set
    `env.bc_label_fn` (mirroring `env.sample_action_fn`) instead of calling
    a label function themselves after the fact. `phase1_labels()` was
    split into `phase1_labels(env, player_id=None)` (a thin wrapper, still
    fine for callers like `record_demonstrations.py` that already compute
    obs+label together, synchronously, from a live `env` they hold) and the
    real logic, `phase1_labels_for_player(player, match)` (the function
    `NeuralPlayerAI.bc_label_fn` actually calls). See
    `curriculum.envs.bc_label_fn_for_phase()` (env-based) vs
    `bc_label_fn_for_phase_player()` (player/match-based) — these are
    NOT interchangeable, using the wrong one for a given call site silently
    reintroduces either this bug or a `TypeError`.
  - **A fourth bug, found while fixing the third**: the counterfactual
    `order.execute()` can cause a real push-kick to fire, and
    `Player.kick_direct()` draws its yaw/pitch noise from `match.rng` — ONE
    `random.Random` shared by the whole simulation (every kick, tackle
    roll, etc.). Without protecting it, this exploratory call silently
    consumed real draws from that stream, desyncing every subsequent
    genuine random outcome in the match. Fixed with
    `match.rng.getstate()`/`setstate()` around the same snapshot/restore
    block that already protects position/velocity/kick_armed/etc.
- The current order's *type* IS legitimate INPUT CONTEXT to the execution
  network (e.g. `ai_type`/context features) — reading order type for context
  is fine; reading order *fields* to derive execution *labels* is not.

**This bug has recurred multiple times** — always audit any BC-label-
generation code that reads an Order's fields OR a Player's real physics
flags and ask: "is this deriving a decision-level label (OK, order fields
are fine there) or an execution-level label (NOT OK unless it's read from
inside the counterfactual sandbox, never from real order fields or real
already-executed Player state)?" Implementation: `phase1_labels()` in
`ai/ppo/bc.py` snapshots player/ball state (including the arm-state fields
above), resets `kicked_this_tick`/`kick_armed`/`tackle_armed` to match what
`Match._process_orders()` itself does before a real order's `execute()`
(so the counterfactual run is genuinely stateless, not contaminated by
whatever really happened this tick), runs the decided order's `execute()`
once, reads back every execution field from the result, then restores
everything — this makes the exploratory call invisible to the real
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
    distributions.py  # IndependentBernoulli, MaskedCategorical, SquashedNormalHead, VonMisesDirectionHead, KickDirectionHead
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

## BC label vector (BC_LABEL_DIM = 27)

`bc.py` stores 27 floats per step (see the module docstring in `bc.py` for
the authoritative up-to-date layout table — do not let this count drift out
of sync, it has already changed several times: 15→16 (added `exec_move`),
16→17 (added `ai_type`), 17→25 (added `opponent_ai_type` + kick
direction/power/spin), 25→27 (added `heading_sin`/`heading_cos`, read
directly off `player.heading_rad` at record time — see `phase1_labels()`)):

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

### BC loss floor-adjusted breakdown logging

`compute_bc_loss_floor()` (analytic minimum achievable BC loss under label
smoothing — see its docstring for the `H(y') = -y'ln(y')-(1-y')ln(1-y')`
derivation) is now a thin wrapper over `compute_bc_loss_floor_components()`,
which returns the same per-head floors broken out by component (`decision`,
`exec_bce`, `sprint`, `move`, `tackle_attempt`, `kick`; `direction`/`region`/
`kick_direction`/`kick_power`/`kick_spin` are always `0.0` — true floor,
not BCE-smoothed). The floor for a smoothed Bernoulli head depends only on
its smoothing constant (and mildly on `pos_weight`), never on label
balance, since `H(y')` is symmetric in the hard label — so per-component
floors are cheap and dataset-independent to compute.

`PPOTrainer`'s per-epoch `breakdown (floor-adj) decision=... exec_bce=...`
log line (both the main Phase 1 BC loop and the BC repair loop in
`pretrain_combined()`) now subtracts each component's own floor before
printing (clamped at 0 for float noise near convergence). This matters
because `exec_bce` sums 4 smoothed Bernoulli heads while `direction` is an
always-floor-0 cosine loss — comparing their *raw* magnitudes overstates
how much worse exec-head imitation is relative to direction/decision. E.g.
with `dec_label_smoothing=0.01`/`exec_label_smoothing=0.02`, `decision`'s 7
heads and `exec_bce`'s 4 heads each carry floors of ≈0.22 nats — a raw
`decision≈0.23` is therefore almost entirely floor (adjusted residual
≈0.01, i.e. decision heads are close to perfectly imitated), while raw
`exec_bce≈0.35`-`0.36` still has a real, non-floor residual of ≈0.13-0.14
after adjustment — the actual dominant BC gap, not an artifact of summing
more smoothed heads. Un-adjusted component numbers should not be compared
to each other directly; use the floor-adjusted log line, or call
`compute_bc_loss_floor_components()` directly against a raw breakdown dict
from `bc_loss_from_tensor(..., return_breakdown=True)`.

**Same mechanism ported to `ai/physics_pretrain`'s event heads (2026-09)**:
the ball/player dynamics encoders' own sigmoid BCE "event" heads (the
shared decoder's per-horizon `out_of_bounds`/`goal_scored` logits,
`crossing_head`'s `crosses_logit`, and — ball only — `event_head`'s
ever-out-of-bounds/ever-goal logits) were observed to interfere with the
regression heads sharing the same encoder; label smoothing was added as a
complement to the pre-existing down-weighting mitigation
(`bce_loss_weight`/`crossing_crosses_loss_weight`/`event_loss_weight`, all
tuned low). `ai/physics_pretrain/event_head_smoothing.py` is a small shared
module (used by both `train_ball_dynamics.py`/`train_player_dynamics.py`)
providing `smooth_target()` (identical formula to `bc.py`'s `_bce()`) and
two floor helpers: `bce_label_smoothing_floor()` (the exact per-row `H(y')`
tensor, mirroring `compute_bc_loss_floor_components`'s `_floor_bce`) and
`expected_bce_floor(smoothing, pos_weight=1.0)` — a **closed-form scalar**
version used for the actual floor-subtraction in both training scripts,
since `crosses_logit`/`event_head` have no `pos_weight` at all (making
their floor a true smoothing-only constant) and the per-horizon
`out_of_bounds`/`goal_scored` heads' `pos_weight` is itself already a
dataset-level inverse-class-frequency constant (`n_neg/n_pos`, computed
once before training) — so `frac_pos = 1/(1+pos_weight)` recovers the same
dataset-level balance without needing to re-derive it from a live batch
tensor, avoiding per-batch sampling noise in what's meant to be a stable
diagnostic number.

Config: `physics_pretrain.ball.label_smoothing`/`physics_pretrain.player.
label_smoothing`, one independently-tunable value per network (matching
every other knob in that config section — NOT split by head like bc.py's
dec/exec smoothing), default `0.0` = no behaviour change. Both training
scripts subtract the appropriate floor at the single aggregation choke
point each metric already funnels through on its way to both the per-epoch
log line and `.history.npz`/the HTML report — `_mean_breakdown`/
`_mean_breakdown_by_horizon` for `oob_bce`/`goal_bce` (per-horizon,
`np.maximum(0.0, raw - floor_by_h)`), and a handful of named scalar sites
for `crossing_crosses_loss`/`event_loss` (player routes this through
`_AuxAccumulator.summary()`'s `crossing_crosses_loss_floor` param). The RAW
(floor-inflated) value is always what's actually backpropagated and what
`compute_loss`/`_crossing_head_loss`/`_event_head_loss` return — only the
reported/logged/saved numbers are adjusted, same "diagnostics only, never
the optimized loss" convention as the BC pipeline. See
`tests/ai_unit/test_event_head_smoothing.py` for the shared helper's own
coverage, and the `test_*_label_smoothing_*` tests in
`test_ball_physics_pretrain.py`/`test_player_physics_pretrain.py`.

**Extended (2026-09) to the "backprop_loss contribution by head" diagnostic
and the crossing_head sub-term splits, not just the standalone oob_bce/
goal_bce/crossing_crosses_loss/event_loss numbers.** Rationale: those
per-head percentage breakdowns (`main=... crossing=... event=...` and, for
crossing_head, `pos=... crosses=... dt=...`) exist specifically to judge
which head is dominating the shared encoder's gradient budget -- left raw,
a head whose entire reported contribution is mostly its own constant
smoothing floor looks artificially significant, exactly the comparison
this diagnostic is for. `_main_head_floor` (bce_weight-, and for player
also main_loss_weight-, scaled sum of `_oob_bce_floor_by_h`/
`_goal_bce_floor_by_h` across horizons) and the existing `_crossing_
crosses_loss_floor`/`_event_loss_floor` (ball only) are subtracted from
EVERY per-head `backprop_contrib` entry, from the percentage-denominator
total (`mean_backprop_loss` -- an independent per-step accumulator on
ball's train side, so adjusted directly there; a value DERIVED from
`backprop_contrib`'s own already-adjusted entries everywhere else, so
automatically consistent for free), and from `aux_summary`/`_AuxAccumulator.
summary()`'s `crossing_loss` (player) / `mean_crossing_loss` (ball) --
the aggregate pos+crosses+dt crossing_head total, which is ALSO what the
crossing pos/(crosses/)dt split uses as its percentage denominator, so
adjusting it keeps that split's own numerator sum matching its denominator
(this was a real latent bug on the player side until this pass: the split's
`crosses` numerator was already floor-adjusted from the original pass but
its `crossing_loss` denominator wasn't, silently breaking the percentages'
100% sum). **Still never touches `train_loss`/`val_loss` themselves**
(the plain main-task loss, mixing continuous regression with BCE) **or
anything used for a real decision** (`best_val_loss` comparisons, early
stopping, LR scheduling) -- those stay exactly as before; only display-only
derived values (dict entries, locally-computed adjusted variables like
`main_val_contrib`) are touched, verified by re-reading each raw variable's
every other use site before mutating it in place vs. computing a fresh
adjusted copy.

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

### Reward design rule: avoid one-off (event-paid) rewards unless terminal

**Be very careful with one-off rewards/penalties; avoid them unless they are
terminal. Prefer per-step or potential-based terms wherever possible.** (Also
a banner in `ai/env/reward.py`'s module docstring, which is where reward
changes get reviewed.)

Why -- measured 2026-09-20 when `gain_possession_bonus` (+1.0) and
`loss_of_possession_penalty` (-0.9) were removed from phase 1:
- The ground-truth return-to-go is a sum of *future* rewards, so it climbs
  toward a one-off payment and *drops* by about that payment the moment it is
  made -- right after the agent did something good (mean step -0.86 in
  return-to-go after a possession gain, n=1,404). The value net must forecast
  when such events happen (largely unpredictable: tackles, turnovers). Events
  added ~14% to the return variance (4.28 vs 3.68 without them); value squared
  error within -3..+1 steps of an event was higher than elsewhere in the same
  episode phase (1.2x early, up to 6.8x late), though that is confounded with
  turnovers genuinely deciding games.
- A non-potential-based event payment changes the optimal policy; gain +1.0 /
  loss -0.9 netted +0.1 per gain/loss cycle (farmable, though only ~13% of
  episodes had 2+ gains).
- It was redundant: with the ball the trainee won ~77% / lost ~9%, without it
  ~19% / ~66% -- the win/lose terminals already separate the two by ~3.5 of
  return, against +1.0 from the bonus.
- After removal (refit run282): value R2 ~0.77 -> ~0.81 on the new targets (not
  strictly comparable across reward definitions), and the return std inside
  `timeout` episodes fell from ~0.46 to ~0.07 (it was mostly possession-event
  noise).

Rules: (1) terminal one-offs (box, lterm, tout, out, prox) are fine -- the
episode ends so there is nothing for the return to drop into, and they are the
objective. (2) Otherwise use a bounded per-step term or potential-based shaping
`F = gamma*Phi(s') - Phi(s)` with `Phi(terminal)=0` (`appr` is the model: it
telescopes and preserves the optimal policy; note it still offsets the value by
`-Phi(s)`, an observable state function, not a function of future event
timing). (3) A per-step term must be bounded over the max episode length --
`per_step * max_steps` well below the terminal win reward, or holding/dawdling
beats winning (use `cumulative_clamped_delta`). (4) Never pair a bonus with a
near-equal offsetting penalty that nets non-zero. (5) If a non-terminal event
term is truly needed: justify it, default it to 0.0, and measure it first
(recompute MC returns with/without it: return variance, value R2, per-outcome
loss). Still event-paid and non-terminal, so review before enabling/retuning:
`ill` (`illegal_action_penalty`, currently 0.0) and the `tatt`/`katt` attempt
bonuses.

### Possession gain/loss reward: real turnovers only

**Both rewards are DISABLED as of 2026-09-20** (`gain_possession_bonus` and
`loss_of_possession_penalty` = 0.0 -- see the rule above). The transition
counting below is still computed and still correct; it just no longer pays.

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

**Armed-kick auto-fire possession touch (fixed -- was previously invisible)**:
`Match._update_loose_ball_pickup`'s armed-kick branch grants possession
(`self._set_possession(player.player_id)`) then immediately fires the kick,
whose `_release_kick` sets `ball.possessed_by = None` again -- BOTH within
the same physics tick. `ScenarioEnv.step()`'s possession-transition scan
only sampled `match.ball.possessed_by` once per tick (after the tick fully
resolved), so this real, if momentary, touch was completely invisible --
not "cancelled via pending_loss" like a genuine same-tick regrab, just
never observed at all (`possessed_by` read `None` both before and after
that tick). Fixed by detecting the armed auto-fire via the SAME
`kicked_this_tick and kick_armed` combination `trainee_armed_kicks_this_step`/
`sec_armed_kicks_this_step` already check (`kick_armed` is never touched by
`_finish_kick`, so it still reads True at this point iff this tick's kick
came from the armed path -- never true for an already-possessing instant
kick, since `apply_nn_action.py`'s kick branch is exclusive) and feeding a
synthetic `possessed_by=this player` sample through
`_possession_transition_step` FIRST (registering the gain) before the real
end-of-tick sample (registering the subsequent loss/pending-loss) --
`_possession_transition_step` itself is unchanged, this just calls it one
extra time for the tick a touch happened. See
`tests/ai_unit/test_possession_transition_step.py`'s
`TestArmedKickAutoFireWithinOneTick`.

**Augmentation log_prob approximation (fixed -- was a real, uncontrolled
noise source)**: `ai/obs/augment.py`'s `augment_batch()` tiles `log_probs`/
`head_log_probs` from the original (s, a) pair onto every augmented copy,
unchanged. Exact for slot-permutation copies (permutation invariance means
`log π(perm(a)|perm(s)) == log π(a|s)` identically), but only an
APPROXIMATION for flip_y copies: the tiled value equals the true
`π_old(flip(a)|flip(s))` only once the network has already learned exact
flip-equivariance, and is off by a real, uncontrolled ratio-scaling factor
otherwise -- confirmed via the epoch-0 baseline diagnostic
(`log_epoch_zero_baseline`): the augmented TRAIN batch's own
`[policy percentiles]` line read p100 in the 20s at ratio=1 (no gradient
step yet this update), while a same-rollout, never-augmented held-out
batch read single digits.

Initially defended (in this file and in `augment.py`'s own docstring) as a
deliberate "provides a gradient signal toward equivariance" mechanism --
on closer, more rigorous derivation this doesn't hold up. The gradient
DIRECTION an augmented row contributes is `d(new_log_prob)/dtheta`, which
depends only on the current weights and the (already-correctly-flipped)
input `(s', a')` -- NOT on which `old_log_prob` constant is subtracted
from it in the ratio's exponent (that constant doesn't depend on theta
either way). Only the MAGNITUDE differs between the tiled (borrowed) value
and the true one. Whatever teaches the network the y-flip symmetry comes
from training on the correctly-flipped `(s', a', A)` sample at all, with
its own correctly-signed advantage -- both the tiled and the proper
version do this identically; the tiled version just also multiplies the
update by an uncontrolled, non-equivariance-dependent factor on top,
which is noise, not signal.

Fixed via `PPOTrainer._recompute_old_log_probs_for_augmented_batch()`,
called from `_ppo_update()` immediately after `augment_batch()` (and after
the physics-encoder full-feature precompute, which it depends on) --
safe specifically because the network is still exactly theta_old at that
point in the call, before any gradient step. Reuses the SAME
`_recompute_log_prob`/`_per_head_new_log_probs` helpers the real
per-minibatch training loop and `_eval_val_episode_losses` already use, so
no new masking/formula logic was introduced -- confirmed consistent with
the ORIGINAL rollout-time log_prob via the held-out epoch-0 baseline
already reading an exact 0.0000 (proving `_recompute_log_prob`'s formula
already matches however the original scalar was stored, for the
never-augmented held-out rows). Only meaningful when
`augment_n_slot_shuffles > 0`; recomputes ALL rows (identity copies
included) rather than special-casing just the flip_y half, since the
identity copies' recomputed value is provably identical to what they
already had.

### Player event callbacks

`Player` has two optional callbacks set on the instance:
```python
player.on_kick    = lambda player: ...   # fired when kick_direct() executes kick physics
player.on_tackle  = lambda player: ...   # fired when ChaseTackleOrder makes contact
```
The engine fires these in `match.py`/`player.py` at the exact tick the action
executes — not when the order is set.  Useful for: BC recording, UI effects,
logging, statistics.  Both default to `None` (no-op).

`on_kick` fires from **any** code path that calls `Player.kick_direct()` or
`Player.kick_with_direction()` — `KickOrder`/`ShootOrder`/`PassOrder.execute()`
delegate to the former, and push-kicks (`MoveOrder`/`GetPossessionOrder` via
`_try_push_kick()` in `orders.py` -- a flat, direction-only kick, no
ballistic solve) and the neural network's direct-drive kick action delegate
to the latter. Alongside the optional callback, both unconditionally set
`player.kicked_this_tick = True` (reset to `False` for every player at the
top of `Match._process_orders()`) — this flag exists specifically so code
that runs *after* order processing (e.g. `bc.py`'s `phase1_labels()`) can
check "did this player kick this tick" without needing `on_kick` wired up
and without inspecting order types (which missed the `MoveOrder` push-kick
case — see the BC label table above).

### Demonstration recording (`record_demonstrations.py`)

**!!!! KNOWN LIMITATION, INTENTIONAL FOR NOW: the secondary opponent is
NEVER neural during demo recording !!!!** `record_episodes()`'s own
opponent-type roll (lines ~571-588, a SEPARATE roll from -- and
unconditionally overriding -- whatever `build_1v1_scenario` itself already
decided) only ever assigns `Phase1RulesAI()` or leaves the opponent
immobile; the `opponent_rules_prob`/`opponent_immobile_prob` args (and the
`--opponent-rules-prob`/`--opponent-immobile-prob` CLI flags,
`phase1_demo_opponent_*_ratio` in ai_config.json) sum to less than 1.0 on
purpose -- the REMAINDER probability mass (what would be "neural") is
deliberately folded into "rules" instead, not left as a real option. Only
the TRAINEE can be neural-driven during recording (`--driver-checkpoint`,
via `driver_trainer`). This means recorded BC datasets never contain a
sample of "what does the opponent's own state distribution look like when
IT is neural" -- relevant if a future neural-secondary-opponent PPO
self-play run (see the "secondary opponent audit" work) ever wants matching
BC-pretraining coverage for that opponent policy too. Also flagged (more
tersely) in `ai_config.json`'s `_comment_phase1_opponent` and inline at the
roll logic itself in `record_demonstrations.py`.

Sampling strategy:
- `env.step()` is called once per real decision interval (the env's own
  `observation.decision_interval_s`, same cadence real training uses — see
  "Timed-sample cadence" below) — episodes terminate correctly
  (box-possession terminal, timeout) because `ScenarioEnv.step()` handles
  those checks.
- Inside each `env.step()` call, the engine fires `player.on_kick` /
  `player.on_tackle` callbacks at the exact physics tick the action executes.
  These callbacks record an extra (obs, label) sample immediately.
- Net result: one regular sample per decision (or every Nth decision, see
  `sample_every_n_decisions`) + one extra sample per kick/tackle event.
- Reward wiring: kick/tackle callback samples used to hardcode `reward=0.0`,
  silently dropping real reward (e.g. `gain_possession_bonus`) that fired on
  exactly that tick. Fixed via a per-player `_pending_reward: dict[str, float]`
  (keyed by player id, not a single shared cell — see "Per-player rewards"
  below for why that distinction matters): `_record_now(reward=None, ...)`
  (the callback default) consumes and clears `_pending_reward[pid]` for that
  ONE player only; the main loop accrues each player's own reward into their
  own dict entry after every `env.step()`, so reward is never double-counted
  or dropped regardless of when a kick/tackle callback fires relative to a
  timed sample.
- Periodic logging (every 10 episodes, or at the final episode) now also
  prints a full reward-component breakdown line (`REWARD_COMP_LABELS` from
  `ppo_trainer.py`, accumulated from `env.last_reward_components` after every
  `env.step()` and reset after each log line) — mirrors `train.py`'s
  pre-training reward diagnostic (`_comp_acc` pattern) so demo-recording
  reward shaping can be sanity-checked the same way.

**Timed-sample cadence is decision-count based, not time based**
(`sample_every_n_decisions`, `bc.demo_sample_every_n_decisions` in
ai_config.json): used to be a sim-seconds interval (`sample_interval_s`,
default 0.2s in code / 0.5s in config -- two different stale defaults, never
actually equal to each other or to `observation.decision_interval_s`
=0.239s) that OVERRODE `env._ticks_per_decision` to force `env.step()` to
advance by exactly that many seconds. That wasn't just a sampling-density
choice — `Phase1RulesAI`/`NeuralPlayerAI` re-evaluate their order/action
once per decision (once per `env.step()` call), so forcing a different
decision cadence during recording than real training/gameplay ever uses
means the recorded trajectories themselves were shaped by a different
policy-update rate, not just logged at a different density. Fixed:
recording no longer touches `_ticks_per_decision` at all — `env.step()`
always advances by the env's own real `decision_interval_s`, identical to
real training — and `sample_every_n_decisions` (default 1 = record every
decision) just controls how many of those genuine `env.step()` calls get a
timed sample (`_do_timed_sample = decision_count % sample_every_n_decisions
== 0`, reset to 0 at the start of each episode). on_kick/on_tackle callback
rows are unaffected either way (always recorded, per-player, regardless of
whether that decision was sampled) — matches the existing "kicks and tackles
always recorded regardless" behavior. `is_decision_step`/`is_trainee`/dones
bookkeeping is unchanged; a skipped (non-sampled) decision simply appends
zero timed-sample rows for that iteration (the reward/component backfill is
a `zip()` over two empty lists, a natural no-op), so at `N>1` that
decision's reward is not captured anywhere — an intentional tradeoff of
thinning the recorded density, not a bug.

**Neural driver/teacher** (`--driver-checkpoint`/`--teacher-checkpoint`): the
trainee can be driven by a loaded checkpoint's own policy instead of
`Phase1RulesAI` (`--driver-checkpoint`, wired via `env.sample_action_fn` +
`ScenarioEnv.reset()`'s existing `NeuralPlayerAI` auto-assignment — no new
mechanism needed), and/or BC labels can be sourced from a neural teacher's own
forward pass instead of the rules-AI order-simulation counterfactual
(`--teacher-checkpoint`, via `bc.phase1_labels_from_teacher()`). The two are
orthogonal: driver picks which states get visited, teacher picks what
supervises them — DAgger-style dataset generation, not circular self-imitation,
since the teacher is a genuinely separate forward pass, decoupled from
whichever AI is physically driving the player that tick.

**Per-player rewards, NOT a shared/duplicated value**: `_record_now(player_id=
None)` (used for every timed sample) records BOTH the trainee's row and the
opponent's row. It used to give both rows the exact SAME reward value (the
trainee's own, from `env.step()`'s scalar return) — silently mislabeling the
opponent's row with someone else's reward AND double-counting that value in
every downstream MC return / episode-total computation. Fixed via
`ScenarioEnv.always_compute_secondary_reward` (opt-in, off by default so real
PPO training is unaffected — see its docstring in `scenario_env.py`): when
set, `env.last_secondary_results` gets a real entry for a secondary player
EVERY step, computed through the exact same `_compute_phase1_reward_for_player()`
call already used for the trainee and for a neural secondary player during
real PPO training — regardless of whether that secondary player is actually
neural-driven. `record_demonstrations.py` sets this flag unconditionally and
reads each row's own reward from it, instead of reusing the trainee's value.

**`compute_returns()`/`compute_component_returns()` are player-track-aware**:
trainee and opponent rows are INTERLEAVED within an episode's row range (one
timed sample appends the trainee's row then the opponent's row, back to
back), not laid out as two contiguous blocks. A flat backward MC scan over
the whole episode range would therefore mix the two players' reward streams
— e.g. the trainee's computed return would incorrectly include the
opponent's future rewards too. Fixed via a new per-row `is_trainee` field
(1.0/0.0, recorded by `record_demonstrations.py`, defaults to all-1.0 for
older files predating it) — `DemonstrationDataset.compute_returns()` now
maintains two independent running accumulators (trainee-track,
opponent-track), each row only updating/reading its own. A second, related
bug found and fixed at the same time: episodes end with 2 CONSECUTIVE
`done=1` rows (one per player, backfilled onto both terminal rows by
`_record_now(player_id=None)`) — naively resetting on every `done=1` row (a
bug that predates the per-track split; the original single-flat-scan version
had it too) makes the SECOND reset wipe out whichever track wasn't at that
exact row, silently discarding that player's terminal reward from every
earlier row's return. Fixed by collapsing consecutive `done=1` rows into a
single reset (`prev_had_done` tracking in both functions).

**!!!! Hardcoded to exactly 2 players (trainee + one opponent) — will need
extending once a demo dataset has more than 2 players !!!!** All of the
per-player-reward/`is_trainee` machinery above assumes exactly one trainee
track and one opponent track:
- `is_trainee` is a binary flag, not a player id — `compute_returns()`'s
  two-accumulator (`running_trainee`/`running_opponent`) design has no room
  for a third track.
- `record_demonstrations.py`'s `_record_now()` hardcodes
  `ids = [env.trainee_player_id, "opponent"]` — a third player's rows would
  need a real multi-id list here, sourced from `secondary_player_ids`
  (already a list on `ScenarioEnv`, so the *reward-computation* side
  — `env.always_compute_secondary_reward` — already generalizes to N
  secondary players for free; only the recording/dataset side is 2-player-
  specific).
- `reward_components` is genuinely per-player (same fix as `rewards` above,
  each secondary player's own breakdown comes from its own
  `last_secondary_results[i]["reward_components"]`), but the 2-track
  assumption in `compute_component_returns()`'s per-track scan is the same
  as `compute_returns()`'s — extending to N players needs the same
  dict-of-accumulators change, just applied to both functions together.
Until Phase 1 (1v1) is the only scenario with demo recording, this is fine —
but the FIRST 2v2+ (or N-secondary-player) demo-recording scenario must
revisit `is_trainee` (→ a real per-row player-id/role field), the two-
accumulator scan in `compute_returns()`/`compute_component_returns()` (→ a
dict of accumulators keyed by that field), and `_record_now()`'s hardcoded
`ids` list, together — not just one of them.

**Per-track GAE segmentation (`RolloutBuffer`, PPO side)**: the same
row-interleaving bug that `compute_returns()` above was fixed for on the BC
dataset side also existed, unfixed, on the PPO rollout-buffer side —
`RolloutBuffer.compute_gae()` used to do one flat backward scan over
`rewards`/`values`/`dones`, with `next_value = all_values[t+1]`. Since
`ppo_trainer.py`'s `train()`/`rollout_worker.py`'s `_collect()` push one
trainee row then immediately that tick's secondary-opponent row(s)
(`env.last_secondary_results`), a trainee step's "next" value in the flat
scan was actually the SAME-TICK secondary player's value, not the
trainee's own next-tick value — and vice versa. This was invisible for a
long time because `phase1_opponent_neural_ratio` has always been 0 (see
the "opponent audit" in `agent_plans/`), so `last_secondary_results` was
always empty and the buffer only ever held a single track in practice — it
would have activated the instant a neural secondary opponent (self-play)
was enabled.

Fixed via a per-row `track_id` field on `RolloutBuffer` (`"trainee"` or the
secondary player's id, e.g. `"opponent"` — defaults to `"trainee"` so every
existing single-track call site/test is unaffected). `compute_gae()` and
`compute_mc_returns()` both segment rows by `track_id` via
`_track_index_groups()`, run the backward recursion independently per
track (in that track's own chronological order — rows for a track are
always appended in order even though interleaved with other tracks in the
flat lists), then scatter results back into the flat, original-index
output arrays. `compute_gae()`'s `last_value` bootstrap argument now
accepts either a bare float (applied to every track — the common
single-track case) or a `{track_id: value}` dict; `PPOTrainer.
_bootstrap_last_values()` builds that dict once per rollout window (the
trainee's own next-obs value, already available from `env.step()`'s
return, plus — for every OTHER track actually present in the buffer — a
fresh `env._get_obs(player_id=track)` re-encode run through the same
critic). Both the single-process (`train()`) and worker (`rollout_worker.
py`'s `_collect()`) rollout-collection loops call this same trainer method,
so the fix is identical in both places. See `tests/ai_unit/test_gae.py`'s
`TestGAEMultiTrack`/`TestMCReturnsMultiTrack` (buffer-level, including a
regression check that the fixture actually would disagree with a flat
scan) and `tests/ai_scenario/test_secondary_opponent_gae.py` (full
`ScenarioEnv`+`PPOTrainer` pipeline with a real neural secondary opponent,
including one call through the actual `PPOTrainer.train()` entry point).

`compute_mc_returns()` (value-pretraining only) was given the identical
segmentation defensively even though it's never actually fed a multi-track
buffer today (value-pretrain rollout collection never records secondary
transitions) — kept consistent so it doesn't become the same landmine if
that changes later.

**"unknown" outcome bucket (`value_mse_by_outcome()`, PPO side)**: same root
cause SHAPE as the per-track bugs above (a kwarg threaded through the
trainee's `buffer.add()` call but silently omitted from the secondary-player
one), different field. `RolloutBuffer.add()`'s `step_outcome` param was only
ever passed at the trainee's own call site in `ppo_trainer.py`/
`rollout_worker.py`/`batched_rollout_worker.py` — every secondary-player
`buffer.add()` call silently took the `""` default, and
`env.last_secondary_results` never even carried an outcome field to source
it from.

`_backfill_step_outcomes()` (propagates a done=1 row's real outcome onto
every earlier row of that same episode) could not rescue these: it scans
the flat, `track_id`-unaware `dones`/`step_outcomes` lists, and only
backfills a done=1 row's OWN preceding span `if outcome:` at THAT row's own
slot. On a terminal tick, the trainee's row (added first, correctly
labeled) triggers a backfill that correctly recovers every earlier row of
the episode (both tracks, since they're interleaved before it) — but each
secondary player's OWN terminal-tick row, appended immediately after in the
same tick, independently also has `done=1.0` with an empty `step_outcome`
slot, so the backfill finds nothing to propagate and skips it. Every
secondary player's terminal-tick row therefore permanently leaked into
`value_mse_by_outcome()`'s fabricated `"unknown"` bucket — exactly the
"no legitimate unknown, all are bugs to fix at the source" case
`DemonstrationDataset.row_outcomes()` already documents for the BC side
(`ai/bc/dataset.py`), just not enforced here.

Fixed by threading the shared, env-level `info.trial_outcome` (one trial
ends the same way for every player on the pitch at once — never
player-specific) into `last_secondary_results["step_outcome"]`
(`scenario_env.py`), then passing `step_outcome=sec.get("step_outcome", "")`
at all 3 secondary `buffer.add()` call sites, mirroring the trainee's own.

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

### Frozen physics-dynamics encoders (`models/physics_encoders.py`)

The standalone-pretrained `ai/physics_pretrain/` encoders (`BallDynamicsEncoder`/
`PlayerDynamicsEncoder`, latent + auxiliary-head outputs concatenated —
matching `physics_value_net.py`'s `PhysicsEncoderValueNet.compute_features()`
diagnostic pattern) are wired into the main `DecisionNetwork`/
`ExecutionNetwork` as an **opt-in** feature — see
`agent_plans/ball_physics_pretrain_plan.md` §8 for the full design rationale
and rejected alternatives (§8.1: why the latent isn't just appended as new
`BallFeatures`/`PlayerFeatures` fields — checkpoint/dataset invalidation, and
an opaque latent has no defined sign under canonical-frame mirroring).

**Config**: `network.ball_physics_encoder_checkpoint`/
`network.player_physics_encoder_checkpoint` (`ai_config.json`, both `null`
by default = feature off, byte-identical behaviour to before this feature
existed). Set a path to a `physics_pretrain` checkpoint (e.g.
`checkpoints/physics_pretrain/ball_encoder_68.midtrain_latest.pt`) to enable.

**Computed once, inside `DecisionNetwork.forward()` ONLY** — never inside
`ExecutionNetwork`. `DecisionNetwork` is the sole owner/loader of
`BallPhysicsFeatureBlock`/`PlayerPhysicsFeatureBlock`
(`models/physics_encoders.py`); its `forward()` concatenates each frozen
block's output onto `ball_feat`/`self_feat`/`other_feat` before they reach
`ball_mlp`/`self_mlp`/`entity_encoder`, and stashes the raw (un-concatenated)
per-entity output on three new `DecisionHeadsRaw` passthrough fields
(`ball_physics_full`/`self_physics_full`/`other_physics_full`) — same "not a
real head, just data ferried to execution net" status as the existing
`latent_vector` field. `ExecutionNetwork.forward()` reads these straight off
the `decision_heads` argument it already receives at every call site in the
codebase (no plumbing changes needed there) and builds its own augmented
tensors via a cheap concat — it **never calls either frozen encoder**. This
means each encoder runs at most once per observation (not once per network),
and at most twice per decision tick in the current 1v1 curriculum (once per
team/canonical-frame — see "Canonical AI frame" above), not once per
player-observation-slot. See `tests/ai_unit/test_physics_encoder_wiring.py`
for the regression coverage (shape correctness, frozen-gradient guarantee,
`is_loose` masking, the compute-once-per-observation call-count property, and
the canonical-mirror-consistency proof).

**Masking**: the ball's combined block is multiplied by `BallFeatures.is_loose`
(zero when the ball is possessed — matches `physics_value_net.py`'s own
`ball_full * is_loose`). The player block is **never masked** — unlike the
ball, a player's dynamics latent is always meaningful regardless of
possession state; padded other-player slots get whatever the encoder
produces on all-zero input, which `exists_mask` already zeroes out
downstream like every other per-slot feature.

**Frozen, no-grad, excluded from the optimizer and from the main PPO
checkpoint**: `requires_grad_(False)` inside `load_frozen_ball_encoder`/
`load_frozen_player_encoder` (`ai/physics_pretrain/live_encoder_features.py`);
`ppo_trainer.py`'s `policy_params` construction explicitly filters out any
parameter whose name starts with `ball_physics_encoder.`/
`player_physics_encoder.`. `DecisionNetwork.state_dict()` is overridden to
drop those same keys before returning (delegated through automatically by
`CanonicalNetworkWrapper.state_dict()`, so every save call site in
`ppo_trainer.py` is covered with no per-call-site changes) — the frozen
encoder is an external, separately-versioned artifact referenced by
checkpoint path, never embedded training state. Correspondingly,
`PPOTrainer.load_checkpoint()` loads `decision_net` via the existing
`_load_state_dict_tolerant()` helper (same one `execution_net`/`value_net`
already used) rather than strict `load_state_dict()`, so the always-missing
physics-encoder keys don't raise — the live submodule already has correct
weights from its own checkpoint load, and tolerant loading simply leaves
them untouched.

**Sizing wrinkle**: `ExecutionNetwork` (including `separate_value_net`'s
standalone `value_net`, also an `ExecutionNetwork`) needs the same widened
`self_dim`/`ball_dim` as `DecisionNetwork` for its own `self_mlp`/`ball_mlp`
(when not shared via `network.share_entity_encoder`)/`entity_encoder`, even
though it never loads the encoder itself — `ExecutionNetwork.from_config()`
calls `peek_ball_physics_output_dim()`/`peek_player_physics_output_dim()`
(`live_encoder_features.py`) to read just the checkpoint's `latent_dim` for
sizing, without constructing a full feature block.

**Known scaling gap (not yet addressed)**: each frozen encoder is computed
once per *observation* (shared between that observation's decision+execution
calls), which only coincidentally bounds to "twice per tick" in the current
1v1 curriculum (at most 2 observations/tick). It does **not** yet achieve
true cross-observation caching — with N>2 simultaneously-deciding players,
the same physical entity's physics block gets redundantly recomputed once
per observation it appears in, rather than once per (entity, team-frame).
See `agent_plans/physics_encoder_cross_observation_caching_plan.md` for the
planned fix (a per-tick cache keyed by (entity_id, team), populated directly
from raw `Player`/`Ball` state since the physics-encoder inputs are entity-
intrinsic, never observer-relative).

### time_remaining_s is caller-managed

The engine only tracks `match.time_s` (elapsed time).  The env wrapper
(`ScenarioEnv`) subtracts from its `max_episode_s` budget and passes the
remainder to `encode_observation()`.  Time is log1p-normalized so the
"urgent endgame" scenarios (1–20s remaining, 10% of curriculum) are
distinguishable from "2 minutes remaining" after normalization - see design
doc section 7.5 for why plain linear /7200 fails here.

### Position normalization: pitch half-diagonal everywhere, NOT per-axis 52.5/34.0

`PlayerFeatures.pos_x/pos_y`, `BallFeatures.pos_x/pos_y`, and every `rel_dx`/
`rel_dy`/`ball_rel_dx`/`ball_rel_dy` field are all normalized by the pitch
**half-diagonal** (`sqrt(52.5²+34.0²) ≈ 62.66`) — the exact same divisor
velocity fields use — via `ai/obs/encoder.py`'s actual `pos_x=.../half_diag`.
There is NO separate per-axis `x/52.5`, `y/34.0` divisor anywhere in the real
encoding, even though `schema.py`'s docstrings claimed exactly that for a long
time (fixed now). Values still land ≈[-1, 1] on a standard pitch either way,
so this silently doesn't matter for training — it only bites when a
standalone script hand-reconstructs a real metre position from a raw feature
value for debugging/diagnostics. Using 52.5/34.0 there under-scales y by
~1.84x and x by ~1.19x, making positions look well within bounds when the
real position is actually at/past the boundary. This has caused real
confusion twice: once in `debug_value_network.py`'s match-log reconstruction
(fixed, see `_episode_rows_to_match_log`'s own comment), and again
independently in a standalone `diagnose_crossing_head.py` script that didn't
reuse that fixed helper (also fixed). If you're hand-computing a real-world
distance/position from a raw `PlayerFeatures`/`BallFeatures` value anywhere
new, use the half-diagonal, not 52.5/34.0.

### Heading and previous-decision movement intent (PlayerFeatures)

`PlayerFeatures` gained 7 fields (appended after `pos_y` specifically so
`ai/physics_pretrain/live_encoder_features.py`'s hardcoded `PF_*` offsets
stay valid — see that section of `ai_trainer_knowledge.md` for the full
dimension-bump note), encoded for **every player slot, self and others
alike** (matches how every other field is already symmetric):

- `heading_sin`/`heading_cos` — `sin`/`cos(player.heading_rad)`, populated
  **unconditionally**, including at standstill and for immobile players. This
  fixes a real gap: `engine/movement.py` constructs `player.velocity` FROM
  `heading_rad` every tick (`velocity = Vector3.from_angle_xy(new_heading,
  new_speed)`), so heading is only recoverable from velocity while `speed >
  0` — at `speed == 0`, velocity collapses to `(0,0,0)` but `heading_rad`
  keeps its last real value, which the schema previously (wrongly) dismissed
  as "irrelevant." A standstill player's facing direction is real signal,
  unlike velocity (which genuinely is noise for a never-moving player, hence
  still zeroed for `is_immobile`).
- `desired_dir_x`/`desired_dir_y` + a `desired_speed_standstill`/`_jog`/
  `_sprint` one-hot — the previous decision's movement intent, still in
  effect until the next decision tick overwrites it (a cheap
  acceleration/intent signal `ai/physics_pretrain` already relies on via BC
  labels — see `ai_trainer_knowledge.md` §6 — now also exposed as a live
  observation input). Direction is sourced from `player.desired_direction`
  (safe to read directly — never auto-cleared). The speed mode is sourced
  from **`player.last_desired_speed_mode`, NOT `player.desired_speed_mode`**:
  `engine/match.py::_apply_movement()` unconditionally resets
  `desired_speed_mode = None` for every player every tick right after
  consuming it, so by the time the next decision's `encode_observation()`
  runs it always reads back `None` — `last_desired_speed_mode` is a sibling
  `Player` field that mirrors it but is deliberately never cleared (set
  alongside the reset in `_apply_movement()`, see its own docstring on
  `Player`). Immobile players and a player with no decision yet both default
  to zero-direction/`STANDSTILL` — "no movement intent" is real signal here,
  same rationale as heading's `is_immobile` exception is the opposite of.

Mirror/flip rule (established first for `ai/ppo/bc.py`'s own
`heading_sin`/`heading_cos` BC label fields and `physics_value_net.py`'s
hand-patched `canonicalize_bc_labels()` call, now made systematic in
`obs/augment.py`'s `PLAYER_FLIP_X_IDX`/`PLAYER_FLIP_Y_IDX` — which
`obs/canonical.py`'s permanent x-mirror also reuses, see "Canonical AI frame"
above): **flip_y negates `heading_sin`/`desired_dir_y` only; the x-mirror
negates `heading_cos`/`desired_dir_x` only.** `desired_dir_x`/`desired_dir_y`
otherwise follow the exact same vector convention as `velocity_x`/
`velocity_y`.

### Last-touch team (`BallFeatures.last_touch_team_direction`)

`+1.0`/`-1.0` for whichever team most recently **gained** possession
(`ball.last_touched_by_player_id`, `entities/ball.py` — set only on a
genuine possession gain by `Match._set_possession()`, `engine/match.py:264-267`,
so it persists through loose-ball periods rather than resetting the instant
the ball becomes loose again), `0.0` until the first possession gain of the
episode. Same sign convention as `PlayerFeatures.attacking_direction` (`+1.0`
= that team attacks +x, i.e. `Team.LEFT`); negated under the canonical
x-mirror via `obs/augment.py`'s `BALL_FLIP_X_IDX` (a team-direction sign, not
a coordinate, so it's **not** in `BALL_FLIP_Y_IDX`).

Exists so the network can see directly who's responsible for an eventual
out-of-bounds/goal outcome, rather than only learning it indirectly through
reward-shaping after the fact — `match.py`'s own
`_run_get_possession_behaviour()` already derives this exact "did my own
team touch it last" fact internally (comparing
`player_by_id(last_toucher_id).team == player.team`) for boundary-braking
logic; this field exposes the same underlying fact to the observation
instead of leaving it as a rules-only/reward-only computation.

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

## Episode-seed replay's advantage ranking now includes self-play secondary rows

`_episode_abs_adv_means()` (`ai/ppo/ppo_trainer.py`, formerly
`_trainee_episode_abs_adv_means`) computes the per-episode mean(|advantage|)
used to pick which episode seeds get re-queued for the next rollout
(`episode_replay_enabled`, PLR-style). It used to filter to
`track_ids == "trainee"` rows only. Extended to include interleaved
secondary-track rows too, because a non-"trainee" row in the buffer is —
today — *always* the current, live network playing itself:
`_batched_worker_main` passes `secondary_trainer=trainer` (identical
weights), and `_secondary_neural_candidates()` excludes rules-based/immobile
opponents entirely (they never produce a buffer row at all). So a
rules/immobile episode still reduces to trainee-only automatically; a
self-play episode's secondary rows are real signal from the same network
being trained and were previously being silently dropped from the ranking.

The tricky part: a secondary row's own `done` mirrors the trainee's shared,
env-level `done` for that tick exactly, and the trainee row for a tick is
always added to the buffer *before* that tick's secondary row(s)
(`buffer.add()` ordering in both `_batched_worker_main` and the
single-process `_collect()`). That means the terminal tick of an episode
looks like `[trainee(done=1), secondary(done=1), ...]` — segmenting by
"close the segment the instant a trainee `done=1` row is seen" would wrongly
push that tick's own secondary rows into the *next* episode. Fixed by
deferring the close until the *next* trainee row is seen instead, so a
terminal tick's trailing secondary rows get pulled into the segment being
closed. See the function's docstring for the full trace-through and the
explicit caution: if a frozen/older-checkpoint opponent is ever added, the
"any secondary row implies current network" invariant this leans on breaks,
and an explicit per-row "is this the live network" signal would be needed
before secondary rows can keep being included unconditionally.

## flip_y consistency (measured 2026-09-20): augmentation, consistency refit, y-canonical frame

**Finding (run283, 68 PPO iterations from a PPG-refit checkpoint).** Eval vs rules slid from about checkpoint 18 onward
while training win rate stayed flat and value loss stayed good. Same pattern in run265 (99 iterations, different reward).
The `[ratio spike]` line (max ratio 1e7 growing to 1e15 over the run) is `move_dir` in 65 of 66 logs; the rows are the
**flip_y augmentation copies**, not ordinary samples: `_ppo_update` recomputes each copy's "old" log-prob with the
pre-update weights (`_recompute_old_log_probs_for_augmented_batch`), and the network is far from flip_y-equivariant.
Measured on 30k real states (mirror of the state vs mirror of the output): median `move_dir` mean-angle error 22 deg at the
PPO start (8% of states off by more than 90 deg) growing to 39 deg / 24% by checkpoint 68; hard-decision (sign) disagreement
between a state and its mirror 9-23% for exec_move / sprint / tackle_attempt; Bernoulli logits saturated (median |logit| 23 for
exec_move) with mirror log-prob tails down to -394. About 27% of active `move_dir` rows had a flip-copy log-prob below -15
(a real sample at kappa~14 essentially never goes below -14; verified on 10M PyTorch VonMises draws). Correlational only: PPO
weight steps are nearly uncorrelated across iterations (cosine ~0), the step size is constant (Adam) while KL per update rose
5x, clipping went from 25% to 82% of steps, and eval declined once KL per update passed ~0.1. Whether the augmentation *causes*
the decline was not isolated by an intervention.

**Also measured:** `val_pre` (pre-update value loss) is inflated from iteration 2 on by `episode_replay` (top-20% |advantage|
seeds re-queued): replayed episodes 0.305 vs fresh 0.175 under the same checkpoint. Not value degradation.

**Fix 1 -- `PPOTrainer.consistency_refit` (`train.py --consistency-refit-only`, `bc.consistency_*`).** A PPG-style phase: every
state is used in both orientations, the orientation where the observer's y >= 0 is "primary", a frozen teacher (the un-reset
`--checkpoint`, or `--consistency-teacher`) is evaluated on it, and the student is trained with the existing analytic per-head
KL (`_ppg_kl_penalty`) toward the teacher's output (mirrored for the mirror orientation). Spread params (kappa etc.), the
value head and the physics encoders are frozen for the call (a trainable kappa would let the KL be lowered by widening
distributions); Bernoulli targets are clamped to +-`consistency_logit_bound`; kappa is set from `consistency_kappa_deg`
(22 deg = 6.78). Helpers are in `ai/ppo/consistency.py`. Result from checkpoint 18 (Bernoulli heads reset with
`--reset-bernoullis` scales fitted so 97% of states fall within +-6.9, kappa 22 deg): mirror error p50/p90 25.0/98.0 deg ->
6.3/20.3 deg, states over 90 deg 11.8% -> 0.5%, hard-decision disagreement (exec_move/sprint/kick/tackle) 10.4/15.4/1.6/16.2%
-> about 2.2/3.5/0.2/5.2%, eval vs rules unchanged within noise (38.4% -> 40.9% win). Validation KL plateaued around 0.47
after ~110 rollouts at batch 8000, 6 epochs (later epochs within a rollout only overfit that rollout; gains come from fresh
data). Caveats: the KL is nearly blind to 0.001 vs 0.0001 probability, so Bernoulli logits re-saturate after the reset (73% of
exec_move states beyond +-6.9) -- bounding needs a separate term or another reset; the loss is heavy-tailed (top 5% of rows carry
~48% of the KL; mirror rows fit worse than primary rows, 1.07 vs 0.48 at run284).

**Fix 2 -- `ppo.y_canonical` (`ai/obs/y_canonical.py`, opt-in).** A second wrapper outside `CanonicalNetworkWrapper` mirrors
every row whose observer is at y < 0 before the network and mirrors the y-signed outputs back, so the policy is *exactly*
flip_y-equivariant (unit-tested on the real networks, with and without the frozen physics encoders). Consequences: flip_y
augmentation copies become exact duplicates -> set `ppo.augment_n_slot_shuffles` to 0 (a warning is logged otherwise; with
n_slot_shuffles=1 the flip is the only thing augmentation does today). `_precompute_physics_full` mirrors first so the cached
encoder outputs match what the wrapper feeds the network. State-dict keys are unchanged. Cost: the policy is discontinuous
across y = 0 wherever the underlying network is asymmetric. Measured: wrapping did not change eval vs rules within noise
(run290 checkpoint 40.9% -> 39.9% win; original checkpoint 18 38.4% -> 41.0%), so the original asymmetry was not doing useful work.
Old snapshot opponents in the neural pool are y-symmetrised too when it is on, so evals vs neural snapshots are not comparable
with earlier runs.

**Diagnostics.** `ppo.log_flip_consistency` adds a `flip` line to every `[PPO]` block (KL between the policy on each rollout
state and the y-mirror of the policy on its flip_y copy, `move_dir` mirror error, hard-decision disagreement), computed from the
pass `_ppo_update` already makes over the augmented batch; needs augmentation on (it reads zero under `y_canonical`).

**Mirror-symmetry tests (`tests/ai_unit/test_mirrored_match.py`, `test_mirrored_match_simulation.py`).** A random real `Match` and its
true mirror (x = team swap, y, both) are encoded with the real `encode_observation` and driven through the wrapped networks with the frozen
physics encoders enabled: every canonical input feature, encoder output, latent vector, value, head and chosen action is identical;
removing any single feature from the flip index lists is caught (17 mutations checked). Simulated forward, the state of a match and its
mirror agree to rounding (~1e-14): neural-driven 300 ticks, rules-AI-driven (dribbling, kicking, tackling, 6 possession changes over 3
matches) 200 ticks -- so the rules-AI opponents are mirror-symmetric too. Two limits, both engine-side and not mirror-specific:
(1) rounding noise (~1e-15) is amplified exponentially by the rules AI's pursuit dynamics (x1e6 in ~65 ticks), so exact agreement lasts
~230 ticks for that driver; (2) **contact knife-edge**: `resolve_all_overlaps` pushes a colliding pair to exactly the touching distance,
after which the second push iteration and `_damp_overlap_velocity` test `distance >= min_distance` on numbers within 1 ulp of it, so the
last rounding bit decides whether velocity damping applies (measured +2.2e-16 in one match, -1.1e-16 in its mirror) and a 1e-16 difference
becomes a 2-6 m/s velocity difference after the first contact between opposing players (`xfail(strict)` test documents it; the other
long-horizon tests switch contact damping off with `CollisionParams(retention=1.0)`). Damping therefore applies to roughly half of
contacts at random; making it depend on the pre-push overlap (or an epsilon) would be a behaviour change and has not been done.

## Neural action application: tackle arming and kick one-shot (measured 2026-09-20 on run293)

`NeuralPlayerAI` re-applies its cached decision every tick of the decision interval (`prepare()` -> `_reapply_cached_gating`),
and `player.tackle_armed`/`kick_armed` are reset every engine tick, so the flags read at the interval boundary by the reward
(`ScenarioEnv._compute_phase1_reward_for_player`) reflect only the LAST tick. A probe (single process, checkpoint 57, 25.6k trainee
decisions vs the rules AI, `scratchpad/arm_probe.py`) found two problems, both now fixed behind opt-in-style flags in
`ai_config.json["action"]` (both set true):

- **Tackle (`arm_tackle_without_carrier`)**: `apply_action_to_player` used to reject a tackle attempt with no opposing carrier as
  illegal and NOT arm it. 96% of sampled tackles were such free no-ops (the head fired 37% of the time while the trainee itself held
  the ball, 26% with nobody holding it, only 3.4% while the opponent did -- inverted, no gradient in the no-op states) and the
  `tackle_armed_while_possessing_multiplier` could never fire (0/3508). Now every sampled attempt arms (unless the player is knocked
  down) and is charged; `Match._check_armed_tackles` already resolves only against an opposing carrier in range, so a pre-armed
  player still runs into a tackle when the ball arrives, and `tackle_attempted_bonus` still pays only on real contact.
- **Kick (`kick_one_shot`)**: the cached kick was re-applied every tick, so an in-possession kick released the ball and then
  re-armed/re-fired: ~3.1 physical kicks per decision (72% >= 2, 50% via the armed auto-fire) and `kick_armed` true at the boundary
  in 96% of possession kicks (the armed-kick cost was charged to ordinary kicks). `NeuralPlayerAI` now drops the cached kick once
  `Player.kick_count` (monotonic, bumped in `_finish_kick`, snapshot/restored in the BC label path) exceeds its count at decision
  time. A fired ARMED kick still pays `karm` (boundary flag OR `kick_attempt_was_armed_this_step`) and earns `katt`; an in-possession
  kick pays neither, as `reward.py`'s docstring always intended. Untested hypothesis: the kick chain (not the small armed penalty,
  ~-0.1 per decision) is why the kick gate collapsed to ~0.15% during run293 -- compare kick_prob in the next run.
- Both flags only take effect in newly started processes (config is read once per process); comparable runs need the same setting.
- **Kicks per episode ([PPO] `kicks` line)**: `StepInfo.trainee_kicks_this_step` / `trainee_armed_kicks_this_step` (per-physics-tick scan of
  `kicked_this_tick`, armed = `kick_armed` still set when it fired = first-touch kicks) are summed per episode in
  `BatchedEnvGroup.collect`, `rollout_worker` and the single-process loop (stats keys `episode_kick_counts`,
  `episode_armed_kick_counts`) and rendered by `_log_rollout_summary`. It counts every real trainee kick, unlike `kick_armed_penalty`
  steps or the `kick_attempted_bonus` count. Test: `tests/ai_scenario/test_episode_kick_counts.py` (exact match vs an independent
  `Player._finish_kick` counter).
- **Possession-conditional kick gate (run302 start, 2026-09-21)**: a probe of run301 showed the trained gate was INVERTED (mean p 28%
  without the ball vs 0.9% with it: 97% of kick decisions fired with no ball). `checkpoint3_posgate.pt` replaces
  `execution_net.kick_logit` by a least-squares read-out of the final trunk layer onto logit(1e-4) (no ball) / logit(1e-2) (ball),
  median-calibrated on train rows, 20% held-out: no ball mean 0.014% (p99 0.08%), ball mean 1.08% (p95 1.8%, p99 2.6%). Only
  `kick_logit.weight/bias` changed; Adam moments for them are stale (old regime). `scratchpad/possession_gate_surgery.py`.
  That first version armed almost never without the ball (run302 rollout 1: 31 `karm` steps, 1 `katt`; PPO also halved the with-ball gate in
  2 rollouts), so no signal reached the first-touch path. **Second version (`checkpoint3_posgate2.pt`, run303)**: an imminent first touch IS
  decodable from the final trunk layer (random arming converts only ~4% of armed decisions to a fired armed kick; a linear read-out reaches
  ~90% precision at ~1k armed attempts/rollout, held-out AUC ~0.97 in the pilot). Fit: force-arm 50% of no-ball decisions, label each armed
  decision by `StepInfo.trainee_armed_kicks_this_step`, then optimise ONE linear head on `E[p(y-mu)]` (mu = price per armed attempt, 0.88) with a
  hinge keeping with-ball gate in [0.3%, 3%] (mean 1.06%). Pitfalls hit on the way: (1) a point target for ball rows (1%) collapses precision
  to ~5% because "imminent touch" and "has the ball" overlap in trunk space -- use a band; (2) never re-shift the bias to hit the attempt
  budget after fitting (drags ball rows to ~0) -- control attempts via mu; (3) high mu from a cold start sticks at the p~0 optimum -- warm-start
  the mu sweep upward; (4) `_collect_value_pretrain_rollout` records no `reward_comps` and drops the trailing incomplete episode -- take
  labels from the env `StepInfo` and truncate them to the batch length. Real-play check (24 workers, 287k trainee rows, no forcing): 387 armed
  attempts / 358 fired (92.5%) vs held-out prediction 979 / 906 per 750k rollout. Scripts: `scratchpad/gate_label_collect.py`, `gate_opt.py`, `apply_gate.py`.

## PPG-style value refit (`PPOTrainer.ppg_value_refit`)

`pretrain_value()`'s shared-trunk mode (`separate_value_net=False`) only
offered two extremes: `value_pretrain_frozen_layers=-1` (default) freezes
encoders + full trunk, leaving only `execution_net.value_head` trainable (a
weak fit — the head can't reshape any upstream feature for value
prediction); `=0` unfreezes everything, so the value gradient reshapes the
exact trunk/encoder weights the policy heads read from, damaging the
policy. `ppg_value_refit()` (`ai/ppo/ppo_trainer.py`) is the middle ground:
full, unfrozen gradient flow through the trunk for the value loss, while an
analytic per-head KL-divergence penalty anchors every policy head's output
distribution to a snapshot taken right before the refit starts. Inspired by
the auxiliary phase of OpenAI's Phasic Policy Gradient (PPG) paper, but
scoped as a standalone/occasional refit — opt-in via `bc.ppg_enabled` (runs
once, automatically, right after whatever pretraining path `train.py` took,
before `checkpoint_pretrained.pt` is saved) or on-demand via `train.py
--ppg-refit-only` against any existing checkpoint — NOT an automatic
alternating cadence inside the main PPO rollout loop (that's a documented,
deliberately out-of-scope future extension, not built here).

Key implementation points, in case any of this needs revisiting:

- **The anchor is a snapshot of raw distribution PARAMETERS, not a cloned
  network.** One `torch.no_grad()` forward pass (`_ppg_snapshot_anchor`,
  minibatch-chunked to bound memory) over the whole (post-augmentation)
  refit batch, under the CURRENT weights, right before the epoch loop
  starts — captures only the state-dependent outputs each head's
  distribution needs (Bernoulli/Categorical logits, and the raw mean vector
  for move_dir/kick_dir/kick_power/kick_spin). This is much cheaper than
  keeping a second live copy of `decision_net`/`execution_net` around.
- **The continuous heads' spread parameters (`move_dir_log_kappa`,
  `kick_dir_log_kappa`, `kick_dir_z_log_std`, `kick_power_log_std`,
  `kick_spin_log_std`) are GLOBAL `nn.Parameter`s, not state-dependent**
  (confirmed directly in `execution_network.py`: each is
  `nn.Parameter(torch.full((1,), ...))`, or `(3,)` for kick_spin) — so the
  anchor only needs a one-time `.detach().clone()` of each, not a per-row
  tensor. This mirrors what `[exec continuous log_kappa]`-style log lines
  have shown all along (always a single-element list).
- **`kick_power`'s KL is computed via the raw (pre-squash) `Normal`, reusing
  `_kick_power_head(...).dist` for both current and anchor.** KL divergence
  is invariant under any invertible transform applied identically to both
  sides, so this is *exact* for the true squashed distributions — and it
  completely sidesteps the sigmoid-Jacobian log-density term responsible
  for an earlier session's `[ratio spike]` blow-up (see the checkpoint-
  surgery/exploding-surrogate incident earlier in this file's history). A
  genuine free correctness win from reusing the existing head constructors
  instead of hand-deriving anything for this head.
- **`torch.distributions` has no `kl_divergence` registered for
  `VonMises`** (confirmed: raises `NotImplementedError`) — move_dir/kick_dir
  needed a hand-derived closed form, `_von_mises_kl()`
  (`ai/action/distributions.py`, next to `_von_mises_entropy`), using the
  same numerically-stable `i0e`/`i1e` (`torch.special`) trick to stay
  well-conditioned at large kappa. `VonMisesDirectionHead`/`KickDirectionHead`
  gained small public `mean_angle`/`kappa`/`theta_mean`/`mean_z`/`std_z`
  properties so this (and any future caller) doesn't need to reach into
  underscore-prefixed internals.
- **Gating mirrors `_recompute_log_prob` exactly**, not `_compute_entropy`'s
  soft `E[parent]` weighting: hard boolean row-subset for the
  pass_target/tackle_target/mark_target categorical KL (folded into their
  parent Bernoulli's breakdown entry, matching how `_recompute_log_prob`
  never surfaces them as a separate scalar either), float exec_move/kick
  masks for their respective sub-heads. Heads in `_inactive_head_lp_keys()`
  (curriculum-frozen decision heads, or permanently-frozen kick_spin) are
  skipped entirely, exactly like the `[per-head KL]` diagnostic.
- **Fresh, throwaway Adam optimizer, not `self.optimizer`** — matches
  `pretrain_value`'s own convention (its shared-trunk branch also builds a
  fresh Adam) rather than mixing this objective's gradient statistics into
  the live PPO Adam state. Grad-norm clipping still splits direction-head
  params into their own `clip_grad_norm_` call via the existing
  `self.direction_param_ids` (computed once in `__init__`) — these heads get
  REAL gradient here (unlike `pretrain_value`'s frozen-trunk case), so
  without the split a single large direction-head KL gradient could force a
  proportional shrink of the value gradient in the same step, the exact
  failure mode `direction_max_grad_norm` already exists to prevent in
  `_ppo_update`.
- Best-val restoration must snapshot/restore the *full* `decision_net`+
  `execution_net` state dicts (not just `value_head`, since the whole
  network is trainable here) — a larger restore scope than
  `pretrain_value`'s own. It must also use `_load_state_dict_tolerant`, NOT
  a plain `load_state_dict` — `DecisionNetwork.state_dict()` is itself
  overridden to deliberately EXCLUDE `ball_physics_encoder.*`/
  `player_physics_encoder.*` (an external, versioned artifact, never part
  of the main checkpoint's training state — see that override's own
  docstring, which literally names the tolerant loader as the fix), so a
  strict load of a snapshot taken via that same `state_dict()` raises
  "missing keys" for those params. First hit as a real crash mid-run.
- **GAE returns, not MC** — `_collect_value_pretrain_rollout` gained an
  opt-in `use_gae` param (default False, `pretrain_value`'s exact prior
  behavior unchanged) that `ppg_value_refit` passes `True`. This was a real
  design correction, not the original plan: MC returns were initially
  copied straight from `pretrain_value`'s own reasoning ("bootstrapping off
  an untrained value net is circular"), but that reasoning is specific to
  `pretrain_value`'s cold-start-from-BC scenario, where the value head has
  genuinely never seen a return. `ppg_value_refit`'s actual target scenario
  — refitting an ALREADY-reasonably-trained value function — isn't cold
  start at all; bootstrapping off an imperfect-but-real value estimate is
  just ordinary TD learning, exactly what real PPO does every rollout. And
  it matters concretely: real PPO's own value loss targets GAE returns, not
  MC returns, so the actual point of a PPG-style refit (converging the
  value head toward the SAME quantity real training already uses, so
  resuming training needs no further readjustment) requires GAE, not MC.
  Mechanically cheap to add: `compute_gae`'s `last_value` bootstrap
  argument is passed as a literal `0.0` and is PROVABLY never used, since
  `truncate_to_last_episode_end()` (already called on both collection
  branches) always leaves the buffer ending exactly on a `done=1` row, and
  `compute_gae`'s backward recursion resets to 0 at every `done=1` — no
  extra "final value" forward pass needed.
- **`num_rollouts`** (default 1): repeats the whole collect→anchor→fit→
  restore cycle that many times in ONE `ppg_value_refit()` call, each cycle
  fully independent (own rollout, split, anchor, early-stop/best-val
  tracking) except the Adam optimizer, which is built ONCE before the loop
  and deliberately persists its momentum across cycles within one call —
  resetting it every cycle would be pure waste, since cycles within a call
  are meant to behave like one continuous session, just periodically
  re-grounded against fresh on-policy data and a fresh anchor. Still not
  the same thing as an automatic in-loop cadence — never touches the main
  `ppo.*` training loop. A checkpoint is saved to
  `checkpoint_dir/checkpoint_pretrained.pt` after EVERY cycle (not just at
  the end) when `self.checkpoint_dir` is set, since a multi-rollout call at
  real `ppg_rollout_steps` sizes can run a long time — bounds how much work
  a crash/interrupt partway through loses.
- **Bernoulli logit clamp (`±12.0`, inside `_ppg_kl_penalty`'s `_bern_kl`)**
  — `torch.distributions.kl_divergence` for two `Bernoulli`s is genuinely
  `+inf` (not a bug) whenever the ANCHOR side's probability has rounded to
  exactly 0.0/1.0 in float32 (any logit past ~±16.7) while the current side
  hasn't — very plausible on a mature, confident checkpoint (e.g. `move`/
  `tackle_attempt` after 500k+ PPO steps). Observed for real: `move`/
  `gp_extra`/`exec_move`/`tackle_attempt` all showed `+inf`, and `sprint`
  showed `nan` specifically because `exec_move_mask=0.0` on non-firing rows
  multiplies that `+inf` (IEEE `0*inf=nan`), poisoning the whole minibatch
  `.mean()`. Fixed the same way this file already handles analogous
  boundary cases for other heads (`SquashedNormalHead`/`VonMisesDirectionHead`
  already clamp `log_std`/`log_kappa` for the same class of reason) — only
  affects this KL penalty term, not the real sampled action/log_prob/
  entropy used anywhere else. The Categorical target heads
  (pass_target/tackle_target/mark_target) share the same architectural risk
  in principle but are UNVERIFIED — not currently protected, and not yet
  observed failing only because they're curriculum-frozen (hence excluded
  via `_inactive_head_lp_keys()`) in the phase this was debugged under.
- **`kl_train`/`kl_val` reported separately** — a second (no-grad,
  diagnostic-only; val is never trained on) forward pass over the held-out
  val set each epoch, against its own `val_anchor` snapshot taken at the
  same time as the train anchor, via `_eval_value_and_kl`.
- **Batched value-pretrain rollout (`ppo.value_pretrain_batched_rollout`,
  default off).** `_collect_value_pretrain_rollout` (shared by
  `pretrain_value`, `pretrain_combined`'s Phase 2/3 and `ppg_value_refit`)
  used to always spawn plain one-env-per-process `rollout_worker.py`
  workers (batch-of-1 network calls, e.g. 20 envs total) while the main PPO
  loop ran 17x12 batched envs — the reason this path felt "crazy slow" at
  `ppg_rollout_steps=580000`. With the flag on it spawns
  `batched_rollout_worker.py` workers instead
  (`value_pretrain_n_processes` x `value_pretrain_envs_per_process` envs;
  `batch_secondary_players` reused from the main loop). Pieces:
  `_spawn_value_pretrain_workers`/`_close_value_pretrain_workers` (one
  worker-lifecycle code path, replacing the spawn logic that had been
  duplicated inside `ppg_value_refit`; the collector takes an optional
  `pool` — None = spawn+close per call exactly as before, given = caller
  owns it, which is how `num_rollouts` > 1 reuses one pool; weights are
  re-synced every call regardless), `_finalize_value_pretrain_result` (pure
  per-result MC/GAE finalizer shared by every branch, unit-tested without
  processes), and an optional shared `progress_value` counter on
  `BatchedEnvGroup.collect`/`_batched_worker_main`/`spawn_batched_workers`
  (parent resets it, workers only add — a worker-side reset races between
  workers; default None = the main loop is untouched).
  **Chunk-streamed, WHOLE EPISODES ONLY** (`ppo.value_pretrain_chunk_steps`,
  default 3000, 0 = one send at the end). An earlier revision of this
  section said value pretrain could NOT be chunked because a flush cuts
  episodes (~20% straddling at ~50-step episodes, wrong MC heads, an episode
  split across train/val). That was the wrong conclusion — the cut itself
  was the bug. See the next bullet: flushes now only ever send completed
  episodes, so chunking is safe here and everywhere.
  **GAE bootstrap change riding along:** with `use_gae=True` both worker
  kinds return a real `last_value`, which is now used as-is with NO
  truncation (a trailing partial episode is kept, correctly bootstrapped)
  instead of the earlier truncate-then-`last_value=0.0`; only the bare
  single-process branch (no `last_value`) still truncates + zero-bootstraps.
  MC mode (`pretrain_value`) is unchanged: truncate, then pure MC returns.
- **LR warmup (`bc.ppg_warmup_steps`, and `ppo.lr_warmup_steps` for the main
  update).** Each refit cycle re-snapshots the anchor, so KL starts at exactly
  0 and the first Adam steps overshoot before the KL pull catches up (epoch-1
  peak `kl_train` ~0.0034 at minibatch ~4 vs ~0.0008 by epoch 5, lr 5e-5,
  kl_coef 100, offline experiment on one 300k-row rollout, 2 cycles per
  setting). Linear warmup over the first N steps of every cycle: N=10 cut the
  peak ~45%, N=25 ~60%, N=50 ~70%; N>=100 (about 2 epochs at that size) just
  moves the bump into later epochs (epoch-5 KL 0.0007-0.0010, no lower than
  no warmup). Equilibrium KL is set by kl_coef vs the value gradient, not by
  warmup. lr 1e-4 without warmup peaked at 0.014 (4x). Val loss was WORSE after
  5 epochs in every setting (0.1916 -> 0.1969-0.1999; longer warmup only
  looks better because it fits less): the refit mostly fits its own rollout's
  episodes, and fresh-data baselines across rollouts stay flat (~0.19).
  `ppo.lr_warmup_steps` applies the same linear ramp to every PPO update
  (`_lr_warmup_scale`); its right N there is untested.
- **Compare-to-original diagnostic (`bc.ppg_compare_to_original`).** The
  per-rollout baseline losses are dominated by rollout-to-rollout noise
  (~0.184-0.200 across rollouts for a flat model), so "is the refit improving"
  can't be read off them. The fix is a PAIRED check: keep a frozen copy of the
  networks as loaded at the start of the call and, on each rollout's fresh rows
  (before training on them), score both it and the current weights on the same
  targets; the delta cancels the data noise. The targets must be PURE-MC
  returns (`mc_returns`, NaN on unfinished tails -- `_mc_returns_nan_tail`),
  NOT the GAE returns the refit trains on: GAE returns = A + V_sampler contain
  the sampling network's own predictions, so any other network scored against
  them loses by ~its squared disagreement with the sampler -- the same order
  as the effect being measured (~0.003). `mc_returns` is attached by the
  worker-side finalize when the returns spec has `"with_mc": true`, and passes
  through merge / `_split_batch_releasing` / `augment_batch` as a plain
  per-row scalar. Cycle 1's delta is 0 by construction (same weights).
- **Value-loss diagnostics (`bc.ppg_loss_diagnostics`).**
  `_value_loss_breakdown` (pure numpy, unit-tested) reports, on a rollout's
  fresh identity-copy rows and pure-MC targets: error by episode outcome x row
  owner (the outcome label is the TRAINEE's view, so the opponent track's
  rows in a 'box_possession' episode are the losing side -- pooled they look
  like ~2.4 std of nonsense; ALWAYS split by track), by row owner x the OTHER
  player's AI type (`other_ai_type` one-hot, [rules, immobile, neural]), by
  position within the episode (`_episode_phase_fraction`: quartiles; per-track
  counters reset at `done` and at the NaN-tail -> finite seam between
  concatenated per-env buffers), the share of squared error from the worst
  1%/10% of rows, the variance left if outcome + owner + episode quartile were
  known (row returns are DISCOUNTED, so within one outcome they still vary with
  time-to-go; an earlier version of this note claimed returns are ~a function of
  the outcome with within-outcome std ~0.09 -- that was the UNDISCOUNTED
  episode total, wrong for per-row returns), a decile calibration table and the
  worst rows. Cycle 1 and every `ppg_diag_every_n_cycles`-th cycle also dump the
  per-row arrays to `<checkpoint_dir>/ppg_diag_cycle<k>.npz`.
  First real reading (deterministic rollouts, R2 0.78, normalised mse 0.234):
  the value net is essentially perfectly CALIBRATED (predicted deciles match
  realised returns to ~0.03), so the error is discrimination not bias; 71% of
  the squared error is in the first half of episodes (0-25%: mse 0.363, 75-100%:
  0.074) -- outcome uncertainty that resolves as the episode plays out; the
  worst 10% of rows carry 69% of the error; the top worst rows are all 'miss'
  (ball out of play, 0.7% of rows, 3.3% of the error) where the net was
  confidently (+3.8) wrong.
- **Refit obs stays on CPU; physics features are precomputed once.** Per row,
  `other_feat` is 21x39 floats but the frozen player-physics encoder's
  `other_physics_full` is 21x74 -- caching it (as `_ppo_update` does, via the
  shared `_precompute_physics_full`) makes the obs ~3x bigger (~17 GB for a
  1.6M-row train set). So `ppg_value_refit` keeps train/val obs on CPU and
  moves ONE chunk at a time to the device in the anchor/eval/epoch loops
  (like PPO's per-minibatch `.to()`); only returns, actions and the per-row
  anchors live on the GPU. History: the original version put the whole
  (uncached-physics) train obs on the GPU; measured no-grad forward incl.
  encoders was ~200k rows/s but the training epoch only ~2.3k rows/s, so
  the encoders were NOT the bottleneck -- suspected VRAM overflow into
  shared memory (12 GB card). Unconfirmed until the CPU-resident version's
  epoch speed is measured.
- **The value plateau (held-out R2 ~0.73 stochastic / ~0.78 deterministic) is
  NOT a capacity or structure limit.** Offline probe on a frozen run276
  policy (3.0M stochastic rows / 2.4M deterministic rows, split by GAME so both
  players' rows stay together; inputs = canonical raw obs + frozen physics
  encodings; target = pure MC return): every fresh value net -- linear 0.60,
  128x2 MLP 0.723, a 256->48->64->1 bottleneck (mimicking a value head that
  reads the 48-wide exec trunk) 0.724, 4x1024 on raw only 0.726, 4x1024 on
  raw+physics 0.728 -- lands at or below the sampling net's own 0.732. The big
  nets memorise (train mse ~0.006 vs val ~0.28 normalised). Learning curve
  is slow log-linear in data (+~0.02 R2 per 3x games: 0.709 / 0.734 / 0.753 at
  10% / 30% / 70% on deterministic rows). The big nets are NOT undertrained:
  train mse ~0.006 (normalised), best val at epoch 1-3 -- they memorise.
  **A deterministic policy does not make the return a function of the
  observation: the sim itself is random during play** (kick yaw/pitch gauss
  error, control-time gauss noise, tackle skill rolls; per-match
  `Match.rng`). Fork test (384 mid-episode states x 16 forks, same state,
  different `match.rng`, deterministic policy rolled to episode end): 18% of
  trainee-track states span >3 in return (different win/loss outcome from
  the identical state); irreducible within-state variance = 13% of total
  => **R2 ceiling ~0.87 for any predictor** (trainee 0.871, opponent 0.872).
  The sampling net scores ~0.70-0.78, so ~0.1-0.17 R2 of *learnable*
  variance remains, and fresh big nets (0.75) don't close it -- it is a
  data/generalisation gap, not capacity (about half the current MSE is
  irreducible noise). Near-boundary `miss`: a dedicated 12-feature classifier
  gets AUC 0.93 (final row) / 0.80 / 0.74 among at-the-line rows. NOTE
  `-value` is a poor miss SCORE (value is dominated by who is winning), so
  the value net's "AUC ~0.55" alone does not show it ignores miss risk; the
  right test is the residual slope (MC - pred) on the classifier's P(miss),
  owner-touched & possessed rows: sampling net -4.6 +/- 0.5, fresh big MLP
  -4.9 (a value that ignores the risk scores ~ -4, one that accounts ~ 0).
  It moves to -4.0 with 10 hand-built geometry columns (dist-to-line,
  outward speed, time-to-line, ...) and to -0.9 +/- 0.5 with those columns
  AND a x300 loss weight on near-line rows (band MSE -10%, overall R2 -0.002).
  So the miss risk is missed because of signal density (near-line rows ~0.3%
  of data, positives ~0.06%; MSE gradient negligible, early stopping hits
  first) plus feature accessibility, NOT capacity. Worth <0.001 of the
  ~0.235 overall MSE. Don't re-chase capacity: widening / bigger heads /
  bottleneck changes have nothing to gain.
  **Correction (2026-09-20): misses ARE decisive once conditioned on speed.**
  Pooling outward speeds from 0.5 m/s hid it. Possessed, owner-touched-last,
  outward 4-6 m/s: P(miss<=3 rows) = 0.63 (0-0.5 m inside the painted line),
  0.92 (0-0.3 m outside), 1.00 (0.3-0.8 m outside); MC -2.7 / -3.95 / -4.5 vs
  value -0.85 / -1.3 / -0.8. White-box checks on the real run276 net
  (143 decisive rows): recomputed value == recorded (no plumbing/lag bug);
  value barely moves for ball 4 m inward (+0.13), velocity reversed (+0.24),
  both (+0.36), vs ideal ~+3; last-touch -> other team only +0.62 (ideal
  ~+2.5); single physics-encoder columns DO separate misses (AUC up to 0.80),
  so the information reaches the net. The PRODUCTION architecture can learn
  it: offline value-only fine-tune (physics encoders frozen, 315k params) with
  near-line rows weighted x100, 3 epochs, held-out decisive rows: mean value
  -0.85 -> -2.12 (MC -2.83), band mse 2.24 -> 2.14, but overall random-row R2
  0.781 -> 0.765 (shared-trunk interference). Root cause is mainly the
  rare-event / gradient-density problem (decisive rows ~0.006% of data).
  The physics encoders' crossing heads are FED to the net (ball
  `crossing_head` = cols 40-43 / `event_head` 48-49 of the 98-dim ball block;
  PLAYER `crossing_head` = cols 24-27 of the 74-dim player block, "will this
  player leave the painted line if its current intent persists", used for
  self and other) but are SATURATED and non-discriminative here: on decisive
  rows ball crosses_prob 0.984 (miss) vs 0.984 (no miss), event 0.95 vs 0.95,
  player(self) crosses_prob 0.94 vs 0.86 (AUC 0.62); every row in the
  near-line outward population "crosses the painted line", the question is
  how far past it the ball/player goes before turning back, which the heads
  don't model (first-crossing only). What DOES discriminate is raw outward
  speed (AUC 0.71) and distance to the line (0.75), which the value net
  underuses (0.54-0.61). Env note: `detect_trial_outcome` ends the episode
  only when |ball.x| > half_length + 1.0 or |ball.y| > half_width + 0.5 (not
  the painted line; no documented rationale, present since outcome.py was
  created; nothing clamps players to the pitch and the ball is glued to the
  carrier). At 4-6 m/s that margin is ~0.1 s so it does NOT explain the
  decisive-state failure; it only adds turn-back ambiguity at low speed.
  **FIXED 2026-09-20**: `detect_trial_outcome` now applies Law 9 (out only
  when the WHOLE ball is over the WHOLE line: centre more than `ball.radius_m`
  past the painted line, both axes, any height) and checks the scoreboard for a
  goal BEFORE "out" (a hard shot moves ~0.7 m/tick, so the goal tick can
  already have the centre past the line). Tests: tests/ai_unit/test_outcome.py,
  tests/scenario/test_ball_out_overrun.py. Returns/value targets from before
  this date used the loose 0.5 m / 1.0 m rule, so they are not comparable.
  Origin: first appears 2026-07-28 in the UI scenario loop's `_trial_outcome`
  ("Ball out of bounds (touchline or behind goal without scoring)" -- the
  x margin is plausibly so a goal can register first), later reused for RL
  termination. The engine has NO out-of-play rule of its own; this function
  is the only place "out" is defined, so the effective RL out line is 0.5 m
  (touchline) / 1.0 m (goal line) beyond the painted line. Size (2.4M det
  rows / 29k games): ball outside the painted line on 0.145% of trainee rows,
  in 2.7% of games at some decision row; of those 795 games 522 ended via the
  out rule and 273 (0.9% of all games) carried on with the ball back in play
  -- the only episodes a painted-line rule would cut short.

## Batched rollouts: whole-episode chunks, worker-side finalization, RAM

**Whole-episode flushes (applies to the MAIN PPO loop too, not just value
pretrain).** `BatchedEnvGroup.collect(chunk_steps=...)` used to flush every
env's whole buffer at a step-count boundary regardless of where its episode
was, and this file called that an "accepted trade-off". It wasn't necessary
and it cost real correctness at every boundary: (1) GAE for the rows next to
the cut was bootstrap-truncated off a value estimate, biasing those
advantages; (2) episode-seed replay's per-episode mean(|advantage|) was
computed from only the half of a straddling episode that shared a chunk with
its terminal row, corrupting the replay ranking; (3) a straddling episode's
halves could land on both sides of a train/val split; (4) per-episode
diagnostics ([advantage |.| by episode], step-outcome backfill) saw half
episodes. Now a mid-collection flush pops each env's COMPLETED episodes
(`RolloutBuffer.pop_complete_episodes()`) and leaves the unfinished tail in
the worker until its episode ends, so a chunk always ends on `done=1`: GAE
needs no bootstrap (`last_value` is 0.0 and the per-flush bootstrap forward
pass is skipped), MC returns are exact, and a chunk's stats describe exactly
the episodes whose rows it holds. The only remaining cut is the unavoidable
one — each env's trailing in-progress episode at the END of a rollout (the
env carries on into the next), bootstrapped with a real `last_value`, one per
env per rollout, exactly as before chunking existed. If a flush is due but
no env has finished an episode yet it retries next round.

**Worker-side finalization + compact wire format.** `collect(returns=
{"mode": "gae"|"mc", "gamma", "lam"})` makes the worker compute returns and
`as_tensors` itself and send flat numpy arrays (`finalize_result_for_wire`;
numpy on purpose, so torch's ForkingPickler shared-memory reductions aren't
involved) instead of a pickled per-row `RolloutBuffer` — millions of tiny
dicts/arrays that were expensive to pickle, unpickle and re-stack in the main
process. `gamma`/`lam` travel in the command (the main process is the source
of truth; a worker's own ai_config.json read can drift if the file is edited
mid-run). No `returns` spec = the legacy `{"buffer","last_value","stats"}`
wire format, unchanged (tests and any other consumer). Both `_train_batched_
parallel` and the batched value-pretrain path request it.

**RAM at the end of a rollout.** The spike was several full copies alive at
once: worker buffer + its pickle, received bytes + unpickled per-row buffers,
the stacked tensors, the `torch.cat` merge, the train/val split copies, the
augmented train set (x2 at `augment_n_slot_shuffles=1`) — with the raw
results/per-worker lists/merged batch/raw split all still referenced.
Fixes: streaming whole-episode chunks (per-message memory bounded), results
ingested as they arrive, `_merge_worker_batches(release_inputs=True)` (delete
each key from the inputs as it's concatenated: ~1x + one key, not ~2x),
`_split_batch_releasing` (slice the merged batch key-by-key while emptying it),
dropping the un-augmented copy after augmentation, and clearing every big
per-cycle name at the end of each `ppg_value_refit` cycle so the previous
cycle isn't resident during the next rollout. NOT changed: `_ppo_update`'s own
`_split_train_val_episodes`/`augment_batch` copies (a separate, delicate
phase).


## Action-opportunity flags and masked action training (2026-09-21)

Plan: `agent_plans/masked_action_training_plan.md`. Problem it fixes: the kick gate, the kick direction/power heads
and the tackle gate were trained on EVERY decision row, but the action can only take effect in a small share of them
(measured on run 303 checkpoint 95, 750k rows: a kick could execute in 37-38% of rows, an armed tackle could reach
contact in only 1.5-1.6%, and only 0.5% of rows had a kick actually fire; 88% of the tackle_attempt=1 samples had no
opposing carrier in reach). The gradient from the other rows is pure noise for those heads and the entropy bonus
inflates them there. Fix: record, per decision interval, whether the action COULD take effect, and train the heads
only on those rows.

**What an "opportunity" is** (one definition, evaluated by the engine, never re-derived in the env/trainer):
* kick: the player could execute a kick at some tick of the interval -- it holds the ball when its kick intent is
  (re)applied (`Player.can_kick`, THE single validity predicate: both kick executors, the fire-vs-arm branch of
  `apply_action_to_player` and the accounting all call it), OR it gains the ball by a LOOSE-BALL PICKUP
  (`Match._update_loose_ball_pickup`, hook inside the engine's own pickup code) -- i.e. first touch counts, both the
  armed redirect-at-pickup and the possession kick from `CONTROLLING_BALL` when the ball was still bouncing. Possession
  won by a tackle is NOT a kick opportunity (it depends on the player's own tackle bit, see the independence rule).
* tackle: an opposing player is the ball carrier within contact range with both players available
  (`Match._tackle_contact_possible` -- the exact predicate armed-tackle resolution uses; it includes the
  `INACTIVE_TACKLED` checks on both sides). Evaluated for every player every tick, before any tackle resolves, whether
  or not the player armed one.
* fired: `Player.kick_count` advanced (kick) / `Player.tackle_fire_count` advanced (an ARMED tackle reached
  `_attempt_tackle_contact`; head-on auto-tackles are not fires).

**Independence rule (I1).** A masked policy gradient is unbiased only if "an opportunity existed" does not depend on the
gate bit being trained. It holds for the FIRST opportunity tick of an interval (the world is identical up to it for either
bit value) but not after: a fired kick releases the ball, a won tackle hands this player the ball. So only whichever head's
opportunity comes FIRST in the interval counts (`ppo.action_opportunity_cross_head_cutoff`, default true; the raw flags
`opp_kick_raw`/`opp_tack_raw` are always recorded). Measured cost: both-raw = 0.5% of intervals; the rule drops 1.2% of raw
kick and 2.2-2.6% of raw tackle opportunities. Tested by running all four (kick, tackle) bit combinations from identical
seeded states: the cut flags never differ (with the rule off the same test fails).

**Plumbing.** `entities/action_opportunity.py` (`ActionOpportunity`, one per `Player`, `begin_interval` at every fresh
`NeuralPlayerAI.apply`) -> `ScenarioEnv._with_opportunity_flags` (end of `step()`; only when
`ppo.log_action_opportunity_stats` or `ppo.mask_untriggered_actions`) -> `raw_exec` -> `_action_to_numpy` -> the batch's
`action/opp_kick, opp_kick_raw, fired_kick, opp_tack, opp_tack_raw, fired_tack, opp_partial` tensors. They ride the
existing `action/*` machinery, so merge / chunked flush / episode replay / train-val split / augmentation tiling need no
changes (augment treats unknown `action/*` keys as flip-invariant). Absent keys => all-ones masks => old behaviour.
`opp_partial` marks the rare (0.1-0.2%) intervals cut short by a non-terminal early exit (flags incomplete).

**Training** (`ppo.mask_untriggered_actions`, default false; needs the flags; phases with frozen pass/tackle/mark heads):
* kick gate trains on rows with a kick opportunity (both bit values), tackle_attempt gate on tackle-opportunity rows,
  kick_dir/kick_power on rows where a kick fired (whether it fires does not depend on dir/power, so conditioning on it
  is unbiased for them). NOT "fire-only positives, all negatives" (Design A) -- that is biased and kept as a negative test.
* the PPO ratio uses log-probs summed over exactly those masked heads on BOTH sides
  (`_mask_total_log_probs`: `sum_h mask_h * col_h` + the total's non-column residual, detached, so a masked head gets
  EXACTLY zero gradient); the stored `head_log_probs` columns must equal the terms of the stored total (tested).
* entropy: kick / tackle_attempt gate entropy is the mask-weighted mean; kick_dir/kick_power entropy is weighted per row by
  `opp_kick * p_kick(row)`. Per-head KL / counterfactual policy loss diagnostics average over mask rows only.
  Whole-policy diagnostics (worst-sample, ratio spike per-head) stay on the unmasked policy.
* not masked: PPG / consistency anchors, BC aux loss.
* `ppo.mask_rescale_by_opportunity_rate` (default false): per-head gradient x 1/mean(mask), value-preserving (plan D9).

**Diagnostics.** `[action opp]` block per rollout (P(opportunity), P(kick|opp), P(kick|no opp), P(fired|kick), rows that
would train each head, invariant-violation counters that must be 0). Instrument first, then the loss change.
Rollout-level numbers at checkpoint 95: kicks are ~97% possession/first-touch kicks that all fire; P(kick=1|opp)=1.3%;
tackle: P(tackle=1|opp)=28% but only 12% of tackle=1 samples have an opportunity, P(fired|tackle=1)=12.5%;
rows that would train: kick gate ~280k, tackle gate ~11-12k, kick dir/power ~3.6-3.8k per 750k-row rollout.

**Known limits.** (1) `opp_partial` intervals. (2) tackle opportunity is contact range, so a tackle blocked by aerial
control / GK immunity still counts as an opportunity (the bit has a reward effect there). (3) The tackle gate and kick
dir/power heads get 1-2 orders of magnitude fewer training rows than before (plan D9/D12) -- watch their learning speed.
