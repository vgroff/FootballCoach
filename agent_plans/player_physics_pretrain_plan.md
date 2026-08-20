# Player Physics Pretraining Plan — Frozen Dynamics Latent for PlayerFeatures

> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

## 0. Status

**Standalone pipeline implemented** (episode generation, dataset, network,
training script, auto-generated HTML report), mirroring
`agent_plans/ball_physics_pretrain_plan.md`'s pipeline and its §12.5 QOL
checklist. This is a follow-up to that plan's §11 sketch, built once the
ball pipeline was validated and in active use.

**Live-network integration (the player-net analogue of the ball plan's §8)
is explicitly OUT OF SCOPE for this pass** — this plan covers ONLY the
standalone offline pretraining pipeline (episode generator → dataset →
network → train script → HTML report → a frozen encoder checkpoint).
`EntityEncoder`, `DecisionNetwork`, `ExecutionNetwork` are untouched. See
the ball plan's §11 for the sketch of how a future integration pass would
differ structurally from the ball's §8 (widening `EntityEncoder.per_
entity_mlp`'s input rather than `DecisionNetwork.forward()`'s top-level
`ball_mlp`/`ball_query_proj`, since a player is one of up to 22 permutation-
invariant entity slots, not a single global entity like the ball) — that
sketch is not re-litigated here.

## 1. File layout (as built)

```
src/footballcoach/ai/physics_pretrain/
  identity_shortcut.py       # NEW shared module -- see §2
  player_episode_gen.py      # PlayerEpisodeGenParams, generate_episode(), generate_shard()
  player_dataset.py          # generate_dataset(), PlayerDynamicsDataset, __main__ CLI
  player_dynamics_net.py     # PlayerDynamicsEncoder / Decoder / Autoencoder
  train_player_dynamics.py   # compute_loss(), compute_confusion_counts(), train(), __main__ CLI
  report.py                  # EXTENDED (not rewritten) to be generic -- see §6
  report_template.html       # EXTENDED (not rewritten) to be generic -- see §6

  # unchanged except the identity_shortcut extraction:
  ball_episode_gen.py
  ball_dataset.py
  ball_dynamics_net.py       # now imports _init_identity_shortcut_{linear,decoder} from identity_shortcut.py
  train_ball_dynamics.py

src/footballcoach/ai/config/ai_config.json      # + "physics_pretrain.player" section
checkpoints/physics_pretrain/                   # player_encoder.pt / .history.npz / .report.html (gitignored)
physics_pretrain_data/player/                   # generated .npz shards (gitignored)
tests/ai_unit/test_player_physics_pretrain.py
```

## 2. Step 0: shared identity-shortcut extraction

`ball_dynamics_net.py`'s `_init_identity_shortcut_linear`/`_init_identity_
shortcut_decoder` were already fully general in `dim`/`noise_std`; the only
ball-specific bit was `_init_identity_shortcut_decoder`'s hardcoded module
constant `Z_FIELD_INDEX = 2`. Extracted both functions verbatim into
`identity_shortcut.py`, generalizing `Z_FIELD_INDEX` into a
`nonneg_field_index: int | None` parameter (`None` = no such field, every
one of the `dim` fields gets the full bidirectional 2-unit treatment, no
freed spare unit). `ball_dynamics_net.py` now imports from there and passes
`nonneg_field_index=Z_FIELD_INDEX` at its one call site; `Z_FIELD_INDEX = 2`
stays defined in `ball_dynamics_net.py` itself. Pure extraction, verified
with `pytest tests/ai_unit/test_ball_physics_pretrain.py -q` (44/44 passing,
unchanged, both before and after) before building anything player-specific.

`player_dynamics_net.py` reuses the same two functions, passing
`nonneg_field_index=STAMINA_FIELD_INDEX` (the player's own provably-
non-negative field — the current stamina fraction, always clamped to
`[0, 1]` by `drain_stamina`/`regen_stamina`).

## 3. Field layout (final, as built)

### 3.1 Input — 24 floats (`N_INPUT_FIELDS`)

The brief's own naive arithmetic (`7 + 12 + 3 = 22`) was wrong (flagged as
such in the brief itself, "count it yourself") — the additional-context
block is actually 14 fields, not 12, once `desired_speed_mode`'s one-hot
(3 fields) and `has_possession` (1 field) are counted individually. Actual
total: **7 + 14 + 3 = 24**.

| # | Field | Notes |
|---|---|---|
| 0-1 | `pos_x, pos_y` | pitch-relative, see §4 normalization |
| 2-3 | `vel_x, vel_y` | |
| 4-5 | `heading_sin, heading_cos` | |
| 6 | `stamina` | current stamina FRACTION (state), `STAMINA_FIELD_INDEX` |
| 7-10 | `top_speed, acceleration, stamina_attr, ball_control` | `PlayerAttributes` fields actually used by movement physics |
| 11 | `has_possession` | fixed for the whole episode |
| 12-13 | `desired_direction_sin, desired_direction_cos` | current intent's direction at t=0/this row |
| 14-16 | `desired_speed_mode` one-hot | `[is_standstill, is_jog, is_sprint]` |
| 17-20 | `pitch_length_norm, pitch_width_norm, goal_width_norm, goal_height_norm` | ratio-to-base, always exact regardless of the normalization flag below |
| 21 | `speed_norm` | engineered, `sqrt(vel_x^2+vel_y^2)` |
| 22-23 | `heading_desired_diff_sin, heading_desired_diff_cos` | engineered, sin/cos of `angle_diff(heading, desired_direction)` |

Fields 0-6 are `N_IDENTITY_SHORTCUT_FIELDS` (7) — the identity-shortcut
block, same order/normalization as the target's own first 7 fields.

**Attributes deliberately not included**: `kick_precision`, `kick_power`,
`dribbling`, `tackling` — none of them feed `step_player_towards`/
`drain_stamina`/`regen_stamina`/`effective_top_speed`, so including them
would be pure noise for this net's actual task (kick physics/tackling
aren't simulated here at all — see §5).

### 3.2 Output — 9 floats × `len(horizons_s)` (`N_TARGET_FIELDS_PER_HORIZON = 9`)

`pos_x, pos_y, vel_x, vel_y, heading_sin, heading_cos, stamina` (7, same
order/normalization as the input's identity block) + `out_of_bounds,
goal_scored` (2, BCE logits at train time). Directly mirrors the ball's
`N_TARGET_FIELDS_PER_HORIZON = 11 = 9 identity + 2 classification` pattern,
just with 7 identity fields instead of 9 (no spin, no height/z axis).

### 3.3 Engineered features — 3, not 6

The brief was explicit not to force a count match with the ball's 6
(drag/Magnus-specific) — only 3 nonlinear combinations here actually map
onto a real term in `movement.py`: `speed_norm` (the turn-rate denominator's
`max(current_speed, 0.5)` term) and `heading_desired_diff_sin/cos` (drives
`turn_fraction`/`turn_speed_penalty` via `angle_diff`). No drag/Magnus
analogue exists for player movement, so no squared-speed or cross-product
terms were added.

## 4. Normalization

Directly mirrors `ball_episode_gen._kinematics_divisors`, simplified per
the brief's explicit instruction ("since there's no z axis... should
actually be simpler... don't add complexity that isn't needed"): one
function, `_kinematics_divisors(pitch, params) -> (div_x, div_y, div_vel)`,
gated by `physics_pretrain.player.normalize_kinematics_by_base_pitch` (same
name/semantics as the ball's flag). `heading_sin/cos`, `desired_direction_
sin/cos`, and `stamina` need no additional normalization (already
unit-scale).

## 5. Episode generation — deviations from a naive ball-copy

### 5.1 No freeze-on-event rule (the biggest deviation)

The brief's own output-field spec never mentions a freeze/latch rule for
`out_of_bounds`/`goal_scored` — it specifies them as `pitch.is_in_bounds`/
`is_goal` "at that horizon" (instantaneous), which matches the ball
pipeline's CURRENT, already-revised convention (`physics_pretrain.ball.
freeze_semantics` now defaults to `false` — see the ball plan's §12.6/§4.3
history). Since the ball pipeline itself moved away from freeze-on-event
after real-world experience, and the player brief's spec was already
written in the non-latched style, the player pipeline was built
**continuous-only from the start**: physics never stops/freezes early, and
there is no `crossings`/`crossing_times` side-channel to carry (a genuine
simplification vs. the ball pipeline, not a missing feature — see
`player_episode_gen.py`'s module docstring).

### 5.2 Intent resampling

A player needs an ongoing "intent" (`desired_direction`, `speed_mode`) to
move at all — the ball has none of this (pure physics from one initial
condition). Resampled exactly at t=0 and at each `horizons_s[i]` boundary
(piecewise-constant within a segment), per the brief's spec: with
probability `intent_continue_prob` (default 0.5), keep `speed_mode` and
jitter the direction by small Gaussian noise; otherwise draw a fresh
direction (uniform) and `speed_mode` (categorical, `speed_mode_weights`).
At t=0, always fresh (no "previous" intent to continue).

### 5.3 Per-tick physics — exact match to `Match._apply_movement`/`_update_state_timers`

Every tick: `step_player_towards(player, direction, speed_mode, dt,
movement_params, has_ball=has_possession)`, THEN `player.stamina =
drain_if_sprinting(...)` (SPRINT only, effort=1.0), THEN SEPARATELY
`player.stamina = regen_stamina(..., dt * 0.3)` (unconditional, every
tick, even while draining) — replicated exactly per the brief's explicit
instruction not to "fix" this quirk, since matching the real engine is the
whole point.

### 5.4 Initial-condition sampling — the heading/velocity invariant

`step_player_towards` always sets `velocity = speed * (cos(heading),
sin(heading))` — heading and velocity direction can never disagree in the
real engine. The sampler enforces this by construction: sample `heading_
rad` uniformly, sample `speed` uniformly in `[0, effective_top_speed(...)]`
(using THIS episode's own sampled attrs/stamina/has_possession), THEN
derive `velocity = Vector3.from_angle_xy(heading, speed)` — never
independently. Verified by `test_heading_velocity_invariant_at_t0` in the
test suite. `PlayerAttributes`' 4 unused-by-movement fields (`kick_
precision`, `kick_power`, `dribbling`, `tackling`) are set to a neutral
`0.5` (validation requires all 8 fields in `[0, 1]`) — never sampled
randomly, since they're provably irrelevant to this net's task and
randomizing them would be pure wasted entropy.

### 5.5 Position sampling

`_sample_position_in_play`: uniform within pitch bounds with margin, z
always 0 (no airborne case — players are grounded). `_sample_position_
already_special` (fraction `out_of_bounds_start_frac`): a simplified 2D
adaptation of the ball's out-of-bounds sampler, including a cheap "already
past the goal line, within goal-mouth width" case (gives the dataset some
`goal_scored=1`-eligible starting states without a dedicated resampling
loop) alongside plain x/y-boundary overshoots.

## 6. Report generalization (`report.py`/`report_template.html`)

Per the brief's explicit instruction ("extend them generically... rather
than writing a second bespoke report generator"), both files were EXTENDED
in place, not copied:

- `report.py`'s `write_report()` gained five new OPTIONAL parameters
  (`panel_defs`, `header_stat_defs`, `best_table_defs`, `title`,
  `config_namespace`), all defaulting to `None`. The ball pipeline's own
  call site in `train_ball_dynamics.py` was **not touched** — it still
  calls `write_report()` with none of these, so `None` flows through and
  `report_template.html`'s JS falls back to its original hardcoded ball
  defaults byte-for-byte (verified: all 44 ball tests, including the report
  ones, still pass unchanged).
- `report_template.html`'s `PANEL_DEFS` is now `DATA.panel_defs || [...ball
  defaults...]`; `buildHeader()`'s headline stat tiles and `buildBestTable()`'s
  rows follow the same `DATA.<x>_defs || <ball default>` pattern, with the
  actual per-metric table/chart rendering logic factored into a shared
  `_renderBestTable()` helper so the generic and ball-hardcoded paths can't
  drift apart silently. `buildFooter()`'s config namespace and `<title>`/`<h1>`
  are now driven by `DATA.config_namespace`/`DATA.title` (again falling back
  to the ball's own strings).
- `train_player_dynamics.py` supplies its own `panel_defs`/`header_stat_
  defs`/`best_table_defs` (9 continuous/BCE panels + 6 classification
  panels + 8 R²/%-of-persistence panels = 23 total) and
  `title="Player Dynamics Training"`, `config_namespace="physics_pretrain.
  player"`.

This is a real (if bounded) change to shared infrastructure, not a "pure
extraction" the way §2's identity-shortcut move was — it was verified by
re-running the FULL ball test suite (44/44) after the change, and confirmed
the player pipeline's own smoke test produces a report whose embedded JSON
parses and whose `<script>` block passes `node --check`.

## 7. Deviations from the ball pipeline's fuller QOL feature set (deliberate scope cuts)

The ball pipeline (as it stands today, per its own §12.6 history) has grown
several features beyond the original design that were deliberately **not**
ported here, to keep this pass's scope bounded to what the brief actually
asked for:

- **No ballistic/straight-line-physics baseline.** The ball's `compute_
  ballistic_baseline_mse` (constant velocity + gravity, no drag/Magnus/
  bounce) has no clean analogue for player movement (there's no simple
  closed-form alternative to `step_player_towards`'s acceleration/turn-rate
  model) — the player pipeline reports the "predict-the-mean" (R²) and
  "persistence" (%-of-displacement) baselines only, matching the rigor the
  brief actually asked for (§12.5's checklist) without inventing a baseline
  that doesn't map onto anything real.
- **No LR cosine-restart schedule, no `crossing_head`.** Both are later
  ball-pipeline additions beyond the original design doc's §7 sketch, not
  part of §11's original follow-up scope or the brief's explicit checklist.
  Flat LR throughout.
- **`build_adjacent_pair_data` has no "exclude already-resolved episodes"
  filter** — meaningful only under a freeze-on-event rule (§5.1), which
  this pipeline doesn't have.

## 8. Testing

`tests/ai_unit/test_player_physics_pretrain.py` (31 tests) mirrors
`test_ball_physics_pretrain.py`'s structure: episode-gen shape/determinism,
the heading/velocity invariant check, engineered-feature hand-computation,
distribution sanity (including the `has_possession=0 => goal_scored` always
0 gating), network shape/NaN checks, identity-shortcut round-trip
(zero-noise exact, noisy-perturbed, adversarial-classification-gradient
regression test with the freed-stamina-unit-stays-live check), dataset
generation/append/split/pos-weight/minibatch coverage, adjacent-pair/
autoencode derived-data re-encoding correctness, and two end-to-end
train-smoke tests (plain, and with both optional pretrain phases enabled).
Full suite (`test_ball_physics_pretrain.py` + `test_player_physics_
pretrain.py`, 75 tests total) passes.

## 9. Final verification

A tiny end-to-end run (a few hundred generated episodes, `train_player_
dynamics.py` for a couple epochs against a tiny config) produces a
checkpoint (`.pt`), a phase checkpoint, an `after_training.pt`, and an HTML
report whose embedded JSON parses and whose `<script>` block passes `node
--check` — see this plan's accompanying session notes / the test suite's
`test_train_smoke` for the automated version of the same check.
