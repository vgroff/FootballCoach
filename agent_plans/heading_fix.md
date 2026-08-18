# Heading Fix — Add `heading_rad` to `PlayerFeatures`

> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

## 0. Status

Not started. Flagged as a follow-up from `agent_plans/ball_physics_pretrain_plan.md`
§11 (the "live-network blind spot adjacent to" the player-dynamics work),
pulled out into its own plan since it stands alone and doesn't depend on
that other work landing first.

---

## 1. Why this matters

`PlayerFeatures` ([obs/schema.py](../src/footballcoach/ai/obs/schema.py))
has no heading field at all. The docstring's justification:

> heading dropped: velocity = (cos(heading)\*speed, sin(heading)\*speed) so
> heading is fully recoverable from velocity when speed > 0, irrelevant when
> speed = 0.

The first half is correct. The second half is not — and the gap is a real
information loss for the network, not a harmless simplification.

**Why "irrelevant at rest" is wrong**: `engine/movement.py::step_player_towards`
reconstructs velocity from heading every tick —
`player.velocity = Vector3.from_angle_xy(new_heading, new_speed)` (line 343)
— and `player.heading_rad` is a persistent field, independent of velocity,
read directly on the *next* tick regardless of current speed (line 308:
`current_heading = player.heading_rad`). That stored heading drives
`heading_diff`/`turn_speed_penalty` (lines 314-326) the instant the player
starts moving again. Concretely: **two standstill players at the same
position, same everything else, but facing different directions, will
diverge onto physically different trajectories the moment either one
starts moving** — one may need to turn 170°, the other 10°, incurring very
different turn-speed penalties. `PlayerFeatures` currently encodes these two
states identically (`velocity_x=velocity_y=speed_mps=0` either way), so the
network cannot distinguish them. This is a genuine aliasing bug in the
observation space, not a design tradeoff.

**Why "just normalize velocity to a unit vector" does not fix it** (raised
and ruled out in the design discussion that produced this doc): `velocity /
|velocity|` is a 0/0 singularity at exactly `speed == 0` — the one case
where the information is missing. Per `movement.py`'s
`_STOP_SNAP_THRESHOLD_MPS`, standstill players hit an exact `0.0` speed,
not just near-zero, so this isn't an epsilon-away edge case. Any function of
velocity alone — raw or normalized — has this same singularity, because
velocity is *constructed from* heading and speed each tick, not the other
way around. `(velocity_x, velocity_y, speed_mps)` and `(cos(heading),
sin(heading), speed_mps)` carry *identical* information whenever `speed >
0` — it's a reparametrization, not a fix. The only real fix is to stop
deriving direction from velocity and read `player.heading_rad` directly, since
it's already a persistent, always-defined field on `Player` independent of
speed.

**Scope of the impact**: likely small in practice (a few ticks of
mis-calibrated turn-speed penalty right after a standstill player starts
moving), which is probably why it hasn't surfaced as an obvious bug so far —
but it's real, cheap to fix, and worth closing given how much of this
codebase's effort already goes into precisely this kind of observation-
accuracy work (canonical frame, augmentation equivariance, etc.).

---

## 2. The fix

### 2.1 Add two fields to `PlayerFeatures`

In [obs/schema.py](../src/footballcoach/ai/obs/schema.py), append (after
the existing `pos_x, pos_y` fields, matching how that pair was appended
last time `PlayerFeatures` grew):

```python
# --- Heading (sin/cos, independent of velocity — see heading_fix.md) ---
heading_cos: float = 1.0
heading_sin: float = 0.0
```

Sin/cos encoding (not raw radians) to avoid the discontinuity at ±π, same
reasoning as every other angular quantity in this codebase.

`PLAYER_FEATURE_DIM` (`len(fields(PlayerFeatures))`) goes from 27/28 → 29/30
automatically — it's derived, never hardcoded.

### 2.2 Populate it in the encoder

In [obs/encoder.py](../src/footballcoach/ai/obs/encoder.py), near the
existing `vel_x`/`vel_y`/`speed` block (~line 226-233):

```python
heading_cos = math.cos(player.heading_rad)
heading_sin = math.sin(player.heading_rad)
```

Read `player.heading_rad` directly — do **not** derive it from `vel_x`/
`vel_y` (that's the exact bug being fixed). Populate for the self slot and
every other-player slot alike (same treatment as every other
`PlayerFeatures` field) — no `is_immobile` special-casing needed. Immobile
players never accelerate, so whatever value sits here is inert either way;
keeping the encoding uniform is simpler than adding a special case, and the
network already has the `is_immobile` flag to condition on if it matters.

### 2.3 Register the new fields with the flip/mirror machinery

This is the step most likely to be forgotten, and forgetting it produces a
correctness bug that's hard to notice (wrong-signed heading only shows up
as subtly-wrong behaviour under geometric augmentation or for `Team.RIGHT`
observers, not a crash).

`heading_cos` transforms exactly like `velocity_x`/`pos_x` (an x-direction
component); `heading_sin` transforms exactly like `velocity_y`/`pos_y` (a
y-direction component). In
[obs/augment.py](../src/footballcoach/ai/obs/augment.py):

```python
PLAYER_FLIP_X_IDX: list[int] = [
    ...,
    _field_index(PlayerFeatures, "velocity_x"),
    _field_index(PlayerFeatures, "heading_cos"),   # add
    ...,
]

PLAYER_FLIP_Y_IDX: list[int] = [
    ...,
    _field_index(PlayerFeatures, "velocity_y"),
    _field_index(PlayerFeatures, "heading_sin"),   # add
    ...,
]
```

**No change needed in `obs/canonical.py`** — `CanonicalNetworkWrapper`
imports `PLAYER_FLIP_X_IDX` directly from `augment.py` rather than
re-deriving its own list (`canonical.py`'s own docstring: *"There is exactly
ONE implementation of the mirror... imports and reuses those same index
lists"*), so the canonical x-mirror for `Team.RIGHT` observers picks up
`heading_cos` automatically once it's added to `PLAYER_FLIP_X_IDX`. This is
the payoff of that existing design — confirm it in the test in §4 rather
than assuming it, but no source change is expected here.

---

## 3. Cost / blast radius

This is a `PlayerFeatures` schema change — same class of break as the prior
26→28 `PLAYER_FEATURE_DIM` change documented in `ai/knowledge.md`:

- **All existing checkpoints** (`decision_net`/`execution_net`) fail to
  load — their first-layer weights (`entity_encoder.per_entity_mlp`'s first
  `Linear`) are sized for the old `PLAYER_FEATURE_DIM`. Expect to retrain
  from scratch or re-run BC pretraining; there's no meaningful partial-reset
  path the way `--reset-value-weights` offers for the value head (heading
  is a first-layer input change, not an isolated output head).
- **All recorded `.npz` BC datasets** (`demonstrations/phase1/` etc.) go
  stale — `record_demonstrations.py` calls the same `encode_observation()`
  path, so old recordings are missing the two new columns. Re-record.
- **No change** to `BALL_FEATURE_DIM`/`GLOBAL_FEATURE_DIM`, `bc.py`'s
  `BC_LABEL_DIM` (24) layout, or anything in `ai/physics_pretrain/` — this
  is orthogonal to the ball-physics-pretrain plan and does not block or get
  blocked by it. If both land around the same time, worth bundling into one
  retrain rather than two, purely to save wall-clock, but there's no
  correctness dependency either direction.

Given the cost is "just" a retrain (not a design risk), the main judgment
call is *timing* — bundle with the next schema-touching change already
planned rather than paying the retrain cost in isolation, unless the
standstill-heading issue is independently suspected of causing a live
problem worth fixing sooner.

---

## 4. Testing

- **`test_obs_schema.py`**: `PLAYER_FEATURE_DIM` reflects the new field
  count (already a generic `len(fields(...))` assertion, should need no
  change beyond the number moving).
- **`test_obs_encoder.py`**: a standstill player (`speed_mps == 0`) with two
  different `heading_rad` values produces two different `heading_cos`/
  `heading_sin` encodings — the regression test for the actual bug being
  fixed (encode the same position/velocity/attributes twice, only
  `heading_rad` differs, assert the output arrays differ at exactly those
  two new indices and nowhere else).
- **New flip/mirror test** (extend `test_ai_type_side_channel.py`-style
  coverage or add alongside `test_obs_encoder.py`'s existing flip tests):
  construct a player with a known `heading_rad`, encode it, apply
  `flip_y`, and assert `heading_sin` negates while `heading_cos` is
  unchanged (and the reverse for a `Team.RIGHT` canonical x-mirror via
  `CanonicalNetworkWrapper`) — confirms §2.3's "no `canonical.py` change
  needed" claim actually holds instead of just asserting it.
- **`test_networks.py`**: forward-pass shape/no-NaN check with the new
  `PLAYER_FEATURE_DIM` — should need no logic change, just confirms nothing
  hardcoded the old dimension.
