# Spin Implementation Plan — Neural AI Kick Spin

## Status quo (as of this document)

The neural network is currently **hardcoded to always kick with
`spin=Vector3.zero()`** — see [apply_nn_action.py](../src/footballcoach/ai/action/apply_nn_action.py).
This was a deliberate stop-gap decision (2026-08-09) to remove an
inconsistent, unclamped, effectively-untrained spin channel from the
live inference path while this plan is implemented. The `kick_spin`
network head, BC label plumbing, augmentation, and rollout-buffer storage
all still exist in the codebase — they are simply not wired to
`kick_with_direction()` any more. This document is the complete plan to
re-enable spin correctly.

This is a large, cross-cutting change. It touches:

- Physical spin cap semantics (`engine/kicking.py`)
- The execution network's output heads (`ai/models/execution_network.py`)
- Action sampling in PPO rollout collection (`ai/ppo/ppo_trainer.py`)
- The gating/apply pipeline (`ai/action/gating.py`, `ai/action/apply_nn_action.py`)
- BC label recording and loss (`ai/ppo/bc.py`)
- Demonstration recording (`ai/scripts/record_demonstrations.py`, `rules_ai.py`)
- Augmentation (`ai/obs/augment.py`)
- Config (`ai/config/ai_config.json`)

Read this document in full before touching any of the above files. Where
a section says "no change needed", it's called out explicitly so you don't
waste time re-deriving it.

---

## 0. IMPORTANT — 2026-08-17: `kick_power`/`kick_spin` PPO sampling landed, `kick_spin` is now FROZEN. Read this before touching sampling/log_prob/entropy code.

This section documents a **scoped, deliberately partial** step taken toward
this plan, and the immediately-following decision to freeze `kick_spin`
back out of training. Do not "clean this up" by assuming it's an oversight
— read the reasoning below first. If you're implementing the rest of this
plan (real spin physics), section 0.4 below is your checklist.

### 0.1 What actually landed

Before this date, `kick_power` and `kick_spin` were **fully deterministic**
at the PPO level — see §1.3/§6 above (kept for historical context, now
partially superseded by this section for these two heads specifically):
`kick_power_phys = float(torch.sigmoid(e_heads.kick_power))` and
`kick_spin_raw = e_heads.kick_spin.squeeze(0)`, no sampling, no log_prob, no
entropy, no PPO trust-region protection. Both `kick_power_log_std`/
`kick_spin_log_std` existed as `nn.Parameter`s but were never read anywhere.

This was fixed **without** the axis/magnitude redesign in §2 above (that
part of the plan is still not implemented — deliberately deferred, see the
scope discussion that led here). What changed instead, keeping `kick_spin`
as the original raw `(batch, 3)` vector head (no dimensionality change):

- `ExecutionNetwork.kick_power_log_std`/`kick_spin_log_std`
  ([execution_network.py](../src/footballcoach/ai/models/execution_network.py))
  now initialize from `ppo.kick_power_log_std_init`/`ppo.kick_spin_log_std_init`
  in `ai_config.json` instead of a hardcoded `torch.zeros(...)`.
- `PPOTrainer` gained two helper methods next to `_move_dir_head`/`_kick_dir_head`
  ([ppo_trainer.py](../src/footballcoach/ai/ppo/ppo_trainer.py)):
  - `_kick_power_head(mean, log_std) -> SquashedNormalHead` — sigmoid-squashed
    to `[0,1]`, same convention `kick_power` always had physically.
  - `_kick_spin_dist(mean, log_std) -> torch.distributions.Normal` — **plain,
    unsquashed** 3D Normal. Neither `DirectionHead` (forces a unit-vector
    mean — wrong, spin isn't unit length) nor `SquashedNormalHead` (forces a
    bounded range — spin has no bounded physical range wired up) fit an
    unbounded raw vector, so this is a small bespoke distribution, not a
    reused class.
- `_sample_action()` now samples both heads for real (mode in
  deterministic-direction mode, else `sample_raw()`/`rsample()`), gated by
  `kick=True` — same gating `kick_direction` already had, since both are
  only ever consumed downstream inside the engine's `if kick_this_tick:`
  block. Raw samples are stored in `raw_exec_samples`/`_action_to_numpy` as
  `kick_power_raw`/`kick_spin_raw`.
- `_compute_log_prob`, `_recompute_log_prob`, `_per_head_new_log_probs`
  (+ `rollout_buffer.HEAD_LP_KEYS`, extended to 15 entries — was 13) all
  gained matching gated terms for both heads.
- `_compute_entropy` gained `p_kick`-weighted entropy terms for both,
  controlled by new `ppo.ent_kick_power_weight`/`ppo.ent_kick_spin_weight`
  config keys (mirroring `ent_dir_weight`'s pattern, not reusing it — neither
  head is a `DirectionHead`).
- The deterministic-mode diagnostic block inside `_ppo_update()` (the
  `diag_act`/`_per_head_new_lp` local stack used for the worst-sample KL
  printout) was extended to match — this is what the regression tests
  (`test_smoke.py::test_ppo_update_produces_finite_losses` et al.) caught
  when it was missed on the first pass: a hardcoded 12-wide local stack
  silently desynced from the now-15-wide `HEAD_LP_KEYS`, causing a tensor
  shape mismatch crash. If you extend `HEAD_LP_KEYS` again in the future,
  grep for `_per_head_names` and `_new_lp_heads_map` in `ppo_trainer.py` —
  there are (at least) two more hand-maintained parallel lists there that
  don't auto-derive their width from `HEAD_LP_KEYS` and will silently
  produce wrong (not crashing) diagnostics, or in the `_old_per_head`
  slicing case, crash, if you add a head and forget them.
- **Physical behaviour is unchanged**: `kick_power_fraction`/`kick_spin`
  still feed the exact same downstream dict keys as before; `apply_nn_action.py`
  still hardcodes `Vector3.zero()` for spin regardless of what
  `gating.kick_spin` contains. This step was PPO-plumbing-only.

### 0.2 Why `kick_spin` was then immediately frozen (and `kick_power` was not)

Once `kick_spin` had a real distribution and real log_prob/entropy, it
started **costing** something for zero benefit: since spin still has no
physical effect, every bit of gradient/exploration noise spent on it is
pure waste, and worse, actively risks polluting the PPO ratio/KL for a
reason that isn't obvious on first read — worth spelling out because it
generalizes to any future "wire up the distribution but not the physics
yet" situation in this codebase:

**Freezing `kick_spin`'s own Linear weights + `kick_spin_log_std` via
`requires_grad_(False)` is necessary but NOT sufficient.**
`e_heads.kick_spin = self.kick_spin(h)` — even with the `kick_spin` Linear
layer's own weights frozen, its *output* keeps drifting every rollout,
because the shared trunk `h` (`ExecutionNetwork.forward()`'s `self.trunk(...)`
output, shared with `move_direction`/`kick_direction`/`kick_power`/every
other live head) keeps moving under gradients from every head that IS
still training. So the stored "old" sample vs. the recomputed "new" mean
at update time keep diverging purely as a side effect of unrelated heads
training — injecting spurious ratio/KL contributions from a head that's
supposed to be inert, exactly as if its log_std were slowly, silently
increasing on its own.

This is the **exact same problem** `_ppo_lp_masked_heads` was already built
to solve for `decision_net`'s phase-frozen heads (see `set_frozen_heads()`'s
docstring: *"explicit masking removes noisy log_prob contributions from
heads that carry no reward signal for this phase"*). The fix here is the
same pattern, not a new idea — freeze the parameters (stops gradient) AND
explicitly zero the term everywhere it's summed into a scalar loss/ratio
(stops trunk-drift noise from leaking in via a frozen-but-still-computed
value). Skipping the second half is the trap: `requires_grad=False` alone
*looks* like it should be enough, and silently isn't.

**`kick_power` was NOT frozen** — it has a real, live physical effect (kick
launch speed via `kick_with_direction()`), so its exploration/log_prob is
legitimate signal, not waste. Only `kick_spin` is a pure no-op right now.

### 0.3 What "frozen" means, concretely

Freeze is **unconditional / permanent**, not curriculum-phase-scoped —
`set_frozen_heads()`'s existing mechanism is phase-conditional because
`decision_net`'s frozen heads (shoot/pass/tackle/mark/hold in phase 1) are
expected to become live again in a later phase. `kick_spin` is different:
it's a hardcoded `Vector3.zero()` no-op in `apply_nn_action.py` regardless
of which phase is running, so there's no phase boundary where un-freezing
it would suddenly make sense — the only correct trigger to unfreeze it is
"real spin physics landed" (section 0.4), not a curriculum transition.

Two things freeze needs to touch, and they're structurally different:

1. **`execution_net.kick_spin`** — an `nn.Linear`, so
   `for p in execution_net.kick_spin.parameters(): p.requires_grad_(False)`
   works exactly like `set_frozen_heads()`'s existing loop.
2. **`execution_net.kick_spin_log_std`** — a **bare `nn.Parameter`**, not
   wrapped in a module. `getattr(execution_net, "kick_spin_log_std")` does
   NOT have a `.parameters()` method — calling `set_frozen_heads()`'s
   existing loop on it as-is will `AttributeError`. Needs its own
   `.requires_grad_(False)` call, or a small extension to whatever freezing
   helper is used so it can accept both raw `nn.Parameter`s and modules.

And the masking side (the actually load-bearing half, per 0.2): `kick_spin`
needs to be excluded from the summed loss/ratio in **all four** of
`_compute_log_prob`, `_recompute_log_prob`, `_per_head_new_log_probs`, and
`_compute_entropy` in `ppo_trainer.py` — not just the sampling side. Easiest
implementation is probably a plain `bool` (`self._kick_spin_frozen = True`,
set once in `__init__`, no curriculum plumbing) guarding each of the four
`kick_spin` blocks added in section 0.1, rather than trying to force this
into the existing `_ppo_lp_masked_heads` frozenset (that set's `in`/`not in`
checks are keyed by `decision_net` attribute name strings like
`"shoot_logit"` and threaded through `set_frozen_heads(list[str])`'s
phase-driven call sites in `train.py` — reusing it for a permanently-frozen
`execution_net` sub-head would work but conflates two conceptually
different kinds of "frozen" for no real benefit; a dedicated flag is
clearer to read later, at the cost of one more `if` per site).

**`kick_power` is deliberately left completely unaffected by any of this**
— no flag, no masking, still fully live per section 0.1.

### 0.4 How to re-enable `kick_spin` training (checklist, for whoever implements real spin physics)

This is the point where the rest of this document (§2 onward: axis+magnitude
output heads, the `topspin_axis`/`sidespin_axis` basis reconstruction in
`apply_nn_action.py`, the BC loss rewrite, `max_spin_rad_s` clamping) finally
gets implemented and `kick_spin` stops being a no-op. At that point:

1. **Remove the freeze**: undo whichever of `requires_grad_(False)` calls
   from §0.3 were added (grep `kick_spin` in `execution_network.py`'s
   construction / `ppo_trainer.py`'s `__init__` for whatever comment marks
   this — reference this section by name in that comment when you add it,
   so it's easy to find later).
2. **Remove the masking**: delete the `if not self._kick_spin_frozen:` (or
   equivalent) guards from `_compute_log_prob`/`_recompute_log_prob`/
   `_per_head_new_log_probs`/`_compute_entropy` added alongside the freeze.
3. **Decide the dimensionality question for real this time**: §2.1/§2.1.1
   argue for replacing the raw `(batch, 3)` `kick_spin` head with a 2D
   `kick_spin_axis` (coefficients over the `kick_direction`-derived
   `topspin_axis`/`sidespin_axis` basis) + `kick_spin_magnitude` scalar,
   rather than keeping the free 3-vector this frozen version used. If you
   do adopt that redesign, `_kick_spin_dist()` (the bespoke unsquashed
   Normal added in §0.1) gets replaced by the `DirectionHead`(dim=2) +
   `SquashedNormalHead` pair described in §2.3, not reused as-is — the
   whole point of the redesign is to stop wasting a degree of freedom on
   the physically-inert axis, which the current frozen 3-vector version
   still has (moot while frozen, not moot once live).
4. **Wire the BC loss properly**: §4.1's projection-onto-basis rewrite,
   including the `kick_precision` plumbing decision (§4.1 options (i) vs
   (ii) — re-recording demo datasets is a real cost, see §5).
5. **Augmentation — do not skip this, it's an easy silent-bug trap**:
   `ai/obs/augment.py`'s `_DIR_ACTION_KEYS` frozenset currently does **NOT**
   include `kick_power_raw` or `kick_spin_raw` — both fall through to the
   `else` branch (`flipped_actions[k] = v`, unchanged pass-through) under
   `flip_y` augmentation. This is **correct and requires no change for
   `kick_power`** (a true scalar, flip-invariant either way). It is
   **currently harmless for `kick_spin` only because it's frozen** — a
   frozen head's stored raw sample never gets compared against a
   recomputed log_prob in a way that matters (see §0.2, the ONLY reason it's
   safe to leave un-augmented right now is that it's excluded from every
   loss sum entirely). The moment `kick_spin` (or its `kick_spin_axis`
   replacement) is unfrozen, this becomes a real correctness bug: augmented
   samples would silently pair a flipped observation with an unflipped
   spin target, corrupting that head's log_prob under augmentation exactly
   the way `augment.py`'s existing pseudovector handling
   (`_BC_KICK_SPIN_X_COL`/`Y_COL`/`Z_COL`) already prevents for the BC
   label path. What to add depends on which representation you picked in
   step 3:
   - **Raw 3-vector kept as-is**: add `kick_spin_raw` to a pseudovector
     variant of the flip handling — negate indices 0 and 2 (x, z) under
     `flip_y`, mirroring `BALL_FLIP_Y_IDX`'s treatment of `ball.spin` — this
     is NOT the same rule `_DIR_ACTION_KEYS` uses (that negates index 1 for
     a *polar* vector), so `kick_spin_raw` cannot simply be added to
     `_DIR_ACTION_KEYS` as-is; it needs its own pseudovector-flip branch.
   - **2D axis+magnitude redesign adopted**: per this document's §2.1.1
     equivariance derivation (verified by direct computation, not just
     "should work"), the topspin coefficient (index 0) is flip-invariant
     and the sidespin coefficient (index 1) negates under `flip_y` — which
     numerically **is** exactly `_DIR_ACTION_KEYS`'s existing rule
     (`v2[:, 1] *= -1.0` under `flip_y`, index 0 untouched). In that case
     `kick_spin_axis_raw` CAN simply be added to `_DIR_ACTION_KEYS`
     directly. `kick_spin_magnitude_raw` needs no entry either way (scalar,
     flip-invariant, same as `kick_power_raw`).
   Either way: add a regression test. `augment.py` has no existing test
   coverage asserting a specific numeric sign-flip outcome for any
   `kick_spin*` action key (the BC-label path's `_BC_KICK_SPIN_*_COL`
   flipping IS tested — check `tests/ai_unit/` for the existing augmentation
   test file and follow its pattern for whichever key you add).
6. **Re-run the full AI test suite** (`uv run pytest tests/ai_unit
   tests/ai_scenario -v`), and specifically add one new test before
   unfreezing: assert `kick_spin`'s log_prob/entropy contribution is
   exactly `0.0` while frozen (protects against a future accidental
   double-unfreeze-by-omission), and a second test asserting it becomes
   nonzero/finite once unfrozen — the two tests should fail loudly if
   someone flips the freeze flag without also removing the masking, or
   vice versa (the exact "necessary but not sufficient" trap from §0.2).

---

## 1. Why the current design is broken

### 1.1 No physical clamp on the neural kick path

`engine/kicking.py` defines a precision-scaled cap:

```python
def max_spin_rad_s(params: KickingParams, kick_precision: float) -> float:
    """Maximum spin magnitude a player can impart, scaling with kick_precision."""
    return params.max_spin_base_rad_s + kick_precision * params.max_spin_precision_scale
```

`physics.json` currently sets `max_spin_base_rad_s=12.0`,
`max_spin_precision_scale=25.0` (note: `KickingParams` dataclass defaults
say `8.0`/`0.2` — those are only used if the physics.json key is
*absent*; the live values come from physics.json and are quite different
from the dataclass fallback, which is itself a smell — see §7.2 below).

This cap is applied for:

- The human player, via the UI drag-to-spin control:
  [input.py](../src/footballcoach/ui/input.py) lines ~303, ~428-433 —
  `max_spin = max_spin_rad_s(KickingParams.from_config(), player.attributes.kick_precision)`
  then `new_mag = max(0.0, min(max_spin, cur_mag + y * 0.5))`.

It is **not** applied anywhere on the neural network's kick path:

- `ExecutionNetwork.forward()` produces `kick_spin = self.kick_spin(h)` — a
  raw, unbounded 3-vector Linear layer output
  ([execution_network.py](../src/footballcoach/ai/models/execution_network.py) line ~179, ~289).
- `apply_action_to_player()` in
  [apply_nn_action.py](../src/footballcoach/ai/action/apply_nn_action.py)
  (pre-this-change) passed `Vector3(*gating.kick_spin)` straight into
  `player.kick_with_direction(...)`.
- `kick_with_direction()` → `kick_ball_from_direction()` in
  [kicking.py](../src/footballcoach/engine/kicking.py) (line ~521) does
  `ball.spin = spin` verbatim — **no clamp, no cap, no sanity check.**

So before this fix, an untrained/early-training network could in
principle output a spin vector of arbitrary magnitude (limited only by
weight init and gradient dynamics, not by game rules), and it would be
applied to the ball unmodified. This is inconsistent with every other
physical quantity the network outputs (`kick_power` is sigmoid-bounded to
`[0,1]` then scaled by `max_kick_speed_mps`; `kick_direction` is
L2-normalized to a unit vector).

### 1.2 Demonstration data teaches the network to output zero

The rules-based AI ([rules_ai.py](../src/footballcoach/rules_ai.py) line
~120) always arms kicks with `player.kick_armed_spin = Vector3.zero()`.
There is no other call site in `rules_ai.py` that sets any nonzero spin.
This means:

- Every BC label recorded from `Phase1RulesAI` demonstrations has
  `kick_spin = [0, 0, 0]` (see `BCLabel.kick_spin` population in
  [bc.py](../src/footballcoach/ai/ppo/bc.py) line ~252-264, sourced from
  `player.last_kick_spin`, which is only ever set to `Vector3.zero()` by
  the rules AI, or to whatever the (currently disabled) neural spin head
  produced).
- The BC MSE loss on `kick_spin` (see §4 below) is therefore trained
  against an all-zero target for 100% of demo rows. This actively
  *teaches* the network's `kick_spin` head to collapse toward zero output
  — which is one reason disabling it changes nothing behaviourally right
  now, and one reason the PPO exploration signal on this head is
  meaningless (there's no reward differential from spin yet, since ball
  physics changes from spin are subtle and not specifically rewarded).

### 1.3 No sampling / log_std / entropy for spin in PPO rollout collection

Compare the three continuous-ish execution heads:

| Head | Sampling distribution | log_std param | Used in ratio/KL? |
|---|---|---|---|
| `move_direction` | `DirectionHead` (Normal on raw 2-vec, then L2-normalize) | `move_dir_log_std` (config: `dir_log_std_init/min/max/target`) | Yes |
| `kick_direction` | `DirectionHead` (Normal on raw 3-vec, then L2-normalize) | `kick_dir_log_std` (config: `kick_dir_log_std_init`, falls back to `dir_log_std_*`) | Yes |
| `kick_power` | **Deterministic** — `torch.sigmoid(e_heads.kick_power)`, no distribution at all | `kick_power_log_std` exists as an `nn.Parameter` but is **never referenced** anywhere else in the codebase | No |
| `kick_spin` | **Deterministic** — raw output used directly, no distribution | `kick_spin_log_std` exists as an `nn.Parameter` but is **never referenced** anywhere else in the codebase | No |

See [ppo_trainer.py](../src/footballcoach/ai/ppo/ppo_trainer.py) lines
~2882-2922 (`_sample_action` or equivalent — the rollout action-sampling
block):

```python
kick_power_phys = float(torch.sigmoid(e_heads.kick_power))
kick_spin_raw = e_heads.kick_spin.squeeze(0)

execution_physical = {
    ...
    "kick_power_fraction": kick_power_phys,
    "kick_spin": kick_spin_raw.cpu().numpy(),
    ...
}
```

Neither line samples from a distribution or computes a log_prob. This
means:

- PPO has **no exploration** on spin (or, as it happens, on kick_power —
  a separate, smaller bug worth flagging but out of scope for this
  document beyond this note) beyond whatever raw-output noise exists from
  weight init and the optimizer.
- There is no PPO ratio/clipping/KL contribution from spin, so a large
  policy update could swing the (currently unclamped) spin output
  arbitrarily between rollouts with zero trust-region protection — this
  is the same class of instability problem `direction_max_grad_norm` and
  `dir_log_std_reg_coef` were introduced to solve for the direction heads
  (see `ai_config.json`'s `ppo` section), and spin currently has none of
  those safeguards.
- The two orphaned `nn.Parameter`s (`kick_power_log_std`,
  `kick_spin_log_std`) are dead weights sitting in every checkpoint,
  contributing to `optimizer.state` and to param-group grouping if
  someone assumes they're live — worth either wiring them up (this plan)
  or removing them if the decision is made to leave kick_power/kick_spin
  deterministic forever (not recommended — see §6).

### 1.4 Unit-vector + magnitude would be more consistent

Every other kick-shape quantity in this codebase is direction ⊕
magnitude, decomposed explicitly:

- `kick_direction` (unit 3-vector, L2-normalized) ⊕ `kick_power` (scalar
  in `[0,1]`, later multiplied by `max_kick_speed_mps(...)`).
- `move_direction` (unit 2-vector) ⊕ `sprint` (binary speed mode, itself
  bounded by `effective_top_speed(...)`).

`kick_spin` breaks this pattern: it's a single raw 3-vector head with no
normalization step and no scalar magnitude head. Two consequences:

1. **No natural clamp point.** With direction+magnitude, you clamp the
   magnitude scalar (e.g. `sigmoid(raw) * max_spin_rad_s(precision)`) and
   the direction is automatically unit-length. With a raw 3-vector, you'd
   have to clamp the vector's *norm* post-hoc (`v * min(1, cap/norm)`),
   which is more awkward to differentiate cleanly and interacts oddly
   with the BC MSE loss (regressing toward a clamped target vs an
   unclamped one).
2. **BC MSE loss is scale-blind to what actually matters.** The current
   loss (`bc.py` line ~648-661) normalizes both prediction and target by
   a flat constant (`spin_norm_max = 30.0`, matching
   `ai_config.json['observation']['ball_spin_norm_max_rad_s']`) and takes
   raw per-component MSE. This conflates "wrong axis" and "wrong
   magnitude" errors into one number and uses an observation-space
   normalization constant (chosen for *observing* the ball's spin, which
   can be higher than the kicker's max achievable kick spin) as if it
   were the action space's natural scale. A magnitude+axis decomposition
   would let you weight axis-cosine-error and magnitude-MSE-error
   separately, exactly as is already done for `kick_direction` (cosine
   loss) vs `kick_power` (MSE loss).

---

## 2. Target design

### 2.1 Network output: in-plane axis (2D) + magnitude, not a raw 3-vector

**Revised 2026-08-17** — the original version of this section kept
`kick_spin_axis` as a free 3D unit vector. That was wrong: a free 3D unit
vector is not constrained to be perpendicular to `kick_direction`, and the
component of spin *parallel* to the kick's velocity contributes exactly
zero to the Magnus force (`magnus_force = spin.cross(velocity) * ...` in
`ball_physics.py` — a vector crossed with a parallel vector is zero). So a
3D unit-axis head still leaves the network free to spend capacity and
exploration noise on a component that can never affect the ball's flight
— it fixes *unbounded magnitude* (the original problem) but not the
*wasted degree of freedom* (a distinct problem, discussed and dismissed in
conversation before this doc existed, and re-raised on review of this
plan).

The correct fix is to output only the **2 coefficients that span the plane
perpendicular to `kick_direction`** — this is not a loss of expressiveness
(topspin, backspin, and sidespin are all still fully representable at any
kick elevation, see §2.1.1) and it structurally guarantees the
physically-inert component is exactly zero rather than hoping training
suppresses it. `ui/kick_trajectory.py::spin_from_mouse()` already
implements this exact idea for the human kick UI — mouse angle `θ` around
the player selects a point on the unit circle of a 2-vector basis
(`topspin_axis`, `sidespin_axis`) that spans the plane ⊥ to the kick's aim
direction, and mouse distance sets the magnitude. This section adopts the
same pattern for the network, generalized to handle elevation (§2.1.1)
since the UI's version is a flat/horizontal approximation (see the
caveat at the end of §2.1.1).

Replace:

```python
self.kick_spin = nn.Linear(trunk_hidden, 3)             # raw spin vector
```

with two heads:

```python
self.kick_spin_axis = nn.Linear(trunk_hidden, 2)         # raw 2-vector (in-plane coeffs), L2-normalized to a unit 2-vector in forward()
self.kick_spin_magnitude = nn.Linear(trunk_hidden, 1)    # raw scalar; sigmoid -> [0, 1] fraction of max_spin_rad_s
```

In `forward()`, alongside the existing `kick_direction` normalization:

```python
raw_spin_axis = self.kick_spin_axis(h)  # (batch, 2) — coefficients over (topspin_axis, sidespin_axis)
...
return ExecutionHeadsRaw(
    ...
    kick_spin_axis=raw_spin_axis / (raw_spin_axis.norm(dim=-1, keepdim=True) + eps),  # (batch, 2) unit vector [cos_theta, sin_theta]
    kick_spin_magnitude=self.kick_spin_magnitude(h),  # raw; sigmoid applied by caller, matches kick_power convention
    ...
)
```

Note the network never sees or constructs the 3D basis vectors themselves
— it only ever outputs 2 numbers (an in-plane angle, represented as a unit
2-vector for the same reasons `move_direction`/`kick_direction` are). The
3D reconstruction happens once, downstream, in `apply_nn_action.py`
(§2.2), where `kick_direction` (this same network's own other output) is
already available to build the basis from.

This still mirrors the existing `kick_direction` (unit vector, computed
in-network) / `kick_power` (raw scalar, sigmoid applied by the caller at
sample time, see `ppo_trainer.py` line ~2912
`kick_power_phys = float(torch.sigmoid(e_heads.kick_power))`) split — just
at 2 output dims instead of 3 for the axis half.

#### 2.1.1 Basis construction (handles elevation correctly; UI does not)

The UI's `topspin_axis = rotate_90_ccw(aim_dir_xy)` is only exactly
perpendicular to the *actual* 3D kick direction when the kick is flat
(`elevation_angle_rad = 0`) — `aim_dir` in `spin_from_mouse()` is built
from the horizontal `AIM_XY` phase only, before the separate `AIM_Z`
(elevation) phase runs, so for any lofted human kick today the UI's spin
basis is a small approximation, not exact. This has presumably gone
unnoticed because lofted spin kicks are rare and the resulting curve error
is subtle.

For the network, do it exactly instead, using the *actual* 3D
`kick_direction` (which can have a nonzero z-component from elevation).
Given unit vector `d = kick_direction`:

```python
world_up = Vector3(0.0, 0.0, 1.0)
topspin_axis = d.cross(world_up).normalized()      # horizontal, guaranteed ⊥ d for any d (property of cross product)
sidespin_axis = topspin_axis.cross(d).normalized()  # ⊥ both d and topspin_axis; tilts with elevation
```

**Corrected 2026-08-17 (implementation pass)** — an earlier draft of this
section had `topspin_axis`/`sidespin_axis` swapped. Checked against the
UI: `spin_from_mouse()`'s `topspin_axis = (-aim_dir_y, aim_dir_x, 0)` is
the **horizontal**, in-plane-rotated-90° vector (matches `d.cross(world_up)`
above, up to an inconsequential sign), and its `sidespin_axis = (0,0,1)`
is the **fixed vertical** one (matches `topspin_axis.cross(d)` reducing to
world-z at zero elevation). Get this backwards and the physical labels
invert: spin about the *horizontal* axis ⊥ to travel produces a *vertical*
Magnus force (dip/lift = topspin/backspin); spin about the *vertical* axis
produces a *horizontal* Magnus force (curl = sidespin) — so the horizontal
cross-product vector must be named `topspin_axis`, not `sidespin_axis`.

`topspin_axis` reduces to the UI's (rotated) `topspin_axis` direction
when `d` is horizontal — the two schemes agree exactly at zero elevation,
they only diverge for lofted kicks, where the network's version stays
exact and the UI's stays approximate. (Not proposing to fix the UI's
approximation in this plan — flagging it only so nobody assumes the UI's
formula can be reused verbatim for the network path; it can't, for lofted
kicks specifically.) Degenerate case: `d` parallel to `world_up` (a
perfectly vertical kick) makes `d.cross(world_up)` zero — not physically
reachable given `max_loft_angle_deg` constraints, but guard with a fallback
axis (e.g. world `+x`) rather than dividing by ~0, same defensive pattern
already used elsewhere in this codebase (`+ eps` on every other
normalization in this plan).

**Equivariance under `flip_y` — now verified by direct computation, not
just "should work" (see the abandoned equivariance note further down in
this doc, superseded by this box):** with `d' = (d_x, -d_y, d_z)` (the
mirrored kick direction) and both axes re-derived fresh from `d'` via the
same formulas, `topspin_axis(d')` equals the **pseudovector transform** of
`topspin_axis(d)` (x,z negate, y unchanged) — exactly like `spin_x`/`spin_z`
in `BALL_FLIP_Y_IDX` — while `sidespin_axis(d')` equals the **polar-vector**
transform of `sidespin_axis(d)` (only y negates) — like `kick_direction`
itself. Working through what this means for the actual spin vector
`spin = mag·(cos_t·topspin_axis + sin_t·sidespin_axis)` under a true
physical mirror: the **topspin coefficient (`cos_t`) is unchanged**, and
the **sidespin coefficient (`sin_t`) negates**. This matches physical
intuition directly — a ball's forward/backward tumble (topspin/backspin)
looks identical mirrored; a ball curling right becomes a ball curling left
in the mirror (sidespin flips handedness). Concretely: if the raw 2-vector
is stored as `[cos_t, sin_t]` (index 0 = topspin, index 1 = sidespin),
`flip_y` must negate **index 1 only** — which is exactly what
`augment.py`'s existing `_DIR_ACTION_KEYS` flip rule already does
(`v2[:, 1] *= -1.0` under `flip_y`), so `kick_spin_axis_raw` can reuse that
existing mechanism directly rather than needing bespoke flip code. (Not
verified for `flip_x` — currently moot, `flip_x` is excluded from
`_FLIP_VARIANTS` per the canonical-AI-frame note elsewhere in this
codebase — flag as unverified if `flip_x` augmentation is ever
reintroduced.)

### 2.2 Where the magnitude gets scaled to physical units

Scaling by `max_spin_rad_s(kick_precision)` must happen **after**
sampling and **using the actual kicking player's `kick_precision`
attribute**, which is not something the network should need to know (the
observation `PlayerFeatures.kick_power` at index reference in
`obs/schema.py` line ~61 already exists, but there's no direct
`kick_precision` scalar fed to the execution network as an observation —
check this before assuming it's visible input; if it isn't, the network
predicts a *fraction* of the player's own current max, and the engine
applies the real physical cap using the player's actual attribute value,
which is the right layering: the network shouldn't need to "know" the
formula, it just says "as much spin as I can, or half, or a specific
fraction" and the engine enforces what that means physically).

Concretely, in `apply_action_to_player()`
([apply_nn_action.py](../src/footballcoach/ai/action/apply_nn_action.py)):

```python
from footballcoach.engine.kicking import KickingParams, max_spin_rad_s

if gating.kick_this_tick:
    kick_dir = gating.kick_direction  # unit Vector3, this same network's kick_direction output
    ...
    cos_t, sin_t = gating.kick_spin_axis  # (2,) unit vector: coeffs over (topspin_axis, sidespin_axis)
    spin_magnitude_frac = gating.kick_spin_magnitude_fraction  # already sigmoid'd, in [0, 1]

    world_up = Vector3(0.0, 0.0, 1.0)
    _top_raw = kick_dir.cross(world_up)
    if _top_raw.length() > 1e-6:
        topspin_axis = _top_raw.normalized()
    else:
        topspin_axis = Vector3(1.0, 0.0, 0.0)  # degenerate near-vertical-kick fallback, see §2.1.1
    sidespin_axis = topspin_axis.cross(kick_dir).normalized()

    if spin_magnitude_frac > 1e-6:
        max_spin = max_spin_rad_s(match.kicking_params, player.attributes.kick_precision)
        spin_dir_unit = topspin_axis * cos_t + sidespin_axis * sin_t  # already unit length: cos_t^2+sin_t^2=1, axes orthonormal
        spin_vec = spin_dir_unit * (spin_magnitude_frac * max_spin)
    else:
        spin_vec = Vector3.zero()
    player.kick_with_direction(
        match,
        direction_3d,
        float(gating.kick_power_fraction) if gating.kick_power_fraction > 0 else 0.85,
        spin_vec,
    )
```

Basis construction is duplicated between here and §2.1.1 conceptually —
consider factoring `kick_spin_basis(kick_direction: Vector3) -> tuple[Vector3, Vector3]`
into a shared helper (e.g. `ai/action/apply_nn_action.py` itself, or a
small physics-adjacent module both it and any future BC/eval code can
import) rather than inlining the cross-product math at every call site —
worth deciding at implementation time, not blocking for this plan.

Note `match.kicking_params` — confirm this attribute name exists on
`Match` (it's referenced as `match.kicking_params` in
`player.kick_direct()` already, e.g.
[player.py](../src/footballcoach/entities/player.py) line ~220, so this
is a safe, already-used accessor).

This is the single most important correctness fix in this whole plan: **it
guarantees the neural network can never exceed the same physical spin cap
the human player and rules-based AI (implicitly, by using zero) are
bound by**, closing the gap identified in the user's own todo list:
_"do we have 2 values of max spin, in UI and engine? BAD."_ — after this
change there is exactly one call site (`max_spin_rad_s`) and exactly one
set of config constants (`physics.json`'s `kicking.max_spin_base_rad_s` /
`max_spin_precision_scale`) governing every kick path (human, rules AI,
neural AI).

### 2.3 Sampling distribution for PPO

**Axis:** reuse `DirectionHead` exactly as `kick_direction` already does
— it already supports arbitrary vector dimensionality (constructed with
`raw_vector: torch.Tensor` of shape `(..., 3)` for `kick_direction`; use
`(..., 2)` for `kick_spin_axis` — confirm `DirectionHead`'s log_prob/
entropy math doesn't hardcode dim=3 anywhere, e.g. in a normalizing
constant; it shouldn't, since it's meant to be dimension-generic, but
verify before assuming). Add a new learnable log_std parameter:

```python
self.kick_spin_axis_log_std = nn.Parameter(torch.full((1,), kick_spin_axis_log_std_init))
```

**Magnitude:** reuse `SquashedNormalHead` (already used for other
squashed-scalar action heads — check for existing usage patterns of
`SquashedNormalHead` in `ppo_trainer.py`; if it isn't yet used for
`kick_power`, treat `kick_power` as needing the *same* fix in a follow-up
— see §6). Construct with `low=0.0, high=1.0, squash="sigmoid"`:

```python
self.kick_spin_magnitude_log_std = nn.Parameter(torch.full((1,), kick_spin_magnitude_log_std_init))
```

In `ppo_trainer.py`'s rollout-collection sampling block (the code shown
in §1.3 above), replace the deterministic block with:

```python
log_std_spin_axis = self.execution_net.kick_spin_axis_log_std
log_std_spin_mag = self.execution_net.kick_spin_magnitude_log_std

spin_axis_head = self._kick_spin_axis_head(e_heads.kick_spin_axis, log_std_spin_axis)
spin_mag_head = SquashedNormalHead(
    mean=e_heads.kick_spin_magnitude, log_std=log_std_spin_mag,
    low=0.0, high=1.0, squash="sigmoid",
)

if det_direction:
    spin_axis_raw = spin_axis_head.mode_physical()          # (1, 2)
    spin_mag_raw = spin_mag_head.mode_physical()            # (1, 1) already in [0,1]
else:
    spin_axis_raw = spin_axis_head.sample_raw()              # (1, 2), pre-normalized already inside DirectionHead
    spin_mag_raw = spin_mag_head.to_physical(spin_mag_head.sample_raw())

spin_axis_phys = (spin_axis_raw / (spin_axis_raw.norm(dim=-1, keepdim=True) + eps)).squeeze(0)
spin_magnitude_frac = float(spin_mag_raw.squeeze())

execution_physical = {
    ...
    "kick_spin_axis": spin_axis_phys.cpu().numpy(),
    "kick_spin_magnitude_fraction": spin_magnitude_frac,
    ...
}
```

Add a `_kick_spin_axis_head` helper next to the existing
`_move_dir_head`/`_kick_dir_head` helper methods (grep for those names in
`ppo_trainer.py` to find the pattern — they construct a `DirectionHead`
from a raw tensor + log_std with the class-configured
`log_std_min`/`log_std_max` clamp bounds).

**Log-prob storage for the PPO ratio:** both new heads need their
`log_prob()` included wherever `kick_direction`'s and `kick_power`'s
log_probs are currently summed into the total action log_prob for the
rollout buffer, and wherever the PPO update recomputes new log_probs for
the ratio. Search `ppo_trainer.py` for every place `kick_direction`'s
`log_prob` is referenced (both in the rollout-collection forward pass and
in `_ppo_update()`'s recomputation) and mirror each site for
`kick_spin_axis`/`kick_spin_magnitude`. Do not skip the `_ppo_update()`
side — a head that has a distribution at sampling time but no log_prob
in the update will silently contribute zero gradient from PPO (it'll
only move via the BC aux loss and entropy bonus, which is not what you
want for a supposedly-explored head).

### 2.4 Entropy bonus

`ent_dir_weight` in `ai_config.json` currently scales `move_dir`/`kick_dir`
entropy contribution relative to the Bernoulli heads. Decide whether spin
axis/magnitude entropy should:

- (a) Share `ent_dir_weight` (simplest — spin axis really is "just
  another direction head"), or
- (b) Get its own `ent_spin_weight` (more control, more config surface).

Recommendation: **(a) for the axis, dedicated new coefficient for the
magnitude.** The axis head is architecturally identical to
`kick_direction` so it should behave identically for entropy purposes.
The magnitude head is architecturally a `SquashedNormalHead`, a new
distribution *type* in the entropy bonus (if `kick_power`'s entropy isn't
already contributing — check `_ppo_update()`'s entropy-bonus computation
to see whether `SquashedNormalHead`-typed heads are summed in at all
currently; if `kick_power` was never actually sampled/entropy'd either,
you're establishing the pattern for both here). Add
`ent_spin_magnitude_weight` (default something small, e.g. `0.02`,
consistent in spirit with `ent_dir_weight=0.05`).

---

## 3. Config additions (`ai_config.json`)

Add to the `network` section, next to the existing dir_log_std_* keys:

```json
"kick_spin_axis_log_std_init": -2.2,
"_comment_kick_spin_axis_log_std_init": "Initial log_std for the kick_spin_axis DirectionHead (isotropic Normal on the raw 2-vector, before L2-normalize, over the (topspin_axis, sidespin_axis) basis derived from kick_direction -- see plan section 2.1.1, NOT a raw 3-vector). Mirrors kick_dir_log_std_init; separate parameter so spin axis exploration can be tuned independently of kick_direction exploration. Falls back to dir_log_std_init if absent.",
"kick_spin_axis_log_std_target": -1.8,
"kick_spin_axis_log_std_reg_coef": 0.0,
"_comment_kick_spin_axis_log_std_reg": "L2 restoring force toward kick_spin_axis_log_std_target, mirrors dir_log_std_reg for move/kick direction heads. 0.0 = disabled.",

"kick_spin_magnitude_log_std_init": -2.0,
"_comment_kick_spin_magnitude_log_std_init": "Initial log_std for the kick_spin_magnitude SquashedNormalHead (sigmoid-squashed scalar in [0,1], fraction of max_spin_rad_s(kick_precision)). No existing analogue since kick_power was never actually given a sampling distribution -- see agent_plans/spin_implementation_plan.md section 6 for the parallel kick_power fix this motivates."
```

Add to the `ppo` section, next to `ent_dir_weight`:

```json
"ent_spin_magnitude_weight": 0.02,
"_comment_ent_spin_magnitude_weight": "Entropy bonus weight for the kick_spin_magnitude SquashedNormalHead. kick_spin_axis reuses ent_dir_weight (same architecture as move_dir/kick_dir -- a DirectionHead). See spin_implementation_plan.md."
```

No new key is needed for the physical cap itself — `max_spin_rad_s()`
already reads `max_spin_base_rad_s`/`max_spin_precision_scale` from
`physics.json`'s `kicking` section, and this plan reuses that single
source of truth rather than duplicating it into `ai_config.json`.
**Do not** add a second spin-cap constant into `ai_config.json` — that
would reintroduce exactly the "two values of max spin" bug this plan is
trying to close.

---

## 4. BC label / loss changes (`ai/ppo/bc.py`)

### 4.1 Label layout

The current flat label layout (see the module docstring in `bc.py`) has:

```
[21] kick_spin_x
[22] kick_spin_y
[23] kick_spin_z
```

Two options:

**Option A (recommended): keep storing spin as a raw 3-vector in the
label, decompose only in the loss function.** This avoids changing
`BC_LABEL_DIM` (currently `25`) or any indices, avoids touching
`augment.py`'s pseudovector flip logic (which already correctly handles
`_BC_KICK_SPIN_X_COL`/`Y_COL`/`Z_COL` as a pseudovector under flips — see
`augment.py` lines ~140-146 and ~257-264/397-407), and avoids touching
`DemonstrationDataset` parsing in `bc/dataset.py`. The loss function
derives axis + magnitude from the stored raw vector at loss-computation
time:

**Equivariance note (only relevant because of the §2.1 axis-dimensionality
change, doesn't apply to the original 3D-axis version of this plan) —
verified by direct computation in §2.1.1, not just "should be fine by
construction":** `augment_batch()`'s `flip_y` variant negates the
y-components of both `kick_direction` and `kick_spin` (the latter as a
pseudovector — spin_x/z negate, per `ai/knowledge.md`) *before* this loss
code ever runs, so `label_kick_dir` and `target_spin_raw` above are always
already-flipped, consistent world-frame vectors by the time the basis is
built. §2.1.1 works through the actual transform: the topspin coefficient
(`target_topspin_component`) is invariant under `flip_y`, the sidespin
coefficient (`target_sidespin_component`) negates — so
`target_spin_axis_2d[:, 0]` (topspin) is unchanged and
`target_spin_axis_2d[:, 1]` (sidespin) negates, matching exactly how
`augment.py` already flips `kick_spin_axis_raw` via the existing
`_DIR_ACTION_KEYS` mechanism (§2.1.1's tail note). Still add a regression
test asserting this end-to-end (a known label run through
`augment_batch(..., flip_y)` should produce a `target_spin_axis_2d` whose
column 0 is unchanged and column 1 is negated) — the derivation is now
verified on paper, not in code, and this is exactly the kind of
silently-always-zero-today code path (§5.1: no current demo has nonzero
spin) that stays untested for a long time by default.

The BC label stays a raw 3-vector (world-frame `kick_spin`), and the loss
now projects it onto the **label's own** `(topspin_axis, sidespin_axis)`
basis — built from the label's `kick_direction` (already stored at indices
18-19/24, see the BC label table) using the exact same cross-product
construction as §2.1.1/§2.2, so target and prediction live in the same
2D coordinate system before comparing:

```python
# --- Execution: kick_spin_axis (cosine, 2D) and kick_spin_magnitude (MSE) ---
kicked_mask = labels[:, _I_KICK_THIS_TICK] > 0.5
if kicked_mask.any():
    target_spin_raw = labels[:, _I_KICK_SPIN_X:_I_KICK_SPIN_Z + 1]  # (N, 3), raw rad/s, world frame
    target_spin_mag = target_spin_raw.norm(dim=-1)                  # (N,)
    has_spin = target_spin_mag > 1e-6

    # Build the same 2D basis used at apply-time (§2.1.1), from the
    # label's OWN kick_direction (indices 18-19/24) -- not the network's
    # kick_direction prediction, since this is the BC *target* side.
    eps = 1e-6
    label_kick_dir = labels[:, [_I_KICK_DIR_X, _I_KICK_DIR_Y, _I_KICK_DIR_Z]]  # (N, 3), already unit
    world_up = torch.tensor([0.0, 0.0, 1.0], device=labels.device).expand_as(label_kick_dir)
    top_raw = torch.cross(label_kick_dir, world_up, dim=-1)
    top_norm = top_raw.norm(dim=-1, keepdim=True)
    topspin_axis = torch.where(top_norm > eps, top_raw / top_norm.clamp_min(eps),
                                torch.tensor([1.0, 0.0, 0.0], device=labels.device))
    sidespin_axis = torch.cross(topspin_axis, label_kick_dir, dim=-1)  # already unit (both inputs orthonormal)

    # Project the raw 3D target spin onto this 2D basis. Any residual
    # component of target_spin_raw parallel to label_kick_dir (should be
    # ~0 for real kicks, since that's the physically-inert axis, but real
    # recorded data / float noise may not be exact) is silently dropped --
    # this is intentional: BC is teaching the physically-relevant subspace
    # only, not the noise floor of the recording pipeline.
    target_topspin_component = (target_spin_raw * topspin_axis).sum(dim=-1)   # (N,)
    target_sidespin_component = (target_spin_raw * sidespin_axis).sum(dim=-1) # (N,)
    target_spin_axis_2d = torch.stack([target_topspin_component, target_sidespin_component], dim=-1)
    target_spin_axis_2d = target_spin_axis_2d / target_spin_mag.clamp_min(eps).unsqueeze(-1)

    pred_spin_axis = exec_heads.kick_spin_axis / (exec_heads.kick_spin_axis.norm(dim=-1, keepdim=True) + eps)  # (N, 2)
    spin_axis_cos_loss = 1.0 - (pred_spin_axis * target_spin_axis_2d).sum(dim=-1)
    spin_axis_loss_per = direction_loss_weight * torch.where(
        kicked_mask & has_spin, spin_axis_cos_loss, torch.zeros_like(spin_axis_cos_loss)
    )
    loss += exec_weight * spin_axis_loss_per

    # Magnitude MSE loss, normalized by the SAME cap the engine enforces --
    # NOTE this now needs kick_precision, which is not currently threaded
    # through BCLabel. See "kick_precision plumbing" note below.
    target_spin_mag_frac = (target_spin_mag / max_spin_cap).clamp(0.0, 1.0)
    pred_spin_mag_frac = torch.sigmoid(exec_heads.kick_spin_magnitude.squeeze(-1))
    spin_mag_mse = (pred_spin_mag_frac - target_spin_mag_frac) ** 2
    spin_mag_loss_per = torch.where(kicked_mask, spin_mag_mse, torch.zeros_like(spin_mag_mse))
    loss += exec_weight * spin_mag_loss_per
```

**Why this is safe today, and what to watch for later:** every current
demo row has `target_spin_raw = [0,0,0]` (rules AI never spins kicks, see
§5.1), so `has_spin` is `False` everywhere and this whole block is
currently a no-op in practice — the 2D-projection machinery only starts
mattering once either (a) a human-recorded demo with spin is added, or
(b) PPO's own on-policy exploration starts generating nonzero spin, which
isn't BC-supervised at all (BC only sees demo-recorded rows). Don't take
"BC loss is ~0 on this term" as validation that the projection math is
correct — write a dedicated unit test with a synthetic label that has a
known `kick_direction` + `kick_spin` and assert the recovered 2D
coefficients match a hand-computed expectation, since this is exactly the
kind of silently-always-zero code path that hides bugs for a long time
(cf. `ai/knowledge.md`'s "Orders vs execution-network labels boundary"
bug history — this codebase has been bitten by silently-untested
label-derivation code before).

**kick_precision plumbing problem:** `max_spin_cap` above needs
`max_spin_rad_s(kicking_params, kick_precision)` computed **per-row**,
since different demo rows come from different randomly-generated players
with different `kick_precision` attributes (see
`phase1_scenario`/attribute randomization in `ui/scenarios.py`). The
current `BCLabel` dataclass does not carry `kick_precision`. You must
either:

  (i) Add a new field `kick_precision: float` to `BCLabel`
      ([bc.py](../src/footballcoach/ai/ppo/bc.py) — the dataclass right
      after the module docstring), populate it in `phase1_labels()`
      (wherever `kick_spin`/`kick_power` are currently read off
      `player.last_kick_spin`/`last_kick_power_fraction`, add
      `kick_precision=player.attributes.kick_precision`), add it to
      `to_array()`/the flat layout (bump `BC_LABEL_DIM` to `26`, new
      index `_I_KICK_PRECISION = 25`), and update `augment.py`'s
      docstring comment about "hardcoded magic numbers... cross-check
      whenever BC_LABEL_DIM's layout changes" (it is NOT position/velocity
      so needs no flip handling, but the comment block should be updated
      to note the new trailing field exists), **or**

  (ii) Simplify and normalize magnitude by a **fixed constant** instead of
      the per-row precision-dependent cap (e.g. reuse
      `ball_spin_norm_max_rad_s=30.0` as today, just applied to the
      *magnitude* rather than per-component). This is simpler but
      reintroduces a mismatch between the BC-loss normalization constant
      and the actual physical cap the engine will apply at inference —
      acceptable ONLY if you also decide the network should learn to
      "leave headroom" naturally rather than being taught the exact
      physical boundary. **Not recommended** — prefer (i), it's a small,
      mechanical addition and keeps the BC loss target consistent with
      what will actually happen physically at inference time.

Recommendation: **do (i).** It's the more correct fix and this codebase's
existing conventions (e.g. `ai_type`/`opponent_ai_type` being added to
`BCLabel` as trailing fields, per the docstring's own layout table)
already establish the pattern of extending `BCLabel` with new trailing
fields when new information is needed for training.

**Option B (more invasive, not recommended for this pass): change the
recorded label itself to axis+magnitude at record time.** Rejected
because it changes `augment.py`'s pseudovector-flip math (magnitude is
flip-invariant, only the axis needs the pseudovector transform — doable,
but touches more surface area for no real benefit over Option A, which
already gets the geometry right at loss-computation time without
touching the label schema's flip semantics).

### 4.2 Loss breakdown dict changes

Update the `return_breakdown` dict (see `bc.py` around line ~712-713 and
the zero-loss early-return around line ~559) to replace `"kick_spin"`
with `"kick_spin_axis"` and `"kick_spin_magnitude"` (two separate
tracked losses, exactly like `"kick_direction"`/`"kick_power"` are
already tracked separately rather than combined).

### 4.3 `EXEC_HEAD_MODULES` in `ppo_trainer.py`

Update the module-attribute mapping used for per-head gradient-norm
diagnostics ([ppo_trainer.py](../src/footballcoach/ai/ppo/ppo_trainer.py)
line ~106):

```python
EXEC_HEAD_MODULES: list[tuple[str, str]] = [
    ...
    ("kick_power", "kick_power"),
    ("kick_spin_axis", "kick_spin_axis"),
    ("kick_spin_magnitude", "kick_spin_magnitude"),
    ...
]
```

And the two `_kick_bkdn_keys` sets at lines ~1891 and ~2160:

```python
_kick_bkdn_keys = {"kick", "kick_direction", "kick_power", "kick_spin_axis", "kick_spin_magnitude"}
```

---

## 5. Demonstration recording changes

### 5.1 Rules-based AI still kicks with zero spin — is that a problem?

**No — this is fine, and you do NOT need to make the rules-based AI
impart spin for this plan.** The rules-based AI's job is to demonstrate
sound *positioning, timing, and shot selection*, not spin technique — real
footballers mostly don't intentionally curl every pass or shot either.
Leaving `rules_ai.py`'s `kick_armed_spin = Vector3.zero()` alone means:

- BC continues to teach "zero spin is a perfectly valid default" (true —
  it is), via the `has_spin` gating in the loss (§4.1) which contributes
  zero axis loss on all-zero-spin rows exactly like `has_kick_dir` already
  skips direction loss on non-kick rows.
- PPO's own exploration (via the newly-added sampling distribution, §2.3)
  is what teaches the network *when* nonzero spin might help — which is
  appropriate, since "should I curl this shot around the keeper" is
  exactly the kind of decision PPO's reward-driven exploration is suited
  to, and BC/demonstration is not (there's no rules-based heuristic for
  "impart topspin to dip a long-range shot" worth hand-coding right now).

### 5.2 What DOES need re-recording

Nothing needs re-recording purely for the *label schema* extension in
§4.1(i) UNLESS you already have a large existing demonstration dataset
you plan to reuse (e.g. `demonstrations/phase1_long/`, referenced in the
active training run in this session's terminal history) and want the new
`kick_precision` field populated for it.

`DemonstrationDataset` in `ai/bc/dataset.py` already has an explicit
schema-version guard — see the comment referenced in the earlier
spin-audit:

```
# (currently 24, after kick_direction/kick_power/kick_spin fields were
# older BC label schema (e.g. missing kick_direction/kick_power/
# kick_spin fields added for full execution-head BC coverage).
```

This confirms the dataset loader already has precedent for detecting and
rejecting stale-schema `.npz` files by array width. When you bump
`BC_LABEL_DIM` from `25` to `26` (§4.1), **all existing recorded
demonstrations under `demonstrations/` become schema-stale** and will
either be rejected by the loader's width check or (worse, if the check
isn't strict) silently misaligned. You must:

1. Confirm the exact guard condition in `ai/bc/dataset.py` (search for
   where `BC_LABEL_DIM` or a hardcoded `25` is checked against loaded
   array shape) and confirm it errors loudly rather than truncating/
   padding silently.
2. **Re-record all demonstration sets** used by any training run that
   will use this updated code, including at minimum:
   - `demonstrations/phase1_long/` (currently in active use per this
     session's terminal history — the exact command used was:
     ```bash
     uv run python -m footballcoach.ai.scripts.record_demonstrations \
         --phase 1 --n-episodes 42000 --episodes-per-file 8 \
         --output demonstrations/phase1_long/ --seed 42
     ```
     Re-run this exact command (same episode count/seed for
     reproducibility) after the `BCLabel`/`BC_LABEL_DIM` change lands.
   - `demonstrations/phase_1_debug_large/` (used by
     `debug_value_network.py` per this session's terminal history):
     check `record_demonstrations.py --info` output for the episode count
     originally used, or just re-record with a similar count (a few
     thousand should suffice for a *debug* value-network sanity script;
     check the script's own recommended size in its `--help` or
     `ai_trainer_knowledge.md`'s Tier-1 file list).
   - `demonstrations/phase_1_debug/`, `demonstrations/phase_1_smoketest/`
     (smaller sets, likely used by tests — grep `tests/` for references
     to these directory names to confirm which test files depend on them
     before deciding whether they need re-recording or are fixture-frozen
     on purpose).
3. Any checkpoint's `checkpoint_pretrained.pt` that was produced via
   `--bc-dataset` pointing at the OLD-schema demonstrations must be
   considered stale for BC-loss purposes (the network weights themselves
   are still valid tensors — nothing about the *model* schema changes,
   only the *label* schema — so old checkpoints will still *load* fine
   via `--from-pretrained`/`--checkpoint`/`--latest-pretrain`; they just
   were never trained with the new `kick_spin_axis`/`kick_spin_magnitude`
   heads' BC supervision, which didn't exist in the model class at their
   save time either — see §6 "Checkpoint compatibility" below, this is
   actually a *model architecture* change too since new `nn.Linear` output
   heads and new `nn.Parameter` log_stds are being added).

### 5.3 Terminal commands to run after implementing this plan

```bash
# 1. Re-record the long demonstration set used for the main training run
#    (same seed/episode-count as the pre-existing set for comparability)
uv run python -m footballcoach.ai.scripts.record_demonstrations \
    --phase 1 --n-episodes 42000 --episodes-per-file 8 \
    --output demonstrations/phase1_long/ --seed 42

# 2. Re-record the smaller debug set used by debug_value_network.py
#    (check record_demonstrations.py --info against the OLD
#    demonstrations/phase_1_debug_large/ dir first to see the original
#    episode count before deleting it, so the new set matches in size)
uv run python -m footballcoach.ai.scripts.record_demonstrations \
    --phase 1 --n-episodes 0 --output demonstrations/phase_1_debug_large/ --info
# (delete/rename the old dir once you've noted the count, then re-record
#  with the same --n-episodes)

# 3. Run the AI test suite -- this MUST pass before any new training run,
#    since it's the fastest signal that the label-schema bump didn't break
#    anything (BC_LABEL_DIM mismatches, augment.py flip index drift, etc.)
uv run pytest tests/ai_unit tests/ai_scenario -v

# 4. Fresh BC pretrain + PPO training run against the re-recorded set.
#    Do NOT reuse --latest-pretrain / --from-pretrained pointing at a
#    checkpoint saved before this change -- the execution network's
#    output-head shapes changed (new kick_spin_axis/kick_spin_magnitude
#    Linear layers replace the old kick_spin Linear layer), so old
#    state_dicts will fail to load (missing/unexpected keys) or, worse,
#    partially load with load_state_dict(strict=False) silently dropping
#    the new heads -- start this run WITHOUT --latest-pretrain/--checkpoint:
uv run python -m footballcoach.ai.scripts.train \
    --phase 1 --seed 42 \
    --total-steps 6000000 --separate-value-net \
    --bc-dataset demonstrations/phase1_long/ 2>&1 | tee -a training_runs.md
```

---

## 6. Follow-up: `kick_power` has the exact same deterministic-sampling bug

This plan focused on `kick_spin` because that's what was asked about, but
§1.3's table shows `kick_power` is in an identical state: a
`kick_power_log_std` parameter exists
([execution_network.py](../src/footballcoach/ai/models/execution_network.py)
line ~237) but is **never used** — `kick_power` is applied deterministically
via `torch.sigmoid(e_heads.kick_power)` with no sampling, no log_prob, no
PPO ratio contribution, no entropy bonus. This is a pre-existing,
separate bug with the same shape as the spin bug, and this document's
`SquashedNormalHead`-based fix for `kick_spin_magnitude` (§2.3) is
*exactly* the fix `kick_power` also needs — construct a
`SquashedNormalHead(mean=e_heads.kick_power, log_std=self.execution_net.kick_power_log_std, low=0.0, high=1.0, squash="sigmoid")`,
sample from it during rollout collection, and thread its log_prob through
the PPO update the same way.

**Recommendation: do NOT bundle this into the same PR/commit as the spin
work**, to keep the diff reviewable and the two changes independently
revertible if either one destabilises training — but implement it as an
immediate follow-up using this document as the template, since by the
time spin is done you'll have just built and tested the exact pattern
`kick_power` needs.

---

## 7. Secondary observations / cleanup opportunities found while researching this plan

These are not required for the spin fix to work, but were noticed during
the investigation and are worth a separate look:

### 7.1 `kick_power_log_std`/`kick_spin_log_std` dead parameters

If for some reason this plan is deprioritized and spin/power stay
deterministic, at minimum delete the two currently-unused
`nn.Parameter`s (`kick_power_log_std`, `kick_spin_log_std`) — an unused
learnable parameter sitting in every checkpoint with zero gradient signal
serves no purpose and adds confusion for anyone reading the model class
expecting it to be load-bearing (this document assumes you're wiring
them up properly instead, in which case `kick_spin_log_std` itself gets
replaced by the two new params in §2.3 and can be deleted regardless).

### 7.2 `KickingParams` dataclass field defaults silently diverge from `physics.json`

`max_spin_base_rad_s: float = 8.0` / `max_spin_precision_scale: float =
0.2` as dataclass defaults, but `physics.json` currently sets `12.0` /
`25.0` — a **125x** difference on the precision-scaling term. These
defaults are only reachable if `physics.json`'s `kicking` section is
ever missing the keys (via `.get(key, default)` in `from_config()`), so
in normal operation this divergence is harmless, but it's a
maintenance trap: anyone constructing `KickingParams(...)` directly in a
test or script without going through `from_config()` gets very different
numbers than production. Consider either (a) removing the dataclass
defaults entirely (force every construction site through
`from_config()` or an explicit value, catching the mistake at
construction time instead of silently using stale numbers), or (b)
updating the dataclass defaults to match `physics.json`'s current
values so they at least agree today. Flagging per this repo's own
practice of surfacing "not convinced by the explanation... bring it up"
type findings (see `Idea2.md`'s NB Immediate Immediate section — noting
only because it's directly adjacent code encountered while doing this
research, not because it's in scope for the spin fix itself).

### 7.3 Spin error/noise is not modelled at all — flagged in your own notes

Per `Idea2.md`'s notes-to-self ("We need - higher spin max, implement
spin error (including for the trajectoris displayed in the ui for the
human player during kick order)") — the kick angle/power error model
(`kick_sigma_rad`, Gaussian noise on yaw/pitch scaled by precision, see
`kicking.py` `kick_sigma_rad()`) has no analogous error term for spin.
A perfectly-precise player and a terrible one impart exactly the
requested spin vector with zero noise today (whether via rules AI,
human, or — once this plan lands — neural network). This is out of scope
for the neural-network wiring plan in this document, but should be
considered together with §2.2's magnitude-cap change, since "how much
spin can this player *reliably* generate" and "what's the maximum spin
this player could ever generate" are related but distinct questions —
right now only the latter has any implementation at all (`max_spin_rad_s`
is a hard ceiling with no probabilistic component).

### 7.4 Observation-side spin normalization constant reused as if it were action-space-appropriate

`ai_config.json['observation']['ball_spin_norm_max_rad_s'] = 30.0` is
described as normalizing the **observed ball's current spin** (which can
be higher than what any kicker could impart from a single kick, since the
ball's spin also depends on bounces — see `bounce_spin_retention=0.5` in
`physics.json` — and can in principle accumulate above a single kick's
max). The BC loss in `bc.py` (§1.4 above) reuses this exact constant to
normalize `kick_spin` target/prediction MSE, which conflates "the range
of spin values the network might ever *observe*" with "the range of spin
values the network might ever *impart*". After this plan's §4.1 fix
(normalizing the magnitude loss by the actual per-row
`max_spin_rad_s(kick_precision)` instead), this conflation goes away for
the loss function specifically — but the raw *observation* encoder
(`obs/encoder.py` line ~298-300, `spin_x = ball.spin.x / max(spin_norm,
1e-3)`) still uses the same constant for its own (correct) purpose of
normalizing the *observed* ball spin, which is fine and doesn't need to
change — just don't let the two uses get re-coupled in a future edit.

---

## 8. Summary checklist

- [x] **Immediate**: hardcode `spin=Vector3.zero()` on the neural kick
      path in `apply_nn_action.py` (done as part of this planning pass).
- [ ] Add `kick_spin_axis` (unit **2**-vector head, coefficients over the
      `kick_direction`-derived `(topspin_axis, sidespin_axis)` basis —
      see §2.1.1, NOT a raw 3-vector) + `kick_spin_magnitude` (scalar
      sigmoid head) to `ExecutionNetwork`, replacing the single raw
      `kick_spin` head; update `ExecutionHeadsRaw`/`ExecutionAction`
      dataclasses in `ai/action/schema.py` accordingly.
- [ ] Add a `kick_spin_basis(kick_direction) -> (topspin_axis, sidespin_axis)`
      helper (§2.1.1/§2.2) and use it from both `apply_nn_action.py` and
      the BC loss (§4.1) rather than duplicating the cross-product math.
- [ ] Add a unit test asserting the 2D axis reconstruction round-trips
      correctly for a hand-picked `kick_direction`/`kick_spin` pair, and a
      second test confirming the projected 2D target is correctly
      equivariant under `augment_batch(..., flip_y)` (§4.1's equivariance
      note) — both currently exercise an always-zero code path in
      practice (no demo data has nonzero spin yet), so don't rely on the
      BC loss number alone to catch a bug here.
- [ ] Add `kick_spin_axis_log_std` (reuse `DirectionHead`) and
      `kick_spin_magnitude_log_std` (reuse/add `SquashedNormalHead`
      sampling) parameters; wire config keys per §3.
- [ ] Wire real sampling + log_prob computation into
      `ppo_trainer.py`'s rollout-collection forward pass AND the
      `_ppo_update()` recomputation — both sides, not just one.
- [ ] Add entropy-bonus contributions (`ent_dir_weight` reuse for axis,
      new `ent_spin_magnitude_weight` for magnitude).
- [ ] Update `gating.py`'s `GatingResult` (`kick_spin` field →
      `kick_spin_axis`/`kick_spin_magnitude_fraction`) and
      `select_action()`'s dict plumbing.
- [ ] Update `apply_nn_action.py` to compute the physical spin vector via
      `max_spin_rad_s(match.kicking_params, player.attributes.kick_precision)`
      — this is the actual bug fix that closes the "two values of max
      spin" gap.
- [ ] Extend `BCLabel` with `kick_precision` (§4.1 option (i)); bump
      `BC_LABEL_DIM` 25→26; add `_I_KICK_PRECISION` index.
- [ ] Rewrite the BC spin loss term (axis cosine + magnitude MSE,
      normalized by the real per-row cap) in `bc.py`; update breakdown
      dict keys.
- [ ] Update `EXEC_HEAD_MODULES` and both `_kick_bkdn_keys` sets in
      `ppo_trainer.py`.
- [ ] Confirm `ai/bc/dataset.py`'s schema-width guard rejects old-format
      `.npz` files loudly.
- [ ] Re-record all demonstration sets in active use (§5.3 commands).
- [ ] Run `uv run pytest tests/ai_unit tests/ai_scenario -v` — must pass
      before any new training run.
- [ ] Start a **fresh** training run (no `--latest-pretrain`/
      `--from-pretrained`/`--checkpoint` pointing at a pre-change
      checkpoint — the execution network's parameter shapes changed).
- [ ] Follow-up (separate PR): apply the identical
      sample/log_prob/entropy fix to `kick_power` (§6) — it has the exact
      same deterministic-sampling bug today.
