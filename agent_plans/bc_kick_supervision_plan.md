# Plan: Full BC supervision for kick_direction, kick_power, kick_spin

## Goal

Every `execution_net` output currently gets BC supervision except
`kick_direction`, `kick_power`, and `kick_spin`. This plan wires all three
in, automatically, for any AI that kicks (not just `Phase1RulesAI`), and
exposes their losses in every place BC loss is already logged (offline
pretrain epochs, BC repair epochs, PPO aux loss line, online pretrain).

Also fixes a smaller pre-existing gap noticed along the way: `kick`'s BCE
loss is computed in `bc_loss_from_tensor()` but only folded into the
aggregate `exec_bce` breakdown key — unlike `tackle_attempt`, which already
gets its own breakdown key. This plan gives `kick` the same treatment.

## Design principle: capture at the engine chokepoint, not per-AI

`Player.kick_direct()` is the single method every kick path already funnels
through — `KickOrder`/`ShootOrder`/`PassOrder.execute()`, `MoveOrder`'s
push-kick (`_do_push_kick()`), and the neural network's direct kick action.
`kicked_this_tick` already uses this pattern (set unconditionally inside
`kick_direct()`, reset once per tick in `Match._process_orders()`) — see
`Player.kicked_this_tick`'s docstring for the precedent and the reason it
was introduced (previously, only order types were inspected, and
`MoveOrder`'s push-kick was invisible to demo recording).

We extend this pattern to also capture the actual physical kick vector
(direction, power_fraction, spin) at the moment `kick_direct()` executes.
This makes BC supervision automatic for **any** current or future AI that
calls `kick_direct()` — rules-based, human input, or another neural net —
with zero additional wiring per-AI. `phase1_labels()` (and any future
`phaseN_labels()`) just reads these fields off `Player` instead of
re-deriving them.

## Files to change

### 1. `src/footballcoach/entities/player.py`

Add three new fields next to `kicked_this_tick`, all reset in
`Match._process_orders()` and populated in `kick_direct()`:

```python
# Unconditional per-tick kick output capture — set inside kick_direct() every
# time it actually executes kick physics, regardless of on_kick being set and
# regardless of which Order (or no Order, e.g. MoveOrder's push-kick)
# triggered it. Mirrors kicked_this_tick's rationale (see its docstring) but
# captures the actual kick VECTOR, not just the boolean fact of kicking, so
# BC label derivation can supervise kick_direction/kick_power/kick_spin for
# ANY AI that kicks (rules-based, human, future neural variants) with no
# per-AI wiring. All three are None/unset when the player did not kick this
# tick (reset alongside kicked_this_tick in Match._process_orders()).
last_kick_direction: Vector3 | None = field(default=None, repr=False, compare=False)
last_kick_power_fraction: float | None = field(default=None, repr=False, compare=False)
last_kick_spin: Vector3 | None = field(default=None, repr=False, compare=False)
```

In `kick_direct()`, after `adjusted_power` is computed and before/alongside
the `kick_ball(...)` call:

```python
_dir_vec = (aim_point - self.position).xy()
self.last_kick_direction = _dir_vec.normalized() if _dir_vec.length() > 1e-6 else None
self.last_kick_power_fraction = float(adjusted_power)
self.last_kick_spin = spin
```

Use `adjusted_power` (not raw `power_fraction`) since that's the actual
value passed to `kick_ball()` — the physically-real kick strength after
`compensate_for_run` adjustment. Use `.xy()` + `.normalized()` (already
available on `Vector3`, see `mathutils/vector3.py`) to get a ground-plane
unit vector, consistent with how `move_direction` is a 2D (x, y) vector
elsewhere in this codebase (`BC_LABEL_DIM` layout, `execution_network.py`).

### 2. `src/footballcoach/engine/match.py`

In `_process_orders()`, reset the three new fields alongside
`kicked_this_tick`:

```python
player.kicked_this_tick = False
player.last_kick_direction = None
player.last_kick_power_fraction = None
player.last_kick_spin = None
```

### 3. `src/footballcoach/ai/ppo/bc.py` — label schema + loss

**Schema bump**: `BC_LABEL_DIM` 18 → 23 (add 5 new floats: kick_dir_x,
kick_dir_y, kick_power_fraction, kick_spin_x, kick_spin_y, kick_spin_z —
note spin is a 3D vector per `execution_network.py`'s `kick_spin: nn.Linear(trunk_hidden, 3)`,
so it's 3 floats, not 2). Recount: 2 (dir) + 1 (power) + 3 (spin) = 6 new
floats, not 5. `BC_LABEL_DIM` 18 → 24.

New indices (append after `_I_OPPONENT_AI_TYPE = 17`):
```python
_I_KICK_DIR_X        = 18
_I_KICK_DIR_Y        = 19
_I_KICK_POWER        = 20
_I_KICK_SPIN_X        = 21
_I_KICK_SPIN_Y        = 22
_I_KICK_SPIN_Z        = 23
```

Update the module docstring's layout table to document all six new fields,
following the existing style (see how `[16]`/`[17]` ai_type fields are
documented).

**`BCLabel` dataclass**: add fields
```python
kick_direction: Optional[np.ndarray] = None      # shape (2,) unit vector, ground plane
kick_power_fraction: Optional[float] = None       # [0, 1], None if not kicking
kick_spin: Optional[np.ndarray] = None            # shape (3,), None if not kicking
```
and pack them in `to_array()` the same way `move_direction`/`move_region_center_m`
are packed (write 0.0 if `None`).

**`phase1_labels()`**: after computing `kick_this_tick`, read the new
`Player` fields directly instead of any manual re-derivation:
```python
kick_direction = None
kick_power_fraction = None
kick_spin = None
if player.kicked_this_tick:
    if player.last_kick_direction is not None:
        kick_direction = np.array(
            [player.last_kick_direction.x, player.last_kick_direction.y], dtype=np.float32
        )
    kick_power_fraction = player.last_kick_power_fraction
    if player.last_kick_spin is not None:
        kick_spin = np.array(
            [player.last_kick_spin.x, player.last_kick_spin.y, player.last_kick_spin.z],
            dtype=np.float32,
        )
```
Pass `kick_direction=kick_direction, kick_power_fraction=kick_power_fraction,
kick_spin=kick_spin` into every `BCLabel(...)` construction in this
function (there are currently two return sites: `MoveOrder` branch and
`GetPossessionOrder` branch — grep for `BCLabel(` before editing to find
all call sites, there may be a third "no possession" branch).

This works automatically for the opponent-labels path too (`phase1_labels(env, player_id="opponent")`)
since it reads the same `Player.last_kick_*` fields — no special-casing
needed.

**`bc_loss_from_tensor()`**: add three things:

1. Give `kick`'s existing BCE its own tracked tensor (currently computed
   inline and folded straight into `exec_bce_loss` — no separate variable):
   ```python
   kick_loss = torch.zeros(labels.shape[0], device=labels.device)
   ...
   if exec_heads is not None:
       kick_loss = _bce(exec_heads.kick_logit, _I_KICK_THIS_TICK, pos_weight_kick)
       exec_bce_loss = move_loss + sprint_loss + kick_loss + tackle_attempt_loss
   ```
   Add `"kick": float(kick_loss[valid].mean())` to the breakdown dict (both
   the early-return zero-dict and the real one).

2. Add a `kick_direction` cosine loss, mirroring `move_direction`'s block
   exactly but gated additionally on `kick_this_tick == 1` (only meaningful
   when a kick actually happened this step — unlike `move_direction` which
   is meaningful whenever `has_dir` is true regardless of move/get_possession
   intent):
   ```python
   kick_dir_loss_per = torch.zeros(labels.shape[0], device=labels.device)
   if exec_heads is not None:
       has_kick_dir = (
           (labels[:, _I_KICK_DIR_X].abs() + labels[:, _I_KICK_DIR_Y].abs()) > 1e-6
       ) & (labels[:, _I_KICK_THIS_TICK] > 0.5)
       if has_kick_dir.any():
           target_kdir = labels[:, _I_KICK_DIR_X:_I_KICK_DIR_Y + 1]
           pred_kdir = exec_heads.kick_direction
           pred_kdir_norm = pred_kdir / (pred_kdir.norm(dim=-1, keepdim=True) + eps)
           kdir_cos_loss = 1.0 - (pred_kdir_norm * target_kdir).sum(dim=-1)
           kick_dir_loss_per = direction_loss_weight * torch.where(
               has_kick_dir, kdir_cos_loss, torch.zeros_like(kdir_cos_loss)
           )
           loss += exec_weight * kick_dir_loss_per
   ```
   Reuses `direction_loss_weight` (same knob as `move_direction`) rather than
   introducing a new config key — keep this simple unless tuning later shows
   they need independent weights. Add `"kick_direction"` breakdown key.

3. Add `kick_power` (MSE, since it's a continuous [0,1] scalar via sigmoid in
   the network, and `kick_spin` (MSE, 3D vector) losses, both gated on
   `kick_this_tick == 1` and `pred is not None`-style validity via the label's
   own zero-vs-set convention (use a dedicated `has_kick_power`/`has_kick_spin`
   mask like `has_kick_dir` above — do NOT reuse `_I_KICK_DIR_X/_Y` nonzero
   check for these since power=0.0 is a legitimately valid value, unlike
   direction where (0,0) unambiguously means "not applicable"). Use the
   `_I_KICK_THIS_TICK > 0.5` mask alone as the gate for power/spin (no
   additional "has magnitude" check needed/possible):
   ```python
   kick_power_loss_per = torch.zeros(labels.shape[0], device=labels.device)
   kick_spin_loss_per = torch.zeros(labels.shape[0], device=labels.device)
   if exec_heads is not None:
       kicked_mask = labels[:, _I_KICK_THIS_TICK] > 0.5
       if kicked_mask.any():
           pred_power = torch.sigmoid(exec_heads.kick_power.squeeze(-1))
           target_power = labels[:, _I_KICK_POWER]
           power_mse = (pred_power - target_power) ** 2
           kick_power_loss_per = torch.where(kicked_mask, power_mse, torch.zeros_like(power_mse))
           loss += exec_weight * kick_power_loss_per

           target_spin = labels[:, _I_KICK_SPIN_X:_I_KICK_SPIN_Z + 1]
           spin_mse = ((exec_heads.kick_spin - target_spin) ** 2).sum(dim=-1)
           kick_spin_loss_per = torch.where(kicked_mask, spin_mse, torch.zeros_like(spin_mse))
           loss += exec_weight * kick_spin_loss_per
   ```
   Add `"kick_power"` and `"kick_spin"` breakdown keys. Note: `kick_spin`'s
   physical units/scale need checking against `execution_network.py`'s
   comment ("physical units determined by to_orders.py") — if spin magnitude
   is large (e.g. rad/s, potentially >>1), the raw MSE could dominate the
   loss; consider normalizing by a config constant (e.g.
   `ai_config.json`'s existing `ball_spin_norm_max_rad_s: 30.0` — reuse this
   for consistency with how ball spin is normalized elsewhere in obs
   encoding) — divide both `exec_heads.kick_spin` and `target_spin` by this
   constant before computing `spin_mse`, OR divide the final `spin_mse` by
   `spin_norm_max ** 2`. Decide during implementation by checking actual
   magnitude of `Player.last_kick_spin` values in practice (print/log a
   sample during a debug run) rather than guessing.

4. Update the zero-dict early return (`if not valid.any(): ...`) to include
   all new keys: `"kick"`, `"kick_direction"`, `"kick_power"`, `"kick_spin"`.

### 4. `src/footballcoach/ai/obs/augment.py` — flip handling

Add column constants mirroring `_BC_DIR_X_COL`/`_BC_REGION_X_COL`:
```python
_BC_KICK_DIR_X_COL: int = 18
_BC_KICK_DIR_Y_COL: int = 19
```
(`kick_power` at 20 is a scalar, invariant under flips — no change needed.
`kick_spin` at 21-23: spin is a pseudovector, same transform rules as
`BALL_FLIP_X_IDX`/`BALL_FLIP_Y_IDX` already apply to ball spin — see
augment.py's existing pseudovector comment block, lines ~57-61. Under
flip_x: spin_y and spin_z negate. Under flip_y: spin_x and spin_z negate.
This must be applied to `_I_KICK_SPIN_X/Y/Z` in the BC label tensor the
same way `BALL_FLIP_X_IDX`/`BALL_FLIP_Y_IDX` already do for ball_feat.)

In `augment_obs_bc()`'s per-flip-variant loop, add:
```python
if flip_x:
    bc[:, _BC_DIR_X_COL]     *= -1.0
    bc[:, _BC_REGION_X_COL]  *= -1.0
    bc[:, _BC_KICK_DIR_X_COL] *= -1.0
    bc[:, _I_KICK_SPIN_Y]    *= -1.0   # pseudovector
    bc[:, _I_KICK_SPIN_Z]    *= -1.0   # pseudovector
if flip_y:
    bc[:, _BC_DIR_Y_COL]     *= -1.0
    bc[:, _BC_REGION_Y_COL]  *= -1.0
    bc[:, _BC_KICK_DIR_Y_COL] *= -1.0
    bc[:, _I_KICK_SPIN_X]    *= -1.0   # pseudovector
    bc[:, _I_KICK_SPIN_Z]    *= -1.0   # pseudovector
```
(Import `_I_KICK_SPIN_X/Y/Z` from `bc.py` or duplicate as local constants —
follow whatever pattern the existing `_BC_DIR_X_COL` etc. constants use;
they're currently hardcoded magic numbers with a comment warning they must
be manually kept in sync with `bc.py`'s `_I_*` constants. Consider fixing
this staleness risk while here: either import the real constants from
`bc.py` directly instead of re-declaring magic numbers, or leave the
existing pattern for consistency and just add a comment cross-reference —
decide based on how invasive the import would be (circular import risk
between `augment.py` and `bc.py`; check before changing the existing
pattern).

**Important**: `augment_batch()` (the PPO rollout path, separate from
`augment_obs_bc()`) already handles `kick_dir_raw` via `_DIR_ACTION_KEYS` —
that's the PPO *action* dict (post-sampling network output), completely
independent of the BC *label* tensor handled by `augment_obs_bc()`. No
change needed there; don't conflate the two paths.

### 5. `src/footballcoach/ai/bc/dataset.py`

Update the schema-guard error message in `from_file()`:
```python
if bc_labels.shape[-1] != BC_LABEL_DIM:
    raise ValueError(
        f"{path}: bc_labels has width {bc_labels.shape[-1]}, expected "
        f"BC_LABEL_DIM={BC_LABEL_DIM}. This .npz was recorded with an "
        f"older BC label schema (e.g. missing kick_direction/kick_power/"
        f"kick_spin fields added for full execution-head BC coverage). "
        f"Re-record demonstrations: ..."
    )
```
Update the comment above it (currently says "currently 18 ... W6 opponent_ai_type").

### 6. `src/footballcoach/ai/ppo/ppo_trainer.py` — logging

In the offline BC epoch loop (`pretrain_combined`'s Phase 1, ~line 745-880)
and the BC repair loop (~line 940-1050), add tracking + printing for:
- `kick_dir_cos` (mean cosine sim between predicted/label `kick_direction`,
  gated on `kick_this_tick==1` rows only) — mirrors the existing `dir_cosines`
  list pattern exactly, just filtered to kicked rows.
- The new `bkdn` keys (`kick`, `kick_direction`, `kick_power`, `kick_spin`)
  will show up automatically in the existing `bkdn_str = "  ".join(f"{k}={v/_bkdn_n:.3f}" ...)`
  line since it iterates whatever keys are present in the dict — no code
  change needed there, just confirm by inspection after `bc.py`'s changes
  land.

Print `kick_dir_cos=` alongside the existing `dir_cos=` in both epoch log
lines (main BC epoch + BC repair epoch).

### 7. `record_demonstrations.py`

No changes should be strictly required — it already calls `label_fn(env, player_id=pid)`
(i.e. `phase1_labels`) which will automatically pick up the new fields once
`bc.py` is updated, and `on_kick`/`on_tackle` callbacks already fire at
exactly the tick `kick_direct()` runs (see file's own header comment on
`on_kick`/`on_tackle` timing). Verify after implementation with a small
manual recording run + inspect `.npz` `bc_labels` columns 18-23 for nonzero
values on rows where `kick_this_tick=1`.

### 8. `ai_config.json`

No new config keys are strictly required (reusing `direction_loss_weight`
for `kick_direction`, `exec_weight`/`dec_weight` for scaling, and
`ball_spin_norm_max_rad_s` if spin normalization is needed per point 3
above). If tuning later shows `kick_direction`/`kick_power`/`kick_spin`
need independent weights from `move_direction`, add
`kick_direction_loss_weight`, `kick_power_loss_weight`, `kick_spin_loss_weight`
to the `bc` section at that time — not speculatively now (avoid
over-engineering per repo convention).

## Tests to write / update

- `tests/ai_unit/test_bc.py`:
  - New test: `Player.kick_direct()` populates `last_kick_direction`/
    `last_kick_power_fraction`/`last_kick_spin` correctly (unit test at the
    `Player`/engine level, not BC-specific — may belong in
    `tests/unit/test_kicking.py` instead, check existing file organization
    convention first).
  - New test: `phase1_labels()` returns non-None `kick_direction`/
    `kick_power_fraction`/`kick_spin` when `player.kicked_this_tick=True`
    with `last_kick_*` fields set, and all-zero/gated-out when not kicking.
  - New test: `bc_loss_from_tensor()` — construct a label tensor with
    `kick_this_tick=1` and known `kick_direction`/`kick_power`/`kick_spin`
    targets, verify the new breakdown keys (`kick`, `kick_direction`,
    `kick_power`, `kick_spin`) are present and their values respond correctly
    to matching vs. mismatching predictions (loss near 0 for exact match,
    positive for mismatch) — mirror existing tests for `direction`/`region`
    breakdown keys if present.
  - Update `BC_LABEL_DIM` literal assertions if any test hardcodes `18`.
- `tests/ai_unit/test_obs_schema.py` or augmentation-specific test file
  (check for existing `augment_obs_bc` tests): verify `kick_direction`/
  `kick_spin` columns flip sign correctly under `flip_x`/`flip_y` in
  `augment_obs_bc()`, and `kick_power` does NOT flip (invariant scalar).
- Any test that constructs a raw `BCLabel(...)` or the flat array manually
  with a hardcoded length (grep for `BC_LABEL_DIM` and `np.zeros(18)`-style
  literals across `tests/`) needs updating to the new width (24).
- Run full `tests/ai_unit/` and `tests/ai_scenario/` suites after the change
  — schema-width changes have historically needed a `.npz` re-record (see
  `ai_trainer_knowledge.md`'s "Schema break" note for the `PLAYER_FEATURE_DIM`
  26→28 precedent) — confirm no other repo file caches `BC_LABEL_DIM`'s
  numeric value.

## Documentation to update

- `src/footballcoach/ai/ppo/bc.py` module docstring — the flat tensor layout
  table (currently documents 18 fields) needs the 6 new entries appended,
  following the exact same style/wording as existing entries.
- `ai_trainer_knowledge.md`:
  - Section "6. Behavioural cloning (BC) from rules-based AI" → "What is
    supervised" subsection currently states: *"Kick, tackle_attempt, and
    kick_direction are not supervised (rules-based AI doesn't kick in Phase 1;
    extend BCLabel if needed later)"* — this sentence is now WRONG on two
    counts (tackle_attempt already was supervised before this change, and
    kick/kick_direction/kick_power/kick_spin all become supervised after
    this change). Rewrite to reflect full coverage.
  - Section 8 "Reading the training log" — add `kick_dir_cos=` to the BC
    epoch line example and field table (mirroring the `dir_cos=`/`mv_p=`
    style already documented there from the earlier `kk_p`/`tk_p` addition).
- `src/footballcoach/ai/knowledge.md` — check for any other reference to
  "kick_direction ... not supervised" or `BC_LABEL_DIM=18` and update.
- `ai_config.json` — if new config keys are added per point 8 above, document
  them inline with `_comment_*` keys following the existing convention.

## Rollout / operational notes

- This is a **breaking schema change** to `.npz` demonstration files
  (`BC_LABEL_DIM` 18 → 24), exactly like the `PLAYER_FEATURE_DIM` 26→28
  break documented in `ai_trainer_knowledge.md`. All existing
  `demonstrations/phase1/*.npz` files must be re-recorded after this lands
  — the `DemonstrationDataset.from_file()` schema guard will raise
  `ValueError` on load otherwise (working as intended — fail loudly, not
  silently corrupt).
- Re-record via:
  ```bash
  rm -f demonstrations/phase1/*.npz
  uv run python -m footballcoach.ai.scripts.record_demonstrations \
      --phase 1 --n-episodes <N> --episodes-per-file 8 \
      --output demonstrations/phase1/ --seed 42
  ```
- Since Phase 1's rules AI (`Phase1RulesAI`) rarely kicks (mostly
  Move/GetPossession — this is exactly why `pos_weight_kick` is auto-computed
  to a large class-imbalance correction, see BC pos_weight logging), expect
  `kick_direction`/`kick_power`/`kick_spin` supervision to be sparse/low-signal
  in Phase 1 specifically. This is fine — the infrastructure will
  automatically light up more once kick-heavy phases/scenarios are
  recorded (Phase 2 shooting uses `build_penalty_scenario`, which kicks far
  more) or once `Phase1RulesAI`'s push-kick behavior in `MoveOrder` fires
  more often per episode. Don't expect `kk_p`/kick-related breakdown values
  to look meaningfully different in Phase 1 runs immediately after this
  change — the win here is architectural completeness + zero manual wiring
  for future phases/AIs, not necessarily an immediate Phase 1 behavior
  change.
- After re-recording, run one short training run (`--total-steps 10000` or
  similar smoke-scale) and confirm the new `kick`, `kick_direction`,
  `kick_power`, `kick_spin` breakdown keys appear with sensible (non-NaN,
  non-exploding) values in the BC epoch log lines before committing to a
  full-length run.
