# Change Plan

## Overview

Five concrete changes in this session, ordered by dependency:

1. **`player.py`** — add atomic action methods
2. **`to_orders.py`** — fix movement to use `move_direction` directly; remove `_apply_move` and `_apply_none`
3. **`evaluate.py`** — add rules-vs-rules baseline (runs by default)
4. **`ppo_trainer.py`** — add `move_direction` diagnostics
5. **`knowledge.md`** — update orders/actions distinction

---

## 1. `src/footballcoach/entities/player.py`

### Why

The rules-based AI and `actions.py` currently set `player.current_order` by
constructing raw Order dataclasses inline (e.g. `player.current_order =
GetPossessionOrder()`). The neural network's `to_orders.py` does the same.
This scatters Order construction across many callers and makes it easy for the
execution network to accidentally construct Orders it should never touch.

The fix: `Player` gets named methods for each high-level action. Callers
(rules-based AI, `to_orders.py`, tests) call `player.get_possession()` etc.
The execution network calls these methods too — it never imports Order types
directly.

### What changes

Add the following methods to the `Player` dataclass (after `is_inactive`):

```python
# --- Atomic action methods -------------------------------------------------
# These are the sole way to assign Orders to a player.  The execution neural
# network calls these directly; rules-based AI and actions.py wrappers call
# them too.  Never set player.current_order to an Order dataclass from
# outside this class.

def kick(self, aim_point: Vector3, power_fraction: float,
         spin: Vector3) -> None:
    """Issue a KickOrder. Only effective if this player has the ball."""
    from footballcoach.orders import KickOrder
    self.current_order = KickOrder(
        aim_point=aim_point, power_fraction=power_fraction, spin=spin
    )

def pass_ball(
    self,
    target_position: Vector3,
    target_player_id: str | None = None,
    power_fraction: float | None = None,
) -> None:
    """Issue a PassOrder toward a position or a specific teammate."""
    from footballcoach.orders import PassOrder
    self.current_order = PassOrder(
        target_position=target_position,
        power_fraction=power_fraction,
        target_player_id=target_player_id,
    )

def get_possession(self) -> None:
    """Chase the ball / dispossess the carrier."""
    from footballcoach.orders import GetPossessionOrder
    self.current_order = GetPossessionOrder()

def tackle_player(self, target_player_id: str) -> None:
    """Chase and tackle a specific opposing player."""
    from footballcoach.orders import ChaseTackleOrder
    self.current_order = ChaseTackleOrder(target_player_id=target_player_id)

def mark_player(self, target_player_id: str) -> None:
    """Mark a specific opposing player."""
    from footballcoach.orders import MarkOrder
    self.current_order = MarkOrder(target_player_id=target_player_id)

def stop(self) -> None:
    """Decelerate to a standstill."""
    from footballcoach.orders import StopOrder
    self.current_order = StopOrder()

def save_goal(self) -> None:
    """Goalkeeper only: track incoming shot and move to intercept."""
    from footballcoach.orders import SaveOrder
    self.current_order = SaveOrder()

def move_to(self, target_position: Vector3, sprint: bool = True,
            max_speed_on_arrival_mps: float | None = None) -> None:
    """Move toward a target position.  Used by the rules-based AI and
    BC label generation only — the neural execution network drives
    movement via step_player_towards directly, not via MoveOrder."""
    from footballcoach.orders import MoveOrder
    self.current_order = MoveOrder(
        target_position=target_position,
        sprint=sprint,
        max_speed_on_arrival_mps=max_speed_on_arrival_mps,
    )
```

Lazy imports (inside each method) avoid circular-import risk since `orders.py`
already imports nothing from `entities/`.

### Callers to update

- `actions.py`: each function now calls the matching player method instead
  of constructing an Order itself. `actions.py` remains as a thin shim (one
  line per function) so all existing call sites in UI, tests, and rules-based
  AI keep working without change.
- `to_orders.py`: same — all `player.current_order = XOrder(...)` lines
  become player method calls.

---

## 2. `src/footballcoach/ai/action/to_orders.py`

### Why

Two bugs:

**Bug A — `_apply_move` ignores `move_direction`.**  
When the decision network fires `MOVE`, the current code uses
`decision_physical["move_region_center_m"]` (the strategic target region
from the *decision* network) as the `MoveOrder` target position. The
execution network's `move_direction` unit vector is completely ignored.  
The design intent: the execution network's `move_direction` is the
instantaneous heading direction — the decision network's `move_region_center`
is strategic context for BC labels and reward shaping, not motor control.

**Bug B — `_apply_none` issues a provisional MoveOrder.**  
When all heads < 0.5 (no action selected), the code creates a 5m
`MoveOrder` in `move_direction` if there's no current order. This is wrong:
`move < 0.5` explicitly means STANDSTILL. The player should decelerate, not
keep drifting.

### What changes

**Remove `_apply_move` entirely.**  
Movement is handled by a new dedicated function `apply_movement_to_player`
that is called unconditionally for every decision tick (whether or not a
high-level action fires). It uses `move_direction` + `sprint` from the
execution network to set a far-target `MoveOrder` (50m) so the engine's
`step_player_towards` picks up the correct direction for all 15 ticks in the
decision interval. When `move < 0.5`, speed mode = STANDSTILL → `player.stop()`.

```python
FAR_TARGET_M = 50.0  # beyond any braking horizon within a 0.5s interval

def apply_movement_to_player(
    player: Player,
    gating: GatingResult,
    move_fires: bool,
) -> None:
    """Apply execution-network movement unconditionally each decision tick.

    move_fires: True if decision-network move_logit >= 0.5.
    When False the player decelerates to standstill (STANDSTILL mode).
    When True the player accelerates/jogs in move_direction (SPRINT or JOG).
    """
    if not move_fires:
        player.stop()
        return

    d = gating.move_direction
    if d is None or np.linalg.norm(d) < 1e-6:
        player.stop()
        return

    target = Vector3(
        player.position.x + float(d[0]) * FAR_TARGET_M,
        player.position.y + float(d[1]) * FAR_TARGET_M,
        0.0,
    )
    sprint = bool(gating.sprint) if gating.sprint is not None else True
    player.move_to(target, sprint=sprint)
```

**`apply_action_to_player` becomes a two-phase call:**
1. Always call `apply_movement_to_player` first (movement is unconditional).
2. Then check high-level decision heads — if one fires, call the corresponding
   player method which overwrites `current_order` with the appropriate Order
   (shoot → KickOrder, pass → PassOrder, etc.). These orders' state machines
   in `match.py` take over for the duration of the order, which may span
   multiple decision intervals.

**`_apply_none` is removed.** When all heads < 0.5, step 1 above already set
STANDSTILL via `player.stop()`. There's nothing left to do.

**`_apply_move` is removed.** Movement is entirely handled by
`apply_movement_to_player` in step 1. The `MOVE` case in the dispatch
switch disappears — `MOVE` is now implicit in every tick, not a discrete branch.

**`HOLD_POSITION` stays as a discrete action** (it's different from MOVE in
reward shaping: stronger penalty for leaving the region), but it also needs
movement — so `HOLD_POSITION` fires movement too, just tagged differently.

### Updated `apply_action_to_player` skeleton:

```python
def apply_action_to_player(gating, player, match, slot_player_ids,
                            decision_physical, move_fires: bool) -> OrderTranslationResult:
    # Phase 1: movement always
    apply_movement_to_player(player, gating, move_fires)

    # Phase 2: high-level order (overwrites movement order only for
    # actions that encapsulate their own movement, e.g. ChaseTackleOrder)
    sel = gating.selected
    if sel == SelectedAction.SHOOT:
        return _apply_shoot(player, match, gating, decision_physical)
    elif sel == SelectedAction.PASS:
        return _apply_pass(player, match, gating, slot_player_ids, decision_physical)
    elif sel == SelectedAction.TACKLE:
        return _apply_tackle(player, match, gating, slot_player_ids)
    elif sel == SelectedAction.GET_POSSESSION:
        return _apply_get_possession(player, match)
    elif sel == SelectedAction.MARK:
        return _apply_mark(player, match, gating, slot_player_ids)
    # MOVE, HOLD_POSITION, NONE: movement already applied above, nothing more.
    return OrderTranslationResult()
```

### Callers

`scenario_env.py` already passes `decision_probs` dict to gating. Add
`move_fires = decision_probs.get("move", 0.0) >= 0.5` and pass to
`apply_action_to_player`. The signature gains one argument.

---

## 3. `src/footballcoach/ai/scripts/evaluate.py`

### Why

Currently evaluate.py only evaluates a neural net checkpoint.  There's no
baseline to compare against.  A rules-vs-rules run (both players driven by
Phase1RulesAI) on the same scenario gives the natural win-rate baseline.

### What changes

- Add `--no-baseline` flag (default: baseline IS run).
- Add `_run_baseline_evaluation(env, n_trials) -> dict` that:
  - Resets the env normally but patches `match._opponent_use_rules_ai = True`
    on every reset (so opponent is always rules-based, not neural or immobile).
  - For the trainee, calls `_PHASE1_TRAINEE_AI(match, tick)` via the
    existing `phase1_training_on_tick` mechanism — i.e., pass a special
    flag to `env.reset()` that forces the rules-based trainee path.
  - Steps env with a no-op action (all probs = 0, move_direction = [1,0]) so
    `apply_action_to_player` fires STANDSTILL and doesn't interfere with the
    on_tick rules AI.
  - Collects same outcome stats as neural eval.

Actually the cleanest approach: use the existing `build_1v1_scenario` /
`_1v1_on_tick` UI hook which already drives BOTH players via rules-based AI.
Build a separate ScenarioEnv with `on_tick=_1v1_on_tick` and a special
`baseline_mode=True` that skips applying neural actions.

### JSON output shape:

```json
{
  "neural_net": {
    "n_trials": 100,
    "mean_reward": ...,
    "outcomes": {"box_possession": 52, "opponent_box_possession": 31, "timeout": 17}
  },
  "baseline_rules_vs_rules": {
    "n_trials": 100,
    "outcomes": {"box_possession": 48, "opponent_box_possession": 39, "timeout": 13}
  },
  "checkpoint_step": 100000
}
```

`box_possession` = trainee won; `opponent_box_possession` = trainee lost.
Win rate = `box_possession / n_trials`.

---

## 4. `src/footballcoach/ai/ppo/ppo_trainer.py`

### Why

`move_direction` is currently excluded from the PPO log_prob ratio because
including it causes KL to explode (mean shifts of 25–50 with std=1 → KL
~2000). Before deciding how to fix this, we need to understand *what* is
actually happening — where is the mean drifting, and how much KL would
`move_direction` contribute if included?

### Diagnostics to add (inside the existing `if not _diag_done:` block)

All added after the existing `lp_movedir` / `lp_kickdir` lines:

```python
# (1) Raw magnitude of the current move_direction means
movedir_mag = e_heads.move_direction.norm(dim=-1)
print(f"    move_dir raw_mag: mean={movedir_mag.mean():.3f}"
      f"  max={movedir_mag.max():.3f}  std={movedir_mag.std():.3f}")

# (2) Stored sample norms vs current mean norms
stored_raw = mb_actions["move_dir_raw"]          # (mb, 2)
stored_norm = stored_raw.norm(dim=-1)
current_norm = e_heads.move_direction.norm(dim=-1)
print(f"    stored_raw_norm: mean={stored_norm.mean():.3f}"
      f"  current_mean_norm: mean={current_norm.mean():.3f}"
      f"  ratio={current_norm.mean()/stored_norm.mean().clamp(min=1e-6):.3f}")

# (3) Mean direction of current policy — is it collapsing to a fixed angle?
mean_vec = e_heads.move_direction.mean(dim=0)
mean_vec_norm = mean_vec.norm().item()
angle_deg = math.degrees(math.atan2(float(mean_vec[1]), float(mean_vec[0])))
print(f"    move_dir mean_vec=({mean_vec[0]:.3f},{mean_vec[1]:.3f})"
      f"  |mean|={mean_vec_norm:.3f}  angle={angle_deg:.1f}deg")

# (4) Hypothetical KL from move_direction if it were in the ratio
#     log_std is fixed (buffer), so we compare old_mean stored sample
#     against new mean (e_heads after backward = e_heads_before_step here,
#     since diag runs before optimizer.step).  True post-step KL needs
#     e_after (already computed below) — we print both.
hyp_movedir_kl = (
    DirectionHead(e_heads.move_direction, log_std_move).log_prob(stored_raw)
    - DirectionHead(e_heads.move_direction, log_std_move).log_prob(stored_raw)
).mean()  # trivially 0 here; real value comes post-step below
print(f"    NOTE: post-step hypothetical movedir_kl printed in per-mb line")
```

And in the per-minibatch print line (after `optimizer.step()`), add:
```python
# After step, compute hypothetical KL if move_dir were included
with torch.no_grad():
    lp_movedir_old_stored = DirectionHead(
        e_heads.move_direction.detach(), log_std_move
    ).log_prob(mb_actions["move_dir_raw"])
    lp_movedir_new_stored = DirectionHead(
        e_after.move_direction, log_std_move
    ).log_prob(mb_actions["move_dir_raw"])
    movedir_hyp_kl = (lp_movedir_old_stored - lp_movedir_new_stored).mean().item()
```

And append `movedir_hyp_kl` to the per-mb print line:
```
  move_shift={movedir_mean_shift:.3f}  movedir_hyp_kl={movedir_hyp_kl:.4f}
```

This gives us per-minibatch visibility into exactly how much KL
`move_direction` would add — the smoking gun.

---

## 5. `src/footballcoach/knowledge.md`

### What changes

Replace the `actions.py` section with the updated architecture:

```
## `actions.py` - thin shim (deprecated, kept for call-site compatibility)

`actions.py` wraps the player methods added in this session.  Its functions
(`move_to`, `shoot`, `pass_to`, `tackle`, `save`, `mark`) now delegate
directly to the corresponding `Player` method.  New code should call player
methods directly; `actions.py` exists only to avoid updating every existing
call site simultaneously.

## Player action methods — the canonical way to issue Orders

`Player` now has named action methods that are the SOLE way to set
`player.current_order`:

- `player.kick(aim_point, power_fraction, spin)` → KickOrder
- `player.pass_ball(target_position, target_player_id, power_fraction)` → PassOrder
- `player.get_possession()` → GetPossessionOrder
- `player.tackle_player(target_player_id)` → ChaseTackleOrder
- `player.mark_player(target_player_id)` → MarkOrder
- `player.stop()` → StopOrder
- `player.save_goal()` → SaveOrder (goalkeeper only)
- `player.move_to(target_position, sprint)` → MoveOrder
  (rules-based AI and BC labels only — see neural AI boundary below)

## Orders vs neural network boundary (IMPORTANT)

`MoveOrder` and all other Order types are for:
  - The rules-based AI (UI scenarios, BC label generation)
  - High-level discrete actions triggered by decision-network head fires
    (shoot, pass, tackle, etc.)

The execution neural network drives movement via `step_player_towards`
(called through `player.move_to` with a 50m far-target) using its
`move_direction` unit vector + `sprint` Bernoulli.  The decision network's
`move_region_center` / `move_arrival_speed` are strategic context for reward
shaping and BC labels — NOT motor control inputs.

When `move_logit < 0.5` → `player.stop()` → STANDSTILL mode.
When `move_logit >= 0.5` → `player.move_to(position + move_direction * 50m)`.
```

---

## Test plan

After each change:
```bash
uv run pytest tests/ -q
```

Expected: all tests pass unchanged.  The changes are either additive (new
player methods, new eval path) or fix incorrect behaviour that no test was
asserting (wrong `move_region_center` target, wrong `_apply_none` movement).

No balance tests need updating — `MoveOrder` semantics are unchanged;
only which code constructs them changes.
