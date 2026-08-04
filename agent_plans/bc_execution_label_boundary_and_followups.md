> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). Otherwise documentation goes stale
> and confusion occurs.

# Plan: Fix Orders/execution-network label boundary + BC loss reporting + log/UX follow-ups

## Context and how we got here

This plan is a direct continuation of `agent_plans/bc_kick_supervision_plan.md`
(already implemented — `kick_direction`/`kick_power`/`kick_spin` BC supervision,
`BC_LABEL_DIM` 18→24, see that file for the full history). While reviewing the
resulting training logs, a real, recurring architectural bug was found in
`phase1_labels()` (`src/footballcoach/ai/ppo/bc.py`), the function that derives
BC supervision labels for the rules-based AI. This plan documents that bug,
the fix, and several smaller follow-ups identified in the same review.

**Do not skip the "Understand the codebase" section below** — the fix touches
a subtle but critical architectural boundary that has been violated multiple
times in this codebase's history. Get this wrong again and BC supervision
will silently degrade the whole training pipeline's quality without any
visible error.

---

## Part 0: The core bug — Orders vs execution-network label boundary

### The rule (already added to `ai/knowledge.md` — reinforce, don't re-argue)

**Orders (`MoveOrder`, `GetPossessionOrder`, `ChaseTackleOrder`, `ShootOrder`,
`PassOrder`, `MarkOrder`, `SaveOrder`, `StopOrder`, `KickOrder`) are:**
- **OUTPUT of the decision network.** The decision network's Bernoulli heads
  (`shoot`, `pass_`, `move`, `tackle`, `get_possession_extra`, `mark`,
  `hold_position`) plus `move_region_center` are legitimately supervised by
  "what order would the rules AI issue right now" — this is an order-level,
  strategic-intent decision. Reading an order's *type* and *fields* (e.g.
  `MoveOrder.target_position`) for this purpose is correct and intended.
- **INPUT to the execution network** (as context, e.g. via `ai_type`/order-type
  features feeding the execution network's forward pass) — again, legitimate.
- **NEVER a source for deriving execution-network *labels*.** The execution
  network's outputs (`move_direction`, `sprint`, `exec_move`, `kick_this_tick`,
  `kick_direction`, `kick_power`, `kick_spin`, `tackle_attempt`) must be
  supervised ONLY from what actually lands on the `Player` object after the
  engine/order machinery has run for that tick:
  - `player.desired_direction` (`Vector3`, already normalized by
    `_compute_movement_intent`/set directly by orders)
  - `player.desired_speed_mode` (`SpeedMode` enum: `SPRINT` / `JOG` /
    `STANDSTILL`, or `None` if no movement command was issued this tick)
  - `player.kicked_this_tick` (`bool`)
  - `player.last_kick_direction` / `last_kick_power_fraction` / `last_kick_spin`
    (`Vector3 | None`, `float | None`, `Vector3 | None`)
  - `player.tackle`-related state (see `ChaseTackleOrder`/`GetPossessionOrder`
    contact-tackle detection, already correctly done via `are_touching()` +
    order-type check in current `phase1_labels()` — this part is fine, see
    "What's already correct" below).

**Why this matters, concretely:** `_compute_movement_intent()` in
`src/footballcoach/orders.py` (called by every movement-issuing order's
`execute()`) applies real, non-trivial logic beyond "point at the target":
- **Braking** via `braking_speed_mode()` (decelerates approaching arrival)
- **Repulsion** via `steering.compute_repulsion()` (steers around other
  players/ball-carrier)
- **Turn-rate limiting** and **brake-to-turn** (large heading changes force a
  `STANDSTILL` tick rather than an instant snap-turn)
- **Close-proximity lateral-overshoot braking**
- **Push-kick short-circuiting** (`MoveOrder`'s push-kick: kicks the ball
  ahead and sprints after it instead of dribbling, changing the direction
  entirely for that tick)

None of this is reproducible by hand-deriving `normalize(target - position)`
in `bc.py`. A hand-derived vector is a *rough, frequently-wrong* proxy for
what the rules AI is physically doing that tick — it ignores repulsion,
braking, turning, and push-kicks entirely, and it produces nonsensical output
at the exact tick the target is reached (see below).

### The concrete bug in `phase1_labels()` today

Read `src/footballcoach/ai/ppo/bc.py`'s `phase1_labels()` function in full
before changing anything — grep for `def phase1_labels`. As of this plan, it:

1. Reads `current_exec = player.current_order` (the *already-executing* order
   for this tick — correct, this is how `kick_this_tick`/`tackle_attempt` are
   derived, from `Player.kicked_this_tick`/order-type checks. **This part is
   correct and must be preserved.**)
2. Clears `player.current_order = None`, calls `Phase1RulesAI().act(player,
   match, trial_tick=0)` to get a **fresh decision** (`order`), then restores
   `player.current_order = current_exec` in a `finally` block. **This part
   (decision-head sourcing) is also correct and must be preserved** — the
   Bernoulli decision heads (`move`, `get_possession_extra`, etc.) SHOULD be
   "what would the rules AI decide right now", and re-deciding from scratch
   (rather than reusing `current_exec`) is intentional so labels don't lag by
   one decision interval.
3. **THE BUG:** for the execution-level fields (`move_direction`, `sprint`,
   `exec_move`), instead of running `order.execute()` and reading back
   `player.desired_direction`/`desired_speed_mode`, it manually computes:
   ```python
   dx, dy = tgt.x - tx, tgt.y - ty   # tgt = order.target_position (MoveOrder)
   # or tgt = ball position (GetPossessionOrder)
   length = math.hypot(dx, dy)
   if length < 1e-6:
       return BCLabel.invalid()   # discards the WHOLE frame including kick/tackle labels!
   direction = np.array([dx / length, dy / length], dtype=np.float32)
   ```
   and hardcodes `exec_move=1.0`, `sprint=1.0 if order.sprint else 0.0` (reading
   `order.sprint` directly — also wrong, see below) in every `BCLabel(...)`
   construction. There's even a computed-but-unused variable,
   `exec_move_now = 1.0 if isinstance(current_exec, (_MoveOrder, _GPOrder)) else 0.0`,
   that was clearly *intended* to be used for `exec_move` but never wired in —
   grep for `exec_move_now` to find it; this dead variable should be removed
   once the real fix lands (or better: replaced entirely by the
   speed-mode-derived value, see the fix below).

**Consequences of the bug:**
- `move_direction` labels don't reflect repulsion/braking/turn-limiting/
  push-kick adjustments — network gets a *systematically imprecise* direction
  target on every single row, not just edge cases.
- `sprint` labels come from the *freshly decided* order's static `sprint`
  field, not from what speed mode the engine actually resolved to after
  braking curves etc. (e.g. a `MoveOrder(sprint=True)` still braking to
  `JOG`/`STANDSTILL` near arrival gets mislabeled as `sprint=1.0`).
- `exec_move=1.0` is hardcoded even at the exact tick the player has fully
  arrived and stopped (`STANDSTILL`) — mislabels a stop as "moving".
- The degenerate-direction guard (`length < 1e-6`) **discards the entire BC
  frame** (`BCLabel.invalid()`), including `kick_this_tick`/`tackle_attempt`/
  `kick_direction`/`kick_power`/`kick_spin` supervision that was already
  correctly computed earlier in the function from `player.kicked_this_tick`
  et al. — these are exactly the arrival-tick/possession-gain-tick frames
  where a push-kick or tackle is *most* likely to have just fired, so this
  silently drops some of the rarest, highest-value supervision signal in the
  entire dataset (recall `pos_weight_kick≈133`, `pos_weight_tackle_attempt≈119`
  from the last training run — these events are already extremely sparse,
  losing any of them is costly).

### The fix

Replace the manual-geometry derivation with: **run the decided order's
`execute()` once (with player state snapshotted/restored so it has zero
effect on the real match), then read `player.desired_direction`/
`player.desired_speed_mode` back.**

**Why snapshot/restore is required:** `Order.execute()` methods have real
side effects beyond setting `desired_direction`/`desired_speed_mode` — e.g.
`MoveOrder.execute()`'s push-kick path calls `player.kick_direct()` (which
mutates `match.ball`, sets `player.kicked_this_tick`, starts release-grace
timers, fires `on_kick` callbacks), and any order can mutate
`self._overshoot_timer_s`/`self.reached_target`/`self.status` (order-internal
state) or (for `ChaseTackleOrder`) actually resolve a tackle via
`match._attempt_tackle_contact()`. None of that may leak into the real
simulation just because we're generating a label — `phase1_labels()` is
called at label-generation time (both offline demo recording and PPO rollout
BC-aux label collection), interleaved with real `env.step()` calls, so any
side effect here would corrupt the actual training rollout.

**Concrete implementation plan** (edit `phase1_labels()` in
`src/footballcoach/ai/ppo/bc.py`):

```python
def phase1_labels(env, player_id: str = None) -> BCLabel:
    from footballcoach.rules_ai import Phase1RulesAI
    from footballcoach.orders import MoveOrder, GetPossessionOrder, ChaseTackleOrder

    try:
        match = env.match
        if player_id is None:
            player_id = env.trainee_player_id
        if match is None:
            return BCLabel.invalid()
        player = match.player_by_id(player_id)
    except (KeyError, AttributeError):
        return BCLabel.invalid()

    # ... ai_type / opponent_ai_type computation: UNCHANGED, keep as-is ...

    # ... kick_this_tick / kick_direction / kick_power_fraction / kick_spin
    #     computation from player.kicked_this_tick / player.last_kick_*:
    #     UNCHANGED, keep as-is (this is already correctly execution-sourced) ...

    # ... tackle_attempt computation via ChaseTackleOrder isinstance check +
    #     are_touching(): UNCHANGED, keep as-is (this is order-TYPE used as
    #     context to detect a physical contact-tackle event, which is fine —
    #     it's not deriving a *direction*/*speed* label from order fields) ...

    current_exec = player.current_order

    # Decision heads: fresh decision from Phase1RulesAI, UNCHANGED mechanism.
    player.current_order = None
    try:
        Phase1RulesAI().act(player, match, trial_tick=0)
        order = player.current_order
    finally:
        player.current_order = current_exec  # always restore

    # --- NEW: derive move_direction / sprint / exec_move from ACTUALLY
    # RUNNING the decided order and reading back player.desired_direction /
    # player.desired_speed_mode, instead of hand-deriving geometry from order
    # fields. See ai/knowledge.md "Orders vs execution-network labels
    # boundary" for why this is required.
    move_direction = None
    sprint_label = 0.0
    exec_move_label = 0.0
    move_region_center = None

    if isinstance(order, (MoveOrder, GetPossessionOrder)):
        # Snapshot everything execute() might mutate, so this exploratory
        # call has ZERO effect on the real simulation.
        _snap_pos = player.position
        _snap_vel = player.velocity
        _snap_heading = player.heading_rad
        _snap_desired_dir = player.desired_direction
        _snap_desired_speed = player.desired_speed_mode
        _snap_kicked_this_tick = player.kicked_this_tick
        _snap_last_kick_dir = player.last_kick_direction
        _snap_last_kick_power = player.last_kick_power_fraction
        _snap_last_kick_spin = player.last_kick_spin
        _snap_current_order = player.current_order
        _snap_on_possession_gained = player.on_possession_gained
        _snap_ball_possessed_by = match.ball.possessed_by
        _snap_ball_velocity = match.ball.velocity
        _snap_ball_position = match.ball.position
        # order is a FRESH object from Phase1RulesAI().act() above (not
        # current_exec), so its own internal state (e.g. MoveOrder's
        # _overshoot_timer_s/reached_target) starts clean — no need to
        # snapshot the order itself, just discard it after reading direction.
        player.current_order = order
        try:
            # dt: use the env's actual sim tick size so braking-distance
            # calculations inside execute() match what real ticks would see.
            _dt = getattr(env, "_dt_s", None) or match.movement_params.__dict__.get("dt_s", 1 / 30)
            order.execute(player, match, _dt)
            if player.desired_speed_mode is not None:
                from footballcoach.engine.movement import SpeedMode
                move_direction_raw = player.desired_direction
                if move_direction_raw.length_xy() > 1e-6:
                    _n = move_direction_raw.normalized()
                    move_direction = np.array([_n.x, _n.y], dtype=np.float32)
                sprint_label = 1.0 if player.desired_speed_mode is SpeedMode.SPRINT else 0.0
                exec_move_label = 0.0 if player.desired_speed_mode is SpeedMode.STANDSTILL else 1.0
        finally:
            # Restore EVERYTHING — this call must be perfectly invisible to
            # the real simulation.
            player.position = _snap_pos
            player.velocity = _snap_vel
            player.heading_rad = _snap_heading
            player.desired_direction = _snap_desired_dir
            player.desired_speed_mode = _snap_desired_speed
            player.kicked_this_tick = _snap_kicked_this_tick
            player.last_kick_direction = _snap_last_kick_dir
            player.last_kick_power_fraction = _snap_last_kick_power
            player.last_kick_spin = _snap_last_kick_spin
            player.current_order = _snap_current_order
            player.on_possession_gained = _snap_on_possession_gained
            match.ball.possessed_by = _snap_ball_possessed_by
            match.ball.velocity = _snap_ball_velocity
            match.ball.position = _snap_ball_position

    if isinstance(order, MoveOrder):
        move_region_center = np.array(
            [order.target_position.x, order.target_position.y], dtype=np.float32
        )
        return BCLabel(
            move=1.0,
            sprint=sprint_label,
            move_direction=move_direction,
            move_region_center_m=move_region_center,
            kick_this_tick=kick_this_tick,
            tackle_attempt=tackle_attempt,
            exec_move=exec_move_label,
            ai_type=ai_type,
            opponent_ai_type=opponent_ai_type,
            kick_direction=kick_direction,
            kick_power_fraction=kick_power_fraction,
            kick_spin=kick_spin,
        )
    elif isinstance(order, GetPossessionOrder):
        ball = match.ball
        move_region_center = np.array([ball.position.x, ball.position.y], dtype=np.float32)
        return BCLabel(
            get_possession_extra=1.0,
            sprint=sprint_label,
            move_direction=move_direction,
            move_region_center_m=move_region_center,
            kick_this_tick=kick_this_tick,
            tackle_attempt=tackle_attempt,
            exec_move=exec_move_label,
            ai_type=ai_type,
            opponent_ai_type=opponent_ai_type,
            kick_direction=kick_direction,
            kick_power_fraction=kick_power_fraction,
            kick_spin=kick_spin,
        )
    else:
        return BCLabel.invalid()
```

**Important implementation notes / things to verify while implementing:**

1. **`env._dt_s` attribute name**: verify the actual attribute name on
   `ScenarioEnv` for the sim tick size (grep `_dt_s` in
   `src/footballcoach/ai/env/scenario_env.py` — it's referenced already in
   `record_demonstrations.py` as `env._dt_s`, e.g.
   `sample_ticks = max(1, round(sample_interval_s / env._dt_s))`). Use that
   exact attribute; do not guess a different one.
2. **`GetPossessionOrder.execute()` has a stateful early-exit** for
   "already have the ball" (`if match.ball.possessed_by == player.player_id
   or self._possession_gained: return True`) — since we're calling `execute()`
   on a **fresh** `GetPossessionOrder` instance (from `Phase1RulesAI.act()`),
   `self._possession_gained` starts `False`, so this reduces to checking
   `match.ball.possessed_by == player.player_id` — safe. But
   `GetPossessionOrder.execute()` also does:
   ```python
   if not self._callback_registered:
       player.on_possession_gained = _on_possession
       self._callback_registered = True
   ```
   This assigns a **real callback closure onto the live `player` object**
   that could fire later on a genuine possession-gain event, if not restored.
   **This is exactly why `player.on_possession_gained` must be snapshotted
   and restored** (already included in the snapshot list above) — do not
   skip this, it's a subtle dangling-closure bug otherwise.
   Also note `GetPossessionOrder.execute()` calls
   `match._run_get_possession_behaviour(player, dt)` when the ball is loose
   or held by someone else — check that function
   (`src/footballcoach/engine/match.py`, grep `_run_get_possession_behaviour`)
   for any additional state it mutates (e.g. does it touch `match.ball`
   directly, or attempt a tackle via `match._attempt_tackle_contact()`?) and
   extend the snapshot/restore list if so. **Read this function fully before
   finalizing the snapshot list — the list above is a best-effort starting
   point based on what's visible in `orders.py`, not a guaranteed-complete
   audit of every transitive side effect.**
3. **Push-kick side effects**: `MoveOrder.execute()`'s push-kick path calls
   `self._do_push_kick()` → `player.kick_direct()` → `kick_ball()` (mutates
   `match.ball.velocity`/`position`/`possessed_by`, calls
   `match._start_release_grace()`, sets `player.kicked_this_tick = True`,
   fires `player.on_kick`). All ball-state mutations are covered by the
   ball snapshot/restore above. `match._start_release_grace()` — check
   `src/footballcoach/engine/match.py` for what state this touches (likely a
   `dict`/`set` keyed by player_id with a grace-period timer) and add it to
   the snapshot/restore list if it's not transient/idempotent. **Do not
   assume — check the function body.**
4. **Do NOT let `player.on_kick`/`player.on_tackle` fire real callbacks
   during this exploratory `execute()` call** — in `record_demonstrations.py`,
   these callbacks are wired to `_record_now(player_id=...)`, i.e. calling
   `order.execute()` here could recursively trigger ANOTHER BC-label sample
   append while already inside `phase1_labels()`! This is a serious
   correctness risk if not handled. **Recommended fix:** temporarily set
   `player.on_kick = None` and `player.on_tackle = None` before calling
   `execute()`, and restore them in the `finally` block (add to the snapshot
   list above: `_snap_on_kick = player.on_kick; _snap_on_tackle =
   player.on_tackle`, set both to `None` before `execute()`, restore after).
   **This is not optional — verify this is in the final implementation.**
5. **Performance**: this makes `phase1_labels()` noticeably more expensive
   (a full order-execute pass with snapshot/restore, vs. simple arithmetic).
   `phase1_labels()` is called extremely frequently (every decision step for
   both players during demo recording AND every PPO rollout step when BC-aux
   is active). After implementing, profile a short demo-recording run (e.g.
   `--n-episodes 50`) before vs. after and confirm recording time hasn't
   regressed by more than ~20-30%; if it has, consider whether the
   snapshot/restore can be narrowed (e.g. skip position/velocity/heading
   snapshot for `GetPossessionOrder`, which doesn't call `kick_direct`
   itself... but it DOES call `_run_get_possession_behaviour` which might).
   Do not prematurely optimize before measuring.
6. **`move_region_center_m`**: keep this sourced from the order's
   `target_position`/ball-position exactly as before — `move_region_center`
   is a **decision-network** output (feeds `move_region_center` on
   `DecisionHeadsRaw`, not an execution-network field), so it is legitimately
   fine to source from the order field per the boundary rule above. Do not
   change this part.

### Update `ai/knowledge.md` and `knowledge.md` documentation

The high-level rule was already added to `ai/knowledge.md` in a prior session
(look for "CRITICAL: Orders vs execution-network labels boundary" near the
top of the file, right after the `Install` section) — **verify it is present
and matches the wording below; if a previous edit attempt was interrupted
(tooling was disabled mid-session), add it now:**

```markdown
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
for the concrete fix history and rationale.
```

Also add a shorter cross-reference note to the top-level
`src/footballcoach/knowledge.md`'s existing "Neural network / Orders boundary
(IMPORTANT)" section (it already documents the NN-never-issues-orders rule;
add one paragraph immediately after it):

```markdown
**Corollary — BC label generation must respect the same boundary in
reverse:** code that derives BC *labels* for the execution network (see
`ai/ppo/bc.py`'s `phase1_labels()`) must source execution-level fields
(`move_direction`, `sprint`, `exec_move`, kick vector, `tackle_attempt`) from
`player.desired_direction`/`player.desired_speed_mode`/`player.kicked_this_tick`/
`player.last_kick_*` — i.e. what the ORDER MACHINERY ACTUALLY PRODUCED on the
player that tick — never by re-deriving geometry from an Order's own fields
(that bypasses braking/repulsion/turning/push-kick logic). See
`ai/knowledge.md`'s "Orders vs execution-network labels boundary" section for
the full rule and `agent_plans/bc_execution_label_boundary_and_followups.md`
for the bug history.
```

Also update the repo memory file (if the executing agent has access to a
memory tool; if using this document as a raw prompt with no memory tool, skip
this step and just do the two doc edits above): append to
`/memories/repo/rules.md` under a "## BC label notes" section (create if not
present) — a short pointer to this rule so future sessions don't reintroduce
the bug. Keep it to 3-5 lines, this is memory not documentation.

### Order-subclass audit (already done during planning — no code changes expected, but VERIFY)

All `Order` subclasses in `src/footballcoach/orders.py` were reviewed during
planning and confirmed to ALREADY correctly restrict their `execute()` output
to `player.desired_direction`/`player.desired_speed_mode` (via
`_compute_movement_intent()` or direct `Vector3.zero()`/`SpeedMode.STANDSTILL`
assignment) plus `player.kick_direct()`/tackle-contact resolution — i.e. the
Order layer itself is clean; **the bug is entirely confined to
`phase1_labels()` misusing Order fields, not to the Orders themselves.**
Confirmed clean: `MoveOrder`, `KickOrder`, `PassOrder`, `ChaseTackleOrder`,
`SaveOrder`, `StopOrder`, `GetPossessionOrder`, `MarkOrder`, `ShootOrder`.

**Still do a final pass during implementation** (grep `player\.\w+\s*=` across
`orders.py`) to be 100% sure no subclass assigns any OTHER `Player` field
that a future BC-label function might be tempted to read directly instead of
`desired_direction`/`desired_speed_mode` — if you find one, flag it, don't
silently "fix" it without understanding why it's there (e.g. `SaveOrder`'s
`player.position = target_position.with_z(...)` snap-to-position on the final
tick IS a legitimate special case for goalkeepers, not a bug — GKs are not a
phase-1 concern here, don't touch it, just note it exists).

---

## Part 1: `bc_loss_from_tensor()` breakdown reporting fix

### The problem

`bc_loss_from_tensor()` in `src/footballcoach/ai/ppo/bc.py` computes
`kick_direction`/`kick_power`/`kick_spin` losses gated by `kicked_mask =
labels[:, _I_KICK_THIS_TICK] > 0.5` — i.e. `torch.where(kicked_mask, mse,
zeros)`. This is CORRECT for the loss itself (you can't supervise "what power
should this kick have" on a non-kick row — there's no target). But the
breakdown dict reported in logs averages this **zeroed-out tensor over ALL
valid rows**, not just the kicked ones:

```python
"kick_power":     float(kick_power_loss_per[valid].mean()),
```

Since kicks are extremely rare (`pos_weight_kick≈133` in the last training
run, i.e. ~1 kick row per 133 total rows), this mean is diluted ~133:1 by
hard zeros and shows as `0.000` in every log line even when the actual
per-kick loss is meaningfully large/small and moving during training. This
makes it look like kick_power/kick_spin/kick_direction supervision is dead or
broken, when it's actually just diluted into invisibility by the reporting
math — a genuinely confusing, misleading diagnostic.

### The fix

Change the breakdown computation for `kick`, `kick_direction`, `kick_power`,
`kick_spin` to average **only over rows where `kicked_mask` is true** (within
the valid set), falling back to `0.0` (not `NaN`) when no kicks occurred in
that minibatch/epoch (common in small batches given how rare kicks are).

Edit `bc_loss_from_tensor()` in `src/footballcoach/ai/ppo/bc.py`. Locate the
`if return_breakdown:` block near the end of the function (grep for
`breakdown = {`):

```python
    if return_breakdown:
        # kick/kick_direction/kick_power/kick_spin are gated to zero on
        # non-kick rows (see kicked_mask above) — averaging over ALL valid
        # rows dilutes them ~pos_weight_kick:1 by hard zeros, making genuine
        # loss changes invisible in logs. Report their mean over KICKED rows
        # only (within the valid set) instead, falling back to 0.0 (not NaN)
        # when no kicks occurred in this batch (common given how rare kicks
        # are — see pos_weight_kick in ai/knowledge.md).
        kicked_valid_mask = valid & (labels[:, _I_KICK_THIS_TICK] > 0.5)
        _n_kicked_valid = int(kicked_valid_mask.sum().item())
        def _kicked_mean(per_row_loss: torch.Tensor) -> float:
            if _n_kicked_valid == 0:
                return 0.0
            return float(per_row_loss[kicked_valid_mask].mean())

        breakdown = {
            "decision":       float(dec_loss[valid].mean()),
            "exec_bce":       float(exec_bce_loss[valid].mean()),
            "sprint":         float(sprint_loss[valid].mean()),
            "move":           float(move_loss[valid].mean()),
            "tackle_attempt": float(tackle_attempt_loss[valid].mean()),
            "direction":      float(dir_loss_per[valid].mean()),
            "region":         float(region_loss_per[valid].mean()),
            "kick":           float(kick_loss[valid].mean()),  # BCE over ALL rows is meaningful (target is 0 or 1 on every row, unlike power/spin/direction which have no target on non-kick rows) — do NOT change this one to kicked-only.
            "kick_direction": _kicked_mean(kick_dir_loss_per),
            "kick_power":     _kicked_mean(kick_power_loss_per),
            "kick_spin":      _kicked_mean(kick_spin_loss_per),
        }
        return total, breakdown
```

**Important distinction to preserve:** `kick` (the Bernoulli BCE for
"did/should a kick happen this tick") has a well-defined target (0 or 1) on
EVERY row, not just kick rows — so its breakdown mean over all valid rows is
already meaningful and should NOT be changed to kicked-only averaging. Only
`kick_direction`/`kick_power`/`kick_spin` (which have NO defined target on
non-kick rows) need the kicked-only treatment. Do not conflate these two —
re-read the code comment above before implementing to keep this distinction
clear in the final diff.

Also add a matching `"kick_dir_cos"` breakdown-adjacent quantity if useful —
NOT required, `ppo_trainer.py` already computes `kick_dir_cos` separately via
its own masked accumulation loop (see Part 2 below); no change needed there
beyond what Part 2 already covers.

### Log precision bump

In `src/footballcoach/ai/ppo/ppo_trainer.py`, the BC epoch log lines print
breakdown values with `:.3f` — bump specifically the kick-related breakdown
keys to more decimals so small-but-real values aren't rounded to `0.000`.
Locate (grep for `bkdn_str = `):

```python
bkdn_str = "  ".join(f"{k}={v/_bkdn_n:.3f}" for k, v in _bkdn_acc.items()) if _bkdn_n else ""
```

Change to format kick-related keys with 5 decimals and everything else with
3 (keeps the common-case line compact while giving kick metrics the
precision they need):

```python
_KICK_BKDN_KEYS = {"kick", "kick_direction", "kick_power", "kick_spin"}
bkdn_str = "  ".join(
    f"{k}={v/_bkdn_n:.5f}" if k in _KICK_BKDN_KEYS else f"{k}={v/_bkdn_n:.3f}"
    for k, v in _bkdn_acc.items()
) if _bkdn_n else ""
```

There are TWO occurrences of this pattern to update — grep `bkdn_str = ` and
`_bkdn_r_acc` (the BC-repair-loop variant, using `_bkdn_r_acc`/`_bkdn_r_n`
instead of `_bkdn_acc`/`_bkdn_n` — same fix, mirrored variable names). Apply
the identical change to both. Define `_KICK_BKDN_KEYS` once near the top of
whichever function scope covers both usages, or duplicate the small literal
set inline at each site if they're in genuinely separate function scopes —
check by reading the surrounding function boundaries before choosing.

---

## Part 2: Re-record demonstrations (mandatory after Part 0)

Part 0's fix changes what `phase1_labels()` returns for `move_direction`/
`sprint`/`exec_move` on essentially every row (not just edge cases) — this is
a **behavioral** change to the recorded dataset, not a schema change (the
`.npz` column layout / `BC_LABEL_DIM` is unchanged from the kick-supervision
plan's final state = 24). Existing recordings under `demonstrations/phase1/`
were captured with the OLD (buggy) label derivation and must be discarded and
re-recorded, or all downstream training will continue learning from the
imprecise geometric-proxy labels.

```bash
rm -f demonstrations/phase1/*.npz
uv run python -m footballcoach.ai.scripts.record_demonstrations \
    --phase 1 --n-episodes 4000 --episodes-per-file 8 \
    --output demonstrations/phase1/ --seed 42 2>&1 | tee /tmp/record_demos.log
```

(4000 episodes matches the most recent recording size used in this repo's
history — check `training_runs.log`/terminal history for the actual most
recent `--n-episodes` value used and match it, don't blindly assume 4000 is
still current by the time this plan is executed.)

After recording, spot-check the new dataset's `move_direction` distribution
looks sane (e.g. via a short Python REPL/script loading a `.npz` file and
checking `bc_labels[:, 7:9]` aren't degenerate/all-zero) before kicking off a
long training run on it — cheap insurance against a snapshot/restore bug in
Part 0 silently zeroing out all `desired_direction` values.

---

## Part 3: Full test suite pass

Run the full suite and confirm no regressions, paying special attention to:
- `tests/ai_unit/test_bc.py` — has extensive `phase1_labels()`/
  `bc_loss_from_tensor()` coverage already; the `TestBCLossFromTensor` class's
  kick-related tests (`test_kick_direction_loss_zero_when_aligned`,
  `test_kick_power_loss_matches_mse`, etc. — added in the prior kick-
  supervision plan) will need their `breakdown["kick_power"]`/
  `breakdown["kick_direction"]`/`breakdown["kick_spin"]` assertions re-checked
  against the new kicked-only-averaging semantics from Part 1 — these tests
  already construct single-kicked-row batches (`n=1`, `kick_this_tick=1.0`),
  so the kicked-only mean should equal the same value as before for those
  specific tests (averaging one row over itself is a no-op) — but VERIFY,
  don't assume, since a batch with `n=1` and kicked-only averaging is
  degenerate in a way that could mask a division-by-zero-guard bug (the
  `_n_kicked_valid == 0` fallback path) if not explicitly tested.
- Add a NEW test in `TestBCLossFromTensor` for the diluted-vs-undiluted
  breakdown distinction: construct a batch with N=10 rows, only 1 kicked,
  assert `breakdown["kick_power"]` equals the loss computed on that ONE row
  (not divided by 10). This directly guards the Part 1 fix and is the kind of
  test that would have caught the original dilution bug.
- Add a NEW test for Part 0's fix: construct a `phase1_labels()` scenario
  where the trainee is exactly at the GetPossessionOrder ball-target (e.g.
  position the ball exactly on the player) and confirm the returned
  `BCLabel.valid is True` (not discarded) with sensible
  `kick_this_tick`/`tackle_attempt` fields intact, and `move_direction`
  reflects `player.desired_direction` post-`execute()` (which may be `None`
  if `desired_speed_mode` ends up `STANDSTILL`/`None`) rather than raising or
  returning `BCLabel.invalid()`. This test should live in
  `tests/ai_unit/test_bc.py` near the existing `phase1_labels` test class
  (grep for `class Test.*Phase1Labels` or similar — check exact existing
  class name before adding).
- Add a test confirming `phase1_labels()`'s exploratory `execute()` call has
  ZERO effect on real match/player state — e.g. call `phase1_labels(env)`
  twice in a row with no `env.step()` between calls and assert the return
  values are IDENTICAL (same `move_direction`, same `kick_this_tick`, etc.)
  AND that `match.ball.position`/`player.position`/`player.velocity` are
  bit-identical to their pre-call values after each call. This is the most
  important regression guard for the snapshot/restore correctness described
  in Part 0 — a subtle omission in the restore list (e.g. forgetting
  `on_possession_gained`) could cause a SECOND call to behave differently
  from the first, which this test would catch.

Full suite command (per this repo's rules — NEVER pipe test output through
head/tail, always show full output):

```bash
uv run pytest tests/ -q
```

If any pre-existing failures are found that are clearly unrelated to this
plan's changes (e.g. flaky BC-fidelity smoke tests under `pytest-xdist`
parallel execution — this repo has a known instance of
`test_bc_pretrain_then_score` occasionally failing under `-n auto` parallel
runs due to shared global RNG state across xdist workers, but passing in
isolation), do not "fix" them as part of this plan unless they are actually
caused by these changes — verify by running the specific failing test alone
(`uv run pytest tests/path/to/test.py::test_name -q`) before concluding it's
pre-existing flakiness vs. a real regression from this plan's changes.

---

## Part 4: PPO epoch log formatting (readability)

### Problem statement

The per-rollout PPO log line (in `_train`'s main loop, `ppo_trainer.py`,
grep for `log.info(\n                    f"step={self._total_steps:,}`) and
the offline BC epoch log lines (`pretrain_combined`'s Phase 1 loop, grep for
`BC epoch {epoch`) are extremely dense single-line log statements packing in
10-20+ named metrics with ad-hoc separators (`|`, `  `, `[...]`). This is
information-dense but very hard to visually parse — the user explicitly asked
for this to be reformatted to be easier to read, "tabulate it maybe, group
stuff together", and is fine with it taking more vertical space.

### Design approach

Convert the single mega-line log statements into small, labeled multi-line
blocks with consistent column alignment, grouping semantically-related
metrics together. Do NOT change what data is logged (all existing fields must
remain present, just reformatted) — this is a pure readability refactor, not
a metrics change. Do NOT remove any diagnostic capability.

**Recommended format** (adapt exact field names/values from the current
single-line implementation — this is illustrative structure, not exact
copy-paste code, since the surrounding variables (`metrics`, `ha`,
`rollout_components`, etc.) need to be read from the actual current
implementation before drafting the multi-line f-strings):

```
────────────────────────────────────────────────────────────────────
[PPO] step=128,000  sps=283  rew=8.76/–  (5 epochs, kl=0.16, early-stop)
  loss    pol=0.0200  val=1.0000(x0.65)=0.6500  ent=0.2500
  value   V=4.32±2.10  R=8.76±5.43  adv=0.12±1.05
  bc      coeff=0.170  loss=2.8400
  moves   log_std=[0.0100,0.0000]  grad=1.2e-03
  heads   mv=22  gp=78  emv=100  spr=66  kck=0  tk=0  sh=0  hld=0
          ta_p=0.0012  kk_p=0.0004
  vs      rules(18): 65%/12%   immobile(20): 90%   neural(16): 44%/25%
  reward  approach=+0.19  retreat=-0.04  heading=-0.05  gain_poss=+3.95
          progress=+0.25  lose_poss=-1.15  ball_out=-0.03  box_poss=+4.80
          speed=+1.95  opp_box=-0.70  stamina=-0.01
────────────────────────────────────────────────────────────────────
```

For the BC epoch lines (`pretrain_combined`'s Phase 1 loop):

```
  BC epoch 3/5  (91s)
    loss    bc=0.5240  val=1.0888(x1.0)=1.0888  rmse=5.67 (returns std=5.4)
    heads   dir_cos=0.965  kick_dir_cos=0.904  mv_p=1.000  spr_p=0.861  kk_p=0.022  tk_p=0.057
    bkdn    decision=0.002  exec_bce=0.414  sprint=0.105  move=0.000
            tackle_attempt=0.219  direction=0.071  region=0.034
            kick=0.09000  kick_direction=0.00200  kick_power=0.00000  kick_spin=0.00000
```

### Implementation guidance

1. Locate the exact log-line-building code blocks (do NOT guess — read the
   current implementation fully first, the exact variable names matter):
   - Main PPO rollout log: `ppo_trainer.py`, inside `train()`, the block
     building `comp_str`/`act_str`/`_val_diag_str`/`outcome_str`/`mv_ls_str`
     then the final `log.info(f"step={self._total_steps:,} | ...")` call.
   - BC pretrain epoch log: `pretrain_combined()`'s Phase 1 loop, the
     `log.info(f"  BC epoch {epoch + 1}/{n_epochs}: ...")` call.
   - BC-repair epoch log: same function/pattern, `_r` variable suffix variant
     (`log.info(f"  BC repair done (...)")` and the per-epoch equivalent
     inside the repair loop).
2. Replace each single f-string with a multi-line f-string (using `\n` and
   consistent leading-space indentation per grouped block) OR multiple
   sequential `log.info(...)` calls — prefer a SINGLE multi-line f-string per
   log "block" (so it appears as one atomic unit in log files/log viewers,
   not interleaved with other threads'/processes' log lines if ever run
   concurrently) unless there's a clear reason to split (there usually isn't
   here since this is single-threaded).
3. **Do not change any of the underlying metric computations** — this is a
   pure string-formatting change. If you find yourself needing to compute a
   NEW value to make the tabulation nicer (e.g. combining two existing values
   into a ratio for display), that's fine, but do not remove or alter
   existing metric computations.
4. Keep column widths reasonably consistent within each block (e.g. pad
   numeric fields to a fixed width with `:>7.4f`-style specifiers) so the
   "tabulated" look is achieved — but don't over-engineer this into a full
   ASCII-table library dependency; plain aligned strings are sufficient and
   keep this dependency-free.
5. After implementing, do a short live check: run a small training smoke run
   (or the existing `tests/ai_scenario/test_smoke.py`-style short rollout) and
   visually inspect the resulting log output in the terminal to confirm
   readability improved and no field was silently dropped. Diff the SET of
   metric names/values present before vs. after (e.g. grep every `=` key in
   an old sample log line vs. a new sample log line) to mechanically verify
   nothing was lost in the reformat.
6. Update any documentation that shows example log lines to match the new
   format — grep `ai_trainer_knowledge.md` for embedded example log lines
   (e.g. the "Offline BC epoch lines (during `pretrain_combined`)" section
   with the `BC epoch 5/50: bc_loss=0.52  dir_cos=0.34...` example, and the
   main PPO rollout log example if one exists elsewhere in that file) and
   update them to the new multi-line format so docs stay accurate.

---

## Part 5: Demonstration-recording logs — kick/tackle counts

### Problem statement

`record_demonstrations.py`'s periodic per-10-episodes log line (grep
`f"Ep {global_ep}/{total_episodes}"`) reports episode outcomes and a reward
component breakdown (via the `_comp_acc`/`REWARD_COMP_LABELS` mechanism), but
does NOT report how many kick/tackle events were actually recorded in that
window — given kicks/tackles are the rarest and most valuable BC signal
(recall `pos_weight_kick≈133`), the user wants explicit visibility into how
many kick/tackle samples are being captured per batch of episodes, printed
alongside the existing reward breakdown, so recording quality/coverage can be
sanity-checked without post-hoc `.npz` inspection.

### Implementation plan

Edit `src/footballcoach/ai/scripts/record_demonstrations.py`. The function
containing the recording loop is the one with the `_record_now(...)` closure
and the `for ep in range(n_episodes):` loop — locate it exactly (grep
`def _record_now` to find the enclosing function's `def` line above it).

1. Add two counters alongside the existing `_comp_acc`/`_comp_acc_episodes`
   accumulators (same scope, reset at the same point):
   ```python
   _kick_count_since_log = 0
   _tackle_count_since_log = 0
   ```
2. Increment these inside `_record_now(...)` (the function that appends a row
   to `bc_labels`) — count directly from the label's `kick_this_tick`/
   `tackle_attempt` fields (NOT from the `on_kick`/`on_tackle` callback firing,
   since `_record_now` is also called for the two-players-at-once timed
   samples where the flag might independently be true/false per player).
   Inside `_record_now`, after `label_arr = label.to_array()` is computed
   (or after `bc_labels.append(label_arr)`), add:
   ```python
   nonlocal _kick_count_since_log, _tackle_count_since_log
   if label.kick_this_tick > 0.5:
       _kick_count_since_log += 1
   if label.tackle_attempt > 0.5:
       _tackle_count_since_log += 1
   ```
   (Add `_kick_count_since_log`/`_tackle_count_since_log` to the existing
   `nonlocal steps_total, steps_valid` statement already present in
   `_record_now`, rather than a separate `nonlocal` line, for consistency
   with existing style — check the exact existing `nonlocal` line before
   editing so you extend it correctly rather than duplicating.)
3. In the periodic logging block (the `if global_ep % 10 == 0 or global_ep ==
   total_episodes:` block), add the kick/tackle counts to the log output.
   Two options — pick based on what reads most naturally next to the existing
   lines (read the exact current log statements before deciding):
   - Append to the main outcome line: add
     `f"  kicks={_kick_count_since_log}  tackles={_tackle_count_since_log}"`
     to the existing `log.info(f"Ep {global_ep}/{total_episodes} | steps: ...")`
     call.
   - OR add as its own line immediately after the reward-breakdown line
     (recommended, keeps the main outcome line from getting even longer):
     ```python
     log.info(
         f"  kick/tackle samples (since last log): "
         f"kicks={_kick_count_since_log}  tackles={_tackle_count_since_log}"
     )
     ```
4. Reset both counters to `0` at the same point `_comp_acc.clear()` /
   `_comp_acc_episodes = 0` already happen (end of the periodic-logging `if`
   block) — find that exact reset code and add the two new resets right next
   to it for consistency.
5. Also consider (optional, nice-to-have, only if trivial to add without
   restructuring): a cumulative kick/tackle total across the WHOLE recording
   run (not just "since last log"), printed once at the very end of
   recording alongside any final summary stats already printed after the
   `for ep in range(n_episodes):` loop finishes (grep for what's printed
   after the loop, e.g. total steps/valid steps summary) — if such a summary
   block exists, add total kick/tackle counts there too using two more
   counters (`_kick_count_total`, `_tackle_count_total`) that are NEVER reset
   (incremented in the same place as the "since log" counters, just not
   cleared). Skip this if the surrounding code structure makes it awkward —
   it's a bonus, not a requirement.

### Tests

Check `tests/ai_scenario/` or wherever `record_demonstrations.py`'s
recording function is unit/integration tested (grep for
`record_demonstrations` or the actual function name it exposes, e.g.
`record_phase_demonstrations` or similar — check the file's public API by
reading its `if __name__ == "__main__":` / argparse setup and any function
that's clearly the "library" entry point vs. CLI glue) for existing coverage.
If a test already runs a short recording and inspects `.npz` output, extend
it to also check that SOME kicks/tackles were recorded (using a fixed seed
and enough episodes to guarantee at least one of each) OR add a lightweight
new test if none exists, following the existing test file's fixture/mocking
patterns exactly (do not introduce a new testing style inconsistent with the
rest of that test file).

---

## Summary checklist for the implementing agent

Work through these IN ORDER (later parts depend on earlier ones):

- [ ] **Part 0a**: Confirm/add the "Orders vs execution-network labels
      boundary" section to `ai/knowledge.md` (verify wording matches this
      plan; check it wasn't partially added already from an interrupted
      prior session).
- [ ] **Part 0b**: Add the corollary paragraph to `src/footballcoach/knowledge.md`'s
      existing "Neural network / Orders boundary (IMPORTANT)" section.
- [ ] **Part 0c**: (Optional, if a memory tool is available) Add a short note
      to repo memory pointing at this rule.
- [ ] **Part 0d**: Re-verify the Order-subclass audit (quick grep pass,
      documented above) — expect no code changes here, just confirmation.
- [ ] **Part 0e**: Rewrite `phase1_labels()`'s execution-field derivation
      (`move_direction`/`sprint`/`exec_move`) per the snapshot/execute/restore
      design above. Read `_run_get_possession_behaviour()` and
      `match._start_release_grace()` FIRST to finalize the exact
      snapshot/restore field list — do not blindly copy the illustrative list
      in this plan without verifying it against the real function bodies.
      Ensure `player.on_kick`/`player.on_tackle` are nulled during the
      exploratory `execute()` call to prevent recursive `_record_now()`
      re-entrancy during demo recording.
- [ ] **Part 1**: Fix `bc_loss_from_tensor()`'s breakdown reporting to average
      `kick_direction`/`kick_power`/`kick_spin` over kicked rows only (NOT
      `kick`, which keeps its all-rows average). Bump log decimal precision
      for kick-related breakdown keys in both `ppo_trainer.py` log sites.
- [ ] **Part 2**: Re-record `demonstrations/phase1/*.npz` (check current
      episode count convention from recent terminal history before running).
      Spot-check the new dataset's `move_direction` column isn't degenerate.
- [ ] **Part 3**: Add the new regression tests described (dilution-fix test,
      arrival-tick-frame-not-discarded test, execute()-is-side-effect-free
      test). Run `uv run pytest tests/ -q` (full output, never piped through
      head/tail per repo convention) and confirm green, investigating any
      failure to determine pre-existing-flake vs. real regression before
      moving on.
- [ ] **Part 4**: Reformat the PPO rollout log line and BC epoch log lines
      into tabulated multi-line blocks without losing any existing metric.
      Update `ai_trainer_knowledge.md`'s embedded example log lines to match.
- [ ] **Part 5**: Add kick/tackle-count logging to
      `record_demonstrations.py`'s periodic log block, plus a test if
      reasonable coverage doesn't already exist.
- [ ] Final: re-run the full test suite one more time after ALL parts are
      complete, and do a short live training smoke run
      (`uv run python -m footballcoach.ai.scripts.train --phase 1 --seed 42
      --bc-dataset demonstrations/phase1/ --total-steps 5000` or similar
      small step count) to visually confirm the new log formatting renders
      correctly end-to-end and no exceptions are raised by the Part 0 fix
      under real PPO-rollout-time BC-aux label collection (not just offline
      demo recording) — this exercises a different call path
      (`_ppo_update`'s BC-aux path, not `record_demonstrations.py`'s loop)
      that must also be verified working.
