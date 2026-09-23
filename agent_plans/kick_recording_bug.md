# `PassOrder`/`ShootOrder` skip `_finish_kick` — kick bookkeeping silently missing

> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

## 0. Status

**Found, not fixed.** Discovered 2026-09-23 while building an unrelated UI feature (a
kick/tackle leg-swing animation) and confirming, by tracing a real `Match`, that a
`PassOrder`-triggered kick launched the ball correctly but never set
`player.last_kick_direction`. Investigated further at the user's request
("that bug seems potentially bad. Where is `last_kick_direction` used?"). Not yet
handed to whoever owns `orders.py`/the training pipeline; this file is that hand-off.

No code has been changed as part of this investigation — this is a report only.

---

## 1. The bug, confirmed by reading the code

[`Player._finish_kick`](../src/footballcoach/entities/player.py#L374-L403) is the
**only** place that does this bookkeeping, after a kick's physics has already run:

```python
self.kicked_this_tick = True
self.kick_count += 1
self.last_kick_direction = (_vel * (1.0 / _vel_len)) if _vel_len > 1e-6 else None
self.last_kick_power_fraction = float(adjusted_power)
self.last_kick_spin = spin
```

(plus clearing first-touch state and firing `on_kick`). `_finish_kick` is called from
exactly two methods, both documented as its callers:
[`kick_direct`](../src/footballcoach/entities/player.py#L407) and
[`kick_with_direction`](../src/footballcoach/entities/player.py#L459).

[`KickOrder.execute`](../src/footballcoach/orders.py#L712) calls `player.kick_direct(...)`
— safe, goes through `_finish_kick`.

[`PassOrder.execute`](../src/footballcoach/orders.py#L744) does **not**. It calls
`pass_ball(...)` ([`engine/kicking.py`](../src/footballcoach/engine/kicking.py#L511)) —
the raw physics function — directly, then fires the callback itself, inline
([orders.py:782](../src/footballcoach/orders.py#L782)):

```python
pass_ball(match.ball, player.position, pass_target, ...)
match._log_debug(f"{player.player_id} passed to {pass_target}")
if player.on_kick is not None:
    player.on_kick(player)
```

[`ShootOrder.execute`](../src/footballcoach/orders.py#L1166) does the exact same thing
with `kick_ball(...)` ([orders.py:1235](../src/footballcoach/orders.py#L1235)):

```python
kick_ball(match.ball, player.position, self.aim_point, ...)
match._log_info(f"{player.player_id} shot at goal  power={self.power_fraction:.2f}")
if player.on_kick is not None:
    player.on_kick(player)
```

**Net effect:** for a kick launched via `PassOrder` or `ShootOrder`, the ball's physics
is correct and `on_kick` *does* fire — but `kicked_this_tick`, `kick_count`,
`last_kick_direction`, `last_kick_power_fraction`, and `last_kick_spin` are all left
exactly as they were before the kick (stale or `None`/`False`), as if nothing had
happened. Anything that reads those fields rather than reacting to `on_kick` itself is
blind to passes and shots.

**Reproduced live** (not just read from the source): a real `Match` with one player
possessing the ball, `actions.pass_to(player, target)`, then `match.step()` — the ball's
velocity was immediately the correct post-kick value, but `player.kicked_this_tick` was
`False` and `player.last_kick_direction` was `None`, on that tick and every tick after.

---

## 2. Where this actually bites — traced, not guessed

**Confirmed safe:** the trainee neural network's own kicks.
[`apply_nn_action.py`](../src/footballcoach/ai/action/apply_nn_action.py#L123) calls
`player.kick_with_direction(match, ...)` **directly** — not through any `Order` at all —
which does go through `_finish_kick` correctly. The core PPO/BC gradient signal driving
the network actually being trained is not affected by this bug.

**Confirmed exposed:** `rules_ai.py` issues `ShootOrder` in
[4 places](../src/footballcoach/rules_ai.py) (grepped: lines with `ShootOrder(` at
~628, ~638, ~645, ~736) and issues `PassOrder` **nowhere** (grepped, zero matches). So
in this codebase specifically, the practical shape of the bug is "a rules-AI player's
shot is invisible to `kicked_this_tick`/`kick_count`/`last_kick_direction`" — passing is
irrelevant here only because nothing currently calls `PassOrder` from AI decision code
(it's still broken and would bite the moment anything does, e.g. a human `P` pass, or
future AI passing logic).

**Confirmed exposed, and worse than first thought:** `bc.py`'s demonstration-label
generation. [`_counterfactual_rules_ai_label`-style
code](../src/footballcoach/ai/ppo/bc.py#L495-L530) (the block that computes `kick_this_tick`/
`kick_direction`/`kick_power_fraction`/`kick_spin` BC labels) works by constructing a
**fresh** order from `Phase1RulesAI().act(...)` and executing it in a sandboxed,
rng-snapshotted, callback-suppressed copy of the tick, then reading
`player.kicked_this_tick` / `player.last_kick_direction` off the result
([bc.py:516-529](../src/footballcoach/ai/ppo/bc.py#L516-L529)):

```python
if player.kicked_this_tick:
    kick_this_tick = 1.0
    if player.last_kick_direction is not None:
        kick_direction = np.array([...])
    kick_power_fraction = player.last_kick_power_fraction
    ...
elif player.kick_armed and player.kick_armed_direction is not None:
    ...
```

`Phase1RulesAI` is a subclass of `_RulesBasedAI`
([rules_ai.py:201](../src/footballcoach/rules_ai.py#L201)), the same base class whose
`decide()` logic contains the 4 `ShootOrder(` call sites above. **So whenever this
counterfactual's fresh rules-AI decision is "shoot", `order.execute()` runs
`ShootOrder.execute()`, which never sets `kicked_this_tick` — the `if
player.kicked_this_tick:` branch is silently skipped, falls through to the `elif
player.kick_armed` branch (which is also unlikely to be true for an open-play shot with
the ball already possessed), and the row gets `kick_this_tick = 0.0`, i.e. "no kick",
even though the counterfactual rules AI's real physics-executed intent was to shoot.**

This means BC (and anything downstream of BC's recorded demonstrations — pretraining,
DAgger aggregation that reuses the same labelling code, etc.) is trained on kick labels
that are missing every counterfactual-rules-AI *shot*, silently downgraded to "did not
kick". This is not a hypothetical edge case — shooting is one of the two things
`_RulesBasedAI` uses a real `Order` for at all (the other, passing, doesn't reach this
code path since it's never called).

**Confirmed exposed, narrower:**
[`scenario_env.py`](../src/footballcoach/ai/env/scenario_env.py#L501-L509) reads
`player.kicked_this_tick` for both the trainee and `sec_players` (e.g. a rules-based
opponent) each physics tick, to build `trainee_kicks_this_step` /
`sec_kicks_this_step` and to detect kick-triggered possession transitions
(`kicked_this_tick and kick_armed`,
[scenario_env.py:585,595](../src/footballcoach/ai/env/scenario_env.py#L585)). A
rules-AI-controlled opponent's real `ShootOrder` shot during a live training episode
would be invisible to `sec_kicks_this_step` and to that possession-transition check. I
have **not** traced how much (if anything) downstream — reward shaping, observation
side-channels — actually depends on `sec_kicks_this_step`'s exact value; only that the
value itself is wrong when it happens.

**Confirmed safe:** `dagger.py`'s `n_kicks` counting
([dagger.py:509-524](../src/footballcoach/ai/ppo/dagger.py#L509)) explicitly reads only
"the trainee's OWN policy" kicks via `kicked_this_tick` on the trainee — same safe path
as §"confirmed safe" above.

**Confirmed exposed, deliberately left unfixed on the UI side:** the UI's kick/tackle
swing animation (`App._kick_cb`, [app.py](../src/footballcoach/ui/app.py)) reads
`player.last_kick_direction` to aim the animation, so it's `None` for the same
majority of real gameplay kicks described above (any AI pass or shot — only a raw
human/UI `KickOrder` sets it). A first pass at this read `match.ball.velocity` instead,
which does work uniformly (it's the actual physics result of every kick path) — but the
user asked for that reverted: keep reading the proper field, and fall back to
"straight ahead" (`cos(heading_rad), sin(heading_rad)`) when it's missing, rather than
reading something else to paper over the engine gap. See `ui/knowledge.md`'s "Action
icons" section for the current (reverted) behaviour. **This means the kick swing
animation shows a straight-ahead guess, not the real kick direction, for most real
gameplay kicks until this bug is actually fixed** — a known, accepted limitation, not
an oversight.

**Not traced / open:** whether anything in
[`agent_plans/masked_action_training_plan.md`](masked_action_training_plan.md)'s
trigger-masking machinery reads `kicked_this_tick` for a non-trainee player anywhere
(the trainee's own gate/mask logic should be safe per the "confirmed safe" point above,
since it's built from the trainee's own `kicked_this_tick`, which is set correctly).
Worth a targeted grep/read before considering this bug fully scoped.

---

## 3. Possible fixes

### Fix A (recommended) — route `PassOrder`/`ShootOrder` through `_finish_kick`

Make both order types call `_finish_kick` (or a thin wrapper around it) the same way
`kick_direct`/`kick_with_direction` do, instead of firing `on_kick` inline themselves.
Concretely, something like adding a small helper on `Player` — e.g.
`_finish_kick_from_order(match, log_label, power_fraction, spin)` — that `pass_ball`/
`kick_ball`'s callers (`PassOrder.execute`, `ShootOrder.execute`) call right after the
physics call, instead of their own `if player.on_kick is not None: player.on_kick(player)`
lines. `_finish_kick` takes `is_first_touch`; `kick_direct` computes it as
`self.state == PlayerState.CONTROLLING_BALL` right before the kick
([player.py:423](../src/footballcoach/entities/player.py#L423)) — a dynamic, per-kick
check, not something specific to how the kick was triggered. A `PassOrder`/`ShootOrder`
kick taken during a first-touch window is a real, ordinary scenario (nothing stops a
player passing or shooting the instant the ball arrives), so the same
`player.state == PlayerState.CONTROLLING_BALL` check belongs in the new call sites too
— don't hardcode `False` there.

**Pros:** one code path for all kick bookkeeping, matches the existing docstring's own
claim ("shared bookkeeping tail for both direct-physics kick methods") — which is
presently wrong (three call sites exist, not two, once you count `PassOrder`/`ShootOrder`
bypassing it). Fixes every consumer in §2 at once.

**Cons/risks:** need to check `PassOrder`'s spin argument (`pass_ball` doesn't appear to
take one — probably always `Vector3.zero()` for a pass) and both orders' exact power
value (pre- vs post-run-compensation — `_finish_kick`'s `adjusted_power` parameter is
documented as "what was actually passed to the physics function") lines up with what
`PassOrder`/`ShootOrder` compute locally as `compensated`. `kick_count` incrementing for
every pass (not just shots/direct kicks) is a behavior change worth confirming is
wanted — passes are far more frequent than shots, so `kick_count` semantics may shift
meaningfully if it's used anywhere as a proxy for "shots taken" rather than "all kicks".

### Fix B — narrower patch: just replicate the missing fields inline

Instead of a shared helper, add the missing four lines
(`kicked_this_tick`/`kick_count`/`last_kick_direction`/`last_kick_power_fraction`/
`last_kick_spin`) directly at both `PassOrder.execute`'s and `ShootOrder.execute`'s
existing `if player.on_kick is not None:` call sites, computed from the post-kick
`match.ball.velocity` (same formula `_finish_kick` already uses) and each order's own
already-computed power value.

**Pros:** smaller diff, no `_finish_kick` signature questions.
**Cons:** duplicates the bookkeeping logic in three places instead of one (the project's
own stated preference, per `CLAUDE.md`'s "single-source validity" note elsewhere in this
codebase, leans towards Fix A) — a future change to what "finishing a kick" means would
need to be applied in three places, and could silently drift out of sync again exactly
the way this bug shows it already has.

### Fix C — per-consumer workaround instead of an engine fix (tried, rejected)

Don't fix the engine; have every *consumer* read something else instead of the
`_finish_kick`-only fields. Tried for the UI specifically (`match.ball.velocity`, which
does work there — the ball's velocity is exactly what the animation needs anyway) and
then **reverted at the user's request**: "please stay reading the `last_kick_direction`,
just assume it's straight if you don't have it but you know we kicked, you should use
the proper fields and we should just fix the bug at some point." The UI now reads
`player.last_kick_direction` and falls back to a straight-ahead guess when it's missing,
rather than reading a different field to hide the gap.

This also wouldn't have helped `bc.py`/`scenario_env.py` even if kept: they specifically
want to know "did a kick *just* happen on *this specific* player", and `ball.velocity`
alone can't distinguish that from "the ball is already moving for some other reason"
the way `kicked_this_tick` (a discrete per-tick flag) can. Not a real option for the
AI-side consumers in §2 regardless of the UI's own decision.

---

## 4. Suggested next step

Fix A, scoped to just `PassOrder`/`ShootOrder`, with the two open questions above
(spin argument, power-value alignment, `kick_count` semantics) checked against the
existing `pass_ball`/`kick_ball` call sites before writing the change — then:

- Re-run the exact repro in §1 (real `Match`, real `PassOrder`) and confirm
  `kicked_this_tick`/`last_kick_direction` now populate.
- Add a regression test alongside the existing `tests/unit/test_kick_label_roundtrip.py`
  / `tests/unit/test_kicking.py` (both already touch this area, per the earlier grep for
  `last_kick_direction`/`kicked_this_tick` usage across the test suite) specifically for
  a `PassOrder`- and `ShootOrder`-triggered kick, not just `KickOrder`.
- Check whether `bc.py`'s counterfactual label code
  (§2 "confirmed exposed, and worse than first thought") needs anything beyond the
  engine fix itself — it already resets `kicked_this_tick`/`kick_armed`/`tackle_armed`
  and suppresses callbacks before executing the counterfactual order, so fixing
  `ShootOrder.execute()` to set `kicked_this_tick` should make that block correctly
  observe a counterfactual shot with no further change needed there — but confirm this
  rather than assuming it, the same way the rest of this investigation did.
- Decide whether existing BC-recorded datasets (`demonstrations/phase1/` etc., if any
  were recorded while this bug was live) need re-recording, the same way a
  `PlayerFeatures` schema change forces a re-record (see `agent_plans/heading_fix.md`
  §3 for the precedent/reasoning) — recorded shot labels from before the fix would be
  the "no kick" mislabelling described in §2, not just missing a new field.
