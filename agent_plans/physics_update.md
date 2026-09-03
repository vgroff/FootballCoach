# Ball Physics Update

> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

## 0. Status

Everything this document originally tracked is **implemented, verified, and
committed** (commit `bf1b271`, 2026-09-03) — the dt/bounce-threshold coupling
fix, the spin-coupled bounce model, the `ball_inertia_shell_factor` retune,
the ground-contact spin-up cost (`_resolve_ground_friction`), the
passing/push-kick speed compensation that followed from it, and the
`ball_max_speed_mps` config-vs-hardcoded-literal fix. The permanent technical
record for all of that now lives in `engine/knowledge.md` (kept in sync
alongside the code, per the note above) — this plan document's job was to
track the *work*, not to be a second permanent copy of it, so the "done"
narrative has been removed now that it's landed. See `git show bf1b271` /
`git log` for the exact diffs and commit history if you need the original
reasoning, numbers, or verification trail.

Downstream ball/player physics-encoder checkpoints and anything trained on
top of them (value nets, PPO policies) are stale relative to these dynamics
and need a retraining pass whenever that's next convenient — this was a
deliberate, accepted tradeoff at the time (see `git log`), not an oversight.

The only thing still open is §9 below — a proposal, not a task.

---

## 9. Proposal (NOT implemented) — a slip-based "is it actually rolling" condition

Raised by the user after the above landed: several places in the codebase
decide "is this ball at rest / rolling / settled" using a **height and/or
raw-speed threshold** — e.g. `ball_physics.py`'s own bounce-vs-rest gate
(grounded + `v_z` vs `bounce_threshold_mps`), and physics-pretrain's
`resting_head` label (`ball_dataset.py`'s `compute_resting_targets`, gated on
`||vel_x, vel_y, vel_z|| < rest_speed_norm`, no height or spin term at all).
Question: now that the engine has a real, physically-grounded
"rolling-without-slipping" test — contact-point **slip**,
`(vx - r·spin.y, vy + r·spin.x)`, already computed every tick inside
`_apply_ground_friction_impulse` — should that replace the older
height/speed heuristics anywhere they're used as a proxy for "rolling"?

**Not implemented.** The user's instinct was that this sounds marginal
given the retraining cost, and asked to record pros/cons rather than build
it. This document agrees with that read (see recommendation below).

### 9.1 What the slip condition would actually buy

A height and/or raw-speed threshold can only tell you the ball is slow
and/or touching the ground — it says nothing about **spin**, so it cannot
distinguish a ball genuinely rolling (spin matches velocity) from one
sliding/skidding with mismatched or zero spin. `|slip| ≈ 0` is the actual
kinematic definition of rolling-without-slipping and is spin-aware by
construction.

### 9.2 Pros

- **Physically correct**, not a proxy — directly tests the thing "is it
  rolling" actually means, rather than inferring it from two loosely
  correlated quantities (height, speed).
- **Consistency** — `step_ball` itself already uses exactly this test
  internally (the `remaining_slip` gate) to decide between kinetic
  ground-friction and gentle rolling-friction. Using a *different*
  definition of "rolling" elsewhere in the pipeline (e.g. the pretrain
  label) is the same "two disconnected definitions of the same concept"
  shape as this session's `ball_max_speed_mps` bug — not broken today, but
  the kind of drift that bites later.
- **Cheap to implement in isolation** — it's a pure function over inputs
  the pipeline already carries (velocity, spin, radius); no new physics,
  just exposing/reusing the existing computation. The cost is entirely in
  retraining, not in writing the code.

### 9.3 Cons

- **The place it would most plausibly apply — `resting_head` — barely
  needs it.** `compute_resting_targets` only labels the position once the
  ball has *already* dropped below `rest_speed_norm`; a ball that's
  "sliding, not truly rolling" but already that slow is, for the purpose
  of "where does it end up," practically indistinguishable from "rolling
  and slow" — both are ~stopped. The rolling-vs-sliding distinction
  matters most at moderate/high speed, which is exactly the regime this
  target already excludes via `min_start_speed_norm`/`rest_speed_norm`
  gating. So the sharper condition would mostly relabel states that were
  already going to resolve to nearly the same target position.
- **The information isn't new to the network.** Velocity and spin are
  already raw encoder inputs; `slip` is a simple (near-linear) function of
  existing inputs. A hand-engineered feature/target built from it doesn't
  hand the encoder information it lacks — at best it's training-signal
  scaffolding, not new signal.
- **No observed problem motivating it.** Every implemented fix in this
  session's history was triggered by a concrete measured discrepancy (a
  failing test, a wrong percentage, a destroyed checkpoint). This one is
  "would be more principled," not "is currently wrong" — nothing has been
  observed to misbehave because of the height/speed heuristics.
- **`step_ball`'s own bounce-vs-rest gate doesn't need this fix either** —
  its job (real bounce vs. settled contact) is a different question from
  "is it rolling true," and height+v_z is already the right tool for that
  specific job. The slip test is already correctly used exactly where it
  matters (the friction/spin-up mechanics themselves) — there's no gap
  there to close.
- **Retraining cost, same calculus as the rest of this project's physics
  changes** — changing `compute_resting_targets`'s definition would need a
  fresh physics-pretrain run for the ball dynamics encoder, with no
  accompanying downstream breakage forcing the timing the way the passing
  balance breakage forced the earlier fixes.

### 9.4 Recommendation

Agrees with the user's instinct: **marginal, defer.** No concrete evidence
of a problem this would fix; the one place it could plausibly help
(`resting_head`) targets a state where the rolling/sliding distinction
barely changes the outcome; the underlying information is already available
to the network via existing inputs. If it's ever picked up, the natural
trigger would be a *new* auxiliary task that specifically needs a clean
rolling/sliding classification (not a retrofit of `resting_head`), bundled
into whatever retraining pass is already planned at that time rather than
justifying one on its own.
