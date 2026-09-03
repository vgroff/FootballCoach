# Ball Physics Update — dt/bounce-threshold coupling, restitution realism, spin-coupled bounce

> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

## 0. Status

**§2 (dt/bounce-threshold coupling) and §4 (spin-coupled bounce) are
implemented** (2026-09-03, `engine/ball_physics.py` + `config/physics.json`
+ `engine/knowledge.md`) — see §2.3/§4 below for what actually landed
(the real fix was implemented for §2, not just the guardrail originally
proposed in §2.3). **§3's restitution retune is NOT separately needed** —
the spin-coupled bounce model in §4 structurally replaced
`bounce_restitution_horizontal` rather than just retuning it, and the
emergent no-spin-bounce retention it produces (see §4.5) already lands
in the ~0.6 range §3 recommended, without hand-tuning a constant to get
there. §5's doc-drift fixes are done as part of the same pass. **This
was landed WITHOUT waiting for a dedicated retrain** — the user explicitly
opted to proceed now rather than defer; downstream ball/player
physics-encoder checkpoints and anything trained on top of them (value
nets, PPO policies) are now stale relative to these dynamics and need a
retraining pass whenever that's next convenient. Full test suite verified
clean (see §7) but no retraining has happened yet as of this writing.

This document assumes the reader has NOT seen the session that produced it —
all necessary context, numbers, and verification are included inline.
"Confirmed" below means directly verified by running real code against
`BallPhysicsParams.from_config()` during this session, not inferred from
reading source alone.

---

## 1. Why this matters

An audit of `ball_physics.py` / `kicking.py` / `possession.py` and their
`physics.json` parameters (`ball_physics`, `kicking`, `passing`,
`ball_pickup`, `control_time` sections) surfaced one concrete correctness bug
and one realism concern, plus a natural follow-on modeling improvement:

1. **`bounce_threshold_mps` is implicitly coupled to the physics timestep
   `dt_s`**, in a way nothing enforces or documents, and the coupling is
   already uncomfortably close to broken at training's current tick rate.
2. **`bounce_restitution_horizontal` (0.8) looks too high relative to real
   ball-bounce physics** — plausibly backwards relative to
   `bounce_restitution_vertical` (0.75), which real surfaces (grass/turf)
   should generally have *lower* than a rigid-surface test bounce, not
   higher than the horizontal figure.
3. **Topspin/backspin currently has no effect on bounces at all** —
   `bounce_restitution_horizontal`/`bounce_spin_retention` are independent
   flat constants, when real physics couples horizontal speed loss and spin
   change through a single friction interaction at the contact point. The
   representation needed to model this (`ball.spin`'s horizontal components)
   already exists for exactly this purpose in the in-flight Magnus force and
   in `kick_trajectory.py`'s topspin/backspin/sidespin convention — it's
   just not consulted during the bounce itself.

Also found, and worth a separate, trivial no-retrain fix: `engine/knowledge.md`'s
prose has drifted from the live `physics.json` values in at least five places
(see §5) — not a behavior bug, but a real risk for anyone (human or agent)
reasoning about ball physics by reading the docs instead of the config.

---

## 2. Finding 1 — `bounce_threshold_mps` is coupled to `dt_s` (bug)

### 2.1 Mechanism

`step_ball()` (`engine/ball_physics.py`) distinguishes a genuine bounce from
ordinary resting/rolling ground contact by checking the ball's
gravity-integrated vertical velocity for this tick against
`bounce_threshold_mps` (0.5). That integrated value is `g · dt_s` (plus a
usually-negligible drag contribution) — so the check is comparing a
**dt-dependent quantity** against a **fixed constant**. There's a second,
protective check on top: even if the raw incoming velocity crosses the
threshold, the code only treats it as a *real* bounce if the resulting
*outgoing* vertical speed (`g · dt_s · bounce_restitution_vertical`) also
exceeds the threshold — otherwise it's zeroed out as ordinary ground contact.
Accounting for both checks, the exact safety boundary is:

```
dt* = bounce_threshold_mps / (gravity_mps2 * bounce_restitution_vertical)
    = 0.5 / (9.81 * 0.75)
    ≈ 0.068s  (~14.7 Hz)
```

Below `dt*`, everything behaves correctly. At or above it, a ball resting or
rolling flat on the ground with pure horizontal velocity **spuriously
registers a full bounce every tick** — the exact bug class
`bounce_threshold_mps` was originally added to prevent (see the existing
`bounce_threshold_mps` comment in `physics.json` / `engine/knowledge.md`'s
"ground contact" note), just reintroduced by a coarse enough `dt_s` instead
of by the threshold's absence.

### 2.2 Confirmed numbers

Two physics ticks are actually in use in this codebase:

- **UI**: `world.dt_s = 1/30 ≈ 0.0333s` (30Hz) — comfortable margin, outgoing
  v_z predicted at 0.245, well under 0.5.
- **Training**: `ai_config.json`'s `observation.sim_dt_s = 0.06` (~16.7Hz),
  injected into `Match.dt_s` by `ScenarioEnv` (`scenario_env.py:145,235`),
  overriding the 30Hz default specifically "to speed up training" per that
  key's own comment.

Directly verified by running `step_ball()` at each dt:

| dt | Hz | predicted outgoing v_z | bounce misfires? |
|---|---|---|---|
| 1/30 (UI) | 30 | 0.245 | no |
| 0.05 | 20 | 0.368 | no |
| **0.06 (training, current)** | **16.7** | **0.442** | **no — 88% of the way to the threshold** |
| 0.067 (`ai_config.json`'s own suggested "15Hz, ~2x faster" option, see its `_comment_sim_dt_s`) | 15 | 0.493 | no — **98.6% of the way there** |
| 0.07 | 14.3 | 0.515 | **yes, fires** |
| 0.10 | 10 | 0.736 | yes, fires — confirmed via full-episode stopping-distance test: expected ~20m rolling distance collapses to 3.7m as the spurious bounce repeatedly kills horizontal speed |

**Training is safe today, but on a thin (~12%) margin — and the very next
step the codebase's own documentation suggests taking (`sim_dt_s=0.067` "for
faster training") would put it at 98.6% of the boundary, one nudge from
silently reintroducing the bug.** No existing test would catch this: every
physics test in `tests/unit/test_ball_physics.py` hardcodes `dt=1/30`.

### 2.3 Implemented fix (2026-09-03)

**Implemented the real, dt-independent-by-construction fix**, not just the
guardrail originally proposed here — the user opted to do the full work
(this + §4) in one pass rather than defer. `step_ball` now checks
`ball.is_grounded()` (the ball's position from BEFORE this tick's
integration) and gates the bounce-vs-rest classification on it
(`was_grounded_before_tick` in `engine/ball_physics.py`): an already-grounded
ball can never misfire as a bounce no matter how coarse `dt_s` is, since the
classification no longer depends on a single tick's gravity-integration
velocity artefact at all. `BOUNCE_THRESHOLD_MPS` is still consulted, but
only for a ball that was genuinely airborne at the start of the tick, where
it reflects real multi-tick-accumulated fall velocity rather than a
one-tick artefact — dt-independent in exactly the case it needs to be.

Re-verified after the fix: the resting-ball misfire and the stopping-distance
collapse (previously 20m → 3.7m at `dt_s=0.1`) are both gone at every tested
`dt_s` up to 0.2s (5Hz) — see §7.

`ai_config.json`'s `_comment_sim_dt_s` was NOT updated with a dt* ceiling
note, since that ceiling no longer exists — `sim_dt_s` is now safe to
increase for training speed without revisiting this bug at all.

### 2.4 (superseded)

The two options originally sketched here (derive the threshold from `dt_s`,
or a dt-independent contact-state test) — the second was the one
implemented, per §2.3 above.

---

## 3. Finding 2 — coefficient realism audit

Checked against sports-aerodynamics literature (Rod Cross's football-bounce
work, University of Sydney Physics; general soccer-ball drag/turf-bounce
research) — see Sources below. All current values are `from_config()`-driven
(`physics.json`'s `ball_physics` section), so this is a tuning question, not
a code-correctness one.

| Param | Current | Realistic range (literature) | Verdict |
|---|---|---|---|
| `drag_coefficient` | 0.2 | ~0.15–0.20 above the "drag crisis" (hard-struck ball, roughly >9–13 m/s); ~0.4–0.5 for slow rolling/passing speeds — real footballs have speed-dependent drag, not constant | Sits at the top of the low-drag band. Since the sim uses one fixed Cd for all speeds, this is realistic for shots/hard passes and deliberately less draggy than reality for slow rolls — i.e. already leaning toward lower energy loss, as intended. No change proposed. |
| `magnus_coefficient` | 0.25 | ~0.1–0.3 depending on spin ratio, ~0.2–0.25 typical for a real shot | Realistic middle. No change proposed. |
| `bounce_restitution_vertical` | 0.75 | FIFA's rigid-plate ball-quality test: ~0.79–0.88 (2m drop → 125–155cm rebound). A real grass pitch absorbs more — artificial turf ≈0.65 (Novel Methodology for Football Rebound Test Method, PMC7146741); natural grass is generally cited lower still | Already at/above the realistic ceiling for grass — essentially as low-loss as a rigid lab-test ball. No further increase proposed. |
| **`bounce_restitution_horizontal`** (removed) | was 0.8 | No round-ball-specific figure found, but the underlying mechanism (Cross, "Bounce of an oval shaped football") is clear: horizontal/tangential COR is **not a fixed constant** — it's governed by friction and spin coupling at contact, and swings hugely with incident spin. Quoting: *"If a ball is incident with topspin, there is only a small change in horizontal speed, whereas if the ball is incident without spin or with backspin there is a large reduction in horizontal speed."* | **Was too high, and backwards relative to the vertical figure** (0.8 > 0.75, when tangential loss should generally be the larger one for non-topspin bounces). **Superseded, not retuned** — §4's spin-coupled friction model replaced this constant entirely rather than adjusting its value; see §4.5 for the emergent no-spin-bounce retention it now produces (~0.6, landing exactly in the range this row would have recommended, without hand-tuning). |
| `rolling_friction_coefficient` | 0.05 | ~0.03–0.10 for grass depending on length/moisture; dry/short/firm grass sits near the low end | Already at the low-friction end of the realistic band. No change proposed. |
| `spin_decay_per_s` | 0.12 | No strong published benchmark found; over a typical 1–2s shot flight this is a mild ~12–24% spin loss, consistent with spin persisting through most of a real shot | Reasonable, not aggressive. No change proposed. |
| `goal_net_restitution` | 0.1 | Real goal nets are strongly energy-absorbing — the ball essentially stops | This is the one place "lean toward lower energy loss" should NOT apply — nets are genuinely high-loss in reality. Explicitly recommend leaving this alone (or lowering further), not raising it. |
| `block_restitution` (player-body deflection) | 0.35 | No literature anchor found; documented in `physics.json` as "a fairly dead deflection" by design | Plausible. No change proposed. |

**Net takeaway: the defaults are already realistic and mostly already lean
toward the low-energy-loss end of the plausible range** — the one clear
outlier is `bounce_restitution_horizontal`, both on its own (too high vs.
literature) and relative to `bounce_restitution_vertical` (wrong ordering).

---

## 4. Implemented — spin-coupled tangential bounce (topspin/backspin)

### 4.1 Current model

`bounce_restitution_horizontal` and `bounce_spin_retention` are two
*independent* constants applied at every real bounce
(`ball_physics.py`'s bounce branch): horizontal velocity is scaled by 0.8,
spin is scaled by 0.5, with no interaction between the two. Topspin,
backspin, and no-spin bounces all behave identically.

### 4.2 What real physics does instead

Per Cross's football-bounce work (and general oblique-sports-ball-bounce
literature), the contact point's slip velocity — the horizontal ball
velocity minus the rotational contribution at the contact point
(`v_horizontal − r·ω_perp`) — determines a friction force during contact
that simultaneously reduces horizontal (linear) velocity and changes spin,
via either a constant-friction sliding model or a tangential-COR gripping
model depending on whether the ball's rotation "catches up" to zero slip
before contact ends.

### 4.3 Implementation (2026-09-03)

The representation this needed already existed: `ball.spin`'s horizontal
(x/y) components already encode topspin/backspin/sidespin about the axis
perpendicular to travel, in the ground plane — confirmed via
`ui/kick_trajectory.py`'s existing convention (`spin.y > 0 → topspin`,
`spin.y < 0 → backspin`, already used for the in-flight Magnus force). So
this landed as a **bounce-branch-only** change — no changes to how spin is
represented, imparted at kick time, or decayed in flight.

New helper `_resolve_bounce_friction()` in `engine/ball_physics.py`,
called from `step_ball`'s real-bounce branch:

1. **Contact-point slip velocity**: `slip = (vx - r·spin.y, vy + r·spin.x)`
   — the ball's horizontal velocity minus the rotational contribution at
   the contact point directly below the centre.
2. **Normal impulse** delivered by the ground this bounce, derived from the
   already-computed vertical restitution: `J_n = m·|v_z,in|·(1+e_v)`.
3. **Friction impulse**, capped at `bounce_friction_coefficient · J_n`
   (Coulomb friction) or, if smaller, the exact impulse needed to drive
   slip to zero (`slip_speed / ((1/m)·(1+1/k))`, from the ball's reduced
   mass for tangential contact — `k` = `ball_inertia_shell_factor`, the
   ball's moment-of-inertia shape factor). Confirmed (§4.5) that the
   "grips" case — friction alone is enough to fully cancel slip mid-bounce
   — is common at realistic bounce speeds, matching Cross's force-plate
   finding that this is the typical real-world outcome, not the exception.
4. That one impulse is applied to **both** horizontal velocity and
   horizontal-axis spin simultaneously (via the ball's moment of inertia,
   `I = k·m·r²`), replacing the old independent
   `bounce_restitution_horizontal` (velocity) / `bounce_spin_retention`
   (spin) constants. `spin.z` (vertical-axis spin, doesn't couple to this
   planar model) keeps the old flat `bounce_spin_retention` decay.

New config fields (`physics.json`'s `ball_physics` section):
`bounce_friction_coefficient` (0.4 — no round-ball-specific literature
figure found, a plausible mid-range value, see §3's table), and
`ball_inertia_shell_factor` (k in `I = k*m*r^2`; **0.5 as of 2026-09-03**,
see §4.6 — the idealized thin-pressurized-shell value is 2/3, vs. 0.4 = 2/5
for a solid sphere). `bounce_restitution_horizontal` was removed from the
config entirely (was only ever read in this one place).

### 4.4 Tradeoff (accepted)

Real, moderately-sized physics change — replaced two independent flat
tunables with a coupled, spin-dependent formula, changing bounce dynamics
for any grounded shot/pass/cross with meaningful spin. The user explicitly
opted to accept the retraining cost (see §6) rather than defer.

### 4.5 Verification

Full test suite: `tests/unit` (211 passed), `tests/balance` + `tests/scenario`
(237 passed, 2 failed — both pre-existing/unrelated: the long-standing
`test_mark_order.py` target-missing failure, and
`test_control_time_balance.py::test_goalkeeper_in_box_control_time_faster_for_high_balls`,
which is explicitly commented in the test itself as "KNOWN FAILING
(2026-08-06)... IGNORE", and doesn't touch ball physics at all). Zero new
regressions from either §2 or §4's changes.

Directly verified the spin-coupling behaves as the literature predicts —
same incoming shot (10 m/s horizontal, 6 m/s downward), varying only
incident spin, current config (k=0.5, mu=0.4):

| Incident spin | Outgoing horizontal speed | Retention |
|---|---|---|
| Backspin (25 rad/s) | 5.80 m/s | 58.0% |
| No spin | 6.67 m/s | 66.7% |
| Topspin (25 rad/s) | 7.58 m/s | 75.8% |

This is an emergent result of the physics, not a hand-tuned lookup table.
All three cases reach the "grip" (zero final slip) branch here — consistent
with Cross's finding that gripping, not pure sliding, is the common
real-world outcome.

**Correction (2026-09-03, later the same session):** the FIRST version of
this table (backspin 48.3% / no-spin 58.6% / topspin 68.9%, at the
then-default k=0.667) was computed with a flawed test harness — the ball
was dropped from 1m and allowed to free-fall before bouncing, so the actual
impact vz wasn't exactly the intended 6 m/s (gravity added extra speed
during the fall), contaminating all three numbers by the same unknown
amount. The ordering (backspin < no-spin < topspin) was still correct, but
the specific percentages were off. Re-verified via a clean method — calling
`_resolve_bounce_friction` directly with the exact intended incoming
velocity, bypassing the free-fall setup entirely — for both this table and
§4.6 below. If you need to characterize this model's behavior again later,
use the direct-call method, not a dropped-ball harness.

---

### 4.6 Post-implementation tuning: `ball_inertia_shell_factor` 0.667 → 0.5

After landing, the user's gut check on the emergent numbers ("0.6 seems
pretty steep for a no-spin grip bounce, is that accurate?") led to a
follow-up round clarifying what each of the two bounce coefficients
actually controls:

- **`bounce_friction_coefficient` (mu)** only decides *whether* a bounce
  reaches "grip" (friction budget = `mu * normal_impulse`, capped) — it has
  **no effect on the outcome** once grip is reached. Lowering it also
  *shrinks* the grip regime (pushes more bounces into "slide," where the
  outcome is nearly spin-independent — same retention regardless of
  topspin/backspin unless spin is strong enough to flip which way the
  contact point is slipping), which would have *undermined* the
  spin-differentiation this whole model exists for. Ruled out for this
  purpose.
- **`ball_inertia_shell_factor` (k)** directly sets the grip outcome — a
  gripped, no-spin bounce retains exactly `1/(1+k)` of its horizontal speed
  (the same relationship behind the textbook "5/7" result for a solid
  sphere sliding-to-rolling under friction; our thin-shell analog is
  `1/(1+2/3) = 3/5 = 0.6`). Lowering k *also widens* the grip regime (more
  bounces reach grip, not fewer) — the opposite, better-aligned side effect
  compared to `mu`.

Real-world caveat also surfaced here: the "grip → exactly zero final slip"
model is a **rigid-body, friction-only** idealization. Cross's football
paper (already cited, §7) found real balls store and release tangential
elastic energy during contact ("spring back") — the same mechanism behind
his measured "ball bounces forward faster than it arrived" surprise — which
a pure-friction rigid model can't capture. That means the true physical
retention in the grip case is plausibly *somewhat higher* than what a rigid
model predicts, independent of whether k=2/3 is the "correct" idealized
shell value. The more rigorous fix for that would be a genuine tangential
coefficient of restitution (`final_slip = -e_t * initial_slip` instead of
assuming `final_slip = 0` at grip, with `e_t=0` reproducing today's
behavior) — flagged here as a follow-up, **not implemented**.

**Decision taken instead (lower-effort, still physically defensible):**
retune `ball_inertia_shell_factor` from 0.667 (idealized thin shell) down to
**0.5**, landing partway toward a solid sphere (0.4) without asserting the
ball is one. Retention for a gripped, no-spin bounce moves from 0.6 to
0.667. Verified: `tests/unit/test_ball_physics.py` still 13/13 passing (no
test pins a specific k-derived retention value, so this was a safe,
contained numeric retune — see §7 for the fuller post-tuning regression
run). `ball_physics.py`'s `from_config()` fallback default was updated to
match (0.5), so a config missing this key falls back to the same tuned
value rather than silently reverting to the idealized 2/3.

## 5. Documentation drift (no-retrain, do anytime)

`engine/knowledge.md`'s prose has fallen out of sync with the live
`physics.json` values in at least five places — worth fixing independently
of everything else above, since it's pure documentation and carries no
behavior risk:

- **Drag**: doc says "C_d = 0.25"; config has 0.2.
- **Vertical bounce restitution**: doc says "e_v = 0.6 ... ball dropped from
  2m rebounding 1.0–1.5m"; config has 0.75 — a materially livelier bounce
  than the doc's own stated rebound-height justification describes.
- **Rolling friction**: doc says "mu_roll = 0.06, ~20m travel at 5 m/s";
  config has 0.05.
- **Kick angle error formula**: doc still states
  `sigma = 0.0107 + 0.0893·(1-precision)` (no exponent); the live formula
  (per `physics.json`'s own `kicking` section comment) is
  `0.0055 + 0.04·(1-precision^0.87)` — a real formula change, not just
  constant retuning, that the prose was never updated for.
- **Kick power formula**: doc says `v_max = 15 + 20·attr`; config has
  `12.0 + 12.3·attr` (deliberately rescaled per `physics.json`'s own
  comment, to compensate for `running_power_coefficient` changing from
  0.3→0.6 — the doc prose was never updated for this either).

---

## 6. Sequencing / retraining notes

**Landed (2026-09-03): §2 (dt/bounce-threshold fix) and §4 (spin-coupled
bounce), together, in one pass** — the user explicitly chose to do both now
rather than wait. §5's doc fixes landed in the same pass (no-retrain, was
always safe to do anytime).

**This means downstream checkpoints are now stale**: the ball/player
physics-encoder checkpoints, and anything trained on top of them (value
nets, PPO policies), were conditioned on the old flat-restitution bounce
model and the pre-fix resting-ball dynamics at coarse `dt_s`. A retraining
pass is needed before those checkpoints' behavior can be trusted to reflect
current physics — not urgent (nothing about the physics change breaks
existing checkpoints, they just no longer exactly match this dynamics
model), but should happen before drawing conclusions from value-net/PPO
diagnostics that assume today's ball physics.

---

## 7. Verification performed this session

All of §2's numbers were produced by actually running `step_ball()` against
`BallPhysicsParams.from_config()` (not reasoned about from reading the code
alone): single-tick resting-ball checks at dt ∈
{1/30, 0.05, 0.06, 0.067, 0.07, 0.1}, a full stopping-distance regression at
each, and an explicit derivation + confirmation of the `dt* ≈ 0.068s`
boundary formula. §3/§4's literature claims came from a handful of
sports-aerodynamics sources (listed below); no soccer-specific tangential-COR
number was found, which is called out explicitly in §3 and §4.3 rather than
papered over.

### Sources consulted
- Rod Cross, "Bounce of an oval shaped football," Physics Dept., University
  of Sydney — https://www.physics.sydney.edu.au/~cross/PUBLICATIONS/49.%20Football.pdf
- "Novel Methodology for Football Rebound Test Method" (PMC7146741) —
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7146741/
- "Modelling bounce of sports balls with friction and tangential
  compliance" — https://www.researchgate.net/publication/259850126
- "Finite element modelling and experimental study of oblique soccer ball
  bounce" — https://www.researchgate.net/publication/51509670

---

## 8. Implemented — ground-contact spin-up cost, and the passing/push-kick fix it required

Follow-on from §4, landed the same day (2026-09-03), prompted by a user
question: "how much would spin up cost slow down the rolling ball?" — which
surfaced that ordinary ground contact never touched spin at all. A ball
"rolling" in this sim was really sliding along the ground with a flat
friction constant; real rolling requires `v = r*omega` at the contact
point, and a ball that doesn't already have that (any zero-spin kick,
which was every kick until this section) has to earn it via the same
kinetic friction that governs bounces. That's a real, missing physical
cost — and, as it turned out, a substantial one.

### 8.1 What was added

`_resolve_ground_friction()` (`engine/ball_physics.py`), sharing the exact
same slip/grip Coulomb-friction physics as the bounce model (§4) via a new
extracted helper, `_apply_ground_friction_impulse()` — `_resolve_bounce_friction`
was refactored to call it too, rather than duplicating the slip/grip math a
second time. The only difference between the two call sites is what normal
impulse is available: a bounce gets one instantaneous hit
(`m·|v_z,in|·(1+e_v)`), ground contact gets a continuous one spread over
each tick (`m·g·dt_s`).

`bounce_friction_coefficient` was renamed to **`ground_friction_coefficient`**
(config key and dataclass field, `physics.json` + `ball_physics.py`) since
it's now used for both — physically the same ball-on-grass friction
coefficient in both cases, just delivered differently.

A real, load-bearing bug was found and fixed while implementing this: after
a ball reaches true rolling (zero slip), the existing flat
`rolling_friction_coefficient` decay only ever reduced *velocity*, never
*spin* — so the very next tick would reopen slip (since `v` dropped but
`r·omega` didn't), re-triggering the much stronger kinetic ground friction,
every tick, forever. Fixed by scaling spin down in lockstep with velocity
(same factor) whenever the ball is confirmed already at zero slip
(`step_ball`'s `remaining_slip` check) — gentle rolling friction now
actually stays gentle once reached, instead of silently re-invoking the
sliding-friction path on every subsequent tick.

### 8.2 The result: same math as the bounce case

Confirmed both analytically and via direct numerical integration of the
classic sliding-to-rolling transition: a ball fully gripping retains
exactly `1/(1+k)` of its pre-slip speed (the same relationship behind the
textbook "5/7 v0" result for a solid sphere) — **independent of the
friction coefficient**, which only controls how long the transition takes,
not how much speed is lost. At the current `k=0.5`, that's a **33.3% loss**,
taking about 0.85s to complete for a 10 m/s launch (using
`ground_friction_coefficient=0.4` as the transition-rate driver). Verified
directly in `step_ball` (not just the isolated integration): a zero-spin
10 m/s ball's slip closes to exactly 0 by t≈0.73s, retention ≈0.61–0.67
depending on how much concurrent drag is folded in (the isolated
friction-only prediction doesn't include drag, which is real and
additional).

### 8.3 Immediate real-world consequence: passing broke

This is not a cosmetic change — every real ground pass in the game
launches with zero spin (nothing in `pass_ball()` ever set spin), so every
pass now paid this same ~33% transition cost before `pass_speed_mps`'s
auto-pace calibration (`v = overshoot_factor·sqrt(2·mu_roll·g·distance)`,
tuned assuming a ball is *already* rolling from the moment it's launched)
ever got a chance to apply. Result: every pass fell meaningfully short of
its intended target, and running the full test suite confirmed it —
**18 failed** (up from the usual 2 pre-existing/unrelated ones), all 16 new
failures in passing: `test_pass_balance.py` (10 accuracy-target tests
across 10m/25m/40m/60m/65m and good/average/low-attribute players),
`test_actions.py::test_pass_to_reaches_teammate`,
`test_move_order.py::test_push_kick_box_to_box_realistic_kick_count`,
`test_pass_getpossession_diagnostic.py` (2 tests),
`test_running_direction_balance.py::test_pass_accuracy_forward_beats_backward_run`.

### 8.4 First attempt (implemented, then reverted): launch already rolling

First fix tried: give passes and push-kicks matching rolling spin at
launch (`kicking.rolling_spin_for_direction(direction, speed_mps,
ball_radius_m)`, returning the spin vector satisfying `v = r*omega` —
exactly `ui/kick_trajectory.py`'s pure-topspin axis, scaled to match),
wired into `kicking.pass_ball()` (computed directly from the pass's own
`aim_dir`/`speed`) and `orders._try_push_kick()` (the single shared
function behind all three rules-AI push-kick call sites). Physically the
more honest fix — real players do strike a grounded ball with enough
forward roll to already be rolling true — and it worked: passing/push-kick
distance was restored.

**Reverted the same session.** It broke something more important:
`tests/ai_scenario/test_rules_ai_nn_replay_equivalence.py` — the test
explicitly flagged earlier this session as critical ("WE need the neural
AI to be able to reproduce the rules-AI exactly"). Its replay mechanism
hardcodes zero kick spin when simulating what the neural network would do,
because the NN's own kick path is *separately, deliberately* hardcoded
spin-free (`ai/action/apply_nn_action.py`, see
`agent_plans/spin_implementation_plan.md` — a large, not-yet-implemented
plan of its own). Once push-kicks genuinely imparted spin, the test's own
built-in safety assertion ("if Phase1RulesAI ever produces nonzero kick
spin, fail loudly — the always-zero-spin replay would otherwise silently
diverge from real physics for a reason unrelated to any genuine bug")
correctly fired: the rules AI was now doing something (rolling true via
imparted spin) the neural network fundamentally cannot reproduce, a real
demonstrator/imitator mismatch for BC training, not a test bug.

Given `spin_implementation_plan.md` is a large, separate undertaking (full
network architecture change, PPO retraining loop changes, BC label schema
bump, demonstration re-recording — not something to fold into this fix),
the user chose to revert the spin approach entirely rather than take on
that scope. `rolling_spin_for_direction` was deleted (dead code, no
remaining callers) and `pass_ball`/`_try_push_kick` both launch spin-free
again, same as before this section started.

### 8.5 Actual fix: kick harder instead of avoiding the cost

Every kick path stays spin-free (NN-replayable), and the lost distance is
compensated by launching faster instead — accepting the ground-contact
spin-up cost as real and paying extra speed up front to still land near
the intended target, rather than avoiding the cost via spin.

- **`kicking.PassingParams`** gained `spinup_speed_boost_base`/
  `spinup_speed_boost_per_m` (default 1.0/0.0 = no compensation), applied
  multiplicatively in `pass_speed_mps` alongside the existing
  `power_overshoot_factor`. Linear in distance, not a flat constant — the
  required compensation was found empirically to grow with distance
  (longer/faster passes spend proportionally longer paying the transition
  cost before genuinely rolling): a flat multiplier undershot short passes
  or overshot long ones. Tuned to **base=1.29, per_m=0.0025** against
  `tests/balance/test_pass_balance.py`'s actual success-rate targets (the
  same empirical-tuning approach the original `power_overshoot_factor` was
  calibrated with).
- **`orders.json`'s `push_kick.spinup_speed_boost`** (default 1.0), applied
  multiplicatively in `_push_kick_power_fraction` (still clamped to
  `power_fraction <= 1.0`). Tuned to **1.3** against
  `tests/scenario/test_move_order.py::test_push_kick_box_to_box_realistic_kick_count`
  (was taking 32 touches to cross the pitch box-to-box at the pre-tuned
  value, expected 4–15; 1.3 restored it to passing).

**A real ceiling was found while tuning passing, not fully closeable by
speed alone**: `kick_sigma_rad`'s angular error scales with
`effective_power^power_error_exponent` (1.8), so pushing launch speed up
to compensate for lost distance *also* inflates angular error — past a
point, more speed stops helping "does the ball reach the target" and
starts actively hurting "does it arrive accurately enough to be
collected." At 60m (the longest tested pass distance), the best
achievable success rate via pure speed compensation was **33.5%**,
short of the pre-existing test's `>35%` threshold — confirmed via a
genuine grid search (7+ `(base, per_m)` combinations tried; results were
not monotonic in either parameter past this point, consistent with a real
local optimum, not an under-explored search). **User-authorized fix**:
lowered that one test's threshold from 35% to 30% (comfortable margin
below the 33.5% actually achieved), documented inline in the test with the
reasoning above — see
`test_good_player_succeeds_over_30_percent_at_60m` (renamed from
`..._35_percent_...`) in `tests/balance/test_pass_balance.py`. Every other
pass/push-kick-distance test passes at its original, untouched threshold.

### 8.6 Verification

- New unit tests in `tests/unit/test_ball_physics.py`:
  `test_zero_spin_ball_pays_spinup_cost_transitioning_to_rolling` (confirms
  the `1/(1+k)` retention, generous tolerance for drag) and
  `test_zero_spin_ball_travels_less_far_than_a_ball_already_rolling`. The
  two pre-existing rolling-friction tests
  (`test_rolling_ball_decelerates_at_the_analytically_correct_rate`,
  `test_rolling_ball_travels_plausible_distance_before_stopping`) were
  updated to start with matching rolling spin, preserving their original
  intent (isolating `rolling_friction_coefficient`'s own decay) now that a
  zero-spin ball no longer starts there for free.
- After reverting spin and adding speed compensation: all 23
  previously-broken tests pass again (16 pass-balance/behaviour tests from
  §8.3, the push-kick count test, and — critically —
  `test_rules_ai_nn_replay_equivalence.py`'s 3 parametrized cases, confirming
  the revert actually fixed the NN-replay break). Full suite re-run clean,
  matching the pre-existing 2-failure baseline plus the one deliberately
  lowered threshold from §8.5, no unexplained regressions.

### 8.6 The distance consequence, once both fixes are in

With the spin-up cost now real AND passes/push-kicks correctly exempted
from paying it, a genuinely airborne/lofted zero-spin kick (which still
pays the cost on landing, per §4) was directly compared against a rolled
zero-spin kick (which now also pays it, immediately, up front) at matched
total launch speed. This reversed an earlier finding from this same
session that predated the spin-up fix: with the rolled ball now correctly
paying its own realistic transition cost too, a lofted kick travels
**42%–132% farther** than a rolled kick at the same total speed across the
range tested (8/12/16 m/s, 15°/30°/45° loft) — shallower angles do best,
since more of the total speed goes toward forward distance during the
free (drag-only, no-friction) flight phase before any ground-contact cost
is paid at all. See conversation transcript for the full table; not
reproduced here since it's empirical output, not a design decision.
