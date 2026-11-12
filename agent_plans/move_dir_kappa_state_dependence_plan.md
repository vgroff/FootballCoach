# State-Dependent move_dir Kappa ("Continuation vs. Correction")

> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

## 0. Status

**Not started — planning only, written up on request during an exploratory
discussion (2026-10-17) about why `move_dir`'s von Mises `log_kappa` keeps
sharpening over the course of `phase1_run321` without its KL ever settling
down.** No code changed. Scope and even whether to proceed past the Phase 0
diagnostic (§5) are open.

## 1. The motivating hypothesis

`move_dir` is emitted every decision interval (0.25s, i.e. 4Hz) — unlike
`kick`/`tackle_attempt`, it is never gated/optional. The user's observation:
most ticks are "continuation" ticks (keep moving in roughly the same
direction as last tick, maybe a small correction), and a small minority are
"correction" ticks (a real, possibly large, change of direction — reacting
to a defender, a loose-ball bounce, a new phase of play). If that's right,
the *correct* conditional distribution over `move_dir` is genuinely
bimodal-in-regime even if each regime's own conditional distribution is
reasonably unimodal: tight around "same as before" most of the time, wide
(or centered somewhere very different) occasionally.

`move_dir_log_kappa` (`execution_network.py:251`) is a single, global,
input-independent `nn.Parameter` — not a function of state at all. Trained
by SGD against the full mixture of ticks, it converges to whatever is best
*on average* across both regimes, and since continuation-ticks vastly
outnumber correction-ticks, the gradient signal from the majority regime
dominates. This is a standard reason a single global concentration/variance
parameter underperforms a state-dependent one in a genuinely heterogeneous-
regime problem — and it would explain both observed symptoms at once:

- Why `move_dir`'s KL doesn't settle: each further increase in the single
  global `kappa` is still net-beneficial in expectation (the majority class
  keeps rewarding it) even though it's making the minority class worse,
  so there's no equilibrium at a value that "satisfies everyone" — the
  optimizer just keeps climbing on the majority's gradient.
- Why naive *global* interventions (the unsharpen-sweep's flat kappa
  reduction, `--reset-dir-log-std`, or a generic entropy bonus) cost
  performance out of proportion to any exploration benefit: they pay the
  "continuation" cost on ~99% of ticks to buy a diffuse, undirected benefit
  on the ~1% of ticks that might actually need more spread. This matches
  the unsharpen-sweep data we already collected: 10% global kappa reduction
  was statistically free, 30% started to show a real (if not yet
  significant) cost — consistent with, but **not proof of**, a bimodal
  regime (a smooth, non-bimodal diminishing-returns relationship would
  produce a qualitatively similar curve; see §5 for how to actually
  distinguish these before committing further).

## 2. What's already available for free

Checked `src/footballcoach/ai/obs/encoder.py:289-304`: the observation
**already includes the player's own previous-decision movement intent**
(`desired_dir_x`/`desired_dir_y`, read from `player.desired_direction`,
real signal even at standstill). So the raw input needed to let a
state-dependent mechanism discriminate "this tick looks like a continuation
of what I just decided" from "this tick looks like a correction" is already
present in the trunk's input — nothing new needs to be added to
observations for this specific idea. (How well the trunk's existing MLP
stack has *compressed* that signal into the final hidden state by the time
it reaches a would-be kappa head is a separate, unverified question — see
§5.)

## 3. Literature grounding

- State-dependent concentration/variance for a continuous action distribution
  is the standard construction in Mixture Density Networks (Bishop 1994) and
  is how SAC computes its Gaussian `log_std` (a function of state, not a
  free parameter) — so the general idea is well precedented.
- However, for **PPO specifically**, Andrychowicz et al.'s large-scale
  implementation-choice study ("What Matters for On-Policy Deep Actor-Critic
  Methods?", 2020) found **state-independent** `log_std` tends to be *more*
  stable than state-dependent for continuous control — the known failure
  mode is variance collapse: a state-dependent variance head can drive
  itself toward near-zero in whatever states dominate early rollouts, and
  then has vanishingly small gradient signal to recover, a rich-get-richer
  dynamic. A **fully free** state-dependent `move_dir_log_kappa` inherits
  this exact risk and is **not** recommended as a first step.
- A single von Mises (however wide) is always unimodal. If the "correction"
  regime genuinely has two or more distinct good directions at once (not
  just "less certain about one direction"), no amount of widening a single
  von Mises represents that correctly — that would need a small mixture of
  von Mises components, which is a structurally bigger change than anything
  in this plan and is explicitly **out of scope** here.
- This codebase already has a precedent for the specific "introduce a new
  parameter that must start as an exact behavioral no-op" pattern this plan
  needs: `agent_plans/network_seam_widening_plan.md`'s §2 Net2WiderNet
  recipe (new *outgoing* weights zero-init so a newly added unit changes
  nothing until it's learned something). The mechanism below reuses that
  same idea at the level of "new head, zero-init, so its initial output is
  a no-op multiplier of 1.0."

## 4. Proposed mechanism (if Phase 0 supports proceeding — see §5)

A small, **bounded** multiplier on the existing global `log_kappa`, not a
fully free state-dependent replacement of it — directly addressing the
collapse risk from §3:

```
raw = small_head(trunk_features)            # scalar per sample
bounded = tanh(raw) * log(10.0)              # in [-log(10), +log(10)]
effective_log_kappa = move_dir_log_kappa + bounded   # multiplier in [0.1x, 10x]
```

- `small_head`: a single `nn.Linear(trunk_hidden, 1)` (or a tiny 2-layer
  MLP if a linear readout proves too weak — start with linear).
- **Zero-init** `small_head`'s weights and bias, so `raw == 0` and
  `bounded == 0` for every input at the moment this is introduced —
  `effective_log_kappa` is then *exactly* `move_dir_log_kappa` as before,
  for every state, with no discontinuity in behavior (same property §2 of
  the widening plan guarantees for its own zero-init case, and for the same
  reason: the new path starts multiplying everything by exactly 1.0).
- The `tanh`-based bound (rather than a raw unclamped linear output) is what
  prevents the variance-collapse failure mode from §3: the multiplier
  structurally cannot drive `kappa` to 0 or to infinity no matter what the
  head learns, it can only ever rescale the existing global value within a
  fixed, bounded window.
- `move_dir_log_kappa` itself is **unchanged** — still a global scalar,
  still clamped by the existing `move_dir_log_kappa_min`/`_max` config
  machinery. The multiplier is a strictly additive-in-log-space adjustment
  on top.

## 5. Recommended first step: a cheap diagnostic, not an architecture change

Per this repo's standing rule to prove a hypothesis before committing to a
fix for it: **before** touching any network code, build a small offline
analysis script (reusing an existing checkpoint, no training) that:

1. Runs a batch of real rollout states (e.g. replay states already captured
   during normal evaluation, or a few hundred fresh rollout steps) through
   the current `execution_net`.
2. For each state, computes a simple **continuation-likelihood** proxy —
   e.g. the cosine similarity between the network's own predicted mean
   `move_direction` and the state's own `desired_dir_x`/`desired_dir_y`
   feature (the previous tick's intent) — without changing anything about
   `log_kappa`.
3. Buckets states by that proxy (e.g. deciles) and, for each bucket,
   computes the *empirical* per-state von Mises log-prob/KL-to-anchor that
   iteration's rollout already logs per-step (or a fresh computation against
   the fixed global `kappa`), to see whether "surprise"/KL really does
   concentrate heavily in the low-continuation-likelihood bucket(s), the
   way the bimodal-regime hypothesis predicts.
4. Report the result as a real distribution (e.g. a plot of mean |log-prob
   residual| vs. continuation-likelihood decile), not just a single
   aggregate number — the whole point is to see whether this is genuinely
   bimodal/concentrated or just a smooth, continuous relationship (which
   would undercut the specific "99%/1%" framing even if a state-dependent
   kappa might still help a little).

**Only proceed to §4's architecture change if this diagnostic shows a real,
concentrated effect.** If the relationship turns out smooth/diffuse rather
than concentrated, a bounded multiplier is much less likely to be worth its
implementation cost (§6), and the honest conclusion would be that `move_dir`
sharpening is just a normal (if slow-converging) optimization trajectory,
not a structural mismatch needing a new parameter.

## 6. If Phase 0 supports it — full scope of call sites needing an update

This is **not** a small, localized change confined to `execution_network.py`.
`move_dir_log_kappa` is currently treated as a bare `(1,)` scalar
`nn.Parameter` at ~25 call sites across `ppo_trainer.py` (grep
`move_dir_log_kappa` for the full current list) — every one of these
assumes a single global value, not a per-sample tensor, and would need
auditing if `effective_log_kappa` becomes per-sample:

- **Action sampling** (forward pass constructing the von Mises
  distribution to sample `move_direction` from).
- **Log-prob computation**, both for the *current* policy (needed every
  minibatch) and for whatever snapshot of the policy produced the original
  action (the PPO ratio's denominator) — these must use the **same**
  per-sample `effective_log_kappa` that was actually in effect when the
  action was taken, not a re-derived one, or the importance ratio is wrong.
- **Entropy computation** (`_compute_entropy`-style code) — von Mises
  entropy is a function of `kappa`; with a per-sample kappa this becomes a
  per-sample entropy that must be aggregated correctly (mean over batch,
  not evaluated at some single representative kappa).
- **KL-anchor computation** (PPG's anchor-vs-current KL penalty,
  `ppo_trainer.py:7783`/`7792` and neighbors) — same per-sample concern.
- **Gradient-norm clip-group logic** (`ppo_trainer.py:1942`): a new head's
  parameters must be explicitly added to the `move_dir_param_ids`/
  `move_dir_params` grouping (currently matched by
  `name.startswith("move_direction.") or name == "move_dir_log_kappa"`) or
  they'll silently fall into the default param group with the wrong
  learning rate / clip norm — an easy, silent-bug-prone spot.
- **Diagnostics, resets, and refit code** that read/write
  `move_dir_log_kappa` as a true scalar directly (`.item()`, `.fill_()`,
  `.data`, `.grad` — e.g. `train.py:530`'s `--reset-dir-log-std`, the
  gradient-probe code around `ppo_trainer.py:9329-9901`, the
  checkpoint-save/restore dict entries at `ppo_trainer.py:6618`/`7206`/
  `9072`). Each of these either needs to keep operating on the *global*
  `move_dir_log_kappa` specifically (the multiplier is a separate,
  additional tensor, so most of these likely stay correct unchanged) or
  needs a decision about whether it should also report/reset the new
  multiplier head — needs a case-by-case pass, not a blanket assumption
  either way.

Given this breadth, a full implementation should budget for an explicit
audit pass over every call site above before writing the new head, not
just "add the head and see what breaks."

## 7. Open questions (unresolved, needs a decision before implementation)

- Whether to condition the new head on the **full** `trunk_hidden` vector
  (more capacity, more overfitting risk, harder to interpret) or on a
  smaller, explicitly hand-built feature like the continuation-likelihood
  proxy from §5 directly (lower capacity, easier to reason about and to
  verify it's using the mechanism this plan actually hypothesizes about,
  rather than learning something else entirely).
- Whether `kick_dir_log_kappa` should eventually get the same treatment
  (kick is gated/optional, so the "always-emitted, mostly-continuation"
  argument from §1 doesn't obviously apply to it the same way — likely out
  of scope unless Phase 0's diagnostic is repeated for kick_dir and shows a
  similar pattern there too).
- Bound width (`log(10)` i.e. [0.1x, 10x] was picked illustratively in §4
  as "a reasonable wide-but-safe range," not derived from any data — worth
  revisiting once/if Phase 0's diagnostic shows how large an effect is
  actually needed.
- Whether the multiplier head needs its own (likely much smaller) learning
  rate, separate from whatever group it gets placed in per §6's clip-group
  concern, given it's a brand-new, zero-initialized parameter rather than
  an already-warmed-up one.

## 8. Files likely touched (not exhaustive — plan only)

- `src/footballcoach/ai/models/execution_network.py` — new small head
  (§4), zero-init.
- `src/footballcoach/ai/ppo/ppo_trainer.py` — every call site in §6's list
  that currently treats `move_dir_log_kappa` as a bare scalar.
- New, scratchpad-or-script-level: the Phase 0 diagnostic (§5) — not
  necessarily a permanent part of the codebase, could live as a one-off
  analysis script against an existing checkpoint.
- Test file(s) under `tests/ai_unit/` or `tests/ai_scenario/` for an
  exact-preservation-at-init regression test (same pattern as the network
  widening plan's own §7: forward pass before vs. after adding the
  zero-init head must be bit-identical) — if/when §6's implementation
  happens.
