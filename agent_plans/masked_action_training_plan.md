# Train action heads only where the action could take effect ("trigger masking")

> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

## 0. Status

**Phases 1 and 3 implemented (2026-09-21, later the same day); Phase 2 folded into Phase 3; Phase 4 = config change
+ PPG refit, see §14.** Written 2026-09-21 on request, while PPO run 299
(`checkpoints/phase1_run299`) was running unchanged (it was later stopped to raise
`kick_armed_penalty_per_second` to -0.2 with a 5-rollout PPG refit, then PPO resumed — a stop-gap
for the same symptom this plan fixes properly). File/line references below are as of the planning date
(line numbers drift — use the function names); §14 records what was actually built and where it differs.

This plan replaces two earlier ideas that were tried or proposed and found wanting:

* **Penalising futile arming** (`tackle_armed_penalty_per_second`,
  `kick_armed_penalty_per_second`) — a shaping hack that pays for the symptom, and
  whose "possessing multiplier" could not even fire until `action.arm_tackle_without_carrier`.
* **State-based eligibility from the current row** (e.g. "the trainee has the ball")
  — **does not work**, see §3. A first-touch kick is armed in a row where the
  player has the ball neither before nor after; a pre-armed tackle is decided
  before the opponent has the ball. Eligibility is only knowable *after* the
  decision interval has played out.

## 1. Why (measured evidence, 2026-09-21)

Probe on run 298 checkpoint 3 vs the rules AI (`scratchpad/arm_probe.py`, 19.7k
trainee decisions, natural sampling):

| | kick | tackle |
|---|---|---|
| decisions sampled `1` | 14.0% | 12.0% |
| … of which the action could not possibly fire | **97%** (2,667 / 2,747 made without the ball) | **93%** (no opposing carrier on any tick) |
| P(action sampled ∣ ball held) | 1.1% (80 / 7,268) | 9.6% |
| P(action sampled ∣ no ball) | 21.5% | 17.2% |
| sampled decisions that actually fired | 12 of 2,667 armed kicks fired (0.4%); all 80 possession kicks fired | contact resolved in ~4% |

Consequences observed in training:

* `kick_prob` rises 5% → 17% over a few checkpoints (entropy bonus `ppo.ent_kick_weight`
  is applied on every row) while **physical kicks per episode stay flat (~0.17–0.19)**.
  The extra probability lives in states where the gate cannot matter.
* The gradient on the gate/direction/power heads from those rows is pure noise
  (the action has zero causal effect on the return), diluting the real signal.
* A large part of the shaping code exists only to make "arm but nothing happens" cost something.

## 2. Goal / non-goals

**Goal.** The kick gate (`kick`), the tackle gate (`tackle_attempt`) and the kick
parameter heads (`kick_dir`, `kick_power`) receive policy-gradient (and entropy)
signal **only on decision rows where the action had a chance to take effect
within that row's decision interval**, so no penalty/illegal-action machinery is
needed to keep them honest.

**Non-goals.** Changing inference behaviour; changing the reward (except the
optional retirement of the armed-penalties in Phase 4); touching the value
function/GAE; masking the movement heads.

## 3. Why the row cannot tell you (and what that forces)

Decision row *t* is made at an instant; the action's effect happens over the
next `_ticks_per_decision` (~6) ticks:

* **Possession kick** — ball held at the decision tick: fires on tick 0.
* **First-touch (armed) kick** — ball loose and *not yet touched*: `kick_armed`
  is set, and the kick fires only if the player reaches the ball within the interval
  (`Match._update_loose_ball_pickup`). At decision time and after the fire the player has no ball.
* **Tackle** — `tackle_armed` resolves only if an opposing carrier is in contact
  range on some tick (`Match._check_armed_tackles`); the opponent may *acquire*
  the ball mid-interval (the pre-arm case the user explicitly wants to keep).

Therefore the flags must be **outcome/event-based, computed by the engine/env
over the interval and stored per row**, and for correctness they must exist for
*both* action values (§5): "fired" alone is not enough, we also need "an
opportunity existed" for the `0` rows.

## 4. Definitions

Per player *p* and decision row *t* (interval = the ticks between the fresh
decision that produced row *t* and the next fresh decision, or episode end):

| flag | meaning |
|---|---|
| `kick_opp` | there was a tick in the interval at which a kick by *p* could have executed: *p* held the ball at the start of that tick, **or** *p* gained the ball (first touch / pickup) during it |
| `kick_fired` | *p* executed a kick in the interval (`Player.kick_count` delta > 0 — already exists, monotonic, added 2026-09-21) |
| `tack_opp` | on some tick an opposing player was the ball carrier within contact range of *p* (the exact predicate `_check_armed_tackles` uses: different teams, both `is_available_to_tackle()`, `dist < auto_tackle_overlap_factor × (r_p + r_carrier)`) |
| `tack_fired` | an **armed** tackle by *p* reached `_attempt_tackle_contact` (win or lose). Head-on *auto* tackles (`_check_head_on_tackles`) are NOT fires |

Invariants (checked at runtime, §8 L1): `kick_fired ⇒ kick_opp ∧ kick=1`;
`tack_fired ⇒ tack_opp ∧ tackle_attempt=1`. **The converse does NOT hold** (verified in
code, 2026-09-21): `Match._update_loose_ball_pickup` fires an armed kick at pickup only if
`player.kick_armed ∧ kick_armed_direction is not None ∧ ball_settled` (`|ball.velocity.z| <
armed_redirect_settle_vz_mps`). If the ball is still bouncing the player just takes control
(first-touch difficulty) and the still-armed intent fires on the *next* tick through the
cached-gating re-apply — which is in the *next decision interval* when the pickup was on the
last tick, and by then the intent has expired. So `kick=1 ∧ kick_opp ∧ ¬kick_fired` exists.
That is fine for the gate (its outcome equals the `kick=0` outcome, i.e. a zero-effect row inside
the opportunity class — dilution, not bias) and is exactly why `kick_dir`/`kick_power` are
masked by `kick_fired`, not by `kick_opp`. Phase 1 must **measure** how often it happens.

**Which rows train which head**

| head | trained when |
|---|---|
| `kick` (gate) | `kick_opp` (both `kick=0` and `kick=1` rows) |
| `kick_dir`, `kick_power` | `kick=1 ∧ kick_fired` (the params only have an effect if it fired) |
| `tackle_attempt` (gate) | `tack_opp` (both values) |
| entropy bonus for the same heads | same masks (the point: exploration where it matters) |

## 5. Statistical correctness — the part most likely to be gotten subtly wrong

Bernoulli gate with logit z, p = σ(z), reward difference ΔQ = Q(fire) − Q(don't) *in rows where the action can take effect* and 0 elsewhere.

* Unmasked expected gradient: `p(1−p)·P(opp)·ΔQ` **plus** pure-noise terms from the
  non-opportunity rows (mean-zero only if V is unbiased; in practice they dominate the variance) **plus** the entropy
  bonus acting everywhere.
* **Design A — "fire-only positives, all negatives"** (train on `kick=1 ∧ fired`
  and *every* `kick=0` row): conditions on the *action outcome* for `1` rows but not for `0` rows → biased. In
  non-opportunity states all `1` samples are dropped and all `0` samples kept: the `(0 − p)·A` terms no longer
  cancel against the dropped `(1 − p)·A` terms, so the gate drifts down (or up) with V's noise. **Do not implement A.**
* **Design B — opportunity mask (this plan)**: mask `0` and `1` rows by the *same* event `opp`.
  Valid iff **(I1)** `opp` is independent of the gate bit given the state and the other heads' actions.
  Under I1 the masked gradient is `p(1−p)·ΔQ` restricted to opportunity rows, i.e. the right direction with
  magnitude scaled by P(opp) (a per-state constant learning-rate factor, see D9).

**I1 holds by construction only for the *first* opportunity.** After the gate acts, the world diverges
(possession kick releases the ball; a=0 keeps it), so `opp` must be defined from the first
opportunity tick, which is identical under a=0 and a=1 (the action only takes effect *at* that tick). That is
why §4 says "a tick at which a kick *could have* executed" rather than "the ball was ever held".

**Known I1 violations (cross-head, same row) — D2 below:**
tackle bit → possession gained by winning the ball → creates a `kick_opp`; own kick fire → releases the ball
→ opponent gets it → creates a `tack_opp`. Both need cut-off rules and a measurement test.

## 6. Design

### 6.1 Engine (single source of truth for events)

* `Player` gets monotonic counters, never reset: `kick_count` (exists), `tackle_fire_count`,
  `kick_opp_count`, `tackle_opp_count`, plus "first tick index in the current interval" markers
  (`interval_first_kick_opp_tick`, …) that the env resets when a fresh decision is applied.
* **Fires:** `kick_count` (already in `Player._finish_kick`); `tackle_fire_count` in `Match._attempt_tackle_contact`
  when reached from `_check_armed_tackles` only.
* **Opportunities:** evaluated *inside the engine, at the same instant the action would resolve*:
  * kick: possession at tick start (`_process_orders` entry, before any kick this tick) and every `_set_possession(p)`
    where `player_id != old` (pickups, tackles won, armed auto-fire — it already goes through `_set_possession`);
  * tackle: in `_check_armed_tackles`, evaluate the predicate for **every** player each tick (armed or not) **without** the
    existing early `return` after the first tackle (that `return` must not truncate opportunity accounting).
  Do **not** re-derive these in `ScenarioEnv` after `Match.step()` returns: positions have already gone
  through `resolve_all_overlaps` by then, so the predicate would disagree with the one that actually resolves.
* Refactor the tackle-contact predicate into one function used by both resolution and opportunity accounting.

### 6.2 Env / transitions

* On every fresh decision (`NeuralPlayerAI.apply`) snapshot the player's counters; when the interval closes
  (next fresh decision, or terminal) compute deltas + cross-head cut-offs and attach four bool fields to
  `last_trainee_transition` and to each `last_secondary_results[i]`.
* **Interval alignment (D1):** `ScenarioEnv.step()` covers one decision interval *except* for the early exits
  (`tick_done` break; "trainee in opponent box with possession" break) and the "trainee not-yet-due" case in
  `BatchedEnvGroup.collect`. Events must be attributed to the *decision's* interval, not the `env.step()` call.

### 6.3 Buffer / workers

* `RolloutBuffer.add(..., kick_opp=, kick_fired=, tack_opp=, tack_fired=)`; `as_tensors` emits `flag/{name}` (N,) float32.
  **Missing keys ⇒ all-ones masks ⇒ bit-identical old behaviour** (opt-in convention).
* Three producers must be updated identically: `ai/ppo/batched_rollout_worker.py`
  (`BatchedEnvGroup.collect`, trainee + secondaries), `ai/ppo/rollout_worker.py` (plain workers), and the single-process
  loop in `ppo_trainer.py` (`buffer.add(head_log_probs=tr.get(...))`, ~2317 / ~2345). Also `_merge_worker_batches`, chunked
  streaming flushes, episode replay re-collection, value-pretrain/PPG paths (they call the same workers).
* `augment_batch` / `_recompute_old_log_probs_for_augmented_batch` tile rows: flags must tile identically
  (currently `augment_n_slot_shuffles=0` and `y_canonical` on, so this is latent — but it must not silently break).

### 6.4 Trainer

* **Ratio.** Stored `log_probs` is the *total* over all heads (sample-time, ~L7470 comment: per-head values "match what went
  into the stored total"). With masking, old and new totals must exclude the same terms. Options:
  (i) recompute old total = Σ_h mask_h·`head_log_probs[:,h]` and new total = Σ_h mask_h·`_per_head_new_log_probs[:,h]`
  (both already exist; requires the stored per-head columns to sum to the stored total *exactly* — test T7);
  (ii) keep the old total and subtract masked-out per-head terms. **Prefer (i)**, gated on the flag, so the flag-off path is untouched.
* Masks applied to: policy ratio, dual-clip, per-head counterfactual policy loss, **entropy** (`_compute_entropy`: `kick`,
  `kick_dir`, `kick_power`, `tackle_attempt`; today `kick_dir`/`kick_power` are weighted by mean p(kick), replace by the mask),
  per-head KL diagnostic (`[per-head KL]`, `[policy heads]`), `head_act` rates, the new `sampled power` stat (use `kick_fired` rows),
  `target_kl` early stop.
* **Not masked (decision):** PPG / consistency anchor KL (keeps behaviour near the reference everywhere), BC aux loss.
* New rollout diagnostics: `P(opp)`, `P(fired|a=1)`, `P(a=1|opp)`, counts per head, invariant-violation counters.

## 7. Difficulties / risk register

| # | difficulty | why it bites | mitigation | test |
|---|---|---|---|---|
| D1 | interval ≠ `env.step()` call (early exits, not-yet-due decisions, terminal mid-interval, secondary players) | flags land on the wrong row or are dropped | attribute by fresh-decision boundaries via player counters, not per-step scans; define terminal rule explicitly; count & assert misalignments | T1, T5, T12 |
| D2 | cross-head dependence: own tackle creates a kick opportunity, own kick creates a tackle opportunity | breaks I1 → biased gate | cut-off rule: an opportunity counts only if its tick ≤ the first tick of this player's *other* head's fire; measure the affected fraction | T2, T9 |
| D3 | I1 for `0` rows needs counterfactual events (a ball pickup that "would" have fired a kick) | naive "fired" flag is not enough | opportunity events recorded from world state, independent of the action | T2, T3 |
| D4 | tackle predicate evaluated after overlap resolution / early `return` after one tackle per tick | opportunity ≠ what resolution used; multi-player truncation | account inside `_check_armed_tackles` for all players, single predicate function | T1, T4 |
| D5 | multiple events per interval (touch → kick → re-touch; tackle → lose → re-tackle) | double counting, wrong first tick | store first-tick markers + counts; masks use "any" | T1 |
| D6 | `kick_opp` does not imply a fire: armed kicks only fire at pickup if the ball is vertically settled, otherwise the fire slips to the next tick / next interval (see §4); also first-touch difficulty, CONTROLLING_BALL, `kick_with_direction` refusals | a=1 ∧ opp ∧ ¬fired would train dir/power on a kick that never happened | `fired` is defined strictly by the `kick_count` delta and is what masks `kick_dir`/`kick_power`; `opp` masks only the gate; measure the `a=1 ∧ opp ∧ ¬fired` share in Phase 1 and log it | L0, L1 |
| D7 | old/new log-prob consistency after masking | ratio ≠ 1 at epoch 0 → spikes, dual-clip storms | per-head decomposition (option i); identity test on real batches; keep "epoch 0 ratio = 1 exactly" baseline check | T7 |
| D8 | stored per-head log-probs zero for inactive heads in a way that differs from the total | mismatch hides in aggregate | audit `head_log_probs` construction vs `_compute_log_prob` for all 15 heads | T7 |
| D9 | masked heads get ~1/P(opp) less gradient | gate learns ~10–100× slower (tackle contact rows ≈ 0.15% of rows) | optional per-head loss rescale by `1/(batch mean of mask)`; make it a config; watch lr interaction and Adam | T6, T9 |
| D10 | `ent_coef` semantics change (entropy over fewer rows) | effective exploration differs from today's tuned values; the `ent_kick_weight=6` calibration (§ in the entropy discussion) no longer applies | retune from measured `P(opp)`; log effective coefficient | T8 |
| D11 | shared trunk: masked rows' outputs drift (no gradient) | arbitrary p(kick) in irrelevant states — harmless *unless* the state later becomes an opportunity within the same interval | opportunity rows are trained *in* those near-touch states; verify with the probe (P(fire|arm) should rise) | manual probe |
| D12 | sample sizes: tackle `tack_opp` rows are rare (~1–5k / iteration), kick opp rows numerous | tackle gate under-trained / noisy | report counts; consider larger batches or aggregate over rollouts; do not mask if count below a floor | T9 |
| D13 | secondary (neural opponent) rows: different `track_id`, weight, buffer segmentation | flags applied to wrong track / GAE segmentation unaffected but masks are | flags travel with the row through `add()`; test with `batch_secondary_players` true/false | T5 |
| D14 | old checkpoints / old logs / tests without flag keys | crashes or silent behaviour change | missing ⇒ all-ones; config default off; golden test for identity with flag off | T12 |
| D15 | `augment_batch` tiling, minibatch indexing, episode replay, chunked flush, merges lose or reorder the new keys | silent misalignment (worst class of bug) | one property test that round-trips random flags through every transformation | T4 |
| D16 | reward shaping interplay: armed penalties still paid on masked rows | noise in advantages of *other* heads | Phase 4: set armed penalties to 0 (needs a PPG refit: value function was trained with them) | T10 |
| D17 | first-touch semantics changed by `action.kick_one_shot` / `arm_tackle_without_carrier` (2026-09-21) | opportunity definitions assume those semantics | write tests under both flag settings; document the required setting | T1 |
| D18 | determinism / performance | extra per-tick work in the hot path | counters only (O(players)/tick); benchmark | T11 |
| D19 | interpretability of existing diagnostics changes (`kick_prob`, `entropy=`) | trends across the restart are not comparable | log both masked and unmasked versions for a transition period | T8 |
| D20 | someone "fixes" the invariant checks by loosening them | flags silently rot | invariants are assertions with counters printed every rollout; treat any nonzero as a failure in tests | L1 |

## 8. Testing strategy (extensive, adversarial — this is the fragile part)

Layers, each a separate file; **flag-off must be bit-identical** everywhere.

**L0 — event semantics (scripted `Match`/`ScenarioEnv`, no network).** Deterministic scenarios ×
{action 0, action 1} × {`kick_one_shot` on/off}:
possession + kick; possession + no kick; loose ball far (expires unfired); loose ball reachable within the interval
(armed fire); ball reachable on the *last* tick; ball reachable one tick *after* the interval; opponent carrier far / in
range / arriving mid-interval (pre-arm) / leaving; own-team carrier; player knocked down (`INACTIVE_TACKLED`);
goal/out/timeout mid-interval; early-exit-in-box; two events in one interval; secondary players.

**L1 — invariants (also compiled into the trainer as per-rollout assertions with counters).**
`fired ⇒ opp ∧ action`; `kick=1 ∧ opp ⇔ fired` (or the documented exceptions); flags ∈ {0,1}; flags aligned with the row that owns them;
`P(opp)` sane; violation counters must be 0 in tests and are printed in the log.

**L2 — the independence property (the important one).** For hundreds of random seeded states,
force the interval twice from an identical snapshot with the gate bit 0 vs 1 (same RNG stream, same other heads) and assert
`opp(a=0) == opp(a=1)` for both heads, measured with and without the cross-head cut-off — quantifies D2 and would catch a
definition that leaks the action. Also assert `fired(a=1) == opp`. Forking: `copy.deepcopy(match)` works and is deterministic (verified 2026-09-21: two copies of a match, RNG
included, stay bit-identical over 60 ticks — `scratchpad/snap_test.py`), so fork from a snapshot, apply the scripted
gating (`apply_action_to_player` with the gate bit forced) to each copy, and step. `NeuralPlayerAI` holds trainer closures
(deep-copying the network is huge and pointless), so the forks must use scripted gating, not the network.

**L3 — plumbing round trips.** Property tests that push random flag arrays through `add → as_tensors → _merge_worker_batches
→ chunk flush → episode replay → augment_batch tiling → minibatch indexing` and assert exact equality per row; batched vs plain
worker parity for identical seeds (deterministic env); trainee vs secondary rows; missing-key default.

**L4 — loss/gradient tests (autograd).** Masked head ⇒ *exactly zero* gradient from masked rows on that head's parameters
(and nonzero from unmasked); trunk still receives gradient from other heads; all-ones masks ⇒ loss and gradients equal to
the current code to float tolerance on a fixed real batch; ratio == 1 exactly at epoch 0 with masks on; dual clip / per-head
policy loss / KL diagnostics use masks consistently; entropy uses masks and the `ent_kick_weight` boost composes.

**L5 — statistical/bandit test.** Synthetic environment where the action only matters when an event flag is set
(known ΔQ): (a) masked training converges to the analytic optimum; (b) unmasked drifts/entropy-inflates in irrelevant states;
(c) **Design A** demonstrably biases (kept as a documented negative test so nobody re-introduces it); (d) verifies the 1/P(opp)
rescale option; (e) cross-head leak test with an engineered dependency.

**L6 — integration.** 3–5 iteration PPO smoke with flag on: no NaN, sane counts, invariants zero, `ratio` diagnostics normal;
flag-off run reproduces a stored golden log line set within tolerance; kill/restart from checkpoint; PPG refit still runs.

**L7 — behavioural probe (manual, before/after).** Re-run `arm_probe.py` and the kick harness: P(kick|ball held),
P(fire|armed), physical kicks/episode, sampled power, win rate. Expected: armed-without-touch decisions stop growing;
P(kick ∣ ball held) responds to `ent_kick_weight`.

**Fault injection.** Flip a flag; shift flags by one row; drop the secondary flags; corrupt one worker's output — each must be
caught by an invariant or a parity test (this is how we know the tripwires work).

## 9. Rollout plan (go / no-go gates)

1. **Phase 1 — instrumentation only.** Engine counters + env attribution + buffer/worker plumbing, **no change to the loss**.
   Log `P(opp)`, `P(fired|a=1)`, `P(a=1|opp)`, invariant counters. Gate: L0–L3 green, invariants zero over a full PPO
   iteration, batched/plain parity. This alone answers "how bad is it" with real numbers and de-risks everything else.
2. **Phase 2 — entropy-only masking** (cheapest behavioural change: makes the kick-entropy boost act where it can matter).
3. **Phase 3 — full policy-gradient masking** for the four heads, behind `ppo.mask_untriggered_actions` (default off), with the per-head
   ratio decomposition. Gate: L4/L5 green, epoch-0 ratio exactly 1, short A/B run vs unmasked.
4. **Phase 4 — retire the hacks.** Set `tackle_armed_penalty_per_second` and `kick_armed_penalty_per_second` to 0, reconsider the
   attempt bonuses, run a PPG value refit (the value function was trained with them), remove or keep `arm_tackle_without_carrier`
   per the semantics below.

A/B protocol: same start checkpoint, same seeds, ≥ 25 iterations each; decide on (a) physical kicks/episode and
P(fire|armed), (b) eval vs rules win / reward with the pooled-eval harness, (c) stability (ratio spikes, KL).

## 10. Config

`ppo.mask_untriggered_actions` (bool, default false) — Phase 3; `ppo.mask_rescale_by_opportunity_rate` (bool, default false) — D9;
`ppo.log_action_opportunity_stats` (bool, default false) — Phase 1 diagnostics. All with `_comment_` entries.

## 11. Open questions for the user

1. **Cross-head cut-off (D2):** accept the "opportunity counts only up to the other head's first fire" rule, or ignore the (rare?) leak and just measure it?
2. **Rescale (D9):** should the masked gate gradients be rescaled by 1/P(opp) (faster learning, higher variance) or left alone?
3. **Anchor losses:** keep PPG/consistency anchors unmasked (my default)?
4. **Armed penalties:** retire them in Phase 4, or keep small ones as a tie-breaker?
5. **Pre-arm across intervals:** each decision re-decides every interval, so an arm never persists past its interval — confirm that is the intended pre-arm semantics.
6. **Secondary/opponent rows:** apply the same masking (default yes).

## 12. Effort / risk (honest estimate)

Phase 1 ≈ 1 day incl. L0–L3 (engine counters, env attribution, three producers, merge/chunk/replay/augment, tests);
Phase 2 ≈ half a day; Phase 3 ≈ 1–1.5 days (ratio decomposition, entropy/KL/dual-clip paths, gradient tests, bandit test);
Phase 4 ≈ half a day + a PPG refit. Highest-risk items: D1 (alignment), D2 (cross-head leak), D7/D8 (log-prob identity), D15 (silent
row misalignment). A run started before Phase 1 lands is unaffected; every phase after Phase 1 requires a restart of the workers.

## 13. Code map (as of 2026-09-21)

* `engine/match.py`: `Match.step` (tick order: timers → `_process_orders` → `_apply_movement` → free flight →
  `_update_loose_ball_pickup` → `_check_armed_tackles` → `_check_head_on_tackles` → overlaps → goal), `_set_possession`
  (single write path, `on_possession_gained`), `_check_armed_tackles`, `_attempt_tackle_contact`.
* `entities/player.py`: `kick_count`, `_finish_kick`, `tackle_armed`, `kick_armed`.
* `ai/action/apply_nn_action.py`: kick/tackle application (`kick_one_shot`, `arm_tackle_without_carrier`).
* `rules_ai.py`: `NeuralPlayerAI.prepare/apply/_reapply_cached_gating` (fresh-decision boundary).
* `ai/env/scenario_env.py`: `ScenarioEnv.step` tick loop (~L476), `last_trainee_transition`, `last_secondary_results`.
* `ai/ppo/rollout_buffer.py`: `HEAD_LP_KEYS`, `add`, `as_tensors`.
* `ai/ppo/batched_rollout_worker.py`, `ai/ppo/rollout_worker.py`, `ppo_trainer.py` single-process loop: `buffer.add` producers.
* `ai/ppo/ppo_trainer.py`: `_sample_action_from_heads` (stored `head_log_probs`), `_compute_log_prob`, `_per_head_new_log_probs`,
  `_ppo_update` (ratio ~L8946, per-head policy loss, `_compute_entropy`, `ent_kick_weight`), `_recompute_old_log_probs_for_augmented_batch`,
  `_ppg_kl_penalty` / `consistency_refit` (anchors).
* Evidence tooling: `scratchpad/arm_probe.py`, `scratchpad/kick_par.py` (session scratchpad, not in the repo — copy into `ai/scripts/` if kept).

## 14. Implementation record and decisions (2026-09-21)

**User decisions on the §11 questions.** (1) Cross-head leak: measure it; a "one action type per decision step" rule is
acceptable. -> implemented as the *first-opportunity-wins* independence rule (only whichever head's opportunity comes
first in the interval counts; `ppo.action_opportunity_cross_head_cutoff`, default true; raw flags always recorded).
Measured on run 303 checkpoint 95 (750k rows): both-raw = 0.5% of intervals, the rule drops 1.2% of raw kick and
2.2-2.6% of raw tackle opportunities. (2) No 1/P(opp) rescale by default (`ppo.mask_rescale_by_opportunity_rate=false`;
implemented as a value-preserving per-head gradient multiplier, capped at 100). (3) PPG / consistency anchors and BC stay
unmasked. (4) Retire the armed penalties (`tackle_armed_penalty_per_second`, `kick_armed_penalty_per_second` -> 0, which also
zeroes the attempt bonuses -- they are derived from the same magnitude) and PPG-refit the value function. (5) Arming
lasts one decision interval and is re-decided each interval -- confirmed. (6) Secondary/opponent neural rows are masked
the same way.

**Where the implementation differs from the plan text.**
* Flags are not a separate `flag/*` buffer field: they are extra keys of the row's action dict (`action/opp_kick, opp_kick_raw,
  fired_kick, opp_tack, opp_tack_raw, fired_tack, opp_partial`), so merge / chunk flush / episode replay / train-val split /
  augmentation need no changes (augment already treats unknown `action/*` keys as flip-invariant). Missing keys => all-ones masks.
* Opportunity events are stamped by `Match.tick_index` and recorded per `Player.opportunity` (`entities/action_opportunity.py`);
  the flags are attached at the end of `ScenarioEnv.step()` (which spans one decision interval) via
  `NeuralPlayerAI.apply`'s `begin_interval`. Not-yet-due / early-exit intervals are marked `opp_partial` (0.1-0.2% of rows) instead of
  being re-attributed.
* Validity has ONE source of truth, shared with the accounting: `Player.can_kick` (kick executors + fire-vs-arm branch +
  accounting) and `Match._tackle_contact_possible` (armed-tackle resolution + accounting; includes the `INACTIVE_TACKLED` checks on
  both players). A kick opportunity also includes loose-ball pickups (first touch), recorded inside `_update_loose_ball_pickup`.
* Masked ratio: the plan's option (i) with the non-column residual of the stored total kept (detached), so masked heads get exactly
  zero gradient; requires phases whose pass/tackle/mark decision heads are frozen (raises otherwise).
* Phase 2 (entropy-only masking) was not built as a separate stage: entropy masking ships inside Phase 3 behind the same flag.
* Not built (deferred): per-head KL/`[policy heads]` diagnostics on the UNmasked policy alongside the masked ones, and the
  `ppo.target_kl` change (the masked KL feeds the existing early stop unchanged).

**Measurements (Phase 1, run 303 checkpoint 95, 750k rows/rollout, 3 rollouts agree to within ~0.5 pp).** P(kick opportunity)
37-38%; P(tackle opportunity) 1.5-1.6%; P(kick=1) 0.47%, of which 97.7% had an opportunity and 97.7% fired (100% of those with
one); P(tackle=1) 3.2-3.7% overall but 27-28% inside opportunities and 12% of tackle=1 samples had an opportunity;
P(fired | tackle=1) 12.5%. Rows that would train: kick gate ~280k, tackle gate ~11-12k, kick dir/power ~3.5-3.8k per rollout.

**Tests.** `tests/ai_scenario/test_action_opportunity_flags.py` (L0 event semantics incl. first touch, inactive players,
first-opportunity-wins, L1 invariants, L2 independence across the four bit combinations, validity single-source guards),
`tests/ai_scenario/test_masked_action_training.py` (L4 log-prob identity, epoch-0 ratio, exact-zero gradients, entropy masks,
rescale, full `_ppo_update`; L5 bandit incl. the Design-A negative test). Mutation-checked: removing the pickup hook, disabling
the cut-off rule, and loosening the tackle accounting each make tests fail.

