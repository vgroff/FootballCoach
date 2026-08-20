# Reward/Recording Pipeline Refactor Plan

## 0. Why this document exists

Across a single working session (2026-08-18), five separate bugs were found
and fixed in the phase-1 reward computation and demonstration-recording
pipeline:

1. `sec_box_terminal` (a secondary player's own box-possession-terminal
   reward check) had no immobile-player gate, letting an immobile opponent
   collect a one-time terminal bonus every tick for as long as a physics
   fluke held (confirmed: 64 consecutive ticks in one real episode).
2. `debug_value_network.py` had a stale, duplicated copy of episode-boundary
   detection logic that double-counted episodes (240,000 vs the correct
   120,000) in three of its diagnostic functions.
3. `env.last_reward_components` was one dict, additively shared between the
   trainee and every secondary player, then blindly copied onto BOTH
   players' recorded rows — so a trainee's genuine win row also carried the
   losing opponent's own loss penalty, and vice versa.
4. `record_demonstrations.py`'s row-backfill loop captured `n_appended`
   (row count) *before* `env.step()` ran, then used it with negative-offset
   indexing *after* `env.step()` returned — but `on_kick`/`on_tackle`
   callbacks fire *synchronously inside* `env.step()` and themselves insert
   extra rows into the same lists, so the stale count silently wrote each
   player's reward/`reward_components` onto the wrong row whenever a
   kick/tackle coincided with a timed sample.
5. `_pending_reward` (a dict tracking "reward accrued since this player's
   last recorded sample") was declared once outside the per-episode loop
   and never reset at `env.reset()`, so a player whose `on_kick`/`on_tackle`
   callback rarely fires (e.g. a neural-driven trainee with ~0% kick rate)
   silently accumulated reward across *many* episodes, then dumped the
   entire backlog onto a single row the next time a callback happened to
   fire (confirmed: one row carried `reward=1474.952`).

Every one of these took multiple diagnostic passes to find, was invisible
for the overwhelming majority of rows (a misattributed zero looks identical
to a correct zero), and was only ever caught by a human eyeballing an
aggregate statistic and noticing an outlier — not by any test, type system,
or structural invariant in the code. Bug #1's fix itself needed three
iterations before it was correct (see `git log`/conversation history for
`scenario_env.py`'s `_can_score_box_terminal`).

This is not a story about carelessness in any one fix. It's a story about a
codebase shape that makes this *entire class* of bug easy to write and hard
to notice, for anyone — human or agent — working in it. This document is
the plan for removing that shape, not just the individual symptoms.

**Read the whole document before starting any of the three refactors below.**
They interact (Refactor A changes what Refactor B consumes; Refactor C
changes what Refactor B produces), and section 7 gives a recommended
sequencing with reasoning, not just a list.

This document assumes the reader has NOT seen the session that produced it
— all necessary context is included inline. Where "confirmed in real data"
appears, it means the claim was directly verified against actual recorded
`.npz` files during that session, not inferred from reading code.

---

## 1. Executive summary

Three refactors are proposed, in increasing order of scope:

- **Refactor A — Unify the trainee/secondary-player code path in
  `ScenarioEnv`.** Right now `ScenarioEnv.step()` has two parallel,
  independently-written code paths: one hardcoded for "the trainee" (its
  own tracking variables, its own reward call, its own possession scanner),
  and a second, generic-looking but separately-maintained loop for
  "secondary players." Every bug in the list above except #2 traces back to
  these two paths drifting apart. The fix is to make the trainee just the
  first entry in a uniform list of driven players, with exactly one code
  path for reward computation, possession tracking, and terminal-condition
  checks, applied identically regardless of role.

- **Refactor B — Turn `record_demonstrations.py`'s recording loop into a
  small stateful class.** Right now `record_episodes()` is an 800+ line
  function whose state (`_pending_reward`, various counters, the growing
  `rewards`/`reward_components`/`dones` lists) lives entirely in closure
  variables with no enforced lifecycle. Nothing stops a variable from
  silently surviving past the scope it's supposed to be reset at (this is
  exactly what happened to `_pending_reward`). A small object with named,
  independently unit-testable methods makes "what does this state mean and
  when does it reset" a visible, checkable thing instead of a comment.

- **Refactor C — Give recorded rows explicit identity.** Right now a
  recorded row's player identity is conveyed by *position* (trainee row,
  then opponent row, alternating) plus a single `is_trainee` float, and
  episode boundaries are inferred from "exactly 2 consecutive `done=1`
  rows." Every bug in the list above that involves the *recording* format
  (not just the live reward computation) — #3, #4, #5 — is a symptom of
  this: some other code path (a callback, a rebuild) doesn't know about the
  positional convention and silently violates it. Giving every row an
  explicit `episode_id` and `player_id` column removes the need to infer
  either from position or counting, and turns most of today's bugs into
  either impossible states or one-line assertion failures.

Recommended order: **A, then C, then B** (reasoning in §7) — but each
section below is written to stand alone if you only want to do one.

---

## 2. Bug catalog (detailed, with exact locations) — read this first

This section is the evidence base the rest of the document argues from.
Skip to §3 if you already lived through the session that found these.

### 2.1 Bug: `sec_box_terminal` missing the immobile-player gate

**Where:** `src/footballcoach/ai/env/scenario_env.py`, inside
`ScenarioEnv.step()`'s secondary-player loop (originally around what is
now the `sec_box_terminal` computation, just before the
`_compute_phase1_reward_for_player()` call for each secondary player).

**What happened:** Phase 1's trainee-win condition
(`box_terminal = in_opponent_box and trainee_has_possession_now`) and
trainee-loss condition (`opponent_box_terminal`) were both explicitly
gated against an immobile opponent — `opponent_box_terminal` had
`and not getattr(match, "_opponent_is_immobile", False)`, with a comment
explaining an immobile player can only satisfy the geometric condition by
coincidence (spawn position, a collision push), never real play. But the
analogous check computed for a *secondary player's own* reward,
`sec_box_terminal`, was a **third, independently hand-written copy** of the
same "is this player in their scoring box with possession" logic — and it
had no such gate:

```python
sec_box_terminal = (
    sec_in_atk_box
    and match.ball.possessed_by == pid
)
```

Because `sec_box_terminal` (unlike `box_terminal`/`opponent_box_terminal`)
does **not** feed into the real episode-ending `done` flag — it only
controls whether *that player's own* reward call sees
`reached_opponent_box_with_possession=True` — an immobile player sitting
near wherever the ball happened to end up could collect the one-time
`box_possession_terminal` (+2.0) and `speed_bonus` reward **every tick**
for as long as the physics fluke held, instead of once. Confirmed directly
in recorded data: 64 consecutive ticks in one real episode, and a
population-level check (`box` component appearing on immobile-labeled rows)
found this pattern repeated across the dataset.

**Root cause pattern:** the same conceptual check
("did player X reach their scoring box, legitimately") was written three
separate times (`box_terminal`, `opponent_box_terminal`, `sec_box_terminal`),
by three different additions to the codebase at different times, and only
two of the three got the immobile-gate applied when it was discovered to be
necessary. This is Refactor A's core motivating example.

**Current state after fix:** consolidated into one shared static method,
`ScenarioEnv._can_score_box_terminal(player_obj) -> bool`, checking
`player_obj.ai is not None` (a per-player, generic check — not the
match-level `_opponent_is_immobile` flag, which only describes "the"
single opponent and can't generalize past 2 players). All three call sites
(`box_terminal`, `opponent_box_terminal`, `sec_box_terminal`) now call this
one function. `opponent_box_terminal` was also changed to look up the
*actual* ball carrier generically (`match.player_by_id(match.ball.possessed_by)`)
rather than assuming "the opponent" is the only possible non-trainee
carrier — a small step toward N-player correctness, though the surrounding
code (`self.trainee_player_id` vs. a hardcoded `"opponent"` id elsewhere)
still assumes exactly 2 players in many other places. See §4 for why this
consolidation, while correct, is still a patch on top of the deeper
structural issue.

### 2.2 Bug: `debug_value_network.py`'s stale duplicate episode-boundary logic

**Where:** `debug_value_network.py` (gitignored, not part of the package —
see `.gitignore`'s `debug_*.py` pattern), a local function
`_episode_row_ranges(dones, row_pool)` used by three of its diagnostic
functions (`_log_reward_component_breakdown`, `_episode_residual_correlation`,
`_save_worst_episode_match_log`).

**What happened:** `src/footballcoach/ai/bc/dataset.py`'s
`DemonstrationDataset` already has a correct episode-boundary function,
`episode_row_ranges()` / the private `_full_dataset_episode_row_ranges()`,
which correctly requires **exactly 2 consecutive** `done=1` rows to close
an episode boundary (matching `record_demonstrations.py`'s convention of
recording both the trainee's and the opponent's row for every timed
sample, both getting `done=1` on the terminal tick). `debug_value_network.py`
had an **older, local, never-updated copy** of this same logic that treated
*any* `done=1` row as its own boundary — so for every real episode it
produced two boundaries instead of one, doubling the reported episode count
(240,000 vs. the correct 120,000) and, more subtly, computing per-episode
reward-component sums over malformed row ranges.

**Root cause pattern:** the SAME logic existed in two places (the package's
`dataset.py` and the debug script's local copy), and when the package
version was fixed (as part of an earlier fix in this same session, to the
consecutive-done-row collapsing), the debug script's copy was not updated
because nothing connected them. This is a duplication bug, structurally
identical in shape to §2.1 even though it lives in unrelated code.

**Current state after fix:** the debug script's local duplicate was
deleted; all three call sites now call `ds.episode_row_ranges(row_pool)`
directly. Also fixed in the same pass: these diagnostics were being
computed over `valid_indices()` (which includes both the trainee's rows
*and* a non-immobile secondary player's own rows), but `classify_outcome()`/
outcome labels are inherently trainee-perspective ("win" means the trainee
won) — so a "win" bucket was silently mixing the trainee's winning returns
with a losing opponent's returns from the same episode. Fixed by
restricting these specific diagnostics to `is_trainee==1` rows
(`trainee_valid_idx`), while leaving actual value-net *training* on the
full `valid_indices()` population (correct there, since `self_feat` is
genuinely self-relative and training benefits from both perspectives).

**Why this bug is out of scope for the refactors below:** `debug_value_network.py`
is a standalone, gitignored debugging script, not a package module — it's
explicitly a "don't regress this by hand" risk rather than something that
should gain its own test infrastructure. It's included here purely as
supporting evidence for the "duplicated logic drifts" pattern; none of the
three refactors below touch it directly, though Refactor C (explicit row
identity) would make this entire class of "reimplement boundary detection"
bug structurally impossible, as a side effect.

### 2.3 Bug: `reward_components` additively shared across players

**Where:** `src/footballcoach/ai/env/scenario_env.py`, `ScenarioEnv.step()`,
in the secondary-player results loop:

```python
# Accumulate secondary components into last_reward_components
for _k, _v in _sec_comps.items():
    self.last_reward_components[_k] = self.last_reward_components.get(_k, 0.0) + _v
```

and `src/footballcoach/ai/scripts/record_demonstrations.py`'s main
recording loop, which used this same combined dict for **every** recorded
row that tick:

```python
_comp_row = np.array(
    [env.last_reward_components.get(k, 0.0) for k in _comp_key_order],
    dtype=np.float32,
)
for _offset, _pid in enumerate(_recorded_ids):
    ...
    reward_components[-_i] = _comp_row   # SAME dict/array for every pid
```

**What happened:** `env.last_reward_components` was designed (by its name
and by every other consumer of it — see `ppo_trainer.py`'s
`reward_comps=dict(getattr(env, "last_reward_components", {}))` when
storing the *trainee's own* rollout-buffer transition) to mean "the
trainee's own per-component reward breakdown for this tick." But the
secondary-player loop additively merged every secondary player's own
components into this same dict, and `record_demonstrations.py` then wrote
this one merged dict onto **every** player's recorded row for the tick —
so a trainee's genuine box-possession-terminal win row also carried the
*losing opponent's own* loss-terminal penalty (`box=+2.0` AND `lterm=-2.5`
on the same row), and the opponent's own row carried the trainee's win
bonus. Confirmed directly in real recorded data: two adjacent rows (one
`is_trainee=1`, one `is_trainee=0`) with *identical* `box=2.0, lterm=-2.5`
values on both.

This also silently affected **live PPO self-play training**, not just demo
recording: `ppo_trainer.py`'s rollout buffer stores
`reward_comps=dict(env.last_reward_components)` for the trainee's own
transition — with a neural self-play opponent (a real, common training
configuration), that dict already included the opponent's own components
merged in, before this fix. This was never separately reported as a
symptom during the session (attention was on the recorded demonstration
data), but it is very likely to have been silently corrupting reward-
component diagnostics logged during live training too, for as long as this
code existed.

**Root cause pattern:** a single shared mutable "this tick's stuff" bucket
being written by multiple logical writers (the trainee's own reward call,
then each secondary player's own reward call) and read by multiple logical
readers (the trainee's own row, every secondary player's own row) with no
way for any reader to know whose contribution is whose. This is the
canonical shape of the "state that should be scoped to one player, one
tick, represented instead as shared state" problem named in the retro.

**Current state after fix:** the merge step was removed. Each secondary
player's own components now travel on that player's own
`last_secondary_results[i]["reward_components"]` entry, kept structurally
separate from `self.last_reward_components` (now purely the trainee's own,
matching what every other consumer already assumed it meant).
`record_demonstrations.py` was updated to build a per-tick
`_comp_by_pid = {trainee_id: env.last_reward_components, **{sec["player_id"]: sec["reward_components"] for sec in env.last_secondary_results}}`
and look up each row's own player's components from it.

### 2.4 Bug: row/reward misattribution from stale `n_appended` across a
reentrant `env.step()` call

**Where:** `src/footballcoach/ai/scripts/record_demonstrations.py`,
`record_episodes()`'s main `while not done:` loop.

**What happened (the actual mechanism, traced row-by-row against real
data):**

```python
n_before = len(rewards)
_recorded_ids = _record_now(reward=0.0, done=False)   # appends 2 rows: [trainee, opponent]
n_appended = len(rewards) - n_before                   # = 2, captured NOW
_obs, _reward, done, last_info = env.step()             # <- on_kick/on_tackle fire INSIDE here
...
for _offset, _pid in enumerate(_recorded_ids):
    _i = n_appended - _offset                            # STALE: still 2, even if step() appended more
    rewards[-_i] = ...                                    # negative-offset indexing into a list
                                                           # that may have grown by more than n_appended
                                                           # since n_appended was captured
```

`on_kick`/`on_tackle` are wired as synchronous player callbacks
(`player.on_kick = _make_on_kick(...)`) that fire **during** the physics
tick loop inside `env.step()` — i.e., *during* the call on the line above,
before it returns. Each callback itself calls `_record_now(player_id=pid)`,
appending **one more row** to the same `rewards`/`reward_components`/`dones`
lists the outer loop is about to index into. But `n_appended` was captured
*before* `env.step()` ran and is never refreshed — so if a callback fires
in the same decision interval as the timed sample (confirmed common: an
immobile player can passively receive the ball via the engine's generic
proximity-based pickup, `Match._update_loose_ball_pickup`, which has *no*
AI check at all — see §2.1's related finding — and the trainee then
legitimately tackles them to reclaim it, firing `on_tackle`), the
negative-offset backfill silently writes into the **wrong absolute rows**:
whichever 2 rows happen to be last at the time of the backfill, which — if
1 extra row was inserted — are actually [the opponent's own timed-sample
row, the injected callback row], not [the trainee's own timed-sample row,
the opponent's own timed-sample row]. The true trainee timed-sample row
(now 3rd-from-last, never targeted by the loop) is left at its untouched
placeholder value.

Traced against real data: rows `[trainee(timed, never backfilled, stuck at
reward=0), opponent(timed, WRONGLY received the trainee's real win
reward/box bonus), <injected on_kick row belonging to the trainee, WRONGLY
received the opponent's real loss penalty>]`.

This bug is invisible for the overwhelming majority of ticks (most timed
samples carry `reward=0`, so a misattributed 0 looks identical to a correct
0) and was only ever caught because it happened to coincide with a
terminal (win/loss) tick, producing a visually obvious, nonsensical
combination (a trainee's "win" episode also containing a `-2.5` loss
penalty on its own row).

**Root cause pattern:** reentrancy — a callback invoked *during* a function
call mutates the same shared, growing data structure the caller is about
to index into with an assumption (a captured length) that was only valid
*before* the call. This is exactly the shape of bug that's easy to miss
when reading either the outer loop or the callback in isolation — you have
to hold both in your head simultaneously, across the reentrant boundary,
to see it.

**Current state after fix:** the outer loop now captures each timed-sample
row's *absolute* index immediately after `_record_now()` returns (before
`env.step()` can insert anything else):
`_recorded_row_indices = list(range(n_before, n_before + len(_recorded_ids)))`,
and backfills by `zip(_recorded_ids, _recorded_row_indices)` — immune to
however many extra rows a callback inserts afterward, because it no longer
relies on "how many rows total" at all.

### 2.5 Bug: `_pending_reward` leaking across episode boundaries

**Where:** `src/footballcoach/ai/scripts/record_demonstrations.py`,
`record_episodes()`. `_pending_reward: dict[str, float] = {}` is declared
**once**, outside the `for ep in range(n_episodes):` loop. Every timed
sample tick accrues into it unconditionally
(`_pending_reward[_pid] = _pending_reward.get(_pid, 0.0) + _r`), and it is
only *drained* for a given player id when an `on_kick`/`on_tackle` callback
fires for that specific player
(`pid_reward = _pending_reward.pop(pid, 0.0) if reward is None else reward`).

**What happened:** `_ep_counts` and `_ep_poss_reward` (two other
per-episode accumulators in the same function) **are** correctly reset at
the top of each episode's setup block, right after `env.reset()`. Nothing
analogous ever reset `_pending_reward`. Semantically, `_pending_reward[pid]`
is meant to represent "reward accrued for this player since their last
recorded sample" — a quantity that can never legitimately span an episode
boundary. But because it was never cleared, a player whose callback rarely
fires (the neural-driven trainee in the dataset that surfaced this had a
**0.00% kick rate** — the trained policy dribbles into the box rather than
kicking — and tackles happen only ~0.7% of ticks) would silently accumulate
reward across *many consecutive episodes* without ever being drained, and
then dump the *entire cross-episode backlog* onto a single row the next
time a callback happened to fire for that player, however many episodes
later that turned out to be. Confirmed directly in real data: one row
carried `reward=1474.952` — several hundred episodes' worth of undiminished
reward flushed onto one row — which `DemonstrationDataset.compute_returns()`'s
backward MC scan (correctly, given its input) then propagated across that
entire episode's `mc1_returns`, producing "episode total reward" statistics
in the hundreds to +1479 for episodes whose real, per-component rewards
were all normal single-digit values.

**Root cause pattern:** a stateful accumulator declared in a wider scope
than its semantics require, with no lifecycle enforcement — the *code*
allowed it to survive `env.reset()`, and nothing (no test, no assertion, no
type-level scoping) said it shouldn't. This is a different flavor of the
same theme as §2.3: state that conceptually belongs to a narrower scope
(one episode) drifting because the language/data-structure gives it a
wider one (the whole recording run) by default, and nothing pushes back.

**Current state after fix:** `_pending_reward.clear()` added to the
per-episode setup block, alongside the other two per-episode resets that
were already there.

**Directly measured severity by driver type:** a stress test with
`Phase1RulesAI` on *both* sides (which kicks/tackles far more than the
neural-driven trainee) found this bug firing in **18 of 300 episodes (6%)**
— i.e., this bug was *more* common with rules-based recording than with
the neural-driven recording where it happened to be first noticed (roughly
1 in 2500 episodes there). This directly answers a question raised during
the session ("was this also happening with the rules-based AI?") — yes,
worse.

### 2.6 Cross-cutting observation: no end-to-end sanity check existed

Every pre-existing reward test (`tests/ai_unit/test_reward.py`) unit-tests
`phase1_reward()` in complete isolation, one call at a time, checking
individual component math (clamping, sign, magnitude for a single
synthetic input). **No test exercised the actual recording pipeline
(`record_episodes()`) end-to-end and checked that a real recorded episode's
aggregate reward stayed within a sane bound.** This is why bugs #2.3–#2.5
shipped into real recorded data and were only caught by a human looking at
aggregate statistics from a training-diagnostic script, rather than by CI.

A first version of this class of test now exists:
`tests/ai_scenario/test_record_demonstrations_reward_sanity.py` (runs 300
real episodes through `record_episodes()` with `Phase1RulesAI` on both
sides — chosen specifically because it stresses the callback-interleaving
code path much harder than a neural-driven trainee — and asserts no
single-tick reward exceeds a generous bound, and no player's per-episode
undiscounted total exceeds a generous bound). It was verified to fail
immediately against bug #2.5 (caught 18/300 corrupted episodes) but was
also verified to **not** catch bug #2.4 on its own — a misattributed value
just moves to a different row, it doesn't become abnormally large — so a
second, complementary test
(`tests/ai_scenario/test_record_demonstrations_row_alignment.py`) exists
specifically to check *row attribution*, not magnitude. Both test styles
are necessary; neither subsumes the other. Any of the refactors below
should preserve or subsume both test intents.

---

## 3. Root cause / shared theme (do not skip this before implementing)

Read narrowly, sections 2.1–2.5 look like five unrelated bugs in five
different functions. They are not. Every one of them is caused by the same
underlying design choice, applied in five different places:

> **State that is conceptually scoped to "one player, at one tick, within
> one episode" is represented in the code as shared, positional, or
> unscoped state instead of state that is explicitly addressed by
> (episode, tick, player).**

Concretely, the codebase currently relies on the reader/maintainer to hold
several *implicit* invariants in their head, none of which are enforced by
the data structures themselves:

1. "`env.last_reward_components` means the trainee's own components" — true
   only because nothing else writes to it, until something did (§2.3).
2. "Rows come in a fixed [trainee, opponent] pair for every timed sample,
   append-ordered" — true only until a callback appends a third row
   in between the pair being recorded and the pair being backfilled (§2.4).
3. "`_pending_reward[pid]` means 'since this player's last sample'" — true
   only if something resets it at the right boundary, which nothing did
   (§2.5).
4. "Exactly 2 consecutive `done=1` rows mark one episode boundary" — true
   of the *current* recording convention, but re-derived from scratch (and
   incorrectly) in a second location (§2.2).
5. "The scoring-box-terminal condition for player X is gated the same way
   regardless of which of three call sites is asking" — true only by
   convention, because it was written three separate times (§2.1).

None of these are hard to state as invariants. All of them are hard to
**violate accidentally** if the data structures make the invariant true by
construction rather than true by convention. That reframing is the design
principle behind all three refactors below:

- **Refactor A** makes "the trainee is just player 0" true by construction
  (one code path, one data structure per player, applied uniformly) instead
  of true by two independently-maintained code paths agreeing to behave the
  same way.
- **Refactor B** makes "this accumulator is scoped to one episode" (or
  "one tick", as appropriate) visible and enforceable in a class's method
  boundaries, instead of implicit in a closure variable's declaration
  point.
- **Refactor C** makes "which player, which episode, this row belongs to"
  an explicit, directly-readable column instead of something inferred from
  position and counting `done=1` rows.

None of the three refactors, alone, would have prevented *every* bug in
§2 — but each would have made its corresponding bug(s) either structurally
impossible or immediately, loudly detectable (an assertion failure at
write time, not a silent corruption discovered weeks later). That is the
bar this document is aiming for: not "fix the bugs we found," which is
already done, but "make this shape of bug impossible to write without
noticing."

---

## 4. Refactor A — Unify the trainee/secondary-player code path

### 4.1 Current state (exact inventory)

`src/footballcoach/ai/env/scenario_env.py`'s `ScenarioEnv` currently
maintains **two parallel sets of per-player state and code paths**:

**Trainee-specific (singular, hardcoded name/field per concept):**
- `self.trainee_player_id: str` — the one distinguished player id.
- `self.last_trainee_transition` — set from
  `player.ai.last_transition` (only populated when the trainee's `.ai` is
  a `NeuralPlayerAI`).
- `self._trainee_had_possession_last_step: bool`,
  `self._trainee_pending_loss: bool` — possession-transition tracking
  state, trainee-only.
- `self._trainee_start_stamina: float`.
- `self._trainee_cumulative_state: dict` — passed into
  `_compute_phase1_reward_for_player()`'s `cumulative_state` param (for
  `cumulative_clamped_delta()`-based reward terms like `prog`/`appr_sq`).
- `self.last_reward_components: dict[str, float]` — (after the §2.3 fix)
  purely the trainee's own per-component reward breakdown for the last
  tick.
- In `step()`: a dedicated, ~150-line block (roughly lines 337–577 as of
  this session) computing `prev_ball_dist`, `prev_box_dist`,
  `trainee_prog_accum`, the tick-loop's `_trainee_poss_prev`/
  `_trainee_pending_loss`/`trainee_gained_count`/`trainee_lost_count`
  scanning, `box_terminal`, `opponent_box_terminal`, and finally one call
  to `_compute_phase1_reward_for_player()` whose result becomes the
  function's own `reward`/`self.last_reward_components` return values.
- In `reset()`: `trainee.ai = NeuralPlayerAI(...)` assigned unconditionally
  whenever `self.sample_action_fn is not None` — no equivalent "should this
  player be neural-driven" decision exists for secondary players in the
  same unconditional way (see below, it's gated by
  `is_rules_episode`/`is_immobile_episode`).

**Secondary-player-specific (generic-*looking*, list/dict-keyed, but a
separately maintained second implementation of almost the same logic):**
- `self.secondary_player_ids: list[str]` — arbitrary-length, in principle
  N-player-ready.
- `self.last_secondary_results: list[dict]` — one dict per secondary
  player per tick, each independently carrying `obs`, `action`, `reward`,
  `done`, and (after the §2.3 fix) its own `reward_components`.
- `self._sec_had_possession_last_step: dict[pid, bool]`,
  `self._sec_pending_loss: dict[pid, bool]`,
  `self._sec_last_ball_dist: dict[pid, float]`,
  `self._sec_start_stamina: dict[pid, float]`,
  `self._sec_cumulative_state: dict[pid, dict]`,
  `self._sec_ema: dict[pid, EMAFilter]` — the same concepts as the
  trainee-only fields above, but dict-keyed by player id, maintained as a
  **second, independent implementation** of the same tracking logic (the
  tick-loop possession scan is called via
  `self._possession_transition_step(pid, ...)` for secondary players in a
  `for pid in sec_pre:` loop that runs the *same* shared helper the
  trainee's own scan calls — so the possession-transition math itself is
  already correctly shared; it is the *surrounding bookkeeping* — which
  variable holds the result, when it gets read, what gates the early-exit
  — that is duplicated).
- In `step()`: a second block (`for pid, pre in sec_pre.items(): ...`)
  re-deriving `sec_curr_ball_dist`, `_sec_box_dist_now`, `sec_ball_prog`,
  `sec_in_atk_box`, `sec_box_terminal`, and calling
  `_compute_phase1_reward_for_player()` again per secondary player.
- In `reset()`: `sec_player.ai = NeuralPlayerAI(...)` is assigned only
  conditionally — `if not is_rules_episode and not is_immobile_episode:` —
  a decision the trainee's own unconditional assignment doesn't make in the
  same way.

**What genuinely is shared (already correctly factored, do not
re-duplicate this when doing Refactor A):**
- `_compute_phase1_reward_for_player()` itself — the single reward-math
  function, taking a `player_id`/`player_obj` and all the per-player scalar
  inputs, called once per player (trainee once, each secondary player
  once). This is the right shape and should be the *model* for how the
  rest of the per-player state should be organized, not something that
  needs changing.
- `_player_speed_and_heading_cos()` — a `@staticmethod` shared helper, with
  a docstring explicitly warning "do not inline this logic at a second call
  site... having two separate inline copies of this is exactly how the
  heading/appr_sq/stamina terms silently went missing for secondary players
  previously" — i.e., this exact class of bug has already bitten this
  codebase once before, in a different field, and was fixed the same way
  Refactor A proposes generalizing.
- `_can_score_box_terminal()` (after the §2.1 fix) — the single
  box-terminal gate, called from all three call sites.
- `_possession_transition_step()` — the single per-tick possession-scan
  step function, called once per player (trainee + each secondary) inside
  the tick loop.

So the shared, correctly-factored pieces are the *math*. What's duplicated
is the *bookkeeping around* the math: where results are stored, what
triggers a fresh computation, what the early-exit conditions are, and
(critically, per §2.1/§2.3) whether a given safety gate got applied
consistently across both paths.

### 4.2 Why this specific duplication caused §2's bugs

- §2.1 (`sec_box_terminal` missing the gate): a new safety gate was added
  to the trainee's own two terminal checks (`box_terminal`,
  `opponent_box_terminal`) but not propagated to the secondary players'
  version, because there was no single place where "the terminal check for
  a player" lived — there were three.
- §2.3 (`reward_components` merged): `last_reward_components` was
  originally a trainee-only concept; when secondary-player component
  reporting was added, the natural (and wrong) thing to do, given no
  existing per-secondary-player component field existed yet, was to merge
  into the one dict that already existed rather than adding a
  properly-scoped new field — the underlying tension being "there is one
  established field for 'this tick's components' and it was implicitly
  trainee-shaped."
- Indirectly, §2.4/§2.5 (in `record_demonstrations.py`, not
  `scenario_env.py` itself, but downstream consumers of this same
  asymmetry): the recorder has to special-case "the trainee's reward comes
  from `env.step()`'s return value" vs. "each secondary player's reward
  comes from iterating `env.last_secondary_results`" — two different
  access patterns for what should be the same kind of information, which
  is exactly the kind of asymmetry that makes off-by-one/misattribution
  bugs easy to introduce when a third thing (a callback-inserted row) needs
  to be reconciled against both patterns at once.

### 4.3 Proposed design

**Core idea:** replace "the trainee" as a distinguished, separately-coded
entity with "the trainee is `driven_players[0]`" — one list of player
records, one per-player state object, one code path through `step()` that
runs identically for every entry in the list. The trainee is not special
at the data-structure level; it is only special in that external callers
(PPO's rollout buffer, the recorder) currently *choose* to treat entry 0
differently, and that choice should move to the callers, not live inside
`ScenarioEnv`.

**New data structure**, replacing the ~12 parallel
trainee-singular/secondary-dict fields listed in §4.1:

```python
@dataclass
class DrivenPlayerState:
    """All per-player state ScenarioEnv.step() tracks for ONE driven
    player, trainee or secondary alike. One instance per entry in
    ScenarioEnv.driven_player_ids, created in reset(), mutated in step().
    Replaces the ~12 separately-named trainee-singular / secondary-dict
    fields this dataclass's docstring enumerates for historical grep-
    ability: last_trainee_transition/last_secondary_results,
    _trainee_had_possession_last_step/_sec_had_possession_last_step,
    _trainee_pending_loss/_sec_pending_loss, _trainee_start_stamina/
    _sec_start_stamina, _trainee_cumulative_state/_sec_cumulative_state,
    _sec_last_ball_dist (trainee's own equivalent was recomputed inline
    from self._last_ball_dist -- also unified here), _sec_ema (trainee's
    own EMA, self._ema, stays match-level/shared -- see note below).
    """
    player_id: str
    is_trainee: bool                     # role flag ONLY -- see 4.3.1
    had_possession_last_step: bool = False
    pending_loss: bool = False
    start_stamina: float = 1.0
    cumulative_state: dict[str, float] = field(default_factory=dict)
    last_ball_dist: float = 0.0
    last_transition: dict | None = None  # from player.ai.last_transition, if any
    last_reward: float = 0.0
    last_reward_components: dict[str, float] = field(default_factory=dict)
    last_done: bool = False
```

`ScenarioEnv` gains:

```python
self.driven_player_ids: list[str]        # trainee_player_id is ALWAYS index 0
self._player_state: dict[str, DrivenPlayerState]   # keyed by player_id
```

`self.trainee_player_id` **stays** (as a plain `str` property/attribute) —
external callers still need to know which id is "the" trainee for their
own purposes (PPO trains only the trainee's policy gradient off entry 0;
the recorder still wants to know which player is being demonstrated). What
changes is that `ScenarioEnv` itself stops branching its *internal*
computation based on that distinction.

`self.secondary_player_ids` becomes a derived property:
`[pid for pid in self.driven_player_ids if pid != self.trainee_player_id]`
— kept for backward compatibility with any external code still reading it
(grep before removing; as of this session, `curriculum/envs.py`'s
`_build_phase1_env()` passes `secondary_player_ids=["opponent"]` into
`ScenarioEnv.__init__`, so the constructor's public API should probably
keep accepting `trainee_player_id` + `secondary_player_ids` separately —
internally immediately compute `driven_player_ids = [trainee_player_id] +
secondary_player_ids`, and construct `_player_state` from that unified
list. This keeps the refactor internal to `ScenarioEnv`; no caller-visible
signature change is required for `__init__`).

#### 4.3.1 `is_trainee` is a role flag, not a behavior switch

This is the crux of the refactor, worth stating explicitly: after this
change, **nothing in `step()`'s reward/possession/terminal computation
should ever read `player_state.is_trainee` to decide what to compute.** It
exists purely so that external code (and the final assembly of
`StepInfo`/return values, see 4.3.4) can find "which one was the trainee"
after the fact. If you find yourself writing
`if player_state.is_trainee: ... else: ...` inside the per-player loop
described in 4.3.2, that is a sign the refactor is being subverted back
into the two-path shape it's meant to remove. The only legitimate
differences between the trainee and secondary players are:

- Whether external callers (PPO, the recorder) choose to store/use the
  result differently — happens *outside* `ScenarioEnv`.
- The `NeuralPlayerAI` assignment condition in `reset()` (see 4.3.5) —
  this is a genuine, currently-real asymmetry (the trainee gets a
  `NeuralPlayerAI` unconditionally when `sample_action_fn` is set; a
  secondary player only does when the episode isn't rules/immobile) that
  reflects an actual product decision (opponents in phase-1 demo recording
  are rules-based or immobile, never neural — see
  `record_demonstrations.py`'s "no neural opponent during demo recording"
  comment), not an accidental duplication. This asymmetry is fine to keep,
  but it should be the *only* remaining place role matters, and it should
  be clearly commented as intentional (which, to be fair, it already is).

#### 4.3.2 Unified `step()` body

Replace the two separate blocks (trainee's ~150 lines, secondary players'
`for pid, pre in sec_pre.items():` loop) with **one loop over
`self.driven_player_ids`**, run twice conceptually per tick (once to
snapshot pre-tick state before the physics loop, once after):

```python
# --- Snapshot pre-tick state for every driven player uniformly ---
_pre: dict[str, dict] = {}
for pid in self.driven_player_ids:
    ps = self._player_state[pid]
    _pre[pid] = {
        "prev_ball_dist": ps.last_ball_dist,
        "prev_box_dist": self._ball_dist_to_opponent_box(pid),
    }

# ... physics tick loop (self._loop.step() N times) -- the possession-
# transition scan already calls the shared _possession_transition_step()
# per player; keep that loop exactly as-is, just iterate
# self.driven_player_ids instead of "trainee once, then sec_pre.items()".
# The early-exit condition (currently trainee-only: "break if the trainee
# already has the ball in the opponent's box") needs a decision here --
# see 4.3.3.

# --- Compute reward for every driven player uniformly, AFTER the tick loop ---
results: dict[str, PlayerStepResult] = {}
for pid in self.driven_player_ids:
    ps = self._player_state[pid]
    player_obj = match.player_by_id(pid)
    in_own_box = self._can_score_box_terminal(player_obj) and self._in_own_attacking_box(player_obj, match)
    ... # ball_dist, box_dist, progress accum -- all read from ps, not from two different variable sets
    reward, comps, ps.cumulative_state = self._compute_phase1_reward_for_player(
        player_id=pid, player_obj=player_obj, ...,
        reached_opponent_box_with_possession=in_own_box,
        opponent_reached_trainee_box=<see 4.3.3>,
        ...
    )
    ps.last_reward = reward
    ps.last_reward_components = comps
    ps.last_done = done
    results[pid] = PlayerStepResult(player_id=pid, reward=reward, reward_components=comps, done=done, obs=..., transition=ps.last_transition)
```

`PlayerStepResult` is a small dataclass (or reuse `DrivenPlayerState`'s
public fields directly) representing "everything a caller might want to
know about one player's outcome this tick" — this becomes the *one*
shape both the trainee and every secondary player's result is returned in.

#### 4.3.3 The two remaining genuinely-asymmetric conditions

Two things in the current code are not simply duplicated — they represent
a real "one player is special" condition that has to be resolved, not
just uniformly generalized:

1. **The early-exit break** (`if trainee_has_possession_now and
   trainee-in-box: break`) — this exists only for the trainee's own tick
   loop, as an optimization ("any further ticks only accumulate spurious
   negative progress"). Under the unified loop, this becomes: break the
   physics tick loop as soon as **any** driven player who has already won
   (reached their own scoring box with possession) is detected — i.e. the
   condition generalizes naturally to "break when any player satisfies
   their own win condition," not just player 0. This is very likely a
   **behavior improvement**, not just a refactor — currently a secondary
   player reaching their own box mid-tick-loop does *not* trigger this
   early exit (only the trainee does), which is itself a minor inconsistency
   worth fixing as part of this work. Flag this as a deliberate,
   documented behavior change when it lands (see §4.6), not a silent
   side effect.

2. **`opponent_reached_trainee_box`** (the "did I just lose" flag passed
   into every player's own reward call) — semantically, for the trainee,
   this means "did a non-trainee player reach the trainee's box." For a
   secondary player, per the existing code's own comment
   (`# from sec's POV, trainee winning = sec losing`), this means "did the
   trainee reach the trainee's own scoring target (the secondary player's
   defending box)." In a strictly 2-player game these are two names for
   the same physical event viewed from each side, but in the N-player case
   this genuinely requires knowing, for player X, "who is the *specific
   other* player whose win is X's loss" — which in a free-for-all
   N-player scenario might not even be well-defined (whose win costs whom,
   with 3+ competing players?). **This is explicitly out of scope for
   Refactor A** — Refactor A should preserve today's exact 2-player
   semantics (compute it generically as "did the CURRENT ball carrier,
   if not player X, satisfy their own win condition" — which is exactly
   what the current `opponent_box_terminal`/`_carrier_can_score` lookup
   already generalizes to, per the §2.1 fix) and leave the "what does
   losing mean with 3+ players" product question for whoever actually
   builds the first N>2 scenario, flagged clearly as a follow-up (§4.7).

#### 4.3.4 What `ScenarioEnv.step()` returns

Currently: `(observation, reward, done, info)` where `reward`/the
`ObservationBatch` describe *only* the trainee, and secondary-player
results are a side channel (`self.last_secondary_results`,
`self.last_reward_components`) callers must separately read after the
call. **Keep this external return signature unchanged** — this is a
public API multiple callers (`ppo_trainer.py`'s rollout loop,
`record_demonstrations.py`) depend on, and changing it is out of scope for
Refactor A (it's a much larger, riskier change for no correctness benefit —
see §4.6's "what NOT to change" list). What changes is only what happens
*internally* to produce that same external shape:

```python
def step(self):
    ...  # unified per-player loop as in 4.3.2, populates results: dict[str, PlayerStepResult]
    trainee_result = results[self.trainee_player_id]
    self.last_trainee_transition = trainee_result.transition
    self.last_reward_components = trainee_result.reward_components
    self.last_secondary_results = [
        {**results[pid].transition_dict(), "player_id": pid, "reward": results[pid].reward,
         "done": 1.0 if done else 0.0, "reward_components": results[pid].reward_components}
        for pid in self.secondary_player_ids
    ]
    return obs, trainee_result.reward, done, info
```

i.e., the public-facing fields (`last_trainee_transition`,
`last_reward_components`, `last_secondary_results`, the 4-tuple return)
all still exist, with the exact same shapes and meanings external code
already relies on — they are now just *views* assembled from the one
unified `results` dict at the end of `step()`, rather than being computed
by two separate code paths throughout. **No changes required to
`ppo_trainer.py` or `record_demonstrations.py` for this part** — this is
the load-bearing design constraint that makes Refactor A safe to do
independently of Refactor B/C.

#### 4.3.5 `reset()`

Same structure: replace the two separate blocks (unconditional
`trainee.ai = NeuralPlayerAI(...)`, conditional
`for pid in secondary_player_ids: sec_player.ai = NeuralPlayerAI(...)`)
with one loop over `driven_player_ids`, special-casing only the condition
from §4.3.1's second bullet:

```python
if self.sample_action_fn is not None:
    for pid in self.driven_player_ids:
        is_trainee = (pid == self.trainee_player_id)
        if is_trainee or (not is_rules_episode and not is_immobile_episode):
            player = match.player_by_id(pid)
            player.ai = NeuralPlayerAI(self.sample_action_fn, ...)
```

Also initialize `self._player_state = {pid: DrivenPlayerState(player_id=pid,
is_trainee=(pid == self.trainee_player_id)) for pid in self.driven_player_ids}`
here, replacing the dozen separate field initializations currently spread
across `reset()`.

### 4.4 Consumers to verify (not necessarily change) after this refactor

Because §4.3.4 preserves the external return shape exactly, the following
should need **zero changes** — but must be re-tested, since they are the
actual behavior contract this refactor must not break:

- `src/footballcoach/ai/ppo/ppo_trainer.py` — reads
  `env.last_trainee_transition`, `env.last_reward_components`,
  `env.last_secondary_results`, and `env.step()`'s 4-tuple. All three
  call sites identified in this session (`buffer.add(reward_comps=dict(
  getattr(env, "last_reward_components", {})), ...)` for the trainee's own
  transition, `for sec in getattr(env, "last_secondary_results", []):
  buffer.add(...)` for secondary transitions, and the rollout/episode
  component-accumulation diagnostics reading
  `env.last_reward_components.items()`) should behave identically,
  benefiting from the same correctness fix §2.3 already gave them (trainee's
  own stored `reward_comps` no longer includes a self-play opponent's
  components merged in).
- `src/footballcoach/ai/scripts/record_demonstrations.py` — same
  external-field reads. Should require zero changes from Refactor A alone
  (Refactor B is a separate, independent rewrite of this file).
- Any test in `tests/ai_unit/test_scenario_env_immobile_opponent_terminal.py`
  or `tests/ai_scenario/` that constructs a `ScenarioEnv` and reads
  `env.last_reward_components`/`env.last_secondary_results` directly
  (several were added this session specifically to test §2.1/§2.3's
  fixes) — these should keep passing unchanged, since they assert on the
  external shape, not on internal structure. Re-run the full suite after
  this refactor as the primary verification, not just new tests.

### 4.5 Migration plan (step-by-step)

1. Add the `DrivenPlayerState` dataclass and `PlayerStepResult` dataclass
   to `scenario_env.py` (additive, no behavior change yet).
2. Add `self._player_state` construction to `reset()`, initialized
   alongside (not yet replacing) the existing trainee-singular/secondary-dict
   fields. Populate both in parallel for one commit, so a debug assertion
   can cross-check they agree (`assert ps.had_possession_last_step ==
   self._trainee_had_possession_last_step` for the trainee entry, etc.) —
   this gives you a correctness oracle for free during the transition
   instead of trusting the refactor blind.
3. Rewrite the tick-loop possession scan to iterate
   `self.driven_player_ids` uniformly, writing into `_player_state`, while
   the cross-check assertions from step 2 remain active. Run the full test
   suite; if the cross-check assertions ever fire, that's a genuine
   behavior difference to resolve *before* proceeding, not paper over.
4. Rewrite the post-tick-loop reward computation as the single loop in
   §4.3.2, producing `results: dict[str, PlayerStepResult]`. At this point
   the OLD trainee-singular/secondary-dict code paths become provably dead
   (nothing reads them except the cross-check assertions) — delete them,
   and delete the cross-check assertions.
5. Rewrite the end-of-`step()` assembly per §4.3.4 to build
   `last_trainee_transition`/`last_reward_components`/`last_secondary_results`
   from `results`.
6. Resolve the early-exit-break decision from §4.3.3 point 1 as its own,
   separately-flagged behavior change (a one-line diff, but call it out
   explicitly in the PR/commit description — it changes when episodes with
   3+ players, once they exist, would terminate early; in the current
   2-player-only reality it only changes behavior in the (currently
   impossible, since there's only one non-trainee player and the trainee's
   own check already existed) case of a *secondary* player reaching their
   own win condition mid-tick-loop before the trainee's own check would —
   verify this is genuinely a no-op today before merging, e.g. via a
   targeted test that forces both a trainee-win and secondary-win
   condition to be geometrically possible in the same tick and asserts the
   outcome is unchanged from before this refactor).
7. Full test suite, plus the two new reward-sanity/row-alignment tests
   from §2.6, plus a fresh multi-thousand-episode `record_episodes()` run
   (mirroring what this session's investigation did manually) to
   statistically re-verify no regression in the rare-event bugs this whole
   document is about — a green test suite alone is not suf­ficient
   confidence given how rare (§2.1: 1/2500 episodes, §2.5: 1/2500 to 1/17
   depending on driver type) these bugs are; a passing unit test suite
   would not have caught most of them the first time either, only the new
   sanity/row-alignment tests specifically targeting these mechanisms
   would, and even those were only added *after* the bugs were found by
   volume-based manual inspection.

### 4.6 Explicitly NOT in scope for Refactor A

- Changing `ScenarioEnv.step()`'s external 4-tuple return signature, or
  the existence/shape of `last_trainee_transition`/`last_reward_components`/
  `last_secondary_results` as public fields. This is a compatibility
  boundary; changing it is a much larger, separate migration touching
  `ppo_trainer.py` and `record_demonstrations.py` directly, with no
  correctness benefit over keeping the current shape as a thin
  assembly step (§4.3.4) — do not conflate the two.
- Solving the N>2-player `opponent_reached_trainee_box` semantics question
  (§4.3.3 point 2) — explicitly deferred.
- Any change to `_compute_phase1_reward_for_player()`'s own signature or
  math — it is already correctly shared and is not part of this
  refactor's problem statement.
- Any change to `record_demonstrations.py` (that's Refactor B) or the
  `.npz` dataset schema (that's Refactor C) — Refactor A is scoped
  entirely to `scenario_env.py`.

### 4.7 Follow-ups this refactor sets up but does not itself resolve

- Once `driven_player_ids` is a real, uniform list internally, extending
  `ScenarioEnv` to actually support 3+ players (a currently-untested,
  partially-unsupported configuration per the "hardcoded to 2 players"
  notes already in `ai/knowledge.md`) becomes primarily a question of (a)
  resolving §4.3.3 point 2's semantics and (b) Refactor C's dataset-format
  work (since `is_trainee: bool` alone cannot distinguish 2+ secondary
  players from each other in the recorded data today) — not a further
  `scenario_env.py` rewrite. Flag this explicitly to whoever picks up
  N-player support next: the hard part will have already been done here.

---

## 5. Refactor B — Class-based recorder for `record_demonstrations.py`

### 5.1 Current state (exact inventory)

`src/footballcoach/ai/scripts/record_demonstrations.py`'s `record_episodes()`
function is, as of this session, roughly 480 lines (from its `def` to its
`return {...}`), containing:

- **Growing output lists**, appended to throughout: `self_feats`,
  `other_feats`, `exists_masks`, `ball_feats`, `global_feats`, `bc_labels`,
  `rewards`, `dones`, `reward_components`, `is_trainee_flags`.
- **Per-run counters/accumulators**, declared once at function top, mutated
  throughout: `steps_total`, `steps_valid`, `_kick_count_since_log`,
  `_tackle_count_since_log`, `_kick_count_total`, `_tackle_count_total`,
  `_comp_acc` (dict), `_comp_acc_episodes`.
- **Per-episode accumulators**, declared once at function top, *intended*
  to be reset per episode (some correctly are, one — `_pending_reward` —
  was not, per §2.5): `_ep_counts` (nested dict, correctly reset),
  `_ep_poss_reward` (dict, correctly reset), `_pending_reward` (dict, was
  NOT reset — now fixed to be reset, but the *pattern* that allowed this
  bug — an accumulator dict declared in the same scope as the correctly-
  reset ones, with no structural distinction between "per-run" and
  "per-episode" scope — remains).
- **Three nested closures** capturing most of the above by reference:
  `_record_now(reward, done, player_id)` (the core row-appending function,
  ~45 lines), `_make_on_kick(role, pid)`/`_make_on_tackle(role, pid)`
  (callback factories, each returning a further closure that calls
  `_record_now`), `_make_on_tackle_result(role)`/
  `_make_on_auto_tackle_result(role)` (stats-only callbacks).
- **The main `for ep in range(n_episodes):` loop**, itself containing a
  `while not done:` inner loop, with the row-backfill logic from §2.4/§2.5
  living inside it.
- **A large final `return {...}` dict**, assembling ~15 keys from the
  accumulated lists via `np.stack`/`np.array` calls.

Every one of these — the growing lists, the counters, the closures — is a
plain local variable or nested function, with **no object boundary**
anywhere. "What is `_pending_reward`'s scope" is a question you can only
answer by reading the whole function and noticing (or, as happened, not
noticing) where it is and isn't touched.

### 5.2 Why this shape caused §2.4/§2.5, and why it will keep causing similar bugs

- §2.5 happened because nothing distinguished "per-episode state" from
  "per-run state" as a *category* — they're both just "a variable declared
  near the top of this long function." A reviewer (or an agent) reading
  the per-episode reset block (`for k in _ep_poss_reward: _ep_poss_reward[k]
  = 0.0`) has no structural cue that `_pending_reward`, declared 8 lines
  earlier, needed the same treatment and doesn't have it — the two
  accumulators look identical in the code, differing only in whether a
  reset call happens to exist for them.
- §2.4 happened because the row-backfill logic has to reason about "how
  many rows were appended, by which mechanism, since when" — a question
  that's only answerable by carefully tracing `_record_now`'s several call
  sites (the main loop's `player_id=None` call, and the three
  callback-triggered `player_id=pid` calls) and their interaction, spread
  across the whole function. There is no single place that owns "the list
  of rows and what's been backfilled so far" as a concept with its own
  invariants.
- More generally: **a function this size, with this much mutable closure
  state, is hard to unit test in isolation.** Every test of this logic
  today (the two added in §2.6) has to construct a real `ScenarioEnv`, run
  real (or monkeypatched) physics, and inspect the *entire returned
  dataset* to infer whether the internal bookkeeping was correct — there
  is no way to, say, directly unit-test "does draining `_pending_reward`
  via a callback correctly zero it out" without running an entire episode
  end-to-end. That is a testability problem as much as a correctness one:
  even the tests written to catch these exact bugs (§2.6) are expensive,
  slow (8+ seconds each, 300-episode runs for the sanity test), and
  necessarily indirect.

### 5.3 Proposed design

**Core idea:** extract the recording state and logic into a class,
`DemonstrationRecorder`, whose methods have narrow, individually-testable
responsibilities and whose instance attributes have explicit, enforced
lifecycles (constructor = per-run scope, an explicit `start_episode()`
method = per-episode scope, `record_row()`/`record_callback()` = per-tick
scope). This does not need to be a large rewrite of the *logic* — the
reward/component computation, the sampling cadence, the multiprocessing
job-splitting are all fine as-is — it is specifically a rewrite of the
*state-management shape* so that scope violations like §2.5 become
impossible to write without visibly breaking the class's contract.

```python
@dataclass
class DemonstrationRecorder:
    """Owns all state for recording ONE worker's share of episodes.
    Per-run state lives in __init__ (self.*_total counters, the growing
    output lists). Per-episode state is explicitly scoped by
    start_episode()/finish_episode() -- nothing declared there survives
    a start_episode() call by construction, unlike the old record_episodes()
    closure where _pending_reward accidentally did.
    """
    env: ScenarioEnv
    label_fn: Callable
    comp_key_order: list[str]

    # --- per-run state (set once in __init__, mutated throughout the run) ---
    self_feats: list = field(default_factory=list)
    other_feats: list = field(default_factory=list)
    # ... (the rest of the growing-list fields, unchanged in kind from today)
    steps_total: int = 0
    steps_valid: int = 0
    _comp_acc: dict[str, float] = field(default_factory=dict)
    _comp_acc_episodes: int = 0

    # --- per-episode state -- ONLY ever mutated between start_episode()
    # and finish_episode(); start_episode() unconditionally resets ALL of
    # these, so nothing can silently survive across an episode boundary
    # the way _pending_reward did. ---
    _pending_reward: dict[str, float] = field(default_factory=dict, init=False)
    _ep_counts: dict = field(default_factory=dict, init=False)
    _ep_poss_reward: dict = field(default_factory=dict, init=False)

    def start_episode(self) -> None:
        """MUST be called once per episode, immediately after env.reset().
        Resets every per-episode accumulator unconditionally -- this
        single method is now the ONE place that has to be correct for
        the whole class of bug in section 2.5 of reward_fixes.md to be
        prevented, instead of that correctness being spread across
        however many accumulators happen to exist, each needing its own
        manually-remembered reset line."""
        self._pending_reward = {}
        self._ep_counts = {role: {k: 0 for k in _ACTION_COUNT_KEYS} for role in _ROLES}
        self._ep_poss_reward = {"poss": 0.0, "lpos": 0.0}

    def record_timed_sample(self) -> list[str]:
        """Equivalent of today's _record_now(reward=0.0, done=False,
        player_id=None). Appends one row per driven player (see Refactor
        A -- this can iterate env.driven_player_ids uniformly instead of
        hardcoding [trainee_id, "opponent"]), returns the recorded ids
        AND their absolute row indices together, as one list of (pid,
        row_idx) pairs -- eliminating the separate n_appended/
        _recorded_row_indices bookkeeping the old code needed, since the
        return value itself is now the single source of truth for "what
        was just appended and where."
        """
        ...

    def record_callback_sample(self, player_id: str) -> None:
        """Equivalent of today's _record_now(player_id=pid) as called
        from on_kick/on_tackle. Pops self._pending_reward[player_id],
        appends one row. Because this is a named method (not a closure
        factory returning a closure), it can be unit-tested directly:
        construct a recorder, call start_episode(), manually set
        self._pending_reward['trainee']=5.0, call
        record_callback_sample('trainee'), assert the popped reward is
        5.0 and self._pending_reward no longer has the key -- no
        ScenarioEnv or physics simulation required for this test.
        """
        ...

    def backfill_step_results(self, recorded: list[tuple[str, int]], results: dict[str, PlayerStepResult], done: bool) -> None:
        """Takes the (pid, row_idx) pairs record_timed_sample() just
        returned and the PlayerStepResult dict env.step() just produced
        (see Refactor A section 4.3.2/4.3.4 -- if Refactor A has NOT
        landed yet, adapt this to read env.last_reward_components /
        env.last_secondary_results directly, as today), and writes each
        row's OWN reward/reward_components/done by absolute index --
        this is the fixed version of section 2.4's bug, now living in
        one named method instead of inline in the main loop, so a test
        can call it directly with a synthetic `recorded` list containing
        extra/reordered entries (simulating a callback insertion) without
        needing to actually trigger a real mid-step() callback.
        """
        ...

    def accrue_pending_reward(self, reward_by_pid: dict[str, float]) -> None:
        """The per-tick accrual step (old code's
        `for _pid, _r in _reward_by_pid.items(): _pending_reward[_pid] += _r`).
        Named and isolated so a test can assert: accrue twice, then
        start_episode(), then assert _pending_reward is empty -- directly
        testing the section 2.5 fix without any episode/physics machinery
        at all.
        """
        ...

    def finish_episode(self, outcome: str) -> None:
        """Rolls _ep_counts/_ep_poss_reward into the per-run history
        lists (episode_action_counts, episode_poss_reward,
        episode_outcomes), matching today's end-of-loop bookkeeping.
        Does NOT reset per-episode state itself -- that's start_episode()'s
        job, kept as a SEPARATE method (not "reset at both start and end")
        specifically so there is exactly one place, not two, that has to
        be remembered/kept correct.
        """
        ...

    def to_dataset_dict(self) -> dict:
        """Equivalent of today's final `return {...}` block."""
        ...
```

The multiprocessing-facing entry points
(`_run_recording_job`, `main()`'s single-process and multi-process paths)
change minimally: instead of calling the free function `record_episodes(...)`,
they construct a `DemonstrationRecorder`, call `.run(n_episodes, ...)` (a
thin method that contains what's left of the old `for ep in range(n_episodes):
... while not done: ...` control flow after the per-tick logic has been
extracted into the named methods above), and call `.to_dataset_dict()` at
the end. `_run_recording_job`'s file-chunking/`.npz`-saving logic is
unaffected either way — it consumes the same dict shape as before.

### 5.4 Testing strategy this design unlocks

The single biggest benefit of this refactor, worth stating explicitly: it
turns the two most important tests from §2.6
(`test_record_demonstrations_row_alignment.py`,
`test_record_demonstrations_reward_sanity.py`) — both of which currently
have to run real (or heavily monkeypatched) `ScenarioEnv` physics for
several seconds each because there is no smaller unit to test against —
into tests that *could* be rewritten as fast, direct unit tests of
`DemonstrationRecorder`'s individual methods, with no environment/physics
dependency at all:

- The `_pending_reward` leak (§2.5) becomes: construct a
  `DemonstrationRecorder` with a fake/minimal `env`, call
  `start_episode()`, `accrue_pending_reward({"trainee": 100.0})` several
  times, call `start_episode()` again (simulating the next episode), call
  `record_callback_sample("trainee")`, and assert the recorded reward is
  ~0, not ~100+. No physics, no monkeypatching `ScenarioEnv.step`, no
  8-second runtime.
- The row-misattribution bug (§2.4) becomes: call `record_timed_sample()`,
  manually append a third synthetic row to the recorder's own lists
  (simulating what `record_callback_sample()` would have inserted mid-
  `step()`), then call `backfill_step_results()` with the *original*
  (pid, row_idx) pairs from the first call, and assert those exact
  absolute rows — not "the last N rows" — got the right values, regardless
  of the extra row in between.

The existing physics-based tests should be **kept** (they test the
integration — that `on_kick`/`on_tackle` really do fire synchronously
inside a real `env.step()`, that `record_timed_sample()`/
`record_callback_sample()` really do get wired to the right callbacks by
whatever replaces today's `_make_on_kick`/`_make_on_tackle` — an
integration bug in the *wiring* is just as real a risk as a bug in the
*logic*, and only an end-to-end test catches wiring bugs). The new,
faster unit tests should be **added alongside**, not instead of, since they
test a different failure mode (the logic in isolation) at a much lower
cost, making them suitable for a tighter dev-loop / pre-commit check.

### 5.5 Migration plan

1. Create `DemonstrationRecorder` as a new class, initially just wrapping/
   delegating to the existing free-function logic with no behavior change
   (i.e., a thin shim) — get the multiprocessing entry points switched
   over to constructing and calling it first, with the *internals* still
   effectively the old function, so the "does the plumbing work" question
   is resolved independently of "is the internal state management
   correct."
2. Extract `start_episode()`/the per-episode accumulator fields, one at a
   time, verifying via the existing (physics-based) tests after each
   extraction that behavior is unchanged.
3. Extract `record_timed_sample()`/`record_callback_sample()`/
   `backfill_step_results()`/`accrue_pending_reward()` as named methods,
   replacing the closures. This is the step where the (pid, row_idx)
   pairing from §5.3's `record_timed_sample()` docstring should replace
   today's separate `_recorded_ids`/`_recorded_row_indices` two-value
   return — consolidate them into one list of pairs as part of this
   extraction, not as a separate later change, since it's the same
   underlying simplification.
4. Write the new fast unit tests from §5.4 directly against the extracted
   methods.
5. Only once all of the above is green: consider whether Refactor A has
   landed, and if so, simplify `backfill_step_results()`/the per-player
   iteration to use `env.driven_player_ids`/`PlayerStepResult` instead of
   the current `[trainee_id, "opponent"]` hardcoding + `env.last_secondary_results`
   list-of-dicts shape. If Refactor A has *not* landed, leave this as a
   `# TODO(refactor-a)`-flagged spot and keep consuming the current
   external `ScenarioEnv` shape — Refactor B does not need to block on
   Refactor A, see §7.

### 5.6 Explicitly NOT in scope for Refactor B

- Changing the `.npz` output schema/keys — that's Refactor C. Keep
  `to_dataset_dict()`'s output identical to today's `return {...}` dict
  during Refactor B; layering a schema change on top of a state-management
  refactor at the same time makes it much harder to isolate which change
  caused a regression if one appears.
- Changing the multiprocessing job-splitting/file-chunking logic in
  `_run_recording_job`/`main()` — unaffected by this refactor, keep as-is.
- Changing the CLI argument surface (`--n-episodes`, `--seed`, etc.) —
  unaffected.

---

## 6. Refactor C — Explicit row identity in the dataset format

### 6.1 Current state (exact inventory)

`DemonstrationDataset` (`src/footballcoach/ai/bc/dataset.py`) and the
`.npz` files it loads currently identify a row's player and episode
membership **entirely implicitly**:

- **Player identity**: a single `is_trainee: np.ndarray[float32]` column
  (1.0 or 0.0). This can distinguish "the trainee" from "not the trainee"
  but **cannot distinguish between two different non-trainee players** —
  a hard blocker already flagged in `ai/knowledge.md`'s "hardcoded to
  exactly 2 players" note for any future 3+ player scenario. There is no
  `player_id` string column at all — the actual id (`"trainee"`,
  `"opponent"`, or whatever a future scenario names its players) is
  discarded at recording time and never stored.
- **Episode identity**: not stored at all. Episode boundaries are inferred
  from the `dones: np.ndarray[float32]` column by requiring **exactly 2
  consecutive** `done=1` rows (`DemonstrationDataset._DONE_ROWS_PER_EPISODE_BOUNDARY
  = 2`, in `_full_dataset_episode_row_ranges()`) — a convention that is
  correct *only* because `record_demonstrations.py` currently always
  records exactly 2 players (trainee + one opponent) per timed sample, and
  backfills `done=1` onto both of that tick's rows when the episode ends.
  This convention is: (a) not self-documenting from the data alone — you
  have to read `dataset.py`'s docstring to know "2" is a hardcoded
  assumption, not something computed from anything else in the file; (b)
  silently wrong for any future N != 2 player count; (c) — and this is the
  part directly implicated in this session's bugs — **fragile to exactly
  the kind of row-insertion bug described in §2.4/§2.5**, because an
  orphaned row (a kick/tackle callback's row, sitting between one episode's
  correctly-paired `done=1` rows and the next episode's first row) has
  `done=0` and is silently absorbed into whichever episode's row range
  happens to start next, contaminating that episode's `is_trainee`-based
  per-track return computation.
- **Row ordering as identity**: within a single timed sample, the trainee's
  row is assumed to precede the opponent's row (append order in
  `_record_now`), and consumers (e.g. `record_demonstrations.py`'s own
  `_recorded_ids = [env.trainee_player_id, "opponent"]` list, used
  positionally against however many rows just got appended) rely on this
  ordering rather than reading it back from the data.
- **`compute_returns()`/`compute_component_returns()`** (in `dataset.py`)
  implement a **two-accumulator backward MC scan** keyed by
  `is_trainee > 0.5`, with a `prev_had_done` flag specifically designed to
  collapse exactly-2-consecutive `done=1` rows into a single reset point.
  This is a second, independent place (beyond
  `_full_dataset_episode_row_ranges()`) that encodes the same "exactly 2,
  exactly this convention" assumption — already flagged in its own
  docstring as something "would need rethinking" (not just extending) once
  a dataset has more than 2 players.
- **`reward_components`** (per `ai/knowledge.md`'s existing note, `see
  compute_component_returns()'s docstring`) is documented as remaining
  "one ENV-LEVEL (all players combined) value duplicated onto every
  player's row" as a **known, deliberately-deferred limitation** — this
  was true when that note was written; the §2.3 fix in this session
  actually *resolved* this specific claim (`reward_components` is now
  correctly per-player, not env-level-combined) — but the note in
  `ai/knowledge.md` should be updated to reflect this once this document's
  refactors are read, since it's now stale.

### 6.2 Why this format caused §2.2/§2.3/§2.4/§2.5, and will keep causing similar bugs

Every one of these bugs is, at its core, a case of the recording/analysis
code needing to answer "which player, which episode, does this row belong
to" — a question the current format only lets you answer by *inference*
(position, counting, the `is_trainee` flag), never by *reading a value
directly off the row*. Inference-based identity is exactly the kind of
implicit invariant §3 argues is the root cause: it holds only as long as
every writer and every reader agrees on the same convention, and the
moment one writer (a kick/tackle callback, per §2.4) doesn't participate
in maintaining that convention, every reader downstream silently gets
wrong answers with no signal that anything went wrong.

Concretely:
- §2.2 (duplicated boundary logic) exists at all because "how do I find
  episode boundaries" is a nontrivial enough *derivation* (not a direct
  read) that it was worth writing a whole function for, and that function
  got duplicated. If episode boundaries were a directly-stored
  `episode_id` column, "how do I find episode boundaries" becomes
  `np.unique(episode_id, return_index=True)` — trivial enough that
  duplicating it would be silly, and any accidental duplicate would be
  obviously equivalent by inspection rather than subtly different.
- §2.3/§2.4 (component/reward misattribution) exist because "which row
  belongs to which player" during recording is maintained by *append
  order*, a convention that any reentrant code (a callback) can violate
  without anyone noticing, because nothing checks it. If every row carried
  its own `player_id` at the moment it was appended (written once, at
  creation, never inferred later), a callback inserting an extra row
  would still be correctly self-identified — there would be no
  "which of the last N rows is whose" question to get wrong, because the
  answer is already written directly on each row.
- §2.5 (the `_pending_reward` leak) is adjacent to this category rather
  than directly caused by it — it's a Refactor-B-shaped bug (state
  lifecycle, not row identity) — but its *symptom* (a corrupted value
  silently spreading across an entire episode's `compute_returns()` output)
  was made *worse* by the episode-boundary-inference fragility: because
  `compute_returns()` has to *infer* episode boundaries from `done`-pairing
  rather than reading an explicit `episode_id`, there was no independent
  check that could have caught "this episode's total is absurd" at the
  boundary-detection level — the corruption and the boundary-inference
  machinery are both working "correctly" by their own local logic, and
  only an external, semantic sanity check (§2.6's new test) can catch the
  combination.

### 6.3 Proposed design

**Add two new columns to the recorded `.npz` format and
`DemonstrationDataset`:**

```
episode_id: np.ndarray[int64]   # monotonically increasing, unique per
                                  # recorded episode, assigned by the
                                  # recorder at record_timed_sample()/
                                  # record_callback_sample() time -- NOT
                                  # inferred later from `dones`.
player_id:  np.ndarray[str]     # the actual player id string
                                  # ("trainee", "opponent", or whatever a
                                  # future N-player scenario names its
                                  # players) -- NOT just is_trainee's
                                  # boolean collapse of it.
```

`is_trainee` and `dones` **both stay** — this is additive, not a breaking
replacement of the existing columns (see §6.5 on backward compatibility).
`is_trainee` remains useful as a fast boolean filter (much cheaper than a
string comparison against `player_id`) for the extremely common
"trainee-only" queries throughout the codebase (`trainee_valid_idx`,
outcome classification, etc.) — keep it as a derived-but-also-stored
convenience column, computed once at record time as
`player_id == env.trainee_player_id`, not removed.

`dones` also stays, for the same reason plus backward compatibility, but
its *role* changes: it is no longer the primary signal
`_full_dataset_episode_row_ranges()` uses to find boundaries — `episode_id`
is. `dones` remains useful as "was this row the terminal tick of its
episode" (a per-row fact, still meaningful and used by
`compute_returns()`'s discounting, which needs to know when a given
player's *own* reward stream ends within their own episode — this is
subtly different from "what episode does this row belong to" and both
facts are worth keeping separately).

**`DemonstrationDataset` changes:**

- `_full_dataset_episode_row_ranges()` becomes trivial:
  ```python
  def _full_dataset_episode_row_ranges(self) -> list[tuple[int, int]]:
      if self._episode_ranges_cache is None:
          # episode_id is monotonically increasing and contiguous per
          # episode by construction (see DemonstrationRecorder.start_episode()
          # in reward_fixes.md section 5) -- this replaces the old
          # done==1-counting scan entirely.
          change_points = np.nonzero(np.diff(self._episode_id))[0] + 1
          starts = np.concatenate([[0], change_points])
          ends = np.concatenate([change_points - 1, [len(self._episode_id) - 1]])
          self._episode_ranges_cache = list(zip(starts.tolist(), ends.tolist()))
      return self._episode_ranges_cache
  ```
  No more "collapse exactly 2 consecutive done rows" logic, no more
  `_DONE_ROWS_PER_EPISODE_BOUNDARY` constant, and — directly relevant to
  §2.2 — no possible future duplicate-with-different-boundary-count bug,
  because there is no boundary-counting logic left to duplicate.

- `compute_returns()`/`compute_component_returns()`'s two-accumulator
  (`running_trainee`/`running_opponent`) scan, keyed by `is_trainee > 0.5`
  with the `prev_had_done` collapsing logic, becomes a **dict of
  accumulators keyed by `player_id`**, reset whenever `episode_id[i] !=
  episode_id[i-1]` (in the backward scan: `episode_id[i] !=
  episode_id[i+1]`) — directly generalizing to any number of players per
  episode, and removing the `prev_had_done`/consecutive-done-counting
  machinery entirely (episode-change detection no longer depends on
  `dones` at all, only on `episode_id`, which is a strictly simpler and
  more directly meaningful signal). This is the change that actually
  resolves the "hardcoded to 2 players" limitation already flagged in
  `ai/knowledge.md` for this exact function — do this as part of Refactor
  C, not as a separate future change, since the accumulator-keying change
  and the episode-boundary-source change are the same underlying fix
  (both replace inference with direct reads) and are easiest to verify
  together (one new set of tests, not two).

- `valid_indices()`, `classify_outcome()`, `row_outcomes()`,
  `episode_row_ranges(row_pool)` (the public, row-pool-filterable
  version) — all become simpler once episode boundaries are a direct
  column rather than a derived scan; audit each for logic that currently
  exists specifically to work around the done-pairing convention (e.g.
  `episode_row_ranges()`'s existing docstring already discusses "a
  filtered pool can never miss or merge an episode boundary" specifically
  *because* boundaries are resolved against the full, unfiltered dataset
  first — with a direct `episode_id` column, this concern mostly
  evaporates, since `episode_id` survives filtering by definition, unlike
  `done`-pairing which can be broken by filtering out one of a pair's two
  rows, e.g. an immobile player's own row via `valid_indices()`).

### 6.4 `DemonstrationRecorder` (Refactor B) changes required to populate the new columns

If Refactor B has landed first (see §7 for sequencing reasoning), this is
almost free: `DemonstrationRecorder.start_episode()` assigns a fresh,
monotonically-increasing `self._current_episode_id` (a simple counter,
incremented once per `start_episode()` call — note this must be a
*global*, run-wide counter, not reset per episode, unlike the
per-episode-scoped accumulators §5's `start_episode()` resets — a subtle
but important distinction to flag clearly in that method's implementation
and its docstring, precisely because "does this counter reset or not" is
exactly the class of question this whole refactor effort is trying to make
impossible to get wrong by accident). `record_timed_sample()`/
`record_callback_sample()` both stamp every row they append with
`self._current_episode_id` and the row's actual `player_id` (already known
directly at the call site — no inference needed) at creation time.

If Refactor B has **not** landed (i.e., Refactor C is being done against
the current closure-based `record_episodes()`), the same two values
(`episode_id`, `player_id`) need to be threaded through the existing
`_record_now()` closure and its three call sites — mechanically
straightforward (add two more `.append()` calls alongside the existing
`is_trainee_flags.append(...)` line, and an `episode_id` counter variable
declared and incremented exactly where the existing per-episode
accumulators are, which — worth noting — makes this a smaller, more
self-contained version of exactly the bug class §2.5 is about: get the
scope of this NEW counter wrong (e.g. accidentally reset it per-episode
instead of incrementing it) and you'd reintroduce a duplicate-episode-id
bug immediately. This is itself a good argument for doing Refactor B
first — see §7.

### 6.5 Backward compatibility / migration for existing recorded data

Existing `.npz` files (recorded before this refactor) will not have
`episode_id`/`player_id` columns. `DemonstrationDataset.from_file()`/
`from_files()` should:

1. Detect their absence (`"episode_id" not in data.files`) and fall back
   to **today's exact inference logic** (done-pairing boundary detection,
   `is_trainee`-only player identity) for those files — i.e., the old code
   paths should be **kept**, not deleted, gated behind this presence
   check, for as long as any pre-Refactor-C `.npz` files are expected to
   still be loaded. Given this session's experience with how quickly
   demonstration directories get regenerated during active development
   (multiple full regens in a single session), it's plausible this
   fallback path has a short practical lifespan — but do not assume that;
   make the fallback explicit and correct, not a "this will never actually
   be hit" shortcut.
2. When mixing old-format and new-format files within a single
   `from_files()` call (a plausible scenario if a directory accumulates
   files across a schema upgrade), either (a) refuse with a clear error
   (simplest, safest — recommend this as the default), or (b) synthesize
   `episode_id`/`player_id` for the old files using the inference fallback
   from point 1, offset so old-file episode ids never collide with
   new-file episode ids, and set `player_id` from `is_trainee` (`"trainee"`
   vs. a generic `"secondary_0"` placeholder, since the old format cannot
   recover the *actual* opponent id string if it was ever anything other
   than `"opponent"` — check `record_demonstrations.py`'s history; as of
   this session it is always literally `"opponent"`, so this is likely a
   non-issue in practice, but verify before relying on it). Recommend (a)
   unless there's a concrete need for (b) — mixing schema versions
   silently is exactly the kind of implicit-convention fragility this
   whole document argues against; an explicit error is more in keeping
   with the spirit of the refactor than a silent best-effort merge.
3. Add a `meta_schema_version` (or similar) field to newly-recorded `.npz`
   files, so this detection can be a direct version check rather than a
   heuristic "does this key exist" probe, going forward — cheap insurance
   against needing a third inference mechanism the next time the schema
   needs to change.

### 6.6 Testing strategy

- Unit tests for the new `_full_dataset_episode_row_ranges()`/
  `compute_returns()`/`compute_component_returns()` implementations,
  directly constructing a small synthetic `DemonstrationDataset` (already
  the pattern used by existing tests in `tests/ai_unit/test_demonstration_dataset.py`
  — check that file for the existing synthetic-construction helpers before
  writing new ones) with explicit `episode_id`/`player_id` arrays,
  including adversarial cases the old format could never represent
  cleanly: 3+ players per episode, an episode with only 1 player's rows
  present (e.g. after filtering), episodes of varying player counts within
  the same dataset.
- A specific regression test reproducing §2.4's exact scenario
  (a callback-inserted row landing between two episodes) against the *new*
  format: assert the inserted row's `episode_id` correctly identifies
  which episode it actually happened in (whichever episode was active
  when `record_callback_sample()`/`record_timed_sample()` was called) —
  proving this format change, unlike the old done-pairing inference, does
  NOT silently reattribute an orphaned row to the wrong episode.
- Backward-compatibility tests: construct an old-format-shaped synthetic
  `.npz` (no `episode_id`/`player_id` keys) and verify `from_file()`/
  `from_files()` still loads it correctly via the fallback path, with
  identical `episode_row_ranges()`/`compute_returns()` output to what the
  pre-Refactor-C code would have produced on the same input — this is the
  regression test that guards the fallback path itself from bit-rotting
  unnoticed once new-format files are the norm.

### 6.7 Explicitly NOT in scope for Refactor C

- Removing `is_trainee`/`dones` — both stay, as documented, useful
  convenience columns even once `episode_id`/`player_id` exist.
- Rewriting `reward_components`' storage layout — the §2.3 fix already
  made it correctly per-player; Refactor C's `player_id` column makes this
  *retrievable more directly* (no need to reconstruct "whose components
  are these" from `is_trainee` + row position) but the underlying array
  shape (`(n_rows, n_components)`) is unaffected.
- Any change to the BC label (`bc_labels`) format or the observation
  (`obs_*`) arrays — unaffected by this refactor.

---

## 7. Recommended sequencing, and why

**Recommended order: Refactor A, then Refactor C, then Refactor B** — with
the caveat that A and B could technically be done in either order relative
to each other (they touch different files), but C should follow A and
precede or accompany B, for the reasons below. If resourcing only allows
one or two of the three, prioritize in this same order — A alone still
removes the largest single class of duplicated-logic bugs (§2.1's shape);
C alone removes the largest single class of implicit-invariant bugs
(§2.2/§2.4's shape); B alone removes §2.5's shape and, importantly, is the
refactor that makes the *other two* easiest to test in isolation going
forward.

**Why A before C:** Refactor C's `player_id` column is populated at
record time by reading the actual player id directly off whatever
`ScenarioEnv` reports for that tick. If Refactor A has already unified
`ScenarioEnv`'s per-player results into one uniform `PlayerStepResult`
shape (§4.3.4), populating `player_id` correctly for every driven player —
trainee or secondary, and for however many secondary players exist in a
future N-player scenario — is a direct, uniform read. Without Refactor A,
the recorder still has to special-case "the trainee's player_id is
`env.trainee_player_id`, read from `env.step()`'s own scope" vs. "each
secondary player's `player_id` is `sec["player_id"]`, read from
`env.last_secondary_results`" — which reproduces, in the *new* column,
the exact same trainee/secondary asymmetry Refactor A exists to remove
everywhere else. Doing C before A would mean either accepting that
asymmetry in the new format too (defeating a chunk of C's purpose), or
doing the uniform-read logic twice (once ad hoc inside the recorder for C,
then again properly inside `ScenarioEnv` for A) — wasted, duplicated work
of exactly the kind this whole document is about avoiding.

**Why C before/alongside B, not after:** §6.4 already notes that adding
the `episode_id` counter to the *current*, closure-based
`record_episodes()` is itself a small instance of the exact bug class
Refactor B exists to prevent (an easy-to-miscount lifecycle: does this
counter reset per-episode, or increment across episodes?). Doing C's
column-population work *as part of* Refactor B's extraction (i.e.,
`DemonstrationRecorder.start_episode()` is written from the start to both
reset per-episode accumulators §5 already describes *and* increment
`self._current_episode_id`, with both responsibilities visible in the same
method and its docstring, rather than added piecemeal to the old function
later) means the new counter's correct lifecycle is established with the
same rigor (and the same "make the mistake impossible to write, not just
fixed once" mindset) as everything else Refactor B is doing. Doing C
first against the old closure-based code, then B afterward, risks
re-threading the new columns through the extraction a second time for no
benefit.

**Given the above, in practice:** do Refactor A on its own first (it's the
most self-contained — entirely within `scenario_env.py`, external
interface unchanged per §4.3.4, so it carries the least risk of touching
anything outside its own file). Then do Refactors B and C **together**, as
one combined effort against `record_demonstrations.py`/`dataset.py` — B's
class extraction and C's new-column population are naturally the same
piece of work once you accept the sequencing argument above, and
splitting them into two separate migrations of the same code would likely
cost more total effort than doing them as one pass with two motivating
sections (this document keeps them as separate sections purely for
clarity of exposition and independent readability, not as a mandate that
they must be separate PRs/commits).

**What NOT to do:** attempt all three simultaneously as one large PR.
Given how each individual bug in §2 took multiple diagnostic passes to
even *find*, let alone fix, correctly, a combined change touching
`scenario_env.py`, `record_demonstrations.py`, and `dataset.py` all at
once — three files that, per this document's own analysis, are exactly
the files where subtle, hard-to-notice bugs have repeatedly hidden — is a
significant risk of introducing a *sixth* bug in the same family while
fixing the structural cause of the first five, with no ability to bisect
which of the three refactors introduced it. Land and fully verify
(full test suite + the volume-based statistical re-verification
§4.5 step 7 describes) each stage before starting the next.

---

## 8. Non-goals (across all three refactors)

Stated explicitly so a future reader doesn't assume silence means
oversight:

- **No reward-design changes.** Nothing in this document proposes changing
  what any reward component means, its magnitude, or when it fires — only
  how correctly and legibly that computation is plumbed through the
  codebase. `reward.py`'s actual `phase1_reward()` math is out of scope
  everywhere in this document.
- **No performance optimization.** Some of the proposed changes (an extra
  `player_id` string column, a class instead of closures) have a nonzero
  but almost certainly negligible runtime/memory cost relative to the
  physics simulation and neural network inference this pipeline already
  does per tick. If a future profiling effort finds otherwise, that's a
  separate, later concern — do not preemptively optimize any of this
  refactor for performance at the cost of the clarity it exists to add.
- **No changes to `debug_value_network.py`** beyond what already landed
  this session (§2.2) — it remains a gitignored, standalone diagnostic
  script, explicitly not part of the package's test-covered surface. If
  Refactor C lands, it becomes possible (not mandatory) to simplify that
  script's own remaining ad hoc episode-boundary-adjacent logic by relying
  on the new `episode_id` column directly — flagged here as a nice-to-have
  cleanup opportunity, not a requirement of any of the three refactors.
- **No solving of the N>2-player product question** (§4.3.3 point 2) —
  every refactor here is explicitly designed to make N>2-player support
  *tractable as a future, separate effort*, not to implement it. Resist
  the temptation to scope-creep any of these three refactors into "and
  also now support 3v3" — that is real, separate product/design work
  (what does "losing" mean with 3+ competing players? what does the
  observation encoder's `MAX_OTHER_PLAYERS`/slot-shuffling logic need to
  change? does the curriculum/scenario-building code support it?) that
  deserves its own planning document once someone actually needs it.
