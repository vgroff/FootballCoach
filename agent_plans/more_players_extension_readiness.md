# More-Players Extension Readiness — FootballCoach

**Status as of 2026-10-12.** New file, written fresh this session (not a
reconciliation of a prior plan). Purpose: the user's explicit next goal
after the upcoming cleanup pass is extending training beyond 1v1 "phase 1"
to more players per side / more game phases. This file collects, in one
place, every concrete hardcoded-1v1/2-player assumption and related risk
found by (a) a fresh read-through of the env/training/network code with
that specific lens, and (b) two findings from earlier this same session
that aren't yet written down anywhere in `agent_plans/`. Every item below
was verified against current code (file:line, or a direct code excerpt) —
nothing here is carried over from a stale document. No code was changed;
this is a planning/documentation file only.

Headline take: the **observation schema and `ScenarioEnv` env-level
plumbing already generalize further than you might expect** (sized for
full 11v11, and already support N "secondary" neural players) — the real
2-player landmines are concentrated in **(1) three per-slot network heads
that are quietly not permutation-invariant, (2) the offline BC/demo-dataset
recording and return-computation pipeline, and (3) one team-vs-trainee
ambiguity in box-terminal detection.** None of these are live bugs in
today's 1v1 phase 1 (verified in each case below), but each would misfire
the moment a scenario has more than one player on a side or more than two
players total.

---

## 1. `EntityEncoder` permutation invariance — exception in the target-logit heads (found this session)

`src/footballcoach/ai/models/entity_encoder.py` is permutation-invariant by
design for everything that goes through it: a shared per-slot MLP followed
by masked `nn.MultiheadAttention` pooling, so which of the
`MAX_OTHER_PLAYERS` (21) slots the real opponent/teammate lands in cannot
affect any head built on top of it.

**Exception, confirmed by reading `src/footballcoach/ai/models/decision_network.py:130-132`:**

```python
self.pass_target_logits = nn.Linear(trunk_hidden, MAX_OTHER_PLAYERS)
self.tackle_target_logits = nn.Linear(trunk_hidden, MAX_OTHER_PLAYERS)
self.mark_target_logits = nn.Linear(trunk_hidden, MAX_OTHER_PLAYERS)
```

These three heads are plain per-slot-indexed `nn.Linear` layers — each of
the 21 output positions has its own independently-learned weight row, with
no attention/pointer mechanism over the per-slot embeddings the way every
other head gets. So "the logit for whichever player happens to be in slot
5" and "the logit for whichever player happens to be in slot 17" are
genuinely different numbers even for the identical player in identical
context.

**Currently dormant/harmless, confirmed by test, not just asserted:** every
phase today (`MaskedCategorical`, see `ai/action/distributions.py`) only
ever has exactly one or zero unmasked candidate slots for these three
heads. Softmax over `[L_valid, -inf, -inf, ...]` is 1.0 on the one valid
slot regardless of `L_valid`'s actual value — so sample/log_prob/entropy
are mathematically unaffected by the non-invariance as long as there's at
most one real candidate.

**Would become a real, live bug once a phase has 2+ simultaneously-valid
pass/tackle/mark targets** — i.e. almost immediately once play moves beyond
1v1 (a trainee with a teammate to pass to, or two markable opponents): the
network would then have a real, if dormant, preference baked into *which
slot* a target sits in rather than *who* the target is, undermining the
whole point of the slot-shuffle augmentation / permutation-invariant
design elsewhere.

**Regression tests already added this session**,
`tests/ai_unit/test_entity_encoder_permutation.py` (5 tests, passing):
confirms every other head is exactly invariant/equivariant across slots,
and separately pins down (not just documents) this exception — one test
proves the raw logits really do differ by slot, another proves
`MaskedCategorical` is nonetheless inert to that in the single-candidate
case, so a future accidental "fix" or regression is caught either way.

**Recommendation for the cleanup pass:** before (or as part of) extending
any phase to multiple simultaneous valid targets, replace these three
heads with the same shared-MLP-+-attention-pool pattern
`ValueAiTypeSideChannel` already uses as a precedent fix for an identical
anti-pattern (see `ai/knowledge.md`'s "Value-only opponent-AI-type side
channel: permutation invariance" section — that side channel used to be
exactly this same flatten+Linear bug, already fixed once).

## 2. `PPOTrainer.load_checkpoint()` — `_checkpoint_count` not restored (found and fixed this session, uncommitted)

`src/footballcoach/ai/ppo/ppo_trainer.py`: `_save_checkpoint()` (~line
11503) increments `self._checkpoint_count` and writes
`checkpoint{N}.pt`/`training_log{N}.txt`, and saves `"checkpoint_count"`
into the checkpoint dict (line 11507). `load_checkpoint()` previously
restored `self._total_steps` from the loaded checkpoint but never restored
`self._checkpoint_count` — so resuming a killed/crashed run from a
`--checkpoint` path (as opposed to an uninterrupted continuous process,
where this counter just lives in memory the whole time) silently reset
numbering to 0, and the new process then overwrote `checkpoint1.pt`,
`checkpoint2.pt`, ... from scratch. Training logs were spared actual data
loss only because `logging.FileHandler` appends rather than truncates;
`checkpoint{N}.pt` files are plain `torch.save()` overwrites with no such
protection.

**Already fixed in code this session** — confirmed via `git diff` on
`src/footballcoach/ai/ppo/ppo_trainer.py` (around line 11618-11631):

```python
self._checkpoint_count = ckpt.get("checkpoint_count", self._checkpoint_count)
```

right after the `_total_steps` restore line, with `.get()` defaulting to
the current value for checkpoints saved before this field existed.

**NOT YET COMMITTED** — confirmed via `git status`: this is still a
working-tree modification, not in any commit. Flagging this explicitly per
the task's instructions: do not assume it's landed just because the code
looks fixed; check `git log`/`git status` again before relying on it.

**Same class of bug, lower severity, found while checking for siblings**:
two more per-process counters in the same file are *also* never
round-tripped through the checkpoint dict and never restored in
`load_checkpoint()`:
- `self._rollouts_since_ppg_interleave` (line ~1852, incremented at
  ~4064, reset at ~4101) — the interleaved-PPG-burst cadence timer.
- `self._nn_snapshot_rollout_count` (line ~1597, incremented at ~3941) —
  the neural-opponent-snapshot cadence timer.

Resuming from a `--checkpoint` path resets both of these to 0, which just
delays the next PPG-interleave burst / neural-snapshot update by up to one
full cadence period — not data loss like the `_checkpoint_count` bug, but
the same underlying pattern ("in-memory counter not part of the
checkpoint's round-trip contract") and worth a single consistent fix
alongside it during cleanup rather than patching one-by-one as each is
separately noticed.

## 3. BC/demo-dataset recording is hardcoded to exactly 2 players (already self-documented, not yet in any `agent_plans/` file)

This is **not a new finding** — it's already flagged, loudly, in
`src/footballcoach/ai/knowledge.md` (search for "Hardcoded to exactly 2
players") — but it isn't referenced anywhere under `agent_plans/`, and it
is directly relevant to the more-players goal, so it's indexed here for
visibility:

- `ai/ppo/bc.py`'s `DemonstrationDataset.compute_returns()` /
  `compute_component_returns()` use a binary `is_trainee` flag and exactly
  two running accumulators (`running_trainee`/`running_opponent`) — no room
  for a third track.
- `record_demonstrations.py`'s `_record_now()` hardcodes
  `ids = [env.trainee_player_id, "opponent"]`.
- Per `ai/knowledge.md`'s own words: "the *reward-computation* side
  (`env.always_compute_secondary_reward`) already generalizes to N
  secondary players for free; only the recording/dataset side is
  2-player-specific." `ScenarioEnv.secondary_player_ids` is already a list
  today (see §4 below).

**Action for the cleanup pass**: when the first 2v2+/N-secondary-player
demo-recording scenario is built, `is_trainee` needs to become a real
per-row player-id/role field and both accumulator functions need a
dict-of-accumulators keyed by that field, done together (per the same
knowledge.md note) — not one fixed and the other left to drift.

## 4. Env/observation layer: more ready than you'd guess, but not wired up

Checked directly, rather than assumed:

- **`MAX_OTHER_PLAYERS = 21`** (`src/footballcoach/ai/obs/schema.py:284`,
  comment: "full 11v11 minus self"). The observation schema was already
  built for full-size matches; it is not a 1v1-only shape.
- **`ScenarioEnv.__init__` already takes `secondary_player_ids: Optional[list]`**
  (`src/footballcoach/ai/env/scenario_env.py:102-113`) — additional player
  IDs driven by the same neural network, with their own transitions
  collected via `last_secondary_results`. This is a list today, not a
  single opponent id.
- **`_compute_phase1_reward_for_player()` is already called once per
  player** (trainee at `scenario_env.py:~697`, and once per entry of
  `secondary_player_ids` in a loop at `~833`) — reward computation is
  already per-player-track, not hardcoded to exactly one trainee + one
  opponent.
- **The PPO rollout buffer already has multi-track GAE segmentation**
  (`RolloutBuffer.compute_gae()`/`compute_mc_returns()`, documented in
  `ai/knowledge.md`'s "Per-track GAE segmentation" section) — this was a
  real bug, already found and fixed, that would otherwise corrupt returns
  the moment a neural secondary (self-play) opponent produces a second
  track of rows. Note `ai_config.json`'s `curriculum.phase1_opponent_neural_ratio`
  is currently **3** (not 0 — `phase1_opponent_rules_ratio=1`,
  `phase1_opponent_immobile_ratio=0`, so self-play is already the
  *majority* of phase-1 episodes by these relative weights), so this fix is
  live and load-bearing today, not theoretical.
- **`ui/scenarios.py` already has `build_2v2_scenario`/`build_1v2_scenario`**
  (lines 716, 861) and `config/scenarios.json` has `"2v2"`/`"1v2"` entries —
  but grepping the whole `src/footballcoach/` tree for references to either
  scenario key shows **zero uses outside `ui/scenarios.py` and
  `config/scenarios.json` themselves** — they are UI-only
  manual-play/freeplay scenarios, never passed to a `CurriculumPhase` or
  any training entry point.
- **`src/footballcoach/ai/curriculum/phases.py` only has two real phases**:
  `PHASE_1_GET_POSSESSION` (`scenario_key="1v1"`) and `PHASE_2_SHOOT`
  (`scenario_key="penalty"`). `PHASE_3_PASSING`/`PHASE_4_TACKLING` are
  explicitly commented "Future phases (stubs for reference, not yet
  implemented)" in the file itself (lines 91-106) — this isn't a stale
  claim, it's the current state of the file.

**Net assessment**: the plumbing to drive N players through one shared
network and collect/credit their rollouts correctly already exists and is
exercised today (via self-play's "neural remainder" opponent). What's
missing for a genuine 2v2+ curriculum phase is (a) fixing §1's target-logit
heads, (b) §3's BC-recording 2-player hardcoding (only matters if 2v2+
demo recording is wanted), (c) actually wiring a `CurriculumPhase` to one
of the existing `"2v2"`/`"1v2"` scenario keys, and (d) §5 below.

## 5. Box-terminal detection conflates "not the trainee" with "an opponent" (found this session)

`src/footballcoach/ai/env/outcome.py`'s `detect_phase1_box_terminal()`
(lines 95-140) is the single source of truth for "who won a phase-1 trial,"
used by both `ScenarioEnv` (training) and `ScenarioLoop` (UI/debug). Reading
it directly:

```python
# "Trainee's own box" is relative to the TRAINEE's own team, not the
# carrier's -- do not swap this for carrier.team (they're not
# necessarily opposite teams once a scenario has more than 2 players).
in_trainee_box = match.pitch.is_in_box(
    ball.position, left=(trainee.team == Team.LEFT),  # trainee's own box
)
if in_trainee_box and can_score_box_terminal(carrier):
    return "opponent_box_possession"
```

The comment *correctly* identifies and handles the question "which box is
the trainee's own box" team-aware (uses `trainee.team`, not
`carrier.team`). But the branch that decides whether a non-trainee ball
carrier counts as "the opponent" at all is **not** team-aware: any
`carrier_id != trainee_player_id` with a real controller reaching the
trainee's own box is unconditionally reported as `"opponent_box_possession"`
(i.e., a loss for the trainee), with no check on whether that carrier is
actually on the opposing team.

**Confirmed harmless today**: phase 1 is strictly 1v1, so "not the
trainee" and "an opponent" are the same player by construction — there is
no third player for this to ever misfire on.

**Would misfire the moment a trainee has a genuine teammate** (e.g. a
real 2v2 curriculum phase with the trainee plus one own-team
`secondary_player_id`): if that teammate — not an opponent — dribbles the
ball into the trainee's own defensive box (an entirely ordinary, non-losing
action, e.g. a defender clearing the ball from deep), this function would
report `"opponent_box_possession"` and the episode would incorrectly score
as a trainee loss. The fix (checking `carrier.team != trainee.team` in
addition to `carrier_id != trainee_player_id`) is small, but it is a real
gap today, not a hypothetical one the comment is already guarding against
— the comment only guards the *box selection*, not the *carrier
attribution*.

## 6. Other stubbed/dead code relevant to opponent scaling

`src/footballcoach/ai/curriculum/opponent_pool.py`'s `OpponentPool` class
and `apply_rules_based_opponent()` function are confirmed **dead code** —
the module's own docstring says so explicitly ("neither `OpponentPool` nor
`apply_rules_based_opponent()` below is constructed/called anywhere in the
real training path"), and a repo-wide grep confirms no call sites exist
outside this module's own file. Real self-play works via
`rules_ai.NeuralPlayerAI`/`maybe_assign_neural_opponent` instead (same
live weights, not a frozen-checkpoint pool). This isn't a bug, but it's
worth flagging for the cleanup pass specifically because of the
more-players context: if a future multi-opponent curriculum phase wants
genuine *checkpoint-pool* diversity (as opposed to today's self-play-only
"neural remainder"), this module looks like a ready-made, half-built home
for that — but it's currently unused scaffolding that would need auditing
(does its design still match the current `Match`/`Player`/checkpoint API?)
before trusting it, not just re-enabling it.

---

## Summary table

| # | Item | File(s) | Severity if untouched | Fix complexity |
|---|------|---------|----|----|
| 1 | pass/tackle/mark target logits not permutation-invariant | `models/decision_network.py:130-132` | Silent policy bias the moment 2+ valid targets exist | Medium (reuse `ValueAiTypeSideChannel` pattern) |
| 2 | `load_checkpoint()` didn't restore `_checkpoint_count` | `ppo/ppo_trainer.py:~11630` | High (silent checkpoint overwrite) if hit, but **fixed, uncommitted** | Done, needs commit |
| 2b | PPG-interleave/snapshot cadence counters not checkpointed | `ppo/ppo_trainer.py:1597,1852` | Low (cadence timer resets, no data loss) | Low |
| 3 | BC dataset recording hardcoded to 2 players | `ai/ppo/bc.py`, `record_demonstrations.py` | N/A until 2v2+ demo recording attempted | Medium, already scoped in knowledge.md |
| 4 | No curriculum phase wired to any multi-player scenario | `curriculum/phases.py` | N/A (nothing broken, just not built) | Medium — scenario builders already exist |
| 5 | Box-terminal detection treats any non-trainee carrier as "opponent" | `ai/env/outcome.py:132-140` | Would silently mis-score a 2v2+ trial the first time a teammate (not opponent) touches the ball in the trainee's own box | Low (one extra team check) |
| 6 | `OpponentPool`/`apply_rules_based_opponent` dead scaffolding | `curriculum/opponent_pool.py` | None today; risk is reusing stale code uncritically later | N/A (audit before reuse, not a fix) |
