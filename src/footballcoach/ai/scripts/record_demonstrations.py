"""Record rules-based AI demonstrations for offline BC pre-training.

Runs N complete episodes of a given phase scenario with rules-based AI on
all sides, collecting (observation, bc_label) pairs at each decision step.
Saves the results as .npz files under a given output directory.

Usage::

    # Record 200 phase-1 episodes, 8 episodes per file (= 25 files)
    uv run python -m footballcoach.ai.scripts.record_demonstrations \\
        --phase 1 --n-episodes 200 --episodes-per-file 8 \\
        --output demonstrations/phase1/

    # Inspect what was recorded
    uv run python -m footballcoach.ai.scripts.record_demonstrations \\
        --phase 1 --n-episodes 0 --output demonstrations/phase1/ --info

Output .npz files contain:
    obs_self_feat, obs_other_feat, obs_exists_mask, obs_ball_feat,
    obs_global_feat, bc_labels, meta_phase, meta_scenario, meta_episode_seeds

Each file is self-contained and can be loaded individually or combined with
DemonstrationDataset.from_directory().
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import time
from pathlib import Path

# Must run BEFORE numpy (or torch, which imports it) is first imported
# ANYWHERE in this process -- numpy's underlying BLAS reads these env vars at
# import/first-use to size its OWN internal thread pool, independent of
# torch.set_num_threads(). Left unset, each process's numpy calls (the
# physics engine and observation encoder are numpy-heavy) AND torch calls
# (when --driver-checkpoint/--teacher-checkpoint load a network) default to
# one thread PER LOGICAL CORE -- fine for a single process, but with
# --n-processes > 1 (the common case here) every worker fights for the same
# cores, oversubscribing badly. Mirrors debug_value_network.py's identical
# top-of-file guard for the same reason -- see its comment for the full
# multiprocessing-spawn-bootstrap rationale (this file uses the same "spawn"
# context, so this must run this early in every worker too, not just main).
# os.environ.setdefault() (not a flat assignment) so an operator's own
# OMP_NUM_THREADS etc from the shell is still respected if already set.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("footballcoach.ai.record")


# ---------------------------------------------------------------------------
# Per-phase env + label-fn factories (shared via curriculum.envs)
# ---------------------------------------------------------------------------

def _build_env_and_label_fn(phase_id: int):
    """Return (env, label_fn, scenario_key) for the given phase."""
    from footballcoach.ai.curriculum.phases import PHASES_BY_ID
    from footballcoach.ai.curriculum.envs import build_env, bc_label_fn_for_phase

    phase = PHASES_BY_ID.get(phase_id)
    if phase is None:
        raise ValueError(f"Unknown phase: {phase_id}")
    label_fn = bc_label_fn_for_phase(phase_id)
    if label_fn is None:
        raise NotImplementedError(f"No BC label function defined for phase {phase_id}")
    env = build_env(phase)
    return env, label_fn, phase.scenario_key


# ---------------------------------------------------------------------------
# Action-count summary (tackle/kick armed, attempted, success/fail)
# ---------------------------------------------------------------------------

def _log_action_stats_summary(
    episode_action_counts: list[dict[str, dict[str, int]]], label: str = "whole run"
) -> None:
    """Log per-episode mean/std/median and grand totals for tackle/kick
    action counters (armed ticks, attempts, win/loss, kicks executed), PER
    ROLE (trainee/opponent) -- see episode_action_counts population in
    record_episodes(). Per-role, not combined: every tackle attempt has
    exactly one winner and one loser, so a trainee+opponent-combined win/loss
    total is always 100% and tells you nothing about actual tackle skill.
    """
    if not episode_action_counts:
        return
    roles = list(episode_action_counts[0].keys())
    keys = list(episode_action_counts[0][roles[0]].keys())
    n_eps = len(episode_action_counts)
    log.info(f"Action-count summary over {n_eps} episode(s) ({label}):")
    for role in roles:
        log.info(f"  [{role}]")
        for k in keys:
            vals = np.array([ep[role][k] for ep in episode_action_counts], dtype=np.float64)
            total = int(vals.sum())
            log.info(
                f"    {k:18s} total={total:6d}  per-ep: mean={vals.mean():6.2f}  "
                f"std={vals.std():6.2f}  median={np.median(vals):6.1f}  "
                f"min={vals.min():.0f}  max={vals.max():.0f}"
            )
        attempts_total = sum(ep[role]["tackle_attempts"] for ep in episode_action_counts)
        wins_total = sum(ep[role]["tackle_wins"] for ep in episode_action_counts)
        if attempts_total > 0:
            log.info(
                f"    tackle win rate: {wins_total}/{attempts_total} = {100.0 * wins_total / attempts_total:.1f}%"
            )
        auto_attempts_total = sum(ep[role]["auto_tackle_attempts"] for ep in episode_action_counts)
        auto_wins_total = sum(ep[role]["auto_tackle_wins"] for ep in episode_action_counts)
        if auto_attempts_total > 0:
            log.info(
                f"    auto-tackle win rate: {auto_wins_total}/{auto_attempts_total} = "
                f"{100.0 * auto_wins_total / auto_attempts_total:.1f}%"
            )


def _log_poss_reward_summary(episode_poss_reward: list[dict[str, float]], label: str = "whole run") -> None:
    """Log per-episode mean/std/median and grand totals for the get_possession
    ("poss")/lose_possession ("lpos") reward components -- env-level
    (trainee+opponent combined, same convention as the reward-breakdown log
    line above), tracked per-episode for the same reason as
    _log_action_stats_summary."""
    if not episode_poss_reward:
        return
    n_eps = len(episode_poss_reward)
    log.info(f"get_possession/lose_possession reward summary over {n_eps} episode(s) ({label}, trainee+opponent):")
    for k, name in (("poss", "get_possession"), ("lpos", "lose_possession")):
        vals = np.array([ep[k] for ep in episode_poss_reward], dtype=np.float64)
        log.info(
            f"  {name:16s} total={vals.sum():+8.2f}  per-ep: mean={vals.mean():+6.3f}  "
            f"std={vals.std():6.3f}  median={np.median(vals):+6.3f}  "
            f"min={vals.min():+.3f}  max={vals.max():+.3f}"
        )


# ---------------------------------------------------------------------------
# Recording logic
# ---------------------------------------------------------------------------

def record_episodes(
    env,
    label_fn,
    n_episodes: int,
    scenario_key: str,
    phase_id: int,
    episode_offset: int = 0,
    total_episodes: int | None = None,
    sample_every_n_decisions: int = 1,
    opponent_rules_prob: float = 0.0,
    opponent_immobile_prob: float | None = None,
    verbose_stats: bool = False,
    driver_trainer=None,
) -> dict:
    """Run *n_episodes* with rules-based AI driving the trainee and opponent
    (or a neural checkpoint driving the trainee, see ``driver_trainer``).

    Sampling strategy:
      - on_kick / on_tackle player callbacks fire at the exact engine tick the
        action executes → always recorded regardless of sample_every_n_decisions.
      - env.step() always advances by exactly ONE real decision interval (the
        env's own ``observation.decision_interval_s``, same cadence used by
        real PPO training/gameplay -- recording no longer overrides this, see
        below). A timed sample is recorded every *sample_every_n_decisions*
        calls to env.step(), so every recorded timed sample lands exactly on
        a genuine decision, never at an independently-configured cadence that
        drifts from how the trained policy actually gets stepped.
      - env.step() handles all terminal conditions normally (box possession,
        timeout) so episodes end correctly.

    Args:
        sample_every_n_decisions: Record a timed sample every this many real
            decision intervals (1 = record every single decision; 2 = every
            other decision, etc.). Kicks and tackles are always recorded via
            callbacks regardless, even on a decision interval that's
            otherwise skipped. Previously this was a sim-seconds interval
            (``sample_interval_s``) that overrode the env's own decision
            cadence (``env._ticks_per_decision``) to match it -- meaning
            recorded episodes ran the trainee/opponent AI at a DIFFERENT
            decision rate than real training ever uses, a real behavioural
            mismatch (Phase1RulesAI/NeuralPlayerAI re-evaluate their order
            once per decision, so a different cadence means different
            trajectories, not just different logging density). Recording
            now always uses the env's real, configured decision cadence;
            this parameter only controls how many of those genuine
            decisions get a timed sample.
        opponent_rules_prob / opponent_immobile_prob: Per-episode
            probabilities for the opponent's driver. **KNOWN LIMITATION,
            intentional for now**: these never actually add a neural
            opponent, even though there's a third implied "remainder"
            probability mass. The roll below (separate from, and
            overriding, whatever ``build_1v1_scenario`` itself decided)
            deliberately folds that remainder into "rules" instead of
            leaving the opponent neural-controlled -- only the TRAINEE can
            be neural during recording (via ``driver_trainer`` below). See
            ``ai/knowledge.md``'s "Demonstration recording" section for the
            full explanation and why it matters (BC data never covers what
            the opponent's own state distribution looks like when IT is
            neural).
        driver_trainer: Optional loaded ``PPOTrainer`` (e.g. via
            ``PPOTrainer.load_for_inference()``). When given, the TRAINEE is
            driven by this checkpoint (via ``NeuralPlayerAI``, auto-assigned
            by ``ScenarioEnv.reset()`` once ``env.sample_action_fn`` is set —
            see below) instead of ``Phase1RulesAI()``, so recorded states
            reflect this policy's own state visitation rather than the rules
            AI's. ``label_fn`` is independent of this — pass
            ``bc.phase1_labels_from_teacher`` (bound to a teacher trainer, via
            ``functools.partial`` or a lambda) as ``label_fn`` to also source
            BC labels from a neural teacher instead of Phase1RulesAI; the two
            are orthogonal (driver = state visitation, label_fn = supervision
            target) and may be the same checkpoint, different checkpoints
            (e.g. across an architecture change), or mixed with the default
            rules-AI label_fn. The opponent is unaffected by this parameter —
            it stays governed by ``opponent_rules_prob``/
            ``opponent_immobile_prob`` as before.

    Returns a dict of numpy arrays ready to be saved as .npz.
    """
    if driver_trainer is not None:
        # ScenarioEnv.reset() (called at the top of the episode loop below)
        # auto-assigns NeuralPlayerAI to the trainee whenever
        # env.sample_action_fn is set -- same wiring PPOTrainer.train() uses,
        # so decision_interval_ticks/max_episode_s/ema_smoothed are all
        # correctly derived from the env instead of guessed here.
        env.sample_action_fn = driver_trainer._sample_action
    # Opt in to a REAL per-player reward for the opponent's row too (via
    # env.last_secondary_results), computed through the exact same
    # _compute_phase1_reward_for_player() the trainee and real PPO training's
    # neural secondary players already use -- not the trainee's own reward
    # copy-pasted onto the opponent's row. Off by default on ScenarioEnv
    # (would crash PPOTrainer's rollout-buffer drain the instant it saw a
    # non-neural secondary player -- see the flag's docstring in
    # scenario_env.py); safe here since record_demonstrations.py only ever
    # reads reward/done off these rows, never feeds them into a PPO buffer.
    env.always_compute_secondary_reward = True
    from footballcoach.rules_ai import Phase1RulesAI
    from footballcoach.orders import JogOrder
    from footballcoach.ai.ppo.ppo_trainer import REWARD_COMP_LABELS

    if sample_every_n_decisions < 1:
        raise ValueError(f"sample_every_n_decisions must be >= 1, got {sample_every_n_decisions}")

    self_feats = []
    other_feats = []
    exists_masks = []
    ball_feats = []
    global_feats = []
    bc_labels = []
    rewards = []
    dones = []
    # 1.0 if this row is the TRAINEE's own row, 0.0 if it's the opponent's --
    # trainee and opponent rows are interleaved within an episode (see
    # _record_now()), so DemonstrationDataset.compute_returns() needs this to
    # segment its backward MC scan per player-track instead of mixing the two
    # players' reward streams together. See "is_trainee" note in dataset.py.
    is_trainee_flags = []
    # 1.0 if this row's own reward corresponds to a genuine NEW decision
    # interval (a real timed sample, recorded every sample_every_n_decisions
    # decisions), 0.0 for a kick/tackle-callback row or the trailing
    # true-terminal row (see _record_terminal_now) -- both fire at some
    # SUB-interval moment inside (or, for the terminal row, one physics tick
    # after) an ALREADY-counted decision interval, not a new one of their own.
    # DemonstrationDataset.compute_returns() uses this so its per-row MC
    # discount (`gamma ** 1` per row) only actually applies once per REAL
    # elapsed decision interval, not once per row regardless of real time --
    # otherwise an episode with a flurry of kicks (many callback rows in a
    # couple of real seconds) gets over-discounted relative to an
    # equal-duration episode with fewer touches, purely as an artifact of
    # how many rows happened to get inserted, not real elapsed time.
    is_decision_step_flags = []
    # Per-step reward-component breakdown, one fixed-width row per sample --
    # column order = REWARD_COMP_LABELS (short-key order), so this stays in
    # sync with ppo_trainer.py's diagnostics/reward.py's component keys
    # without a separate schema. Mid-step samples (timed samples, and every
    # sample except the one right after env.step()) carry all-zero rows,
    # same convention as `rewards`/`dones` -- the real per-step breakdown is
    # only ever known right after env.step() returns.
    reward_components: list[np.ndarray] = []
    _comp_key_order = [k for k, _ in REWARD_COMP_LABELS]

    steps_total = 0
    steps_valid = 0

    # Outcome counters, and the ground-truth per-episode outcome list (the
    # exact `info.trial_outcome` string ScenarioEnv computed -- see
    # ai/env/outcome.py's outcome vocabulary and ScenarioEnv.step()'s
    # "invalid" split). Persisted verbatim (meta_episode_outcomes below) so
    # DemonstrationDataset never has to INFER an episode's outcome from
    # per-row reward components again -- that inference is fundamentally
    # incomplete (e.g. an "invalid" ball-out with no toucher fires no
    # per-player reward component at all, so there is nothing to infer from).
    outcome_counts: dict[str, int] = {}
    episode_outcomes: list[str] = []
    # Per-episode seed passed to build_1v1_scenario(seed=...) (see below,
    # drawn fresh each episode from this worker's already-seeded --seed
    # stream) -- persisted verbatim (meta_episode_seeds below) so any
    # recorded episode can be rebuilt EXACTLY later (positions, attributes,
    # ball state, opponent-type roll, and every subsequent tackle-roll/kick-
    # noise draw, since build_1v1_scenario's returned Match keeps using this
    # same seeded rng for all of its own physics too -- see that function's
    # own docstring) via scripts/replay_episode.py, instead of only ever
    # being able to inspect a static post-hoc log. Previously no seed was
    # ever captured -- the --seed CLI flag only seeded the global numpy/
    # random modules, which build_1v1_scenario's own rng=random.Random()
    # (freshly OS-entropy-seeded, never connected to that global state) never
    # used, so recorded episodes were never actually reproducible despite
    # --seed's docstring implying otherwise.
    episode_seeds: list[int] = []

    # Reward component breakdown accumulator (mirrors train.py's diagnostic
    # "_comp_acc" pattern), reset after each periodic log line so the
    # printed line reflects the average since the last log, not the whole run.
    _comp_acc: dict[str, float] = {}
    _comp_acc_episodes = 0

    # Kick/tackle sample counters (since last periodic log), and running
    # totals for the whole recording run — visibility into how many of the
    # rarest, highest-value BC rows (kicks/tackles) are actually being
    # captured, without post-hoc .npz inspection. See
    # agent_plans/bc_execution_label_boundary_and_followups.md Part 5.
    _kick_count_since_log = 0
    _tackle_count_since_log = 0
    _kick_count_total = 0
    _tackle_count_total = 0

    # Engine-level action counters (armed intent, attempted contact, and
    # win/loss outcome), tracked PER ROLE (trainee/opponent, not summed
    # together -- every tackle attempt has exactly one winner and one loser,
    # so a combined win/loss total is always 100% and tells you nothing;
    # see debugging notes). Tracked per-episode so both the periodic log
    # lines (since-last-log window) and the end-of-run summary (whole run)
    # can report mean/std/median over episodes as well as grand totals.
    # tackle_armed/kick_armed are per-tick flags on Player, sampled once per
    # env.step() via definition.on_tick; on_tackle/on_tackle_result fire
    # exactly once per real tackle attempt/outcome regardless of
    # sample_every_n_decisions.
    # auto_tackle_attempts/wins/losses are the collision-based fallback path
    # (_check_head_on_tackles, on_auto_tackle_result) -- separate from
    # tackle_attempts/wins/losses (the intentional/armed path, on_tackle/
    # on_tackle_result) since the two paths never fire the same callback
    # pair; see player.py's on_auto_tackle_result docstring.
    _ACTION_COUNT_KEYS = (
        "tackle_armed_ticks", "kick_armed_ticks",
        "tackle_attempts", "tackle_wins", "tackle_losses",
        "auto_tackle_attempts", "auto_tackle_wins", "auto_tackle_losses",
        "kicks_executed",
    )
    _ROLES = ("trainee", "opponent")
    episode_action_counts: list[dict[str, dict[str, int]]] = []
    _ep_counts: dict[str, dict[str, int]] = {
        role: {k: 0 for k in _ACTION_COUNT_KEYS} for role in _ROLES
    }
    # Since-last-periodic-log accumulator (mirrors _comp_acc's reset pattern).
    _ep_counts_since_log: list[dict[str, dict[str, int]]] = []

    # Per-episode cumulative get_possession/lose_possession reward (env-level,
    # i.e. trainee+opponent combined -- same convention as _comp_acc/rewards
    # above, there's no separate per-player reward signal at this
    # granularity). Tracked per-episode (not just since-last-log) so the
    # final summary can report mean/std/median across the whole run,
    # matching the action-count summary's level of detail.
    episode_poss_reward: list[dict[str, float]] = []
    _ep_poss_reward: dict[str, float] = {"poss": 0.0, "lpos": 0.0}
    _poss_reward_since_log: list[dict[str, float]] = []

    if total_episodes is None:
        total_episodes = n_episodes


    def _record_now(reward: float | None = None, done: bool = False, player_id: str | None = None) -> list[str]:
        """Append one (obs, label) sample per player. Returns the list of
        player ids recorded, in append order, so the caller can backfill
        each row's OWN reward after env.step() (see the main loop below).

        player_id=None (used for timed samples) records BOTH the trainee and
        the opponent in one call. A specific player_id (used for on_kick/
        on_tackle callback samples, which fire per-player) records only that
        player.

        reward=None (the default, used by kick/tackle callbacks): these rows
        carry reward=0.0 always. The real reward for whatever env.step()
        call is in progress gets attributed EXACTLY ONCE, to that step's own
        timed-sample row (see the main loop below) — a kick/tackle callback
        firing synchronously inside that same env.step() call exists purely
        to record a genuine BC-label row at the exact tick the action
        happened (kick_this_tick=1 etc.), not to carry its own reward.
        Earlier this also drained a per-player pending-reward accrual here,
        which double-counted: the reward had already been written to its
        timed-sample row before any callback could fire, so the accrual was
        never actually needed to avoid losing anything — it only added a
        second, unlabelled copy of the same value on whatever row happened
        to consume it next. Timed samples explicitly pass reward=0.0 (a
        placeholder) and get each player's real, INDIVIDUALLY-COMPUTED
        reward backfilled after env.step() — the trainee's own reward
        (env.step()'s return) is NOT the same value as the opponent's own
        reward (env.last_secondary_results, computed via the same
        _compute_phase1_reward_for_player() call PPO training itself uses
        for a neural secondary player — see env.always_compute_secondary_
        reward).
        """
        ids = [env.trainee_player_id, "opponent"] if player_id is None else [player_id]
        nonlocal steps_total, steps_valid
        nonlocal _kick_count_since_log, _tackle_count_since_log, _kick_count_total, _tackle_count_total
        for pid in ids:
            pid_reward = 0.0 if reward is None else reward
            obs = env._get_obs(player_id=pid)
            label = label_fn(env, player_id=pid)
            label_arr = label.to_array()
            self_feats.append(obs.self_feat.copy())
            other_feats.append(obs.other_feat.copy())
            exists_masks.append(obs.exists_mask.copy())
            ball_feats.append(obs.ball_feat.copy())
            global_feats.append(obs.global_feat.copy())
            bc_labels.append(label_arr)
            rewards.append(np.float32(pid_reward))
            dones.append(np.float32(done))
            is_trainee_flags.append(np.float32(1.0 if pid == env.trainee_player_id else 0.0))
            # player_id is None only for the timed-sample call (line ~557) --
            # a genuine new decision interval; kick/tackle callbacks always
            # pass an explicit player_id (see is_decision_step_flags' own
            # comment above).
            is_decision_step_flags.append(np.float32(1.0 if player_id is None else 0.0))
            reward_components.append(np.zeros(len(_comp_key_order), dtype=np.float32))
            steps_total += 1
            if label.valid:
                steps_valid += 1
                if label.kick_this_tick > 0.5:
                    _kick_count_since_log += 1
                    _kick_count_total += 1
                if label.tackle_attempt > 0.5:
                    _tackle_count_since_log += 1
                    _tackle_count_total += 1
        return ids

    def _record_terminal_now() -> list[int]:
        """Append the TRUE final row for both players, one real physics tick
        later than any `_record_now()` call could ever capture -- see
        ScenarioLoop.last_completed_trial_match's docstring for exactly why
        that tick was previously unrecordable (env.step() only returns
        AFTER the next trial's match has already been built, so the episode
        that just ended can no longer be read from env._loop.match/
        env._get_obs() by that point). Every row here is real, recorded
        engine state -- NOT extrapolated or replayed.

        Returns the row indices just appended (always exactly 2: trainee,
        opponent) so the caller can mark them (not the previous timed
        sample's rows) as the episode's real dones=1 boundary.

        reward=0.0 / an invalid BCLabel: this tick corresponds to no new
        decision (the episode is already over) and no new reward (the
        terminal reward was already correctly attributed to the last timed
        sample by env.step()'s own return, exactly as before) -- this row
        exists ONLY to carry the true final observation, e.g. for match-log
        reconstruction. `label.valid=False` means BC training already skips
        it via the same steps_valid/label.valid convention every other
        padding row uses.
        """
        from footballcoach.ai.ppo.bc import BCLabel

        nonlocal steps_total
        row_indices = []
        for pid in (env.trainee_player_id, "opponent"):
            obs = env._get_obs(player_id=pid, match=env.last_terminal_match)
            label_arr = BCLabel(valid=False).to_array()
            self_feats.append(obs.self_feat.copy())
            other_feats.append(obs.other_feat.copy())
            exists_masks.append(obs.exists_mask.copy())
            ball_feats.append(obs.ball_feat.copy())
            global_feats.append(obs.global_feat.copy())
            bc_labels.append(label_arr)
            rewards.append(np.float32(0.0))
            dones.append(np.float32(0.0))  # caller overwrites -- see docstring
            is_trainee_flags.append(np.float32(1.0 if pid == env.trainee_player_id else 0.0))
            is_decision_step_flags.append(np.float32(0.0))  # see its own comment above
            reward_components.append(np.zeros(len(_comp_key_order), dtype=np.float32))
            row_indices.append(len(rewards) - 1)
            steps_total += 1
        return row_indices

    # Sample tackle_armed/kick_armed (transient per-tick flags on Player,
    # reset every tick by Match._process_orders) once per physics tick via
    # definition.on_tick -- the only hook that runs at tick granularity
    # inside env.step()'s multi-tick loop. Wraps whatever on_tick the
    # scenario already had (phase1_training_on_tick is currently a no-op,
    # but this must not silently drop it if that changes).
    _orig_on_tick = env.definition.on_tick

    def _sample_armed_flags(match, trial_tick):  # noqa: ANN001
        if _orig_on_tick is not None:
            _orig_on_tick(match, trial_tick)
        for role, pid in (("trainee", env.trainee_player_id), ("opponent", "opponent")):
            try:
                p = match.player_by_id(pid)
            except KeyError:
                continue
            if p.tackle_armed:
                _ep_counts[role]["tackle_armed_ticks"] += 1
            if p.kick_armed:
                _ep_counts[role]["kick_armed_ticks"] += 1

    env.definition.on_tick = _sample_armed_flags

    def _make_on_tackle_result(role: str):
        def _cb(player, tackler_won, was_tackler):  # noqa: ANN001
            # Only tally wins/losses for attempts THIS role initiated (was_tackler)
            # -- otherwise a role's win+loss total (as tacklee) inflates past its
            # own tackle_attempts count, which only counts attempts it initiated.
            if not was_tackler:
                return
            _ep_counts[role]["tackle_wins" if tackler_won else "tackle_losses"] += 1
        return _cb

    def _make_on_auto_tackle_result(role: str):
        # Auto-tackle (collision path) has no separate "armed"/"attempt"
        # callback -- on_auto_tackle_result IS the attempt signal, fired
        # once the outcome is already known, so count the attempt here too.
        def _cb(player, tackler_won, was_tackler):  # noqa: ANN001
            if not was_tackler:
                return
            _ep_counts[role]["auto_tackle_attempts"] += 1
            _ep_counts[role]["auto_tackle_wins" if tackler_won else "auto_tackle_losses"] += 1
        return _cb

    def _make_on_kick(role: str, pid: str):
        def _cb(player):  # noqa: ANN001
            _ep_counts[role]["kicks_executed"] += 1
            _record_now(player_id=pid)
        return _cb

    def _make_on_tackle(role: str, pid: str):
        def _cb(player):  # noqa: ANN001
            _ep_counts[role]["tackle_attempts"] += 1
            _record_now(player_id=pid)
        return _cb

    for ep in range(n_episodes):
        # Drawn from this worker's own --seed-derived numpy stream (already
        # seeded once at worker start -- see _run_recording_job), so the
        # WHOLE run is reproducible given the same --seed + worker index +
        # episode order, and each individual episode is ALSO independently
        # reproducible on its own via just this one int (see
        # episode_seeds/meta_episode_seeds above). int() so it round-trips
        # cleanly through np.int64 -> plain Python int -> build_1v1_
        # scenario's own `seed: int | None` param.
        episode_seed = int(np.random.randint(0, 2**31 - 1))
        env.reset(seed=episode_seed)
        for role in _ROLES:
            for k in _ep_counts[role]:
                _ep_counts[role][k] = 0
        for k in _ep_poss_reward:
            _ep_poss_reward[k] = 0.0
        # Drive trainee with rules-based AI (or a neural checkpoint, when
        # driver_trainer is given -- env.reset() just above already assigned
        # NeuralPlayerAI to it via env.sample_action_fn, so don't override
        # that here) and attach action callbacks. Callbacks fire inside
        # env.step()'s 15-tick loop regardless of which AI drives the player,
        # so episodes still terminate correctly via env.step()'s
        # box-possession / timeout checks.
        try:
            player = env._loop.match.player_by_id(env.trainee_player_id)
            if driver_trainer is None:
                player.ai = Phase1RulesAI()
            player.on_kick = _make_on_kick("trainee", env.trainee_player_id)
            player.on_tackle = _make_on_tackle("trainee", env.trainee_player_id)
            player.on_tackle_result = _make_on_tackle_result("trainee")
            player.on_auto_tackle_result = _make_on_auto_tackle_result("trainee")
        except (AttributeError, KeyError):
            pass

        # Randomise opponent: rules-based with probability opponent_rules_prob,
        # immobile otherwise (no neural opponent during demo recording).
        # on_kick/on_tackle are wired unconditionally for code simplicity — the
        # immobile branch never kicks/tackles, so the callbacks are harmless but
        # inert in that case.
        #
        # NOTE this is a SEPARATE roll from (and unconditionally OVERRIDES)
        # whatever build_1v1_scenario itself already decided internally using
        # its own seeded rng -- a real pre-existing quirk (this one folds the
        # "would-be-neural" probability mass into "rules" instead, matching
        # "no neural opponent during demo recording" above; build_1v1_
        # scenario's own internal roll instead leaves that region as ai=None,
        # relying on ScenarioEnv to assign a NeuralPlayerAI, which doesn't
        # happen during plain recording), deliberately preserved as-is rather
        # than unified, to avoid silently changing the recorded opponent-type
        # distribution. Sourced from a LOCAL rng seeded by THIS episode's own
        # seed (not the global numpy stream) so it's captured by episode_seed
        # like everything else -- scripts/replay_episode.py MUST reproduce
        # this exact second roll (same seed, same formula) after building the
        # scenario, not just call build_1v1_scenario(seed=...) alone, or a
        # replayed episode's opponent type can differ from the recorded one.
        try:
            match = env._loop.match
            opp = match.player_by_id("opponent")
            # opponent_immobile_prob (if given) lets immobile-prob be set
            # independently of rules_prob instead of implicitly = 1 - rules_prob.
            _roll = random.Random(episode_seed).random()
            if _roll < opponent_rules_prob:
                opp.ai = Phase1RulesAI()
                match._opponent_use_rules_ai = True
                match._opponent_is_immobile = False
            elif opponent_immobile_prob is not None and _roll >= opponent_rules_prob + opponent_immobile_prob:
                opp.ai = Phase1RulesAI()
                match._opponent_use_rules_ai = True
                match._opponent_is_immobile = False
            else:
                # ai stays None (Phase1RulesAI.decide() and others key off
                # `opponent.ai is None` as their "can this opponent ever
                # move/contest" signal -- see JogOrder's own docstring) --
                # but current_order is set directly so the opponent still
                # moves like a real player (accel/turn-rate/heading via
                # step_player_towards) instead of Match._apply_movement's
                # "no order this tick" branch, which now RAISES rather than
                # silently coasting (see that method's docstring) -- this
                # mirrors the identical fix already applied to
                # build_1v1_scenario's own internal immobile-opponent roll
                # in ui/scenarios.py; this is the separate, deliberately-
                # duplicated roll described above, so it needed the same
                # fix applied here too.
                opp.ai = None
                opp.current_order = JogOrder(direction=opp.velocity)
                match._opponent_use_rules_ai = False
                match._opponent_is_immobile = True
            opp.on_kick = _make_on_kick("opponent", "opponent")
            opp.on_tackle = _make_on_tackle("opponent", "opponent")
            opp.on_tackle_result = _make_on_tackle_result("opponent")
            opp.on_auto_tackle_result = _make_on_auto_tackle_result("opponent")
        except (AttributeError, KeyError):
            pass

        done = False
        last_info = None
        # Counts real decision intervals (= env.step() calls) within THIS
        # episode, so a timed sample is recorded on decision 0, then every
        # sample_every_n_decisions'th one after that -- always starts each
        # episode on the very first decision rather than carrying a
        # cross-episode phase offset.
        _decision_count = 0
        while not done:
            _do_timed_sample = (_decision_count % sample_every_n_decisions == 0)
            _decision_count += 1
            # Timed sample on every sample_every_n_decisions'th real decision
            # (reward=0 placeholder; each player's OWN real reward is
            # assigned to their row(s) below). player_id=None -> records BOTH
            # trainee and opponent -> appends 2 rows. Skipped entirely on a
            # non-sampled decision -- _recorded_ids/_recorded_row_indices
            # stay empty, so the reward/component backfill below (a zip over
            # these two lists) naturally no-ops for this iteration.
            if _do_timed_sample:
                n_before = len(rewards)
                _recorded_ids = _record_now(reward=0.0, done=False)
                # ABSOLUTE row indices, captured now (before env.step() can insert
                # anything else) -- NOT a relative/negative-offset count. on_kick/
                # on_tackle callbacks fire SYNCHRONOUSLY inside env.step() below
                # and themselves call _record_now(player_id=pid), appending
                # MORE rows to these same lists mid-call. A stale "how many did
                # I append" count combined with negative indexing (rewards[-i])
                # would then silently backfill the WRONG rows once any kick/
                # tackle happens in the same decision interval as this timed
                # sample -- confirmed in real recorded data: a trainee's genuine
                # box-possession-terminal row ended up misattributed to the
                # opponent's row (and vice versa for the loss penalty) this way.
                # Absolute indices are immune to however many extra rows a
                # callback inserts afterward.
                _recorded_row_indices = list(range(n_before, n_before + len(_recorded_ids)))
            else:
                _recorded_ids = []
                _recorded_row_indices = []
            # Advance exactly one real decision interval (the env's own,
            # unmodified decision_interval_s -- see this function's
            # docstring); kick/tackle callbacks fire inside regardless of
            # _do_timed_sample.
            _obs, _reward, done, last_info = env.step()
            # Per-player reward for this step: the trainee's own (env.step()'s
            # return) plus the opponent's own (env.last_secondary_results,
            # populated regardless of the opponent's driver type because
            # env.always_compute_secondary_reward=True was set above) — NOT
            # the same value duplicated onto both, see _record_now()'s
            # docstring for why that was wrong.
            _reward_by_pid = {env.trainee_player_id: float(_reward)}
            for _sec in env.last_secondary_results:
                _reward_by_pid[_sec["player_id"]] = float(_sec["reward"])
            # Backfill reward/done onto each row just appended for this timed
            # sample with THAT row's own player's reward AND reward_components
            # (env.last_reward_components is the trainee's own; each
            # secondary player's own breakdown lives on its
            # last_secondary_results entry — see scenario_env.py's
            # last_secondary_results docstring for why these are no longer
            # merged together. Previously every row got the SAME env-level-
            # combined dict regardless of which player it belonged to,
            # e.g. a trainee "win" row would also carry the losing
            # opponent's own loss_terminal penalty.
            _comp_by_pid = {env.trainee_player_id: env.last_reward_components}
            for _sec in env.last_secondary_results:
                _comp_by_pid[_sec["player_id"]] = _sec.get("reward_components", {})
            for _pid, _row_idx in zip(_recorded_ids, _recorded_row_indices):
                rewards[_row_idx] = np.float32(_reward_by_pid.get(_pid, _reward))
                _pid_comps = _comp_by_pid.get(_pid, {})
                reward_components[_row_idx] = np.array(
                    [_pid_comps.get(k, 0.0) for k in _comp_key_order], dtype=np.float32,
                )
                # dones=1 is NOT set here even when done=True -- it belongs
                # on the TRUE final row recorded below (_record_terminal_now,
                # via env.last_terminal_match), one real physics tick later
                # than this row. Reward/components stay here (correctly
                # attributed to the tick that earned them); dones alone
                # moves to mark the real episode boundary in the right place.
            # NOTE: previously also accrued each player's reward into
            # _pending_reward here, for a LATER kick/tackle callback's own
            # _record_now(player_id=pid, reward=None) call to pop() as ITS
            # OWN reward. That was a genuine double-count, not a deferred
            # attribution: the reward computed by THIS env.step() call was
            # ALREADY written onto this timed sample's own row two lines
            # above, unconditionally, every single call -- there was never a
            # scenario where it still needed a second home. Confirmed in
            # real recorded data: row N carried get_possession=+1.0 with a
            # real component breakdown (the genuine timed-sample event), and
            # row N+6 -- the kick-callback row six rows later, where the
            # trainee kicked the ball away -- carried ANOTHER +1.0 with an
            # EMPTY component breakdown (reward_components is only ever
            # backfilled for timed-sample rows, never callback rows) -- an
            # orphaned duplicate of the same value, silently inflating every
            # downstream MC return computed across that stretch. Removed
            # entirely; _pending_reward/_record_now's reward=None path now
            # always resolves to 0.0 (see _record_now's own docstring),
            # matching how kick/tackle-callback rows already carry
            # meaningful BC labels but zero reward of their own -- exactly
            # like the terminal row _record_terminal_now adds.
            # Accumulate reward component breakdown for periodic logging (see
            # train.py's "_comp_acc" diagnostic for the analogous pattern).
            # This periodic run-wide SUMMARY is deliberately env-level (all
            # players' components summed) -- there's no per-player breakdown
            # at this granularity (see _ep_poss_reward's own docstring
            # above) -- so explicitly sum every _comp_by_pid dict here,
            # now that env.last_reward_components alone no longer includes
            # the secondary players' own contribution (see the backfill
            # above).
            for _pid_comps in _comp_by_pid.values():
                for _k, _v in _pid_comps.items():
                    _comp_acc[_k] = _comp_acc.get(_k, 0.0) + _v
            _ep_poss_reward["poss"] += sum(c.get("poss", 0.0) for c in _comp_by_pid.values())
            _ep_poss_reward["lpos"] += sum(c.get("lpos", 0.0) for c in _comp_by_pid.values())

        # `while not done` has just exited, so done=True and env.last_terminal_match
        # holds the trial's real final tick (see ScenarioLoop.last_completed_trial_match).
        # Record it as the episode's true last row -- see _record_terminal_now's
        # own docstring for why this exists and why it's real, not reconstructed.
        if env.last_terminal_match is not None:
            for _row_idx in _record_terminal_now():
                dones[_row_idx] = np.float32(1.0)
        else:
            # Should not happen (done=True always sets last_terminal_match --
            # see ScenarioEnv.step()) but fail loudly rather than silently
            # recording an episode with no dones=1 row at all, which would
            # corrupt every downstream episode-boundary computation.
            raise AssertionError(
                "episode ended (done=True) but env.last_terminal_match is None -- "
                "ScenarioEnv.step()/ScenarioLoop.last_completed_trial_match is broken."
            )

        _comp_acc_episodes += 1
        episode_poss_reward.append(dict(_ep_poss_reward))
        _poss_reward_since_log.append(dict(_ep_poss_reward))
        _ep_counts_snapshot = {role: dict(_ep_counts[role]) for role in _ROLES}
        episode_action_counts.append(_ep_counts_snapshot)
        _ep_counts_since_log.append(_ep_counts_snapshot)

        # Track episode outcome
        outcome = getattr(last_info, "trial_outcome", None) or "unknown"
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        episode_outcomes.append(outcome)
        episode_seeds.append(episode_seed)

        global_ep = episode_offset + ep + 1
        if global_ep % 10 == 0 or global_ep == total_episodes:
            total_eps = sum(outcome_counts.values())
            parts = []
            for k in ("box_possession", "opponent_box_possession", "timeout"):
                c = outcome_counts.get(k, 0)
                pct = 100.0 * c / total_eps if total_eps else 0.0
                short = {"box_possession": "trainee_box", "opponent_box_possession": "opp_box", "timeout": "timeout"}[k]
                parts.append(f"{short}={c}({pct:.0f}%)")
            for k, c in sorted(outcome_counts.items()):
                if k not in ("box_possession", "opponent_box_possession", "timeout"):
                    parts.append(f"{k}={c}")
            log.info(
                f"Ep {global_ep}/{total_episodes} | steps: {steps_total:,} ({steps_valid:,} valid) | "
                + "  ".join(parts)
            )
            if _comp_acc_episodes > 0:
                from footballcoach.ai.ppo.ppo_trainer import REWARD_COMP_LABELS as _CL
                _key_order = {k: i for i, (k, _) in enumerate(_CL)}
                _cl_map = dict(_CL)
                _comp_sorted = sorted(_comp_acc.items(), key=lambda x: _key_order.get(x[0], 999))
                _comp_str = "  ".join(
                    f"{_cl_map.get(k, k)}={v / _comp_acc_episodes:+.2f}"
                    for k, v in _comp_sorted
                )
                log.info(
                    f"  reward breakdown (per ep, since last log, trainee+opponent): {_comp_str}"
                )
            if verbose_stats:
                log.info(
                    f"  kick/tackle samples (since last log): "
                    f"kicks={_kick_count_since_log}  tackles={_tackle_count_since_log}"
                    f"  (totals: kicks={_kick_count_total}  tackles={_tackle_count_total})"
                )
                _log_action_stats_summary(_ep_counts_since_log, label="since last log")
                _log_poss_reward_summary(_poss_reward_since_log, label="since last log")
            _comp_acc.clear()
            _comp_acc_episodes = 0
            _kick_count_since_log = 0
            _tackle_count_since_log = 0
            _ep_counts_since_log.clear()
            _poss_reward_since_log.clear()

    env.definition.on_tick = _orig_on_tick  # restore original

    return {
        "obs_self_feat":   np.stack(self_feats).astype(np.float32),
        "obs_other_feat":  np.stack(other_feats).astype(np.float32),
        "obs_exists_mask": np.stack(exists_masks).astype(np.float32),
        "obs_ball_feat":   np.stack(ball_feats).astype(np.float32),
        "obs_global_feat": np.stack(global_feats).astype(np.float32),
        "bc_labels":       np.stack(bc_labels).astype(np.float32),
        "rewards":         np.array(rewards, dtype=np.float32),
        "is_trainee":      np.array(is_trainee_flags, dtype=np.float32),
        "is_decision_step": np.array(is_decision_step_flags, dtype=np.float32),
        "dones":           np.array(dones, dtype=np.float32),
        "reward_components": np.stack(reward_components).astype(np.float32),
        "meta_reward_component_keys": np.array(_comp_key_order),
        "meta_phase":      np.array(phase_id, dtype=np.int32),
        "meta_scenario":   np.bytes_(scenario_key),
        "meta_episode_action_counts": np.array(
            [
                [ep[role][k] for role in _ROLES for k in _ACTION_COUNT_KEYS]
                for ep in episode_action_counts
            ],
            dtype=np.int64,
        ),
        # Flattened "role.key" column labels, matching meta_episode_action_counts'
        # column order -- per-role dicts can't round-trip through .npz directly.
        "meta_episode_action_count_keys": np.array(
            [f"{role}.{k}" for role in _ROLES for k in _ACTION_COUNT_KEYS]
        ),
        "meta_episode_poss_reward": np.array(
            [[ep["poss"], ep["lpos"]] for ep in episode_poss_reward], dtype=np.float32
        ),
        # Ground-truth per-episode outcome strings (see episode_outcomes
        # comment above) -- one entry per complete episode, same order as
        # `dones`' done=1 rows.
        "meta_episode_outcomes": np.array(episode_outcomes, dtype="U32"),
        # Per-episode seed (see episode_seeds comment above) -- same order/
        # length as meta_episode_outcomes, one int per complete episode.
        "meta_episode_seeds": np.array(episode_seeds, dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# Batch-loop worker (shared by single-process main() and each subprocess)
# ---------------------------------------------------------------------------

def _run_recording_job(job: dict) -> dict:
    """Record ``job['n_episodes']`` episodes in ``episodes_per_file``-sized
    .npz files, starting from ``job['file_idx_start']``.

    Must stay top-level/picklable -- used both directly (single-process path)
    and as the target of a ``multiprocessing`` worker (see
    ``--n-processes``/``bc.demo_recording_n_processes``). Each job carries
    checkpoint PATHS (``driver_checkpoint``/``teacher_checkpoint``), not
    loaded trainer objects -- torch modules aren't reliably picklable across
    a spawn-context Pool boundary, so each worker loads its own copy here
    (once per job, not per episode/file). Otherwise fully self-contained --
    each job just needs its own RNG seed (to avoid identical episodes across
    workers) and its own disjoint file_idx range (to avoid filename
    collisions); no weight sync is needed even with a checkpoint involved,
    unlike ai/ppo/rollout_worker.py, since nothing here is being trained.
    """
    import random as _random
    np.random.seed(job["seed"])
    _random.seed(job["seed"])

    env, label_fn, scenario_key = _build_env_and_label_fn(job["phase_id"])

    driver_trainer = None
    if job.get("driver_checkpoint") or job.get("teacher_checkpoint"):
        # Explicit redundant safeguard on top of the module-level
        # OMP_NUM_THREADS-etc guard above -- torch's own internal thread pool
        # doesn't reliably pick up those env vars in every build, so cap it
        # directly too (mirrors debug_value_network.py's rollout workers).
        import torch as _torch
        _torch.set_num_threads(max(1, int(job.get("worker_torch_threads", 1))))

    if job.get("driver_checkpoint"):
        from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
        driver_trainer = PPOTrainer.load_for_inference(job["driver_checkpoint"])
        log.info(f"[worker {job.get('worker_idx', 0)}] Driver checkpoint loaded: {job['driver_checkpoint']}")

    if job.get("teacher_checkpoint"):
        from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
        from footballcoach.ai.ppo.bc import phase1_labels_from_teacher
        # Reuse the driver trainer instead of loading a second copy when both
        # checkpoints are identical -- the common "distill this checkpoint
        # against itself, on-policy" case.
        if job.get("driver_checkpoint") == job["teacher_checkpoint"] and driver_trainer is not None:
            teacher_trainer = driver_trainer
        else:
            teacher_trainer = PPOTrainer.load_for_inference(job["teacher_checkpoint"])
        log.info(f"[worker {job.get('worker_idx', 0)}] Teacher checkpoint loaded: {job['teacher_checkpoint']}")
        label_fn = lambda env, player_id=None, _t=teacher_trainer: phase1_labels_from_teacher(env, _t, player_id)

    n_eps = job["n_episodes"]
    eps_per_file = job["episodes_per_file"]
    file_idx = job["file_idx_start"]
    remaining = n_eps
    episodes_done = 0
    total_steps = 0
    n_files_written = 0
    all_episode_action_counts: list[dict[str, int]] = []
    all_episode_poss_reward: list[dict[str, float]] = []

    while remaining > 0:
        batch = min(eps_per_file, remaining)
        t0 = time.time()
        data = record_episodes(
            env=env,
            label_fn=label_fn,
            n_episodes=batch,
            scenario_key=scenario_key,
            phase_id=job["phase_id"],
            episode_offset=episodes_done,
            total_episodes=n_eps,
            sample_every_n_decisions=job["sample_every_n_decisions"],
            opponent_rules_prob=job["opponent_rules_prob"],
            opponent_immobile_prob=job["opponent_immobile_prob"],
            verbose_stats=job.get("verbose_stats", False),
            driver_trainer=driver_trainer,
        )
        elapsed = time.time() - t0

        n_steps = len(data["bc_labels"])
        total_steps += n_steps
        # Unflatten "role.key" columns (see record_episodes()'s
        # meta_episode_action_count_keys comment) back into per-role dicts.
        _flat_keys = list(data["meta_episode_action_count_keys"])
        for row in data["meta_episode_action_counts"]:
            _ep_dict: dict[str, dict[str, int]] = {}
            for flat_k, v in zip(_flat_keys, row.tolist()):
                role, _, k = str(flat_k).partition(".")
                _ep_dict.setdefault(role, {})[k] = v
            all_episode_action_counts.append(_ep_dict)
        for poss, lpos in data["meta_episode_poss_reward"]:
            all_episode_poss_reward.append({"poss": float(poss), "lpos": float(lpos)})

        fname = Path(job["output_dir"]) / f"phase{job['phase_id']}_{file_idx:04d}.npz"
        np.savez_compressed(fname, **data)

        log.info(
            f"[worker {job.get('worker_idx', 0)}] Saved {fname.name} | "
            f"{batch} episodes, {n_steps} steps | {elapsed:.1f}s"
        )

        file_idx += 1
        n_files_written += 1
        remaining -= batch
        episodes_done += batch

    if job.get("verbose_stats", False):
        _log_action_stats_summary(all_episode_action_counts)
        _log_poss_reward_summary(all_episode_poss_reward)

    return {"total_steps": total_steps, "n_files": n_files_written}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record rules-based AI demonstrations for BC pre-training"
    )
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--n-episodes", type=int, default=200,
                        help="Total episodes to record (default: 200)")
    parser.add_argument("--episodes-per-file", type=int, default=50,
                        help="Episodes per output .npz file (default: 8)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for .npz files")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed. Default: None -- draws a fresh random seed each run "
                             "(logged so the run can be reproduced later). Pass an explicit "
                             "value for reproducible/comparable recordings across runs. You "
                             "don't need to remember this to replay one specific episode later "
                             "though -- every recorded episode's own build_1v1_scenario seed is "
                             "saved individually (meta_episode_seeds), so "
                             "scripts/replay_episode.py only needs THAT one int, not this "
                             "whole-run --seed.")
    _cfg = __import__("footballcoach.ai.config", fromlist=["load_ai_config"]).load_ai_config()
    _default_sample_every_n = int(_cfg.get("bc", {}).get("demo_sample_every_n_decisions", 1))
    parser.add_argument("--sample-every-n-decisions", type=int, default=_default_sample_every_n,
                        help=f"Record a timed sample every this many real decision intervals "
                             f"(default: {_default_sample_every_n}; 1 = every decision). Always "
                             "lands exactly on a genuine decision -- kicks and tackles are always "
                             "recorded via callbacks regardless.")
    _demo_curr = _cfg.get("curriculum", {})
    _demo_rules_r = float(_demo_curr.get("phase1_demo_opponent_rules_ratio", 1.0))
    _demo_immobile_r = float(_demo_curr.get("phase1_demo_opponent_immobile_ratio", 1.0))
    _demo_neural_r = float(_demo_curr.get("phase1_demo_opponent_neural_ratio", 0.0))
    _demo_total = _demo_rules_r + _demo_immobile_r + _demo_neural_r
    _default_opp_rules_prob = (_demo_rules_r / _demo_total) if _demo_total > 0 else 0.5
    _default_opp_immobile_prob = (_demo_immobile_r / _demo_total) if _demo_total > 0 else 0.5
    parser.add_argument("--opponent-rules-prob", type=float, default=_default_opp_rules_prob,
                        help=f"Probability (0–1) that the opponent uses the rules-based AI each "
                             f"demo episode (default: {_default_opp_rules_prob:.2f} from config ratios). "
                             "Remainder are immobile, unless --opponent-immobile-prob is also given.")
    parser.add_argument("--opponent-immobile-prob", type=float, default=_default_opp_immobile_prob,
                        help=f"Probability (0–1) that the opponent is immobile (default: {_default_opp_immobile_prob:.2f} from config ratios).")
    parser.add_argument("--verbose-stats", action="store_true",
                        help="Print per-log-interval kick/tackle/possession detail stats (noisy; off by default)")
    parser.add_argument("--driver-checkpoint", type=str, default=None,
                        help="Path to a trained checkpoint (.pt). When given, the TRAINEE is "
                             "driven by this checkpoint's own policy instead of Phase1RulesAI, "
                             "so recorded states reflect this policy's own on-policy state "
                             "visitation. Independent of --teacher-checkpoint (may be the same "
                             "path, a different checkpoint, or omitted to keep BC labels from "
                             "Phase1RulesAI as before). The opponent is unaffected -- still "
                             "governed by --opponent-rules-prob/--opponent-immobile-prob.")
    parser.add_argument("--teacher-checkpoint", type=str, default=None,
                        help="Path to a trained checkpoint (.pt). When given, BC labels are "
                             "read directly from this checkpoint's decision/execution network "
                             "forward pass (soft probabilities, every head) instead of "
                             "Phase1RulesAI's order-simulation counterfactual -- see "
                             "footballcoach.ai.ppo.bc.phase1_labels_from_teacher(). Independent "
                             "of --driver-checkpoint: this only changes what supervises each "
                             "recorded state, not which states get visited.")
    parser.add_argument("--info", action="store_true",
                        help="Print info about existing files and exit")
    _default_n_processes = int(_cfg.get("bc", {}).get("demo_recording_n_processes", 1))
    parser.add_argument("--n-processes", type=int, default=_default_n_processes,
                        help=f"Number of worker processes to split --n-episodes across "
                             f"(default: {_default_n_processes} from config). 1 = current "
                             "single-process behaviour. Unlike PPO's --n-parallel-envs there's "
                             "no weight sync -- each worker just records its own share of "
                             "episodes into its own disjoint .npz file range (and, with "
                             "--driver-checkpoint/--teacher-checkpoint, loads its own copy of "
                             "the checkpoint independently).")
    _default_worker_torch_threads = int(_cfg.get("bc", {}).get("demo_worker_torch_threads", 1))
    parser.add_argument("--worker-torch-threads", type=int, default=_default_worker_torch_threads,
                        help=f"torch.set_num_threads() per worker (default: "
                             f"{_default_worker_torch_threads} from config). Only matters with "
                             "--driver-checkpoint/--teacher-checkpoint (no torch is loaded "
                             "otherwise). Keep this at 1 when --n-processes > 1 -- torch defaults "
                             "to one thread per logical core, so N processes each left uncapped "
                             "oversubscribes badly; this is on top of the module-level "
                             "OMP_NUM_THREADS-etc guard, which caps numpy/BLAS the same way.")
    args = parser.parse_args()

    import random
    if args.seed is None:
        args.seed = random.SystemRandom().randrange(2**31)
        log.info(f"--seed not given: drew random seed {args.seed} for this run")
    np.random.seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output)

    if args.info:
        _print_info(output_dir)
        return

    if args.n_episodes <= 0:
        log.info("n-episodes=0, nothing to record.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume: start file indices after the highest existing file so we never overwrite.
    import re as _re
    _existing = [
        int(m.group(1))
        for p in output_dir.glob(f"phase{args.phase}_*.npz")
        if (m := _re.search(r"_(\d+)\.npz$", p.name))
    ]
    _file_idx_offset = (max(_existing) + 1) if _existing else 0
    if _file_idx_offset > 0:
        log.warning(
            f"Output dir already contains {len(_existing)} file(s) (up to index {_file_idx_offset - 1}). "
            f"Appending new files from index {_file_idx_offset} — existing files will NOT be overwritten."
        )

    n_eps = args.n_episodes
    eps_per_file = args.episodes_per_file
    n_files = (n_eps + eps_per_file - 1) // eps_per_file
    n_processes = max(1, args.n_processes)

    if n_processes == 1:
        _, _, scenario_key = _build_env_and_label_fn(args.phase)
        log.info(
            f"Recording {n_eps} episodes of phase {args.phase} ({scenario_key}) "
            f"→ {n_files} file(s) in {output_dir} "
            f"[sample_every_n_decisions={args.sample_every_n_decisions}, opponent_rules_prob={args.opponent_rules_prob:.0%}]"
        )
        result = _run_recording_job({
            "phase_id": args.phase,
            "n_episodes": n_eps,
            "episodes_per_file": eps_per_file,
            "output_dir": str(output_dir),
            "file_idx_start": _file_idx_offset,
            "verbose_stats": args.verbose_stats,
            "seed": args.seed,
            "sample_every_n_decisions": args.sample_every_n_decisions,
            "opponent_rules_prob": args.opponent_rules_prob,
            "opponent_immobile_prob": args.opponent_immobile_prob,
            "driver_checkpoint": args.driver_checkpoint,
            "teacher_checkpoint": args.teacher_checkpoint,
            "worker_torch_threads": args.worker_torch_threads,
        })
        log.info(
            f"Done. {result['n_files']} file(s), {result['total_steps']:,} total steps → {output_dir}"
        )
        return

    # --- Multi-process path: split n_episodes evenly, each worker gets its
    # own disjoint file_idx range (via cumulative n_files-per-worker) and its
    # own RNG seed so workers don't record identical episodes. ---
    import multiprocessing as mp

    _, _, scenario_key = _build_env_and_label_fn(args.phase)
    base_eps = n_eps // n_processes
    remainder = n_eps % n_processes
    jobs: list[dict] = []
    file_idx_cursor = _file_idx_offset
    for i in range(n_processes):
        worker_eps = base_eps + (1 if i < remainder else 0)
        if worker_eps == 0:
            continue
        jobs.append({
            "phase_id": args.phase,
            "n_episodes": worker_eps,
            "episodes_per_file": eps_per_file,
            "output_dir": str(output_dir),
            "file_idx_start": file_idx_cursor,
            "verbose_stats": args.verbose_stats,
            "seed": args.seed + i,
            "sample_every_n_decisions": args.sample_every_n_decisions,
            "opponent_rules_prob": args.opponent_rules_prob,
            "opponent_immobile_prob": args.opponent_immobile_prob,
            "driver_checkpoint": args.driver_checkpoint,
            "teacher_checkpoint": args.teacher_checkpoint,
            "worker_torch_threads": args.worker_torch_threads,
            "worker_idx": i,
        })
        file_idx_cursor += (worker_eps + eps_per_file - 1) // eps_per_file

    log.info(
        f"Recording {n_eps} episodes of phase {args.phase} ({scenario_key}) "
        f"across {len(jobs)} process(es) → {n_files} file(s) in {output_dir} "
        f"[sample_every_n_decisions={args.sample_every_n_decisions}, opponent_rules_prob={args.opponent_rules_prob:.0%}]"
    )

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(jobs)) as pool:
        results = pool.map(_run_recording_job, jobs)

    total_steps = sum(r["total_steps"] for r in results)
    total_files = sum(r["n_files"] for r in results)
    log.info(
        f"Done. {total_files} file(s), {total_steps:,} total steps → {output_dir}"
    )


def _print_info(directory: Path) -> None:
    """Print summary of existing .npz files in directory."""
    files = sorted(directory.glob("*.npz")) if directory.exists() else []
    if not files:
        print(f"No .npz files found in {directory}")
        return
    total_steps = 0
    valid_steps = 0
    for f in files:
        data = np.load(f)
        n = len(data["bc_labels"])
        v = int((data["bc_labels"][:, -1] > 0.5).sum())
        total_steps += n
        valid_steps += v
        print(f"  {f.name}: {n} steps ({v} valid)")
    print(f"Total: {len(files)} files, {total_steps:,} steps ({valid_steps:,} valid)")


if __name__ == "__main__":
    main()
