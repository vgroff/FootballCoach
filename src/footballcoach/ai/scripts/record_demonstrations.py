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
    obs_global_feat, bc_labels, meta_phase, meta_scenario

Each file is self-contained and can be loaded individually or combined with
DemonstrationDataset.from_directory().
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

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
    sample_interval_s: float = 0.2,
    opponent_rules_prob: float = 0.0,
    opponent_immobile_prob: float | None = None,
    verbose_stats: bool = False,
) -> dict:
    """Run *n_episodes* with rules-based AI driving the trainee and opponent.

    Sampling strategy:
      - on_kick / on_tackle player callbacks fire at the exact engine tick the
        action executes → always recorded regardless of sample interval.
      - Time-based sampling at *sample_interval_s* cadence (default 0.2s),
        independent of the neural-network decision interval (0.5s).
      - env.step() handles all terminal conditions normally (box possession,
        timeout) so episodes end correctly.

    Args:
        sample_interval_s: How often (in sim-seconds) to record a timed sample.
            Kicks and tackles are always recorded via callbacks regardless.
            Default is 0.2s.

    Returns a dict of numpy arrays ready to be saved as .npz.
    """
    from footballcoach.rules_ai import Phase1RulesAI
    from footballcoach.ai.ppo.ppo_trainer import REWARD_COMP_LABELS

    # Override the env's ticks-per-decision to match sample_interval_s so that
    # env.step() advances by sample_interval_s and we sample at that cadence.
    # Kicks/tackle callbacks are unaffected (they fire inside each env.step()).
    orig_ticks = env._ticks_per_decision
    sample_ticks = max(1, round(sample_interval_s / env._dt_s))
    env._ticks_per_decision = sample_ticks

    self_feats = []
    other_feats = []
    exists_masks = []
    ball_feats = []
    global_feats = []
    bc_labels = []
    rewards = []
    dones = []
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
    # sample_interval_s.
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

    # Mutable cell holding the reward accrued since the last time it was
    # consumed by a sample (timed sample or kick/tackle callback). Cleared to
    # 0.0 every time it's read so reward is never double-counted across rows,
    # and never silently dropped when a kick/tackle callback fires between
    # timed samples (see the main loop below, which accrues into this cell
    # after each env.step()).
    _pending_reward: list[float] = [0.0]

    def _record_now(reward: float | None = None, done: bool = False, player_id: str | None = None):
        """Append one (obs, label) sample per player.

        player_id=None (used for timed samples) records BOTH the trainee and
        the opponent in one call. A specific player_id (used for on_kick/
        on_tackle callback samples, which fire per-player) records only that
        player.

        reward=None (the default, used by kick/tackle callbacks) consumes and
        clears whatever reward has accrued since the last sample via
        ``_pending_reward`` — previously this was hardcoded to 0.0, silently
        dropping the actual step reward (e.g. gain_possession_bonus) that
        fired at exactly the kick/tackle tick. Timed samples explicitly pass
        reward=0.0 and get their real reward backfilled after env.step().
        """
        ids = [env.trainee_player_id, "opponent"] if player_id is None else [player_id]
        nonlocal steps_total, steps_valid
        nonlocal _kick_count_since_log, _tackle_count_since_log, _kick_count_total, _tackle_count_total
        if reward is None:
            reward = _pending_reward[0]
            _pending_reward[0] = 0.0
        for pid in ids:
            obs = env._get_obs(player_id=pid)
            label = label_fn(env, player_id=pid)
            label_arr = label.to_array()
            self_feats.append(obs.self_feat.copy())
            other_feats.append(obs.other_feat.copy())
            exists_masks.append(obs.exists_mask.copy())
            ball_feats.append(obs.ball_feat.copy())
            global_feats.append(obs.global_feat.copy())
            bc_labels.append(label_arr)
            rewards.append(np.float32(reward))
            dones.append(np.float32(done))
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
        env.reset()
        for role in _ROLES:
            for k in _ep_counts[role]:
                _ep_counts[role][k] = 0
        for k in _ep_poss_reward:
            _ep_poss_reward[k] = 0.0

        # Drive trainee with rules-based AI and attach action callbacks.
        # Callbacks fire inside env.step()'s 15-tick loop, so episodes still
        # terminate correctly via env.step()'s box-possession / timeout checks.
        try:
            player = env._loop.match.player_by_id(env.trainee_player_id)
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
        try:
            match = env._loop.match
            opp = match.player_by_id("opponent")
            # opponent_immobile_prob (if given) lets immobile-prob be set
            # independently of rules_prob instead of implicitly = 1 - rules_prob.
            _roll = np.random.random()
            if _roll < opponent_rules_prob:
                opp.ai = Phase1RulesAI()
                match._opponent_use_rules_ai = True
                match._opponent_is_immobile = False
            elif opponent_immobile_prob is not None and _roll >= opponent_rules_prob + opponent_immobile_prob:
                opp.ai = Phase1RulesAI()
                match._opponent_use_rules_ai = True
                match._opponent_is_immobile = False
            else:
                opp.ai = None
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
        while not done:
            # Timed sample at sample_interval_s cadence (reward=0 for mid-step samples;
            # the actual reward from env.step() is assigned to the NEXT sample(s) or
            # appended to the episode-end step below).
            # player_id=None -> records BOTH trainee and opponent -> appends 2 rows.
            n_before = len(rewards)
            _record_now(reward=0.0, done=False)
            n_appended = len(rewards) - n_before
            # Advance sim by sample_interval_s; kick/tackle callbacks fire inside
            _obs, _reward, done, last_info = env.step()
            # Backfill reward/done/components onto every row just appended for
            # this timed sample (trainee + opponent both share the env-level
            # reward/done/components — there is no separate per-player signal
            # at this granularity).
            _comp_row = np.array(
                [env.last_reward_components.get(k, 0.0) for k in _comp_key_order],
                dtype=np.float32,
            )
            for i in range(1, n_appended + 1):
                rewards[-i] = np.float32(_reward)
                reward_components[-i] = _comp_row
                if done:
                    dones[-i] = np.float32(1.0)
            # Accrue this step's reward for the NEXT kick/tackle callback (if
            # any) that fires before the next timed sample — see _record_now.
            _pending_reward[0] += float(_reward)
            # Accumulate reward component breakdown for periodic logging (see
            # train.py's "_comp_acc" diagnostic for the analogous pattern).
            for _k, _v in env.last_reward_components.items():
                _comp_acc[_k] = _comp_acc.get(_k, 0.0) + _v
            _ep_poss_reward["poss"] += env.last_reward_components.get("poss", 0.0)
            _ep_poss_reward["lpos"] += env.last_reward_components.get("lpos", 0.0)

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
    env._ticks_per_decision = orig_ticks  # restore original

    return {
        "obs_self_feat":   np.stack(self_feats).astype(np.float32),
        "obs_other_feat":  np.stack(other_feats).astype(np.float32),
        "obs_exists_mask": np.stack(exists_masks).astype(np.float32),
        "obs_ball_feat":   np.stack(ball_feats).astype(np.float32),
        "obs_global_feat": np.stack(global_feats).astype(np.float32),
        "bc_labels":       np.stack(bc_labels).astype(np.float32),
        "rewards":         np.array(rewards, dtype=np.float32),
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
    }


# ---------------------------------------------------------------------------
# Batch-loop worker (shared by single-process main() and each subprocess)
# ---------------------------------------------------------------------------

def _run_recording_job(job: dict) -> dict:
    """Record ``job['n_episodes']`` episodes in ``episodes_per_file``-sized
    .npz files, starting from ``job['file_idx_start']``.

    Must stay top-level/picklable -- used both directly (single-process path)
    and as the target of a ``multiprocessing`` worker (see
    ``--n-processes``/``bc.demo_recording_n_processes``). Independent of any
    neural network, so unlike ai/ppo/rollout_worker.py there is no weight
    sync needed -- each job is fully self-contained and just needs its own
    RNG seed (to avoid identical episodes across workers) and its own
    disjoint file_idx range (to avoid filename collisions).
    """
    import random as _random
    np.random.seed(job["seed"])
    _random.seed(job["seed"])

    env, label_fn, scenario_key = _build_env_and_label_fn(job["phase_id"])

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
            sample_interval_s=job["sample_interval_s"],
            opponent_rules_prob=job["opponent_rules_prob"],
            opponent_immobile_prob=job["opponent_immobile_prob"],
            verbose_stats=job.get("verbose_stats", False),
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
    parser.add_argument("--episodes-per-file", type=int, default=8,
                        help="Episodes per output .npz file (default: 8)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for .npz files")
    parser.add_argument("--seed", type=int, default=42)
    _cfg = __import__("footballcoach.ai.config", fromlist=["load_ai_config"]).load_ai_config()
    _default_interval = float(_cfg.get("bc", {}).get("demo_sample_interval_s", 0.2))
    parser.add_argument("--sample-interval", type=float, default=_default_interval,
                        help=f"Sim-seconds between timed samples (default: {_default_interval}). "
                             "Kicks and tackles are always recorded regardless.")
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
    parser.add_argument("--info", action="store_true",
                        help="Print info about existing files and exit")
    _default_n_processes = int(_cfg.get("bc", {}).get("demo_recording_n_processes", 1))
    parser.add_argument("--n-processes", type=int, default=_default_n_processes,
                        help=f"Number of worker processes to split --n-episodes across "
                             f"(default: {_default_n_processes} from config). 1 = current "
                             "single-process behaviour. No neural network is involved in "
                             "recording, so unlike PPO's --n-parallel-envs there's no weight "
                             "sync -- each worker just records its own share of episodes into "
                             "its own disjoint .npz file range.")
    args = parser.parse_args()

    import random
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
            f"[sample_interval={args.sample_interval}s, opponent_rules_prob={args.opponent_rules_prob:.0%}]"
        )
        result = _run_recording_job({
            "phase_id": args.phase,
            "n_episodes": n_eps,
            "episodes_per_file": eps_per_file,
            "output_dir": str(output_dir),
            "file_idx_start": _file_idx_offset,
            "verbose_stats": args.verbose_stats,
            "seed": args.seed,
            "sample_interval_s": args.sample_interval,
            "opponent_rules_prob": args.opponent_rules_prob,
            "opponent_immobile_prob": args.opponent_immobile_prob,
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
            "sample_interval_s": args.sample_interval,
            "opponent_rules_prob": args.opponent_rules_prob,
            "opponent_immobile_prob": args.opponent_immobile_prob,
            "worker_idx": i,
        })
        file_idx_cursor += (worker_eps + eps_per_file - 1) // eps_per_file

    log.info(
        f"Recording {n_eps} episodes of phase {args.phase} ({scenario_key}) "
        f"across {len(jobs)} process(es) → {n_files} file(s) in {output_dir} "
        f"[sample_interval={args.sample_interval}s, opponent_rules_prob={args.opponent_rules_prob:.0%}]"
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
