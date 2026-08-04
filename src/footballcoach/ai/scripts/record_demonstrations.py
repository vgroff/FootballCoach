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

    steps_total = 0
    steps_valid = 0

    # Outcome counters
    outcome_counts: dict[str, int] = {}

    # Reward component breakdown accumulator (mirrors train.py's diagnostic
    # "_comp_acc" pattern), reset after each periodic log line so the
    # printed line reflects the average since the last log, not the whole run.
    _comp_acc: dict[str, float] = {}
    _comp_acc_episodes = 0

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
            steps_total += 1
            if label.valid:
                steps_valid += 1

    for ep in range(n_episodes):
        env.reset()

        # Drive trainee with rules-based AI and attach action callbacks.
        # Callbacks fire inside env.step()'s 15-tick loop, so episodes still
        # terminate correctly via env.step()'s box-possession / timeout checks.
        try:
            player = env._loop.match.player_by_id(env.trainee_player_id)
            player.ai = Phase1RulesAI()
            player.on_kick = lambda p: _record_now(player_id=env.trainee_player_id)
            player.on_tackle = lambda p: _record_now(player_id=env.trainee_player_id)
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
            opp.on_kick = lambda p: _record_now(player_id="opponent")
            opp.on_tackle = lambda p: _record_now(player_id="opponent")
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
            # Backfill reward/done onto every row just appended for this timed
            # sample (trainee + opponent both share the env-level reward/done —
            # there is no separate per-player reward signal at this granularity).
            for i in range(1, n_appended + 1):
                rewards[-i] = np.float32(_reward)
                if done:
                    dones[-i] = np.float32(1.0)
            # Accrue this step's reward for the NEXT kick/tackle callback (if
            # any) that fires before the next timed sample — see _record_now.
            _pending_reward[0] += float(_reward)
            # Accumulate reward component breakdown for periodic logging (see
            # train.py's "_comp_acc" diagnostic for the analogous pattern).
            for _k, _v in env.last_reward_components.items():
                _comp_acc[_k] = _comp_acc.get(_k, 0.0) + _v

        _comp_acc_episodes += 1

        # Track episode outcome
        outcome = getattr(last_info, "trial_outcome", None) or "unknown"
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

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
            _comp_acc.clear()
            _comp_acc_episodes = 0

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
        "meta_phase":      np.array(phase_id, dtype=np.int32),
        "meta_scenario":   np.bytes_(scenario_key),
    }


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
    _default_opp_rules_prob = float(_cfg.get("curriculum", {}).get("phase1_demo_opponent_rules_prob", 0.3))
    parser.add_argument("--opponent-rules-prob", type=float, default=_default_opp_rules_prob,
                        help=f"Probability (0–1) that the opponent uses the rules-based AI each "
                             f"demo episode (default: {_default_opp_rules_prob} from config). "
                             "Remainder are immobile, unless --opponent-immobile-prob is also given.")
    _default_opp_immobile_prob = float(_cfg.get("curriculum", {}).get("phase1_demo_opponent_immobile_prob", 0.0))
    parser.add_argument("--opponent-immobile-prob", type=float, default=_default_opp_immobile_prob,
                        help=f"Probability (0–1) that the opponent is immobile, independent of "
                             f"--opponent-rules-prob (default: {_default_opp_immobile_prob} from config). "
                             "0.0 (default) preserves old behaviour: remainder after rules-prob is immobile.")
    parser.add_argument("--info", action="store_true",
                        help="Print info about existing files and exit")
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
    env, label_fn, scenario_key = _build_env_and_label_fn(args.phase)

    n_eps = args.n_episodes
    eps_per_file = args.episodes_per_file
    n_files = (n_eps + eps_per_file - 1) // eps_per_file

    log.info(
        f"Recording {n_eps} episodes of phase {args.phase} ({scenario_key}) "
        f"→ {n_files} file(s) in {output_dir} "
        f"[sample_interval={args.sample_interval}s, opponent_rules_prob={args.opponent_rules_prob:.0%}]"
    )

    total_steps = 0
    file_idx = 0
    remaining = n_eps
    episodes_done = 0

    while remaining > 0:
        batch = min(eps_per_file, remaining)
        t0 = time.time()
        data = record_episodes(
            env=env,
            label_fn=label_fn,
            n_episodes=batch,
            scenario_key=scenario_key,
            phase_id=args.phase,
            episode_offset=episodes_done,
            total_episodes=n_eps,
            sample_interval_s=args.sample_interval,
            opponent_rules_prob=args.opponent_rules_prob,
            opponent_immobile_prob=args.opponent_immobile_prob,
        )
        elapsed = time.time() - t0

        n_steps = len(data["bc_labels"])
        total_steps += n_steps

        fname = output_dir / f"phase{args.phase}_{file_idx:04d}.npz"
        np.savez_compressed(fname, **data)

        log.info(
            f"Saved {fname.name} | {batch} episodes, {n_steps} steps | {elapsed:.1f}s"
        )

        file_idx += 1
        remaining -= batch
        episodes_done += batch

    log.info(
        f"Done. {file_idx} file(s), {total_steps:,} total steps → {output_dir}"
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
