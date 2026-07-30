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

    steps_total = 0
    steps_valid = 0

    # Outcome counters
    outcome_counts: dict[str, int] = {}

    if total_episodes is None:
        total_episodes = n_episodes

    def _record_now():
        obs = env._get_obs()
        label = label_fn(env)
        label_arr = label.to_array()
        self_feats.append(obs.self_feat.copy())
        other_feats.append(obs.other_feat.copy())
        exists_masks.append(obs.exists_mask.copy())
        ball_feats.append(obs.ball_feat.copy())
        global_feats.append(obs.global_feat.copy())
        bc_labels.append(label_arr)
        nonlocal steps_total, steps_valid
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
            player.on_kick = lambda p: _record_now()
            player.on_tackle = lambda p: _record_now()
        except (AttributeError, KeyError):
            pass

        done = False
        last_info = None
        while not done:
            # Timed sample at sample_interval_s cadence
            _record_now()
            # Advance sim by sample_interval_s; kick/tackle callbacks fire inside
            _obs, _reward, done, last_info = env.step()

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

    env._ticks_per_decision = orig_ticks  # restore original

    return {
        "obs_self_feat":   np.stack(self_feats).astype(np.float32),
        "obs_other_feat":  np.stack(other_feats).astype(np.float32),
        "obs_exists_mask": np.stack(exists_masks).astype(np.float32),
        "obs_ball_feat":   np.stack(ball_feats).astype(np.float32),
        "obs_global_feat": np.stack(global_feats).astype(np.float32),
        "bc_labels":       np.stack(bc_labels).astype(np.float32),
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
        f"[sample_interval={args.sample_interval}s]"
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
