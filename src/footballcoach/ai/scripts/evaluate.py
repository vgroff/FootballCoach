"""Evaluation script: run N trials headless, report statistics.

Mirrors the reporting style of tests/balance/ (full stats, not just pass/fail).
Results are written to a JSON file for later inspection.

Usage:
    uv run python -m footballcoach.ai.scripts.evaluate \\
        --checkpoint checkpoints/checkpoint_00500000.pt \\
        --phase 2 --n-trials 200 --output results/eval_500k.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("footballcoach.ai.evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained football AI checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint .pt file (omit with --baseline-only)")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Curriculum phase to evaluate (default: 1)")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Number of episodes to evaluate (default: 100)")
    parser.add_argument("--output", type=str, default="results/eval_latest.json",
                        help="Output JSON file for results")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip the rules-vs-rules baseline evaluation (baseline runs by default).")
    parser.add_argument("--baseline-trials", type=int, default=40,
                        help="Number of episodes for the rules-vs-rules baseline (default: 40).")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Run only the rules-vs-rules baseline (no checkpoint required).")
    args = parser.parse_args()

    if not args.baseline_only and args.checkpoint is None:
        parser.error("--checkpoint is required unless --baseline-only is set")

    import torch
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)

    from footballcoach.ai.curriculum.phases import PHASES_BY_ID
    from footballcoach.ai.curriculum.envs import build_env

    phase = PHASES_BY_ID.get(args.phase)
    if phase is None:
        log.error(f"Unknown phase: {args.phase}")
        return

    # Baseline-only mode: no checkpoint needed
    if args.baseline_only:
        log.info(f"Running rules-vs-rules baseline ({args.baseline_trials} episodes, phase={args.phase})...")
        baseline_stats = _run_baseline_evaluation(None, args.baseline_trials)
        combined = {
            "checkpoint": None,
            "checkpoint_step": 0,
            "phase": args.phase,
            "n_trials": 0,
            "neural_net": None,
            "baseline_rules_vs_rules": baseline_stats,
        }
        bl = baseline_stats
        log.info(f"  Rules baseline: win={bl['win_rate_pct']:.1f}%  outcomes={bl['outcomes']}")
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(combined, f, indent=2)
        log.info(f"Results written to {out_path}")
        return

    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer

    # Load trainer + checkpoint
    trainer = PPOTrainer.from_config(device=device)
    step = trainer.load_checkpoint(Path(args.checkpoint))

    # Build env (same as training)
    env = build_env(phase)

    # Run neural-net evaluation
    log.info(f"Evaluating {args.n_trials} episodes (phase={phase.name}, step={step:,})")
    neural_stats = _run_evaluation(trainer, env, args.n_trials)

    # Run rules-vs-rules baseline (unless suppressed)
    baseline_stats: dict | None = None
    if not args.no_baseline and args.phase == 1:
        log.info(f"Running rules-vs-rules baseline ({args.baseline_trials} episodes)...")
        baseline_stats = _run_baseline_evaluation(env, args.baseline_trials)

    # Combine results
    combined = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": step,
        "phase": args.phase,
        "n_trials": args.n_trials,
        "neural_net": neural_stats,
    }
    if baseline_stats is not None:
        combined["baseline_rules_vs_rules"] = baseline_stats

    # Report
    _print_combined_stats(combined)

    # Write results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    log.info(f"Results written to {out_path}")


def _run_evaluation(trainer, env, n_trials: int) -> dict:
    """Run n_trials episodes with the neural network and collect statistics."""
    import numpy as np
    rewards = []
    outcomes: dict[str, int] = {}
    times = []

    # Wire NeuralPlayerAI into the env (same as training)
    env.sample_action_fn = trainer._sample_action

    for trial in range(n_trials):
        t0 = time.time()
        env.reset()
        ep_reward = 0.0
        done = False
        info = None

        while not done:
            _, reward, done, info = env.step()
            ep_reward += reward

        rewards.append(ep_reward)
        outcome = info.trial_outcome if info else "unknown"
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        times.append(time.time() - t0)

        if (trial + 1) % 10 == 0:
            log.info(f"  [neural] trial {trial+1}/{n_trials}: outcome={outcome}, reward={ep_reward:.2f}")

    n = len(rewards)
    win_count = outcomes.get("box_possession", 0)
    return {
        "n_trials": n,
        "win_rate_pct": 100.0 * win_count / n,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "outcomes": outcomes,
        "mean_episode_time_s": float(np.mean(times)),
    }


def _run_baseline_evaluation(env, n_trials: int) -> dict:  # env unused, kept for API compat
    """Run n_trials episodes with rules-based AI on BOTH sides.

    build_1v1_scenario now assigns Phase1RulesAI to both trainee and opponent
    via player.ai; Match.step() fires them automatically.  We use ScenarioEnv
    with a noop action — ScenarioEnv's apply_action_to_player still fires, but
    since player.ai runs first (inside _process_orders) and overwrites the order
    every tick the noop StopOrder is immediately replaced.
    """
    import numpy as np
    import functools
    from footballcoach.ui.scenarios import build_1v1_scenario, ScenarioDefinition
    from footballcoach.rules_ai import Phase1RulesAI
    from footballcoach.ai.env.scenario_env import ScenarioEnv

    def _baseline_build(*args, **kwargs):
        match = build_1v1_scenario(*args, **kwargs)
        # Force both players to rules-based regardless of the 50/50 flag.
        for pid in ("trainee", "opponent"):
            try:
                match.player_by_id(pid).ai = Phase1RulesAI()
            except KeyError:
                pass
        return match

    defn = ScenarioDefinition(
        key="baseline_1v1",
        label="Baseline: rules vs rules",
        description="Rules-based AI on both sides for win-rate baseline",
        build=functools.partial(_baseline_build, ball_max_speed_mps=4.0),
        on_tick=None,
    )
    baseline_env = ScenarioEnv(
        definition=defn,
        trainee_player_id="trainee",
        phase=1,
        secondary_player_ids=[],
    )

    outcomes: dict[str, int] = {}
    times = []

    for trial in range(n_trials):
        t0 = time.time()
        baseline_env.reset()
        done = False
        info = None
        while not done:
            _, _, done, info = baseline_env.step()

        outcome = info.trial_outcome if info else "unknown"
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        times.append(time.time() - t0)

        if (trial + 1) % 10 == 0:
            log.info(f"  [baseline] trial {trial+1}/{n_trials}: outcome={outcome}")

    n = len(times)
    win_count = outcomes.get("box_possession", 0)
    return {
        "n_trials": n,
        "win_rate_pct": 100.0 * win_count / n,
        "outcomes": outcomes,
        "mean_episode_time_s": float(np.mean(times)),
    }


def _print_combined_stats(combined: dict) -> None:
    log.info("=" * 60)
    log.info("EVALUATION RESULTS")
    log.info("=" * 60)
    log.info(f"  Checkpoint:  {combined['checkpoint']}  (step {combined['checkpoint_step']:,})")
    log.info(f"  Phase:       {combined['phase']}   Trials: {combined['n_trials']}")

    nn = combined["neural_net"]
    log.info(f"  Neural net:  win={nn['win_rate_pct']:.1f}%  "
             f"reward={nn['mean_reward']:.2f}±{nn['std_reward']:.2f}  "
             f"outcomes={nn['outcomes']}")

    bl = combined.get("baseline_rules_vs_rules")
    if bl:
        log.info(f"  Rules base:  win={bl['win_rate_pct']:.1f}%  outcomes={bl['outcomes']}")
        diff = nn["win_rate_pct"] - bl["win_rate_pct"]
        log.info(f"  Delta vs baseline: {diff:+.1f}pp")

    log.info("=" * 60)


if __name__ == "__main__":
    main()
