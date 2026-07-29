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
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("footballcoach.ai.evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained football AI checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--phase", type=int, default=2, choices=[1, 2, 3, 4],
                        help="Curriculum phase to evaluate (default: 2)")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Number of episodes to evaluate (default: 100)")
    parser.add_argument("--output", type=str, default="results/eval_latest.json",
                        help="Output JSON file for results")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)

    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    from footballcoach.ai.curriculum.phases import PHASES_BY_ID

    phase = PHASES_BY_ID.get(args.phase)
    if phase is None:
        log.error(f"Unknown phase: {args.phase}")
        return

    # Load trainer + checkpoint
    trainer = PPOTrainer.from_config(device=device)
    step = trainer.load_checkpoint(Path(args.checkpoint))

    # Build env (same as training)
    from footballcoach.ai.scripts.train import _build_env
    env = _build_env(phase)

    # Run evaluation
    log.info(f"Evaluating {args.n_trials} episodes (phase={phase.name}, step={step:,})")
    stats = _run_evaluation(trainer, env, args.n_trials)

    # Report
    _print_stats(stats)

    # Write results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Results written to {out_path}")


def _run_evaluation(trainer, env, n_trials: int) -> dict:
    """Run n_trials episodes and collect statistics."""
    rewards = []
    outcomes = {}
    times = []
    goal_count = 0

    for trial in range(n_trials):
        t0 = time.time()
        obs = env.reset()
        ep_reward = 0.0
        done = False
        info = None

        while not done:
            action, log_prob, value, decision_probs, exec_phys, dec_phys, tgt_slots = (
                trainer._sample_action(obs.to_torch_dict())
            )
            env_action = {
                "decision_probs": decision_probs,
                "execution_physical": exec_phys,
                "decision_physical": dec_phys,
                "target_slots": tgt_slots,
                "slot_player_ids": [None] * 21,
                "decision": action,
            }
            obs, reward, done, info = env.step(env_action)
            ep_reward += reward

        rewards.append(ep_reward)
        outcome = info.trial_outcome if info else "unknown"
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if outcome == "goal":
            goal_count += 1
        times.append(time.time() - t0)

        if (trial + 1) % 10 == 0:
            log.info(f"Trial {trial+1}/{n_trials}: outcome={outcome}, reward={ep_reward:.2f}")

    import numpy as np
    return {
        "n_trials": n_trials,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "goal_rate_pct": 100.0 * goal_count / n_trials,
        "outcomes": outcomes,
        "mean_episode_time_s": float(np.mean(times)),
    }


def _print_stats(stats: dict) -> None:
    log.info("=" * 60)
    log.info("EVALUATION RESULTS")
    log.info("=" * 60)
    log.info(f"  Trials:       {stats['n_trials']}")
    log.info(f"  Mean reward:  {stats['mean_reward']:.3f} ± {stats['std_reward']:.3f}")
    log.info(f"  Goal rate:    {stats['goal_rate_pct']:.1f}%")
    log.info(f"  Outcomes:     {stats['outcomes']}")
    log.info(f"  Ep time:      {stats['mean_episode_time_s']:.2f}s avg")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
