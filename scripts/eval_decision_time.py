"""Evaluate how decision interval affects phase-1 performance.

Sweeps decision_interval_s × opponent_mode and prints a summary table of
win rate and mean reward for each combination.

Usage (from repo root):
    uv run python scripts/eval_decision_time.py \
        --checkpoint checkpoints/phase1_run194/checkpoint_pretrained.pt

    # Custom options:
    uv run python scripts/eval_decision_time.py \
        --checkpoint checkpoints/phase1_run194/checkpoint_pretrained.pt \
        --n-trials 100 \
        --decision-times 0.1 0.3 0.5 1.0 2.0 \
        --output results/decision_time_sweep.json
"""
from __future__ import annotations

import argparse
import functools
import json
import logging
import random
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval_decision_time")

DEFAULT_DECISION_TIMES = [0.3, 0.7, 1.2]
OPPONENT_MODES = ["immobile", "self_play", "rules"]
OPPONENT_LABELS = {
    "immobile":  "Immobile",
    "self_play": "Self-play",
    "rules":     "Rules AI",
}
COMP_SHORT = {
    "approach":        "appr",
    "retreat":         "retr",
    "approach_speed":  "appr_spd",
    "heading":         "hdg",
    "get_possession":  "poss",
    "progress":        "prog",
    "lose_possession": "lpos",
    "ball_out":        "out",
    "illegal":         "ill",
    "box_possession":  "box",
    "speed_bonus":     "spd",
    "opponent_box":    "opp_box",
    "timeout":         "tout",
    "proximity_bonus": "prox",
    "stamina_penalty": "stam",
}


def _build_env(opponent_mode: str, max_episode_s: float = 240.0):
    """Build a fresh ScenarioEnv for the given opponent mode (phase 1)."""
    from footballcoach.ai.env.scenario_env import ScenarioEnv
    from footballcoach.ui.scenarios import (
        build_1v1_scenario,
        phase1_training_on_tick,
        ScenarioDefinition,
    )

    opp_cfg = {
        "immobile":  dict(opponent_rules_prob=0.0, opponent_immobile_prob=1.0),
        "self_play": dict(opponent_rules_prob=0.0, opponent_immobile_prob=0.0),
        "rules":     dict(opponent_rules_prob=1.0, opponent_immobile_prob=0.0),
    }[opponent_mode]

    defn = ScenarioDefinition(
        key=f"eval_phase1_{opponent_mode}",
        label=f"Phase-1 eval ({opponent_mode})",
        description=f"Phase-1 1v1 evaluation with {opponent_mode} opponent",
        build=functools.partial(build_1v1_scenario, ball_max_speed_mps=10.0, **opp_cfg),
        on_tick=phase1_training_on_tick,
    )
    secondary = ["opponent"] if opponent_mode == "self_play" else []
    return ScenarioEnv(
        definition=defn,
        trainee_player_id="trainee",
        phase=1,
        secondary_player_ids=secondary,
        max_episode_s=max_episode_s,
    )


def _run_trials(
    trainer, env, n_trials: int, decision_interval_s: float, seed: int
) -> dict:
    """Patch env's decision interval and run n_trials episodes.

    Patching _decision_interval_s + _ticks_per_decision before each reset()
    ensures NeuralPlayerAI (created inside reset()) and the step() tick-loop
    both use the new interval throughout every episode of this condition.
    """
    import numpy as np

    env._decision_interval_s = decision_interval_s
    env._ticks_per_decision = max(1, round(decision_interval_s / env._dt_s))
    env.sample_action_fn = trainer._sample_action
    env.rng = random.Random(seed)

    rewards: list[float] = []
    outcomes: dict[str, int] = {}
    comp_acc: dict[str, float] = {}

    for trial in range(n_trials):
        # Seed global RNG before each trial so that build_1v1_scenario's
        # internal random.Random() draws the same scenario layout regardless
        # of decision_interval_s. Trial N is identical across all conditions.
        trial_seed = seed * 10_000 + trial
        random.seed(trial_seed)
        np.random.seed(trial_seed % (2**31))

        env.reset()
        ep_reward = 0.0
        done = False
        info = None
        while not done:
            _, reward, done, info = env.step()
            ep_reward += reward
            for k, v in getattr(env, "last_reward_components", {}).items():
                comp_acc[k] = comp_acc.get(k, 0.0) + v
        rewards.append(ep_reward)
        outcome = info.trial_outcome if info else "unknown"
        outcomes[outcome] = outcomes.get(outcome, 0) + 1

        if (trial + 1) % 20 == 0:
            done_so_far = trial + 1
            win_so_far = outcomes.get("box_possession", 0)
            log.info(
                f"    trial {done_so_far}/{n_trials}  "
                f"win={100.0*win_so_far/done_so_far:.0f}%  "
                f"last_outcome={outcome}  last_rew={ep_reward:.1f}"
            )

    n = len(rewards)
    win_count = outcomes.get("box_possession", 0)
    return {
        "decision_interval_s": decision_interval_s,
        "ticks_per_decision": env._ticks_per_decision,
        "n_trials": n,
        "win_rate_pct": 100.0 * win_count / n,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "outcomes": outcomes,
        "reward_components": {k: float(v / n) for k, v in comp_acc.items()},
    }


def _print_summary(results: dict, decision_times: list[float], opponents: list[str] | None = None) -> None:
    if opponents is None:
        opponents = OPPONENT_MODES
    W = 84
    sep = "─" * W

    print()
    print(sep)
    print(f"  DECISION TIME SWEEP  ·  {results['_meta']['checkpoint']}")
    print(f"  {results['_meta']['n_trials']} trials per condition  ·  seed={results['_meta']['seed']}")
    print(sep)
    print(f"  {'Decision':>14}  {'Opponent':>12}  {'Win%':>6}  {'MeanRew':>9}  {'Std':>7}  Outcomes")
    print(sep)

    for i_dt, dt in enumerate(decision_times):
        if i_dt > 0:
            print()
        for mode in opponents:
            key = f"{dt}_{mode}"
            r = results.get(key)
            if r is None:
                continue
            ticks = r["ticks_per_decision"]
            dt_str = f"{dt:.2f}s ({ticks}t)" if mode == opponents[0] else ""
            label = OPPONENT_LABELS[mode]
            oc_str = "  ".join(f"{k}:{v}" for k, v in sorted(r["outcomes"].items()))
            print(
                f"  {dt_str:>14}  {label:>12}  {r['win_rate_pct']:>5.1f}%"
                f"  {r['mean_reward']:>9.2f}  {r['std_reward']:>7.2f}  {oc_str}"
            )

    print(sep)

    # Reward component breakdown
    print()
    print("  Reward components (mean per episode, sorted by magnitude)")
    print(sep)

    all_keys: list[str] = []
    for dt in decision_times:
        for mode in opponents:
            for k in results.get(f"{dt}_{mode}", {}).get("reward_components", {}):
                if k not in all_keys:
                    all_keys.append(k)
    all_keys.sort(
        key=lambda k: -max(
            abs(results.get(f"{dt}_{mode}", {}).get("reward_components", {}).get(k, 0.0))
            for dt in decision_times
            for mode in opponents
        )
    )

    for i_dt, dt in enumerate(decision_times):
        if i_dt > 0:
            print()
        for mode in opponents:
            key = f"{dt}_{mode}"
            r = results.get(key)
            if r is None:
                continue
            comps = r.get("reward_components", {})
            label = OPPONENT_LABELS[mode]
            dt_str = f"{dt:.2f}s" if mode == OPPONENT_MODES[0] else ""
            parts = [
                f"{COMP_SHORT.get(k, k)}={comps[k]:+.2f}"
                for k in all_keys
                if abs(comps.get(k, 0.0)) > 0.005
            ]
            print(f"  {dt_str:>8}  {label:>12}  {'  '.join(parts)}")

    print(sep)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep decision interval vs. opponent mode in phase 1."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to checkpoint .pt file",
    )
    parser.add_argument(
        "--n-trials", type=int, default=200,
        help="Trials per (decision_time × opponent) condition (default: 200)",
    )
    parser.add_argument(
        "--decision-times", type=float, nargs="+", default=DEFAULT_DECISION_TIMES,
        metavar="S",
        help=f"Decision interval(s) in seconds (default: {DEFAULT_DECISION_TIMES})",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional path to write results JSON",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-episode-s", type=float, default=240.0,
        help="Max episode duration in sim-seconds (default: 240.0)",
    )
    parser.add_argument(
        "--opponents", nargs="+", default=OPPONENT_MODES,
        choices=OPPONENT_MODES, metavar="MODE",
        help=f"Opponent modes to include (default: all). Choices: {OPPONENT_MODES}",
    )
    args = parser.parse_args()

    import torch
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer

    log.info(f"Loading checkpoint: {args.checkpoint}")
    trainer = PPOTrainer.load_for_inference(args.checkpoint)

    all_results: dict = {
        "_meta": {
            "checkpoint": args.checkpoint,
            "n_trials": args.n_trials,
            "decision_times": args.decision_times,
            "opponents": args.opponents,
            "seed": args.seed,
        }
    }

    conditions = [
        (dt, mode) for dt in args.decision_times for mode in args.opponents
    ]
    t_start = time.time()

    for ci, (dt, mode) in enumerate(conditions):
        log.info(
            f"[{ci+1}/{len(conditions)}] decision={dt}s  opponent={mode}  ({args.n_trials} trials)"
        )
        t0 = time.time()
        env = _build_env(mode, max_episode_s=args.max_episode_s)
        stats = _run_trials(trainer, env, args.n_trials, dt, seed=args.seed + ci)
        elapsed = time.time() - t0
        log.info(
            f"    win={stats['win_rate_pct']:.1f}%  "
            f"reward={stats['mean_reward']:.2f}±{stats['std_reward']:.2f}  "
            f"outcomes={stats['outcomes']}  ({elapsed:.0f}s)"
        )
        all_results[f"{dt}_{mode}"] = stats

    log.info(f"All {len(conditions)} conditions done in {time.time()-t_start:.0f}s")

    _print_summary(all_results, args.decision_times, opponents=args.opponents)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        log.info(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
