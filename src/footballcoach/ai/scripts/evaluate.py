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
from typing import Optional

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
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Number of distinct eval seeds (default: ai_config.json eval.eval_n_seeds).")
    parser.add_argument("--repeats-per-seed", type=int, default=None,
                        help="Episodes run per seed, averaged into the same stats (default: ai_config.json eval.eval_repeats_per_seed).")
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
    parser.add_argument("--n-parallel-workers", type=int, default=None,
                        help="Subprocess workers for eval (default: ai_config.json eval.eval_n_parallel_workers). 1 = sequential.")
    parser.add_argument("--deterministic", action="store_true",
                        help="Run the checkpoint's policy fully deterministically (mode/mean of every "
                             "action head instead of a stochastic PPO sample) -- no exploration noise. "
                             "Shorthand for --deterministic-decision --deterministic-direction.")
    parser.add_argument("--deterministic-decision", action="store_true",
                        help="Only the discrete decision/execution heads (shoot/pass/move/tackle/etc. "
                             "intents and pass/tackle/mark targets) use their mode; move/kick direction "
                             "still sample stochastically.")
    parser.add_argument("--deterministic-direction", action="store_true",
                        help="Only the move_direction/kick_direction heads use their mean; discrete "
                             "decision/execution heads still sample stochastically.")
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
        baseline_stats = _run_baseline_evaluation(None, args.baseline_trials, args.repeats_per_seed, args.n_parallel_workers)
        combined = {
            "checkpoint": None,
            "checkpoint_step": 0,
            "phase": args.phase,
            "n_trials": 0,
            "neural_net": None,
            "baseline_rules_vs_rules": baseline_stats,
        }
        bl = baseline_stats
        log.info(f"  Rules baseline: win={bl['win_rate_pct']:.1f}%  outcomes={bl['outcomes']}"
                 f"  (win/loss/tout/miss: {bl['outcome_breakdown']})")
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
    log.info(f"Evaluating checkpoint (phase={phase.name}, step={step:,})...")
    neural_stats = _run_evaluation(
        trainer, env, args.n_trials, args.repeats_per_seed, args.checkpoint, args.n_parallel_workers,
        deterministic=args.deterministic,
        deterministic_decision=args.deterministic_decision,
        deterministic_direction=args.deterministic_direction,
    )
    log.info(f"  Neural net eval done: {neural_stats['n_trials']} episodes, "
             f"win={neural_stats['win_rate_pct']:.1f}%")

    # Run rules-vs-rules baseline (unless suppressed)
    baseline_stats: dict | None = None
    if not args.no_baseline and args.phase == 1:
        log.info(f"Running rules-vs-rules baseline ({args.baseline_trials} episodes)...")
        baseline_stats = _run_baseline_evaluation(env, args.baseline_trials, args.repeats_per_seed, args.n_parallel_workers)
        log.info(f"  Baseline eval done: win={baseline_stats['win_rate_pct']:.1f}%")

    # Combine results
    combined = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": step,
        "phase": args.phase,
        "n_trials": neural_stats["n_trials"],
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


def _checkpoint_eval_env_factory(
    checkpoint_path: str, device: str, definition_kwargs: dict,
    deterministic: bool = False,
    deterministic_decision: bool = False,
    deterministic_direction: bool = False,
) -> tuple:
    """Module-level (picklable) worker factory for parallel evaluate.py runs
    -- each subprocess reloads the checkpoint fresh from disk (live
    nn.Module/optimizer objects aren't picklable across the process
    boundary), mirroring ai/ppo/rollout_worker.py's pattern."""
    import functools
    import torch
    from footballcoach.ai.env.scenario_env import ScenarioEnv
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer

    trainer = PPOTrainer.load_for_inference(checkpoint_path)

    def _env_factory(seed: int) -> ScenarioEnv:
        return ScenarioEnv(**definition_kwargs, seed=seed)

    sample_fn = trainer._sample_action
    if deterministic or deterministic_decision or deterministic_direction:
        sample_fn = functools.partial(
            trainer._sample_action,
            deterministic=deterministic,
            deterministic_decision=deterministic_decision,
            deterministic_direction=deterministic_direction,
        )
    return _env_factory, sample_fn


def _run_evaluation(
    trainer, env, n_trials: Optional[int] = None, repeats_per_seed: Optional[int] = None,
    checkpoint_path: Optional[str] = None, n_parallel_workers: Optional[int] = None,
    deterministic: bool = False,
    deterministic_decision: bool = False,
    deterministic_direction: bool = False,
) -> dict:
    """Run the shared seeded evaluation (ai/eval/seeded_eval.py) against the
    checkpoint's own env/definition, reusing the SAME fixed seed list as
    PPOTrainer._eval_vs_rules() so numbers are comparable across training
    runs and standalone evaluate.py calls. n_trials/repeats_per_seed override
    ai_config.json's eval.eval_n_seeds/eval_repeats_per_seed when given.
    n_parallel_workers>1 requires checkpoint_path (each subprocess reloads
    the checkpoint itself -- see _checkpoint_eval_env_factory). deterministic=True
    runs the policy's mode/mean instead of sampling; deterministic_decision/
    deterministic_direction narrow this to just the discrete or just the
    direction heads respectively (see PPOTrainer._sample_action)."""
    from footballcoach.ai.config import load_ai_config
    from footballcoach.ai.env.scenario_env import ScenarioEnv
    from footballcoach.ai.eval.seeded_eval import (
        default_eval_seeds, run_seeded_evaluation, run_seeded_evaluation_parallel,
    )

    cfg = load_ai_config()
    seeds = default_eval_seeds(cfg)
    if n_trials is not None:
        seeds = seeds[:n_trials] if n_trials <= len(seeds) else seeds + list(
            range(seeds[-1] + 1, seeds[-1] + 1 + (n_trials - len(seeds)))
        )
    repeats = repeats_per_seed if repeats_per_seed is not None else int(
        cfg.get("eval", {}).get("eval_repeats_per_seed", 2)
    )
    n_workers = n_parallel_workers if n_parallel_workers is not None else int(
        cfg.get("eval", {}).get("eval_n_parallel_workers", 1)
    )

    if n_workers > 1 and checkpoint_path is not None:
        import functools
        _def_kwargs = dict(
            definition=env.definition,
            trainee_player_id=env.trainee_player_id,
            phase=env.phase,
            secondary_player_ids=env.secondary_player_ids,
            max_episode_s=env.max_episode_s,
        )
        worker_factory = functools.partial(
            _checkpoint_eval_env_factory, checkpoint_path, "cpu", _def_kwargs,
            deterministic, deterministic_decision, deterministic_direction,
        )
        result = run_seeded_evaluation_parallel(worker_factory, seeds, repeats, n_workers=n_workers)
    else:
        def _env_factory(seed: int) -> ScenarioEnv:
            return ScenarioEnv(
                definition=env.definition,
                trainee_player_id=env.trainee_player_id,
                phase=env.phase,
                secondary_player_ids=env.secondary_player_ids,
                max_episode_s=env.max_episode_s,
                seed=seed,
            )

        import functools
        sample_fn = trainer._sample_action
        if deterministic or deterministic_decision or deterministic_direction:
            sample_fn = functools.partial(
                trainer._sample_action,
                deterministic=deterministic,
                deterministic_decision=deterministic_decision,
                deterministic_direction=deterministic_direction,
            )
        result = run_seeded_evaluation(_env_factory, sample_fn, seeds, repeats)

    d = result.as_dict()
    d["min_reward"] = float(min(result.rewards)) if result.rewards else float("nan")
    d["max_reward"] = float(max(result.rewards)) if result.rewards else float("nan")
    return d


def _baseline_env_worker_factory() -> tuple:
    """Module-level (picklable) worker factory for the rules-vs-rules
    baseline -- no checkpoint/network involved, so sample_action_fn is None."""
    from footballcoach.ui.scenarios import build_1v1_scenario, ScenarioDefinition
    from footballcoach.rules_ai import Phase1RulesAI
    from footballcoach.ai.env.scenario_env import ScenarioEnv

    def _baseline_build(*args, **kwargs):
        match = build_1v1_scenario(*args, **kwargs)
        for pid in ("trainee", "opponent"):
            try:
                match.player_by_id(pid).ai = Phase1RulesAI()
            except KeyError:
                pass
        return match

    def _env_factory(seed: int) -> ScenarioEnv:
        def _build(*_a, **_kw):
            return _baseline_build(*_a, ball_max_speed_mps=4.0, seed=seed, **_kw)

        defn = ScenarioDefinition(
            key="baseline_1v1",
            label="Baseline: rules vs rules",
            description="Rules-based AI on both sides for win-rate baseline",
            build=_build,
            on_tick=None,
        )
        return ScenarioEnv(definition=defn, trainee_player_id="trainee", phase=1, secondary_player_ids=[])

    return _env_factory, None


def _run_baseline_evaluation(
    env, n_trials: int, repeats_per_seed: Optional[int] = None, n_parallel_workers: Optional[int] = None,
) -> dict:  # env unused, kept for API compat
    """Run the shared seeded evaluation with rules-based AI on BOTH sides.

    build_1v1_scenario assigns Phase1RulesAI to both trainee and opponent via
    player.ai; Match.step() fires them automatically.
    """
    from footballcoach.ai.config import load_ai_config
    from footballcoach.ai.eval.seeded_eval import run_seeded_evaluation, run_seeded_evaluation_parallel

    cfg = load_ai_config()
    seeds = list(range(2_000_000, 2_000_000 + n_trials))
    repeats = repeats_per_seed if repeats_per_seed is not None else int(
        cfg.get("eval", {}).get("eval_repeats_per_seed", 2)
    )
    n_workers = n_parallel_workers if n_parallel_workers is not None else int(
        cfg.get("eval", {}).get("eval_n_parallel_workers", 1)
    )

    if n_workers > 1:
        result = run_seeded_evaluation_parallel(
            _baseline_env_worker_factory, seeds, repeats, n_workers=n_workers,
        )
    else:
        env_factory, sample_action_fn = _baseline_env_worker_factory()
        result = run_seeded_evaluation(env_factory, sample_action_fn, seeds, repeats)

    d = result.as_dict()
    del d["mean_value_pred"]  # no neural player in this scenario
    return d


def _print_combined_stats(combined: dict) -> None:
    log.info("=" * 60)
    log.info("EVALUATION RESULTS")
    log.info("=" * 60)
    log.info(f"  Checkpoint:  {combined['checkpoint']}  (step {combined['checkpoint_step']:,})")
    log.info(f"  Phase:       {combined['phase']}   Trials: {combined['n_trials']}")

    nn = combined["neural_net"]
    log.info(f"  Neural net:  win={nn['win_rate_pct']:.1f}%  "
             f"reward={nn['mean_reward']:.2f}±{nn['std_reward']:.2f} (sem={nn['sem_reward']:.2f})  "
             f"outcomes={nn['outcomes']}  (win/loss/tout/miss: {nn['outcome_breakdown']})")

    bl = combined.get("baseline_rules_vs_rules")
    if bl:
        log.info(f"  Rules base:  win={bl['win_rate_pct']:.1f}%  outcomes={bl['outcomes']}"
                 f"  (win/loss/tout/miss: {bl['outcome_breakdown']})")
        diff = nn["win_rate_pct"] - bl["win_rate_pct"]
        log.info(f"  Delta vs baseline: {diff:+.1f}pp")

    log.info("=" * 60)


if __name__ == "__main__":
    main()
