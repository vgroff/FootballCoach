"""CLI entry point for PPO training.

Usage:
    uv run python -m footballcoach.ai.scripts.train --phase 1
    uv run python -m footballcoach.ai.scripts.train --phase 2 --total-steps 500000
    uv run python -m footballcoach.ai.scripts.train --phase 1 --checkpoint path/to/ckpt.pt

See ai_design_doc.md section 5 for the two MVP experiments.
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("footballcoach.ai.train")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the football AI with PPO")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Curriculum phase to train (default: 1)")
    parser.add_argument("--total-steps", type=int, default=500_000,
                        help="Total training decision-steps (default: 500000)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a checkpoint .pt file to resume from")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/",
                        help="Directory to save training checkpoints")
    parser.add_argument("--device", type=str, default="cpu",
                        help="PyTorch device (default: cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    import torch
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)

    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    from footballcoach.ai.curriculum.phases import PHASES_BY_ID
    from footballcoach.ai.env.scenario_env import ScenarioEnv

    phase = PHASES_BY_ID.get(args.phase)
    if phase is None:
        log.error(f"Unknown phase: {args.phase}")
        return

    log.info(f"Starting training: phase={phase.name}, total_steps={args.total_steps:,}")
    log.info(f"Phase description: {phase.description}")

    # Build environment
    env = _build_env(phase)

    # Build trainer
    trainer = PPOTrainer.from_config(device=device, checkpoint_dir=checkpoint_dir)

    # Optionally resume from checkpoint
    if args.checkpoint:
        trainer.load_checkpoint(Path(args.checkpoint))

    # Train
    trainer.train(env, total_steps=args.total_steps)


def _build_env(phase):
    """Build a ScenarioEnv for the given curriculum phase."""
    from footballcoach.ai.env.scenario_env import ScenarioEnv

    # MVP: use the existing UI scenarios as training environments.
    # Phase 1: a simple 1v1 GetPossession/Move scenario.
    # Phase 2: the penalty scenario (attacker + optional GK).
    if phase.phase_id == 1:
        env = _build_phase1_env(phase)
    elif phase.phase_id == 2:
        env = _build_phase2_env(phase)
    else:
        raise NotImplementedError(f"Phase {phase.phase_id} not yet implemented")
    return env


def _build_phase1_env(phase):
    """Phase 1: 1v1 get possession and bring ball toward goal."""
    from footballcoach.ai.env.scenario_env import ScenarioEnv
    from footballcoach.ui.scenarios import build_1v1_scenario, ScenarioDefinition

    defn = ScenarioDefinition(
        key="phase1_1v1",
        label="Phase 1: 1v1 Get Possession",
        description="1v1 scenario for curriculum phase 1",
        build=build_1v1_scenario,
    )
    return ScenarioEnv(
        definition=defn,
        trainee_player_id="trainee",
        phase=1,
        **phase.env_kwargs,
    )


def _build_phase2_env(phase):
    """Phase 2: shooting (penalty + GK)."""
    from footballcoach.ai.env.scenario_env import ScenarioEnv
    from footballcoach.ui.scenarios import build_penalty_scenario, ScenarioDefinition

    defn = ScenarioDefinition(
        key="phase2_penalty",
        label="Phase 2: Shoot",
        description="Penalty scenario for curriculum phase 2",
        build=build_penalty_scenario,
    )
    return ScenarioEnv(
        definition=defn,
        trainee_player_id="kicker",
        phase=2,
        **phase.env_kwargs,
    )


if __name__ == "__main__":
    main()
