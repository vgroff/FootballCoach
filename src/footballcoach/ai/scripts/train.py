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
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("footballcoach.ai.train")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the football AI with PPO")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Curriculum phase to train (default: 1)")
    parser.add_argument("--total-steps", type=int, default=500_000,
                        help="PPO decision-steps to train for, excluding pre-training (default: 500000)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a checkpoint .pt file to resume from")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/",
                        help="Directory to save training checkpoints")
    parser.add_argument("--device", type=str, default="cpu",
                        help="PyTorch device (default: cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--bc-pretrain-steps", type=int, default=None,
                        help=(
                            "Supervised BC pre-training steps before PPO starts. "
                            "Overrides bc.pretrain_steps in ai_config.json when set. "
                            "Set to 0 to disable pre-training entirely."
                        ))
    parser.add_argument("--no-bc-aux", action="store_true",
                        help="Disable BC auxiliary loss during PPO (ignores ai_config.json bc.aux_coeff).")
    parser.add_argument("--bc-dataset", type=str, default=None,
                        help=(
                            "Path to a directory of .npz demonstration files for offline BC "
                            "pre-training. When provided, pre-training uses the dataset instead "
                            "of online env collection. Use record_demonstrations.py to create one."
                        ))
    parser.add_argument("--bc-pretrain-epochs", type=int, default=None,
                        help="Epochs over dataset for offline BC pre-training (default: bc.bc_pretrain_epochs in ai_config.json).")
    parser.add_argument("--bc-pretrain-batch-size", type=int, default=None,
                        help="Minibatch size for offline BC pre-training (default: bc.bc_pretrain_batch_size in ai_config.json).")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug-level logs (per-minibatch details, per-head diagnostics).")
    parser.add_argument("--pre-ppo-eval-trials", type=int, default=40,
                        help="Episodes to evaluate vs rules-based AI after pre-training, before "
                             "PPO starts. Set to 0 to skip (default: 40).")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("footballcoach.ai").setLevel(logging.DEBUG)

    import torch
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)

    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    from footballcoach.ai.curriculum.phases import PHASES_BY_ID
    from footballcoach.ai.config import load_ai_config

    phase = PHASES_BY_ID.get(args.phase)
    if phase is None:
        log.error(f"Unknown phase: {args.phase}")
        return

    log.info(f"Starting training: phase={phase.name}, total_steps={args.total_steps:,}")
    log.info(f"Phase description: {phase.description}")

    # Build environment
    from footballcoach.ai.curriculum.envs import build_env, bc_label_fn_for_phase
    env = build_env(phase)

    # BC label function (None = no BC)
    label_fn = bc_label_fn_for_phase(args.phase)

    # BC pre-training steps: CLI flag overrides config
    cfg = load_ai_config()
    bc_cfg = cfg.get("bc", {})
    pretrain_steps = (
        args.bc_pretrain_steps
        if args.bc_pretrain_steps is not None
        else int(bc_cfg.get("pretrain_steps", 0))
    )

    # Build trainer (--no-bc-aux zeros out the aux coeff in config)
    if args.no_bc_aux:
        import copy
        cfg = copy.deepcopy(cfg)
        cfg.setdefault("bc", {})["aux_coeff_start"] = 0.0
        cfg["bc"]["aux_coeff_end"] = 0.0
        trainer = PPOTrainer(
            decision_net=__import__("footballcoach.ai.models.decision_network",
                                    fromlist=["DecisionNetwork"]).DecisionNetwork.from_config(),
            execution_net=__import__("footballcoach.ai.models.execution_network",
                                     fromlist=["ExecutionNetwork"]).ExecutionNetwork.from_config(),
            cfg=cfg,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )
    else:
        trainer = PPOTrainer.from_config(device=device, checkpoint_dir=checkpoint_dir)

    # Optionally resume from checkpoint
    if args.checkpoint:
        trainer.load_checkpoint(Path(args.checkpoint))

    # Pre-training phase: BC + value jointly when a dataset is available,
    # otherwise fall back to online BC then separate value pre-training.
    value_pretrain_steps = int(bc_cfg.get("value_pretrain_steps", 0))
    value_pretrain_epochs = int(bc_cfg.get("value_pretrain_epochs", 20))
    value_pretrain_lr = float(bc_cfg.get("value_pretrain_lr", 1e-3))

    if not args.checkpoint:
        from footballcoach.ai.bc.dataset import DemonstrationDataset
        dataset = None
        if args.bc_dataset:
            dataset = DemonstrationDataset.from_directory(args.bc_dataset)
            log.info(f"Offline BC dataset: {len(dataset):,} steps from {args.bc_dataset}")

        bc_pretrain_epochs = (
            args.bc_pretrain_epochs
            if args.bc_pretrain_epochs is not None
            else int(bc_cfg.get("bc_pretrain_epochs", 30))
        )
        bc_pretrain_batch_size = (
            args.bc_pretrain_batch_size
            if args.bc_pretrain_batch_size is not None
            else int(bc_cfg.get("bc_pretrain_batch_size", 256))
        )

        if dataset is not None and label_fn is not None:
            # Combined joint pre-training (BC + value in one pass)
            trainer.pretrain_combined(
                env=env,
                dataset=dataset,
                n_epochs=bc_pretrain_epochs,
                batch_size=bc_pretrain_batch_size,
                bc_lr=float(bc_cfg.get("pretrain_lr", 3e-4)),
                value_lr=value_pretrain_lr,
                repair_lr=float(bc_cfg.get("bc_repair_lr", bc_cfg.get("pretrain_lr", 3e-4))),
                rollout_steps=max(value_pretrain_steps, trainer.rollout_steps),
                value_epochs=value_pretrain_epochs,
            )
        else:
            # Online BC pre-training (noisy but works without a dataset)
            if pretrain_steps > 0 and label_fn is not None:
                from footballcoach.ai.ppo.bc import BCPretrainer
                pretrainer = BCPretrainer(
                    trainer.decision_net, trainer.execution_net, cfg, device
                )
                pretrainer.pretrain(env, n_steps=pretrain_steps, label_fn=label_fn)

            # Separate value pre-training
            if value_pretrain_steps > 0:
                trainer.pretrain_value(
                    env,
                    n_steps=value_pretrain_steps,
                    n_epochs=value_pretrain_epochs,
                    lr=value_pretrain_lr,
                )

    # Save a checkpoint of the pre-trained model before PPO starts
    if not args.checkpoint and checkpoint_dir is not None:
        pretrain_ckpt = checkpoint_dir / "checkpoint_pretrained.pt"
        trainer._save_checkpoint_to(pretrain_ckpt)
        log.info(f"Pre-trained checkpoint saved: {pretrain_ckpt}")

    # Quick pre-PPO evaluation: neural vs rules-based AND vs immobile opponent
    if args.pre_ppo_eval_trials > 0:
        from footballcoach.ai.scripts.evaluate import _run_evaluation
        log.info(f"Pre-PPO eval: {args.pre_ppo_eval_trials} episodes (mixed rules/immobile)...")
        eval_stats = _run_evaluation(trainer, env, args.pre_ppo_eval_trials)
        log.info(
            f"Pre-PPO eval result: win={eval_stats['win_rate_pct']:.1f}%  "
            f"mean_rew={eval_stats['mean_reward']:.3f}  "
            f"outcomes={eval_stats['outcomes']}"
        )

        # Also evaluate explicitly against immobile opponent only
        from footballcoach.ui.scenarios import build_1v1_scenario, ScenarioDefinition
        from footballcoach.ai.env.scenario_env import ScenarioEnv
        def _immobile_build(*args, **kwargs):
            match = build_1v1_scenario(*args, **kwargs)
            opp = match.player_by_id("opponent")
            opp.ai = None
            match._opponent_use_rules_ai = False
            return match
        immobile_defn = ScenarioDefinition(key="1v1_immobile", label="1v1 immobile", description="1v1 vs immobile opponent", build=_immobile_build)
        immobile_env = ScenarioEnv(
            immobile_defn, trainee_player_id="trainee", phase=1,
            max_episode_s=env.max_episode_s,
        )
        immobile_stats = _run_evaluation(trainer, immobile_env, args.pre_ppo_eval_trials)
        log.info(
            f"Pre-PPO eval (immobile opp): win={immobile_stats['win_rate_pct']:.1f}%  "
            f"mean_rew={immobile_stats['mean_reward']:.3f}  "
            f"outcomes={immobile_stats['outcomes']}"
        )

    # PPO training (with optional BC aux loss if label_fn and aux_coeff > 0)
    aux_label_fn = None if (args.no_bc_aux or label_fn is None) else label_fn
    trainer.train(env, total_steps=args.total_steps, bc_label_fn=aux_label_fn)


# env building and label functions live in curriculum.envs — no duplication here.


if __name__ == "__main__":
    main()
