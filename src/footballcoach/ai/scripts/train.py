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


def _rules_1v1_build(*args, **kwargs):
    """Module-level (picklable) scenario builder for pre-PPO rules-based eval --
    parallel eval workers pickle this via ScenarioDefinition.build, so it can't
    be a closure defined inside main()."""
    from footballcoach.rules_ai import Phase1RulesAI
    from footballcoach.ui.scenarios import build_1v1_scenario

    match = build_1v1_scenario(*args, **kwargs)
    opp = match.player_by_id("opponent")
    opp.ai = Phase1RulesAI()
    match._opponent_use_rules_ai = True
    match._opponent_is_immobile = False
    return match


def _rules_vs_rules_1v1_build(*args, **kwargs):
    """Module-level (picklable) scenario builder for the pre-training reward
    diagnostic -- rules AI on BOTH sides (see main())."""
    from footballcoach.rules_ai import Phase1RulesAI
    from footballcoach.ui.scenarios import build_1v1_scenario

    match = build_1v1_scenario(*args, **kwargs)
    for p in match.players:
        p.ai = Phase1RulesAI()
    match._opponent_use_rules_ai = True
    match._opponent_is_immobile = False
    return match


def _immobile_1v1_build(*args, **kwargs):
    """Module-level (picklable) scenario builder for pre-PPO immobile-opponent eval."""
    from footballcoach.ui.scenarios import build_1v1_scenario

    match = build_1v1_scenario(*args, **kwargs)
    opp = match.player_by_id("opponent")
    opp.ai = None
    match._opponent_use_rules_ai = False
    match._opponent_is_immobile = True
    return match


def _neural_1v1_build(*args, **kwargs):
    """Module-level (picklable) scenario builder for pre-PPO self-play eval."""
    from footballcoach.ui.scenarios import build_1v1_scenario

    match = build_1v1_scenario(*args, **kwargs)
    opp = match.player_by_id("opponent")
    opp.ai = None
    match._opponent_use_rules_ai = False
    match._opponent_is_immobile = False
    return match


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the football AI with PPO")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Curriculum phase to train (default: 1)")
    parser.add_argument("--total-steps", type=int, default=500_000,
                        help="PPO decision-steps to train for, excluding pre-training (default: 500000)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a checkpoint .pt file to resume from")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory to save training checkpoints (default: auto-generated as checkpoints/phase{N}_run{next}/)")
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
    parser.add_argument("--experiment-separate-value-net", action="store_true",
                        help=(
                            "EXPERIMENTAL: alongside the normal shared-trunk value_head training in "
                            "pretrain_value() (the --bc-dataset=None fallback path only), also train a "
                            "second, fully independent ExecutionNetwork (fresh init, fully unfrozen) on "
                            "the exact same rollout data/returns, and log a side-by-side val_rmse "
                            "comparison each epoch. Purely diagnostic -- the second network is discarded "
                            "and has no effect on the real value_head or subsequent PPO training."
                        ))
    parser.add_argument("--separate-value-net", action="store_true",
                        help=(
                            "Use a permanent, fully independent ExecutionNetwork as the sole critic "
                            "for the ENTIRE training run (BC pre-training, value warm-up, and PPO), "
                            "instead of the default shared-trunk value_head. Unlike "
                            "--experiment-separate-value-net (a throwaway diagnostic comparison), this "
                            "is a real architecture switch: the dedicated critic never receives BC "
                            "gradients (execution_net's own value_head/value_ai_type_channel are frozen "
                            "and unused), so it is free to learn its own features purely for value "
                            "prediction rather than reading through a BC-primed policy trunk. Persisted "
                            "in checkpoints under 'value_net'/'value_net_optimizer'; "
                            "PPOTrainer.load_for_inference() auto-detects this from the checkpoint, so "
                            "no flag is needed when evaluating/running a checkpoint trained with this on."
                        ))
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
    parser.add_argument("--from-pretrained", type=str, default=None, metavar="PATH",
                        help=(
                            "Load a pre-trained checkpoint and skip all BC/value pre-training. "
                            "If PATH is a directory, checkpoint_pretrained.pt inside it is used "
                            "automatically. Useful for iterating on PPO hyperparameters without "
                            "re-running the expensive BC pre-training stage."
                        ))
    parser.add_argument("--pretrain-from-checkpoint", type=str, default=None, metavar="PATH",
                        help=(
                            "Load weights from a checkpoint then still run the full BC/value "
                            "pre-training loop. Unlike --checkpoint (which skips pre-training), "
                            "this lets you re-pretrain an existing policy with new config or demos."
                        ))
    parser.add_argument("--pre-ppo-eval-trials", type=int, default=None,
                        help="Number of distinct eval seeds to evaluate vs rules-based AI after "
                             "pre-training, before PPO starts (default: ai_config.json "
                             "eval.eval_n_seeds; each seed is repeated eval.eval_repeats_per_seed "
                             "times). Set to 0 to skip.")
    parser.add_argument("--no-head-freeze", action="store_true",
                        help="Skip frozen_heads from the curriculum phase definition — all "
                             "decision-network heads are trained during PPO. Default: the "
                             "phase's frozen_heads list is applied before PPO starts.")
    parser.add_argument("--latest", action="store_true",
                        help="Resume from the most recent checkpoint across all phase{N}_run* "
                             "dirs. Equivalent to --checkpoint <latest.pt>: skips pretraining "
                             "and continues the step counter. Errors if no checkpoints exist.")
    parser.add_argument("--latest-pretrain", action="store_true",
                        help="Load the most recent checkpoint then still run the full BC/value "
                             "pre-training loop (equivalent to combining --latest resolution "
                             "with --pretrain-from-checkpoint). Resets the step counter. "
                             "Requires --bc-dataset for combined pre-training.")
    parser.add_argument("--reset-dir-log-std", action="store_true",
                        help="Reset move_dir_log_std/kick_dir_log_std to ppo.dir_log_std_init "
                             "from ai_config.json after loading any checkpoint (--checkpoint, "
                             "--pretrain-from-checkpoint, --from-pretrained, --latest[-pretrain]). "
                             "Useful when a loaded policy's log_std has drifted/collapsed and is "
                             "causing move_dir KL to dominate early-stop.")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("footballcoach.ai").setLevel(logging.DEBUG)

    import torch

    from footballcoach.ai.config import load_ai_config

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(int(load_ai_config().get("ppo", {}).get("main_process_torch_threads", 4)))

    device = torch.device(args.device)

    # Auto-generate checkpoint dir if not specified: checkpoints/phase{N}_run{next}
    if args.checkpoint_dir is None:
        _ckpt_base = Path("checkpoints")
        _prefix = f"phase{args.phase}_run"
        _existing = sorted(
            int(p.name[len(_prefix):]) for p in _ckpt_base.glob(f"{_prefix}*")
            if p.is_dir() and p.name[len(_prefix):].isdigit()
        ) if _ckpt_base.exists() else []
        _next = (_existing[-1] + 1) if _existing else 1
        checkpoint_dir = _ckpt_base / f"{_prefix}{_next}"
    else:
        checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Checkpoint dir: {checkpoint_dir}")

    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
    from footballcoach.ai.curriculum.phases import PHASES_BY_ID

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

    # When re-pretraining from an existing checkpoint (--pretrain-from-checkpoint or
    # --latest-pretrain), swap in the _from_ckpt variants for pretrain epochs and
    # aux coefficients — typically fewer epochs and a lower BC aux start since the
    # policy is already close to the demonstrations.
    _is_pretrain_from_ckpt = bool(args.pretrain_from_checkpoint) or bool(args.latest_pretrain)
    if _is_pretrain_from_ckpt:
        _from_ckpt_keys = (
            "bc_pretrain_epochs",
            "demo_value_pretrain_epochs",
            "value_pretrain_epochs",
            "aux_coeff_start",
            "aux_coeff_end",
            "aux_coeff_anneal_fraction",
        )
        for _k in _from_ckpt_keys:
            _ckpt_k = f"{_k}_from_ckpt"
            if _ckpt_k in bc_cfg:
                log.info(f"_from_ckpt: overriding {_k}={bc_cfg[_k]} → {bc_cfg[_ckpt_k]}")
                bc_cfg[_k] = bc_cfg[_ckpt_k]

    pretrain_steps = (
        args.bc_pretrain_steps
        if args.bc_pretrain_steps is not None
        else int(bc_cfg.get("bc_online_steps", bc_cfg.get("pretrain_steps", 0)))
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
            separate_value_net=args.separate_value_net,
        )
    else:
        trainer = PPOTrainer.from_config(
            device=device, checkpoint_dir=checkpoint_dir,
            separate_value_net=args.separate_value_net,
        )

    # --latest / --latest-pretrain: auto-discover the most recent checkpoint.
    if args.latest or args.latest_pretrain:
        import glob as _glob, re as _re, os as _os
        _ckpt_base = Path("checkpoints")
        _prefix = f"phase{args.phase}_run"
        # Prefer latest.pt symlinks first (always the newest in their run dir),
        # then fall back to the numerically-sorted regular checkpoints.
        _run_dirs = sorted(
            _glob.glob(str(_ckpt_base / f"{_prefix}*/")),
            key=lambda d: int(m.group(1)) if (m := _re.search(rf"{_prefix}(\d+)", d)) else -1,
        )
        _latest_path: Path | None = None
        for _rd in reversed(_run_dirs):
            _symlink = Path(_rd) / "latest.pt"
            if _symlink.exists():
                _latest_path = _symlink
                break
            # No latest.pt — find highest-numbered checkpoint in this dir
            _pts = [
                Path(p) for p in _glob.glob(str(Path(_rd) / "checkpoint*.pt"))
                if not _os.path.basename(p).startswith("checkpoint_pretrained")
            ]
            if _pts:
                _latest_path = max(
                    _pts,
                    key=lambda p: int(m.group(1)) if (m := _re.search(r"(\d+)", p.name)) else -1,
                )
                break
            # No numbered checkpoint either — this run dir got no further than
            # pre-training, but it's still the most recent run: use its
            # checkpoint_pretrained.pt rather than silently falling through to
            # an OLDER run dir that happens to have a numbered checkpoint.
            _pretrained_only = Path(_rd) / "checkpoint_pretrained.pt"
            if _pretrained_only.exists():
                _latest_path = _pretrained_only
                break
        if _latest_path is None:
            log.error(f"--latest: no checkpoints found under {_ckpt_base}/{_prefix}*/")
            return
        log.info(f"--latest{'(-pretrain)' if args.latest_pretrain else ''}: resolved to {_latest_path}")
        if args.latest_pretrain:
            # Route through pretrain-from-checkpoint logic instead of resume
            args.pretrain_from_checkpoint = str(_latest_path)
        else:
            if args.checkpoint:
                log.warning("--latest overrides --checkpoint")
            args.checkpoint = str(_latest_path)

    def _reset_dir_log_std() -> None:
        ppo_cfg_r = cfg.get("ppo", {})
        move_init = float(ppo_cfg_r.get("dir_log_std_init", -2.0))
        kick_init = float(ppo_cfg_r.get("kick_dir_log_std_init", move_init))
        with torch.no_grad():
            trainer.execution_net.move_dir_log_std.fill_(move_init)
            trainer.execution_net.kick_dir_log_std.fill_(kick_init)
        log.info(f"--reset-dir-log-std: move_dir_log_std={move_init}  kick_dir_log_std={kick_init}")

    # Optionally resume from checkpoint
    if args.checkpoint:
        trainer.load_checkpoint(Path(args.checkpoint))
        if args.reset_dir_log_std:
            _reset_dir_log_std()

    # --pretrain-from-checkpoint: load weights but still run pretraining
    if args.pretrain_from_checkpoint:
        ptrain_path = Path(args.pretrain_from_checkpoint)
        if not ptrain_path.exists():
            log.error(f"--pretrain-from-checkpoint: file not found: {ptrain_path}")
            return
        trainer.load_checkpoint(ptrain_path)
        trainer._total_steps = 0  # reset step counter so pretraining + full PPO run from scratch
        log.info(f"Loaded checkpoint for re-pretraining: {ptrain_path} — will still run BC/value pre-training")
        if args.reset_dir_log_std:
            _reset_dir_log_std()
        if not args.bc_dataset:
            log.warning(
                "--pretrain-from-checkpoint used without --bc-dataset: "
                "will fall back to noisy online BC pre-training. "
                "Pass --bc-dataset demonstrations/phase1/ for full combined pre-training."
            )

    # --from-pretrained: load a pre-trained checkpoint and skip all pre-training
    if args.from_pretrained:
        pretrained_path = Path(args.from_pretrained)
        if pretrained_path.is_dir():
            # Auto-discover: use checkpoint_pretrained.pt in that directory
            candidate = pretrained_path / "checkpoint_pretrained.pt"
            if not candidate.exists():
                log.error(f"--from-pretrained: no checkpoint_pretrained.pt found in {pretrained_path}")
                return
            pretrained_path = candidate
        if not pretrained_path.exists():
            log.error(f"--from-pretrained: file not found: {pretrained_path}")
            return
        trainer.load_checkpoint(pretrained_path)
        if args.reset_dir_log_std:
            _reset_dir_log_std()
        log.info(f"Loaded pre-trained checkpoint: {pretrained_path} — skipping BC/value pre-training")

    # Pre-training phase: BC + value jointly when a dataset is available,
    # otherwise fall back to online BC then separate value pre-training.
    value_pretrain_steps = int(bc_cfg.get("value_pretrain_steps", 0))
    combined_pretrain_rollout_steps = int(bc_cfg.get("combined_pretrain_rollout_steps", 0))
    value_pretrain_epochs = int(bc_cfg.get("value_pretrain_epochs", 20))
    value_pretrain_lr = float(bc_cfg.get("value_pretrain_lr", 1e-3))

    if not args.checkpoint and not args.from_pretrained:
        from footballcoach.ai.bc.dataset import DemonstrationDataset

        # Computed before deciding whether to load the dataset file (below)
        # so the decision reflects any bc_pretrain_epochs_from_ckpt /
        # demo_value_pretrain_epochs_from_ckpt swap already applied to
        # bc_cfg in-place above (--pretrain-from-checkpoint / --latest-pretrain).
        bc_pretrain_epochs = (
            args.bc_pretrain_epochs
            if args.bc_pretrain_epochs is not None
            else int(bc_cfg.get("bc_pretrain_epochs", 30))
        )
        _demo_epochs_eff = int(bc_cfg.get("demo_value_pretrain_epochs", 0))

        # Loading .npz demonstration files (DemonstrationDataset.from_directory)
        # can take ~1 minute on a large dataset directory. If both BC-epoch
        # counts that would actually consume it are 0 (after any _from_ckpt
        # override), the dataset would never be touched -- skip loading it
        # entirely and run value-only pre-training instead (identical to what
        # pretrain_combined()'s Phase 2/3 would do -- see below).
        dataset = None
        _skip_dataset_load = bool(args.bc_dataset) and bc_pretrain_epochs == 0 and _demo_epochs_eff == 0
        if args.bc_dataset and not _skip_dataset_load:
            dataset = DemonstrationDataset.from_directory(args.bc_dataset)
            log.info(f"Offline BC dataset: {len(dataset):,} steps from {args.bc_dataset}")
        elif _skip_dataset_load:
            log.info(
                f"Skipping BC dataset load ({args.bc_dataset}): bc_pretrain_epochs=0 and "
                f"demo_value_pretrain_epochs=0 (after any _from_ckpt override), so the dataset "
                f"would never be used -- running value-only pre-training instead."
            )

        bc_pretrain_batch_size = (
            args.bc_pretrain_batch_size
            if args.bc_pretrain_batch_size is not None
            else int(bc_cfg.get("bc_pretrain_batch_size", 256))
        )

        if _skip_dataset_load:
            # Value-only pre-training: exactly what pretrain_combined()'s
            # Phase 2/3 would do (it's a thin wrapper over this same
            # pretrain_value() call, see ppo_trainer.py) -- reusing
            # combined_pretrain_rollout_steps for n_steps so the rollout size
            # matches what Phase 2/3 would have used, without needing the BC
            # dataset loaded (Phase 0/1 are skipped entirely since their
            # epoch counts are 0, per _skip_dataset_load's condition above).
            trainer.pretrain_value(
                env,
                n_steps=combined_pretrain_rollout_steps,
                n_epochs=max(1, value_pretrain_epochs),
                lr=value_pretrain_lr,
                batch_size=bc_pretrain_batch_size,
                experiment_separate_value_net=args.experiment_separate_value_net,
                phase_id=args.phase,
            )
        elif dataset is not None and label_fn is not None:
            # Diagnostic: run 40 rules-vs-rules episodes before any training to show
            # the baseline reward component breakdown. Helps calibrate reward shaping.
            _diag_n = 40
            log.info(f"Reward diagnostic: running {_diag_n} rules-vs-rules episodes...")
            try:
                from footballcoach.ui.scenarios import ScenarioDefinition
                from footballcoach.ai.env.scenario_env import ScenarioEnv
                _diag_defn = ScenarioDefinition(key="diag_rr", label="diag", description="rules vs rules diagnostic", build=_rules_vs_rules_1v1_build)
                _diag_env = ScenarioEnv(_diag_defn, trainee_player_id="trainee", phase=1, max_episode_s=env.max_episode_s)
                _comp_acc: dict[str, float] = {}
                _ep_rewards: list[float] = []
                for _ in range(_diag_n):
                    _diag_env.reset()
                    _ep_r = 0.0
                    _done = False
                    while not _done:
                        _, _r, _done, _ = _diag_env.step()
                        _ep_r += _r
                        for _k, _v in _diag_env.last_reward_components.items():
                            _comp_acc[_k] = _comp_acc.get(_k, 0.0) + _v
                    _ep_rewards.append(_ep_r)
                from footballcoach.ai.ppo.ppo_trainer import REWARD_COMP_LABELS as _CL
                _key_order = {k: i for i, (k, _) in enumerate(_CL)}
                _comp_sorted = sorted(
                    _comp_acc.items(),
                    key=lambda x: _key_order.get(x[0], 999),
                )
                _cl_map = dict(_CL)
                _comp_str = "  ".join(
                    f"{_cl_map.get(k, k)}={v / _diag_n:+.2f}"
                    for k, v in _comp_sorted
                    if abs(v / _diag_n) > 0.005
                )
                log.info(
                    f"Reward diagnostic ({_diag_n} ep, rules vs rules): "
                    f"mean_ep_rew={sum(_ep_rewards) / _diag_n:.2f}  "
                    f"per_episode: {_comp_str}"
                )
            except Exception as _diag_exc:
                log.warning(f"Reward diagnostic failed (non-fatal): {_diag_exc}")

            # Combined joint pre-training (BC + value in one pass)
            trainer.pretrain_combined(
                env=env,
                dataset=dataset,
                n_epochs=bc_pretrain_epochs,
                batch_size=bc_pretrain_batch_size,
                bc_lr=float(bc_cfg.get("bc_learning_rate", bc_cfg.get("pretrain_lr", 3e-4))),
                value_lr=value_pretrain_lr,
                repair_lr=float(bc_cfg.get("bc_repair_lr", bc_cfg.get("bc_learning_rate", bc_cfg.get("pretrain_lr", 3e-4)))),
                rollout_steps=combined_pretrain_rollout_steps,
                value_epochs=value_pretrain_epochs,
                experiment_separate_value_net=args.experiment_separate_value_net,
                phase_id=args.phase,
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
                    experiment_separate_value_net=args.experiment_separate_value_net,
                    phase_id=args.phase,
                )

    # Save pre-trained checkpoint NOW so parallel pre-PPO eval workers can reload it from disk.
    # (parallel eval requires a file path — live nn.Modules aren't picklable across subprocesses)
    _eval_ckpt_path: str | None = None
    if not args.checkpoint and not args.from_pretrained:
        _pretrained_path = checkpoint_dir / "checkpoint_pretrained.pt"
        trainer._save_checkpoint_to(_pretrained_path)
        log.info(f"Pre-trained checkpoint saved: {_pretrained_path}")
        _eval_ckpt_path = str(_pretrained_path)
    elif args.checkpoint:
        _eval_ckpt_path = args.checkpoint
    elif args.from_pretrained:
        _fp = Path(args.from_pretrained)
        _eval_ckpt_path = str(_fp / "checkpoint_pretrained.pt" if _fp.is_dir() else _fp)

    if args.pre_ppo_eval_trials != 0:
        _pre_ppo_n_seeds = args.pre_ppo_eval_trials  # None = fall back to ai_config.json eval.eval_n_seeds
        from footballcoach.ai.scripts.evaluate import _run_evaluation
        from footballcoach.ui.scenarios import ScenarioDefinition
        from footballcoach.ai.env.scenario_env import ScenarioEnv

        # Evaluate against rules-based opponent only
        rules_defn = ScenarioDefinition(key="1v1_rules", label="1v1 rules", description="1v1 vs rules-based opponent", build=_rules_1v1_build)
        rules_env = ScenarioEnv(
            rules_defn, trainee_player_id="trainee", phase=1,
            max_episode_s=env.max_episode_s,
        )
        rules_stats = _run_evaluation(trainer, rules_env, _pre_ppo_n_seeds, checkpoint_path=_eval_ckpt_path)
        from footballcoach.ai.ppo.ppo_trainer import REWARD_COMP_LABELS as _CL
        _cl_map = dict(_CL)
        _comp_str = "  ".join(
            f"{_cl_map.get(k, k)}={v:+.2f}" for k, v in sorted(
                rules_stats.get("reward_components", {}).items(), key=lambda x: -abs(x[1])
            ) if abs(v) > 0.005
        )
        log.info(
            f"Pre-PPO eval (rules opp): win={rules_stats['win_rate_pct']:.1f}%  "
            f"mean_rew={rules_stats['mean_reward']:.3f}  "
            f"V={rules_stats['mean_step_v']:.3f}  R={rules_stats['mean_step_r']:.3f}  "
            f"gap={rules_stats['mean_step_v'] - rules_stats['mean_step_r']:+.3f}  "
            f"outcomes={rules_stats['outcomes']}"
        )
        if _comp_str:
            log.info(f"  rew breakdown (rules, per ep): {_comp_str}")

        # Evaluate against immobile opponent only
        immobile_defn = ScenarioDefinition(key="1v1_immobile", label="1v1 immobile", description="1v1 vs immobile opponent", build=_immobile_1v1_build)
        immobile_env = ScenarioEnv(
            immobile_defn, trainee_player_id="trainee", phase=1,
            max_episode_s=env.max_episode_s,
        )
        immobile_stats = _run_evaluation(trainer, immobile_env, _pre_ppo_n_seeds, checkpoint_path=_eval_ckpt_path)
        _imm_comp_str = "  ".join(
            f"{_cl_map.get(k, k)}={v:+.2f}" for k, v in sorted(
                immobile_stats.get("reward_components", {}).items(), key=lambda x: -abs(x[1])
            ) if abs(v) > 0.005
        )
        log.info(
            f"Pre-PPO eval (immobile opp): win={immobile_stats['win_rate_pct']:.1f}%  "
            f"mean_rew={immobile_stats['mean_reward']:.3f}  "
            f"V={immobile_stats['mean_step_v']:.3f}  R={immobile_stats['mean_step_r']:.3f}  "
            f"gap={immobile_stats['mean_step_v'] - immobile_stats['mean_step_r']:+.3f}  "
            f"outcomes={immobile_stats['outcomes']}"
        )
        if _imm_comp_str:
            log.info(f"  rew breakdown (immobile, per ep): {_imm_comp_str}")

        # Evaluate neural vs neural (self-play): the BC-pretrained policy plays itself.
        # Uses secondary_player_ids so the opponent also runs NeuralPlayerAI with the
        # same sample_action_fn — true self-play, no rules involvement.
        neural_defn = ScenarioDefinition(
            key="1v1_neural", label="1v1 self-play",
            description="1v1 neural vs neural (self-play)",
            build=_neural_1v1_build,
        )
        neural_env = ScenarioEnv(
            neural_defn, trainee_player_id="trainee", phase=1,
            max_episode_s=env.max_episode_s,
            secondary_player_ids=["opponent"],
        )
        neural_stats = _run_evaluation(trainer, neural_env, _pre_ppo_n_seeds, checkpoint_path=_eval_ckpt_path)
        _nn_comp_str = "  ".join(
            f"{_cl_map.get(k, k)}={v:+.2f}" for k, v in sorted(
                neural_stats.get("reward_components", {}).items(), key=lambda x: -abs(x[1])
            ) if abs(v) > 0.005
        )
        log.info(
            f"Pre-PPO eval (self-play):   win={neural_stats['win_rate_pct']:.1f}%  "
            f"mean_rew={neural_stats['mean_reward']:.3f}  "
            f"V={neural_stats['mean_step_v']:.3f}  R={neural_stats['mean_step_r']:.3f}  "
            f"gap={neural_stats['mean_step_v'] - neural_stats['mean_step_r']:+.3f}  "
            f"outcomes={neural_stats['outcomes']}"
        )
        if _nn_comp_str:
            log.info(f"  rew breakdown (self-play, per ep): {_nn_comp_str}")

    # Rules vs rules baseline (always runs, 12 trials)
    try:
        from footballcoach.ai.scripts.evaluate import _run_baseline_evaluation
        baseline_stats = _run_baseline_evaluation(env, n_trials=12)
        log.info(
            f"Baseline (rules vs rules, 12 trials): trainee_win={baseline_stats['win_rate_pct']:.1f}%  "
            f"outcomes={baseline_stats['outcomes']}"
        )
    except Exception as _e:
        log.warning(f"Baseline eval failed: {_e}")

    # Apply curriculum head freezing (after pre-training, before PPO)
    if not args.no_head_freeze and phase.frozen_heads:
        trainer.set_frozen_heads(phase.frozen_heads)

    # PPO training (with optional BC aux loss if label_fn and aux_coeff > 0)
    aux_label_fn = None if (args.no_bc_aux or label_fn is None) else label_fn
    trainer.train(env, total_steps=args.total_steps, bc_label_fn=aux_label_fn, phase_id=args.phase)


# env building and label functions live in curriculum.envs — no duplication here.


if __name__ == "__main__":
    main()
