"""Small BC learning-rate sweep from the most recent checkpoint.

Loads a fresh copy of the latest checkpoint per LR (so each run starts from
the exact same weights, not contaminated by a previous LR's updates), trains
for a couple epochs over a SUBSET of the demonstration dataset (a handful of
files, not the full multi-GB set), and reports the resulting loss trajectory
per LR -- a quick, cheap read on "does this LR converge stably, blow up, or
barely move" without committing to a full run.

Reuses dagger.py's train_bc_epochs() (the same production BC training step
Phase 1/DAgger use) rather than reimplementing the training loop, so the
loss/grad-norm/head-breakdown diagnostics are directly comparable to real
training logs.

Usage:
    uv run python debug_lr_sweep.py
    uv run python debug_lr_sweep.py --lrs 1e-3 1e-4 1e-5 1e-6 --n-episodes 3000 --epochs 2
"""
from __future__ import annotations

import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import glob
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("footballcoach.ai.debug_lr_sweep")


def _find_latest_checkpoint(checkpoints_dir: str = "checkpoints") -> Path:
    candidates = [
        p for pattern in ("*/checkpoint_pretrained.pt", "*/checkpoint_final.pt", "*/latest.pt")
        for p in Path(checkpoints_dir).glob(pattern)
    ]
    if not candidates:
        raise SystemExit(f"No checkpoint files found under {checkpoints_dir}/*/ -- pass --checkpoint explicitly.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default=None, help="Default: newest checkpoint under checkpoints/*/.")
    parser.add_argument("--demonstrations", type=str, default="demonstrations/phase1_rules_immobile")
    parser.add_argument("--n-episodes", type=int, default=3000,
                         help="Approximate episode count to train on (rounded to whole demo files). Default: 3000.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 1e-4, 1e-5, 1e-6])
    parser.add_argument("--batch-size", type=int, default=None, help="Default: bc.bc_pretrain_batch_size from config.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(args.device) if args.device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    log.info(f"Device: {device}")

    from footballcoach.ai.bc.dataset import DemonstrationDataset
    from footballcoach.ai.config import load_ai_config
    from footballcoach.ai.ppo.dagger import train_bc_epochs
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer

    checkpoint_path = Path(args.checkpoint) if args.checkpoint is not None else _find_latest_checkpoint()
    log.info(f"Checkpoint: {checkpoint_path}")

    cfg = load_ai_config()
    batch_size = args.batch_size if args.batch_size is not None else int(cfg["bc"]["bc_pretrain_batch_size"])

    # ~250 episodes/file is typical for this dataset (confirmed via a direct
    # check on file 0 earlier this session) -- rounded to whole files, not
    # trying to hit --n-episodes exactly. Loading only a handful of files
    # (not the full ~154-file, multi-GB dataset via from_directory) keeps
    # this cheap and avoids competing for RAM with anything else running.
    all_files = sorted(glob.glob(f"{args.demonstrations}/*.npz"))
    n_files = max(1, round(args.n_episodes / 250))
    chosen_files = all_files[:n_files]
    log.info(f"Loading {len(chosen_files)} demo file(s) (targeting ~{args.n_episodes} episodes)...")
    ds = DemonstrationDataset.from_files(chosen_files)
    train_idx = ds.valid_indices()
    n_eps = ds.n_episodes(train_idx)
    log.info(f"Loaded {len(train_idx)} valid rows / {n_eps} episode(s) -- batch_size={batch_size}")

    # pos_weight: compute ONCE from this subset (same auto-compute logic
    # pretrain_combined() runs), reused identically across every LR run
    # below so differences in the results are only ever about LR, not a
    # different pos_weight draw each time.
    auto_weights = ds.compute_pos_weights()
    log.info(f"pos_weight (from this subset): kick={auto_weights['kick']:.2f}  tackle_attempt={auto_weights['tackle_attempt']:.2f}")

    results: dict[float, tuple[float, float, int]] = {}
    for lr in args.lrs:
        log.info(f"########## LR={lr:.0e} ##########")
        trainer = PPOTrainer.from_config(device=device, inference_only=True)
        trainer.load_checkpoint(checkpoint_path)
        trainer.optimizer = torch.optim.Adam(
            list(trainer.decision_net.parameters()) + list(trainer.execution_net.parameters()),
            lr=lr, eps=1e-5,
        )
        trainer._bc_pos_weight_kick = auto_weights["kick"]
        trainer._bc_pos_weight_tackle_attempt = auto_weights["tackle_attempt"]

        mean_loss, floor, epochs_run = train_bc_epochs(
            trainer, ds, train_idx, trainer.optimizer,
            max_epochs=args.epochs, batch_size=batch_size, log_every=1,
        )
        results[lr] = (mean_loss, floor, epochs_run)
        log.info(f"########## LR={lr:.0e} done: mean_loss={mean_loss:.4f}  floor={floor:.4f}  bc_adj={mean_loss - floor:.4f} ##########")

    log.info("========== SWEEP SUMMARY ==========")
    for lr, (mean_loss, floor, epochs_run) in results.items():
        log.info(f"  LR={lr:.0e}: final mean_loss={mean_loss:.4f}  floor={floor:.4f}  bc_adj={mean_loss - floor:.4f}  ({epochs_run} epoch(s))")


if __name__ == "__main__":
    main()
