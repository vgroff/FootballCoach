"""CLI: train the ball-dynamics encoder from a generated .npz dataset.

> **Documentation must stay in sync with code.** Any significant change, and
> any change that conflicts with existing documentation, must be followed by
> additions or edits to the relevant documentation (this file, other
> knowledge.md files, design docs, plans). When writing plans, design
> documents, prompts, or other work-related files, always include this same
> statement at the top of that file. Otherwise documentation goes stale and
> confusion occurs.

See agent_plans/ball_physics_pretrain_plan.md section 7.

Usage::

    uv run python -m footballcoach.ai.physics_pretrain.train_ball_dynamics \\
        --dataset physics_pretrain_data/ball/ \\
        --output checkpoints/physics_pretrain/ball_encoder.pt \\
        --epochs 50 --batch-size 1024

To generate the dataset first, see ``ball_dataset.py``'s own ``__main__``
entry point (``python -m footballcoach.ai.physics_pretrain.ball_dataset
--help``).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from footballcoach.ai.physics_pretrain.ball_dataset import BallDynamicsDataset
from footballcoach.ai.physics_pretrain.ball_dynamics_net import BallDynamicsAutoencoder
from footballcoach.ai.physics_pretrain.ball_episode_gen import N_TARGET_FIELDS_PER_HORIZON

log = logging.getLogger("footballcoach.ai.physics_pretrain.train_ball_dynamics")


class LossBreakdown:
    """Per-horizon loss components, each a length-``n_horizons`` list of floats.

    ``pos_mse``/``vel_mse``/``spin_mse`` are reported SEPARATELY (not one
    blended "continuous MSE") because they're different physical quantities
    on different normalized scales (position vs velocity vs spin) -- a
    single combined number can look flat while one of the three is actually
    moving a lot and another is stuck, exactly the same masking problem as
    merging out_of_bounds/goal_scored BCE (see ``oob_bce``/``goal_bce``
    below).
    """
    __slots__ = ("pos_mse", "vel_mse", "spin_mse", "oob_bce", "goal_bce")

    def __init__(self):
        self.pos_mse: list[float] = []
        self.vel_mse: list[float] = []
        self.spin_mse: list[float] = []
        self.oob_bce: list[float] = []
        self.goal_bce: list[float] = []


def compute_loss(
    pred_heads: list[torch.Tensor], target: torch.Tensor, pos_weight: torch.Tensor,
) -> tuple[torch.Tensor, LossBreakdown]:
    """Sum over horizons of (continuous MSE + event BCE), per §5.

    ``pos_weight``: ``(n_horizons, 2)`` tensor from
    ``BallDynamicsDataset.compute_pos_weights()``.

    Returns ``(total, breakdown)`` where ``breakdown`` is a
    ``LossBreakdown`` with 5 SEPARATE per-horizon components (position MSE,
    velocity MSE, spin MSE, out_of_bounds BCE, goal_scored BCE) -- never
    merged into fewer numbers, since each hides a different quantity's own
    learning progress (see ``LossBreakdown``'s docstring).
    """
    total = target.new_zeros(())
    breakdown = LossBreakdown()
    for h, head_out in enumerate(pred_heads):
        base = h * N_TARGET_FIELDS_PER_HORIZON
        target_h = target[:, base:base + N_TARGET_FIELDS_PER_HORIZON]
        pos_mse = F.mse_loss(head_out[:, 0:3], target_h[:, 0:3])
        vel_mse = F.mse_loss(head_out[:, 3:6], target_h[:, 3:6])
        spin_mse = F.mse_loss(head_out[:, 6:9], target_h[:, 6:9])
        oob_bce = F.binary_cross_entropy_with_logits(
            head_out[:, 9], target_h[:, 9], pos_weight=pos_weight[h, 0],
        )
        goal_bce = F.binary_cross_entropy_with_logits(
            head_out[:, 10], target_h[:, 10], pos_weight=pos_weight[h, 1],
        )
        total = total + pos_mse + vel_mse + spin_mse + oob_bce + goal_bce
        breakdown.pos_mse.append(float(pos_mse.item()))
        breakdown.vel_mse.append(float(vel_mse.item()))
        breakdown.spin_mse.append(float(spin_mse.item()))
        breakdown.oob_bce.append(float(oob_bce.item()))
        breakdown.goal_bce.append(float(goal_bce.item()))
    return total, breakdown


def _physics_config_hash() -> str:
    """Hash of physics.json's ball_physics section, saved alongside the
    checkpoint so a later load can warn if the real physics has drifted
    since this encoder was trained (§7, §8.4's staleness check)."""
    from footballcoach.config import load_physics_config, require_section
    section = require_section(load_physics_config(), "ball_physics")
    return hashlib.sha256(json.dumps(section, sort_keys=True).encode()).hexdigest()[:16]


def train(
    dataset_dir: str,
    output_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    val_frac: float = 0.15,
    seed: int = 0,
    pos_weight_max: float | None = None,
    device: str = "cpu",
) -> dict:
    from footballcoach.ai.config import load_ai_config
    cfg = load_ai_config()["physics_pretrain"]["ball"]

    ds = BallDynamicsDataset.from_directory(dataset_dir)
    train_idx, val_idx = ds.split_train_val(val_frac=val_frac, seed=seed)
    log.info(f"Dataset: {len(ds):,} episodes ({len(train_idx):,} train / {len(val_idx):,} val)")

    n_horizons = len(cfg["horizons_s"])
    pos_weight_np = ds.compute_pos_weights(n_horizons, indices=train_idx, max_weight=pos_weight_max)
    pos_weight = torch.from_numpy(pos_weight_np).to(device)

    model = BallDynamicsAutoencoder.from_config().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    early_stop_patience = int(cfg.get("early_stop_patience", 5))
    early_stop_min_delta = float(cfg.get("early_stop_min_delta", 1e-4))
    early_stop_enabled = early_stop_patience > 0 and len(val_idx) > 0

    best_val_loss = float("inf")
    best_state: dict | None = None
    patience_ctr = 0
    stopped_early = False
    rng = np.random.default_rng(seed)

    # Full per-epoch history (every epoch, not just the log tail) -- saved
    # alongside the checkpoint below (as .history.npz) so it can be
    # inspected/plotted after the fact without re-parsing terminal output.
    history: list[dict] = []

    n_h = len(cfg["horizons_s"])
    _COMPONENTS = ("pos_mse", "vel_mse", "spin_mse", "oob_bce", "goal_bce")

    def _mean_breakdown(items: list[LossBreakdown]) -> dict[str, np.ndarray]:
        if not items:
            return {c: np.full(n_h, np.nan) for c in _COMPONENTS}
        return {c: np.mean([getattr(b, c) for b in items], axis=0) for c in _COMPONENTS}

    for epoch in range(epochs):
        model.train()
        train_losses: list[float] = []
        train_breakdowns: list[LossBreakdown] = []
        for x, y in ds.iterate_minibatches(batch_size, train_idx, shuffle=True, device=device, rng=rng):
            _, heads = model(x)
            loss, breakdown = compute_loss(heads, y, pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            train_breakdowns.append(breakdown)
        mean_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        train_means = _mean_breakdown(train_breakdowns)

        val_line = ""
        val_loss = float("nan")
        val_means = _mean_breakdown([])
        if len(val_idx) > 0:
            model.eval()
            val_losses: list[float] = []
            val_breakdowns: list[LossBreakdown] = []
            with torch.no_grad():
                for x, y in ds.iterate_minibatches(batch_size, val_idx, shuffle=False, device=device):
                    _, heads = model(x)
                    loss, breakdown = compute_loss(heads, y, pos_weight)
                    val_losses.append(float(loss.item()))
                    val_breakdowns.append(breakdown)
            val_loss = float(np.mean(val_losses))
            val_means = _mean_breakdown(val_breakdowns)
            improved = val_loss < (best_val_loss - early_stop_min_delta)
            val_line = f"  val_loss={val_loss:.4f}  best={min(best_val_loss, val_loss):.4f}"
            if early_stop_enabled:
                val_line += "  (improved)" if improved else f"  (patience {patience_ctr + 1}/{early_stop_patience})"
            if improved:
                best_val_loss = val_loss
                if early_stop_enabled:
                    best_state = copy.deepcopy(model.state_dict())
                    patience_ctr = 0
            elif early_stop_enabled:
                patience_ctr += 1

        log.info(
            f"epoch {epoch + 1}/{epochs}: train_loss={mean_train_loss:.4f}{val_line}"
        )
        for c in _COMPONENTS:
            log.info(f"    train {c:9s} by horizon: {np.array2string(train_means[c], precision=4)}")
        if len(val_idx) > 0:
            for c in _COMPONENTS:
                log.info(f"    val   {c:9s} by horizon: {np.array2string(val_means[c], precision=4)}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": mean_train_loss,
            "val_loss": val_loss,
            **{f"train_{c}": train_means[c] for c in _COMPONENTS},
            **{f"val_{c}": val_means[c] for c in _COMPONENTS},
        })

        if early_stop_enabled and patience_ctr >= early_stop_patience:
            log.info(f"Early stop at epoch {epoch + 1} (val stagnant for {early_stop_patience} epochs, best={best_val_loss:.4f})")
            stopped_early = True
            break

    if stopped_early and best_state is not None:
        model.load_state_dict(best_state)
        log.info(f"Restored best-val weights (val_loss={best_val_loss:.4f})")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "encoder_state_dict": model.encoder.state_dict(),
        "config_snapshot": cfg,
        "dataset_stats": {
            "n_episodes": len(ds),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
        },
        "physics_config_hash": _physics_config_hash(),
    }
    torch.save(artifact, output_path)
    log.info(f"Saved encoder checkpoint to {output_path}")

    history_path = output_path.with_suffix(".history.npz")
    history_arrays = {
        "epoch": np.array([h["epoch"] for h in history]),
        "train_loss": np.array([h["train_loss"] for h in history]),
        "val_loss": np.array([h["val_loss"] for h in history]),
        "horizons_s": np.array(cfg["horizons_s"]),
    }
    for c in _COMPONENTS:
        history_arrays[f"train_{c}"] = np.stack([h[f"train_{c}"] for h in history])
        history_arrays[f"val_{c}"] = np.stack([h[f"val_{c}"] for h in history])
    np.savez_compressed(history_path, **history_arrays)
    log.info(f"Saved per-epoch history to {history_path}")

    artifact["history"] = history
    return artifact


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from footballcoach.ai.config import load_ai_config
    cfg = load_ai_config()["physics_pretrain"]["ball"]

    parser = argparse.ArgumentParser(description="Train the ball-dynamics encoder (physics pretraining).")
    parser.add_argument("--dataset", required=True, help="Directory of .npz shards (see ball_dataset.py's __main__).")
    parser.add_argument("--output", required=True, help="Output path for the frozen encoder checkpoint.")
    parser.add_argument("--epochs", type=int, default=cfg["epochs"])
    parser.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    parser.add_argument("--lr", type=float, default=cfg["lr"])
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pos-weight-max", type=float, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train(
        dataset_dir=args.dataset,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
        seed=args.seed,
        pos_weight_max=args.pos_weight_max,
        device=args.device,
    )


if __name__ == "__main__":
    main()
