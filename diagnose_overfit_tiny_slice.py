"""Diagnostic: can PhysicsEncoderValueNet's trainable MLP head even MEMORIZE
a small, fixed slice of real data? Classic first move in ML debugging: if a
~10k-parameter MLP can't drive train loss to near-zero on a few hundred
FIXED examples (no val split, no resampling, no generalization required),
something in the loss/gradient/data path is definitively broken. If it CAN,
that rules out wiring bugs entirely -- the real question becomes
generalization/representation adequacy, not plumbing.

No new model, no new physics -- reuses PhysicsEncoderValueNet exactly as
debug_value_network.py builds it.

Usage:
    uv run python diagnose_overfit_tiny_slice.py \\
        --ball-checkpoint checkpoints/physics_pretrain/ball_encoder_63.midtrain_latest_train.pt \\
        --player-checkpoint checkpoints/physics_pretrain/player_encoder_45.midtrain_latest_train.pt \\
        --data demonstrations/phase1_physics_smoketest --n-rows 1000 --epochs 3000
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from footballcoach.ai.bc.dataset import DemonstrationDataset, _to_tensor
from footballcoach.ai.physics_pretrain.physics_value_net import PhysicsEncoderValueNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ball-checkpoint", type=str, required=True)
    p.add_argument("--player-checkpoint", type=str, required=True)
    p.add_argument("--data", type=str, default="demonstrations/phase1_physics_smoketest")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-rows", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n-files", type=int, default=20, help="how many files to load before subsampling rows")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    all_files = sorted(Path(args.data).glob("*.npz"))
    rng = random.Random(args.seed)
    rng.shuffle(all_files)
    files = all_files[: args.n_files]
    print(f"loading {len(files)} files...")
    ds = DemonstrationDataset.from_files(files)
    valid_idx = ds.valid_indices()
    print(f"{len(ds):,} rows loaded, {len(valid_idx):,} valid")

    rng_np = np.random.default_rng(args.seed)
    n_rows = min(args.n_rows, len(valid_idx))
    row_idx = rng_np.choice(valid_idx, size=n_rows, replace=False)
    row_idx.sort()
    print(f"overfitting on a FIXED {n_rows} row subset (no val split, full-batch every epoch)")

    returns = ds.compute_returns(gamma=args.gamma)
    ret_batch = _to_tensor(returns[row_idx], device)

    value_net = PhysicsEncoderValueNet.from_checkpoints(
        args.ball_checkpoint, args.player_checkpoint, mlp_hidden=64,
    ).to(device)
    n_params = sum(pp.numel() for pp in value_net.mlp.parameters())
    print(f"value_net.mlp trainable_params={n_params:,}")

    # Precompute once -- compute_features() has no trainable-parameter
    # dependency, so the frozen-encoder pass only needs to happen once for
    # this fixed subset, same optimization debug_value_network.py already
    # relies on.
    self_feat = _to_tensor(ds._self_feat[row_idx], device)
    other_feat = _to_tensor(ds._other_feat[row_idx], device)
    ball_feat = _to_tensor(ds._ball_feat[row_idx], device)
    global_feat = _to_tensor(ds._global_feat[row_idx], device)
    labels = _to_tensor(ds._labels[row_idx], device)
    exists_mask = _to_tensor(ds._exists_mask[row_idx], device)
    with torch.no_grad():
        features = value_net.compute_features(self_feat, other_feat, ball_feat, global_feat, labels, exists_mask)

    optimizer = torch.optim.Adam(value_net.mlp.parameters(), lr=args.lr)
    print(f"\ntarget return stats: mean={ret_batch.mean().item():+.3f} std={ret_batch.std().item():.3f}")

    log_every = max(1, args.epochs // 20)
    for epoch in range(1, args.epochs + 1):
        pred = value_net.mlp(features).squeeze(-1)
        loss = ((pred - ret_batch) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % log_every == 0 or epoch == 1:
            rmse = float(loss.sqrt())
            norm = float(loss) / (ret_batch.var().item() + 1e-8)
            print(f"epoch {epoch:>5}/{args.epochs}  train_mse={float(loss):.5f}  "
                  f"train_rmse={rmse:.5f}  normalized_mse={norm:.5f}")

    final_mse = float(loss)
    final_norm = final_mse / (ret_batch.var().item() + 1e-8)
    print(f"\n=== FINAL: train_mse={final_mse:.6f}  normalized_mse={final_norm:.6f} ===")
    if final_norm < 0.05:
        print("VERDICT: memorized cleanly (normalized_mse << 1) -- the training/gradient/loss "
              "wiring is NOT broken. The val-time ceiling is a generalization/representation "
              "question, not a plumbing bug.")
    elif final_norm < 0.3:
        print("VERDICT: mostly memorized but not clean -- some real friction in optimization "
              "even on a trivial fixed target; worth a closer look (lr? clip? feature scale?).")
    else:
        print("VERDICT: could NOT memorize a fixed 1000-row subset even after thousands of "
              "epochs -- this points to something genuinely broken in the loss/gradient/data "
              "path, not a hard generalization problem.")


if __name__ == "__main__":
    main()
