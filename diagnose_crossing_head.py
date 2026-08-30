"""Diagnostic: is the ball physics encoder's EXISTING, ALREADY-TRAINED
crossing_head (predicts out-of-bounds/goal crossing point + time -- see
project memory `project_physics_pretrain_crossing_head`) actually any good
at predicting ball-out timing on REAL match data?

No new training. No new model. Reuses exactly the same reconstruction path
`PhysicsEncoderValueNet.compute_features()` already uses
(`load_frozen_ball_encoder`, `ball_live_to_physics_input`,
`canonicalize_obs`) -- this script only adds pulling `crossing_head`'s own
raw output back out and comparing it against what actually happened in the
recorded episode, which nothing in the existing pipeline currently checks.

crossing_head's ball-side output is `(pos_x, pos_y, delta_t)` from a single
unmasked `Linear(latent_dim, 3)` (unlike the player's crossing_head, which
was split into a classifier + masked regression -- the ball's was NOT given
that fix). Ball position/velocity is NOT player-relative, so this evaluates
every row (both trainee-owned and opponent-owned rows are valid -- the ball
state is the same physical quantity regardless of whose row it's attached
to).

Usage:
    uv run python diagnose_crossing_head.py \\
        --ball-checkpoint checkpoints/physics_pretrain/ball_encoder_63.midtrain_latest_train.pt \\
        --data demonstrations/phase1_physics_smoketest
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from footballcoach.ai.bc.dataset import DemonstrationDataset
from footballcoach.ai.obs.canonical import canonicalize_obs, x_sign_of
from footballcoach.ai.physics_pretrain.live_encoder_features import (
    ball_live_to_physics_input,
    load_frozen_ball_encoder,
)

import math

# BallFeatures field indices (declaration order, see obs/schema.py).
_BF_POS_X, _BF_POS_Y, _BF_VEL_X, _BF_VEL_Y = 0, 1, 3, 4
_BF_IS_POSSESSED, _BF_IS_LOOSE = 9, 10
# GlobalFeatures.time_remaining_norm index.
_GF_TIME_REMAINING_NORM = 1
# BOTH ball position AND velocity are normalized by the pitch half-diagonal
# (matches ai/obs/encoder.py's actual pos_x=.../half_diag, NOT the
# per-axis 52.5/34.0 the schema.py docstrings used to (wrongly) claim --
# see schema.py's PlayerFeatures/BallFeatures docstrings for the fix and
# ai/knowledge.md for the debugging cost of trusting the stale comment).
_HALF_DIAG = math.hypot(52.5, 34.0)
_TIME_NORM_MAX_S = 7200.0  # matches PhysicsEncoderValueNet's default


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ball-checkpoint", type=str, required=True)
    p.add_argument("--data", type=str, default="demonstrations/phase1_physics_smoketest")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--no-canon", action="store_true",
                    help="Skip canonicalize_obs mirroring on ball_feat -- test "
                         "whether the ball encoder (trained on un-mirrored "
                         "world-frame ball_episode_gen data) is being fed "
                         "out-of-distribution input by the mirror.")
    args = p.parse_args()

    all_files = sorted(Path(args.data).glob("*.npz"))
    rng = random.Random(args.seed)
    rng.shuffle(all_files)
    val_files = all_files[: max(1, round(len(all_files) * args.val_frac))]
    print(f"loading {len(val_files)} val files...")
    ds = DemonstrationDataset.from_files(val_files)
    print(f"{len(ds):,} rows loaded")

    ball_encoder, ball_aux_heads, ball_cfg = load_frozen_ball_encoder(args.ball_checkpoint)
    if "crossing_head" not in ball_aux_heads:
        print(f"FATAL: ball_aux_heads has no 'crossing_head' key -- found: {list(ball_aux_heads.keys())}")
        return
    crossing_head = ball_aux_heads["crossing_head"]

    from footballcoach.ai.physics_pretrain.live_encoder_features import load_physics_pitch_constants
    pitch_constants = load_physics_pitch_constants()

    n = len(ds)
    pred_delta_t = np.empty(n, dtype=np.float32)
    pred_crosses_logit = np.empty(n, dtype=np.float32)
    time_remaining_s = np.empty(n, dtype=np.float64)
    ballistic_will_exit_10s = np.empty(n, dtype=bool)
    ball_speed = np.empty(n, dtype=np.float32)
    dist_to_boundary = np.empty(n, dtype=np.float32)
    is_possessed = np.empty(n, dtype=bool)
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            chunk = slice(start, min(start + args.batch_size, n))
            self_feat = torch.from_numpy(ds._self_feat[chunk]).float()
            other_feat = torch.from_numpy(ds._other_feat[chunk]).float()
            ball_feat = torch.from_numpy(ds._ball_feat[chunk]).float()
            global_feat = torch.from_numpy(ds._global_feat[chunk]).float()
            # Same canonicalization compute_features() applies -- x-mirroring
            # doesn't change WHETHER the ball crosses a boundary (symmetric),
            # only which physical side, but mirroring the input to match how
            # the encoder was actually fed data during training removes any
            # doubt about whether that matters.
            if args.no_canon:
                ball_feat_c = ball_feat
            else:
                x_sign = x_sign_of(self_feat)
                _, _, ball_feat_c, _ = canonicalize_obs(self_feat, other_feat, ball_feat, x_sign=x_sign)
            ball_in = ball_live_to_physics_input(ball_feat_c, global_feat, pitch_constants)
            latent = ball_encoder(ball_in)
            # (batch, 4): pos_x, pos_y, crosses_logit, delta_t -- crossing_head
            # was split into a classifier + masked regression (see
            # train_ball_dynamics._crossing_head_loss's docstring); delta_t
            # is now only meaningful for rows crosses_logit says will cross.
            out = crossing_head(latent)
            pred_crosses_logit[chunk] = out[:, 2].numpy()
            pred_delta_t[chunk] = out[:, 3].numpy()

            # Exact real seconds remaining in the episode -- inverts
            # GlobalFeatures.time_remaining_norm's log1p compression, same
            # formula time_remaining_and_elapsed_fraction() uses internally
            # (see live_encoder_features.py), just kept in raw seconds here
            # instead of a fraction.
            norm = global_feat[:, _GF_TIME_REMAINING_NORM]
            time_remaining_s[chunk] = torch.expm1(norm * math.log1p(_TIME_NORM_MAX_S)).numpy()

            # Closed-form ballistic check (NOT crossing_head, NOT trained --
            # pure linear extrapolation from raw position/velocity, both
            # identity-shortcut-preserved in the latent per the class
            # docstring): does the ball, continuing in a straight line at
            # its CURRENT velocity, exit the standard pitch boundary
            # (+-52.5m x, +-34.0m y) within 10 seconds?
            px = ball_feat[:, _BF_POS_X] * _HALF_DIAG
            py = ball_feat[:, _BF_POS_Y] * _HALF_DIAG
            vx = ball_feat[:, _BF_VEL_X] * _HALF_DIAG
            vy = ball_feat[:, _BF_VEL_Y] * _HALF_DIAG

            def _t_to_wall(pos, vel, bound):
                # time to reach `bound` moving at `vel` from `pos`; inf if
                # not moving toward that wall at all.
                t = (bound - pos) / vel
                return torch.where((vel != 0) & (t > 0), t, torch.full_like(t, float("inf")))

            t_exit = torch.minimum(
                torch.minimum(_t_to_wall(px, vx, 52.5), _t_to_wall(px, vx, -52.5)),
                torch.minimum(_t_to_wall(py, vy, 34.0), _t_to_wall(py, vy, -34.0)),
            )
            ballistic_will_exit_10s[chunk] = (t_exit <= 10.0).numpy()
            ball_speed[chunk] = torch.hypot(vx, vy).numpy()
            dist_to_boundary[chunk] = torch.minimum(
                torch.minimum(52.5 - px, 52.5 + px), torch.minimum(34.0 - py, 34.0 + py),
            ).numpy()
            is_possessed[chunk] = (ball_feat[:, _BF_IS_POSSESSED] > 0.5).numpy()

    print(f"\npred_delta_t over ALL {n:,} rows: "
          f"mean={pred_delta_t.mean():+.3f} std={pred_delta_t.std():.3f} "
          f"min={pred_delta_t.min():+.3f} max={pred_delta_t.max():+.3f} "
          f"p10={np.percentile(pred_delta_t, 10):+.3f} p50={np.percentile(pred_delta_t, 50):+.3f} "
          f"p90={np.percentile(pred_delta_t, 90):+.3f}")

    # --- Ground truth, restricted to "invalid" outcome episodes ONLY.
    # "ball_out" = the ball goes out WITH a toucher -- i.e. the PLAYER hit
    # it out (see ScenarioEnv's outcome vocabulary / _OUTCOME_LABEL_MAP:
    # raw "miss" -> "ball_out"). That's actively player-controlled right up
    # to the event (confirmed: ball_out episodes averaged 1.28 recorded
    # kicks), NOT the clean free-flight trajectory crossing_head was
    # actually trained on. "invalid" = the ball goes out with NO toucher --
    # untouched physics, no player intervention -- the correct, matching
    # test case. Using "ball_out" for this test was wrong; this section
    # redoes it against "invalid" instead. ---
    ranges = ds.episode_row_ranges(np.arange(n))
    seconds_until_end = np.full(n, -1.0, dtype=np.float64)
    is_invalid_episode = np.zeros(n, dtype=bool)
    for s, e in ranges:
        outcome = ds.classify_outcome(e) if ds.has_episode_outcomes else "unknown"
        seconds_until_end[s:e + 1] = time_remaining_s[s:e + 1] - time_remaining_s[e]
        if outcome == "invalid":
            is_invalid_episode[s:e + 1] = True

    inv = is_invalid_episode & (seconds_until_end >= 0)
    print(f"\n'invalid'-episode rows: {inv.sum():,}")

    in_horizon = inv & (seconds_until_end <= 10.0)
    beyond_horizon = inv & (seconds_until_end > 10.0)
    print(f"  within 10s of the actual end: {in_horizon.sum():,}   beyond 10s: {beyond_horizon.sum():,}")

    # 1) Correlation between predicted delta_t and actual real seconds
    # until the ball actually exits -- rows within the 10s horizon only.
    if in_horizon.sum() > 10:
        corr = float(np.corrcoef(pred_delta_t[in_horizon], seconds_until_end[in_horizon])[0, 1])
        print(f"\n  corr(pred_delta_t, actual seconds_until_end), invalid episodes, within 10s: {corr:+.3f}")

    # 2) Boolean accuracy: predicted_crossing = (pred_delta_t > 0),
    # true_crossing = (this row is within the 10s horizon of the actual
    # exit). Evaluated over ALL invalid-episode rows (both within and
    # beyond the horizon), so this is a genuine 2x2 confusion matrix, not
    # just a recall number on an all-positive set.
    # crosses_logit is the trained classifier now (crossing_head split into
    # a classifier + masked regression) -- use it directly rather than the
    # sign of delta_t, which was only ever an implicit proxy for this.
    pred_crossing = pred_crosses_logit[inv] > 0
    true_crossing = seconds_until_end[inv] <= 10.0
    accuracy = float((pred_crossing == true_crossing).mean())
    tp = int((pred_crossing & true_crossing).sum())
    fp = int((pred_crossing & ~true_crossing).sum())
    fn = int((~pred_crossing & true_crossing).sum())
    tn = int((~pred_crossing & ~true_crossing).sum())
    print(f"\n  boolean accuracy (crosses_logit>0 == 'crossing within 10s'), invalid episodes: {accuracy:.1%}")
    print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    fn_mask_local = (~pred_crossing) & true_crossing
    tp_mask_local = pred_crossing & true_crossing
    logits_inv = pred_crosses_logit[inv]
    if fn_mask_local.sum() == 0:
        print("\n  no FN rows -- perfect recall on this checkpoint, skipping FN-vs-TP breakdown.")
    else:
        print(f"\n  crosses_logit on FN rows: mean={logits_inv[fn_mask_local].mean():+.3f} "
              f"std={logits_inv[fn_mask_local].std():.3f} "
              f"p90={np.percentile(logits_inv[fn_mask_local], 90):+.3f} "
              f"max={logits_inv[fn_mask_local].max():+.3f}")
        print(f"  crosses_logit on TP rows: mean={logits_inv[tp_mask_local].mean():+.3f} "
              f"std={logits_inv[tp_mask_local].std():.3f}")

        speed_inv = ball_speed[inv]
        dist_inv = dist_to_boundary[inv]
        print(f"\n  ball speed (m/s): FN mean={speed_inv[fn_mask_local].mean():.2f} "
              f"median={np.median(speed_inv[fn_mask_local]):.2f}  |  "
              f"TP mean={speed_inv[tp_mask_local].mean():.2f} median={np.median(speed_inv[tp_mask_local]):.2f}")
        print(f"  dist to nearest boundary (m): FN mean={dist_inv[fn_mask_local].mean():.2f} "
              f"median={np.median(dist_inv[fn_mask_local]):.2f}  |  "
              f"TP mean={dist_inv[tp_mask_local].mean():.2f} median={np.median(dist_inv[tp_mask_local]):.2f}")

        poss_inv = is_possessed[inv]
        print(f"\n  fraction with is_possessed=True (a player currently has the ball -- "
              f"NOT free-flight physics, outside crossing_head's trained domain):")
        print(f"    FN rows: {poss_inv[fn_mask_local].mean():.1%}  |  TP rows: {poss_inv[tp_mask_local].mean():.1%}  |  "
              f"all invalid-episode rows: {poss_inv.mean():.1%}")

        ballistic_inv = ballistic_will_exit_10s[inv]
        print(f"\n  fraction where a NAIVE straight-line/constant-velocity extrapolation "
              f"ALSO says 'exits within 10s' (ballistic_will_exit_10s):")
        print(f"    FN rows: {ballistic_inv[fn_mask_local].mean():.1%}  |  TP rows: {ballistic_inv[tp_mask_local].mean():.1%}")


if __name__ == "__main__":
    main()
