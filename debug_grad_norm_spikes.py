"""Offline gradient-norm-spike investigation.

Loads a trained checkpoint and runs its decision_net/execution_net forward/
backward over real recorded demonstration data -- WITHOUT ever calling
optimizer.step() (weights never move) -- so every minibatch is evaluated
against the exact same fixed network. Any grad-norm spike is then
attributable to the INPUT DATA for that minibatch alone, not to training-time
weight drift, which makes it a cleaner signal for "which real recorded
transition provokes an oversized gradient" than watching for spikes live
during actual training.

Reuses the exact same spike-detection/investigation machinery real BC/DAgger
training already uses (ppo_trainer.py's _measure_exec_head_grad_norms_and_
maybe_log_spike -> _maybe_log_grad_norm_spike -> _describe_weight_grad_columns)
-- nothing about the detection itself is reimplemented here. What this script
adds on top: when a spike fires, it maps the offending minibatch's rows back
to the recorded episode(s)/seed(s) they came from (via DemonstrationDataset's
own episode_row_ranges/episode_seed/classify_outcome), and prints a ready-to-
run replay_episode.py command so the exact moment can be watched directly.

Usage:
    uv run python debug_grad_norm_spikes.py
    uv run python debug_grad_norm_spikes.py --checkpoint checkpoints/phase1_run44/checkpoint_pretrained.pt
    uv run python debug_grad_norm_spikes.py --spike-k 2.5 --epochs 3
"""
from __future__ import annotations

# Must run BEFORE numpy/torch is first imported anywhere in this process --
# see debug_policy_net.py's identical block for the full rationale.
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import dataclasses
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("footballcoach.ai.debug_grad_norm_spikes")


def _forward_loss(trainer, obs_dict, bc_labels):
    """One BC forward pass + loss, identical to dagger.py's train_bc_epochs()
    / ppo_trainer.py's Phase 1 minibatch step -- factored out so both the
    main per-minibatch loop and the bisection search below (which needs to
    re-run this on shrinking row subsets) share one implementation."""
    from footballcoach.ai.obs.canonical import canonicalize_bc_labels, x_sign_of
    from footballcoach.ai.ppo.bc import bc_loss_from_tensor, direction_magnitude_reg
    from footballcoach.ai.ppo.ppo_trainer import _ai_types

    sat, oat = _ai_types(obs_dict)
    d_heads = trainer.decision_net(
        obs_dict["self_feat"], obs_dict["other_feat"], obs_dict["exists_mask"],
        obs_dict["ball_feat"], obs_dict["global_feat"], sat, oat,
    )
    e_heads = trainer.execution_net(
        obs_dict["self_feat"], obs_dict["other_feat"], obs_dict["exists_mask"],
        obs_dict["ball_feat"], obs_dict["global_feat"], d_heads, sat, oat,
    )
    bc_labels_c = canonicalize_bc_labels(bc_labels, x_sign_of(obs_dict["self_feat"]))
    loss, _bkdn = bc_loss_from_tensor(
        bc_labels_c, d_heads, e_heads,
        direction_loss_weight=trainer._bc_dir_loss_w,
        direction_loss_mode=trainer._bc_dir_loss_mode,
        region_loss_weight=trainer._bc_region_loss_w,
        pos_weight_kick=trainer._bc_pos_weight_kick,
        pos_weight_tackle_attempt=trainer._bc_pos_weight_tackle_attempt,
        dec_weight=trainer._bc_dec_weight,
        exec_weight=trainer._bc_exec_weight,
        dec_label_smoothing=trainer._bc_dec_label_smoothing,
        exec_label_smoothing=trainer._bc_exec_label_smoothing,
        return_breakdown=True,
    )
    loss = loss + direction_magnitude_reg(e_heads, trainer._bc_dir_mag_reg_coef)
    return loss


def _raw_grad_norm_for_rows(trainer, ds, idx, device, all_params) -> float:
    """Forward+backward on exactly the rows in `idx` (a sorted row-index
    array), return the combined decision_net+execution_net raw grad norm --
    measurement only (max_norm=inf), grads left populated for the caller."""
    obs_dict, bc_labels = next(iter(ds.iterate_minibatches(
        len(idx), shuffle=False, device=device, indices_override=idx,
    )))
    loss = _forward_loss(trainer, obs_dict, bc_labels)
    for _p in all_params:
        _p.grad = None
    loss.backward()
    return torch.nn.utils.clip_grad_norm_(all_params, float("inf")).item()


def _bisect_to_culprit_rows(trainer, ds, row_slice: np.ndarray, device, all_params, min_size: int = 4) -> np.ndarray:
    """Recursively halve `row_slice` (the full spiking minibatch's row
    indices), keeping whichever half reproduces the larger raw grad norm on
    its own, until at most `min_size` rows remain.

    A minibatch is thousands of rows spanning 100+ recorded episodes --
    listing all of them (as the coarse per-minibatch report does) doesn't
    localize anything. This greedy halving assumes the single dominant
    contributor lives entirely within one half at each step (true whenever
    one outlier row's gradient dominates the batch sum by a wide margin --
    confirmed the normal case here: per_entity_mlp.0's top column is
    routinely 10-20x the next-largest column, see the per-column breakdown).
    It is NOT exhaustive (could in principle miss two moderate,
    similar-magnitude contributors split across the two halves), but is
    cheap (O(log(batch_size)) forward/backward passes, only run on the
    handful of minibatches that actually spike) and reliably finds the
    single-outlier case this diagnostic exists for.
    """
    current = row_slice
    while len(current) > min_size:
        mid = len(current) // 2
        left, right = current[:mid], current[mid:]
        left_norm = _raw_grad_norm_for_rows(trainer, ds, left, device, all_params)
        right_norm = _raw_grad_norm_for_rows(trainer, ds, right, device, all_params)
        current = left if left_norm >= right_norm else right
    return current


def _physics_encoder_segments(encoder) -> list[tuple[str, int]]:
    """[(name, width), ...] covering a PlayerPhysicsFeatureBlock's full
    output_dim, in the exact order PlayerPhysicsFeatureBlock.forward()
    torch.cats them: the raw latent, then each frozen aux head (declared
    order), then the frozen linear decoder's unpadded tail if the checkpoint
    has one. Lets a raw column index within that concatenated output be
    mapped back to a human-meaningful name (e.g. "crossing_head[2]") instead
    of an opaque col_N."""
    latent_dim = encoder.output_dim - sum(h.out_features for h in encoder.aux_heads.values())
    if encoder.linear_decoder is not None:
        latent_dim -= encoder.linear_decoder.unpadded_output_dim
    segments = [("latent", latent_dim)]
    for name, head in encoder.aux_heads.items():
        segments.append((name, head.out_features))
    if encoder.linear_decoder is not None:
        segments.append(("linear_decoder", encoder.linear_decoder.unpadded_output_dim))
    return segments


def _describe_physics_encoder_output(encoder, feat_row: np.ndarray, global_feat_row: np.ndarray, device, top_k: int = 5) -> str:
    """Run the frozen player-physics encoder on ONE row's raw features and
    report its top-|value| output dims, named via _physics_encoder_segments.

    Exists because this encoder's output is concatenated onto the raw named
    PlayerFeatures before ever reaching per_entity_mlp (see decision_network.
    py's self_feat_aug), so a per_entity_mlp weight-gradient column beyond
    index PLAYER_FEATURE_DIM (39) is NOT one of the human-named fields
    _describe_weight_grad_columns already prints -- it's one of THESE,
    otherwise invisible. The encoder runs under torch.no_grad() internally
    (frozen, see PlayerPhysicsFeatureBlock.forward()) so this is a plain
    inference call, not a backward pass.
    """
    with torch.no_grad():
        feat_t = torch.from_numpy(feat_row).float().unsqueeze(0).to(device)
        global_t = torch.from_numpy(global_feat_row).float().unsqueeze(0).to(device)
        full = encoder(feat_t, global_t)[0]
    labels = []
    offset = 0
    for name, width in _physics_encoder_segments(encoder):
        for i in range(width):
            labels.append(f"{name}[{i}]" if name != "latent" else f"latent[{i}]")
        offset += width
    pairs = sorted(zip(labels, full.tolist()), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    return "  ".join(f"{k}={v:.3f}" for k, v in pairs)


def _find_latest_checkpoint(checkpoints_dir: str = "checkpoints") -> Path:
    """Newest checkpoint_pretrained.pt/checkpoint_final.pt/latest.pt under
    checkpoints/*/, by mtime -- "my latest network" with no path typing
    required. Pass --checkpoint explicitly to pick a specific run instead."""
    candidates = [
        p for pattern in ("*/checkpoint_pretrained.pt", "*/checkpoint_final.pt", "*/latest.pt")
        for p in Path(checkpoints_dir).glob(pattern)
    ]
    if not candidates:
        raise SystemExit(
            f"No checkpoint files found under {checkpoints_dir}/*/ -- pass --checkpoint explicitly."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demonstrations", type=str, default="demonstrations/phase1_rules_immobile",
                        help="Directory of recorded demonstration .npz files. Default: demonstrations/phase1_rules_immobile.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint to load (decision_net/execution_net weights). Default: newest "
                             "checkpoint_pretrained.pt/checkpoint_final.pt/latest.pt under checkpoints/*/.")
    parser.add_argument("--epochs", type=int, default=1, help="Passes over the full dataset. Default: 1.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Default: bc.bc_pretrain_batch_size from ai_config.json (same size real BC training uses).")
    parser.add_argument("--max-minibatches", type=int, default=None,
                        help="Stop after this many minibatches total (across all epochs), regardless of --epochs. "
                             "Default: no cap (run every minibatch every epoch).")
    parser.add_argument("--spike-k", type=float, default=None,
                        help="Override bc.grad_norm_spike_k for this run only (lower = more sensitive, fires on "
                             "smaller outliers, e.g. try 2.0-2.5 if the live default of 4.0 finds nothing here). "
                             "Default: whatever bc.grad_norm_spike_k resolves to in config.")
    parser.add_argument("--device", type=str, default=None, help="PyTorch device. Default: auto-detect.")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed. Default: 0.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device) if args.device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    log.info(f"Device: {device}")

    from footballcoach.ai.bc.dataset import DemonstrationDataset
    from footballcoach.ai.config import load_ai_config
    from footballcoach.ai.obs.schema import PlayerFeatures
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer, _measure_exec_head_grad_norms_and_maybe_log_spike

    checkpoint_path = Path(args.checkpoint) if args.checkpoint is not None else _find_latest_checkpoint()
    log.info(f"Checkpoint: {checkpoint_path}")

    cfg = load_ai_config()
    batch_size = args.batch_size if args.batch_size is not None else int(cfg["bc"]["bc_pretrain_batch_size"])

    trainer = PPOTrainer.from_config(device=device, inference_only=True)
    trainer.load_checkpoint(checkpoint_path)
    if args.spike_k is not None:
        trainer.grad_norm_spike_k = args.spike_k
        log.info(f"--spike-k: overriding grad_norm_spike_k to {args.spike_k}")
    else:
        log.info(f"grad_norm_spike_k={trainer.grad_norm_spike_k} (from config)")

    ds = DemonstrationDataset.from_directory(args.demonstrations)
    valid_idx = ds.valid_indices()
    n_minibatches_per_epoch = (len(valid_idx) + batch_size - 1) // batch_size
    log.info(
        f"Dataset: {len(valid_idx)} valid rows, batch_size={batch_size} "
        f"-> {n_minibatches_per_epoch} minibatch(es)/epoch"
    )

    # pos_weight_*: same auto-compute-from-dataset-if-unset-in-config logic
    # pretrain_combined() runs at the top of real Phase 1 (ppo_trainer.py) --
    # reused here so the loss this script computes matches production's
    # exactly rather than silently falling back to pos_weight=1.0 for both.
    if trainer._bc_pos_weight_kick_cfg is None or trainer._bc_pos_weight_tackle_attempt_cfg is None:
        _auto_weights = ds.compute_pos_weights(max_weight=trainer._bc_pos_weight_max)
        if trainer._bc_pos_weight_kick_cfg is None:
            trainer._bc_pos_weight_kick = _auto_weights["kick"]
        if trainer._bc_pos_weight_tackle_attempt_cfg is None:
            trainer._bc_pos_weight_tackle_attempt = _auto_weights["tackle_attempt"]
        log.info(
            f"BC pos_weight (auto-computed from dataset): "
            f"kick={trainer._bc_pos_weight_kick:.2f}  tackle_attempt={trainer._bc_pos_weight_tackle_attempt:.2f}"
        )

    # train() (not eval()): matches the exact mode real BC/DAgger training
    # runs in (dagger.py's train_bc_epochs, ppo_trainer.py's Phase 1) -- if
    # any module here has train/eval-sensitive behaviour, we want the same
    # one that actually produces the spikes seen live.
    trainer.decision_net.train()
    trainer.execution_net.train()
    all_params = list(trainer.decision_net.parameters()) + list(trainer.execution_net.parameters())

    n_spikes = 0
    n_mb_total = 0
    losses: list[float] = []
    done = False
    for epoch in range(1, args.epochs + 1):
        if done:
            break
        # shuffle=False + indices_override=valid_idx: minibatch mb_i is then
        # GUARANTEED to be exactly valid_idx[mb_i*batch_size : (mb_i+1)*batch_size]
        # (iterate_minibatches only shuffles BLOCK order, never row order
        # within a block, and here there's nothing to shuffle since block
        # order is already left as-is) -- lets this script recover which
        # dataset rows produced a given minibatch without dataset.py needing
        # to expose row indices itself.
        for mb_i, (obs_dict, bc_labels) in enumerate(ds.iterate_minibatches(
            batch_size, shuffle=False, device=device, indices_override=valid_idx,
        )):
            row_slice = valid_idx[mb_i * batch_size: mb_i * batch_size + batch_size]

            loss = _forward_loss(trainer, obs_dict, bc_labels)

            for _p in all_params:
                _p.grad = None
            loss.backward()

            # Snapshot the running history BEFORE calling the shared spike
            # helper (which both fires the [grad norm SPIKE] warning AND
            # appends this minibatch's raw norm to the same history) -- so
            # this script can independently recover, from the RETURNED raw
            # norm, whether THIS minibatch was the one that just fired,
            # without needing the helper to return that as a bool itself.
            _hist_snapshot = list(trainer._grad_norm_history)
            gn_vals = _measure_exec_head_grad_norms_and_maybe_log_spike(trainer, "debug", epoch, mb_i)
            raw = gn_vals["raw"]
            spiked = False
            if trainer.grad_norm_spike_k is not None and len(_hist_snapshot) >= 20:
                _m, _s = float(np.mean(_hist_snapshot)), float(np.std(_hist_snapshot))
                spiked = raw > _m + trainer.grad_norm_spike_k * _s

            losses.append(loss.item())
            n_mb_total += 1

            if spiked:
                n_spikes += 1
                n_episodes_touched = len(ds.episode_row_ranges(row_slice))
                log.warning(
                    f"  [investigate] epoch={epoch} mb={mb_i} minibatch rows=[{row_slice[0]},{row_slice[-1]}] "
                    f"({len(row_slice)} rows, {n_episodes_touched} episodes) loss={loss.item():.4f} -- "
                    f"bisecting to find the culprit row(s)..."
                )
                culprit_rows = _bisect_to_culprit_rows(trainer, ds, row_slice, device, all_params, min_size=4)
                culprit_norm = _raw_grad_norm_for_rows(trainer, ds, culprit_rows, device, all_params)
                log.warning(
                    f"  [investigate] narrowed to {len(culprit_rows)} row(s) "
                    f"(raw grad norm on just these: {culprit_norm:.2f}, vs {raw:.2f} for the full minibatch): "
                    f"{culprit_rows.tolist()}"
                )
                for _row in culprit_rows.tolist():
                    _eps = ds.episode_row_ranges(np.array([_row]))
                    if _eps:
                        _s, _e = _eps[0]
                        _seed = ds.episode_seed(_e)
                        _outcome = ds.classify_outcome(_e)
                        log.warning(
                            f"    row {_row}: episode rows[{_s},{_e}] seed={_seed} outcome={_outcome}"
                            + (f"  -- replay: uv run python -m footballcoach.ai.scripts.replay_episode "
                               f"--seed {_seed} --ui" if _seed is not None else "")
                        )
                    _row_self = ds._self_feat[_row]
                    _row_other = ds._other_feat[_row]
                    _row_global = ds._global_feat[_row]
                    _names = [f.name for f in dataclasses.fields(PlayerFeatures)]
                    if len(_names) == _row_self.shape[-1]:
                        _self_top = sorted(zip(_names, _row_self.tolist()), key=lambda kv: abs(kv[1]), reverse=True)[:6]
                        log.warning(f"      self_feat  top: " + "  ".join(f"{k}={v:.3f}" for k, v in _self_top))
                        for _slot in range(_row_other.shape[0]):
                            _slot_vals = _row_other[_slot]
                            if not np.any(_slot_vals):
                                continue
                            _other_top = sorted(zip(_names, _slot_vals.tolist()), key=lambda kv: abs(kv[1]), reverse=True)[:6]
                            log.warning(f"      other_feat[{_slot}] top: " + "  ".join(f"{k}={v:.3f}" for k, v in _other_top))
                    # Also decode the frozen player-physics-encoder's own
                    # output for this row -- per_entity_mlp's in_features
                    # extends beyond the 39 raw named fields (physics
                    # features are concatenated on, see decision_network.py's
                    # self_feat_aug), so a per-column gradient breakdown that
                    # falls back to "col_65" etc. (see the [grad norm SPIKE]
                    # log above) is actually pointing INTO this block, not
                    # any raw named feature -- this makes it visible by name.
                    if trainer.decision_net.player_physics_encoder is not None:
                        _phys_self = _describe_physics_encoder_output(
                            trainer.decision_net.player_physics_encoder, _row_self, _row_global, device,
                        )
                        log.warning(f"      self_feat physics-encoder output top: {_phys_self}")

            if args.max_minibatches is not None and n_mb_total >= args.max_minibatches:
                done = True
                break

    log.info(
        f"Done: {n_spikes} spike(s) fired over {n_mb_total} minibatch(es) "
        f"({args.epochs} epoch(s) requested, mean loss={float(np.mean(losses)) if losses else float('nan'):.4f})."
    )
    if n_spikes == 0:
        log.info(
            "No spikes fired -- try --spike-k with a smaller value (e.g. 2.0-2.5) to see smaller outliers, "
            "or --epochs > 1 / a larger dataset pass to sample more minibatches."
        )


if __name__ == "__main__":
    main()
