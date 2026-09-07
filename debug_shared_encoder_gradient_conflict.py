"""Are decision_net's and execution_net's gradients on the SHARED
entity_encoder (share_entity_encoder=True) aligned or conflicting?

Context: decision_net and execution_net share ONE EntityEncoder instance
(same per_entity_mlp/attention/ln_query/ln_kv/*_query_proj weight tensors,
not just the same architecture) -- every training step, BOTH networks'
losses contribute gradient onto that same set of parameters. Whether that's
a good idea depends on whether the two objectives actually agree on what a
good per-player embedding looks like (aligned -> free multi-task lunch) or
actively disagree (conflicting -> negative transfer, the shared layer has
to compromise between two pulls). Plain gradient MAGNITUDE (what the
existing [grad norm SPIKE] diagnostic reports) can't tell these apart --
two huge but well-aligned gradients look the same as two huge, opposed
ones by norm alone. Cosine similarity between the two SEPARATE
contributions (before they get summed into one .grad) directly answers
this: +1 = perfectly aligned, 0 = unrelated, -1 = directly fighting.

Method: one forward pass, two INDEPENDENT loss scalars (a decision-heads-
only BCE loss, and an execution-heads-only BCE+direction loss), each
backpropagated separately via torch.autograd.grad(..., retain_graph=True)
-- NOT loss.backward(), which would sum both contributions into one .grad
and destroy exactly the distinction we want to measure. Read-only: no
optimizer, no checkpoint written, single small demo file (a few dozen
episodes), CPU-cheap.

Usage:
    uv run python debug_shared_encoder_gradient_conflict.py
    uv run python debug_shared_encoder_gradient_conflict.py --checkpoint checkpoints/phase1_run45/checkpoint_pretrained.pt
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
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("footballcoach.ai.debug_shared_encoder_gradient_conflict")


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
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Default: newest checkpoint under checkpoints/*/.")
    parser.add_argument("--demo-file", type=str, default=None,
                        help="Single .npz demo file (default: first file under "
                             "demonstrations/phase1_rules_immobile) -- deliberately ONE file "
                             "(a few dozen episodes), not the full multi-GB dataset.")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Rows to use from that file. Default: 4096.")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    log.info(f"Device: {device}")

    from footballcoach.ai.bc.dataset import DemonstrationDataset
    from footballcoach.ai.obs.canonical import canonicalize_bc_labels, x_sign_of
    from footballcoach.ai.ppo.bc import (
        _I_DIR_X, _I_DIR_Y, _I_EXEC_MOVE, _I_GP_EXTRA, _I_HOLD, _I_KICK_THIS_TICK,
        _I_MARK, _I_MOVE, _I_PASS, _I_SHOOT, _I_SPRINT, _I_TACKLE, _I_TACKLE_ATTEMPT, _I_VALID,
    )
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer, _ai_types

    checkpoint_path = Path(args.checkpoint) if args.checkpoint is not None else _find_latest_checkpoint()
    log.info(f"Checkpoint: {checkpoint_path}")

    demo_file = args.demo_file or sorted(glob.glob("demonstrations/phase1_rules_immobile/*.npz"))[0]
    log.info(f"Demo file: {demo_file}")

    trainer = PPOTrainer.from_config(device=device, inference_only=True)
    trainer.load_checkpoint(checkpoint_path)
    trainer.decision_net.train()
    trainer.execution_net.train()

    ds = DemonstrationDataset.from_files([str(demo_file)])
    valid_idx = ds.valid_indices()
    n_eps = ds.n_episodes(valid_idx)
    idx = valid_idx[: args.batch_size]
    log.info(f"{len(valid_idx)} valid rows / {n_eps} episode(s) in this file -- using {len(idx)} row(s)")

    obs_dict, bc_labels = next(iter(ds.iterate_minibatches(
        len(idx), shuffle=False, device=device, indices_override=idx,
    )))
    sat, oat = _ai_types(obs_dict)
    d_heads = trainer.decision_net(
        obs_dict["self_feat"], obs_dict["other_feat"], obs_dict["exists_mask"],
        obs_dict["ball_feat"], obs_dict["global_feat"], sat, oat,
    )
    e_heads = trainer.execution_net(
        obs_dict["self_feat"], obs_dict["other_feat"], obs_dict["exists_mask"],
        obs_dict["ball_feat"], obs_dict["global_feat"], d_heads, sat, oat,
    )
    labels = canonicalize_bc_labels(bc_labels, x_sign_of(obs_dict["self_feat"]))
    valid = labels[:, _I_VALID] > 0.5
    if not valid.any():
        raise SystemExit("No valid BC rows in this batch -- try a different --demo-file or larger --batch-size.")

    def _bce(logit: torch.Tensor, col: int) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logit.squeeze(-1), labels[:, col], reduction="none")

    # Decision-only loss: the 7 decision Bernoulli heads. Depends ONLY on
    # decision_net's own forward pass -- a clean, single-source signal.
    dec_loss_per_row = (
        _bce(d_heads.shoot_logit, _I_SHOOT) + _bce(d_heads.pass_logit, _I_PASS)
        + _bce(d_heads.move_logit, _I_MOVE) + _bce(d_heads.tackle_logit, _I_TACKLE)
        + _bce(d_heads.get_possession_raw, _I_GP_EXTRA) + _bce(d_heads.mark_logit, _I_MARK)
        + _bce(d_heads.hold_position_logit, _I_HOLD)
    )
    dec_loss = dec_loss_per_row[valid].mean()

    # Execution-only loss: execution Bernoulli heads + move_direction cosine
    # loss. Depends on execution_net's own forward pass -- which itself
    # depends on decision_net's output (d_heads, passed in UNDETACHED), so
    # this genuinely captures "execution's total pull on the shared encoder"
    # via every path that exists right now, not just its own separate call.
    exec_bce_per_row = (
        _bce(e_heads.sprint_logit, _I_SPRINT) + _bce(e_heads.exec_move_logit, _I_EXEC_MOVE)
        + _bce(e_heads.tackle_attempt_logit, _I_TACKLE_ATTEMPT) + _bce(e_heads.kick_logit, _I_KICK_THIS_TICK)
    )
    has_dir = (labels[:, _I_DIR_X].abs() + labels[:, _I_DIR_Y].abs()) > 1e-6
    target_dir = labels[:, _I_DIR_X:_I_DIR_Y + 1]
    pred_dir = e_heads.move_direction
    pred_norm = pred_dir / (pred_dir.norm(dim=-1, keepdim=True) + 1e-6)
    dir_cos_loss = torch.where(has_dir, 1.0 - (pred_norm * target_dir).sum(-1), torch.zeros_like(exec_bce_per_row))
    exec_loss_per_row = exec_bce_per_row + dir_cos_loss
    exec_loss = exec_loss_per_row[valid].mean()

    log.info(f"dec_loss={dec_loss.item():.4f}  exec_loss={exec_loss.item():.4f}  (n={int(valid.sum())} valid rows)")

    ee = trainer.decision_net.entity_encoder
    if ee is not trainer.execution_net.entity_encoder:
        raise SystemExit(
            "decision_net.entity_encoder is NOT the same object as execution_net.entity_encoder "
            "(share_entity_encoder=False in this config?) -- nothing shared to measure."
        )

    submodules = {"per_entity_mlp": ee.per_entity_mlp, "attention": ee.attention,
                  "ln_query": ee.ln_query, "ln_kv": ee.ln_kv}
    if ee.ball_query_proj is not None:
        submodules["ball_query_proj"] = ee.ball_query_proj
    if ee.global_query_proj is not None:
        submodules["global_query_proj"] = ee.global_query_proj
    if ee.inter_player_attn is not None:
        submodules["inter_player_attn"] = ee.inter_player_attn

    log.info("Shared submodule gradient alignment (decision-only loss vs execution-only loss):")
    for name, module in submodules.items():
        params = [p for p in module.parameters() if p.requires_grad]
        if not params:
            continue
        grad_dec = torch.autograd.grad(dec_loss, params, retain_graph=True, allow_unused=True)
        grad_exec = torch.autograd.grad(exec_loss, params, retain_graph=True, allow_unused=True)
        flat_dec = torch.cat([g.flatten() for g in grad_dec if g is not None])
        flat_exec = torch.cat([g.flatten() for g in grad_exec if g is not None])
        if flat_dec.numel() == 0 or flat_exec.numel() == 0:
            log.info(f"  {name}: no gradient from one side (unused this pass) -- skipping")
            continue
        cos = F.cosine_similarity(flat_dec.unsqueeze(0), flat_exec.unsqueeze(0)).item()
        verdict = "ALIGNED" if cos > 0.3 else "CONFLICTING" if cos < -0.3 else "~orthogonal/unrelated"
        log.info(
            f"  {name:<18} cos_sim={cos:+.3f}  ({verdict})   "
            f"|dec_grad|={flat_dec.norm().item():.4f}  |exec_grad|={flat_exec.norm().item():.4f}  "
            f"(n_params={flat_dec.numel()})"
        )


if __name__ == "__main__":
    main()
