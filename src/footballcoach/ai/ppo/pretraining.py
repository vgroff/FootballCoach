"""PPOTrainer's BC + value pretraining (pretrain_combined, pretrain_value) and the value-pretrain rollout collection and worker pool."""
from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from footballcoach.ai.models.execution_network import ExecutionNetwork
from footballcoach.ai.obs.augment import N_FLIP_VARIANTS, augment_batch, augment_obs_bc
from footballcoach.ai.progress import ProgressReporter
from footballcoach.ai.obs.canonical import CanonicalNetworkWrapper, canonicalize_bc_labels, x_sign_of
from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer
from footballcoach.ai.ppo.trainer_common import REWARD_COMP_LABELS, _PHASE1_OUTCOME_KEYS, _PHASE1_OUTCOME_LEGEND, _load_state_dict_tolerant, log, outcome_breakdown
from footballcoach.ai.ppo.trainer_common import EXEC_HEAD_MODULES, _action_to_numpy, _ai_types, _detach_decision_heads, _episode_abs_adv_means, _maybe_log_grad_norm_spike, _merge_worker_batches, _split_batch_releasing, format_outcome_rmse_breakdown, value_mse_by_outcome


def _measure_exec_head_grad_norms_and_maybe_log_spike(
    trainer: "PPOTrainer", label: str, epoch_i: int, mb_i: int,
) -> dict[str, float]:
    """Measurement-only (max_norm=inf, never rescales) per-execution-head
    grad norm snapshot for BC/DAgger's minibatch loops (pretrain_combined()'s
    Phase 1, dagger.py's train_bc_epochs()) -- mirrors PPO's own per-head
    grad-norm diagnostic (_ppo_update's [exec head grad norm]) so BC/DAgger
    rounds get the same per-head visibility PPO already has, instead of only
    a single aggregate clip norm. Also runs the grad-norm-spike diagnostic
    (see _maybe_log_grad_norm_spike) against the same combined
    decision_net+execution_net raw norm PPO's own spike check uses, so an
    outlier minibatch during BC/DAgger gets the same per-parameter-tensor
    breakdown PPO gets.

    Must be called AFTER loss.backward() and BEFORE the caller's own real
    clip_grad_norm_()/optimizer.step() (grads must still reflect the
    ORIGINAL, un-clipped magnitudes for this measurement to mean anything --
    this function never itself clips; the caller's real cap is unaffected).

    Returns {head_name: grad_norm}, plus "raw" for the combined norm across
    both networks. `label` is a short tag (e.g. "BC", "DAgger") passed
    through to _maybe_log_grad_norm_spike.
    """
    _gn_tensors: dict[str, torch.Tensor] = {}
    for _head_name, _attr in EXEC_HEAD_MODULES:
        _head_params = list(getattr(trainer.execution_net, _attr).parameters())
        if _head_params:
            _gn_tensors[_head_name] = torch.nn.utils.clip_grad_norm_(_head_params, float("inf"))
    _gn_tensors["raw"] = torch.nn.utils.clip_grad_norm_(
        list(trainer.decision_net.parameters()) + list(trainer.execution_net.parameters()),
        float("inf"),
    )
    _gn_names = list(_gn_tensors.keys())
    _gn_vals = dict(zip(
        _gn_names,
        torch.stack([_gn_tensors[k].to(trainer.device) for k in _gn_names]).tolist(),
    ))

    _maybe_log_grad_norm_spike(trainer, label, epoch_i, mb_i, _gn_vals["raw"])
    return _gn_vals

def log_episode_reward_stats_by_outcome(
    episode_returns: list[float], episode_outcome_raw: list[str], log_prefix: str = "",
) -> None:
    """Log mean/std/min/max of per-episode TOTAL reward (undiscounted sum
    across the whole episode -- i.e. Monte Carlo return with gamma=1, not
    GAE/bootstrapped), grouped by outcome (mapped through
    ``_PHASE1_OUTCOME_KEYS``' short labels; "unknown" for anything else or
    missing). ``episode_returns``/``episode_outcome_raw`` must be the same
    length and in the same per-episode order -- index i of one must
    correspond to index i of the other. No-op if empty."""
    if not episode_returns:
        return
    _label_map = dict(_PHASE1_OUTCOME_KEYS)
    returns_arr = np.array(episode_returns, dtype=np.float64)
    labels = np.array([_label_map.get(o, "unknown") for o in episode_outcome_raw], dtype=object)
    log.info(f"  {log_prefix}episode total reward (gamma=1, {len(returns_arr)} episode(s)):")
    for name in sorted(set(labels)):
        vals = returns_arr[labels == name]
        log.info(
            f"    {name:<6} n={len(vals):>5}  mean={vals.mean():+.3f}  "
            f"std={vals.std():.3f}  min={vals.min():+.3f}  max={vals.max():+.3f}"
        )

def _binary_confusion_counts(
    pred_logit: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, threshold: float = 0.5,
) -> tuple[float, float, float]:
    """Return (tp, fp, fn) counts for a Bernoulli head over valid rows.

    ``pred_logit``/``target`` are raw (pre-sigmoid) logits and 0/1 float
    labels respectively, shape (N,) or (N, 1) (squeezed internally).
    ``valid_mask`` selects which rows to include (e.g. BC's _I_VALID mask).
    Used to accumulate precision/recall/F1 diagnostics for rare-positive
    heads (kick_this_tick, tackle_attempt), where a flat/low mean predicted
    probability alone can't distinguish over-firing (low precision) from
    under-firing (low recall) -- see ai_trainer_knowledge.md BC diagnostics.
    """
    with torch.no_grad():
        p = torch.sigmoid(pred_logit.squeeze(-1))[valid_mask]
        t = target[valid_mask] > 0.5
        pred_pos = p > threshold
        tp = float((pred_pos & t).sum())
        fp = float((pred_pos & ~t).sum())
        fn = float((~pred_pos & t).sum())
    return tp, fp, fn

def _precision_recall_f1(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    """Precision/recall/F1 from accumulated (tp, fp, fn) counts.

    NaN-safe: returns float('nan') for any ratio with a zero denominator (no
    positive predictions this epoch / no positive labels seen this epoch)
    rather than raising or silently returning 0.0, so it's visually distinct
    from a genuine 0.0 score in logs.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    f1 = (
        2 * precision * recall / (precision + recall)
        if (tp + fp) > 0 and (tp + fn) > 0 and (precision + recall) > 0
        else float('nan')
    )
    return precision, recall, f1

@dataclass
class _ValuePretrainWorkers:
    """A live pool of value-pretrain rollout worker processes -- either
    plain one-env-per-process ``rollout_worker.py`` workers, or (when
    ``ppo.value_pretrain_batched_rollout`` is on) ``batched_rollout_worker.py``
    workers each owning ``envs_per_process`` envs. Built by
    ``PPOTrainer._spawn_value_pretrain_workers``; ``progress_value`` is the
    shared ``ctx.Value`` step counter the workers add to (always present).
    """
    handles: list
    batched: bool
    progress_value: object
    n_processes: int
    envs_per_process: int

def _finalize_value_pretrain_result(
    r: dict, gamma: float, lam: float, use_gae: bool, allow_empty: bool = False,
    with_mc_returns: bool = False,
) -> tuple[Optional[dict], int]:
    """Turn ONE worker/env result dict (``{"buffer", "last_value"?, "stats"}``)
    from the value-pretrain rollout into an ``as_tensors()`` batch dict with
    its ``returns`` filled in. Returns ``(batch_dict, n_rows_dropped)``.

    ``allow_empty`` (MC mode only): when the buffer holds NO completed
    episode at all -- with per-env whole-episode chunking that is common (an
    env's final flush is often just its unfinished tail) -- MC returns for
    those rows would be truncated garbage and ``truncate_to_last_episode_end``
    (which never empties a buffer) would leave them in place. With
    ``allow_empty=True`` the whole buffer is dropped instead and
    ``(None, n_rows)`` is returned; callers must skip a None batch. Default
    False keeps the old keep-everything behaviour for the callers that can't
    tolerate an empty result (single-process collection, plain workers whose
    single per-worker buffer always spans many episodes).

    ``with_mc_returns``: additionally attach an ``"mc_returns"`` tensor (pure
    Monte-Carlo returns, NaN on each track's unfinished tail -- see
    ``_mc_returns_nan_tail``) alongside whatever ``"returns"`` the mode above
    produced. Used by ``ppg_value_refit``'s compare-to-original diagnostic.

    Never concatenate raw transitions across result boundaries before this
    (same per-worker discipline as ``_merge_worker_batches``'s docstring).

    - ``use_gae=False`` (``pretrain_value``): drop any trailing incomplete
      episode (``truncate_to_last_episode_end``), then pure Monte Carlo
      discounted returns -- unchanged from the original behaviour.
    - ``use_gae=True`` AND the result carries a ``"last_value"`` (both worker
      kinds compute one via ``PPOTrainer._bootstrap_last_values``): GAE with
      that REAL bootstrap value and NO truncation, so a trailing partial
      episode's data is kept (correctly bootstrapped) instead of thrown away.
    - ``use_gae=True`` but no ``"last_value"`` (the single-process
      collection branch, which builds a bare buffer): truncate, then GAE with
      a literal 0.0 bootstrap -- provably never used, since truncation leaves
      the buffer ending on a ``done=1`` row and GAE's backward recursion
      resets at every ``done=1``.
    """
    buf = r["buffer"]
    if use_gae and r.get("last_value") is not None:
        advantages, returns = buf.compute_gae(gamma, lam, r["last_value"])
        n_dropped = 0
    else:
        if allow_empty and not use_gae and buf.last_complete_episode_end() < 0:
            return None, len(buf)
        n_dropped = buf.truncate_to_last_episode_end()
        if use_gae:
            advantages, returns = buf.compute_gae(gamma, lam, 0.0)
        else:
            advantages = [0.0] * len(buf.rewards)
            returns = buf.compute_mc_returns(gamma)
    tensors = buf.as_tensors(advantages, returns)
    if with_mc_returns:
        tensors["mc_returns"] = _mc_returns_nan_tail(buf, gamma)
    return tensors, n_dropped

def _mc_returns_nan_tail(buf, gamma: float) -> torch.Tensor:
    """Pure Monte-Carlo discounted returns for every row of ``buf``, with NaN
    on each track's unfinished tail (rows after that track's last ``done=1``:
    their true return isn't known yet, so a truncated sum would be garbage).

    Independent of ANY value net -- unlike GAE returns (``A + V_sampler``),
    which contain the sampling network's own predictions and therefore
    always favour whichever network generated the rollout when two value
    nets are scored against them. That independence is why
    ``ppg_value_refit``'s compare-to-original diagnostic scores against
    THESE targets."""
    mc = np.asarray(buf.compute_mc_returns(gamma), dtype=np.float32)
    for _track, idxs in buf._track_index_groups().items():
        last_done_pos = -1
        for pos, i in enumerate(idxs):
            if buf.dones[i] > 0.5:
                last_done_pos = pos
        for i in idxs[last_done_pos + 1:]:
            mc[i] = np.nan
    return torch.from_numpy(mc)


class PretrainingMixin:
    """BC and value pretraining for PPOTrainer. Mixed into PPOTrainer; relies on its attributes."""

    # -----------------------------------------------------------------------
    # Curriculum helpers
    # -----------------------------------------------------------------------

    def _get_value_pretrain_freeze_params(self) -> list:
        """Return params to set requires_grad=False during value warm-up.

        Controlled by ``bc.value_pretrain_frozen_layers`` in ai_config.json:
          -1 / 3 : encoders + full trunk (default — matches prior behaviour)
          2       : encoders + first trunk Linear only
          1       : pre-trunk encoders only
          0       : nothing frozen (gradients flow through trunk freely)
        """
        n = self._value_pretrain_frozen_layers
        if n == 0:
            return []
        params: list[nn.Parameter] = []
        for net in (self.decision_net, self.execution_net):
            pre_trunk: list[nn.Module] = [
                net.entity_encoder, net.self_mlp, net.ball_mlp, net.global_mlp,
            ]
            if hasattr(net, "decision_mlp"):
                pre_trunk.append(net.decision_mlp)
            layer_groups = [
                pre_trunk,
                [net.trunk[0]],
                [net.trunk[2]],
            ]
            count = len(layer_groups) if n == -1 else min(n, len(layer_groups))
            for group in layer_groups[:count]:
                for m in group:
                    params.extend(m.parameters())
        return params

    def _freeze_for_value_only_bc(self) -> None:
        """Freeze decision_net entirely, and execution_net entirely EXCEPT
        its value head(s) -- pair with ``_unfreeze_after_value_only_bc()``.

        Used by ``pretrain_combined()`` when ``bc.bc_train_value_only`` is
        set: Phase 0's forward pass still needs decision_net (the value head
        reads decision-network context) and execution_net's trunk (the value
        head sits on top of it), but with both frozen here no gradient flows
        past the value head's own parameters, so Adam only ever sees nonzero
        grad on the value head. ``decision_net.value_head`` is already
        permanently frozen elsewhere (single value head convention) and
        stays that way regardless -- this does not touch it either way.

        Saves each param's prior ``requires_grad`` onto
        ``self._voe_saved_grad_state`` so ``_unfreeze_after_value_only_bc()``
        can restore it exactly, rather than assuming everything was
        previously trainable.
        """
        self._voe_saved_grad_state = [(p, p.requires_grad) for p in self.decision_net.parameters()]
        self._voe_saved_grad_state += [(p, p.requires_grad) for p in self.execution_net.parameters()]
        self.decision_net.requires_grad_(False)
        self.execution_net.requires_grad_(False)
        self.execution_net.value_head.requires_grad_(True)
        if hasattr(self.execution_net, "value_ai_type_channel"):
            self.execution_net.value_ai_type_channel.requires_grad_(True)

    def _unfreeze_after_value_only_bc(self) -> None:
        """Restore every param's ``requires_grad`` saved by
        ``_freeze_for_value_only_bc()``, so a value-only BC pass never
        leaks a frozen policy into what runs after it (Phase 2/3's rollout
        warm-up, then PPO)."""
        for p, orig in self._voe_saved_grad_state:
            p.requires_grad_(orig)
        self._voe_saved_grad_state = []

    # -----------------------------------------------------------------------
    # Value pre-training
    # -----------------------------------------------------------------------

    def pretrain_combined(
        self,
        env,
        dataset,
        n_epochs: int,
        batch_size: int,
        bc_lr: float,
        value_lr: float,
        rollout_steps: int,
        value_epochs: int = 5,
        repair_lr: Optional[float] = None,
        experiment_separate_value_net: bool = False,
        phase_id: Optional[int] = None,
        checkpoint_dir: Optional[Path] = None,
    ) -> None:
        """Joint BC + value pre-training in a single pass.

        Collects one rollout to get GAE returns for the value loss, then for
        each epoch iterates over the BC dataset in minibatches.  Each iteration
        does two backward passes:

          1. BC loss (all parameters) — teaches the policy trunk to imitate the
             rules-based AI.
          2. Value loss (value heads only, trunk detached) — warm-starts the
             critic to predict actual returns, without corrupting the trunk.

        This replaces the two separate ``BCPretrainer.pretrain`` +
        ``pretrain_value`` calls and avoids the oscillation of online BC
        pre-training.

        Args:
            env: ScenarioEnv
            dataset: DemonstrationDataset
            n_epochs: epochs over the BC dataset
            batch_size: minibatch size for both losses
            bc_lr: learning rate for BC (all params)
            value_lr: learning rate for value heads only
            rollout_steps: steps to collect for value targets (≥ rollout_steps in config)
            checkpoint_dir: If given, ``checkpoint_dir/checkpoint_pretrained.pt``
                is saved (overwritten in place) after EACH phase completes
                (Phase 0, Phase 1, Phase 2/3, Phase 4) instead of only once
                at the very end (the caller's own post-call save, e.g.
                train.py's, still happens too -- this just means a crash or
                interrupt partway through no longer loses everything back to
                the start of pretraining; whatever the last COMPLETED phase
                produced is always on disk). ``None`` (default) = no
                intermediate saves, matching prior behaviour.
        """
        from footballcoach.ai.ppo.bc import (
            bc_loss_from_tensor, compute_bc_loss_floor, compute_bc_loss_floor_components,
            direction_magnitude_reg,
        )
        from footballcoach.ai.bc.dataset import DemonstrationDataset

        # pos_weight_*: auto-compute from this dataset if not overridden in config.
        if self._bc_pos_weight_kick_cfg is None or self._bc_pos_weight_tackle_attempt_cfg is None:
            _auto_weights = dataset.compute_pos_weights(max_weight=self._bc_pos_weight_max)
            if self._bc_pos_weight_kick_cfg is None:
                self._bc_pos_weight_kick = _auto_weights["kick"]
            if self._bc_pos_weight_tackle_attempt_cfg is None:
                self._bc_pos_weight_tackle_attempt = _auto_weights["tackle_attempt"]
            log.info(
                f"BC pos_weight (auto-computed from dataset): "
                f"kick={self._bc_pos_weight_kick:.2f}  "
                f"tackle_attempt={self._bc_pos_weight_tackle_attempt:.2f}"
            )

        log.info(
            f"Combined BC + value pre-training: {n_epochs} epoch(s), "
            f"batch_size={batch_size}, dataset={len(dataset):,} steps, "
            f"rollout_steps={rollout_steps}"
        )

        def _save_pretrain_checkpoint(phase_label: str) -> None:
            """Overwrite checkpoint_dir/checkpoint_pretrained.pt with the
            current weights -- see checkpoint_dir's own docstring above for
            why this runs after every phase instead of only once at the end."""
            if checkpoint_dir is None:
                return
            _path = Path(checkpoint_dir) / "checkpoint_pretrained.pt"
            self._save_checkpoint_to(_path)
            log.info(f"  Pre-trained checkpoint saved after {phase_label}: {_path}")

        bc_opt = torch.optim.Adam(
            list(self.decision_net.parameters()) + list(self.execution_net.parameters()),
            lr=bc_lr, eps=1e-5,
        )
        # NOTE: the value-only optimizer used to be built here, but Phase 2/3's
        # value warm-up now delegates to pretrain_value(), which builds its own
        # internal value_opt over decision_net.value_head + execution_net.value_head
        # (identical param set) — see the pretrain_value() call further below.

        if self._bc_train_value_only:
            log.info(
                "bc_train_value_only=True -- decision_net + execution_net policy "
                "heads frozen for Phase 0; Phase 1's BC epoch loop and the BC "
                "repair epochs are both skipped entirely (nothing to repair -- "
                "the policy never moves during this call). Only the value "
                "head(s) train, via Phase 0 alone. bc_opt above is still built "
                "over the full param set but will only ever see gradient on "
                "the value head once frozen."
            )
            self._freeze_for_value_only_bc()

        # --- Phase 0: decision-network-only warm-up on demo data (before any BC epochs) ---
        # Combined decision-heads-only BC loss + value MSE loss, ONE backward pass.
        # Optimizer covers ALL decision_net parameters (encoders + trunk;
        # decision_net.value_head itself stays frozen — single value head
        # convention, see __init__) PLUS the one live critic's value_head
        # (execution_net.value_head normally, or self.value_net when
        # separate_value_net is enabled — see ai/knowledge.md "Phase 0"
        # note). The rest of execution_net (encoders/trunk/action heads) is
        # NOT trained here; it gets its BC training in Phase 1 below.
        # execution_net still needs a forward pass every minibatch (to
        # produce e_heads.value from d_heads) even when separate_value_net
        # is on and its own value_head is frozen/unused — the actual critic
        # forward goes through self._value_heads() instead. Uses stored
        # rewards/dones so no env interaction is needed. Skipped if the
        # dataset has no reward data or demo_value_pretrain_epochs=0.
        # decision_net's BC warm-up runs regardless of separate_value_net —
        # only WHICH network's value_head is trained alongside it changes;
        # self.value_net gets its own Adam param group (self.value_net_
        # optimizer, built in __init__) instead of the ad-hoc demo_opt used
        # for execution_net.value_head's params in the non-separate case.
        _demo_epochs = self._demo_value_pretrain_epochs
        if _demo_epochs > 0 and dataset.has_rewards:
            if self.separate_value_net:
                demo_opt = torch.optim.Adam(
                    list(self.decision_net.parameters()),
                    lr=self._demo_value_pretrain_lr, eps=1e-5,
                )
                # Fresh optimizer, NOT self.value_net_optimizer -- that one is
                # built at ppo.value_learning_rate (5e-5 by default), tuned
                # for gentle per-rollout updates against a slowly-shifting
                # on-policy return distribution during real PPO, not a bulk
                # offline regression pass over the full demo dataset. Reusing
                # it here made Phase 0 (and Phase 1's later separate-value
                # branch, before this fix) converge far slower than
                # demo_value_pretrain_lr actually allows, and dragged
                # whatever Adam momentum state this builds up straight into
                # real PPO training afterward. demo_value_pretrain_lr is the
                # same rate Phase 1's own separate-value branch now uses, for
                # the identical underlying task (demo-return regression).
                _value_opt = torch.optim.Adam(
                    self.value_net.parameters(), lr=self._demo_value_pretrain_lr, eps=1e-5,
                )
                _value_clip_params = list(self.value_net.parameters())
            else:
                demo_opt = torch.optim.Adam(
                    list(self.decision_net.parameters())
                    + list(self.execution_net.value_head.parameters()),
                    lr=self._demo_value_pretrain_lr, eps=1e-5,
                )
                _value_opt = None
                _value_clip_params = list(self.execution_net.value_head.parameters())
            demo_returns = dataset.compute_returns(gamma=self._demo_value_pretrain_gamma)
            ret_t_all = torch.from_numpy(demo_returns).to(self.device)
            ret_std = ret_t_all.std().clamp(min=1.0)

            _p0_train_idx, _p0_val_idx = dataset.split_train_val_indices(val_frac=0.15, valid_only=True)
            _p0_early_stop_enabled = self._p0_early_stop_patience > 0 and len(_p0_val_idx) > 0
            log.info(
                f"Phase 0 — decision-net warm-up (BC + "
                f"{'self.value_net' if self.separate_value_net else 'execution_net.value_head'} "
                f"MSE; single value head convention): {_demo_epochs} epoch(s), "
                f"gamma={self._demo_value_pretrain_gamma}, "
                f"returns mean={ret_t_all.mean():.2f}  std={ret_std:.2f}  "
                f"lr={self._demo_value_pretrain_lr}  "
                f"phase0_value_coef={self._phase0_value_coef}  "
                f"split: {len(_p0_train_idx):,} train / {len(_p0_val_idx):,} val rows"
            )

            # Requires ground-truth episode outcomes (see
            # DemonstrationDataset.has_episode_outcomes) -- falls back to an
            # all-"n/a" lookup for older recordings rather than raising, so
            # Phase 0 warm-up still runs without the outcome breakdown.
            if dataset.has_episode_outcomes:
                _p0_outcome_by_row = dataset.outcome_by_row()
            else:
                log.warning("Dataset has no ground-truth episode outcomes -- "
                            "Phase 0 val-loss-by-outcome breakdown will be skipped.")
                _p0_outcome_by_row = np.full(len(dataset), "n/a", dtype=object)

            def _eval_p0_val_loss() -> tuple[float, float, float, dict[str, tuple[float, int, float, float]]]:
                """Combined dec_bc + value MSE on the held-out val rows (no grad).
                Returns (combined, bc_adj, val_mse, val_mse_by_outcome) where
                bc_adj = bc - floor and val_mse_by_outcome is the per-outcome
                (raw value MSE, n_rows, gt_mean, gt_std) breakdown (see
                value_mse_by_outcome())."""
                self.decision_net.eval()
                _v_losses: list[float] = []
                _v_bc_losses: list[float] = []
                _v_mse_losses: list[float] = []
                _v_floors: list[float] = []
                _outc_sq_err: dict[str, float] = {}
                _outc_n: dict[str, int] = {}
                _outc_gt_sum: dict[str, float] = {}
                _outc_gt_sqsum: dict[str, float] = {}
                _pos = 0
                with torch.no_grad():
                    for _obs_v, _lbl_v, _ret_v in dataset.iterate_minibatches(
                        batch_size=batch_size, shuffle=False, device=self.device,
                        indices_override=_p0_val_idx, returns=demo_returns,
                    ):
                        # iterate_minibatches(shuffle=False) with indices_override yields
                        # contiguous batch_size-sized slices of _p0_val_idx in order, so
                        # this chunk (for the outcome lookup only) is reconstructable here
                        # without the generator itself yielding row indices.
                        _chunk = _p0_val_idx[_pos:_pos + len(_ret_v)]
                        _pos += len(_ret_v)
                        _sat_v, _oat_v = _ai_types(_obs_v)
                        _d_v = self.decision_net(
                            _obs_v["self_feat"], _obs_v["other_feat"],
                            _obs_v["exists_mask"], _obs_v["ball_feat"], _obs_v["global_feat"],
                            _sat_v, _oat_v,
                        )
                        _lbl_v_c = canonicalize_bc_labels(_lbl_v, x_sign_of(_obs_v["self_feat"]))
                        _bc_v, _ = bc_loss_from_tensor(
                            _lbl_v_c, _d_v, exec_heads=None,
                            direction_loss_weight=self._bc_dir_loss_w,
                            direction_loss_mode=self._bc_dir_loss_mode,
                                region_loss_weight=self._bc_region_loss_w,
                            dec_weight=self._bc_dec_weight,
                            dec_label_smoothing=self._bc_dec_label_smoothing,
                            return_breakdown=True,
                        )
                        _floor_v = compute_bc_loss_floor(
                            _lbl_v,
                            dec_weight=self._bc_dec_weight,
                            dec_label_smoothing=self._bc_dec_label_smoothing,
                            has_exec=False,
                        )
                        _value_v = self._value_heads(
                            _obs_v["self_feat"], _obs_v["other_feat"],
                            _obs_v["exists_mask"], _obs_v["ball_feat"], _obs_v["global_feat"],
                            _d_v, _sat_v, _oat_v,
                        )
                        _pred_v = _value_v.squeeze(-1)
                        _mse_v = F.mse_loss(_pred_v, _ret_v) / (ret_std ** 2)
                        _v_losses.append((_bc_v + self._phase0_value_coef * _mse_v).item())
                        _v_bc_losses.append(_bc_v.item())
                        _v_mse_losses.append(_mse_v.item())
                        _v_floors.append(_floor_v)
                        for _name, (_mse, _n, _gt_mean, _gt_mean_sq) in value_mse_by_outcome(
                            _pred_v, _ret_v, list(_p0_outcome_by_row[_chunk])
                        ).items():
                            _outc_sq_err[_name] = _outc_sq_err.get(_name, 0.0) + _mse * _n
                            _outc_n[_name] = _outc_n.get(_name, 0) + _n
                            _outc_gt_sum[_name] = _outc_gt_sum.get(_name, 0.0) + _gt_mean * _n
                            _outc_gt_sqsum[_name] = _outc_gt_sqsum.get(_name, 0.0) + _gt_mean_sq * _n
                self.decision_net.train()
                _combined = float(np.mean(_v_losses)) if _v_losses else float("nan")
                _bc_mean = float(np.mean(_v_bc_losses)) if _v_bc_losses else float("nan")
                _floor_mean = float(np.mean(_v_floors)) if _v_floors else 0.0
                _mse_mean = float(np.mean(_v_mse_losses)) if _v_mse_losses else float("nan")
                _by_outcome = {
                    name: (
                        _outc_sq_err[name] / max(_outc_n[name], 1),
                        _outc_n[name],
                        _outc_gt_sum[name] / max(_outc_n[name], 1),
                        _outc_gt_sqsum[name] / max(_outc_n[name], 1),
                    )
                    for name in _outc_sq_err
                }
                return _combined, _bc_mean - _floor_mean, _mse_mean, _by_outcome

            _p0_best_val_loss = float("inf")
            _p0_best_state: Optional[dict] = None
            _p0_patience_ctr = 0
            _p0_stopped_early = False

            for epoch in range(_demo_epochs):
                epoch_losses: list[float] = []
                epoch_bc_losses: list[float] = []
                epoch_val_losses: list[float] = []
                epoch_floors: list[float] = []
                if self._downsample_trivial_enabled:
                    _p0_ds_frac = (
                        self._downsample_trivial_frac_high_epoch
                        if epoch >= self._downsample_trivial_epoch_threshold
                        else self._downsample_trivial_frac_default
                    )
                else:
                    _p0_ds_frac = 0.0
                _p0_progress = ProgressReporter(
                    len(_p0_train_idx), prefix=f"  Phase 0 epoch {epoch + 1}/{_demo_epochs}: ",
                )
                _p0_rows_done = 0
                for obs_dict, bc_labels, ret_batch in dataset.iterate_minibatches(
                    batch_size=batch_size, shuffle=True, device=self.device,
                    indices_override=_p0_train_idx, returns=demo_returns,
                    downsample_trivial_frac=_p0_ds_frac,
                    downsample_trivial_cos_threshold=self._downsample_trivial_cos_threshold,
                    downsample_trivial_exclude_radius_steps=self._downsample_trivial_exclude_radius_steps,
                ):
                    _sat, _oat = _ai_types(obs_dict)
                    d_heads = self.decision_net(
                        obs_dict["self_feat"], obs_dict["other_feat"],
                        obs_dict["exists_mask"], obs_dict["ball_feat"], obs_dict["global_feat"],
                        _sat, _oat,
                    )
                    bc_labels = canonicalize_bc_labels(bc_labels, x_sign_of(obs_dict["self_feat"]))
                    # decision_net.value_head is frozen (single value head
                    # convention — see __init__); Phase 0's decision-net BC
                    # loss is heads-only (no value term from decision_net).
                    dec_bc_loss, _ = bc_loss_from_tensor(
                        bc_labels, d_heads, exec_heads=None,
                        direction_loss_weight=self._bc_dir_loss_w,
                        direction_loss_mode=self._bc_dir_loss_mode,
                        region_loss_weight=self._bc_region_loss_w,
                        dec_weight=self._bc_dec_weight,
                        dec_label_smoothing=self._bc_dec_label_smoothing,
                        return_breakdown=True,
                    )
                    epoch_floors.append(compute_bc_loss_floor(
                        bc_labels,
                        dec_weight=self._bc_dec_weight,
                        dec_label_smoothing=self._bc_dec_label_smoothing,
                        has_exec=False,
                    ))
                    # The one live critic IS trained here against the same
                    # demo returns — self.value_net when separate_value_net
                    # is on (via _value_heads(), routed to value_opt below),
                    # otherwise execution_net.value_head (via demo_opt). No
                    # other execution-network output (move/kick/tackle/etc
                    # heads) is used or optimized in this phase.
                    _value = self._value_heads(
                        obs_dict["self_feat"], obs_dict["other_feat"],
                        obs_dict["exists_mask"], obs_dict["ball_feat"], obs_dict["global_feat"],
                        d_heads, _sat, _oat,
                    )
                    val_loss = F.mse_loss(_value.squeeze(-1), ret_batch) / (ret_std ** 2)
                    combined = dec_bc_loss + self._phase0_value_coef * val_loss
                    demo_opt.zero_grad()
                    if _value_opt is not None:
                        _value_opt.zero_grad()
                    combined.backward()
                    if self._bc_max_grad_norm is not None:
                        nn.utils.clip_grad_norm_(
                            list(self.decision_net.parameters()) + _value_clip_params,
                            self._bc_max_grad_norm,
                        )
                    demo_opt.step()
                    if _value_opt is not None:
                        _value_opt.step()
                    epoch_losses.append(combined.item())
                    epoch_bc_losses.append(dec_bc_loss.item())
                    epoch_val_losses.append(val_loss.item())
                    _p0_rows_done += len(ret_batch)
                    _p0_progress.update(_p0_rows_done, postfix=f"loss={np.mean(epoch_losses):.4f}")
                # Force the bar to its final 100% render (with the trailing
                # newline that unlocks) even though _p0_rows_done usually
                # falls short of len(_p0_train_idx) -- downsample_trivial_frac
                # (see iterate_minibatches() above) skips a chunk of "trivial"
                # rows every epoch, so the running count legitimately never
                # reaches the pre-downsample total on its own. Without this,
                # the live bar never renders "finished" and the next log.info()
                # call runs straight onto its unterminated line.
                _p0_progress.update(_p0_progress.total, postfix=f"loss={np.mean(epoch_losses):.4f}")
                _mean_bc = float(np.mean(epoch_bc_losses))
                _mean_floor = float(np.mean(epoch_floors)) if epoch_floors else 0.0
                log.info(
                    f"  Phase 0 epoch {epoch + 1}/{_demo_epochs}: "
                    f"loss={np.mean(epoch_losses):.4f}  "
                    f"dec_bc={_mean_bc:.4f}  bc_adj={_mean_bc - _mean_floor:.4f}"
                    f"(floor={_mean_floor:.4f})  "
                    f"val_mse={np.mean(epoch_val_losses):.4f}(x{self._phase0_value_coef})="
                    f"{self._phase0_value_coef * np.mean(epoch_val_losses):.4f}"
                )
                if len(_p0_val_idx) > 0:
                    _p0_vl, _p0_vl_bc_adj, _p0_vl_mse, _p0_vl_by_outc = _eval_p0_val_loss()
                    _p0_improved = _p0_vl < (_p0_best_val_loss - self._p0_early_stop_min_delta)
                    _val_core = (
                        f"    val  p0_val_loss={_p0_vl:.4f}  bc_adj={_p0_vl_bc_adj:.4f}  "
                        f"val_mse={_p0_vl_mse:.4f}  best={min(_p0_best_val_loss, _p0_vl):.4f}"
                    )
                    if _p0_early_stop_enabled:
                        log.info(
                            _val_core
                            + ("  (improved)" if _p0_improved else f"  (patience {_p0_patience_ctr + 1}/{self._p0_early_stop_patience})")
                        )
                    else:
                        log.info(_val_core)
                    if _p0_vl_by_outc:
                        log.info(f"    val_rmse by outcome: {format_outcome_rmse_breakdown(_p0_vl_by_outc)}")
                    if _p0_improved:
                        _p0_best_val_loss = _p0_vl
                        if _p0_early_stop_enabled:
                            _p0_best_state = {
                                "decision_net": copy.deepcopy(self.decision_net.state_dict()),
                                **(({"value_net": copy.deepcopy(self.value_net.state_dict())} if self.separate_value_net
                                    else {"exec_value_head": copy.deepcopy(self.execution_net.value_head.state_dict())})),
                            }
                            _p0_patience_ctr = 0
                    elif _p0_early_stop_enabled:
                        _p0_patience_ctr += 1
                        if _p0_patience_ctr >= self._p0_early_stop_patience:
                            log.info(
                                f"  [Phase 0] early stop at epoch {epoch + 1} "
                                f"(val stagnant for {self._p0_early_stop_patience} epochs, "
                                f"best={_p0_best_val_loss:.4f})"
                            )
                            _p0_stopped_early = True
                            break
            if _p0_stopped_early and _p0_best_state is not None:
                _load_state_dict_tolerant(self.decision_net, _p0_best_state["decision_net"], "decision_net")
                if self.separate_value_net:
                    self.value_net.load_state_dict(_p0_best_state["value_net"])
                else:
                    self.execution_net.value_head.load_state_dict(_p0_best_state["exec_value_head"])
                log.info(f"  [Phase 0] restored best-val weights (p0_val_loss={_p0_best_val_loss:.4f})")
            log.info(f"Phase 0 done (decision-net BC + critic value_head warm-up, {_demo_epochs} epoch(s))")
        elif _demo_epochs > 0 and not dataset.has_rewards:
            log.info(
                "Phase 0 skipped — dataset has no reward data "
                "(re-record demonstrations to enable demo value pretrain)"
            )

        if self._bc_train_value_only:
            self._unfreeze_after_value_only_bc()

        _save_pretrain_checkpoint("Phase 0")

        # --- Phase 1: BC epochs over the dataset ---
        # If dataset has reward data and bc_value_coef > 0, also train value
        # against demo returns every minibatch. Two different mechanisms
        # depending on separate_value_net:
        #   - not separate_value_net: value loss (against execution_net.
        #     value_head) is folded into total_loss and shares bc_opt's
        #     single backward/step with the BC loss -- see total_loss below.
        #   - separate_value_net: value loss (against the independent
        #     self.value_net) gets its OWN backward()/self.value_net_
        #     optimizer.step(), computed from a DETACHED d_heads, so it
        #     never contributes gradient into decision_net and never
        #     touches bc_opt/bc_losses/_eval_bc_val_loss's early-stop
        #     patience -- mirrors _ppo_update's identical detach-before-
        #     value_net pattern (see _detach_decision_heads' call sites),
        #     and fixes the gap where self.value_net previously only ever
        #     got demo-return training from Phase 0 (a separate, usually
        #     much shorter epoch budget), never from the bulk of Phase 1.
        _use_joint_val = (
            not self.separate_value_net
            and self._bc_value_coef > 0.0
            and dataset.has_rewards
        )
        _use_separate_value_training = (
            self.separate_value_net
            and self._bc_value_coef > 0.0
            and dataset.has_rewards
        )
        if _use_joint_val or _use_separate_value_training:
            _joint_returns = dataset.compute_returns(gamma=self._demo_value_pretrain_gamma)
            _joint_ret_std = float(np.std(_joint_returns).clip(1.0))
            log.info(
                f"Phase 1 BC epochs will include "
                f"{'joint' if _use_joint_val else 'separate (self.value_net, detached from decision_net)'} "
                f"value loss (coef={self._bc_value_coef}, gamma={self._demo_value_pretrain_gamma}, "
                f"returns std={_joint_ret_std:.2f})"
            )
        else:
            _joint_returns = None
            _joint_ret_std = 1.0
        # Fresh optimizer, NOT self.value_net_optimizer -- see Phase 0's
        # identical fix/rationale above (that one is tuned for gentle
        # per-rollout PPO updates, ~20x slower than demo_value_pretrain_lr,
        # and reusing it here also drags Phase 1's momentum state into real
        # PPO training afterward). Only actually built/used when
        # _use_separate_value_training is True.
        _sep_value_opt = (
            torch.optim.Adam(self.value_net.parameters(), lr=self._demo_value_pretrain_lr, eps=1e-5)
            if _use_separate_value_training else None
        )

        # Do BC first so the rollout is collected with the BC-warmed policy,
        # giving on-policy value targets instead of random-init targets.
        # bc_losses is declared here (not just inside the loop) so the
        # "BC pre-training done" log line below stays well-defined even when
        # n_epochs=0 (e.g. bc.bc_pretrain_epochs_from_ckpt=0 with
        # --latest-pretrain/--pretrain-from-checkpoint) -- the loop body then
        # never runs and bc_losses is simply empty.
        bc_losses: list[float] = []
        bc_floors: list[float] = []

        # --- Episode-level train/val split, reported every epoch (see
        # bc.bc_pretrain_early_stop_patience in ai_config.json for the
        # OPTIONAL early-stop behaviour layered on top). The split itself
        # (and the per-epoch "val bc_val_loss=" log line below) is always
        # computed whenever the dataset has >=2 episodes, regardless of
        # whether early stop is enabled (patience=0 = report-only, no
        # stopping/best-weight-restore -- matches prior "train blind"
        # behaviour for the actual training, just with visibility added). ---
        _bc_early_stop_enabled = self._bc_pretrain_early_stop_patience > 0
        _bc_train_idx, _bc_val_idx = dataset.split_train_val_indices(val_frac=0.15, valid_only=True)
        log.info(
            f"  BC pretrain split: {len(_bc_train_idx):,} train rows"
            + (f"  |  {len(_bc_val_idx):,} val rows" if len(_bc_val_idx) > 0 else "  (val split empty -- <2 episodes, val loss unavailable)")
        )
        _bc_best_val_loss = float("inf")
        _bc_best_state: Optional[dict] = None
        _bc_patience_ctr = 0
        _bc_stopped_early = False

        def _eval_bc_val_loss() -> float:
            """Mean BC loss (no grad) over the held-out val rows, current weights."""
            self.decision_net.eval()
            self.execution_net.eval()
            _losses: list[float] = []
            with torch.no_grad():
                for _obs, _labels in dataset.iterate_minibatches(
                    batch_size=batch_size, shuffle=False, device=self.device,
                    indices_override=_bc_val_idx,
                ):
                    _sat_v, _oat_v = _ai_types(_obs)
                    _d_v = self.decision_net(
                        _obs["self_feat"], _obs["other_feat"], _obs["exists_mask"],
                        _obs["ball_feat"], _obs["global_feat"], _sat_v, _oat_v,
                    )
                    _e_v = self.execution_net(
                        _obs["self_feat"], _obs["other_feat"], _obs["exists_mask"],
                        _obs["ball_feat"], _obs["global_feat"], _d_v, _sat_v, _oat_v,
                    )
                    _labels = canonicalize_bc_labels(_labels, x_sign_of(_obs["self_feat"]))
                    _losses.append(bc_loss_from_tensor(
                        _labels, _d_v, _e_v,
                        direction_loss_weight=self._bc_dir_loss_w,
                        direction_loss_mode=self._bc_dir_loss_mode,
                        region_loss_weight=self._bc_region_loss_w,
                        pos_weight_kick=self._bc_pos_weight_kick,
                        pos_weight_tackle_attempt=self._bc_pos_weight_tackle_attempt,
                        dec_weight=self._bc_dec_weight,
                        exec_weight=self._bc_exec_weight,
                        dec_label_smoothing=self._bc_dec_label_smoothing,
                        exec_label_smoothing=self._bc_exec_label_smoothing,
                    ).item())
            self.decision_net.train()
            self.execution_net.train()
            return float(np.mean(_losses)) if _losses else float("nan")

        def _eval_p1_value_val_loss() -> tuple[float, float]:
            """Mean value MSE (normalized) + RMSE (raw), no grad, over the
            held-out val rows -- mirrors _eval_bc_val_loss but for the value
            head/net (execution_net.value_head, or self.value_net when
            separate_value_net), so Phase 1's value loss has a genuine
            held-out counterpart. Only meaningful when _use_joint_val or
            _use_separate_value_training (returns are otherwise undefined)."""
            self.decision_net.eval()
            if _use_separate_value_training:
                self.value_net.eval()
            else:
                self.execution_net.eval()
            _losses: list[float] = []
            _raw_mses: list[float] = []
            with torch.no_grad():
                for _obs_v, _lbl_v, _ret_v in dataset.iterate_minibatches(
                    batch_size=batch_size, shuffle=False, device=self.device,
                    indices_override=_bc_val_idx, returns=_joint_returns,
                ):
                    _sat_v, _oat_v = _ai_types(_obs_v)
                    _d_v = self.decision_net(
                        _obs_v["self_feat"], _obs_v["other_feat"], _obs_v["exists_mask"],
                        _obs_v["ball_feat"], _obs_v["global_feat"], _sat_v, _oat_v,
                    )
                    if _use_separate_value_training:
                        _d_v_for_value = _detach_decision_heads(_d_v)
                        _value_v = self.value_net(
                            _obs_v["self_feat"], _obs_v["other_feat"], _obs_v["exists_mask"],
                            _obs_v["ball_feat"], _obs_v["global_feat"], _d_v_for_value, _sat_v, _oat_v,
                            value_only=True,
                        )
                    else:
                        _value_v = self.execution_net(
                            _obs_v["self_feat"], _obs_v["other_feat"], _obs_v["exists_mask"],
                            _obs_v["ball_feat"], _obs_v["global_feat"], _d_v, _sat_v, _oat_v,
                            value_only=True,
                        )
                    _v = _value_v.squeeze(-1)
                    _raw_mse = F.mse_loss(_v, _ret_v)
                    _raw_mses.append(_raw_mse.item())
                    _losses.append((_raw_mse / (_joint_ret_std ** 2)).item())
            self.decision_net.train()
            if _use_separate_value_training:
                self.value_net.train()
            else:
                self.execution_net.train()
            if not _losses:
                return float("nan"), float("nan")
            return float(np.mean(_losses)), float(np.sqrt(np.mean(_raw_mses)))

        _p1_n_epochs = 0 if self._bc_train_value_only else n_epochs
        if self._bc_train_value_only:
            log.info("  Phase 1 skipped (bc_train_value_only=True) -- Phase 0 already covers value training.")
        # Per-execution-head grad norm, accumulated across every epoch/
        # minibatch of Phase 1 -- same diagnostic PPO's _ppo_update already
        # gets (see _measure_exec_head_grad_norms_and_maybe_log_spike),
        # summarized once at the end of Phase 1 below. Also runs the
        # grad-norm-spike per-parameter breakdown inline, same as PPO.
        _bc_head_grad_norm: dict[str, list[float]] = {name: [] for name, _ in EXEC_HEAD_MODULES}
        # Total (combined decision_net+execution_net, pre-clip) grad norm
        # per minibatch -- Phase 1 never logged this at all before (only
        # DAgger's mean_grad_norm did, and only a mean, not the full spread).
        _bc_raw_grad_norm: list[float] = []
        for epoch in range(_p1_n_epochs):
            _epoch_t0 = time.monotonic()
            bc_losses = []
            bc_floors = []
            val_losses: list[float] = []
            val_raw_mse_losses: list[float] = []
            sep_val_losses: list[float] = []
            sep_val_raw_mse_losses: list[float] = []
            dir_cosines: list[float] = []
            kick_dir_cosines: list[float] = []
            move_probs: list[float] = []
            sprint_probs: list[float] = []
            kick_probs: list[float] = []
            tackle_attempt_probs: list[float] = []
            _kick_tp = _kick_fp = _kick_fn = 0.0
            _tackle_tp = _tackle_fp = _tackle_fn = 0.0
            _bkdn_acc: dict[str, float] = {}
            _bkdn_floor_acc: dict[str, float] = {}
            _bkdn_n: int = 0
            if self._downsample_trivial_enabled:
                _ds_frac = (
                    self._downsample_trivial_frac_high_epoch
                    if epoch >= self._downsample_trivial_epoch_threshold
                    else self._downsample_trivial_frac_default
                )
                _ds_stats = dataset.downsample_trivial_stats(
                    valid_only=True,
                    cos_threshold=self._downsample_trivial_cos_threshold,
                    exclude_radius_steps=self._downsample_trivial_exclude_radius_steps,
                    frac=_ds_frac,
                )
                log.info(
                    f"  Downsample trivial rows (epoch {epoch + 1}): "
                    f"{_ds_stats['n_trivial']:,}/{_ds_stats['n_total']:,} "
                    f"({_ds_stats['trivial_frac']:.1%}) rows classified trivial, "
                    f"excluding ~{_ds_stats['n_excluded_at_frac']:,} this epoch "
                    f"(frac={_ds_frac:.2f})"
                )
            else:
                _ds_frac = 0.0
            _p1_progress = ProgressReporter(
                len(_bc_train_idx), prefix=f"  Phase 1 epoch {epoch + 1}/{_p1_n_epochs}: ",
            )
            _p1_rows_done = 0
            for _p1_mb_i, mb in enumerate(dataset.iterate_minibatches(
                batch_size=batch_size, shuffle=True, device=self.device,
                valid_only=True, returns=_joint_returns,
                downsample_trivial_frac=_ds_frac,
                downsample_trivial_cos_threshold=self._downsample_trivial_cos_threshold,
                downsample_trivial_exclude_radius_steps=self._downsample_trivial_exclude_radius_steps,
                indices_override=_bc_train_idx,
            )):
                if _use_joint_val or _use_separate_value_training:
                    obs_dict, bc_labels, ret_batch = mb
                else:
                    obs_dict, bc_labels = mb
                    ret_batch = None
                _p1_rows_done += bc_labels.shape[0]  # pre-augmentation row count
                # Augment with geometric flips + slot permutations (ALWAYS applied).
                if self.augment_n_slot_shuffles > 0:
                    obs_dict, bc_labels = augment_obs_bc(
                        obs_dict, bc_labels, self.augment_n_slot_shuffles, self._aug_rng
                    )
                    if ret_batch is not None:
                        n_aug = N_FLIP_VARIANTS * max(1, self.augment_n_slot_shuffles)
                        ret_batch = ret_batch.repeat(n_aug)
                _sat, _oat = _ai_types(obs_dict)
                d_heads = self.decision_net(
                    obs_dict["self_feat"], obs_dict["other_feat"],
                    obs_dict["exists_mask"], obs_dict["ball_feat"], obs_dict["global_feat"],
                    _sat, _oat,
                )
                e_heads = self.execution_net(
                    obs_dict["self_feat"], obs_dict["other_feat"],
                    obs_dict["exists_mask"], obs_dict["ball_feat"], obs_dict["global_feat"],
                    d_heads, _sat, _oat,
                )
                bc_labels = canonicalize_bc_labels(bc_labels, x_sign_of(obs_dict["self_feat"]))
                bc_loss, bkdn = bc_loss_from_tensor(
                    bc_labels, d_heads, e_heads,
                    direction_loss_weight=self._bc_dir_loss_w,
                    region_loss_weight=self._bc_region_loss_w,
                    pos_weight_kick=self._bc_pos_weight_kick,
                    pos_weight_tackle_attempt=self._bc_pos_weight_tackle_attempt,
                    dec_weight=self._bc_dec_weight,
                    exec_weight=self._bc_exec_weight,
                    dec_label_smoothing=self._bc_dec_label_smoothing,
                    exec_label_smoothing=self._bc_exec_label_smoothing,
                    return_breakdown=True,
                )
                dir_mag_reg = direction_magnitude_reg(e_heads, self._bc_dir_mag_reg_coef)
                total_loss = bc_loss + dir_mag_reg
                if _use_joint_val and ret_batch is not None:
                    # Single value head: execution_net only (decision_net.value
                    # is frozen — see __init__ note). Folded into total_loss,
                    # sharing bc_opt's one backward/step with the BC loss --
                    # only reached when NOT separate_value_net (see
                    # _use_separate_value_training's own detached branch
                    # below for that case).
                    v_exc = e_heads.value.squeeze(-1)
                    val_loss = F.mse_loss(v_exc, ret_batch) / (_joint_ret_std ** 2)
                    total_loss = bc_loss + dir_mag_reg + self._bc_value_coef * val_loss
                    val_losses.append(val_loss.item())
                    # raw MSE for RMSE reporting (values already in raw space)
                    with torch.no_grad():
                        raw_mse = F.mse_loss(v_exc, ret_batch)
                        val_raw_mse_losses.append(raw_mse.item())
                bc_opt.zero_grad()
                total_loss.backward()
                _bc_gn_vals = _measure_exec_head_grad_norms_and_maybe_log_spike(self, "BC", epoch, _p1_mb_i)
                for _head_name, _ in EXEC_HEAD_MODULES:
                    if _head_name in _bc_gn_vals:
                        _bc_head_grad_norm[_head_name].append(_bc_gn_vals[_head_name])
                _bc_raw_grad_norm.append(_bc_gn_vals["raw"])
                if self._bc_max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(
                        list(self.decision_net.parameters()) + list(self.execution_net.parameters()),
                        self._bc_max_grad_norm,
                    )
                bc_opt.step()

                if _use_separate_value_training and ret_batch is not None:
                    # Own backward/optimizer step, entirely separate from
                    # bc_opt/total_loss above -- d_heads is detached first so
                    # this loss contributes ZERO gradient to decision_net
                    # (mirrors _ppo_update's identical pattern, see
                    # _detach_decision_heads' other call sites/docstring).
                    # Deliberately NOT scaled by self._bc_value_coef: that
                    # coefficient exists to balance a value term AGAINST a BC
                    # term sharing the same backward pass (the _use_joint_val
                    # branch above) -- here value_loss is the only term in
                    # its own graph, so scaling it would just be equivalent
                    # to rescaling _sep_value_opt's own LR, adding a second,
                    # redundant knob for the same effect. Does NOT feed
                    # bc_losses/_eval_bc_val_loss, so it has no effect on
                    # Phase 1's BC early-stop patience (that stays a pure
                    # BC-only metric). Uses _sep_value_opt (fresh, demo_value_
                    # pretrain_lr), NOT self.value_net_optimizer -- see that
                    # variable's own setup comment above for why.
                    d_heads_for_value = _detach_decision_heads(d_heads)
                    _value_sep = self.value_net(
                        obs_dict["self_feat"], obs_dict["other_feat"], obs_dict["exists_mask"],
                        obs_dict["ball_feat"], obs_dict["global_feat"], d_heads_for_value, _sat, _oat,
                        value_only=True,
                    )
                    v_sep = _value_sep.squeeze(-1)
                    sep_val_loss = F.mse_loss(v_sep, ret_batch) / (_joint_ret_std ** 2)
                    _sep_value_opt.zero_grad()
                    sep_val_loss.backward()
                    if self._bc_max_grad_norm is not None:
                        nn.utils.clip_grad_norm_(list(self.value_net.parameters()), self._bc_max_grad_norm)
                    _sep_value_opt.step()
                    sep_val_losses.append(sep_val_loss.item())
                    with torch.no_grad():
                        sep_raw_mse = F.mse_loss(v_sep, ret_batch)
                        sep_val_raw_mse_losses.append(sep_raw_mse.item())
                bc_losses.append(bc_loss.item())
                bc_floors.append(compute_bc_loss_floor(
                    bc_labels,
                    pos_weight_kick=self._bc_pos_weight_kick,
                    pos_weight_tackle_attempt=self._bc_pos_weight_tackle_attempt,
                    dec_weight=self._bc_dec_weight,
                    exec_weight=self._bc_exec_weight,
                    dec_label_smoothing=self._bc_dec_label_smoothing,
                    exec_label_smoothing=self._bc_exec_label_smoothing,
                    has_exec=True,
                ))
                for k, v in bkdn.items():
                    _bkdn_acc[k] = _bkdn_acc.get(k, 0.0) + v
                _bkdn_floor = compute_bc_loss_floor_components(
                    bc_labels,
                    pos_weight_kick=self._bc_pos_weight_kick,
                    pos_weight_tackle_attempt=self._bc_pos_weight_tackle_attempt,
                    dec_label_smoothing=self._bc_dec_label_smoothing,
                    exec_label_smoothing=self._bc_exec_label_smoothing,
                    has_exec=True,
                )
                for k, v in _bkdn_floor.items():
                    _bkdn_floor_acc[k] = _bkdn_floor_acc.get(k, 0.0) + v
                _bkdn_n += 1

                # Accumulate cosine similarity between predicted and label move directions.
                # Use _I_VALID (index 14) not -1 (_I_OPPONENT_AI_TYPE = 0.0 for rules
                # demos, which makes valid_mask always False and causes dir_cos=nan).
                with torch.no_grad():
                    valid_mask = bc_labels[:, 14] > 0.5  # _I_VALID
                    has_dir = (bc_labels[:, 7].abs() + bc_labels[:, 8].abs()) > 1e-6
                    sel = valid_mask & has_dir
                    if sel.any():
                        pred_dir = e_heads.move_direction[sel]
                        tgt_dir = bc_labels[sel, 7:9]
                        eps = 1e-6
                        pred_n = pred_dir / (pred_dir.norm(dim=-1, keepdim=True) + eps)
                        cos_vals = (pred_n * tgt_dir).sum(dim=-1)
                        dir_cosines.append(cos_vals.mean().item())
                    kicked_mask = valid_mask & (bc_labels[:, 12] > 0.5)  # _I_KICK_THIS_TICK
                    has_kick_dir = (bc_labels[:, 18].abs() + bc_labels[:, 19].abs() + bc_labels[:, 24].abs()) > 1e-6  # _I_KICK_DIR_X/Y/Z
                    ksel = kicked_mask & has_kick_dir
                    if ksel.any():
                        pred_kdir = e_heads.kick_direction[ksel]
                        tgt_kdir = torch.stack([bc_labels[ksel, 18], bc_labels[ksel, 19], bc_labels[ksel, 24]], dim=-1)
                        eps = 1e-6
                        pred_kn = pred_kdir / (pred_kdir.norm(dim=-1, keepdim=True) + eps)
                        kcos_vals = (pred_kn * tgt_kdir).sum(dim=-1)
                        kick_dir_cosines.append(kcos_vals.mean().item())
                    if valid_mask.any():
                        move_probs.append(torch.sigmoid(e_heads.exec_move_logit.squeeze(-1)[valid_mask]).mean().item())
                        sprint_probs.append(torch.sigmoid(e_heads.sprint_logit.squeeze(-1)[valid_mask]).mean().item())
                        kick_probs.append(torch.sigmoid(e_heads.kick_logit.squeeze(-1)[valid_mask]).mean().item())
                        tackle_attempt_probs.append(torch.sigmoid(e_heads.tackle_attempt_logit.squeeze(-1)[valid_mask]).mean().item())
                        _tp, _fp, _fn = _binary_confusion_counts(e_heads.kick_logit, bc_labels[:, 12], valid_mask)
                        _kick_tp += _tp; _kick_fp += _fp; _kick_fn += _fn
                        _tp, _fp, _fn = _binary_confusion_counts(e_heads.tackle_attempt_logit, bc_labels[:, 13], valid_mask)
                        _tackle_tp += _tp; _tackle_fp += _fp; _tackle_fn += _fn
                _p1_progress.update(_p1_rows_done, postfix=f"bc_loss={np.mean(bc_losses):.4f}")
            # Force the final 100% render -- see the identical comment on
            # _p0_progress above (same downsample_trivial_frac cause).
            _p1_progress.update(_p1_progress.total, postfix=f"bc_loss={np.mean(bc_losses):.4f}")

            mean_cos = float(np.mean(dir_cosines)) if dir_cosines else float('nan')
            mean_kick_cos = float(np.mean(kick_dir_cosines)) if kick_dir_cosines else float('nan')
            mean_mv = float(np.mean(move_probs)) if move_probs else float('nan')
            mean_spr = float(np.mean(sprint_probs)) if sprint_probs else float('nan')
            mean_kk = float(np.mean(kick_probs)) if kick_probs else float('nan')
            mean_tk = float(np.mean(tackle_attempt_probs)) if tackle_attempt_probs else float('nan')
            kk_prec, kk_rec, kk_f1 = _precision_recall_f1(_kick_tp, _kick_fp, _kick_fn)
            tk_prec, tk_rec, tk_f1 = _precision_recall_f1(_tackle_tp, _tackle_fp, _tackle_fn)
            _kick_bkdn_keys = {"kick", "kick_direction", "kick_power", "kick_spin"}
            # Subtract each component's own analytic label-smoothing floor (see
            # compute_bc_loss_floor_components) before printing, so raw-magnitude
            # differences between components (e.g. exec_bce summing 4 smoothed
            # heads vs. direction's true 0 floor) don't masquerade as real
            # imitation-quality differences. Clamp at 0 — mean-of-floors vs.
            # mean-of-raw over the same minibatches can differ by float noise
            # right at convergence, and a tiny negative "adjusted" value reads
            # as a bug rather than the near-zero residual it actually is.
            bkdn_str = "  ".join(
                f"{k}={max(0.0, v/_bkdn_n - _bkdn_floor_acc.get(k, 0.0)/_bkdn_n):.5f}"
                if k in _kick_bkdn_keys else
                f"{k}={max(0.0, v/_bkdn_n - _bkdn_floor_acc.get(k, 0.0)/_bkdn_n):.3f}"
                for k, v in _bkdn_acc.items()
            ) if _bkdn_n else ""
            if val_losses:
                # NOTE: despite the name, val_losses/val_rmse below are accumulated
                # over TRAINING minibatches (_bc_train_idx), not the held-out val
                # split -- they're the epoch's mean training-batch value loss. See
                # the genuine held-out "value_val_loss=" figure on the "val" line
                # further down (_eval_p1_value_val_loss) for an actual val metric.
                val_rmse = float(np.sqrt(np.mean(val_raw_mse_losses))) if val_raw_mse_losses else float('nan')
                _mean_val = np.mean(val_losses)
                val_str = (
                    f"  train_value_loss={_mean_val:.4f}"
                    f"(x{self._bc_value_coef})={_mean_val * self._bc_value_coef:.4f}"
                    f"  rmse={val_rmse:.2f} (returns std={_joint_ret_std:.1f})"
                )
            else:
                val_str = ""
            if sep_val_losses:
                # Separate, detached self.value_net training (see
                # _use_separate_value_training above) -- NOT scaled by
                # self._bc_value_coef (no combined loss to weight against),
                # so no "(x coef)=" term here unlike val_str above. Same
                # training-batch-not-held-out-val caveat as val_str above --
                # see "value_val_loss=" on the "val" line for the real metric.
                sep_val_rmse = float(np.sqrt(np.mean(sep_val_raw_mse_losses))) if sep_val_raw_mse_losses else float('nan')
                sep_val_str = (
                    f"    value_net  train_loss={np.mean(sep_val_losses):.4f}  "
                    f"rmse={sep_val_rmse:.2f} (returns std={_joint_ret_std:.1f}, detached from decision_net)"
                )
            else:
                sep_val_str = ""
            _epoch_elapsed = time.monotonic() - _epoch_t0
            # Tabulated multi-line epoch summary (readability refactor only — every
            # field from the old single-line format is preserved, just grouped, with
            # full-word labels and long lines wrapped to avoid overflow. See
            # agent_plans/bc_execution_label_boundary_and_followups.md Part 4.
            _mean_bc_loss = float(np.mean(bc_losses))
            _mean_bc_floor = float(np.mean(bc_floors)) if bc_floors else 0.0
            _bc_lines = [
                f"  BC epoch {epoch + 1}/{n_epochs}  ({_epoch_elapsed:.1f}s)",
                f"    loss       bc={_mean_bc_loss:.4f}  bc_adj={_mean_bc_loss - _mean_bc_floor:.4f}"
                f"(floor={_mean_bc_floor:.4f})" + (val_str.strip() and f"  {val_str.strip()}" or ""),
                f"    heads      dir_cos={mean_cos:.3f}  kick_dir_cos={mean_kick_cos:.3f}",
                f"               move_prob={mean_mv:.3f}  sprint_prob={mean_spr:.3f}  "
                f"kick_prob={mean_kk:.3f}  tackle_prob={mean_tk:.3f}",
                f"    pr/rec     kick:   p={kk_prec:.3f}  r={kk_rec:.3f}  f1={kk_f1:.3f}  "
                f"(tp={_kick_tp:.0f} fp={_kick_fp:.0f} fn={_kick_fn:.0f})",
                f"               tackle: p={tk_prec:.3f}  r={tk_rec:.3f}  f1={tk_f1:.3f}  "
                f"(tp={_tackle_tp:.0f} fp={_tackle_fp:.0f} fn={_tackle_fn:.0f})",
            ]
            if sep_val_str:
                _bc_lines.append(sep_val_str)
            if bkdn_str:
                _bkdn_parts = bkdn_str.split("  ")
                _mid = (len(_bkdn_parts) + 1) // 2
                _bc_lines.append(f"    breakdown (floor-adj)  {'  '.join(_bkdn_parts[:_mid])}")
                if _bkdn_parts[_mid:]:
                    _bc_lines.append(f"                           {'  '.join(_bkdn_parts[_mid:])}")
            log.info("\n".join(_bc_lines))
            if _bc_raw_grad_norm:
                log.info(
                    f"    [BC grad norm total] mean={np.mean(_bc_raw_grad_norm):.3f}  "
                    f"min={np.min(_bc_raw_grad_norm):.3f}  max={np.max(_bc_raw_grad_norm):.3f}  "
                    f"std={np.std(_bc_raw_grad_norm):.3f}  (n={len(_bc_raw_grad_norm)})"
                )
            _bc_raw_grad_norm.clear()
            if any(_bc_head_grad_norm.values()):
                _bc_head_gn_str = "  ".join(
                    f"{name}={np.mean(vals):.3f}" for name, vals in _bc_head_grad_norm.items() if vals
                )
                log.info(f"    [BC exec head grad norm] {_bc_head_gn_str}")
            for _v in _bc_head_grad_norm.values():
                _v.clear()
            dir_cosines.clear()
            kick_dir_cosines.clear()
            move_probs.clear()
            sprint_probs.clear()
            _bkdn_acc.clear()
            _bkdn_floor_acc.clear()
            _bkdn_n = 0

            # --- BC pretrain val loss, reported every epoch; early stop is opt-in
            # (see bc.bc_pretrain_early_stop_patience). Also reports a genuine
            # held-out value loss/RMSE (_eval_p1_value_val_loss) on the same line
            # when a value term is being trained this phase -- separate from the
            # "train_value_loss="/"value_net  train_loss=" figures printed above,
            # which are training-batch metrics, not a val-set metric, despite the
            # similar name. Does NOT feed BC's own early-stop bookkeeping below
            # (that stays a pure BC-only metric, see _use_separate_value_training's
            # own note above for why). ---
            if len(_bc_val_idx) > 0:
                _bc_val_loss = _eval_bc_val_loss()
                _improved = _bc_val_loss < (_bc_best_val_loss - self._bc_pretrain_early_stop_min_delta)
                _val_line = f"    val        bc_val_loss={_bc_val_loss:.4f}  best={min(_bc_best_val_loss, _bc_val_loss):.4f}"
                if _bc_early_stop_enabled:
                    _val_line += ("  (improved)" if _improved else f"  (patience {_bc_patience_ctr + 1}/{self._bc_pretrain_early_stop_patience})")
                elif _improved:
                    _val_line += "  (improved, checkpointed)"
                if _use_joint_val or _use_separate_value_training:
                    _val_value_loss, _val_value_rmse = _eval_p1_value_val_loss()
                    _val_line += f"  value_val_loss={_val_value_loss:.4f}  rmse={_val_value_rmse:.2f}"
                log.info(_val_line)
                # Best-val tracking/checkpointing is independent of whether
                # early STOPPING is enabled (patience=0 = report-only for
                # the stop decision, but "checkpoint the actual best" is
                # useful regardless) -- previously this whole block was
                # gated on _bc_early_stop_enabled, so with early stop
                # disabled (the default), a new best was never captured
                # in memory OR on disk at all, and a crash mid-Phase-1
                # lost all progress back to whatever Phase 0 left on disk.
                if _improved:
                    _bc_best_val_loss = _bc_val_loss
                    _bc_best_state = {
                        "decision_net": copy.deepcopy(self.decision_net.state_dict()),
                        "execution_net": copy.deepcopy(self.execution_net.state_dict()),
                    }
                    _save_pretrain_checkpoint(f"Phase 1 (new best bc_val_loss={_bc_best_val_loss:.4f}, epoch {epoch + 1})")
                    if _bc_early_stop_enabled:
                        _bc_patience_ctr = 0
                elif _bc_early_stop_enabled:
                    _bc_patience_ctr += 1
                    if _bc_patience_ctr >= self._bc_pretrain_early_stop_patience:
                        log.info(
                            f"  [BC pretrain] early stop at epoch {epoch + 1} "
                            f"(val stagnant for {self._bc_pretrain_early_stop_patience} epochs, "
                            f"best={_bc_best_val_loss:.4f})"
                        )
                        _bc_stopped_early = True
                        break
        # Restore the best-val weights before Phase 1's own final checkpoint
        # below, regardless of whether early stop actually triggered --
        # otherwise a full (non-early-stopped) run's final on-disk state is
        # just whatever the LAST epoch happened to leave (which can be worse
        # than an earlier epoch, e.g. from overfitting), silently overwriting
        # every "new best" checkpoint saved above with something worse.
        if _bc_best_state is not None:
            _load_state_dict_tolerant(self.decision_net, _bc_best_state["decision_net"], "decision_net")
            self.execution_net.load_state_dict(_bc_best_state["execution_net"])
            log.info(
                f"  [BC pretrain] restored best-val weights (bc_val_loss={_bc_best_val_loss:.4f})"
                + ("" if _bc_stopped_early else " before final Phase 1 checkpoint")
            )
        if bc_losses:
            log.info(f"BC pre-training done ({n_epochs} epoch(s), final bc_loss={np.mean(bc_losses):.4f})")
        else:
            log.info(f"BC pre-training done ({n_epochs} epoch(s) -- no BC epochs ran, dataset/pretrain skipped)")

        _save_pretrain_checkpoint("Phase 1")

        # --- Eval: BC-pretrained policy, before DAgger ---
        # Runs regardless of whether DAgger itself is enabled below, so a
        # Phase-1-only run still gets a "how good is the network right now"
        # readout right after BC pretraining, not just loss curves.
        log.info("Evaluating BC-pretrained policy (before DAgger)...")
        self._eval_vs_rules(env.max_episode_s)

        # --- DAgger (ai/ppo/dagger.py): alternate full-dataset BC training
        # with closed-loop rollout-and-aggregate, correcting exposure bias
        # in the BC-pretrained policy before value pre-training / Phase 4
        # repair run on top of it. Opt-in (bc.dagger_iterations, default 0). ---
        if self._dagger_iterations > 0:
            if phase_id != 1:
                log.warning(
                    f"bc.dagger_iterations={self._dagger_iterations} but phase_id={phase_id} "
                    f"(!= 1) -- DAgger's opponent-roll/phase1_labels() machinery is Phase-1-"
                    f"specific, skipping."
                )
            elif self._bc_train_value_only:
                log.info("DAgger skipped (bc_train_value_only=True) -- policy was never trained in Phase 1.")
            else:
                from footballcoach.ai.ppo.dagger import run_dagger_phase
                log.info(
                    f"DAgger: {self._dagger_iterations} iteration(s), "
                    f"{self._dagger_bc_epochs_per_iter} full-dataset BC epoch(s)/iteration, "
                    f"buffer cap {self._dagger_buffer_max_size}"
                )
                run_dagger_phase(
                    self, dataset, _bc_train_idx, bc_opt,
                    iterations=self._dagger_iterations,
                    bc_epochs_per_iter=self._dagger_bc_epochs_per_iter,
                    buffer_max_size=self._dagger_buffer_max_size,
                    max_replay_steps=self._dagger_max_replay_steps,
                    episodes_per_iteration=self._dagger_episodes_per_iteration,
                    n_workers=self._dagger_n_workers,
                    batch_size=batch_size,
                    checkpoint_fn=_save_pretrain_checkpoint,
                )
                _save_pretrain_checkpoint("DAgger")
                log.info("Evaluating policy after DAgger...")
                self._eval_vs_rules(env.max_episode_s)

        # --- Phase 2/3: collect on-policy rollout + value head warm-up ---
        # Delegates to pretrain_value(), which collects rollout_steps of
        # experience with the BC-warmed policy, computes GAE returns, applies
        # augmentation, and fits the value heads for value_epochs (with trunk
        # freezing retained — a DIFFERENT freezing decision than Phase 0 above,
        # see pretrain_value()'s docstring). This used to be duplicated inline
        # here; extracted so pretrain_value() remains the single source of
        # truth and is also usable standalone.
        self.pretrain_value(
            env,
            n_steps=rollout_steps,
            n_epochs=max(1, value_epochs),
            lr=value_lr,
            batch_size=batch_size,
            experiment_separate_value_net=experiment_separate_value_net,
            phase_id=phase_id,
        )

        # --- BC degradation check: re-evaluate BC loss over dataset after value warm-up ---
        # Meaningless (and bc_losses is empty, so the "before" mean would be
        # NaN over an empty list) when bc_train_value_only=True -- the policy
        # was never trained in this call, so it cannot have degraded.
        if self._bc_train_value_only:
            log.info("BC degradation check skipped (bc_train_value_only=True) -- policy was never trained.")
        else:
            self.decision_net.eval()
            self.execution_net.eval()
            post_bc_losses = []
            with torch.no_grad():
                for obs_dict, bc_labels in dataset.iterate_minibatches(
                    batch_size=batch_size, shuffle=False, device=self.device, valid_only=True
                ):
                    _sat, _oat = _ai_types(obs_dict)
                    d_check = self.decision_net(
                        obs_dict["self_feat"], obs_dict["other_feat"],
                        obs_dict["exists_mask"], obs_dict["ball_feat"], obs_dict["global_feat"],
                        _sat, _oat,
                    )
                    e_check = self.execution_net(
                        obs_dict["self_feat"], obs_dict["other_feat"],
                        obs_dict["exists_mask"], obs_dict["ball_feat"], obs_dict["global_feat"],
                        d_check, _sat, _oat,
                    )
                    bc_labels = canonicalize_bc_labels(bc_labels, x_sign_of(obs_dict["self_feat"]))
                    post_bc_losses.append(bc_loss_from_tensor(
                        bc_labels, d_check, e_check,
                        pos_weight_kick=self._bc_pos_weight_kick,
                        pos_weight_tackle_attempt=self._bc_pos_weight_tackle_attempt,
                        dec_weight=self._bc_dec_weight,
                        exec_weight=self._bc_exec_weight,
                        dec_label_smoothing=self._bc_dec_label_smoothing,
                        exec_label_smoothing=self._bc_exec_label_smoothing,
                    ).item())
            self.decision_net.train()
            self.execution_net.train()
            post_bc_loss = float(np.mean(post_bc_losses))
            bc_loss_before_value = float(np.mean(bc_losses))
            delta = post_bc_loss - bc_loss_before_value
            degraded = delta > 0.05
            log.info(
                f"BC check after value warm-up: bc_loss={post_bc_loss:.4f} "
                f"(before={bc_loss_before_value:.4f}, delta={delta:+.4f})"
                + ("  *** WARNING: significant BC degradation!" if degraded else "  OK")
            )

        _save_pretrain_checkpoint("Phase 2/3")

        # --- Phase 4: joint BC repair epoch (all params, bc_lr) ---
        # Runs the BC dataset once more with all parameters trainable to restore
        # any policy quality lost during value-only warm-up, while also keeping
        # the freshly-warmed value head in the training graph.
        repair_epochs = 0 if self._bc_train_value_only else int(self._bc_cfg.get("bc_repair_epochs", 1))
        if self._bc_train_value_only:
            log.info("  BC repair epochs skipped (bc_train_value_only=True) -- policy was never trained, nothing to repair.")
        if repair_epochs > 0:
            _repair_lr = repair_lr if repair_lr is not None else bc_lr
            repair_opt = torch.optim.Adam(
                list(self.decision_net.parameters()) + list(self.execution_net.parameters()),
                lr=_repair_lr, eps=1e-5,
            )
            for epoch in range(repair_epochs):
                repair_losses = []
                dir_cosines_r: list[float] = []
                kick_dir_cosines_r: list[float] = []
                move_probs_r: list[float] = []
                sprint_probs_r: list[float] = []
                kick_probs_r: list[float] = []
                tackle_attempt_probs_r: list[float] = []
                _kick_tp_r = _kick_fp_r = _kick_fn_r = 0.0
                _tackle_tp_r = _tackle_fp_r = _tackle_fn_r = 0.0
                _bkdn_r_acc: dict[str, float] = {}
                _bkdn_r_floor_acc: dict[str, float] = {}
                _bkdn_r_n: int = 0
                _p4_progress = ProgressReporter(
                    len(dataset.valid_indices()), prefix=f"  Phase 4 (repair) epoch {epoch + 1}/{repair_epochs}: ",
                )
                _p4_rows_done = 0
                for obs_dict, bc_labels in dataset.iterate_minibatches(
                    batch_size=batch_size, shuffle=True, device=self.device, valid_only=True
                ):
                    _p4_rows_done += bc_labels.shape[0]  # pre-augmentation row count
                    # Augment repair minibatch (ALWAYS applied).
                    if self.augment_n_slot_shuffles > 0:
                        obs_dict, bc_labels = augment_obs_bc(
                            obs_dict, bc_labels, self.augment_n_slot_shuffles, self._aug_rng
                        )
                    _sat, _oat = _ai_types(obs_dict)
                    d_r = self.decision_net(
                        obs_dict["self_feat"], obs_dict["other_feat"],
                        obs_dict["exists_mask"], obs_dict["ball_feat"], obs_dict["global_feat"],
                        _sat, _oat,
                    )
                    e_r = self.execution_net(
                        obs_dict["self_feat"], obs_dict["other_feat"],
                        obs_dict["exists_mask"], obs_dict["ball_feat"], obs_dict["global_feat"],
                        d_r, _sat, _oat,
                    )
                    bc_labels = canonicalize_bc_labels(bc_labels, x_sign_of(obs_dict["self_feat"]))
                    loss_r, bkdn_r = bc_loss_from_tensor(
                        bc_labels, d_r, e_r,
                        direction_loss_weight=self._bc_dir_loss_w,
                        direction_loss_mode=self._bc_dir_loss_mode,
                        region_loss_weight=self._bc_region_loss_w,
                        pos_weight_kick=self._bc_pos_weight_kick,
                        pos_weight_tackle_attempt=self._bc_pos_weight_tackle_attempt,
                        dec_weight=self._bc_dec_weight,
                        exec_weight=self._bc_exec_weight,
                        dec_label_smoothing=self._bc_dec_label_smoothing,
                        exec_label_smoothing=self._bc_exec_label_smoothing,
                        return_breakdown=True,
                    )
                    loss_r = loss_r + direction_magnitude_reg(e_r, self._bc_dir_mag_reg_coef)
                    repair_opt.zero_grad()
                    loss_r.backward()
                    if self._bc_max_grad_norm is not None:
                        nn.utils.clip_grad_norm_(
                            list(self.decision_net.parameters()) + list(self.execution_net.parameters()),
                            self._bc_max_grad_norm,
                        )
                    repair_opt.step()
                    repair_losses.append(loss_r.item())
                    for k, v in bkdn_r.items():
                        _bkdn_r_acc[k] = _bkdn_r_acc.get(k, 0.0) + v
                    _bkdn_r_floor = compute_bc_loss_floor_components(
                        bc_labels,
                        pos_weight_kick=self._bc_pos_weight_kick,
                        pos_weight_tackle_attempt=self._bc_pos_weight_tackle_attempt,
                        dec_label_smoothing=self._bc_dec_label_smoothing,
                        exec_label_smoothing=self._bc_exec_label_smoothing,
                        has_exec=True,
                    )
                    for k, v in _bkdn_r_floor.items():
                        _bkdn_r_floor_acc[k] = _bkdn_r_floor_acc.get(k, 0.0) + v
                    _bkdn_r_n += 1

                    with torch.no_grad():
                        valid_mask = bc_labels[:, 14] > 0.5  # _I_VALID (not -1 = _I_OPPONENT_AI_TYPE)
                        has_dir = (bc_labels[:, 7].abs() + bc_labels[:, 8].abs()) > 1e-6
                        sel = valid_mask & has_dir
                        if sel.any():
                            pred_dir = e_r.move_direction[sel]
                            tgt_dir = bc_labels[sel, 7:9]
                            eps = 1e-6
                            pred_n = pred_dir / (pred_dir.norm(dim=-1, keepdim=True) + eps)
                            dir_cosines_r.append((pred_n * tgt_dir).sum(dim=-1).mean().item())
                        kicked_mask_r = valid_mask & (bc_labels[:, 12] > 0.5)  # _I_KICK_THIS_TICK
                        has_kick_dir_r = (bc_labels[:, 18].abs() + bc_labels[:, 19].abs() + bc_labels[:, 24].abs()) > 1e-6
                        ksel_r = kicked_mask_r & has_kick_dir_r
                        if ksel_r.any():
                            pred_kdir_r = e_r.kick_direction[ksel_r]
                            tgt_kdir_r = torch.stack([bc_labels[ksel_r, 18], bc_labels[ksel_r, 19], bc_labels[ksel_r, 24]], dim=-1)
                            eps = 1e-6
                            pred_kn_r = pred_kdir_r / (pred_kdir_r.norm(dim=-1, keepdim=True) + eps)
                            kick_dir_cosines_r.append((pred_kn_r * tgt_kdir_r).sum(dim=-1).mean().item())
                        if valid_mask.any():
                            move_probs_r.append(torch.sigmoid(e_r.exec_move_logit.squeeze(-1)[valid_mask]).mean().item())
                            sprint_probs_r.append(torch.sigmoid(e_r.sprint_logit.squeeze(-1)[valid_mask]).mean().item())
                            kick_probs_r.append(torch.sigmoid(e_r.kick_logit.squeeze(-1)[valid_mask]).mean().item())
                            tackle_attempt_probs_r.append(torch.sigmoid(e_r.tackle_attempt_logit.squeeze(-1)[valid_mask]).mean().item())
                            _tp, _fp, _fn = _binary_confusion_counts(e_r.kick_logit, bc_labels[:, 12], valid_mask)
                            _kick_tp_r += _tp; _kick_fp_r += _fp; _kick_fn_r += _fn
                            _tp, _fp, _fn = _binary_confusion_counts(e_r.tackle_attempt_logit, bc_labels[:, 13], valid_mask)
                            _tackle_tp_r += _tp; _tackle_fp_r += _fp; _tackle_fn_r += _fn
                    _p4_progress.update(_p4_rows_done, postfix=f"loss={np.mean(repair_losses):.4f}")
                # Force the final 100% render -- see the identical comment on
                # _p0_progress above. Phase 4 doesn't downsample today, but
                # this keeps all three phases' bars consistent/robust if that
                # ever changes.
                _p4_progress.update(_p4_progress.total, postfix=f"loss={np.mean(repair_losses):.4f}")

                mean_cos_r = float(np.mean(dir_cosines_r)) if dir_cosines_r else float('nan')
                mean_kick_cos_r = float(np.mean(kick_dir_cosines_r)) if kick_dir_cosines_r else float('nan')
                mean_mv_r = float(np.mean(move_probs_r)) if move_probs_r else float('nan')
                mean_spr_r = float(np.mean(sprint_probs_r)) if sprint_probs_r else float('nan')
                mean_kk_r = float(np.mean(kick_probs_r)) if kick_probs_r else float('nan')
                mean_tk_r = float(np.mean(tackle_attempt_probs_r)) if tackle_attempt_probs_r else float('nan')
                kk_prec_r, kk_rec_r, kk_f1_r = _precision_recall_f1(_kick_tp_r, _kick_fp_r, _kick_fn_r)
                tk_prec_r, tk_rec_r, tk_f1_r = _precision_recall_f1(_tackle_tp_r, _tackle_fp_r, _tackle_fn_r)
                # Also measure value loss on the stored rollout returns (no gradient)
                with torch.no_grad():
                    val_losses_r = []
                    for start_r in range(0, n_rollout, batch_size):
                        mb_obs_r = {k.replace("obs/", ""): rollout_batch[k][start_r:start_r+batch_size].to(self.device)
                                    for k in rollout_batch if k.startswith("obs/")}
                        mb_ret_r = returns_t[start_r:start_r+batch_size]
                        _sat_r, _oat_r = _ai_types(mb_obs_r)
                        d_vr = self.decision_net(
                            mb_obs_r["self_feat"], mb_obs_r["other_feat"],
                            mb_obs_r["exists_mask"], mb_obs_r["ball_feat"], mb_obs_r["global_feat"],
                            _sat_r, _oat_r,
                        )
                        _val_net_r = self.value_net if self.separate_value_net else self.execution_net
                        _value_vr = _val_net_r(
                            mb_obs_r["self_feat"], mb_obs_r["other_feat"],
                            mb_obs_r["exists_mask"], mb_obs_r["ball_feat"], mb_obs_r["global_feat"],
                            d_vr, _sat_r, _oat_r, value_only=True,
                        )
                        pred_vr = _value_vr.squeeze(-1)  # single critic (execution_net, or self.value_net)
                        val_losses_r.append(F.mse_loss(pred_vr, mb_ret_r).item() / (ret_std ** 2).item())
                _kick_bkdn_keys = {"kick", "kick_direction", "kick_power", "kick_spin"}
                # Floor-adjusted, same rationale as the main BC epoch loop above —
                # see compute_bc_loss_floor_components.
                bkdn_r_str = "  ".join(
                    f"{k}={max(0.0, v/_bkdn_r_n - _bkdn_r_floor_acc.get(k, 0.0)/_bkdn_r_n):.5f}"
                    if k in _kick_bkdn_keys else
                    f"{k}={max(0.0, v/_bkdn_r_n - _bkdn_r_floor_acc.get(k, 0.0)/_bkdn_r_n):.3f}"
                    for k, v in _bkdn_r_acc.items()
                ) if _bkdn_r_n else ""
                # Tabulated multi-line epoch summary (readability refactor only — every
                # field from the old single-line format is preserved, just grouped, with
                # full-word labels and long lines wrapped to avoid overflow. See
                # agent_plans/bc_execution_label_boundary_and_followups.md Part 4.
                _bc_r_lines = [
                    f"  BC repair epoch {epoch + 1}/{repair_epochs}",
                    f"    loss       bc={np.mean(repair_losses):.4f}  val={np.mean(val_losses_r):.4f}",
                    f"    heads      dir_cos={mean_cos_r:.3f}  kick_dir_cos={mean_kick_cos_r:.3f}",
                    f"               move_prob={mean_mv_r:.3f}  sprint_prob={mean_spr_r:.3f}  "
                    f"kick_prob={mean_kk_r:.3f}  tackle_prob={mean_tk_r:.3f}",
                    f"    pr/rec     kick:   p={kk_prec_r:.3f}  r={kk_rec_r:.3f}  f1={kk_f1_r:.3f}  "
                    f"(tp={_kick_tp_r:.0f} fp={_kick_fp_r:.0f} fn={_kick_fn_r:.0f})",
                    f"               tackle: p={tk_prec_r:.3f}  r={tk_rec_r:.3f}  f1={tk_f1_r:.3f}  "
                    f"(tp={_tackle_tp_r:.0f} fp={_tackle_fp_r:.0f} fn={_tackle_fn_r:.0f})",
                ]
                if bkdn_r_str:
                    _bkdn_r_parts = bkdn_r_str.split("  ")
                    _mid_r = (len(_bkdn_r_parts) + 1) // 2
                    _bc_r_lines.append(f"    breakdown (floor-adj)  {'  '.join(_bkdn_r_parts[:_mid_r])}")
                    if _bkdn_r_parts[_mid_r:]:
                        _bc_r_lines.append(f"                           {'  '.join(_bkdn_r_parts[_mid_r:])}")
                log.info("\n".join(_bc_r_lines))
            log.info(
                f"BC repair done ({repair_epochs} epoch(s), final bc_loss={np.mean(repair_losses):.4f}  "
                f"val_loss={np.mean(val_losses_r):.4f})\n"
                f"  final heads  dir_cos={mean_cos_r:.3f}  kick_dir_cos={mean_kick_cos_r:.3f}  "
                f"move_prob={mean_mv_r:.3f}  sprint_prob={mean_spr_r:.3f}\n"
                f"               kick_prob={mean_kk_r:.3f}  tackle_prob={mean_tk_r:.3f}"
            )

        _save_pretrain_checkpoint("Phase 4")

        log.info("Combined pre-training complete.")

    def _spawn_value_pretrain_workers(self, phase_id: Optional[int]) -> Optional["_ValuePretrainWorkers"]:
        """Spawn the parallel rollout workers for the value-pretrain
        rollout, or return None when parallel collection doesn't apply
        (``ppo.value_pretrain_n_processes <= 1`` or no ``phase_id`` -- the
        single-process branch of ``_collect_value_pretrain_rollout`` then
        steps ``env`` directly).

        ``ppo.value_pretrain_batched_rollout`` (default off) selects the
        worker kind: True = ``batched_rollout_worker.py`` workers, each
        owning ``ppo.value_pretrain_envs_per_process`` envs and batching all
        of their decisions into one network call per round (the same
        mechanism the main PPO loop uses; ``batch_secondary_players`` is
        reused from the main loop's setting so self-play opponents get
        batched too); False = today's plain one-env-per-process
        ``rollout_worker.py`` workers. Both share one ``ctx.Value`` step
        counter for a single aggregate progress bar. Batched workers stream
        WHOLE-EPISODE chunks every ``ppo.value_pretrain_chunk_steps`` rows
        (0 = one send at the end): a flush only ever sends completed
        episodes and keeps each env's unfinished tail, so MC returns stay
        exact and the train/val episode split stays clean while per-process
        memory stays bounded to one chunk -- see ai/knowledge.md "PPG-style
        value refit" and batched_rollout_worker.py's "Chunked streaming".

        Callers own the lifecycle: pair with ``_close_value_pretrain_workers``.
        """
        if not (self.value_pretrain_n_processes > 1 and phase_id is not None):
            return None
        import multiprocessing

        n_workers = self.value_pretrain_n_processes
        base_seed = random.randint(0, 2**31 - 1)
        # Must be created with the SAME "spawn" context the spawn functions
        # use, and passed at process-creation time (a raw ctx.Value can't be
        # sent to an already-running worker afterwards).
        ctx = multiprocessing.get_context("spawn")
        progress_value = ctx.Value("l", 0)
        if self.value_pretrain_batched_rollout:
            from footballcoach.ai.ppo.batched_rollout_worker import spawn_batched_workers

            envs_per_process = max(1, self.value_pretrain_envs_per_process)
            handles = spawn_batched_workers(
                phase_id, n_workers, envs_per_process, base_seed,
                self.separate_value_net, self.worker_torch_threads,
                batch_secondary_players=self.batch_secondary_players,
                chunk_steps=self.value_pretrain_chunk_steps or None, progress_value=progress_value,
            )
            return _ValuePretrainWorkers(handles, True, progress_value, n_workers, envs_per_process)
        from footballcoach.ai.ppo.rollout_worker import spawn_workers

        handles = spawn_workers(
            phase_id, n_workers, base_seed, self.separate_value_net, self.worker_torch_threads,
            progress_value=progress_value,
        )
        return _ValuePretrainWorkers(handles, False, progress_value, n_workers, 1)

    def _close_value_pretrain_workers(self, pool: Optional["_ValuePretrainWorkers"]) -> None:
        if pool is None:
            return
        if pool.batched:
            from footballcoach.ai.ppo.batched_rollout_worker import close_batched_workers

            close_batched_workers(pool.handles)
        else:
            from footballcoach.ai.ppo.rollout_worker import close_workers

            close_workers(pool.handles)

    def _collect_value_pretrain_rollout(
        self, env, n_steps: int, phase_id: Optional[int], use_gae: bool = False,
        pool: Optional["_ValuePretrainWorkers"] = None, with_mc_returns: bool = False,
        deterministic: bool = False, replay_seeds: Optional[list[int]] = None,
        collect_episode_replay_stats: bool = False,
    ) -> tuple[dict, dict]:
        """Collect ``n_steps`` of on-policy experience for value warm-up.

        Returns ``(batch, stats)`` -- ``batch`` is the GAE-processed dict (same
        shape as ``RolloutBuffer.as_tensors()``); ``stats`` has
        ``episode_returns``/``outcomes_vs_rules``/``outcomes_vs_immobile``/
        ``outcomes_vs_neural`` for the caller's return value.

        Single-process when ``ppo.value_pretrain_n_processes == 1`` (uses
        ``env`` directly, exactly the previous inline behaviour). When
        ``ppo.value_pretrain_n_processes > 1`` and ``phase_id`` is given,
        spawns that many plain one-env-per-process ``rollout_worker.py``
        workers (its own process count, DECOUPLED from the main PPO loop's
        ``ppo.n_processes``/``ppo.batched_rollout`` -- this call has no
        batched-rollout-aware path of its own, see ``value_pretrain_n_processes``'s
        own config comment for why sharing ``n_processes`` directly would
        be a regression here) -- no weight sync needed since this is called
        once per pretraining stage, not per rollout; each worker's GAE is
        computed independently before merging, for the same reason as
        ``_train_parallel()`` (concatenating raw transitions across worker
        boundaries before GAE would corrupt advantage estimates).

        Args:
            use_gae: False (default, ``pretrain_value``'s behaviour, UNCHANGED)
                uses pure Monte Carlo discounted returns
                (``compute_mc_returns``) -- correct for a BC-fresh/untrained
                value net, where bootstrapping off its own (effectively
                random) predictions would fit targets that circularly depend
                on the very net being warm-started. True (``ppg_value_refit``)
                uses ``compute_gae`` instead: for an ALREADY-reasonably-
                trained value net (ppg_value_refit's actual use case -- a
                decent checkpoint, not a cold start), bootstrapping off its
                own predictions is just ordinary TD learning, exactly what
                real PPO training already does on every rollout -- and
                matters here because real PPO's OWN value loss targets GAE
                returns, not MC returns, so fitting the same quantity during
                a refit means the value head needs no further readjustment
                once real training resumes (the actual point of PPG's
                auxiliary phase: converge toward the SAME target, not a
                different-but-correlated one). Bootstrap handling lives in
                ``_finalize_value_pretrain_result``: parallel workers (both
                kinds) return a real ``last_value``, used as-is with NO
                truncation so a trailing partial episode's data is kept;
                the single-process branch has no ``last_value`` and falls
                back to truncate-then-GAE with a literal ``0.0`` bootstrap,
                which is provably never used (truncation leaves the buffer
                ending on a ``done=1`` row and GAE's recursion resets there).
            pool: an already-spawned worker pool (from
                ``_spawn_value_pretrain_workers``) to REUSE for this call
                instead of spawning a fresh one -- the worker main loops
                (both kinds) already serve repeated "collect" commands (only
                "close"/the pipe closing ends them), so nothing on the
                worker side needs to change. When given, this call does NOT
                spawn or close any process; the CALLER owns that lifecycle
                (``ppg_value_refit`` does this for ``num_rollouts`` > 1, so
                a fresh process pool isn't paid for every cycle). Weights
                are re-synced to the workers at the start of EVERY call
                regardless, which is what makes reuse correct while the
                policy keeps moving. ``None`` (default) = spawn a pool for
                this call alone and close it before returning -- exactly
                the original per-call behaviour (``pretrain_value``,
                ``pretrain_combined``).
            replay_seeds: episode seeds to force-replay in this collection,
                chunked round-robin across worker processes exactly like
                ``_train_batched_parallel``'s own episode-seed replay (see
                ``_update_episode_replay``) -- reuses the SAME
                ``BatchedRolloutWorkerHandle.collect(replay_seeds=...)``
                wire parameter the main loop already sends. Only takes
                effect on the batched worker path (``pool.batched``); a
                non-empty list given with a plain/unbatched pool (or no
                pool at all, i.e. the single-process branch) is logged and
                otherwise ignored, since ``rollout_worker.py`` has no
                seed-injection support. None (default) = no replay, i.e.
                today's ordinary on-policy sampling.
            collect_episode_replay_stats: False (default) = today's exact
                behaviour, no extra cost. True adds
                ``stats["episode_seed_mse"]``: a ``[(seed, mean_squared_error), ...]``
                list, one entry per completed episode with a recorded seed,
                using each episode's mean ``(values - returns) ** 2`` (the
                critic's own at-collection-time prediction vs the GAE/MC
                target) -- computed via ``_episode_abs_adv_means`` (squared
                errors are already non-negative, so its internal ``abs()``
                is a no-op here). For ``ppg_value_refit``'s own rollout-to-
                rollout episode replay (``ppg_episode_replay_enabled``), the
                value-fitting analog of ``_update_episode_replay``'s
                top-|advantage| selection. Only populated on the batched
                worker path; empty list otherwise (matches ``replay_seeds``'
                own plain-pool limitation).
        """
        _owns_pool = pool is None
        if _owns_pool:
            pool = self._spawn_value_pretrain_workers(phase_id)
        if replay_seeds and pool is None:
            log.info(
                f"  [value pretrain rollout] {len(replay_seeds)} replay seed(s) given but ignored "
                "-- no parallel worker pool for this call (single-process branch has no "
                "seed-injection support)"
            )
        if pool is not None:
            import multiprocessing
            import multiprocessing.connection

            n_workers = pool.n_processes
            steps_per_worker = max(1, n_steps // n_workers)
            _kind = (
                f"batched: {n_workers} proc x {pool.envs_per_process} envs = "
                f"{n_workers * pool.envs_per_process} envs"
                if pool.batched else f"{n_workers} worker(s), 1 env each"
            )
            log.info(
                f"  [value pretrain rollout] parallel collection ({_kind}), "
                f"~{steps_per_worker} steps/process"
                + ("" if _owns_pool else " (reusing already-spawned workers)")
            )
            _wall_start = time.perf_counter()

            # Results are INGESTED AS THEY ARRIVE (each one is finalized,
            # its stats folded in, and the raw result dropped) rather than
            # collected into a list first -- the old "gather everything, then
            # convert everything, then merge" shape kept the raw results, the
            # converted per-worker tensors and the merged batch alive
            # together (the rollout-end RAM spike). Batched workers now also
            # finalize in the worker and stream whole-episode chunks, so what
            # arrives here is already compact and bounded per message.
            worker_batches: list[dict] = []
            episode_returns: list[float] = []
            episode_outcome_labels: list[str] = []
            outcomes_vs_rules: list[str] = []
            outcomes_vs_immobile: list[str] = []
            outcomes_vs_neural: list[str] = []
            episode_comp_list: list[dict[str, float]] = []
            episode_durations_s: list[float] = []
            episode_seed_mse: list[tuple[int, float]] = []
            n_dropped_total = 0
            _n_rows_total = 0

            def _ingest(r: dict) -> None:
                nonlocal n_dropped_total, _n_rows_total
                if "batch" in r:
                    # Worker-finalized (batched workers): numpy arrays on the wire
                    # (None = MC mode had no completed episode in this result).
                    from footballcoach.ai.ppo.batched_rollout_worker import batch_from_wire

                    tensors = batch_from_wire(r["batch"]) if r["batch"] is not None else None
                    n_dropped = r["n_dropped"]
                    if collect_episode_replay_stats and tensors is not None:
                        # Squared errors are already non-negative, so _episode_abs_adv_means's
                        # internal abs() is a no-op here -- this reuses it, unmodified, to get
                        # per-episode mean-squared-error instead of mean-|advantage|. .tolist()
                        # first, matching _decode_rollout_result's own convention (plain floats,
                        # not 0-dim tensors, for downstream sort/format/log use).
                        _sq_err = ((tensors["values"] - tensors["returns"]) ** 2).tolist()
                        _mse_means = _episode_abs_adv_means(
                            tensors["track_ids"], tensors["dones"].tolist(), _sq_err,
                        )
                        for _seed, _mse in zip(r["stats"].get("episode_seeds", []), _mse_means):
                            if _seed is not None:
                                episode_seed_mse.append((_seed, _mse))
                else:
                    # Legacy shape ({"buffer", ...}): plain workers, or a batched
                    # worker spoken to without a returns spec.
                    tensors, n_dropped = _finalize_value_pretrain_result(
                        r, self.gamma, self.lam, use_gae, with_mc_returns=with_mc_returns,
                    )
                n_dropped_total += n_dropped
                _n_rows_total += (int(tensors["returns"].shape[0]) if tensors is not None else 0) + n_dropped
                if tensors is not None:
                    worker_batches.append(tensors)
                stats = r["stats"]
                episode_returns.extend(stats["episode_rewards"])
                episode_outcome_labels.extend(stats["episode_outcome_labels"])
                outcomes_vs_rules.extend(stats["episode_outcomes_vs_rules"])
                outcomes_vs_immobile.extend(stats["episode_outcomes_vs_immobile"])
                outcomes_vs_neural.extend(stats["episode_outcomes_vs_neural"])
                episode_comp_list.extend(stats["episode_comp_list"])
                episode_durations_s.extend(stats["episode_durations_s"])

            try:
                dec_state = self.decision_net.state_dict()
                exec_state = self.execution_net.state_dict()
                val_state = self.value_net.state_dict() if self.value_net is not None else None
                for w in pool.handles:
                    w.set_weights(dec_state, exec_state, val_state)
                # The PARENT resets the shared counter (workers never do --
                # a worker-side reset would race between workers sharing it).
                with pool.progress_value.get_lock():
                    pool.progress_value.value = 0
                # Batched workers finalize their own results with OUR gamma/lam
                # (explicit, so a worker's separately-read config can't drift).
                _returns_spec = {
                    "mode": "gae" if use_gae else "mc", "gamma": self.gamma, "lam": self.lam,
                }
                if with_mc_returns:
                    _returns_spec["with_mc"] = True
                if deterministic and not pool.batched:
                    raise ValueError(
                        "deterministic value-pretrain rollouts need the BATCHED collection path "
                        "(ppo.value_pretrain_batched_rollout=true)"
                    )
                if replay_seeds and not pool.batched:
                    log.info(
                        f"  [value pretrain rollout] {len(replay_seeds)} replay seed(s) given but "
                        "ignored -- the plain (unbatched) worker pool has no seed-injection support "
                        "(needs ppo.value_pretrain_batched_rollout=true)"
                    )
                # Round-robin chunking, identical to _train_batched_parallel's own
                # episode-seed replay (_update_episode_replay's caller) -- each
                # worker gets every len(pool.handles)-th seed.
                _replay_chunks = (
                    [c for c in (replay_seeds[i::n_workers] for i in range(n_workers)) if c]
                    if (replay_seeds and pool.batched) else []
                )
                for _wi, w in enumerate(pool.handles):
                    if pool.batched:
                        w.collect(
                            steps_per_worker, returns=_returns_spec, deterministic=deterministic,
                            replay_seeds=(_replay_chunks[_wi] if _wi < len(_replay_chunks) else None),
                        )
                    else:
                        w.collect(steps_per_worker, progress=0.0)
                _agg_progress = ProgressReporter(
                    steps_per_worker * n_workers,
                    prefix=f"  [value pretrain rollout] ({n_workers} workers): ", live=True,
                )
                _pending = {w.conn: w for w in pool.handles}
                while _pending:
                    ready = multiprocessing.connection.wait(list(_pending.keys()), timeout=0.2)
                    _agg_progress.update(int(pool.progress_value.value))
                    for conn in ready:
                        if pool.batched:
                            # Uniform batched protocol: zero-or-more
                            # {"chunk": [per-env results]} then {"done": True}.
                            msg = conn.recv()
                            for r in msg.get("chunk", []):
                                _ingest(r)
                            if msg.get("done"):
                                _pending.pop(conn, None)
                        else:
                            _ingest(_pending.pop(conn).recv_result())
            finally:
                if _owns_pool:
                    self._close_value_pretrain_workers(pool)
            _wall_elapsed = time.perf_counter() - _wall_start
            log.info(
                f"  [value pretrain rollout] parallel total: {_wall_elapsed:.1f}s wall  "
                f"({1000.0 * _wall_elapsed / max(_n_rows_total, 1):.2f} ms/step aggregate, "
                f"{_n_rows_total / max(_wall_elapsed, 1e-9):.1f} steps/s aggregate across "
                f"{n_workers} process(es)"
                + (", includes process spawn)" if _owns_pool else ")")
            )
            if n_dropped_total:
                log.info(
                    f"  [value pretrain rollout] dropped {n_dropped_total} trailing "
                    f"(incomplete-episode) step(s) across workers before "
                    f"{'GAE' if use_gae else 'MC'}-return fit"
                )
            if not worker_batches:
                raise RuntimeError(
                    "value-pretrain rollout produced no usable rows (no episode completed in "
                    f"{n_steps} steps across {n_workers} process(es)); raise n_steps or shorten episodes"
                )
            # release_inputs: peak ~1x the rollout during the merge, not ~2x.
            batch = _merge_worker_batches(worker_batches, release_inputs=True)
            worker_batches.clear()
        else:
            env.sample_action_fn = self._sample_action
            env.reset()
            buffer = RolloutBuffer()
            episode_returns = []
            episode_outcome_labels = []
            outcomes_vs_rules = []
            outcomes_vs_immobile = []
            outcomes_vs_neural = []
            episode_accum = 0.0
            next_obs = None
            episode_comp_accum: dict[str, float] = {}
            episode_comp_list = []
            episode_durations_s = []
            episode_seed_mse = []  # single-process branch has no seed-injection support
            progress = ProgressReporter(n_steps, prefix="  [value pretrain rollout] ", live=True)

            for _step_i in range(n_steps):
                progress.update(_step_i + 1)
                next_obs, reward, done, info = env.step()
                tr = env.last_trainee_transition
                if tr is not None:
                    buffer.add(
                        obs=tr["obs"],
                        action=_action_to_numpy(tr["action"], tr["raw_exec"]),
                        log_prob=tr["log_prob"],
                        value=tr["value"],
                        reward=reward,
                        done=1.0 if done else 0.0,
                    )
                episode_accum += reward
                for _k, _v in getattr(env, "last_reward_components", {}).items():
                    episode_comp_accum[_k] = episode_comp_accum.get(_k, 0.0) + _v
                if done:
                    episode_returns.append(episode_accum)
                    episode_accum = 0.0
                    episode_outcome_labels.append(
                        info.trial_outcome if (info is not None and info.trial_outcome is not None) else "unknown"
                    )
                    if episode_comp_accum:
                        episode_comp_list.append(dict(episode_comp_accum))
                    episode_comp_accum = {}
                    if info is not None and info.trial_outcome is not None:
                        if info.is_rules_episode:
                            outcomes_vs_rules.append(info.trial_outcome)
                        elif info.is_immobile_episode:
                            outcomes_vs_immobile.append(info.trial_outcome)
                        else:
                            outcomes_vs_neural.append(info.trial_outcome)
                    if info is not None:
                        episode_durations_s.append(info.ticks_elapsed * env._dt_s)
                    env.reset()
                else:
                    obs = next_obs
            progress.finish(n_steps, n_episodes=len(episode_outcome_labels))

            # No "last_value" in this bare result dict -> the helper takes its
            # truncate-then-(MC | zero-bootstrap GAE) path, same as before.
            batch, n_dropped = _finalize_value_pretrain_result(
                {"buffer": buffer}, self.gamma, self.lam, use_gae, with_mc_returns=with_mc_returns,
            )
            if n_dropped:
                log.info(
                    f"  [value pretrain rollout] dropped {n_dropped} trailing "
                    f"(incomplete-episode) step(s) before {'GAE' if use_gae else 'MC'}-return fit"
                )

        log.info(
            f"  [value pretrain rollout] mean_return={np.mean(episode_returns) if episode_returns else float('nan'):.2f} "
            f"({len(episode_returns)} episode(s))  "
            f"vs[{_PHASE1_OUTCOME_LEGEND}]  "
            f"vs_rules({len(outcomes_vs_rules)}): {outcome_breakdown(outcomes_vs_rules)}  "
            f"vs_immobile({len(outcomes_vs_immobile)}): {outcome_breakdown(outcomes_vs_immobile)}  "
            f"vs_neural({len(outcomes_vs_neural)}): {outcome_breakdown(outcomes_vs_neural)}"
        )
        if episode_durations_s:
            _dur_arr = np.array(episode_durations_s)
            log.info(
                f"  [value pretrain rollout] ep_len {_dur_arr.mean():.1f}\u00b1{_dur_arr.std():.1f}s"
                f"  (n={len(episode_durations_s)}, min={_dur_arr.min():.1f}s, max={_dur_arr.max():.1f}s)"
            )
        log_episode_reward_stats_by_outcome(
            episode_returns, episode_outcome_labels, log_prefix="[value pretrain rollout] "
        )

        # Per-episode reward statistics table (mean/std/min/max per type),
        # identical formatting to the main PPO rollout loop's table.
        if episode_comp_list:
            _all_keys = [k for k, _ in REWARD_COMP_LABELS
                         if any(k in ep for ep in episode_comp_list)]
            if _all_keys:
                _col_w = 14
                _lbl_map = {k: lbl for k, lbl in REWARD_COMP_LABELS}
                _rew_stats_lines = [
                    f"  {'component':<{_col_w}}  {'mean':>8}  {'std':>7}  {'min':>8}  {'max':>8}",
                    "  " + "-" * _col_w + "  " + "-" * 8 + "  " + "-" * 7 + "  " + "-" * 8 + "  " + "-" * 8,
                ]
                for _k in _all_keys:
                    _vals = [ep[_k] for ep in episode_comp_list if _k in ep]
                    if not _vals:
                        continue
                    _arr = np.array(_vals)
                    # Same "never fired / always exactly zero this rollout"
                    # skip as the "rew/step" table's own "_cs['mean'] == 0.0
                    # and _cs['std'] == 0.0" check -- mean==0 and std==0
                    # together mean every episode's value for this
                    # component was exactly 0.0, so there's nothing to show.
                    # Component identity/whether it's config-disabled can
                    # change between rollouts (a coefficient retuned in
                    # ai_config.json, a curriculum phase change), so this is
                    # a per-rollout runtime check, not a permanent removal
                    # from REWARD_COMP_LABELS -- a component that starts
                    # firing again just reappears on its own next rollout,
                    # nothing to keep in sync by hand.
                    if _arr.mean() == 0.0 and _arr.std() == 0.0:
                        continue
                    _lbl = _lbl_map.get(_k, _k)
                    _rew_stats_lines.append(
                        f"  {_lbl:<{_col_w}}  {_arr.mean():>+8.3f}  {_arr.std():>7.3f}"
                        f"  {_arr.min():>+8.3f}  {_arr.max():>+8.3f}"
                    )
                log.info(
                    f"  [value pretrain rollout] rew/ep (mean/std/min/max per episode, "
                    f"{len(episode_comp_list)} ep)\n" + "\n".join(_rew_stats_lines)
                )

        return batch, {
            "episode_returns": episode_returns,
            "outcomes_vs_rules": outcomes_vs_rules,
            "outcomes_vs_immobile": outcomes_vs_immobile,
            "outcomes_vs_neural": outcomes_vs_neural,
            "episode_seed_mse": episode_seed_mse,
        }

    def pretrain_value(
        self,
        env,
        n_steps: int,
        n_epochs: int,
        lr: float,
        batch_size: Optional[int] = None,
        experiment_separate_value_net: bool = False,
        phase_id: Optional[int] = None,
    ) -> dict:
        """Warm-start the value heads to predict actual returns before PPO starts.

        Collects n_steps of experience using the current (BC-warm-started) policy,
        computes Monte Carlo returns via GAE, then trains ONLY the value loss for
        n_epochs at the given lr. This prevents the enormous value gradient from
        destroying the policy on the very first PPO update.

        Trunk/encoder freezing (via ``_get_value_pretrain_freeze_params()``) is
        always applied here — this is a distinct stage from
        ``pretrain_combined()``'s Phase 0, which deliberately has freezing
        REMOVED (see ai/knowledge.md "Phase 0" note). Do not conflate the two.

        Also called internally by ``pretrain_combined()``'s Phase 2/3 (rollout
        collection + value warm-up), which used to duplicate this logic inline.

        Args:
            env: ScenarioEnv. Ignored (may be None) when
                ``ppo.value_pretrain_n_processes > 1`` and ``phase_id`` is
                given -- rollout collection spawns its own
                ``rollout_worker.py`` subprocess workers instead (a process
                count decoupled from the main PPO loop's ``n_processes``,
                see that config key's comment for why).
            n_steps: Steps to collect (should be >= rollout_steps, e.g. 4096)
            n_epochs: Epochs to fit the value network per collected rollout
            lr: Learning rate for value pre-training (higher than PPO lr, e.g. 1e-3)
            batch_size: Minibatch size. Defaults to ``self.minibatch_size``.
            experiment_separate_value_net: EXPERIMENTAL (see Idea2.md /
                ai_trainer_knowledge.md "separate value network" discussion) —
                when True, also constructs a second, completely independent
                ``ExecutionNetwork`` (same class + config, fresh random init, NOT
                sharing any weights with ``self.execution_net``) and trains it
                fully unfrozen (no ``_get_value_pretrain_freeze_params()``
                freezing) on the identical rollout data/returns as the main
                shared-trunk value head. It still reads ``decision_heads`` from
                the real (frozen) ``self.decision_net``, so the comparison
                isolates "separate execution-net trunk for the value head" as
                the only variable — same input information, same architecture,
                same data, only the trunk-sharing differs. Logs a side-by-side
                val_rmse comparison each epoch. Purely a read-only experiment:
                the second network is discarded when this method returns (no
                checkpoint save, no effect on the real value_head or PPO).
            phase_id: Curriculum phase id. Required to use parallel rollout
                collection (``ppo.value_pretrain_n_processes > 1``) -- each
                worker rebuilds its own env from this id, same as the main PPO
                training loop. Ignored when ``ppo.value_pretrain_n_processes
                == 1`` (uses ``env`` directly).

        Returns:
            dict with diagnostic stats from the rollout collection:
            ``{"episode_returns": list[float], "outcomes_vs_rules": list[str],
            "outcomes_vs_immobile": list[str], "outcomes_vs_neural": list[str]}``
        """
        # Decoupled from ppo.minibatch_size -- see self._value_pretrain_batch_size.
        _batch_size = batch_size if batch_size is not None else self._value_pretrain_batch_size
        log.info(f"Value pre-training: {n_steps} steps, {n_epochs} epochs, lr={lr}, batch_size={_batch_size}")
        if self.separate_value_net:
            # self.value_net is fully independent of the BC-primed policy trunk --
            # nothing to freeze/protect, train the whole thing (this is the whole
            # point: a critic that never saw a BC gradient, free to organise its
            # own features purely for value prediction from step one).
            value_opt = self.value_net_optimizer
        else:
            # Freeze trunk layers so BC-learned policy weights are not corrupted.
            _freeze_params = self._get_value_pretrain_freeze_params()
            for p in _freeze_params:
                p.requires_grad_(False)
            value_opt = torch.optim.Adam(
                list(self.execution_net.value_head.parameters()),
                lr=lr, eps=1e-5, weight_decay=self._value_pretrain_weight_decay,
            )

        # --- Experimental: separate-trunk value network (see docstring) ---
        _sep_net = None
        _sep_opt = None
        if experiment_separate_value_net:
            from footballcoach.ai.models.execution_network import ExecutionNetwork
            _sep_net = CanonicalNetworkWrapper(ExecutionNetwork.from_config()).to(self.device)
            # Fully unfrozen: every parameter of this fresh network trains,
            # unlike the main value_head-only optimizer above.
            _sep_opt = torch.optim.Adam(_sep_net.parameters(), lr=lr, eps=1e-5)
            log.info(
                "  [separate value net experiment] constructed a second, independent "
                "ExecutionNetwork (fresh init, fully unfrozen) for side-by-side "
                "value-loss comparison against the shared-trunk value_head above."
            )

        batch, _rollout_stats = self._collect_value_pretrain_rollout(env, n_steps, phase_id)

        # --- Episode-level 85/15 train/val split (overfit detection) ---
        # Split by complete episodes so no episode spans both sets.
        dones_arr = batch["dones"].numpy()
        episode_end_idxs = np.where(dones_arr > 0.5)[0]
        n_complete_eps = len(episode_end_idxs)
        n_val_eps = max(1, round(0.15 * n_complete_eps)) if n_complete_eps >= 2 else 0
        n_train_eps = n_complete_eps - n_val_eps
        n_total = len(dones_arr)
        val_mask = np.zeros(n_total, dtype=bool)
        if n_val_eps > 0:
            ep_starts = np.concatenate([[0], episode_end_idxs[:-1] + 1])
            for _i in range(n_train_eps, n_complete_eps):
                val_mask[ep_starts[_i]:episode_end_idxs[_i] + 1] = True
        train_mask = ~val_mask

        # ret_std from all returns for consistent normalisation scale --
        # taken BEFORE the split below empties ``batch`` key by key.
        all_returns_t = batch["returns"].to(self.device)

        # Split WITHOUT keeping the full batch alive next to its two slices
        # (see _split_batch_releasing): peak ~1x the rollout, not ~2x.
        train_batch_raw, val_batch_raw = _split_batch_releasing(
            batch, train_mask, val_mask if n_val_eps > 0 else None,
        )

        log.info(
            f"  Value pretrain split: {n_train_eps} train eps ({int(train_mask.sum())} steps)"
            + (f"  |  {n_val_eps} val eps ({int(val_mask.sum())} steps)" if n_val_eps > 0 else "")
        )

        # Augment only the train portion (geometric flips + slot permutations).
        if self.augment_n_slot_shuffles > 0:
            train_batch = augment_batch(train_batch_raw, self.augment_n_slot_shuffles, self._aug_rng)
            del train_batch_raw  # the un-augmented copy is dead weight from here on
        else:
            train_batch = train_batch_raw

        ret_std = all_returns_t.std().clamp(min=1.0)
        log.debug(f"  [value pretrain] returns: mean={all_returns_t.mean():.2f}  std={ret_std:.2f}"
                  f"  min={all_returns_t.min():.2f}  max={all_returns_t.max():.2f}")

        returns_t = train_batch["returns"].to(self.device)

        # Pre-load val tensors to device once.
        val_returns_t = None
        val_obs_dict = None
        val_outcomes: list[str] = []
        if val_batch_raw is not None:
            val_returns_t = val_batch_raw["returns"].to(self.device)
            val_obs_dict = {k.replace("obs/", ""): val_batch_raw[k].to(self.device)
                            for k in val_batch_raw if k.startswith("obs/")}
            val_outcomes = val_batch_raw.get("step_outcomes", [])

        n = len(returns_t)
        mean_loss = float("nan")
        epochs_done = 0
        _best_val_loss = float("inf")
        _best_val_state: Optional[dict] = None
        _patience = 0
        _EARLY_STOP_PATIENCE = self._value_pretrain_early_stop_patience
        _EARLY_STOP_MIN_DELTA = self._value_pretrain_early_stop_min_delta

        # --- Baseline: loss BEFORE any gradient step, train/val separately ---
        # Same normalized-MSE forward pass (and the same self.value_net vs.
        # self.execution_net routing) as the per-epoch train/val blocks below,
        # just under no_grad with no optimizer step -- so "epoch 1"'s
        # improvement can be read against a real starting point instead of
        # assumed. Minibatched the same way as the training loop purely to
        # bound peak memory on a large (post-augmentation) train_batch.
        def _eval_loss_over(obs_source: dict, ret_source: torch.Tensor) -> tuple[float, float]:
            total_sq = 0.0
            n_rows = 0
            with torch.no_grad():
                for start in range(0, len(ret_source), _batch_size):
                    mb_ret = ret_source[start:start + _batch_size]
                    mb_obs = {k: v[start:start + _batch_size] for k, v in obs_source.items()}
                    sat, oat = _ai_types(mb_obs)
                    d_heads = self.decision_net(
                        mb_obs["self_feat"], mb_obs["other_feat"], mb_obs["exists_mask"],
                        mb_obs["ball_feat"], mb_obs["global_feat"], sat, oat,
                    )
                    _val_net = self.value_net if self.separate_value_net else self.execution_net
                    _value = _val_net(
                        mb_obs["self_feat"], mb_obs["other_feat"], mb_obs["exists_mask"],
                        mb_obs["ball_feat"], mb_obs["global_feat"], d_heads, sat, oat,
                        value_only=True,
                    )
                    preds = _value.squeeze(-1)
                    total_sq += float(((preds - mb_ret) ** 2).sum())
                    n_rows += len(mb_ret)
            mse = total_sq / max(n_rows, 1)
            norm_mse = mse / float(ret_std ** 2)
            return norm_mse, float(ret_std) * math.sqrt(norm_mse)

        _baseline_train_obs = {
            k.replace("obs/", ""): v.to(self.device)
            for k, v in train_batch.items() if k.startswith("obs/")
        }
        _baseline_train_loss, _baseline_train_rmse = _eval_loss_over(_baseline_train_obs, returns_t)
        if val_obs_dict is not None and val_returns_t is not None:
            _baseline_val_loss, _baseline_val_rmse = _eval_loss_over(val_obs_dict, val_returns_t)
            log.info(
                f"  Value epoch   0/{n_epochs} (baseline, no training yet): "
                f"train={_baseline_train_loss:.4f} rmse={_baseline_train_rmse:.2f}  "
                f"val={_baseline_val_loss:.4f} val_rmse={_baseline_val_rmse:.2f} "
                f"(std={float(ret_std):.1f})"
            )
        else:
            log.info(
                f"  Value epoch   0/{n_epochs} (baseline, no training yet): "
                f"train_loss={_baseline_train_loss:.4f}  rmse={_baseline_train_rmse:.2f} "
                f"(returns std={float(ret_std):.1f})"
            )

        ep_losses_sep: list[float] = []  # populated only when experiment_separate_value_net
        for ep in range(n_epochs):
            indices = torch.randperm(n)
            ep_losses = []
            ep_losses_sep = []
            ep_pred_means: list[float] = []
            ep_ret_means: list[float] = []
            for start in range(0, n, _batch_size):
                mb_idx = indices[start:start + _batch_size]
                mb_obs = {k.replace("obs/", ""): train_batch[k][mb_idx].to(self.device)
                          for k in train_batch if k.startswith("obs/")}
                mb_ret = returns_t[mb_idx]

                sf = mb_obs["self_feat"]
                of = mb_obs["other_feat"]
                em = mb_obs["exists_mask"]
                bf = mb_obs["ball_feat"]
                gf = mb_obs["global_feat"]
                sat, oat = _ai_types(mb_obs)

                if self.separate_value_net:
                    # decision_net stays fully frozen/detached here -- self.value_net
                    # is completely independent, no need for its gradient to reach
                    # decision_net at all (unlike the shared-trunk path below, where
                    # decision_net params are included in the optimizer/clip so its
                    # BC-pretrained-but-still-nominally-trainable params get the
                    # gradient too under the old convention).
                    with torch.no_grad():
                        d_heads = self.decision_net(sf, of, em, bf, gf, sat, oat)
                    _value = self.value_net(sf, of, em, bf, gf, d_heads, sat, oat, value_only=True)
                    new_values = _value.squeeze(-1)

                    value_loss = F.mse_loss(new_values, mb_ret) / (ret_std ** 2)

                    value_opt.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                    value_opt.step()
                    ep_losses.append(value_loss.item())
                else:
                    d_heads = self.decision_net(sf, of, em, bf, gf, sat, oat)
                    _value = self.execution_net(sf, of, em, bf, gf, d_heads, sat, oat, value_only=True)
                    new_values = _value.squeeze(-1)  # single value head (execution_net)

                    # Normalised MSE so the loss is O(1) regardless of return scale
                    value_loss = F.mse_loss(new_values, mb_ret) / (ret_std ** 2)

                    value_opt.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.decision_net.parameters()) + list(self.execution_net.parameters()),
                        self.max_grad_norm,
                    )
                    value_opt.step()
                    ep_losses.append(value_loss.item())

                ep_pred_means.append(new_values.detach().mean().item())
                ep_ret_means.append(mb_ret.mean().item())

                # --- Experimental separate-trunk value net: identical mb_obs/mb_ret,
                # identical decision-head VALUES (from the same frozen decision_net) —
                # only the execution-net trunk/encoders/value_head differ (fresh,
                # unfrozen, zero weight sharing with self.execution_net). d_heads is
                # detached here so this experimental path is totally independent: no
                # gradient flows back into decision_net, and it does not try to reuse
                # the main value path's autograd graph (already freed by the
                # .backward() call above).
                if _sep_net is not None:
                    d_heads_detached = _detach_decision_heads(d_heads)
                    _value_sep2 = _sep_net(sf, of, em, bf, gf, d_heads_detached, sat, oat, value_only=True)
                    new_values_sep = _value_sep2.squeeze(-1)
                    value_loss_sep = F.mse_loss(new_values_sep, mb_ret) / (ret_std ** 2)

                    _sep_opt.zero_grad()
                    value_loss_sep.backward()
                    nn.utils.clip_grad_norm_(_sep_net.parameters(), self.max_grad_norm)
                    _sep_opt.step()
                    ep_losses_sep.append(value_loss_sep.item())

            mean_loss = float(np.mean(ep_losses))
            epochs_done = ep + 1
            _train_rmse = float(ret_std) * math.sqrt(mean_loss)
            _train_pred_mean = float(np.mean(ep_pred_means)) if ep_pred_means else float("nan")
            _train_ret_mean = float(np.mean(ep_ret_means)) if ep_ret_means else float("nan")
            _mean_loss_sep = float(np.mean(ep_losses_sep)) if ep_losses_sep else float("nan")
            _train_rmse_sep = float(ret_std) * math.sqrt(_mean_loss_sep) if ep_losses_sep else float("nan")

            _vl_sep = float("nan")
            _val_rmse_sep = float("nan")
            if val_obs_dict is not None and val_returns_t is not None:
                with torch.no_grad():
                    _sat_v, _oat_v = _ai_types(val_obs_dict)
                    d_v = self.decision_net(
                        val_obs_dict["self_feat"], val_obs_dict["other_feat"],
                        val_obs_dict["exists_mask"], val_obs_dict["ball_feat"],
                        val_obs_dict["global_feat"], _sat_v, _oat_v,
                    )
                    _val_net = self.value_net if self.separate_value_net else self.execution_net
                    _value_v = _val_net(
                        val_obs_dict["self_feat"], val_obs_dict["other_feat"],
                        val_obs_dict["exists_mask"], val_obs_dict["ball_feat"],
                        val_obs_dict["global_feat"], d_v, _sat_v, _oat_v,
                        value_only=True,
                    )
                    _val_preds = _value_v.squeeze(-1)
                    _vl = float(F.mse_loss(_val_preds, val_returns_t) / (ret_std ** 2))
                    _val_pred_mean = float(_val_preds.mean().item())
                    _val_ret_mean = float(val_returns_t.mean().item())
                    if _sep_net is not None:
                        _value_v_sep = _sep_net(
                            val_obs_dict["self_feat"], val_obs_dict["other_feat"],
                            val_obs_dict["exists_mask"], val_obs_dict["ball_feat"],
                            val_obs_dict["global_feat"], d_v, _sat_v, _oat_v,
                            value_only=True,
                        )
                        _vl_sep = float(F.mse_loss(
                            _value_v_sep.squeeze(-1), val_returns_t
                        ) / (ret_std ** 2))
                        _val_rmse_sep = float(ret_std) * math.sqrt(_vl_sep)
                _val_rmse = float(ret_std) * math.sqrt(_vl)
                log.info(
                    f"  Value epoch {epochs_done}/{n_epochs}: "
                    f"train={mean_loss:.4f} rmse={_train_rmse:.2f}  "
                    f"val={_vl:.4f} val_rmse={_val_rmse:.2f} "
                    f"(std={float(ret_std):.1f})"
                    f"\n    V(train)={_train_pred_mean:+.3f}  R(train)={_train_ret_mean:+.3f}"
                    f"  |  V(val)={_val_pred_mean:+.3f}  R(val)={_val_ret_mean:+.3f}"
                    + (
                        f"\n    [separate-trunk value net] train={_mean_loss_sep:.4f} rmse={_train_rmse_sep:.2f}  "
                        f"val={_vl_sep:.4f} val_rmse={_val_rmse_sep:.2f}"
                        f"  (shared-trunk val_rmse={_val_rmse:.2f} \u2014 "
                        f"{'separate is BETTER' if _val_rmse_sep < _val_rmse else 'shared is better or equal'})"
                        if _sep_net is not None else ""
                    )
                )
                if val_outcomes:
                    _val_by_outc = value_mse_by_outcome(_val_preds, val_returns_t, val_outcomes)
                    if _val_by_outc:
                        log.info(f"    val_rmse by outcome: {format_outcome_rmse_breakdown(_val_by_outc)}")
                if _vl < _best_val_loss - _EARLY_STOP_MIN_DELTA:
                    _best_val_loss = _vl
                    _patience = 0
                    if self.separate_value_net:
                        _best_val_state = copy.deepcopy(self.value_net.state_dict())
                    else:
                        _best_val_state = copy.deepcopy(self.execution_net.value_head.state_dict())
                else:
                    _patience += 1
                    if _patience >= _EARLY_STOP_PATIENCE:
                        log.info(
                            f"  [value pretrain] early stop at epoch {epochs_done} "
                            f"(val stagnant for {_EARLY_STOP_PATIENCE} epochs, best={_best_val_loss:.4f})"
                        )
                        break
            else:
                log.info(
                    f"  Value epoch {epochs_done}/{n_epochs}: "
                    f"train_loss={mean_loss:.4f}  rmse={_train_rmse:.2f} "
                    f"(returns std={float(ret_std):.1f})"
                    f"\n    V(train)={_train_pred_mean:+.3f}  R(train)={_train_ret_mean:+.3f}"
                    + (
                        f"\n    [separate-trunk value net] train_loss={_mean_loss_sep:.4f}  rmse={_train_rmse_sep:.2f}"
                        if _sep_net is not None else ""
                    )
                )
        if _best_val_state is not None:
            if self.separate_value_net:
                self.value_net.load_state_dict(_best_val_state)
            else:
                self.execution_net.value_head.load_state_dict(_best_val_state)
            log.info(f"  [value pretrain] restored best-val weights (val_loss={_best_val_loss:.4f})")
        log.info(f"Value pre-training done ({epochs_done} epoch(s), final train_loss={mean_loss:.4f})")
        if _sep_net is not None:
            log.info(
                f"  [separate value net experiment] final train_loss={_mean_loss_sep:.4f}"
                f"  (compare against shared-trunk final train_loss={mean_loss:.4f} above)"
            )
        if not self.separate_value_net:
            for p in _freeze_params:
                p.requires_grad_(True)

        return _rollout_stats

    def _bootstrap_last_values(self, env, next_obs, buffer: "RolloutBuffer") -> dict[str, float]:
        """Per-track bootstrap values for ``RolloutBuffer.compute_gae()``.

        The trainee's own next-state value comes from ``next_obs`` (already
        available at the caller's rollout-window boundary — the return value
        of the most recent ``env.step()``). For every OTHER track actually
        present in the buffer (i.e. a secondary player whose transitions
        were recorded this rollout — only possible when that player is
        neural-controlled), re-encode that player's own current observation
        via ``env._get_obs(player_id=...)`` and run it through the same
        critic. Without this, ``compute_gae`` would silently fall back to
        0.0 for that track's final-step bootstrap — see ``compute_gae``'s
        docstring for why that's only an approximation, not a crash.
        """
        with torch.no_grad():
            last_obs_dict = {
                k: v.unsqueeze(0).to(self.device) for k, v in next_obs.to_torch_dict().items()
            }
            last_values = {"trainee": self._get_value(last_obs_dict)}
            for track in set(buffer.track_ids):
                if track == "trainee" or track in last_values:
                    continue
                sec_obs = env._get_obs(player_id=track)
                sec_obs_dict = {
                    k: v.unsqueeze(0).to(self.device) for k, v in sec_obs.to_torch_dict().items()
                }
                last_values[track] = self._get_value(sec_obs_dict)
        return last_values
