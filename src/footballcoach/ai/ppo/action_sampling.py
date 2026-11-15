"""PPOTrainer's action sampling, log-probabilities, entropy and value lookups."""
from __future__ import annotations

from typing import Optional
import numpy as np
import torch
from footballcoach.ai.action.distributions import IndependentBernoulli, KickDirectionHead, MaskedCategorical, SquashedNormalHead, VonMisesDirectionHead, _von_mises_entropy
from footballcoach.ai.action.schema import DecisionAction
from footballcoach.ai.models.decision_network import derive_get_possession_prob
from footballcoach.ai.obs.canonical import mirror_x, x_sign_of


# _LP_HEAD_NAMES (decision_net attribute names, as returned by
# PPOTrainer._ppo_lp_masked_heads) -> their short HEAD_LP_KEYS-style name
# (rollout_buffer.py) -- the per-head diagnostic lines (entropy breakdown,
# [per-head KL], [per-head policy_loss (counterfactual...)], [policy heads])
# all key on the short form, so masked-head filtering needs this mapping.
_LP_HEAD_NAME_TO_SHORT: dict[str, str] = {
    "shoot_logit": "shoot", "pass_logit": "pass_", "move_logit": "move",
    "tackle_logit": "tackle", "get_possession_raw": "gp_extra",
    "mark_logit": "mark", "hold_position_logit": "hold",
}


class ActionSamplingMixin:
    """Action sampling and log-prob/entropy maths for PPOTrainer. Mixed into PPOTrainer; relies on its attributes."""

    def _value_heads(self, sf, of, em, bf, gf, d_heads, sat, oat) -> "torch.Tensor":
        """Return THE critic value estimate (batch, 1) -- and ONLY that,
        every caller of this method must never need anything else.

        When ``separate_value_net`` is disabled (default), this is just
        ``self.execution_net(...)`` (the normal shared-trunk forward pass,
        already computed by the caller in nearly every call site — this
        method exists so call sites that only need value can go through
        one path). When enabled, forwards through the dedicated
        ``self.value_net`` instead — a fully independent ExecutionNetwork
        with its own trunk/encoders, never touched by BC losses. Both take
        identical inputs (same decision_heads too).

        Always calls with ``value_only=True`` (see ExecutionNetwork.forward's
        own docstring) -- this returns the raw value tensor directly, NOT an
        ExecutionHeadsRaw, so a caller that (wrongly) expected e.g.
        ``.move_direction`` off the result fails immediately and loudly
        rather than silently reading a stale/placeholder field.
        """
        if self.separate_value_net:
            return self.value_net(sf, of, em, bf, gf, d_heads, sat, oat, value_only=True)
        return self.execution_net(sf, of, em, bf, gf, d_heads, sat, oat, value_only=True)

    def _inactive_head_lp_keys(self) -> frozenset[str]:
        """HEAD_LP_KEYS-style short names (rollout_buffer.py) of every head
        that's currently forced to ratio=1 / zero entropy everywhere --
        curriculum-frozen decision heads (_ppo_lp_masked_heads, translated
        via _LP_HEAD_NAME_TO_SHORT) plus kick_spin when permanently frozen
        (_kick_spin_frozen, a separate flag from the curriculum mask).
        These heads' per-head diagnostic values are either an exact 0.0000
        (entropy, KL -- both literally zero when old==new==0) or a
        content-free "-mean(this slice's advantages)" residual (the
        counterfactual policy_loss breakdown, since ratio=1 collapses the
        clipped surrogate to -adv regardless of the head) -- never a real
        gradient/training signal either way, so every per-head diagnostic
        line drops these entirely rather than clutter the output with
        guaranteed-uninformative entries."""
        keys = {_LP_HEAD_NAME_TO_SHORT[name] for name in self._ppo_lp_masked_heads}
        if self._kick_spin_frozen:
            keys.add("kick_spin")
        return frozenset(keys)

    # -----------------------------------------------------------------------
    # Policy sampling
    # -----------------------------------------------------------------------

    def _move_dir_head(self, raw_vec: torch.Tensor, log_kappa_param: torch.Tensor) -> "VonMisesDirectionHead":
        return VonMisesDirectionHead(raw_vec, log_kappa_param,
                             log_kappa_min=self.move_dir_log_kappa_min,
                             log_kappa_max=self.move_dir_log_kappa_max)

    def _kick_dir_head(self, raw_vec: torch.Tensor, log_kappa_param: torch.Tensor,
                        log_std_z_param: torch.Tensor) -> "KickDirectionHead":
        return KickDirectionHead(raw_vec, log_kappa_param, log_std_z_param,
                             log_kappa_min=self.kick_dir_log_kappa_min,
                             log_kappa_max=self.kick_dir_log_kappa_max,
                             log_std_z_min=self.kick_dir_z_log_std_min,
                             log_std_z_max=self.kick_dir_z_log_std_max)

    def _kick_power_head(self, raw_mean: torch.Tensor, log_std_param: torch.Tensor) -> "SquashedNormalHead":
        """kick_power: sigmoid-squashed scalar in [0,1] (power_fraction).

        Was previously applied fully deterministically (sigmoid(mean), no
        sampling/log_prob/entropy) -- see agent_plans/spin_implementation_plan.md
        section 6. SquashedNormalHead internally clamps its own log_std to
        (-5.0, 2.0), so no extra config bounds are needed here (unlike the
        VonMisesDirectionHead/KickDirectionHead heads above, which take
        externally-configured clamp bounds because their log_kappa/log_std
        also feed an explicit restoring-force regularizer this scoped fix
        does not add for kick_power/kick_spin).
        """
        # low = ppo.kick_power_floor: the executed power fraction is
        # floor + (1 - floor) * sigmoid(raw). Only to_physical/mode_physical see
        # it; log_prob/entropy/KL are on the raw draw, so PPO's maths is unchanged.
        return SquashedNormalHead(raw_mean, log_std_param, low=self.kick_power_floor, high=1.0, squash="sigmoid")

    def _kick_spin_dist(self, raw_mean: torch.Tensor, log_std_param: torch.Tensor) -> torch.distributions.Normal:
        """kick_spin: plain unsquashed 3D Normal, no L2-normalize.

        kick_spin has no bounded physical range wired up (it's still disabled
        at the apply_nn_action.py chokepoint -- real spin physics/clamping is
        deferred, see the plan doc). Neither VonMisesDirectionHead/
        KickDirectionHead (which force a unit-vector mean) nor SquashedNormalHead (which forces a squashed
        [low,high] range) fits an unbounded raw vector, so this uses
        torch.distributions.Normal directly -- same log_std clamp convention
        as SquashedNormalHead (-5.0, 2.0) for consistency.
        """
        std = torch.exp(log_std_param.clamp(-5.0, 2.0))
        return torch.distributions.Normal(raw_mean, std)

    def _per_head_new_log_probs(self, d_heads, e_heads, mb_actions: dict, exists_mask) -> torch.Tensor:
        """Per-sample, per-head log_prob under the CURRENT policy for stored
        actions, stacked in ``HEAD_LP_KEYS`` order (shape ``(batch, 15)``).

        Mirrors the gating in ``_recompute_log_prob`` (sprint/move_dir gated
        by exec_move; kick_dir/kick_power/kick_spin gated by kick) so summing
        this over heads
        matches the scalar log_prob used for the PPO ratio/KL. Used for the
        per-head KL diagnostic: ``batch["head_log_probs"][mb_idx] - this``,
        averaged over the batch dim, gives a per-head KL breakdown instead
        of just the scalar total.
        """
        def _b(logit, key):
            return IndependentBernoulli(logit).log_prob(mb_actions[key]).squeeze(-1)

        log_kappa_move = self.execution_net.move_dir_log_kappa.to(self.device)
        log_kappa_kick = self.execution_net.kick_dir_log_kappa.to(self.device)
        log_std_z_kick = self.execution_net.kick_dir_z_log_std.to(self.device)
        log_std_power = self.execution_net.kick_power_log_std.to(self.device)
        log_std_spin = self.execution_net.kick_spin_log_std.to(self.device)
        exec_move_mask = (mb_actions["exec_move"].squeeze(-1) > 0.5).float()
        kick_mask = (mb_actions["kick"].squeeze(-1) > 0.5).float()

        lp_move_dir = exec_move_mask * (
            self._move_dir_head(e_heads.move_direction, log_kappa_move).log_prob(
                mb_actions["move_dir_raw"]
            )
        )
        lp_kick_dir = kick_mask * (
            self._kick_dir_head(e_heads.kick_direction, log_kappa_kick, log_std_z_kick).log_prob(
                mb_actions["kick_dir_raw"]
            )
        )
        lp_kick_power = kick_mask * (
            self._kick_power_head(e_heads.kick_power, log_std_power).log_prob(
                mb_actions["kick_power_raw"]
            )
        )
        # kick_spin is permanently frozen (see agent_plans/spin_implementation_plan.md
        # section 0) -- masked to exactly zero here rather than computed and
        # discarded, since freezing requires_grad alone does NOT stop its
        # trunk-drift-induced log_prob contribution from drifting (see the
        # plan doc's section 0.2 for why).
        lp_kick_spin = torch.zeros(exists_mask.shape[0], device=self.device) if self._kick_spin_frozen else kick_mask * (
            self._kick_spin_dist(e_heads.kick_spin, log_std_spin).log_prob(
                mb_actions["kick_spin_raw"]
            ).sum(dim=-1)
        )
        lp_sprint = exec_move_mask * _b(e_heads.sprint_logit, "sprint")

        masked = self._ppo_lp_masked_heads
        _zero = torch.zeros(exists_mask.shape[0], device=self.device)
        return torch.stack([
            _zero if "shoot_logit" in masked else _b(d_heads.shoot_logit, "shoot"),
            _zero if "pass_logit" in masked else _b(d_heads.pass_logit, "pass_"),
            _zero if "move_logit" in masked else _b(d_heads.move_logit, "move"),
            _zero if "tackle_logit" in masked else _b(d_heads.tackle_logit, "tackle"),
            _zero if "get_possession_raw" in masked else _b(d_heads.get_possession_raw, "get_possession_extra"),
            _zero if "mark_logit" in masked else _b(d_heads.mark_logit, "mark"),
            _zero if "hold_position_logit" in masked else _b(d_heads.hold_position_logit, "hold_position"),
            _b(e_heads.exec_move_logit, "exec_move"),
            lp_sprint,
            _b(e_heads.kick_logit, "kick"),
            _b(e_heads.tackle_attempt_logit, "tackle_attempt"),
            lp_move_dir,
            lp_kick_dir,
            lp_kick_power,
            lp_kick_spin,
        ], dim=-1)

    @torch.no_grad()
    def _sample_action_networks(self, obs_dict_batch: dict) -> tuple:
        """Batched network-forward part of action sampling: decision_net +
        execution_net + value forward passes, run ONCE for however many rows
        ``obs_dict_batch`` holds (batch dim already present -- callers that
        want a single row pass a batch of 1). This is the expensive,
        batchable part -- profiled this session at ~88% of rollout wall
        time, of which only ~11% is actual matmul compute (the rest is
        Python/PyTorch dispatch overhead on tiny batch-of-1 calls); a direct
        benchmark measured 6 separate batch=1 CPU calls at 51.2s vs 1 batch=6
        CPU call at 10.5s for this exact 3-network combination (~4.88x).
        The sampling/log_prob/decanonicalization tail is
        ``_sample_action_from_heads_batch``; both ``_sample_action`` (a
        batch of 1) and ``_sample_action_batch`` go through it, so production
        has exactly one implementation of that math.

        Returns ``(d_heads, e_heads, value_batch, x_sign_batch, em)`` -- all
        still batch-shaped ``(K, ...)``. ``em`` (exists_mask) is returned too
        since the sampling tail needs it again for the masked
        categorical target distributions (pass/tackle/mark target) -- it's
        part of the original observation, not something the network forward
        pass produces.
        """
        dev = self.device
        sf = obs_dict_batch["self_feat"].to(dev)
        of = obs_dict_batch["other_feat"].to(dev)
        em = obs_dict_batch["exists_mask"].to(dev)
        bf = obs_dict_batch["ball_feat"].to(dev)
        gf = obs_dict_batch["global_feat"].to(dev)
        sat = obs_dict_batch["self_ai_type"].to(dev) if "self_ai_type" in obs_dict_batch else None
        oat = obs_dict_batch["other_ai_type"].to(dev) if "other_ai_type" in obs_dict_batch else None

        # Canonical AI frame: mirror world-frame obs so self always attacks
        # +x (see ai/obs/canonical.py). Reused by the sampling tail
        # to decanonicalize move_direction/
        # kick_direction before they're returned to the caller.
        # (decision_net/execution_net wrap-canonicalize sf/of/bf automatically —
        # see CanonicalNetworkWrapper — so only x_sign itself is needed here.)
        # x_sign_of already returns a (K,) tensor for batched input -- see its
        # own docstring (ai/obs/canonical.py) -- no change needed here for
        # batch>1.
        x_sign_batch = x_sign_of(sf)

        d_heads = self.decision_net(sf, of, em, bf, gf, sat, oat)
        e_heads = self.execution_net(sf, of, em, bf, gf, d_heads, sat, oat)

        # Single value head: execution_net only (decision_net.value_head is
        # frozen — see __init__ note), OR self.value_net when
        # separate_value_net is enabled (see _value_heads() docstring).
        if self.separate_value_net:
            with torch.no_grad():
                value_batch = self.value_net(sf, of, em, bf, gf, d_heads, sat, oat, value_only=True).squeeze(-1)
        else:
            value_batch = e_heads.value.squeeze(-1)

        return d_heads, e_heads, value_batch, x_sign_batch, em

    @torch.no_grad()
    def _sample_action_from_heads(
        self,
        d_heads,
        e_heads,
        value: float,
        x_sign: float,
        em,
        deterministic: bool = False,
        deterministic_decision: bool = False,
        deterministic_direction: bool = False,
    ) -> tuple:
        """TEST REFERENCE ONLY -- not called by any production code path.

        Per-row sampling + log_prob + decanonicalization for ONE row of
        already-computed batch-of-1 network output. This is the original
        scalar implementation (``_sample_action``'s body before batching),
        kept as an independent oracle: production uses
        ``_sample_action_from_heads_batch`` for every call (``_sample_action``
        is a batch of 1 through it), and
        tests/ai_unit/test_sample_action_from_heads_batch.py checks that
        vectorized tail row by row against this function. Its gating is
        written as Python ``if``s and ``_compute_log_prob``, independently of
        the tensor masks in ``_recompute_log_prob``/``_per_head_new_log_probs``,
        which is what makes it a useful cross-check. Do not delete it as a
        "duplicate" without moving that check elsewhere. The comparison runs
        with ``deterministic=True``: stochastic draws differ by construction
        because a batched ``.sample()`` consumes the RNG in a different order
        than K per-row calls.

        Args:
            deterministic: if True, use each head's mode/mean instead of a
                stochastic sample for EVERY head (Bernoulli -> >=0.5,
                Categorical -> argmax, Normal/Direction heads -> mean) —
                shorthand for deterministic_decision=deterministic_direction=True.
            deterministic_decision: if True, only the discrete decision/execution
                heads (Bernoulli intents incl. exec_move/sprint/kick/tackle_attempt,
                and the pass/tackle/mark Categorical targets) use their mode;
                move/kick direction still sample.
            deterministic_direction: if True, only the continuous move_direction/
                kick_direction heads use their mean; discrete heads still sample.
            These are for evaluating a checkpoint without (all or part of) PPO
            exploration noise. Never used during rollout collection/training,
            only via load_for_inference() callers.

        Returns:
            (decision_action, log_prob, value, decision_probs,
             execution_physical, target_slots, raw_exec_samples,
             head_log_probs)

            There is no decanonicalized move-region entry: the old
            ``decision_physical`` dict (world-frame move_region_center_m,
            size_m, arrival_speed_mps) was removed on 2026-11-15 because
            ``apply_action_to_player`` never read it. The raw
            move_region_* values are still in ``decision_action``.
        """
        det_decision = deterministic or deterministic_decision
        det_direction = deterministic or deterministic_direction

        # Sample from each decision head
        shoot_dist = IndependentBernoulli(d_heads.shoot_logit)
        pass_dist = IndependentBernoulli(d_heads.pass_logit)
        move_dist = IndependentBernoulli(d_heads.move_logit)
        tackle_dist = IndependentBernoulli(d_heads.tackle_logit)
        gp_extra_dist = IndependentBernoulli(d_heads.get_possession_raw)
        mark_dist = IndependentBernoulli(d_heads.mark_logit)
        hold_dist = IndependentBernoulli(d_heads.hold_position_logit)
        shoot = shoot_dist.mode() if det_decision else shoot_dist.sample()
        pass_ = pass_dist.mode() if det_decision else pass_dist.sample()
        move = move_dist.mode() if det_decision else move_dist.sample()
        tackle = tackle_dist.mode() if det_decision else tackle_dist.sample()
        gp_extra = gp_extra_dist.mode() if det_decision else gp_extra_dist.sample()
        mark = mark_dist.mode() if det_decision else mark_dist.sample()
        hold = hold_dist.mode() if det_decision else hold_dist.sample()

        # Categorical targets (masked)
        pass_tgt_dist = MaskedCategorical(d_heads.pass_target_logits, em)
        tackle_tgt_dist = MaskedCategorical(d_heads.tackle_target_logits, em)
        mark_tgt_dist = MaskedCategorical(d_heads.mark_target_logits, em)
        pass_tgt = pass_tgt_dist.mode() if det_decision else pass_tgt_dist.sample()
        tackle_tgt = tackle_tgt_dist.mode() if det_decision else tackle_tgt_dist.sample()
        mark_tgt = mark_tgt_dist.mode() if det_decision else mark_tgt_dist.sample()

        # Continuous decision heads (pre-squash raw samples for PPO)
        mv_center_raw = d_heads.move_region_center  # (1, 2), no extra noise for now - use mean
        mv_size_raw = d_heads.move_region_size
        mv_speed_raw = d_heads.move_arrival_speed
        ad_raw = d_heads.attack_defence_raw

        # Decision probs for gating
        tackle_prob, gp_prob = derive_get_possession_prob(
            d_heads.tackle_logit, d_heads.get_possession_raw
        )
        decision_probs = {
            "shoot": float(torch.sigmoid(d_heads.shoot_logit)),
            "pass_": float(torch.sigmoid(d_heads.pass_logit)),
            "move": float(torch.sigmoid(d_heads.move_logit)),
            "tackle": float(tackle_prob),
            "get_possession": float(gp_prob),
            "mark": float(torch.sigmoid(d_heads.mark_logit)),
            "hold_position": float(torch.sigmoid(d_heads.hold_position_logit)),
        }
        target_slots = {
            "pass_": int(pass_tgt),
            "tackle": int(tackle_tgt),
            "mark": int(mark_tgt),
        }

        # Sample execution heads
        exec_move_dist = IndependentBernoulli(e_heads.exec_move_logit)
        sprint_dist = IndependentBernoulli(e_heads.sprint_logit)
        kick_dist = IndependentBernoulli(e_heads.kick_logit)
        tackle_attempt_dist = IndependentBernoulli(e_heads.tackle_attempt_logit)
        exec_move = exec_move_dist.mode() if det_decision else exec_move_dist.sample()
        sprint = sprint_dist.mode() if det_decision else sprint_dist.sample()
        kick = kick_dist.mode() if det_decision else kick_dist.sample()
        tackle_attempt = tackle_attempt_dist.mode() if det_decision else tackle_attempt_dist.sample()

        # Direction heads: move_dir is a von Mises (azimuthal-only); kick_dir
        # is a von Mises (azimuthal) x plain Normal (elevation) composite --
        # see VonMisesDirectionHead/KickDirectionHead in ai/action/distributions.py.
        # We store the noisy raw sample (not the mean) so that log_prob ratios
        # during the PPO update are meaningful — new_mean vs stored sample.
        # In deterministic (direction) mode we use the (normalized) mean direction instead.
        eps = 1e-6
        log_kappa_move = self.execution_net.move_dir_log_kappa
        log_kappa_kick = self.execution_net.kick_dir_log_kappa
        log_std_z_kick = self.execution_net.kick_dir_z_log_std
        move_dir_head = self._move_dir_head(e_heads.move_direction, log_kappa_move)
        kick_dir_head = self._kick_dir_head(e_heads.kick_direction, log_kappa_kick, log_std_z_kick)
        if det_direction:
            move_dir_raw = move_dir_head.mode_physical()  # (1, 2)
            kick_dir_raw = kick_dir_head.mode_physical()   # (1, 3)
        else:
            move_dir_raw = move_dir_head.sample_raw()  # (1, 2)
            kick_dir_raw = kick_dir_head.sample_raw()   # (1, 3)
        move_dir_phys = (move_dir_raw / (move_dir_raw.norm(dim=-1, keepdim=True) + eps)).squeeze(0)
        kick_dir_phys = (kick_dir_raw / (kick_dir_raw.norm(dim=-1, keepdim=True) + eps)).squeeze(0)

        # kick_power/kick_spin: previously applied fully deterministically
        # (sigmoid(mean) / raw mean, no sampling/log_prob/entropy) -- see
        # agent_plans/spin_implementation_plan.md section 6. Now sampled the
        # same way as move_dir/kick_dir above (mode in deterministic-direction
        # mode, else a real stochastic sample), with the raw (pre-squash)
        # sample stored for PPO log_prob recomputation at update time.
        log_std_power = self.execution_net.kick_power_log_std
        log_std_spin = self.execution_net.kick_spin_log_std
        kick_power_head = self._kick_power_head(e_heads.kick_power, log_std_power)
        kick_spin_dist = self._kick_spin_dist(e_heads.kick_spin, log_std_spin)
        if det_direction:
            kick_power_raw = kick_power_head.dist.mean  # (1, 1), pre-squash
            kick_spin_raw = kick_spin_dist.mean.squeeze(0)  # (3,)
        else:
            kick_power_raw = kick_power_head.sample_raw()  # (1, 1), pre-squash
            kick_spin_raw = kick_spin_dist.rsample().squeeze(0)  # (3,)
        kick_power_phys = float(kick_power_head.to_physical(kick_power_raw))

        # Decanonicalize: these are world-frame physical directions from here on.
        move_dir_world = mirror_x(move_dir_phys, x_sign)
        kick_dir_world = mirror_x(kick_dir_phys, x_sign)

        execution_physical = {
            "exec_move": bool(exec_move.item() > 0.5),
            "move_direction": move_dir_world.cpu().numpy(),
            "sprint": bool(sprint.item() > 0.5),
            "kick_this_tick": bool(kick.item() > 0.5),
            "kick_direction": kick_dir_world.cpu().numpy(),
            "kick_power_fraction": kick_power_phys,
            "kick_spin": kick_spin_raw.cpu().numpy(),
            "tackle_attempt": bool(tackle_attempt.item() > 0.5),
        }

        # Combined log_prob
        # (value was already computed by _sample_action_networks and passed
        # in as a plain float -- see that method's docstring.)
        log_prob = self._compute_log_prob(d_heads, e_heads, {
            "shoot": shoot, "pass_": pass_, "move": move,
            "tackle": tackle, "gp_extra": gp_extra, "mark": mark, "hold": hold,
            "pass_tgt": pass_tgt, "tackle_tgt": tackle_tgt, "mark_tgt": mark_tgt,
            "exec_move": exec_move, "sprint": sprint, "kick": kick,
            "tackle_attempt": tackle_attempt,
            "move_dir_raw": move_dir_raw, "kick_dir_raw": kick_dir_raw,
            "kick_power_raw": kick_power_raw, "kick_spin_raw": kick_spin_raw.unsqueeze(0),
        }, em)

        # Per-head log_probs for DEBUG KL breakdown (stored alongside total in buffer)
        _lsm = self.execution_net.move_dir_log_kappa
        _lsk = self.execution_net.kick_dir_log_kappa
        _lskz = self.execution_net.kick_dir_z_log_std
        # Debug per-head log_probs: apply same masking as _compute_log_prob
        # so these values match what went into the stored total log_prob.
        # Frozen/masked decision heads (self._ppo_lp_masked_heads, set via
        # set_frozen_heads() for the current curriculum phase) are zeroed
        # here too -- _per_head_new_log_probs() (used at update time to
        # recompute the "after" side of the per-head KL diagnostic) already
        # zeroes these same heads, so without masking here the stored
        # "before" side was nonzero while "after" was zero, producing a
        # spurious per-head KL (= the raw sampled log_prob, not a real KL)
        # for every frozen head -- see ai_trainer_knowledge.md "per-head KL
        # masking" note.
        _exec_move_active = float(exec_move) > 0.5
        _kick_active = float(kick) > 0.5
        _masked = self._ppo_lp_masked_heads
        head_log_probs = np.array([
            0.0 if "shoot_logit" in _masked else float(IndependentBernoulli(d_heads.shoot_logit).log_prob(shoot).sum()),
            0.0 if "pass_logit" in _masked else float(IndependentBernoulli(d_heads.pass_logit).log_prob(pass_).sum()),
            0.0 if "move_logit" in _masked else float(IndependentBernoulli(d_heads.move_logit).log_prob(move).sum()),
            0.0 if "tackle_logit" in _masked else float(IndependentBernoulli(d_heads.tackle_logit).log_prob(tackle).sum()),
            0.0 if "get_possession_raw" in _masked else float(IndependentBernoulli(d_heads.get_possession_raw).log_prob(gp_extra).sum()),
            0.0 if "mark_logit" in _masked else float(IndependentBernoulli(d_heads.mark_logit).log_prob(mark).sum()),
            0.0 if "hold_position_logit" in _masked else float(IndependentBernoulli(d_heads.hold_position_logit).log_prob(hold).sum()),
            float(IndependentBernoulli(e_heads.exec_move_logit).log_prob(exec_move).sum()),
            # sprint: only when exec_move=True
            float(IndependentBernoulli(e_heads.sprint_logit).log_prob(sprint).sum()) if _exec_move_active else 0.0,
            float(IndependentBernoulli(e_heads.kick_logit).log_prob(kick).sum()),
            float(IndependentBernoulli(e_heads.tackle_attempt_logit).log_prob(tackle_attempt).sum()),
            # move_dir: only when exec_move=True
            float(self._move_dir_head(e_heads.move_direction, _lsm).log_prob(move_dir_raw)) if _exec_move_active else 0.0,
            # kick_dir: only when kick=True
            float(self._kick_dir_head(e_heads.kick_direction, _lsk, _lskz).log_prob(kick_dir_raw)) if _kick_active else 0.0,
            # kick_power: only when kick=True (same gating as kick_dir --
            # only ever used downstream inside the `if kick_this_tick:` block)
            float(kick_power_head.log_prob(kick_power_raw)) if _kick_active else 0.0,
            # kick_spin: permanently frozen (see spin_implementation_plan.md section 0) --
            # masked to exactly 0.0 rather than computed and discarded.
            0.0 if self._kick_spin_frozen else (
                float(kick_spin_dist.log_prob(kick_spin_raw.unsqueeze(0)).sum(dim=-1)) if _kick_active else 0.0
            ),
        ], dtype=np.float32)

        # Build DecisionAction for storage
        action = DecisionAction(
            shoot=float(shoot),
            pass_=float(pass_),
            move=float(move),
            tackle=float(tackle),
            get_possession_extra=float(gp_extra),
            mark=float(mark),
            hold_position=float(hold),
            pass_target=int(pass_tgt),
            tackle_target=int(tackle_tgt),
            mark_target=int(mark_tgt),
            move_region_center_raw=mv_center_raw.squeeze(0).cpu().numpy(),
            move_region_size_raw=float(mv_size_raw),
            move_arrival_speed_raw=float(mv_speed_raw),
            attack_defence_raw=float(ad_raw),
        )

        # Raw execution samples needed to recompute log_prob during PPO update
        raw_exec_samples = {
            "exec_move": np.array([float(exec_move)], dtype=np.float32),
            "sprint": np.array([float(sprint)], dtype=np.float32),
            "kick": np.array([float(kick)], dtype=np.float32),
            "tackle_attempt": np.array([float(tackle_attempt)], dtype=np.float32),
            "move_dir_raw": move_dir_raw.squeeze(0).cpu().numpy().astype(np.float32),
            "kick_dir_raw": kick_dir_raw.squeeze(0).cpu().numpy().astype(np.float32),
            "kick_power_raw": kick_power_raw.squeeze(0).cpu().numpy().astype(np.float32),
            "kick_spin_raw": kick_spin_raw.cpu().numpy().astype(np.float32),
        }

        return (
            action,
            float(log_prob),
            value,
            decision_probs,
            execution_physical,
            target_slots,
            raw_exec_samples,
            head_log_probs,
        )

    @torch.no_grad()
    def _sample_action_from_heads_batch(
        self,
        d_heads,
        e_heads,
        value_batch: torch.Tensor,
        x_sign_batch: torch.Tensor,
        em,
        deterministic: bool = False,
        deterministic_decision: bool = False,
        deterministic_direction: bool = False,
    ) -> list:
        """Vectorized tail: sampling + log_prob + decanonicalization for ALL
        K rows in ONE pass. This is the only production implementation;
        it replaced a Python loop of K individual
        ``_sample_action_from_heads`` calls (that loop was ~53% of rollout
        decision time despite the network forward pass already being batched,
        since every distribution/log_prob/masking call it made was batch=1;
        ``_sample_action_from_heads`` survives as the test reference). Returns a
        list of K tuples, each identical in shape/semantics to what
        ``_sample_action_from_heads`` returns for that row (see its
        docstring) -- this is a drop-in replacement for
        ``_sample_action_batch``'s inner loop, not a new/different sampling
        policy.

        Why this is safe to vectorize: every distribution class involved
        (IndependentBernoulli, MaskedCategorical, VonMisesDirectionHead,
        KickDirectionHead, SquashedNormalHead, torch.distributions.Normal)
        already operates on an arbitrary leading batch dimension -- the
        batch=1 restriction in ``_sample_action_from_heads`` came entirely
        from ITS OWN ``.squeeze(0)``/``float()``/``.item()`` calls, never
        from the underlying sampling machinery. Likewise ``mirror_x``/
        ``x_sign_of`` (ai/obs/canonical.py) already accept a batched
        x_sign. The one place with real per-row CONDITIONAL logic -- gating
        sprint/move_dir by exec_move and kick_dir/kick_power/kick_spin by
        kick, plus masking frozen heads (``self._ppo_lp_masked_heads``) --
        is deliberately NOT reimplemented here: this method builds an
        ``mb_actions`` dict from the freshly sampled actions and hands it to
        ``_recompute_log_prob``/``_per_head_new_log_probs``, the SAME
        already-vectorized functions the PPO update loop already calls every
        rollout to recompute log_prob for stored actions (verified
        line-for-line equivalent to ``_compute_log_prob``'s scalar gating
        used above by the single-row reference). The masking logic that was
        flagged as the highest-risk part of this vectorization therefore has
        exactly one implementation either way, never a second one that could
        silently drift -- this method's own new code is limited to sampling,
        decanonicalization, and packaging already-computed batched tensors
        into K per-row output tuples.
        """
        det_decision = deterministic or deterministic_decision
        det_direction = deterministic or deterministic_direction
        K = int(value_batch.shape[0])

        # Decision heads (Bernoulli)
        shoot_dist = IndependentBernoulli(d_heads.shoot_logit)
        pass_dist = IndependentBernoulli(d_heads.pass_logit)
        move_dist = IndependentBernoulli(d_heads.move_logit)
        tackle_dist = IndependentBernoulli(d_heads.tackle_logit)
        gp_extra_dist = IndependentBernoulli(d_heads.get_possession_raw)
        mark_dist = IndependentBernoulli(d_heads.mark_logit)
        hold_dist = IndependentBernoulli(d_heads.hold_position_logit)
        shoot = shoot_dist.mode() if det_decision else shoot_dist.sample()
        pass_ = pass_dist.mode() if det_decision else pass_dist.sample()
        move = move_dist.mode() if det_decision else move_dist.sample()
        tackle = tackle_dist.mode() if det_decision else tackle_dist.sample()
        gp_extra = gp_extra_dist.mode() if det_decision else gp_extra_dist.sample()
        mark = mark_dist.mode() if det_decision else mark_dist.sample()
        hold = hold_dist.mode() if det_decision else hold_dist.sample()

        # Categorical targets (masked)
        pass_tgt_dist = MaskedCategorical(d_heads.pass_target_logits, em)
        tackle_tgt_dist = MaskedCategorical(d_heads.tackle_target_logits, em)
        mark_tgt_dist = MaskedCategorical(d_heads.mark_target_logits, em)
        pass_tgt = pass_tgt_dist.mode() if det_decision else pass_tgt_dist.sample()
        tackle_tgt = tackle_tgt_dist.mode() if det_decision else tackle_tgt_dist.sample()
        mark_tgt = mark_tgt_dist.mode() if det_decision else mark_tgt_dist.sample()

        # Continuous decision heads (pre-squash raw samples for PPO)
        mv_center_raw = d_heads.move_region_center  # (K, 2)
        mv_size_raw = d_heads.move_region_size       # (K, 1)
        mv_speed_raw = d_heads.move_arrival_speed    # (K, 1)
        ad_raw = d_heads.attack_defence_raw          # (K, 1)

        tackle_prob, gp_prob = derive_get_possession_prob(
            d_heads.tackle_logit, d_heads.get_possession_raw
        )
        shoot_prob = torch.sigmoid(d_heads.shoot_logit)
        pass_prob = torch.sigmoid(d_heads.pass_logit)
        move_prob = torch.sigmoid(d_heads.move_logit)
        mark_prob = torch.sigmoid(d_heads.mark_logit)
        hold_prob = torch.sigmoid(d_heads.hold_position_logit)

        # Sample execution heads
        exec_move_dist = IndependentBernoulli(e_heads.exec_move_logit)
        sprint_dist = IndependentBernoulli(e_heads.sprint_logit)
        kick_dist = IndependentBernoulli(e_heads.kick_logit)
        tackle_attempt_dist = IndependentBernoulli(e_heads.tackle_attempt_logit)
        exec_move = exec_move_dist.mode() if det_decision else exec_move_dist.sample()
        sprint = sprint_dist.mode() if det_decision else sprint_dist.sample()
        kick = kick_dist.mode() if det_decision else kick_dist.sample()
        tackle_attempt = tackle_attempt_dist.mode() if det_decision else tackle_attempt_dist.sample()

        # Direction heads (see _sample_action_from_heads for the full
        # rationale comment -- unchanged here, just batched).
        eps = 1e-6
        log_kappa_move = self.execution_net.move_dir_log_kappa
        log_kappa_kick = self.execution_net.kick_dir_log_kappa
        log_std_z_kick = self.execution_net.kick_dir_z_log_std
        move_dir_head = self._move_dir_head(e_heads.move_direction, log_kappa_move)
        kick_dir_head = self._kick_dir_head(e_heads.kick_direction, log_kappa_kick, log_std_z_kick)
        if det_direction:
            move_dir_raw = move_dir_head.mode_physical()  # (K, 2)
            kick_dir_raw = kick_dir_head.mode_physical()   # (K, 3)
        else:
            move_dir_raw = move_dir_head.sample_raw()  # (K, 2)
            kick_dir_raw = kick_dir_head.sample_raw()   # (K, 3)
        move_dir_phys = move_dir_raw / (move_dir_raw.norm(dim=-1, keepdim=True) + eps)
        kick_dir_phys = kick_dir_raw / (kick_dir_raw.norm(dim=-1, keepdim=True) + eps)

        log_std_power = self.execution_net.kick_power_log_std
        log_std_spin = self.execution_net.kick_spin_log_std
        kick_power_head = self._kick_power_head(e_heads.kick_power, log_std_power)
        kick_spin_dist = self._kick_spin_dist(e_heads.kick_spin, log_std_spin)
        if det_direction:
            kick_power_raw = kick_power_head.dist.mean  # (K, 1), pre-squash
            kick_spin_raw = kick_spin_dist.mean          # (K, 3)
        else:
            kick_power_raw = kick_power_head.sample_raw()  # (K, 1), pre-squash
            kick_spin_raw = kick_spin_dist.rsample()        # (K, 3)
        kick_power_phys = kick_power_head.to_physical(kick_power_raw)  # (K, 1)

        # Decanonicalize: these are world-frame physical directions from here on.
        move_dir_world = mirror_x(move_dir_phys, x_sign_batch)  # (K, 2)
        kick_dir_world = mirror_x(kick_dir_phys, x_sign_batch)  # (K, 3)

        # Combined + per-head log_prob: reuse the update-time vectorized
        # masking (_recompute_log_prob/_per_head_new_log_probs) rather than
        # reimplementing exec_move/kick/frozen-head gating here -- see
        # docstring above.
        mb_actions = {
            "shoot": shoot, "pass_": pass_, "move": move, "tackle": tackle,
            "get_possession_extra": gp_extra, "mark": mark, "hold_position": hold,
            "pass_target": pass_tgt.unsqueeze(-1).float(),
            "tackle_target": tackle_tgt.unsqueeze(-1).float(),
            "mark_target": mark_tgt.unsqueeze(-1).float(),
            "exec_move": exec_move, "sprint": sprint, "kick": kick,
            "tackle_attempt": tackle_attempt,
            "move_dir_raw": move_dir_raw, "kick_dir_raw": kick_dir_raw,
            "kick_power_raw": kick_power_raw, "kick_spin_raw": kick_spin_raw,
        }
        log_prob_batch = self._recompute_log_prob(d_heads, e_heads, mb_actions, em)  # (K,)
        head_log_probs_batch = self._per_head_new_log_probs(d_heads, e_heads, mb_actions, em)  # (K, 15)

        # Unpack into K per-row tuples -- cheap indexing/dict-building only,
        # no distribution construction or network calls below this point.
        results = []
        for i in range(K):
            action = DecisionAction(
                shoot=float(shoot[i]),
                pass_=float(pass_[i]),
                move=float(move[i]),
                tackle=float(tackle[i]),
                get_possession_extra=float(gp_extra[i]),
                mark=float(mark[i]),
                hold_position=float(hold[i]),
                pass_target=int(pass_tgt[i]),
                tackle_target=int(tackle_tgt[i]),
                mark_target=int(mark_tgt[i]),
                move_region_center_raw=mv_center_raw[i].cpu().numpy(),
                move_region_size_raw=float(mv_size_raw[i]),
                move_arrival_speed_raw=float(mv_speed_raw[i]),
                attack_defence_raw=float(ad_raw[i]),
            )
            decision_probs = {
                "shoot": float(shoot_prob[i]),
                "pass_": float(pass_prob[i]),
                "move": float(move_prob[i]),
                "tackle": float(tackle_prob[i]),
                "get_possession": float(gp_prob[i]),
                "mark": float(mark_prob[i]),
                "hold_position": float(hold_prob[i]),
            }
            target_slots = {
                "pass_": int(pass_tgt[i]),
                "tackle": int(tackle_tgt[i]),
                "mark": int(mark_tgt[i]),
            }
            execution_physical = {
                "exec_move": bool(exec_move[i].item() > 0.5),
                "move_direction": move_dir_world[i].cpu().numpy(),
                "sprint": bool(sprint[i].item() > 0.5),
                "kick_this_tick": bool(kick[i].item() > 0.5),
                "kick_direction": kick_dir_world[i].cpu().numpy(),
                "kick_power_fraction": float(kick_power_phys[i]),
                "kick_spin": kick_spin_raw[i].cpu().numpy(),
                "tackle_attempt": bool(tackle_attempt[i].item() > 0.5),
            }
            raw_exec_samples = {
                "exec_move": np.array([float(exec_move[i])], dtype=np.float32),
                "sprint": np.array([float(sprint[i])], dtype=np.float32),
                "kick": np.array([float(kick[i])], dtype=np.float32),
                "tackle_attempt": np.array([float(tackle_attempt[i])], dtype=np.float32),
                "move_dir_raw": move_dir_raw[i].cpu().numpy().astype(np.float32),
                "kick_dir_raw": kick_dir_raw[i].cpu().numpy().astype(np.float32),
                "kick_power_raw": kick_power_raw[i].cpu().numpy().astype(np.float32),
                "kick_spin_raw": kick_spin_raw[i].cpu().numpy().astype(np.float32),
            }
            results.append((
                action,
                float(log_prob_batch[i]),
                float(value_batch[i]),
                decision_probs,
                execution_physical,
                target_slots,
                raw_exec_samples,
                head_log_probs_batch[i].cpu().numpy(),
            ))
        return results

    @torch.no_grad()
    def _sample_action(
        self,
        obs_dict: dict,
        deterministic: bool = False,
        deterministic_decision: bool = False,
        deterministic_direction: bool = False,
    ) -> tuple:
        """Forward pass + sample from all distributions, for ONE observation.

        Thin wrapper: builds a batch of 1 and routes through
        ``_sample_action_batch`` (see its docstring, and
        ``_sample_action_networks``/``_sample_action_from_heads_batch``) so the
        single-row path can never diverge from the batched path -- it IS the
        batched path, called with batch_size=1.

        See ``_sample_action_from_heads`` for the deterministic*/return-value
        documentation (unchanged).
        """
        obs_batch = {k: v.unsqueeze(0) for k, v in obs_dict.items()}
        return self._sample_action_batch(
            obs_batch,
            deterministic=deterministic,
            deterministic_decision=deterministic_decision,
            deterministic_direction=deterministic_direction,
        )[0]

    @torch.no_grad()
    def _sample_action_batch(
        self,
        obs_dict_batch: dict,
        deterministic: bool = False,
        deterministic_decision: bool = False,
        deterministic_direction: bool = False,
    ) -> list:
        """Batched variant of ``_sample_action``: ``obs_dict_batch`` already
        has a batch dimension (K rows, K>=1). Runs the 3 network forward
        passes ONCE (``_sample_action_networks``) and then the vectorized
        tail (``_sample_action_from_heads_batch``) ONCE, for all K rows
        together -- see that method's docstring for why this is safe.
        Returns a list of K tuples, each identical in shape/semantics to
        what a single ``_sample_action(...)`` call would return for that
        row's own observation (see ``_sample_action_from_heads``'s
        docstring for the exact tuple contents; ``_sample_action_from_heads_batch``
        returns the same shape, just computed vectorized).
        """
        d_heads_b, e_heads_b, value_b, x_sign_b, em_b = self._sample_action_networks(obs_dict_batch)
        return self._sample_action_from_heads_batch(
            d_heads_b, e_heads_b, value_b, x_sign_b, em_b,
            deterministic=deterministic,
            deterministic_decision=deterministic_decision,
            deterministic_direction=deterministic_direction,
        )

    def _get_value(self, obs_dict: dict) -> float:
        sf = obs_dict["self_feat"].to(self.device)
        of = obs_dict["other_feat"].to(self.device)
        em = obs_dict["exists_mask"].to(self.device)
        bf = obs_dict["ball_feat"].to(self.device)
        gf = obs_dict["global_feat"].to(self.device)
        sat = obs_dict["self_ai_type"].to(self.device) if "self_ai_type" in obs_dict else None
        oat = obs_dict["other_ai_type"].to(self.device) if "other_ai_type" in obs_dict else None
        d_heads = self.decision_net(sf, of, em, bf, gf, sat, oat)
        _value = self._value_heads(sf, of, em, bf, gf, d_heads, sat, oat)
        return float(_value.mean())  # single critic (execution_net, or self.value_net)

    def _compute_log_prob(self, d_heads, e_heads, samples: dict, exists_mask) -> torch.Tensor:
        """Compute combined log_prob across all action heads."""
        lp = torch.zeros(1, device=self.device)
        masked = self._ppo_lp_masked_heads

        # Bernoulli decision heads (skip heads masked for current phase)
        if "shoot_logit" not in masked:
            lp += IndependentBernoulli(d_heads.shoot_logit).log_prob(samples["shoot"]).sum()
        if "pass_logit" not in masked:
            lp += IndependentBernoulli(d_heads.pass_logit).log_prob(samples["pass_"]).sum()
        if "move_logit" not in masked:
            lp += IndependentBernoulli(d_heads.move_logit).log_prob(samples["move"]).sum()
        if "tackle_logit" not in masked:
            lp += IndependentBernoulli(d_heads.tackle_logit).log_prob(samples["tackle"]).sum()
        if "get_possession_raw" not in masked:
            lp += IndependentBernoulli(d_heads.get_possession_raw).log_prob(samples["gp_extra"]).sum()
        if "mark_logit" not in masked:
            lp += IndependentBernoulli(d_heads.mark_logit).log_prob(samples["mark"]).sum()
        if "hold_position_logit" not in masked:
            lp += IndependentBernoulli(d_heads.hold_position_logit).log_prob(samples["hold"]).sum()

        # Categorical target heads (gated by intent; also skip when parent is masked)
        if "pass_logit" not in masked and samples["pass_"] > 0.5:
            lp += MaskedCategorical(d_heads.pass_target_logits, exists_mask).log_prob(
                samples["pass_tgt"]
            )
        if "tackle_logit" not in masked and samples["tackle"] > 0.5:
            lp += MaskedCategorical(d_heads.tackle_target_logits, exists_mask).log_prob(
                samples["tackle_tgt"]
            )
        if "mark_logit" not in masked and samples["mark"] > 0.5:
            lp += MaskedCategorical(d_heads.mark_target_logits, exists_mask).log_prob(
                samples["mark_tgt"]
            )

        # Unconditional execution Bernoulli heads
        lp += IndependentBernoulli(e_heads.exec_move_logit).log_prob(samples["exec_move"]).sum()
        lp += IndependentBernoulli(e_heads.kick_logit).log_prob(samples["kick"]).sum()
        lp += IndependentBernoulli(e_heads.tackle_attempt_logit).log_prob(
            samples["tackle_attempt"]
        ).sum()

        # Sub-parameters gated by parent action — only contribute to log_prob
        # when the parent was actually taken.  Unconditional inclusion injects
        # large-variance noise from unused heads and inflates KL divergence.
        log_kappa_move = self.execution_net.move_dir_log_kappa
        log_kappa_kick = self.execution_net.kick_dir_log_kappa
        log_std_z_kick = self.execution_net.kick_dir_z_log_std
        log_std_power = self.execution_net.kick_power_log_std
        log_std_spin = self.execution_net.kick_spin_log_std
        # sprint + move_dir: only when exec_move=True (player was moving)
        if float(samples["exec_move"]) > 0.5:
            lp += IndependentBernoulli(e_heads.sprint_logit).log_prob(samples["sprint"]).sum()
            lp += self._move_dir_head(e_heads.move_direction, log_kappa_move).log_prob(
                samples["move_dir_raw"]
            )
        # kick_dir/kick_power: only when kick=True (a kick was taken) -- both
        # are only ever consumed downstream inside the engine's
        # `if kick_this_tick:` block.
        if float(samples["kick"]) > 0.5:
            lp += self._kick_dir_head(e_heads.kick_direction, log_kappa_kick, log_std_z_kick).log_prob(
                samples["kick_dir_raw"]
            )
            lp += self._kick_power_head(e_heads.kick_power, log_std_power).log_prob(
                samples["kick_power_raw"]
            )
        # kick_spin is permanently frozen (see agent_plans/spin_implementation_plan.md
        # section 0) -- deliberately excluded here, not just gated by kick=True,
        # since freezing requires_grad alone does not stop trunk-drift noise
        # from leaking into the ratio (see the plan doc's section 0.2).
        if not self._kick_spin_frozen and float(samples["kick"]) > 0.5:
            lp += self._kick_spin_dist(e_heads.kick_spin, log_std_spin).log_prob(
                samples["kick_spin_raw"]
            ).sum(dim=-1)

        return lp

    def _recompute_log_prob(self, d_heads, e_heads, mb_actions: dict, exists_mask) -> torch.Tensor:
        """Recompute log_probs for stored actions under the current policy."""
        lp = torch.zeros(exists_mask.shape[0], device=self.device)
        masked = self._ppo_lp_masked_heads

        def _b(logit, key):
            return IndependentBernoulli(logit).log_prob(mb_actions[key]).squeeze(-1)

        if "shoot_logit" not in masked:
            lp += _b(d_heads.shoot_logit, "shoot")
        if "pass_logit" not in masked:
            lp += _b(d_heads.pass_logit, "pass_")
        if "move_logit" not in masked:
            lp += _b(d_heads.move_logit, "move")
        if "tackle_logit" not in masked:
            lp += _b(d_heads.tackle_logit, "tackle")
        if "get_possession_raw" not in masked:
            lp += _b(d_heads.get_possession_raw, "get_possession_extra")
        if "mark_logit" not in masked:
            lp += _b(d_heads.mark_logit, "mark")
        if "hold_position_logit" not in masked:
            lp += _b(d_heads.hold_position_logit, "hold_position")
        lp += _b(e_heads.exec_move_logit, "exec_move")
        lp += _b(e_heads.kick_logit, "kick")
        lp += _b(e_heads.tackle_attempt_logit, "tackle_attempt")

        # Target categorical log_probs (gated by parent intent).
        # Automatically skipped when the parent Bernoulli is masked — if pass_logit
        # is masked then pass_ samples are excluded from the ratio and the categorical
        # target contribution would be spurious noise too.
        pass_mask = mb_actions["pass_"].squeeze(-1) > 0.5
        tackle_mask = mb_actions["tackle"].squeeze(-1) > 0.5
        mark_mask = mb_actions["mark"].squeeze(-1) > 0.5

        for parent_name, mask, logits, key in [
            ("pass_logit",   pass_mask,   d_heads.pass_target_logits,   "pass_target"),
            ("tackle_logit", tackle_mask, d_heads.tackle_target_logits, "tackle_target"),
            ("mark_logit",   mark_mask,   d_heads.mark_target_logits,   "mark_target"),
        ]:
            if parent_name in masked:
                continue
            if mask.any():
                cat_lp = MaskedCategorical(logits, exists_mask).log_prob(
                    mb_actions[key].long().squeeze(-1)
                )
                lp[mask] += cat_lp[mask]

        # Sub-parameters gated by parent action (vectorised float mask over minibatch).
        # sprint + move_dir only contribute when exec_move=True.
        # kick_dir only contributes when kick=True.
        # Without this gating, unused heads inject large-variance log_prob noise
        # that inflates KL and triggers spurious early stops every rollout.
        log_kappa_move = self.execution_net.move_dir_log_kappa.to(self.device)
        log_kappa_kick = self.execution_net.kick_dir_log_kappa.to(self.device)
        log_std_z_kick = self.execution_net.kick_dir_z_log_std.to(self.device)
        log_std_power = self.execution_net.kick_power_log_std.to(self.device)
        log_std_spin = self.execution_net.kick_spin_log_std.to(self.device)
        exec_move_mask = (mb_actions["exec_move"].squeeze(-1) > 0.5).float()
        kick_mask = (mb_actions["kick"].squeeze(-1) > 0.5).float()
        lp += exec_move_mask * _b(e_heads.sprint_logit, "sprint")
        lp += exec_move_mask * (
            self._move_dir_head(e_heads.move_direction, log_kappa_move).log_prob(
                mb_actions["move_dir_raw"]
            )
        )
        lp += kick_mask * (
            self._kick_dir_head(e_heads.kick_direction, log_kappa_kick, log_std_z_kick).log_prob(
                mb_actions["kick_dir_raw"]
            )
        )
        lp += kick_mask * (
            self._kick_power_head(e_heads.kick_power, log_std_power).log_prob(
                mb_actions["kick_power_raw"]
            )
        )
        # kick_spin is permanently frozen (see agent_plans/spin_implementation_plan.md
        # section 0) -- excluded here rather than just gated by kick_mask, see
        # the plan doc's section 0.2 for why requires_grad alone isn't enough.
        if not self._kick_spin_frozen:
            lp += kick_mask * (
                self._kick_spin_dist(e_heads.kick_spin, log_std_spin).log_prob(
                    mb_actions["kick_spin_raw"]
                ).sum(dim=-1)
            )

        return lp

    def _compute_entropy(self, d_heads, e_heads, exists_mask, return_breakdown: bool = False,
                         return_kick_boost: bool = False, opp_masks: Optional[dict] = None):
        """Entropy bonus, consistent with the masked log_prob.

        Restricted to exactly the heads in HEAD_LP_KEYS (rollout_buffer.py)
        -- the same 15 heads _recompute_log_prob/the per-head KL diagnostic
        track. pass_target/tackle_target/mark_target are NOT included: they
        never enter the PPO importance ratio at all (BC-aux-only, see
        set_frozen_heads' docstring), so an entropy bonus over them doesn't
        serve PPO's own exploration-regularization purpose. Any head
        currently in self._ppo_lp_masked_heads (frozen for this curriculum
        phase, no reward signal) is zeroed here too, matching
        _recompute_log_prob exactly -- otherwise the entropy bonus would
        still pull the shared decision_net trunk towards more uncertainty on
        a "frozen" head via that head's own (still-differentiable) forward
        pass, even though PPO's policy-gradient term for that head is zero.

        Sub-parameter heads (sprint, move_dir, kick_dir) are weighted by
        E[parent active] to match the masking in _recompute_log_prob where
        those terms are 0 when the parent is not taken.

        decision_net's 7 Bernoulli heads (shoot/pass/move/tackle/
        get_possession_raw/mark/hold_position) are additionally scaled by
        ent_decision_weight, separate from execution_net's exec_move/kick/
        tackle_attempt (always weight 1.0). See ai_config.json's
        ent_decision_weight comment.

        ``opp_masks`` (ppo.mask_untriggered_actions, agent_plans/masked_action_training_plan.md):
        ``{"kick": (B,), "tackle_attempt": (B,)}`` per-row opportunity masks (1 = the action could take
        effect on that row), optionally with ``"kick_scale"``/``"tackle_attempt_scale"`` floats (the
        1/P(opportunity) rescale). The kick gate / tackle_attempt entropy is then the mask-weighted mean over
        rows, and kick_dir / kick_power (/ spin) entropy is weighted per row by ``mask_kick * p_kick(row)`` (the
        E[kick] weighting restricted to rows where a kick could happen) instead of by the batch-mean p_kick.
        ``None`` = the unmasked formulas below, unchanged.

        ``return_kick_boost`` returns the extra bonus terms that only enter the PPO loss (never the logged entropy): the
        ``ent_kick_weight`` boost on the kick heads, ``ent_move_dir_weight`` on move_dir, and
        ``ent_kick_power_only_weight`` / ``ent_kick_dir_azimuth_only_weight`` / ``ent_kick_dir_z_only_weight`` on
        kick_power / kick_dir's azimuth / kick_dir's elevation specifically (detached kick-gate weighting, so
        unlike ``ent_kick_weight`` these never also push on ``kick_logit``, and never on each other's parameter).

        Args:
            return_breakdown: if True, also return a dict of each head's own
                (already E[parent active]-weighted, matching what's actually
                added into the total below) entropy contribution — used by
                the per-rollout "[per-head entropy]" diagnostic log line to
                show which heads are driving a rising aggregate entropy,
                rather than just the summed scalar. See ai/knowledge.md's
                "Per-head entropy diagnostics" note.
        """
        ent = torch.zeros(1, device=self.device)
        # Per-head breakdown tensors, NOT yet synced to CPU -- see the single
        # combined torch.stack(...).tolist() call below. Called every PPO
        # minibatch with return_breakdown=True, so 15 separate .item() calls
        # here (one CPU<->GPU round trip each) was a real, measurable per-
        # minibatch cost; batching into one round trip changes none of the
        # returned values, only how many syncs it takes to get them.
        _bkdn_tensors: dict[str, torch.Tensor] = {}
        masked = self._ppo_lp_masked_heads
        # Unconditional heads (no parent gate)
        for name, logit, mask_key in [
            ("shoot", d_heads.shoot_logit, "shoot_logit"),
            ("pass_", d_heads.pass_logit, "pass_logit"),
            ("move", d_heads.move_logit, "move_logit"),
            ("tackle", d_heads.tackle_logit, "tackle_logit"),
            ("gp_extra", d_heads.get_possession_raw, "get_possession_raw"),
            ("mark", d_heads.mark_logit, "mark_logit"),
            ("hold", d_heads.hold_position_logit, "hold_position_logit"),
            ("exec_move", e_heads.exec_move_logit, None),
            ("kick", e_heads.kick_logit, None),
            ("tackle_attempt", e_heads.tackle_attempt_logit, None),
        ]:
            if mask_key is not None and mask_key in masked:
                h = torch.zeros((), device=self.device)
            elif opp_masks is not None and name in ("kick", "tackle_attempt"):
                h = (opp_masks[name] * IndependentBernoulli(logit).entropy().reshape(-1)).mean() * opp_masks.get(f"{name}_scale", 1.0)
            else:
                h = IndependentBernoulli(logit).entropy().mean()
                if mask_key is not None:
                    # mask_key is only set for decision_net's own heads (the
                    # 7 in _LP_HEAD_NAMES) -- exec_move/kick/tackle_attempt
                    # (mask_key=None) always keep weight 1.0 regardless of
                    # ent_decision_weight. See its ai_config.json comment.
                    h = self.ent_decision_weight * h
            ent += h
            _bkdn_tensors[name] = h
        # Sub-parameters: scale by E[parent active] to match masked log_prob.
        log_kappa_move = self.execution_net.move_dir_log_kappa
        log_kappa_kick = self.execution_net.kick_dir_log_kappa
        log_std_z_kick = self.execution_net.kick_dir_z_log_std
        log_std_power = self.execution_net.kick_power_log_std
        log_std_spin = self.execution_net.kick_spin_log_std
        p_exec_move = torch.sigmoid(e_heads.exec_move_logit).mean()
        p_kick = torch.sigmoid(e_heads.kick_logit).mean()
        h_sprint = p_exec_move * IndependentBernoulli(e_heads.sprint_logit).entropy().mean()
        _h_move_dir_raw = self._move_dir_head(e_heads.move_direction, log_kappa_move).entropy().mean()
        h_move_dir = p_exec_move * _h_move_dir_raw
        # Raw (un-gated) entropy of kick_dir/kick_power, kept UNREDUCED (per-row) here so the
        # opportunity-weighted mean below (masked branch) can be reproduced exactly for the
        # detached-gate boosts further down -- mean(w * x) != mean(w) * mean(x) when w varies per
        # row, so this can't just be a scalar the way _h_move_dir_raw is (move_dir's gate,
        # p_exec_move, is always a plain batch-mean scalar, never per-row).
        _kick_dir_head_obj = self._kick_dir_head(e_heads.kick_direction, log_kappa_kick, log_std_z_kick)
        _h_kick_dir_raw = _kick_dir_head_obj.entropy()
        # Diagnostic-only split of the combined kick_dir entropy above into its two constituent
        # components (azimuth: von Mises on kappa; elevation/z: plain Gaussian on std_z) -- both
        # global scalar parameters (not state-dependent, see the class comment), so these are
        # themselves scalar/broadcastable, same shape story as _h_kick_dir_raw itself. Logged
        # (via _bkdn_tensors below) so azimuth+z sharpening can be told apart in the training log
        # and future plots, since KickDirectionHead.entropy() only ever returns their SUM.
        _h_kick_dir_azimuth_raw = _von_mises_entropy(_kick_dir_head_obj.kappa)
        _h_kick_dir_z_raw = _kick_dir_head_obj.dist_z.entropy()
        _h_kick_power_raw = self._kick_power_head(e_heads.kick_power, log_std_power).entropy()
        if opp_masks is None:
            h_kick_dir = p_kick * _h_kick_dir_raw.mean()
            h_kick_dir_azimuth = p_kick * _h_kick_dir_azimuth_raw.mean()
            h_kick_dir_z = p_kick * _h_kick_dir_z_raw.mean()
            h_kick_power = p_kick * self.ent_kick_power_weight * _h_kick_power_raw.mean()
            # kick_spin is permanently frozen (see agent_plans/spin_implementation_plan.md
            # section 0) -- its entropy term is masked to exactly zero rather than
            # computed and discarded, same rationale as the log_prob masking above.
            h_kick_spin = (
                torch.zeros((), device=self.device) if self._kick_spin_frozen else
                p_kick * self.ent_kick_spin_weight * self._kick_spin_dist(e_heads.kick_spin, log_std_spin).entropy().sum(dim=-1).mean()
            )
            # Detached-gate versions for ent_kick_power_only_weight/ent_kick_dir_azimuth_only_weight/
            # ent_kick_dir_z_only_weight below: p_kick.detach() has the identical forward value, so
            # these still scale correctly by "how often kicks happen right now", but carry no
            # gradient into kick_logit.
            _kick_power_only_raw = p_kick.detach() * self.ent_kick_power_weight * _h_kick_power_raw.mean()
            _kick_dir_azimuth_only_raw = p_kick.detach() * _h_kick_dir_azimuth_raw.mean()
            _kick_dir_z_only_raw = p_kick.detach() * _h_kick_dir_z_raw.mean()
        else:
            _w_kick = opp_masks["kick"] * torch.sigmoid(e_heads.kick_logit).reshape(-1) * opp_masks.get("kick_scale", 1.0)
            h_kick_dir = (_w_kick * _h_kick_dir_raw.reshape(-1)).mean()
            h_kick_dir_azimuth = (_w_kick * _h_kick_dir_azimuth_raw.reshape(-1)).mean()
            h_kick_dir_z = (_w_kick * _h_kick_dir_z_raw.reshape(-1)).mean()
            h_kick_power = (_w_kick * self.ent_kick_power_weight * _h_kick_power_raw.reshape(-1)).mean()
            h_kick_spin = (
                torch.zeros((), device=self.device) if self._kick_spin_frozen else
                (_w_kick * self.ent_kick_spin_weight * self._kick_spin_dist(e_heads.kick_spin, log_std_spin).entropy().sum(dim=-1).reshape(-1)).mean()
            )
            # opp_masks["kick"] carries no gradient either way (an env-recorded flag, not a network
            # output); only the sigmoid(kick_logit) factor needs detaching.
            _w_kick_detached = opp_masks["kick"] * torch.sigmoid(e_heads.kick_logit).detach().reshape(-1) * opp_masks.get("kick_scale", 1.0)
            _kick_power_only_raw = (_w_kick_detached * self.ent_kick_power_weight * _h_kick_power_raw.reshape(-1)).mean()
            _kick_dir_azimuth_only_raw = (_w_kick_detached * _h_kick_dir_azimuth_raw.reshape(-1)).mean()
            _kick_dir_z_only_raw = (_w_kick_detached * _h_kick_dir_z_raw.reshape(-1)).mean()
        ent += h_sprint + h_move_dir + h_kick_dir + h_kick_power + h_kick_spin
        _bkdn_tensors["sprint"] = h_sprint
        _bkdn_tensors["move_dir"] = h_move_dir
        _bkdn_tensors["kick_dir"] = h_kick_dir
        # Diagnostic only (see the comment where these are built above): NOT included in the
        # `ent` sum above (that already has h_kick_dir, their combined total) -- these two entries
        # exist purely so the training log / future plots can see kick_dir's azimuth-vs-elevation
        # split, which kick_dir alone can't show since KickDirectionHead.entropy() sums them.
        _bkdn_tensors["kick_dir_azimuth"] = h_kick_dir_azimuth
        _bkdn_tensors["kick_dir_z"] = h_kick_dir_z
        _bkdn_tensors["kick_power"] = h_kick_power
        _bkdn_tensors["kick_spin"] = h_kick_spin
        # ent_kick_weight: extra bonus on the kick gate + kick_dir + kick_power
        # entropy terms, returned SEPARATELY (return_kick_boost) so the logged
        # `entropy` / per-head breakdown keep meaning "the unboosted entropy"
        # and the caller adds the boost only where it builds the loss.
        kick_boost = (self.ent_kick_weight - 1.0) * (_bkdn_tensors["kick"] + h_kick_dir + h_kick_power)
        # ent_move_dir_weight: extra bonus on the move_dir entropy, carried in the same separate term (so the logged entropy and
        # breakdown stay unboosted). p_exec_move is detached: the von Mises entropy depends only on kappa, so this boost pulls
        # move_dir_log_kappa down (slower sharpening) without also pushing the exec_move gate toward "stand still".
        kick_boost = kick_boost + (self.ent_move_dir_weight - 1.0) * p_exec_move.detach() * _h_move_dir_raw
        # ent_kick_power_only_weight / ent_kick_dir_azimuth_only_weight / ent_kick_dir_z_only_weight:
        # same pattern as ent_move_dir_weight, but for the kick heads' own spread parameters. Unlike
        # ent_kick_weight above (which deliberately also pushes p(kick) itself -- see its comment --
        # it was added to stop the kick gate collapsing early), these use the DETACHED kick-gate
        # weighting, so the extra boost reaches only kick_power_log_std, or only kick_dir_log_kappa
        # (azimuth) / kick_dir_z_log_std (elevation) separately, never kick_logit. The azimuth/z
        # split (2026-09-23) replaced a single combined ent_kick_dir_only_weight that turned out to
        # push BOTH kick_dir_log_kappa and kick_dir_z_log_std at once (KickDirectionHead.entropy()
        # sums them, and _kick_dir_only_raw was built from that sum) -- measured directly (isolated
        # backward pass on a real checkpoint) to split roughly 2:1 in favor of kick_dir_z_log_std,
        # NOT azimuth-only as this comment previously (incorrectly) claimed. The two gradients are
        # mathematically independent (additive branches), so this split changes NOTHING about how
        # hard azimuth is pushed -- it only stops the accidental extra push into elevation.
        kick_boost = kick_boost + (self.ent_kick_power_only_weight - 1.0) * _kick_power_only_raw
        kick_boost = kick_boost + (self.ent_kick_dir_azimuth_only_weight - 1.0) * _kick_dir_azimuth_only_raw
        kick_boost = kick_boost + (self.ent_kick_dir_z_only_weight - 1.0) * _kick_dir_z_only_raw
        result = [ent]
        if return_breakdown:
            _names = list(_bkdn_tensors.keys())
            _vals = torch.stack([_bkdn_tensors[n] for n in _names]).tolist()
            result.append(dict(zip(_names, _vals)))
        if return_kick_boost:
            result.append(kick_boost)
        return result[0] if len(result) == 1 else tuple(result)
