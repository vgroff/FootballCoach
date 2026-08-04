"""Behavioural cloning (BC) support for rules-based AI bootstrapping.

Two modes, both configurable via ai_config.json['bc']:

1. **Pre-training** (before PPO starts):
   Run ``BCPretrainer.pretrain(env, n_steps)`` which rolls the rules-based AI
   in the environment and trains the network via pure supervised cross-entropy
   on the resulting (obs, label) pairs. This gives the network a decent
   starting point before PPO exploration begins.

2. **Auxiliary loss during PPO** (annealed to zero):
   At each PPO rollout-collection step, ``label_fn(env)`` is called to get a
   ``BCLabel`` for the current state. These labels are stored in the rollout
   buffer and later used in ``_ppo_update()`` as an auxiliary cross-entropy
   term weighted by ``bc_aux_coeff`` (linearly annealed to 0.0).

   This keeps the network from drifting too far from sensible behaviour early
   in PPO training while still letting the RL signal take over as it matures.

Design rule: BC labels do NOT go through the PPO importance ratio / clipping.
They are a separate, additive loss term. This means they can use actions taken
by the rules-based AI (which has no π_old) without corrupting PPO's math.

Flat tensor layout for stored BC labels (17 floats per step):
  [0]  shoot
  [1]  pass_
  [2]  move
  [3]  tackle
  [4]  get_possession_extra
  [5]  mark
  [6]  hold_position
  [7]  move_dir_x        (unit vector component; 0.0 if dir not applicable)
  [8]  move_dir_y
  [9]  sprint
  [10] move_region_x_m   (physical x of move target in metres; 0.0 if not applicable)
  [11] move_region_y_m   (physical y of move target in metres; 0.0 if not applicable)
  [12] kick_this_tick    (1.0 = player is kicking this step, 0.0 otherwise)
  [13] tackle_attempt    (1.0 = player is tackling this step, 0.0 otherwise)
  [14] valid             (1.0 = use this label, 0.0 = skip BC loss for this step)
  [15] exec_move         (1.0 = player is moving, 0.0 = standstill)
  [16] ai_type           (0.0=rules, 1.0=immobile, 2.0=neural [reserved, unused] -
                          which AI controlled this player when the label was
                          recorded; visible to the VALUE head only, never the
                          policy heads - see ai/knowledge.md)
  [17] opponent_ai_type  (same 0/1/2 coding as [16], but for the OTHER player
                          in the match at this same tick. In 1v1 phases this is
                          unambiguous - the single other player. Recorded
                          directly at demo-recording time rather than joined
                          across rows after the fact, since the two players'
                          rows are not otherwise correlatable once interleaved.
                          Used to build the value-only ``other_ai_type``
                          side-channel for BC/value pretraining - see
                          ai/knowledge.md "Opponent-AI-type (value-only)".)
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("footballcoach.ai.bc")

BC_LABEL_DIM = 18  # elements in the flat label vector (see module docstring)

# Indices into the flat label vector
_I_SHOOT          = 0
_I_PASS           = 1
_I_MOVE           = 2
_I_TACKLE         = 3
_I_GP_EXTRA       = 4
_I_MARK           = 5
_I_HOLD           = 6
_I_DIR_X          = 7
_I_DIR_Y          = 8
_I_SPRINT         = 9
_I_REGION_X       = 10
_I_REGION_Y       = 11
_I_KICK_THIS_TICK = 12
_I_TACKLE_ATTEMPT = 13
_I_VALID          = 14
_I_EXEC_MOVE      = 15
_I_AI_TYPE        = 16
_I_OPPONENT_AI_TYPE = 17

# ai_type integer codes (see module docstring layout table)
AI_TYPE_RULES    = 0.0
AI_TYPE_IMMOBILE = 1.0
AI_TYPE_NEURAL   = 2.0  # reserved, unused - no neural demo-recording mode yet

# Standard pitch half-dimensions used to normalise move_region supervision.
# Kept as constants here so bc_loss_from_tensor doesn't need a pitch object.
_PITCH_HALF_LENGTH_M = 52.5
_PITCH_HALF_WIDTH_M  = 34.0


@dataclass
class BCLabel:
    """Rules-based supervision label for one decision step.

    All Bernoulli targets are floats in {0.0, 1.0}.
    ``move_direction`` is a (dx, dy) unit vector or ``None`` to skip the
    direction loss (e.g. when movement is not the selected action).
    ``valid=False`` skips BC loss entirely for this step.
    """
    shoot: float = 0.0
    pass_: float = 0.0
    move: float = 0.0
    tackle: float = 0.0
    get_possession_extra: float = 0.0  # raw gp head, not derived gp_prob
    mark: float = 0.0
    hold_position: float = 0.0
    move_direction: Optional[np.ndarray] = None  # shape (2,) unit vector
    sprint: float = 1.0
    move_region_center_m: Optional[np.ndarray] = None  # shape (2,) physical metres
    kick_this_tick: float = 0.0   # execution: 1.0 if player is kicking this step
    tackle_attempt: float = 0.0   # execution: 1.0 if player is attempting a tackle
    exec_move: float = 1.0        # execution: 1.0 if player is moving, 0.0 = standstill
    valid: bool = True
    ai_type: float = AI_TYPE_RULES  # which AI produced this label (see AI_TYPE_* constants)
    opponent_ai_type: float = AI_TYPE_IMMOBILE  # which AI controls the OTHER player (see AI_TYPE_* constants)

    def to_array(self) -> np.ndarray:
        """Pack into a flat float32 array of length BC_LABEL_DIM."""
        arr = np.zeros(BC_LABEL_DIM, dtype=np.float32)
        arr[_I_SHOOT]    = self.shoot
        arr[_I_PASS]     = self.pass_
        arr[_I_MOVE]     = self.move
        arr[_I_TACKLE]   = self.tackle
        arr[_I_GP_EXTRA] = self.get_possession_extra
        arr[_I_MARK]     = self.mark
        arr[_I_HOLD]     = self.hold_position
        if self.move_direction is not None:
            arr[_I_DIR_X] = float(self.move_direction[0])
            arr[_I_DIR_Y] = float(self.move_direction[1])
        arr[_I_SPRINT]         = self.sprint
        if self.move_region_center_m is not None:
            arr[_I_REGION_X] = float(self.move_region_center_m[0])
            arr[_I_REGION_Y] = float(self.move_region_center_m[1])
        arr[_I_KICK_THIS_TICK] = self.kick_this_tick
        arr[_I_TACKLE_ATTEMPT] = self.tackle_attempt
        arr[_I_VALID]          = 1.0 if self.valid else 0.0
        arr[_I_EXEC_MOVE]      = self.exec_move
        arr[_I_AI_TYPE]        = self.ai_type
        arr[_I_OPPONENT_AI_TYPE] = self.opponent_ai_type
        return arr

    @staticmethod
    def invalid() -> "BCLabel":
        """A label that contributes zero BC loss."""
        return BCLabel(valid=False)


# ---------------------------------------------------------------------------
# Rules-based label generators
# ---------------------------------------------------------------------------

def phase1_labels(env, player_id: str = None) -> BCLabel:
    """Derive BC labels for Phase 1 by asking Phase1RulesAI what it would do.

    Instantiates a temporary Phase1RulesAI, calls act() on the current match
    state, and reads back the order it sets — so the labels are always exactly
    in sync with the rules AI behaviour, with no duplicated logic here.
    """
    from footballcoach.rules_ai import Phase1RulesAI
    from footballcoach.orders import MoveOrder, GetPossessionOrder, ShootOrder, ChaseTackleOrder, KickOrder, PassOrder

    try:
        match = env.match
        if player_id is None:
            player_id = env.trainee_player_id
        if match is None:
            return BCLabel.invalid()
        player = match.player_by_id(player_id)
    except (KeyError, AttributeError):
        return BCLabel.invalid()

    # ai_type reflects which AI actually controls *player* right now (rules
    # vs immobile) — NOT which AI the label was derived from (this function
    # always queries Phase1RulesAI internally regardless of the caller).
    ai_type = AI_TYPE_RULES if isinstance(player.ai, Phase1RulesAI) else AI_TYPE_IMMOBILE

    # opponent_ai_type: which AI controls the OTHER player in the match right
    # now. Phase 1 is strictly 1v1, so "the other player" is unambiguous —
    # any player in match.players that isn't `player`. Falls back to
    # AI_TYPE_IMMOBILE if no other player is found (should not happen in a
    # well-formed 1v1 scenario).
    opponent_ai_type = AI_TYPE_IMMOBILE
    for _p in match.players:
        if _p.player_id != player.player_id:
            opponent_ai_type = AI_TYPE_RULES if isinstance(_p.ai, Phase1RulesAI) else AI_TYPE_IMMOBILE
            break

    # Execution heads reflect what the rules AI is physically doing RIGHT NOW —
    # i.e. the current order it's executing, before any new decision is made.
    from footballcoach.orders import MoveOrder as _MoveOrder, GetPossessionOrder as _GPOrder
    current_exec = player.current_order
    kick_this_tick = 1.0 if isinstance(current_exec, (ShootOrder, KickOrder, PassOrder)) else 0.0
    # tackle_attempt: ChaseTackleOrder always, OR GetPossessionOrder when the player
    # is currently touching the ball carrier — that's when GP behaviour resolves a tackle.
    _is_gp_tackling = False
    if isinstance(current_exec, _GPOrder):
        try:
            carrier = match.ball_carrier()
            if carrier is not None and carrier.player_id != player.player_id:
                from footballcoach.engine.collision import are_touching
                _is_gp_tackling = are_touching(player, carrier)
        except Exception:
            pass
    tackle_attempt = 1.0 if (isinstance(current_exec, ChaseTackleOrder) or _is_gp_tackling) else 0.0
    # exec_move: 1 if the rules AI is currently executing any movement order
    exec_move_now = 1.0 if isinstance(current_exec, (_MoveOrder, _GPOrder)) else 0.0

    # Decision heads reflect what the rules AI DECIDES next.
    # Temporarily clear current_order so the AI always produces a fresh decision.
    player.current_order = None
    try:
        Phase1RulesAI().act(player, match, trial_tick=0)
        order = player.current_order
    finally:
        player.current_order = current_exec  # always restore

    if isinstance(order, MoveOrder):
        tx, ty = player.position.x, player.position.y
        tgt = order.target_position
        dx, dy = tgt.x - tx, tgt.y - ty
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return BCLabel.invalid()
        direction = np.array([dx / length, dy / length], dtype=np.float32)
        return BCLabel(
            move=1.0,
            sprint=1.0 if order.sprint else 0.0,
            move_direction=direction,
            move_region_center_m=np.array([tgt.x, tgt.y], dtype=np.float32),
            kick_this_tick=kick_this_tick,
            tackle_attempt=tackle_attempt,
            exec_move=1.0,
            ai_type=ai_type,
            opponent_ai_type=opponent_ai_type,
        )
    elif isinstance(order, GetPossessionOrder):
        ball = match.ball
        bx, by = ball.position.x, ball.position.y
        tx, ty = player.position.x, player.position.y
        dx, dy = bx - tx, by - ty
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return BCLabel.invalid()
        direction = np.array([dx / length, dy / length], dtype=np.float32)
        return BCLabel(
            get_possession_extra=1.0,
            sprint=1.0 if order.sprint else 0.0,
            move_direction=direction,
            move_region_center_m=np.array([bx, by], dtype=np.float32),
            kick_this_tick=kick_this_tick,
            tackle_attempt=tackle_attempt,
            exec_move=1.0,
            ai_type=ai_type,
            opponent_ai_type=opponent_ai_type,
        )
    else:
        return BCLabel.invalid()


# ---------------------------------------------------------------------------
# BC loss computation
# ---------------------------------------------------------------------------

def bc_loss_from_tensor(
    labels: torch.Tensor,
    decision_heads,
    exec_heads=None,
    direction_loss_weight: float = 3.0,
    region_loss_weight: float = 1.0,
    pos_weight_kick: float = 1.0,
    pos_weight_tackle_attempt: float = 1.0,
    return_breakdown: bool = False,
    dec_weight: float = 1.0,
    exec_weight: float = 1.0,
):
    """Compute BC loss for a minibatch, given packed label tensors.

    Args:
        labels: (N, BC_LABEL_DIM) float32 tensor from the rollout buffer.
        decision_heads: DecisionHeadsRaw from decision_net forward pass.
        exec_heads: ExecutionHeadsRaw from execution_net forward pass, or
            ``None`` to compute a decision-heads-only loss (skips all
            exec-dependent terms: exec_move/sprint/kick/tackle_attempt BCE
            and the move_direction cosine loss). Used by Phase 0 of
            ``PPOTrainer.pretrain_combined()``, which only trains
            ``decision_net`` — see ai/knowledge.md "Phase 0" note.
        direction_loss_weight: Multiplier on the move_direction cosine loss.
            From ai_config.json['bc']['direction_loss_weight'] (default 3.0).
            Upweights direction relative to ~11 Bernoulli BCE heads so it gets
            proportional gradient pressure. Ignored when exec_heads is None.
        region_loss_weight: Multiplier on the move_region_center MSE loss.
            From ai_config.json['bc']['region_loss_weight'] (default 1.0).
            Separate from direction_loss_weight: in Phase 1 the region target
            (ball position proxy) is noisy and less critical.
        pos_weight_kick: Positive-class weight (``F.binary_cross_entropy_with_logits``
            ``pos_weight`` semantics) applied to the ``kick_this_tick`` BCE
            term only, to counter class imbalance (rare positives). Default
            1.0 = no reweighting. See ``DemonstrationDataset.compute_pos_weights()``.
            Ignored when exec_heads is None.
        pos_weight_tackle_attempt: Same as ``pos_weight_kick`` but for the
            ``tackle_attempt`` BCE term. Ignored when exec_heads is None.
        return_breakdown: If True, also return a dict of per-group mean losses
            {decision, exec_bce, sprint, move, direction, region} for diagnostics.
            When exec_heads is None, the exec-dependent entries are all 0.0.

    Returns:
        Scalar BC loss (mean over valid steps), or (loss, breakdown_dict) if
        return_breakdown=True.  Returns zero tensor if no valid steps.
    """
    valid = labels[:, _I_VALID] > 0.5  # (N,) bool
    _zero = torch.zeros(1, device=labels.device)
    if not valid.any():
        if return_breakdown:
            return _zero, {"decision": 0.0, "exec_bce": 0.0, "sprint": 0.0, "move": 0.0, "tackle_attempt": 0.0, "direction": 0.0, "region": 0.0}
        return _zero

    loss = torch.zeros(labels.shape[0], device=labels.device)

    # --- Bernoulli decision heads (BCE from logits) ---
    def _bce(logit: torch.Tensor, col: int, pos_weight: float = 1.0) -> torch.Tensor:
        target = labels[:, col]
        pw = None
        if pos_weight != 1.0:
            pw = torch.tensor(pos_weight, device=labels.device, dtype=logit.dtype)
        return F.binary_cross_entropy_with_logits(
            logit.squeeze(-1), target, reduction="none", pos_weight=pw
        )

    dec_loss = (
        _bce(decision_heads.shoot_logit,          _I_SHOOT)
        + _bce(decision_heads.pass_logit,          _I_PASS)
        + _bce(decision_heads.move_logit,          _I_MOVE)
        + _bce(decision_heads.tackle_logit,        _I_TACKLE)
        + _bce(decision_heads.get_possession_raw,  _I_GP_EXTRA)
        + _bce(decision_heads.mark_logit,          _I_MARK)
        + _bce(decision_heads.hold_position_logit, _I_HOLD)
    )
    loss += dec_weight * dec_loss

    exec_bce_loss = torch.zeros(labels.shape[0], device=labels.device)
    sprint_loss = torch.zeros(labels.shape[0], device=labels.device)
    move_loss = torch.zeros(labels.shape[0], device=labels.device)
    tackle_attempt_loss = torch.zeros(labels.shape[0], device=labels.device)
    dir_loss_per = torch.zeros(labels.shape[0], device=labels.device)

    if exec_heads is not None:
        # --- Execution BCE heads: exec_move, sprint, kick, tackle_attempt ---
        sprint_loss        = _bce(exec_heads.sprint_logit,          _I_SPRINT)
        move_loss          = _bce(exec_heads.exec_move_logit,       _I_EXEC_MOVE)
        tackle_attempt_loss = _bce(exec_heads.tackle_attempt_logit, _I_TACKLE_ATTEMPT, pos_weight_tackle_attempt)
        exec_bce_loss = (
            move_loss
            + sprint_loss
            + _bce(exec_heads.kick_logit,               _I_KICK_THIS_TICK, pos_weight_kick)
            + tackle_attempt_loss
        )
        loss += exec_weight * exec_bce_loss

        # --- Execution: move_direction cosine loss ---
        # Upweighted by direction_loss_weight so the single continuous head gets
        # proportional gradient against ~11 Bernoulli heads.
        # 1 - cosine_similarity → 0 when aligned, 2 when opposite.
        has_dir = (labels[:, _I_DIR_X].abs() + labels[:, _I_DIR_Y].abs()) > 1e-6
        if has_dir.any():
            target_dir = labels[:, _I_DIR_X:_I_DIR_Y + 1]  # (N, 2)
            pred_dir = exec_heads.move_direction             # (N, 2) raw (pre-normalize)
            eps = 1e-6
            pred_norm = pred_dir / (pred_dir.norm(dim=-1, keepdim=True) + eps)
            cos_loss = 1.0 - (pred_norm * target_dir).sum(dim=-1)
            dir_loss_per = direction_loss_weight * torch.where(has_dir, cos_loss, torch.zeros_like(cos_loss))
            loss += exec_weight * dir_loss_per

    # --- Decision: move_region_center MSE (normalised to [-1, 1]) ---
    # Weighted by region_loss_weight (default 1.0, separate from direction).
    # In Phase 1 the region target is a noisy proxy for ball position; keep
    # weight low so it does not inflate the BC loss floor unnecessarily.
    region_loss_per = torch.zeros(labels.shape[0], device=labels.device)
    has_region = (labels[:, _I_REGION_X].abs() + labels[:, _I_REGION_Y].abs()) > 1e-6
    if has_region.any():
        pitch_scale = torch.tensor(
            [[_PITCH_HALF_LENGTH_M, _PITCH_HALF_WIDTH_M]], device=labels.device
        )
        target_region = labels[:, _I_REGION_X:_I_REGION_Y + 1]  # (N, 2) physical m
        target_norm = (target_region / pitch_scale).clamp(-1.0, 1.0)
        pred_norm_region = torch.tanh(decision_heads.move_region_center)  # (N, 2) in [-1,1]
        region_mse = ((pred_norm_region - target_norm) ** 2).sum(dim=-1)
        region_loss_per = region_loss_weight * torch.where(has_region, region_mse, torch.zeros_like(region_mse))
        loss += region_loss_per

    # Mask to valid steps only and mean
    valid_loss = loss[valid]
    total = valid_loss.mean() if len(valid_loss) > 0 else _zero

    if return_breakdown:
        breakdown = {
            "decision":       float(dec_loss[valid].mean()),
            "exec_bce":       float(exec_bce_loss[valid].mean()),
            "sprint":         float(sprint_loss[valid].mean()),
            "move":           float(move_loss[valid].mean()),
            "tackle_attempt": float(tackle_attempt_loss[valid].mean()),
            "direction":      float(dir_loss_per[valid].mean()),
            "region":         float(region_loss_per[valid].mean()),
        }
        return total, breakdown
    return total


# ---------------------------------------------------------------------------
# Pre-training
# ---------------------------------------------------------------------------

class BCPretrainer:
    """Runs a short supervised pre-training phase before PPO begins.

    For each step:
      1. Encode the current match observation.
      2. Compute rules-based BC labels for the trainee.
      3. Forward pass both networks.
      4. Compute BC loss; backprop; optimizer step.
      5. Advance one decision interval in the env (using the label's implied
         action, NOT the network's sampled action) so the env state changes.

    The env is stepped with a dummy rules-based action derived from the label
    so the trainee actually moves toward the ball, producing a varied stream
    of states rather than replaying the same initial state forever.
    """

    def __init__(self, decision_net, execution_net, cfg: dict, device: torch.device):
        self.decision_net = decision_net
        self.execution_net = execution_net
        self.device = device
        bc_cfg = cfg.get("bc", {})
        lr = float(bc_cfg.get("bc_learning_rate", bc_cfg.get("pretrain_lr", 1e-3)))
        self._online_batch_size = int(bc_cfg.get("bc_online_batch_size", bc_cfg.get("pretrain_online_batch_size", 16)))
        all_params = (
            list(decision_net.parameters()) + list(execution_net.parameters())
        )
        self.optimizer = torch.optim.Adam(all_params, lr=lr, eps=1e-5)

    def pretrain(
        self,
        env,
        n_steps: int,
        label_fn: Callable,
        dataset=None,
        n_epochs: int = 1,
        batch_size: int = 64,
    ) -> None:
        """Run BC pre-training.

        Two modes:
          - **Online** (default, ``dataset=None``): step the env with rules-based
            AI, compute labels on-the-fly, one gradient step per env step.
            Simple but noisy (single-sample updates, oscillates on episode reset).
          - **Offline** (``dataset`` provided): sample minibatches from a pre-recorded
            ``DemonstrationDataset`` for ``n_epochs`` epochs of ``n_steps`` steps.
            Stable, low-variance gradients; decoupled from env randomness.

        Args:
            env: ScenarioEnv — used only in online mode.
            n_steps: Online steps OR offline steps-per-epoch.
            label_fn: Callable(env) -> BCLabel — online mode only.
            dataset: Optional DemonstrationDataset for offline mode.
            n_epochs: Offline mode only — number of passes over the dataset.
            batch_size: Offline mode only — minibatch size.
        """
        if n_steps <= 0:
            return

        if dataset is not None:
            self._pretrain_offline(dataset, n_steps, n_epochs, batch_size)
        else:
            self._pretrain_online(env, n_steps, label_fn)

    def _pretrain_offline(self, dataset, n_steps: int, n_epochs: int, batch_size: int) -> None:
        """Offline BC pre-training from a DemonstrationDataset."""
        log.info(
            f"BC pre-training (offline): {n_epochs} epoch(s), "
            f"batch_size={batch_size}, dataset={len(dataset):,} steps"
        )
        import torch.nn as nn

        total_steps_done = 0
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            for obs_dict, labels in dataset.iterate_minibatches(
                batch_size=batch_size, shuffle=True, device=self.device, valid_only=True
            ):
                self.optimizer.zero_grad()
                _sat, _oat = obs_dict.get("self_ai_type"), obs_dict.get("other_ai_type")
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
                loss = bc_loss_from_tensor(labels, d_heads, e_heads)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.decision_net.parameters()) + list(self.execution_net.parameters()),
                    1.0,
                )
                self.optimizer.step()
                epoch_loss += float(loss.item())
                epoch_batches += 1
                total_steps_done += 1

            mean_loss = epoch_loss / max(1, epoch_batches)
            log.info(
                f"BC pre-train epoch {epoch + 1}/{n_epochs} | "
                f"mean_bc_loss={mean_loss:.4f} ({epoch_batches} batches)"
            )

        log.info("BC pre-training (offline) complete.")

    def _pretrain_online(self, env, n_steps: int, label_fn: Callable) -> None:
        """Online BC pre-training: accumulate a mini-batch, then gradient step.

        Collects ``pretrain_online_batch_size`` (obs, label) pairs before each
        update.  This gives low-variance gradients at the cost of slightly
        fewer updates per step, which is a much better trade-off than the
        previous 1-sample-per-update scheme.
        """
        batch_size = max(1, self._online_batch_size)
        log.info(f"BC pre-training (online): {n_steps} steps, batch_size={batch_size}")
        import torch.nn as nn
        from footballcoach.rules_ai import Phase1RulesAI
        obs = env.reset()
        try:
            env._loop.match.player_by_id(env.trainee_player_id).ai = Phase1RulesAI()
        except (AttributeError, KeyError):
            pass
        total_loss = 0.0
        valid_steps = 0

        # Accumulators for the current mini-batch
        batch_obs: list[dict] = []
        batch_labels: list = []

        for step in range(n_steps):
            label = label_fn(env)
            obs_dict = {k: v.to(self.device) for k, v in obs.to_torch_dict().items()}
            batch_obs.append(obs_dict)
            batch_labels.append(torch.from_numpy(label.to_array()).to(self.device))
            if label.valid:
                valid_steps += 1

            # Gradient step once the mini-batch is full (or at the last step)
            if len(batch_obs) >= batch_size or step == n_steps - 1:
                # Stack into batched tensors
                stacked = {
                    k: torch.stack([b[k] for b in batch_obs], dim=0)
                    for k in batch_obs[0]
                }
                label_t = torch.stack(batch_labels, dim=0)

                self.optimizer.zero_grad()
                _sat, _oat = stacked.get("self_ai_type"), stacked.get("other_ai_type")
                d_heads = self.decision_net(
                    stacked["self_feat"], stacked["other_feat"],
                    stacked["exists_mask"], stacked["ball_feat"], stacked["global_feat"],
                    _sat, _oat,
                )
                e_heads = self.execution_net(
                    stacked["self_feat"], stacked["other_feat"],
                    stacked["exists_mask"], stacked["ball_feat"], stacked["global_feat"],
                    d_heads, _sat, _oat,
                )
                loss = bc_loss_from_tensor(label_t, d_heads, e_heads)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.decision_net.parameters()) + list(self.execution_net.parameters()),
                    1.0,
                )
                self.optimizer.step()
                total_loss += float(loss.item())
                batch_obs.clear()
                batch_labels.clear()

            next_obs, _reward, done, _info = env.step()
            if done:
                obs = env.reset()
                try:
                    env._loop.match.player_by_id(env.trainee_player_id).ai = Phase1RulesAI()
                except (AttributeError, KeyError):
                    pass
            else:
                obs = next_obs

            if (step + 1) % 200 == 0:
                updates = (step + 1) // batch_size
                mean_loss = total_loss / max(1, updates)
                log.info(
                    f"BC pre-train step {step + 1}/{n_steps} | "
                    f"mean_bc_loss={mean_loss:.4f} (over {valid_steps} valid steps)"
                )
                total_loss = 0.0
                valid_steps = 0

        log.info("BC pre-training complete.")


def _label_to_env_action(label: BCLabel, env) -> dict:
    """Build a minimal env action dict from a BCLabel so the env can be
    stepped during pre-training.  The trainee's order is set by the env's
    apply_action_to_player via to_orders.py — we just need plausible
    decision_probs and execution_physical.
    """
    from footballcoach.ai.action.schema import ExecutionAction

    # Derive decision probs from label (0/1 → 0.1/0.9 to avoid log(0) in any
    # downstream callers; the env's gating uses a 0.5 threshold anyway).
    def _prob(v: float) -> float:
        return 0.9 if v > 0.5 else 0.1

    direction = label.move_direction
    if direction is None:
        direction = np.array([0.0, 0.0], dtype=np.float32)

    return {
        "decision_probs": {
            "shoot":          _prob(label.shoot),
            "pass_":          _prob(label.pass_),
            "move":           _prob(label.move),
            "tackle":         _prob(label.tackle),
            "get_possession": _prob(label.get_possession_extra),
            "mark":           _prob(label.mark),
            "hold_position":  _prob(label.hold_position),
        },
        "execution_physical": {
            "move_direction":    direction,
            "sprint":            label.sprint > 0.5,
            "kick_this_tick":    False,
            "kick_direction":    np.array([1.0, 0.0], dtype=np.float32),
            "kick_power_fraction": 0.0,
            "kick_spin":         np.zeros(3, dtype=np.float32),
            "tackle_attempt":    False,
        },
        "decision_physical": {
            "move_region_center_m":   (
                label.move_region_center_m
                if label.move_region_center_m is not None
                else np.zeros(2, dtype=np.float32)
            ),
            "move_region_size_m":     2.0,
            "move_arrival_speed_mps": 7.0,
        },
        "target_slots": {"pass_": 0, "tackle": 0, "mark": 0},
        "slot_player_ids": [None] * 21,
        "decision": None,
        "execution": ExecutionAction(),
    }
