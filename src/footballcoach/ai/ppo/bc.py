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
  [18] kick_dir_x        (unit vector component of the kick target direction;
                          0.0 if not applicable. Read from
                          Player.last_kick_direction, set inside kick_direct().)
  [19] kick_dir_y
  [20] kick_power        (power_fraction actually used for the kick, [0, 1];
                          0.0 if not applicable. Read from
                          Player.last_kick_power_fraction.)
  [21] kick_spin_x       (raw kick spin vector; 0.0 if not applicable. Read
                          from Player.last_kick_spin.)
  [22] kick_spin_y
  [23] kick_spin_z
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

BC_LABEL_DIM = 25  # elements in the flat label vector (see module docstring)

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
_I_KICK_DIR_X     = 18
_I_KICK_DIR_Y     = 19
_I_KICK_POWER     = 20
_I_KICK_SPIN_X    = 21
_I_KICK_SPIN_Y    = 22
_I_KICK_SPIN_Z    = 23
_I_KICK_DIR_Z     = 24  # kick_direction z component (3D unit vector with indices 18, 19, 24)

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
    kick_direction: Optional[np.ndarray] = None      # shape (3,) 3D unit vector of actual launch direction
    kick_power_fraction: Optional[float] = None       # [0, 1], None if not kicking
    kick_spin: Optional[np.ndarray] = None            # shape (3,), None if not kicking

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
        if self.kick_direction is not None:
            arr[_I_KICK_DIR_X] = float(self.kick_direction[0])
            arr[_I_KICK_DIR_Y] = float(self.kick_direction[1])
            arr[_I_KICK_DIR_Z] = float(self.kick_direction[2]) if len(self.kick_direction) > 2 else 0.0
        if self.kick_power_fraction is not None:
            arr[_I_KICK_POWER] = float(self.kick_power_fraction)
        if self.kick_spin is not None:
            arr[_I_KICK_SPIN_X] = float(self.kick_spin[0])
            arr[_I_KICK_SPIN_Y] = float(self.kick_spin[1])
            arr[_I_KICK_SPIN_Z] = float(self.kick_spin[2])
        return arr

    @staticmethod
    def invalid() -> "BCLabel":
        """A label that contributes zero BC loss."""
        return BCLabel(valid=False)


# ---------------------------------------------------------------------------
# Rules-based label generators
# ---------------------------------------------------------------------------

def _ai_type_of(ai) -> float:
    """Classify a ``player.ai`` instance into an AI_TYPE_* code for the
    value-only ``ai_type``/``opponent_ai_type`` side channel. Checked in this
    order: rules-based first (``Phase1RulesAI``), then neural
    (``NeuralPlayerAI`` — also matches ``HybridPlayerAI``, a subclass), else
    immobile (``ai is None`` or anything else)."""
    from footballcoach.rules_ai import Phase1RulesAI, NeuralPlayerAI
    if isinstance(ai, Phase1RulesAI):
        return AI_TYPE_RULES
    if isinstance(ai, NeuralPlayerAI):
        return AI_TYPE_NEURAL
    return AI_TYPE_IMMOBILE


def phase1_labels(env, player_id: str = None) -> BCLabel:
    """Derive BC labels for Phase 1 by asking Phase1RulesAI what it would do.

    Instantiates a temporary Phase1RulesAI, calls act() on the current match
    state, and reads back the order it sets — so the labels are always exactly
    in sync with the rules AI behaviour, with no duplicated logic here.

    All quantities returned here (``move_direction``, ``move_region_center_m``,
    ``kick_direction``, ``kick_spin``, ...) are in raw WORLD/ENGINE-FRAME
    coordinates, matching the world-frame observations from ``obs/encoder.py``.
    No canonical-AI-frame mirroring happens here — that transform is applied
    later, as a thin wrapper around network forward calls (see
    ``ai/obs/canonical.py``), not baked into recorded/generated labels.
    """
    from footballcoach.rules_ai import Phase1RulesAI
    from footballcoach.orders import MoveOrder, GetPossessionOrder, ChaseTackleOrder

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
    # vs neural vs immobile) — NOT which AI the label was derived from (this
    # function always queries Phase1RulesAI internally regardless of the
    # caller).
    ai_type = _ai_type_of(player.ai)

    # opponent_ai_type: which AI controls the OTHER player in the match right
    # now. Phase 1 is strictly 1v1, so "the other player" is unambiguous —
    # any player in match.players that isn't `player`. Falls back to
    # AI_TYPE_IMMOBILE if no other player is found (should not happen in a
    # well-formed 1v1 scenario).
    opponent_ai_type = AI_TYPE_IMMOBILE
    for _p in match.players:
        if _p.player_id != player.player_id:
            opponent_ai_type = _ai_type_of(_p.ai)
            break

    # Execution heads reflect what the rules AI is physically doing RIGHT NOW —
    # i.e. the current order it's executing, before any new decision is made.
    from footballcoach.orders import MoveOrder as _MoveOrder, GetPossessionOrder as _GPOrder
    current_exec = player.current_order
    # kick_this_tick: read Player.kicked_this_tick directly rather than
    # inspecting order types. kick_direct() sets this flag unconditionally
    # whenever it actually executes kick physics — including MoveOrder's
    # push-kick path (rules_ai.py's box-run), which never creates a
    # ShootOrder/KickOrder/PassOrder and was previously missed entirely,
    # causing recorded demonstrations to have zero "kick" labels despite
    # visible kicks in the UI. See Player.kicked_this_tick docstring.
    kick_this_tick = 1.0 if (player.kicked_this_tick or player.kick_armed) else 0.0
    kick_direction = None
    kick_power_fraction = None
    kick_spin = None
    if player.kicked_this_tick:
        # Actual kick this tick — use real post-physics values.
        if player.last_kick_direction is not None:
            kick_direction = np.array(
                [player.last_kick_direction.x, player.last_kick_direction.y, player.last_kick_direction.z],
                dtype=np.float32,
            )
        kick_power_fraction = player.last_kick_power_fraction
        if player.last_kick_spin is not None:
            kick_spin = np.array(
                [player.last_kick_spin.x, player.last_kick_spin.y, player.last_kick_spin.z],
                dtype=np.float32,
            )
    elif player.kick_armed and player.kick_armed_aim_point is not None:
        # Approach tick: arm intent with estimated direction/power toward the pre-computed target.
        _d = player.kick_armed_aim_point - player.position
        _d_len = (_d.x**2 + _d.y**2 + _d.z**2) ** 0.5
        if _d_len > 1e-6:
            kick_direction = np.array([_d.x / _d_len, _d.y / _d_len, _d.z / _d_len], dtype=np.float32)
        kick_power_fraction = player.kick_armed_power_fraction
        kick_spin = np.zeros(3, dtype=np.float32)
    # tackle_attempt: ChaseTackleOrder always, OR GetPossessionOrder when tackle_armed
    # is True — set every tick during the approach by _run_get_possession_behaviour,
    # so BC sees tackle intent across the full approach, not only at contact range.
    _is_gp_tackling = isinstance(current_exec, _GPOrder) and player.tackle_armed
    tackle_attempt = 1.0 if (isinstance(current_exec, ChaseTackleOrder) or _is_gp_tackling) else 0.0

    # Decision heads reflect what the rules AI DECIDES next.
    # Temporarily clear current_order so the AI always produces a fresh decision.
    player.current_order = None
    try:
        Phase1RulesAI().act(player, match, trial_tick=0)
        order = player.current_order
    finally:
        player.current_order = current_exec  # always restore

    # Execution-level fields (move_direction / sprint / exec_move) must NEVER
    # be hand-derived from Order fields (bypasses braking/repulsion/turn-
    # limiting/push-kick logic in _compute_movement_intent()). Instead, run
    # the decided order's execute() once (fully snapshotted/restored so it
    # has ZERO effect on the real match) and read back what actually landed
    # on player.desired_direction/player.desired_speed_mode. See
    # ai/knowledge.md "Orders vs execution-network labels boundary".
    move_direction = None
    sprint_label = 0.0
    exec_move_label = 0.0
    move_region_center = None

    if isinstance(order, (MoveOrder, GetPossessionOrder)):
        # Snapshot everything execute() might mutate, so this exploratory
        # call has ZERO effect on the real simulation.
        _snap_pos = player.position
        _snap_vel = player.velocity
        _snap_heading = player.heading_rad
        _snap_desired_dir = player.desired_direction
        _snap_desired_speed = player.desired_speed_mode
        _snap_kicked_this_tick = player.kicked_this_tick
        _snap_last_kick_dir = player.last_kick_direction
        _snap_last_kick_power = player.last_kick_power_fraction
        _snap_last_kick_spin = player.last_kick_spin
        _snap_current_order = player.current_order
        _snap_on_possession_gained = player.on_possession_gained
        _snap_on_kick = player.on_kick
        _snap_on_tackle = player.on_tackle
        _snap_on_tackle_result = player.on_tackle_result
        _snap_on_auto_tackle_result = player.on_auto_tackle_result
        _snap_ball_possessed_by = match.ball.possessed_by
        _snap_ball_velocity = match.ball.velocity
        _snap_ball_position = match.ball.position
        # order is a FRESH object from Phase1RulesAI().act() above (not
        # current_exec), so its own internal state starts clean.
        player.current_order = order
        # Prevent on_kick/on_tackle/on_tackle_result from firing real callbacks
        # during this exploratory execute() (e.g. record_demonstrations.py wires
        # these to _record_now(), which would otherwise recursively re-enter here).
        player.on_kick = None
        player.on_tackle = None
        player.on_tackle_result = None
        player.on_auto_tackle_result = None
        try:
            _dt = env._dt_s
            order.execute(player, match, _dt)
            if player.desired_speed_mode is not None:
                from footballcoach.engine.movement import SpeedMode
                move_direction_raw = player.desired_direction
                if move_direction_raw.length_xy() > 1e-6:
                    _n = move_direction_raw.normalized()
                    move_direction = np.array([_n.x, _n.y], dtype=np.float32)
                sprint_label = 1.0 if player.desired_speed_mode is SpeedMode.SPRINT else 0.0
                exec_move_label = 0.0 if player.desired_speed_mode is SpeedMode.STANDSTILL else 1.0
        finally:
            # Restore EVERYTHING — this call must be perfectly invisible to
            # the real simulation.
            player.position = _snap_pos
            player.velocity = _snap_vel
            player.heading_rad = _snap_heading
            player.desired_direction = _snap_desired_dir
            player.desired_speed_mode = _snap_desired_speed
            player.kicked_this_tick = _snap_kicked_this_tick
            player.last_kick_direction = _snap_last_kick_dir
            player.last_kick_power_fraction = _snap_last_kick_power
            player.last_kick_spin = _snap_last_kick_spin
            player.current_order = _snap_current_order
            player.on_possession_gained = _snap_on_possession_gained
            player.on_kick = _snap_on_kick
            player.on_tackle = _snap_on_tackle
            player.on_tackle_result = _snap_on_tackle_result
            player.on_auto_tackle_result = _snap_on_auto_tackle_result
            match.ball.possessed_by = _snap_ball_possessed_by
            match.ball.velocity = _snap_ball_velocity
            match.ball.position = _snap_ball_position

    if isinstance(order, MoveOrder):
        move_region_center = np.array(
            [order.target_position.x, order.target_position.y], dtype=np.float32
        )
        return BCLabel(
            move=1.0,
            sprint=sprint_label,
            move_direction=move_direction,
            move_region_center_m=move_region_center,
            kick_this_tick=kick_this_tick,
            tackle_attempt=tackle_attempt,
            exec_move=exec_move_label,
            ai_type=ai_type,
            opponent_ai_type=opponent_ai_type,
            kick_direction=kick_direction,
            kick_power_fraction=kick_power_fraction,
            kick_spin=kick_spin,
        )
    elif isinstance(order, GetPossessionOrder):
        ball = match.ball
        move_region_center = np.array([ball.position.x, ball.position.y], dtype=np.float32)
        return BCLabel(
            get_possession_extra=1.0,
            sprint=sprint_label,
            move_direction=move_direction,
            move_region_center_m=move_region_center,
            kick_this_tick=kick_this_tick,
            tackle_attempt=tackle_attempt,
            exec_move=exec_move_label,
            ai_type=ai_type,
            opponent_ai_type=opponent_ai_type,
            kick_direction=kick_direction,
            kick_power_fraction=kick_power_fraction,
            kick_spin=kick_spin,
        )
    else:
        return BCLabel.invalid()


def phase1_labels_from_teacher(env, teacher_trainer, player_id: str = None) -> BCLabel:
    """Derive BC labels for Phase 1 from a neural teacher's own forward pass,
    instead of ``phase1_labels()``'s Phase1RulesAI order-simulation
    counterfactual.

    Every head — decision Bernoulli probabilities, move_direction/
    kick_direction unit vectors, sprint/exec_move/kick_this_tick/
    tackle_attempt probabilities, kick_power, kick_spin, move_region_center —
    comes directly from ONE forward call through
    ``teacher_trainer.decision_net``/``teacher_trainer.execution_net``, fully
    decoupled from whatever ``player.ai`` is physically doing to *player*
    this tick. This sidesteps the real-state-vs-counterfactual gap
    ``phase1_labels()``'s ``kick_this_tick``/``tackle_attempt`` fields have
    (those read real player/order state because Phase1RulesAI only speaks in
    Orders, not raw action probabilities, so it needs a snapshot/restore
    sandbox to get a genuine counterfactual for move/sprint but not for
    kick/tackle — see that function's docstring and
    ai/knowledge.md "Orders vs execution-network labels boundary"). A neural
    teacher has no such gap: every head is read straight off its own output
    tensors, so kick/tackle labels are just as counterfactual as move/sprint.

    Labels are SOFT (the teacher's own sigmoid probabilities / raw outputs,
    not hard 0/1) — ``bc_loss_from_tensor()``'s BCE/cosine/MSE terms accept
    continuous targets directly, no BCLabel schema change needed.

    This is the intended mechanism for neural-to-neural distillation /
    DAgger-style dataset generation: whoever/whatever physically drives
    *player* (rules AI, immobile, or another neural checkpoint via
    ``record_demonstrations.py --driver-checkpoint``) determines which states
    get visited; ``teacher_trainer`` supplies what to imitate at each visited
    state, independent of the driver's identity — so this works whether the
    driver and teacher are the same checkpoint, different checkpoints (e.g.
    across an architecture change), or the driver is rules-based/immobile.

    ``teacher_trainer`` must be a ``PPOTrainer`` with real ``decision_net``/
    ``execution_net`` weights loaded (e.g. via
    ``PPOTrainer.load_for_inference()``) — called directly, under
    ``torch.no_grad()``, regardless of its own ``.training``/``.eval()``
    mode.

    All quantities returned here are decanonicalized back to raw WORLD/
    ENGINE-FRAME coordinates before storage, matching ``phase1_labels()``'s
    own convention (the network's raw outputs are in CANONICAL frame per
    ``CanonicalNetworkWrapper`` — see ai/knowledge.md "Canonical AI frame") —
    the stored dataset stays plain world-frame; ``canonicalize_bc_labels()``
    re-canonicalizes at consumption time, same as every other label source.
    """
    from footballcoach.ai.obs.canonical import x_sign_of, mirror_x

    try:
        match = env.match
        if player_id is None:
            player_id = env.trainee_player_id
        if match is None:
            return BCLabel.invalid()
        player = match.player_by_id(player_id)
    except (KeyError, AttributeError):
        return BCLabel.invalid()

    ai_type = _ai_type_of(player.ai)
    opponent_ai_type = AI_TYPE_IMMOBILE
    for _p in match.players:
        if _p.player_id != player.player_id:
            opponent_ai_type = _ai_type_of(_p.ai)
            break

    obs = env._get_obs(player_id=player_id)
    device = next(teacher_trainer.decision_net.parameters()).device
    obs_dict = {k: v.unsqueeze(0).to(device) for k, v in obs.to_torch_dict().items()}
    x_sign = float(x_sign_of(obs.self_feat))

    with torch.no_grad():
        d_heads = teacher_trainer.decision_net(
            obs_dict["self_feat"], obs_dict["other_feat"], obs_dict["exists_mask"],
            obs_dict["ball_feat"], obs_dict["global_feat"],
            obs_dict["self_ai_type"], obs_dict["other_ai_type"],
        )
        e_heads = teacher_trainer.execution_net(
            obs_dict["self_feat"], obs_dict["other_feat"], obs_dict["exists_mask"],
            obs_dict["ball_feat"], obs_dict["global_feat"],
            d_heads, obs_dict["self_ai_type"], obs_dict["other_ai_type"],
        )

    move_direction = mirror_x(e_heads.move_direction.squeeze(0), x_sign).cpu().numpy().astype(np.float32)
    kick_direction = mirror_x(e_heads.kick_direction.squeeze(0), x_sign).cpu().numpy().astype(np.float32)
    mv_center_phys = torch.tanh(d_heads.move_region_center) * torch.tensor(
        [[_PITCH_HALF_LENGTH_M, _PITCH_HALF_WIDTH_M]], device=device
    )
    move_region_center = mirror_x(mv_center_phys.squeeze(0), x_sign).cpu().numpy().astype(np.float32)
    # kick_spin is left unmirrored, matching PPOTrainer._sample_action()'s own
    # direct-drive decanonicalize step (which mirrors move_direction/
    # kick_direction but not kick_spin) — kick_spin's PPO sampling is
    # currently frozen (see agent_plans/spin_implementation_plan.md section 0)
    # and no call site in the codebase mirrors it, so this stays consistent
    # rather than introducing a new, unvalidated transform.
    kick_spin = e_heads.kick_spin.squeeze(0).cpu().numpy().astype(np.float32)

    return BCLabel(
        shoot=float(torch.sigmoid(d_heads.shoot_logit)),
        pass_=float(torch.sigmoid(d_heads.pass_logit)),
        move=float(torch.sigmoid(d_heads.move_logit)),
        tackle=float(torch.sigmoid(d_heads.tackle_logit)),
        get_possession_extra=float(torch.sigmoid(d_heads.get_possession_raw)),
        mark=float(torch.sigmoid(d_heads.mark_logit)),
        hold_position=float(torch.sigmoid(d_heads.hold_position_logit)),
        move_direction=move_direction,
        sprint=float(torch.sigmoid(e_heads.sprint_logit)),
        move_region_center_m=move_region_center,
        kick_this_tick=float(torch.sigmoid(e_heads.kick_logit)),
        tackle_attempt=float(torch.sigmoid(e_heads.tackle_attempt_logit)),
        exec_move=float(torch.sigmoid(e_heads.exec_move_logit)),
        valid=True,
        ai_type=ai_type,
        opponent_ai_type=opponent_ai_type,
        kick_direction=kick_direction,
        kick_power_fraction=float(torch.sigmoid(e_heads.kick_power)),
        kick_spin=kick_spin,
    )


# ---------------------------------------------------------------------------
# BC loss floor (analytic minimum achievable loss under label smoothing)
# ---------------------------------------------------------------------------

def compute_bc_loss_floor_components(
    labels: torch.Tensor,
    pos_weight_kick: float = 1.0,
    pos_weight_tackle_attempt: float = 1.0,
    dec_label_smoothing: float = 0.0,
    exec_label_smoothing: float = 0.0,
    has_exec: bool = True,
) -> dict:
    """Per-component analytic BC loss floors, unweighted — i.e. matching the
    scale of ``bc_loss_from_tensor(..., return_breakdown=True)``'s dict
    *before* ``dec_weight``/``exec_weight`` are applied (those breakdown
    entries are raw, unweighted per-group means — see that function).

    Label smoothing (and, for positive rows, ``pos_weight``) makes the
    BCE-optimal prediction ``p* = y'`` (the smoothed target) rather than the
    hard 0/1 label, so the achievable minimum loss per smoothed Bernoulli
    head is no longer 0 — it's the (weighted) binary entropy of the smoothed
    target: ``H(y') = -y'*ln(y') - (1-y')*ln(1-y')``, scaled by
    ``pos_weight`` on positive-labelled rows (matching
    ``F.binary_cross_entropy_with_logits``'s ``pos_weight`` semantics, which
    multiplies the WHOLE per-sample loss on positive rows, not just the
    ``-log(p)`` term). Note ``H(y')`` is symmetric in the hard label ``y``
    (``H(0.5*eps) == H(1-0.5*eps)``), so each head's floor depends only on
    its smoothing constant (and mildly on ``pos_weight``), not on how
    balanced its labels are — this is what makes per-component floors cheap
    to reason about without touching the underlying dataset.

    Only the smoothed Bernoulli BCE components (``decision``, ``exec_bce``,
    and its constituents ``sprint``/``move``/``tackle_attempt``/``kick``)
    have a nonzero floor — the cosine (``direction``/``kick_direction``) and
    MSE (``region``/``kick_power``/``kick_spin``) components have a true
    floor of 0 regardless of smoothing, since they aren't binary-target BCE.

    Subtracting these per-component floors from a logged breakdown (e.g.
    ppo_trainer.py's per-epoch "breakdown  decision=... exec_bce=..." line)
    keeps components directly comparable to each other and to zero-floor
    components like ``direction`` — without it, summed multi-head terms
    (``exec_bce`` sums 4 smoothed heads) look disproportionately large next
    to unfloored ones purely from the smoothing offset, not real residual
    error. See ai_trainer_knowledge.md's BC loss / label smoothing
    discussion, and ``compute_bc_loss_floor`` below (the pre-existing
    single-scalar total, now a thin wrapper over this function).

    Args:
        labels: (N, BC_LABEL_DIM) float32 tensor, same batch passed to
            ``bc_loss_from_tensor``.
        has_exec: whether the caller passed a real ``exec_heads`` (i.e.
            Phase 1/2/3, not Phase 0's decision-only path) — mirrors
            ``bc_loss_from_tensor``'s ``exec_heads is None`` branch.

    Returns:
        Dict with keys {decision, exec_bce, sprint, move, tackle_attempt,
        kick, direction, region, kick_direction, kick_power, kick_spin} —
        the last five always 0.0 (true zero floor). All-zero dict if no
        valid rows.
    """
    zero = {
        "decision": 0.0, "exec_bce": 0.0, "sprint": 0.0, "move": 0.0,
        "tackle_attempt": 0.0, "direction": 0.0, "region": 0.0,
        "kick": 0.0, "kick_direction": 0.0, "kick_power": 0.0, "kick_spin": 0.0,
    }
    valid = labels[:, _I_VALID] > 0.5
    if not valid.any():
        return zero

    def _floor_bce(col: int, pos_weight: float = 1.0, smoothing: float = 0.0) -> torch.Tensor:
        y = labels[:, col]
        if smoothing <= 0.0:
            return torch.zeros_like(y)
        y_prime = y * (1.0 - smoothing) + 0.5 * smoothing
        eps = 1e-12
        h = -(
            y_prime * torch.log(y_prime.clamp_min(eps))
            + (1.0 - y_prime) * torch.log((1.0 - y_prime).clamp_min(eps))
        )
        if pos_weight != 1.0:
            w = torch.where(y > 0.5, torch.full_like(y, pos_weight), torch.ones_like(y))
            h = h * w
        return h

    dec_floor = (
        _floor_bce(_I_SHOOT,    smoothing=dec_label_smoothing)
        + _floor_bce(_I_PASS,    smoothing=dec_label_smoothing)
        + _floor_bce(_I_MOVE,    smoothing=dec_label_smoothing)
        + _floor_bce(_I_TACKLE,  smoothing=dec_label_smoothing)
        + _floor_bce(_I_GP_EXTRA, smoothing=dec_label_smoothing)
        + _floor_bce(_I_MARK,    smoothing=dec_label_smoothing)
        + _floor_bce(_I_HOLD,    smoothing=dec_label_smoothing)
    )
    result = dict(zero)
    result["decision"] = float(dec_floor[valid].mean())

    if has_exec:
        move_floor = _floor_bce(_I_EXEC_MOVE, smoothing=exec_label_smoothing)
        sprint_floor = _floor_bce(_I_SPRINT, smoothing=exec_label_smoothing)
        kick_floor = _floor_bce(_I_KICK_THIS_TICK, pos_weight_kick, smoothing=exec_label_smoothing)
        tackle_attempt_floor = _floor_bce(
            _I_TACKLE_ATTEMPT, pos_weight_tackle_attempt, smoothing=exec_label_smoothing
        )
        result["move"] = float(move_floor[valid].mean())
        result["sprint"] = float(sprint_floor[valid].mean())
        result["kick"] = float(kick_floor[valid].mean())
        result["tackle_attempt"] = float(tackle_attempt_floor[valid].mean())
        result["exec_bce"] = float(
            (move_floor + sprint_floor + kick_floor + tackle_attempt_floor)[valid].mean()
        )

    return result


def compute_bc_loss_floor(
    labels: torch.Tensor,
    pos_weight_kick: float = 1.0,
    pos_weight_tackle_attempt: float = 1.0,
    dec_weight: float = 1.0,
    exec_weight: float = 1.0,
    dec_label_smoothing: float = 0.0,
    exec_label_smoothing: float = 0.0,
    has_exec: bool = True,
) -> float:
    """Analytic minimum achievable ``bc_loss_from_tensor`` value for this batch.

    Thin wrapper over ``compute_bc_loss_floor_components`` (see its
    docstring for the ``H(y')`` derivation) that applies ``dec_weight``/
    ``exec_weight`` and sums to the single scalar comparable against
    ``bc_loss_from_tensor``'s (weighted) total, useful as
    ``bc_adj = bc_loss - floor`` so epoch-to-epoch / config-to-config loss
    comparisons aren't confounded by a smoothing-only additive offset (see
    ai_trainer_knowledge.md discussion on BC loss and label smoothing).

    Args:
        labels: (N, BC_LABEL_DIM) float32 tensor, same batch passed to
            ``bc_loss_from_tensor``.
        has_exec: whether the caller passed a real ``exec_heads`` (i.e.
            Phase 1/2/3, not Phase 0's decision-only path) — mirrors
            ``bc_loss_from_tensor``'s ``exec_heads is None`` branch.

    Returns:
        Scalar float floor value, or 0.0 if no valid rows.
    """
    components = compute_bc_loss_floor_components(
        labels,
        pos_weight_kick=pos_weight_kick,
        pos_weight_tackle_attempt=pos_weight_tackle_attempt,
        dec_label_smoothing=dec_label_smoothing,
        exec_label_smoothing=exec_label_smoothing,
        has_exec=has_exec,
    )
    total_floor = dec_weight * components["decision"]
    if has_exec:
        total_floor += exec_weight * components["exec_bce"]
    return total_floor


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
    dec_label_smoothing: float = 0.0,
    exec_label_smoothing: float = 0.0,
    split_kick: bool = False,
    split_tackle: bool = False,
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
        dec_label_smoothing: Label smoothing applied to the 7 decision Bernoulli
            targets (shoot/pass/move/tackle/gp_extra/mark/hold) before BCE:
            target = target*(1-eps) + 0.5*eps. 0.0 (default) = no smoothing.
            From ai_config.json['bc']['dec_label_smoothing']. Softens
            overconfident BC-primed decision probabilities.
        exec_label_smoothing: Same, applied to the 4 execution Bernoulli
            targets (exec_move, sprint, kick_this_tick, tackle_attempt).
            Ignored when exec_heads is None. From
            ai_config.json['bc']['exec_label_smoothing']. Motivated by PPO's
            per-minibatch KL early-stop firing on minibatch 0 of the very
            first rollout — exec_move/sprint sit at confident (~0.68-0.79)
            post-BC probabilities where Bernoulli KL is highly nonlinear, so
            even a small logit shift from the first gradient step overshoots
            target_kl. Smoothing these targets toward 0.5 pre-PPO reduces
            that initial overconfidence.
        return_breakdown: If True, also return a dict of per-group mean losses
            {decision, exec_bce, sprint, move, direction, region} for diagnostics.
            When exec_heads is None, the exec-dependent entries are all 0.0.
        split_kick: If True, also return a dict of two *differentiable*
            (non-detached) group-loss tensors: {"kick_group_loss",
            "other_group_loss"}. "kick_group_loss" is the mean (over valid
            rows) of kick_this_tick BCE + kick_direction cosine + kick_power
            MSE + kick_spin MSE, each already scaled by exec_weight;
            "other_group_loss" is everything else in `total` (decision BCEs,
            move_region_center MSE, exec_move/sprint/tackle_attempt BCE,
            move_direction cosine). Unlike return_breakdown's dict (whose
            values are `float(...)`-cast and carry no gradient — diagnostic
            only), these are plain tensors still attached to the autograd
            graph, so a caller can do
            ``coeff_kick * split["kick_group_loss"] + coeff_other *
            split["other_group_loss"]`` and backprop through it. Combine
            freely with return_breakdown and/or split_tackle (independent
            flags) — when either split flag is set, one extra dict is
            appended to the return tuple (see Returns below); "other_group_loss"
            always excludes whichever of the kick/tackle groups are split out.
        split_tackle: If True, also return "tackle_group_loss" in the same
            split dict as split_kick (created even if split_kick is False):
            the mean (over valid rows) of the tackle_attempt BCE term alone,
            scaled by exec_weight — tackle has no direction/power/spin
            sub-heads the way kicking does, so this is a single-term group.
            Same differentiable-tensor rationale as split_kick.

    Returns:
        Scalar BC loss (mean over valid steps). If return_breakdown and/or
        (split_kick or split_tackle) are True, additionally returns their
        dict(s) in that order: (loss[, breakdown_dict][, split_dict]).
        split_dict always has "other_group_loss"; "kick_group_loss" is
        present iff split_kick, "tackle_group_loss" iff split_tackle.
        Returns zero tensor(s) if no valid steps.
    """
    valid = labels[:, _I_VALID] > 0.5  # (N,) bool
    _zero = torch.zeros(1, device=labels.device)
    if not valid.any():
        result = (_zero,)
        if return_breakdown:
            result += ({
                "decision": 0.0, "exec_bce": 0.0, "sprint": 0.0, "move": 0.0,
                "tackle_attempt": 0.0, "direction": 0.0, "region": 0.0,
                "kick": 0.0, "kick_direction": 0.0, "kick_power": 0.0, "kick_spin": 0.0,
            },)
        if split_kick or split_tackle:
            _zero_split = {"other_group_loss": _zero}
            if split_kick:
                _zero_split["kick_group_loss"] = _zero
            if split_tackle:
                _zero_split["tackle_group_loss"] = _zero
            result += (_zero_split,)
        return result if len(result) > 1 else result[0]

    loss = torch.zeros(labels.shape[0], device=labels.device)

    # --- Bernoulli decision heads (BCE from logits) ---
    def _bce(logit: torch.Tensor, col: int, pos_weight: float = 1.0, smoothing: float = 0.0) -> torch.Tensor:
        target = labels[:, col]
        smoothed_target = target * (1.0 - smoothing) + 0.5 * smoothing if smoothing > 0.0 else target
        loss = F.binary_cross_entropy_with_logits(
            logit.squeeze(-1), smoothed_target, reduction="none"
        )
        # pos_weight scales loss on HARD-positive rows only (target > 0.5, pre-smoothing) —
        # torch's built-in pos_weight= multiplies the (possibly smoothed) target itself,
        # so raising pos_weight also inflates the optimal sigmoid for true negatives
        # (their smoothed target has a nonzero residual y=0.5*smoothing) and destroys
        # precision. Weighting by the hard label avoids that leak.
        if pos_weight != 1.0:
            w = torch.where(target > 0.5, torch.full_like(target, pos_weight), torch.ones_like(target))
            loss = loss * w
        return loss

    dec_loss = (
        _bce(decision_heads.shoot_logit,          _I_SHOOT,   smoothing=dec_label_smoothing)
        + _bce(decision_heads.pass_logit,          _I_PASS,    smoothing=dec_label_smoothing)
        + _bce(decision_heads.move_logit,          _I_MOVE,    smoothing=dec_label_smoothing)
        + _bce(decision_heads.tackle_logit,        _I_TACKLE,  smoothing=dec_label_smoothing)
        + _bce(decision_heads.get_possession_raw,  _I_GP_EXTRA, smoothing=dec_label_smoothing)
        + _bce(decision_heads.mark_logit,          _I_MARK,    smoothing=dec_label_smoothing)
        + _bce(decision_heads.hold_position_logit, _I_HOLD,    smoothing=dec_label_smoothing)
    )
    loss += dec_weight * dec_loss

    exec_bce_loss = torch.zeros(labels.shape[0], device=labels.device)
    sprint_loss = torch.zeros(labels.shape[0], device=labels.device)
    move_loss = torch.zeros(labels.shape[0], device=labels.device)
    tackle_attempt_loss = torch.zeros(labels.shape[0], device=labels.device)
    kick_loss = torch.zeros(labels.shape[0], device=labels.device)
    dir_loss_per = torch.zeros(labels.shape[0], device=labels.device)
    kick_dir_loss_per = torch.zeros(labels.shape[0], device=labels.device)
    kick_power_loss_per = torch.zeros(labels.shape[0], device=labels.device)
    kick_spin_loss_per = torch.zeros(labels.shape[0], device=labels.device)

    if exec_heads is not None:
        # --- Execution BCE heads: exec_move, sprint, kick, tackle_attempt ---
        sprint_loss        = _bce(exec_heads.sprint_logit,          _I_SPRINT,          smoothing=exec_label_smoothing)
        move_loss          = _bce(exec_heads.exec_move_logit,       _I_EXEC_MOVE,       smoothing=exec_label_smoothing)
        tackle_attempt_loss = _bce(exec_heads.tackle_attempt_logit, _I_TACKLE_ATTEMPT, pos_weight_tackle_attempt, smoothing=exec_label_smoothing)
        kick_loss           = _bce(exec_heads.kick_logit,           _I_KICK_THIS_TICK, pos_weight_kick, smoothing=exec_label_smoothing)
        exec_bce_loss = (
            move_loss
            + sprint_loss
            + kick_loss
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

        # --- Execution: kick_direction cosine loss (3D, only on kick ticks) ---
        has_kick_dir = (
            (labels[:, _I_KICK_DIR_X].abs() + labels[:, _I_KICK_DIR_Y].abs() + labels[:, _I_KICK_DIR_Z].abs()) > 1e-6
        ) & (labels[:, _I_KICK_THIS_TICK] > 0.5)
        if has_kick_dir.any():
            eps = 1e-6
            target_kdir = torch.stack(
                [labels[:, _I_KICK_DIR_X], labels[:, _I_KICK_DIR_Y], labels[:, _I_KICK_DIR_Z]], dim=-1
            )  # (N, 3)
            pred_kdir = exec_heads.kick_direction  # (N, 3) raw
            pred_kdir_norm = pred_kdir / (pred_kdir.norm(dim=-1, keepdim=True) + eps)
            kdir_cos_loss = 1.0 - (pred_kdir_norm * target_kdir).sum(dim=-1)
            kick_dir_loss_per = direction_loss_weight * torch.where(
                has_kick_dir, kdir_cos_loss, torch.zeros_like(kdir_cos_loss)
            )
            loss += exec_weight * kick_dir_loss_per

        # --- Execution: kick_power (MSE) and kick_spin (MSE, normalized) ---
        kicked_mask = labels[:, _I_KICK_THIS_TICK] > 0.5
        if kicked_mask.any():
            pred_power = torch.sigmoid(exec_heads.kick_power.squeeze(-1))
            target_power = labels[:, _I_KICK_POWER]
            power_mse = (pred_power - target_power) ** 2
            kick_power_loss_per = torch.where(kicked_mask, power_mse, torch.zeros_like(power_mse))
            loss += exec_weight * kick_power_loss_per

            spin_norm_max = 30.0  # matches ai_config.json obs['ball_spin_norm_max_rad_s']
            target_spin = labels[:, _I_KICK_SPIN_X:_I_KICK_SPIN_Z + 1] / spin_norm_max
            pred_spin = exec_heads.kick_spin / spin_norm_max
            spin_mse = ((pred_spin - target_spin) ** 2).sum(dim=-1)
            kick_spin_loss_per = torch.where(kicked_mask, spin_mse, torch.zeros_like(spin_mse))
            loss += exec_weight * kick_spin_loss_per

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

    split = None
    if split_kick or split_tackle:
        # The per-row component tensors below (kick_loss, kick_dir_loss_per,
        # kick_power_loss_per, kick_spin_loss_per, tackle_attempt_loss) are
        # already zeroed on non-applicable rows by the torch.where(...) gates
        # above -- exactly the row-level structure that return_breakdown's
        # detached scalar means throw away. Scale each by exec_weight
        # (matching how they were folded into `loss` above) and sum per-row
        # *before* reducing to a mean, so the resulting scalars stay attached
        # to the autograd graph.
        kick_group_per_row = torch.zeros(labels.shape[0], device=labels.device)
        tackle_group_per_row = torch.zeros(labels.shape[0], device=labels.device)
        if split_kick:
            kick_group_per_row = exec_weight * (
                kick_loss + kick_dir_loss_per + kick_power_loss_per + kick_spin_loss_per
            )
        if split_tackle:
            tackle_group_per_row = exec_weight * tackle_attempt_loss
        other_group_per_row = loss - kick_group_per_row - tackle_group_per_row

        def _group_mean(per_row: torch.Tensor) -> torch.Tensor:
            v = per_row[valid]
            return v.mean() if len(v) > 0 else _zero

        split = {"other_group_loss": _group_mean(other_group_per_row)}
        if split_kick:
            split["kick_group_loss"] = _group_mean(kick_group_per_row)
        if split_tackle:
            split["tackle_group_loss"] = _group_mean(tackle_group_per_row)

    if return_breakdown:
        # kick_direction/kick_power/kick_spin are gated to zero on non-kick
        # rows (see kicked_mask above) — averaging over ALL valid rows
        # dilutes them ~pos_weight_kick:1 by hard zeros, making genuine loss
        # changes invisible in logs. Report their mean over KICKED rows only
        # (within the valid set) instead, falling back to 0.0 (not NaN) when
        # no kicks occurred in this batch. NOTE: "kick" (the BCE for
        # did/should-kick) has a well-defined target on EVERY row, so it
        # keeps all-valid-rows averaging — do not change that one.
        kicked_valid_mask = valid & (labels[:, _I_KICK_THIS_TICK] > 0.5)
        _n_kicked_valid = int(kicked_valid_mask.sum().item())

        def _kicked_mean(per_row_loss: torch.Tensor) -> float:
            if _n_kicked_valid == 0:
                return 0.0
            return float(per_row_loss[kicked_valid_mask].mean())

        breakdown = {
            "decision":       float(dec_loss[valid].mean()),
            "exec_bce":       float(exec_bce_loss[valid].mean()),
            "sprint":         float(sprint_loss[valid].mean()),
            "move":           float(move_loss[valid].mean()),
            "tackle_attempt": float(tackle_attempt_loss[valid].mean()),
            "direction":      float(dir_loss_per[valid].mean()),
            "region":         float(region_loss_per[valid].mean()),
            "kick":           float(kick_loss[valid].mean()),
            "kick_direction": _kicked_mean(kick_dir_loss_per),
            "kick_power":     _kicked_mean(kick_power_loss_per),
            "kick_spin":      _kicked_mean(kick_spin_loss_per),
        }
        if split_kick or split_tackle:
            return total, breakdown, split
        return total, breakdown
    if split_kick or split_tackle:
        return total, split
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
    stepped during pre-training.  The trainee's physical action is applied
    directly onto the Player object by the env's call into
    apply_nn_action.py::apply_action_to_player() (no Order is ever set) —
    we just need plausible decision_probs and execution_physical.
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
