"""Builds Match instances for the UI's two non-freeplay modes:

- Training mode: one player + ball on a full pitch, no opponent, goals in
  either net are counted and the ball/player reset on a goal.
- Balance scenarios: small, illustrative re-creations of the statistical
  scenarios from tests/balance/, played out live/visually one trial at a
  time (rather than the thousands-of-trials-headless style used by pytest).

These are UI conveniences for *watching* the mechanics that the balance
tests already validate statistically - they are not a replacement for the
pytest balance suite.

AI wiring: each non-trainee player that needs per-tick logic gets a
``PlayerAI`` subclass assigned to ``player.ai`` (from ``footballcoach.rules_ai``).
``Match.step()`` calls ``player.ai.act(player, match, tick)`` automatically —
no ``on_tick`` hook needed for per-player logic.  ``on_tick`` is only used for
rare cross-player coordination (e.g. pass receiver pickup radius check).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from footballcoach.config import load_gameplay_config, load_scenarios_config
from footballcoach.engine.ball_physics import BallPhysicsParams
from footballcoach.engine.kicking import PassingParams, pass_speed_mps
from footballcoach.engine.match import Match
from footballcoach.engine.movement import MovementParams, effective_top_speed
from footballcoach.entities import Ball, Pitch, PlayerAttributes, Team
from footballcoach.entities.player import Player, PlayerAI
from footballcoach.generation import generate_attributes
from footballcoach.mathutils import Vector3
from footballcoach.orders import (
    ChaseTackleOrder,
    GetPossessionOrder,
    JogOrder,
    KickOrder,
    MoveOrder,
    PassOrder,
    SaveOrder,
    ShootOrder,
    StopOrder,
)


# ---------------------------------------------------------------------------
# ScenarioParam
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioParam:
    """Describes one adjustable knob exposed in the SCENARIO_PARAMS UI screen.

    ``name`` must match the kwarg accepted by the corresponding ``build_*``
    function.  ``default`` is used both as the initial value in the UI and
    as the fallback when the kwarg is omitted (preserving backward
    compatibility for callers that call ``build(rng_reduction)`` without
    extra kwargs).
    """
    name: str
    label: str
    min_value: float
    max_value: float
    step: float
    default: float


@dataclass(frozen=True)
class ScenarioChoiceParam:
    """A dropdown-style parameter: one of a fixed list of string options.

    The build function receives the selected string as a kwarg.
    ``default`` must be one of ``choices``.
    """
    name: str
    label: str
    choices: tuple[str, ...]
    default: str


@dataclass(frozen=True)
class ScenarioGroupedChoiceParam:
    """A two-level dropdown-style parameter: options are organised into named
    groups (e.g. checkpoint directories) and the UI first asks the user to
    pick a group, then a value within that group.

    ``groups`` is an ordered tuple of ``(group_label, (value, ...))`` pairs.
    The build function receives the selected value string as a kwarg (the
    same flat value space as ``ScenarioChoiceParam`` — grouping is purely a
    UI convenience for browsing long option lists). ``default`` must be one
    of the values across all groups.
    """
    name: str
    label: str
    groups: tuple[tuple[str, tuple[str, ...]], ...]
    default: str

    def flat_choices(self) -> tuple[str, ...]:
        return tuple(v for _group, values in self.groups for v in values)


@dataclass(frozen=True)
class ScenarioBoolParam:
    """A checkbox parameter (True/False).

    The build function receives a bool kwarg.
    """
    name: str
    label: str
    default: bool


def make_training_match(rng_reduction: float = 0.3) -> Match:
    """One player + ball, full pitch, both goals live, no opponent."""
    from footballcoach.entities.attributes import PlayerAttributes
    pitch = Pitch.standard()
    attrs = PlayerAttributes(
        top_speed=0.78, acceleration=0.78, stamina=0.78,
        kick_precision=0.78, kick_power=0.78, dribbling=0.78,
        ball_control=0.78, tackling=0.78,
    )
    player = Player.create("trainee", Team.LEFT, attrs, position=Vector3(0, 0, 0))
    ball = Ball.at_rest(Vector3(3, 0, 0))
    ui_cfg = load_gameplay_config().get("ui", {})
    return Match(
        pitch=pitch, players=[player], ball=ball,
        rng_reduction=rng_reduction, rng=random.Random(),
        goal_linger_s=ui_cfg.get("goal_linger_s", 3.0),
    )


def discover_all_phase1_checkpoints() -> list[str]:
    """Scan every ``checkpoints/phase1_run*/`` and ``checkpoints/longterm/``
    dir for .pt checkpoints.

    Thin wrapper around ``_discover_checkpoints`` used by both the Phase 1
    UI scenario picker and Training mode's neural-control checkpoint picker
    (see ``App._toggle_training_ai_mode`` in ``ui/app.py``), so the two stay
    in sync without duplicating the discovery/sorting logic.
    """
    import glob as _glob
    import re as _re
    all_run_dirs = sorted(
        _glob.glob("checkpoints/phase1_run*/"),
        key=lambda d: int(m.group(1)) if (m := _re.search(r"phase1_run(\d+)", d)) else -1,
    )
    checkpoints: list[str] = []
    for d in all_run_dirs:
        checkpoints.extend(_discover_checkpoints(d.rstrip("/")))
    checkpoints.extend(_discover_checkpoints("checkpoints/longterm"))
    return checkpoints


def load_trainer_for_ui(checkpoint_path: str):
    """Public wrapper around ``_load_trainer`` for use outside this module
    (e.g. training-mode neural control in ``ui/app.py``). Returns
    ``(trainer_or_None, error_message_or_None)``."""
    return _load_trainer(checkpoint_path)


AnyScenarioParam = ScenarioParam | ScenarioChoiceParam | ScenarioGroupedChoiceParam | ScenarioBoolParam

# Universal params appended to every scenario's params screen by the UI.
# They are NOT forwarded to build(); app.py pops them before calling build().
UNIVERSAL_PARAMS: list[AnyScenarioParam] = [
    ScenarioParam(
        name="timeout_s",
        label="Timeout (sim-seconds)",
        min_value=1.0,
        max_value=300.0,
        step=1.0,
        default=float(load_gameplay_config().get("ui", {}).get("scenario_timeout_s", 30.0)),
    ),
    ScenarioParam(
        name="sim_speed",
        label="Sim speed (match-sec/real-sec)",
        min_value=0.1,
        max_value=32.0,
        step=0.5,
        default=1.0,
    ),
]


@dataclass
class ScenarioDefinition:
    key: str
    label: str
    description: str
    build: Callable[..., Match]   # (rng_reduction, **kwargs) -> Match
    params: list[AnyScenarioParam] = field(default_factory=list)
    on_tick: Callable[[Match, int], None] | None = None
    box_possession_terminal: bool = True
    """If False, ScenarioLoop never ends a trial on box possession alone.
    Use for scenarios (e.g. 1v2) where the attacker is expected to enter
    the box and should be allowed to shoot, get tackled, or score naturally."""
    win_outcome: str | None = None
    """Outcome key treated as 'Win' in the HUD counter — matches seeded_eval's
    win_outcome param so the UI tally is directly comparable to training/eval
    metrics.  None = fall back to the generic goal/saved/miss display."""
    outcome_remap: "Callable[[Match, str, str | None], str] | None" = None
    """Optional per-outcome label transformer called as remap(match, raw_outcome,
    last_ball_toucher_id) before the outcome is recorded.  Used by phase1 to
    apply the same label transforms ScenarioEnv does (goal→miss, miss→invalid
    when untouched) so UI tallies match training/eval breakdowns exactly."""
    phase1_trainee_player_id: str | None = None
    """player_id of the trainee, for phase-1 1v1 scenarios only. When set,
    forwarded to detect_trial_outcome() (ai/env/outcome.py) so box-possession
    detection uses the SAME trainee/opponent-differentiated, can_score_box_
    terminal()-gated check ScenarioEnv (training/eval) uses -- instead of the
    generic "any player, any box" check every other scenario gets. Without
    this, a UI/debug-script playback of a phase-1 1v1 scenario can end (and
    declare a winner) purely because the ball first neared a box, on a trial
    eval's own gated check would correctly keep running and score the other
    way once the ball changed hands again -- confirmed as a real, repeated
    source of "the UI shows a different outcome than eval recorded" bugs."""


def build_penalty_scenario(rng_reduction: float = 0.3) -> Match:
    """Recreates tests/balance/test_penalty_balance.py's penalty setup: one
    kicker on the penalty spot, aiming at a bottom corner, no goalkeeper."""
    pitch = Pitch.standard()
    penalty_spot = pitch.penalty_spot(left=False)
    attrs = generate_attributes(tier="premier_league")
    kicker = Player.create("kicker", Team.LEFT, attrs, position=penalty_spot)

    ball = Ball.at_rest(penalty_spot)
    ball.set_initial_possession(kicker.player_id)
    match = Match(pitch=pitch, players=[kicker], ball=ball, rng_reduction=rng_reduction, rng=random.Random())

    corner_offset_y = pitch.goal_width_m / 2.0 - 0.475
    aim_point = pitch.right_goal_centre + Vector3(0, corner_offset_y, 0.475)
    kicker.current_order = KickOrder(aim_point=aim_point, power_fraction=0.65, spin=Vector3.zero())
    return match


def build_tackle_scenario(
    rng_reduction: float = 0.3,
    *,
    separation_min_m: float = 1.0,
    separation_max_m: float = 10.0,
    tackler_tackling_min: float = 0.8,
    tackler_tackling_max: float = 0.8,
    dribbler_dribbling_min: float = 0.6,
    dribbler_dribbling_max: float = 0.6,
) -> Match:
    """Tackle challenge with randomised separation and optional skill range."""
    rng = random.Random()
    pitch = Pitch.standard()
    separation = rng.uniform(separation_min_m, separation_max_m)
    tackler_tackling = rng.uniform(tackler_tackling_min, tackler_tackling_max)
    dribbler_dribbling = rng.uniform(dribbler_dribbling_min, dribbler_dribbling_max)

    defender_attrs = PlayerAttributes(
        top_speed=0.6, acceleration=0.6, stamina=0.6, kick_precision=0.6,
        kick_power=0.6, dribbling=0.6, ball_control=0.6, tackling=tackler_tackling,
    )
    attacker_attrs = PlayerAttributes(
        top_speed=0.6, acceleration=0.6, stamina=0.6, kick_precision=0.6,
        kick_power=0.6, dribbling=dribbler_dribbling, ball_control=0.6, tackling=0.6,
    )
    attacker_pos = Vector3(0.0, 0.0, 0)
    defender_pos = Vector3(-separation, 0.0, 0)
    defender = Player.create("defender", Team.LEFT, defender_attrs, position=defender_pos)
    attacker = Player.create("attacker", Team.RIGHT, attacker_attrs, position=attacker_pos)

    heading = rng.uniform(-math.pi, math.pi)
    attacker.heading_rad = heading
    far_point = Vector3(
        attacker_pos.x + math.cos(heading) * 30.0,
        attacker_pos.y + math.sin(heading) * 30.0,
        0.0,
    )
    ball = Ball.at_rest(attacker_pos)
    ball.set_initial_possession(attacker.player_id)
    ui_cfg = load_gameplay_config().get("ui", {})
    match = Match(
        pitch=pitch, players=[defender, attacker], ball=ball,
        rng_reduction=rng_reduction, rng=rng,
        goal_linger_s=ui_cfg.get("goal_linger_s", 3.0),
    )
    attacker.current_order = MoveOrder(target_position=far_point, sprint=False)
    defender.current_order = ChaseTackleOrder(target_player_id=attacker.player_id)
    # Both orders self-complete (MoveOrder on arrival; ChaseTackleOrder once
    # contact is resolved) -- neither player has any other AI to pick up
    # movement intent afterwards, so hold position rather than crash/coast.
    attacker.ai = StopWhenIdleAI()
    defender.ai = StopWhenIdleAI()
    return match


# SprintController replaced by SprintWaypointAI (rules_ai.py) — a proper PlayerAI subclass.


def _make_sprint_waypoints(
    rng: random.Random, pitch: Pitch, start: Vector3,
    n_legs: int, leg_min_m: float, leg_max_m: float, max_retries: int = 10,
) -> list[Vector3]:
    waypoints: list[Vector3] = []
    current = start
    for _ in range(n_legs):
        candidate = start  # fallback
        for _attempt in range(max_retries):
            angle = rng.uniform(-math.pi, math.pi)
            length = rng.uniform(leg_min_m, leg_max_m)
            candidate = Vector3(current.x + math.cos(angle) * length,
                                current.y + math.sin(angle) * length, 0.0)
            if pitch.is_in_bounds(candidate):
                break
        clamped = Vector3(
            max(-pitch.half_length + 1.0, min(pitch.half_length - 1.0, candidate.x)),
            max(-pitch.half_width + 1.0, min(pitch.half_width - 1.0, candidate.y)),
            0.0,
        )
        waypoints.append(clamped)
        current = clamped
    return waypoints


def build_sprint_scenario(
    rng_reduction: float = 0.3,
    *,
    leg_min_m: float = 5.0,
    leg_max_m: float = 25.0,
    runner_skill_min: float = 0.7,
    runner_skill_max: float = 0.8,
) -> Match:
    """Sprint across a random 5-waypoint course on the pitch."""
    rng = random.Random()
    pitch = Pitch.standard()
    skill = rng.uniform(runner_skill_min, runner_skill_max)
    attrs = PlayerAttributes(skill, skill, skill, 0.5, 0.5, 0.5, 0.5, 0.5)
    start_x = rng.uniform(-pitch.half_length + 2.0, pitch.half_length - 2.0)
    start_y = rng.uniform(-pitch.half_width + 2.0, pitch.half_width - 2.0)
    start = Vector3(start_x, start_y, 0.0)
    player = Player.create("runner", Team.LEFT, attrs, position=start)
    waypoints = _make_sprint_waypoints(rng, pitch, start, n_legs=5,
                                       leg_min_m=leg_min_m, leg_max_m=leg_max_m)
    ball = Ball.at_rest(Vector3(0.0, pitch.half_width - 5.0, 0.0))
    ui_cfg = load_gameplay_config().get("ui", {})
    match = Match(
        pitch=pitch, players=[player], ball=ball,
        rng_reduction=rng_reduction, rng=rng,
        goal_linger_s=ui_cfg.get("goal_linger_s", 3.0),
    )
    player.current_order = MoveOrder(target_position=waypoints[0], sprint=True)
    player.ai = SprintWaypointAI(waypoints, start_idx=1)
    return match


def build_goal_to_goal_sprint_scenario(
    rng_reduction: float = 0.3,
    *,
    top_speed: float = 1.0,
    acceleration: float = 1.0,
) -> Match:
    """Max-stat player sprints from one goal line to the other."""
    pitch = Pitch.standard()
    attrs = PlayerAttributes(
        top_speed=top_speed, acceleration=acceleration, stamina=1.0,
        kick_precision=0.5, kick_power=0.5, dribbling=0.5,
        ball_control=0.5, tackling=0.5,
    )
    start = Vector3(-pitch.half_length, 0.0, 0.0)
    target = Vector3(pitch.half_length, 0.0, 0.0)
    player = Player.create("sprinter", Team.LEFT, attrs, position=start)
    ball = Ball.at_rest(Vector3(0.0, pitch.half_width - 3.0, 0.0))
    ui_cfg = load_gameplay_config().get("ui", {})
    match = Match(
        pitch=pitch, players=[player], ball=ball,
        rng_reduction=rng_reduction, rng=random.Random(),
        goal_linger_s=ui_cfg.get("goal_linger_s", 3.0),
    )
    player.current_order = MoveOrder(target_position=target, sprint=True, arrival_tolerance_m=1.0)
    # MoveOrder self-completes on arrival at the far goal line; nothing else
    # drives this player afterwards, so hold position instead of crashing.
    player.ai = StopWhenIdleAI()
    return match


def build_close_range_save_scenario(
    rng_reduction: float = 0.3,
    *,
    distance_min_m: float = 8.0,
    distance_max_m: float = 16.0,
    shooter_y_offset_m: float = 5.0,
    shooter_precision_min: float = 0.65,
    shooter_precision_max: float = 0.85,
    shooter_power_min: float = 0.70,
    shooter_power_max: float = 0.95,
    gk_skill_min: float = 0.65,
    gk_skill_max: float = 0.85,
) -> Match:
    """Shot vs keeper with randomised setup each trial."""
    pitch = Pitch.standard()
    half_goal_w = pitch.goal_width_m / 2.0
    rng = random.Random()

    gk_speed = rng.uniform(gk_skill_min, gk_skill_max)
    gk_control = rng.uniform(gk_skill_min, gk_skill_max)
    gk_attrs = PlayerAttributes(
        top_speed=gk_speed, acceleration=gk_speed, stamina=0.8,
        kick_precision=0.5, kick_power=0.5, dribbling=0.5,
        ball_control=gk_control, tackling=0.5,
    )
    shot_dist = rng.uniform(distance_min_m, distance_max_m)
    shooter_y = rng.uniform(-shooter_y_offset_m, shooter_y_offset_m)

    if rng.random() < 0.5:
        gk_start_y = rng.uniform(-half_goal_w + 0.3, half_goal_w - 0.3)
    else:
        shift_fraction = rng.uniform(0.3, 0.6)
        gk_start_y = max(-half_goal_w + 0.3,
                         min(half_goal_w - 0.3,
                             math.copysign(half_goal_w * shift_fraction, shooter_y)))

    gk = Player.create(
        "keeper", Team.LEFT, gk_attrs,
        position=pitch.left_goal_centre + Vector3(0, gk_start_y, 0),
        is_goalkeeper=True,
    )
    shooter_precision = rng.uniform(shooter_precision_min, shooter_precision_max)
    shooter_speed = rng.uniform(gk_skill_min, gk_skill_max)
    d_range = max(distance_max_m - distance_min_m, 1e-3)
    shooter_power = shooter_power_min + (shot_dist - distance_min_m) / d_range * (shooter_power_max - shooter_power_min)
    shooter_power = min(shooter_power_max + 0.02, max(shooter_power_min - 0.02,
                                                       shooter_power + rng.uniform(-0.04, 0.04)))
    shooter_attrs = PlayerAttributes(
        top_speed=shooter_speed, acceleration=shooter_speed, stamina=0.7,
        kick_precision=shooter_precision, kick_power=shooter_power,
        dribbling=0.5, ball_control=0.5, tackling=0.5,
    )
    shooter_pos = Vector3(-(pitch.half_length - shot_dist), shooter_y, 0)
    shooter = Player.create("striker", Team.RIGHT, shooter_attrs, position=shooter_pos)

    # Live, goalkeeper-aware corner pick (see shot_selection.choose_shot_target)
    # instead of a fixed bias-away-from-gk_start_y random point -- shooter and
    # gk are both already positioned above, so this is effectively "at the
    # moment of shooting" for this scenario (the shot fires on tick 1 anyway).
    from footballcoach.shot_selection import choose_shot_target
    aim_point = choose_shot_target(shooter, shooter_power, gk, pitch, rng)

    aim_dir = aim_point - shooter_pos
    aim_xy_len = math.hypot(aim_dir.x, aim_dir.y)
    mvmt = MovementParams.from_config()
    run_speed = effective_top_speed(
        mvmt, shooter.attributes.top_speed, shooter.stamina,
        has_ball=True, ball_control_attr=shooter.attributes.ball_control,
    )
    shooter.velocity = Vector3(aim_dir.x / aim_xy_len * run_speed,
                               aim_dir.y / aim_xy_len * run_speed, 0.0)
    shooter.heading_rad = math.atan2(aim_dir.y, aim_dir.x)

    ball = Ball.at_rest(shooter_pos)
    ball.set_initial_possession(shooter.player_id)
    ui_cfg = load_gameplay_config().get("ui", {})
    match = Match(
        pitch=pitch, players=[gk, shooter], ball=ball,
        rng_reduction=rng_reduction, rng=rng,
        goal_linger_s=ui_cfg.get("goal_linger_s", 3.0),
    )
    gk.current_order = SaveOrder()
    gk.ai = StopWhenIdleAI()
    shooter.current_order = KickOrder(aim_point=aim_point, power_fraction=shooter_power,
                                      spin=Vector3.zero(), compensate_for_run=False)
    shooter.ai = StopWhenIdleAI()
    return match


def build_pass_scenario(
    rng_reduction: float = 0.3,
    *,
    max_distance_m: float = 30.0,
    attr_clamp_min: float = 0.70,
    attr_clamp_max: float = 0.80,
) -> Match:
    """Randomised ground pass, clamping kick_precision/ball_control into range."""
    rng = random.Random()
    pitch = Pitch.standard()
    passer_attrs_raw = generate_attributes(tier="premier_league", rng=rng)
    receiver_attrs_raw = generate_attributes(tier="premier_league", rng=rng)

    def clamp_attr(v: float) -> float:
        return v if attr_clamp_min <= v <= attr_clamp_max else rng.uniform(attr_clamp_min, attr_clamp_max)

    passer_attrs = PlayerAttributes(
        top_speed=passer_attrs_raw.top_speed, acceleration=passer_attrs_raw.acceleration,
        stamina=passer_attrs_raw.stamina, kick_precision=clamp_attr(passer_attrs_raw.kick_precision),
        kick_power=passer_attrs_raw.kick_power, dribbling=passer_attrs_raw.dribbling,
        ball_control=clamp_attr(passer_attrs_raw.ball_control), tackling=passer_attrs_raw.tackling,
    )
    receiver_attrs = PlayerAttributes(
        top_speed=receiver_attrs_raw.top_speed, acceleration=receiver_attrs_raw.acceleration,
        stamina=receiver_attrs_raw.stamina, kick_precision=clamp_attr(receiver_attrs_raw.kick_precision),
        kick_power=receiver_attrs_raw.kick_power, dribbling=receiver_attrs_raw.dribbling,
        ball_control=clamp_attr(receiver_attrs_raw.ball_control), tackling=receiver_attrs_raw.tackling,
    )

    dist = rng.uniform(5.0, max_distance_m)
    angle = rng.uniform(-math.pi, math.pi)
    px = rng.uniform(-pitch.half_length * 0.5, pitch.half_length * 0.5)
    py = rng.uniform(-pitch.half_width * 0.5, pitch.half_width * 0.5)
    passer_pos = Vector3(px, py, 0)
    receiver_pos = Vector3(
        max(-pitch.half_length + 1.0, min(pitch.half_length - 1.0, px + math.cos(angle) * dist)),
        max(-pitch.half_width + 1.0, min(pitch.half_width - 1.0, py + math.sin(angle) * dist)),
        0,
    )

    passer = Player.create("passer", Team.LEFT, passer_attrs, position=passer_pos)
    receiver = Player.create("receiver", Team.LEFT, receiver_attrs, position=receiver_pos)

    mvmt = MovementParams.from_config()
    top_speed = effective_top_speed(
        mvmt, passer.attributes.top_speed, passer.stamina,
        has_ball=True, ball_control_attr=passer.attributes.ball_control,
    )
    # Run/face toward the actual pass target (receiver_pos, post pitch-edge
    # clamping) rather than a hardcoded +x -- previously this was always
    # (top_speed*0.5, 0, 0) regardless of the randomised pass angle, making
    # the passer visibly run toward the goal instead of toward the pass.
    pass_dir = receiver_pos - passer_pos
    pass_dir_xy_len = math.hypot(pass_dir.x, pass_dir.y)
    # Cap the run-up speed well below the ball's own auto-computed pace
    # (same formula PassOrder/pass_ball will use at execution time -- see
    # pass_speed_mps) so the ball is guaranteed to pull away from the
    # passer instead of the two travelling at near-identical speed. Without
    # this, a short pass (low auto speed, ~2-4 m/s) barely outran a jogging
    # top_speed*0.5 run-up, so the passer immediately re-picked up their own
    # pass (can_pick_up_ball()'s closing/deadzone check never saw real
    # separation) -- reproduced on 30/30 short (5-8m) trials.
    ball_phys = BallPhysicsParams.from_config()
    passing_params = PassingParams.from_config()
    auto_pass_speed = pass_speed_mps(
        passing_params, pass_dir_xy_len, ball_phys.gravity_mps2, ball_phys.rolling_friction_coefficient,
    )
    run_speed = min(top_speed * 0.5, auto_pass_speed * 0.5)
    passer.velocity = Vector3(pass_dir.x / pass_dir_xy_len * run_speed,
                              pass_dir.y / pass_dir_xy_len * run_speed, 0.0)
    passer.heading_rad = math.atan2(pass_dir.y, pass_dir.x)

    ball = Ball.at_rest(passer_pos)
    ball.set_initial_possession(passer.player_id)
    ui_cfg = load_gameplay_config().get("ui", {})
    match = Match(
        pitch=pitch, players=[passer, receiver], ball=ball,
        rng_reduction=rng_reduction, rng=rng,
        goal_linger_s=ui_cfg.get("goal_linger_s", 3.0),
    )
    passer.current_order = PassOrder(target_position=receiver_pos)
    passer.ai = StopWhenIdleAI()
    # PassReceiverAI leaves current_order untouched while still waiting for
    # the ball to come within range (see _IdleFallbackAI's own docstring for
    # why that gap can't just be closed with an initial order at build time)
    # -- wrap it so the receiver has movement intent set every tick even
    # before the ball is close enough to react to.
    receiver.ai = _IdleFallbackAI(PassReceiverAI(get_possession_radius_m=8.0))
    return match


# ---------------------------------------------------------------------------
# Reusable per-player AI primitives — live in footballcoach.rules_ai
# ---------------------------------------------------------------------------

from footballcoach.rules_ai import (
    BallCarrierAttackerAI,
    BallReceiverThenShootAI,
    NeuralPlayerAI,
    PassReceiverAI,
    StagedGoalkeeperAI,
    Phase1RulesAI,
    SprintWaypointAI,
    StopWhenIdleAI,
)


class _IdleFallbackAI(PlayerAI):
    """Wraps another ``PlayerAI`` and, after delegating to it each tick,
    re-issues a ``StopOrder`` if the wrapped AI left ``current_order`` as
    ``None``.

    A couple of shared ``rules_ai.py`` AIs have a genuinely-idle branch that
    deliberately leaves ``current_order`` untouched instead of setting a
    fresh order every tick -- e.g. ``StagedGoalkeeperAI`` once parked at goal
    centre (only fixes heading, see ``_face_outfield_if_parked``), and
    ``PassReceiverAI`` while still waiting for the ball to come within
    ``get_possession_radius_m`` (its docstring even says to pair it with an
    initial persistent order at build time -- but any such order, e.g.
    ``StopOrder``/``MoveOrder`` to a stationary target, self-completes and
    clears itself the very first tick since the player is already at rest,
    which just relocates the gap by one tick instead of closing it). Both
    were harmless before ``Match._apply_movement`` started requiring every
    player to have movement intent set EVERY tick (see its docstring) --
    the player was already at ~0 velocity, so silently coasting looked like
    standing still. Now it has to be an explicit ``StopOrder``, same as
    everywhere else in this codebase (see ``StopWhenIdleAI``, whose
    docstring explicitly invites reusing this exact one-line pattern).

    Deliberately NOT folded into ``rules_ai.py`` itself: both wrapped AIs are
    shared well beyond this file's own scenarios (real matches, other tests),
    so the fix is composed locally here instead of changing their shared
    behaviour.
    """

    def __init__(self, inner: PlayerAI) -> None:
        super().__init__()
        self._inner = inner

    def act(self, player: Player, match: Match, trial_tick: int) -> None:
        self._inner.act(player, match, trial_tick)
        if player.current_order is None:
            player.current_order = StopOrder()


def _2v2_cfg() -> dict:
    return load_scenarios_config().get("2v2", {})


def build_2v2_scenario(
    rng_reduction: float = 0.3,
    *,
    attacker_skill_min: float | None = None,
    attacker_skill_max: float | None = None,
    defender_skill_min: float | None = None,
    defender_skill_max: float | None = None,
    gk_skill_min: float | None = None,
    gk_skill_max: float | None = None,
    shoot_immediately_probability: float | None = None,
    attacker_a_dist_from_goal_m: float | None = None,
    attacker_a_y_m: float | None = None,
    attacker_b_x_offset_m: float | None = None,
    attacker_b_y_fraction: float | None = None,
    attacker_b_running: bool | None = None,
    defender_dist_from_goal_m: float | None = None,
    defender_y_m: float | None = None,
    pass_power_multiplier: float | None = None,
) -> Match:
    """2v2: attacker A (ball) passes to attacker B; defender + GK defend.

    Team convention: Team.LEFT attacks the right goal (+x); Team.RIGHT defends
    the right goal.  Attackers are Team.LEFT, defender+GK are Team.RIGHT.

    ``attacker_a_dist_from_goal_m`` — distance from the right goal line where attacker A starts.
    ``attacker_a_y_m`` — lateral (y) position of attacker A.
    ``attacker_b_x_offset_m`` — how far behind attacker A (in x) attacker B starts.
    ``attacker_b_y_fraction`` — 1.0 = original far-side y; 0.0 = same y as attacker A.
    ``attacker_b_running`` — if True, attacker B starts at full sprint speed.
    ``defender_dist_from_goal_m`` — distance of the defender from the right goal line.
    ``defender_y_m`` — lateral (y) position of the defender.
    ``pass_power_multiplier`` — scales the auto-computed pass speed (1.0 = unmodified).
    """
    _cfg = _2v2_cfg()
    if attacker_skill_min is None: attacker_skill_min = float(_cfg.get("attacker_skill_min", 0.7))
    if attacker_skill_max is None: attacker_skill_max = float(_cfg.get("attacker_skill_max", 0.85))
    if defender_skill_min is None: defender_skill_min = float(_cfg.get("defender_skill_min", 0.55))
    if defender_skill_max is None: defender_skill_max = float(_cfg.get("defender_skill_max", 0.7))
    if gk_skill_min is None: gk_skill_min = float(_cfg.get("gk_skill_min", 0.55))
    if gk_skill_max is None: gk_skill_max = float(_cfg.get("gk_skill_max", 0.7))
    if shoot_immediately_probability is None: shoot_immediately_probability = float(_cfg.get("shoot_immediately_probability", 0.5))
    if attacker_a_dist_from_goal_m is None: attacker_a_dist_from_goal_m = float(_cfg.get("attacker_a_dist_from_goal_m", 18.0))
    if attacker_a_y_m is None: attacker_a_y_m = float(_cfg.get("attacker_a_y_m", -3.16))
    if attacker_b_x_offset_m is None: attacker_b_x_offset_m = float(_cfg.get("attacker_b_x_offset_m", 4.0))
    if attacker_b_y_fraction is None: attacker_b_y_fraction = float(_cfg.get("attacker_b_y_fraction", 0.8))
    if attacker_b_running is None: attacker_b_running = bool(_cfg.get("attacker_b_running", True))
    if defender_dist_from_goal_m is None: defender_dist_from_goal_m = float(_cfg.get("defender_dist_from_goal_m", 8.0))
    if defender_y_m is None: defender_y_m = float(_cfg.get("defender_y_m", 0.0))
    if pass_power_multiplier is None: pass_power_multiplier = float(_cfg.get("pass_power_multiplier", 1.5))

    _attr_names = ("top_speed", "acceleration", "stamina", "kick_precision",
                   "kick_power", "dribbling", "ball_control", "tackling")

    def _make_attrs(skill: float, overrides: dict) -> PlayerAttributes:
        """Build PlayerAttributes from a uniform skill, applying per-attribute overrides.
        If 'the_rest' key is present, it replaces the uniform skill as the default
        for any attribute not explicitly listed."""
        base = float(overrides.get("the_rest", skill))
        vals = {a: float(overrides.get(a, base)) for a in _attr_names}
        return PlayerAttributes(**vals)

    rng = random.Random()
    pitch = Pitch.standard()

    def make_player(pid: str, team: Team, skill: float, pos: Vector3, gk: bool = False) -> Player:
        overrides = _cfg.get(f"{pid}_attrs", {})
        attrs = _make_attrs(skill, overrides)
        return Player.create(pid, team, attrs, position=pos, is_goalkeeper=gk)

    half_goal_w = pitch.goal_width_m / 2.0
    a_x = pitch.half_length - attacker_a_dist_from_goal_m
    a_y = attacker_a_y_m
    # attacker B: configurable offset from attacker A
    b_y_far = half_goal_w - 0.5
    b_y = a_y + attacker_b_y_fraction * (b_y_far - a_y)
    b_x = a_x - attacker_b_x_offset_m

    # Team.LEFT attacks right goal; Team.RIGHT defends right goal.
    attacker_a = make_player("atk_a", Team.LEFT,
                             rng.uniform(attacker_skill_min, attacker_skill_max),
                             Vector3(a_x, a_y, 0))
    attacker_b = make_player("atk_b", Team.LEFT,
                             rng.uniform(attacker_skill_min, attacker_skill_max),
                             Vector3(b_x, b_y, 0))
    defender = make_player("defender", Team.RIGHT,
                           rng.uniform(defender_skill_min, defender_skill_max),
                           Vector3(pitch.half_length - defender_dist_from_goal_m, defender_y_m, 0))
    gk_sk = rng.uniform(gk_skill_min, gk_skill_max)
    gk = Player.create("keeper", Team.RIGHT,
                       _make_attrs(gk_sk, _cfg.get("keeper_attrs", {})),
                       position=pitch.right_goal_centre, is_goalkeeper=True)

    ball = Ball.at_rest(attacker_a.position)
    ball.set_initial_possession(attacker_a.player_id)
    ui_cfg = load_gameplay_config().get("ui", {})
    match = Match(
        pitch=pitch, players=[attacker_a, attacker_b, defender, gk],
        ball=ball, rng_reduction=rng_reduction, rng=rng,
        goal_linger_s=ui_cfg.get("goal_linger_s", 3.0),
    )

    move_target = Vector3(pitch.half_length - 5.0, b_y, 0)
    attacker_b.current_order = MoveOrder(target_position=move_target, sprint=True)

    if attacker_b_running:
        mvmt = MovementParams.from_config()
        sprint_speed = effective_top_speed(
            mvmt, attacker_b.attributes.top_speed, attacker_b.stamina, has_ball=False,
        )
        attacker_b.velocity = Vector3(sprint_speed, 0.0, 0.0)
        attacker_b.heading_rad = 0.0  # facing +x toward right goal

    attacker_a.current_order = PassOrder(
        target_position=attacker_b.position,
        target_player_id=attacker_b.player_id,
        power_multiplier=pass_power_multiplier,
    )
    defender.current_order = GetPossessionOrder()
    gk.current_order = MoveOrder(target_position=pitch.right_goal_centre, sprint=False,
                                  max_speed_on_arrival_mps=0.0)

    # attacker_a's PassOrder always completes in a single tick; nothing else
    # drives it afterwards, so hold position instead of crashing next tick.
    attacker_a.ai = StopWhenIdleAI()

    # goal_aim_point omitted -> BallCarrierAttackerAI (via BallReceiverThenShootAI)
    # picks a live, goalkeeper-aware corner at the moment of shooting.
    # Wrapped: BallCarrierAttackerAI.decide()'s "lost the ball" branch (used
    # once attacker_b is delegated to it, post-reception) deliberately
    # abandons current_order without setting a fresh one -- see
    # _IdleFallbackAI's own docstring.
    attacker_b.ai = _IdleFallbackAI(BallReceiverThenShootAI(
        shoot_immediately=rng.random() < shoot_immediately_probability,
    ))
    # Phase1RulesAI.decide() unconditionally sets a fresh order every call
    # (see its own docstring) -- no idle-fallback gap to wrap.
    defender.ai = Phase1RulesAI()
    gk.ai = _IdleFallbackAI(StagedGoalkeeperAI())
    return match


# ---------------------------------------------------------------------------
# 1v2 scenario
# ---------------------------------------------------------------------------

def build_1v2_scenario(
    rng_reduction: float = 0.3,
    *,
    attacker_skill: float = 0.9,
    defender_skill: float = 0.55,
    gk_skill: float = 0.55,
    attacker_start_min_m: float = 18.0,
    attacker_start_max_m: float = 32.0,
    attacker_y_offset_m: float = 12.0,
    defender_fraction_min: float = 0.3,
    defender_fraction_max: float = 0.7,
    defender_jitter_m: float = 2.0,
    gk_start_jitter_m: float = 1.0,
    move_fraction_min: float = 0.25,
    move_fraction_max: float = 0.50,
) -> Match:
    """1v2: elite attacker vs. average defender + GK."""
    rng = random.Random()
    pitch = Pitch.standard()
    min_gap = 2.0 * 0.3 + 0.5
    goal_centre = pitch.right_goal_centre

    for _r in range(20):
        d0 = rng.uniform(attacker_start_min_m, attacker_start_max_m)
        y0 = rng.uniform(-attacker_y_offset_m, attacker_y_offset_m)
        attacker_start = Vector3(pitch.half_length - d0, y0, 0)
        if pitch.is_in_bounds(attacker_start):
            break
    else:
        attacker_start = Vector3(pitch.half_length - 25.0, 0.0, 0)

    line = goal_centre - attacker_start
    line_len = math.hypot(line.x, line.y)
    perp_x, perp_y = -line.y / line_len, line.x / line_len

    for _r in range(20):
        frac = rng.uniform(defender_fraction_min, defender_fraction_max)
        jitter = rng.uniform(-defender_jitter_m, defender_jitter_m)
        mid = attacker_start + line * frac
        defender_start = Vector3(mid.x + perp_x * jitter, mid.y + perp_y * jitter, 0)
        if (pitch.is_in_bounds(defender_start)
                and math.hypot(defender_start.x - attacker_start.x,
                               defender_start.y - attacker_start.y) > min_gap
                and math.hypot(defender_start.x - goal_centre.x,
                               defender_start.y - goal_centre.y) > min_gap):
            break
    else:
        defender_start = Vector3(pitch.half_length - 12.0, 0.0, 0)

    gk_start = Vector3(pitch.right_goal_centre.x, rng.uniform(-gk_start_jitter_m, gk_start_jitter_m), 0)

    # Attacker runs toward the RIGHT goal (+x) so must be Team.LEFT (attacks +x).
    # GK and defender are Team.RIGHT (defend the right goal).
    attacker = Player.create("attacker", Team.LEFT, PlayerAttributes.average(attacker_skill), position=attacker_start)
    defender = Player.create("defender", Team.RIGHT, PlayerAttributes.average(defender_skill), position=defender_start)
    gk = Player.create("keeper", Team.RIGHT, PlayerAttributes.average(gk_skill), position=gk_start, is_goalkeeper=True)

    ball = Ball.at_rest(attacker_start)
    ball.set_initial_possession(attacker.player_id)
    move_frac = rng.uniform(move_fraction_min, move_fraction_max)
    move_target_raw = attacker_start + (goal_centre - attacker_start) * move_frac
    move_target = Vector3(
        max(-pitch.half_length + 1, min(pitch.half_length - 1, move_target_raw.x)),
        max(-pitch.half_width + 1, min(pitch.half_width - 1, move_target_raw.y)),
        0.0,
    )

    ui_cfg = load_gameplay_config().get("ui", {})
    match = Match(
        pitch=pitch, players=[attacker, defender, gk],
        ball=ball, rng_reduction=rng_reduction, rng=rng,
        goal_linger_s=ui_cfg.get("goal_linger_s", 3.0),
    )

    attacker.current_order = MoveOrder(target_position=move_target, sprint=True)
    defender.current_order = GetPossessionOrder()
    gk.current_order = MoveOrder(target_position=goal_centre, sprint=False, max_speed_on_arrival_mps=0.0)

    # Start the attacker already at the speed/heading their MoveOrder will
    # drive them to, rather than from a standstill -- the scenario is meant
    # to snapshot a run already in progress, not an explosive first step.
    run_dir = (move_target - attacker_start).xy()
    if run_dir.length() > 1e-9:
        run_dir = run_dir.normalized()
        mvmt = MovementParams.from_config()
        run_speed = effective_top_speed(
            mvmt, attacker.attributes.top_speed, attacker.stamina, has_ball=True,
            ball_control_attr=attacker.attributes.ball_control,
        )
        attacker.velocity = Vector3(run_dir.x * run_speed, run_dir.y * run_speed, 0.0)
        attacker.heading_rad = math.atan2(run_dir.y, run_dir.x)

    # aim_point omitted -> BallCarrierAttackerAI picks a live, goalkeeper-aware
    # corner at the moment of shooting (see shot_selection.choose_shot_target).
    # Wrapped: its decide() deliberately abandons current_order without
    # setting a fresh one once the attacker loses the ball (e.g. after
    # shooting) -- see _IdleFallbackAI's own docstring.
    attacker.ai = _IdleFallbackAI(BallCarrierAttackerAI(power_fraction=0.9))
    # Phase1RulesAI.decide() unconditionally sets a fresh order every call
    # (see its own docstring) -- no idle-fallback gap to wrap.
    defender.ai = Phase1RulesAI()
    gk.ai = _IdleFallbackAI(StagedGoalkeeperAI())
    return match


# ---------------------------------------------------------------------------
# on_tick dispatch helpers for the SCENARIOS list
# ---------------------------------------------------------------------------

_PASS_SCENARIO_GET_POSSESSION_RADIUS_M = 6.0


def _pass_on_tick(match: Match, trial_tick: int) -> None:
    """Once the ball is within 4 m of the receiver, give them a GetPossessionOrder
    so they actively move to collect it (rather than standing still)."""
    try:
        receiver = match.player_by_id("receiver")
    except KeyError:
        return
    if match.ball.possessed_by is not None:
        return
    if isinstance(receiver.current_order, GetPossessionOrder):
        return
    dist = receiver.position.xy().distance_to(match.ball.position.xy())
    if dist <= _PASS_SCENARIO_GET_POSSESSION_RADIUS_M:
        receiver.current_order = GetPossessionOrder()


def _sprint_on_tick(match: Match, trial_tick: int) -> None:
    pass  # runner.ai = SprintWaypointAI assigned in build; Match.step() fires it.


# _2v2_on_tick and _1v2_on_tick removed: players now carry their own AI
# via player.ai — Match.step() calls it automatically.


# ---------------------------------------------------------------------------
# Repulsion obstacle scenario
# ---------------------------------------------------------------------------

def build_repulsion_obstacle_scenario(
    rng_reduction: float = 0.3,
    *,
    obstacle_on_path: float = 1.0,
    attacker_skill: float = 0.9,
) -> Match:
    """Ball carrier runs from x=-20 to x=+20 with a stationary obstacle at
    x=0. obstacle_on_path=1.0 puts it dead on the line; 0.0 puts it 5 m
    to the side. Watch for orbiting vs. clean passage."""
    pitch = Pitch.standard()
    rng = random.Random()

    start = Vector3(-20.0, 0.0, 0.0)
    target = Vector3(20.0, 0.0, 0.0)
    obstacle_y = (1.0 - obstacle_on_path) * 5.0

    obstacle = Player.create(
        "obstacle", Team.RIGHT,
        PlayerAttributes.average(0.5),
        position=Vector3(0.0, obstacle_y, 0.0),
    )
    attacker = Player.create(
        "attacker", Team.LEFT,
        PlayerAttributes.average(attacker_skill),
        position=start,
    )
    ball = Ball.at_rest(start)
    ball.set_initial_possession(attacker.player_id)
    ui_cfg = load_gameplay_config().get("ui", {})
    match = Match(
        pitch=pitch, players=[attacker, obstacle], ball=ball,
        rng_reduction=rng_reduction, rng=rng,
        goal_linger_s=ui_cfg.get("goal_linger_s", 3.0),
    )
    attacker.current_order = MoveOrder(target_position=target, sprint=True)
    # obstacle never gets any order at all (it's meant to just stand there);
    # attacker's MoveOrder self-completes on arrival at the far side with
    # nothing else to pick up movement intent afterwards. Both need an
    # explicit idle AI rather than being left to crash/coast.
    obstacle.ai = StopWhenIdleAI()
    attacker.ai = StopWhenIdleAI()
    return match


def build_repulsion_obstacle_no_ball_scenario(
    rng_reduction: float = 0.3,
    *,
    obstacle_on_path: float = 1.0,
    attacker_skill: float = 0.9,
) -> Match:
    """Identical to build_repulsion_obstacle_scenario except the runner never
    takes possession of the ball -- isolates the pure movement/repulsion
    avoidance mechanic from ball-carrying effects (effective_top_speed's
    has_ball speed reduction, dribble-loss risk) that the possessed version
    always has active. Ball sits at rest at the start position, uninvolved."""
    pitch = Pitch.standard()
    rng = random.Random()

    start = Vector3(-20.0, 0.0, 0.0)
    target = Vector3(20.0, 0.0, 0.0)
    obstacle_y = (1.0 - obstacle_on_path) * 5.0

    obstacle = Player.create(
        "obstacle", Team.RIGHT,
        PlayerAttributes.average(0.5),
        position=Vector3(0.0, obstacle_y, 0.0),
    )
    runner = Player.create(
        "attacker", Team.LEFT,
        PlayerAttributes.average(attacker_skill),
        position=start,
    )
    # Parked well off the x=-20->+20, y~0 run path (not just unpossessed --
    # if it sat at `start` like the ball-carrier version does, the runner
    # would be within the 0.55m auto-pickup radius on the very first tick
    # and pick it up immediately regardless of set_initial_possession).
    ball = Ball.at_rest(Vector3(0.0, pitch.half_width - 5.0, 0.0))
    ui_cfg = load_gameplay_config().get("ui", {})
    match = Match(
        pitch=pitch, players=[runner, obstacle], ball=ball,
        rng_reduction=rng_reduction, rng=rng,
        goal_linger_s=ui_cfg.get("goal_linger_s", 3.0),
    )
    runner.current_order = MoveOrder(target_position=target, sprint=True)
    obstacle.ai = StopWhenIdleAI()
    runner.ai = StopWhenIdleAI()
    return match


# ---------------------------------------------------------------------------
# Phase 1 curriculum: 1v1 get-possession / move-toward-goal
# ---------------------------------------------------------------------------

# Phase 1 on_tick functions removed: both trainee and opponent now carry
# player.ai assigned in build_1v1_scenario — Match.step() fires them automatically.
# Kept as no-ops for backward compat with any external references.

def _1v1_on_tick(match: Match, trial_tick: int) -> None:
    pass


def phase1_training_on_tick(match: Match, trial_tick: int) -> None:
    pass


def _load_trainer(checkpoint_path: str):
    """Load a PPOTrainer in inference-only mode (no optimizer).

    Uses PPOTrainer.load_for_inference() which skips torch.optim.Adam creation,
    preventing the torch._dynamo → triton import that segfaults inside pygame.
    """
    import os
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None, f"No checkpoint found ({checkpoint_path})"
    try:
        from footballcoach.ai.ppo.ppo_trainer import PPOTrainer
        trainer = PPOTrainer.load_for_inference(checkpoint_path)
        return trainer, None
    except Exception as e:
        return None, str(e)


def _discover_checkpoints(checkpoint_dir: str) -> list[str]:
    """Return sorted list of .pt checkpoint paths in checkpoint_dir.

    Matches any *.pt file, including checkpoint_pretrained.pt and
    arbitrarily-named snapshots (e.g. competent_pretrain.pt).
    latest.pt is always placed last as the most recent.

    Sorting is by numeric suffix so checkpoint14.pt > checkpoint9.pt;
    files with no numeric suffix sort before numbered ones.
    """
    import glob
    import os
    import re
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return []
    all_paths = set(glob.glob(os.path.join(checkpoint_dir, "*.pt")))

    def _sort_key(p: str) -> tuple[int, str]:
        name = os.path.basename(p)
        m = re.search(r"(\d+)", name)
        return (int(m.group(1)) if m else -1, name)

    numbered = sorted(
        (p for p in all_paths if os.path.basename(p) != "latest.pt"),
        key=_sort_key,
    )
    # latest.pt goes at the very end so the default picker lands on it
    latest = os.path.join(checkpoint_dir, "latest.pt")
    if os.path.exists(latest):
        numbered.append(latest)
    return numbered


def _phase1_scenario_cfg() -> dict:
    """Return the phase1_scenario section from ai_config.json, with fallback defaults."""
    try:
        from footballcoach.ai.config import load_ai_config
        return load_ai_config().get("phase1_scenario", {})
    except Exception:
        return {}


def _phase1_training_cfg() -> dict:
    """Returns the actual values PPO training uses for decision cadence and
    episode length -- sourced from ai_config.json's ``observation``/
    ``curriculum`` sections (NOT ``phase1_scenario``, which only covers the
    randomised-scenario knobs) -- so the Phase 1 UI's defaults can match
    training instead of drifting from it via separately-hardcoded numbers.
    Falls back to the same numbers currently in ai_config.json if it can't
    be loaded, so a missing/broken config degrades gracefully rather than
    crashing the scenario picker."""
    try:
        from footballcoach.ai.config import load_ai_config
        cfg = load_ai_config()
        obs = cfg.get("observation", {})
        curriculum = cfg.get("curriculum", {})
        return {
            "decision_interval_s": float(obs.get("decision_interval_s", 0.2499)),
            "max_episode_s": float(curriculum.get("phase1_max_episode_s", 18.5)),
        }
    except Exception:
        return {"decision_interval_s": 0.2499, "max_episode_s": 18.5}


def _make_phase1_scenario_pair(checkpoint_dir: str = "checkpoints/phase1_run1"):
    """Factory returning (build_fn, None, params_list) for the Phase 1 UI scenario
    -- the second element is always None; neural players are driven by a plain
    ``player.ai = NeuralPlayerAI(...)`` assignment in ``build()`` (Match.step()
    calls it once per tick like any other AI), not a per-tick scenario hook.

    Params:
      trainee_checkpoint   — ScenarioChoiceParam dropdown over all .pt files in checkpoint_dir
      trainee_rules        — ScenarioBoolParam checkbox (overrides to rules-based)
      opponent_checkpoint  — same dropdown for opponent
      opponent_rules       — ScenarioBoolParam checkbox
      ball_max_speed_mps   — ScenarioParam slider

    Trainers are cached per resolved path so switching checkpoints only re-loads on change.
    """
    import logging
    log_ui = logging.getLogger("footballcoach.ui.scenarios")

    _train_cfg = _phase1_training_cfg()
    # Matches the real-world decision cadence PPO training actually uses
    # (ai_config.json observation.decision_interval_s, currently 0.2499s) --
    # NOT the tick count training uses (5 ticks at its own 20Hz), since the
    # UI intentionally ticks the engine at a different rate (see UI_TICK_HZ
    # below); matching on real seconds is what keeps the network seeing the
    # same decision cadence it was trained on regardless of that difference.
    # This replaced a stale hardcoded 500ms (a leftover 15-tick-at-30Hz
    # default that was never actually training's value) -- exactly 2x too
    # slow a decision cadence versus training.
    DECISION_INTERVAL_MS_DEFAULT = _train_cfg["decision_interval_s"] * 1000.0
    UI_TICK_HZ = 30.0  # UI always ticks the engine at 30Hz for smooth human playback -- deliberately NOT ai_config.json's sim_dt_s (0.05s/20Hz, see its own _comment_sim_dt_s: "UI ignores this and always uses 30Hz")
    # Matches ai_config.json curriculum.phase1_max_episode_s -- the real
    # episode length training uses. NeuralPlayerAI derives a "time_remaining"
    # observation feature from this (see rules_ai.py); leaving it at the old
    # hardcoded 1e9 effectively froze that feature at a huge constant,
    # meaning the network never saw the time-pressure signal here that it
    # saw throughout training.
    PHASE1_MAX_EPISODE_S = _train_cfg["max_episode_s"]

    _trainer_cache: dict[str, object] = {}

    def _get_trainer(checkpoint_path: str | None):
        if not checkpoint_path:
            return None
        if checkpoint_path not in _trainer_cache:
            trainer, err = _load_trainer(checkpoint_path)
            if err:
                log_ui.warning(f"Phase 1 UI: {err} — using rules-based AI")
            _trainer_cache[checkpoint_path] = trainer
        return _trainer_cache[checkpoint_path]

    # Build the params list: scan ALL phase1_run* dirs and longterm/ so
    # checkpoints from every run are visible, not just the one dir passed in.
    import glob as _glob, re as _re
    _all_run_dirs = sorted(
        _glob.glob("checkpoints/phase1_run*/"),
        key=lambda d: int(m.group(1)) if (m := _re.search(r"phase1_run(\d+)", d)) else -1,
    )
    checkpoints = []
    for _d in _all_run_dirs:
        checkpoints.extend(_discover_checkpoints(_d.rstrip("/")))
    _phase1_count = len(checkpoints)  # index boundary before longterm entries
    checkpoints.extend(_discover_checkpoints("checkpoints/longterm"))
    # Friendly display names: run{N}/filename
    def _ckpt_label(p: str) -> str:
        parts = Path(p).parts
        return f"{parts[-2]}/{parts[-1]}" if len(parts) >= 2 else parts[-1]
    ckpt_labels = tuple(_ckpt_label(c) for c in checkpoints) if checkpoints else ("(none)",)
    # Default: most recent phase1_run checkpoint; fall back to last item if no phase1 runs exist
    ckpt_default = ckpt_labels[_phase1_count - 1] if _phase1_count > 0 else ckpt_labels[-1]

    # Group labels by folder (the part before the "/") so the UI can offer a
    # folder-first, then-checkpoint two-level dropdown instead of one huge
    # flat list.
    def _grouped_ckpt_labels() -> tuple[tuple[str, tuple[str, ...]], ...]:
        groups: dict[str, list[str]] = {}
        for label in ckpt_labels:
            folder, _sep, _rest = label.partition("/")
            groups.setdefault(folder, []).append(label)
        return tuple((folder, tuple(vals)) for folder, vals in groups.items())
    ckpt_groups = _grouped_ckpt_labels()

    _cfg = _phase1_scenario_cfg()
    params_list: list = [
        ScenarioGroupedChoiceParam("trainee_checkpoint", "Trainee checkpoint", ckpt_groups, ckpt_default),
        ScenarioBoolParam("trainee_rules", "Trainee: rules-based override", False),
        ScenarioBoolParam("trainee_immobile", "Trainee: immobile override", False),
        ScenarioGroupedChoiceParam("opponent_checkpoint", "Opponent checkpoint", ckpt_groups, ckpt_default),
        ScenarioBoolParam("opponent_rules", "Opponent: rules-based override", False),
        ScenarioBoolParam("opponent_immobile", "Opponent: immobile override", False),
        ScenarioChoiceParam("trainee_tier", "Trainee tier", ("generic", "amateur", "semi_pro", "premier_league"), str(_cfg.get("trainee_tier", "generic"))),
        ScenarioChoiceParam("opponent_tier", "Opponent tier", ("generic", "amateur", "semi_pro", "premier_league"), str(_cfg.get("opponent_tier", "generic"))),
        ScenarioParam("ball_max_speed_mps", "Ball max speed (m/s)", 0.0, 60.0, 0.5, float(_cfg.get("ball_max_speed_mps", 10.0))),
        ScenarioParam("ball_max_dist_from_trainee_m", "Ball max dist from trainee (m)", 1.0, 75.0, 1.0, float(_cfg.get("ball_max_dist_from_trainee_m", 45.0))),
        ScenarioParam("stamina_min", "Stamina min", 0.0, 1.0, 0.05, float(_cfg.get("stamina_min", 0.3))),
        ScenarioParam("stamina_max", "Stamina max", 0.0, 1.0, 0.05, float(_cfg.get("stamina_max", 1.0))),
        ScenarioParam("restitution_sigma", "Restitution σ", 0.0, 1.0, 0.01, float(_cfg.get("restitution_sigma", 0.08))),
        ScenarioParam("decision_interval_ms", "Decision interval (ms)", 50.0, 2000.0, 50.0, DECISION_INTERVAL_MS_DEFAULT),
    ]

    def _resolve_trainer_from_name(ckpt_name: str, use_rules: bool):
        if use_rules:
            return None
        # Map friendly name back to full path
        # Match by "runN/filename" label (new format) or plain filename (fallback)
        match_path = next(
            (c for c in checkpoints if _ckpt_label(c) == ckpt_name or Path(c).name == ckpt_name),
            None,
        )
        if match_path is None:
            return None
        return _get_trainer(match_path)

    def build(
        rng_reduction: float = 0.3,
        *,
        trainee_checkpoint: str = ckpt_default,
        trainee_rules: bool = False,
        trainee_immobile: bool = False,
        opponent_checkpoint: str = ckpt_default,
        opponent_rules: bool = False,
        opponent_immobile: bool = False,
        trainee_tier: str = str(_cfg.get("trainee_tier", "generic")),
        opponent_tier: str = str(_cfg.get("opponent_tier", "generic")),
        ball_max_speed_mps: float = float(_cfg.get("ball_max_speed_mps", 10.0)),
        ball_max_dist_from_trainee_m: float = float(_cfg.get("ball_max_dist_from_trainee_m", 45.0)),
        stamina_min: float = float(_cfg.get("stamina_min", 0.3)),
        stamina_max: float = float(_cfg.get("stamina_max", 1.0)),
        restitution_sigma: float = float(_cfg.get("restitution_sigma", 0.08)),
        decision_interval_ms: float = DECISION_INTERVAL_MS_DEFAULT,
        **_ignored,  # absorb sim_dt_s etc injected by ScenarioEnv
    ) -> Match:
        # immobile takes priority over both the checkpoint and rules-based
        # override — an immobile player never gets a trainer and never gets
        # Phase1RulesAI (ai stays None, current_order set directly instead).
        trainee_trainer = None if trainee_immobile else _resolve_trainer_from_name(trainee_checkpoint, trainee_rules)
        opponent_trainer = None if opponent_immobile else _resolve_trainer_from_name(opponent_checkpoint, opponent_rules)
        decision_interval_ticks = max(1, round(decision_interval_ms / 1000.0 * UI_TICK_HZ))

        # opponent_rules_prob/opponent_immobile_prob=0.0: this scenario has
        # its OWN complete, explicit control over both players via the UI's
        # checkpoint/rules/immobile params below -- without this,
        # build_1v1_scenario falls back to ITS OWN defaults
        # (opponent_immobile_prob=1.0), unconditionally builds an opponent
        # JogOrder (see JogOrder's own docstring: "never completes" -- it
        # re-asserts a FIXED direction every tick via the engine's own order
        # execution, unconditionally, regardless of what player.ai does),
        # and hands it to code below that only ever meant to override
        # `.ai`, not clean up a `.current_order` it didn't know existed.
        # That mismatch was a real, shipped bug: the opponent's NeuralPlayerAI
        # was correctly assigned and correctly deciding every tick, but the
        # engine kept re-applying the never-cleared JogOrder right on top of
        # it every tick after, so the opponent visibly just jogged in a fixed
        # line regardless of what the network wanted -- indistinguishable
        # from "immobile" to the eye, and to a naive "did it move at all"
        # check. Passing explicit zero probabilities here means
        # build_1v1_scenario's own roll always lands on its "neural" branch
        # (ai=None, no order touched at all -- see its own opponent-roll
        # comment), leaving a clean slate for the assignment below with
        # nothing left over to clean up, rather than relying on remembering
        # to undo a side effect of a roll this scenario never wanted run.
        match = build_1v1_scenario(
            rng_reduction,
            trainee_tier=trainee_tier,
            opponent_tier=opponent_tier,
            ball_max_speed_mps=ball_max_speed_mps,
            restitution_sigma=restitution_sigma,
            ball_max_dist_from_trainee_m=ball_max_dist_from_trainee_m,
            stamina_min=stamina_min,
            stamina_max=stamina_max,
            opponent_rules_prob=0.0,
            opponent_immobile_prob=0.0,
        )

        # Neural driving goes through NeuralPlayerAI directly (Match.step()
        # calls player.ai.act() once per tick, which already handles the
        # decision-interval gating/cached-direction-reapplication this used
        # to hand-roll via a bespoke on_tick hook -- see git history for the
        # old _apply_neural_action/_reapply_cached_neural_action, removed
        # because that hook fires TWICE per physics tick (ScenarioLoop.step()
        # calls definition.on_tick both before AND after match.step(), see
        # its own docstring), silently double-counting a hand-rolled
        # decision-interval counter and leaving the opponent's ai at None
        # forever in at least one code path -- a real, previously-shipped
        # bug this refactor removes by construction, not by patching it.
        #
        # current_order is explicitly cleared in the non-immobile branches
        # below too (belt-and-suspenders, not just relying on the clean-slate
        # roll above) -- a player driven by ai must never also have a stale
        # current_order lying around: the engine's order-processing step
        # executes ANY current_order every tick unconditionally, regardless
        # of what ai.act() just decided, and would silently win/fight over
        # desired_direction/desired_speed_mode exactly like the JogOrder bug
        # above -- this is a real, general hazard whenever code reassigns
        # `.ai` without also touching `.current_order`, not unique to jogging.
        try:
            if trainee_immobile:
                # ai stays None (see JogOrder's own docstring) -- current_order
                # set directly so Match._apply_movement (which now RAISES if a
                # player has no movement intent, rather than silently coasting)
                # is satisfied.
                _trainee = match.player_by_id("trainee")
                _trainee.ai = None
                _trainee.current_order = JogOrder(direction=_trainee.velocity)
            elif trainee_trainer is not None:
                _trainee = match.player_by_id("trainee")
                _trainee.current_order = None
                _trainee.ai = NeuralPlayerAI(
                    trainee_trainer._sample_action,
                    decision_interval_ticks=decision_interval_ticks,
                    max_episode_s=PHASE1_MAX_EPISODE_S,
                )
            else:
                _trainee = match.player_by_id("trainee")
                _trainee.current_order = None
                _trainee.ai = Phase1RulesAI()
        except KeyError:
            pass
        try:
            if opponent_immobile:
                _opponent = match.player_by_id("opponent")
                _opponent.ai = None
                _opponent.current_order = JogOrder(direction=_opponent.velocity)
            elif opponent_trainer is not None:
                _opponent = match.player_by_id("opponent")
                _opponent.current_order = None
                _opponent.ai = NeuralPlayerAI(
                    opponent_trainer._sample_action,
                    decision_interval_ticks=decision_interval_ticks,
                    max_episode_s=PHASE1_MAX_EPISODE_S,
                )
            else:
                _opponent = match.player_by_id("opponent")
                _opponent.current_order = None
                _opponent.ai = Phase1RulesAI()
        except KeyError:
            pass

        # This build() already gave BOTH players their final, explicit .ai
        # (checkpoint-picked NeuralPlayerAI, Phase1RulesAI, or immobile
        # JogOrder) from the UI's own params -- not from build_1v1_scenario's
        # opponent-type roll, which was only forced to its "neural" branch
        # above for a clean current_order slate (see the opponent_rules_prob/
        # opponent_immobile_prob=0.0 comment on the build_1v1_scenario call).
        # That means match._opponent_use_rules_ai/_opponent_is_immobile are
        # now ALWAYS False/False here regardless of what the user actually
        # selected, so app.py's generic _wire_fresh_match ->
        # maybe_assign_neural_opponent("opponent", ...) would otherwise treat
        # every trial as "rolled neural" and silently re-assign the opponent
        # a SECOND NeuralPlayerAI (always the latest checkpoint) on top of
        # whatever this function just picked -- overriding a user-selected
        # non-latest checkpoint, or a rules/immobile choice, right after this
        # function set it correctly. Marking the match as fully wired here
        # makes maybe_assign_neural_opponent a guaranteed no-op on it.
        match._ai_fully_wired = True

        tr_label = "immobile" if trainee_immobile else (trainee_checkpoint if not trainee_rules else "rules")
        op_label = "immobile" if opponent_immobile else (opponent_checkpoint if not opponent_rules else "rules")
        log_ui.info(f"Phase 1 UI: trainee={tr_label}  opponent={op_label}")
        return match

    # Attach the params list so the SCENARIOS entry can reference it
    build._phase1_params = params_list  # type: ignore[attr-defined]

    return build, None


def _phase1_outcome_remap(match: Match, outcome: str, last_toucher_id: str | None) -> str:
    """Adapter: maps ScenarioLoop's (match, outcome, last_toucher_id) signature to
    remap_phase1_outcome() in outcome.py so there is one source of truth."""
    from footballcoach.ai.env.outcome import remap_phase1_outcome
    # box_terminal / opponent_box_terminal are already resolved by
    # detect_trial_outcome -> ScenarioLoop._trial_outcome before we get here,
    # so outcome is already the final raw key from that function; we only need
    # the goal/miss/invalid transforms that live in remap_phase1_outcome.
    return remap_phase1_outcome(
        outcome,
        last_ball_toucher_id=last_toucher_id,
        box_terminal=(outcome == "box_possession"),
        opponent_box_terminal=False,
        timeout=(outcome == "timeout"),
    )


def _find_latest_phase1_checkpoint_dir() -> str:
    """Return the highest-numbered checkpoints/phase1_run* directory found at import time."""
    import glob
    import re
    dirs = glob.glob("checkpoints/phase1_run*/")
    def _run_num(d: str) -> int:
        m = re.search(r"phase1_run(\d+)", d)
        return int(m.group(1)) if m else -1
    dirs = sorted(dirs, key=_run_num)
    return dirs[-1].rstrip("/") if dirs else "checkpoints/phase1_run1"


_phase1_build, _phase1_on_tick = _make_phase1_scenario_pair(_find_latest_phase1_checkpoint_dir())


def build_1v1_scenario(
    rng_reduction: float = 0.3,
    *,
    trainee_tier: str | None = None,
    opponent_tier: str | None = None,
    ball_max_speed_mps: float | None = None,
    restitution_sigma: float | None = None,
    ball_max_dist_from_trainee_m: float | None = None,
    stamina_min: float | None = None,
    stamina_max: float | None = None,
    trainee_team: "Team | None" = None,
    sim_dt_s: float = 1.0 / 30.0,
    opponent_rules_prob: float = 0.0,
    opponent_immobile_prob: float = 1.0,
    opponent_min_dist_m: float | None = None,
    opponent_max_dist_m: float | None = None,
    seed: int | None = None,
) -> Match:
    """Phase 1 curriculum: 1v1 get-possession/move-toward-goal.

    Both players and the ball are placed randomly across the full pitch with
    randomised attributes, stamina, headings, and (for the ball) velocity,
    spin, and restitution coefficient.

    ``seed``: when given, the ENTIRE scenario (trainee/opponent team choice,
    positions, attributes, stamina, headings, ball placement/velocity/spin/
    restitution, and the opponent-type roll) is fully deterministic for that
    seed value -- every draw below goes through the single ``rng`` instance.
    Used by the shared seeded-eval helper (ai/eval/seeded_eval.py) to replay
    a fixed set of scenarios for pre-training and every rollout's periodic
    eval. ``None`` (default) = fully random, matching prior behaviour;
    PPO training itself is NOT seeded by this -- only evaluation call sites
    pass a seed.

    Trainee (random team each episode by default, player_id='trainee') has no
    initial order so the AI can drive it.  Opponent gets the opposite team and
    is immobile (no order) - the first sub-phase of the phase-1 curriculum.
    Pass ``trainee_team`` to pin the side (used by tests / UI scenarios that
    need a fixed attacking direction).

    An immobile opponent can never reach ITS OWN box with possession by any
    means other than spawning inside it and getting lucky with a ball roll
    (it never moves) -- that coincidence is excluded outright by re-rolling
    the opponent's spawn position out of their own box whenever this
    episode's opponent-type roll lands on immobile. Rules-based and neural
    opponents are unaffected (they can legitimately end up in their own box
    through real play).

    Ball velocity is resampled (up to 20 times) until a linear extrapolation
    at the initial speed for 2.0 seconds stays in bounds (ignoring friction -
    conservative).  If all attempts fail the ball starts at rest.

    Restitution: sampled from Gaussian(base_restitution, sigma=0.08),
    clamped to [0.2, 0.95].

    See ai_design_doc.md sections 4 and 9.2 for the full design rationale.
    """
    _cfg = _phase1_scenario_cfg()
    if trainee_tier is None:
        trainee_tier = str(_cfg.get("trainee_tier", "generic"))
    if opponent_tier is None:
        opponent_tier = str(_cfg.get("opponent_tier", "generic"))
    if ball_max_speed_mps is None:
        ball_max_speed_mps = float(_cfg.get("ball_max_speed_mps", 10.0))
    if restitution_sigma is None:
        restitution_sigma = float(_cfg.get("restitution_sigma", 0.08))
    if ball_max_dist_from_trainee_m is None:
        ball_max_dist_from_trainee_m = float(_cfg.get("ball_max_dist_from_trainee_m", 45.0))
    if stamina_min is None:
        stamina_min = float(_cfg.get("stamina_min", 0.3))
    if stamina_max is None:
        stamina_max = float(_cfg.get("stamina_max", 1.0))
    if opponent_min_dist_m is None:
        opponent_min_dist_m = float(_cfg.get("opponent_min_dist_m", 0.0))
    if opponent_max_dist_m is None:
        opponent_max_dist_m = float(_cfg.get("opponent_max_dist_m", 9999.0))

    rng = random.Random(seed)
    pitch = Pitch.standard()

    def _rand_pos() -> Vector3:
        return Vector3(
            rng.uniform(-pitch.half_length + 1.5, pitch.half_length - 1.5),
            rng.uniform(-pitch.half_width + 1.5, pitch.half_width - 1.5),
            0.0,
        )

    # --- Randomise which end the trainee attacks (unless pinned by caller) ---
    if trainee_team is None:
        trainee_team = rng.choice([Team.LEFT, Team.RIGHT])
    opponent_team = Team.RIGHT if trainee_team == Team.LEFT else Team.LEFT

    _movement_params = MovementParams.from_config()

    def _rand_capped_velocity(attrs: PlayerAttributes, stamina: float, direction_rad: float) -> Vector3:
        """Magnitude uniform in [0, this player's own effective_top_speed] at
        their sampled stamina, in the given direction -- has_ball=False
        (nobody starts in possession here) and is_goalkeeper=False (no
        goalkeepers in this 1v1 scenario). direction_rad is passed in
        (rather than drawn here) so the caller can reuse the SAME angle for
        heading_rad -- a player's heading and velocity should never be two
        independent random draws (see below)."""
        max_speed = effective_top_speed(_movement_params, attrs.top_speed, stamina, has_ball=False)
        speed = rng.uniform(0.0, max_speed)
        return Vector3(math.cos(direction_rad) * speed, math.sin(direction_rad) * speed, 0.0)

    # --- Trainee (random team) ---
    trainee_attrs = generate_attributes(tier=trainee_tier, rng=rng)
    trainee = Player.create("trainee", trainee_team, trainee_attrs, position=_rand_pos())
    trainee.stamina = rng.uniform(stamina_min, stamina_max)
    # heading_rad and velocity share ONE random draw, not two independent
    # ones -- a player facing a direction other than the one they're moving
    # in is a real Match invariant violation (see Match.__init__'s heading/
    # velocity alignment check) that used to happen here silently.
    _trainee_dir_rad = rng.uniform(-math.pi, math.pi)
    trainee.heading_rad = _trainee_dir_rad
    trainee.velocity = _rand_capped_velocity(trainee_attrs, trainee.stamina, _trainee_dir_rad)

    # --- Opponent (opposite team) --- immobile, no order ---
    opponent_attrs = generate_attributes(tier=opponent_tier, rng=rng)
    # Opponent placement: respect min/max distance from trainee if configured
    _opp_pos = _rand_pos()
    if opponent_min_dist_m > 0.0 or opponent_max_dist_m < 9999.0:
        for _opp_attempt in range(50):
            _opp_pos = _rand_pos()
            _opp_dist = math.hypot(
                _opp_pos.x - trainee.position.x,
                _opp_pos.y - trainee.position.y,
            )
            if opponent_min_dist_m <= _opp_dist <= opponent_max_dist_m:
                break
    opponent = Player.create("opponent", opponent_team, opponent_attrs, position=_opp_pos)
    opponent.stamina = rng.uniform(stamina_min, stamina_max)
    _opponent_dir_rad = rng.uniform(-math.pi, math.pi)
    opponent.heading_rad = _opponent_dir_rad
    # When this opponent ends up immobile (no AI, the common case --
    # opponent_immobile_prob default 1.0), it's given a JogOrder below in
    # this same direction instead of being left to coast forever on this
    # raw initial velocity (Match._apply_movement's "no order this tick"
    # branch applies zero deceleration and never updates heading) -- see
    # JogOrder's own docstring (orders.py).
    opponent.velocity = _rand_capped_velocity(opponent_attrs, opponent.stamina, _opponent_dir_rad)

    # --- Ball: random placement within ball_max_dist_from_trainee_m of trainee, in bounds ---
    for _ball_attempt in range(50):
        ball_pos = _rand_pos()
        dist = math.hypot(
            ball_pos.x - trainee.position.x,
            ball_pos.y - trainee.position.y,
        )
        if dist <= ball_max_dist_from_trainee_m:
            break
    # _rand_pos already guarantees in-bounds; worst case we just use the last sample
    ball_vel = Vector3.zero()
    max_speed = ball_max_speed_mps
    for _attempt in range(20):
        speed = rng.uniform(0.0, max_speed)
        direction = rng.uniform(-math.pi, math.pi)
        vx = math.cos(direction) * speed
        vy = math.sin(direction) * speed
        # Conservative (no-friction) 2.5-second extrapolation check (lowered
        # from 3.0s -- the longer window was suppressing genuinely-escaping-
        # ball trajectories almost entirely: 20,000 recorded demo episodes
        # and a live 300-episode PPO-scenario sample both came back at 0%
        # "invalid" outcome. Still rules out a ball that's an instant
        # write-off, but lets through faster/closer-to-boundary draws that
        # take a bit longer to actually exit -- combined with
        # ball_max_dist_from_trainee_m, the trainee should now sometimes
        # genuinely fail to reach it in time. Empirically (300-episode
        # sample, at other threshold values tried along the way):
        # 3.0s->0% invalid, 2.0s->3.7%, 1.5s->9%.)
        if (abs(ball_pos.x + vx * 2.5) < pitch.half_length
                and abs(ball_pos.y + vy * 2.5) < pitch.half_width):
            ball_vel = Vector3(vx, vy, 0.0)
            break
        max_speed = max(1.0, max_speed * 0.7)

    ball_spin = Vector3(
        rng.gauss(0.0, 1.0),
        rng.gauss(0.0, 1.0),
        rng.gauss(0.0, 1.5),
    )
    ball = Ball.at_rest(ball_pos)
    ball.velocity = ball_vel
    ball.spin = ball_spin

    # --- Randomised ball restitution ---
    base_params = BallPhysicsParams.from_config()
    restitution = max(0.2, min(0.95, rng.gauss(base_params.bounce_restitution_vertical, restitution_sigma)))
    ball_params = replace(base_params, bounce_restitution_vertical=restitution)

    ui_cfg = load_gameplay_config().get("ui", {})
    match = Match(
        pitch=pitch,
        players=[trainee, opponent],
        ball=ball,
        rng_reduction=rng_reduction,
        rng=rng,
        ball_physics_params=ball_params,
        goal_linger_s=ui_cfg.get("goal_linger_s", 3.0),
        dt_s=sim_dt_s,
    )
    # Randomise opponent type per episode based on caller-supplied probabilities.
    # rules_prob + immobile_prob must be <= 1; the remainder becomes neural (secondary player).
    _r = rng.random()
    if _r < opponent_rules_prob:
        from footballcoach.rules_ai import Phase1RulesAI as _Phase1RulesAI
        opponent.ai = _Phase1RulesAI()
        match._opponent_use_rules_ai = True
        match._opponent_is_immobile = False
    elif _r < opponent_rules_prob + opponent_immobile_prob:
        # ai stays None (Phase1RulesAI.decide() and others key off `opponent.
        # ai is None` as their "can this opponent ever move/contest" signal
        # -- see JogOrder's own docstring for why this must not become a real
        # AI) -- but current_order is set directly so the opponent still
        # moves like a real player (proper accel/turn-rate/heading) instead
        # of coasting forever on its raw initial velocity.
        opponent.ai = None
        opponent.current_order = JogOrder(
            direction=Vector3(math.cos(_opponent_dir_rad), math.sin(_opponent_dir_rad), 0.0)
        )
        match._opponent_use_rules_ai = False
        match._opponent_is_immobile = True
    else:
        # Neural — ScenarioEnv assigns NeuralPlayerAI on reset when sample_action_fn is set.
        opponent.ai = None
        match._opponent_use_rules_ai = False
        match._opponent_is_immobile = False

    if match._opponent_is_immobile:
        # This re-roll only protects the STARTING position -- "immobile" now
        # jogs (see the JogOrder assigned above), so it can still walk into
        # its own box over the course of an episode even after spawning
        # clear of it. Not handled here (yet): revisit if that shows up as a
        # real problem in practice. It is also not perfectly frozen even
        # ignoring JogOrder: resolve_all_overlaps() (engine/collision.py)
        # pushes apart any overlapping ACTIVE players purely from physical
        # collision, with no regard for whether either side has an AI. If
        # the trainee repeatedly contests the ball right next to the
        # opponent's own box, each contact can nudge the immobile opponent
        # a little -- a spawn point merely OUTSIDE the box with zero margin
        # could still get walked across the line over the course of an
        # episode. A real clearance margin (not just non-overlap) is needed
        # to make this genuinely impossible rather than just less likely.
        #
        # This matters because reaching its own box with possession is the
        # phase-1 terminal condition that, from the opponent's side, reads
        # as a trainee loss -- for an opponent that can never chase or even
        # meaningfully hold a defensive line, that's a pure coincidence
        # with zero skill/intent behind it. It produced badly-mispredicted,
        # high-variance value-loss rows (opponent_box_possession MSE ~48
        # against a return std of ~1.8) while also unfairly zeroing the
        # trainee's terminal reward for what is, from its perspective, a
        # coin flip. Re-rolling the spawn point (with margin) is sufficient
        # since position is otherwise stable -- still respects
        # opponent_min_dist_m/opponent_max_dist_m if the caller set them.
        _immobile_own_box_margin_m = 2.0

        def _immobile_pos_ok(pos: Vector3) -> bool:
            half_box_w = pitch.box_width_m / 2.0 + _immobile_own_box_margin_m
            if opponent_team == Team.LEFT:
                in_x = pos.x <= -pitch.half_length + pitch.box_length_m + _immobile_own_box_margin_m
            else:
                in_x = pos.x >= pitch.half_length - pitch.box_length_m - _immobile_own_box_margin_m
            in_y = -half_box_w <= pos.y <= half_box_w
            if in_x and in_y:
                return False
            if opponent_min_dist_m > 0.0 or opponent_max_dist_m < 9999.0:
                d = math.hypot(pos.x - trainee.position.x, pos.y - trainee.position.y)
                if not (opponent_min_dist_m <= d <= opponent_max_dist_m):
                    return False
            return True

        if not _immobile_pos_ok(opponent.position):
            for _reroll_attempt in range(50):
                _candidate = _rand_pos()
                if _immobile_pos_ok(_candidate):
                    opponent.position = _candidate
                    break
            # else: 50 attempts exhausted (only possible with a pathologically
            # tight opponent_min_dist_m/max_dist_m window) -- fall back to
            # whatever the last candidate was, same lenient philosophy as the
            # ball-placement loop above.

    return match


def _build_1v1_phase1_ui_scenario(rng_reduction: float = 0.3, **kwargs) -> Match:
    """SCENARIOS["1v1_phase1"]'s ``build`` -- thin wrapper around
    ``build_1v1_scenario`` that explicitly assigns the trainee's AI.

    ``build_1v1_scenario`` itself deliberately leaves ``trainee.ai`` unset
    (see its own docstring: "Trainee ... has no initial order so the AI can
    drive it") so its real callers -- PPO training/eval/recording
    (ai/scripts/train.py, evaluate.py, record_demonstrations.py,
    ppo_trainer.py, ...) -- can assign whatever drives the trainee
    themselves (a NeuralPlayerAI wired up outside player.ai, or nothing at
    all if the env steps it directly); build_1v1_scenario is shared with all
    of those non-test callers, so it must not gain a hardcoded trainee AI.
    This "1v1: Phase 1 get possession (rules-based only)" UI/test scenario
    entry has no such external driver, so -- mirroring exactly what
    ai/scripts/replay_episode.py's own UI launch path already does for the
    same reason (see its comment there) -- assign Phase1RulesAI here,
    locally, rather than changing the shared builder.
    """
    match = build_1v1_scenario(rng_reduction, **kwargs)
    try:
        match.player_by_id("trainee").ai = Phase1RulesAI()
    except KeyError:
        pass
    return match


# ---------------------------------------------------------------------------
# 1v1 escaping-ball scenario (deliberately produces "invalid" outcomes)
# ---------------------------------------------------------------------------

def build_1v1_escaping_ball_scenario(
    rng_reduction: float = 0.3,
    *,
    trainee_tier: str | None = None,
    opponent_tier: str | None = None,
    ball_speed_mps: float | None = None,
    stamina_min: float | None = None,
    stamina_max: float | None = None,
    trainee_team: "Team | None" = None,
    sim_dt_s: float = 1.0 / 30.0,
    seed: int | None = None,
) -> Match:
    """1v1 variant deliberately engineered to produce a genuine "invalid"
    outcome (ball leaves the pitch with nobody having touched it -- see
    ai/env/outcome.py's remap_phase1_outcome) at a realistic, non-trivial
    rate: the ball starts near a random pitch corner aimed outward (away
    from the pitch centre, toward that corner's own boundary), and the
    trainee starts near the DIAGONALLY OPPOSITE corner -- far enough away
    that reaching the ball before it exits is not realistic, unlike
    build_1v1_scenario's own ball placement/velocity sampling, which
    deliberately keeps the ball within ball_max_dist_from_trainee_m and
    resamples any velocity that would exit within 3 seconds specifically
    to AVOID this outcome (see that function's own docstring) -- this is
    the intentional inverse of that safeguard, for recording/eval runs that
    specifically want "invalid" coverage rather than trying to eliminate it.

    Opponent is always immobile (this scenario is about the trainee's own
    inability to reach a fast, distant ball -- an opponent isn't part of
    the point). The trainee still gets no initial order, so real AI
    (e.g. Phase1RulesAI during demo recording) genuinely attempts the chase
    and realistically fails -- the recorded states are "gave chase, didn't
    make it", not an artificially inert trainee.

    ``seed``: fully deterministic for a given value (matches
    build_1v1_scenario's own convention), including which corner is picked.
    """
    _cfg = _phase1_scenario_cfg()
    if trainee_tier is None:
        trainee_tier = str(_cfg.get("trainee_tier", "generic"))
    if opponent_tier is None:
        opponent_tier = str(_cfg.get("opponent_tier", "generic"))
    if ball_speed_mps is None:
        ball_speed_mps = float(_cfg.get("ball_max_speed_mps", 10.0))
    if stamina_min is None:
        stamina_min = float(_cfg.get("stamina_min", 0.3))
    if stamina_max is None:
        stamina_max = float(_cfg.get("stamina_max", 1.0))

    rng = random.Random(seed)
    pitch = Pitch.standard()

    if trainee_team is None:
        trainee_team = rng.choice([Team.LEFT, Team.RIGHT])
    opponent_team = Team.RIGHT if trainee_team == Team.LEFT else Team.LEFT

    # --- Pick a ball corner and its diagonally opposite trainee corner ---
    corner_margin_m = 3.0  # how far in from the true corner point to start (avoid the exact edge)
    corner_signs = rng.choice([(1, 1), (1, -1), (-1, 1), (-1, -1)])
    bx_sign, by_sign = corner_signs
    ball_corner = Vector3(
        bx_sign * (pitch.half_length - corner_margin_m),
        by_sign * (pitch.half_width - corner_margin_m),
        0.0,
    )
    trainee_corner = Vector3(-ball_corner.x, -ball_corner.y, 0.0)

    def _jitter(base: Vector3, spread_m: float) -> Vector3:
        return Vector3(
            max(-pitch.half_length + 0.5, min(pitch.half_length - 0.5, base.x + rng.uniform(-spread_m, spread_m))),
            max(-pitch.half_width + 0.5, min(pitch.half_width - 0.5, base.y + rng.uniform(-spread_m, spread_m))),
            0.0,
        )

    ball_pos = _jitter(ball_corner, 4.0)
    trainee_pos = _jitter(trainee_corner, 4.0)

    # --- Trainee ---
    trainee_attrs = generate_attributes(tier=trainee_tier, rng=rng)
    trainee = Player.create("trainee", trainee_team, trainee_attrs, position=trainee_pos)
    trainee.stamina = rng.uniform(stamina_min, stamina_max)
    trainee.heading_rad = rng.uniform(-math.pi, math.pi)

    # --- Opponent: always immobile, parked well out of the way ---
    opponent_attrs = generate_attributes(tier=opponent_tier, rng=rng)
    opponent = Player.create("opponent", opponent_team, opponent_attrs, position=Vector3(0.0, 0.0, 0.0))
    opponent.stamina = rng.uniform(stamina_min, stamina_max)
    opponent.heading_rad = rng.uniform(-math.pi, math.pi)
    # ai stays None (see JogOrder's own docstring for why) -- current_order
    # set directly instead, so Match._apply_movement (which now RAISES if a
    # player has no movement intent, rather than silently coasting) is
    # satisfied. opponent.velocity is zero here, so JogOrder resolves to
    # "stay put forever" via its own zero-direction handling -- exactly
    # matching this scenario's "always immobile, parked" intent.
    opponent.ai = None
    opponent.current_order = JogOrder(direction=opponent.velocity)

    # --- Ball: aimed from its corner toward the NEAREST boundary intersection
    # (i.e. straight out through that same corner), small angular jitter so
    # episodes vary without ever pointing back into the pitch. ---
    outward_angle = math.atan2(ball_corner.y, ball_corner.x)
    angle = outward_angle + rng.uniform(-math.radians(20.0), math.radians(20.0))
    speed = rng.uniform(0.6 * ball_speed_mps, ball_speed_mps)
    ball_vel = Vector3(math.cos(angle) * speed, math.sin(angle) * speed, 0.0)
    ball_spin = Vector3(rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.5))
    ball = Ball.at_rest(ball_pos)
    ball.velocity = ball_vel
    ball.spin = ball_spin

    base_params = BallPhysicsParams.from_config()
    restitution = max(0.2, min(0.95, rng.gauss(base_params.bounce_restitution_vertical, 0.08)))
    ball_params = replace(base_params, bounce_restitution_vertical=restitution)

    ui_cfg = load_gameplay_config().get("ui", {})
    match = Match(
        pitch=pitch,
        players=[trainee, opponent],
        ball=ball,
        rng_reduction=rng_reduction,
        rng=rng,
        ball_physics_params=ball_params,
        goal_linger_s=ui_cfg.get("goal_linger_s", 3.0),
        dt_s=sim_dt_s,
    )
    match._opponent_use_rules_ai = False
    match._opponent_is_immobile = True
    return match


# ---------------------------------------------------------------------------
# Mark standoff stability scenario
# ---------------------------------------------------------------------------

def build_mark_standoff_scenario(
    rng_reduction: float = 0.3,
    *,
    marker_skill: float = 0.8,
) -> Match:
    """MarkOrder standoff (3-player): carrier (Team.RIGHT) holds the ball at (-5, 8);
    target (Team.RIGHT) is an off-ball runner at (20, 0); marker (Team.LEFT) marks
    the target, standing 1.5 m between them and the carrier.
    Watch whether the marker settles on the ideal standoff point or oscillates."""
    pitch = Pitch.standard()
    rng = random.Random()

    carrier_pos = Vector3(-5, 8, 0)
    target_pos = Vector3(20, 0, 0)
    # Start marker away from the standoff so the approach is visible.
    marker_start = Vector3(15, 3, 0)

    carrier = Player.create(
        "carrier", Team.RIGHT,
        PlayerAttributes.average(0.6),
        position=carrier_pos,
    )
    target = Player.create(
        "target", Team.RIGHT,
        PlayerAttributes.average(0.5),
        position=target_pos,
    )
    marker = Player.create(
        "marker", Team.LEFT,
        PlayerAttributes.average(marker_skill),
        position=marker_start,
    )

    ball = Ball.at_rest(carrier_pos)
    ball.set_initial_possession(carrier.player_id)
    ui_cfg = load_gameplay_config().get("ui", {})
    match = Match(
        pitch=pitch, players=[carrier, target, marker], ball=ball,
        rng_reduction=rng_reduction, rng=rng,
        goal_linger_s=ui_cfg.get("goal_linger_s", 3.0),
    )
    carrier.stop()  # carrier stays put holding the ball
    # StopOrder self-completes the instant the player is already at rest
    # (carrier and target both start stationary), so without a fallback AI
    # both would be left with current_order=None after tick 1. marker's
    # MarkOrder is persistent (never auto-completes) so it needs no fallback.
    carrier.ai = StopWhenIdleAI()
    target.ai = StopWhenIdleAI()
    marker.mark_player(target_player_id=target.player_id)
    return match


# ---------------------------------------------------------------------------
# Penalty corner accuracy scenario
# ---------------------------------------------------------------------------

def build_penalty_corner_accuracy_scenario(
    rng_reduction: float = 0.3,
    *,
    kicker_precision: float = 0.5,
) -> Match:
    """Penalty kicker running at full pace, aiming at the bottom corner of the
    right goal, no goalkeeper. Mirrors test_penalty_balance.py exactly:
    compensate_for_run=False, power_fraction=0.8.
    At precision=0.5 the target is 50-80% scored; at 0.8 it is 85-95%."""
    pitch = Pitch.standard()
    penalty_spot = pitch.penalty_spot(left=False)
    rng = random.Random()

    kicker_attrs = PlayerAttributes(
        top_speed=0.7, acceleration=0.7, stamina=1.0,
        kick_precision=kicker_precision, kick_power=0.7,
        dribbling=0.5, ball_control=0.5, tackling=0.5,
    )
    kicker = Player.create("kicker", Team.LEFT, kicker_attrs, position=penalty_spot)

    mvmt = MovementParams.from_config()
    v_run = effective_top_speed(
        mvmt, kicker.attributes.top_speed, kicker.stamina,
        has_ball=True, ball_control_attr=kicker.attributes.ball_control,
    )
    kicker.velocity = Vector3(v_run, 0.0, 0.0)
    kicker.heading_rad = 0.0  # facing +x toward right goal

    ball = Ball.at_rest(penalty_spot)
    ball.set_initial_possession(kicker.player_id)
    ui_cfg = load_gameplay_config().get("ui", {})
    match = Match(
        pitch=pitch, players=[kicker], ball=ball,
        rng_reduction=rng_reduction, rng=rng,
        goal_linger_s=ui_cfg.get("goal_linger_s", 3.0),
    )
    # Extreme corner: 0.15 m inside the post, 0.25 m off the ground.
    corner_offset_y = pitch.goal_width_m / 2.0 - 0.15
    aim_point = pitch.right_goal_centre + Vector3(0, corner_offset_y, 0.25)
    kicker.current_order = KickOrder(
        aim_point=aim_point, power_fraction=0.9,
        spin=Vector3.zero(), compensate_for_run=False,
    )
    # KickOrder always completes in a single tick; nothing else drives the
    # kicker afterwards, so hold position instead of crashing next tick.
    kicker.ai = StopWhenIdleAI()
    return match


# ---------------------------------------------------------------------------
# GK far-post speed scenario
# ---------------------------------------------------------------------------

def build_gk_far_post_scenario(
    rng_reduction: float = 0.3,
    *,
    gk_attr: float = 0.5,
    shot_distance_m: float = 25.0,
) -> Match:
    """GK starts pinned at the near post; shot aimed at the far corner.
    Mirrors test_save_balance.py's far-post test: shooter at shot_distance_m
    from the left goal, power=0.9, precision=0.95, compensate_for_run=False.
    Vary gk_attr to see whether speed/accel makes a meaningful difference."""
    pitch = Pitch.standard()
    half_goal_w = pitch.goal_width_m / 2.0
    rng = random.Random()

    gk_attrs = PlayerAttributes(
        top_speed=gk_attr, acceleration=gk_attr, stamina=0.8,
        kick_precision=0.5, kick_power=0.5, dribbling=0.5,
        ball_control=0.6, tackling=0.5,
    )
    gk = Player.create(
        "keeper", Team.LEFT, gk_attrs,
        position=pitch.left_goal_centre + Vector3(0, -half_goal_w + 0.3, 0),
        is_goalkeeper=True,
    )

    shot_x = -(pitch.half_length - shot_distance_m)
    shooter_attrs = PlayerAttributes(
        top_speed=0.7, acceleration=0.7, stamina=0.8,
        kick_precision=0.95, kick_power=0.9,
        dribbling=0.5, ball_control=0.5, tackling=0.5,
    )
    shooter = Player.create("shooter", Team.RIGHT, shooter_attrs,
                            position=Vector3(shot_x, 0, 0))

    mvmt = MovementParams.from_config()
    run_speed = effective_top_speed(
        mvmt, shooter.attributes.top_speed, shooter.stamina,
        has_ball=True, ball_control_attr=shooter.attributes.ball_control,
    )
    shooter.velocity = Vector3(-run_speed, 0.0, 0.0)
    shooter.heading_rad = math.pi  # facing -x toward left goal

    ball = Ball.at_rest(shooter.position)
    ball.set_initial_possession(shooter.player_id)
    aim_point = pitch.left_goal_centre + Vector3(0, half_goal_w - 0.3, 1.8)

    ui_cfg = load_gameplay_config().get("ui", {})
    match = Match(
        pitch=pitch, players=[gk, shooter], ball=ball,
        rng_reduction=rng_reduction, rng=rng,
        goal_linger_s=ui_cfg.get("goal_linger_s", 3.0),
    )
    gk.current_order = SaveOrder()
    shooter.current_order = KickOrder(
        aim_point=aim_point, power_fraction=0.95,
        spin=Vector3.zero(), compensate_for_run=False,
    )
    # SaveOrder self-completes once a goal is scored (stands down during the
    # goal linger); KickOrder always completes in a single tick. Neither
    # player has any other AI to pick up movement intent afterwards.
    gk.ai = StopWhenIdleAI()
    shooter.ai = StopWhenIdleAI()
    return match


# ---------------------------------------------------------------------------
# SCENARIOS list (Phase H: trimmed to 6, all parameterized)
# ---------------------------------------------------------------------------

def _build_2v2_params() -> list:
    """Build ScenarioParam list for the 2v2 scenario, pulling defaults from scenarios.json."""
    c = _2v2_cfg()
    return [
        ScenarioParam("attacker_skill_min", "Attacker skill min", 0.3, 1.0, 0.05, float(c.get("attacker_skill_min", 0.7))),
        ScenarioParam("attacker_skill_max", "Attacker skill max", 0.3, 1.0, 0.05, float(c.get("attacker_skill_max", 0.85))),
        ScenarioParam("defender_skill_min", "Defender skill min", 0.3, 1.0, 0.05, float(c.get("defender_skill_min", 0.55))),
        ScenarioParam("defender_skill_max", "Defender skill max", 0.3, 1.0, 0.05, float(c.get("defender_skill_max", 0.7))),
        ScenarioParam("gk_skill_min", "GK skill min", 0.3, 1.0, 0.05, float(c.get("gk_skill_min", 0.55))),
        ScenarioParam("gk_skill_max", "GK skill max", 0.3, 1.0, 0.05, float(c.get("gk_skill_max", 0.7))),
        ScenarioParam("shoot_immediately_probability", "Shoot immediately prob", 0.0, 1.0, 0.1, float(c.get("shoot_immediately_probability", 0.5))),
        ScenarioParam("attacker_a_dist_from_goal_m", "Attacker A: dist from goal (m)", 1.0, 50.0, 0.5, float(c.get("attacker_a_dist_from_goal_m", 18.0))),
        ScenarioParam("attacker_a_y_m", "Attacker A: lateral position (m)", -20.0, 20.0, 0.5, float(c.get("attacker_a_y_m", -3.16))),
        ScenarioParam("attacker_b_x_offset_m", "Attacker B: x offset behind A (m)", 0.0, 20.0, 0.5, float(c.get("attacker_b_x_offset_m", 4.0))),
        ScenarioParam("attacker_b_y_fraction", "Attacker B: y spread (0=same as A, 1=far side)", 0.0, 1.0, 0.1, float(c.get("attacker_b_y_fraction", 0.8))),
        ScenarioBoolParam("attacker_b_running", "Attacker B starts at full sprint", default=bool(c.get("attacker_b_running", True))),
        ScenarioParam("defender_dist_from_goal_m", "Defender: dist from goal (m)", 1.0, 50.0, 0.5, float(c.get("defender_dist_from_goal_m", 8.0))),
        ScenarioParam("defender_y_m", "Defender: lateral position (m)", -20.0, 20.0, 0.5, float(c.get("defender_y_m", 0.0))),
        ScenarioParam("pass_power_multiplier", "Pass power multiplier", 0.5, 3.0, 0.1, float(c.get("pass_power_multiplier", 1.5))),
    ]


SCENARIOS: list[ScenarioDefinition] = [
    # ---- AI scenarios (shown first in menu) ----
    ScenarioDefinition(
        key="phase1_neural_ai",
        label="Phase 1: Neural AI vs opponent (checkpoint picker)",
        description=(
            "1v1 Phase 1 scenario. Use the dropdowns to select a checkpoint for each player. "
            "Tick the 'rules-based override' checkbox to force that player to use the rules AI "
            "regardless of the checkpoint selection. Checkpoints are loaded from the most recent "
            "checkpoints/phase1_run*/ directory."
        ),
        build=_phase1_build,
        on_tick=_phase1_on_tick,
        params=_phase1_build._phase1_params,  # type: ignore[attr-defined]
        win_outcome="box_possession",  # matches seeded_eval win_outcome
        outcome_remap=_phase1_outcome_remap,
    ),
    ScenarioDefinition(
        key="1v1_phase1",
        label="1v1: Phase 1 get possession (rules-based only)",
        description=(
            "Both players randomly placed, random ball, random attributes. "
            "Phase 1 curriculum scenario. Trainee chases ball; opponent immobile."
        ),
        build=_build_1v1_phase1_ui_scenario,
        on_tick=_1v1_on_tick,
        params=[
            ScenarioParam("ball_max_speed_mps", "Ball max speed (m/s)", 0.0, 60.0, 0.5, 10.0),
            ScenarioParam("restitution_sigma", "Restitution randomness (sigma)", 0.0, 1.0, 0.01, 0.08),
        ],
        phase1_trainee_player_id="trainee",
    ),
    # ---- Balance scenarios ----
    ScenarioDefinition(
        key="save_close",
        label="Shot vs keeper (close range, mixed outcome)",
        description="Mid-tier striker shoots from varied distance; expect saves and goals.",
        build=build_close_range_save_scenario,
        params=[
            ScenarioParam("distance_min_m", "Min shot distance (m)", 0.5, 105.0, 0.5, 8.0),
            ScenarioParam("distance_max_m", "Max shot distance (m)", 1.0, 105.0, 0.5, 16.0),
            ScenarioParam("shooter_y_offset_m", "Shooter lateral offset (m)", 0.0, 34.0, 0.5, 5.0),
            ScenarioParam("shooter_precision_min", "Shooter precision min", 0.3, 1.0, 0.05, 0.65),
            ScenarioParam("shooter_precision_max", "Shooter precision max", 0.3, 1.0, 0.05, 0.85),
            ScenarioParam("gk_skill_min", "GK skill min", 0.3, 1.0, 0.05, 0.65),
            ScenarioParam("gk_skill_max", "GK skill max", 0.3, 1.0, 0.05, 0.85),
        ],
    ),
    ScenarioDefinition(
        key="pass",
        label="Ground pass (randomised distance)",
        description="Two Premier-League-tier players, randomised distance/angle pass.",
        build=build_pass_scenario,
        on_tick=_pass_on_tick,
        params=[
            ScenarioParam("max_distance_m", "Max pass distance (m)", 0.5, 105.0, 0.5, 30.0),
            ScenarioParam("attr_clamp_min", "Attr clamp min", 0.3, 1.0, 0.05, 0.70),
            ScenarioParam("attr_clamp_max", "Attr clamp max", 0.3, 1.0, 0.05, 0.80),
        ],
    ),
    ScenarioDefinition(
        key="tackle",
        label="Tackle challenge (randomised separation)",
        description="Defender chases and tackles a jogging attacker from varied distances.",
        build=build_tackle_scenario,
        params=[
            ScenarioParam("separation_min_m", "Min separation (m)", 0.5, 105.0, 0.5, 1.0),
            ScenarioParam("separation_max_m", "Max separation (m)", 0.5, 105.0, 0.5, 10.0),
            ScenarioParam("tackler_tackling_min", "Tackler tackling min", 0.0, 1.0, 0.05, 0.8),
            ScenarioParam("tackler_tackling_max", "Tackler tackling max", 0.0, 1.0, 0.05, 0.8),
            ScenarioParam("dribbler_dribbling_min", "Dribbler dribbling min", 0.0, 1.0, 0.05, 0.6),
            ScenarioParam("dribbler_dribbling_max", "Dribbler dribbling max", 0.0, 1.0, 0.05, 0.6),
        ],
    ),
    ScenarioDefinition(
        key="goal_to_goal_sprint",
        label="Sprint: goal-to-goal (max stats)",
        description="Max-stat player sprints from one goal line to the other (~105 m). Use to verify game speed.",
        build=build_goal_to_goal_sprint_scenario,
        on_tick=_sprint_on_tick,
        params=[
            ScenarioParam("top_speed", "Top speed attr", 0.0, 1.0, 0.05, 1.0),
            ScenarioParam("acceleration", "Acceleration attr", 0.0, 1.0, 0.05, 1.0),
        ],
    ),
    ScenarioDefinition(
        key="sprint",
        label="Sprint: random 5-waypoint course",
        description="A runner follows a random 5-leg course across the pitch.",
        build=build_sprint_scenario,
        on_tick=_sprint_on_tick,
        params=[
            ScenarioParam("leg_min_m", "Min leg length (m)", 0.5, 105.0, 0.5, 5.0),
            ScenarioParam("leg_max_m", "Max leg length (m)", 1.0, 105.0, 0.5, 25.0),
            ScenarioParam("runner_skill_min", "Runner skill min", 0.0, 1.0, 0.05, 0.7),
            ScenarioParam("runner_skill_max", "Runner skill max", 0.0, 1.0, 0.05, 0.8),
        ],
    ),
    ScenarioDefinition(
        key="2v2",
        label="2v2: pass and shoot vs defender+GK",
        description="Two attackers combine with a pass before shooting; one defender + GK.",
        build=build_2v2_scenario,
        on_tick=None,
        params=_build_2v2_params(),
    ),
    ScenarioDefinition(
        key="1v2",
        label="1v2: elite attacker vs. average defender+GK",
        description="Elite attacker (skill=0.9) runs at goal; average defender+GK defend.",
        build=build_1v2_scenario,
        on_tick=None,
        box_possession_terminal=False,  # attacker enters box to shoot; only end on goal/out/dispossession
        params=[
            ScenarioParam("attacker_skill", "Attacker skill", 0.3, 1.0, 0.05, 0.9),
            ScenarioParam("defender_skill", "Defender skill", 0.3, 1.0, 0.05, 0.55),
            ScenarioParam("gk_skill", "GK skill", 0.3, 1.0, 0.05, 0.55),
            ScenarioParam("attacker_start_min_m", "Attacker start min dist (m)", 0.5, 105.0, 0.5, 18.0),
            ScenarioParam("attacker_start_max_m", "Attacker start max dist (m)", 1.0, 105.0, 0.5, 32.0),
            ScenarioParam("move_fraction_min", "Move fraction min", 0.0, 1.0, 0.05, 0.25),
            ScenarioParam("move_fraction_max", "Move fraction max", 0.0, 1.0, 0.05, 0.50),
        ],
    ),
    ScenarioDefinition(
        key="repulsion_obstacle",
        label="Repulsion: ball carrier past stationary obstacle",
        description="Ball carrier runs x=-20 to x=+20 with a stationary player on the path. obstacle_on_path=1 is dead-centre, 0 is 5 m aside.",
        build=build_repulsion_obstacle_scenario,
        params=[
            ScenarioParam("obstacle_on_path", "Obstacle on path (1=centre, 0=5m aside)", 0.0, 1.0, 0.1, 1.0),
            ScenarioParam("attacker_skill", "Attacker skill", 0.3, 1.0, 0.05, 0.9),
        ],
    ),
    ScenarioDefinition(
        key="repulsion_obstacle_no_ball",
        label="Repulsion: player past stationary obstacle (no ball)",
        description="Same as the ball-carrier repulsion scenario, but the runner never takes possession -- isolates the avoidance steering from ball-carrying speed/dribble effects.",
        build=build_repulsion_obstacle_no_ball_scenario,
        params=[
            ScenarioParam("obstacle_on_path", "Obstacle on path (1=centre, 0=5m aside)", 0.0, 1.0, 0.1, 1.0),
            ScenarioParam("attacker_skill", "Attacker skill", 0.3, 1.0, 0.05, 0.9),
        ],
    ),
    ScenarioDefinition(
        key="mark_standoff",
        label="Mark standoff stability",
        description=(
            "Marker holds 1.5 m standoff between target and ball (target stationary, ball fixed). "
            "Mirrors test_marker_standoff_stability. Watch for oscillation vs. stable hovering."
        ),
        build=build_mark_standoff_scenario,
        params=[
            ScenarioParam("marker_skill", "Marker skill", 0.0, 1.0, 0.05, 0.8),
        ],
    ),
    ScenarioDefinition(
        key="penalty_corner_accuracy",
        label="Penalty: corner accuracy (no GK)",
        description=(
            "Kicker running at full pace aims at the bottom corner, no goalkeeper. "
            "At precision=0.5 the balance target is 50-80% scored; at 0.8 it is 85-95%. "
            "Mirrors test_penalty_balance.py exactly (compensate_for_run=False, power=0.8)."
        ),
        build=build_penalty_corner_accuracy_scenario,
        params=[
            ScenarioParam("kicker_precision", "Kicker precision", 0.0, 1.0, 0.05, 0.5),
        ],
    ),
    ScenarioDefinition(
        key="gk_far_post",
        label="GK far-post speed test",
        description=(
            "GK starts pinned at near post; shot aimed at far corner. "
            "Vary gk_attr to see whether speed/acceleration makes a meaningful difference. "
            "Mirrors test_save_balance.py's far-post test (shooter 25 m out, power=0.9, precision=0.95)."
        ),
        build=build_gk_far_post_scenario,
        params=[
            ScenarioParam("gk_attr", "GK top_speed + acceleration attr", 0.0, 1.0, 0.05, 0.5),
            ScenarioParam("shot_distance_m", "Shot distance from goal (m)", 5.0, 50.0, 1.0, 25.0),
        ],
    ),
]


# ---------------------------------------------------------------------------
# ScenarioLoop
# ---------------------------------------------------------------------------

@dataclass
class ScenarioLoop:
    """Runs a ScenarioDefinition repeatedly for a fixed number of trials,
    rebuilding a fresh Match after each one completes.  The UI drives it by
    calling ``step()`` once per frame; the caller inspects ``complete`` to
    know when all trials are done.

    After an outcome is detected, the match keeps running for ``linger_s``
    sim-seconds before the new trial is built (so the UI can watch the ball
    settle / celebrate a goal before resetting).

    Trial-end detection (in priority order):
    1. Ball crosses the touchline or goal line (out of bounds / goal).
    2. Scoreboard changed (goal scored into a net).
    3. Initial ball carrier released ball and ball resolved (saved, dispossessed).
    4. Timeout failsafe (default 500 ticks ≈ 16.7 s at 30 Hz).
    """

    definition: ScenarioDefinition
    max_trials: int = 0
    timeout_ticks: int = 500
    rng_reduction: float = 0.3
    kwargs: dict = field(default_factory=dict)
    linger_s: float = field(
        default_factory=lambda: load_gameplay_config().get("ui", {}).get("scenario_linger_s", 3.0)
    )
    terminal_outcomes: frozenset | None = None
    """Allowlist of outcome keys that may actually end a trial.  Any outcome
    returned by ``_trial_outcome`` that is *not* in this set is silently
    suppressed — the episode continues until a permitted outcome fires or the
    env's own timeout triggers.  ``None`` (default) permits all outcomes,
    which is the correct behaviour for UI scenarios.  Phase-1 training passes
    ``frozenset({"miss", "goal"})`` so that dispossessed/saved/box_possession/
    timeout never terminate the loop directly; the env catches box-possession
    and timeout itself with its own authoritative tick budget."""

    _trial_count: int = field(default=0, init=False, repr=False)
    _trial_tick: int = field(default=0, init=False, repr=False)
    _match: Match = field(init=False, repr=False)
    _initial_carrier_id: str | None = field(default=None, init=False, repr=False)
    _initial_scoreboard: tuple[int, int] = field(default=(0, 0), init=False, repr=False)
    _ball_released: bool = field(default=False, init=False, repr=False)
    # The Match object exactly as it stood on the trial's FINAL tick --
    # captured (a cheap reference save, not a copy) immediately before
    # _start_trial() reassigns self._match to a brand-new Match for the next
    # trial -- step() returns True only after the rebuild has already
    # happened, so self._match itself is unusable for reading the trial
    # that just ended; this is the one place that state survives.
    # last_completed_trial_toucher_id (property, below) derives from this
    # rather than keeping its own separate snapshot, for the same reason.
    # See last_completed_trial_match property below; record_demonstrations.py
    # uses this to record the TRUE final observation (the real recorded
    # physics tick the episode actually ended on), instead of the episode's
    # last recorded row always being the tick BEFORE termination.
    _last_completed_trial_match: Match | None = field(default=None, init=False, repr=False)
    outcomes: dict[str, int] = field(
        default_factory=dict,
        init=False, repr=False,
    )
    _pending_outcome: str | None = field(default=None, init=False, repr=False)
    _linger_remaining_s: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        # Pre-seed every raw outcome key at 0 so consumers (app.py's HUD
        # tally) can safely display/iterate any key without needing to
        # special-case ones that haven't happened yet this session. Outcomes
        # produced via definition.outcome_remap (e.g. phase-1's "invalid"/
        # "opponent_box_possession") are still added on the fly as before --
        # this only guarantees the raw detect_trial_outcome() vocabulary.
        from footballcoach.ai.env.outcome import RAW_OUTCOME_KEYS
        self.outcomes.update({key: 0 for key in RAW_OUTCOME_KEYS})
        self._start_trial()

    def _start_trial(self) -> None:
        self._match = self.definition.build(self.rng_reduction, **self.kwargs)
        self._trial_tick = 0
        self._initial_carrier_id = self._match.ball.possessed_by
        self._initial_scoreboard = (
            self._match.scoreboard.left_goals,
            self._match.scoreboard.right_goals,
        )
        self._ball_released = False
        self._pending_outcome = None
        self._linger_remaining_s = 0.0

    def _apply_remap(self, outcome: str) -> str:
        if self.definition.outcome_remap is not None:
            return self.definition.outcome_remap(
                self._match, outcome, self._match.ball.last_touched_by_player_id,
            )
        return outcome

    @property
    def last_completed_trial_toucher_id(self) -> str | None:
        """Who last touched the ball in the trial that just ended -- derived
        from last_completed_trial_match's own ball (Ball.
        last_touched_by_player_id, the single authoritative write-path for
        this fact -- see that field's own docstring; Match._set_possession()
        keeps it current every tick of live play, so nothing here needs its
        own separate tracking/snapshot logic beyond reusing
        last_completed_trial_match, which already has to exist for other
        reasons). Valid to read any time after step() returns True for that
        trial, up until the NEXT trial itself ends -- same lifetime as
        last_completed_trial_match itself. None before any trial has
        completed."""
        return (
            self._last_completed_trial_match.ball.last_touched_by_player_id
            if self._last_completed_trial_match is not None else None
        )

    @property
    def last_completed_trial_match(self) -> Match | None:
        """The Match exactly as it stood on the trial's TRUE final tick --
        valid to read any time after step() returns True for that trial, up
        until the NEXT trial itself ends. See _last_completed_trial_match's
        own field comment for why this is the only way to observe that
        state at all (self.match/self._match is already the NEXT trial's
        fresh state by the time step() returns True)."""
        return self._last_completed_trial_match

    @property
    def match(self) -> Match:
        return self._match

    @property
    def trial_count(self) -> int:
        return self._trial_count

    @property
    def complete(self) -> bool:
        return self.max_trials > 0 and self._trial_count >= self.max_trials

    def step(self) -> bool:
        """Advance the current trial by one physics tick.

        Returns ``True`` the tick a trial ends (after the linger period), at
        which point the loop has already rebuilt a fresh match for the next
        trial (or set ``complete`` if max_trials reached).
        """
        if self._pending_outcome is not None:
            if self.definition.on_tick is not None:
                self.definition.on_tick(self._match, self._trial_tick)
            self._match.step()
            self._trial_tick += 1
            self._linger_remaining_s -= self._match.dt_s
            if self._linger_remaining_s <= 0.0:
                self.outcomes[self._pending_outcome] = self.outcomes.get(self._pending_outcome, 0) + 1
                self._trial_count += 1
                self._last_completed_trial_match = self._match
                if not self.complete:
                    self._start_trial()
                return True
            return False

        if self.definition.on_tick is not None:
            self.definition.on_tick(self._match, self._trial_tick)
        self._match.step()
        self._trial_tick += 1
        # Call on_tick again after the step so that controllers can reissue
        # orders in the same tick they were cleared by match.step(). Without
        # this, a MoveOrder completing inside match.step() leaves
        # current_order=None for the rest of this tick, causing _trial_outcome
        # to fire the "all orders None + ball still" early-termination check
        # before the controller has had a chance to issue the next waypoint.
        if self.definition.on_tick is not None:
            self.definition.on_tick(self._match, self._trial_tick)

        if not self._ball_released and self._initial_carrier_id is not None:
            if self._match.ball.possessed_by != self._initial_carrier_id:
                self._ball_released = True

        outcome, linger = self._trial_outcome()
        if (outcome is not None
                and self.terminal_outcomes is not None
                and outcome not in self.terminal_outcomes):
            outcome = None
        if outcome is not None:
            outcome = self._apply_remap(outcome)
            if linger > 0.0:
                self._pending_outcome = outcome
                self._linger_remaining_s = linger
                return False
            self.outcomes[outcome] = self.outcomes.get(outcome, 0) + 1
            self._trial_count += 1
            self._last_completed_trial_match = self._match
            if not self.complete:
                self._start_trial()
            return True
        return False

    def _trial_outcome(self) -> tuple[str | None, float]:
        """Returns (outcome_key, linger_seconds) if the trial is over, else (None, 0).

        Out-of-bounds events linger for half the normal duration so the UI
        can briefly show the ball leaving the pitch without a full pause.
        All other outcomes use the full ``linger_s``. Detection itself lives
        in ``detect_trial_outcome`` (shared with ``ScenarioEnv``) so the UI
        and training never disagree on what ended a trial.
        """
        from footballcoach.ai.env.outcome import detect_trial_outcome
        outcome, half_linger = detect_trial_outcome(
            self._match,
            initial_scoreboard=self._initial_scoreboard,
            initial_carrier_id=self._initial_carrier_id,
            ball_released=self._ball_released,
            box_possession_terminal=self.definition.box_possession_terminal,
            trial_tick=self._trial_tick,
            timeout_ticks=self.timeout_ticks,
            phase1_trainee_player_id=self.definition.phase1_trainee_player_id,
        )
        if outcome is None:
            return None, 0.0
        return outcome, self.linger_s * 0.5 if half_linger else self.linger_s
