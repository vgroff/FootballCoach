"""Builds Match instances for the UI's two non-freeplay modes:

- Training mode: one player + ball on a full pitch, no opponent, goals in
  either net are counted and the ball/player reset on a goal.
- Balance scenarios: small, illustrative re-creations of the statistical
  scenarios from tests/balance/, played out live/visually one trial at a
  time (rather than the thousands-of-trials-headless style used by pytest).

These are UI conveniences for *watching* the mechanics that the balance
tests already validate statistically - they are not a replacement for the
pytest balance suite.

Phase H additions:
- ``ScenarioParam``: frozen dataclass describing one adjustable UI parameter.
- ``ScenarioDefinition`` now carries a ``params`` list and an optional
  ``on_tick`` hook (called each physics tick before ``match.step()``).
- Removed from SCENARIOS: penalty, save, shoot (kept as private helpers).
- Parameterized: save_close, pass, tackle, sprint.
- New: 2v2, 1v2.
- ``ScenarioLoop``: added ``dispossessed`` outcome, ``linger_s`` cooldown.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Callable

from footballcoach.config import load_physics_config
from footballcoach.engine.ball_physics import BallPhysicsParams
from footballcoach.engine.match import Match
from footballcoach.engine.movement import MovementParams, effective_top_speed
from footballcoach.entities import Ball, Pitch, PlayerAttributes, Team
from footballcoach.entities.player import Player, PlayerState
from footballcoach.generation import generate_attributes
from footballcoach.mathutils import Vector3
from footballcoach.orders import (
    ChaseTackleOrder,
    GetPossessionOrder,
    KickOrder,
    MoveOrder,
    PassOrder,
    SaveOrder,
    ShootOrder,
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


def make_training_match(rng_reduction: float = 0.3, tier: str = "premier_league") -> Match:
    """One player + ball, full pitch, both goals live, no opponent."""
    pitch = Pitch.standard()
    attrs = generate_attributes(tier=tier)
    player = Player.create("trainee", Team.LEFT, attrs, position=Vector3(0, 0, 0))
    ball = Ball.at_rest(Vector3(3, 0, 0))
    ui_cfg = load_physics_config().get("ui", {})
    return Match(
        pitch=pitch, players=[player], ball=ball,
        rng_reduction=rng_reduction, rng=random.Random(),
        goal_linger_s=ui_cfg.get("goal_linger_s", 3.0),
    )


@dataclass
class ScenarioDefinition:
    key: str
    label: str
    description: str
    build: Callable[..., Match]   # (rng_reduction, **kwargs) -> Match
    params: list[ScenarioParam] = field(default_factory=list)
    on_tick: Callable[[Match, int], None] | None = None


def build_penalty_scenario(rng_reduction: float = 0.3) -> Match:
    """Recreates tests/balance/test_penalty_balance.py's penalty setup: one
    kicker on the penalty spot, aiming at a bottom corner, no goalkeeper."""
    pitch = Pitch.standard()
    penalty_spot = pitch.penalty_spot(left=False)
    attrs = generate_attributes(tier="premier_league")
    kicker = Player.create("kicker", Team.LEFT, attrs, position=penalty_spot)

    ball = Ball.at_rest(penalty_spot)
    ball.possessed_by = kicker.player_id

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
    ball.possessed_by = attacker.player_id

    ui_cfg = load_physics_config().get("ui", {})
    match = Match(
        pitch=pitch, players=[defender, attacker], ball=ball,
        rng_reduction=rng_reduction, rng=rng,
        goal_linger_s=ui_cfg.get("goal_linger_s", 3.0),
    )
    attacker.current_order = MoveOrder(target_position=far_point, sprint=False)
    defender.current_order = ChaseTackleOrder(target_player_id=attacker.player_id)
    return match


class SprintController:
    """Issues sequential MoveOrder waypoints each time the previous completes."""

    def __init__(self, player_id: str, waypoints: list[Vector3]) -> None:
        self.player_id = player_id
        self.waypoints = waypoints
        self._next_idx = 0

    def __call__(self, match: Match, trial_tick: int) -> None:
        try:
            player = match.player_by_id(self.player_id)
        except KeyError:
            return
        if self._next_idx >= len(self.waypoints):
            return
        if player.current_order is None:
            player.current_order = MoveOrder(
                target_position=self.waypoints[self._next_idx], sprint=True
            )
            self._next_idx += 1


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
    ui_cfg = load_physics_config().get("ui", {})
    match = Match(
        pitch=pitch, players=[player], ball=ball,
        rng_reduction=rng_reduction, rng=rng,
        goal_linger_s=ui_cfg.get("goal_linger_s", 3.0),
    )
    controller = SprintController(player.player_id, waypoints)
    player.current_order = MoveOrder(target_position=waypoints[0], sprint=True)
    controller._next_idx = 1
    match._sprint_controller = controller  # type: ignore[attr-defined]
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

    aim_half_sign = -1.0 if gk_start_y >= 0 else 1.0
    aim_y = aim_half_sign * rng.uniform(half_goal_w * 0.3, half_goal_w - 0.3)
    aim_z = rng.uniform(0.2, 1.8)
    aim_point = pitch.left_goal_centre + Vector3(0, aim_y, aim_z)

    aim_dir = aim_point - shooter_pos
    aim_xy_len = math.hypot(aim_dir.x, aim_dir.y)
    mvmt = MovementParams.from_config()
    run_speed = effective_top_speed(
        mvmt, shooter.attributes.top_speed, shooter.stamina,
        has_ball=True, ball_control_attr=shooter.attributes.ball_control,
    )
    shooter.velocity = Vector3(aim_dir.x / aim_xy_len * run_speed,
                               aim_dir.y / aim_xy_len * run_speed, 0.0)

    ball = Ball.at_rest(shooter_pos)
    ball.possessed_by = shooter.player_id

    ui_cfg = load_physics_config().get("ui", {})
    match = Match(
        pitch=pitch, players=[gk, shooter], ball=ball,
        rng_reduction=rng_reduction, rng=rng,
        goal_linger_s=ui_cfg.get("goal_linger_s", 3.0),
    )
    gk.current_order = SaveOrder()
    shooter.current_order = KickOrder(aim_point=aim_point, power_fraction=shooter_power,
                                      spin=Vector3.zero(), compensate_for_run=False)
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
    passer.velocity = Vector3(top_speed * 0.5, 0.0, 0.0)

    ball = Ball.at_rest(passer_pos)
    ball.possessed_by = passer.player_id

    ui_cfg = load_physics_config().get("ui", {})
    match = Match(
        pitch=pitch, players=[passer, receiver], ball=ball,
        rng_reduction=rng_reduction, rng=rng,
        goal_linger_s=ui_cfg.get("goal_linger_s", 3.0),
    )
    passer.current_order = PassOrder(target_position=receiver_pos)
    return match


# ---------------------------------------------------------------------------
# Reusable on-tick AI primitives
# ---------------------------------------------------------------------------

class BallCarrierAttackerAI:
    """Reusable on-tick AI for a player who holds the ball and must shoot.

    Each tick, if the player has the ball:
    - If their current order is a ``MoveOrder`` and the distance to the
      move target is *increasing* (repulsion / obstruction is pushing them
      away), immediately switches to a ``ShootOrder`` at the configured
      aim point.
    - If their current order is ``None`` (e.g. a prior ``MoveOrder``
      completed), issues a ``ShootOrder``.
    - No-ops when the player does not have the ball.

    Compose with ``StagedGoalkeeperAI`` and others via ``CompositeAI``
    when building scenario on_tick hooks.
    """

    def __init__(self, player_id: str, aim_point: Vector3, power_fraction: float = 0.9) -> None:
        self.player_id = player_id
        self.aim_point = aim_point
        self.power_fraction = power_fraction
        self._prev_dist_to_target: float | None = None

    def __call__(self, match: Match, trial_tick: int) -> None:
        try:
            player = match.player_by_id(self.player_id)
        except KeyError:
            return
        if match.ball.possessed_by != self.player_id:
            self._prev_dist_to_target = None
            return
        order = player.current_order
        if isinstance(order, MoveOrder):
            dist = player.position.xy().distance_to(order.target_position.xy())
            prev = self._prev_dist_to_target
            self._prev_dist_to_target = dist
            if prev is not None and dist > prev:
                # Repulsion/obstruction is pushing us away — shoot now.
                player.current_order = ShootOrder(
                    aim_point=self.aim_point, power_fraction=self.power_fraction
                )
        elif order is None:
            self._prev_dist_to_target = None
            player.current_order = ShootOrder(
                aim_point=self.aim_point, power_fraction=self.power_fraction
            )


class StagedGoalkeeperAI:
    """Reusable on-tick AI for a goalkeeper that starts with a positioning
    ``MoveOrder`` and transitions to ``SaveOrder`` once in place.

    Assumes the GK's initial order is a ``MoveOrder`` (e.g. walk to goal
    centre with ``max_speed_on_arrival_mps=0.0``).  As soon as that order
    completes (``current_order`` becomes ``None``) and the GK does not have
    the ball, this AI issues a ``SaveOrder`` so the engine handles shot
    prediction from that point on.

    Works correctly even when the GK starts exactly at the goal centre
    (the MoveOrder completes in one or two ticks and the transition is
    immediate).
    """

    def __init__(self, gk_id: str) -> None:
        self.gk_id = gk_id

    def __call__(self, match: Match, trial_tick: int) -> None:
        try:
            gk = match.player_by_id(self.gk_id)
        except KeyError:
            return
        if gk.current_order is None and match.ball.possessed_by != self.gk_id:
            gk.current_order = SaveOrder()


class CompositeAI:
    """Composes multiple on-tick AI callables into a single callable.

    Each controller is called in insertion order every tick.  Useful for
    combining ``BallCarrierAttackerAI``, ``StagedGoalkeeperAI``, and any
    custom per-scenario logic into a single on_tick hook.
    """

    def __init__(self, *controllers: Callable[[Match, int], None]) -> None:
        self._controllers = controllers

    def __call__(self, match: Match, trial_tick: int) -> None:
        for ctrl in self._controllers:
            ctrl(match, trial_tick)


# ---------------------------------------------------------------------------
# 2v2 scenario
# ---------------------------------------------------------------------------

class TwoVTwoController:
    """Drives the 2v2 scenario via the on_tick hook.

    Handles the pass A→B hand-off: once B receives the ball, either shoots
    immediately or runs 30 % toward the goal first.  Ongoing "if B has the
    ball and no order, shoot" logic is delegated to ``BallCarrierAttackerAI``,
    which also handles the repulsion-pushback-→-shoot shortcut.
    """

    def __init__(self, attacker_a_id: str, attacker_b_id: str, goal_aim_point: Vector3,
                 shoot_immediately_probability: float = 0.5, rng: random.Random | None = None) -> None:
        self.attacker_a_id = attacker_a_id
        self.attacker_b_id = attacker_b_id
        self.goal_aim_point = goal_aim_point
        self._rng = rng or random.Random()
        self._b_received = False
        self._shoot_immediately = self._rng.random() < shoot_immediately_probability
        self._b_ai = BallCarrierAttackerAI(attacker_b_id, goal_aim_point, power_fraction=0.85)

    def __call__(self, match: Match, trial_tick: int) -> None:
        try:
            b = match.player_by_id(self.attacker_b_id)
        except KeyError:
            return
        if not self._b_received:
            if match.ball.possessed_by == self.attacker_b_id:
                self._b_received = True
                if self._shoot_immediately:
                    b.current_order = ShootOrder(aim_point=self.goal_aim_point, power_fraction=0.85)
                else:
                    run_target = Vector3(
                        b.position.x + (self.goal_aim_point.x - b.position.x) * 0.3,
                        b.position.y, 0.0,
                    )
                    b.current_order = MoveOrder(target_position=run_target, sprint=True)
        # Delegate ongoing attacker logic to the shared AI primitive.
        self._b_ai(match, trial_tick)


def build_2v2_scenario(
    rng_reduction: float = 0.3,
    *,
    attacker_skill_min: float = 0.7,
    attacker_skill_max: float = 0.85,
    defender_skill_min: float = 0.55,
    defender_skill_max: float = 0.7,
    gk_skill_min: float = 0.55,
    gk_skill_max: float = 0.7,
    shoot_immediately_probability: float = 0.5,
) -> Match:
    """2v2: attacker A (ball) passes to attacker B; defender + GK defend."""
    rng = random.Random()
    pitch = Pitch.standard()

    def make_player(pid: str, team: Team, skill: float, pos: Vector3, gk: bool = False) -> Player:
        attrs = PlayerAttributes(skill, skill, skill, skill, skill, skill, skill, skill)
        return Player.create(pid, team, attrs, position=pos, is_goalkeeper=gk)

    half_goal_w = pitch.goal_width_m / 2.0
    a_x = pitch.half_length - pitch.box_length_m - 2.0
    attacker_a = make_player("atk_a", Team.RIGHT,
                             rng.uniform(attacker_skill_min, attacker_skill_max),
                             Vector3(a_x, -half_goal_w + 0.5, 0))
    attacker_b = make_player("atk_b", Team.RIGHT,
                             rng.uniform(attacker_skill_min, attacker_skill_max),
                             Vector3(a_x - 8.0, half_goal_w - 0.5, 0))
    defender = make_player("defender", Team.LEFT,
                           rng.uniform(defender_skill_min, defender_skill_max),
                           Vector3(pitch.half_length - 8.0, 0.0, 0))
    gk_sk = rng.uniform(gk_skill_min, gk_skill_max)
    gk = Player.create("keeper", Team.LEFT,
                       PlayerAttributes(gk_sk, gk_sk, gk_sk, 0.5, 0.5, 0.5, gk_sk, 0.5),
                       position=pitch.right_goal_centre, is_goalkeeper=True)

    ball = Ball.at_rest(attacker_a.position)
    ball.possessed_by = attacker_a.player_id

    ui_cfg = load_physics_config().get("ui", {})
    match = Match(
        pitch=pitch, players=[attacker_a, attacker_b, defender, gk],
        ball=ball, rng_reduction=rng_reduction, rng=rng,
        goal_linger_s=ui_cfg.get("goal_linger_s", 3.0),
    )

    attacker_b.current_order = MoveOrder(
        target_position=Vector3(pitch.half_length - 5.0, attacker_b.position.y, 0), sprint=True
    )
    attacker_a.current_order = PassOrder(target_position=attacker_b.position)
    defender.current_order = GetPossessionOrder()
    gk.current_order = SaveOrder()

    aim_z = rng.uniform(0.2, 1.5)
    aim_point = pitch.right_goal_centre + Vector3(
        0, rng.uniform(-half_goal_w * 0.6, half_goal_w * 0.6), aim_z
    )
    controller = TwoVTwoController(
        attacker_a.player_id, attacker_b.player_id,
        goal_aim_point=aim_point,
        shoot_immediately_probability=shoot_immediately_probability,
        rng=rng,
    )
    match._2v2_controller = controller  # type: ignore[attr-defined]
    return match


# ---------------------------------------------------------------------------
# 1v2 scenario
# ---------------------------------------------------------------------------

class OneVTwoController:
    """Drives the 1v2 scenario.

    Composes two reusable AI primitives:

    - ``BallCarrierAttackerAI``: attacker carries ball toward goal on a
      ``MoveOrder``; switches to ``ShootOrder`` when the move completes or
      repulsion starts pushing them away from the target.
    - ``StagedGoalkeeperAI``: GK starts on a ``MoveOrder`` to the goal
      centre; transitions to ``SaveOrder`` once in position.

    The defender uses a ``GetPossessionOrder`` set at build time and needs
    no on-tick logic.
    """

    def __init__(self, attacker_id: str, gk_id: str, shoot_at: Vector3) -> None:
        self._ai = CompositeAI(
            BallCarrierAttackerAI(attacker_id, shoot_at, power_fraction=0.9),
            StagedGoalkeeperAI(gk_id),
        )

    def __call__(self, match: Match, trial_tick: int) -> None:
        self._ai(match, trial_tick)


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
    move_fraction_min: float = 0.10,
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
    ball.possessed_by = attacker.player_id

    move_frac = rng.uniform(move_fraction_min, move_fraction_max)
    move_target_raw = attacker_start + (goal_centre - attacker_start) * move_frac
    move_target = Vector3(
        max(-pitch.half_length + 1, min(pitch.half_length - 1, move_target_raw.x)),
        max(-pitch.half_width + 1, min(pitch.half_width - 1, move_target_raw.y)),
        0.0,
    )

    half_goal_w = pitch.goal_width_m / 2.0
    aim_point = goal_centre + Vector3(
        0, rng.uniform(-half_goal_w * 0.7, half_goal_w * 0.7), rng.uniform(0.2, 1.8)
    )

    ui_cfg = load_physics_config().get("ui", {})
    match = Match(
        pitch=pitch, players=[attacker, defender, gk],
        ball=ball, rng_reduction=rng_reduction, rng=rng,
        goal_linger_s=ui_cfg.get("goal_linger_s", 3.0),
    )

    attacker.current_order = MoveOrder(target_position=move_target, sprint=True)
    defender.current_order = GetPossessionOrder()
    # GK starts by walking to goal centre (standstill), then on_tick upgrades to SaveOrder.
    gk.current_order = MoveOrder(target_position=goal_centre, sprint=False, max_speed_on_arrival_mps=0.0)

    controller = OneVTwoController(attacker.player_id, gk.player_id, aim_point)
    match._1v2_controller = controller  # type: ignore[attr-defined]
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
    ctrl = getattr(match, "_sprint_controller", None)
    if ctrl is not None:
        ctrl(match, trial_tick)


def _2v2_on_tick(match: Match, trial_tick: int) -> None:
    ctrl = getattr(match, "_2v2_controller", None)
    if ctrl is not None:
        ctrl(match, trial_tick)


def _1v2_on_tick(match: Match, trial_tick: int) -> None:
    ctrl = getattr(match, "_1v2_controller", None)
    if ctrl is not None:
        ctrl(match, trial_tick)


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
    ball.possessed_by = attacker.player_id

    ui_cfg = load_physics_config().get("ui", {})
    match = Match(
        pitch=pitch, players=[attacker, obstacle], ball=ball,
        rng_reduction=rng_reduction, rng=rng,
        goal_linger_s=ui_cfg.get("goal_linger_s", 3.0),
    )
    attacker.current_order = MoveOrder(target_position=target, sprint=True)
    return match


# ---------------------------------------------------------------------------
# Phase 1 curriculum: 1v1 get-possession / move-toward-goal
# ---------------------------------------------------------------------------

def _1v1_on_tick(match: Match, trial_tick: int) -> None:
    """UI fallback for the 1v1 training scenario: give the trainee a
    ``GetPossessionOrder`` whenever it has no current order.

    In training the AI overrides the trainee's order each decision interval,
    so this only fires between AI steps (or when no AI is driving the env).
    In the UI scenario picker this makes the trainee chase the ball so the
    scenario is watchable.
    """
    try:
        trainee = match.player_by_id("trainee")
    except KeyError:
        return
    if trainee.current_order is None:
        trainee.current_order = GetPossessionOrder()


def build_1v1_scenario(
    rng_reduction: float = 0.3,
    *,
    trainee_tier: str = "generic",
    opponent_tier: str = "generic",
    ball_max_speed_mps: float = 8.0,
    restitution_sigma: float = 0.08,
    ball_max_dist_from_trainee_m: float = 35.0,
) -> Match:
    """Phase 1 curriculum: 1v1 get-possession/move-toward-goal.

    Both players and the ball are placed randomly across the full pitch with
    randomised attributes, stamina, headings, and (for the ball) velocity,
    spin, and restitution coefficient.

    Trainee (Team.LEFT, player_id='trainee') has no initial order so the AI
    can drive it.  Opponent (Team.RIGHT, player_id='opponent') is immobile
    (no order) - the first sub-phase of the phase-1 curriculum.

    Ball velocity is resampled (up to 20 times) until a linear extrapolation
    at the initial speed for 3 seconds stays in bounds (ignoring friction -
    conservative).  If all attempts fail the ball starts at rest.

    Restitution: sampled from Gaussian(base_restitution, sigma=0.08),
    clamped to [0.2, 0.95].

    See ai_design_doc.md sections 4 and 9.2 for the full design rationale.
    """
    rng = random.Random()
    pitch = Pitch.standard()

    def _rand_pos() -> Vector3:
        return Vector3(
            rng.uniform(-pitch.half_length + 1.5, pitch.half_length - 1.5),
            rng.uniform(-pitch.half_width + 1.5, pitch.half_width - 1.5),
            0.0,
        )

    # --- Trainee (Team.LEFT, attacks +x) ---
    trainee_attrs = generate_attributes(tier=trainee_tier, rng=rng)
    trainee = Player.create("trainee", Team.LEFT, trainee_attrs, position=_rand_pos())
    trainee.stamina = rng.uniform(0.3, 1.0)
    trainee.heading_rad = rng.uniform(-math.pi, math.pi)

    # --- Opponent (Team.RIGHT, attacks -x) --- immobile, no order ---
    opponent_attrs = generate_attributes(tier=opponent_tier, rng=rng)
    opponent = Player.create("opponent", Team.RIGHT, opponent_attrs, position=_rand_pos())
    opponent.stamina = rng.uniform(0.3, 1.0)
    opponent.heading_rad = rng.uniform(-math.pi, math.pi)

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
        # Conservative (no-friction) 3-second extrapolation check
        if (abs(ball_pos.x + vx * 3.0) < pitch.half_length
                and abs(ball_pos.y + vy * 3.0) < pitch.half_width):
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

    ui_cfg = load_physics_config().get("ui", {})
    return Match(
        pitch=pitch,
        players=[trainee, opponent],
        ball=ball,
        rng_reduction=rng_reduction,
        rng=rng,
        ball_physics_params=ball_params,
        goal_linger_s=ui_cfg.get("goal_linger_s", 3.0),
    )


# ---------------------------------------------------------------------------
# SCENARIOS list (Phase H: trimmed to 6, all parameterized)
# ---------------------------------------------------------------------------

SCENARIOS: list[ScenarioDefinition] = [
    ScenarioDefinition(
        key="save_close",
        label="Shot vs keeper (close range, mixed outcome)",
        description="Mid-tier striker shoots from varied distance; expect saves and goals.",
        build=build_close_range_save_scenario,
        params=[
            ScenarioParam("distance_min_m", "Min shot distance (m)", 3.0, 20.0, 1.0, 8.0),
            ScenarioParam("distance_max_m", "Max shot distance (m)", 5.0, 30.0, 1.0, 16.0),
            ScenarioParam("shooter_y_offset_m", "Shooter lateral offset (m)", 0.0, 10.0, 0.5, 5.0),
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
            ScenarioParam("max_distance_m", "Max pass distance (m)", 5.0, 60.0, 5.0, 30.0),
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
            ScenarioParam("separation_min_m", "Min separation (m)", 0.5, 20.0, 0.5, 1.0),
            ScenarioParam("separation_max_m", "Max separation (m)", 0.5, 30.0, 1.0, 10.0),
            ScenarioParam("tackler_tackling_min", "Tackler tackling min", 0.0, 1.0, 0.05, 0.8),
            ScenarioParam("tackler_tackling_max", "Tackler tackling max", 0.0, 1.0, 0.05, 0.8),
            ScenarioParam("dribbler_dribbling_min", "Dribbler dribbling min", 0.0, 1.0, 0.05, 0.6),
            ScenarioParam("dribbler_dribbling_max", "Dribbler dribbling max", 0.0, 1.0, 0.05, 0.6),
        ],
    ),
    ScenarioDefinition(
        key="sprint",
        label="Sprint: random 5-waypoint course",
        description="A runner follows a random 5-leg course across the pitch.",
        build=build_sprint_scenario,
        on_tick=_sprint_on_tick,
        params=[
            ScenarioParam("leg_min_m", "Min leg length (m)", 2.0, 30.0, 1.0, 5.0),
            ScenarioParam("leg_max_m", "Max leg length (m)", 5.0, 50.0, 2.0, 25.0),
            ScenarioParam("runner_skill_min", "Runner skill min", 0.0, 1.0, 0.05, 0.7),
            ScenarioParam("runner_skill_max", "Runner skill max", 0.0, 1.0, 0.05, 0.8),
        ],
    ),
    ScenarioDefinition(
        key="2v2",
        label="2v2: pass and shoot vs defender+GK",
        description="Two attackers combine with a pass before shooting; one defender + GK.",
        build=build_2v2_scenario,
        on_tick=_2v2_on_tick,
        params=[
            ScenarioParam("attacker_skill_min", "Attacker skill min", 0.3, 1.0, 0.05, 0.7),
            ScenarioParam("attacker_skill_max", "Attacker skill max", 0.3, 1.0, 0.05, 0.85),
            ScenarioParam("defender_skill_min", "Defender skill min", 0.3, 1.0, 0.05, 0.55),
            ScenarioParam("defender_skill_max", "Defender skill max", 0.3, 1.0, 0.05, 0.7),
            ScenarioParam("gk_skill_min", "GK skill min", 0.3, 1.0, 0.05, 0.55),
            ScenarioParam("gk_skill_max", "GK skill max", 0.3, 1.0, 0.05, 0.7),
            ScenarioParam("shoot_immediately_probability", "Shoot immediately prob", 0.0, 1.0, 0.1, 0.5),
        ],
    ),
    ScenarioDefinition(
        key="1v2",
        label="1v2: elite attacker vs. average defender+GK",
        description="Elite attacker (skill=0.9) runs at goal; average defender+GK defend.",
        build=build_1v2_scenario,
        on_tick=_1v2_on_tick,
        params=[
            ScenarioParam("attacker_skill", "Attacker skill", 0.3, 1.0, 0.05, 0.9),
            ScenarioParam("defender_skill", "Defender skill", 0.3, 1.0, 0.05, 0.55),
            ScenarioParam("gk_skill", "GK skill", 0.3, 1.0, 0.05, 0.55),
            ScenarioParam("attacker_start_min_m", "Attacker start min dist (m)", 5.0, 40.0, 1.0, 18.0),
            ScenarioParam("attacker_start_max_m", "Attacker start max dist (m)", 10.0, 50.0, 1.0, 32.0),
            ScenarioParam("move_fraction_min", "Move fraction min", 0.0, 1.0, 0.05, 0.10),
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
        key="1v1_phase1",
        label="1v1: Phase 1 get possession (AI training)",
        description=(
            "Both players randomly placed, random ball, random attributes. "
            "Phase 1 curriculum scenario. Trainee chases ball; opponent immobile."
        ),
        build=build_1v1_scenario,
        on_tick=_1v1_on_tick,
        params=[
            ScenarioParam("ball_max_speed_mps", "Ball max speed (m/s)", 0.0, 15.0, 0.5, 8.0),
            ScenarioParam("restitution_sigma", "Restitution randomness (sigma)", 0.0, 0.3, 0.01, 0.08),
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
    3. Initial ball carrier released ball and ball resolved (save, miss, dispossessed).
    4. All non-persistent orders resolved and ball settled.
    5. Timeout failsafe (default 500 ticks ≈ 16.7 s at 30 Hz).
    """

    definition: ScenarioDefinition
    max_trials: int = 0
    timeout_ticks: int = 500
    rng_reduction: float = 0.3
    kwargs: dict = field(default_factory=dict)
    linger_s: float = field(
        default_factory=lambda: load_physics_config().get("ui", {}).get("scenario_linger_s", 3.0)
    )

    _trial_count: int = field(default=0, init=False, repr=False)
    _trial_tick: int = field(default=0, init=False, repr=False)
    _match: Match = field(init=False, repr=False)
    _initial_carrier_id: str | None = field(default=None, init=False, repr=False)
    _initial_scoreboard: tuple[int, int] = field(default=(0, 0), init=False, repr=False)
    _ball_released: bool = field(default=False, init=False, repr=False)
    outcomes: dict[str, int] = field(
        default_factory=lambda: {"goal": 0, "saved": 0, "miss": 0, "dispossessed": 0, "other": 0},
        init=False, repr=False,
    )
    _pending_outcome: str | None = field(default=None, init=False, repr=False)
    _linger_remaining_s: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
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
                self.outcomes[self._pending_outcome] += 1
                self._trial_count += 1
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
        if outcome is not None:
            if linger > 0.0:
                self._pending_outcome = outcome
                self._linger_remaining_s = linger
                return False
            self.outcomes[outcome] += 1
            self._trial_count += 1
            if not self.complete:
                self._start_trial()
            return True
        return False

    def _trial_outcome(self) -> tuple[str | None, float]:
        """Returns (outcome_key, linger_seconds) if the trial is over, else (None, 0).

        Out-of-bounds events linger for half the normal duration so the UI
        can briefly show the ball leaving the pitch without a full pause.
        All other outcomes (goal, saved, dispossessed, other) use the full
        ``linger_s``.
        """
        pitch = self._match.pitch
        ball = self._match.ball
        scoreboard = self._match.scoreboard

        if abs(ball.position.x) > pitch.half_length + 1.0:
            return "miss", self.linger_s * 0.5
        if abs(ball.position.y) > pitch.half_width + 0.5:
            return "miss", self.linger_s * 0.5

        if (scoreboard.left_goals, scoreboard.right_goals) != self._initial_scoreboard:
            return "goal", self.linger_s

        if self._ball_released:
            if ball.possessed_by is not None and ball.possessed_by != self._initial_carrier_id:
                try:
                    repossessor = self._match.player_by_id(ball.possessed_by)
                    initial_carrier = (
                        self._match.player_by_id(self._initial_carrier_id)
                        if self._initial_carrier_id else None
                    )
                    if initial_carrier is not None and repossessor.team != initial_carrier.team:
                        if repossessor.is_goalkeeper:
                            return "saved", self.linger_s
                        return "dispossessed", self.linger_s
                except KeyError:
                    pass
                return "saved", self.linger_s

            any_controlling = any(
                p.state == PlayerState.CONTROLLING_BALL for p in self._match.players
            )
            if ball.possessed_by is None and ball.velocity.length() < 0.1 and not any_controlling:
                return "miss", self.linger_s * 0.5

        if self._trial_tick >= 30 and all(p.current_order is None for p in self._match.players):
            if ball.velocity.length() < 0.1:
                return "other", self.linger_s

        if self._trial_tick >= self.timeout_ticks:
            return "other", self.linger_s

        return None, 0.0
