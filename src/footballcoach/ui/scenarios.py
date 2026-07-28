"""Builds Match instances for the UI's two non-freeplay modes:

- Training mode: one player + ball on a full pitch, no opponent, goals in
  either net are counted and the ball/player reset on a goal.
- Balance scenarios: small, illustrative re-creations of the statistical
  scenarios from tests/balance/, played out live/visually one trial at a
  time (rather than the thousands-of-trials-headless style used by pytest).

These are UI conveniences for *watching* the mechanics that the balance
tests already validate statistically - they are not a replacement for the
pytest balance suite.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from footballcoach import actions
from footballcoach.engine.match import Match
from footballcoach.engine.movement import MovementParams, effective_top_speed
from footballcoach.entities import Ball, Pitch, PlayerAttributes, Team
from footballcoach.entities.player import Player, PlayerState
from footballcoach.generation import generate_attributes
from footballcoach.mathutils import Vector3
from footballcoach.orders import KickOrder, MoveOrder, PassOrder, SaveOrder, TackleOrder


def make_training_match(rng_reduction: float = 0.3, tier: str = "premier_league") -> Match:
    """One player + ball, full pitch, both goals live, no opponent."""
    pitch = Pitch.standard()
    attrs = generate_attributes(tier=tier)
    player = Player.create("trainee", Team.LEFT, attrs, position=Vector3(0, 0, 0))
    ball = Ball.at_rest(Vector3(3, 0, 0))
    return Match(pitch=pitch, players=[player], ball=ball, rng_reduction=rng_reduction, rng=random.Random())


@dataclass
class ScenarioDefinition:
    key: str
    label: str
    description: str
    build: Callable[[float], Match]


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


def build_tackle_scenario(rng_reduction: float = 0.3) -> Match:
    """Recreates tests/balance/test_tackling_balance.py's headline matchup:
    tackling=0.8 defender vs dribbling=0.6 attacker in possession, touching."""
    pitch = Pitch.standard()

    defender_attrs = PlayerAttributes(
        top_speed=0.6, acceleration=0.6, stamina=0.6, kick_precision=0.6,
        kick_power=0.6, dribbling=0.6, ball_control=0.6, tackling=0.8,
    )
    attacker_attrs = PlayerAttributes(
        top_speed=0.6, acceleration=0.6, stamina=0.6, kick_precision=0.6,
        kick_power=0.6, dribbling=0.6, ball_control=0.6, tackling=0.6,
    )
    defender = Player.create("defender", Team.LEFT, defender_attrs, position=Vector3(0, 0, 0))
    attacker = Player.create("attacker", Team.RIGHT, attacker_attrs, position=Vector3(0.5, 0, 0))

    ball = Ball.at_rest(Vector3(0.5, 0, 0))
    ball.possessed_by = attacker.player_id

    match = Match(pitch=pitch, players=[defender, attacker], ball=ball, rng_reduction=rng_reduction, rng=random.Random())
    defender.current_order = TackleOrder(target_player_id=attacker.player_id)
    return match


def build_sprint_scenario(rng_reduction: float = 0.3) -> Match:
    """Recreates tests/balance/test_sprint_balance.py: a top-attribute and a
    bottom-attribute player racing 100m from the same start line."""
    pitch = Pitch.standard()
    start_x = -pitch.half_length + 1

    top_attrs = PlayerAttributes(
        top_speed=1.0, acceleration=1.0, stamina=1.0, kick_precision=0.5,
        kick_power=0.5, dribbling=0.5, ball_control=0.5, tackling=0.5,
    )
    bottom_attrs = PlayerAttributes(
        top_speed=0.0, acceleration=0.0, stamina=0.5, kick_precision=0.5,
        kick_power=0.5, dribbling=0.5, ball_control=0.5, tackling=0.5,
    )
    top_player = Player.create("fast", Team.LEFT, top_attrs, position=Vector3(start_x, -3, 0))
    bottom_player = Player.create("slow", Team.RIGHT, bottom_attrs, position=Vector3(start_x, 3, 0))

    ball = Ball.at_rest(Vector3(start_x, 20, 0))  # kept well away so it's not auto-picked-up
    match = Match(
        pitch=pitch, players=[top_player, bottom_player], ball=ball,
        rng_reduction=rng_reduction, rng=random.Random(),
    )

    for player in (top_player, bottom_player):
        player.current_order = MoveOrder(target_position=Vector3(pitch.half_length - 1, player.position.y, 0), sprint=True)
    return match


def build_close_range_save_scenario(rng_reduction: float = 0.3) -> Match:
    """Shot vs keeper with randomised setup each trial: varied distances
    (12-20m), GK start position (anywhere across goal), shot placement, and
    attribute spreads.  Decoupled from the balance tests - this is purely a
    visual/UI scenario designed to produce a genuine mix of saves and goals
    across the 20-trial loop."""
    import math as _math
    pitch = Pitch.standard()
    half_goal_w = pitch.goal_width_m / 2.0
    rng = random.Random()  # fresh seed every call → different each trial

    # --- Goalkeeper ---
    gk_speed = rng.uniform(0.65, 0.85)
    gk_control = rng.uniform(0.65, 0.85)
    gk_attrs = PlayerAttributes(
        top_speed=gk_speed, acceleration=gk_speed, stamina=0.8,
        kick_precision=0.5, kick_power=0.5, dribbling=0.5,
        ball_control=gk_control, tackling=0.5,
    )
    # --- Shooter (position decided before GK so smart GK can read it) ---
    shot_dist = rng.uniform(8.0, 16.0)       # 8-16m from goal line
    shooter_y = rng.uniform(-5.0, 5.0)       # lateral spread

    # GK positioning: 50% random across goal, 50% "smart" - shifted toward the
    # shooter's y-side as a real keeper would (covering the near post / near
    # angle).  Since the aim always goes to the OPPOSITE side from gk_start_y,
    # smart positioning still forces a full cross-goal dive; it just looks more
    # realistic than a keeper stranded on the wrong side by pure chance.
    if rng.random() < 0.5:
        # Random placement anywhere across the goal.
        gk_start_y = rng.uniform(-half_goal_w + 0.3, half_goal_w - 0.3)
    else:
        # Smart: align with shooter's side at 30-60% of goal half-width so
        # the keeper covers the near post but the far side is still exposed.
        shift_fraction = rng.uniform(0.3, 0.6)
        gk_start_y = max(-half_goal_w + 0.3,
                         min(half_goal_w - 0.3,
                             _math.copysign(half_goal_w * shift_fraction, shooter_y)))

    gk = Player.create(
        "keeper", Team.LEFT, gk_attrs,
        position=pitch.left_goal_centre + Vector3(0, gk_start_y, 0),
        is_goalkeeper=True,
    )
    shooter_precision = rng.uniform(0.65, 0.85)
    shooter_speed = rng.uniform(0.65, 0.85)
    # Power scales with distance: closer shots need less pace, farther ones more.
    # Linearly interpolated: 8m→0.70, 16m→0.90.
    shooter_power = 0.70 + (shot_dist - 8.0) / (16.0 - 8.0) * 0.20
    shooter_power = min(0.92, max(0.68, shooter_power + rng.uniform(-0.04, 0.04)))
    shooter_attrs = PlayerAttributes(
        top_speed=shooter_speed, acceleration=shooter_speed, stamina=0.7,
        kick_precision=shooter_precision, kick_power=shooter_power,
        dribbling=0.5, ball_control=0.5, tackling=0.5,
    )
    shooter_pos = Vector3(-(pitch.half_length - shot_dist), shooter_y, 0)
    shooter = Player.create("striker", Team.RIGHT, shooter_attrs, position=shooter_pos)

    # Aim away from the goalkeeper: pick a point in the opposite half of the
    # goal from where the keeper is standing, so the shot forces a dive.
    # gk_start_y > 0 → keeper is on positive side → aim at negative side, and vice versa.
    aim_half_sign = -1.0 if gk_start_y >= 0 else 1.0
    # Aim anywhere from the post inward to just past the centre of the opposite half.
    aim_y = aim_half_sign * rng.uniform(half_goal_w * 0.3, half_goal_w - 0.3)
    aim_z = rng.uniform(0.2, 1.8)
    aim_point = pitch.left_goal_centre + Vector3(0, aim_y, aim_z)

    # Velocity aligned to aim direction for running power boost
    aim_dir = aim_point - shooter_pos
    aim_xy_len = _math.hypot(aim_dir.x, aim_dir.y)
    mvmt = MovementParams.from_config()
    run_speed = effective_top_speed(
        mvmt, shooter.attributes.top_speed, shooter.stamina,
        has_ball=True, ball_control_attr=shooter.attributes.ball_control,
    )
    shooter.velocity = Vector3(
        aim_dir.x / aim_xy_len * run_speed,
        aim_dir.y / aim_xy_len * run_speed,
        0.0,
    )

    ball = Ball.at_rest(shooter_pos)
    ball.possessed_by = shooter.player_id

    match = Match(pitch=pitch, players=[gk, shooter], ball=ball,
                  rng_reduction=rng_reduction, rng=rng)

    gk.current_order = SaveOrder()
    shooter.current_order = KickOrder(
        aim_point=aim_point, power_fraction=shooter_power, spin=Vector3.zero()
    )
    return match


def build_save_scenario(rng_reduction: float = 0.3) -> Match:
    """GK pinned to near post, precise shot from 22m aimed at the far corner
    — keeper must dive across the whole goal to reach it."""
    pitch = Pitch.standard()
    half_goal_w = pitch.goal_width_m / 2.0

    gk_attrs = PlayerAttributes(
        top_speed=1.0, acceleration=1.0, stamina=0.8, kick_precision=0.5,
        kick_power=0.5, dribbling=0.5, ball_control=0.9, tackling=0.5,
    )
    gk = Player.create(
        "keeper", Team.LEFT, gk_attrs,
        position=pitch.left_goal_centre + Vector3(0, -half_goal_w + 0.3, 0),
        is_goalkeeper=True,
    )

    shooter_attrs = PlayerAttributes(
        top_speed=0.7, acceleration=0.7, stamina=0.6, kick_precision=0.8,
        kick_power=0.8, dribbling=0.5, ball_control=0.5, tackling=0.5,
    )
    # 22m from the left goal line.
    shooter_pos = Vector3(-(pitch.half_length - 22.0), 0, 0)
    shooter = Player.create("striker", Team.RIGHT, shooter_attrs, position=shooter_pos)

    aim_point = pitch.left_goal_centre + Vector3(0, half_goal_w - 0.3, 0.3)
    # Velocity aligned to the aim direction for the running power boost.
    import math as _math
    aim_dir = aim_point - shooter_pos
    aim_xy_len = _math.hypot(aim_dir.x, aim_dir.y)
    mvmt = MovementParams.from_config()
    run_speed = effective_top_speed(
        mvmt, shooter.attributes.top_speed, shooter.stamina,
        has_ball=True, ball_control_attr=shooter.attributes.ball_control,
    )
    shooter.velocity = Vector3(
        aim_dir.x / aim_xy_len * run_speed,
        aim_dir.y / aim_xy_len * run_speed,
        0.0,
    )

    ball = Ball.at_rest(shooter_pos)
    ball.possessed_by = shooter.player_id

    match = Match(pitch=pitch, players=[gk, shooter], ball=ball,
                  rng_reduction=rng_reduction, rng=random.Random())

    gk.current_order = SaveOrder()
    shooter.current_order = KickOrder(aim_point=aim_point, power_fraction=0.95, spin=Vector3.zero())
    return match


def build_shoot_scenario(rng_reduction: float = 0.3) -> Match:
    """Recreates tests/balance/test_shoot_balance.py: a Premier-League-tier
    player shoots from inside the box at dead centre of goal, no goalkeeper."""
    pitch = Pitch.standard()
    # Midway between the penalty spot and the goal line, slightly off-centre.
    x = pitch.half_length - 14.0
    y = 5.0
    attrs = generate_attributes(tier="premier_league")
    kicker = Player.create("striker", Team.LEFT, attrs, position=Vector3(x, y, 0))

    ball = Ball.at_rest(kicker.position)
    ball.possessed_by = kicker.player_id

    match = Match(pitch=pitch, players=[kicker], ball=ball,
                  rng_reduction=rng_reduction, rng=random.Random())

    actions.shoot(kicker, pitch)
    return match


def build_pass_scenario(rng_reduction: float = 0.3) -> Match:
    """A 20m ground pass between two Premier-League-tier players at midfield."""
    pitch = Pitch.standard()
    passer_attrs = generate_attributes(tier="premier_league")
    receiver_attrs = generate_attributes(tier="premier_league")
    passer = Player.create("passer", Team.LEFT, passer_attrs, position=Vector3(-10, 0, 0))
    receiver = Player.create("receiver", Team.LEFT, receiver_attrs, position=Vector3(10, 0, 0))

    # Passer running at half speed toward receiver, matching calibration setup.
    mvmt = MovementParams.from_config()
    top_speed = effective_top_speed(
        mvmt, passer.attributes.top_speed, passer.stamina,
        has_ball=True, ball_control_attr=passer.attributes.ball_control,
    )
    passer.velocity = Vector3(top_speed * 0.5, 0.0, 0.0)

    ball = Ball.at_rest(passer.position)
    ball.possessed_by = passer.player_id

    match = Match(pitch=pitch, players=[passer, receiver], ball=ball,
                  rng_reduction=rng_reduction, rng=random.Random())

    passer.current_order = PassOrder(target_position=receiver.position)
    return match


SCENARIOS: list[ScenarioDefinition] = [
    ScenarioDefinition(
        key="penalty",
        label="Penalty (corner aim, no keeper)",
        description="A Premier-League-tier kicker takes a penalty aiming at the bottom corner.",
        build=build_penalty_scenario,
    ),
    ScenarioDefinition(
        key="save_close",
        label="Shot vs keeper (close range, mixed outcome)",
        description="Mid-tier striker shoots from 12m, centred GK - expect saves and goals.",
        build=build_close_range_save_scenario,
    ),
    ScenarioDefinition(
        key="save",
        label="Goalkeeper save (far-post dive, 22m)",
        description="Fast GK pinned to near post, precise shot aimed at the far corner.",
        build=build_save_scenario,
    ),
    ScenarioDefinition(
        key="shoot",
        label="Shoot from box (no keeper)",
        description="A Premier-League-tier player shoots from inside the box.",
        build=build_shoot_scenario,
    ),
    ScenarioDefinition(
        key="pass",
        label="20m ground pass",
        description="A 20m ground pass between two Premier-League-tier players.",
        build=build_pass_scenario,
    ),
    ScenarioDefinition(
        key="tackle",
        label="Tackle challenge (0.8 tackling vs 0.6 dribbling)",
        description="A tackling=0.8 defender challenges a dribbling=0.6 attacker in possession.",
        build=build_tackle_scenario,
    ),
    ScenarioDefinition(
        key="sprint",
        label="Sprint race (top vs bottom attributes)",
        description="A max-attribute player races a min-attribute player over 100m.",
        build=build_sprint_scenario,
    ),
]


@dataclass
class ScenarioLoop:
    """Runs a ScenarioDefinition repeatedly for a fixed number of trials,
    rebuilding a fresh Match after each one completes.  The UI drives it by
    calling ``step()`` once per frame; the caller inspects ``complete`` to
    know when all trials are done.

    Trial-end detection (in priority order):
    1. Ball crosses the touchline or goal line (out of bounds / goal).
    2. Scoreboard changed (goal scored into a net).
    3. The initial ball carrier has released the ball AND the ball has since
       been picked up by someone else, gone dead (speed < 0.1), or a goal
       was recorded.  This correctly handles GK saves: as soon as the GK
       controls the ball the trial ends, even though SaveOrder keeps running.
    4. All non-persistent orders resolved and ball has stopped (covers
       orderless scenarios like sprint, tackle with no SaveOrder).
    5. Timeout failsafe (default 500 ticks ≈ 16.7 s at 30 Hz).
    """

    definition: ScenarioDefinition
    max_trials: int = 0  # 0 = run indefinitely; any positive value = stop after that many
    timeout_ticks: int = 500
    rng_reduction: float = 0.3

    _trial_count: int = field(default=0, init=False, repr=False)
    _trial_tick: int = field(default=0, init=False, repr=False)
    _match: Match = field(init=False, repr=False)
    # Player ID that started with the ball; None if ball was already loose.
    _initial_carrier_id: str | None = field(default=None, init=False, repr=False)
    _initial_scoreboard: tuple[int, int] = field(default=(0, 0), init=False, repr=False)
    # Set to True once the initial carrier has released the ball.
    _ball_released: bool = field(default=False, init=False, repr=False)
    # Running tally of outcomes across all completed trials.
    outcomes: dict[str, int] = field(default_factory=lambda: {"goal": 0, "saved": 0, "miss": 0, "other": 0}, init=False, repr=False)

    def __post_init__(self) -> None:
        self._start_trial()

    def _start_trial(self) -> None:
        self._match = self.definition.build(self.rng_reduction)
        self._trial_tick = 0
        self._initial_carrier_id = self._match.ball.possessed_by
        self._initial_scoreboard = (
            self._match.scoreboard.left_goals,
            self._match.scoreboard.right_goals,
        )
        self._ball_released = False

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

        Returns ``True`` the tick a trial ends (the loop has already rebuilt
        a fresh match for the next trial, or set ``complete`` if max_trials
        reached).  Returns ``False`` on every other tick.
        """
        self._match.step()
        self._trial_tick += 1

        # Track when the initial carrier releases the ball.
        if not self._ball_released and self._initial_carrier_id is not None:
            if self._match.ball.possessed_by != self._initial_carrier_id:
                self._ball_released = True

        outcome = self._trial_outcome()
        if outcome is not None:
            self.outcomes[outcome] += 1
            self._trial_count += 1
            if not self.complete:
                self._start_trial()
            return True
        return False

    def _trial_outcome(self) -> str | None:
        """Returns the outcome key if the trial is over, else None."""
        pitch = self._match.pitch
        ball = self._match.ball
        scoreboard = self._match.scoreboard

        # Ball out of bounds (touchline or behind goal without scoring).
        if abs(ball.position.x) > pitch.half_length + 1.0:
            return "miss"
        if abs(ball.position.y) > pitch.half_width + 0.5:
            return "miss"

        # Scoreboard changed - a goal was scored.
        if (scoreboard.left_goals, scoreboard.right_goals) != self._initial_scoreboard:
            return "goal"

        # After the initial ball carrier has kicked/released: end as soon as
        # the ball resolves - picked up by anyone (GK save, tackle win,
        # receiver controls a pass) or gone dead (shot missed, rolled out).
        if self._ball_released:
            # Someone else now has possession - covered (GK save immediately
            # on control, without waiting for them to stop moving).
            if ball.possessed_by is not None and ball.possessed_by != self._initial_carrier_id:
                return "saved"
            # Ball loose and stationary - missed shot / failed pass rolled dead.
            # IMPORTANT: exclude the case where a player is mid-control-time
            # (CONTROLLING_BALL): the ball is frozen (v=0, possessed_by=None)
            # during the control delay, so without this guard we'd declare
            # "miss" the instant a GK starts catching the ball, before they
            # actually receive possession.  The balance tests use exactly this
            # same guard (gk.state != CONTROLLING_BALL).
            any_controlling = any(
                p.state == PlayerState.CONTROLLING_BALL for p in self._match.players
            )
            if ball.possessed_by is None and ball.velocity.length() < 0.1 and not any_controlling:
                return "miss"

        # Fallback for scenarios with no initial carrier (sprint, loose-ball
        # tackle): all non-persistent orders resolved and ball settled.
        if self._trial_tick >= 30 and all(p.current_order is None for p in self._match.players):
            if ball.velocity.length() < 0.1:
                return "other"

        # Timeout failsafe.
        if self._trial_tick >= self.timeout_ticks:
            return "other"

        return None
