"""Player-player collision resolution ("soft-body" overlap push-apart), plus
loose-ball blocking by inactive players' cylinders.

Players are cylinders viewed from above as circles of radius `radius_m`.
Per the design spec: the distance between the centres of any two players
must be >= r1 + r2. If violated, both players are pushed apart along the
line connecting their centres, weighted by their velocity component along
that line - this lets a faster-moving/harder-charging player "win" more of
the push, approximating momentum without a full mass/impulse system.

Inactive players (just tackled, or having just failed a tackle - see
`Player.is_inactive`) are excluded from this push-apart logic entirely: per
the design spec you can run straight through an inactive player rather than
being blocked by them. They can, however, still physically block a loose
ball that flies through their cylinder from outside it (see
`resolve_ball_block_by_inactive_players`) - a player lying on the ground
after a tackle can still deflect a stray shot, they just can't obstruct
other *players*' movement.

Velocity damping (applied after position push-apart) reduces the closing
speed of colliding players, even for inactive pairs: without this, a
just-tackled player would keep gliding at full speed into the opponent.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from footballcoach.config import load_physics_config, require_section
from footballcoach.entities.ball import Ball
from footballcoach.entities.player import Player
from footballcoach.mathutils import Vector3

log = logging.getLogger("footballcoach.collision")


@dataclass(frozen=True)
class CollisionParams:
    """Config-driven constants for player-player collision resolution."""
    collision_velocity_retention: float   # fraction of closing speed retained per tick (0.5 = 50% reduction)
    collision_damping_min_closing_speed_mps: float  # closing speeds below this are not damped

    @staticmethod
    def from_config() -> "CollisionParams":
        d = require_section(load_physics_config(), "collision")
        return CollisionParams(
            collision_velocity_retention=d["collision_velocity_retention"],
            collision_damping_min_closing_speed_mps=d["collision_damping_min_closing_speed_mps"],
        )


def _damp_overlap_velocity(
    player_a: Player,
    player_b: Player,
    params: CollisionParams,
) -> None:
    """Damps the closing velocity components for a potentially overlapping
    player pair. Only applies when the pair is actually overlapping AND the
    closing component exceeds the floor threshold, so it self-limits to a
    few ticks of sustained contact (see the worked example in the phase D
    design doc for the compounding analysis).

    Unlike the position push-apart, this runs for ALL pairs — including
    those where one or both players are inactive — so a just-tackled player
    does not keep gliding at full sprint speed into the opponent.

    Note: position z is preserved; only xy closing components are damped.
    """
    delta = player_b.position.xy() - player_a.position.xy()
    distance = delta.length()
    min_distance = player_a.radius_m + player_b.radius_m

    if distance >= min_distance:
        return  # not overlapping, nothing to damp

    if distance < 1e-9:
        direction = Vector3(1.0, 0.0, 0.0)
    else:
        direction = delta / distance  # unit vector from a toward b (xy, z=0)

    retention = params.collision_velocity_retention
    floor_speed = params.collision_damping_min_closing_speed_mps

    # Player A's closing component toward player B (positive = moving toward B)
    closing_a = player_a.velocity.xy().dot(direction)
    if closing_a > floor_speed:
        reduction_a = closing_a * (1.0 - retention)
        player_a.velocity = player_a.velocity - direction * reduction_a

    # Player B's closing component toward player A (positive = moving toward A)
    closing_b = -player_b.velocity.xy().dot(direction)
    if closing_b > floor_speed:
        reduction_b = closing_b * (1.0 - retention)
        player_b.velocity = player_b.velocity + direction * reduction_b


def resolve_player_overlap(player_a: Player, player_b: Player) -> None:
    """Mutates both players' positions to resolve any overlap between them."""
    delta = player_b.position.xy() - player_a.position.xy()
    distance = delta.length()
    min_distance = player_a.radius_m + player_b.radius_m

    if distance >= min_distance:
        return

    if distance < 1e-9:
        # Degenerate case: identical positions, push apart along an arbitrary axis.
        direction = Vector3(1.0, 0.0, 0.0)
        distance = 0.0
    else:
        direction = delta / distance

    overlap = min_distance - distance

    # Weight the push by each player's velocity component along the
    # separation axis (clamped to >= 0, since a player moving *away* along
    # this axis shouldn't be blamed for the overlap). A player charging hard
    # into another is pushed back less; the other is displaced more.
    speed_a_towards_b = max(0.0, player_a.velocity.xy().dot(direction))
    speed_b_towards_a = max(0.0, -player_b.velocity.xy().dot(direction))
    total_speed = speed_a_towards_b + speed_b_towards_a

    if total_speed < 1e-9:
        # Neither is moving into the other (e.g. both stationary) - split evenly.
        weight_a = 0.5
    else:
        # The player pushing harder (moving faster into the other) yields less
        # ground: weight_a is how much *player_a* gets pushed back.
        weight_a = speed_b_towards_a / total_speed

    weight_b = 1.0 - weight_a

    push_a = direction * (-overlap * weight_a)
    push_b = direction * (overlap * weight_b)

    log.debug(
        "[collision] %s<->%s  dist=%.3f  min=%.3f  overlap=%.3f  "
        "weight_a=%.3f  push_a=(%.3f,%.3f)  push_b=(%.3f,%.3f)",
        player_a.player_id, player_b.player_id,
        distance, min_distance, overlap, weight_a,
        push_a.x, push_a.y, push_b.x, push_b.y,
    )

    player_a.position = player_a.position + push_a
    player_b.position = player_b.position + push_b


def resolve_all_overlaps(
    players: list[Player],
    iterations: int = 2,
    collision_params: CollisionParams | None = None,
) -> None:
    """Resolves overlaps across all player pairs. Multiple iterations help
    settle chains of overlapping players (e.g. 3+ players bunched up).

    Position push-apart: pairs where either player is currently `is_inactive`
    are skipped entirely - per the design spec, you can run straight through
    an inactive player, rather than merely reducing the push.

    Velocity damping: applied once after all push-apart iterations, for ALL
    pairs (including those involving inactive players). This prevents a
    just-tackled player from continuing to glide at full sprint speed into
    the opponent, while keeping the position push-apart rule unchanged.
    """
    params = collision_params or CollisionParams.from_config()

    for _ in range(iterations):
        for i in range(len(players)):
            if players[i].is_inactive:
                continue
            for j in range(i + 1, len(players)):
                if players[j].is_inactive:
                    continue
                resolve_player_overlap(players[i], players[j])

    # Velocity damping — single pass over ALL pairs (active and inactive).
    # This is intentionally separate from the position push-apart loop above:
    # position push-apart is skipped for inactive pairs, but velocity damping
    # is not (see module docstring for rationale). Compounding is self-limiting
    # because the floor threshold stops damping once closing speed drops below
    # collision_damping_min_closing_speed_mps.
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            _damp_overlap_velocity(players[i], players[j], params)


def are_touching(player_a: Player, player_b: Player, tolerance_m: float = 0.05) -> bool:
    """Returns True if two players are close enough to be considered
    "touching" (e.g. for tackle eligibility)."""
    distance = player_a.position.xy().distance_to(player_b.position.xy())
    return distance <= player_a.radius_m + player_b.radius_m + tolerance_m


def resolve_ball_block_by_inactive_players(
    ball: Ball,
    players: list[Player],
    previous_ball_position: Vector3,
    block_restitution: float = 0.35,
) -> None:
    """Blocks a loose, in-flight ball against any *inactive* player's
    cylinder that it passed through this tick, per the design spec: "you
    shouldn't be able to bump into inactive players... however they can
    still block the ball from being shot if it is shot from outside their
    cylinder and crosses in (but not if it is shot from inside the
    cylinder)".

    Checks the ball's ground-plane movement segment (`previous_ball_position`
    -> `ball.position`) against each inactive player's circle. If the
    segment starts outside the cylinder and ends inside/beyond it (i.e. it
    was struck from outside and is crossing in), the ball is stopped at the
    entry point and its velocity is reflected/damped (a simple deflection,
    not a full rebound simulation) - it does NOT block a ball that started
    the tick already inside the cylinder (e.g. the ball being dribbled past
    them or just released at their feet), matching "not if it is shot from
    inside the cylinder".

    Only meaningful for a loose ball (this is a no-op if `ball.possessed_by`
    is set, since a possessed ball doesn't undergo free-flight physics
    anyway).
    """
    if ball.possessed_by is not None:
        return

    start = previous_ball_position.xy()
    end = ball.position.xy()
    segment = end - start
    segment_len = segment.length()
    if segment_len < 1e-9:
        return
    direction = segment / segment_len

    for player in players:
        if not player.is_inactive:
            continue

        centre = player.position.xy()
        radius = player.radius_m

        start_offset = start - centre
        # Distance from segment start to the circle boundary along `direction`,
        # via the standard ray-circle intersection quadratic.
        b = start_offset.dot(direction)
        c = start_offset.dot(start_offset) - radius * radius

        if c <= 0.0:
            # Ball started inside the cylinder this tick - per the spec,
            # don't block in this case (e.g. it was already at their feet).
            continue

        discriminant = b * b - c
        if discriminant < 0.0:
            continue  # ray never reaches the circle

        t_entry = -b - math.sqrt(discriminant)
        if t_entry < 0.0 or t_entry > segment_len:
            continue  # entry point isn't within this tick's travel segment

        # Struck from outside and crossing in this tick: stop the ball at
        # the entry point and deflect it (damped reflection off the
        # cylinder's surface normal at the entry point).
        entry_point_xy = start + direction * t_entry
        normal = (entry_point_xy - centre)
        normal_len = normal.length()
        normal = normal / normal_len if normal_len > 1e-9 else direction * -1.0

        velocity_xy = ball.velocity.xy()
        reflected_xy = velocity_xy - normal * (2.0 * velocity_xy.dot(normal))
        ball.velocity = (reflected_xy * block_restitution).with_z(ball.velocity.z * block_restitution)
        ball.position = Vector3(entry_point_xy.x, entry_point_xy.y, ball.position.z)
        return  # only block against the first inactive player hit this tick
