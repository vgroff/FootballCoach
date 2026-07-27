"""Player-player collision resolution ("soft-body" overlap push-apart).

Players are cylinders viewed from above as circles of radius `radius_m`.
Per the design spec: the distance between the centres of any two players
must be >= r1 + r2. If violated, both players are pushed apart along the
line connecting their centres, weighted by their velocity component along
that line - this lets a faster-moving/harder-charging player "win" more of
the push, approximating momentum without a full mass/impulse system.
"""
from __future__ import annotations

from footballcoach.entities.player import Player
from footballcoach.mathutils import Vector3


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

    player_a.position = player_a.position + push_a
    player_b.position = player_b.position + push_b


def resolve_all_overlaps(players: list[Player], iterations: int = 2) -> None:
    """Resolves overlaps across all player pairs. Multiple iterations help
    settle chains of overlapping players (e.g. 3+ players bunched up)."""
    for _ in range(iterations):
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                resolve_player_overlap(players[i], players[j])


def are_touching(player_a: Player, player_b: Player, tolerance_m: float = 0.05) -> bool:
    """Returns True if two players are close enough to be considered
    "touching" (e.g. for tackle eligibility)."""
    distance = player_a.position.xy().distance_to(player_b.position.xy())
    return distance <= player_a.radius_m + player_b.radius_m + tolerance_m
