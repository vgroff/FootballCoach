"""Opponent pool management for self-play and curriculum training.

See ai_design_doc.md section 4 (curriculum item: "opponent progression:
immobile -> sometimes rules-based AI -> sometimes older AI generations").

This module manages:
1. Rules-based opponents (the existing orders/actions API)
2. Frozen checkpoint opponents (old AI generations for self-play)

For the MVP experiments, only rules-based opponents are used.
Checkpoint-based self-play opponents are stubbed for future extension.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

log = logging.getLogger("footballcoach.ai.curriculum")


@dataclass
class OpponentPool:
    """Manages a pool of opponents for curriculum training.

    Args:
        rules_based_fraction: Probability of using a rules-based opponent
            (vs. a frozen checkpoint opponent) at each episode.
        checkpoint_dir: Directory containing frozen checkpoint files
            (for self-play). If None or empty, always uses rules-based.
        max_checkpoints: Maximum number of checkpoint opponents to keep
            in the pool at once (oldest are evicted).
    """
    rules_based_fraction: float = 1.0
    checkpoint_dir: Optional[Path] = None
    max_checkpoints: int = 5
    _checkpoints: list[Path] = field(default_factory=list, init=False, repr=False)
    _rng: random.Random = field(default_factory=random.Random, init=False, repr=False)

    def should_use_rules_based(self) -> bool:
        """Returns True if the rules-based opponent should be used this episode."""
        if not self._checkpoints:
            return True
        return self._rng.random() < self.rules_based_fraction

    def add_checkpoint(self, path: Path) -> None:
        """Add a new checkpoint to the pool (evicting oldest if full)."""
        self._checkpoints.append(path)
        if len(self._checkpoints) > self.max_checkpoints:
            evicted = self._checkpoints.pop(0)
            log.debug(f"Opponent pool: evicted checkpoint {evicted}")
        log.info(f"Opponent pool: added checkpoint {path} ({len(self._checkpoints)} total)")

    def sample_checkpoint(self) -> Optional[Path]:
        """Sample a random checkpoint opponent, or None if pool is empty."""
        if not self._checkpoints:
            return None
        return self._rng.choice(self._checkpoints)


def apply_rules_based_opponent(match, opponent_player_id: str, ball_carrier_id: Optional[str]) -> None:
    """Apply rules-based orders to an opponent player for one decision tick.

    Implements the simplest rules-based logic from curriculum phase 1:
      - If no one has the ball: GetPossessionOrder
      - If opponent has the ball: GetPossessionOrder + ChaseTackleOrder
      - If self (trainee) has the ball: MoveOrder (toward trainee's box)

    This logic mirrors what the user's ai_plan.md describes as the
    "rules-based AI to use as positive examples and adversary."

    Args:
        match: The running Match instance.
        opponent_player_id: The player_id of the rules-based opponent.
        ball_carrier_id: player_id of the current ball carrier (None if loose).
    """
    from footballcoach.orders import ChaseTackleOrder, GetPossessionOrder, MoveOrder

    try:
        opp = match.player_by_id(opponent_player_id)
    except (KeyError, AttributeError):
        return

    if ball_carrier_id is None:
        # Loose ball: chase it
        opp.current_order = GetPossessionOrder()
    elif ball_carrier_id == opponent_player_id:
        # Opponent has the ball: move toward trainee's goal
        # (trainee is LEFT => their goal is at -x; opponent is RIGHT => attacks -x)
        from footballcoach.entities.player import Team
        if opp.team == Team.LEFT:
            goal_x = match.pitch.half_length
        else:
            goal_x = -match.pitch.half_length
        from footballcoach.mathutils import Vector3
        opp.current_order = MoveOrder(
            target_position=Vector3(goal_x, opp.position.y, 0.0),
            sprint=True,
        )
    else:
        # Trainee has the ball: tackle them
        opp.current_order = ChaseTackleOrder(target_player_id=ball_carrier_id)
