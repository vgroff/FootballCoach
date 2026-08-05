"""Curriculum phase definitions.

See ai_design_doc.md section 4 for the full curriculum spec.

Each CurriculumPhase describes a training phase (what scenarios to use,
which heads to freeze, what reward function to use, etc.).

The MVP focuses on phases 1 and 2 only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from footballcoach.ai.config import load_ai_config


@dataclass
class CurriculumPhase:
    """Describes one curriculum training phase.

    Args:
        name: Human-readable phase name.
        phase_id: Integer ID (1-based, matching ai_design_doc.md section 4).
        env_kwargs: Extra kwargs passed to ScenarioEnv (e.g. scenario key,
            max_episode_s, etc.).
        frozen_heads: List of decision-network head names to freeze during
            this phase (weights not updated).  Empty list = all heads unfrozen.
        reward_phase: Which reward function to use (1 or 2, matching
            env/reward.py's phase1_reward / phase2_reward).
        description: Narrative description for logging/debugging.
    """
    name: str
    phase_id: int
    scenario_key: str              # Key into SCENARIOS dict from ui/scenarios.py
    env_kwargs: dict = field(default_factory=dict)
    frozen_heads: list[str] = field(default_factory=list)
    reward_phase: int = 1
    description: str = ""


# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

def _phase1_max_episode_s() -> float:
    return float(load_ai_config().get("curriculum", {}).get("phase1_max_episode_s", 240.0))


PHASE_1_GET_POSSESSION = CurriculumPhase(
    name="phase1_get_possession",
    phase_id=1,
    scenario_key="1v1",
    env_kwargs={
        "max_episode_s": _phase1_max_episode_s(),
    },
    # Freeze all decision heads except the latent vector in early training.
    # Gradually unfreeze as per the curriculum (this is done manually by the
    # training script, not automatically here).
    frozen_heads=[
        "shoot_logit", "pass_logit", "tackle_logit", "get_possession_raw",
        "mark_logit", "hold_position_logit",
        "pass_target_logits", "tackle_target_logits", "mark_target_logits",
    ],
    reward_phase=1,
    description=(
        "1v1 scenario: learn to get possession and bring the ball toward the "
        "opponent box.  Decision network frozen except Move/GetPossession and "
        "latent vector."
    ),
)

PHASE_2_SHOOT = CurriculumPhase(
    name="phase2_shoot",
    phase_id=2,
    scenario_key="penalty",
    env_kwargs={
        "max_episode_s": 60.0,
    },
    frozen_heads=[],  # All heads active; shoot-probability head specifically rewarded
    reward_phase=2,
    description=(
        "Shooting scenarios (empty goal, GK, static defender): learn to shoot "
        "quickly and accurately.  GK remains rules-based."
    ),
)

# Future phases (stubs for reference, not yet implemented):
PHASE_3_PASSING = CurriculumPhase(
    name="phase3_passing",
    phase_id=3,
    scenario_key="pass",
    reward_phase=1,
    description="Passing scenarios with immobile/moving teammates.",
)

PHASE_4_TACKLING = CurriculumPhase(
    name="phase4_tackling",
    phase_id=4,
    scenario_key="tackle",
    reward_phase=1,
    description="Tackling scenarios; uses phase-2 shooting AI as the dribbler.",
)

ALL_PHASES: list[CurriculumPhase] = [
    PHASE_1_GET_POSSESSION,
    PHASE_2_SHOOT,
    PHASE_3_PASSING,
    PHASE_4_TACKLING,
]

PHASES_BY_ID: dict[int, CurriculumPhase] = {p.phase_id: p for p in ALL_PHASES}
