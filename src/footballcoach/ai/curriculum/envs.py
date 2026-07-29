"""Factory functions for building training environments and BC label functions.

Single source of truth for all phase → ScenarioEnv mappings. Used by both
the training script (train.py) and the demonstration recorder
(record_demonstrations.py), so there is no duplication between them.

Add a new elif branch here when adding a new phase.
"""
from __future__ import annotations

from typing import Callable, Optional

from footballcoach.ai.curriculum.phases import CurriculumPhase


def build_env(phase: CurriculumPhase):
    """Build a ScenarioEnv for *phase*."""
    if phase.phase_id == 1:
        return _build_phase1_env(phase)
    elif phase.phase_id == 2:
        return _build_phase2_env(phase)
    else:
        raise NotImplementedError(f"Phase {phase.phase_id} not yet implemented")


def bc_label_fn_for_phase(phase_id: int) -> Optional[Callable]:
    """Return the rules-based BC label function for *phase_id*, or None."""
    if phase_id == 1:
        from footballcoach.ai.ppo.bc import phase1_labels
        return phase1_labels
    return None


# ---------------------------------------------------------------------------
# Per-phase builders (private)
# ---------------------------------------------------------------------------

def _build_phase1_env(phase: CurriculumPhase):
    import functools
    from footballcoach.ai.env.scenario_env import ScenarioEnv
    from footballcoach.ui.scenarios import (
        build_1v1_scenario,
        phase1_training_on_tick,
        ScenarioDefinition,
    )

    defn = ScenarioDefinition(
        key="phase1_1v1",
        label="Phase 1: 1v1 Get Possession",
        description="1v1 scenario for curriculum phase 1",
        build=functools.partial(build_1v1_scenario, ball_max_speed_mps=4.0),
        on_tick=phase1_training_on_tick,
    )
    return ScenarioEnv(
        definition=defn,
        trainee_player_id="trainee",
        phase=1,
        secondary_player_ids=["opponent"],
        **phase.env_kwargs,
    )


def _build_phase2_env(phase: CurriculumPhase):
    from footballcoach.ai.env.scenario_env import ScenarioEnv
    from footballcoach.ui.scenarios import build_penalty_scenario, ScenarioDefinition

    defn = ScenarioDefinition(
        key="phase2_penalty",
        label="Phase 2: Shoot",
        description="Penalty scenario for curriculum phase 2",
        build=build_penalty_scenario,
    )
    return ScenarioEnv(
        definition=defn,
        trainee_player_id="kicker",
        phase=2,
        **phase.env_kwargs,
    )
