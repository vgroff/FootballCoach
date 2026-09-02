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
    """Return the ``(env, player_id=None) -> BCLabel`` rules-based BC label
    function for *phase_id*, or None.

    For callers that already have an ``env`` and want a label for its
    CURRENT state, synchronously, at the point they call this (e.g.
    record_demonstrations.py's recording loop, always called either before
    that decision interval's own env.step() or from inside an on_kick/
    on_tackle callback -- both points where "current state" and "the state
    the observation was captured from" are the same instant; also
    BCPretrainer._pretrain_online(), which similarly labels BEFORE stepping
    the env). See bc_label_fn_for_phase_player() for the OTHER calling
    convention, needed by any caller that can't guarantee that -- notably
    on-policy PPO/DAgger rollout collection, where the label must be
    computed synchronously inside NeuralPlayerAI.act() instead (see that
    function's own bc_label_fn parameter).
    """
    if phase_id == 1:
        from footballcoach.ai.ppo.bc import phase1_labels
        return phase1_labels
    return None


def bc_label_fn_for_phase_player(phase_id: int) -> Optional[Callable]:
    """Return the ``(player, match) -> BCLabel`` rules-based BC label
    function for *phase_id*, or None.

    This is the convention ``NeuralPlayerAI``/``ScenarioEnv.bc_label_fn``
    need (see ``NeuralPlayerAI.bc_label_fn``'s own docstring): the label
    must be computed SYNCHRONOUSLY inside ``NeuralPlayerAI.act()``, at the
    exact same instant the observation is encoded -- before
    ``Match._apply_movement()`` advances the player for that tick -- or it
    describes a state one physics tick later than the observation it's
    paired with (see ``phase1_labels_for_player()``'s "CRITICAL --
    CALLER-SIDE TIMING" docstring section for the full story). Used by
    on-policy PPO training (``PPOTrainer.train()``/``rollout_worker.py``)
    and DAgger (``ai/ppo/dagger.py``) -- never call this AFTER an
    ``env.step()`` has already returned and expect it to describe the
    observation THAT step produced; it won't.
    """
    if phase_id == 1:
        from footballcoach.ai.ppo.bc import phase1_labels_for_player
        return phase1_labels_for_player
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

    from footballcoach.ai.config import load_ai_config
    _curriculum_cfg = load_ai_config().get("curriculum", {})
    _rules_ratio = float(_curriculum_cfg.get("phase1_opponent_rules_ratio", 0.0))
    _immobile_ratio = float(_curriculum_cfg.get("phase1_opponent_immobile_ratio", 1.0))
    _neural_ratio = float(_curriculum_cfg.get("phase1_opponent_neural_ratio", 0.0))
    _total = _rules_ratio + _immobile_ratio + _neural_ratio
    _rules_prob = (_rules_ratio / _total) if _total > 0 else 0.0
    _immobile_prob = (_immobile_ratio / _total) if _total > 0 else 1.0
    defn = ScenarioDefinition(
        key="phase1_1v1",
        label="Phase 1: 1v1 Get Possession",
        description="1v1 scenario for curriculum phase 1",
        build=functools.partial(
            build_1v1_scenario,
            ball_max_speed_mps=10.0,
            opponent_rules_prob=_rules_prob,
            opponent_immobile_prob=_immobile_prob,
        ),
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
