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

def opponent_type_probs(
    rules_ratio: float, immobile_ratio: float, neural_ratio: float,
) -> tuple[float, float]:
    """Convert the three curriculum ratios (``ai_config.json``'s
    ``phase1_opponent_{rules,immobile,neural}_ratio``) into the
    ``(opponent_rules_prob, opponent_immobile_prob)`` pair
    ``build_1v1_scenario`` actually takes -- the neural probability is
    whatever's left over (``1 - rules_prob - immobile_prob``), matching
    ``build_1v1_scenario``'s own roll (see its docstring: "the remainder
    becomes neural").

    Extracted from ``_build_phase1_env`` as its own pure function
    specifically so this arithmetic -- the thing that turns e.g. a
    ``1 : 0 : 3`` config ratio into "75% of phase-1 training is self-play"
    -- can be unit-tested directly. Every existing self-play test
    deliberately bypasses ``_build_phase1_env``/``load_ai_config()``
    entirely (pytest-xdist workers share the config file, so mutating it
    mid-run is unsafe), which meant this exact conversion had never been
    exercised by anything. This file has already shipped one silent
    "config value has zero effect on real training" bug before (see the
    ``ball_max_speed_mps`` comment below) -- not a hypothetical risk.

    Falls back to ``(0.0, 1.0)`` (always immobile) when all three ratios
    sum to <= 0, matching ``phase1_opponent_immobile_ratio``'s own config
    default of ``1.0``.
    """
    total = rules_ratio + immobile_ratio + neural_ratio
    if total <= 0:
        return 0.0, 1.0
    return rules_ratio / total, immobile_ratio / total


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
    _rules_prob, _immobile_prob = opponent_type_probs(_rules_ratio, _immobile_ratio, _neural_ratio)
    defn = ScenarioDefinition(
        key="phase1_1v1",
        label="Phase 1: 1v1 Get Possession",
        description="1v1 scenario for curriculum phase 1",
        build=functools.partial(
            build_1v1_scenario,
            # ball_max_speed_mps deliberately NOT passed here (was hardcoded
            # to 10.0 until 2026-09-03) -- omitting it lets build_1v1_scenario
            # fall back to its own config-driven default
            # (ai_config.json["phase1_scenario"]["ball_max_speed_mps"]),
            # matching every other scenario param in this call. The hardcoded
            # literal meant this config value had zero effect on real
            # training regardless of what it was set to; found while
            # debugging why debug_rulesai_score.py's outcomes looked
            # suspiciously clean.
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
