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
        reward_phase: Which reward function to use. Phase 1 uses
            env/reward.py's phase1_reward; role-based phases ignore it.
        description: Narrative description for logging/debugging.
        roles: player_id -> role (e.g. "attacker", "defender",
            "goalkeeper"). Non-empty makes this a ROLE-BASED phase, built by
            curriculum/envs.py's build_role_env as an ai/env/role_env.py
            RoleScenarioEnv: controllers rolled per role, terminal-only
            rewards from a per-role table. Phase 1 leaves it empty and keeps
            its own builder. The order is the order the episode's trainee is
            chosen in (first neural player).
        config_key: section name under ai_config.json ``curriculum`` (env
            settings: max_episode_s, controllers) and ``reward`` (the
            ``terminal`` table) for a role-based phase.
        outcome_kind: which outcome vocabulary / classifier a role-based
            phase uses (curriculum/envs.py OUTCOME_KINDS).
        eval_specs: role-based phases only: the periodic seeded evals, each
            {"name": ..., "controllers": {role: "rules" | "neural"}} pinning
            every role (ai/eval/role_phase_eval.py). A spec with no neural
            role is a rules baseline. A spec may also carry "scenario_kwargs",
            overriding the configured build kwargs for that eval (e.g.
            {"no_keeper_prob": 1.0}).
        optional_roles: roles the scenario may leave out of an episode (e.g.
            phase 2's goalkeeper under ``no_keeper_prob``). Such an episode
            simply has no player in that role; a missing role NOT listed here
            is an error.
    """
    name: str
    phase_id: int
    scenario_key: str              # Key into SCENARIOS dict from ui/scenarios.py
    env_kwargs: dict = field(default_factory=dict)
    frozen_heads: list[str] = field(default_factory=list)
    reward_phase: int = 1
    description: str = ""
    roles: dict[str, str] = field(default_factory=dict)
    config_key: str = ""
    outcome_kind: str = ""
    eval_specs: list[dict] = field(default_factory=list)
    optional_roles: frozenset = frozenset()

    @property
    def is_role_based(self) -> bool:
        return bool(self.roles)


# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

def _phase1_max_episode_s() -> float:
    return float(load_ai_config().get("curriculum", {}).get("phase1_max_episode_s", 240.0))


def _phase1_max_episode_s_variability() -> float:
    """Extra, uniformly-sampled seconds ADDED to _phase1_max_episode_s() each
    episode (see ScenarioEnv's own max_episode_s_variability_s docstring) --
    0.0 (default, and the live value as of this writing) means every episode
    gets exactly _phase1_max_episode_s(), identical to before this knob
    existed."""
    return float(load_ai_config().get("curriculum", {}).get("phase1_max_episode_s_variability", 0.0))


PHASE_1_GET_POSSESSION = CurriculumPhase(
    name="phase1_get_possession",
    phase_id=1,
    scenario_key="1v1",
    env_kwargs={
        "max_episode_s": _phase1_max_episode_s(),
        "max_episode_s_variability_s": _phase1_max_episode_s_variability(),
    },
    # Freeze all decision heads except the latent vector in early training.
    # Gradually unfreeze as per the curriculum (this is done manually by the
    # training script, not automatically here).
    frozen_heads=[
        "shoot_logit", "pass_logit", "tackle_logit",
        "mark_logit", "hold_position_logit",
        "pass_target_logits", "tackle_target_logits", "mark_target_logits",
    ],
    reward_phase=1,
    description=(
        "1v1 scenario: learn to get possession and bring the ball toward the "
        "opponent box.  Decision network frozen except Move/GetPossession "
        "and the shared latent vector -- Move and GetPossession are the "
        "only two decision-level actions the rules-AI actually exercises "
        "in this scenario, kept together deliberately (confirmed "
        "2026-09-01): either both stay trainable or neither does."
    ),
)

def _role_phase_max_episode_s(config_key: str) -> float:
    return float(load_ai_config()["curriculum"][config_key]["max_episode_s"])


PHASE_2_1V2 = CurriculumPhase(
    name="phase2_1v2",
    phase_id=2,
    scenario_key="1v2",
    env_kwargs={"max_episode_s": _role_phase_max_episode_s("phase2")},
    # Same frozen heads as phase 1 (shooting uses the execution kick head), so
    # phases 1 and 2 can share one PPO update's head mask when trained together.
    frozen_heads=list(PHASE_1_GET_POSSESSION.frozen_heads),
    reward_phase=2,
    description=(
        "1v2: an attacker on the ball against a defender and a goalkeeper. "
        "Terminal-only rewards per role (ai_config.json reward.phase2.terminal); "
        "each role's controller (rules / neural) is rolled per episode from "
        "curriculum.phase2.controllers."
    ),
    roles={"attacker": "attacker", "defender": "defender", "keeper": "goalkeeper"},
    config_key="phase2",
    outcome_kind="attack_defence",
    # The main specs always keep the keeper (comparable with runs before
    # no_keeper_prob); the *_no_keeper specs are the empty-goal case alone.
    eval_specs=[
        {"name": "rules_baseline", "controllers": {"attacker": "rules", "defender": "rules", "goalkeeper": "rules"},
         "scenario_kwargs": {"no_keeper_prob": 0.0}},
        {"name": "attacker_vs_rules", "controllers": {"attacker": "neural", "defender": "rules", "goalkeeper": "rules"},
         "scenario_kwargs": {"no_keeper_prob": 0.0}},
        {"name": "defender_vs_rules", "controllers": {"attacker": "rules", "defender": "neural", "goalkeeper": "rules"},
         "scenario_kwargs": {"no_keeper_prob": 0.0}},
        # Neural keeper (2026-11-19): vs the rules attacker (compare rules_baseline)
        # and vs the AI attacker (compare attacker_vs_rules, the same attacker
        # against the rules keeper). Evaluated whenever the keeper trains.
        {"name": "goalkeeper_vs_rules", "controllers": {"attacker": "rules", "defender": "rules", "goalkeeper": "neural"},
         "scenario_kwargs": {"no_keeper_prob": 0.0}},
        {"name": "attacker_vs_ai_keeper", "controllers": {"attacker": "neural", "defender": "rules", "goalkeeper": "neural"},
         "scenario_kwargs": {"no_keeper_prob": 0.0}},
        # All three roles neural (user, 2026-11-20: the most interesting
        # overall view -- scored / saved / defended in the all-AI game).
        {"name": "all_neural", "controllers": {"attacker": "neural", "defender": "neural", "goalkeeper": "neural"},
         "scenario_kwargs": {"no_keeper_prob": 0.0}},
        {"name": "rules_baseline_no_keeper", "controllers": {"attacker": "rules", "defender": "rules", "goalkeeper": "rules"},
         "scenario_kwargs": {"no_keeper_prob": 1.0}},
        {"name": "attacker_vs_rules_no_keeper", "controllers": {"attacker": "neural", "defender": "rules", "goalkeeper": "rules"},
         "scenario_kwargs": {"no_keeper_prob": 1.0}},
    ],
    optional_roles=frozenset({"goalkeeper"}),
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
    PHASE_2_1V2,
    PHASE_3_PASSING,
    PHASE_4_TACKLING,
]

PHASES_BY_ID: dict[int, CurriculumPhase] = {p.phase_id: p for p in ALL_PHASES}
