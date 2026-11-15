"""Seeded evals for role-based curriculum phases (phase 2 onward).

Each role-based CurriculumPhase lists ``eval_specs``: a name plus pinned
controllers per role, e.g. {"name": "attacker_vs_rules", "controllers":
{"attacker": "neural", "defender": "rules", "goalkeeper": "rules"}}. Every
spec runs the same fixed seed list (ai_config.json curriculum[<config_key>]
eval_episodes / eval_seed_offset) through the real env, so numbers are
comparable across a run. With one neural role, the reported reward is that
role's (it is the episode's trainee); a spec with no neural role is a rules
baseline that never changes and can be cached.
"""
from __future__ import annotations

from dataclasses import dataclass

from footballcoach.ai.curriculum.phases import CurriculumPhase


@dataclass
class RolePhaseEvalResult:
    phase_name: str
    spec_name: str
    episode_tag: str
    outcomes: list
    rewards: list          # the trainee's (first neural role's) episode reward; all-rules: the first role's

    def summary(self) -> str:
        from footballcoach.ai.ppo.rollout_logging import format_tag_outcomes
        return format_tag_outcomes(self.episode_tag, self.outcomes, self.rewards)


def eval_seeds(phase: CurriculumPhase) -> list[int]:
    from footballcoach.ai.config import load_ai_config
    cfg = load_ai_config()["curriculum"][phase.config_key]
    n, offset = int(cfg["eval_episodes"]), int(cfg["eval_seed_offset"])
    return list(range(offset, offset + n))


def run_role_phase_eval(phase: CurriculumPhase, spec: dict, sample_action_fn, seeds=None) -> RolePhaseEvalResult:
    """Run ``spec`` of ``phase`` over ``seeds`` (default: the phase's fixed
    eval seeds), one episode per seed, stochastic sampling like the phase-1
    evals."""
    from footballcoach.ai.curriculum.envs import build_role_env
    from footballcoach.ai.eval.seeded_eval import run_seeded_evaluation

    controllers = dict(spec["controllers"])
    roles = set(phase.roles.values())
    if set(controllers) != roles:
        raise ValueError(f"{phase.name}/{spec['name']}: controllers must pin exactly the roles {sorted(roles)}")
    weights = {role: {ctrl: 1} for role, ctrl in controllers.items()}
    last_env = {}

    def _factory(seed: int, _swap: bool):
        env = build_role_env(phase, controller_weights=weights)
        env.controller_overrides = controllers
        env.scenario_kwargs = {**env.scenario_kwargs, "seed": seed}
        last_env["env"] = env
        return env

    result = run_seeded_evaluation(
        _factory, sample_action_fn, list(seeds if seeds is not None else eval_seeds(phase)),
        repeats_per_seed=1, win_outcome="goal",
    )
    return RolePhaseEvalResult(
        phase_name=phase.name, spec_name=spec["name"], episode_tag=last_env["env"].episode_tag,
        outcomes=list(result.outcomes_list), rewards=list(result.rewards),
    )
