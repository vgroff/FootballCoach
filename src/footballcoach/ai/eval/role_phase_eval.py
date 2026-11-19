"""Seeded evals for role-based curriculum phases (phase 2 onward), generic over
phases: a new role-based phase gets its periodic evals from its own
``eval_specs`` and ``curriculum[<config_key>].eval_seed_offset`` alone.

Each role-based CurriculumPhase lists ``eval_specs``: a name plus pinned
controllers per role, e.g. {"name": "attacker_vs_rules", "controllers":
{"attacker": "neural", "defender": "rules", "goalkeeper": "rules"}}. Every
spec runs the same fixed seed plan, sized like the phase-1 evals (the shared
``eval.eval_n_seeds`` x ``eval.eval_repeats_per_seed``) and offset per phase
(``eval_seed_offset``), through the real env, so numbers are comparable across
a run. With one neural role, the reported reward is that role's (it is the
episode's trainee); a spec with no neural role is a rules baseline, which the
trainer runs once and caches.

Which specs a training run evaluates (``specs_to_evaluate``): the rules-only
ones, plus every spec whose neural roles this run all trains (neural weight in
``curriculum[<config_key>].controllers`` above zero) -- evaluating a role
nobody trains says nothing about the run.
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


def _phase_cfg(phase: CurriculumPhase) -> dict:
    from footballcoach.ai.config import load_ai_config
    return load_ai_config()["curriculum"][phase.config_key]


def eval_plan(phase: CurriculumPhase) -> tuple[list[int], int]:
    """(seeds, repeats_per_seed): the shared eval budget (eval.eval_n_seeds x
    eval.eval_repeats_per_seed, the same as the phase-1 evals) on this phase's
    own seed range, starting at its eval_seed_offset."""
    from footballcoach.ai.config import load_ai_config
    ev = load_ai_config()["eval"]
    offset = int(_phase_cfg(phase)["eval_seed_offset"])
    return list(range(offset, offset + int(ev["eval_n_seeds"]))), int(ev["eval_repeats_per_seed"])


def _resolve(phase: CurriculumPhase, seeds, repeats_per_seed) -> tuple[list[int], int]:
    """Default: the phase's eval_plan. Explicit seeds without repeats run once each."""
    if seeds is None:
        seeds, plan_repeats = eval_plan(phase)
        return seeds, int(plan_repeats if repeats_per_seed is None else repeats_per_seed)
    return list(seeds), int(1 if repeats_per_seed is None else repeats_per_seed)


def is_rules_only(spec: dict) -> bool:
    return all(c == "rules" for c in spec["controllers"].values())


def trained_roles(phase: CurriculumPhase) -> set[str]:
    """Roles this phase's training can put a neural player in."""
    ctrl = _phase_cfg(phase).get("controllers", {})
    return {role for role, w in ctrl.items() if float(w.get("neural", 0)) > 0}


def specs_to_evaluate(phase: CurriculumPhase) -> list[dict]:
    """The rules-only specs, plus each spec whose neural roles this run ALL
    trains (a spec pairing a trained role with an untrained neural one, e.g.
    the AI attacker vs a never-trained AI keeper, says nothing about the run)."""
    trained = trained_roles(phase)
    return [s for s in phase.eval_specs
            if is_rules_only(s) or all(r in trained for r, c in s["controllers"].items() if c == "neural")]


def run_role_phase_eval(phase: CurriculumPhase, spec: dict, sample_action_fn, seeds=None,
                        repeats_per_seed=None) -> RolePhaseEvalResult:
    """Run ``spec`` of ``phase`` in this process over ``seeds`` x
    ``repeats_per_seed`` (see ``_resolve`` for the defaults), stochastic
    sampling like the phase-1 evals."""
    from footballcoach.ai.curriculum.envs import build_role_env
    from footballcoach.ai.eval.seeded_eval import run_seeded_evaluation

    controllers = dict(spec["controllers"])
    roles = set(phase.roles.values())
    if set(controllers) != roles:
        raise ValueError(f"{phase.name}/{spec['name']}: controllers must pin exactly the roles {sorted(roles)}")
    seeds, repeats_per_seed = _resolve(phase, seeds, repeats_per_seed)
    weights = {role: {ctrl: 1} for role, ctrl in controllers.items()}
    last_env = {}

    def _factory(seed: int, _swap: bool):
        env = build_role_env(phase, controller_weights=weights)
        env.controller_overrides = controllers
        env.scenario_kwargs = {**env.scenario_kwargs, **dict(spec.get("scenario_kwargs") or {}), "seed": seed}
        last_env["env"] = env
        return env

    result = run_seeded_evaluation(
        _factory, sample_action_fn, seeds, repeats_per_seed=repeats_per_seed, win_outcome="goal",
    )
    return RolePhaseEvalResult(
        phase_name=phase.name, spec_name=spec["name"], episode_tag=last_env["env"].episode_tag,
        outcomes=list(result.outcomes_list), rewards=list(result.rewards),
    )


def merge_role_results(results: list[RolePhaseEvalResult]) -> RolePhaseEvalResult:
    """One result from per-worker chunks of the same spec."""
    first = results[0]
    return RolePhaseEvalResult(
        phase_name=first.phase_name, spec_name=first.spec_name, episode_tag=first.episode_tag,
        outcomes=[o for r in results for o in r.outcomes], rewards=[x for r in results for x in r.rewards],
    )


def run_role_phase_eval_on_workers(workers, phase: CurriculumPhase, spec: dict, seeds=None,
                                   repeats_per_seed=None) -> RolePhaseEvalResult:
    """``run_role_phase_eval`` spread over a persistent eval worker pool
    (ai/eval/eval_worker.py) whose weights the caller has already set. Seeds
    are dealt round-robin; each worker runs every repeat of its own seeds."""
    seeds, repeats_per_seed = _resolve(phase, seeds, repeats_per_seed)
    chunks = [c for c in (seeds[i::len(workers)] for i in range(len(workers))) if c]
    active = workers[:len(chunks)]
    for w, chunk in zip(active, chunks):
        w.role_eval(phase.phase_id, spec, chunk, repeats_per_seed)
    return merge_role_results([w.recv_result() for w in active])
