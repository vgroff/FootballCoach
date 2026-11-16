"""Training several curriculum phases in one run.

A phase mix is {phase_id: weight}. Each rollout env slot is given one phase
for the whole run (assign_env_phases), so the weights are shares of env
slots, not of rows or episodes: short-episode phases contribute more
episodes per slot, and every phase's rows come at the rate its episodes
produce them. The rollout summary logs per-phase outcome tallies to tune
against.

Episode-seed replay stays phase-correct by tagging seeds with their phase
(tag_seed): a phase-1 seed is its plain value, so a phase-1-only run is
unchanged; a phase-N seed is offset by (N - 1) * PHASE_SEED_STRIDE, and
workers hand each replayed seed only to an env of its own phase.
"""
from __future__ import annotations

from footballcoach.ai.curriculum.phases import PHASES_BY_ID, CurriculumPhase

# Fresh episode seeds are drawn from [0, 2**31), so a phase offset of 2**31
# per phase id can never collide with another phase's seeds.
PHASE_SEED_STRIDE = 2 ** 31


def tag_seed(seed: int, phase_id: int) -> int:
    if not 0 <= seed < PHASE_SEED_STRIDE:
        raise ValueError(f"episode seed {seed} outside [0, 2**31)")
    return seed + (phase_id - 1) * PHASE_SEED_STRIDE


def untag_seed(tagged: int) -> tuple[int, int]:
    """(seed, phase_id) for a tag_seed() value."""
    return tagged % PHASE_SEED_STRIDE, tagged // PHASE_SEED_STRIDE + 1


def parse_phase_mix(spec) -> dict[int, float]:
    """{phase_id: weight} from a dict ({"1": 3, "2": 1} or int keys) or a
    "1:3,2:1" string. A bare "2" means {2: 1.0}."""
    if isinstance(spec, dict):
        items = spec.items()
    else:
        items = []
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            pid, _, w = part.partition(":")
            items.append((pid, w or "1"))
    mix: dict[int, float] = {}
    for pid, w in items:
        pid_i, w_f = int(pid), float(w)
        if pid_i in mix:
            raise ValueError(f"phase {pid_i} listed twice in the phase mix")
        if w_f < 0:
            raise ValueError(f"phase {pid_i}: negative weight {w_f}")
        if w_f > 0:
            mix[pid_i] = w_f
    if not mix:
        raise ValueError(f"phase mix {spec!r} has no phase with a positive weight")
    return dict(sorted(mix.items()))


def validate_training_mix(mix: dict[int, float]) -> list[CurriculumPhase]:
    """The mix's phases, after checking each can be trained alongside the
    others: it exists and is implemented, all share one frozen-head set (one
    PPO update applies one head mask), and a role-based phase has at least one
    role that can roll neural (an episode with no neural player has no
    trainee to collect)."""
    from footballcoach.ai.config import load_ai_config

    phases = []
    for pid in mix:
        if pid not in PHASES_BY_ID:
            raise ValueError(f"phase mix names unknown phase {pid}; known: {sorted(PHASES_BY_ID)}")
        phase = PHASES_BY_ID[pid]
        if not (phase.is_role_based or pid == 1):
            raise ValueError(f"phase {pid} ({phase.name}) is not implemented")
        if phase.is_role_based:
            controllers = load_ai_config()["curriculum"][phase.config_key]["controllers"]
            if not any(float(controllers[role].get("neural", 0)) > 0 for role in set(phase.roles.values())):
                raise ValueError(
                    f"{phase.name}: no role can roll 'neural' (ai_config.json curriculum."
                    f"{phase.config_key}.controllers), so its episodes have no trainee"
                )
        phases.append(phase)
    heads = {tuple(sorted(p.frozen_heads)) for p in phases}
    if len(heads) > 1:
        raise ValueError(
            "phases in one training mix must share frozen_heads (one PPO update applies one head mask); "
            + "; ".join(f"{p.name}: {sorted(p.frozen_heads)}" for p in phases)
        )
    return phases


def assign_env_phases(mix: dict[int, float], n_slots: int) -> list[int]:
    """Phase id for each of n_slots env slots: counts by largest remainder,
    interleaved (each slot goes to the phase furthest below its share so far),
    so every worker's contiguous block of slots gets a similar mix."""
    if n_slots <= 0:
        raise ValueError("n_slots must be positive")
    total = sum(mix.values())
    exact = {pid: n_slots * w / total for pid, w in mix.items()}
    counts = {pid: int(x) for pid, x in exact.items()}
    for pid in sorted(exact, key=lambda p: (-(exact[p] - counts[p]), p))[: n_slots - sum(counts.values())]:
        counts[pid] += 1
    out: list[int] = []
    given = {pid: 0 for pid in mix}
    for k in range(n_slots):
        pid = max((p for p in mix if given[p] < counts[p]),
                  key=lambda p: (counts[p] * (k + 1) / n_slots - given[p], -p))
        out.append(pid)
        given[pid] += 1
    return out


def top_fraction_per_phase(items, fraction: float, *, seed_of, score_of) -> list:
    """The top ``fraction`` of ``items`` by ``score_of`` (descending), taken
    WITHIN each phase separately (phase from the tagged seed ``seed_of(item)``)
    and at least one per phase that has items. Phases have different reward
    and value scales, so ranking them together would let one phase take every
    replay slot. With a single phase this is exactly the plain top fraction
    (same ceil, same stable order). Returned grouped by phase id, each group
    in score order."""
    import math

    by_phase: dict[int, list] = {}
    for it in items:
        by_phase.setdefault(untag_seed(seed_of(it))[1], []).append(it)
    out = []
    for _pid, group in sorted(by_phase.items()):
        n_top = max(1, math.ceil(fraction * len(group)))
        out.extend(sorted(group, key=score_of, reverse=True)[:n_top])
    return out


def route_seeds_to_workers(seeds: list[int], worker_phases: list[set[int]]) -> list[list[int]]:
    """Split tagged replay seeds across workers, each seed only to workers that
    run an env of its phase (a worker without one could never replay it),
    round-robin within each phase. ``worker_phases[w]`` is the set of phases
    worker w's envs run. With every worker running every phase (a single-phase
    run) this is the plain round-robin ``seeds[w::n_workers]``. Seeds of a
    phase no worker runs are dropped."""
    out: list[list[int]] = [[] for _ in worker_phases]
    next_idx: dict[int, int] = {}
    for s in seeds:
        pid = untag_seed(s)[1]
        eligible = [w for w, ph in enumerate(worker_phases) if pid in ph]
        if not eligible:
            continue
        k = next_idx.get(pid, 0)
        out[eligible[k % len(eligible)]].append(s)
        next_idx[pid] = k + 1
    return out
