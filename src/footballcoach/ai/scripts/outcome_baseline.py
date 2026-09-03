"""Outcome-distribution baseline: run N phase-1 episodes with the rules AI
as trainee and report the % of each outcome (win/loss/timeout/ball_out/
invalid/...).

Exists as a fast "before vs after" sanity check for engine-behaviour changes
(e.g. the boundary-braking/pickup-radius work) -- run once before a change,
once after, diff the percentages. Not the same thing as evaluate.py's
``--baseline-only`` baseline: that one is a fixed rules-vs-rules/rules-vs-
immobile NOISE-FLOOR probe. This script instead reproduces the REAL phase-1
training/demonstration distribution -- same ``build_1v1_scenario`` call the
curriculum env (``curriculum/envs.py``) and ``record_demonstrations.py``
actually use, including the real, config-driven ``ball_max_speed_mps``
(``ai_config.json["phase1_scenario"]["ball_max_speed_mps"]``) and the
config-driven opponent-type mix (``ai_config.json["curriculum"]``
phase1_opponent_*_ratio, immobile by default) -- so the outcome mix here
should match what shows up in real recorded episodes, not a separate easier
regime. (Until 2026-09-03, ``curriculum/envs.py`` and this script's own
``--ball-max-speed-mps`` default both silently hardcoded 10.0 instead of
reading ``phase1_scenario.ball_max_speed_mps`` -- found and fixed together;
see ``debug_rulesai_score.py``'s matching fix for the full story.)

Every run is fully deterministic (same seed -> byte-identical episode, see
_env_worker_factory's own docstring) and always includes KNOWN_SEEDS -- a
handful of specific seeds from real diagnosed worst-episode bugs -- folded
into the same sample/percentages as the random seeds (--extra-seeds to
change the list), with each one ALSO echoed individually afterwards so you
can see at a glance whether a fix changed its specific outcome.

Usage:
    uv run python -m footballcoach.ai.scripts.outcome_baseline \\
        --n-episodes 500 --output results/outcome_baseline.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("footballcoach.ai.outcome_baseline")

# Distinct from evaluate.py's baseline range (2_000_000+) and ai_config's
# default eval seeds (1_000_000+) so runs from all three never collide.
DEFAULT_SEED_BASE = 3_000_000

# Specific seeds from real recorded worst-episode diagnostics (debug_value_
# network.py's --physics-encoder-value-net worst-episode logs), confirmed via
# replay_episode.py/build_replay_match to reproduce exactly. Included in
# every run (see --extra-seeds) so a before/after diff always covers these
# known failure modes directly, rather than relying on a large-n random
# sample to happen to hit their (very rare) natural occurrence rate.
KNOWN_SEEDS: dict[int, str] = {
    35361186: "ball_out -- pickup granted off an already-out-of-bounds ball "
               "position, converting a would-be 'invalid' into a penalised "
               "'ball_out' (diagnosed 2026-08-31)",
    1594255696: "invalid -- genuine near-miss, chaser fell ~9cm short of "
                "pickup radius at the exact tick the ball crossed out even "
                "at full unbraked sprint (diagnosed 2026-08-31)",
}


def _real_phase1_max_episode_s() -> float:
    """The actual phase-1 episode timeout (ai_config.json["curriculum"]
    ["phase1_max_episode_s"], 18.5s by default) -- see curriculum/phases.py's
    PHASE_1_GET_POSSESSION.env_kwargs, which is what real training actually
    passes to ScenarioEnv. Read here rather than hardcoded so the two can
    never drift apart."""
    from footballcoach.ai.curriculum.phases import PHASE_1_GET_POSSESSION
    return float(PHASE_1_GET_POSSESSION.env_kwargs["max_episode_s"])


def _env_worker_factory(
    ball_max_speed_mps: float, max_episode_s: float, decision_interval_ticks: int = 1,
) -> tuple:
    """Module-level (picklable) factory: rules-AI trainee, opponent type
    drawn from the real curriculum config ratios (build_1v1_scenario's own
    opponent_rules_prob/opponent_immobile_prob defaults -- immobile-only
    unless ai_config.json["curriculum"] says otherwise).

    Deliberately deterministic: unlike evaluate.py's noise-floor baseline
    (which reseeds match.rng with fresh entropy per episode so repeated runs
    of the SAME seed sample independent noise), this script's whole point is
    a reproducible before/after diff across code changes -- the same seed
    must produce the exact same episode every run, or a diff between two
    runs can't tell a real behaviour change from random noise. So
    build_1v1_scenario's own seeded RNG is left untouched (matches
    replay_episode.py's build_replay_match, confirmed elsewhere this session
    to reproduce recorded episodes exactly)."""
    from footballcoach.ui.scenarios import build_1v1_scenario, ScenarioDefinition
    from footballcoach.rules_ai import Phase1RulesAI
    from footballcoach.ai.env.scenario_env import ScenarioEnv

    def _build(*args, **kwargs):
        match = build_1v1_scenario(*args, ball_max_speed_mps=ball_max_speed_mps, **kwargs)
        match.player_by_id("trainee").ai = Phase1RulesAI(decision_interval_ticks=decision_interval_ticks)
        # Opponent AI (rules/immobile/neural-less) is already assigned by
        # build_1v1_scenario itself according to opponent_rules_prob/
        # opponent_immobile_prob -- nothing to override here.
        return match

    def _env_factory(seed: int) -> ScenarioEnv:
        defn = ScenarioDefinition(
            key="outcome_baseline_1v1",
            label="Outcome baseline: rules trainee vs curriculum-mix opponent",
            description="Real phase-1 scenario params, rules AI trainee, for outcome-% baselining",
            build=lambda *a, **kw: _build(*a, seed=seed, **kw),
            on_tick=None,
        )
        return ScenarioEnv(
            definition=defn, trainee_player_id="trainee", phase=1, secondary_player_ids=["opponent"],
            max_episode_s=max_episode_s,
        )

    return _env_factory, None


def _real_ball_max_speed_mps() -> float:
    """The actual phase-1 ball speed cap (ai_config.json["phase1_scenario"]
    ["ball_max_speed_mps"]) -- what curriculum/envs.py's real training env
    now actually uses (fixed 2026-09-03; previously hardcoded 10.0 and
    never read this config value at all). Mirrors
    _real_phase1_max_episode_s()'s pattern so this script's own default can
    never silently drift from what real training uses again."""
    from footballcoach.ai.config import load_ai_config
    return float(load_ai_config().get("phase1_scenario", {}).get("ball_max_speed_mps", 10.0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-episodes", type=int, default=500)
    ap.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    ap.add_argument("--ball-max-speed-mps", type=float, default=_real_ball_max_speed_mps(),
                     help="Default: ai_config.json's phase1_scenario.ball_max_speed_mps -- "
                          "the same value curriculum/envs.py's real training env now uses.")
    ap.add_argument("--max-episode-s", type=float, default=None,
                     help="Episode timeout in sim-seconds. Default: the real phase-1 config "
                          "value (ai_config.json curriculum.phase1_max_episode_s, 18.5s) -- "
                          "NOT ScenarioEnv's own generic default (120s), which would silently "
                          "suppress almost all real timeouts.")
    ap.add_argument("--decision-interval-ticks", type=int, default=1,
                     help="Throttle the rules-AI trainee's order/sprint-flag decision to once "
                          "every N physics ticks (dt_s=0.06s -> N=2 is ~0.12s, N=4 is ~0.24s), "
                          "instead of every tick (default 1, unchanged). NOTE: per-tick steering "
                          "(the intercept solve inside order.execute()) still runs every tick "
                          "regardless -- see Phase1RulesAI's own docstring for why this only "
                          "throttles the higher-level order choice, not continuous movement.")
    ap.add_argument("--n-parallel-workers", type=int, default=8)
    ap.add_argument("--extra-seeds", type=str, default=",".join(str(s) for s in KNOWN_SEEDS),
                     help="Comma-separated seeds always folded into the sample, on top of the "
                          "--n-episodes random range (default: KNOWN_SEEDS, the diagnosed "
                          "worst-episode seeds). Pass '' to skip.")
    ap.add_argument("--output", type=str, default="results/outcome_baseline.json")
    args = ap.parse_args()

    from footballcoach.ai.eval.seeded_eval import run_seeded_evaluation, run_seeded_evaluation_parallel

    max_episode_s = args.max_episode_s if args.max_episode_s is not None else _real_phase1_max_episode_s()
    extra_seeds = [int(s) for s in args.extra_seeds.split(",") if s.strip()]
    seeds = list(range(args.seed_base, args.seed_base + args.n_episodes)) + extra_seeds
    import functools
    worker_factory = functools.partial(
        _env_worker_factory, args.ball_max_speed_mps, max_episode_s, args.decision_interval_ticks,
    )

    log.info(f"Running {len(seeds)} phase-1 episodes (rules trainee, ball_max_speed_mps="
              f"{args.ball_max_speed_mps}, max_episode_s={max_episode_s}, "
              f"decision_interval_ticks={args.decision_interval_ticks}, "
              f"seeds {seeds[0]}..{seeds[args.n_episodes - 1]} + {len(extra_seeds)} known)...")
    if args.n_parallel_workers > 1:
        result = run_seeded_evaluation_parallel(worker_factory, seeds, repeats_per_seed=1, n_workers=args.n_parallel_workers)
    else:
        env_factory, sample_fn = worker_factory()
        result = run_seeded_evaluation(env_factory, sample_fn, seeds, repeats_per_seed=1)

    n = result.n_episodes
    log.info("=" * 60)
    log.info(f"OUTCOME BASELINE  (n={n} episodes, deterministic)")
    log.info("=" * 60)
    for outcome, count in sorted(result.outcomes.items(), key=lambda kv: -kv[1]):
        log.info(f"  {outcome:<24} {count:>5}  ({100.0 * count / n:5.1f}%)")
    log.info("-" * 60)
    log.info(f"  mean_reward = {result.mean_reward:.3f} +/- {result.std_reward:.3f}")
    log.info("=" * 60)

    known_results: dict[int, dict] = {}
    if extra_seeds:
        log.info("KNOWN SEEDS (included above; echoed individually here for tracking)")
        log.info("-" * 60)
        seed_to_result = dict(zip(result.seeds, zip(result.outcomes_list, result.rewards)))
        for seed in extra_seeds:
            outcome, reward = seed_to_result[seed]
            note = KNOWN_SEEDS.get(seed, "")
            log.info(f"  seed={seed:<12} outcome={outcome:<12} reward={reward:+.3f}  {note}")
            known_results[seed] = {"outcome": outcome, "reward": reward, "note": note}
        log.info("=" * 60)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    d = result.as_dict()
    d["ball_max_speed_mps"] = args.ball_max_speed_mps
    d["max_episode_s"] = max_episode_s
    d["decision_interval_ticks"] = args.decision_interval_ticks
    d["known_seeds"] = known_results
    with open(out_path, "w") as f:
        json.dump(d, f, indent=2)
    log.info(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
