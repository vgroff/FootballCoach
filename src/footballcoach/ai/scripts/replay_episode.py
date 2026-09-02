"""Rebuild and replay ONE specific recorded phase-1 demonstration episode
from its stored seed (see record_demonstrations.py's ``meta_episode_seeds`` /
``DemonstrationDataset.episode_seed()``) -- exactly, not approximately: same
spawn positions/attributes/ball state and opponent-type roll, and every
subsequent physics RNG draw (tackle rolls, kick noise) the whole way through,
since ``build_1v1_scenario``'s returned ``Match`` keeps using that one seeded
``rng`` for everything it does afterward too.

Two ways to use it::

    # Headless: run it and save a real MatchLogger event log (an ACTUAL live
    # match, not dataset rows replayed through debug_value_network.py's
    # reconstruction -- richer and not subject to that reconstruction's own
    # "compact subset of rows" simplification).
    uv run python -m footballcoach.ai.scripts.replay_episode --seed 12345 \\
        --output results/replay_12345.json
    uv run python scripts/visualise_match_log.py results/replay_12345.json

    # Interactive: launch the UI with this exact episode loaded, paused --
    # press Space to watch it live.
    uv run python -m footballcoach.ai.scripts.replay_episode --seed 12345 --ui

Only reproduces the RAW MATCH (positions/physics/rules-AI behaviour) --
deliberately bypasses ScenarioEnv entirely (no observation encoding, no
reward computation, no BC labels), since watching what physically happened
is the whole point here, not re-deriving the RL-training-time view of it.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger("footballcoach.replay_episode")


def _real_sim_dt_s() -> float:
    """The REAL physics tick size real phase-1 recording actually runs at
    -- ai_config.json's observation.sim_dt_s (0.06s / ~16.67Hz by default,
    NOT the interactive UI's fixed 30Hz -- see that key's own config
    comment: "UI ignores this and always uses 30Hz"). This was a real,
    confirmed bug in every function below: each one used to hardcode
    dt_s=1/30 (the UI's rate), silently simulating a completely different
    physics timestep than the one the seed was actually recorded at.

    Confirmed via direct comparison: build_replay_match's constructed Match
    and a genuine ScenarioEnv.reset(seed=...) start with byte-IDENTICAL
    match.rng state and positions, but after even a single decision
    interval of stepping (dt_s=1/30 vs the real 0.06), positions had
    already diverged -- proving the RNG draws matched (same sequence, same
    count) and the divergence was purely from the wrong timestep, not any
    RNG/ordering difference. Over a multi-second episode this compounds
    into a completely different trajectory (confirmed: a real recorded
    episode with 87 real kicks replayed down to just 1-4 kicks under the
    wrong dt_s)."""
    from footballcoach.ai.config import load_ai_config

    return float(load_ai_config().get("observation", {}).get("sim_dt_s", 1.0 / 30.0))


def _phase1_opponent_probs() -> tuple[float, float]:
    """(opponent_rules_prob, opponent_immobile_prob), computed EXACTLY like
    ai/curriculum/envs.py's _build_phase1_env does -- the real recording
    path's own source of these two numbers. Reads the CURRENT ai_config.json,
    not whatever it was at record time -- if the curriculum ratios have
    changed since, replay's second opponent-type roll (see build_replay_
    match's docstring) will diverge from what was actually recorded. Every
    other part of the scenario (positions, attributes, ball state,
    build_1v1_scenario's own internal roll) is unaffected by this drift,
    since those are fully determined by `seed` alone."""
    from footballcoach.ai.config import load_ai_config

    cfg = load_ai_config().get("curriculum", {})
    rules_ratio = float(cfg.get("phase1_opponent_rules_ratio", 0.0))
    immobile_ratio = float(cfg.get("phase1_opponent_immobile_ratio", 1.0))
    neural_ratio = float(cfg.get("phase1_opponent_neural_ratio", 0.0))
    total = rules_ratio + immobile_ratio + neural_ratio
    rules_prob = (rules_ratio / total) if total > 0 else 0.0
    immobile_prob = (immobile_ratio / total) if total > 0 else 1.0
    return rules_prob, immobile_prob


def apply_phase1_opponent_roll(match, opponent, seed: int) -> None:
    """Apply the SECOND, OVERRIDING phase-1 opponent-type roll to an
    already-constructed Match + opponent Player, regardless of which
    construction path built them (``build_1v1_scenario`` via
    ``build_replay_match()``/``ScenarioLoop``, or ``ScenarioEnv.reset()``
    via a fresh phase-1 env) -- call this AFTER construction. Mutates
    ``opponent.ai``, ``match._opponent_use_rules_ai``,
    ``match._opponent_is_immobile`` in place.

    See ``build_replay_match()``'s docstring for WHY this second roll
    exists (record_demonstrations.py never trusts build_1v1_scenario's own
    internal seed-determined roll). Single source of truth for this logic --
    previously duplicated inline in ``build_replay_match()``, ``run_headless()``
    (which was missing the ``_opponent_use_rules_ai``/``_opponent_is_immobile``
    assignments -- fixed by routing through here), and ``debug_policy_net.py``.
    """
    import random as _random

    from footballcoach.rules_ai import Phase1RulesAI

    opponent_rules_prob, opponent_immobile_prob = _phase1_opponent_probs()
    roll = _random.Random(seed).random()
    if roll < opponent_rules_prob:
        opponent.ai = Phase1RulesAI()
        match._opponent_use_rules_ai, match._opponent_is_immobile = True, False
    elif opponent_immobile_prob is not None and roll >= opponent_rules_prob + opponent_immobile_prob:
        opponent.ai = Phase1RulesAI()
        match._opponent_use_rules_ai, match._opponent_is_immobile = True, False
    else:
        opponent.ai = None
        match._opponent_use_rules_ai, match._opponent_is_immobile = False, True


def build_replay_match(seed: int, *, rng_reduction: float = 0.3):
    """Rebuild the EXACT ``Match`` a phase-1 demonstration episode with this
    seed would have been recorded from, including AI assignment
    (``Phase1RulesAI`` on the trainee always; the opponent per a SECOND,
    OVERRIDING roll -- see below).

    ``rng_reduction`` default (0.3) matches ``ScenarioEnv``'s own dataclass
    default, which real phase-1 recording never overrides (no
    ``--rng-reduction`` CLI flag exists in record_demonstrations.py).

    Why a second opponent-type roll: record_demonstrations.py's own
    recording loop does NOT rely on ``build_1v1_scenario``'s internal
    opponent-type roll (which already fully determines it from ``seed``
    alone, per that function's own docstring) -- it unconditionally
    OVERRIDES it right after, with a second roll seeded from THIS episode's
    seed via a fresh ``random.Random(seed)`` (see that file's own long
    comment on this for the reasoning: it deliberately folds the
    "would-be-neural" probability region into "rules" instead, since demo
    recording never uses a neural opponent). This function mirrors that
    exact second roll -- formula, region boundaries, and all -- so replay's
    opponent type matches what was actually recorded rather than
    build_1v1_scenario's own (overridden, and therefore never actually
    observed during real recording) internal choice.
    """
    from footballcoach.rules_ai import Phase1RulesAI
    from footballcoach.ui.scenarios import build_1v1_scenario

    opponent_rules_prob, opponent_immobile_prob = _phase1_opponent_probs()

    match = build_1v1_scenario(
        rng_reduction, seed=seed, ball_max_speed_mps=10.0,
        opponent_rules_prob=opponent_rules_prob,
        opponent_immobile_prob=opponent_immobile_prob,
        sim_dt_s=_real_sim_dt_s(),
    )

    trainee = match.player_by_id("trainee")
    trainee.ai = Phase1RulesAI()

    opponent = match.player_by_id("opponent")
    apply_phase1_opponent_roll(match, opponent, seed)

    return match


def run_headless(seed: int, output: str, *, rng_reduction: float = 0.3) -> None:
    """Run the replayed match to completion (box possession / ball out /
    timeout / invalid), logging every real engine event via ``MatchLogger``,
    and save it to *output* -- ready for ``scripts/visualise_match_log.py``.

    Termination is ``ScenarioLoop``'s own native ``detect_trial_outcome``
    check (default ``terminal_outcomes=None`` -- every raw outcome ends the
    trial), NOT ``ScenarioEnv``'s wrapped version (which suppresses that in
    favour of its own per-tick box-possession/timeout check, run here
    deliberately bypassed -- see module docstring). Timing can differ
    slightly from the original recording as a result (a real, pre-existing
    structural gap between the two termination paths, not specific to this
    tool -- up to roughly one decision interval's worth of overshoot).

    No reward is computed or logged (ScenarioEnv's job, not reproduced
    here) -- this tool is for watching what physically happened, not
    re-deriving training-time reward.
    """
    from footballcoach.engine.match_logger import MatchLogger
    from footballcoach.ai.config import load_ai_config
    from footballcoach.ui.scenarios import ScenarioDefinition, ScenarioLoop, phase1_training_on_tick

    opponent_rules_prob, opponent_immobile_prob = _phase1_opponent_probs()

    import functools
    from footballcoach.ui.scenarios import build_1v1_scenario
    dt_s = _real_sim_dt_s()
    defn = ScenarioDefinition(
        key="phase1_1v1_replay",
        label="Phase 1 replay",
        description="Standalone replay of one recorded episode by seed.",
        build=functools.partial(
            build_1v1_scenario, ball_max_speed_mps=10.0,
            opponent_rules_prob=opponent_rules_prob,
            opponent_immobile_prob=opponent_immobile_prob,
            sim_dt_s=dt_s,
        ),
        on_tick=phase1_training_on_tick,
    )
    max_episode_s = float(load_ai_config().get("curriculum", {}).get("phase1_max_episode_s", 18.5))
    max_ticks = max(1, int(max_episode_s / dt_s))

    loop = ScenarioLoop(
        definition=defn, max_trials=1, rng_reduction=rng_reduction,
        kwargs={"seed": seed}, timeout_ticks=max_ticks,
    )
    # dt_s is already correctly set via sim_dt_s above (build_1v1_scenario
    # applies it to the constructed Match) -- no need to override it again
    # here; a prior version of this line hardcoded dt_s=1/30 AFTER
    # construction, silently overwriting whatever the real config-derived
    # value would otherwise have been.
    # Same AI assignment / overriding second roll as build_replay_match(),
    # via the shared apply_phase1_opponent_roll() -- ScenarioLoop must own
    # construction here so its native trial-outcome detection applies, so
    # this can't just call build_replay_match() directly, but the roll
    # logic itself is no longer duplicated.
    from footballcoach.rules_ai import Phase1RulesAI
    loop.match.player_by_id("trainee").ai = Phase1RulesAI()
    opponent = loop.match.player_by_id("opponent")
    apply_phase1_opponent_roll(loop.match, opponent, seed)

    logger = MatchLogger()
    loop.match.match_logger = logger
    logger.record_start(loop.match)

    outcomes_before = dict(loop.outcomes)
    done = False
    for _tick in range(max_ticks):
        done = loop.step()
        if done:
            break
    if not done:
        log.warning(f"Replay did not terminate within {max_ticks} ticks ({max_episode_s}s) -- saving as-is.")

    resolved_outcome = "unknown"
    for k, v in loop.outcomes.items():
        if v > outcomes_before.get(k, 0):
            resolved_outcome = k
            break
    logger.notify_episode_end(
        loop.match.time_s, loop.match.ball.position, resolved_outcome,
        reward_total=0.0, components={},
    )
    log.info(f"Episode ended at t={loop.match.time_s:.2f}s: {resolved_outcome}")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.save(out_path)
    log.info(f"Saved to {out_path}")


def run_ui(seed: int, *, rng_reduction: float = 0.3) -> None:
    """Launch the interactive UI with this exact episode pre-loaded,
    reusing ``App._start_scenario`` (the SAME machinery the menu's own
    "1v1: Phase 1 get possession" entry uses) rather than a bespoke launch
    path -- just pre-supplies ``seed``/the curriculum-matched opponent
    probabilities as kwargs instead of going through the menu."""
    from footballcoach.ui.app import App
    from footballcoach.ui.scenarios import SCENARIOS

    opponent_rules_prob, opponent_immobile_prob = _phase1_opponent_probs()
    defn = next(d for d in SCENARIOS if d.key == "1v1_phase1")

    app = App()
    app._start_scenario(
        defn,
        kwargs={
            "seed": seed,
            "ball_max_speed_mps": 10.0,
            "opponent_rules_prob": opponent_rules_prob,
            "opponent_immobile_prob": opponent_immobile_prob,
            # Real recording runs at ai_config.json's observation.sim_dt_s
            # (0.06s), NOT the UI's normal fixed 30Hz (see _real_sim_dt_s's
            # own docstring for the confirmed bug this fixes) -- without
            # this the replayed match plays out completely different
            # physics than what was actually recorded, even at the exact
            # same seed. Trade-off: the match now advances more simulated
            # time per engine tick than a normal 30Hz UI scenario, so it
            # will visibly run faster than real-time -- a real, accepted
            # cost of correctly reproducing the recorded episode instead of
            # a cosmetically smoother but WRONG one.
            "sim_dt_s": _real_sim_dt_s(),
        },
    )
    # Trainee needs Phase1RulesAI explicitly (build_1v1_scenario doesn't
    # assign it -- see build_replay_match()'s own docstring); the opponent's
    # AI is already whatever build_1v1_scenario's OWN internal roll decided
    # (unlike run_headless()/build_replay_match(), the UI path does NOT
    # apply record_demonstrations.py's second overriding roll -- there is no
    # recorded episode to match fidelity against when launching fresh from
    # the menu's own scenario kwargs, so this only matters if you're
    # cross-checking a specific already-recorded seed's opponent type,
    # which will need a manual look at meta_episode_outcomes for now).
    from footballcoach.rules_ai import Phase1RulesAI
    app.match.player_by_id("trainee").ai = Phase1RulesAI()
    app.match.paused = True
    app.run()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True, help="Episode seed (see meta_episode_seeds / DemonstrationDataset.episode_seed()).")
    parser.add_argument("--rng-reduction", type=float, default=0.3, help="Physics randomness level (default: 0.3, matching real recording).")
    parser.add_argument("--output", type=str, default="results/replay_episode.json", help="Where to save the headless match log (ignored with --ui).")
    parser.add_argument("--ui", action="store_true", help="Launch the interactive UI with this episode loaded instead of running headless.")
    args = parser.parse_args()

    if args.ui:
        run_ui(args.seed, rng_reduction=args.rng_reduction)
    else:
        run_headless(args.seed, args.output, rng_reduction=args.rng_reduction)


if __name__ == "__main__":
    main()
