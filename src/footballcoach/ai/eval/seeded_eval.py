"""Shared seeded evaluation helper for PPO periodic eval and the standalone
evaluate.py CLI (previously duplicated logic -- see ai_trainer_knowledge.md
"Evaluation" note).

Runs a FIXED list of scenario seeds (not fresh unseeded draws every call),
each repeated ``repeats_per_seed`` times, and reports aggregate reward/
outcome stats. This means pre-training eval and every subsequent rollout's
eval (and any standalone evaluate.py run) see the exact same set of N
scenarios -- comparable numbers across an entire run instead of noise from a
different random scenario draw every time. PPO's own training rollouts are
NOT seeded by this module -- only evaluation call sites use it.

Not scenario-specific: ``env_factory(seed) -> ScenarioEnv`` is supplied by
the caller (see PPOTrainer._eval_vs_rules / evaluate.py for the current
factories: rules-opponent eval and rules-vs-rules baseline). Reuse for any
future phase/AI-type by writing a new ``env_factory``.
"""
from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from footballcoach.ai.env.scenario_env import ScenarioEnv

log = logging.getLogger("footballcoach.ai.eval.seeded_eval")

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


@contextlib.contextmanager
def _capped_thread_env_for_pool_spawn(threads: str = "1"):
    """Temporarily cap BLAS/OMP thread-count env vars in THIS process
    before constructing a fresh ``multiprocessing.Pool`` (which spawns all
    its worker processes immediately, at construction time) for the
    throwaway seeded-eval pools below.

    Every OTHER worker-spawn point in this codebase (ai/ppo/rollout_worker.py,
    ai/ppo/batched_rollout_worker.py, ai/eval/eval_worker.py) sets these same
    env vars INSIDE the freshly-spawned child's own entry function, before
    that function's own ``import torch``/``import numpy`` -- see
    rollout_worker.py's ``_worker_main`` docstring for why this matters
    (OpenBLAS/MKL read these vars once, at first use, to size their OWN
    native thread pool -- entirely independent of, and not shrunk by, a
    later ``torch.set_num_threads()`` call). That per-child-body pattern
    does NOT work here: THIS module imports numpy at module level (needed
    to resolve/unpickle ``_eval_worker_entry``/``_eval_worker_entry_batched``
    by name in the child), so numpy's own BLAS thread pool would already be
    sized off the ambient (uncapped) environment by the time either entry
    function's body ran, no matter what it set internally. Setting the env
    vars HERE instead, in the parent, works regardless of import order --
    a spawned child inherits the parent's ``os.environ`` as its OS-level
    process environment from the moment it's created, before any of its own
    Python code (including module-level imports) runs.

    Left uncapped, each throwaway worker process defaults to one BLAS
    thread per logical core; with eval.eval_n_parallel_workers processes
    spawned on top of the training run's already-running persistent
    rollout/eval worker pools, this is real, confirmed oversubscription
    that can exhaust process/thread memory outright (observed in production
    as ``OMP: Error #111: Memory allocation failed.`` killing the whole
    training run), not just a slowdown.

    Restores the previous values afterward -- this only needs to affect
    processes spawned during the ``with`` block, not the long-lived caller.
    """
    _prev = {v: os.environ.get(v) for v in _THREAD_ENV_VARS}
    for v in _THREAD_ENV_VARS:
        os.environ[v] = threads
    try:
        yield
    finally:
        for v, val in _prev.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val


def swap_player_states(match, id_a: str, id_b: str) -> None:
    """Swap two players' full physical/skill state (team, attributes,
    position, velocity, heading, stamina) while leaving player_id -- and
    therefore whichever AI/network gets wired to that id afterward -- fully
    untouched. Used for eval "swap_sides" fairness (ai_config.json
    eval.swap_sides): call once, right after build_1v1_scenario() returns,
    before any .ai is assigned.

    Rationale: a 1v1 scenario's random draw (positions, attributes, team/
    attacking-direction, stamina) can hand one role a spatial or skill
    advantage the other role didn't get. Running a seed once normally and
    once with player states swapped means 'trainee' (always driven by
    whichever network is under test, since only the STATE moves, not the
    id) experiences BOTH roles that seed's draw produced, canceling out any
    per-role bias when the two passes are averaged together -- a true role
    exchange, not a geometric mirror (the ball's position is untouched:
    handing 'trainee' the other role's spatial relationship to the ball,
    including its original distance/angle, is exactly the point).

    Silently a no-op if either id is missing (mirrors other eval build
    functions' tolerance for scenarios without both ids).
    """
    try:
        a = match.player_by_id(id_a)
        b = match.player_by_id(id_b)
    except KeyError:
        return
    a.team, b.team = b.team, a.team
    a.attributes, b.attributes = b.attributes, a.attributes
    a.position, b.position = b.position, a.position
    a.velocity, b.velocity = b.velocity, a.velocity
    a.heading_rad, b.heading_rad = b.heading_rad, a.heading_rad
    a.stamina, b.stamina = b.stamina, a.stamina


def default_eval_seeds(cfg: dict) -> list[int]:
    """Fixed seed list from ai_config.json['eval']. Shared by PPOTrainer and
    evaluate.py so both use IDENTICAL seeds -- the whole point of this
    module is that pre-training eval, every rollout's eval, and any
    standalone evaluate.py run see the same N scenarios."""
    eval_cfg = cfg.get("eval", {})
    n_seeds = int(eval_cfg.get("eval_n_seeds", 30))
    seed_base = int(eval_cfg.get("eval_seed_base", 1_000_000))
    return list(range(seed_base, seed_base + n_seeds))


@dataclass
class SeededEvalResult:
    n_episodes: int
    seeds: list[int]
    repeats_per_seed: int
    mean_reward: float
    std_reward: float
    win_rate_pct: float
    outcomes: dict = field(default_factory=dict)
    outcome_breakdown: str = ""
    mean_value_pred: float = float("nan")
    rewards: list = field(default_factory=list)
    outcomes_list: list = field(default_factory=list)
    # Per-episode wall-clock duration (seconds), same index order as
    # rewards/outcomes_list -- info.ticks_elapsed * env._dt_s, matching
    # ppo_trainer.py's episode_durations_s ("ep_len" log line) computation.
    episode_lengths_s: list = field(default_factory=list)
    # Per-episode reward-component breakdown (one dict per episode, same
    # index order as rewards/outcomes_list/episode_lengths_s) -- unlike
    # reward_component_sums below (pooled across ALL episodes), this lets a
    # caller inspect e.g. "what made up the total reward for episode i"
    # (see evaluate.py's noise-floor probe: explaining why a specific
    # outlier episode's reward came out the way it did).
    episode_reward_components: list = field(default_factory=list)
    value_preds: list = field(default_factory=list)
    # Per-step (V, discounted-return-to-episode-end) pairs, pooled across all
    # episodes -- V-vs-R diagnostic (see mean_step_v/mean_step_r below).
    step_v: list = field(default_factory=list)
    step_r: list = field(default_factory=list)
    # Sum of each named reward component across all episodes; divide by
    # n_episodes for the per-episode mean (see reward_components property).
    reward_component_sums: dict = field(default_factory=dict)

    @property
    def reward_components(self) -> dict:
        n = self.n_episodes or 1
        return {k: v / n for k, v in self.reward_component_sums.items()}

    @property
    def mean_step_v(self) -> float:
        return float(np.mean(self.step_v)) if self.step_v else float("nan")

    @property
    def mean_step_r(self) -> float:
        return float(np.mean(self.step_r)) if self.step_r else float("nan")

    @property
    def sem_reward(self) -> float:
        """Standard error of mean_reward across the n_episodes trials (std_reward / sqrt(n))."""
        n = self.n_episodes
        return float(self.std_reward / np.sqrt(n)) if n > 0 else float("nan")

    def as_dict(self) -> dict:
        return {
            "n_trials": self.n_episodes,
            "seeds": self.seeds,
            "repeats_per_seed": self.repeats_per_seed,
            "win_rate_pct": self.win_rate_pct,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "sem_reward": self.sem_reward,
            "min_reward": float(min(self.rewards)) if self.rewards else float("nan"),
            "max_reward": float(max(self.rewards)) if self.rewards else float("nan"),
            "outcomes": self.outcomes,
            "outcome_breakdown": self.outcome_breakdown,
            "mean_value_pred": self.mean_value_pred,
            "mean_step_v": self.mean_step_v,
            "mean_step_r": self.mean_step_r,
            "reward_components": self.reward_components,
            # Per-episode detail, all in the same index order -- lets a
            # caller correlate "this episode's reward/outcome/duration/
            # component breakdown" rather than only seeing pooled aggregates.
            "rewards": self.rewards,
            "outcomes_list": self.outcomes_list,
            "episode_lengths_s": self.episode_lengths_s,
            "episode_reward_components": self.episode_reward_components,
        }


def run_seeded_evaluation(
    env_factory: Callable[[int, bool], ScenarioEnv],
    sample_action_fn,
    seeds: list[int],
    repeats_per_seed: int = 2,
    win_outcome: str = "box_possession",
    trial_log_every: int = 0,
    gamma: float = 0.98,
    swap_sides: bool = False,
) -> SeededEvalResult:
    """Run ``len(seeds) * repeats_per_seed`` (x2 if ``swap_sides``) episodes,
    return aggregate stats.

    Args:
        env_factory: ``(seed, swap) -> freshly-built ScenarioEnv``. ``swap``
            is True for the "swapped" pass when ``swap_sides=True`` (False
            otherwise) -- factories that support it call
            ``swap_player_states(match, "trainee", "opponent")`` right after
            building the match when ``swap`` is True; factories that don't
            care about side-fairness can just ignore the second argument.
            Repeating the same seed still gives different episodes since
            match physics/action sampling keep their own residual
            randomness -- only the initial scenario setup (positions/
            attributes/ball state/opponent-type roll) is pinned by the seed.
        sample_action_fn: forwarded to env.sample_action_fn (e.g.
            trainer._sample_action).
        seeds: fixed list of scenario seeds (see default_eval_seeds()).
        repeats_per_seed: episodes run per seed (per side, if swap_sides);
            all repeats are pooled into the same aggregate stats as every
            other episode (no separate per-seed breakdown -- not needed for
            the current use case). Deliberately independent of swap_sides --
            it never changes what repeats_per_seed means, only whether each
            seed is also run a second time with roles swapped.
        gamma: discount factor used for the per-step V-vs-R diagnostic
            (mean_step_v/mean_step_r on the result) -- no bootstrap at
            episode end, matching the prior evaluate.py behaviour.
        swap_sides: when True, run each seed once with the scenario's
            natural (unswapped) player roles AND once with
            swap_player_states() applied (see that function's docstring) --
            doubles total episode count. See ai_config.json's
            eval.swap_sides.
    """
    from footballcoach.ai.ppo.ppo_trainer import outcome_breakdown  # local: avoid import cycle

    rewards: list[float] = []
    outcomes: dict[str, int] = {}
    outcomes_list: list[str] = []
    episode_lengths_s: list[float] = []
    episode_reward_components: list[dict] = []
    value_preds: list[float] = []
    step_v: list[float] = []
    step_r: list[float] = []
    reward_component_sums: dict[str, float] = {}

    n = 0
    _sides = (False, True) if swap_sides else (False,)
    total = len(seeds) * max(1, repeats_per_seed) * len(_sides)
    for seed in seeds:
        for do_swap in _sides:
            for _ in range(max(1, repeats_per_seed)):
                env = env_factory(seed, do_swap)
                env.sample_action_fn = sample_action_fn
                env.reset()
                done = False
                info = None
                ep_reward = 0.0
                ep_step_v: list[float] = []
                ep_step_rew: list[float] = []
                ep_component_sums: dict[str, float] = {}
                while not done:
                    _, reward, done, info = env.step()
                    ep_reward += reward
                    for _k, _v in getattr(env, "last_reward_components", {}).items():
                        # Cast to plain float -- numpy/torch scalar types here make
                        # the running sum (and therefore reward_components/as_dict())
                        # non-JSON-serializable downstream in evaluate.py.
                        _v = float(_v.item()) if hasattr(_v, "item") else float(_v)
                        reward_component_sums[_k] = reward_component_sums.get(_k, 0.0) + _v
                        ep_component_sums[_k] = ep_component_sums.get(_k, 0.0) + _v
                    tr = getattr(env, "last_trainee_transition", None)
                    if tr is not None and "value" in tr:
                        _v = tr["value"]
                        _v = float(_v.item()) if hasattr(_v, "item") else float(_v)
                        value_preds.append(_v)
                        ep_step_v.append(_v)
                        ep_step_rew.append(reward)
                # Discounted return-to-go for each step in this episode (no
                # bootstrap at the end) -- V-vs-R diagnostic.
                G = 0.0
                for i in reversed(range(len(ep_step_rew))):
                    G = ep_step_rew[i] + gamma * G
                    step_v.append(ep_step_v[i])
                    step_r.append(G)
                rewards.append(ep_reward)
                outcome = info.trial_outcome if info else "unknown"
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
                outcomes_list.append(outcome)
                # Matches ppo_trainer.py's episode_durations_s ("ep_len" log line)
                # computation exactly: ticks_elapsed * the env's own physics dt.
                episode_lengths_s.append(
                    float(info.ticks_elapsed) * env._dt_s if info is not None else float("nan")
                )
                episode_reward_components.append(ep_component_sums)
                n += 1
                if trial_log_every and n % trial_log_every == 0:
                    log.info(f"  [seeded eval] trial {n}/{total}: seed={seed} "
                             f"outcome={outcome} reward={ep_reward:.2f}")

    win_count = outcomes.get(win_outcome, 0)
    return SeededEvalResult(
        n_episodes=n,
        seeds=list(seeds),
        repeats_per_seed=repeats_per_seed,
        mean_reward=float(np.mean(rewards)) if rewards else float("nan"),
        std_reward=float(np.std(rewards)) if rewards else float("nan"),
        win_rate_pct=100.0 * win_count / n if n else float("nan"),
        outcomes=outcomes,
        outcome_breakdown=outcome_breakdown(outcomes_list),
        mean_value_pred=float(np.mean(value_preds)) if value_preds else float("nan"),
        rewards=rewards,
        outcomes_list=outcomes_list,
        episode_lengths_s=episode_lengths_s,
        episode_reward_components=episode_reward_components,
        value_preds=value_preds,
        step_v=step_v,
        step_r=step_r,
        reward_component_sums=reward_component_sums,
    )


def run_seeded_evaluation_batched(
    env_factory: Callable[[int, bool], ScenarioEnv],
    trainer,
    seeds: list[int],
    repeats_per_seed: int = 2,
    win_outcome: str = "box_possession",
    gamma: float = 0.98,
    swap_sides: bool = False,
    envs_per_process: int = 8,
    secondary_trainer=None,
    deterministic: bool = False,
    deterministic_decision: bool = False,
    deterministic_direction: bool = False,
) -> SeededEvalResult:
    """Batched variant of ``run_seeded_evaluation()``: instead of stepping one
    ``ScenarioEnv`` at a time (batch-of-1 network calls), runs up to
    ``envs_per_process`` envs TOGETHER, batching every round's trainee
    decisions (and, when ``secondary_trainer`` is given, every due secondary/
    opponent decision too) into one ``_sample_action_batch()`` call each --
    mirrors ``ai/ppo/batched_rollout_worker.py``'s real-rollout-collection
    speedup, applied to evaluation instead. Same total episode count and
    identical AGGREGATE stats as ``run_seeded_evaluation`` (mean/std reward,
    win rate, pooled outcome counts) -- this is purely a speed optimization
    over the unbatched path, not a different evaluation.

    NOTE on ordering: the per-episode lists on the returned
    ``SeededEvalResult`` (``rewards``/``outcomes_list``/``episode_lengths_s``/
    ``episode_reward_components``) are appended in COMPLETION order, not
    seed/task-submission order -- envs with different episode lengths
    finish out of order under concurrent batched stepping, unlike
    ``run_seeded_evaluation``'s strictly sequential one-task-at-a-time loop.
    The four lists stay mutually consistent with each other (index i is
    always the same episode across all four), just not with ``seeds``'
    original order -- see test_batched_seeded_eval.py for why this is
    correct, not a bug.

    Args:
        env_factory: same ``(seed, swap) -> freshly-built ScenarioEnv``
            convention as ``run_seeded_evaluation``. Each built env gets
            ``env.sample_action_fn = trainer._sample_action`` set here
            (purely so ``ScenarioEnv.reset()`` wires up a ``NeuralPlayerAI``
            at all -- see ``BatchedEnvGroup``'s own docstring for the same
            requirement); the actual sampling always goes through
            ``trainer._sample_action_batch()`` below, this fallback is never
            actually invoked.
        trainer: the actual ``PPOTrainer`` (not just its bound
            ``sample_action_fn``) -- needed for ``_sample_action_batch()``.
        envs_per_process: how many episodes to run concurrently, batched
            together, in this call. As one episode finishes, its slot is
            immediately refilled from the remaining seed/swap/repeat task
            queue (if any remain) -- keeps every slot busy until the whole
            task list is drained, rather than waiting for the slowest of a
            fixed batch each round.
        secondary_trainer: same convention as ``BatchedEnvGroup`` -- ``None``
            (default) leaves secondary/opponent players unbatched (fine when
            they're rules-based/immobile, or simply not present); pass the
            opponent snapshot's trainer for neural-vs-neural eval to batch
            its decisions too.
        deterministic/deterministic_decision/deterministic_direction:
            forwarded to every ``_sample_action_batch`` call (trainee AND
            secondary, when applicable) -- real eval calls leave these at
            the default (``False``, matching ``run_seeded_evaluation``'s own
            always-stochastic usage via a plain, non-deterministic
            ``sample_action_fn``); set them for a fully reproducible,
            RNG-free comparison (e.g. tests checking this against
            ``run_seeded_evaluation``'s output on the same seeds).
    """
    from footballcoach.ai.ppo.ppo_trainer import outcome_breakdown  # local: avoid import cycle
    from footballcoach.ai.ppo.batched_rollout_worker import _batch_due_decisions, _secondary_neural_candidates

    _sides = (False, True) if swap_sides else (False,)
    tasks = [
        (seed, do_swap)
        for seed in seeds
        for do_swap in _sides
        for _ in range(max(1, repeats_per_seed))
    ]

    rewards: list[float] = []
    outcomes: dict[str, int] = {}
    outcomes_list: list[str] = []
    episode_lengths_s: list[float] = []
    episode_reward_components: list[dict] = []
    value_preds: list[float] = []
    step_v: list[float] = []
    step_r: list[float] = []
    reward_component_sums: dict[str, float] = {}
    n = 0

    if not tasks:
        return SeededEvalResult(
            n_episodes=0, seeds=list(seeds), repeats_per_seed=repeats_per_seed,
            mean_reward=float("nan"), std_reward=float("nan"), win_rate_pct=float("nan"),
        )

    task_iter = iter(tasks)

    def _new_slot():
        seed, do_swap = next(task_iter)  # raises StopIteration when exhausted
        env = env_factory(seed, do_swap)
        env.sample_action_fn = trainer._sample_action
        env.reset()
        return env, {"reward": 0.0, "step_v": [], "step_rew": [], "comps": {}}

    def _trainee_player_and_ai(env):
        player = env.match.player_by_id(env.trainee_player_id)
        return player, player.ai

    n_slots = min(max(1, envs_per_process), len(tasks))
    slots = [_new_slot() for _ in range(n_slots)]  # list of (env, accum)

    while slots:
        trainee_candidates = [(*_trainee_player_and_ai(env), env.match) for env, _ in slots]
        _batch_due_decisions(
            trainee_candidates, trainer, deterministic, deterministic_decision, deterministic_direction,
        )

        if secondary_trainer is not None:
            secondary_candidates = [
                triple for env, _ in slots for triple in _secondary_neural_candidates(env)
            ]
            _batch_due_decisions(
                secondary_candidates, secondary_trainer,
                deterministic, deterministic_decision, deterministic_direction,
            )

        next_slots = []
        for env, acc in slots:
            _, reward, done, info = env.step()
            acc["reward"] += reward
            for _k, _v in getattr(env, "last_reward_components", {}).items():
                _v = float(_v.item()) if hasattr(_v, "item") else float(_v)
                reward_component_sums[_k] = reward_component_sums.get(_k, 0.0) + _v
                acc["comps"][_k] = acc["comps"].get(_k, 0.0) + _v
            tr = getattr(env, "last_trainee_transition", None)
            if tr is not None and "value" in tr:
                _v = tr["value"]
                _v = float(_v.item()) if hasattr(_v, "item") else float(_v)
                value_preds.append(_v)
                acc["step_v"].append(_v)
                acc["step_rew"].append(reward)

            if not done:
                next_slots.append((env, acc))
                continue

            G = 0.0
            for i in reversed(range(len(acc["step_rew"]))):
                G = acc["step_rew"][i] + gamma * G
                step_v.append(acc["step_v"][i])
                step_r.append(G)
            rewards.append(acc["reward"])
            outcome = info.trial_outcome if info else "unknown"
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            outcomes_list.append(outcome)
            episode_lengths_s.append(
                float(info.ticks_elapsed) * env._dt_s if info is not None else float("nan")
            )
            episode_reward_components.append(acc["comps"])
            n += 1
            try:
                next_slots.append(_new_slot())
            except StopIteration:
                pass  # no more tasks -- this slot drops out, group shrinks
        slots = next_slots

    win_count = outcomes.get(win_outcome, 0)
    return SeededEvalResult(
        n_episodes=n,
        seeds=list(seeds),
        repeats_per_seed=repeats_per_seed,
        mean_reward=float(np.mean(rewards)) if rewards else float("nan"),
        std_reward=float(np.std(rewards)) if rewards else float("nan"),
        win_rate_pct=100.0 * win_count / n if n else float("nan"),
        outcomes=outcomes,
        outcome_breakdown=outcome_breakdown(outcomes_list),
        mean_value_pred=float(np.mean(value_preds)) if value_preds else float("nan"),
        rewards=rewards,
        outcomes_list=outcomes_list,
        episode_lengths_s=episode_lengths_s,
        episode_reward_components=episode_reward_components,
        value_preds=value_preds,
        step_v=step_v,
        step_r=step_r,
        reward_component_sums=reward_component_sums,
    )


def _eval_worker_entry_batched(
    worker_factory, seed_chunk, repeats_per_seed, win_outcome, swap_sides, envs_per_process,
) -> SeededEvalResult:
    """Module-level (picklable) subprocess entry point for the batched eval
    path -- mirrors ``_eval_worker_entry`` above, except ``worker_factory()``
    returns ``(env_factory, trainer, secondary_trainer)`` (the trainer
    OBJECTS, not just a bound sample_action_fn, since batched sampling needs
    ``_sample_action_batch``)."""
    import torch
    torch.set_num_threads(1)
    env_factory, trainer, secondary_trainer = worker_factory()
    return run_seeded_evaluation_batched(
        env_factory, trainer, seed_chunk, repeats_per_seed, win_outcome,
        swap_sides=swap_sides, envs_per_process=envs_per_process, secondary_trainer=secondary_trainer,
    )


def run_seeded_evaluation_parallel_batched(
    worker_factory: Callable[[], tuple],
    seeds: list[int],
    repeats_per_seed: int = 2,
    n_workers: int = 1,
    win_outcome: str = "box_possession",
    swap_sides: bool = False,
    envs_per_process: int = 8,
    pool=None,
) -> SeededEvalResult:
    """Batched variant of ``run_seeded_evaluation_parallel()``: same seed-
    chunking across ``n_workers`` processes, but each worker batches its own
    chunk's episodes across up to ``envs_per_process`` envs (see
    ``run_seeded_evaluation_batched``) instead of running them one at a
    time. ``worker_factory`` must return ``(env_factory, trainer,
    secondary_trainer)`` -- see ``_eval_worker_factory_batched``/
    ``_eval_worker_factory_neural_snapshot_batched`` in ppo_trainer.py.

    ``pool``: an already-created ``multiprocessing.Pool`` (spawn context) to
    dispatch onto instead of creating (and closing) a fresh one -- pass this
    when a caller needs MULTIPLE sequential calls with the same
    ``n_workers`` (e.g. comparing against several snapshots in a row) so
    those calls share one set of already-warm worker processes instead of
    spawning-and-tearing-down a whole new process pool every single call.
    The caller owns the pool's lifecycle (close/join it themselves) when
    passing one in -- this function never closes a pool it didn't create.
    Ignored when ``n_workers <= 1`` (no pool needed at all in that case).
    """
    if n_workers <= 1:
        env_factory, trainer, secondary_trainer = worker_factory()
        return run_seeded_evaluation_batched(
            env_factory, trainer, seeds, repeats_per_seed, win_outcome,
            swap_sides=swap_sides, envs_per_process=envs_per_process, secondary_trainer=secondary_trainer,
        )

    chunks = [c for c in (seeds[i::n_workers] for i in range(n_workers)) if c]
    _args = [
        (worker_factory, chunk, repeats_per_seed, win_outcome, swap_sides, envs_per_process)
        for chunk in chunks
    ]
    if pool is not None:
        results = pool.starmap(_eval_worker_entry_batched, _args)
        return merge_eval_results(results, repeats_per_seed)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    log.info(f"  [seeded eval batched] running {len(seeds)}x{repeats_per_seed} episodes "
             f"across {len(chunks)} worker process(es), {envs_per_process} envs/process...")
    with _capped_thread_env_for_pool_spawn("1"), ctx.Pool(processes=len(chunks)) as _pool:
        results = _pool.starmap(_eval_worker_entry_batched, _args)
    log.info("  [seeded eval batched] all workers finished, merging results.")
    return merge_eval_results(results, repeats_per_seed)


def merge_eval_results(results: list[SeededEvalResult], repeats_per_seed: int) -> SeededEvalResult:
    from footballcoach.ai.ppo.ppo_trainer import outcome_breakdown

    rewards: list[float] = []
    outcomes_list: list[str] = []
    episode_lengths_s: list[float] = []
    episode_reward_components: list[dict] = []
    value_preds: list[float] = []
    seeds: list[int] = []
    outcomes: dict[str, int] = {}
    step_v: list[float] = []
    step_r: list[float] = []
    reward_component_sums: dict[str, float] = {}
    for r in results:
        rewards.extend(r.rewards)
        outcomes_list.extend(r.outcomes_list)
        episode_lengths_s.extend(r.episode_lengths_s)
        episode_reward_components.extend(r.episode_reward_components)
        value_preds.extend(r.value_preds)
        seeds.extend(r.seeds)
        step_v.extend(r.step_v)
        step_r.extend(r.step_r)
        for k, v in r.outcomes.items():
            outcomes[k] = outcomes.get(k, 0) + v
        for k, v in r.reward_component_sums.items():
            reward_component_sums[k] = reward_component_sums.get(k, 0.0) + v
    win_count = outcomes.get("box_possession", 0)
    n = len(rewards)
    return SeededEvalResult(
        n_episodes=n,
        seeds=seeds,
        repeats_per_seed=repeats_per_seed,
        mean_reward=float(np.mean(rewards)) if rewards else float("nan"),
        std_reward=float(np.std(rewards)) if rewards else float("nan"),
        win_rate_pct=100.0 * win_count / n if n else float("nan"),
        outcomes=outcomes,
        outcome_breakdown=outcome_breakdown(outcomes_list),
        mean_value_pred=float(np.mean(value_preds)) if value_preds else float("nan"),
        rewards=rewards,
        outcomes_list=outcomes_list,
        episode_lengths_s=episode_lengths_s,
        episode_reward_components=episode_reward_components,
        value_preds=value_preds,
        step_v=step_v,
        step_r=step_r,
        reward_component_sums=reward_component_sums,
    )


def _eval_worker_entry(worker_factory, seed_chunk, repeats_per_seed, win_outcome, swap_sides=False) -> SeededEvalResult:
    """Module-level (picklable) subprocess entry point -- each worker builds
    its OWN (env_factory, sample_action_fn) pair via ``worker_factory()``
    rather than trying to pickle live nn.Module/optimizer state across the
    process boundary, mirroring ai/ppo/rollout_worker.py's pattern."""
    # Each worker only ever does batch-of-1 CPU forward passes, so letting
    # torch/BLAS use its default multi-threaded pool means N worker
    # processes each spin up a full thread pool and fight over the same
    # cores -- this is what made parallel eval "hang" (severe CPU
    # oversubscription, not an actual deadlock). Mirrors
    # ai/ppo/rollout_worker.py's worker_torch_threads=1 default.
    import torch
    torch.set_num_threads(1)
    env_factory, sample_action_fn = worker_factory()
    return run_seeded_evaluation(
        env_factory, sample_action_fn, seed_chunk, repeats_per_seed, win_outcome, swap_sides=swap_sides,
    )


def run_seeded_evaluation_parallel(
    worker_factory: Callable[[], tuple],
    seeds: list[int],
    repeats_per_seed: int = 2,
    n_workers: int = 1,
    win_outcome: str = "box_possession",
    swap_sides: bool = False,
    pool=None,
) -> SeededEvalResult:
    """Parallel variant of run_seeded_evaluation().

    ``worker_factory`` must be a picklable, zero-arg, MODULE-LEVEL callable
    (not a closure/lambda) that each subprocess calls once to build its own
    ``(env_factory, sample_action_fn)`` pair (e.g. load a checkpoint fresh
    in that process) -- see ai/ppo/rollout_worker.py for the same pattern
    used by PPO rollout collection. ``n_workers<=1`` runs sequentially in
    the caller's process with no subprocess overhead. ``swap_sides``: see
    run_seeded_evaluation(). ``pool``: see
    ``run_seeded_evaluation_parallel_batched``'s identical parameter --
    reuse an already-running pool across several sequential calls instead
    of spawning a fresh one each time.
    """
    if n_workers <= 1:
        env_factory, sample_action_fn = worker_factory()
        return run_seeded_evaluation(
            env_factory, sample_action_fn, seeds, repeats_per_seed, win_outcome, swap_sides=swap_sides,
        )

    chunks = [c for c in (seeds[i::n_workers] for i in range(n_workers)) if c]
    _args = [(worker_factory, chunk, repeats_per_seed, win_outcome, swap_sides) for chunk in chunks]
    if pool is not None:
        results = pool.starmap(_eval_worker_entry, _args)
        return merge_eval_results(results, repeats_per_seed)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    log.info(f"  [seeded eval] running {len(seeds)}x{repeats_per_seed} episodes "
             f"across {len(chunks)} worker process(es)...")
    with _capped_thread_env_for_pool_spawn("1"), ctx.Pool(processes=len(chunks)) as _pool:
        results = _pool.starmap(_eval_worker_entry, _args)
    log.info("  [seeded eval] all workers finished, merging results.")
    return merge_eval_results(results, repeats_per_seed)
