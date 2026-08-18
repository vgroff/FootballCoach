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

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from footballcoach.ai.env.scenario_env import ScenarioEnv

log = logging.getLogger("footballcoach.ai.eval.seeded_eval")


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
    env_factory: Callable[[int], ScenarioEnv],
    sample_action_fn,
    seeds: list[int],
    repeats_per_seed: int = 2,
    win_outcome: str = "box_possession",
    trial_log_every: int = 0,
    gamma: float = 0.98,
) -> SeededEvalResult:
    """Run ``len(seeds) * repeats_per_seed`` episodes, return aggregate stats.

    Args:
        env_factory: seed -> freshly-built ScenarioEnv. Repeating the same
            seed still gives different episodes since match physics/action
            sampling keep their own residual randomness -- only the initial
            scenario setup (positions/attributes/ball state/opponent-type
            roll) is pinned by the seed.
        sample_action_fn: forwarded to env.sample_action_fn (e.g.
            trainer._sample_action).
        seeds: fixed list of scenario seeds (see default_eval_seeds()).
        repeats_per_seed: episodes run per seed; all repeats are pooled into
            the same aggregate stats as every other episode (no separate
            per-seed breakdown -- not needed for the current use case).
        gamma: discount factor used for the per-step V-vs-R diagnostic
            (mean_step_v/mean_step_r on the result) -- no bootstrap at
            episode end, matching the prior evaluate.py behaviour.
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
    total = len(seeds) * max(1, repeats_per_seed)
    for seed in seeds:
        for _ in range(max(1, repeats_per_seed)):
            env = env_factory(seed)
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


def _merge_eval_results(results: list[SeededEvalResult], repeats_per_seed: int) -> SeededEvalResult:
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


def _eval_worker_entry(worker_factory, seed_chunk, repeats_per_seed, win_outcome) -> SeededEvalResult:
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
    return run_seeded_evaluation(env_factory, sample_action_fn, seed_chunk, repeats_per_seed, win_outcome)


def run_seeded_evaluation_parallel(
    worker_factory: Callable[[], tuple],
    seeds: list[int],
    repeats_per_seed: int = 2,
    n_workers: int = 1,
    win_outcome: str = "box_possession",
) -> SeededEvalResult:
    """Parallel variant of run_seeded_evaluation().

    ``worker_factory`` must be a picklable, zero-arg, MODULE-LEVEL callable
    (not a closure/lambda) that each subprocess calls once to build its own
    ``(env_factory, sample_action_fn)`` pair (e.g. load a checkpoint fresh
    in that process) -- see ai/ppo/rollout_worker.py for the same pattern
    used by PPO rollout collection. ``n_workers<=1`` runs sequentially in
    the caller's process with no subprocess overhead.
    """
    if n_workers <= 1:
        env_factory, sample_action_fn = worker_factory()
        return run_seeded_evaluation(env_factory, sample_action_fn, seeds, repeats_per_seed, win_outcome)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    chunks = [c for c in (seeds[i::n_workers] for i in range(n_workers)) if c]
    log.info(f"  [seeded eval] running {len(seeds)}x{repeats_per_seed} episodes "
             f"across {len(chunks)} worker process(es)...")
    with ctx.Pool(processes=len(chunks)) as pool:
        results = pool.starmap(
            _eval_worker_entry,
            [(worker_factory, chunk, repeats_per_seed, win_outcome) for chunk in chunks],
        )
    log.info("  [seeded eval] all workers finished, merging results.")
    return _merge_eval_results(results, repeats_per_seed)
