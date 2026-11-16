"""PPOTrainer's periodic evaluations (vs rules, immobile, opponent types, neural snapshots) and their env/worker factories."""
from __future__ import annotations

from typing import Optional
import torch
from footballcoach.ai.eval.seeded_eval import _capped_thread_env_for_pool_spawn, run_seeded_evaluation, run_seeded_evaluation_batched, run_seeded_evaluation_parallel, run_seeded_evaluation_parallel_batched
from footballcoach.ai.ppo.trainer_common import _load_state_dict_tolerant, log


def rebuild_inference_trainer(
    decision_state: dict, execution_state: dict, separate_value_net: bool = False,
    value_state: Optional[dict] = None,
) -> "PPOTrainer":
    """Rebuild a fresh CPU, inference-only PPOTrainer from plain state dicts
    -- the single source of truth for "give a worker subprocess a usable
    trainer" across every parallel-worker path in this file (live
    nn.Module/optimizer objects aren't picklable across a process boundary,
    so every one of these workers rebuilds from scratch instead of
    inheriting a live trainer). Used by _eval_worker_factory below (parallel
    seeded eval) and ai/ppo/dagger.py's parallel rollout collection --
    previously duplicated inline in the former only; extracted here once a
    second caller needed the exact same rebuild logic.
    """
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer  # local: PPOTrainer's own module imports this one

    trainer = PPOTrainer.from_config(
        device=torch.device("cpu"), inference_only=True, separate_value_net=separate_value_net,
    )
    # Tolerant, not module.load_state_dict(..., strict=...) directly: a
    # physics-encoder-enabled DecisionNetwork's state_dict() deliberately
    # excludes ball_physics_encoder.*/player_physics_encoder.* (see that
    # method's own docstring -- they're an external checkpoint artifact,
    # never part of the main PPO checkpoint), but a freshly-constructed
    # DecisionNetwork here still HAS those params (loaded fresh from their
    # own physics_pretrain checkpoint path by from_config() above) -- a
    # strict load would then fail on "missing keys" for something that was
    # never supposed to travel through decision_state at all. Mirrors
    # load_checkpoint()'s own tolerant load for the exact same reason.
    _load_state_dict_tolerant(trainer.decision_net, decision_state, "decision_net")
    _load_state_dict_tolerant(trainer.execution_net, execution_state, "execution_net")
    if value_state is not None and trainer.value_net is not None:
        _load_state_dict_tolerant(trainer.value_net, value_state, "value_net")
    return trainer

def _build_eval_env_factory(use_rules_ai: bool, max_episode_s: float):
    """Builds a ``seed -> ScenarioEnv`` factory for periodic phase-1 eval
    (rules or immobile opponent). Extracted out of ``_eval_worker_factory``
    below so ``ai/eval/eval_worker.py``'s PERSISTENT eval workers can reuse
    the exact same env-building logic without also paying for a trainer
    rebuild on every eval call -- their trainer is cached across many eval
    calls and only refreshed on an explicit ``set_weights`` message, unlike
    the throwaway-Pool path below which rebuilds fresh every single call."""
    from footballcoach.rules_ai import Phase1RulesAI
    from footballcoach.ui.scenarios import build_1v1_scenario, ScenarioDefinition
    from footballcoach.ai.env.scenario_env import ScenarioEnv
    from footballcoach.ai.eval.seeded_eval import swap_player_states

    _label = "rules" if use_rules_ai else "immobile"

    def _eval_env_factory(seed: int, swap: bool = False) -> ScenarioEnv:
        def _eval_build(*_a, **_kw):
            _m = build_1v1_scenario(*_a, seed=seed, **_kw)
            if swap:
                swap_player_states(_m, "trainee", "opponent")
            if use_rules_ai:
                _m.player_by_id("opponent").ai = Phase1RulesAI()
            _m._opponent_use_rules_ai = use_rules_ai
            _m._opponent_is_immobile = not use_rules_ai
            return _m

        return ScenarioEnv(
            ScenarioDefinition(key=f"_eval_{_label}", label=f"eval_{_label}",
                               description=f"periodic {_label} eval", build=_eval_build),
            trainee_player_id="trainee",
            max_episode_s=max_episode_s,
        )

    return _eval_env_factory

def _eval_worker_factory(
    decision_state: dict, execution_state: dict, separate_value_net: bool,
    value_state: Optional[dict], use_rules_ai: bool, max_episode_s: float,
) -> tuple:
    """Module-level (picklable) zero-arg-after-partial factory for parallel
    seeded eval (ai/eval/seeded_eval.py's run_seeded_evaluation_parallel) --
    each subprocess rebuilds its own inference-only PPOTrainer from the
    passed state dicts via rebuild_inference_trainer(), mirroring
    ai/ppo/rollout_worker.py."""
    trainer = rebuild_inference_trainer(decision_state, execution_state, separate_value_net, value_state)
    return _build_eval_env_factory(use_rules_ai, max_episode_s), trainer._sample_action

def _eval_worker_factory_batched(
    decision_state: dict, execution_state: dict, separate_value_net: bool,
    value_state: Optional[dict], use_rules_ai: bool, max_episode_s: float,
) -> tuple:
    """Batched-eval counterpart to ``_eval_worker_factory``: returns
    ``(env_factory, trainer, secondary_trainer)`` -- the trainer OBJECT
    (not just its bound ``sample_action_fn``), since
    ``run_seeded_evaluation_batched`` needs ``_sample_action_batch()``.
    ``secondary_trainer`` is always ``None`` here: rules/immobile opponents
    have no neural decision to batch (see ``_secondary_neural_candidates``'s
    duck-typed skip of anything without ``is_due_for_decision()``)."""
    trainer = rebuild_inference_trainer(decision_state, execution_state, separate_value_net, value_state)
    return _build_eval_env_factory(use_rules_ai, max_episode_s), trainer, None

def _build_neural_vs_neural_env_factory(max_episode_s: float, opponent_sample_action_fn):
    """Builds a ``(seed, swap) -> ScenarioEnv`` factory for neural-vs-neural
    snapshot eval (see ``PPOTrainer._eval_vs_neural_snapshot``): the trainee
    uses whatever ``sample_action_fn`` the caller assigns onto the returned
    env (same convention as ``_build_eval_env_factory``), while the
    'opponent' secondary player is pinned to a DIFFERENT, fixed
    ``opponent_sample_action_fn`` (a frozen snapshot) via a small
    ``ScenarioEnv`` subclass -- ``reset()``'s normal
    ``maybe_assign_neural_opponent()`` (``ai/env/scenario_env.py``) always
    mirrors the trainee's OWN live weights onto secondary players (real
    self-play), so this overrides the opponent's ``.ai`` right after
    ``super().reset()`` assigns it, with an otherwise-identical
    ``NeuralPlayerAI`` just pointed at the snapshot instead."""
    from footballcoach.ui.scenarios import build_1v1_scenario, ScenarioDefinition
    from footballcoach.ai.env.scenario_env import ScenarioEnv
    from footballcoach.ai.eval.seeded_eval import swap_player_states

    class _NeuralOpponentEnv(ScenarioEnv):
        def reset(self, *, seed: Optional[int] = None):
            obs = super().reset(seed=seed)
            from footballcoach.ai.obs.encoder import rollout_task_phase
            from footballcoach.rules_ai import NeuralPlayerAI
            match = self._loop.match
            opponent = match.player_by_id_or_none("opponent")
            if opponent is None:
                return obs
            opponent.ai = NeuralPlayerAI(
                opponent_sample_action_fn,
                decision_interval_ticks=self._ticks_per_decision,
                max_episode_s=self.max_episode_s,
                ema_smoothed=self._sec_ema["opponent"].smoothed,
                rng=self.rng,
                task_phase=rollout_task_phase(self.phase),
            )
            return obs

    def _eval_env_factory(seed: int, swap: bool = False) -> ScenarioEnv:
        def _eval_build(*_a, **_kw):
            # opponent_rules_prob/opponent_immobile_prob=0.0: force
            # build_1v1_scenario's OWN internal opponent-type roll to land on
            # its "neural" branch (ai=None, no order touched at all) instead
            # of its default opponent_immobile_prob=1.0, which would assign a
            # JogOrder that re-asserts a fixed direction every tick
            # regardless of what player.ai does (see that function's own
            # opponent-roll comment, and ui/scenarios.py's
            # build_checkpoint_vs_checkpoint-style scenario for the exact
            # same fix already applied once before for the identical reason).
            # Without this, the opponent.ai reassignment below is completely
            # neutralized by the engine re-applying the leftover JogOrder on
            # top of it every tick -- confirmed via a real self-play sanity
            # check (checkpoint vs itself came back ~90% "trainee" win rate
            # instead of ~50%, i.e. the opponent was just jogging in a
            # straight line, never actually running any network).
            _m = build_1v1_scenario(
                *_a, seed=seed, opponent_rules_prob=0.0, opponent_immobile_prob=0.0, **_kw,
            )
            if swap:
                swap_player_states(_m, "trainee", "opponent")
            return _m

        return _NeuralOpponentEnv(
            ScenarioDefinition(key="_eval_neural_snapshot", label="eval_neural_snapshot",
                               description="neural-vs-neural snapshot eval", build=_eval_build),
            trainee_player_id="trainee",
            secondary_player_ids=["opponent"],
            max_episode_s=max_episode_s,
        )

    return _eval_env_factory

def _eval_worker_factory_neural_snapshot(
    trainee_decision_state: dict, trainee_execution_state: dict, trainee_value_state: Optional[dict],
    opponent_decision_state: dict, opponent_execution_state: dict, opponent_value_state: Optional[dict],
    separate_value_net: bool, max_episode_s: float,
) -> tuple:
    """Module-level (picklable) factory for parallel neural-vs-neural
    snapshot eval (``PPOTrainer._eval_vs_neural_snapshot``) -- mirrors
    ``_eval_worker_factory``, but rebuilds TWO inference-only PPOTrainers
    (the trainee's current live weights, and a frozen snapshot for the
    opponent) instead of one."""
    trainee_trainer = rebuild_inference_trainer(
        trainee_decision_state, trainee_execution_state, separate_value_net, trainee_value_state)
    opponent_trainer = rebuild_inference_trainer(
        opponent_decision_state, opponent_execution_state, separate_value_net, opponent_value_state)
    env_factory = _build_neural_vs_neural_env_factory(max_episode_s, opponent_trainer._sample_action)
    return env_factory, trainee_trainer._sample_action

def _eval_worker_factory_neural_snapshot_batched(
    trainee_decision_state: dict, trainee_execution_state: dict, trainee_value_state: Optional[dict],
    opponent_decision_state: dict, opponent_execution_state: dict, opponent_value_state: Optional[dict],
    separate_value_net: bool, max_episode_s: float,
) -> tuple:
    """Batched-eval counterpart to ``_eval_worker_factory_neural_snapshot``:
    returns ``(env_factory, trainee_trainer, opponent_trainer)`` instead of
    ``(env_factory, trainee_sample_action_fn)`` -- ``run_seeded_evaluation_batched``
    batches the opponent's decisions too (secondary_trainer=opponent_trainer)
    the same way ``BatchedEnvGroup.secondary_trainer`` does for training
    self-play, since neural-vs-neural eval's 'opponent' is exactly that kind
    of secondary neural player (see ``_build_neural_vs_neural_env_factory``'s
    ``_NeuralOpponentEnv``)."""
    trainee_trainer = rebuild_inference_trainer(
        trainee_decision_state, trainee_execution_state, separate_value_net, trainee_value_state)
    opponent_trainer = rebuild_inference_trainer(
        opponent_decision_state, opponent_execution_state, separate_value_net, opponent_value_state)
    env_factory = _build_neural_vs_neural_env_factory(max_episode_s, opponent_trainer._sample_action)
    return env_factory, trainee_trainer, opponent_trainer

def _resolve_snapshot_lookback(k: int | float, current_count: int) -> tuple[str, int] | None:
    """Turn one ``eval.neural_snapshot_lookbacks`` entry into (label, target checkpoint number),
    or None if there isn't enough run history yet for it (silently skipped by the caller, same as
    before this function existed).

    A WHOLE-NUMBER entry (1, 5, 2.0, ...) is a FIXED number of rollouts back: target =
    current_count - k, label "{k}_back" -- exactly the original behaviour, and int-vs-float typing
    of an already-integral value doesn't change that (json.load hands back a plain int for "5" and
    a float for "5.0", and both mean the same fixed lookback here). A NON-integral entry (e.g. 0.5)
    is instead a FRACTION of the run's checkpoint count so far: target = current_count -
    round(k * current_count), so e.g. 0.5 always points at the checkpoint sitting at the run's
    current halfway point -- a target that moves forward as current_count grows, unlike a fixed
    lookback. The label keeps the CONFIGURED fraction (e.g. "0.5_back"), not the drifting computed
    distance, so it groups consistently across a whole run's logs/graphs despite the underlying
    checkpoint number changing every time it's evaluated.
    """
    is_fraction = float(k) != int(k)
    k_effective = round(k * current_count) if is_fraction else int(k)
    if k_effective < 1:
        return None
    target_count = current_count - k_effective
    if target_count < 1:
        return None
    label = f"{k:g}_back" if is_fraction else f"{int(k)}_back"
    return label, target_count

def format_outcomes_with_pct(outcomes: dict) -> str:
    """Format an outcome-count dict as ``{'key': N (P%), ...}`` so relative
    shares are visible in logs alongside the raw counts, without requiring
    mental division while reading them.
    """
    total = sum(outcomes.values())
    if total == 0:
        return "{}"
    parts = [f"'{k}': {v} ({v / total * 100:.1f}%)" for k, v in outcomes.items()]
    return "{" + ", ".join(parts) + "}"


class TrainerEvalMixin:
    """Periodic evaluation for PPOTrainer. Mixed into PPOTrainer; relies on its attributes."""

    def _eval_vs_rules(self, max_episode_s: float) -> None:
        """Quick periodic eval vs immobile AND rules-based AI, shared by
        both the single-process and parallel training loops. Uses a FIXED
        seed list (ai_config.json['eval']) via ai/eval/seeded_eval.py so
        pre-training eval and every rollout's eval see identical scenarios
        -- comparable numbers across the whole run instead of noise from a
        fresh random scenario draw each time. PPO training itself stays
        unseeded. vs-immobile runs first (cheaper baseline sanity check,
        e.g. catching the "runs in circles vs immobile" regression) then
        vs-rules."""
        self._eval_vs_immobile(max_episode_s)
        self._eval_vs_opponent_type(max_episode_s, use_rules_ai=True)

    def _eval_role_phases(self) -> None:
        """Seeded evals of every role-based phase in the training mix, one log
        line per eval spec (ai/eval/role_phase_eval.py). Rules-only specs give
        the same result every time, so each runs once and is cached. No-op
        without a phase mix, so a phase-1-only run is unchanged."""
        if not self.phase_mix:
            return
        from footballcoach.ai.curriculum.phases import PHASES_BY_ID
        from footballcoach.ai.eval.role_phase_eval import run_role_phase_eval
        for pid in sorted(self.phase_mix):
            phase = PHASES_BY_ID[pid]
            if not phase.is_role_based:
                continue
            for spec in phase.eval_specs:
                rules_only = all(c == "rules" for c in spec["controllers"].values())
                key = (phase.name, spec["name"])
                result = self._role_eval_cache.get(key) if rules_only else None
                if result is None:
                    result = run_role_phase_eval(phase, spec, self._sample_action)
                    if rules_only:
                        self._role_eval_cache[key] = result
                log.info(f"  [eval {phase.name}/{spec['name']}] {result.summary()}")

    def _eval_vs_immobile(self, max_episode_s: float) -> None:
        """Seeded eval vs a standing-still opponent -- see _eval_vs_rules()."""
        self._eval_vs_opponent_type(max_episode_s, use_rules_ai=False)

    def _compute_rules_vs_rules_baseline(self, max_episode_s: float):
        """Rules-AI-vs-rules-AI reference point on the EXACT SAME fixed eval
        seed set (self._eval_seeds/_eval_repeats_per_seed) as the periodic
        "[eval vs rules]" check, so a trained policy's numbers can be
        compared directly against "what does a rules-AI trainee itself score
        on these scenarios" -- e.g. beating this baseline's win rate is a
        much more meaningful bar than 0%/100%.

        Computed and cached ONCE (returns the cached result on every
        subsequent call) since it never changes: same seeds, same
        deterministic rules-AI decision logic on both sides (only residual
        match-physics RNG varies run to run, which repeats_per_seed already
        averages over the same way the main eval does). Deliberately
        sequential, not parallel like _eval_vs_opponent_type's main path --
        this only ever runs once per training process, so the extra
        worker-pool code path isn't worth it for a one-time cost.
        """
        if self._rules_vs_rules_baseline is not None:
            return self._rules_vs_rules_baseline
        from footballcoach.rules_ai import Phase1RulesAI
        from footballcoach.ui.scenarios import build_1v1_scenario, ScenarioDefinition
        from footballcoach.ai.env.scenario_env import ScenarioEnv
        from footballcoach.ai.eval.seeded_eval import swap_player_states

        def _baseline_env_factory(seed: int, swap: bool = False) -> ScenarioEnv:
            def _build(*_a, **_kw):
                _m = build_1v1_scenario(*_a, seed=seed, **_kw)
                if swap:
                    swap_player_states(_m, "trainee", "opponent")
                for p in _m.players:
                    p.ai = Phase1RulesAI()
                _m._opponent_use_rules_ai = True
                _m._opponent_is_immobile = False
                return _m

            return ScenarioEnv(
                ScenarioDefinition(key="_eval_rules_vs_rules", label="eval_rules_vs_rules",
                                   description="rules-vs-rules eval baseline", build=_build),
                trainee_player_id="trainee",
                max_episode_s=max_episode_s,
            )

        try:
            # sample_action_fn=None: trainee's ai is already Phase1RulesAI()
            # from _build above, so ScenarioEnv.reset() never assigns a
            # NeuralPlayerAI (see its "if self.sample_action_fn is not None"
            # gate) -- both players stay pure rules-AI.
            self._rules_vs_rules_baseline = run_seeded_evaluation(
                _baseline_env_factory, None, self._eval_seeds, self._eval_repeats_per_seed,
                swap_sides=self._eval_swap_sides,
            )
        except Exception as _e:
            log.warning(f"  [eval baseline rules-vs-rules] failed: {_e}")
            self._rules_vs_rules_baseline = False  # sentinel: tried, failed, don't retry every rollout
        return self._rules_vs_rules_baseline

    def _eval_vs_opponent_type(self, max_episode_s: float, use_rules_ai: bool) -> None:
        _label = "rules" if use_rules_ai else "immobile"
        # If the curriculum never actually trains against an immobile
        # opponent, this check is pure sanity insurance (catching e.g. a
        # "runs in circles vs immobile" regression), not a tracked metric --
        # cut it down to 10 total episodes instead of the full
        # eval_seeds x eval_repeats_per_seed budget every rollout.
        if not use_rules_ai and self._phase1_opponent_immobile_ratio == 0:
            _seeds = self._eval_seeds[:10]
            _repeats = 1
        else:
            _seeds = self._eval_seeds
            _repeats = self._eval_repeats_per_seed
        try:
            _persistent_workers = getattr(self, "_persistent_eval_workers", None)
            if self._eval_n_parallel_workers > 1 and _persistent_workers:
                # Persistent-pool path (see ai/eval/eval_worker.py): workers
                # were already spawned once by _train_parallel() -- just
                # push current weights and dispatch, no process spawn here.
                _decision_state, _execution_state, _value_state = self._cpu_state_dicts()
                result = self._run_persistent_eval(
                    _persistent_workers, _decision_state, _execution_state, _value_state,
                    _seeds, _repeats, use_rules_ai, max_episode_s,
                    swap_sides=self._eval_swap_sides,
                    batched=self._eval_batched, envs_per_process=self._eval_envs_per_process,
                )
            elif self._eval_n_parallel_workers > 1:
                # Fallback parallel path (no persistent pool available --
                # e.g. evaluate.py CLI or the single-process train() loop):
                # each subprocess rebuilds its own trainer from these state
                # dicts (see _eval_worker_factory) -- must snapshot weights
                # now, not capture self._sample_action, since bound methods/
                # live nn.Modules aren't picklable.
                import functools
                _decision_state, _execution_state, _value_state = self._cpu_state_dicts()
                if self._eval_batched:
                    worker_factory = functools.partial(
                        _eval_worker_factory_batched, _decision_state, _execution_state,
                        self.separate_value_net, _value_state, use_rules_ai, max_episode_s,
                    )
                    result = run_seeded_evaluation_parallel_batched(
                        worker_factory, _seeds, _repeats,
                        n_workers=self._eval_n_parallel_workers,
                        swap_sides=self._eval_swap_sides,
                        envs_per_process=self._eval_envs_per_process,
                    )
                else:
                    worker_factory = functools.partial(
                        _eval_worker_factory, _decision_state, _execution_state,
                        self.separate_value_net, _value_state, use_rules_ai, max_episode_s,
                    )
                    result = run_seeded_evaluation_parallel(
                        worker_factory, _seeds, _repeats,
                        n_workers=self._eval_n_parallel_workers,
                        swap_sides=self._eval_swap_sides,
                    )
            else:
                # Reuse the exact same factory the persistent-pool/fallback-
                # parallel branches above use (_build_eval_env_factory) rather
                # than duplicating the build logic a third time -- keeps
                # swap_sides handling (swap_player_states) in one place.
                _eval_env_factory = _build_eval_env_factory(use_rules_ai, max_episode_s)
                if self._eval_batched:
                    result = run_seeded_evaluation_batched(
                        _eval_env_factory, self,
                        _seeds, _repeats, swap_sides=self._eval_swap_sides,
                        envs_per_process=self._eval_envs_per_process,
                    )
                else:
                    result = run_seeded_evaluation(
                        _eval_env_factory, self._sample_action,
                        _seeds, _repeats, swap_sides=self._eval_swap_sides,
                    )
            log.info(
                f"  [eval vs {_label}] step={self._total_steps:,}  "
                f"seeds={len(_seeds)}x{_repeats}  "
                f"win={result.win_rate_pct:.0f}%  "
                f"mean_rew={result.mean_reward:.3f}±{result.std_reward:.3f} "
                f"(sem={result.sem_reward:.3f})  "
                f"V={result.mean_value_pred:.3f}  gap={result.mean_value_pred - result.mean_reward:+.3f}  "
                f"outcomes={format_outcomes_with_pct(result.outcomes)}"
            )
            if use_rules_ai:
                _baseline = self._compute_rules_vs_rules_baseline(max_episode_s)
                if _baseline:
                    log.info(
                        f"  [eval baseline rules-vs-rules] seeds={len(self._eval_seeds)}x{self._eval_repeats_per_seed}  "
                        f"win={_baseline.win_rate_pct:.0f}%  "
                        f"mean_rew={_baseline.mean_reward:.3f}±{_baseline.std_reward:.3f} "
                        f"(sem={_baseline.sem_reward:.3f})  "
                        f"outcomes={format_outcomes_with_pct(_baseline.outcomes)}"
                    )
        except Exception as _e:
            log.warning(f"  [eval vs {_label}] failed: {_e}")

    def _eval_vs_neural_snapshot(
        self, max_episode_s: float, label: str, opponent_snapshot: dict, pool=None,
    ) -> None:
        """One neural-vs-neural comparison: current live weights (trainee
        side) vs a frozen snapshot (opponent side), on the SAME fixed eval
        seed set as _eval_vs_rules() (self._eval_seeds/_eval_repeats_per_seed
        -- 'use the same number of episodes as the existing evals'), with the
        same eval.swap_sides fairness treatment. Unlike _eval_vs_opponent_type,
        this doesn't use the persistent eval worker pool (that pool's IPC
        protocol only supports ONE trainer's weights per call, see
        ai/eval/eval_worker.py) -- it uses a throwaway parallel pool via
        run_seeded_evaluation_parallel[_batched], sized by
        eval.eval_n_parallel_workers. Pass ``pool`` (see
        _maybe_run_neural_snapshot_eval, which spawns ONE pool per rollout
        cycle and reuses it across every lookback comparison) rather than
        letting each call spawn its own -- with
        eval.neural_snapshot_lookbacks holding several entries,
        respawning eval_n_parallel_workers fresh PROCESSES per lookback
        (on top of the already-running persistent rollout+eval pools) is
        real, confirmed oversubscription that can stall a live training run
        for many minutes with every process fighting for the same cores.
        Failure is caught and logged, never raised, matching
        _eval_vs_opponent_type -- a broken comparison here must not take
        down the training run."""
        try:
            import functools
            _decision_state, _execution_state, _value_state = self._cpu_state_dicts()
            if self._eval_batched:
                worker_factory = functools.partial(
                    _eval_worker_factory_neural_snapshot_batched,
                    _decision_state, _execution_state, _value_state,
                    opponent_snapshot["decision"], opponent_snapshot["execution"], opponent_snapshot["value"],
                    self.separate_value_net, max_episode_s,
                )
                result = run_seeded_evaluation_parallel_batched(
                    worker_factory, self._eval_seeds, self._eval_repeats_per_seed,
                    n_workers=self._eval_n_parallel_workers,
                    swap_sides=self._eval_swap_sides,
                    envs_per_process=self._eval_envs_per_process,
                    pool=pool,
                )
            else:
                worker_factory = functools.partial(
                    _eval_worker_factory_neural_snapshot,
                    _decision_state, _execution_state, _value_state,
                    opponent_snapshot["decision"], opponent_snapshot["execution"], opponent_snapshot["value"],
                    self.separate_value_net, max_episode_s,
                )
                result = run_seeded_evaluation_parallel(
                    worker_factory, self._eval_seeds, self._eval_repeats_per_seed,
                    n_workers=self._eval_n_parallel_workers,
                    swap_sides=self._eval_swap_sides,
                    pool=pool,
                )
            log.info(
                f"  [eval vs neural:{label}] step={self._total_steps:,}  "
                f"opp_step={opponent_snapshot['step']:,}  "
                f"seeds={len(self._eval_seeds)}x{self._eval_repeats_per_seed}  "
                f"win={result.win_rate_pct:.0f}%  "
                f"mean_rew={result.mean_reward:.3f}±{result.std_reward:.3f} "
                f"(sem={result.sem_reward:.3f})  "
                f"outcomes={format_outcomes_with_pct(result.outcomes)}"
            )
        except Exception as _e:
            log.warning(f"  [eval vs neural:{label}] failed: {_e}")

    def _maybe_run_neural_snapshot_eval(self, max_episode_s: float) -> None:
        """Every eval.neural_snapshot_eval_every_n_rollouts rollouts: compare
        the CURRENT live policy against the checkpoint saved
        eval.neural_snapshot_lookbacks[i] rollouts ago (or, for a float
        entry, at the corresponding fraction of the run's checkpoint count
        so far -- see _resolve_snapshot_lookback), for every entry in that
        list, plus 'original' (this run's checkpoint1.pt) -- using the
        checkpoint FILES _save_checkpoint() already writes every rollout
        regardless of this feature, rather than keeping a separate in-memory
        snapshot ring. Requires self.checkpoint_dir (no-op without one,
        since there's nothing on disk to compare against). Lookbacks with
        not-yet-enough history are silently skipped, not logged as failures
        -- this is expected early in a run and self-resolves as more
        checkpoints accumulate.

        All comparisons this cycle share ONE throwaway worker pool (spawned
        once here, closed at the end) instead of each
        _eval_vs_neural_snapshot() call spawning its own -- with several
        lookbacks configured, respawning eval_n_parallel_workers fresh
        processes per comparison stacks on top of the already-running
        persistent rollout+eval pools and can genuinely stall a live run for
        minutes (confirmed live: processes starved of CPU before they could
        even finish importing). Reusing one pool cuts that spawn/teardown
        cost from N times per cycle to once.
        """
        if self._neural_snapshot_every_n <= 0 or self.checkpoint_dir is None:
            return
        self._nn_snapshot_rollout_count += 1
        if self._nn_snapshot_rollout_count % self._neural_snapshot_every_n != 0:
            return

        current_count = self._checkpoint_count  # _save_checkpoint() already ran this rollout
        comparisons: list[tuple[str, dict]] = []
        for k in self._neural_snapshot_lookbacks:
            resolved = _resolve_snapshot_lookback(k, current_count)
            if resolved is None:
                continue
            label, target_count = resolved
            ckpt_path = self.checkpoint_dir / f"checkpoint{target_count}.pt"
            if not ckpt_path.exists():
                continue
            try:
                snapshot = self._load_snapshot_dict_from_checkpoint(ckpt_path)
            except Exception as _e:
                log.warning(f"  [eval vs neural:{label}] failed to load {ckpt_path}: {_e}")
                continue
            comparisons.append((label, snapshot))

        # checkpoint1.pt always exists by this point (current_count >= 1 is
        # guaranteed -- _save_checkpoint() already ran this rollout, see the
        # comment above) -- always attempt "original", even on rollout 1
        # itself, where current_count == 1 means this IS checkpoint1.pt: a
        # degenerate but still useful self-play comparison (current live
        # weights vs an identical copy of themselves), since both sides
        # still sample stochastically -- a ~50% win rate here is the same
        # "eval pipeline isn't systematically biased" sanity check as this
        # session's earlier manual self-play probes, now free every run.
        original_path = self.checkpoint_dir / "checkpoint1.pt"
        if original_path.exists():
            try:
                comparisons.append(("original", self._load_snapshot_dict_from_checkpoint(original_path)))
            except Exception as _e:
                log.warning(f"  [eval vs neural:original] failed to load {original_path}: {_e}")

        if not comparisons:
            return

        pool = None
        if self._eval_n_parallel_workers > 1:
            import multiprocessing as mp
            # See _capped_thread_env_for_pool_spawn's own docstring: without
            # this, each of these eval_n_parallel_workers fresh processes
            # defaults to one BLAS thread per logical core, on top of the
            # already-running persistent rollout+eval worker pools -- real,
            # confirmed to crash a live run outright with
            # "OMP: Error #111: Memory allocation failed.", not just slow it
            # down. The other throwaway seeded-eval pools (seeded_eval.py's
            # own run_seeded_evaluation_parallel[_batched] fallback path)
            # are capped the same way at their own Pool(...) construction.
            with _capped_thread_env_for_pool_spawn("1"):
                pool = mp.get_context("spawn").Pool(processes=self._eval_n_parallel_workers)
        try:
            for label, snapshot in comparisons:
                self._eval_vs_neural_snapshot(max_episode_s, label, snapshot, pool=pool)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
