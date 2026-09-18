"""Batched multi-environment rollout collection.

``rollout_worker.py`` runs ``n_processes`` worker PROCESSES, each with
its own single ``ScenarioEnv``, each doing its own batch-of-1 CPU network
forward pass whenever its trainee needs a decision. Profiled this session:
rollout collection is ~88% CPU neural-network inference (not physics), and a
direct benchmark showed 6 separate batch=1 CPU forward passes take 51.2s for
a fixed workload where 1 batch=6 CPU call takes only 10.5s (~4.88x) -- nearly
all of that gap is Python/PyTorch dispatch overhead on tiny batch-of-1 calls,
not actual compute. This module gets that speedup by running
``envs_per_process`` environments TOGETHER in one process, sharing one
trainer, and batching every environment's trainee decision into ONE
``PPOTrainer._sample_action_batch()`` call per round instead of one
``_sample_action()`` call per environment per decision.

Near-zero behavior change: ``ScenarioEnv.step()`` is architecturally almost
always exactly one full decision interval (``NeuralPlayerAI``'s
``decision_interval_ticks`` is constructed equal to
``ScenarioEnv._ticks_per_decision``, the same loop bound ``step()`` uses
internally -- ``ai/env/scenario_env.py:152,299,313,414``), so the trainee is
*usually* due for a decision on the very first physics tick of every
``env.step()`` call, whether continuing an existing episode or immediately
following a fresh ``env.reset()`` (which itself needs no decision at all).
That's what lets calling every environment's next ``env.step()`` (or
``env.reset()``, for one that just finished) together, once per round, keep
every environment's decisions naturally, mostly synchronized -- no waiting,
no held-back episodes, no global tick counter needed.

NOT a hard guarantee, though: ``ScenarioEnv.step()`` has at least one
legitimate early-exit (trainee already in the opponent box with possession
-- see its own comment) that can end a decision interval several ticks
short, leaving the NEXT round's would-be decision genuinely not due yet for
that one env. An earlier version of this module asserted every env must be
due at the start of a round and crashed a real training run the first time
this fired in practice -- ``collect()`` now checks
``ai.is_due_for_decision()`` first and simply skips a not-due env from that
round's batch (see ``collect()``'s own docstring for the full mechanism);
its ``_precomputed_result`` stays unset, so its own ``env.step()`` call
falls through to ``act()``'s completely normal cached-gating path for that
tick -- identical to what happens for that same tick in the unbatched
``rollout_worker.py`` path, so this is not a behavior change, just
tolerance for something the original design incorrectly assumed could never
happen.

Mechanism, per round:
  1. For each env: check its trainee's ``NeuralPlayerAI.is_due_for_decision()``;
     for due envs only, call ``prepare()`` (see ``rules_ai.py`` -- the exact
     same call that would fire naturally as the first physics tick inside
     the upcoming ``env.step()``, just invoked one line earlier so its obs
     can be pooled with every other due env's). Not-due envs are skipped
     entirely this round (see above).
  2. Concatenate the collected obs dicts (due envs only) into one batch;
     call ``trainer._sample_action_batch(...)`` ONCE.
  3. Stash each due env's result on its trainee AI's ``_precomputed_result``.
  4. Call ``env.step()`` normally for every env (due or not) -- a due env's
     internal first tick's ``player.ai.act()`` (fired by the completely
     UNMODIFIED ``Match._process_orders()``) sees the precomputed result and
     consumes it via ``apply()`` instead of calling ``sample_action_fn``
     again; every later tick within that same ``env.step()`` call, and every
     tick of a not-due env's ``env.step()`` call, behaves exactly as today
     (cached-gating reapplication via ``act()``'s normal, un-precomputed path).
  5. Normal per-env bookkeeping, mirroring ``rollout_worker.py``'s
     ``_collect()`` line for line, just once per env per round instead of
     once per worker process.

Secondary neural players (e.g. a self-play opponent) are batched too, but
only opt-in via ``ppo.batch_secondary_players`` (default ``False``) --
``_batched_worker_main`` passes ``secondary_trainer=trainer`` (the SAME
trainer as the primary, i.e. true self-play with identical weights) to
``BatchedEnvGroup`` when set, which batches every due secondary player's
decision into its own extra ``_sample_action_batch()`` call per round (see
``BatchedEnvGroup``'s own docstring and ``_batch_due_decisions()``). Default
``False`` preserves the original behavior exactly: secondary players decide
unbatched, one at a time, via their own ``env.sample_action_fn`` fallback
inside ``env.step()``. Worth turning on whenever a curriculum phase leans on
neural self-play heavily (e.g. phase 1's default
``phase1_opponent_neural_ratio``, often the majority of episodes).

Episode-seed replay (opt-in, ``ppo.episode_replay_enabled``): every
mid-collection episode reset is now given an EXPLICIT seed (drawn via
``random.randint`` when not replaying -- statistically equivalent to the
prior bare ``env.reset()``, see ``ScenarioEnv.reset()``'s own docstring on
explicit-seed vs OS-entropy being the same underlying RNG) and recorded on
``stats[i]["episode_seeds"]`` in exact 1:1 order with
``stats[i]["episode_rewards"]``. ``collect(..., replay_seeds=[...])`` lets a
caller (``PPOTrainer._train_batched_parallel``, see its own docstring for
the "highest mean |advantage| episode" selection this feeds) force specific
seeds onto the next available episode slots instead of drawing fresh ones --
one shared FCFS queue for the whole call, consumed by whichever env resets
first; once exhausted, resets fall back to fresh random seeds exactly as
before this feature existed. The FIRST episode per env slot (from
``__init__``'s own initial reset) is deliberately left unseeded/unrecorded
(``None`` in ``episode_seeds``) -- one episode per slot, ever, not worth the
churn.

Chunked streaming (opt-in, ``ppo.batched_rollout_chunk_steps``): fixes a
real production crash -- ``_batched_worker_main`` used to run ``collect()``
to full completion (accumulating ALL ``steps_per_worker`` transitions, e.g.
21,000 rows across ``envs_per_process`` envs) before pickling the entire
result list into ONE ``conn.send()`` call. With many worker processes all
finishing around the same time, that many simultaneous huge pickle
allocations spiked system RAM past what was available (``MemoryError``
inside ``_ForkingPickler.dumps``). ``collect(chunk_steps=K,
on_chunk=callback)`` periodically flushes (every ``K`` steps collected,
checked once per round) instead of only at the very end.

**Flushes are WHOLE-EPISODE ONLY.** Each mid-collection flush sends, per
env, exactly the rows up to and including that env's last completed episode
(``RolloutBuffer.pop_complete_episodes()``) and KEEPS the unfinished tail in
the worker's buffer until its episode completes -- an episode is never cut in
half at a flush boundary. (An earlier version flushed everything regardless
and just accepted the cut; that bootstrap-truncated GAE for the rows next to
every boundary, made episode-seed-replay's per-episode mean(|advantage|) use
only half an episode's rows, and let one episode's halves land on both sides
of a train/val split -- all avoidable, so it's gone.) Consequences: a
mid-collection chunk always ends on a ``done=1`` row, so its GAE needs no
bootstrap value (``last_value`` is a literal 0.0 and the per-flush value-net
bootstrap forward pass is skipped entirely) and its MC returns are exact; a
chunk's ``stats`` describe exactly the episodes whose rows it contains
(each episode's stats travel with its rows); the buffers handed to
``on_chunk`` are NEW objects (not the live per-env buffers), so a consumer
may hold them. If a flush is due but no env has completed an episode yet,
nothing is sent and the next round tries again (buffers just keep growing
until an episode ends -- episodes always end, at worst by timeout).
Per-episode mid-episode accumulators (``episode_reward_accum``,
``_current_episode_seed``, ...) are untouched by flushes, as before.

The final flush at the end of ``collect()`` still sends everything left --
the last complete episodes AND each env's trailing in-progress episode, with
a real bootstrap ``last_value`` -- so, as always, exactly one partial
episode per env per rollout is bootstrap-truncated (unavoidable: the env
carries on into the next rollout). ``chunk_steps=None`` (default) is a
complete no-op -- ``on_chunk`` is never called, ``collect()`` returns the
full list exactly as before this feature existed.

Worker-side finalization (opt-in per ``collect`` command, ``returns`` spec):
building the ``as_tensors`` batch and the GAE/MC returns is done INSIDE the
worker (``finalize_result_for_wire``), and only the finished flat arrays go
over the pipe (as numpy, not torch tensors -- no torch shared-memory
reductions involved) instead of a pickled per-row ``RolloutBuffer`` (millions
of tiny dicts/arrays -- heavy to pickle, unpickle and then re-stack in the
main process). Peak memory on both sides is then bounded by one chunk. With
no ``returns`` spec the wire format is exactly the legacy
``{"buffer", "last_value", "stats"}``.

Two non-obvious things discovered (the hard way, via a failing test) while
building this:

- ``BatchedEnvGroup.collect()`` NEVER calls ``env.sample_action_fn`` or the
  trainee AI's own ``.sample_action_fn`` -- every decision is always
  precomputed via ``trainer._sample_action_batch(...)`` before ``env.step()``
  runs, so ``NeuralPlayerAI.act()``'s ``sample_action_fn`` fallback branch is
  never reached in batched collection. ``collect()`` exposes its own
  ``deterministic``/``deterministic_decision``/``deterministic_direction``
  kwargs (forwarded straight to ``_sample_action_batch``) as the actual,
  single source of truth for sampling behavior -- setting them on
  ``env.sample_action_fn`` instead is silently ignored.
- Despite the above, ``env.sample_action_fn`` still MUST be set (to
  anything, even a plain ``trainer._sample_action`` never actually called)
  before ``env.reset()`` runs -- ``ScenarioEnv.reset()`` itself checks it to
  decide whether/how to wire a ``NeuralPlayerAI`` onto the trainee at all;
  without it the trainee's ``.ai`` stays ``None`` and every ``prepare()``
  call in this module crashes with ``AttributeError``.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Optional

import torch

log = logging.getLogger("footballcoach.ai.ppo.batched_rollout_worker")


def _stack_obs_dicts(obs_dicts: list) -> dict:
    """Stack a list of unbatched obs dicts (as ``ObservationBatch.to_torch_dict()``
    produces) into one batched dict, ready for ``_sample_action_batch``."""
    keys = obs_dicts[0].keys()
    return {k: torch.stack([o[k] for o in obs_dicts]) for k in keys}


def _secondary_neural_candidates(env) -> list[tuple]:
    """(player, ai, match) triples for every one of this env's
    secondary_player_ids currently driven by a NeuralPlayerAI --
    rules-based/immobile/None secondaries (duck-typed: anything without
    an is_due_for_decision() method) are excluded, exactly as they
    already are from the unbatched path (they never had a
    _precomputed_result hook to consume in the first place). Module-level
    (not a BatchedEnvGroup method) so ai/eval/seeded_eval.py's batched eval
    consumer can reuse the exact same logic without needing a full
    BatchedEnvGroup instance."""
    out = []
    for pid in env.secondary_player_ids:
        try:
            sec_player = env.match.player_by_id(pid)
        except KeyError:
            continue
        sec_ai = sec_player.ai
        if not hasattr(sec_ai, "is_due_for_decision"):
            continue
        out.append((sec_player, sec_ai, env.match))
    return out


def _batch_due_decisions(
    candidates: list[tuple], trainer,
    deterministic: bool, deterministic_decision: bool, deterministic_direction: bool,
) -> None:
    """Shared by BatchedEnvGroup's trainee/secondary collection phases AND
    ai/eval/seeded_eval.py's batched eval consumer: given (player, ai, match)
    candidate triples, check which are actually due this round, prepare()
    each due one's observation, run ONE batched
    ``trainer._sample_action_batch()`` call over all of them together, and
    stash the result on each ai as ``_precomputed_result`` -- consumed the
    next time ``Match._process_orders()`` calls that ai's ``act()`` (see
    ``NeuralPlayerAI.act()``'s own docstring for the hook). No-op if nothing
    in `candidates` is due this round. Module-level (not a method) so it has
    exactly one implementation shared by both consumers."""
    obs_dicts = []
    due_ais = []
    for player, ai, match in candidates:
        if not ai.is_due_for_decision():
            continue
        obs_dict = ai.prepare(player, match, 0)
        assert obs_dict is not None, (
            "is_due_for_decision() said this player was due, but "
            "prepare() returned None anyway -- the two are now out "
            "of sync (see is_due_for_decision()'s docstring for the "
            "exact condition it checks)."
        )
        obs_dicts.append(obs_dict)
        due_ais.append(ai)

    if not obs_dicts:
        return
    obs_batch = _stack_obs_dicts(obs_dicts)
    results = trainer._sample_action_batch(
        obs_batch,
        deterministic=deterministic,
        deterministic_decision=deterministic_decision,
        deterministic_direction=deterministic_direction,
    )
    for ai, result in zip(due_ais, results):
        ai._precomputed_result = result


def _new_episode_stats() -> dict:
    return {
        "episode_rewards": [],
        "episode_seeds": [],
        "episode_outcome_labels": [],
        "secondary_episode_rewards": [],
        "episode_outcomes_vs_rules": [],
        "episode_outcomes_vs_neural": [],
        "episode_outcomes_vs_immobile": [],
        "episode_comp_list": [],
        "episode_durations_s": [],
    }


class BatchedEnvGroup:
    """Owns ``n_envs`` ``ScenarioEnv`` instances and steps them together,
    one round at a time, batching the trainee's decision network call
    across all of them. Single-process, directly testable without any
    multiprocessing involved -- see ``_batched_worker_main`` below for the
    Process+Pipe wrapper real parallel collection uses.

    ``secondary_trainer`` (default ``None``): when given, ALSO batches every
    secondary/opponent player's decision (across every env's
    ``secondary_player_ids``) into one extra ``secondary_trainer.
    _sample_action_batch()`` call per round, via the exact same
    ``prepare()``/``_precomputed_result`` hook used for the trainee (see
    ``NeuralPlayerAI.act()``'s own docstring -- that hook is already
    per-AI-instance and player-agnostic, so no changes to ``NeuralPlayerAI``
    or ``ScenarioEnv`` are needed for this). ``None`` (default) preserves
    the exact prior behavior: secondary players decide unbatched, one at a
    time, via their own ``env.sample_action_fn`` fallback inside
    ``env.step()``'s internal tick loop -- see this module's own top
    docstring ("OUT OF SCOPE for batching in this first pass"). Pass the
    SAME trainer as the primary one for real self-play (identical weights,
    just batched for speed); pass a DIFFERENT trainer for e.g. neural-vs-
    neural eval (a frozen snapshot on the opponent side). Only ONE shared
    secondary_trainer for the whole group -- every env's secondary
    player(s) must be driven by the same network for a given
    ``collect()``/``collect_eval_episodes()`` call.
    """

    def __init__(
        self, envs: list, trainer, seeds: Optional[list] = None,
        secondary_trainer=None,
    ) -> None:
        assert len(envs) > 0, "BatchedEnvGroup needs at least one env"
        if seeds is not None:
            assert len(seeds) == len(envs), "seeds must have exactly one entry per env"
        self.envs = envs
        self.trainer = trainer
        self.secondary_trainer = secondary_trainer
        from footballcoach.ai.ppo.rollout_buffer import RolloutBuffer
        self._buffers = [RolloutBuffer() for _ in envs]
        # Always resets every env exactly once here, regardless of whether
        # the caller already reset it -- pass `seeds` for reproducible
        # construction (e.g. tests comparing against a controlled reference
        # run) rather than relying on a pre-reset env, which this would
        # otherwise silently discard.
        if seeds is not None:
            self._last_obs = [env.reset(seed=s) for env, s in zip(envs, seeds)]
        else:
            self._last_obs = [env.reset() for env in envs]
        # Seed of each env's CURRENTLY IN-PROGRESS episode, for episode-seed
        # replay (see collect()'s replay_seeds kwarg and this module's own
        # top docstring). This first episode per slot is deliberately left
        # unseeded/unrecorded (None) -- every LATER reset, inside collect(),
        # is given an explicit seed and recorded here before being
        # overwritten for the next episode.
        self._current_episode_seed: list[Optional[int]] = [None] * len(envs)

    def _trainee_player_and_ai(self, env):
        player = env.match.player_by_id(env.trainee_player_id)
        return player, player.ai

    def _secondary_neural_candidates(self, env) -> list[tuple]:
        return _secondary_neural_candidates(env)

    def _batch_due_decisions(
        self, candidates: list[tuple], trainer,
        deterministic: bool, deterministic_decision: bool, deterministic_direction: bool,
    ) -> None:
        _batch_due_decisions(candidates, trainer, deterministic, deterministic_decision, deterministic_direction)

    def collect(
        self, n_steps: int,
        deterministic: bool = False,
        deterministic_decision: bool = False,
        deterministic_direction: bool = False,
        replay_seeds: Optional[list[int]] = None,
        chunk_steps: Optional[int] = None,
        on_chunk=None,
        progress_value=None,
    ) -> list[dict]:
        """Collect until at least ``n_steps`` total trainee+secondary steps
        have been recorded across the whole group (mirrors
        ``rollout_worker.py``'s ``_collect(n_steps, ...)`` stopping
        condition, just incrementing by however many envs/secondaries
        contribute a step each round instead of exactly 1). Returns a list
        of ``len(self.envs)`` dicts, each shaped exactly like
        ``rollout_worker.py._collect()``'s return value
        (``buffer``/``last_value``/``stats``) -- a drop-in per-env
        replacement for what one whole rollout_worker.py process returns
        today.

        ``deterministic``/``deterministic_decision``/``deterministic_direction``:
        forwarded to every ``_sample_action_batch`` call this round -- real
        rollout collection always leaves these at the default (``False``,
        matching ``rollout_worker.py``'s own always-stochastic
        ``_sample_action`` usage); set them for a fully reproducible,
        RNG-free comparison (e.g. tests). NOTE: each env's own
        ``env.sample_action_fn`` is NOT consulted here for envs that ARE due
        this round (that indirection only matters for the single-env
        ``act()`` fallback path) -- these three explicit kwargs are the
        actual, single source of truth for how THIS group samples envs it
        batches. The one exception: an env that is NOT yet due this round
        (see below) steps itself completely normally, including its own
        potential ``sample_action_fn`` fallback if it somehow becomes due
        mid-``env.step()`` -- structurally the same as any single-env
        rollout, since this method never touches that env at all this round.

        Not every env is guaranteed to be due for a decision at the start of
        every round: ``ScenarioEnv.step()`` almost always runs a full
        ``decision_interval_ticks``-tick interval (see this module's
        top docstring), but it has at least one legitimate early-exit
        (trainee already in the opponent box with possession -- further
        ticks would only accumulate spurious negative progress) that can
        end an interval several ticks short. When that happens, the
        NEXT round's would-be decision is not actually due yet -- calling
        ``prepare()`` on it would legitimately return ``None`` (exactly the
        cached-gating-reapply behavior the unbatched path already relies
        on). Each round therefore checks ``ai.is_due_for_decision()`` FIRST
        (a pure peek, see its own docstring for why calling ``prepare()``
        itself to check would silently desync the cadence) and only
        includes due envs in this round's batch; a not-due env's
        ``_precomputed_result`` is simply left ``None``, so its own
        ``env.step()`` call falls through to ``act()``'s normal per-tick
        path (which will itself re-check due-ness and correctly just
        re-apply cached gating) -- identical to what the unbatched
        rollout_worker.py path already does for that same tick, so this
        adds no behavior change, just tolerance for an env not being due.

        ``replay_seeds``: optional list of seeds to force onto the next
        available episode slots in THIS call, instead of drawing fresh
        random ones -- see this module's top docstring ("Episode-seed
        replay") and ``PPOTrainer._train_batched_parallel``'s
        highest-mean-|advantage| seed selection. One shared queue for the
        whole call, consumed FCFS by whichever env resets first; once
        exhausted (or if ``None``/empty), every later reset this call draws
        a fresh random seed exactly as if this parameter didn't exist.
        Every episode's seed (replayed or fresh) is recorded on
        ``stats[i]["episode_seeds"]``, one entry per completed episode in
        exact 1:1 order with ``stats[i]["episode_rewards"]``.

        ``chunk_steps``/``on_chunk``: see this module's top docstring
        ("Chunked streaming", WHOLE-EPISODE flushes) -- when ``chunk_steps``
        is set, every time that many steps have been collected since the
        last flush (checked once per round, never mid-round), ``on_chunk``
        is called with a per-env result list (same shape as this method's own
        return value) holding each env's COMPLETED episodes only (rows +
        the stats of exactly those episodes); the unfinished tail of each
        env's buffer stays put. If no env has a completed episode yet,
        nothing is sent and the flush is retried next round. A final flush
        happens after the collection loop ends for whatever wasn't already
        flushed (including each env's trailing in-progress episode, with a
        real bootstrap ``last_value``). In this mode the method itself
        returns ``[]`` -- everything was already delivered via ``on_chunk``.
        ``chunk_steps=None`` (default) disables this entirely: ``on_chunk``
        is never called and the full result list is returned normally,
        exactly as before this parameter existed.

        ``progress_value``: optional shared ``ctx.Value("l", 0)`` (same
        pattern as ``rollout_worker.py``'s own ``progress_value``); after
        every round, the rows collected that round are ADDED to it under
        its lock, so a caller in another process can poll one aggregate
        counter and render a live progress bar across all workers. This
        method never resets it -- the caller owns that (resetting here
        would race between workers all sharing one counter). ``None``
        (default) = no effect.

        NOTE for ``on_chunk`` implementers: the ``"buffer"`` in every
        mid-collection chunk is a NEW ``RolloutBuffer`` (from
        ``pop_complete_episodes``), safe to keep. Only the FINAL flush's
        buffers are the live ``self._buffers[i]`` objects (nothing mutates
        them after that flush; ``clear_buffers()`` -- called by the worker
        right after sending -- empties them), so a consumer that outlives
        ``collect()`` should still serialize/copy the final chunk before
        clearing (exactly what pickling for a ``conn.send()`` already does --
        see ``_batched_worker_main``'s usage).
        """
        import random as _random

        from footballcoach.ai.ppo.ppo_trainer import _action_to_numpy

        n = len(self.envs)
        stats = [_new_episode_stats() for _ in range(n)]
        episode_reward_accum = [0.0] * n
        secondary_episode_reward_accum = [0.0] * n
        episode_comp_accum = [dict() for _ in range(n)]
        _replay_queue: list[int] = list(replay_seeds) if replay_seeds else []

        def _next_episode_seed() -> int:
            return _replay_queue.pop(0) if _replay_queue else _random.randint(0, 2**31 - 1)

        collected = 0
        _last_flush_at = 0  # only meaningful when chunk_steps is set
        _last_progress_at = 0  # only meaningful when progress_value is set
        while collected < n_steps:
            # --- 0. Figure out, per env, whether ANYTHING is due this round
            # (trainee or, when secondary batching is on, any secondary
            # player) -- a pure peek, same is_due_for_decision() check
            # _batch_due_decisions does internally, just done once upfront
            # here. For every such env, advance its match's state timers
            # (stamina drain/regen, inactive-tackled/controlling-ball
            # countdowns) RIGHT NOW, before prepare() encodes any
            # observation below -- Match.step() normally does this itself,
            # BEFORE processing orders/decisions, but a batched decision is
            # encoded externally, before env.step() (hence before its
            # internal Match.step() call) even runs. Without this, every
            # batched decision would see one tick's worth of stale timer
            # state relative to the identical unbatched path -- confirmed as
            # a real (if small) source of divergence between BatchedEnvGroup
            # and plain unbatched stepping. See Match.step()'s own
            # docstring for the full rationale. ---
            env_has_due_decision = [False] * n
            for i, env in enumerate(self.envs):
                _player, _ai = self._trainee_player_and_ai(env)
                if _ai.is_due_for_decision():
                    env_has_due_decision[i] = True
                elif self.secondary_trainer is not None:
                    for _p, sec_ai, _m in self._secondary_neural_candidates(env):
                        if sec_ai.is_due_for_decision():
                            env_has_due_decision[i] = True
                            break
            for i, env in enumerate(self.envs):
                if env_has_due_decision[i]:
                    env.match.advance_state_timers()

            # --- 1, 2 & 3. Prepare each DUE env's trainee decision for this
            # round (see docstring above for why not every env is
            # guaranteed to be due, and why is_due_for_decision() is checked
            # first rather than just calling prepare() and branching on
            # None), then ONE batched network call over the due envs only
            # (not-due envs' _precomputed_result stays None -- see docstring
            # above). ---
            trainee_candidates = [
                (*self._trainee_player_and_ai(env), env.match) for env in self.envs
            ]
            self._batch_due_decisions(
                trainee_candidates, self.trainer,
                deterministic, deterministic_decision, deterministic_direction,
            )

            # --- Same thing, for secondary/opponent players, only when the
            # caller opted in via secondary_trainer (see class docstring) --
            # None (default) leaves every secondary player fully unbatched,
            # exactly as before this was added. ---
            if self.secondary_trainer is not None:
                secondary_candidates = [
                    triple
                    for env in self.envs
                    for triple in self._secondary_neural_candidates(env)
                ]
                self._batch_due_decisions(
                    secondary_candidates, self.secondary_trainer,
                    deterministic, deterministic_decision, deterministic_direction,
                )

            # --- 4 & 5. Step every env; per-env bookkeeping (mirrors
            # rollout_worker.py._collect() exactly, once per env). ---
            for i, env in enumerate(self.envs):
                next_obs, reward, done, info = env.step(timers_already_advanced=env_has_due_decision[i])
                tr = env.last_trainee_transition
                if tr is None:
                    # Structurally shouldn't happen given the precomputed
                    # result is always set before env.step() runs (see
                    # assert above) -- kept for parity/robustness with
                    # rollout_worker.py's identical defensive branch rather
                    # than assuming it can truly never fire.
                    if done:
                        stats[i]["episode_rewards"].append(episode_reward_accum[i])
                        stats[i]["episode_seeds"].append(self._current_episode_seed[i])
                        episode_reward_accum[i] = 0.0
                        stats[i]["episode_outcome_labels"].append(
                            info.trial_outcome if (info is not None and info.trial_outcome is not None) else "unknown"
                        )
                        _seed = _next_episode_seed()
                        self._current_episode_seed[i] = _seed
                        self._last_obs[i] = env.reset(seed=_seed)
                    else:
                        self._last_obs[i] = next_obs
                    continue

                bc_label_arr = tr.get("bc_label")
                self._buffers[i].add(
                    obs=tr["obs"],
                    action=_action_to_numpy(tr["action"], tr["raw_exec"]),
                    log_prob=float(tr["log_prob"]),
                    value=float(tr["value"]),
                    reward=reward,
                    done=1.0 if done else 0.0,
                    bc_label=bc_label_arr,
                    head_log_probs=tr.get("head_log_probs"),
                    reward_comps=dict(getattr(env, "last_reward_components", {})),
                    step_outcome=(info.trial_outcome or "") if (done and info is not None) else "",
                )
                collected += 1

                for sec in getattr(env, "last_secondary_results", []):
                    self._buffers[i].add(
                        obs=sec["obs"],
                        action=_action_to_numpy(sec["action"], sec["raw_exec"]),
                        log_prob=sec["log_prob"],
                        value=sec["value"],
                        reward=sec["reward"],
                        done=sec["done"],
                        bc_label=None,
                        # See ppo_trainer.py's single-process rollout loop's
                        # matching secondary buffer.add() call for why this
                        # is real data (from NeuralPlayerAI.last_transition,
                        # same for trainee and secondary players), not a
                        # placeholder -- omitting it silently corrupted every
                        # per-head KL/policy_loss diagnostic for every
                        # secondary/opponent row via RolloutBuffer.add()'s
                        # all-zero default.
                        head_log_probs=sec.get("head_log_probs"),
                        weight=self.trainer._secondary_weight,
                        track_id=sec["player_id"],
                        # Shared env-level episode outcome -- see
                        # scenario_env.py's last_secondary_results["step_outcome"]
                        # and ai/knowledge.md "unknown outcome bucket".
                        step_outcome=sec.get("step_outcome", ""),
                    )
                    secondary_episode_reward_accum[i] += sec["reward"]
                    if sec["done"]:
                        stats[i]["secondary_episode_rewards"].append(secondary_episode_reward_accum[i])
                        secondary_episode_reward_accum[i] = 0.0
                    collected += 1

                episode_reward_accum[i] += reward
                for _k, _v in getattr(env, "last_reward_components", {}).items():
                    episode_comp_accum[i][_k] = episode_comp_accum[i].get(_k, 0.0) + _v

                if done:
                    stats[i]["episode_rewards"].append(episode_reward_accum[i])
                    stats[i]["episode_seeds"].append(self._current_episode_seed[i])
                    episode_reward_accum[i] = 0.0
                    stats[i]["episode_outcome_labels"].append(
                        info.trial_outcome if (info is not None and info.trial_outcome is not None) else "unknown"
                    )
                    if episode_comp_accum[i]:
                        stats[i]["episode_comp_list"].append(dict(episode_comp_accum[i]))
                    episode_comp_accum[i] = {}
                    if info is not None and info.trial_outcome is not None:
                        if info.is_rules_episode:
                            stats[i]["episode_outcomes_vs_rules"].append(info.trial_outcome)
                        elif info.is_immobile_episode:
                            stats[i]["episode_outcomes_vs_immobile"].append(info.trial_outcome)
                        else:
                            stats[i]["episode_outcomes_vs_neural"].append(info.trial_outcome)
                    if info is not None:
                        stats[i]["episode_durations_s"].append(info.ticks_elapsed * env._dt_s)
                    _seed = _next_episode_seed()
                    self._current_episode_seed[i] = _seed
                    self._last_obs[i] = env.reset(seed=_seed)
                else:
                    self._last_obs[i] = next_obs

            if progress_value is not None:
                with progress_value.get_lock():
                    progress_value.value += collected - _last_progress_at
                _last_progress_at = collected

            # --- Periodic flush (chunked streaming, opt-in) -- checked once
            # per round, never mid-round, so a flush always lands on a clean
            # "every env has stepped once" boundary. WHOLE EPISODES ONLY: see
            # this module's top docstring ("Chunked streaming"). If nothing
            # is sendable yet, _last_flush_at is left alone so the very next
            # round retries. ---
            if chunk_steps and (collected - _last_flush_at) >= chunk_steps:
                chunk = self._pop_complete_episode_results(range(n), stats)
                if chunk:
                    on_chunk(chunk)
                    _last_flush_at = collected

        # Bootstrap value per env (never concatenate raw transitions across
        # envs before this -- see _train_parallel()'s identical per-worker
        # discipline).
        if chunk_steps:
            # Final flush of whatever's left since the last periodic one --
            # everything has already been (or is about to be) delivered via
            # on_chunk, so there's nothing left to return.
            chunk = self._build_results(range(n), stats)
            if chunk:
                on_chunk(chunk)
            return []
        return self._build_results(range(n), stats)

    def _pop_complete_episode_results(self, indices, stats: list[dict]) -> list[dict]:
        """Whole-episode counterpart of ``_build_results`` for a MID-collection
        flush: for each given env, split off its completed episodes into a new
        buffer (``RolloutBuffer.pop_complete_episodes``), leave the unfinished
        tail in place, and hand over that env's stats (resetting ``stats[i]``
        to a fresh dict so the sent one is never mutated again). Envs with no
        completed episode yet are skipped and keep both their rows and their
        stats. ``last_value`` is a literal 0.0 -- the popped buffer ends on a
        ``done=1`` row, so ``compute_gae`` never reads it -- which also means
        no value-net bootstrap forward pass per flush (unlike
        ``_build_results``)."""
        out = []
        for i in indices:
            popped = self._buffers[i].pop_complete_episodes()
            if popped is None:
                continue
            out.append({"buffer": popped, "last_value": 0.0, "stats": stats[i]})
            stats[i] = _new_episode_stats()
        return out

    def _build_results(self, indices, stats: list[dict]) -> list[dict]:
        """Build the ``{"buffer", "last_value", "stats"}`` result dict for
        each given env index -- shared by ``collect()``'s final return AND
        its periodic ``chunk_steps`` flush. ``stats`` is that call's own
        per-env stats list (indexed the same way as ``self.envs``). Skips
        any index whose buffer is currently empty (nothing collected for
        that env since the last flush/start) -- an empty buffer would crash
        ``RolloutBuffer.as_tensors()``'s ``self.obs[0]`` indexing, and can
        legitimately happen at a flush boundary if an env's episode just
        ended right before its next decision was due."""
        out = []
        for i in indices:
            if len(self._buffers[i]) == 0:
                continue
            env = self.envs[i]
            last_value = self.trainer._bootstrap_last_values(env, self._last_obs[i], self._buffers[i])
            out.append({"buffer": self._buffers[i], "last_value": last_value, "stats": stats[i]})
        return out

    def clear_buffers(self) -> None:
        for b in self._buffers:
            b.clear()


def _batch_to_wire(batch: dict) -> dict:
    """``as_tensors()`` dict -> same dict with torch tensors replaced by numpy
    views (zero-copy). Numpy arrays pickle by value through the pipe, so no
    torch shared-memory reductions (``ForkingPickler`` registers those for
    ``torch.Tensor``) are involved; the lists (``reward_comps_raw``,
    ``step_outcomes``, ``track_ids``) pass through unchanged."""
    return {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def batch_from_wire(d: dict) -> dict:
    """Inverse of ``_batch_to_wire`` (numpy -> torch via ``from_numpy``,
    zero-copy). Call in the RECEIVING process on a finalized result's
    ``"batch"``."""
    import numpy as _np

    return {k: (torch.from_numpy(v) if isinstance(v, _np.ndarray) else v) for k, v in d.items()}


def finalize_result_for_wire(r: dict, returns_spec: dict) -> dict:
    """Finalize ONE per-env result inside the worker: compute its returns
    (``returns_spec["mode"]``: ``"gae"`` = GAE with the result's own
    ``last_value``, no truncation; ``"mc"`` = truncate the trailing partial
    episode then pure MC -- exactly ``_finalize_value_pretrain_result``'s
    contract) using the CALLER's ``gamma``/``lam``, pack it with
    ``as_tensors``, and convert to numpy for the pipe. Returns
    ``{"batch", "stats", "n_dropped"}``; the receiver applies
    ``batch_from_wire`` to ``"batch"``. ``"batch"`` is ``None`` when MC mode
    had nothing usable in this result (a buffer with no completed episode --
    e.g. an env's final flush that is only its unfinished tail): its rows are
    all counted in ``n_dropped`` and the receiver must skip the batch (its
    ``stats`` are still valid and must still be folded in)."""
    from footballcoach.ai.ppo.ppo_trainer import _finalize_value_pretrain_result

    tensors, n_dropped = _finalize_value_pretrain_result(
        r, float(returns_spec["gamma"]), float(returns_spec["lam"]), returns_spec["mode"] == "gae",
        allow_empty=True,
    )
    return {
        "batch": _batch_to_wire(tensors) if tensors is not None else None,
        "stats": r["stats"], "n_dropped": n_dropped,
    }


def _batched_worker_main(
    conn, phase_id: int, base_seed: int, worker_idx: int, envs_per_process: int,
    separate_value_net: bool = False, worker_torch_threads: int = 1,
    batch_secondary_players: bool = False, chunk_steps: Optional[int] = None,
    progress_value=None,
) -> None:
    """Entry point run inside each persistent batched-worker process. Must
    stay picklable/top-level (mirrors rollout_worker.py's ``_worker_main``,
    including the critical BLAS-threading env-var ordering)."""
    import os
    _t = str(max(1, worker_torch_threads))
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = _t

    import random

    import torch as _torch

    from footballcoach.ai.curriculum.envs import build_env, bc_label_fn_for_phase_player
    from footballcoach.ai.curriculum.phases import PHASES_BY_ID
    from footballcoach.ai.ppo.ppo_trainer import PPOTrainer, _load_state_dict_tolerant

    _torch.set_num_threads(max(1, worker_torch_threads))

    phase = PHASES_BY_ID[phase_id]
    bc_label_fn = bc_label_fn_for_phase_player(phase_id)

    trainer = PPOTrainer.from_config(
        device=_torch.device("cpu"), inference_only=True, separate_value_net=separate_value_net,
    )
    if phase.frozen_heads:
        trainer.set_frozen_heads(phase.frozen_heads)

    envs = []
    for j in range(envs_per_process):
        seed = base_seed + j
        _torch.manual_seed(seed)
        random.seed(seed)
        env = build_env(phase)
        env.sample_action_fn = trainer._sample_action
        env.bc_label_fn = bc_label_fn
        envs.append(env)

    group = BatchedEnvGroup(
        envs, trainer,
        secondary_trainer=trainer if batch_secondary_players else None,
    )

    while True:
        try:
            msg = conn.recv()
        except (EOFError, KeyboardInterrupt):
            break
        cmd = msg.get("cmd")
        if cmd == "collect":
            # Uniform protocol regardless of chunking: zero-or-more
            # {"chunk": [...]} messages, always followed by exactly one
            # {"done": True} -- see this module's top docstring ("Chunked
            # streaming") for why (fixes a real MemoryError from pickling
            # one giant end-of-rollout result all at once).
            returns_spec = msg.get("returns")

            def _emit(chunk: list, _spec=returns_spec) -> None:
                # Opt-in worker-side finalization (see this module's
                # "Worker-side finalization" docstring paragraph): only the
                # finished flat arrays cross the pipe, not the per-row buffer.
                if _spec is not None:
                    chunk = [finalize_result_for_wire(r, _spec) for r in chunk]
                conn.send({"chunk": chunk})

            if chunk_steps:
                group.collect(
                    msg["n_steps"], replay_seeds=msg.get("replay_seeds"),
                    chunk_steps=chunk_steps,
                    on_chunk=_emit,
                    progress_value=progress_value,
                )
            else:
                results = group.collect(
                    msg["n_steps"], replay_seeds=msg.get("replay_seeds"),
                    progress_value=progress_value,
                )
                if results:
                    _emit(results)
            conn.send({"done": True})
            group.clear_buffers()  # harmless no-op if chunking already cleared everything
        elif cmd == "set_weights":
            _load_state_dict_tolerant(trainer.decision_net, msg["decision_net"], "decision_net")
            _load_state_dict_tolerant(trainer.execution_net, msg["execution_net"], "execution_net")
            if msg.get("value_net") is not None and trainer.value_net is not None:
                _load_state_dict_tolerant(trainer.value_net, msg["value_net"], "value_net")
            conn.send({"ok": True})
        elif cmd == "close":
            break
        else:
            log.warning(f"batched_rollout_worker[{worker_idx}]: unknown cmd {cmd!r}")
    conn.close()


@dataclass
class BatchedRolloutWorkerHandle:
    """Main-process handle to one spawned batched-worker process (owns
    ``envs_per_process`` environments internally)."""
    process: "mp.process.BaseProcess"
    conn: "mp.connection.Connection"
    worker_idx: int

    def collect(
        self, n_steps: int, replay_seeds: Optional[list[int]] = None,
        returns: Optional[dict] = None,
    ) -> None:
        """``returns`` (optional): ``{"mode": "gae" | "mc", "gamma": float,
        "lam": float}`` -- ask the worker to finalize each result (GAE/MC
        returns + ``as_tensors``) itself and send compact flat arrays
        instead of per-row buffers; see ``finalize_result_for_wire`` /
        ``batch_from_wire``. ``gamma``/``lam`` are passed explicitly (the
        caller's values are the source of truth -- a worker reads its own
        ai_config.json at spawn time, which can drift from the main process's
        if the file is edited mid-run). None (default) = legacy wire format
        (``{"buffer", "last_value", "stats"}``)."""
        self.conn.send({
            "cmd": "collect", "n_steps": n_steps, "replay_seeds": replay_seeds, "returns": returns,
        })

    def recv_result(self) -> list:
        return self.conn.recv()

    def set_weights(self, decision_state: dict, execution_state: dict, value_state: Optional[dict]) -> None:
        self.conn.send({
            "cmd": "set_weights",
            "decision_net": decision_state,
            "execution_net": execution_state,
            "value_net": value_state,
        })
        self.conn.recv()  # block until applied, keeps weight sync deterministic

    def close(self) -> None:
        try:
            self.conn.send({"cmd": "close"})
        except (BrokenPipeError, OSError):
            pass
        self.process.join(timeout=5.0)
        if self.process.is_alive():
            self.process.terminate()


def spawn_batched_workers(
    phase_id: int, n_processes: int, envs_per_process: int, base_seed: int,
    separate_value_net: bool = False, worker_torch_threads: int = 1,
    batch_secondary_players: bool = False, chunk_steps: Optional[int] = None,
    progress_value=None,
) -> list[BatchedRolloutWorkerHandle]:
    """Spawn ``n_processes`` batched-worker processes, each internally
    owning ``envs_per_process`` environments (total envs =
    ``n_processes * envs_per_process``). ``envs_per_process=1`` reduces to
    exactly today's ``rollout_worker.py`` topology (for empirical A/B
    comparison, at the same ``n_processes``); ``n_processes=1`` is the
    "everything in one process" extreme this module was built to test;
    anything in between is a tunable hybrid. ``chunk_steps``: see
    ``_batched_worker_main``/``BatchedEnvGroup.collect()``'s "Chunked
    streaming" docstring -- fixed for the life of this worker.

    ``progress_value``: optional shared ``ctx.Value("l", 0)`` that every
    worker adds its collected rows to (see ``BatchedEnvGroup.collect()``'s
    ``progress_value`` docstring). Must be created from the SAME
    ``mp.get_context("spawn")`` used here and passed at process-creation
    time -- a raw ``Synchronized`` value can't be sent to an already-running
    worker afterwards. The caller resets it to 0 before each collect.
    ``None`` (default) = no effect (the main PPO loop never passes it).
    """
    ctx = mp.get_context("spawn")
    handles: list[BatchedRolloutWorkerHandle] = []
    for i in range(n_processes):
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(
            target=_batched_worker_main,
            args=(
                child_conn, phase_id, base_seed + i * envs_per_process, i, envs_per_process,
                separate_value_net, worker_torch_threads, batch_secondary_players, chunk_steps,
                progress_value,
            ),
            daemon=True,
        )
        proc.start()
        handles.append(BatchedRolloutWorkerHandle(process=proc, conn=parent_conn, worker_idx=i))
    return handles


def close_batched_workers(handles: list[BatchedRolloutWorkerHandle]) -> None:
    for h in handles:
        h.close()
