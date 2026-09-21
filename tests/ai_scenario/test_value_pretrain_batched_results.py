"""Coverage for the batched value-pretrain rollout path
(``PPOTrainer._spawn_value_pretrain_workers`` /
``_collect_value_pretrain_rollout(pool=...)`` /
``_finalize_value_pretrain_result`` in ai/ppo/ppo_trainer.py, plus the
optional ``progress_value`` counter added to ai/ppo/batched_rollout_worker.py).

No test in this repo spawns real subprocesses, so nothing here does either:

1. ``_finalize_value_pretrain_result`` is a pure function over real
   ``BatchedEnvGroup.collect()`` results (built in-process, same fixture
   pattern as test_episode_replay.py) -- MC / GAE-with-bootstrap / GAE-
   without-bootstrap paths.
2. ``BatchedEnvGroup.collect(progress_value=...)`` adds exactly the rows it
   collected to a plain ``multiprocessing.Value`` (default None = unchanged).
3. ``_spawn_value_pretrain_workers`` / ``_close_value_pretrain_workers``
   worker-kind selection and argument wiring, with the real spawn functions
   monkeypatched to recorders.
4. The full ``_collect_value_pretrain_rollout`` consumption loop (batched
   AND plain protocol) driven by fake worker handles that speak the real
   wire protocol over real in-process ``multiprocessing.Pipe``s -- exercises
   ``connection.wait``, chunk/done handling, finalize + merge, the parent-
   side progress reset, and that a caller-owned pool is NOT closed.
"""
from __future__ import annotations

import copy
import functools
import multiprocessing
import threading

import pytest
import torch

from footballcoach.ai.curriculum.phases import PHASES_BY_ID
from footballcoach.ai.env.scenario_env import ScenarioEnv
import pickle

from footballcoach.ai.ppo.batched_rollout_worker import (
    BatchedEnvGroup,
    batch_from_wire,
    finalize_result_for_wire,
)
from footballcoach.ai.ppo.ppo_trainer import (
    PPOTrainer,
    _ValuePretrainWorkers,
    _decode_rollout_result,
    _finalize_value_pretrain_result,
    _merge_worker_batches,
)
from footballcoach.ui.scenarios import ScenarioDefinition, build_1v1_scenario, phase1_training_on_tick

_PHASE = PHASES_BY_ID[1]
# Short on purpose: episodes end within a handful of decisions, so a modest
# collect(n_steps=...) reliably yields several completed episodes per env AND
# a trailing incomplete one (which is what the truncate/bootstrap paths differ on).
_SHORT_MAX_EPISODE_S = 2.0
_GAMMA = 0.99
_LAM = 0.95


@pytest.fixture(scope="module")
def trainer():
    t = PPOTrainer.from_config(device=torch.device("cpu"), inference_only=True, separate_value_net=True)
    if _PHASE.frozen_heads:
        t.set_frozen_heads(_PHASE.frozen_heads)
    return t


def _make_short_envs(trainer, n: int) -> list[ScenarioEnv]:
    envs = []
    for _ in range(n):
        defn = ScenarioDefinition(
            key="value_pretrain_batched_test", label="Value-pretrain batched test scenario",
            description="1v1, short episodes, rules-based opponent -- config-independent",
            build=functools.partial(
                build_1v1_scenario, opponent_rules_prob=1.0, opponent_immobile_prob=0.0,
            ),
            on_tick=phase1_training_on_tick,
        )
        env = ScenarioEnv(
            definition=defn, trainee_player_id="trainee", phase=1,
            secondary_player_ids=["opponent"], max_episode_s=_SHORT_MAX_EPISODE_S,
        )
        env.sample_action_fn = trainer._sample_action
        envs.append(env)
    return envs


def _collect_results(trainer, n_envs: int = 3, n_steps: int = 300) -> list[dict]:
    group = BatchedEnvGroup(_make_short_envs(trainer, n_envs), trainer)
    return group.collect(n_steps=n_steps)


class TestFinalizeValuePretrainResult:
    def test_mc_mode_truncates_to_last_done_and_matches_reference_mc(self, trainer):
        for r in _collect_results(trainer):
            ref = copy.deepcopy(r["buffer"])
            expected_dropped = ref.truncate_to_last_episode_end()
            expected_returns = ref.compute_mc_returns(_GAMMA)
            n_before = len(r["buffer"])

            tensors, n_dropped = _finalize_value_pretrain_result(r, _GAMMA, _LAM, use_gae=False)

            assert n_dropped == expected_dropped
            assert len(tensors["returns"]) == n_before - n_dropped
            assert tensors["returns"].tolist() == pytest.approx(expected_returns)
            # A buffer with at least one completed episode must end exactly on it.
            if float(tensors["dones"].sum()) > 0:
                assert float(tensors["dones"][-1]) == 1.0

    def test_gae_with_bootstrap_keeps_every_row_and_uses_real_last_value(self, trainer):
        saw_trailing_partial = False
        for r in _collect_results(trainer):
            assert r.get("last_value") is not None, "batched results always carry a real bootstrap value"
            ref = copy.deepcopy(r["buffer"])
            expected_adv, expected_ret = ref.compute_gae(_GAMMA, _LAM, r["last_value"])
            n_before = len(r["buffer"])
            ended_mid_episode = float(r["buffer"].dones[-1]) < 0.5
            saw_trailing_partial = saw_trailing_partial or ended_mid_episode

            tensors, n_dropped = _finalize_value_pretrain_result(r, _GAMMA, _LAM, use_gae=True)

            assert n_dropped == 0, "GAE-with-bootstrap must NOT truncate -- the tail is correctly bootstrapped"
            assert len(tensors["returns"]) == n_before
            assert tensors["returns"].tolist() == pytest.approx(expected_ret)
            assert tensors["advantages"].tolist() == pytest.approx(expected_adv)
        assert saw_trailing_partial, (
            "fixture should produce at least one env ending mid-episode -- otherwise this test "
            "can't distinguish 'kept the tail' from 'nothing to keep'"
        )

    def test_gae_without_last_value_falls_back_to_truncate_then_zero_bootstrap(self, trainer):
        for r in _collect_results(trainer):
            bare = {"buffer": r["buffer"]}  # what the single-process branch passes
            ref = copy.deepcopy(r["buffer"])
            expected_dropped = ref.truncate_to_last_episode_end()
            _, expected_ret = ref.compute_gae(_GAMMA, _LAM, 0.0)

            tensors, n_dropped = _finalize_value_pretrain_result(bare, _GAMMA, _LAM, use_gae=True)

            assert n_dropped == expected_dropped
            assert tensors["returns"].tolist() == pytest.approx(expected_ret)

    def test_output_keys_match_plain_as_tensors_and_merge_cleanly(self, trainer):
        results = _collect_results(trainer)
        reference_keys = set(copy.deepcopy(results[0]["buffer"]).as_tensors(
            [0.0] * len(results[0]["buffer"]), [0.0] * len(results[0]["buffer"]),
        ).keys())
        batches = []
        for use_gae in (False, True):
            for r in copy.deepcopy(results):
                tensors, _ = _finalize_value_pretrain_result(r, _GAMMA, _LAM, use_gae)
                assert set(tensors.keys()) == reference_keys
                batches.append(tensors)
        merged = _merge_worker_batches(batches)
        assert len(merged["returns"]) == sum(len(b["returns"]) for b in batches)
        assert len(merged["step_outcomes"]) == len(merged["returns"])


class TestBatchedCollectProgressValue:
    def test_progress_value_receives_exactly_the_collected_rows(self, trainer):
        progress = multiprocessing.Value("l", 0)
        group = BatchedEnvGroup(_make_short_envs(trainer, 3), trainer)
        results = group.collect(n_steps=200, progress_value=progress)
        total_rows = sum(len(r["buffer"]) for r in results)
        assert progress.value == total_rows
        assert progress.value >= 200

    def test_progress_value_defaults_to_no_effect(self, trainer):
        group = BatchedEnvGroup(_make_short_envs(trainer, 2), trainer)
        results = group.collect(n_steps=100)  # no progress_value -- must just work
        assert sum(len(r["buffer"]) for r in results) >= 100


class TestSpawnAndCloseWiring:
    def test_returns_none_when_parallel_collection_does_not_apply(self, trainer):
        old = trainer.value_pretrain_n_processes
        try:
            trainer.value_pretrain_n_processes = 4
            assert trainer._spawn_value_pretrain_workers(phase_id=None) is None
            trainer.value_pretrain_n_processes = 1
            assert trainer._spawn_value_pretrain_workers(phase_id=1) is None
        finally:
            trainer.value_pretrain_n_processes = old

    def test_close_is_a_noop_for_none(self, trainer):
        trainer._close_value_pretrain_workers(None)  # must not raise

    def test_batched_flag_selects_batched_workers_with_expected_args(self, trainer, monkeypatch):
        import footballcoach.ai.ppo.batched_rollout_worker as bw

        calls: dict = {}

        def fake_spawn(*args, **kwargs):
            calls["args"], calls["kwargs"] = args, kwargs
            return ["h0", "h1"]

        closed: list = []
        monkeypatch.setattr(bw, "spawn_batched_workers", fake_spawn)
        monkeypatch.setattr(bw, "close_batched_workers", lambda handles: closed.append(list(handles)))
        monkeypatch.setattr(trainer, "value_pretrain_n_processes", 2)
        monkeypatch.setattr(trainer, "value_pretrain_batched_rollout", True)
        monkeypatch.setattr(trainer, "value_pretrain_envs_per_process", 5)
        monkeypatch.setattr(trainer, "batch_secondary_players", True)

        pool = trainer._spawn_value_pretrain_workers(phase_id=1)

        assert isinstance(pool, _ValuePretrainWorkers)
        assert pool.batched is True and pool.n_processes == 2 and pool.envs_per_process == 5
        assert pool.handles == ["h0", "h1"] and pool.progress_value is not None
        # (phase_id, n_processes, envs_per_process, base_seed, separate_value_net, worker_torch_threads)
        assert calls["args"][0] == 1 and calls["args"][1] == 2 and calls["args"][2] == 5
        assert calls["kwargs"]["batch_secondary_players"] is True
        assert calls["kwargs"]["chunk_steps"] == (trainer.value_pretrain_chunk_steps or None), (
            "value pretrain streams whole-episode chunks of ppo.value_pretrain_chunk_steps rows"
        )
        assert calls["kwargs"]["progress_value"] is pool.progress_value

        trainer._close_value_pretrain_workers(pool)
        assert closed == [["h0", "h1"]]

    @pytest.mark.parametrize("configured,expected", [(1234, 1234), (0, None)])
    def test_chunk_steps_config_is_passed_through_and_zero_disables(self, trainer, monkeypatch, configured, expected):
        import footballcoach.ai.ppo.batched_rollout_worker as bw

        seen: dict = {}
        monkeypatch.setattr(bw, "spawn_batched_workers", lambda *a, **k: seen.update(k) or ["h"])
        monkeypatch.setattr(trainer, "value_pretrain_n_processes", 1 + 1)
        monkeypatch.setattr(trainer, "value_pretrain_batched_rollout", True)
        monkeypatch.setattr(trainer, "value_pretrain_chunk_steps", configured)

        trainer._spawn_value_pretrain_workers(phase_id=1)

        assert seen["chunk_steps"] == expected

    def test_flag_off_selects_plain_workers(self, trainer, monkeypatch):
        import footballcoach.ai.ppo.rollout_worker as rw

        calls: dict = {}

        def fake_spawn(*args, **kwargs):
            calls["args"], calls["kwargs"] = args, kwargs
            return ["p0", "p1", "p2"]

        closed: list = []
        monkeypatch.setattr(rw, "spawn_workers", fake_spawn)
        monkeypatch.setattr(rw, "close_workers", lambda handles: closed.append(list(handles)))
        monkeypatch.setattr(trainer, "value_pretrain_n_processes", 3)
        monkeypatch.setattr(trainer, "value_pretrain_batched_rollout", False)

        pool = trainer._spawn_value_pretrain_workers(phase_id=1)

        assert pool.batched is False and pool.n_processes == 3 and pool.envs_per_process == 1
        assert calls["kwargs"]["progress_value"] is pool.progress_value
        trainer._close_value_pretrain_workers(pool)
        assert closed == [["p0", "p1", "p2"]]


class _FakeHandle:
    """Speaks the real worker wire protocol over a real Pipe, from a thread
    (a big pickled message sent from the SAME thread that later reads would
    block on the pipe buffer, exactly why real workers are separate processes)."""

    def __init__(self, messages: list, batched: bool):
        self.parent_conn, self._child_conn = multiprocessing.Pipe()
        self.conn = self.parent_conn
        self._messages = messages
        self._batched = batched
        self.weights_calls = 0
        self.collect_calls: list = []
        self.returns_specs: list = []
        self.deterministic_flags: list = []

    def set_weights(self, dec, exec_, val):
        self.weights_calls += 1

    def collect(self, n_steps, progress=None, returns=None, deterministic=False):
        self.collect_calls.append(n_steps)
        self.returns_specs.append(returns)
        self.deterministic_flags.append(deterministic)

        def _send():
            for m in self._messages:
                self._child_conn.send(m)

        threading.Thread(target=_send, daemon=True).start()

    def recv_result(self):
        return self.parent_conn.recv()


class TestCollectConsumptionLoop:
    def _pool(self, handles, batched: bool, envs_per_process: int = 3):
        return _ValuePretrainWorkers(
            handles=handles, batched=batched, progress_value=multiprocessing.Value("l", 123),
            n_processes=len(handles), envs_per_process=envs_per_process,
        )

    def test_batched_protocol_end_to_end(self, trainer, monkeypatch):
        per_worker = [_collect_results(trainer, n_envs=2, n_steps=200) for _ in range(2)]
        expected_rows = sum(len(r["buffer"]) for results in per_worker for r in results)
        handles = [
            _FakeHandle([{"chunk": results}, {"done": True}], batched=True) for results in per_worker
        ]
        pool = self._pool(handles, batched=True, envs_per_process=2)
        closed: list = []
        monkeypatch.setattr(trainer, "_close_value_pretrain_workers", lambda p: closed.append(p))

        batch, stats = trainer._collect_value_pretrain_rollout(
            env=None, n_steps=400, phase_id=1, use_gae=True, pool=pool,
        )

        # GAE-with-bootstrap keeps every row from every env result.
        assert len(batch["returns"]) == expected_rows
        assert all(h.weights_calls == 1 for h in handles)
        assert all(h.collect_calls == [200] for h in handles), "steps split evenly across processes"
        assert pool.progress_value.value == 0, "parent must reset the shared counter before collecting"
        assert closed == [], "a caller-owned pool must NOT be closed by the collector"
        assert len(stats["episode_returns"]) > 0

    def test_batched_protocol_mc_mode_drops_only_trailing_partials(self, trainer, monkeypatch):
        results = _collect_results(trainer, n_envs=3, n_steps=300)
        expected = sum(
            len(r["buffer"]) - copy.deepcopy(r["buffer"]).truncate_to_last_episode_end() for r in results
        )
        handle = _FakeHandle([{"chunk": results}, {"done": True}], batched=True)
        pool = self._pool([handle], batched=True, envs_per_process=3)
        monkeypatch.setattr(trainer, "_close_value_pretrain_workers", lambda p: None)

        batch, _ = trainer._collect_value_pretrain_rollout(
            env=None, n_steps=300, phase_id=1, use_gae=False, pool=pool,
        )

        assert len(batch["returns"]) == expected
        assert float(batch["dones"][-1]) == 1.0

    def test_plain_protocol_still_works(self, trainer, monkeypatch):
        # rollout_worker.py replies with ONE result dict per worker (no chunk/done).
        results = _collect_results(trainer, n_envs=2, n_steps=200)
        handles = [_FakeHandle([r], batched=False) for r in results]
        pool = self._pool(handles, batched=False, envs_per_process=1)
        monkeypatch.setattr(trainer, "_close_value_pretrain_workers", lambda p: None)

        batch, _ = trainer._collect_value_pretrain_rollout(
            env=None, n_steps=200, phase_id=1, use_gae=False, pool=pool,
        )

        assert len(batch["returns"]) > 0
        assert all(h.collect_calls == [100] for h in handles)

    def test_owned_pool_is_closed_even_when_collection_fails(self, trainer, monkeypatch):
        class _Boom(_FakeHandle):
            def set_weights(self, *a):
                raise RuntimeError("worker died")

        pool = self._pool([_Boom([], batched=True)], batched=True)
        closed: list = []
        monkeypatch.setattr(trainer, "_spawn_value_pretrain_workers", lambda phase_id: pool)
        monkeypatch.setattr(trainer, "_close_value_pretrain_workers", lambda p: closed.append(p))

        with pytest.raises(RuntimeError, match="worker died"):
            trainer._collect_value_pretrain_rollout(env=None, n_steps=100, phase_id=1)

        assert closed == [pool], "a pool the collector spawned itself must be closed in a finally"


def _spec(mode: str) -> dict:
    return {"mode": mode, "gamma": _GAMMA, "lam": _LAM}


def _assert_tensor_dicts_equal(a: dict, b: dict) -> None:
    assert set(a.keys()) == set(b.keys())
    for k in a:
        if isinstance(a[k], torch.Tensor):
            assert torch.equal(a[k], b[k]), k
        else:
            assert a[k] == b[k], k


class TestWorkerSideFinalizationWireFormat:
    @pytest.mark.parametrize("mode", ["gae", "mc"])
    def test_wire_result_round_trips_to_the_same_batch_as_local_finalization(self, trainer, mode):
        for r in _collect_results(trainer):
            local_tensors, local_dropped = _finalize_value_pretrain_result(
                copy.deepcopy(r), _GAMMA, _LAM, use_gae=(mode == "gae"),
            )
            wire = finalize_result_for_wire(r, _spec(mode))

            assert wire["n_dropped"] == local_dropped
            assert set(wire.keys()) == {"batch", "stats", "n_dropped"}
            # numpy (not torch) on the wire -- no torch shared-memory reductions involved
            assert not any(isinstance(v, torch.Tensor) for v in wire["batch"].values())
            # must survive a real pickle round trip (this is what conn.send does)
            wire_after_pipe = pickle.loads(pickle.dumps(wire))
            _assert_tensor_dicts_equal(batch_from_wire(wire_after_pipe["batch"]), local_tensors)

    def test_caller_gamma_and_lam_are_used_not_a_default(self, trainer):
        r = _collect_results(trainer, n_envs=1, n_steps=200)[0]
        a = batch_from_wire(finalize_result_for_wire(copy.deepcopy(r), {"mode": "gae", "gamma": 0.5, "lam": 0.5})["batch"])
        b = batch_from_wire(finalize_result_for_wire(copy.deepcopy(r), {"mode": "gae", "gamma": 0.99, "lam": 0.95})["batch"])
        assert not torch.allclose(a["returns"], b["returns"]), "gamma/lam in the spec must actually drive the returns"

    @pytest.mark.parametrize("want_replay_inputs", [True, False])
    def test_main_loop_decode_is_identical_for_legacy_and_worker_finalized_results(self, trainer, want_replay_inputs):
        """The correctness claim of worker-side finalization for the MAIN
        PPO loop: what _train_batched_parallel gets is the same whether the
        main process computed GAE from the buffer (legacy) or the worker did."""
        for r in _collect_results(trainer):
            legacy = _decode_rollout_result(copy.deepcopy(r), _GAMMA, _LAM, want_replay_inputs)
            wire = finalize_result_for_wire(copy.deepcopy(r), _spec("gae"))
            finalized = _decode_rollout_result(pickle.loads(pickle.dumps(wire)), _GAMMA, _LAM, want_replay_inputs)

            _assert_tensor_dicts_equal(legacy[0], finalized[0])
            assert legacy[2] == finalized[2], "track_ids"
            if want_replay_inputs:
                assert legacy[1] == pytest.approx(finalized[1]), "advantages list"
                assert list(legacy[3]) == pytest.approx(finalized[3]), "dones list"
            else:
                assert finalized[1] is None and finalized[3] is None


class TestCollectConsumesWorkerFinalizedChunks:
    def _pool(self, handles, envs_per_process: int = 2):
        return _ValuePretrainWorkers(
            handles=handles, batched=True, progress_value=multiprocessing.Value("l", 7),
            n_processes=len(handles), envs_per_process=envs_per_process,
        )

    @pytest.mark.parametrize("use_gae", [True, False])
    def test_finalized_stream_gives_the_same_batch_as_the_legacy_buffer_stream(self, trainer, monkeypatch, use_gae):
        # ONE worker: messages within a pipe are ordered, so the merged row
        # order is deterministic and the two streams can be compared exactly
        # (with several fake workers, which one's message arrives first races).
        per_worker = [_collect_results(trainer, n_envs=4, n_steps=300)]
        mode = "gae" if use_gae else "mc"
        monkeypatch.setattr(trainer, "_close_value_pretrain_workers", lambda p: None)

        # legacy: raw per-env buffers on the wire, finalized in the main process
        legacy_handles = [
            _FakeHandle([{"chunk": copy.deepcopy(results)}, {"done": True}], batched=True) for results in per_worker
        ]
        legacy_batch, legacy_stats = trainer._collect_value_pretrain_rollout(
            env=None, n_steps=400, phase_id=1, use_gae=use_gae, pool=self._pool(legacy_handles),
        )

        # new: workers finalize; the SAME rows arrive as several small chunks
        new_handles = []
        for results in per_worker:
            msgs = [
                {"chunk": [finalize_result_for_wire(copy.deepcopy(r), _spec(mode) | {"gamma": trainer.gamma, "lam": trainer.lam})]}
                for r in results
            ] + [{"done": True}]
            new_handles.append(_FakeHandle(msgs, batched=True))
        new_batch, new_stats = trainer._collect_value_pretrain_rollout(
            env=None, n_steps=400, phase_id=1, use_gae=use_gae, pool=self._pool(new_handles),
        )

        _assert_tensor_dicts_equal(new_batch, legacy_batch)
        assert sorted(new_stats["episode_returns"]) == pytest.approx(sorted(legacy_stats["episode_returns"]))
        # the collector must have asked the workers to finalize, with OUR gamma/lam
        for h in legacy_handles:
            assert h.returns_specs == [_spec(mode) | {"gamma": trainer.gamma, "lam": trainer.lam}]
        for h in new_handles:
            assert h.returns_specs[0]["mode"] == mode
            assert h.returns_specs[0]["gamma"] == trainer.gamma and h.returns_specs[0]["lam"] == trainer.lam

    def test_n_dropped_from_worker_finalized_mc_results_is_reported(self, trainer, monkeypatch, caplog):
        results = _collect_results(trainer, n_envs=2, n_steps=200)
        expected_dropped = sum(
            copy.deepcopy(r["buffer"]).truncate_to_last_episode_end() for r in results
        )
        msgs = [{"chunk": [finalize_result_for_wire(copy.deepcopy(r), _spec("mc"))]} for r in results] + [{"done": True}]
        monkeypatch.setattr(trainer, "_close_value_pretrain_workers", lambda p: None)
        import logging

        with caplog.at_level(logging.INFO, logger="footballcoach.ai.ppo"):
            batch, _ = trainer._collect_value_pretrain_rollout(
                env=None, n_steps=200, phase_id=1, use_gae=False,
                pool=self._pool([_FakeHandle(msgs, batched=True)]),
            )

        assert len(batch["returns"]) == sum(len(r["buffer"]) for r in results) - expected_dropped
        if expected_dropped:
            assert any(f"dropped {expected_dropped}" in rec.getMessage() for rec in caplog.records)


def _tail_only_result(trainer) -> tuple[dict, int]:
    """A real result whose buffer is ONLY an unfinished tail (no done row):
    exactly what an env's final flush looks like when it completed no episode
    since the previous flush. Returns (result, n_tail_rows)."""
    for r in _collect_results(trainer, n_envs=3, n_steps=300):
        head = r["buffer"].pop_complete_episodes()
        if head is not None and len(r["buffer"]) > 0:
            assert all(d < 0.5 for d in r["buffer"].dones)
            return r, len(r["buffer"])
    pytest.skip("fixture produced no env with both a completed episode and a trailing tail")


class TestTailOnlyResults:
    def test_mc_with_allow_empty_drops_a_tail_only_buffer(self, trainer):
        r, n_tail = _tail_only_result(trainer)
        tensors, n_dropped = _finalize_value_pretrain_result(r, _GAMMA, _LAM, use_gae=False, allow_empty=True)
        assert tensors is None and n_dropped == n_tail

    def test_default_keeps_rows_for_callers_that_cannot_handle_empty(self, trainer):
        r, n_tail = _tail_only_result(trainer)
        tensors, n_dropped = _finalize_value_pretrain_result(r, _GAMMA, _LAM, use_gae=False)
        assert tensors is not None and n_dropped == 0 and len(tensors["returns"]) == n_tail

    def test_gae_with_bootstrap_still_keeps_a_tail_only_buffer(self, trainer):
        r, n_tail = _tail_only_result(trainer)
        tensors, n_dropped = _finalize_value_pretrain_result(r, _GAMMA, _LAM, use_gae=True, allow_empty=True)
        assert tensors is not None and n_dropped == 0 and len(tensors["returns"]) == n_tail

    def test_wire_result_carries_none_batch_but_keeps_stats(self, trainer):
        r, n_tail = _tail_only_result(trainer)
        wire = finalize_result_for_wire(r, _spec("mc"))
        assert wire["batch"] is None and wire["n_dropped"] == n_tail
        assert "episode_rewards" in wire["stats"]
        pickle.loads(pickle.dumps(wire))  # still a valid message

    def test_collector_skips_none_batches_but_folds_their_stats(self, trainer, monkeypatch):
        good = finalize_result_for_wire(_collect_results(trainer, n_envs=1, n_steps=300)[0], _spec("mc"))
        assert good["batch"] is not None
        tail_result, n_tail = _tail_only_result(trainer)
        # marker: this result's stats must still be counted even though its batch is dropped.
        # (kept internally consistent -- one episode, one label -- since the collector logs
        # per-outcome reward tables that pair rewards with labels)
        tail_result["stats"].update({
            "episode_rewards": [12345.0], "episode_outcome_labels": ["timeout"],
            "episode_outcomes_vs_rules": ["timeout"], "episode_outcomes_vs_immobile": [],
            "episode_outcomes_vs_neural": [], "episode_comp_list": [], "episode_durations_s": [],
        })
        tail_wire = finalize_result_for_wire(tail_result, _spec("mc"))
        pool = _ValuePretrainWorkers(
            handles=[_FakeHandle([{"chunk": [good, tail_wire]}, {"done": True}], batched=True)],
            batched=True, progress_value=multiprocessing.Value("l", 0), n_processes=1, envs_per_process=2,
        )
        monkeypatch.setattr(trainer, "_close_value_pretrain_workers", lambda p: None)

        batch, stats = trainer._collect_value_pretrain_rollout(
            env=None, n_steps=300, phase_id=1, use_gae=False, pool=pool,
        )

        assert len(batch["returns"]) == len(good["batch"]["returns"]), "the dropped tail must not be in the batch"
        assert 12345.0 in stats["episode_returns"], "stats of a dropped-batch result must still be folded in"

    def test_collector_raises_clearly_when_nothing_usable_was_collected(self, trainer, monkeypatch):
        tail_result, _ = _tail_only_result(trainer)
        pool = _ValuePretrainWorkers(
            handles=[_FakeHandle([{"chunk": [finalize_result_for_wire(tail_result, _spec("mc"))]}, {"done": True}], batched=True)],
            batched=True, progress_value=multiprocessing.Value("l", 0), n_processes=1, envs_per_process=1,
        )
        monkeypatch.setattr(trainer, "_close_value_pretrain_workers", lambda p: None)
        with pytest.raises(RuntimeError, match="no usable rows"):
            trainer._collect_value_pretrain_rollout(env=None, n_steps=100, phase_id=1, use_gae=False, pool=pool)


class TestMCReturnsForCompareToOriginal:
    """``with_mc_returns`` (ppg_value_refit's compare-to-original diagnostic):
    pure-MC targets alongside the normal returns, NaN on each track's
    unfinished tail, independent of any value net."""

    def test_mc_returns_match_reference_and_are_nan_only_on_unfinished_tails(self, trainer):
        saw_nan = False
        for r in _collect_results(trainer):
            buf = r["buffer"]
            expected = buf.compute_mc_returns(_GAMMA)
            tail_start = {}
            for track, idxs in buf._track_index_groups().items():
                last_done = max((p for p, i in enumerate(idxs) if buf.dones[i] > 0.5), default=-1)
                tail_start[track] = {i for i in idxs[last_done + 1:]}
            tail_rows = set().union(*tail_start.values()) if tail_start else set()

            tensors, n_dropped = _finalize_value_pretrain_result(
                r, _GAMMA, _LAM, use_gae=True, with_mc_returns=True,
            )
            mc = tensors["mc_returns"]
            assert n_dropped == 0 and len(mc) == len(expected)
            for i, e in enumerate(expected):
                if i in tail_rows:
                    assert torch.isnan(mc[i]), f"unfinished-tail row {i} must be NaN"
                    saw_nan = True
                else:
                    assert float(mc[i]) == pytest.approx(e, rel=1e-5, abs=1e-6)
        assert saw_nan, "fixture should produce at least one unfinished tail"

    def test_absent_unless_requested(self, trainer):
        r = _collect_results(trainer, n_envs=1, n_steps=200)[0]
        tensors, _ = _finalize_value_pretrain_result(r, _GAMMA, _LAM, use_gae=True)
        assert "mc_returns" not in tensors

    def test_mc_returns_do_not_depend_on_the_recorded_values(self, trainer):
        """The whole point: unlike GAE returns, MC returns must not move when
        the value estimates stored in the buffer change."""
        r = _collect_results(trainer, n_envs=1, n_steps=300)[0]
        r2 = copy.deepcopy(r)
        r2["buffer"].values = [v + 5.0 for v in r2["buffer"].values]
        lv = r["last_value"]
        r2["last_value"] = {k: v + 5.0 for k, v in lv.items()} if isinstance(lv, dict) else float(lv) + 5.0
        a, _ = _finalize_value_pretrain_result(r, _GAMMA, _LAM, use_gae=True, with_mc_returns=True)
        b, _ = _finalize_value_pretrain_result(r2, _GAMMA, _LAM, use_gae=True, with_mc_returns=True)
        assert torch.allclose(a["mc_returns"], b["mc_returns"], equal_nan=True)
        assert not torch.allclose(a["returns"], b["returns"]), "GAE returns DO depend on the values"

    def test_survives_the_wire_and_the_spec_flag_is_honoured(self, trainer):
        r = _collect_results(trainer, n_envs=1, n_steps=200)[0]
        with_mc = finalize_result_for_wire(copy.deepcopy(r), _spec("gae") | {"with_mc": True})
        without = finalize_result_for_wire(copy.deepcopy(r), _spec("gae"))
        assert "mc_returns" in with_mc["batch"] and "mc_returns" not in without["batch"]
        after_pipe = batch_from_wire(pickle.loads(pickle.dumps(with_mc))["batch"])
        local, _ = _finalize_value_pretrain_result(
            copy.deepcopy(r), _GAMMA, _LAM, use_gae=True, with_mc_returns=True,
        )
        assert torch.allclose(after_pipe["mc_returns"], local["mc_returns"], equal_nan=True)


class TestDeterministicRolloutFlag:
    def _pool(self, handles, batched: bool):
        return _ValuePretrainWorkers(
            handles=handles, batched=batched, progress_value=multiprocessing.Value("l", 0),
            n_processes=len(handles), envs_per_process=2,
        )

    @pytest.mark.parametrize("flag", [False, True])
    def test_flag_reaches_every_batched_worker(self, trainer, monkeypatch, flag):
        results = _collect_results(trainer, n_envs=2, n_steps=200)
        handles = [
            _FakeHandle([{"chunk": copy.deepcopy(results)}, {"done": True}], batched=True) for _ in range(2)
        ]
        monkeypatch.setattr(trainer, "_close_value_pretrain_workers", lambda p: None)
        trainer._collect_value_pretrain_rollout(
            env=None, n_steps=400, phase_id=1, use_gae=True, pool=self._pool(handles, True),
            deterministic=flag,
        )
        assert [h.deterministic_flags for h in handles] == [[flag], [flag]]

    def test_plain_workers_refuse_deterministic_instead_of_silently_ignoring_it(self, trainer, monkeypatch):
        handles = [_FakeHandle([], batched=False)]
        monkeypatch.setattr(trainer, "_close_value_pretrain_workers", lambda p: None)
        with pytest.raises(ValueError, match="BATCHED"):
            trainer._collect_value_pretrain_rollout(
                env=None, n_steps=100, phase_id=1, use_gae=True, pool=self._pool(handles, False),
                deterministic=True,
            )
        assert handles[0].collect_calls == [], "must fail BEFORE dispatching any work"

    def test_handle_message_carries_the_flag(self):
        from footballcoach.ai.ppo.batched_rollout_worker import BatchedRolloutWorkerHandle

        parent, child = multiprocessing.Pipe()
        h = BatchedRolloutWorkerHandle(process=None, conn=parent, worker_idx=0)
        h.collect(100, deterministic=True)
        assert child.recv()["deterministic"] is True
        h.collect(100)
        assert child.recv()["deterministic"] is False
