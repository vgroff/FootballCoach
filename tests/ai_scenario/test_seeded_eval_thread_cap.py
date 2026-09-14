"""Regression coverage for a real production crash: ``OMP: Error #111:
Memory allocation failed.`` killing a live training run right after the
first rollout's evals (PPOTrainer._maybe_run_neural_snapshot_eval spawning
its throwaway eval.eval_n_parallel_workers process pool).

Root cause: every OTHER worker-spawn point in this codebase
(ai/ppo/rollout_worker.py, ai/ppo/batched_rollout_worker.py,
ai/eval/eval_worker.py) sets OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/
MKL_NUM_THREADS/etc INSIDE the freshly-spawned child's own entry function,
before that function's own numpy/torch import -- see
rollout_worker.py's _worker_main docstring: BLAS reads these vars once, at
first use, to size its OWN native thread pool, entirely independent of (and
not shrunk by) a later torch.set_num_threads() call. Left uncapped, each
worker process defaults to one BLAS thread per logical core.

ai/eval/seeded_eval.py's throwaway pools (used by
run_seeded_evaluation_parallel[_batched] whenever no persistent pool is
passed in -- exactly PPOTrainer._maybe_run_neural_snapshot_eval's case) were
missing this guard entirely: _eval_worker_entry/_eval_worker_entry_batched
only ever called torch.set_num_threads(1), AND the per-child-body pattern
used elsewhere would not even have worked here, because seeded_eval.py
imports numpy at MODULE level (needed to resolve the pickled worker-entry
function in the spawned child) -- by the time either entry function's body
ran, numpy's own BLAS thread pool would already be sized off the ambient,
uncapped environment. The fix (_capped_thread_env_for_pool_spawn) sets the
env vars in the PARENT process immediately before constructing each
throwaway Pool instead -- a spawned child inherits the parent's os.environ
as its own OS-level process environment from the moment it's created,
before any of its own Python code (module-level imports included) runs.
"""
from __future__ import annotations

import multiprocessing as mp
import os

from footballcoach.ai.eval.seeded_eval import _capped_thread_env_for_pool_spawn


def _report_omp_env() -> str | None:
    """Module-level (picklable) subprocess entry point -- reports what its
    OWN os.environ looks like, to directly verify inheritance rather than
    just asserting on the parent process's os.environ."""
    import os as _os
    return _os.environ.get("OMP_NUM_THREADS")


class TestCappedThreadEnvForPoolSpawn:
    def test_sets_all_thread_env_vars_inside_the_block(self):
        _prev = {
            v: os.environ.get(v) for v in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            )
        }
        try:
            with _capped_thread_env_for_pool_spawn("1"):
                assert os.environ["OMP_NUM_THREADS"] == "1"
                assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
                assert os.environ["MKL_NUM_THREADS"] == "1"
                assert os.environ["NUMEXPR_NUM_THREADS"] == "1"
                assert os.environ["VECLIB_MAXIMUM_THREADS"] == "1"
        finally:
            for v, val in _prev.items():
                if val is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = val

    def test_restores_previous_value_on_exit(self):
        prev = os.environ.get("OMP_NUM_THREADS")
        try:
            os.environ["OMP_NUM_THREADS"] = "8"
            with _capped_thread_env_for_pool_spawn("1"):
                assert os.environ["OMP_NUM_THREADS"] == "1"
            assert os.environ["OMP_NUM_THREADS"] == "8"
        finally:
            if prev is None:
                os.environ.pop("OMP_NUM_THREADS", None)
            else:
                os.environ["OMP_NUM_THREADS"] = prev

    def test_restores_to_unset_when_previously_unset(self):
        prev = os.environ.pop("OMP_NUM_THREADS", None)
        try:
            with _capped_thread_env_for_pool_spawn("1"):
                assert os.environ["OMP_NUM_THREADS"] == "1"
            assert "OMP_NUM_THREADS" not in os.environ
        finally:
            if prev is not None:
                os.environ["OMP_NUM_THREADS"] = prev

    def test_spawned_child_process_actually_inherits_the_cap(self):
        """The real-world claim this fix depends on: a real
        multiprocessing 'spawn'-context child process, created INSIDE the
        context manager, must see OMP_NUM_THREADS=1 in its OWN os.environ
        -- not just the parent's. This is what a plain assertion on the
        parent process's os.environ (the tests above) cannot prove by
        itself: env-var inheritance into a genuinely separate OS process is
        the actual mechanism the fix relies on."""
        ctx = mp.get_context("spawn")
        prev = os.environ.get("OMP_NUM_THREADS")
        try:
            with _capped_thread_env_for_pool_spawn("1"):
                with ctx.Pool(processes=1) as pool:
                    result = pool.apply(_report_omp_env)
            assert result == "1", (
                "spawned child did not inherit OMP_NUM_THREADS=1 from the "
                "parent's os.environ at Pool-construction time"
            )
        finally:
            if prev is None:
                os.environ.pop("OMP_NUM_THREADS", None)
            else:
                os.environ["OMP_NUM_THREADS"] = prev
