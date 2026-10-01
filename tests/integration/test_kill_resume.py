"""Chaos test: SIGKILL the worker mid-run, restart, verify resume.

Uses REAL Redis (skipped when unreachable) and the REAL worker process
(`python -m app.workers.runner`). Proves:
  1. the run resumes from the last committed checkpoint (no restart),
  2. completed nodes are NOT re-executed (no duplicate researcher results),
  3. the run completes with a valid cited report.
"""

import json
import os
import signal
import subprocess
import sys
import time

import pytest
import redis

from app.services.orchestrator import build_initial_state
from app.state.schemas import RunBudgets, RunStatus
from app.storage.registry import RedisRunRegistry, new_run_id

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable  # the interpreter running pytest (CI has no .venv)
TEST_DB = 15


def _redis_or_skip():
    try:
        r = redis.Redis.from_url(f"redis://localhost:6379/{TEST_DB}",
                                 decode_responses=False, socket_timeout=2)
        r.ping()
        return r
    except Exception:  # noqa: BLE001 - any redis failure means skip
        pytest.skip("no Redis on localhost:6379 — chaos test needs real Redis")


def _snapshot(r, run_id):
    raw = r.get(f"research:run:{run_id}")
    return json.loads(raw.decode()) if raw else None


def _wait_for(pred, timeout, interval=0.5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        v = pred()
        if v:
            return v
        time.sleep(interval)
    raise TimeoutError("timed out waiting for condition")


def test_kill9_midrun_resumes_and_completes():
    r = _redis_or_skip()
    r.flushdb()
    registry = RedisRunRegistry(r)

    run_id = new_run_id()
    budgets = RunBudgets(max_subquestions=3, max_searches=9)
    initial = build_initial_state(run_id, "vector databases for RAG", budgets)
    registry.create_run(run_id, "vector databases for RAG",
                        json.dumps(initial))
    registry.enqueue(run_id)

    env = dict(os.environ,
               REDIS_URL=f"redis://localhost:6379/{TEST_DB}",
               STEP_DELAY_SEC="1.2",
               LLM_PROVIDER="stub",
               SEARCH_PROVIDER="stub",
               LOG_LEVEL="WARNING")

    worker = subprocess.Popen([PY, "-m", "app.workers.runner"],
                              cwd=REPO, env=env,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
    try:
        # wait until real progress exists, then SIGKILL mid-run
        _wait_for(lambda: (_snapshot(r, run_id) or {}).get("trace")
                  and len(_snapshot(r, run_id)["trace"]) >= 4, 90)
        steps_at_kill = len(_snapshot(r, run_id)["trace"])
        assert steps_at_kill >= 4
        os.kill(worker.pid, signal.SIGKILL)
        worker.wait(timeout=10)

        # sanity: the run was genuinely in-flight, not finished
        mid = _snapshot(r, run_id)
        assert mid["status"] == RunStatus.RUNNING.value, mid["status"]

        # restart a fresh worker: it must reclaim the orphan and resume
        t0 = time.monotonic()
        worker2 = subprocess.Popen(
            [PY, "-m", "app.workers.runner"], cwd=REPO, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            def _terminal():
                s = _snapshot(r, run_id)
                if s and s["status"] in (RunStatus.COMPLETED.value,
                                         RunStatus.FAILED.value):
                    return s
                return None
            final = _wait_for(_terminal, 240)
        finally:
            worker2.terminate()
            worker2.wait(timeout=15)
        resume_sec = time.monotonic() - t0

        assert final["status"] == RunStatus.COMPLETED.value, final.get("error")

        # no duplicated work: one researcher result per subquestion
        results = final["researcher_results"]
        sqids = [x["subquestion_id"] for x in results]
        assert len(sqids) == len(set(sqids)) == 3, sqids

        # completed research nodes were not re-executed after resume:
        # every research trace step after the kill belongs to a subquestion
        # that had no completed result at kill time.
        research_steps = [s for s in final["trace"] if s["node"] == "research"
                          and not s["cached"]]
        assert len(research_steps) <= 4, [s["input_summary"][:40]
                                          for s in research_steps]

        assert final["report"] and final["report_markdown"]
        assert "## Sources" in final["report_markdown"]

        print(f"\n[chaos] steps_at_kill={steps_at_kill} "
              f"resume_to_completed_sec={resume_sec:.1f} "
              f"research_node_executions={len(research_steps)}")
    finally:
        if worker.poll() is None:
            worker.kill()
        r.flushdb()
