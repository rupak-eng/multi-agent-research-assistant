"""Background worker: claims queued runs and drives the graph.

Run:  python -m app.workers.runner

Crash recovery: on startup the worker reclaims items stranded in the
`processing` queue (a previous worker died holding them) and resumes each
from its LangGraph checkpoint — completed nodes are NOT re-executed.
"""

from __future__ import annotations

import json
import time
import traceback

from ..config import get_settings
from ..logging import get_logger
from ..services.orchestrator import build_worker_graph
from ..state.schemas import ErrorCode, RunError, RunStatus, utcnow
from ..storage.registry import RedisRunRegistry

log = get_logger(component="worker")


def _mark_failed(registry: RedisRunRegistry, run_id: str, message: str) -> None:
    raw = registry.get_snapshot(run_id)
    if not raw:
        return
    snap = json.loads(raw)
    snap["status"] = RunStatus.FAILED.value
    snap["error"] = RunError(code=ErrorCode.INTERNAL_ERROR, message=message,
                             node="worker", at=utcnow()).model_dump(mode="json")
    registry.save_snapshot(run_id, json.dumps(snap))


def _drive_guarded(graph, registry: RedisRunRegistry, run_id: str,
                   initial: dict | None, wall_backstop_sec: float) -> None:
    started = time.monotonic()
    config = {"configurable": {"thread_id": run_id}}
    stream_input = initial
    seen_trace = 0
    for _chunk in graph.stream(stream_input, config):
        snap = dict(graph.get_state(config).values)
        # mirror new trace steps to the live trace list as they complete
        steps = snap.get("trace") or []
        for s in steps[seen_trace:]:
            registry.append_trace(run_id, json.dumps(s))
        seen_trace = len(steps)
        registry.save_snapshot(run_id, json.dumps(snap))
        if time.monotonic() - started > wall_backstop_sec:
            raise TimeoutError(
                f"worker backstop: run exceeded {wall_backstop_sec:.0f}s")
    final = dict(graph.get_state(config).values)
    if final.get("status") not in (RunStatus.COMPLETED.value,
                                   RunStatus.FAILED.value):
        final["status"] = RunStatus.FAILED.value
        final["error"] = RunError(
            code=ErrorCode.INTERNAL_ERROR, node="worker",
            message="graph stream ended without terminal status",
            at=utcnow()).model_dump(mode="json")
    registry.save_snapshot(run_id, json.dumps(final))


def main() -> None:
    settings = get_settings()
    graph, deps, registry = build_worker_graph(settings)
    assert isinstance(registry, RedisRunRegistry)
    backstop = settings.wall_clock_timeout_sec + 300

    # --- crash recovery: resume runs orphaned by a dead worker ------------
    # reclaim_orphans removes them from `processing` (no re-queue); we drive
    # each directly from its LangGraph checkpoint, so completed nodes are not
    # re-executed and the run can never be picked up twice.
    orphans = registry.reclaim_orphans()
    for run_id in orphans:
        log.info("resuming orphaned run", run_id=run_id)
        try:
            _drive_guarded(graph, registry, run_id, None, backstop)
        except Exception:
            log.error("run failed", run_id=run_id, error=traceback.format_exc())
            _mark_failed(registry, run_id, "resume after worker crash failed")
    # snapshots stuck in RUNNING that never made it to the processing queue
    for run_id in registry.running_run_ids():
        log.info("resuming stuck RUNNING run", run_id=run_id)
        try:
            _drive_guarded(graph, registry, run_id, None, backstop)
        except Exception:
            _mark_failed(registry, run_id, "resume of stuck run failed")

    log.info("worker started, waiting for runs")
    while True:
        run_id = registry.claim(timeout=5)
        if run_id is None:
            continue
        log.info("claimed run", run_id=run_id)
        try:
            raw = registry.get_snapshot(run_id)
            if raw is None:
                log.error("no snapshot for claimed run", run_id=run_id)
                continue
            snap = json.loads(raw)
            status = snap.get("status")
            if status == RunStatus.RUNNING.value:
                # worker died mid-run (or orphan path): resume from checkpoint
                _drive_guarded(graph, registry, run_id, None, backstop)
            elif status == RunStatus.QUEUED.value:
                snap["status"] = RunStatus.RUNNING.value
                registry.save_snapshot(run_id, json.dumps(snap))
                _drive_guarded(graph, registry, run_id, snap, backstop)
            else:
                # terminal (COMPLETED/FAILED): never re-drive; just ack it.
                log.warning("claimed terminal run, skipping re-drive",
                            run_id=run_id, status=status)
            log.info("run finished", run_id=run_id,
                     status=json.loads(registry.get_snapshot(run_id) or "{}").get("status"))
        except Exception:
            log.error("run failed", run_id=run_id, error=traceback.format_exc())
            _mark_failed(registry, run_id, "worker exception")
        finally:
            registry.ack(run_id)


if __name__ == "__main__":
    from ..logging import configure_logging
    configure_logging(get_settings().log_level)
    main()
