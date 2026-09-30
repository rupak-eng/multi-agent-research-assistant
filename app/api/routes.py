"""HTTP API: submit research, poll status, fetch the execution trace."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from ..api.deps import QUEUE, get_redis_client, redis_ping, run_key, trace_key
from ..config import Settings, get_settings
from ..logging import get_logger
from ..services.orchestrator import (
    build_initial_state,
    snapshot_to_status,
    trace_to_response,
)
from ..state.schemas import (
    ResearchCreateRequest,
    ResearchCreateResponse,
    ResearchStatusResponse,
    RunBudgets,
    RunStatus,
    TraceResponse,
)
from ..storage.registry import new_run_id

log = get_logger(component="api")
router = APIRouter()


def _settings() -> Settings:
    return get_settings()


@router.post("/research", response_model=ResearchCreateResponse,
             status_code=202)
async def create_research(req: ResearchCreateRequest,
                          settings: Settings = Depends(_settings)):  # noqa: B008 - FastAPI DI
    """Enqueue a research run. Returns immediately with a run_id."""
    client = get_redis_client(settings)
    if not await redis_ping(client):
        raise HTTPException(status_code=503, detail="state store unavailable")
    run_id = new_run_id()
    budgets = req.budgets or RunBudgets(
        max_tokens=settings.max_tokens_per_run,
        max_searches=settings.max_searches_per_run,
        max_subquestions=settings.max_subquestions_per_run,
        max_revisions=settings.max_revisions_per_run,
        wall_clock_sec=settings.wall_clock_timeout_sec,
    )
    state = build_initial_state(run_id, req.question, budgets)
    await client.set(run_key(run_id), json.dumps(state))
    await client.lpush(QUEUE, run_id)
    log.info("run enqueued", run_id=run_id,
             question=req.question[:80])
    return ResearchCreateResponse(run_id=run_id, status=RunStatus.QUEUED)


@router.get("/research/{run_id}", response_model=ResearchStatusResponse)
async def get_research(run_id: str, settings: Settings = Depends(_settings)):  # noqa: B008 - FastAPI DI
    client = get_redis_client(settings)
    raw = await client.get(run_key(run_id))
    if raw is None:
        raise HTTPException(status_code=404, detail="unknown run_id")
    return snapshot_to_status(raw, settings)


@router.get("/research/{run_id}/trace", response_model=TraceResponse)
async def get_trace(run_id: str, settings: Settings = Depends(_settings)):  # noqa: B008 - FastAPI DI
    client = get_redis_client(settings)
    if await client.get(run_key(run_id)) is None:
        raise HTTPException(status_code=404, detail="unknown run_id")
    # Live trace list, mirrored step-by-step by the worker. Falls back to the
    # snapshot if the worker wrote only snapshots (e.g. resumed file runs).
    raw_steps = await client.lrange(trace_key(run_id), 0, -1)
    if not raw_steps:
        snap = json.loads(await client.get(run_key(run_id)))
        raw_steps = [json.dumps(s) for s in (snap.get("trace") or [])]
    return trace_to_response(run_id, list(raw_steps))


@router.get("/health")
async def health():
    return {"status": "ok", "service": "multi-agent-research-assistant"}


@router.get("/ready")
async def ready(settings: Settings = Depends(_settings)):  # noqa: B008 - FastAPI DI
    client = get_redis_client(settings)
    ok = await redis_ping(client)
    body = {"ready": bool(ok),
            "llm_provider": settings.llm_provider,
            "search_provider": settings.search_provider}
    return JSONResponse(status_code=200 if ok else 503, content=body)
