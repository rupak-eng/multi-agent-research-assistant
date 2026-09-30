"""Integration: full graph run with stub providers (file backend)."""

import json
import tempfile

import pytest

from app.agents.citations import validate_citations
from app.config import Settings
from app.services.orchestrator import (
    build_initial_state,
    build_worker_graph,
    drive_run,
)
from app.state.schemas import (
    Finding,
    ResearchReport,
    RunBudgets,
    RunStatus,
)
from app.storage.checkpoint import FileCheckpointer
from app.storage.registry import FileRunRegistry


@pytest.fixture()
def harness():
    tmp = tempfile.mkdtemp()
    settings = Settings(llm_provider="stub", search_provider="stub")
    registry = FileRunRegistry(tmp + "/reg")
    ckpt = FileCheckpointer(tmp + "/ckpt")
    graph, deps, _ = build_worker_graph(settings, registry, checkpointer=ckpt)
    return settings, registry, graph


def _run(harness, question, budgets):
    settings, registry, graph = harness
    run_id = "run_integ1"
    initial = build_initial_state(run_id, question, budgets)
    registry.create_run(run_id, question, json.dumps(initial))
    return run_id, drive_run(graph, registry, run_id, initial)


def test_full_run_completes_with_cited_report(harness):
    run_id, final = _run(harness, "vector databases for RAG",
                         RunBudgets(max_subquestions=2, max_searches=6))
    assert final["status"] == RunStatus.COMPLETED.value
    assert final["error"] is None
    report = ResearchReport.model_validate(final["report"])
    findings = [Finding.model_validate(f) for f in final["findings"]]
    assert validate_citations(report, findings) == []
    md = final["report_markdown"]
    assert "## Sources" in md
    assert "stub-search.local" in md
    # every finding URL appears in the report sources
    for f in findings:
        assert f.url in md


def test_trace_covers_all_agents(harness):
    _, final = _run(harness, "LangGraph supervisor patterns",
                    RunBudgets(max_subquestions=2, max_searches=6))
    agents = {s["agent"] for s in final["trace"]}
    assert {"planner", "supervisor", "researcher", "writer"} <= agents
    nodes = [s["node"] for s in final["trace"]]
    assert nodes[0] == "plan" and nodes[-1] == "validate"
    for s in final["trace"]:
        assert s["latency_ms"] >= 0
    # routing decisions recorded on supervise steps
    sup_steps = [s for s in final["trace"] if s["node"] == "supervise"]
    assert all(s["routing_decision"] for s in sup_steps)


def test_search_budget_routes_to_write(harness):
    # 1 search total: researcher spends it on sq1, supervisor must not
    # research sq2 but write from what exists.
    _, final = _run(harness, "vector databases for RAG",
                    RunBudgets(max_subquestions=3, max_searches=1))
    assert final["status"] == RunStatus.COMPLETED.value
    reasons = [s["routing_decision"]["reason"]
               for s in final["trace"] if s["node"] == "supervise"]
    assert any("search budget exhausted" in r for r in reasons)


def test_zero_search_budget_fails_cleanly(harness):
    _, final = _run(harness, "vector databases for RAG",
                    RunBudgets(max_subquestions=2, max_searches=0))
    # max_searches has ge=1 in Settings but RunBudgets allows explicit 0
    assert final["status"] == RunStatus.FAILED.value
    assert final["error"]["code"] == "BUDGET_EXCEEDED"
    assert "search budget" in final["error"]["message"]


def test_token_budget_fails_cleanly(harness):
    _, final = _run(harness, "vector databases for RAG",
                    RunBudgets(max_subquestions=2, max_searches=6,
                               max_tokens=10))
    assert final["status"] == RunStatus.FAILED.value
    assert final["error"]["code"] == "BUDGET_EXCEEDED"
