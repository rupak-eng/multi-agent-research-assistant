"""Node idempotency: re-running a node with identical inputs performs no
duplicate side effects and returns the same result."""


from app.config import Settings
from app.graph.nodes import make_deps, research_node
from app.providers.llm_stub import StubLLMProvider
from app.providers.search_stub import StubSearchProvider
from app.services.orchestrator import build_initial_state
from app.state.schemas import (
    ResearcherResult,
    ResearcherStatus,
    RunBudgets,
    SubQuestion,
    SubQuestionStatus,
)
from app.storage.registry import FileRunRegistry


def _deps(tmp_path):
    settings = Settings(llm_provider="stub", search_provider="stub")
    registry = FileRunRegistry(str(tmp_path / "reg"))
    search = StubSearchProvider()
    llm = StubLLMProvider()
    deps = make_deps(settings, registry, llm=llm, search=search)
    return deps, registry, search


def _state_with_active_sq(run_id="run_idem1"):
    state = build_initial_state(run_id, "vector databases", RunBudgets())
    sq = SubQuestion(id="sq1", question="What are vector databases?",
                     status=SubQuestionStatus.IN_PROGRESS, attempts=1)
    state["active_subquestion"] = sq.model_dump(mode="json")
    return state


def test_research_node_idempotent(tmp_path):
    deps, _registry, search = _deps(tmp_path)
    state = _state_with_active_sq()

    out1 = research_node(state, deps)
    calls_after_first = len(search.calls)
    assert calls_after_first > 0

    # second invocation with identical inputs: cache hit, no new searches
    out2 = research_node(state, deps)
    assert len(search.calls) == calls_after_first

    r1 = ResearcherResult.model_validate(out1["researcher_results"][0])
    r2 = ResearcherResult.model_validate(out2["researcher_results"][0])
    assert r1 == r2
    assert r1.status == ResearcherStatus.ANSWERED
    assert len(r1.findings) > 0


def test_research_node_cached_trace_flag(tmp_path):
    deps, _registry, _search = _deps(tmp_path)
    state = _state_with_active_sq()
    research_node(state, deps)
    out2 = research_node(state, deps)
    assert out2["trace"][0]["cached"] is True
    assert out2["trace"][0]["tools_used"]  # tools still reported honestly
