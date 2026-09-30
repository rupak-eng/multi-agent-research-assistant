"""Unit tests: schemas, supervisor routing, citations, tokens, stubs."""

import pytest

from app.state.schemas import (
    ErrorCode,
    ResearchPlan,
    ResearcherResult,
    ResearcherStatus,
    RoutingTarget,
    RunBudgets,
    SubQuestion,
    SubQuestionStatus,
    TokenUsage,
)


def _sq(i, status=SubQuestionStatus.PENDING):
    return SubQuestion(id=f"sq{i}", question=f"question {i}?", status=status)


def _plan(*statuses):
    return ResearchPlan(
        subquestions=[_sq(i + 1, s) for i, s in enumerate(statuses)],
        strategy_notes="test",
    )


def _result(sqid, status=ResearcherStatus.ANSWERED):
    return ResearcherResult(subquestion_id=sqid, summary="s", status=status)


# ------------------------------------------------------------- schemas ---
def test_plan_requires_unique_ids():
    with pytest.raises(Exception):
        ResearchPlan(subquestions=[_sq(1), _sq(1)])


def test_plan_requires_at_least_one():
    with pytest.raises(Exception):
        ResearchPlan(subquestions=[])


def test_cost_math():
    u = TokenUsage(input_tokens=1_000_000, output_tokens=500_000, searches=3)
    assert u.cost_usd(0.15, 0.60) == pytest.approx(0.45)
    assert u.total_tokens == 1_500_000


# ---------------------------------------------------------- supervisor ---
def _view(**kw):
    from app.agents.supervisor import SupervisorView
    base = dict(plan=_plan(SubQuestionStatus.PENDING, SubQuestionStatus.PENDING),
                usage=TokenUsage(), budgets=RunBudgets())
    base.update(kw)
    return SupervisorView(**base)


def test_supervisor_routes_to_first_pending():
    from app.agents.supervisor import SupervisorAgent
    d = SupervisorAgent().decide(_view())
    assert d.next == RoutingTarget.RESEARCH and d.subquestion_id == "sq1"


def test_supervisor_token_budget_fail():
    from app.agents.supervisor import SupervisorAgent
    v = _view(usage=TokenUsage(input_tokens=60_000))
    d = SupervisorAgent().decide(v)
    assert d.next == RoutingTarget.FAIL
    assert d.error_code == ErrorCode.BUDGET_EXCEEDED


def test_supervisor_wall_clock_fail():
    from app.agents.supervisor import SupervisorAgent
    v = _view(elapsed_sec=9999)
    d = SupervisorAgent().decide(v)
    assert d.next == RoutingTarget.FAIL
    assert d.error_code == ErrorCode.WALL_CLOCK_TIMEOUT


def test_supervisor_insufficient_retries_once():
    from app.agents.supervisor import SupervisorAgent
    plan = _plan(SubQuestionStatus.PENDING)
    v = _view(plan=plan, researcher_results=[_result("sq1", ResearcherStatus.INSUFFICIENT)])
    d = SupervisorAgent().decide(v)
    assert d.next == RoutingTarget.RESEARCH and d.subquestion_id == "sq1"


def test_supervisor_insufficient_exhausts_revisions():
    from app.agents.supervisor import SupervisorAgent
    plan = _plan(SubQuestionStatus.PENDING)
    budgets = RunBudgets(max_revisions=1)
    v = _view(plan=plan, budgets=budgets, revision_count=1,
              researcher_results=[_result("sq1", ResearcherStatus.INSUFFICIENT)])
    d = SupervisorAgent().decide(v)
    # sq1 marked failed, nothing else usable -> FAIL with NO_USABLE_SOURCES
    assert d.next == RoutingTarget.FAIL
    assert d.error_code == ErrorCode.NO_USABLE_SOURCES


def test_supervisor_write_when_search_budget_out_with_findings():
    from app.agents.supervisor import SupervisorAgent
    v = _view(usage=TokenUsage(searches=12), findings_count=4)
    d = SupervisorAgent().decide(v)
    assert d.next == RoutingTarget.WRITE


def test_supervisor_fail_search_budget_out_no_findings():
    from app.agents.supervisor import SupervisorAgent
    v = _view(usage=TokenUsage(searches=12), findings_count=0)
    d = SupervisorAgent().decide(v)
    assert d.next == RoutingTarget.FAIL
    assert d.error_code == ErrorCode.BUDGET_EXCEEDED


def test_supervisor_deterministic():
    from app.agents.supervisor import SupervisorAgent
    v = _view()
    a = SupervisorAgent().decide(v)
    b = SupervisorAgent().decide(_view())
    assert a == b


# ----------------------------------------------------------- citations ---
def _report_and_findings():
    from app.state.schemas import Citation, Finding, ReportSection, ResearchReport
    findings = [
        Finding(finding_id="f1", subquestion_id="sq1", claim="c1",
                url="https://x/1", title="t1", snippet="s1"),
        Finding(finding_id="f2", subquestion_id="sq1", claim="c2",
                url="https://x/2", title="t2", snippet="s2"),
    ]
    report = ResearchReport(
        title="t", summary="s",
        sections=[ReportSection(heading="h", body_markdown="claim one [1] and two [2]")],
        citations=[
            Citation(marker="[1]", finding_id="f1", url="https://x/1", title="t1"),
            Citation(marker="[2]", finding_id="f2", url="https://x/2", title="t2"),
        ],
    )
    return report, findings


def test_citations_valid():
    from app.agents.citations import validate_citations
    report, findings = _report_and_findings()
    assert validate_citations(report, findings) == []


def test_citations_marker_out_of_range():
    from app.agents.citations import validate_citations
    report, findings = _report_and_findings()
    report.sections[0].body_markdown = "bogus [9]"
    assert validate_citations(report, findings)


def test_citations_unknown_finding():
    from app.agents.citations import validate_citations
    report, findings = _report_and_findings()
    report.citations[0].finding_id = "fZZ"
    assert validate_citations(report, findings)


def test_citations_unreferenced_entry():
    from app.agents.citations import validate_citations
    report, findings = _report_and_findings()
    report.sections[0].body_markdown = "only [1]"
    problems = validate_citations(report, findings)
    assert any("never referenced" in p for p in problems)


def test_citations_no_markers():
    from app.agents.citations import validate_citations
    report, findings = _report_and_findings()
    report.sections[0].body_markdown = "no citations here"
    assert validate_citations(report, findings)


# -------------------------------------------------------------- tokens ---
def test_token_counter_char_estimate():
    from app.providers.tokens import TokenCounter
    c = TokenCounter()
    assert c.count("abcd" * 100) == 100  # 400 chars / 4
    assert c.method in ("tiktoken", "char-estimate")


# ----------------------------------------------------------------- stubs ---
def test_stub_search_deterministic():
    from app.providers.search_stub import StubSearchProvider
    s = StubSearchProvider()
    a = s.search("vector databases")
    b = s.search("vector databases")
    assert [h.url for h in a] == [h.url for h in b]
    assert all(h.url.startswith("https://stub-search.local/") for h in a)


def test_stub_llm_plan_validates():
    from app.providers.base import LLMRequest
    from app.providers.llm_stub import StubLLMProvider
    from app.agents.workers import PlannerAgent
    from app.state.schemas import PlanRequest
    llm = StubLLMProvider()
    plan, resp = PlannerAgent(llm).plan(
        PlanRequest(question="What is LangGraph?", max_subquestions=3))
    assert len(plan.subquestions) == 3
    assert resp.tokens_in > 0 and resp.tokens_out > 0
