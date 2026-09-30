"""Supervisor: deterministic routing, not LLM-driven.

Control flow must never depend on model nondeterminism. The supervisor is a
pure function of (plan, results, usage, budgets, elapsed): given the same
inputs it always returns the same routing decision, which makes runs
reproducible and the trace auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..state.schemas import (
    ErrorCode,
    ResearcherResult,
    ResearcherStatus,
    ResearchPlan,
    RoutingDecision,
    RoutingTarget,
    RunBudgets,
    RunError,
    SubQuestion,
    SubQuestionStatus,
    TokenUsage,
    utcnow,
)


@dataclass
class SupervisorView:
    plan: ResearchPlan
    researcher_results: list[ResearcherResult] = field(default_factory=list)
    findings_count: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)
    budgets: RunBudgets = field(default_factory=RunBudgets)
    revision_count: int = 0
    elapsed_sec: float = 0.0
    failed: bool = False  # an error is already recorded on the run


class SupervisorAgent:
    name = "supervisor"

    def decide(self, view: SupervisorView) -> RoutingDecision:
        if view.failed:
            return RoutingDecision(next=RoutingTarget.END,
                                   reason="run already marked failed")

        # --- hard budgets first: never route into work we can't pay for ---
        if view.usage.total_tokens >= view.budgets.max_tokens:
            return RoutingDecision(
                next=RoutingTarget.FAIL, error_code=ErrorCode.BUDGET_EXCEEDED,
                reason=(f"token budget exhausted "
                        f"({view.usage.total_tokens} >= {view.budgets.max_tokens})"))
        if view.elapsed_sec >= view.budgets.wall_clock_sec:
            return RoutingDecision(
                next=RoutingTarget.FAIL, error_code=ErrorCode.WALL_CLOCK_TIMEOUT,
                reason=(f"wall-clock budget exceeded "
                        f"({view.elapsed_sec:.0f}s >= {view.budgets.wall_clock_sec}s)"))

        # --- absorb ALL researcher results into sub-question state ---
        # (not just the latest: after a resume/merge the plan may have been
        # re-run and reset to PENDING while results are already present).
        for res in view.researcher_results:
            sq = self._find(view.plan, res.subquestion_id)
            if sq is None:
                continue
            if res.status == ResearcherStatus.ANSWERED:
                sq.status = SubQuestionStatus.ANSWERED
            elif res.status == ResearcherStatus.INSUFFICIENT:
                # handled below for the latest insufficient result only
                pass
            else:  # FAILED
                sq.status = SubQuestionStatus.FAILED
        # retry logic acts on the latest insufficient result, if any
        if view.researcher_results:
            last = view.researcher_results[-1]
            if last.status == ResearcherStatus.INSUFFICIENT:
                sq = self._find(view.plan, last.subquestion_id)
                if sq is not None and view.revision_count < view.budgets.max_revisions:
                    sq.status = SubQuestionStatus.INSUFFICIENT
                    sq.attempts = last_attempts_plus_one(view, last)
                    return RoutingDecision(
                        next=RoutingTarget.RESEARCH,
                        reason=(f"{sq.id} insufficient; revision "
                                f"{view.revision_count + 1}/{view.budgets.max_revisions}"),
                        subquestion_id=sq.id,
                    )
                elif sq is not None:
                    sq.status = SubQuestionStatus.FAILED

        pending = [s for s in view.plan.subquestions
                   if s.status in (SubQuestionStatus.PENDING, SubQuestionStatus.INSUFFICIENT)]
        # cap: never research more sub-questions than budgeted
        answered_or_failed = [s for s in view.plan.subquestions
                              if s.status in (SubQuestionStatus.ANSWERED, SubQuestionStatus.FAILED)]

        if pending and len(answered_or_failed) < view.budgets.max_subquestions:
            if view.usage.searches >= view.budgets.max_searches:
                if view.findings_count:
                    return RoutingDecision(next=RoutingTarget.WRITE, reason=(
                        f"search budget exhausted ({view.usage.searches}); "
                        f"writing from {view.findings_count} findings"))
                return RoutingDecision(
                    next=RoutingTarget.FAIL, error_code=ErrorCode.BUDGET_EXCEEDED,
                    reason="search budget exhausted with no findings")
            nxt = pending[0]
            return RoutingDecision(next=RoutingTarget.RESEARCH,
                                   reason=f"next pending sub-question {nxt.id}",
                                   subquestion_id=nxt.id)

        if view.findings_count:
            return RoutingDecision(next=RoutingTarget.WRITE, reason=(
                f"{len(answered_or_failed)} sub-questions resolved, "
                f"{view.findings_count} findings -> write report"))
        return RoutingDecision(
            next=RoutingTarget.FAIL, error_code=ErrorCode.NO_USABLE_SOURCES,
            reason="no usable sources for any sub-question")

    @staticmethod
    def _find(plan: ResearchPlan, sqid: str) -> SubQuestion | None:
        for s in plan.subquestions:
            if s.id == sqid:
                return s
        return None

    def fail(self, code: ErrorCode, message: str, node: str = "supervise",
             **detail) -> RunError:
        return RunError(code=code, message=message, node=node, detail=detail,
                        at=utcnow())


def last_attempts_plus_one(view: SupervisorView, last: ResearcherResult) -> int:
    sq = SupervisorAgent._find(view.plan, last.subquestion_id)
    return (sq.attempts if sq else 0) + 1
