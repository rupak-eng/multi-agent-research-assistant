"""Agent implementations. Every public method takes/returns Pydantic models."""

from __future__ import annotations

from ..providers.base import LLMProvider, LLMRequest, LLMResponse, SearchProvider, SearchProviderError
from ..providers.tokens import TokenCounter
from ..state.schemas import (
    ErrorCode,
    Finding,
    PlanRequest,
    ResearchPlan,
    ResearchReport,
    ResearchRequest,
    ResearcherResult,
    ResearcherStatus,
    RunError,
    SubQuestion,
    SynthesisOutput,
    WriteRequest,
    utcnow,
)
from urllib.parse import urlparse


# ---------------------------------------------------------------- planner ---
class PlannerAgent:
    SYSTEM = (
        "RESEARCH-PLANNER\n"
        "You are a research planner. Decompose the user's research question into "
        "a small set of concrete, non-overlapping sub-questions that together "
        "cover the topic. Respond with JSON matching the ResearchPlan schema "
        "(subquestions with id, question, status, attempts; plus strategy_notes). "
        "Keep sub-questions specific and searchable."
    )

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def plan(self, req: PlanRequest) -> tuple[ResearchPlan, LLMResponse]:
        resp = self.llm.complete(
            LLMRequest(
                system=self.SYSTEM,
                user=f"QUESTION: {req.question}\nMAX_SUBQUESTIONS: {req.max_subquestions}",
                response_model=ResearchPlan,
            )
        )
        plan = ResearchPlan.model_validate_json(resp.text)
        # Enforce the cap deterministically, re-number ids for stability.
        subqs = plan.subquestions[: req.max_subquestions]
        fixed = [
            SubQuestion(id=f"sq{i+1}", question=s.question,
                        status=s.status, attempts=s.attempts)
            for i, s in enumerate(subqs)
        ]
        return ResearchPlan(subquestions=fixed, strategy_notes=plan.strategy_notes), resp


# -------------------------------------------------------------- researcher ---
class ResearcherAgent:
    SYSTEM = (
        "RESEARCH-SYNTHESIZER\n"
        "You synthesize web-search findings into a short summary for one "
        "sub-question. Respond with JSON: {\"summary\": \"...\", \"status\": "
        "\"answered\" | \"insufficient\"}. Use \"insufficient\" only when the "
        "findings cannot support any substantive claim."
    )
    SEARCHES_PER_SUBQUESTION = 2
    MAX_FINDINGS_PER_DOMAIN = 3

    def __init__(self, llm: LLMProvider, search: SearchProvider,
                 counter: TokenCounter | None = None):
        self.llm = llm
        self.search = search
        self.counter = counter or TokenCounter()

    def _queries(self, req: ResearchRequest) -> list[str]:
        base = req.subquestion.question
        queries = [base]
        if req.refinement_hint:
            queries.append(f"{base} {req.refinement_hint}")
        else:
            queries.append(f"{base} evidence benchmarks")
        return queries[: self.SEARCHES_PER_SUBQUESTION]

    def research(self, req: ResearchRequest, *, searches_left: int,
                 finding_id_offset: int = 0) -> ResearcherResult:
        findings: list = []
        searches_used = 0
        seen_urls: set[str] = set()
        domain_counts: dict[str, int] = {}
        search_errors: list[str] = []

        for q in self._queries(req):
            if searches_left - searches_used <= 0:
                break
            try:
                hits = self.search.search(q, max_results=5)
            except SearchProviderError as e:
                search_errors.append(str(e))
                continue
            searches_used += 1
            for h in hits:
                if h.url in seen_urls:
                    continue
                domain = urlparse(h.url).netloc or "unknown"
                if domain_counts.get(domain, 0) >= self.MAX_FINDINGS_PER_DOMAIN:
                    continue
                seen_urls.add(h.url)
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
                fid = f"f{finding_id_offset + len(findings) + 1}"
                claim = h.snippet.strip().split(". ")[0][:280]
                findings.append(Finding(
                    finding_id=fid, subquestion_id=req.subquestion.id,
                    claim=claim or h.title, url=h.url, title=h.title,
                    snippet=h.snippet, fetched_at=utcnow(),
                ))

        findings_json = "[" + ",".join(f.model_dump_json() for f in findings) + "]"
        resp = self.llm.complete(
            LLMRequest(
                system=self.SYSTEM,
                user=(
                    f"SUBQUESTION_ID: {req.subquestion.id}\n"
                    f"SUBQUESTION: {req.subquestion.question}\n"
                    f"FINDINGS_JSON: {findings_json}"
                ),
                response_model=SynthesisOutput,
            )
        )
        try:
            synth = SynthesisOutput.model_validate_json(resp.text)
        except Exception:
            synth = SynthesisOutput(
                summary=f"Collected {len(findings)} findings.",
                status=ResearcherStatus.ANSWERED if findings else ResearcherStatus.INSUFFICIENT,
            )

        if not findings:
            status = ResearcherStatus.FAILED if search_errors else ResearcherStatus.INSUFFICIENT
            error = (RunError(code=ErrorCode.SEARCH_FAILED,
                              message="; ".join(search_errors) or "no usable sources",
                              node="research") if search_errors else None)
        else:
            status, error = synth.status, None

        return ResearcherResult(
            subquestion_id=req.subquestion.id,
            summary=synth.summary,
            findings=findings,
            searches_used=searches_used,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            status=status,
            error=error,
        )


# ------------------------------------------------------------------ writer ---
class WriterAgent:
    SYSTEM = (
        "REPORT-WRITER\n"
        "You write a cited research report from findings. Every factual claim in "
        "section bodies MUST carry a citation marker [n] referring to the "
        "citations list, where n is the 1-based index into the findings you were "
        "given (finding f1 -> [1], f2 -> [2], ...). Respond with JSON matching "
        "the ResearchReport schema."
    )

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def write(self, req: WriteRequest) -> tuple[ResearchReport, LLMResponse]:
        findings_json = "[" + ",".join(f.model_dump_json() for f in req.findings) + "]"
        user = f"QUESTION: {req.question}\nFINDINGS_JSON: {findings_json}"
        if req.validation_feedback:
            user += f"\nPREVIOUS_DRAFT_FEEDBACK: {req.validation_feedback}"
        resp = self.llm.complete(
            LLMRequest(system=self.SYSTEM, user=user,
                       response_model=ResearchReport, max_tokens=4096)
        )
        report = ResearchReport.model_validate_json(resp.text)
        return report, resp
