"""Deterministic stub LLM for tests, benchmarks and offline demos.

Behavior is keyed off markers in the system prompt:
- "RESEARCH-PLANNER"    -> returns a ResearchPlan JSON for the question
- "RESEARCH-SYNTHESIZER"-> returns a ResearcherResult JSON from findings JSON
- "REPORT-WRITER"       -> returns a ResearchReport JSON from findings JSON

Everything is derived from the actual inputs (no randomness), so runs are
reproducible and benchmarks are comparable.
"""

from __future__ import annotations

import json
import re
import time

from .base import LLMProvider, LLMRequest, LLMResponse
from .tokens import TokenCounter


def _subqs_for(question: str, n: int) -> list[dict]:
    q = question.strip().rstrip("?")
    templates = [
        f"What are the main approaches to {q}?",
        f"What do benchmarks and evaluations say about {q}?",
        f"What are the production trade-offs and costs of {q}?",
        f"What are common failure modes when working with {q}?",
        f"What is the current state of the art for {q}?",
    ]
    return [
        {"id": f"sq{i+1}", "question": templates[i], "status": "pending", "attempts": 0}
        for i in range(min(n, len(templates)))
    ]


class StubLLMProvider(LLMProvider):
    name = "stub"

    def __init__(self) -> None:
        self.counter = TokenCounter()
        self.calls = 0

    def complete(self, req: LLMRequest) -> LLMResponse:
        start = time.monotonic()
        self.calls += 1
        system = req.system
        if "RESEARCH-PLANNER" in system:
            payload = self._plan(req.user)
        elif "RESEARCH-SYNTHESIZER" in system:
            payload = self._synthesize(req.user)
        elif "REPORT-WRITER" in system:
            payload = self._write(req.user)
        else:
            payload = {"echo": req.user[:200]}
        text = json.dumps(payload)
        latency_ms = int((time.monotonic() - start) * 1000)
        return LLMResponse(
            text=text,
            tokens_in=self.counter.count_messages(req.system, req.user),
            tokens_out=self.counter.count(text),
            model="stub-llm-v1",
            latency_ms=latency_ms,
        )

    # -- planner ---------------------------------------------------------
    def _plan(self, user: str) -> dict:
        m = re.search(r"QUESTION:\s*(.+)", user, re.DOTALL)
        question = (m.group(1).strip() if m else user.strip()) or "the topic"
        m2 = re.search(r"MAX_SUBQUESTIONS:\s*(\d+)", user)
        n = int(m2.group(1)) if m2 else 3
        return {
            "subquestions": _subqs_for(question, n),
            "strategy_notes": (
                f"Decompose '{question[:80]}' into sub-questions, research each "
                "with web search, then synthesize a cited report."
            ),
        }

    # -- researcher synthesis --------------------------------------------
    def _synthesize(self, user: str) -> dict:
        m = re.search(r"FINDINGS_JSON:\s*(\[.*\])", user, re.DOTALL)
        findings = json.loads(m.group(1)) if m else []
        m2 = re.search(r"SUBQUESTION_ID:\s*(\S+)", user)
        sqid = m2.group(1) if m2 else "sq?"
        claims = "; ".join(f.get("claim", "")[:90] for f in findings[:3])
        summary = (
            f"Synthesis for {sqid}: {len(findings)} findings collected. "
            f"Key claims — {claims}." if findings
            else f"Synthesis for {sqid}: no usable findings."
        )
        return {
            "subquestion_id": sqid,
            "summary": summary,
            "findings": findings,
            "searches_used": 0,  # filled in by the agent from actual calls
            "tokens_in": 0,
            "tokens_out": 0,
            "status": "answered" if findings else "insufficient",
            "error": None,
        }

    # -- writer ------------------------------------------------------------
    def _write(self, user: str) -> dict:
        m = re.search(r"FINDINGS_JSON:\s*(\[.*\])", user, re.DOTALL)
        findings = json.loads(m.group(1)) if m else []
        m2 = re.search(r"QUESTION:\s*(.+)", user, re.DOTALL)
        question = (m2.group(1).strip().splitlines()[0] if m2 else "Research report")
        citations = [
            {
                "marker": f"[{i+1}]",
                "finding_id": f.get("finding_id", f"f{i+1}"),
                "url": f.get("url", ""),
                "title": f.get("title", ""),
            }
            for i, f in enumerate(findings)
        ]
        by_sq: dict[str, list[dict]] = {}
        for f in findings:
            by_sq.setdefault(f.get("subquestion_id", "?"), []).append(f)
        sections = []
        for idx, (sqid, fs) in enumerate(by_sq.items()):
            _n0 = findings.index(fs[0]) + 1
            body = " ".join(
                f"{f.get('claim','')} [{findings.index(f)+1}]" for f in fs
            )
            sections.append({"heading": f"Finding group {idx+1} ({sqid})", "body_markdown": body})
        if not sections:
            sections = [{"heading": "Summary", "body_markdown": "No findings were available."}]
        return {
            "title": f"Research report: {question[:90]}",
            "summary": (
                f"This report synthesizes {len(findings)} findings from "
                f"{len({f.get('url') for f in findings})} sources."
            ),
            "sections": sections,
            "citations": citations,
        }
