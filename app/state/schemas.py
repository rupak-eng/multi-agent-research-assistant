"""Typed schemas for every handoff between agents.

Design rule: agents never pass raw dicts. Each agent method takes and returns
a validated Pydantic model. The LangGraph state stores plain JSON (dicts) so
checkpoints serialize cleanly; node wrappers validate on every read/write.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Error taxonomy — runs never hang; they end FAILED with one of these codes.
# --------------------------------------------------------------------------
class ErrorCode(str, Enum):
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"          # token / search / subq cap hit
    WALL_CLOCK_TIMEOUT = "WALL_CLOCK_TIMEOUT"    # wall-clock budget exceeded
    PROVIDER_ERROR = "PROVIDER_ERROR"            # LLM provider failed
    SEARCH_FAILED = "SEARCH_FAILED"              # search provider failed
    CITATION_INVALID = "CITATION_INVALID"        # report citations don't resolve
    NO_USABLE_SOURCES = "NO_USABLE_SOURCES"      # nothing retrievable
    PLAN_FAILED = "PLAN_FAILED"                  # planner produced no plan
    INTERNAL_ERROR = "INTERNAL_ERROR"            # unexpected bug


class RunError(BaseModel):
    code: ErrorCode
    message: str
    node: str = ""          # node that raised it
    detail: dict = Field(default_factory=dict)
    at: str = Field(default_factory=utcnow)


class RunStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# --------------------------------------------------------------------------
# Planner handoff
# --------------------------------------------------------------------------
class SubQuestionStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    ANSWERED = "answered"
    INSUFFICIENT = "insufficient"   # researched but sources too thin; may retry
    FAILED = "failed"               # exhausted revisions


class SubQuestion(BaseModel):
    id: str = Field(description="stable id, e.g. 'sq1'")
    question: str
    status: SubQuestionStatus = SubQuestionStatus.PENDING
    attempts: int = 0

    @field_validator("id", mode="before")
    @classmethod
    def _coerce_id(cls, v):
        # Small models sometimes emit integer ids (1, 2, 3). Coerce instead
        # of failing the whole run; planner re-numbers ids anyway.
        return v if isinstance(v, str) else str(v)

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, v):
        # Real LLMs emit near-synonyms ("unanswered", "done", ...). Normalize
        # to the closed enum instead of failing the whole run.
        if isinstance(v, SubQuestionStatus):
            return v
        s = str(v).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "unanswered": "pending", "unstarted": "pending",
            "todo": "pending", "not_started": "pending", "queued": "pending",
            "new": "pending",
            "inprogress": "in_progress", "started": "in_progress",
            "researching": "in_progress", "ongoing": "in_progress",
            "done": "answered", "complete": "answered", "completed": "answered",
            "finished": "answered", "resolved": "answered",
            "partial": "insufficient", "thin": "insufficient",
            "skipped": "failed", "error": "failed",
        }
        s = aliases.get(s, s)
        try:
            return SubQuestionStatus(s)
        except ValueError:
            # Unknown LLM variant: a fresh sub-question is pending by
            # definition; never fail the run on a synonym we missed.
            return SubQuestionStatus.PENDING


class ResearchPlan(BaseModel):
    subquestions: list[SubQuestion] = Field(min_length=1, max_length=10)
    strategy_notes: str = ""

    @field_validator("subquestions")
    @classmethod
    def _unique_ids(cls, v: list[SubQuestion]) -> list[SubQuestion]:
        ids = [s.id for s in v]
        if len(set(ids)) != len(ids):
            raise ValueError("subquestion ids must be unique")
        return v


class PlanRequest(BaseModel):
    question: str
    max_subquestions: int = 5


# --------------------------------------------------------------------------
# Researcher handoff
# --------------------------------------------------------------------------
class SearchRecord(BaseModel):
    url: str
    title: str
    snippet: str
    domain: str = ""
    fetched_at: str = Field(default_factory=utcnow)


class Finding(BaseModel):
    """One atomic claim with provenance. Every claim -> source URL."""
    finding_id: str                    # e.g. "f3"
    subquestion_id: str
    claim: str
    url: str
    title: str
    snippet: str
    fetched_at: str = Field(default_factory=utcnow)


class ResearcherStatus(str, Enum):
    ANSWERED = "answered"
    INSUFFICIENT = "insufficient"
    FAILED = "failed"


class SynthesisOutput(BaseModel):
    """LLM synthesis of findings for one sub-question (findings stay typed in
    the agent; the model only writes summary + status to save tokens)."""
    summary: str = Field(min_length=1)
    status: ResearcherStatus = ResearcherStatus.ANSWERED


class ResearcherResult(BaseModel):
    subquestion_id: str
    summary: str
    findings: list[Finding] = Field(default_factory=list)
    searches_used: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    status: ResearcherStatus = ResearcherStatus.ANSWERED
    error: RunError | None = None


class ResearchRequest(BaseModel):
    subquestion: SubQuestion
    attempt: int = 1
    refinement_hint: str = ""      # supervisor feedback from a previous attempt


# --------------------------------------------------------------------------
# Writer handoff
# --------------------------------------------------------------------------
class ReportSection(BaseModel):
    heading: str
    body_markdown: str              # may contain citation markers like [1], [2]


class Citation(BaseModel):
    marker: str                     # "[1]"
    finding_id: str
    url: str
    title: str


class ResearchReport(BaseModel):
    title: str
    summary: str
    sections: list[ReportSection] = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1)
    generated_at: str = Field(default_factory=utcnow)


class WriteRequest(BaseModel):
    question: str
    plan: ResearchPlan
    findings: list[Finding]
    validation_feedback: str = ""   # set when a previous draft failed validation


# --------------------------------------------------------------------------
# Supervisor routing — deterministic, not LLM-driven (control flow must not
# depend on model nondeterminism).
# --------------------------------------------------------------------------
class RoutingTarget(str, Enum):
    RESEARCH = "research"   # run researcher on next pending subquestion
    WRITE = "write"         # enough material -> write the report
    END = "end"             # finished (COMPLETED or FAILED already set)
    FAIL = "fail"           # abort now with state.error


class RoutingDecision(BaseModel):
    next: RoutingTarget
    reason: str
    subquestion_id: str | None = None
    error_code: ErrorCode | None = None  # set when next == FAIL


# --------------------------------------------------------------------------
# Budgets & usage
# --------------------------------------------------------------------------
class RunBudgets(BaseModel):
    max_tokens: int = 60_000
    max_searches: int = 12
    max_subquestions: int = 5
    max_revisions: int = 2
    wall_clock_sec: int = 600


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    searches: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost_usd(self, price_in_per_1m: float, price_out_per_1m: float) -> float:
        return (
            self.input_tokens / 1_000_000 * price_in_per_1m
            + self.output_tokens / 1_000_000 * price_out_per_1m
        )


# --------------------------------------------------------------------------
# Trace — one step per node execution, persisted to Redis.
# --------------------------------------------------------------------------
class TraceStep(BaseModel):
    seq: int
    run_id: str
    node: str                       # graph node name
    agent: str                      # planner | researcher | writer | supervisor | system
    input_summary: str
    output_summary: str
    tools_used: list[str] = Field(default_factory=list)
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cached: bool = False            # served from idempotency cache, no side effects
    error: RunError | None = None
    routing_decision: RoutingDecision | None = None
    at: str = Field(default_factory=utcnow)


# --------------------------------------------------------------------------
# Full run state (serialized to JSON in the run snapshot; the LangGraph
# state holds these as plain dicts — see graph/state.py).
# --------------------------------------------------------------------------
class RunSnapshot(BaseModel):
    run_id: str
    question: str
    status: RunStatus
    plan: ResearchPlan | None = None
    findings: list[Finding] = Field(default_factory=list)
    researcher_results: list[ResearcherResult] = Field(default_factory=list)
    report: ResearchReport | None = None
    report_markdown: str = ""
    usage: TokenUsage = Field(default_factory=TokenUsage)
    budgets: RunBudgets = Field(default_factory=RunBudgets)
    revision_count: int = 0
    error: RunError | None = None
    trace: list[TraceStep] = Field(default_factory=list)
    created_at: str = Field(default_factory=utcnow)
    updated_at: str = Field(default_factory=utcnow)
    token_counter: str = "unknown"  # "tiktoken" | "char-estimate"


# --------------------------------------------------------------------------
# API models
# --------------------------------------------------------------------------
class ResearchCreateRequest(BaseModel):
    question: str = Field(min_length=10, max_length=2000)
    budgets: RunBudgets | None = None


class ResearchCreateResponse(BaseModel):
    run_id: str
    status: RunStatus


class Progress(BaseModel):
    subquestions_total: int = 0
    subquestions_answered: int = 0
    searches_used: int = 0
    tokens_used: int = 0


class ResearchStatusResponse(BaseModel):
    run_id: str
    status: RunStatus
    question: str
    progress: Progress = Field(default_factory=Progress)
    report_markdown: str | None = None
    error: RunError | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    estimated_cost_usd: float = 0.0
    created_at: str = ""
    updated_at: str = ""


class TraceResponse(BaseModel):
    run_id: str
    steps: list[TraceStep]
