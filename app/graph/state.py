"""LangGraph state.

The graph carries plain JSON (dicts/lists/str). Node wrappers validate every
read/write against the Pydantic schemas in app/state/schemas.py, so agent
handoffs stay typed while checkpoints remain trivially serializable.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class GraphState(TypedDict, total=False):
    run_id: str
    question: str
    status: str                       # RunStatus value
    plan: dict | None                 # ResearchPlan JSON
    active_subquestion: dict | None   # SubQuestion JSON (+ attempt info)
    researcher_results: Annotated[list[dict], operator.add]
    findings: Annotated[list[dict], operator.add]
    report: dict | None               # ResearchReport JSON
    report_markdown: str
    validation_feedback: str
    writer_attempts: int
    trace: Annotated[list[dict], operator.add]
    usage: dict                       # {"input_tokens","output_tokens","searches"}
    budgets: dict                     # RunBudgets JSON
    revision_count: int
    error: dict | None                # RunError JSON
    started_at: str                   # ISO timestamp
    routing: dict | None              # last RoutingDecision JSON
