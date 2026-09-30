"""Assemble the supervisor-routed research graph."""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from ..state.schemas import RoutingTarget
from .nodes import (
    NodeDeps,
    plan_node,
    research_node,
    supervise_node,
    validate_node,
    write_node,
)
from .state import GraphState


def build_graph(deps: NodeDeps,
               checkpointer: BaseCheckpointSaver | None = None):
    """Nodes are bound to deps via closures so tests can inject stubs."""
    g = StateGraph(GraphState)

    g.add_node("plan", lambda s: plan_node(s, deps))
    g.add_node("supervise", lambda s: supervise_node(s, deps))
    g.add_node("research", lambda s: research_node(s, deps))
    g.add_node("write", lambda s: write_node(s, deps))
    g.add_node("validate", lambda s: validate_node(s, deps))

    g.add_edge(START, "plan")
    g.add_edge("plan", "supervise")
    g.add_conditional_edges(
        "supervise",
        lambda s: _route_supervise(s),
        {"research": "research", "write": "write", "end": END},
    )
    g.add_edge("research", "supervise")
    g.add_edge("write", "validate")
    g.add_conditional_edges(
        "validate",
        lambda s: _route_validate(s),
        {"ok": END, "retry": "supervise", "fail": END},
    )
    return g.compile(checkpointer=checkpointer)


def _route_supervise(state: GraphState) -> str:
    from ..state.schemas import RoutingDecision
    routing = state.get("routing")
    if not routing:
        return "end"
    decision = RoutingDecision.model_validate(routing)
    if decision.next == RoutingTarget.RESEARCH:
        return "research"
    if decision.next == RoutingTarget.WRITE:
        return "write"
    return "end"


def _route_validate(state: GraphState) -> str:
    from ..state.schemas import RunStatus
    if state.get("status") == RunStatus.COMPLETED.value:
        return "ok"
    if state.get("status") == RunStatus.FAILED.value:
        return "fail"
    return "retry"
