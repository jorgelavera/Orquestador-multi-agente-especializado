"""Shared typed graph state and attributed agent contribution records."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import MessagesState


AgentName = Literal["research", "analysis", "supervisor", "synthesis", "validator"]
RouteName = Literal["research", "analysis", "synthesis", "safe_terminal"]


class Contribution(TypedDict):
    agent: AgentName
    kind: Literal["evidence", "calculation", "answer"]
    summary: str
    evidence_ids: list[str]
    data: dict[str, object]


class RouteEvent(TypedDict):
    source: str
    destination: str
    reason: str
    provider: str
    step: int


class AgentState(MessagesState, total=False):
    request: str
    contributions: Annotated[list[Contribution], operator.add]
    route_trace: Annotated[list[RouteEvent], operator.add]
    next_route: RouteName
    supervisor_reason: str
    step_count: int
    max_steps: int
    required_agents: list[str]
    refinement_used: bool
    validation_errors: list[str]
    is_valid: bool
    final_answer: str
    citations: list[str]
    status: Literal["running", "complete", "needs_human"]
    router_provider: str
