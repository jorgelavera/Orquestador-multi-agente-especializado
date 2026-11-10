"""Supervisor-led LangGraph topology with bounded refinement and validation."""

from __future__ import annotations

import ipaddress
import json
import math
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from openai import AsyncOpenAI

from orchestrator.agents.analysis_agent import analysis_agent
from orchestrator.agents.research_agent import make_research_agent
from orchestrator.config import Settings
from orchestrator.state import AgentState, Contribution, RouteEvent, RouteName
from orchestrator.tools.knowledge_base import KnowledgeBase


class Router(Protocol):
    async def choose(
        self,
        *,
        request: str,
        summary: dict[str, Any],
        candidates: list[RouteName],
        fallback: RouteName,
    ) -> tuple[RouteName, str, str]: ...


class SupervisorRouter:
    """Ask the configured OpenAI-compatible local model for constrained JSON routing."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: AsyncOpenAI | None = None
        if settings.provider != "offline":
            parsed_url = urlsplit(settings.base_url)
            host = parsed_url.hostname
            if parsed_url.scheme not in {"http", "https"} or not host or not (
                host.casefold() == "localhost"
                or _is_loopback_address(host)
            ):
                raise ValueError("ORCHESTRATOR_BASE_URL must use HTTP(S) and a localhost/loopback host")
            self._client = AsyncOpenAI(
                api_key="ollama",
                base_url=settings.base_url,
                timeout=settings.timeout_seconds,
                max_retries=0,
            )

    async def choose(
        self,
        *,
        request: str,
        summary: dict[str, Any],
        candidates: list[RouteName],
        fallback: RouteName,
    ) -> tuple[RouteName, str, str]:
        if self._client is not None:
            try:
                response = await self._client.chat.completions.create(
                    model=self.settings.model,
                    temperature=0,
                    max_tokens=1024,
                    response_format={"type": "json_object"},
                    extra_body={"reasoning_effort": "none"},
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a workflow supervisor. Return one JSON object with keys "
                                "route and reason. route MUST be one of the supplied candidates. "
                                "Choose only a candidate; do not invent work, evidence, or metrics. "
                                "Prefer the specialist required by the current state."
                            ),
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                {"request": request, "state_summary": summary, "candidates": candidates},
                                ensure_ascii=False,
                            ),
                        },
                    ],
                )
                content = (response.choices[0].message.content or "{}").strip()
                # Some local chat templates wrap otherwise valid JSON in literal single quotes.
                if content.startswith("'{") and content.endswith("}'"):
                    content = content[1:-1]
                payload = json.loads(content)
                route = payload.get("route")
                reason = payload.get("reason")
                if route in candidates:
                    explanation = reason[:300] if isinstance(reason, str) and reason.strip() else "Local model selected an allowed route."
                    return route, explanation, "ollama"
                return fallback, "Local model returned a route outside the allowed candidate set.", "deterministic-fallback"
            except Exception as exc:
                return fallback, f"Local model unavailable or invalid; deterministic fallback: {type(exc).__name__}.", "deterministic-fallback"
        return fallback, "Offline deterministic route selected from the bounded candidate set.", "offline"


def _is_loopback_address(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _latest_contribution(state: AgentState, agent: str, kind: str | None = None) -> Contribution | None:
    for contribution in reversed(state.get("contributions", [])):
        if contribution["agent"] == agent and (kind is None or contribution["kind"] == kind):
            return contribution
    return None


def _candidate_routes(state: AgentState) -> tuple[list[RouteName], RouteName, str]:
    if state.get("validation_errors") and state.get("refinement_used"):
        errors = " ".join(state["validation_errors"]).lower()
        if "calculation" in errors or "metric" in errors:
            refinement_route: RouteName = "analysis"
            refinement_reason = "Refine the invalid or missing deterministic calculation."
        else:
            refinement_route = "research"
            refinement_reason = "Refine evidence or citation provenance."
        last_route = (state.get("route_trace") or [])[-1:]
        if last_route and last_route[0]["source"] == "supervisor" and last_route[0]["destination"] == refinement_route:
            return ["synthesis"], "synthesis", "Revalidate the result after its single targeted refinement."
        return [refinement_route], refinement_route, refinement_reason
    if _latest_contribution(state, "research", "evidence") is None:
        return ["research", "analysis"], "research", "Retrieve grounded local evidence before analysis."
    if _latest_contribution(state, "analysis", "calculation") is None:
        return ["analysis", "synthesis"], "analysis", "Compute requested metrics from retrieved evidence."
    return ["synthesis"], "synthesis", "Both specialist outputs are available; synthesize and validate."


async def supervisor_node(
    state: AgentState,
    *,
    router: Router,
    settings: Settings,
) -> dict[str, Any]:
    current_steps = state.get("step_count", 0)
    max_steps = state.get("max_steps", settings.max_steps)
    if current_steps >= max_steps:
        event: RouteEvent = {
            "source": "supervisor",
            "destination": "safe_terminal",
            "reason": f"Maximum supervisor step count ({max_steps}) reached.",
            "provider": "step-guard",
            "step": current_steps,
        }
        return {
            "next_route": "safe_terminal",
            "supervisor_reason": event["reason"],
            "route_trace": [event],
            "status": "running",
        }

    candidates, fallback, fallback_reason = _candidate_routes(state)
    summary = {
        "completed_agents": [
            name
            for name in ("research", "analysis")
            if _latest_contribution(state, name) is not None
        ],
        "source_ids": sorted(
            {
                source_id
                for contribution in state.get("contributions", [])
                for source_id in contribution["evidence_ids"]
            }
        ),
        "has_calculation": _latest_contribution(state, "analysis", "calculation") is not None,
        "refinement_used": state.get("refinement_used", False),
        "validation_errors": state.get("validation_errors", []),
    }
    route, reason, provider = await router.choose(
        request=state["request"],
        summary=summary,
        candidates=candidates,
        fallback=fallback,
    )
    if route not in candidates:
        route, reason, provider = fallback, fallback_reason, "deterministic-fallback"
    if fallback in {"research", "analysis"} and route != fallback:
        route = fallback
        reason = fallback_reason
        provider = "deterministic-guard"
    step = current_steps + 1
    event = {
        "source": "supervisor",
        "destination": route,
        "reason": reason or fallback_reason,
        "provider": provider,
        "step": step,
    }
    return {
        "next_route": route,
        "supervisor_reason": event["reason"],
        "step_count": step,
        "router_provider": provider,
        "route_trace": [event],
        "status": "running",
    }


def route_from_supervisor(state: AgentState) -> Literal["research", "analysis", "synthesis", "safe_terminal"]:
    """Typed conditional-edge router; invalid state safely terminates."""
    route = state.get("next_route", "safe_terminal")
    if route in {"research", "analysis", "synthesis", "safe_terminal"}:
        return route
    return "safe_terminal"


async def synthesis_node(state: AgentState) -> dict[str, Any]:
    research = _latest_contribution(state, "research", "evidence")
    analysis = _latest_contribution(state, "analysis", "calculation")
    sources = research["data"].get("sources", []) if research else []
    metrics = analysis["data"].get("metrics", {}) if analysis else {}
    source_ids = [
        source["id"]
        for source in sources
        if isinstance(source, dict)
        and isinstance(source.get("id"), str)
        and (source.get("retrieval_fixture") is not None or source["id"] == "metric-definitions-v1")
    ]

    if not isinstance(metrics, dict) or "macro_precision_at_k" not in metrics:
        answer = "I could not produce a supported metric result from the available sources. Please review the request with a human."
        status = "needs_human"
    else:
        source_labels = ", ".join(f"[{source_id}]" for source_id in source_ids)
        answer = (
            f"For the included {metrics['query_count']}-query retrieval fixture at k={metrics['k']}, "
            f"macro Precision@{metrics['k']} is {metrics['macro_precision_at_k']:.4f}, "
            f"macro Recall@{metrics['k']} is {metrics['macro_recall_at_k']:.4f}, "
            f"the mean attainable Precision@{metrics['k']} ceiling is {metrics['precision_ceiling_at_k']:.4f}, "
            f"and normalized precision is {metrics['normalized_precision']:.4f}. "
            f"These values are computed deterministically from the retrieved fixture. Sources: {source_labels}."
        )
        status = "running"
    contribution: Contribution = {
        "agent": "synthesis",
        "kind": "answer",
        "summary": "Synthesized a concise answer from specialist outputs.",
        "evidence_ids": source_ids,
        "data": {"answer": answer},
    }
    return {
        "contributions": [contribution],
        "final_answer": answer,
        "citations": source_ids,
        "status": status,
        "messages": [AIMessage(content=answer)],
    }


def validator_node(state: AgentState) -> dict[str, Any]:
    """Check citation provenance and every required deterministic metric field."""
    research = _latest_contribution(state, "research", "evidence")
    analysis = _latest_contribution(state, "analysis", "calculation")
    answer = _latest_contribution(state, "synthesis", "answer")
    retrieved_ids = {
        str(source["id"])
        for contribution in state.get("contributions", [])
        if contribution["agent"] == "research"
        for source in contribution["data"].get("sources", [])
        if isinstance(source, dict) and isinstance(source.get("id"), str)
    }
    citations = state.get("citations", [])
    errors: list[str] = []
    if research is None or not retrieved_ids:
        errors.append("Research evidence is missing.")
    if not citations:
        errors.append("Citations are missing.")
    if any(citation not in retrieved_ids for citation in citations):
        errors.append("A citation does not point to a retrieved source ID.")
    if answer is None or not state.get("final_answer"):
        errors.append("A synthesized answer is missing.")
    if analysis is None:
        errors.append("Calculation result is missing.")
    else:
        metrics = analysis["data"].get("metrics")
        required = (
            "macro_precision_at_k",
            "macro_recall_at_k",
            "precision_ceiling_at_k",
            "normalized_precision",
        )
        if not isinstance(metrics, dict) or any(key not in metrics for key in required):
            errors.append("Calculation metrics are incomplete.")
        else:
            for key in required:
                value = metrics[key]
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                    errors.append(f"Calculation metric {key} is invalid.")
            if metrics.get("source_id") not in retrieved_ids:
                errors.append("Calculation evidence is not among the retrieved source IDs.")

    used_refinement = state.get("refinement_used", False)
    valid = not errors
    if valid:
        destination = "END"
    elif used_refinement:
        destination = "safe_terminal"
    else:
        destination = "supervisor"
    event: RouteEvent = {
        "source": "validator",
        "destination": destination,
        "reason": "Validation passed." if valid else "; ".join(errors),
        "provider": "validator",
        "step": state.get("step_count", 0),
    }
    return {
        "is_valid": valid,
        "status": "complete" if valid else "running",
        "validation_errors": errors,
        "refinement_used": used_refinement or not valid,
        "route_trace": [event],
    }


def route_after_validation(state: AgentState) -> Literal["end", "supervisor", "safe_terminal"]:
    if state.get("is_valid"):
        return "end"
    if state.get("refinement_used") and state.get("validation_errors"):
        # A first failure marks refinement_used before returning to the supervisor.
        prior_validator_failures = sum(event["source"] == "validator" and event["destination"] != "END" for event in state.get("route_trace", []))
        if prior_validator_failures > 1:
            return "safe_terminal"
        return "supervisor"
    return "safe_terminal"


def safe_terminal_node(state: AgentState) -> dict[str, Any]:
    message = (
        "The workflow stopped safely because its step bound was reached or the result did not pass validation. "
        "Please review the request and evidence with a human."
    )
    return {
        "status": "needs_human",
        "final_answer": message,
        "messages": [AIMessage(content=message)],
    }


def build_graph(
    *,
    settings: Settings | None = None,
    router: Router | None = None,
    knowledge_base: KnowledgeBase | None = None,
    checkpointer: Any | None = None,
):
    """Build and compile the hierarchical graph with thread-scoped checkpointing."""
    runtime = settings or Settings.from_env()
    supervisor_router = router or SupervisorRouter(runtime)
    local_kb = knowledge_base or KnowledgeBase()

    async def supervisor(state: AgentState) -> dict[str, Any]:
        return await supervisor_node(state, router=supervisor_router, settings=runtime)

    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("research", make_research_agent(local_kb))
    builder.add_node("analysis", analysis_agent)
    builder.add_node("synthesis", synthesis_node)
    builder.add_node("validator", validator_node)
    builder.add_node("safe_terminal", safe_terminal_node)
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {"research": "research", "analysis": "analysis", "synthesis": "synthesis", "safe_terminal": "safe_terminal"},
    )
    builder.add_edge("research", "supervisor")
    builder.add_edge("analysis", "supervisor")
    builder.add_edge("synthesis", "validator")
    builder.add_conditional_edges(
        "validator",
        route_after_validation,
        {"end": END, "supervisor": "supervisor", "safe_terminal": "safe_terminal"},
    )
    builder.add_edge("safe_terminal", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())
