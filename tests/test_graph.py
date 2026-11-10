from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import HumanMessage

import orchestrator.graph as graph_module
from orchestrator.config import Settings
from orchestrator.graph import SupervisorRouter, build_graph, route_after_validation, validator_node
from orchestrator.state import RouteName


class OrderedRouter:
    def __init__(self) -> None:
        self.choices: list[str] = []

    async def choose(
        self,
        *,
        request: str,
        summary: dict[str, Any],
        candidates: list[RouteName],
        fallback: RouteName,
    ) -> tuple[RouteName, str, str]:
        choice = candidates[0]
        self.choices.append(choice)
        return choice, f"fake router selected {choice}", "test-fake"


def initial_state(request: str, max_steps: int = 8) -> dict[str, Any]:
    return {
        "request": request,
        "messages": [HumanMessage(content=request)],
        "contributions": [],
        "route_trace": [],
        "step_count": 0,
        "max_steps": max_steps,
        "required_agents": ["research", "analysis"],
        "refinement_used": False,
        "validation_errors": [],
        "status": "running",
    }


def test_ollama_client_never_receives_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    received: dict[str, Any] = {}

    def fake_client(**kwargs: Any) -> object:
        received.update(kwargs)
        return object()

    monkeypatch.setenv("OPENAI_API_KEY", "user-secret")
    monkeypatch.setattr("orchestrator.graph.AsyncOpenAI", fake_client)

    SupervisorRouter(Settings(provider="ollama"))

    assert received["api_key"] == "ollama"
    assert "user-secret" not in received.values()


def test_ollama_client_rejects_remote_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("orchestrator.graph.AsyncOpenAI", lambda **_: pytest.fail("client must not be created"))

    with pytest.raises(ValueError, match="localhost/loopback"):
        SupervisorRouter(Settings(provider="ollama", base_url="https://api.example.com/v1"))


@pytest.mark.asyncio
async def test_routes_research_analysis_synthesis_and_validates_checkpoint() -> None:
    router = OrderedRouter()
    graph = build_graph(settings=Settings(provider="offline"), router=router)
    config = {"configurable": {"thread_id": "test-research-analysis"}}

    result = await graph.ainvoke(
        initial_state("Compute retrieval fixture metrics and cite sources."),
        config=config,
    )

    assert router.choices == ["research", "analysis", "synthesis"]
    assert result["status"] == "complete"
    assert result["is_valid"] is True
    assert [item["agent"] for item in result["contributions"]] == ["research", "analysis", "synthesis"]
    research = result["contributions"][0]
    analysis = result["contributions"][1]
    assert research["evidence_ids"]
    assert analysis["evidence_ids"] == ["eval-fixture-v1"]
    assert set(result["citations"]) == {"metric-definitions-v1", "eval-fixture-v1"}
    assert result["step_count"] == 3
    assert [(event["source"], event["destination"]) for event in result["route_trace"]] == [
        ("supervisor", "research"),
        ("supervisor", "analysis"),
        ("supervisor", "synthesis"),
        ("validator", "END"),
    ]

    saved = await graph.aget_state(config)
    assert saved.values["final_answer"] == result["final_answer"]
    assert saved.config["configurable"]["thread_id"] == "test-research-analysis"


@pytest.mark.asyncio
async def test_failed_validation_gets_one_targeted_refinement_before_revalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_synthesis = graph_module.synthesis_node
    synthesis_calls = 0

    async def fail_first_synthesis(state: dict[str, Any]) -> dict[str, Any]:
        nonlocal synthesis_calls
        result = await original_synthesis(state)
        synthesis_calls += 1
        if synthesis_calls == 1:
            return {**result, "citations": [*result["citations"], "not-retrieved"]}
        return result

    monkeypatch.setattr(graph_module, "synthesis_node", fail_first_synthesis)
    router = OrderedRouter()
    graph = build_graph(settings=Settings(provider="offline"), router=router)

    result = await graph.ainvoke(
        initial_state("Compute retrieval fixture metrics and cite sources."),
        config={"configurable": {"thread_id": "test-single-refinement"}},
    )

    assert result["status"] == "complete"
    assert result["is_valid"] is True
    assert synthesis_calls == 2
    assert router.choices == ["research", "analysis", "synthesis", "research", "synthesis"]
    assert [event["destination"] for event in result["route_trace"] if event["source"] == "supervisor"] == [
        "research",
        "analysis",
        "synthesis",
        "research",
        "synthesis",
    ]
    validator_events = [event for event in result["route_trace"] if event["source"] == "validator"]
    assert len(validator_events) == 2
    assert "citation does not point to a retrieved source ID" in validator_events[0]["reason"]
    assert validator_events[1]["destination"] == "END"


@pytest.mark.asyncio
async def test_maximum_step_guard_stops_a_repeated_route() -> None:
    router = OrderedRouter()
    graph = build_graph(settings=Settings(provider="offline", max_steps=1), router=router)

    result = await graph.ainvoke(
        initial_state("This request deliberately cannot finish within one supervisor step.", max_steps=1),
        config={"configurable": {"thread_id": "test-step-bound"}},
    )

    assert result["status"] == "needs_human"
    assert result["step_count"] == 1
    assert router.choices == ["research"]
    assert result["route_trace"][-1]["destination"] == "safe_terminal"
    assert len(result["route_trace"]) <= 3


def test_validator_allows_only_one_refinement() -> None:
    invalid: dict[str, Any] = {
        "contributions": [],
        "route_trace": [],
        "citations": ["not-retrieved"],
        "final_answer": "Unsupported answer.",
        "step_count": 1,
        "refinement_used": False,
        "validation_errors": [],
    }

    first = validator_node(invalid)  # type: ignore[arg-type]
    after_first = {**invalid, **first, "route_trace": first["route_trace"]}
    assert route_after_validation(after_first) == "supervisor"  # type: ignore[arg-type]

    second = validator_node(after_first)  # type: ignore[arg-type]
    after_second = {
        **after_first,
        **second,
        "route_trace": after_first["route_trace"] + second["route_trace"],
    }
    assert route_after_validation(after_second) == "safe_terminal"  # type: ignore[arg-type]
