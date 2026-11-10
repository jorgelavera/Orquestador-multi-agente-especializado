"""Analysis specialist: validate retrieved evidence and compute metrics deterministically."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from orchestrator.state import AgentState, Contribution
from orchestrator.tools.calculations import calculate_macro_metrics


async def analysis_agent(state: AgentState) -> dict[str, Any]:
    evidence = [
        source
        for contribution in state.get("contributions", [])
        if contribution["agent"] == "research"
        for source in contribution["data"].get("sources", [])
        if isinstance(source, dict) and source.get("retrieval_fixture") is not None
    ]
    if not evidence:
        result: dict[str, Any] = {"error": "No retrieved source contains a valid metric fixture."}
        evidence_ids: list[str] = []
        summary = "Calculation unavailable: no fixture evidence was retrieved."
    else:
        evidence_ids = [str(evidence[0].get("id", ""))]
        try:
            result = calculate_macro_metrics(evidence[0])
            summary = "Computed ranking metrics from validated fixture evidence."
        except ValueError as exc:
            result = {"error": str(exc)}
            summary = f"Calculation validation failed: {exc}"
    contribution: Contribution = {
        "agent": "analysis",
        "kind": "calculation",
        "summary": summary,
        "evidence_ids": evidence_ids,
        "data": {"metrics": result},
    }
    return {
        "contributions": [contribution],
        "messages": [AIMessage(content=summary)],
    }
