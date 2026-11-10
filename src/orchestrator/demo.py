"""Run the notebook's end-to-end request and expose the persisted thread state."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.messages import HumanMessage

from orchestrator.config import Settings
from orchestrator.graph import build_graph


DEMO_REQUEST = (
    "Using the local retrieval-evaluation fixture, report macro Precision@5, macro Recall@5, "
    "the attainable Precision@5 ceiling, and normalized precision. Cite the source IDs."
)


async def run_demo(request: str = DEMO_REQUEST, thread_id: str = "pre-entrega-6-demo") -> dict[str, Any]:
    settings = Settings.from_env()
    graph = build_graph(settings=settings)
    config = {"configurable": {"thread_id": thread_id}}
    result = await graph.ainvoke(
        {
            "request": request,
            "messages": [HumanMessage(content=request)],
            "contributions": [],
            "route_trace": [],
            "step_count": 0,
            "max_steps": settings.max_steps,
            "required_agents": ["research", "analysis"],
            "refinement_used": False,
            "validation_errors": [],
            "status": "running",
        },
        config=config,
    )
    checkpoint = await graph.aget_state(config)
    return {
        "request": request,
        "thread_id": thread_id,
        "status": result.get("status"),
        "final_answer": result.get("final_answer"),
        "contributions": result.get("contributions", []),
        "route_trace": result.get("route_trace", []),
        "step_count": result.get("step_count"),
        "checkpoint_preserved": checkpoint.values.get("final_answer") == result.get("final_answer"),
        "checkpoint_thread_id": checkpoint.config.get("configurable", {}).get("thread_id"),
    }


def main() -> None:
    print(json.dumps(asyncio.run(run_demo()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
