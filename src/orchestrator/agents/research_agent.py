"""Research specialist: retrieve local evidence and preserve source attribution."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from orchestrator.state import AgentState, Contribution
from orchestrator.tools.knowledge_base import KnowledgeBase, search_local_corpus


def make_research_agent(knowledge_base: KnowledgeBase):
    async def research_agent(state: AgentState) -> dict[str, Any]:
        # The specialist gets the bounded task and request, not the full message history.
        task = f"Find local corpus evidence and source IDs for this request: {state['request']}"
        sources = search_local_corpus(task, knowledge_base)
        contribution: Contribution = {
            "agent": "research",
            "kind": "evidence",
            "summary": f"Retrieved {len(sources)} Chroma source(s).",
            "evidence_ids": [source["id"] for source in sources],
            "data": {"sources": sources, "task": task},
        }
        return {
            "contributions": [contribution],
            "messages": [AIMessage(content=f"Research retrieved {len(sources)} source(s).")],
        }

    return research_agent
