from __future__ import annotations

import pytest

from orchestrator.tools.calculations import calculate_macro_metrics
from orchestrator.tools.knowledge_base import KnowledgeBase, hash_embedding, search_local_corpus


def test_chroma_search_returns_hydrated_source_ids() -> None:
    knowledge_base = KnowledgeBase()

    results = search_local_corpus("retrieval Precision@5 fixture relevant documents", knowledge_base)

    assert results
    assert "eval-fixture-v1" in {result["id"] for result in results}
    fixture = next(result for result in results if result["id"] == "eval-fixture-v1")
    assert fixture["retrieval_fixture"]["k"] == 5
    assert fixture["distance"] >= 0


def test_hash_embedding_is_repeatable_and_normalized() -> None:
    first = hash_embedding("stable local vector")
    second = hash_embedding("stable local vector")

    assert first == second
    assert sum(value * value for value in first) == pytest.approx(1.0)


def test_calculation_computes_expected_macro_metrics() -> None:
    source = next(
        item for item in KnowledgeBase().search("retrieval evaluation fixture") if item["id"] == "eval-fixture-v1"
    )

    metrics = calculate_macro_metrics(source)

    assert metrics == {
        "source_id": "eval-fixture-v1",
        "k": 5,
        "query_count": 3,
        "macro_precision_at_k": 0.4,
        "macro_recall_at_k": 0.6389,
        "precision_ceiling_at_k": 0.6,
        "normalized_precision": 0.6667,
    }


@pytest.mark.parametrize(
    "fixture, message",
    [
        ({"k": 0, "queries": [{"relevant_total": 1, "relevant_retrieved": 1}]}, "k must be"),
        ({"k": 5, "queries": [{"relevant_total": 0, "relevant_retrieved": 0}]}, "relevant_total must be positive"),
        ({"k": 5, "queries": [{"relevant_total": 2, "relevant_retrieved": 3}]}, "exceeds its valid maximum"),
    ],
)
def test_calculation_rejects_invalid_inputs(fixture: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        calculate_macro_metrics({"id": "bad-fixture", "retrieval_fixture": fixture})
