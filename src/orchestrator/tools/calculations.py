"""Validated deterministic ranking-metric computation."""

from __future__ import annotations

from typing import Any


def calculate_macro_metrics(source: dict[str, Any]) -> dict[str, Any]:
    """Compute macro P@k, macro recall, attainable ceiling, and normalized P@k."""
    source_id = source.get("id")
    fixture = source.get("retrieval_fixture")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("evidence source must have a non-empty id")
    if not isinstance(fixture, dict):
        raise ValueError(f"source {source_id!r} has no retrieval fixture")
    k = fixture.get("k")
    queries = fixture.get("queries")
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("fixture k must be a positive integer")
    if not isinstance(queries, list) or not queries:
        raise ValueError("fixture queries must be a non-empty list")

    precisions: list[float] = []
    recalls: list[float] = []
    ceilings: list[float] = []
    for index, row in enumerate(queries):
        if not isinstance(row, dict):
            raise ValueError(f"query row {index} must be an object")
        relevant_total = row.get("relevant_total")
        relevant_retrieved = row.get("relevant_retrieved")
        for field, value in (("relevant_total", relevant_total), ("relevant_retrieved", relevant_retrieved)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"query row {index} {field} must be a non-negative integer")
        if relevant_total == 0:
            raise ValueError(f"query row {index} relevant_total must be positive")
        if relevant_retrieved > min(k, relevant_total):
            raise ValueError(f"query row {index} relevant_retrieved exceeds its valid maximum")
        precisions.append(relevant_retrieved / k)
        recalls.append(relevant_retrieved / relevant_total)
        ceilings.append(min(relevant_total, k) / k)

    macro_precision = sum(precisions) / len(precisions)
    macro_recall = sum(recalls) / len(recalls)
    ceiling = sum(ceilings) / len(ceilings)
    return {
        "source_id": source_id,
        "k": k,
        "query_count": len(queries),
        "macro_precision_at_k": round(macro_precision, 4),
        "macro_recall_at_k": round(macro_recall, 4),
        "precision_ceiling_at_k": round(ceiling, 4),
        "normalized_precision": round(macro_precision / ceiling, 4),
    }
