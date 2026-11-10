"""Chroma-backed local corpus search with deterministic hash embeddings."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb


DEFAULT_CORPUS = Path(__file__).resolve().parents[3] / "data" / "sources.json"
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_DIMENSIONS = 96


def hash_embedding(text: str) -> list[float]:
    """Make a stable, dependency-free embedding suitable for this tiny demo corpus."""
    vector = [0.0] * _DIMENSIONS
    tokens = _TOKEN_PATTERN.findall(text.lower())
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % _DIMENSIONS
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


@dataclass
class KnowledgeBase:
    """A local Chroma vector collection hydrated from the included JSON corpus."""

    corpus_path: Path = DEFAULT_CORPUS

    def __post_init__(self) -> None:
        raw: list[dict[str, Any]] = json.loads(self.corpus_path.read_text(encoding="utf-8"))
        self._sources = {source["id"]: source for source in raw}
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(
            name="research_corpus",
            metadata={"hnsw:space": "cosine"},
        )
        ids = list(self._sources)
        if ids:
            self._collection.upsert(
                ids=ids,
                documents=[f"{item['title']} {item['text']}" for item in self._sources.values()],
                embeddings=[hash_embedding(f"{item['title']} {item['text']}") for item in self._sources.values()],
                metadatas=[{"title": item["title"]} for item in self._sources.values()],
            )

    def search(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        """Search Chroma and return hydrated, source-attributed evidence records."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit < 1:
            raise ValueError("limit must be positive")
        result = self._collection.query(
            query_embeddings=[hash_embedding(query)],
            n_results=min(limit, len(self._sources)),
            include=["distances", "documents", "metadatas"],
        )
        ids = result["ids"][0]
        distances = result.get("distances", [[]])[0]
        records: list[dict[str, Any]] = []
        for source_id, distance in zip(ids, distances, strict=True):
            source = self._sources[source_id]
            records.append(
                {
                    "id": source_id,
                    "title": source["title"],
                    "text": source["text"],
                    "distance": float(distance),
                    "retrieval_fixture": source.get("retrieval_fixture"),
                }
            )
        return records


def search_local_corpus(query: str, knowledge_base: KnowledgeBase) -> list[dict[str, Any]]:
    """Research tool entry point: performs actual vector search and returns source IDs."""
    return knowledge_base.search(query)
