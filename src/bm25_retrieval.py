"""BM25-based lexical retrieval."""

from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None
        self._tokenized: list[list[str]] = []

    def build(self, documents: list[dict[str, Any]]) -> None:
        self.documents = list(documents or [])
        self._tokenized = [_tokenize(doc.get("text", "")) for doc in self.documents]
        if self._tokenized and any(self._tokenized):
            self._bm25 = BM25Okapi(self._tokenized)
        else:
            self._bm25 = None

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.documents or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(
            range(len(self.documents)),
            key=lambda i: scores[i],
            reverse=True,
        )
        results: list[dict[str, Any]] = []
        for i in ranked[:top_k]:
            score = float(scores[i])
            if score <= 0.0:
                continue
            doc = self.documents[i]
            result = dict(doc)
            result["bm25_score"] = score
            results.append(result)
        return results