"""Retrieval over the JSON index built by ingest.py.

Retriever is an abstract base class (same pattern as RULModel): swap in a
ChromaDB- or Qdrant-backed implementation for big corpora without touching the
QA engine. The default, BruteCosineRetriever, is exact and embarrassingly fast
below ~50k chunks; MMR re-ranking trades a little relevance for diversity so
top-k passages do not all come from one page.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from enginepulse.docqa.chunk import Chunk
from enginepulse.docqa.ingest import Embedder


class Retriever(ABC):
    """Anything that can return top chunks for a query string."""

    @abstractmethod
    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        """Return [(chunk, score)] best-first."""


class BruteCosineRetriever(Retriever):
    def __init__(self, index_path: str | Path, embedder: Embedder | None = None) -> None:
        payload = json.loads(Path(index_path).read_text())
        self.chunks = [Chunk(**c) for c in payload["chunks"]]
        self.vectors = np.asarray(payload["vectors"], dtype=np.float32)
        self.embedding_model = payload["embedding_model"]
        # Queries must be embedded with the SAME model as the index, or the
        # vector spaces will not line up — fall back config wins here.
        self.embedder = embedder or Embedder(self.embedding_model)
        if self.embedding_model != self.embedder.model_name:
            raise ValueError(
                f"index was built with '{self.embedding_model}' but embedder is "
                f"'{self.embedder.model_name}'; rebuild the index or pass a matching Embedder."
            )

    def _scores(self, query: str) -> np.ndarray:
        q = self.embedder.encode([query])[0]
        # Vectors are stored normalised; cosine == dot product.
        return self.vectors @ q

    def search(self, query: str, k: int, mmr: bool = True, lambda_: float = 0.65) -> list[tuple[Chunk, float]]:
        scores = self._scores(query)
        pool_n = min(max(4 * k, 12), len(self.chunks))
        cand = np.argsort(-scores)[:pool_n]
        if not mmr:
            return [(self.chunks[i], float(scores[i])) for i in cand[:k]]

        # MMR on candidate vectors: relevance minus redundancy against chosen set.
        cand_vecs = self.vectors[cand]
        selected: list[int] = []
        remaining = list(range(len(cand)))
        while len(selected) < k and remaining:
            best_i, best_val = None, -np.inf
            for ri in remaining:
                rel = float(scores[cand[ri]])
                red = max((float(cand_vecs[ri] @ cand_vecs[s]) for s in selected), default=0.0)
                val = lambda_ * rel - (1.0 - lambda_) * red
                if val > best_val:
                    best_i, best_val = ri, val
            selected.append(best_i)
            remaining.remove(best_i)
        return [(self.chunks[cand[s]], float(scores[cand[s]])) for s in selected]
