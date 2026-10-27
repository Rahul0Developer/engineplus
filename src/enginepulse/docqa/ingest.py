"""PDF ingestion: extract text per page, chunk it, build an index.

Uses pypdf when available; falls back to plain .txt/.md files so tests and
demo corpora do not need binary fixtures. Embeddings come from
sentence-transformers if installed, otherwise a deterministic hashed bag-of-
words projection (clearly labelled as 'hash-ngram-v0') — retrieval quality
drops, but the pipeline runs anywhere with zero downloads.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from enginepulse.docqa.chunk import Chunk, chunk_pages
from enginepulse.utils import get_logger

logger = get_logger(__name__)

FALLBACK_EMBEDDING_MODEL = "hash-ngram-v0"


def extract_pages(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number 1-based, text)] for pdf/txt/md files."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        text = path.read_text(errors="replace")
        # Heuristic page markers so txt corpora still get page citations.
        parts = text.split("\f")
        return [(i + 1, p) for i, p in enumerate(parts)]
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError("PDF ingestion needs `pip install 'enginepulse[docqa]'`") from exc
        reader = PdfReader(str(path))
        return [(i + 1, (pg.extract_text() or "")) for i, pg in enumerate(reader.pages)]
    raise ValueError(f"Unsupported file type: {path}")


def fallback_embed(texts: list[str], dim: int = 256) -> "object":
    """Deterministic hashed word/bigram embedding. No ML dependencies.

    This is NOT competitive with real embeddings; it exists so retrieval-only
    mode works on a laptop with no downloads and so the eval harness can run
    in CI. The index records which model produced it.
    """
    import numpy as np

    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        toks = [w for w in "".join(c if c.isalnum() else " " for c in t.lower()).split()]
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            idx = h % dim
            out[i, idx] += 1.0 if (h >> 63) & 1 else -1.0
        norm = np.linalg.norm(out[i])
        if norm > 0:
            out[i] /= norm
    return out


class Embedder:
    """Wraps sentence-transformers if present; otherwise the hash fallback."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or FALLBACK_EMBEDDING_MODEL
        self._st = None
        if self.model_name != FALLBACK_EMBEDDING_MODEL:
            try:
                from sentence_transformers import SentenceTransformer

                self._st = SentenceTransformer(self.model_name)
            except ImportError as exc:
                raise ImportError(
                    f"embedding model '{self.model_name}' needs "
                    "`pip install 'enginepulse[docqa]'`"
                ) from exc

    def encode(self, texts: list[str]):
        if self._st is not None:
            e = self._st.encode(texts, normalize_embeddings=True)
            return e.astype("float32")
        return fallback_embed(texts)

    @property
    def dim(self) -> int:
        return 256 if self._st is None else self._st.get_sentence_embedding_dimension()


def build_index(
    source_dir: str | Path,
    out_path: str | Path,
    model_name: str | None = None,
    max_tokens: int = 240,
    overlap_tokens: int = 40,
) -> dict:
    """Ingest every supported doc in a folder into a JSON index file.

    We ship our own tiny index format (chunks + vectors) instead of requiring
    ChromaDB: at portfolio-corpus scale (<10k chunks) brute-force cosine is
    instant, dependency-free and diffable. ``retrieve.py`` documents how to
    swap in ChromaDB behind the same Retriever interface for larger corpora.
    """
    src = Path(source_dir)
    files = sorted([p for p in src.rglob("*") if p.suffix.lower() in (".pdf", ".txt", ".md")])
    if not files:
        raise FileNotFoundError(f"No ingestible documents (.pdf/.txt/.md) found in {src}")

    all_chunks: list[Chunk] = []
    for f in files:
        pages = extract_pages(f)
        all_chunks.extend(chunk_pages(pages, document=f.name,
                                      max_tokens=max_tokens, overlap_tokens=overlap_tokens))
    logger.info("Chunked %d documents into %d passages", len(files), len(all_chunks))

    emb = Embedder(model_name)
    vectors = emb.encode([c.text for c in all_chunks])

    payload = {
        "embedding_model": emb.model_name,
        "chunk_config": {"max_tokens": max_tokens, "overlap_tokens": overlap_tokens},
        "chunks": [
            {"id": c.id, "text": c.text, "document": c.document,
             "page": c.page, "heading": c.heading}
            for c in all_chunks
        ],
        "vectors": [[float(x) for x in row] for row in vectors],
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    logger.info("Wrote index (%d chunks) to %s", len(all_chunks), out)
    return payload
