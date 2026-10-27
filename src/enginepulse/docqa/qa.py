"""The DocQA engine: retrieval-only answers, optional LLM synthesis.

Retrieval mode returns the best passages verbatim with document/page citations
— no LLM, no API key, nothing leaves the machine.

LLM mode talks to any OpenAI-compatible /chat/completions endpoint you point
it at (set ENGINEPULSE_LLM_BASE_URL / OPENAI_API_KEY style config). If the
top retrieval score falls below ``min_score``, we refuse instead of hallucinating —
refusal accuracy is measured in eval.py, because a RAG system that never says
"I don't know" is lying about half the time.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass

from enginepulse.docqa.retrieve import BruteCosineRetriever, Retriever
from enginepulse.utils import get_logger

logger = get_logger(__name__)

REFUSAL_MESSAGE = (
    "I could not find anything in the indexed documents that answers this. "
    "Try `enginepulse docqa-index` on a folder containing the relevant manual."
)


@dataclass
class LLMConfig:
    base_url: str          # e.g. http://localhost:11434/v1 or https://api.openai.com/v1
    model: str             # e.g. "gpt-4o-mini", "qwen2.5:7b-instruct"
    api_key: str | None = None
    timeout_s: int = 60


def llm_config_from_env() -> LLMConfig | None:
    base = os.environ.get("ENGINEPULSE_LLM_BASE_URL")
    model = os.environ.get("ENGINEPULSE_LLM_MODEL")
    if not base or not model:
        return None
    return LLMConfig(base_url=base.rstrip("/"), model=model,
                     api_key=os.environ.get("ENGINEPULSE_LLM_API_KEY"))


class DocQAEngine:
    """Question -> {answer, citations, refusal}. Composes a Retriever + optional LLM."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMConfig | None = None,
        min_score: float = 0.08,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        # Threshold tuned on the sample corpus with the hash embedder; real
        # embedding models score much higher, so override per-corpus.
        self.min_score = min_score

    @classmethod
    def from_index(cls, index_path: str, llm: LLMConfig | None = None) -> "DocQAEngine":
        r = BruteCosineRetriever(index_path)
        return cls(r, llm=llm)

    def ask(self, question: str, k: int = 4) -> dict:
        hits = self.retriever.search(question, k=k)
        if not hits or hits[0][1] < self.min_score:
            return {"answer": REFUSAL_MESSAGE, "citations": [], "refusal": True}

        citations = [
            {"document": c.document, "page": c.page,
             "text": c.text[:400], "score": round(s, 4)}
            for c, s in hits
        ]

        if self.llm is None:
            # Retrieval-only mode: the passages ARE the answer, labelled by source.
            body = "\n\n".join(
                f"[{c['document']} p.{c['page']}] {c['text']}" for c in citations
            )
            return {"answer": f"Most relevant passages:\n\n{body}",
                    "citations": citations, "refusal": False}

        answer = self._llm_answer(question, citations)
        if answer is None:
            body = "\n\n".join(f"[{c['document']} p.{c['page']}] {c['text']}" for c in citations)
            return {"answer": f"(LLM unavailable — raw passages)\n\n{body}",
                    "citations": citations, "refusal": False}
        return {"answer": answer, "citations": citations, "refusal": False}

    def _llm_answer(self, question: str, citations: list[dict]) -> str | None:
        ctx = "\n\n".join(
            f"[source: {c['document']} page {c['page']}]\n{c['text']}" for c in citations
        )
        prompt = (
            "Answer ONLY from the context. If it is not in the context, reply exactly: "
            "'Not found in documents.' Cite sources as [doc p.page].\n\n"
            f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
        )
        payload = {
            "model": self.llm.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 512,
        }
        try:
            req = urllib.request.Request(
                f"{self.llm.base_url}/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json",
                         **({"Authorization": f"Bearer {self.llm.api_key}"} if self.llm.api_key else {})},
            )
            with urllib.request.urlopen(req, timeout=self.llm.timeout_s) as resp:
                out = json.loads(resp.read())
            return out["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.warning("LLM call failed (%s); falling back to retrieval-only.", exc)
            return None
