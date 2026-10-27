"""Local document Q&A over maintenance PDFs (optional extra: [docqa]).

Retrieval-only mode works with NO API key and no LLM. LLM mode talks to any
OpenAI-compatible endpoint (including local servers like llama.cpp/ollama's
OpenAI shim) — the base URL and model name are user-supplied, nothing is sent
to us.
"""

from enginepulse.docqa.qa import DocQAEngine

__all__ = ["DocQAEngine"]
