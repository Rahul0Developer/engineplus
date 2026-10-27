"""Heading-aware text chunking.

A "heading" here is a short ALL-CAPS or Title-Case line followed by body text
— the pattern that survives PDF->text conversion for most manuals. Chunks keep
their heading path ("Section 3 > 3.2 Inspection") so citations can show where
the passage came from, and overlap keeps sentences split across boundaries
retrievable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    document: str
    page: int
    heading: str = ""
    id: str = field(default="")

    def citation(self) -> dict:
        return {"document": self.document, "page": self.page,
                "heading": self.heading, "text": self.text}


_HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+)*\s+)?[A-Z][A-Za-z0-9 ,\-/&()]{2,60}$")


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 61 or line.endswith("."):
        return False
    words = line.split()
    if len(words) > 8:
        return False
    if line.isupper():
        return True
    return bool(_HEADING_RE.match(line))


def chunk_pages(
    pages: list[tuple[int, str]],
    document: str,
    max_tokens: int = 240,
    overlap_tokens: int = 40,
) -> list[Chunk]:
    """Split (page_no, text) pairs into overlapping chunks that never cross a page.

    Keeping chunks page-boundary-clean is deliberate: it makes every citation
    verifiable against one physical page, which matters more than perfect
    semantic continuity for maintenance docs.
    """
    chunks: list[Chunk] = []
    for page_no, text in pages:
        heading = ""
        buf: list[str] = []
        buf_len = 0

        def flush():
            nonlocal buf, buf_len
            if not buf:
                return
            body = " ".join(buf).strip()
            if body:
                cid = f"{document}:{page_no}:{len(chunks)}"
                chunks.append(Chunk(text=body, document=document, page=page_no,
                                    heading=heading, id=cid))
            # Overlap: carry the tail of this chunk into the next one.
            tail_words = body.split()[-overlap_tokens:] if overlap_tokens else []
            buf = [" ".join(tail_words)] if tail_words else []
            buf_len = sum(len(w) for w in tail_words)

        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            if _looks_like_heading(s):
                flush()
                heading = s
                continue
            words = s.split()
            for wd in words:
                if buf_len + len(wd) > max_tokens * 5:  # ~5 chars per token
                    flush()
                buf.append(wd)
                buf_len += len(wd) + 1
        flush()
    return chunks
