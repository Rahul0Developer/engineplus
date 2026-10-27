"""DocQA evaluation harness.

Question set format (JSONL, one object per line):
    {"question": "...", "answerable": true,
     "evidence_doc": "manual.txt", "evidence_page": 2,
     "must_contain": ["400 hours"]}

Metrics:
  hit@k            : gold document+page appears in the top-k retrieved chunks
  MRR              : mean reciprocal rank of the first gold hit
  citation_validity: fraction of returned citations that exist in the index
                     (document + page actually present)
  refusal_accuracy : on answerable=false questions, how often we refuse
  faithfulness     : on answerable=true, fraction where all must_contain
                     strings appear in the returned passages (cheap proxy for
                     "the answer is actually supported"; a manual review of a
                     sample should accompany this number, see REPORT.md)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from enginepulse.docqa.qa import DocQAEngine


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    qs = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            obj = json.loads(line)
            obj.setdefault("answerable", True)
            qs.append(obj)
    return qs


def evaluate(engine: DocQAEngine, questions: list[dict], k: int = 4) -> dict[str, float]:
    n_hit = n_rr = 0.0
    n_cites = n_valid_cites = 0
    n_unans = n_correct_refusal = 0
    n_ans = n_faithful = 0

    # Valid (doc, page) pairs from the index itself — citations must be real.
    valid_pairs: set[tuple[str, int]] = set()
    chunks = getattr(engine.retriever, "chunks", [])
    for c in chunks:
        valid_pairs.add((c.document, c.page))

    for q in questions:
        res = engine.ask(q["question"], k=k)
        if q.get("answerable", True):
            n_ans += 1
            gold_doc, gold_page = q.get("evidence_doc"), q.get("evidence_page")
            ranks = [i + 1 for i, c in enumerate(res["citations"])
                     if c["document"] == gold_doc and c["page"] == gold_page]
            if ranks:
                n_hit += 1
                n_rr += 1.0 / ranks[0]
            must = q.get("must_contain", [])
            joined = " ".join(c["text"].lower() for c in res["citations"])
            if must and all(m.lower() in joined for m in must):
                n_faithful += 1
        else:
            n_unans += 1
            if res["refusal"]:
                n_correct_refusal += 1

        for c in res["citations"]:
            n_cites += 1
            if not valid_pairs or (c["document"], c["page"]) in valid_pairs:
                n_valid_cites += 1

    n_q = len(questions)
    return {
        "n_questions": n_q,
        "hit@k": n_hit / max(n_ans, 1),
        "mrr": n_rr / max(n_ans, 1),
        "citation_validity": n_valid_cites / max(n_cites, 1),
        "refusal_accuracy": n_correct_refusal / max(n_unans, 1) if n_unans else float("nan"),
        "faithfulness_must_contain": n_faithful / max(n_ans, 1),
    }
