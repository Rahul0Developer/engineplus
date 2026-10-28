"""CLI wiring tests: argument parsing + offline subcommands.

The heavy subcommands (train/benchmark) are exercised through the trainer
tests; here we only check the parser contract and the commands that need no
network or GPU, so CI stays fast.
"""

from __future__ import annotations

import json

import pytest

from enginepulse.cli import build_parser, main


def test_parser_requires_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_train_parses_common_flags():
    args = build_parser().parse_args(
        ["train", "--model", "gru", "--dataset", "FD002", "--seed", "7",
         "--window-len", "32", "--max-rul", "150"]
    )
    assert args.command == "train"
    assert (args.model, args.dataset, args.seed) == ("gru", "FD002", 7)
    assert (args.window_len, args.max_rul) == (32, 150)


def test_benchmark_defaults():
    args = build_parser().parse_args(["benchmark"])
    assert args.datasets == "FD001,FD002,FD003,FD004"
    assert args.seeds == "42"


def test_main_models_lists_registered(capsys):
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    for name in ("baseline", "ridge", "xgb", "cnn", "gru"):
        assert name in out


def test_main_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "enginepulse" in capsys.readouterr().out


def test_ask_refuses_without_index(tmp_path, capsys):
    rc = main(["ask", "what is a borescope?", "--index", str(tmp_path / "nope.json")])
    assert rc == 1
    assert "docqa-index" in capsys.readouterr().err


def test_docqa_index_and_ask_roundtrip(tmp_path, capsys):
    """Index a tiny text 'manual', ask it a question — retrieval-only, no LLM."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "manual.txt").write_text(
        "Inspection Intervals\n"
        "The low-pressure turbine must be borescoped every 400 operating hours.\n\n"
        "Oil Change\n"
        "Replace lubricant every 250 hours or after any shutdown longer than 30 days.\n"
    )
    index = tmp_path / "idx.json"
    assert main(["docqa-index", "--docs-dir", str(docs), "--out", str(index)]) == 0
    assert index.exists()

    assert main(["ask", "When must the LPT be borescoped?", "--index", str(index)]) == 0
    out = capsys.readouterr().out
    assert "400" in out  # the answer lives in the retrieved passage


def test_docqa_eval_on_tiny_question_set(tmp_path, capsys):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "manual.txt").write_text(
        "Filters\nAir filters are serviced every 100 cycles.\n"
        "Valves\nFuel valves are leak-tested at every 500-cycle inspection.\n"
    )
    index = tmp_path / "idx.json"
    main(["docqa-index", "--docs-dir", str(docs), "--out", str(index)])

    qs = tmp_path / "q.jsonl"
    rows = [
        {"question": "How often are air filters serviced?", "answerable": True,
         "must_contain": ["100"]},
        {"question": "What is the capital of France?", "answerable": False},
    ]
    qs.write_text("\n".join(json.dumps(r) for r in rows))
    assert main(["docqa-eval", "--index", str(index), "--questions", str(qs)]) == 0
    summary = capsys.readouterr().out
    assert "hit@" in summary
