"""EnginePulse command-line interface.

Subcommands mirror the pipeline stages, so you can run one step at a time or
the whole thing:

    enginepulse download
    enginepulse preprocess --dataset FD001
    enginepulse train --model ridge --dataset FD001 [--config configs/ridge.yaml]
    enginepulse evaluate --dataset FD001 --model ridge
    enginepulse benchmark [--models ridge,xgb] [--datasets FD001] [--seeds 42,43]
    enginepulse explain --dataset FD001 --model xgb
    enginepulse serve [--models-dir ...] [--port 8000]
    enginepulse docqa-index --docs-dir examples/manuals --out docqa_index.json
    enginepulse ask "When must the LPT be borescoped?" --index docqa_index.json
    enginepulse docqa-eval --index docqa_index.json --questions examples/questions.jsonl

`train --config X.yaml` merges: defaults < YAML < explicit CLI flags.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import enginepulse
from enginepulse.utils import get_logger, load_config

logger = get_logger("enginepulse.cli")


def _version_string() -> str:
    """Package version plus git hash when running from a checkout."""
    from enginepulse.utils import git_hash

    h = git_hash()
    return f"enginepulse {enginepulse.__version__}" + (f" ({h})" if h != "unknown" else "")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--window-len", type=int, default=None)
    p.add_argument("--max-rul", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", default="benchmarks/results")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="enginepulse", description=__doc__.splitlines()[0])
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--version", action="version", version=_version_string())
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="Fetch/verify the NASA C-MAPSS files")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--datasets", default="FD001,FD002,FD003,FD004")

    p = sub.add_parser("preprocess", help="Run the leakage-safe pipeline and report shapes")
    _add_common(p)

    p = sub.add_parser("train", help="Train one model on one dataset")
    _add_common(p)
    p.add_argument("--model", required=True,
                   help="one of: baseline|ridge|xgb|cnn|gru|transformer (see 'models')")
    p.add_argument("--config", default=None, help="YAML experiment config")
    p.add_argument("--epochs", type=int, default=None, help="deep models only")
    p.add_argument("--mc-dropout", action="store_true",
                   help="enable MC-dropout intervals on gru/cnn")
    p.add_argument("--no-artifacts", action="store_true")

    p = sub.add_parser("evaluate", help="Retrain + evaluate; prints metrics JSON")
    _add_common(p)
    p.add_argument("--model", required=True)
    p.add_argument("--config", default=None)

    p = sub.add_parser("benchmark", help="Train every model on every dataset, write the table")
    p.add_argument("--models", default=None, help="comma list; default all available")
    p.add_argument("--datasets", default="FD001,FD002,FD003,FD004")
    p.add_argument("--seeds", default="42")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="benchmarks/results")
    p.add_argument("--table-out", default="benchmarks/results/table.md")
    p.add_argument("--config", default=None)

    p = sub.add_parser("explain", help="Per-sensor importance for a trained classical model")
    _add_common(p)
    p.add_argument("--model", default="xgb")
    p.add_argument("--top", type=int, default=8)

    p = sub.add_parser("serve", help="Start the FastAPI server")
    p.add_argument("--models-dir", default="benchmarks/results/artifacts")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    p = sub.add_parser("docqa-index", help="Build the DocQA vector index from a docs folder")
    p.add_argument("--docs-dir", required=True)
    p.add_argument("--out", default="docqa_index.json")
    p.add_argument("--embedding-model", default=None,
                   help="sentence-transformers name; default = offline hash embedder")

    p = sub.add_parser("ask", help="Query the DocQA index (retrieval-only unless LLM env is set)")
    p.add_argument("question")
    p.add_argument("--index", default="docqa_index.json")
    p.add_argument("--k", type=int, default=4)

    p = sub.add_parser("docqa-eval", help="Evaluate DocQA against a JSONL question set")
    p.add_argument("--index", default="docqa_index.json")
    p.add_argument("--questions", required=True)
    p.add_argument("--k", type=int, default=4)

    sub.add_parser("models", help="List models available in this environment")
    return ap


def _cfg_from_args(args, extra_keys=()) -> dict:
    cfg = {}
    if getattr(args, "config", None):
        raw = load_config(args.config)
        # YAML nests model kwargs under `model_params`; keep them separate.
        cfg = {k: v for k, v in raw.items() if k != "model_params"}
        cfg["model_params"] = raw.get("model_params", {})
    # Explicit flags beat the file — that is how you ablate quickly.
    for key in ("window_len", "max_rul", "seed", "epochs"):
        v = getattr(args, key, None)
        if v is not None:
            if key == "epochs":
                cfg.setdefault("model_params", {})[key] = v
            else:
                cfg[key] = v
    if getattr(args, "mc_dropout", False):
        cfg.setdefault("model_params", {})["mc_dropout"] = True
    del extra_keys
    return cfg


def cmd_train(args, evaluate_only: bool = False) -> int:
    from enginepulse.train.trainer import TrainConfig, train_and_evaluate

    d = _cfg_from_args(args)
    tc = TrainConfig(
        dataset=args.dataset,
        model=args.model,
        seed=d.get("seed", 42),
        window_len=d.get("window_len", 24),
        max_rul=d.get("max_rul", 125),
        val_fraction=d.get("val_fraction", 0.15),
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_params=d.get("model_params", {}),
        save_artifacts=not evaluate_only and not getattr(args, "no_artifacts", False),
    )
    res = train_and_evaluate(tc)
    print(json.dumps(res["metrics"], indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verbose:
        logger.setLevel(10)  # DEBUG

    try:
        if args.command == "download":
            from enginepulse.data.download import download_all

            ds = tuple(s.strip() for s in args.datasets.split(","))
            download_all(args.data_dir, ds)
            print(f"C-MAPSS ready in {args.data_dir}")

        elif args.command == "preprocess":
            from enginepulse.data.loader import load_dataset
            from enginepulse.data.preprocessing import prepare_dataset

            d = _cfg_from_args(args)
            ds = load_dataset(args.data_dir, args.dataset)
            prep = prepare_dataset(ds, window_len=d.get("window_len", 24),
                                   max_rul=d.get("max_rul", 125),
                                   seed=d.get("seed", 42))
            overlap = set(prep.train_units) & set(prep.val_units)
            print(json.dumps({
                "dataset": prep.dataset,
                "features": len(prep.feature_cols),
                "train_windows": len(prep.train_X),
                "val_windows": len(prep.val_X),
                "test_engines": len(prep.test_X),
                "train_val_engine_overlap": sorted(int(u) for u in overlap),
                "meta": prep.meta,
            }, indent=2, default=str))
            if overlap:
                print("LEAKAGE DETECTED — this should never happen; please file an issue.",
                      file=sys.stderr)
                return 2

        elif args.command in ("train", "evaluate"):
            return cmd_train(args, evaluate_only=(args.command == "evaluate"))

        elif args.command == "benchmark":
            from enginepulse.models.registry import available_models
            from enginepulse.train.benchmark import make_table, run_benchmark, write_table

            d = _cfg_from_args(args)
            models = [m.strip() for m in args.models.split(",")] if args.models else None
            seeds = [int(s) for s in args.seeds.split(",")]
            datasets = [s.strip() for s in args.datasets.split(",")]
            known = available_models()
            bad = [m for m in (models or []) if m not in known]
            if bad:
                print(f"Unknown/unavailable models: {bad}. Available: {known}", file=sys.stderr)
                return 2
            results = run_benchmark(models=models, datasets=datasets, seeds=seeds,
                                    base={k: v for k, v in d.items() if k != "model_params"},
                                    data_dir=args.data_dir, output_dir=args.output_dir)
            if not results:
                print("Benchmark produced no results — see the errors above.", file=sys.stderr)
                return 1
            table = make_table(results, title=f"Benchmark ({len(results)} runs)")
            print(table)
            out = write_table(results, args.table_out)
            print(f"\nTable written to {out}")

        elif args.command == "explain":
            from enginepulse.data.loader import load_dataset
            from enginepulse.data.preprocessing import prepare_dataset
            from enginepulse.explain import native_sensor_importance, permutation_sensor_importance, top_sensors
            from enginepulse.models.registry import build_model
            from enginepulse.utils import set_seed

            d = _cfg_from_args(args)
            set_seed(d.get("seed", 42))
            ds = load_dataset(args.data_dir, args.dataset)
            prep = prepare_dataset(ds, window_len=d.get("window_len", 24),
                                   max_rul=d.get("max_rul", 125), seed=d.get("seed", 42))
            model = build_model(args.model, seed=d.get("seed", 42),
                                max_rul=d.get("max_rul", 125)).fit(
                prep.train_X, prep.train_y, prep.val_X, prep.val_y)
            imp = native_sensor_importance(model, prep.feature_cols)
            if not imp:
                logger.info("No native importances; falling back to permutation importance.")
                # Use VAL engines' windows: permuting on training data would
                # measure memorisation, not generalisation sensitivity.
                imp = permutation_sensor_importance(model, prep.val_X[:500], prep.val_y[:500],
                                                    prep.feature_cols, seed=d.get("seed", 42))
            print(f"Top sensors for {args.dataset}/{args.model} (sensor code = official column):")
            for sensor, val in top_sensors(imp, args.top):
                print(f"  sensor {sensor:>3}: {val:.4f}")

        elif args.command == "serve":
            try:
                import uvicorn

                from enginepulse.api.app import create_app
            except ImportError as exc:
                print(str(exc) + "\nInstall with: pip install 'enginepulse[api]'", file=sys.stderr)
                return 2
            uvicorn.run(create_app(args.models_dir), host=args.host, port=args.port)

        elif args.command == "docqa-index":
            from enginepulse.docqa.ingest import build_index

            build_index(args.docs_dir, args.out, model_name=args.embedding_model)
            print(f"Index written to {args.out}")

        elif args.command == "ask":
            from enginepulse.docqa.qa import DocQAEngine, llm_config_from_env

            engine = DocQAEngine.from_index(args.index, llm=llm_config_from_env())
            res = engine.ask(args.question, k=args.k)
            print(res["answer"])
            if res["citations"]:
                print("\nCitations:")
                for c in res["citations"]:
                    print(f"  - {c['document']} p.{c['page']} (score {c['score']})")

        elif args.command == "docqa-eval":
            from enginepulse.docqa.eval import evaluate, load_questions
            from enginepulse.docqa.qa import DocQAEngine, llm_config_from_env

            engine = DocQAEngine.from_index(args.index, llm=llm_config_from_env())
            scores = evaluate(engine, load_questions(args.questions), k=args.k)
            print(json.dumps(scores, indent=2))

        elif args.command == "models":
            from enginepulse.models.registry import available_models

            print("\n".join(available_models()))

        else:  # pragma: no cover
            build_parser().print_help()
            return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
