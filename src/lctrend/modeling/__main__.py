"""Run the three modeling levels independently of ingest.

Level 1 (dataset): pilot, export-pilot, export-full, compact-subgraphs,
enrich-neighbors, label-llm, prepare. Level 2 (training): train-catboost,
train-hgt, explain-catboost, explain-hgt. Level 3 (labeling):
score-catboost. Merging duplicates, the first step of level 1, runs
as ``python -m lctrend.modeling.dataset.deduplication``. Commands with
``--run`` write into ``artifacts/modeling/<run>/`` when no output path is
given.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .dataset.annotations import (
    build_pilot_queue,
    compact_subgraph_file,
    export_full_history,
    export_pilot_features,
    write_pilot_queue,
)
from .dataset.labels import TARGETS, prepare_from_files
from .dataset.llm_labels import label_file
from .dataset.neighbors import enrich_file
from .storage import RunLayout, write_manifest
from .training.catboost_model import train_from_file as train_catboost
from .training.explanations import explain_catboost_file, explain_hgt_file
from .training.hgt_model import train_from_files as train_hgt


def _output(args, level, name):
    """An explicit --output, else <run>/<level>/<name>."""
    if args.output:
        return args.output
    return getattr(RunLayout.at(args.run), level) / name


def _run_arguments(command):
    command.add_argument("--output", type=Path)
    command.add_argument(
        "--run", help="Run name under artifacts/modeling (default: today)"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m lctrend.modeling")
    commands = parser.add_subparsers(dest="command", required=True)
    pilot = commands.add_parser("pilot")
    pilot.add_argument("--output", type=Path, required=True)
    pilot.add_argument("--technologies", type=int, default=100)
    pilot.add_argument("--as-known", action="store_true")
    export = commands.add_parser("export-pilot")
    export.add_argument("--review", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--subgraphs", type=Path, required=True)
    export.add_argument("--as-known", action="store_true")
    export.add_argument("--no-taxonomy", action="store_true")
    full = commands.add_parser("export-full")
    full.add_argument("--output", type=Path, required=True)
    full.add_argument("--review", type=Path, required=True)
    full.add_argument("--subgraphs", type=Path, required=True)
    full.add_argument("--as-known", action="store_true")
    full.add_argument("--no-taxonomy", action="store_true")
    compact = commands.add_parser("compact-subgraphs")
    compact.add_argument("--source", type=Path, required=True)
    compact.add_argument("--target", type=Path, required=True)
    llm_labels = commands.add_parser("label-llm")
    llm_labels.add_argument("--review", type=Path, required=True)
    llm_labels.add_argument("--reference", type=Path, required=True)
    llm_labels.add_argument("--output", type=Path, required=True)
    llm_labels.add_argument("--limit", type=int, default=10)
    llm_labels.add_argument("--all-snapshots", action="store_true")
    llm_labels.add_argument("--dry-run", action="store_true")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--dataset", type=Path, required=True)
    prepare.add_argument("--review-1", type=Path, required=True)
    prepare.add_argument("--review-2", type=Path, required=True)
    prepare.add_argument("--adjudication", type=Path)
    prepare.add_argument("--data-end", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    catboost = commands.add_parser("train-catboost")
    catboost.add_argument("--dataset", type=Path, required=True)
    catboost.add_argument("--output-dir", type=Path, required=True)
    catboost.add_argument("--target", choices=TARGETS, default="signal_36m")
    catboost.add_argument(
        "--split", choices=("temporal", "family", "cohort"), default="temporal"
    )
    catboost.add_argument("--fold", type=int, default=0)
    hgt = commands.add_parser("train-hgt")
    hgt.add_argument("--dataset", type=Path, required=True)
    hgt.add_argument("--subgraphs", type=Path, required=True)
    hgt.add_argument("--output-dir", type=Path, required=True)
    hgt.add_argument("--target", choices=TARGETS, default="signal_36m")
    hgt.add_argument(
        "--split", choices=("temporal", "family", "cohort"), default="temporal"
    )
    hgt.add_argument("--fold", type=int, default=0)
    for name in ("explain-catboost", "explain-hgt"):
        command = commands.add_parser(name)
        command.add_argument("--model-dir", type=Path, required=True)
        command.add_argument("--technology-id", required=True)
        command.add_argument("--snapshot", required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--target", choices=TARGETS, default="signal_36m")
        command.add_argument(
            "--split",
            choices=("temporal", "family", "cohort"),
            default="temporal",
        )
        command.add_argument("--fold", type=int, default=0)
        if name == "explain-catboost":
            command.add_argument("--dataset", type=Path, required=True)
        else:
            command.add_argument("--subgraphs", type=Path, required=True)
            command.add_argument(
                "--dataset",
                type=Path,
                help="History rows that fill neighbour technologies",
            )
    neighbors = commands.add_parser(
        "enrich-neighbors",
        help="Level 1: add neighbour aggregates from subgraphs to the rows",
    )
    neighbors.add_argument("--dataset", type=Path, required=True)
    neighbors.add_argument(
        "--subgraphs",
        type=Path,
        required=True,
        help="Subgraph JSONL, or a zip holding one",
    )
    _run_arguments(neighbors)
    score = commands.add_parser(
        "score-catboost",
        help="Level 3: calibrated probability for every technology",
    )
    score.add_argument("--model-dir", type=Path, required=True)
    score.add_argument("--dataset", type=Path, required=True)
    score.add_argument(
        "--snapshot", help="Score this date (default: each latest)"
    )
    score.add_argument("--target", choices=TARGETS, default="signal_36m")
    score.add_argument(
        "--split", choices=("temporal", "family", "cohort"), default="temporal"
    )
    score.add_argument("--fold", type=int, default=0)
    _run_arguments(score)
    args = parser.parse_args(argv)
    if args.command == "pilot":
        from ..cli import _graph, _temporal_data

        corpus = asyncio.run(
            _graph(lambda store: _temporal_data(store, args.as_known))
        )
        rows = build_pilot_queue(corpus, args.technologies)
        print(
            json.dumps(
                {"rows": write_pilot_queue(args.output, rows)},
                ensure_ascii=False,
            )
        )
    elif args.command == "export-pilot":
        from ..cli import _graph, _temporal_data

        corpus = asyncio.run(
            _graph(lambda store: _temporal_data(store, args.as_known))
        )
        count = export_pilot_features(
            corpus,
            args.review,
            args.output,
            args.subgraphs,
            include_novelty=not args.no_taxonomy,
        )
        print(json.dumps({"rows": count}, ensure_ascii=False))
    elif args.command == "export-full":
        from ..cli import _graph, _temporal_data

        corpus = asyncio.run(
            _graph(lambda store: _temporal_data(store, args.as_known))
        )
        report = export_full_history(
            corpus,
            args.output,
            args.review,
            args.subgraphs,
            include_novelty=not args.no_taxonomy,
        )
        print(json.dumps(report, ensure_ascii=False))
    elif args.command == "compact-subgraphs":
        count = compact_subgraph_file(args.source, args.target)
        print(json.dumps({"rows": count}, ensure_ascii=False))
    elif args.command == "label-llm":
        result = asyncio.run(
            label_file(
                args.review,
                args.output,
                args.reference,
                limit=args.limit,
                all_snapshots=args.all_snapshots,
                dry_run=args.dry_run,
            )
        )
        print(json.dumps(result, ensure_ascii=False))
    elif args.command == "prepare":
        result = prepare_from_files(
            args.dataset,
            args.review_1,
            args.review_2,
            args.output,
            args.data_end,
            args.adjudication,
        )
        print(json.dumps(result, ensure_ascii=False))
    elif args.command == "train-catboost":
        result = train_catboost(
            args.dataset,
            args.output_dir,
            target=args.target,
            strategy=args.split,
            fold=args.fold,
        )
        print(json.dumps(result["metrics"], ensure_ascii=False))
    elif args.command == "train-hgt":
        result = train_hgt(
            args.dataset,
            args.subgraphs,
            args.output_dir,
            target=args.target,
            strategy=args.split,
            fold=args.fold,
        )
        print(json.dumps(result["metrics"], ensure_ascii=False))
    elif args.command == "explain-catboost":
        result = explain_catboost_file(
            args.model_dir,
            args.dataset,
            args.technology_id,
            args.snapshot,
            args.output,
            args.target,
            args.split,
            args.fold,
        )
        print(json.dumps({"probability": result["probability"]}))
    elif args.command == "explain-hgt":
        result = explain_hgt_file(
            args.model_dir,
            args.subgraphs,
            args.technology_id,
            args.snapshot,
            args.output,
            args.target,
            args.split,
            args.fold,
            args.dataset,
        )
        print(json.dumps({"probability": result["probability"]}))
    elif args.command == "enrich-neighbors":
        output = _output(args, "dataset", "history-neighbors.csv")
        summary = enrich_file(args.dataset, args.subgraphs, output)
        write_manifest(
            output,
            "dataset.enrich-neighbors",
            [args.dataset, args.subgraphs],
            summary=summary,
        )
        print(json.dumps(summary, ensure_ascii=False))
    elif args.command == "score-catboost":
        from .labeling.scoring import score_catboost_file

        output = _output(args, "labeling", "scores.csv")
        summary = score_catboost_file(
            args.model_dir,
            args.dataset,
            output,
            target=args.target,
            strategy=args.split,
            fold=args.fold,
            snapshot=args.snapshot,
        )
        write_manifest(
            output,
            "labeling.score-catboost",
            [args.dataset],
            parameters={
                "model_dir": str(args.model_dir),
                "target": args.target,
                "split": args.split,
                "fold": args.fold,
                "snapshot": args.snapshot,
            },
            summary=summary,
        )
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
