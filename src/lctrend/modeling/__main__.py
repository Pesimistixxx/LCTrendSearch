"""Run the expert-review and model-training stages independently of ingest."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .annotations import (
    build_pilot_queue,
    compact_subgraph_file,
    export_full_history,
    export_pilot_features,
    write_pilot_queue,
)
from .catboost_model import train_from_file as train_catboost
from .dataset import TARGETS, prepare_from_files
from .explanations import explain_catboost_file, explain_hgt_file
from .hgt_model import train_from_files as train_hgt
from .llm_labels import label_file


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
        )
        print(json.dumps({"probability": result["probability"]}))


if __name__ == "__main__":
    main()
