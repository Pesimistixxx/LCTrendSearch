"""Score every technology of the graph and write the result back to it.

The final marking straight from Neo4j, without the training exports:

1. merge duplicates (``dataset.deduplication``, by default with the
   parameters the presented run was built with), so a technology's
   history is not spread over several nodes;
2. read the graph as it was on one date (default: its latest observation)
   and compute each technology's feature row and two-hop subgraph with the
   same functions the history export uses (``export_snapshot``);
3. add the neighbour aggregates (``dataset.builder.enrich_and_relate``);
4. apply the CatBoost, HGT and stacked models of a trained run exactly as
   ``training.pipeline`` saved them: features, scaler, temperature,
   intercept, F1 threshold and winner come from ``models/report.json``;
   nothing is refitted (``score_rows``);
5. write ``scores.csv`` and SET ``signal_*`` on the Technology nodes
   (``graph_labels.write_labels``); scores of technologies not scored this
   time are removed, so the search never shows a stale probability.

Intermediate files go to ``<run>/dataset/graph-<date>/`` (large, not in
git), the scores to ``<run>/labeling/graph-<date>/``.

    python scripts/score_graph.py --run 2026-09-29-dedup
    python scripts/score_graph.py --dry-run          # plan merges, no writes
    python scripts/score_graph.py --skip-dedup --no-write
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..dataset.builder import enrich_and_relate, read_history, write_rows
from ..dataset.neighbors import iter_samples
from ..storage import RunLayout, write_manifest
from .graph_labels import PROPERTIES, graph_rows, read_answers, write_labels

logger = logging.getLogger(__name__)

MODELS = ("catboost", "hgt", "stacked")
SIGNAL_PROPERTIES = tuple(name for name in PROPERTIES if name[:7] == "signal_")
# The presented run was deduplicated at 0.90 with an LLM judge below 0.95
# (artifacts/modeling/dedup-090.json); scoring merges the same way.
DEDUP_THRESHOLD = 0.90
DEDUP_JUDGE_BELOW = 0.95


def _key(row):
    return row["technology_id"], row["snapshot_date"]


# -- 2. features and subgraphs at one date ----------------------------


def export_snapshot(
    corpus: Any,
    when: date,
    history_csv: Path,
    subgraphs_jsonl: Path,
    include_novelty: bool = True,
) -> Dict[str, Any]:
    """Feature rows and subgraphs of every documented technology at ``when``.

    The single-date part of ``annotations.export_full_history``: the same
    rows, the same neighbourhood sampling, the same files, so the models
    see exactly what they were trained on.
    """
    from ...core.config import load_catalog
    from ...graph.subgraphs import sample_neighborhood, write_subgraph_rows
    from ...graph.training import (
        IDENTITY_FIELDS,
        build_snapshot_rows,
        dataset_feature_fields,
    )
    from ..dataset.annotations import without_unused_embeddings

    feature_names = dataset_feature_fields()
    fields = list(dict.fromkeys((*IDENTITY_FIELDS, *feature_names)))
    config = load_catalog("dataset")["neighborhood"]
    view = corpus.view(when)
    rows = build_snapshot_rows(
        corpus, when, min_documents=1, include_novelty=include_novelty
    )
    history_csv = Path(history_csv)
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    with history_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def samples():
        for row in rows:
            yield without_unused_embeddings(
                sample_neighborhood(
                    view,
                    row["technology_id"],
                    config=config,
                    features={name: row.get(name) for name in feature_names},
                )
            )

    graphs = write_subgraph_rows(subgraphs_jsonl, samples())
    return {
        "snapshot": when.isoformat(),
        "technologies": len(corpus.labels),
        "technologies_with_documents": len(rows),
        "subgraphs": graphs,
    }


def scoring_inputs(history_csv: Path, subgraphs_jsonl: Path):
    """Rows as the training read them (CSV strings, neighbour aggregates)
    and the subgraph of each row by (technology_id, snapshot_date)."""
    rows = read_history(history_csv)
    samples = {
        (str(sample["technology_id"]), str(sample["snapshot"])): sample
        for sample in iter_samples(subgraphs_jsonl)
    }
    enrich_and_relate(rows, samples.values())
    return rows, samples


# -- 4. the trained models --------------------------------------------


def load_models(model_dir: Path) -> Dict[str, Any]:
    """The models of a run and the report that says how to apply them."""
    from catboost import CatBoostClassifier

    model_dir = Path(model_dir)
    report = json.loads(
        (model_dir / "report.json").read_text(encoding="utf-8")
    )
    loaded: Dict[str, Any] = {"report": report, "dir": str(model_dir)}
    for name in ("catboost", "stacked"):
        if name in report["models"]:
            model = CatBoostClassifier()
            model.load_model(str(model_dir / f"{name}.cbm"))
            if list(model.feature_names_) != report["models"][name]["features"]:
                raise ValueError(
                    f"{name}.cbm and report.json list different features: "
                    "the model files and the report come from different runs"
                )
            loaded[name] = model
    if "hgt" in report["models"]:
        loaded["hgt"] = _hgt(
            model_dir / "hgt.pt", len(report["models"]["hgt"]["features"])
        )
    if "stacked" in report["models"]:
        folds = int(report["models"]["stacked"]["folds"])
        paths = [model_dir / f"hgt_fold{fold}.pt" for fold in range(folds)]
        missing = [path.name for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Stacked model needs {folds} fold models; missing {missing}"
            )
        loaded["folds"] = [
            _hgt(path, len(report["models"]["hgt"]["features"]))
            for path in paths
        ]
    return loaded


def _hgt(path: Path, features: int):
    import torch

    from ..training.hgt_model import build_model

    model = build_model(features)
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


def _calibrated(logits, report):
    from ..training.catboost_model import calibrated

    return calibrated(
        logits,
        report["temperature_valid_only"],
        # Reports written before the intercept existed calibrated with the
        # temperature alone.
        report.get("intercept_valid_only", 0.0),
    )


def score_rows(
    models: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    samples: Mapping[tuple, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Calibrated probabilities of every model and the winner's verdict.

    Mirrors ``training.pipeline``: HGT sees the root and its neighbour
    technologies through the training scaler; the stacked CatBoost gets
    the mean fold-HGT logit as ``hgt_oof_probability``. A row without a
    subgraph keeps its CatBoost probability; when the winner is a graph
    model the verdict then falls back to CatBoost, and ``model`` says so.
    """
    from ..training.catboost_model import _sigmoid, catboost_logits
    from ..training.hgt_model import hgt_data, predict_logits
    from ..training.pipeline import STACK_FEATURE

    report = models["report"]["models"]
    rows = list(rows)
    features = report["catboost"]["features"]
    absent = [name for name in features if rows and name not in rows[0]]
    if absent:
        raise ValueError(
            f"Rows lack model features {absent[:5]}: the export and the "
            "models are of different versions"
        )
    probability: Dict[str, Dict[tuple, float]] = {name: {} for name in MODELS}
    logits = catboost_logits(models["catboost"], rows, features)
    for row, value in zip(rows, _calibrated(logits, report["catboost"])):
        probability["catboost"][_key(row)] = float(value)

    graphed = [row for row in rows if _key(row) in samples]
    if "hgt" in models and graphed:
        names = report["hgt"]["features"]
        scaler = report["hgt"]["scaler_train_only"]
        neighbours = {_key(row): row for row in rows}
        graphs = [
            hgt_data(samples[_key(row)], names, scaler, neighbours)
            for row in graphed
        ]
        values = _calibrated(
            predict_logits(models["hgt"], graphs), report["hgt"]
        )
        probability["hgt"] = {
            _key(row): float(value) for row, value in zip(graphed, values)
        }
        if "stacked" in models:
            folds = models["folds"]
            mean = sum(predict_logits(model, graphs) for model in folds)
            stacked_rows = [
                {**row, STACK_FEATURE: float(_sigmoid(value / len(folds)))}
                for row, value in zip(graphed, mean)
            ]
            names = report["stacked"]["features"]
            values = _calibrated(
                catboost_logits(models["stacked"], stacked_rows, names),
                report["stacked"],
            )
            probability["stacked"] = {
                _key(row): float(value)
                for row, value in zip(graphed, values)
            }

    winner = models["report"]["winner"]["model"]
    records = []
    for row in rows:
        key = _key(row)
        record = {
            "technology_id": row["technology_id"],
            "technology": row.get("technology"),
            "snapshot_date": row["snapshot_date"],
        }
        for name in MODELS:
            value = probability[name].get(key)
            record[f"p_{name}"] = "" if value is None else round(value, 6)
        chosen = winner if key in probability[winner] else "catboost"
        value = probability[chosen][key]
        record.update(
            model=chosen,
            probability=round(value, 6),
            signal=int(value >= report[chosen]["threshold_valid_f1"]),
            above_0_75=int(value >= 0.75),
        )
        records.append(record)
    records.sort(key=lambda item: -item["probability"])
    return records


# -- 5. the graph -----------------------------------------------------


async def clear_stale_scores(store: Any, keep: Sequence[str]) -> int:
    """Remove signal_* from Technology nodes outside ``keep``: a node that
    lost its documents or was merged must not keep an old probability."""
    removed = ", ".join(f"t.{name}" for name in SIGNAL_PROPERTIES)
    async with store._driver.session(database=store._database) as session:

        async def clear(tx):
            result = await tx.run(
                "MATCH (t:Technology) WHERE t.signal_probability IS NOT NULL "
                "AND NOT t.concept_id IN $keep "
                f"REMOVE {removed} RETURN count(t) AS n",
                keep=list(keep),
            )
            record = await result.single()
            return record["n"]

        return await session.execute_write(clear)


async def write_scores(
    store: Any,
    records: Sequence[Mapping[str, Any]],
    answers: Optional[Mapping[str, Mapping[str, Any]]] = None,
    keep_stale: bool = False,
) -> Dict[str, Any]:
    result = await write_labels(store, graph_rows(records, answers or {}))
    if not keep_stale:
        result["stale_scores_removed"] = await clear_stale_scores(
            store, [record["technology_id"] for record in records]
        )
    return result


# -- the whole pass ---------------------------------------------------


def _deduplicate(args, log: Path) -> Dict[str, Any]:
    from ..dataset.deduplication import main as deduplicate

    argv = [
        "--threshold",
        str(args.dedup_threshold),
        "--judge-below",
        str(args.judge_below),
        "--log",
        str(log),
    ]
    if args.llm_judge:
        argv.append("--llm-judge")
    if args.dry_run:
        argv.append("--dry-run")
    return deduplicate(argv)


@contextmanager
def only_key(name: str):
    """Point GIGACHAT_KEYS_FILE at a pool of the one key ``name``.

    The labelling key is ``enabled: false`` in the shared pool so that
    ingestion workers never take it; every LLM call of this pass (missing
    vectors, the duplicate judge) goes through it alone. The temporary
    pool file holds a secret and is removed on exit.
    """
    from ..dataset.llm_outcomes import labelling_key

    source = os.getenv("GIGACHAT_KEYS_FILE", "")
    if not source:
        raise ValueError("GIGACHAT_KEYS_FILE is not set: no key to use")
    entry = labelling_key(source, name)
    handle, path = tempfile.mkstemp(prefix="lctrend-key-", suffix=".json")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump({"keys": [{**entry, "enabled": True}]}, stream)
        os.environ["GIGACHAT_KEYS_FILE"] = path
        yield entry.get("name")
    finally:
        os.environ["GIGACHAT_KEYS_FILE"] = source
        Path(path).unlink(missing_ok=True)


def _summary(records):
    by_model: Dict[str, int] = {}
    for record in records:
        by_model[record["model"]] = by_model.get(record["model"], 0) + 1
    return {
        "technologies": len(records),
        "signal": sum(record["signal"] for record in records),
        "above_0_75": sum(record["above_0_75"] for record in records),
        "by_model": by_model,
        "top": [
            [record["technology"], record["probability"]]
            for record in records[:10]
        ],
    }


def main(argv=None) -> Dict[str, Any]:
    from ...session import temporal_corpus, with_graph
    from ...core.config import load_catalog, load_environment
    from ..dataset.llm_outcomes import LABELING_KEY

    run = load_catalog("training").get("run")
    parser = argparse.ArgumentParser(
        prog="python scripts/score_graph.py",
        description=(
            "Deduplicate the graph, score every technology with a trained "
            "run and write the probabilities onto Technology nodes."
        ),
    )
    parser.add_argument("--run", default=run, help=f"Default: {run}")
    parser.add_argument("--model-dir", type=Path, help="Default: <run>/models")
    parser.add_argument(
        "--snapshot", help="Score as of YYYY-MM-DD (default: latest data)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan merges only and write nothing to the graph",
    )
    parser.add_argument("--skip-dedup", action="store_true")
    parser.add_argument(
        "--dedup-threshold", type=float, default=DEDUP_THRESHOLD
    )
    parser.add_argument(
        "--judge-below", type=float, default=DEDUP_JUDGE_BELOW
    )
    parser.add_argument(
        "--no-llm-judge",
        dest="llm_judge",
        action="store_false",
        help="Merge by names and vectors only, without the LLM judge",
    )
    parser.add_argument(
        "--key",
        default=LABELING_KEY,
        help=(
            "The only GigaChat key of GIGACHAT_KEYS_FILE to use, by name, "
            f"even if disabled in the pool (default: {LABELING_KEY})"
        ),
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Merge duplicates but only write scores.csv",
    )
    parser.add_argument(
        "--keep-stale",
        action="store_true",
        help="Keep old signal_* of technologies not scored now",
    )
    parser.add_argument(
        "--llm",
        type=Path,
        help="LLM labels JSONL to write too (llm_* properties)",
    )
    parser.add_argument("--as-known", action="store_true")
    parser.add_argument("--no-novelty", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    load_environment()

    layout = RunLayout.at(args.run)
    stamp = f"graph-{date.today().isoformat()}"
    work, output = layout.dataset / stamp, layout.labeling / stamp
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Any] = {"run": args.run, "output": str(output)}
    # Models first: a broken run must fail before the graph is touched.
    models = load_models(args.model_dir or layout.models)
    result["winner"] = models["report"]["winner"]["model"]

    if not args.skip_dedup:
        logger.info("1/5 Merging duplicates (LLM key %s only)", args.key)
        with only_key(args.key):
            result["dedup"] = _deduplicate(args, work / "dedup.json")

    logger.info("2/5 Reading the graph and computing features")
    corpus = asyncio.run(
        with_graph(lambda store: temporal_corpus(store, args.as_known))
    )
    if corpus.latest_date is None:
        raise ValueError("The graph has no dated observations")
    when = (
        date.fromisoformat(args.snapshot)
        if args.snapshot
        else corpus.latest_date
    )
    history, subgraphs = work / "history.csv", work / "subgraphs.jsonl"
    result["export"] = export_snapshot(
        corpus, when, history, subgraphs, not args.no_novelty
    )
    del corpus

    logger.info("3/5 Neighbour aggregates")
    rows, samples = scoring_inputs(history, subgraphs)

    logger.info("4/5 Scoring %d technologies", len(rows))
    records = score_rows(models, rows, samples)
    scores = output / "scores.csv"
    write_rows(scores, records)
    result["scores"] = _summary(records)
    write_manifest(
        scores,
        "labeling.score-graph",
        [history, subgraphs, Path(models["dir"]) / "report.json"],
        parameters={
            "run": args.run,
            "model_dir": models["dir"],
            "snapshot": when.isoformat(),
            "dedup": None if args.skip_dedup else str(work / "dedup.json"),
            "as_known": args.as_known,
            "novelty": not args.no_novelty,
        },
        summary=result["scores"],
    )

    if args.dry_run or args.no_write:
        logger.info("5/5 Skipped: nothing written to the graph")
    else:
        logger.info("5/5 Writing signal_* onto Technology nodes")
        answers = read_answers(args.llm) if args.llm else {}
        result["graph"] = asyncio.run(
            with_graph(
                lambda store: write_scores(
                    store, records, answers, args.keep_stale
                )
            )
        )
        (output / "graph.json").write_text(
            json.dumps(result["graph"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
