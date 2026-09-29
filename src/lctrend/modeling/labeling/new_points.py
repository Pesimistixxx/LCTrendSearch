"""Score the technologies that appeared after training, in the graph.

Ingestion keeps adding technologies; the trained models do not need a new
export to score them. One run:

1. embeds concepts that have no vector yet (GigaChat, the key pool);
2. merges near-duplicates first (``dataset.deduplication``: cosine at or
   above 0.95, versions held), so a new name for a known technology joins
   it instead of getting a score of its own;
3. computes the features of every technology at the latest date (the same
   ``graph.training.build_snapshot_rows`` as the export), the subgraph of
   each new one (``graph.subgraphs.sample_neighborhood``) and its
   neighbour aggregates;
4. applies the winning model of the run's ``report.json``: the stack
   (mean of the HGT fold models → CatBoost) or CatBoost, with the same
   calibration and threshold as in training;
5. with ``--write`` puts ``signal_*`` on the Technology nodes
   (``labeling.graph_labels``).

    python -m lctrend.modeling.labeling.new_points          # plan only
    python -m lctrend.modeling.labeling.new_points --write  # and write

New technologies get LLM trajectory labels from
``python -m lctrend.modeling.dataset.llm_outcomes --from-graph``, which
takes only those it has not labelled yet.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

STACK_FEATURE = "hgt_oof_probability"


async def unscored_ids(store: Any) -> List[str]:
    """Active technologies without a model probability on the node."""
    from ...graph.store import _records

    async with store._driver.session(database=store._database) as session:
        records = await _records(
            session,
            "MATCH (t:Technology) WHERE coalesce(t.status, '') <> 'merged' "
            "AND t.signal_probability IS NULL RETURN t.concept_id AS id",
        )
    return [record["id"] for record in records]


def feature_rows(corpus: Any, ids: Optional[Sequence[str]] = None):
    """Rows of every technology at the latest date, the subgraphs of the
    requested ones, and those rows with neighbour aggregates added."""
    from ...core.config import load_catalog
    from ...graph.subgraphs import sample_neighborhood
    from ...graph.training import build_snapshot_rows
    from ..dataset.neighbors import neighbor_features

    when = corpus.latest_date
    rows = build_snapshot_rows(corpus, when, min_documents=1)
    snapshot = when.isoformat()
    for row in rows:
        row["snapshot_date"] = snapshot
    index = {(row["technology_id"], snapshot): row for row in rows}
    wanted = (
        set(ids) if ids is not None else set(r["technology_id"] for r in rows)
    )
    view = corpus.view(when)
    config = load_catalog("dataset")["neighborhood"]
    chosen, samples = [], []
    for row in rows:
        if row["technology_id"] not in wanted:
            continue
        sample = sample_neighborhood(
            view,
            row["technology_id"],
            config=config,
            features={
                key: value
                for key, value in row.items()
                if isinstance(value, (int, float, bool)) or value is None
            },
        )
        scored = {**row, **neighbor_features(sample, index)}
        chosen.append(scored)
        samples.append(sample)
    return chosen, samples, index


def _catboost(path: Path):
    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def hgt_probability(report, models_dir: Path, samples, technology_rows):
    """Mean logit of the HGT fold models, as the stack saw on valid."""
    import torch

    from ..training.catboost_model import _sigmoid
    from ..training.hgt_model import build_model, hgt_data, predict_logits

    settings = report["models"]["hgt"]
    names, scaler = settings["features"], settings["scaler_train_only"]
    graphs = [
        hgt_data(sample, names, scaler, technology_rows) for sample in samples
    ]
    folds = sorted(Path(models_dir).glob("hgt_fold*.pt"))
    if not folds:
        raise FileNotFoundError("No hgt_fold*.pt models for the stack")
    total = np.zeros(len(graphs))
    for path in folds:
        model = build_model(len(names))
        model.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )
        total += predict_logits(model, graphs)
    return _sigmoid(total / len(folds))


def score(
    rows: List[Dict[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    technology_rows: Mapping[tuple, Mapping[str, Any]],
    models_dir: Path,
) -> List[Dict[str, Any]]:
    """Calibrated probability and signal flag of the run's winner."""
    from ..training.catboost_model import calibrated, catboost_logits

    report = json.loads((Path(models_dir) / "report.json").read_text("utf-8"))
    winner = report["winner"]["model"]
    name = winner if winner in ("catboost", "stacked") else "catboost"
    settings = report["models"][name]
    if name == "stacked":
        probability = hgt_probability(
            report, models_dir, samples, technology_rows
        )
        rows = [
            {**row, STACK_FEATURE: float(value)}
            for row, value in zip(rows, probability)
        ]
    model = _catboost(Path(models_dir) / f"{name}.cbm")
    logits = catboost_logits(model, rows, settings["features"])
    probabilities = calibrated(
        logits,
        settings["temperature_valid_only"],
        settings.get("intercept_valid_only", 0.0),
    )
    threshold = settings["threshold_valid_f1"]
    return [
        {
            "technology_id": row["technology_id"],
            "technology": row.get("technology"),
            "snapshot_date": row["snapshot_date"],
            "probability": round(float(value), 6),
            "signal": int(value >= threshold),
            "model": name,
        }
        for row, value in zip(rows, probabilities)
    ]


async def run(
    store: Any,
    config: Mapping[str, Any],
    write: bool = False,
    rescore_all: bool = False,
    dedup: bool = True,
) -> Dict[str, Any]:
    from ...session import embed_concepts, temporal_corpus
    from ..dataset.deduplication import ConceptDeduplicator
    from ..storage import RunLayout
    from .graph_labels import graph_rows, write_labels

    layout = RunLayout.at(config["run"])
    embedded = await embed_concepts(store)
    merged = None
    if dedup:
        family = ConceptDeduplicator(["Technology", "Method", "Material"])
        plan = await family.run(store, embedded["model"], apply=write)
        merged = {
            "planned": plan["merges"],
            "merged": len(plan["merged"]),
            "held": len(plan["held"]),
        }
    ids = None if rescore_all else await unscored_ids(store)
    if ids is not None and not ids:
        return {
            "embedded": embedded["embedded"],
            "duplicates": merged,
            "scored": 0,
            "written": 0,
        }
    corpus = await temporal_corpus(store)
    rows, samples, index = await asyncio.to_thread(feature_rows, corpus, ids)
    scores = await asyncio.to_thread(
        score, rows, samples, index, layout.models
    )
    output = layout.labeling / "new_points.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        fields = list(scores[0]) if scores else ["technology_id"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scores)
    written = 0
    if write and scores:
        result = await write_labels(store, graph_rows(scores, {}))
        written = result["written"]
    return {
        "embedded": embedded["embedded"],
        "duplicates": merged,
        "scored": len(scores),
        "signals": sum(row["signal"] for row in scores),
        "written": written,
        "output": str(output),
    }


def main(argv=None) -> Dict[str, Any]:
    from ...core.config import load_environment
    from ...session import with_graph
    from ..config import load_config

    parser = argparse.ArgumentParser(
        prog="python -m lctrend.modeling.labeling.new_points",
        description="Merge duplicates, score and mark new technologies.",
    )
    parser.add_argument("--write", action="store_true", help="Merge and write")
    parser.add_argument(
        "--all", action="store_true", help="Rescore every technology"
    )
    parser.add_argument(
        "--no-dedup", action="store_true", help="Skip the duplicate check"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    load_environment()
    bundle = os.getenv("GIGACHAT_CA_BUNDLE_FILE", "")
    if not bundle or not Path(bundle).exists():
        os.environ["GIGACHAT_CA_BUNDLE_FILE"] = str(
            Path(__file__).resolve().parents[4]
            / "certs/gigachat-ca-bundle.pem"
        )
    config = load_config()
    result = asyncio.run(
        with_graph(
            lambda store: run(
                store, config, args.write, args.all, not args.no_dedup
            )
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
