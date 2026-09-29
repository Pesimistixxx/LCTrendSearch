"""External check: the organisers' 100 weak signals against our graph.

The list (``dataset.annotation_batch.read_list``) is 100 technologies that
methodologists judged weak signals in September 2026, described in
Russian. Each is matched to the graph by meaning: its text is embedded
with the same model as the concept vectors (the GigaChat key pool) and
compared with every active technology; the three nearest are kept with
their cosine. A match is only a candidate — ``match_cosine`` says how
close it is, and a person confirms it in the annotation workbook.

For every matched item the report adds what our model and the LLM say
about that technology today. The list has positives only, so it measures
recall («do we flag what the methodologists flag»), not precision.

    python -m lctrend.modeling.labeling.benchmark \\
        --list outputs/annotation-batch-2026-09-29/batch.xlsx \\
        --output artifacts/modeling/R/benchmark.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

TOP = 3


def item_text(item: Mapping[str, Any]) -> str:
    """The name and why it is a weak signal, as a concept's
    «label: definition» text is built."""
    why = str(item.get("why_weak") or "").strip()
    return (
        f"{item['technology']}: {why[:400]}"
        if why
        else str(item["technology"])
    )


def nearest(
    item_vectors: Sequence[Sequence[float]],
    concepts: Sequence[Mapping[str, Any]],
    top: int = TOP,
) -> List[List[Dict[str, Any]]]:
    """For every item, the ``top`` concepts by cosine, closest first."""
    matrix = np.asarray([row["vector"] for row in concepts], dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-9)
    items = np.asarray(item_vectors, dtype=np.float32)
    items /= np.linalg.norm(items, axis=1, keepdims=True).clip(min=1e-9)
    similarity = items @ matrix.T
    result = []
    for row in similarity:
        order = np.argsort(-row)[:top]
        result.append(
            [
                {
                    "concept_id": concepts[index]["concept_id"],
                    "label": concepts[index]["label"],
                    "cosine": round(float(row[index]), 4),
                }
                for index in order
            ]
        )
    return result


async def match_list(
    store: Any, items: Sequence[Mapping[str, Any]], model: str
) -> List[Dict[str, Any]]:
    """Graph candidates for every list item (reads Neo4j, embeds texts)."""
    from ...extraction.processing import _semantic_deduplicator
    from ..dataset.deduplication import read_candidates

    concepts = await read_candidates(store, model, ["Technology"])
    semantic = _semantic_deduplicator()
    vectors = await asyncio.to_thread(
        semantic.embed, [item_text(item) for item in items]
    )
    if vectors is None:
        raise RuntimeError(
            f"Embedding endpoint unavailable ({semantic.failure})"
        )
    rows = []
    for item, candidates in zip(items, nearest(vectors, concepts)):
        best = candidates[0]
        rows.append(
            {
                "source_no": item.get("source_no"),
                "technology": item.get("technology"),
                "domain": item.get("domain"),
                "stage": item.get("stage"),
                "graph_technology_id": best["concept_id"],
                "graph_technology": best["label"],
                "match_cosine": best["cosine"],
                "alternatives": json.dumps(candidates[1:], ensure_ascii=False),
            }
        )
    return rows


def attach_scores(
    matches: Sequence[Dict[str, Any]],
    scores: Sequence[Mapping[str, str]],
    answers: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Add our model's and the LLM's view of each matched technology."""
    by_id = {row["technology_id"]: row for row in scores}
    result = []
    for row in matches:
        score = by_id.get(row["graph_technology_id"], {})
        answer = answers.get(row["graph_technology_id"], {})
        years = sorted(
            answer.get("years") or [], key=lambda item: item["year"]
        )
        result.append(
            {
                **row,
                "probability": score.get("probability", ""),
                "signal": score.get("signal", ""),
                "in_scores": bool(score),
                "llm_verdict": answer.get("verdict", ""),
                "llm_is_technology": answer.get("is_technology", ""),
                "llm_score_latest": years[-1]["score"] if years else "",
            }
        )
    return result


def summary(
    rows: Sequence[Mapping[str, Any]],
    min_cosine: float,
    weak_max: float = 0.3,
) -> Dict[str, Any]:
    """Recall on the list among confident matches."""
    matched = [row for row in rows if float(row["match_cosine"]) >= min_cosine]
    scored = [row for row in matched if row["in_scores"]]
    flagged = [row for row in scored if str(row["signal"]) == "1"]
    llm_weak = [
        row
        for row in matched
        if row["llm_score_latest"] not in ("", None)
        and float(row["llm_score_latest"]) <= weak_max
    ]
    return {
        "items": len(rows),
        "matched_at_cosine": min_cosine,
        "matched": len(matched),
        "scored_by_model": len(scored),
        "model_flags_signal": len(flagged),
        "model_recall": round(len(flagged) / len(scored), 3)
        if scored
        else None,
        "llm_weak_now": len(llm_weak),
        "llm_recall": round(len(llm_weak) / len(matched), 3)
        if matched
        else None,
    }


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> Dict[str, Any]:
    from ...cli import _graph
    from ...core.config import load_environment
    from ...extraction.processing import _semantic_deduplicator
    from ..config import load_config
    from ..dataset.annotation_batch import read_list
    from ..storage import RunLayout
    from .graph_labels import read_answers, read_scores

    parser = argparse.ArgumentParser(
        prog="python -m lctrend.modeling.labeling.benchmark",
        description="Match the organisers' 100 signals to the graph.",
    )
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-cosine", type=float, default=0.8)
    args = parser.parse_args(argv)
    load_environment()
    bundle = os.getenv("GIGACHAT_CA_BUNDLE_FILE", "")
    if not bundle or not Path(bundle).exists():
        os.environ["GIGACHAT_CA_BUNDLE_FILE"] = str(
            Path(__file__).resolve().parents[4]
            / "certs/gigachat-ca-bundle.pem"
        )
    config = load_config()
    layout = RunLayout.at(config["run"])
    items = read_list(args.list)
    model = _semantic_deduplicator().embedding_model_name
    matches = asyncio.run(
        _graph(lambda store: match_list(store, items, model))
    )
    scores_path = layout.labeling / "scores.csv"
    rows = attach_scores(
        matches,
        read_scores(scores_path) if scores_path.exists() else [],
        read_answers(layout.dataset / "llm_labels.csv.jsonl"),
    )
    output = args.output or layout.root / "benchmark.csv"
    write_rows(output, rows)
    result = summary(rows, args.min_cosine, config["labels"]["weak_max"])
    result["output"] = str(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
