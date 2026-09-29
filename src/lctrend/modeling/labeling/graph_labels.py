"""The final marking of technologies, written onto their graph nodes.

Two sources meet on each ``Technology`` node:

- the trained model (``labeling/scores.csv``): ``signal_probability`` of a
  weak signal at the technology's latest snapshot, ``signal_flag`` (above
  the valid-F1 threshold), ``signal_model``, ``signal_snapshot``;
- the LLM trajectory labels (``llm_labels.csv.jsonl``): ``llm_verdict``,
  ``llm_is_technology`` (False = noise), the latest year's ``llm_score``,
  ``llm_hype`` and ``llm_maturity``, ``llm_rationale``, ``llm_model`` and
  every year as JSON in ``llm_years``.

Every property starts with ``signal_`` or ``llm_``, so ``clear_labels``
removes them all and nothing else. Only existing nodes are updated; a
technology merged or absent from the graph is reported, never created.

    python -m lctrend.modeling.labeling.graph_labels \\
        --scores artifacts/modeling/R/labeling/scores.csv \\
        --llm artifacts/modeling/R/dataset/llm_labels.csv.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROPERTIES = (
    "signal_probability",
    "signal_flag",
    "signal_model",
    "signal_snapshot",
    "signal_scored_at",
    "llm_verdict",
    "llm_is_technology",
    "llm_score",
    "llm_hype",
    "llm_maturity",
    "llm_rationale",
    "llm_model",
    "llm_years",
    "llm_labelled_at",
)
_BATCH = 500


def read_scores(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_answers(path: Path) -> Dict[str, Dict[str, Any]]:
    """The latest successful LLM answer per technology."""
    answers: Dict[str, Dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                item = json.loads(line)
                if item.get("status") == "ok":
                    answers[item["technology_id"]] = item
    return answers


def graph_rows(
    scores: Sequence[Mapping[str, Any]],
    answers: Mapping[str, Mapping[str, Any]],
    now: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """One {id, props} per technology found in either source."""
    now = now or datetime.now(timezone.utc).isoformat()
    props: Dict[str, Dict[str, Any]] = {}
    for row in scores:
        if row.get("probability") in (None, ""):
            continue
        props.setdefault(row["technology_id"], {}).update(
            {
                "signal_probability": float(row["probability"]),
                "signal_flag": str(row.get("signal")) == "1",
                "signal_model": row.get("model"),
                "signal_snapshot": row.get("snapshot_date"),
                "signal_scored_at": now,
            }
        )
    for technology_id, answer in answers.items():
        years = sorted(answer.get("years") or [], key=lambda y: y["year"])
        latest = years[-1] if years else {}
        props.setdefault(technology_id, {}).update(
            {
                "llm_verdict": answer.get("verdict"),
                "llm_is_technology": bool(answer.get("is_technology")),
                "llm_score": latest.get("score"),
                "llm_hype": latest.get("hype"),
                "llm_maturity": latest.get("maturity"),
                "llm_rationale": answer.get("rationale"),
                "llm_model": answer.get("model"),
                "llm_years": json.dumps(years, ensure_ascii=False),
                "llm_labelled_at": now,
            }
        )
    return [
        {
            "id": key,
            "props": {k: v for k, v in value.items() if v is not None},
        }
        for key, value in sorted(props.items())
    ]


async def write_labels(
    store: Any, rows: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """SET the properties on existing Technology nodes, in batches."""
    from ...graph.store import _records

    written, found = 0, set()
    async with store._driver.session(database=store._database) as session:
        for start in range(0, len(rows), _BATCH):
            batch = list(rows[start : start + _BATCH])

            async def write(tx, batch=batch):
                result = await tx.run(
                    "UNWIND $rows AS row "
                    "MATCH (t:Technology {concept_id: row.id}) "
                    "WHERE coalesce(t.status, '') <> 'merged' "
                    "SET t += row.props RETURN t.concept_id AS id",
                    rows=batch,
                )
                return [record["id"] async for record in result]

            ids = await session.execute_write(write)
            found.update(ids)
            written += len(ids)
        totals = await _records(
            session,
            "MATCH (t:Technology) WHERE coalesce(t.status, '') <> 'merged' "
            "RETURN count(t) AS all, "
            "count(t.signal_probability) AS scored, "
            "count(t.llm_verdict) AS llm, "
            "sum(CASE WHEN t.llm_is_technology = false THEN 1 ELSE 0 END) "
            "AS noise",
        )
    return {
        "written": written,
        "not_in_graph": sorted({row["id"] for row in rows} - found),
        "graph": dict(totals[0]) if totals else {},
    }


async def clear_labels(store: Any) -> int:
    """Remove every signal_/llm_ property from Technology nodes."""
    removed = ", ".join(f"t.{name}" for name in PROPERTIES)
    async with store._driver.session(database=store._database) as session:

        async def clear(tx):
            result = await tx.run(
                "MATCH (t:Technology) WHERE t.signal_probability IS NOT NULL "
                f"OR t.llm_verdict IS NOT NULL REMOVE {removed} "
                "RETURN count(t) AS n"
            )
            record = await result.single()
            return record["n"]

        return await session.execute_write(clear)


def main(argv=None) -> Dict[str, Any]:
    from ...cli import _graph

    parser = argparse.ArgumentParser(
        prog="python -m lctrend.modeling.labeling.graph_labels",
        description="Write model scores and LLM labels onto Technology nodes.",
    )
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--llm", type=Path)
    parser.add_argument(
        "--clear", action="store_true", help="Remove the properties instead"
    )
    args = parser.parse_args(argv)
    if args.clear:
        result = {"cleared": asyncio.run(_graph(clear_labels))}
    else:
        rows = graph_rows(
            read_scores(args.scores) if args.scores else [],
            read_answers(args.llm) if args.llm else {},
        )
        result = asyncio.run(_graph(lambda store: write_labels(store, rows)))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
