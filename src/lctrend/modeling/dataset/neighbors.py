"""Point-in-time aggregates of each technology's sampled neighbourhood.

A subgraph sample keeps neighbour technologies as bare nodes: the export
gives them only a document count. Their own features at the same date are
rows of the history CSV. The aggregates are therefore a join: neighbour
ids from the sample, their rows at the sample's snapshot. Both are cut at
that snapshot, so nothing from the future enters.

Two neighbour groups are kept apart because they mean different things:
``related`` technologies share a typed edge with the root (parent, child),
``comentioned`` ones appear in a document that mentions the root. A
technology with no neighbours in a group gets empty means, not zeros: no
neighbour is not the same as neighbours without growth.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Tuple

# Neighbour features chosen by meaning: activity relative to its
# contemporaries, growth, freshness of authors, novelty, age and fading.
SOURCE_FEATURES = (
    "documents_last_year_snapshot_pct",
    "mention_growth_12m",
    "burst_score",
    "new_author_rate",
    "semantic_novelty",
    "technology_age_days",
    "decline_rate",
)
GROUPS = ("related", "comentioned")
ORGANIZATION_TYPES = ("Company", "University", "Organization")
TECHNOLOGY_PREFIX = "Technology:"

NEIGHBOR_FEATURES = (
    *(f"nb_{group}_count" for group in GROUPS),
    *(
        f"nb_{group}_{name}_{stat}"
        for group in GROUPS
        for name in SOURCE_FEATURES
        for stat in ("mean", "max")
    ),
    "nb_growing_share",
    "nb_organization_count",
    "nb_company_share",
)

Key = Tuple[str, str]


def _number(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return float(value)
    text = str(value).strip().lower()
    if text in ("true", "false"):
        return float(text == "true")
    try:
        return float(text)
    except ValueError:
        return None


def _technology_id(node_id: str) -> str:
    return node_id[len(TECHNOLOGY_PREFIX) :]


def neighbor_groups(sample: Mapping[str, Any]) -> Dict[str, set]:
    """Technology ids of the root's related and co-mentioned neighbours."""
    root = sample["root_id"]
    types = {node["id"]: node["type"] for node in sample["nodes"]}
    related, root_documents = set(), set()
    mentions: Dict[str, set] = {}
    for edge in sample["edges"]:
        source, target = edge["source"], edge["target"]
        if edge["type"] == "MENTIONED_IN":
            if source == root:
                root_documents.add(target)
            elif types.get(source) == "Technology":
                mentions.setdefault(target, set()).add(source)
        elif (
            root in (source, target)
            and types.get(source) == types.get(target) == "Technology"
        ):
            related.add(target if source == root else source)
    comentioned = {
        technology
        for document in root_documents
        for technology in mentions.get(document, ())
    }
    related.discard(root)
    comentioned.discard(root)
    return {
        "related": {_technology_id(item) for item in related},
        "comentioned": {_technology_id(item) for item in comentioned},
    }


def neighbor_features(
    sample: Mapping[str, Any], rows: Mapping[Key, Mapping[str, Any]]
) -> Dict[str, Optional[float]]:
    """One row of NEIGHBOR_FEATURES for a sample; rows are keyed by
    (technology_id, snapshot_date)."""
    snapshot = sample["snapshot"]
    groups = neighbor_groups(sample)
    result: Dict[str, Optional[float]] = {}
    for group in GROUPS:
        found = [
            rows[(technology, snapshot)]
            for technology in sorted(groups[group])
            if (technology, snapshot) in rows
        ]
        result[f"nb_{group}_count"] = float(len(found))
        for name in SOURCE_FEATURES:
            values = [
                value
                for value in (_number(row.get(name)) for row in found)
                if value is not None
            ]
            result[f"nb_{group}_{name}_mean"] = (
                sum(values) / len(values) if values else None
            )
            result[f"nb_{group}_{name}_max"] = max(values) if values else None
    growth = [
        value
        for technology in sorted(groups["related"] | groups["comentioned"])
        if (technology, snapshot) in rows
        for value in [
            _number(rows[(technology, snapshot)].get("mention_growth_12m"))
        ]
        if value is not None
    ]
    result["nb_growing_share"] = (
        sum(value > 0 for value in growth) / len(growth) if growth else None
    )
    organizations = [
        node["type"]
        for node in sample["nodes"]
        if node["type"] in ORGANIZATION_TYPES
    ]
    result["nb_organization_count"] = float(len(organizations))
    result["nb_company_share"] = (
        organizations.count("Company") / len(organizations)
        if organizations
        else None
    )
    return result


def iter_samples(path: Path) -> Iterator[Dict[str, Any]]:
    """Samples of a JSONL file, or of the JSONL inside a zip archive."""
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [
                name for name in archive.namelist() if name.endswith(".jsonl")
            ]
            if len(members) != 1:
                raise ValueError(
                    f"Expected one .jsonl in {path}, found {members}"
                )
            with archive.open(members[0]) as raw:
                for line in io.TextIOWrapper(raw, encoding="utf-8"):
                    if line.strip():
                        yield json.loads(line)
        return
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _format(value: Optional[float]) -> str:
    return "" if value is None else repr(float(value))


def enrich_rows(
    rows: Iterable[Dict[str, Any]], samples: Iterable[Mapping[str, Any]]
) -> Dict[str, int]:
    """Add NEIGHBOR_FEATURES to the rows in place; returns counts.

    Rows without a sample keep empty neighbour cells.
    """
    rows = list(rows)
    index = {(row["technology_id"], row["snapshot_date"]): row for row in rows}
    enriched = 0
    for sample in samples:
        key = (str(sample["technology_id"]), str(sample["snapshot"]))
        row = index.get(key)
        if row is None:
            continue
        values = neighbor_features(sample, index)
        row.update({name: _format(values[name]) for name in NEIGHBOR_FEATURES})
        enriched += 1
    for row in rows:
        for name in NEIGHBOR_FEATURES:
            row.setdefault(name, "")
    return {"rows": len(rows), "enriched": enriched}


def enrich_file(dataset: Path, subgraphs: Path, output: Path) -> Dict:
    csv.field_size_limit(1 << 30)
    with Path(dataset).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    summary = enrich_rows(rows, iter_samples(subgraphs))
    filled = {
        group: sum(
            1 for row in rows if float(row[f"nb_{group}_count"] or 0) > 0
        )
        for group in GROUPS
    }
    summary["rows_with_neighbors"] = filled
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(dict.fromkeys([*fields, *NEIGHBOR_FEATURES])),
        )
        writer.writeheader()
        writer.writerows(rows)
    return summary
