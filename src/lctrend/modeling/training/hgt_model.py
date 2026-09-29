"""Heterogeneous Graph Transformer for technology snapshot subgraphs.

Only four predictive node roles are kept. Countries, domains, authors and
source/evidence records remain in the auditable JSONL; their aggregate
point-in-time features enter the technology/document vectors instead of
becoming high-degree message-passing hubs.
"""

from __future__ import annotations

import copy
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..dataset.labels import (
    FEATURES,
    TARGETS,
    _read_csv,
    require_trainable,
    split_for,
)
from .catboost_model import _metrics, _sigmoid, fit_temperature, model_stem

NODE_TYPES = ["Technology", "Document", "Organization", "Task"]
RELATIONS = [
    ("Technology", "mentions_recent", "Document"),
    ("Document", "recently_mentions", "Technology"),
    ("Technology", "mentions_middle", "Document"),
    ("Document", "middle_mentions", "Technology"),
    ("Technology", "mentions_old", "Document"),
    ("Document", "old_mentions", "Technology"),
    ("Document", "associated_with", "Organization"),
    ("Organization", "associated_document", "Document"),
    ("Technology", "developed_by", "Organization"),
    ("Organization", "develops", "Technology"),
    ("Technology", "used_by", "Organization"),
    ("Organization", "uses", "Technology"),
    ("Technology", "funded_by", "Organization"),
    ("Organization", "funds", "Technology"),
    ("Technology", "solves", "Task"),
    ("Task", "solved_by", "Technology"),
    ("Technology", "parent_of", "Technology"),
    ("Technology", "child_of", "Technology"),
]
METADATA = (NODE_TYPES, RELATIONS)
DOCUMENT_FEATURES = (
    "reliability_tier",
    "fulltext_available",
    "age_years",
    "scholarly",
    "code",
    "package_registry",
    "patent",
    "funding",
    "labor_market",
)
ORG_FEATURES = ("company", "university", "organization")


def fit_scaler(rows, names=FEATURES):
    """Fit medians and robust scales on training roots only."""
    result = {}
    for name in names:
        values = []
        for row in rows:
            raw = row.get(name)
            if raw in (None, ""):
                continue
            try:
                values.append(
                    float(str(raw).lower() == "true")
                    if str(raw).lower() in ("true", "false")
                    else float(raw)
                )
            except (TypeError, ValueError):
                continue
        center = float(np.median(values)) if values else 0.0
        spread = (
            float(np.percentile(values, 75) - np.percentile(values, 25))
            if values
            else 0.0
        )
        result[name] = {"center": center, "scale": max(spread, 1.0)}
    return result


def _technology_vector(node, names, scaler):
    result, missing = [], []
    for name in names:
        raw = node["features"].get(name)
        if raw is None:
            result.append(0.0)
            missing.append(1.0)
            continue
        if isinstance(raw, bool):
            raw = float(raw)
        value = (float(raw) - scaler[name]["center"]) / scaler[name]["scale"]
        result.append(max(-5.0, min(5.0, value)))
        missing.append(0.0)
    return result + missing


def _document_vector(node, snapshot):
    family = node.get("source_family", "")
    from datetime import date

    age = max(
        0,
        (
            date.fromisoformat(snapshot)
            - date.fromisoformat(node["timestamp"][:10])
        ).days,
    )
    features = node["features"]
    return [
        float(features.get("reliability_tier") or 0) / 5,
        float(features.get("fulltext_available") or 0),
        min(age / 365.25 / 30, 1),
        *[float(family == value) for value in DOCUMENT_FEATURES[3:]],
    ]


def _role(node):
    kind = node["type"]
    if kind == "Technology":
        return kind
    if kind == "DocumentVersion":
        return "Document"
    if kind in ("Company", "University", "Organization"):
        return "Organization"
    if kind == "Task":
        return kind
    return None


def _neighbor_node(node, sample, technology_rows):
    """A neighbour technology with its own history row at the snapshot.

    The exported subgraph carries only a document count for neighbours;
    without their rows every neighbour would enter as an all-missing vector.
    """
    if not technology_rows or node["id"] == sample["root_id"]:
        return node
    row = technology_rows.get(
        (node["id"][len("Technology:") :], sample["snapshot"])
    )
    if row is None:
        return node
    return {**node, "features": {**node["features"], **_row_features(row)}}


def _row_features(row):
    result = {}
    for name, raw in row.items():
        if raw in (None, ""):
            result[name] = None
        elif str(raw).lower() in ("true", "false"):
            result[name] = str(raw).lower() == "true"
        else:
            try:
                result[name] = float(raw)
            except (TypeError, ValueError):
                continue
    return result


def hgt_data(sample, names, scaler, technology_rows=None):
    """Convert one existing two-hop JSONL sample to stable HGT roles.

    ``technology_rows`` maps (technology_id, snapshot_date) to history rows
    and fills the features of neighbour technologies.
    """
    import torch
    from torch_geometric.data import HeteroData

    data = HeteroData()
    kept, indices = defaultdict(list), {}
    for node in sample["nodes"]:
        role = _role(node)
        if role:
            indices[node["id"]] = (role, len(kept[role]))
            kept[role].append(node)
    if sample["root_id"] not in indices:
        raise ValueError("Sample root is not a Technology node")
    for role in NODE_TYPES:
        vectors = []
        for node in kept[role]:
            if role == "Technology":
                node = _neighbor_node(node, sample, technology_rows)
                vectors.append(_technology_vector(node, names, scaler))
            elif role == "Document":
                vectors.append(_document_vector(node, sample["snapshot"]))
            elif role == "Organization":
                vectors.append(
                    [
                        float(node["type"] == kind)
                        for kind in ("Company", "University", "Organization")
                    ]
                )
            else:
                vectors.append([1.0])
        width = {
            "Technology": 2 * len(names),
            "Document": len(DOCUMENT_FEATURES),
            "Organization": len(ORG_FEATURES),
            "Task": 1,
        }[role]
        data[role].x = torch.tensor(vectors, dtype=torch.float32).reshape(
            -1, width
        )
        data[role].node_ids = [node["id"] for node in kept[role]]
    edges = defaultdict(list)

    def add(left, relation, right):
        edges[(left[0], relation, right[0])].append((left[1], right[1]))

    for edge in sample["edges"]:
        left = indices.get(edge["source"])
        right = indices.get(edge["target"])
        if not left or not right:
            continue
        relation = edge["type"]
        if (
            relation == "MENTIONED_IN"
            and left[0] == "Technology"
            and right[0] == "Document"
        ):
            from datetime import date

            age = (
                date.fromisoformat(sample["snapshot"])
                - date.fromisoformat(edge["timestamp"][:10])
            ).days
            bucket = (
                "recent" if age <= 365 else "middle" if age <= 1095 else "old"
            )
            add(left, "mentions_" + bucket, right)
            add(
                right,
                {
                    "recent": "recently_mentions",
                    "middle": "middle_mentions",
                    "old": "old_mentions",
                }[bucket],
                left,
            )
        elif (
            relation == "ASSOCIATED_WITH"
            and left[0] == "Document"
            and right[0] == "Organization"
        ):
            add(left, "associated_with", right)
            add(right, "associated_document", left)
        elif (
            relation in ("DEVELOPED_BY", "USED_BY", "FUNDED_BY")
            and left[0] == "Technology"
            and right[0] == "Organization"
        ):
            forward, backward = {
                "DEVELOPED_BY": ("developed_by", "develops"),
                "USED_BY": ("used_by", "uses"),
                "FUNDED_BY": ("funded_by", "funds"),
            }[relation]
            add(left, forward, right)
            add(right, backward, left)
        elif (
            relation == "SOLVES"
            and left[0] == "Technology"
            and right[0] == "Task"
        ):
            add(left, "solves", right)
            add(right, "solved_by", left)
        elif (
            relation == "SUBTECHNOLOGY_OF"
            and left[0] == right[0] == "Technology"
        ):
            add(left, "child_of", right)
            add(right, "parent_of", left)
    for relation in RELATIONS:
        pairs = edges[relation]
        data[relation].edge_index = (
            torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).T.contiguous()
        )
    data.root_index = indices[sample["root_id"]][1]
    return data


def build_model(in_features, hidden=64, heads=4, dropout=0.2):
    """Factory keeps PyG an optional import for non-model workflows."""
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch_geometric.nn import HGTConv

    class SignalHGT(nn.Module):
        def __init__(self):
            super().__init__()
            widths = {
                "Technology": 2 * in_features,
                "Document": len(DOCUMENT_FEATURES),
                "Organization": len(ORG_FEATURES),
                "Task": 1,
            }
            self.projections = nn.ModuleDict(
                {
                    kind: nn.Linear(width, hidden)
                    for kind, width in widths.items()
                }
            )
            self.convolutions = nn.ModuleList(
                [
                    HGTConv(hidden, hidden, METADATA, heads=heads)
                    for _ in range(2)
                ]
            )
            self.tabular = nn.Sequential(
                nn.Linear(2 * in_features, 32), nn.ReLU(), nn.Dropout(dropout)
            )
            self.head = nn.Sequential(
                nn.Linear(hidden + 32, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 2),
            )

        def forward(self, graph):
            representations = {
                kind: F.gelu(self.projections[kind](graph[kind].x))
                for kind in NODE_TYPES
            }
            for convolution in self.convolutions:
                update = convolution(representations, graph.edge_index_dict)
                representations = {
                    kind: F.dropout(
                        update[kind]
                        if update.get(kind) is not None
                        else representations[kind],
                        p=dropout,
                        training=self.training,
                    )
                    for kind in NODE_TYPES
                }
            index = graph.root_index
            if torch.is_tensor(index) and index.dim() == 1:
                # A Batch: root positions are local to each graph.
                index = graph["Technology"].ptr[:-1] + index
            vector = torch.cat(
                (
                    representations["Technology"][index],
                    self.tabular(graph["Technology"].x[index]),
                ),
                dim=-1,
            )
            return self.head(vector)

    return SignalHGT()


def predict_logits(model, graphs, batch_size=256):
    """Signal logits of HeteroData graphs, in their order."""
    import torch
    from torch_geometric.data import Batch

    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch = Batch.from_data_list(graphs[start : start + batch_size])
            result.append(model(batch)[:, 0].numpy())
    return np.concatenate(result) if result else np.zeros(0)


def _predict(model, samples, names, scaler, technology_rows=None):
    return predict_logits(
        model,
        [
            hgt_data(sample, names, scaler, technology_rows)
            for sample in samples
        ],
    )


def fit_hgt(
    train,
    in_features,
    valid=None,
    epochs=60,
    patience=10,
    batch_size=128,
    learning_rate=1e-3,
    seed=13,
):
    """Train on (graph, label, weight, trend) items; returns the best state.

    With ``valid`` (graph, label) items training stops when valid PR-AUC
    has not improved for ``patience`` epochs. Without it the model trains
    exactly ``epochs`` epochs: out-of-fold models must not look at valid.
    """
    import torch
    from sklearn.metrics import average_precision_score
    from torch import nn
    from torch_geometric.data import Batch

    torch.manual_seed(seed)
    generator = random.Random(seed)
    model = build_model(in_features)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    order = list(range(len(train)))
    best, best_score, best_epoch, waited = None, -1.0, epochs, 0
    history = []
    for epoch in range(epochs):
        model.train()
        generator.shuffle(order)
        for start in range(0, len(order), batch_size):
            items = [train[i] for i in order[start : start + batch_size]]
            batch = Batch.from_data_list([item[0] for item in items])
            logits = model(batch)
            labels = torch.tensor([float(item[1]) for item in items])
            weights = torch.tensor([float(item[2]) for item in items])
            loss = (
                nn.functional.binary_cross_entropy_with_logits(
                    logits[:, 0], labels, reduction="none"
                )
                * weights
            ).sum() / weights.sum()
            trend = [
                (i, float(item[3]))
                for i, item in enumerate(items)
                if item[3] is not None
            ]
            if trend:
                rows = torch.tensor([i for i, _ in trend])
                loss = (
                    loss
                    + 0.5
                    * nn.functional.binary_cross_entropy_with_logits(
                        logits[rows, 1],
                        torch.tensor([value for _, value in trend]),
                    )
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if valid is None:
            continue
        scores = _sigmoid(predict_logits(model, [item[0] for item in valid]))
        score = float(
            average_precision_score([item[1] for item in valid], scores)
        )
        history.append(round(score, 5))
        if score > best_score + 1e-4:
            best_score, waited = score, 0
            best = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
        else:
            waited += 1
            if waited >= patience:
                break
    if best is None:
        best = copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    return {
        "model": model,
        "state": best,
        "best_epoch": best_epoch,
        "valid_pr_auc_by_epoch": history,
    }


def train_hgt(
    rows,
    samples,
    output_dir,
    epochs=60,
    seed=13,
    min_train_families_per_class=20,
    min_valid_families_per_class=10,
    target="signal_36m",
    strategy="temporal",
    fold=0,
    technology_rows=None,
):
    """Train the signal head; optional trend labels train a second head.

    Neighbour technologies take their features from ``technology_rows``,
    by default every row passed in, labelled or not.
    """
    import torch

    require_trainable(
        rows,
        min_train_families_per_class,
        min_valid_families_per_class,
        target,
        strategy,
        fold,
    )
    months = TARGETS[target]
    if technology_rows is None:
        technology_rows = {
            (row["technology_id"], row["snapshot_date"]): row for row in rows
        }
    by_key = {
        (row["technology_id"], row["snapshot_date"]): row
        for row in rows
        if str(row.get(target)) in ("0", "1")
    }
    sets = defaultdict(list)
    for sample in samples:
        row = by_key.get((sample["technology_id"], sample["snapshot"]))
        if row:
            part = split_for(row, target, strategy, fold)
            if part in ("train", "valid", "test"):
                sets[part].append((row, sample))
    if not sets["train"] or not sets["valid"]:
        raise ValueError("HGT needs reviewed train and valid subgraphs")
    for part in ("train", "valid"):
        if len({int(row[target]) for row, _ in sets[part]}) != 2:
            raise ValueError(f"HGT {part} subgraphs need both classes")
    names = [
        name
        for name in FEATURES
        if any(row.get(name) not in (None, "") for row, _ in sets["train"])
    ]
    scaler = fit_scaler([row for row, _ in sets["train"]], names)
    graphs = {
        part: [
            (row, hgt_data(sample, names, scaler, technology_rows))
            for row, sample in values
        ]
        for part, values in sets.items()
    }

    def trend(row):
        value = row.get(f"trend_{months}m")
        return float(value) if str(value) in ("0", "1") else None

    fitted = fit_hgt(
        [
            (graph, int(row[target]), 1.0, trend(row))
            for row, graph in graphs["train"]
        ],
        len(names),
        valid=[(graph, int(row[target])) for row, graph in graphs["valid"]],
        epochs=epochs,
        seed=seed,
    )
    model, best, best_epoch = (
        fitted["model"],
        fitted["state"],
        fitted["best_epoch"],
    )
    logits = {
        part: predict_logits(model, [graph for _, graph in graphs[part]])
        for part in graphs
    }
    temperature = fit_temperature(
        logits["valid"], [int(row[target]) for row, _ in graphs["valid"]]
    )
    metrics = {}
    for part in ("valid", "test"):
        if graphs.get(part):
            labels = [int(row[target]) for row, _ in graphs[part]]
            metrics[part] = _metrics(
                labels,
                _sigmoid(logits[part] / temperature),
                np.ones(len(labels)),
            )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = model_stem("hgt", target, strategy, fold)
    torch.save(best, output_dir / f"{stem}.pt")
    report = {
        "model": "two-layer four-role HGT",
        "target": target,
        "split_strategy": strategy,
        "family_fold": fold if strategy == "family" else None,
        "features": names,
        "scaler_train_only": scaler,
        "temperature_valid_only": temperature,
        "best_epoch": best_epoch,
        "metrics": metrics,
        "explanation": (
            "node ablation; raw attention is not treated as importance"
        ),
    }
    (output_dir / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def explain_hgt(model, sample, report, top_k=10, technology_rows=None):
    """Signed changes from deleting each neighbour or replacing root inputs."""
    import torch

    names, scaler = report["features"], report["scaler_train_only"]
    temperature = report["temperature_valid_only"]

    def probability(current):
        model.eval()
        with torch.no_grad():
            raw = model(hgt_data(current, names, scaler, technology_rows))[
                0
            ].item()
        return float(_sigmoid(raw / temperature))

    full = probability(sample)
    neighbors = []
    for node in sample["nodes"]:
        if node["id"] == sample["root_id"] or _role(node) is None:
            continue
        reduced = dict(sample)
        reduced["nodes"] = [
            item for item in sample["nodes"] if item["id"] != node["id"]
        ]
        reduced["edges"] = [
            edge
            for edge in sample["edges"]
            if node["id"] not in (edge["source"], edge["target"])
        ]
        change = 100 * (full - probability(reduced))
        neighbors.append(
            {
                "node_id": node["id"],
                "type": node["type"],
                "timestamp": node["timestamp"],
                "source_family": node.get("source_family"),
                "document_type": node.get("document_type"),
                "title": node.get("title"),
                "url": node.get("url"),
                "effect_probability_points": round(change, 3),
            }
        )
    neighbors.sort(
        key=lambda item: abs(item["effect_probability_points"]), reverse=True
    )
    root = next(
        node for node in sample["nodes"] if node["id"] == sample["root_id"]
    )
    features = []
    for name in names:
        original = root["features"].get(name)
        if original is None:
            continue
        reduced = copy.deepcopy(sample)
        target = next(
            node
            for node in reduced["nodes"]
            if node["id"] == reduced["root_id"]
        )
        baseline = scaler[name]["center"]
        target["features"][name] = baseline
        change = 100 * (full - probability(reduced))
        features.append(
            {
                "feature": name,
                "value": original,
                "train_median": baseline,
                "effect_probability_points": round(change, 3),
            }
        )
    features.sort(
        key=lambda item: abs(item["effect_probability_points"]), reverse=True
    )
    return {
        "probability": full,
        "entities": neighbors[:top_k],
        "features": features[:top_k],
        "method": "one-at-a-time model ablation, not causal influence",
    }


def train_from_files(
    dataset,
    subgraphs,
    output_dir,
    epochs=60,
    target="signal_36m",
    strategy="temporal",
    fold=0,
):
    samples = [
        json.loads(line)
        for line in Path(subgraphs).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return train_hgt(
        _read_csv(dataset),
        samples,
        output_dir,
        epochs,
        target=target,
        strategy=strategy,
        fold=fold,
    )
