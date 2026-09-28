"""Exported CSVs follow the current predictor schema.

Committed exports once drifted to a 21-column legacy format without a
manifest; these tests fail when any export or artifact leaves the schema.
"""

import csv
import json
from pathlib import Path

from lctrend.core.config import load_catalog
from lctrend.graph.temporal import TemporalCorpus
from lctrend.graph.training import (
    IDENTITY_FIELDS,
    OUTCOME_FIELDS,
    dataset_feature_fields,
    write_dataset_rows,
    write_snapshot_rows,
)

REPOSITORY = Path(__file__).resolve().parents[2]


def expected_header(training):
    fields = IDENTITY_FIELDS + dataset_feature_fields()
    return list(dict.fromkeys(fields + (OUTCOME_FIELDS if training else [])))


def header(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return next(csv.reader(stream))


def test_exports_write_exactly_the_shared_feature_schema(tmp_path):
    corpus = TemporalCorpus({})
    features, training = tmp_path / "features.csv", tmp_path / "train.csv"
    write_snapshot_rows(features, [], corpus)
    write_dataset_rows(training, [], corpus)
    assert header(features) == expected_header(False)
    assert header(training) == expected_header(True)
    manifest = json.loads(
        training.with_suffix(".csv.manifest.json").read_text()
    )
    assert manifest["feature_columns"] == dataset_feature_fields()


def test_committed_exports_match_the_current_schema():
    for directory, training in (("features", False), ("training", True)):
        for path in sorted((REPOSITORY / "artifacts" / directory).glob(
            "*.csv"
        )):
            manifest = path.with_suffix(".csv.manifest.json")
            assert manifest.exists(), f"{path.name} has no manifest"
            assert header(path) == expected_header(training), path.name


def test_runtime_catalog_keeps_only_live_output_paths():
    runtime = load_catalog("runtime")
    # Snapshot and label parameters live in dataset.json only.
    assert "training" not in runtime
    assert runtime["training_output"].endswith(".csv")
    assert runtime["features_output"].endswith(".csv")
