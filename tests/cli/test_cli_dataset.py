"""CLI integration through an offline, dated graph-store boundary."""

import csv
import json
import sys

import pytest

from lctrend import cli


class TemporalStore:
    def __init__(self, data):
        self.data = data
        self.reads = 0
        self.audits = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def ensure_schema(self):
        pass

    async def read_temporal_data(self):
        self.reads += 1
        return self.data

    def read_training_data(self):
        pytest.fail("Dataset commands must read dated graph data")

    def read_signal_data(self):
        pytest.fail("Dataset commands must share the temporal corpus")

    async def write_crawl_run(self, run):
        self.audits.append(dict(run))


@pytest.fixture
def temporal_store(monkeypatch, tmp_path):
    versions = []
    mentions = []
    for identifier, published, kind, group in (
        ("paper1", "2018-01-01", "article", "research-a"),
        ("paper2", "2019-01-01", "article", "research-b"),
        ("repo", "2021-06-01", "repository", "implementation-a"),
        ("patent", "2022-06-01", "patent", "implementation-b"),
        ("tail", "2024-01-01", "article", "tail"),
    ):
        version = identifier + "-v1"
        versions.append({
            "document_id": identifier,
            "version_id": version,
            "document_type": kind,
            "source_id": "source:" + kind,
            "independence_group": group,
            "document_published_at": published,
            "version_published_at": published,
            "retrieved_at": published,
            "extracted_at": published,
            "metrics_observed_at": published,
            "metrics_json": {"citation_count": 3} if kind == "article" else {},
            "extracted": True,
        })
        if identifier != "tail":
            mentions.append({
                "technology_id": "t1", "technology": "Local technology",
                "version_id": version, "observed_at": published,
                "mentions": 1, "accepted": 1,
            })
    # An update to an old article must not leak today's metrics into 2020.
    versions.append({
        **versions[0], "version_id": "paper1-future",
        "version_published_at": "2023-01-01",
        "retrieved_at": "2023-01-01", "extracted_at": "2023-01-01",
        "metrics_observed_at": "2023-01-01",
        "metrics_json": {"citation_count": 999},
    })
    mentions.append({
        **mentions[0], "version_id": "paper1-future",
        "observed_at": "2023-01-01",
    })
    store = TemporalStore({
        "versions": versions, "mentions": mentions,
        "maturity": [{
            "technology_id": "t1", "version_id": "paper2-v1",
            "observed_at": "2019-01-01", "stage_rank": 3,
        }],
        "technologies": [{
            "technology_id": "t1", "technology": "Local technology",
        }],
    })
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr(cli, "setup_logging", lambda: tmp_path / "log.txt")
    monkeypatch.setattr(cli, "_store", lambda: store)
    monkeypatch.delenv("LCTREND_CONFIG_DIR", raising=False)
    return store


def invoke(monkeypatch, *arguments):
    monkeypatch.setattr(sys, "argv", ["lctrend", *map(str, arguments)])
    cli.main()


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_snapshot_cli_uses_dated_versions_and_metrics(
    monkeypatch, tmp_path, temporal_store
):
    output = tmp_path / "snapshot.csv"
    invoke(monkeypatch, "export-features", "--snapshot", "2020-01-01",
           "--output", output, "--no-taxonomy")
    row, = read_csv(output)
    assert temporal_store.reads == 1
    assert row["snapshot_date"] == "2020-01-01"
    assert row["document_count"] == "2"
    assert float(row["citation_count"]) == 6
    assert "label_realized" not in row
    manifest = json.loads(output.with_suffix(".csv.manifest.json").read_text())
    assert manifest["label_column"] is None
    assert "document_count" in manifest["feature_columns"]


def test_dataset_cli_exports_labels_and_matching_snapshot_subgraphs(
    monkeypatch, tmp_path, temporal_store
):
    output, graphs = tmp_path / "dataset.csv", tmp_path / "graphs.jsonl"
    invoke(monkeypatch, "build-training-set", "--start-year", "2020",
           "--horizon-years", "3", "--end-date", "2024-01-01",
           "--output", output, "--subgraphs-output", graphs, "--no-taxonomy")
    rows = read_csv(output)
    first = next(row for row in rows if row["snapshot_date"] == "2020-01-01")
    assert temporal_store.reads == 1
    assert first["document_count"] == "2"
    assert first["future_repositories"] == "1"
    assert first["future_patents"] == "1"
    assert first["label_realized"] == "1"
    assert first["horizon_end"] == "2023-01-01"
    last = next(row for row in rows if row["snapshot_date"] == "2024-01-01")
    assert last["label_realized"] == ""
    assert last["label_reason"] == "horizon_censored"
    assert last["split"] == "unlabeled"
    manifest = json.loads(output.with_suffix(".csv.manifest.json").read_text())
    assert not any(name.startswith("future_")
                   for name in manifest["feature_columns"])
    assert "label_realized" not in manifest["feature_columns"]
    samples = [json.loads(line) for line in graphs.read_text().splitlines()]
    assert len(samples) == len(rows)
    sample = samples[0]
    assert sample["label"] == 1
    assert sample["split"] == first["split"]
    root = next(node for node in sample["nodes"]
                if node["id"] == sample["root_id"])
    assert root["features"]["mention_growth_3m"] == float(
        first["mention_growth_3m"]
    )
    assert not any(name.startswith(("future_", "label_"))
                   for name in root["features"])
    graph_manifest = json.loads(
        graphs.with_suffix(".jsonl.manifest.json").read_text()
    )
    assert "mention_growth_3m" in graph_manifest["node_features"]["Technology"]
    assert all(node["timestamp"] <= "2020-01-01" for node in sample["nodes"])
    assert not any(node["id"] == "DocumentVersion:paper1-future"
                   for node in sample["nodes"])


@pytest.mark.parametrize("flag", [
    "--positive-future-documents", "--negative-future-documents",
])
def test_removed_popularity_threshold_flags_are_rejected(
    monkeypatch, temporal_store, flag
):
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, "build-training-set", flag, "5")
    assert error.value.code == 2
    assert temporal_store.reads == 0


def test_empty_limited_crawl_records_search_without_claiming_absence(
    monkeypatch, tmp_path, temporal_store
):
    monkeypatch.setattr(cli, "fetch_openalex_page", lambda *args: {
        "results": [], "meta": {"next_cursor": None},
    })
    cli._crawl_openalex(
        "fixture", 2, 2, tmp_path / "checkpoint.json", False,
        fulltext=False,
        filter="from_publication_date:2018-01-01,to_publication_date:2024-01-01",
    )
    run = temporal_store.audits[-1]
    assert run["status"] == "completed" and run["exhaustive"] is False
    assert run["source_family"] == "scholarly"
    assert run["period_start"] == "2018-01-01"
    assert run["period_end"] == "2024-01-01"
    assert run["records_seen"] == run["records_ingested"] == 0
    assert run["finished_at"]
    assert json.loads(run["checkpoint_json"])["cursor"] is None


def test_failed_source_page_finalizes_crawl_audit(
    monkeypatch, tmp_path, temporal_store
):
    def fail(*args):
        raise RuntimeError("offline source failed")

    monkeypatch.setattr(cli, "fetch_openalex_page", fail)
    with pytest.raises(RuntimeError, match="offline source failed"):
        cli._crawl_openalex("fixture", 2, 2, tmp_path / "checkpoint.json",
                            False, fulltext=False)
    assert temporal_store.audits[-1]["status"] == "failed"
    assert temporal_store.audits[-1]["finished_at"]


def test_pypi_crawl_persists_success_and_failure_counts(
    monkeypatch, tmp_path, temporal_store
):
    def fetch(name):
        if name == "bad":
            raise ValueError("bad package fixture")
        return {"info": {"name": name, "version": "1"}, "releases": {}}

    async def write(*args, **kwargs):
        return None

    monkeypatch.setattr(cli, "fetch_pypi", fetch)
    monkeypatch.setattr(cli, "_snapshot", lambda document, raw: document)
    monkeypatch.setattr(cli, "_write_ingested_async", write)
    cli._crawl_pypi(2, tmp_path / "pypi.json", False,
                    requested_packages=["good", "bad"])
    run = temporal_store.audits[-1]
    assert run["source_family"] == "package_registry"
    assert run["records_seen"] == 2
    assert run["records_ingested"] == 1 and run["failures"] == 1
    assert run["status"] == "completed" and run["exhaustive"] is False


@pytest.fixture
def collected_store(monkeypatch, tmp_path, temporal_store):
    """The same corpus, collected and processed in 2026-09."""
    for row in temporal_store.data["versions"]:
        row.update(
            retrieved_at="2026-09-20", metrics_observed_at="2026-09-20",
            extracted_at="2026-09-21",
        )
    for row in temporal_store.data["mentions"]:
        row["recorded_at"] = "2026-09-21"
    return temporal_store


def test_dataset_cli_dates_content_by_publication(
    monkeypatch, tmp_path, collected_store
):
    output = tmp_path / "dataset.csv"
    invoke(monkeypatch, "build-training-set", "--start-year", "2020",
           "--output", output, "--no-taxonomy")
    rows = read_csv(output)
    first = next(row for row in rows if row["snapshot_date"] == "2020-01-01")
    assert first["first_seen_date"] == "2018-01-01"
    assert first["document_count"] == "2"
    assert first["citation_count"] == ""
    manifest = json.loads(output.with_suffix(".csv.manifest.json").read_text())
    assert manifest["as_known"] is False


def test_as_known_flag_keeps_the_strict_collection_gate(
    monkeypatch, tmp_path, collected_store
):
    output = tmp_path / "dataset.csv"
    invoke(monkeypatch, "build-training-set", "--start-year", "2020",
           "--output", output, "--no-taxonomy", "--as-known")
    assert [row for row in read_csv(output)
            if row["snapshot_date"] < "2026"] == []
    manifest = json.loads(output.with_suffix(".csv.manifest.json").read_text())
    assert manifest["as_known"] is True
    features = tmp_path / "features.csv"
    for flag, expected in ((), 1), (("--as-known",), 0):
        invoke(monkeypatch, "export-features", "--snapshot", "2020-01-01",
               "--output", features, "--no-taxonomy", *flag)
        assert len(read_csv(features)) == expected
