import copy
import csv
import json

from lctrend.graph.temporal import TemporalCorpus
from lctrend.graph.training import (
    build_dataset_rows,
    build_snapshot_rows,
    temporal_split,
    write_dataset_rows,
    write_snapshot_rows,
)


def version(document_id, when, family="scholarly", group="origin"):
    return {
        "document_id": document_id,
        "version_id": document_id + "-v1",
        "document_type": {
            "code": "repository",
            "package_registry": "package",
            "patent": "patent",
        }.get(family, "article"),
        "source_family": family,
        "source_id": family,
        "document_published_at": when,
        "version_published_at": when,
        "retrieved_at": when,
        "metrics_observed_at": when,
        "independence_group": group,
        "coverage": "full_text",
        "domains": ["ai"],
        "countries": ["US"],
        "extracted": True,
        "extracted_at": when,
    }


def dataset(family="code"):
    versions = [
        version("past1", "2018-01-01"),
        version("past2", "2019-01-01"),
        version("future1", "2021-01-01", family, "team-one"),
        version("future2", "2022-01-01", "scholarly", "team-two"),
        version("end", "2026-01-01"),
    ]
    return {
        "versions": versions,
        "technologies": [{"technology_id": "t", "technology": "Example"}],
        "mentions": [
            {
                "technology_id": "t",
                "version_id": row["version_id"],
                "observed_at": row["version_published_at"],
                "mentions": 1,
                "accepted": 1,
            }
            for row in versions
            if row["document_id"] != "end"
        ],
        "maturity": [
            {
                "technology_id": "t",
                "version_id": "past2-v1",
                "observed_at": "2019-01-01",
                "stage_rank": 3,
                "trl": 4,
            }
        ],
    }


COLLECTED = "2026-09-20T10:00:00+00:00"
PROCESSED = "2026-09-21T08:00:00+00:00"


def collected_in_2026():
    """Articles published 2016-2023, all collected and processed in 2026-09.

    This is what the system builds for itself: every retrieval, metric and
    extraction timestamp is from the collection run, not from publication.
    """
    versions, mentions = [], []
    for year in range(2016, 2024):
        row = version(f"a{year}", f"{year}-03-01", group=f"team-{year}")
        row.update(
            retrieved_at=COLLECTED,
            metrics_observed_at=COLLECTED,
            metrics_json={"citation_count": 5},
            extracted_at=PROCESSED,
        )
        versions.append(row)
        mentions.append(
            {
                "technology_id": "t",
                "version_id": row["version_id"],
                "observed_at": row["version_published_at"],
                "recorded_at": PROCESSED,
                "mentions": 1,
                "accepted": 1,
            }
        )
    return {
        "versions": versions,
        "technologies": [{"technology_id": "t", "technology": "Example"}],
        "mentions": mentions,
        "maturity": [
            {
                "technology_id": "t",
                "version_id": "a2016-v1",
                "observed_at": "2016-03-01",
                "recorded_at": PROCESSED,
                "stage_rank": 3,
            }
        ],
    }


def test_corpus_collected_today_yields_historical_rows():
    rows = build_dataset_rows(
        TemporalCorpus(collected_in_2026()), start_year=2016
    )
    historical = [row for row in rows if row["snapshot_date"] < "2026"]
    assert historical
    row = next(row for row in rows if row["snapshot_date"] == "2020-01-01")
    assert row["first_seen_date"] == "2016-03-01"
    assert row["document_count"] == 4
    assert row["max_maturity_rank"] == 3
    # Metrics are mutable: known only from their own observation in 2026.
    assert row["citation_count"] is None
    current = next(r for r in rows if r["snapshot_date"] == "2026-01-01")
    assert current["documents_last_year"] == 0
    assert current["technology_age_days"] > 3000


def test_as_known_mode_keeps_collection_and_processing_gates():
    corpus = TemporalCorpus(collected_in_2026(), as_known=True)
    rows = build_dataset_rows(corpus, start_year=2016)
    assert [row for row in rows if row["snapshot_date"] < "2026"] == []
    assert build_snapshot_rows(corpus, "2026-09-19") == []
    (row,) = build_snapshot_rows(corpus, "2026-09-21")
    assert row["first_seen_date"] == "2026-09-21"


def with_undated_upload(data):
    """A file uploaded in 2026 without any publication date."""
    upload = version("upload", None)
    upload.update(
        retrieved_at=COLLECTED, metrics_observed_at=None, extracted_at=None
    )
    data["versions"].append(upload)
    data["mentions"] += [
        {
            "technology_id": technology,
            "version_id": "upload-v1",
            # Legacy graphs stored the upload time as the mention date.
            "observed_at": COLLECTED[:10],
            "mentions": 5,
            "accepted": 5,
        }
        for technology in ("t", "only-upload")
    ]
    return data


def test_undated_documents_stay_out_of_dynamics_and_first_seen(tmp_path):
    corpus = TemporalCorpus(with_undated_upload(collected_in_2026()))
    # Collected in 2026, so no earlier snapshot can know the upload.
    assert build_snapshot_rows(corpus, "2025-01-01")[0]["document_count"] == 8
    rows = {
        row["technology_id"]: row
        for row in build_snapshot_rows(corpus, "2026-09-21")
    }
    row = rows["t"]
    assert row["first_seen_date"] == "2016-03-01"
    assert row["document_count"] == 9
    assert row["mention_count"] == 13
    assert row["documents_last_year"] == 0
    assert row["publication_growth"] == 0.0
    assert row["mention_growth_12m"] == 0.0
    assert row["burst_score"] == 0.0
    upload_only = rows["only-upload"]
    assert upload_only["first_seen_date"] is None
    assert upload_only["technology_age_days"] is None
    assert upload_only["documents_last_year"] == 0
    output = tmp_path / "snapshot.csv"
    write_snapshot_rows(output, list(rows.values()), corpus)
    manifest = json.loads(output.with_suffix(".csv.manifest.json").read_text())
    assert manifest["undated_documents"] == 1


def at_2020(data):
    return next(
        row
        for row in build_dataset_rows(
            TemporalCorpus(data),
            start_year=2020,
            min_documents=2,
        )
        if row["snapshot_date"] == "2020-01-01"
    )


def cover_outcomes(data):
    data["crawls"] = [
        {
            "source_family": family,
            "query": "Example",
            "exhaustive": True,
            "status": "completed",
            "failures": 0,
            "period_start": "2020-01-01",
            "period_end": "2023-01-01",
            "finished_at": "2023-01-01",
        }
        for family in (
            "scholarly",
            "code",
            "package_registry",
            "patent",
            "commercial",
        )
    ]


def test_training_and_snapshot_share_all_predictors():
    data = dataset()
    snapshot = build_snapshot_rows(
        TemporalCorpus(data),
        "2020-01-01",
        min_documents=2,
    )[0]
    row = at_2020(data)
    assert {name: row[name] for name in snapshot} == snapshot
    assert row["label_realized"] == 1
    assert row["future_repositories"] == 1
    assert row["future_independent_sources"] == 2
    assert row["document_count"] == 2
    assert "mention_growth_3m" in row and "pagerank_delta_12m" in row


def test_many_future_articles_are_not_realization():
    data = dataset("scholarly")
    cover_outcomes(data)
    row = at_2020(data)
    assert row["future_document_count"] == 2
    assert row["future_independent_sources"] == 2
    assert row["label_realized"] == 0


def test_missing_outcome_sources_do_not_become_negative_labels():
    row = at_2020(dataset("scholarly"))
    assert row["label_realized"] is None
    assert row["label_reason"] == "source_coverage_incomplete"
    assert row["split"] == "unlabeled"


def test_linked_package_and_repository_are_one_confirmation():
    data = dataset()
    data["versions"][3].update(
        source_family="package_registry",
        document_type="package",
        independence_group="team-one",
    )
    cover_outcomes(data)
    row = at_2020(data)
    assert row["future_repositories"] == row["future_packages"] == 1
    assert row["future_independent_sources"] == 1
    assert row["label_realized"] == 0


def test_unidentified_documents_do_not_prove_independence():
    data = dataset()
    for row in data["versions"][2:4]:
        row["independence_group"] = None
    row = at_2020(data)
    assert row["future_independent_sources"] == 0
    assert row["label_realized"] is None


def test_unknown_or_commercial_snapshot_maturity_excludes_labels():
    data = dataset()
    data["maturity"] = []
    assert at_2020(data)["label_reason"] == "maturity_unknown"
    data = dataset()
    data["maturity"][0]["stage_rank"] = 5
    row = at_2020(data)
    assert row["label_realized"] is None
    assert row["label_reason"] == "already_commercial"


def test_censored_horizon_remains_unlabeled_even_if_positive_signal_seen():
    rows = build_dataset_rows(
        TemporalCorpus(dataset()),
        start_year=2020,
        end_date="2022-02-01",
    )
    row = rows[0]
    assert row["future_repositories"] == 1
    assert row["future_independent_sources"] == 2
    assert row["label_realized"] is None
    assert row["label_reason"] == "horizon_censored"
    assert row["outcome_observation_complete"] is False


def test_candidate_and_planned_commercial_evidence_do_not_label_positive():
    data = dataset("scholarly")
    cover_outcomes(data)
    data["economics"] = [
        {
            "technology_id": "t",
            "version_id": "future1-v1",
            "observed_at": "2021-01-01",
            "category": "commercialization",
            "polarity": "affirmed",
            "modality": modality,
            "status": status,
        }
        for modality, status in (
            ("reported", "candidate"),
            ("planned", "accepted"),
        )
    ]
    row = at_2020(data)
    assert row["future_commercial_evidence"] == 0
    assert row["label_realized"] == 0


def test_late_metadata_and_mentions_cannot_change_old_predictors():
    data = dataset()
    before = build_snapshot_rows(TemporalCorpus(data), "2020-01-01")
    changed = copy.deepcopy(data)
    late = version("past1", "2025-01-01")
    late.update(
        version_id="past1-late",
        contributors=["new-author"],
        countries=["GB"],
        metrics_json='{"citation_count": 10000}',
        metadata_json='{"release_dates": ["2019-01-01"]}',
    )
    changed["versions"].append(late)
    changed["mentions"].append(
        {
            "technology_id": "t",
            "version_id": "past1-v1",
            "observed_at": "2026-01-01",
            "mentions": 999,
        }
    )
    after = build_snapshot_rows(TemporalCorpus(changed), "2020-01-01")
    assert before == after


def test_temporal_splits_purge_horizons_and_retain_usable_validation():
    rows = [
        {
            "snapshot_date": f"{year}-01-01",
            "horizon_end": f"{year + 3}-01-01",
            "label_realized": 1,
        }
        for year in range(2000, 2011)
    ]
    temporal_split(rows)
    splits = {row["snapshot_date"]: row["split"] for row in rows}
    assert splits["2010-01-01"] == "test"
    assert splits["2006-01-01"] == "valid"
    assert splits["2002-01-01"] == "train"
    assert splits["2003-01-01"] == splits["2009-01-01"] == "purged"
    train = [row for row in rows if row["split"] == "train"]
    valid = [row for row in rows if row["split"] == "valid"]
    assert max(row["horizon_end"] for row in train) < valid[0]["snapshot_date"]
    assert max(row["horizon_end"] for row in valid) < "2010-01-01"


def test_csv_and_manifest_separate_outcomes_from_predictors(tmp_path):
    corpus = TemporalCorpus(dataset())
    output = tmp_path / "train.csv"
    rows = build_dataset_rows(corpus, start_year=2020)
    assert write_dataset_rows(output, rows) == len(rows)
    manifest = json.loads(output.with_suffix(".csv.manifest.json").read_text())
    assert "mention_growth_12m" in manifest["feature_columns"]
    assert not any(
        name.startswith("future_") for name in manifest["feature_columns"]
    )
    assert "label_realized" not in manifest["feature_columns"]
    # Duplicate columns are gone; the manifest names their replacement.
    assert manifest["column_aliases"]["taxonomy_depth"] == "taxonomy_level"
    assert not set(manifest["column_aliases"]) & set(
        manifest["feature_columns"]
    )
    with output.open(newline="", encoding="utf-8") as stream:
        exported = list(csv.DictReader(stream))
    assert exported[0]["label_realized"] == "1"
    features = tmp_path / "snapshot.csv"
    write_snapshot_rows(features, build_snapshot_rows(corpus, "2020-01-01"))
    with features.open(newline="", encoding="utf-8") as stream:
        names = csv.DictReader(stream).fieldnames
    assert not any(name.startswith("future_") for name in names)
    assert "label_realized" not in names


def test_empty_dataset_exports_the_same_schema(tmp_path):
    output = tmp_path / "empty.csv"
    assert build_dataset_rows(TemporalCorpus({})) == []
    assert write_dataset_rows(output, []) == 0
    with output.open(newline="", encoding="utf-8") as stream:
        names = csv.DictReader(stream).fieldnames
    assert "mention_growth_12m" in names
    assert "label_realized" in names
