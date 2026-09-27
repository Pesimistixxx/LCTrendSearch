from lctrend.graph.training import build_feature_rows, build_training_rows


def test_training_rows_use_only_past_features_and_future_label():
    rows = build_training_rows(
        [
            {
                "technology_id": "t1",
                "technology": "Ner",
                "document_id": "d1",
                "mentions": 2,
            },
            {
                "technology_id": "t1",
                "technology": "Ner",
                "document_id": "d2",
                "mentions": 1,
            },
            {
                "technology_id": "t1",
                "technology": "Ner",
                "document_id": "d3",
                "mentions": 1,
            },
            {
                "technology_id": "t1",
                "technology": "Ner",
                "document_id": "d4",
                "mentions": 1,
            },
        ],
        [
            {
                "document_id": "d1",
                "created_at": "2019-01-01",
                "source_id": "openalex",
                "countries": ["US"],
                "companies": [],
                "universities": [],
                "domains": ["ai"],
            },
            {
                "document_id": "d2",
                "created_at": "2020-01-01",
                "source_id": "openalex",
                "countries": ["GB"],
                "companies": ["c1"],
                "universities": [],
                "domains": ["ai"],
            },
            {
                "document_id": "d3",
                "created_at": "2021-01-01",
                "source_id": "openalex",
                "countries": [],
                "companies": [],
                "universities": [],
                "domains": ["ai"],
            },
            {
                "document_id": "d4",
                "created_at": "2022-01-01",
                "source_id": "openalex",
                "countries": [],
                "companies": [],
                "universities": [],
                "domains": ["ai"],
            },
            {
                "document_id": "latest",
                "created_at": "2025-01-01",
                "source_id": "openalex",
                "countries": [],
                "companies": [],
                "universities": [],
                "domains": [],
            },
        ],
        [],
        start_year=2020,
        horizon_years=3,
        min_documents=2,
        positive_future_documents=2,
        negative_future_documents=0,
    )
    row = next(row for row in rows if row["snapshot_date"] == "2020-01-01")
    assert row["document_count"] == 2
    assert row["country_count"] == 2
    assert row["future_document_count"] == 2
    assert row["label_realized_3y"] == 1


def test_feature_rows_mark_missing_source_families_unavailable():
    rows = build_feature_rows(
        [
            {
                "technology_id": "t",
                "technology": "Tech",
                "document_id": "d",
                "mentions": 1,
            }
        ],
        [
            {
                "document_id": "d",
                "created_at": "2020-01-01",
                "source_id": "s",
                "source_family": "scholarly",
                "independence_group": "s",
                "metrics_json": '{"citation_count": 2}',
                "countries": [],
                "companies": [],
                "universities": [],
                "domains": [],
            }
        ],
        [],
        "2020-12-31",
    )
    assert rows[0]["citation_count"] == 2
    assert rows[0]["patent_data_available"] is False


def test_text_signals_count_only_what_was_observed_by_the_snapshot():
    mentions = [
        {
            "technology_id": "t",
            "technology": "Tech",
            "document_id": "d",
            "mentions": 1,
        }
    ]
    documents = [
        {
            "document_id": "d",
            "created_at": "2020-01-01",
            "source_id": "s",
            "source_family": "scholarly",
            "independence_group": None,
            "metrics_json": "{}",
            "countries": [],
            "companies": [],
            "universities": [],
            "domains": [],
        }
    ]
    signals = [
        {
            "technology_id": "t",
            "signal": "DEVELOPED_BY",
            "target_id": "o1",
            "value": None,
            "observed_at": "2020-02-01",
        },
        {
            "technology_id": "t",
            "signal": "DEVELOPED_BY",
            "target_id": "o1",
            "value": None,
            "observed_at": "2020-03-01",
        },
        {
            "technology_id": "t",
            "signal": "USED_BY",
            "target_id": "o2",
            "value": None,
            "observed_at": "2020-03-01",
        },
        {
            "technology_id": "t",
            "signal": "MATURITY",
            "target_id": 6,
            "value": 4,
            "observed_at": "2020-04-01",
        },
        {
            "technology_id": "t",
            "signal": "ECONOMIC",
            "target_id": "investment",
            "value": 5e6,
            "observed_at": "2020-05-01",
        },
        {
            "technology_id": "t",
            "signal": "FUNDED_BY",
            "target_id": "o3",
            "value": None,
            "observed_at": "2022-01-01",
        },
    ]
    row = build_feature_rows(mentions, documents, [], "2020-12-31", signals)[0]
    assert row["developer_count"] == 1
    assert row["user_count"] == 1
    assert row["funder_count"] == 0
    assert (row["max_maturity_rank"], row["max_trl"]) == (4, 6)
    assert row["economic_evidence_count"] == 1
    assert row["economic_data_available"] is True
    empty = build_feature_rows(mentions, documents, [], "2020-12-31")[0]
    assert empty["economic_data_available"] is False
    assert empty["max_trl"] is None
