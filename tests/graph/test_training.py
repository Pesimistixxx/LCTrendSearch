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
