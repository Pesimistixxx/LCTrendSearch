from datetime import date, datetime

import pytest

from lctrend.graph.temporal import TemporalCorpus, parse_date


def version(identifier="v1", document="d1", published="2020-01-01", **extra):
    return {
        "document_id": document,
        "version_id": identifier,
        "document_type": "repository",
        "version_published_at": published,
        "retrieved_at": published,
        **extra,
    }


def mention(identifier="v1", observed="2020-01-01", technology="t1", **extra):
    return {
        "technology_id": technology,
        "version_id": identifier,
        "observed_at": observed,
        "mentions": 1,
        **extra,
    }


def crawl(**extra):
    return {
        "source_family": "patent",
        "status": "completed",
        "finished_at": "2021-01-01",
        "exhaustive": True,
        "query": "Technology One",
        **extra,
    }


def test_history_and_parties_are_taken_from_the_visible_version():
    corpus = TemporalCorpus(
        {
            "versions": [
                version(
                    metadata_json={"release_dates": ["2020-02-01"]},
                    companies=["early-company"],
                ),
                version(
                    "v2",
                    published="2022-01-01",
                    metadata_json={
                        "release_dates": ["2019-01-01", "2020-02-01"]
                    },
                    companies=["future-company"],
                ),
            ],
            "mentions": [mention(), mention("v2", "2022-01-01")],
        }
    )
    trace = corpus.view(date(2021, 1, 1)).technologies["t1"].documents[0]
    assert trace.version.version_id == "v1"
    assert trace.version.companies == ("early-company",)
    assert trace.history["release_dates"] == ["2020-02-01"]


def test_content_is_visible_from_publication_despite_later_retrieval():
    data = {
        "versions": [version(retrieved_at="2022-01-01")],
        "mentions": [mention()],
    }
    trace = (
        TemporalCorpus(data)
        .view(date(2021, 1, 1))
        .technologies["t1"]
        .documents[0]
    )
    assert trace.first_visible == date(2020, 1, 1)


def test_as_known_mode_does_not_backdate_a_later_retrieval():
    corpus = TemporalCorpus(
        {
            "versions": [version(retrieved_at="2022-01-01")],
            "mentions": [mention()],
        },
        as_known=True,
    )
    assert corpus.view(date(2021, 1, 1)).technologies == {}
    trace = corpus.view(date(2022, 1, 1)).technologies["t1"].documents[0]
    assert trace.first_visible == date(2022, 1, 1)
    assert trace.version.version_date == date(2020, 1, 1)


def test_versions_are_selected_by_publication_with_separate_retrieval_gate():
    corpus = TemporalCorpus(
        {
            "versions": [
                version("v1", retrieved_at="2023-01-01"),
                version("v2", published="2021-01-01"),
            ]
        }
    )
    assert corpus.visible_version("d1", date(2022, 1, 1)).version_id == "v2"
    assert corpus.visible_version("d1", date(2024, 1, 1)).version_id == "v2"


@pytest.mark.parametrize(
    "as_known, visible_2021", [(False, {"t2"}), (True, set())]
)
def test_undated_mentions_are_excluded_and_extraction_gates_as_known(
    as_known, visible_2021
):
    corpus = TemporalCorpus(
        {
            "versions": [version()],
            "mentions": [
                mention(observed=None),
                mention(technology="t2", recorded_at="2022-01-01"),
            ],
        },
        as_known=as_known,
    )
    assert set(corpus.view(date(2021, 1, 1)).technologies) == visible_2021
    assert set(corpus.view(date(2022, 1, 1)).technologies) == {"t2"}


@pytest.mark.parametrize("as_known", [False, True])
@pytest.mark.parametrize(
    "kind", ["relations", "maturity", "economics", "assertions"]
)
def test_events_require_visible_dated_source_versions(kind, as_known):
    corpus = TemporalCorpus(
        {
            "versions": [version(), version("v2", published="2022-01-01")],
            "mentions": [mention()],
            kind: [
                {
                    "technology_id": "t1",
                    "version_id": "v2",
                    "observed_at": "2020-01-01",
                },
                {
                    "technology_id": "t1",
                    "version_id": "missing",
                    "observed_at": "2020-01-01",
                },
                {
                    "technology_id": "t1",
                    "version_id": "v1",
                    "observed_at": None,
                },
                {
                    "technology_id": "t1",
                    "version_id": "v1",
                    "observed_at": "2020-01-01",
                    "recorded_at": "2023-01-01",
                },
            ],
        },
        as_known=as_known,
    )
    early = getattr(corpus.view(date(2021, 1, 1)).technologies["t1"], kind)
    # Extraction time gates content only in the as_known mode.
    assert [event.observed for event in early] == (
        [] if as_known else [date(2020, 1, 1)]
    )
    events = getattr(corpus.view(date(2023, 1, 1)).technologies["t1"], kind)
    assert sorted(event.observed for event in events) == sorted([
        date(2022, 1, 1),
        date(2023, 1, 1) if as_known else date(2020, 1, 1),
    ])


def test_projected_maturity_rejects_unreviewed_or_speculative_rows():
    rows = [
        {
            "technology_id": "t1",
            "version_id": "v1",
            "observed_at": "2020-01-01",
            "stage_rank": 5,
            **fields,
        }
        for fields in (
            {"status": "candidate"},
            {"modality": "planned"},
            {"polarity": "negated"},
            {"verification_status": "unverified"},
            {
                "status": "accepted",
                "verification_status": "supported",
                "modality": "observed",
                "polarity": "affirmed",
            },
        )
    ]
    corpus = TemporalCorpus(
        {"versions": [version()], "mentions": [mention()], "maturity": rows}
    )
    assert len(corpus.view(date(2021, 1, 1)).technologies["t1"].maturity) == 1


def test_metric_observation_gates_both_counters_and_yearly_histories():
    corpus = TemporalCorpus(
        {
            "versions": [
                version(
                    document_type="article",
                    metrics_json={"citation_count": 99},
                    metrics_observed_at="2022-01-01",
                    metadata_json={
                        "counts_by_year": [
                            {"year": 2019, "cited_by_count": 99}
                        ]
                    },
                )
            ],
            "mentions": [mention()],
        }
    )
    early = corpus.view(date(2021, 1, 1)).technologies["t1"].documents[0]
    assert early.metrics is None
    assert "citations_by_year" not in early.history
    late = corpus.view(date(2022, 1, 1)).technologies["t1"].documents[0]
    assert late.metrics == {"citation_count": 99}
    assert late.history["citations_by_year"] == {2019: 99}


def test_coverage_requires_a_completed_relevant_exhaustive_observation():
    corpus = TemporalCorpus(
        {
            "technologies": [
                {"technology_id": "t1", "technology": "Technology One"}
            ],
            "crawls": [
                crawl(**fields)
                for fields in (
                    {"finished_at": "2023-01-01"},
                    {"query": "Another technology"},
                    {"status": "running"},
                    {"failures": 1},
                    {"finished_at": None},
                    {"retrieved_at": "2023-01-01"},
                )
            ],
        }
    )
    assert corpus.covered_families(date(2022, 1, 1), "t1") == set()
    corpus = TemporalCorpus(
        {
            "technologies": [
                {"technology_id": "t1", "technology": "Technology One"}
            ],
            "crawls": [crawl(query='"TECHNOLOGY ONE"')],
        }
    )
    assert corpus.covered_families(date(2022, 1, 1), "t1") == {"patent"}
    assert corpus.covered_families(date(2022, 1, 1), "t2") == set()
    assert corpus.covered_families(date(2022, 1, 1)) == set()


def test_completed_sample_is_observed_but_cannot_prove_outcome_absence():
    corpus = TemporalCorpus({"crawls": [crawl(query=None, exhaustive=False)]})
    assert corpus.covered_families(date(2022, 1, 1), "t1") == {"patent"}
    assert (
        corpus.covered_families(
            date(2022, 1, 1), "t1", date(2020, 1, 1), date(2021, 1, 1)
        )
        == set()
    )


def test_a_search_completed_before_the_horizon_cannot_prove_future_absence():
    corpus = TemporalCorpus(
        {"crawls": [crawl(query=None, period_end="2023-01-01")]}
    )
    assert (
        corpus.covered_families(
            date(2023, 1, 1), "t1", date(2020, 1, 1), date(2023, 1, 1)
        )
        == set()
    )


def test_outcome_absence_requires_full_interval_coverage_not_documents():
    corpus = TemporalCorpus(
        {
            "versions": [version(document_type="patent")],
            "mentions": [mention()],
            "crawls": [
                crawl(
                    query=None,
                    period_start="2020-01-01",
                    period_end="2021-01-01",
                )
            ],
        }
    )
    assert corpus.covered_families(date(2022, 1, 1), "t1") == {"patent"}
    assert (
        corpus.covered_families(
            date(2022, 1, 1), "t1", date(2020, 1, 1), date(2022, 1, 1)
        )
        == set()
    )
    assert (
        corpus.covered_families(
            date(2022, 1, 1), "t1", date(2019, 1, 1), date(2021, 1, 1)
        )
        == set()
    )
    assert corpus.covered_families(
        date(2022, 1, 1), "t1", date(2020, 1, 1), date(2021, 1, 1)
    ) == {"patent"}


def test_document_presence_is_scoped_to_the_technology():
    corpus = TemporalCorpus({"versions": [version()], "mentions": [mention()]})
    assert corpus.covered_families(date(2021, 1, 1), "t1") == {"code"}
    assert corpus.covered_families(date(2021, 1, 1), "t2") == set()


def test_current_embeddings_need_their_own_timestamp():
    corpus = TemporalCorpus(
        {
            "technologies": [
                {"technology_id": "undated", "embedding": [1, 0]},
                {
                    "technology_id": "later",
                    "embedding": [0, 1],
                    "embedding_observed_at": "2022-01-01",
                },
                {
                    "technology_id": "early",
                    "embedding": [1, 1],
                    "embedding_observed_at": "2020-01-01",
                },
            ]
        }
    )
    assert corpus.embeddings_at(date(2021, 1, 1)) == {"early": [1, 1]}


@pytest.mark.parametrize("as_known", [False, True])
def test_extraction_completion_and_latest_observation_have_separate_dates(
    as_known,
):
    corpus = TemporalCorpus(
        {
            "versions": [
                version(
                    extracted=True,
                    extracted_at="2022-01-01",
                    metrics_observed_at="2023-01-01",
                )
            ],
            "crawls": [crawl(finished_at="2024-01-01")],
        },
        as_known=as_known,
    )
    # Extraction describes published content, so only as_known waits for it.
    early = corpus.visible_version("d1", date(2021, 1, 1))
    assert early.extracted is not as_known
    assert corpus.visible_version("d1", date(2022, 1, 1)).extracted is True
    assert corpus.latest_date == date(2024, 1, 1)


def test_datetime_input_and_invalid_history_dates_are_normalized():
    assert parse_date(datetime(2020, 1, 1, 12)) == date(2020, 1, 1)
    corpus = TemporalCorpus(
        {
            "versions": [
                version(
                    metadata_json={
                        "release_dates": [
                            "",
                            "invalid",
                            "2020-02-01",
                            "2024-01-01",
                        ],
                        "commit_weeks": {
                            "": 10,
                            "2020-02-01": 2,
                            "2024-01-01": 99,
                        },
                        "priority_date": "2024-01-01",
                    }
                )
            ],
            "mentions": [mention()],
        }
    )
    history = (
        corpus.view(date(2021, 1, 1)).technologies["t1"].documents[0].history
    )
    assert history == {
        "release_dates": ["2020-02-01"],
        "commit_weeks": {"2020-02-01": 2},
    }
