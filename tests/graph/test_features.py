from datetime import date

import pytest

from lctrend.graph.features import (
    activity_features,
    classify_economic,
    coverage_features,
    economic_features,
    log_growth,
    months_before,
)
from lctrend.graph.temporal import Event, TechnologyView


@pytest.mark.parametrize(
    "status", ["candidate", "needs_review", "rejected", None]
)
def test_reported_unconfirmed_economics_remain_candidates(status):
    assert (
        classify_economic(
            {
                "status": status,
                "polarity": "affirmed",
                "modality": "reported",
                "verification_status": "supported",
            },
            ["reported", "observed"],
        )
        == "candidate"
    )


@pytest.mark.parametrize("modality", ["planned", "hypothetical"])
def test_reviewed_forecasts_are_never_confirmed_economics(modality):
    assert (
        classify_economic(
            {
                "status": "accepted",
                "polarity": "affirmed",
                "modality": modality,
            },
            ["reported", "observed"],
        )
        == "forecast"
    )


def test_only_factual_affirmative_confirmed_economics_contribute_to_amounts():
    fields = [
        {"status": "candidate", "modality": "reported"},
        {"status": "accepted", "modality": "planned"},
        {"status": "accepted", "modality": "hypothetical"},
        {"status": "accepted", "modality": "observed", "polarity": "negated"},
        {"status": "accepted", "modality": "observed", "amount_value": 7},
    ]
    view = TechnologyView("t1", "Technology")
    view.economics = [
        Event(
            date(2020, 1, 1),
            {
                "polarity": "affirmed",
                "category": "investment",
                "currency": "USD",
                "amount_value": 1000,
                **extra,
            },
        )
        for extra in fields
    ]
    row = economic_features(
        view, date(2021, 1, 1), ["reported", "observed"], 3
    )
    assert row["economic_evidence_count"] == 1
    assert row["economic_candidate_count"] == 1
    assert row["economic_forecast_count"] == 2
    assert row["economic_planned_count"] == 1
    assert row["economic_hypothetical_count"] == 1
    assert row["economic_negated_count"] == 1
    # 7 USD of 2020 in dollars of the base year (money.json).
    assert row["funding_amount_usd_real"] == pytest.approx(
        7 * 321.962 / 258.856, rel=1e-3
    )


def test_supported_performance_forecasts_do_not_count_as_realized_gains():
    view = TechnologyView("t1", "Technology")
    view.assertions = [
        Event(
            date(2020, 1, 1),
            {
                "predicate": "changes_metric",
                "verification_status": "supported",
                "status": "accepted",
                "polarity": "affirmed",
                "modality": modality,
            },
        )
        for modality in ("reported", "planned", "hypothetical")
    ]
    row = economic_features(
        view, date(2021, 1, 1), ["reported", "observed"], 3
    )
    assert row["performance_gain_evidence"] == 1
    assert row["funding_amount_usd_real"] is None
    assert row["economic_signal_recency_days"] is None


def test_uncollected_source_count_stays_missing_while_searched_zero_is_zero():
    row = coverage_features(
        TechnologyView("t1", "Technology"),
        {"scholarly"},
        {"scholarly": "article", "patent": "patent"},
    )
    assert row["article_count"] == 0
    assert row["article_count_missing"] is False
    assert row["patent_count"] is None
    assert row["patent_count_missing"] is True


def test_calendar_windows_and_growth_support_leap_days_and_new_signals():
    assert months_before(date(2020, 3, 31), 1) == date(2020, 2, 29)
    assert months_before(date(2020, 2, 29), 12) == date(2019, 2, 28)
    assert log_growth(1, 0) > 0
    assert log_growth(1, 0) == -log_growth(0, 1)
    assert (
        activity_features(
            TechnologyView("t1", "Technology"), date(2021, 1, 1)
        )["citation_count"]
        is None
    )
