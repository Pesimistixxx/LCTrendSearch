"""F05: agreement and conflict of independent sources on one claim."""

from datetime import date

from lctrend.graph.features import claim_slot_features
from lctrend.graph.temporal import Event, TechnologyView

FACTUAL = ["reported", "observed"]


def event(version, key, polarity="affirmed", **data):
    """A claim of slot ``key``: developed by the organization ``key``."""
    return Event(
        date(2022, 1, 1),
        {
            "version_id": version,
            "predicate": "developed_by",
            "roles": [["SUBJECT", "t1"], ["ORGANIZATION", key]] if key else [],
            "qualifiers_json": "{}",
            "polarity": polarity,
            "status": "accepted",
            "modality": "reported",
            **data,
        },
    )


def features(*events):
    view = TechnologyView("t1", "Technology")
    view.assertions = list(events)
    return claim_slot_features(view, FACTUAL)


def test_two_supporting_and_two_refuting_sources_conflict_fully():
    row = features(
        event("v1", "s1"),
        event("v2", "s1"),
        event("v3", "s1", "negated"),
        event("v4", "s1", "negated"),
        # A second slot every source agrees on.
        event("v1", "s2"),
        event("v5", "s2"),
    )

    assert row["comparable_claim_slot_count"] == 2
    assert row["corroborated_claim_slot_count"] == 2
    assert row["independent_claim_conflict_strength"] == 1.0
    assert row["independent_refutation_share"] == 2 / 6


def test_all_refuting_sources_agree_but_refute():
    row = features(event("v1", "s1", "negated"), event("v2", "s1", "negated"))

    assert row["independent_claim_conflict_strength"] == 0.0
    assert row["independent_refutation_share"] == 1.0


def test_mixed_sources_plans_and_unreviewed_claims_do_not_count():
    row = features(
        # One source both affirms and negates: mixed, neither P nor N.
        event("v1", "s1"),
        event("v1", "s1", "negated"),
        event("v2", "s1"),
        event("v3", "s1", "negated", modality="planned"),
        event("v4", "s1", "negated", status="needs_review"),
        event("v5", None),
    )

    assert row["mixed_origin_group_count"] == 1
    # One unambiguous source is not a comparison: unknown, not zero.
    assert row["comparable_claim_slot_count"] == 0
    assert row["independent_claim_conflict_strength"] is None
    assert row["independent_refutation_share"] is None
