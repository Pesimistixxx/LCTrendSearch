from copy import deepcopy

import pytest

from lctrend.core.catalog_validation import CATALOG_NAMES, validate_catalogs
from lctrend.core.config import load_catalog


@pytest.fixture
def catalogs():
    return {name: deepcopy(load_catalog(name)) for name in CATALOG_NAMES}


def test_bundled_resources_are_consistent():
    validate_catalogs()


@pytest.mark.parametrize(
    "catalog,path,value,expected",
    [
        ("sources", ("http", "max_attempts"), 0, "sources.http.max_attempts"),
        ("sources", ("markdown", "overlap_chars"), 1000, "overlap_chars"),
        ("extraction", ("economics", "money_pattern"), "[", "money_pattern"),
        (
            "extraction",
            ("assertions", "subject_kinds"),
            ["Typo"],
            "subject_kinds",
        ),
        ("pipeline", ("settings", "primary_chunks"), 99, "pipeline.settings"),
        ("pipeline", ("graph_context", "max_chars"), True, "max_chars"),
        (
            "resolver",
            ("semantic", "embedding_provider"),
            "missing",
            "embedding_provider",
        ),
        ("dataset", ("snapshots", "month"), 13, "snapshots.month"),
        (
            "dataset",
            ("commercial_economic_categories",),
            ["invented"],
            "commercial_economic_categories",
        ),
        ("taxonomy", ("branching",), 1, "taxonomy.branching"),
        ("ranking", ("top_k",), 0, "ranking.top_k"),
        (
            "ranking",
            ("candidates", "kinds"),
            ["Typo"],
            "ranking.candidates.kinds",
        ),
        (
            "ranking",
            ("score", "weights", "mention_growth_12m"),
            "high",
            "ranking.score.weights.mention_growth_12m",
        ),
        (
            "ranking",
            ("score", "labels"),
            {},
            "ranking.score.labels",
        ),
        ("llm", ("model_ladder",), ["same", "same"], "model_ladder"),
        ("countries", ("iso_alpha2",), ["RU", "ZZ"], "countries.iso_alpha2"),
        (
            "llm_schema",
            ("currency_aliases", "JPY"),
            [],
            "currency_aliases.JPY",
        ),
        (
            "llm_schema",
            (
                "predicates",
                "reports_maturity_stage",
                "integer_qualifier_markers",
                "trl",
            ),
            "[",
            "integer_qualifier_markers",
        ),
    ],
)
def test_invalid_resource_fails_with_actionable_path(
    catalogs, catalog, path, value, expected
):
    target = catalogs[catalog]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=expected):
        validate_catalogs(catalogs)


def test_unknown_projection_predicate_and_role_are_rejected(catalogs):
    projection = catalogs["graph"]["projections"]["developed_by"]
    projection["target_role"] = "country"
    projection["target_kinds"] = ["Company"]
    with pytest.raises(ValueError, match="graph.projections.developed_by"):
        validate_catalogs(catalogs)


def test_cypher_projection_identifiers_cannot_be_injected(catalogs):
    catalogs["graph"]["projections"]["solves_task"]["relationship"] = (
        "X) DELETE n"
    )
    with pytest.raises(ValueError, match="invalid Cypher identifier"):
        validate_catalogs(catalogs)


def test_maturity_stage_contract_and_rank_map_cannot_drift(catalogs):
    catalogs["graph"]["maturity_stage_rank"]["invented_stage"] = 7
    with pytest.raises(ValueError, match="maturity_stage_rank"):
        validate_catalogs(catalogs)


def test_invalid_calendar_date_and_overlapping_modalities(catalogs):
    catalogs["dataset"]["snapshots"].update(month=2, day=31)
    with pytest.raises(ValueError, match="dataset.snapshots"):
        validate_catalogs(catalogs)
    catalogs["dataset"]["snapshots"].update(month=1, day=1)
    catalogs["dataset"]["speculative_modalities"].append("reported")
    with pytest.raises(ValueError, match="overlaps speculative"):
        validate_catalogs(catalogs)


def test_complete_override_must_contain_required_fields(catalogs):
    catalogs["sources"] = {"domains": []}
    with pytest.raises(ValueError, match="sources.domains"):
        validate_catalogs(catalogs)


def test_negative_labels_require_only_families_with_a_source(catalogs):
    label = catalogs["dataset"]["label"]
    families = {
        platform["source_family"]
        for platform in catalogs["sources"]["platforms"].values()
    }
    assert set(label["required_negative_families"]) <= families
    label["required_negative_families"].append("commercial")
    with pytest.raises(
        ValueError, match="required_negative_families.*no source"
    ):
        validate_catalogs(catalogs)
