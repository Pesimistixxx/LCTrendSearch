from datetime import date

from lctrend.graph.subgraphs import sample_neighborhood
from lctrend.graph.temporal import TemporalCorpus


def _version(identifier, year, **parties):
    return {
        "document_id": f"d-{identifier}",
        "version_id": identifier,
        "document_type": "article",
        "source_family": "scholarly",
        "source_id": "openalex",
        "document_published_at": f"{year}-01-01",
        "version_published_at": f"{year}-01-01",
        "retrieved_at": f"{year}-01-01",
        "extracted": True,
        "extracted_at": f"{year}-01-01",
        "coverage": "full_text",
        **parties,
    }


def _mention(technology, version, year):
    return {
        "technology_id": technology,
        "version_id": version,
        "observed_at": f"{year}-01-01",
        "mentions": 1,
        "accepted": 1,
    }


def _corpus():
    """r: co-mentioned with c in v1; parent p with its own paper v3 that
    also mentions x; f shares only a country and a company with r; the
    survey v4 mentions r, s1 and s2.
    """
    hubs = {
        "countries": ["US"],
        "companies": ["Acme"],
        "contributors": ["Ann"],
    }
    return TemporalCorpus(
        {
            "versions": [
                _version("v1", 2020, **hubs),
                _version("v2", 2019, **hubs),
                _version("v3", 2019),
                _version("v4", 2018),
            ],
            "technologies": [
                {"technology_id": key, "technology": key.upper()}
                for key in ("r", "c", "p", "x", "f", "s1", "s2")
            ],
            "mentions": [
                _mention("r", "v1", 2020),
                _mention("c", "v1", 2020),
                _mention("f", "v2", 2019),
                _mention("p", "v3", 2019),
                _mention("x", "v3", 2019),
                _mention("r", "v4", 2018),
                _mention("s1", "v4", 2018),
                _mention("s2", "v4", 2018),
            ],
            "relations": [
                {
                    "technology_id": "r",
                    "version_id": "v1",
                    "relation": "SUBTECHNOLOGY_OF",
                    "target_id": "p",
                    "target_kind": "Technology",
                    "observed_at": "2020-01-01",
                }
            ],
        }
    )


CONFIG = {"max_document_technologies": 3}


def _types(sample):
    return {node["id"]: node["type"] for node in sample["nodes"]}


def test_two_hops_through_technologies_and_documents():
    sample = sample_neighborhood(
        _corpus().view(date(2021, 1, 1)), "r", config=CONFIG
    )
    nodes = _types(sample)
    # Hop 1: its document, its parent; hop 2: the co-mentioned technology
    # and the parent's paper.
    for expected in (
        "Technology:c",
        "Technology:p",
        "DocumentVersion:v1",
        "DocumentVersion:v3",
    ):
        assert expected in nodes
    # x is three steps away (r -> p -> v3 -> x).
    assert "Technology:x" not in nodes


def test_hubs_are_dead_ends_and_authors_stay_out():
    nodes = _types(
        sample_neighborhood(
            _corpus().view(date(2021, 1, 1)), "r", config=CONFIG
        )
    )
    assert "Country" in nodes.values() and "Company" in nodes.values()
    # f is reachable only through the shared country or company.
    assert "Technology:f" not in nodes
    assert "DocumentVersion:v2" not in nodes
    assert not {"Author", "Source", "Document"} & set(nodes.values())


def test_a_survey_is_kept_but_not_crossed():
    nodes = _types(
        sample_neighborhood(
            _corpus().view(date(2021, 1, 1)), "r", config=CONFIG
        )
    )
    assert "DocumentVersion:v4" in nodes
    assert "Technology:s1" not in nodes and "Technology:s2" not in nodes
    # With a higher survey threshold the same paper is crossed.
    wider = _types(
        sample_neighborhood(
            _corpus().view(date(2021, 1, 1)),
            "r",
            config={"max_document_technologies": 10},
        )
    )
    assert "Technology:s1" in wider


def test_only_documents_visible_at_the_snapshot():
    nodes = _types(
        sample_neighborhood(
            _corpus().view(date(2019, 6, 1)), "r", config=CONFIG
        )
    )
    assert "DocumentVersion:v1" not in nodes
    assert "Technology:c" not in nodes


def test_node_cap_and_stable_order():
    view = _corpus().view(date(2021, 1, 1))
    capped = sample_neighborhood(view, "r", config={**CONFIG, "max_nodes": 3})
    assert len(capped["nodes"]) == 3
    assert sample_neighborhood(view, "r", config=CONFIG) == (
        sample_neighborhood(view, "r", config=CONFIG)
    )
