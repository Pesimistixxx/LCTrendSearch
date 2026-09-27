from datetime import date

import pytest

from lctrend.taxonomy import TaxonomyConcept, build_taxonomy


def concept(identifier, vector=(1.0, 0.0), model="model-a", docs=None):
    evidence = docs or [("doc-1", date(2025, 6, 1))]
    return TaxonomyConcept(
        identifier,
        identifier,
        "Technology",
        vector,
        date(2024, 1, 1),
        [day for _, day in evidence],
        evidence,
        model,
    )


def test_shared_documents_count_once_and_same_day_documents_remain_distinct():
    docs = [("doc-1", date(2025, 6, 1)), ("doc-2", date(2025, 6, 1))]
    taxonomy = build_taxonomy(
        [concept("a", docs=docs), concept("b", docs=docs)], "2026-01-01"
    )
    assert taxonomy.root().documents_last_year == 2


def test_input_vectors_and_reviewed_parent_edges_change_taxonomy_version():
    baseline = build_taxonomy([concept("a"), concept("b")], "2026-01-01")
    changed_vector = build_taxonomy(
        [concept("a", (0.9, 0.1)), concept("b")], "2026-01-01"
    )
    changed_parent = build_taxonomy(
        [concept("a"), concept("b")], "2026-01-01", [("a", "b")]
    )
    changed_docs = build_taxonomy(
        [concept("a", docs=[("other", date(2025, 6, 1))]), concept("b")],
        "2026-01-01",
    )
    assert (
        len(
            {
                item.version
                for item in [
                    baseline,
                    changed_vector,
                    changed_parent,
                    changed_docs,
                ]
            }
        )
        == 4
    )


@pytest.mark.parametrize(
    "bad", [(0.0, 0.0), (float("nan"), 0.1), (), (1.0, 0.0, 0.0)]
)
def test_invalid_or_incompatible_embeddings_fail_explicitly(bad):
    with pytest.raises(ValueError, match="embedding"):
        build_taxonomy([concept("a"), concept("b", bad)], "2026-01-01")


def test_different_models_cannot_share_the_same_taxonomy():
    with pytest.raises(ValueError, match="model"):
        build_taxonomy(
            [concept("a"), concept("b", model="model-b")], "2026-01-01"
        )


def test_embedding_observed_after_snapshot_is_excluded():
    old = concept("a")
    old.embedding_observed_at = date(2026, 2, 1)
    taxonomy = build_taxonomy([old], "2026-01-01")
    assert not taxonomy.concepts
