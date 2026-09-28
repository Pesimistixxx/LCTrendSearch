"""Concept identity survives the graph round trip (C-4, C-5)."""

import asyncio
import json

import pytest

from lctrend.core.models import (
    Chunk,
    Concept,
    ConceptKind,
    ExtractionResult,
    Mention,
    ProcessingRun,
    ResolutionDecision,
)
from lctrend.graph.store import GraphStore, _concept_from_properties
from lctrend.ingest.adapters import parse_openalex


class Result(list):
    def consume(self):
        return None

    def single(self):
        return {"count": 0, "published": 0}


class Transaction:
    """Answers the stored-label probe from ``stored``: concept_id → labels."""

    def __init__(self, stored=None):
        self.stored = stored or {}
        self.queries = []

    def run(self, query, **parameters):
        self.queries.append((query, parameters))
        if "RETURN id AS concept_id" in query:
            return Result(
                {
                    "concept_id": concept_id,
                    **{
                        label: label in self.stored.get(concept_id, ())
                        for label in ("Technology", "Method", "Material")
                    },
                }
                for concept_id in parameters["ids"]
            )
        return Result()


def extraction(kind):
    document = parse_openalex(
        {"id": "https://openalex.org/W7", "title": "Graphene"}
    )
    document.chunks = [
        Chunk(chunk_id="c1", kind="abstract", text="graphene", order=0)
    ]
    concept = Concept(
        concept_id="concept:graphene",
        kind=kind,
        preferred_label="graphene",
        identity_key="graphene",
        label_counts={"graphene": 3, "Graphene": 1},
    )
    result = ExtractionResult(
        document_version_id=document.document_version_id,
        run=ProcessingRun(
            run_id="run:1",
            parser="llm_packets",
            config_hash="config",
            started_at="2026-09-28T00:00:00Z",
        ),
        concepts=[concept],
        mentions=[
            Mention(
                mention_id="m1",
                chunk_id="c1",
                surface_text="graphene",
                start=0,
                end=8,
                type_candidates=[kind],
            )
        ],
        resolutions=[
            ResolutionDecision(
                resolution_id="r1",
                mention_id="m1",
                status="provisional",
                concept_id="concept:graphene",
            )
        ],
    )
    return document, result


def write(kind, stored):
    document, result = extraction(kind)
    tx = Transaction(stored)
    asyncio.run(GraphStore._write_extraction(tx, document, result))
    return tx.queries


def test_identity_key_and_form_counts_are_stored_and_read_back():
    queries = write(ConceptKind.MATERIAL, {})
    _, parameters = next(
        item for item in queries if "c.preferred_label" in item[0]
    )
    row = parameters["rows"][0]
    assert row["identity_key"] == "graphene"
    assert json.loads(row["label_counts_json"]) == {
        "graphene": 3,
        "Graphene": 1,
    }
    loaded = _concept_from_properties(
        {
            "concept_id": "concept:graphene",
            "kind": "Material",
            "preferred_label": "graphene",
            "identity_key": "graphene",
            "label_counts_json": row["label_counts_json"],
        }
    )
    assert loaded.identity_key == "graphene"
    assert loaded.label_counts == {"graphene": 3, "Graphene": 1}


def test_unreviewed_technology_cannot_relabel_a_stored_material():
    with pytest.raises(ValueError, match="reviewed definition"):
        write(ConceptKind.TECHNOLOGY, {"concept:graphene": ["Material"]})


def test_a_lower_family_kind_does_not_inherit_a_stored_technology_type():
    with pytest.raises(ValueError, match="kind conflict"):
        write(ConceptKind.MATERIAL, {"concept:graphene": ["Technology"]})


def ambiguous_extraction():
    document, result = extraction(ConceptKind.TECHNOLOGY)
    result.concepts = []
    result.resolutions = [
        ResolutionDecision(
            resolution_id="r1",
            mention_id="m1",
            status="ambiguous",
            method="deterministic_alias_collision",
            candidates=[
                {"concept_id": "concept:a", "kind": "Technology", "score": 1},
                {"concept_id": "concept:b", "kind": "Method", "score": 1},
            ],
        )
    ]
    return document, result


def test_an_ambiguous_mention_links_every_candidate():
    document, result = ambiguous_extraction()
    tx = Transaction({"concept:b": ["Technology"]})
    asyncio.run(GraphStore._write_extraction(tx, document, result))
    rows = {
        (row["concept_id"], row["resolution_status"], label)
        for query, parameters in tx.queries
        if "MERGE (chunk)-[r:MENTIONS" in query
        for label in ("Technology", "Method")
        if f"concept:{label} {{" in query
        for row in parameters["rows"]
    }
    # concept:b is stored as a Technology already: the link follows it.
    assert rows == {
        ("concept:a", "ambiguous", "Technology"),
        ("concept:b", "ambiguous", "Technology"),
    }


class ReadSession:
    def __init__(self, queries):
        self.queries = queries

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def run(self, query, **parameters):
        self.queries.append(" ".join(query.split()))
        if "db.labels" in query:
            return [{"label": "Technology"}]
        return []


def reader(queries):
    store = GraphStore.__new__(GraphStore)
    store._driver = type(
        "Driver", (), {"session": lambda self, **_: ReadSession(queries)}
    )()
    store._database = "neo4j"
    return store


def test_ambiguous_links_are_not_counted_as_mentions():
    queries = []
    asyncio.run(reader(queries).read_temporal_data())
    mentions = next(q for q in queries if "[m:MENTIONS]" in q)
    assert "m.resolution_status, '') <> 'ambiguous'" in mentions
    queries.clear()
    asyncio.run(reader(queries).read_taxonomy_input(["Technology"]))
    taxonomy = next(q for q in queries if "MENTIONS" in q)
    assert "m.resolution_status, '') <> 'ambiguous'" in taxonomy


def test_an_unchanged_vector_keeps_its_observation_date():
    document, result = extraction(ConceptKind.MATERIAL)
    result.concept_embeddings = {"concept:graphene": [0.6, 0.8]}
    result.embedding_model = "EmbeddingsGigaR"
    tx = Transaction()
    asyncio.run(GraphStore._write_extraction(tx, document, result))
    query, parameters = next(
        (query, parameters)
        for query, parameters in tx.queries
        if "c.embedding_observed_at" in query
    )
    query = " ".join(query.split())
    # Re-processing with the same vector and model must not move the date
    # a past snapshot saw (N-2).
    assert (
        "c.embedding = row.vector AND c.embedding_model = $model "
        "AND c.embedding_observed_at IS NOT NULL AS unchanged"
    ) in query
    assert (
        "c.embedding_observed_at = CASE WHEN unchanged "
        "THEN c.embedding_observed_at ELSE $recorded_at END"
    ) in query
    assert query.index("AS unchanged") < query.index("SET")
    assert parameters["model"] == "EmbeddingsGigaR"


class VotingTransaction(Transaction):
    """A stored node with kind votes another job already wrote."""

    def __init__(self, stored, votes):
        super().__init__(stored)
        self.votes = votes

    def run(self, query, **parameters):
        if "RETURN id AS concept_id" in query:
            self.queries.append((query, parameters))
            return Result(
                {
                    "concept_id": concept_id,
                    **{
                        label: label in self.stored.get(concept_id, ())
                        for label in ("Technology", "Method", "Material")
                    },
                    "kind_counts_json": json.dumps(self.votes),
                }
                for concept_id in parameters["ids"]
            )
        return super().run(query, **parameters)


def write_voted(kind, own_votes, stored_votes):
    document, result = extraction(kind)
    result.concepts[0].kind_counts = own_votes
    tx = VotingTransaction({"concept:graphene": ["Technology"]}, stored_votes)
    asyncio.run(GraphStore._write_extraction(tx, document, result))
    return tx.queries


@pytest.mark.parametrize(
    "stored_votes", [{"Technology": 1}, {"Technology": 3}]
)
def test_votes_cannot_retype_a_stored_technical_concept(stored_votes):
    with pytest.raises(ValueError, match="Stored concept kind conflict"):
        write_voted(ConceptKind.MATERIAL, {"Material": 3}, stored_votes)


def test_a_definition_is_written_without_erasing_a_stored_one():
    document, result = extraction(ConceptKind.MATERIAL)
    result.concepts[0].definition = "two-dimensional carbon"
    tx = Transaction({})
    asyncio.run(GraphStore._write_extraction(tx, document, result))
    query, parameters = next(
        item for item in tx.queries if "c.preferred_label" in item[0]
    )
    assert "coalesce(row.definition" in query
    assert parameters["rows"][0]["definition"] == "two-dimensional carbon"


def test_a_declared_alias_candidate_is_stored_for_review():
    document, result = extraction(ConceptKind.MATERIAL)
    result.resolutions[0].candidates = [
        {
            "concept_id": "concept:other",
            "kind": "Material",
            "score": 1.0,
            "method": "declared_alias",
            "alias": "G",
        }
    ]
    tx = Transaction({})
    asyncio.run(GraphStore._write_extraction(tx, document, result))
    query, parameters = next(
        item for item in tx.queries if "POSSIBLY_SAME_AS" in item[0]
    )
    assert "coalesce(r.review_status, 'pending')" in query
    (row,) = parameters["rows"]
    assert (row["source"], row["target"], row["method"], row["alias"]) == (
        "concept:graphene",
        "concept:other",
        "declared_alias",
        "G",
    )


def test_a_reviewed_kind_is_not_re_voted():
    document, result = extraction(ConceptKind.MATERIAL)
    result.concepts[0].status = "accepted"
    result.concepts[0].kind_counts = {"Material": 1}
    tx = VotingTransaction(
        {"concept:graphene": ["Material"]}, {"Technology": 9}
    )
    asyncio.run(GraphStore._write_extraction(tx, document, result))
    assert not [q for q, _ in tx.queries if "REMOVE c:" in q]
