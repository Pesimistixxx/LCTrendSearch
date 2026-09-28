"""Merging duplicate concepts (C-3): MERGED_INTO with MENTIONS moved."""

import asyncio
import json
import sys

import pytest

from lctrend import cli
from lctrend.core.models import Concept, ConceptKind, ConceptName, Mention
from lctrend.extraction.resolver import resolve_mentions
from lctrend.graph.merge import merged_concept
from lctrend.graph.store import GraphStore

T = ConceptKind.TECHNOLOGY


def concept(concept_id, label, kind=T, **fields):
    return Concept(
        concept_id=concept_id,
        kind=kind,
        preferred_label=label,
        status="provisional",
        names=[
            ConceptName(
                name_id=f"name:{concept_id}",
                text=label,
                normalized_text=label.casefold(),
                name_kind="observed",
                status="provisional",
            )
        ],
        **fields,
    )


SOURCE = concept(
    "concept:b",
    "Квантового отжига",
    label_counts={"Квантового отжига": 2},
)
TARGET = concept(
    "concept:a",
    "квантовый отжиг",
    kind=T,
    identity_key="квантов отжиг",
    label_counts={"квантовый отжиг": 1},
)


def test_merged_concept_adopts_names_and_counts_within_one_kind():
    merged = merged_concept(TARGET, SOURCE)
    assert merged.concept_id == "concept:a"
    assert merged.kind == T
    assert merged.identity_key == "квантов отжиг"
    assert merged.label_counts == {
        "квантовый отжиг": 1,
        "Квантового отжига": 2,
    }
    assert merged.preferred_label == "Квантового отжига"
    # A merge is a review decision: the source's names become accepted
    # names of the target, so the next document resolves them to it.
    accepted = {
        name.text for name in merged.names if name.status == "accepted"
    }
    assert accepted == {"Квантового отжига"}


def test_a_merged_source_name_resolves_to_the_target():
    mention = Mention(
        mention_id="m1",
        chunk_id="c1",
        surface_text="Квантового отжига",
        canonical_text="Квантового отжига",
        start=0,
        end=17,
        type_candidates=[T],
    )
    _, decisions = resolve_mentions(
        [mention], [merged_concept(TARGET, SOURCE)]
    )
    assert decisions[0].status == "accepted"
    assert decisions[0].concept_id == "concept:a"


class Result(list):
    def consume(self):
        return None

    def single(self):
        return self[0] if self else None


class Session:
    def __init__(self, stored, queries):
        self.stored = stored
        self.queries = queries

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def run(self, query, **parameters):
        self.queries.append((" ".join(query.split()), parameters))
        if "RETURN properties(c) AS properties" in query:
            return Result(
                {"properties": self.stored[item]}
                for item in parameters.get("ids", ())
                if item in self.stored
            )
        return Result()

    async def execute_write(self, callback, *args):
        return await callback(self, *args)


def properties(item):
    return {
        **item.model_dump(mode="json", exclude={"names", "label_counts"}),
        "names_json": json.dumps(
            [name.model_dump(mode="json") for name in item.names]
        ),
        "label_counts_json": json.dumps(item.label_counts),
    }


def store_with(*concepts):
    queries = []
    stored = {item.concept_id: properties(item) for item in concepts}
    store = GraphStore.__new__(GraphStore)
    store._database = "neo4j"
    store._driver = type(
        "Driver",
        (),
        {"session": lambda self, **_: Session(stored, queries)},
    )()
    return store, queries


def test_merge_moves_mentions_and_links_source_into_target():
    store, queries = store_with(SOURCE, TARGET)
    summary = asyncio.run(
        store.merge_concepts("concept:b", "concept:a", reason="duplicate")
    )
    assert summary["target_kind"] == "Technology"
    text = [query for query, _ in queries]
    assert not any("REMOVE c:" in q for q in text)
    move = next(q for q in text if "moved:MENTIONS" in q)
    assert "(s:Technology {concept_id: $source})<-[r:MENTIONS]-" in move
    assert "(t:Technology {concept_id: $target})" in move
    assert "DELETE r" in move
    assert any("[:SUBJECT]->(t)" in q for q in text)
    assert any("moved:DEVELOPED_BY" in q for q in text)
    assert any("moved:HAS_MATURITY_EVIDENCE" in q for q in text)
    link = next(q for q in text if "MERGED_INTO" in q)
    assert "s.status = 'merged'" in link
    parameters = next(p for q, p in queries if "MERGED_INTO" in q)
    assert parameters["reason"] == "duplicate"
    update = next(p for q, p in queries if "t.names_json" in q)
    assert "Квантового отжига" in update["aliases"]


@pytest.mark.parametrize(
    "source,target,message",
    [
        ("concept:a", "concept:a", "itself"),
        ("concept:x", "concept:a", "not found"),
        ("concept:c", "concept:a", "family"),
    ],
)
def test_merge_refuses_self_missing_and_other_family(source, target, message):
    country = concept("concept:c", "DE", kind=ConceptKind.COUNTRY)
    store, queries = store_with(SOURCE, TARGET, country)
    with pytest.raises(ValueError, match=message):
        asyncio.run(store.merge_concepts(source, target))
    assert not [q for q, _ in queries if "MERGED_INTO" in q]


def test_merged_concepts_leave_the_registry_and_the_features():
    store, queries = store_with()
    asyncio.run(store.read_concepts())
    assert all(
        "coalesce(c.status, '') <> 'merged'" in query for query, _ in queries
    )


def test_cli_merge_command_calls_the_store(monkeypatch):
    calls = []

    class Store:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def merge_concepts(self, source, target, reason=None):
            calls.append((source, target, reason))
            return {"source": source, "target": target}

    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr(cli, "_store", Store)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lctrend",
            "merge-concepts",
            "concept:b",
            "concept:a",
            "--reason",
            "duplicate",
        ],
    )
    cli.main()
    assert calls == [("concept:b", "concept:a", "duplicate")]


def test_a_merge_sums_kind_votes_and_keeps_a_definition():
    target = concept(
        "concept:a",
        "ML-236B",
        kind=ConceptKind.MATERIAL,
        kind_counts={"Material": 1},
    )
    source = concept(
        "concept:b",
        "compactin",
        kind=ConceptKind.MATERIAL,
        kind_counts={"Material": 4},
        definition="HMG-CoA reductase inhibitor",
    )
    merged = merged_concept(target, source)
    assert merged.kind == ConceptKind.MATERIAL
    assert merged.kind_counts == {"Material": 5}
    assert merged.definition == "HMG-CoA reductase inhibitor"


def test_set_concept_kind_relabels_within_the_family_and_accepts():
    from lctrend.graph import merge as merge_module

    queries = []

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def execute_write(self, callback):
            class Tx:
                async def run(self, query, **parameters):
                    queries.append((query, parameters))

                    class Done:
                        async def consume(self):
                            return None

                    return Done()

            return await callback(Tx())

    store = GraphStore.__new__(GraphStore)
    store._database = "neo4j"
    store._driver = type("D", (), {"session": lambda self, **_: Session()})()

    async def found(_, ids):
        return {"c:ml": (concept("c:ml", "ML-236B"), "provisional")}

    original = merge_module.read_concepts_by_id
    merge_module.read_concepts_by_id = found
    try:
        summary = asyncio.run(store.set_concept_kind("c:ml", "Material"))
        with pytest.raises(ValueError):
            asyncio.run(store.set_concept_kind("c:ml", "Company"))
        with pytest.raises(ValueError, match="reviewed definition"):
            asyncio.run(store.set_concept_kind("c:ml", "Technology"))
    finally:
        merge_module.read_concepts_by_id = original
    assert summary["to"] == "Material"
    ((query, parameters),) = queries
    assert "REMOVE c:Technology" in query and "SET c:Material" in query
    assert "c.status = 'accepted'" in query


def test_merge_cannot_promote_method_or_join_different_meanings():
    method = TARGET.model_copy(update={"kind": ConceptKind.METHOD})
    with pytest.raises(ValueError, match="Different entity kinds"):
        merged_concept(method, SOURCE)
    other = SOURCE.model_copy(update={"identity_scope": "different meaning"})
    with pytest.raises(ValueError, match="Different entity meanings"):
        merged_concept(TARGET, other)
