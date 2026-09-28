"""Key v2 migration of an existing graph: recompute keys, merge duplicates."""

import asyncio
import sys

from lctrend import cli
from lctrend.core.models import Concept, ConceptKind, stable_id
from lctrend.graph.migration import apply_key_migration, plan_key_migration

T = ConceptKind.TECHNOLOGY


def concept(concept_id, label, kind=T, status="provisional"):
    return Concept(
        concept_id=concept_id,
        kind=kind,
        preferred_label=label,
        status=status,
    )


def test_duplicates_under_key_v2_are_merged_into_one_target():
    concepts = [
        concept("concept:1", "языковые модели"),
        concept("concept:2", "языковой модели"),
        concept("concept:3", "LLM"),
        concept("concept:4", "больших языковых моделей"),
        concept("concept:5", "AI"),
        concept("concept:6", "AM"),
        concept("concept:7", "federated learning", ConceptKind.METHOD),
        concept("concept:8", "Federated learning"),
        concept("concept:9", "IS", ConceptKind.COUNTRY),
        concept("concept:10", "BE", ConceptKind.COUNTRY),
    ]
    mentions = {"concept:2": 5, "concept:1": 2, "concept:4": 3}
    plan = plan_key_migration(concepts, mention_counts=mentions)
    merges = {(item.source, item.target) for item in plan.merges}
    # Most mentions wins among equals; the LLM synonym group is one
    # concept; Method and Technology retain their types; AI, AM and the
    # countries stay apart.
    assert merges == {
        ("concept:1", "concept:2"),
        ("concept:3", "concept:4"),
    }
    keys = {item.concept_id: item.identity_key for item in plan.updates}
    assert keys["concept:9"] == "iso:IS"
    assert keys["concept:10"] == "iso:BE"
    assert keys["concept:5"] != keys["concept:6"]


def test_reviewed_and_canonical_concepts_are_preferred_targets():
    key = "квантов отжиг"
    canonical = stable_id("concept", "Technology", key)
    concepts = [
        concept("concept:old", "квантового отжига"),
        concept(canonical, "квантовый отжиг"),
        concept("concept:reviewed", "Квантовый отжиг", status="accepted"),
    ]
    plan = plan_key_migration(concepts, mention_counts={"concept:old": 9})
    assert {item.target for item in plan.merges} == {"concept:reviewed"}
    plan = plan_key_migration(concepts[:2], mention_counts={"concept:old": 9})
    assert {item.target for item in plan.merges} == {canonical}


def test_label_counts_are_seeded_from_mention_forms():
    plan = plan_key_migration(
        [concept("concept:1", "Quantum annealer")],
        form_counts={
            "concept:1": {"quantum annealers": 4, "Quantum annealer": 1}
        },
    )
    (update,) = plan.updates
    assert update.label_counts == {
        "quantum annealers": 4,
        "Quantum annealer": 1,
    }
    assert update.preferred_label == "quantum annealers"


def test_a_second_run_plans_no_merges():
    concepts = [
        concept("concept:1", "языковые модели"),
        concept("concept:2", "языковой модели"),
    ]
    plan = plan_key_migration(concepts)
    survivors = [
        item
        for item in concepts
        if item.concept_id not in {m.source for m in plan.merges}
    ]
    assert plan_key_migration(survivors).merges == []


class Store:
    def __init__(self, concepts):
        self.concepts = concepts
        self.calls = []

    async def read_concepts(self):
        return self.concepts

    async def read_concept_forms(self):
        return {}, {}

    async def write_concept_identities(self, updates):
        self.calls.append(("identities", [u.concept_id for u in updates]))

    async def merge_concepts(self, source, target, reason=None):
        self.calls.append(("merge", source, target, reason))
        return {}


def test_apply_writes_identities_before_merging():
    store = Store(
        [
            concept("concept:1", "языковые модели"),
            concept("concept:2", "языковой модели"),
        ]
    )
    plan = asyncio.run(apply_key_migration(store, apply=True))
    assert store.calls[0][0] == "identities"
    assert store.calls[1][0] == "merge"
    assert store.calls[1][3] == "lexical-key/2 migration"
    assert len(plan.merges) == 1


def test_dry_run_changes_nothing():
    store = Store(
        [
            concept("concept:1", "языковые модели"),
            concept("concept:2", "языковой модели"),
        ]
    )
    plan = asyncio.run(apply_key_migration(store, apply=False))
    assert store.calls == []
    assert len(plan.merges) == 1


def test_cli_migration_is_a_dry_run_by_default(monkeypatch, capsys):
    store = Store([concept("concept:1", "AI"), concept("concept:2", "ai")])

    class Opened(Store):
        async def __aenter__(self):
            return store

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr(cli, "_store", lambda: Opened([]))
    monkeypatch.setattr(sys, "argv", ["lctrend", "migrate-concept-keys"])
    cli.main()
    assert store.calls == []
    assert '"apply": false' in capsys.readouterr().out


class Rows(list):
    async def consume(self):
        return None


class Session:
    def __init__(self, rows, queries):
        self.rows = rows
        self.queries = queries

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def run(self, query, **parameters):
        self.queries.append((query, parameters))
        return Rows(self.rows if "RETURN c.concept_id" in query else [])

    async def execute_write(self, callback):
        return await callback(self)


def graph_store(rows=()):
    from lctrend.graph.store import GraphStore

    queries = []
    store = GraphStore.__new__(GraphStore)
    store._database = "neo4j"
    store._driver = type(
        "Driver",
        (),
        {"session": lambda self, **_: Session(list(rows), queries)},
    )()
    return store, queries


def test_store_counts_forms_and_writes_identities_by_label():
    from lctrend.graph.migration import IdentityUpdate

    store, queries = graph_store(
        [
            {"concept_id": "c1", "form": "LLM", "mentions": 2},
            {"concept_id": "c1", "form": "LLMs", "mentions": 1},
        ]
    )
    mentions, forms = asyncio.run(store.read_concept_forms())
    assert mentions == {"c1": 3}
    assert forms == {"c1": {"LLM": 2, "LLMs": 1}}
    assert "<> 'ambiguous'" in queries[0][0]
    queries.clear()
    asyncio.run(
        store.write_concept_identities(
            [IdentityUpdate("c1", "Method", "llm", {"LLM": 2}, "LLM")]
        )
    )
    query, parameters = queries[0]
    assert "MATCH (c:Method {concept_id: row.concept_id})" in query
    assert parameters["key_version"] == "lexical-key/2"
    assert parameters["rows"][0]["identity_key"] == "llm"


def test_mention_votes_preserve_technical_kind_and_quotes_are_dropped():
    from lctrend.core.models import ConceptName

    quote = "ML-236A, ML-236B and ML-236C, new inhibitors of cholesterogenesis"
    stored = Concept(
        concept_id="concept:ml",
        kind=T,
        preferred_label="ML-236B",
        names=[
            ConceptName(
                name_id="n1",
                text=quote,
                normalized_text=quote.casefold(),
                name_kind="observed",
                status="provisional",
            ),
            ConceptName(
                name_id="n2",
                text="ml236b",
                normalized_text="ml236b",
                name_kind="observed",
                status="provisional",
            ),
        ],
    )
    plan = plan_key_migration(
        [stored], kind_counts={"concept:ml": {"Material": 4, "Technology": 1}}
    )
    (update,) = plan.updates
    assert (update.kind, update.new_kind) == ("Technology", None)
    assert [name.text for name in update.names] == ["ml236b"]
    summary = plan.summary(False)
    assert summary["retyped"] == 0 and summary["names_cleaned"] == 1


def test_a_reviewed_concept_keeps_its_kind():
    stored = concept("concept:1", "ML-236B", status="accepted")
    plan = plan_key_migration(
        [stored], kind_counts={"concept:1": {"Material": 9}}
    )
    assert plan.updates[0].new_kind is None


def test_store_reads_mention_kinds_and_relabels_retyped_concepts():
    from lctrend.graph.migration import IdentityUpdate

    store, queries = graph_store(
        [
            {"concept_id": "c1", "kind": "Material", "mentions": 3},
            {"concept_id": "c1", "kind": "Technology", "mentions": 1},
        ]
    )
    assert asyncio.run(store.read_concept_kinds()) == {
        "c1": {"Material": 3, "Technology": 1}
    }
    queries.clear()
    asyncio.run(
        store.write_concept_identities(
            [
                IdentityUpdate(
                    "c1",
                    "Technology",
                    "ml 236 b",
                    {"ML-236B": 4},
                    "ML-236B",
                    kind_counts={"Material": 3, "Technology": 1},
                    new_kind="Material",
                )
            ]
        )
    )
    relabel = [q for q, _ in queries if "REMOVE c:Technology" in q]
    assert relabel and "SET c:Material" in relabel[0]
    rows = next(p for q, p in queries if "c.identity_key" in q)["rows"]
    assert rows[0]["kind_counts_json"] == '{"Material":3,"Technology":1}'
    assert rows[0]["names_json"] is None
