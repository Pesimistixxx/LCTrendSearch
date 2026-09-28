"""Semantic gold cases plus adversarial contract tests.

Replay responses are hand-reviewed proposals, not an LLM quality benchmark.
The expected labels are independent of lexical matching and name length.
"""

import asyncio
import json
from copy import deepcopy
from pathlib import Path

import pytest

from lctrend.core.models import ConceptKind, validate_extraction
from lctrend.extraction.resolver import ConceptRegistry
from lctrend.graph.store import GraphStore, _concept_from_properties
from lctrend.graph.technology_migration import technology_dry_run
from lctrend.llm.client import ReplayProvider
from lctrend.llm.pipeline import process_document
from tests.graph.test_concept_identity_store import Transaction
from tests.llm.test_llm_pipeline import document, settings

CASES = json.loads(
    (
        Path(__file__).parents[1] / "fixtures/technology_contract.json"
    ).read_text()
)


def run_case(case, registry=()):
    doc = document([case["text"]])
    doc.document_version_id = "version:" + case["id"]
    result = asyncio.run(
        process_document(
            doc,
            ReplayProvider([case["extraction"], case["review"]]),
            registry=registry,
            settings=settings(),
        )
    )
    return doc, result


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_gold_entity_types_and_graph_roundtrip(case):
    doc, result = run_case(case)
    assert result.run.status == "succeeded"
    assert len(result.concepts) == 1
    concept = result.concepts[0]
    assert concept.kind.value == case["expected_kind"]
    assert len(result.entity_assessments) == 1
    validate_extraction(doc, result)
    tx = Transaction()
    asyncio.run(GraphStore._write_extraction(tx, doc, result))
    rows = [
        row
        for q, p in tx.queries
        if "c.preferred_label = row.preferred_label" in q
        for row in p["rows"]
    ]
    loaded = _concept_from_properties(rows[0])
    assert loaded == concept
    if concept.kind == ConceptKind.TECHNOLOGY:
        assert loaded.technology and loaded.definition
        assert all(
            doc.chunks[0].text[s.start : s.end] == s.quote
            for s in loaded.technology.evidence
        )
        assert any("ASSESSED_ENTITY" in q for q, _ in tx.queries)
    else:
        assert not any("MERGE (c:Technology" in q for q, _ in tx.queries)


def test_completeness_all_concrete_approaches_without_claims_or_whitelist():
    cases = CASES[:6]
    text = "\n".join(c["text"] for c in cases)
    doc = document([text])
    extraction = {"entities": [c["extraction"]["entities"][0] for c in cases]}
    review = {"entity_items": [c["review"]["entity_items"][0] for c in cases]}
    result = asyncio.run(
        process_document(
            doc, ReplayProvider([extraction, review]), settings=settings()
        )
    )
    assert len(result.concepts) == 6
    assert all(c.kind == ConceptKind.TECHNOLOGY for c in result.concepts)
    assert len(result.entity_assessments) == 6
    assert result.assertions == []


@pytest.mark.parametrize(
    "attack",
    [
        "missing_review",
        "missing_field",
        "foreign_quote",
        "invented_quote",
        "stitched",
        "results_in_definition",
        "application_only",
    ],
)
def test_bad_definitions_do_not_mint_technology(attack):
    case = deepcopy(CASES[1])
    entity = case["extraction"]["entities"][0]
    review = case["review"]["entity_items"][0]
    if attack == "missing_review":
        case["review"]["entity_items"] = []
    elif attack == "missing_field":
        entity["technology"]["evidence"][0]["supports_fields"].remove(
            "mechanism"
        )
    elif attack == "foreign_quote":
        entity["technology"]["evidence"][0]["chunk_id"] = "foreign:c1"
    elif attack == "invented_quote":
        entity["technology"]["evidence"][0]["quote"] = (
            "It uses quantum teleportation."
        )
    elif attack == "stitched":
        review["coherent"] = False
    elif attack == "results_in_definition":
        review["definition_only"] = False
    else:
        review["adaptation_or_base"] = False
    _, result = run_case(case)
    assert all(c.kind != ConceptKind.TECHNOLOGY for c in result.concepts)
    assert result.entity_assessments[0].decision == "unresolved"
    assert result.entity_assessments[0].reason


def test_variants_share_identity_but_homonyms_do_not():
    registry = ConceptRegistry()
    _, first = run_case(CASES[1], registry)
    variant = deepcopy(CASES[1])
    variant["id"] = "vit_variant"
    variant["text"] = variant["text"].replace(
        "Vision Transformer", "vision-transformer"
    )
    entity = variant["extraction"]["entities"][0]
    entity["label"] = "vision-transformer"
    entity["evidence"][0]["quote"] = "vision-transformer"
    entity["technology"]["evidence"][0]["quote"] = variant["text"]
    variant["review"]["entity_items"][0]["evidence"][0]["quote"] = variant[
        "text"
    ]
    _, second = run_case(variant, registry)
    assert first.concepts[0].concept_id == second.concepts[0].concept_id
    homonym = deepcopy(CASES[5])
    homonym["id"] = "fis_optics"
    # Same source name, different documented meaning.
    homonym["text"] = (
        "FIS-ML is an optical pulse detector using a "
        "matched-lobe interferometer."
    )
    profile = homonym["extraction"]["entities"][0]["technology"]
    profile.update(
        definition="Optical pulse detection by matched-lobe interference.",
        function="Detect optical pulses",
        mechanism="matched-lobe interferometer",
        boundary="Optical pulse detector",
        identity_scope="optical detector",
    )
    profile["evidence"][0]["quote"] = homonym["text"]
    review = homonym["review"]["entity_items"][0]
    review["identity_scope"] = profile["identity_scope"]
    review["evidence"][0]["quote"] = homonym["text"]
    _, one = run_case(CASES[5], registry)
    _, two = run_case(homonym, registry)
    assert one.concepts[0].concept_id != two.concepts[0].concept_id


def test_dry_run_preserves_inputs_and_never_writes_graph(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry run must not construct a graph writer")

    monkeypatch.setattr(GraphStore, "__init__", forbidden)
    doc, _ = run_case(CASES[0])
    snapshot = {
        "documents": [
            {
                "document": doc.model_dump(mode="json"),
                "answers": [CASES[0]["extraction"], CASES[0]["review"]],
            }
        ]
    }
    before = deepcopy(snapshot)
    plan = asyncio.run(
        technology_dry_run(snapshot, replay=True, settings=settings())
    )
    assert snapshot == before == plan["original_snapshot"]
    assert plan["graph_writes"] == 0
    assert plan["summary"] == {"add": 1}


def test_consistent_reviews_with_different_reasons_do_not_lose_technology():
    from lctrend.llm.contracts import EntityReviewItem, Extraction
    from lctrend.llm.validation import assess_entity, validate_local_extraction

    case = CASES[1]
    doc = document([case["text"]])
    local, _ = validate_local_extraction(
        doc, Extraction.model_validate(case["extraction"]), {"c1"}
    )
    first = EntityReviewItem.model_validate(case["review"]["entity_items"][0])
    second = first.model_copy(
        update={"reason": "Patch embedding is explicit."}
    )
    assessment = assess_entity(
        doc, local.entities[0], [first, second], {"c1"}, "run:test"
    )
    assert assessment.resolved_kind == ConceptKind.TECHNOLOGY
    second.specific = False
    assessment = assess_entity(
        doc, local.entities[0], [first, second], {"c1"}, "run:test"
    )
    assert assessment.resolved_kind == ConceptKind.CANDIDATE


def test_method_can_become_technology_only_after_definition_review():
    case = deepcopy(CASES[1])
    case["extraction"]["entities"][0]["kind"] = "Method"
    _, result = run_case(case)
    assert result.concepts[0].kind == ConceptKind.TECHNOLOGY
    case["review"]["entity_items"] = []
    _, result = run_case(case)
    assert result.concepts[0].kind == ConceptKind.CANDIDATE


def test_definition_can_use_several_chunks_of_one_source():
    case = deepcopy(CASES[1])
    name, mechanism = case["text"].split(". ", 1)
    doc = document([name + ".", mechanism], shared_stream=True)
    spans = case["extraction"]["entities"][0]["technology"]["evidence"]
    fields = spans[0]["supports_fields"]
    spans[:] = [
        {
            "chunk_id": "c1",
            "quote": name + ".",
            "supports_fields": ["canonical_name", "function"],
        },
        {
            "chunk_id": "c2",
            "quote": mechanism,
            "supports_fields": [
                f for f in fields if f not in {"canonical_name", "function"}
            ],
        },
    ]
    case["review"]["entity_items"][0]["evidence"] = [
        {"chunk_id": "c2", "quote": mechanism}
    ]
    result = asyncio.run(
        process_document(
            doc,
            ReplayProvider([case["extraction"], case["review"]]),
            settings=settings(primary_chunks=2),
        )
    )
    assert result.concepts[0].kind == ConceptKind.TECHNOLOGY
    assert {s.chunk_id for s in result.concepts[0].technology.evidence} == {
        "c1",
        "c2",
    }


def test_new_profile_does_not_change_search_scores():
    from tests.ranking.test_search import T, corpus, search_response

    data = corpus()
    before = search_response(data, "финтех", T)
    profile = (
        run_case(CASES[0])[1].concepts[0].technology.model_dump(mode="json")
    )
    data.technology_profiles["pay"] = profile
    after = search_response(data, "финтех", T)
    # The reader exposes the definition; selection and ranking stay identical.
    assert before["signals"] and after["signals"]
    assert [(s["id"], s["score"]) for s in before["signals"]] == [
        (s["id"], s["score"]) for s in after["signals"]
    ]
    assert (
        next(s for s in after["signals"] if s["id"] == "pay")[
            "technologyDefinition"
        ]
        == profile
    )


def test_reviewed_identity_survives_an_explicit_merge():
    from lctrend.core.models import ConceptName

    _, result = run_case(CASES[1])
    saved = result.concepts[0].model_copy(deep=True)
    saved.concept_id = "retained-id-after-reviewed-merge"
    saved.names.append(
        ConceptName(
            name_id="reviewed-name",
            text="Vision Transformer",
            normalized_text="vision transformer",
            status="accepted",
        )
    )
    _, later = run_case(CASES[1], [saved])
    assert later.concepts[0].concept_id == saved.concept_id


def test_legacy_key_migration_skips_reviewed_meanings():
    from lctrend.graph.migration import plan_key_migration

    _, result = run_case(CASES[1])
    plan = plan_key_migration(result.concepts)
    assert plan.updates == plan.merges == []


def test_dry_run_cli_cannot_overwrite_its_input(tmp_path):
    from argparse import Namespace

    from lctrend.cli import _run

    source = tmp_path / "snapshot.json"
    source.write_text('{"documents": []}')
    with pytest.raises(ValueError, match="must not overwrite"):
        _run(
            Namespace(
                command="technology-dry-run",
                input=source,
                output=source,
                replay=True,
            )
        )
    assert source.read_text() == '{"documents": []}'


def test_short_homonym_with_a_defined_mechanism_is_not_blacklisted():
    # ML deliberately has a different, explicit meaning in this fixture.
    case = json.loads(json.dumps(CASES[5]).replace("FIS-ML", "ML"))
    _, result = run_case(case)
    assert result.concepts[0].kind == ConceptKind.TECHNOLOGY
    assert result.concepts[0].technology.mechanism


def test_reviewed_explicit_alias_keeps_existing_identity():
    registry = ConceptRegistry()
    _, first = run_case(CASES[0], registry)
    case = deepcopy(CASES[0])
    entity = case["extraction"]["entities"][0]
    entity["technology"]["canonical_name"] = "CNN"
    entity["aliases"] = ["convolutional neural network"]
    case["review"]["entity_items"][0]["canonical_name"] = "CNN"
    _, second = run_case(case, registry)
    assert second.concepts[0].concept_id == first.concepts[0].concept_id
    assert second.concepts[0].preferred_label == "Convolutional neural network"
    assert second.mentions[0].declared_aliases == [
        "convolutional neural network"
    ]


def test_real_chunk_aliases_decode_definition_and_keep_evidence_nodes():
    from lctrend.graph.store import evidence_chunk_ids

    case = deepcopy(CASES[0])
    doc = document(["CNN", case["text"]], shared_stream=True)
    ids = ["chunk:" + "a" * 24, "chunk:" + "b" * 24]
    for chunk, chunk_id in zip(doc.chunks, ids):
        chunk.chunk_id = chunk_id
    entity = case["extraction"]["entities"][0]
    entity["technology"]["evidence"][0]["chunk_id"] = "c2"
    review = case["review"]["entity_items"][0]
    review["evidence"][0]["chunk_id"] = ids[1]
    result = asyncio.run(process_document(
        doc,
        ReplayProvider([case["extraction"], case["review"]]),
        settings=settings(primary_chunks=2),
    ))
    assert result.concepts[0].kind == ConceptKind.TECHNOLOGY
    assert result.concepts[0].technology.evidence[0].chunk_id == ids[1]
    validate_extraction(doc, result)
    assert set(ids) <= evidence_chunk_ids(result)
    assert result.concepts[0].profile["classification_status"] == "validated"
