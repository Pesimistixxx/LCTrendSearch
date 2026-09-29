"""Technology vertex contract (docs/technology-contract.md): a Technology
exists only with a source-backed mechanism and function; otherwise the
mention stays a ConceptCandidate with the reasons.
"""

import asyncio

import pytest

from lctrend.core.models import (
    Artifact,
    Chunk,
    ConceptKind,
    DocumentEnvelope,
    DocumentType,
    SourceRef,
)
from lctrend.llm.client import ReplayProvider
from lctrend.llm.contracts import Extraction
from lctrend.llm.pipeline import process_document
from lctrend.llm.validation import validate_local_extraction
from tests.llm.test_parties_and_maturity import settings

NAME = "KV cache offloading"
MECHANISM = "moves the KV cache of long contexts from GPU memory to DRAM"
FUNCTION = "serves contexts longer than GPU memory allows"
TEXT = (
    f"We present {NAME}, which {MECHANISM} and NVMe tiers. "
    f"The system {FUNCTION}. Acme Inference developed the {NAME}."
)
DEVELOPED = f"Acme Inference developed the {NAME}."


def document():
    return DocumentEnvelope(
        document_id="doc",
        document_version_id="version",
        document_type=DocumentType.REPORT,
        title="KV cache report",
        published_at="2026-03-01",
        source=SourceRef(
            source_id="fixture",
            name="fixture",
            source_type="test",
            record_id="1",
        ),
        artifact=Artifact(
            uri="memory://fixture", sha256="a" * 64, media_type="text/plain"
        ),
        chunks=[Chunk(chunk_id="c1", kind="paragraph", text=TEXT, order=0)],
    )


def technology(**overrides):
    value = {
        "local_id": "kv",
        # A canonical name summarizing the source, not a verbatim quote.
        "label": "тиринг KV-кэша с выгрузкой контекста в DRAM/NVMe",
        "kind": "Technology",
        "source_names": [{"name": NAME, "context": "abstract"}],
        "definition": "Выгрузка KV-кэша длинных контекстов из памяти GPU "
        "в DRAM и NVMe.",
        "technical_mechanism": "перенос KV-кэша между уровнями памяти",
        "technical_function": "обслуживание контекстов длиннее памяти GPU",
        "technology_type": "software_system",
        "boundary": "не любой кэш LLM: только выгрузка KV-кэша по уровням",
        "support": [
            {"chunk_id": "c1", "quote": MECHANISM, "supports": ["mechanism"]},
            {"chunk_id": "c1", "quote": FUNCTION, "supports": ["function"]},
        ],
        "evidence": [{"chunk_id": "c1", "quote": NAME}],
    }
    value.update(overrides)
    return value


def extraction(entity=None):
    return {
        "entities": [
            entity or technology(),
            {
                "local_id": "acme",
                "label": "Acme Inference",
                "kind": "Company",
                "evidence": [{"chunk_id": "c1", "quote": "Acme Inference"}],
            },
        ],
        "claims": [
            {
                "claim_id": "developer",
                "predicate": "developed_by",
                "roles": {"subject": "kv", "organization": "acme"},
                "polarity": "affirmed",
                "modality": "reported",
                "evidence": [{"chunk_id": "c1", "quote": DEVELOPED}],
            }
        ],
        "context_requests": [],
    }


def validate(entity):
    notes = []
    result, issues = validate_local_extraction(
        document(),
        Extraction.model_validate(extraction(entity)),
        {"c1"},
        notes,
    )
    return result.entities[0], issues, notes


def test_complete_technology_is_validated_with_a_summarized_name():
    entity, issues, _ = validate(technology())
    assert "entity:kv" not in issues
    assert entity.kind == ConceptKind.TECHNOLOGY
    assert entity.classification_status == "validated"
    assert entity.contract_issues == []
    # The verbatim name becomes identity evidence.
    assert NAME in entity.aliases
    assert [span.start for span in entity.support] == [
        TEXT.index(MECHANISM),
        TEXT.index(FUNCTION),
    ]


def test_unmet_conditions_keep_a_candidate_with_reasons():
    entity, issues, notes = validate(
        technology(
            technical_mechanism=None,
            support=[
                {
                    "chunk_id": "c1",
                    "quote": "not in the text",
                    "supports": ["function"],
                }
            ],
            uncertainty="механизм в тексте не описан",
        )
    )
    assert entity.kind == ConceptKind.CANDIDATE
    assert entity.classification_status == "proposed"
    assert set(entity.contract_issues) == {
        "missing:technical_mechanism",
        "unsupported:mechanism",
        "unsupported:function",
        "model_uncertainty",
    }
    assert "entity:kv" not in issues
    unmet = next(
        note for note in notes if note["code"] == "technology_contract_unmet"
    )
    assert unmet["uncertainty"] == "механизм в тексте не описан"
    assert any(note["code"] == "support_dropped" for note in notes)
    # A claim about an undefined technology cannot attach to it.
    assert any(
        issue.startswith("role_type_mismatch")
        for issue in issues["claim:developer"]
    )


@pytest.mark.shipped_technology_contract
def test_shipped_contract_marks_unmet_technologies_without_demoting_them():
    # 2026-09-29, live: GigaChat-3-Ultra leaves the contract fields empty on
    # abstracts and READMEs, so enforcement demoted every technology and
    # dropped every claim about it. Shipped: reported, not enforced.
    entity, issues, notes = validate(technology(technical_mechanism=None))
    assert entity.kind == ConceptKind.TECHNOLOGY
    assert entity.classification_status == "proposed"
    assert "missing:technical_mechanism" in entity.contract_issues
    assert "claim:developer" not in issues
    assert not any(
        note["code"] == "technology_contract_unmet" for note in notes
    )


def test_bare_abbreviation_and_phantom_names_are_not_technologies():
    entity, _, _ = validate(
        technology(
            label="KV",
            source_names=[{"name": "KV"}],
            evidence=[{"chunk_id": "c1", "quote": "KV"}],
        )
    )
    assert "abbreviation_not_expanded" in entity.contract_issues
    assert entity.kind == ConceptKind.CANDIDATE
    entity, issues, notes = validate(
        technology(source_names=[{"name": "quantum KV teleport"}])
    )
    # Neither the summarized label nor a real source name is in the text.
    assert entity.source_names == []
    assert "label_not_grounded" in issues["entity:kv"]
    assert any(note["code"] == "source_name_dropped" for note in notes)


def test_invalid_type_and_overlong_field_are_reported():
    entity, _, _ = validate(
        technology(technology_type="startup", boundary="word " * 60)
    )
    assert {"invalid_technology_type", "too_long:boundary"} <= set(
        entity.contract_issues
    )


def test_profile_reaches_the_concept_and_the_run_audit():
    result = asyncio.run(
        process_document(
            document(),
            ReplayProvider(
                [
                    extraction(),
                    {
                        "items": [
                            {
                                "claim_id": "developer",
                                "decision": "supported",
                                "reason": "Stated.",
                            }
                        ]
                    },
                ]
            ),
            settings=settings(),
        )
    )
    concept = next(
        item for item in result.concepts if item.kind == ConceptKind.TECHNOLOGY
    )
    profile = concept.profile
    assert profile["classification_status"] == "validated"
    assert profile["technology_type"] == "software_system"
    assert profile["source_names"] == [{"name": NAME, "context": "abstract"}]
    # Retrospective labels exist and stay empty.
    assert profile["signal_36m"] is None and profile["trend_36m"] is None
    assert {tuple(item["supports"]) for item in profile["evidence"]} == {
        ("mechanism",),
        ("function",),
    }
    assert all(
        item["document_version_id"] == "version"
        for item in profile["evidence"]
    )
    bindings = result.run.metadata["entity_bindings"]
    assert any(
        binding.get("technology_profile") == profile
        for binding in bindings.values()
    )
    assert any(
        assertion.predicate == "developed_by"
        for assertion in result.assertions
    )


def test_graph_write_carries_the_profile_to_the_vertex():
    from lctrend.graph.store import GraphStore, evidence_chunk_ids
    from tests.llm.test_parties_and_maturity import Transaction

    doc = document()
    result = asyncio.run(
        process_document(
            doc,
            ReplayProvider(
                [
                    extraction(),
                    {
                        "items": [
                            {
                                "claim_id": "developer",
                                "decision": "supported",
                                "reason": "Stated.",
                            }
                        ]
                    },
                ]
            ),
            settings=settings(),
        )
    )
    tx = Transaction()
    asyncio.run(
        GraphStore._write_document(
            tx, doc, kept_chunk_ids=evidence_chunk_ids(result)
        )
    )
    asyncio.run(GraphStore._write_extraction(tx, doc, result))
    query, parameters = next(
        (query, parameters)
        for query, parameters in tx.queries
        if "MERGE (c:Technology {concept_id" in query
    )
    assert "c.classification_status" in query
    [row] = parameters["rows"]
    assert row["profile"]["classification_status"] == "validated"
    assert row["profile_json"] and '"signal_36m":null' in row["profile_json"]
    # Mention rows do not repeat the profile.
    assert not any(
        "profile" in row
        for query, parameters in tx.queries
        if "MENTIONS" in query
        for row in parameters.get("rows", [])
    )


def titled_document():
    """An abstract that never names the method; only the title does."""
    doc = document()
    abstract = doc.chunks[0].model_copy(
        update={"text": TEXT.replace(NAME, "it")}
    )
    title = Chunk(
        chunk_id="c2", kind="title", text=f"{NAME} for long contexts", order=1
    )
    return doc.model_copy(update={"chunks": [abstract, title]})


def test_title_is_context_of_every_packet_not_a_packet():
    from lctrend.llm.context import DOCUMENT_CONTEXT, plan_packets

    plan = plan_packets(titled_document(), settings())
    assert [packet.focus_chunk_ids for packet in plan.packets] == [["c1"]]
    [packet] = plan.packets
    assert packet.support_chunk_ids == ["c2"]
    assert packet.selection_reasons["c2"] == DOCUMENT_CONTEXT


def test_a_method_named_only_in_the_title_is_extracted_in_one_call():
    doc = titled_document()
    answer = extraction(
        technology(
            evidence=[{"chunk_id": "c2", "quote": NAME}],
            support=[
                {
                    "chunk_id": "c1",
                    "quote": MECHANISM,
                    "supports": ["mechanism"],
                },
                {
                    "chunk_id": "c1",
                    "quote": FUNCTION,
                    "supports": ["function"],
                },
            ],
        )
    )
    answer["claims"] = []
    provider = ReplayProvider([answer, {"items": []}])
    result = asyncio.run(process_document(doc, provider, settings=settings()))
    assert result.run.status == "succeeded"
    assert len(provider.calls) == 1
    assert result.run.metadata["coverage"]["unprocessed_chunk_ids"] == []
    [concept] = [
        item for item in result.concepts if item.kind == ConceptKind.TECHNOLOGY
    ]
    assert concept.profile["classification_status"] == "validated"


def test_missing_name_quote_and_verb_form_are_repaired():
    doc = titled_document()
    entity = technology(
        support=[
            {"chunk_id": "c1", "quote": MECHANISM, "supports": ["mechanism"]},
            {"chunk_id": "c1", "quote": FUNCTION, "supports": ["function"]},
        ]
    )
    del entity["evidence"]
    answer = extraction(entity)
    answer["claims"][0]["predicate"] = "develop_by"
    notes = []
    result, issues = validate_local_extraction(
        doc, Extraction.model_validate(answer), {"c1", "c2"}, notes
    )
    # The name is only in the title: its quote moved there.
    [span] = result.entities[0].evidence
    assert (span.chunk_id, span.quote) == ("c2", NAME)
    assert "entity:kv" not in issues
    assert result.entities[0].classification_status == "validated"
    assert any(note["code"] == "name_quote_relocated" for note in notes)
    claim = Extraction.model_validate(answer).claims[0]
    claim.predicate = "reports_measurement"
    from lctrend.llm.validation import _normalize_predicate

    _normalize_predicate(claim, "claim:k", {"reported_measurement": {}}, notes)
    assert claim.predicate == "reported_measurement"


def test_languages_and_libraries_are_candidates_not_technologies():
    entity, _, notes = validate(
        technology(label="PyTorch", source_names=[{"name": NAME}])
    )
    assert entity.kind == ConceptKind.CANDIDATE
    assert any(
        note["code"] == "kind_normalized" and note["to"] == "ConceptCandidate"
        for note in notes
    )


def test_a_dropped_item_does_not_make_the_document_partial():
    from lctrend.llm.pipeline import _item_issue

    assert _item_issue({"code": "invalid_items", "stage": "extract"})
    assert not _item_issue({"code": "call_budget"})


@pytest.mark.technology_triage
def test_triage_turns_products_and_operations_into_candidates():
    save = technology(
        local_id="save",
        label="Save",
        source_names=[{"name": "Acme Inference"}],
        evidence=[{"chunk_id": "c1", "quote": "Acme Inference"}],
    )
    answer = extraction()
    answer["entities"].append(save)
    triage = {
        "items": [
            {"id": "t1", "verdict": "technology", "reason": "подход"},
            {
                "id": "t2",
                "verdict": "software_component",
                "reason": "операция",
            },
        ]
    }
    review = {
        "items": [
            {"claim_id": "developer", "decision": "supported", "reason": "Ok."}
        ]
    }
    provider = ReplayProvider([answer, review, triage])
    result = asyncio.run(
        process_document(document(), provider, settings=settings())
    )
    technologies = [
        concept.preferred_label
        for concept in result.concepts
        if concept.kind == ConceptKind.TECHNOLOGY
    ]
    assert technologies == ["тиринг KV-кэша с выгрузкой контекста в DRAM/NVMe"]
    rejected = [
        binding["technology_profile"]
        for binding in result.run.metadata["entity_bindings"].values()
        if binding.get("technology_profile", {}).get("classification_status")
        == "rejected"
    ]
    assert len(rejected) == 1
    assert "triage:software_component" in rejected[0]["contract_issues"]
    assert result.run.metadata["technology_triage"] == {
        "candidates": 2,
        "judged": 2,
        "rejected": {"software_component": 1},
    }


@pytest.mark.technology_triage
def test_failed_triage_keeps_the_candidates():
    provider = ReplayProvider(
        [
            extraction(),
            {
                "items": [
                    {
                        "claim_id": "developer",
                        "decision": "supported",
                        "reason": "Ok.",
                    }
                ]
            },
        ]
    )
    result = asyncio.run(
        process_document(document(), provider, settings=settings())
    )
    assert any(
        concept.kind == ConceptKind.TECHNOLOGY for concept in result.concepts
    )
    assert result.run.metadata["technology_triage"]["failed_calls"]
