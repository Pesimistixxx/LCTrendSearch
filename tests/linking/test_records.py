"""Model-free links of grants and vacancies to known technologies."""

import asyncio

import pytest

from lctrend.core.models import (
    Artifact,
    Chunk,
    Concept,
    ConceptKind,
    DocumentEnvelope,
    DocumentType,
    SourceRef,
    stable_id,
    validate_extraction,
)
from lctrend.extraction.processing import process_material
from lctrend.extraction.resolver import ConceptRegistry
from lctrend.linking.records import link_known_technologies
from lctrend.llm.client import LLMError


def concept(label, kind=ConceptKind.TECHNOLOGY, concept_id=None):
    return Concept(
        concept_id=concept_id or stable_id("concept", label),
        kind=kind,
        preferred_label=label,
    )


def document(text, document_type=DocumentType.JOB_POSTING):
    return DocumentEnvelope(
        document_id="doc",
        document_version_id="v1",
        document_type=document_type,
        title="Vacancy",
        source=SourceRef(
            source_id="hh", name="hh", source_type="api", record_id="1"
        ),
        artifact=Artifact(
            uri="memory://fixture",
            sha256="a" * 64,
            media_type="application/json",
        ),
        chunks=[Chunk(chunk_id="c1", kind="description", text=text, order=0)],
    )


REGISTRY = [
    concept("Graph Neural Network"),
    concept("графовые нейронные сети"),
    concept("GPT"),
    concept("Kubernetes"),
    concept("machine learning"),
]


class CountingModel:
    """Counts the calls that reach the model; each one fails."""

    def __init__(self):
        self.requests = 0

    async def generate(self, *args, **kwargs):
        self.requests += 1
        raise LLMError("auth", "No model in this test", retryable=False)


def linked_names(result):
    return [
        (mention.surface_text, mention.canonical_text)
        for mention in result.mentions
    ]


def test_known_names_link_without_the_model():
    record = document(
        "Experience with graph neural networks; Kubernetes, Kubernetes."
    )
    result = link_known_technologies(record, REGISTRY)

    assert linked_names(result) == [
        ("graph neural networks", "Graph Neural Network"),
        ("Kubernetes", "Kubernetes"),
    ]
    assert {d.status for d in result.resolutions} == {"accepted"}
    assert result.run.metadata["model_calls"] == 0
    assert result.run.metadata["input_coverage"] == record.coverage
    validate_extraction(record, result)


def test_inflected_russian_name_links():
    result = link_known_technologies(
        document("Опыт работы с графовыми нейронными сетями."), REGISTRY
    )

    assert [m.canonical_text for m in result.mentions] == [
        "графовые нейронные сети"
    ]


@pytest.mark.parametrize(
    "text",
    [
        # Part of a word is not the name: GPT4 is not GPT.
        "Experience with GPT4.",
        # An umbrella term is a domain, not a technology to link.
        "Knowledge of machine learning.",
        "No technology here.",
    ],
)
def test_nothing_known_needs_the_model(text):
    assert link_known_technologies(document(text), REGISTRY) is None


def test_name_of_two_concepts_links_nothing():
    registry = [
        concept("Transformer", concept_id="a"),
        concept("transformers", ConceptKind.METHOD, concept_id="b"),
    ]

    assert link_known_technologies(document("Transformers."), registry) is None


def test_economic_record_skips_the_model():
    model = CountingModel()
    result = asyncio.run(
        process_material(
            document("Kubernetes operator"),
            provider=model,
            registry=ConceptRegistry(REGISTRY),
        )
    )

    assert result.run.parser == "registry_match"
    assert result.run.status == "succeeded"
    assert model.requests == 0


@pytest.mark.parametrize(
    "record",
    [
        document("Kubernetes operator", DocumentType.ARTICLE),
        document("Nothing known"),
    ],
)
def test_other_records_and_unknown_texts_take_the_model_path(record):
    model = CountingModel()
    result = asyncio.run(
        process_material(record, provider=model, registry=REGISTRY)
    )

    assert result.run.parser == "llm_packets"
    assert model.requests > 0
