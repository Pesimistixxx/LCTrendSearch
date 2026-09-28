"""Explicit synthetic sensor definitions for graph serialization tests.

These fixtures bypass inference deliberately: the tests exercise persistence,
not the scientific meaning of the placeholder label.
"""

from lctrend.core.models import (
    TECHNOLOGY_FIELDS,
    Chunk,
    ConceptKind,
    EvidenceSpan,
    TechnologyProfile,
)


def define_test_sensors(document, result):
    for concept in result.concepts:
        if concept.kind != ConceptKind.TECHNOLOGY:
            continue
        text = (
            f"{concept.preferred_label} is the name of a "
            "synthetic test sensor. "
            "It measures incident light by collecting photodiode charge. "
            "The device boundary is the photodiode and its charge readout."
        )
        chunk = Chunk(
            chunk_id="definition:" + concept.concept_id,
            kind="paragraph",
            text=text,
            order=len(document.chunks),
        )
        document.chunks.append(chunk)
        profile = TechnologyProfile(
            canonical_name=concept.preferred_label,
            definition="Light sensor based on photodiode charge collection.",
            function="Measure incident light",
            mechanism="Photodiode charge collection",
            boundary="Photodiode and charge readout",
            identity_scope="test photodiode sensor",
            document_version_id=document.document_version_id,
            run_id=result.run.run_id,
            review_reason="The synthetic fixture specifies all device parts.",
            evidence=[
                EvidenceSpan(
                    chunk_id=chunk.chunk_id,
                    quote=text,
                    start=0,
                    end=len(text),
                    supports_fields=sorted(TECHNOLOGY_FIELDS),
                )
            ],
        )
        concept.technology = profile
        concept.definition = profile.definition
        concept.identity_scope = profile.identity_scope
    return result
