"""Entity labels and country codes must be grounded in the source (B-3)."""

import pytest

from lctrend.core.models import Chunk
from lctrend.llm.contracts import Extraction
from lctrend.llm.validation import validate_local_extraction
from tests.llm.test_parties_and_maturity import (
    FIRST,
    claim,
    document,
    entity,
    extraction,
)


def issues(response, text=None):
    source = document()
    if text is not None:
        source.chunks = [
            Chunk(chunk_id="c1", kind="paragraph", text=text, order=0)
        ]
    _, found = validate_local_extraction(
        source, Extraction.model_validate(response), {"c1"}
    )
    return found


def test_a_phantom_label_behind_a_real_quote_is_rejected():
    response = extraction()
    response["entities"][0] = entity(
        "battery",
        "GPT-7 quantum lithium-sulfur battery",
        "Technology",
        "pilot",
    )
    found = issues(response)
    assert "label_not_grounded" in found["entity:battery"]
    # Its claims cannot survive a rejected entity.
    assert "invalid_entity:battery" in found["claim:developer"]


@pytest.mark.parametrize(
    "label,kind,quote",
    [
        # Inflection and plural are one key; the label may come from the
        # chunk rather than the short quote.
        ("solid-state batteries", "Technology", "solid-state battery"),
        ("Solid-State Battery", "Technology", "battery"),
        ("Германия", "Country", "Германии"),
        ("Acme Energy", "Company", "Acme"),
    ],
)
def test_labels_grounded_up_to_case_and_inflection_pass(label, kind, quote):
    response = extraction()
    extra = {"country_code": "DE"} if kind == "Country" else {}
    response["entities"][0] = entity("x", label, kind, quote, **extra)
    response["claims"] = []
    assert "label_not_grounded" not in issues(response).get("entity:x", [])


def test_a_synonym_of_the_text_grounds_a_label():
    text = "Большие языковые модели ускоряют поиск."
    response = {
        "entities": [
            entity("llm", "LLM", "Technology", "Большие языковые модели")
        ],
        "claims": [],
        "context_requests": [],
    }
    assert issues(response, text).get("entity:llm", []) == []


@pytest.mark.parametrize(
    "label,code,expected",
    [
        ("Германии", "US", ["country_code_mismatch"]),
        ("Германии", "DE", []),
        ("ФРГ", "DE", []),
        ("немецкой", "DE", []),
        ("США", "US", []),
    ],
)
def test_country_code_must_name_the_quoted_country(label, code, expected):
    text = FIRST + " Затем ФРГ, США и немецкой компанией."
    response = {
        "entities": [
            entity("battery", "solid-state battery", "Technology"),
            entity("c", label, "Country", country_code=code),
        ],
        "claims": [
            claim(
                "country",
                "developed_in",
                {"subject": "battery", "country": "c"},
                FIRST,
            )
        ],
        "context_requests": [],
    }
    found = issues(response, text)
    assert [
        item for item in found.get("entity:c", []) if "country" in item
    ] == expected


def test_descriptive_kinds_may_name_what_the_quote_describes():
    # A metric is named in the model's own words ("consumes 8 mW").
    text = "Sensor S consumes 8 mW for monitoring."
    response = {
        "entities": [
            entity("power", "power consumption", "Metric", "consumes")
        ],
        "claims": [],
        "context_requests": [],
    }
    assert issues(response, text).get("entity:power", []) == []
