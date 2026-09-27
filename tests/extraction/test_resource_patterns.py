import re

import pytest

from lctrend.core.config import load_catalog
from lctrend.core.models import (
    Chunk,
    ConceptKind,
)
from lctrend.extraction.economics import (
    _currency,
    amount_value,
    extract_economic_evidence,
)
from lctrend.extraction.resolver import resolve_mentions
from lctrend.ingest.adapters import _domains_from_values
from lctrend.ingest.fulltext import section_role


@pytest.mark.parametrize(
    "amount,value,currency",
    [
        ("$5 thousand", 5000, "USD"),
        ("5 тыс. RUB", 5000, "RUB"),
        ("1,5 млн. рублей", 1500000, "RUB"),
        ("CNY 2 million", 2000000, "CNY"),
        ("3 thousand JPY", 3000, "JPY"),
        ("CN¥ 200", 200, "CNY"),
        ("JP¥ 300", 300, "JPY"),
    ],
)
def test_currency_and_scale_are_extracted_together(amount, value, currency):
    rules = load_catalog("extraction")["economics"]
    matches = list(re.finditer(rules["money_pattern"], amount, re.I))
    assert len(matches) == 1
    raw = matches[0].group().strip()
    assert raw == amount
    assert amount_value(raw, rules["scales"]) == value
    assert _currency(raw, rules["currencies"]) == currency


def test_ambiguous_yen_symbol_is_not_assigned_to_a_currency():
    rules = load_catalog("extraction")["economics"]
    assert re.search(rules["money_pattern"], "¥300", re.I) is None


def test_thousand_abbreviation_does_not_split_currency_from_amount():
    from lctrend.core.models import Mention

    text = "SensorX costs 5 тыс. RUB."
    mention = Mention(
        mention_id="m",
        chunk_id="c",
        surface_text="SensorX",
        start=0,
        end=7,
        type_candidates=[ConceptKind.TECHNOLOGY],
    )
    concepts, resolutions = resolve_mentions([mention], ())
    evidence = extract_economic_evidence(
        [Chunk(chunk_id="c", kind="paragraph", order=0, text=text)],
        [mention],
        concepts,
        resolutions,
    )
    assert len(evidence) == 1
    assert evidence[0].amount_value == 5000
    assert evidence[0].currency == "RUB"


def test_russian_domains_do_not_expand_search_queries_implicitly():
    domains = _domains_from_values(["компьютерное зрение и робототехника"])
    assert {item.name for item in domains} == {"Computer vision", "Robotics"}
    rules = load_catalog("sources")["domains"]
    vision = next(item for item in rules if item["name"] == "Computer vision")
    assert "компьютерное зрение" in vision["aliases"]
    assert "компьютерное зрение" not in vision["search_aliases"]


@pytest.mark.parametrize(
    "heading,role",
    [
        ("Методика", "method"),
        ("Валидация", "results"),
        ("Постановка задачи", "problem_statement"),
        ("Практическое применение", "deployment"),
        ("Экономическая оценка", "economics"),
    ],
)
def test_section_roles_support_russian_navigation(heading, role):
    assert section_role(heading) == role
