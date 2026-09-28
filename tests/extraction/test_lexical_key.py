"""Lexical identity key v2: the regression table of the 2026-09-28 audit.

Each row resolves two mentions against an empty registry. A false merge
must end in two concepts, a false split in one.
"""

import pytest

from lctrend.core.models import ConceptKind, Mention
from lctrend.extraction.resolver import resolve_mentions

T = ConceptKind.TECHNOLOGY
COUNTRY = ConceptKind.COUNTRY


def mention(mention_id, text, kind):
    return Mention(
        mention_id=mention_id,
        chunk_id="c1",
        surface_text=text,
        canonical_text=text,
        start=0,
        end=len(text),
        type_candidates=[kind],
    )


def same_concept(left, right, kind=T, right_kind=None):
    _, decisions = resolve_mentions(
        [mention("m1", left, kind), mention("m2", right, right_kind or kind)],
        [],
    )
    first, second = decisions
    return (
        second.concept_id is not None
        and first.concept_id == second.concept_id
    )


# C-1: English lemmatization merged acronyms and unrelated words.
FALSE_MERGES_C1 = [
    ("AI", "AM", T),
    ("AI", "IS", T),
    ("AI", "BE", T),
    ("IS", "BE", T),
    ("GAN", "gin", T),
    ("LED", "lead", T),
    ("Faster R-CNN", "Fast R-CNN", T),
    # Countries are identified by the ISO code alone: Iceland is not Belgium.
    ("IS", "BE", COUNTRY),
    ("AM", "IS", COUNTRY),
    ("US", "USA", COUNTRY),
]

FALSE_SPLITS_C1 = [
    ("models", "model", T),
    ("graph neural networks", "graph neural network", T),
    ("LLMs", "LLM", T),
    ("LEDs", "LED", T),
    ("batteries", "battery", T),
    ("LARGE LANGUAGE MODELS", "large language model", T),
    ("de", "DE", COUNTRY),
]


@pytest.mark.parametrize("left,right,kind", FALSE_MERGES_C1)
def test_false_merges_are_separated(left, right, kind):
    assert not same_concept(left, right, kind)


@pytest.mark.parametrize("left,right,kind", FALSE_SPLITS_C1)
def test_false_splits_are_joined(left, right, kind):
    assert same_concept(left, right, kind)


# C-2: Russian inflected forms of one name were split; ё/е too.
FALSE_SPLITS_C2 = [
    ("языковая модель", "языковые модели", T),
    ("языковая модель", "языковой модели", T),
    ("большая языковая модель", "большой языковой модели", T),
    ("большие языковые модели", "больших языковых моделей", T),
    ("квантовый отжиг", "квантового отжига", T),
    ("федеративное обучение", "федеративного обучения", T),
    ("нейронная сеть", "нейронных сетей", T),
    ("цифровой двойник", "цифровые двойники", T),
    ("твердотельный аккумулятор", "твердотельных аккумуляторов", T),
    ("интернет вещей", "интернета вещей", T),
    ("обучение с подкреплением", "обучения с подкреплением", T),
    ("твёрдый электролит", "твердый электролит", T),
    ("ёмкостный датчик", "емкостные датчики", T),
    ("БОЛЬШИЕ ЯЗЫКОВЫЕ МОДЕЛИ", "большая языковая модель", T),
]

FALSE_MERGES_C2 = [
    ("литий-ионный аккумулятор", "натрий-ионный аккумулятор", T),
    ("модель", "моделирование", T),
    ("сеть", "сетка", T),
    # A capitalized Russian acronym is not stemmed into a letter.
    ("ИИ", "и", T),
]


@pytest.mark.parametrize("left,right,kind", FALSE_MERGES_C2)
def test_russian_false_merges_are_separated(left, right, kind):
    assert not same_concept(left, right, kind)


@pytest.mark.parametrize("left,right,kind", FALSE_SPLITS_C2)
def test_russian_forms_are_joined(left, right, kind):
    assert same_concept(left, right, kind)


# C-7: symbols were dropped as punctuation, script twins and digit
# boundaries split one name.
FALSE_MERGES_C7 = [
    ("C", "C++", T),
    ("C", "C#", T),
    ("C++", "C#", T),
    ("F#", "F", T),
    ("LoRA", "LoRa", T),
]

FALSE_SPLITS_C7 = [
    ("C++", "c++", T),
    ("GPT-4", "GPT4", T),
    ("H100", "H 100", T),
    ("5G", "5 G", T),
    # Cyrillic С and О written for Latin C and O, and the reverse.
    ("С", "C", T),
    ("СО2", "CO2", T),
    ("Тrаnsformer", "Transformer", T),
    ("cеть", "сеть", T),
    ("захват CO2", "захват СО2", T),
    ("naïve Bayes", "naive Bayes", T),
    ("ＧＰＴ－４", "GPT-4", T),
]


@pytest.mark.parametrize("left,right,kind", FALSE_MERGES_C7)
def test_symbol_names_are_separated(left, right, kind):
    assert not same_concept(left, right, kind)


@pytest.mark.parametrize("left,right,kind", FALSE_SPLITS_C7)
def test_script_twins_and_digit_boundaries_are_joined(left, right, kind):
    assert same_concept(left, right, kind)


METHOD = ConceptKind.METHOD

# C-6: synonym groups matched only the exact written form, and matched it
# case-insensitively, so GaN joined the GAN group.
FALSE_SPLITS_C6 = [
    ("LLM", "больших языковых моделей", T),
    ("LLM", "Large Language Models", T),
    ("NLP", "обработки естественного языка", T),
    ("цифровых двойников", "digital twin", T),
    ("графовых нейронных сетей", "GNN", T),
    ("федеративного обучения", "federated learning", METHOD),
    ("свёрточных нейронных сетей", "CNNs", T),
]

FALSE_MERGES_C6 = [
    ("GAN", "GaN", T),
    ("LoRA", "LoRa", METHOD),
]


@pytest.mark.parametrize("left,right,kind", FALSE_SPLITS_C6)
def test_synonym_groups_match_by_key(left, right, kind):
    assert same_concept(left, right, kind)


@pytest.mark.parametrize("left,right,kind", FALSE_MERGES_C6)
def test_synonym_groups_do_not_ignore_acronym_case(left, right, kind):
    assert not same_concept(left, right, kind)
