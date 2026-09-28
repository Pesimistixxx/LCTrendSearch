"""Lexical identity key v2 of concept names.

The key decides which surface forms name the same concept, so it must
neither merge unrelated names nor split inflected forms of one name:

- text is NFKC-normalized; + and # become words (C, C++ and C# differ)
  before the other punctuation and symbols split it; digits and letters
  are separate tokens (GPT-4 is GPT4);
- look-alike Cyrillic and Latin letters are written in one script, and
  Latin accents are dropped;
- acronyms written in capitals (AI, ИИ, GAN, LED) are never lemmatized;
  short mixed-case acronyms keep their case (GaN is not GAN);
- ё is е; Cyrillic words are stemmed with Snowball, Latin words only
  lose a plural ending;
- a country is identified by its ISO 3166-1 code alone.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Optional, Tuple

KEY_VERSION = "lexical-key/2"

# Longer all-capital words mean a shouted label, not an acronym.
_ACRONYM_MAX = 5
_LATIN_KEEP = ("ss", "us", "is", "ics", "as")
_COUNTRY_CODE = re.compile(r"[A-Za-z]{2}")
# Symbols that distinguish names (C, C++, C#) are words before the
# remaining punctuation and symbols are dropped.
_SYMBOLS = {"+": " plus ", "#": " sharp "}
# Letters that look the same in Cyrillic and Latin.
_CYRILLIC_TWINS = "АВЕКМНОРСТУХІЈЅаеорсухіјѕ"
_LATIN_TWINS = "ABEKMHOPCTYXIJSaeopcyxijs"
_TO_LATIN = str.maketrans(_CYRILLIC_TWINS, _LATIN_TWINS)
_TO_CYRILLIC = str.maketrans(_LATIN_TWINS, _CYRILLIC_TWINS)


def _script(char: str) -> Optional[str]:
    if not char.isalpha():
        return None
    name = unicodedata.name(char, "")
    if name.startswith("LATIN"):
        return "latin"
    if name.startswith("CYRILLIC"):
        return "cyrillic"
    return "other"


def _distinct(token: str, script: str, twins: str) -> bool:
    return any(_script(char) == script and char not in twins for char in token)


def _fold_twins(token: str, cyrillic_context: bool) -> str:
    """Write a token in one script when look-alike letters mix them.

    Letters unique to one script decide; a token of look-alikes only is
    Latin when capitalized (CO2, C) and follows the label otherwise.
    """
    scripts = {_script(char) for char in token}
    if not scripts & {"latin", "cyrillic"}:
        return token
    cyrillic = _distinct(token, "cyrillic", _CYRILLIC_TWINS)
    latin = _distinct(token, "latin", _LATIN_TWINS)
    if cyrillic and latin:
        return token
    if latin or (not cyrillic and (token.isupper() or not cyrillic_context)):
        return token.translate(_TO_LATIN)
    return token.translate(_TO_CYRILLIC)


def _strip_accents(token: str) -> str:
    return unicodedata.normalize(
        "NFC",
        "".join(
            char
            for char in unicodedata.normalize("NFD", token)
            if unicodedata.category(char) != "Mn"
        ),
    )


def _is_acronym(token: str) -> bool:
    capitals = sum(char.isupper() for char in token)
    return 2 <= len(token) <= _ACRONYM_MAX and capitals >= 2


def _singular(token: str) -> str:
    """Strip an English plural ending; nothing else is lemmatized."""
    if len(token) <= 3 or not token.endswith("s"):
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("sses", "shes", "ches", "xes")):
        return token[:-2]
    if token.endswith(_LATIN_KEEP):
        return token
    return token[:-1]


@lru_cache(maxsize=1)
def _russian():
    import snowballstemmer

    return snowballstemmer.stemmer("russian")


def _cyrillic(token: str) -> str:
    return _russian().stemWord(token)


def _word(token: str, shouted: bool) -> str:
    if not shouted:
        if token.endswith("s") and len(token) > 2 and _is_acronym(token[:-1]):
            token = token[:-1]
        if token.isupper() and _is_acronym(token):
            return token.casefold().replace("ё", "е")
        if _is_acronym(token):
            return token
    token = token.casefold().replace("ё", "е")
    scripts = {_script(char) for char in token} - {None}
    if scripts == {"latin"}:
        return _singular(_strip_accents(token))
    if scripts == {"cyrillic"}:
        return _cyrillic(token)
    return token


@lru_cache(maxsize=262144)
def lexical_tokens(value: str) -> Tuple[str, ...]:
    value = unicodedata.normalize("NFKC", value)
    value = "".join(
        _SYMBOLS.get(char)
        or (" " if unicodedata.category(char)[0] in {"P", "S"} else char)
        for char in value
    )
    # Digits and letters are separate tokens: GPT-4 is GPT4.
    tokens = re.findall(r"\d+|[^\W\d_]+", value)
    cyrillic_context = _distinct(value, "cyrillic", _CYRILLIC_TWINS)
    tokens = [_fold_twins(token, cyrillic_context) for token in tokens]
    words = [token for token in tokens if not token.isdigit()]
    shouted = (
        len(words) > 1
        and all(token.isupper() for token in words)
        and any(len(token) > _ACRONYM_MAX for token in words)
    )
    return tuple(_word(token, shouted) for token in tokens)


def lexical_key(value: str) -> str:
    return " ".join(lexical_tokens(value))


@lru_cache(maxsize=1)
def _iso_codes() -> frozenset:
    from ..core.config import load_catalog

    return frozenset(load_catalog("countries")["iso_alpha2"])


def country_code(value: str) -> Optional[str]:
    """The ISO 3166-1 alpha-2 code the value spells, if it is one."""
    code = value.strip().upper()
    if _COUNTRY_CODE.fullmatch(code) and code in _iso_codes():
        return code
    return None


# One technology can be reported as a Technology, a Method or a Material;
# identity must not depend on which one a document chose.
_FAMILIES = {
    "Technology": "technology",
    "Method": "technology",
    "Material": "technology",
}


def kind_family(kind: object) -> str:
    """The identity family of a concept kind; other kinds are their own."""
    value = str(getattr(kind, "value", kind))
    return _FAMILIES.get(value, value)


def identity_key(value: str, kind: object = None) -> str:
    """The deterministic identity key of a name of a concept kind."""
    if getattr(kind, "value", kind) == "Country":
        code = country_code(value)
        if code is not None:
            return f"iso:{code}"
    return lexical_key(value)
