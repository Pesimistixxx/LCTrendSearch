"""Lexical identity key v2 of concept names.

The key decides which surface forms name the same concept, so it must
neither merge unrelated names nor split inflected forms of one name:

- text is NFKC-normalized and split on punctuation and symbols;
- acronyms written in capitals (AI, ИИ, GAN, LED) are never lemmatized;
  short mixed-case acronyms keep their case (GaN is not GAN);
- Latin words only lose a plural ending;
- a country is identified by its ISO 3166-1 code alone.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from functools import lru_cache
from typing import Optional, Tuple

KEY_VERSION = "lexical-key/2"

# Longer all-capital words mean a shouted label, not an acronym.
_ACRONYM_MAX = 5
_LATIN_KEEP = ("ss", "us", "is", "ics", "as")
_COUNTRY_CODE = re.compile(r"[A-Za-z]{2}")

_SIMPLEMMA_MISSING = False


def _lemmatizer():
    """The optional simplemma function; a failed import is tried only once."""
    global _SIMPLEMMA_MISSING
    module = sys.modules.get("simplemma")
    if module is None and not _SIMPLEMMA_MISSING:
        try:
            import simplemma as module
        except ImportError:
            _SIMPLEMMA_MISSING = True
    return getattr(module, "lemmatize", None)


def _script(char: str) -> Optional[str]:
    if not char.isalpha():
        return None
    name = unicodedata.name(char, "")
    if name.startswith("LATIN"):
        return "latin"
    if name.startswith("CYRILLIC"):
        return "cyrillic"
    return "other"


def _is_acronym(token: str) -> bool:
    capitals = sum(char.isupper() for char in token)
    return len(token) >= 2 and (
        token.isupper() or capitals >= 2 and len(token) <= _ACRONYM_MAX
    )


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


def _cyrillic(token: str) -> str:
    lemmatize = _lemmatizer()
    return lemmatize(token, lang="ru") if lemmatize else token


def _word(token: str, shouted: bool) -> str:
    if not shouted:
        if (
            token.endswith("s")
            and len(token) > 2
            and _is_acronym(token[:-1])
        ):
            token = token[:-1]
        if token.isupper() and len(token) >= 2:
            return token.casefold()
        if _is_acronym(token):
            return token
    token = token.casefold()
    scripts = {_script(char) for char in token} - {None}
    if scripts == {"latin"}:
        return _singular(token)
    if scripts == {"cyrillic"}:
        return _cyrillic(token)
    return token


@lru_cache(maxsize=262144)
def lexical_tokens(value: str) -> Tuple[str, ...]:
    value = unicodedata.normalize("NFKC", value)
    value = "".join(
        " " if unicodedata.category(char)[0] in {"P", "S"} else char
        for char in value
    )
    tokens = re.findall(r"[^\W_]+", value)
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


def identity_key(value: str, kind: object = None) -> str:
    """The deterministic identity key of a name of a concept kind."""
    if getattr(kind, "value", kind) == "Country":
        code = country_code(value)
        if code is not None:
            return f"iso:{code}"
    return lexical_key(value)
