"""Names of organizations and countries shared by the adapters, the
extraction gates and the graph writer.

A company is one organization wherever it is registered: OpenAlex splits
it by country ("Intel (United States)", "Intel (Germany)"), a patent
names its legal form ("INTEL CORP"), a GitHub owner is a login ("intel").
All of them get one identity, keyed by the name without the country
suffix and the legal form.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, Optional, Tuple

from .config import load_catalog
from .models import stable_id

# Source types that say nothing about the kind of organization ("funder"
# is a role); the name decides them (Samsung as a funder is a company).
UNTYPED = frozenset({"other", "funder", "nonprofit", "facility", "archive"})

_SUFFIX = re.compile(r"^(?P<name>.*\S)\s*\((?P<inner>[^()]+)\)\s*$")


def _words(value: str) -> str:
    return " " + " ".join(re.findall(r"\w+", value.casefold())) + " "


def organization_type(name: str, default: str = "other") -> str:
    """Type an organization the source left untyped by its name.

    An explicit university marker wins; then known names (MIT, IBM, Сбер)
    that carry no legal form or university marker; then the legal form.
    GitHub logins ("sberbank-ai") are split into words.
    """
    catalog = load_catalog("sources")
    lowered = name.casefold()
    words = _words(name)
    patterns = catalog["organization_type_patterns"]

    def marked(kind: str) -> bool:
        pattern = patterns.get(kind)
        return bool(
            pattern
            and (re.search(pattern, lowered) or re.search(pattern, words))
        )

    if marked("university"):
        return "university"
    for kind, names in catalog.get("known_organizations", {}).items():
        for known in names:
            alias = _words(known).strip()
            if alias and f" {alias} " in words:
                return kind
    for kind in patterns:
        if marked(kind):
            return kind
    return default


def source_organization_type(name: str, source_type: Optional[str]) -> str:
    """A source's type, unless it is a role or says nothing."""
    source_type = source_type or "other"
    if source_type in UNTYPED:
        return organization_type(name, default=source_type)
    return source_type


@lru_cache(maxsize=1)
def _country_keys() -> Dict[str, str]:
    from ..extraction.lexical import lexical_key

    keys = {}
    for code, names in load_catalog("countries")["names"].items():
        for name in names:
            key = lexical_key(name)
            if key:
                keys.setdefault(key, code)
    return keys


def country_code_of(name: Optional[str]) -> Optional[str]:
    """The ISO code a country name spells ("UNITED STATES", "Германия")."""
    from ..extraction.lexical import country_code, lexical_key

    if not name:
        return None
    return country_code(name) or _country_keys().get(lexical_key(name))


def country_suffix(name: str) -> Tuple[str, Optional[str]]:
    """Split "Intel (United States)" into ("Intel", "US")."""
    from ..extraction.lexical import lexical_key

    match = _SUFFIX.match(name.strip())
    if match:
        code = _country_keys().get(lexical_key(match["inner"]))
        if code:
            return match["name"].strip(), code
    return name.strip(), None


def company_key(name: str) -> str:
    """The identity of a company: its name without country and legal form."""
    base, _ = country_suffix(name)
    pattern = load_catalog("sources")["organization_type_patterns"]["company"]
    words = re.sub(pattern, " ", _words(base))
    key = " ".join(re.findall(r"\w+", words))
    return key or _words(base).strip()


def display_rank(name: str) -> int:
    """Lower is a better display name: written in mixed case, then short.

    "Intel" beats "intel" (a login) and "INTEL CORP" (a patent register).
    """
    letters = [char for char in name if char.isalpha()]
    uniform = bool(letters) and (
        all(char.isupper() for char in letters)
        or all(char.islower() for char in letters)
    )
    return (1000 if uniform else 0) + len(name)


def organization_identity(
    name: str, organization_type: str, source: str, external_id: str
) -> Tuple[str, str]:
    """(organization_id, display name) of an organization from a source.

    A company is identified by its name across sources and countries;
    other organizations keep the source's own identifier, since the same
    name can mean different bodies ("Ministry of Health (Brazil)").
    """
    if organization_type == "company":
        base, _ = country_suffix(name)
        return stable_id("organization", "company", company_key(base)), base
    return stable_id("organization", source, external_id), name.strip()


def country_names(code: str) -> Tuple[str, str]:
    """(Russian, English) full name of an ISO 3166-1 alpha-2 code.

    Names are listed English first, then the Russian CLDR name; an
    unknown code is its own name.
    """
    names = load_catalog("countries")["names"].get(code) or []
    english = names[0] if names else code
    russian = next(
        (name for name in names if re.search("[а-яё]", name, re.I)),
        english,
    )
    return russian, english
