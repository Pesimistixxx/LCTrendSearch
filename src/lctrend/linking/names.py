"""Known names of the registry found in a text.

Names are compared by lexical identity tokens (extraction.lexical), the
same key the resolver uses, so "graph neural networks" in a text finds the
concept "Graph Neural Network" and "графовыми нейронными сетями" finds
"графовые нейронные сети". A match covers whole words ("GPT4" is not
"GPT"); an umbrella term ("machine learning") is never a name to find.
"""

from __future__ import annotations

import re
from typing import Collection, Dict, Iterable, List, Optional, Tuple

from ..core.models import Concept
from ..extraction.lexical import kind_family, lexical_tokens
from ..extraction.resolver import _aliases

# A name shorter than this (letters) matches too many unrelated words.
MIN_LETTERS = 3
INACTIVE = frozenset({"merged", "rejected"})
_WORD = re.compile(r"[^\W_]+[+#]*")

Token = Tuple[str, int, int, bool, bool]


def tokens(text: str) -> List[Token]:
    """Identity tokens with the span of their word; the flags mark the
    first and last token of a word, so a match never splits one."""
    found: List[Token] = []
    for word in _WORD.finditer(text):
        parts = lexical_tokens(word.group())
        for position, part in enumerate(parts):
            found.append(
                (
                    part,
                    word.start(),
                    word.end(),
                    position == 0,
                    position == len(parts) - 1,
                )
            )
    return found


def _specific(name: str, key: Tuple[str, ...]) -> bool:
    from ..llm.validation import generic_technology

    letters = sum(len(token) for token in key if not token.isdigit())
    return letters >= MIN_LETTERS and not generic_technology(name)


class NameIndex:
    """Reviewed names of active registry concepts, by identity tokens.

    ``families`` limits the concepts to kind families (extraction.lexical:
    Technology, Method and Material are the "technology" family); None
    keeps every kind.
    """

    def __init__(
        self,
        concepts: Iterable[Concept],
        families: Optional[Collection[str]] = None,
    ) -> None:
        self.names: Dict[Tuple[str, ...], set] = {}
        self.concepts: Dict[str, Concept] = {}
        for concept in concepts:
            if concept.status in INACTIVE or (
                families is not None
                and kind_family(concept.kind) not in families
            ):
                continue
            self.concepts[concept.concept_id] = concept
            for name in _aliases(concept):
                key = tuple(token[0] for token in tokens(name))
                if key and _specific(name, key):
                    self.names.setdefault(key, set()).add(concept.concept_id)
        self.longest = max((len(key) for key in self.names), default=0)

    def __bool__(self) -> bool:
        return bool(self.names)

    def find(self, text: str) -> List[Tuple[str, int, int]]:
        """(concept_id, start, end) of the longest known names in the text.

        A name several concepts share is consumed but links nothing: the
        text alone cannot say which one it means.
        """
        found: List[Tuple[str, int, int]] = []
        words = tokens(text)
        position = 0
        while position < len(words):
            for size in range(min(self.longest, len(words) - position), 0, -1):
                window = words[position : position + size]
                owners = self.names.get(tuple(token[0] for token in window))
                if owners and window[0][3] and window[-1][4]:
                    if len(owners) == 1:
                        found.append(
                            (next(iter(owners)), window[0][1], window[-1][2])
                        )
                    position += size
                    break
            else:
                position += 1
        return found
