"""Search topics for thematic crawls, proposed by the language model.

The model only proposes OpenAlex/GitHub search queries around a direction;
nothing it says becomes graph data. Queries are cleaned, deduplicated
against each other and against topics already crawled, and capped.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

from pydantic import BaseModel, ConfigDict, Field

MAX_TOPICS = 30
MAX_QUERY_CHARS = 120

SYSTEM = """You plan bulk literature crawls for a technology knowledge
graph. The goal is coverage: each query should pull as many relevant
papers from the OpenAlex scholarly API and repositories from GitHub search
as possible. Given a direction, propose distinct search queries.

Rules:
- Each query is 1-4 words in English and names a broad, well-established
  field or major sub-field with a large body of literature and code (e.g.
  "machine learning", "computer vision", "natural language processing",
  "reinforcement learning", "robotics", "computer networks"). Never a
  narrow niche, a specific product, an application to one industry, or a
  combination of two fields ("homomorphic encryption finance" is wrong).
- If the direction is itself a field (e.g. "ML"), return its main
  sub-fields; if it is "any" / "любое" / very general, return the major
  domains of computer science and engineering.
- Order from the largest and most central field to smaller ones; pick
  the canonical names, not exotic or speculative ones.
- Queries must not duplicate each other or the topics in "exclude", even
  as synonyms or abbreviations.
- "why" is one short sentence in Russian: what the field covers.
- Return exactly "count" topics unless the direction has fewer
  sub-fields.
Return JSON: {"topics": [{"query": "...", "why": "..."}]}"""


class SuggestedTopic(BaseModel):
    model_config = ConfigDict(extra="ignore")
    query: str = Field(min_length=1, max_length=400)
    why: str = Field(default="", max_length=1000)


class TopicSuggestions(BaseModel):
    model_config = ConfigDict(extra="ignore")
    topics: List[SuggestedTopic] = Field(default_factory=list)


def _key(query: str) -> str:
    return re.sub(r"[\W_]+", " ", query.casefold()).strip()


def clean_topics(
    topics: Iterable[SuggestedTopic],
    count: int,
    exclude: Iterable[str] = (),
) -> List[dict]:
    """Trim, drop empty/overlong/repeated queries, keep at most ``count``."""
    seen = {_key(item) for item in exclude if _key(item)}
    output = []
    for topic in topics:
        query = re.sub(r"\s+", " ", topic.query).strip(" \"'.,;")
        key = _key(query)
        if not key or len(query) > MAX_QUERY_CHARS or key in seen:
            continue
        seen.add(key)
        output.append({"query": query, "why": topic.why.strip()})
        if len(output) >= count:
            break
    return output


async def suggest_topics(
    provider,
    direction: str,
    count: int = 10,
    exclude: Optional[Iterable[str]] = None,
) -> List[dict]:
    """Ask the model for ``count`` search topics around ``direction``."""
    direction = direction.strip()
    if not direction:
        raise ValueError("direction must not be empty")
    count = max(1, min(MAX_TOPICS, int(count)))
    exclude = [item for item in (exclude or []) if item and item.strip()]
    answer = await provider.generate(
        TopicSuggestions,
        SYSTEM,
        {
            "direction": direction,
            "count": count,
            # The most recent topics are enough to steer away from repeats.
            "exclude": exclude[:100],
        },
        # The short review budget fits a list of queries.
        stage="review",
    )
    return clean_topics(answer.topics, count, exclude)
