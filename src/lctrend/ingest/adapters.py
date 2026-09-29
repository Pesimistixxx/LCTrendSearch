from __future__ import annotations

import base64
import hashlib
import html
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ..core.config import load_catalog
from ..core.models import (
    Artifact,
    Chunk,
    Contributor,
    Country,
    DocumentEnvelope,
    DocumentType,
    Domain,
    ExternalId,
    Organization,
    SourceRef,
    stable_id,
)
from ..core.organizations import (
    organization_identity,
    source_organization_type,
)
from ..core.organizations import organization_type as _organization_type
from .connectors import normalize_openalex_work_id


def _source(
    platform: str,
    record_id: str,
    url: str,
    independence_group: Optional[str] = None,
) -> SourceRef:
    settings = load_catalog("sources")["platforms"][platform]
    return SourceRef(
        source_id=f"source:{platform}",
        name=settings["name"],
        source_type=settings["source_type"],
        source_family=settings["source_family"],
        reliability_tier=settings["reliability_tier"],
        record_id=record_id,
        canonical_url=url,
        independence_group=independence_group,
    )


def _project_group(urls: Iterable[Any]) -> Optional[str]:
    """An explicit repository link ties records to one project, not one
    platform.
    """
    for url in urls:
        match = re.search(
            r"https?://github\.com/([^/?#]+/[^/?#]+)", str(url), re.I
        )
        if match:
            return (
                "project:github:"
                + match.group(1).removesuffix(".git").casefold()
            )
    return None


def _catalog_domain(rule: Mapping[str, Any]) -> Domain:
    return Domain(
        domain_id=stable_id("domain", rule["name"]),
        name=rule["name"],
        parent_name=rule["parent_name"],
    )


def _without_parents(domains: Iterable[Domain]) -> List[Domain]:
    """Drop a domain whose subdomain is also present.

    "NLP" already implies "Artificial intelligence"; keeping both invents
    a cross-domain pair (A-9).
    """
    domains = list(domains)
    parents = {
        domain.parent_name.casefold()
        for domain in domains
        if domain.parent_name
    }
    return [
        domain for domain in domains if domain.name.casefold() not in parents
    ]


def _domains_from_values(values: Iterable[Any]) -> List[Domain]:
    """Every catalog domain the text names; several domains are what makes
    a cross-domain signal visible.
    """
    text = " ".join(str(value).casefold() for value in values)
    found: Dict[str, Domain] = {}
    for rule in load_catalog("sources")["domains"]:
        if any(
            re.search(rf"\b{re.escape(alias)}\b", text)
            for alias in rule["aliases"]
        ):
            domain = _catalog_domain(rule)
            found.setdefault(domain.domain_id, domain)
    return _without_parents(found.values())


def _domains_from_topics(
    topics: Iterable[Mapping[str, Any]],
) -> List[Domain]:
    """Catalog domains for source-ranked OpenAlex topics.

    A topic outside the catalog keeps its OpenAlex subfield (under its field)
    instead of being dropped.
    """
    settings = load_catalog("sources")
    fallback = settings["platforms"]["openalex"]["fallback_domain"]
    catalog = {rule["name"].casefold(): rule for rule in settings["domains"]}
    found: Dict[str, Domain] = {}
    for topic in topics:
        values = [topic.get("display_name", "")]
        values.extend(
            (topic.get(key) or {}).get("display_name", "")
            for key in settings["openalex_topic_fields"]
        )
        matched = _domains_from_values(values)
        subfield = topic.get(fallback["name_field"]) or {}
        name = subfield.get("display_name")
        # The subfield is kept when the catalog does not cover it: "ML in
        # Materials Science" is ML *and* materials, not ML alone (A-9).
        covered = {
            value.casefold()
            for domain in matched
            for value in (domain.name, domain.parent_name or "")
        }
        if name and name.casefold() not in covered:
            if name.casefold() in catalog:
                if not matched:
                    matched = [_catalog_domain(catalog[name.casefold()])]
            else:
                matched = matched + [
                    Domain(
                        domain_id=stable_id("domain", name),
                        name=name,
                        parent_name=(
                            topic.get(fallback["parent_field"]) or {}
                        ).get("display_name"),
                        external_ids=[
                            ExternalId(
                                scheme="openalex", value=str(subfield["id"])
                            )
                        ]
                        if subfield.get("id")
                        else [],
                    )
                ]
        for domain in matched:
            found.setdefault(domain.domain_id, domain)
    return _without_parents(found.values())


def _bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# Fields that change between fetches of unchanged content: fetch time,
# search rank and counters. They are observations of the version (stored as
# metrics with their own date), never a new version: a new version means a
# new paid extraction.
VOLATILE_FIELDS = {
    "common": frozenset({"_retrieved_at"}),
    "openalex": frozenset(
        {
            "relevance_score",
            "cited_by_count",
            "counts_by_year",
            "updated_date",
            "fwci",
            "citation_normalized_percentile",
            "cited_by_percentile_year",
            "summary_stats",
            "is_authors_truncated",
        }
    ),
    "github": frozenset(
        {
            "stargazers_count",
            "watchers_count",
            "watchers",
            "subscribers_count",
            "forks_count",
            "forks",
            "network_count",
            "open_issues_count",
            "open_issues",
            "size",
            "score",
            "pushed_at",
            "updated_at",
            # Weekly activity windows slide with the fetch date.
            "commit_activity",
            "contributor_stats",
        }
    ),
    "pypi": frozenset(
        {"downloads", "last_serial", "releases", "vulnerabilities"}
    ),
}


def _without(value: Any, volatile: frozenset) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without(item, volatile)
            for key, item in value.items()
            if key not in volatile
        }
    if isinstance(value, list):
        return [_without(item, volatile) for item in value]
    return value


def _observation_hash(payload: Mapping[str, Any], source: str = "") -> str:
    """Hash of the source record's content.

    Fetch time, search rank and counters are not a version: re-fetching an
    unchanged record must map to the already processed version instead of
    a new one.
    """
    volatile = VOLATILE_FIELDS["common"] | VOLATILE_FIELDS.get(
        source, frozenset()
    )
    observation = _without(dict(payload), volatile)
    return _bytes_hash(
        json.dumps(observation, sort_keys=True, default=str).encode("utf-8")
    )


def _artifact(uri: str, raw: bytes, media_type: str) -> Artifact:
    return Artifact(
        uri=uri,
        sha256=_bytes_hash(raw),
        media_type=media_type,
        byte_length=len(raw),
    )


def _chunk(
    version_id: str, kind: str, text: str, order: int, **locator: Any
) -> Chunk:
    section_path = locator.pop("section_path", [kind])
    return Chunk(
        chunk_id=stable_id("chunk", version_id, kind, order, text),
        kind=kind,
        text=text,
        order=order,
        section_path=section_path,
        locator=locator,
    )


def _markdown_chunks(
    version_id: str,
    kind: str,
    text: str,
    start_order: int = 0,
    max_chars: Optional[int] = None,
    overlap_chars: Optional[int] = None,
) -> List[Chunk]:
    """Keep contiguous original Markdown slices; headings are navigation
    metadata.
    """
    settings = load_catalog("sources")["markdown"]
    max_chars = settings["max_chars"] if max_chars is None else max_chars
    overlap_chars = (
        settings["overlap_chars"] if overlap_chars is None else overlap_chars
    )
    if max_chars <= overlap_chars or overlap_chars < 0:
        raise ValueError("Markdown chunk size must exceed nonnegative overlap")
    headings: List[str] = []
    sections: List[tuple[List[str], int, int]] = []
    section_start = 0
    offset = 0
    for line in text.splitlines(keepends=True):
        match = re.match(r"^(#{1,6})[ \t]+(.+?)[\r\n]*$", line)
        if match:
            if offset > section_start:
                sections.append(([kind, *headings], section_start, offset))
            level = len(match.group(1))
            headings[:] = headings[: level - 1]
            headings.append(match.group(2).strip())
            section_start = offset
        offset += len(line)
    if section_start < len(text):
        sections.append(([kind, *headings], section_start, len(text)))
    chunks: List[Chunk] = []
    for section_path, left, right in sections:
        if not text[left:right].strip():
            continue
        cursor = left
        while cursor < right:
            source_start = max(left, cursor - overlap_chars)
            overlap_size = cursor - source_start
            end = min(right, cursor + max_chars - overlap_size)
            if end < right:
                boundary = max(
                    text.rfind("\n", cursor, end), text.rfind(" ", cursor, end)
                )
                if boundary > cursor + (end - cursor) // 2:
                    end = boundary + 1
            value = text[source_start:end]
            chunks.append(
                _chunk(
                    version_id,
                    kind,
                    value,
                    start_order + len(chunks),
                    section_path=section_path,
                    source_start=source_start,
                    source_end=end,
                    overlap_chars=overlap_size,
                    core_text_start=overlap_size,
                    source_stream_id=stable_id("stream", version_id, kind),
                    source_segments=[
                        {
                            "chunk_start": 0,
                            "chunk_end": len(value),
                            "source_start": source_start,
                            "source_end": end,
                        }
                    ],
                    text_basis="decoded_markdown_source",
                )
            )
            cursor = end
    return chunks


def _pypi_people(value: Any) -> List[str]:
    return [
        name.strip()
        for name in re.split(r"\s*[,;]\s*", str(value or ""))
        if name.strip()
    ]


def _abstract_from_inverted_index(
    index: Optional[Mapping[str, Iterable[int]]],
) -> str:
    if not isinstance(index, Mapping):
        return ""
    positioned = [
        (position, word)
        for word, positions in index.items()
        if isinstance(word, str) and isinstance(positions, (list, tuple))
        for position in positions
        if (
            isinstance(position, int)
            and not isinstance(position, bool)
            and position >= 0
        )
    ]
    return " ".join(word for _, word in sorted(positioned))


def _plain_title(value: Any) -> Optional[str]:
    """OpenAlex titles carry publisher markup (<i>, <sub>, &amp;), which
    leaked into the graph and the UI (A-14)."""
    if not isinstance(value, str):
        return None
    text = html.unescape(re.sub(r"<[^>]{1,200}>", "", value))
    return re.sub(r"\s+", " ", text).strip() or None


def parse_openalex(
    payload: Mapping[str, Any], raw: Optional[bytes] = None
) -> DocumentEnvelope:
    raw = raw or repr(dict(payload)).encode("utf-8")
    openalex_id = normalize_openalex_work_id(str(payload["id"]))
    if not openalex_id.startswith("W"):
        raise ValueError("OpenAlex work record must have a W-prefixed ID")
    doi_value = payload.get("doi") or (payload.get("ids") or {}).get("doi")
    doi = ""
    if doi_value:
        doi_id = normalize_openalex_work_id(str(doi_value))
        if not doi_id.startswith("doi:"):
            raise ValueError("OpenAlex DOI field must contain a DOI")
        doi = doi_id.removeprefix("doi:")
    identity = doi or openalex_id
    document_id = stable_id("document", "openalex", identity)
    version_id = stable_id(
        "version", document_id, _observation_hash(payload, "openalex")
    )
    abstract = _abstract_from_inverted_index(
        payload.get("abstract_inverted_index")
    )
    canonical_url = (
        f"https://doi.org/{doi}"
        if doi
        else f"https://openalex.org/{openalex_id}"
    )

    contributors = []
    organizations: Dict[str, Organization] = {}
    countries: Dict[str, Country] = {}
    for authorship in payload.get("authorships") or []:
        author = authorship.get("author") or {}
        name = author.get("display_name")
        affiliation_ids = []
        for institution in authorship.get("institutions") or []:
            institution_name = institution.get("display_name")
            if not institution_name:
                continue
            external_id = str(institution.get("id") or institution_name)
            country_code = (
                institution.get("country_code") or ""
            ).upper() or None
            organization_type = source_organization_type(
                institution_name,
                load_catalog("sources")["platforms"]["openalex"][
                    "organization_types"
                ].get(institution.get("type"), institution.get("type")),
            )
            # "Intel (United States)" and "Intel (Germany)" are one Intel.
            organization_id, institution_name = organization_identity(
                institution_name, organization_type, "openalex", external_id
            )
            organizations[organization_id] = Organization(
                organization_id=organization_id,
                name=institution_name,
                organization_type=organization_type,
                country_code=country_code,
                role="affiliation",
                external_ids=[
                    ExternalId(scheme="openalex", value=external_id)
                ],
            )
            affiliation_ids.append(organization_id)
            if country_code:
                countries[country_code] = Country(
                    country_id=stable_id("country", country_code),
                    code=country_code,
                    role="affiliation",
                )
        for country_code in authorship.get("countries") or []:
            code = str(country_code).upper()
            countries[code] = Country(
                country_id=stable_id("country", code),
                code=code,
                role="authorship",
            )
        if name:
            author_id = str(author.get("id") or name)
            contributors.append(
                Contributor(
                    contributor_id=stable_id("person", "openalex", author_id),
                    name=name,
                    external_ids=[
                        ExternalId(scheme="openalex", value=author_id)
                    ]
                    + (
                        [ExternalId(scheme="orcid", value=author["orcid"])]
                        if author.get("orcid")
                        else []
                    ),
                    affiliation_ids=affiliation_ids,
                )
            )

    grants = []
    funders = [
        {
            "id": grant.get("funder"),
            "display_name": grant.get("funder_display_name"),
        }
        for grant in payload.get("grants") or []
    ]
    funders.extend(payload.get("funders") or [])
    awards = payload.get("awards") or []
    funders.extend(
        {
            "id": award.get("funder_id"),
            "display_name": award.get("funder_display_name"),
        }
        for award in awards
    )
    for funder in funders:
        name = funder.get("display_name")
        if not name:
            continue
        external_id = str(funder.get("id") or name)
        # "Funder" is a role: Samsung funding a paper is still a company.
        organization_type = source_organization_type(name, "funder")
        organization_id, name = organization_identity(
            name, organization_type, "openalex", external_id
        )
        organizations.setdefault(
            organization_id,
            Organization(
                organization_id=organization_id,
                name=name,
                organization_type=organization_type,
                country_code=(funder.get("country_code") or "").upper()
                or None,
                role="funder",
                external_ids=[
                    ExternalId(scheme="openalex", value=external_id)
                ],
            ),
        )
    for grant in payload.get("grants") or []:
        if grant.get("award_id"):
            grants.append(
                {
                    "funder": grant.get("funder_display_name"),
                    "award_id": grant["award_id"],
                }
            )
    for award in awards:
        if award.get("funder_award_id"):
            grant = {
                "funder": award.get("funder_display_name"),
                "award_id": award["funder_award_id"],
            }
            if grant not in grants:
                grants.append(grant)

    topics = payload.get("topics") or (
        [payload["primary_topic"]] if payload.get("primary_topic") else []
    )
    domains = _domains_from_topics(topics)
    location = payload.get("primary_location") or {}

    identifiers = [ExternalId(scheme="openalex", value=openalex_id)]
    if doi:
        identifiers.append(ExternalId(scheme="doi", value=doi))
    for scheme in ("pmid", "pmcid"):
        value = (payload.get("ids") or {}).get(scheme)
        if value:
            identifiers.append(ExternalId(scheme=scheme, value=str(value)))

    title = _plain_title(payload.get("title")) or _plain_title(
        payload.get("display_name")
    )
    chunks = (
        [
            _chunk(
                version_id,
                "abstract",
                abstract,
                0,
                json_pointer="/abstract_inverted_index",
            ),
            # A paper often names its method only in the title ("YOLOv4:
            # ..." whose abstract never says YOLOv4): the title is text an
            # entity can cite. After the abstract, so abstract chunk ids of
            # processed works do not change; only with an abstract, so a
            # card without text is not sent to the model for its title.
            *(
                [_chunk(version_id, "title", title, 1, json_pointer="/title")]
                if title
                else []
            ),
        ]
        if abstract
        else []
    )
    return DocumentEnvelope(
        document_id=document_id,
        document_version_id=version_id,
        document_type=DocumentType.ARTICLE,
        title=title or identity,
        language=payload.get("language"),
        published_at=payload.get("publication_date"),
        version_published_at=payload.get("publication_date"),
        retrieved_at=payload.get("_retrieved_at"),
        metrics_observed_at=payload.get("_retrieved_at"),
        source=_source(
            "openalex",
            openalex_id,
            str(canonical_url),
            f"work:doi:{doi.casefold()}" if doi else None,
        ),
        artifact=_artifact(str(canonical_url), raw, "application/json"),
        identifiers=identifiers,
        contributors=contributors,
        organizations=list(organizations.values()),
        countries=list(countries.values()),
        domains=domains,
        chunks=chunks,
        metadata={
            "type": payload.get("type"),
            "topics": topics,
            "venue": (location.get("source") or {}).get("display_name"),
            "venue_type": (location.get("source") or {}).get("type"),
            "open_access": (payload.get("open_access") or {}).get("oa_status"),
            "is_retracted": bool(payload.get("is_retracted")),
            "grants": grants,
            "awards": awards,
            "counts_by_year": payload.get("counts_by_year") or [],
            "referenced_works": payload.get("referenced_works") or [],
        },
        metrics={
            **(
                {"fwci": float(payload["fwci"])}
                if isinstance(payload.get("fwci"), (int, float))
                else {}
            ),
            "citation_count": float(payload.get("cited_by_count") or 0),
            "reference_count": float(
                payload.get("referenced_works_count")
                if payload.get("referenced_works_count") is not None
                else len(payload.get("referenced_works") or [])
            ),
            "authorship_count": float(len(payload.get("authorships") or [])),
        },
        coverage="abstract_only" if abstract else "metadata_only",
    )


def _week_date(timestamp: Any) -> Optional[str]:
    try:
        moment = datetime.fromtimestamp(int(timestamp), timezone.utc)
        return moment.date().isoformat()
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _github_history(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Dated repository activity: release dates, weekly commit totals and
    the first commit week of every contributor.
    """
    commit_weeks = {
        week: int(item.get("total") or 0)
        for item in payload.get("commit_activity") or []
        if isinstance(item, Mapping)
        for week in [_week_date(item.get("week"))]
        if week
    }
    first_weeks = []
    for author in payload.get("contributor_stats") or []:
        if not isinstance(author, Mapping):
            continue
        active = [
            week
            for item in author.get("weeks") or []
            if isinstance(item, Mapping) and int(item.get("c") or 0) > 0
            for week in [_week_date(item.get("w"))]
            if week
        ]
        if active:
            first_weeks.append(min(active))
    history: Dict[str, Any] = {}
    if isinstance(payload.get("releases"), list):
        history["release_dates"] = sorted(
            str(release["published_at"])
            for release in payload.get("releases") or []
            if isinstance(release, Mapping) and release.get("published_at")
        )
    if isinstance(payload.get("commit_activity"), list):
        history["commit_weeks"] = dict(sorted(commit_weeks.items()))
    if isinstance(payload.get("contributor_stats"), list):
        history["contributor_first_weeks"] = sorted(first_weeks)
    return history


_LINK_LINE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+.*\[[^\]]+\]\([^)]+\)")


def link_catalog(markdown: str) -> bool:
    """Whether a README is mostly list rows with a link (an awesome list)."""
    settings = load_catalog("sources")["platforms"]["github"]
    lines = [line for line in markdown.splitlines() if line.strip()]
    links = sum(bool(_LINK_LINE.match(line)) for line in lines)
    return links >= int(settings["catalog_min_link_lines"]) and links >= float(
        settings["catalog_min_link_share"]
    ) * len(lines)


def parse_github(
    payload: Mapping[str, Any], raw: Optional[bytes] = None
) -> DocumentEnvelope:
    repo = payload.get("repository") or payload
    raw = raw or repr(dict(payload)).encode("utf-8")
    repo_id = str(repo.get("node_id") or repo.get("id") or repo["full_name"])
    document_id = stable_id("document", "github", repo_id)
    commit = payload.get("commit") or {}
    commit_sha = commit.get("sha")
    commit_data = commit.get("commit") or {}
    content_date = (commit_data.get("committer") or {}).get("date")
    readme = payload.get("readme") if "repository" in payload else None
    content = None
    if isinstance(readme, Mapping):
        content = readme.get("text")
        if not content and readme.get("content"):
            content = base64.b64decode(readme["content"]).decode(
                "utf-8", errors="replace"
            )
    # Counters are dated metric observations of the version, not content:
    # a new star must not pay for a new extraction.
    version_id = stable_id(
        "version",
        document_id,
        commit_sha,
        _observation_hash(payload, "github"),
    )
    canonical_url = (
        repo.get("html_url")
        or f"{load_catalog('sources')['platforms']['github']['public_base']}"
        f"/{repo['full_name']}"
    )
    chunks: List[Chunk] = []
    organizations: List[Organization] = []
    warnings: List[str] = []

    if isinstance(readme, Mapping):
        if content and link_catalog(content):
            # An awesome-style list: one "library - one line" per row; the
            # model made each row a technology (sources.json catalog_note).
            warnings.append("readme_link_catalog_skipped")
        elif content:
            readme_chunks = _markdown_chunks(
                version_id, "readme", content, len(chunks)
            )
            for chunk in readme_chunks:
                chunk.locator.update(
                    commit_sha=commit_sha,
                    path=readme.get("path", "README.md"),
                    observed_at=content_date,
                    retrieved_at=payload.get("_retrieved_at"),
                    source_stream_id=stable_id(
                        "stream",
                        version_id,
                        "readme",
                        readme.get("path", "README.md"),
                    ),
                )
            chunks.extend(readme_chunks)

    for release in payload.get("releases") or []:
        body = release.get("body") or ""
        if body:
            release_chunks = _markdown_chunks(
                version_id, "release", body, len(chunks)
            )
            for chunk in release_chunks:
                chunk.locator.update(
                    release_id=release.get("id"),
                    tag=release.get("tag_name"),
                    observed_at=release.get("published_at"),
                    retrieved_at=payload.get("_retrieved_at"),
                    content_date_status=(
                        "release_body_snapshot; "
                        "publication_date_does_not_prove_text_immutability"
                    ),
                    source_stream_id=stable_id(
                        "stream",
                        version_id,
                        "release",
                        release.get("id"),
                        release.get("tag_name"),
                        _bytes_hash(body.encode("utf-8")),
                    ),
                )
            chunks.extend(release_chunks)

    owner = repo.get("owner") or {}
    contributors = []
    if owner.get("login"):
        owner_key = owner.get("node_id") or owner["login"]
        if owner.get("type") == "Organization":
            organization_type = _organization_type(owner["login"])
            organizations.append(
                Organization(
                    organization_id=organization_identity(
                        owner["login"], organization_type, "github", owner_key
                    )[0],
                    name=owner["login"],
                    organization_type=organization_type,
                    role="owner",
                    external_ids=[
                        ExternalId(scheme="github", value=str(owner_key))
                    ],
                )
            )
        else:
            contributors.append(
                Contributor(
                    contributor_id=stable_id("person", "github", owner_key),
                    name=owner["login"],
                    role="owner",
                    external_ids=[
                        ExternalId(scheme="github", value=str(owner_key))
                    ],
                )
            )

    return DocumentEnvelope(
        document_id=document_id,
        document_version_id=version_id,
        document_type=DocumentType.REPOSITORY,
        title=repo.get("name") or repo["full_name"],
        language=repo.get("language"),
        published_at=repo.get("created_at"),
        version_published_at=content_date,
        retrieved_at=payload.get("_retrieved_at"),
        metrics_observed_at=payload.get("_retrieved_at"),
        source=_source(
            "github", repo_id, canonical_url, _project_group([canonical_url])
        ),
        artifact=_artifact(canonical_url, raw, "application/json"),
        identifiers=[ExternalId(scheme="github", value=repo_id)],
        contributors=contributors,
        organizations=organizations,
        domains=_domains_from_values(
            [
                repo.get("name") or "",
                repo.get("description") or "",
                *(repo.get("topics") or []),
            ]
        ),
        chunks=chunks,
        metadata={
            "full_name": repo.get("full_name"),
            **({"parse_warnings": warnings} if warnings else {}),
            "commit_sha": commit_sha,
            "content_date_status": "commit_metadata"
            if content_date
            else "unavailable",
            "description": repo.get("description"),
            "topics": repo.get("topics") or [],
            "license": (repo.get("license") or {}).get("spdx_id"),
            "stars": repo.get("stargazers_count"),
            "fork": bool(repo.get("fork")),
            **_github_history(payload),
        },
        metrics={
            "stars": float(repo.get("stargazers_count") or 0),
            "forks": float(repo.get("forks_count") or 0),
            "watchers": float(repo.get("subscribers_count") or 0),
            "open_issues": float(repo.get("open_issues_count") or 0),
        },
        coverage="selected_files" if chunks else "metadata_only",
    )


def parse_pypi(
    payload: Mapping[str, Any], raw: Optional[bytes] = None
) -> DocumentEnvelope:
    raw = raw or repr(dict(payload)).encode("utf-8")
    info = payload["info"]
    name = info["name"]
    version = info.get("version") or "unknown"
    document_id = stable_id("document", "pypi", name.lower())
    version_id = stable_id(
        "version", document_id, version, _observation_hash(payload, "pypi")
    )
    canonical_url = (
        info.get("package_url")
        or f"{load_catalog('sources')['platforms']['pypi']['public_base']}"
        f"/{name}/"
    )
    uploaded_at = sorted(
        item["upload_time_iso_8601"]
        for item in payload.get("urls") or []
        if item.get("upload_time_iso_8601")
    )
    # First upload of every release: the package's history is dated, so a
    # snapshot can count releases known at its date. The project exists
    # since its first release, not since the current version's upload.
    release_dates = sorted(
        min(uploads)
        for uploads in (
            [
                item["upload_time_iso_8601"]
                for item in files or []
                if item.get("upload_time_iso_8601")
            ]
            for files in (payload.get("releases") or {}).values()
        )
        if uploads
    )
    first_release = min([*release_dates, *uploaded_at[:1]], default=None)
    domains = _domains_from_values(
        [name, info.get("summary") or "", *(info.get("classifiers") or [])]
    )
    country_code = str(
        info.get("country_code") or info.get("country") or ""
    ).upper()
    countries = (
        [
            Country(
                country_id=stable_id("country", country_code),
                code=country_code,
                role="metadata",
            )
        ]
        if re.fullmatch(r"[A-Z]{2}", country_code)
        else []
    )
    description = info.get("description") or info.get("summary") or ""
    chunks = (
        _markdown_chunks(version_id, "description", description)
        if description
        else []
    )
    for chunk in chunks:
        chunk.locator["json_pointer"] = "/info/description"
    contributors = []
    seen = set()
    for role, field in load_catalog("sources")["platforms"]["pypi"][
        "people_fields"
    ].items():
        for person_name in _pypi_people(info.get(field)):
            key = (person_name.casefold(), role)
            if key in seen:
                continue
            seen.add(key)
            contributors.append(
                Contributor(
                    contributor_id=stable_id(
                        "person", "pypi", person_name.casefold()
                    ),
                    name=person_name,
                    role=role,
                )
            )

    return DocumentEnvelope(
        document_id=document_id,
        document_version_id=version_id,
        document_type=DocumentType.PACKAGE,
        title=name,
        published_at=first_release,
        version_published_at=uploaded_at[0] if uploaded_at else None,
        retrieved_at=payload.get("_retrieved_at"),
        metrics_observed_at=payload.get("_retrieved_at"),
        source=_source(
            "pypi",
            name,
            canonical_url,
            _project_group((info.get("project_urls") or {}).values()),
        ),
        artifact=_artifact(canonical_url, raw, "application/json"),
        identifiers=[ExternalId(scheme="pypi", value=name)],
        contributors=contributors,
        countries=countries,
        domains=domains,
        chunks=chunks,
        metadata={
            "version": version,
            "summary": info.get("summary"),
            "classifiers": info.get("classifiers") or [],
            "project_urls": info.get("project_urls") or {},
            "requires_python": info.get("requires_python"),
            "country_status": "provided" if countries else "unavailable",
            "release_dates": release_dates,
        },
        metrics={"release_file_count": float(len(payload.get("urls") or []))},
        coverage="full_text" if chunks else "metadata_only",
    )


def _local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _elements(root: ET.Element, name: str) -> Iterable[ET.Element]:
    return (element for element in root.iter() if _local_name(element) == name)


def _first_text(root: ET.Element, name: str) -> Optional[str]:
    for element in _elements(root, name):
        text = " ".join("".join(element.itertext()).split())
        if text:
            return text
    return None


def _name_country(value: str) -> Tuple[str, Optional[str]]:
    """EPO epodoc names carry residence: "SIEMENS AG [DE]"."""
    match = re.fullmatch(r"(.*?)[\s,]*\[([A-Z]{2})\]\s*", value)
    if match:
        return match.group(1).strip(), match.group(2)
    return value.strip(), None


def _party_names(
    root: ET.Element, tag: str
) -> List[Tuple[str, Optional[str]]]:
    settings = load_catalog("sources")["platforms"]["epo"]
    elements = list(_elements(root, tag))
    preferred = [
        element
        for element in elements
        if element.get("data-format") == settings["preferred_name_format"]
    ]
    names: Dict[str, Tuple[str, Optional[str]]] = {}
    for element in preferred or elements:
        name = next(
            (
                " ".join("".join(candidate.itertext()).split())
                for candidate in element.iter()
                if _local_name(candidate) in settings["name_tags"]
                and "".join(candidate.itertext()).strip()
            ),
            None,
        )
        if name:
            name, country = _name_country(name)
            key = re.sub(r"[\W_]+", " ", name.casefold()).strip()
            if key not in names or (country and not names[key][1]):
                names[key] = (name, country)
    return list(names.values())


def _publication_date(root: ET.Element) -> Optional[str]:
    """Publication, not priority or application, date."""
    reference = next(_elements(root, "publication-reference"), None)
    date = (
        _first_text(reference, "date") if reference is not None else None
    ) or _first_text(root, "date")
    if date and re.fullmatch(r"\d{8}", date):
        date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    return date


def _priority_date(root: ET.Element) -> Optional[str]:
    dates = []
    for claim in _elements(root, "priority-claim"):
        value = _first_text(claim, "date")
        if value and re.fullmatch(r"\d{8}", value):
            dates.append(f"{value[:4]}-{value[4:6]}-{value[6:]}")
    return min(dates, default=None)


def parse_epo(
    xml: str,
    uri: str = "local://epo.xml",
    retrieved_at: Optional[str] = None,
) -> DocumentEnvelope:
    raw = xml.encode("utf-8")
    root = ET.fromstring(xml)
    settings = load_catalog("sources")["platforms"]["epo"]
    reference = next(_elements(root, "publication-reference"), root)
    country = _first_text(reference, "country") or ""
    doc_number = _first_text(reference, "doc-number") or _first_text(
        root, "publication-reference"
    )
    if not doc_number:
        raise ValueError("EPO XML has no publication number")
    kind = _first_text(reference, "kind") or ""
    publication_id = "".join(
        part for part in (country, doc_number, kind) if part
    )
    document_id = stable_id("document", "epo", publication_id)
    version_id = stable_id("version", document_id, _bytes_hash(raw))
    title = _first_text(root, "invention-title") or publication_id
    abstract_parts = []
    language = None
    for abstract in _elements(root, "abstract"):
        text = " ".join("".join(abstract.itertext()).split())
        if text and text not in abstract_parts:
            abstract_parts.append(text)
            language = language or abstract.get("lang")
    abstract = "\n".join(abstract_parts)
    chunks = (
        [
            _chunk(
                version_id,
                "abstract",
                abstract,
                0,
                xpath="//*[local-name()='abstract']",
            )
        ]
        if abstract
        else []
    )
    # Full-text OPS responses add claims and description; biblio has none.
    for section, item_tag in settings["text_sections"].items():
        container = next(_elements(root, section), None)
        if container is None:
            continue
        items = [
            " ".join("".join(item.itertext()).split())
            for item in _elements(container, item_tag)
        ]
        text = "\n\n".join(item for item in items if item)
        if not text:
            continue
        section_chunks = _markdown_chunks(
            version_id, section, text, len(chunks)
        )
        for chunk in section_chunks:
            chunk.locator.update(
                xpath=f"//*[local-name()='{section}']",
                text_basis="epo_xml_item_text_joined",
            )
        chunks.extend(section_chunks)

    inventors = _party_names(root, settings["people_tags"]["inventor"])
    inventor_keys = {
        re.sub(r"[\W_]+", " ", name.casefold()).strip()
        for name, _ in inventors
    }
    contributors = [
        Contributor(
            contributor_id=stable_id("person", "epo", name),
            name=name,
            role="inventor",
        )
        for name, _ in inventors
    ]
    organizations = []
    countries: Dict[str, Country] = {}
    iso = set(load_catalog("countries")["iso_alpha2"])
    # EP, WO, EA... are patent offices, not countries (A-7).
    office = country.upper() if country.upper() not in iso else None
    if country and not office:
        countries[country.upper()] = Country(
            country_id=stable_id("country", country.upper()),
            code=country.upper(),
            role="jurisdiction",
        )
    for name, residence in _party_names(
        root, settings["people_tags"]["applicant"]
    ):
        key = re.sub(r"[\W_]+", " ", name.casefold()).strip()
        if key in inventor_keys:
            # An inventor filing in their own name is a person, not a company.
            contributors.append(
                Contributor(
                    contributor_id=stable_id("person", "epo", name),
                    name=name,
                    role="applicant",
                )
            )
            continue
        organization_type = _organization_type(name)
        organizations.append(
            Organization(
                organization_id=organization_identity(
                    name, organization_type, "epo", name
                )[0],
                name=name,
                organization_type=organization_type,
                country_code=residence if residence in iso else None,
                role="applicant",
            )
        )
        if residence in iso:
            countries.setdefault(
                residence,
                Country(
                    country_id=stable_id("country", residence),
                    code=residence,
                    role="applicant_residence",
                ),
            )

    date = _publication_date(root)

    return DocumentEnvelope(
        document_id=document_id,
        document_version_id=version_id,
        document_type=DocumentType.PATENT,
        title=title,
        language=language,
        published_at=date,
        version_published_at=date,
        retrieved_at=retrieved_at,
        source=_source("epo", publication_id, uri),
        artifact=_artifact(uri, raw, "application/xml"),
        identifiers=[
            ExternalId(scheme="patent-publication", value=publication_id)
        ],
        contributors=contributors,
        organizations=organizations,
        countries=list(countries.values()),
        domains=_domains_from_values([title, abstract]),
        chunks=chunks,
        metadata={
            "country": country or None,
            "patent_office": office,
            "kind": kind or None,
            "classifications": [
                " ".join("".join(item.itertext()).split())
                for item in _elements(root, "classification-ipcr")
            ],
            # A family groups filings of one invention; priority dates the
            # invention itself (patent age), publication dates the document.
            "family_id": next(
                (
                    element.get("family-id")
                    for element in root.iter()
                    if element.get("family-id")
                ),
                None,
            ),
            "priority_date": _priority_date(root),
        },
        coverage=(
            "full_text"
            if len(chunks) > (1 if abstract else 0)
            else "abstract_only"
            if chunks
            else "metadata_only"
        ),
    )
