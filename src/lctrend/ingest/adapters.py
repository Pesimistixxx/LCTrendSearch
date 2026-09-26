from __future__ import annotations

import base64
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, Iterable, List, Mapping, Optional

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


def _domain_from_values(values: Iterable[Any]) -> Optional[Domain]:
    text = " ".join(str(value).casefold() for value in values)
    for rule in load_catalog("sources")["domains"]:
        if any(
            re.search(rf"\b{re.escape(alias)}\b", text)
            for alias in rule["aliases"]
        ):
            return Domain(
                domain_id=stable_id("domain", rule["name"]),
                name=rule["name"],
                parent_name=rule["parent_name"],
            )
    return None


def _domain_from_topics(
    topics: Iterable[Mapping[str, Any]],
) -> Optional[Domain]:
    """Choose one canonical broad domain from source-ranked OpenAlex topics."""
    for topic in topics:
        values = [topic.get("display_name", "")]
        values.extend(
            (topic.get(key) or {}).get("display_name", "")
            for key in load_catalog("sources")["openalex_topic_fields"]
        )
        domain = _domain_from_values(values)
        if domain:
            return domain
    return None


def _bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
    if not index:
        return ""
    positioned = [
        (position, word)
        for word, positions in index.items()
        for position in positions
    ]
    return " ".join(word for _, word in sorted(positioned))


def parse_openalex(
    payload: Mapping[str, Any], raw: Optional[bytes] = None
) -> DocumentEnvelope:
    raw = raw or repr(dict(payload)).encode("utf-8")
    openalex_id = str(payload["id"]).rsplit("/", 1)[-1]
    doi = str(payload.get("doi") or "").removeprefix("https://doi.org/")
    identity = doi or openalex_id
    document_id = stable_id("document", "openalex", identity)
    version_id = stable_id("version", document_id, _bytes_hash(raw))
    abstract = _abstract_from_inverted_index(
        payload.get("abstract_inverted_index")
    )
    canonical_url = payload.get("doi") or payload.get("id")

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
            organization_id = stable_id(
                "organization", "openalex", external_id
            )
            country_code = (
                institution.get("country_code") or ""
            ).upper() or None
            organization_type = load_catalog("sources")["platforms"][
                "openalex"
            ]["organization_types"].get(
                institution.get("type"), institution.get("type") or "other"
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
                    ],
                    affiliation_ids=affiliation_ids,
                )
            )

    domain = _domain_from_topics(payload.get("topics") or [])

    identifiers = [ExternalId(scheme="openalex", value=openalex_id)]
    if doi:
        identifiers.append(ExternalId(scheme="doi", value=doi))

    chunks = (
        [
            _chunk(
                version_id,
                "abstract",
                abstract,
                0,
                json_pointer="/abstract_inverted_index",
            )
        ]
        if abstract
        else []
    )
    return DocumentEnvelope(
        document_id=document_id,
        document_version_id=version_id,
        document_type=DocumentType.ARTICLE,
        title=payload.get("title") or payload.get("display_name") or identity,
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
        domains=[domain] if domain else [],
        chunks=chunks,
        metadata={
            "type": payload.get("type"),
            "topics": payload.get("topics") or [],
        },
        metrics={
            "citation_count": float(payload.get("cited_by_count") or 0),
            "reference_count": float(
                payload.get("referenced_works_count") or 0
            ),
            "authorship_count": float(len(payload.get("authorships") or [])),
        },
        coverage="abstract_only" if abstract else "metadata_only",
    )


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
    # Mutable counters need their own snapshot so later collection cannot
    # overwrite the metrics attached to a previous version. Fetch time alone
    # is not a version.
    observation = {
        key: value for key, value in payload.items() if key != "_retrieved_at"
    }
    snapshot_hash = _bytes_hash(
        json.dumps(observation, sort_keys=True).encode("utf-8")
    )
    version_id = stable_id("version", document_id, commit_sha, snapshot_hash)
    canonical_url = (
        repo.get("html_url")
        or f"{load_catalog('sources')['platforms']['github']['public_base']}"
        f"/{repo['full_name']}"
    )
    chunks: List[Chunk] = []
    organizations: List[Organization] = []

    if isinstance(readme, Mapping):
        if content:
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
            organizations.append(
                Organization(
                    organization_id=stable_id(
                        "organization", "github", owner_key
                    ),
                    name=owner["login"],
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
        chunks=chunks,
        metadata={
            "full_name": repo.get("full_name"),
            "commit_sha": commit_sha,
            "content_date_status": "commit_metadata"
            if content_date
            else "unavailable",
            "description": repo.get("description"),
            "topics": repo.get("topics") or [],
            "license": (repo.get("license") or {}).get("spdx_id"),
            "stars": repo.get("stargazers_count"),
            "fork": bool(repo.get("fork")),
        },
        metrics={
            "stars": float(repo.get("stargazers_count") or 0),
            "forks": float(repo.get("forks_count") or 0),
            "watchers": float(repo.get("subscribers_count") or 0),
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
    version_id = stable_id("version", document_id, version, _bytes_hash(raw))
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
    domain = _domain_from_values(
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
        published_at=uploaded_at[0] if uploaded_at else None,
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
        domains=[domain] if domain else [],
        chunks=chunks,
        metadata={
            "version": version,
            "summary": info.get("summary"),
            "classifiers": info.get("classifiers") or [],
            "project_urls": info.get("project_urls") or {},
            "requires_python": info.get("requires_python"),
            "country_status": "provided" if countries else "unavailable",
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


def parse_epo(xml: str, uri: str = "local://epo.xml") -> DocumentEnvelope:
    raw = xml.encode("utf-8")
    root = ET.fromstring(xml)
    country = _first_text(root, "country") or ""
    doc_number = _first_text(root, "doc-number") or _first_text(
        root, "publication-reference"
    )
    if not doc_number:
        raise ValueError("EPO XML has no publication number")
    kind = _first_text(root, "kind") or ""
    publication_id = "".join(
        part for part in (country, doc_number, kind) if part
    )
    document_id = stable_id("document", "epo", publication_id)
    version_id = stable_id("version", document_id, _bytes_hash(raw))
    title = _first_text(root, "invention-title") or publication_id
    abstract_parts = []
    for abstract in _elements(root, "abstract"):
        text = " ".join("".join(abstract.itertext()).split())
        if text and text not in abstract_parts:
            abstract_parts.append(text)
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

    contributors = []
    organizations = []
    for role, tag in load_catalog("sources")["platforms"]["epo"][
        "people_tags"
    ].items():
        for element in _elements(root, tag):
            name = next(
                (
                    " ".join("".join(candidate.itertext()).split())
                    for candidate in element.iter()
                    if _local_name(candidate)
                    in load_catalog("sources")["platforms"]["epo"]["name_tags"]
                    and "".join(candidate.itertext()).strip()
                ),
                None,
            )
            if name:
                if role == "applicant":
                    organizations.append(
                        Organization(
                            organization_id=stable_id(
                                "organization", "epo", name
                            ),
                            name=name,
                            role=role,
                        )
                    )
                else:
                    contributors.append(
                        Contributor(
                            contributor_id=stable_id("person", "epo", name),
                            name=name,
                            role=role,
                        )
                    )

    date = _first_text(root, "date")
    if date and re.fullmatch(r"\d{8}", date):
        date = f"{date[:4]}-{date[4:6]}-{date[6:]}"

    return DocumentEnvelope(
        document_id=document_id,
        document_version_id=version_id,
        document_type=DocumentType.PATENT,
        title=title,
        published_at=date,
        version_published_at=date,
        source=_source("epo", publication_id, uri),
        artifact=_artifact(uri, raw, "application/xml"),
        identifiers=[
            ExternalId(scheme="patent-publication", value=publication_id)
        ],
        contributors=contributors,
        organizations=organizations,
        countries=(
            [
                Country(
                    country_id=stable_id("country", country.upper()),
                    code=country.upper(),
                    role="jurisdiction",
                )
            ]
            if country
            else []
        ),
        chunks=chunks,
        metadata={"country": country or None, "kind": kind or None},
        coverage="abstract_only" if chunks else "metadata_only",
    )
