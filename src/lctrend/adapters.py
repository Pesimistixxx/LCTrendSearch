from __future__ import annotations

import base64
import hashlib
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .models import (
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


DOMAIN_RULES = (
    ("Bioinformatics", ("bioinformatics", "computational biology", "genomics")),
    ("Edge computing", ("edge computing", "fog computing", "edge ai")),
    ("Artificial intelligence", ("artificial intelligence", "machine learning", "natural language", "computer vision", "deep learning", "named entity recognition", "ner")),
    ("Robotics", ("robotics", "robotic")),
    ("Cybersecurity", ("cybersecurity", "computer security", "information security")),
    ("Quantum computing", ("quantum computing", "quantum information")),
    ("Semiconductors", ("semiconductor", "integrated circuit")),
    ("Energy technology", ("renewable energy", "energy storage", "smart grid")),
)


def _domain_from_values(values: Iterable[Any]) -> Optional[Domain]:
    text = " ".join(str(value).casefold() for value in values)
    for name, aliases in DOMAIN_RULES:
        if any(re.search(rf"\b{re.escape(alias)}\b", text) for alias in aliases):
            return Domain(domain_id=stable_id("domain", name), name=name)
    return None


def _domain_from_topics(topics: Iterable[Mapping[str, Any]]) -> Optional[Domain]:
    """Choose one canonical broad domain from source-ranked OpenAlex topics."""
    for topic in topics:
        values = [topic.get("display_name", "")]
        values.extend((topic.get(key) or {}).get("display_name", "") for key in ("subfield", "field", "domain"))
        domain = _domain_from_values(values)
        if domain:
            return domain
    return None


def _bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _artifact(uri: str, raw: bytes, media_type: str) -> Artifact:
    return Artifact(uri=uri, sha256=_bytes_hash(raw), media_type=media_type, byte_length=len(raw))


def _chunk(version_id: str, kind: str, text: str, order: int, **locator: Any) -> Chunk:
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
    max_chars: int = 1000,
    overlap_chars: int = 150,
) -> List[Chunk]:
    """Split Markdown by headings and paragraphs with intra-section overlap."""
    headings: List[str] = []
    sections: List[tuple[List[str], str]] = []
    current: List[str] = []

    def flush() -> None:
        body = "".join(current).strip()
        if body or headings:
            content = f"{headings[-1]}\n\n{body}".strip() if headings else body
            sections.append(([kind, *headings], content))

    for line in text.splitlines(keepends=True):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            flush()
            current = []
            level = len(match.group(1))
            headings[:] = headings[: level - 1]
            headings.append(match.group(2).strip())
        else:
            current.append(line)
    flush()

    chunks: List[Chunk] = []
    search_from = 0
    for section_path, body in sections:
        budget = max_chars - overlap_chars - 2
        blocks = [block.strip() for block in re.split(r"\n\s*\n", body) if block.strip()]
        parts: List[str] = []
        pending = ""
        for block in blocks:
            while len(block) > budget:
                cut = block.rfind(" ", 0, budget + 1)
                cut = cut if cut > budget // 2 else budget
                if pending:
                    parts.append(pending)
                    pending = ""
                parts.append(block[:cut].strip())
                block = block[cut:].strip()
            candidate = f"{pending}\n\n{block}" if pending else block
            if len(candidate) <= budget:
                pending = candidate
            else:
                parts.append(pending)
                pending = block
        if pending:
            parts.append(pending)

        previous = ""
        for part in parts:
            overlap = previous[-overlap_chars:]
            if overlap and " " in overlap:
                overlap = overlap.split(" ", 1)[1]
            chunk_text = f"{overlap}\n\n{part}" if overlap else part
            offset = text.find(part, search_from)
            if offset < 0:
                offset = text.find(part)
            search_from = max(search_from, offset + len(part))
            chunks.append(
                _chunk(
                    version_id,
                    kind,
                    chunk_text,
                    start_order + len(chunks),
                    section_path=section_path,
                    source_start=offset if offset >= 0 else None,
                    source_end=offset + len(part) if offset >= 0 else None,
                    overlap_chars=len(overlap),
                )
            )
            previous = part
    return chunks


def _pypi_people(value: Any) -> List[str]:
    return [name.strip() for name in re.split(r"\s*[,;]\s*", str(value or "")) if name.strip()]


def _abstract_from_inverted_index(index: Optional[Mapping[str, Iterable[int]]]) -> str:
    if not index:
        return ""
    positioned = [(position, word) for word, positions in index.items() for position in positions]
    return " ".join(word for _, word in sorted(positioned))


def parse_openalex(payload: Mapping[str, Any], raw: Optional[bytes] = None) -> DocumentEnvelope:
    raw = raw or repr(dict(payload)).encode("utf-8")
    openalex_id = str(payload["id"]).rsplit("/", 1)[-1]
    doi = str(payload.get("doi") or "").removeprefix("https://doi.org/")
    identity = doi or openalex_id
    document_id = stable_id("document", "openalex", identity)
    version_id = stable_id("version", document_id, _bytes_hash(raw))
    abstract = _abstract_from_inverted_index(payload.get("abstract_inverted_index"))
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
            organization_id = stable_id("organization", "openalex", external_id)
            country_code = (institution.get("country_code") or "").upper() or None
            organization_type = {
                "education": "university",
                "company": "company",
            }.get(institution.get("type"), institution.get("type") or "other")
            organizations[organization_id] = Organization(
                organization_id=organization_id,
                name=institution_name,
                organization_type=organization_type,
                country_code=country_code,
                role="affiliation",
                external_ids=[ExternalId(scheme="openalex", value=external_id)],
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
                country_id=stable_id("country", code), code=code, role="authorship"
            )
        if name:
            author_id = str(author.get("id") or name)
            contributors.append(
                Contributor(
                    contributor_id=stable_id("person", "openalex", author_id),
                    name=name,
                    external_ids=[ExternalId(scheme="openalex", value=author_id)],
                    affiliation_ids=affiliation_ids,
                )
            )

    domain = _domain_from_topics(payload.get("topics") or [])

    identifiers = [ExternalId(scheme="openalex", value=openalex_id)]
    if doi:
        identifiers.append(ExternalId(scheme="doi", value=doi))

    chunks = [_chunk(version_id, "abstract", abstract, 0, json_pointer="/abstract_inverted_index")] if abstract else []
    return DocumentEnvelope(
        document_id=document_id,
        document_version_id=version_id,
        document_type=DocumentType.ARTICLE,
        title=payload.get("title") or payload.get("display_name") or identity,
        language=payload.get("language"),
        published_at=payload.get("publication_date"),
        source=SourceRef(
            source_id="source:openalex",
            name="OpenAlex",
            source_type="scholarly_api",
            record_id=openalex_id,
            canonical_url=canonical_url,
        ),
        artifact=_artifact(str(canonical_url), raw, "application/json"),
        identifiers=identifiers,
        contributors=contributors,
        organizations=list(organizations.values()),
        countries=list(countries.values()),
        domains=[domain] if domain else [],
        chunks=chunks,
        metadata={"type": payload.get("type"), "topics": payload.get("topics") or []},
        coverage="abstract_only" if abstract else "metadata_only",
    )


def parse_github(payload: Mapping[str, Any], raw: Optional[bytes] = None) -> DocumentEnvelope:
    repo = payload.get("repository") or payload
    raw = raw or repr(dict(payload)).encode("utf-8")
    repo_id = str(repo.get("node_id") or repo.get("id") or repo["full_name"])
    document_id = stable_id("document", "github", repo_id)
    version_marker = repo.get("pushed_at") or repo.get("updated_at") or _bytes_hash(raw)
    version_id = stable_id("version", document_id, version_marker)
    canonical_url = repo.get("html_url") or f"https://github.com/{repo['full_name']}"
    chunks: List[Chunk] = []
    organizations: List[Organization] = []

    readme = payload.get("readme") if "repository" in payload else None
    if isinstance(readme, Mapping):
        content = readme.get("text")
        if not content and readme.get("content"):
            content = base64.b64decode(readme["content"]).decode("utf-8", errors="replace")
        if content:
            readme_chunks = _markdown_chunks(version_id, "readme", content, len(chunks))
            for chunk in readme_chunks:
                chunk.locator.update(
                    commit_sha=version_marker, path=readme.get("path", "README.md")
                )
            chunks.extend(readme_chunks)

    for release in payload.get("releases") or []:
        body = release.get("body") or ""
        if body:
            release_chunks = _markdown_chunks(version_id, "release", body, len(chunks))
            for chunk in release_chunks:
                chunk.locator.update(
                    release_id=release.get("id"), tag=release.get("tag_name")
                )
            chunks.extend(release_chunks)

    owner = repo.get("owner") or {}
    contributors = []
    if owner.get("login"):
        owner_key = owner.get("node_id") or owner["login"]
        if owner.get("type") == "Organization":
            organizations.append(
                Organization(
                    organization_id=stable_id("organization", "github", owner_key),
                    name=owner["login"],
                    role="owner",
                    external_ids=[ExternalId(scheme="github", value=str(owner_key))],
                )
            )
        else:
            contributors.append(
                Contributor(
                    contributor_id=stable_id("person", "github", owner_key),
                    name=owner["login"],
                    role="owner",
                    external_ids=[ExternalId(scheme="github", value=str(owner_key))],
                )
            )

    return DocumentEnvelope(
        document_id=document_id,
        document_version_id=version_id,
        document_type=DocumentType.REPOSITORY,
        title=repo.get("name") or repo["full_name"],
        language=repo.get("language"),
        published_at=repo.get("created_at"),
        source=SourceRef(
            source_id="source:github",
            name="GitHub",
            source_type="code_host",
            record_id=repo_id,
            canonical_url=canonical_url,
        ),
        artifact=_artifact(canonical_url, raw, "application/json"),
        identifiers=[ExternalId(scheme="github", value=repo_id)],
        contributors=contributors,
        organizations=organizations,
        chunks=chunks,
        metadata={
            "full_name": repo.get("full_name"),
            "description": repo.get("description"),
            "topics": repo.get("topics") or [],
            "license": (repo.get("license") or {}).get("spdx_id"),
            "stars": repo.get("stargazers_count"),
            "fork": bool(repo.get("fork")),
        },
        coverage="selected_files" if chunks else "metadata_only",
    )


def parse_pypi(payload: Mapping[str, Any], raw: Optional[bytes] = None) -> DocumentEnvelope:
    raw = raw or repr(dict(payload)).encode("utf-8")
    info = payload["info"]
    name = info["name"]
    version = info.get("version") or "unknown"
    document_id = stable_id("document", "pypi", name.lower())
    version_id = stable_id("version", document_id, version, _bytes_hash(raw))
    canonical_url = info.get("package_url") or f"https://pypi.org/project/{name}/"
    uploaded_at = sorted(
        item["upload_time_iso_8601"]
        for item in payload.get("urls") or []
        if item.get("upload_time_iso_8601")
    )
    domain = _domain_from_values(
        [name, info.get("summary") or "", *(info.get("classifiers") or [])]
    )
    country_code = str(info.get("country_code") or info.get("country") or "").upper()
    countries = (
        [Country(country_id=stable_id("country", country_code), code=country_code, role="metadata")]
        if re.fullmatch(r"[A-Z]{2}", country_code)
        else []
    )
    description = info.get("description") or info.get("summary") or ""
    chunks = _markdown_chunks(version_id, "description", description) if description else []
    for chunk in chunks:
        chunk.locator["json_pointer"] = "/info/description"
    contributors = []
    seen = set()
    for role, field in (("author", "author"), ("maintainer", "maintainer")):
        for person_name in _pypi_people(info.get(field)):
            key = (person_name.casefold(), role)
            if key in seen:
                continue
            seen.add(key)
            contributors.append(
                Contributor(
                    contributor_id=stable_id("person", "pypi", person_name.casefold()),
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
        source=SourceRef(
            source_id="source:pypi",
            name="PyPI",
            source_type="package_registry",
            record_id=name,
            canonical_url=canonical_url,
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
    doc_number = _first_text(root, "doc-number") or _first_text(root, "publication-reference")
    if not doc_number:
        raise ValueError("EPO XML has no publication number")
    kind = _first_text(root, "kind") or ""
    publication_id = "".join(part for part in (country, doc_number, kind) if part)
    document_id = stable_id("document", "epo", publication_id)
    version_id = stable_id("version", document_id, _bytes_hash(raw))
    title = _first_text(root, "invention-title") or publication_id
    abstract_parts = []
    for abstract in _elements(root, "abstract"):
        text = " ".join("".join(abstract.itertext()).split())
        if text and text not in abstract_parts:
            abstract_parts.append(text)
    abstract = "\n".join(abstract_parts)
    chunks = [_chunk(version_id, "abstract", abstract, 0, xpath="//*[local-name()='abstract']")] if abstract else []

    contributors = []
    organizations = []
    for role, tag in (("applicant", "applicant"), ("inventor", "inventor")):
        for element in _elements(root, tag):
            name = next(
                (
                    " ".join("".join(candidate.itertext()).split())
                    for candidate in element.iter()
                    if _local_name(candidate) in {"name", "last-name"} and "".join(candidate.itertext()).strip()
                ),
                None,
            )
            if name:
                if role == "applicant":
                    organizations.append(
                        Organization(
                            organization_id=stable_id("organization", "epo", name),
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
        source=SourceRef(
            source_id="source:epo",
            name="EPO OPS",
            source_type="patent_api",
            record_id=publication_id,
            canonical_url=uri,
        ),
        artifact=_artifact(uri, raw, "application/xml"),
        identifiers=[ExternalId(scheme="patent-publication", value=publication_id)],
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
