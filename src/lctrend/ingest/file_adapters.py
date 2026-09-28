"""Local file snapshots to addressable text, without invented publication
dates.

Dates, language and parties come only from the file's own embedded metadata
or from an explicit sidecar (``report.pdf.meta.json``); each value records
its basis in document.metadata.
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
import re
import tempfile
import threading
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from zipfile import BadZipFile, ZipFile

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
from .snapshots import snapshot_bytes

ADAPTER_VERSION = "local-files/1"


class FileAdapterError(ValueError):
    pass


class UnsupportedFileFormat(FileAdapterError):
    pass


def _text(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise FileAdapterError(
            "Text files must be UTF-8; undecodable bytes were not replaced"
        ) from None


def _chunks(
    version_id: str,
    text: str,
    kind: str,
    locator: Dict[str, Any],
    section_path: Optional[List[str]] = None,
    source_start: int = 0,
    source_stream_id: Optional[str] = None,
) -> List[Chunk]:
    """Split without altering source text; offsets are in the stated decoded
    text basis.
    """
    budget = int(load_catalog("sources")["markdown"]["max_chars"])
    if budget < 1:
        raise FileAdapterError("File chunk size must be positive")
    stream_id = source_stream_id or stable_id("stream", version_id, locator)
    chunks = []
    cursor = 0
    while cursor < len(text):
        end = min(len(text), cursor + budget)
        if end < len(text):
            boundary = max(
                text.rfind("\n", cursor, end), text.rfind(" ", cursor, end)
            )
            if boundary > cursor + budget // 2:
                end = boundary + 1
        value = text[cursor:end]
        if value.strip():
            start = source_start + cursor
            stop = source_start + end
            chunks.append(
                Chunk(
                    chunk_id=stable_id(
                        "chunk", version_id, stream_id, start, stop, value
                    ),
                    kind=kind,
                    text=value,
                    order=0,
                    section_path=section_path or [],
                    locator={
                        **locator,
                        "source_stream_id": stream_id,
                        "source_start": start,
                        "source_end": stop,
                        "source_segments": [
                            {
                                "chunk_start": 0,
                                "chunk_end": len(value),
                                "source_start": start,
                                "source_end": stop,
                            }
                        ],
                    },
                )
            )
        cursor = end
    return chunks


def _plain(
    version_id: str, text: str, path: Path, markdown: bool
) -> List[Chunk]:
    chunks = []
    headings: List[str] = []
    stream_id = stable_id("stream", version_id, "decoded_file_text")
    # Structural Markdown boundaries and paragraph boundaries do not change
    # the text.
    heading_starts = (
        [
            match.start()
            for match in re.finditer(r"(?m)^#{1,6}\s+[^\r\n]+", text)
        ]
        if markdown
        else []
    )
    boundaries = sorted({0, *heading_starts, len(text)})
    for left, right in zip(boundaries, boundaries[1:]):
        section = text[left:right]
        heading = (
            re.match(r"^(#{1,6})\s+([^\r\n]+)", section) if markdown else None
        )
        if heading:
            level = len(heading.group(1))
            headings[:] = headings[: level - 1]
            headings.append(heading.group(2).strip())
        paragraph_starts = [0]
        spans = []
        for delimiter in re.finditer(r"\r?\n[\t \r]*\r?\n", section):
            spans.append((paragraph_starts[-1], delimiter.start()))
            paragraph_starts.append(delimiter.end())
        spans.append((paragraph_starts[-1], len(section)))
        for start_in_section, end_in_section in spans:
            original = section[start_in_section:end_in_section]
            body = original.strip()
            if not body:
                continue
            start = (
                left
                + start_in_section
                + len(original)
                - len(original.lstrip())
            )
            produced = _chunks(
                version_id,
                body,
                "markdown" if markdown else "paragraph",
                {"path": str(path), "text_basis": "utf8_sig_decoded_source"},
                headings.copy(),
                start,
                stream_id,
            )
            for chunk in produced:
                chunk.locator["line_start"] = (
                    text.count("\n", 0, chunk.locator["source_start"]) + 1
                )
                chunk.locator["line_end"] = (
                    text.count("\n", 0, chunk.locator["source_end"]) + 1
                )
            chunks.extend(produced)
    return chunks


def _xml(raw: bytes) -> ET.Element:
    if re.search(rb"<!\s*(?:DOCTYPE|ENTITY)\b", raw, re.I):
        raise FileAdapterError("DTD and custom XML entities are not supported")
    try:
        return ET.fromstring(raw)
    except ET.ParseError:
        raise FileAdapterError("Invalid OOXML document") from None


def _xml_nodes(root: ET.Element, path: str = ""):
    current = path or "/" + root.tag + "[1]"
    yield root, current
    counts: Dict[str, int] = {}
    for child in root:
        counts[child.tag] = counts.get(child.tag, 0) + 1
        yield from _xml_nodes(
            child,
            current + "/" + child.tag + "[" + str(counts[child.tag]) + "]",
        )


def _docx(
    raw: bytes, version_id: str, path: Path
) -> Tuple[str, List[Chunk], List[str]]:
    limits = load_catalog("pipeline")["file_limits"]
    title = path.stem
    try:
        with ZipFile(io.BytesIO(raw)) as archive:
            entry = archive.getinfo("word/document.xml")
            if entry.file_size > limits["docx_document_xml_bytes"]:
                raise FileAdapterError(
                    "DOCX document XML exceeds the configured size limit"
                )
            root = _xml(archive.read(entry))
            if "docProps/core.xml" in archive.namelist():
                if (
                    archive.getinfo("docProps/core.xml").file_size
                    > limits["docx_core_xml_bytes"]
                ):
                    raise FileAdapterError(
                        "DOCX metadata exceeds the configured size limit"
                    )
                core = _xml(archive.read("docProps/core.xml"))
                title = next(
                    (
                        node.text
                        for node in core.iter()
                        if node.tag.rsplit("}", 1)[-1] == "title" and node.text
                    ),
                    title,
                )
    except (BadZipFile, KeyError, RuntimeError):
        raise FileAdapterError(
            "Invalid DOCX package or missing word/document.xml"
        ) from None
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

    def paragraph_text(node: ET.Element) -> str:
        result = []
        for child in node:
            if child.tag == ns + "p":
                continue  # A nested text-box paragraph is emitted separately.
            if child.tag == ns + "t":
                result.append(child.text or "")
            elif child.tag == ns + "tab":
                result.append("\t")
            elif child.tag in (ns + "br", ns + "cr"):
                result.append("\n")
            result.append(paragraph_text(child))
        return "".join(result)

    chunks = []
    headings: List[str] = []
    for node, element_path in _xml_nodes(root):
        if node.tag != ns + "p":
            continue
        text = paragraph_text(node)
        if not text.strip():
            continue
        style = next(
            (
                item.attrib.get(ns + "val", "")
                for item in node.iter()
                if item.tag == ns + "pStyle"
            ),
            "",
        )
        heading = re.fullmatch(r"Heading([1-6])", style, re.I)
        if heading:
            level = int(heading.group(1))
            headings[:] = headings[: level - 1]
            headings.append(text.strip())
        locator = {
            "path": str(path),
            "kind": "ooxml",
            "part": "word/document.xml",
            "element_path": element_path,
            "path_syntax": "namespace_qualified_element_path",
            "text_basis": "w_t_tab_br_v1",
        }
        kind = (
            "table_cell_paragraph"
            if ns + "tc[" in element_path
            else "paragraph"
        )
        chunks.extend(
            _chunks(
                version_id,
                text,
                kind,
                locator,
                headings.copy(),
                source_stream_id=stable_id("stream", version_id, element_path),
            )
        )
    return (
        title,
        chunks,
        list(load_catalog("pipeline")["file_warnings"]["docx"]),
    )


class _HTMLNode:
    def __init__(self, tag: str, path: str, excluded: bool = False):
        self.tag, self.path, self.excluded = tag, path, excluded
        self.content: List[Any] = []
        self.counts: Dict[str, int] = {}

    def text(self) -> str:
        if self.excluded:
            return ""
        return "".join(
            item if isinstance(item, str) else item.text()
            for item in self.content
        )


class _HTMLTree(HTMLParser):
    def __init__(self, excluded_tags: Iterable[str]):
        super().__init__(convert_charrefs=True)
        self.root = _HTMLNode("document", "")
        self.stack = [self.root]
        self.excluded_tags = set(excluded_tags)
        self.void_tags = set(load_catalog("pipeline")["html"]["void_tags"])

    def handle_starttag(self, tag: str, attrs) -> None:
        parent = self.stack[-1]
        parent.counts[tag] = parent.counts.get(tag, 0) + 1
        node = _HTMLNode(
            tag,
            parent.path + "/" + tag + "[" + str(parent.counts[tag]) + "]",
            parent.excluded or tag in self.excluded_tags,
        )
        parent.content.append(node)
        if tag == "br":
            parent.content.append("\n")
        elif tag not in self.void_tags:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        self.stack[-1].content.append(data)


def _html(
    text: str, version_id: str, path: Path
) -> Tuple[str, List[Chunk], List[str]]:
    settings = load_catalog("pipeline")["html"]
    tree = _HTMLTree(settings["excluded_tags"])
    tree.feed(text)
    tree.close()
    title, chunks, headings = path.stem, [], []
    block_tags = set(settings["block_tags"])

    def emit(value: str, node: _HTMLNode, part: int = 0) -> None:
        if not value.strip():
            return
        if re.fullmatch(r"h[1-6]", node.tag):
            level = int(node.tag[1])
            headings[:] = headings[: level - 1]
            headings.append(value.strip())
        locator = {
            "path": str(path),
            "kind": "html",
            "element_path": node.path,
            "text_part": part,
            "text_basis": "htmlparser_rendered_text_v1",
        }
        chunks.extend(
            _chunks(
                version_id,
                value,
                "heading"
                if node.tag.startswith("h") and node.tag in block_tags
                else "paragraph",
                locator,
                headings.copy(),
                source_stream_id=stable_id(
                    "stream", version_id, node.path, part
                ),
            )
        )

    def has_block(node: _HTMLNode) -> bool:
        return any(
            isinstance(item, _HTMLNode)
            and (item.tag in block_tags or has_block(item))
            for item in node.content
        )

    def visit(node: _HTMLNode) -> None:
        nonlocal title
        if node.excluded:
            return
        if node.tag == "title":
            if node.text().strip():
                title = node.text().strip()
            return
        if node.tag == "head":
            for item in node.content:
                if isinstance(item, _HTMLNode) and item.tag == "title":
                    visit(item)
            return
        if node.tag in block_tags and not has_block(node):
            emit(node.text(), node)
            return
        pending = []
        part = 0
        for item in node.content:
            if isinstance(item, str):
                pending.append(item)
            elif (
                not item.excluded
                and item.tag not in block_tags
                and not has_block(item)
                and item.tag not in {"head", "title"}
            ):
                pending.append(item.text())
            else:
                emit("".join(pending), node, part)
                pending.clear()
                part += 1
                visit(item)
        emit("".join(pending), node, part)

    visit(tree.root)
    return (
        title,
        chunks,
        list(load_catalog("pipeline")["file_warnings"]["html"]),
    )


_CONVERTER = None
# Docling loads its layout/OCR models once per converter and is CPU-bound;
# one shared converter, one conversion at a time.
_CONVERTER_LOCK = threading.Lock()


def _convert(path: Path):
    global _CONVERTER
    from docling.document_converter import DocumentConverter

    limits = load_catalog("pipeline")["file_limits"]
    with _CONVERTER_LOCK:
        if not isinstance(_CONVERTER, DocumentConverter):
            _CONVERTER = DocumentConverter()
        options = {}
        if "max_num_pages" in inspect.signature(_CONVERTER.convert).parameters:
            # A book-sized PDF would hold the single converter for hours.
            options["max_num_pages"] = int(limits.get("pdf_max_pages", 10**9))
        return _CONVERTER.convert(path, **options)


def _pdf(
    raw: bytes, version_id: str, path: Path
) -> Tuple[str, List[Chunk], List[str]]:
    try:
        from docling.document_converter import (  # noqa: F401
            DocumentConverter,
        )
    except ImportError:
        raise UnsupportedFileFormat(
            "PDF parsing requires the optional Docling PDF dependency"
        ) from None
    # Convert the bytes whose hash we report, not a file that may change
    # during conversion.
    snapshot_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="lctrend-pdf-", suffix=".pdf", delete=False
        ) as snapshot:
            snapshot.write(raw)
            snapshot_path = Path(snapshot.name)
        conversion = _convert(snapshot_path)
    finally:
        if snapshot_path is not None:
            snapshot_path.unlink(missing_ok=True)
    document = conversion.document
    chunks, warnings = (
        [],
        list(load_catalog("pipeline")["file_warnings"]["pdf"]),
    )
    for item, _level in document.iterate_items():
        label = getattr(item.label, "value", str(item.label))
        value = (
            item.export_to_markdown(doc=document)
            if label == "table"
            else getattr(item, "text", "")
        )
        if not value or not value.strip():
            continue
        provenance = [
            prov.model_dump(mode="json") for prov in getattr(item, "prov", [])
        ]
        locator = {
            "path": str(path),
            "kind": "docling",
            "self_ref": item.self_ref,
            "provenance": provenance,
            "text_basis": "docling_table_markdown"
            if label == "table"
            else "docling_text",
        }
        chunks.extend(
            _chunks(
                version_id,
                value,
                label,
                locator,
                source_stream_id=stable_id(
                    "stream", version_id, item.self_ref
                ),
            )
        )
    status = getattr(conversion.status, "value", str(conversion.status))
    if status != "success":
        warnings.append("docling_conversion_status:" + status)
    if any(not chunk.locator["provenance"] for chunk in chunks):
        warnings.append("some_pdf_items_have_no_page_provenance")
    return path.stem, chunks, warnings


def section_role(heading: Optional[str]) -> Optional[str]:
    """Rhetorical role of a paper section (method, results, limitations...).

    The packet stream stays one "fulltext" kind; the role is navigation
    metadata, so a heading never becomes evidence.
    """
    if not heading:
        return None
    for role, pattern in load_catalog("pipeline")["section_roles"].items():
        if re.search(pattern, heading, re.IGNORECASE):
            return role
    return None


def pdf_body(chunks: List[Chunk], **locator: Any) -> List[Chunk]:
    """Paper body of Docling chunks, shared by local and OpenAlex PDFs.

    Page headers, footers and the bibliography are dropped. One kind keeps a
    paper in one packet stream (a new Docling label must not close a
    packet); the Docling label and section stay visible in the locator.
    """
    settings = load_catalog("pipeline")["openalex_fulltext"]
    skipped = set(settings["skipped_labels"])
    bibliography = re.compile(settings["bibliography_heading"], re.IGNORECASE)
    kept, heading, in_bibliography = [], None, False
    for chunk in chunks:
        label = chunk.kind
        if label == "section_header":
            heading = chunk.text.strip()
            in_bibliography = bool(bibliography.match(heading))
        if label in skipped or in_bibliography:
            continue
        chunk.kind = "fulltext"
        chunk.section_path = ["fulltext"]
        chunk.locator.update(
            docling_label=label,
            section_heading=heading,
            section_role=section_role(heading),
            **locator,
        )
        kept.append(chunk)
    return kept


HTML_DATE_FIELDS = (
    "citation_publication_date",
    "citation_date",
    "article:published_time",
    "dc.date",
    "dcterms.created",
    "dc.date.issued",
    "date",
    "pubdate",
)


class _HTMLMeta(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: Dict[str, str] = {}
        self.authors: List[str] = []
        self.language: Optional[str] = None

    def handle_starttag(self, tag: str, attrs) -> None:
        values = {key.lower(): value for key, value in attrs if value}
        if tag == "html" and values.get("lang"):
            self.language = values["lang"]
        if tag != "meta":
            return
        key = (values.get("name") or values.get("property") or "").lower()
        content = values.get("content", "").strip()
        if not key or not content:
            return
        if key in ("citation_author", "author", "dc.creator"):
            self.authors.append(content)
        self.meta.setdefault(key, content)


def _iso_date(value: Optional[str]) -> Optional[str]:
    """Keep a date only when it is a real calendar date."""
    if not value:
        return None
    value = value.strip()
    match = re.match(r"(\d{4})[-/.](\d{1,2})(?:[-/.](\d{1,2}))?", value)
    if not match:
        return None
    year, month, day = match.group(1), match.group(2), match.group(3) or "1"
    try:
        return datetime(int(year), int(month), int(day)).date().isoformat()
    except ValueError:
        return None


def _embedded_facts(adapter: str, raw: bytes) -> Dict[str, Any]:
    """Facts the file itself states, with their basis. Only HTML
    publication tags give a publication date; DOCX/PDF creation dates
    describe the file.
    """
    facts: Dict[str, Any] = {}
    if adapter == "html":
        parser = _HTMLMeta()
        try:
            parser.feed(_text(raw))
        except FileAdapterError:
            return facts
        for field in HTML_DATE_FIELDS:
            date = _iso_date(parser.meta.get(field))
            if date:
                facts["published_at"] = (date, "html_meta:" + field)
                break
        language = parser.meta.get("citation_language") or parser.language
        if language:
            facts["language"] = (language.split("-")[0].lower(), "html_lang")
        if parser.authors:
            facts["authors"] = (parser.authors, "html_meta:author")
    elif adapter == "docx":
        try:
            with ZipFile(io.BytesIO(raw)) as archive:
                if "docProps/core.xml" not in archive.namelist():
                    return facts
                core = _xml(archive.read("docProps/core.xml"))
        except (BadZipFile, FileAdapterError):
            return facts
        values = {
            node.tag.rsplit("}", 1)[-1]: (node.text or "").strip()
            for node in core.iter()
        }
        date = _iso_date(values.get("created"))
        if date:
            facts["file_created_at"] = (date, "docx_core:created")
        if values.get("language"):
            facts["language"] = (
                values["language"].split("-")[0].lower(),
                "docx_core:language",
            )
        if values.get("creator"):
            facts["authors"] = ([values["creator"]], "docx_core:creator")
    elif adapter == "pdf":
        match = re.search(rb"/CreationDate\s*\(D:(\d{4})(\d{2})?(\d{2})?", raw)
        if match:
            date = _iso_date(
                "-".join(part.decode() for part in match.groups(default=b"01"))
            )
            if date:
                facts["file_created_at"] = (date, "pdf_info:CreationDate")
    return facts


def _guess_language(chunks: List[Chunk]) -> Optional[str]:
    settings = load_catalog("sources")["local_files"]
    text = " ".join(chunk.text for chunk in chunks[:20])
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return None
    cyrillic = sum("Ѐ" <= char <= "ӿ" for char in letters)
    if cyrillic / len(letters) >= settings["cyrillic_share_for_ru"]:
        return "ru"
    words = set(re.findall(r"[a-z]+", text.casefold()))
    if len(words & set(settings["english_markers"])) >= 2:
        return "en"
    return None


def _sidecar(path: Path) -> Dict[str, Any]:
    """Operator-supplied metadata for a local file (dataset exports,
    manual curation). It is data about the file, never file text.
    """
    candidate = path.with_name(
        path.name + load_catalog("sources")["local_files"]["sidecar_suffix"]
    )
    if not candidate.is_file():
        return {}
    try:
        value = json.loads(candidate.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        raise FileAdapterError(
            "Invalid metadata sidecar: " + candidate.name
        ) from None
    if not isinstance(value, dict):
        raise FileAdapterError("Metadata sidecar must be a JSON object")
    return value


def _sidecar_parties(
    sidecar: Dict[str, Any],
) -> Tuple[List[Contributor], List[Organization], List[Country], List[Domain]]:
    organizations: Dict[str, Organization] = {}
    for item in sidecar.get("organizations") or []:
        item = {"name": item} if isinstance(item, str) else item
        if not item.get("name"):
            continue
        organization_id = stable_id("organization", "local", item["name"])
        code = (item.get("country") or "").upper() or None
        organizations[organization_id] = Organization(
            organization_id=organization_id,
            name=item["name"],
            organization_type=item.get("type") or "other",
            country_code=code
            if code and re.fullmatch(r"[A-Z]{2}", code)
            else None,
            role=item.get("role") or "associated",
        )
    by_name = {item.name: key for key, item in organizations.items()}
    contributors = []
    for item in sidecar.get("authors") or []:
        item = {"name": item} if isinstance(item, str) else item
        if not item.get("name"):
            continue
        contributors.append(
            Contributor(
                contributor_id=stable_id("person", "local", item["name"]),
                name=item["name"],
                role=item.get("role") or "author",
                affiliation_ids=[
                    by_name[name]
                    for name in item.get("affiliations") or []
                    if name in by_name
                ],
            )
        )
    codes = {
        code.upper()
        for code in sidecar.get("countries") or []
        if isinstance(code, str) and re.fullmatch(r"[A-Za-z]{2}", code)
    }
    codes |= {
        item.country_code
        for item in organizations.values()
        if item.country_code
    }
    countries = [
        Country(
            country_id=stable_id("country", code),
            code=code,
            role="metadata",
        )
        for code in sorted(codes)
    ]
    domains = {}
    for item in sidecar.get("domains") or []:
        item = {"name": item} if isinstance(item, str) else item
        if item.get("name"):
            domain = Domain(
                domain_id=stable_id("domain", item["name"]),
                name=item["name"],
                parent_name=item.get("parent"),
            )
            domains.setdefault(domain.domain_id, domain)
    return (
        contributors,
        list(organizations.values()),
        countries,
        list(domains.values()),
    )


def parse_file(path: Union[Path, str]) -> DocumentEnvelope:
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise FileAdapterError("Local source must be a file")
    config = load_catalog("pipeline")
    spec = config["file_formats"].get(path.suffix.casefold())
    if spec is None:
        raise UnsupportedFileFormat(
            "Unsupported local file extension: " + path.suffix
        )
    max_bytes = config["file_limits"]["max_file_bytes"]
    with path.open("rb") as source:
        raw = source.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise FileAdapterError(
            "Local source exceeds the configured size limit"
        )
    sha256 = hashlib.sha256(raw).hexdigest()
    snapshot_path = snapshot_bytes(raw)
    document_id = stable_id("document", "local", os.path.normcase(str(path)))
    version_id = stable_id("version", document_id, sha256, ADAPTER_VERSION)
    adapter = spec["adapter"]
    warnings, title = [], path.stem
    if adapter in ("text", "markdown"):
        chunks = _plain(version_id, _text(raw), path, adapter == "markdown")
        coverage = "full_text"
    elif adapter == "docx":
        title, chunks, warnings = _docx(raw, version_id, path)
        coverage = "selected_text"
    elif adapter == "html":
        title, chunks, warnings = _html(_text(raw), version_id, path)
        coverage = "selected_text"
    elif adapter == "pdf":
        title, chunks, warnings = _pdf(raw, version_id, path)
        chunks = pdf_body(chunks)
        coverage = "parsed_text"
    else:
        raise UnsupportedFileFormat(
            "Unknown configured file adapter: " + str(adapter)
        )
    if not chunks:
        warnings.append("no_text_chunks")
    for order, chunk in enumerate(chunks):
        chunk.order = order
        chunk.locator.update(
            artifact_sha256=sha256, adapter_version=ADAPTER_VERSION
        )
    retrieved_at = datetime.now(timezone.utc).isoformat()
    uri = path.as_uri()
    sidecar = _sidecar(path)
    facts = _embedded_facts(adapter, raw)
    basis: Dict[str, str] = {}
    published_at = _iso_date(sidecar.get("published_at"))
    if published_at:
        basis["published_at"] = "sidecar"
    elif "published_at" in facts:
        published_at, basis["published_at"] = facts["published_at"]
    language = sidecar.get("language")
    if language:
        basis["language"] = "sidecar"
    elif "language" in facts:
        language, basis["language"] = facts["language"]
    else:
        language = _guess_language(chunks)
        if language:
            basis["language"] = "script_heuristic"
    contributors, organizations, countries, domains = _sidecar_parties(sidecar)
    if not contributors and "authors" in facts:
        names, basis["authors"] = facts["authors"]
        contributors = [
            Contributor(
                contributor_id=stable_id("person", "local", name),
                name=name,
            )
            for name in dict.fromkeys(names)
        ]
    source = sidecar.get("source") or {}
    try:
        document_type = DocumentType(
            sidecar.get("document_type") or DocumentType.REPORT.value
        )
    except ValueError:
        raise FileAdapterError(
            "Unknown document_type in metadata sidecar"
        ) from None
    return DocumentEnvelope(
        document_id=document_id,
        document_version_id=version_id,
        document_type=document_type,
        title=sidecar.get("title") or title,
        language=language,
        retrieved_at=retrieved_at,
        published_at=published_at,
        version_published_at=published_at,
        source=SourceRef(
            source_id="source:local"
            if not source.get("name")
            else stable_id("source", "local", source["name"]),
            name=source.get("name") or "Local file",
            source_type=source.get("type") or "local_file",
            source_family=source.get("family") or "local",
            reliability_tier=int(source.get("reliability_tier") or 1),
            independence_group=source.get("independence_group"),
            record_id=str(path),
            canonical_url=source.get("url") or uri,
        ),
        identifiers=[
            ExternalId(scheme=str(scheme), value=str(value))
            for scheme, value in (sidecar.get("identifiers") or {}).items()
        ],
        contributors=contributors,
        organizations=organizations,
        countries=countries,
        domains=domains,
        artifact=Artifact(
            uri=snapshot_path.as_uri(),
            sha256=sha256,
            media_type=spec["media_type"],
            byte_length=len(raw),
        ),
        chunks=chunks,
        metadata={
            "parser": ADAPTER_VERSION,
            "parse_warnings": warnings,
            "text_encoding": "utf-8-sig"
            if adapter in ("text", "markdown", "html")
            else None,
            "metadata_basis": basis,
            # Authoring/production date of the file, not a publication date.
            "file_created_at": facts.get("file_created_at", (None,))[0],
            "sidecar": bool(sidecar),
        },
        coverage=coverage,
        quality_status="needs_review" if warnings else "accepted",
    )
