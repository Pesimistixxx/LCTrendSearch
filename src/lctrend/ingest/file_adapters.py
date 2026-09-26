"""Local file snapshots to addressable text, without invented publication
dates.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import tempfile
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
    DocumentEnvelope,
    DocumentType,
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


def _pdf(
    raw: bytes, version_id: str, path: Path
) -> Tuple[str, List[Chunk], List[str]]:
    try:
        from docling.document_converter import DocumentConverter
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
        conversion = DocumentConverter().convert(snapshot_path)
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
    return DocumentEnvelope(
        document_id=document_id,
        document_version_id=version_id,
        document_type=DocumentType.REPORT,
        title=title,
        retrieved_at=retrieved_at,
        published_at=None,
        version_published_at=None,
        source=SourceRef(
            source_id="source:local",
            name="Local file",
            source_type="local_file",
            source_family="local",
            record_id=str(path),
            canonical_url=uri,
        ),
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
        },
        coverage=coverage,
        quality_status="needs_review" if warnings else "accepted",
    )
