import hashlib
import sys
from types import ModuleType, SimpleNamespace
from zipfile import ZipFile

import pytest

from lctrend.ingest.adapters import _markdown_chunks, parse_github
from lctrend.ingest.file_adapters import (
    FileAdapterError,
    UnsupportedFileFormat,
    parse_file,
)


@pytest.fixture(autouse=True)
def raw_directory(tmp_path, monkeypatch):
    directory = tmp_path / "raw"
    monkeypatch.setenv("LCTREND_RAW_DIR", str(directory))
    return directory


def test_txt_snapshot_identity_dates_and_source_offsets(
    tmp_path, raw_directory
):
    path = tmp_path / "paper.txt"
    raw = "First paragraph.\r\n\r\nСенсор не прошёл испытание.".encode("utf-8")
    path.write_bytes(raw)
    first = parse_file(path)
    second = parse_file(path)
    assert first.document_id == second.document_id
    assert first.document_version_id == second.document_version_id
    assert first.published_at is None
    assert first.version_published_at is None
    assert first.retrieved_at
    assert first.source.canonical_url == path.resolve().as_uri()
    snapshot = raw_directory / hashlib.sha256(raw).hexdigest()
    assert snapshot.read_bytes() == raw
    assert first.artifact.uri == snapshot.resolve().as_uri()
    source = raw.decode("utf-8")
    assert len(first.chunks) == 2
    for chunk in first.chunks:
        segment = chunk.locator["source_segments"][0]
        assert (
            chunk.text
            == source[segment["source_start"] : segment["source_end"]]
        )
        assert chunk.locator["line_start"] >= 1
        assert "page" not in chunk.locator
    # A-6: identity is the content, not the path. Other bytes at the same
    # path are another document; the old snapshot is untouched.
    path.write_text("Changed content", encoding="utf-8")
    third = parse_file(path)
    assert third.document_id != first.document_id
    assert third.document_version_id != first.document_version_id
    assert snapshot.read_bytes() == raw


def test_the_same_bytes_uploaded_twice_are_one_document(tmp_path):
    # A-6: every web upload lands in a new folder.
    raw = b"Sparse attention reduces memory use."
    first_path = tmp_path / "upload-1" / "paper.txt"
    second_path = tmp_path / "upload-2" / "renamed.txt"
    for path in (first_path, second_path):
        path.parent.mkdir()
        path.write_bytes(raw)
    first, second = parse_file(first_path), parse_file(second_path)
    assert first.document_id == second.document_id
    assert first.document_version_id == second.document_version_id
    assert [chunk.chunk_id for chunk in first.chunks] == [
        chunk.chunk_id for chunk in second.chunks
    ]


def test_markdown_sections_and_long_fragments_keep_literal_source(tmp_path):
    path = tmp_path / "paper.md"
    text = (
        "# Intro\nText.\n## Results\n" + "Word " * 700 + "\n\nNo improvement."
    )
    path.write_bytes(text.encode("utf-8"))
    document = parse_file(path)
    assert document.chunks[0].section_path == ["Intro"]
    assert document.chunks[-1].section_path == ["Intro", "Results"]
    assert len(document.chunks) > 3
    assert all(len(chunk.text) <= 1000 for chunk in document.chunks)
    for chunk in document.chunks:
        assert (
            text[chunk.locator["source_start"] : chunk.locator["source_end"]]
            == chunk.text
        )
    assert document.chunks[0].text.startswith("# Intro")


def make_docx(path, xml, core=None):
    with ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", xml)
        if core:
            archive.writestr("docProps/core.xml", core)


def test_docx_preserves_runs_tables_nested_text_boxes_and_truthful_locators(
    tmp_path,
):
    path = tmp_path / "paper.docx"
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    make_docx(
        path,
        f'''<w:document xmlns:w="{ns}"><w:body>
      <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Results</w:t></w:r></w:p>
      <w:p><w:r><w:t>A</w:t><w:tab/><w:t>B</w:t><w:br/><w:t>C</w:t></w:r></w:p>
      <w:tbl><w:tr><w:tc><w:p><w:r><w:t>Cell value</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
      <w:p><w:r><w:t>Outer.</w:t><w:drawing><w:txbxContent><w:p><w:r><w:t>Inner.</w:t></w:r></w:p></w:txbxContent></w:drawing></w:r></w:p>
    </w:body></w:document>''',  # noqa: E501
        '<core xmlns="urn:test"><title>Source title</title>'
        "<created>1990-01-01</created></core>",
    )
    document = parse_file(path)
    assert document.title == "Source title"
    texts = [chunk.text for chunk in document.chunks]
    assert "A\tB\nC" in texts
    assert texts.count("Outer.") == texts.count("Inner.") == 1
    assert any(
        chunk.kind == "table_cell_paragraph" and chunk.text == "Cell value"
        for chunk in document.chunks
    )
    assert all(
        "page" not in chunk.locator
        and chunk.locator["part"] == "word/document.xml"
        for chunk in document.chunks
    )
    assert all(chunk.section_path == ["Results"] for chunk in document.chunks)
    assert document.published_at is None
    assert document.metadata["parse_warnings"]


def test_html_keeps_visible_literal_rendered_text_once_and_excludes_script(
    tmp_path,
):
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><title>Title &amp; subtitle</title>"
        "<script>invented claim</script></head>"
        "<body>Intro<h1>Results</h1>"
        "<p>A <b>sensor</b> &amp; a baseline.</p>"
        "<ul><li><p>Nested statement.</p></li></ul>Tail</body></html>",
        encoding="utf-8",
    )
    document = parse_file(path)
    texts = [chunk.text for chunk in document.chunks]
    assert document.title == "Title & subtitle"
    assert texts == [
        "Intro",
        "Results",
        "A sensor & a baseline.",
        "Nested statement.",
        "Tail",
    ]
    assert document.chunks[2].section_path == ["Results"]
    assert all(
        chunk.locator["text_basis"] == "htmlparser_rendered_text_v1"
        for chunk in document.chunks
    )
    assert all("page" not in chunk.locator for chunk in document.chunks)


def test_empty_bad_encoding_and_unsupported_input_are_explicit(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_bytes(b"")
    document = parse_file(path)
    assert document.chunks == []
    assert "no_text_chunks" in document.metadata["parse_warnings"]
    # A-10: an empty file used to claim coverage "full_text".
    assert document.coverage == "metadata_only"
    path.write_bytes(b"\xff\xff")
    with pytest.raises(FileAdapterError, match="UTF-8"):
        parse_file(path)
    other = tmp_path / "unknown.bin"
    other.write_bytes(b"text")
    with pytest.raises(UnsupportedFileFormat):
        parse_file(other)


def test_docx_rejects_missing_document_and_custom_entities(tmp_path):
    path = tmp_path / "bad.docx"
    path.write_bytes(b"not a ZIP")
    with pytest.raises(FileAdapterError, match="DOCX"):
        parse_file(path)
    make_docx(path, '<!DOCTYPE x [<!ENTITY a "expanded">]><x>&a;</x>')
    with pytest.raises(FileAdapterError, match="entities"):
        parse_file(path)


def test_corrupted_existing_snapshot_is_not_overwritten(
    tmp_path, raw_directory
):
    path = tmp_path / "paper.txt"
    path.write_bytes(b"original")
    parse_file(path)
    snapshot = raw_directory / hashlib.sha256(b"original").hexdigest()
    snapshot.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="content hash"):
        parse_file(path)
    assert snapshot.read_bytes() == b"corrupted"


def test_pdf_dependency_is_optional_and_lazy(tmp_path, monkeypatch):
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-test")
    monkeypatch.setitem(sys.modules, "docling.document_converter", None)
    with pytest.raises(UnsupportedFileFormat, match="optional Docling"):
        parse_file(path)


def test_pdf_uses_exact_snapshot_and_only_reported_provenance(
    tmp_path, monkeypatch
):
    path = tmp_path / "paper.pdf"
    original = b"%PDF-original"
    path.write_bytes(original)

    class Provenance:
        def model_dump(self, mode):
            return {"page_no": 4, "bbox": {"l": 1, "t": 2, "r": 3, "b": 4}}

    item = SimpleNamespace(
        label=SimpleNamespace(value="text"),
        text="Original extracted text.",
        self_ref="#/texts/0",
        prov=[Provenance()],
    )

    class Converter:
        def convert(self, snapshot):
            assert snapshot.read_bytes() == original
            path.write_bytes(b"%PDF-changed-during-parse")
            return SimpleNamespace(
                document=SimpleNamespace(iterate_items=lambda: [(item, 0)]),
                status=SimpleNamespace(value="success"),
            )

    module = ModuleType("docling.document_converter")
    module.DocumentConverter = Converter
    monkeypatch.setitem(sys.modules, "docling.document_converter", module)
    document = parse_file(path)
    assert document.artifact.sha256 == hashlib.sha256(original).hexdigest()
    assert document.chunks[0].locator["provenance"][0]["page_no"] == 4
    assert "page" not in document.chunks[0].locator


def test_markdown_overlap_segments_map_only_real_source_slices():
    text = "word " * 100
    chunks = _markdown_chunks(
        "v", "readme", text, max_chars=100, overlap_chars=20
    )
    assert any(chunk.locator["overlap_chars"] for chunk in chunks)
    for chunk in chunks:
        for span in chunk.locator["source_segments"]:
            assert (
                chunk.text[span["chunk_start"] : span["chunk_end"]]
                == text[span["source_start"] : span["source_end"]]
            )
    no_overlap = _markdown_chunks(
        "v", "readme", text, max_chars=100, overlap_chars=0
    )
    assert all(
        chunk.locator["overlap_chars"] == 0 and len(chunk.text) <= 100
        for chunk in no_overlap
    )


def test_api_markdown_chunks_preserve_heading_markers_blank_lines_and_repeated_source_positions():  # noqa: E501
    text = (
        "Preface.\r\n\r\n# Intro\r\n  A statement.  \r\n \r\n"
        + "Repeated sentence. " * 30
        + "\r\n## Limits\r\nAuthors do not claim improvement.\r\n"
    )
    chunks = _markdown_chunks(
        "v", "description", text, max_chars=100, overlap_chars=20
    )
    assert chunks[0].section_path == ["description"]
    assert any(
        chunk.section_path == ["description", "Intro"]
        and "# Intro" in chunk.text
        for chunk in chunks
    )
    assert chunks[-1].section_path == ["description", "Intro", "Limits"]
    covered = set()
    for chunk in chunks:
        start, end = chunk.locator["source_start"], chunk.locator["source_end"]
        assert chunk.text == text[start:end]
        assert len(chunk.text) <= 100
        assert chunk.locator["source_segments"] == [
            {
                "chunk_start": 0,
                "chunk_end": len(chunk.text),
                "source_start": start,
                "source_end": end,
            }
        ]
        covered.update(range(start, end))
    assert all(
        position in covered
        for position, char in enumerate(text)
        if not char.isspace()
    )


def test_each_release_has_its_own_source_stream_for_overlap_coordinates():
    document = parse_github(
        {
            "repository": {"full_name": "org/repo"},
            "releases": [
                {"id": 1, "body": "A release"},
                {"id": 2, "body": "B release"},
            ],
        }
    )
    assert (
        document.chunks[0].locator["source_stream_id"]
        != document.chunks[1].locator["source_stream_id"]
    )


def test_snapshot_is_published_only_after_complete_fsynced_bytes(
    tmp_path, monkeypatch
):
    from lctrend.ingest import snapshots

    raw = b"immutable source" * 100
    original_link = snapshots.os.link

    def checked_link(source, destination):
        assert source.read_bytes() == raw
        assert not destination.exists()
        assert source.parent == destination.parent
        original_link(source, destination)

    monkeypatch.setattr(snapshots.os, "link", checked_link)
    path = snapshots.snapshot_bytes(raw, tmp_path / "snapshots")
    assert path.read_bytes() == raw
    assert list(path.parent.iterdir()) == [path]


def test_failure_before_snapshot_publication_leaves_no_partial_hash_file(
    tmp_path, monkeypatch
):
    from lctrend.ingest import snapshots

    def interrupted_fsync(descriptor):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(snapshots.os, "fsync", interrupted_fsync)
    directory = tmp_path / "snapshots"
    with pytest.raises(RuntimeError, match="interruption"):
        snapshots.snapshot_bytes(b"original", directory)
    assert list(directory.iterdir()) == []


def test_concurrent_snapshot_writers_publish_once_without_false_corruption(
    tmp_path, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from lctrend.ingest import snapshots

    barrier = Barrier(2)
    original_link = snapshots.os.link
    raw = b"simultaneous source" * 10000

    def concurrent_link(source, destination):
        assert source.read_bytes() == raw
        barrier.wait(timeout=5)
        original_link(source, destination)

    monkeypatch.setattr(snapshots.os, "link", concurrent_link)
    directory = tmp_path / "snapshots"
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(snapshots.snapshot_bytes, raw, directory)
        second = pool.submit(snapshots.snapshot_bytes, raw, directory)
        paths = [first.result(timeout=10), second.result(timeout=10)]
    assert paths[0] == paths[1]
    assert paths[0].read_bytes() == raw
    assert list(directory.iterdir()) == [paths[0]]


def test_local_pdf_gets_the_same_body_as_an_openalex_full_text(
    tmp_path, monkeypatch
):
    # A-3: Docling labels must not split a local PDF into many packets.
    from lctrend.llm.context import PipelineSettings, plan_packets

    monkeypatch.setenv("LCTREND_RAW_DIR", str(tmp_path / "raw"))
    items = [
        ("page_header", "Journal of Fixtures, vol. 1"),
        ("section_header", "2 Methods"),
        ("text", "Sparse attention reduces memory use by 40%."),
        ("caption", "Figure 1. Memory use."),
        ("text", "It was tested on long documents."),
        ("page_footer", "Page 3"),
        ("section_header", "References"),
        ("list_item", "[1] Someone. A cited paper. 2020."),
    ]
    document = SimpleNamespace(
        iterate_items=lambda: [
            (
                SimpleNamespace(
                    label=SimpleNamespace(value=label),
                    text=text,
                    self_ref=f"#/texts/{index}",
                    prov=[],
                ),
                0,
            )
            for index, (label, text) in enumerate(items)
        ]
    )

    class Converter:
        def convert(self, path, **options):
            return SimpleNamespace(
                document=document, status=SimpleNamespace(value="success")
            )

    module = ModuleType("docling.document_converter")
    module.DocumentConverter = Converter
    monkeypatch.setitem(sys.modules, "docling.document_converter", module)
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-local")
    parsed = parse_file(path)
    assert [chunk.text for chunk in parsed.chunks] == [
        "2 Methods",
        "Sparse attention reduces memory use by 40%.",
        "Figure 1. Memory use.",
        "It was tested on long documents.",
    ]
    assert {chunk.kind for chunk in parsed.chunks} == {"fulltext"}
    assert parsed.chunks[1].locator["docling_label"] == "text"
    assert parsed.chunks[1].locator["section_role"] == "method"
    plan = plan_packets(
        parsed,
        PipelineSettings.from_catalog().model_copy(
            update={"primary_chunks": 6}
        ),
    )
    assert len(plan.packets) == 1


FAKE_DOCLING = """
import time
from pathlib import Path
from types import SimpleNamespace


class DocumentConverter:
    def convert(self, path, max_num_pages=None):
        raw = Path(path).read_bytes()
        if b"stuck" in raw:
            time.sleep(120)
        item = SimpleNamespace(
            label=SimpleNamespace(value="text"),
            text=raw.decode(),
            self_ref="#/texts/0",
            prov=[],
        )
        return SimpleNamespace(
            document=SimpleNamespace(iterate_items=lambda: [(item, 0)]),
            status=SimpleNamespace(value="success"),
        )
"""


def test_stuck_pdf_is_killed_and_the_next_pdf_converts(tmp_path, monkeypatch):
    # A-4: a thread timeout left Docling running with the converter held.
    import time

    from lctrend.ingest import file_adapters

    package = tmp_path / "site" / "docling"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "document_converter.py").write_text(
        FAKE_DOCLING, encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    for name in ("docling", "docling.document_converter"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(file_adapters, "IN_PROCESS_DOCLING", False)
    worker = file_adapters._DoclingProcess()
    monkeypatch.setattr(file_adapters, "_DOCLING", worker)
    monkeypatch.setenv("LCTREND_RAW_DIR", str(tmp_path / "raw"))
    catalog = file_adapters.load_catalog

    def limits(timeout):
        def load(name):
            value = catalog(name)
            if name == "pipeline":
                value["file_limits"]["pdf_timeout_seconds"] = timeout
            return value

        monkeypatch.setattr(file_adapters, "load_catalog", load)

    def pdf(name, text):
        path = tmp_path / name
        path.write_bytes(text.encode())
        return path

    try:
        limits(60)  # the first call also starts the worker
        first = parse_file(pdf("first.pdf", "Sparse attention works."))
        assert first.chunks[0].text == "Sparse attention works."
        limits(1)
        started = time.monotonic()
        with pytest.raises(file_adapters.DoclingTimeout):
            parse_file(pdf("stuck.pdf", "stuck forever"))
        assert time.monotonic() - started < 10
        assert worker.process is None
        limits(60)
        after = parse_file(pdf("after.pdf", "Next paper converts."))
        assert after.chunks[0].text == "Next paper converts."
    finally:
        worker.stop()


def test_docling_pool_converts_pdfs_in_parallel():
    import threading
    import time
    from pathlib import Path

    from lctrend.ingest import file_adapters

    running, peak, lock = [0], [0], threading.Lock()

    class FakeProcess:
        def convert(self, path, max_pages, timeout):
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            time.sleep(0.2)
            with lock:
                running[0] -= 1
            return ([], "success")

    pool = file_adapters._DoclingPool()
    # Two converter processes, four PDFs: two convert at once.
    pool._free = [FakeProcess(), FakeProcess()]
    pool._slots = threading.BoundedSemaphore(2)
    threads = [
        threading.Thread(target=pool.convert, args=(Path("a.pdf"), 5, 10))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert peak[0] == 2
    assert len(pool._free) == 2
