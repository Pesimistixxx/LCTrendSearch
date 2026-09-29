"""Link catalogs (awesome lists) are not technology sources."""

from lctrend.ingest.adapters import link_catalog, parse_github

CATALOG = "# Awesome ML\n" + "\n".join(
    f"- [Lib{index}](https://example.org/{index}) - a library"
    for index in range(40)
)


def test_link_catalog_is_detected_but_a_readme_with_links_is_not():
    assert link_catalog(CATALOG)
    assert not link_catalog(
        "# Project\n\nWe train a model.\n\n- [Docs](https://d)\n- Setup\n"
    )


def test_catalog_readme_is_not_sent_to_the_model():
    payload = {
        "repository": {
            "id": 1,
            "full_name": "someone/awesome-ml",
            "html_url": "https://github.com/someone/awesome-ml",
        },
        "readme": {"text": CATALOG, "path": "README.md"},
    }
    document = parse_github(payload)
    assert not [chunk for chunk in document.chunks if chunk.kind == "readme"]
    assert document.metadata["parse_warnings"] == [
        "readme_link_catalog_skipped"
    ]
