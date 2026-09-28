"""One work, several documents: OpenAlex, PDF, preprint, A1 and B1."""

import asyncio
import sys
from datetime import date
from itertools import permutations

from lctrend import cli
from lctrend.graph.temporal import TemporalCorpus
from lctrend.graph.works import (
    FoundWork,
    WorkIndex,
    WorkKey,
    link_stored_works,
    normalize_arxiv,
    patent_number,
    plan_work,
    stored_work_keys,
    work_keys,
)

TITLE = "A new inhibitor of cholesterol synthesis produced by fungi"


def keys(document_type="article", title=TITLE, **identifiers):
    return work_keys(document_type, list(identifiers.items()), title)


def test_identifiers_are_normalized_to_one_key_per_work():
    assert {key.key for key in keys(doi="https://doi.org/10.1234/ABC")} == {
        "doi:10.1234/abc",
        "title:" + " ".join(key for key in keys()[0].key[6:].split()),
    }
    # arXiv's DOI, URL and versioned id name one preprint.
    for value in (
        "10.48550/arXiv.2101.00001",
        "https://arxiv.org/abs/2101.00001v3",
        "arXiv:2101.00001v1",
    ):
        assert normalize_arxiv(value) == "2101.00001"
    assert keys(doi="10.48550/arXiv.2101.00001", title=None) == [
        WorkKey("arxiv:2101.00001", "arxiv")
    ]
    assert normalize_arxiv("hep-th/9901001v2") == "hep-th/9901001"
    assert patent_number("EP 1 234 567 B1") == "EP1234567"
    assert patent_number("ep1234567a1") == "EP1234567"
    assert keys(pypi="Scikit_Learn", title=None) == [
        WorkKey("pypi:scikit-learn", "pypi")
    ]


def test_stored_external_ids_give_the_same_keys_as_the_envelope():
    fresh = work_keys(
        "patent",
        [("patent-publication", "EP1234567A1"), ("openalex", "W12")],
        "Method",
        {"family_id": "5501"},
    )
    stored = stored_work_keys(
        "patent",
        ["patent-publication:ep1234567a1", "openalex:w12"],
        "Method",
        {"family_id": "5501"},
    )
    assert fresh == stored
    assert WorkKey("patent-family:5501", "patent-family") in fresh


def test_short_titles_patents_grants_and_postings_get_no_title_key():
    assert keys(title="Editorial note") == []
    assert keys("patent", title=TITLE) == []
    assert keys("repository", title=TITLE) == []
    assert keys("grant", title=TITLE, nih="123") == [
        WorkKey("nih:123", "nih")
    ]


def title_key():
    return next(key for key in keys() if not key.strong)


def test_a_title_joins_a_preprint_and_its_journal_version():
    journal = FoundWork(
        "work:j", [WorkKey("doi:10.1/j", "doi"), title_key()], {title_key().key}
    )
    preprint_keys = [WorkKey("arxiv:2101.00001", "arxiv"), title_key()]
    plan = plan_work("document:p", preprint_keys, [journal])
    assert plan.work_id == "work:j" and plan.rejected == []


def test_a_title_never_joins_two_papers_with_different_dois():
    other = FoundWork(
        "work:o", [WorkKey("doi:10.1/o", "doi"), title_key()], {title_key().key}
    )
    plan = plan_work(
        "document:n", [WorkKey("doi:10.1/n", "doi"), title_key()], [other]
    )
    assert plan.work_id != "work:o"
    assert plan.rejected == ["work:o"]
    # The title now names both works, so it joins neither from now on.
    assert title_key() in plan.keys


def test_a_record_with_both_identifiers_folds_two_works():
    doi = FoundWork("work:b", [WorkKey("doi:10.1/x", "doi")], {"doi:10.1/x"})
    arxiv = FoundWork(
        "work:a", [WorkKey("arxiv:1", "arxiv")], {"arxiv:1"}
    )
    plan = plan_work(
        "document:both",
        [WorkKey("doi:10.1/x", "doi"), WorkKey("arxiv:1", "arxiv")],
        [doi, arxiv],
    )
    assert (plan.work_id, plan.absorbed) == ("work:a", ["work:b"])


DOCUMENTS = [
    ("openalex", keys(doi="10.1234/abc", openalex="W1")),
    ("pdf", keys(doi="https://doi.org/10.1234/ABC")),
    ("preprint", keys(doi="10.48550/arXiv.2101.00001", openalex="W2")),
    ("other", keys(doi="10.9999/other", openalex="W3")),
    ("a1", keys("patent", patent="EP1234567A1")),
    ("b1", keys("patent", patent="EP1234567B1")),
]


def test_strong_keys_group_documents_in_any_order():
    for order in permutations(range(len(DOCUMENTS))):
        index = WorkIndex()
        for position in order:
            index.add(*DOCUMENTS[position])
        groups = index.groups()
        assert {"a1", "b1"} in groups
        assert any({"openalex", "pdf"} <= group for group in groups)
        # A different DOI is never the same work, whatever came first.
        assert not any(
            "other" in group and "openalex" in group for group in groups
        )
        # The preprint joins at most one of two papers with different DOIs
        # (which one can depend on arrival order when it comes first).
        assert not any(
            {"preprint", "other", "openalex"} <= group for group in groups
        )


def test_a_title_naming_two_papers_joins_neither():
    index = WorkIndex()
    index.add("other", keys(doi="10.9999/other"))
    index.add("paper", keys(doi="10.1234/abc"))
    index.add("preprint", keys(doi="10.48550/arXiv.2101.00001"))
    assert {"preprint"} in index.groups()
    assert index.rejected == 1


def test_a_work_is_counted_once_in_a_snapshot():
    corpus = TemporalCorpus(
        {
            "versions": [
                {
                    "document_id": document,
                    "work_id": "work:1",
                    "version_id": version,
                    "document_type": "article",
                    "version_published_at": "2020-01-01",
                    "retrieved_at": retrieved,
                    "title": "Compactin",
                    "url": url,
                }
                for document, version, retrieved, url in (
                    ("openalex", "v1", "2020-01-02", "https://doi.org/x"),
                    ("pdf", "v2", "2020-02-01", None),
                )
            ],
            "mentions": [
                {
                    "technology_id": "t1",
                    "version_id": version,
                    "observed_at": "2020-01-01",
                    "mentions": count,
                }
                for version, count in (("v1", 1), ("v2", 4))
            ],
        }
    )
    documents = corpus.view(date(2021, 1, 1)).technologies["t1"].documents
    assert [item.document_id for item in documents] == ["work:1"]
    assert documents[0].mention_count == 4
    assert documents[0].version.source_document_id == "pdf"
    assert corpus.document_info["work:1"]["url"] == "https://doi.org/x"


class Store:
    def __init__(self, rows):
        self.rows = rows
        self.written = None

    async def read_document_identities(self):
        return self.rows

    async def write_works(self, documents):
        self.written = documents


ROWS = [
    {
        "document_id": "d1",
        "document_type": "patent",
        "title": "Method",
        "external_ids": ["patent-publication:ep1a1"],
        "metadata_json": ['{"family_id": "77"}'],
    },
    {
        "document_id": "d2",
        "document_type": "patent",
        "title": "Method",
        "external_ids": ["patent-publication:us2b2"],
        "metadata_json": [None, '{"family_id": "77"}'],
    },
]


def test_backfill_groups_a_patent_family_and_is_a_dry_run_by_default():
    store = Store(ROWS)
    summary = asyncio.run(link_stored_works(store))
    assert (summary["documents"], summary["works"]) == (2, 1)
    assert summary["documents_in_shared_works"] == 2
    assert store.written is None
    asyncio.run(link_stored_works(store, apply=True))
    assert [document for document, _ in store.written] == ["d1", "d2"]


def test_cli_link_works_is_a_dry_run_by_default(monkeypatch, capsys):
    store = Store(ROWS)

    class Opened:
        async def __aenter__(self):
            return store

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setattr(cli, "_store", lambda: Opened())
    monkeypatch.setattr(sys, "argv", ["lctrend", "link-works"])
    cli.main()
    assert store.written is None
    assert '"works": 1' in capsys.readouterr().out
