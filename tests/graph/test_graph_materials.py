"""Prior graph extractions retain their material identities in crawl state."""

from lctrend.graph.store import GraphStore


class Record(dict):
    def data(self):
        return dict(self)


class Session:
    def __init__(self, records):
        self.records = records

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def run(self, query):
        if "db.labels" in query:
            return [
                {"label": label}
                for label in ("ProcessingRun", "Source", "Document")
            ]
        return self.records


class Driver:
    def __init__(self, records):
        self.records = records

    def session(self, **_):
        return Session(self.records)


def test_prior_repository_uses_url_not_github_node_id_and_paper_uses_doi():
    store = GraphStore.__new__(GraphStore)
    store._database = None
    store._driver = Driver(
        [
            Record(
                source="source:github",
                source_id="R_kgDOABC123",
                title="Project",
                url="https://github.com/Owner/Project",
                external_ids=["github:r_kgdoabc123"],
                statuses=["succeeded"],
            ),
            Record(
                source="source:openalex",
                source_id="W123",
                title="Paper",
                url="https://doi.org/10.1234/Paper",
                external_ids=["doi:10.1234/paper"],
                statuses=["partial"],
            ),
        ]
    )
    repo, paper = list(store.processed_materials())
    assert repo["canonical_id"] == "github:owner/project"
    assert repo["source_id"] == "owner/project"
    assert repo["status"] == "parsed"
    assert paper["canonical_id"] == "doi:10.1234/paper"
    assert paper["status"] == "partial"
