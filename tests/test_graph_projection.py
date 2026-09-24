from lctrend.adapters import parse_openalex
from lctrend.graph import GraphStore


class Result:
    def consume(self):
        return None


class Transaction:
    def __init__(self):
        self.queries = []

    def run(self, query, **parameters):
        self.queries.append((query, parameters))
        return Result()


def test_document_projection_builds_queries_without_dynamic_cypher_errors():
    document = parse_openalex(
        {
            "id": "https://openalex.org/W1",
            "title": "Example",
            "authorships": [
                {"author": {"id": "https://openalex.org/A1", "display_name": "Ada"}}
            ],
        }
    )
    tx = Transaction()
    GraphStore._write_document(tx, document)
    assert any(":Person" in query for query, _ in tx.queries)
    assert any("IDENTIFIED_BY" in query for query, _ in tx.queries)
