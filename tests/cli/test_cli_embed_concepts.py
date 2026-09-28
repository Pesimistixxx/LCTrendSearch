import asyncio

from lctrend import cli


class FakeStore:
    def __init__(self, pending):
        self.pending = pending
        self.asked = None
        self.written = []

    async def read_concepts_to_embed(self, kinds, model, force=False):
        self.asked = (tuple(kinds), model, force)
        return list(self.pending)

    async def write_concept_embeddings(self, rows, model):
        self.written.append((rows, model))


class FakeSemantic:
    embedding_model_name = "EmbeddingsGigaR"
    batch_size = 2
    failure = None

    def __init__(self, fail=False):
        self.fail = fail

    def embed(self, texts):
        if self.fail:
            self.failure = "LLMError"
            return None
        return [[float(len(text)), 1.0] for text in texts]


def test_missing_vectors_are_backfilled_in_batches(monkeypatch):
    from lctrend.extraction import processing

    monkeypatch.setattr(
        processing, "_semantic_deduplicator", lambda: FakeSemantic()
    )
    pending = [
        {"concept_id": f"c{i}", "label": name, "node_label": "Technology"}
        for i, name in enumerate(["edge ai", "digital twin", "lidar"])
    ]
    store = FakeStore(pending)
    summary = asyncio.run(cli._embed_concepts(store))
    assert summary == {
        "model": "EmbeddingsGigaR",
        "pending": 3,
        "embedded": 3,
    }
    assert "Technology" in store.asked[0] and store.asked[1:] == (
        "EmbeddingsGigaR",
        False,
    )
    assert [len(rows) for rows, _ in store.written] == [2, 1]
    assert store.written[0][0][0]["vector"] == [7.0, 1.0]


def test_backfill_stops_when_the_endpoint_is_unavailable(monkeypatch):
    import pytest

    from lctrend.extraction import processing

    monkeypatch.setattr(
        processing, "_semantic_deduplicator", lambda: FakeSemantic(True)
    )
    store = FakeStore(
        [{"concept_id": "c", "label": "x", "node_label": "Method"}]
    )
    with pytest.raises(RuntimeError, match="0 vectors written"):
        asyncio.run(cli._embed_concepts(store))
    assert store.written == []


def test_semantic_cache_is_seeded_once_per_model(monkeypatch):
    from lctrend.extraction import processing
    from lctrend.extraction.resolver import SemanticDeduplicator

    semantic = SemanticDeduplicator(embedder=object())
    monkeypatch.setattr(processing, "_llm_semantic", lambda: semantic)
    reads = []

    class Store:
        async def read_label_vectors(self, kinds, model):
            reads.append(model)
            return [("edge ai", [1.0, 0.0])]

    asyncio.run(processing.seed_semantic(Store()))
    asyncio.run(processing.seed_semantic(Store()))
    assert reads == ["EmbeddingsGigaR"]
    assert semantic.embed(["Edge AI"]) == [[1.0, 0.0]]
