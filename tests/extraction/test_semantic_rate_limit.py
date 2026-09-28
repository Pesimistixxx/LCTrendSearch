"""HTTP 429 on embeddings is a busy moment, not a broken semantic layer."""

from lctrend.extraction.resolver import SemanticDeduplicator
from lctrend.llm.client import LLMError


class Busy:
    def __init__(self, status):
        self.status = status
        self.calls = 0

    async def embed(self, texts, model):
        self.calls += 1
        raise LLMError(
            "http_error", "busy", True, retry_after=0, status=self.status
        )


def semantic(status):
    return SemanticDeduplicator(
        embedding_provider="gigachat", embedder=Busy(status)
    )


def test_rate_limited_embeddings_keep_the_layer_on():
    layer = semantic(429)

    assert layer.embed(["graph neural network"]) is None
    assert layer.failure is None and layer.available()


def test_other_failures_still_pause_the_layer():
    layer = semantic(500)

    assert layer.embed(["graph neural network"]) is None
    assert layer.failure == "LLMError" and not layer.available()
