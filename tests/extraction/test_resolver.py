from lctrend.core.models import Concept, ConceptKind, ConceptName, Mention
from lctrend.extraction.resolver import (
    alias_keys,
    normalize_name,
    resolve_exact_mentions,
    resolve_mentions,
)


def mention(text="Натрий-ионный аккумулятор"):
    return Mention(
        mention_id="m1",
        chunk_id="c1",
        surface_text=text,
        start=0,
        end=len(text),
        type_candidates=[ConceptKind.TECHNOLOGY],
    )


def test_reviewed_exact_name_is_resolved():
    concept = Concept(
        concept_id="tech:sodium",
        kind=ConceptKind.TECHNOLOGY,
        preferred_label="Натрий-ионный аккумулятор",
        status="accepted",
        names=[
            ConceptName(
                name_id="n1",
                text="Натрий-ионный аккумулятор",
                normalized_text=normalize_name("Натрий-ионный аккумулятор"),
            )
        ],
    )
    new, decisions = resolve_exact_mentions(
        [mention("  НАТРИЙ-ИОННЫЙ   аккумулятор ")], [concept]
    )
    assert new == [concept]
    assert decisions[0].concept_id == "tech:sodium"
    assert decisions[0].status == "accepted"


def test_similar_but_distinct_technology_is_not_merged():
    lithium = Concept(
        concept_id="tech:lithium",
        kind=ConceptKind.TECHNOLOGY,
        preferred_label="Литий-ионный аккумулятор",
        status="accepted",
    )
    new, decisions = resolve_exact_mentions([mention()], [lithium])
    assert new[0].preferred_label == "Натрий-ионный аккумулятор"
    assert new[0].status == "provisional"
    assert decisions[0].concept_id != "tech:lithium"


def test_repeated_exact_mentions_share_one_provisional_concept():
    first = mention("Sensor")
    second = mention("sensor")
    second.mention_id = "m2"
    new, decisions = resolve_exact_mentions([first, second], [])
    assert len(new) == 1
    assert decisions[0].concept_id == decisions[1].concept_id


def test_abbreviation_and_full_name_share_one_concept():
    first = mention("NLP")
    second = mention("natural language processing")
    second.mention_id = "m2"
    concepts, decisions = resolve_exact_mentions([first, second], [])
    assert len(concepts) == 1
    assert decisions[0].concept_id == decisions[1].concept_id
    assert {name.text for name in concepts[0].names} == {
        "NLP",
        "natural language processing",
    }


def test_normalization_and_stemming():
    assert normalize_name("  Graph-based_NER™ ") == "graph based ner"
    # Key v2 strips an English plural and stems Russian with Snowball; the
    # optional simplemma lemmatizer is no longer part of identity.
    assert alias_keys("models") & alias_keys("model")
    assert alias_keys("  MODEL ") & alias_keys("model")
    assert alias_keys("языковые модели") & alias_keys("языковой модели")


def test_same_initials_do_not_establish_identity():
    first = mention("carbon capture")
    second = mention("cloud computing")
    second.mention_id = "m2"
    concepts, decisions = resolve_exact_mentions([first, second], [])
    assert len(concepts) == 2
    assert decisions[0].concept_id != decisions[1].concept_id
    assert not alias_keys("carbon capture") & alias_keys("cloud computing")


def test_semantic_similarity_is_only_a_review_candidate():
    existing = Concept(
        concept_id="tech:one",
        kind=ConceptKind.TECHNOLOGY,
        preferred_label="carbon capture",
        status="accepted",
    )

    class FakeSemantic:
        def best_match(self, text, concepts):
            return existing, 0.99, 0.99

    touched, decisions = resolve_mentions(
        [mention("carbon conversion")], [existing], FakeSemantic()
    )
    assert touched == [existing]
    assert decisions[0].status == "ambiguous"
    assert decisions[0].concept_id is None
    assert decisions[0].review_status == "pending"
    assert decisions[0].candidates == [
        {"concept_id": "tech:one", "score": 0.99}
    ]
    assert existing.names == []


def test_unreviewed_observed_alias_does_not_merge_a_future_mention():
    existing = Concept(
        concept_id="tech:one",
        kind=ConceptKind.TECHNOLOGY,
        preferred_label="carbon capture",
        status="accepted",
        names=[
            ConceptName(
                name_id="observed",
                text="carbon conversion",
                normalized_text="carbon conversion",
                status="provisional",
            )
        ],
    )
    touched, decisions = resolve_exact_mentions(
        [mention("carbon conversion")], [existing]
    )
    assert touched[0].concept_id != existing.concept_id
    assert decisions[0].status == "provisional"


def test_gigachat_embeddings_batch_and_cache_candidates():
    from lctrend.extraction.resolver import SemanticDeduplicator

    vectors = {
        "carbon conversion": [1.0, 0.1],
        "carbon capture": [1.0, 0.0],
        "quantum computing": [0.0, 1.0],
    }
    requests = []

    class FakeEmbedder:
        def embed(self, texts, model):
            requests.append((list(texts), model))
            return [vectors[text] for text in texts]

    semantic = SemanticDeduplicator(embedder=FakeEmbedder())
    semantic._decision_score = lambda left, right: 0.9
    concepts = [
        Concept(
            concept_id=f"tech:{index}",
            kind=ConceptKind.TECHNOLOGY,
            preferred_label=label,
            status="accepted",
        )
        for index, label in enumerate(["carbon capture", "quantum computing"])
    ]
    match, cosine, decision = semantic.best_match(
        "carbon conversion", concepts
    )
    assert semantic.embedding_provider == "gigachat"
    assert match.concept_id == "tech:0" and cosine > 0.99 and decision == 0.9
    assert requests == [
        (
            ["carbon conversion", "carbon capture", "quantum computing"],
            "EmbeddingsGigaR",
        )
    ]
    semantic.best_match("Carbon  Capture", concepts)
    assert len(requests) == 1, "normalized texts are embedded once"


def _semantic_with(embedder):
    from lctrend.extraction.resolver import SemanticDeduplicator

    return SemanticDeduplicator(embedder=embedder)


def test_transient_embedding_errors_are_retried(monkeypatch):
    from lctrend.extraction import resolver
    from lctrend.llm.client import LLMError

    monkeypatch.setattr(resolver, "sleep", lambda seconds: None)
    calls = []

    class FlakyEmbedder:
        def embed(self, texts, model):
            calls.append(list(texts))
            if len(calls) < 3:
                raise LLMError("timeout", "slow", retryable=True)
            return [[1.0, 0.0] for _ in texts]

    semantic = _semantic_with(FlakyEmbedder())
    assert semantic.embed(["edge ai"]) == [[1.0, 0.0]]
    assert len(calls) == 3 and semantic.failure is None


def test_permanent_embedding_error_is_not_retried(monkeypatch):
    from lctrend.extraction import resolver
    from lctrend.llm.client import LLMError

    monkeypatch.setattr(resolver, "sleep", lambda seconds: None)
    calls = []

    class BrokenEmbedder:
        def embed(self, texts, model):
            calls.append(list(texts))
            raise LLMError("http_error", "forbidden", retryable=False)

    semantic = _semantic_with(BrokenEmbedder())
    assert semantic.embed(["edge ai"]) is None
    assert len(calls) == 1 and semantic.failure == "LLMError"


def test_missing_cross_encoder_keeps_the_embedding_layer():
    class Embedder:
        def embed(self, texts, model):
            return [[1.0, 0.0] for _ in texts]

    semantic = _semantic_with(Embedder())

    def no_torch(left, right):
        raise ModuleNotFoundError("torch")

    semantic._decision_score = no_torch
    concept = Concept(
        concept_id="tech:a",
        kind=ConceptKind.TECHNOLOGY,
        preferred_label="carbon capture",
        status="accepted",
    )
    match, cosine, decision = semantic.best_match(
        "carbon conversion", [concept]
    )
    assert match is None and cosine > 0.99 and decision == 0.0
    assert semantic.decision_failure == "ModuleNotFoundError"
    assert semantic.available() and semantic.failure is None
    assert semantic.embed(["quantum annealing"]) == [[1.0, 0.0]]


def test_document_names_are_embedded_in_one_request():
    requests = []

    class Embedder:
        def embed(self, texts, model):
            requests.append(list(texts))
            return [[1.0, float(index)] for index, _ in enumerate(texts)]

    semantic = _semantic_with(Embedder())
    semantic._decision_score = lambda left, right: 0.0
    mentions = [
        mention(text).model_copy(update={"mention_id": f"m{index}"})
        for index, text in enumerate(
            ["edge ai", "quantum annealing", "digital twin"]
        )
    ]
    resolve_mentions(mentions, [], semantic)
    assert requests[0] == ["edge ai", "quantum annealing", "digital twin"]
    # Later lookups hit the cache: no request per mention.
    assert all(len(batch) > 1 for batch in requests)


def test_stored_label_vectors_seed_the_cache():
    requests = []

    class Embedder:
        def embed(self, texts, model):
            requests.append(list(texts))
            return [[0.0, 1.0] for _ in texts]

    semantic = _semantic_with(Embedder())
    assert semantic.seed([("Carbon capture", [3.0, 4.0]), ("", [1.0])]) == 1
    assert semantic.seeded_model == "EmbeddingsGigaR"
    assert semantic.embed(["carbon  capture"]) == [[0.6, 0.8]]
    assert requests == [], "a preloaded label is not requested again"
