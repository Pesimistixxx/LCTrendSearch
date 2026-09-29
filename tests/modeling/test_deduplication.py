import asyncio

import pytest

from lctrend.modeling.dataset import deduplication
from lctrend.modeling.dataset.deduplication import (
    ConceptDeduplicator,
    kind_groups,
    lexical_support,
    variant_reason,
)


def _concept(identifier, vector, mentions=1, kind="Technology", label=None):
    return {
        "concept_id": identifier,
        "label": label or identifier,
        "kind": kind,
        "status": "accepted",
        "mention_count": mentions,
        "first_seen_at": "2020-01-01",
        "vector": vector,
    }


def _plan(rows, kinds=("Technology", "Method", "Material"), **options):
    deduplicator = ConceptDeduplicator(kinds, **options)
    deduplicator.load(rows)
    return deduplicator.plan()


def test_star_takes_only_concepts_similar_to_the_survivor():
    # a-b and b-c are above 0.95, a-c is not. b has the most mentions and
    # takes both; labels confirm the merge.
    rows = [
        _concept("a", [1.0, 0.0], 1, label="Backpropagation"),
        _concept("b", [0.97, 0.243], 5, label="back-propagation"),
        _concept("c", [0.88, 0.475], 1, label="Backpropagation algorithm"),
        _concept("far", [0.0, 1.0], 9),
    ]
    plan = _plan(rows)
    assert len(plan["groups"]) == 1
    group = plan["groups"][0]
    assert group["canonical"]["concept_id"] == "b"
    assert {m["concept_id"] for m in group["members"]} == {"a", "c"}


def test_chain_member_is_not_merged_through_a_middle_concept():
    rows = [
        _concept("a", [1.0, 0.0], 9, label="Backpropagation"),
        _concept("b", [0.97, 0.243], 5, label="back-propagation"),
        _concept("c", [0.88, 0.475], 1, label="Backpropagation algorithm"),
    ]
    plan = _plan(rows)
    merged = {m["concept_id"] for g in plan["groups"] for m in g["members"]}
    # c is similar to b only, and b is already taken by a.
    assert merged == {"b"}


@pytest.mark.parametrize(
    "left, right, reason",
    [
        ("GPT3-175B", "GPT3-13B", "different numbers or versions"),
        ("YOLOv3", "YOLOv1", "different numbers or versions"),
        ("ResNet-18", "ResNet", "different numbers or versions"),
        (
            "python核心编程(第三版)",
            "python核心编程(第二版)",
            "different numbers or versions",
        ),
        ("Bidirectional LSTM", "Bidirectional GRU", "different acronyms"),
        (
            "i.i.d. partitioning",
            "non-i.i.d. partitioning",
            "negation on one side",
        ),
        (
            "Explainable Artificial Intelligence (XAI)",
            "Explainable AI (XAI)",
            None,
        ),
        ("MobileNet V2", "MobileNetV2", None),
    ],
)
def test_variants_are_held(left, right, reason):
    assert variant_reason(left, right) == reason


@pytest.mark.parametrize(
    "left, right, support",
    [
        ("Backpropagation", "Backpropagation algorithm", "words contained"),
        ("privacy protection", "Privacy Protections", "words contained"),
        ("Federated Averaging", "FederatedAveraging", "words contained"),
        (
            "Decentralised Identifiers",
            "Decentralized Identifier",
            "spelling variant",
        ),
        ("IoT devices", "Internet of Things devices", "acronym expansion"),
        ("Paxos", "PoP", None),
        ("Block-based DAG", "Transaction-based DAG", None),
    ],
)
def test_lexical_support(left, right, support):
    assert lexical_support(left, right) == support


def test_similar_embedding_alone_goes_to_review_unless_very_close():
    rows = [
        _concept("a", [1.0, 0.0], 9, label="Paxos"),
        _concept("b", [0.96, 0.28], 1, label="PoP"),
    ]
    plan = _plan(rows)
    assert plan["merges"] == 0
    assert [item["concept_id"] for item in plan["review"]] == ["b"]
    # The same pair merges when the cosine clears the semantic threshold.
    assert _plan(rows, semantic_threshold=0.955)["merges"] == 1


def test_families_are_separate_and_grouped():
    assert kind_groups(["Technology", "Task", "Method", "Problem"]) == [
        ["Technology", "Method"],
        ["Task"],
        ["Problem"],
    ]
    with pytest.raises(ValueError, match="kind family"):
        ConceptDeduplicator(["Technology", "Task"])
    rows = [
        _concept("tech", [1.0, 0.0], label="LoRA"),
        _concept("task", [1.0, 0.0], kind="Task", label="LoRA"),
    ]
    assert _plan(rows)["groups"] == []


class _Store:
    def __init__(self, merged=()):
        self.calls = []
        self.merged = set(merged)

    async def merge_concepts(self, source, target, reason=None):
        if source in self.merged:
            raise ValueError(f"concept {source} is already merged")
        self.calls.append((source, target, reason))
        return {"source": source, "target": target}


def test_run_merges_each_star_and_reports_survivor_links(monkeypatch):
    rows = [
        _concept("a", [1.0, 0.0], 9, label="Federated Averaging"),
        _concept("b", [0.99, 0.141], 1, label="Federated Averaging (FedAvg)"),
        _concept("c", [0.99, -0.141], 1, label="federated averaging"),
    ]
    counts = iter([{"a": 3}, {"a": 11}])

    async def candidates(store, model, kinds):
        return rows

    async def links(store, ids):
        return next(counts)

    monkeypatch.setattr(deduplication, "read_candidates", candidates)
    monkeypatch.setattr(deduplication, "relationship_counts", links)
    store = _Store(merged={"c"})
    deduplicator = ConceptDeduplicator(["Technology"])
    result = asyncio.run(deduplicator.run(store, "model"))
    assert [call[:2] for call in store.calls] == [("b", "a")]
    assert "model cosine" in store.calls[0][2]
    assert [item["source"] for item in result["skipped"]] == ["c"]
    assert result["survivor_links"] == {"before": 3, "after": 11}
    assert result["groups"][0]["links_after"] == 11


def test_dry_run_writes_nothing(monkeypatch):
    async def candidates(store, model, kinds):
        return [
            _concept("a", [1.0, 0.0], 9, label="GAN"),
            _concept("b", [0.99, 0.141], 1, label="GAN model"),
        ]

    async def links(store, ids):
        return {}

    monkeypatch.setattr(deduplication, "read_candidates", candidates)
    monkeypatch.setattr(deduplication, "relationship_counts", links)
    store = _Store()
    result = asyncio.run(
        ConceptDeduplicator(["Technology"]).run(store, "m", apply=False)
    )
    assert result["merges"] == 1 and store.calls == []


class _JudgeClient:
    model = "fake-judge"

    def __init__(self, answers):
        self.answers = answers  # (first, second) -> (same, confidence)
        self.asked = []

    async def generate(self, schema, system, payload, stage=None):
        verdicts = []
        for pair in payload["pairs"]:
            self.asked.append((pair["first"], pair["second"]))
            same, confidence = self.answers.get(
                (pair["first"], pair["second"]), (False, 0.9)
            )
            verdicts.append(
                {"id": pair["id"], "same": same, "confidence": confidence}
            )
        return schema.model_validate({"verdicts": verdicts})


def test_llm_judge_decides_pairs_below_the_bar():
    rows = [
        _concept("a", [1.0, 0.0], 9, label="gender sensitivity english"),
        _concept("b", [0.92, 0.392], 1, label="gender sensitivity chinese"),
        _concept("c", [0.93, -0.368], 1, label="CoT prompting"),
        _concept("d", [0.999, 0.045], 1, label="gender sensitivity (english)"),
    ]
    client = _JudgeClient(
        {
            ("gender sensitivity english", "gender sensitivity chinese"): (
                False,
                0.95,
            ),
            ("gender sensitivity english", "CoT prompting"): (True, 0.5),
            ("gender sensitivity chinese", "CoT prompting"): (False, 0.9),
        }
    )
    deduplicator = ConceptDeduplicator(
        ["Technology"],
        threshold=0.8,
        pair_judge=deduplication.PairJudge(client, batch_size=2),
        judge_below=0.95,
    )
    deduplicator.load(rows)
    stats = asyncio.run(deduplicator.adjudicate())
    plan = deduplicator.plan()
    merged = {
        m["concept_id"]: m["reason"]
        for g in plan["groups"]
        for m in g["members"]
    }
    # d is above 0.95: rules decide, the judge is not asked.
    assert set(merged) == {"d"}
    assert (
        "gender sensitivity english",
        "gender sensitivity (english)",
    ) not in client.asked
    reasons = {item["concept_id"]: item["reason"] for item in plan["review"]}
    assert reasons["b"].startswith("llm different")
    # "same" with low confidence is not enough.
    assert reasons["c"].startswith("llm same (0.50)")
    assert stats["asked"] == stats["answered"] == 4
