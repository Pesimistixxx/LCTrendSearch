"""Packets of one document run concurrently on a key pool; chunk IDs travel
to the model as short aliases."""

import asyncio

import pytest

from lctrend.core.models import stable_id
from lctrend.llm.client import LLMError, ReplayProvider
from lctrend.llm.context import PipelineSettings, plan_packets
from lctrend.llm.pipeline import _ChunkAliases, process_document
from tests.llm.test_llm_pipeline import (
    FullPacketProvider,
    document,
    extracted,
    full_text,
    reviewed,
    settings,
)

pytestmark = pytest.mark.legacy_technology_entities


class ConcurrentProvider(FullPacketProvider):
    """FullPacketProvider serving several requests at once."""

    def __init__(self, capacity, fail_first_extract=False):
        super().__init__()
        self.max_concurrency = capacity
        self.active = self.peak = 0
        self.fail_first_extract = fail_first_extract

    async def generate(self, schema, system, payload, *, stage="extract"):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.01)
            if self.fail_first_extract and stage == "extract":
                self.fail_first_extract = False
                self.calls.append({"stage": stage})
                raise LLMError("incomplete_response", "output limit")
            return super().generate(schema, system, payload, stage=stage)
        finally:
            self.active -= 1


def run(doc, provider, **overrides):
    shipped = PipelineSettings.from_catalog().model_copy(update=overrides)
    return asyncio.run(process_document(doc, provider, settings=shipped))


def outcome(result):
    return [(a.assertion_id, a.status) for a in result.assertions]


def test_packets_run_at_once_and_keep_the_plan_order():
    doc = full_text()
    sequential = run(doc, FullPacketProvider())
    packets = len(plan_packets(doc, PipelineSettings.from_catalog()).packets)
    assert packets >= 3
    provider = ConcurrentProvider(capacity=8)
    parallel = run(doc, provider)
    # Every packet at once; each packet's review follows its extraction.
    assert provider.peak == packets
    assert outcome(parallel) == outcome(sequential)
    assert parallel.run.status == "succeeded"
    assert parallel.run.metadata["timing"]["packet_workers"] == 8


def test_packet_workers_never_exceed_the_provider_capacity():
    provider = ConcurrentProvider(capacity=2)
    result = run(full_text(), provider, packet_workers=8)
    assert provider.peak == 2
    assert result.run.metadata["timing"]["packet_workers"] == 2
    # A provider without a concurrency limit is served one packet at a time.
    single = run(full_text(), FullPacketProvider())
    assert single.run.metadata["timing"]["packet_workers"] == 1


def test_split_packet_halves_take_its_place_under_concurrency():
    doc = full_text()
    sequential = run(doc, FullPacketProvider())
    provider = ConcurrentProvider(capacity=3, fail_first_extract=True)
    parallel = run(doc, provider)
    assert "incomplete_response" in {
        issue.get("code") for issue in parallel.run.metadata["issues"]
    }
    assert sorted(outcome(parallel)) == sorted(outcome(sequential))
    coverage = parallel.run.metadata["coverage"]
    assert coverage["unprocessed_chunk_ids"] == []
    assert coverage["failed_packet_ids"] == []


def hex_ids(doc):
    for chunk in doc.chunks:
        chunk.chunk_id = stable_id("chunk", chunk.text)
    return doc


class Recording(ReplayProvider):
    def __init__(self, answers):
        super().__init__(answers)
        self.payloads = []

    async def generate(self, schema, system, payload, *, stage="extract"):
        self.payloads.append((stage, payload))
        return await super().generate(schema, system, payload, stage=stage)


def test_real_chunk_ids_reach_the_model_as_aliases_and_come_back():
    doc = hex_ids(document())
    real = doc.chunks[0].chunk_id
    answer = extracted(document())  # cites the alias "c1"
    provider = Recording([answer, reviewed()])
    result = asyncio.run(process_document(doc, provider, settings=settings()))
    stage, payload = provider.payloads[0]
    assert stage == "extract"
    assert [chunk["chunk_id"] for chunk in payload["chunks"]] == ["c1"]
    assert real not in str(payload)
    assert result.assertions[0].evidence[0].chunk_id == real
    assert result.assertions[0].status == "accepted"
    assert {m.chunk_id for m in result.mentions} == {real}


def test_aliases_leave_fixture_ids_and_foreign_ids_alone():
    assert _ChunkAliases(document()).forward == {}
    doc = hex_ids(document(["One.", "Two."]))
    aliases = _ChunkAliases(doc)
    foreign = stable_id("chunk", "another document")
    first, second = (chunk.chunk_id for chunk in doc.chunks)
    encoded = aliases.encode(
        {
            first: [second, f'{{"chunk_id": "{first}"}}', foreign],
            "text": "no ids here",
        }
    )
    assert encoded == {
        "c1": ["c2", '{"chunk_id": "c1"}', foreign],
        "text": "no ids here",
    }


def test_a_call_cut_by_a_stopping_server_is_not_a_bad_answer():
    # A server stop closes the loop's executor under a request in flight;
    # that must not be recorded as the model answering garbage.
    from lctrend.llm.pipeline import _Budget

    class Stopped:
        async def generate(self, schema, system, payload, *, stage="extract"):
            raise RuntimeError("cannot schedule new futures after shutdown")

    trace = []
    budget = _Budget(Stopped(), PipelineSettings(), trace)
    try:
        asyncio.run(budget.call(object, "s", {}, "extract"))
    except LLMError as exc:
        assert exc.code == "interrupted"
    else:
        raise AssertionError("the call must fail")
    assert trace[-1]["code"] == "interrupted"
