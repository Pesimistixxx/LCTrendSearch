"""Weak-signal cards: clusters, counted stage and trend, the model's words
checked against the dossier, and the analyst table file.
"""

import asyncio
import csv
import json
from datetime import date

import pytest

from lctrend.core.catalog_validation import validate_catalogs
from lctrend.core.config import load_catalog
from lctrend.graph.temporal import TemporalCorpus
from lctrend.llm.client import LLMError
from lctrend.ranking.export import COLUMNS, write_report
from lctrend.ranking.signals import (
    SignalAnswer,
    build_cards,
    cluster_candidates,
    signal_config,
)
from tests.ranking.test_scoring import corpus_data

T = date(2026, 9, 1)


def data():
    base = corpus_data(
        {
            "mcp": (
                "MCP server security scanner",
                [
                    ("2025-06-01", "a"),
                    ("2026-01-10", "b"),
                    ("2026-03-01", "c"),
                    ("2026-05-01", "d"),
                    ("2026-07-01", "e"),
                ],
            ),
            "mcp2": (
                "MCP tool poisoning detection",
                [("2026-02-01", "f"), ("2026-06-01", "g")],
            ),
            "flash": (
                "Speculative decoding with flash offloading",
                [("2025-03-01", "h"), ("2026-04-01", "i")],
            ),
        }
    )
    base["technologies"][0]["definition"] = "сканер конфигураций MCP"
    # Document parties are ids; only named ones reach the table.
    base["versions"][0]["companies"] = ["org:acme", "org:ghost"]
    base["organizations"] = [
        {"organization_id": "org:acme", "name": "Acme Security"}
    ]
    for row, vector in zip(
        base["technologies"], ([1.0, 0.0], [0.95, 0.05], [0.0, 1.0])
    ):
        row["embedding"] = vector
        row["embedding_model"] = "test"
    base["relations"] = [
        {
            "technology_id": "mcp",
            "relation": "DEVELOPED_BY",
            "target_id": "org:invariant",
            "target_label": "Invariant Labs",
            "target_kind": "Company",
            "version_id": "mcp-1-v1",
            "observed_at": "2026-01-10",
        }
    ]
    base["assertions"] = [
        {
            "technology_id": "mcp",
            "assertion_id": "a1",
            "predicate": "reports_market_event",
            "status": "accepted",
            "verification_status": "supported",
            "polarity": "affirmed",
            "modality": "observed",
            "qualifiers_json": json.dumps(
                {"event": "funding_round", "round": "seed"}
            ),
            "role_labels": [
                ["SUBJECT", "MCP server security scanner", "Technology"],
                ["ORGANIZATION", "Cyata", "Company"],
            ],
            "version_id": "mcp-3-v1",
            "observed_at": "2026-05-01",
            "quote": "Cyata emerged from stealth with $8.5 million seed",
        },
        {
            "technology_id": "mcp",
            "assertion_id": "a2",
            "predicate": "reports_market_event",
            "status": "rejected",
            "qualifiers_json": json.dumps({"event": "acquisition"}),
            "role_labels": [["ORGANIZATION", "Rumored Corp", "Company"]],
            "version_id": "mcp-4-v1",
            "observed_at": "2026-07-01",
            "quote": "rumor",
        },
    ]
    base["maturity"] = [
        {
            "technology_id": "mcp",
            "version_id": "mcp-0-v1",
            "observed_at": "2025-06-01",
            "stage": "prototype",
            "stage_rank": 3,
        },
        {
            "technology_id": "mcp",
            "version_id": "mcp-4-v1",
            "observed_at": "2026-07-01",
            "stage": "commercial_deployment",
            "stage_rank": 5,
        },
    ]
    return base


def config():
    value = signal_config()
    value["include_novelty"] = False
    return value


class Model:
    def __init__(self, answers):
        self.answers = answers
        self.payloads = []

    async def generate(self, schema, system, payload, *, stage="extract"):
        assert schema is SignalAnswer and stage == "review"
        assert "слабого технологического сигнала" in system
        self.payloads.append(payload)
        [answer] = [
            self.answers[item["label"]]
            for item in payload["dossier"]["technologies"]
            if item["label"] in self.answers
        ]
        if isinstance(answer, Exception):
            raise answer
        return SignalAnswer.model_validate(answer)


def test_shipped_signal_config_is_valid():
    validate_catalogs()
    settings = load_catalog("ranking")["signals"]
    assert settings["stages"]["points"] == {
        "concept": 1,
        "prototype": 2,
        "pilot": 3,
        "early_adoption": 4,
    }
    assert settings["trend"]["points"]["fast"] == 3


def test_clusters_follow_ranking_order_and_cosine():
    rows = [{"technology_id": key} for key in ("a", "b", "c", "d")]
    vectors = {"a": [1.0, 0.0], "b": [0.99, 0.1], "c": [0.0, 1.0]}
    clusters = cluster_candidates(rows, vectors, 0.9, 5)
    assert [
        [item["technology_id"] for item in group] for group in clusters
    ] == [
        ["a", "b"],
        ["c"],
        ["d"],
    ]
    single = cluster_candidates(rows, vectors, 0.9, 1)
    assert len(single) == 4


def test_card_counts_stage_trend_score_and_checks_companies():
    model = Model(
        {
            "MCP server security scanner": {
                "title": "Сканеры безопасности MCP-серверов",
                "area": "Защита ИИ",
                "companies": [
                    "Invariant Labs (mcp-scan)",
                    "Cyata",
                    "OpenAI",
                ],
                "why_weak": "Два seed-раунда и ни одной категории закупок.",
                "stage": "prototype",
                "stage_note": "у лидера",
                "trend_note": "от одной работы в 2025 до шести в 2026",
            },
            "Speculative decoding with flash offloading": {
                "is_signal": False,
                "reject_reason": "узкая техника без участников",
            },
        }
    )
    result = asyncio.run(
        build_cards(TemporalCorpus(data()), T, model, config())
    )
    assert result["stats"]["clusters"] == 2
    assert result["stats"]["rejected_by_model"] == 1
    [card] = result["cards"]
    assert sorted(card["technology_ids"]) == ["mcp", "mcp2"]
    assert card["title"] == "Сканеры безопасности MCP-серверов"
    # Only names found in the dossier; a rejected claim adds nobody.
    assert card["companies"] == ["Invariant Labs (mcp-scan)", "Cyata"]
    # Evidence wins over the model's stage and shows the transition.
    assert card["stage"] == "early_adoption"
    assert card["stage_label"] == (
        "Прототип/PoC → Раннее внедрение (у лидера)"
    )
    assert card["llm_stage"] == "prototype"
    # 6 documents in the last 12 months against 1 before: fast.
    assert card["trend"] == "fast"
    assert card["trend_label"].startswith("Растёт быстро — от одной")
    assert card["score"] == 4 + 3
    assert card["sources"][0]["url"].startswith("https://example.org/")
    dossier = model.payloads[0]["dossier"]
    assert {item.get("definition") for item in dossier["technologies"]} == {
        "сканер конфигураций MCP",
        None,
    }
    assert [item["event"] for item in dossier["events"]] == ["funding_round"]
    assert "Защита ИИ" in model.payloads[0]["areas"]


def test_card_without_model_is_built_from_the_dossier():
    model = Model(
        {
            "MCP server security scanner": LLMError("invalid_schema", "bad"),
            "Speculative decoding with flash offloading": {
                "title": "Спекулятивный декодинг с выгрузкой во флеш",
                "area": "Неизвестная область",
                "stage": "concept",
            },
        }
    )
    result = asyncio.run(
        build_cards(TemporalCorpus(data()), T, model, config())
    )
    cards = {
        key: card for card in result["cards"] for key in card["technology_ids"]
    }
    fallback = cards["mcp"]
    assert fallback["llm"] is False
    assert fallback["title"] in fallback["technologies"]
    # Organizations with roles first, then authors of materials.
    assert set(fallback["companies"][:2]) == {"Invariant Labs", "Cyata"}
    assert fallback["companies"][2:] == ["Acme Security"]
    assert "Документов: 7" in fallback["why_weak"]
    flash = cards["flash"]
    # No maturity evidence: the model's stage counts; unknown area -> last.
    assert flash["stage_label"] == "Концепция/Исследование"
    assert flash["area"] == "Другое"
    assert flash["trend"] == "stable"
    assert flash["score"] == 1 + 1
    assert result["stats"]["without_model"] == 1
    assert [card["number"] for card in result["cards"]] == [1, 2]


def test_report_files_follow_the_analyst_columns(tmp_path):
    result = asyncio.run(
        build_cards(TemporalCorpus(data()), T, None, config())
    )
    csv_path = write_report(result, tmp_path / "signals.csv")
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        table = list(csv.reader(handle))
    assert table[0] == COLUMNS
    assert table[1][0] == "1"
    assert table[1][8].startswith("[Paper ")
    saved = json.loads(
        write_report(result, tmp_path / "signals.json").read_text("utf-8")
    )
    assert saved["cards"][0]["number"] == 1
    with pytest.raises(ValueError):
        write_report(result, tmp_path / "signals.txt")


def test_xlsx_report(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    result = asyncio.run(
        build_cards(TemporalCorpus(data()), T, None, config())
    )
    path = write_report(result, tmp_path / "signals.xlsx")
    sheet = openpyxl.load_workbook(path).active
    assert sheet["A1"].value.startswith("2 слабых технологических сигналов")
    assert "сентябрь 2026" in sheet["A1"].value
    assert [cell.value for cell in sheet[2]] == COLUMNS
    assert sheet.max_row == 4
