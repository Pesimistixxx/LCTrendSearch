import asyncio
import csv
import json

from lctrend.modeling.dataset.llm_outcomes import (
    Labeller,
    TrajectoryAssessment,
    label_packets,
    packets,
    write_labels,
)


def _history():
    rows = []
    for technology, dates in (
        ("t", ("2019-01-01", "2019-07-01", "2020-01-01", "2021-01-01")),
        ("u", ("2020-01-01",)),
    ):
        for index, when in enumerate(dates):
            rows.append(
                {
                    "technology_id": technology,
                    "technology": technology.upper(),
                    "snapshot_date": when,
                    "first_seen_date": dates[0],
                    "document_count": str(index + 1),
                    "documents_last_year": "1",
                    "independence_group_diversity": "1",
                }
            )
    return rows


def _review():
    return [
        {
            "technology_id": "t",
            "recent_documents": json.dumps(
                [
                    {
                        "document_id": "d1",
                        "title": "A paper on T",
                        "date": "2019-02-01",
                        "source_family": "scholarly",
                    }
                ]
            ),
        }
    ]


def test_packet_is_a_yearly_timeline_with_dated_titles():
    items = packets(_history(), _review())
    timeline = items["t"]["history"]
    # 2019 appears once, from its last snapshot (July).
    assert [entry["year"] for entry in timeline] == [2019, 2020, 2021]
    assert timeline[0]["documents"] == 2
    assert items["t"]["documents"] == [
        {"date": "2019-02-01", "title": "A paper on T", "source": "scholarly"}
    ]


class _Client:
    """Answers every year except 2020, to test gap filling."""

    def __init__(self, fail=()):
        self.fail = set(fail)

    async def generate(self, schema, system, packet, stage=None):
        if packet["technology"] in self.fail:
            raise RuntimeError("rate limited")
        years = [
            {"year": entry["year"], "score": 0.2, "hype": 0.5, "maturity": 0.3}
            for entry in packet["history"]
            if entry["year"] != 2020 or len(packet["history"]) == 1
        ]
        return schema.model_validate(
            {
                "is_technology": True,
                "verdict": "niche",
                "rationale": "test",
                "years": years,
            }
        )


def test_labels_every_snapshot_and_resumes_after_errors(tmp_path):
    history = _history()
    items = packets(history)
    log = tmp_path / "labels.jsonl"
    first = asyncio.run(
        label_packets(items, [Labeller("llm", _Client({"U"}), "m")], log)
    )
    assert first["ok"] == 1 and first["error"] == 1
    # The rerun takes only what failed.
    second = asyncio.run(
        label_packets(items, [Labeller("llm", _Client(), "m")], log)
    )
    assert second["pending"] == 1 and second["ok"] == 1
    output = tmp_path / "labels.csv"
    summary = write_labels(history, items, log, output)
    assert summary["rows"] == len(history)
    with output.open(encoding="utf-8") as stream:
        rows = {
            (row["technology_id"], row["snapshot_date"]): row
            for row in csv.DictReader(stream)
        }
    # A year the model skipped takes the previous year's values.
    assert rows[("t", "2020-01-01")]["llm_score"] == "0.2"
    assert rows[("t", "2019-07-01")]["llm_verdict"] == "niche"
    assert rows[("u", "2020-01-01")]["label_source"] == "llm_trajectory"


def test_scores_are_bounded():
    try:
        TrajectoryAssessment.model_validate(
            {
                "is_technology": True,
                "verdict": "niche",
                "rationale": "",
                "years": [
                    {"year": 2020, "score": 1.5, "hype": 0, "maturity": 0}
                ],
            }
        )
    except ValueError:
        return
    raise AssertionError("score above 1 was accepted")


def test_packets_from_the_graph_count_documents_by_year():
    from lctrend.graph.temporal import TemporalCorpus
    from lctrend.modeling.dataset.llm_outcomes import packets_from_corpus

    def version(year, companies):
        return {
            "document_id": f"d{year}",
            "version_id": f"v{year}",
            "document_type": "article",
            "source_family": "scholarly",
            "source_id": "openalex",
            "document_published_at": f"{year}-03-01",
            "version_published_at": f"{year}-03-01",
            "retrieved_at": f"{year}-03-01",
            "extracted": True,
            "coverage": "full_text",
            "companies": companies,
            "title": f"Paper {year}",
        }

    corpus = TemporalCorpus(
        {
            "versions": [
                version(2020, ["Acme"]),
                version(2022, ["Acme"]),
                version(2022, ["Beta"])
                | {"document_id": "e", "version_id": "w"},
            ],
            "technologies": [{"technology_id": "t", "technology": "T"}],
            "mentions": [
                {
                    "technology_id": "t",
                    "version_id": key,
                    "observed_at": when,
                    "mentions": 1,
                    "accepted": 1,
                }
                for key, when in (
                    ("v2020", "2020-03-01"),
                    ("v2022", "2022-03-01"),
                    ("w", "2022-03-01"),
                )
            ],
        }
    )
    packet = packets_from_corpus(corpus)["t"]
    years = {entry["year"]: entry for entry in packet["history"]}
    assert sorted(years) == [2020, 2021, 2022]
    assert years[2021]["documents"] == 1
    assert years[2021]["new_documents_last_year"] == 0
    assert years[2022]["documents"] == 3
    # Acme's two papers are one group; Beta is another.
    assert years[2022]["independent_groups"] == 2
    assert years[2022]["companies"] == 2
    assert [item["title"] for item in packet["documents"]][0] == "Paper 2020"


def test_llm_scores_become_the_binary_target_and_noise_is_unlabelled():
    from lctrend.modeling.dataset.llm_outcomes import llm_label_rows

    def snapshot(technology, when, active=1):
        return {
            "technology_id": technology,
            "snapshot_date": when,
            "documents_last_year": str(active),
        }

    def year(technology, value, score, is_technology=True):
        return {
            "technology_id": technology,
            "year": str(value),
            "llm_score": str(score),
            "llm_hype": "0.2",
            "llm_maturity": "0.3",
            "llm_verdict": "niche",
            "llm_is_technology": str(is_technology),
            "llm_model": "m",
        }

    history = [
        snapshot("weak", "2020-04-01"),
        snapshot("mature", "2020-04-01"),
        snapshot("unsure", "2020-04-01"),
        snapshot("noise", "2020-04-01"),
        snapshot("silent", "2020-04-01", active=0),
    ]
    by_year = [
        year("weak", 2020, 0.1),
        year("mature", 2020, 0.9),
        year("unsure", 2020, 0.5),
        year("noise", 2020, 0.1, is_technology=False),
        year("silent", 2020, 0.1),
    ]
    rows = {
        row["technology_id"]: row for row in llm_label_rows(history, by_year)
    }
    assert rows["weak"]["signal_llm"] == "1"
    assert rows["mature"]["signal_llm"] == "0"
    assert rows["unsure"]["signal_llm"] == ""
    assert rows["noise"]["llm_bucket"] == "not_technology"
    assert rows["noise"]["signal_llm"] == ""
    assert "silent" not in rows
    assert rows["weak"]["label_source"] == "llm_trajectory"
