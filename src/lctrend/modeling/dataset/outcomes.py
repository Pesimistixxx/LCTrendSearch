"""Labels from what happened to a technology after the snapshot.

The row at T holds only the past; the label compares it with the same
technology's row at the end of the horizon (the latest snapshot at or
before T + horizon). Features and label therefore never share a window.

- a snapshot is labelled only if the technology is active at T (enough
  documents in the last year): the question "is this an early signal?"
  is meaningless for a technology silent at T, and its copies of the
  same state would flood the sample with easy negatives;
- 1, a signal: within the horizon it gained new documents and new
  independent groups (other teams took it up);
- 0, noise: within the horizon it gained no documents at all;
- 0, mainstream: at T it was already at the top of its contemporaries
  (weight reduced, see ``sample_weight_factor``);
- anything in between (one new document, new documents from the same
  teams) stays unlabelled: fewer examples beat disputed ones;
- a horizon ending after ``data_end`` stays unlabelled (censored).

Silence in our corpus is weaker evidence than silence in the world: a
technology we saw once may live on in sources we never crawled. The
labels carry ``label_source = auto_future_outcome`` so that they are never
mistaken for expert labels.
"""

from __future__ import annotations

import bisect
from calendar import monthrange
from collections import defaultdict
from datetime import date
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .labels import AUTO_LABEL_SOURCE, TARGETS

DEFAULT_RULE = {
    "active_min_documents_last_year": 1,
    "positive_min_new_documents": 2,
    "positive_min_new_groups": 1,
    "negative_max_new_documents": 0,
    "mainstream_min_pct": 0.95,
    "mainstream_min_documents_last_year": 10,
    "mainstream_weight": 0.5,
}


def _int(value: Any) -> int:
    return int(float(value)) if value not in (None, "") else 0


def _float(value: Any) -> Optional[float]:
    return float(value) if value not in (None, "") else None


def add_months(when: date, months: int) -> date:
    index = when.month - 1 + months
    year, month = when.year + index // 12, index % 12 + 1
    return date(year, month, min(when.day, monthrange(year, month)[1]))


def outcome(
    row: Mapping[str, Any],
    future: Mapping[str, Any],
    rule: Mapping[str, Any],
) -> Dict[str, Any]:
    """Label of one snapshot from its row and the horizon-end row."""
    new_documents = _int(future["document_count"]) - _int(
        row["document_count"]
    )
    new_groups = _int(future["independence_group_diversity"]) - _int(
        row["independence_group_diversity"]
    )
    pct = _float(row.get("documents_last_year_snapshot_pct"))
    if (
        pct is not None
        and pct >= rule["mainstream_min_pct"]
        and _int(row["documents_last_year"])
        >= rule["mainstream_min_documents_last_year"]
    ):
        label, reason = 0, "mainstream"
    elif (
        new_documents >= rule["positive_min_new_documents"]
        and new_groups >= rule["positive_min_new_groups"]
    ):
        label, reason = 1, "adopted"
    elif new_documents <= rule["negative_max_new_documents"]:
        label, reason = 0, "silent"
    else:
        label, reason = None, "ambiguous"
    return {
        "label": label,
        "reason": reason,
        "new_documents": new_documents,
        "new_groups": new_groups,
    }


def label_rows(
    rows: Iterable[Dict[str, Any]],
    data_end: date,
    rule: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Rows active at T with signal_12m/36m, reasons and weights added.

    Inactive snapshots are dropped; censored or ambiguous horizons keep an
    empty label.
    """
    rule = {**DEFAULT_RULE, **(rule or {})}
    history = defaultdict(list)
    for row in rows:
        history[row["technology_id"]].append(row)
    result = []
    for technology_rows in history.values():
        technology_rows.sort(key=lambda item: item["snapshot_date"])
        dates = [item["snapshot_date"] for item in technology_rows]
        for row in technology_rows:
            if (
                _int(row["documents_last_year"])
                < rule["active_min_documents_last_year"]
            ):
                continue
            labelled = dict(row)
            when = date.fromisoformat(row["snapshot_date"])
            weight = 1.0
            for target, months in TARGETS.items():
                end = add_months(when, months)
                labelled[f"horizon_{months}m_end"] = end.isoformat()
                if end > data_end:
                    labelled[target] = ""
                    labelled[f"outcome_{months}m"] = "censored"
                    continue
                future = technology_rows[
                    bisect.bisect_right(dates, end.isoformat()) - 1
                ]
                found = outcome(row, future, rule)
                labelled[target] = (
                    "" if found["label"] is None else str(found["label"])
                )
                labelled[f"outcome_{months}m"] = found["reason"]
                labelled[f"new_documents_{months}m"] = found["new_documents"]
                labelled[f"new_groups_{months}m"] = found["new_groups"]
                if found["reason"] == "mainstream":
                    weight = rule["mainstream_weight"]
            labelled["sample_weight_factor"] = weight
            labelled["label_source"] = AUTO_LABEL_SOURCE
            result.append(labelled)
    return result
