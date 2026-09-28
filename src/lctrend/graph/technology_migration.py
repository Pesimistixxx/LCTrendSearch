"""Offline, additive Technology reprocessing plan. This module has no writer.

Input: {"documents": [{"document": DocumentEnvelope,
"previous_result": ExtractionResult (optional), "answers": [...] (replay)}]}.
Original documents, runs, quotes and decisions are retained in the report.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict

from ..core.models import (
    DocumentEnvelope,
    ExtractionResult,
    json_value,
    stable_id,
)
from ..extraction.resolver import ConceptRegistry
from ..llm.client import JsonLLM, ReplayProvider
from ..llm.context import PipelineSettings
from ..llm.pipeline import process_document


async def technology_dry_run(
    snapshot: Dict[str, Any],
    *,
    replay: bool = False,
    provider=None,
    settings: PipelineSettings | None = None,
) -> Dict[str, Any]:
    """Reprocess only supplied source text; never connect to Neo4j.

    Matching old and new concepts uses document-local evidence overlap, not
    the names that may have caused the original homonym collision.
    """
    registry = ConceptRegistry()
    rows = []
    counts: Counter = Counter()
    for item in snapshot["documents"]:
        document = DocumentEnvelope.model_validate(item["document"])
        previous = (
            ExtractionResult.model_validate(item["previous_result"])
            if item.get("previous_result")
            else None
        )
        if (
            previous
            and previous.document_version_id != document.document_version_id
        ):
            raise ValueError("previous extraction belongs to another document")
        active_provider = (
            ReplayProvider(item["answers"])
            if replay
            else (provider or JsonLLM.from_environment())
        )
        result = await process_document(
            document,
            active_provider,
            registry=registry,
            settings=settings,
        )
        new_mentions = {m.mention_id: m for m in result.mentions}
        new_concepts = {c.concept_id: c for c in result.concepts}
        changes = []
        matched = set()
        for old in previous.concepts if previous else []:
            old_ids = {
                d.mention_id
                for d in previous.resolutions
                if d.concept_id == old.concept_id
            }
            old_spans = [
                m for m in previous.mentions if m.mention_id in old_ids
            ]
            targets = set()
            for decision in result.resolutions:
                mention = new_mentions.get(decision.mention_id)
                if mention is not None and any(
                    m.chunk_id == mention.chunk_id
                    and max(m.start, mention.start) < min(m.end, mention.end)
                    for m in old_spans
                ):
                    targets.add(decision.concept_id)
            targets.discard(None)
            matched |= targets
            proposals = [
                {
                    "concept_id": cid,
                    "kind": new_concepts[cid].kind.value,
                    "name": new_concepts[cid].preferred_label,
                }
                for cid in sorted(targets)
            ]
            action = (
                "needs_context"
                if not targets
                else "split"
                if len(targets) > 1
                else "reclassify"
                if proposals[0]["kind"] != old.kind.value
                else "retain"
                if proposals[0]["concept_id"] == old.concept_id
                else "replace_identity"
            )
            counts[action] += 1
            changes.append(
                {
                    "old_id": old.concept_id,
                    "old_name": old.preferred_label,
                    "old_kind": old.kind.value,
                    "action": action,
                    "proposals": proposals,
                    "assessment_ids": [
                        a.assessment_id for a in result.entity_assessments
                    ],
                }
            )
        for cid in sorted(set(new_concepts) - matched):
            counts["add"] += 1
            changes.append(
                {
                    "action": "add",
                    "proposals": [
                        {
                            "concept_id": cid,
                            "name": new_concepts[cid].preferred_label,
                            "kind": new_concepts[cid].kind.value,
                        }
                    ],
                }
            )
        rows.append(
            {
                "document_version_id": document.document_version_id,
                "run_status": result.run.status,
                "changes": changes,
                "proposed_result": result.model_dump(mode="json"),
            }
        )
    # One legacy node may have accumulated different meanings across sources.
    # Never choose a single replacement just because it appears first.
    identities: Dict[str, Any] = {}
    for row in rows:
        for change in row["changes"]:
            if "old_id" not in change:
                continue
            identity = identities.setdefault(
                change["old_id"],
                {
                    "old_id": change["old_id"],
                    "old_name": change["old_name"],
                    "proposals": {},
                    "document_version_ids": [],
                    "missing_context": False,
                },
            )
            identity["document_version_ids"].append(row["document_version_id"])
            identity["missing_context"] |= change["action"] == "needs_context"
            for proposal in change["proposals"]:
                identity["proposals"][proposal["concept_id"]] = proposal
    for identity in identities.values():
        identity["proposals"] = list(identity["proposals"].values())
        identity["action"] = (
            "needs_context"
            if identity["missing_context"]
            else "split"
            if len(identity["proposals"]) > 1
            else "review_mapping"
        )
    return {
        "contract_version": "technology/1",
        "dry_run": True,
        "graph_writes": 0,
        "evaluation_mode": "replay" if replay else "model",
        "source_snapshot_id": stable_id("snapshot", json_value(snapshot)),
        "summary": dict(counts),
        "original_snapshot": snapshot,
        "documents": rows,
        "identity_plan": list(identities.values()),
        "limitations": [
            "Replay verifies pipeline behavior, not independent model quality."
            if replay
            else "Semantic decisions require human sample review.",
            (
                "An incomplete snapshot cannot establish "
                "graph-wide split/merge coverage."
            ),
            "No deletions or apply operation are generated.",
        ],
    }
