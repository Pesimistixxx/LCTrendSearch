"""New claims reconciled with what the graph already holds.

Extraction is local: every relation comes from one document's quote, and
nothing compared a new claim with the old ones. After a document is
published, this step pulls the accepted claims about the same concepts
from the graph and links what can be linked:

- ``claim_key`` (the slot): predicate + concepts in their roles +
  normalized qualifiers (conditions, stage), without polarity. Claims of
  one slot say the same thing about the same things under the same
  conditions; different conditions are different slots, so they never
  conflict. Measurements and money (llm_schema ``measurement_predicates``)
  are left out: their values need their own comparison.
- ``CORROBORATES``: a claim of another work (fc's Work layer: an arXiv
  preprint and its journal version are one work) says the same, with the
  same polarity and modality class; each claim points to the earliest
  one of every other work, so a popular slot stays linear, not quadratic.
- ``CONTRADICTS``: a factual claim of another work in the same slot with
  the opposite polarity (a plan does not contradict a fact).
- ``SHARES_CONTEXT_WITH``: two technologies of one kind family stated in
  at least ``min_shared`` identical contexts (same predicate with the
  same partners: developed by the same company, solving the same task).
  A review candidate for a duplicate or a close relative; contexts whose
  partners are only countries or domains do not count. Existing
  SIMILAR_TO and POSSIBLY_SAME_AS edges get ``shared_claims`` too.

Every edge is computed (``method``, ``computed_at``) and dated by the
later of its claims (``observed_at``), so a snapshot can use it without
seeing the future. None of them is a reported relation: graph.novelty
leaves them out of structural metrics. The claim key is also a property
of the assertion; snapshot features (features.claim_slot_features) group
by it within the snapshot itself.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..core.config import cypher_identifier, load_catalog
from ..core.models import json_value, stable_id
from ..extraction.lexical import kind_family

logger = logging.getLogger(__name__)

METHOD = "claim_slot"
COMPUTED = ("CORROBORATES", "CONTRADICTS", "SHARES_CONTEXT_WITH")
FACTUAL = frozenset({"observed", "reported", "unknown"})
OPPOSITE = {"affirmed": "negated", "negated": "affirmed"}
_WRITE_BATCH = 1000


def settings() -> Dict[str, Any]:
    return dict(
        load_catalog("pipeline").get("linking", {}).get("reconcile", {})
    )


@lru_cache(maxsize=1)
def _skipped_predicates() -> frozenset:
    return frozenset(load_catalog("llm_schema")["measurement_predicates"])


def _normalized(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    if isinstance(value, Mapping):
        return {key: _normalized(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    return value


def claim_key(
    predicate: str,
    roles: Mapping[str, str],
    qualifiers: Optional[Mapping[str, Any]] = None,
    ignored_qualifiers: Iterable[str] = ("trl",),
) -> Optional[str]:
    """The slot of a claim, or None for a measurement or money claim."""
    if not predicate or not roles or predicate in _skipped_predicates():
        return None
    ignored = set(ignored_qualifiers)
    conditions = {
        key: _normalized(value)
        for key, value in (qualifiers or {}).items()
        if key not in ignored and value not in (None, "", [], {})
    }
    return stable_id(
        "claim-slot",
        predicate,
        json_value(sorted(roles.items())),
        json_value(conditions),
    )


def slot_of_row(row: Mapping[str, Any]) -> Optional[str]:
    """The slot of a claim read as [relation, concept_id] role pairs and
    qualifiers JSON (the temporal assertions rows): computed at read time,
    so it follows concepts moved by a merge."""
    names = _role_names()
    roles = {
        names[relation]: concept
        for relation, concept in row.get("roles") or []
        if relation in names
    }
    try:
        qualifiers = json.loads(row.get("qualifiers_json") or "{}")
    except ValueError:
        qualifiers = {}
    return claim_key(
        row.get("predicate"),
        roles,
        qualifiers if isinstance(qualifiers, dict) else {},
        settings().get("ignored_qualifiers", ["trl"]),
    )


def _modality_class(claim: Mapping[str, Any]) -> str:
    return "factual" if claim.get("modality") in FACTUAL else "speculative"


def _order(claim: Mapping[str, Any]) -> tuple:
    """Earliest first; an undated claim is never the earliest."""
    observed = claim.get("observed_at")
    return (observed is None, str(observed or ""), claim["assertion_id"])


def _later(*dates: Any) -> Optional[str]:
    known = [str(item) for item in dates if item]
    return max(known) if len(known) == len(dates) else None


def claim_links(
    claims: Sequence[Mapping[str, Any]],
    new_ids: Optional[set] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """CORROBORATES and CONTRADICTS rows between claims of other works.

    A work speaks once per slot and polarity: by its earliest claim. Each
    such claim links to the earliest agreeing (and, if factual, the
    earliest opposing) claim of another work, so a popular slot stays
    linear. The later claim points to the earlier one. ``new_ids`` limits
    the links to those starting from these claims (an incremental run
    after one document); None links everything.
    """
    slots: Dict[str, List[Mapping]] = {}
    for claim in claims:
        if claim.get("claim_key"):
            slots.setdefault(claim["claim_key"], []).append(claim)
    links: Dict[str, Dict[tuple, Dict[str, Any]]] = {
        "CORROBORATES": {},
        "CONTRADICTS": {},
    }
    for key, members in slots.items():
        speakers: Dict[tuple, Mapping] = {}
        for claim in sorted(members, key=_order):
            speakers.setdefault(
                (
                    claim["work_id"],
                    claim.get("polarity"),
                    _modality_class(claim),
                ),
                claim,
            )
        ordered = sorted(speakers.values(), key=_order)
        for claim in ordered:
            if new_ids is not None and claim["assertion_id"] not in new_ids:
                continue
            factual = _modality_class(claim) == "factual"
            agreeing = next(
                (
                    other
                    for other in ordered
                    if other["work_id"] != claim["work_id"]
                    and other.get("polarity") == claim.get("polarity")
                    and _modality_class(other) == _modality_class(claim)
                ),
                None,
            )
            opposing = next(
                (
                    other
                    for other in ordered
                    if factual
                    and other["work_id"] != claim["work_id"]
                    and other.get("polarity")
                    == OPPOSITE.get(claim.get("polarity"))
                    and _modality_class(other) == "factual"
                ),
                None,
            )
            for relation, target in (
                ("CORROBORATES", agreeing),
                ("CONTRADICTS", opposing),
            ):
                if target is None:
                    continue
                later, earlier = sorted((claim, target), key=_order)[::-1]
                links[relation][
                    (later["assertion_id"], earlier["assertion_id"])
                ] = {
                    "source": later["assertion_id"],
                    "target": earlier["assertion_id"],
                    "claim_key": key,
                    "observed_at": _later(
                        claim.get("observed_at"), target.get("observed_at")
                    ),
                }
    return {
        relation: [rows[pair] for pair in sorted(rows)]
        for relation, rows in links.items()
    }


def _signatures(
    claims: Sequence[Mapping[str, Any]], excluded_kinds: set
) -> Dict[str, Dict[tuple, Optional[str]]]:
    """Subject technology -> its contexts (predicate + partners) with the
    earliest date each was stated."""
    found: Dict[str, Dict[tuple, Optional[str]]] = {}
    for claim in claims:
        subject = claim["roles"].get("subject")
        if subject is None or claim.get("polarity") != "affirmed":
            continue
        if kind_family(claim["kinds"].get(subject)) != "technology":
            continue
        partners = tuple(
            sorted(
                (role, concept)
                for role, concept in claim["roles"].items()
                if role != "subject"
            )
        )
        if not partners or all(
            claim["kinds"].get(concept) in excluded_kinds
            for _, concept in partners
        ):
            continue
        context = (claim["predicate"], partners)
        dates = found.setdefault(subject, {})
        observed = claim.get("observed_at")
        if context not in dates or (
            observed and (dates[context] is None or observed < dates[context])
        ):
            dates[context] = observed
    return found


def shared_contexts(
    claims: Sequence[Mapping[str, Any]],
    kinds: Mapping[str, str],
    min_shared: int = 1,
    excluded_kinds: Iterable[str] = ("Country", "Domain"),
    subjects: Optional[set] = None,
    max_members: int = 200,
) -> List[Dict[str, Any]]:
    """Technology pairs of one family stated in the same contexts.

    A context shared by more than ``max_members`` technologies (developed
    by one giant, solving "classification") says nothing about any pair
    and would make the pairing quadratic; it is skipped. With ``subjects``
    (an incremental run) only pairs with these technologies are formed,
    linear in the members of a context.
    """
    signatures = _signatures(claims, set(excluded_kinds))
    by_context: Dict[tuple, List[str]] = {}
    for subject, contexts in signatures.items():
        for context in contexts:
            by_context.setdefault(context, []).append(subject)
    pairs: Dict[tuple, List[tuple]] = {}
    for context, members in by_context.items():
        members = sorted(set(members))
        if len(members) > max_members:
            continue
        anchors = (
            members
            if subjects is None
            else [item for item in members if item in subjects]
        )
        for left in anchors:
            for right in members:
                if left == right or kind_family(
                    kinds.get(left)
                ) != kind_family(kinds.get(right)):
                    continue
                found = pairs.setdefault(
                    (min(left, right), max(left, right)), []
                )
                if context not in found:
                    found.append(context)
    rows = []
    for (left, right), contexts in sorted(pairs.items()):
        if len(contexts) < min_shared:
            continue
        rows.append(
            {
                "left": left,
                "right": right,
                "left_kind": kinds[left],
                "right_kind": kinds[right],
                "shared_claims": len(contexts),
                "examples": [
                    f"{predicate}: "
                    + ", ".join(f"{role}={concept}" for role, concept in parts)
                    for predicate, parts in sorted(contexts)[:3]
                ],
                # The pair is linked once both stated a shared context.
                "observed_at": min(
                    (
                        day
                        for context in contexts
                        if (
                            day := _later(
                                signatures[left][context],
                                signatures[right][context],
                            )
                        )
                    ),
                    default=None,
                ),
            }
        )
    return rows


# ------------------------------------------------------------------ graph


_CLAIMS = """
MATCH (d:Document)-[:HAS_VERSION]->(v:DocumentVersion)
      -[:HAS_ASSERTION]->(a:Assertion)
WHERE a.status = 'accepted' {where}
MATCH (a)-[r]->(c)
WHERE type(r) IN $role_types AND c.concept_id IS NOT NULL
WITH d, v, a, collect([type(r), c.concept_id, c.kind]) AS roles
RETURN a.assertion_id AS assertion_id, a.predicate AS predicate,
       a.polarity AS polarity, a.modality AS modality,
       a.qualifiers_json AS qualifiers_json, a.observed_at AS observed_at,
       v.document_version_id AS version_id,
       coalesce(d.work_id, d.document_id) AS work_id, roles
"""


def _role_names() -> Dict[str, str]:
    return {
        relation: role
        for role, relation in load_catalog("graph")["assertion_roles"].items()
    }


def _claim(record: Mapping[str, Any], names: Mapping[str, str]) -> dict:
    roles, kinds = {}, {}
    for relation, concept, kind in record["roles"]:
        roles[names[relation]] = concept
        kinds[concept] = kind
    try:
        qualifiers = json.loads(record.get("qualifiers_json") or "{}")
    except ValueError:
        qualifiers = {}
    return {
        "assertion_id": record["assertion_id"],
        "predicate": record["predicate"],
        "polarity": record.get("polarity"),
        "modality": record.get("modality"),
        "observed_at": record.get("observed_at"),
        "version_id": record["version_id"],
        "work_id": record["work_id"],
        "roles": roles,
        "kinds": kinds,
        "qualifiers": qualifiers if isinstance(qualifiers, dict) else {},
    }


async def read_claims(
    store: Any,
    version_ids: Optional[List[str]] = None,
    concept_ids: Optional[Dict[str, List[str]]] = None,
) -> List[dict]:
    """Accepted claims: all, of these versions, or touching these concepts
    (``concept_ids``: kind -> ids)."""
    from ..graph.store import _records

    names = _role_names()
    where = ""
    if version_ids is not None:
        where = "AND v.document_version_id IN $version_ids"
    query = _CLAIMS.format(where=where)
    if concept_ids is not None:
        # Start from labeled concept nodes (their concept_id index), not
        # from every assertion.
        starts = " UNION ".join(
            f"MATCH (x:{cypher_identifier(kind)}) "
            f"WHERE x.concept_id IN $concepts['{kind}'] RETURN x"
            for kind in sorted(concept_ids)
        )
        query = (
            f"CALL {{ {starts} }} "
            "MATCH (x)<-[]-(a:Assertion) WITH DISTINCT a "
            + _CLAIMS.format(where="")
        )
    async with store._driver.session(database=store._database) as session:
        records = await _records(
            session,
            query,
            role_types=list(names),
            version_ids=version_ids or [],
            concepts=concept_ids or {},
        )
    claims = [_claim(record, names) for record in records]
    ignored = settings().get("ignored_qualifiers", ["trl"])
    for claim in claims:
        claim["claim_key"] = claim_key(
            claim["predicate"], claim["roles"], claim["qualifiers"], ignored
        )
    return claims


async def _write(
    store: Any,
    claims: List[dict],
    links: Dict[str, List[dict]],
    contexts: List[dict],
    replace: bool,
) -> None:
    from ..graph.store import _run

    computed_at = datetime.now(timezone.utc).isoformat()
    keys = [
        {"assertion_id": c["assertion_id"], "claim_key": c["claim_key"]}
        for c in claims
    ]
    groups: Dict[tuple, List[dict]] = {}
    for row in contexts:
        groups.setdefault((row["left_kind"], row["right_kind"]), []).append(
            row
        )
    min_shared = int(settings().get("min_shared", 2))
    async with store._driver.session(database=store._database) as session:

        async def write(tx):
            if replace:
                for relation in COMPUTED:
                    await _run(
                        tx,
                        f"MATCH ()-[r:{relation} {{method: $method}}]->() "
                        "DELETE r",
                        method=METHOD,
                    )
            for start in range(0, len(keys), _WRITE_BATCH):
                await _run(
                    tx,
                    "UNWIND $rows AS row "
                    "MATCH (a:Assertion {assertion_id: row.assertion_id}) "
                    "SET a.claim_key = row.claim_key",
                    rows=keys[start : start + _WRITE_BATCH],
                )
            for relation, rows in links.items():
                for start in range(0, len(rows), _WRITE_BATCH):
                    await _run(
                        tx,
                        f"""
                        UNWIND $rows AS row
                        MATCH (a:Assertion {{assertion_id: row.source}})
                        MATCH (b:Assertion {{assertion_id: row.target}})
                        MERGE (a)-[r:{relation} {{method: $method}}]->(b)
                        SET r.claim_key = row.claim_key,
                            r.observed_at = row.observed_at,
                            r.computed_at = $computed_at
                        """,
                        rows=rows[start : start + _WRITE_BATCH],
                        method=METHOD,
                        computed_at=computed_at,
                    )
            for (left_kind, right_kind), rows in sorted(groups.items()):
                left = cypher_identifier(left_kind)
                right = cypher_identifier(right_kind)
                for start in range(0, len(rows), _WRITE_BATCH):
                    batch = rows[start : start + _WRITE_BATCH]
                    # Similarity and duplicate candidates learn how many
                    # contexts their concepts share; an incremental run
                    # sees part of them, so a count never decreases here.
                    await _run(
                        tx,
                        f"""
                        UNWIND $rows AS row
                        MATCH (x:{left} {{concept_id: row.left}})
                              -[r:SIMILAR_TO|POSSIBLY_SAME_AS]-
                              (y:{right} {{concept_id: row.right}})
                        SET r.shared_claims = CASE
                            WHEN r.shared_claims IS NULL
                              OR row.shared_claims > r.shared_claims
                            THEN row.shared_claims ELSE r.shared_claims END
                        """,
                        rows=batch,
                    )
                    strong = [
                        row
                        for row in batch
                        if row["shared_claims"] >= min_shared
                    ]
                    if not strong:
                        continue
                    await _run(
                        tx,
                        f"""
                        UNWIND $rows AS row
                        MATCH (x:{left} {{concept_id: row.left}})
                        MATCH (y:{right} {{concept_id: row.right}})
                        MERGE (x)-[r:SHARES_CONTEXT_WITH
                                   {{method: $method}}]->(y)
                        SET r.shared_claims = CASE
                              WHEN r.shared_claims IS NULL
                                OR row.shared_claims > r.shared_claims
                              THEN row.shared_claims ELSE r.shared_claims END,
                            r.examples = row.examples,
                            r.observed_at = row.observed_at,
                            r.computed_at = $computed_at
                        """,
                        rows=strong,
                        method=METHOD,
                        computed_at=computed_at,
                    )

        await session.execute_write(write)


async def reconcile(
    store: Any, version_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Link the claims of these versions with the graph (None: rebuild
    the whole layer, replacing earlier computed edges)."""
    options = settings()
    if version_ids is None:
        pool = await read_claims(store)
        new_ids = None
        subjects = None
    else:
        fresh = await read_claims(store, version_ids=version_ids)
        excluded = set(
            options.get("excluded_partner_kinds", ["Country", "Domain"])
        )
        # A country or a domain is shared by a large part of the graph;
        # every slot of a new claim also names its other concepts.
        concepts: Dict[str, List[str]] = {}
        for claim in fresh:
            for concept, kind in claim["kinds"].items():
                known = concepts.setdefault(kind, [])
                if kind not in excluded and concept not in known:
                    known.append(concept)
        concepts = {kind: ids for kind, ids in concepts.items() if ids}
        pool = (
            await read_claims(store, concept_ids=concepts) if concepts else []
        )
        new_ids = {claim["assertion_id"] for claim in fresh}
        subjects = {
            claim["roles"]["subject"]
            for claim in fresh
            if "subject" in claim["roles"]
        }
    kinds = {
        concept: kind
        for claim in pool
        for concept, kind in claim["kinds"].items()
    }
    links = claim_links(pool, new_ids)
    contexts = shared_contexts(
        pool,
        kinds,
        min_shared=1,
        excluded_kinds=options.get(
            "excluded_partner_kinds", ["Country", "Domain"]
        ),
        subjects=subjects,
        max_members=int(options.get("max_context_members", 200)),
    )
    await _write(
        store,
        pool
        if version_ids is None
        else [claim for claim in pool if claim["assertion_id"] in new_ids],
        links,
        contexts,
        replace=version_ids is None,
    )
    summary = {
        "claims": len(pool),
        "slots": len({c["claim_key"] for c in pool if c["claim_key"]}),
        "corroborates": len(links["CORROBORATES"]),
        "contradicts": len(links["CONTRADICTS"]),
        "shared_context_pairs": sum(
            row["shared_claims"] >= int(options.get("min_shared", 2))
            for row in contexts
        ),
        "scope": "graph" if version_ids is None else "versions",
    }
    logger.info("Claims reconciled: %s", summary)
    return summary


async def reconcile_quietly(store: Any, version_ids: List[str]) -> None:
    """The ingestion hook: a failed reconciliation never fails a document
    (``lctrend reconcile-claims`` rebuilds the layer)."""
    if not settings().get("on_ingest", True) or not hasattr(store, "_driver"):
        # Test doubles and other stores without a Neo4j driver.
        return
    try:
        await reconcile(store, version_ids)
    except Exception as exc:
        logger.warning(
            "Claim reconciliation skipped for %s (%s)",
            version_ids,
            type(exc).__name__,
        )
        logger.debug("Reconciliation traceback", exc_info=True)
