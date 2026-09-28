"""Point-in-time technology features.

Every function takes a :class:`TechnologyView` of one snapshot (plus shared
snapshot context) and returns plain columns. Missing is ``None``, never 0:
a model must tell "searched, nothing found" from "not collected".
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .temporal import (
    CODE,
    PACKAGE,
    PATENT,
    SCHOLARLY,
    DocumentTrace,
    SnapshotView,
    TechnologyView,
    independence_groups,
    parse_date,
)

Row = Dict[str, Any]


def months_before(value: date, months: int) -> date:
    """The same day ``months`` earlier (clamped to the month's end)."""
    index = value.year * 12 + value.month - 1 - months
    year, month = divmod(index, 12)
    month += 1
    for day in (value.day, 30, 29, 28):
        try:
            return date(year, month, min(day, value.day))
        except ValueError:
            continue
    raise AssertionError("unreachable")


def log_growth(recent: float, previous: float) -> float:
    """log((recent + 1) / (previous + 1)): 0 -> 1 grows, 1 -> 1 does not,
    and growth and decline are symmetric.
    """
    return math.log((recent + 1.0) / (previous + 1.0))


def entropy(counts: Iterable[float]) -> Optional[float]:
    values = [value for value in counts if value > 0]
    total = sum(values)
    if not total:
        return None
    return -sum(value / total * math.log(value / total) for value in values)


def hhi(counts: Iterable[float]) -> Optional[float]:
    values = [value for value in counts if value > 0]
    total = sum(values)
    if not total:
        return None
    return sum((value / total) ** 2 for value in values)


def ratio(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator if denominator else None


def _window(
    dated: Sequence[Tuple[date, float]], start: date, end: date
) -> float:
    """Sum of values dated in (start, end]."""
    return sum(value for when, value in dated if start < when <= end)


def _days(later: Optional[date], earlier: Optional[date]) -> Optional[int]:
    if later is None or earlier is None:
        return None
    return (later - earlier).days


# ---------------------------------------------------------------- activity


def activity_features(view: TechnologyView, cutoff: date) -> Row:
    documents = view.documents
    first_seen = view.first_seen
    year_ago = months_before(cutoff, 12)
    two_years_ago = months_before(cutoff, 24)
    # Undated documents count in totals but cannot be placed in a window.
    dated_documents = [
        (item.first_visible, 1.0) for item in view.dated_documents
    ]
    recent = _window(dated_documents, year_ago, cutoff)
    previous = _window(dated_documents, two_years_ago, year_ago)
    citations, citations_known = 0.0, False
    for document in documents:
        by_year = document.history.get("citations_by_year")
        if by_year:
            citations += sum(by_year.values())
            citations_known = True
        elif document.metrics and "citation_count" in document.metrics:
            citations += document.metrics["citation_count"]
            citations_known = True
    scholarly = any(item.family == SCHOLARLY for item in documents)
    return {
        "first_seen_date": first_seen.isoformat() if first_seen else None,
        "technology_age_days": _days(cutoff, first_seen),
        "document_count": len(documents),
        "mention_count": sum(item.mention_count for item in documents),
        "documents_last_year": int(recent),
        "publication_growth": log_growth(recent, previous),
        "citation_count": citations if citations_known else None,
        "citation_count_missing": scholarly and not citations_known,
    }


# ---------------------------------------------------------------- dynamics


def _mention_series(view: TechnologyView) -> List[Tuple[date, float]]:
    return [
        (day.observed, float(day.mentions))
        for document in view.dated_documents
        for day in document.mentions
    ]


def dynamics_features(
    view: TechnologyView, cutoff: date, families: Dict[str, str]
) -> Row:
    series = _mention_series(view)
    row: Row = {}
    for months in (3, 6, 12, 24):
        start = months_before(cutoff, months)
        row[f"mention_growth_{months}m"] = log_growth(
            _window(series, start, cutoff),
            _window(series, months_before(cutoff, 2 * months), start),
        )
    year_ago = months_before(cutoff, 12)
    two_years_ago = months_before(cutoff, 24)
    for family, name in families.items():
        dated = [
            (item.first_visible, 1.0)
            for item in view.dated_documents
            if item.family == family
        ]
        row[f"publication_growth_{name}"] = log_growth(
            _window(dated, year_ago, cutoff),
            _window(dated, two_years_ago, year_ago),
        )
    six, twelve, eighteen = (months_before(cutoff, m) for m in (6, 12, 18))
    row["growth_acceleration"] = log_growth(
        _window(series, six, cutoff), _window(series, twelve, six)
    ) - log_growth(
        _window(series, twelve, six), _window(series, eighteen, twelve)
    )
    three = months_before(cutoff, 3)
    recent = _window(series, three, cutoff)
    expected = 3.0 * _window(series, two_years_ago, three) / 21.0
    row["burst_score"] = (recent - expected) / math.sqrt(expected + 1.0)
    quarters = [
        _window(
            series,
            months_before(cutoff, 3 * (index + 1)),
            months_before(cutoff, 3 * index),
        )
        for index in reversed(range(8))
    ]
    row["growth_persistence"] = (
        sum(later > earlier for earlier, later in zip(quarters, quarters[1:]))
        / 7.0
    )
    first_seen = view.first_seen
    span = 24
    if first_seen is not None:
        age_months = (cutoff.year - first_seen.year) * 12 + (
            cutoff.month - first_seen.month
        )
        span = max(1, min(24, age_months + 1))
    active = sum(
        _window(
            series,
            months_before(cutoff, index + 1),
            months_before(cutoff, index),
        )
        > 0
        for index in range(span)
    )
    row["active_month_ratio"] = active / span
    recent_year = _window(series, year_ago, cutoff)
    peak = recent_year
    if first_seen is not None:
        months = (cutoff.year - first_seen.year) * 12 + cutoff.month
        for offset in range(0, months - first_seen.month + 1):
            end = months_before(cutoff, offset)
            peak = max(peak, _window(series, months_before(end, 12), end))
    row["decline_rate"] = 1.0 - recent_year / peak if peak else 0.0
    row["time_since_last_signal_days"] = _days(cutoff, view.last_signal())
    return row


# ------------------------------------------------------------- convergence


def convergence_features(
    view: TechnologyView,
    cutoff: date,
    commercial_users: Sequence[date],
) -> Row:
    documents = view.documents
    families = Counter(item.family for item in documents)
    sources = independence_groups(item.version for item in documents)
    # A document naming nobody is its own source.
    groups = Counter(
        sources[item.version.version_id] or f"document:{item.document_id}"
        for item in documents
    )
    year_ago = months_before(cutoff, 12)
    dated = view.dated_documents
    first = {
        family: min(
            (item.first_visible for item in dated if item.family == family),
            default=None,
        )
        for family in (SCHOLARLY, CODE, PACKAGE, PATENT)
    }
    paper = first[SCHOLARLY]
    company_use = min(commercial_users, default=None)
    chain = 0
    if paper is not None:
        chain = sum(
            stage is not None and stage >= paper
            for stage in (
                first[PACKAGE],
                first[CODE],
                first[PATENT],
                company_use,
            )
        )
    return {
        "source_type_diversity": len(families),
        "independence_group_diversity": len(groups),
        "source_entropy": entropy(families.values()),
        "independence_group_entropy": entropy(groups.values()),
        "recent_source_convergence": len(
            {item.family for item in dated if item.first_visible > year_ago}
        ),
        "paper_to_package_lag_days": _days(first[PACKAGE], paper),
        "paper_to_repository_lag_days": _days(first[CODE], paper),
        "paper_to_patent_lag_days": _days(first[PATENT], paper),
        "paper_to_company_use_lag_days": _days(company_use, paper),
        # Paper -> package -> repository -> patent -> company use: each later
        # stage is stronger evidence than more papers.
        "evidence_chain_length": chain,
    }


# ---------------------------------------------------------- participants


def _parties(document: DocumentTrace) -> Tuple[str, ...]:
    version = document.version
    return tuple(
        dict.fromkeys(
            [*version.organizations, *version.companies, *version.universities]
        )
    )


def participant_features(view: TechnologyView, cutoff: date) -> Row:
    year_ago = months_before(cutoff, 12)
    documents = view.documents
    dated = view.dated_documents
    recent = [item for item in dated if item.first_visible > year_ago]
    earlier = [item for item in dated if item.first_visible <= year_ago]
    authors = {a for item in documents for a in item.version.contributors}
    recent_authors = {a for item in recent for a in item.version.contributors}
    earlier_authors = {
        a for item in earlier for a in item.version.contributors
    }
    countries = Counter(
        c for item in documents for c in item.version.countries
    )
    organizations = Counter(o for item in documents for o in _parties(item))
    company_documents = sum(bool(item.version.companies) for item in documents)
    university_documents = sum(
        bool(item.version.universities) for item in documents
    )
    recent_companies = {c for item in recent for c in item.version.companies}
    earlier_companies = {c for item in earlier for c in item.version.companies}
    total_organizations = sum(organizations.values())
    return {
        "country_count": len(countries),
        "company_count": len(
            {c for item in documents for c in item.version.companies}
        ),
        "university_count": len(
            {u for item in documents for u in item.version.universities}
        ),
        "domain_count": len(
            {d for item in documents for d in item.version.domains}
        ),
        "unique_author_count": len(authors),
        "new_author_rate": ratio(
            len(recent_authors - earlier_authors), len(recent_authors)
        ),
        # Retention: authors of earlier work who kept working on it.
        "returning_author_rate": ratio(
            len(earlier_authors & recent_authors), len(earlier_authors)
        ),
        "author_country_entropy": entropy(countries.values()),
        "organization_entropy": entropy(organizations.values()),
        "organization_hhi": hhi(organizations.values()),
        "top_organization_share": ratio(
            max(organizations.values(), default=0), total_organizations
        ),
        # +1: only companies, -1: only universities.
        "university_company_balance": ratio(
            company_documents - university_documents,
            company_documents + university_documents,
        ),
        "new_company_rate": ratio(
            len(recent_companies - earlier_companies), len(recent_companies)
        ),
    }


# ---------------------------------------------------- relations from text


RELATION_COUNTS = {
    "DEVELOPED_BY": "developer_count",
    "USED_BY": "user_count",
    "FUNDED_BY": "funder_count",
    "DEVELOPED_IN": "text_country_count",
    "SUBTECHNOLOGY_OF": "parent_technology_count",
    "SOLVES": "task_count",
}


def is_company(event_data: Dict[str, Any]) -> bool:
    labels = set(event_data.get("target_labels") or [])
    return "Company" in labels or event_data.get("target_kind") == "Company"


def relation_features(view: TechnologyView) -> Row:
    targets: Dict[str, set] = {name: set() for name in RELATION_COUNTS}
    for event in view.relations:
        relation = event.data.get("relation")
        if relation in targets:
            targets[relation].add(event.data.get("target_id"))
    row = {
        column: len(targets[relation])
        for relation, column in RELATION_COUNTS.items()
    }
    row["commercial_user_count"] = len(
        {
            event.data.get("target_id")
            for event in view.relations
            if event.data.get("relation") == "USED_BY"
            and is_company(event.data)
        }
    )
    return row


# ------------------------------------------------------------ credibility


def credibility_features(
    view: TechnologyView,
    tier_max: int,
    speculative: Sequence[str],
    confirmed_evidence: float,
) -> Row:
    mentions = sum(item.mention_count for item in view.documents)
    days = [day for item in view.documents for day in item.mentions]
    weighted = sum(
        day.mentions * item.version.reliability_tier / tier_max
        for item in view.documents
        for day in item.mentions
    )
    accepted = sum(day.accepted for day in days)
    hashes = [h for day in days for h in day.content_hashes]
    claims = [event.data for event in view.assertions]
    supported = [
        claim
        for claim in claims
        if claim.get("verification_status") == "supported"
    ]
    versions = view_versions(view)
    sources = independence_groups(versions.values())

    def source(version) -> str:
        # A document naming nobody is its own source.
        return (
            sources[version.version_id] or f"document:{version.document_id}"
        )

    families = {
        claim.get("evidence_family_id")
        or claim.get("claim_group_id")
        or (
            source(versions[claim["version_id"]])
            if claim.get("version_id") in versions
            else claim.get("version_id")
        )
        for claim in supported
    }
    confidences = [
        float(claim["confidence"])
        for claim in claims
        if claim.get("confidence") is not None
    ]
    return {
        "reliability_weighted_mentions": weighted,
        "accepted_mention_ratio": ratio(accepted, mentions),
        "supported_claim_ratio": ratio(len(supported), len(claims)),
        "independent_evidence_family_count": len(families),
        "speculative_claim_ratio": ratio(
            sum(claim.get("modality") in speculative for claim in claims),
            len(claims),
        ),
        "negated_claim_ratio": ratio(
            sum(claim.get("polarity") == "negated" for claim in claims),
            len(claims),
        ),
        "duplicate_content_ratio": (
            1.0 - len(set(hashes)) / len(hashes) if hashes else None
        ),
        # Many claims resting on few independent sources.
        "claim_to_evidence_gap": math.log1p(len(claims))
        - math.log1p(len(families)),
        "mean_extraction_confidence": (
            sum(confidences) / len(confidences) if confidences else None
        ),
        # Attention without confirmation is a hype pattern.
        "attention_evidence_gap": math.log1p(mentions)
        - math.log1p(confirmed_evidence),
    }


def view_versions(view: TechnologyView) -> Dict[str, Any]:
    return {
        day.version_id: item.version
        for item in view.documents
        for day in item.mentions
    }


# -------------------------------------------------------------- economics


def classify_economic(data: Dict[str, Any], factual: Sequence[str]) -> str:
    """Separate confirmation from forecasts and extraction candidates."""
    if data.get("polarity") == "negated":
        return "negated"
    if data.get("modality") in ("planned", "hypothetical"):
        return "forecast"
    if (
        data.get("polarity") == "affirmed"
        and data.get("modality") in factual
        and data.get("status") in ("accepted", "supported")
        and data.get("verification_status", "supported") == "supported"
    ):
        return "reviewed"
    return "candidate"


def economic_features(
    view: TechnologyView, cutoff: date, factual: Sequence[str], tier_max: int
) -> Row:
    classified = [
        (classify_economic(event.data, factual), event)
        for event in view.economics
    ]
    confirmed = [
        event for kind, event in classified if kind in ("fact", "reviewed")
    ]
    categories = Counter(event.data.get("category") for event in confirmed)
    usd_funding = [
        float(event.data["amount_value"])
        for event in confirmed
        if event.data.get("category") == "investment"
        and event.data.get("amount_value") is not None
        and event.data.get("currency") == "USD"
    ]
    confidences = [
        float(event.data["confidence"])
        for event in confirmed
        if event.data.get("confidence") is not None
    ]
    tiers = [
        int(event.data["reliability_tier"]) / tier_max
        for event in confirmed
        if event.data.get("reliability_tier") is not None
    ]
    performance = sum(
        event.data.get("predicate") == "changes_metric"
        and event.data.get("verification_status") == "supported"
        and event.data.get("status") in ("accepted", "supported")
        and event.data.get("polarity") == "affirmed"
        and event.data.get("modality") in factual
        for event in view.assertions
    )
    return {
        "economic_evidence_count": len(confirmed),
        "economic_forecast_count": sum(k == "forecast" for k, _ in classified),
        "economic_candidate_count": sum(
            k == "candidate" for k, _ in classified
        ),
        "economic_planned_count": sum(
            event.data.get("polarity") != "negated"
            and event.data.get("modality") == "planned"
            for event in view.economics
        ),
        "economic_hypothetical_count": sum(
            event.data.get("polarity") != "negated"
            and event.data.get("modality") == "hypothetical"
            for event in view.economics
        ),
        "economic_negated_count": sum(k == "negated" for k, _ in classified),
        "economic_reviewed_count": sum(k == "reviewed" for k, _ in classified),
        "economic_category_diversity": len(categories),
        "investment_evidence_count": categories.get("investment", 0),
        "funding_amount_usd": sum(usd_funding) if usd_funding else None,
        "market_size_evidence": categories.get("market", 0),
        "cost_reduction_evidence": categories.get("cost", 0)
        + categories.get("savings", 0),
        "performance_gain_evidence": performance,
        "procurement_evidence": categories.get("procurement", 0),
        "economic_signal_recency_days": _days(
            cutoff, max((e.observed for e in confirmed), default=None)
        ),
        "economic_confidence": (
            sum(confidences) / len(confidences) if confidences else None
        ),
        "economic_source_reliability": (
            sum(tiers) / len(tiers) if tiers else None
        ),
    }


# --------------------------------------------------------------- maturity


def _max_rank(events: Iterable[Any], until: date) -> Optional[int]:
    ranks = [
        int(event.data["stage_rank"])
        for event in events
        if event.observed <= until and event.data.get("stage_rank") is not None
    ]
    return max(ranks, default=None)


def _first_rank(events: Iterable[Any], rank: int) -> Optional[date]:
    return min(
        (
            event.observed
            for event in events
            if event.data.get("stage_rank") is not None
            and int(event.data["stage_rank"]) >= rank
        ),
        default=None,
    )


def maturity_features(
    view: TechnologyView,
    snapshot: SnapshotView,
    stages: Dict[str, int],
    families: Dict[str, int],
) -> Row:
    cutoff = snapshot.cutoff
    year_ago = months_before(cutoff, 12)
    two_years_ago = months_before(cutoff, 24)
    rank = _max_rank(view.maturity, cutoff)
    rank_year_ago = _max_rank(view.maturity, year_ago)
    trls = [
        int(event.data["trl"])
        for event in view.maturity
        if event.data.get("trl") is not None
    ]
    prototype = _first_rank(view.maturity, stages["prototype"])
    pilot = _first_rank(view.maturity, stages["pilot"])
    repositories = [item for item in view.documents if item.family == CODE]
    packages = [item for item in view.documents if item.family == PACKAGE]
    patents = [item for item in view.documents if item.family == PATENT]
    repository_releases = [
        parse_date(value)
        for item in repositories
        for value in item.history.get("release_dates", [])
    ]
    package_releases = [
        parse_date(value)
        for item in packages
        for value in item.history.get("release_dates", [])
    ]
    commits = sum(
        total
        for item in repositories
        for week, total in item.history.get("commit_weeks", {}).items()
        if parse_date(week) and parse_date(week) > year_ago
    )
    contributor_weeks = [
        parse_date(value)
        for item in repositories
        for value in item.history.get("contributor_first_weeks", [])
    ]
    patent_dates = [
        when
        for item in patents
        if (
            when := parse_date(item.history.get("priority_date"))
            or item.version.document_date
            or (None if item.undated else item.first_visible)
        )
        is not None
    ]
    family_ids = [
        item.history.get("family_id")
        for item in patents
        if item.history.get("family_id")
    ]
    parties = Counter(
        event.data.get("target_id")
        for event in view.relations
        if event.data.get("relation") in ("DEVELOPED_BY", "USED_BY")
    )
    return {
        "max_maturity_rank": rank,
        # No reviewed maturity claim by T (e.g. the "none" extractor).
        "max_maturity_rank_missing": rank is None,
        "max_trl": max(trls, default=None),
        "maturity_growth": (
            (rank or 0) - (rank_year_ago or 0) if rank is not None else None
        ),
        "time_in_prototype_stage_days": (
            _days(pilot or cutoff, prototype) if prototype else None
        ),
        "time_to_pilot_days": _days(pilot, view.first_seen),
        "implementation_count": len(repositories) + len(packages),
        "repository_release_frequency": (
            sum(1 for d in repository_releases if d and d > two_years_ago)
            / 2.0
            if repositories
            else None
        ),
        "package_activity": (
            sum(1 for d in package_releases if d and d > year_ago)
            if packages
            else None
        ),
        "commit_activity_12m": (
            commits
            if any("commit_weeks" in item.history for item in repositories)
            else None
        ),
        "contributor_growth_12m": (
            sum(1 for d in contributor_weeks if d and d > year_ago)
            if any(
                "contributor_first_weeks" in item.history
                for item in repositories
            )
            else None
        ),
        "patent_age_days": _days(cutoff, min(patent_dates, default=None)),
        "patent_family_size": (
            max(families.get(family, 1) for family in family_ids)
            if family_ids
            else (1 if patents else None)
        ),
        "standardization_presence": any(
            item.version.document_type == "standard" for item in view.documents
        ),
        "market_concentration": hhi(parties.values()),
    }


def growth_plateau_score(mention_count: int, growth_12m: float) -> float:
    """High for large, flat volume: attention that stopped growing."""
    volume = min(1.0, math.log1p(mention_count) / math.log1p(100))
    return (1.0 - min(1.0, abs(growth_12m))) * volume


# ------------------------------------------------------------ data quality


def quality_features(view: TechnologyView) -> Row:
    documents = view.documents
    days = [day for item in documents for day in item.mentions]
    mentions = sum(day.mentions for day in days)
    return {
        "fulltext_coverage": ratio(
            sum(
                item.version.coverage in ("full_text", "selected_files")
                for item in documents
            ),
            len(documents),
        ),
        "metadata_only_ratio": ratio(
            sum(
                item.version.coverage == "metadata_only" for item in documents
            ),
            len(documents),
        ),
        "provisional_resolution_ratio": ratio(
            sum(day.provisional for day in days), mentions
        ),
        "ambiguous_resolution_ratio": ratio(
            sum(day.ambiguous for day in days), mentions
        ),
    }


def coverage_features(
    view: TechnologyView, covered: set, families: Dict[str, str]
) -> Row:
    """Per source family: searched by T, count found, count unknown."""
    row: Row = {}
    for family, name in families.items():
        is_covered = family in covered
        count = sum(item.family == family for item in view.documents)
        row[f"{name}_source_covered"] = is_covered
        row[f"{name}_count"] = count if is_covered else None
        row[f"{name}_count_missing"] = not is_covered
    row["source_coverage_count"] = sum(
        family in covered for family in families
    )
    return row


def company_use_dates(view: TechnologyView) -> List[date]:
    return [
        event.observed
        for event in view.relations
        if event.data.get("relation") == "USED_BY" and is_company(event.data)
    ]


def year_ago_of(cutoff: date) -> date:
    return months_before(cutoff, 12)


__all__ = [
    "activity_features",
    "company_use_dates",
    "convergence_features",
    "coverage_features",
    "credibility_features",
    "dynamics_features",
    "economic_features",
    "entropy",
    "growth_plateau_score",
    "hhi",
    "log_growth",
    "maturity_features",
    "months_before",
    "participant_features",
    "quality_features",
    "relation_features",
    "timedelta",
    "year_ago_of",
]
