"""Point-in-time view of the graph for the temporal dataset.

The graph holds every document version, mention and relation ever
collected, including those published or observed after a snapshot date.
A :class:`SnapshotView` keeps only what was knowable at the date ``T``:

- a document version is visible when published and retrieved by ``T``;
- a mention counts when its own date (``MENTIONS.observed_at``) is by ``T``
  and it belongs to a visible version;
- metrics (citations, stars) count only from a version whose metrics were
  observed by ``T``;
- dated histories inside version metadata (releases, weekly commits,
  citations per year) are cut at ``T``;
- relations, maturity, economic evidence and assertions count when observed
  by ``T``. Undated rows are never used: their date cannot be placed.

Parties of a version (authors, organizations, countries, domains) come from
the version itself, so later updates of the Document node cannot leak.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Source families that matter for implementation outcomes.
SCHOLARLY, CODE, PACKAGE, PATENT = (
    "scholarly",
    "code",
    "package_registry",
    "patent",
)
FAMILY_BY_TYPE = {
    "article": SCHOLARLY,
    "repository": CODE,
    "package": PACKAGE,
    "patent": PATENT,
}


def parse_date(value: object) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _json(value: object) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        loaded = json.loads(str(value))
    except ValueError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _tuple(value: object) -> Tuple[str, ...]:
    return tuple(str(item) for item in value or () if item is not None)


@dataclass(frozen=True)
class Version:
    document_id: str
    version_id: str
    document_type: str
    family: str
    source_id: str
    independence_group: Optional[str]
    reliability_tier: int
    document_date: Optional[date]
    version_date: Optional[date]
    metrics_date: Optional[date]
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    coverage: str
    countries: Tuple[str, ...]
    companies: Tuple[str, ...]
    universities: Tuple[str, ...]
    organizations: Tuple[str, ...]
    domains: Tuple[str, ...]
    contributors: Tuple[str, ...]
    extracted: bool
    retrieved_date: Optional[date] = None
    extracted_date: Optional[date] = None

    @property
    def available_date(self) -> date:
        """Publication and collection must both precede a snapshot."""
        return max(
            when
            for when in (self.version_date, self.retrieved_date)
            if when is not None
        )

    @property
    def independence_key(self) -> str:
        """Who stands behind the document.

        An explicit independence group (a project shared by a repository and
        its package) wins; otherwise the first organization or author team;
        otherwise the document alone.
        """
        if self.independence_group:
            return f"group:{self.independence_group}"
        if self.organizations or self.companies or self.universities:
            return "org:" + min(
                self.organizations or self.companies or self.universities
            )
        if self.contributors:
            return "people:" + ",".join(sorted(self.contributors)[:3])
        return f"document:{self.document_id}"


@dataclass(frozen=True)
class MentionDay:
    """Mentions of one technology in one version on one content date."""

    version_id: str
    observed: date
    mentions: int
    confidence_sum: float
    confidence_count: int
    accepted: int
    provisional: int
    ambiguous: int
    content_hashes: Tuple[str, ...]


@dataclass(frozen=True)
class Event:
    """A dated per-technology row: relation, maturity, economics, claim."""

    observed: date
    data: Dict[str, Any]


@dataclass
class DocumentTrace:
    """One document as seen by one technology at a snapshot."""

    document_id: str
    # Latest version published by T: parties, coverage, source.
    version: Version
    # When the technology became visible in this document.
    first_visible: date
    # Mentions of the latest visible version that carries this technology.
    mentions: List[MentionDay]
    # Metrics of the latest version whose metrics were observed by T.
    metrics: Optional[Dict[str, float]]
    # Dated histories cut at T (release_dates, commit_weeks, ...).
    history: Dict[str, Any]

    @property
    def family(self) -> str:
        return self.version.family

    @property
    def mention_count(self) -> int:
        return sum(item.mentions for item in self.mentions)


@dataclass
class TechnologyView:
    technology_id: str
    label: str
    documents: List[DocumentTrace] = field(default_factory=list)
    relations: List[Event] = field(default_factory=list)
    maturity: List[Event] = field(default_factory=list)
    economics: List[Event] = field(default_factory=list)
    assertions: List[Event] = field(default_factory=list)

    @property
    def first_seen(self) -> Optional[date]:
        return min(
            (item.first_visible for item in self.documents), default=None
        )

    def last_signal(self) -> Optional[date]:
        dates = [
            day.observed
            for document in self.documents
            for day in document.mentions
        ]
        for events in (
            self.relations,
            self.maturity,
            self.economics,
            self.assertions,
        ):
            dates.extend(event.observed for event in events)
        return max(dates, default=None)


def _history(
    metadata: Dict[str, Any], cutoff: date, metrics_known: bool = True
) -> Dict[str, Any]:
    """Dated lists from version metadata, cut at the snapshot date."""
    history: Dict[str, Any] = {}
    for key in ("release_dates", "contributor_first_weeks"):
        values = metadata.get(key)
        if isinstance(values, list):
            history[key] = [
                when.isoformat()
                for value in values
                if (when := parse_date(value)) is not None and when <= cutoff
            ]
    weeks = metadata.get("commit_weeks")
    if isinstance(weeks, dict):
        history["commit_weeks"] = {
            week: total
            for week, total in weeks.items()
            if parse_date(week) is not None and parse_date(week) <= cutoff
        }
    counts = metadata.get("counts_by_year")
    if isinstance(counts, list) and metrics_known:
        # Only years complete by the snapshot are known then.
        history["citations_by_year"] = {
            int(item["year"]): float(item.get("cited_by_count") or 0)
            for item in counts
            if isinstance(item, dict)
            and str(item.get("year", "")).isdigit()
            and int(item["year"]) < cutoff.year
        }
    if metadata.get("family_id"):
        history["family_id"] = metadata["family_id"]
    priority = parse_date(metadata.get("priority_date"))
    if priority is not None and priority <= cutoff:
        history["priority_date"] = priority.isoformat()
    return history


class TemporalCorpus:
    """All dated graph data, indexed for point-in-time snapshots."""

    def __init__(self, data: Dict[str, List[Dict[str, Any]]]) -> None:
        self.versions: Dict[str, Version] = {}
        self.document_versions: Dict[str, List[Version]] = {}
        skipped = 0
        for row in data.get("versions", []):
            document_date = parse_date(row.get("document_published_at"))
            version_date = (
                parse_date(row.get("version_published_at"))
                or document_date
                or parse_date(row.get("retrieved_at"))
            )
            if version_date is None:
                skipped += 1
                continue
            document_type = str(row.get("document_type") or "")
            version = Version(
                document_id=str(row["document_id"]),
                version_id=str(row["version_id"]),
                document_type=document_type,
                family=str(
                    row.get("source_family")
                    or FAMILY_BY_TYPE.get(document_type)
                    or "other"
                ),
                source_id=str(row.get("source_id") or ""),
                independence_group=row.get("independence_group"),
                reliability_tier=int(row.get("reliability_tier") or 1),
                document_date=document_date,
                version_date=version_date,
                metrics_date=parse_date(row.get("metrics_observed_at")),
                metrics={
                    key: float(value)
                    for key, value in _json(row.get("metrics_json")).items()
                    if isinstance(value, (int, float))
                },
                metadata=_json(row.get("metadata_json")),
                coverage=str(row.get("coverage") or "metadata_only"),
                countries=_tuple(row.get("countries")),
                companies=_tuple(row.get("companies")),
                universities=_tuple(row.get("universities")),
                organizations=_tuple(row.get("organizations")),
                domains=_tuple(row.get("domains")),
                contributors=_tuple(row.get("contributors")),
                extracted=bool(row.get("extracted")),
                retrieved_date=parse_date(row.get("retrieved_at")),
                extracted_date=parse_date(row.get("extracted_at")),
            )
            self.versions[version.version_id] = version
            self.document_versions.setdefault(version.document_id, []).append(
                version
            )
        for versions in self.document_versions.values():
            versions.sort(
                key=lambda item: (item.version_date, item.version_id)
            )
        self.labels: Dict[str, str] = {}
        self.embeddings: Dict[str, List[float]] = {}
        self.embedding_dates: Dict[str, date] = {}
        for row in data.get("technologies", []):
            technology_id = str(row["technology_id"])
            self.labels[technology_id] = str(
                row.get("technology") or technology_id
            )
            embedding_date = parse_date(row.get("embedding_observed_at"))
            if row.get("embedding") and embedding_date is not None:
                self.embeddings[technology_id] = list(row["embedding"])
                self.embedding_dates[technology_id] = embedding_date

        # technology -> document -> version -> mention days
        self.mentions: Dict[str, Dict[str, Dict[str, List[MentionDay]]]] = {}
        for row in data.get("mentions", []):
            version = self.versions.get(str(row["version_id"]))
            if version is None:
                continue
            observed = parse_date(row.get("observed_at"))
            if observed is None:
                continue
            recorded = parse_date(row.get("recorded_at"))
            # A backfilled mention becomes a signal when it was knowable.
            available = max(
                observed,
                version.available_date,
                recorded or version.available_date,
            )
            technology_id = str(row["technology_id"])
            self.labels.setdefault(
                technology_id, str(row.get("technology") or technology_id)
            )
            self.mentions.setdefault(technology_id, {}).setdefault(
                version.document_id, {}
            ).setdefault(version.version_id, []).append(
                MentionDay(
                    version_id=version.version_id,
                    observed=available,
                    mentions=int(row.get("mentions") or 0),
                    confidence_sum=float(row.get("confidence_sum") or 0.0),
                    confidence_count=int(row.get("confidence_count") or 0),
                    accepted=int(row.get("accepted") or 0),
                    provisional=int(row.get("provisional") or 0),
                    ambiguous=int(row.get("ambiguous") or 0),
                    content_hashes=_tuple(row.get("content_hashes")),
                )
            )

        self.events: Dict[str, Dict[str, List[Event]]] = {}
        undated = 0
        for kind in ("relations", "maturity", "economics", "assertions"):
            for row in data.get(kind, []):
                # Direct projections are reviewed in the store. Explicit
                # review fields on imported projections must agree too.
                if kind in ("relations", "maturity") and (
                    row.get("polarity", "affirmed") != "affirmed"
                    or row.get("modality", "reported")
                    not in ("reported", "observed")
                    or (
                        "status" in row
                        and row["status"] not in ("accepted", "supported")
                    )
                    or (
                        "verification_status" in row
                        and row["verification_status"] != "supported"
                    )
                ):
                    continue
                observed = parse_date(row.get("observed_at"))
                version = self.versions.get(str(row.get("version_id") or ""))
                if observed is None or version is None:
                    undated += 1
                    continue
                observed = max(
                    observed,
                    version.available_date,
                    parse_date(row.get("recorded_at"))
                    or version.available_date,
                )
                self.events.setdefault(
                    str(row["technology_id"]), {}
                ).setdefault(kind, []).append(Event(observed, dict(row)))
        for kinds in self.events.values():
            for events in kinds.values():
                events.sort(key=lambda item: item.observed)

        self.crawls = [
            {
                **row,
                "period_start": parse_date(row.get("period_start")),
                "period_end": parse_date(row.get("period_end")),
                "finished_at": parse_date(row.get("finished_at")),
                "observed_at": parse_date(row.get("observed_at")),
                "retrieved_at": parse_date(row.get("retrieved_at")),
            }
            for row in data.get("crawls", [])
        ]
        dates = [
            day.observed
            for documents in self.mentions.values()
            for versions in documents.values()
            for days in versions.values()
            for day in days
        ] + [version.available_date for version in self.versions.values()]
        dates.extend(
            event.observed
            for kinds in self.events.values()
            for events in kinds.values()
            for event in events
        )
        dates.extend(
            version.metrics_date
            for version in self.versions.values()
            if version.metrics_date is not None
        )
        dates.extend(
            version.extracted_date
            for version in self.versions.values()
            if version.extracted_date is not None
        )
        dates.extend(
            crawl["finished_at"]
            or crawl["observed_at"]
            or crawl["retrieved_at"]
            for crawl in self.crawls
            if crawl["finished_at"]
            or crawl["observed_at"]
            or crawl["retrieved_at"]
        )
        self.latest_date: Optional[date] = max(dates, default=None)
        self.earliest_date: Optional[date] = min(dates, default=None)
        if skipped or undated:
            logger.info(
                "Temporal corpus: %d undated versions and %d undated "
                "relations are excluded",
                skipped,
                undated,
            )
        self._views: Dict[date, SnapshotView] = {}

    def visible_version(
        self, document_id: str, cutoff: date
    ) -> Optional[Version]:
        visible = [
            version
            for version in self.document_versions.get(document_id, [])
            if version.available_date <= cutoff
        ]
        if not visible:
            return None
        version = max(
            visible, key=lambda item: (item.version_date, item.version_id)
        )
        return replace(
            version,
            extracted=(
                version.extracted
                and version.extracted_date is not None
                and version.extracted_date <= cutoff
            ),
        )

    def metrics_at(
        self, document_id: str, cutoff: date
    ) -> Optional[Dict[str, float]]:
        for version in reversed(self.document_versions.get(document_id, [])):
            if (
                version.available_date <= cutoff
                and version.metrics_date is not None
                and version.metrics_date <= cutoff
                and version.metrics
            ):
                return version.metrics
        return None

    def covered_families(
        self,
        cutoff: date,
        technology_id: Optional[str] = None,
        period_start: Optional[date] = None,
        period_end: Optional[date] = None,
    ) -> set:
        """Completed relevant searches known at T.

        An explicit interval requires a complete crawl covering that whole
        interval. A found document establishes presence, never absence.
        An unrestricted, exhaustive crawl applies to every technology;
        query-specific crawls apply only to their technology or exact label.
        Without an interval, sampled crawls establish source observation.
        """
        families = set()
        for crawl in self.crawls:
            observed = max(
                (
                    when
                    for when in (
                        crawl["finished_at"],
                        crawl["observed_at"],
                        crawl["retrieved_at"],
                    )
                    if when is not None
                ),
                default=None,
            )
            if (
                observed is None
                or observed > cutoff
                or crawl.get("status") not in ("succeeded", "completed")
                or crawl.get("failures", 0)
                or (
                    (period_start is not None or period_end is not None)
                    and crawl.get("exhaustive") is not True
                )
                or not self._crawl_relevant(crawl, technology_id)
            ):
                continue
            start, end = crawl["period_start"], crawl["period_end"]
            if (
                period_start is not None
                and start is not None
                and start > period_start
            ):
                continue
            if period_end is not None and end is not None and end < period_end:
                continue
            if period_end is not None and observed < period_end:
                continue
            if period_start is None and start is not None and start > cutoff:
                continue
            if crawl.get("source_family"):
                families.add(crawl["source_family"])
        if period_start is None and period_end is None:
            if technology_id is None:
                versions = self.versions.values()
            else:
                versions = (
                    self.versions[version_id]
                    for document in self.mentions.get(
                        technology_id, {}
                    ).values()
                    for version_id, days in document.items()
                    if any(day.observed <= cutoff for day in days)
                )
            families.update(
                version.family
                for version in versions
                if version.available_date <= cutoff
            )
        return families

    def _crawl_relevant(
        self, crawl: Dict[str, Any], technology_id: Optional[str]
    ) -> bool:
        ids = set(crawl.get("technology_ids") or [])
        if crawl.get("technology_id"):
            ids.add(crawl["technology_id"])
        query = " ".join(
            re.findall(r"\w+", str(crawl.get("query") or "").casefold())
        )
        if not ids and not query:
            return True
        if technology_id is None:
            return False
        label = " ".join(
            re.findall(r"\w+", self.labels.get(technology_id, "").casefold())
        )
        return technology_id in ids or bool(label and query == label)

    def embeddings_at(self, cutoff: date) -> Dict[str, List[float]]:
        """Current vectors whose actual observation is known by T."""
        return {
            key: vector
            for key, vector in self.embeddings.items()
            if self.embedding_dates[key] <= cutoff
        }

    def event_visible(self, event: Event, cutoff: date) -> bool:
        version = self.versions.get(str(event.data.get("version_id") or ""))
        recorded = parse_date(event.data.get("recorded_at"))
        return (
            event.observed <= cutoff
            and version is not None
            and version.available_date <= cutoff
            and (recorded is None or recorded <= cutoff)
        )

    def first_visible(
        self, technology_id: str, document_id: str
    ) -> Optional[date]:
        """When a technology first became visible in a document: the
        earliest version that carries it, at the later of the version date
        and the mention date.
        """
        versions = self.mentions.get(technology_id, {}).get(document_id, {})
        candidates = [
            max(
                self.versions[version_id].available_date,
                min(d.observed for d in days),
            )
            for version_id, days in versions.items()
            if days
        ]
        return min(candidates, default=None)

    def view(self, cutoff: date) -> "SnapshotView":
        if cutoff not in self._views:
            self._views[cutoff] = SnapshotView(self, cutoff)
        return self._views[cutoff]


class SnapshotView:
    """Everything knowable at one snapshot date."""

    def __init__(self, corpus: TemporalCorpus, cutoff: date) -> None:
        self.corpus = corpus
        self.cutoff = cutoff
        self.technologies: Dict[str, TechnologyView] = {}
        for technology_id, documents in corpus.mentions.items():
            view = TechnologyView(technology_id, corpus.labels[technology_id])
            for document_id, versions in documents.items():
                trace = self._trace(document_id, versions)
                if trace is not None:
                    view.documents.append(trace)
            events = corpus.events.get(technology_id, {})
            for kind in ("relations", "maturity", "economics", "assertions"):
                setattr(
                    view,
                    kind,
                    [
                        event
                        for event in events.get(kind, [])
                        if corpus.event_visible(event, cutoff)
                    ],
                )
            if view.documents:
                self.technologies[technology_id] = view
        # Technologies known only through relations have no documents; they
        # are not candidates but still count as known concepts.
        self.documents: Dict[str, Version] = {}
        for document_id in corpus.document_versions:
            version = corpus.visible_version(document_id, cutoff)
            if version is not None:
                self.documents[document_id] = version

    def _trace(
        self, document_id: str, versions: Dict[str, List[MentionDay]]
    ) -> Optional[DocumentTrace]:
        cutoff = self.cutoff
        corpus = self.corpus
        carrying = [
            (
                corpus.versions[version_id],
                [d for d in days if d.observed <= cutoff],
            )
            for version_id, days in versions.items()
            if corpus.versions[version_id].available_date <= cutoff
        ]
        carrying = [(version, days) for version, days in carrying if days]
        if not carrying:
            return None
        first_visible = min(
            max(version.available_date, min(day.observed for day in days))
            for version, days in carrying
        )
        _, mentions = max(
            carrying,
            key=lambda item: (item[0].version_date, item[0].version_id),
        )
        visible = corpus.visible_version(document_id, cutoff)
        return DocumentTrace(
            document_id=document_id,
            version=visible,
            first_visible=first_visible,
            mentions=mentions,
            metrics=corpus.metrics_at(document_id, cutoff),
            history=_history(
                visible.metadata,
                cutoff,
                metrics_known=(
                    visible.metrics_date is not None
                    and visible.metrics_date <= cutoff
                ),
            ),
        )

    def known_technologies(self) -> Iterable[str]:
        return self.technologies.keys()
