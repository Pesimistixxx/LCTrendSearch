"""Work identity: the documents that are one work.

A paper reaches the graph as an OpenAlex record, an uploaded PDF and an
arXiv preprint; a patent is published as A1 and B1 and filed in several
offices. Each source keeps its own Document (its versions are that source's
observations); a Work groups the Documents of one work, so features count
works, not copies (graph.temporal reads the work of each document).

A document's keys come from its identifiers. A strong key identifies a work
alone: DOI, arXiv id, PMID, PMCID, OpenAlex work, patent publication number
without its kind code, patent family, repository, package, any other
source record id. The weak title key joins documents that share a long
title only when the title names one compatible work and no persistent
identifier of one scheme disagrees: a preprint (arXiv id) and its journal
version (DOI) join, two papers with different DOIs never do, and a title
two such papers share joins neither.

Grants and job postings stay one work per record: each award year and each
posting carries its own money (graph.features reads the economic facts of
one version per work).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)
from urllib.parse import unquote, urlsplit

from ..core.models import DocumentEnvelope, stable_id
from ..extraction.lexical import lexical_tokens

# Document types whose title names one work; a repository, a package, a
# grant or a posting is identified by its record, and patent titles
# ("Method and apparatus for ...") repeat across inventions.
TITLE_TYPES = frozenset({"article", "report", "standard", "news"})
# Shorter titles ("Introduction", "Editorial note") name many works.
TITLE_MIN_TOKENS = 5
# Persistent identifiers of a work: two documents holding different values
# of one of these schemes are different works whatever their titles say.
CONFLICT_SCHEMES = frozenset(
    {"doi", "arxiv", "pmid", "pmcid", "patent", "patent-family"}
)
# One work per record: no key beyond the record itself.
RECORD_TYPES = frozenset({"grant", "job_posting"})

_ARXIV_NEW = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
_ARXIV_OLD = re.compile(r"([a-z-]+(?:\.[a-z]{2})?/\d{7})(?:v\d+)?", re.I)
_ARXIV_DOI = re.compile(r"10\.48550/arxiv\.(.+)", re.I)
_PATENT = re.compile(r"([A-Z]{2})0*(\d+)(?:[A-Z]\d?)?")


@dataclass(frozen=True)
class WorkKey:
    key: str
    scheme: str
    strong: bool = True


def normalize_doi(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if value.lower().startswith(("http://", "https://")):
        parsed = urlsplit(value)
        if (parsed.hostname or "").lower() not in {
            "doi.org",
            "dx.doi.org",
            "www.doi.org",
        }:
            return None
        value = parsed.path.lstrip("/")
    value = re.sub(r"^doi\s*:\s*", "", value, flags=re.I)
    value = unquote(value).strip().casefold()
    return value if re.fullmatch(r"10\.\d{4,9}/\S+", value) else None


def normalize_arxiv(value: Any) -> Optional[str]:
    """An arXiv id without version: 2101.00001, hep-th/9901001."""
    if not isinstance(value, str):
        return None
    value = unquote(value.strip())
    doi = normalize_doi(value)
    if doi is not None:
        match = _ARXIV_DOI.fullmatch(doi)
        if match is None:
            return None
        value = match.group(1)
    if value.lower().startswith(("http://", "https://")):
        parsed = urlsplit(value)
        if not (parsed.hostname or "").lower().endswith("arxiv.org"):
            return None
        value = re.sub(r"^/(?:abs|pdf)/", "", parsed.path)
        value = re.sub(r"\.pdf$", "", value, flags=re.I)
    value = re.sub(r"^arxiv\s*:\s*", "", value, flags=re.I).strip()
    for pattern in (_ARXIV_NEW, _ARXIV_OLD):
        match = pattern.fullmatch(value)
        if match:
            return match.group(1).casefold()
    return None


def patent_number(value: Any) -> Optional[str]:
    """A publication number without its kind code: EP1234567A1 and
    EP1234567B1 are one patent."""
    if not isinstance(value, str):
        return None
    compact = re.sub(r"[\s.,/-]", "", value).upper()
    match = _PATENT.fullmatch(compact)
    return f"{match.group(1)}{match.group(2)}" if match else None


def _package(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value.strip()).casefold()


def _identifier_keys(scheme: str, value: str) -> List[WorkKey]:
    scheme = scheme.strip().casefold()
    value = str(value).strip()
    if not value:
        return []
    if scheme == "doi":
        doi = normalize_doi(value)
        if doi is None:
            return []
        arxiv = normalize_arxiv(doi)
        # arXiv's DOI names the preprint, which the arXiv id already does.
        return [
            WorkKey(f"arxiv:{arxiv}", "arxiv")
            if arxiv
            else WorkKey(f"doi:{doi}", "doi")
        ]
    if scheme == "arxiv":
        arxiv = normalize_arxiv(value)
        return [WorkKey(f"arxiv:{arxiv}", "arxiv")] if arxiv else []
    if scheme == "pmid":
        digits = re.sub(r"\D", "", value.rsplit("/", 1)[-1])
        return [WorkKey(f"pmid:{int(digits)}", "pmid")] if digits else []
    if scheme == "pmcid":
        match = re.search(r"PMC\d+", value, re.I)
        return (
            [WorkKey(f"pmcid:{match.group(0).upper()}", "pmcid")]
            if match
            else []
        )
    if scheme == "openalex":
        match = re.search(r"W\d+", value, re.I)
        return (
            [WorkKey(f"openalex:{match.group(0).upper()}", "openalex")]
            if match
            else []
        )
    if scheme == "pypi":
        return [WorkKey(f"pypi:{_package(value)}", "pypi")]
    if scheme in ("patent-publication", "patent"):
        number = patent_number(value)
        return [WorkKey(f"patent:{number}", "patent")] if number else []
    return [WorkKey(f"{scheme}:{value.casefold()}", scheme)]


def work_keys(
    document_type: str,
    identifiers: Iterable[Tuple[str, str]],
    title: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> List[WorkKey]:
    """The keys of a document's work, strong ones first, without repeats."""
    keys: List[WorkKey] = []
    for scheme, value in identifiers:
        keys += _identifier_keys(scheme, value)
    if document_type in RECORD_TYPES:
        return list(dict.fromkeys(keys))
    family = (metadata or {}).get("family_id")
    if document_type == "patent" and family:
        keys.append(WorkKey(f"patent-family:{family}", "patent-family"))
    if document_type in TITLE_TYPES and title:
        tokens = lexical_tokens(title)
        if len([token for token in tokens if not token.isdigit()]) >= (
            TITLE_MIN_TOKENS
        ):
            keys.append(
                WorkKey("title:" + " ".join(tokens), "title", strong=False)
            )
    return list(dict.fromkeys(keys))


def _external_pair(external_id: str) -> Tuple[str, str]:
    scheme, _, value = external_id.partition(":")
    return scheme, value


def document_work_keys(document: DocumentEnvelope) -> List[WorkKey]:
    return work_keys(
        document.document_type.value,
        [(item.scheme, item.value) for item in document.identifiers],
        document.title,
        document.metadata,
    )


def stored_work_keys(
    document_type: str,
    external_ids: Sequence[str],
    title: Optional[str],
    metadata: Optional[Mapping[str, Any]] = None,
) -> List[WorkKey]:
    """Keys of a stored Document (``external_ids`` as ``scheme:value``)."""
    return work_keys(
        document_type,
        [_external_pair(item) for item in external_ids or ()],
        title,
        metadata,
    )


@dataclass
class FoundWork:
    """A stored work reached by some of a document's keys."""

    work_id: str
    keys: List[WorkKey]
    via: Set[str] = field(default_factory=set)
    current: bool = False


@dataclass
class WorkPlan:
    work_id: str
    # Stored works that are this work too; they are folded into it.
    absorbed: List[str]
    # The document's keys; a strong one names only this work, a title
    # may name other works as well.
    keys: List[WorkKey]
    # Works the title matched but a persistent identifier contradicts.
    rejected: List[str] = field(default_factory=list)


def _conflicts(left: Iterable[WorkKey], right: Iterable[WorkKey]) -> bool:
    def by_scheme(keys: Iterable[WorkKey]) -> Dict[str, Set[str]]:
        values: Dict[str, Set[str]] = {}
        for key in keys:
            if key.strong and key.scheme in CONFLICT_SCHEMES:
                values.setdefault(key.scheme, set()).add(key.key)
        return values

    ours, theirs = by_scheme(left), by_scheme(right)
    return any(
        not ours[scheme] & theirs[scheme] for scheme in ours.keys() & theirs
    )


def plan_work(
    document_id: str, keys: Sequence[WorkKey], found: Sequence[FoundWork]
) -> WorkPlan:
    """The work of a document and the stored works it joins.

    Works reached by a strong key, and the document's current work, are
    this work. A work reached only by a title joins when the title names
    no other compatible work and no persistent identifier contradicts what
    is joined so far; a title that names several works is ambiguous and
    joins none. The smallest id survives.
    """
    strong = {key.key for key in keys if key.strong}
    forced = [work for work in found if work.current or work.via & strong]
    joined: List[WorkKey] = list(keys)
    for work in forced:
        joined += work.keys
    members = [work.work_id for work in forced]
    forced_ids = set(members)
    candidates = sorted(
        (work for work in found if work.work_id not in forced_ids),
        key=lambda item: item.work_id,
    )
    rejected = [
        work.work_id for work in candidates if _conflicts(joined, work.keys)
    ]
    compatible = [work for work in candidates if work.work_id not in rejected]
    named: Dict[str, int] = {}
    for work in compatible:
        for key in work.via:
            named[key] = named.get(key, 0) + 1
    for work in compatible:
        if not any(named[key] == 1 for key in work.via):
            continue
        if _conflicts(joined, work.keys):
            rejected.append(work.work_id)
            continue
        members.append(work.work_id)
        joined += work.keys
    members = sorted(set(members))
    if members:
        work_id = members[0]
    else:
        anchor = sorted(key.key for key in keys if key.strong)
        work_id = stable_id("work", anchor[0] if anchor else document_id)
    return WorkPlan(
        work_id=work_id,
        absorbed=[item for item in members if item != work_id],
        keys=list(keys),
        rejected=sorted(rejected),
    )


class WorkIndex:
    """Works of documents in memory, decided as the graph decides them
    (GraphStore._write_work): the dry run of ``lctrend link-works`` and
    the reference for tests. A strong key names one work; a title may name
    several."""

    def __init__(self) -> None:
        self.key_works: Dict[str, Set[str]] = {}
        self.work_keys: Dict[str, List[WorkKey]] = {}
        self.document_work: Dict[str, str] = {}
        self.work_documents: Dict[str, Set[str]] = {}
        self.rejected = 0

    def add(self, document_id: str, keys: Sequence[WorkKey]) -> WorkPlan:
        found: Dict[str, FoundWork] = {}

        def reach(work_id: str) -> FoundWork:
            return found.setdefault(
                work_id,
                FoundWork(work_id, list(self.work_keys.get(work_id, []))),
            )

        for key in keys:
            for work_id in sorted(self.key_works.get(key.key, ())):
                reach(work_id).via.add(key.key)
        if document_id in self.document_work:
            reach(self.document_work[document_id]).current = True
        plan = plan_work(document_id, keys, list(found.values()))
        work_id = plan.work_id
        self.work_keys.setdefault(work_id, [])
        self.work_documents.setdefault(work_id, set())
        for absorbed in plan.absorbed:
            for key in self.work_keys.pop(absorbed, []):
                works = self.key_works[key.key]
                works.discard(absorbed)
                works.add(work_id)
                if key not in self.work_keys[work_id]:
                    self.work_keys[work_id].append(key)
            for document in self.work_documents.pop(absorbed, set()):
                self.document_work[document] = work_id
                self.work_documents[work_id].add(document)
        for key in plan.keys:
            works = self.key_works.setdefault(key.key, set())
            if key.strong and works - {work_id}:
                continue
            works.add(work_id)
            if key not in self.work_keys[work_id]:
                self.work_keys[work_id].append(key)
        previous = self.document_work.get(document_id)
        if previous and previous != work_id:
            self.work_documents.get(previous, set()).discard(document_id)
        self.document_work[document_id] = work_id
        self.work_documents[work_id].add(document_id)
        self.rejected += len(plan.rejected)
        return plan

    def groups(self) -> List[Set[str]]:
        return [docs for docs in self.work_documents.values() if docs]



def _family_id(metadata_json: Iterable[Optional[str]]) -> Optional[str]:
    import json

    for value in metadata_json:
        try:
            metadata = json.loads(value) if value else {}
        except ValueError:
            continue
        if isinstance(metadata, dict) and metadata.get("family_id"):
            return str(metadata["family_id"])
    return None


def stored_document_keys(row: Mapping[str, Any]) -> List[WorkKey]:
    """Keys of a Document row read by GraphStore.read_document_identities."""
    family = _family_id(row.get("metadata_json") or [])
    return stored_work_keys(
        str(row.get("document_type") or ""),
        list(row.get("external_ids") or []),
        row.get("title"),
        {"family_id": family} if family else None,
    )


async def link_stored_works(store: Any, apply: bool = False) -> Dict[str, Any]:
    """Give every stored document its work; without ``apply`` only report
    what the works would be. A second run changes nothing."""
    from ..core.aio import resolve

    rows = sorted(
        await resolve(store.read_document_identities()),
        key=lambda row: str(row["document_id"]),
    )
    index = WorkIndex()
    keyed = []
    for row in rows:
        keys = stored_document_keys(row)
        index.add(str(row["document_id"]), keys)
        keyed.append((str(row["document_id"]), keys))
    groups = sorted(index.groups(), key=lambda docs: (-len(docs), min(docs)))
    titles = {str(row["document_id"]): row.get("title") for row in rows}
    summary = {
        "apply": apply,
        "documents": len(rows),
        "works": len(groups),
        "documents_in_shared_works": sum(
            len(docs) for docs in groups if len(docs) > 1
        ),
        "title_matches_kept_apart": index.rejected,
        "largest_works": [
            {
                "documents": len(docs),
                "titles": sorted(
                    {str(titles.get(doc) or doc) for doc in docs}
                )[:3],
            }
            for docs in groups[:10]
            if len(docs) > 1
        ],
    }
    if apply:
        # The key constraint is what serializes concurrent writers.
        ensure = getattr(store, "ensure_schema", None)
        if ensure is not None:
            await resolve(ensure())
        await resolve(store.write_works(keyed))
    return summary
