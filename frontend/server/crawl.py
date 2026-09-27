"""Resumable thematic discovery and ingestion ledger, separate from Neo4j.

SQLite stores queue positions, material identities and processing statuses.
Extraction and the domain graph remain in the existing parser and Neo4j.
"""

from __future__ import annotations

import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, RLock
from time import sleep
from uuid import uuid4

from lctrend.core import aio
from lctrend.core.config import load_catalog

from .jobs import _error, default_workers


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


async def _hydrate(item):
    from lctrend.ingest.connectors import fetch_github, fetch_pypi

    if item["source"] == "github":
        payload = item.get("payload")
        if (
            isinstance(payload, dict)
            and "repository" in payload
            and "commit" in payload
        ):
            return payload
        return await fetch_github(item["source_id"], os.getenv("GITHUB_TOKEN"))
    if item["source"] == "pypi":
        return item.get("payload") or await fetch_pypi(item["source_id"])
    return item["payload"]


class CrawlManager:
    def __init__(
        self,
        job_manager,
        directory=None,
        *,
        discoverers=None,
        hydrator=None,
        pypi_discoverer=None,
        domains=None,
        batch_size=25,
        processed_reader=None,
    ):
        from lctrend.ingest.discovery import (
            discover_github,
            discover_openalex,
            discover_pypi_from_github_payload,
        )

        if not 1 <= batch_size <= 100:
            raise ValueError("batch_size must be 1..100")
        self.job_manager = job_manager
        self.directory = Path(
            directory or Path(job_manager.directory).parent / "crawl"
        ).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._db = sqlite3.connect(
            self.directory / "ledger.sqlite",
            check_same_thread=False,
            timeout=30,
        )
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS crawls (
                crawl_id TEXT PRIMARY KEY, topic TEXT NOT NULL,
                domains_json TEXT NOT NULL,
                status TEXT NOT NULL, stage TEXT NOT NULL, error_json TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS streams (
                crawl_id TEXT NOT NULL REFERENCES
                crawls(crawl_id), domain TEXT NOT NULL,
                query TEXT NOT NULL, source TEXT NOT NULL, cursor TEXT,
                status TEXT NOT NULL DEFAULT 'pending', pages
                INTEGER NOT NULL DEFAULT 0,
                total INTEGER, complete INTEGER NOT NULL
                DEFAULT 0, limitations_json TEXT NOT NULL
                DEFAULT '[]',
                error_json TEXT, PRIMARY KEY (crawl_id,domain,query,source)
            );
            CREATE TABLE IF NOT EXISTS materials (
                material_id TEXT PRIMARY KEY, canonical_id TEXT
                UNIQUE NOT NULL,
                title TEXT NOT NULL, source TEXT NOT NULL,
                source_id TEXT NOT NULL,
                url TEXT, payload_json TEXT, hydrated INTEGER
                NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'pending', job_id
                TEXT, doc_id TEXT,
                error_json TEXT, updated_at TEXT NOT NULL, claimed_by TEXT,
                stage TEXT, llm_status TEXT, gliner_status TEXT
            );
            CREATE TABLE IF NOT EXISTS links (
                crawl_id TEXT NOT NULL REFERENCES crawls(crawl_id),
                material_id TEXT NOT NULL REFERENCES materials(material_id),
                domain TEXT NOT NULL, source TEXT NOT NULL,
                duplicate INTEGER NOT NULL,
                PRIMARY KEY (crawl_id,material_id,domain,source)
            );
            CREATE INDEX IF NOT EXISTS material_status ON
            materials(status,updated_at);
            CREATE INDEX IF NOT EXISTS crawl_links ON
            links(crawl_id,source,material_id);
        """)
        existing_columns = {
            row[1] for row in self._db.execute("PRAGMA table_info(materials)")
        }
        for column in ["claimed_by", "stage", "llm_status", "gliner_status"]:
            if column not in existing_columns:
                self._db.execute(
                    f"ALTER TABLE materials ADD COLUMN {column} TEXT"
                )
        crawl_columns = {
            row[1] for row in self._db.execute("PRAGMA table_info(crawls)")
        }
        if "max_per_source" not in crawl_columns:
            # Materials per direction and search source; NULL is unlimited.
            self._db.execute(
                "ALTER TABLE crawls ADD COLUMN max_per_source INTEGER"
            )
        self._db.commit()
        self._discoverers = (
            discoverers
            if discoverers is not None
            else {"openalex": discover_openalex, "github": discover_github}
        )
        self._hydrator = hydrator or _hydrate
        self._pypi_discoverer = (
            pypi_discoverer or discover_pypi_from_github_payload
        )
        if domains is None:
            catalog = load_catalog("sources")
            selected = catalog.get("crawl_directions")
            domains = [
                domain
                for domain in catalog["domains"]
                if selected is None or domain["name"] in selected
            ]
        self._domains = domains
        self._batch_size = batch_size
        self._processed_reader = processed_reader or self._read_processed
        self._seeded = False
        self._paused = {}
        self._children = {}
        self._futures = {}
        self._closed = False
        self._recover()
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="thematic-crawl"
        )

    def _await(self, function, *args):
        """Run a sync or async source/graph call from the crawl thread.

        Network calls share the job manager's event loop.
        """
        coroutine = aio.call(function, *args)
        runner = getattr(self.job_manager, "run", None)
        return runner(coroutine) if runner else aio.run_sync(coroutine)

    def _read_processed(self):
        factory = getattr(self.job_manager, "_store_factory", None)
        if factory is None:
            return []

        async def collect():
            store = await aio.call(factory)
            try:
                verify = getattr(store, "verify_connectivity", None)
                if verify is not None:
                    await aio.call(verify)
                materials = store.processed_materials()
                if hasattr(materials, "__aiter__"):
                    return [item async for item in materials]
                return list(materials)
            finally:
                await aio.call(store.close)

        return self._await(collect)

    def append_seed(self, materials):
        from lctrend.ingest.discovery import material_identity

        with self._lock, self._db:
            for item in materials:
                if item.get("status") not in {"parsed", "partial"}:
                    continue
                canonical = item.get("canonical_id") or material_identity(
                    item["source"], item["source_id"]
                )
                self._db.execute(
                    """INSERT INTO
materials(material_id,canonical_id,title,source,source_id,url,status,updated_at)
                    VALUES (?,?,?,?,?,?,?,?) ON
                    CONFLICT(canonical_id) DO UPDATE SET
                    status=excluded.status,
                    title=CASE WHEN excluded.title='' THEN
                    materials.title ELSE excluded.title
                    END,
                    url=COALESCE(excluded.url,materials.url),updated_at=excluded.updated_at
                    WHERE materials.status='pending'""",
                    (
                        uuid4().hex,
                        canonical,
                        item.get("title") or item["source_id"],
                        item["source"],
                        item["source_id"],
                        item.get("url"),
                        item["status"],
                        _now(),
                    ),
                )

    def _ensure_seeded(self):
        if not self._seeded:
            self.append_seed(self._processed_reader())
            self._seeded = True

    def _recover(self):
        with self._lock, self._db:
            self._db.execute(
                (
                    "UPDATE crawls SET "
                    "status='paused',stage='paused',updated_at=? WHERE status "
                    "IN ('queued','running','pausing')"
                ),
                (_now(),),
            )
            records = self._db.execute(
                (
                    "SELECT * FROM materials WHERE status='processing' OR "
                    "claimed_by IS NOT NULL"
                )
            ).fetchall()
            for record in records:
                status, error = "pending", None
                if record["job_id"] and record["doc_id"]:
                    try:
                        job = self.job_manager.get_job(record["job_id"])
                        doc = next(
                            item
                            for item in job["documents"]
                            if item["doc_id"] == record["doc_id"]
                        )
                        status = {
                            "succeeded": "parsed",
                            "partial": "partial",
                            "failed": "failed",
                        }.get(doc["status"], "pending")
                        error = doc.get("error")
                    except (KeyError, StopIteration):
                        pass
                self._db.execute(
                    (
                        "UPDATE materials SET "
                        "status=?,error_json=?,updated_at=?,claimed_by=NULL "
                        "WHERE material_id=?"
                    ),
                    (
                        status,
                        _json(error) if error else None,
                        _now(),
                        record["material_id"],
                    ),
                )

    def _known(self, crawl_id):
        row = self._db.execute(
            "SELECT * FROM crawls WHERE crawl_id=?", (crawl_id,)
        ).fetchone()
        if row is None:
            raise KeyError(crawl_id)
        return row

    def create(self, topic="", limit=None):
        if not isinstance(topic, str) or len(topic) > 1000:
            raise ValueError(
                "topic must be a string of at most 1000 characters"
            )
        if limit is not None and (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 10000
        ):
            raise ValueError("limit must be 1..10000")
        topic = topic.strip()
        domains = [{"name": topic, "aliases": []}] if topic else self._domains
        if not domains:
            raise ValueError("No configured domains")
        with self._lock, self._db:
            if self._closed:
                raise RuntimeError("Crawl manager is closed")
            crawl_id, created = uuid4().hex, _now()
            self._db.execute(
                "INSERT INTO crawls(crawl_id,topic,domains_json,status,"
                "stage,error_json,created_at,updated_at,max_per_source) "
                "VALUES (?,?,?,'queued','queued',NULL,?,?,?)",
                (crawl_id, topic, _json(domains), created, created, limit),
            )
            for domain in domains:
                seen = set()
                queries = [domain["name"], *domain.get(
                    "search_aliases", domain.get("aliases", [])
                )]
                for query in queries:
                    if query.strip().casefold() in seen:
                        continue
                    seen.add(query.strip().casefold())
                    for source in self._discoverers:
                        self._db.execute(
                            (
                                "INSERT INTO "
                                "streams(crawl_id,domain,query,source,cursor) "
                                "VALUES (?,?,?,?,?)"
                            ),
                            (
                                crawl_id,
                                domain["name"],
                                query.strip(),
                                source,
                                "*",
                            ),
                        )
            self._paused[crawl_id] = Event()
            self._futures[crawl_id] = self._pool.submit(self._run, crawl_id)
        return self.get_crawl(crawl_id)

    def _counts(self, crawl_id, source=None):
        where = "l.crawl_id=?" + (" AND l.source=?" if source else "")
        params = (crawl_id, source) if source else (crawl_id,)
        rows = self._db.execute(
            "SELECT status,COUNT(*) n FROM materials m JOIN "
            "(SELECT DISTINCT material_id FROM links l "
            f"WHERE {where}) x ON x.material_id=m.material_id GROUP BY status",
            params,
        ).fetchall()
        counts = {
            status: 0
            for status in [
                "parsed",
                "pending",
                "processing",
                "partial",
                "failed",
            ]
        }
        counts.update({row["status"]: row["n"] for row in rows})
        counts["discovered"] = sum(counts.values())
        counts["duplicates"] = self._db.execute(
            f"SELECT COALESCE(SUM(duplicate),0) FROM links l WHERE {where}",
            params,
        ).fetchone()[0]
        return counts

    def get_crawl(self, crawl_id):
        with self._lock:
            row = self._known(crawl_id)
            sources = []
            source_names = list(
                dict.fromkeys(
                    [*load_catalog("sources")["platforms"], *self._discoverers]
                )
            )
            for source in source_names:
                streams = self._db.execute(
                    "SELECT * FROM streams WHERE crawl_id=? AND source=?",
                    (crawl_id, source),
                ).fetchall()
                counts = self._counts(crawl_id, source)
                limitations = []
                for stream in streams:
                    for item in json.loads(stream["limitations_json"]):
                        if item not in limitations:
                            limitations.append(item)
                if source == "pypi":
                    limitations.append(
                        {
                            "code": "linked_packages_only",
                            "message": (
                                "PyPI: пакеты из ссылок в найденных "
                                "GitHub-репозиториях; тематического "
                                "поискового API нет."
                            ),
                        }
                    )
                    status = "linked"
                    complete = False
                    total = None
                elif not streams:
                    limitations.append(
                        {
                            "code": "discovery_unavailable",
                            "message": (
                                "Для этого источника пока нет "
                                "подключённого обхода."
                            ),
                        }
                    )
                    status, complete, total = "unsupported", False, None
                else:
                    complete = all(stream["complete"] for stream in streams)
                    if any(stream["status"] == "failed" for stream in streams):
                        status = "failed"
                    elif any(
                        stream["status"] == "pending" for stream in streams
                    ):
                        status = (
                            "discovering"
                            if row["status"] == "running"
                            else "pending"
                        )
                    elif any(
                        item.get("code") == "user_limit"
                        for item in limitations
                    ):
                        status = "capped"
                    else:
                        status = "complete" if complete else "limited"
                    # Overlapping domain/alias queries cannot have their totals
                    # added without double counting. One stream has a total.
                    total = streams[0]["total"] if len(streams) == 1 else None
                source_errors = [
                    json.loads(stream["error_json"])
                    for stream in streams
                    if stream["error_json"]
                ]
                if source == "github":
                    metadata_failure = self._db.execute(
                        "SELECT m.error_json FROM materials m "
                        "WHERE m.source='github' AND m.hydrated=-1 "
                        "AND EXISTS(SELECT 1 FROM links l WHERE l.crawl_id=? "
                        "AND l.material_id=m.material_id) LIMIT 1",
                        (crawl_id,),
                    ).fetchone()
                    if metadata_failure is not None:
                        status, complete = "failed", False
                        source_errors.append(
                            json.loads(metadata_failure["error_json"])
                            if metadata_failure["error_json"]
                            else {
                                "code": "metadata_unavailable",
                                "message": (
                                    "Не удалось получить ссылки пакетов."
                                ),
                            }
                        )
                sources.append(
                    {
                        "source": source,
                        "total": total,
                        "discovered": counts["discovered"],
                        "parsed": counts["parsed"],
                        "pending": counts["pending"],
                        "processing": counts["processing"],
                        "partial": counts["partial"],
                        "failed": counts["failed"],
                        "status": status,
                        "complete": complete,
                        "limitations": limitations,
                        "error": source_errors[0] if source_errors else None,
                    }
                )
            return {
                "crawl_id": crawl_id,
                "topic": row["topic"],
                "status": row["status"],
                "stage": row["stage"],
                "error": json.loads(row["error_json"])
                if row["error_json"]
                else None,
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "domains": [
                    domain["name"]
                    for domain in json.loads(row["domains_json"])
                ],
                "counts": self._counts(crawl_id),
                "sources": sources,
                "limit": row["max_per_source"],
                "directions": self._direction_counts(crawl_id),
            }

    def _direction_counts(self, crawl_id):
        """Discovered materials and their states per direction and source."""
        rows = self._db.execute(
            "SELECT l.domain,l.source,m.status,COUNT(*) n FROM links l "
            "JOIN materials m ON m.material_id=l.material_id "
            "WHERE l.crawl_id=? GROUP BY l.domain,l.source,m.status",
            (crawl_id,),
        ).fetchall()
        directions = {
            domain["name"]: {}
            for domain in json.loads(self._known(crawl_id)["domains_json"])
        }
        for row in rows:
            counts = directions.setdefault(row["domain"], {}).setdefault(
                row["source"], {"discovered": 0}
            )
            counts[row["status"]] = counts.get(row["status"], 0) + row["n"]
            counts["discovered"] += row["n"]
        return [
            {"name": name, "sources": sources}
            for name, sources in directions.items()
        ]

    def list_crawls(self):
        with self._lock:
            ids = [
                row[0]
                for row in self._db.execute(
                    (
                        "SELECT crawl_id FROM crawls ORDER BY created_at DESC "
                        "LIMIT 100"
                    )
                )
            ]
        return [self.get_crawl(crawl_id) for crawl_id in ids]

    def list_materials(self, crawl_id, status=None, limit=100, offset=0):
        if status is not None and status not in {
            "parsed",
            "pending",
            "processing",
            "partial",
            "failed",
        }:
            raise ValueError("Invalid material status")
        if (
            not isinstance(limit, int)
            or not 1 <= limit <= 100
            or not isinstance(offset, int)
            or offset < 0
        ):
            raise ValueError("Invalid pagination")
        with self._lock:
            self._known(crawl_id)
            where = (
                "EXISTS(SELECT 1 FROM links l WHERE l.crawl_id=? "
                "AND l.material_id=m.material_id)"
            )
            params = [crawl_id]
            if status:
                where += " AND m.status=?"
                params.append(status)
            total = self._db.execute(
                f"SELECT COUNT(*) FROM materials m WHERE {where}", params
            ).fetchone()[0]
            rows = self._db.execute(
                "SELECT material_id,title,source,canonical_id,status,job_id,"
                "doc_id,url,error_json,stage,llm_status,gliner_status "
                f"FROM materials m WHERE {where} "
                "ORDER BY updated_at DESC,material_id LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
            items = []
            for row in rows:
                item = dict(row)
                error_json = item.pop("error_json")
                item["error"] = json.loads(error_json) if error_json else None
                items.append(item)
            return {
                "items": items,
                "total": total,
                "limit": limit,
                "offset": offset,
            }

    def _set_state(self, crawl_id, status, stage=None, error=None):
        with self._lock, self._db:
            if (
                status == "running"
                and self._known(crawl_id)["status"] == "pausing"
            ):
                status = "pausing"
            self._db.execute(
                (
                    "UPDATE crawls SET "
                    "status=?,stage=?,error_json=?,updated_at=? WHERE "
                    "crawl_id=?"
                ),
                (
                    status,
                    stage or status,
                    _json(error) if error else None,
                    _now(),
                    crawl_id,
                ),
            )

    def pause(self, crawl_id):
        with self._lock:
            row = self._known(crawl_id)
            if row["status"] in {"queued", "running", "pausing"}:
                event = self._paused.setdefault(crawl_id, Event())
                event.set()
                future = self._futures.get(crawl_id)
                if future is not None and future.cancel():
                    self._set_state(crawl_id, "paused")
                else:
                    self._set_state(crawl_id, "pausing")
                    if crawl_id in self._children:
                        self.job_manager.cancel_job(self._children[crawl_id])
        return self.get_crawl(crawl_id)

    def resume(self, crawl_id):
        with self._lock, self._db:
            row = self._known(crawl_id)
            if self._closed:
                raise RuntimeError("Crawl manager is closed")
            if row["status"] in {"queued", "running", "pausing"}:
                return self.get_crawl(crawl_id)
            # Failed discovery is retried only by this explicit action. Failed
            # material extraction stays failed until retry_failed is requested.
            self._db.execute(
                (
                    "UPDATE streams SET status='pending',error_json=NULL "
                    "WHERE "
                    "crawl_id=? AND status='failed'"
                ),
                (crawl_id,),
            )
            self._paused[crawl_id] = Event()
            self._db.execute(
                "UPDATE materials SET hydrated=0,error_json=NULL "
                "WHERE hydrated=-1 AND material_id IN "
                "(SELECT material_id FROM links WHERE crawl_id=?)",
                (crawl_id,),
            )
            self._set_state(crawl_id, "queued")
            self._futures[crawl_id] = self._pool.submit(self._run, crawl_id)
        return self.get_crawl(crawl_id)

    def retry_failed(self, crawl_id):
        with self._lock, self._db:
            row = self._known(crawl_id)
            if row["status"] in {"queued", "running", "pausing"}:
                raise ValueError(
                    "Pause the crawl before retrying failed materials"
                )
            self._db.execute(
                (
                    "UPDATE materials SET "
                    "status='pending',error_json=NULL,updated_at=? WHERE "
                    "status='failed' AND material_id IN (SELECT material_id "
                    "FROM links WHERE crawl_id=?)"
                ),
                (_now(), crawl_id),
            )
        return self.resume(crawl_id)

    def _enqueue(self, crawl_id, domain, items):
        from lctrend.ingest.discovery import material_identity

        for item in items:
            canonical = item.get("canonical_id") or material_identity(
                item["source"], item["source_id"], item.get("payload")
            )
            existing = self._db.execute(
                "SELECT * FROM materials WHERE canonical_id=?", (canonical,)
            ).fetchone()
            if existing is None:
                material_id = uuid4().hex
                payload = item.get("payload")
                self._db.execute(
                    (
                        "INSERT INTO "
                        "materials(material_id,canonical_id,title,source,source_id,url,payload_json,updated_at)"
                        " VALUES (?,?,?,?,?,?,?,?)"
                    ),
                    (
                        material_id,
                        canonical,
                        str(item.get("title") or item["source_id"]),
                        item["source"],
                        item["source_id"],
                        item.get("url"),
                        _json(payload) if payload is not None else None,
                        _now(),
                    ),
                )
            else:
                material_id = existing["material_id"]
            self._db.execute(
                "INSERT OR IGNORE INTO links VALUES (?,?,?,?,?)",
                (
                    crawl_id,
                    material_id,
                    domain,
                    item["source"],
                    int(existing is not None),
                ),
            )
            if (
                existing is not None
                and item["source"] == "github"
                and existing["hydrated"] == 1
                and existing["payload_json"]
            ):
                self._enqueue(
                    crawl_id,
                    domain,
                    self._pypi_discoverer(
                        json.loads(existing["payload_json"])
                    ),
                )

    def _discover_page(self, crawl_id, source):
        with self._lock:
            stream = self._db.execute(
                (
                    "SELECT * FROM streams WHERE crawl_id=? AND source=? AND "
                    "status='pending' ORDER BY pages,domain,query LIMIT 1"
                ),
                (crawl_id, source),
            ).fetchone()
        if stream is None:
            return False
        remaining = self._remaining(crawl_id, stream["domain"], source)
        if remaining == 0:
            with self._lock, self._db:
                self._close_capped(crawl_id, stream["domain"], source)
            return True
        try:
            page = self._await(
                self._discoverers[source], stream["query"], stream["cursor"]
            )
            items = page["items"]
            cursor = page.get("next_cursor")
            if cursor is not None and str(cursor) == stream["cursor"]:
                raise ValueError("Repeated discovery cursor")
            if remaining is not None:
                items = items[:remaining]
            with self._lock, self._db:
                self._enqueue(crawl_id, stream["domain"], items)
                if self._remaining(crawl_id, stream["domain"], source) == 0:
                    self._close_capped(crawl_id, stream["domain"], source)
                    return True
                limitations = json.loads(stream["limitations_json"])
                for limitation in page.get("limitations", []):
                    if limitation not in limitations:
                        limitations.append(limitation)
                self._db.execute(
                    (
                        "UPDATE streams SET "
                        "cursor=?,status=?,pages=pages+1,total=?,complete=?,limitations_json=?,error_json=NULL"
                        " WHERE crawl_id=? AND domain=? AND query=? AND "
                        "source=?"
                    ),
                    (
                        str(cursor) if cursor is not None else None,
                        "pending" if cursor is not None else "done",
                        page.get("total"),
                        int(page.get("complete", False) and not limitations),
                        _json(limitations),
                        crawl_id,
                        stream["domain"],
                        stream["query"],
                        source,
                    ),
                )
        except Exception as exc:
            error = _error(exc, "discovery")
            error["message"] = (
                f"Не удалось получить страницу источника {source}. "
                "Другие источники продолжают обрабатываться."
            )
            with self._lock, self._db:
                self._db.execute(
                    (
                        "UPDATE streams SET status='failed',error_json=? "
                        "WHERE "
                        "crawl_id=? AND domain=? AND query=? AND source=?"
                    ),
                    (
                        _json(error),
                        crawl_id,
                        stream["domain"],
                        stream["query"],
                        source,
                    ),
                )
        return True

    def _remaining(self, crawl_id, domain, source):
        """Materials still allowed for a direction/source; None: no limit.

        Already processed duplicates count too: they are part of the
        direction's coverage even though they are not analysed again.
        """
        with self._lock:
            limit = self._known(crawl_id)["max_per_source"]
            if limit is None:
                return None
            used = self._db.execute(
                "SELECT COUNT(*) FROM links WHERE crawl_id=? AND domain=? "
                "AND source=?",
                (crawl_id, domain, source),
            ).fetchone()[0]
        return max(limit - used, 0)

    def _close_capped(self, crawl_id, domain, source):
        limit = self._known(crawl_id)["max_per_source"]
        limitation = {
            "code": "user_limit",
            "limit": limit,
            "message": (
                f"Достигнут лимит запуска: {limit} материалов на направление."
            ),
        }
        for stream in self._db.execute(
            "SELECT * FROM streams WHERE crawl_id=? AND domain=? AND "
            "source=? AND status='pending'",
            (crawl_id, domain, source),
        ).fetchall():
            limitations = json.loads(stream["limitations_json"])
            if limitation not in limitations:
                limitations.append(limitation)
            self._db.execute(
                "UPDATE streams SET status='done',complete=0,"
                "limitations_json=? WHERE crawl_id=? AND domain=? AND "
                "query=? AND source=?",
                (
                    _json(limitations),
                    crawl_id,
                    domain,
                    stream["query"],
                    source,
                ),
            )

    def _process_pending(self, crawl_id):
        with self._lock, self._db:
            first = self._db.execute(
                (
                    "SELECT m.source FROM materials m WHERE status='pending' "
                    "AND claimed_by IS NULL AND EXISTS(SELECT 1 FROM links l "
                    "WHERE l.crawl_id=? AND l.material_id=m.material_id) "
                    "ORDER "
                    "BY updated_at LIMIT 1"
                ),
                (crawl_id,),
            ).fetchone()
            if first is None:
                return False
            candidates = self._db.execute(
                (
                    "SELECT m.* FROM materials m WHERE status='pending' AND "
                    "claimed_by IS NULL AND source=? AND EXISTS(SELECT 1 FROM "
                    "links l WHERE l.crawl_id=? AND "
                    "l.material_id=m.material_id) ORDER BY updated_at LIMIT ?"
                ),
                (first["source"], crawl_id, self._batch_size),
            ).fetchall()
            records = []
            for record in candidates:
                claimed = self._db.execute(
                    (
                        "UPDATE materials SET claimed_by=? "
                        "WHERE material_id=? "
                        "AND status='pending' AND claimed_by IS NULL"
                    ),
                    (crawl_id, record["material_id"]),
                )
                if claimed.rowcount:
                    records.append(record)
        ready, payloads, cached_results = [], [], []
        for record in records:
            if self._paused[crawl_id].is_set():
                self._material_status(record["material_id"], "pending")
                continue
            item = dict(record)
            item["payload"] = (
                json.loads(record["payload_json"])
                if record["payload_json"]
                else None
            )
            try:
                cached = self._cached_publication(record)
                with self._lock, self._db:
                    self._db.execute(
                        (
                            "UPDATE materials SET stage='hydration' WHERE "
                            "material_id=?"
                        ),
                        (record["material_id"],),
                    )
                self._set_state(
                    crawl_id, "running", f"hydration:{record['source']}"
                )
                payload = (
                    self._await(self._hydrator, item)
                    if cached is None and not record["hydrated"]
                    else item["payload"]
                )
                with self._lock, self._db:
                    self._db.execute(
                        (
                            "UPDATE materials SET payload_json=?,hydrated=1 "
                            "WHERE material_id=?"
                        ),
                        (_json(payload), record["material_id"]),
                    )
                    if record["source"] == "github":
                        domains = self._db.execute(
                            (
                                "SELECT DISTINCT domain FROM links WHERE "
                                "crawl_id=? AND material_id=?"
                            ),
                            (crawl_id, record["material_id"]),
                        ).fetchall()
                        refs = self._pypi_discoverer(payload)
                        for domain in domains:
                            self._enqueue(crawl_id, domain[0], refs)
                ready.append(record)
                payloads.append(payload)
                cached_results.append(cached)
            except Exception as exc:
                error = _error(exc, "parse")
                error["message"] = (
                    "Не удалось загрузить содержимое материала из источника."
                )
                self._material_status(record["material_id"], "failed", error)
        if not ready:
            return True
        if self._paused[crawl_id].is_set():
            for record in ready:
                self._material_status(record["material_id"], "pending")
            return True
        try:
            with self._lock:
                if self._paused[crawl_id].is_set():
                    for record in ready:
                        self._material_status(record["material_id"], "pending")
                    return True

                def register(job):
                    self._children[crawl_id] = job["job_id"]
                    with self._db:
                        for record, doc in zip(ready, job["documents"]):
                            self._db.execute(
                                "UPDATE materials SET job_id=?,doc_id=? "
                                "WHERE material_id=?",
                                (
                                    job["job_id"],
                                    doc["doc_id"],
                                    record["material_id"],
                                ),
                            )

                job = self.job_manager.create_payloads(
                    first["source"],
                    payloads,
                    direction=self._known(crawl_id)["topic"],
                    workers=default_workers(),
                    on_created=register,
                    cached_results=cached_results,
                )
            while True:
                job = self.job_manager.get_job(job["job_id"])
                self._sync_child(crawl_id, ready, job)
                if job["status"] not in {"queued", "running", "cancelling"}:
                    break
                if (
                    self._paused[crawl_id].is_set()
                    and job["status"] != "cancelling"
                ):
                    self.job_manager.cancel_job(job["job_id"])
                sleep(0.1)
            for record, doc in zip(ready, job["documents"]):
                status = {
                    "succeeded": "parsed",
                    "partial": "partial",
                    "failed": "failed",
                    "cancelled": "pending",
                }.get(doc["status"], "pending")
                error = doc.get("error") or (
                    job.get("error")
                    if status == "pending" and job["status"] == "failed"
                    else None
                )
                if job["status"] == "failed" and doc["status"] == "cancelled":
                    # Preflight failure is a failed attempt, not an endless
                    # pending/model-reconfiguration loop.
                    status = "failed"
                self._material_status(record["material_id"], status, error)
        except Exception as exc:
            for record in ready:
                self._material_status(
                    record["material_id"], "failed", _error(exc, "processing")
                )
        finally:
            with self._lock:
                self._children.pop(crawl_id, None)
        return True

    def _expand_known_github(self, crawl_id):
        """Fetch missing README/package links for graph-seeded repositories.

        Their successful extraction stays parsed; this does not call a model.
        A failed metadata fetch is retried only through explicit resume.
        """
        with self._lock:
            records = self._db.execute(
                "SELECT m.* FROM materials m WHERE m.source='github' "
                "AND m.status IN ('parsed','partial') AND m.hydrated=0 "
                "AND EXISTS(SELECT 1 FROM links l WHERE l.crawl_id=? "
                "AND l.material_id=m.material_id) LIMIT ?",
                (crawl_id, self._batch_size),
            ).fetchall()
        for record in records:
            if self._paused[crawl_id].is_set():
                break
            self._set_state(crawl_id, "running", "metadata:github")
            item = dict(record)
            item["payload"] = (
                json.loads(record["payload_json"])
                if record["payload_json"]
                else None
            )
            try:
                payload = self._await(self._hydrator, item)
                refs = self._pypi_discoverer(payload)
                with self._lock, self._db:
                    self._db.execute(
                        "UPDATE materials SET hydrated=1,payload_json=?,"
                        "error_json=NULL WHERE material_id=?",
                        (_json(payload), record["material_id"]),
                    )
                    domains = self._db.execute(
                        "SELECT DISTINCT domain FROM links WHERE crawl_id=? "
                        "AND material_id=?",
                        (crawl_id, record["material_id"]),
                    ).fetchall()
                    for domain in domains:
                        self._enqueue(crawl_id, domain[0], refs)
            except Exception as exc:
                error = _error(exc, "parse")
                error["message"] = (
                    "Материал уже обработан, но дополнительные ссылки "
                    "на пакеты пока не удалось получить."
                )
                with self._lock, self._db:
                    self._db.execute(
                        "UPDATE materials SET hydrated=-1,error_json=? "
                        "WHERE material_id=?",
                        (_json(error), record["material_id"]),
                    )
        return bool(records)

    def _material_status(self, material_id, status, error=None):
        with self._lock, self._db:
            self._db.execute(
                (
                    "UPDATE materials SET "
                    "status=?,error_json=?,updated_at=?,claimed_by=NULL WHERE "
                    "material_id=?"
                ),
                (status, _json(error) if error else None, _now(), material_id),
            )

    def _cached_publication(self, record):
        if not record["job_id"] or not record["doc_id"]:
            return None
        try:
            job = self.job_manager.get_job(record["job_id"])
            doc = next(
                item
                for item in job["documents"]
                if item["doc_id"] == record["doc_id"]
            )
            publication = (
                doc.get("stage") == "publication"
                or doc.get("interrupted_stage") == "publication"
            )
            if not publication or not doc.get("result_ready"):
                return None
            cached = self.job_manager.get_result(
                record["job_id"], record["doc_id"]
            )
            if cached["extraction"]["run"]["status"] in {
                "succeeded",
                "partial",
            }:
                return cached
        except (KeyError, StopIteration, OSError, ValueError):
            pass
        return None

    def _sync_child(self, crawl_id, records, job):
        with self._lock, self._db:
            for record, doc in zip(records, job["documents"]):
                status = {
                    "queued": "pending",
                    "running": "processing",
                    "succeeded": "parsed",
                    "partial": "partial",
                    "failed": "failed",
                    "cancelled": "pending",
                }.get(doc["status"], "pending")
                error = doc.get("error")
                if job["status"] == "failed" and doc["status"] == "cancelled":
                    status, error = "failed", job.get("error")
                claim = (
                    crawl_id
                    if doc["status"] in {"queued", "running"}
                    else None
                )
                error_json = _json(error) if error else None
                values = (
                    status,
                    doc.get("stage"),
                    doc.get("llm_status"),
                    doc.get("gliner_status"),
                    error_json,
                    claim,
                )
                current = self._db.execute(
                    (
                        "SELECT "
                        "status,stage,llm_status,gliner_status,error_json,claimed_by"
                        " FROM materials WHERE material_id=?"
                    ),
                    (record["material_id"],),
                ).fetchone()
                if tuple(current) != values:
                    self._db.execute(
                        (
                            "UPDATE materials SET "
                            "status=?,stage=?,llm_status=?,gliner_status=?,error_json=?,claimed_by=?,updated_at=?"
                            " WHERE material_id=?"
                        ),
                        (*values, _now(), record["material_id"]),
                    )
            running = next(
                (
                    doc
                    for doc in job["documents"]
                    if doc["status"] == "running"
                ),
                None,
            )
            stage = (
                running.get("stage")
                if running
                else job.get("stage", "processing")
            )
            self._db.execute(
                "UPDATE crawls SET stage=?,updated_at=? WHERE crawl_id=?",
                (stage or "processing", _now(), crawl_id),
            )

    def _run(self, crawl_id):
        try:
            if self._paused[crawl_id].is_set():
                self._set_state(crawl_id, "paused")
                return
            self._ensure_seeded()
            self._set_state(crawl_id, "running", "discovery")
            while not self._paused[crawl_id].is_set():
                found_page = False
                with self._lock:
                    pending = self._counts(crawl_id)["pending"]
                # Apply backpressure to discovery while its durable input queue
                # already contains several processing batches.
                if pending < self._batch_size * 4:
                    for source in self._discoverers:
                        if self._paused[crawl_id].is_set():
                            break
                        found_page = (
                            self._discover_page(crawl_id, source) or found_page
                        )
                if self._paused[crawl_id].is_set():
                    break
                metadata_expanded = self._expand_known_github(crawl_id)
                self._set_state(crawl_id, "running", "processing")
                processed = self._process_pending(crawl_id)
                if not found_page and not processed and not metadata_expanded:
                    break
                self._set_state(crawl_id, "running", "discovery")
            if self._paused[crawl_id].is_set():
                self._set_state(crawl_id, "paused")
            else:
                state = self.get_crawl(crawl_id)
                failed_sources = any(
                    source["status"] == "failed" for source in state["sources"]
                )
                error = (
                    {
                        "code": "source_failures",
                        "message": (
                            "Часть источников недоступна. Уже найденные "
                            "материалы обработаны; повтор обхода доступен "
                            "через продолжение."
                        ),
                    }
                    if failed_sources
                    else None
                )
                self._set_state(
                    crawl_id,
                    "failed" if failed_sources else "completed",
                    error=error,
                )
        except Exception as exc:
            self._set_state(
                crawl_id,
                "paused" if self._paused[crawl_id].is_set() else "failed",
                error=_error(exc, "processing"),
            )

    def close(self, wait=False):
        with self._lock:
            self._closed = True
            rows = self._db.execute(
                (
                    "SELECT crawl_id FROM crawls WHERE status IN "
                    "('queued','running','pausing')"
                )
            ).fetchall()
            for row in rows:
                self.pause(row[0])
        self._pool.shutdown(wait=wait, cancel_futures=True)
        if wait:
            with self._lock:
                self._db.close()

    shutdown = close
