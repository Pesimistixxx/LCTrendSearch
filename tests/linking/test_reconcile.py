"""New claims linked with the claims the graph already holds."""

import asyncio

from lctrend.linking.reconcile import (
    METHOD,
    claim_key,
    claim_links,
    reconcile,
    shared_contexts,
    slot_of_row,
)

SENSOR, ACME, TASK, US = "tech:sensor", "org:acme", "task:monitor", "c:US"
OTHER = "tech:probe"


def claim(
    assertion_id,
    work,
    predicate="developed_by",
    roles=None,
    polarity="affirmed",
    modality="reported",
    observed="2022-01-01",
    qualifiers=None,
):
    roles = roles or {"subject": SENSOR, "organization": ACME}
    item = {
        "assertion_id": assertion_id,
        "work_id": work,
        "predicate": predicate,
        "roles": roles,
        "kinds": {
            SENSOR: "Technology",
            OTHER: "Technology",
            ACME: "Company",
            TASK: "Task",
            US: "Country",
        },
        "polarity": polarity,
        "modality": modality,
        "observed_at": observed,
        "qualifiers": qualifiers or {},
    }
    item["claim_key"] = claim_key(predicate, roles, item["qualifiers"])
    return item


def test_the_slot_ignores_polarity_case_and_trl_but_not_conditions():
    roles = {"subject": SENSOR, "organization": ACME}
    assert claim_key("developed_by", roles) == claim_key(
        "developed_by", dict(reversed(list(roles.items())))
    )
    pilot = claim_key(
        "reports_maturity_stage", {"subject": SENSOR}, {"stage": "Pilot"}
    )
    assert pilot == claim_key(
        "reports_maturity_stage",
        {"subject": SENSOR},
        {"stage": "pilot", "trl": 6},
    )
    assert pilot != claim_key(
        "reports_maturity_stage", {"subject": SENSOR}, {"stage": "prototype"}
    )
    hot = claim_key(
        "solves_task", {"subject": SENSOR, "task": TASK}, {"condition": "80 C"}
    )
    assert hot != claim_key("solves_task", {"subject": SENSOR, "task": TASK})
    # Measurements and money need their own value comparison.
    assert claim_key("reported_measurement", {"subject": SENSOR}) is None
    assert claim_key("reported_investment", roles) is None


def test_other_works_corroborate_and_contradict_the_earliest_claim():
    claims = [
        claim("a1", "w1", observed="2020-01-01"),
        claim("a2", "w1", observed="2020-06-01"),  # same work: no link
        claim("b1", "w2", observed="2021-01-01"),
        claim("c1", "w3", polarity="negated", observed="2022-01-01"),
        # A plan does not contradict a fact, nor corroborate it.
        claim("d1", "w4", polarity="negated", modality="planned"),
    ]
    links = claim_links(claims)

    corroborates = {(r["source"], r["target"]) for r in links["CORROBORATES"]}
    assert corroborates == {("b1", "a1")}
    contradicts = {(r["source"], r["target"]) for r in links["CONTRADICTS"]}
    assert contradicts == {("c1", "a1"), ("c1", "b1")}
    # A link is dated by its later claim.
    assert links["CORROBORATES"][0]["observed_at"] == "2021-01-01"


def test_an_incremental_run_links_only_the_new_claims():
    claims = [
        claim("a1", "w1", observed="2020-01-01"),
        claim("b1", "w2", observed="2021-01-01"),
        claim("c1", "w3", observed="2022-01-01"),
    ]
    links = claim_links(claims, new_ids={"c1"})

    pairs = {(r["source"], r["target"]) for r in links["CORROBORATES"]}
    assert pairs == {("c1", "a1")}


def test_technologies_in_the_same_contexts_share_them():
    claims = [
        claim("a1", "w1"),
        claim(
            "a2",
            "w1",
            predicate="solves_task",
            roles={"subject": SENSOR, "task": TASK},
        ),
        claim(
            "b1",
            "w2",
            roles={"subject": OTHER, "organization": ACME},
            observed="2023-01-01",
        ),
        claim(
            "b2",
            "w2",
            predicate="solves_task",
            roles={"subject": OTHER, "task": TASK},
        ),
        # Countries alone are too common to relate two technologies.
        claim(
            "a3",
            "w1",
            predicate="developed_in",
            roles={"subject": SENSOR, "country": US},
        ),
        claim(
            "b3",
            "w2",
            predicate="developed_in",
            roles={"subject": OTHER, "country": US},
        ),
    ]
    kinds = claims[0]["kinds"]
    [pair] = shared_contexts(claims, kinds)

    assert (pair["left"], pair["right"]) == (OTHER, SENSOR)
    assert pair["shared_claims"] == 2
    assert pair["observed_at"] == "2022-01-01"


class Store:
    def __init__(self, rows):
        self.rows = rows
        self.writes = []
        self._database = None
        self._driver = self

    def session(self, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def run(self, query, **parameters):
        if "RETURN a.assertion_id" in query:
            return self.rows
        self.writes.append((query, parameters))
        return self

    def consume(self):
        return None

    async def execute_write(self, callback):
        return await callback(self)


def record(assertion_id, work, polarity="affirmed", observed="2021-01-01"):
    return {
        "assertion_id": assertion_id,
        "predicate": "developed_by",
        "polarity": polarity,
        "modality": "reported",
        "qualifiers_json": "{}",
        "observed_at": observed,
        "version_id": "v-" + assertion_id,
        "work_id": work,
        "roles": [
            ["SUBJECT", SENSOR, "Technology"],
            ["ORGANIZATION", ACME, "Company"],
        ],
    }


def test_a_rebuild_replaces_the_layer_and_writes_claim_keys():
    store = Store(
        [record("a1", "w1", observed="2020-01-01"), record("b1", "w2")]
    )
    summary = asyncio.run(reconcile(store))

    assert summary["corroborates"] == 1 and summary["slots"] == 1
    queries = [query for query, _ in store.writes]
    assert sum("DELETE r" in query for query in queries) == 3
    keys = next(p for q, p in store.writes if "SET a.claim_key" in q)
    assert {row["assertion_id"] for row in keys["rows"]} == {"a1", "b1"}
    link = next(
        p for q, p in store.writes if "CORROBORATES" in q and "MERGE" in q
    )
    assert link["method"] == METHOD
    assert link["rows"][0]["source"] == "b1"


def test_a_slot_read_from_roles_follows_a_merge():
    row = {
        "predicate": "developed_by",
        "roles": [["SUBJECT", SENSOR], ["ORGANIZATION", ACME]],
        "qualifiers_json": "{}",
    }
    assert slot_of_row(row) == claim_key(
        "developed_by", {"subject": SENSOR, "organization": ACME}
    )
    merged = {**row, "roles": [["SUBJECT", OTHER], ["ORGANIZATION", ACME]]}
    assert slot_of_row(merged) != slot_of_row(row)


def contexts_of(names, **options):
    kinds = {name: "Technology" for name in names} | {ACME: "Company"}
    claims = []
    for index, name in enumerate(names):
        item = claim(
            f"a{index}",
            f"w{index}",
            roles={"subject": name, "organization": ACME},
        )
        item["kinds"] = kinds
        claims.append(item)
    return shared_contexts(claims, kinds, **options)


def test_a_hub_context_relates_nothing():
    subjects = [f"tech:{index}" for index in range(5)]

    assert len(contexts_of(subjects)) == 10
    assert contexts_of(subjects, max_members=4) == []


def test_an_incremental_run_pairs_only_the_new_subjects():
    subjects = [f"tech:{index}" for index in range(5)]
    rows = contexts_of(subjects, subjects={"tech:3"})

    assert {(row["left"], row["right"]) for row in rows} == {
        ("tech:0", "tech:3"),
        ("tech:1", "tech:3"),
        ("tech:2", "tech:3"),
        ("tech:3", "tech:4"),
    }


class QueryLog(Store):
    def __init__(self, rows):
        super().__init__(rows)
        self.reads = []

    def run(self, query, **parameters):
        if "RETURN a.assertion_id" in query:
            self.reads.append((query, parameters))
        return super().run(query, **parameters)


def test_an_incremental_run_starts_from_labeled_concepts_without_countries():
    row = record("b1", "w2")
    row["roles"] = row["roles"] + [["COUNTRY", US, "Country"]]
    store = QueryLog([row])
    asyncio.run(reconcile(store, ["v-b1"]))

    fresh, pool = store.reads
    assert "IN $version_ids" in fresh[0]
    assert pool[0].startswith("CALL { MATCH (x:")
    assert pool[1]["concepts"] == {
        "Company": [ACME],
        "Technology": [SENSOR],
    }
