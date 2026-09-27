"""Validate complete resource catalogs at startup, before network/model work.

Overrides remain whole-file replacements. This module deliberately does not
validate individual load_catalog calls: callers sometimes load partial test
fixtures, and cross-catalog constraints require the complete resource set.
"""

from __future__ import annotations

import math
import re
from datetime import date
from typing import Any, Mapping, Optional

from .models import ConceptKind, DocumentType

CATALOG_NAMES = (
    "sources",
    "extraction",
    "pipeline",
    "resolver",
    "graph",
    "runtime",
    "llm",
    "llm_schema",
    "dataset",
    "taxonomy",
    "countries",
)
KINDS = {kind.value for kind in ConceptKind}
DOCUMENT_TYPES = {kind.value for kind in DocumentType}


def _fail(path: str, message: str) -> None:
    raise ValueError(f"Invalid resource {path}: {message}")


def _mapping(value: Any, path: str) -> Mapping:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    return value


def _required(value: Mapping, key: str, path: str) -> Any:
    if key not in value:
        _fail(f"{path}.{key}", "required field missing")
    return value[key]


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "must be a nonempty string")
    return value


def _number(value: Any, path: str, minimum=0, maximum=None, integer=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (integer and not isinstance(value, int))
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        _fail(path, "number outside the allowed range or wrong type")
    return value


def _bool(value: Any, path: str) -> None:
    if not isinstance(value, bool):
        _fail(path, "must be a boolean")


def _strings(value: Any, path: str, allowed=None, nonempty=False) -> list:
    if not isinstance(value, list) or (nonempty and not value):
        _fail(path, "must be a list of strings")
    for index, item in enumerate(value):
        _text(item, f"{path}[{index}]")
        if allowed is not None and item not in allowed:
            _fail(f"{path}[{index}]", f"unknown value {item!r}")
    if len(value) != len(set(value)):
        _fail(path, "duplicate entries")
    return value


def _regex(value: Any, path: str) -> None:
    _text(value, path)
    try:
        re.compile(value)
    except re.error as exc:
        _fail(path, f"invalid regular expression: {exc}")


def _patterns(value: Any, path: str) -> None:
    for key, pattern in _mapping(value, path).items():
        _regex(pattern, f"{path}.{key}")


def _identifier(value: Any, path: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", _text(value, path)):
        _fail(path, "invalid Cypher identifier")


def _sources(catalog: Mapping) -> None:
    domains = _required(catalog, "domains", "sources")
    if not isinstance(domains, list) or not domains:
        _fail("sources.domains", "must be a nonempty list")
    names = set()
    for index, value in enumerate(domains):
        path = f"sources.domains[{index}]"
        domain = _mapping(value, path)
        name = _text(_required(domain, "name", path), path + ".name")
        if name.casefold() in names:
            _fail(path + ".name", "duplicate domain name")
        names.add(name.casefold())
        if _required(domain, "parent_name", path) is not None:
            _text(domain["parent_name"], path + ".parent_name")
        _strings(
            _required(domain, "aliases", path),
            path + ".aliases",
            nonempty=True,
        )
        if "search_aliases" in domain:
            _strings(domain["search_aliases"], path + ".search_aliases")
    for name in _strings(
        catalog.get("crawl_directions", []), "sources.crawl_directions"
    ):
        if name.casefold() not in names:
            _fail("sources.crawl_directions", f"unknown domain {name!r}")
    platforms = _mapping(
        _required(catalog, "platforms", "sources"), "sources.platforms"
    )
    for name in ("openalex", "github", "pypi", "epo"):
        platform = _mapping(
            _required(platforms, name, "sources.platforms"),
            f"sources.platforms.{name}",
        )
        for key in ("name", "source_type", "source_family"):
            _text(
                _required(platform, key, f"sources.platforms.{name}"),
                f"sources.platforms.{name}.{key}",
            )
        _number(
            platform["reliability_tier"],
            f"sources.platforms.{name}.reliability_tier",
            1,
            integer=True,
        )
    http = _mapping(_required(catalog, "http", "sources"), "sources.http")
    _text(
        _required(http, "user_agent", "sources.http"),
        "sources.http.user_agent",
    )
    for key, minimum in (
        ("timeout_seconds", 0.001),
        ("max_attempts", 1),
        ("backoff_seconds", 0),
        ("max_backoff_seconds", 0),
        ("max_rate_limit_wait_seconds", 0),
    ):
        _number(
            _required(http, key, "sources.http"),
            "sources.http." + key,
            minimum,
            integer=key == "max_attempts",
        )
    if http["backoff_seconds"] > http["max_backoff_seconds"]:
        _fail("sources.http.backoff_seconds", "exceeds max_backoff_seconds")
    statuses = _required(http, "retry_statuses", "sources.http")
    if not isinstance(statuses, list):
        _fail("sources.http.retry_statuses", "must be a list")
    for status in statuses:
        _number(status, "sources.http.retry_statuses", 100, 599, integer=True)
    markdown = _mapping(
        _required(catalog, "markdown", "sources"), "sources.markdown"
    )
    size = _number(
        markdown["max_chars"], "sources.markdown.max_chars", 1, integer=True
    )
    overlap = _number(
        markdown["overlap_chars"],
        "sources.markdown.overlap_chars",
        0,
        integer=True,
    )
    if overlap >= size:
        _fail("sources.markdown.overlap_chars", "must be below max_chars")
    _patterns(
        catalog["organization_type_patterns"],
        "sources.organization_type_patterns",
    )
    local = _mapping(catalog["local_files"], "sources.local_files")
    _text(local["sidecar_suffix"], "sources.local_files.sidecar_suffix")
    _number(
        local["cyrillic_share_for_ru"],
        "sources.local_files.cyrillic_share_for_ru",
        0,
        1,
    )
    _strings(local["english_markers"], "sources.local_files.english_markers")


def _extraction(catalog: Mapping) -> None:
    _regex(catalog["sentence_pattern"], "extraction.sentence_pattern")
    ner = _mapping(catalog["ner"], "extraction.ner")
    _number(ner["threshold"], "extraction.ner.threshold", 0, 1)
    _bool(
        ner["composite_technologies_enabled"],
        "extraction.ner.composite_technologies_enabled",
    )
    _strings(
        ner["generic_technologies"], "extraction.ner.generic_technologies"
    )
    for label, kind in _mapping(
        ner["labels"], "extraction.ner.labels"
    ).items():
        _text(label, "extraction.ner.labels")
        if kind not in KINDS:
            _fail("extraction.ner.labels." + label, "unknown ConceptKind")
    for rule in ner["composite_technology_rules"]:
        for key in ("method", "application"):
            _regex(
                rule[key], "extraction.ner.composite_technology_rules." + key
            )
        _text(
            rule["canonical"],
            "extraction.ner.composite_technology_rules.canonical",
        )
    hybrid = _mapping(catalog["hybrid"], "extraction.hybrid")
    for key in ("hint_kinds", "candidate_kinds"):
        _strings(hybrid[key], "extraction.hybrid." + key, KINDS)
    for key in ("hint_min_score", "candidate_min_score"):
        _number(hybrid[key], "extraction.hybrid." + key, 0, 1)
    _number(
        hybrid["max_hints_per_packet"],
        "extraction.hybrid.max_hints_per_packet",
        1,
        integer=True,
    )
    _patterns(catalog["polarity"], "extraction.polarity")
    assertions = _mapping(catalog["assertions"], "extraction.assertions")
    _strings(
        assertions["subject_kinds"],
        "extraction.assertions.subject_kinds",
        KINDS,
    )
    for predicate, kinds in assertions["object_kinds"].items():
        _strings(
            kinds, "extraction.assertions.object_kinds." + predicate, KINDS
        )
    for rule in assertions["relation_rules"]:
        if rule["predicate"] not in assertions["object_kinds"]:
            _fail(
                "extraction.assertions.relation_rules",
                "predicate has no object kinds",
            )
        _regex(rule["pattern"], "extraction.assertions.relation_rules.pattern")
    economics = _mapping(catalog["economics"], "extraction.economics")
    for key in (
        "clause_split_pattern",
        "ownership_prefix",
        "ownership_suffix",
        "non_monetary_cost",
        "money_pattern",
    ):
        _regex(economics[key], "extraction.economics." + key)
    _patterns(economics["categories"], "extraction.economics.categories")
    for scale, value in economics["scales"].items():
        _number(value, "extraction.economics.scales." + scale, 1)
    for token, currency in economics["currencies"].items():
        _text(token, "extraction.economics.currencies")
        if not re.fullmatch(r"[A-Z]{3}", str(currency)):
            _fail(
                "extraction.economics.currencies." + token,
                "invalid currency code",
            )


def _pipeline(catalog: Mapping) -> None:
    # Lazy import keeps config loading independent from this validator.
    from ..llm.context import PipelineSettings

    try:
        PipelineSettings.model_validate(catalog["settings"])
    except (ValueError, KeyError) as exc:
        _fail("pipeline.settings", str(exc))
    for key in ("metadata_fields", "navigation_locator_fields"):
        _strings(catalog[key], "pipeline." + key)
    _patterns(catalog["section_roles"], "pipeline.section_roles")
    context = _mapping(catalog["graph_context"], "pipeline.graph_context")
    _bool(context["enabled"], "pipeline.graph_context.enabled")
    for key in (
        "max_documents",
        "max_chunks",
        "max_chars",
        "max_search_chars",
    ):
        _number(context[key], "pipeline.graph_context." + key, 1, integer=True)
    for extension, spec in catalog["file_formats"].items():
        if not extension.startswith("."):
            _fail(
                "pipeline.file_formats." + extension,
                "extension must start with a dot",
            )
        if spec["adapter"] not in {"text", "markdown", "docx", "html", "pdf"}:
            _fail(
                "pipeline.file_formats." + extension,
                "adapter is not implemented",
            )
        _text(
            spec["media_type"],
            "pipeline.file_formats." + extension + ".media_type",
        )
    for key, value in catalog["file_limits"].items():
        _number(
            value,
            "pipeline.file_limits." + key,
            1,
            integer=key != "pdf_timeout_seconds",
        )
    fulltext = catalog["openalex_fulltext"]
    _number(
        fulltext["max_candidates"],
        "pipeline.openalex_fulltext.max_candidates",
        1,
        integer=True,
    )
    _strings(
        fulltext["skipped_labels"], "pipeline.openalex_fulltext.skipped_labels"
    )
    _regex(
        fulltext["bibliography_heading"],
        "pipeline.openalex_fulltext.bibliography_heading",
    )
    for key, value in catalog["html"].items():
        _strings(value, "pipeline.html." + key)


def _contracts(catalogs: Mapping) -> None:
    schema, graph = catalogs["llm_schema"], catalogs["graph"]
    predicates = _mapping(schema["predicates"], "llm_schema.predicates")
    roles = _mapping(graph["assertion_roles"], "graph.assertion_roles")
    for pattern in ("number_pattern", "country_code_pattern"):
        _regex(schema[pattern], "llm_schema." + pattern)
    _strings(
        schema["measurement_predicates"],
        "llm_schema.measurement_predicates",
        set(predicates),
    )
    for predicate, category in schema.get("economic_predicates", {}).items():
        if predicate not in predicates:
            _fail(
                "llm_schema.economic_predicates." + predicate,
                "unknown predicate",
            )
        _text(category, "llm_schema.economic_predicates." + predicate)
    for code, aliases in schema.get("currency_aliases", {}).items():
        if code not in {"$", "¥"} and not re.fullmatch(r"[A-Z]{3}", code):
            _fail(
                "llm_schema.currency_aliases." + code, "invalid currency code"
            )
        _strings(aliases, "llm_schema.currency_aliases." + code, nonempty=True)
    for role, relationship in roles.items():
        _text(role, "graph.assertion_roles")
        _identifier(relationship, "graph.assertion_roles." + role)
    for name, rule in predicates.items():
        path = "llm_schema.predicates." + name
        allowed = _strings(
            rule["allowed_roles"], path + ".allowed_roles", set(roles)
        )
        _strings(
            rule["required_roles"], path + ".required_roles", set(allowed)
        )
        for role, kinds in rule["role_types"].items():
            if role not in allowed:
                _fail(path + ".role_types", "role not in allowed_roles")
            _strings(kinds, path + ".role_types." + role, KINDS, nonempty=True)
        if set(allowed) != set(rule["role_types"]):
            _fail(
                path + ".role_types",
                "every allowed role needs a type contract",
            )
        if "value_contract" in rule:
            contract = _mapping(
                rule["value_contract"], path + ".value_contract"
            )
            _strings(
                contract["required_fields"],
                path + ".value_contract.required_fields",
                {"raw", "value", "currency", "unit", "period"},
                nonempty=True,
            )
            _bool(
                contract["numeric_value"],
                path + ".value_contract.numeric_value",
            )
            _strings(
                contract["grounded_fields"],
                path + ".value_contract.grounded_fields",
                {"unit", "period"},
            )
        markers = rule.get("integer_qualifier_markers", {})
        _patterns(markers, path + ".integer_qualifier_markers")
        for qualifier in markers:
            if qualifier not in rule.get("grounded_integer_qualifiers", {}):
                _fail(
                    path + ".integer_qualifier_markers",
                    "marker has no integer bounds",
                )
        for qualifier, bounds in rule.get(
            "grounded_integer_qualifiers", {}
        ).items():
            if not isinstance(bounds, list) or len(bounds) != 2:
                _fail(path + "." + qualifier, "must have two bounds")
            low = _number(bounds[0], path + "." + qualifier, integer=True)
            _number(bounds[1], path + "." + qualifier, low, integer=True)
    for name, projection in graph["projections"].items():
        path = "graph.projections." + name
        if name not in predicates:
            _fail(path, "unknown LLM predicate")
        _identifier(projection["relationship"], path + ".relationship")
        for endpoint in ("source", "target"):
            role = projection[endpoint + "_role"]
            if role not in predicates[name]["allowed_roles"]:
                _fail(
                    path + "." + endpoint + "_role",
                    "role not allowed by predicate",
                )
            _strings(
                projection[endpoint + "_kinds"],
                path + "." + endpoint + "_kinds",
                set(predicates[name]["role_types"][role]),
                nonempty=True,
            )
    maturity = graph["maturity_predicate"]
    if maturity not in predicates:
        _fail("graph.maturity_predicate", "unknown LLM predicate")
    stages = predicates[maturity].get("qualifier_enums", {}).get("stage", [])
    if set(stages) != set(graph["maturity_stage_rank"]):
        _fail(
            "graph.maturity_stage_rank", "stages differ from the LLM contract"
        )
    for name, rank in graph["maturity_stage_rank"].items():
        _number(rank, "graph.maturity_stage_rank." + name, 1, integer=True)
    for mapping in ("organization_relationships", "organization_labels"):
        for key, value in graph[mapping].items():
            _identifier(value, "graph." + mapping + "." + key)
    if "solution_predicates" in graph:
        _strings(
            graph["solution_predicates"],
            "graph.solution_predicates",
            set(predicates),
        )
    for rule in catalogs["extraction"]["assertions"]["relation_rules"]:
        if rule["predicate"] not in predicates or rule["role"] not in roles:
            _fail(
                "extraction.assertions.relation_rules",
                "unknown predicate or graph role",
            )


def _models(catalogs: Mapping) -> None:
    runtime, resolver, llm = (
        catalogs["runtime"],
        catalogs["resolver"],
        catalogs["llm"],
    )
    _text(runtime["ner_model"], "runtime.ner_model")
    if runtime["default_extractor"] not in {"none", "gliner", "llm", "hybrid"}:
        _fail("runtime.default_extractor", "unknown extractor")
    for group in resolver["explicit_aliases"]:
        if group["kind"] not in KINDS:
            _fail("resolver.explicit_aliases.kind", "unknown ConceptKind")
        _strings(
            group["names"], "resolver.explicit_aliases.names", nonempty=True
        )
    semantic = resolver["semantic"]
    _strings(
        semantic["embedded_kinds"], "resolver.semantic.embedded_kinds", KINDS
    )
    for key in ("cosine_threshold", "decision_threshold"):
        _number(semantic[key], "resolver.semantic." + key, 0, 1)
    if semantic["embedding_provider"] not in semantic["embedding_models"]:
        _fail(
            "resolver.semantic.embedding_provider",
            "no model configured for provider",
        )
    for provider, model in semantic["embedding_models"].items():
        _text(model, "resolver.semantic.embedding_models." + provider)
    _text(semantic["decision_model"], "resolver.semantic.decision_model")
    for key in ("embedding_batch_size", "max_length"):
        _number(semantic[key], "resolver.semantic." + key, 1, integer=True)
    if llm["provider"] not in {"openai_compatible", "gigachat"}:
        _fail("llm.provider", "unknown provider")
    for section in (llm, llm["gigachat"]):
        _strings(section["model_ladder"], "llm.model_ladder")
        _number(
            section["max_concurrent_requests"],
            "llm.max_concurrent_requests",
            1,
            integer=True,
        )
    for stage in ("extract", "review"):
        _number(
            llm["max_output_tokens"][stage],
            "llm.max_output_tokens." + stage,
            1,
            integer=True,
        )
    for key, value in llm["timeouts_seconds"].items():
        _number(value, "llm.timeouts_seconds." + key, 0.001)


def _analytics(catalogs: Mapping) -> None:
    dataset, taxonomy = catalogs["dataset"], catalogs["taxonomy"]
    snapshots = dataset["snapshots"]
    for key, low, high in (
        ("start_year", 1, 9999),
        ("month", 1, 12),
        ("day", 1, 31),
        ("step_months", 1, None),
    ):
        _number(
            snapshots[key], "dataset.snapshots." + key, low, high, integer=True
        )
    try:
        date(snapshots["start_year"], snapshots["month"], snapshots["day"])
    except ValueError as exc:
        _fail("dataset.snapshots", str(exc))
    for key in ("horizon_years", "min_documents", "reliability_tier_max"):
        _number(dataset[key], "dataset." + key, 1, integer=True)
    for key in ("valid_snapshots", "test_snapshots"):
        _number(dataset["split"][key], "dataset.split." + key, 1, integer=True)
    label = dataset["label"]
    _number(
        label["min_future_independent_sources"],
        "dataset.label.min_future_independent_sources",
        1,
        integer=True,
    )
    outcomes = {
        "future_patents",
        "future_repositories",
        "future_packages",
        "future_users",
        "future_commercial_evidence",
    }
    _strings(
        label["implementation_outcomes"],
        "dataset.label.implementation_outcomes",
        outcomes,
        nonempty=True,
    )
    _number(
        label["commercial_stage_rank"],
        "dataset.label.commercial_stage_rank",
        1,
        max(catalogs["graph"]["maturity_stage_rank"].values()),
        integer=True,
    )
    categories = set(catalogs["extraction"]["economics"]["categories"])
    _strings(
        dataset["commercial_economic_categories"],
        "dataset.commercial_economic_categories",
        categories,
    )
    modalities = {"reported", "observed", "planned", "hypothetical", "unknown"}
    factual = _strings(
        dataset["factual_modalities"], "dataset.factual_modalities", modalities
    )
    speculative = _strings(
        dataset["speculative_modalities"],
        "dataset.speculative_modalities",
        modalities,
    )
    if set(factual) & set(speculative):
        _fail("dataset.factual_modalities", "overlaps speculative_modalities")
    for key, value in dataset["coverage_families"].items():
        if value not in DOCUMENT_TYPES:
            _fail("dataset.coverage_families." + key, "unknown document type")
    for key, value in dataset["subgraph"].items():
        _number(value, "dataset.subgraph." + key, 1, integer=True)
    _strings(taxonomy["kinds"], "taxonomy.kinds", KINDS, nonempty=True)
    for key in (
        "branching",
        "max_depth",
        "min_cluster_size",
        "kmeans_iterations",
        "label_terms",
        "known_age_days",
    ):
        _number(taxonomy[key], "taxonomy." + key, 1, integer=True)
    if taxonomy["branching"] < 2:
        _fail("taxonomy.branching", "must be at least two")
    for key in (
        "general_term_margin",
        "explicit_parent_weight",
        "new_branch_share",
    ):
        _number(taxonomy[key], "taxonomy." + key, 0, 1)


def validate_catalogs(catalogs: Optional[Mapping[str, Any]] = None) -> None:
    """Raise ValueError naming the invalid resource in a complete catalog set.

    Supplying a mapping makes validation deterministic and network-free in
    tests. Without one, the active files (including config overrides) are read.
    """
    if catalogs is None:
        from .config import load_catalog

        catalogs = {name: load_catalog(name) for name in CATALOG_NAMES}
    for name in CATALOG_NAMES:
        catalog = _mapping(_required(catalogs, name, "catalogs"), name)
        if "schema_version" in catalog:
            _number(
                catalog["schema_version"],
                name + ".schema_version",
                1,
                integer=True,
            )
    codes = _strings(
        catalogs["countries"]["iso_alpha2"],
        "countries.iso_alpha2",
        nonempty=True,
    )
    for code in codes:
        if not re.fullmatch(r"[A-Z]{2}", code) or code in {"ZZ", "EU", "XK"}:
            _fail(
                "countries.iso_alpha2",
                f"invalid assigned ISO country code {code!r}",
            )
    try:
        _sources(catalogs["sources"])
        _extraction(catalogs["extraction"])
        _pipeline(catalogs["pipeline"])
        _contracts(catalogs)
        _models(catalogs)
        _analytics(catalogs)
    except (KeyError, TypeError, AttributeError) as exc:
        _fail("catalogs", f"missing or malformed required field: {exc}")
