"""Evaluation harness -- three checks, run together via ``python -m lp_uworld_rag eval``:

  1. Metadata integrity -- scans every stored chunk's metadata directly (no retrieval involved) and
     flags structural problems: missing required fields, malformed stable ids, dangling
     cross-references, exact bidirectional mismatches (a controller's used_by vs. the endpoints that
     actually point to it), and completeness gaps (a db-collection no endpoint's own ``db_ids``
     references, or an endpoint with an empty ``db_ids``). This is
     what caught real bugs in the ``Order(api)`` content this session (inconsistent repo naming, a
     multi-value ``controller`` field corrupting ``controller_id`` into a Python-list string) --
     exactly the kind of thing that's invisible from a query's top-3 results looking plausible, so
     it's checked directly against the stored data, not inferred from retrieval.
  2. Endpoint -> DB-collection mapping -- exercises the actual runtime resolve path
     (``index._resolve_citations``) for every endpoint chunk, not a second static recomputation of
     the same metadata. Confirms that when an endpoint is genuinely retrieved, its DB-collection
     citations really do come back -- catches a bug in the resolve/join logic itself, which tier 1's
     pure-metadata check can't (tier 1 can only tell you the data supports a mapping, not that the
     code that's supposed to surface it at query time actually does).
  3. Retrieval quality -- a curated set of deep-reasoning queries spanning the three intents this
     project is designed for (bugfix / enhancement / new-feature, see the plan's Document model
     section), each asserting the top hit is one of a small set of acceptable ids, PLUS a diversity
     check: a broad "give me the full picture" query should surface a mix across feature-hub,
     technical-hub, endpoint, AND db-collection together in one result set, not just one doc_type
     dominating -- confirms the quota/citation machinery actually delivers cross-collection,
     cross-doc_type coverage for the queries that need it, not just narrow single-doc_type retrieval.
     A "fail" here means retrieval quality regressed, not necessarily a hard bug -- worth a look either
     way.

All four are read-only against whatever is currently ingested -- run after any ingest to catch
regressions before they reach an actual agent session.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .chunker import decode_list_field
from .config import RagConfig
from .index import (COLLECTIONS, RetrieverCache, _resolve_citations, build_retriever_cache,
                     get_chroma_client, get_collection, query)

_CTRL_ID_RE = re.compile(r"^ctrl::[^:\s\[\]'\"/\\]+::[^:\s\[\]'\"/\\]+$")
_EP_ID_RE = re.compile(r"^ep::[^:\s\[\]'\"/\\]+::[^:\s\[\]'\"/\\{}]+$")
_DB_ID_RE = re.compile(r"^db::[^:\s\[\]'\"/\\]+::[^:\s\[\]'\"/\\]+$")

# WS0 (token-economy plan): tiktoken's cl100k_base as a consistent, real proxy for Claude token
# count (Claude's own tokenizer isn't public) -- same choice ReportsRagPy's token_usage_bench.py
# makes, so numbers are comparable across both projects. Encoder built once, lazily, since eval's
# --help/config-loading path shouldn't require tiktoken installed.
_ENC = None


def count_tokens(text: str | None) -> int:
    global _ENC
    if not text:
        return 0
    if _ENC is None:
        import tiktoken
        _ENC = tiktoken.get_encoding("cl100k_base")
    return len(_ENC.encode(text, disallowed_special=()))


def _chunks_tokens(chunks: list[dict]) -> int:
    """Total tokens across a retrieval result's chunk ``content`` -- what would actually land in
    Claude's context if this response were handed to an agent verbatim (WS0's "tokens_in")."""
    return sum(count_tokens(c.get("content")) for c in chunks)


@dataclass
class Finding:
    severity: str  # "error" | "warn"
    where: str
    message: str


@dataclass
class QueryCase:
    scope: str  # "bugfix" | "enhancement" | "new-feature" | "functional" | "out-of-domain"
    question: str
    accept_ids: list[str] = field(default_factory=list)  # any of these appearing in top-5 passes
    note: str = ""
    informational: bool = False  # shown but never counted toward pass/fail (e.g. out-of-domain probes)


@dataclass
class DiversityCase:
    """A broad query that should return coverage across doc_types together, not one type dominating
    -- a different failure mode than QueryCase's "is the right single answer at the top" (a query can
    pass every QueryCase and still, say, never surface a feature-hub alongside its technical detail)."""
    question: str
    required_doc_types: list[str]
    top_k: int = 8


# -- Tier 1: metadata integrity -------------------------------------------------

def check_metadata(cfg: RagConfig) -> list[Finding]:
    findings: list[Finding] = []
    all_metas: dict[str, list[dict]] = {}
    for c in COLLECTIONS:
        client = get_chroma_client(cfg, c)
        coll = get_collection(cfg, client, c)
        got = coll.get(include=["metadatas", "documents"])
        all_metas[c] = got.get("metadatas", [])

        # Oversize-chunk check: MAX_CHUNK_CHARS is a defensive ceiling, not a target -- a node that
        # actually hit it lost real content (confirmed live once on
        # ep::reports::GetFacultyLedGroupPerformance's Gotchas section before the cap was raised
        # 3000->8000). Flag both the definitive case (the truncation marker itself, meaning content
        # was cut THIS ingest) and the near-cap warning (>=80%, meaning the next edit to that page
        # risks the same silent loss) so it's visible instead of trusted to the authoring convention.
        from .chunker import MAX_CHUNK_CHARS
        docs = got.get("documents") or []
        for md, text in zip(all_metas[c], docs):
            if not text:
                continue
            where = f"{c}/{md.get('doc_type')}/{md.get('title')!r}"
            if text.rstrip().endswith("_[... truncated]_"):
                findings.append(Finding("error", where,
                                         f"content truncated at MAX_CHUNK_CHARS ({MAX_CHUNK_CHARS}) -- real content lost"))
            elif len(text) >= 0.8 * MAX_CHUNK_CHARS:
                findings.append(Finding("warn", where,
                                         f"chunk is {len(text)} chars, within 20% of MAX_CHUNK_CHARS "
                                         f"({MAX_CHUNK_CHARS}) -- at risk of truncation on its next edit"))

    known_endpoint_ids: set[str] = set()
    known_controller_ids: set[str] = set()
    known_db_ids: set[str] = set()
    referenced_db_ids: set[str] = set()  # via endpoints' own forward db_ids (dependency-direction-correct)
    endpoints_by_controller: dict[str, set[str]] = {}
    feature_hub_features: set[str] = set()
    technical_hub_features: set[str] = set()
    endpoint_features: set[str] = set()

    for c, metas in all_metas.items():
        for md in metas:
            doc_type = md.get("doc_type")
            where = f"{c}/{doc_type}/{md.get('title')!r}"

            if doc_type == "endpoint":
                eid, cid, repo = md.get("endpoint_id"), md.get("controller_id"), md.get("repo")
                if not eid:
                    findings.append(Finding("error", where, "missing endpoint_id"))
                elif not _EP_ID_RE.match(eid):
                    findings.append(Finding("error", where, f"malformed endpoint_id: {eid!r}"))
                else:
                    known_endpoint_ids.add(eid)
                if not cid:
                    findings.append(Finding("error", where, "missing controller_id"))
                elif not _CTRL_ID_RE.match(cid):
                    findings.append(Finding("error", where, f"malformed controller_id: {cid!r}"))
                if not repo:
                    findings.append(Finding("error", where, "missing repo"))
                feats = decode_list_field(md.get("feature"))
                if not feats:
                    findings.append(Finding("warn", where, "empty feature list"))
                endpoint_features |= set(feats)
                if not md.get("route"):
                    findings.append(Finding("error", where, "missing route"))
                if cid and _CTRL_ID_RE.match(cid) and eid:
                    endpoints_by_controller.setdefault(cid, set()).add(eid)
                # db_ids -- the endpoint's own forward reference to the data stores it reads
                # (dependency-direction-correct: endpoint -> data store). Malformed entries are
                # checked here; dangling-reference and completeness checks run below, once every
                # endpoint's and db-collection's own db_id is known.
                db_ids = decode_list_field(md.get("db_ids"))
                if not db_ids:
                    findings.append(Finding("warn", where, "empty db_ids -- endpoint reads no data store?"))
                for did in db_ids:
                    if not _DB_ID_RE.match(did):
                        findings.append(Finding("error", where, f"malformed db_ids entry: {did!r}"))
                    else:
                        referenced_db_ids.add(did)

            elif doc_type == "db-collection":
                for key in ("store", "type", "collection", "db_id"):
                    if not md.get(key):
                        findings.append(Finding("error", where, f"missing {key}"))
                if md.get("db_id"):
                    if not _DB_ID_RE.match(md["db_id"]):
                        findings.append(Finding("error", where, f"malformed db_id: {md['db_id']!r}"))
                    else:
                        known_db_ids.add(md["db_id"])

            elif doc_type == "controller-context":
                cid = md.get("controller_id")
                if not cid:
                    findings.append(Finding("error", where, "missing controller_id"))
                elif not _CTRL_ID_RE.match(cid):
                    findings.append(Finding("error", where, f"malformed controller_id: {cid!r}"))
                else:
                    known_controller_ids.add(cid)
                if not decode_list_field(md.get("used_by")):
                    findings.append(Finding("warn", where, "empty used_by -- unreferenced controller?"))

            elif doc_type in ("technical-hub", "feature-hub"):
                feats = decode_list_field(md.get("feature"))
                if not feats:
                    findings.append(Finding("error", where, "empty feature list"))
                (technical_hub_features if doc_type == "technical-hub" else feature_hub_features).update(feats)

    # Cross-reference checks -- dangling references between doc types.
    for c, metas in all_metas.items():
        for md in metas:
            if md.get("doc_type") == "endpoint":
                cid = md.get("controller_id")
                if cid and _CTRL_ID_RE.match(cid) and cid not in known_controller_ids:
                    findings.append(Finding("error", f"{c}/endpoint/{md.get('title')!r}",
                                             f"controller_id {cid!r} has no matching controller-context chunk"))
                for did in decode_list_field(md.get("db_ids")):
                    if _DB_ID_RE.match(did) and did not in known_db_ids:
                        findings.append(Finding("error", f"{c}/endpoint/{md.get('title')!r}",
                                                 f"db_ids entry {did!r} has no matching db-collection chunk"))
            if md.get("doc_type") == "controller-context":
                for eid in decode_list_field(md.get("used_by")):
                    if eid not in known_endpoint_ids:
                        findings.append(Finding("warn", f"{c}/controller-context/{md.get('title')!r}",
                                                 f"used_by endpoint_id {eid!r} matches no known endpoint"))

    # Exact bidirectional match: a controller's used_by must be EXACTLY the endpoints that point at
    # it -- not just "not dangling" (already checked above), which alone would miss the reverse case
    # (an endpoint points at a controller whose used_by forgot to list it back).
    for c, metas in all_metas.items():
        for md in metas:
            if md.get("doc_type") != "controller-context":
                continue
            cid = md.get("controller_id")
            if not cid:
                continue
            where = f"{c}/controller-context/{md.get('title')!r}"
            used_by = set(decode_list_field(md.get("used_by")))
            pointing = endpoints_by_controller.get(cid, set())
            missing = pointing - used_by  # endpoints that point here but aren't listed back
            extra = used_by - pointing    # listed here but no endpoint actually points here
            if missing:
                findings.append(Finding("error", where,
                                         f"used_by is missing endpoint(s) that point to this controller: {sorted(missing)}"))
            if extra:
                findings.append(Finding("warn", where,
                                         f"used_by lists endpoint(s) that no longer point to this controller: {sorted(extra)}"))

    # Completeness: every db-collection should be referenced by at least one endpoint's db_ids --
    # a db-collection nothing points to is either genuinely unreferenced (safe to remove, see the
    # root page's teardown checklist) or a missed db_ids entry on whichever endpoint actually reads
    # it. Reverse direction from the old consumed_by-based check (that checked "every endpoint has
    # a consumer-listing db-collection"); now the endpoint side is authoritative, so this checks the
    # data-store side isn't orphaned instead. Endpoints missing db_ids entirely are already flagged
    # above ("empty db_ids").
    for c, metas in all_metas.items():
        for md in metas:
            if md.get("doc_type") == "db-collection" and md.get("db_id") and md["db_id"] not in referenced_db_ids:
                findings.append(Finding("warn", f"{c}/db-collection/{md.get('title')!r}",
                                         "no endpoint's db_ids references this collection"))

    # Every feature an endpoint or technical-hub references should have both a Feature Hub and a
    # Technical Hub chunk -- a missing Feature Hub is expected/current for not-yet-authored features
    # (warn, not error); a missing Technical Hub means the feature's own endpoints have nothing
    # indexing them as a group (error).
    for feat in sorted(endpoint_features | technical_hub_features):
        if feat not in feature_hub_features:
            findings.append(Finding("warn", f"feature/{feat!r}", "no feature-hub chunk (not yet authored?)"))
        if feat not in technical_hub_features:
            findings.append(Finding("error", f"feature/{feat!r}", "no technical-hub chunk"))

    return findings


# -- Tier 2: endpoint -> DB-collection mapping, via the real resolve path ------

def check_endpoint_db_mapping(cache: RetrieverCache) -> list[Finding]:
    """For every endpoint chunk, call the actual query-time resolve function
    (``index._resolve_citations``) and confirm every data store its own ``db_ids`` names comes back
    as a citation. Unlike tier 1 (which only checks the metadata is internally consistent), this
    exercises the same code path a live query hits -- it would catch, say, a typo in the
    ``id_index`` lookup inside ``_resolve_citations`` that tier 1's pure-data check can't see
    because the metadata itself is perfectly fine (a malformed/dangling ``db_ids`` entry is tier 1's
    job; this is "does the resolve mechanism actually turn a well-formed one into a citation")."""
    findings: list[Finding] = []
    docstore = cache.docstores["technical"]
    if docstore is None:
        findings.append(Finding("error", "technical", "no docstore loaded -- has ingest run?"))
        return findings

    for node_id, doc in docstore.docs.items():
        md = doc.metadata or {}
        if md.get("doc_type") != "endpoint":
            continue
        where = f"technical/endpoint/{md.get('title')!r}"
        expected_db_ids = set(decode_list_field(md.get("db_ids")))
        citations = _resolve_citations(md, node_id, "technical", cache, exclude_ids=set())
        cited_db_ids = {c["id"] for c in citations if c["doc_type"] == "db-collection"}

        missing = expected_db_ids - cited_db_ids
        if missing:
            findings.append(Finding("error", where,
                                     f"resolve_citations did not surface expected db-collection(s): {sorted(missing)}"))
        if not expected_db_ids:
            findings.append(Finding("warn", where, "no db-collection maps to this endpoint at all"))

    return findings


# -- Tier 3: retrieval quality (deep-reasoning queries) -------------------------
# Ids are drawn from the live reports-domain corpus (see confluence-manifest.json / live Confluence).
# Kept small and specific rather than exhaustive -- each one probes a distinct reasoning shape.

QUERIES: list[QueryCase] = [
    QueryCase("bugfix", "Why would GetTotalPrepGroupPerformance return an empty result for a valid FpGroupId?",
              ["ep::reports::GetTotalPrepGroupPerformance"]),
    QueryCase("bugfix", "What happens when IsFacultyQbank is null on the group-performance endpoint?",
              ["ep::reports::GetFacultyLedGroupPerformance"]),
    QueryCase("bugfix",
              "Why would GetFacultyLedStudentPerformanceByDate return different totals than "
              "GetFacultyLedStudentPerformance for the same student?",
              ["ep::reports::GetFacultyLedStudentPerformanceByDate", "ep::reports::GetFacultyLedStudentPerformance"],
              note="either sibling is an acceptable top hit -- the query is genuinely about their relationship"),
    QueryCase("enhancement",
              "Adding a field to the faculty-led group performance response -- what other endpoints "
              "need to stay consistent?",
              ["ep::reports::GetFacultyLedGroupPerformance", "ep::reports::GetFacultyLedGroupPerformanceByDate"]),
    QueryCase("enhancement", "What MongoDB collections does the group-performance-by-date endpoint touch?",
              ["db::ReportsDatabase::group-daily-activity", "db::ReportsDatabase::group-performance",
               "db::ReportsDatabase::qbank-national-performance"]),
    QueryCase("new-feature",
              "How should a new self-study endpoint that filters by course type be structured, "
              "similar to Total Prep?",
              ["ep::reports::GetTotalPrepGroupPerformance"]),
    QueryCase("functional",
              "Why does the Faculty-Led Customizable Report support both group and individual student views?",
              ["7475626269"],  # Feature Hub page_id (no stable id scheme for feature-hub chunks)
              note="feature-hub chunks fall back to page_id as their public id"),
    QueryCase("bugfix",
              "What's the authentication requirement and DI setup for the controller behind the "
              "faculty-led group performance endpoint?",
              ["ctrl::reports::FacultyLedPerformanceController"],
              note="targets controller-context specifically -- the one doc_type none of the other cases hit"),
    QueryCase("new-feature",
              "What endpoints make up the Total Prep Series feature end to end?",
              ["7475658832"],  # Total Prep Technical Hub page_id (same fallback as feature-hub)
              note="targets technical-hub specifically -- the other doc_type none of the other cases hit"),
    QueryCase("out-of-domain",
              "How do I reset a user's password in the admin portal?",
              informational=True,
              note="nothing in this corpus should genuinely answer this -- checking for graceful "
                   "degradation (a plausible-looking but wrong low-confidence guess, not a crash), "
                   "not a specific correct answer"),
]


def run_queries(cfg: RagConfig, cache=None) -> list[dict]:
    """Retrieve top-5 per case (not just top-1) so a case that's "almost right" -- correct answer at
    rank 2 or 3 -- is visible as a near-miss rather than an indistinguishable-from-random failure.
    Also times each call: query latency is part of "is this actually usable," not just correctness.

    Passes on either of two conditions, each representing something a caller actually sees without
    already knowing the answer:
      * direct_rank <= 3 -- the correct chunk is independently present in what a normal caller would
        actually read (the top few results), even if not literally first. Note the design's own
        "cite, don't inline" rule means a node independently retrieved at, say, rank 2 is correctly
        *excluded* from rank 1's citations (no point duplicating it) -- so a rank-2/3 hit and a
        citation are two different, non-overlapping ways the right answer reaches the caller, not
        one masking the other.
      * cited_on_top -- the #1 hit's own citations list names the correct id+title directly.
    Strict rank-1 is tracked separately (see "rank" in the returned dict) so ranking quality is still
    visible in the report even when the looser pass/fail bar is satisfied.
    """
    import time

    cache = cache or build_retriever_cache(cfg)
    results = []
    for case in QUERIES:
        t0 = time.perf_counter()
        r = query(cfg, case.question, top_k=5, cache=cache)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        chunks = r["chunks"]
        direct_rank = next((i + 1 for i, c in enumerate(chunks) if c["id"] in case.accept_ids), None)
        top_citations = {cit["id"] for cit in (chunks[0].get("citations") or [])} if chunks else set()
        cited_on_top = bool(set(case.accept_ids) & top_citations)
        results.append({
            "case": case,
            "top": chunks[0] if chunks else None,
            "rank": direct_rank,  # 1-indexed position of first accepted id as a direct hit, or None
            "cited_on_top": cited_on_top,  # accept_id appears in the #1 hit's own citations
            "passed": (direct_rank is not None and direct_rank <= 3) or cited_on_top,
            "elapsed_ms": elapsed_ms,
            "tokens_in": _chunks_tokens(chunks),  # WS0: what this response would cost in Claude's context
        })
    return results


DIVERSITY_CASES: list[DiversityCase] = [
    DiversityCase(
        "Give me the full picture of the Faculty-Led Customizable Report -- the business purpose, "
        "the technical index of its endpoints, and what data it's built on",
        required_doc_types=["feature-hub", "technical-hub", "endpoint", "db-collection"],
    ),
]


def run_diversity_cases(cfg: RagConfig, cache=None) -> list[dict]:
    cache = cache or build_retriever_cache(cfg)
    results = []
    for case in DIVERSITY_CASES:
        r = query(cfg, case.question, top_k=case.top_k, cache=cache)
        present = {c["docType"] for c in r["chunks"]}
        missing = [dt for dt in case.required_doc_types if dt not in present]
        results.append({"case": case, "present": sorted(present), "missing": missing, "passed": not missing})
    return results


# -- Tier 4: expand() correctness ------------------------------------------------
# One id of each stable-id family (endpoint, controller, db-collection) plus a bare page_id
# (feature-hub/technical-hub's fallback) -- confirms expand() actually returns real, non-empty
# content of the right doc_type for every id shape query_rag can hand back as a citation.

EXPAND_SAMPLES: list[tuple[str, str]] = [
    ("ep::reports::GetFacultyLedGroupPerformance", "endpoint"),
    ("ctrl::reports::FacultyLedPerformanceController", "controller-context"),
    ("db::ReportsDatabase::group-performance", "db-collection"),
    ("7475626269", "feature-hub"),  # bare page_id -- Faculty-Led Customizable Report
]


def check_expand(cfg: RagConfig, cache: RetrieverCache) -> list[Finding]:
    from .index import expand

    findings: list[Finding] = []
    for id_, expected_doc_type in EXPAND_SAMPLES:
        where = f"expand({id_!r})"
        result = expand(cfg, id_, cache=cache)
        if not result:
            findings.append(Finding("error", where, "returned no results"))
            continue
        doc_types = {r["docType"] for r in result}
        if expected_doc_type not in doc_types:
            findings.append(Finding("error", where,
                                     f"expected doc_type {expected_doc_type!r}, got {sorted(doc_types)}"))
        if not all(r.get("content") for r in result):
            findings.append(Finding("error", where, "at least one result has empty content"))
    return findings


# -- Tier 5: orchestrator routing (docs -> repo -> code) -----------------------
# Requires at least one registered repo manifest (config.json's "repos.manifests" --
# see docs/repo-rag-contract.md); skipped entirely otherwise so eval stays runnable before any
# repo has onboarded.

@dataclass
class OrchestratorCase:
    question: str
    expected_repos: list[str] | None  # None -> routing is expected to fall back (any repo set ok)
    expect_fallback: bool
    accept_files: list[str] = field(default_factory=list)  # basenames; any hit across routed repos passes
    informational: bool = False  # shown but never counted toward pass/fail -- see the fallback case below


ORCHESTRATOR_CASES: list[OrchestratorCase] = [
    OrchestratorCase(
        "why does the faculty-led group performance endpoint return null for a valid FpGroupId",
        expected_repos=["reports"], expect_fallback=False,
        accept_files=["FacultyLedPerformanceController.cs", "FacultyLedPerformanceRepository.cs"],
    ),
    # Intended to exercise the fallback-to-every-registered-repo path (guardrail R2) with a
    # question matching no registered repo. Kept informational rather than scored: today's corpus
    # is entirely "reports" content, so even a weak/irrelevant doc hit still carries repo: reports
    # metadata and genuinely never triggers fallback (confirmed directly against _route with a
    # synthetic no-match/unregistered-repo doc set -- that logic path itself is sound). This case
    # starts actually asserting fallback the moment a second, non-overlapping repo onboards.
    OrchestratorCase(
        "how does OAuth token refresh work across the entire platform",
        expected_repos=None, expect_fallback=True, informational=True,
    ),
]


def run_orchestrator_cases(cfg: RagConfig, registry, cache=None) -> list[dict] | None:
    if registry is None or not registry.repo_keys():
        return None
    from .orchestrator import deep_query

    cache = cache or build_retriever_cache(cfg)
    results = []
    for case in ORCHESTRATOR_CASES:
        r = deep_query(cfg, registry, case.question, cache=cache)
        routing = r["routing"]
        repos_ok = set(routing["repos"]) == set(case.expected_repos) if case.expected_repos is not None else True
        fallback_ok = routing["fallback_used"] == case.expect_fallback
        files_seen = {c.get("filePath") for chunks in r["code"].values() for c in chunks}
        files_ok = True
        if case.accept_files:
            files_ok = any(f and any(f.endswith(basename) for basename in case.accept_files) for f in files_seen)
        # WS0: the full assembled payload tokens -- docs chunks + every routed repo's code chunks --
        # is the actual per-`deep_query`-call cost landing in an agent's context today (both stages
        # inline full `content` for every hit; see the plan's WS2/WS4 for the levers that shrink this).
        code_chunks = [c for chunks in r["code"].values() for c in chunks]
        payload_tokens = _chunks_tokens(r["docs"]["chunks"]) + _chunks_tokens(code_chunks)
        results.append({
            "case": case, "routing": routing, "files_seen": sorted(f for f in files_seen if f),
            "passed": repos_ok and fallback_ok and files_ok,
            "payload_tokens": payload_tokens,
        })
    return results


# -- report ----------------------------------------------------------------------

def run(cfg: RagConfig) -> bool:
    """Run all five tiers, print a report, return True iff nothing failed at error severity."""
    print("=" * 90)
    print("TIER 1 -- metadata integrity")
    print("=" * 90)
    findings = check_metadata(cfg)
    errors = [f for f in findings if f.severity == "error"]
    warnings = [f for f in findings if f.severity == "warn"]
    if not findings:
        print("  no issues found")
    for f in findings:
        print(f"  [{f.severity.upper():5}] {f.where}: {f.message}")
    print(f"\n  {len(errors)} error(s), {len(warnings)} warning(s)")

    # Build the retriever cache once, shared by tiers 2 and 3 -- both need the embed model + docstores
    # loaded, no reason to pay for that twice.
    cache = build_retriever_cache(cfg)

    print()
    print("=" * 90)
    print("TIER 2 -- endpoint -> DB-collection mapping (live resolve check)")
    print("=" * 90)
    db_findings = check_endpoint_db_mapping(cache)
    db_errors = [f for f in db_findings if f.severity == "error"]
    db_warnings = [f for f in db_findings if f.severity == "warn"]
    if not db_findings:
        print("  no issues found -- every endpoint's expected db-collection(s) resolved via the real citation path")
    for f in db_findings:
        print(f"  [{f.severity.upper():5}] {f.where}: {f.message}")
    print(f"\n  {len(db_errors)} error(s), {len(db_warnings)} warning(s)")

    print()
    print("=" * 90)
    print("TIER 3 -- retrieval quality (deep-reasoning queries, top-5 + latency)")
    print("=" * 90)
    query_results = run_queries(cfg, cache=cache)
    scored = [r for r in query_results if not r["case"].informational]
    query_failures = 0
    for r in query_results:
        case, top, rank, passed = r["case"], r["top"], r["rank"], r["passed"]
        top_desc = f"{top['id']} ({top['docType']})" if top else "<no results>"
        if case.informational:
            status = "INFO"
        elif rank == 1:
            status = "PASS"
        elif passed and rank is not None:
            status = f"PASS (rank {rank})"  # within top-3 -- counts as passing, ranking still worth tightening
        elif passed and r["cited_on_top"]:
            status = "PASS (via citation on #1 hit)"
        elif rank is not None:
            status = f"NEAR (rank {rank})"  # found, but beyond the top-3 pass threshold
            query_failures += 1
        else:
            status = "FAIL"
            query_failures += 1
        print(f"  [{status}] [{case.scope}] {case.question}")
        print(f"         top hit: {top_desc}  ({r['elapsed_ms']:.0f}ms, {r['tokens_in']} tokens)"
              + (f"  -- {case.note}" if case.note else ""))

    avg_ms = sum(r["elapsed_ms"] for r in query_results) / len(query_results)
    total_tokens = sum(r["tokens_in"] for r in query_results)
    avg_tokens = total_tokens / len(query_results)
    print()
    print(f"  {len(scored) - query_failures}/{len(scored)} scored queries passed (rank <=3 or cited on #1 hit) "
          f"({len(query_results) - len(scored)} informational, not scored)")
    print(f"  avg query latency: {avg_ms:.0f}ms")
    print(f"  tokens_in: {total_tokens} total across {len(query_results)} queries "
          f"({avg_tokens:.0f} avg/query) -- WS0 baseline for the token-economy plan (WS1-WS5)")

    print()
    print("-" * 90)
    print("  cross-doc_type diversity (broad queries should span multiple doc_types, not one)")
    print("-" * 90)
    diversity_results = run_diversity_cases(cfg, cache=cache)
    diversity_failures = sum(0 if d["passed"] else 1 for d in diversity_results)
    for d in diversity_results:
        status = "PASS" if d["passed"] else "FAIL"
        print(f"  [{status}] {d['case'].question}")
        print(f"         doc_types present: {d['present']}" + (f"  MISSING: {d['missing']}" if d["missing"] else ""))
    print(f"\n  {len(diversity_results) - diversity_failures}/{len(diversity_results)} diversity cases passed")

    print()
    print("=" * 90)
    print("TIER 4 -- expand() correctness")
    print("=" * 90)
    expand_findings = check_expand(cfg, cache)
    expand_errors = [f for f in expand_findings if f.severity == "error"]
    if not expand_findings:
        print(f"  no issues found -- all {len(EXPAND_SAMPLES)} stable-id shapes (endpoint/controller/"
              f"db-collection/bare page_id) expand to real, non-empty content")
    for f in expand_findings:
        print(f"  [{f.severity.upper():5}] {f.where}: {f.message}")
    print(f"\n  {len(expand_errors)} error(s)")

    print()
    print("=" * 90)
    print("TIER 5 -- orchestrator routing (docs -> repo -> code)")
    print("=" * 90)
    from .repo_registry import RepoRegistry

    registry = RepoRegistry.from_config(cfg)
    orchestrator_failures = 0
    orchestrator_results = run_orchestrator_cases(cfg, registry, cache=cache)
    if orchestrator_results is None:
        print("  skipped -- no repo manifests registered (see config.json's \"repos.manifests\")")
    else:
        scored_orchestrator = [r for r in orchestrator_results if not r["case"].informational]
        for r in orchestrator_results:
            if r["case"].informational:
                status = "INFO"
            else:
                status = "PASS" if r["passed"] else "FAIL"
                if not r["passed"]:
                    orchestrator_failures += 1
            print(f"  [{status}] {r['case'].question}")
            print(f"         routed: {r['routing']['repos']}  fallback_used: {r['routing']['fallback_used']}"
                  f"  files_seen: {r['files_seen']}  payload_tokens: {r['payload_tokens']}")
        print(f"\n  {len(scored_orchestrator) - orchestrator_failures}/{len(scored_orchestrator)} scored "
              f"orchestrator cases passed ({len(orchestrator_results) - len(scored_orchestrator)} informational, not scored)")
        total_payload_tokens = sum(r["payload_tokens"] for r in orchestrator_results)
        print(f"  assembled deep_query payload tokens: {total_payload_tokens} total across "
              f"{len(orchestrator_results)} call(s) -- WS0 baseline for orchestrator-level token cost")
    if registry.errors:
        for path, err in registry.errors.items():
            print(f"  [WARN ] (manifest error) {path}: {err}")

    print()
    ok = (not errors and not db_errors and query_failures == 0 and diversity_failures == 0
          and not expand_errors and orchestrator_failures == 0)
    print("=" * 90)
    print("RESULT: " + ("PASS" if ok else "FAIL — see findings above"))
    print("=" * 90)
    return ok
