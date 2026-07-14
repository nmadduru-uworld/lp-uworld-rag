"""Domain-specific eval fixtures -- the concrete queries, expand samples, and orchestrator-routing
cases the generic harness in ``eval.py`` runs. Kept out of ``eval.py`` so the harness itself stays
domain-agnostic: its tier logic scans/exercises whatever is currently ingested, while these cases
hardcode the ids of today's corpus (the ``reports`` domain). A future repo/domain can supply its own
case list without touching the harness.

Ids are drawn from the live reports-domain corpus (see confluence-manifest.json / live Confluence).
Kept small and specific rather than exhaustive -- each one probes a distinct reasoning shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field


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


@dataclass
class OrchestratorCase:
    question: str
    expected_repos: list[str] | None  # None -> routing is expected to fall back (any repo set ok)
    expect_fallback: bool
    accept_files: list[str] = field(default_factory=list)  # basenames; any hit across routed repos passes
    informational: bool = False  # shown but never counted toward pass/fail -- see the fallback case below


# -- Tier 3: retrieval quality (deep-reasoning queries) -------------------------

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


# -- Tier 3 (diversity): a broad query should span multiple doc_types, not one --

DIVERSITY_CASES: list[DiversityCase] = [
    DiversityCase(
        "Give me the full picture of the Faculty-Led Customizable Report -- the business purpose, "
        "the technical index of its endpoints, and what data it's built on",
        required_doc_types=["feature-hub", "technical-hub", "endpoint", "db-collection"],
    ),
]


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


# -- Tier 5: orchestrator routing (docs -> repo -> code) -----------------------

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
