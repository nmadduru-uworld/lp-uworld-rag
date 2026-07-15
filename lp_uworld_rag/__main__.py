"""CLI entry point.

    python -m lp_uworld_rag ingest [--full] [--strict]
    python -m lp_uworld_rag query "<text>" [--collection functional|technical] [--top-k N] [--no-siblings]
    python -m lp_uworld_rag expand <id>
    python -m lp_uworld_rag status
    python -m lp_uworld_rag eval
    python -m lp_uworld_rag validate
    python -m lp_uworld_rag mcp
    python -m lp_uworld_rag ingest-code (--repo REPO | --all) [--full]
    python -m lp_uworld_rag deep-query "<text>" [--repo REPO ...] [--top-k-docs N] [--top-k-code N]
    python -m lp_uworld_rag query-code "<text>" [--repo REPO] [--top-k N] [--file-hint PATH ...]
    python -m lp_uworld_rag repos [--validate]
    python -m lp_uworld_rag validate-citations

Heavy imports are deferred into each handler so ``--help`` and config loading work without
llama-index/chromadb/mcp installed.
"""
from __future__ import annotations

import argparse
import json
import sys

# Confluence content routinely carries non-ASCII characters (arrows, em dashes, curly quotes);
# Windows consoles default to a legacy codepage (cp1252) that can't encode them, so a plain
# print(json.dumps(..., ensure_ascii=False)) crashes there -- force UTF-8 on stdout/stderr instead
# of falling back to ensure_ascii=True, which would just replace the crash with unreadable \uXXXX
# escapes for anyone actually reading this output at a glance.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def _load():
    """Load config -- imported lazily (like every handler) so ``--help`` works without deps."""
    from .config import load_config
    return load_config()


def _registry(cfg):
    """Build the repo registry from config -- the two lines the four repo-facing handlers share."""
    from .repo_registry import RepoRegistry
    return RepoRegistry.from_config(cfg)


def _cmd_ingest(args: argparse.Namespace) -> int:
    from .config import load_config
    from .ingest import run_ingest

    ok = run_ingest(load_config(), full=args.full, strict=args.strict)
    return 0 if ok else 1


def _cmd_query(args: argparse.Namespace) -> int:
    from .config import load_config
    from .retrieval.docs_index import query as run_query

    result = run_query(load_config(), args.text, collection=args.collection, top_k=args.top_k,
                        include_siblings=not args.no_siblings)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _cmd_expand(args: argparse.Namespace) -> int:
    from .config import load_config
    from .retrieval.docs_index import expand as run_expand

    print(json.dumps(run_expand(load_config(), args.id), indent=2, ensure_ascii=False))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    try:
        from .config import load_config
        from .retrieval.docs_index import status as run_status
    except ImportError as exc:  # deps not installed yet -- scaffold-friendly
        print(f"lp-uworld-rag: dependencies not installed ({exc}). Run `pip install -e .` first.",
              file=sys.stderr)
        return 0
    print(run_status(load_config()))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .config import load_config
    from .eval import run as run_eval

    ok = run_eval(load_config())
    return 0 if ok else 1


def _cmd_validate(args: argparse.Namespace) -> int:
    """Metadata-integrity check only (eval tier 1), against whatever is currently ingested --
    no re-ingest, no retrieval-model load. The same check ``ingest --strict`` gates on."""
    from .config import load_config
    from .eval import check_metadata

    findings = check_metadata(load_config())
    errors = [f for f in findings if f.severity == "error"]
    warnings = [f for f in findings if f.severity == "warn"]
    if not findings:
        print("no issues found")
    for f in findings:
        print(f"[{f.severity.upper():5}] {f.where}: {f.message}")
    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


def _cmd_mcp(args: argparse.Namespace) -> int:
    from .mcp_server import serve
    serve()
    return 0


def _cmd_ingest_code(args: argparse.Namespace) -> int:
    """Ingest one repo's (or every configured repo's) code into its own code_stores/<repoKey> index
    via the shared engine (repo_ingest), pointed at the checkout path in config.json."""
    from .config import load_config
    from .repo_ingest.pipeline import run_code_ingest
    from .repo_ingest.spec import discover_repo_keys, load_job

    cfg = load_config()
    if args.all:
        keys = discover_repo_keys(cfg.repos_dir())
        if not keys:
            print("no repo specs found under repos/*/ingest.json", file=sys.stderr)
            return 1
    else:
        keys = [args.repo]

    failed = False
    for key in keys:
        try:
            job = load_job(cfg, key)
        except Exception as exc:
            print(f"[ingest-code] {key}: cannot start -- {exc}", file=sys.stderr)
            failed = True
            continue
        run_code_ingest(job, full=args.full)
    return 1 if failed else 0


def _cmd_deep_query(args: argparse.Namespace) -> int:
    from .retrieval.orchestrator import deep_query

    cfg = _load()
    registry = _registry(cfg)
    result = deep_query(cfg, registry, args.text, top_k_docs=args.top_k_docs,
                         top_k_code=args.top_k_code, repos=args.repo)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _cmd_query_code(args: argparse.Namespace) -> int:
    from .retrieval.orchestrator import query_code

    cfg = _load()
    registry = _registry(cfg)
    result = query_code(registry, args.text, repo=args.repo, file_hints=args.file_hint,
                         top_k=args.top_k)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _cmd_repos(args: argparse.Namespace) -> int:
    cfg = _load()
    registry = _registry(cfg)
    ok = True

    if not args.validate:
        for key in registry.repo_keys():
            m = registry.manifests[key]
            print(f"{key} | {m.displayName} | backend={registry.backend(key)} | {m.manifest_path}")
        for path, err in registry.errors.items():
            print(f"[FAIL] (manifest error) {path}: {err}", file=sys.stderr)
            ok = False
        return 0 if ok else 1

    for key in registry.repo_keys():
        result = registry.validate(key)
        print(f"[{'PASS' if result['ok'] else 'FAIL'}] {key} (backend={result.get('backend', '?')})")
        for step in result["steps"]:
            detail = "" if step["ok"] else f" -- {step['error']}"
            print(f"    {step['step']}: {'ok' if step['ok'] else 'FAIL'}{detail}")
        if result.get("known_risk"):
            print(f"    known_risk: {result['known_risk']}" +
                  (f" (tracked: {result['tracked_in']})" if result.get("tracked_in") else ""))
        if not result["ok"]:
            ok = False
    for path, err in registry.errors.items():
        print(f"[FAIL] (manifest error) {path}: {err}")
        ok = False
    return 0 if ok else 1


def _cmd_validate_citations(args: argparse.Namespace) -> int:
    from .retrieval.orchestrator import validate_citations

    cfg = _load()
    registry = _registry(cfg)
    stale = validate_citations(cfg, registry)
    print(json.dumps(stale, indent=2, ensure_ascii=False))
    return 1 if stale else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lp_uworld_rag",
        description="Shared functional/technical RAG for the UWorld LP Confluence knowledge base.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Crawl + chunk + embed + upsert into Chroma.")
    p_ingest.add_argument("--full", action="store_true",
                           help="Ignore delta state and rebuild both collections from scratch.")
    p_ingest.add_argument("--strict", action="store_true",
                           help="Exit non-zero if the post-ingest metadata-integrity check finds "
                                "any error-severity finding (CI-ready gate).")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_query = sub.add_parser("query", help="Run a retrieval query (for testing).")
    p_query.add_argument("text", help="Natural-language question.")
    p_query.add_argument("--collection", choices=["functional", "technical"], default=None)
    p_query.add_argument("--top-k", type=int, default=None)
    p_query.add_argument("--no-siblings", action="store_true",
                          help="Skip resolving linked nodes as citations (narrowest response).")
    p_query.set_defaults(func=_cmd_query)

    p_expand = sub.add_parser("expand", help="Fetch a stable id's or page_id's full chunk content.")
    p_expand.add_argument("id", help="e.g. ep::reports::GetFacultyLedGroupPerformance")
    p_expand.set_defaults(func=_cmd_expand)

    p_status = sub.add_parser("status", help="Show indexed chunk counts by collection/doc_type.")
    p_status.set_defaults(func=_cmd_status)

    p_eval = sub.add_parser("eval", help="Run metadata integrity + retrieval quality checks.")
    p_eval.set_defaults(func=_cmd_eval)

    p_validate = sub.add_parser("validate", help="Metadata-integrity check only (eval tier 1), no re-ingest.")
    p_validate.set_defaults(func=_cmd_validate)

    p_mcp = sub.add_parser("mcp", help="Run as an MCP server over stdio.")
    p_mcp.set_defaults(func=_cmd_mcp)

    p_ingest_code = sub.add_parser(
        "ingest-code", help="Ingest a repo's code into its own index via the shared engine.")
    ic_group = p_ingest_code.add_mutually_exclusive_group(required=True)
    ic_group.add_argument("--repo", help="repoKey of a repo with a repos/<repoKey>/ingest.json spec.")
    ic_group.add_argument("--all", action="store_true", help="Ingest every configured repo.")
    p_ingest_code.add_argument("--full", action="store_true",
                                help="Delete and rebuild the collection instead of delta-syncing.")
    p_ingest_code.set_defaults(func=_cmd_ingest_code)

    p_deep_query = sub.add_parser("deep-query", help="Docs -> route -> per-repo code retrieval.")
    p_deep_query.add_argument("text", help="Natural-language question.")
    p_deep_query.add_argument("--repo", action="append", default=None,
                               help="Pin routing to this repo key (repeatable); omit to derive from doc hits.")
    p_deep_query.add_argument("--top-k-docs", type=int, default=None)
    p_deep_query.add_argument("--top-k-code", type=int, default=None)
    p_deep_query.set_defaults(func=_cmd_deep_query)

    p_query_code = sub.add_parser("query-code", help="Query one or every registered repo's code-RAG directly.")
    p_query_code.add_argument("text", help="Natural-language question.")
    p_query_code.add_argument("--repo", default=None, help="Registered repo key; omit to query all.")
    p_query_code.add_argument("--top-k", type=int, default=None)
    p_query_code.add_argument("--file-hint", action="append", default=None,
                               help="Advisory file path/symbol to rank-boost (repeatable).")
    p_query_code.set_defaults(func=_cmd_query_code)

    p_repos = sub.add_parser("repos", help="List registered repos, or validate their conformance.")
    p_repos.add_argument("--validate", action="store_true",
                          help="Launch each repo's MCP server and probe query_rag/rag_status for contract conformance.")
    p_repos.set_defaults(func=_cmd_repos)

    p_validate_citations = sub.add_parser(
        "validate-citations", help="Check every File.cs:line doc citation against its repo's checkout + code-RAG.")
    p_validate_citations.set_defaults(func=_cmd_validate_citations)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
