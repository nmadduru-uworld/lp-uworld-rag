"""Tree crawl + per-page Metadata-line tagging + cross-reference pass.

Discovery is a live crawl of two containers -- ``confluence.rootPageId`` (Technical Hub tree) and
``confluence.functionalRootPageId`` (the sibling Feature Hub tree) -- not a hand-maintained page-id
manifest. Every content page already carries everything needed to tag it in its own ``**Metadata**``
line. See the plan's Findings section for why a manifest drifts and this doesn't.
"""
from __future__ import annotations

import json
import logging
import re
from collections import deque
from pathlib import Path

from . import chunker, confluence_client
from .config import RagConfig

log = logging.getLogger(__name__)

_METADATA_KEY_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:")
_BACKTICK_RE = re.compile(r"`([^`]*)`")
_CONTROLLERS_SECTION_RE = re.compile(r"^##\s+Controllers\s*$", re.MULTILINE)
_H2_RE = re.compile(r"^##\s+\S", re.MULTILINE)
_H2_HEADING_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_H3_RE = re.compile(r"^###\s+(.+)$", re.MULTILINE)

# WS4 (token-economy plan): doc_types with rich multi-section bodies worth splitting small-to-big.
# Every other doc_type keeps today's one-node-per-page/section behavior (chunker.py's module
# docstring explains why -- they're already thin by authoring convention).
_SECTION_SPLIT_DOC_TYPES = frozenset({"endpoint", "db-collection"})
_VERB_TITLE_RE = re.compile(r"^(GET|POST|PUT|PATCH|DELETE)\s")
_BRACKET_TYPE_TITLE_RE = re.compile(r"\[[^\]]+\]\s*$")


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def parse_metadata_line(markdown: str) -> dict | None:
    """Parse the leading ``**Metadata** — key: `value` · key: `value` ...`` line into a dict.

    Returns None if no Metadata line is present -- a pure navigation/index page (``Repos``,
    ``Data Stores``, ``Features``, ``Artifacts``, a bare ``Repos/<repo>`` folder), which gets skipped
    rather than ingested as a chunk.
    """
    head = "\n".join(markdown.splitlines()[:3])
    if "**Metadata**" not in head:
        return None
    line = next((ln for ln in markdown.splitlines() if "**Metadata**" in ln), "")
    line = line.split("**Metadata**", 1)[1]
    line = re.sub(r"^\s*[—\-:]\s*", "", line)  # strip the leading "— "/"- " separator
    # markdownify escapes underscores as `\_` (a literal `_` would otherwise read as markdown
    # emphasis) -- undo it before parsing, or every underscored key (`endpoint_id`, `consumed_by`,
    # `served_by`) silently fails to match and its value is lost.
    line = line.replace("\\_", "_")
    fields: dict[str, str | list[str]] = {}
    for segment in re.split(r"\s*·\s*", line):
        key_match = _METADATA_KEY_RE.match(segment)
        if not key_match:
            continue
        # A list field (e.g. consumed_by) carries several backtick-wrapped values in one segment,
        # only the first preceded by "key:" -- collect every backtick value in the segment, not just
        # the one immediately after the key, or later list entries silently get dropped.
        values = _BACKTICK_RE.findall(segment)
        if not values:
            continue
        fields[key_match.group(1)] = values[0] if len(values) == 1 else values
    return fields or None


def classify(metadata: dict, title: str, parent_title: str | None) -> str | None:
    """doc_type priority table -- first match wins. See the plan's confluence_reader.py section."""
    if "docType" in metadata:
        return metadata["docType"]
    if "endpoint_id" in metadata:
        return "endpoint"
    # "store:"/"collection:" is the original hand-written convention; lp-datastore-docs
    # generated pages use the more precise "database:" plus "table:" (SQL) / "collection:"
    # (Mongo) -- both classify identically.
    if (("store" in metadata or "database" in metadata)
            and ("collection" in metadata or "table" in metadata)):
        return "db-collection"
    if "repoType" in metadata:
        return "repo-registry"
    if _BRACKET_TYPE_TITLE_RE.search(title):
        return "db-collection"
    if title.endswith("· Technical Hub"):
        return "technical-hub"
    if _VERB_TITLE_RE.match(title):
        return "endpoint"
    if parent_title == "Features":
        return "feature-hub"
    return None


def _crawl_from(cfg: RagConfig, root_page_id: str) -> list[dict]:
    """Breadth-first walk from ``root_page_id``. True Confluence tree edges only (each page has
    exactly one parent) -- no de-duplication needed, unlike the feature<->endpoint relationship,
    which is metadata, not a tree edge (see the plan's Document model section). A page whose title is
    in ``confluence.excludedTitles`` is skipped entirely -- not fetched, not chunked, and its own
    children are never visited (e.g. ``Artifacts``, which only links out to design-reference pages)."""
    excluded = set(cfg.confluence.excludedTitles)
    pages: list[dict] = []
    queue: deque[dict] = deque(
        p for p in confluence_client.get_children(cfg.confluence, root_page_id) if p["title"] not in excluded
    )
    while queue:
        page = queue.popleft()
        pages.append(page)
        children = confluence_client.get_children(cfg.confluence, page["id"])
        queue.extend(c for c in children if c["title"] not in excluded)
    return pages


def _split_page_sections(markdown: str) -> tuple[str, list[tuple[str, str]]] | None:
    """WS4: split an endpoint/db-collection page into ``(overview_text, [(heading, section_text), ...])``
    for small-to-big chunking. The overview folds in everything before the first ``##`` heading
    (the Metadata line) plus the ``## Purpose`` section itself if present, matching the Template
    v3 page's own "Purpose/summary lead ... chunked as the overview parent" rule; every other ``##``
    section becomes its own leaf.

    Returns ``None`` (whole-page fallback, unchanged from before WS4) when the page has no ``##``
    headings at all, or has only a Purpose section and nothing else worth splitting out.
    """
    headings = list(_H2_HEADING_RE.finditer(markdown))
    if not headings:
        return None

    overview_text = markdown[: headings[0].start()].strip()
    sections: list[tuple[str, str]] = []
    for i, h in enumerate(headings):
        start = h.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(markdown)
        sections.append((h.group(1).strip(), markdown[start:end].strip()))

    if sections and sections[0][0].lower() == "purpose":
        overview_text = f"{overview_text}\n\n## {sections[0][0]}\n\n{sections[0][1]}".strip()
        sections = sections[1:]

    return (overview_text, sections) if sections else None


def _split_controller_sections(markdown: str) -> list[tuple[str, str]]:
    """Split a repo-registry page's ``## Controllers`` section into (ControllerName, text) pairs."""
    m = _CONTROLLERS_SECTION_RE.search(markdown)
    if not m:
        return []
    rest = markdown[m.end():]
    next_h2 = _H2_RE.search(rest)
    section = rest[: next_h2.start()] if next_h2 else rest

    headers = list(_H3_RE.finditer(section))
    out: list[tuple[str, str]] = []
    for i, h in enumerate(headers):
        start = h.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(section)
        out.append((h.group(1).strip(), section[start:end].strip()))
    return out


def _build_tags(doc_type: str, metadata: dict) -> dict:
    if doc_type in ("feature-hub", "technical-hub"):
        return {"feature": _as_list(metadata.get("feature"))}
    if doc_type == "endpoint":
        repo = metadata.get("repo")
        controller = metadata.get("controller")
        return {
            "repo": repo,
            "feature": _as_list(metadata.get("feature")),
            "endpoint_id": metadata.get("endpoint_id"),
            "controller_id": f"ctrl::{repo}::{controller}" if repo and controller else None,
            # Forward reference to the data stores this endpoint reads (dependency-direction-correct
            # replacement for db-collection's consumed_by -- see the template's "Dependency direction"
            # section). Authored directly on the endpoint's Metadata line, not derived.
            "db_ids": _as_list(metadata.get("db_ids")),
        }
    if doc_type == "db-collection":
        # Alias mapping: generated pages say database:/table: (SQL) or database:/collection:
        # (Mongo); hand-written ones say store:/collection:. Same identity either way.
        store = metadata.get("store") or metadata.get("database")
        collection = metadata.get("collection") or metadata.get("table")
        return {
            "store": store,
            "type": metadata.get("type"),
            "collection": collection,
            "db_id": f"db::{store}::{collection}" if store and collection else None,
            "consumed_by": _as_list(metadata.get("consumed_by")),
        }
    return dict(metadata)


def crawl(cfg: RagConfig) -> list[dict]:
    """Breadth-first walk of both containers -- the Technical Hub tree (``rootPageId``, feeds
    docs_technical) and the sibling Functional Feature Hub tree (``functionalRootPageId``, feeds
    docs_functional). The two are disjoint subtrees under different root pages, so no de-duplication
    is needed between them -- see ``_crawl_from``."""
    return (_crawl_from(cfg, cfg.confluence.rootPageId)
            + _crawl_from(cfg, cfg.confluence.functionalRootPageId))


def read_all(cfg: RagConfig, pages: list[dict] | None = None) -> list:
    """Fetch bodies, classify, tag, chunk every page. Returns every ``TextNode`` to ingest.

    Always fetches every page's body -- at this space's current scale (a few dozen pages) that's a
    handful of seconds, not worth a fetch-skip cache. ``ingest.py`` gets its actual delta savings from
    comparing the resulting nodes' content-hash ids against what's already in Chroma, which is also
    strictly more robust than trusting Confluence's ``version`` number alone (it catches a
    cross-referenced field like a controller's ``used_by`` changing even when that controller's own
    page didn't). Pass ``pages`` (from a prior ``crawl()`` call) to avoid crawling twice in one run.
    """
    pages = pages if pages is not None else crawl(cfg)
    title_by_id = {p["id"]: p["title"] for p in pages}

    # Body cache keyed by Confluence version: at Data-Stores-catalog scale (hundreds of
    # pages) refetching every body each run dominates wall-clock; an unchanged version
    # number means an identical body, so reuse it. Written incrementally so an interrupted
    # run keeps its fetch progress. Gitignored local artifact.
    cache_path = cfg.root_path / ".ingest_body_cache.json"
    try:
        body_cache: dict = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        body_cache = {}

    def _save_cache() -> None:
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(body_cache), encoding="utf-8")
        tmp.replace(cache_path)

    nodes = []
    endpoint_records: list[dict] = []  # for the controller used_by cross-reference pass
    repo_registry_pages: list[tuple[dict, dict, dict]] = []

    cache_hits = 0
    for n, page in enumerate(pages, 1):
        # Progress heartbeat -- a silent fetch loop looks hung; flush so it's visible
        # through redirected/buffered stdout.
        if n % 25 == 0 or n == len(pages):
            print(f"[ingest]   fetched {n}/{len(pages)} pages ({cache_hits} from cache) …",
                  flush=True)
            _save_cache()
        cached = body_cache.get(page["id"])
        if cached is not None and page.get("version") is not None \
                and cached.get("version") == page["version"]:
            body = {"id": page["id"], "title": page["title"], "markdown": cached["markdown"]}
            cache_hits += 1
        else:
            body = confluence_client.get_page_body(cfg.confluence, page["id"])
            if body is not None and page.get("version") is not None:
                body_cache[page["id"]] = {"version": page["version"],
                                          "markdown": body["markdown"]}
        if body is None:
            continue
        metadata = parse_metadata_line(body["markdown"])
        if metadata is None:
            continue  # pure navigation/index page
        parent_title = title_by_id.get(page.get("parentId"))
        doc_type = classify(metadata, page["title"], parent_title)
        if doc_type is None:
            log.warning("page %s (%r) has a Metadata line but no doc_type matched -- skipping",
                        page["id"], page["title"])
            continue

        if doc_type == "repo-registry":
            # Produces no chunk of its own -- only its ## Controllers sections are ingested, once
            # `used_by` can be cross-referenced against every endpoint below.
            repo_registry_pages.append((page, metadata, body))
            continue

        tags = _build_tags(doc_type, metadata)
        tags["page_id"] = page["id"]
        if doc_type == "endpoint":
            # The page title is verbatim "<VERB> <route>" -- kept as its own field for readability
            # and as a required-field check (eval.py tier 1). The endpoint->db-collection join no
            # longer runs through route strings; it reads db_ids directly (see _build_tags above).
            tags["route"] = page["title"]

        page_title = body["title"] or page["title"]
        split = _split_page_sections(body["markdown"]) if doc_type in _SECTION_SPLIT_DOC_TYPES else None
        if split is not None:
            # WS4: overview (Purpose) parent + one leaf per remaining section, small-to-big --
            # every leaf carries the SAME tags as a whole-page node would (so eval.py's per-doc_type
            # checks, and stable-id lookups like endpoint_id, work identically whether a hit is the
            # overview or one of its sections), plus its own `parent_id` back to the overview.
            overview_text, sections = split
            overview_id = chunker.compute_id(doc_type, page["id"], page_title, overview_text)
            nodes.append(chunker.build_node(
                text=overview_text, title=page_title, key=page["id"], doc_type=doc_type, metadata=tags,
            ))
            for heading, section_text in sections:
                leaf_tags = {**tags, "parent_id": overview_id, "section": heading}
                nodes.append(chunker.build_node(
                    text=section_text, title=f"{page_title} — {heading}",
                    key=f'{page["id"]}::{heading}', doc_type=doc_type, metadata=leaf_tags,
                ))
        else:
            nodes.append(chunker.build_node(
                text=body["markdown"], title=page_title,
                key=page["id"], doc_type=doc_type, metadata=tags,
            ))

        if doc_type == "endpoint":
            endpoint_records.append({
                "endpoint_id": tags.get("endpoint_id"),
                "repo": tags.get("repo"),
                "controller": metadata.get("controller"),
            })

    for page, metadata, body in repo_registry_pages:
        repo = metadata.get("repo")
        for name, text in _split_controller_sections(body["markdown"]):
            used_by = [r["endpoint_id"] for r in endpoint_records
                       if r["repo"] == repo and r["controller"] == name and r["endpoint_id"]]
            tags = {
                "repo": repo,
                "controller_id": f"ctrl::{repo}::{name}",
                "used_by": used_by,
                "page_id": page["id"],
            }
            node = chunker.build_node(
                text=text, title=name, key=f'{page["id"]}::{name}',
                doc_type="controller-context", metadata=tags,
            )
            nodes.append(node)

    if cfg.confluence.manifestPath:
        _audit_manifest(cfg.confluence.manifestPath, {p["id"] for p in pages})

    # Prune cache entries for pages that no longer exist, then persist the final cache.
    current_ids = {p["id"] for p in pages}
    for stale in set(body_cache) - current_ids:
        del body_cache[stale]
    _save_cache()

    return nodes


def _collect_page_ids(obj) -> set[str]:
    """Recursively collect string values under any key ending in 'PageId' -- generic on purpose, no
    dependency on any particular manifest schema."""
    ids: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower().endswith("pageid") and isinstance(v, str):
                ids.add(v)
            ids |= _collect_page_ids(v)
    elif isinstance(obj, list):
        for item in obj:
            ids |= _collect_page_ids(item)
    return ids


def _audit_manifest(manifest_path: str, crawled_ids: set[str]) -> None:
    """Informational only -- never blocks ingest. See the plan's Optional manifest audit step."""
    path = Path(manifest_path)
    if not path.exists():
        log.info("manifest audit: %s not found -- skipping", manifest_path)
        return
    try:
        manifest_ids = _collect_page_ids(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("manifest audit: could not read %s (%s) -- skipping", manifest_path, exc)
        return
    stale = manifest_ids - crawled_ids
    if stale:
        log.warning("manifest audit: %d id(s) in %s not found in the live crawl: %s",
                    len(stale), manifest_path, ", ".join(sorted(stale)))
    else:
        log.info("manifest audit: every id in %s matches the live crawl", manifest_path)
