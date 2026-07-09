"""Confluence Cloud REST API v2 client -- Basic Auth via the api.atlassian.com gateway, no MCP/agent
dependency.

Standalone by design: ``python -m lp_uworld_rag ingest`` must run unattended from a terminal or CI,
where no live agent session (and therefore no Confluence MCP tool) exists.

Two things had to be verified live against this org's Atlassian site, both now confirmed:
  * Auth is HTTP Basic (email + token) -- the normal mechanism for a personal API token.
  * But it must go through ``https://api.atlassian.com/ex/confluence/{cloudId}/...``
    (``cfg.api_base()``), never the site-direct domain (``cfg.baseUrl``). The identical Basic Auth
    call against the site-direct domain returns a 401 + ``WWW-Authenticate: OAuth`` (this org blocks
    personal-token Basic Auth on the site domain itself); the same call against the gateway succeeds.
"""
from __future__ import annotations

import logging
from typing import Any

import requests
from markdownify import ATX, markdownify

from .config import ConfluenceConfig

log = logging.getLogger(__name__)


def _auth(cfg: ConfluenceConfig) -> tuple[str, str]:
    return (cfg.email(), cfg.api_token())


def get_children(cfg: ConfluenceConfig, page_id: str) -> list[dict[str, Any]]:
    """Direct children of ``page_id`` -- id/title/parentId/version, no body. Paginated via cursor."""
    results: list[dict[str, Any]] = []
    url = f"{cfg.api_base()}/wiki/api/v2/pages/{page_id}/children"
    params: dict[str, Any] = {"limit": 100}
    while url:
        resp = requests.get(url, params=params, auth=_auth(cfg), timeout=30)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("results", []):
            # The /children response has no `parentId` field of its own -- it's implicit (we asked
            # for THIS page's children), so set it directly rather than reading a key that isn't there.
            version = item.get("version") or {}
            results.append({
                "id": item["id"],
                "title": item.get("title", ""),
                "parentId": page_id,
                "version": version.get("number"),
            })
        next_link = (data.get("_links") or {}).get("next")
        if next_link:
            # `_links.next` is a path relative to the site's own /wiki root (e.g. from `_links.base`
            # in the response), not the api.atlassian.com gateway -- rebuild against api_base().
            url = next_link if next_link.startswith("http") else f"{cfg.api_base()}{next_link.removeprefix('/wiki')}"
            params = {}  # cursor is already encoded into next_link
        else:
            url = None
    # Fallback: if the v2 response omitted version numbers, fetch each page's own metadata for it.
    if results and any(r["version"] is None for r in results):
        for r in results:
            if r["version"] is None:
                r["version"] = _get_version_only(cfg, r["id"])
    return results


def _get_version_only(cfg: ConfluenceConfig, page_id: str) -> int | None:
    """Cheap no-body fallback for a single page's version number."""
    resp = requests.get(f"{cfg.api_base()}/wiki/api/v2/pages/{page_id}", auth=_auth(cfg), timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return (resp.json().get("version") or {}).get("number")


def get_page_body(cfg: ConfluenceConfig, page_id: str) -> dict[str, Any] | None:
    """Full page body as markdown. Returns None on 404 (log + skip, never raise).

    ``body-format=export_view`` is confirmed live against this org's gateway (root, endpoint, DB
    collection, technical hub, and feature hub pages all succeeded on the first try) -- no retry
    chain across body-format values or a v1 fallback needed.
    """
    resp = requests.get(
        f"{cfg.api_base()}/wiki/api/v2/pages/{page_id}",
        params={"body-format": "export_view"},
        auth=_auth(cfg),
        timeout=30,
    )
    if resp.status_code == 404:
        log.warning("page %s not found (404) -- skipping", page_id)
        return None
    resp.raise_for_status()
    data = resp.json()
    html = ((data.get("body") or {}).get("export_view") or {}).get("value", "")
    # ATX (`#`, `##`, `###`) throughout -- markdownify's default is setext style for H1/H2 (an
    # underline of `===`/`---`), which silently breaks every ``##``/``###``-anchored regex in
    # confluence_reader.py (e.g. finding the ``## Controllers`` section) since setext has no H3+
    # equivalent, so real pages end up a mix of two heading syntaxes.
    return {"id": page_id, "title": data.get("title", ""),
            "markdown": markdownify(html, heading_style=ATX).strip()}
