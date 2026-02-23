"""MCP server for Yellow Network documentation."""
from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from yellow_docs_mcp.indexer import RepoManager
from yellow_docs_mcp.parser import DocPage
from yellow_docs_mcp.search import SearchEngine, SearchResult
from yellow_docs_mcp.store import save_index, load_index
from yellow_docs_mcp.telemetry import log_metric, timed_metric

logger = logging.getLogger(__name__)

mcp = FastMCP("yellow-docs")

_engine = SearchEngine()
_initialized = False


def _ensure_initialized() -> None:
    global _initialized
    if _initialized:
        return

    repo = RepoManager()
    cached = load_index()
    with timed_metric(logger, "server.initialize"):
        content_changed = True
        try:
            logger.info("Syncing docs repos...")
            content_changed = repo.sync_repo()
        except Exception as exc:
            if cached:
                logger.warning("Repo sync failed; using cached index: %s", exc)
                _engine.load_index_data(cached)
                _initialized = True
                log_metric(logger, "server.initialize_from_cache", reason="sync_failed")
                return
            raise

        if cached and not content_changed and not repo.needs_reindex():
            logger.info("Loading cached index...")
            _engine.load_index_data(cached)
            _initialized = True
            log_metric(logger, "server.initialize_from_cache", reason="no_changes")
            return

        logger.info("Parsing documents...")
        pages = repo.parse_all_docs()
        logger.info("Building search index...")
        _engine.build_index(pages, previous_index=cached)
        save_index(_engine.get_index_data())
        repo.save_hash()
        _initialized = True
        log_metric(logger, "server.initialize_reindexed", page_count=len(pages))


def _format_search_results(results: list[SearchResult]) -> str:
    if not results:
        return "No results found."
    parts = []
    for i, r in enumerate(results, 1):
        part = f"### Result {i}\n"
        part += f"**Source:** {r.source}\n"
        part += f"**Page:** {r.page_title}\n"
        part += f"**Section:** {r.section_title}\n"
        part += f"**Path:** `{r.path}`\n"
        part += f"**Category:** {r.category}\n"
        part += f"**Score:** {r.score:.4f}\n\n"
        part += r.text + "\n"
        if r.code_blocks:
            for cb in r.code_blocks:
                part += f"\n```{cb['language']}\n{cb['code']}\n```\n"
        parts.append(part)
    return "\n---\n\n".join(parts)


def _format_page(page: DocPage) -> str:
    parts = [f"# {page.title}\n"]
    if page.description:
        parts.append(f"*{page.description}*\n")
    parts.append(f"**Source:** {page.source} | **Path:** `{page.path}` | **Category:** {page.category}\n")
    if page.keywords:
        parts.append(f"**Keywords:** {', '.join(page.keywords)}\n")
    parts.append("")
    for section in page.sections:
        prefix = "#" * min(section.level + 1, 4)
        parts.append(f"{prefix} {section.title}\n")
        parts.append(section.text + "\n")
        for cb in section.code_blocks:
            parts.append(f"```{cb.language}\n{cb.code}\n```\n")
    return "\n".join(parts)


def _format_sections_tree(
    pages: list[DocPage],
    category: str | None = None,
    source: str | None = None,
) -> str:
    filtered = pages
    if source:
        filtered = [p for p in filtered if p.source == source]
    if category:
        filtered = [p for p in filtered if p.category == category]
    if not filtered:
        if category and source:
            return f"No pages found for source '{source}' and category '{category}'."
        if source:
            return f"No pages found for source: {source}"
        if category:
            return f"No pages found for category: {category}"
        return "No pages indexed."

    by_source: dict[str, dict[str, list[DocPage]]] = {}
    for page in filtered:
        src = page.source or "docs"
        cat = page.category or "root"
        by_source.setdefault(src, {}).setdefault(cat, []).append(page)

    parts = ["# Documentation Structure\n"]
    for src in sorted(by_source.keys()):
        parts.append(f"\n## Source: {src}\n")
        for cat in sorted(by_source[src].keys()):
            parts.append(f"\n### {cat}/\n")
            for page in sorted(by_source[src][cat], key=lambda p: p.path):
                desc = f" - {page.description}" if page.description else ""
                parts.append(f"- **{page.title}** (`{page.path}`){desc}")
                for section in page.sections:
                    indent = "  " * section.level
                    parts.append(f"{indent}- {section.title}")
    return "\n".join(parts)


def _format_api_methods(
    pages: list[DocPage],
    method: str | None = None,
    source: str | None = None,
) -> str:
    api_pages = [p for p in pages if p.category in ("protocol", "api-reference")]
    if source:
        api_pages = [p for p in api_pages if p.source == source]
    if method:
        results = []
        method_lower = method.lower().replace("_", "").replace("-", "")
        for page in api_pages:
            for section in page.sections:
                section_lower = section.title.lower().replace("_", "").replace("-", "")
                if method_lower in section_lower or section_lower in method_lower:
                    results.append((page, section))
        if not results:
            return f"No API method found matching: {method}"
        parts = []
        for page, section in results:
            part = f"## {section.title}\n"
            part += f"**Source:** {page.source}\n"
            part += f"**Path:** `{page.path}`\n\n"
            part += section.text + "\n"
            for cb in section.code_blocks:
                part += f"\n```{cb.language}\n{cb.code}\n```\n"
            parts.append(part)
        return "\n---\n\n".join(parts)
    else:
        parts = ["# API Methods\n"]
        for page in sorted(api_pages, key=lambda p: p.path):
            parts.append(f"\n## {page.title} (`{page.path}`)\n")
            for section in page.sections:
                has_code = " [code]" if section.code_blocks else ""
                parts.append(f"- `{section.title}`{has_code}")
        return "\n".join(parts)


def _format_code_examples(
    pages: list[DocPage],
    topic: str,
    language: str = "typescript",
    max_examples: int = 5,
    source: str | None = None,
) -> str:
    topic_lower = topic.lower()
    lang_lower = language.lower()
    max_examples = max(1, min(max_examples, 25))
    if source:
        pages = [p for p in pages if p.source == source]
    results = []
    for page in pages:
        for section in page.sections:
            if topic_lower in section.title.lower() or topic_lower in section.text.lower():
                for cb in section.code_blocks:
                    if cb.language.lower() == lang_lower or language == "all":
                        results.append((page, section, cb))
    if not results:
        for page in pages:
            for section in page.sections:
                if topic_lower in section.title.lower() or topic_lower in section.text.lower():
                    for cb in section.code_blocks:
                        results.append((page, section, cb))
    if not results:
        return f"No code examples found for topic: {topic}"
    parts = [f"# Code Examples: {topic}\n"]
    for page, section, cb in results[:max_examples]:
        parts.append(f"## From: {page.title} > {section.title}")
        parts.append(f"**Source:** {page.source}")
        parts.append(f"**Path:** `{page.path}`\n")
        parts.append(f"```{cb.language}\n{cb.code}\n```\n")
    if len(results) > max_examples:
        parts.append(
            f"...and {len(results) - max_examples} more examples. "
            "Narrow the topic or language to reduce output."
        )
    return "\n".join(parts)


@mcp.tool()
async def search_docs(
    query: str,
    limit: int = 5,
    category: str | None = None,
    source: str | None = None,
) -> str:
    """Search Yellow Network documentation using hybrid semantic + keyword search.

    Args:
        query: Search query (e.g. "how do state channels work", "create_channel params")
        limit: Max number of results (default 5)
        category: Optional filter by doc category: "learn", "protocol", "api-reference", "guides", "manuals", "build", "tutorials"
        source: Optional source repo filter (e.g. "docs", "clearsync", "docs-gitbook", "nitro", "release-process")
    """
    _ensure_initialized()
    limit = max(1, min(limit, 20))
    with timed_metric(
        logger,
        "tool.search_docs",
        query_len=len(query),
        limit=limit,
        category=category or "all",
        source=source or "all",
    ):
        results = _engine.search(query, limit=limit, category=category, source=source)
    return _format_search_results(results)


@mcp.tool()
async def get_page(path: str, source: str | None = None) -> str:
    """Get a full documentation page by path or topic name.

    Args:
        path: Doc path (e.g. "protocol/off-chain/channel-methods.mdx") or topic name (e.g. "state channels", "authentication")
        source: Optional source repo filter (e.g. "docs", "clearsync", "docs-gitbook", "nitro", "release-process")
    """
    _ensure_initialized()
    with timed_metric(logger, "tool.get_page", path=path, source=source or "all"):
        page = _engine.get_page_by_path(path, source=source)
        if page is None:
            results = _engine.search(path, limit=1, source=source)
            if results:
                page = _engine.get_page_by_path(results[0].path, source=results[0].source)
    if page is None:
        return f"Page not found: {path}. Use list_sections() to see available pages."
    return _format_page(page)


@mcp.tool()
async def list_sections(category: str | None = None, source: str | None = None) -> str:
    """Browse documentation structure. Shows all pages and their sections.

    Args:
        category: Optional filter: "learn", "protocol", "api-reference", "guides", "manuals", "build", "tutorials"
        source: Optional source repo filter (e.g. "docs", "clearsync", "docs-gitbook", "nitro", "release-process")
    """
    _ensure_initialized()
    with timed_metric(
        logger,
        "tool.list_sections",
        category=category or "all",
        source=source or "all",
    ):
        return _format_sections_tree(_engine.get_all_pages(), category, source)


@mcp.tool()
async def get_api_reference(method: str | None = None, source: str | None = None) -> str:
    """Look up Yellow Network protocol RPC methods and SDK API.

    Args:
        method: Optional method name (e.g. "create_channel", "auth_request", "NitroliteRPC"). If omitted, lists all methods.
        source: Optional source repo filter (e.g. "docs", "docs-gitbook")
    """
    _ensure_initialized()
    with timed_metric(
        logger,
        "tool.get_api_reference",
        method=method or "all",
        source=source or "all",
    ):
        return _format_api_methods(_engine.get_all_pages(), method, source)


@mcp.tool()
async def get_code_examples(
    topic: str,
    language: str = "typescript",
    limit: int = 5,
    source: str | None = None,
) -> str:
    """Find code examples from the docs matching a topic.

    Args:
        topic: Topic to find examples for (e.g. "connect to clearnode", "create channel", "authentication")
        language: Programming language filter (default "typescript"). Use "all" for any language.
        limit: Max number of examples to return (default 5, max 25)
        source: Optional source repo filter (e.g. "docs", "clearsync", "docs-gitbook", "nitro", "release-process")
    """
    _ensure_initialized()
    with timed_metric(
        logger,
        "tool.get_code_examples",
        topic=topic,
        language=language,
        limit=limit,
        source=source or "all",
    ):
        return _format_code_examples(
            _engine.get_all_pages(),
            topic,
            language,
            max_examples=limit,
            source=source,
        )


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
