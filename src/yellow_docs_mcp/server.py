"""MCP server for Yellow Network documentation."""
from __future__ import annotations

import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from yellow_docs_mcp.indexer import RepoManager
from yellow_docs_mcp.parser import DocPage
from yellow_docs_mcp.search import SearchEngine, SearchResult
from yellow_docs_mcp.store import save_index, load_index

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
    if cached and not repo.needs_reindex():
        logger.info("Loading cached index...")
        _engine.load_index_data(cached)
        _initialized = True
        return

    logger.info("Syncing docs repo...")
    repo.sync_repo()
    logger.info("Parsing documents...")
    pages = repo.parse_all_docs()
    logger.info("Building search index...")
    _engine.build_index(pages)
    save_index(_engine.get_index_data())
    repo.save_hash()
    _initialized = True
    logger.info("Initialization complete: %d pages indexed", len(pages))


def _format_search_results(results: list[SearchResult]) -> str:
    if not results:
        return "No results found."
    parts = []
    for i, r in enumerate(results, 1):
        part = f"### Result {i}\n"
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
    parts.append(f"**Path:** `{page.path}` | **Category:** {page.category}\n")
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


def _format_sections_tree(pages: list[DocPage], category: str | None = None) -> str:
    filtered = pages
    if category:
        filtered = [p for p in pages if p.category == category]
    if not filtered:
        return f"No pages found for category: {category}" if category else "No pages indexed."
    by_category: dict[str, list[DocPage]] = {}
    for page in filtered:
        cat = page.category or "root"
        by_category.setdefault(cat, []).append(page)
    parts = ["# Documentation Structure\n"]
    for cat in sorted(by_category.keys()):
        parts.append(f"\n## {cat}/\n")
        for page in sorted(by_category[cat], key=lambda p: p.path):
            desc = f" - {page.description}" if page.description else ""
            parts.append(f"- **{page.title}** (`{page.path}`){desc}")
            for section in page.sections:
                indent = "  " * section.level
                parts.append(f"{indent}- {section.title}")
    return "\n".join(parts)


def _format_api_methods(pages: list[DocPage], method: str | None = None) -> str:
    api_pages = [p for p in pages if p.category in ("protocol", "api-reference")]
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
            part += f"**Source:** `{page.path}`\n\n"
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


def _format_code_examples(pages: list[DocPage], topic: str, language: str = "typescript") -> str:
    topic_lower = topic.lower()
    lang_lower = language.lower()
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
    for page, section, cb in results:
        parts.append(f"## From: {page.title} > {section.title}")
        parts.append(f"**Path:** `{page.path}`\n")
        parts.append(f"```{cb.language}\n{cb.code}\n```\n")
    return "\n".join(parts)


@mcp.tool()
async def search_docs(query: str, limit: int = 5, category: str | None = None) -> str:
    """Search Yellow Network documentation using hybrid semantic + keyword search.

    Args:
        query: Search query (e.g. "how do state channels work", "create_channel params")
        limit: Max number of results (default 5)
        category: Optional filter by doc category: "learn", "protocol", "api-reference", "guides", "manuals", "build", "tutorials"
    """
    _ensure_initialized()
    results = _engine.search(query, limit=limit, category=category)
    return _format_search_results(results)


@mcp.tool()
async def get_page(path: str) -> str:
    """Get a full documentation page by path or topic name.

    Args:
        path: Doc path (e.g. "protocol/off-chain/channel-methods.mdx") or topic name (e.g. "state channels", "authentication")
    """
    _ensure_initialized()
    page = _engine.get_page_by_path(path)
    if page is None:
        results = _engine.search(path, limit=1)
        if results:
            page = _engine.get_page_by_path(results[0].path)
    if page is None:
        return f"Page not found: {path}. Use list_sections() to see available pages."
    return _format_page(page)


@mcp.tool()
async def list_sections(category: str | None = None) -> str:
    """Browse documentation structure. Shows all pages and their sections.

    Args:
        category: Optional filter: "learn", "protocol", "api-reference", "guides", "manuals", "build", "tutorials"
    """
    _ensure_initialized()
    return _format_sections_tree(_engine.get_all_pages(), category)


@mcp.tool()
async def get_api_reference(method: str | None = None) -> str:
    """Look up Yellow Network protocol RPC methods and SDK API.

    Args:
        method: Optional method name (e.g. "create_channel", "auth_request", "NitroliteRPC"). If omitted, lists all methods.
    """
    _ensure_initialized()
    return _format_api_methods(_engine.get_all_pages(), method)


@mcp.tool()
async def get_code_examples(topic: str, language: str = "typescript") -> str:
    """Find code examples from the docs matching a topic.

    Args:
        topic: Topic to find examples for (e.g. "connect to clearnode", "create channel", "authentication")
        language: Programming language filter (default "typescript"). Use "all" for any language.
    """
    _ensure_initialized()
    return _format_code_examples(_engine.get_all_pages(), topic, language)


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
