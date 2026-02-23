"""Tests for MCP server tool definitions."""
import pytest
import yellow_docs_mcp.server as server_mod
from yellow_docs_mcp.server import (
    _format_search_results,
    _format_page,
    _format_sections_tree,
    _format_api_methods,
    _format_code_examples,
)
from yellow_docs_mcp.parser import DocPage, DocSection, CodeBlock
from yellow_docs_mcp.search import SearchEngine, SearchResult


def _sample_result():
    return SearchResult(
        path="protocol/off-chain/channel-methods.mdx",
        category="protocol",
        page_title="Channel Management Methods",
        section_title="create_channel",
        text="Creates a new payment channel.",
        score=0.032,
        code_blocks=[{"language": "json", "code": '{"req": [1, "create_channel"]}'}],
    )


def _sample_page():
    return DocPage(
        path="learn/introduction/what-yellow-solves.mdx",
        category="learn",
        title="What Yellow Solves",
        description="Core problems solved",
        keywords=["state channels"],
        sections=[
            DocSection(title="Fragmentation", level=2, text="Chains are fragmented.", code_blocks=[]),
            DocSection(title="Speed", level=2, text="L1 is slow.", code_blocks=[
                CodeBlock(language="typescript", code="const x = 1;"),
            ]),
        ],
        raw_content="",
    )


def test_format_search_results():
    text = _format_search_results([_sample_result()])
    assert "create_channel" in text
    assert "Channel Management Methods" in text
    assert "protocol/off-chain/channel-methods.mdx" in text


def test_format_page():
    text = _format_page(_sample_page())
    assert "What Yellow Solves" in text
    assert "Source:" in text
    assert "Fragmentation" in text
    assert "Speed" in text


def test_format_sections_tree():
    pages = [_sample_page()]
    text = _format_sections_tree(pages)
    assert "learn" in text.lower()
    assert "Source: docs" in text
    assert "What Yellow Solves" in text


def test_format_sections_tree_source_filter():
    docs_page = _sample_page()
    nitro_page = _sample_page()
    nitro_page.source = "nitro"
    text = _format_sections_tree([docs_page, nitro_page], source="nitro")
    assert "Source: nitro" in text
    assert "Source: docs" not in text


def test_format_code_examples():
    pages = [_sample_page()]
    text = _format_code_examples(pages, "speed")
    assert "const x = 1" in text


def test_format_api_methods_source_filter():
    docs_page = _sample_page()
    docs_page.category = "protocol"
    docs_page.sections = [DocSection(title="create_channel", level=2, text="Create", code_blocks=[])]

    nitro_page = _sample_page()
    nitro_page.source = "nitro"
    nitro_page.category = "protocol"
    nitro_page.sections = [DocSection(title="close_channel", level=2, text="Close", code_blocks=[])]

    text = _format_api_methods([docs_page, nitro_page], source="nitro")
    assert "close_channel" in text
    assert "create_channel" not in text


def test_format_code_examples_limit():
    pages = [_sample_page(), _sample_page()]
    text = _format_code_examples(pages, "speed", max_examples=1)
    assert text.count("```typescript") == 1
    assert "more examples" in text


def test_ensure_initialized_falls_back_to_cache_when_sync_fails(monkeypatch):
    page = _sample_page()
    cached_engine = SearchEngine(enable_vectors=False)
    cached_engine.build_index([page])
    cached_index = cached_engine.get_index_data()

    class FailingRepo:
        def sync_repo(self):
            raise RuntimeError("network down")

        def needs_reindex(self):
            return False

        def parse_all_docs(self):
            return []

        def save_hash(self):
            return None

    monkeypatch.setattr(server_mod, "RepoManager", lambda: FailingRepo())
    monkeypatch.setattr(server_mod, "load_index", lambda: cached_index)
    server_mod._engine = SearchEngine(enable_vectors=False)
    server_mod._initialized = False

    server_mod._ensure_initialized()
    assert server_mod._initialized is True
    results = server_mod._engine.search("speed")
    assert len(results) > 0
