"""Tests for MCP server tool definitions."""
import pytest
from yellow_docs_mcp.server import (
    _format_search_results,
    _format_page,
    _format_sections_tree,
    _format_api_methods,
    _format_code_examples,
)
from yellow_docs_mcp.parser import DocPage, DocSection, CodeBlock
from yellow_docs_mcp.search import SearchResult


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
    assert "Fragmentation" in text
    assert "Speed" in text


def test_format_sections_tree():
    pages = [_sample_page()]
    text = _format_sections_tree(pages)
    assert "learn" in text.lower()
    assert "What Yellow Solves" in text


def test_format_code_examples():
    pages = [_sample_page()]
    text = _format_code_examples(pages, "speed")
    assert "const x = 1" in text
