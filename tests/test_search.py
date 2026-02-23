"""Tests for hybrid search engine."""
from yellow_docs_mcp.parser import CodeBlock, DocPage, DocSection
from yellow_docs_mcp.search import SearchEngine, SearchResult


def _make_pages() -> list[DocPage]:
    return [
        DocPage(
            path="learn/introduction/what-yellow-solves.mdx", category="learn",
            title="What Yellow Solves", description="Core problems Yellow Network addresses",
            keywords=["state channels", "scaling"],
            sections=[
                DocSection(title="The Fragmentation Problem", level=2,
                    text="Today's blockchain ecosystem is fragmented across multiple chains. Yellow Network unifies liquidity.",
                    code_blocks=[]),
                DocSection(title="The Speed Problem", level=2,
                    text="Traditional L1 transactions are slow. State channels provide instant finality.",
                    code_blocks=[CodeBlock(language="typescript", code="const client = new NitroliteClient();")]),
            ], raw_content="",
        ),
        DocPage(
            path="protocol/off-chain/channel-methods.mdx", category="protocol",
            title="Channel Management Methods", description="RPC methods for channel operations",
            keywords=["channels", "RPC"],
            sections=[
                DocSection(title="create_channel", level=2,
                    text="Creates a new payment channel between two participants. Requires a funded custody contract.",
                    code_blocks=[CodeBlock(language="json", code='{"req": [1, "create_channel", {}]}')]),
                DocSection(title="close_channel", level=2,
                    text="Closes an existing channel and settles balances on-chain.", code_blocks=[]),
            ], raw_content="",
        ),
        DocPage(
            path="api-reference/index.md", category="api-reference",
            title="SDK API Reference", description="TypeScript SDK types and methods",
            keywords=["SDK", "API", "TypeScript"],
            sections=[
                DocSection(title="NitroliteRPC", level=2,
                    text="The main RPC client class for interacting with ClearNode WebSocket endpoints.",
                    code_blocks=[CodeBlock(language="typescript", code="class NitroliteRPC { connect(url: string): void }")]),
            ], raw_content="", source="nitro",
        ),
    ]


def test_search_engine_build():
    engine = SearchEngine(enable_vectors=False)
    engine.build_index(_make_pages())
    assert engine.is_ready()


def test_search_returns_results():
    engine = SearchEngine(enable_vectors=False)
    engine.build_index(_make_pages())
    results = engine.search("state channels scaling")
    assert len(results) > 0
    assert isinstance(results[0], SearchResult)


def test_search_relevance():
    engine = SearchEngine(enable_vectors=False)
    engine.build_index(_make_pages())
    results = engine.search("create_channel")
    assert results[0].path == "protocol/off-chain/channel-methods.mdx"


def test_search_category_filter():
    engine = SearchEngine(enable_vectors=False)
    engine.build_index(_make_pages())
    results = engine.search("channel", category="protocol")
    for r in results:
        assert r.category == "protocol"


def test_search_limit():
    engine = SearchEngine(enable_vectors=False)
    engine.build_index(_make_pages())
    results = engine.search("channel", limit=1)
    assert len(results) <= 1


def test_search_result_has_content():
    engine = SearchEngine(enable_vectors=False)
    engine.build_index(_make_pages())
    results = engine.search("fragmentation")
    assert len(results) > 0
    assert results[0].text
    assert results[0].section_title


def test_search_bm25_only_mode():
    engine = SearchEngine(enable_vectors=False)
    engine.build_index(_make_pages())
    results = engine.search("create_channel")
    assert len(results) > 0
    assert results[0].path == "protocol/off-chain/channel-methods.mdx"


def test_search_source_filter():
    engine = SearchEngine(enable_vectors=False)
    engine.build_index(_make_pages())
    results = engine.search("nitrolite", source="nitro")
    assert len(results) > 0
    assert all(r.source == "nitro" for r in results)


def test_search_vector_failure_falls_back(monkeypatch):
    engine = SearchEngine(enable_vectors=True)
    monkeypatch.setattr(engine, "_load_model", lambda: False)
    engine.build_index(_make_pages())
    results = engine.search("create_channel")
    assert len(results) > 0
    assert results[0].path == "protocol/off-chain/channel-methods.mdx"


def test_search_chunk_splitting(monkeypatch):
    monkeypatch.setenv("YELLOW_DOCS_MAX_CHUNK_CHARS", "80")
    monkeypatch.setenv("YELLOW_DOCS_CHUNK_OVERLAP_CHARS", "20")
    engine = SearchEngine(enable_vectors=False)

    pages = [
        DocPage(
            path="learn/long.md",
            category="learn",
            title="Long Page",
            description="",
            keywords=[],
            sections=[
                DocSection(
                    title="Long Section",
                    level=2,
                    text=" ".join(["state-channel"] * 200),
                    code_blocks=[],
                )
            ],
            raw_content="",
        )
    ]

    engine.build_index(pages)
    assert engine.is_ready()
    assert len(engine._chunks) > 1
