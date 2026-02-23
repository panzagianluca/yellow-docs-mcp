"""Live smoke test against the upstream docs repository."""
from __future__ import annotations

import os

import pytest

from yellow_docs_mcp.indexer import RepoManager, RepoSource
from yellow_docs_mcp.search import SearchEngine


pytestmark = pytest.mark.live


@pytest.mark.skipif(
    os.getenv("YELLOW_DOCS_LIVE_TESTS") != "1",
    reason="Set YELLOW_DOCS_LIVE_TESTS=1 to run live smoke tests",
)
def test_live_docs_repo_smoke(tmp_path):
    source = RepoSource(name="docs", url="https://github.com/layer-3/docs.git", doc_paths=("docs",))
    manager = RepoManager(base_dir=tmp_path, sources=[source], max_file_bytes=1_000_000)
    manager.sync_repo()
    pages = manager.parse_all_docs()
    assert len(pages) > 20

    engine = SearchEngine(enable_vectors=False)
    engine.build_index(pages)
    results = engine.search("state channels", source="docs", limit=5)
    assert len(results) > 0
