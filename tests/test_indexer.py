"""Tests for multi-repo indexing manager."""
from yellow_docs_mcp.indexer import (
    DEFAULT_REPO_SOURCES,
    INDEX_DIR,
    RepoManager,
    RepoSource,
)


def test_repo_manager_defaults():
    rm = RepoManager()
    assert rm.base_dir == INDEX_DIR
    assert len(rm.sources) == len(DEFAULT_REPO_SOURCES)
    assert [s.name for s in rm.sources] == [
        "docs",
        "clearsync",
        "docs-gitbook",
        "nitro",
        "release-process",
    ]


def test_content_hash_changes_with_content(tmp_path):
    source = RepoSource(name="docs", url="https://example.com/docs.git", doc_paths=("docs",))
    rm = RepoManager(base_dir=tmp_path, sources=[source])
    repo_dir = tmp_path / "repos" / "docs"
    docs_dir = repo_dir / "docs"
    docs_dir.mkdir(parents=True)

    (docs_dir / "test.md").write_text("# Hello")
    hash1 = rm._compute_content_hash(repo_dir, [docs_dir])
    (docs_dir / "test.md").write_text("# Changed")
    hash2 = rm._compute_content_hash(repo_dir, [docs_dir])
    assert hash1 != hash2


def test_content_hash_stable(tmp_path):
    source = RepoSource(name="docs", url="https://example.com/docs.git", doc_paths=("docs",))
    rm = RepoManager(base_dir=tmp_path, sources=[source])
    repo_dir = tmp_path / "repos" / "docs"
    docs_dir = repo_dir / "docs"
    docs_dir.mkdir(parents=True)

    (docs_dir / "test.md").write_text("# Hello")
    hash1 = rm._compute_content_hash(repo_dir, [docs_dir])
    hash2 = rm._compute_content_hash(repo_dir, [docs_dir])
    assert hash1 == hash2


def test_resolve_doc_roots_falls_back_to_repo_root(tmp_path):
    source = RepoSource(name="nitro", url="https://example.com/nitro.git")
    rm = RepoManager(base_dir=tmp_path, sources=[source])
    repo_dir = tmp_path / "repos" / "nitro"
    repo_dir.mkdir(parents=True)
    (repo_dir / "README.md").write_text("# Nitro")

    roots = rm._resolve_doc_roots(source, repo_dir)
    assert roots == [repo_dir]


def test_parse_all_docs_multi_source(tmp_path):
    docs_source = RepoSource(name="docs", url="https://example.com/docs.git", doc_paths=("docs",))
    nitro_source = RepoSource(name="nitro", url="https://example.com/nitro.git", doc_paths=("docs",))
    rm = RepoManager(base_dir=tmp_path, sources=[docs_source, nitro_source])

    docs_root = tmp_path / "repos" / "docs" / "docs" / "protocol"
    docs_root.mkdir(parents=True)
    (docs_root / "channel-methods.mdx").write_text("## create_channel\nCreates channel")

    nitro_root = tmp_path / "repos" / "nitro" / "docs"
    nitro_root.mkdir(parents=True)
    (nitro_root / "overview.md").write_text("# Nitro\n## Intro\nFast state channels")

    pages = rm.parse_all_docs()
    assert len(pages) == 2
    assert {page.source for page in pages} == {"docs", "nitro"}
