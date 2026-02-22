"""Tests for git repo cloning and doc indexing."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from yellow_docs_mcp.indexer import RepoManager, INDEX_DIR


def test_repo_manager_defaults():
    rm = RepoManager()
    assert rm.repo_url == "https://github.com/layer-3/docs.git"
    assert rm.repo_dir == INDEX_DIR / "repo"
    assert rm.branch == "master"


def test_docs_dir_path():
    rm = RepoManager()
    assert rm.docs_dir == INDEX_DIR / "repo" / "docs"


def test_content_hash_changes_with_content(tmp_path):
    """Content hash should change when doc files change."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "test.md").write_text("# Hello")
    rm = RepoManager(base_dir=tmp_path)
    hash1 = rm._compute_content_hash(docs_dir)
    (docs_dir / "test.md").write_text("# Changed")
    hash2 = rm._compute_content_hash(docs_dir)
    assert hash1 != hash2


def test_content_hash_stable(tmp_path):
    """Same content should produce same hash."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "test.md").write_text("# Hello")
    rm = RepoManager(base_dir=tmp_path)
    hash1 = rm._compute_content_hash(docs_dir)
    hash2 = rm._compute_content_hash(docs_dir)
    assert hash1 == hash2
