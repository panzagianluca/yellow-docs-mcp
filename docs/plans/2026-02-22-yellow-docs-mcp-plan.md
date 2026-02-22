# Yellow Network Docs MCP - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an MCP server that indexes the Yellow Network docs repo and exposes 5 structured retrieval tools for AI assistants.

**Architecture:** Python FastMCP server with local sentence-transformers embeddings + BM25 keyword index. Clones/pulls the docs repo on startup, parses MDX/MD into structured chunks, builds hybrid search index. Serves 5 tools over stdio.

**Tech Stack:** Python 3.11+, FastMCP, sentence-transformers (all-MiniLM-L6-v2), rank-bm25, numpy, gitpython, pyyaml

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/yellow_docs_mcp/__init__.py`
- Create: `src/yellow_docs_mcp/server.py` (stub)
- Create: `README.md`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "yellow-docs-mcp"
version = "0.1.0"
description = "MCP server for Yellow Network documentation"
requires-python = ">=3.11"
dependencies = [
    "mcp[cli]>=1.0",
    "sentence-transformers>=2.2",
    "rank-bm25>=0.2",
    "numpy>=1.24",
    "pyyaml>=6.0",
    "gitpython>=3.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**Step 2: Create package init**

```python
# src/yellow_docs_mcp/__init__.py
"""Yellow Network Docs MCP Server."""
```

**Step 3: Create minimal server stub**

```python
# src/yellow_docs_mcp/server.py
"""MCP server for Yellow Network documentation."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("yellow-docs")


@mcp.tool()
async def search_docs(query: str, limit: int = 5) -> str:
    """Search Yellow Network documentation using hybrid semantic + keyword search."""
    return "Not implemented yet"


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
```

**Step 4: Create README.md**

```markdown
# Yellow Network Docs MCP

MCP server that indexes [Yellow Network documentation](https://github.com/layer-3/docs) and provides structured retrieval tools for AI assistants.

## Setup

```bash
uv sync
```

## Usage

Add to Claude Code (`~/.claude/mcp.json`):

```json
{
  "mcpServers": {
    "yellow-docs": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/yellow-docs-mcp", "python", "-m", "yellow_docs_mcp.server"]
    }
  }
}
```
```

**Step 5: Verify the stub runs**

Run: `cd /Users/panza/Documents/Yellow\ MPC && uv sync && uv run python -c "from yellow_docs_mcp.server import mcp; print('OK')"`
Expected: `OK`

**Step 6: Commit**

```bash
git add pyproject.toml src/ README.md
git commit -m "feat: scaffold yellow-docs-mcp project"
```

---

### Task 2: Document Parser

**Files:**
- Create: `src/yellow_docs_mcp/parser.py`
- Create: `tests/test_parser.py`

**Step 1: Write the failing tests**

```python
# tests/test_parser.py
"""Tests for MDX/MD document parser."""
import pytest
from yellow_docs_mcp.parser import parse_document, DocPage, DocSection


SAMPLE_MDX = """---
sidebar_position: 1
title: What Yellow Solves
description: Understand the core problems Yellow Network addresses
keywords: [Yellow Network, state channels, blockchain scaling]
---

import Tooltip from '@site/src/components/Tooltip';
import { tooltipDefinitions } from '@site/src/constants/tooltipDefinitions';

# What Yellow Solves

Yellow Network addresses key challenges in blockchain.

## The Fragmentation Problem

Today's blockchain ecosystem is fragmented across multiple <Tooltip content={tooltipDefinitions.blockchain}>chains</Tooltip>.

:::tip Key Insight
State channels solve this by moving operations off-chain.
:::

```typescript
const client = new NitroliteClient({
  endpoint: "wss://clearnet.yellow.com/ws",
});
```

## The Speed Problem

Traditional L1 transactions are slow.

### Gas Costs

Every on-chain operation costs gas.
"""

SAMPLE_MD_NO_FRONTMATTER = """# Running ClearNode Locally

This guide walks you through setting up a ClearNode.

## Prerequisites

- Docker installed
- Node.js 18+
"""


def test_parse_frontmatter():
    page = parse_document(SAMPLE_MDX, "learn/introduction/what-yellow-solves.mdx")
    assert page.title == "What Yellow Solves"
    assert page.description == "Understand the core problems Yellow Network addresses"
    assert "state channels" in page.keywords
    assert page.path == "learn/introduction/what-yellow-solves.mdx"
    assert page.category == "learn"


def test_parse_no_frontmatter():
    page = parse_document(SAMPLE_MD_NO_FRONTMATTER, "manuals/running-clearnode-locally.md")
    assert page.title == "Running ClearNode Locally"
    assert page.category == "manuals"


def test_parse_sections():
    page = parse_document(SAMPLE_MDX, "learn/introduction/what-yellow-solves.mdx")
    # Should have sections for intro, "The Fragmentation Problem", "The Speed Problem", "Gas Costs"
    assert len(page.sections) >= 3
    titles = [s.title for s in page.sections]
    assert "The Fragmentation Problem" in titles
    assert "The Speed Problem" in titles


def test_strip_imports():
    page = parse_document(SAMPLE_MDX, "learn/introduction/what-yellow-solves.mdx")
    for section in page.sections:
        assert "import Tooltip" not in section.text
        assert "tooltipDefinitions" not in section.text or "Tooltip" not in section.text.split("tooltipDefinitions")[0]


def test_strip_tooltip_tags():
    page = parse_document(SAMPLE_MDX, "learn/introduction/what-yellow-solves.mdx")
    frag_section = [s for s in page.sections if s.title == "The Fragmentation Problem"][0]
    assert "<Tooltip" not in frag_section.text
    assert "chains" in frag_section.text


def test_extract_code_blocks():
    page = parse_document(SAMPLE_MDX, "learn/introduction/what-yellow-solves.mdx")
    frag_section = [s for s in page.sections if s.title == "The Fragmentation Problem"][0]
    assert len(frag_section.code_blocks) == 1
    assert frag_section.code_blocks[0].language == "typescript"
    assert "NitroliteClient" in frag_section.code_blocks[0].code


def test_preserve_admonitions():
    page = parse_document(SAMPLE_MDX, "learn/introduction/what-yellow-solves.mdx")
    frag_section = [s for s in page.sections if s.title == "The Fragmentation Problem"][0]
    assert "State channels solve this" in frag_section.text
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/panza/Documents/Yellow\ MPC && uv run pytest tests/test_parser.py -v`
Expected: FAIL (ModuleNotFoundError)

**Step 3: Implement the parser**

```python
# src/yellow_docs_mcp/parser.py
"""Parse MDX/MD documents into structured sections."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CodeBlock:
    """A fenced code block extracted from a document."""
    language: str
    code: str


@dataclass
class DocSection:
    """A section of a document (split by headings)."""
    title: str
    level: int  # heading level (1, 2, 3)
    text: str  # prose content (code blocks removed)
    code_blocks: list[CodeBlock] = field(default_factory=list)


@dataclass
class DocPage:
    """A parsed document page."""
    path: str
    category: str
    title: str
    description: str
    keywords: list[str]
    sections: list[DocSection]
    raw_content: str


def _extract_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML frontmatter from document content."""
    if not content.startswith("---"):
        return {}, content

    end = content.find("---", 3)
    if end == -1:
        return {}, content

    import yaml
    fm_text = content[3:end].strip()
    body = content[end + 3:].strip()

    try:
        fm = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError:
        fm = {}

    return fm, body


def _strip_imports(content: str) -> str:
    """Remove JSX import statements."""
    lines = content.split("\n")
    filtered = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") and ("from " in stripped or "require(" in stripped):
            # Check if it looks like a JS/JSX import (has quotes)
            if "'" in stripped or '"' in stripped:
                continue
        filtered.append(line)
    return "\n".join(filtered)


def _strip_tooltip_tags(content: str) -> str:
    """Replace <Tooltip ...>text</Tooltip> with just text."""
    # Match <Tooltip content={...}>text</Tooltip>
    pattern = r'<Tooltip[^>]*>(.*?)</Tooltip>'
    return re.sub(pattern, r'\1', content, flags=re.DOTALL)


def _strip_tabs_components(content: str) -> str:
    """Strip Tabs/TabItem JSX wrappers, keep inner content."""
    content = re.sub(r'<Tabs[^>]*>', '', content)
    content = re.sub(r'</Tabs>', '', content)
    content = re.sub(r'<TabItem[^>]*>', '', content)
    content = re.sub(r'</TabItem>', '', content)
    return content


def _extract_code_blocks(text: str) -> tuple[str, list[CodeBlock]]:
    """Extract fenced code blocks from text, return cleaned text and blocks."""
    blocks = []
    pattern = r'```(\w*)\n(.*?)```'

    def replacer(match):
        lang = match.group(1) or "text"
        code = match.group(2).strip()
        blocks.append(CodeBlock(language=lang, code=code))
        return ""

    cleaned = re.sub(pattern, replacer, text, flags=re.DOTALL)
    return cleaned.strip(), blocks


def _extract_admonition_text(text: str) -> str:
    """Convert :::type content ::: to plain text, keeping content."""
    # Replace :::type Title with just the content inside
    # Keep the text between ::: markers
    lines = text.split("\n")
    result = []
    in_admonition = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(":::") and not in_admonition:
            in_admonition = True
            # Extract title if present after :::type
            parts = stripped.split(" ", 1)
            if len(parts) > 1 and not parts[1].startswith(":::"):
                pass  # Title is part of admonition, skip the marker line
            continue
        elif stripped == ":::" and in_admonition:
            in_admonition = False
            continue
        result.append(line)
    return "\n".join(result)


def _split_sections(content: str) -> list[DocSection]:
    """Split content into sections by headings."""
    # Match ## and ### headings
    heading_pattern = r'^(#{1,3})\s+(.+)$'
    lines = content.split("\n")

    sections: list[DocSection] = []
    current_title = ""
    current_level = 1
    current_lines: list[str] = []

    for line in lines:
        match = re.match(heading_pattern, line)
        if match:
            # Save previous section if it has content
            if current_lines or current_title:
                raw_text = "\n".join(current_lines).strip()
                text, code_blocks = _extract_code_blocks(raw_text)
                text = _extract_admonition_text(text)
                text = text.strip()
                if text or code_blocks:
                    sections.append(DocSection(
                        title=current_title,
                        level=current_level,
                        text=text,
                        code_blocks=code_blocks,
                    ))

            current_level = len(match.group(1))
            current_title = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Save last section
    if current_lines or current_title:
        raw_text = "\n".join(current_lines).strip()
        text, code_blocks = _extract_code_blocks(raw_text)
        text = _extract_admonition_text(text)
        text = text.strip()
        if text or code_blocks:
            sections.append(DocSection(
                title=current_title,
                level=current_level,
                text=text,
                code_blocks=code_blocks,
            ))

    return sections


def _title_from_content(content: str) -> str:
    """Extract title from first # heading if no frontmatter title."""
    match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    return match.group(1).strip() if match else "Untitled"


def parse_document(content: str, path: str) -> DocPage:
    """Parse a MDX/MD document into structured sections.

    Args:
        content: Raw file content
        path: Relative path from docs/ root (e.g. "learn/introduction/what-yellow-solves.mdx")

    Returns:
        Parsed DocPage with metadata and sections
    """
    # Extract frontmatter
    frontmatter, body = _extract_frontmatter(content)

    # Clean up JSX
    body = _strip_imports(body)
    body = _strip_tooltip_tags(body)
    body = _strip_tabs_components(body)

    # Extract metadata
    title = frontmatter.get("title", _title_from_content(body))
    description = frontmatter.get("description", "")
    keywords = frontmatter.get("keywords", [])
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",")]

    # Get category from path
    category = path.split("/")[0] if "/" in path else ""

    # Split into sections
    sections = _split_sections(body)

    return DocPage(
        path=path,
        category=category,
        title=title,
        description=description,
        keywords=keywords,
        sections=sections,
        raw_content=content,
    )


def parse_docs_directory(docs_dir: Path) -> list[DocPage]:
    """Parse all MDX/MD files in a docs directory.

    Args:
        docs_dir: Path to docs/ directory

    Returns:
        List of parsed DocPage objects
    """
    pages = []
    for ext in ("*.md", "*.mdx"):
        for filepath in docs_dir.rglob(ext):
            rel_path = str(filepath.relative_to(docs_dir))
            content = filepath.read_text(encoding="utf-8")
            page = parse_document(content, rel_path)
            pages.append(page)
    return pages
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/panza/Documents/Yellow\ MPC && uv run pytest tests/test_parser.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add src/yellow_docs_mcp/parser.py tests/test_parser.py
git commit -m "feat: add MDX/MD document parser with section extraction"
```

---

### Task 3: Git Repo Manager (Indexer)

**Files:**
- Create: `src/yellow_docs_mcp/indexer.py`
- Create: `tests/test_indexer.py`

**Step 1: Write the failing tests**

```python
# tests/test_indexer.py
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
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/panza/Documents/Yellow\ MPC && uv run pytest tests/test_indexer.py -v`
Expected: FAIL (ModuleNotFoundError)

**Step 3: Implement the indexer**

```python
# src/yellow_docs_mcp/indexer.py
"""Git repo management and document indexing."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from yellow_docs_mcp.parser import DocPage, parse_docs_directory

logger = logging.getLogger(__name__)

INDEX_DIR = Path.home() / ".yellow-docs-mcp"
REPO_URL = "https://github.com/layer-3/docs.git"
BRANCH = "master"


class RepoManager:
    """Manages the local clone of the docs repo."""

    def __init__(
        self,
        base_dir: Path | None = None,
        repo_url: str = REPO_URL,
        branch: str = BRANCH,
    ):
        self.base_dir = base_dir or INDEX_DIR
        self.repo_url = repo_url
        self.branch = branch
        self.repo_dir = self.base_dir / "repo"
        self.docs_dir = self.repo_dir / "docs"
        self.hash_file = self.base_dir / "content_hash.txt"

    def sync_repo(self) -> bool:
        """Clone or pull the docs repo. Returns True if content changed."""
        import git

        self.base_dir.mkdir(parents=True, exist_ok=True)

        if not self.repo_dir.exists():
            logger.info("Cloning %s...", self.repo_url)
            git.Repo.clone_from(
                self.repo_url,
                str(self.repo_dir),
                branch=self.branch,
                depth=1,
            )
            return True
        else:
            logger.info("Pulling latest changes...")
            repo = git.Repo(str(self.repo_dir))
            origin = repo.remotes.origin
            old_hash = self._compute_content_hash(self.docs_dir)
            origin.pull()
            new_hash = self._compute_content_hash(self.docs_dir)
            changed = old_hash != new_hash
            if changed:
                logger.info("Docs content changed, re-indexing needed.")
            else:
                logger.info("No content changes detected.")
            return changed

    def needs_reindex(self) -> bool:
        """Check if docs have changed since last index build."""
        if not self.hash_file.exists():
            return True
        stored_hash = self.hash_file.read_text().strip()
        current_hash = self._compute_content_hash(self.docs_dir)
        return stored_hash != current_hash

    def save_hash(self) -> None:
        """Save current content hash after successful indexing."""
        current_hash = self._compute_content_hash(self.docs_dir)
        self.hash_file.write_text(current_hash)

    def parse_all_docs(self) -> list[DocPage]:
        """Parse all documents in the docs directory."""
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Docs directory not found: {self.docs_dir}")
        return parse_docs_directory(self.docs_dir)

    def _compute_content_hash(self, docs_dir: Path) -> str:
        """Compute a hash of all doc file contents for change detection."""
        hasher = hashlib.sha256()
        files = sorted(docs_dir.rglob("*"))
        for f in files:
            if f.is_file() and f.suffix in (".md", ".mdx"):
                hasher.update(f.read_bytes())
                hasher.update(str(f.relative_to(docs_dir)).encode())
        return hasher.hexdigest()
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/panza/Documents/Yellow\ MPC && uv run pytest tests/test_indexer.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/yellow_docs_mcp/indexer.py tests/test_indexer.py
git commit -m "feat: add git repo manager with clone/pull and change detection"
```

---

### Task 4: Search Engine (BM25 + Vector + RRF)

**Files:**
- Create: `src/yellow_docs_mcp/search.py`
- Create: `src/yellow_docs_mcp/store.py`
- Create: `tests/test_search.py`

**Step 1: Write the failing tests**

```python
# tests/test_search.py
"""Tests for hybrid search engine."""
import pytest
from yellow_docs_mcp.parser import DocPage, DocSection, CodeBlock
from yellow_docs_mcp.search import SearchEngine, SearchResult


def _make_pages() -> list[DocPage]:
    """Create test doc pages."""
    return [
        DocPage(
            path="learn/introduction/what-yellow-solves.mdx",
            category="learn",
            title="What Yellow Solves",
            description="Core problems Yellow Network addresses",
            keywords=["state channels", "scaling"],
            sections=[
                DocSection(
                    title="The Fragmentation Problem",
                    level=2,
                    text="Today's blockchain ecosystem is fragmented across multiple chains. Yellow Network unifies liquidity.",
                    code_blocks=[],
                ),
                DocSection(
                    title="The Speed Problem",
                    level=2,
                    text="Traditional L1 transactions are slow. State channels provide instant finality.",
                    code_blocks=[CodeBlock(language="typescript", code="const client = new NitroliteClient();")],
                ),
            ],
            raw_content="",
        ),
        DocPage(
            path="protocol/off-chain/channel-methods.mdx",
            category="protocol",
            title="Channel Management Methods",
            description="RPC methods for channel operations",
            keywords=["channels", "RPC"],
            sections=[
                DocSection(
                    title="create_channel",
                    level=2,
                    text="Creates a new payment channel between two participants. Requires a funded custody contract.",
                    code_blocks=[CodeBlock(language="json", code='{"req": [1, "create_channel", {}]}')],
                ),
                DocSection(
                    title="close_channel",
                    level=2,
                    text="Closes an existing channel and settles balances on-chain.",
                    code_blocks=[],
                ),
            ],
            raw_content="",
        ),
        DocPage(
            path="api-reference/index.md",
            category="api-reference",
            title="SDK API Reference",
            description="TypeScript SDK types and methods",
            keywords=["SDK", "API", "TypeScript"],
            sections=[
                DocSection(
                    title="NitroliteRPC",
                    level=2,
                    text="The main RPC client class for interacting with ClearNode WebSocket endpoints.",
                    code_blocks=[CodeBlock(language="typescript", code="class NitroliteRPC { connect(url: string): void }")],
                ),
            ],
            raw_content="",
        ),
    ]


def test_search_engine_build():
    engine = SearchEngine()
    pages = _make_pages()
    engine.build_index(pages)
    assert engine.is_ready()


def test_search_returns_results():
    engine = SearchEngine()
    engine.build_index(_make_pages())
    results = engine.search("state channels scaling")
    assert len(results) > 0
    assert isinstance(results[0], SearchResult)


def test_search_relevance():
    """Searching for 'create_channel' should rank the protocol page highest."""
    engine = SearchEngine()
    engine.build_index(_make_pages())
    results = engine.search("create_channel")
    assert results[0].path == "protocol/off-chain/channel-methods.mdx"


def test_search_category_filter():
    engine = SearchEngine()
    engine.build_index(_make_pages())
    results = engine.search("channel", category="protocol")
    for r in results:
        assert r.category == "protocol"


def test_search_limit():
    engine = SearchEngine()
    engine.build_index(_make_pages())
    results = engine.search("channel", limit=1)
    assert len(results) <= 1


def test_search_result_has_content():
    engine = SearchEngine()
    engine.build_index(_make_pages())
    results = engine.search("fragmentation")
    assert len(results) > 0
    assert results[0].text  # Should have text content
    assert results[0].section_title  # Should have section title
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/panza/Documents/Yellow\ MPC && uv run pytest tests/test_search.py -v`
Expected: FAIL (ModuleNotFoundError)

**Step 3: Implement the store (persistence)**

```python
# src/yellow_docs_mcp/store.py
"""Index persistence - save/load search index to disk."""
from __future__ import annotations

import pickle
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

INDEX_DIR = Path.home() / ".yellow-docs-mcp"


def save_index(data: dict[str, Any], path: Path | None = None) -> None:
    """Save index data to pickle file."""
    path = path or (INDEX_DIR / "index.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    logger.info("Index saved to %s", path)


def load_index(path: Path | None = None) -> dict[str, Any] | None:
    """Load index data from pickle file. Returns None if not found."""
    path = path or (INDEX_DIR / "index.pkl")
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        logger.info("Index loaded from %s", path)
        return data
    except Exception as e:
        logger.warning("Failed to load index: %s", e)
        return None
```

**Step 4: Implement the search engine**

```python
# src/yellow_docs_mcp/search.py
"""Hybrid search engine combining BM25 keyword search and vector similarity."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from yellow_docs_mcp.parser import DocPage

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""
    path: str
    category: str
    page_title: str
    section_title: str
    text: str
    score: float
    code_blocks: list[dict] | None = None


@dataclass
class Chunk:
    """An indexed document chunk (one per section)."""
    chunk_id: int
    path: str
    category: str
    page_title: str
    section_title: str
    text: str
    code_blocks: list[dict]
    embedding: np.ndarray | None = None


class SearchEngine:
    """Hybrid BM25 + vector search with Reciprocal Rank Fusion."""

    def __init__(self):
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._embeddings: np.ndarray | None = None
        self._model = None
        self._pages: list[DocPage] = []
        self._ready = False

    def is_ready(self) -> bool:
        return self._ready

    def build_index(self, pages: list[DocPage]) -> None:
        """Build search index from parsed document pages."""
        self._pages = pages
        self._chunks = []

        # Create chunks from all sections
        chunk_id = 0
        for page in pages:
            for section in page.sections:
                # Combine section text with page context for better search
                search_text = f"{page.title} - {section.title}\n{section.text}"
                code_block_dicts = [
                    {"language": cb.language, "code": cb.code}
                    for cb in section.code_blocks
                ]
                self._chunks.append(Chunk(
                    chunk_id=chunk_id,
                    path=page.path,
                    category=page.category,
                    page_title=page.title,
                    section_title=section.title,
                    text=section.text,
                    code_blocks=code_block_dicts,
                ))
                chunk_id += 1

        if not self._chunks:
            logger.warning("No chunks to index!")
            return

        # Build BM25 index
        tokenized = [self._tokenize(c.text + " " + c.section_title + " " + c.page_title) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)

        # Build vector index
        self._build_embeddings()

        self._ready = True
        logger.info("Index built: %d chunks from %d pages", len(self._chunks), len(pages))

    def _build_embeddings(self) -> None:
        """Generate embeddings for all chunks."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model...")
            self._model = SentenceTransformer("all-MiniLM-L6-v2")

        texts = [f"{c.page_title} - {c.section_title}\n{c.text}" for c in self._chunks]
        self._embeddings = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        logger.info("Generated %d embeddings", len(self._embeddings))

    def search(
        self,
        query: str,
        limit: int = 5,
        category: str | None = None,
    ) -> list[SearchResult]:
        """Search using hybrid BM25 + vector with RRF merge.

        Args:
            query: Search query string
            limit: Max results to return
            category: Optional category filter (e.g. "protocol", "learn")

        Returns:
            Ranked list of SearchResult
        """
        if not self._ready:
            return []

        # Filter chunks by category if specified
        if category:
            valid_ids = {c.chunk_id for c in self._chunks if c.category == category}
        else:
            valid_ids = {c.chunk_id for c in self._chunks}

        # BM25 search
        bm25_scores = self._bm25.get_scores(self._tokenize(query))
        bm25_ranked = sorted(
            [(i, s) for i, s in enumerate(bm25_scores) if self._chunks[i].chunk_id in valid_ids],
            key=lambda x: x[1],
            reverse=True,
        )[:20]

        # Vector search
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")

        query_emb = self._model.encode(query, convert_to_numpy=True)
        similarities = np.dot(self._embeddings, query_emb) / (
            np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )
        vector_ranked = sorted(
            [(i, float(s)) for i, s in enumerate(similarities) if self._chunks[i].chunk_id in valid_ids],
            key=lambda x: x[1],
            reverse=True,
        )[:20]

        # RRF merge
        k = 60
        rrf_scores: dict[int, float] = {}
        for rank, (idx, _) in enumerate(bm25_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)
        for rank, (idx, _) in enumerate(vector_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)

        # Sort by RRF score
        top_indices = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:limit]

        # Build results
        results = []
        for idx in top_indices:
            chunk = self._chunks[idx]
            results.append(SearchResult(
                path=chunk.path,
                category=chunk.category,
                page_title=chunk.page_title,
                section_title=chunk.section_title,
                text=chunk.text,
                score=rrf_scores[idx],
                code_blocks=chunk.code_blocks if chunk.code_blocks else None,
            ))

        return results

    def get_all_pages(self) -> list[DocPage]:
        """Return all indexed pages."""
        return self._pages

    def get_page_by_path(self, path: str) -> DocPage | None:
        """Get a page by exact or fuzzy path match."""
        # Exact match first
        for page in self._pages:
            if page.path == path:
                return page

        # Fuzzy: match end of path
        for page in self._pages:
            if page.path.endswith(path):
                return page

        # Fuzzy: match by title (case-insensitive)
        path_lower = path.lower().replace("-", " ").replace("_", " ")
        for page in self._pages:
            if path_lower in page.title.lower():
                return page

        return None

    def get_index_data(self) -> dict:
        """Export index data for persistence."""
        return {
            "chunks": self._chunks,
            "embeddings": self._embeddings,
            "pages": self._pages,
        }

    def load_index_data(self, data: dict) -> None:
        """Load index data from persistence."""
        self._chunks = data["chunks"]
        self._embeddings = data["embeddings"]
        self._pages = data["pages"]

        # Rebuild BM25 (can't pickle it reliably)
        tokenized = [self._tokenize(c.text + " " + c.section_title + " " + c.page_title) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)

        self._ready = True
        logger.info("Index loaded: %d chunks from %d pages", len(self._chunks), len(self._pages))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer."""
        import re
        return re.findall(r'\w+', text.lower())
```

**Step 5: Run tests to verify they pass**

Run: `cd /Users/panza/Documents/Yellow\ MPC && uv run pytest tests/test_search.py -v`
Expected: All 6 tests PASS (may take ~10s first run for model download)

**Step 6: Commit**

```bash
git add src/yellow_docs_mcp/search.py src/yellow_docs_mcp/store.py tests/test_search.py
git commit -m "feat: add hybrid search engine with BM25 + vector + RRF"
```

---

### Task 5: MCP Server with All 5 Tools

**Files:**
- Modify: `src/yellow_docs_mcp/server.py`
- Create: `tests/test_server.py`

**Step 1: Write the failing tests**

```python
# tests/test_server.py
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
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/panza/Documents/Yellow\ MPC && uv run pytest tests/test_server.py -v`
Expected: FAIL (ImportError)

**Step 3: Implement the full server**

```python
# src/yellow_docs_mcp/server.py
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

# Global state
_engine = SearchEngine()
_initialized = False


def _ensure_initialized() -> None:
    """Initialize the search engine on first use."""
    global _initialized
    if _initialized:
        return

    repo = RepoManager()

    # Try loading cached index first
    cached = load_index()
    if cached and not repo.needs_reindex():
        logger.info("Loading cached index...")
        _engine.load_index_data(cached)
        _initialized = True
        return

    # Sync repo and build fresh index
    logger.info("Syncing docs repo...")
    repo.sync_repo()

    logger.info("Parsing documents...")
    pages = repo.parse_all_docs()

    logger.info("Building search index...")
    _engine.build_index(pages)

    # Cache the index
    save_index(_engine.get_index_data())
    repo.save_hash()

    _initialized = True
    logger.info("Initialization complete: %d pages indexed", len(pages))


# --- Formatting helpers ---

def _format_search_results(results: list[SearchResult]) -> str:
    """Format search results for display."""
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
    """Format a full page for display."""
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
    """Format a tree view of all documentation sections."""
    filtered = pages
    if category:
        filtered = [p for p in pages if p.category == category]

    if not filtered:
        return f"No pages found for category: {category}" if category else "No pages indexed."

    # Group by category
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
    """Format API method reference."""
    # Find API-related pages
    api_pages = [
        p for p in pages
        if p.category in ("protocol", "api-reference")
    ]

    if method:
        # Search for specific method
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
        # List all methods grouped by category
        parts = ["# API Methods\n"]
        for page in sorted(api_pages, key=lambda p: p.path):
            parts.append(f"\n## {page.title} (`{page.path}`)\n")
            for section in page.sections:
                has_code = " [code]" if section.code_blocks else ""
                parts.append(f"- `{section.title}`{has_code}")
        return "\n".join(parts)


def _format_code_examples(pages: list[DocPage], topic: str, language: str = "typescript") -> str:
    """Find and format code examples matching a topic."""
    topic_lower = topic.lower()
    lang_lower = language.lower()

    results = []
    for page in pages:
        for section in page.sections:
            # Check if topic matches section title or text
            if topic_lower in section.title.lower() or topic_lower in section.text.lower():
                for cb in section.code_blocks:
                    if cb.language.lower() == lang_lower or language == "all":
                        results.append((page, section, cb))

    if not results:
        # Fallback: search all code blocks regardless of language
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


# --- MCP Tools ---

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
        # Try searching by topic
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


# --- Entry point ---

def main():
    """Run the MCP server."""
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/panza/Documents/Yellow\ MPC && uv run pytest tests/test_server.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/yellow_docs_mcp/server.py tests/test_server.py
git commit -m "feat: implement all 5 MCP tools with formatting helpers"
```

---

### Task 6: __main__.py Entry Point + MCP Config

**Files:**
- Create: `src/yellow_docs_mcp/__main__.py`
- Create: `claude_mcp_config.json`

**Step 1: Create __main__.py**

```python
# src/yellow_docs_mcp/__main__.py
"""Entry point for running as module: python -m yellow_docs_mcp"""
from yellow_docs_mcp.server import main

main()
```

**Step 2: Create example MCP config**

```json
{
  "mcpServers": {
    "yellow-docs": {
      "command": "uv",
      "args": ["run", "--directory", "/Users/panza/Documents/Yellow MPC", "python", "-m", "yellow_docs_mcp"]
    }
  }
}
```

**Step 3: Verify it can start (quick sanity check)**

Run: `cd /Users/panza/Documents/Yellow\ MPC && timeout 5 uv run python -c "from yellow_docs_mcp.server import mcp; print('Server ready:', mcp.name)" 2>&1 || true`
Expected: `Server ready: yellow-docs`

**Step 4: Commit**

```bash
git add src/yellow_docs_mcp/__main__.py claude_mcp_config.json
git commit -m "feat: add module entry point and example MCP config"
```

---

### Task 7: Integration Test - Full Pipeline

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_integration.py
"""Integration test: parse real-like docs, build index, search."""
import pytest
from pathlib import Path
from yellow_docs_mcp.parser import parse_docs_directory, DocPage
from yellow_docs_mcp.search import SearchEngine
from yellow_docs_mcp.store import save_index, load_index


@pytest.fixture
def sample_docs(tmp_path):
    """Create a mini docs directory with realistic content."""
    docs = tmp_path / "docs"

    # learn/
    learn = docs / "learn" / "introduction"
    learn.mkdir(parents=True)
    (learn / "what-yellow-solves.mdx").write_text("""---
title: What Yellow Solves
description: Core problems Yellow Network addresses
keywords: [state channels, scaling, off-chain]
sidebar_position: 1
---

import Tooltip from '@site/src/components/Tooltip';

# What Yellow Solves

Yellow Network uses <Tooltip content="A protocol">state channels</Tooltip> to solve blockchain fragmentation.

## The Fragmentation Problem

Liquidity is split across chains.

## The Speed Problem

L1 transactions are slow. State channels provide instant finality.

```typescript
const client = new NitroliteClient({
  endpoint: "wss://clearnet.yellow.com/ws",
});
```
""")

    # protocol/off-chain/
    protocol = docs / "protocol" / "off-chain"
    protocol.mkdir(parents=True)
    (protocol / "channel-methods.mdx").write_text("""---
title: Channel Management Methods
sidebar_position: 4
---

## create_channel

Creates a new payment channel between two participants.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| participants | address[] | Yes | Channel participants |
| chain_id | uint32 | Yes | Target blockchain |

```json
{"req": [1, "create_channel", {"participants": ["0xA", "0xB"]}]}
```

## close_channel

Closes an existing channel and settles on-chain.
""")

    # api-reference/
    api = docs / "api-reference"
    api.mkdir(parents=True)
    (api / "index.md").write_text("""---
title: SDK API Reference
description: TypeScript SDK types and methods
---

## NitroliteRPC

The main client class.

```typescript
class NitroliteRPC {
  connect(url: string): Promise<void>
  createChannel(params: CreateChannelParams): Promise<Channel>
}
```

## Types

```typescript
interface Channel {
  participants: Address[];
  adjudicator: Address;
  nonce: bigint;
}
```
""")

    return docs


def test_full_pipeline(sample_docs, tmp_path):
    """Test: parse → index → search → find results."""
    # Parse
    pages = parse_docs_directory(sample_docs)
    assert len(pages) == 3

    # Index
    engine = SearchEngine()
    engine.build_index(pages)
    assert engine.is_ready()

    # Search - semantic
    results = engine.search("how do state channels work")
    assert len(results) > 0
    assert any("yellow" in r.page_title.lower() for r in results)

    # Search - keyword (exact method name)
    results = engine.search("create_channel")
    assert len(results) > 0
    assert results[0].path == "protocol/off-chain/channel-methods.mdx"

    # Search with category filter
    results = engine.search("channel", category="protocol")
    assert all(r.category == "protocol" for r in results)

    # Get page by path
    page = engine.get_page_by_path("protocol/off-chain/channel-methods.mdx")
    assert page is not None
    assert page.title == "Channel Management Methods"

    # Get page by fuzzy title
    page = engine.get_page_by_path("channel management")
    assert page is not None


def test_index_persistence(sample_docs, tmp_path):
    """Test: save index → load index → search still works."""
    pages = parse_docs_directory(sample_docs)

    # Build and save
    engine1 = SearchEngine()
    engine1.build_index(pages)
    index_path = tmp_path / "test_index.pkl"
    save_index(engine1.get_index_data(), index_path)

    # Load into new engine
    engine2 = SearchEngine()
    data = load_index(index_path)
    assert data is not None
    engine2.load_index_data(data)

    # Search should still work
    results = engine2.search("create_channel")
    assert len(results) > 0
    assert results[0].path == "protocol/off-chain/channel-methods.mdx"
```

**Step 2: Run integration tests**

Run: `cd /Users/panza/Documents/Yellow\ MPC && uv run pytest tests/test_integration.py -v`
Expected: All tests PASS

**Step 3: Run all tests**

Run: `cd /Users/panza/Documents/Yellow\ MPC && uv run pytest tests/ -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for full parse → index → search pipeline"
```

---

### Task 8: Wire Up to Claude Code

**Step 1: Add MCP config to Claude Code**

Add to `~/.claude/mcp.json` (or create if doesn't exist):

```json
{
  "mcpServers": {
    "yellow-docs": {
      "command": "uv",
      "args": ["run", "--directory", "/Users/panza/Documents/Yellow MPC", "python", "-m", "yellow_docs_mcp"]
    }
  }
}
```

**Step 2: Test the MCP server manually**

Run: `cd /Users/panza/Documents/Yellow\ MPC && echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1"}}}' | timeout 10 uv run python -m yellow_docs_mcp 2>/dev/null | head -1`
Expected: JSON response with server capabilities

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: yellow-docs-mcp v0.1.0 - complete MCP server for Yellow Network docs"
```
