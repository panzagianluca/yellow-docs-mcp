"""Tests for MDX/MD document parser."""
import pytest
from yellow_docs_mcp.parser import parse_document, parse_docs_directory


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
    assert page.source == "docs"


def test_parse_no_frontmatter():
    page = parse_document(SAMPLE_MD_NO_FRONTMATTER, "manuals/running-clearnode-locally.md")
    assert page.title == "Running ClearNode Locally"
    assert page.category == "manuals"


def test_parse_sections():
    page = parse_document(SAMPLE_MDX, "learn/introduction/what-yellow-solves.mdx")
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


def test_parse_docs_directory_with_source_and_prefix(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "guide.md").write_text("# Guide\n\n## Step\nRun it")

    pages = parse_docs_directory(docs_dir, source="nitro", path_prefix="manuals")
    assert len(pages) == 1
    assert pages[0].source == "nitro"
    assert pages[0].path == "manuals/guide.md"


def test_parse_docs_directory_excludes_node_modules(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    node_modules = docs_dir / "node_modules"
    node_modules.mkdir()
    (node_modules / "bad.md").write_text("# Should be skipped")
    (docs_dir / "ok.md").write_text("# Included")

    pages = parse_docs_directory(docs_dir)
    assert len(pages) == 1
    assert pages[0].path == "ok.md"
