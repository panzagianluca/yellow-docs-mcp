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
    """Test: parse -> index -> search -> find results."""
    pages = parse_docs_directory(sample_docs)
    assert len(pages) == 3

    engine = SearchEngine(enable_vectors=False)
    engine.build_index(pages)
    assert engine.is_ready()

    results = engine.search("how do state channels work")
    assert len(results) > 0
    assert any("yellow" in r.page_title.lower() for r in results)

    results = engine.search("create_channel")
    assert len(results) > 0
    assert results[0].path == "protocol/off-chain/channel-methods.mdx"

    results = engine.search("channel", category="protocol")
    assert all(r.category == "protocol" for r in results)

    results = engine.search("channel", source="docs")
    assert all(r.source == "docs" for r in results)

    page = engine.get_page_by_path("protocol/off-chain/channel-methods.mdx")
    assert page is not None
    assert page.title == "Channel Management Methods"

    page = engine.get_page_by_path("channel management")
    assert page is not None


def test_index_persistence(sample_docs, tmp_path):
    """Test: save index -> load index -> search still works."""
    pages = parse_docs_directory(sample_docs)

    engine1 = SearchEngine(enable_vectors=False)
    engine1.build_index(pages)
    index_path = tmp_path / "test_index.json"
    save_index(engine1.get_index_data(), index_path)

    engine2 = SearchEngine(enable_vectors=False)
    data = load_index(index_path)
    assert data is not None
    engine2.load_index_data(data)

    results = engine2.search("create_channel")
    assert len(results) > 0
    assert results[0].path == "protocol/off-chain/channel-methods.mdx"


def test_incremental_index_reuse(sample_docs):
    pages = parse_docs_directory(sample_docs)
    engine1 = SearchEngine(enable_vectors=False)
    engine1.build_index(pages)

    engine2 = SearchEngine(enable_vectors=False)
    engine2.build_index(pages, previous_index=engine1.get_index_data())
    results = engine2.search("state channels")
    assert len(results) > 0
