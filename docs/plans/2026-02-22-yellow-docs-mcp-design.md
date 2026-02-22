# Yellow Network Docs MCP - Design Document

**Date:** 2026-02-22
**Status:** Approved
**Approach:** Structured Index + Hybrid Vector/BM25 Search (Approach B)

## Problem

AI assistants (Claude Code, Claude Desktop, etc.) need to answer questions about Yellow Network (ERC-7824 Nitrolite protocol) accurately. Without grounding in the actual docs, answers are generic or hallucinated. An MCP server that indexes the docs and exposes structured retrieval tools lets any MCP-compatible AI give precise, sourced answers.

## Target Users

- Developers building on Yellow Network (SDK integration, API reference)
- Internal team (customer support, BD) answering protocol questions
- Anyone using Claude Code/Desktop who needs Yellow Network knowledge

## Architecture

```
MCP Client (Claude Code / Desktop / etc.)
    │ stdio (MCP protocol)
    ▼
yellow-docs-mcp (Python, FastMCP)
    ├── Tools Layer (5 tools)
    ├── Search Engine (hybrid BM25 + vector, RRF merge)
    ├── Indexer (parse MDX/MD, extract structure, embed)
    └── Store (local pickle/numpy index)
    │
    │ git clone/pull on startup
    ▼
github.com/layer-3/docs (~50 MDX/MD pages, Docusaurus)
```

### Startup Sequence

1. Check `~/.yellow-docs-mcp/repo/` exists
   - No → `git clone https://github.com/layer-3/docs.git`
   - Yes → `git pull` (fast-forward)
2. Hash all doc files → compare to stored hash
   - Changed → re-parse and re-index
   - Same → load cached index
3. Start MCP server on stdio

### Storage

All local in `~/.yellow-docs-mcp/`:
```
~/.yellow-docs-mcp/
├── repo/              # Cloned docs repo
├── index.pkl          # Pickled index (chunks, embeddings, BM25, API registry)
└── content_hash.txt   # Hash of doc files for cache invalidation
```

## MCP Tools

### 1. `search_docs`
**Purpose:** Hybrid semantic + keyword search across all docs
**Input:** `query: str`, `limit: int = 5`, `category: str = None`
**Returns:** Ranked results with: content snippet, source path, section title, relevance score
**Search method:** RRF merge of BM25 + cosine similarity

### 2. `get_page`
**Purpose:** Fetch a full doc page by path or fuzzy topic match
**Input:** `path: str` (e.g. `"protocol/off-chain/channels"` or `"state channels"`)
**Returns:** Full page content with frontmatter metadata (title, description, keywords)
**Matching:** Exact path match first, then fuzzy match on title/description

### 3. `list_sections`
**Purpose:** Browse documentation structure
**Input:** `category: str = None` (optional filter: `"learn"`, `"protocol"`, `"api-reference"`, `"guides"`, `"manuals"`, `"build"`, `"tutorials"`)
**Returns:** Tree of pages with titles, descriptions, and paths

### 4. `get_api_reference`
**Purpose:** Look up protocol RPC methods and SDK API
**Input:** `method: str = None` (optional, e.g. `"create_channel"`, `"auth_request"`)
**Returns:** If method specified: signature, parameters, description, examples. If not: list of all available methods grouped by category.

### 5. `get_code_examples`
**Purpose:** Extract code snippets for a topic
**Input:** `topic: str`, `language: str = "typescript"`
**Returns:** All code blocks matching the topic, with surrounding context and source path

## Indexing Pipeline

### Document Parsing

```
MDX/MD file
  → Read raw content
  → Extract YAML frontmatter (title, description, keywords, sidebar_position)
  → Strip Docusaurus JSX components (keep inner text content)
      - <Tooltip> → extract text
      - <Tabs>/<TabItem> → extract code per tab
      - :::tip/:::warning → keep as-is with label
  → Split by headings (## and ###) into sections
  → Per section:
      → Separate prose text from code blocks
      → Tag code blocks with language
      → Extract API method signatures (from protocol/ and api-reference/ paths)
  → Generate embedding for prose text (code blocks indexed by keyword only)
```

### Chunk Strategy

- **Unit:** One chunk per H2/H3 section
- **Size:** ~200-500 tokens each (natural doc sections)
- **Total:** ~250 chunks from ~50 pages
- **Metadata per chunk:** source path, section title, category, heading level, has_code flag

### API Method Extraction

Special parsing for protocol docs (`docs/protocol/off-chain/`) and API reference (`docs/api-reference/`):
- Extract method name, category (auth/channel/transfer/query/app-session)
- Extract parameters with types
- Extract return type/description
- Extract code examples
- Store in structured API registry (dict, not vector-searched)

## Search Engine

### Hybrid Search (Reciprocal Rank Fusion)

```python
def hybrid_search(query, limit=5, category=None):
    # 1. BM25 keyword search
    bm25_results = bm25_index.get_top_n(tokenize(query), chunks, n=20)

    # 2. Vector cosine similarity
    query_embedding = model.encode(query)
    similarities = cosine_similarity(query_embedding, chunk_embeddings)
    vector_results = top_k(similarities, k=20)

    # 3. Optional category filter
    if category:
        bm25_results = [r for r in bm25_results if r.category == category]
        vector_results = [r for r in vector_results if r.category == category]

    # 4. RRF merge
    k = 60
    scores = {}
    for rank, result in enumerate(bm25_results):
        scores[result.id] = scores.get(result.id, 0) + 1 / (k + rank + 1)
    for rank, result in enumerate(vector_results):
        scores[result.id] = scores.get(result.id, 0) + 1 / (k + rank + 1)

    return sorted(scores, reverse=True)[:limit]
```

### Embedding Model

- **Model:** `all-MiniLM-L6-v2` (sentence-transformers)
- **Dimensions:** 384
- **Size:** ~80MB download
- **Speed:** ~50ms per embedding, full index builds in <5 seconds

## Project Structure

```
yellow-docs-mcp/
├── pyproject.toml
├── README.md
├── src/
│   └── yellow_docs_mcp/
│       ├── __init__.py
│       ├── server.py        # MCP server, tool definitions, startup logic
│       ├── indexer.py        # Git operations + orchestrate parse/embed
│       ├── parser.py         # MDX/MD parsing, section splitting, API extraction
│       ├── search.py         # BM25 + vector search + RRF merge
│       └── store.py          # Index persistence (pickle + numpy arrays)
└── claude_mcp_config.json   # Example MCP config for Claude Code/Desktop
```

## Dependencies

```toml
[project]
dependencies = [
    "mcp[cli]>=1.0",
    "sentence-transformers>=2.2",
    "rank-bm25>=0.2",
    "numpy>=1.24",
    "pyyaml>=6.0",
    "gitpython>=3.1",
]
```

## Configuration (for users)

### Claude Code (`~/.claude/mcp.json`)
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

### Claude Desktop (`claude_desktop_config.json`)
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

## Success Criteria

1. `search_docs("how do state channels work")` returns relevant sections from learn/ and protocol/
2. `get_api_reference("create_channel")` returns the full method spec with params and examples
3. `get_code_examples("connect to clearnode")` returns TypeScript snippets
4. `list_sections("protocol")` returns the full protocol doc tree
5. Full startup (clone + index) < 30 seconds; cached startup < 3 seconds
6. Works in both Claude Code and Claude Desktop
