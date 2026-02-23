# Yellow Network Docs MCP

> **Unofficial** — This is a community-built MCP server, not officially maintained by Yellow Network.

MCP server that indexes Yellow Network documentation repositories and provides structured retrieval tools for AI assistants (Claude Code, Claude Desktop, etc.).

**5 tools** for searching, browsing, and retrieving Yellow Network / Nitrolite protocol docs across multiple repos.

## Install

```bash
pip install yellow-docs-mcp
```

Or with uv:
```bash
uv pip install yellow-docs-mcp
```

## Configure

### Claude Code

Add to `~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "yellow-docs": {
      "command": "yellow-docs-mcp"
    }
  }
}
```

### Claude Desktop

Add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "yellow-docs": {
      "command": "yellow-docs-mcp"
    }
  }
}
```

## Tools

| Tool | Description |
|------|-------------|
| `search_docs` | Hybrid semantic + keyword search across all indexed docs (with optional `source` filter) |
| `get_page` | Fetch a full doc page by path or topic (with optional `source` filter) |
| `list_sections` | Browse documentation structure by source/category |
| `get_api_reference` | Look up protocol RPC methods and SDK API (with optional `source` filter) |
| `get_code_examples` | Find code snippets by topic/language with output limits |

## Indexed Repositories (default)

- [layer-3/docs](https://github.com/layer-3/docs)
- [layer-3/clearsync](https://github.com/layer-3/clearsync)
- [layer-3/docs-gitbook](https://github.com/layer-3/docs-gitbook)
- [layer-3/nitro](https://github.com/layer-3/nitro)
- [layer-3/release-process](https://github.com/layer-3/release-process)

## How it works

On first run, the server:
1. Clones configured Layer-3 repositories locally
2. Parses all MDX/MD files into structured sections
3. Builds a hybrid search index (BM25 keywords + optional sentence-transformers embeddings)
4. Caches everything for fast startup (~3s on subsequent runs)

On subsequent runs, it syncs repos and re-indexes only if content changed. Index cache format is versioned and migrates forward automatically.

## Runtime tuning

The server can be tuned via environment variables:

- `YELLOW_DOCS_REPOS`: override sources (comma-separated `name=url` or URLs)
- `YELLOW_DOCS_DISABLE_VECTORS`: disable vector embeddings (`1/true/yes/on`)
- `YELLOW_DOCS_MAX_CHUNK_CHARS`: max chunk size for long sections
- `YELLOW_DOCS_CHUNK_OVERLAP_CHARS`: chunk overlap size
- `YELLOW_DOCS_BM25_WEIGHT`: BM25 weight in weighted RRF
- `YELLOW_DOCS_VECTOR_WEIGHT`: vector weight in weighted RRF
- `YELLOW_DOCS_RRF_K`: RRF denominator constant
- `YELLOW_DOCS_CANDIDATE_POOL`: max candidates per ranking stage
- `YELLOW_DOCS_MAX_FILE_BYTES`: skip markdown files above this size

## Example queries

Once configured, you can ask your AI assistant:

- *"How do state channels work in Yellow Network?"*
- *"What are the parameters for create_channel?"*
- *"Show me code examples for connecting to ClearNode"*
- *"List all available RPC query methods"*
- *"What chains does Yellow Network support?"*

## Development

```bash
git clone https://github.com/panzagianluca/yellow-docs-mcp.git
cd yellow-docs-mcp
uv sync
uv run pytest tests/ -v
```

## Author

Built by [Gianluca Panza](https://x.com/gianlucapanz) ([@gianlucapanz](https://x.com/gianlucapanz))

## License

MIT
