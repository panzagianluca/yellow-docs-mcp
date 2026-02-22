# Yellow Network Docs MCP

> **Unofficial** — This is a community-built MCP server, not officially maintained by Yellow Network.

MCP server that indexes [Yellow Network documentation](https://github.com/layer-3/docs) and provides structured retrieval tools for AI assistants (Claude Code, Claude Desktop, etc.).

**5 tools** for searching, browsing, and retrieving Yellow Network / Nitrolite protocol docs — grounded in the actual documentation, not hallucinated.

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
| `search_docs` | Hybrid semantic + keyword search across all docs |
| `get_page` | Fetch a full doc page by path or topic |
| `list_sections` | Browse documentation structure |
| `get_api_reference` | Look up protocol RPC methods and SDK API |
| `get_code_examples` | Find code snippets by topic |

## How it works

On first run, the server:
1. Clones the [docs repo](https://github.com/layer-3/docs) locally
2. Parses all MDX/MD files into structured sections
3. Builds a hybrid search index (BM25 keywords + sentence-transformers vector embeddings)
4. Caches everything for fast startup (~3s on subsequent runs)

On subsequent runs, it pulls the latest changes and re-indexes only if content changed.

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
