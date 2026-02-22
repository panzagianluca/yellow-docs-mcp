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
