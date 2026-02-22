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
