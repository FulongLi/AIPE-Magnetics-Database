"""``python -m aipe_magnetics_database.mcp`` starts the stdio MCP server (same as ``python -m aipe_magnetics_database.mcp.server``)."""

from aipe_magnetics_database.mcp.server import main

if __name__ == "__main__":
    raise SystemExit(main())
