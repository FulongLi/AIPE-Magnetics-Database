"""``python -m aipe_magnetics.mcp`` starts the stdio MCP server (same as ``python -m aipe_magnetics.mcp.server``)."""

from aipe_magnetics.mcp.server import main

if __name__ == "__main__":
    raise SystemExit(main())
