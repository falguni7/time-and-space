"""PressLens MCP server: the engine's read-only tools (tools.py) over the Model Context Protocol.

    bin/pl python -m presslens.mcp_server                 # stdio (how the agents start it)
    bin/pl python -m presslens.mcp_server --http 8765     # streamable HTTP, e.g. for an MCP inspector
"""

from __future__ import annotations

import argparse
import functools

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from . import tools

INSTRUCTIONS = """Read-only football metrics for the PressLens league (8 fictional clubs, 56 matches).
Start with list_matches / list_moments to find ids, get_moment for one moment's numbers, and
pass_options / pressure_at to see the situation at that frame. query_metrics answers aggregate
questions (call list_metrics for the catalogue). Quote only numbers these tools return, with
their sample size n when you compare groups."""

READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)


def _explain_errors(fn):
    """Return bad-argument errors as a normal result: the MCP server and the agent framework both
    hide an exception's text, and the model needs it ("allowed: [...]") to correct its call."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ValueError, KeyError) as exc:
            return {"error": str(exc).strip("'\"")}
    return wrapper


def build_server() -> MCPServer:
    server = MCPServer(name="presslens-metrics", instructions=INSTRUCTIONS)
    for fn in tools.TOOLS:
        server.add_tool(_explain_errors(fn), annotations=READ_ONLY)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="PressLens MCP server")
    parser.add_argument("--http", type=int, metavar="PORT", help="serve streamable HTTP instead of stdio")
    args = parser.parse_args()
    server = build_server()
    if args.http:
        server.run("streamable-http", port=args.http)
    else:
        server.run("stdio")


if __name__ == "__main__":
    main()
