"""How an agent attaches the metrics server: an Agent Framework MCP tool that starts it over stdio.

    async with metrics_tool() as mcp:
        agent = Agent(client=..., tools=[mcp], ...)
"""

from __future__ import annotations

import os
import sys

from agent_framework import MCPStdioTool

from ..paths import ROOT


class _StaticMCPStdioTool(MCPStdioTool):
    """Our server's tool list never changes at runtime, so skip the framework's list-change watcher
    (agent-framework-core 1.23 needs the optional `tenacity` package just to start it)."""

    async def _listen_capability_list_changes(self, mcp_client) -> None:
        return None


def metrics_tool(allowed_tools: list[str] | None = None, name: str = "presslens_metrics") -> MCPStdioTool:
    """The server runs with this interpreter (i.e. inside the container) and the repo's src/ importable."""
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    return _StaticMCPStdioTool(
        name=name,
        command=sys.executable,
        args=["-m", "presslens.mcp_server"],
        env=env,
        allowed_tools=allowed_tools,
        load_prompts=False,
        request_timeout=60,
    )
