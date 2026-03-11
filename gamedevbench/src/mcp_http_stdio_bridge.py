#!/usr/bin/env python3
"""
Bridge an MCP streamable-http endpoint to stdio transport.

Gemini CLI can then use this bridge as a stdio MCP server to avoid
proxy-related issues on local HTTP MCP endpoints.
"""

import argparse
import asyncio
import os
from datetime import timedelta
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, ListToolsResult
from mcp.types import Tool
from mcp.server.lowlevel.server import NotificationOptions


DEFAULT_REMOTE_URL = "http://127.0.0.1:6601/mcp"
DEFAULT_TIMEOUT_SECONDS = 60


def _force_local_no_proxy() -> None:
    # This bridge should always reach a local MCP endpoint directly.
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
    os.environ["no_proxy"] = "127.0.0.1,localhost,::1"


class HttpToStdioBridge:
    def __init__(self, remote_url: str, timeout_seconds: int):
        self.remote_url = remote_url
        self.timeout_seconds = timeout_seconds
        self.server = Server("threejs-http-stdio-bridge")
        self._register_handlers()

    async def _with_remote_session(self) -> Any:
        client_cm = streamablehttp_client(url=self.remote_url)
        streams = await client_cm.__aenter__()
        try:
            try:
                read, write, _ = streams
            except Exception:
                read, write = streams
            session = ClientSession(
                read,
                write,
                read_timeout_seconds=timedelta(seconds=self.timeout_seconds),
            )
            await session.__aenter__()
            try:
                await session.initialize()
                return client_cm, session
            except Exception:
                await session.__aexit__(None, None, None)
                await client_cm.__aexit__(None, None, None)
                raise
        except Exception:
            await client_cm.__aexit__(None, None, None)
            raise

    async def _close_remote_session(self, client_cm: Any, session: ClientSession) -> None:
        await session.__aexit__(None, None, None)
        await client_cm.__aexit__(None, None, None)

    def _register_handlers(self) -> None:
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            client_cm, session = await self._with_remote_session()
            try:
                result: ListToolsResult = await session.list_tools()
                return result.tools
            finally:
                await self._close_remote_session(client_cm, session)

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
            client_cm, session = await self._with_remote_session()
            try:
                result: CallToolResult = await session.call_tool(name, arguments or {})
                # Return the full CallToolResult so structuredContent is preserved.
                return result
            finally:
                await self._close_remote_session(client_cm, session)

    async def run(self) -> None:
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="threejs-http-stdio-bridge",
                    server_version="0.1.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge MCP streamable-http endpoint to stdio.")
    parser.add_argument("--url", default=DEFAULT_REMOTE_URL, help="Remote MCP streamable-http endpoint URL")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="Remote MCP read timeout (seconds)")
    return parser.parse_args()


def main() -> None:
    _force_local_no_proxy()
    args = parse_args()
    bridge = HttpToStdioBridge(remote_url=args.url, timeout_seconds=args.timeout)
    asyncio.run(bridge.run())


if __name__ == "__main__":
    main()
