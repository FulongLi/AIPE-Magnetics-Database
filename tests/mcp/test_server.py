"""MCP server: in-process client, structured output, error surfacing, write gating, stdio subprocess, offline."""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import anyio
import pytest

pytest.importorskip("mcp", reason="the MCP server tests need the 'mcp' extra; install aipe-magnetics[mcp]")

from mcp import Client  # noqa: E402
from mcp.client.stdio import StdioServerParameters  # noqa: E402

from aipe_magnetics.mcp.server import ServerConfig, build_server  # noqa: E402

EXPECTED_TOOLS = {"list_capabilities", "list_materials", "get_material", "compare_materials", "estimate_core_loss", "evaluate_hysteresis",
                  "design_inductor", "design_transformer", "validate_design", "generate_report", "verification_status"}


def run(coro):
    return anyio.run(coro)


def test_tool_listing_annotations_and_schemas():
    async def main():
        async with Client(build_server()) as client:
            tools = (await client.list_tools()).tools
            by_name = {t.name: t for t in tools}
            assert set(by_name) == EXPECTED_TOOLS
            for name, tool in by_name.items():
                assert tool.input_schema["type"] == "object"
                assert tool.output_schema is not None, name
                assert tool.annotations is not None
                if name != "generate_report":
                    assert tool.annotations.read_only_hint is True and tool.annotations.destructive_hint is False
            assert by_name["generate_report"].annotations.destructive_hint is False
    run(main)


def test_structured_results_and_errors_without_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network access attempted")

    # asyncio's Windows event loop uses a local socketpair for its self-pipe, so forbid name resolution
    # and outbound connections instead of socket creation itself
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)

    async def main():
        async with Client(build_server()) as client:
            caps = await client.call_tool("list_capabilities", {})
            assert caps.is_error is False and caps.structured_content["status"] == "ok"
            bad = await client.call_tool("estimate_core_loss", {"material": "N87", "frequency": {"value": 100, "unit": "kHz"},
                                                                 "flux_density_peak": 5})
            assert bad.is_error is True
            hyst = await client.call_tool("evaluate_hysteresis", {"field_peak": {"value": 500, "unit": "A/m"}, "periods": 3,
                                                                   "samples_per_period": 200})
            assert hyst.is_error is False
            assert hyst.structured_content["values"]["loop_energy_density"]["quantity"]["value"] > 0
            status = await client.call_tool("verification_status", {"capability": "aipe.magnetics.model.igse"})
            assert status.is_error is False and "verification" in status.structured_content
    run(main)


def test_write_is_gated(tmp_path):
    async def main():
        async with Client(build_server(ServerConfig(allow_write=False))) as client:
            caps = await client.call_tool("list_capabilities", {})
            refused = await client.call_tool("generate_report", {"results": [caps.structured_content], "output_path": str(tmp_path / "r.md")})
            assert refused.is_error is True and "authorization_required" in refused.content[0].text
            inline = await client.call_tool("generate_report", {"results": [caps.structured_content]})
            assert inline.is_error is False and inline.structured_content["data"]["content"].startswith("# ")
        async with Client(build_server(ServerConfig(allow_write=True, output_dir=tmp_path))) as client:
            caps = await client.call_tool("list_capabilities", {})
            ok = await client.call_tool("generate_report", {"results": [caps.structured_content], "output_path": "report.md"})
            assert ok.is_error is False and Path(ok.structured_content["data"]["written_to"]).exists()
            outside = await client.call_tool("generate_report", {"results": [caps.structured_content], "output_path": "../escape.md"})
            assert outside.is_error is True and "outside" in outside.content[0].text
            dup = await client.call_tool("generate_report", {"results": [caps.structured_content], "output_path": "report.md"})
            assert dup.is_error is True and "exists" in dup.content[0].text
    run(main)


def test_stdio_subprocess_transport():
    params = StdioServerParameters(command=sys.executable, args=["-m", "aipe_magnetics.mcp.server"])

    async def main():
        async with Client(params) as client:
            tools = (await client.list_tools()).tools
            assert set(t.name for t in tools) == EXPECTED_TOOLS
            result = await client.call_tool("list_capabilities", {})
            assert result.is_error is False
            assert client.server_info is not None and client.server_info.name == "aipe-magnetics"
    run(main)


def test_self_test_entry_point(capsys):
    from aipe_magnetics.mcp.server import main
    assert main(["--self-test"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["tools"]) == EXPECTED_TOOLS and payload["allow_write"] is False


def test_allow_write_requires_output_dir():
    from aipe_magnetics.mcp.server import main
    with pytest.raises(SystemExit):
        main(["--allow-write", "--self-test"])
