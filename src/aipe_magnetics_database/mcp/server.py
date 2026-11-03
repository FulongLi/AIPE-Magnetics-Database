"""Local stdio MCP server of the AIPE Magnetics Database (read-only data access).

* Protocol: Model Context Protocol, Python SDK ``mcp>=2`` (``MCPServer``), stdio transport only; no network
  listener, no API keys, no cloud services; the server never writes files or executes repository content.
* Tools return plain JSON: materials, cores, measurement tables (row-limited), provenance and the source inventory.
  Physical conditions carry units (``"100 kHz"``, ``"50 kHz..200 kHz"``); bare numbers are rejected.
* No loss prediction or design tools: those belong to the AIPE-Magnetics-Design server.
* Unknown or misspelled tool arguments are rejected instead of being dropped silently.

Run: ``aipe-magnetics-db-mcp`` (or ``python -m aipe_magnetics_database.mcp.server``); ``--self-test`` lists the tools.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from aipe_magnetics_database import TOOL_NAME, __version__
from aipe_magnetics_database.common.errors import AIPEMagneticsError, UnsupportedOperationError

INSTRUCTIONS = (
    "AIPE Magnetics Database: standardised magnetic material, core and measurement data (schema v0.2, SI units). Every record "
    "states its source, licence, data_class (experimental_measurement, manufacturer_datasheet, manufacturer_tool_export, "
    "derived_from_measurement, fitted_parameters) and review status (third-party values are unreviewed). Give conditions with "
    "units, e.g. '100 kHz' or '50 kHz..200 kHz'. Report only values present in tool results, keep measurements and fitted "
    "parameters apart, and repeat quality flags. Loss prediction and design are not offered here (AIPE-Magnetics-Design)."
)
MAX_ROWS = 5000


def build_server():
    try:
        from mcp.server.mcpserver import MCPServer
        from mcp.server.mcpserver.exceptions import ToolError
        from mcp.types import ToolAnnotations
    except ImportError as error:  # pragma: no cover - depends on environment
        raise UnsupportedOperationError("the MCP server requires the 'mcp' package (pip install 'aipe-magnetics-database[mcp]')") from error

    import aipe_magnetics_database.api as api

    server = MCPServer("aipe-magnetics-database", title=TOOL_NAME, instructions=INSTRUCTIONS, version=__version__)
    server.middleware.append(_RejectUnknownArguments(server))
    ro = ToolAnnotations(title=None, read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)

    def run(func, *args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except AIPEMagneticsError as error:
            raise ToolError(f"{error.code}: {error.message}") from None
        except (TypeError, ValueError) as error:
            raise ToolError(f"invalid_input: {error}") from None

    @server.tool(name="database_summary", title="Database summary", annotations=ro, description="Record counts per source and of the canonical catalogue.")
    def database_summary() -> dict[str, Any]:
        return run(api.summary)

    @server.tool(name="list_sources", title="List sources", annotations=ro,
                 description="Every investigated data source: status (integrated / reference / unavailable), version, licence, redistribution and reason.")
    def list_sources(status: str | None = None) -> list[dict[str, Any]]:
        return run(api.list_sources, status=status)

    @server.tool(name="list_materials", title="List materials", annotations=ro,
                 description="Canonical materials (deduplicated across sources) filtered by text, manufacturer, material_class or source id.")
    def list_materials(query: str | None = None, manufacturer: str | None = None, material_class: str | None = None, source: str | None = None,
                       limit: int = 100) -> list[dict[str, Any]]:
        return run(api.list_materials, query=query, manufacturer=manufacturer, material_class=material_class, source=source)[: max(1, min(limit, 1000))]

    @server.tool(name="get_material", title="Get material", annotations=ro,
                 description="Canonical material record by id, name ('TDK N87') or unique grade; lists member source records and measurements.")
    def get_material(material: str) -> dict[str, Any]:
        return run(api.get_material, material)

    @server.tool(name="material_properties", title="Material properties", annotations=ro,
                 description="Typed properties from every contributing source record plus fitted loss-model coefficients (fitted_parameters).")
    def material_properties(material: str) -> dict[str, Any]:
        return run(api.material_properties, material)

    @server.tool(name="list_cores", title="List cores", annotations=ro,
                 description="Core shapes (dimension tolerance ranges) and manufacturer parts; kind: all | shapes | parts.")
    def list_cores(query: str | None = None, family: str | None = None, manufacturer: str | None = None, material: str | None = None,
                   kind: str = "all", limit: int = 50) -> list[dict[str, Any]]:
        return run(api.list_cores, query=query, family=family, manufacturer=manufacturer, material=material, kind=kind, limit=max(1, min(limit, 500)))

    @server.tool(name="get_core", title="Get core", annotations=ro, description="Core shape or part record by id, name, part number or alias.")
    def get_core(core: str) -> dict[str, Any]:
        return run(api.get_core, core)

    @server.tool(name="find_measurements", title="Find measurements", annotations=ro,
                 description="Measurement tables whose condition ranges overlap frequency / temperature / flux_density (peak) / dc_bias "
                             "(e.g. '100 kHz', '25 degC..100 degC'); filter by material, quantity (core_loss, complex_permeability, bh_curve ...), "
                             "data_class, source, waveform. Duplicates of other sources are hidden unless include_duplicates.")
    def find_measurements(material: str | None = None, quantity: str | None = None, data_class: str | None = None, source: str | None = None,
                          frequency: str | None = None, temperature: str | None = None, flux_density: str | None = None, dc_bias: str | None = None,
                          waveform: str | None = None, include_duplicates: bool = False) -> list[dict[str, Any]]:
        return run(api.find_measurements, material=material, quantity=quantity, data_class=data_class, source=source, frequency=frequency,
                   temperature=temperature, flux_density=flux_density, dc_bias=dc_bias, waveform=waveform, include_duplicates=include_duplicates)

    @server.tool(name="get_table", title="Get measurement table", annotations=ro,
                 description=f"Rows of one measurement table (SI units, column definitions), filtered by condition ranges; at most {MAX_ROWS} rows.")
    def get_table(record_id: str, frequency: str | None = None, temperature: str | None = None, flux_density: str | None = None,
                  dc_bias: str | None = None, waveform: str | None = None, limit: int = 200) -> dict[str, Any]:
        return run(api.get_table, record_id, frequency=frequency, temperature=temperature, flux_density=flux_density, dc_bias=dc_bias,
                   waveform=waveform, limit=max(1, min(limit, MAX_ROWS)))

    @server.tool(name="provenance", title="Provenance", annotations=ro,
                 description="Evidence, licence, transformation history, quality flags, dataset and duplicate status of a record.")
    def provenance(record_id: str) -> dict[str, Any]:
        return run(api.provenance, record_id)

    return server


class _RejectUnknownArguments:
    """Server middleware: a ``tools/call`` whose arguments name a property the tool does not declare is refused.

    The SDK's generated argument model ignores unknown keys, so a misspelled optional argument (``model_paramters``)
    would silently fall back to the default and return a plausible number. The refusal is an ``isError`` tool result
    naming the unknown argument(s) and the accepted ones.
    """

    def __init__(self, server: Any):
        self._server = server
        self._properties: dict[str, set[str]] | None = None

    async def _declared(self) -> dict[str, set[str]]:
        if self._properties is None:
            tools = await self._server.list_tools()
            self._properties = {t.name: set((t.input_schema or {}).get("properties", {})) for t in tools}
        return self._properties

    async def __call__(self, ctx: Any, call_next: Any) -> Any:
        if ctx.method == "tools/call" and isinstance(ctx.params, dict):
            name = ctx.params.get("name")
            arguments = ctx.params.get("arguments") or {}
            declared = (await self._declared()).get(name)
            if declared is not None and isinstance(arguments, dict):
                unknown = sorted(set(arguments) - declared)
                if unknown:
                    from mcp.types import CallToolResult, TextContent
                    message = (f"invalid_input: unknown argument(s) for tool {name}: {', '.join(unknown)}; accepted arguments: "
                               f"{', '.join(sorted(declared))}. Unknown arguments are refused rather than ignored so that a typo cannot "
                               "silently change the calculation.")
                    return CallToolResult(content=[TextContent(type="text", text=message)], isError=True)
        return await call_next(ctx)



def self_test() -> int:
    """Build the server and list its tools; exit code 0 when tools are registered."""
    import anyio

    server = build_server()

    async def _list() -> list[str]:
        return [t.name for t in await server.list_tools()]

    names = anyio.run(_list)
    print(json.dumps({"server": "aipe-magnetics-database", "version": __version__, "tools": names, "writes_files": False}, indent=2))
    return 0 if names else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aipe-magnetics-db-mcp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true", help="build the server, list its tools and exit")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    build_server().run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
