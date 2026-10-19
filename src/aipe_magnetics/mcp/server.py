"""Local stdio MCP server exposing the AIPE Magnetics engineering API.

* Protocol: Model Context Protocol, Python SDK ``mcp>=2`` (``MCPServer``), stdio transport only.
* No API keys, no cloud services, no network listener.
* Read-only data queries and deterministic calculations are available by default. The only tool that
  can modify user files (``generate_report`` with ``output_path``) requires the server to be started
  with ``--allow-write`` (or ``AIPE_MAGNETICS_MCP_ALLOW_WRITE=1``) *and* an explicit output path inside
  the allowed output directory.
* The server never executes repository content or shell commands.

Run: ``python -m aipe_magnetics.mcp.server`` (or the ``aipe-magnetics-mcp`` console script).
Self-test: ``python -m aipe_magnetics.mcp.server --self-test`` lists the tools and exits.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from aipe_magnetics import TOOL_NAME, __version__
from aipe_magnetics.common.errors import AIPEMagneticsError, UnsupportedOperationError
from aipe_magnetics.common.results import EngineeringResult

INSTRUCTIONS = (
    "AIPE Magnetics engineering tools. Every tool returns an engineering result envelope "
    "(aipe.magnetics.result/0.2) with values + units, assumptions, model identity/version, sources, verification status, "
    "warnings, limitations and explicit errors (status 'rejected' when inputs are invalid or outside the validated domain). "
    "Physical values must be given as {'value': number, 'unit': '<unit>'} in any registered unit (e.g. kHz, mT, degC). "
    "Never report numbers that are not in a tool result; repeat the limitations and verification status to the user."
)

QUANTITY_DOC = "{'value': number, 'unit': str} in any registered unit (SI preferred; kHz, mT, degC accepted)"


class ServerConfig:
    def __init__(self, allow_write: bool = False, output_dir: str | Path | None = None):
        self.allow_write = allow_write
        self.output_dir = Path(output_dir).resolve() if output_dir else None


def _envelope(result: EngineeringResult) -> dict[str, Any]:
    return result.to_json_dict()


def build_server(config: ServerConfig | None = None):
    """Create the MCPServer with all tools registered (importable without starting any transport)."""
    try:
        from mcp.server.mcpserver import MCPServer
        from mcp.server.mcpserver.exceptions import ToolError
        from mcp.types import ToolAnnotations
    except ImportError as error:  # pragma: no cover - depends on environment
        raise UnsupportedOperationError("the MCP server requires the 'mcp' package (pip install 'aipe-magnetics[mcp]')") from error

    import aipe_magnetics.api as api

    config = config or ServerConfig()
    server = MCPServer("aipe-magnetics", title=TOOL_NAME, instructions=INSTRUCTIONS, version=__version__)
    read_only = ToolAnnotations(title=None, read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)

    def run(func, *args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return _envelope(func(*args, **kwargs))
        except AIPEMagneticsError as error:
            raise ToolError(f"{error.code}: {error.message}") from None
        except (TypeError, ValueError) as error:
            raise ToolError(f"invalid_input: {error}") from None

    @server.tool(name="list_capabilities", title="List capabilities", annotations=read_only,
                 description="List the engineering capabilities of this server with their ids and kinds.")
    def list_capabilities() -> dict[str, Any]:
        return run(api.list_capabilities)

    @server.tool(name="list_materials", title="List materials", annotations=read_only,
                 description="List catalogue magnetic materials, optionally filtered by a text query (id, name, grade) or material class.")
    def list_materials(query: str | None = None, material_class: str | None = None) -> dict[str, Any]:
        return run(api.list_materials, query, material_class)

    @server.tool(name="get_material", title="Get material", annotations=read_only,
                 description="Datasheet parameters, Steinmetz parameter sets with validity ranges, and source references of one material "
                             "(by id, name or grade such as 'N87').")
    def get_material(material: str) -> dict[str, Any]:
        return run(api.get_material, material)

    @server.tool(name="compare_materials", title="Compare materials", annotations=read_only,
                 description=f"Compare materials at an operating point: Steinmetz loss density and saturation margin. Quantities are {QUANTITY_DOC}.")
    def compare_materials(materials: list[str], frequency: dict[str, Any] | None = None, flux_density_peak: dict[str, Any] | None = None,
                          temperature: dict[str, Any] | None = None) -> dict[str, Any]:
        return run(api.compare_materials, materials, frequency, flux_density_peak, temperature)

    @server.tool(name="estimate_core_loss", title="Estimate core loss", annotations=read_only,
                 description=("Volumetric core loss (and total core loss when a core id is given) with the classical Steinmetz equation "
                              "(sinusoidal), iGSE (triangular/trapezoidal/piecewise-linear/sampled flux) or Bertotti (user coefficients). "
                              f"Quantities are {QUANTITY_DOC}. waveform: {{'kind': 'sinusoidal'|'triangular'|'trapezoidal'|'piecewise_linear'|'sampled', "
                              "'duty': 0.5, 'duty_rise': ..., 'duty_flat': ..., 'breakpoints': [[t_frac, B_frac], ...], 'samples': [...]}. "
                              "model: 'auto'|'steinmetz'|'igse'|'bertotti'. policy: 'reject' (default) or 'warn' for out-of-domain inputs."))
    def estimate_core_loss(material: str, frequency: dict[str, Any], flux_density_peak: dict[str, Any], temperature: dict[str, Any] | None = None,
                           waveform: dict[str, Any] | None = None, model: str = "auto", core: str | None = None,
                           parameter_set: str | None = None, model_parameters: dict[str, float] | None = None,
                           policy: str = "reject") -> dict[str, Any]:
        operating_point = {"frequency": frequency, "flux_density_peak": flux_density_peak}
        if temperature is not None:
            operating_point["temperature"] = temperature
        if waveform is not None:
            operating_point["waveform"] = waveform
        return run(api.estimate_core_loss, {"material": material, "operating_point": operating_point, "model": model, "core": core,
                                            "parameter_set": parameter_set, "model_parameters": model_parameters, "policy": policy})

    @server.tool(name="evaluate_hysteresis", title="Evaluate hysteresis loop", annotations=read_only,
                 description=("Jiles-Atherton B-H loop evaluation via the ODE framework: peak B, remanence, coercivity, loop energy density, "
                              f"solver diagnostics. field_peak and field_dc are {QUANTITY_DOC} (A/m); parameters: {{Ms, a, alpha, k, c}} in SI "
                              "(default: illustrative set, not a material fit)."))
    def evaluate_hysteresis(field_peak: dict[str, Any], frequency: dict[str, Any] | None = None, field_dc: dict[str, Any] | None = None,
                            waveform: str = "sinusoidal", periods: int = 4, parameters: dict[str, float] | None = None,
                            samples_per_period: int = 400, policy: str = "reject") -> dict[str, Any]:
        payload: dict[str, Any] = {"field_peak": field_peak, "waveform": waveform, "periods": periods, "parameters": parameters,
                                   "samples_per_period": samples_per_period, "policy": policy}
        if frequency is not None:
            payload["frequency"] = frequency
        if field_dc is not None:
            payload["field_dc"] = field_dc
        return run(api.evaluate_hysteresis, payload)

    @server.tool(name="design_inductor", title="Design inductor", annotations=read_only,
                 description=("Basic gapped inductor sizing: turns, air gap, peak flux density, saturation margin, core and copper loss, window fill. "
                              f"Quantities are {QUANTITY_DOC}. core: catalogue core id or null to screen the catalogue. winding: "
                              "{'conductor_type': 'round'|'litz'|'foil', 'current_density': {...A/m^2}, 'fill_factor': 0.4, ...}. "
                              "flux_limit: {'mode': 'fraction_of_saturation', 'fraction': 0.7} or {'mode': 'absolute', 'absolute': {...T}}."))
    def design_inductor(inductance: dict[str, Any], current_peak: dict[str, Any], current_rms: dict[str, Any], frequency: dict[str, Any],
                        material: str, core: str | None = None, current_ripple_peak_to_peak: dict[str, Any] | None = None,
                        temperature: dict[str, Any] | None = None, flux_limit: dict[str, Any] | None = None,
                        winding: dict[str, Any] | None = None, loss_model: str = "auto", max_candidates: int = 5,
                        policy: str = "warn") -> dict[str, Any]:
        payload: dict[str, Any] = {"inductance": inductance, "current_peak": current_peak, "current_rms": current_rms, "frequency": frequency,
                                   "material": material, "core": core, "loss_model": loss_model, "max_candidates": max_candidates, "policy": policy}
        for key, value in (("current_ripple_peak_to_peak", current_ripple_peak_to_peak), ("temperature", temperature),
                           ("flux_limit", flux_limit), ("winding", winding)):
            if value is not None:
                payload[key] = value
        return run(api.design_inductor, payload)

    @server.tool(name="design_transformer", title="Design transformer", annotations=read_only,
                 description=("Basic transformer sizing: candidate core screening, turns per winding, peak flux density, saturation margin, core loss "
                              "(iGSE for square/trapezoidal voltage), winding losses (Dowell), window fill, ranked candidates. "
                              f"Quantities are {QUANTITY_DOC}. primary/secondaries: {{'name': str, 'voltage': {{...V}}, 'current_rms'?: {{...A}}, "
                              "'winding'?: {...}}. voltage_waveform: 'square'|'sinusoidal'|'trapezoidal'; duty <= 0.5. cores: list of core ids or null."))
    def design_transformer(power: dict[str, Any], frequency: dict[str, Any], primary: dict[str, Any], secondaries: list[dict[str, Any]],
                           material: str, voltage_waveform: str = "square", duty: float = 0.5, cores: list[str] | None = None,
                           temperature: dict[str, Any] | None = None, flux_limit: dict[str, Any] | None = None,
                           loss_density_target: dict[str, Any] | None = None, loss_model: str = "auto", max_candidates: int = 5,
                           policy: str = "warn") -> dict[str, Any]:
        payload: dict[str, Any] = {"power": power, "frequency": frequency, "primary": primary, "secondaries": secondaries, "material": material,
                                   "voltage_waveform": voltage_waveform, "duty": duty, "cores": cores, "loss_model": loss_model,
                                   "max_candidates": max_candidates, "policy": policy}
        for key, value in (("temperature", temperature), ("flux_limit", flux_limit), ("loss_density_target", loss_density_target)):
            if value is not None:
                payload[key] = value
        return run(api.design_transformer, payload)

    @server.tool(name="validate_design", title="Validate design", annotations=read_only,
                 description=("Saturation screening and operating-point validation of an existing design (material, core, turns, voltage or flux "
                              f"density, frequency, temperature, optional current and air gap). Quantities are {QUANTITY_DOC}."))
    def validate_design(material: str, core: str, turns: int, frequency: dict[str, Any], temperature: dict[str, Any] | None = None,
                        voltage_peak: dict[str, Any] | None = None, voltage_waveform: str = "square", duty: float = 0.5,
                        current_peak: dict[str, Any] | None = None, air_gap: dict[str, Any] | None = None,
                        flux_density_peak: dict[str, Any] | None = None, flux_limit: dict[str, Any] | None = None,
                        loss_model: str = "auto") -> dict[str, Any]:
        payload: dict[str, Any] = {"material": material, "core": core, "turns": turns, "frequency": frequency, "voltage_waveform": voltage_waveform,
                                   "duty": duty, "loss_model": loss_model}
        for key, value in (("temperature", temperature), ("voltage_peak", voltage_peak), ("current_peak", current_peak), ("air_gap", air_gap),
                           ("flux_density_peak", flux_density_peak), ("flux_limit", flux_limit)):
            if value is not None:
                payload[key] = value
        return run(api.validate_design, payload)

    @server.tool(name="generate_report", title="Generate engineering report",
                 annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False),
                 description=("Render one or more previously obtained result envelopes into a Markdown or JSON engineering report. Returns the text. "
                              "Writing a file needs output_path AND a server started with --allow-write (the path must lie inside the allowed "
                              "output directory); otherwise the tool returns the content inline only."))
    def generate_report(results: list[dict[str, Any]], title: str = "AIPE Magnetics engineering report", format: str = "markdown",
                        output_path: str | None = None, overwrite: bool = False, include_inputs: bool = True) -> dict[str, Any]:
        try:
            envelopes = [EngineeringResult.model_validate(r) for r in results]
        except Exception as error:  # noqa: BLE001 - validation message is returned to the model
            raise ToolError(f"invalid_input: results must be aipe.magnetics.result/0.2 envelopes ({error})") from None
        if output_path and not config.allow_write:
            raise ToolError("authorization_required: file writing is disabled; start the server with --allow-write and an --output-dir")
        request = {"title": title, "format": format, "output_path": output_path, "allow_write": bool(output_path and config.allow_write),
                   "overwrite": overwrite, "include_inputs": include_inputs}
        payload = run(api.generate_report, envelopes, request, allowed_directory=config.output_dir)
        if payload.get("status") != "ok":
            # authorisation and path problems are protocol-level failures for the caller, not engineering results
            messages = "; ".join(f"{e.get('code')}: {e.get('message')}" for e in payload.get("errors", []))
            raise ToolError(messages or "report generation failed")
        return payload

    @server.tool(name="verification_status", title="Verification status", annotations=read_only,
                 description=("Verification evidence per capability and operating range (levels L1 software, L2 numerical/physical, "
                              f"L3 engineering). Optional capability id (e.g. 'aipe.magnetics.model.igse') and conditions ({QUANTITY_DOC})."))
    def verification_status(capability: str | None = None, frequency: dict[str, Any] | None = None, temperature: dict[str, Any] | None = None,
                            flux_density_peak: dict[str, Any] | None = None, material: str | None = None,
                            excitation: str | None = None) -> dict[str, Any]:
        return run(api.verification_status, capability, frequency=frequency, temperature=temperature, flux_density_peak=flux_density_peak,
                   material=material, excitation=excitation)

    return server


def self_test(config: ServerConfig) -> int:
    """List the registered tools synchronously; exit code 0 when the server builds and lists tools."""
    import anyio

    server = build_server(config)

    async def _list() -> list[str]:
        tools = await server.list_tools()
        return [t.name for t in tools]

    names = anyio.run(_list)
    print(json.dumps({"server": "aipe-magnetics", "version": __version__, "tools": names, "allow_write": config.allow_write,
                      "output_dir": str(config.output_dir) if config.output_dir else None}, indent=2))
    return 0 if names else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aipe-magnetics-mcp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--allow-write", action="store_true", default=os.environ.get("AIPE_MAGNETICS_MCP_ALLOW_WRITE", "0") == "1",
                        help="allow generate_report to write files (inside --output-dir)")
    parser.add_argument("--output-dir", default=os.environ.get("AIPE_MAGNETICS_MCP_OUTPUT_DIR"),
                        help="directory that written reports must stay inside (required with --allow-write)")
    parser.add_argument("--self-test", action="store_true", help="build the server, list its tools and exit")
    args = parser.parse_args(argv)
    if args.allow_write and not args.output_dir:
        parser.error("--allow-write requires --output-dir (or AIPE_MAGNETICS_MCP_OUTPUT_DIR)")
    config = ServerConfig(allow_write=args.allow_write, output_dir=args.output_dir)
    if args.self_test:
        return self_test(config)
    server = build_server(config)
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
