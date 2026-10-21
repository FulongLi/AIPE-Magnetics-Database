# MCP server

`aipe-magnetics-mcp` (or `python -m aipe_magnetics.mcp.server`) is a local **stdio** Model Context
Protocol server built on the official Python SDK `mcp>=2` (`mcp.server.mcpserver.MCPServer`,
protocol revision 2026-07-28). It exposes the Python API one-to-one; the server contains no
engineering logic of its own.

| Property | Value |
| --- | --- |
| transport | stdio only (no network listener) |
| credentials | none; no API keys, no cloud services |
| default mode | read-only data queries and deterministic calculations |
| file writes | only `generate_report` with `output_path`, only when started with `--allow-write --output-dir <dir>` (or `AIPE_MAGNETICS_MCP_ALLOW_WRITE=1|true|yes` plus `AIPE_MAGNETICS_MCP_OUTPUT_DIR`; any other spelling of the flag is an error), only inside that directory, never overwriting without `overwrite=true` (`destination_exists`) |
| code execution | the server never executes repository content or shell commands; `design_transformer_from_waveforms` only *reads* the waveform file named by the caller (HDF5 / CSV / JSON) |
| validation | every tool has a JSON input schema (pydantic) and a structured output schema; invalid argument types return `isError` results with the validation message; unknown or misspelled argument names are refused (`isError`, naming the accepted arguments) instead of being ignored; quantity parameters are free-form objects in the schema whose `{value, unit}` shape, dimension and sign are validated by the engine |

## Tools

| Tool | Annotation | Returns |
| --- | --- | --- |
| `list_capabilities` | read-only | capability table |
| `list_materials(query?, material_class?)` | read-only | catalogue listing |
| `get_material(material)` | read-only | datasheet parameters, Steinmetz sets, sources |
| `compare_materials(materials, frequency?, flux_density_peak?, temperature?)` | read-only | ranked comparison |
| `estimate_core_loss(material, frequency, flux_density_peak, temperature?, waveform?, model?, core?, parameter_set?, model_parameters?, policy?)` | read-only | loss density, core loss, model identity, domain check |
| `evaluate_hysteresis(field_peak, frequency?, field_dc?, waveform?, periods?=10, parameters?, samples_per_period?, steady_state_tol?, policy?)` | read-only | Jiles-Atherton loop quantities and solver diagnostics (defaults identical to the Python API) |
| `design_inductor(..., ripple_duty?=0.5)` | read-only | turns, gap, flux density, losses, candidates (every catalogue core offered in the material when `core` is null) |
| `design_transformer(...)` (secondaries may carry `power_share`) | read-only | turns, flux density, losses, ranked feasible candidates; incompletely evaluable cores are reported as not feasible |
| `design_transformer_from_waveforms(frequency, waveforms, turns_ratio_target, ...)` | read-only (reads the named waveform file) | ranked integer-turn candidates from the actual winding waveforms (exact RMS / harmonics, iGSE on the real flux, per-layer harmonic Dowell, leakage and external inductance, magnetising inductance, fill, thermal estimate), square-current comparison, material comparison, measurement requests; `export_state=true` with `created_at` adds the AIPE-Core Engineering State |
| `measurement_requests(frequency, materials?, flux_density_peak?, temperature?, temperature_range?, excitation?, dc_bias_field_max?, harmonics?, include_available?)` | read-only | per-material coverage of B-H, core loss vs f/B/T, DC bias, saturation vs T, complex permeability, winding AC resistance, thermal conductivity; requests with test conditions and priority |
| `validate_design(...)` | read-only | saturation and operating-point checks |
| `generate_report(results, title?, format?, output_path?, overwrite?, include_inputs?)` | non-destructive write (gated) | report text, written path when authorised |
| `verification_status(capability?, frequency?, temperature?, flux_density_peak?, material?, excitation?)` | read-only | evidence per capability / range |

Every result is an `aipe.magnetics.result/0.2` envelope: `status` (`ok`/`rejected`/`error`), `values` with
units and status (`calculated`, `estimated`, `looked_up`, `specified`, `assumed`), `assumptions`, `model`
(id, version, family), `sources`, `verification` (level and state: `not_run`, `verified`, `partially_verified`,
`out_of_range`, `unverified`, `failed`), `warnings`, `limitations`, `errors`. Quantities are passed as `{"value": 100, "unit": "kHz"}`; any registered unit is accepted
and normalised.

## Running and registering

```sh
python -m aipe_magnetics.mcp.server --self-test                 # lists tools, exit 0
python -m aipe_magnetics.mcp.server                             # stdio server (read-only)
python -m aipe_magnetics.mcp.server --allow-write --output-dir ./reports
```

Claude Code (project scope `.mcp.json` or CLI):

```sh
claude mcp add --transport stdio aipe-magnetics -- python -m aipe_magnetics.mcp.server
```

```json
{ "mcpServers": { "aipe-magnetics": { "command": "python", "args": ["-m", "aipe_magnetics.mcp.server"] } } }
```

Codex CLI (`~/.codex/config.toml`):

```toml
[mcp_servers.aipe-magnetics]
command = "python"
args = ["-m", "aipe_magnetics.mcp.server"]
```

or `codex mcp add aipe-magnetics -- python -m aipe_magnetics.mcp.server`.

## Testing

`tests/mcp/` connects an in-process `mcp.Client` to `build_server()` and also launches the server as
a subprocess over stdio (`StdioServerParameters`) to verify the real transport: tool listing,
structured results, error surfacing, write gating, and the absence of network access.
