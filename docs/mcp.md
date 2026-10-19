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
| file writes | only `generate_report` with `output_path`, only when started with `--allow-write --output-dir <dir>` (or `AIPE_MAGNETICS_MCP_ALLOW_WRITE=1`, `AIPE_MAGNETICS_MCP_OUTPUT_DIR`), only inside that directory, never overwriting without `overwrite=true` |
| code execution | the server never executes repository content or shell commands |
| validation | every tool has a JSON input schema (pydantic) and a structured output schema; invalid arguments return `isError` results with the validation message |

## Tools

| Tool | Annotation | Returns |
| --- | --- | --- |
| `list_capabilities` | read-only | capability table |
| `list_materials(query?, material_class?)` | read-only | catalogue listing |
| `get_material(material)` | read-only | datasheet parameters, Steinmetz sets, sources |
| `compare_materials(materials, frequency?, flux_density_peak?, temperature?)` | read-only | ranked comparison |
| `estimate_core_loss(material, frequency, flux_density_peak, temperature?, waveform?, model?, core?, parameter_set?, model_parameters?, policy?)` | read-only | loss density, core loss, model identity, domain check |
| `evaluate_hysteresis(field_peak, frequency?, field_dc?, waveform?, periods?, parameters?, samples_per_period?, policy?)` | read-only | Jiles-Atherton loop quantities and solver diagnostics |
| `design_inductor(...)` | read-only | turns, gap, flux density, losses, candidates |
| `design_transformer(...)` | read-only | turns, flux density, losses, ranked candidates |
| `validate_design(...)` | read-only | saturation and operating-point checks |
| `generate_report(results, title?, format?, output_path?, overwrite?, include_inputs?)` | non-destructive write (gated) | report text, written path when authorised |
| `verification_status(capability?, frequency?, temperature?, flux_density_peak?, material?, excitation?)` | read-only | evidence per capability / range |

Every result is an `aipe.magnetics.result/0.2` envelope: `status` (`ok`/`rejected`), `values` with
units and status (`calculated`, `estimated`, `looked_up`, `assumed`), `assumptions`, `model`
(id, version, family), `sources`, `verification` (level and state), `warnings`, `limitations`,
`errors`. Quantities are passed as `{"value": 100, "unit": "kHz"}`; any registered unit is accepted
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
