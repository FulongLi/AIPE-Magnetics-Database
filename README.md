# AIPE — Magnetics Database

AIPE Magnetics is the evidence-first magnetic engineering expert of **AI for Power Engineering**: a Python
engine with a unified magnetic data standard, core-loss and hysteresis models with per-capability
verification evidence, and inductor/transformer design calculations. Every result is an envelope with
values and units, assumptions, model identity, sources, verification status, warnings and limitations;
out-of-domain inputs are rejected, never clipped.

## Features

- Magnetic Data Schema v0.2 in SI with checksums, provenance and licence; bundled TDK N87 datasheet
  values, ten ferrites with MagNet-derived Steinmetz sets and five TDK cores
- Steinmetz, iGSE, Bertotti, Jiles-Atherton and data-driven models behind one contract with declared
  validity domains
- Material comparison, core-loss estimation, inductor and transformer sizing over the catalogue,
  saturation screening, Dowell winding loss and design validation
- Waveform-informed transformer design from the converter's actual waveforms, plus structured material
  measurement requests
- Verification evidence (L1 software, L2 numerical/physical, L3 engineering) in every result
- Library, CLI, stdio MCP server, Codex and Claude Code plugins, AIPE-Core round-trip adapter

## Installation

Python 3.10 or newer; no orchestrator, registry service, commercial software or LLM credentials. Extras:
`[mcp]` MCP server, `[hdf5]` waveform store, `[torch]` data-driven backend. Codex and Claude Code plugins:
[docs/plugins.md](docs/plugins.md).

```sh
python -m pip install -e ".[dev]"
```

## Usage

```sh
aipe-magnetics info
aipe-magnetics materials compare N87 3C94 N49 -f "100 kHz" -b "100 mT" -t "90 degC"
aipe-magnetics loss estimate N87 -f "100 kHz" -b "150 mT" -t "100 degC" --waveform triangular --core core.tdk-e65-32-27
aipe-magnetics demo dab-10kw --output reports/demo
python -m pytest -q
```

```python
import aipe_magnetics.api as api
r = api.estimate_core_loss(material="N87", frequency={"value": 100, "unit": "kHz"},
                           flux_density_peak={"value": 150, "unit": "mT"}, temperature={"value": 100, "unit": "degC"},
                           waveform={"kind": "triangular"})
print(r.values["loss_density"].quantity, r.model.id, r.verification.state)
```

The demo sizes a 10 kW, 800 V : 400 V, 100 kHz dual-active-bridge transformer in N87 (E 65/32/27, 21 : 10
turns, 39.2 W estimated loss; [docs/demo-dab-10kw.md](docs/demo-dab-10kw.md)): analytical estimates with
stated assumptions, nothing built or measured.

## Project Information

- Version: 0.2.0 (`pyproject.toml`, package `aipe-magnetics`; unreleased additions are listed in
  [CHANGELOG.md](CHANGELOG.md))
- Status: Prototype (`aipe.yaml`: `experimental` / `prototype`; package classifier
  `Development Status :: 3 - Alpha`); model results report `L0 unverified` until the maintainer's bundled
  evidence file exists
- Developer: Fulong Li
- Organisation: Spirit Connect / AIPE Labs
- License: not declared — `NOASSERTION` in `pyproject.toml`, `aipe.yaml` and the plugin manifests; code
  rights are reserved by the contributors until the owner selects a licence. Third-party components keep
  their own licences and nothing is relicensed ([LICENSING.md](LICENSING.md),
  [docs/sources.md](docs/sources.md))

## Documentation

- [Architecture and capability overview](docs/architecture.md) ·
  [Data standard v0.2](docs/data-standard-v0.2.md) · [Models](docs/models.md)
- [Python API and CLI](docs/api.md) · [MCP server](docs/mcp.md) · [Plugins](docs/plugins.md)
- [Verification](docs/verification.md) · [Release verification v0.2.0](docs/release-verification-v0.2.0.md)
- [DAB demo](docs/demo-dab-10kw.md) · [Waveform-informed design](docs/waveform-informed-design.md) ·
  [Limitations](docs/limitations.md)
- [Sources, adapters and attribution](docs/sources.md) · [Core/Registry integration](docs/core-integration.md) ·
  [v0.1 audit](docs/audit-v0.1.md)
- [CHANGELOG.md](CHANGELOG.md) · [LICENSING.md](LICENSING.md) · [CONTRIBUTING.md](CONTRIBUTING.md)
