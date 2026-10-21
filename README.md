# AIPE Magnetics

**Evidence-first magnetic engineering expert for power electronics** - a Python engine with a
unified magnetic data standard, loss and hysteresis models with per-capability verification evidence,
inductor/transformer design calculations, usable as a library, a CLI, a local MCP server
and as a standalone plugin for **OpenAI Codex** and **Claude Code**. Part of
[AIPE - AI for Power Engineering](https://github.com/FulongLi/AIPE-Core).

Version 0.2.0 (`feature/aipe-magnetics-v0.2`). The v0.1 record scaffold is preserved unchanged; see
[docs/audit-v0.1.md](docs/audit-v0.1.md) and [CHANGELOG.md](CHANGELOG.md).

## Capabilities

| Area | What you get |
| --- | --- |
| Data | Magnetic Data Schema v0.2 (materials, manufacturers, datasheet parameters, cores, windings, measurements, B-H loops, loss tables, waveforms, model metadata), SI with original units, checksums, provenance, licence; v0.1 migration; HDF5 waveform store; bundled catalogue (TDK N87 datasheet, 10 ferrites with MagNet-derived Steinmetz sets, 5 TDK cores); source adapters for Princeton MagNet, MagNet Challenge 2, Paderborn materialdatabase and mag-net-hub |
| Models | Steinmetz, iGSE, Bertotti, Jiles-Atherton (ODE framework), data-driven adapter (ANN / LSTM / Transformer / Neural ODE; numpy + optional torch / magnethub backends) behind one Magnetic Model Contract with declared validity domains |
| Engineering | material lookup and comparison, flux density, core-loss estimation, inductor and transformer sizing (every catalogue core offered in the material evaluated and ranked), saturation screening, Dowell winding loss, operating-point validation, Markdown/JSON reports; **waveform-informed transformer design** from the converter's actual waveforms (HDF5 / CSV / JSON import or SPS synthesis: exact RMS and harmonics, flux from the voltage integral, iGSE on the real flux, per-layer harmonic Dowell copper loss, leakage and external inductance, thermal estimate, ranked integer-turn candidates, material comparison) and structured **material measurement requests** |
| Verification | evidence per capability and operating range (L1 software, L2 numerical/physical, L3 engineering), check suite with machine- and human-readable reports, verification status in every result derived from the bundled evidence file (`L0 unverified`, stated explicitly, until the release step records that file) |
| Interfaces | typed Python API, `aipe-magnetics` CLI, stdio MCP server, Codex plugin, Claude Code plugin, AIPE-Core round-trip adapter, AIPE-Registry manifest |

Every result is an envelope with values + units, assumptions, model identity and version, input
conditions, sources, verification status, warnings and limitations. Invalid or out-of-domain inputs
are rejected explicitly, never clipped; uncertainty is reported as unknown when it is unknown.

## Architecture

```
api / cli / mcp / plugins  ->  engineering  ->  models (contract, empirical, physics, dynamic, data-driven)
                                    |                |
                                   data (standard, catalogue, HDF5, adapters)   verification (evidence, checks)
                                                      integrations (AIPE-Core, AIPE-Registry)
```

Mathematics never imports from the AI layers; see [docs/architecture.md](docs/architecture.md).

## Installation

Python 3.10 or newer. No orchestrator, registry service, other domain expert, commercial software or
LLM credentials are required.

```sh
python -m pip install "aipe-magnetics[mcp] @ git+https://github.com/FulongLi/AIPE-Magnetics-Database@feature/aipe-magnetics-v0.2"
# development checkout
git clone https://github.com/FulongLi/AIPE-Magnetics-Database && cd AIPE-Magnetics-Database
python -m pip install -e ".[dev]"
python -m pytest -q
```

Extras: `[mcp]` MCP server, `[hdf5]` waveform store, `[torch]` optional data-driven backend.

### Codex

```sh
codex plugin marketplace add https://github.com/FulongLi/AIPE-Magnetics-Database
codex plugin add aipe-magnetics@aipe-magnetics        # skills + MCP server; start a new session
```

### Claude Code

```text
/plugin marketplace add FulongLi/AIPE-Magnetics-Database     # once v0.2 is on the default branch; until then: /plugin marketplace add /path/to/checkout
/plugin install aipe-magnetics@aipe-magnetics
```

Details, uninstall and troubleshooting: [docs/plugins.md](docs/plugins.md).

## Quick start

```sh
aipe-magnetics info
aipe-magnetics materials get N87
aipe-magnetics materials compare N87 3C94 N49 -f "100 kHz" -b "100 mT" -t "90 degC"
aipe-magnetics loss estimate N87 -f "100 kHz" -b "150 mT" -t "100 degC" --waveform triangular --core core.tdk-e65-32-27
aipe-magnetics design transformer -P "10 kW" -f "100 kHz" --primary-voltage "800 V" --secondary-voltage "400 V" --material N87 -t "100 degC" --loss-target "300 kW/m^3"
aipe-magnetics demo dab-10kw --output reports/demo
aipe-magnetics verify --output reports/verification --check-bundled   # exits 1 until the maintainer's bundled evidence file exists
aipe-magnetics mcp self-test
```

```python
import aipe_magnetics.api as api
r = api.estimate_core_loss(material="N87", frequency={"value": 100, "unit": "kHz"},
                           flux_density_peak={"value": 150, "unit": "mT"}, temperature={"value": 100, "unit": "degC"},
                           waveform={"kind": "triangular"})
print(r.values["loss_density"].quantity, r.model.id, r.verification.state)
```

## Example results

`aipe-magnetics demo dab-10kw` sizes a 10 kW, 800 V : 400 V, 100 kHz dual-active-bridge transformer in
TDK N87 (square voltage, triangular flux, 100 degC, flux limited to 70 % of Bsat and to a 300 kW/m^3
loss density). Output of the release run (`docs/demo-dab-10kw-report.md`, selected core E 65/32/27):

| Quantity | Value | Status |
| --- | --- | --- |
| turns primary : secondary | 21 : 10 | calculated (Faraday, integer turns) |
| peak flux density | 0.178 T (Bsat at 100 degC: 0.39 T, margin 54 %) | calculated |
| core loss density / core loss | 274 kW/m^3 / 21.5 W | estimated (iGSE, TDK 100 degC datasheet-fit set) |
| copper loss (litz, Dowell 1-D) | 17.7 W | estimated |
| total estimated loss | 39.2 W | estimated |
| window fill | 0.30 | estimated |
| verification | L2 (sizing relations and iGSE verified against analytical references) | - |

Other candidates: E 70/33/32 44.8 W, PM 87/70 44.9 W; ETD 59/31/22 rejected (window fill 0.51 > 0.40);
UU 93/152/30 not evaluable (no mean turn length / window area in the datasheet record). See
[docs/demo-dab-10kw.md](docs/demo-dab-10kw.md) for the specification and assumptions and
[docs/release-verification-v0.2.0.md](docs/release-verification-v0.2.0.md) for the record of what was
executed for the release. Values are analytical design estimates with stated assumptions; no
hardware was built or measured.

## Current limitations

* `design_transformer` is analytical sizing with the `P / V` square-current approximation; the
  waveform-informed design adds exact waveform RMS / harmonics, 1-D leakage, per-layer Dowell and a
  network / lumped thermal estimate, still without finite-element, insulation or manufacturing analysis
  (the v0.3 FEA / thermal-FE workflow layer is present but not yet released). Results are design
  estimates with explicit assumptions.
* Core-loss models need a parameter set covering the operating temperature, frequency and flux
  density; the engine refuses to extrapolate. Steinmetz/iGSE ignore DC bias and minor loops (flagged).
* Datasheet parameters are transcribed for N87 only; other catalogue materials carry MagNet-derived
  loss parameters but no Bsat/Curie data yet (reported as missing, not guessed).
* Jiles-Atherton ships an illustrative parameter set, not a material fit. Data-driven inference is
  demonstrated on a synthetic reference; real challenge models need the optional `magnethub` package.
* Verification checks are L1/L2 (software, numerical and physical consistency). No L3 experimental
  validation is bundled; optional external tests exist for users with the source datasets.
* The bundled evidence file `src/aipe_magnetics/verification/evidence/v0.2.0.json` is written by the
  maintainer's release step (`verify --update-bundled`); until then every model-based result reports
  `L0 unverified` with a statement naming the missing file, and `verify --check-bundled` exits 1.
* The repository licence is `NOASSERTION` pending the owner's decision ([LICENSING.md](LICENSING.md)).

## Reference projects and attribution

Princeton MagNet (MIT), MagNet Challenge 2 (no licence; reference only), Paderborn LEA
materialdatabase (GPL-3.0; no code or data copied), Paderborn mag-net-hub (MIT), AIPE-Core and
AIPE-Registry (MIT). Equations follow Steinmetz, Venkatachalam et al. 2002 (iGSE), Bertotti 1988,
Jiles & Atherton 1986 / Jiles, Thoelke & Devine 1992, Dowell 1966. See [docs/sources.md](docs/sources.md).

## Documentation

[architecture](docs/architecture.md) · [v0.1 audit](docs/audit-v0.1.md) · [data standard v0.2](docs/data-standard-v0.2.md) ·
[models](docs/models.md) · [verification](docs/verification.md) · [API and CLI](docs/api.md) · [MCP](docs/mcp.md) ·
[plugins](docs/plugins.md) · [Core/Registry integration](docs/core-integration.md) · [sources](docs/sources.md) ·
[DAB demo](docs/demo-dab-10kw.md) · [waveform-informed design](docs/waveform-informed-design.md) · [licensing](LICENSING.md) · [contributing](CONTRIBUTING.md)

## Licence

No licence has been declared by the repository owner (`NOASSERTION`). Third-party components keep
their own licences; nothing is relicensed. See [LICENSING.md](LICENSING.md).
