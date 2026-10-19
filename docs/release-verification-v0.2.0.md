# Release verification record - AIPE Magnetics 0.2.0

Branch `feature/aipe-magnetics-v0.2`, release run on 2026-10-19. Host: Windows 11 Pro for
Workstations (10.0.26300), Python 3.10.11 (project virtualenv created with uv 0.12.18), numpy 2.2.6,
scipy 1.15.3, pydantic 2.14.1, jsonschema 4.26.0, h5py 3.16.0, mcp 2.3.0 (protocol 2026-07-28),
Claude Code 2.1.80, codex-cli 0.163.0 (installed into a scratch directory for packaging checks).

This document records what was **actually executed**. Items marked *not executed* were not run and
are not claimed.

## 1. Automated test suite (executed)

| Command | Result |
| --- | --- |
| `python -m pytest tests -q` (whole repository, offline, project venv) | **1002 passed, 11 skipped, 2 warnings** in 28 s. Skips are the optional external tests, each with an explicit reason (`AIPE_ECOSYSTEM_CHECKOUTS not set`, `set AIPE_MAGNET_CHECKOUT`, `set AIPE_ALLOW_DOWNLOAD=1`, ...) and the `magnethub` package test (upstream needs Python ≥ 3.13). The 2 warnings are `PytestUnhandledThreadExceptionWarning` from the MCP stdio reader threads. |
| `python -m unittest discover -s tests -v` (the unchanged v0.1 workflow command) | `Ran 17 tests ... OK` |
| `python scripts/validate.py examples/synthetic-transformer.json` (v0.1 tool) | `PASS 1 records` |
| `python -m pytest tests -m external` with `AIPE_ECOSYSTEM_CHECKOUTS`, `AIPE_MAGNET_CHECKOUT`, `AIPE_MAGNETHUB_CHECKOUT`, `AIPE_MATERIALDATABASE_CHECKOUT`, `AIPE_MAGNETCHALLENGE2_CHECKOUT` pointing at local clones (commits 4e664f0 / f419260 for Core / Registry, 8a6892e, 1ed635d, 04fff8e, ab4913c) and `AIPE_ALLOW_DOWNLOAD=1` | **10 passed, 1 skipped** (the skip: `magnethub` not installed). Includes AIPE-Core's own `scripts/validate.py` run on our exported state, AIPE-Registry's own `scripts/validate.py` on our manifest, Princeton MagNet database extraction (N87 142 871 rows; 3E6 6 996 rows, pinned SHA-256 match), mag-net-hub fixture import (90 rows, 15 materials), materialdatabase discovery (75 items, blob SHA-1 match at the pinned commit, mT anomaly rejected), MagNet Challenge 2 user-file import, and one real pinned download (3E6 database, SHA-256 verified). |

## 2. Verification suite (executed)

`aipe-magnetics verify --output reports/verification --update-bundled` → **58 checks, 58 passed**
(stable after re-run). `aipe-magnetics verify --check-bundled` → `bundled evidence matches this run`.
Evidence file: `src/aipe_magnetics/verification/evidence/v0.2.0.json`; Markdown twin:
`docs/verification-report-v0.2.0.md`. Highest passing level per capability:

| Capability | Level | Capability | Level |
| --- | --- | --- | --- |
| model.steinmetz | L2 | model.igse | L2 |
| model.bertotti | L2 | model.jiles_atherton | L2 |
| model.ode_framework | L2 | model.datadriven.reference-steinmetz-ann (synthetic) | L2 |
| model.datadriven.lstm / neural-ode | L2 | model.datadriven (adapter) / model.waveforms | L1 |
| design_transformer | L2 | design_inductor | L2 |
| validate_design | L2 | engineering.formulas / engineering.winding | L2 |
| data.units | L2 | data.schema / catalog / hdf5 / import / migration / sources | L1 |
| integration.aipe-core / aipe-registry / magnethub | L1 | | |

L2 means agreement with analytical references, independent numerical implementations, published
reference values and physical-consistency requirements. **No L3 (experimental) evidence is bundled.**

## 3. Demonstration (executed)

`aipe-magnetics demo dab-10kw --output reports/demo` (public API only; see `docs/demo-dab-10kw.md`).
Report committed as `docs/demo-dab-10kw-report.md`. Headline numbers (E 65/32/27 selected):
21 : 10 turns, B_pk 0.178 T (Bsat(100 degC) 0.39 T, margin 54 %), iGSE loss density 273.9 kW/m^3,
core loss 21.54 W, copper loss 17.68 W (litz, Dowell), total 39.22 W, window fill 0.30; E 70/33/32
44.8 W and PM 87/70 44.9 W feasible; ETD 59/31/22 rejected (fill 0.51); UU 93/152/30 not evaluable.
The verification check `engineering.transformer-design-self-consistent` recomputes these numbers by
hand from the datasheet parameters and the stored Steinmetz set (max relative deviation 1.8e-11).
The exported Engineering State `reports/demo/dab-10kw-engineering-state.json` passed the pinned
offline Core validator **and** AIPE-Core's own `scripts/validate.py` (`PASS ... (structure and
references; no physical validation)`).

## 4. Ecosystem compatibility (executed)

| Check | Result |
| --- | --- |
| AIPE-Core round trip (`tests/integrations/test_core_roundtrip.py`, offline + external subprocess) | pass |
| AIPE-Registry validator, explicit manifests `AIPE-Core/aipe.yaml` + `aipe.yaml` (`python scripts/validate.py ...` in the Registry checkout) | exit 0; diagnostics: one warning `unresolved_license` (repository licence is `NOASSERTION` by design) |
| Registry acceptance check `integration.registry-manifest-accepted` (manifest references exist, verification claims ≤ bundled evidence) | pass |

## 5. Host / plugin installation tests (executed)

| Host | What was executed | Result |
| --- | --- | --- |
| Claude Code 2.1.80 | `claude plugin validate plugins/claude/aipe-magnetics`; `claude plugin validate .` (repository-root marketplace) | `Validation passed` (both) |
| Codex CLI 0.163.0 (scratch `CODEX_HOME`) | `codex plugin marketplace add <repo root>` → `Added marketplace aipe-magnetics`; `codex plugin add aipe-magnetics@aipe-magnetics --json` → `installedPath .../plugins/cache/aipe-magnetics/aipe-magnetics/0.2.0`; `codex mcp list` → `aipe-magnetics  python -m aipe_magnetics.mcp.server ... enabled`; five `skills/*/SKILL.md` copied | pass |
| MCP stdio transport | `python -m aipe_magnetics.mcp.server --self-test` (11 tools); `tests/mcp/test_server.py` in-process client and real subprocess over stdio, write gating, unknown-argument rejection | pass |
| Interactive session with a model | *not executed* | - |

## 6. Clean-environment installation (executed on the release host)

```
python -m build                                   -> aipe_magnetics-0.2.0-py3-none-any.whl (183 files) + sdist
uv venv <scratch>/venv --python 3.10              -> fresh environment
pip install "aipe_magnetics-0.2.0-py3-none-any.whl[mcp,hdf5]"
aipe-magnetics info                               -> 11 materials, 6 cores, 4 manufacturers; evidence 58 records / 25 capabilities
aipe-magnetics demo dab-10kw --output <dir>       -> exit 0, total_loss 39.2218 W, report written
python -m aipe_magnetics.mcp.server --self-test   -> 11 tools
offline API call with socket.getaddrinfo / create_connection disabled -> estimate_core_loss(N87, 100 kHz, 100 mT, 100 degC) = 94815.4 W/m^3, L2 verified
```

The wheel contains the catalogue, the derived Steinmetz fits, the v0.2 schemas, the pinned Core /
Registry schemas with NOTICE files, the reference ANN and the evidence file. Linux and macOS legs
are defined in `.github/workflows/ci.yml` (ubuntu/windows/macos × Python 3.10/3.12) and are
*not claimed* until that workflow has run.

## 7. Not executed / not claimed

* No hardware was built or measured; no thermal, finite-element, insulation or manufacturing analysis.
* No model was fitted to new measurements by this release; the Princeton-derived Steinmetz sets are
  least-squares fits of existing MIT-licensed data with recorded metrics.
* PyTorch, `magnethub` (needs Python ≥ 3.13) and the external datasets are optional and were not part
  of the clean-environment test.
* GitHub Actions had not run on this branch at the time of writing.
