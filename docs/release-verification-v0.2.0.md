# Release verification record - AIPE Magnetics 0.2.0

Branch `feature/aipe-magnetics-v0.2`. Release host: Windows 11 Pro for Workstations (10.0.26300),
Python 3.10.11 (project virtualenv created with uv 0.12.18), numpy 2.2.6, scipy 1.15.3,
pydantic 2.14.1, jsonschema 4.26.0, h5py 3.16.0, mcp 2.3.0 (protocol 2026-07-28),
Claude Code 2.1.80, codex-cli 0.163.0 (installed locally in a scratch directory for packaging checks).

This document records what was actually executed for the release. Items marked *not executed*
were not run and are not claimed.

## 1. Automated test suite

_(final numbers inserted by the release run; see section 7 for the exact command lines)_

| Suite | Result |
| --- | --- |
| `pytest -q` (whole repository, offline) | see section 7 |
| legacy v0.1 `python -m unittest discover -s tests -p "test_records.py"` | see section 7 |
| legacy `python scripts/validate.py examples/synthetic-transformer.json` | see section 7 |
| optional external tests with local reference checkouts (`AIPE_ECOSYSTEM_CHECKOUTS`, `AIPE_MAGNET_CHECKOUT`, `AIPE_MAGNETHUB_CHECKOUT`, `AIPE_MATERIALDATABASE_CHECKOUT`, `AIPE_MAGNETCHALLENGE2_CHECKOUT`, `AIPE_ALLOW_DOWNLOAD=1`) | see section 7 |

## 2. Verification suite (evidence per capability)

`aipe-magnetics verify --output reports/verification --update-bundled` produced
`src/aipe_magnetics/verification/evidence/v0.2.0.json`; the human-readable report of the same run
is committed as `docs/verification-report-v0.2.0.md`. `aipe-magnetics verify --check-bundled`
re-runs every check and fails when the recorded `(check id, status, level, capability)` set differs.

## 3. Demonstration (10 kW / 100 kHz DAB transformer)

`aipe-magnetics demo dab-10kw --output reports/demo` executed through the public API only
(see `docs/demo-dab-10kw.md`). Outputs: `dab-10kw-report.md`, `dab-10kw-results.json`,
`dab-10kw-engineering-state.json`. The exported Engineering State was validated with the pinned
offline Core validator **and** with AIPE-Core's own `scripts/validate.py` from a checkout of
https://github.com/FulongLi/AIPE-Core (result: `PASS ... (structure and references; no physical
validation)`).

Headline numbers of the release run are listed in section 7; they are analytical design estimates
with the assumptions printed in the report (no hardware was built or measured).

## 4. Ecosystem compatibility

| Check | How | Result |
| --- | --- | --- |
| AIPE-Core round trip | `tests/integrations/test_core_roundtrip.py` (offline, pinned schemas) and the external subprocess test running Core's `scripts/validate.py` | pass (see section 7) |
| AIPE-Registry manifest | `tests/integrations/test_registry_manifest.py` (offline schema + semantic rules) and Registry's `scripts/validate.py` executed with explicit manifests `AIPE-Core/aipe.yaml` + `aipe.yaml` | pass once the release files exist (the validator checks that every referenced documentation/evidence path is a file) |

## 5. Host / plugin installation tests

| Host | What was executed | Result |
| --- | --- | --- |
| Claude Code 2.1.80 | `claude plugin validate plugins/claude/aipe-magnetics` and `claude plugin validate .` (marketplace) | Validation passed |
| Codex CLI 0.163.0 | with a scratch `CODEX_HOME`: `codex plugin marketplace add <repo root>`, `codex plugin add aipe-magnetics@aipe-magnetics --json`, `codex mcp list` | plugin installed to `plugins/cache/aipe-magnetics/aipe-magnetics/0.2.0`, the `aipe-magnetics` MCP server is listed, all five skills copied |
| MCP stdio transport | `python -m aipe_magnetics.mcp.server --self-test`; `tests/mcp/test_server.py` (in-process client and real subprocess over stdio) | pass |
| Interactive session with a model | *not executed* (no LLM session was driven as part of the release) | - |

## 6. Clean-environment installation

Executed on the release host: build sdist/wheel with `python -m build`, create a fresh virtual
environment, `pip install <wheel>[mcp,hdf5]`, then `aipe-magnetics info`, `aipe-magnetics demo dab-10kw`,
`python -m aipe_magnetics.mcp.server --self-test` and an offline API call with sockets disabled.
Result and command lines in section 7. Linux and macOS runs are defined in `.github/workflows/ci.yml`
(matrix ubuntu/windows/macos × Python 3.10/3.12) and are *not claimed* until the workflow has run.

## 7. Release run log

_(appended by the release run)_
