# Verification framework

There is **no global "validated" flag**. Evidence is recorded per capability (model or engineering
function) and per operating range, at three levels:

| Level | Meaning | Typical checks |
| --- | --- | --- |
| **L1** software and schema correctness | the code does what it says | invalid-input rejection, determinism, schema validity, round trips, overflow detection |
| **L2** numerical and physical consistency | the implementation agrees with mathematics and physics | analytical reference cases, independent numerical implementations, published reference values, limiting cases, zero excitation, scaling laws, time-step and sampling convergence, energy consistency, loop closure/dissipation |
| **L3** engineering application validation | agreement with experiments or reviewed engineering cases | comparison with measured data where licensing permits (optional, external) |

## How it works

* Checks are plain functions registered with `@check(...)` in
  `aipe_magnetics/verification/checks/check_*.py`. Each declares its capability id, level, kind,
  method, optional operating range and reference, and returns a `CheckResult(passed, metrics, notes)`.
* `aipe-magnetics verify --output reports/verification` runs every check and writes
  `verification-report.json` (machine-readable `EvidenceRecord`s) and `verification-report.md`.
* The **bundled evidence file** `src/aipe_magnetics/verification/evidence/v0.2.0.json` is produced by
  `aipe-magnetics verify --update-bundled` before a release and compared against a fresh run by
  `--check-bundled` (CI). Every API result derives its `verification` block from this file for the
  requested conditions: `verified` (passing L2/L3 evidence covers the conditions),
  `partially_verified` (only L1), `out_of_range` (conditions outside all verified ranges),
  `unverified` (no evidence), `failed` (failing evidence exists). A condition that is *not supplied* for a
  declared range dimension makes the range undecidable (`in_verified_range: null`, the statement names
  the unchecked dimension); only a supplied condition outside a declared range yields `out_of_range`.
* The bundled file is produced by the maintainer's release step (`--update-bundled` on a green suite).
  Until it is committed the API does not pretend otherwise: every model-based result reports
  `L0 unverified` with a statement naming the missing file,
  `aipe-magnetics info` prints `verification evidence: MISSING`, `verification_status()` warns
  `bundled_evidence_missing`, `verify --check-bundled` exits 1, and the Registry acceptance check
  `integration.registry-manifest-accepted` fails closed.
* `pytest` executes the same check functions, so a failing check fails the test suite.

## Model declarations

Every model declares (see `aipe_magnetics.models.contract.ValidityDomain`): supported materials,
excitation kinds, frequency / temperature / flux-density ranges, DC-bias handling
(`unsupported`, `ignored`, `supported`), whether magnetic history is required, numerical constraints
and known limitations. Predictions outside the declared domain are rejected (`policy="reject"`,
default) or flagged (`policy="warn"`) - never silently clipped.

## Uncertainty

Confidence intervals are never fabricated. Quantities carry an `uncertainty` field only when a
source or a fit provides one (`fit_metrics` are stored with every fitted parameter set); otherwise
the uncertainty is reported as unknown.

## Reports

`reports/verification/verification-report.md` (human) and `.json` (machine) are produced by
`aipe-magnetics verify` (CI runs it on every push). The released evidence file is committed with the
package by the maintainer's release step: `verify --update-bundled` writes the bundle, re-runs the suite
against it (some checks consult the bundle) until the outcome is stable, and omits wall-clock durations
so that identical outcomes give identical files; the release procedure then also places the Markdown
report and the record of the ecosystem run under `docs/`.
