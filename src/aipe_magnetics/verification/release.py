"""Release verification runner: run every check, write machine- and human-readable reports, compare with the bundled evidence.

``aipe-magnetics verify --output reports/verification [--check-bundled] [--update-bundled]``

The bundled evidence file (``src/aipe_magnetics/verification/evidence/v<version>.json``) is what the
API consults when it reports a verification status. ``--check-bundled`` fails when the set of
``(check id, status, level, capability)`` of a fresh run differs from the bundled file, so stale or
optimistic evidence cannot ship. ``--update-bundled`` rewrites the file (maintainers, before release).

Some checks consult the bundled file themselves (the AIPE-Registry acceptance check cross-checks the
manifest's verification claims against it), so ``--update-bundled`` writes the bundle, re-runs the suite
against it and repeats until the fingerprint is stable; the persisted bundle omits wall-clock durations
so that two runs with identical outcomes produce identical files.
"""

from __future__ import annotations

import sys
from pathlib import Path

from aipe_magnetics import __version__
from aipe_magnetics.common.jsonio import load
from aipe_magnetics.verification.evidence import VerificationReport
from aipe_magnetics.verification.suite import build_report, bundled_evidence_path, run_suite, write_reports


MAX_UPDATE_PASSES = 3


def fingerprint(report: VerificationReport) -> set[tuple[str, str, str, str]]:
    return {(r.id, r.status, r.level, r.capability) for r in report.records}


def bundle_payload(report: VerificationReport) -> dict:
    """The report as persisted in the bundled evidence file: no wall-clock duration, so identical outcomes give identical bytes."""
    payload = report.model_dump(mode="json", exclude_none=True)
    totals = payload.get("summary", {}).get("_totals")
    if isinstance(totals, dict):
        totals.pop("duration_s", None)
    return payload


def main_cli(*, output: Path, capability: str | None = None, check_bundled: bool = False, update_bundled: bool = False) -> int:
    outcomes = run_suite(capability=capability)
    report = build_report(outcomes)
    json_path, md_path = write_reports(report, output)
    totals = report.summary.get("_totals", {})
    failed = [o for o in outcomes if o.status != "pass"]
    print(f"verification: {totals.get('pass', 0)} passed, {totals.get('fail', 0)} failed, {len(outcomes)} checks in {totals.get('duration_s', 0)} s")
    for o in failed:
        print(f"  FAIL {o.spec.id} ({o.spec.capability})" + (f": {o.error.strip().splitlines()[-1]}" if o.error else ""))
    print(f"reports: {json_path} ; {md_path}")
    status = 0 if not failed else 1
    bundled = bundled_evidence_path()
    if update_bundled:
        if capability:
            print("refusing --update-bundled for a partial (capability-filtered) run", file=sys.stderr)
            return 1
        bundled.parent.mkdir(parents=True, exist_ok=True)
        from aipe_magnetics.common.jsonio import dump
        stable = False
        for _ in range(MAX_UPDATE_PASSES):
            dump(bundle_payload(report), bundled, overwrite=True)
            rerun = build_report(run_suite())
            if fingerprint(rerun) == fingerprint(report):
                stable = True
                break
            print("checks that consult the bundled evidence changed outcome after the update; re-running the suite")
            report = rerun
        if not stable:
            print(f"bundled evidence did not stabilise within {MAX_UPDATE_PASSES} passes", file=sys.stderr)
            return 1
        json_path, md_path = write_reports(report, output)
        totals = report.summary.get("_totals", {})
        failed = [r for r in report.records if r.status != "pass"]
        status = 0 if not failed else 1
        print(f"bundled evidence updated: {bundled} ({totals.get('pass', 0)} passed, {totals.get('fail', 0)} failed after re-run)")
        for r in failed:
            print(f"  FAIL {r.check_id} ({r.capability})")
    if check_bundled:
        if not bundled.exists():
            print(f"bundled evidence missing: {bundled}", file=sys.stderr)
            return 1
        stored = VerificationReport.model_validate(load(bundled))
        fresh = fingerprint(report)
        if capability:
            stored_fp = {x for x in fingerprint(stored) if x[3] == capability}
        else:
            stored_fp = fingerprint(stored)
        if fresh != stored_fp:
            missing = sorted(stored_fp - fresh)
            extra = sorted(fresh - stored_fp)
            print("bundled evidence differs from this run:", file=sys.stderr)
            for item in missing:
                print(f"  bundled-only: {item}", file=sys.stderr)
            for item in extra:
                print(f"  run-only:     {item}", file=sys.stderr)
            return 1
        if stored.tool_version != __version__:
            print(f"bundled evidence tool version {stored.tool_version} != {__version__}", file=sys.stderr)
            return 1
        print("bundled evidence matches this run")
    return status
