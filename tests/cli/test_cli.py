"""CLI behaviour that does not depend on catalogue content."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aipe_magnetics import __version__
from aipe_magnetics.cli.main import _qty, build_parser, main


def test_quantity_parsing():
    assert _qty("100 kHz", "f") == {"value": 100.0, "unit": "kHz"}
    assert _qty("0.2T", "b") == {"value": 0.2, "unit": "T"}
    assert _qty("1e5 Hz", "f") == {"value": 1e5, "unit": "Hz"}
    assert _qty("-3 degC", "t") == {"value": -3.0, "unit": "degC"}
    with pytest.raises(SystemExit):
        _qty("100", "f")


def test_version_and_help():
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])
    assert exc.value.code == 0
    out = subprocess.run([sys.executable, "-m", "aipe_magnetics.cli.main", "--help"], capture_output=True, text=True)
    assert out.returncode == 0 and "materials" in out.stdout and "design" in out.stdout


def test_hysteresis_command_json(capsys):
    code = main(["hysteresis", "evaluate", "--field-peak", "300 A/m", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok" and "loop_energy_density" in payload["values"]


def test_verify_command_writes_reports(tmp_path: Path, capsys):
    code = main(["verify", "--output", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0, out
    assert (tmp_path / "verification-report.json").exists() and (tmp_path / "verification-report.md").exists()
    report = json.loads((tmp_path / "verification-report.json").read_text(encoding="utf-8"))
    assert report["tool_version"] == __version__ and report["records"]
    assert all(r["status"] == "pass" for r in report["records"])


def test_mcp_self_test_command(capsys):
    assert main(["mcp", "self-test"]) == 0
    assert "estimate_core_loss" in capsys.readouterr().out


def test_rejected_results_exit_with_code_2(capsys):
    code = main(["loss", "estimate", "definitely-not-a-material", "-f", "100 kHz", "-b", "100 mT", "-t", "100 degC"])
    assert code == 2
    assert "rejected" in capsys.readouterr().out
