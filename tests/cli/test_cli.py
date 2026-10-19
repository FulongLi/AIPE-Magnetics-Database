"""CLI behaviour that does not depend on catalogue content, plus explicit (non-traceback) error handling."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aipe_magnetics import __version__
from aipe_magnetics.cli.main import _qty, build_parser, main
from aipe_magnetics.verification.suite import bundled_evidence_path


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
    assert (tmp_path / "verification-report.json").exists() and (tmp_path / "verification-report.md").exists()
    report = json.loads((tmp_path / "verification-report.json").read_text(encoding="utf-8"))
    assert report["tool_version"] == __version__ and report["records"]
    failing = sorted(r["id"] for r in report["records"] if r["status"] != "pass")
    if bundled_evidence_path().exists():
        assert code == 0 and failing == [], out
    else:
        # The AIPE-Registry acceptance check fails closed, by design, while the maintainer's bundled evidence file
        # (src/aipe_magnetics/verification/evidence/v<version>.json) does not exist; nothing else may fail and the
        # command must say so with a non-zero exit code instead of hiding it.
        assert failing == ["check:integration.registry-manifest-accepted"], out
        assert code == 1 and "integration.registry-manifest-accepted" in out


def test_mcp_self_test_command(capsys):
    assert main(["mcp", "self-test"]) == 0
    assert "estimate_core_loss" in capsys.readouterr().out


def test_rejected_results_exit_with_code_2(capsys):
    code = main(["loss", "estimate", "definitely-not-a-material", "-f", "100 kHz", "-b", "100 mT", "-t", "100 degC"])
    assert code == 2
    assert "rejected" in capsys.readouterr().out


def test_explicit_flux_fraction_zero_is_rejected_not_dropped(capsys):
    code = main(["design", "inductor", "-L", "100 uH", "--current-peak", "20 A", "--current-rms", "14 A", "-f", "100 kHz",
                 "--material", "N87", "--flux-fraction", "0", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2 and payload["status"] == "rejected"
    assert "fraction" in payload["errors"][0]["message"]
    code = main(["design", "transformer", "-P", "10 kW", "-f", "100 kHz", "--primary-voltage", "800 V", "--secondary-voltage", "400 V",
                 "--material", "N87", "--flux-fraction", "0", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2 and payload["status"] == "rejected" and "fraction" in payload["errors"][0]["message"]


def test_user_errors_are_one_line_messages_not_tracebacks(tmp_path: Path, capsys):
    assert main(["hysteresis", "evaluate", "--field-peak", "500 A/m", "--param", "Ms"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("ERROR invalid_input") and "NAME=VALUE" in err
    assert main(["hysteresis", "evaluate", "--field-peak", "500 A/m", "--param", "Ms=abc"]) == 1
    assert "not a number" in capsys.readouterr().err
    assert main(["report", str(tmp_path / "missing.json")]) == 1
    assert capsys.readouterr().err.startswith("ERROR")
    (tmp_path / "notenv.json").write_text('{"hello": 1}', encoding="utf-8")
    assert main(["report", str(tmp_path / "notenv.json")]) == 1
    assert "envelopes" in capsys.readouterr().err
    assert main(["schema", "validate", str(tmp_path / "missing.json")]) == 1
    assert capsys.readouterr().err.startswith("ERROR")
    assert main(["loss", "estimate", "N87", "-f", "100 kHz", "-b", "150 mT", "-t", "100 degC", "--waveform", "trapezoidal", "--duty-flat", "0.2"]) == 1
    assert "--duty-rise" in capsys.readouterr().err


def test_report_accepts_envelopes_json_reports_and_demo_output(tmp_path: Path, capsys):
    assert main(["loss", "estimate", "N87", "-f", "100 kHz", "-b", "100 mT", "-t", "100 degC", "--json"]) == 0
    envelope = json.loads(capsys.readouterr().out)
    (tmp_path / "single.json").write_text(json.dumps(envelope), encoding="utf-8")
    (tmp_path / "demo.json").write_text(json.dumps({"steps": [{"title": "step", "result": envelope}]}), encoding="utf-8")
    (tmp_path / "report.json").write_text(json.dumps({"report_schema": "aipe.magnetics.report/0.2", "results": [envelope]}), encoding="utf-8")
    assert main(["report", str(tmp_path / "single.json"), str(tmp_path / "demo.json"), str(tmp_path / "report.json")]) == 0
    out = capsys.readouterr().out
    assert "## 3. aipe.magnetics.estimate_core_loss" in out
    assert main(["report", str(tmp_path / "single.json"), "--output", str(tmp_path / "out.md")]) == 0
    assert (tmp_path / "out.md").exists()
    assert main(["report", str(tmp_path / "single.json"), "--output", str(tmp_path / "out.md")]) == 2
    assert "destination_exists" in capsys.readouterr().out


def test_duty_rise_alone_derives_the_symmetric_trapezoid(capsys):
    code = main(["loss", "estimate", "N87", "-f", "100 kHz", "-b", "150 mT", "-t", "100 degC", "--waveform", "trapezoidal",
                 "--duty-rise", "0.25", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0, payload.get("errors")
    assert payload["inputs"]["operating_point"]["waveform"]["duty_flat"] == pytest.approx(0.25)


def test_mcp_serve_allow_write_requires_output_dir():
    with pytest.raises(SystemExit) as exc:
        main(["mcp", "serve", "--allow-write"])
    assert exc.value.code == 2


def test_info_reports_the_bundled_evidence_state_loudly(capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    if bundled_evidence_path().exists():
        assert "records" in out
    else:
        assert "MISSING" in out and "--update-bundled" in out
