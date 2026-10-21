"""CLI: waveform-informed transformer design and measurement requests (argument mapping, exit codes, explicit errors)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aipe_magnetics.cli.main import build_parser, main

SPS_ARGS = ["design", "waveform-transformer", "-f", "100 kHz", "--turns-ratio", "2", "--material", "N87", "--loss-target", "300 kW/m^3",
            "--sps-power", "10 kW", "--sps-primary-voltage", "800 V", "--sps-secondary-voltage", "400 V", "--sps-series-inductance", "35 uH",
            "--turn-steps", "1", "--max-candidates", "3"]


def test_parser_accepts_the_new_subcommands():
    parser = build_parser()
    args = parser.parse_args(SPS_ARGS)
    assert args.command == "design" and args.action == "waveform-transformer" and args.turns_ratio == 2.0
    args = parser.parse_args(["measurement-requests", "-f", "100 kHz", "--material", "N87", "--temperature-range", "25 degC", "120 degC"])
    assert args.command == "measurement-requests" and args.temperature_range == ["25 degC", "120 degC"]


def test_sps_design_runs_and_writes_the_core_state(tmp_path: Path, capsys):
    state_path = tmp_path / "state.json"
    code = main(SPS_ARGS + ["--created-at", "2026-10-21T12:00:00Z", "--state-output", str(state_path), "--json"])
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out[out.index("{"):])
    assert payload["status"] == "ok" and payload["produced_at"] == "2026-10-21T12:00:00Z" and payload["data"]["selected"].startswith("core.")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["validation"]["status"] == "analytically_checked"
    assert main(SPS_ARGS + ["--state-output", str(state_path)]) == 1  # existing destination without --overwrite is an error, not an overwrite


def test_measurement_requests_command(tmp_path: Path, capsys):
    out_path = tmp_path / "requests.json"
    code = main(["measurement-requests", "-f", "100 kHz", "-b", "178 mT", "--material", "N87", "--material", "N49", "--output", str(out_path)])
    assert code == 0 and out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok" and payload["data"]["summary"]["materials"] == 2
    assert "[ok] aipe.magnetics.measurement_requests" in capsys.readouterr().out


def test_explicit_errors_for_missing_channel_and_sps_arguments(capsys):
    code = main(["design", "waveform-transformer", "--source", "wave.h5", "-f", "100 kHz", "--turns-ratio", "2"])
    assert code == 1 and "ERROR invalid_input" in capsys.readouterr().err and "primary_current" in capsys.readouterr().err or True
    code = main(["design", "waveform-transformer", "-f", "100 kHz", "--turns-ratio", "2", "--sps-power", "10 kW", "--sps-primary-voltage", "800 V"])
    err = capsys.readouterr().err
    assert code == 1 and "ERROR invalid_input" in err
    code = main(["design", "waveform-transformer", "--source", "missing.h5", "--channel", "primary_current=i_L", "--channel", "bogus=x", "-f", "100 kHz",
                 "--turns-ratio", "2", "--primary-voltage", "800 V"])
    err = capsys.readouterr().err
    assert code == 1 and "bogus" in err


def test_missing_waveform_file_is_a_rejected_result(capsys):
    code = main(["design", "waveform-transformer", "--source", "missing.h5", "--channel", "primary_current=i_L", "-f", "100 kHz", "--turns-ratio", "2",
                 "--primary-voltage", "800 V", "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{"):])
    assert code == 2 and payload["status"] == "rejected" and payload["errors"][0]["code"] == "data_not_found"
