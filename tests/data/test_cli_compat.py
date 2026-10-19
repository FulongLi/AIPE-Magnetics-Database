"""The ``aipe-magnetics`` CLI consumes the data layer: these integration tests keep the two from drifting apart.

They drive the real CLI entry point (``aipe_magnetics.cli.main.main``) for the schema and import
subcommands, exercise the compatibility shims the data layer keeps for command-line consumers, and
make sure every ``python -m aipe_magnetics.data...`` entry point runs exactly once without a
``RuntimeWarning`` (the package exports are lazy).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from aipe_magnetics.cli.main import main
from aipe_magnetics.common.jsonio import dump, load
from aipe_magnetics.data.fixtures import v01_fixture_paths
from aipe_magnetics.data.importers.json_record import import_record
from aipe_magnetics.data.schema import check_exported, check_exported_schemas, export_schemas, validate_record

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cli_schema_export_and_check(tmp_path: Path, capsys):
    out = tmp_path / "schemas"
    assert main(["schema", "export", str(out)]) == 0
    assert (out / "magnetic-record.schema.json").is_file() and check_exported_schemas(out) == []
    assert main(["schema", "export", str(out), "--check"]) == 0
    (out / "quantity.schema.json").write_bytes(b"{}\n")
    assert main(["schema", "export", str(out), "--check"]) == 1
    assert "quantity.schema.json" in capsys.readouterr().out


def test_cli_import_json_with_migration_and_catalogue_files(tmp_path: Path, v01_example_path: Path, capsys):
    destination = tmp_path / "imported.json"
    assert main(["import", "json", str(v01_example_path), str(destination), "--migrate"]) == 0
    printed = capsys.readouterr().out
    assert "imported" in printed and "warning: record migrated" in printed
    assert validate_record(load(destination)) == []
    # an existing destination is a one-line user error (exit 1), never a traceback and never an overwrite
    before = destination.read_bytes()
    assert main(["import", "json", str(v01_example_path), str(destination), "--migrate"]) == 1
    assert destination.read_bytes() == before
    # --catalogue FILE supplies the reference context
    core = tmp_path / "core.json"
    dump(load(destination) | {"id": "core.cli-context", "kind": "core_geometry", "properties": {}, "extensions": {}}, core)
    transformer = load(destination)
    transformer["id"], transformer["properties"], transformer["extensions"] = "transformer.cli-context", {"core_ref": "core.cli-context"}, {}
    source = tmp_path / "transformer.json"
    dump(transformer, source)
    assert main(["import", "json", str(source), str(tmp_path / "t.json"), "--catalogue", str(core)]) == 0
    assert validate_record(load(tmp_path / "t.json")) == []


def test_cli_schema_validate_and_migrate(tmp_path: Path, v01_example_path: Path, capsys):
    assert main(["schema", "validate", str(v01_example_path)]) == 0
    assert capsys.readouterr().out.startswith("PASS ")
    out = tmp_path / "migrated.json"
    assert main(["schema", "migrate", str(v01_example_path), str(out)]) == 0
    assert validate_record(load(out)) == []
    assert main(["schema", "validate", str(out)]) == 0
    broken = tmp_path / "broken.json"
    record = load(out)
    record["properties"]["leakage_inductance"] = {"value": -1e-6, "unit": "H"}
    dump(record, broken)
    assert main(["schema", "validate", str(broken)]) == 1
    assert "[physical]" in capsys.readouterr().out


def test_compat_shims_kept_for_command_line_consumers(tmp_path: Path, v01_example_path: Path, capsys):
    out = tmp_path / "s"
    export_schemas(out)
    assert check_exported(out) == 0 and check_exported(tmp_path / "nope") == 1
    result = import_record(v01_example_path, tmp_path / "r.json", catalogue=[], migrate=True)
    assert result.written == [str(tmp_path / "r.json")] and result.record.id == "magnetic.example-transformer"


@pytest.mark.parametrize("argv", [
    ["aipe_magnetics.data.schema", "--check", str(REPO_ROOT / "schemas" / "v0.2")],
    ["aipe_magnetics.data.schema", "--validate", *[str(p) for p in v01_fixture_paths()]],
    ["aipe_magnetics.data.migrate", "--help"],
    ["aipe_magnetics.data.importers.json_record", "--help"],
])
def test_module_entry_points_run_once_without_runtime_warning(argv):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep))
    proc = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-m", *argv], capture_output=True, text=True,
                          cwd=str(REPO_ROOT), env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "RuntimeWarning" not in proc.stderr
