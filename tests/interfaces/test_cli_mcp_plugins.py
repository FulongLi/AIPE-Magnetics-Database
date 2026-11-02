"""CLI, MCP server and agent-plugin packaging of the Database."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from aipe_magnetics_database import __version__
from aipe_magnetics_database.cli.main import main

ROOT = Path(__file__).resolve().parents[2]


def test_cli_queries(capsys, tmp_path):
    assert main(["info", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["canonical"]["canonical_materials"] > 1000
    assert main(["measurements", "find", "--material", "TDK N87", "-f", "100 kHz", "-t", "90 degC", "--json"]) == 0
    assert "measurement.princeton-magnet-n87-core-loss" in capsys.readouterr().out
    assert main(["measurements", "table", "measurement.princeton-magnet-n87-core-loss", "-f", "99 kHz..101 kHz", "--csv", str(tmp_path / "t.csv")]) == 0
    assert (tmp_path / "t.csv").read_text().startswith("frequency [Hz],flux_density_peak [T]")
    assert main(["materials", "get", "nonexistent-xyz"]) == 1
    assert "ERROR" in capsys.readouterr().err
    assert main(["validate"]) == 0


def test_mcp_self_test_lists_read_only_tools():
    pytest.importorskip("mcp")
    proc = subprocess.run([sys.executable, "-m", "aipe_magnetics_database.mcp.server", "--self-test"], capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["writes_files"] is False
    assert {"find_measurements", "get_table", "provenance", "get_material"} <= set(payload["tools"])
    assert not {t for t in payload["tools"] if re.search(r"design|estimate|optimi", t)}


@pytest.mark.parametrize("manifest", ["plugins/claude/aipe-magnetics-database/.claude-plugin/plugin.json",
                                      "plugins/codex/aipe-magnetics-database/.codex-plugin/plugin.json"])
def test_plugin_manifests(manifest):
    data = json.loads((ROOT / manifest).read_text(encoding="utf-8"))
    assert data["name"] == "aipe-magnetics-database" and data["version"] == __version__
    mcp = json.loads((ROOT / Path(manifest).parent.parent / ".mcp.json").read_text(encoding="utf-8"))
    assert mcp["mcpServers"]["aipe-magnetics-database"]["args"] == ["-m", "aipe_magnetics_database.mcp.server"]
    for market, source in ((".claude-plugin/marketplace.json", "./plugins/claude/aipe-magnetics-database"),
                           (".agents/plugins/marketplace.json", "./plugins/codex/aipe-magnetics-database")):
        entry = json.loads((ROOT / market).read_text(encoding="utf-8"))["plugins"][0]
        assert entry["source"] == source and entry["version"] == __version__ and (ROOT / source).is_dir()


def test_plugin_skills_in_sync():
    proc = subprocess.run([sys.executable, str(ROOT / "tools" / "sync_plugin_skills.py"), "--check"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_database_has_no_design_or_simulation_dependencies():
    """The Database must run without the Design package and without scipy / plotting / CAD / ML libraries."""
    banned = re.compile(r"^\s*(from|import)\s+(aipe_magnetics_design|scipy|matplotlib|torch|gmsh|cadquery)\b", re.M)
    for path in (ROOT / "src" / "aipe_magnetics_database").rglob("*.py"):
        assert not banned.search(path.read_text(encoding="utf-8")), path
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    deps = pyproject.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert not re.search(r"scipy|matplotlib|torch|aipe-magnetics-design", deps)
