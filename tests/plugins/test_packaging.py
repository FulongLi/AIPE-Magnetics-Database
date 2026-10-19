"""Structural validation of the Codex and Claude Code plugin packages (no host CLI required)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CLAUDE_PLUGIN = ROOT / "plugins" / "claude" / "aipe-magnetics"
CODEX_PLUGIN = ROOT / "plugins" / "codex" / "aipe-magnetics"
SKILLS = ["magnetic-material-analysis", "inductor-design", "transformer-design", "magnetic-loss-modelling", "magnetic-engineering-review"]
NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert match, f"{path} has no YAML frontmatter"
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


@pytest.mark.parametrize("plugin, manifest", [(CLAUDE_PLUGIN, ".claude-plugin/plugin.json"), (CODEX_PLUGIN, ".codex-plugin/plugin.json")])
def test_plugin_manifest_fields(plugin: Path, manifest: str):
    data = _json(plugin / manifest)
    assert data["name"] == "aipe-magnetics" and NAME_RE.match(data["name"])
    assert not data["name"].startswith(("claude-", "anthropic-", "cc-plugin-"))
    assert data["version"] == "0.2.0"
    assert data["description"] and data["author"]["name"]
    assert data["homepage"].startswith("https://") and data["repository"].startswith("https://")
    assert data["license"] == "NOASSERTION"
    for key in ("skills", "mcpServers"):
        assert data[key].startswith("./"), f"{key} path must be plugin-root relative and start with ./"
        assert (plugin / data[key]).exists()
    # components live at the plugin root, never inside the manifest directory
    assert not (plugin / Path(manifest).parent / "skills").exists()


def test_codex_interface_metadata():
    data = _json(CODEX_PLUGIN / ".codex-plugin" / "plugin.json")
    interface = data["interface"]
    assert interface["displayName"] and interface["category"] and interface["shortDescription"]
    assert len(interface.get("defaultPrompt", [])) <= 3 and all(len(p) <= 128 for p in interface.get("defaultPrompt", []))


@pytest.mark.parametrize("plugin", [CLAUDE_PLUGIN, CODEX_PLUGIN])
def test_mcp_declaration_points_at_the_engine(plugin: Path):
    data = _json(plugin / ".mcp.json")
    server = data["mcpServers"]["aipe-magnetics"]
    assert server["command"] == "python" and server["args"] == ["-m", "aipe_magnetics.mcp.server"]
    assert "AIPE_MAGNETICS_MCP_ALLOW_WRITE" in server.get("env", {}) and server["env"]["AIPE_MAGNETICS_MCP_ALLOW_WRITE"] == "0"
    for key, value in server.items():
        assert "api_key" not in key.lower() and "token" not in key.lower()
        assert not (isinstance(value, str) and re.search(r"sk-[A-Za-z0-9]{8,}", value))


@pytest.mark.parametrize("plugin", [CLAUDE_PLUGIN, CODEX_PLUGIN])
def test_skills_follow_the_agent_skills_rules(plugin: Path):
    found = sorted(p.parent.name for p in (plugin / "skills").glob("*/SKILL.md"))
    assert found == sorted(SKILLS)
    for name in SKILLS:
        fm = _frontmatter(plugin / "skills" / name / "SKILL.md")
        assert fm["name"] == name, "frontmatter name must match the directory"
        assert NAME_RE.match(name) and 1 <= len(name) <= 64
        assert 1 <= len(fm["description"]) <= 1024
        body = (plugin / "skills" / name / "SKILL.md").read_text(encoding="utf-8")
        assert body.count("\n") < 500


def test_plugin_skills_are_synced_with_canonical_sources():
    result = subprocess.run([sys.executable, str(ROOT / "tools" / "sync_plugin_skills.py"), "--check"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("manifest, expected_source", [
    (ROOT / ".claude-plugin" / "marketplace.json", "./plugins/claude/aipe-magnetics"),
    (ROOT / ".agents" / "plugins" / "marketplace.json", "./plugins/codex/aipe-magnetics"),
])
def test_marketplaces_point_at_existing_plugins(manifest: Path, expected_source: str):
    data = _json(manifest)
    assert data["name"] == "aipe-magnetics"
    entries = data["plugins"]
    assert len(entries) == 1 and entries[0]["name"] == "aipe-magnetics" and entries[0]["source"] == expected_source
    target = ROOT / expected_source
    assert (target / ".claude-plugin" / "plugin.json").exists() or (target / ".codex-plugin" / "plugin.json").exists()
    assert entries[0]["version"] == "0.2.0"
    if "policy" in entries[0]:
        assert entries[0]["policy"]["installation"] in {"AVAILABLE", "INSTALLED_BY_DEFAULT", "NOT_AVAILABLE"}
        assert entries[0]["policy"]["authentication"] in {"ON_INSTALL", "ON_USE"}


def test_no_numerical_code_inside_plugins():
    for plugin in (CLAUDE_PLUGIN, CODEX_PLUGIN):
        assert not list(plugin.rglob("*.py")), "plugins must not duplicate numerical implementations"
