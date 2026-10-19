"""Release-level consistency: versions, manifest paths, licence declarations, legacy artefacts untouched."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib_shim  # noqa: F401  (see conftest: provides tomllib on Python 3.10)
from pathlib import Path

import yaml

from aipe_magnetics import __version__

ROOT = Path(__file__).resolve().parents[1]


def _toml(path: Path) -> dict:
    import tomllib_shim as shim
    return shim.loads(path.read_text(encoding="utf-8"))


def test_versions_agree_everywhere():
    assert _toml(ROOT / "pyproject.toml")["project"]["version"] == __version__
    for manifest in (ROOT / "plugins/claude/aipe-magnetics/.claude-plugin/plugin.json",
                     ROOT / "plugins/codex/aipe-magnetics/.codex-plugin/plugin.json"):
        assert json.loads(manifest.read_text(encoding="utf-8"))["version"] == __version__
    for marketplace in (ROOT / ".claude-plugin/marketplace.json", ROOT / ".agents/plugins/marketplace.json"):
        data = json.loads(marketplace.read_text(encoding="utf-8"))
        assert data["plugins"][0]["version"] == __version__
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## {__version__}" in changelog
    manifest = yaml.safe_load((ROOT / "aipe.yaml").read_text(encoding="utf-8"))
    assert manifest["extensions"]["aipe.magnetics"]["package"] == f"aipe-magnetics {__version__}"


def test_manifest_documentation_and_evidence_paths_exist():
    manifest = yaml.safe_load((ROOT / "aipe.yaml").read_text(encoding="utf-8"))
    for ref in manifest["documentation"] + manifest["validation"]["evidence"]:
        if ref.startswith("https://"):
            continue
        assert (ROOT / ref).exists(), f"aipe.yaml references missing path {ref}"
    assert manifest["license"] == "NOASSERTION"
    assert manifest["compatibility"]["integration"] == "native"
    standalone = manifest["extensions"]["aipe.magnetics"]["standalone_operation"]
    assert not any(standalone.values()), "standalone operation must not require orchestrator, registry, experts, commercial software or LLM keys"


def test_legacy_v01_artefacts_are_unchanged_since_the_v01_release():
    """The v0.1 files must stay byte-identical (modulo line endings) to commit f73b0cd (merge of PR #1)."""
    import subprocess

    legacy = ["schemas/magnetic-record.schema.json", "scripts/validate.py", "importers/json_record.py",
              "examples/synthetic-transformer.json", "tests/test_records.py"]
    for rel in legacy:
        assert (ROOT / rel).exists(), rel
        blob = subprocess.run(["git", "show", f"f73b0cd:{rel}"], capture_output=True, cwd=ROOT)
        if blob.returncode != 0:
            continue  # not a git checkout with history (e.g. sdist); presence was checked above
        assert blob.stdout.replace(b"\r\n", b"\n") == (ROOT / rel).read_bytes().replace(b"\r\n", b"\n"), f"{rel} changed since v0.1"


def test_no_api_keys_or_cloud_endpoints_in_package():
    pattern = re.compile(r"(sk-[A-Za-z0-9]{16,}|api[_-]?key\s*=\s*['\"][A-Za-z0-9]{8,}|https://api\.(openai|anthropic)\.com)")
    for path in (ROOT / "src").rglob("*.py"):
        assert not pattern.search(path.read_text(encoding="utf-8")), path
