"""Hugging Face layer: staging, publishing (against an in-memory Hub), sync, version-pinned reads, transfer restore."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from aipe_magnetics_database.common.errors import DataIntegrityError, DataNotFoundError
from aipe_magnetics_database.hub import layout
from aipe_magnetics_database.hub.publish import make_plan, publish, remote_files
from aipe_magnetics_database.hub.stage import build_stage, load_checksums, publishable, verify_stage
from aipe_magnetics_database.ingest import registry
from aipe_magnetics_database.ingest.common import git_blob_sha1, sha256_path

ROOT = Path(__file__).resolve().parents[2]
SMALL = ["zenodo_21626116_smc_ferrite_triangular", "zenodo_8292861_hefmag"]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(scope="module")
def stage_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("stage") / "release"
    build_stage(out, root=ROOT, sources=SMALL, log=lambda _m: None)
    return out


# -- layout and policy -------------------------------------------------------------------------------------------
def test_layout_paths():
    assert layout.raw_path("x", "files/a.csv") == "raw/x/files/a.csv"
    assert layout.normalized_path("x", "tables/t.jsonl") == "normalized/v0.2/x/tables/t.jsonl"
    with pytest.raises(ValueError):
        layout.raw_path("x", "../../escape")
    assert layout.resolve_url("o/r", "abc", "raw/x/a b.csv").endswith("/datasets/o/r/resolve/abc/raw/x/a%20b.csv")
    assert layout.OIDC_RESOURCE == "datasets/FulongLi/AIPE-Magnetics-Dataset"


def test_only_redistributable_integrated_sources_are_publishable():
    decisions = {s["id"]: publishable(s) for s in registry.SOURCES}
    assert all(ok for sid, (ok, _) in decisions.items() if sid in {s["id"] for s in registry.integrated()} and sid != "aipe_curated")
    for sid in ("psma_core_loss_studies", "magnet_challenge_2", "asu_mhz_core_loss_aggregate", "manufacturer_tdk", "aipe_curated"):
        ok, reason = decisions[sid]
        assert not ok and reason
    assert not publishable({"id": "x", "status": "integrated", "directory": "x", "license": "NOASSERTION"})[0]


# -- staging -----------------------------------------------------------------------------------------------------
def test_stage_preserves_raw_and_normalised_bytes(stage_dir: Path):
    assert verify_stage(stage_dir) == []
    for sid in SMALL:
        source = registry.by_id(sid)
        manifest = json.loads((ROOT / "data/raw" / source["directory"] / "SOURCES.json").read_text(encoding="utf-8"))
        for item in manifest["files"]:
            staged = stage_dir / layout.raw_path(sid, f"files/{item['path']}")
            assert sha256_path(staged) == item["sha256"]
            assert staged.read_bytes() == (ROOT / "data/raw" / source["directory"] / "files" / item["path"]).read_bytes()
        base = ROOT / "data/normalized" / source["directory"]
        for path in base.rglob("*"):
            if path.is_file():
                assert (stage_dir / layout.normalized_path(sid, path.relative_to(base).as_posix())).read_bytes() == path.read_bytes()


def test_stage_manifests(stage_dir: Path):
    index = json.loads((stage_dir / "manifests/dataset-index.json").read_text(encoding="utf-8"))
    assert [s["id"] for s in index["sources"]] == [s["id"] for s in registry.SOURCES if s["id"] in SMALL]
    assert all(s["citation"] and s["license"] == "CC-BY-4.0" and s["retrieved_at"] for s in index["sources"])
    sources = json.loads((stage_dir / "manifests/sources.json").read_text(encoding="utf-8"))
    assert len(sources) == len(registry.SOURCES)
    by_id = {s["id"]: s for s in sources}
    assert by_id["psma_core_loss_studies"]["huggingface"]["published"] is False
    raw = by_id["zenodo_8292861_hefmag"]["huggingface"]["raw_files"][0]
    assert raw["format"] == "ZIP" and len(raw["sha256"]) == 64 and raw["url"].startswith("https://zenodo.org/")
    checksums = load_checksums(stage_dir)
    on_disk = {p.relative_to(stage_dir).as_posix() for p in stage_dir.rglob("*") if p.is_file()}
    assert set(checksums) == on_disk
    card = (stage_dir / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---\nlicense: other") and "Schema validity is not proof" in card
    quality = json.loads((stage_dir / "manifests/quality-report.json").read_text(encoding="utf-8"))
    assert quality["summary"]["errors"] == 0


def test_stage_refuses_non_empty_dir_and_detects_tampering(stage_dir: Path, tmp_path: Path):
    (tmp_path / "x").write_text("x")
    with pytest.raises(FileExistsError):
        build_stage(tmp_path, root=ROOT, sources=SMALL, log=lambda _m: None)
    copy = tmp_path / "copy"
    shutil.copytree(stage_dir, copy)
    victim = next(p for p in (copy / "raw").rglob("*.txt"))
    victim.write_bytes(victim.read_bytes() + b"\n")
    (copy / "stray.bin").write_bytes(b"0")
    problems = verify_stage(copy)
    assert any("checksum mismatch" in p for p in problems) and "unlisted file: stray.bin" in problems


# -- publishing against an in-memory Hub ---------------------------------------------------------------------------
class FakeHub:
    """Minimal stand-in for huggingface_hub.HfApi: branches and PR refs of path -> bytes."""

    def __init__(self):
        self.refs: dict[str, dict[str, bytes]] = {"main": {".gitattributes": b"*.zip filter=lfs\n"}}
        self.prs: list[SimpleNamespace] = []
        self.uploads: list[tuple[str, int]] = []
        self.deleted: list[str] = []

    # read
    def auth_check(self, repo_id, repo_type=None):
        return True

    def list_repo_tree(self, repo_id, repo_type=None, revision=None, recursive=True):
        from huggingface_hub.hf_api import RepoFile
        files = self.refs[revision or "main"]
        out = []
        for path, data in sorted(files.items()):
            lfs = {"oid": sha(data), "size": len(data), "pointerSize": 130} if len(data) > 1000 else None
            out.append(RepoFile(path=path, size=len(data), oid=hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest(), lfs=lfs))
        return out

    def repo_info(self, repo_id, repo_type=None, revision=None):
        files = self.refs[revision]
        return SimpleNamespace(sha=hashlib.sha1(json.dumps(sorted((k, sha(v)) for k, v in files.items())).encode()).hexdigest())

    def get_repo_discussions(self, repo_id, repo_type=None, discussion_type=None):
        return iter(self.prs)

    # write
    def create_pull_request(self, repo_id, title, description=None, repo_type=None):
        raise AssertionError("the discussions endpoint rejects Trusted Publisher tokens; PRs must come from create_pr commits")
        num = len(self.prs) + 1
        pr = SimpleNamespace(num=num, title=title, status="draft", git_reference=f"refs/pr/{num}")
        self.prs.append(pr)
        self.refs[pr.git_reference] = dict(self.refs["main"])
        return pr

    def upload_folder(self, repo_id, repo_type, folder_path, revision=None, commit_message=None, commit_description=None, create_pr=False):
        if create_pr:  # the commit endpoint opens the pull request (title = commit message)
            num = len(self.prs) + 1
            pr = SimpleNamespace(num=num, title=commit_message, status="open", git_reference=f"refs/pr/{num}")
            self.prs.append(pr)
            self.refs[pr.git_reference] = dict(self.refs["main"])
            revision = pr.git_reference
        target = self.refs[revision or "main"]
        n = 0
        for p in Path(folder_path).rglob("*"):
            if p.is_file():
                target[p.relative_to(folder_path).as_posix()] = p.read_bytes()
                n += 1
        self.uploads.append((revision or "main", n))
        if create_pr:
            return SimpleNamespace(pr_num=pr.num, pr_revision=pr.git_reference, pr_url=f"https://hub/pr/{pr.num}")

    def change_discussion_status(self, repo_id, num, new_status, repo_type=None, comment=None):
        self.prs[num - 1].status = new_status

    def merge_pull_request(self, repo_id, num, repo_type=None, comment=None):
        pr = self.prs[num - 1]
        assert pr.status == "open", "draft pull requests cannot be merged"
        self.refs["main"] = dict(self.refs[pr.git_reference])
        pr.status = "merged"


@pytest.fixture()
def hub(monkeypatch):
    import huggingface_hub
    fake = FakeHub()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: fake)
    monkeypatch.delenv("HF_OIDC_RESOURCE", raising=False)
    return fake


def test_publish_pr_verify_merge_then_idempotent(stage_dir: Path, hub: FakeHub, tmp_path: Path):
    receipt = publish(stage_dir, mode="pr", merge=True, receipt_path=tmp_path / "r.json", log=lambda _m: None)
    assert receipt["verified"] and receipt["merged"] and receipt["pull_request"]["num"] == 1
    assert hub.refs["main"].keys() >= load_checksums(stage_dir).keys() and ".gitattributes" in hub.refs["main"]
    assert json.loads((tmp_path / "r.json").read_text())["commit"] == receipt["commit"]
    again = publish(stage_dir, mode="pr", log=lambda _m: None)
    assert again["plan"]["upload"] == 0 and len(hub.uploads) == 1 and len(hub.prs) == 1  # nothing re-uploaded, no new PR


def test_publish_dry_run_uploads_nothing(stage_dir: Path, hub: FakeHub):
    receipt = publish(stage_dir, dry_run=True, log=lambda _m: None)
    assert receipt["plan"]["upload"] == len(load_checksums(stage_dir)) and not hub.uploads and not hub.prs


def test_publish_refuses_to_replace_raw_and_never_deletes(stage_dir: Path, hub: FakeHub):
    publish(stage_dir, mode="main", log=lambda _m: None)
    raw = next(k for k in hub.refs["main"] if k.startswith("raw/") and k.endswith(".txt"))
    hub.refs["main"][raw] = b"different upstream bytes"
    hub.refs["main"]["normalized/v0.2/legacy/keep.json"] = b"{}"
    with pytest.raises(DataIntegrityError, match="immutable"):
        publish(stage_dir, mode="main", log=lambda _m: None)
    del hub.refs["main"][raw]
    receipt = publish(stage_dir, mode="main", log=lambda _m: None)
    assert "normalized/v0.2/legacy/keep.json" in hub.refs["main"] and receipt["plan"]["retained"] == 1


def test_publish_resumes_open_pull_request(stage_dir: Path, hub: FakeHub, monkeypatch):
    calls = {"n": 0}
    real = hub.upload_folder

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("interrupted")
        return real(**kwargs)

    monkeypatch.setattr(hub, "upload_folder", flaky)
    with pytest.raises(ConnectionError):
        publish(stage_dir, mode="pr", log=lambda _m: None)
    receipt = publish(stage_dir, mode="pr", log=lambda _m: None)
    assert len(hub.prs) == 1 and receipt["revision"] == "refs/pr/1" and receipt["verified"]


def test_publish_rejects_wrong_oidc_resource(stage_dir: Path, hub: FakeHub, monkeypatch):
    monkeypatch.setenv("HF_OIDC_RESOURCE", "datasets/someone/else")
    with pytest.raises(ValueError, match="HF_OIDC_RESOURCE"):
        publish(stage_dir, dry_run=True, log=lambda _m: None)


def test_plan_matches_small_files_by_git_blob(stage_dir: Path):
    from aipe_magnetics_database.hub.publish import RemoteFile
    checksums = load_checksums(stage_dir)
    rel = "README.md"
    remote = {rel: RemoteFile(rel, checksums[rel]["size"], git_blob_sha1(stage_dir / rel), None)}
    plan = make_plan(stage_dir, remote)
    assert rel in plan.unchanged and rel not in plan.upload


# -- sync and version-pinned reads ---------------------------------------------------------------------------------
def _serve(files: dict[str, bytes], commit: str):
    def get(url: str) -> bytes:
        if url.endswith(f"/revision/main"):
            return json.dumps({"sha": commit, "lastModified": "2026-11-07T00:00:00.000Z"}).encode()
        if "/tree/" in url:
            return json.dumps([{"type": "file", "path": k, "size": len(v), "oid": "x",
                                "lfs": {"oid": sha(v), "size": len(v), "pointerSize": 1}} for k, v in files.items()]).encode()
        path = url.split(f"/resolve/{commit}/", 1)[1]
        from urllib.parse import unquote
        return files[unquote(path)]
    return get


def test_sync_records_exact_commit(stage_dir: Path, tmp_path: Path):
    from aipe_magnetics_database.hub.sync import sync_release
    files = {k: (stage_dir / k).read_bytes() for k in load_checksums(stage_dir)}
    root = tmp_path / "checkout"
    (root / "data").mkdir(parents=True)
    commit = "c" * 40
    release = sync_release(revision="main", root=root, get=_serve(files, commit), log=lambda _m: None)
    stored = json.loads((root / "data/huggingface.json").read_text(encoding="utf-8"))
    assert stored["commit"] == commit and set(stored["files"]) == set(files) and release["sources"].keys() == set(SMALL)
    files["manifests/dataset-index.json"] = b"{}"  # tampered on the Hub: size/sha mismatch
    files_bad = dict(files)
    with pytest.raises(Exception):
        sync_release(revision="main", root=root, get=_serve(files_bad, commit), log=lambda _m: None)


@pytest.fixture()
def data_store(tmp_path: Path) -> Path:
    """A data store whose Paderborn table sidecar exists only in the 'published release'."""
    root = tmp_path / "data"
    shutil.copytree(ROOT / "data/normalized/paderborn-lea", root / "normalized/paderborn-lea")
    shutil.copytree(ROOT / "data/canonical", root / "canonical")
    return root


def test_store_reads_missing_sidecar_from_published_release(data_store: Path, tmp_path: Path, monkeypatch):
    from aipe_magnetics_database.data.sources import cache as cache_mod
    from aipe_magnetics_database.store import MagneticDatabase
    rel = "tables/paderborn-lea-n87-complex-permeability-magnet.jsonl"
    sidecar = data_store / "normalized/paderborn-lea" / rel
    body = sidecar.read_bytes()
    sidecar.unlink()
    path = layout.normalized_path("paderborn_lea_materialdatabase", rel)
    (data_store / "huggingface.json").write_text(json.dumps({"repo_id": layout.DEFAULT_REPO_ID, "commit": "a" * 40,
                                                             "files": {path: {"sha256": sha(body), "size": len(body)}}}))
    record = next(r for r in (data_store / "normalized/paderborn-lea/measurements/measurements-001.jsonl").read_text().splitlines()
                  if rel in r)
    rid = json.loads(record)["id"]
    served = []

    def transport(url, timeout):
        served.append(url)
        assert f"/resolve/{'a' * 40}/" in url
        return io.BytesIO(body)

    monkeypatch.setattr(cache_mod, "urllib_transport", transport)
    monkeypatch.setenv("AIPE_MAGNETICS_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("AIPE_MAGNETICS_ALLOW_DOWNLOAD", raising=False)
    with pytest.raises(DataNotFoundError, match="AIPE_MAGNETICS_ALLOW_DOWNLOAD"):
        MagneticDatabase(data_store).table(rid, limit=1)
    table = MagneticDatabase(data_store, allow_download=True).table(rid)
    assert len(table.rows) == table.total_rows == len(body.splitlines()) and len(served) == 1
    MagneticDatabase(data_store).table(rid)  # cached: works offline, no new request
    assert len(served) == 1
    assert MagneticDatabase(data_store).summary()["huggingface"]["commit"] == "a" * 40


def test_store_without_release_keeps_offline_behaviour():
    import aipe_magnetics_database.api as api
    assert api.hub_release()["published"] in (False, True)
    table = api.get_table("measurement.princeton-magnet-n87-core-loss", frequency="100 kHz", temperature="90 degC", limit=2)
    assert table["returned_rows"] == 2


# -- transfer assets -----------------------------------------------------------------------------------------------
def _tar(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_safe_extract_rejects_traversal_and_unreferenced_members(tmp_path: Path):
    from aipe_magnetics_database.hub.transfer import safe_extract
    archive = tmp_path / "a.tar.gz"
    archive.write_bytes(_tar({"../evil": b"x"}))
    with pytest.raises(DataIntegrityError):
        safe_extract(archive, tmp_path / "out", {"../evil"})
    archive.write_bytes(_tar({"tables-local/other.jsonl": b"x"}))
    with pytest.raises(DataIntegrityError, match="unexpected member"):
        safe_extract(archive, tmp_path / "out", {"tables-local/t.jsonl"})


def test_restore_tables_verifies_each_sidecar(tmp_path: Path, monkeypatch):
    from aipe_magnetics_database.data.sources import cache as cache_mod
    from aipe_magnetics_database.hub import transfer
    rows = b'[1,2]\n[3,4]\n'
    base = tmp_path / "data/normalized/src"
    (base / "measurements").mkdir(parents=True)
    (base / "index.json").write_text(json.dumps({"shards": {"measurements": ["measurements/m.jsonl"]}, "tables_local": ["tables-local/t.jsonl"]}))
    (base / "measurements/m.jsonl").write_text(json.dumps({"id": "measurement.t", "extensions": {"aipe.table": {
        "rows_file": "tables-local/t.jsonl", "rows_sha256": sha(rows)}}}) + "\n")
    good, bad = _tar({"tables-local/t.jsonl": rows}), _tar({"tables-local/t.jsonl": b"[9,9]\n"})
    for blob, ok in ((bad, False), (good, True)):
        pins = [{"id": "t", "url": f"https://example.org/{sha(blob)[:8]}.tar.gz", "sha256": sha(blob), "extract_to": "data/normalized/src"}]
        monkeypatch.setattr(transfer, "pins", lambda pins=pins: pins)
        monkeypatch.setattr(cache_mod, "urllib_transport", lambda url, timeout, blob=blob: io.BytesIO(blob))
        if not ok:
            with pytest.raises(DataIntegrityError, match="rows_sha256"):
                transfer.restore(root=tmp_path, cache_dir=str(tmp_path / "c"), allow_download=True, log=lambda _m: None)
            assert not (base / "tables-local/t.jsonl").exists()
        else:
            assert transfer.restore(root=tmp_path, cache_dir=str(tmp_path / "c"), allow_download=True, log=lambda _m: None)[0]["restored"] == 1
            assert (base / "tables-local/t.jsonl").read_bytes() == rows
            assert transfer.restore(root=tmp_path, cache_dir=str(tmp_path / "c"), log=lambda _m: None)[0]["restored"] == 0


def test_transfer_pin_is_well_formed():
    from aipe_magnetics_database.hub.transfer import pins
    for asset in pins():
        assert len(asset["sha256"]) == 64 and asset["url"].startswith("https://github.com/FulongLi/AIPE-Magnetics-Database/releases/")
        assert (ROOT / asset["extract_to"] / "index.json").is_file()


def test_publish_leaves_verified_pr_open_when_merge_is_refused(stage_dir: Path, hub: FakeHub, monkeypatch):
    import huggingface_hub

    class Refused(huggingface_hub.errors.HfHubHTTPError):
        def __init__(self):
            Exception.__init__(self, "401")
            self.response = SimpleNamespace(status_code=401)

    def refuse(*a, **k):
        raise Refused()

    monkeypatch.setattr(hub, "merge_pull_request", refuse)
    receipt = publish(stage_dir, mode="pr", merge=True, log=lambda _m: None)
    assert receipt["verified"] and receipt["merged"] is False and "HTTP 401" in receipt["merge_error"] and receipt["revision"] == "refs/pr/1"
    assert not any(k.startswith("raw/") for k in hub.refs["main"])  # nothing reached main


def test_publish_tolerates_unlistable_discussions(stage_dir: Path, hub: FakeHub, monkeypatch):
    import huggingface_hub

    class Refused(huggingface_hub.errors.HfHubHTTPError):
        def __init__(self):
            Exception.__init__(self, "401")
            self.response = SimpleNamespace(status_code=401)

    def refuse(*a, **k):
        raise Refused()

    monkeypatch.setattr(hub, "get_repo_discussions", refuse)
    assert publish(stage_dir, mode="pr", log=lambda _m: None)["verified"]
