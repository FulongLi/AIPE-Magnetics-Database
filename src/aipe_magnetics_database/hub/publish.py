"""Publish a staged release to the Hugging Face dataset and verify it (``aipe-magnetics-db hub publish``).

Authentication is whatever ``huggingface_hub`` resolves. In the GitHub Actions workflow
(``.github/workflows/publish-huggingface.yml``) that is a Trusted Publisher exchange: the job has
``permissions: id-token: write`` and ``HF_OIDC_RESOURCE=datasets/FulongLi/AIPE-Magnetics-Dataset``; no token is
stored and none is ever printed.

Rules enforced before anything is sent:

* the stage is intact (every file re-hashed against ``manifests/checksums.json``);
* files already on the Hub with identical content are skipped (repeated runs upload nothing);
* a file under ``raw/`` that exists on the Hub with *different* bytes aborts the publication (raw is immutable);
* nothing is ever deleted on the Hub; remote files absent from the stage are reported as ``retained``.

``mode="pr"`` (default) uploads into a new Hub pull request opened by the commit endpoint (``create_pr``), or resumes
the open pull request of the same release; ``merge=True`` merges it only after verification succeeded (if the token
may not merge, the verified pull request is left open and the receipt says so). ``mode="main"`` commits directly.
Verification compares SHA-256 (LFS / Xet files) or the Git blob id (small files) and the size of every staged
file at the resulting revision.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from aipe_magnetics_database.common.errors import DataIntegrityError
from aipe_magnetics_database.hub.layout import DEFAULT_REPO_ID, IMMUTABLE_PREFIXES, MANIFEST_PREFIX, OIDC_RESOURCE, REPO_TYPE
from aipe_magnetics_database.hub.stage import load_checksums, verify_stage
from aipe_magnetics_database.ingest.common import dumps_pretty, git_blob_sha1, sha256_path

RELEASE_TITLE = "AIPE Magnetics release"


def _hf():
    try:
        import huggingface_hub
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise ImportError("publishing needs huggingface_hub: pip install 'aipe-magnetics-database[hub]'") from error
    return huggingface_hub


@dataclass
class RemoteFile:
    path: str
    size: int
    blob_id: str
    sha256: str | None  # LFS / Xet files only


@dataclass
class Plan:
    upload: list[str] = field(default_factory=list)
    replace: list[str] = field(default_factory=list)  # subset of upload: paths that exist on the Hub with other content
    unchanged: list[str] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    retained: list[str] = field(default_factory=list)
    upload_bytes: int = 0

    def to_json(self) -> dict[str, Any]:
        return {"upload": len(self.upload), "replace": self.replace, "unchanged": len(self.unchanged), "conflicts": self.conflicts, "retained": len(self.retained),
                "retained_paths": self.retained[:50], "upload_bytes": self.upload_bytes}


def remote_files(api: Any, repo_id: str, revision: str | None = None) -> dict[str, RemoteFile]:
    hf = _hf()
    out: dict[str, RemoteFile] = {}
    try:
        items = api.list_repo_tree(repo_id, repo_type=REPO_TYPE, revision=revision, recursive=True)
        for item in items:
            if isinstance(item, hf.hf_api.RepoFile):
                out[item.path] = RemoteFile(item.path, item.size, item.blob_id, item.lfs.sha256 if item.lfs else None)
    except hf.errors.RevisionNotFoundError:
        return {}
    except hf.errors.EntryNotFoundError:
        return {}
    return out


def matches(local: Path, meta: dict[str, Any], remote: RemoteFile) -> bool:
    if remote.size != meta["size"]:
        return False
    if remote.sha256 is not None:
        return remote.sha256 == meta["sha256"]
    return remote.blob_id == git_blob_sha1(local)


def make_plan(stage_dir: Path, remote: dict[str, RemoteFile]) -> Plan:
    plan = Plan()
    checksums = load_checksums(stage_dir)
    for rel in sorted(checksums):
        meta = checksums[rel]
        other = remote.get(rel)
        if other is None:
            plan.upload.append(rel)
            plan.upload_bytes += meta["size"]
        elif matches(stage_dir / rel, meta, other):
            plan.unchanged.append(rel)
        elif rel.startswith(IMMUTABLE_PREFIXES):
            plan.conflicts.append({"path": rel, "local_sha256": meta["sha256"], "remote_sha256": other.sha256, "remote_size": other.size,
                                   "local_size": meta["size"]})
        else:
            plan.upload.append(rel)
            plan.replace.append(rel)
            plan.upload_bytes += meta["size"]
    plan.retained = sorted(set(remote) - set(checksums) - {".gitattributes"})
    return plan


def verify_remote(stage_dir: Path, remote: dict[str, RemoteFile]) -> list[str]:
    problems = []
    for rel, meta in load_checksums(stage_dir).items():
        other = remote.get(rel)
        if other is None:
            problems.append(f"missing on the Hub: {rel}")
        elif not matches(stage_dir / rel, meta, other):
            problems.append(f"content differs on the Hub: {rel}")
    return problems


def _release_id(stage_dir: Path) -> str:
    return sha256_path(stage_dir / MANIFEST_PREFIX / "checksums.json")[:12]


def _delta(stage_dir: Path, paths: list[str]) -> Path:
    delta = stage_dir.parent / f".{stage_dir.name}-delta"
    if delta.exists():
        shutil.rmtree(delta)
    for rel in paths:
        dst = delta / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(stage_dir / rel, dst)
        except OSError:
            shutil.copy2(stage_dir / rel, dst)
    return delta


def _open_release_pr(api: Any, repo_id: str, title: str, log: Callable[[str], None] = print) -> Any | None:
    """The open Hub pull request of this release (to resume into), or ``None``; listing failures are not fatal."""
    hf = _hf()
    try:
        for disc in api.get_repo_discussions(repo_id, repo_type=REPO_TYPE, discussion_type="pull_request"):
            if disc.title == title and disc.status in ("open", "draft"):
                return disc
    except hf.errors.HfHubHTTPError as error:
        log(f"could not list Hub pull requests ({type(error).__name__}); a new pull request will be opened")
    return None


def publish(stage_dir: str | Path, *, repo_id: str = DEFAULT_REPO_ID, mode: str = "pr", merge: bool = False, dry_run: bool = False,
            receipt_path: str | Path | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Upload the stage (see module docstring); returns the publication receipt."""
    hf = _hf()
    stage_dir = Path(stage_dir)
    if mode not in ("pr", "main"):
        raise ValueError("mode must be 'pr' or 'main'")
    problems = verify_stage(stage_dir)
    if problems:
        raise DataIntegrityError(f"stage {stage_dir} is not intact: {problems[:5]}")
    api = hf.HfApi()
    if os.environ.get("HF_OIDC_RESOURCE") and os.environ["HF_OIDC_RESOURCE"] != f"datasets/{repo_id}":
        raise ValueError(f"HF_OIDC_RESOURCE is {os.environ['HF_OIDC_RESOURCE']!r} but the target is datasets/{repo_id}")
    api.auth_check(repo_id, repo_type=REPO_TYPE)
    remote = remote_files(api, repo_id)
    plan = make_plan(stage_dir, remote)
    release = _release_id(stage_dir)
    index = json.loads((stage_dir / MANIFEST_PREFIX / "dataset-index.json").read_text(encoding="utf-8"))
    log(f"plan: upload {len(plan.upload)} file(s) ({plan.upload_bytes:,} B; {len(plan.replace)} replace an older version: "
        f"{plan.replace[:10]}), unchanged {len(plan.unchanged)}, retained {len(plan.retained)}, "
        f"conflicts {len(plan.conflicts)}")
    receipt: dict[str, Any] = {"repo_id": repo_id, "repo_type": REPO_TYPE, "mode": mode, "release_id": release, "dry_run": dry_run,
                               "builder": index.get("builder"), "plan": plan.to_json(), "verified": False}
    if plan.conflicts:
        raise DataIntegrityError(f"{len(plan.conflicts)} raw file(s) already on the Hub with different content (raw is immutable): "
                                 f"{[c['path'] for c in plan.conflicts[:5]]}", details={"conflicts": plan.conflicts[:20]})
    if dry_run:
        return _write_receipt(receipt, receipt_path)
    commit = index["builder"].get("commit") or "unknown"
    title = f"{RELEASE_TITLE} {release}"
    description = (f"Release {release} built by aipe-magnetics-database {index['builder']['version']} from "
                   f"https://github.com/FulongLi/AIPE-Magnetics-Database/commit/{commit}.\n\n"
                   f"Upload {len(plan.upload)} file(s), {plan.upload_bytes:,} bytes; {len(plan.unchanged)} unchanged; nothing deleted.\n"
                   f"Replaced (new version of a non-raw file): {', '.join(plan.replace) or 'none'}.\n"
                   f"Every file is listed with SHA-256 in manifests/checksums.json.")
    revision = None
    pr = None
    if plan.upload:
        delta = _delta(stage_dir, plan.upload)
        if mode == "pr":
            pr = _open_release_pr(api, repo_id, title, log)
            if pr is not None:
                log(f"resuming Hub pull request #{pr.num}")
                revision = pr.git_reference
        if mode == "pr" and pr is None:
            # The pull request is opened by the commit endpoint (create_pr), which accepts the repo-scoped Trusted
            # Publisher token; the discussions endpoint used by create_pull_request rejects it (401).
            log("uploading into a new Hub pull request ...")
            info = api.upload_folder(repo_id=repo_id, repo_type=REPO_TYPE, folder_path=delta, create_pr=True, commit_message=title,
                                     commit_description=description)
            pr = SimpleNamespace(num=info.pr_num, git_reference=info.pr_revision, status="open")
            revision = info.pr_revision
            log(f"opened Hub pull request #{pr.num}")
        else:
            log(f"uploading to {revision or 'main'} ...")
            api.upload_folder(repo_id=repo_id, repo_type=REPO_TYPE, folder_path=delta, revision=revision, commit_message=title,
                              commit_description=description)
        shutil.rmtree(delta, ignore_errors=True)
    elif mode == "pr":
        pr = _open_release_pr(api, repo_id, title, log)
        revision = pr.git_reference if pr is not None else None
    verify_at = revision or "main"
    problems = verify_remote(stage_dir, remote_files(api, repo_id, verify_at))
    if problems:
        receipt["verification_problems"] = problems[:50]
        _write_receipt(receipt, receipt_path)
        raise DataIntegrityError(f"verification at {verify_at} failed: {problems[:5]}")
    info = api.repo_info(repo_id, repo_type=REPO_TYPE, revision=verify_at)
    receipt.update({"verified": True, "revision": verify_at, "commit": info.sha})
    if pr is not None:
        receipt["pull_request"] = {"num": pr.num, "url": f"https://huggingface.co/datasets/{repo_id}/discussions/{pr.num}",
                                   "ref": pr.git_reference}
    log(f"verified {len(load_checksums(stage_dir))} file(s) at {verify_at} ({info.sha})")
    if merge and pr is not None:
        try:
            if getattr(pr, "status", "open") == "draft":  # pull requests from create_pull_request start as drafts
                api.change_discussion_status(repo_id, pr.num, "open", repo_type=REPO_TYPE, comment="Every file verified; ready to merge.")
            api.merge_pull_request(repo_id, pr.num, repo_type=REPO_TYPE, comment="Merged after SHA-256 verification of every file.")
        except hf.errors.HfHubHTTPError as error:
            receipt.update({"merged": False, "merge_error": f"{type(error).__name__}: HTTP {getattr(error.response, 'status_code', '?')}",
                            "merge_hint": "merge the verified pull request on the Hub, or re-run with mode=main"})
            receipt["files"] = load_checksums(stage_dir)
            log(f"could not merge #{pr.num} with this token ({receipt['merge_error']}); the pull request is verified and left open")
            return _write_receipt(receipt, receipt_path)
        problems = verify_remote(stage_dir, remote_files(api, repo_id, "main"))
        if problems:
            raise DataIntegrityError(f"verification on main after merging #{pr.num} failed: {problems[:5]}")
        main = api.repo_info(repo_id, repo_type=REPO_TYPE, revision="main")
        receipt.update({"revision": "main", "commit": main.sha, "merged": True})
        log(f"merged #{pr.num}; main is {main.sha}")
    receipt["files"] = load_checksums(stage_dir)
    return _write_receipt(receipt, receipt_path)


def _write_receipt(receipt: dict[str, Any], path: str | Path | None) -> dict[str, Any]:
    if path is not None:
        Path(path).write_text(dumps_pretty(receipt), encoding="utf-8")
    return receipt


__all__ = ["OIDC_RESOURCE", "Plan", "RemoteFile", "make_plan", "matches", "publish", "remote_files", "verify_remote"]
