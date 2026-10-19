"""Local cache for on-demand downloads and pinned checkouts of external sources.

* Location: ``root`` argument, else the environment variable ``AIPE_MAGNETICS_CACHE``, else
  ``.aipe-cache`` in the current working directory. Cached files are never modified by the adapters.
* Manifest: ``<root>/manifest.json`` with one entry per URL ``{url, path, sha256, size, commit,
  license, retrieved_at (caller supplied or omitted), source_id}`` and one per pinned checkout.
* :meth:`LocalCache.fetch` **never downloads unless** ``allow_download=True`` (otherwise
  :class:`DownloadNotAllowedError`); after a download the SHA-256 is verified against
  ``expected_sha256`` and a mismatch removes the file and raises :class:`DataIntegrityError`.
* :meth:`LocalCache.clone_pinned` makes a shallow ``git`` clone of exactly one commit (subprocess,
  no shell, no credentials, no prompts) and verifies ``HEAD`` equals the pinned commit; nothing
  inside the checkout is ever executed or imported. An existing cached checkout is served only after
  :meth:`LocalCache.checkout_problems` found nothing (HEAD at the pinned commit, no tracked file
  modified or deleted; untracked additions are not detected).
* :meth:`LocalCache.verify` re-hashes every cached file and re-checks every checkout the same way.

Transport is the standard library (``urllib`` with timeouts, http(s) only, no credentials in URLs;
every redirect target is re-validated and an https -> http downgrade is refused); an alternative
``transport`` callable can be injected so that tests and verification checks stay offline.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, BinaryIO, Callable

from pydantic import BaseModel, ConfigDict, Field

from aipe_magnetics import __version__
from aipe_magnetics.common.errors import AuthorizationError, DataIntegrityError, InvalidInputError, UnsupportedOperationError
from aipe_magnetics.common.jsonio import sha256_file

DEFAULT_CACHE_DIRNAME = ".aipe-cache"
CACHE_ENV_VAR = "AIPE_MAGNETICS_CACHE"
MANIFEST_VERSION = "aipe.magnetics.cache/0.2"
MANIFEST_NAME = "manifest.json"
USER_AGENT = f"aipe-magnetics/{__version__} (+https://github.com/FulongLi/AIPE-Magnetics-Database)"
_SHA_RE = re.compile(r"^[a-f0-9]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
#: device names that Windows refuses as file stems (with or without an extension), case-insensitively
WINDOWS_RESERVED_NAMES: frozenset[str] = frozenset({"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))})

Transport = Callable[[str, float], BinaryIO]


def is_windows_reserved_name(name: str) -> bool:
    """True when ``name`` (or its stem before the first dot) is a Windows reserved device name."""
    return name.split(".", 1)[0].strip().upper() in WINDOWS_RESERVED_NAMES


class DownloadNotAllowedError(AuthorizationError):
    """A network download (or clone) was needed but the caller did not pass ``allow_download=True``."""

    code = "download_not_allowed"


class CacheEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str
    path: str = Field(description="path relative to the cache root (POSIX separators)")
    filename: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size: int = Field(ge=0)
    commit: str | None = None
    license: str = "NOASSERTION"
    retrieved_at: str | None = Field(default=None, description="caller-supplied RFC 3339 timestamp; omitted otherwise")
    source_id: str | None = None


class CheckoutEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repository_url: str
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    path: str
    license: str = "NOASSERTION"
    retrieved_at: str | None = None
    source_id: str | None = None


def check_fetch_url(url: str, *, previous: str | None = None) -> None:
    """Reject anything but http(s) URLs without embedded credentials; with ``previous`` also an https -> http downgrade."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("https", "http"):
        raise InvalidInputError(f"only http(s) URLs can be fetched, got {url!r}", details={"url": url})
    if parts.username or parts.password:
        raise InvalidInputError("URLs with embedded credentials are refused", details={"url": parts.hostname})
    if previous is not None and urllib.parse.urlsplit(previous).scheme == "https" and parts.scheme != "https":
        raise InvalidInputError(f"redirect from {previous!r} to the non-https URL {url!r} is refused (downgrade)",
                                details={"from": previous, "to": url})


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validates every redirect target with :func:`check_fetch_url` before urllib follows it."""

    def redirect_request(self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        try:
            check_fetch_url(newurl, previous=req.full_url)
        except InvalidInputError:
            close = getattr(fp, "close", None)
            if close is not None:
                close()
            raise
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener: urllib.request.OpenerDirector | None = None


def _safe_opener() -> urllib.request.OpenerDirector:
    global _opener
    if _opener is None:
        _opener = urllib.request.build_opener(SafeRedirectHandler())
    return _opener


def urllib_transport(url: str, timeout: float) -> BinaryIO:
    """Open ``url`` (http/https only, no credentials, redirects re-validated, no https -> http downgrade) with a timeout."""
    check_fetch_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    response = _safe_opener().open(request, timeout=timeout)  # noqa: S310 - scheme validated above and on every redirect
    final = response.geturl()
    if final != url:
        try:
            check_fetch_url(final, previous=url)
        except InvalidInputError:
            response.close()
            raise
    return response


def _safe_filename(url: str, filename: str | None) -> str:
    name = filename or urllib.parse.unquote(urllib.parse.urlsplit(url).path.rsplit("/", 1)[-1]) or "download"
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "download"
    if is_windows_reserved_name(name):
        name = f"file-{name}"  # CON, NUL, COM1 ... cannot be created on Windows
    return name[:120]


def _force_remove(func: Callable[..., Any], path: str, _exc: Any) -> None:  # pragma: no cover - Windows read-only files
    os.chmod(path, stat.S_IWRITE)
    func(path)


def rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, onerror=_force_remove)


class LocalCache:
    """Manifest-backed local cache (see module docstring)."""

    def __init__(self, root: str | Path | None = None, *, transport: Transport | None = None, max_bytes: int = 2 << 30):
        if root is None:
            root = os.environ.get(CACHE_ENV_VAR) or DEFAULT_CACHE_DIRNAME
        self.root = Path(root)
        self._transport: Transport = transport or urllib_transport
        self.max_bytes = int(max_bytes)

    # -- manifest ---------------------------------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    def _load(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return {"manifest_version": MANIFEST_VERSION, "files": {}, "checkouts": {}}
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise DataIntegrityError(f"cache manifest {self.manifest_path} is unreadable: {error}") from None
        if not isinstance(data, dict) or data.get("manifest_version") != MANIFEST_VERSION:
            raise DataIntegrityError(f"cache manifest {self.manifest_path} has an unexpected version {data.get('manifest_version')!r}")
        data.setdefault("files", {})
        data.setdefault("checkouts", {})
        return data

    def _save(self, data: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.root, delete=False, newline="\n", suffix=".tmp") as stream:
            stream.write(text)
            temporary = Path(stream.name)
        os.replace(temporary, self.manifest_path)

    def entries(self) -> dict[str, CacheEntry]:
        return {url: CacheEntry.model_validate(item) for url, item in self._load()["files"].items()}

    def checkouts(self) -> dict[str, CheckoutEntry]:
        return {key: CheckoutEntry.model_validate(item) for key, item in self._load()["checkouts"].items()}

    def lookup(self, url: str) -> CacheEntry | None:
        item = self._load()["files"].get(url)
        return None if item is None else CacheEntry.model_validate(item)

    def path_of(self, entry: CacheEntry | CheckoutEntry) -> Path:
        return self.root / Path(entry.path)

    # -- files ------------------------------------------------------------------------------------
    def fetch(self, url: str, expected_sha256: str | None = None, allow_download: bool = False, *, filename: str | None = None,
              commit: str | None = None, license: str = "NOASSERTION", source_id: str | None = None,
              retrieved_at: str | None = None, timeout: float = 60.0) -> Path:
        """Return the local path of ``url``; download only with ``allow_download=True`` and verify the digest."""
        if not isinstance(url, str) or not url:
            raise InvalidInputError("url must be a non-empty string")
        if expected_sha256 is not None:
            expected_sha256 = expected_sha256.lower()
            if not _SHA_RE.match(expected_sha256):
                raise InvalidInputError("expected_sha256 must be 64 lowercase hex characters")
        cached = self.lookup(url)
        if cached is not None:
            path = self.path_of(cached)
            if path.is_file():
                actual = sha256_file(path)
                if actual != cached.sha256:
                    raise DataIntegrityError(f"cached file {path} no longer matches its manifest digest "
                                             f"(manifest {cached.sha256[:12]}..., file {actual[:12]}...); remove it and fetch again",
                                             details={"url": url, "path": str(path)})
                if expected_sha256 is not None and actual != expected_sha256:
                    raise DataIntegrityError(f"cached file for {url} has digest {actual[:12]}... but {expected_sha256[:12]}... was "
                                             "expected (different upstream version); remove the cache entry to refetch",
                                             details={"url": url, "path": str(path)})
                return path
        if not allow_download:
            raise DownloadNotAllowedError(
                f"{url} is not in the local cache ({self.root}) and downloading was not authorised; pass allow_download=True "
                "to fetch it (network access is opt-in)", details={"url": url, "cache_root": str(self.root)})
        return self._download(url, expected_sha256, filename=filename, commit=commit, license=license, source_id=source_id,
                              retrieved_at=retrieved_at, timeout=timeout)

    def _download(self, url: str, expected_sha256: str | None, *, filename: str | None, commit: str | None, license: str,
                  source_id: str | None, retrieved_at: str | None, timeout: float) -> Path:
        name = _safe_filename(url, filename)
        self.root.mkdir(parents=True, exist_ok=True)
        tmp_dir = self.root / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        with tempfile.NamedTemporaryFile("wb", dir=tmp_dir, delete=False, suffix=".part") as out:
            partial = Path(out.name)
            try:
                stream = self._transport(url, timeout)
                try:
                    while True:
                        block = stream.read(1 << 20)
                        if not block:
                            break
                        size += len(block)
                        if size > self.max_bytes:
                            raise DataIntegrityError(f"download of {url} exceeds the cache limit of {self.max_bytes} bytes")
                        digest.update(block)
                        out.write(block)
                finally:
                    close = getattr(stream, "close", None)
                    if close is not None:
                        close()
            except Exception:
                out.close()
                partial.unlink(missing_ok=True)
                raise
        actual = digest.hexdigest()
        if expected_sha256 is not None and actual != expected_sha256:
            partial.unlink(missing_ok=True)
            raise DataIntegrityError(
                f"downloaded {url} has sha256 {actual} but {expected_sha256} was expected; the file was discarded",
                details={"url": url, "expected": expected_sha256, "actual": actual, "size": size})
        target_dir = self.root / "files" / actual[:2] / actual
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / name
        os.replace(partial, target)
        entry = CacheEntry(url=url, path=target.relative_to(self.root).as_posix(), filename=name, sha256=actual, size=size,
                           commit=commit, license=license, retrieved_at=retrieved_at, source_id=source_id)
        data = self._load()
        data["files"][url] = entry.model_dump(mode="json", exclude_none=True)
        self._save(data)
        return target

    # -- checkouts --------------------------------------------------------------------------------
    @staticmethod
    def _git(args: list[str], *, cwd: Path | None, timeout: float, git: str, strip: bool = True) -> str:
        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"
        env.setdefault("GIT_ASKPASS", "")
        try:
            completed = subprocess.run([git, *args], cwd=None if cwd is None else str(cwd), capture_output=True, text=True,
                                       timeout=timeout, env=env, check=False)
        except FileNotFoundError:
            raise UnsupportedOperationError("git is required for pinned checkouts but was not found on PATH",
                                            details={"git": git}) from None
        except subprocess.TimeoutExpired:
            raise DataIntegrityError(f"git {' '.join(args[:2])} timed out after {timeout} s") from None
        if completed.returncode != 0:
            raise DataIntegrityError(f"git {' '.join(args[:2])} failed ({completed.returncode}): {completed.stderr.strip()[:500]}",
                                     details={"args": args, "stderr": completed.stderr[-2000:]})
        return completed.stdout.strip() if strip else completed.stdout

    @classmethod
    def git_head(cls, path: str | Path, *, git: str = "git", timeout: float = 30.0) -> str:
        return cls._git(["-C", str(path), "rev-parse", "HEAD"], cwd=None, timeout=timeout, git=git)

    @classmethod
    def checkout_problems(cls, path: str | Path, commit: str, *, git: str = "git", timeout: float = 60.0) -> list[str]:
        """Problems of a pinned checkout: HEAD must equal ``commit`` and no tracked file may be modified or deleted.

        Uses ``git status --porcelain --untracked-files=no`` with ``--no-optional-locks`` (the checkout is
        not written to). Untracked additions are deliberately ignored; they cannot change what the adapters
        read because every file is addressed by its path in the pinned tree.
        """
        problems: list[str] = []
        try:
            head = cls.git_head(path, git=git, timeout=timeout)
        except (DataIntegrityError, UnsupportedOperationError) as error:
            return [f"cannot read HEAD: {error.message}"]
        if head != commit:
            problems.append(f"HEAD {head} differs from pinned {commit}")
        try:
            status = cls._git(["-C", str(path), "--no-optional-locks", "status", "--porcelain", "--untracked-files=no"],
                              cwd=None, timeout=timeout, git=git, strip=False)
        except (DataIntegrityError, UnsupportedOperationError) as error:
            return problems + [f"cannot read the working-tree status: {error.message}"]
        changed = [line[3:].rstrip("\r") for line in status.splitlines() if line.strip()]
        if changed:
            problems.append(f"{len(changed)} tracked file(s) modified or deleted since the pinned checkout: {changed[:10]}")
        return problems

    def find_checkout(self, repository_url: str, commit: str) -> Path | None:
        """Path of a cached checkout *without* integrity checks (use :meth:`clone_pinned`, which verifies, to obtain data)."""
        entry = self._load()["checkouts"].get(f"{repository_url}@{commit}")
        if entry is None:
            return None
        path = self.path_of(CheckoutEntry.model_validate(entry))
        return path if path.is_dir() else None

    def clone_pinned(self, repository_url: str, commit: str, *, allow_download: bool = False, license: str = "NOASSERTION",
                     source_id: str | None = None, retrieved_at: str | None = None, timeout: float = 600.0,
                     git: str = "git") -> Path:
        """Shallow clone of exactly ``commit`` into the cache; verified with ``git rev-parse HEAD``."""
        if not _COMMIT_RE.match(commit or ""):
            raise InvalidInputError("commit must be a 40-character lowercase hexadecimal hash")
        if not repository_url.startswith("https://"):
            raise InvalidInputError("repository_url must be an https:// URL")
        existing = self.find_checkout(repository_url, commit)
        if existing is not None:
            problems = self.checkout_problems(existing, commit, git=git)
            if problems:
                raise DataIntegrityError(f"cached checkout {existing} is not the pristine pinned commit {commit[:12]}: "
                                         + "; ".join(problems) + "; remove the directory and the manifest entry to re-clone",
                                         details={"path": str(existing), "commit": commit, "problems": problems})
            return existing
        if not allow_download:
            raise DownloadNotAllowedError(
                f"no cached checkout of {repository_url} at {commit} in {self.root} and cloning was not authorised; pass "
                "allow_download=True to clone (network access is opt-in)",
                details={"repository_url": repository_url, "commit": commit, "cache_root": str(self.root)})
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", repository_url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")) or "checkout"
        dest = self.root / "checkouts" / f"{name}-{commit[:12]}"
        rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._git(["init", "-q", str(dest)], cwd=None, timeout=timeout, git=git)
            self._git(["-C", str(dest), "remote", "add", "origin", repository_url], cwd=None, timeout=timeout, git=git)
            self._git(["-C", str(dest), "-c", "protocol.version=2", "fetch", "-q", "--depth", "1", "origin", commit],
                      cwd=None, timeout=timeout, git=git)
            self._git(["-C", str(dest), "-c", "advice.detachedHead=false", "checkout", "-q", "--detach", "FETCH_HEAD"],
                      cwd=None, timeout=timeout, git=git)
            head = self.git_head(dest, git=git)
        except Exception:
            rmtree(dest)
            raise
        if head != commit:
            rmtree(dest)
            raise DataIntegrityError(f"clone of {repository_url} ended at {head}, not the pinned commit {commit}; removed",
                                     details={"head": head, "commit": commit})
        entry = CheckoutEntry(repository_url=repository_url, commit=commit, path=dest.relative_to(self.root).as_posix(),
                              license=license, retrieved_at=retrieved_at, source_id=source_id)
        data = self._load()
        data["checkouts"][f"{repository_url}@{commit}"] = entry.model_dump(mode="json", exclude_none=True)
        self._save(data)
        return dest

    # -- verification -----------------------------------------------------------------------------
    def verify(self, *, git: str = "git") -> list[str]:
        """Re-check every cached file (existence, size, SHA-256) and checkout (HEAD, modified / deleted tracked files)."""
        problems: list[str] = []
        data = self._load()
        for url, raw in sorted(data["files"].items()):
            entry = CacheEntry.model_validate(raw)
            path = self.path_of(entry)
            if not path.is_file():
                problems.append(f"{url}: cached file missing ({entry.path})")
                continue
            if path.stat().st_size != entry.size:
                problems.append(f"{url}: size {path.stat().st_size} differs from manifest {entry.size}")
            actual = sha256_file(path)
            if actual != entry.sha256:
                problems.append(f"{url}: sha256 {actual} differs from manifest {entry.sha256}")
        for key, raw in sorted(data["checkouts"].items()):
            entry = CheckoutEntry.model_validate(raw)
            path = self.path_of(entry)
            if not path.is_dir():
                problems.append(f"{key}: checkout directory missing ({entry.path})")
                continue
            problems.extend(f"{key}: {problem}" for problem in self.checkout_problems(path, entry.commit, git=git))
        return problems


__all__ = ["CACHE_ENV_VAR", "DEFAULT_CACHE_DIRNAME", "MANIFEST_VERSION", "WINDOWS_RESERVED_NAMES", "CacheEntry", "CheckoutEntry",
           "DownloadNotAllowedError", "LocalCache", "SafeRedirectHandler", "Transport", "check_fetch_url", "is_windows_reserved_name",
           "urllib_transport", "rmtree"]
