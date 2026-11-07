# Hugging Face dataset integration

The data live in three layers:

| Layer | Repository | Role |
| --- | --- | --- |
| 1 | [`FulongLi/AIPE-Magnetics-Dataset`](https://huggingface.co/datasets/FulongLi/AIPE-Magnetics-Dataset) (Hugging Face) | storage: original files, normalised v0.2 records, large tables, manifests, versioned releases |
| 2 | this repository | Database Manager: source registry, acquisition, importers, validation, quality checks, publishing, API / CLI / MCP, Data Explorer |
| 3 | [AIPE-Magnetics-Design](https://github.com/FulongLi/AIPE-Magnetics-Design) | engineering application; reads data only through this package |

GitHub keeps the code, the registry, the normalised records small enough for Git (bundled into the wheel for offline
use) and `data/huggingface.json`, which pins the published release. Large sidecars and the original files live on
Hugging Face only.

## Dataset layout

```
raw/<source-id>/files/...          original files, byte-for-byte (immutable)
raw/<source-id>/SOURCES.json       pinned URL, version and SHA-256 of every original; upstream LICENSE / NOTICE beside it
normalized/v0.2/<source-id>/       index.json, record shards, datasets/, tables/ and tables-local/ sidecars
manifests/sources.json             every investigated source: licence, status, published or not and why
manifests/dataset-index.json       per-source counts, table rows, sizes, formats, licence, citation, retrieval date
manifests/checksums.json           SHA-256 and size of every file
manifests/quality-report.json      automated checks (see docs/quality-summary.md)
README.md, LICENSE.md              dataset card, per-source licences
```

`<source-id>` is the registry id (`princeton_magnet`, `openmagnetics_mas`, ...). Only sources whose status is
`integrated` and whose licence permits redistribution are staged (`hub.stage.publishable`); the rest appear in
`manifests/sources.json` as metadata only. Records are copied unchanged, so a record's `rows_file` resolves under its
source prefix on the Hub exactly as in `data/normalized/<directory>/`.

## Publishing (GitHub Actions, Trusted Publisher)

`.github/workflows/publish-huggingface.yml` runs on manual dispatch only:

1. **stage** (no publishing rights): restores the large `tables-local/` sidecars from pinned transfer assets
   (`hub restore-tables`), validates every record, downloads the pinned raw inputs into the SHA-256 verifying cache,
   builds the release tree (`hub stage`), re-hashes it (`hub verify-stage`) and fails on quality errors.
2. **publish** (`permissions: id-token: write`, `HF_OIDC_RESOURCE=datasets/FulongLi/AIPE-Magnetics-Dataset`):
   `huggingface_hub` exchanges the GitHub OIDC token for a one-hour token scoped to the dataset. No HF token is
   stored in GitHub and none is printed. `hub publish` then
   * re-verifies the stage, lists the Hub files and skips identical ones (repeated runs upload nothing),
   * aborts if any `raw/` path exists on the Hub with different bytes (raw is immutable) and never deletes anything,
   * opens a Hub pull request (or resumes the open one of the same release; interrupted uploads resume because
     `upload_folder` skips committed files and Xet deduplicates uploaded chunks),
   * verifies the SHA-256 (LFS/Xet) or Git blob id and size of every file at the PR revision, and
   * with `merge: true` opens and merges the pull request, then verifies `main` again.

Inputs: `mode` (`pr` or `main`), `merge`, `sources` (subset), `dry_run` (authenticate and plan only).

The Trusted Publisher on the dataset (Settings -> Trusted Publishers -> GitHub Actions) must match
`repository = FulongLi/AIPE-Magnetics-Database`, `workflow = publish-huggingface.yml` and, if a branch is set, the
branch the workflow is dispatched on (`main`). It must not be read-only. On a mismatch the token exchange fails with
`invalid_grant`; the job summary then lists the claims the run presented.

## Recording a release in GitHub

```sh
aipe-magnetics-db hub sync --revision main     # resolve to the exact commit, verify every file, write data/huggingface.json
aipe-magnetics-db hub status
```

`data/huggingface.json` holds the dataset repository, the exact commit, per-source prefixes and the SHA-256 of every
published file; it is bundled into the wheel. `aipe-magnetics-db quality` compares GitHub's normalised files with it
(`hub_consistency`).

## Reading published data

Bundled data work offline as before. A table that is only on Hugging Face (for example the 81 MAS B-H cycles in
`tables-local/`) is downloaded on request from `resolve/<exact commit>/<path>`, verified against the SHA-256 in
`data/huggingface.json` *and* the record's `rows_sha256`, and cached in `AIPE_MAGNETICS_CACHE` (default
`~/.cache/aipe-magnetics`). No account or `huggingface_hub` is needed.

```sh
AIPE_MAGNETICS_ALLOW_DOWNLOAD=1 aipe-magnetics-db measurements table measurement.mas-dmegc-dmr95-0400-bh-cycle --limit 5
aipe-magnetics-db hub fetch measurement.mas-dmegc-dmr95-0400-bh-cycle    # download once, then offline
```

```python
db.get_table("measurement.mas-dmegc-dmr95-0400-bh-cycle", allow_download=True)
```

## Local commands

| Command | Purpose |
| --- | --- |
| `hub stage --out DIR [--source ID] [--allow-download]` | build the release tree |
| `hub verify-stage DIR` | re-hash a stage against its checksums |
| `hub publish DIR [--mode pr|main] [--merge] [--dry-run]` | publish (needs Hub write access; normally run by the workflow) |
| `hub sync [--revision REV]` | record a verified release in `data/huggingface.json` |
| `hub restore-tables [--allow-download]` | restore `tables-local/` sidecars from pinned transfer assets |
| `quality` | run the quality checks; writes `data/reports/quality-report.json` and `docs/quality-summary.md` |
