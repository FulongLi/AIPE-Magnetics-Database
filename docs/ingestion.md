# Ingestion and refresh

```sh
aipe-magnetics-db refresh --allow-download                    # all integrated sources, pinned inputs into the cache
aipe-magnetics-db refresh --source princeton_magnet --local princeton_magnet=/path/to/magnet/src/magnet/data
aipe-magnetics-db validate                                    # schema + references of every committed record
```

For each source the pipeline (`aipe_magnetics_database.ingest.pipeline.refresh`):

1. **Obtains pinned inputs**: `RawFile(name, url, sha256)` per file (`ingest/sources/*.py`, `ingest/pins/`). Local copies
   (`--local`, or `data/raw/<source>/files/` when committed) must hash to the pin; otherwise the `LocalCache`
   (`AIPE_MAGNETICS_CACHE`, default `.aipe-cache/`) serves or downloads them - downloads only with `--allow-download`,
   streamed to a temporary file and admitted only after the SHA-256 matches, so interrupted runs simply resume.
2. **Checks rights** from `ingest/registry.py`: only `integrated` sources are written; `reference` sources have
   user-supplied-file adapters whose output stays local (`aipe-magnetics-db user-import asu-mhz-csv|psma-coreloss-db`,
   `ingest.external.user_table`).
3. **Normalises** into schema v0.2 (SI units, original units retained, explicit column definitions, data class).
4. **Validates** every record (`MagneticRecord`, cross-record references) - a failure aborts the refresh.
5. **Deduplicates** and builds `data/canonical/` (`ingest/canonical.py`).
6. **Reports**: `data/reports/import-<source>.json` (inputs, counts, table rows, flags, every exclusion with its
   reason), `data/reports/coverage.json`, `data/sources.json`, `docs/source-catalogue.md`.

Output is deterministic (stable order, timestamps from the source pin, exact floats), so a re-run on the same inputs
writes no file; `tests/ingest/test_reproducible_imports.py` re-imports the committed raw files and compares bytes.
Streaming: sidecar tables are read and written line by line; MAS B-H loops (1.5 M points) go to `tables-local/`.

| Source | Importer | Pin | Raw policy |
| --- | --- | --- | --- |
| OpenMagnetics MAS | `ingest/sources/mas.py` | commit `c33ccfec`, SHA-256 per file (`data/MANIFEST.sha256` upstream; LFS object for advanced curves) | on demand (145 MB LFS) |
| Princeton MagNet | `ingest/sources/princeton.py` (reuses `PrincetonMagnetAdapter`) | commit `8a6892e9`, SHA-256 of the 10 `.h5` files | on demand (37 MB) |
| Paderborn LEA | `ingest/sources/paderborn.py` | commit `04fff8e9`, SHA-256 + git blob of 79 CSV + LICENSE | committed (GPL-3.0, 3.9 MB) |
| HEFMAG | `ingest/sources/zenodo.py:run_hefmag` | DOI 10.5281/zenodo.8292861 v1.0.0, SHA-256 of the zip | committed (CC-BY-4.0) |
| Zenodo 21626116 | `ingest/sources/zenodo.py:run_triangular` | DOI, SHA-256 of 9 files | committed (CC-BY-4.0) |

Adding a source: write `ingest/sources/<name>.py` with `RAW_FILES` and `run(output_root, *, local_dir, allow_download,
cache) -> ImportSummary` using `ingest.common` (record builders, `measurement_record`, `SourceWriter`,
`finish_source`), register it in `ingest/registry.py` and `pipeline.importers()`, and add a reproducibility test.
