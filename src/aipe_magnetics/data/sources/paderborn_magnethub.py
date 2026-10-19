"""Adapter for the Paderborn **MagNet-Hub** repository (https://github.com/upb-lea/mag-net-hub, MIT).

Pinned commit ``1ed635d7bfac3c2aa024471af363beadc05f8932`` (pyproject version 0.0.11; HEAD also carries
the undocumented sequence-model API). Facts recorded from the audit:

* loss models for **15 materials** (``3C90 3C92 3C94 3C95 3E6 3F4 77 78 79 ML95S N27 N30 N49 N87 T37``)
  from **2 teams**: ``paderborn`` (HARDCORE residual dilated CNN, TorchScript, 1024 samples per period)
  and ``sydney`` (MMINN state dicts, 128 samples per period); input units B T, f Hz, T degC; outputs
  p W/m^3 and h A/m; the Paderborn graph reports ``|oint H dB| f`` while Sydney's trapezoid has no abs;
* sequence (B -> H) models (``paderborn``, GRU8 ``.eqx``) for 15 materials assume the training sample
  rate of **16 MHz** (no frequency input; per-sample derivatives);
* the Paderborn waveform classifier is **phase sensitive** (fixed sample indices; sine rule needs
  |b[0]| < 0.1 max b), so a rotated period changes the one-hot features;
* test fixtures: ``tests/test_files/unit_test_data_ploss_at_450kWpm3.csv`` (90 measured rows, columns
  ``B_t_0..B_t_1023`` [T], ``freq`` [Hz], ``temp`` [degC], ``ploss`` [W/m^3], ``material``; one zero-mean
  period per row; sha256 ``b728a796...5047``) and ``unit_test_data_sequence_3C90.csv`` (6 rows B/H/temp,
  multi-period, DC offsets; sha256 ``0f9a97a3...5cb7``).

Discovery looks at an installed ``magnethub`` package through ``importlib.util.find_spec`` only (the
package is **not imported** unless :meth:`MagnetHubAdapter.run_model` is called), at a checkout
``root`` and - with ``allow_download=True`` - at the pinned raw fixture URLs (MIT permits caching).
:meth:`MagnetHubAdapter.import_test_fixture` is a genuine measured-data import example.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from aipe_magnetics.common.errors import DataIntegrityError, DataNotFoundError, InvalidInputError, UnsupportedOperationError
from aipe_magnetics.common.jsonio import sha256_file
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.data.hdf5 import WaveformMeta, WaveformStore
from aipe_magnetics.data.importers.base import IMPORT_EXTENSION, ImportResult, is_file_uri, source_artifact
from aipe_magnetics.data.records import MagneticRecord
from aipe_magnetics.data.sources.base import (
    SOURCES_EXTENSION,
    ExtractedData,
    SourceAdapter,
    SourceDescriptor,
    SourceItem,
    cross_validate,
    dataset_token,
    make_evidence,
    range_problems,
    record_envelope,
    slug,
    source_uri,
    sources_extension,
    write_records,
)
from aipe_magnetics.data.sources.cache import LocalCache
from aipe_magnetics.models.datadriven.magnethub_backend import (
    MAGNETHUB_IMPORT_NAME,
    MAGNETHUB_INSTALL_HINT,
    MAGNETHUB_REFERENCE_COMMIT,
    MAGNETHUB_URL,
    MagnetHubBackend,
    magnethub_package,
)

SOURCE_ID = "paderborn_magnethub"
COMMIT = MAGNETHUB_REFERENCE_COMMIT
COMMIT_DATE = "2026-06-20T11:22:26+02:00"
LICENSE = "MIT"
RAW_URL_TEMPLATE = "https://raw.githubusercontent.com/upb-lea/mag-net-hub/{commit}/{path}"
LOSS_MATERIALS: tuple[str, ...] = ("3C90", "3C92", "3C94", "3C95", "3E6", "3F4", "77", "78", "79", "ML95S", "N27", "N30", "N49", "N87", "T37")
SEQUENCE_MATERIALS: tuple[str, ...] = ("3C90", "3C92", "3C94", "3C95", "3E6", "3F4", "77", "78", "FEC007", "FEC014", "N27", "N30", "N49", "N87", "T37")
TEAMS: dict[str, dict[str, Any]] = {
    "paderborn": {"samples_per_period": 1024, "model": "HARDCORE residual dilated CNN (TorchScript, torch.jit)", "paper": "arXiv:2401.11488",
                  "loss_formula": "ploss = f * corr * |sum_t B_t (H_{t-1} - H_{t+1})| (abs of the loop integral; corr in [0.4, 0.6])"},
    "sydney": {"samples_per_period": 128, "model": "MMINN stop-operator + eddy RNN (PyTorch state dict)", "paper": "moetomg/magnet-engine",
               "loss_formula": "Pv = f * trapz(H, B) over 128 points (no abs; sign follows the traversal direction)"},
}
SEQUENCE_MODEL_SAMPLE_RATE_HZ = 16e6
INPUT_UNITS = {"flux_density": "T", "frequency": "Hz", "temperature": "degC"}
OUTPUT_UNITS = {"loss_density": "W/m^3", "field": "A/m"}
FIXTURES: dict[str, dict[str, Any]] = {
    "ploss": {"path": "tests/test_files/unit_test_data_ploss_at_450kWpm3.csv", "sha256": "b728a796ebff32dce1e2026aea6c59d2292b467b4790557767e9b9de9b575047",
              "size_bytes": 1524321, "rows": 90, "columns": 1028, "samples_per_period": 1024,
              "column_units": {"B_t_<k>": "T", "freq": "Hz", "temp": "degC", "ploss": "W/m^3", "material": "label"},
              "description": "90 measured zero-mean single-period flux-density waveforms with operating point and measured loss density "
                             "(all rows have ploss within 450.0-451.0 kW/m^3); 15 materials; temp in {25, 50, 70, 90} degC"},
    "sequence_3C90": {"path": "tests/test_files/unit_test_data_sequence_3C90.csv", "sha256": "0f9a97a3d26f9ae66282112d58cfe9b2f17d659861200a0af5982c4c6e5dc3b7",
                      "size_bytes": 243632, "rows": 6, "columns": 2049, "column_units": {"B_t_<k>": "T", "H_t_<k>": "A/m", "temp": "degC"},
                      "description": "6 measured 3C90 B/H trajectories of 1024 samples (multi-period, with DC offsets; sample rate not stored)"},
}
FIXTURE_SAMPLES = 1024

DESCRIPTOR = SourceDescriptor(
    source_id=SOURCE_ID, title="Paderborn MagNet-Hub: certified MagNet Challenge models (upb-lea/mag-net-hub)", repository_url=MAGNETHUB_URL,
    pinned_commit=COMMIT, pinned_commit_date=COMMIT_DATE, license=LICENSE, redistribution_permitted=True,
    citation="upb-lea/mag-net-hub (MIT, Copyright (c) 2024 Paderborn University - LEA), commit 1ed635d; Paderborn HARDCORE model: "
             "N. Foerster et al., arXiv:2401.11488; Paderborn sequence model: upb-lea/RHINO-MAG; Sydney MMINN model: moetomg/magnet-engine "
             "(University of Sydney)",
    publisher="Paderborn University, Department of Power Electronics and Electrical Drives (LEA)",
    data_locations=[f"{MAGNETHUB_URL}/tree/{COMMIT}/src_py/magnethub/models (20 Paderborn TorchScript .pt, 15 Sydney .pt, 15 .eqx sequence models, ~1.1 MB)",
                    RAW_URL_TEMPLATE.format(commit=COMMIT, path=FIXTURES["ploss"]["path"]),
                    RAW_URL_TEMPLATE.format(commit=COMMIT, path=FIXTURES["sequence_3C90"]["path"]),
                    "PyPI 'mag-net-hub' (requires Python >= 3.13, torch, pandas, jax, equinox)"],
    notes=["Loss models: 15 materials, teams paderborn (1024 samples/period) and sydney (128); inputs B [T], f [Hz], T [degC]; outputs p [W/m^3], h [A/m].",
           "Sequence models (paderborn, GRU8 .eqx): no frequency input; assume the 16 MHz training sample rate; T_max 70 degC in every normaliser.",
           "Paderborn waveform classifier is phase sensitive (fixed sample indices); keep the upstream zero-crossing start convention.",
           "Fixture origin: measured targets only, measurement provenance not stated upstream (presumably MagNet Challenge 2023 / Princeton data).",
           "Model weights are never vendored by AIPE Magnetics; inference needs the optional magnethub package (models.datadriven.magnethub_backend)."])


def installed_package_location() -> Path | None:
    """Directory of an installed ``magnethub`` package located with ``find_spec`` (never imported)."""
    try:
        spec = importlib.util.find_spec(MAGNETHUB_IMPORT_NAME)
    except (ImportError, ValueError):  # pragma: no cover - broken installations
        return None
    if spec is None:
        return None
    if spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            if (Path(location) / "models").is_dir():
                return Path(location)
        return Path(next(iter(spec.submodule_search_locations)))
    return Path(spec.origin).parent if spec.origin else None


def _fixture_item(key: str, path: Path | None, *, available: bool, kind: str = "file") -> SourceItem:
    """``SourceItem.sha256`` is the *computed* digest of a local file (never the pinned value assumed); the pinned
    digest is ``extra['expected_sha256']`` and is only used as ``sha256`` for remote items that do not exist locally yet."""
    info = FIXTURES[key]
    local = path is not None and path.is_file()
    return SourceItem(source_id=SOURCE_ID, item_id=info["path"], kind=kind, path=str(path) if path else None,  # type: ignore[arg-type]
                      url=RAW_URL_TEMPLATE.format(commit=COMMIT, path=info["path"]), size_bytes=path.stat().st_size if local else info["size_bytes"],
                      sha256=sha256_file(path) if local else info["sha256"], available=available, license=LICENSE, description=info["description"],
                      extra={"fixture": key, "expected_sha256": info["sha256"], "expected_rows": info["rows"], "expected_columns": info["columns"],
                             "column_units": info["column_units"]})


class MagnetHubAdapter(SourceAdapter):
    """Discovery of MagNet-Hub models and fixtures, fixture import, and delegated inference."""

    descriptor = DESCRIPTOR
    importer_id = "aipe.magnetics.sources.paderborn_magnethub"
    _license_policy = {"read": True, "cache": True, "bundle": True, "redistribute": True}

    def __init__(self, cache: LocalCache | None = None):
        self.cache = cache or LocalCache()

    # -- discovery ------------------------------------------------------------------------------------
    def discover(self, root: str | Path | None = None, allow_download: bool = False, **options: Any) -> list[SourceItem]:
        items: list[SourceItem] = []
        found: set[str] = set()
        if root is not None:
            root = Path(root)
            if not root.is_dir():
                raise DataNotFoundError(f"checkout directory not found: {root}")
            for key, info in FIXTURES.items():
                path = root / info["path"]
                if path.is_file():
                    items.append(_fixture_item(key, path, available=True))
                    found.add(key)
            models = root / "src_py" / "magnethub" / "models"
            if models.is_dir():
                for team_dir in sorted(p for p in models.iterdir() if p.is_dir()):
                    files = sorted(p for p in team_dir.iterdir() if p.suffix in (".pt", ".eqx"))
                    items.append(SourceItem(source_id=SOURCE_ID, item_id=f"src_py/magnethub/models/{team_dir.name}", kind="directory", path=str(team_dir),
                                            size_bytes=sum(p.stat().st_size for p in files), available=True, license=LICENSE,
                                            description=f"{len(files)} model files of team directory {team_dir.name} (opaque weights; not read by AIPE)",
                                            extra={"files": [p.name for p in files], "team": team_dir.name}))
        else:
            location = installed_package_location()
            if location is not None:
                models = location / "models"
                files = sorted(p for p in models.rglob("*") if p.suffix in (".pt", ".eqx")) if models.is_dir() else []
                items.append(SourceItem(source_id=SOURCE_ID, item_id="installed:magnethub", kind="package_data", path=str(location),
                                        size_bytes=sum(p.stat().st_size for p in files), available=True, license=LICENSE,
                                        description="installed magnethub package (located with importlib.util.find_spec; not imported)",
                                        extra={"model_files": [p.relative_to(location).as_posix() for p in files], "teams": sorted(TEAMS),
                                               "loss_materials": list(LOSS_MATERIALS), "sequence_materials": list(SEQUENCE_MATERIALS)}))
        for key, info in FIXTURES.items():
            if key in found:
                continue
            url = RAW_URL_TEMPLATE.format(commit=COMMIT, path=info["path"])
            cached = self.cache.lookup(url)
            if cached is not None and self.cache.path_of(cached).is_file():
                item = _fixture_item(key, self.cache.path_of(cached), available=True)
                item.extra["cached"] = True
                items.append(item)
            else:
                item = _fixture_item(key, None, available=False, kind="remote")
                item.description = (info["description"] + f"; download requires allow_download=True ({'authorised' if allow_download else 'not authorised'})")
                items.append(item)
        return items

    def fetch(self, item: SourceItem, *, allow_download: bool = False, retrieved_at: str | None = None, timeout: float = 120.0) -> Path:
        if item.path and Path(item.path).is_file():
            return Path(item.path)
        if not item.url:
            raise DataNotFoundError(f"item {item.item_id} has neither a local path nor a URL")
        self.check_license("cache")
        return self.cache.fetch(item.url, item.extra.get("expected_sha256") or item.sha256, allow_download, filename=Path(item.item_id).name,
                                commit=COMMIT, license=LICENSE, source_id=SOURCE_ID, retrieved_at=retrieved_at, timeout=timeout)

    # -- inspection -----------------------------------------------------------------------------------
    def inspect(self, item: SourceItem, **options: Any) -> dict[str, Any]:
        if item.kind in ("directory", "package_data"):
            return {"item_id": item.item_id, "path": item.path, "files": item.extra.get("files") or item.extra.get("model_files"),
                    "teams": TEAMS, "loss_materials": list(LOSS_MATERIALS), "sequence_materials": list(SEQUENCE_MATERIALS),
                    "note": "model files are opaque weights; AIPE reads only their names and sizes"}
        path = self.fetch(item, allow_download=False)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            rows = 0
            materials: dict[str, int] = {}
            widths: set[int] = set()
            material_index = header.index("material") if "material" in header else None
            for cells in reader:
                if not cells:
                    continue
                rows += 1
                widths.add(len(cells))
                if material_index is not None:
                    materials[cells[material_index]] = materials.get(cells[material_index], 0) + 1
        b_columns = sum(1 for h in header if h.startswith("B_t_"))
        h_columns = sum(1 for h in header if h.startswith("H_t_"))
        info = {"item_id": item.item_id, "path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
                "expected_sha256": item.extra.get("expected_sha256"), "rows": rows, "columns": len(header), "rectangular": widths <= {len(header)},
                "b_samples": b_columns, "h_samples": h_columns, "scalar_columns": [h for h in header if not h.startswith(("B_t_", "H_t_"))],
                "materials": dict(sorted(materials.items())), "column_units": item.extra.get("column_units"), "license": LICENSE}
        info["matches_pinned_commit"] = info["sha256"] == info["expected_sha256"] if info["expected_sha256"] else None
        return info

    # -- extraction -----------------------------------------------------------------------------------
    def extract(self, item: SourceItem, *, materials: Sequence[str] | None = None, allow_download: bool = False, **filters: Any) -> ExtractedData:
        if filters:
            raise InvalidInputError(f"unknown filters {sorted(filters)}; supported: materials, allow_download")
        if item.extra.get("fixture") != "ploss":
            raise InvalidInputError("extract() supports the loss fixture unit_test_data_ploss_at_450kWpm3.csv (B waveforms + operating point + "
                                    "measured loss); model directories hold opaque weights")
        path = self.fetch(item, allow_download=allow_download)
        digest = sha256_file(path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            expected = [f"B_t_{k}" for k in range(FIXTURE_SAMPLES)] + ["freq", "temp", "ploss", "material"]
            if header != expected:
                raise DataIntegrityError(f"{path.name}: header differs from the audited layout (B_t_0..B_t_1023, freq, temp, ploss, material)")
            rows = [cells for cells in reader if cells]
        if not rows:
            raise DataIntegrityError(f"{path.name}: no data rows")
        if any(len(r) != len(header) for r in rows):
            raise DataIntegrityError(f"{path.name}: ragged rows")
        labels = [r[-1] for r in rows]
        wanted = list(materials) if materials else sorted(set(labels))
        unknown = sorted(set(wanted) - set(labels))
        if unknown:
            raise DataNotFoundError(f"materials {unknown} not present in the fixture; present: {sorted(set(labels))}")
        keep = [i for i, lab in enumerate(labels) if lab in wanted]
        try:
            block = np.array([[float(c) for c in rows[i][:-1]] for i in keep], dtype=np.float64)
        except ValueError as error:
            raise DataIntegrityError(f"{path.name}: non-numeric cell: {error}") from None
        b = block[:, :FIXTURE_SAMPLES]
        freq, temp, ploss = block[:, FIXTURE_SAMPLES], block[:, FIXTURE_SAMPLES + 1], block[:, FIXTURE_SAMPLES + 2]
        categories = sorted(set(labels[i] for i in keep))
        codes = np.array([categories.index(labels[i]) for i in keep], dtype=np.int64)
        return ExtractedData(
            source_id=SOURCE_ID, item=item, kind="sequences", material=wanted[0] if len(wanted) == 1 else None,
            arrays={"flux_density": (b, "T")}, conditions={"frequency": (freq, "Hz"), "temperature": (temp, "degC"), "loss_density": (ploss, "W/m^3")},
            categorical={"material": (codes, categories)}, timebase="uniform_per_period",
            source_files={path.name: digest}, source_paths={path.name: str(path)},
            column_sources={"flux_density": "B_t_0..B_t_1023", "frequency": "freq", "temperature": "temp", "loss_density": "ploss", "material": "material"},
            metadata={"n_sequences": len(keep), "samples_per_period": FIXTURE_SAMPLES, "materials": categories,
                      "rows_per_material": {m: int((codes == i).sum()) for i, m in enumerate(categories)}, "expected_sha256": item.extra.get("expected_sha256"),
                      "materials_by_row": [labels[i] for i in keep], "ploss_range_W_per_m3": [float(ploss.min()), float(ploss.max())]},
            filters={"materials": wanted} if materials else {},
            notes=["one zero-mean steady-state period per row (1024 samples); measured loss density per row; measurement origin not stated upstream"])

    # -- validation -----------------------------------------------------------------------------------
    def validate(self, extracted: ExtractedData) -> list[str]:
        problems = super().validate(extracted)
        if extracted.kind != "sequences":
            return problems
        b = extracted.arrays.get("flux_density")
        if b is not None:
            problems += range_problems("flux_density peak", np.nanmax(np.abs(b[0]), axis=1), b[1], minimum=0.0, maximum=2.0, strict_min=True)
            means = np.abs(np.nanmean(b[0], axis=1))
            if np.any(means > 1e-3 * np.nanmax(np.abs(b[0]), axis=1)):
                problems.append("flux_density: rows are not zero-mean periods (|mean B| > 0.1 % of peak)")
        for name, lo, hi in (("frequency", 1e3, 1e7), ("loss_density", 0.0, 1e8)):
            if name in extracted.conditions:
                problems += range_problems(name, extracted.conditions[name][0], extracted.conditions[name][1], minimum=lo, maximum=hi, strict_min=True)
        if "temperature" in extracted.conditions:
            problems += range_problems("temperature", extracted.conditions["temperature"][0], extracted.conditions["temperature"][1], minimum=-50.0, maximum=200.0)
        expected = extracted.metadata.get("expected_sha256")
        for name, digest in extracted.source_files.items():
            if expected and digest != expected:
                problems.append(f"{name}: sha256 differs from the pinned upstream fixture ({expected[:12]}...)")
        return problems

    # -- records --------------------------------------------------------------------------------------
    def to_records(self, extracted: ExtractedData, *, destination_dir: str | Path, dataset_id: str | None = None,
                   material_refs: Mapping[str, str] | None = None, context: Sequence[MagneticRecord] = (), retrieved_at: str | None = None,
                   allow_unpinned: bool = False, **options: Any) -> ImportResult:
        """One waveform file + ``waveform_dataset`` record per material, one ``dataset`` record, identity material records.

        ``allow_unpinned``: a fixture that is not byte-identical to the file at the pinned commit is refused
        (:class:`DataIntegrityError`) unless ``True``; then every record is flagged
        ``source_differs_from_pinned_commit`` and versioned by the file's SHA-256 instead of the commit.
        """
        if options:
            raise InvalidInputError(f"unknown options {sorted(options)}")
        if extracted.kind != "sequences" or extracted.source_id != SOURCE_ID:
            raise InvalidInputError("to_records expects the loss fixture extracted by MagnetHubAdapter")
        problems = self.validate(extracted)
        blocking = [p for p in problems if "differs from the pinned upstream fixture" not in p]
        if blocking:
            raise DataIntegrityError("fixture failed validation: " + "; ".join(blocking), details={"problems": problems})
        warnings = [p for p in problems if p not in blocking]
        pin = self._unpinned_policy(extracted, allow_unpinned=allow_unpinned)
        destination_dir = Path(destination_dir)
        source_name, source_digest = next(iter(extracted.source_files.items()))
        source_path = Path(extracted.source_paths[source_name])
        raw_uri = source_uri(source_path, destination_dir, pinned_url=extracted.item.url if pin.pinned else None)
        if is_file_uri(raw_uri):
            warnings.append(f"{source_name}: no relative path from {destination_dir} to the source file; the raw artifact uses the absolute URI {raw_uri}")
        file_version = pin.version or COMMIT
        origin = f"the pinned fixture (commit {COMMIT[:12]})" if pin.pinned else f"an UNPINNED copy of the fixture (sha256 {source_digest})"
        base = dataset_token(dataset_id) if dataset_id else "paderborn-magnethub-ploss-fixture"
        dataset_id = dataset_id or f"dataset.{base}"
        codes, categories = extracted.categorical["material"]
        refs = dict(material_refs or {})
        repo_evidence = self._repository_evidence(description="MagNet-Hub repository at the pinned commit (MIT, Copyright (c) 2024 Paderborn University - LEA); "
                                                              "test fixtures and model files under src_py/magnethub and tests/test_files.")
        file_evidence = make_evidence(
            evidence_id="evidence.paderborn-magnethub-ploss-fixture", kind="source",
            reference=f"upb-lea/mag-net-hub {FIXTURES['ploss']['path']}, commit {COMMIT}",
            description=("Measured single-period B(t) waveforms with frequency, temperature and measured loss density used upstream as accuracy "
                         "reference for the certified loss models; the measurement campaign is not identified in the repository (presumably "
                         "MagNet Challenge 2023 data); units per upstream LossModel docstring: B T, freq Hz, temp degC, ploss W/m^3."),
            created_at=COMMIT_DATE, license=LICENSE, locator="columns B_t_0..B_t_1023, freq, temp, ploss, material", url=extracted.item.url,
            version=file_version, sha256=source_digest, accessed_at=retrieved_at)
        evidence = [repo_evidence, file_evidence]
        normalization = ("B [T] stored as given; frequency Hz as given; temperature degC -> K; loss density W/m^3 as given; one waveform file per "
                         "material; original units kept as HDF5 attributes")
        provenance = self._provenance(f"imported {extracted.n_rows} fixture rows from {source_name} split by material", inputs=[e.id for e in evidence],
                                      normalization=normalization, retrieved_at=retrieved_at, version=pin.version)
        raw_artifact = source_artifact(source_path, media_type="text/csv", sha256=source_digest, role="raw", uri=raw_uri,
                                       description="original upstream fixture as read (unmodified)")
        pin_quality = {"status": "normalized", "flags": list(pin.flags), "notes": list(warnings)} if (pin.flags or warnings) else None
        base_extension = self._import_extension(source_name=source_name, source_sha256=source_digest, normalization=normalization,
                                                extra={"column_units": FIXTURES["ploss"]["column_units"], "samples_per_period": FIXTURE_SAMPLES,
                                                       "rows": extracted.n_rows, "filters": dict(extracted.filters)})
        records: list[MagneticRecord] = []
        artifacts = [raw_artifact]
        checksums = {source_name: source_digest}
        written_files: list[str] = []
        identity_ids: list[str] = []
        store = WaveformStore()
        try:
            for code, material in enumerate(categories):
                token = slug(material)
                rows = np.flatnonzero(codes == code)
                material_ref = refs.get(material)
                if material_ref is None:
                    identity = self._material_identity_record(material=material, evidence=[repo_evidence],
                                                              description=f"Identity record for MagNet-Hub material label {material!r} (one of the 15 loss-model "
                                                                          "materials); no datasheet parameters.")
                    records.append(identity)
                    identity_ids.append(identity.id)
                    material_ref = identity.id
                h5_path = destination_dir / f"{base}-{token}.h5"
                if h5_path.exists():
                    raise FileExistsError(f"{h5_path} exists; adapters never overwrite")
                meta = WaveformMeta(source_id=SOURCE_ID, dataset_version=file_version, license=LICENSE, timebase="uniform_per_period", material_ref=material_ref,
                                    source_dataset_ref=dataset_id,
                                    description=f"MagNet-Hub loss fixture rows of {material}: one zero-mean period per row (1024 samples), measured loss density")
                arrays = {"flux_density": (extracted.arrays["flux_density"][0][rows], extracted.arrays["flux_density"][1])}
                conditions = {name: (values[rows], unit) for name, (values, unit) in extracted.conditions.items()}
                properties = store.write(h5_path, arrays, conditions, meta=meta)
                written_files.append(str(h5_path))
                h5_artifact = store.artifact(h5_path, uri=h5_path.name, description="AIPE waveform file (layout aipe.magnetics.waveforms/0.2)")
                artifacts.append(h5_artifact)
                checksums[h5_path.name] = h5_artifact.sha256 or ""
                extensions = {IMPORT_EXTENSION: dict(base_extension, material=material, rows=int(rows.size)),
                              SOURCES_EXTENSION: sources_extension(self.descriptor, extracted.item, extracted, waveform_file=h5_path.name, material=material,
                                                                   fixture_rows=[int(i) for i in rows.tolist()], **pin.extension())}
                records.append(record_envelope(
                    record_id=f"waveform-dataset.{base}-{token}", kind="waveform_dataset",
                    name=f"MagNet-Hub loss fixture waveforms - {material}", properties=properties.model_dump(mode="json", exclude_none=True),
                    evidence=evidence, license=LICENSE, provenance=provenance, artifacts=[h5_artifact, raw_artifact], dataset_ref=dataset_id,
                    tags=["external-source", SOURCE_ID, "fixture"], extensions=extensions,
                    quality={"status": "normalized", "flags": list(pin.flags), "notes": list(extracted.notes) + list(warnings)},
                    description=f"{rows.size} measured single-period B(t) rows of {material} with frequency [Hz], temperature [K] and loss density [W/m^3]."))
            extensions = {IMPORT_EXTENSION: base_extension, SOURCES_EXTENSION: sources_extension(self.descriptor, extracted.item, extracted, **pin.extension())}
            dataset_record = self._dataset_record(
                dataset_id=dataset_id, name="MagNet-Hub loss-model test fixture (measured B waveforms and losses)", evidence=evidence, checksums=checksums,
                materials=list(categories), record_count=extracted.n_rows, provenance=provenance, artifacts=artifacts, extensions=extensions,
                retrieved_at=retrieved_at, title="upb-lea/mag-net-hub tests/test_files/unit_test_data_ploss_at_450kWpm3.csv",
                url=extracted.item.url, version=pin.version, quality=pin_quality,
                description=(f"{extracted.n_rows} measured rows ({len(categories)} materials) from {origin}; one waveform file per material; "
                             "licence MIT; measurement campaign not identified upstream."))
            records.insert(0, dataset_record)
            cross_validate(records, context)
            written = written_files + write_records(records, destination_dir, reusable_ids=identity_ids)
        except Exception:
            for path in written_files:
                Path(path).unlink(missing_ok=True)
            raise
        return self._import_result(records=records, artifacts=artifacts, provenance=provenance, warnings=warnings, written=written, source_sha256=source_digest)

    def import_test_fixture(self, checkout_root: str | Path, destination_dir: str | Path, *, materials: Sequence[str] | None = None,
                            dataset_id: str | None = None, material_refs: Mapping[str, str] | None = None, context: Sequence[MagneticRecord] = (),
                            retrieved_at: str | None = None, allow_unpinned: bool = False) -> ImportResult:
        """Import ``tests/test_files/unit_test_data_ploss_at_450kWpm3.csv`` of a checkout (genuine measured data, MIT)."""
        items = [i for i in self.discover(checkout_root) if i.extra.get("fixture") == "ploss" and i.available]
        if not items:
            raise DataNotFoundError(f"{FIXTURES['ploss']['path']} not found under {checkout_root}")
        extracted = self.extract(items[0], materials=materials)
        return self.to_records(extracted, destination_dir=destination_dir, dataset_id=dataset_id, material_refs=material_refs, context=context,
                               retrieved_at=retrieved_at, allow_unpinned=allow_unpinned)

    # -- inference ------------------------------------------------------------------------------------
    def run_model(self, material: str, team: str, b_sequence: Any, frequency: Any, temperature: Any) -> dict[str, Any]:
        """Delegate to ``aipe_magnetics.models.datadriven.magnethub_backend`` (optional ``magnethub`` package).

        ``b_sequence``: (L,) or (N, L) flux density in **T** with exactly the team's samples per period;
        ``frequency`` / ``temperature``: :class:`Quantity` or ``(value, unit)`` pairs (bare numbers are
        rejected - no unit inference). Returns ``{"loss_density": (N,) W/m^3, "field": (N, L) A/m, ...}``.
        Raises :class:`UnsupportedOperationError` when ``magnethub`` is not installed.
        """
        if team not in TEAMS:
            raise InvalidInputError(f"team must be one of {sorted(TEAMS)}, got {team!r}")
        f_q = Quantity.parse(frequency, name="frequency")
        t_q = Quantity.parse(temperature, name="temperature")
        b = np.asarray(b_sequence, dtype=np.float64)
        if b.ndim == 1:
            b = b[None, :]
        n = b.shape[0]
        package = magnethub_package(material, team, sequence_length=None)
        backend = MagnetHubBackend()
        try:
            backend.load(package)
        except UnsupportedOperationError as error:
            raise UnsupportedOperationError(f"run_model needs the optional package {MAGNETHUB_IMPORT_NAME!r}: {error.message}; {MAGNETHUB_INSTALL_HINT}",
                                            details=dict(error.details, material=material, team=team)) from None
        outputs = backend.run(package, {"flux_density": b, "frequency": np.full(n, f_q.magnitude("Hz")), "temperature": np.full(n, t_q.magnitude("K"))})
        return {"loss_density": outputs["loss_density"], "field": outputs["field"], "units": dict(OUTPUT_UNITS), "team": team, "material": material,
                "samples_per_period": backend.expected_sequence_length, "attribution": [s.model_dump(mode="json", exclude_none=True) for s in self.attribution()]}


__all__ = ["SOURCE_ID", "DESCRIPTOR", "COMMIT", "LICENSE", "LOSS_MATERIALS", "SEQUENCE_MATERIALS", "TEAMS", "SEQUENCE_MODEL_SAMPLE_RATE_HZ", "INPUT_UNITS",
           "OUTPUT_UNITS", "FIXTURES", "FIXTURE_SAMPLES", "RAW_URL_TEMPLATE", "MagnetHubAdapter", "installed_package_location"]
