"""Import user-supplied waveform sequences (HDF5 / npz / CSV) into the AIPE waveform layout.

Upstream waveform sets come in many containers (MagNet-style HDF5 files with arbitrary dataset
names, ``.npz`` bundles, one CSV per quantity with one sequence per row). This importer reads the
arrays the caller names **explicitly** - ``arrays = {"flux_density": ("B_seq", "mT"), ...}`` and
``conditions = {"frequency": ("f", "kHz"), ...}`` where the key is the dataset path (HDF5), the
array name (npz) or the file name (CSV directory) - converts them to SI through
:class:`aipe_magnetics.data.hdf5.WaveformStore` (original units retained, NaN = missing, strict
shape checks, per-array SHA-256) and produces a ``waveform_dataset`` record whose artifacts carry
the SHA-256 of the written file and of the original source bytes.

Only the *container format* is taken from the file suffix (``.h5``/``.hdf5``, ``.npz``, ``.csv`` or
a directory of CSV files); ``source_format`` overrides it. Units, quantities and sequence/condition
roles are never inferred.

Order of operations (nothing is written when any step fails):

1. argument, evidence and destination checks (an existing ``destination_record`` or, without
   ``overwrite_h5``, an existing ``destination_h5`` is refused before anything is read);
2. the mapped arrays are read and **prepared** (:meth:`WaveformStore.prepare`: units, shapes,
   missing values, timebase) without touching the file system;
3. the complete record is built with a provisional waveform artifact (checksum unknown until the
   file exists) and validated - JSON Schema, semantics, physical rules and catalogue cross-references;
4. only then the HDF5 file is written, the artifact checksum is filled in, the record is re-validated
   and (optionally) written. Should anything fail after step 4 started, the waveform file produced by
   this call is removed again so no orphan is left behind.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from aipe_magnetics.common.errors import InvalidInputError, UnsupportedOperationError
from aipe_magnetics.common.jsonio import canonical_sha256, sha256_file
from aipe_magnetics.data.catalog import Catalog
from aipe_magnetics.data.hdf5 import HDF5_MEDIA_TYPE, WaveformMeta, WaveformStore, h5py_available
from aipe_magnetics.data.importers.base import (
    IMPORT_EXTENSION,
    ImportResult,
    build_record,
    check_against_catalogue,
    coerce_evidence,
    import_extension,
    is_file_uri,
    relative_uri,
    source_artifact,
    tool_attribution,
    write_record,
)
from aipe_magnetics.data.records import SCHEMA_VERSION, Artifact, Evidence, MagneticRecord, Provenance

IMPORTER_ID = "aipe.magnetics.import.hdf5_waveforms"
SourceFormat = str  # "hdf5" | "npz" | "csv"
_SUFFIXES = {".h5": "hdf5", ".hdf5": "hdf5", ".npz": "npz", ".csv": "csv"}
_MEDIA = {"hdf5": HDF5_MEDIA_TYPE, "npz": "application/octet-stream", "csv": "text/csv"}
_H5_DESCRIPTION = "AIPE waveform file (layout aipe.magnetics.waveforms/0.2)"


def _detect_format(source: Path, source_format: str | None) -> str:
    if source_format is not None:
        if source_format not in _MEDIA:
            raise InvalidInputError(f"source_format must be one of {sorted(_MEDIA)}, got {source_format!r}")
        return source_format
    if source.is_dir():
        return "csv"
    fmt = _SUFFIXES.get(source.suffix.lower())
    if fmt is None:
        raise InvalidInputError(f"cannot tell the container format of {source.name!r}; pass source_format=",
                                details={"path": str(source)})
    return fmt


def _validate_mapping(label: str, mapping: Mapping[str, Any] | None) -> dict[str, tuple[str, str]]:
    if mapping is None:
        return {}
    if not isinstance(mapping, Mapping):
        raise InvalidInputError(f"{label} must be a mapping quantity -> (source key, unit)")
    out: dict[str, tuple[str, str]] = {}
    for quantity, spec in mapping.items():
        if not isinstance(spec, (tuple, list)) or len(spec) != 2 or not all(isinstance(s, str) for s in spec):
            raise InvalidInputError(f"{label}[{quantity!r}] must be a (source key, unit) pair of strings, got {spec!r}")
        out[str(quantity)] = (spec[0], spec[1])
    return out


class _Source:
    """Reads named arrays from one container; records which files were read and their digests."""

    def __init__(self, path: Path, fmt: str, *, delimiter: str):
        self.path, self.fmt, self.delimiter = path, fmt, delimiter
        self.files_read: dict[str, str] = {}
        self._npz: Any = None
        self._h5: Any = None

    def __enter__(self) -> "_Source":
        if self.fmt == "npz":
            if not self.path.is_file():
                raise InvalidInputError(f"source file not found: {self.path}")
            self._npz = np.load(self.path, allow_pickle=False)
            self.files_read[self.path.name] = sha256_file(self.path)
        elif self.fmt == "hdf5":
            if not h5py_available():
                raise UnsupportedOperationError("reading HDF5 sources needs h5py: install aipe-magnetics[hdf5]")
            import h5py  # type: ignore
            if not self.path.is_file():
                raise InvalidInputError(f"source file not found: {self.path}")
            self._h5 = h5py.File(self.path, "r")
            self.files_read[self.path.name] = sha256_file(self.path)
        elif self.fmt == "csv":
            if not (self.path.is_dir() or self.path.is_file()):
                raise InvalidInputError(f"source not found: {self.path}")
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._npz is not None:
            self._npz.close()
        if self._h5 is not None:
            self._h5.close()

    def available(self) -> list[str]:
        if self.fmt == "npz":
            return sorted(self._npz.files)
        if self.fmt == "hdf5":
            names: list[str] = []
            self._h5.visititems(lambda name, obj: names.append(name) if hasattr(obj, "shape") else None)
            return sorted(names)
        if self.path.is_dir():
            return sorted(p.name for p in self.path.glob("*.csv"))
        return [self.path.name]

    def array(self, key: str) -> np.ndarray:
        if self.fmt == "npz":
            if key not in self._npz.files:
                raise InvalidInputError(f"array {key!r} not found in {self.path.name}; available: {self.available()}")
            return np.asarray(self._npz[key])
        if self.fmt == "hdf5":
            if key not in self._h5 or not hasattr(self._h5[key], "shape"):
                raise InvalidInputError(f"dataset {key!r} not found in {self.path.name}; available: {self.available()}")
            return np.asarray(self._h5[key][()])
        file = self.path / key if self.path.is_dir() else self.path
        if self.path.is_file() and key != self.path.name:
            raise InvalidInputError(f"single CSV source: key must be the file name {self.path.name!r}, got {key!r}")
        if not file.is_file():
            raise InvalidInputError(f"CSV file {key!r} not found in {self.path}; available: {self.available()}")
        self.files_read[file.name] = sha256_file(file)
        try:
            return np.loadtxt(file, delimiter=self.delimiter, ndmin=1, dtype=np.float64)
        except ValueError as error:
            raise InvalidInputError(f"{file}: cannot parse numeric CSV: {error}") from None


def import_waveform_sequences(source: str | Path, destination_h5: str | Path, *, arrays: Mapping[str, Any],
                              meta: WaveformMeta, record_id: str, name: str, evidence: Evidence | Mapping[str, Any],
                              conditions: Mapping[str, Any] | None = None, destination_record: str | Path | None = None,
                              catalogue: Catalog | Iterable[Any] | None = None, synthetic: bool = False,
                              dataset_ref: str | None = None, description: str | None = None, tags: Iterable[str] = (),
                              source_format: str | None = None, delimiter: str = ",", overwrite_h5: bool = False,
                              store: WaveformStore | None = None) -> ImportResult:
    """Read the mapped arrays from ``source``, write the AIPE waveform file and build the describing record.

    See the module docstring for the order of operations; nothing is written unless the record validates.
    """
    source, destination_h5 = Path(source), Path(destination_h5)
    fmt = _detect_format(source, source_format)
    array_map = _validate_mapping("arrays", arrays)
    condition_map = _validate_mapping("conditions", conditions)
    if not array_map:
        raise InvalidInputError("arrays mapping must name at least one sequence quantity")
    item = coerce_evidence(evidence)
    if not isinstance(meta, WaveformMeta):
        raise InvalidInputError("meta must be a WaveformMeta instance")
    if store is not None and not isinstance(store, WaveformStore):
        raise InvalidInputError("store must be a WaveformStore instance")
    store = store or WaveformStore()
    record_path = None if destination_record is None else Path(destination_record)
    # 1. destinations are refused up front: nothing is read, converted or written when one is taken
    if record_path is not None and record_path.exists():
        raise FileExistsError(f"{record_path} exists; importers never overwrite")
    if destination_h5.exists() and not overwrite_h5:
        raise FileExistsError(f"{destination_h5} exists; pass overwrite_h5=True to replace it")
    if not h5py_available():
        raise UnsupportedOperationError("writing the AIPE waveform file needs h5py: install aipe-magnetics[hdf5]",
                                        details={"extra": "hdf5"})
    # 2. read and prepare (validation + SI normalisation, no file-system writes)
    with _Source(source, fmt, delimiter=delimiter) as reader:
        loaded_arrays = {quantity: (reader.array(key), unit) for quantity, (key, unit) in array_map.items()}
        loaded_conditions = {quantity: (reader.array(key), unit) for quantity, (key, unit) in condition_map.items()}
        files_read = dict(reader.files_read)
    warnings: list[str] = []
    for quantity, (values, _unit) in loaded_conditions.items():
        if np.asarray(values).ndim == 2 and np.asarray(values).shape[1] == 1:
            warnings.append(f"condition {quantity!r}: column array flattened to one entry per sequence")
    prepared = store.prepare(loaded_arrays, loaded_conditions, meta=meta)
    properties = prepared.properties
    # 3. the complete record, validated before anything is written
    if len(files_read) == 1:
        source_name, source_digest = next(iter(files_read.items()))
        digest_method = "sha256 of the source file bytes"
    else:
        source_name = source.name
        source_digest = canonical_sha256(dict(sorted(files_read.items())))
        digest_method = "sha256 of the canonical JSON {file name: sha256 of file bytes} of every source file read"
    normalization = ("arrays converted to canonical SI units (" + ", ".join(f"{a.quantity}: {a.original_unit} -> {a.unit}" for a in properties.arrays)
                     + "); original units recorded as HDF5 attributes; missing values written as NaN")
    h5_uri = relative_uri(destination_h5, record_path)
    raw_artifacts: list[Artifact] = []
    for file_name, digest in sorted(files_read.items()):
        file_path = source / file_name if source.is_dir() else source
        raw_artifacts.append(source_artifact(file_path, media_type=_MEDIA[fmt], sha256=digest, role="raw",
                                             uri=relative_uri(file_path, record_path), description="original source as supplied"))
    for uri in [h5_uri, *(a.uri for a in raw_artifacts)]:
        if is_file_uri(uri):
            warnings.append(f"artifact recorded with an absolute file URI ({uri}) because no relative path from the record file exists")

    def record_dict(waveform_artifact: Artifact) -> dict[str, Any]:
        artifacts = [waveform_artifact, *raw_artifacts]
        data: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION, "id": record_id, "kind": "waveform_dataset", "name": name, "synthetic": synthetic,
            "license": meta.license, "properties": properties.model_dump(mode="json", exclude_none=True),
            "evidence_refs": [item.id], "evidence": [item.model_dump(mode="json", exclude_none=True)],
            "provenance": Provenance(activity=f"imported {properties.n_sequences} waveform sequences from {source_name} ({fmt})",
                                     tool=tool_attribution(), inputs=[item.id], dataset_version=meta.dataset_version,
                                     normalization=normalization).model_dump(mode="json", exclude_none=True),
            "artifacts": [a.model_dump(mode="json", exclude_none=True) for a in artifacts],
            "tags": list(tags),
            "extensions": {IMPORT_EXTENSION: import_extension(
                source_name=source_name, source_sha256=source_digest, importer=IMPORTER_ID, normalization=normalization,
                extra={"source_format": fmt, "source_sha256_method": digest_method, "source_files": dict(sorted(files_read.items())),
                       "arrays": {q: {"key": k, "unit": u} for q, (k, u) in array_map.items()},
                       "conditions": {q: {"key": k, "unit": u} for q, (k, u) in condition_map.items()}})},
        }
        if description is not None:
            data["description"] = description
        if dataset_ref is not None:
            data["dataset_ref"] = dataset_ref
        return data

    provisional = Artifact(uri=h5_uri, media_type=HDF5_MEDIA_TYPE, role="waveform_data", description=_H5_DESCRIPTION)
    record: MagneticRecord = build_record(record_dict(provisional))  # schema + semantics + physical, before any write
    check_against_catalogue(record, catalogue)                       # cross-references, before any write
    # 4. write; remove the waveform file again if anything fails afterwards
    try:
        store.write_prepared(destination_h5, prepared, overwrite=overwrite_h5)
        final_artifact = store.artifact(destination_h5, uri=h5_uri, description=_H5_DESCRIPTION)
        record = build_record(record_dict(final_artifact))
        written: list[str] = [str(destination_h5)]
        if record_path is not None:
            written.append(str(write_record(record, record_path)))
    except BaseException:
        if destination_h5.exists():
            destination_h5.unlink()  # produced (or replaced) by this call; leave nothing behind on failure
        raise
    artifacts = [final_artifact, *raw_artifacts]
    return ImportResult(importer=IMPORTER_ID, records=[record], artifacts=artifacts, provenance=record.provenance,  # type: ignore[arg-type]
                        warnings=warnings, written=written, source_sha256=source_digest)


__all__ = ["IMPORTER_ID", "import_waveform_sequences"]
