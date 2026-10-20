"""Shared machinery of the end-to-end workflows: artifact store, state builders, report helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from aipe_magnetics import __version__
from aipe_magnetics.state.model import (
    ArtifactRecord,
    CoreSelection,
    GeometryRecord,
    MaterialSelection,
    QV,
    ResultSet,
    SolverConfiguration,
    WindingDefinition,
    WindingLayer,
)

MEDIA = {".json": "application/json", ".csv": "text/csv", ".md": "text/markdown", ".png": "image/png", ".step": "model/step",
         ".stl": "model/stl", ".brep": "application/octet-stream", ".npz": "application/x-npz"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return {"re": value.real, "im": value.imag}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass
class ArtifactStore:
    """Writes files below ``root`` and records them (relative path, media type, role, SHA-256, size)."""

    root: Path
    records: list[ArtifactRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _register(self, path: Path, role: str, description: str, produced_by: str | None = None, artifact_id: str | None = None) -> str:
        data = path.read_bytes()
        rel = path.relative_to(self.root).as_posix()
        aid = artifact_id or "artifact." + rel.replace("/", ".")
        self.records = [r for r in self.records if r.id != aid]
        self.records.append(ArtifactRecord(id=aid, path=rel, media_type=MEDIA.get(path.suffix, "application/octet-stream"), role=role,
                                           sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data), description=description,
                                           produced_by=produced_by))
        return aid

    def json(self, rel: str, data: Any, role: str, description: str, produced_by: str | None = None) -> str:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable(data), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        return self._register(path, role, description, produced_by)

    def text(self, rel: str, text: str, role: str, description: str, produced_by: str | None = None) -> str:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        return self._register(path, role, description, produced_by)

    def csv(self, rel: str, rows: list[dict[str, Any]], role: str, description: str, produced_by: str | None = None) -> str:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        keys: list[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with path.open("w", newline="", encoding="utf-8") as handle:
            w = csv.DictWriter(handle, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: json.dumps(_jsonable(v)) if isinstance(v, (dict, list)) else v for k, v in r.items()})
        return self._register(path, role, description, produced_by)

    def file(self, path: Path, role: str, description: str, produced_by: str | None = None) -> str:
        return self._register(Path(path), role, description, produced_by)

    def compact(self) -> list[dict[str, Any]]:
        return [{"id": r.id, "path": r.path, "role": r.role, "bytes": r.size_bytes} for r in self.records]


def qv(value: float, unit: str, method: str | None = None, description: str | None = None) -> QV:
    return QV(value=float(value), unit=unit, method=method, description=description)  # type: ignore[arg-type]


def material_selection(ctx: Any, *, selected: bool, notes: list[str] | None = None) -> MaterialSelection:
    d = ctx.data
    props: dict[str, QV] = {}
    if d.initial_permeability:
        props["initial_permeability"] = qv(d.initial_permeability, "1", "looked_up")
    if d.density:
        props["density"] = qv(d.density, "kg/m^3", "looked_up")
    if d.curie_temperature:
        props["curie_temperature"] = qv(d.curie_temperature, "K", "looked_up")
    for t, b, h in d.saturation_points:
        if not math.isnan(t):
            props[f"saturation_flux_density_{t - 273.15:.0f}C"] = qv(b, "T", "looked_up")
    from aipe_magnetics.database import physical_property
    props["thermal_conductivity"] = QV(**{**physical_property(d.material_class or "ferrite_mnzn", "thermal_conductivity").model_dump(exclude={"original"}),
                                          "method": "looked_up", "description": "property library (generic class value with bounds)"})
    return MaterialSelection(role="core", record_id=d.id, name=d.name, origin="catalogue", record_schema_version=d.record_schema_version,
                             properties=props, sources=list(d.sources)[:8], selected=selected, notes=notes or [])


def core_selection(core: Any, *, selected: bool, notes: list[str] | None = None) -> CoreSelection:
    eff = {}
    for k, unit in (("effective_area", "m^2"), ("effective_length", "m"), ("effective_volume", "m^3"), ("bobbin_window_area", "m^2"),
                    ("mean_turn_length", "m"), ("mass", "kg")):
        v = getattr(core, k)
        if v is not None:
            eff[k] = qv(v, unit, "looked_up")
    return CoreSelection(record_id=core.id, name=core.name, shape=core.shape, part_number=core.part_number, selected=selected,
                         dimensions={k: qv(v, "m", "looked_up") for k, v in core.dimensions.items()}, effective_parameters=eff, notes=notes or [])


def winding_definitions(geometry: Any, temperature: float) -> list[WindingDefinition]:
    from aipe_magnetics.engineering import formulas as F
    out = []
    rho = F.copper_resistivity(temperature)
    for name in geometry.winding_names():
        layers = [l for l in geometry.layers if l.winding == name]
        cond = layers[0].conductor
        length = sum(l.turns * l.mean_turn_length for l in layers)
        out.append(WindingDefinition(
            name=name, turns=sum(l.turns for l in layers), conductor=cond.model_dump(exclude_none=True) | {"description": cond.describe()},
            sections=len({l.section for l in layers}),
            layers=[WindingLayer(id=l.id, turns=l.turns, inner_distance=qv(l.d_in, "m", "derived"), outer_distance=qv(l.d_out, "m", "derived"),
                                 mean_turn_length=qv(l.mean_turn_length, "m", "derived")) for l in layers],
            conductor_length=qv(length, "m", "derived"),
            resistance_dc=qv(rho * length / cond.copper_area(), "Ohm", "analytical", f"at {temperature - 273.15:.0f} degC")))
    return out


def geometry_record(geometry: Any, artifacts: list[str]) -> GeometryRecord:
    from aipe_magnetics.geometry.ecore import GEOMETRY_MODEL_ID, GEOMETRY_MODEL_VERSION
    params = {k: qv(v["value"], v["unit"], "specified") for k, v in geometry.model.parameters().items()}
    d = geometry.derived
    derived = {"core_volume": qv(d["core_volume_m3"], "m^3", "derived"), "bounding_volume": qv(d["bounding_volume_m3"], "m^3", "derived"),
               "copper_volume": qv(d["copper_volume_m3"], "m^3", "derived"), "window_area_per_side": qv(d["window_area_per_side_m2"], "m^2", "derived"),
               "window_fill_bare_copper": qv(d["window_fill_bare_copper"], "1", "derived"),
               "core_exposed_area": qv(d["core_exposed_area_m2"], "m^2", "derived"),
               "winding_exposed_area": qv(d["winding_exposed_area_m2"], "m^2", "derived")}
    return GeometryRecord(model_id=GEOMETRY_MODEL_ID, model_version=GEOMETRY_MODEL_VERSION, parameter_hash=geometry.model.parameter_hash(),
                          parameters=params, derived=derived, checks=[c.__dict__ for c in geometry.checks],
                          model=geometry.model.model_dump(mode="json"), artifacts=artifacts)


def em_result_set(rid: str, solver_ref: str, op_ref: str | None, report: Any, artifacts: list[str]) -> ResultSet:
    q = {k: QV(value=v["value"], unit=v["unit"], method=v["method"], description=v["description"]) for k, v in report.quantities.items()}
    return ResultSet(id=rid, solver_ref=solver_ref, operating_point_ref=op_ref, quantities=q,
                     data={"winding_losses": _jsonable(report.winding_losses), "harmonic_solves": _jsonable(report.diagnostics.get("harmonic_solves", []))},
                     artifacts=artifacts, warnings=list(report.warnings))


def em_solver_record(sid: str, report: Any, config_artifact: str) -> SolverConfiguration:
    s = report.solver
    return SolverConfiguration(id=sid, domain="electromagnetic", backend=s["id"], version=s["version"], fidelity="fea",
                               formulation=s["formulation"], settings={k: _jsonable(v) for k, v in s.items() if k not in ("formulation",)},
                               configuration_artifact=config_artifact, limitations=list(report.limitations))


def thermal_solver_record(sid: str, thermal: Any, config_artifact: str) -> SolverConfiguration:
    from aipe_magnetics.simulation.thermal.fem2d import THERMAL_FEM_ID, THERMAL_FEM_VERSION
    return SolverConfiguration(id=sid, domain="thermal", backend=THERMAL_FEM_ID, version=THERMAL_FEM_VERSION, fidelity="fea_2p5d",
                               formulation="2.5-D steady conduction on the half cross-section with out-of-plane sinks and convective "
                                           "boundaries (linear triangles)", settings=_jsonable(thermal.settings), configuration_artifact=config_artifact,
                               limitations=list(thermal.assumptions))


def thermal_result_set(rid: str, solver_ref: str, op_ref: str | None, thermal: Any, artifacts: list[str]) -> ResultSet:
    q: dict[str, QV] = {}
    for name, st in thermal.region_stats.items():
        key = name.replace(":", "_")
        q[f"{key}_T_max"] = qv(st["T_max_K"], "K", "fea_2p5d", f"maximum temperature of {name}")
        q[f"{key}_T_mean"] = qv(st["T_mean_K"], "K", "fea_2p5d", f"area-mean temperature of {name}")
    q["energy_balance_relative_residual"] = qv(thermal.balance["relative_residual"], "1", "fea_2p5d", "sources - sinks - boundary flow")
    return ResultSet(id=rid, solver_ref=solver_ref, operating_point_ref=op_ref, quantities=q,
                     data={"hotspots": _jsonable(thermal.hotspots), "balance": _jsonable(thermal.balance)}, artifacts=artifacts)


def save_thermal_fields(thermal: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    m = thermal.mesh
    np.savez_compressed(path, nodes=m.nodes, triangles=m.triangles, element_region=m.element_region,
                        region_names=np.array([r.name for r in m.section.regions]), temperature_K=thermal.T,
                        element_source_W_per_m3=thermal.element_source, element_conductivity=thermal.element_k)
    return path


def md_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def software_versions() -> dict[str, str]:
    import platform

    import numpy
    import scipy
    out = {"aipe-magnetics": __version__, "python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__}
    for mod in ("gmsh", "cadquery", "matplotlib"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001 - optional dependency probe
            out[mod] = "not installed"
    return out


__all__ = ["ArtifactStore", "core_selection", "em_result_set", "em_solver_record", "geometry_record", "material_selection", "md_table",
           "qv", "save_thermal_fields", "software_versions", "thermal_result_set", "thermal_solver_record", "winding_definitions"]
