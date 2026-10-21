"""Waveform-informed DAB transformer design (capability ``aipe.magnetics.design_transformer_from_waveforms``).

The public ``design_transformer`` sizes windings with the square-current approximation ``I_rms = P / V`` and Dowell's
factor at the fundamental. This capability instead takes the converter's **actual** winding waveforms - sampled
(HDF5 / CSV / JSON, see :mod:`aipe_magnetics.engineering.waveform_import`) or synthesised from the single-phase-shift
model (:mod:`aipe_magnetics.design.transformers.dab`) - and evaluates every integer-turn candidate under these
identical operating conditions:

1. exact RMS and Fourier harmonics of the primary and secondary currents (closed form on the piecewise-linear period);
2. flux linkage ``lambda(t) = integral v_p dt`` (drift removed, zero mean) -> flux density ``B(t) = lambda / (N1 A_e)``;
   peak flux density, saturation margin against ``Bsat(T)``, allowed flux from the flux-limit policy (and an optional
   loss-density ceiling) -> minimum primary turns, ``N2 = round(N1 / n_target)`` within the ratio tolerance;
3. iGSE core loss on the **actual** flux waveform at the design temperature (:class:`~aipe_magnetics.database.LossEvaluator`,
   temperature interpolation between fitted sets; extrapolation is flagged, never silent);
4. winding layout per candidate (parametric E-core layers for cores with the E outline, a generic layer stack on the
   datasheet coil-former data otherwise), per-layer **harmonic Dowell** copper loss with the MMF profile of the
   arrangement (``P-S``, ``P-S-P`` ...) - litz bundles are expanded into ``sqrt(n_strands)`` strand sub-layers, foil
   layers are used as such - each winding with its own current spectrum; the 1-D litz proximity model is reported as
   a cross-reference;
5. 1-D window-field leakage inductance referred to the primary, the external inductance still required
   ``L_ext = L_total - L_leak``, and the magnetising inductance from the core reluctance (residual mating gap recorded);
6. window fill against the coil-former window, current density, insulation margins;
7. thermal estimate with the existing network model (E cores: core / winding nodes with hot-spot estimates, coupled
   electrothermally) or a lumped surface estimate (other outlines, labelled as such);
8. ranking of feasible candidates by total loss at the design temperature; comparison with the square-current
   approximation (same geometry, ``P/V`` square currents, and the public ``design_transformer`` result);
9. material comparison for this excitation (every catalogue material in the reference core; missing data listed) and
   an optional AIPE-Core Engineering State export whose evidence links to the waveform source (path + SHA-256).

Everything is analytical; nothing here is a measurement. Limitations are listed in every result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aipe_magnetics import TOOL_ID, __version__
from aipe_magnetics.api.types import WaveformInformedTransformerRequest
from aipe_magnetics.common.errors import AIPEMagneticsError, DataNotFoundError, InvalidInputError, OutOfDomainError
from aipe_magnetics.common.quantity import Quantity
from aipe_magnetics.common.results import EngineeringResult, ErrorRecord, ModelIdentity, SourceReference
from aipe_magnetics.common.units import MU_0
from aipe_magnetics.database import CoreData, FluxShape, LossEvaluator, MaterialData, core_data, list_core_data, material_data
from aipe_magnetics.design.transformers import dab
from aipe_magnetics.engineering import formulas as F
from aipe_magnetics.engineering import waveform_import as wi
from aipe_magnetics.engineering.periodic import PiecewiseLinear
from aipe_magnetics.engineering.winding_harmonic import (
    dowell_layer_stack_loss,
    leakage_inductance_1d,
    litz_proximity_loss_1d,
    litz_strand_skin_factor,
)
from aipe_magnetics.geometry.conductors import ConductorSpec, litz_for_area

CAP_WAVEFORM_DESIGN = "aipe.magnetics.design_transformer_from_waveforms"
IGSE_MODEL_ID = "aipe.magnetics.model.igse"
RESIDUAL_GAP_PER_LEG = 10e-6          # m, residual mating gap of an "ungapped" set (assumption, as in the v0.3 candidate model)
COPPER_DENSITY = 8933.0               # kg/m^3 (property library value)
BOBBIN_TUBE = 1.3e-3                  # m, generic layout defaults (same as geometry.ecore.BobbinSpec)
BOBBIN_CLEARANCE = 0.3e-3
BOBBIN_FLANGE = 1.7e-3
OUTER_CLEARANCE = 0.5e-3
LAYER_INSULATION = 0.05e-3
OUTER_WRAP = 0.1e-3
MAX_TURN_SEARCH = 400
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy >= 2 renamed trapz


# -------------------------------------------------------------------------------------------------
# excitation
# -------------------------------------------------------------------------------------------------
@dataclass
class TransformerExcitation:
    """One steady-state period of the transformer waveforms, in SI, with the provenance of every piece."""

    frequency: float
    primary_current: PiecewiseLinear
    secondary_current: PiecewiseLinear
    flux_linkage: PiecewiseLinear
    primary_voltage: PiecewiseLinear | None
    magnetising_current: PiecewiseLinear | None
    basis: str
    secondary_basis: str
    flux_basis: str
    source: dict[str, Any]
    period_detection: dict[str, Any] | None
    steady_state: dict[str, Any]
    series_inductance_total: float | None
    series_inductance_basis: str | None
    metadata_hints: dict[str, Any] = field(default_factory=dict)
    warnings: list[tuple[str, str]] = field(default_factory=list)
    assumptions: list[tuple[str, str, str]] = field(default_factory=list)
    _harmonics: dict[tuple[str, int], list[tuple[int, float]]] = field(default_factory=dict, repr=False)

    @property
    def period(self) -> float:
        return 1.0 / self.frequency

    @property
    def flux_linkage_peak_to_peak(self) -> float:
        return self.flux_linkage.peak_to_peak()

    def flux_shape(self) -> FluxShape:
        """Normalised flux shape ``(t/T, lambda / (lambda_pp / 2))`` for the vectorised iGSE evaluator."""
        half = 0.5 * self.flux_linkage_peak_to_peak
        pts = [(t, v / half) for t, v in self.flux_linkage.points]
        pts[-1] = (1.0, pts[0][1])
        return FluxShape.piecewise_linear(pts)

    def harmonics(self, winding: str, n_max: int) -> list[tuple[int, float]]:
        key = (winding, n_max)
        if key not in self._harmonics:
            wave = self.primary_current if winding == "primary" else self.secondary_current
            self._harmonics[key] = [(n, a) for n, a, _ in wave.harmonic_amplitudes(n_max) if a > 1e-9 * max(1.0, wave.peak())]
        return self._harmonics[key]

    def throughput_power(self) -> float | None:
        """Mean of ``v_p(t) i_p(t)`` over the period (W); ``None`` without a voltage waveform."""
        if self.primary_voltage is None:
            return None
        x = np.linspace(0.0, 1.0, 8193)
        v, i = self.primary_voltage.value(x), self.primary_current.value(x)
        return float(_trapezoid(v * i, x))

    def describe(self, n_harmonics: int = 15) -> dict[str, Any]:
        def wave(pl: PiecewiseLinear | None, n: int) -> dict[str, Any] | None:
            if pl is None:
                return None
            return {"unit": pl.unit, "rms": pl.rms(), "peak": pl.peak(), "mean": pl.mean(), "peak_to_peak": pl.peak_to_peak(),
                    "n_breakpoints": len(pl.points), "harmonics": [{"n": k, "amplitude": a, "phase_rad": p} for k, a, p in pl.harmonic_amplitudes(n)],
                    "harmonic_energy_fraction": pl.harmonic_energy_fraction(n)}
        return {"basis": self.basis, "frequency_Hz": self.frequency, "period_s": self.period,
                "primary_current": wave(self.primary_current, n_harmonics), "secondary_current": wave(self.secondary_current, n_harmonics),
                "secondary_basis": self.secondary_basis, "primary_voltage": wave(self.primary_voltage, 9), "flux_basis": self.flux_basis,
                "flux_linkage_peak_to_peak_Vs": self.flux_linkage_peak_to_peak,
                "magnetising_current": wave(self.magnetising_current, 5), "throughput_power_W": self.throughput_power(),
                "series_inductance_total_H": self.series_inductance_total, "series_inductance_basis": self.series_inductance_basis,
                "source": self.source, "period_detection": self.period_detection, "steady_state": self.steady_state,
                "metadata_hints": self.metadata_hints, "warnings": [{"code": c, "message": m} for c, m in self.warnings]}


def _triangle(peak_to_peak: float, unit: str, label: str) -> PiecewiseLinear:
    return PiecewiseLinear(points=((0.0, -0.5 * peak_to_peak), (0.5, 0.5 * peak_to_peak), (1.0, -0.5 * peak_to_peak)), unit=unit, label=label)


def build_excitation(request: WaveformInformedTransformerRequest) -> TransformerExcitation:
    """Load (or synthesise) the waveforms named by the request and reduce them to one steady-state period."""
    src = request.waveforms
    f = request.frequency.value
    n_target = request.turns_ratio_target
    warnings: list[tuple[str, str]] = []
    assumptions: list[tuple[str, str, str]] = []
    if src.kind == "sps":
        sps = src.sps
        assert sps is not None
        power, v1, v2 = sps.power.value, sps.primary_voltage.value, sps.secondary_voltage.value
        if sps.series_inductance is not None:
            inductance = sps.series_inductance.value
            phi = dab.phase_shift_for_power(power, v1, v2, n_target, f, inductance)
        else:
            phi = sps.phase_shift.value  # type: ignore[union-attr]
            inductance = dab.inductance_for_phase_shift(power, v1, v2, n_target, f, phi)
        i_p = dab.inductor_current(v1, v2, n_target, f, inductance, phi)
        i_s = i_p.scaled(n_target, unit="A", label="secondary current (n x primary, ideal transformer)")
        lam = _triangle(v1 / (2.0 * f), "V*s", "flux linkage of the ideal +-V1 square wave")
        v_p = wi.square_piecewise(v1, 0.5, unit="V", label="ideal square primary voltage")
        assumptions.append(("sps_waveforms", f"Transformer waveforms synthesised with the lossless single-phase-shift model: +-{v1:g} V / +-{v2:g} V "
                            f"square voltages, phase shift {math.degrees(phi):.2f} deg, total series inductance {inductance * 1e6:.2f} uH referred to "
                            "the primary, no dead time, magnetising current neglected in the winding currents.", "model_assumption"))
        total = request.series_inductance_total.value if request.series_inductance_total is not None else inductance
        basis = "request" if request.series_inductance_total is not None else "sps_model"
        return TransformerExcitation(frequency=f, primary_current=i_p, secondary_current=i_s, flux_linkage=lam, primary_voltage=v_p,
                                     magnetising_current=None, basis="synthesised_sps", secondary_basis="ideal_transformer_times_turns_ratio_target",
                                     flux_basis="ideal_square_voltage", source={"kind": "sps", "power_W": power, "v1_V": v1, "v2_V": v2,
                                                                                 "phase_shift_rad": phi, "series_inductance_H": inductance},
                                     period_detection=None, steady_state={"cycles": 1, "note": "analytical waveform"},
                                     series_inductance_total=total, series_inductance_basis=basis, warnings=warnings, assumptions=assumptions)

    channels = src.channels
    assert channels is not None
    names = [n for n in (channels.primary_current, channels.secondary_current, channels.primary_voltage, channels.magnetising_current) if n]
    window = None if src.window is None else {"t0": None if src.window.t0 is None else src.window.t0.value,
                                                "t1": None if src.window.t1 is None else src.window.t1.value}
    if src.kind == "hdf5":
        ws = wi.load_waveforms_hdf5(src.path, channels=names, time_dataset=src.time_dataset, channel_group=src.channel_group, window=window)  # type: ignore[arg-type]
    elif src.kind == "csv":
        columns = {n: (n, (src.units or {})[n]) for n in names}
        ws = wi.load_waveforms_csv(src.path, time_column=src.time_column, time_unit=src.time_unit, columns=columns,  # type: ignore[arg-type]
                                   delimiter=src.delimiter, window=window)
    else:
        ws = wi.load_waveforms_json(src.document if src.document is not None else src.path, window=window)  # type: ignore[arg-type]
        missing = [n for n in names if n not in ws.channels]
        if missing:
            raise DataNotFoundError(f"channels {missing} not in the JSON document; available: {sorted(ws.channels)}")

    def require_unit(name: str, unit: str) -> None:
        ch = ws.channel(name)
        if ch.unit != unit:
            raise InvalidInputError(f"channel {name!r} has unit {ch.original_unit!r} ({ch.unit}); expected a {unit} quantity")

    require_unit(channels.primary_current, "A")
    estimate = wi.detect_period(ws.time, ws.channel(channels.primary_current).values, frequency_hint=f)
    for note in estimate.notes:
        warnings.append(("period_detection", note))
    period = estimate.period
    if ws.duration < period * (1 - 1e-9):
        raise InvalidInputError(f"the waveform record ({ws.duration:g} s) is shorter than one period ({period:g} s)")
    i_slice = wi.extract_period(ws.time, ws.channel(channels.primary_current).values, period, unit="A")
    i_p = i_slice.to_piecewise_linear(f"primary current ({channels.primary_current})")
    if i_slice.info["closure_flagged"]:
        warnings.append(("period_closure", f"primary current does not close over the extracted period (residual {i_slice.closure_residual:.2%} of the "
                         "peak-to-peak value); the end value was set equal to the start value"))
    steady = wi.steady_state_metrics(ws.time, ws.channel(channels.primary_current).values, period)
    if steady.get("cycles", 0) >= 2 and steady["rms_relative_spread"] > 0.01:
        warnings.append(("steady_state", f"primary-current RMS varies {steady['rms_relative_spread']:.2%} between the {steady['cycles']} analysed "
                         "cycles; the record may not be in steady state"))
    if channels.secondary_current:
        require_unit(channels.secondary_current, "A")
        s_slice = wi.extract_period(ws.time, ws.channel(channels.secondary_current).values, period, unit="A")
        i_s = s_slice.to_piecewise_linear(f"secondary current ({channels.secondary_current})")
        secondary_basis = "imported"
        if s_slice.info["closure_flagged"]:
            warnings.append(("period_closure", f"secondary current closure residual {s_slice.closure_residual:.2%}"))
    else:
        i_s = i_p.scaled(n_target, unit="A", label="secondary current (primary x turns-ratio target)")
        secondary_basis = "derived_from_primary_and_turns_ratio_target"
        assumptions.append(("secondary_current_derived", f"No secondary current channel: the secondary current is the primary current times the "
                            f"turns-ratio target {n_target:g} (ideal transformer, magnetising current neglected).", "model_assumption"))
    v_p = None
    if channels.primary_voltage:
        require_unit(channels.primary_voltage, "V")
        lam_slice = wi.flux_linkage_from_voltage(ws.time, ws.channel(channels.primary_voltage).values, period)
        lam = lam_slice.to_piecewise_linear("flux linkage (integral of the primary voltage)")
        v_p = wi.extract_period(ws.time, ws.channel(channels.primary_voltage).values, period, unit="V").to_piecewise_linear("primary voltage")
        flux_basis = "integral_of_voltage_channel"
        imbalance = lam_slice.info["volt_second_relative_imbalance"]
        if imbalance > 1e-3:
            warnings.append(("volt_second_imbalance", f"net volt-seconds per period are {imbalance:.2%} of the flux-linkage swing (drift removed "
                             "before the loss evaluation; a real DC flux offset is not modelled by iGSE)"))
        flux_info = lam_slice.info
    else:
        v1 = request.primary_voltage.value  # type: ignore[union-attr]
        lam = _triangle(v1 / (2.0 * f), "V*s", "flux linkage of the ideal +-V1 square wave")
        flux_basis = "ideal_square_voltage"
        flux_info = {"flux_linkage_peak_to_peak_Vs": lam.peak_to_peak()}
        assumptions.append(("ideal_square_voltage", f"No primary voltage channel: the flux is the integral of an ideal +-{v1:g} V, 50 % square wave "
                            "(triangular flux); device drops, dead time and ringing are not represented.", "model_assumption"))
    i_m = None
    if channels.magnetising_current:
        require_unit(channels.magnetising_current, "A")
        i_m = wi.extract_period(ws.time, ws.channel(channels.magnetising_current).values, period, unit="A").to_piecewise_linear("magnetising current")
    hints: dict[str, Any] = {}
    spec = ws.metadata.get("spec") if isinstance(ws.metadata.get("spec"), dict) else {}
    for key in ("f_sw", "n_ratio", "l_ext", "l_lk_p", "l_lk_s", "l_m", "v1", "v2", "phi_deg", "name"):
        if key in spec:
            hints[key] = spec[key]
    if "f_sw" in hints and isinstance(hints["f_sw"], (int, float)) and abs(hints["f_sw"] - f) > 1e-6 * f:
        warnings.append(("frequency_metadata", f"the request frequency {f:g} Hz differs from the record metadata f_sw = {hints['f_sw']:g} Hz"))
    if "n_ratio" in hints and isinstance(hints["n_ratio"], (int, float)) and abs(hints["n_ratio"] - n_target) > 1e-9:
        warnings.append(("turns_ratio_metadata", f"the simulated turns ratio {hints['n_ratio']:g} differs from the target {n_target:g}; the "
                         "imported secondary current belongs to the simulated ratio"))
    total = None
    basis = None
    if request.series_inductance_total is not None:
        total, basis = request.series_inductance_total.value, "request"
    elif all(isinstance(hints.get(k), (int, float)) for k in ("l_ext", "l_lk_p", "l_lk_s", "n_ratio")):
        total = float(hints["l_ext"]) + float(hints["l_lk_p"]) + float(hints["l_lk_s"]) * float(hints["n_ratio"]) ** 2
        basis = "waveform_metadata (l_ext + l_lk_p + l_lk_s n^2)"
        assumptions.append(("series_inductance_from_metadata", f"Total series inductance {total * 1e6:.2f} uH (referred to the primary) taken from "
                            "the waveform record metadata (l_ext + l_lk_p + l_lk_s n^2).", "user_input"))
    else:
        warnings.append(("series_inductance_unknown", "no total series inductance given or found in the record metadata; the external "
                         "inductance still required cannot be reported"))
    source = dict(ws.source)
    source.update({"channels": {"primary_current": channels.primary_current, "secondary_current": channels.secondary_current,
                                "primary_voltage": channels.primary_voltage, "magnetising_current": channels.magnetising_current},
                   "window_s": None if ws.window is None else list(ws.window), "n_samples_in_window": ws.n_samples,
                   "period_extracted_s": [i_slice.t_start, i_slice.t_end], "flux_linkage": flux_info})
    return TransformerExcitation(frequency=f, primary_current=i_p, secondary_current=i_s, flux_linkage=lam, primary_voltage=v_p,
                                 magnetising_current=i_m, basis="imported", secondary_basis=secondary_basis, flux_basis=flux_basis,
                                 source=source, period_detection=estimate.to_json(), steady_state=steady, series_inductance_total=total,
                                 series_inductance_basis=basis, metadata_hints=hints, warnings=warnings, assumptions=assumptions)


# -------------------------------------------------------------------------------------------------
# materials and cores
# -------------------------------------------------------------------------------------------------
@dataclass
class MaterialContext:
    data: MaterialData
    evaluator: LossEvaluator | None
    bsat: float | None
    bsat_status: str | None
    reasons: list[str] = field(default_factory=list)
    loss_notes: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.data.id

    @property
    def evaluable(self) -> bool:
        return not self.reasons

    def missing(self) -> list[str]:
        """Catalogue data this material lacks for the design and its refinements (reported, never guessed)."""
        d = self.data
        out = []
        if not d.has_saturation_data():
            out.append("saturation flux density vs temperature")
        if d.initial_permeability is None:
            out.append("initial permeability")
        if d.density is None:
            out.append("density")
        if d.curie_temperature is None:
            out.append("Curie temperature")
        if d.resistivity is None:
            out.append("resistivity")
        if not d.steinmetz_sets:
            out.append("core-loss parameter sets")
        props = d.properties()
        if not props.loss_reference_points:
            out.append("datasheet core-loss reference points")
        if not props.coercivity and not props.remanence:
            out.append("B-H loop data (coercivity / remanence)")
        if not props.dc_bias_points:
            out.append("DC-bias (premagnetisation) data")
        if not props.amplitude_permeability and not props.permeability_vs_temperature:
            out.append("permeability vs amplitude / temperature (complex permeability)")
        return out


def material_contexts(request: WaveformInformedTransformerRequest, frequency: float, temperature: float) -> list[MaterialContext]:
    from aipe_magnetics.data.catalog import Catalog
    cat = Catalog.default()
    if request.materials:
        ids = [material_data(q, cat).id for q in request.materials]
    else:
        ids = [m.id for m in cat.materials() if not m.synthetic]
    out: list[MaterialContext] = []
    for mid in ids:
        md = material_data(mid, cat)
        reasons: list[str] = []
        notes: list[str] = []
        bsat = bsat_status = None
        if md.synthetic:
            reasons.append("synthetic record (illustrative only)")
        if not md.has_saturation_data():
            reasons.append("no saturation flux density data")
        else:
            try:
                bsat, bsat_status = md.saturation_flux_density(temperature)
            except (OutOfDomainError, DataNotFoundError) as error:
                reasons.append(f"Bsat at {temperature - 273.15:.0f} degC: {error.message}")
        evaluator = None
        if not md.steinmetz_sets:
            reasons.append("no Steinmetz parameter set")
        else:
            evaluator = LossEvaluator(md.id, md.steinmetz_sets)
            try:
                _sets, _w, sel_warn = evaluator.select(frequency, temperature, policy=request.policy)
                notes.extend(sel_warn)
            except AIPEMagneticsError as error:
                reasons.append(f"loss model: {error.message}")
                evaluator = None
        out.append(MaterialContext(data=md, evaluator=evaluator, bsat=bsat, bsat_status=bsat_status, reasons=reasons, loss_notes=notes))
    return out


def core_candidates(request: WaveformInformedTransformerRequest, materials: list[MaterialContext]) -> list[CoreData]:
    if request.cores:
        return [core_data(c) for c in request.cores]
    cores = list_core_data()
    if request.material_offer_policy == "catalogue":
        mids = {m.id for m in materials}
        cores = [c for c in cores if set(c.material_refs) & mids]
    return cores


# -------------------------------------------------------------------------------------------------
# layouts
# -------------------------------------------------------------------------------------------------
@dataclass
class LayerInfo:
    winding: str
    section: int
    d_in: float
    d_out: float
    turns: int
    mean_turn_length: float
    conductor: ConductorSpec


@dataclass
class Layout:
    fidelity: str
    layers: list[LayerInfo]
    winding_height: float
    checks: list[dict[str, Any]]
    derived: dict[str, Any]
    geometry: Any = None
    notes: list[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return all(c["passed"] for c in self.checks)

    def windings(self) -> list[str]:
        out: list[str] = []
        for layer in self.layers:
            if layer.winding not in out:
                out.append(layer.winding)
        return out

    def turns(self, winding: str) -> int:
        return sum(l.turns for l in self.layers if l.winding == winding)

    def conductor(self, winding: str) -> ConductorSpec:
        return next(l.conductor for l in self.layers if l.winding == winding)

    def length(self, winding: str) -> float:
        return sum(l.turns * l.mean_turn_length for l in self.layers if l.winding == winding)


def _sections(arrangement: str, turns: dict[str, int], conductors: dict[str, ConductorSpec]) -> list[tuple[str, int, ConductorSpec]]:
    letters = [p.strip() for p in arrangement.split("-") if p.strip()]
    names = {"P": "primary", "S": "secondary"}
    if not letters or any(l not in names for l in letters):
        raise InvalidInputError(f"arrangement {arrangement!r} must consist of 'P' and 'S' letters separated by '-'")
    if set(names[l] for l in letters) != {"primary", "secondary"}:
        raise InvalidInputError(f"arrangement {arrangement!r} must place both the primary and the secondary")
    counts = {l: letters.count(l) for l in set(letters)}
    parts: dict[str, list[int]] = {}
    for letter, count in counts.items():
        n = turns[names[letter]]
        if n < count:
            raise InvalidInputError(f"{names[letter]} has {n} turns and cannot be split into {count} sections")
        base, extra = divmod(n, count)
        parts[letter] = [base + (1 if i < extra else 0) for i in range(count)]
    used = {l: 0 for l in counts}
    out = []
    for letter in letters:
        out.append((names[letter], parts[letter][used[letter]], conductors[names[letter]]))
        used[letter] += 1
    return out


def build_layout(core: CoreData, material_id: str, n1: int, n2: int, cond_p: ConductorSpec, cond_s: ConductorSpec, arrangement: str,
                 creepage_margin: float, section_insulation: float) -> Layout:
    """Parametric E-core layout when the outline is recorded, otherwise a generic layer stack on the coil-former data."""
    if core.has_e_geometry():
        from aipe_magnetics.geometry.ecore import BobbinSpec, ECoreDimensions, ECoreMagnetic, GapSpec, InsulationSpec, WindingLayout, arrangement_sections
        dims = ECoreDimensions.from_core_data(core)
        sections = arrangement_sections(arrangement, {"P": (n1, cond_p), "S": (n2, cond_s)})
        model = ECoreMagnetic(name=f"{core.name}", core=dims, core_material=material_id, gap=GapSpec(spacer=RESIDUAL_GAP_PER_LEG, note="residual mating gap"),
                              bobbin=BobbinSpec(), layout=WindingLayout(sections=sections, arrangement=arrangement,
                                                                        insulation=InsulationSpec(creepage_margin=creepage_margin,
                                                                                                  section_insulation=section_insulation)))
        geo = model.build()
        layers = [LayerInfo(l.winding, l.section, l.d_in, l.d_out, l.turns, l.mean_turn_length, l.conductor) for l in geo.layers]
        checks = [{"code": c.code, "passed": c.passed, "message": c.message} for c in geo.checks]
        derived = {"core_volume_m3": geo.derived["core_volume_m3"], "copper_volume_m3": geo.derived["copper_volume_m3"],
                   "window_fill_geometric": geo.derived["window_fill_bare_copper"], "core_exposed_area_m2": geo.derived["core_exposed_area_m2"],
                   "winding_exposed_area_m2": geo.derived["winding_exposed_area_m2"], "bounding_volume_m3": geo.derived["bounding_volume_m3"]}
        return Layout(fidelity="parametric_e_core", layers=layers, winding_height=model.winding_space()["height"] - 2 * creepage_margin,
                      checks=checks, derived=derived, geometry=geo,
                      notes=["parametric E-core layout: per-layer radial position and mean turn length from the core outline"])
    # generic stack from the coil-former window and the outline dimensions that exist
    if core.mean_turn_length is None or (core.bobbin_window_area or core.window_area) is None:
        raise DataNotFoundError(f"core {core.id} has no coil-former data (mean turn length / window area); windings cannot be laid out")
    window_area = core.bobbin_window_area or core.window_area
    d = core.dimensions
    notes = []
    if "window_height_half" in d:
        height_total = 2.0 * d["window_height_half"]
    elif "window_height" in d:
        height_total = d["window_height"]
    elif "window_height_single_u" in d:
        height_total = d["window_height_single_u"]
    else:
        height_total = math.sqrt(2.0 * window_area)
        notes.append("window height estimated as sqrt(2 A_N) (2:1 window aspect assumption)")
    height = height_total - 2 * BOBBIN_FLANGE - 2 * creepage_margin
    radial_available = window_area / height_total - OUTER_CLEARANCE
    notes.append("generic layer stack: radial space = coil-former window area / window height; every layer uses the datasheet mean turn length")
    checks: list[dict[str, Any]] = []
    layers: list[LayerInfo] = []
    if height <= 0:
        checks.append({"code": "winding_height", "passed": False, "message": "creepage margins leave no winding height"})
        return Layout("generic_layer_stack", layers, 0.0, checks, {}, None, notes)
    pos = BOBBIN_CLEARANCE + BOBBIN_TUBE
    for si, (winding, turns, cond) in enumerate(_sections(arrangement, {"primary": n1, "secondary": n2}, {"primary": cond_p, "secondary": cond_s})):
        if si > 0:
            pos += section_insulation
        if cond.kind == "foil":
            width = height - 2 * cond.insulation
            cond = cond.model_copy(update={"width": width})
            per_layer, n_layers = 1, turns
        else:
            pitch = cond.axial_pitch()
            fit = int(math.floor(height / pitch + 1e-9))
            if fit < 1:
                checks.append({"code": f"{winding}.turn_fit", "passed": False, "message": f"{winding}: one turn ({pitch * 1e3:.2f} mm) does not fit "
                                                                                           f"the winding height {height * 1e3:.2f} mm"})
                fit = 1
            n_layers = int(math.ceil(turns / fit))
            per_layer = int(math.ceil(turns / n_layers))
        remaining = turns
        for li in range(n_layers):
            if li > 0:
                pos += LAYER_INSULATION
            n_here = min(per_layer, remaining)
            remaining -= n_here
            build = cond.radial_build()
            layers.append(LayerInfo(winding, si, pos, pos + build, n_here, core.mean_turn_length, cond))
            pos += build
    pos += OUTER_WRAP
    checks.append({"code": "radial_fit", "passed": pos <= radial_available * (1 + 1e-9),
                   "message": f"radial build {pos * 1e3:.2f} mm (incl. bobbin and insulation) vs available {radial_available * 1e3:.2f} mm"})
    copper_volume = sum(l.turns * l.mean_turn_length * l.conductor.copper_area() for l in layers)
    exposed = None
    if all(k in d for k in ("length_a", "height_b_half", "depth_c")):
        a, b2, c = d["length_a"], 2 * d["height_b_half"], d["depth_c"]
        exposed = 2 * (a * b2 + a * c + b2 * c) - 4 * (height_total * 0.5 * (d.get("window_width_total", a / 2) - d.get("centre_leg_diameter", d.get("centre_leg_width", 0.0))))
    elif all(k in d for k in ("diameter", "height_total")):
        exposed = math.pi * d["diameter"] * d["height_total"] + math.pi * d["diameter"] ** 2 / 2.0
    derived = {"copper_volume_m3": copper_volume, "exposed_area_m2": exposed, "radial_build_m": pos, "radial_available_m": radial_available}
    return Layout("generic_layer_stack", layers, height, checks, derived, None, notes)


# -------------------------------------------------------------------------------------------------
# winding losses, leakage, magnetising inductance
# -------------------------------------------------------------------------------------------------
def _stack_entries(layout: Layout, n1: int, n2: int, reference: str, temperature: float, expand_litz: bool) -> list[dict[str, Any]]:
    """Radially ordered Dowell stack for 1 A of the ``reference`` winding; the other winding's layers shape the MMF only."""
    rho = F.copper_resistivity(temperature)
    ratio = {"primary": 1.0 if reference == "primary" else n2 / n1, "secondary": n1 / n2 if reference == "primary" else 1.0}
    sign = {"primary": 1.0, "secondary": -1.0}
    out: list[dict[str, Any]] = []
    for layer in sorted(layout.layers, key=lambda l: l.d_in):
        c = layer.conductor
        at = sign[layer.winding] * ratio[layer.winding] * layer.turns
        r_dc = rho * layer.turns * layer.mean_turn_length / c.copper_area() if layer.winding == reference else 0.0
        if c.kind == "foil":
            out.append({"winding": layer.winding, "ampere_turns": at, "turns": layer.turns, "r_dc": r_dc, "kind": "foil", "size": c.thickness,
                        "pitch": layout.winding_height, "x0": layer.d_in, "x1": layer.d_out})
        elif c.kind == "litz" and expand_litz:
            k = max(1, int(round(math.sqrt(c.strand_count))))  # strand sub-layers across the bundle
            strand_pitch = c.envelope_diameter() / math.sqrt(c.strand_count)
            for _ in range(k):
                out.append({"winding": layer.winding, "ampere_turns": at / k, "turns": layer.turns, "r_dc": r_dc * k, "kind": "round",
                            "size": c.strand_diameter, "pitch": strand_pitch, "x0": layer.d_in, "x1": layer.d_out})
        else:
            size = c.bare_diameter if c.kind == "round" else c.equivalent_square_side()
            out.append({"winding": layer.winding, "ampere_turns": at, "turns": layer.turns, "r_dc": r_dc, "kind": "round", "size": size,
                        "pitch": c.axial_pitch(), "x0": layer.d_in, "x1": layer.d_out})
    return out


def winding_losses(layout: Layout, excitation: TransformerExcitation, n1: int, n2: int, temperature: float, n_harmonics: int) -> dict[str, dict[str, Any]]:
    """Per-winding copper loss: per-layer harmonic Dowell with each winding's own spectrum (litz as strand sub-layers)."""
    rho = F.copper_resistivity(temperature)
    out: dict[str, dict[str, Any]] = {}
    for winding in ("primary", "secondary"):
        wave = excitation.primary_current if winding == "primary" else excitation.secondary_current
        harmonics = excitation.harmonics(winding, n_harmonics)
        stack = _stack_entries(layout, n1, n2, winding, temperature, expand_litz=True)
        dow = dowell_layer_stack_loss(stack, harmonics, excitation.frequency, temperature, dc=abs(wave.mean()))
        r_dc = sum(rho * l.turns * l.mean_turn_length / l.conductor.copper_area() for l in layout.layers if l.winding == winding)
        i_rms = wave.rms()
        p_dc_equiv = r_dc * i_rms ** 2
        entry: dict[str, Any] = {"method": "dowell_per_layer_harmonic" + ("_litz_strand_sublayers" if layout.conductor(winding).kind == "litz" else ""),
                                 "loss_W": dow.get(winding, 0.0), "r_dc_Ohm": r_dc, "current_rms_A": i_rms, "loss_dc_resistance_W": p_dc_equiv,
                                 "effective_ac_factor": dow.get(winding, 0.0) / p_dc_equiv if p_dc_equiv > 0 else None,
                                 "turns": layout.turns(winding), "conductor": layout.conductor(winding).describe(),
                                 "copper_area_m2": layout.conductor(winding).copper_area(layout.winding_height),
                                 "current_density_A_per_m2": i_rms / layout.conductor(winding).copper_area(layout.winding_height),
                                 "layers": sum(1 for l in layout.layers if l.winding == winding), "harmonics_used": len(harmonics)}
        cond = layout.conductor(winding)
        if cond.kind == "litz":
            stack_1d = [{"x0": s["x0"], "x1": s["x1"], "ampere_turns": s["ampere_turns"], "turns": s["turns"], "mlt": 0.0, "strands": 0, "strand_diameter": None}
                        for s in _stack_entries(layout, n1, n2, winding, temperature, expand_litz=False)]
            for s, layer in zip(stack_1d, sorted(layout.layers, key=lambda l: l.d_in)):
                if layer.winding == winding:
                    s.update({"strands": cond.strand_count, "strand_diameter": cond.strand_diameter, "mlt": layer.mean_turn_length})
            prox = litz_proximity_loss_1d(stack_1d, layout.winding_height, harmonics, excitation.frequency, temperature)
            skin = r_dc * wave.mean() ** 2 + sum(r_dc * a * a / 2.0 * litz_strand_skin_factor(cond.strand_diameter, F.skin_depth(n * excitation.frequency, rho))
                                                for n, a in harmonics)
            entry["litz_1d_reference"] = {"proximity_W": prox, "dc_and_skin_W": skin, "total_W": prox + skin,
                                          "method": "strand skin factor + 1-D window-field proximity (Sullivan 1999), reference only"}
        out[winding] = entry
    return out


def leakage_inductance(layout: Layout, n1: int, n2: int) -> float:
    stack = _stack_entries(layout, n1, n2, "primary", 293.15, expand_litz=False)
    turns_total = sum(l.turns for l in layout.layers)
    mlt = sum(l.turns * l.mean_turn_length for l in layout.layers) / turns_total
    return leakage_inductance_1d([{"x0": s["x0"], "x1": s["x1"], "ampere_turns": s["ampere_turns"]} for s in stack], layout.winding_height, mlt)


def magnetising_inductance(core: CoreData, material: MaterialContext, n1: int) -> dict[str, Any]:
    out: dict[str, Any] = {"residual_gap_per_leg_m": RESIDUAL_GAP_PER_LEG}
    mu_i = material.data.initial_permeability
    if mu_i is None or core.effective_length is None or core.effective_area is None:
        out["reluctance_model_H"] = None
        out["note"] = "initial permeability or effective length missing; reluctance estimate not possible"
    else:
        reluctance = (core.effective_length / mu_i + 2 * RESIDUAL_GAP_PER_LEG) / (MU_0 * core.effective_area)
        out["reluctance_model_H"] = n1 * n1 / reluctance
        out["effective_permeability"] = core.effective_length / (core.effective_length / mu_i + 2 * RESIDUAL_GAP_PER_LEG)
    from aipe_magnetics.data.catalog import Catalog
    record = Catalog.default().get(core.id)
    ext = (record.extensions or {}).get("aipe.datasheet", {}) if isinstance(record.extensions, dict) else {}
    grade = material.data.grade or ""
    key = f"ungapped_{grade}"
    if isinstance(ext, dict) and isinstance(ext.get(key), dict) and isinstance(ext[key].get("AL_nH"), (int, float)):
        out["datasheet_AL_H"] = float(ext[key]["AL_nH"]) * 1e-9 * n1 * n1
        out["datasheet_AL_nH"] = float(ext[key]["AL_nH"])
    return out


# -------------------------------------------------------------------------------------------------
# thermal
# -------------------------------------------------------------------------------------------------
def _cooling(request: WaveformInformedTransformerRequest):
    from aipe_magnetics.simulation.thermal.convection import Cooling
    th = request.thermal
    h = th.heat_transfer_coefficient.value if th.heat_transfer_coefficient is not None else (40.0 if th.cooling == "forced" else None)
    return Cooling(mode=th.cooling, ambient=th.ambient.value, forced_h=h if th.cooling == "forced" else None, radiation=th.radiation,
                   emissivity=th.emissivity)


def _lumped_surface(p_total: float, area: float, cooling: Any, height: float) -> dict[str, Any]:
    t = cooling.ambient + 40.0
    for _ in range(200):
        h = cooling.h_total(t, height)
        t_new = cooling.ambient + p_total / (h * area)
        if abs(t_new - t) < 0.01:
            t = t_new
            break
        t = 0.5 * (t + t_new)
    return {"method": "lumped_surface_estimate", "surface_temperature_K": t, "heat_transfer_coefficient_W_per_m2K": cooling.h_total(t, height),
            "exposed_area_m2": area, "assumptions": [cooling.describe(), "uniform surface temperature; no internal gradient (hot spots are "
                                                                            "not resolved: optimistic)"]}


def thermal_estimate(layout: Layout, core: CoreData, material: MaterialContext, request: WaveformInformedTransformerRequest, excitation: TransformerExcitation,
                     n1: int, n2: int, core_loss_at: Any, t_design: float) -> dict[str, Any]:
    """Electrothermal estimate: E cores through the thermal network (core + winding nodes), others as a lumped surface."""
    from aipe_magnetics.database import physical_property
    cooling = _cooling(request)
    k_core = physical_property(material.data.material_class or "ferrite_mnzn", "thermal_conductivity").value
    if layout.fidelity == "parametric_e_core" and layout.geometry is not None:
        from aipe_magnetics.design.common import electrothermal, pack_conductivities
        from aipe_magnetics.simulation.thermal.network import solve_thermal_network
        geo = layout.geometry
        kp = pack_conductivities()

        def losses(tc: float, tw: float) -> tuple[float, dict[str, float]]:
            pc = core_loss_at(tc)
            wl = winding_losses(layout, excitation, n1, n2, tw, request.harmonics)
            return pc, {w: v["loss_W"] for w, v in wl.items()}

        def thermal(pc: float, pw: dict[str, float]):
            res = solve_thermal_network(geo, core_loss=pc, winding_losses=pw, cooling=cooling, k_core=k_core, k_bobbin=0.3, k_insulation=0.12, k_pack=kp)
            tw = max(v for k, v in res.temperatures.items() if k != "core")
            return res.temperatures["core"], tw, res

        et = electrothermal(losses, thermal, t_design, t_design, tol=0.5, max_iterations=15)
        net = et.thermal
        hot_w = max(v for k, v in net.hotspots.items() if k not in ("core", "primary", "secondary"))
        return {"method": "thermal_network_electrothermal", "converged": et.converged, "iterations": et.iterations,
                "temperatures_K": {k: float(v) for k, v in net.temperatures.items()}, "hotspots_K": {k: float(v) for k, v in net.hotspots.items()},
                "hotspot_winding_K": float(hot_w), "hotspot_core_K": float(net.hotspots["core"]), "ambient_K": cooling.ambient,
                "temperature_rise_K": float(max(hot_w, net.hotspots["core"]) - cooling.ambient),
                "losses_at_estimated_temperatures": {"core_W": et.core_loss, "windings_W": et.winding_losses,
                                                     "total_W": et.core_loss + sum(et.winding_losses.values())},
                "assumptions": net.assumptions, "cooling": cooling.to_json()}
    area = layout.derived.get("exposed_area_m2")
    if not area:
        return {"method": "not_evaluated", "reason": "no outline dimensions for a surface estimate", "cooling": cooling.to_json()}
    wl = winding_losses(layout, excitation, n1, n2, t_design, request.harmonics)
    p_total = core_loss_at(t_design) + sum(v["loss_W"] for v in wl.values())
    d = core.dimensions
    height = d.get("height_total") or (2 * d["height_b_half"] if "height_b_half" in d else math.sqrt(area))
    lumped = _lumped_surface(p_total, area, cooling, height)
    t_s = lumped["surface_temperature_K"]
    return {**lumped, "converged": True, "hotspot_winding_K": t_s, "hotspot_core_K": t_s, "ambient_K": cooling.ambient,
            "temperature_rise_K": t_s - cooling.ambient, "losses_at_estimated_temperatures": None, "cooling": cooling.to_json(),
            "note": "losses evaluated at the design temperature; surface temperature is a lower bound for the hot spots"}


# -------------------------------------------------------------------------------------------------
# candidate evaluation
# -------------------------------------------------------------------------------------------------
def _conductor_options(request: WaveformInformedTransformerRequest, i1: float, i2: float) -> list[tuple[str, ConductorSpec, ConductorSpec, dict[str, Any]]]:
    opts = request.windings
    out = []
    for kind in opts.conductors:
        if kind == "litz":
            for d in opts.litz_strand_diameters:
                for j in opts.current_densities:
                    out.append((kind, litz_for_area(i1 / j.value, d.value), litz_for_area(i2 / j.value, d.value),
                                {"current_density_A_per_mm2": j.value / 1e6, "strand_diameter_mm": d.value * 1e3}))
        else:
            for t in opts.foil_thicknesses:
                out.append((kind, ConductorSpec(kind="foil", thickness=t.value), ConductorSpec(kind="foil", thickness=t.value * i2 / i1),
                            {"foil_thickness_mm": t.value * 1e3}))
    return out


def evaluate_candidate(*, request: WaveformInformedTransformerRequest, excitation: TransformerExcitation, material: MaterialContext, core: CoreData,
                       n1: int, n2: int, kind: str, cond_p: ConductorSpec, cond_s: ConductorSpec, arrangement: str, tags: dict[str, Any],
                       core_loss_density: Any, bsat: float, b_allow: float) -> dict[str, Any]:
    f = excitation.frequency
    t_design = request.temperature.value
    ae, ve = core.effective_area, core.effective_volume
    assert ae is not None and ve is not None
    lam_pp = excitation.flux_linkage_peak_to_peak
    b_pk = lam_pp / (2.0 * n1 * ae)
    b_max = max(abs(v) for _, v in excitation.flux_linkage.points) / (n1 * ae)
    row: dict[str, Any] = {
        "id": f"{core.name.replace(' ', '')}|{material.data.grade or material.id}|{n1}:{n2}|{cond_p.describe()}/{cond_s.describe()}|{arrangement}",
        "core": core.id, "core_name": core.name, "material": material.id, "material_grade": material.data.grade, "turns": {"primary": n1, "secondary": n2},
        "turns_ratio": n1 / n2, "turns_ratio_target": request.turns_ratio_target, "turns_ratio_error": (n1 / n2) / request.turns_ratio_target - 1.0,
        "conductor": kind, "arrangement": arrangement, "tags": tags, "feasible": True, "evaluation": "complete", "violations": [], "violation_codes": [],
        "warnings": [],
        "standard_size_equivalent": material.id not in core.material_refs,
        "flux_density_peak_T": b_pk, "flux_density_max_T": b_max, "flux_density_allowed_T": b_allow, "saturation_flux_density_T": bsat,
        "saturation_margin": 1.0 - b_max / bsat, "flux_linkage_peak_to_peak_Vs": lam_pp,
    }

    def violate(code: str, message: str) -> None:
        row["violations"].append(message)
        row["violation_codes"].append(code)

    if abs(row["turns_ratio_error"]) > request.turns_ratio_tolerance + 1e-9:
        violate("turns_ratio", f"turns ratio {n1}:{n2} deviates {row['turns_ratio_error']:+.1%} from the target")
    if b_max >= bsat:
        violate("saturation", f"saturation: B_max {b_max:.3f} T >= Bsat {bsat:.3f} T")
    elif b_max > b_allow * (1 + 1e-9):
        violate("flux_limit", f"flux limit: B_max {b_max:.3f} T > allowed {b_allow:.3f} T")
    # core loss at the design temperature (actual flux waveform)
    pv, pv_notes = core_loss_density(n1, t_design)
    row["loss_density_W_per_m3"] = pv
    row["core_loss_W"] = pv * ve
    row["core_loss_notes"] = pv_notes
    if request.loss_density_target is not None and pv > request.loss_density_target.value * (1 + 1e-9):
        violate("core_loss_density_target", f"core loss density {pv / 1e3:.0f} kW/m^3 exceeds the target {request.loss_density_target.value / 1e3:.0f} kW/m^3")
    # layout
    try:
        layout = build_layout(core, material.id, n1, n2, cond_p, cond_s, arrangement, request.isolation.creepage_margin.value,
                              request.isolation.interwinding_insulation.value)
    except (AIPEMagneticsError, ValueError) as error:
        row.update(feasible=False, evaluation="incomplete")
        violate("layout", f"layout: {getattr(error, 'message', str(error))}")
        return row
    row["layout"] = {"fidelity": layout.fidelity, "checks": layout.checks, "notes": layout.notes, "winding_height_m": layout.winding_height,
                     "layers": [{"winding": l.winding, "section": l.section, "turns": l.turns, "d_in_m": l.d_in, "d_out_m": l.d_out,
                                 "mean_turn_length_m": l.mean_turn_length} for l in layout.layers], **layout.derived}
    for check in layout.checks:
        if not check["passed"]:
            violate("layout_" + check["code"].split(".")[-1], f"layout: {check['message']}")
    if not layout.feasible:
        row["feasible"] = False
    window_area = core.bobbin_window_area or core.window_area
    copper_area = n1 * cond_p.copper_area(layout.winding_height) + n2 * cond_s.copper_area(layout.winding_height)
    row["window_fill"] = copper_area / window_area if window_area else None
    if row["window_fill"] is not None and row["window_fill"] > request.windings.max_window_fill:
        violate("window_fill", f"window fill {row['window_fill']:.2f} > {request.windings.max_window_fill:.2f} (bare copper / coil-former window)")
    # winding losses at the design temperature (each winding with its own harmonics)
    wl = winding_losses(layout, excitation, n1, n2, t_design, request.harmonics)
    row["windings"] = wl
    row["copper_loss_W"] = sum(v["loss_W"] for v in wl.values())
    row["total_loss_W"] = row["core_loss_W"] + row["copper_loss_W"]
    j_max = request.windings.max_current_density.value
    for w, v in wl.items():
        if v["current_density_A_per_m2"] > j_max * (1 + 1e-9):
            violate("current_density", f"{w}: current density {v['current_density_A_per_m2'] / 1e6:.2f} A/mm^2 > {j_max / 1e6:.1f} A/mm^2")
        if v["effective_ac_factor"] is not None and v["effective_ac_factor"] > 3:
            row["warnings"].append(f"{w}: effective AC/DC resistance factor {v['effective_ac_factor']:.2f} is high")
    # leakage, external inductance, magnetising inductance
    leak = leakage_inductance(layout, n1, n2)
    row["leakage_inductance_H"] = leak
    total = excitation.series_inductance_total
    row["series_inductance_total_H"] = total
    row["external_inductance_required_H"] = None if total is None else total - leak
    if total is not None and leak > total:
        violate("leakage_exceeds_series_inductance", f"leakage {leak * 1e6:.1f} uH exceeds the total series inductance {total * 1e6:.1f} uH")
    lm = magnetising_inductance(core, material, n1)
    row["magnetising_inductance"] = lm
    l_m = lm.get("reluctance_model_H")
    if l_m:
        row["magnetising_current_peak_A"] = 0.5 * lam_pp / l_m
        row["magnetising_current_share_of_primary_peak"] = row["magnetising_current_peak_A"] / excitation.primary_current.peak()
    # mass and volume
    copper_mass = COPPER_DENSITY * sum(l.turns * l.mean_turn_length * l.conductor.copper_area(layout.winding_height) for l in layout.layers)
    core_mass = core.mass if core.mass is not None else (material.data.density * ve if material.data.density else None)
    row["mass_kg"] = None if core_mass is None else core_mass + copper_mass
    row["copper_mass_kg"] = copper_mass
    row["core_volume_m3"] = ve
    # thermal
    try:
        th = thermal_estimate(layout, core, material, request, excitation, n1, n2, lambda t: core_loss_density(n1, t)[0] * ve, t_design)
    except AIPEMagneticsError as error:
        th = {"method": "not_evaluated", "reason": error.message}
    row["thermal"] = th
    if th.get("method") == "not_evaluated":
        row.update(feasible=False, evaluation="incomplete")
        violate("thermal_not_evaluated", f"thermal estimate not possible: {th.get('reason')}")
    else:
        row["temperature_rise_K"] = th["temperature_rise_K"]
        if not th.get("converged", True):
            violate("electrothermal_convergence", "electrothermal iteration did not converge")
        if th["hotspot_winding_K"] > request.thermal.max_winding_hotspot.value:
            violate("winding_hotspot", f"winding hot spot {th['hotspot_winding_K'] - 273.15:.0f} degC > {request.thermal.max_winding_hotspot.value - 273.15:.0f} degC")
        if th["hotspot_core_K"] > request.thermal.max_core_hotspot.value:
            violate("core_hotspot", f"core hot spot {th['hotspot_core_K'] - 273.15:.0f} degC > {request.thermal.max_core_hotspot.value - 273.15:.0f} degC")
        if th.get("method") == "thermal_network_electrothermal" and th["temperatures_K"]["core"] > t_design + 5:
            row["warnings"].append(f"estimated core temperature {th['temperatures_K']['core'] - 273.15:.0f} degC exceeds the design temperature "
                                   f"{t_design - 273.15:.0f} degC at which the losses were evaluated")
    if row["violations"]:
        row["feasible"] = False
    return row


# -------------------------------------------------------------------------------------------------
# comparison with the square-current approximation
# -------------------------------------------------------------------------------------------------
def square_current_comparison(best: dict[str, Any], excitation: TransformerExcitation, request: WaveformInformedTransformerRequest,
                              material: MaterialContext, core: CoreData, cond_p: ConductorSpec, cond_s: ConductorSpec) -> dict[str, Any]:
    """Same geometry, square P/V currents (v0.2 assumption) and Dowell at the fundamental, plus the public design_transformer."""
    power = excitation.throughput_power()
    n1, n2 = best["turns"]["primary"], best["turns"]["secondary"]
    out: dict[str, Any] = {"throughput_power_W": power}
    v1 = request.primary_voltage.value if request.primary_voltage is not None else (excitation.primary_voltage.peak() if excitation.primary_voltage else None)
    if power is None or v1 is None:
        out["note"] = "throughput power or primary voltage unknown; square-current comparison not possible"
        return out
    i1_sq, i2_sq = power / v1, power / v1 * request.turns_ratio_target
    i1, i2 = excitation.primary_current.rms(), excitation.secondary_current.rms()
    layout = build_layout(core, material.id, n1, n2, cond_p, cond_s, best["arrangement"], request.isolation.creepage_margin.value,
                          request.isolation.interwinding_insulation.value)
    t = request.temperature.value
    actual = best["windings"]
    # v0.2 method on the same geometry: Dowell at the fundamental (strand-equivalent layers for litz) with the P/V RMS current,
    # then the same with the actual RMS (isolates the RMS effect from the harmonic / MMF-profile effect)
    rho = F.copper_resistivity(t)
    delta = F.skin_depth(excitation.frequency, rho)
    fundamental_square: dict[str, float] = {}
    fundamental_actual: dict[str, float] = {}
    factors: dict[str, float] = {}
    for w, cond, i_sq, i_act in (("primary", cond_p, i1_sq, i1), ("secondary", cond_s, i2_sq, i2)):
        layers = [l for l in layout.layers if l.winding == w]
        r_dc = sum(rho * l.turns * l.mean_turn_length / cond.copper_area(layout.winding_height) for l in layers)
        if cond.kind == "litz":
            strand_area = math.pi * cond.strand_diameter ** 2 / 4
            porosity = max(0.1, min(1.0, cond.strand_count * strand_area / (math.pi * cond.envelope_diameter() ** 2 / 4)))
            m = max(1, int(round(len(layers) * math.sqrt(cond.strand_count))))
            fr = F.dowell_ac_resistance_factor(m, F.round_wire_equivalent_thickness(cond.strand_diameter), delta, porosity)
        else:
            fr = F.dowell_ac_resistance_factor(len(layers), cond.thickness, delta, 1.0)
        factors[w] = fr
        fundamental_square[w] = r_dc * fr * i_sq ** 2
        fundamental_actual[w] = r_dc * fr * i_act ** 2
    out.update({
        "square_currents": {"primary_rms_A": i1_sq, "secondary_rms_A": i2_sq, "basis": "I_rms = P / V (public design_transformer assumption)"},
        "actual_currents": {"primary_rms_A": i1, "secondary_rms_A": i2, "primary_peak_A": excitation.primary_current.peak(),
                            "secondary_peak_A": excitation.secondary_current.peak()},
        "rms_ratio_actual_over_square": {"primary": i1 / i1_sq, "secondary": i2 / i2_sq},
        "dowell_factor_at_fundamental": factors,
        "copper_loss_W": {"square_rms_dowell_at_fundamental_v02_method": fundamental_square,
                          "actual_rms_dowell_at_fundamental": fundamental_actual,
                          "actual_waveform_per_layer_harmonic_dowell": {w: v["loss_W"] for w, v in actual.items()}},
        "copper_loss_total_W": {"square_rms_dowell_at_fundamental_v02_method": sum(fundamental_square.values()),
                                "actual_rms_dowell_at_fundamental": sum(fundamental_actual.values()),
                                "actual_waveform_per_layer_harmonic_dowell": sum(v["loss_W"] for v in actual.values())},
        "note": "same core, turns, conductors and arrangement for all three columns; the square approximation defines only an RMS value, "
                "so no harmonic content is attributed to it",
    })
    # the public v0.2 capability on the same core / material / conductor
    try:
        from aipe_magnetics.engineering.transformer import design_transformer
        from aipe_magnetics.api.types import TransformerDesignRequest
        winding = {"conductor_type": cond_p.kind, "current_density": {"value": best["tags"].get("current_density_A_per_mm2", 4.0), "unit": "A/mm^2"}}
        if cond_p.kind == "litz":
            winding["strand_diameter"] = {"value": cond_p.strand_diameter, "unit": "m"}
        else:
            winding["foil_thickness"] = {"value": cond_p.thickness, "unit": "m"}
        req = TransformerDesignRequest.model_validate({
            "power": {"value": power, "unit": "W"}, "frequency": {"value": excitation.frequency, "unit": "Hz"},
            "primary": {"name": "primary", "voltage": {"value": v1, "unit": "V"}, "winding": winding},
            "secondaries": [{"name": "secondary", "voltage": {"value": v1 / request.turns_ratio_target, "unit": "V"}, "winding": winding}],
            "voltage_waveform": "square", "duty": 0.5, "material": material.id, "cores": [core.id], "temperature": {"value": t, "unit": "K"},
            "flux_limit": request.flux_limit.model_dump(mode="json", exclude_none=True),
            **({"loss_density_target": {"value": request.loss_density_target.value, "unit": "W/m^3"}} if request.loss_density_target is not None else {}),
            "policy": "warn"})
        v02 = design_transformer(req)
        if v02.status == "ok":
            out["public_design_transformer"] = {"status": "ok", "turns": {k[len("turns_"):]: v.quantity.value for k, v in v02.values.items() if k.startswith("turns_")},
                                                "flux_density_peak_T": v02.value("flux_density_peak").value, "core_loss_W": v02.value("core_loss").value,
                                                "copper_loss_W": v02.value("copper_loss").value, "total_loss_W": v02.value("total_loss").value,
                                                "window_fill": v02.value("window_fill").value,
                                                "current_basis": "P/V square approximation, Dowell at the fundamental, datasheet mean turn length"}
        else:
            out["public_design_transformer"] = {"status": v02.status, "errors": [e.message for e in v02.errors]}
    except AIPEMagneticsError as error:
        out["public_design_transformer"] = {"status": "error", "message": error.message}
    return out


# -------------------------------------------------------------------------------------------------
# material comparison for this excitation
# -------------------------------------------------------------------------------------------------
def material_comparison(materials: list[MaterialContext], core: CoreData, excitation: TransformerExcitation, request: WaveformInformedTransformerRequest,
                        n1: int) -> list[dict[str, Any]]:
    """Every catalogue material in the reference core with the selected turns: loss density, saturation margin, missing data."""
    f, t = excitation.frequency, request.temperature.value
    shape = excitation.flux_shape()
    ae, ve = core.effective_area, core.effective_volume
    rows = []
    for m in materials:
        b_pk = excitation.flux_linkage_peak_to_peak / (2.0 * n1 * ae) if ae else None
        row: dict[str, Any] = {"material": m.id, "grade": m.data.grade, "manufacturer_ref": m.data.properties().manufacturer_ref,
                               "evaluable_for_design": m.evaluable, "reasons_not_evaluable": list(m.reasons), "missing_data": m.missing(),
                               "bsat_T": m.bsat, "bsat_status": m.bsat_status, "flux_density_peak_T": b_pk,
                               "saturation_margin": None if (m.bsat is None or b_pk is None) else 1.0 - b_pk / m.bsat,
                               "minimum_primary_turns": None, "loss_density_W_per_m3": None, "core_loss_W": None, "loss_notes": [],
                               "offered_in_reference_core": m.id in core.material_refs}
        if m.bsat is not None and ae:
            limit = request.flux_limit.fraction * m.bsat if request.flux_limit.mode == "fraction_of_saturation" else request.flux_limit.absolute.value  # type: ignore[union-attr]
            row["minimum_primary_turns"] = int(math.ceil(excitation.flux_linkage_peak_to_peak / (2.0 * ae * limit) - 1e-9))
        if m.evaluator is not None and b_pk:
            try:
                ev = m.evaluator.loss_density(f, b_pk, t, shape, policy="warn")
                row.update({"loss_density_W_per_m3": float(ev.loss_density), "core_loss_W": float(ev.loss_density) * (ve or 0.0),
                            "parameter_sets": ev.sets, "interpolated": ev.interpolated, "loss_notes": list(ev.warnings)})
            except AIPEMagneticsError as error:
                row["loss_notes"].append(error.message)
        rows.append(row)
    rows.sort(key=lambda r: (r["loss_density_W_per_m3"] is None, r["loss_density_W_per_m3"] or 0.0))
    return rows


# -------------------------------------------------------------------------------------------------
# capability
# -------------------------------------------------------------------------------------------------
def design_transformer_from_waveforms(request: WaveformInformedTransformerRequest, *, created_at: str | None = None,
                                      export_state: bool = False) -> EngineeringResult:
    inputs = request.model_dump(mode="json", exclude_none=True)
    if export_state and not created_at:
        raise InvalidInputError("export_state needs created_at (RFC 3339 timestamp supplied by the caller)")
    try:
        excitation = build_excitation(request)
    except AIPEMagneticsError as error:
        return EngineeringResult.rejected(CAP_WAVEFORM_DESIGN, error, inputs=inputs)
    result = EngineeringResult(capability=CAP_WAVEFORM_DESIGN, inputs=inputs, produced_at=created_at)
    f = excitation.frequency
    t_design = request.temperature.value
    for code, message in excitation.warnings:
        result.warn(code, message)
    for aid, statement, basis in excitation.assumptions:
        result.assume(aid, statement, basis, ["copper_loss", "core_loss"])  # type: ignore[arg-type]
    result.data["excitation"] = excitation.describe()
    try:
        materials = material_contexts(request, f, t_design)
        evaluable = [m for m in materials if m.evaluable]
        cores = core_candidates(request, evaluable)
    except AIPEMagneticsError as error:
        return EngineeringResult.rejected(CAP_WAVEFORM_DESIGN, error, inputs=inputs)
    result.data["materials_considered"] = [{"material": m.id, "grade": m.data.grade, "evaluable": m.evaluable, "reasons": m.reasons, "loss_notes": m.loss_notes,
                                 "bsat_T": m.bsat, "bsat_status": m.bsat_status} for m in materials]
    for m in evaluable:
        for s in m.data.sources:
            result.cite(s)
    if not evaluable:
        result.status = "rejected"
        result.errors.append(ErrorRecord(code="no_evaluable_material", message="No candidate material has saturation data and a loss parameter set "
                                         "covering the operating point; see data.materials_considered for the reasons.",
                                         details={"materials": {m.id: m.reasons for m in materials}}))
        return result
    if not cores:
        result.status = "rejected"
        result.errors.append(ErrorRecord(code="no_candidate_core", message="No catalogue core is offered in the evaluable material(s); pass cores= or "
                                         "material_offer_policy='standard_size_equivalent'."))
        return result
    # policy assumptions
    from aipe_magnetics.engineering.saturation import flux_limit_from_policy
    result.assume("zero_dc_flux", "The flux linkage integral is detrended and taken with zero mean (volt-second balanced operation); iGSE ignores any DC "
                  "flux component.", "model_assumption", ["core_loss", "flux_density_peak"])
    result.assume("residual_gap", f"Magnetising inductance uses an ungapped core set with a residual mating gap of {RESIDUAL_GAP_PER_LEG * 1e6:.0f} um per leg.",
                  "engineering_judgement", ["magnetising_inductance"])
    cooling = _cooling(request)
    result.assume("cooling", f"Thermal estimate with {cooling.describe()}; limits: winding hot spot <= {request.thermal.max_winding_hotspot.value - 273.15:.0f} degC, "
                  f"core <= {request.thermal.max_core_hotspot.value - 273.15:.0f} degC.", "user_input" if request.thermal.heat_transfer_coefficient is not None
                  else "default", ["temperature_rise"])
    result.assume("isolation_margins", f"Creepage margin {request.isolation.creepage_margin.value * 1e3:.2f} mm at every layer end and "
                  f"{request.isolation.interwinding_insulation.value * 1e3:.2f} mm between winding sections; the insulation system itself is not designed.",
                  "default" if request.isolation.working_voltage is None else "user_input", ["leakage_inductance", "window_fill"])
    i1, i2 = excitation.primary_current.rms(), excitation.secondary_current.rms()
    options = _conductor_options(request, i1, i2)
    shape = excitation.flux_shape()
    candidates: list[dict[str, Any]] = []
    loss_cache: dict[tuple[str, str, int, float], tuple[float, list[str]]] = {}
    for material in evaluable:
        assert material.bsat is not None and material.evaluator is not None
        b_policy, flux_assumption = flux_limit_from_policy(request.flux_limit, material.bsat)
        result.assume(flux_assumption.id, flux_assumption.statement, flux_assumption.basis, ["turns_primary", "flux_density_peak"])
        if material.bsat_status == "estimated":
            result.assume("bsat_interpolated", f"Bsat at {t_design - 273.15:.0f} degC linearly interpolated between datasheet points.", "engineering_judgement")
        for core in cores:
            if request.material_offer_policy == "catalogue" and not request.cores and material.id not in core.material_refs:
                continue
            ae, ve = core.effective_area, core.effective_volume
            if ae is None or ve is None:
                candidates.append({"id": f"{core.name}|{material.data.grade}", "core": core.id, "core_name": core.name, "material": material.id, "feasible": False,
                                   "evaluation": "incomplete", "violations": ["core record lacks effective area / volume"], "violation_codes": ["incomplete_evaluation"], "warnings": []})
                continue
            if core.mean_turn_length is None or (core.bobbin_window_area or core.window_area) is None:
                candidates.append({"id": f"{core.name}|{material.data.grade}", "core": core.id, "core_name": core.name, "material": material.id, "feasible": False,
                                   "evaluation": "incomplete", "violations": ["not evaluable: core record has no coil-former data (mean turn length / window area)"],
                                   "violation_codes": ["incomplete_evaluation"], "warnings": []})
                continue
            evaluator = material.evaluator

            def core_loss_density(n1: int, t: float, _ev=evaluator, _ae=ae, _mid=material.id) -> tuple[float, list[str]]:
                key = (_mid, core.id, n1, round(t, 3))
                if key not in loss_cache:
                    b = excitation.flux_linkage_peak_to_peak / (2.0 * n1 * _ae)
                    ev = _ev.loss_density(f, b, t, shape, policy="warn")
                    loss_cache[key] = (float(ev.loss_density), sorted(set(ev.warnings)) + ([f"temperature interpolation between {ev.sets}"] if ev.interpolated else []))
                return loss_cache[key]

            n1_min = int(math.ceil(excitation.flux_linkage_peak_to_peak / (2.0 * ae * b_policy) - 1e-9))
            n1_min = max(1, n1_min)
            if request.loss_density_target is not None:
                n = n1_min
                while n < MAX_TURN_SEARCH and core_loss_density(n, t_design)[0] > request.loss_density_target.value * (1 + 1e-9):
                    n += 1
                if n > n1_min:
                    result.assume("loss_density_target", f"Primary turns raised so that the core loss density stays below "
                                  f"{request.loss_density_target.value / 1e3:.0f} kW/m^3 (user ceiling).", "user_input", ["turns_primary"])
                n1_min = n
            for n1 in range(n1_min, n1_min + request.turn_steps + 1):
                n2 = max(1, int(round(n1 / request.turns_ratio_target)))
                if abs((n1 / n2) / request.turns_ratio_target - 1.0) > request.turns_ratio_tolerance + 1e-9:
                    continue
                for kind, cond_p, cond_s, tags in options:
                    for arrangement in request.windings.arrangements:
                        try:
                            row = evaluate_candidate(request=request, excitation=excitation, material=material, core=core, n1=n1, n2=n2, kind=kind,
                                                     cond_p=cond_p, cond_s=cond_s, arrangement=arrangement, tags=tags, core_loss_density=core_loss_density,
                                                     bsat=material.bsat, b_allow=b_policy)
                        except AIPEMagneticsError as error:
                            row = {"id": f"{core.name}|{material.data.grade}|{n1}:{n2}|{kind}|{arrangement}", "core": core.id, "core_name": core.name,
                                   "material": material.id, "feasible": False, "evaluation": "incomplete", "violations": [error.message],
                                   "violation_codes": ["incomplete_evaluation"], "warnings": []}
                        candidates.append(row)
    feasible = [c for c in candidates if c["feasible"] and c.get("total_loss_W") is not None]
    feasible.sort(key=lambda c: c["total_loss_W"])
    infeasible = [c for c in candidates if not c["feasible"]]
    result.data["candidates"] = feasible[: request.max_candidates]
    result.data["candidate_counts"] = {"evaluated": len(candidates), "feasible": len(feasible), "infeasible": len(infeasible)}
    reasons: dict[str, int] = {}
    for c in infeasible:
        for code in c.get("violation_codes") or ["incomplete_evaluation"]:
            reasons[code] = reasons.get(code, 0) + 1
    result.data["infeasible_summary"] = [{"reason": k, "candidates": v} for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])]
    result.data["infeasible_examples"] = [{"id": c["id"], "violations": c["violations"]} for c in infeasible[:12]]
    best_by_core: dict[str, dict[str, Any]] = {}
    for c in feasible:
        best_by_core.setdefault(c["core"], c)
    result.data["best_by_core"] = [{"core": k, "id": v["id"], "turns": v["turns"], "total_loss_W": v["total_loss_W"], "temperature_rise_K": v.get("temperature_rise_K")}
                                   for k, v in best_by_core.items()]
    result.data["conductor_options"] = [{"kind": k, "primary": p.describe(), "secondary": s.describe(), **t} for k, p, s, t in options]
    _limitations(result, excitation)
    if not feasible:
        result.status = "rejected"
        shown = "; ".join(f"{r['reason']} ({r['candidates']})" for r in result.data["infeasible_summary"][:6])
        result.errors.append(ErrorRecord(code="no_feasible_candidate", message="No candidate satisfied the saturation, ratio, fill, current-density, "
                                         f"leakage and thermal constraints with a complete evaluation. Reasons: {shown}",
                                         details={"infeasible_summary": result.data["infeasible_summary"]}))
        return result
    best = feasible[0]
    result.data["selected"] = best["core"]              # core record id (URI reference for the Core export)
    result.data["selected_candidate"] = best["id"]
    result.data["selected_core"] = best["core"]
    result.data["material"] = best["material"]
    result.data["component_kind"] = "transformer"
    for message in best["warnings"]:
        result.warn("selected_candidate", f"{best['core_name']}: {message}")
    for note in best.get("core_loss_notes", []):
        result.warn("core_loss_model", note)
    n1, n2 = best["turns"]["primary"], best["turns"]["secondary"]
    sel_material = next(m for m in evaluable if m.id == best["material"])
    sel_core = next(c for c in cores if c.id == best["core"])
    option = next((o for o in options if o[1].describe() == best["windings"]["primary"]["conductor"] and o[2].describe() == best["windings"]["secondary"]["conductor"]), options[0])
    result.add_value("turns_primary", "Turns of the primary", Quantity(value=float(n1), unit="1"), status="calculated",
                     derived_from=["flux_linkage_peak_to_peak", "effective_area", "flux_density_allowed"])
    result.add_value("turns_secondary", "Turns of the secondary", Quantity(value=float(n2), unit="1"), status="calculated", derived_from=["turns_primary", "turns_ratio_target"])
    result.add_value("turns_ratio", "Realised turns ratio N1/N2", Quantity(value=best["turns_ratio"], unit="1"), status="calculated")
    result.add_value("flux_linkage_peak_to_peak", "Primary flux-linkage swing per cycle", Quantity(value=best["flux_linkage_peak_to_peak_Vs"], unit="V*s"),
                     status="calculated" if excitation.flux_basis == "integral_of_voltage_channel" else "assumed",
                     note=excitation.flux_basis)
    result.add_value("flux_density_peak", "Peak flux density (selected core)", Quantity(value=best["flux_density_peak_T"], unit="T"), status="calculated")
    result.add_value("saturation_flux_density", f"Bsat at {t_design - 273.15:.0f} degC", Quantity(value=best["saturation_flux_density_T"], unit="T"),
                     status=sel_material.bsat_status or "looked_up")
    result.add_value("saturation_margin", "Margin to saturation (1 - B_max / Bsat)", Quantity(value=best["saturation_margin"], unit="1"), status="calculated")
    result.add_value("loss_density", "Core loss density (iGSE on the actual flux waveform, design temperature)", Quantity(value=best["loss_density_W_per_m3"], unit="W/m^3"), status="estimated")
    result.add_value("core_loss", "Core loss (design temperature)", Quantity(value=best["core_loss_W"], unit="W"), status="estimated")
    result.add_value("copper_loss", "Copper loss, both windings (per-layer harmonic Dowell, design temperature)", Quantity(value=best["copper_loss_W"], unit="W"), status="estimated")
    result.add_value("total_loss", "Total estimated loss (core + copper, design temperature)", Quantity(value=best["total_loss_W"], unit="W"), status="estimated",
                     derived_from=["core_loss", "copper_loss"])
    result.add_value("leakage_inductance", "Leakage inductance attributable to the transformer (1-D window field, referred to the primary)",
                     Quantity(value=best["leakage_inductance_H"], unit="H"), status="estimated")
    if best.get("external_inductance_required_H") is not None:
        result.add_value("external_inductance_required", "External series inductance still required (L_total - L_leak, referred to the primary)",
                         Quantity(value=best["external_inductance_required_H"], unit="H"), status="estimated", derived_from=["leakage_inductance"],
                         note=f"L_total basis: {excitation.series_inductance_basis}")
    if best["magnetising_inductance"].get("reluctance_model_H"):
        result.add_value("magnetising_inductance", "Magnetising inductance (reluctance model with residual gap)",
                         Quantity(value=best["magnetising_inductance"]["reluctance_model_H"], unit="H"), status="estimated")
    if best["magnetising_inductance"].get("datasheet_AL_H"):
        result.add_value("magnetising_inductance_datasheet_al", "Magnetising inductance from the datasheet AL value (ungapped set)",
                         Quantity(value=best["magnetising_inductance"]["datasheet_AL_H"], unit="H"), status="looked_up")
    result.add_value("window_fill", "Window fill (bare copper / coil-former window)", Quantity(value=best["window_fill"], unit="1"), status="estimated")
    if best.get("temperature_rise_K") is not None:
        result.add_value("temperature_rise", "Hot-spot temperature rise above ambient (thermal estimate)", Quantity(value=best["temperature_rise_K"], unit="K"),
                         status="estimated", note=best["thermal"]["method"])
        result.add_value("hotspot_winding", "Winding hot-spot temperature estimate", Quantity(value=best["thermal"]["hotspot_winding_K"], unit="K"), status="estimated")
        result.add_value("hotspot_core", "Core hot-spot temperature estimate", Quantity(value=best["thermal"]["hotspot_core_K"], unit="K"), status="estimated")
    status_i = "specified" if excitation.basis == "imported" else "assumed"
    result.add_value("current_rms_primary", "Primary RMS current (from the waveform)", Quantity(value=i1, unit="A"), status=status_i, note=excitation.basis)
    result.add_value("current_rms_secondary", "Secondary RMS current (from the waveform)", Quantity(value=i2, unit="A"), status=status_i, note=excitation.secondary_basis)
    if best.get("mass_kg") is not None:
        result.add_value("mass", "Core + copper mass", Quantity(value=best["mass_kg"], unit="kg"), status="estimated")
    result.data["square_current_comparison"] = square_current_comparison(best, excitation, request, sel_material, sel_core, option[1], option[2])
    ref_core = sel_core
    if request.reference_core:
        try:
            ref_core = core_data(request.reference_core)
        except AIPEMagneticsError as error:
            result.warn("reference_core", f"reference core not used: {error.message}")
    # every non-synthetic catalogue material is compared in the reference core, whatever the request restricted the design to
    try:
        all_materials = materials if request.materials is None else material_contexts(request.model_copy(update={"materials": None}), f, t_design)
    except AIPEMagneticsError as error:  # pragma: no cover - the bundled catalogue always loads
        result.warn("material_comparison", f"catalogue-wide comparison not possible: {error.message}")
        all_materials = materials
    result.data["material_comparison"] = {"reference_core": ref_core.id, "primary_turns": n1, "flux_waveform": "actual (imported / synthesised) flux shape",
                                          "temperature_K": t_design, "frequency_Hz": f, "rows": material_comparison(all_materials, ref_core, excitation, request, n1)}
    # measurement requests for the materials considered, with the selected winding as context
    try:
        from aipe_magnetics.api.types import MeasurementRequestSpec
        from aipe_magnetics.engineering.measurement_requests import measurement_requests as _requests
        spec = MeasurementRequestSpec(materials=[m.id for m in all_materials], frequency=Quantity(value=f, unit="Hz"),
                                      flux_density_peak=Quantity(value=best["flux_density_peak_T"], unit="T"), temperature=Quantity(value=t_design, unit="K"),
                                      excitation="piecewise_linear", harmonics=min(request.harmonics, 11), include_available=False)
        context = {"conductor": {w: v["conductor"] for w, v in best["windings"].items()}, "turns": best["turns"], "core": best["core"],
                   "arrangement": best["arrangement"], "winding_temperature_degC": round(t_design - 273.15, 1),
                   "current_harmonics_primary_A": [{"n": n, "amplitude": a} for n, a in excitation.harmonics("primary", min(request.harmonics, 11))]}
        requests = _requests(spec, winding_context=context)
        result.data["measurement_requests"] = requests.data if requests.status == "ok" else {"status": requests.status, "errors": [e.message for e in requests.errors]}
    except AIPEMagneticsError as error:  # pragma: no cover - defensive: requests never block the design
        result.warn("measurement_requests", f"measurement requests not generated: {error.message}")
    # model identity and verification
    from aipe_magnetics.models.empirical.igse import IGSEModel
    result.model = IGSEModel.identity
    from aipe_magnetics.engineering._verification import capability_status, combine
    from aipe_magnetics.verification.suite import load_bundled_registry
    registry = load_bundled_registry()
    design_status = capability_status(registry, CAP_WAVEFORM_DESIGN, frequency=f, temperature=t_design, flux_density_peak=best["flux_density_peak_T"], material=best["material"])
    loss_status = registry.status_for(IGSE_MODEL_ID, frequency=f, temperature=t_design, flux_density_peak=best["flux_density_peak_T"], material=best["material"],
                                      excitation="piecewise_linear")
    result.data["loss_model_verification"] = loss_status.model_dump(mode="json")
    result.verification = combine(design_status, loss_status, design_label="Waveform-informed sizing (exact RMS / Fourier, Faraday, per-layer Dowell, 1-D "
                                  "leakage, thermal network)", model_label="Core-loss model (iGSE) of the selected core")
    if export_state:
        try:
            result.data["engineering_state"] = engineering_state_from_waveform_design(result, created_at=created_at)  # type: ignore[arg-type]
        except (AIPEMagneticsError, ValueError) as error:
            result.warn("engineering_state_export", f"AIPE-Core export failed: {getattr(error, 'message', str(error))}", "error")
    return result


def _limitations(result: EngineeringResult, excitation: TransformerExcitation) -> None:
    result.limit("analytical_models", "Analytical models only: 1-D Dowell / 1-D leakage field, lumped or network thermal model, iGSE core loss on the "
                 "volume-average flux; no finite-element, fringing, insulation-system or manufacturing analysis.")
    result.limit("igse_scope", "iGSE uses sinusoidal-fit Steinmetz parameters on the actual flux waveform; DC bias, minor loops and relaxation losses are "
                 "not modelled; parameter sets outside their fitted temperature / flux range are flagged, not corrected.")
    result.limit("dowell_litz", "Litz copper loss uses Dowell's layer model on strand sub-layers (bundle layers x sqrt(strands)) with the MMF profile of the "
                 "arrangement; strand twisting, bundle-level eddy currents and end-turn fields are not modelled.")
    result.limit("thermal_estimate", "Temperature rise comes from a steady-state thermal network (E cores) or a lumped surface estimate (other outlines) with "
                 "generic class thermal conductivities and the stated cooling assumption; no CFD, no transient, no hardware validation.")
    result.limit("waveform_scope", f"Results hold for the supplied operating point only ({excitation.basis}); other voltages, loads, phase shifts or "
                 "modulation schemes change the currents and the flux and must be evaluated separately.")
    result.limit("no_hardware_validation", "No hardware was built or measured; results are design estimates, not qualification data.")
    if excitation.basis == "imported":
        result.limit("imported_waveforms", "The imported waveforms are taken as given (one steady-state period, no smoothing); simulation idealisations of "
                     "the source (ideal switches, no parasitics) carry over into every estimate.")


# -------------------------------------------------------------------------------------------------
# AIPE-Core Engineering State export with waveform-source evidence
# -------------------------------------------------------------------------------------------------
def engineering_state_from_waveform_design(result: EngineeringResult, *, created_at: str, project_id: str = "project.dab-waveform-informed-design",
                                           project_name: str = "AIPE Magnetics: waveform-informed DAB transformer design",
                                           converter_topology: str = "dual-active-bridge", role: str = "DAB isolation transformer") -> dict[str, Any]:
    """Core Engineering State 0.1.0 of the selected design; its analytical evidence derives from the waveform source evidence (path + SHA-256)."""
    from aipe_magnetics.integrations.aipe_core.bridge import component_summary_from_design
    from aipe_magnetics.integrations.aipe_core.export import CoreEvidence, CoreValidationCheck, DesignRequirements, to_engineering_state

    if result.status != "ok":
        raise InvalidInputError(f"cannot export a {result.status} design result")
    summary = component_summary_from_design(result, role=role, created_at=created_at, component_id="magnetic.dab-waveform-informed-transformer")
    excitation = result.data.get("excitation", {})
    source = excitation.get("source", {})
    tool = {"id": TOOL_ID, "version": __version__, "name": "AIPE Magnetics"}
    if source.get("kind") == "sps":
        wave_evidence = CoreEvidence(id="evidence.magnetic-dab-waveform-informed-transformer-waveform-sps", kind="assumption",
                                     description=(f"Transformer waveforms synthesised with the lossless single-phase-shift DAB model: P = {source.get('power_W'):g} W, "
                                                  f"V1 = {source.get('v1_V'):g} V, V2 = {source.get('v2_V'):g} V, phase shift {math.degrees(source.get('phase_shift_rad', 0)):.2f} deg, "
                                                  f"series inductance {source.get('series_inductance_H', 0) * 1e6:.2f} uH (analytical, not a measurement)."),
                                     source={"type": "document", "reference": "aipe_magnetics.design.transformers.dab (SPS model, De Doncker 1991 / Kheraluwala 1992)"},
                                     provenance={"created_at": created_at, "activity": "Synthesise the design excitation", "inputs": [], "tool": tool})
    else:
        channels = source.get("channels", {})
        locator = "channels " + ", ".join(f"{k}={v}" for k, v in channels.items() if v)
        if source.get("window_s"):
            locator += f"; window {source['window_s'][0]:g}..{source['window_s'][1]:g} s"
        if source.get("period_extracted_s"):
            locator += f"; period {source['period_extracted_s'][0]:g}..{source['period_extracted_s'][1]:g} s"
        artifacts = []
        if source.get("uri") and source.get("sha256"):
            artifacts.append({"uri": source["uri"], "media_type": source.get("media_type", "application/octet-stream"), "sha256": source["sha256"],
                              "description": f"converter waveform record ({source.get('kind')}); SHA-256 of the file bytes"})
        wave_evidence = CoreEvidence(id="evidence.magnetic-dab-waveform-informed-transformer-waveform-source", kind="source",
                                     description=(f"Converter waveform record used as the design excitation ({source.get('kind')} file "
                                                  f"{source.get('path') or 'inline document'}, SHA-256 {source.get('sha256')}); one steady-state period of the "
                                                  "primary / secondary currents and the primary voltage. The record is a circuit simulation or measurement "
                                                  "supplied by the caller; this tool did not produce it."),
                                     source={"type": "document", "reference": source.get("path") or "inline JSON waveform document", "locator": locator},
                                     artifacts=artifacts, provenance={"created_at": created_at, "activity": "Import the converter waveforms (window, period "
                                                                      "detection, closure, flux-linkage integration)", "inputs": [], "tool": tool})
    evidence = list(summary.evidence)
    evidence.append(wave_evidence)
    for index, record in enumerate(evidence):
        if record.kind == "analytical":
            provenance = record.provenance.model_copy(update={"inputs": [*record.provenance.inputs, wave_evidence.id]})
            evidence[index] = record.model_copy(update={"provenance": provenance})
            analytical_id = record.id
    leakage = result.values.get("leakage_inductance")
    inductance = result.values.get("magnetising_inductance")
    summary = summary.model_copy(update={"evidence": evidence, "leakage_inductance": leakage.quantity if leakage else None,
                                         "inductance": inductance.quantity if inductance else None,
                                         "database_ref": result.data.get("selected_core"), "design_status": "candidate"})
    requirements = None
    hints = excitation.get("metadata_hints", {})
    power = excitation.get("throughput_power_W")
    req_fields: dict[str, Any] = {}
    if power:
        req_fields["nominal_power"] = Quantity(value=float(power), unit="W")
    if isinstance(hints.get("v1"), (int, float)):
        req_fields["input_voltage"] = Quantity(value=float(hints["v1"]), unit="V")
    if isinstance(hints.get("v2"), (int, float)):
        req_fields["output_voltage"] = Quantity(value=float(hints["v2"]), unit="V")
    if source.get("kind") == "sps":
        req_fields.update({"nominal_power": Quantity(value=float(source["power_W"]), unit="W"), "input_voltage": Quantity(value=float(source["v1_V"]), unit="V"),
                           "output_voltage": Quantity(value=float(source["v2_V"]), unit="V")})
    req_fields["switching_frequency"] = Quantity(value=float(excitation.get("frequency_Hz")), unit="Hz")
    requirements = DesignRequirements(**req_fields, evidence=[CoreEvidence(
        id="evidence.magnetic-dab-waveform-informed-transformer-requirements", kind="assumption",
        description="Converter operating point as carried by the waveform source metadata / request (switching frequency, throughput power from the mean of "
                    "v_p i_p, DC voltages when recorded).", source={"type": "user_requirement", "reference": "waveform-informed design request"},
        provenance={"created_at": created_at, "activity": "Record the operating point", "inputs": [], "tool": tool})])
    checks = [CoreValidationCheck(id="check.magnetic-dab-waveform-informed-transformer-analytical", status="pass",
                                  description="Waveform-informed analytical design relations evaluated for the selected candidate (saturation, ratio, fill, "
                                              "leakage, thermal limits satisfied)", evidence_refs=[analytical_id])]
    return to_engineering_state(summary, created_at=created_at, project_id=project_id, project_name=project_name, converter_topology=converter_topology,
                                requirements=requirements, checks=checks)


__all__ = ["CAP_WAVEFORM_DESIGN", "Layout", "MaterialContext", "TransformerExcitation", "build_excitation", "build_layout", "core_candidates",
           "design_transformer_from_waveforms", "engineering_state_from_waveform_design", "evaluate_candidate", "leakage_inductance",
           "magnetising_inductance", "material_comparison", "material_contexts", "square_current_comparison", "thermal_estimate", "winding_losses"]
