"""Verification checks for the engineering layer (closed-form relations and design capabilities).

The design capabilities are checked against independent re-derivations of their closed forms and
against physical consistency requirements, on the bundled N87 catalogue (TDK datasheet parameters)
for the ranged checks. None of these checks compares with measurements; they establish L1/L2
evidence only.
"""

from __future__ import annotations

import json
import math

from scipy.integrate import quad

from aipe_magnetics.common.results import SourceReference
from aipe_magnetics.common.units import MU_0
from aipe_magnetics.models.contract import ValidityRange
from aipe_magnetics.verification.evidence import OperatingRange
from aipe_magnetics.verification.suite import CheckResult, check

CAP_FORMULAS = "aipe.magnetics.engineering.formulas"
CAP_WINDING = "aipe.magnetics.engineering.winding"
CAP_TRANSFORMER = "aipe.magnetics.design_transformer"
CAP_INDUCTOR = "aipe.magnetics.design_inductor"
CAP_VALIDATE = "aipe.magnetics.validate_design"

N87_RANGE = OperatingRange(materials=["material.tdk-n87"], excitation=["triangular", "sinusoidal"],
                           frequency=ValidityRange(min=50e3, max=500e3, unit="Hz"),
                           temperature=ValidityRange(min=368.15, max=378.15, unit="K"),
                           flux_density_peak=ValidityRange(min=0.0, max=0.3, unit="T"),
                           note="bundled N87 record with the TDK 100 degC datasheet-fit Steinmetz set")

DOWELL = SourceReference(id="ref.dowell-1966", kind="publication",
                         reference="P.L. Dowell, 'Effects of eddy currents in transformer windings', Proc. IEE 113(8), 1966, pp. 1387-1394",
                         license="NOASSERTION")
REATTI = SourceReference(id="ref.reatti-kazimierczuk-2002", kind="publication",
                         reference="A. Reatti, M.K. Kazimierczuk, 'Comparison of various methods for calculating the AC resistance of inductors', "
                                   "IEEE Trans. Magn. 38(3), 2002, pp. 1512-1518 (exponential form of Dowell's equation)",
                         license="NOASSERTION")

DEMO_SPEC = {
    "power": {"value": 10, "unit": "kW"}, "frequency": {"value": 100, "unit": "kHz"},
    "primary": {"name": "primary", "voltage": {"value": 800, "unit": "V"}, "winding": {"conductor_type": "litz"}},
    "secondaries": [{"name": "secondary", "voltage": {"value": 400, "unit": "V"}, "winding": {"conductor_type": "litz"}}],
    "voltage_waveform": "square", "duty": 0.5, "material": "N87", "cores": None, "temperature": {"value": 100, "unit": "degC"},
    "flux_limit": {"mode": "fraction_of_saturation", "fraction": 0.7}, "loss_density_target": {"value": 300, "unit": "kW/m^3"},
    "loss_model": "auto", "max_candidates": 10, "policy": "warn",
}


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


# -------------------------------------------------------------------------------------------------
# closed forms
# -------------------------------------------------------------------------------------------------
@check(id="engineering.faraday-relations", capability=CAP_FORMULAS, level="L2", kind="analytical_reference",
       description="Flux-density / turns relations for square (duty), trapezoidal and sinusoidal voltages match independent arithmetic and turn rounding never exceeds the limit.",
       method="Compare formulas.flux_density_peak_square/sine and turns_for_flux_density_* with direct evaluation of V*D/(2 f N Ae) and V_rms/(sqrt(2) pi f N Ae) on a grid.")
def faraday_relations() -> CheckResult:
    from aipe_magnetics.engineering import formulas as F
    worst = 0.0
    violations = 0
    for v in (48.0, 400.0, 800.0):
        for f in (20e3, 100e3, 500e3):
            for ae in (1e-5, 368e-6, 535e-6):
                for duty in (0.2, 0.5):
                    for n in (1, 7, 21):
                        expected = v * duty / (2 * f * n * ae)
                        worst = max(worst, _rel(F.flux_density_peak_square(v, f, n, ae, duty), expected))
                    for blim in (0.05, 0.15, 0.3):
                        n_calc = F.turns_for_flux_density_square(v, f, ae, blim, duty)
                        if F.flux_density_peak_square(v, f, n_calc, ae, duty) > blim * (1 + 1e-12):
                            violations += 1
                        if n_calc > 1 and F.flux_density_peak_square(v, f, n_calc - 1, ae, duty) <= blim * (1 - 1e-9):
                            violations += 1
                for n in (5, 50):
                    expected = v / (math.sqrt(2) * math.pi * f * n * ae)
                    worst = max(worst, _rel(F.flux_density_peak_sine(v, f, n, ae), expected))
    return CheckResult(passed=worst < 1e-12 and violations == 0, metrics={"max_rel_error": worst, "rounding_violations": violations})


@check(id="engineering.gapped-inductor-relations", capability=CAP_FORMULAS, level="L2", kind="analytical_reference",
       description="Gapped-core inductance and gap length are mutual inverses and reduce to the ungapped reluctance limit.",
       method="Round trip gapped_inductance -> air_gap_for_inductance on a grid; compare the zero-gap value with mu0 N^2 Ae mu_r / le; ungapped overshoot raises.")
def gapped_inductor_relations() -> CheckResult:
    from aipe_magnetics.common.errors import InvalidInputError
    from aipe_magnetics.engineering import formulas as F
    worst = 0.0
    for n in (3, 20, 150):
        for ae in (50e-6, 535e-6):
            for g in (0.0, 1e-4, 2e-3):
                le, mur = 0.147, 2200.0
                L = F.gapped_inductance(n, ae, g, le, mur)
                worst = max(worst, abs(F.air_gap_for_inductance(L, n, ae, le, mur) - g))
                if g == 0.0:
                    worst = max(worst, _rel(L, MU_0 * n * n * ae * mur / le))
    try:
        F.air_gap_for_inductance(MU_0 * 20 ** 2 * 535e-6 * 2200 / 0.147 * 1.001, 20, 535e-6, 0.147, 2200)
        rejected = False
    except InvalidInputError:
        rejected = True
    return CheckResult(passed=worst < 1e-12 and rejected, metrics={"max_abs_error": worst, "ungapped_overshoot_rejected": rejected})


@check(id="engineering.dowell-independent-form", capability=CAP_WINDING, level="L2", kind="independent_numerical",
       description="The Dowell AC-resistance factor agrees with the exponential form given by Reatti & Kazimierczuk and with its thin- and thick-conductor asymptotes.",
       method="Evaluate F_R = Delta[(e^{2D}-e^{-2D}+2 sin 2D)/(e^{2D}+e^{-2D}-2 cos 2D) + (2/3)(m^2-1)(e^{D}-e^{-D}-2 sin D)/(e^{D}+e^{-D}+2 cos D)] independently; compare for m in 1..8, Delta in 0.05..6; check asymptotes.",
       reference=REATTI)
def dowell_independent_form() -> CheckResult:
    from aipe_magnetics.engineering import formulas as F

    def reatti(m: int, d: float) -> float:
        a = (math.exp(2 * d) - math.exp(-2 * d) + 2 * math.sin(2 * d)) / (math.exp(2 * d) + math.exp(-2 * d) - 2 * math.cos(2 * d))
        b = (math.exp(d) - math.exp(-d) - 2 * math.sin(d)) / (math.exp(d) + math.exp(-d) + 2 * math.cos(d))
        return d * (a + 2.0 * (m * m - 1) / 3.0 * b)

    skin = 2.365e-4
    worst = 0.0
    for m in range(1, 9):
        for d in (0.05, 0.3, 1.0, 2.5, 6.0):
            worst = max(worst, _rel(F.dowell_ac_resistance_factor(m, d * skin, skin), reatti(m, d)))
    thin = F.dowell_ac_resistance_factor(4, 0.01 * skin, skin)
    thick = F.dowell_ac_resistance_factor(4, 12 * skin, skin)
    thin_ok = abs(thin - 1.0) < 1e-3
    thick_ok = _rel(thick, 12 * (1 + 2 * 15 / 3)) < 2e-3
    porosity_ok = F.dowell_ac_resistance_factor(3, skin, skin, porosity=0.5) < F.dowell_ac_resistance_factor(3, skin, skin)
    return CheckResult(passed=worst < 1e-12 and thin_ok and thick_ok and porosity_ok,
                       metrics={"max_rel_error_vs_exponential_form": worst, "thin_limit": thin, "thick_limit_rel_error": _rel(thick, 12 * 11)})


# -------------------------------------------------------------------------------------------------
# design capabilities on the bundled N87 record
# -------------------------------------------------------------------------------------------------
def _n87_set(material_summary: dict, set_id: str) -> dict:
    for s in material_summary["steinmetz_parameter_sets"]:
        if s["id"] == set_id:
            return s
    raise LookupError(set_id)


def _igse_triangular(k: float, alpha: float, beta: float, bpk: float, f: float) -> float:
    integral = quad(lambda th: abs(math.cos(th)) ** alpha, 0, 2 * math.pi, points=[math.pi / 2, math.pi, 3 * math.pi / 2])[0]
    k_i = k / ((2 * math.pi) ** (alpha - 1) * integral * 2 ** (beta - alpha))
    return k_i * 2 ** (alpha + beta) * bpk ** beta * f ** alpha


@check(id="engineering.transformer-design-self-consistent", capability=CAP_TRANSFORMER, level="L2", kind="physical_consistency",
       description="The 10 kW / 100 kHz DAB transformer design on N87 reproduces by hand: Faraday turns and peak flux density, secondary turns, iGSE triangular core loss from the stored parameter set, copper loss from the reported winding data, window fill and loss totals.",
       method="Run design_transformer with the demonstration specification; recompute every headline value independently from the selected core's datasheet parameters and the N87 100 degC parameter set (closed forms), tolerance 1e-9 relative.",
       range=N87_RANGE)
def transformer_design_self_consistent() -> CheckResult:
    import aipe_magnetics.api as api
    r = api.design_transformer(DEMO_SPEC)
    if r.status != "ok":
        return CheckResult(passed=False, notes=[f"design rejected: {[e.message for e in r.errors]}"])
    sel = next(c for c in r.data["candidates"] if c["core"] == r.data["selected"])
    n1 = int(r.values["turns_primary"].quantity.value)
    n2 = int(r.values["turns_secondary"].quantity.value)
    ae = sel["effective_area_m2"]
    bpk_expected = 800.0 * 0.5 / (2 * 1e5 * n1 * ae)
    errors = {"flux_density_peak": _rel(r.values["flux_density_peak"].quantity.value, bpk_expected)}
    n2_expected = max(1, round(n1 * 400 / 800))
    errors["turns_secondary"] = 0.0 if n2 == n2_expected else 1.0
    allowed = r.values["flux_density_allowed"].quantity.value
    errors["flux_within_allowed"] = 0.0 if bpk_expected <= allowed * (1 + 1e-9) else 1.0
    material = api.get_material("N87").data["material"]
    pset = _n87_set(material, "set.tdk-datasheet-fit-n87-100c")
    pv_expected = _igse_triangular(pset["k"]["value"], pset["alpha"], pset["beta"], bpk_expected, 1e5)
    errors["loss_density"] = _rel(r.values["loss_density"].quantity.value, pv_expected)
    errors["core_loss"] = _rel(r.values["core_loss"].quantity.value, pv_expected * sel["effective_volume_m3"])
    copper = 0.0
    fill_area = 0.0
    for name, w in sel["windings"].items():
        copper += w["current_rms_A"] ** 2 * w["resistance_dc_Ohm"] * w["ac_resistance_factor"]
        fill_area += w["turns"] * w["conductor_area_m2"]
    errors["copper_loss"] = _rel(r.values["copper_loss"].quantity.value, copper)
    errors["total_loss"] = _rel(r.values["total_loss"].quantity.value, r.values["core_loss"].quantity.value + r.values["copper_loss"].quantity.value)
    errors["currents_p_over_v"] = max(_rel(sel["windings"]["primary"]["current_rms_A"], 10000 / 800),
                                      _rel(sel["windings"]["secondary"]["current_rms_A"], 10000 / 400))
    passed = all(v < 1e-9 for v in errors.values()) and r.values["loss_density"].quantity.value <= 300e3 * (1 + 1e-9)
    return CheckResult(passed=passed, metrics={**errors, "selected_core": sel["core"], "turns_primary": n1, "turns_secondary": n2,
                                               "flux_density_peak_T": bpk_expected, "loss_density_W_per_m3": pv_expected})


@check(id="engineering.transformer-power-split", capability=CAP_TRANSFORMER, level="L2", kind="physical_consistency",
       description="With several secondaries the throughput power is shared (equal split by default, explicit power_share otherwise) so that the secondary volt-amperes add up to the rated power.",
       method="Design a 10 kW transformer with 400 V and 200 V secondaries; check I_rms = P_share / V per winding and sum(V I) = P.",
       range=N87_RANGE)
def transformer_power_split() -> CheckResult:
    import aipe_magnetics.api as api
    spec = json.loads(json.dumps(DEMO_SPEC))
    spec["secondaries"] = [{"name": "sec_a", "voltage": {"value": 400, "unit": "V"}, "winding": {"conductor_type": "litz"}},
                           {"name": "sec_b", "voltage": {"value": 200, "unit": "V"}, "winding": {"conductor_type": "litz"}}]
    spec["loss_density_target"] = None
    r = api.design_transformer(spec)
    if r.status != "ok":
        return CheckResult(passed=False, notes=[f"design rejected: {[e.message for e in r.errors]}"])
    currents = r.data["winding_currents"]
    ia, ib = currents["sec_a"]["current_rms_A"], currents["sec_b"]["current_rms_A"]
    va_total = ia * 400 + ib * 200
    return CheckResult(passed=_rel(ia, 5000 / 400) < 1e-9 and _rel(ib, 5000 / 200) < 1e-9 and _rel(va_total, 10000) < 1e-9,
                       metrics={"sec_a_A": ia, "sec_b_A": ib, "secondary_VA_total": va_total})


@check(id="engineering.inductor-design-self-consistent", capability=CAP_INDUCTOR, level="L2", kind="physical_consistency",
       description="The gapped inductor design satisfies the reluctance model exactly: L = mu0 N^2 Ae/(g + le/mu_i), B_pk = L I_pk/(N Ae) <= allowed, AC amplitude = L dI/(2 N Ae).",
       method="Design a 100 uH / 25 A inductor on E 65/32/27 in N87 and recompute the relations from the catalogue parameters.",
       range=N87_RANGE)
def inductor_design_self_consistent() -> CheckResult:
    import aipe_magnetics.api as api
    from aipe_magnetics.engineering import formulas as F
    r = api.design_inductor(inductance={"value": 100, "unit": "uH"}, current_peak={"value": 25, "unit": "A"}, current_rms={"value": 18, "unit": "A"},
                            current_ripple_peak_to_peak={"value": 5, "unit": "A"}, frequency={"value": 100, "unit": "kHz"}, material="N87",
                            core="core.tdk-e65-32-27", temperature={"value": 100, "unit": "degC"}, winding={"conductor_type": "litz"})
    if r.status != "ok":
        return CheckResult(passed=False, notes=[f"design rejected: {[e.message for e in r.errors]}"])
    n = int(r.values["turns"].quantity.value)
    gap = r.values["air_gap"].quantity.value
    ae, le, mu_i = 535e-6, 0.147, 2200.0
    errors = {"inductance_roundtrip": _rel(F.gapped_inductance(n, ae, gap, le, mu_i), 100e-6),
              "flux_density_peak": _rel(r.values["flux_density_peak"].quantity.value, 100e-6 * 25 / (n * ae)),
              "ac_amplitude": _rel(r.values["flux_density_ac_amplitude"].quantity.value, 100e-6 * 5 / (2 * n * ae)),
              "within_allowed": 0.0 if r.values["flux_density_peak"].quantity.value <= r.values["flux_density_allowed"].quantity.value * (1 + 1e-9) else 1.0,
              "gap_nonnegative": 0.0 if gap >= 0 else 1.0}
    return CheckResult(passed=all(v < 1e-9 for v in errors.values()), metrics={**errors, "turns": n, "air_gap_m": gap})


@check(id="engineering.saturation-screening", capability=CAP_VALIDATE, level="L2", kind="physical_consistency",
       description="Saturation screening interpolates Bsat linearly between the datasheet temperature points, refuses extrapolation, and validate_design fails a saturating design while passing a sane one.",
       method="Bsat(62.5 degC) on N87 must be the midpoint of the 25/100 degC datasheet values; 150 degC must be refused; validate_design with 4 turns (B_pk > Bsat) must fail the saturation check and with 20 turns pass it.",
       range=N87_RANGE)
def saturation_screening() -> CheckResult:
    import aipe_magnetics.api as api
    from aipe_magnetics.common.errors import OutOfDomainError
    from aipe_magnetics.data.catalog import Catalog
    from aipe_magnetics.data.records import MaterialProperties
    from aipe_magnetics.engineering.saturation import screen_saturation
    props = MaterialProperties.model_validate(Catalog.default().find_material("N87").properties)
    mid = screen_saturation(flux_density_peak=0.2, temperature=273.15 + 62.5, material=props)
    mid_ok = abs(mid.saturation_flux_density - 0.44) < 1e-12 and mid.saturation_status == "estimated"
    try:
        screen_saturation(flux_density_peak=0.2, temperature=423.15, material=props)
        refused = False
    except OutOfDomainError:
        refused = True
    bad = api.validate_design(material="N87", core="core.tdk-e65-32-27", turns=4, frequency={"value": 100, "unit": "kHz"},
                              temperature={"value": 100, "unit": "degC"}, voltage_peak={"value": 800, "unit": "V"})
    good = api.validate_design(material="N87", core="core.tdk-e65-32-27", turns=20, frequency={"value": 100, "unit": "kHz"},
                               temperature={"value": 100, "unit": "degC"}, voltage_peak={"value": 800, "unit": "V"})
    bad_ok = bad.status == "rejected" and "saturation" in bad.data.get("failed_checks", [])
    good_ok = good.status == "ok" and not good.data.get("failed_checks")
    return CheckResult(passed=mid_ok and refused and bad_ok and good_ok,
                       metrics={"bsat_62_5C_T": mid.saturation_flux_density, "extrapolation_refused": refused,
                                "saturating_design_rejected": bad_ok, "sane_design_passes": good_ok})


@check(id="engineering.invalid-inputs-rejected", capability=CAP_TRANSFORMER, level="L1", kind="invalid_input_rejection",
       description="Impossible or out-of-domain design inputs are returned as explicit rejections (never exceptions, never silently adjusted results).",
       method="Call the API with negative/zero/huge values, wrong dimensions, unknown material and core, out-of-range temperature and an infeasible specification; every call must return status 'rejected' with an error record.")
def invalid_inputs_rejected() -> CheckResult:
    import aipe_magnetics.api as api
    cases = [
        lambda: api.design_transformer({**DEMO_SPEC, "power": {"value": -10, "unit": "kW"}}),
        lambda: api.design_transformer({**DEMO_SPEC, "frequency": {"value": 100, "unit": "mT"}}),
        lambda: api.design_transformer({**DEMO_SPEC, "material": "no-such-material"}),
        lambda: api.design_transformer({**DEMO_SPEC, "cores": ["core.does-not-exist"]}),
        lambda: api.design_transformer({**DEMO_SPEC, "temperature": {"value": 150, "unit": "degC"}}),
        lambda: api.design_transformer({**DEMO_SPEC, "duty": 0.9}),
        lambda: api.design_transformer({**DEMO_SPEC, "power": {"value": 1, "unit": "MW"}, "frequency": {"value": 20, "unit": "kHz"},
                                        "cores": ["core.tdk-etd59-31-22"], "loss_density_target": None}),
        lambda: api.design_inductor(inductance={"value": 100, "unit": "uH"}, current_peak={"value": 10, "unit": "A"},
                                    current_rms={"value": 14, "unit": "A"}, frequency={"value": 100, "unit": "kHz"}, material="N87"),
        lambda: api.validate_design(material="N87", core="core.tdk-e65-32-27", turns=0, frequency={"value": 100, "unit": "kHz"},
                                    voltage_peak={"value": 800, "unit": "V"}),
    ]
    outcomes = []
    for case in cases:
        try:
            res = case()
            outcomes.append(res.status == "rejected" and bool(res.errors))
        except Exception as error:  # noqa: BLE001 - an exception escaping the API is a failed check
            outcomes.append(False)
            return CheckResult(passed=False, notes=[f"exception escaped the API: {type(error).__name__}: {error}"])
    return CheckResult(passed=all(outcomes), metrics={"cases": len(cases), "rejected": sum(outcomes)})


@check(id="engineering.demo-deterministic", capability=CAP_TRANSFORMER, level="L1", kind="determinism",
       description="Two independent runs of the demonstration design produce byte-identical result envelopes (no timestamps, no randomness).",
       method="Run design_transformer twice with the demonstration specification and compare the JSON serialisations.")
def demo_deterministic() -> CheckResult:
    import aipe_magnetics.api as api
    a = json.dumps(api.design_transformer(DEMO_SPEC).to_json_dict(), sort_keys=True)
    b = json.dumps(api.design_transformer(DEMO_SPEC).to_json_dict(), sort_keys=True)
    return CheckResult(passed=a == b, metrics={"bytes": len(a)})
