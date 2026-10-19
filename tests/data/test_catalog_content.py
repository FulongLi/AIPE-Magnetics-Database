"""Provenance and consistency of the bundled catalogue and the derived Steinmetz fits."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from scipy.integrate import quad

from aipe_magnetics.common.jsonio import load
from aipe_magnetics.data.catalog import Catalog
from aipe_magnetics.data.records import MaterialProperties

ROOT = Path(__file__).resolve().parents[2]
FITS = ROOT / "src" / "aipe_magnetics" / "data" / "derived" / "princeton-magnet-steinmetz-fits.json"

# material_steinmetz_param from PrincetonUniversity/magnet src/magnet/constants.py @ 8a6892e (documented there as (k_i, alpha, beta))
PRINCETON_CONSTANTS = {"N87": (0.79822, 1.3453, 2.5752), "3C90": (0.23732, 1.3932, 2.5481)}


def test_catalogue_validates_and_every_real_record_has_source_evidence():
    cat = Catalog.default()
    assert cat.validate() == []
    for record in cat.records():
        if record.synthetic:
            continue
        kinds = {e.kind for e in record.evidence if e.id in record.evidence_refs}
        assert kinds & {"source", "measurement"}, record.id
        for e in record.evidence:
            assert e.license, record.id
            if e.kind == "source":
                assert e.url or e.locator, f"{record.id}: source evidence {e.id} needs a url or locator"


def test_n87_datasheet_values_are_transcribed_with_locators():
    props = MaterialProperties.model_validate(Catalog.default().find_material("N87").properties)
    assert props.initial_permeability.value.value == 2200
    bsat = {round(p.temperature.value - 273.15): p.value.value for p in props.saturation_flux_density}
    assert bsat == {25: pytest.approx(0.49), 100: pytest.approx(0.39)}
    assert props.curie_temperature.value.value == pytest.approx(273.15 + 210)
    assert props.density.value.value == 4850 and props.resistivity.value.value == 10
    assert props.recommended_frequency_range.min == 25e3 and props.recommended_frequency_range.max == 500e3
    assert len(props.loss_reference_points) == 4 and all(p.basis == "datasheet_typical" for p in props.loss_reference_points)
    for p in props.saturation_flux_density + props.loss_reference_points:
        assert p.source_ref == "evidence.tdk-n87-datasheet-2025-06"


def test_parameter_sets_declare_basis_ranges_and_metrics():
    cat = Catalog.default()
    for record in cat.materials():
        props = MaterialProperties.model_validate(record.properties)
        for s in props.steinmetz_parameter_sets:
            assert s.basis in {"datasheet_fit", "measurement_fit", "publication", "manufacturer_supplied", "synthetic"}
            assert s.source_ref is not None, f"{record.id}/{s.id} has no source_ref"
            assert s.frequency_range is not None and s.flux_density_range is not None
            if s.basis == "measurement_fit":
                assert "max_rel_error" in s.fit_metrics and "n_points" in s.fit_metrics
            assert s.k.unit == "W/m^3" and s.alpha > 0 and s.beta > 0


def test_derived_fits_reproduce_princeton_published_constants_for_n87_and_3c90():
    document = load(FITS)
    assert document["source"]["commit"] == "8a6892e91ac622e42f8b9182a6dfe9094ceea97a"
    for material, (k_i_pub, alpha_pub, beta_pub) in PRINCETON_CONSTANTS.items():
        fit = next(f for f in document["fits"] if f["material"] == material and f["temperature_degC"] == 25)
        assert fit["alpha"] == pytest.approx(alpha_pub, rel=2e-4)
        assert fit["beta"] == pytest.approx(beta_pub, rel=2e-4)
        a, b = fit["alpha"], fit["beta"]
        integral = quad(lambda th: abs(math.cos(th)) ** a, 0, 2 * math.pi, points=[math.pi / 2, math.pi, 3 * math.pi / 2])[0]
        k_i = fit["k_W_per_m3"] / ((2 * math.pi) ** (a - 1) * integral * 2 ** (b - a))
        assert k_i == pytest.approx(k_i_pub, rel=5e-3)
        assert document["source"]["files"][material]["sha256"]


def test_cores_have_effective_parameters_from_datasheets():
    cat = Catalog.default()
    cores = {c.id: c for c in cat.cores()}
    assert {"core.tdk-e65-32-27", "core.tdk-e70-33-32", "core.tdk-etd59-31-22", "core.tdk-pm87-70", "core.tdk-uu93-152-30"} <= set(cores)
    from aipe_magnetics.data.records import CoreGeometryProperties
    e65 = CoreGeometryProperties.model_validate(cores["core.tdk-e65-32-27"].properties)
    assert e65.effective_area.value == pytest.approx(535e-6) and e65.effective_length.value == pytest.approx(0.147)
    assert e65.effective_volume.value == pytest.approx(78650e-9) and e65.mass.value == pytest.approx(0.394)
    assert e65.bobbin_window_area.value == pytest.approx(426e-6) and e65.mean_turn_length.value == pytest.approx(0.188)
    assert e65.effective_area.original.unit == "mm^2"
    assert "material.tdk-n87" in e65.material_refs
    assert cat.cores("material.tdk-n87")


# coil-former rows (AN mm^2, lN mm, AR uOhm) as printed in the TDK October 2024 datasheets for the ordering code named in each record
COIL_FORMERS = {
    "core.tdk-e65-32-27": (426.0, 188.0, 15.2, "B66388B1000T001"),
    "core.tdk-e70-33-32": (456.0, 209.0, 15.8, "B66372B1000T001"),   # one-core-set former; 230.5 mm / 20 uOhm are the two-set formers
    "core.tdk-etd59-31-22": (365.6, 106.1, 10.0, "B66398W1024T001"),
    "core.tdk-pm87-70": (657.0, 158.0, 8.27, "B65714K1020T001"),
}


@pytest.mark.parametrize("core_id", sorted(COIL_FORMERS))
def test_coil_former_rows_are_transcribed_from_one_variant(core_id):
    from aipe_magnetics.data.records import CoreGeometryProperties
    an_mm2, ln_mm, ar_uohm, code = COIL_FORMERS[core_id]
    record = Catalog.default().get(core_id)
    props = CoreGeometryProperties.model_validate(record.properties)
    assert props.bobbin_window_area.value == pytest.approx(an_mm2 * 1e-6)
    assert props.mean_turn_length.value == pytest.approx(ln_mm * 1e-3)
    datasheet = record.extensions["aipe.datasheet"]
    assert datasheet["coil_former_AR_uOhm"] == pytest.approx(ar_uohm)
    assert code in datasheet["coil_former"]
    # transcription consistency: TDK's AR = rho lN / AN per turn^2 uses one copper constant, so AR / (lN / AN) is the same
    # (34.4-34.5 uOhm mm) for every former; a value mixed from two variants (456 mm2 with 230.5 mm / 20 uOhm) gives 39.6
    assert ar_uohm / (ln_mm / an_mm2) == pytest.approx(34.45, rel=0.01)
    evidence = next(e for e in record.evidence if e.id in record.evidence_refs)
    assert code in (evidence.locator or "")


def test_uu93_window_height_is_the_drawn_single_u_value_with_the_derived_total_marked():
    from aipe_magnetics.data.records import CoreGeometryProperties
    record = Catalog.default().get("core.tdk-uu93-152-30")
    props = CoreGeometryProperties.model_validate(record.properties)
    dims = props.dimensions.values
    assert dims["window_height_single_u"].value == pytest.approx(48e-3) and "window_height" not in dims
    assert props.bobbin_window_area is None and props.mean_turn_length is None
    derived = record.extensions["aipe.datasheet"]["derived"]
    assert derived["window_height_uu_mm"] == 96.0 and "derived" in derived["note"]
    evidence = next(e for e in record.evidence if e.id in record.evidence_refs)
    assert "U 93/76/30 with I 93/28/30" in evidence.reference and "no coil former" in evidence.description


def test_muhlethaler_set_validity_is_the_table_i_fit_span():
    props = MaterialProperties.model_validate(Catalog.default().find_material("N87").properties)
    s = next(s for s in props.steinmetz_parameter_sets if s.id == "set.muhlethaler-2012-n87-25c")
    assert (s.frequency_range.min, s.frequency_range.max) == (20e3, 50e3)
    assert (s.flux_density_range.min, s.flux_density_range.max) == (0.05, 0.1)
    assert "triangular" in s.note and "20-100 kHz" in s.note


def test_application_tags_follow_the_transcribed_application_statements():
    """'power' only where the cited source says so: N87 datasheet; Princeton MagNet material_applications for the rest."""
    cat = Catalog.default()
    not_power = {"material.ferroxcube-3e6", "material.tdk-n30", "material.fair-rite-77"}  # wideband / broadband / inductive designs
    for record in cat.materials():
        if record.synthetic:
            continue
        description = record.properties.get("description") or ""
        assert "power ferrite" not in description or record.id == "material.tdk-n87"
        if record.id in not_power:
            assert "power" not in record.tags, record.id
        else:
            assert "power" in record.tags, record.id
        if record.id != "material.tdk-n87":
            assert "material_applications" in description


def test_family_standards_are_only_asserted_where_the_datasheet_prints_them():
    cat = Catalog.default()
    families = {r.id: r for r in cat.records() if r.kind == "core_family"}
    assert families["core-family.tdk-etd"].properties["standard"] == "IEC 63093-6"
    assert families["core-family.tdk-pm"].properties["standard"] == "IEC 63093-10"
    assert "standard" not in families["core-family.tdk-e"].properties
    assert "standard" not in families["core-family.tdk-u"].properties
