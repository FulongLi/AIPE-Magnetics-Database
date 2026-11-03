"""Data access through the store / API, with values cross-checked against the original source files."""
from __future__ import annotations

import json

import pytest

import aipe_magnetics_database.api as api
from aipe_magnetics_database.common.errors import DataNotFoundError, InvalidInputError


def test_summary_counts_match_canonical_index():
    s = api.summary()
    assert s["canonical"]["canonical_materials"] >= 1100 and s["canonical"]["measurement_records"] >= 1700
    assert set(s["sources"]) >= {"openmagnetics_mas", "princeton_magnet", "paderborn_lea_materialdatabase", "zenodo_8292861_hefmag",
                                 "zenodo_21626116_smc_ferrite_triangular"}


def test_material_resolution_and_consolidation():
    n87 = api.get_material("N87")
    assert n87["id"] == "material.canonical-tdk-n87" and n87["name"] == "TDK N87"
    members = {m["source_id"] for m in n87["extensions"]["aipe.canonical"]["members"]}
    assert members == {"aipe_curated", "openmagnetics_mas", "paderborn_lea_materialdatabase", "princeton_magnet"}
    assert api.get_material("material.mas-tdk-n87-0015")["id"] == n87["id"]
    with pytest.raises(DataNotFoundError):
        api.get_material("no such grade 123")


def test_listing_filters():
    nano = api.list_materials(material_class="nanocrystalline")
    assert nano and all(m["material_class"] == "nanocrystalline" for m in nano)
    assert any("zenodo_8292861_hefmag" in m["sources"] for m in nano)
    assert all("TDK" in m["name"] for m in api.list_materials(manufacturer="TDK"))


def test_conditions_require_units():
    with pytest.raises(InvalidInputError, match="unit"):
        api.find_measurements(frequency=100e3)
    with pytest.raises(InvalidInputError, match="reversed"):
        api.find_measurements(frequency="200 kHz..100 kHz")


def test_find_measurements_by_conditions_and_duplicates_hidden():
    found = api.find_measurements(material="Ferroxcube 3C90", quantity="core_loss", frequency="100 kHz", temperature="25 degC",
                                  waveform="triangular")
    ids = [e["id"] for e in found]
    assert "measurement.princeton-magnet-3c90-core-loss" in ids
    assert "measurement.mas-ferroxcube-3c90-0032-core-loss-advanced-magnet" not in ids  # duplicate of MagNet, hidden
    everything = api.find_measurements(material="Ferroxcube 3C90", quantity="core_loss", include_duplicates=True)
    assert "measurement.mas-ferroxcube-3c90-0032-core-loss-advanced-magnet" in [e["id"] for e in everything]


# ---- cross-checks against the original files (values read from the upstream files during the audit) -----------------
def test_princeton_rows_equal_the_packaged_database():
    t = api.get_table("measurement.princeton-magnet-3e6-core-loss", limit=1)
    assert t["total_rows"] == 6996  # 3E6_database.h5: 6,996 rows
    assert t["rows"][0] == [50020.0, 0.0405, 0.0, None, None, "sinusoidal", 298.15, 2874.5]
    assert api.get_table("measurement.princeton-magnet-n87-core-loss", limit=1)["total_rows"] == 142871


def test_mas_values_cross_checked():
    n87 = api.get_record("material.mas-tdk-n87-0015")["properties"]
    assert n87["curie_temperature"]["value"]["original"] == {"value": 210, "unit": "degC"}
    assert {"value": 0.38980000000000004, "unit": "T"} in [s["value"] for s in n87["saturation_flux_density"]]
    fit = next(r for r in api.material_properties("TDK N27")["fitted_loss_models"] if r["architecture"] == "steinmetz")
    first = fit["coefficients"]["ranges"][0]
    assert (first["k"], first["alpha"], first["beta"]) == (8.993268138861152, 1.3654727643443896, 2.4255213193041425)
    c350 = api.get_table("measurement.mas-tdk-c350-0001-complex-permeability", limit=1)["rows"][0]
    assert c350 == [1000000.0, 9.121, 0.032]


def test_paderborn_values_cross_checked():
    lea = api.get_table("measurement.paderborn-lea-n95-complex-permeability-lea-mtb", limit=1)["rows"][0]
    assert lea[:3] == ["R29.5x19x14.9", 59629.0, 30.145 + 273.15]
    assert lea[3] == 0.0401250199192159 and lea[5:] == [3955.516035056361, 333.48617320056104]
    eps = api.get_table("measurement.paderborn-lea-n87-complex-permittivity-lea-mtb", limit=1)["rows"][0]
    assert eps == ["SF5", 100578.0, 303.15, 107576.6780220766, 81060.13154810607]
    assert api.find_measurements(source="paderborn_lea_materialdatabase", data_class="derived_from_measurement") == []  # derived views hidden
    derived = api.find_measurements(source="paderborn_lea_materialdatabase", data_class="derived_from_measurement", include_duplicates=True)
    assert len(derived) == 11 and all(e["status"] == "derived_view" for e in derived)


def test_zenodo_values_cross_checked():
    hef = api.get_table("measurement.hefmag-co67-ht-d25um-fluxmetric", flux_density="100 mT", limit=1)["rows"][0]
    assert hef == [0.1, 2.0, hef[2], hef[3], 0.08765]
    smc = api.get_table("measurement.zenodo-21626116-xflux-ultra-sinusoidal", limit=1)["rows"][0]
    assert smc == [0.1, 100.0, 7.52219337676448]
    assert "temperature_not_stated" in api.get_record("measurement.zenodo-21626116-tdk-n30-triangular")["quality"]["flags"]


def test_cores_and_provenance():
    shape = api.get_core("E 42/21/15")
    assert shape["extensions"]["aipe.mas"]["dimensions_range_m"]["A"] == {"minimum_m": 0.0413, "maximum_m": 0.043}
    parts = api.list_cores(kind="parts", material="Ferroxcube 3C90", limit=5)
    assert parts and all(p["manufacturer"] == "Ferroxcube" for p in parts)
    prov = api.provenance("measurement.mas-ferroxcube-3c90-0032-core-loss-advanced-magnet")
    assert prov["canonical"]["status"] == "duplicate" and prov["canonical"]["duplicate_match"]["match_fraction"] == 1.0
    assert prov["license"] == "Apache-2.0" and prov["data"]["data_class"] == "experimental_measurement"


def test_export_writes_schema_valid_jsonl(tmp_path):
    from aipe_magnetics_database.data.records import MagneticRecord
    out = api.export(["measurement.princeton-magnet-3e6-core-loss", "material.canonical-ferroxcube-3e6"], tmp_path / "x.jsonl")
    lines = (tmp_path / "x.jsonl").read_text(encoding="utf-8").splitlines()
    assert out["records"] == 2 and len(lines) == 2
    for line in lines:
        MagneticRecord.model_validate_json(line)
    assert len(json.loads(lines[0])["extensions"]["aipe.table"]["rows"]) == 6996


def test_measurements_and_fitted_parameters_are_kept_apart():
    for entry in api.find_measurements(material="TDK N87", include_duplicates=True):
        assert entry["data_class"] != "fitted_parameters"
    for model in api.material_properties("TDK N87")["fitted_loss_models"]:
        assert model["data_class"] == "fitted_parameters"


def test_readme_example_is_a_valid_excerpt_of_a_committed_record():
    import re
    from pathlib import Path

    from aipe_magnetics_database.data.records import MagneticRecord
    text = (Path(__file__).resolve().parents[2] / "README.md").read_text(encoding="utf-8")
    example = json.loads(re.search(r"```json\n(.*?)```", text, re.S).group(1))
    MagneticRecord.model_validate(example)
    committed = api.get_record(example["id"])
    assert example["properties"]["saturation_flux_density"][0] in committed["properties"]["saturation_flux_density"]
    assert example["evidence"][0]["version"] == committed["evidence"][0]["version"]
    lines = text.splitlines()
    assert 60 <= len(lines) <= 120


def test_readme_python_example_runs():
    hits = api.find_measurements(material="TDK N87", quantity="core_loss", data_class="experimental_measurement",
                                 frequency="100 kHz", temperature="90 degC")
    table = api.get_table(hits[0]["id"], frequency="95 kHz..105 kHz", temperature="90 degC", waveform="sinusoidal")
    assert hits[0]["id"] == "measurement.princeton-magnet-n87-core-loss" and table["rows"]
    assert api.provenance(hits[0]["id"])["license"] == "MIT"
