"""Executable demonstration: 10 kW dual-active-bridge transformer at 100 kHz (see docs/demo-dab-10kw.md).

Everything is computed through the public API; nothing is hard-coded except the specification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aipe_magnetics.api as api
from aipe_magnetics.common.results import EngineeringResult

SPEC: dict[str, Any] = {
    "power": {"value": 10, "unit": "kW"},
    "frequency": {"value": 100, "unit": "kHz"},
    "primary_voltage": {"value": 800, "unit": "V"},
    "secondary_voltage": {"value": 400, "unit": "V"},
    "voltage_waveform": "square",
    "duty": 0.5,
    "material": "N87",
    "loss_density_target": {"value": 300, "unit": "kW/m^3"},
    "flux_limit_fraction": 0.7,
    "conductor": "litz",
    "strand_diameter": {"value": 0.1, "unit": "mm"},
    "current_density": {"value": 4, "unit": "A/mm^2"},
    "fill_factor": 0.4,
}


def _print(title: str, result: EngineeringResult) -> None:
    from aipe_magnetics.cli.main import _summary
    print(f"\n=== {title} ===")
    print(_summary(result))


def run_dab_demo(*, output: Path | None = None, temperature: dict[str, Any] | None = None, as_json: bool = False) -> int:
    temperature = temperature or {"value": 100, "unit": "degC"}
    steps: list[tuple[str, EngineeringResult]] = []

    # 1-2. material lookup and parameter inspection
    material = api.get_material(SPEC["material"])
    steps.append(("1-2 material lookup and parameter inspection", material))
    if material.status != "ok":
        _print("material", material)
        return 2
    # 3-9. candidate selection, turns, flux density, saturation, core loss, winding loss, comparison
    winding = {"conductor_type": SPEC["conductor"], "strand_diameter": SPEC["strand_diameter"], "current_density": SPEC["current_density"],
               "fill_factor": SPEC["fill_factor"]}
    design = api.design_transformer({
        "power": SPEC["power"], "frequency": SPEC["frequency"],
        "primary": {"name": "primary", "voltage": SPEC["primary_voltage"], "winding": winding},
        "secondaries": [{"name": "secondary", "voltage": SPEC["secondary_voltage"], "winding": winding}],
        "voltage_waveform": SPEC["voltage_waveform"], "duty": SPEC["duty"], "material": SPEC["material"], "cores": None,
        "temperature": temperature, "flux_limit": {"mode": "fraction_of_saturation", "fraction": SPEC["flux_limit_fraction"]},
        "loss_density_target": SPEC["loss_density_target"], "loss_model": "auto", "max_candidates": 10, "policy": "warn"})
    steps.append(("3-9 core candidates, turns, flux density, saturation, core loss, winding loss, comparison", design))
    validation = None
    loss = None
    if design.status == "ok":
        selected = design.data["selected"]
        n1 = int(design.values["turns_primary"].quantity.value)
        validation = api.validate_design({"material": SPEC["material"], "core": selected, "turns": n1, "frequency": SPEC["frequency"],
                                          "temperature": temperature, "voltage_peak": SPEC["primary_voltage"],
                                          "voltage_waveform": SPEC["voltage_waveform"], "duty": SPEC["duty"],
                                          "flux_limit": {"mode": "fraction_of_saturation", "fraction": SPEC["flux_limit_fraction"]}})
        steps.append(("6 saturation screening and operating-point validation (selected core)", validation))
        loss = api.estimate_core_loss({"material": SPEC["material"], "core": selected, "model": "igse", "policy": "warn",
                                       "operating_point": {"frequency": SPEC["frequency"],
                                                           "flux_density_peak": design.values["flux_density_peak"].quantity.model_dump(mode="json", exclude_none=True),
                                                           "temperature": temperature, "waveform": {"kind": "triangular", "duty": 0.5}}})
        steps.append(("7 core-loss estimation (iGSE, triangular flux) for the selected core", loss))
    # 10. report
    report = api.generate_report([r for _, r in steps], {"title": "10 kW / 100 kHz DAB transformer - AIPE Magnetics demonstration",
                                                        "format": "markdown"})
    steps.append(("10 engineering report", report))
    core_state = None
    if design.status == "ok":
        try:
            from aipe_magnetics.integrations.aipe_core.bridge import component_summary_from_design
            from aipe_magnetics.integrations.aipe_core.export import to_engineering_state
            summary = component_summary_from_design(design, role="dab-transformer", created_at="2026-10-19T00:00:00Z")
            core_state = to_engineering_state(summary, created_at="2026-10-19T00:00:00Z", project_id="project.dab-10kw-demo",
                                              requirements={"nominal_power": SPEC["power"], "input_voltage": SPEC["primary_voltage"],
                                                            "output_voltage": SPEC["secondary_voltage"], "switching_frequency": SPEC["frequency"]})
        except Exception as error:  # noqa: BLE001 - the export is optional for the demonstration
            print(f"(AIPE-Core export not available: {type(error).__name__}: {error})")

    if as_json:
        print(json.dumps({"steps": [{"title": t, "result": r.to_json_dict()} for t, r in steps]}, indent=2))
    else:
        for title, result in steps:
            if result.capability.endswith("generate_report"):
                continue
            _print(title, result)
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "dab-10kw-report.md").write_text(report.data["content"], encoding="utf-8", newline="\n")
        (output / "dab-10kw-results.json").write_text(json.dumps({"steps": [{"title": t, "result": r.to_json_dict()} for t, r in steps]},
                                                                 indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
        if core_state is not None:
            (output / "dab-10kw-engineering-state.json").write_text(json.dumps(core_state, indent=2, ensure_ascii=False, sort_keys=True),
                                                                   encoding="utf-8", newline="\n")
        print(f"\nreport written to {output / 'dab-10kw-report.md'}")
    return 0 if design.status == "ok" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run_dab_demo(output=Path("reports/demo")))
