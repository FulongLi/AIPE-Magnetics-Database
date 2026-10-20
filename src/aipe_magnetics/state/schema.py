"""JSON Schema export, validation, load / dump (round trip) of the Magnetic Engineering State."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aipe_magnetics.common.errors import DataIntegrityError, InvalidInputError
from aipe_magnetics.state.model import STATE_SCHEMA, STATE_VERSION, MagneticEngineeringState

SCHEMA_ID = "https://github.com/FulongLi/AIPE-Magnetics-Database/schemas/state/v0.3.0/magnetic-engineering-state.schema.json"
PACKAGED_SCHEMA = Path(__file__).resolve().parents[1] / "schemas" / "state" / "v0.3.0" / "magnetic-engineering-state.schema.json"


def state_json_schema() -> dict[str, Any]:
    schema = MagneticEngineeringState.model_json_schema(mode="validation")
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": SCHEMA_ID,
              "title": "AIPE Magnetics - Magnetic Engineering State 0.3.0",
              "description": f"{STATE_SCHEMA}: versioned record of a magnetic component design lifecycle (AIPE Magnetics).", **schema}
    return schema


def export_state_schema(path: str | Path | None = None) -> Path:
    path = Path(path) if path else PACKAGED_SCHEMA
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state_json_schema(), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return path


def check_state_schema(path: str | Path | None = None) -> list[str]:
    """Differences between a committed schema file and a fresh export (empty = in sync)."""
    path = Path(path) if path else PACKAGED_SCHEMA
    if not path.exists():
        return [f"{path} is missing"]
    committed = json.loads(path.read_text(encoding="utf-8"))
    return [] if committed == state_json_schema() else [f"{path} differs from a fresh export of MagneticEngineeringState"]


def validate_state(data: dict[str, Any]) -> list[str]:
    """Validate a state document: JSON Schema (structure) and the model's cross-reference rules. Empty = valid."""
    import jsonschema
    errors: list[str] = []
    validator = jsonschema.Draft202012Validator(state_json_schema())
    for e in sorted(validator.iter_errors(data), key=lambda e: list(e.path)):
        errors.append(f"$.{'.'.join(str(p) for p in e.path)}: {e.message}")
    if not errors:
        try:
            MagneticEngineeringState.model_validate(data)
        except ValidationError as error:
            errors += [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors()]
    return errors


def dump_state(state: MagneticEngineeringState, path: str | Path, *, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(state.model_dump(mode="json", exclude_none=True), indent=2, ensure_ascii=False, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")
    return path


def load_state(path: str | Path, *, migrate: bool = True) -> MagneticEngineeringState:
    from aipe_magnetics.common.jsonio import load
    data = load(path)
    if not isinstance(data, dict):
        raise InvalidInputError(f"{path} does not contain a JSON object")
    if migrate:
        from aipe_magnetics.state.migrate import migrate_state
        data = migrate_state(data)
    errors = validate_state(data)
    if errors:
        raise DataIntegrityError(f"{path} is not a valid Magnetic Engineering State {STATE_VERSION}", details={"errors": errors[:50]})
    return MagneticEngineeringState.model_validate(data)


__all__ = ["PACKAGED_SCHEMA", "SCHEMA_ID", "check_state_schema", "dump_state", "export_state_schema", "load_state", "state_json_schema",
           "validate_state"]
