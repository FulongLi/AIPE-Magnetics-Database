"""Offline, evidence-first validation of a magnetic-record catalogue."""

import argparse
import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]


def load_record(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(f"Non-finite number: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=unique, parse_constant=reject)


def validate_records(records):
    schema = load_record(ROOT / "schemas/magnetic-record.schema.json")
    Draft202012Validator.check_schema(schema)
    checker = FormatChecker()
    if not {"date-time", "uri-reference"}.issubset(checker.checkers):
        raise ValueError("Install requirements-dev.txt including JSON Schema format checks")
    validator = Draft202012Validator(schema, format_checker=checker)
    errors = []
    for index, record in enumerate(records):
        errors.extend(f"record[{index}]{error.json_path}: {error.message}"
                      for error in validator.iter_errors(record))
    if errors:
        return errors
    identifiers = set()
    by_id = {record["id"]: record for record in records}
    reference_kinds = {
        "material_ref": {"magnetic_material", "core_material"},
        "family_ref": {"core_family"},
        "core_ref": {"core_geometry"},
        "winding_refs": {"winding"},
        "component_ref": {"inductor", "transformer", "magnetic_component"},
        "operating_condition_ref": {"operating_condition"},
        "operating_condition_refs": {"operating_condition"},
    }
    for record in records:
        if record["id"] in identifiers:
            errors.append(f"Duplicate record ID: {record['id']}")
        identifiers.add(record["id"])
    for record in records:
        evidence = {item["id"]: item for item in record["evidence"]}
        if len(evidence) != len(record["evidence"]):
            errors.append(f"{record['id']}: duplicate evidence ID")
        for reference in record["evidence_refs"]:
            if reference not in evidence:
                errors.append(f"{record['id']}: unknown evidence {reference}")
            elif evidence[reference]["review_status"] == "rejected":
                errors.append(f"{record['id']}: rejected evidence {reference}")
        selected = [evidence[ref] for ref in record["evidence_refs"] if ref in evidence]
        if not record["synthetic"] and all(item["kind"] == "assumption" for item in selected):
            errors.append(f"{record['id']}: real records require non-assumption source evidence")
        if record["kind"] == "measurement" and not any(
            evidence[reference]["kind"] == "measurement"
            for reference in record["evidence_refs"] if reference in evidence
        ):
            errors.append(f"{record['id']}: measurement requires measurement evidence")
        for key, value in record["properties"].items():
            references = [value] if key.endswith("_ref") else value if key.endswith("_refs") else []
            for reference in references:
                if reference not in identifiers:
                    errors.append(f"{record['id']}: unknown record reference {reference}")
                elif by_id[reference]["kind"] not in reference_kinds[key]:
                    errors.append(f"{record['id']}: {key} points to wrong record kind")
        def finite(value):
            if isinstance(value, float) and not math.isfinite(value):
                errors.append(f"{record['id']}: quantities must be finite")
            elif isinstance(value, dict):
                for item in value.values():
                    finite(item)
            elif isinstance(value, list):
                for item in value:
                    finite(item)
        finite(record)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        errors = validate_records([load_record(path) for path in args.records])
    except (ValueError, OSError) as error:
        errors = [str(error)]
    if errors:
        print("\n".join(errors))
        return 1
    print(f"PASS {len(args.records)} records (schema/reference checks; no physical validation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
