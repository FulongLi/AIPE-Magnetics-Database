import copy
import hashlib
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from importers.json_record import import_record
from scripts.validate import ROOT, load_record, validate_records


class MagneticRecordsTests(unittest.TestCase):
    def setUp(self):
        self.path = ROOT / "examples/synthetic-transformer.json"
        self.record = load_record(self.path)

    def invalid(self, record=None):
        self.assertTrue(validate_records([record or self.record]))

    def test_synthetic_example_passes_offline_without_engineering_data(self):
        with patch.object(socket, "socket", side_effect=AssertionError("network forbidden")):
            self.assertEqual(validate_records([self.record]), [])
        self.assertEqual(set(self.record["properties"]), {"description"})

    def test_no_unknown_domain_fields(self):
        self.record["properties"]["permeablity"] = 23
        self.invalid()

    def test_wrong_units_and_untyped_values_fail(self):
        for quantity in [1e-6, {"value": 1, "unit": "uH"}, {"value": 1, "unit": "F"}]:
            self.record["properties"]["leakage_inductance"] = quantity
            self.invalid()

    def test_negative_and_non_finite_quantity_fail(self):
        for value in [-1, float("inf"), float("nan")]:
            self.record["properties"]["leakage_inductance"] = {"value": value, "unit": "H"}
            self.invalid()

    def test_duplicate_ids_and_missing_evidence_fail(self):
        self.assertTrue(validate_records([self.record, self.record]))
        self.record["evidence_refs"] = ["evidence.absent"]
        self.invalid()

    def test_duplicate_evidence_fails(self):
        self.record["evidence"].append(copy.deepcopy(self.record["evidence"][0]))
        self.invalid()

    def test_rejected_evidence_fails(self):
        self.record["evidence"][0]["review_status"] = "rejected"
        self.invalid()

    def test_invalid_timestamp_fails(self):
        self.record["evidence"][0]["created_at"] = "2026-99-10"
        self.invalid()

    def test_unknown_component_reference_fails(self):
        self.record["properties"]["core_ref"] = "core.missing"
        self.invalid()

    def test_reference_resolves_in_supplied_catalogue(self):
        core = copy.deepcopy(self.record)
        core.update(id="core.example", kind="core_geometry", properties={"description":"Synthetic geometry identity"})
        self.record["properties"]["core_ref"] = "core.example"
        self.assertEqual(validate_records([core, self.record]), [])

    def test_assumption_cannot_be_relabelled_real(self):
        self.record["synthetic"] = False
        self.invalid()

    def test_wrong_reference_kind_fails(self):
        self.record["properties"]["core_ref"] = self.record["id"]
        self.invalid()

    def test_unreferenced_source_cannot_validate_real_claim(self):
        self.record["synthetic"] = False
        source = copy.deepcopy(self.record["evidence"][0])
        source.update(id="evidence.unused", kind="source")
        self.record["evidence"].append(source)
        self.invalid()

    def test_measurement_requires_acquisition_context(self):
        self.record["kind"] = "measurement"
        self.invalid()

    def test_import_preserves_raw_records_provenance_and_refuses_overwrite(self):
        raw = self.path.read_bytes()
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "candidate.json"
            result = import_record(self.path, destination)
            self.assertEqual(result["extensions"]["aipe.import"]["source_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(validate_records([load_record(destination)]), [])
            with self.assertRaises(FileExistsError):
                import_record(self.path, destination)
            with self.assertRaises(ValueError):
                import_record(destination, Path(temp) / "second.json")
        self.assertEqual(raw, self.path.read_bytes())

    def test_import_rejects_ambiguous_data_without_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            source, destination = Path(temp) / "raw.json", Path(temp) / "candidate.json"
            source.write_text('{"schema_version":"9.0.0"}')
            with self.assertRaises(ValueError):
                import_record(source, destination)
            self.assertFalse(destination.exists())

    def test_loader_rejects_duplicate_keys_and_nonfinite_tokens(self):
        for text in ['{"id":1,"id":2}', '{"value":NaN}']:
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "bad.json"
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_record(path)


if __name__ == "__main__":
    unittest.main()
