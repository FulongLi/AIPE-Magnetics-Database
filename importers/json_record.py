"""Reviewable JSON import: validate, record raw checksum, never overwrite."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from scripts.validate import load_record, validate_records


def import_record(source, destination, catalogue=()):
    source, destination = Path(source), Path(destination)
    record = load_record(source)
    errors = validate_records([*catalogue, record])
    if errors:
        raise ValueError("; ".join(errors))
    result = copy.deepcopy(record)
    extensions = result.setdefault("extensions", {})
    if "aipe.import" in extensions:
        raise ValueError("Existing import provenance must not be replaced")
    extensions["aipe.import"] = {
        "source_filename": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "importer_version": "0.1.0",
        "normalization": "none; source must already use explicit SI units",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--catalogue", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    import_record(args.source, args.destination, [load_record(path) for path in args.catalogue])


if __name__ == "__main__":
    main()
