# Notice - modified GPL-3.0 material

The records in this directory are a **modified version** of data from the Paderborn University LEA
`materialdatabase` (https://github.com/upb-lea/materialdatabase), copyright UPB-LEA, commit
`04fff8e98ae3ddb72809223c42e8c939ed1073bb`, licensed under the GNU General Public License v3.0 (see `LICENSE`, verbatim upstream text).

Modification (2026-11-02, AIPE Magnetics Database 0.3.0): the CSV files were converted into AIPE Magnetic Data
Schema v0.2 JSON/JSONL records with SI units (degC -> K), explicit column definitions and provenance; anomalous
rows were excluded or unit-flagged as documented in `docs/provenance-and-limitations.md`. Values were not otherwise changed.

This directory as a whole is released under **GPL-3.0-only**. Corresponding source: the original CSV files at the
pinned commit above (SHA-256 of each file in `src/aipe_magnetics_database/ingest/pins/paderborn-lea.json`) together with
the conversion program `src/aipe_magnetics_database/ingest/sources/paderborn.py` of this repository.

The GPL grant covers UPB-LEA's work only. Records classified `derived_from_measurement` rest on Princeton MagNet
measurements (MIT), `manufacturer_tool_export` on TDK Magnetic Design Tool output and `manufacturer_datasheet` on
digitised manufacturer datasheets; the rights of those third parties are not cleared by this licence.
