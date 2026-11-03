"""AIPE Magnetics Database: standardised, provenance-tracked magnetic data for power electronics.

* ``aipe_magnetics_database.common``   - quantities, SI units, errors, result envelopes, JSON I/O
* ``aipe_magnetics_database.data``     - Magnetic Data Schema v0.2 records, HDF5 waveform/table layout,
  importers and external source adapters
* ``aipe_magnetics_database.ingest``   - reproducible source importers, deduplication and catalogue build
* ``aipe_magnetics_database.store``    - the repository data store (normalised + canonical records) and queries
* ``aipe_magnetics_database.database`` - query layer over the curated catalogue
* ``aipe_magnetics_database.cli`` / ``mcp`` - command line and local stdio MCP server (read-only data access)

The package contains no loss prediction, design or optimisation code; that is AIPE-Magnetics-Design.
"""

from __future__ import annotations

__version__ = "0.3.0"
TOOL_ID = "aipe.magnetics.database"
TOOL_NAME = "AIPE Magnetics Database"
SCHEMA_VERSION = "0.2.0"

__all__ = ["__version__", "TOOL_ID", "TOOL_NAME", "SCHEMA_VERSION"]
