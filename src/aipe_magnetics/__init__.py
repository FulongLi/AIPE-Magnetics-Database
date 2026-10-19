"""AIPE Magnetics: evidence-first magnetic engineering expert for power electronics.

The package is organised in five functional modules plus thin interaction layers:

* ``aipe_magnetics.data``          - data standard v0.2, units, HDF5 waveforms, importers, source adapters
* ``aipe_magnetics.models``        - magnetic model contract and model families (empirical, physics, ODE, data-driven)
* ``aipe_magnetics.engineering``   - practical design and estimation calculations
* ``aipe_magnetics.verification``  - independent verification levels and evidence registry
* ``aipe_magnetics.integrations``  - AIPE-Core / AIPE-Registry compatibility adapters
* ``aipe_magnetics.api``           - stable typed Python API (no AI agent required)
* ``aipe_magnetics.cli``           - command line interface
* ``aipe_magnetics.mcp``           - local stdio MCP server exposing the API

All mathematics lives in ``models`` / ``engineering`` and is callable without any language model.
"""

from __future__ import annotations

__version__ = "0.2.0"
TOOL_ID = "aipe.magnetics"
TOOL_NAME = "AIPE Magnetics"
SCHEMA_VERSION = "0.2.0"

__all__ = ["__version__", "TOOL_ID", "TOOL_NAME", "SCHEMA_VERSION"]
