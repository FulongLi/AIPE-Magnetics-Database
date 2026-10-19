# Licensing and data rights

## This repository

The repository has **no licence declaration** by its owner. The v0.2 package therefore keeps
`license = NOASSERTION` in `pyproject.toml`, `aipe.yaml` and both plugin manifests. No blanket
licence is assigned to historical material, to the v0.2 code or to imported datasets. Selecting a
licence is an owner decision; until then, rights in the code are reserved by its contributors.

## Third-party components

| Component | Licence | How it is used |
| --- | --- | --- |
| AIPE-Core schemas v0.1.0 (pinned copy in `src/aipe_magnetics/integrations/aipe_core/schemas/`) and the contract checks of Core's `scripts/validate.py` re-implemented in `integrations/aipe_core/state.py` | MIT | offline compatibility validation; NOTICE file and the `state.py` header carry the copyright, commit and attribution |
| AIPE-Registry manifest schema (pinned copy in `src/aipe_magnetics/integrations/aipe_registry/schema/`) | MIT | offline manifest validation; NOTICE file carries the copyright and commit |
| numpy, scipy, h5py | BSD-3-Clause | runtime |
| pydantic, jsonschema, referencing, PyYAML, mcp | MIT | runtime / optional |
| torch (optional) | BSD-3-Clause | optional data-driven backend |
| Princeton MagNet (`PrincetonUniversity/magnet`) | MIT | read on demand; *derived* Steinmetz fits are bundled with provenance (see `docs/sources.md`) |
| Paderborn `mag-net-hub` | MIT | optional model backend through the `magnethub` package; no weights vendored |
| Paderborn `materialdatabase` | GPL-3.0 | **no code copied, no data bundled**; adapter reads a user checkout; records produced from it carry `license: GPL-3.0` |
| MagNet Challenge 2 (`minjiechen/magnetchallenge-2`) | none published | reference and user-supplied import only |
| TDK / EPCOS datasheets | manufacturer copyright | individual typical values transcribed with page/table locators as evidence; documents are referenced, never copied |

Each source evidence record declares its known licence or `NOASSERTION`. Public availability is
not redistribution permission. Preserve manufacturer notices and measurement partner terms;
reference restricted source documents instead of copying them.
