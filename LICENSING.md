# Licensing and data rights

**Code**: the owner has not declared a licence (`NOASSERTION` in `pyproject.toml`, `aipe.yaml` and the plugin
manifests); rights in the code are reserved by its contributors until a licence is chosen.

**Data** keep the licence of their source; every record carries it in `license` and in each evidence item. Nothing
is relicensed.

| Data | Licence | Notes |
| --- | --- | --- |
| `data/normalized/mas`, `data/raw/mas` (OpenMagnetics MAS, commit `c33ccfec`) | Apache-2.0 | upstream licence in `data/raw/mas/LICENSE.upstream.md`; most values originate from manufacturer datasheets, whose terms the MAS licence does not cover |
| `data/normalized/princeton-magnet` (PrincetonUniversity/magnet `8a6892e9`) | MIT | cite the three MagNet IEEE TPEL papers (dataset record) |
| `data/normalized/paderborn-lea`, `data/raw/paderborn-lea` (upb-lea/materialdatabase `04fff8e9`) | GPL-3.0-only | separate copyleft subtree with LICENSE and NOTICE (modification statement, source offer) |
| `data/normalized/hefmag`, `data/raw/hefmag` (Zenodo 8292861) | CC-BY-4.0 | attribution: Ragusa, Dobak, Beatrice, Fiorillo (EMPIR 19ENG06 HEFMAG) |
| `data/normalized/zenodo-smc-ferrite-triangular`, `data/raw/...` (Zenodo 21626116) | CC-BY-4.0 | attribution: Dobak, Onderko, Fuzer, Kollar |
| curated catalogue in `src/aipe_magnetics_database/data/catalog` | NOASSERTION | individual datasheet values transcribed with locators; documents referenced, not copied |
| PSMA core-loss studies, ASU MHz aggregate, MagNet Challenge 1/2, manufacturer tools | not redistributed | reference entries and user-supplied-file adapters only ([docs/source-catalogue.md](docs/source-catalogue.md)) |

Third-party code dependencies (numpy BSD-3-Clause; pydantic, jsonschema, referencing, mcp MIT; h5py BSD-3-Clause) keep
their licences. Public availability is not redistribution permission; rights holders may request removal of values
through an issue.
