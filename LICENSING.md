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
| `data/normalized/nabih-magneto`, `data/raw/nabih-magneto` (asalahnabih/magneto `0a3a0f1e`) | MIT | cite per the upstream CITATION.cff (A. Nabih, Virginia Tech); `derived/` is a TSV export, not an original |
| `data/normalized/guillod-n87-dielectric`, `data/raw/guillod-n87-dielectric` (otvam/large_signal_ferrite_apec24 `dd068898`) | MIT | Guillod, Roberts, Sullivan, APEC 2024; the repository's datasheet- and MagNet-derived permeability folders are not included |
| `data/normalized/zenodo-7368936-cwh-n87` (Zenodo 7368936; raw on Hugging Face) | CC-BY-4.0 | attribution: Guillod, Lee, Li, Wang, Chen, Sullivan; measurements from MagNet |
| `data/normalized/zenodo-10725890-inrim`, `data/raw/zenodo-10725890-inrim` (Zenodo 10725890) | CC-BY-4.0 | attribution: M. Pasquale, INRIM |
| curated catalogue in `src/aipe_magnetics_database/data/catalog` | NOASSERTION | individual datasheet values transcribed with locators; documents referenced, not copied |
| PSMA core-loss studies, ASU MHz aggregate, MagNet Challenge 1/2, DTU IEEE DataPort set, unlicensed GitHub data, manufacturer tools | not redistributed | reference entries and user-supplied-file adapters only ([docs/source-catalogue.md](docs/source-catalogue.md)) |

Third-party code dependencies (numpy BSD-3-Clause; pydantic, jsonschema, referencing, mcp MIT; h5py BSD-3-Clause) keep
their licences. Public availability is not redistribution permission; rights holders may request removal of values
through an issue.

The Hugging Face dataset [FulongLi/AIPE-Magnetics-Dataset](https://huggingface.co/datasets/FulongLi/AIPE-Magnetics-Dataset)
republishes only the sources above whose licence permits redistribution (not the NOASSERTION curated catalogue); each
`raw/<source-id>/` and `normalized/v0.2/<source-id>/` subtree keeps its source licence (`LICENSE.md` there).
